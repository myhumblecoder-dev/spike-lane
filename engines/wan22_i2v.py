# SPDX-License-Identifier: Apache-2.0
"""End-to-end FastMetal-5B-QAD generation on Apple Silicon (MLX DiT + MLX TAEHV).

This is the Wan2.2 TI2V entrypoint. Use FastVideo/FastMetal-5B-QAD:

    hf download FastVideo/FastMetal-5B-QAD --local-dir ./FastMetal-5B-QAD
    python examples/inference/basic/mlx_wan22_generate.py \\
      --mlx-checkpoint ./FastMetal-5B-QAD \\
      --text-encoder-root ./FastMetal-5B-QAD \\
      --vae-root ./FastMetal-5B-QAD/vae

Pipeline: torch/MPS UMT5 encode (shared with 1.3B) → MLXWan22DiT 3-step DMD
(warped schedule, flow_shift=5) → MLX TAEHV decode (taew2_2.pth). Fully MLX
on the heavy DiT + decode path.

Decoder backends: ``taehv`` (default, MLX, ~seconds), ``taehv-torch`` (parity),
``wan-vae`` (full AutoencoderKLWan on MPS, slow).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from fastvideo.mlx_runtime.fast_spatial import DEFAULT_FAST_SPATIAL_SHARPEN
from fastvideo.mlx_runtime.frame_upsample import DEFAULT_PIXEL_UPSAMPLE_MODE, PIXEL_UPSAMPLE_MODES
from fastvideo.mlx_runtime.memory import cleanup_mlx
from fastvideo.mlx_runtime.prompt_cache import (
    fingerprint_digest,
    load_prompt_cache,
    save_prompt_cache,
    text_encoder_fingerprint,
)
from fastvideo.mlx_runtime.checkpoint_compat import (
    UnsupportedMLXCheckpointError,
    raise_if_unsupported_mlx_checkpoint,
    resolve_mlx_checkpoint,
)
from fastvideo.mlx_runtime.rife_interp import aligned_keyframe_count

FASTWAN21_MODEL_ID = "FastVideo/FastWan2.1-T2V-1.3B-Diffusers"
FASTWAN22_MODEL_ID = "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers"
DEFAULT_HEIGHT = 448
DEFAULT_WIDTH = 832
DEFAULT_NUM_FRAMES = 121

def _resolve_model_paths(
    *,
    text_encoder_root: Path | None,
    dit_checkpoint: Path | None,
    dit_config: Path | None,
    vae_root: Path | None,
    mlx_checkpoint: Path | None,
    decode_backend: str,
) -> tuple[Path, Path | None, Path | None, Path | None]:
    """Download only the missing assets required by the selected Wan2.2 path."""
    from huggingface_hub import snapshot_download

    if text_encoder_root is None:
        text_encoder_root = Path(snapshot_download(
            FASTWAN21_MODEL_ID,
            allow_patterns=["tokenizer/*", "text_encoder/*"],
        ))
    if mlx_checkpoint is None and (dit_checkpoint is None or dit_config is None):
        patterns = []
        if dit_checkpoint is None:
            patterns.append("transformer/diffusion_pytorch_model.safetensors")
        if dit_config is None:
            patterns.append("transformer/config.json")
        model_root = Path(snapshot_download(FASTWAN22_MODEL_ID, allow_patterns=patterns))
        dit_checkpoint = dit_checkpoint or model_root / "transformer/diffusion_pytorch_model.safetensors"
        dit_config = dit_config or model_root / "transformer/config.json"
    if decode_backend == "wan-vae" and vae_root is None:
        model_root = Path(snapshot_download(FASTWAN22_MODEL_ID, allow_patterns=["vae/*"]))
        vae_root = model_root / "vae"
    return text_encoder_root, dit_checkpoint, dit_config, vae_root


def _prompt_cache_fingerprint(
    *,
    prompt: str,
    prompt_used: str,
    enhance_prompt: bool,
    enhance_prompt_backend: str,
    text_encoder_root: Path,
    max_sequence_length: int,
    dtype: str,
) -> dict[str, object]:
    return {
        "prompt": prompt,
        "prompt_used": prompt_used,
        "enhance_prompt": enhance_prompt,
        "enhance_prompt_backend": enhance_prompt_backend,
        "text_encoder": text_encoder_fingerprint(text_encoder_root),
        "max_sequence_length": max_sequence_length,
        "dtype": dtype,
    }


def _default_prompt_cache_path(fingerprint: dict[str, object]) -> Path:
    """Content-addressed default cache file for a prompt fingerprint.

    The Wan2.1 entrypoint caches prompt embeddings by default; this one only
    did so when handed an explicit ``--prompt-embeds-cache`` path, so every 5B
    run paid a full UMT5 encode (~45s on an M4 Max) even for a repeat prompt.
    The fingerprint already covers everything that changes the embedding, so
    hash it for the filename.
    """
    digest = fingerprint_digest(fingerprint)[:32]
    return Path.home() / ".cache" / "fastvideo" / "prompt_embeds" / f"wan22_{digest}.npy"


def _encode_first_frame(image_path, vae_dir, *, height, width, frames=1):
    """Encode a still held for `frames` frames to normalized Wan2.2 latents [1, C, T, h, w] (mlx fp16)."""
    import mlx.core as mx
    import torch
    from diffusers import AutoencoderKLWan
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    scale = max(width / img.width, height / img.height)  # cover, then center-crop
    img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    left, top = (img.width - width) // 2, (img.height - height) // 2
    img = img.crop((left, top, left + width, top + height))
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1)[None, :, None].repeat(1, 1, frames, 1, 1)
    vae = AutoencoderKLWan.from_pretrained(vae_dir, torch_dtype=torch.float32, local_files_only=True).to("mps")
    with torch.no_grad():
        z = vae.encode(x.to("mps")).latent_dist.mode()
    mean = torch.tensor(vae.config.latents_mean, device="mps").view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device="mps").view(1, -1, 1, 1, 1)
    z = ((z - mean) / std).float().cpu().numpy()
    del vae
    torch.mps.empty_cache()
    return mx.array(z).astype(mx.float16)


def _encode_video(video_path, vae_dir, *, height, width, frames, fps=24.0, start=0.0):
    """Encode `frames` frames of a real clip (conformed to `fps`, cover-cropped) to normalized
    Wan2.2 latents [1, C, T, h, w] (mlx fp16)."""
    import subprocess
    import mlx.core as mx
    import torch
    from diffusers import AutoencoderKLWan

    vf = (f"fps={fps},scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos,"
          f"crop={width}:{height}")
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(start), "-i", str(video_path), "-vf", vf,
                          "-frames:v", str(frames), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         check=True, capture_output=True).stdout
    clip = np.frombuffer(raw, np.uint8).reshape(-1, height, width, 3)
    if len(clip) < frames:  # hold the last frame if the clip runs short
        clip = np.concatenate([clip, np.repeat(clip[-1:], frames - len(clip), axis=0)])
    x = torch.from_numpy(clip.astype(np.float32) / 127.5 - 1.0).permute(3, 0, 1, 2)[None]
    vae = AutoencoderKLWan.from_pretrained(vae_dir, torch_dtype=torch.float32, local_files_only=True).to("mps")
    with torch.no_grad():
        z = vae.encode(x.to("mps")).latent_dist.mode()
    mean = torch.tensor(vae.config.latents_mean, device="mps").view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device="mps").view(1, -1, 1, 1, 1)
    z = ((z - mean) / std).float().cpu().numpy()
    del vae
    torch.mps.empty_cache()
    return mx.array(z).astype(mx.float16)


def _sample_v2v(model, encoder_hidden_states, noise_latents, freqs_cis, *, init_latent, init_sigma,
                image_latent=None, steps=3, flow_shift=5.0, seed=0, **_):
    """Restyle a real clip: start from its latents re-noised to `init_sigma` and denoise toward the
    prompt in `steps` DMD steps. Lower sigma keeps more of the real motion; higher restyles more.
    With `image_latent`, latent frame 0 is also pinned to that image (TI2V conditioning)."""
    import mlx.core as mx
    from fastvideo.mlx_runtime.sampling import dmd_step, pred_noise_to_pred_video
    from fastvideo.mlx_runtime.wan22_sample import build_wan22_dmd_schedule

    schedule, _ = build_wan22_dmd_schedule(None, flow_shift=flow_shift, warp_denoising_step=True)
    # After warping, sigma(t) = t / 1000, so pick timesteps directly on the sigma axis.
    timesteps = [1000.0 * init_sigma * (1 - k / steps) for k in range(steps)]
    rng = np.random.default_rng(seed)
    batch, _c, frames, height, width = noise_latents.shape
    pt, ph, pw = model.patch_size
    per_frame = (height // ph) * (width // pw)
    tokens = (frames // pt) * per_frame
    pinned = 0 if image_latent is None else per_frame
    img = None if image_latent is None else image_latent[:, :, :1].astype(mx.float32)

    def pin(lat):
        return lat if img is None else mx.concatenate([img.astype(lat.dtype), lat[:, :, 1:]], axis=2)

    sigma = float(schedule.sigma_for(timesteps[0]))
    latents = pin((1.0 - sigma) * init_latent.astype(mx.float32) + sigma * noise_latents.astype(mx.float32))
    print(f"[v2v] init sigma={sigma:.3f}, timesteps={[round(t) for t in timesteps]}, pinned first frame={img is not None}", flush=True)
    last = len(timesteps) - 1
    for i, t in enumerate(timesteps):
        ts = mx.concatenate([mx.zeros((batch, pinned), dtype=mx.float32),
                             mx.full((batch, tokens - pinned), float(t), dtype=mx.float32)], axis=1)
        pred = model(latents.astype(mx.float16), encoder_hidden_states, ts, freqs_cis)
        ni, pn = latents.astype(mx.float32), pred.astype(mx.float32)
        if i < last:
            renoise = mx.array(rng.standard_normal(tuple(latents.shape)).astype(np.float32))
            latents = dmd_step(latents=ni, noise_input_latent=ni, pred_noise=pn, schedule=schedule,
                               timestep=float(t), next_timestep=float(timesteps[i + 1]),
                               noise=renoise).astype(latents.dtype)
        else:
            latents = pred_noise_to_pred_video(pn, ni, schedule.sigma_for(float(t))).astype(latents.dtype)
        latents = pin(latents)
        mx.eval(latents)
    return latents


def _sample_i2v(model, encoder_hidden_states, noise_latents, freqs_cis, *, image_latent, anchor_start=0,
                dmd_denoising_steps=None, flow_shift=5.0, warp_denoising_step=True, seed=0):
    """sample_wan22_dmd with TI2V conditioning: latent frame 0 is pinned to the image
    and its tokens get timestep 0 (clean), every other token the current level."""
    import mlx.core as mx
    from fastvideo.mlx_runtime.sampling import dmd_step, pred_noise_to_pred_video
    from fastvideo.mlx_runtime.wan22_sample import build_wan22_dmd_schedule

    schedule, timesteps = build_wan22_dmd_schedule(dmd_denoising_steps, flow_shift=flow_shift,
                                                   warp_denoising_step=warp_denoising_step)
    rng = np.random.default_rng(seed)
    batch, _c, frames, height, width = noise_latents.shape
    pt, ph, pw = model.patch_size
    per_frame = (height // ph) * (width // pw)
    tokens = (frames // pt) * per_frame
    still = image_latent.astype(mx.float32)          # [1, C, T or 1, h, w]
    img = still[:, :, :1]

    def pin(lat):
        return mx.concatenate([img.astype(lat.dtype), lat[:, :, 1:]], axis=2)

    if anchor_start:
        # Start every frame from the storyboard still, re-noised to the level of the
        # first step we keep: x_t = (1 - sigma) * x0 + sigma * noise (flow matching).
        timesteps = timesteps[anchor_start:]
        sigma = float(schedule.sigma_for(float(timesteps[0])))
        noise_latents = ((1.0 - sigma) * still + sigma * noise_latents.astype(mx.float32))
        print(f"[i2v] anchored init: start t={timesteps[0]:.0f} sigma={sigma:.3f}, {len(timesteps)} steps", flush=True)
    latents = pin(noise_latents)
    last = len(timesteps) - 1
    for i, t in enumerate(timesteps):
        ts = mx.concatenate([mx.zeros((batch, per_frame), dtype=mx.float32),
                             mx.full((batch, tokens - per_frame), float(t), dtype=mx.float32)], axis=1)
        pred = model(latents.astype(mx.float16), encoder_hidden_states, ts, freqs_cis)
        ni, pn = latents.astype(mx.float32), pred.astype(mx.float32)
        if i < last:
            renoise = mx.array(rng.standard_normal(tuple(latents.shape)).astype(np.float32))
            latents = dmd_step(latents=ni, noise_input_latent=ni, pred_noise=pn, schedule=schedule,
                               timestep=float(t), next_timestep=float(timesteps[i + 1]),
                               noise=renoise).astype(latents.dtype)
        else:
            latents = pred_noise_to_pred_video(pn, ni, schedule.sigma_for(float(t))).astype(latents.dtype)
        latents = pin(latents)
        mx.eval(latents)
    return latents


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MLX Wan2.2-5B T2V (encode → DiT DMD → TAEHV/VAE decode)"
    )
    parser.add_argument(
        "--prompt",
        default="A red fox trotting through a snowy pine forest at golden hour, cinematic",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("video_samples/demo_5b/fox_5b_mlx.mp4"),
    )
    parser.add_argument(
        "--text-encoder-root",
        type=Path,
        default=None,
        help="Root with text_encoder/ + tokenizer/",
    )
    parser.add_argument(
        "--prompt-embeds-cache",
        type=Path,
        default=None,
        help="Explicit .npy UMT5 embedding cache file. Overrides the automatic "
        "content-addressed cache (--prompt-cache).",
    )
    parser.add_argument(
        "--prompt-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cache prompt embeddings under ~/.cache/fastvideo/prompt_embeds so "
        "repeat runs skip the text encoder entirely. Default: on.",
    )
    parser.add_argument(
        "--text-encoder-device",
        choices=("auto", "cpu", "mps"),
        default="cpu",
        help="Device for UMT5 encoding. CPU is safest beside the 5B MLX DiT.",
    )
    parser.add_argument(
        "--enhance-prompt",
        action="store_true",
        help="Apply deterministic local cinematic prompt enrichment before UMT5.",
    )
    parser.add_argument(
        "--enhance-prompt-backend",
        choices=("template",),
        default="template",
        help="Prompt enrichment backend.",
    )
    parser.add_argument(
        "--dit-checkpoint",
        type=Path,
        default=None,
    )
    parser.add_argument("--dit-config", type=Path, default=None)
    parser.add_argument(
        "--mlx-checkpoint",
        type=Path,
        default=None,
        help="Packed FastMetal-5B-QAD MLX DiT directory (mlx_dit.json + mlx_dit.safetensors). "
        "If omitted, a FastMetal directory passed as --text-encoder-root is used when it "
        "already contains those files.",
    )
    parser.add_argument("--vae-root", type=Path, default=None)
    parser.add_argument("--image", type=Path, default=None, help="I2V: first frame to animate from")
    parser.add_argument("--anchor-start", type=int, default=0, help="I2V: skip this many DMD steps and start "
                        "every frame from a re-noised still of --image (0 = off, pure noise)")
    parser.add_argument("--init-video", type=Path, default=None, help="V2V: restyle this real clip (its motion "
                        "and layout seed the generation)")
    parser.add_argument("--init-sigma", type=float, default=0.75, help="V2V: noise added to the clip, 0-1 "
                        "(lower keeps more real motion, higher restyles more)")
    parser.add_argument("--init-start", type=float, default=0.0, help="V2V: start time in the clip, seconds")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument(
        "--num-frames",
        type=int,
        default=DEFAULT_NUM_FRAMES,
        help="Pixel frames (121 at 24fps = 5.04 seconds)",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--renoise-seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--flow-shift", type=float, default=5.0)
    parser.add_argument("--dmd-denoising-steps", default="1000,757,522")
    parser.add_argument(
        "--no-warp",
        action="store_true",
        help="Disable schedule warping (debug only).",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Generate fewer frames then RIFE-interpolate to --num-frames.",
    )
    parser.add_argument("--fast-factor", type=int, default=2)
    parser.add_argument("--fast-sharpen", type=float, default=0.6)
    parser.add_argument(
        "--fast-spatial",
        action="store_true",
        help="Denoise and decode at reduced spatial resolution, then resample "
        "the decoded frames up to the target size.",
    )
    parser.add_argument("--fast-spatial-scale", type=int, default=2)
    parser.add_argument(
        "--fast-spatial-upsample-mode",
        choices=PIXEL_UPSAMPLE_MODES,
        default=DEFAULT_PIXEL_UPSAMPLE_MODE,
    )
    parser.add_argument("--fast-spatial-sharpen", type=float, default=DEFAULT_FAST_SPATIAL_SHARPEN)
    parser.add_argument(
        "--refine",
        action="store_true",
        help="Two-pass DMD: coarse denoise, upsample/re-noise, full-res denoise.",
    )
    parser.add_argument("--refine-scale", type=int, default=2)
    parser.add_argument(
        "--refine-upsample-mode",
        choices=("bilinear", "nearest"),
        default="bilinear",
    )
    parser.add_argument("--no-refine-add-noise", action="store_true")
    parser.add_argument(
        "--decode-backend",
        choices=("taehv", "taehv-torch", "wan-vae"),
        default="taehv",
    )
    parser.add_argument("--save-latents", type=Path, default=None)
    parser.add_argument("--metrics-json", type=Path, default=None,
                        help="Write measured run metadata as JSON for reports or galleries.")
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile the DiT forward with mx.compile; fallback to eager on failure.",
    )
    args = parser.parse_args()

    if args.fast_factor < 2:
        parser.error("--fast-factor must be at least 2")
    # --fast-spatial used to be rejected here because it upsampled the completed
    # 48-channel latent, which is out of distribution for the decoder and gave
    # black or noisy video. The upsample now runs on decoded frames, so the
    # latent never leaves the grid it was denoised on and the mode is usable.
    if args.refine and args.fast_spatial:
        print("[wan22] --refine takes precedence over --fast-spatial")

    args.mlx_checkpoint = resolve_mlx_checkpoint(args.mlx_checkpoint, args.text_encoder_root)
    if args.mlx_checkpoint is not None:
        if args.text_encoder_root is None and (args.mlx_checkpoint / "text_encoder").is_dir():
            args.text_encoder_root = args.mlx_checkpoint
        if args.vae_root is None and (args.mlx_checkpoint / "vae").is_dir():
            args.vae_root = args.mlx_checkpoint / "vae"
    try:
        raise_if_unsupported_mlx_checkpoint(args.mlx_checkpoint, args.dit_checkpoint)
    except UnsupportedMLXCheckpointError as exc:
        raise SystemExit(str(exc)) from exc

    args.text_encoder_root, args.dit_checkpoint, args.dit_config, args.vae_root = _resolve_model_paths(
        text_encoder_root=args.text_encoder_root,
        dit_checkpoint=args.dit_checkpoint,
        dit_config=args.dit_config,
        vae_root=args.vae_root,
        mlx_checkpoint=args.mlx_checkpoint,
        decode_backend=args.decode_backend,
    )
    target_frames = args.num_frames
    if args.fast:
        args.num_frames = aligned_keyframe_count(target_frames, args.fast_factor)
        print(
            f"[wan22 fast] generating {args.num_frames} frames, "
            f"RIFE {args.fast_factor}x -> {target_frames}"
        )

    import mlx.core as mx
    import torch

    from examples.inference.basic.mlx_wan_prompt_to_video import (
        _postprocess_video,
        encode_prompt,
        make_rotary_embeddings,
    )
    from fastvideo.mlx_runtime.fast_spatial import plan_fast_spatial
    from fastvideo.mlx_runtime.refine import (
        default_refine_timesteps,
        plan_refine_resolutions,
        prepare_refine_latents,
    )
    from fastvideo.mlx_runtime.wan22 import (
        mlx_wan22_dit_from_diffusers_safetensors,
        mlx_wan22_dit_from_mlx_checkpoint,
    )
    from fastvideo.mlx_runtime.wan22_sample import build_wan22_dmd_schedule, sample_wan22_dmd
    from fastvideo.mlx_runtime.wan_vae import decode_latents_to_video

    if args.mlx_checkpoint is not None:
        config = json.loads((args.mlx_checkpoint / "mlx_dit.json").read_text())["config"]
    else:
        config = json.loads(args.dit_config.read_text())
    patch_size = tuple(config.get("patch_size", (1, 2, 2)))
    if args.refine:
        active_plan = plan_refine_resolutions(
            height=args.height, width=args.width, num_frames=args.num_frames,
            spatial_scale=args.refine_scale, vae_spatial_compression=16,
            vae_temporal_compression=4, patch_size=patch_size, enabled=True,
        )
        spatial_mode = "refine"
    elif args.fast_spatial:
        fast_spatial_plan = plan_fast_spatial(
            height=args.height, width=args.width, num_frames=args.num_frames,
            spatial_scale=args.fast_spatial_scale, vae_spatial_compression=16,
            vae_temporal_compression=4, patch_size=patch_size,
            upsample_mode=args.fast_spatial_upsample_mode,
            sharpen=args.fast_spatial_sharpen, enabled=True,
        )
        active_plan = fast_spatial_plan.plan
        spatial_mode = "fast_spatial"
    else:
        active_plan = plan_refine_resolutions(
            height=args.height, width=args.width, num_frames=args.num_frames,
            spatial_scale=1, vae_spatial_compression=16, vae_temporal_compression=4,
            patch_size=patch_size, enabled=False,
        )
        spatial_mode = "off"
    lat_h, lat_w = active_plan.stage1_latent_height, active_plan.stage1_latent_width
    lat_t = active_plan.latent_frames
    in_ch = int(config["in_channels"])
    print(f"[5B] latent {in_ch}x{lat_t}x{lat_h}x{lat_w}", flush=True)

    total_start = time.perf_counter()
    prompt_for_encode = args.prompt
    enhance_backend = None
    enhance_elapsed_s = 0.0
    if args.enhance_prompt:
        from fastvideo.mlx_runtime.prompt_enhance import enhance_prompt

        enhancement = enhance_prompt(args.prompt, backend=args.enhance_prompt_backend)
        prompt_for_encode = enhancement.enhanced
        enhance_backend = enhancement.backend
        enhance_elapsed_s = enhancement.elapsed_s
        print(f"[enhance] backend={enhance_backend} in {enhance_elapsed_s:.2f}s", flush=True)
        print(f"[enhance] prompt: {prompt_for_encode}", flush=True)

    t0 = time.perf_counter()
    prompt_cache_fingerprint = _prompt_cache_fingerprint(
        prompt=args.prompt,
        prompt_used=prompt_for_encode,
        enhance_prompt=args.enhance_prompt,
        enhance_prompt_backend=args.enhance_prompt_backend,
        text_encoder_root=args.text_encoder_root,
        max_sequence_length=512,
        dtype="fp16",
    )
    prompt_cache_path = args.prompt_embeds_cache
    if prompt_cache_path is None and args.prompt_cache:
        prompt_cache_path = _default_prompt_cache_path(prompt_cache_fingerprint)
    cached_embeds = load_prompt_cache(
        prompt_cache_path,
        prompt_cache_fingerprint,
    )
    if cached_embeds is not None:
        embeds = torch.from_numpy(cached_embeds).contiguous()
    else:
        embeds = encode_prompt(
            model_root=args.text_encoder_root,
            prompt=prompt_for_encode,
            max_sequence_length=512,
            device_arg=args.text_encoder_device,
            dtype_arg="fp16",
        )
        save_prompt_cache(
            prompt_cache_path,
            embeds.cpu().numpy(),
            prompt_cache_fingerprint,
        )
    ehs = mx.array(embeds.numpy()).astype(mx.float16)
    prompt_encode_s = time.perf_counter() - t0
    print(f"[5B] prompt encoded {tuple(ehs.shape)} in {prompt_encode_s:.1f}s", flush=True)

    t1 = time.perf_counter()
    if args.mlx_checkpoint is not None:
        dit = mlx_wan22_dit_from_mlx_checkpoint(
            args.mlx_checkpoint,
            compile=args.compile,
        )
    else:
        dit = mlx_wan22_dit_from_diffusers_safetensors(
            args.dit_checkpoint,
            args.dit_config,
            dtype="fp16",
            compile=args.compile,
        )
    dit_load_s = time.perf_counter() - t1
    print(f"[5B] DiT loaded in {dit_load_s:.1f}s", flush=True)

    freqs = make_rotary_embeddings(config, latent_frames=lat_t, latent_height=lat_h, latent_width=lat_w)
    gen = torch.Generator().manual_seed(args.seed)
    noise = mx.array(
        torch.randn(1, in_ch, lat_t, lat_h, lat_w, generator=gen, dtype=torch.float32).numpy()).astype(mx.float16)

    image_latent = None
    if args.image is not None:
        image_latent = _encode_first_frame(args.image, args.vae_root, height=lat_h * 16, width=lat_w * 16,
                                           frames=args.num_frames if args.anchor_start else 1)
        print(f"[i2v] first frame encoded {tuple(image_latent.shape)} from {args.image}", flush=True)
    init_latent = None
    if args.init_video is not None:
        init_latent = _encode_video(args.init_video, args.vae_root, height=lat_h * 16, width=lat_w * 16,
                                    frames=args.num_frames, fps=args.fps, start=args.init_start)
        print(f"[v2v] clip encoded {tuple(init_latent.shape)} from {args.init_video} @ {args.init_start}s", flush=True)
    steps = [int(s) for s in args.dmd_denoising_steps.split(",") if s.strip()]
    t2 = time.perf_counter()
    mx.reset_peak_memory()
    if init_latent is not None:
        sampler = lambda *a, **k: _sample_v2v(*a, init_latent=init_latent, init_sigma=args.init_sigma,
                                              image_latent=image_latent, **k)
    else:
        sampler = sample_wan22_dmd if image_latent is None else (
            lambda *a, **k: _sample_i2v(*a, image_latent=image_latent, anchor_start=args.anchor_start, **k))
    latents = sampler(
        dit,
        ehs,
        noise,
        freqs,
        dmd_denoising_steps=steps,
        flow_shift=args.flow_shift,
        warp_denoising_step=not args.no_warp,
        seed=args.renoise_seed,
    )
    if spatial_mode == "refine":
        schedule, warped_steps = build_wan22_dmd_schedule(
            steps, flow_shift=args.flow_shift, warp_denoising_step=not args.no_warp,
        )
        # The grid opens at sigma == 1, where the hand-off
        # `(1 - sigma) * upsampled + sigma * noise` weights stage 1 at zero and
        # refine silently becomes a plain full-res run. Drop the leading
        # full-noise steps so stage 1 actually reaches stage 2.
        stage2_warped = default_refine_timesteps(schedule, warped_steps)
        stage2_steps = steps[len(warped_steps) - len(stage2_warped):]
        sigma = schedule.sigma_for(stage2_warped[0])
        print(f"[5B refine] stage-2 steps={stage2_steps} sigma={sigma:.4f} "
              f"(stage-1 weight {1.0 - sigma:.4f})", flush=True)
        latents = prepare_refine_latents(
            latents, scale=args.refine_scale, sigma=sigma,
            add_noise_flag=not args.no_refine_add_noise,
            upsample_mode=args.refine_upsample_mode, seed=args.renoise_seed + 1,
        )
        freqs_stage2 = make_rotary_embeddings(
            config, latent_frames=lat_t,
            latent_height=active_plan.stage2_latent_height,
            latent_width=active_plan.stage2_latent_width,
        )
        latents = sample_wan22_dmd(
            dit, ehs, latents, freqs_stage2, dmd_denoising_steps=stage2_steps,
            flow_shift=args.flow_shift, warp_denoising_step=not args.no_warp,
            seed=args.renoise_seed + 2,
        )
    # spatial_mode == "fast_spatial" leaves the latents on the stage-1 grid;
    # the resample happens after decode, in _postprocess_video.
    denoise_s = time.perf_counter() - t2
    peak = mx.get_peak_memory() / (1024**3)
    print(f"[5B] denoise {len(steps)} steps in {denoise_s:.1f}s, peak {peak:.2f} GiB", flush=True)

    latents_np = np.array(latents.astype(mx.float32))
    if args.save_latents is not None:
        args.save_latents.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.save_latents, latents=latents_np, prompt=args.prompt, seed=args.seed)
        print(f"[5B] wrote latents {args.save_latents}", flush=True)

    if spatial_mode == "refine":
        del freqs_stage2
    del dit, latents, ehs, noise, freqs
    cleanup_mlx()

    metrics = decode_latents_to_video(
        latents_np,
        args.output_path,
        fps=args.fps,
        backend=args.decode_backend,
        vae_dir=args.vae_root if args.decode_backend == "wan-vae" else None,
        z_dim=in_ch,
    )
    # One h264 round-trip for both post-decode passes (see _postprocess_video).
    rife_s = 0.0
    rife_request = ({
        "factor": args.fast_factor,
        "target_frames": target_frames,
        "sharpen": args.fast_sharpen,
    } if args.fast else None)
    spatial_request = fast_spatial_plan if spatial_mode == "fast_spatial" else None
    if rife_request is not None or spatial_request is not None:
        rife_start = time.perf_counter()
        _postprocess_video(
            video_path=args.output_path, fps=args.fps,
            rife=rife_request, spatial=spatial_request,
        )
        rife_s = time.perf_counter() - rife_start
    print(f"[5B] decoded via {metrics['backend']} in {metrics['decode_s']:.1f}s → {args.output_path}", flush=True)
    summary = {
        "output_path": str(args.output_path.resolve()),
        "prompt": args.prompt,
        "prompt_used": prompt_for_encode,
        "enhance_prompt": args.enhance_prompt,
        "enhance_backend": enhance_backend,
        "enhance_elapsed_s": round(enhance_elapsed_s, 3),
        "height": args.height,
        "width": args.width,
        "fps": args.fps,
        "target_frames": target_frames,
        "generated_frames": args.num_frames,
        "seed": args.seed,
        "renoise_seed": args.renoise_seed,
        "dmd_denoising_steps": steps,
        "flow_shift": args.flow_shift,
        "warp": not args.no_warp,
        "spatial_mode": spatial_mode,
        "fast": args.fast,
        "fast_factor": args.fast_factor if args.fast else None,
        "fast_spatial_scale": args.fast_spatial_scale if args.fast_spatial else None,
        "refine_scale": args.refine_scale if args.refine else None,
        "decode_backend": args.decode_backend,
        "prompt_encode_s": round(prompt_encode_s, 3),
        "dit_load_s": round(dit_load_s, 3),
        "denoise_s": round(denoise_s, 3),
        "decode_s": round(metrics["decode_s"], 3),
        "rife_s": round(rife_s, 3),
        "wall_total_s": round(time.perf_counter() - total_start, 3),
        "peak_gib": round(peak, 3),
        "latent_shape": [in_ch, lat_t, lat_h, lat_w],
        "stage2_latent_shape": [in_ch, lat_t, active_plan.stage2_latent_height, active_plan.stage2_latent_width],
        "mlx_checkpoint": str(args.mlx_checkpoint.resolve()) if args.mlx_checkpoint else None,
    }
    if args.metrics_json is not None:
        args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_json.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"[5B] wrote metrics {args.metrics_json}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
