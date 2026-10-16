"""engines/wan22_i2v.py: encoding a still or clip to Wan2.2 latents without the GPU's memory piling up.
Encoding a 61-frame held still at 1280x704 the way diffusers does peaked at 51 GB and came back wrong from
the fifth latent frame on; the freed buffers of each 4-frame chunk stayed cached until the end. The engine
runs the same chunk loop and hands those buffers back after every chunk. Runs a small real Wan VAE (the
5B's own layout, scaled down) on the CPU in FastVideo's venv."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
VENV = Path.home() / "FastVideo" / ".venv" / "bin" / "python"
pytestmark = pytest.mark.skipif(not VENV.exists(), reason="needs FastVideo's venv (torch, diffusers)")

PROBE = r'''
import json, sys
import torch
from diffusers import AutoencoderKLWan
sys.path.insert(0, sys.argv[1])
import wan22_i2v

torch.manual_seed(0)
vae = AutoencoderKLWan(base_dim=8, decoder_base_dim=8, z_dim=4, dim_mult=[1, 2, 2, 2], num_res_blocks=1,
                       attn_scales=[], temperal_downsample=[False, True, True], in_channels=12, out_channels=12,
                       patch_size=2, is_residual=True).eval()
x = torch.randn(1, 3, 13, 64, 64)
flushes = []
with torch.no_grad():
    want = vae.encode(x).latent_dist.mode()
    got = wan22_i2v._vae_encode(vae, x, flush=lambda: flushes.append(1))
print(json.dumps({"shape": list(got.shape), "want": list(want.shape), "diff": float((got - want).abs().max()),
                  "flushes": len(flushes)}))
'''


def test_the_still_is_encoded_exactly_as_diffusers_does_but_freed_after_every_chunk():
    r = subprocess.run([str(VENV), "-c", PROBE, str(ROOT / "engines")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    got = json.loads(r.stdout.strip().splitlines()[-1])
    assert got["shape"] == got["want"]
    assert got["diff"] == 0.0
    assert got["flushes"] == 4          # 13 frames: the first alone, then three chunks of four


CACHE = r'''
import json, shutil, sys
from pathlib import Path
import numpy as np, torch
from diffusers import AutoencoderKLWan
from PIL import Image
sys.path.insert(0, sys.argv[1])
import wan22_i2v

tmp = Path(sys.argv[2])
torch.manual_seed(0)
AutoencoderKLWan(base_dim=8, decoder_base_dim=8, z_dim=4, dim_mult=[1, 2, 2, 2], num_res_blocks=1, attn_scales=[],
                 temperal_downsample=[False, True, True], in_channels=12, out_channels=12, patch_size=2,
                 is_residual=True, latents_mean=[0.0] * 4, latents_std=[1.0] * 4).save_pretrained(tmp / "vae")
rng = np.random.default_rng(0)
for name in ("a", "b"):
    Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(tmp / f"{name}.png")
enc = lambda name: np.array(wan22_i2v._encode_first_frame(tmp / f"{name}.png", tmp / "vae", height=64, width=64,
                                                          frames=9, cache=tmp / "cache"))
first = enc("a")
shutil.rmtree(tmp / "vae")                     # a second encode of the same still must not need the VAE
again = enc("a")
try:
    enc("b"); other = "encoded without a VAE"
except Exception:
    other = "needed the VAE"
print(json.dumps({"same": bool((first == again).all()), "shape": list(first.shape), "other": other}))
'''


def test_a_stills_latents_are_kept_so_retakes_skip_the_encode(tmp_path):
    r = subprocess.run([str(VENV), "-c", CACHE, str(ROOT / "engines"), str(tmp_path)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    got = json.loads(r.stdout.strip().splitlines()[-1])
    assert got["same"] and got["shape"] == [1, 4, 3, 4, 4]   # 9 frames -> 3 latent frames; 64 px / 16 -> 4
    assert got["other"] == "needed the VAE"     # another picture is encoded afresh
