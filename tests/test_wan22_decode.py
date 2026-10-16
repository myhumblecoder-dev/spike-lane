"""engines/wan22_i2v.py: decoding latents to frames a few latent frames at a time.
FastVideo's MLX TAEHV decoder evaluates the whole clip as one graph; at 1280x704 its 64-channel full-size
layers alone are ~15 GB per tensor, so the last seconds of an HD clip spiked the job to ~30 GB. Every layer
works frame by frame except the MemBlocks, which also read the previous frame of their input, so carrying
that one frame across chunks gives the same frames for a fraction of the memory. Runs the real taew2_2
weights on small latents in FastVideo's venv."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
VENV = Path.home() / "FastVideo" / ".venv" / "bin" / "python"
CKPT = Path.home() / ".cache" / "fastvideo" / "taehv" / "taew2_2.pth"
pytestmark = pytest.mark.skipif(not (VENV.exists() and CKPT.exists()), reason="needs FastVideo's venv and taew2_2.pth")

PROBE = r'''
import json, sys
import numpy as np, mlx.core as mx
sys.path.insert(0, sys.argv[1])
import wan22_i2v
from fastvideo.mlx_runtime.wan_vae import decode_latents_taehv_mlx

h, w = int(sys.argv[2]), int(sys.argv[3])
lat = np.random.default_rng(0).standard_normal((1, 48, 6, h, w)).astype(np.float32)
mx.reset_peak_memory()
want = decode_latents_taehv_mlx(lat)
whole = mx.get_peak_memory()
mx.clear_cache(); mx.reset_peak_memory()
flushes = []
got = wan22_i2v._decode_taehv(lat, flush=lambda: flushes.append(1))
chunked = mx.get_peak_memory()
print(json.dumps({"shape": list(got.shape), "want": list(want.shape), "diff": float(np.abs(got - want).max()),
                  "flushes": len(flushes), "peak_ratio": chunked / whole}))
'''


def _probe(h, w):
    r = subprocess.run([str(VENV), "-c", PROBE, str(ROOT / "engines"), str(h), str(w)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_chunked_decode_gives_the_same_frames_as_the_whole_clip_decode_for_far_less_memory():
    got = _probe(22, 40)                        # a quarter of an HD clip's latent area
    assert got["shape"] == got["want"] == [1, 21, 352, 640, 3]   # 6 latent frames -> 24 - 3 trimmed
    assert got["diff"] < 1e-4
    assert got["flushes"] == 6                  # one latent frame at a time
    assert got["peak_ratio"] < 0.5
