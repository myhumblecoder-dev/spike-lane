"""spike-image runs Z-Image Turbo (mflux) inside the lane. The generator is
replaced by a stand-in executable; the real one downloads ~12 GB of weights."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import spike_lane  # noqa: E402

IMAGE = ROOT / "bin" / "spike-image"

FAKE_GEN = """#!/bin/sh
printf '%s\\n' "$@" > "$FAKE_ARGS"
echo "$SPIKE_LANE_HOLDER" > "$FAKE_HOLDER"
"""


@pytest.fixture
def env(tmp_path, monkeypatch):
    gen = tmp_path / "bin" / "mflux-generate-z-image-turbo"
    gen.parent.mkdir()
    gen.write_text(FAKE_GEN)
    gen.chmod(0o755)
    monkeypatch.setenv("SPIKE_LANE_DIR", str(tmp_path / "lane"))
    e = os.environ.copy()
    e.update(IMAGE_OUT=str(tmp_path / "out"), IMAGE_MODEL=str(tmp_path / "zi-q8"), FAKE_ARGS=str(tmp_path / "args"),
             FAKE_HOLDER=str(tmp_path / "holder"),
             PATH=f"{gen.parent}:/usr/bin:/bin")  # no ollama: eviction no-ops
    return tmp_path, e


def _flag(args, name):
    return args[args.index(name) + 1]


def test_runs_z_image_turbo_with_prompt_output_and_defaults(env):
    tmp, e = env
    r = subprocess.run([str(IMAGE), "a puffin on a sea cliff", "--seed", "7"],
                       capture_output=True, text=True, env=e)
    assert r.returncode == 0, r.stderr
    args = (tmp / "args").read_text().splitlines()
    assert _flag(args, "--prompt") == "a puffin on a sea cliff"
    out = Path(_flag(args, "--output"))
    assert out.parent == tmp / "out" and out.suffix == ".png"
    assert _flag(args, "--steps") == "9"
    # The local mflux-community q8 build is already quantized: point at it,
    # never re-quantize (the official repo is a 33 GB download).
    assert _flag(args, "--model") == str(tmp / "zi-q8")
    assert "-q" not in args and "--quantize" not in args
    assert _flag(args, "--seed") == "7"  # extra flags pass through
    assert (tmp / "holder").read_text().strip() == "image"


def test_refuses_when_lane_busy(env):
    tmp, e = env
    fd = spike_lane.acquire("video", ["spike-video"])
    try:
        r = subprocess.run([str(IMAGE), "x"], capture_output=True, text=True, env=e)
    finally:
        os.close(fd)
    assert r.returncode == 75
    assert "video" in r.stderr
    assert not (tmp / "args").exists()


def test_requires_a_prompt(env):
    _, e = env
    r = subprocess.run([str(IMAGE)], capture_output=True, text=True, env=e)
    assert r.returncode == 2
    assert "usage" in r.stderr.lower()
