"""spike-video runs Wan generation inside the lane. The model run itself is
replaced by a stand-in interpreter (the real one is 10 minutes of GPU)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import spike_lane  # noqa: E402

VIDEO = ROOT / "bin" / "spike-video"

FAKE_PY = """#!/bin/sh
printf '%s\\n' "$@" > "$FAKE_ARGS"
echo "$SPIKE_LANE_HOLDER" > "$FAKE_HOLDER"
"""


@pytest.fixture
def env(tmp_path, monkeypatch):
    fv = tmp_path / "FastVideo"
    (fv / ".venv" / "bin").mkdir(parents=True)
    py = fv / ".venv" / "bin" / "python"
    py.write_text(FAKE_PY)
    py.chmod(0o755)
    monkeypatch.setenv("SPIKE_LANE_DIR", str(tmp_path / "lane"))
    e = os.environ.copy()
    e.update(FASTVIDEO_HOME=str(fv), WAN_MODEL=str(tmp_path / "model"),
             WAN_OUT=str(tmp_path / "out"), FAKE_ARGS=str(tmp_path / "args"),
             FAKE_HOLDER=str(tmp_path / "holder"),
             PATH="/usr/bin:/bin")  # no ollama: eviction must be a quiet no-op
    return tmp_path, e


def test_runs_wan_script_with_model_prompt_and_output(env):
    tmp, e = env
    r = subprocess.run([str(VIDEO), "a red fox in snow", "--fast"],
                       capture_output=True, text=True, env=e)
    assert r.returncode == 0, r.stderr
    args = (tmp / "args").read_text().splitlines()
    assert args[0].endswith("mlx_wan_prompt_to_video.py")
    assert args[args.index("--prompt") + 1] == "a red fox in snow"
    assert args[args.index("--model-root") + 1] == str(tmp / "model")
    assert args[args.index("--mlx-checkpoint") + 1] == str(tmp / "model")
    out = Path(args[args.index("--output-path") + 1])
    assert out.parent == tmp / "out" and out.suffix == ".mp4"
    assert "--fast" in args
    assert (tmp / "holder").read_text().strip() == "video"


def test_refuses_when_lane_busy(env):
    tmp, e = env
    fd = spike_lane.acquire("coder", ["dag-coder"])
    try:
        r = subprocess.run([str(VIDEO), "x"], capture_output=True, text=True, env=e)
    finally:
        os.close(fd)
    assert r.returncode == 75
    assert "coder" in r.stderr
    assert not (tmp / "args").exists()


def test_requires_a_prompt(env):
    _, e = env
    r = subprocess.run([str(VIDEO)], capture_output=True, text=True, env=e)
    assert r.returncode == 2
    assert "usage" in r.stderr.lower()
