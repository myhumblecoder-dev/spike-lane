"""The lane is real flock semantics between real processes — no mocks."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import spike_lane  # noqa: E402

CLI = ROOT / "bin" / "spike-lane"


@pytest.fixture(autouse=True)
def lane_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SPIKE_LANE_DIR", str(tmp_path))
    return tmp_path


def _hold(name: str, seconds: float = 30) -> subprocess.Popen:
    """Another process holding the lane via the module, until killed."""
    code = (
        "import sys, time; sys.path.insert(0, %r); import spike_lane;"
        "fd = spike_lane.acquire(%r, ['holder']); print('held', flush=True);"
        "time.sleep(%r)" % (str(ROOT), name, seconds)
    )
    p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE,
                         text=True, env=os.environ.copy())
    assert p.stdout.readline().strip() == "held"
    return p


def test_acquire_free_lane_records_holder(lane_dir):
    fd = spike_lane.acquire("video", ["wan", "--prompt", "x"])
    try:
        rec = json.loads((lane_dir / "lane.lock").read_text())
        assert rec["name"] == "video"
        assert rec["pid"] == os.getpid()
        assert rec["cmd"] == ["wan", "--prompt", "x"]
    finally:
        os.close(fd)


def test_busy_lane_refuses_without_wait_and_names_holder():
    p = _hold("coder")
    try:
        with pytest.raises(spike_lane.LaneBusy) as e:
            spike_lane.acquire("video", wait=0)
        assert e.value.holder["name"] == "coder"
        assert "coder" in str(e.value)
    finally:
        p.kill(); p.wait()


def test_waiter_does_not_wipe_holders_record(lane_dir):
    p = _hold("coder")
    try:
        with pytest.raises(spike_lane.LaneBusy):
            spike_lane.acquire("video", wait=0)
        assert json.loads((lane_dir / "lane.lock").read_text())["name"] == "coder"
    finally:
        p.kill(); p.wait()


def test_dead_holder_frees_the_lane():
    p = _hold("coder")
    p.kill(); p.wait()
    fd = spike_lane.acquire("video", wait=0)
    os.close(fd)


def test_waiting_acquire_gets_lane_when_holder_finishes():
    p = _hold("coder", seconds=1.0)
    t0 = time.time()
    fd = spike_lane.acquire("video", wait=20, poll=0.1)
    os.close(fd)
    assert time.time() - t0 < 15
    p.wait()


def test_holder_reports_free_then_busy():
    assert spike_lane.holder() is None
    p = _hold("image")
    try:
        assert spike_lane.holder()["name"] == "image"
    finally:
        p.kill(); p.wait()
    assert spike_lane.holder() is None


def test_parse_ollama_ps():
    out = (
        "NAME                     ID              SIZE     PROCESSOR    UNTIL\n"
        "spike-coder-v3:latest    abc123          19 GB    100% GPU     4 minutes from now\n"
    )
    assert spike_lane.parse_ollama_ps(out) == ["spike-coder-v3:latest"]
    assert spike_lane.parse_ollama_ps("NAME    ID    SIZE    PROCESSOR    UNTIL \n") == []
    assert spike_lane.parse_ollama_ps("") == []


# ------------------------------------------------------------------- CLI

def _cli(*args, **kw):
    return subprocess.run([sys.executable, str(CLI), *args], capture_output=True,
                          text=True, env=os.environ.copy(), **kw)


def test_cli_run_holds_lane_for_the_child_and_releases_after():
    # The child asks for status while it runs: it must see itself as holder.
    r = _cli("run", "video", "--", sys.executable, str(CLI), "status")
    assert r.returncode == 0, r.stderr
    assert "video" in r.stdout
    assert spike_lane.holder() is None


def test_cli_run_propagates_child_exit_code():
    r = _cli("run", "video", "--", sys.executable, "-c", "raise SystemExit(7)")
    assert r.returncode == 7


def test_cli_run_refuses_busy_lane_with_tempfail():
    p = _hold("coder")
    try:
        r = _cli("run", "video", "--", "true")
        assert r.returncode == 75
        assert "coder" in r.stderr
    finally:
        p.kill(); p.wait()


def test_cli_status_free():
    r = _cli("status")
    assert r.returncode == 0
    assert "free" in r.stdout
