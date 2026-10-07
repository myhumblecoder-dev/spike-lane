"""One heavy workload at a time on this machine.

Spike's Mac Studio is a 36 GB M4 Max. Its heavy workloads each want most of
that memory on their own: a coder grind holds ~17 GB of Ollama weights plus KV
cache; Wan video generation peaks at ~22 GB for the 14B model. Two at once
never errors, it swaps, and everything gets several times slower.

The lane is a single exclusive flock. Whoever holds it is the one heavy
workload running. The kernel drops a flock when its holder dies, so a crash
can never leave the lane stuck.

The protocol is deliberately small enough to reimplement anywhere without
importing this module (dag-coder does exactly that, so OpenClaw and the media
tools never depend on each other):

  1. open  $SPIKE_LANE_DIR/lane.lock  (default ~/.local/state/spike-lane)
     WITHOUT truncating it — a waiter must not wipe the holder's record;
  2. flock(LOCK_EX), polling with LOCK_NB while willing to wait;
  3. once held, truncate the file and write one JSON object:
     {"name", "pid", "cmd", "started"};
  4. keep the fd open for the life of the work; closing it (or dying) frees
     the lane.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

EX_TEMPFAIL = 75  # sysexits.h: "try again later" — exactly what a busy lane means


class LaneBusy(Exception):
    """The lane is held by someone else. `.holder` is their record, if any."""

    def __init__(self, holder: dict | None):
        self.holder = holder
        super().__init__(describe(holder))


def lane_path() -> Path:
    d = Path(os.environ.get("SPIKE_LANE_DIR")
             or Path.home() / ".local" / "state" / "spike-lane")
    d.mkdir(parents=True, exist_ok=True)
    return d / "lane.lock"


def _read_record(fd: int) -> dict | None:
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        raw = os.read(fd, 64 * 1024).decode(errors="replace").strip()
        return json.loads(raw) if raw else None
    except (OSError, ValueError):
        return None


def acquire(name: str, cmd: list[str] | None = None, *,
            wait: float | None = 0.0, poll: float = 1.0) -> int:
    """Take the lane and return the fd that owns it (close it to release).

    `wait` is how many seconds to keep trying: 0 refuses at once, None waits
    forever. Raises LaneBusy with the holder's record when time runs out.
    """
    fd = os.open(lane_path(), os.O_RDWR | os.O_CREAT, 0o644)  # never O_TRUNC
    deadline = None if wait is None else time.monotonic() + wait
    announced = False
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if deadline is not None and time.monotonic() >= deadline:
                rec = _read_record(fd)
                os.close(fd)
                raise LaneBusy(rec)
            if not announced:
                print(f"⏳ waiting for the lane — held by {describe(_read_record(fd))}",
                      file=sys.stderr)
                announced = True
            time.sleep(poll)
    rec = {"name": name, "pid": os.getpid(), "cmd": list(cmd or []),
           "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    os.ftruncate(fd, 0)
    os.lseek(fd, 0, os.SEEK_SET)
    os.write(fd, (json.dumps(rec) + "\n").encode())
    return fd


def holder() -> dict | None:
    """The current holder's record, or None if the lane is free."""
    fd = os.open(lane_path(), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return _read_record(fd) or {"name": "unknown"}
        fcntl.flock(fd, fcntl.LOCK_UN)
        return None
    finally:
        os.close(fd)


def describe(h: dict | None) -> str:
    if not h:
        return "an unknown holder"
    cmd = " ".join(h.get("cmd") or [])
    s = f"{h.get('name', 'unknown')} (pid {h.get('pid', '?')}, since {h.get('started', '?')})"
    return f"{s}: {cmd}" if cmd else s


# ---------------------------------------------------------------- ollama

def parse_ollama_ps(out: str) -> list[str]:
    """Model names from `ollama ps` output (first column, header skipped)."""
    lines = [ln for ln in (out or "").splitlines() if ln.strip()]
    return [ln.split()[0] for ln in lines[1:]]


def evict_ollama(run=subprocess.run) -> list[str]:
    """Unload every model Ollama holds. A coder grind releases the lane when
    it ends, but OLLAMA_KEEP_ALIVE keeps its ~17 GB of weights resident — the
    next workload has to clear them itself. Best-effort: no ollama, no-op."""
    try:
        p = run(["ollama", "ps"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return []
    names = parse_ollama_ps(p.stdout) if p.returncode == 0 else []
    for n in names:
        print(f"⏏  unloading ollama model {n}", file=sys.stderr)
        try:
            run(["ollama", "stop", n], capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            pass
    return names


# ------------------------------------------------------------------- CLI

def _wait_arg(v: str) -> float | None:
    return None if v == "forever" else float(v)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="spike-lane", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status", help="show who holds the lane")
    r = sub.add_parser("run", help="run CMD holding the lane")
    r.add_argument("--wait", type=_wait_arg, default=0.0,
                   help="seconds to wait for a busy lane, or 'forever' (default: refuse at once)")
    r.add_argument("--evict-ollama", action="store_true",
                   help="unload resident Ollama models before running")
    r.add_argument("name")
    r.add_argument("command", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)

    if a.cmd == "status":
        h = holder()
        print("free" if h is None else f"held by {describe(h)}")
        return 0

    command = a.command[1:] if a.command[:1] == ["--"] else a.command
    if not command:
        ap.error("run needs a command after --")
    try:
        fd = acquire(a.name, command, wait=a.wait)
    except LaneBusy as e:
        print(f"✗ lane busy — held by {e}. Retry later, or pass --wait.", file=sys.stderr)
        return EX_TEMPFAIL
    if a.evict_ollama:
        evict_ollama()
    # exec, not spawn: the child inherits the locked fd and keeps the same
    # pid, so the lane lives exactly as long as the workload does.
    os.set_inheritable(fd, True)
    os.environ["SPIKE_LANE_HOLDER"] = a.name
    try:
        os.execvp(command[0], command)
    except OSError as e:
        print(f"✗ cannot run {command[0]}: {e}", file=sys.stderr)
        return 127
