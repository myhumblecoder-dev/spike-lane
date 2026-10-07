# spike-lane

## Problem
This Mac Studio has 36 GB of unified memory. Each of Spike's heavy workloads,
a coder grind (Ollama, ~17 GB of weights) or Wan video generation (~22 GB
peak for the 14B model), needs most of it on its own. Two at once doesn't
error out. The machine swaps and everything slows down.

## Solution
One machine-wide lane: an exclusive `flock` on `~/.local/state/spike-lane/lane.lock`.
Whoever holds it is the only heavy workload running. The kernel releases it
when the holder exits or crashes, so it can't get stuck. The protocol is
documented in `spike_lane.py` and is small enough to reimplement:
OpenClaw's `dag-coder` does exactly that, so neither side imports the other.

| Workload | Takes the lane via | When busy |
|---|---|---|
| Coder grind | `dag-coder run` (OpenClaw) | waits 30 min, then reports `lane-busy` |
| Video | `spike-video` | refuses right away (exit 75) unless `SPIKE_LANE_WAIT` is set |
| Anything new | `spike-lane run NAME -- CMD` | `--wait SECONDS` or `--wait forever` |

## Getting Started
```bash
ln -sf ~/spike-lane/bin/spike-lane  ~/.local/bin/spike-lane
ln -sf ~/spike-lane/bin/spike-video ~/.local/bin/spike-video

spike-lane status                                   # free | held by <name> ...
spike-video "a red fox trotting through fresh snow" # 14B, 480p, ~10 min
spike-video "..." --fast                            # ~3.5 min, RIFE interpolation
SPIKE_LANE_WAIT=forever spike-video "..."           # queue behind a grind
python3 -m pytest -q tests
```
Video needs `~/FastVideo` (with its `.venv`) and `~/wan-models/FastMetal-14B-QAD`.
Clips land in `~/Movies/spike-video/`.
