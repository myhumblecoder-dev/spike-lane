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
| Image | `spike-image` | same as video |
| Animated film | `spike-animate` (one lane call per step) | waits for the lane |
| Anything new | `spike-lane run NAME -- CMD` | `--wait SECONDS` or `--wait forever` |

## Getting Started
```bash
ln -sf ~/spike-lane/bin/spike-lane  ~/.local/bin/spike-lane
ln -sf ~/spike-lane/bin/spike-video ~/.local/bin/spike-video

spike-lane status                                   # free | held by <name> ...
spike-video "a red fox trotting through fresh snow" # 14B, 480p, ~10 min
spike-video "..." --fast                            # ~3.5 min, RIFE interpolation
SPIKE_LANE_WAIT=forever spike-video "..."           # queue behind a grind
spike-image "a puffin on a sea cliff at dawn"       # Z-Image Turbo q8, 1024², 9 steps
python3 -m pytest -q tests
```
Video needs `~/FastVideo` (with its `.venv`) and `~/wan-models/FastMetal-14B-QAD`.
Clips land in `~/Movies/spike-video/`.
Image needs `mflux` (`uv tool install mflux`) and `~/image-models/z-image-turbo-mflux-q8`; images land in `~/Pictures/spike-image/`.

## spike-animate: storyboard → film

Make a short cartoon from one `story.toml`: characters, look, and per shot a still, a motion,
an optional real wildlife clip, sound, narration and story intensity; plus a narrator voice and a
music brief. Example: `~/Movies/spike-video/fox-hunt-v2/story.toml`.

```bash
ln -sf ~/spike-lane/bin/spike-animate ~/.local/bin/spike-animate
cd ~/Movies/spike-video/my-film
spike-animate new my-film "a one-line idea"       # local gemma4 drafts my-film/story.toml (~1 min)
spike-animate board story.toml                    # stills + board/contact-sheet.jpg, then stop
spike-animate retake story.toml 06-pounce --seeds 7,42,99   # alternates; keep one with `seed = N`
spike-animate film story.toml                     # → film/<title>.mp4 + film/review/ + film/checks.json
spike-animate status story.toml
```

| Step | Engine | Per 8-shot film | Peak memory |
|---|---|---|---|
| Stills | Z-Image Turbo q8 (mflux) | ~8 min | 11.6 GB |
| Shots | Wan 2.2 5B FastMetal (`engines/wan22_i2v.py`): starts from the still; or restyles a real clip with a cartoon first frame | ~17 min | 8.3 GB |
| Effects | MMAudio large_44k_v2, generated from each shot's picture | ~4 min | 9.4 GB |
| Narration | Qwen3-TTS 1.7B: VoiceDesign once, Base clones it per line; Parakeet checks every line | ~1 min | small |
| Score | ACE-Step 1.5 turbo (`engines/score_gen.py`), ranked by `engines/score_rank.py` | ~2 min | 15.7 GB |

Rules it enforces: every shot is 121 frames at 24 fps (5.04 s); the score runs at 95 BPM so each
shot is two bars; stills whose stored prompt doesn't match the story are rejected; only outputs whose
inputs changed are re-rendered (`.spike-animate/keys.json`).

Needs `~/FastVideo` + `~/wan-models/FastMetal-5B-QAD`, `~/MMAudio` (own venv), `~/voice` (venv with
`mlx-audio`), `~/ACE-Step-1.5` (official repo, `uv sync`), each overridable by `FASTVIDEO_HOME`,
`WAN5B_MODEL`, `MMAUDIO_HOME`, `VOICE_HOME`, `ACESTEP_HOME`. MMAudio's weights are CC-BY-NC (not for
commercial use); the other models are Apache-2.0 / MIT.

### Characters who talk
Give a character a table with a `voice` (how it sounds), and put lines on shots:

```toml
[characters.TINY]
look  = "a tiny rusty robot with one glowing blue eye"
voice = "small, bright, chirpy robot voice, fast and excitable"

[[shot]]
id = "04-hello"
dialogue = [{ speaker = "TINY", line = "Hello? Can you hear me?", emotion = "hopeful" }]

[dialogue]
mode = "clone"   # default: one designed voice per character, every line cloned from it (fast, steady)
                 # "scene": MOSS-TTSD performs all the dialogue in one pass (natural turn-taking,
                 #          minutes per scene, ~23-31 GB: runs alone in the lane)
```
A shot may have narration then dialogue; keep all its speech to 12 words so it fits 5 s. Mouths are
not lip-synced: stage talking shots medium/over-the-shoulder and cut to listeners' reactions.
Every line is checked with speech-to-text (`film/checks.json` → `dialogue`).
