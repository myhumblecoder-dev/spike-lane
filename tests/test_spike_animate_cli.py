"""spike-animate end to end with stand-in engines. Each stand-in logs how it was called (and which
lane holder it ran under) and writes small but real media with ffmpeg, so the light steps
(colour-matching aside) — concat, padding, mixing, length checks — run for real."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import spike_animate as sa  # noqa: E402

ANIMATE = ROOT / "bin" / "spike-animate"
FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
pytestmark = pytest.mark.skipif(not Path(FFMPEG).exists(), reason="needs ffmpeg")

# One stand-in for every engine; it decides what to fake from its own name and arguments.
FAKE = r'''#!/usr/bin/env python3
import json, os, subprocess, sys
from pathlib import Path
me, args = Path(sys.argv[0]).name, sys.argv[1:]
with open(os.environ["FAKE_LOG"], "a") as f:
    f.write(json.dumps({"tool": me, "args": args, "holder": os.environ.get("SPIKE_LANE_HOLDER")}) + "\n")
def flag(n, d=None):
    return args[args.index(n) + 1] if n in args else d
def ff(*a):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *a], check=True)
def tone(path, secs):
    ff("-f", "lavfi", "-i", f"sine=frequency=440:duration={secs}", "-ac", "1", "-ar", "24000", str(path))
if me.startswith("mflux"):
    out = flag("--output")
    if Path(out).exists():   # like the real mflux: never overwrite, save beside it as NAME_1.png
        out = str(Path(out).with_name(Path(out).stem + "_1" + Path(out).suffix))
    w, h = flag("--width", "64"), flag("--height", "64")
    ff("-f", "lavfi", "-i", f"color=c=white:s={w}x{h}", "-frames:v", "1", out)
    prompt = "" if os.environ.get("FAKE_LOSE_PROMPT") and os.environ["FAKE_LOSE_PROMPT"] in out else flag("--prompt")
    with open(out, "ab") as f:
        f.write(json.dumps({"mflux_version": "fake", "prompt": prompt, "seed": flag("--seed")}).encode())   # like mflux
elif me == "python" and args and args[0].endswith("colormatch.py"):
    ff("-i", args[-2], "-c", "copy", args[-1])
elif me == "python" and args and args[0].endswith("wan22_i2v.py") and os.environ.get("FAKE_OOM_ONCE") \
        and not Path(os.environ["FAKE_OOM_ONCE"]).exists():
    Path(os.environ["FAKE_OOM_ONCE"]).write_text("x")      # the first run runs out of memory, like a busy Mac
    print("RuntimeError: MPS backend out of memory (MPS allocated: 8.61 GiB, other allocations: 39.00 GiB)")
    sys.exit(1)
elif me == "python" and args and args[0].endswith("wan22_i2v.py"):
    import hashlib  # like the real model, a different prompt or seed gives a different picture
    c = hashlib.sha256((flag("--prompt") + flag("--seed", "")).encode()).hexdigest()[:6]
    y, u, v = 16 + int(c[:2], 16) // 2, int(c[2:4], 16), int(c[4:6], 16)
    box = f"geq=lum='if(between(X\\,T*120\\,T*120+60)*between(Y\\,40\\,100)\\,235\\,{y})':cb={u}:cr={v}"
    ff("-f", "lavfi", "-i", f"color=c=black:size={flag('--width', '832')}x{flag('--height', '448')}:rate=24", "-vf", box,
       "-frames:v", flag("--num-frames", "121"), "-pix_fmt", "yuv420p", flag("--output-path"))
elif me == "python" and args and args[0] == "demo.py":
    out = Path(flag("--output")); out.mkdir(parents=True, exist_ok=True)
    tone(out / (Path(flag("--video")).stem + ".flac"), float(flag("--duration")) - 0.03)  # MMAudio runs short
elif me == "python" and args and args[0].endswith("score_gen.py"):
    spec = json.loads(Path(args[1]).read_text())
    for s in spec["seeds"]:
        tone(Path(spec["out"]) / f"take-s{s}.wav", spec["duration"])
elif me == "python" and args and args[0].endswith("titles.py"):
    spec = json.loads(Path(args[1]).read_text())   # a white bar where the words would be
    w, h = spec["size"]
    for it in spec["items"]:
        a = f"if(between(X\\,{w // 6}\\,{w * 5 // 6})*between(Y\\,{int(h * 0.82)}\\,{int(h * 0.9)})\\,255\\,0)"
        ff("-f", "lavfi", "-i", f"color=c=white:s={w}x{h},format=rgba", "-vf", f"geq=r=255:g=255:b=255:a='{a}'",
           "-frames:v", "1", it["out"])
elif me == "python" and args and args[0].endswith("upscale.py"):
    ff("-i", args[2], "-vf", f"scale={args[4]}:{args[5]}:force_original_aspect_ratio=increase,crop={args[4]}:{args[5]}",
       "-pix_fmt", "yuv420p", args[3])
elif me == "python" and args and args[0].endswith("screen.py"):
    spec = json.loads(Path(args[1]).read_text())   # white where the screen's corners say
    w, h = spec["size"]
    for it in spec["items"]:
        xs, ys = [int(c[0]) for c in it["corners"]], [int(c[1]) for c in it["corners"]]
        a = f"if(between(X\\,{min(xs)}\\,{max(xs)})*between(Y\\,{min(ys)}\\,{max(ys)})\\,255\\,0)"
        ff("-f", "lavfi", "-i", f"color=c=white:s={w}x{h},format=rgba", "-vf", f"geq=r=255:g=255:b=255:a='{a}'",
           "-frames:v", "1", it["out"])
elif me == "python" and args and args[0].endswith("kenburns.py"):
    still, out, w, h, n = args[1], args[2], args[3], args[4], args[5]   # the picture (and its screen), held
    src = ["-f", "lavfi", "-i", f"color=c=gray:s={w}x{h}:rate=24"]
    over = ["-i", args[7], "-filter_complex", f"[1]scale={w}:{h}[o];[0][o]overlay"] if len(args) > 7 else []
    ff(*src, *over, "-frames:v", n, "-pix_fmt", "yuv420p", out)
elif me == "python" and args and args[0].endswith("score_rank.py"):
    takes = args[3:]
    Path(args[2]).write_text(json.dumps([{"take": t, "score": 1.0 - i / 10} for i, t in enumerate(takes)]))
elif me == "mlx_audio.tts.generate" and "MOSS-TTSD" in flag("--model"):
    import struct, wave  # a scene: one tone per [S..] line, with pauses between them
    out = Path(flag("--output_path")); out.mkdir(parents=True, exist_ok=True)
    n = flag("--text").count("[S")
    with wave.open(str(out / f"{flag('--file_prefix')}_000.wav"), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
        import math
        for i in range(n):
            w.writeframes(b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * (300 + 50 * i) * t / 24000)))
                                   for t in range(int(24000 * 1.2))))
            if i < n - 1:
                w.writeframes(b"\x00\x00" * int(24000 * 0.6))
elif me == "mlx_audio.tts.generate":
    out = Path(flag("--output_path")); out.mkdir(parents=True, exist_ok=True)
    wav = out / f"{flag('--file_prefix')}_000.wav"   # like the real model: half a second of silence before the words
    ff("-f", "lavfi", "-i", f"sine=frequency={200 + len(flag('--text'))}:duration=2", "-af", "adelay=500:all=1,apad=pad_dur=0.3",
       "-ac", "1", "-ar", "24000", str(wav))
    by_name = Path(os.environ["FAKE_LOG"]).with_name("said-by-name"); by_name.mkdir(exist_ok=True)
    (by_name / flag("--file_prefix")).write_text(flag("--text"))   # lines are trimmed after, so their bytes change
    said = Path(os.environ["FAKE_LOG"]).with_name("said"); said.mkdir(exist_ok=True)
    import hashlib  # remember what this audio says, by content, for the stand-in speech-to-text
    (said / hashlib.sha256(wav.read_bytes()).hexdigest()).write_text(flag("--text"))
elif me == "mlx_audio.stt.generate":
    import hashlib
    said = Path(os.environ["FAKE_LOG"]).with_name("said") / hashlib.sha256(Path(flag("--audio")).read_bytes()).hexdigest()
    if not said.exists():
        said = Path(os.environ["FAKE_LOG"]).with_name("said-by-name") / Path(flag("--audio")).stem
    Path(flag("--output-path") + ".txt").write_text(said.read_text() if said.exists() else "")
'''

STORY = """
title = "Fox Hunt"
seed = 1024
look = "2D cartoon animation"
still_style = "cartoon still"

[characters]
FOX = "a slender red fox"

[narrator]
voice = "a deep warm storyteller"
sample = "Deep in the winter woods."

[music]
caption = "dark chamber orchestra"
takes = 2

[[shot]]
id = "01-walk"
still = "FOX walking in snow"
motion = "the fox walks"
sfx = "paws in snow"
narration = "The fox was hungry."
intensity = 0.2

[[shot]]
id = "02-pounce"
still = "FOX pouncing"
motion = "the fox pounces"
anchor = 0
intensity = 1.0

[[shot]]
id = "03-stalk"
still = "FOX creeping"
motion = "the fox creeps"
intensity = 0.4

[shot.reference]
clip = "reference/stalk.mp4"
start = 0.5
"""


@pytest.fixture
def env(tmp_path, monkeypatch):
    bins = tmp_path / "fakebin"
    homes = {"FASTVIDEO_HOME": tmp_path / "FastVideo", "MMAUDIO_HOME": tmp_path / "MMAudio",
             "VOICE_HOME": tmp_path / "voice", "ACESTEP_HOME": tmp_path / "ACE-Step", "UPSCALE_HOME": tmp_path / "upscale"}
    targets = [bins / "mflux-generate-z-image-turbo", bins / "mflux-generate-flux2-edit"]
    targets += [h / ".venv" / "bin" / "python" for h in homes.values()]
    targets += [homes["VOICE_HOME"] / ".venv" / "bin" / n for n in ("mlx_audio.tts.generate", "mlx_audio.stt.generate")]
    for t in targets:
        t.parent.mkdir(parents=True, exist_ok=True)
        t.write_text(FAKE)
        t.chmod(0o755)
    film = tmp_path / "fox-hunt"
    (film / "reference").mkdir(parents=True)
    subprocess.run([FFMPEG, "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30", "-t", "6",
                    "-pix_fmt", "yuv420p", str(film / "reference" / "stalk.mp4")], check=True)
    (film / "story.toml").write_text(STORY)
    monkeypatch.setenv("SPIKE_LANE_DIR", str(tmp_path / "lane"))
    e = os.environ.copy()
    e.update({k: str(v) for k, v in homes.items()})
    e.update(OLLAMA_HOST="http://127.0.0.1:9",   # nothing listens: the continuity check is skipped, never the real model
             FAKE_LOG=str(tmp_path / "log.jsonl"), IMAGE_MODEL=str(tmp_path / "zi"), WAN5B_MODEL=str(tmp_path / "wan5b"),
             PATH=f"{bins}:{Path(FFMPEG).parent}:/usr/bin:/bin")  # no ollama: eviction no-ops
    return film, e, tmp_path / "log.jsonl"


def run(env, *argv):
    film, e, _ = env
    return subprocess.run([str(ANIMATE), *argv], capture_output=True, text=True, env=e, cwd=film)


def calls(log, tool=None):
    if not log.exists():
        return []
    rows = [json.loads(l) for l in log.read_text().splitlines()]
    return [r for r in rows if tool is None or r["tool"] == tool]


def flag(args, name):
    return args[args.index(name) + 1]


def made(args):
    """The picture an image call ends up as: it renders to .NAME-new.png, then moves it to NAME.png."""
    out = Path(flag(args, "--output"))
    return out.with_name(out.name.removeprefix(".").replace("-new.", "."))


def duration(path):
    out = subprocess.run([str(Path(FFMPEG).with_name("ffprobe")), "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True).stdout
    return float(out)


# --- board ------------------------------------------------------------------

def test_board_renders_every_still_in_the_lane_then_stops_for_review(env):
    film, _, log = env
    r = run(env, "board", "story.toml")
    assert r.returncode == 0, r.stderr
    stills = calls(log, "mflux-generate-z-image-turbo")
    assert [made(c["args"]).name for c in stills] == ["01-walk.png", "02-pounce.png", "03-stalk.png"]
    first = stills[0]["args"]
    assert flag(first, "--prompt") == "a slender red fox walking in snow, cartoon still"
    assert flag(first, "--seed") == "1024" and flag(first, "--width") == "1248" and flag(first, "--height") == "720"
    assert all(c["holder"] == "image" for c in stills)
    assert (film / "board" / "contact-sheet.jpg").exists()
    assert "review" in r.stdout.lower()              # tells the user to look before running film
    assert not calls(log, "python")                  # no video yet


def test_board_rerenders_only_what_changed(env):
    film, _, log = env
    assert run(env, "board", "story.toml").returncode == 0
    log.unlink()
    assert run(env, "board", "story.toml").returncode == 0
    assert calls(log) == []                          # nothing changed, nothing rendered
    story = film / "story.toml"
    story.write_text(story.read_text().replace('still = "FOX pouncing"', 'still = "FOX pouncing high"'))
    assert run(env, "board", "story.toml").returncode == 0
    assert [made(c["args"]).name for c in calls(log, "mflux-generate-z-image-turbo")] == ["02-pounce.png"]


def test_a_redrawn_still_replaces_the_old_picture(env):
    film, _, log = env
    assert run(env, "board", "story.toml").returncode == 0
    story = film / "story.toml"
    story.write_text(story.read_text().replace('still = "FOX pouncing"', 'still = "FOX leaping high"'))
    r = run(env, "board", "story.toml")
    assert r.returncode == 0, r.stdout + r.stderr
    import spike_animate as sa
    assert sa.png_prompt(film / "board" / "02-pounce.png") == "FOX leaping high, cartoon still".replace("FOX", "a slender red fox")
    assert not list((film / "board").glob("*_1.png"))


def test_board_flags_a_still_whose_prompt_was_lost(env):
    film, e, _ = env
    e["FAKE_LOSE_PROMPT"] = "02-pounce"
    r = run(env, "board", "story.toml")
    assert r.returncode == 1
    assert "02-pounce" in r.stderr and "prompt" in r.stderr.lower()


def test_retake_renders_alternates_beside_the_board(env):
    film, _, log = env
    r = run(env, "retake", "story.toml", "02-pounce", "--seeds", "7,42")
    assert r.returncode == 0, r.stderr
    outs = [made(c["args"]) for c in calls(log, "mflux-generate-z-image-turbo")]
    assert [o.name for o in outs] == ["02-pounce-s7.png", "02-pounce-s42.png"]
    assert all(o.parent == film / "board" / "retakes" for o in outs)
    assert (film / "board" / "retakes" / "02-pounce-sheet.jpg").exists()
    assert "seed" in r.stdout                        # explains how to keep one


# --- film -------------------------------------------------------------------

def test_film_needs_the_board_first(env):
    r = run(env, "film", "story.toml")
    assert r.returncode == 2
    assert "board" in r.stderr


def test_film_animates_scores_and_mixes_the_whole_story(env):
    film, _, log = env
    assert run(env, "board", "story.toml").returncode == 0
    log.unlink()
    r = run(env, "film", "story.toml")
    assert r.returncode == 0, r.stdout + r.stderr

    video = [c for c in calls(log, "python") if c["args"][0].endswith("wan22_i2v.py")]
    assert len(video) == 3 and all(c["holder"] == "video" for c in video)
    walk, pounce, stalk = (c["args"] for c in video)
    assert flag(walk, "--image") == str(film / "board" / "01-walk.png") and flag(walk, "--anchor-start") == "1"
    assert flag(walk, "--prompt") == "2D cartoon animation, the fox walks"
    assert flag(pounce, "--anchor-start") == "0"
    # Real-clip shot: motion from the clip, look from a cartoonized first frame.
    assert flag(stalk, "--init-video") == str(film / "reference" / "stalk.mp4")
    assert flag(stalk, "--init-start") == "0.5" and flag(stalk, "--init-sigma") == "0.88"
    first = [c for c in calls(log, "mflux-generate-z-image-turbo") if "--image-path" in c["args"]]
    assert len(first) == 1 and flag(first[0]["args"], "--image-strength") == "0.4"
    assert flag(stalk, "--image") == str(made(first[0]["args"]))

    sfx = [c for c in calls(log, "python") if c["args"][0] == "demo.py"]
    assert len(sfx) == 3 and all(c["holder"] == "sfx" for c in sfx)
    assert flag(sfx[0]["args"], "--prompt") == "paws in snow"
    tts = calls(log, "mlx_audio.tts.generate")
    assert [flag(c["args"], "--text") for c in tts] == ["Deep in the winter woods.", "The fox was hungry."]
    assert "VoiceDesign" in flag(tts[0]["args"], "--model") and "Base" in flag(tts[1]["args"], "--model")
    assert all(c["holder"] == "voice" for c in tts)
    gen = [c for c in calls(log, "python") if c["args"][0].endswith("score_gen.py")]
    assert len(gen) == 1 and gen[0]["holder"] == "music"
    spec = json.loads(Path(gen[0]["args"][1]).read_text())
    assert spec["bpm"] == 95 and spec["seeds"] == [1, 2]

    out = film / "film" / "fox-hunt.mp4"
    assert out.exists()
    assert duration(out) == pytest.approx(3 * 121 / 24, abs=0.06)
    assert (film / "film" / "review" / "index.html").exists()
    checks = json.loads((film / "film" / "checks.json").read_text())
    assert checks["narration"]["01-walk"] >= 0.8
    assert checks["score"]["chosen"].endswith("take-s1.wav")


def test_second_film_run_renders_nothing_and_one_edit_reruns_one_shot(env):
    film, _, log = env
    assert run(env, "board", "story.toml").returncode == 0
    assert run(env, "film", "story.toml").returncode == 0
    log.unlink()
    assert run(env, "film", "story.toml").returncode == 0
    assert [c["tool"] for c in calls(log)] in ([], ["mlx_audio.stt.generate"] * len(calls(log)))
    story = film / "story.toml"
    story.write_text(story.read_text().replace('motion = "the fox pounces"', 'motion = "the fox leaps"'))
    log.unlink(missing_ok=True)
    assert run(env, "film", "story.toml").returncode == 0
    video = [c for c in calls(log, "python") if c["args"][0].endswith("wan22_i2v.py")]
    assert [Path(flag(c["args"], "--output-path")).name for c in video] == ["02-pounce-raw.mp4"]
    sfx = [c for c in calls(log, "python") if c["args"][0] == "demo.py"]
    assert len(sfx) == 1                               # only the changed shot is re-scored for effects
    assert not calls(log, "mlx_audio.tts.generate")   # narration untouched


def test_status_reports_progress(env):
    assert run(env, "board", "story.toml").returncode == 0
    r = run(env, "status", "story.toml")
    assert r.returncode == 0
    assert "stills 3/3" in r.stdout and "shots 0/3" in r.stdout


# --- characters who talk ------------------------------------------------------

TALK_STORY = """
title = "Robots"
look = "2D cartoon animation"
still_style = "cartoon still"

[characters.TINY]
look = "a tiny rusty robot"
voice = "small chirpy robot voice"

[characters.NEWBIE]
look = "a big clumsy robot"
voice = "big deep gentle voice"
sample = "Everything is so big."

[[shot]]
id = "01-hello"
still = "TINY waving"
motion = "static camera, a small robot talking"
dialogue = [{ speaker = "TINY", line = "Hello? Can you hear me?", emotion = "hopeful" }]

[[shot]]
id = "02-reply"
still = "NEWBIE blinking"
motion = "static camera, a big robot talking"
dialogue = [{ speaker = "NEWBIE", line = "I can hear you." }, { speaker = "NEWBIE", line = "Where am I?" }]
"""


def talking(env, mode=None):
    film = env[0]
    (film / "story.toml").write_text(TALK_STORY + (f'\n[dialogue]\nmode = "{mode}"\n' if mode else ""))
    return env


def test_characters_get_their_own_voice_and_every_line_is_cloned_from_it(env):
    film, _, log = talking(env)
    assert run(env, "board", "story.toml").returncode == 0
    r = run(env, "film", "story.toml")
    assert r.returncode == 0, r.stdout + r.stderr
    tts = calls(log, "mlx_audio.tts.generate")
    design = [c["args"] for c in tts if "VoiceDesign" in flag(c["args"], "--model")]
    clone = [c["args"] for c in tts if "Base" in flag(c["args"], "--model")]
    assert sorted(flag(a, "--instruct") for a in design) == ["big deep gentle voice", "small chirpy robot voice"]
    assert {flag(a, "--text") for a in design} == {sa_default_sample(), "Everything is so big."}
    assert [flag(a, "--text") for a in clone] == ["Hello? Can you hear me?", "I can hear you.", "Where am I?"]
    assert Path(flag(clone[0], "--ref_audio")).name == "TINY.wav" and Path(flag(clone[2], "--ref_audio")).name == "NEWBIE.wav"
    assert flag(clone[2], "--ref_text") == "Everything is so big."
    assert all(c["holder"] == "voice" for c in tts)
    checks = json.loads((film / "film" / "checks.json").read_text())
    assert set(checks["dialogue"]) == {"01-hello-1", "02-reply-1", "02-reply-2"}
    assert min(checks["dialogue"].values()) >= 0.8
    assert duration(film / "film" / "robots.mp4") == pytest.approx(2 * 121 / 24, abs=0.06)
    assert "Where am I?" in (film / "film" / "review" / "index.html").read_text()
    log.unlink()
    assert run(env, "film", "story.toml").returncode == 0
    assert not calls(log, "mlx_audio.tts.generate")          # nothing changed, no voices re-rendered


def test_scene_mode_performs_all_the_dialogue_in_one_pass_and_cuts_it_into_lines(env):
    film, _, log = talking(env, "scene")
    assert run(env, "board", "story.toml").returncode == 0
    r = run(env, "film", "story.toml")
    assert r.returncode == 0, r.stdout + r.stderr
    tts = calls(log, "mlx_audio.tts.generate")
    scene = [c["args"] for c in tts if "MOSS-TTSD" in flag(c["args"], "--model")]
    assert len(scene) == 1 and not [c for c in tts if "Base" in flag(c["args"], "--model")]
    a = scene[0]
    assert flag(a, "--text") == "[S1] Hello? Can you hear me? [S2] I can hear you. [S2] Where am I?"
    refs = [Path(a[i + 1]).name for i, x in enumerate(a) if x == "--ref_audio"]
    assert refs == ["TINY.wav", "NEWBIE.wav"]                 # S1, S2 in order of first appearance
    lines = sorted(p.name for p in (film / "voice" / "dialogue").glob("0*.wav"))
    assert lines == ["01-hello-1-TINY.wav", "02-reply-1-NEWBIE.wav", "02-reply-2-NEWBIE.wav"]
    assert (film / "film" / "robots.mp4").exists()


def sa_default_sample():
    import spike_animate
    return spike_animate.DEFAULT_VOICE_SAMPLE


# --- one step at a time (the studio confirms each before the next) -------------

def _tools(log):
    out = []
    for c in calls(log):
        a = c["args"]
        out.append(c["tool"] if c["tool"] != "python" else Path(a[0]).name)
    return set(out)


def test_video_needs_the_board_first(env):
    r = run(env, "video", "story.toml")
    assert r.returncode == 2 and "board" in r.stderr


def test_video_animates_the_shots_and_nothing_else(env):
    film, _, log = env
    assert run(env, "board", "story.toml").returncode == 0
    log.unlink()
    r = run(env, "video", "story.toml")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "wan22_i2v.py" in _tools(log)
    assert not _tools(log) & {"demo.py", "mlx_audio.tts.generate", "score_gen.py"}
    assert duration(film / "film" / "picture.mp4") == pytest.approx(3 * 121 / 24, abs=0.06)


def test_sound_needs_the_video_first(env):
    assert run(env, "board", "story.toml").returncode == 0
    r = run(env, "sound", "story.toml")
    assert r.returncode == 2 and "video" in r.stderr


def test_sound_adds_effects_only(env):
    film, _, log = env
    assert run(env, "board", "story.toml").returncode == 0
    assert run(env, "video", "story.toml").returncode == 0
    log.unlink()
    r = run(env, "sound", "story.toml")
    assert r.returncode == 0, r.stdout + r.stderr
    assert _tools(log) == {"demo.py"}
    assert (film / "film" / "sfx.wav").exists()


def test_voices_record_every_line_and_save_their_checks(env):
    film, _, log = env
    r = run(env, "voices", "story.toml")
    assert r.returncode == 0, r.stdout + r.stderr
    assert _tools(log) <= {"mlx_audio.tts.generate", "mlx_audio.stt.generate"}
    assert (film / "voice" / "01-walk.wav").exists()
    checks = json.loads((film / "film" / "checks.json").read_text())
    assert checks["narration"]["01-walk"] >= 0.8


def test_music_scores_the_film_length_and_saves_the_choice(env):
    film, _, log = env
    r = run(env, "music", "story.toml")
    assert r.returncode == 0, r.stdout + r.stderr
    assert _tools(log) == {"score_gen.py", "score_rank.py"}
    gen = [c for c in calls(log, "python") if c["args"][0].endswith("score_gen.py")]
    spec = json.loads(Path(gen[0]["args"][1]).read_text())
    assert spec["duration"] == round(3 * 121 / 24 + 1.2, 2)
    checks = json.loads((film / "film" / "checks.json").read_text())
    assert checks["score"]["chosen"].endswith("take-s1.wav")


def test_after_every_step_the_film_only_mixes(env):
    film, _, log = env
    for step in ("board", "video", "sound", "voices", "music"):
        assert run(env, step, "story.toml").returncode == 0, step
    log.unlink()
    r = run(env, "film", "story.toml")
    assert r.returncode == 0, r.stdout + r.stderr
    assert not _tools(log) & {"wan22_i2v.py", "demo.py", "mlx_audio.tts.generate", "score_gen.py",
                              "mflux-generate-z-image-turbo"}
    assert (film / "film" / "fox-hunt.mp4").exists()


def loudness(path):
    err = subprocess.run([FFMPEG, "-hide_banner", "-i", str(path), "-map", "0:a", "-af", "ebur128", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    return float(err.rsplit("I:", 1)[1].split("LUFS")[0])


def test_the_mix_turns_each_layer_down_to_its_level(env):
    film, _, _ = env
    story = film / "story.toml"
    full = story.read_text().replace("[[shot]]", "[mix]\nsound = 100\nmusic = 100\nvoices = 100\n\n[[shot]]", 1)
    story.write_text(full)
    assert run(env, "board", "story.toml").returncode == 0
    r = run(env, "film", "story.toml")
    assert r.returncode == 0, r.stdout + r.stderr
    loud = loudness(film / "film" / "fox-hunt.mp4")
    assert loud == pytest.approx(-16, abs=1.5)              # 100% everywhere = the old normalized mix

    story.write_text(STORY)                                  # defaults: sound 20, music 20, voices 50
    r = run(env, "film", "story.toml")
    assert r.returncode == 0, r.stdout + r.stderr
    assert loudness(film / "film" / "fox-hunt.mp4") < loud - 6


# --- one scene at a time -------------------------------------------------------

def test_board_can_draw_one_scene(env):
    film, _, log = env
    r = run(env, "board", "story.toml", "--shot", "02-pounce")
    assert r.returncode == 0, r.stdout + r.stderr
    drawn = [made(c["args"]) for c in calls(log, "mflux-generate-z-image-turbo")]
    assert drawn == [film / "board" / "02-pounce.png"]


def test_video_sound_and_voices_can_make_one_scene(env):
    film, _, log = env
    assert run(env, "board", "story.toml").returncode == 0
    log.unlink()
    r = run(env, "video", "story.toml", "--shot", "01-walk")
    assert r.returncode == 0, r.stdout + r.stderr
    assert _tools(log) == {"wan22_i2v.py", "colormatch.py"}             # animate, then match the still's colours
    assert len([c for c in calls(log, "python") if c["args"][0].endswith("wan22_i2v.py")]) == 1
    assert (film / "shots" / "01-walk.mp4").exists() and not (film / "shots" / "02-pounce.mp4").exists()
    assert not (film / "film" / "picture.mp4").exists()            # the whole picture waits for the end

    log.unlink()
    r = run(env, "sound", "story.toml", "--shot", "01-walk")
    assert r.returncode == 0, r.stdout + r.stderr
    assert [flag(c["args"], "--prompt") for c in calls(log, "python")] == ["paws in snow"]

    log.unlink()
    r = run(env, "voices", "story.toml", "--shot", "01-walk")
    assert r.returncode == 0, r.stdout + r.stderr
    said = [flag(c["args"], "--text") for c in calls(log, "mlx_audio.tts.generate")]
    assert said == ["Deep in the winter woods.", "The fox was hungry."]   # narrator's voice, then the line
    checks = json.loads((film / "film" / "checks.json").read_text())
    assert set(checks["narration"]) == {"01-walk"}


def test_one_scenes_voices_keep_the_other_scenes_checks(env):
    film, _, _ = env
    talk = talking(env)
    assert run(env, "voices", "story.toml").returncode == 0
    before = json.loads((film / "film" / "checks.json").read_text())
    assert set(before["dialogue"]) == {"01-hello-1", "02-reply-1", "02-reply-2"}
    r = run(talk, "voices", "story.toml", "--shot", "01-hello")
    assert r.returncode == 0, r.stdout + r.stderr
    after = json.loads((film / "film" / "checks.json").read_text())
    assert after["dialogue"] == before["dialogue"]


def test_scene_preview_is_one_shot_with_its_own_sound(env):
    film, _, log = env
    assert run(env, "scene", "story.toml", "01-walk").returncode == 2          # needs its video first
    assert run(env, "board", "story.toml", "--shot", "01-walk").returncode == 0
    for step in ("video", "sound", "voices"):
        assert run(env, step, "story.toml", "--shot", "01-walk").returncode == 0, step
    log.unlink()
    r = run(env, "scene", "story.toml", "01-walk")
    assert r.returncode == 0, r.stdout + r.stderr
    assert not _tools(log) & {"wan22_i2v.py", "demo.py", "mlx_audio.tts.generate", "score_gen.py"}
    out = film / "scenes" / "01-walk.mp4"
    assert duration(out) == pytest.approx(121 / 24, abs=0.06)
    assert loudness(out) > -40                                                  # it has sound


def test_unknown_scene_is_refused(env):
    r = run(env, "video", "story.toml", "--shot", "99-nope")
    assert r.returncode == 2 and "no shot '99-nope'" in r.stderr


# --- trimming a shot, and new takes of its motion ------------------------------------------

def frames(path):
    out = subprocess.run([str(Path(FFMPEG).with_name("ffprobe")), "-v", "error", "-count_frames", "-select_streams", "v",
                          "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return int(out)


def distinct_frames(path, first, last):
    """How many different pictures frames first..last of a clip show."""
    out = subprocess.run([FFMPEG, "-v", "error", "-i", str(path), "-vf", f"select=between(n\\,{first}\\,{last})",
                          "-fps_mode", "passthrough", "-f", "framemd5", "-"], capture_output=True, text=True, check=True).stdout
    return len({l.rsplit(",", 1)[1].strip() for l in out.splitlines() if l and not l.startswith("#")})


def set_shot(film, sid, extra):
    """Add lines to one [[shot]] table of the story."""
    text = (film / "story.toml").read_text()
    head = f'id = "{sid}"\n'
    (film / "story.toml").write_text(text.replace(head, head + extra + "\n", 1))


def wan(log):
    return [c for c in calls(log, "python") if c["args"][0].endswith("wan22_i2v.py")]


def test_a_shot_can_end_early_and_still_fill_its_five_seconds(env):
    film, _, log = env
    assert run(env, "board", "story.toml", "--shot", "01-walk").returncode == 0
    assert run(env, "video", "story.toml", "--shot", "01-walk").returncode == 0
    log.unlink()
    for fill in ("slow", "hold"):
        (film / "story.toml").write_text(STORY)
        set_shot(film, "01-walk", f'end = 3.0\nfill = "{fill}"')
        r = run(env, "video", "story.toml", "--shot", "01-walk")
        assert r.returncode == 0, r.stdout + r.stderr
        clip = film / "shots" / "01-walk.mp4"
        assert frames(clip) == 121, fill                         # still exactly one shot long
        tail = distinct_frames(clip, 80, 120)
        assert tail == 1 if fill == "hold" else tail > 20, (fill, tail)   # frozen, or still moving (slowed down)
    assert not wan(log)                                           # trimming never animates again


def test_a_new_motion_seed_makes_a_new_take_and_keeps_the_old_one(env):
    film, _, log = env
    assert run(env, "board", "story.toml", "--shot", "01-walk").returncode == 0
    assert run(env, "video", "story.toml", "--shot", "01-walk").returncode == 0
    first = (film / "shots" / "01-walk.mp4").read_bytes()
    log.unlink()
    set_shot(film, "01-walk", "motion_seed = 99")
    r = run(env, "video", "story.toml", "--shot", "01-walk")
    assert r.returncode == 0, r.stdout + r.stderr
    assert [flag(c["args"], "--seed") for c in wan(log)] == ["99"]
    assert (film / "shots" / "01-walk.mp4").read_bytes() != first
    takes = sorted(p.name for p in (film / "shots" / "takes").glob("01-walk-m*.mp4"))
    assert takes == ["01-walk-m1024.mp4", "01-walk-m99.mp4"]


def test_shots_made_before_takes_existed_are_not_animated_again(env):
    film, _, log = env
    assert run(env, "board", "story.toml", "--shot", "01-walk").returncode == 0
    assert run(env, "video", "story.toml", "--shot", "01-walk").returncode == 0
    # as an older spike-animate left it: only the shot itself, keyed by what it was made from
    shutil.rmtree(film / "shots" / "takes")
    (film / "shots" / "01-walk-full.mp4").unlink()
    keys = json.loads((film / ".spike-animate" / "keys.json").read_text())
    keys = {k: v for k, v in keys.items() if "-full" not in k and "takes/" not in k}
    import spike_animate as sa
    story = sa.load_story(film / "story.toml")
    keys["shots/01-walk.mp4"] = sa.shot_key(story, story.shots[0])
    (film / ".spike-animate" / "keys.json").write_text(json.dumps(keys))
    log.unlink()
    assert run(env, "video", "story.toml", "--shot", "01-walk").returncode == 0
    assert not wan(log)


def test_a_picture_is_still_drawn_when_the_continuity_check_cannot_run(env):
    film, _, _ = env
    text = (film / "story.toml").read_text().replace('title = ', 'lighting = "winter dusk, cold blue light"\ntitle = ', 1)
    (film / "story.toml").write_text(text)
    r = run(env, "board", "story.toml", "--shot", "01-walk")
    assert r.returncode == 0, r.stdout + r.stderr
    assert (film / "board" / "01-walk.png").exists()
    assert "continuity check skipped" in r.stdout


# --- the Look: character sheets and place pictures, then scenes drawn from them --------------

def with_place(film):
    text = (film / "story.toml").read_text().replace(
        '[narrator]', '[places]\nWOODS = "a snowy pine forest at dusk"\n\n[narrator]', 1)
    (film / "story.toml").write_text(text.replace('still = "FOX walking in snow"', 'still = "FOX walking through the WOODS"'))


def tool(c):
    return c["tool"]


def test_look_draws_a_character_sheet_and_a_place_picture(env):
    film, _, log = env
    with_place(film)
    for name in ("FOX", "WOODS"):
        r = run(env, "look", "story.toml", name)
        assert r.returncode == 0, r.stdout + r.stderr
        assert (film / "looks" / f"{name}.png").exists()
    fox, woods = calls(log, "mflux-generate-z-image-turbo")
    assert "character sheet" in flag(fox["args"], "--prompt") and "a slender red fox" in flag(fox["args"], "--prompt")
    assert "a snowy pine forest at dusk" in flag(woods["args"], "--prompt") and "no people" in flag(woods["args"], "--prompt")
    assert fox["holder"] == "image"
    log.unlink()
    assert run(env, "look", "story.toml", "FOX").returncode == 0
    assert not log.exists()                                          # nothing changed: nothing drawn


def test_a_character_sheet_is_drawn_from_an_uploaded_picture(env):
    film, _, log = env
    (film / "reference" / "FOX.png").write_bytes(b"\x89PNG the author's fox")
    r = run(env, "look", "story.toml", "FOX")
    assert r.returncode == 0, r.stdout + r.stderr
    [c] = calls(log)
    assert tool(c) == "mflux-generate-flux2-edit"
    assert flag(c["args"], "--image-paths") == str(film / "reference" / "FOX.png")
    assert "a slender red fox" in flag(c["args"], "--prompt")


def test_look_can_draw_more_versions(env):
    film, _, _ = env
    r = run(env, "look", "story.toml", "FOX", "--seeds", "7,9")
    assert r.returncode == 0, r.stdout + r.stderr
    assert sorted(p.name for p in (film / "looks" / "takes").glob("FOX-s*.png")) == ["FOX-s7.png", "FOX-s9.png"]


def test_scene_pictures_are_drawn_from_the_looks(env):
    film, _, log = env
    with_place(film)
    assert run(env, "look", "story.toml", "FOX").returncode == 0
    assert run(env, "look", "story.toml", "WOODS").returncode == 0
    log.unlink()
    r = run(env, "board", "story.toml", "--shot", "01-walk")
    assert r.returncode == 0, r.stdout + r.stderr
    [c] = [c for c in calls(log) if tool(c).startswith("mflux")]
    assert tool(c) == "mflux-generate-flux2-edit"
    assert c["args"][c["args"].index("--image-paths") + 1:c["args"].index("--image-paths") + 3] == \
        [str(film / "looks" / "FOX.png"), str(film / "looks" / "WOODS.png")]
    prompt = flag(c["args"], "--prompt")
    assert "image 1" in prompt and "image 2" in prompt and "a slender red fox walking through" in prompt
    log.unlink()
    assert run(env, "board", "story.toml", "--shot", "01-walk").returncode == 0
    assert not [c for c in calls(log) if tool(c).startswith("mflux")]        # up to date
    assert run(env, "look", "story.toml", "FOX", "--seeds", "5").returncode == 0
    text = (film / "story.toml").read_text().replace("[places]", "[looks]\nFOX = 5\n\n[places]", 1)
    (film / "story.toml").write_text(text)                                     # use another version of the fox
    assert run(env, "look", "story.toml", "FOX").returncode == 0
    log.unlink()
    assert run(env, "board", "story.toml", "--shot", "01-walk").returncode == 0
    assert [tool(c) for c in calls(log) if tool(c).startswith("mflux")] == ["mflux-generate-flux2-edit"]   # redrawn


# --- reference pictures for one scene, and voices chosen up front -----------------------------

def test_a_scene_can_be_drawn_with_its_own_reference_picture(env):
    film, _, log = env
    (film / "reference" / "01-walk-1.png").write_bytes(b"\x89PNG a screenshot")
    text = (film / "story.toml").read_text().replace(
        'id = "01-walk"\n', 'id = "01-walk"\npictures = [{ file = "reference/01-walk-1.png", note = "what the sign in the snow shows" }]\n', 1)
    (film / "story.toml").write_text(text)
    r = run(env, "board", "story.toml", "--shot", "01-walk")
    assert r.returncode == 0, r.stdout + r.stderr
    [c] = [c for c in calls(log) if c["tool"].startswith("mflux")]
    assert c["tool"] == "mflux-generate-flux2-edit"
    assert flag(c["args"], "--image-paths") == str(film / "reference" / "01-walk-1.png")
    assert "image 1 is what the sign in the snow shows" in flag(c["args"], "--prompt")


def test_a_missing_scene_picture_is_named(env):
    film, _, _ = env
    text = (film / "story.toml").read_text().replace(
        'id = "01-walk"\n', 'id = "01-walk"\npictures = [{ file = "reference/gone.png", note = "x" }]\n', 1)
    (film / "story.toml").write_text(text)
    r = run(env, "board", "story.toml", "--shot", "01-walk")
    assert r.returncode != 0 and "reference/gone.png" in r.stderr


def test_voices_are_designed_and_chosen_before_any_line(env):
    film, _, log = env
    r = run(env, "voice", "story.toml", "narrator")
    assert r.returncode == 0, r.stdout + r.stderr
    [c] = calls(log, "mlx_audio.tts.generate")
    assert "VoiceDesign" in flag(c["args"], "--model") and flag(c["args"], "--instruct") == "a deep warm storyteller"
    assert (film / "voice" / "card.wav").exists() and (film / "voice" / "takes" / "narrator-1.wav").exists()
    log.unlink()
    assert run(env, "voice", "story.toml", "narrator", "--takes", "2").returncode == 0
    assert len(calls(log, "mlx_audio.tts.generate")) == 2
    assert sorted(p.name for p in (film / "voice" / "takes").glob("narrator-*.wav")) == \
        ["narrator-1.wav", "narrator-2.wav", "narrator-3.wav"]
    log.unlink()
    assert run(env, "voices", "story.toml", "--shot", "01-walk").returncode == 0
    said = [flag(c["args"], "--text") for c in calls(log, "mlx_audio.tts.generate")]
    assert said == ["The fox was hungry."]                      # the chosen voice is reused, not designed again
    assert run(env, "voice", "story.toml", "NOBODY").returncode == 2


def test_a_voice_made_before_takes_existed_becomes_take_one(env):
    film, _, _ = env
    assert run(env, "voices", "story.toml", "--shot", "01-walk").returncode == 0   # designs the narrator on the way
    shutil.rmtree(film / "voice" / "takes", ignore_errors=True)
    card = (film / "voice" / "card.wav").read_bytes()
    assert run(env, "voice", "story.toml", "narrator", "--takes", "1").returncode == 0
    assert (film / "voice" / "takes" / "narrator-1.wav").read_bytes() == card
    assert (film / "voice" / "takes" / "narrator-2.wav").exists()


def test_a_step_that_runs_out_of_memory_is_tried_once_more(env):
    film, e, log = env
    assert run(env, "board", "story.toml", "--shot", "01-walk").returncode == 0
    log.unlink()
    e.update(FAKE_OOM_ONCE=str(film.parent / "oom-once"), SPIKE_OOM_PAUSE="0")
    r = run(env, "video", "story.toml", "--shot", "01-walk")
    assert r.returncode == 0, r.stdout + r.stderr
    assert len(wan(log)) == 2 and "out of memory" in r.stdout


# --- voice pace: every shot as long as its lines ----------------------------------------------

def voice_paced(env):
    film, _, _ = env
    (film / "story.toml").write_text(STORY.replace('seed = 1024\n', 'seed = 1024\npace = "voice"\n', 1))
    return film


def test_a_voice_paced_shot_waits_for_its_line_then_lasts_as_long_as_it(env):
    film, _, log = env
    voice_paced(env)
    assert run(env, "board", "story.toml", "--shot", "01-walk").returncode == 0
    r = run(env, "video", "story.toml", "--shot", "01-walk")
    assert r.returncode == 2 and "record" in r.stderr and not wan(log)       # the line sets the length
    assert run(env, "voices", "story.toml", "--shot", "01-walk").returncode == 0
    r = run(env, "video", "story.toml", "--shot", "01-walk")
    assert r.returncode == 0, r.stdout + r.stderr
    line = sa.media_seconds(film / "voice" / "01-walk.wav")
    assert line < 2.4                                                          # 2 s of words, cut to them
    want = sa.wan_frames(0.5 + line + sa.TAIL)                                 # lead, the line, a breath
    assert flag(wan(log)[-1]["args"], "--num-frames") == str(want)
    assert frames(film / "shots" / "01-walk.mp4") == want


def test_a_voice_paced_film_is_as_long_as_its_shots(env):
    film, _, log = env
    voice_paced(env)
    set_shot(film, "02-pounce", "seconds = 4.2")
    assert run(env, "board", "story.toml").returncode == 0
    assert run(env, "voices", "story.toml").returncode == 0
    r = run(env, "film", "story.toml")
    assert r.returncode == 0, r.stdout + r.stderr
    line = sa.media_seconds(film / "voice" / "01-walk.wav")
    lens = [sa.wan_frames(0.5 + line + sa.TAIL) / 24, sa.wan_frames(4.2) / 24, sa.wan_frames(sa.SILENT_SECONDS) / 24]
    assert duration(film / "film" / "picture.mp4") == pytest.approx(sum(lens), abs=0.08)
    assert duration(film / "film" / "sfx.wav") == pytest.approx(sum(lens), abs=0.08)
    sfx = [float(flag(c["args"], "--duration")) for c in calls(log, "python") if c["args"][0] == "demo.py"]
    assert sfx == pytest.approx(lens, abs=0.01)                               # effects as long as each shot
    spec = json.loads((film / "score" / "spec.json").read_text())
    assert spec["duration"] == pytest.approx(sum(lens) + 1.2, abs=0.08)
    assert duration(film / "film" / "fox-hunt.mp4") == pytest.approx(sum(lens), abs=0.1)


def test_a_voice_paced_scene_preview_is_its_shot_long(env):
    film, _, _ = env
    voice_paced(env)
    assert run(env, "board", "story.toml", "--shot", "01-walk").returncode == 0
    for step in ("voices", "video", "sound"):
        assert run(env, step, "story.toml", "--shot", "01-walk").returncode == 0, step
    assert run(env, "scene", "story.toml", "01-walk").returncode == 0
    assert duration(film / "scenes" / "01-walk.mp4") == pytest.approx(sa.wan_frames(0.5 + sa.media_seconds(film / "voice" / "01-walk.wav") + sa.TAIL) / 24, abs=0.06)


# --- titles: captions, the end card, and the finished size ------------------------------------

END_CARD = '\n[end_card]\ntext = "Put your agent to work."\nsub = "BotGuild.ai"\n'


def size(path):
    out = subprocess.run([str(Path(FFMPEG).with_name("ffprobe")), "-v", "error", "-select_streams", "v",
                          "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout.strip()
    return tuple(int(x) for x in out.split(","))


def bar(path, t):
    """How bright the caption band of the frame at `t` is (0-255)."""
    out = subprocess.run([FFMPEG, "-v", "error", "-ss", str(t), "-i", str(path), "-frames:v", "1", "-vf",
                          "crop=iw*2/3:ih/14:iw/6:ih*0.83,scale=1:1,format=gray", "-f", "rawvideo", "-"],
                         capture_output=True, check=True).stdout
    return out[0]


def test_the_film_is_finished_at_720p_with_captions_and_an_end_card(env):
    film, _, log = env
    (film / "story.toml").write_text(STORY.replace("\n[[shot]]", END_CARD + "\n[[shot]]", 1))
    assert run(env, "board", "story.toml").returncode == 0
    r = run(env, "film", "story.toml")
    assert r.returncode == 0, r.stdout + r.stderr
    spec = json.loads((film / "film" / "titles" / "spec.json").read_text())
    assert spec["size"] == [1280, 720]
    assert [i["text"] for i in spec["items"]] == ["The fox was hungry.", "Put your agent to work.\nBotGuild.ai"]
    out = film / "film" / "fox-hunt.mp4"
    L = 121 / 24
    assert size(out) == (1280, 720)
    assert duration(out) == pytest.approx(3 * L + 1.5, abs=0.1)              # the last frame holds for the card
    assert bar(out, 1.0) > 220 and bar(out, L + 1.0) < 160                    # a caption over shot 1 only
                                                                              # (the stand-in video is at most 143)
    assert bar(out, 3 * L + 1.0) > 220                                        # the end card, over the held frame


def test_without_captions_the_film_is_still_finished_at_720p(env):
    film, _, log = env
    (film / "story.toml").write_text(STORY.replace("seed = 1024\n", "seed = 1024\ncaptions = false\n", 1))
    assert run(env, "board", "story.toml").returncode == 0
    r = run(env, "film", "story.toml")
    assert r.returncode == 0, r.stdout + r.stderr
    assert not [c for c in calls(log, "python") if c["args"][0].endswith("titles.py")]
    out = film / "film" / "fox-hunt.mp4"
    assert size(out) == (1280, 720) and bar(out, 1.0) < 160
    assert duration(out) == pytest.approx(3 * 121 / 24, abs=0.1)


@pytest.mark.skipif(not (Path.home() / "MMAudio" / ".venv" / "bin" / "python").exists(), reason="needs Pillow's venv")
def test_the_title_engine_draws_readable_words(tmp_path):
    spec = {"size": [1280, 720], "font": sa.TITLE_FONT, "items": [
        {"text": "Built the agent.", "kind": "caption", "out": str(tmp_path / "c.png")},
        {"text": "Put your agent to work.\nBotGuild.ai", "kind": "end", "out": str(tmp_path / "e.png")}]}
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    subprocess.run([str(Path.home() / "MMAudio" / ".venv" / "bin" / "python"), str(ROOT / "engines" / "titles.py"),
                    str(tmp_path / "spec.json")], check=True)
    for name, rows in (("c.png", (0.75, 0.95)), ("e.png", (0.35, 0.65))):
        alpha = subprocess.run([FFMPEG, "-v", "error", "-i", str(tmp_path / name), "-vf", "alphaextract,format=gray",
                                "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
        ink = [i // 1280 / 720 for i, a in enumerate(alpha) if a > 128]
        assert ink and rows[0] <= min(ink) and max(ink) <= rows[1], name     # words, where they belong


def rgb(path, t):
    out = subprocess.run([FFMPEG, "-v", "error", "-ss", str(t), "-i", str(path), "-frames:v", "1", "-vf",
                          "scale=1:1,format=rgb24", "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    return tuple(out[:3])


def test_a_warmer_grade_refinishes_the_film_without_animating_again(env):
    film, _, log = env
    assert run(env, "board", "story.toml").returncode == 0
    assert run(env, "film", "story.toml").returncode == 0
    before = rgb(film / "film" / "fox-hunt.mp4", 1.0)
    log.unlink()
    (film / "story.toml").write_text(STORY + "\n[grade]\nwarmth = 0.3\n")
    r = run(env, "film", "story.toml")
    assert r.returncode == 0, r.stdout + r.stderr
    assert not wan(log)
    after = rgb(film / "film" / "fox-hunt.mp4", 1.0)
    assert after[0] - after[2] > before[0] - before[2] + 10                    # redder, less blue


# --- screens ------------------------------------------------------------------------------------

def patch(path, t, box):
    """How bright a box (x, y, w, h in film pixels) of the frame at `t` is."""
    x, y, w, h = box
    out = subprocess.run([FFMPEG, "-v", "error", "-ss", str(t), "-i", str(path), "-frames:v", "1", "-vf",
                          f"crop={w}:{h}:{x}:{y},scale=1:1,format=gray", "-f", "rawvideo", "-"],
                         capture_output=True, check=True).stdout
    return out[0]


def test_a_real_picture_is_laid_onto_the_screen_in_its_shot(env):
    film, _, log = env
    subprocess.run([FFMPEG, "-v", "error", "-f", "lavfi", "-i", "color=c=gray:s=640x400", "-frames:v", "1",
                    str(film / "reference" / "page.png")], check=True)
    corners = [[0.3, 0.2], [0.7, 0.2], [0.7, 0.6], [0.3, 0.6]]
    (film / "story.toml").write_text(STORY.replace("seed = 1024\n", "seed = 1024\ncaptions = false\n", 1))
    set_shot(film, "02-pounce", f'screen = {{ picture = "reference/page.png", corners = {corners} }}')
    assert run(env, "board", "story.toml").returncode == 0
    r = run(env, "film", "story.toml")
    assert r.returncode == 0, r.stdout + r.stderr
    spec = json.loads((film / "film" / "screens" / "spec.json").read_text())
    want = [sa.still_to_finish(tuple(c), sa.STILL_SIZE, (832, 448), sa.FINISH_SIZE) for c in corners]
    assert spec["items"][0]["corners"] == [pytest.approx(list(c)) for c in want]
    assert spec["items"][0]["top"] == 0.0
    out, L = film / "film" / "fox-hunt.mp4", 121 / 24
    (x0, y0), (x1, y1) = want[0], want[2]
    box = (int(x0) + 10, int(y0) + 10, int(x1 - x0) - 20, int(y1 - y0) - 20)
    assert patch(out, L + 1.0, box) > 220                       # the picture, over shot 2
    assert patch(out, 1.0, box) < 160 and patch(out, 2 * L + 1.0, box) < 160   # and only there


# --- quality ------------------------------------------------------------------------------------

def upscales(log):
    return [c for c in calls(log, "python") if c["args"][0].endswith("upscale.py")]


def test_an_hd_film_animates_at_the_models_full_size_and_needs_no_upscaling(env):
    film, _, log = env
    (film / "story.toml").write_text(STORY.replace("seed = 1024\n", 'seed = 1024\nquality = "hd"\n', 1))
    assert run(env, "board", "story.toml").returncode == 0
    r = run(env, "film", "story.toml")
    assert r.returncode == 0, r.stdout + r.stderr
    assert {(flag(c["args"], "--width"), flag(c["args"], "--height")) for c in wan(log)} == {("1280", "704")}
    assert size(film / "shots" / "01-walk.mp4") == (1280, 704) and not upscales(log)
    assert size(film / "film" / "fox-hunt.mp4") == (1280, 720)


def test_a_small_film_has_each_clip_upscaled_once_before_it_is_finished(env):
    film, _, log = env
    assert run(env, "board", "story.toml").returncode == 0
    assert run(env, "film", "story.toml").returncode == 0
    ups = upscales(log)
    assert sorted(Path(c["args"][2]).name for c in ups) == ["01-walk.mp4", "02-pounce.mp4", "03-stalk.mp4"]
    assert all(c["args"][1].endswith("realesr-general-x4v3.pth") and c["args"][4:6] == ["1280", "720"] for c in ups)
    assert size(film / "film" / "picture.mp4") == (1280, 720)
    log.unlink()
    (film / "story.toml").write_text(STORY.replace("intensity = 0.2", "intensity = 0.3", 1))   # nothing visual
    assert run(env, "film", "story.toml").returncode == 0
    assert not upscales(log)


# --- motion levels ----------------------------------------------------------------------------

def glides(log):
    return [c for c in calls(log, "python") if c["args"][0].endswith("kenburns.py")]


def frames(path):
    out = subprocess.run([str(Path(FFMPEG).with_name("ffprobe")), "-v", "error", "-count_frames", "-select_streams", "v",
                          "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return int(out.strip())


def test_a_still_shot_glides_over_its_picture_without_the_video_model(env):
    film, _, log = env
    set_shot(film, "02-pounce", 'tier = "still"')
    assert run(env, "board", "story.toml").returncode == 0
    r = run(env, "film", "story.toml")
    assert r.returncode == 0, r.stdout + r.stderr
    assert len(wan(log)) == 2 and not any("pounce" in flag(c["args"], "--output-path") for c in wan(log))
    [g] = glides(log)
    assert g["args"][1] == str(film / "board" / "02-pounce.png") and g["args"][3:7] == ["1280", "720", "121", "push"]
    clip = film / "shots" / "02-pounce.mp4"
    assert size(clip) == (1280, 720) and frames(clip) == 121
    assert sorted(Path(c["args"][2]).name for c in upscales(log)) == ["01-walk.mp4", "03-stalk.mp4"]   # already big


def test_a_new_camera_move_glides_again_and_animates_nothing(env):
    film, _, log = env
    set_shot(film, "02-pounce", 'tier = "still"')
    assert run(env, "board", "story.toml").returncode == 0
    assert run(env, "film", "story.toml").returncode == 0
    log.unlink()
    set_shot(film, "02-pounce", 'camera = "pull"')
    assert run(env, "film", "story.toml").returncode == 0
    assert not wan(log) and [c["args"][6] for c in glides(log)] == ["pull"]


def test_a_still_shots_screen_is_laid_on_its_picture_so_it_moves_with_the_glide(env):
    film, _, log = env
    subprocess.run([FFMPEG, "-v", "error", "-f", "lavfi", "-i", "color=c=gray:s=640x400", "-frames:v", "1",
                    str(film / "reference" / "page.png")], check=True)
    corners = [[0.3, 0.2], [0.7, 0.2], [0.7, 0.6], [0.3, 0.6]]
    set_shot(film, "02-pounce", f'tier = "still"\nscreen = {{ picture = "reference/page.png", corners = {corners} }}')
    assert run(env, "board", "story.toml").returncode == 0
    r = run(env, "film", "story.toml")
    assert r.returncode == 0, r.stdout + r.stderr
    [g] = glides(log)
    assert g["args"][7] == str(film / "shots" / "screens" / "02-pounce.png")
    spec = json.loads((film / "shots" / "screens" / "02-pounce.json").read_text())
    w, h = size(film / "board" / "02-pounce.png")
    assert spec["size"] == [w, h] and spec["items"][0]["corners"] == [pytest.approx([x * w, y * h]) for x, y in corners]
    assert not (film / "film" / "screens").exists()             # not laid on again when the film is finished


def gray_frame(path, index, w, h):
    raw = subprocess.run([FFMPEG, "-v", "error", "-i", str(path), "-vf", f"select=eq(n\\,{index}),format=gray",
                          "-frames:v", "1", "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    return [raw[r * w:(r + 1) * w] for r in range(h)]


def lit(rows):
    """The span of bright columns along the middle row, and of bright rows down the middle column."""
    mid = rows[len(rows) // 2]
    xs = [x for x, v in enumerate(mid) if v > 128]
    ys = [y for y, r in enumerate(rows) if r[len(mid) // 2] > 128]
    return xs[0], xs[-1], ys[0], ys[-1]


@pytest.mark.parametrize("camera", ["push", "pull", "left"])
def test_the_glide_engine_moves_the_camera_smoothly_over_the_picture(tmp_path, camera):
    still, out = tmp_path / "still.png", tmp_path / "out.mp4"
    subprocess.run([FFMPEG, "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=1248x720", "-vf",
                    "drawbox=x=524:y=260:w=200:h=200:c=white:t=fill", "-frames:v", "1", str(still)], check=True)
    subprocess.run([str(Path.home() / "FastVideo" / ".venv" / "bin" / "python"), str(ROOT / "engines" / "kenburns.py"),
                    str(still), str(out), "1280", "720", "49", camera], check=True)
    assert size(out) == (1280, 720) and frames(out) == 49
    a, b = lit(gray_frame(out, 0, 1280, 720)), lit(gray_frame(out, 48, 1280, 720))
    wa, wb = a[1] - a[0], b[1] - b[0]
    if camera == "push":
        assert wb / wa == pytest.approx(1 + sa.GLIDE, abs=0.02)      # closer in by the end
    elif camera == "pull":
        assert wa / wb == pytest.approx(1 + sa.GLIDE, abs=0.02)
    else:
        assert wa == pytest.approx(wb, abs=2) and b[0] > a[0] + 20     # camera pans left: the scene slides right
    assert abs((a[2] + a[3]) - (b[2] + b[3])) <= 4 if camera == "left" else True


def test_the_end_card_is_drawn_where_the_story_places_it(env):
    film, _, log = env
    (film / "story.toml").write_text(STORY.replace("\n[[shot]]", END_CARD + 'place = "top"\n' + "\n[[shot]]", 1))
    assert run(env, "board", "story.toml").returncode == 0
    r = run(env, "film", "story.toml")
    assert r.returncode == 0, r.stdout + r.stderr
    spec = json.loads((film / "film" / "titles" / "spec.json").read_text())
    assert [(i["kind"], i["place"]) for i in spec["items"]] == [("caption", "middle"), ("end", "top")]


def test_the_title_engine_puts_a_high_end_card_in_the_top_third(tmp_path):
    spec = {"size": [1280, 720], "font": sa.TITLE_FONT, "items": [
        {"text": "Put your agent to work.\nBotGuild.ai", "kind": "end", "place": "top", "out": str(tmp_path / "e.png")}]}
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    subprocess.run([str(Path.home() / "MMAudio" / ".venv" / "bin" / "python"), str(ROOT / "engines" / "titles.py"),
                    str(tmp_path / "spec.json")], check=True)
    alpha = subprocess.run([FFMPEG, "-v", "error", "-i", str(tmp_path / "e.png"), "-vf", "alphaextract,format=gray",
                            "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    ink = [i // 1280 / 720 for i, a in enumerate(alpha) if a > 128]
    assert ink and 0.05 <= min(ink) and max(ink) <= 0.36



# --- pace: lines trimmed to their words ------------------------------------------------------------

def silence_at_start(path):
    r = subprocess.run([FFMPEG, "-i", str(path), "-af", "silencedetect=n=-40dB:d=0.05", "-f", "null", "-"],
                       capture_output=True, text=True).stderr
    import re
    starts = [float(x) for x in re.findall(r"silence_start: (-?[\d.]+)", r)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", r)]
    return ends[0] if starts and starts[0] < 0.01 else 0.0


def test_a_recorded_line_is_trimmed_to_its_words_and_long_pauses_shortened(tmp_path):
    wav = tmp_path / "line.wav"
    subprocess.run([FFMPEG, "-v", "error", "-f", "lavfi", "-i", "sine=frequency=300:duration=1", "-f", "lavfi", "-i",
                    "sine=frequency=500:duration=1", "-filter_complex",
                    "[0]adelay=500:all=1,apad=pad_dur=0.6[a];[1]apad=pad_dur=0.5[b];[a][b]concat=n=2:v=0:a=1",
                    "-ar", "24000", "-ac", "1", str(wav)], check=True)
    assert sa.media_seconds(wav) == pytest.approx(3.6, abs=0.05)
    sa.trim_speech(wav)
    assert silence_at_start(wav) < 0.1                                # the words start at once
    assert 2.2 <= sa.media_seconds(wav) <= 2.6                        # 1 s + a short pause + 1 s, little after


def test_narration_is_recorded_without_the_voice_models_lead_in_silence(env):
    film, _, log = env
    assert run(env, "voices", "story.toml").returncode == 0
    wav = film / "voice" / "01-walk.wav"
    assert silence_at_start(wav) < 0.1 and sa.media_seconds(wav) < 2.4


def test_every_animated_scene_gets_a_motion_check_that_never_stops_the_film(env):
    film, _, log = env
    set_shot(film, "02-pounce", 'tier = "still"')
    assert run(env, "board", "story.toml").returncode == 0
    r = run(env, "video", "story.toml", "--shot", "01-walk")
    assert r.returncode == 0 and "motion check skipped for 01-walk" in r.stdout     # no model here: advice only
    r = run(env, "film", "story.toml")
    assert r.returncode == 0, r.stdout + r.stderr
    checked = {l.split("motion check skipped for ")[1].split(":")[0] for l in r.stdout.splitlines() if "motion check skipped" in l}
    assert checked == {"01-walk", "03-stalk"}                                    # a glide has no motion to check


def test_lines_recorded_by_the_film_step_start_where_their_voice_paced_shots_start(env, monkeypatch):
    film, e, _ = env
    voice_paced(env)
    set_shot(film, "03-stalk", 'narration = "And then it crept closer."')
    for k, v in e.items():
        monkeypatch.setenv(k, v)
    story = sa.load_story(film / "story.toml")
    lines = sa.narrate(story, sa.Keys(film), {})     # nothing recorded yet, as when `film` runs straight from the board
    assert [round(at, 3) for _, at in lines] == [round(at, 3) for _, at in sa.narration_offsets(story)]
    assert lines[-1][1] < sa.film_length(story)       # 5 s guesses put the last line past the end of a short film


def test_the_film_names_a_line_missing_from_the_final_mix(env):
    film, _, log = env
    set_shot(film, "02-pounce", 'narration = "It leapt into the snow."')
    assert run(env, "board", "story.toml").returncode == 0
    heard = log.with_name("said-by-name"); heard.mkdir(exist_ok=True)
    (heard / "fox-hunt").write_text("The fox was hungry.")   # the finished film: the second line got lost
    r = run(env, "film", "story.toml")
    assert r.returncode == 0, r.stdout + r.stderr
    checks = json.loads((film / "film" / "checks.json").read_text())
    assert checks["final_lines"]["01-walk"] >= 0.9 and checks["final_lines"]["02-pounce"] < 0.6
    assert "02-pounce's line is missing or buried in the final mix" in r.stdout
    assert "01-walk's line" not in r.stdout
