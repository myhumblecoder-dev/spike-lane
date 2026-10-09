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
        f.write(json.dumps({"mflux_version": "fake", "prompt": prompt}).encode())
elif me == "python" and args and args[0].endswith("colormatch.py"):
    ff("-i", args[-2], "-c", "copy", args[-1])
elif me == "python" and args and args[0].endswith("wan22_i2v.py"):
    import hashlib  # like the real model, a different prompt gives a different picture
    c = hashlib.sha256(flag("--prompt").encode()).hexdigest()[:6]
    ff("-f", "lavfi", "-i", f"color=c=0x{c}:size=832x448:rate=24", "-frames:v", "121", "-pix_fmt", "yuv420p", flag("--output-path"))
elif me == "python" and args and args[0] == "demo.py":
    out = Path(flag("--output")); out.mkdir(parents=True, exist_ok=True)
    tone(out / (Path(flag("--video")).stem + ".flac"), float(flag("--duration")) - 0.03)  # MMAudio runs short
elif me == "python" and args and args[0].endswith("score_gen.py"):
    spec = json.loads(Path(args[1]).read_text())
    for s in spec["seeds"]:
        tone(Path(spec["out"]) / f"take-s{s}.wav", spec["duration"])
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
    wav = out / f"{flag('--file_prefix')}_000.wav"
    ff("-f", "lavfi", "-i", f"sine=frequency={200 + len(flag('--text'))}:duration=2", "-ac", "1", "-ar", "24000", str(wav))
    said = Path(os.environ["FAKE_LOG"]).with_name("said"); said.mkdir(exist_ok=True)
    import hashlib  # remember what this audio says, by content, for the stand-in speech-to-text
    (said / hashlib.sha256(wav.read_bytes()).hexdigest()).write_text(flag("--text"))
elif me == "mlx_audio.stt.generate":
    import hashlib
    said = Path(os.environ["FAKE_LOG"]).with_name("said") / hashlib.sha256(Path(flag("--audio")).read_bytes()).hexdigest()
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
             "VOICE_HOME": tmp_path / "voice", "ACESTEP_HOME": tmp_path / "ACE-Step"}
    targets = [bins / "mflux-generate-z-image-turbo"]
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
    e.update(FAKE_LOG=str(tmp_path / "log.jsonl"), IMAGE_MODEL=str(tmp_path / "zi"), WAN5B_MODEL=str(tmp_path / "wan5b"),
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
