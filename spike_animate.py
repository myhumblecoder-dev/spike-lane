"""spike-animate — storyboard, animate, score and mix a short cartoon from one story file.

A film is a directory holding `story.toml`. Every stage writes into that directory and
records a key (a hash of everything that went into an output) in `.spike-animate/keys.json`,
so re-running a command only re-renders what changed. Heavy steps go through spike-lane.

The rules here were proven on the fox-hunt film (2026-10):
  * every shot is 121 frames at 24 fps (5.04 s); effects, narration timing and the score's
    tempo all rely on that, so it is fixed, not configurable;
  * stills: Z-Image Turbo, characters pasted verbatim from the story + a fixed seed;
  * video: Wan 2.2 5B started from the still (latent frame 0 pinned, light anchoring),
    colour-matched back to the still; or restyled from a real clip with a cartoon first frame;
  * score tempo = 8 beats per shot, so every cut falls on a bar line.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

SHOT_FRAMES = 121
FPS = 24
STILL_SIZE = (1248, 720)        # storyboard stills (Z-Image)
FIRST_FRAME_SIZE = (1248, 672)  # cartoon first frame for real-clip shots: the video's 832x448 aspect


class StoryError(ValueError):
    """The story file is missing something or contradicts itself."""


@dataclass
class Reference:
    clip: Path
    start: float = 0.0
    sigma: float = 0.88          # noise added to the real clip: lower keeps more real motion
    first_strength: float = 0.4  # how closely the cartoon first frame follows the real one


@dataclass
class Shot:
    id: str
    still: str
    motion: str
    seed: int
    anchor: int = 1
    sfx: str = ""
    narration: str = ""
    intensity: float = 0.5
    colormatch: bool = True
    reference: Reference | None = None
    still_style: str = ""        # overrides the film's still style for this shot
    dialogue: list["Line"] = field(default_factory=list)


@dataclass
class Line:
    speaker: str
    line: str
    emotion: str = ""


DEFAULT_VOICE_SAMPLE = "Hello there! It is a lovely day, and I have so much to tell you about it."
DIALOGUE_MODES = ("clone", "scene")   # clone: one voice card per character, each line cloned from it;
                                      # scene: MOSS-TTSD performs all the dialogue in one pass


@dataclass
class CharVoice:
    voice: str                   # a description of how the character sounds (Qwen3-TTS VoiceDesign)
    sample: str = DEFAULT_VOICE_SAMPLE


@dataclass
class Narrator:
    voice: str
    sample: str
    lead: float = 0.5


@dataclass
class Music:
    caption: str
    structure: str = "[Instrumental]"
    key: str = ""
    takes: int = 8
    take: int | None = None      # pin a seed instead of ranking
    db: float = -10.0


@dataclass
class Story:
    path: Path
    title: str
    seed: int
    look: str
    still_style: str
    characters: dict[str, str]
    shots: list[Shot]
    narrator: Narrator | None = None
    music: Music | None = None
    sfx_negative: str = "music, speech, human voice, talking, singing"
    sfx_db: float = -4.0
    voices: dict[str, CharVoice] = field(default_factory=dict)
    dialogue_mode: str = "clone"

    @property
    def root(self) -> Path:
        return self.path.parent

    @property
    def slug(self) -> str:
        return re.sub(r"[^a-z0-9]+", "-", self.title.lower()).strip("-") or "film"


def _need(table: dict, key: str, where: str):
    if not table.get(key):
        raise StoryError(f"{where} is missing '{key}'")
    return table[key]


def _lines(s: dict, sid: str, characters: dict, voices: dict) -> list[Line]:
    lines = []
    for x in s.get("dialogue", []):
        who = _need(x, "speaker", f"shot {sid} dialogue")
        if who not in characters:
            raise StoryError(f"shot {sid}: speaker '{who}' is not a character")
        if who not in voices:
            raise StoryError(f"shot {sid}: {who} speaks but has no voice; give it a table with look and voice")
        lines.append(Line(who, _need(x, "line", f"shot {sid} dialogue"), x.get("emotion", "")))
    return lines


def load_story(path: Path) -> Story:
    path = Path(path).resolve()
    try:
        d = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as e:
        raise StoryError(f"{path.name}: {e}") from e
    seed = int(d.get("seed", 1024))
    characters, voices = {}, {}
    for name, c in d.get("characters", {}).items():
        if isinstance(c, str):
            characters[name] = c
        else:
            characters[name] = _need(c, "look", f"character {name}")
            if c.get("voice"):
                voices[name] = CharVoice(voice=c["voice"], sample=c.get("sample", DEFAULT_VOICE_SAMPLE))
    raw_shots = d.get("shot", [])
    if not raw_shots:
        raise StoryError("the story has no shots ([[shot]] tables)")
    shots, seen = [], set()
    for i, s in enumerate(raw_shots, 1):
        sid = _need(s, "id", f"shot {i}")
        if sid in seen:
            raise StoryError(f"duplicate shot id '{sid}'")
        seen.add(sid)
        ref = None
        if "reference" in s:
            r = s["reference"]
            ref = Reference(clip=(path.parent / _need(r, "clip", f"shot {sid} reference")).resolve(),
                            start=float(r.get("start", 0.0)), sigma=float(r.get("sigma", 0.88)),
                            first_strength=float(r.get("first_strength", 0.4)))
        shots.append(Shot(id=sid, still=_need(s, "still", f"shot {sid}"), motion=_need(s, "motion", f"shot {sid}"),
                          seed=int(s.get("seed", seed)), anchor=int(s.get("anchor", 1)), sfx=s.get("sfx", ""),
                          narration=s.get("narration", ""), intensity=float(s.get("intensity", 0.5)),
                          colormatch=bool(s.get("colormatch", ref is None)), reference=ref,
                          still_style=s.get("still_style", ""), dialogue=_lines(s, sid, characters, voices)))
    narrator = None
    if "narrator" in d:
        n = d["narrator"]
        narrator = Narrator(voice=_need(n, "voice", "[narrator]"), sample=_need(n, "sample", "[narrator]"),
                            lead=float(n.get("lead", 0.5)))
    if narrator is None and any(s.narration for s in shots):
        raise StoryError("shots have narration but the story has no [narrator] (voice + sample)")
    music = None
    if "music" in d:
        m = d["music"]
        music = Music(caption=_need(m, "caption", "[music]"), structure=m.get("structure", "[Instrumental]"),
                      key=m.get("key", ""), takes=int(m.get("takes", 8)),
                      take=int(m["take"]) if "take" in m else None, db=float(m.get("db", -10.0)))
    mode = d.get("dialogue", {}).get("mode", "clone")
    if mode not in DIALOGUE_MODES:
        raise StoryError(f"[dialogue] mode must be one of {', '.join(DIALOGUE_MODES)}, not '{mode}'")
    sound = d.get("sound", {})
    return Story(path=path, title=_need(d, "title", "the story"), seed=seed, look=_need(d, "look", "the story"),
                 still_style=_need(d, "still_style", "the story"), characters=characters,
                 shots=shots, narrator=narrator, music=music, voices=voices, dialogue_mode=mode,
                 sfx_negative=sound.get("negative", Story.sfx_negative), sfx_db=float(sound.get("db", -4.0)))


# --- prompts ----------------------------------------------------------------

def expand(text: str, characters: dict[str, str]) -> str:
    """Replace each character NAME (whole word) with its fixed description, so every shot
    describes the character with exactly the same words; that is what keeps it consistent."""
    if not characters:
        return text
    pat = re.compile(r"\b(" + "|".join(map(re.escape, characters)) + r")\b")
    return pat.sub(lambda m: characters[m.group(1)], text)


def still_prompt(story: Story, shot: Shot) -> str:
    return f"{expand(shot.still, story.characters)}, {shot.still_style or story.still_style}"


def motion_prompt(story: Story, shot: Shot) -> str:
    return f"{story.look}, {expand(shot.motion, story.characters)}"


# --- timing -----------------------------------------------------------------

def shot_seconds() -> float:
    return SHOT_FRAMES / FPS


def score_bpm(shot_len: float, beats: int = 8) -> int:
    return round(beats * 60 / shot_len)


def narration_offsets(story: Story) -> list[tuple[str, float]]:
    lead = story.narrator.lead if story.narrator else 0.0
    return [(s.id, i * shot_seconds() + lead) for i, s in enumerate(story.shots) if s.narration]


def intensity_arc(story: Story) -> list[float]:
    return [s.intensity for s in story.shots]


def hit_time(story: Story) -> float:
    arc = intensity_arc(story)
    return arc.index(max(arc)) * shot_seconds()


def dialogue_lines(story: Story) -> list[tuple[str, Line]]:
    return [(s.id, l) for s in story.shots for l in s.dialogue]


def spoken_text(story: Story) -> str:
    """Everything said in the film, in order: narration and dialogue."""
    return " ".join(x for s in story.shots for x in ([s.narration] if s.narration else []) + [l.line for l in s.dialogue])


def place_lines(shot_start: float, durations: list[float], lead: float = 0.4, gap: float = 0.3) -> tuple[list[float], float]:
    """Start times for a shot's lines, one after another; and how far the last runs past the cut (0 if not)."""
    starts, t = [], shot_start + lead
    for d in durations:
        starts.append(t)
        t += d + gap
    end = (starts[-1] + durations[-1]) if durations else shot_start
    return starts, max(0.0, end - (shot_start + shot_seconds()))


def dialogue_lead(narration_seconds: float | None, narrator_lead: float, gap: float = 0.3,
                  default: float = 0.4) -> float:
    """Seconds into a shot when its dialogue starts: after the narrator's line, if the shot has one."""
    return default if narration_seconds is None else narrator_lead + narration_seconds + gap


def split_at_silences(silences: list[tuple[float, float]], total: float, n: int) -> list[tuple[float, float]]:
    """Cut a one-pass scene recording into its `n` lines at the n-1 longest pauses."""
    if len(silences) < n - 1:
        raise StageError(f"the scene recording should hold {n} lines but has only {len(silences)} pause(s)")
    longest = sorted(sorted(silences, key=lambda x: x[1] - x[0], reverse=True)[:n - 1])
    cuts = [0.0] + [round((a + b) / 2, 6) for a, b in longest] + [total]
    return list(zip(cuts[:-1], cuts[1:]))


# --- caching ----------------------------------------------------------------

def key_for(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


class Keys:
    """Which inputs produced each output. An output is stale when it is missing or its key changed."""

    def __init__(self, root: Path):
        self.file = Path(root) / ".spike-animate" / "keys.json"
        self.root = Path(root)
        self.keys = json.loads(self.file.read_text()) if self.file.exists() else {}

    def _rel(self, out: Path) -> str:
        return str(Path(out).resolve().relative_to(self.root.resolve()))

    def stale(self, out: Path, key: str) -> bool:
        return not Path(out).exists() or self.keys.get(self._rel(out)) != key

    def record(self, out: Path, key: str) -> None:
        self.keys[self._rel(out)] = key
        self.file.parent.mkdir(parents=True, exist_ok=True)
        self.file.write_text(json.dumps(self.keys, indent=1, sort_keys=True))


# --- checks -----------------------------------------------------------------

def png_prompt(path: Path) -> str | None:
    """The prompt mflux stored in the image's metadata. A missing or empty prompt means the
    still was generated from nothing (the 'empty forest' bug), not that the model refused."""
    data = Path(path).read_bytes()
    i = data.find(b'{"mflux_version"')
    if i < 0:
        return None
    try:
        meta, _ = json.JSONDecoder().raw_decode(data[i:].decode("latin-1"))
    except ValueError:
        return None
    return meta.get("prompt") or None


def _letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())


def words_match(expected: str, heard: str) -> float:
    """Similarity of what the narrator should say to what speech-to-text heard (0-1). Compared
    letter by letter with spaces removed, so 'a scent' and 'ascent' count as the same."""
    return difflib.SequenceMatcher(None, _letters(expected), _letters(heard)).ratio()


# --- mixing -----------------------------------------------------------------

def mix_filtergraph(narration_ms: list[int], total: float, sfx_db: float, music_db: float) -> str:
    """ffmpeg graph for the final mix. Inputs: 0 picture, 1 sound effects, 2 music,
    3.. one narration line each, delayed to its start. Music and effects duck under the voice."""
    fmt = "aresample=48000,aformat=channel_layouts=stereo"
    g = (f"[1:a]{fmt},volume={sfx_db:g}dB[sfx];"
         f"[2:a]{fmt},volume={music_db:g}dB,apad,atrim=0:{total:g}[mus];"
         "[sfx][mus]amix=inputs=2:normalize=0[bed];")
    master = "loudnorm=I=-16:TP=-1.5,aresample=48000[out]"
    if not narration_ms:
        return g + f"[bed]{master}"
    lines = "".join(f"[{3 + i}:a]{fmt},adelay={ms}:all=1[n{i}];" for i, ms in enumerate(narration_ms))
    names = "".join(f"[n{i}]" for i in range(len(narration_ms)))
    return (g + lines + f"{names}amix=inputs={len(narration_ms)}:normalize=0,apad,atrim=0:{total:g},asplit[nar][key];"
            "[bed][key]sidechaincompress=threshold=0.02:ratio=9:attack=20:release=450[duck];"
            f"[duck][nar]amix=inputs=2:normalize=0,{master}")


# --- engines ----------------------------------------------------------------
# Every heavy step runs through spike-lane (one heavy workload on the machine at a time).
# Engine locations come from the environment so tests can swap in stand-ins.

import argparse  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402

HERE = Path(__file__).resolve().parent
ENGINES = HERE / "engines"
TTS_DESIGN = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"
TTS_CLONE = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"
TTS_SCENE = "OpenMOSS-Team/MOSS-TTSD-v1.0"   # up to 5 speakers in one pass; ~23 GB resident, minutes per scene
STT = "mlx-community/parakeet-tdt-0.6b-v3"


class StageError(RuntimeError):
    """A step failed; the message says which and where its log is."""


def _home(var: str, default: str) -> Path:
    return Path(os.environ.get(var) or Path.home() / default)


def _venv(var: str, default: str, exe: str = "python") -> str:
    return str(_home(var, default) / ".venv" / "bin" / exe)


def _say(msg: str) -> None:
    print(msg, flush=True)


def heavy(name: str, cmd: list, log: Path, cwd: Path | None = None) -> None:
    """Run one heavy step in the lane, waiting for it if something else holds it."""
    lane = [str(HERE / "bin" / "spike-lane"), "run", "--wait", "forever"]
    if name in ("image", "video"):
        lane.append("--evict-ollama")
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as f:
        rc = subprocess.run([*lane, name, "--", *map(str, cmd)], stdout=f, stderr=subprocess.STDOUT, cwd=cwd).returncode
    if rc != 0:
        tail = log.read_text(errors="replace").strip().splitlines()[-8:]
        raise StageError(f"{name} step failed (exit {rc}); log {log}\n  " + "\n  ".join(tail))


def ffmpeg(*args) -> None:
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", *map(str, args)], capture_output=True, text=True)
    if r.returncode != 0:
        raise StageError(f"ffmpeg failed: {r.stderr.strip()[-400:]}")


def media_seconds(path: Path) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        raise StageError(f"cannot read the length of {path}") from None


def render_still(prompt: str, seed: int, out: Path, size=STILL_SIZE, from_image: Path | None = None,
                 strength: float | None = None) -> None:
    model = os.environ.get("IMAGE_MODEL") or str(Path.home() / "image-models" / "z-image-turbo-mflux-q8")
    cmd = ["mflux-generate-z-image-turbo", "--model", model, "--prompt", prompt, "--output", out, "--steps", "9",
           "--seed", seed, "--width", size[0], "--height", size[1]]
    if from_image is not None:
        cmd += ["--image-path", from_image, "--image-strength", strength]
    out.parent.mkdir(parents=True, exist_ok=True)
    heavy("image", cmd, out.with_suffix(".log"))


def contact_sheet(images: list[Path], out: Path, cols: int = 4) -> None:
    lst = out.with_suffix(".txt")
    lst.write_text("".join(f"file '{p}'\nduration 1\n" for p in images) + f"file '{images[-1]}'\n")
    rows = -(-len(images) // cols)
    ffmpeg("-f", "concat", "-safe", "0", "-i", lst, "-vf",
           f"scale=416:240:force_original_aspect_ratio=decrease,pad=416:240:(ow-iw)/2:(oh-ih)/2,"
           f"tile={min(cols, len(images))}x{rows}:padding=6:color=white", "-frames:v", "1", "-update", "1", out)
    lst.unlink()


# --- board ------------------------------------------------------------------

def still_path(story: Story, shot: Shot) -> Path:
    return story.root / "board" / f"{shot.id}.png"


def still_key(story: Story, shot: Shot) -> str:
    return key_for("still", still_prompt(story, shot), shot.seed, STILL_SIZE)


def check_stills(story: Story, paths: dict[str, Path]) -> list[str]:
    """Shots whose PNG doesn't carry the prompt it was asked for (a lost prompt renders an empty scene)."""
    return [sid for sid, p in paths.items()
            if png_prompt(p) != still_prompt(story, next(s for s in story.shots if s.id == sid))]


def cmd_board(story: Story) -> int:
    keys = Keys(story.root)
    for shot in story.shots:
        out, k = still_path(story, shot), still_key(story, shot)
        if keys.stale(out, k):
            _say(f"still {shot.id} …")
            render_still(still_prompt(story, shot), shot.seed, out)
            keys.record(out, k)
    bad = check_stills(story, {s.id: still_path(story, s) for s in story.shots})
    sheet = story.root / "board" / "contact-sheet.jpg"
    contact_sheet([still_path(story, s) for s in story.shots], sheet)
    if bad:
        for sid in bad:
            print(f"✗ {sid}: the still's stored prompt doesn't match the story (lost prompt?); "
                  f"delete board/{sid}.png and run board again", file=sys.stderr)
        return 1
    _say(f"Storyboard ready: {sheet}\nReview it. Retake a shot with `spike-animate retake story.toml SHOT --seeds 7,42`, "
         f"then run `spike-animate film story.toml`.")
    return 0


def cmd_retake(story: Story, shot_id: str, seeds: list[int]) -> int:
    shot = next((s for s in story.shots if s.id == shot_id), None)
    if shot is None:
        print(f"no shot '{shot_id}' in the story", file=sys.stderr)
        return 2
    outs = []
    for seed in seeds:
        out = story.root / "board" / "retakes" / f"{shot.id}-s{seed}.png"
        _say(f"retake {shot.id} seed {seed} …")
        render_still(still_prompt(story, shot), seed, out)
        outs.append(out)
    sheet = story.root / "board" / "retakes" / f"{shot.id}-sheet.jpg"
    contact_sheet(outs, sheet, cols=len(outs))
    _say(f"Retakes: {sheet} (left to right: seeds {', '.join(map(str, seeds))}).\n"
         f"To keep one, set `seed = N` on shot {shot.id} in story.toml and run board again.")
    return 0


# --- film -------------------------------------------------------------------

def shot_path(story: Story, shot: Shot, suffix: str = "") -> Path:
    return story.root / "shots" / f"{shot.id}{suffix}.mp4"


def animate_shot(story: Story, shot: Shot, keys: Keys) -> bool:
    """Render one shot if its inputs changed. Returns True when it re-rendered."""
    still, out, raw = still_path(story, shot), shot_path(story, shot), shot_path(story, shot, "-raw")
    ref = shot.reference
    parts = ["shot", file_digest(still), motion_prompt(story, shot), shot.anchor, shot.seed, shot.colormatch]
    if ref:
        parts += [file_digest(ref.clip), ref.start, ref.sigma, ref.first_strength]
    k = key_for(*parts)
    if not keys.stale(out, k):
        return False
    _say(f"shot {shot.id} …")
    model = _home("WAN5B_MODEL", "wan-models/FastMetal-5B-QAD")
    cmd = [_venv("FASTVIDEO_HOME", "FastVideo"), ENGINES / "wan22_i2v.py", "--mlx-checkpoint", model,
           "--text-encoder-root", model, "--vae-root", model / "vae", "--prompt", motion_prompt(story, shot),
           "--seed", shot.seed, "--output-path", raw]
    if ref:
        # Motion from the real clip; look from a cartoon version of the clip's first frame.
        real, first = shot_path(story, shot, "-first-real").with_suffix(".png"), shot_path(story, shot, "-first").with_suffix(".png")
        w, h = FIRST_FRAME_SIZE
        ffmpeg("-ss", ref.start, "-i", ref.clip, "-frames:v", 1, "-update", 1, "-vf",
               f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}", real)
        render_still(still_prompt(story, shot), shot.seed, first, size=FIRST_FRAME_SIZE, from_image=real,
                     strength=ref.first_strength)
        cmd += ["--init-video", ref.clip, "--init-start", ref.start, "--init-sigma", ref.sigma, "--image", first]
    else:
        cmd += ["--image", still, "--anchor-start", shot.anchor]
    heavy("video", cmd, raw.with_suffix(".log"), cwd=_home("FASTVIDEO_HOME", "FastVideo"))
    if ref:
        # Drop the first frames (the hand-over from the pinned cartoon frame), hold the last to keep 121.
        ffmpeg("-ss", 4 / FPS, "-i", raw, "-vf", "tpad=stop_mode=clone:stop=4", "-frames:v", SHOT_FRAMES,
               "-c:v", "libx264", "-crf", 16, "-pix_fmt", "yuv420p", "-an", out)
    elif shot.colormatch:
        r = subprocess.run([_venv("FASTVIDEO_HOME", "FastVideo"), str(ENGINES / "colormatch.py"), str(still),
                            str(raw), str(out)], capture_output=True, text=True)
        if r.returncode != 0:
            raise StageError(f"colour-match failed for {shot.id}: {r.stderr.strip()[-300:]}")
    else:
        shutil.copyfile(raw, out)
    secs = media_seconds(out)
    if abs(secs - shot_seconds()) > 0.06:
        raise StageError(f"shot {shot.id} is {secs:.2f} s, expected {shot_seconds():.2f} s")
    keys.record(out, k)
    return True


def build_picture(story: Story, keys: Keys) -> Path:
    out = story.root / "film" / "picture.mp4"
    shots = [shot_path(story, s) for s in story.shots]
    k = key_for("picture", [file_digest(p) for p in shots])
    if keys.stale(out, k):
        out.parent.mkdir(parents=True, exist_ok=True)
        lst = out.with_suffix(".txt")
        lst.write_text("".join(f"file '{p}'\n" for p in shots))
        ffmpeg("-f", "concat", "-safe", "0", "-i", lst, "-c:v", "libx264", "-crf", 18, "-pix_fmt", "yuv420p", out)
        keys.record(out, k)
    return out


def sound_shot(story: Story, shot: Shot, keys: Keys) -> Path:
    clip, out = shot_path(story, shot), story.root / "sfx" / f"{shot.id}.flac"
    k = key_for("sfx", file_digest(clip), shot.sfx, story.sfx_negative)
    if keys.stale(out, k):
        _say(f"effects {shot.id} …")
        heavy("sfx", [_venv("MMAUDIO_HOME", "MMAudio"), "demo.py", "--video", clip, "--duration", f"{shot_seconds():.6f}",
                      "--prompt", shot.sfx, "--negative_prompt", story.sfx_negative, "--skip_video_composite",
                      "--output", out.parent], out.with_suffix(".log"), cwd=_home("MMAUDIO_HOME", "MMAudio"))
        keys.record(out, k)
    return out


def build_soundtrack(story: Story, tracks: list[Path], keys: Keys) -> Path:
    """Pad or trim every shot's effects to the exact shot length so the track never drifts off the cuts."""
    out = story.root / "film" / "sfx.wav"
    k = key_for("soundtrack", [file_digest(t) for t in tracks])
    if keys.stale(out, k):
        L = f"{shot_seconds():.6f}"
        ins = [a for t in tracks for a in ("-i", t)]
        g = "".join(f"[{i}:a]apad,atrim=0:{L}[a{i}];" for i in range(len(tracks)))
        g += "".join(f"[a{i}]" for i in range(len(tracks))) + f"concat=n={len(tracks)}:v=0:a=1,loudnorm=I=-16:TP=-1.5[out]"
        ffmpeg(*ins, "-filter_complex", g, "-map", "[out]", "-ar", 44100, "-c:a", "pcm_s16le", out)
        keys.record(out, k)
    return out


def tts(model: str, text: str, out: Path, extra: list) -> None:
    tmp = out.parent / f".{out.stem}"
    heavy("voice", [_venv("VOICE_HOME", "voice", "mlx_audio.tts.generate"), "--model", model, "--text", text, *extra,
                    "--output_path", tmp, "--file_prefix", out.stem, "--audio_format", "wav"], out.with_suffix(".log"))
    made = tmp / f"{out.stem}_000.wav"
    if not made.exists():
        raise StageError(f"voice step wrote nothing for {out.name}; log {out.with_suffix('.log')}")
    made.replace(out)
    shutil.rmtree(tmp, ignore_errors=True)


def heard(audio: Path, keys: Keys) -> str:
    """What speech-to-text hears in `audio` (cached by the audio's content)."""
    txt = audio.with_suffix(".heard.txt")
    k = key_for("stt", file_digest(audio))
    if keys.stale(txt, k):
        stem = audio.parent / f".{audio.stem}-stt"
        heavy("voice", [_venv("VOICE_HOME", "voice", "mlx_audio.stt.generate"), "--model", STT, "--audio", audio,
                        "--output-path", stem, "--format", "txt"], audio.with_suffix(".stt.log"))
        Path(f"{stem}.txt").replace(txt)
        keys.record(txt, k)
    return txt.read_text().strip()


def narrate(story: Story, keys: Keys, checks: dict) -> list[tuple[Path, float]]:
    """Design the narrator's voice once, clone it for every line, and check each line is intelligible."""
    offsets = narration_offsets(story)
    if not offsets:
        return []
    vdir = story.root / "voice"
    vdir.mkdir(exist_ok=True)
    n = story.narrator
    card = vdir / "card.wav"
    k = key_for("card", n.voice, n.sample)
    if keys.stale(card, k):
        _say("narrator voice …")
        tts(TTS_DESIGN, n.sample, card, ["--instruct", n.voice])
        keys.record(card, k)
    lines, checks["narration"] = [], {}
    for sid, at in offsets:
        shot = next(s for s in story.shots if s.id == sid)
        out = vdir / f"{sid}.wav"
        k = key_for("line", file_digest(card), n.sample, shot.narration)
        if keys.stale(out, k):
            _say(f"narration {sid} …")
            tts(TTS_CLONE, shot.narration, out, ["--ref_audio", card, "--ref_text", n.sample])
            keys.record(out, k)
        checks["narration"][sid] = round(words_match(shot.narration, heard(out, keys)), 2)
        lines.append((out, at))
    return lines


def voice_cards(story: Story, keys: Keys) -> dict[str, Path]:
    """One designed voice per speaking character, made once and reused for every line."""
    cdir = story.root / "voice" / "cards"
    cdir.mkdir(parents=True, exist_ok=True)
    cards = {}
    for name in dict.fromkeys(l.speaker for _, l in dialogue_lines(story)):
        v, card = story.voices[name], cdir / f"{name}.wav"
        k = key_for("card", v.voice, v.sample)
        if keys.stale(card, k):
            _say(f"voice for {name} …")
            tts(TTS_DESIGN, v.sample, card, ["--instruct", v.voice])
            keys.record(card, k)
        cards[name] = card
    return cards


def _silences(audio: Path) -> list[tuple[float, float]]:
    r = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(audio), "-af", "silencedetect=n=-35dB:d=0.25", "-f", "null", "-"],
                       capture_output=True, text=True)
    starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", r.stderr)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", r.stderr)]
    return list(zip(starts, ends))


def speak_dialogue(story: Story, keys: Keys, checks: dict) -> list[tuple[Path, float]]:
    """Voice every dialogue line, check it is intelligible, and place it in its shot."""
    said = dialogue_lines(story)
    if not said:
        return []
    cards = voice_cards(story, keys)
    ddir = story.root / "voice" / "dialogue"
    ddir.mkdir(parents=True, exist_ok=True)
    count: dict[str, int] = {}
    outs = []
    for sid, l in said:
        count[sid] = count.get(sid, 0) + 1
        outs.append((sid, l, ddir / f"{sid}-{count[sid]}-{l.speaker}.wav"))
    if story.dialogue_mode == "scene":
        speakers = list(cards)
        if len(speakers) > 5:
            raise StageError("scene mode handles up to 5 speakers; use mode = \"clone\"")
        script = " ".join(f"[S{speakers.index(l.speaker) + 1}] {l.line}" for _, l in said)
        k = key_for("scene", script, [(file_digest(cards[n]), story.voices[n].sample) for n in speakers])
        if any(keys.stale(o, k) for *_, o in outs):
            _say(f"dialogue scene ({len(said)} lines, {len(speakers)} voices) …")
            scene = ddir / "scene.wav"
            refs = [a for n in speakers for a in ("--ref_audio", cards[n], "--ref_text", story.voices[n].sample)]
            tts(TTS_SCENE, script, scene, refs)
            for (a, b), (*_, o) in zip(split_at_silences(_silences(scene), media_seconds(scene), len(said)), outs):
                ffmpeg("-ss", f"{a:.3f}", "-to", f"{b:.3f}", "-i", scene, "-af",
                       "silenceremove=start_periods=1:start_threshold=-40dB,areverse,"
                       "silenceremove=start_periods=1:start_threshold=-40dB,areverse", o)
                keys.record(o, k)
    else:
        for sid, l, o in outs:
            k = key_for("dline", file_digest(cards[l.speaker]), story.voices[l.speaker].sample, l.line)
            if keys.stale(o, k):
                _say(f"dialogue {o.stem} …")
                tts(TTS_CLONE, l.line, o, ["--ref_audio", cards[l.speaker], "--ref_text", story.voices[l.speaker].sample])
                keys.record(o, k)
    checks["dialogue"], checks["dialogue_overflow"] = {}, {}
    placed = []
    index = {s.id: i for i, s in enumerate(story.shots)}
    for sid in dict.fromkeys(sid for sid, *_ in outs):
        mine = [(l, o) for s2, l, o in outs if s2 == sid]
        narr = story.root / "voice" / f"{sid}.wav"
        lead = dialogue_lead(media_seconds(narr) if story.shots[index[sid]].narration and narr.exists() else None,
                             story.narrator.lead if story.narrator else 0.0)
        starts, over = place_lines(index[sid] * shot_seconds(), [media_seconds(o) for _, o in mine], lead=lead)
        if over > 0.5:   # a little spill into the next shot is a natural L-cut; more than that won't read
            checks["dialogue_overflow"][sid] = round(over, 2)
        for (l, o), at in zip(mine, starts):
            checks["dialogue"][o.stem.rsplit("-", 1)[0]] = round(words_match(l.line, heard(o, keys)), 2)
            placed.append((o, at))
    return placed


def score(story: Story, keys: Keys, checks: dict, total: float) -> Path | None:
    """Generate score takes with the tempo locked to the edit, then keep the one that best follows the story."""
    m = story.music
    if m is None:
        return None
    sdir = story.root / "score"
    sdir.mkdir(exist_ok=True)
    seeds = [m.take] if m.take is not None else list(range(1, m.takes + 1))
    spec = {"out": str(sdir), "duration": round(total + 1.2, 2), "bpm": score_bpm(shot_seconds()), "key": m.key,
            "caption": m.caption, "structure": m.structure, "seeds": seeds}
    takes = [sdir / f"take-s{s}.wav" for s in seeds]
    k = key_for("score", spec)
    if any(keys.stale(t, k) for t in takes):
        _say(f"score: {len(seeds)} take(s) at {spec['bpm']} BPM …")
        (sdir / "spec.json").write_text(json.dumps(spec, indent=1))
        heavy("music", [_venv("ACESTEP_HOME", "ACE-Step-1.5"), ENGINES / "score_gen.py", sdir / "spec.json"],
              sdir / "gen.log", cwd=_home("ACESTEP_HOME", "ACE-Step-1.5"))
        for t in takes:
            keys.record(t, k)
    rank = sdir / "rank.json"
    rk = key_for("rank", [file_digest(t) for t in takes], intensity_arc(story))
    if len(takes) > 1 and keys.stale(rank, rk):
        rspec = sdir / "rank-spec.json"
        rspec.write_text(json.dumps({"shot_len": shot_seconds(), "arc": intensity_arc(story),
                                     "hit": hit_time(story), "bpm": spec["bpm"]}))
        r = subprocess.run([_venv("MMAUDIO_HOME", "MMAudio"), str(ENGINES / "score_rank.py"), str(rspec), str(rank),
                            *map(str, takes)], capture_output=True, text=True)
        if r.returncode != 0:
            raise StageError(f"score ranking failed: {r.stderr.strip()[-300:]}")
        keys.record(rank, rk)
    ranking = json.loads(rank.read_text()) if len(takes) > 1 else [{"take": str(takes[0]), "score": None}]
    chosen = Path(ranking[0]["take"])
    checks["score"] = {"chosen": str(chosen), "ranking": ranking}
    return chosen


def mix(story: Story, keys: Keys, picture: Path, soundtrack: Path, music: Path | None,
        lines: list[tuple[Path, float]], total: float) -> Path:
    out = story.root / "film" / f"{story.slug}.mp4"
    mdb = story.music.db if story.music else 0
    k = key_for("mix", file_digest(picture), file_digest(soundtrack), music and file_digest(music),
                [(file_digest(p), at) for p, at in lines], story.sfx_db, mdb)
    if keys.stale(out, k):
        _say("mix …")
        music_in = ["-i", music] if music else ["-f", "lavfi", "-t", f"{total:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
        ins = ["-i", picture, "-i", soundtrack, *music_in]
        for p, _ in lines:
            ins += ["-i", p]
        g = mix_filtergraph([round(at * 1000) for _, at in lines], round(total, 4), story.sfx_db, mdb)
        ffmpeg(*ins, "-filter_complex", g, "-map", "0:v", "-map", "[out]", "-c:v", "copy", "-c:a", "aac",
               "-b:a", "192k", "-shortest", "-movflags", "+faststart", out)
        keys.record(out, k)
    return out


def _beat(s: Shot) -> str:
    if s.dialogue:
        return " ".join(f"{l.speaker}: “{l.line}”" for l in s.dialogue)
    return f"“{s.narration}”" if s.narration else s.motion


def review_page(story: Story, film: Path, checks: dict) -> Path:
    """A self-contained folder (page + film + storyboard) that plays on a phone."""
    rdir = story.root / "film" / "review"
    rdir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(film, rdir / film.name)
    sheet = story.root / "board" / "contact-sheet.jpg"
    if sheet.exists():
        shutil.copyfile(sheet, rdir / "storyboard.jpg")
    esc = lambda s: (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))  # noqa: E731
    beats = "".join(
        f"<li><span>{i * shot_seconds() // 60:.0f}:{i * shot_seconds() % 60:04.1f}</span>"
        f"<b>{esc(s.id)}</b> {esc(_beat(s))}</li>"
        for i, s in enumerate(story.shots))
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>{esc(story.title)}</title><style>
:root{{--bg:#12191c;--fg:#eef1ee;--dim:#9fb0b0;--acc:#e9a99a;color-scheme:dark}}
body{{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 -apple-system,Helvetica,Arial,sans-serif;padding:24px 16px 48px}}
main{{max-width:860px;margin:0 auto;display:grid;gap:20px}}h1{{font:700 2.4rem/1 Georgia,serif;margin:0}}
video,img{{width:100%;border-radius:6px;display:block}}ol{{list-style:none;padding:0;margin:0}}
li{{padding:8px 0;border-top:1px solid #23302f}}li span{{color:var(--acc);margin-right:10px;font-variant-numeric:tabular-nums}}
li b{{color:var(--dim);font-weight:600;margin-right:6px}}p{{color:var(--dim);margin:0}}
</style></head><body><main><h1>{esc(story.title)}</h1>
<video controls playsinline preload="metadata" src="{film.name}"></video>
<p>{len(story.shots)} shots · {media_seconds(film):.1f} s · made with spike-animate</p>
<ol>{beats}</ol>{'<img src="storyboard.jpg" alt="Storyboard">' if sheet.exists() else ''}
</main></body></html>"""
    (rdir / "index.html").write_text(page)
    return rdir / "index.html"


def cmd_film(story: Story) -> int:
    keys = Keys(story.root)
    missing = [s.id for s in story.shots if keys.stale(still_path(story, s), still_key(story, s))]
    if missing:
        print(f"stills missing or out of date for {', '.join(missing)}: run `spike-animate board` first",
              file=sys.stderr)
        return 2
    checks: dict = {}
    for shot in story.shots:
        animate_shot(story, shot, keys)
    picture = build_picture(story, keys)
    total = media_seconds(picture)
    soundtrack = build_soundtrack(story, [sound_shot(story, s, keys) for s in story.shots], keys)
    lines = narrate(story, keys, checks) + speak_dialogue(story, keys, checks)
    music = score(story, keys, checks, total)
    film = mix(story, keys, picture, soundtrack, music, lines, total)
    checks["length"] = {"picture": round(total, 3), "film": round(media_seconds(film), 3)}
    if lines:
        checks["final_narration"] = round(words_match(spoken_text(story), heard(film, keys)), 2)
    (story.root / "film" / "checks.json").write_text(json.dumps(checks, indent=1))
    page = review_page(story, film, checks)
    weak = [sid for part in ("narration", "dialogue") for sid, v in checks.get(part, {}).items() if v < 0.8]
    _say(f"Film: {film} ({checks['length']['film']:.1f} s)\nReview page: {page}")
    for sid in weak:
        _say(f"⚠ the line for {sid} is hard to make out; retry with a reworded line")
    for sid, over in checks.get("dialogue_overflow", {}).items():
        _say(f"⚠ dialogue in {sid} runs {over} s past the cut; shorten it or split it across shots")
    if lines and checks["final_narration"] < 0.8:
        _say(f"⚠ speech is hard to hear in the final mix (match {checks['final_narration']}); lower music or effects")
    return 0


def cmd_status(story: Story) -> int:
    keys = Keys(story.root)
    n = len(story.shots)
    stills = sum(not keys.stale(still_path(story, s), still_key(story, s)) for s in story.shots)
    shots = sum(shot_path(story, s).exists() for s in story.shots)
    sfx = sum((story.root / "sfx" / f"{s.id}.flac").exists() for s in story.shots)
    lines = [s for s in story.shots if s.narration]
    voiced = sum((story.root / "voice" / f"{s.id}.wav").exists() for s in lines)
    film = story.root / "film" / f"{story.slug}.mp4"
    _say(f"{story.title}: stills {stills}/{n} · shots {shots}/{n} · effects {sfx}/{n} · "
         f"narration {voiced}/{len(lines)} · film {'ready' if film.exists() else 'not yet'}")
    return 0


# --- new: draft a story from an idea ----------------------------------------
# A local model (Ollama, gemma4:26b by default) writes the story; a JSON schema shapes its output,
# the linter enforces the story rules, and a draft that breaks them goes back with the problems.

import urllib.request  # noqa: E402

DRAFT_SCHEMA = {
    "type": "object",
    "required": ["title", "look", "still_style", "characters", "narrator", "music", "shots"],
    "properties": {
        "title": {"type": "string"},
        "look": {"type": "string"},
        "still_style": {"type": "string"},
        "characters": {"type": "array", "items": {"type": "object", "required": ["name", "description", "voice"],
                       "properties": {"name": {"type": "string"}, "description": {"type": "string"},
                                      "voice": {"type": "string"}}}},
        "narrator": {"type": "object", "required": ["voice", "sample"],
                     "properties": {"voice": {"type": "string"}, "sample": {"type": "string"}}},
        "music": {"type": "object", "required": ["caption", "structure", "key"],
                  "properties": {"caption": {"type": "string"}, "structure": {"type": "string"}, "key": {"type": "string"}}},
        "shots": {"type": "array", "items": {"type": "object",
                  "required": ["id", "still", "motion", "sfx", "narration", "intensity", "action"],
                  "properties": {"id": {"type": "string"}, "still": {"type": "string"}, "motion": {"type": "string"},
                                 "sfx": {"type": "string"}, "narration": {"type": "string"},
                                 "intensity": {"type": "number"}, "action": {"type": "boolean"},
                                 "dialogue": {"type": "array", "items": {"type": "object",
                                              "required": ["speaker", "line", "emotion"],
                                              "properties": {"speaker": {"type": "string"}, "line": {"type": "string"},
                                                             "emotion": {"type": "string"}}}}}}},
    },
}

MAX_NARRATION_WORDS = 12   # ~3.5 s of unhurried narration: fits inside a 5 s shot with its lead-in
MAX_DIALOGUE_WORDS = 12    # all the dialogue in one shot, for the same reason

WRITER_EXAMPLE = {
    "title": "Fox Hunt",
    "look": "2D cartoon animation, soft muted storybook colors, dusky teal pine trees, soft white snow, gentle shading",
    "still_style": "2D cartoon animation still, bold black outlines, flat vibrant colors, simple shapes, "
                   "Saturday-morning cartoon style, snowy pine forest",
    "characters": [{"name": "FOX", "description": "a slender red fox with bright orange fur, white chest and muzzle, "
                                                  "black legs, bushy tail with a white tip, amber eyes",
                    "voice": "a smooth, low, purring voice, sly and unhurried, a villain who enjoys himself"},
                   {"name": "RAB", "description": "a plump brown rabbit with long ears and a fluffy white tail",
                    "voice": ""}],
    "narrator": {"voice": "A deep, warm, older male storyteller with a gentle British accent, slow and hushed, "
                          "like reading a dark fairy tale aloud by firelight",
                 "sample": "Deep in the winter woods, where the snow falls soft and silent, every creature knows one "
                           "simple rule. The hungry hunt, and the careless are eaten."},
    "music": {"caption": "dark fairy tale film score, small chamber orchestra, solo cello, pizzicato strings, celesta, "
                         "timpani; starts hushed and curious, tightens into suspense, explodes into a violent attack, "
                         "ends cold and quiet; cinematic, no vocals",
              "structure": "[Intro - hushed celesta, snowy forest]\n\n[Build - tremolo strings, creeping suspense]\n\n"
                           "[Climax - pounding timpani, brass stabs]\n\n[Outro - low cello drone, fade out]",
              "key": "D minor"},
    "shots": [
        {"id": "01-establish", "still": "wide shot of FOX walking through a snowy pine forest at dawn, soft pink sky",
         "motion": "static camera, a slender red fox walks steadily through the snowy pine forest at dawn",
         "sfx": "soft paw footsteps crunching in snow, quiet winter wind", "narration":
         "Deep in the winter woods, the fox was hungry.", "intensity": 0.15, "action": False, "dialogue": []},
        {"id": "02-reveal", "still": "point-of-view shot through snowy branches of RAB nibbling grass, unaware",
         "motion": "static camera, a brown rabbit nibbles grass, chewing, ears twitching",
         "sfx": "rabbit nibbling, soft rustling, quiet birdsong", "narration": "Breakfast. And it had no idea.",
         "intensity": 0.3, "action": False, "dialogue": []},
        {"id": "03-taunt", "still": "medium shot of FOX crouched behind a snowy log, grinning, eyes on RAB",
         "motion": "static camera, a red fox grins and talks quietly, eyes fixed ahead",
         "sfx": "soft wind, a twig creaking", "narration": "",
         "dialogue": [{"speaker": "FOX", "line": "Good morning, little breakfast.", "emotion": "sly, purring"}],
         "intensity": 0.6, "action": False},
        {"id": "03-pounce", "still": "action shot of FOX leaping through the air to pounce on RAB, snow spraying",
         "motion": "a red fox leaps through the air and pounces on a brown rabbit, snow spraying, fast dynamic action",
         "sfx": "fast whoosh of a leap, paws thudding into snow, rabbit squeal", "narration": "",
         "intensity": 1.0, "action": True, "dialogue": []},
    ],
}


def writer_prompt(idea: str, shots: int, problems: list[str] | None = None, previous: dict | None = None,
                  notes: str = "") -> str:
    ref = f"\nREFERENCE NOTES from the author (follow them: names, looks, setting, tone, events):\n{notes.strip()}\n" if notes.strip() else ""
    p = f"""You are the story artist for a short 2D cartoon. Turn the idea below into a shot list of exactly {shots} shots.

IDEA: {idea}
{ref}
How the film is made, so write for it:
- Every shot is a single 5 seconds clip animated from one storyboard still. One clear action per shot.
- Stills are rendered by an image model, so each still is a concrete visual description: framing (wide shot,
  close-up, low angle, point-of-view), who, doing what, where, lighting. No camera moves, no time passing.
- Characters keep the same look only if they are described with the same words every time. So give each
  character a name that is one UPPERCASE word, letters only (FOX, KEEPER, BOLT), and a full visual
  description once, in "characters".
  In every still, write the UPPERCASE name, never re-describe the character.
- "motion" says what moves during the 5 seconds, in plain words without the UPPERCASE names; start calm shots
  with "static camera,". Set "action": true only for fast, violent or acrobatic shots.
- "sfx" lists the sounds of that shot (no music, no speech).
- "narration" is optional and at most {MAX_NARRATION_WORDS} words; leave it "" on action shots. Narrate about half
  the shots at most. The narrator speaks over the film.
- Characters may talk if the idea calls for it. Give every character who speaks a "voice": how they sound
  (pitch, pace, texture, personality), e.g. "small, bright, chirpy robot voice, fast and excitable"; leave
  "voice" "" for characters who never speak. Put spoken lines in a shot's "dialogue" list as
  {{"speaker": NAME, "line": ..., "emotion": ...}}. Keep one speaker per shot and cut between speakers
  (shot / reverse shot); at most {MAX_DIALOGUE_WORDS} words of speech per shot, counting narration and dialogue
  together (the narrator speaks first, then the character). Frame talking shots as medium shots or over the shoulder and say "talking" in the motion.
  Reaction shots of the listener, with no dialogue, make conversations read well. Use "dialogue": [] otherwise.
- "intensity" is the story's energy in that shot, from 0 (calm) to 1 (peak). Build to one peak near the end.
- "look" starts with "2D cartoon animation," then the palette and setting. "still_style" starts with
  "2D cartoon animation still, bold black outlines, flat vibrant colors," then the setting.
- "music" describes an instrumental score that follows the intensity arc; "structure" is 3-6 section tags
  like "[Intro - ...]" separated by blank lines; "key" like "D minor" or "C major".
- Shot ids look like 01-name, 02-name, ... in order.

Example of the format and the level of detail (a different film, 4 shots; FOX talks, RAB does not):
{json.dumps(WRITER_EXAMPLE, indent=1)}

Write the story for the IDEA as one JSON object in the same format, with exactly {shots} shots."""
    if problems:
        if previous is not None:
            p += "\n\nYour previous draft:\n" + json.dumps(previous, indent=1)
        p += "\n\nIt broke these rules:\n" + "\n".join(f"- {x}" for x in problems) + \
             "\nReturn the same story with only these problems fixed; keep everything else as it is."
    return p


def _ngrams(words: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def lint_draft(d: dict, shots: int) -> list[str]:
    """Story rules a draft must follow; each problem says which shot and how to fix it."""
    problems = []
    got = d.get("shots", [])
    if len(got) != shots:
        problems.append(f"the story needs exactly {shots} shots, it has {len(got)}")
    names = {c["name"]: c["description"] for c in d.get("characters", [])}
    for s in got:
        sid = s.get("id", "?")
        if not re.fullmatch(r"\d\d-[a-z0-9-]+", sid):
            problems.append(f"shot id '{sid}' must look like 01-name (two digits, a dash, lowercase words)")
        words = len(s.get("narration", "").split())
        if words > MAX_NARRATION_WORDS:
            problems.append(f"shot {sid} narration has {words} words; keep it to {MAX_NARRATION_WORDS} or fewer so it fits a 5 second shot")
        if not 0 <= s.get("intensity", 0) <= 1:
            problems.append(f"shot {sid} intensity {s.get('intensity')} must be between 0 and 1")
        said = s.get("dialogue", [])
        talk = sum(len(x.get("line", "").split()) for x in said)
        if talk > MAX_DIALOGUE_WORDS:
            problems.append(f"shot {sid} dialogue has {talk} words; keep it to {MAX_DIALOGUE_WORDS} or fewer so it fits a 5 second shot")
        voiced = {c["name"] for c in d.get("characters", []) if c.get("voice")}
        for x in said:
            who = x.get("speaker")
            if who not in names:
                problems.append(f"shot {sid}: speaker {who} is not one of the characters ({', '.join(names)})")
            elif who not in voiced:
                problems.append(f"{who} speaks in shot {sid} but has no voice; give {who} a voice description")
        both = talk + len(s.get("narration", "").split())
        if said and s.get("narration") and both > MAX_DIALOGUE_WORDS:
            problems.append(f"shot {sid} has {both} words of narration and dialogue together; keep it to "
                            f"{MAX_DIALOGUE_WORDS} or fewer so it fits a 5 second shot")
        still = s.get("still", "")
        still_words = re.findall(r"[a-z']+", still.lower())
        for name, desc in names.items():
            if re.search(rf"\b{re.escape(name)}\b", still):
                continue
            if _ngrams(re.findall(r"[a-z']+", desc.lower()), 4) & _ngrams(still_words, 4):
                problems.append(f"shot {sid} describes {name} in words; write {name} instead so the character looks the same in every shot")
    for name in names:
        if not re.fullmatch(r"[A-Z][A-Z]*", name):
            problems.append(f"character name {name} must be one UPPERCASE word, letters only (like FOX or KEEPER)")
        if not any(re.search(rf"\b{re.escape(name)}\b", s.get("still", "")) for s in got):
            problems.append(f"character {name} is never used in a still; use it or remove it")
    return problems


def _toml_str(v: str) -> str:
    return json.dumps(v, ensure_ascii=False)   # a JSON string is a valid TOML basic string


def draft_to_toml(d: dict, idea: str, notes: str = "") -> str:
    q = _toml_str
    out = [f"# {d['title']}: drafted by spike-animate new from the idea:", f"#   {idea}",
           "# Edit freely, then: spike-animate board story.toml", "",
           f"idea = {q(idea)}", *([f"notes = {q(notes)}"] if notes else []),
           f"title = {q(d['title'])}", "seed = 1024", f"look = {q(d['look'])}", f"still_style = {q(d['still_style'])}",
           "", "[characters]"]
    out += [f"{c['name']} = {q(c['description'])}" for c in d["characters"] if not c.get("voice")]
    for c in d["characters"]:
        if c.get("voice"):
            out += ["", f"[characters.{c['name']}]", f"look = {q(c['description'])}", f"voice = {q(c['voice'])}"]
    n, m = d["narrator"], d["music"]
    out += ["", "[narrator]", f"voice = {q(n['voice'])}", f"sample = {q(n['sample'])}", "lead = 0.5",
            "", "[music]", f"caption = {q(m['caption'])}", f"key = {q(m['key'])}", "takes = 8", "db = -10",
            f"structure = {q(m['structure'])}"]
    for s in d["shots"]:
        out += ["", "[[shot]]", f"id = {q(s['id'])}", f"still = {q(s['still'])}", f"motion = {q(s['motion'])}"]
        if s.get("action"):
            out.append("anchor = 0   # action shot: pin only the first frame, let it move")
        if s.get("sfx"):
            out.append(f"sfx = {q(s['sfx'])}")
        if s.get("narration"):
            out.append(f"narration = {q(s['narration'])}")
        if s.get("dialogue"):
            out.append("dialogue = [" + ", ".join(
                f"{{ speaker = {q(x['speaker'])}, line = {q(x['line'])}, emotion = {q(x.get('emotion', ''))} }}"
                for x in s["dialogue"]) + "]")
        out.append(f"intensity = {float(s['intensity']):g}")
    return "\n".join(out) + "\n"


def ollama_generate(prompt: str, model: str) -> dict:
    host = os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434"
    if "://" not in host:
        host = "http://" + host
    body = {"model": model, "prompt": prompt, "stream": False, "think": False, "format": DRAFT_SCHEMA,
            "keep_alive": 0, "options": {"temperature": 0.7, "num_ctx": 8192}}
    req = urllib.request.Request(f"{host.rstrip('/')}/api/generate", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            reply = json.loads(r.read())
    except OSError as e:
        raise StageError(f"could not reach Ollama at {host} ({e}); is `ollama serve` running?") from None
    try:
        return json.loads(reply["response"])
    except (KeyError, ValueError):
        raise StageError("the writer model did not return a JSON story") from None


def cmd_new(folder: Path, idea: str, shots: int, model: str, attempts: int = 3, notes: str = "") -> int:
    import spike_lane
    story_file = folder / "story.toml"
    if story_file.exists():
        print(f"{story_file} already exists; pick a new folder or edit that story", file=sys.stderr)
        return 2
    _say(f"drafting a {shots}-shot story with {model} …")
    fd = spike_lane.acquire("writer", ["spike-animate", "new", idea], wait=None)
    try:
        problems, d = None, None
        for attempt in range(1, attempts + 1):
            d = ollama_generate(writer_prompt(idea, shots, problems, previous=d if problems else None, notes=notes), model)
            problems = lint_draft(d, shots)
            if not problems:
                break
            _say(f"draft {attempt} broke {len(problems)} rule(s); asking for a fix …")
    finally:
        os.close(fd)
    if problems:
        print("the writer couldn't produce a valid story:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    folder.mkdir(parents=True, exist_ok=True)
    story_file.write_text(draft_to_toml(d, idea, notes))
    load_story(story_file)   # must load cleanly
    _say(f"Story drafted: {story_file} ({d['title']}, {len(d['shots'])} shots)\n"
         f"Read and edit it, then run `spike-animate board {story_file}`.")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="spike-animate", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, helptext in [("board", "render the storyboard stills and a contact sheet, then stop for review"),
                           ("film", "animate, add effects, narration and score, and mix the film"),
                           ("status", "show what is done")]:
        sub.add_parser(name, help=helptext).add_argument("story", type=Path)
    nw = sub.add_parser("new", help="draft story.toml for a new film from a one-line idea (local model)")
    nw.add_argument("folder", type=Path)
    nw.add_argument("idea")
    nw.add_argument("--shots", type=int, default=8)
    nw.add_argument("--notes", default="", help="reference material for the writer: names, looks, setting, tone")
    nw.add_argument("--model", default=os.environ.get("SPIKE_WRITER_MODEL", "gemma4:26b"))
    rt = sub.add_parser("retake", help="render alternate seeds for one shot's still")
    rt.add_argument("story", type=Path)
    rt.add_argument("shot")
    rt.add_argument("--seeds", required=True, help="comma-separated, e.g. 7,42,99")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "new":
            return cmd_new(a.folder, a.idea, a.shots, a.model, notes=a.notes)
        story = load_story(a.story)
        if a.cmd == "board":
            return cmd_board(story)
        if a.cmd == "retake":
            return cmd_retake(story, a.shot, [int(s) for s in a.seeds.split(",") if s.strip()])
        if a.cmd == "film":
            return cmd_film(story)
        return cmd_status(story)
    except (StoryError, StageError) as e:
        print(f"spike-animate: {e}", file=sys.stderr)
        return 1
