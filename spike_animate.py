"""spike-animate — storyboard, animate, score and mix a short cartoon from one story file.

A film is a directory holding `story.toml`. Every stage writes into that directory and
records a key (a hash of everything that went into an output) in `.spike-animate/keys.json`,
so re-running a command only re-renders what changed. Heavy steps go through spike-lane.

The rules here were proven on the fox-hunt film (2026-10):
  * every shot is 121 frames at 24 fps (5.04 s), unless the story says pace = "voice": then each
    shot lasts its own lines plus a breath (49-121 frames, a count the video model can make);
  * stills: Z-Image Turbo, characters pasted verbatim from the story + a fixed seed;
  * video: Wan 2.2 5B started from the still (latent frame 0 pinned, light anchoring),
    colour-matched back to the still; or restyled from a real clip with a cartoon first frame;
  * score tempo = 8 beats per shot, so every cut falls on a bar line.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import math
import re
import tomllib
import os
from dataclasses import dataclass, field
from pathlib import Path

SHOT_FRAMES = 121
MIN_FRAMES = 49                 # the shortest clip a voice-paced shot gets (about 2 s)
FPS = 24
TAIL = 0.3                      # voice pace: a breath after a shot's last line before the cut
SILENT_SECONDS = 3.0            # voice pace: a shot with nothing said
PACES = ("fixed", "voice")
VIDEO_DIMS = {"sd": (832, 448), "hd": (1280, 704)}   # the video model's size: hd is Wan 2.2 5B's own 720p
END_PLACES = ("middle", "top", "bottom")
TIERS = ("still", "subtle", "full")   # how a shot moves: a camera glide over its picture, small movement, or full motion
CAMERAS = ("push", "pull", "left", "right")   # a still shot's glide: in, out, or across
GLIDE = 0.08                    # a still shot's camera move (engines/kenburns.py has its own copy)
SUBTLE_MOTION = "almost still, only small gentle movement, steady camera"
UPSCALER = "realesr-general-x4v3.pth"                # sd clips are upscaled to the finished size with it
STILL_SIZE = (1248, 720)        # storyboard stills (Z-Image)
FIRST_FRAME_SIZE = (1248, 672)
FINISH_SIZE = (1280, 720)       # the finished film: the 832x448 clips scaled up to fill it, words drawn at this size
TITLE_FONT = os.environ.get("TITLE_FONT", "/System/Library/Fonts/Supplemental/Arial Bold.ttf")  # cartoon first frame for real-clip shots: the video's 832x448 aspect


class StoryError(ValueError):
    """The story file is missing something or contradicts itself."""


@dataclass
class Reference:
    clip: Path
    start: float = 0.0
    sigma: float = 0.88          # noise added to the real clip: lower keeps more real motion
    first_strength: float = 0.4  # how closely the cartoon first frame follows the real one


@dataclass
class Screen:
    """A real picture (a web page, an app) laid onto a screen in the shot, so its words stay readable.
    Corners are where the screen is on the shot's still, 0-1 across and down: top-left, top-right,
    bottom-right, bottom-left. The picture is cropped to the screen's shape from its top, like a browser shows a
    page; `top` skips that much of it first (0-1 of its height), e.g. a phone's address bar."""
    picture: Path
    corners: list[tuple[float, float]]
    top: float = 0.0


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
    angle: str = ""                  # a named camera angle from [angles]
    pictures: list[tuple[Path, str]] = field(default_factory=list)   # this scene's own reference pictures + notes
    motion_seed: int | None = None   # the video's seed when it differs from the still's (a new take)
    end: float | None = None         # use the motion only up to here (seconds) ...
    fill: str = "slow"               # ... then "slow" it down to fill the shot, or "hold" its last frame
    seconds: float | None = None     # voice pace: this shot's own length instead of its lines'
    caption: str | None = None       # on-screen words; None shows the narration, "" shows nothing
    screen: Screen | None = None
    tier: str = "full"               # "still": glide over the picture, no video model; "subtle": small movement
    camera: str = "push"             # a still shot's glide
    lighting: str = ""               # this shot's own light (another time of day); "" uses the film's

    @property
    def video_seed(self) -> int:
        return self.seed if self.motion_seed is None else self.motion_seed


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
class Levels:
    """How loud each layer plays in the final film, in percent of the old normalized mix."""
    sound: float = 20
    music: float = 20
    voices: float = 50


@dataclass
class Angle:
    """A camera setup shared by shots: the same framing words and the same seed every time."""
    frame: str
    seed: int | None = None


@dataclass
class EndCard:
    """The closing words, over the last shot, whose last frame then holds for `hold` seconds."""
    text: str
    sub: str = ""
    hold: float = 1.5
    place: str = "middle"   # or "top" / "bottom": where it sits, clear of a face


@dataclass
class Grade:
    """One colour grade over every shot, so the film reads as one light."""
    warmth: float = 0.0          # -1 cooler .. 1 warmer
    saturation: float = 1.0
    contrast: float = 1.0


@dataclass
class Title:
    text: str
    kind: str        # "caption" (lower third) or "end" (the end card, centred)
    start: float
    end: float
    place: str = "middle"   # the end card's height; captions always sit in the lower third


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
    levels: Levels = field(default_factory=Levels)
    lighting: str = ""                                          # the film bible: light in every shot,
    places: dict[str, str] = field(default_factory=dict)        # places named like characters,
    angles: dict[str, Angle] = field(default_factory=dict)      # and camera setups shots share
    look_seeds: dict[str, int] = field(default_factory=dict)    # the version of each character sheet / place picture
    pace: str = "fixed"                                         # "fixed": 5 s shots; "voice": as long as their lines
    captions: bool = True                                       # burn the narration (or each shot's caption) in
    end_card: EndCard | None = None
    grade: Grade = field(default_factory=Grade)
    quality: str = "sd"                                         # "hd": animate at 1280x704 (2-5 min a clip)

    @property
    def names(self) -> dict[str, str]:
        """Every NAME a still or motion can use, with the words it stands for."""
        return {**self.characters, **self.places}

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
    places = {str(k): str(v) for k, v in d.get("places", {}).items()}
    for name in places:
        if name in characters:
            raise StoryError(f"{name} is both a character and a place; give one of them another name")
    angles = {}
    for name, a in d.get("angles", {}).items():
        angles[name] = Angle(frame=_need(a, "frame", f"angle {name}"), seed=int(a["seed"]) if "seed" in a else None)
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
        angle = s.get("angle", "")
        if angle and angle not in angles:
            raise StoryError(f"shot {sid} uses angle '{angle}', which isn't one of the [angles] ({', '.join(angles) or 'none'})")
        pictures = []
        for pic in s.get("pictures", []):
            f = (path.parent / _need(pic, "file", f"shot {sid} picture")).resolve()
            if not f.is_file():
                raise StoryError(f"shot {sid}: reference picture {pic['file']} is missing")
            pictures.append((f, str(pic.get("note", "")).strip() or "a reference for this scene"))
        shot_seed = int(s["seed"]) if "seed" in s else (angles[angle].seed if angle and angles[angle].seed is not None else seed)
        screen = None
        if "screen" in s:
            sc = s["screen"]
            pic = (path.parent / _need(sc, "picture", f"shot {sid} screen")).resolve()
            if not pic.is_file():
                raise StoryError(f"shot {sid}: screen picture {sc['picture']} is missing")
            corners = [(float(x), float(y)) for x, y in sc.get("corners", [])]
            if len(corners) != 4 or not all(-0.2 <= v <= 1.2 for c in corners for v in c):
                raise StoryError(f"shot {sid}: a screen needs 4 corners (top-left, top-right, bottom-right, "
                                 "bottom-left), each 0-1 across and down the picture")
            top = float(sc.get("top", 0.0))
            if not 0 <= top < 0.9:
                raise StoryError(f"shot {sid}: a screen's top must be between 0 and 0.9")
            screen = Screen(pic, corners, top)
        secs = float(s["seconds"]) if "seconds" in s else None
        if secs is not None and not MIN_FRAMES / FPS <= secs <= shot_seconds():
            raise StoryError(f"shot {sid}: seconds must be between {MIN_FRAMES / FPS:.2f} and {shot_seconds():.2f}")
        end = float(s["end"]) if "end" in s else None
        if end is not None and not 1.0 <= end < shot_seconds():
            raise StoryError(f"shot {sid}: end must be between 1 and {shot_seconds():.2f} seconds")
        if s.get("fill", "slow") not in ("slow", "hold"):
            raise StoryError(f"shot {sid}: fill must be \"slow\" or \"hold\"")
        tier, camera = s.get("tier", "full"), s.get("camera", "push")
        if tier not in TIERS:
            raise StoryError(f"shot {sid}: tier must be one of {', '.join(TIERS)}, not '{tier}'")
        if camera not in CAMERAS:
            raise StoryError(f"shot {sid}: camera must be one of {', '.join(CAMERAS)}, not '{camera}'")
        shots.append(Shot(id=sid, still=_need(s, "still", f"shot {sid}"), motion=_need(s, "motion", f"shot {sid}"),
                          seed=shot_seed, anchor=int(s.get("anchor", 1)), sfx=s.get("sfx", ""),
                          narration=s.get("narration", ""), intensity=float(s.get("intensity", 0.5)),
                          colormatch=bool(s.get("colormatch", ref is None)), reference=ref,
                          still_style=s.get("still_style", ""), dialogue=_lines(s, sid, characters, voices),
                          motion_seed=int(s["motion_seed"]) if "motion_seed" in s else None, end=end,
                          fill=s.get("fill", "slow"), angle=angle, pictures=pictures, seconds=secs,
                          caption=str(s["caption"]).strip() if "caption" in s else None, screen=screen,
                          tier=tier, camera=camera, lighting=str(s.get("lighting", "")).strip()))
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
    lv = {k: float(v) for k, v in d.get("mix", {}).items() if k in ("sound", "music", "voices")}
    if any(v < 0 for v in lv.values()):
        raise StoryError("[mix] levels are percentages and cannot be negative")
    look_seeds = {str(k): int(v) for k, v in d.get("looks", {}).items()}
    pace = d.get("pace", "fixed")
    if pace not in PACES:
        raise StoryError(f"pace must be one of {', '.join(PACES)}, not '{pace}'")
    end_card = None
    if "end_card" in d:
        e = d["end_card"]
        end_card = EndCard(text=_need(e, "text", "[end_card]"), sub=str(e.get("sub", "")), hold=float(e.get("hold", 1.5)),
                           place=str(e.get("place", "middle")))
        if end_card.place not in END_PLACES:
            raise StoryError(f"[end_card] place must be one of {', '.join(END_PLACES)}, not '{end_card.place}'")
        if not 0 <= end_card.hold <= 5:
            raise StoryError("[end_card] hold must be between 0 and 5 seconds")
    gd = d.get("grade", {})
    grade = Grade(warmth=float(gd.get("warmth", 0.0)), saturation=float(gd.get("saturation", 1.0)),
                  contrast=float(gd.get("contrast", 1.0)))
    if not -1 <= grade.warmth <= 1:
        raise StoryError("[grade] warmth must be between -1 and 1")
    if not 0 <= grade.saturation <= 3 or not 0.5 <= grade.contrast <= 2:
        raise StoryError("[grade] saturation must be 0-3 and contrast 0.5-2")
    quality = d.get("quality", "sd")
    if quality not in VIDEO_DIMS:
        raise StoryError(f"quality must be one of {', '.join(VIDEO_DIMS)}, not '{quality}'")
    return Story(quality=quality, grade=grade, pace=pace, captions=bool(d.get("captions", True)), end_card=end_card, levels=Levels(**lv), lighting=d.get("lighting", ""), places=places, angles=angles, look_seeds=look_seeds, path=path, title=_need(d, "title", "the story"), seed=seed, look=_need(d, "look", "the story"),
                 still_style=_need(d, "still_style", "the story"), characters=characters,
                 shots=shots, narrator=narrator, music=music, voices=voices, dialogue_mode=mode,
                 sfx_negative=sound.get("negative", Story.sfx_negative), sfx_db=float(sound.get("db", -4.0)))


# --- prompts ----------------------------------------------------------------

def expand(text: str, characters: dict[str, str]) -> str:
    """Replace each character NAME (whole word) with its fixed description, so every shot
    describes the character with exactly the same words; that is what keeps it consistent."""
    if not characters:
        return text
    # "on the TERRACE" with TERRACE = "a seaside terrace" reads "on a seaside terrace", not "on the a ..."
    pat = re.compile(r"\b(?:(?P<art>the|a|an) )?(?P<name>" + "|".join(map(re.escape, characters)) + r")\b", re.I)

    def sub(m):
        name = next((n for n in characters if n == m.group("name")), None)
        if name is None:   # matched case-insensitively only: not a NAME
            return m.group(0)
        desc = characters[name]
        art = m.group("art")
        if art and not re.match(r"(the|a|an)\b", desc, re.I):
            return f"{art} {desc}"
        return desc

    return pat.sub(sub, text)


def shot_light(story: Story, shot: Shot) -> str:
    return shot.lighting or story.lighting


def still_prompt(story: Story, shot: Shot) -> str:
    frame = f"{expand(story.angles[shot.angle].frame, story.names)}, " if shot.angle else ""
    light = f"{shot_light(story, shot)}, " if shot_light(story, shot) else ""
    return f"{frame}{expand(shot.still, story.names)}, {light}{shot.still_style or story.still_style}"


def motion_prompt(story: Story, shot: Shot) -> str:
    light = f"{shot_light(story, shot)}, " if shot_light(story, shot) else ""
    calm = f"{SUBTLE_MOTION}, " if shot.tier == "subtle" else ""
    return f"{story.look}, {light}{calm}{expand(shot.motion, story.names)}"


# --- timing -----------------------------------------------------------------

def shot_seconds() -> float:
    return SHOT_FRAMES / FPS


def wan_frames(seconds: float) -> int:
    """The clip length for `seconds`: frames the video model can make (4n+1), between about 2 and 5 s."""
    n = math.ceil(round(seconds * FPS, 6))
    return min(SHOT_FRAMES, max(MIN_FRAMES, n + (-(n - 1)) % 4))


def spoken_seconds(story: Story, shot: Shot) -> float | None:
    """When a shot's last line ends, from its recordings; 0 when nothing is said, None while a line isn't recorded."""
    end, narr = 0.0, story.root / "voice" / f"{shot.id}.wav"
    if shot.narration:
        if not narr.exists():
            return None
        end = (story.narrator.lead if story.narrator else 0.0) + media_seconds(narr)
    if shot.dialogue:
        wavs = [story.root / "voice" / "dialogue" / f"{shot.id}-{n}-{l.speaker}.wav" for n, l in enumerate(shot.dialogue, 1)]
        if not all(w.exists() for w in wavs):
            return None
        lead = dialogue_lead(media_seconds(narr) if shot.narration else None, story.narrator.lead if story.narrator else 0.0)
        starts, _ = place_lines(0.0, [media_seconds(w) for w in wavs], lead=lead)
        end = starts[-1] + media_seconds(wavs[-1])
    return end


def shot_frames(story: Story, shot: Shot) -> int | None:
    """How many frames the shot lasts; None while a voice-paced shot's lines aren't recorded."""
    if story.pace != "voice":
        return SHOT_FRAMES
    if shot.seconds is not None:
        return wan_frames(shot.seconds)
    said = spoken_seconds(story, shot)
    if said is None:
        return None
    return wan_frames(said + TAIL if said else SILENT_SECONDS)


def video_dims(story: Story) -> tuple[int, int]:
    return VIDEO_DIMS[story.quality]


def shot_len(story: Story, shot: Shot) -> float:
    """The shot's length in seconds (5.04 s until a voice-paced shot's lines are recorded)."""
    return (shot_frames(story, shot) or SHOT_FRAMES) / FPS


def shot_starts(story: Story) -> list[float]:
    starts, t = [], 0.0
    for s in story.shots:
        starts.append(t)
        t += shot_len(story, s)
    return starts


def film_length(story: Story) -> float:
    """The whole film: every shot, then the end card's held last frame."""
    return sum(shot_len(story, s) for s in story.shots) + (story.end_card.hold if story.end_card else 0.0)


def caption_text(story: Story, shot: Shot) -> str:
    if not story.captions:
        return ""
    return shot.narration.strip() if shot.caption is None else shot.caption


def title_plan(story: Story) -> list[Title]:
    """Every caption over its shot, and the end card from the last shot's lead-in to the end."""
    starts, last = shot_starts(story), len(story.shots) - 1
    out = []
    for i, (s, at) in enumerate(zip(story.shots, starts)):
        if i == last and story.end_card:
            break
        if text := caption_text(story, s):
            out.append(Title(text, "caption", at, at + shot_len(story, s)))
    if story.end_card:
        e, lead = story.end_card, story.narrator.lead if story.narrator else 0.5
        out.append(Title(e.text + (f"\n{e.sub}" if e.sub else ""), "end", starts[last] + lead, film_length(story),
                         e.place))
    return out


def score_bpm(shot_len: float, beats: int = 8) -> int:
    return round(beats * 60 / shot_len)


def narration_offsets(story: Story) -> list[tuple[str, float]]:
    lead = story.narrator.lead if story.narrator else 0.0
    return [(s.id, at + lead) for s, at in zip(story.shots, shot_starts(story)) if s.narration]


def intensity_arc(story: Story) -> list[float]:
    return [s.intensity for s in story.shots]


def hit_time(story: Story) -> float:
    arc = intensity_arc(story)
    return shot_starts(story)[arc.index(max(arc))]


def dialogue_lines(story: Story) -> list[tuple[str, Line]]:
    return [(s.id, l) for s in story.shots for l in s.dialogue]


def spoken_text(story: Story) -> str:
    """Everything said in the film, in order: narration and dialogue."""
    return " ".join(x for s in story.shots for x in ([s.narration] if s.narration else []) + [l.line for l in s.dialogue])


def place_lines(shot_start: float, durations: list[float], lead: float = 0.4, gap: float = 0.3,
                length: float | None = None) -> tuple[list[float], float]:
    """Start times for a shot's lines, one after another; and how far the last runs past the cut (0 if not)."""
    starts, t = [], shot_start + lead
    for d in durations:
        starts.append(t)
        t += d + gap
    end = (starts[-1] + durations[-1]) if durations else shot_start
    return starts, max(0.0, end - (shot_start + (shot_seconds() if length is None else length)))


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

def _pct_db(pct: float) -> float:
    return 20 * math.log10(pct / 100) if pct > 0 else -120.0


def mix_filtergraph(narration_ms: list[int], total: float, sfx_db: float, music_db: float,
                    levels: Levels | None = None, master_db: float = 0.0) -> str:
    """ffmpeg graph for the final mix. Inputs: 0 picture, 1 sound effects, 2 music,
    3.. one narration line each, delayed to its start. Music and effects duck under the voice.
    Without levels it is the plain mix normalized to -16 LUFS (also used to measure master_db,
    the gain that normalization applies). With levels, every layer gets master_db plus its own
    share, and only a limiter follows, so turning a layer down really makes it quieter."""
    fmt = "aresample=48000,aformat=channel_layouts=stereo"
    sfx_db, music_db = (sfx_db, music_db) if levels is None else \
        (round(sfx_db + master_db + _pct_db(levels.sound), 4), round(music_db + master_db + _pct_db(levels.music), 4))
    g = (f"[1:a]{fmt},volume={sfx_db:g}dB[sfx];"
         f"[2:a]{fmt},volume={music_db:g}dB,apad,atrim=0:{total:g}[mus];"
         "[sfx][mus]amix=inputs=2:normalize=0[bed];")
    master = ("loudnorm=I=-16:TP=-1.5" if levels is None else "alimiter=limit=0.84:level=0") + ",aresample=48000[out]"
    if not narration_ms:
        return g + f"[bed]{master}"
    lines = "".join(f"[{3 + i}:a]{fmt},adelay={ms}:all=1[n{i}];" for i, ms in enumerate(narration_ms))
    names = "".join(f"[n{i}]" for i in range(len(narration_ms)))
    g += lines + (f"{names}amix=inputs={len(narration_ms)}:normalize=0,apad,atrim=0:{total:g},asplit[nar][key];"
                  "[bed][key]sidechaincompress=threshold=0.02:ratio=9:attack=20:release=450[duck];")
    if levels is not None:   # the ducking key stays at full level, so ducking works as before
        g += f"[nar]volume={round(master_db + _pct_db(levels.voices), 4):g}dB[narv];"
    return g + f"[duck][{'nar' if levels is None else 'narv'}]amix=inputs=2:normalize=0,{master}"


# --- engines ----------------------------------------------------------------
# Every heavy step runs through spike-lane (one heavy workload on the machine at a time).
# Engine locations come from the environment so tests can swap in stand-ins.

import argparse  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
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
    for attempt in (1, 2):
        with open(log, "w") as f:
            rc = subprocess.run([*lane, name, "--", *map(str, cmd)], stdout=f, stderr=subprocess.STDOUT, cwd=cwd).returncode
        # Memory the previous job (or another app) held may not be free yet: wait, then try once more.
        if rc == 0 or attempt == 2 or "out of memory" not in log.read_text(errors="replace").lower():
            break
        pause = float(os.environ.get("SPIKE_OOM_PAUSE", "30"))
        _say(f"{name} step ran out of memory; trying again in {pause:g} s …")
        time.sleep(pause)
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
                 strength: float | None = None, refs: list[Path] | None = None) -> None:
    """Draw one picture: from the prompt alone (Z-Image), or from reference pictures (FLUX.2 klein edit)."""
    model = os.environ.get("IMAGE_MODEL") or str(Path.home() / "image-models" / "z-image-turbo-mflux-q8")
    # mflux never overwrites: given an existing file it saves NAME_1.png beside it. Render to a fresh
    # name, then move it into place.
    tmp = out.with_name(f".{out.stem}-new{out.suffix}")
    tmp.unlink(missing_ok=True)
    if refs:
        cmd = ["mflux-generate-flux2-edit", "--model", os.environ.get("KLEIN_MODEL") or "flux2-klein-4b", "--quantize", "8",
               "--image-paths", *refs, "--prompt", prompt, "--output", tmp, "--seed", seed,
               "--width", size[0], "--height", size[1]]
    else:
        cmd = ["mflux-generate-z-image-turbo", "--model", model, "--prompt", prompt, "--output", tmp, "--steps", "9",
               "--seed", seed, "--width", size[0], "--height", size[1]]
    if from_image is not None:
        cmd += ["--image-path", from_image, "--image-strength", strength]
    out.parent.mkdir(parents=True, exist_ok=True)
    heavy("image", cmd, out.with_suffix(".log"))
    if not tmp.exists():
        raise StageError(f"the image step wrote nothing for {out.name}; log {out.with_suffix('.log')}")
    os.replace(tmp, out)


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


# --- the Look: one character sheet per character and one picture per place --------------------
# Drawn and approved before the scenes; every scene picture is then drawn from the ones it shows.

IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def look_names(story: Story) -> list[str]:
    return [*story.characters, *story.places]


def look_path(story: Story, name: str) -> Path:
    return story.root / "looks" / f"{name}.png"


def look_take_path(story: Story, name: str, seed: int) -> Path:
    return story.root / "looks" / "takes" / f"{name}-s{seed}.png"


def look_reference(story: Story, name: str) -> Path | None:
    """A picture the author uploaded for a character (reference/NAME.png or .jpg)."""
    if name not in story.characters or not (story.root / "reference").is_dir():
        return None
    found = [p for p in (story.root / "reference").iterdir()
             if p.is_file() and p.stem.lower() == name.lower() and p.suffix.lower() in IMAGE_EXTS]
    return sorted(found)[0] if found else None


def look_seed(story: Story, name: str) -> int:
    return story.look_seeds.get(name, story.seed)


def look_prompt(story: Story, name: str) -> str:
    style = story.still_style
    if name in story.characters:   # the film's look without its setting, so the sheet has a plain background
        style = story.look
    if name in story.places:
        light = f"{story.lighting}, " if story.lighting else ""
        return f"wide establishing view of {story.places[name]}, no people, {light}{style}"
    look = story.characters[name]
    if look_reference(story, name):
        return (f"Character sheet of the person in image 1, keeping exactly their face, hair, skin, markings and "
                f"style. They look like this: {look}. Show them twice side by side: a full body front view and a "
                f"full body three-quarter view, standing in a relaxed neutral pose, on a plain off-white background. {style}")
    return (f"character sheet, the same character shown twice side by side: full body front view and full body "
            f"three-quarter view, standing, neutral pose, plain off-white background, {look}, {style}")


def look_key(story: Story, name: str, seed: int | None = None) -> str:
    ref = look_reference(story, name)
    return key_for("look", look_prompt(story, name), look_seed(story, name) if seed is None else seed,
                   file_digest(ref) if ref else None)


def draw_look(story: Story, name: str, seed: int, keys: Keys) -> Path:
    """One version of a character sheet or place picture, kept in looks/takes/."""
    take, k = look_take_path(story, name, seed), look_key(story, name, seed)
    if keys.stale(take, k):
        _say(f"look {name} seed {seed} …")
        ref = look_reference(story, name)
        render_still(look_prompt(story, name), seed, take, refs=[ref] if ref else None)
        keys.record(take, k)
    return take


def cmd_look(story: Story, name: str, seeds: list[int] | None = None) -> int:
    if name not in look_names(story):
        print(f"'{name}' is not a character or a place in the story", file=sys.stderr)
        return 2
    keys = Keys(story.root)
    if seeds:
        for seed in seeds:
            draw_look(story, name, seed, keys)
        _say(f"{len(seeds)} version(s) of {name} in looks/takes/; keep one with `{name} = SEED` under [looks]")
        return 0
    out, k = look_path(story, name), look_key(story, name)
    if keys.stale(out, k):
        take = draw_look(story, name, look_seed(story, name), keys)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(take, out)
        keys.record(out, k)
    _say(f"Look: {out}")
    return 0


# --- voices, designed and chosen before any line is recorded -----------------------------------

def voice_names(story: Story) -> list[str]:
    return (["narrator"] if story.narrator else []) + list(story.voices)


def card_path(story: Story, name: str) -> Path:
    return story.root / "voice" / ("card.wav" if name == "narrator" else f"cards/{name}.wav")


def card_key(story: Story, name: str) -> str:
    return narrator_card_key(story) if name == "narrator" else character_card_key(story, name)


def voice_take_path(story: Story, name: str, n: int) -> Path:
    return story.root / "voice" / "takes" / f"{name}-{n}.wav"


def _voice_desc(story: Story, name: str) -> tuple[str, str]:
    v = story.narrator if name == "narrator" else story.voices[name]
    return v.voice, v.sample


def voice_takes(story: Story, name: str, keys: Keys) -> list[tuple[int, Path]]:
    """This voice's takes made from its current description and sample line."""
    found = []
    for p in (story.root / "voice" / "takes").glob(f"{name}-*.wav"):
        m = re.fullmatch(rf"{re.escape(name)}-(\d+)\.wav", p.name)
        if m and keys.keys.get(keys._rel(p)) == card_key(story, name):
            found.append((int(m.group(1)), p))
    return sorted(found)


def design_voice(story: Story, name: str, keys: Keys) -> Path:
    """One more take of a voice, kept in voice/takes/ (each run of the voice designer sounds a little different)."""
    taken = [int(m.group(1)) for p in (story.root / "voice" / "takes").glob(f"{name}-*.wav")
             if (m := re.fullmatch(rf"{re.escape(name)}-(\d+)\.wav", p.name))]
    take = voice_take_path(story, name, max(taken, default=0) + 1)
    take.parent.mkdir(parents=True, exist_ok=True)
    voice, sample = _voice_desc(story, name)
    _say(f"voice {name} take {take.stem.rsplit('-', 1)[1]} …")
    tts(TTS_DESIGN, sample, take, ["--instruct", voice])
    keys.record(take, card_key(story, name))
    return take


def cmd_voice(story: Story, name: str, takes: int | None = None) -> int:
    if name not in voice_names(story):
        print(f"'{name}' has no voice in the story (the narrator, or a character with a voice)", file=sys.stderr)
        return 2
    keys = Keys(story.root)
    card = card_path(story, name)
    if not voice_takes(story, name, keys) and not keys.stale(card, card_key(story, name)):
        take = voice_take_path(story, name, 1)   # the voice in use before takes existed: keep it as take 1
        take.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(card, take)
        keys.record(take, card_key(story, name))
    if takes:
        for _ in range(takes):
            design_voice(story, name, keys)
        _say(f"{takes} more take(s) of {name}'s voice in voice/takes/")
        return 0
    if keys.stale(card, card_key(story, name)):
        take = design_voice(story, name, keys)
        card.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(take, card)
        keys.record(card, card_key(story, name))
    _say(f"Voice: {card}")
    return 0


def scene_refs(story: Story, shot: Shot) -> list[tuple[str, Path]]:
    """The looks a scene's picture is drawn from: its characters, then its places (those drawn)."""
    text = shot.still + (" " + story.angles[shot.angle].frame if shot.angle else "")
    names = [n for n in story.characters if re.search(rf"\b{re.escape(n)}\b", text)]
    names += [n for n in story.places if re.search(rf"\b{re.escape(n)}\b", text)]
    return [(n, look_path(story, n)) for n in names if look_path(story, n).is_file()]


def render_prompt(story: Story, shot: Shot) -> tuple[str, list[Path]]:
    """What a scene's picture is drawn from: the prompt, and the reference pictures (none: text only):
    the looks it shows, then the scene's own pictures."""
    refs = scene_refs(story, shot) + [(note, f) for f, note in shot.pictures]
    prompt = still_prompt(story, shot)
    if not refs:
        if any(look_path(story, n).is_file() for n in [*story.characters, *story.places]):
            # the other scenes copy their style from the looks; lead with it here, or a prop alone comes out a photo
            return f"an illustration in this style: {story.look}: {prompt}", []
        return prompt, []

    def say(i, n):
        if n in story.characters:
            return f"image {i} is {n}, keep their face, hair, clothes and markings exactly"
        if n in story.places:
            return (f"image {i} is the place {n}, keep its colours, light and furnishings, seen from this shot's own "
                    f"camera angle and distance")
        return f"image {i} is {n}, copy it as shown"
    said = "; ".join(say(i, n) for i, (n, _) in enumerate(refs, 1))
    return f"{prompt}. Reference pictures: {said}. Match the drawing style of the references.", [p for _, p in refs]


def still_key(story: Story, shot: Shot) -> str:
    prompt, refs = render_prompt(story, shot)
    if not refs:   # the same key as before looks existed, so older pictures stay current
        return key_for("still", prompt, shot.seed, STILL_SIZE)
    return key_for("still", prompt, shot.seed, STILL_SIZE, [file_digest(r) for r in refs])


def check_stills(story: Story, paths: dict[str, Path]) -> list[str]:
    """Shots whose PNG doesn't carry the prompt it was asked for (a lost prompt renders an empty scene)."""
    return [sid for sid, p in paths.items()
            if png_prompt(p) != render_prompt(story, next(s for s in story.shots if s.id == sid))[0]]


def cmd_board(story: Story, shot_id: str | None = None) -> int:
    keys = Keys(story.root)
    shots = _scene(story, shot_id)
    if shots is None:
        return 2
    for shot in shots:
        out, k = still_path(story, shot), still_key(story, shot)
        if keys.stale(out, k):
            _say(f"still {shot.id} …")
            prompt, refs = render_prompt(story, shot)
            render_still(prompt, shot.seed, out, refs=refs)
            keys.record(out, k)
    for shot in shots:
        try:
            _report_check(shot, check_still(story, shot, keys))
        except StageError as e:   # the picture stands; the check is advice
            _say(f"continuity check skipped for {shot.id}: {e}")
    bad = check_stills(story, {s.id: still_path(story, s) for s in shots})
    sheet = story.root / "board" / "contact-sheet.jpg"
    contact_sheet([still_path(story, s) for s in story.shots if still_path(story, s).exists()], sheet)
    if bad:
        for sid in bad:
            print(f"✗ {sid}: the still's stored prompt doesn't match the story (lost prompt?); "
                  f"delete board/{sid}.png and run board again", file=sys.stderr)
        return 1
    _say(f"Storyboard ready: {sheet}\nReview it. Retake a shot with `spike-animate retake story.toml SHOT --seeds 7,42`, "
         f"then run `spike-animate video story.toml`.")
    return 0


# --- continuity: does a picture match the film bible? -------------------------------------

CHECK_SCHEMA = {"type": "object", "required": ["misses"],
                "properties": {"misses": {"type": "array", "items": {"type": "string"}}}}


def _named(text: str, names) -> list[str]:
    return [n for n in names if re.search(rf"\b{re.escape(n)}\b", text)]


def continuity_facts(story: Story, shot: Shot) -> list[str]:
    """The bible facts this picture should show: its characters, its places, the light and the framing."""
    facts = [f"{n}: {story.characters[n]}" for n in _named(shot.still, story.characters)]
    facts += [f"Place {n}: {story.places[n]}" for n in _named(shot.still, story.places)]
    if shot_light(story, shot):
        facts.append(f"Lighting: {shot_light(story, shot)}")
    if shot.angle:
        facts.append(f"Framing: {story.angles[shot.angle].frame}")
    return facts


def continuity_prompt(story: Story, shot: Shot) -> str:
    facts = "\n".join(f"- {f}" for f in continuity_facts(story, shot))
    return f"""You check one storyboard picture of an animated film against the film's fixed design, so every
scene matches the others. Compare the picture with each fact below. List only the facts the picture clearly
breaks, each as one short sentence that says what you see instead, for example "ELENA's shirt is blue, not
cream linen" or "the light comes from the right, not the left". Ignore the drawing style and anything the
facts don't mention. If the picture matches, return an empty list.

Facts:
{facts}

Reply with JSON: {{"misses": [...]}}"""


def check_path(story: Story, shot: Shot) -> Path:
    return story.root / "board" / f"{shot.id}.check.json"


def check_still(story: Story, shot: Shot, keys: Keys, model: str | None = None) -> list[str] | None:
    """Ask the local vision model whether this shot's picture keeps the bible. None when there is
    nothing to check; the answer is kept until the picture or the bible changes."""
    import base64
    if not continuity_facts(story, shot):
        return None
    still, out = still_path(story, shot), check_path(story, shot)
    prompt = continuity_prompt(story, shot)
    k = key_for("check", file_digest(still), prompt)
    if keys.stale(out, k):
        import spike_lane
        _say(f"continuity {shot.id} …")
        fd = spike_lane.acquire("writer", ["spike-animate", "check", shot.id], wait=None)
        try:
            reply = ollama_generate(prompt, model or os.environ.get("SPIKE_WRITER_MODEL", "gemma4:26b"), CHECK_SCHEMA,
                                    [base64.b64encode(still.read_bytes()).decode()])
        finally:
            os.close(fd)
        misses = [" ".join(str(m).split()) for m in reply.get("misses", []) if str(m).strip()]
        out.write_text(json.dumps({"misses": misses}, indent=1))
        keys.record(out, k)
    return json.loads(out.read_text())["misses"]


def _report_check(shot: Shot, misses: list[str] | None) -> None:
    if misses:
        _say(f"⚠ {shot.id} doesn't match the film bible:\n  " + "\n  ".join(misses))
    elif misses is not None:
        _say(f"✓ {shot.id} matches the film bible")


def cmd_check(story: Story, shot_id: str | None = None, motion: bool = False) -> int:
    shots = _scene(story, shot_id)
    if shots is None:
        return 2
    keys = Keys(story.root)
    for shot in shots:
        if motion:
            _report_motion(shot, check_motion(story, shot, keys))
        elif still_path(story, shot).exists():
            _report_check(shot, check_still(story, shot, keys))
    return 0


# --- the motion check: does anything pass through anything else? -------------------------------

MOTION_FRAMES = 6
MOTION_SCHEMA = {"type": "object", "required": ["faults"],
                 "properties": {"faults": {"type": "array", "items": {"type": "string"}}}}


def motion_check_prompt(story: Story, shot: Shot) -> str:
    return f"""These {MOTION_FRAMES} pictures are frames from one shot of an animated film, in order from start to
end. The shot shows: {expand(shot.still, story.names)}; what moves: {expand(shot.motion, story.names)}.
Look for mistakes the animation made, the kind a viewer notices:
- a body part passing through furniture, a person or an object (legs through a table, a hand through a laptop);
- someone sitting on nothing, or an object floating or sinking into another;
- a limb, a hand or an object that disappears, appears from nowhere or doubles;
- an object or a face that changes shape or turns into something else (a mug becoming a paper cup).
List each mistake as one short sentence saying what happens and when ("her right arm disappears as she
lowers it"). Ignore the drawing style, the camera moving, and small wobbles. If there are none, return an
empty list.

Reply with JSON: {{"faults": [...]}}"""


def motion_path(story: Story, shot: Shot) -> Path:
    return story.root / "shots" / f"{shot.id}.motion.json"


def motion_model(model: str | None = None) -> str:
    return model or os.environ.get("SPIKE_WRITER_MODEL", "gemma4:26b")


def motion_key(story: Story, shot: Shot, model: str | None = None) -> str:
    return key_for("motion-check", file_digest(shot_path(story, shot)), motion_check_prompt(story, shot),
                   motion_model(model))


def _motion_verdict(out: Path) -> list[str] | None:
    try:
        return [str(f) for f in json.loads(out.read_text())["faults"]]
    except (OSError, ValueError, KeyError, TypeError):
        return None


def check_motion(story: Story, shot: Shot, keys: Keys, model: str | None = None) -> list[str] | None:
    """Ask the local vision model whether this shot's clip has things passing through each other. None for a
    glide (nothing in the picture moves), or while the clip is missing or older than the shot's picture and
    motion; kept until the clip or the model changes."""
    import base64
    import tempfile
    clip, out = shot_path(story, shot), motion_path(story, shot)
    if shot.tier == "still" or not clip.exists():
        return None
    if not clip_fresh(story, shot, keys):
        _say(f"{shot.id}'s clip is older than its picture or motion: animate it again before checking its motion")
        return None
    prompt, k = motion_check_prompt(story, shot), motion_key(story, shot, model)
    faults = None if keys.stale(out, k) else _motion_verdict(out)
    if faults is None:
        import spike_lane
        _say(f"motion check {shot.id} …")
        secs = media_seconds(clip)
        pics = []
        with tempfile.TemporaryDirectory() as frames:       # fresh every time: never a frame of an older take
            for i in range(MOTION_FRAMES):
                f = Path(frames) / f"{i}.jpg"
                ffmpeg("-ss", f"{secs * (i + 0.5) / MOTION_FRAMES:.3f}", "-i", clip, "-frames:v", 1, "-update", 1,
                       "-vf", "scale=640:-2", f)
                if not f.exists():
                    raise StageError(f"could not take frame {i + 1} of {clip}")
                pics.append(base64.b64encode(f.read_bytes()).decode())
        fd = spike_lane.acquire("writer", ["spike-animate", "check", "--motion", shot.id], wait=None)
        try:
            reply = ollama_generate(prompt, motion_model(model), MOTION_SCHEMA, pics)
        finally:
            os.close(fd)
        faults = [" ".join(str(m).split()) for m in reply.get("faults", []) if str(m).strip()]
        out.write_text(json.dumps({"faults": faults}, indent=1))
        keys.record(out, k)
    return faults


def _report_motion(shot: Shot, faults: list[str] | None) -> None:
    if faults:
        _say(f"⚠ {shot.id}'s motion has mistakes:\n  " + "\n  ".join(faults))
    elif faults is not None:
        _say(f"✓ {shot.id}'s motion looks clean")


def check_motions(story: Story, shots: list[Shot], keys: Keys) -> None:
    """The motion check after animating; it is advice, so a check that can't run never stops the film."""
    for shot in shots:
        try:
            _report_motion(shot, check_motion(story, shot, keys))
        except (StageError, OSError, ValueError) as e:
            _say(f"motion check skipped for {shot.id}: {e}")


# --- ranking: which version of a scene's picture is best? --------------------------------------

RANK_SCHEMA = {"type": "object", "required": ["score", "faults"],
               "properties": {"score": {"type": "integer"}, "faults": {"type": "array", "items": {"type": "string"}}}}


def rank_prompt(story: Story, shot: Shot) -> str:
    facts = "\n".join(f"- {f}" for f in continuity_facts(story, shot)) or "- (none)"
    who = _named(shot.still, story.characters)
    people = (f"exactly {len(who)} {'person' if len(who) == 1 else 'people'} ({', '.join(who)}), each drawn once"
              if who else "no people, unless the scene mentions hands or someone")
    return f"""You judge one storyboard picture for an animated film; several versions of the same scene are drawn
and the best one is kept. Judge how well it:
- shows the scene: {shot.still} (the NAMEs are described in the facts);
- has {people};
- keeps the film's fixed design (the facts below);
- reads as a finished illustration: well-formed faces and hands, no extra or missing limbs, no garbled
  lettering, no smears or broken shapes, a clear composition;
- makes physical sense: nobody sitting on nothing, no legs or arms passing through a table, chair or each
  other, nothing floating or cut through, a table either in front of a person or behind them, not both.
List its faults, each one short sentence ("her left hand has six fingers"); none if it has none.
Score it by this rubric:
- 9-10: no faults, or only a tiny one nobody would notice;
- 7-8: small faults (a detail of the design is off, a slightly odd hand);
- 4-6: a clear fault (wrong clothes or hair, garbled shapes, the wrong framing, legs passing through a
  table, someone sitting on nothing);
- 1-3: broken: a character drawn twice or missing, someone who isn't in the scene, the wrong place.

Facts:
{facts}

Reply with JSON: {{"score": N, "faults": [...]}}"""


def versions(story: Story, shot: Shot) -> list[tuple[int, Path]]:
    """The scene's picture and every other version drawn from the same description: (seed, file)."""
    prompt = render_prompt(story, shot)[0]
    out = [(shot.seed, still_path(story, shot))] if still_path(story, shot).exists() else []
    for p in sorted((story.root / "board" / "retakes").glob(f"{shot.id}-s*.png")):
        m = re.fullmatch(rf"{re.escape(shot.id)}-s(\d+)\.png", p.name)
        if m and int(m[1]) != shot.seed and png_prompt(p) == prompt:
            out.append((int(m[1]), p))
    return out


def rank_versions(story: Story, shot: Shot, keys: Keys, model: str | None = None) -> list[dict]:
    """Score every version with the local vision model (each one once), best first; saved beside the board."""
    import base64
    import spike_lane
    prompt, rdir = rank_prompt(story, shot), story.root / "board" / "ranks"
    rdir.mkdir(parents=True, exist_ok=True)
    rows, fd = [], None
    try:
        for seed, pic in versions(story, shot):
            out, k = rdir / f"{shot.id}-s{seed}.json", key_for("rank", file_digest(pic), prompt)
            if keys.stale(out, k):
                if fd is None:
                    _say(f"ranking the versions of {shot.id} …")
                    fd = spike_lane.acquire("writer", ["spike-animate", "rank", shot.id], wait=None)
                reply = ollama_generate(prompt, model or os.environ.get("SPIKE_WRITER_MODEL", "gemma4:26b"), RANK_SCHEMA,
                                        [base64.b64encode(pic.read_bytes()).decode()])
                faults = [" ".join(str(f).split()) for f in reply.get("faults", []) if str(f).strip()]
                out.write_text(json.dumps({"score": max(1, min(10, int(reply.get("score", 1)))), "faults": faults}))
                keys.record(out, k)
            rows.append({"seed": seed, **json.loads(out.read_text())})
    finally:
        if fd is not None:
            os.close(fd)
    rows.sort(key=lambda r: -r["score"])        # stable: the scene's own picture wins a tie
    (story.root / "board" / f"{shot.id}.rank.json").write_text(json.dumps(rows, indent=1))
    return rows


def cmd_rank(story: Story, shot_id: str) -> int:
    shots = _scene(story, shot_id)
    if shots is None:
        return 2
    rows = rank_versions(story, shots[0], Keys(story.root))
    for r in rows:
        _say(f"seed {r['seed']}: {r['score']}/10" + (f" ({'; '.join(r['faults'])})" if r["faults"] else ""))
    if rows:
        _say(f"Best: seed {rows[0]['seed']}")
    return 0


def cmd_retake(story: Story, shot_id: str, seeds: list[int], rank: bool = False) -> int:
    shot = next((s for s in story.shots if s.id == shot_id), None)
    if shot is None:
        print(f"no shot '{shot_id}' in the story", file=sys.stderr)
        return 2
    outs = []
    for seed in seeds:
        out = story.root / "board" / "retakes" / f"{shot.id}-s{seed}.png"
        _say(f"retake {shot.id} seed {seed} …")
        prompt, refs = render_prompt(story, shot)
        render_still(prompt, seed, out, refs=refs)
        outs.append(out)
    sheet = story.root / "board" / "retakes" / f"{shot.id}-sheet.jpg"
    contact_sheet(outs, sheet, cols=len(outs))
    _say(f"Retakes: {sheet} (left to right: seeds {', '.join(map(str, seeds))}).\n"
         f"To keep one, set `seed = N` on shot {shot.id} in story.toml and run board again.")
    if rank:
        try:
            cmd_rank(story, shot_id)
        except StageError as e:   # the versions stand; the ranking is advice
            _say(f"ranking skipped: {e}")
    return 0


# --- film -------------------------------------------------------------------

def shot_path(story: Story, shot: Shot, suffix: str = "") -> Path:
    return story.root / "shots" / f"{shot.id}{suffix}.mp4"


def full_path(story: Story, shot: Shot) -> Path:
    """The whole 5 seconds as the video model made it; shot_path is this, trimmed."""
    return story.root / "shots" / f"{shot.id}-full.mp4"


def take_path(story: Story, shot: Shot, seed: int) -> Path:
    return story.root / "shots" / "takes" / f"{shot.id}-m{seed}.mp4"


def shot_key(story: Story, shot: Shot, seed: int | None = None) -> str | None:
    """What a shot's motion is made from (None while its still is missing); `seed` asks about another take."""
    still, ref = still_path(story, shot), shot.reference
    if not still.exists() or (ref and not Path(ref.clip).exists()):
        return None
    n = shot_frames(story, shot)
    if n is None:
        return None
    if shot.tier == "still":
        sc = shot.screen
        return key_for("glide", file_digest(still), shot.camera, n, FINISH_SIZE,
                       [file_digest(sc.picture), sc.corners, sc.top] if sc else None)
    parts = ["shot", file_digest(still), motion_prompt(story, shot), shot.anchor,
             shot.video_seed if seed is None else seed, shot.colormatch]
    if story.pace == "voice":
        parts.append(n)
    if story.quality != "sd":
        parts.append(video_dims(story))
    if ref:
        parts += [file_digest(ref.clip), ref.start, ref.sigma, ref.first_strength]
    return key_for(*parts)


def clip_key(story: Story, shot: Shot) -> str | None:
    full = full_path(story, shot)
    return key_for("clip", file_digest(full), shot.end, shot.fill if shot.end else None) if full.exists() else None


def adopt_old_shot(story: Story, shot: Shot, keys: Keys) -> None:
    """Shots made before takes existed: the shot itself is the whole take."""
    out, full = shot_path(story, shot), full_path(story, shot)
    k = shot_key(story, shot)
    if not full.exists() and out.exists() and k and keys.keys.get(keys._rel(out)) == k:
        for dest in (full, take_path(story, shot, shot.video_seed)):
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(out, dest)
            keys.record(dest, k)
        keys.record(out, clip_key(story, shot)) if shot.end is None else None


def clip_fresh(story: Story, shot: Shot, keys: Keys) -> bool:
    """The shot's clip matches its picture, motion, seed and trim (also for shots made before takes existed)."""
    out, full, k = shot_path(story, shot), full_path(story, shot), shot_key(story, shot)
    if k is None:
        return False
    if not full.exists():
        return shot.end is None and out.exists() and keys.keys.get(keys._rel(out)) == k
    return not keys.stale(full, k) and not keys.stale(out, clip_key(story, shot))


def trim_clip(story: Story, shot: Shot, keys: Keys) -> None:
    """The shot from its whole take: as is, or ended early and filled back to its length (slowed down or held)."""
    full, out = full_path(story, shot), shot_path(story, shot)
    k = clip_key(story, shot)
    if not keys.stale(out, k):
        return
    if shot.end is None:
        shutil.copyfile(full, out)
    else:
        n = shot_frames(story, shot) or SHOT_FRAMES
        E, L = min(shot.end, n / FPS - 0.2), n / FPS
        vf = (f"trim=end={E},setpts=(PTS-STARTPTS)*{L / E:.6f},minterpolate=fps={FPS}:mi_mode=mci:mc_mode=aobmc:vsbmc=1,"
              f"tpad=stop_mode=clone:stop=12" if shot.fill == "slow" else
              f"trim=end={E},setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop={n}")
        _say(f"trim {shot.id} at {E:g} s ({'slowed down' if shot.fill == 'slow' else 'last frame held'}) …")
        ffmpeg("-i", full, "-vf", vf, "-frames:v", n, "-c:v", "libx264", "-crf", 16, "-pix_fmt", "yuv420p", "-an", out)
    keys.record(out, k)


def animate_shot(story: Story, shot: Shot, keys: Keys) -> bool:
    """Render one shot if its inputs changed, then trim it. Returns True when the video model ran."""
    still, out, raw = still_path(story, shot), full_path(story, shot), shot_path(story, shot, "-raw")
    ref = shot.reference
    adopt_old_shot(story, shot, keys)
    k = shot_key(story, shot)
    if not keys.stale(out, k):
        trim_clip(story, shot, keys)
        return False
    n = shot_frames(story, shot) or SHOT_FRAMES
    if shot.tier == "still":
        glide_shot(story, shot, keys, k, n)
        return False
    _say(f"shot {shot.id} ({n / FPS:.2f} s) …")
    model = _home("WAN5B_MODEL", "wan-models/FastMetal-5B-QAD")
    cmd = [_venv("FASTVIDEO_HOME", "FastVideo"), ENGINES / "wan22_i2v.py", "--mlx-checkpoint", model,
           "--text-encoder-root", model, "--vae-root", model / "vae", "--prompt", motion_prompt(story, shot),
           "--seed", shot.video_seed, "--num-frames", n, "--output-path", raw]
    if story.quality != "sd":
        w, h = video_dims(story)
        cmd += ["--width", w, "--height", h]
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
        # Drop the first frames (the hand-over from the pinned cartoon frame), hold the last to keep the length.
        ffmpeg("-ss", 4 / FPS, "-i", raw, "-vf", "tpad=stop_mode=clone:stop=4", "-frames:v", n,
               "-c:v", "libx264", "-crf", 16, "-pix_fmt", "yuv420p", "-an", out)
    elif shot.colormatch:
        r = subprocess.run([_venv("FASTVIDEO_HOME", "FastVideo"), str(ENGINES / "colormatch.py"), str(still),
                            str(raw), str(out)], capture_output=True, text=True)
        if r.returncode != 0:
            raise StageError(f"colour-match failed for {shot.id}: {r.stderr.strip()[-300:]}")
    else:
        shutil.copyfile(raw, out)
    secs = media_seconds(out)
    if abs(secs - n / FPS) > 0.06:
        raise StageError(f"shot {shot.id} is {secs:.2f} s, expected {n / FPS:.2f} s")
    keys.record(out, k)
    take = take_path(story, shot, shot.video_seed)   # every take is kept, to go back to
    take.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(out, take)
    keys.record(take, k)
    trim_clip(story, shot, keys)
    return True


def screen_on_still(story: Story, shot: Shot) -> Path:
    """A still shot's screen picture warped onto its picture, so the glide moves it with the scene."""
    still, sc = still_path(story, shot), shot.screen
    out = story.root / "shots" / "screens" / f"{shot.id}.png"
    w, h = video_size(still)
    spec = {"size": [w, h], "items": [{"picture": str(sc.picture), "out": str(out), "top": sc.top,
                                       "corners": [[x * w, y * h] for x, y in sc.corners]}]}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(spec, indent=1))
    r = subprocess.run([_venv("FASTVIDEO_HOME", "FastVideo"), str(ENGINES / "screen.py"), str(out.with_suffix(".json"))],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise StageError(f"laying the screen picture on {shot.id} failed: {r.stderr.strip()[-300:]}")
    return out


def glide_shot(story: Story, shot: Shot, keys: Keys, k: str, n: int) -> None:
    """A still shot: a slow camera move over its picture at the finished size; the video model doesn't run."""
    still, out = still_path(story, shot), full_path(story, shot)
    _say(f"shot {shot.id} ({n / FPS:.2f} s, a {shot.camera} over the picture) …")
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [_venv("FASTVIDEO_HOME", "FastVideo"), ENGINES / "kenburns.py", still, out, *FINISH_SIZE, n, shot.camera]
    if shot.screen:
        cmd.append(screen_on_still(story, shot))
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if r.returncode != 0:
        raise StageError(f"the glide over {shot.id} failed: {r.stderr.strip()[-300:]}")
    secs = media_seconds(out)
    if abs(secs - n / FPS) > 0.06:
        raise StageError(f"shot {shot.id} is {secs:.2f} s, expected {n / FPS:.2f} s")
    keys.record(out, k)
    trim_clip(story, shot, keys)


def finish_clip(story: Story, shot: Shot, keys: Keys) -> Path:
    """The shot's clip at the finished size: as it is when it was animated that big, else upscaled once."""
    clip = shot_path(story, shot)
    if video_size(clip)[0] >= FINISH_SIZE[0]:
        return clip
    out = story.root / "shots" / "hd" / f"{shot.id}.mp4"
    k = key_for("upscale", file_digest(clip), UPSCALER, FINISH_SIZE)
    if keys.stale(out, k):
        _say(f"upscale {shot.id} …")
        out.parent.mkdir(parents=True, exist_ok=True)
        heavy("upscale", [_venv("UPSCALE_HOME", "upscale"), ENGINES / "upscale.py", _home("UPSCALE_HOME", "upscale") /
                          "models" / UPSCALER, clip, out, *FINISH_SIZE], out.with_suffix(".log"))
        keys.record(out, k)
    return out


def build_picture(story: Story, keys: Keys) -> Path:
    out = story.root / "film" / "picture.mp4"
    shots = [finish_clip(story, s, keys) for s in story.shots]
    k = key_for("picture", [file_digest(p) for p in shots])
    if keys.stale(out, k):
        out.parent.mkdir(parents=True, exist_ok=True)
        lst = out.with_suffix(".txt")
        lst.write_text("".join(f"file '{p}'\n" for p in shots))
        ffmpeg("-f", "concat", "-safe", "0", "-i", lst, "-c:v", "libx264", "-crf", 18, "-pix_fmt", "yuv420p", out)
        keys.record(out, k)
    return out


def _cover(pt: tuple[float, float], src: tuple[int, int], dst: tuple[int, int]) -> tuple[float, float]:
    """Where a pixel of `src` lands when it is scaled to cover `dst` and centre-cropped."""
    k = max(dst[0] / src[0], dst[1] / src[1])
    return pt[0] * k - (src[0] * k - dst[0]) / 2, pt[1] * k - (src[1] * k - dst[1]) / 2


def still_to_finish(pt: tuple[float, float], still: tuple[int, int], clip: tuple[int, int],
                    finish: tuple[int, int]) -> tuple[float, float]:
    """A point on the still (0-1) in the finished film's pixels: the clip crops the still, the film crops the clip."""
    return _cover(_cover((pt[0] * still[0], pt[1] * still[1]), still, clip), clip, finish)


def grade_filter(story: Story) -> str:
    """ffmpeg filters for the film's grade ("" when it is neutral)."""
    g, out = story.grade, []
    if g.saturation != 1 or g.contrast != 1:
        out.append(f"eq=saturation={g.saturation:g}:contrast={g.contrast:g}")
    if g.warmth:
        w = g.warmth
        out.append(f"colorbalance=rs={w / 2:g}:rm={w:g}:rh={w / 2:g}:bs={-w / 2:g}:bm={-w:g}:bh={-w / 2:g}")
    return ",".join(out)


def render_titles(story: Story, keys: Keys, plan: list[Title]) -> list[Path]:
    """One transparent picture per title, drawn by the title engine (this ffmpeg has no text filters)."""
    tdir = story.root / "film" / "titles"
    outs = [tdir / f"{i:02d}-{t.kind}.png" for i, t in enumerate(plan, 1)]
    spec = {"size": list(FINISH_SIZE), "font": TITLE_FONT,
            "items": [{"text": t.text, "kind": t.kind, "place": t.place, "out": str(o)}
                      for t, o in zip(plan, outs)]}
    k = key_for("titles", spec)
    if any(keys.stale(o, k) for o in outs):
        _say(f"titles ({len(plan)}) …")
        tdir.mkdir(parents=True, exist_ok=True)
        (tdir / "spec.json").write_text(json.dumps(spec, indent=1))
        r = subprocess.run([_venv("MMAUDIO_HOME", "MMAudio"), str(ENGINES / "titles.py"), str(tdir / "spec.json")],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise StageError(f"drawing the titles failed: {r.stderr.strip()[-300:]}")
        for o in outs:
            keys.record(o, k)
    return outs


def render_screens(story: Story, keys: Keys) -> list[tuple[Path, float, float]]:
    """Each shot's screen picture warped onto its corners at the finished size, with the shot's time span.
    A still shot's screen is already on its picture (it moves with the glide)."""
    shots = [(s, at) for s, at in zip(story.shots, shot_starts(story)) if s.screen and s.tier != "still"]
    if not shots:
        return []
    sdir = story.root / "film" / "screens"
    outs = [sdir / f"{s.id}.png" for s, _ in shots]
    spec = {"size": list(FINISH_SIZE), "items": [
        {"picture": str(s.screen.picture), "out": str(o), "top": s.screen.top,
         "corners": [list(still_to_finish(c, STILL_SIZE, video_size(shot_path(story, s)), FINISH_SIZE))
                     for c in s.screen.corners]}
        for (s, _), o in zip(shots, outs)]}
    k = key_for("screens", spec, [file_digest(s.screen.picture) for s, _ in shots])
    if any(keys.stale(o, k) for o in outs):
        _say(f"screens ({len(shots)}) …")
        sdir.mkdir(parents=True, exist_ok=True)
        (sdir / "spec.json").write_text(json.dumps(spec, indent=1))
        r = subprocess.run([_venv("FASTVIDEO_HOME", "FastVideo"), str(ENGINES / "screen.py"), str(sdir / "spec.json")],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise StageError(f"laying the screen pictures failed: {r.stderr.strip()[-300:]}")
        for o in outs:
            keys.record(o, k)
    return [(o, at, at + shot_len(story, s)) for (s, at), o in zip(shots, outs)]


def video_size(path: Path) -> tuple[int, int]:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=width,height",
                        "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    try:
        w, h = r.stdout.strip().split(",")[:2]
        return int(w), int(h)
    except ValueError:
        raise StageError(f"cannot read the size of {path}") from None


def finish_picture(story: Story, keys: Keys, picture: Path) -> Path:
    """The picture at the finished size: graded, real screens laid on, then captions and the end card over
    it, the last frame held for the card."""
    out = story.root / "film" / "finished.mp4"
    plan = title_plan(story)
    layers = render_screens(story, keys)                            # screens first, words on top
    pngs = render_titles(story, keys, plan) if plan else []
    layers += [(png, t.start, t.end) for t, png in zip(plan, pngs)]
    hold = story.end_card.hold if story.end_card else 0.0
    k = key_for("finish", file_digest(picture), [(file_digest(p), a, b) for p, a, b in layers], hold, FINISH_SIZE,
                grade_filter(story))
    if keys.stale(out, k):
        _say("finish: size" + (", screens and titles" if layers else "") + " …")
        w, h = FINISH_SIZE
        g = f"[0:v]scale={w}:{h}:force_original_aspect_ratio=increase:flags=lanczos,crop={w}:{h},setsar=1"
        g += f",{grade_filter(story)}" if grade_filter(story) else ""
        g += (f",tpad=stop_mode=clone:stop_duration={hold:g}" if hold else "") + "[v0];"
        ins = ["-i", picture]
        for i, (png, a, b) in enumerate(layers, 1):
            ins += ["-loop", 1, "-framerate", FPS, "-t", f"{b - a:.3f}", "-i", png]
            fade = "fade=in:st=0:d=0.2:alpha=1," if png.parent.name == "titles" else ""
            g += (f"[{i}:v]format=rgba,{fade}setpts=PTS-STARTPTS+{a:.3f}/TB[t{i}];"
                  f"[v{i - 1}][t{i}]overlay=eof_action=pass:format=auto[v{i}];")
        g = g.rstrip(";")
        ffmpeg(*ins, "-filter_complex", g, "-map", f"[v{len(layers)}]", "-c:v", "libx264", "-crf", 17,
               "-pix_fmt", "yuv420p", "-r", FPS, out)
        keys.record(out, k)
    return out


def sfx_path(story: Story, shot: Shot) -> Path:
    return story.root / "sfx" / f"{shot.id}.flac"


def sfx_key(story: Story, shot: Shot) -> str | None:
    clip = shot_path(story, shot)
    return key_for("sfx", file_digest(clip), shot.sfx, story.sfx_negative) if clip.exists() else None


def sound_shot(story: Story, shot: Shot, keys: Keys) -> Path:
    clip, out = shot_path(story, shot), sfx_path(story, shot)
    k = sfx_key(story, shot)
    if keys.stale(out, k):
        _say(f"effects {shot.id} …")
        heavy("sfx", [_venv("MMAUDIO_HOME", "MMAudio"), "demo.py", "--video", clip, "--duration", f"{shot_len(story, shot):.6f}",
                      "--prompt", shot.sfx, "--negative_prompt", story.sfx_negative, "--skip_video_composite",
                      "--output", out.parent], out.with_suffix(".log"), cwd=_home("MMAUDIO_HOME", "MMAudio"))
        keys.record(out, k)
    return out


def build_soundtrack(story: Story, tracks: list[Path], keys: Keys) -> Path:
    """Pad or trim every shot's effects to the exact shot length so the track never drifts off the cuts."""
    out = story.root / "film" / "sfx.wav"
    lens = [shot_len(story, s) for s in story.shots]
    k = key_for("soundtrack", [file_digest(t) for t in tracks], *([lens] if story.pace == "voice" else []))
    if keys.stale(out, k):
        ins = [a for t in tracks for a in ("-i", t)]
        g = "".join(f"[{i}:a]apad,atrim=0:{L:.6f}[a{i}];" for i, L in enumerate(lens))
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


def speech_cuts(silences: list[tuple[float, float]], total: float, pause: float = 0.2, before: float = 0.05,
                after: float = 0.2) -> list[tuple[float, float]]:
    """The parts of a line to keep: its words, `before` ahead of the first, `after` past the last, and every
    pause inside it cut to at most `pause` (half kept on each side)."""
    words, at = [], 0.0
    for a, b in sorted(silences):
        if a > at:
            words.append((at, a))
        at = max(at, b)
    if at < total:
        words.append((at, total))
    if not words:
        return [(0.0, total)]
    keep = [[max(0.0, words[0][0] - before), words[0][1]]]
    for a, b in words[1:]:
        gap = a - keep[-1][1]
        if gap <= pause:
            keep[-1][1] = b
        else:
            keep[-1][1] += pause / 2
            keep.append([a - pause / 2, b])
    keep[-1][1] = min(total, keep[-1][1] + after)
    return [(round(a, 6), round(b, 6)) for a, b in keep]


def trim_speech(wav: Path) -> None:
    """A line cut to its words: the voice model's lead-in silence gone, long pauses (and the end) cut to 0.2 s.
    Voice-paced shots last as long as their lines, so dead air there is dead air in the film."""
    r = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(wav), "-af", "silencedetect=n=-40dB:d=0.15", "-f", "null", "-"],
                       capture_output=True, text=True)
    starts = [float(x) for x in re.findall(r"silence_start: (-?[\d.]+)", r.stderr)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", r.stderr)]
    total = media_seconds(wav)
    keep = speech_cuts(list(zip(starts, ends + [total] * (len(starts) - len(ends)))), total)
    if keep == [(0.0, total)]:
        return
    pick = "+".join(f"between(t,{a:.4f},{b:.4f})" for a, b in keep)
    tmp = wav.with_name(f".{wav.stem}-trim.wav")
    ffmpeg("-i", wav, "-af", f"aselect='{pick}',asetpts=N/SR/TB", tmp)
    tmp.replace(wav)


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


def narrator_card_key(story: Story) -> str:
    return key_for("card", story.narrator.voice, story.narrator.sample)


def narration_key(story: Story, shot: Shot) -> str | None:
    card = story.root / "voice" / "card.wav"
    return key_for("line", file_digest(card), story.narrator.sample, shot.narration) if card.exists() else None


def character_card_key(story: Story, name: str) -> str:
    v = story.voices[name]
    return key_for("card", v.voice, v.sample)


def dialogue_key(story: Story, line) -> str | None:
    """A cloned dialogue line's key (clone mode); None while the speaker has no voice yet."""
    card = story.root / "voice" / "cards" / f"{line.speaker}.wav"
    return key_for("dline", file_digest(card), story.voices[line.speaker].sample, line.line) if card.exists() else None


def narrate(story: Story, keys: Keys, checks: dict, only: set[str] | None = None) -> list[tuple[Path, float]]:
    """Design the narrator's voice once, clone it for every line, and check each line is intelligible.
    `only` limits it to those shots (one scene at a time)."""
    offsets = [(sid, at) for sid, at in narration_offsets(story) if only is None or sid in only]
    if not offsets:
        return []
    vdir = story.root / "voice"
    vdir.mkdir(exist_ok=True)
    n = story.narrator
    card = vdir / "card.wav"
    k = narrator_card_key(story)
    if keys.stale(card, k):
        _say("narrator voice …")
        tts(TTS_DESIGN, n.sample, card, ["--instruct", n.voice])
        keys.record(card, k)
    checks["narration"] = {}
    for sid, _ in offsets:
        shot = next(s for s in story.shots if s.id == sid)
        out = vdir / f"{sid}.wav"
        k = narration_key(story, shot)
        if keys.stale(out, k):
            _say(f"narration {sid} …")
            tts(TTS_CLONE, shot.narration, out, ["--ref_audio", card, "--ref_text", n.sample])
            trim_speech(out)
            keys.record(out, k)
        checks["narration"][sid] = round(words_match(shot.narration, heard(out, keys)), 2)
    # placed only now: in a voice-paced film the recordings set the shot lengths, and so where each line starts
    return [(vdir / f"{sid}.wav", at) for sid, at in narration_offsets(story) if only is None or sid in only]


def voice_cards(story: Story, keys: Keys, only: set[str] | None = None) -> dict[str, Path]:
    """One designed voice per speaking character, made once and reused for every line."""
    cdir = story.root / "voice" / "cards"
    cdir.mkdir(parents=True, exist_ok=True)
    cards = {}
    for name in dict.fromkeys(l.speaker for sid, l in dialogue_lines(story) if only is None or sid in only):
        v, card = story.voices[name], cdir / f"{name}.wav"
        k = character_card_key(story, name)
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


def speak_dialogue(story: Story, keys: Keys, checks: dict, only: set[str] | None = None) -> list[tuple[Path, float]]:
    """Voice every dialogue line, check it is intelligible, and place it in its shot.
    `only` limits it to those shots (one scene at a time; scene mode still performs them together)."""
    said = dialogue_lines(story)
    if only is not None and story.dialogue_mode != "scene":
        said = [(sid, l) for sid, l in said if sid in only]
    if not said:
        return []
    cards = voice_cards(story, keys, None if story.dialogue_mode == "scene" else only)
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
            k = dialogue_key(story, l)
            if keys.stale(o, k):
                _say(f"dialogue {o.stem} …")
                tts(TTS_CLONE, l.line, o, ["--ref_audio", cards[l.speaker], "--ref_text", story.voices[l.speaker].sample])
                trim_speech(o)
                keys.record(o, k)
    checks["dialogue"], checks["dialogue_overflow"] = {}, {}
    placed = []
    index = {s.id: i for i, s in enumerate(story.shots)}
    for sid in dict.fromkeys(sid for sid, *_ in outs):
        mine = [(l, o) for s2, l, o in outs if s2 == sid]
        narr = story.root / "voice" / f"{sid}.wav"
        lead = dialogue_lead(media_seconds(narr) if story.shots[index[sid]].narration and narr.exists() else None,
                             story.narrator.lead if story.narrator else 0.0)
        shot = story.shots[index[sid]]
        starts, over = place_lines(shot_starts(story)[index[sid]], [media_seconds(o) for _, o in mine], lead=lead,
                                   length=shot_len(story, shot))
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
    bpm = score_bpm(shot_seconds()) if story.pace != "voice" else min(140, max(70, score_bpm(total / len(story.shots))))
    spec = {"out": str(sdir), "duration": round(total + 1.2, 2), "bpm": bpm, "key": m.key,
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
        rspec.write_text(json.dumps({"shot_len": shot_seconds(), "starts": shot_starts(story), "total": total,
                                     "arc": intensity_arc(story), "hit": hit_time(story), "bpm": spec["bpm"]}))
        r = subprocess.run([_venv("MMAUDIO_HOME", "MMAudio"), str(ENGINES / "score_rank.py"), str(rspec), str(rank),
                            *map(str, takes)], capture_output=True, text=True)
        if r.returncode != 0:
            raise StageError(f"score ranking failed: {r.stderr.strip()[-300:]}")
        keys.record(rank, rk)
    ranking = json.loads(rank.read_text()) if len(takes) > 1 else [{"take": str(takes[0]), "score": None}]
    chosen = Path(ranking[0]["take"])
    checks["score"] = {"chosen": str(chosen), "ranking": ranking}
    return chosen


def mix_gain(ins: list, starts: list[int], total: float, sfx_db: float, music_db: float) -> float:
    """The gain (dB) that normalizing the plain mix to -16 LUFS would apply: 100% on every layer
    then sounds exactly as loud as films did before levels existed."""
    g = mix_filtergraph(starts, total, sfx_db, music_db).replace("loudnorm=I=-16:TP=-1.5", "loudnorm=I=-16:TP=-1.5:print_format=json")
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", *map(str, ins), "-filter_complex", g, "-map", "[out]",
                        "-f", "null", "-"], capture_output=True, text=True)
    m = re.search(r'"input_i"\s*:\s*"(-?[\d.]+|-inf)"', r.stderr)
    if r.returncode != 0 or not m:
        raise StageError(f"could not measure the mix: {r.stderr.strip()[-400:]}")
    return 0.0 if m[1] == "-inf" else round(-16 - float(m[1]), 2)


def mix(story: Story, keys: Keys, picture: Path, soundtrack: Path, music: Path | None,
        lines: list[tuple[Path, float]], total: float) -> Path:
    out = story.root / "film" / f"{story.slug}.mp4"
    mdb = story.music.db if story.music else 0
    lv = story.levels
    k = key_for("mix", file_digest(picture), file_digest(soundtrack), music and file_digest(music),
                [(file_digest(p), at) for p, at in lines], story.sfx_db, mdb, lv.sound, lv.music, lv.voices)
    if keys.stale(out, k):
        _say("mix …")
        music_in = ["-i", music] if music else ["-f", "lavfi", "-t", f"{total:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
        ins = ["-i", picture, "-i", soundtrack, *music_in]
        for p, _ in lines:
            ins += ["-i", p]
        starts, total = [round(at * 1000) for _, at in lines], round(total, 4)
        g = mix_filtergraph(starts, total, story.sfx_db, mdb, lv, mix_gain(ins, starts, total, story.sfx_db, mdb))
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
        f"<li><span>{at // 60:.0f}:{at % 60:04.1f}</span>"
        f"<b>{esc(s.id)}</b> {esc(_beat(s))}</li>"
        for s, at in zip(story.shots, shot_starts(story)))
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


def _stills_ready(story: Story, keys: Keys) -> bool:
    missing = [s.id for s in story.shots if keys.stale(still_path(story, s), still_key(story, s))]
    if missing:
        print(f"stills missing or out of date for {', '.join(missing)}: run `spike-animate board` first",
              file=sys.stderr)
    return not missing


def _lines_ready(story: Story) -> bool:
    missing = [s.id for s in story.shots if shot_frames(story, s) is None]
    if missing:
        print(f"lines not recorded for {', '.join(missing)}: shots last as long as their lines, "
              "so run `spike-animate voices` first", file=sys.stderr)
    return not missing


def save_checks(story: Story, part: dict, merge: bool = False) -> None:
    """Merge one step's checks into film/checks.json, so steps can run one at a time. With merge,
    per-shot tables are updated entry by entry (one scene's results keep the other scenes')."""
    f = story.root / "film" / "checks.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    try:
        checks = json.loads(f.read_text())
    except (OSError, ValueError):
        checks = {}
    for k, v in part.items():
        checks[k] = {**checks.get(k, {}), **v} if merge and isinstance(v, dict) else v
    f.write_text(json.dumps(checks, indent=1))


def film_seconds(story: Story) -> float:
    """The picture's length: the shots butt-joined (what build_picture produces)."""
    picture = story.root / "film" / "picture.mp4"
    hold = story.end_card.hold if story.end_card else 0.0
    return media_seconds(picture) + hold if picture.exists() else film_length(story)


def _scene(story: Story, shot_id: str | None) -> list[Shot] | None:
    """The shots a step works on: one scene, or all of them. None (after saying why) for an unknown id."""
    if shot_id is None:
        return story.shots
    shot = next((s for s in story.shots if s.id == shot_id), None)
    if shot is None:
        print(f"no shot '{shot_id}' in the story", file=sys.stderr)
    return [shot] if shot else None


def cmd_video(story: Story, shot_id: str | None = None) -> int:
    keys = Keys(story.root)
    shots = _scene(story, shot_id)
    if shots is None:
        return 2
    if shot_id:
        if keys.stale(still_path(story, shots[0]), still_key(story, shots[0])):
            print(f"no still for {shot_id} yet: run `spike-animate board --shot {shot_id}` first", file=sys.stderr)
            return 2
        if shot_frames(story, shots[0]) is None:
            print(f"{shot_id} lasts as long as its lines: record them first (`spike-animate voices --shot {shot_id}`)",
                  file=sys.stderr)
            return 2
        animate_shot(story, shots[0], keys)
        check_motions(story, shots, keys)
        _say(f"Video: {shot_path(story, shots[0])}")
        return 0
    if not _stills_ready(story, keys) or not _lines_ready(story):
        return 2
    for shot in story.shots:
        animate_shot(story, shot, keys)
    check_motions(story, story.shots, keys)
    picture = build_picture(story, keys)
    _say(f"Video: {picture} ({media_seconds(picture):.1f} s, no sound yet)")
    return 0


def cmd_sound(story: Story, shot_id: str | None = None) -> int:
    keys = Keys(story.root)
    shots = _scene(story, shot_id)
    if shots is None:
        return 2
    missing = [s.id for s in shots if not shot_path(story, s).exists()]
    if missing:
        print(f"no video yet for {', '.join(missing)}: run `spike-animate video` first", file=sys.stderr)
        return 2
    if shot_id:
        _say(f"Sound effects: {sound_shot(story, shots[0], keys)}")
        return 0
    out = build_soundtrack(story, [sound_shot(story, s, keys) for s in story.shots], keys)
    _say(f"Sound effects: {out}")
    return 0


def cmd_voices(story: Story, shot_id: str | None = None) -> int:
    keys, checks = Keys(story.root), {}
    shots = _scene(story, shot_id)
    if shots is None:
        return 2
    only = {shot_id} if shot_id else None
    lines = narrate(story, keys, checks, only) + speak_dialogue(story, keys, checks, only)
    save_checks(story, checks, merge=only is not None)
    _say(f"Voices: {len(lines)} line(s) recorded")
    return 0


def cmd_music(story: Story) -> int:
    keys, checks = Keys(story.root), {}
    chosen = score(story, keys, checks, film_seconds(story))
    save_checks(story, checks)
    _say(f"Music: {chosen}" if chosen else "Music: the story has no [music]")
    return 0


def scene_path(story: Story, shot: Shot) -> Path:
    return story.root / "scenes" / f"{shot.id}.mp4"


def scene_lines(story: Story, shot: Shot, keys: Keys) -> list[tuple[Path, float]]:
    """This scene's narration and dialogue, each with its start time inside the shot."""
    checks: dict = {}
    start = shot_starts(story)[story.shots.index(shot)]
    lines = narrate(story, keys, checks, {shot.id}) + speak_dialogue(story, keys, checks, {shot.id})
    return [(p, round(at - start, 3)) for p, at in lines if p.stem.startswith(shot.id)]


def cmd_scene(story: Story, shot_id: str) -> int:
    """Preview one scene: its clip with its own sound effects and lines, at the film's levels (no music)."""
    keys = Keys(story.root)
    shots = _scene(story, shot_id)
    if shots is None:
        return 2
    shot = shots[0]
    clip = shot_path(story, shot)
    if not clip.exists():
        print(f"no video for {shot.id} yet: run `spike-animate video --shot {shot.id}` first", file=sys.stderr)
        return 2
    sfx = story.root / "sfx" / f"{shot.id}.flac"
    lines = scene_lines(story, shot, keys)
    out, L = scene_path(story, shot), shot_len(story, shot)
    lv = story.levels
    k = key_for("scene", file_digest(clip), sfx.exists() and file_digest(sfx), [(file_digest(p), at) for p, at in lines],
                story.sfx_db, lv.sound, lv.voices)
    if keys.stale(out, k):
        out.parent.mkdir(parents=True, exist_ok=True)
        bed = out.with_name(f".{shot.id}-sfx.wav")   # normalized like the film's soundtrack
        src = ["-i", sfx] if sfx.exists() else ["-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono"]
        ffmpeg(*src, "-af", f"apad,atrim=0:{L:.6f}" + (",loudnorm=I=-16:TP=-1.5" if sfx.exists() else ""),
               "-ar", 44100, "-c:a", "pcm_s16le", bed)
        ins = ["-i", clip, "-i", bed, "-f", "lavfi", "-t", f"{L:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
        for p, _ in lines:
            ins += ["-i", p]
        starts = [max(0, round(at * 1000)) for _, at in lines]
        g = mix_filtergraph(starts, round(L, 4), story.sfx_db, 0, lv, mix_gain(ins, starts, round(L, 4), story.sfx_db, 0))
        ffmpeg(*ins, "-filter_complex", g, "-map", "0:v", "-map", "[out]", "-c:v", "copy", "-c:a", "aac",
               "-b:a", "192k", "-t", f"{L:.6f}", "-movflags", "+faststart", out)
        bed.unlink(missing_ok=True)
        keys.record(out, k)
    _say(f"Scene: {out}")
    return 0


def cmd_film(story: Story) -> int:
    keys = Keys(story.root)
    if not _stills_ready(story, keys):
        return 2
    checks: dict = {}
    lines = narrate(story, keys, checks) + speak_dialogue(story, keys, checks)   # first: they can set the shot lengths
    for shot in story.shots:
        animate_shot(story, shot, keys)
    check_motions(story, story.shots, keys)
    picture = finish_picture(story, keys, build_picture(story, keys))
    total = media_seconds(picture)
    soundtrack = build_soundtrack(story, [sound_shot(story, s, keys) for s in story.shots], keys)
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
    "required": ["title", "look", "still_style", "lighting", "characters", "places", "angles", "narrator", "music",
                 "end_card", "shots"],
    "properties": {
        "title": {"type": "string"},
        "look": {"type": "string"},
        "still_style": {"type": "string"},
        "lighting": {"type": "string"},
        "end_card": {"type": "object", "required": ["text", "sub"],
                     "properties": {"text": {"type": "string"}, "sub": {"type": "string"}}},
        "places": {"type": "array", "items": {"type": "object", "required": ["name", "description"],
                   "properties": {"name": {"type": "string"}, "description": {"type": "string"}}}},
        "angles": {"type": "array", "items": {"type": "object", "required": ["name", "frame"],
                   "properties": {"name": {"type": "string"}, "frame": {"type": "string"}}}},
        "characters": {"type": "array", "items": {"type": "object", "required": ["name", "description", "voice"],
                       "properties": {"name": {"type": "string"}, "description": {"type": "string"},
                                      "voice": {"type": "string"}}}},
        "narrator": {"type": "object", "required": ["voice", "sample"],
                     "properties": {"voice": {"type": "string"}, "sample": {"type": "string"}}},
        "music": {"type": "object", "required": ["caption", "structure", "key"],
                  "properties": {"caption": {"type": "string"}, "structure": {"type": "string"}, "key": {"type": "string"}}},
        "shots": {"type": "array", "items": {"type": "object",
                  "required": ["id", "angle", "still", "motion", "sfx", "narration", "caption", "intensity", "action",
                               "tier", "camera", "lighting"],
                  "properties": {"id": {"type": "string"}, "angle": {"type": "string"}, "caption": {"type": "string"},
                                 "still": {"type": "string"}, "motion": {"type": "string"},
                                 "sfx": {"type": "string"}, "narration": {"type": "string"},
                                 "intensity": {"type": "number"}, "action": {"type": "boolean"},
                                 "tier": {"type": "string", "enum": list(TIERS)},
                                 "camera": {"type": "string", "enum": list(CAMERAS)},
                                 "lighting": {"type": "string"},
                                 "dialogue": {"type": "array", "items": {"type": "object",
                                              "required": ["speaker", "line", "emotion"],
                                              "properties": {"speaker": {"type": "string"}, "line": {"type": "string"},
                                                             "emotion": {"type": "string"}}}}}}},
    },
}

@dataclass
class Parts:
    """Which optional parts the author wants in the film."""
    narrator: bool = True
    dialogue: bool = True
    music: bool = True


def draft_schema(parts: Parts = Parts()) -> dict:
    """DRAFT_SCHEMA without the parts the author left out."""
    s = json.loads(json.dumps(DRAFT_SCHEMA))
    for name, on in (("narrator", parts.narrator), ("music", parts.music)):
        if not on:
            s["required"].remove(name)
            del s["properties"][name]
    if not parts.dialogue:
        s["properties"]["characters"]["items"]["required"].remove("voice")
    return s


MAX_NARRATION_WORDS = 10   # ~3 s of narration: a voice-paced shot is its line plus a breath, at most 5 s
MAX_DIALOGUE_WORDS = 12    # all the dialogue in one shot, for the same reason
MAX_CAPTION_WORDS = 6      # on-screen words: read at a glance
DRAFT_LEAD = 0.2           # the narrator comes in right after a cut: an ad's voice-over runs on
MAX_FULL_SHOTS = 3         # full motion is busy to watch: keep it for the gestures and the action

WRITER_EXAMPLE = {
    "title": "Fox Hunt",
    "look": "2D cartoon animation, soft muted storybook colors, dusky teal pine trees, soft white snow, gentle shading",
    "still_style": "2D cartoon animation still, bold black outlines, flat vibrant colors, simple shapes, "
                   "Saturday-morning cartoon style, snowy pine forest",
    "lighting": "winter dawn, low pale pink sunlight from the right, long blue shadows on the snow",
    "places": [{"name": "CLEARING", "description": "a small snowy clearing ringed by dark teal pine trees, a fallen "
                                                  "log half buried in snow on one side"}],
    "angles": [{"name": "WIDE", "frame": "wide shot of the clearing from its edge, pines framing both sides"},
               {"name": "LOG", "frame": "medium shot from low behind the fallen log, looking across the clearing"},
               {"name": "CLOSE", "frame": "close-up at eye level, the snowy pines blurred behind"}],
    "characters": [{"name": "FOX", "description": "a slender red fox with bright orange fur, white chest and muzzle, "
                                                  "black legs, bushy tail with a white tip, amber eyes, a frayed "
                                                  "green scarf around its neck",
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
    "end_card": {"text": "", "sub": ""},
    "shots": [
        {"id": "01-establish", "angle": "WIDE", "still": "FOX standing at the edge of the CLEARING, looking out",
         "motion": "static camera, a slender red fox stands still at the edge of the snowy pine forest at dawn, mouth closed",
         "tier": "still", "camera": "push", "lighting": "",
         "sfx": "soft paw footsteps crunching in snow, quiet winter wind", "narration":
         "Deep in the winter woods, the fox was hungry.", "caption": "The fox was hungry.", "intensity": 0.15, "action": False, "dialogue": []},
        {"id": "02-reveal", "angle": "LOG", "still": "RAB nibbling grass in the CLEARING, unaware",
         "motion": "static camera, a brown rabbit nibbles grass, chewing, ears twitching",
         "sfx": "rabbit nibbling, soft rustling, quiet birdsong", "narration": "Breakfast. And it had no idea.",
         "caption": "Breakfast.", "intensity": 0.3, "action": False, "tier": "subtle", "camera": "push", "lighting": "", "dialogue": []},
        {"id": "03-taunt", "angle": "CLOSE", "still": "FOX crouched behind the log in the CLEARING, grinning",
         "motion": "static camera, a red fox grins, licks its lips and talks quietly, eyes fixed ahead",
         "sfx": "soft wind, a twig creaking", "narration": "", "caption": "",
         "dialogue": [{"speaker": "FOX", "line": "Good morning, little breakfast.", "emotion": "sly, purring"}],
         "intensity": 0.6, "action": False, "tier": "subtle", "camera": "push", "lighting": ""},
        {"id": "03-pounce", "angle": "WIDE", "still": "FOX leaping through the air to pounce on RAB in the CLEARING, snow spraying",
         "motion": "a red fox leaps through the air and pounces on a brown rabbit, snow spraying, fast dynamic action",
         "sfx": "fast whoosh of a leap, paws thudding into snow, rabbit squeal", "narration": "", "caption": "",
         "intensity": 1.0, "action": True, "tier": "full", "camera": "push", "lighting": "", "dialogue": []},
    ],
}


# The writer's rules for the optional parts, used when the author wants that part.
NARRATION_RULE = '- The voice-over carries the film: give most shots a "narration" line of at most {max_words} words,\n  short and punchy like an ad or a trailer ("She built the agent."). Lines follow on from each other;\n  a shot may stay silent ("") for a beat. The last shot holds the closing line.\n'
DIALOGUE_RULE = '- Characters may talk if the idea calls for it. Give every character who speaks a "voice": how they sound\n  (pitch, pace, texture, personality), e.g. "small, bright, chirpy robot voice, fast and excitable"; leave\n  "voice" "" for characters who never speak. Put spoken lines in a shot\'s "dialogue" list as\n  {{"speaker": NAME, "line": ..., "emotion": ...}}. Keep one speaker per shot and cut between speakers\n  (shot / reverse shot); at most {max_words} words of speech per shot, counting narration and dialogue\n  together (the narrator speaks first, then the character). Frame talking shots as medium shots or over the shoulder and say "talking" in the motion.\n  Reaction shots of the listener, with no dialogue, make conversations read well. Use "dialogue": [] otherwise.\n'
MUSIC_RULE = '- "music" describes an instrumental score that follows the intensity arc; "structure" is 3-6 section tags\n  like "[Intro - ...]" separated by blank lines; "key" like "D minor" or "C major".\n'


LOOKS = ("keyart", "flat")
KEYART_LOOK = "anime key art, richly detailed backgrounds, cinematic lighting, vibrant saturated colors, clean linework"
KEYART_STILL = ("anime key visual illustration, highly detailed background, cinematic lighting, rim light, depth, "
                "painterly sky, vibrant saturated colors, clean confident linework")


def _example(parts: Parts, look: str = "keyart") -> dict:
    ex = json.loads(json.dumps(WRITER_EXAMPLE))
    if look == "keyart":
        ex["look"] = f"{KEYART_LOOK}, dusky teal pine trees, soft white snow"
        ex["still_style"] = f"{KEYART_STILL}, snowy pine forest"
    for name, on in (("narrator", parts.narrator), ("music", parts.music)):
        if not on:
            del ex[name]
    for s in ex["shots"]:
        if not parts.narrator:
            s["narration"] = ""
        if not parts.dialogue:
            s["dialogue"] = []
            s["motion"] = s["motion"].replace("grins and talks quietly", "grins silently, mouth closed")
    if not parts.dialogue:
        for c in ex["characters"]:
            c["voice"] = ""
    return ex


def writer_prompt(idea: str, shots: int, problems: list[str] | None = None, previous: dict | None = None,
                  notes: str = "", style: str = "", parts: Parts = Parts(), look: str = "keyart") -> str:
    ref = f"\nREFERENCE NOTES from the author (follow them: names, looks, setting, tone, events):\n{notes.strip()}\n" if notes.strip() else ""
    if style.strip():
        ref += (f"\nVISUAL STYLE, read from the author's reference picture:\n{style.strip()}\n"
                "Write \"look\" and \"still_style\" in this style (keep its medium, palette, linework and texture words), "
                "not the default cartoon style; end \"still_style\" with the setting.\n")
        looks = '- "look" and "still_style" follow the VISUAL STYLE above.'
    elif look == "flat":
        looks = ('- "look" starts with "2D cartoon animation," then the palette and setting. "still_style" starts with\n'
                 '  "2D cartoon animation still, bold black outlines, flat vibrant colors," then the setting.')
    else:
        looks = (f'- "look" starts with "{KEYART_LOOK}," then the palette and setting. "still_style" starts with\n'
                 f'  "{KEYART_STILL}," then the setting. Every picture is a finished illustration, full of detail.')
    narration_rule = (NARRATION_RULE.format(max_words=MAX_NARRATION_WORDS) if parts.narrator else
                      '- There is no narrator: leave every "narration" "".\n')
    dialogue_rule = (DIALOGUE_RULE.format(max_words=MAX_DIALOGUE_WORDS) if parts.dialogue else
                     '- No one speaks in this film: every character\'s "voice" is "" and every "dialogue" is []. Tell the\n'
                     '  story through what we see and hear.\n')
    music_rule = MUSIC_RULE if parts.music else '- There is no score: leave out "music".\n'
    p = f"""You are the story artist for a short 2D cartoon. Turn the idea below into a shot list of exactly {shots} shots.

IDEA: {idea}
(If the author wrote a whole story rather than an idea, follow its events in order, keep its characters and
names, and use the author's own dialogue lines word for word where they fit; split a long line across shots.)
{ref}
How the film is made, so write for it:
- Every shot is one short clip animated from one storyboard still, as long as its line plus a breath:
  2 to 5 seconds (a shot with nothing said lasts 3 s). The whole film runs about 15 to 20 seconds, so every
  shot must earn its place. One clear action per shot.
- Stills are rendered by an image model, so each still is a concrete visual description: framing (wide shot,
  close-up, low angle, point-of-view), who, doing what, where, lighting. No camera moves, no time passing.
- Characters keep the same look only if they are described with the same words every time. So give each
  character a name that is one UPPERCASE word, letters only (FOX, KEEPER, BOLT), and a full visual
  description once, in "characters".
  In every still, write the UPPERCASE name, never re-describe the character. The description includes the
  exact clothes and accessories (colours, fabrics, fit), so the outfit stays the same in every shot.
- "lighting" is one line used in every picture and every motion: time of day, where the light comes from,
  its colour, the shadows. A shot at another time of day (the night turns to dawn) gives its own "lighting"
  line the same way; every other shot leaves its "lighting" "" and keeps the film's.
- "places" names each location once, one UPPERCASE word (TERRACE, CLEARING), with a full visual description
  and its set dressing: 4 to 6 concrete details that fill the frame (string lights, palm trees, thatched
  umbrellas, potted flowers), so every shot there looks rich and the same.
  Write the place's NAME in every still set there, never re-describe it.
- "angles" are 3 or 4 camera setups, each one UPPERCASE word with a "frame" (shot size and where the camera
  stands, e.g. "medium shot across the cafe table, the sea behind"). Make them really different: a wide
  shot, a medium shot, a close-up, a low or high angle, from different sides of the place. Every shot picks
  one "angle", and never two shots in a row from the same angle: each cut shows a new view. The still then
  says who and what, not the framing again.
- "motion" says what moves during the shot, in plain words without the UPPERCASE names; start calm shots
  with "static camera,". Set "action": true only for fast, violent or acrobatic shots.
- "tier" is how much the shot moves. "still": nothing needs to move (a place, a product, a screen, a pose,
  the closing shot); the camera glides slowly over the finished picture, and "camera" says how: "push" in,
  "pull" out, or a pan "left" or "right". "subtle": people are there but barely move (look, smile, breathe,
  sip, talk). "full": real action. Use "full", with "static camera," for a shot whose gesture is the point;
  at most {MAX_FULL_SHOTS} shots "full".
  Give every shot with a person one clear gesture or use of a prop that a viewer sees at once: she throws
  both fists in the air, sips her coffee, leans back and stretches, spins her chair, points at the screen.
  Never just "looks" or "thinks"; vary the gesture from shot to shot.
  A shot of a screen (a laptop, a phone, a monitor) showing a web page or an app is "still", and its still
  shows "its bright blank white screen": the real page is laid onto the screen afterwards, so the words
  stay readable. Never ask the image model to draw words on a screen.
  Give every shot a "camera" ("push" when it isn't still).
  The motion moves only what the still shows: never bring in a prop, person or action the still doesn't
  contain (the video model draws whatever the words suggest, so "as she reads" grows a book; write "her eyes
  follow the laptop screen"). A character who doesn't speak in the shot keeps their mouth closed: add
  "mouth closed" to the motion, or the video model animates their lips.
- Anyone sitting at a table is framed from the waist up, the table's edge in the foreground; never show
  their legs with the table's or chairs' legs, which the image model tangles. Say what they sit on and where
  the table is: "ELENA sitting on a chair behind the table, seen from the waist up".
- "sfx" lists the sounds of that shot (no music, no speech).
- "caption" is the words burned in on screen over the shot: at most {MAX_CAPTION_WORDS} words, a shorter, punchier echo of
  its line ("Built the agent.", "Now make it earn."), or "" for none.
- "end_card" is the closing words over the last shot, e.g. for an ad the tagline and the website:
  {{"text": "Put your agent to work.", "sub": "BotGuild.ai"}}. Use {{"text": "", "sub": ""}} when the film needs none.
{narration_rule}{dialogue_rule}- "intensity" is the story's energy in that shot, from 0 (calm) to 1 (peak). Build to one peak near the end.
{looks}
{music_rule}- Shot ids look like 01-name, 02-name, ... in order.

Example of the format and the level of detail (a different film, 4 shots; FOX talks, RAB does not):
{json.dumps(_example(parts, look), indent=1)}

Write the story for the IDEA as one JSON object in the same format, with exactly {shots} shots."""
    if problems:
        if previous is not None:
            p += "\n\nYour previous draft:\n" + json.dumps(previous, indent=1)
        p += "\n\nIt broke these rules:\n" + "\n".join(f"- {x}" for x in problems) + \
             "\nReturn the same story with only these problems fixed; keep everything else as it is."
    return p


def _ngrams(words: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def lint_draft(d: dict, shots: int, parts: Parts = Parts()) -> list[str]:
    """Story rules a draft must follow; each problem says which shot and how to fix it."""
    problems = []
    for s in d.get("shots", []):
        if not parts.narrator and s.get("narration", "").strip():
            problems.append(f"shot {s.get('id', '?')} has narration but this film has no narrator; leave \"narration\" empty")
        if not parts.dialogue and s.get("dialogue"):
            problems.append(f"shot {s.get('id', '?')} has dialogue but no one speaks in this film; leave \"dialogue\" empty")
    got = d.get("shots", [])
    if len(got) != shots:
        problems.append(f"the story needs exactly {shots} shots, it has {len(got)}")
    full = [s.get("id", "?") for s in got if s.get("tier") == "full"]
    if len(full) > MAX_FULL_SHOTS:
        problems.append(f"shots {', '.join(full)} are all \"full\" motion; keep it to the {MAX_FULL_SHOTS} with real "
                        "action and make the others \"subtle\" or \"still\"")
    names = {c["name"]: c["description"] for c in d.get("characters", [])}
    angles = {a.get("name") for a in d.get("angles", [])}
    for pl in d.get("places", []):
        if not re.fullmatch(r"[A-Z]+", pl.get("name", "")):
            problems.append(f"place name {pl.get('name')} must be one UPPERCASE word, letters only (like TERRACE)")
        elif pl["name"] in names:
            problems.append(f"{pl['name']} is both a character and a place; rename the place")
    for a, b in zip(got, got[1:]):
        if a.get("angle") and a.get("angle") == b.get("angle"):
            problems.append(f"shots {a.get('id', '?')} and {b.get('id', '?')} are in a row from the same angle "
                            f"{a['angle']}; give one of them another angle so the cut shows a new view")
    for s in got:
        if s.get("angle") and s["angle"] not in angles:
            problems.append(f"shot {s.get('id', '?')} uses angle {s['angle']}, which isn't one of the angles "
                            f"({', '.join(sorted(a for a in angles if a)) or 'none'}); pick one or add it")
    for s in got:
        sid = s.get("id", "?")
        if not re.fullmatch(r"\d\d-[a-z0-9-]+", sid):
            problems.append(f"shot id '{sid}' must look like 01-name (two digits, a dash, lowercase words)")
        cw = len(s.get("caption", "").split())
        if cw > MAX_CAPTION_WORDS:
            problems.append(f"shot {sid} caption has {cw} words; keep it to {MAX_CAPTION_WORDS} or fewer")
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
        if re.search(r"\b(screen|monitor|display)s?\b[^.;]*?\b(show|shows|showing|display|displays|displaying)\b", still, re.I) \
                and "blank white screen" not in still.lower():
            problems.append(f"shot {sid} asks the image model to draw what a screen shows, which comes out as garbled "
                            "words; describe \"its bright blank white screen\" instead (the real page goes on afterwards)")
        if names and re.search(r"\b(shoulders?|hands?|arms?|fingers?)\b", still, re.I) \
                and not any(re.search(rf"\b{re.escape(n)}\b", still) for n in names):
            problems.append(f"shot {sid} shows someone's hands or shoulder without saying whose; write the character's "
                            f"NAME ({', '.join(names)}) so it is drawn as them, not a stranger")
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


def draft_to_toml(d: dict, idea: str, notes: str = "", style_image: str = "", style: str = "") -> str:
    q = _toml_str
    out = [f"# {d['title']}: drafted by spike-animate new from the idea:", f"#   {idea}",
           "# Edit freely, then: spike-animate board story.toml", "",
           f"idea = {q(idea)}", *([f"notes = {q(notes)}"] if notes else []),
           *([f"style_image = {q(style_image)}", f"style = {q(style)}"] if style_image else []),
           f"title = {q(d['title'])}", "seed = 1024", 'pace = "voice"', 'quality = "hd"', f"look = {q(d['look'])}", f"still_style = {q(d['still_style'])}",
           *([f"lighting = {q(d['lighting'])}"] if d.get("lighting") else []),
           "", "[characters]"]
    out += [f"{c['name']} = {q(c['description'])}" for c in d["characters"] if not c.get("voice")]
    for c in d["characters"]:
        if c.get("voice"):
            out += ["", f"[characters.{c['name']}]", f"look = {q(c['description'])}", f"voice = {q(c['voice'])}"]
    if d.get("places"):
        out += ["", "[places]"] + [f"{pl['name']} = {q(pl['description'])}" for pl in d["places"]]
    for i, a in enumerate(d.get("angles", [])):
        out += ["", f"[angles.{a['name']}]", f"frame = {q(a['frame'])}", f"seed = {1024 + 1000 * (i + 1)}"]
    if d.get("narrator") and any(s.get("narration") for s in d["shots"]):
        n = d["narrator"]
        out += ["", "[narrator]", f"voice = {q(n['voice'])}", f"sample = {q(n['sample'])}", f"lead = {DRAFT_LEAD:g}"]
    if d.get("music"):
        m = d["music"]
        out += ["", "[music]", f"caption = {q(m['caption'])}", f"key = {q(m['key'])}", "takes = 8", "db = -10",
                f"structure = {q(m['structure'])}"]
    e = d.get("end_card") or {}
    if e.get("text", "").strip():
        out += ["", "[end_card]", f"text = {q(e['text'].strip())}", f"sub = {q(e.get('sub', '').strip())}"]
    for s in d["shots"]:
        out += ["", "[[shot]]", f"id = {q(s['id'])}"]
        if s.get("angle"):
            out.append(f"angle = {q(s['angle'])}")
        out += [f"still = {q(s['still'])}", f"motion = {q(s['motion'])}"]
        if s.get("action"):
            out.append("anchor = 0   # action shot: pin only the first frame, let it move")
        if s.get("tier", "full") != "full":
            out.append(f"tier = {q(s['tier'])}")
        if s.get("tier") == "still":
            out.append(f"camera = {q(s.get('camera', 'push'))}")
        if s.get("lighting", "").strip():
            out.append(f"lighting = {q(s['lighting'].strip())}")
        if s.get("sfx"):
            out.append(f"sfx = {q(s['sfx'])}")
        if s.get("narration"):
            out.append(f"narration = {q(s['narration'])}")
        if s.get("caption", "").strip() or ("caption" in s and s.get("narration")):
            out.append(f"caption = {q(s['caption'].strip())}")
        if s.get("dialogue"):
            out.append("dialogue = [" + ", ".join(
                f"{{ speaker = {q(x['speaker'])}, line = {q(x['line'])}, emotion = {q(x.get('emotion', ''))} }}"
                for x in s["dialogue"]) + "]")
        out.append(f"intensity = {float(s['intensity']):g}")
    return "\n".join(out) + "\n"


STYLE_SCHEMA = {"type": "object", "required": ["style"], "properties": {"style": {"type": "string"}}}
STYLE_PROMPT = """Describe the visual style of this picture so an image model can paint new, different scenes in
the same style. Cover the medium (watercolor, cel animation, 3D render, ink, ...), the palette, the linework,
the shading and lighting, and the texture. Do not describe what is in the picture (no subjects, no setting).
Reply with one line of comma-separated phrases, at most 40 words, in "style"."""


def ollama_generate(prompt: str, model: str, schema: dict | None = None, images: list[str] | None = None) -> dict:
    host = os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434"
    if "://" not in host:
        host = "http://" + host
    body = {"model": model, "prompt": prompt, "stream": False, "think": False, "format": schema or DRAFT_SCHEMA,
            "keep_alive": 0, "options": {"temperature": 0.7, "num_ctx": 8192}}
    if images:
        body["images"] = images
    req = urllib.request.Request(f"{host.rstrip('/')}/api/generate", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            reply = json.loads(r.read())
    except OSError as e:
        raise StageError(f"could not reach Ollama at {host} ({e}); is `ollama serve` running?") from None
    except ValueError:
        raise StageError(f"Ollama at {host} sent back something that is not JSON") from None
    try:
        return json.loads(reply["response"])
    except (KeyError, ValueError):
        raise StageError("the writer model did not return a JSON story") from None


def describe_style(image: Path, model: str) -> str:
    """The style of a reference picture, in words (a local vision model; the caller holds the lane)."""
    import base64
    reply = ollama_generate(STYLE_PROMPT, model, STYLE_SCHEMA, [base64.b64encode(Path(image).read_bytes()).decode()])
    return " ".join(str(reply.get("style", "")).split())


def cmd_new(folder: Path, idea: str, shots: int, model: str, attempts: int = 3, notes: str = "",
            style_image: Path | None = None, parts: Parts = Parts(), look: str = "keyart") -> int:
    import spike_lane
    story_file = folder / "story.toml"
    if story_file.exists():
        print(f"{story_file} already exists; pick a new folder or edit that story", file=sys.stderr)
        return 2
    _say(f"drafting a {shots}-shot story with {model} …")
    fd = spike_lane.acquire("writer", ["spike-animate", "new", idea], wait=None)
    try:
        style = ""
        if style_image is not None:
            _say("reading the style of the reference picture …")
            style = describe_style(style_image, model)
        problems, d = None, None
        for attempt in range(1, attempts + 1):
            d = ollama_generate(writer_prompt(idea, shots, problems, previous=d if problems else None, notes=notes,
                                              style=style, parts=parts, look=look), model, draft_schema(parts))
            problems = lint_draft(d, shots, parts)
            if not problems:
                break
            _say(f"draft {attempt} broke {len(problems)} rule(s); asking for a fix …")
    finally:
        os.close(fd)
    if problems:
        print("the writer couldn't produce a valid story:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    folder.mkdir(parents=True, exist_ok=True)
    rel = ""
    if style_image is not None:
        dest = folder / "reference" / f"style{Path(style_image).suffix.lower()}"
        if Path(style_image).resolve() != dest.resolve():
            dest.parent.mkdir(exist_ok=True)
            shutil.copyfile(style_image, dest)
        rel = str(dest.relative_to(folder))
    story_file.write_text(draft_to_toml(d, idea, notes, style_image=rel, style=style))
    load_story(story_file)   # must load cleanly
    _say(f"Story drafted: {story_file} ({d['title']}, {len(d['shots'])} shots)\n"
         f"Read and edit it, then run `spike-animate board {story_file}`.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="spike-animate", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, helptext in [("board", "render the storyboard stills and a contact sheet, then stop for review"),
                           ("video", "animate every shot from its still (no sound)"),
                           ("sound", "add sound effects to the animated shots"),
                           ("voices", "record the narration and dialogue"),
                           ("music", "score the film"),
                           ("film", "do whatever is left of the steps above, then mix the film"),
                           ("status", "show what is done")]:
        sp = sub.add_parser(name, help=helptext)
        sp.add_argument("story", type=Path)
        if name in ("board", "video", "sound", "voices"):
            sp.add_argument("--shot", help="work on this one scene only")
    rk = sub.add_parser("rank", help="score a scene's picture versions with the local vision model, best first")
    rk.add_argument("story", type=Path)
    rk.add_argument("shot")
    ck = sub.add_parser("check", help="check pictures against the film bible with the local vision model")
    ck.add_argument("story", type=Path)
    ck.add_argument("--shot")
    ck.add_argument("--motion", action="store_true", help="check the scenes' motion for clipping instead")
    lk = sub.add_parser("look", help="draw a character sheet or a place picture that scenes are drawn from")
    lk.add_argument("story", type=Path)
    lk.add_argument("name")
    lk.add_argument("--seeds", help="draw these versions instead, e.g. 7,42,99")
    vc = sub.add_parser("voice", help="design a voice (narrator or a character) before any line is recorded")
    vc.add_argument("story", type=Path)
    vc.add_argument("name")
    vc.add_argument("--takes", type=int, help="make this many more takes to choose from")
    sc = sub.add_parser("scene", help="preview one scene: its clip with its own sound and lines")
    sc.add_argument("story", type=Path)
    sc.add_argument("shot")
    nw = sub.add_parser("new", help="draft story.toml for a new film from a one-line idea (local model)")
    nw.add_argument("folder", type=Path)
    nw.add_argument("idea")
    nw.add_argument("--shots", type=int, default=6)
    nw.add_argument("--notes", default="", help="reference material for the writer: names, looks, setting, tone")
    nw.add_argument("--style-image", type=Path, help="a picture whose style the film should copy")
    nw.add_argument("--model", default=os.environ.get("SPIKE_WRITER_MODEL", "gemma4:26b"))
    nw.add_argument("--no-narrator", action="store_true", help="no narrator: the story is told by picture and dialogue")
    nw.add_argument("--no-dialogue", action="store_true", help="the characters never speak")
    nw.add_argument("--no-music", action="store_true", help="no score")
    nw.add_argument("--look", choices=LOOKS, default="keyart",
                    help="without a style picture: rich anime key art (default) or a flat 2D cartoon")
    rt = sub.add_parser("retake", help="render alternate seeds for one shot's still")
    rt.add_argument("story", type=Path)
    rt.add_argument("shot")
    rt.add_argument("--seeds", required=True, help="comma-separated, e.g. 7,42,99")
    rt.add_argument("--rank", action="store_true", help="then score every version with the local vision model")
    return ap


def main(argv: list[str]) -> int:
    a = build_parser().parse_args(argv)
    try:
        if a.cmd == "new":
            return cmd_new(a.folder, a.idea, a.shots, a.model, notes=a.notes, style_image=a.style_image,
                           parts=Parts(narrator=not a.no_narrator, dialogue=not a.no_dialogue, music=not a.no_music),
                           look=a.look)
        story = load_story(a.story)
        if a.cmd == "board":
            return cmd_board(story, a.shot)
        if a.cmd == "scene":
            return cmd_scene(story, a.shot)
        if a.cmd == "check":
            return cmd_check(story, a.shot, motion=a.motion)
        if a.cmd == "rank":
            return cmd_rank(story, a.shot)
        if a.cmd == "voice":
            return cmd_voice(story, a.name, a.takes)
        if a.cmd == "look":
            return cmd_look(story, a.name, [int(s) for s in a.seeds.split(",") if s.strip()] if a.seeds else None)
        if a.cmd == "retake":
            return cmd_retake(story, a.shot, [int(s) for s in a.seeds.split(",") if s.strip()], rank=a.rank)
        if a.cmd in ("video", "sound", "voices"):
            return {"video": cmd_video, "sound": cmd_sound, "voices": cmd_voices}[a.cmd](story, a.shot)
        steps = {"film": cmd_film, "music": cmd_music}
        if a.cmd in steps:
            return steps[a.cmd](story)
        return cmd_status(story)
    except (StoryError, StageError) as e:
        print(f"spike-animate: {e}", file=sys.stderr)
        return 1
