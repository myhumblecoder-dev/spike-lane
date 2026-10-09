"""spike-studio — a web UI over spike-animate, made for a phone.

The studio never renders anything itself. It edits `story.toml`, starts spike-animate steps as
background jobs (one per film; spike-lane still runs one heavy workload on the machine at a time),
and shows what each step left on disk. A film is a folder under the studio root holding story.toml.

Access needs a token: open the URL that `spike-studio` prints once, and the browser keeps a cookie.
"""
from __future__ import annotations

import argparse
import hmac
import json
import mimetypes
import os
import re
import secrets
import shutil
import signal
import socket
import socketserver
import subprocess
import sys
import threading
import time
import tomllib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

import spike_animate as sa

HERE = Path(__file__).resolve().parent
PAGE = HERE / "studio" / "index.html"


class Busy(RuntimeError):
    """The film already has a step running."""


# --- writing story.toml ---------------------------------------------------------
# The editor works on the parsed TOML (a plain dict) and writes it back. Only the shapes a story uses
# are needed: top-level values, tables (with sub-tables such as [characters.ZAP]), and [[shot]] arrays
# whose nested tables and lists are written inline.

def _key(k: str) -> str:
    return k if re.fullmatch(r"[A-Za-z0-9_-]+", k) else json.dumps(k, ensure_ascii=False)


def _value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)   # a JSON string is a valid TOML basic string
    if isinstance(v, dict):
        return "{ " + ", ".join(f"{_key(k)} = {_value(x)}" for k, x in v.items()) + " }" if v else "{}"
    if isinstance(v, list):
        return "[" + ", ".join(_value(x) for x in v) + "]"
    raise TypeError(f"can't write {type(v).__name__} to TOML")


def _is_table_array(v) -> bool:
    return isinstance(v, list) and bool(v) and all(isinstance(x, dict) for x in v)


def _table(out: list[str], name: str, d: dict) -> None:
    plain = {k: v for k, v in d.items() if not isinstance(v, dict)}
    subs = {k: v for k, v in d.items() if isinstance(v, dict)}
    if plain or not subs:
        out += ["", f"[{name}]"] + [f"{_key(k)} = {_value(v)}" for k, v in plain.items()]
    for k, v in subs.items():
        _table(out, f"{name}.{_key(k)}", v)


def doc_to_toml(doc: dict) -> str:
    out = [f"{_key(k)} = {_value(v)}" for k, v in doc.items() if not isinstance(v, dict) and not _is_table_array(v)]
    for k, v in doc.items():
        if isinstance(v, dict):
            _table(out, _key(k), v)
    for k, v in doc.items():
        if _is_table_array(v):
            for item in v:
                out += ["", f"[[{_key(k)}]]"] + [f"{_key(i)} = {_value(x)}" for i, x in item.items()]
    return "\n".join(out).lstrip("\n") + "\n"


def save_story(folder: Path, doc: dict | None = None, toml: str | None = None) -> list[str]:
    """Write story.toml if the new text loads as a story; otherwise leave the file alone and say why."""
    folder = Path(folder)
    try:
        text = toml if toml is not None else doc_to_toml(doc)
    except TypeError as e:
        return [str(e)]
    check = folder / ".story-check.toml"
    check.write_text(text)
    try:
        sa.load_story(check)
    except sa.StoryError as e:
        check.unlink()
        return [str(e).replace(check.name, "story.toml")]
    os.replace(check, folder / "story.toml")
    return []


def _doc(folder: Path) -> dict:
    return tomllib.loads((Path(folder) / "story.toml").read_text())


def pick_seed(folder: Path, shot_id: str, seed: int) -> list[str]:
    doc = _doc(folder)
    shot = next((s for s in doc.get("shot", []) if s.get("id") == shot_id), None)
    if shot is None:
        return [f"no shot '{shot_id}' in the story"]
    shot["seed"] = int(seed)
    return save_story(folder, doc=doc)


def pick_take(folder: Path, take: int | None) -> list[str]:
    doc = _doc(folder)
    if "music" not in doc:
        return ["the story has no [music]"]
    if take is None:
        doc["music"].pop("take", None)
    else:
        doc["music"]["take"] = int(take)
    return save_story(folder, doc=doc)


def _current_retakes(folder: Path, story, shot) -> list[tuple[int, Path]]:
    """This shot's retakes drawn from its current description (older ones would redraw differently)."""
    want = sa.still_prompt(story, shot) if story else None
    found = []
    for p in (Path(folder) / "board" / "retakes").glob(f"{shot.id if story else shot}-s*.png"):
        m = re.fullmatch(rf"{re.escape(shot.id if story else shot)}-s(\d+)\.png", p.name)
        if m and (want is None or sa.png_prompt(p) == want):
            found.append((int(m.group(1)), p))
    return sorted(found)


def use_retake(folder: Path, shot_id: str, seed: int) -> list[str]:
    """Keep a retake: its seed goes into the story and its picture becomes the shot's still, with no redraw."""
    folder = Path(folder)
    story = sa.load_story(folder / "story.toml")
    shot = next((x for x in story.shots if x.id == shot_id), None)
    if shot is None:
        return [f"no shot '{shot_id}' in the story"]
    pic = dict(_current_retakes(folder, story, shot)).get(int(seed))
    if pic is None:
        return [f"no retake with seed {seed} for the current description of {shot_id}"]
    errors = pick_seed(folder, shot_id, int(seed))
    if errors:
        return errors
    story = sa.load_story(folder / "story.toml")
    shot = next(x for x in story.shots if x.id == shot_id)
    out = sa.still_path(story, shot)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(pic, out)
    sa.Keys(folder).record(out, sa.still_key(story, shot))
    return []


# --- one step at a time ------------------------------------------------------------

STAGES = ("story", "board", "video", "sound", "voices", "music", "film")
STEP_STAGE = {"board": "board", "retake": "board", "video": "video", "sound": "sound", "voices": "voices",
              "music": "music", "film": "film"}
STAGE_NAMES = {"story": "Story", "board": "Pictures", "video": "Video", "sound": "Sound effects",
               "voices": "Voices & dialogue", "music": "Music", "film": "Final film"}


def _outputs(folder: Path, doc: dict, story) -> dict[str, list[Path] | None]:
    """The files each step makes; None while the step isn't finished."""
    shots = doc.get("shot", [])
    ids = [x.get("id", "") for x in shots]

    def every(paths):
        return paths if all(p.is_file() for p in paths) else None

    out: dict[str, list[Path] | None] = {"story": [] if story else None}
    board = [folder / "board" / f"{i}.png" for i in ids]
    if story:
        keys = sa.Keys(folder)
        fresh = all(not keys.stale(sa.still_path(story, sh), sa.still_key(story, sh)) for sh in story.shots)
        out["board"] = board if fresh else None
    else:
        out["board"] = None
    out["video"] = every([folder / "shots" / f"{i}.mp4" for i in ids])
    out["sound"] = every([folder / "sfx" / f"{i}.flac" for i in ids] + [folder / "film" / "sfx.wav"])
    voices = [folder / "voice" / f"{x['id']}.wav" for x in shots if x.get("narration")]
    for x in shots:
        voices += [folder / "voice" / "dialogue" / f"{x['id']}-{n}-{l.get('speaker', '')}.wav"
                   for n, l in enumerate(x.get("dialogue", []), 1)]
    out["voices"] = every(voices)
    music = doc.get("music")
    if music is None:
        out["music"] = []
    elif "take" in music:
        out["music"] = every([folder / "score" / f"take-s{int(music['take'])}.wav"])
    else:
        try:
            chosen = json.loads((folder / "film" / "checks.json").read_text())["score"]["chosen"]
            out["music"] = every([Path(chosen)])
        except (OSError, ValueError, KeyError, TypeError):
            out["music"] = None
    out["film"] = every([folder / "film" / f"{_slug(doc.get('title', '')) or 'film'}.mp4"])
    return out


def _fingerprint(files: list[Path]) -> str:
    rows = []
    for p in files:
        st_ = p.stat()
        rows.append([str(p), st_.st_size, st_.st_mtime_ns])
    return sa.key_for(rows)


def _approvals_file(folder: Path) -> Path:
    return Path(folder) / ".spike-animate" / "approved.json"


def _steps(folder: Path, doc: dict | None, story) -> list[dict]:
    """Each step: done (its files exist), approved (and nothing it or an earlier step made has changed since),
    unlocked (every earlier step is approved). An approval covers the files of its step and all before it."""
    folder = Path(folder)
    try:
        saved = json.loads(_approvals_file(folder).read_text())
    except (OSError, ValueError):
        saved = {}
    outs = _outputs(folder, doc or {}, story) if doc is not None else {k: None for k in STAGES}
    steps, upstream, ok = [], [], True
    for stage in STAGES:
        files = outs[stage]
        done = files is not None
        fp = _fingerprint(upstream + files) if done else None
        approved = ok and done and saved.get(stage) == fp
        steps.append({"stage": stage, "name": STAGE_NAMES[stage], "done": done, "approved": approved,
                      "unlocked": ok, "fingerprint": fp})
        ok = approved
        upstream += files or []
    return steps


def _load(folder: Path):
    doc = _doc(folder) if (Path(folder) / "story.toml").exists() else None
    try:
        story = sa.load_story(Path(folder) / "story.toml") if doc is not None else None
    except sa.StoryError:
        story = None
    return doc, story


def approve(folder: Path, stage: str, undo: bool = False) -> list[str]:
    if stage not in STAGES:
        return [f"unknown step '{stage}'"]
    steps = {x["stage"]: x for x in _steps(folder, *_load(folder))}
    f = _approvals_file(folder)
    try:
        saved = json.loads(f.read_text())
    except (OSError, ValueError):
        saved = {}
    me = steps[stage]
    if undo:
        saved.pop(stage, None)
    elif not me["unlocked"]:
        before = next(x for x in steps.values() if not x["approved"])
        return [f"approve {before['name']} first"]
    elif not me["done"]:
        return [f"{me['name']} isn't finished yet"]
    else:
        saved[stage] = me["fingerprint"]
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(saved, indent=1))
    return []


def gate(folder: Path, step: str) -> str | None:
    """Why `step` can't run yet, or None when every step before it is approved."""
    stage = STEP_STAGE.get(step)
    if stage is None:
        return None
    steps = _steps(folder, *_load(folder))
    if next(x for x in steps if x["stage"] == stage)["unlocked"]:
        return None
    before = next(x for x in steps if not x["approved"])
    return f"approve {before['name']} first"


# --- what a film looks like on disk ----------------------------------------------

def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def film_name(title: str) -> str:
    name = _slug(title)[:48].strip("-")
    if not name:
        raise ValueError(f"'{title}' doesn't make a folder name")
    return name


def inside(folder: Path, rel: str) -> Path | None:
    """`rel` inside `folder`, or None if it points anywhere else."""
    if not rel or rel.startswith("/") or "\0" in rel:
        return None
    base = Path(folder).resolve()
    p = (base / rel).resolve()
    return p if p.is_relative_to(base) else None


def _url(folder: Path, rel: str) -> str | None:
    p = folder / rel
    return f"{rel}?v={int(p.stat().st_mtime)}" if p.is_file() else None


def film_state(folder: Path) -> dict:
    """Everything the page shows about one film: the story, its errors, and every rendered file."""
    folder = Path(folder)
    sf = folder / "story.toml"
    state = {"name": folder.name, "title": folder.name, "toml": "", "doc": None, "errors": [], "shots": [],
             "voices": {}, "score": {"takes": [], "pinned": None, "chosen": None}, "checks": {}, "film": None,
             "board": _url(folder, "board/contact-sheet.jpg"), "stage": "story", "idea": "", "notes": "",
             "references": sorted(str(p.relative_to(folder)) for p in (folder / "reference").glob("*") if p.is_file())}
    if not sf.exists():
        state["errors"] = ["no story yet"]
        state["steps"] = [{k: v for k, v in x.items() if k != "fingerprint"} for x in _steps(folder, None, None)]
        return state
    state["toml"] = sf.read_text()
    try:
        doc = tomllib.loads(state["toml"])
    except tomllib.TOMLDecodeError as e:
        state["errors"] = [f"story.toml: {e}"]
        state["steps"] = [{k: v for k, v in x.items() if k != "fingerprint"} for x in _steps(folder, None, None)]
        return state
    state["doc"] = doc
    state.update(title=doc.get("title") or folder.name, idea=doc.get("idea", ""), notes=doc.get("notes", ""))
    story = None
    try:
        story = sa.load_story(sf)
    except sa.StoryError as e:
        state["errors"] = [str(e)]
    keys = sa.Keys(folder) if story else None
    for s in doc.get("shot", []):
        sid = s.get("id", "")
        shot_obj = next((x for x in story.shots if x.id == sid), None) if story else None
        retakes = _current_retakes(folder, story, shot_obj) if shot_obj else _current_retakes(folder, None, sid)
        lines = sorted((folder / "voice" / "dialogue").glob(f"{sid}-*-*.wav"),
                       key=lambda p: int(p.name[len(sid) + 1:].split("-")[0]) if p.name[len(sid) + 1:].split("-")[0].isdigit() else 0)
        still_fresh = None
        if story:
            shot = next(x for x in story.shots if x.id == sid)
            still_fresh = not keys.stale(sa.still_path(story, shot), sa.still_key(story, shot))
        state["shots"].append({
            "id": sid, "still_text": s.get("still", ""), "motion": s.get("motion", ""),
            "narration": s.get("narration", ""), "dialogue": s.get("dialogue", []),
            "seed": s.get("seed", doc.get("seed", 1024)), "still_fresh": still_fresh,
            "still": _url(folder, f"board/{sid}.png"),
            "retakes": [{"seed": seed, "url": _url(folder, str(p.relative_to(folder)))} for seed, p in retakes],
            "clip": _url(folder, f"shots/{sid}.mp4"), "sfx": _url(folder, f"sfx/{sid}.flac"),
            "narration_audio": _url(folder, f"voice/{sid}.wav"),
            "dialogue_audio": [_url(folder, str(p.relative_to(folder))) for p in lines],
        })
    voices = {"narrator": _url(folder, "voice/card.wav")} if "narrator" in doc else {}
    for name, c in doc.get("characters", {}).items():
        if isinstance(c, dict) and c.get("voice"):
            voices[name] = _url(folder, f"voice/cards/{name}.wav")
    state["voices"] = voices
    rank_file = folder / "score" / "rank.json"
    ranking = {}
    if rank_file.exists():
        try:
            ranking = {Path(r["take"]).name: r.get("score") for r in json.loads(rank_file.read_text())}
        except (ValueError, KeyError, TypeError):
            ranking = {}
    takes = sorted((int(m.group(1)), p) for p in (folder / "score").glob("take-s*.wav")
                   if (m := re.fullmatch(r"take-s(\d+)\.wav", p.name)))
    checks_file = folder / "film" / "checks.json"
    if checks_file.exists():
        try:
            state["checks"] = json.loads(checks_file.read_text())
        except ValueError:
            pass
    chosen = state["checks"].get("score", {}).get("chosen")
    state["score"] = {"takes": [{"seed": seed, "url": _url(folder, str(p.relative_to(folder))), "score": ranking.get(p.name)}
                                for seed, p in takes],
                      "pinned": doc.get("music", {}).get("take"),
                      "chosen": int(m.group(1)) if chosen and (m := re.search(r"take-s(\d+)", chosen)) else None}
    state["film"] = _url(folder, f"film/{_slug(doc.get('title', '')) or 'film'}.mp4")
    state["steps"] = [{k: v for k, v in x.items() if k != "fingerprint"} for x in _steps(folder, doc, story)]
    sh = state["shots"]
    state["stage"] = ("film" if state["film"] else "shots" if any(x["clip"] for x in sh)
                      else "board" if any(x["still"] for x in sh) else "story")
    return state


def list_films(root: Path) -> list[dict]:
    films = []
    for sf in Path(root).glob("*/story.toml"):
        folder = sf.parent
        try:
            title = tomllib.loads(sf.read_text()).get("title") or folder.name
        except tomllib.TOMLDecodeError:
            title = folder.name
        stills = sorted((folder / "board").glob("[0-9]*.png"))
        poster = _url(folder, "board/contact-sheet.jpg") or (_url(folder, str(stills[0].relative_to(folder))) if stills else None)
        film = _url(folder, f"film/{_slug(title) or 'film'}.mp4")
        films.append({"name": folder.name, "title": title, "poster": poster, "film": film,
                      "updated": max(p.stat().st_mtime for p in [sf, *folder.glob("*/*")] if p.exists())})
    return sorted(films, key=lambda f: -f["updated"])


# --- steps --------------------------------------------------------------------

def step_argv(folder: Path, step: str, params: dict) -> list[str]:
    """The spike-animate arguments for one step, checked: anything odd is a ValueError, never a command."""
    folder = Path(folder)
    story = str(folder / "story.toml")
    if step in ("board", "video", "sound", "voices", "music", "film", "status"):
        return [step, story]
    if step == "retake":
        ids = [s.get("id") for s in _doc(folder).get("shot", [])]
        if params.get("shot") not in ids:
            raise ValueError(f"no shot '{params.get('shot')}' in the story")
        seeds = [int(str(x).strip()) for x in params.get("seeds", [])]
        if not 1 <= len(seeds) <= 6 or any(s < 0 for s in seeds):
            raise ValueError("give 1 to 6 seeds")
        return ["retake", story, params["shot"], "--seeds", ",".join(map(str, seeds))]
    if step == "new":
        idea = str(params.get("idea", "")).strip().lstrip("-").strip()
        notes = str(params.get("notes", "")).strip().lstrip("-").strip()
        shots = int(params.get("shots", 8))
        if not idea:
            raise ValueError("the idea is empty")
        if not 3 <= shots <= 16:
            raise ValueError("a film has 3 to 16 shots")
        argv = ["new", str(folder), idea, "--shots", str(shots)] + (["--notes", notes] if notes else [])
        if params.get("style_image"):
            pic = inside(folder, str(params["style_image"]))
            if pic is None or not pic.is_file() or pic.suffix.lower() not in IMAGE_TYPES:
                raise ValueError("the style picture must be an uploaded image")
            argv += ["--style-image", str(pic)]
        return argv
    raise ValueError(f"unknown step '{step}'")


def _animate_bin() -> str:
    return os.environ.get("SPIKE_ANIMATE_BIN") or str(HERE / "bin" / "spike-animate")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class Jobs:
    """One background spike-animate run per film. The record lives in the film folder, so a job keeps
    running, and its result stays readable, across studio restarts."""

    def __init__(self):
        self._lock = threading.Lock()

    @staticmethod
    def _dir(folder: Path) -> Path:
        return Path(folder) / ".spike-animate"

    def start(self, folder: Path, argv: list[str]) -> dict:
        with self._lock:
            j = self.status(folder)
            if j and j["running"]:
                raise Busy(f"{j['step']} is still running")
            d = self._dir(folder)
            d.mkdir(parents=True, exist_ok=True)
            log, rc = d / "job.log", d / "job.rc"
            rc.unlink(missing_ok=True)
            env = os.environ.copy()   # launchd starts the studio with a bare PATH; the engines need these
            env["PATH"] = os.pathsep.join([str(Path.home() / ".local" / "bin"), "/opt/homebrew/bin", "/usr/local/bin",
                                           env.get("PATH", "/usr/bin:/bin")])
            proc = subprocess.Popen(["/bin/sh", "-c", '"$@" > "$JOB_LOG" 2>&1; echo $? > "$JOB_RC"', "sh",
                                     _animate_bin(), *argv], env={**env, "JOB_LOG": str(log), "JOB_RC": str(rc)},
                                    start_new_session=True, stdin=subprocess.DEVNULL)
            (d / "job.json").write_text(json.dumps({"step": argv[0], "argv": argv, "pid": proc.pid, "started": time.time()}))
            threading.Thread(target=proc.wait, daemon=True).start()   # reap it
            return {"step": argv[0], "running": True}

    def status(self, folder: Path) -> dict | None:
        d = self._dir(folder)
        try:
            meta = json.loads((d / "job.json").read_text())
        except (OSError, ValueError):
            return None
        rc_file, log_file = d / "job.rc", d / "job.log"
        rc = None
        if rc_file.exists():
            try:
                rc = int(rc_file.read_text().strip())
            except ValueError:
                rc = None
        running = rc is None and _alive(meta["pid"])
        log = log_file.read_text(errors="replace")[-6000:] if log_file.exists() else ""
        ended = rc_file.stat().st_mtime if rc_file.exists() else None
        return {"step": meta["step"], "running": running, "rc": rc, "log": log, "started": meta["started"],
                "seconds": round((ended or time.time()) - meta["started"])}

    def stop(self, folder: Path) -> bool:
        j = self.status(folder)
        if not j or not j["running"]:
            return False
        pid = json.loads((self._dir(folder) / "job.json").read_text())["pid"]
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            return False
        return True


# --- the server ------------------------------------------------------------------

TYPES = {".flac": "audio/flac", ".wav": "audio/wav", ".mp3": "audio/mpeg", ".mp4": "video/mp4", ".mov": "video/quicktime",
         ".m4v": "video/mp4", ".toml": "text/plain; charset=utf-8", ".json": "application/json", ".log": "text/plain; charset=utf-8"}
IMAGE_TYPES = {".png", ".jpg", ".jpeg"}
UPLOAD_TYPES = {".mp4", ".mov", ".m4v", ".webm"} | IMAGE_TYPES
MAX_UPLOAD = 1 << 30
MAX_JSON = 4 << 20


class Studio(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr, root: Path, token: str):
        self.root, self.token, self.jobs = Path(root).resolve(), token, Jobs()
        super().__init__(addr, Handler)

    def server_bind(self):   # skip HTTPServer's reverse-DNS lookup (seconds on macOS)
        if self.address_family == socket.AF_INET6:
            self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = "studio", self.socket.getsockname()[1]


def make_server(root: Path, token: str, host: str = "127.0.0.1", port: int = 8765) -> Studio:
    return Studio((host, port), root, token)


class Handler(BaseHTTPRequestHandler):
    server: Studio
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    # helpers
    def _send(self, status: int, body: bytes = b"", ctype: str = "application/json", headers: dict | None = None):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, status: int, data, headers=None):
        self._send(status, json.dumps(data).encode(), headers=headers)

    def _body(self, limit: int = MAX_JSON) -> bytes:
        n = int(self.headers.get("Content-Length") or 0)
        if n > limit:
            raise ValueError("too large")
        return self.rfile.read(n)

    def _authed(self, query: dict) -> tuple[bool, dict]:
        tok = self.server.token.encode()
        cookie = re.search(r"(?:^|;\s*)studio=([^;]+)", self.headers.get("Cookie", ""))
        if cookie and hmac.compare_digest(cookie.group(1).encode(), tok):
            return True, {}
        given = (query.get("t") or [""])[0]
        if given and hmac.compare_digest(given.encode(), tok):
            return True, {"Set-Cookie": f"studio={self.server.token}; Path=/; HttpOnly; SameSite=Lax; Max-Age=31536000"}
        return False, {}

    def _film(self, name: str) -> Path | None:
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", name or ""):
            return None
        folder = self.server.root / name
        return folder if folder.is_dir() else None

    def _state(self, folder: Path) -> dict:
        s = film_state(folder)
        s["job"] = self.server.jobs.status(folder)
        return s

    # routing
    def do_HEAD(self):   # noqa: N802
        self.do_GET()

    def do_GET(self):    # noqa: N802
        self._route("GET")

    def do_POST(self):   # noqa: N802
        self._route("POST")

    def do_PUT(self):    # noqa: N802
        self._route("PUT")

    def _route(self, method: str):
        url = urlsplit(self.path)
        query = parse_qs(url.query)
        ok, cookie = self._authed(query)
        if not ok:
            return self._send(401, b"Open the studio link with its token (spike-studio prints it).", "text/plain")
        parts = [unquote(p) for p in url.path.split("/")[1:]]
        try:
            if method == "GET" and url.path == "/":
                return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8", cookie)
            if parts[:1] == ["media"] and method == "GET" and len(parts) >= 3:
                return self._media(parts[1], "/".join(parts[2:]))
            if parts[:2] == ["api", "films"]:
                return self._api(method, parts[2:], query)
            return self._send(404, b"not found", "text/plain")
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _api(self, method: str, rest: list[str], query: dict):
        root, jobs = self.server.root, self.server.jobs
        if not rest:
            if method == "GET":
                return self._json(200, list_films(root))
            if method == "POST":
                try:
                    p = json.loads(self._body() or b"{}")
                    name = film_name(p.get("title") or p.get("idea", "")[:40])
                except ValueError as e:
                    return self._json(400, {"errors": [str(e)]})
                folder = root / name
                j = jobs.status(folder) if folder.exists() else None
                if (folder / "story.toml").exists() or (j and j["running"]):
                    return self._json(409, {"errors": [f"a film called '{name}' already exists"]})
                if not p.get("idea"):            # make the film now; write its story after (e.g. once a style picture is in)
                    folder.mkdir(parents=True, exist_ok=True)
                    return self._json(201, {"name": name})
                try:
                    argv = step_argv(folder, "new", p)
                except ValueError as e:
                    return self._json(400, {"errors": [str(e)]})
                folder.mkdir(parents=True, exist_ok=True)
                jobs.start(folder, argv)
                return self._json(202, {"name": name})
            return self._json(405, {"errors": ["method not allowed"]})
        folder = self._film(rest[0])
        if folder is None:
            return self._json(404, {"errors": ["no such film"]})
        action = rest[1] if len(rest) > 1 else ""
        if method == "GET" and not action:
            return self._json(200, self._state(folder))
        if method == "PUT" and action == "story":
            p = json.loads(self._body() or b"{}")
            errors = save_story(folder, doc=p.get("doc"), toml=p.get("toml"))
            return self._json(400, {"errors": errors}) if errors else self._json(200, self._state(folder))
        if method == "POST" and action == "run":
            p = json.loads(self._body() or b"{}")
            try:
                argv = step_argv(folder, p.get("step", ""), p)
                if p.get("step") == "new" and (folder / "story.toml").exists():
                    raise Busy("this film already has a story")
                why = gate(folder, p.get("step", ""))
                if why:
                    raise Busy(why)
                jobs.start(folder, argv)
            except ValueError as e:
                return self._json(400, {"errors": [str(e)]})
            except (Busy, OSError) as e:
                return self._json(409, {"errors": [str(e)]})
            return self._json(202, self._state(folder))
        if method == "POST" and action == "approve":
            p = json.loads(self._body() or b"{}")
            errors = approve(folder, str(p.get("stage", "")), undo=bool(p.get("undo")))
            return self._json(400, {"errors": errors}) if errors else self._json(200, self._state(folder))
        if method == "POST" and action == "use":
            p = json.loads(self._body() or b"{}")
            try:
                errors = use_retake(folder, str(p["shot"]), int(p["seed"]))
            except (KeyError, ValueError, TypeError, sa.StoryError) as e:
                errors = [f"bad request: {e}"]
            return self._json(400, {"errors": errors}) if errors else self._json(200, self._state(folder))
        if method == "POST" and action == "stop":
            return self._json(200, {"stopped": jobs.stop(folder)})
        if method == "POST" and action in ("seed", "take"):
            p = json.loads(self._body() or b"{}")
            try:
                errors = (pick_seed(folder, p["shot"], int(p["seed"])) if action == "seed"
                          else pick_take(folder, None if p.get("take") is None else int(p["take"])))
            except (KeyError, ValueError, TypeError) as e:
                errors = [f"bad request: {e}"]
            return self._json(400, {"errors": errors}) if errors else self._json(200, self._state(folder))
        if method == "POST" and action == "upload":
            raw = (query.get("name") or [""])[0]
            ext = Path(raw).suffix.lower()
            try:
                stem = film_name(Path(raw).stem)
            except ValueError as e:
                return self._json(400, {"errors": [str(e)]})
            if ext not in UPLOAD_TYPES:
                return self._json(400, {"errors": [f"upload a video or image ({', '.join(sorted(UPLOAD_TYPES))})"]})
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_UPLOAD:
                return self._json(413, {"errors": ["file is over 1 GB"]})
            dest = folder / "reference" / f"{stem}{ext}"
            dest.parent.mkdir(exist_ok=True)
            part = dest.with_name(dest.name + ".part")
            with open(part, "wb") as f:
                left = n
                while left:
                    chunk = self.rfile.read(min(left, 1 << 20))
                    if not chunk:
                        break
                    f.write(chunk)
                    left -= len(chunk)
            if left:
                part.unlink()
                return self._json(400, {"errors": ["upload was cut short"]})
            os.replace(part, dest)
            return self._json(201, {"path": str(dest.relative_to(folder))})
        return self._json(404, {"errors": ["not found"]})

    def _media(self, name: str, rel: str):
        folder = self._film(name)
        p = inside(folder, rel) if folder else None
        if p is None or not p.is_file():
            return self._send(404, b"not found", "text/plain")
        size = p.stat().st_size
        ctype = TYPES.get(p.suffix.lower()) or mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        start, end, status = 0, size - 1, 200
        m = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", "").strip())
        if m and size and (m.group(1) or m.group(2)):
            if m.group(1):
                start = int(m.group(1))
                end = min(int(m.group(2)), size - 1) if m.group(2) else size - 1
            else:
                start = max(0, size - int(m.group(2)))
            if start > end:
                return self._send(416, b"", "text/plain", {"Content-Range": f"bytes */{size}"})
            status = 206
        length = end - start + 1 if size else 0
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "private, max-age=3600")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if self.command == "HEAD":
            return
        with open(p, "rb") as f:
            f.seek(start)
            left = length
            while left:
                chunk = f.read(min(left, 1 << 20))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)


# --- command line ------------------------------------------------------------------

def _token(path: Path) -> str:
    if path.exists():
        return path.read_text().strip()
    path.parent.mkdir(parents=True, exist_ok=True)
    tok = secrets.token_urlsafe(18)
    path.write_text(tok + "\n")
    path.chmod(0o600)
    return tok


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="spike-studio", description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, default=Path.home() / "Movies" / "spike-video", help="folder holding the films")
    ap.add_argument("--host", default="127.0.0.1", help="loopback only; share it with `tailscale serve`")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--token-file", type=Path, default=Path.home() / ".config" / "spike-studio" / "token")
    a = ap.parse_args(argv)
    a.root.mkdir(parents=True, exist_ok=True)
    token = _token(a.token_file)
    srv = make_server(a.root, token, a.host, a.port)
    print(f"spike-studio: films in {a.root}\n  open http://{a.host}:{srv.server_port}/?t={token} (or your tailscale serve address)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
