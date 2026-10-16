"""spike-studio: the web UI over spike-animate. Editing a story from the page, what the page shows about
a film, running steps as background jobs, serving media so a phone can play it, and the access token."""
from __future__ import annotations

import http.client
import json
import os
import subprocess
import sys
import threading
import time
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import spike_animate as sa  # noqa: E402
import spike_studio as st  # noqa: E402

STORY = '''idea = "two robots argue over a bolt"
title = "The Last Bolt"
seed = 1024
look = "2D cartoon animation, junkyard"
still_style = "2D cartoon animation still, junkyard"

[characters]
JUNK = "a pile of scrap"

[characters.ZAP]
look = "a small yellow robot"
voice = "high, squeaky, fast"

[characters.CLANK]
look = "a tall rusty robot"
voice = "deep, gravelly, slow"
sample = "Hmph. Another day, another heap of rust."

[narrator]
voice = "warm storyteller"
sample = "Once, in a junkyard."
lead = 0.5

[music]
caption = "playful score"
structure = "[Intro - calm]\\n\\n[Outro - warm]"
key = "C major"
takes = 8
take = 2
db = -10

[dialogue]
mode = "clone"

[sound]
db = -5

[[shot]]
id = "01-heap"
still = "wide shot of JUNK under a sunset"
motion = "static camera, dust drifts"
narration = "In the scrap heap, life was quiet."
intensity = 0.2

[[shot]]
id = "02-claim"
still = "ZAP holding a bolt, talking"
motion = "static camera, a small robot talks"
seed = 77
dialogue = [{ speaker = "ZAP", line = "I saw it first! It's mine!", emotion = "smug" }]
intensity = 0.6

[[shot]]
id = "03-lunge"
still = "CLANK lunging"
motion = "a tall robot lunges"
anchor = 0
reference = { clip = "reference/lunge.mp4", start = 1.5, sigma = 0.85 }
intensity = 1.0
'''


def write_story(folder: Path, text: str = STORY) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "story.toml").write_text(text)
    return folder / "story.toml"


# --- editing the story -------------------------------------------------------

def test_a_story_round_trips_through_the_editor_unchanged():
    doc = tomllib.loads(STORY)
    assert tomllib.loads(st.doc_to_toml(doc)) == doc


def test_awkward_text_survives_the_round_trip():
    doc = tomllib.loads(STORY)
    doc["shots_note"] = 'quotes " and \\ backslashes\nand newlines'
    doc["shot"][0]["narration"] = 'She said "run" — now.'
    assert tomllib.loads(st.doc_to_toml(doc)) == doc


def test_saving_a_valid_story_writes_it(tmp_path):
    p = write_story(tmp_path / "bolt")
    doc = tomllib.loads(STORY)
    doc["shot"][0]["narration"] = "All was still."
    assert st.save_story(p.parent, doc=doc) == []
    assert sa.load_story(p).shots[0].narration == "All was still."


def test_saving_a_broken_story_is_refused_and_the_file_is_untouched(tmp_path):
    p = write_story(tmp_path / "bolt")
    doc = tomllib.loads(STORY)
    doc["shot"][1]["dialogue"][0]["speaker"] = "GHOST"
    errors = st.save_story(p.parent, doc=doc)
    assert errors and "GHOST" in errors[0]
    assert p.read_text() == STORY
    assert st.save_story(p.parent, toml="title = [unclosed")
    assert p.read_text() == STORY


def test_raw_toml_can_be_saved_as_typed(tmp_path):
    p = write_story(tmp_path / "bolt")
    text = STORY.replace("life was quiet", "nothing moved")
    assert st.save_story(p.parent, toml=text) == []
    assert p.read_text() == text            # raw edits keep the author's comments and layout


def test_picking_a_seed_and_a_score_take(tmp_path):
    p = write_story(tmp_path / "bolt")
    assert st.pick_seed(p.parent, "03-lunge", 42) == []
    assert sa.load_story(p).shots[2].seed == 42
    assert st.pick_take(p.parent, 5) == []
    assert sa.load_story(p).music.take == 5
    assert st.pick_take(p.parent, None) == []        # back to ranking every take
    assert sa.load_story(p).music.take is None
    assert st.pick_seed(p.parent, "99-nope", 1)


# --- what the page shows -------------------------------------------------------

def touch(folder: Path, *rels: str) -> None:
    for rel in rels:
        f = folder / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"x")


def test_a_fresh_story_shows_nothing_rendered(tmp_path):
    folder = write_story(tmp_path / "bolt").parent
    s = st.film_state(folder)
    assert s["title"] == "The Last Bolt" and s["errors"] == []
    assert [x["id"] for x in s["shots"]] == ["01-heap", "02-claim", "03-lunge"]
    assert all(x["still"] is None and x["clip"] is None for x in s["shots"])
    assert s["film"] is None and s["stage"] == "story"


def test_rendered_files_show_up_as_links(tmp_path):
    folder = write_story(tmp_path / "bolt").parent
    story = sa.load_story(folder / "story.toml")
    fake_still(folder, "board/retakes/01-heap-s7.png", sa.still_prompt(story, story.shots[0]))
    touch(folder, "board/01-heap.png", "board/contact-sheet.jpg",
          "shots/01-heap.mp4", "sfx/01-heap.flac", "voice/card.wav", "voice/01-heap.wav",
          "voice/cards/ZAP.wav", "voice/dialogue/02-claim-1-ZAP.wav", "score/take-s2.wav",
          "film/the-last-bolt.mp4")
    (folder / "film" / "checks.json").write_text(json.dumps({"narration": {"01-heap": 1.0}}))
    s = st.film_state(folder)
    heap = s["shots"][0]
    assert heap["still"].startswith("board/01-heap.png?v=")      # cache-busted when re-rendered
    assert heap["retakes"] == [{"seed": 7, "url": heap["retakes"][0]["url"]}]
    assert heap["clip"].startswith("shots/01-heap.mp4") and heap["sfx"].startswith("sfx/01-heap.flac")
    assert heap["narration_audio"].startswith("voice/01-heap.wav")
    assert s["shots"][1]["dialogue_audio"][0].startswith("voice/dialogue/02-claim-1-ZAP.wav")
    assert s["voices"]["narrator"].startswith("voice/card.wav")
    assert s["voices"]["ZAP"].startswith("voice/cards/ZAP.wav") and s["voices"]["CLANK"] is None
    assert [t["seed"] for t in s["score"]["takes"]] == [2] and s["score"]["pinned"] == 2
    assert s["film"].startswith("film/the-last-bolt.mp4") and s["checks"]["narration"]["01-heap"] == 1.0
    assert s["board"].startswith("board/contact-sheet.jpg")
    assert s["stage"] == "film"


def test_a_story_that_does_not_load_still_shows_its_errors(tmp_path):
    folder = write_story(tmp_path / "bolt", "title = [broken").parent
    s = st.film_state(folder)
    assert s["errors"] and s["toml"] == "title = [broken"


def test_films_lists_every_folder_with_a_story(tmp_path):
    write_story(tmp_path / "bolt")
    write_story(tmp_path / "fox", STORY.replace("The Last Bolt", "Fox Hunt"))
    (tmp_path / "loose-notes").mkdir()
    assert sorted((f["name"], f["title"]) for f in st.list_films(tmp_path)) == [("bolt", "The Last Bolt"), ("fox", "Fox Hunt")]


def test_film_names_are_plain_folder_names():
    assert st.film_name("My Penguin Film!") == "my-penguin-film"
    with pytest.raises(ValueError):
        st.film_name("../..")


def test_paths_cannot_leave_the_film_folder(tmp_path):
    folder = write_story(tmp_path / "bolt").parent
    assert st.inside(folder, "story.toml") == folder / "story.toml"
    assert st.inside(folder, "../other/story.toml") is None
    assert st.inside(folder, "/etc/passwd") is None


# --- steps run as jobs ----------------------------------------------------------

def test_each_step_becomes_a_spike_animate_command(tmp_path):
    folder = write_story(tmp_path / "bolt").parent
    story = str(folder / "story.toml")
    assert st.step_argv(folder, "board", {}) == ["board", story]
    assert st.step_argv(folder, "film", {}) == ["film", story]
    assert st.step_argv(folder, "retake", {"shot": "02-claim", "seeds": [7, 42]}) == \
        ["retake", story, "02-claim", "--seeds", "7,42"]
    assert st.step_argv(folder, "new", {"idea": "a penguin", "notes": "snowy", "shots": 6}) == \
        ["new", str(folder), "a penguin", "--shots", "6", "--notes", "snowy"]
    for bad in [("retake", {"shot": "99-nope", "seeds": [1]}), ("retake", {"shot": "02-claim", "seeds": ["1; rm"]}),
                ("explode", {})]:
        with pytest.raises(ValueError):
            st.step_argv(folder, *bad)


FAKE_ANIMATE = '''#!/bin/sh
echo "fake $1 starting"
sleep "${FAKE_SLEEP:-0}"
echo "fake $1 done"
[ "$1" = film ] && exit 3
exit 0
'''


@pytest.fixture
def fake_bin(tmp_path, monkeypatch):
    b = tmp_path / "fake-animate"
    b.write_text(FAKE_ANIMATE)
    b.chmod(0o755)
    monkeypatch.setenv("SPIKE_ANIMATE_BIN", str(b))
    return b


def wait_done(jobs, folder, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        j = jobs.status(folder)
        if j and not j["running"]:
            return j
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_a_job_runs_in_the_background_and_reports_its_result(tmp_path, fake_bin):
    folder = write_story(tmp_path / "bolt").parent
    jobs = st.Jobs()
    jobs.start(folder, ["board", str(folder / "story.toml")])
    j = wait_done(jobs, folder)
    assert j["step"] == "board" and j["rc"] == 0 and "fake board done" in j["log"]
    jobs.start(folder, ["film", str(folder / "story.toml")])
    assert wait_done(jobs, folder)["rc"] == 3


def test_one_job_per_film_at_a_time(tmp_path, fake_bin, monkeypatch):
    monkeypatch.setenv("FAKE_SLEEP", "1")
    folder = write_story(tmp_path / "bolt").parent
    jobs = st.Jobs()
    jobs.start(folder, ["board", str(folder / "story.toml")])
    with pytest.raises(st.Busy):
        jobs.start(folder, ["film", str(folder / "story.toml")])
    wait_done(jobs, folder)


def test_a_job_result_survives_a_studio_restart(tmp_path, fake_bin):
    folder = write_story(tmp_path / "bolt").parent
    st.Jobs().start(folder, ["board", str(folder / "story.toml")])
    j = wait_done(st.Jobs(), folder)         # a new Jobs knows nothing in memory, only what's on disk
    assert j["rc"] == 0 and j["step"] == "board"


# --- the server -----------------------------------------------------------------

@pytest.fixture
def server(tmp_path, fake_bin):
    films = tmp_path / "films"
    write_story(films / "bolt")
    touch(films / "bolt", "film/the-last-bolt.mp4")
    (films / "bolt" / "film" / "the-last-bolt.mp4").write_bytes(bytes(range(256)) * 4)
    srv = st.make_server(films, token="sekrit", host="127.0.0.1", port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv, films
    srv.shutdown()


def call(srv, method, path, body=None, headers=None, cookie=True):
    c = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=10)
    h = dict(headers or {})
    if cookie:
        h["Cookie"] = "studio=sekrit"
    data = json.dumps(body).encode() if body is not None else None
    if data:
        h["Content-Type"] = "application/json"
    c.request(method, path, body=data, headers=h)
    r = c.getresponse()
    return r.status, dict(r.getheaders()), r.read()


def test_the_studio_needs_the_token(server):
    srv, _ = server
    assert call(srv, "GET", "/api/films", cookie=False)[0] == 401
    status, headers, _ = call(srv, "GET", "/?t=sekrit", cookie=False)
    assert status == 200 and "studio=sekrit" in headers["Set-Cookie"] and "HttpOnly" in headers["Set-Cookie"]
    assert call(srv, "GET", "/?t=wrong", cookie=False)[0] == 401


def test_the_api_lists_shows_and_edits_films(server):
    srv, films = server
    status, _, body = call(srv, "GET", "/api/films")
    assert status == 200 and json.loads(body)[0]["name"] == "bolt"
    state = json.loads(call(srv, "GET", "/api/films/bolt")[2])
    assert state["title"] == "The Last Bolt"
    doc = state["doc"]
    doc["title"] = "The Very Last Bolt"
    assert call(srv, "PUT", "/api/films/bolt/story", {"doc": doc})[0] == 200
    assert sa.load_story(films / "bolt" / "story.toml").title == "The Very Last Bolt"
    doc["shot"][0]["id"] = doc["shot"][1]["id"]
    status, _, body = call(srv, "PUT", "/api/films/bolt/story", {"doc": doc})
    assert status == 400 and "duplicate" in json.loads(body)["errors"][0]
    assert call(srv, "GET", "/api/films/nope")[0] == 404


def test_the_api_runs_steps(server):
    srv, films = server
    assert call(srv, "POST", "/api/films/bolt/approve", {"stage": "story"})[0] == 200
    status, _, body = call(srv, "POST", "/api/films/bolt/run", {"step": "board"})
    assert status == 202
    end = time.time() + 10
    while time.time() < end:
        job = json.loads(call(srv, "GET", "/api/films/bolt")[2])["job"]
        if job and not job["running"]:
            break
        time.sleep(0.05)
    assert job["rc"] == 0 and "fake board done" in job["log"]
    assert call(srv, "POST", "/api/films/bolt/run", {"step": "retake", "shot": "99-x", "seeds": [1]})[0] == 400


def test_a_new_film_starts_from_an_idea(server):
    srv, films = server
    status, _, body = call(srv, "POST", "/api/films", {"title": "Penguin Shop", "idea": "a penguin sells ice cream",
                                                        "notes": "", "shots": 6})
    assert status == 202 and json.loads(body)["name"] == "penguin-shop"
    assert (films / "penguin-shop").is_dir()
    assert call(srv, "POST", "/api/films", {"title": "bolt", "idea": "x", "shots": 6})[0] == 409


def test_media_plays_on_a_phone(server):
    srv, _ = server
    status, headers, body = call(srv, "GET", "/media/bolt/film/the-last-bolt.mp4", headers={"Range": "bytes=10-19"})
    assert status == 206 and body == bytes(range(10, 20))
    assert headers["Content-Range"] == "bytes 10-19/1024" and headers["Content-Type"] == "video/mp4"
    assert headers["Accept-Ranges"] == "bytes"
    status, headers, body = call(srv, "GET", "/media/bolt/film/the-last-bolt.mp4")
    assert status == 200 and len(body) == 1024
    assert call(srv, "GET", "/media/bolt/../../etc/passwd")[0] == 404


def test_reference_clips_can_be_uploaded(server):
    srv, films = server
    c = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=10)
    c.request("POST", "/api/films/bolt/upload?name=Fox%20Pounce.mp4", body=b"clipdata",
              headers={"Cookie": "studio=sekrit", "Content-Type": "video/mp4"})
    r = c.getresponse()
    assert r.status == 201 and json.loads(r.read())["path"] == "reference/fox-pounce.mp4"
    assert (films / "bolt" / "reference" / "fox-pounce.mp4").read_bytes() == b"clipdata"
    state = json.loads(call(srv, "GET", "/api/films/bolt")[2])
    assert "reference/fox-pounce.mp4" in state["references"]


def test_a_film_whose_story_never_got_written_can_be_started_again(server):
    srv, films = server
    (films / "penguin-shop").mkdir()                 # a draft that failed: folder, no story
    status, _, body = call(srv, "POST", "/api/films", {"title": "Penguin Shop", "idea": "a penguin", "shots": 6})
    assert status == 202 and json.loads(body)["name"] == "penguin-shop"


# --- one step at a time: each step is approved before the next unlocks ---------------------

def fake_still(folder: Path, rel: str, prompt: str) -> Path:
    p = folder / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\x89PNG" + json.dumps({"mflux_version": "fake", "prompt": prompt}).encode())
    return p


def render_board(folder: Path) -> None:
    story = sa.load_story(folder / "story.toml")
    keys = sa.Keys(folder)
    for shot in story.shots:
        out = fake_still(folder, f"board/{shot.id}.png", sa.still_prompt(story, shot))
        keys.record(out, sa.still_key(story, shot))


def stages(folder: Path) -> dict:
    return {s["stage"]: s for s in st.film_state(folder)["steps"]}


def open_scenes(folder: Path) -> None:
    """Approve the story, draw and approve every look and every voice, so the scenes open."""
    assert st.approve(folder, "story") == []
    story = sa.load_story(folder / "story.toml")
    for name in sa.look_names(story):
        draw_look(folder, name)
        assert st.approve(folder, f"look:{name}") == []
    for name in sa.voice_names(story):
        design(folder, name, 1)
        assert st.use_voice(folder, name, 1) == []
        assert st.approve(folder, f"voice:{name}") == []


def make(folder: Path, sid: str, part: str) -> None:
    """Stand in for spike-animate making one part of one scene, with the keys it would record."""
    story = sa.load_story(folder / "story.toml")
    keys = sa.Keys(folder)
    shot = next(s for s in story.shots if s.id == sid)
    time.sleep(0.01)
    if part == "picture":
        keys.record(fake_still(folder, f"board/{sid}.png", sa.still_prompt(story, shot)), sa.still_key(story, shot))
    elif part == "video":
        if shot.reference and not Path(shot.reference.clip).exists():
            touch(folder, str(Path(shot.reference.clip).relative_to(folder)))
        touch(folder, f"shots/{sid}.mp4")
        keys.record(folder / "shots" / f"{sid}.mp4", sa.shot_key(story, shot))
    elif part == "sound":
        touch(folder, f"sfx/{sid}.flac")
        keys.record(sa.sfx_path(story, shot), sa.sfx_key(story, shot))
    elif part == "voices":
        if shot.narration:
            if not (folder / "voice" / "card.wav").exists():
                touch(folder, "voice/card.wav")
            touch(folder, f"voice/{sid}.wav")
            keys.record(folder / "voice" / "card.wav", sa.narrator_card_key(story))
            keys.record(folder / "voice" / f"{sid}.wav", sa.narration_key(story, shot))
        for n, line in enumerate(shot.dialogue, 1):
            if not (folder / "voice" / "cards" / f"{line.speaker}.wav").exists():   # the voice chosen in the Look step
                touch(folder, f"voice/cards/{line.speaker}.wav")
            touch(folder, f"voice/dialogue/{sid}-{n}-{line.speaker}.wav")
            keys.record(folder / "voice" / "cards" / f"{line.speaker}.wav", sa.character_card_key(story, line.speaker))
            keys.record(folder / "voice" / "dialogue" / f"{sid}-{n}-{line.speaker}.wav", sa.dialogue_key(story, line))
    elif part == "preview":
        touch(folder, f"scenes/{sid}.mp4")


def finish(folder: Path, sid: str) -> None:
    """Make and approve every part of one scene, in order."""
    for part in st.SCENE_PARTS:
        s = stages(folder)[f"scene:{sid}:{part}"]
        if s["skip"]:
            continue
        make(folder, sid, part)
        assert st.approve(folder, f"scene:{sid}:{part}") == [], part


IDS = ("01-heap", "02-claim", "03-lunge")


def test_only_the_story_is_open_at_first(tmp_path):
    folder = write_story(tmp_path / "bolt").parent
    s = stages(folder)
    assert list(s) == ["story", "look:JUNK", "look:ZAP", "look:CLANK", "voice:narrator", "voice:ZAP", "voice:CLANK"] + \
        [f"scene:{i}:{p}" for i in IDS for p in st.SCENE_PARTS] + ["music", "film"]
    assert s["story"]["unlocked"] and not s["story"]["approved"]
    assert not any(v["unlocked"] for k, v in s.items() if k != "story")
    assert st.gate(folder, "board", {"shot": "01-heap"}) is not None and st.gate(folder, "film") is not None


def test_a_scene_is_made_one_part_at_a_time(tmp_path):
    folder = write_story(tmp_path / "bolt").parent
    open_scenes(folder)
    assert st.gate(folder, "board", {"shot": "01-heap"}) is None
    assert st.gate(folder, "retake", {"shot": "01-heap"}) is None
    assert st.gate(folder, "video", {"shot": "01-heap"}) is not None          # approve its picture first
    assert st.gate(folder, "board", {"shot": "02-claim"}) is not None         # finish scene 1 first
    assert st.approve(folder, "scene:01-heap:picture") != []                   # nothing drawn yet
    make(folder, "01-heap", "picture")
    assert st.approve(folder, "scene:01-heap:picture") == []
    assert st.gate(folder, "video", {"shot": "01-heap"}) is None
    make(folder, "01-heap", "video")
    assert st.approve(folder, "scene:01-heap:video") == []
    s = stages(folder)
    assert s["scene:01-heap:sound"]["skip"] and s["scene:01-heap:sound"]["approved"]   # no sounds written
    assert s["scene:01-heap:voices"]["unlocked"] and not s["scene:01-heap:voices"]["skip"]
    make(folder, "01-heap", "voices")
    assert st.approve(folder, "scene:01-heap:voices") == []
    assert st.gate(folder, "scene", {"shot": "01-heap"}) is None
    make(folder, "01-heap", "preview")
    assert st.approve(folder, "scene:01-heap:preview") == []
    assert st.gate(folder, "board", {"shot": "02-claim"}) is None               # on to scene 2


def test_changing_a_picture_reopens_only_that_scene(tmp_path):
    folder = write_story(tmp_path / "bolt").parent
    open_scenes(folder)
    finish(folder, "01-heap")
    finish(folder, "02-claim")
    doc = st._doc(folder)
    doc["shot"][0]["still"] = "close-up of JUNK at night"
    st.save_story(folder, doc=doc)
    s = stages(folder)
    assert not s["scene:01-heap:picture"]["done"]
    assert not any(s[f"scene:01-heap:{p}"]["approved"] for p in ("picture", "video", "voices", "preview"))
    assert all(s[f"scene:02-claim:{p}"]["approved"] for p in st.SCENE_PARTS)    # scene 2 is untouched
    assert s["scene:02-claim:picture"]["unlocked"]                              # and can still be reopened
    make(folder, "01-heap", "picture")
    assert st.approve(folder, "scene:01-heap:picture") == []
    assert not stages(folder)["scene:01-heap:video"]["done"]                     # made from the old picture


def test_an_edited_line_must_be_recorded_again(tmp_path):
    folder = write_story(tmp_path / "bolt").parent
    open_scenes(folder)
    finish(folder, "01-heap")
    finish(folder, "02-claim")
    doc = st._doc(folder)
    doc["shot"][1]["dialogue"][0]["line"] = "Mine! I saw it first!"
    st.save_story(folder, doc=doc)
    s = stages(folder)
    assert not s["scene:02-claim:voices"]["done"] and not s["scene:02-claim:preview"]["approved"]
    assert s["scene:02-claim:video"]["approved"] and s["scene:01-heap:voices"]["approved"]


def test_music_and_the_film_wait_for_every_scene(tmp_path):
    folder = write_story(tmp_path / "bolt").parent
    open_scenes(folder)
    finish(folder, "01-heap")
    finish(folder, "02-claim")
    assert st.gate(folder, "music") is not None
    finish(folder, "03-lunge")
    assert st.gate(folder, "music") is None and st.gate(folder, "film") is not None
    touch(folder, "score/take-s2.wav")
    assert st.approve(folder, "music") == []
    assert st.gate(folder, "film") is None


def test_no_music_means_the_music_step_is_skipped(tmp_path):
    folder = write_story(tmp_path / "bolt", STORY.split("[music]")[0] + "[dialogue]" + STORY.split("[dialogue]")[1]).parent
    open_scenes(folder)
    for sid in IDS:
        finish(folder, sid)
    s = stages(folder)
    assert s["music"]["skip"] and s["music"]["approved"] and st.gate(folder, "film") is None


def test_each_scene_step_becomes_a_spike_animate_command(tmp_path):
    folder = write_story(tmp_path / "bolt").parent
    story = str(folder / "story.toml")
    for step in ("board", "video", "sound", "voices"):
        assert st.step_argv(folder, step, {"shot": "02-claim"}) == [step, story, "--shot", "02-claim"]
    assert st.step_argv(folder, "scene", {"shot": "02-claim"}) == ["scene", story, "02-claim"]
    for bad in [("video", {"shot": "99-nope"}), ("scene", {})]:
        with pytest.raises(ValueError):
            st.step_argv(folder, *bad)


def test_the_page_shows_what_each_line_was_heard_as_and_the_scene_preview(tmp_path):
    folder = write_story(tmp_path / "bolt").parent
    touch(folder, "voice/dialogue/02-claim-1-ZAP.wav", "scenes/02-claim.mp4")
    (folder / "voice" / "dialogue" / "02-claim-1-ZAP.heard.txt").write_text("I saw it first. It's mine.")
    (folder / "voice" / "01-heap.heard.txt").write_text("In the scrap heap")
    s1, s2 = st.film_state(folder)["shots"][:2]
    assert s2["dialogue_heard"] == ["I saw it first. It's mine."] and s1["narration_heard"] == "In the scrap heap"
    assert s2["preview"].startswith("scenes/02-claim.mp4")


# --- many rounds of pictures --------------------------------------------------------------

def test_using_a_retake_takes_effect_at_once_without_redrawing(tmp_path):
    folder = write_story(tmp_path / "bolt").parent
    render_board(folder)
    story = sa.load_story(folder / "story.toml")
    shot = story.shots[1]
    fake_still(folder, "board/retakes/02-claim-s555.png", sa.still_prompt(story, shot))
    assert st.use_retake(folder, "02-claim", 555) == []
    assert st._doc(folder)["shot"][1]["seed"] == 555
    assert (folder / "board" / "02-claim.png").read_bytes() == (folder / "board/retakes/02-claim-s555.png").read_bytes()
    s = st.film_state(folder)["shots"][1]
    assert s["still_fresh"] is True


def test_retakes_of_an_old_description_are_not_offered(tmp_path):
    folder = write_story(tmp_path / "bolt").parent
    story = sa.load_story(folder / "story.toml")
    fake_still(folder, "board/retakes/02-claim-s1.png", "an older description, cartoon")
    fake_still(folder, "board/retakes/02-claim-s2.png", sa.still_prompt(story, story.shots[1]))
    assert [r["seed"] for r in st.film_state(folder)["shots"][1]["retakes"]] == [2]
    assert st.use_retake(folder, "02-claim", 1) != []


# --- starting a film from a story and a style picture --------------------------------------

def test_new_can_read_a_style_picture(tmp_path):
    folder = tmp_path / "pip"
    touch(folder, "reference/style.png")
    argv = st.step_argv(folder, "new", {"idea": "Pip paints.", "shots": 4, "style_image": "reference/style.png"})
    assert argv[-2:] == ["--style-image", str(folder / "reference" / "style.png")]
    for bad in ("../outside.png", "reference/missing.png", "reference/clip.mp4"):
        touch(folder, "reference/clip.mp4")
        with pytest.raises(ValueError):
            st.step_argv(folder, "new", {"idea": "x", "shots": 4, "style_image": bad})


def test_a_film_can_be_made_first_and_written_after(server):
    srv, films = server
    status, _, body = call(srv, "POST", "/api/films", {"title": "Pip Paints"})
    assert status == 201 and json.loads(body)["name"] == "pip-paints"
    assert (films / "pip-paints").is_dir() and not (films / "pip-paints" / "story.toml").exists()
    status, _, _ = call(srv, "POST", "/api/films/pip-paints/run", {"step": "new", "idea": "Pip paints.", "shots": 4})
    assert status == 202


def test_the_api_approves_steps_and_refuses_locked_ones(server):
    srv, films = server
    status, _, body = call(srv, "POST", "/api/films/bolt/run", {"step": "video", "shot": "01-heap"})
    assert status == 409 and "approve" in json.loads(body)["errors"][0].lower()
    status, _, body = call(srv, "POST", "/api/films/bolt/approve", {"stage": "story"})
    assert status == 200 and {s["stage"]: s for s in json.loads(body)["steps"]}["story"]["approved"]
    assert call(srv, "POST", "/api/films/bolt/approve", {"stage": "scene:01-heap:picture"})[0] == 400
    assert call(srv, "POST", "/api/films/bolt/approve", {"stage": "scene:99-x:picture"})[0] == 400
    assert call(srv, "POST", "/api/films/bolt/approve", {"stage": "story", "undo": True})[0] == 200
    assert not stages(films / "bolt")["story"]["approved"]


def test_a_new_film_can_leave_out_the_narrator_dialogue_and_music(tmp_path):
    folder = tmp_path / "pip"
    argv = st.step_argv(folder, "new", {"idea": "Pip paints.", "shots": 4, "narrator": False, "dialogue": False, "music": False})
    assert argv[-3:] == ["--no-narrator", "--no-dialogue", "--no-music"]
    assert "--no-narrator" not in st.step_argv(folder, "new", {"idea": "Pip paints.", "shots": 4})


# --- new takes of a scene's motion, and trimming it -----------------------------------------

def take(folder: Path, sid: str, seed: int, body: bytes) -> Path:
    """A take of a shot's motion as spike-animate keeps it."""
    story = sa.load_story(folder / "story.toml")
    shot = next(s for s in story.shots if s.id == sid)
    p = sa.take_path(story, shot, seed)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(body)
    sa.Keys(folder).record(p, sa.shot_key(story, shot, seed))
    return p


def test_animate_again_asks_for_a_new_take(tmp_path):
    folder = write_story(tmp_path / "bolt").parent
    render_board(folder)
    seeds = set()
    for _ in range(3):
        assert st.new_take(folder, "01-heap") == []
        seeds.add(st._doc(folder)["shot"][0]["motion_seed"])
    assert len(seeds) == 3 and 1024 not in seeds
    assert st.new_take(folder, "99-x") != []


def test_an_earlier_take_can_be_used_again_at_once(tmp_path):
    folder = write_story(tmp_path / "bolt").parent
    open_scenes(folder)
    make(folder, "01-heap", "picture")
    take(folder, "01-heap", 1024, b"first take")
    take(folder, "01-heap", 555, b"second take")
    state = st.film_state(folder)["shots"][0]
    assert [t["seed"] for t in state["takes"]] == [555, 1024]
    assert st.use_take(folder, "01-heap", 555) == []
    assert st._doc(folder)["shot"][0]["motion_seed"] == 555
    assert (folder / "shots" / "01-heap.mp4").read_bytes() == b"second take"
    assert stages(folder)["scene:01-heap:video"]["done"]
    assert st.use_take(folder, "01-heap", 1024) == []           # back to the first one
    assert "motion_seed" not in st._doc(folder)["shot"][0]
    assert (folder / "shots" / "01-heap.mp4").read_bytes() == b"first take"
    assert st.use_take(folder, "01-heap", 9) != []


def test_takes_of_an_old_picture_are_not_offered(tmp_path):
    folder = write_story(tmp_path / "bolt").parent
    make(folder, "01-heap", "picture")
    take(folder, "01-heap", 1024, b"made from the first picture")
    doc = st._doc(folder)
    doc["shot"][0]["still"] = "close-up of JUNK at night"
    st.save_story(folder, doc=doc)
    make(folder, "01-heap", "picture")
    assert st.film_state(folder)["shots"][0]["takes"] == []


def test_the_api_starts_a_new_take(server):
    srv, films = server
    folder = films / "bolt"
    open_scenes(folder)
    make(folder, "01-heap", "picture")
    st.approve(folder, "scene:01-heap:picture")
    status, _, _ = call(srv, "POST", "/api/films/bolt/run", {"step": "video", "shot": "01-heap", "new_take": True})
    assert status == 202 and "motion_seed" in st._doc(folder)["shot"][0]


def test_draw_again_asks_for_a_new_picture_and_keeps_the_old_one_as_a_version(tmp_path):
    folder = write_story(tmp_path / "bolt").parent
    render_board(folder)
    old = (folder / "board" / "01-heap.png").read_bytes()
    seeds = set()
    for _ in range(3):
        assert st.new_picture(folder, "01-heap") == []
        seeds.add(st._doc(folder)["shot"][0]["seed"])
    assert len(seeds) == 3 and 1024 not in seeds
    assert (folder / "board" / "retakes" / "01-heap-s1024.png").read_bytes() == old
    assert 1024 in [r["seed"] for r in st.film_state(folder)["shots"][0]["retakes"]]   # offered to go back to
    assert st.new_picture(folder, "99-x") != []


def test_the_api_draws_a_new_picture(server):
    srv, films = server
    folder = films / "bolt"
    open_scenes(folder)
    make(folder, "01-heap", "picture")
    status, _, _ = call(srv, "POST", "/api/films/bolt/run", {"step": "board", "shot": "01-heap", "new_picture": True})
    assert status == 202 and st._doc(folder)["shot"][0]["seed"] != 1024


def test_draw_again_asks_for_a_new_look(tmp_path):
    folder = write_story(tmp_path / "bolt").parent
    st.approve(folder, "story")
    draw_look(folder, "ZAP")
    first = sa.look_seed(sa.load_story(folder / "story.toml"), "ZAP")
    assert st.new_look(folder, "ZAP") == []
    story = sa.load_story(folder / "story.toml")
    assert sa.look_seed(story, "ZAP") != first
    assert first in [t["seed"] for t in next(l for l in st.film_state(folder)["looks"] if l["name"] == "ZAP")["takes"]]
    assert st.new_look(folder, "NOBODY") != []


def test_a_shot_made_before_takes_existed_is_kept_when_animating_again(tmp_path):
    folder = write_story(tmp_path / "bolt").parent
    make(folder, "01-heap", "picture")
    make(folder, "01-heap", "video")                       # the shot alone, as older films have it
    (folder / "shots" / "01-heap.mp4").write_bytes(b"the old shot")
    story = sa.load_story(folder / "story.toml")
    sa.Keys(folder).record(folder / "shots" / "01-heap.mp4", sa.shot_key(story, story.shots[0]))
    assert st.new_take(folder, "01-heap") == []
    assert (folder / "shots" / "takes" / "01-heap-m1024.mp4").read_bytes() == b"the old shot"


# --- the film bible -------------------------------------------------------------------------

def test_the_page_shows_whether_a_picture_keeps_the_film_bible(tmp_path):
    folder = write_story(tmp_path / "bolt", STORY.replace("seed = 1024\n", 'seed = 1024\nlighting = "sunset, orange light"\n', 1)).parent
    make(folder, "02-claim", "picture")
    story = sa.load_story(folder / "story.toml")
    shot = story.shots[1]
    assert st.film_state(folder)["shots"][1]["continuity"] is None          # not checked yet
    out = sa.check_path(story, shot)
    out.write_text(json.dumps({"misses": ["ZAP is red, not yellow"]}))
    sa.Keys(folder).record(out, sa.key_for("check", sa.file_digest(sa.still_path(story, shot)), sa.continuity_prompt(story, shot)))
    assert st.film_state(folder)["shots"][1]["continuity"] == ["ZAP is red, not yellow"]
    make(folder, "02-claim", "picture")                                      # a new picture: the old check no longer applies
    (folder / "board" / "02-claim.png").write_bytes(b"\x89PNG another picture")
    assert st.film_state(folder)["shots"][1]["continuity"] is None
    assert st.step_argv(folder, "check", {"shot": "02-claim"}) == ["check", str(folder / "story.toml"), "--shot", "02-claim"]


def test_a_film_bible_survives_the_editor(tmp_path):
    text = STORY.replace("seed = 1024\n", 'seed = 1024\nlighting = "sunset, orange light from the left"\n', 1) + '''
[places]
HEAP = "a mountain of rusty scrap under a red sky"

[angles.LOW]
frame = "low wide shot from the ground"
seed = 2024
'''
    text = text.replace('id = "01-heap"\n', 'id = "01-heap"\nangle = "LOW"\n')
    doc = tomllib.loads(text)
    folder = tmp_path / "bolt"
    folder.mkdir()
    assert st.save_story(folder, doc=doc) == []
    assert tomllib.loads((folder / "story.toml").read_text()) == doc
    story = sa.load_story(folder / "story.toml")
    assert story.shots[0].seed == 2024 and story.places["HEAP"].startswith("a mountain")


def test_a_scene_shows_the_seed_its_picture_is_drawn_with(tmp_path):
    text = STORY + '\n[angles.LOW]\nframe = "low wide shot from the ground"\nseed = 2024\n'
    folder = write_story(tmp_path / "bolt", text.replace('id = "01-heap"\n', 'id = "01-heap"\nangle = "LOW"\n')).parent
    assert [s["seed"] for s in st.film_state(folder)["shots"]] == [2024, 77, 1024]   # the angle's, its own, the film's
    assert st.new_picture(folder, "01-heap") == []
    assert st.film_state(folder)["shots"][0]["seed"] not in (2024, 1024)


# --- deleting a film ------------------------------------------------------------------------

def test_deleting_a_film_moves_it_to_the_trash(tmp_path, monkeypatch):
    trash = tmp_path / "Trash"
    trash.mkdir()
    monkeypatch.setattr(st, "TRASH", trash)
    folder = write_story(tmp_path / "films" / "bolt").parent
    touch(folder, "board/01-heap.png", "shots/01-heap.mp4")
    assert st.trash_film(folder) == []
    assert not folder.exists() and (trash / "bolt" / "shots" / "01-heap.mp4").exists()
    again = write_story(tmp_path / "films" / "bolt").parent       # a second film of the same name
    assert st.trash_film(again) == []
    assert sorted(p.name for p in trash.iterdir()) == ["bolt", "bolt 2"]


def test_the_api_deletes_a_film_but_not_while_it_is_working(server, monkeypatch, tmp_path):
    srv, films = server
    trash = tmp_path / "Trash"
    trash.mkdir()
    monkeypatch.setattr(st, "TRASH", trash)
    monkeypatch.setenv("FAKE_SLEEP", "3")
    st.approve(films / "bolt", "story")
    assert call(srv, "POST", "/api/films/bolt/run", {"step": "board"})[0] == 202
    status, _, body = call(srv, "DELETE", "/api/films/bolt")
    assert status == 409 and "running" in json.loads(body)["errors"][0]
    call(srv, "POST", "/api/films/bolt/stop")
    time.sleep(0.5)
    assert call(srv, "DELETE", "/api/films/bolt")[0] == 200
    assert not (films / "bolt").exists() and (trash / "bolt").exists()
    assert call(srv, "GET", "/api/films/bolt")[0] == 404


# --- the Look step ----------------------------------------------------------------------------

def draw_look(folder: Path, name: str, seed: int | None = None) -> None:
    """Stand in for `spike-animate look`: the picture, and the key it records."""
    story = sa.load_story(folder / "story.toml")
    keys = sa.Keys(folder)
    seed = sa.look_seed(story, name) if seed is None else seed
    take = sa.look_take_path(story, name, seed)
    fake_still(folder, str(take.relative_to(folder)), f"{name} {seed}")
    keys.record(take, sa.look_key(story, name, seed))
    if seed == sa.look_seed(story, name):
        out = sa.look_path(story, name)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(take.read_bytes())
        keys.record(out, sa.look_key(story, name))


def test_every_character_and_place_has_a_look_to_approve_before_the_scenes(tmp_path):
    folder = write_story(tmp_path / "bolt", STORY + '\n[places]\nHEAP = "a mountain of rusty scrap"\n').parent
    s = stages(folder)
    looks = [k for k in s if k.startswith("look:")]
    assert looks == ["look:JUNK", "look:ZAP", "look:CLANK", "look:HEAP"]
    assert list(s).index("look:HEAP") < list(s).index("scene:01-heap:picture")
    st.approve(folder, "story")
    assert all(stages(folder)[k]["unlocked"] for k in looks)                 # in any order
    assert st.gate(folder, "board", {"shot": "01-heap"}) is not None
    for name in ("JUNK", "ZAP", "CLANK", "HEAP"):
        assert st.gate(folder, "look", {"name": name}) is None
        draw_look(folder, name)
        assert st.approve(folder, f"look:{name}") == []
    for name in ("narrator", "ZAP", "CLANK"):
        design(folder, name, 1)
        st.use_voice(folder, name, 1)
        assert st.approve(folder, f"voice:{name}") == []
    assert st.gate(folder, "board", {"shot": "01-heap"}) is None
    assert st.step_argv(folder, "look", {"name": "ZAP"}) == ["look", str(folder / "story.toml"), "ZAP"]
    assert st.step_argv(folder, "look", {"name": "ZAP", "seeds": [4, 5]})[-2:] == ["--seeds", "4,5"]
    with pytest.raises(ValueError):
        st.step_argv(folder, "look", {"name": "NOPE"})


def test_a_new_look_reopens_the_scenes_drawn_from_it(tmp_path):
    folder = write_story(tmp_path / "bolt").parent
    open_scenes(folder)
    finish(folder, "01-heap")
    finish(folder, "02-claim")
    draw_look(folder, "ZAP", 99)
    assert st.use_look(folder, "ZAP", 99) == []
    assert st._doc(folder)["looks"]["ZAP"] == 99
    s = stages(folder)
    assert not s["look:ZAP"]["approved"] and s["look:ZAP"]["done"]
    assert not s["scene:02-claim:picture"]["done"]                          # ZAP is in it: draw it again
    assert s["scene:01-heap:picture"]["approved"]                           # JUNK only: untouched
    assert st.use_look(folder, "ZAP", 12345) != []


def test_a_character_can_be_drawn_from_an_uploaded_picture(tmp_path):
    folder = write_story(tmp_path / "bolt").parent
    touch(folder, "reference/zap.jpg")                                       # uploads are saved in lower case
    story = sa.load_story(folder / "story.toml")
    assert sa.look_reference(story, "ZAP") == folder / "reference" / "zap.jpg"
    look = next(x for x in st.film_state(folder)["looks"] if x["name"] == "ZAP")
    assert look["kind"] == "character" and look["reference"].startswith("reference/zap.jpg")


# --- voices chosen in the Look step -------------------------------------------------------------

def design(folder: Path, name: str, n: int) -> None:
    """Stand in for one take of `spike-animate voice`."""
    story = sa.load_story(folder / "story.toml")
    take = sa.voice_take_path(story, name, n)
    touch(folder, str(take.relative_to(folder)))
    take.write_bytes(f"{name} take {n}".encode())
    sa.Keys(folder).record(take, sa.card_key(story, name))


def test_voices_are_chosen_in_the_look_step(tmp_path):
    folder = write_story(tmp_path / "bolt").parent
    s = stages(folder)
    assert [k for k in s if k.startswith("voice:")] == ["voice:narrator", "voice:ZAP", "voice:CLANK"]
    st.approve(folder, "story")
    assert st.gate(folder, "voice", {"name": "ZAP"}) is None
    assert st.step_argv(folder, "voice", {"name": "ZAP", "takes": 3})[-3:] == ["ZAP", "--takes", "3"]
    design(folder, "ZAP", 1)
    design(folder, "ZAP", 2)
    assert st.use_voice(folder, "ZAP", 2) == []
    assert (folder / "voice" / "cards" / "ZAP.wav").read_bytes() == b"ZAP take 2"
    assert stages(folder)["voice:ZAP"]["done"]
    zap = next(v for v in st.film_state(folder)["voice_looks"] if v["name"] == "ZAP")
    assert [t["n"] for t in zap["takes"]] == [1, 2] and zap["in_use"] == 2
    assert st.use_voice(folder, "ZAP", 7) != []


def test_scenes_wait_for_the_voices_too(tmp_path):
    folder = write_story(tmp_path / "bolt").parent
    st.approve(folder, "story")
    for name in ("JUNK", "ZAP", "CLANK"):
        draw_look(folder, name)
        st.approve(folder, f"look:{name}")
    assert st.gate(folder, "board", {"shot": "01-heap"}) is not None
    for name in ("narrator", "ZAP", "CLANK"):
        design(folder, name, 1)
        assert st.use_voice(folder, name, 1) == []
        assert st.approve(folder, f"voice:{name}") == []
    assert st.gate(folder, "board", {"shot": "01-heap"}) is None


# --- adding and removing scenes -----------------------------------------------------------------

def test_a_scene_added_after_another_takes_its_place_in_the_order(tmp_path):
    folder = write_story(tmp_path / "bolt").parent
    errors, sid = st.add_scene(folder, after="01-heap", still="BOLT sniffing the ground", motion="static camera, BOLT sniffs")
    assert errors == []
    shots = sa.load_story(folder / "story.toml").shots
    assert [s.id for s in shots] == ["01-heap", sid, "02-claim", "03-lunge"]
    added = shots[1]
    assert added.still == "BOLT sniffing the ground" and added.motion == "static camera, BOLT sniffs"


def test_a_scene_added_without_a_place_goes_last_and_never_reuses_an_id(tmp_path):
    folder = write_story(tmp_path / "bolt").parent
    first = st.add_scene(folder, after=None, still="a wide sky", motion="static camera")[1]
    second = st.add_scene(folder, after=None, still="a wide sky", motion="static camera")[1]
    ids = [s.id for s in sa.load_story(folder / "story.toml").shots]
    assert ids[-2:] == [first, second] and len(set(ids)) == len(ids)


def test_a_scene_needs_a_picture_description(tmp_path):
    folder = write_story(tmp_path / "bolt").parent
    before = (folder / "story.toml").read_text()
    errors, sid = st.add_scene(folder, after=None, still="  ", motion="static camera")
    assert errors and "picture" in errors[0] and sid is None
    assert (folder / "story.toml").read_text() == before


def test_a_new_scene_opens_once_the_one_before_is_finished_and_others_stay_approved(tmp_path):
    folder = write_story(tmp_path / "bolt").parent
    open_scenes(folder)
    finish(folder, "01-heap")
    finish(folder, "02-claim")
    sid = st.add_scene(folder, after="01-heap", still="BOLT sniffing the ground", motion="static camera, BOLT sniffs")[1]
    s = stages(folder)
    assert s[f"scene:{sid}:picture"]["unlocked"] and not s[f"scene:{sid}:picture"]["approved"]
    assert all(s[f"scene:{x}:{p}"]["approved"] for x in ("01-heap", "02-claim") for p in st.SCENE_PARTS)


def test_removing_a_scene_drops_it_and_its_approvals_and_opens_the_next(tmp_path):
    folder = write_story(tmp_path / "bolt").parent
    open_scenes(folder)
    finish(folder, "01-heap")
    finish(folder, "02-claim")
    assert st.remove_scene(folder, "02-claim") == []
    assert [x.id for x in sa.load_story(folder / "story.toml").shots] == ["01-heap", "03-lunge"]
    assert not any(k.startswith("scene:02-claim:") for k in st._saved(folder))
    s = stages(folder)
    assert s["scene:03-lunge:picture"]["unlocked"]
    assert all(s[f"scene:01-heap:{p}"]["approved"] for p in st.SCENE_PARTS)


def test_the_last_scene_and_unknown_scenes_cannot_be_removed(tmp_path):
    folder = write_story(tmp_path / "bolt").parent
    assert "no scene" in st.remove_scene(folder, "99-nope")[0]
    assert st.remove_scene(folder, "01-heap") == [] and st.remove_scene(folder, "02-claim") == []
    assert "only scene" in st.remove_scene(folder, "03-lunge")[0]
    assert [x.id for x in sa.load_story(folder / "story.toml").shots] == ["03-lunge"]


def test_the_api_adds_and_removes_scenes_but_not_while_working(server, monkeypatch):
    srv, films = server
    status, _, body = call(srv, "POST", "/api/films/bolt/scenes", {"after": "01-heap", "still": "BOLT sniffing", "motion": "static camera"})
    assert status == 200
    state = json.loads(body)
    sid = state["added"]
    assert [s["id"] for s in state["film"]["shots"]][:2] == ["01-heap", sid]
    assert call(srv, "POST", "/api/films/bolt/scenes", {"still": "", "motion": "x"})[0] == 400
    monkeypatch.setenv("FAKE_SLEEP", "3")
    st.approve(films / "bolt", "story")
    assert call(srv, "POST", "/api/films/bolt/run", {"step": "board"})[0] == 202
    assert call(srv, "DELETE", f"/api/films/bolt/scenes/{sid}")[0] == 409
    call(srv, "POST", "/api/films/bolt/stop")
    time.sleep(0.5)
    status, _, body = call(srv, "DELETE", f"/api/films/bolt/scenes/{sid}")
    assert status == 200 and sid not in [s["id"] for s in json.loads(body)["shots"]]


# --- voice pace: a scene's lines come before its motion -----------------------------------------

VOICE_PACED = STORY.replace("seed = 1024\n", 'seed = 1024\npace = "voice"\n', 1)


def tone(path: Path, secs: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"sine=frequency=300:duration={secs}",
                    "-ac", "1", "-ar", "24000", str(path)], check=True)


def speak(folder: Path, sid: str, secs: float = 2.0) -> None:
    """Record a scene's lines as real audio, so their length can set the shot's."""
    make(folder, sid, "voices")
    for wav in [folder / "voice" / f"{sid}.wav", *(folder / "voice" / "dialogue").glob(f"{sid}-*.wav")]:
        if wav.exists():
            tone(wav, secs)


def test_a_voice_paced_scene_records_its_lines_before_its_motion(tmp_path):
    fixed = write_story(tmp_path / "fixed").parent
    assert st.film_state(fixed)["parts"] == list(st.SCENE_PARTS)
    folder = write_story(tmp_path / "bolt", VOICE_PACED).parent
    assert st.film_state(folder)["parts"] == ["picture", "voices", "video", "sound", "preview"]
    open_scenes(folder)
    make(folder, "01-heap", "picture")
    assert st.approve(folder, "scene:01-heap:picture") == []
    s = stages(folder)
    assert s["scene:01-heap:voices"]["unlocked"] and not s["scene:01-heap:video"]["unlocked"]
    assert "lines" in (st.gate(folder, "video", {"shot": "01-heap"}) or "")
    speak(folder, "01-heap")
    assert st.approve(folder, "scene:01-heap:voices") == []
    assert stages(folder)["scene:01-heap:video"]["unlocked"]
    assert st.gate(folder, "video", {"shot": "01-heap"}) is None


def test_re_recording_a_line_to_a_new_length_reopens_the_motion(tmp_path):
    folder = write_story(tmp_path / "bolt", VOICE_PACED).parent
    open_scenes(folder)
    make(folder, "01-heap", "picture")
    assert st.approve(folder, "scene:01-heap:picture") == []
    speak(folder, "01-heap", 2.0)
    assert st.approve(folder, "scene:01-heap:voices") == []
    make(folder, "01-heap", "video")
    assert st.approve(folder, "scene:01-heap:video") == []
    tone(folder / "voice" / "01-heap.wav", 3.5)                     # a longer take of the same line
    s = stages(folder)
    assert not s["scene:01-heap:voices"]["approved"] and not s["scene:01-heap:video"]["done"]


def test_each_scene_shows_how_long_it_lasts(tmp_path):
    folder = write_story(tmp_path / "bolt", VOICE_PACED).parent
    assert [s["seconds"] for s in st.film_state(folder)["shots"]] == [None, None, pytest.approx(sa.wan_frames(3.0) / 24)]
    speak(folder, "01-heap", 2.0)                                       # known once its line is recorded
    assert st.film_state(folder)["shots"][0]["seconds"] == pytest.approx(sa.wan_frames(0.5 + 2.0 + sa.TAIL) / 24)
    fixed = write_story(tmp_path / "fixed").parent
    assert [s["seconds"] for s in st.film_state(fixed)["shots"]] == pytest.approx([121 / 24] * 3)


def test_the_finish_settings_save_with_the_story(tmp_path):
    folder = write_story(tmp_path / "bolt").parent
    doc = tomllib.loads((folder / "story.toml").read_text())
    doc.update(captions=False, end_card={"text": "Put your agent to work.", "sub": "BotGuild.ai"},
               grade={"warmth": 0.15, "saturation": 1.1})
    doc["shot"][0]["caption"] = "Quiet."
    assert st.save_story(folder, doc=doc) == []
    story = sa.load_story(folder / "story.toml")
    assert story.captions is False and story.end_card == sa.EndCard("Put your agent to work.", "BotGuild.ai")
    assert story.grade == sa.Grade(warmth=0.15, saturation=1.1) and story.shots[0].caption == "Quiet."


def test_a_new_film_can_ask_for_a_flat_cartoon_look(tmp_path):
    argv = st.step_argv(tmp_path / "f", "new", {"idea": "a bakery ad", "look": "flat"})
    assert argv[argv.index("--look") + 1] == "flat"
    assert "--look" not in st.step_argv(tmp_path / "f", "new", {"idea": "a bakery ad"})
    with pytest.raises(ValueError):
        st.step_argv(tmp_path / "f", "new", {"idea": "a bakery ad", "look": "oil"})


# --- ranked versions of a scene's picture -------------------------------------------------------

def test_more_versions_can_be_ranked(tmp_path):
    folder = write_story(tmp_path / "bolt").parent
    argv = st.step_argv(folder, "retake", {"shot": "01-heap", "seeds": [1, 2, 3, 4, 5], "rank": True})
    assert argv[-1] == "--rank"
    assert "--rank" not in st.step_argv(folder, "retake", {"shot": "01-heap", "seeds": [1, 2, 3]})


def test_each_scene_shows_its_versions_best_first(tmp_path):
    folder = write_story(tmp_path / "bolt").parent
    render_board(folder)
    story = sa.load_story(folder / "story.toml")
    shot = story.shots[0]
    prompt = sa.render_prompt(story, shot)[0]
    for seed in (7, 9):
        fake_still(folder, f"board/retakes/01-heap-s{seed}.png", prompt)
    (folder / "board" / "01-heap.rank.json").write_text(json.dumps(
        [{"seed": 9, "score": 9, "faults": []}, {"seed": shot.seed, "score": 6, "faults": ["blurred hands"]},
         {"seed": 7, "score": 3, "faults": ["no sunset"]}, {"seed": 123, "score": 8, "faults": []}]))   # 123: gone since
    s = st.film_state(folder)["shots"][0]
    assert s["ranking"] == [{"seed": 9, "score": 9, "faults": []}, {"seed": shot.seed, "score": 6, "faults": ["blurred hands"]},
                            {"seed": 7, "score": 3, "faults": ["no sunset"]}]
    assert st.film_state(folder)["shots"][1]["ranking"] == []


def test_the_page_shows_the_motion_checks_mistakes_for_the_current_clip(tmp_path):
    folder = write_story(tmp_path / "bolt", STORY).parent
    story = sa.load_story(folder / "story.toml")
    shot = story.shots[1]
    clip = sa.shot_path(story, shot)
    clip.parent.mkdir(parents=True, exist_ok=True)
    clip.write_bytes(b"a clip")
    assert st.film_state(folder)["shots"][1]["motion_faults"] is None       # not checked yet
    out = sa.motion_path(story, shot)
    out.write_text(json.dumps({"faults": ["his arm passes through the table"]}))
    sa.Keys(folder).record(out, sa.motion_key(story, shot))
    assert st.film_state(folder)["shots"][1]["motion_faults"] == ["his arm passes through the table"]
    clip.write_bytes(b"a new take")                                          # the old check no longer applies
    assert st.film_state(folder)["shots"][1]["motion_faults"] is None


def test_a_scenes_motion_can_be_checked_again_from_the_page(tmp_path):
    folder = write_story(tmp_path / "bolt", STORY).parent
    assert st.step_argv(folder, "motioncheck", {"shot": "02-claim"}) == \
        ["check", str(folder / "story.toml"), "--shot", "02-claim", "--motion"]
    assert st.STEP_PART["motioncheck"] == "video"
    with pytest.raises(ValueError):
        st.step_argv(folder, "motioncheck", {"shot": "nope"})


def test_the_page_and_the_server_agree_on_which_part_of_a_scene_each_step_needs():
    import re
    page = (Path(st.__file__).parent / "studio" / "index.html").read_text()
    js = re.search(r"const STEP_PART = \{(.*?)\};", page).group(1)
    assert dict(re.findall(r"(\w+): \"(\w+)\"", js)) == st.STEP_PART


def test_a_job_is_named_by_its_studio_step(tmp_path, fake_bin):
    folder = write_story(tmp_path / "bolt").parent
    jobs = st.Jobs()
    jobs.start(folder, st.step_argv(folder, "motioncheck", {"shot": "02-claim"}), step="motioncheck")
    assert wait_done(jobs, folder)["step"] == "motioncheck"
    page = (Path(st.__file__).parent / "studio" / "index.html").read_text()
    assert "motioncheck:" in page.split("const STEP_NAMES")[1].split("};")[0]


# --- the page itself, run in Node against a stand-in document ----------------------------------

NODE = __import__("shutil").which("node")


def on_page(js: str):
    """Run `js` with the studio page loaded (its functions in scope) and return what it prints as JSON."""
    harness = Path(__file__).with_name("studio_page.cjs")
    prog = (f"const {{ load, fieldOf }} = require({json.dumps(str(harness))}); const page = load();"
            f"page.fieldOf = fieldOf; const vm = require('vm');"
            f"Promise.resolve(vm.runInContext({json.dumps('(async () => {' + js + '})()')}, page))"
            f".then((v) => console.log(JSON.stringify(v)));")
    r = subprocess.run([NODE, "-e", prog], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


@pytest.mark.skipif(not NODE, reason="needs node")
def test_a_scene_card_shows_and_edits_its_own_lighting():
    got = on_page("""
      __S.film = { shots: [{ id: "05-dawn" }], references: [] };
      const d = { lighting: "late night, purple neon", angles: {}, shot: [{ id: "05-dawn", still: "x", motion: "y", lighting: "dawn, golden light" }] };
      const s = d.shot[0], f = fieldOf(shotCard(d, s, 0, () => [], () => {}), "Lighting");
      const shown = { value: f.value, placeholder: f.placeholder };
      f.value = "dusk, orange light"; f.dispatch("input"); const typed = s.lighting;
      f.value = ""; f.dispatch("input");
      return { ...shown, typed, cleared: "lighting" in s };
    """)
    assert got["value"] == "dawn, golden light"
    assert "late night, purple neon" in got["placeholder"]       # empty means the film's own light
    assert got["typed"] == "dusk, orange light" and got["cleared"] is False


FASTVIDEO_PY = Path.home() / "FastVideo" / ".venv" / "bin" / "python"


@pytest.mark.skipif(not FASTVIDEO_PY.exists(), reason="needs FastVideo's venv (OpenCV)")
def test_the_studio_finds_a_scenes_drawn_screen(server):
    srv, films = server
    draw = ("import sys, cv2, numpy as np; img = np.full((720, 1248, 3), 30, np.uint8);"
            "cv2.fillConvexPoly(img, np.int32([[300, 200], [900, 220], [880, 560], [320, 540]]), (252, 252, 252));"
            "cv2.imwrite(sys.argv[1], img)")
    (films / "bolt" / "board").mkdir(exist_ok=True)
    subprocess.run([str(FASTVIDEO_PY), "-c", draw, str(films / "bolt" / "board" / "02-claim.png")], check=True)
    status, _, body = call(srv, "POST", "/api/films/bolt/findscreen", {"shot": "02-claim"})
    assert status == 200, body
    got = json.loads(body)["corners"]
    for (x, y), (wx, wy) in zip(got, [[300, 200], [900, 220], [880, 560], [320, 540]]):
        assert abs(x - wx / 1248) < 0.005 and abs(y - wy / 720) < 0.005
    status, _, body = call(srv, "POST", "/api/films/bolt/findscreen", {"shot": "01-heap"})   # never drawn
    assert status == 400 and "picture" in json.loads(body)["errors"][0]


@pytest.mark.skipif(not NODE, reason="needs node")
def test_the_screen_editor_finds_the_screen_and_shows_it_up_close():
    got = on_page("""
      __S.name = "night"; __S.film = { doc: { shot: [{ id: "02-page" }] }, references: ["reference/gigs.png"], shots: [] };
      const asked = [];
      globalThis.fetch = async (url, o) => { asked.push([url, o.method]);
        return { ok: true, json: async () => ({ corners: [[0.6, 0.42], [0.87, 0.46], [0.82, 0.81], [0.55, 0.7]] }) }; };
      const el = screenEditor({ id: "02-page", still: "/media/x/board/02-page.png" });
      const btn = (t) => el.all().find((n) => n.tagName === "BUTTON" && n.textContent === t);
      const zoom = el.all().find((n) => n.className === "zoom");
      const before = { saveable: !btn("Save screen").getAttribute("disabled") && !btn("Save screen").disabled, zoomHidden: zoom.hidden };
      const pick = el.all().find((n) => n.tagName === "SELECT"); pick.value = "reference/gigs.png"; pick.dispatch("change");
      btn("Find the screen").dispatch("click");
      await new Promise((r) => setTimeout(r, 0));
      const polys = zoom.all().filter((n) => n.tagName === "POLYGON").map((n) => n.getAttribute("points"));
      return { before, asked, polys, zoomHidden: zoom.hidden, saveable: !btn("Save screen").disabled };
    """)
    assert got["before"]["zoomHidden"] is True and got["before"]["saveable"] is False
    assert got["asked"] == [["/api/films/night/findscreen", "POST"]]
    assert got["polys"] == ["0.6,0.42 0.87,0.46 0.82,0.81 0.55,0.7"]     # the found outline, up close
    assert got["zoomHidden"] is False and got["saveable"] is True


@pytest.mark.skipif(not NODE, reason="needs node")
def test_the_film_checks_name_a_line_missing_from_the_final_mix():
    got = on_page("""
      const chips = (checks) => {
        __S.film = { title: "T", film: "film/t.mp4", shots: [], references: [], score: {}, errors: [], steps: [], checks };
        return filmTab().all().filter((n) => n.className.startsWith("chip")).map((n) => [n.className, n.textContent]);
      };
      return { lost: chips({ narration: { "01-a": 1, "05-b": 1, "06-c": 1 }, final_narration: 0.81, final_lines: { "01-a": 1, "05-b": 0.3, "06-c": 0.2 } }),
               fine: chips({ narration: { "01-a": 1 }, final_narration: 0.95, final_lines: { "01-a": 0.97 } }) };
    """)
    lost = [c for c in got["lost"] if "final mix" in c[1]]
    assert lost == [["chip bad", "missing from the final mix: 05-b, 06-c"]]   # 0.81 overall, yet two lines gone
    assert [c for c in got["fine"] if "final mix" in c[1]] == [["chip ok", "speech in final mix 0.95"]]


@pytest.mark.skipif(not NODE, reason="needs node")
def test_draw_again_on_an_up_to_date_picture_asks_for_a_new_one():
    got = on_page("""
      __S.name = "pond"; __S.film = { errors: [], steps: [], references: [], doc: { shot: [{ id: "03-strike" }] }, shots: [] };
      globalThis.renderFilm = () => {}; globalThis.loadFilm = () => {};
      const sent = [];
      globalThis.fetch = async (url, o) => { sent.push(JSON.parse(o.body || "{}")); return { ok: true, json: async () => __S.film }; };
      const click = async (root) => { root.all().find((n) => n.tagName === "BUTTON" && /^Draw( again)?$/.test(n.textContent)).dispatch("click");
                                       await new Promise((r) => setTimeout(r, 0)); };
      const scene = (fresh) => h("div", {}, picturePart({ id: "03-strike", still: "board/03-strike.png", still_fresh: fresh, retakes: [], seed: 4024 },
                                                         { done: true, unlocked: true }));
      await click(scene(true)); await click(scene(false));
      const look = (fresh) => lookCard({ name: "PIER", kind: "place", url: "looks/PIER.png", fresh, text: "", seed: 1024, takes: [] });
      await click(look(true)); await click(look(false));
      return sent;
    """)
    assert got[0] == {"step": "board", "shot": "03-strike", "new_picture": True}   # nothing changed: a new picture
    assert got[1] == {"step": "board", "shot": "03-strike"}                       # description changed: draw it
    assert got[2] == {"step": "look", "name": "PIER", "new_look": True}
    assert got[3] == {"step": "look", "name": "PIER"}
