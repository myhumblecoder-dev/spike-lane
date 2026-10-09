"""spike-studio: the web UI over spike-animate. Editing a story from the page, what the page shows about
a film, running steps as background jobs, serving media so a phone can play it, and the access token."""
from __future__ import annotations

import http.client
import json
import os
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
            touch(folder, "voice/card.wav", f"voice/{sid}.wav")
            keys.record(folder / "voice" / "card.wav", sa.narrator_card_key(story))
            keys.record(folder / "voice" / f"{sid}.wav", sa.narration_key(story, shot))
        for n, line in enumerate(shot.dialogue, 1):
            touch(folder, f"voice/cards/{line.speaker}.wav", f"voice/dialogue/{sid}-{n}-{line.speaker}.wav")
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
    assert list(s) == ["story"] + [f"scene:{i}:{p}" for i in IDS for p in st.SCENE_PARTS] + ["music", "film"]
    assert s["story"]["unlocked"] and not s["story"]["approved"]
    assert not any(v["unlocked"] for k, v in s.items() if k != "story")
    assert st.gate(folder, "board", {"shot": "01-heap"}) is not None and st.gate(folder, "film") is not None


def test_a_scene_is_made_one_part_at_a_time(tmp_path):
    folder = write_story(tmp_path / "bolt").parent
    assert st.approve(folder, "story") == []
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
    st.approve(folder, "story")
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
    st.approve(folder, "story")
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
    st.approve(folder, "story")
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
    st.approve(folder, "story")
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
    st.approve(folder, "story")
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
    st.approve(folder, "story")
    make(folder, "01-heap", "picture")
    st.approve(folder, "scene:01-heap:picture")
    status, _, _ = call(srv, "POST", "/api/films/bolt/run", {"step": "video", "shot": "01-heap", "new_take": True})
    assert status == 202 and "motion_seed" in st._doc(folder)["shot"][0]


def test_a_shot_made_before_takes_existed_is_kept_when_animating_again(tmp_path):
    folder = write_story(tmp_path / "bolt").parent
    make(folder, "01-heap", "picture")
    make(folder, "01-heap", "video")                       # the shot alone, as older films have it
    (folder / "shots" / "01-heap.mp4").write_bytes(b"the old shot")
    story = sa.load_story(folder / "story.toml")
    sa.Keys(folder).record(folder / "shots" / "01-heap.mp4", sa.shot_key(story, story.shots[0]))
    assert st.new_take(folder, "01-heap") == []
    assert (folder / "shots" / "takes" / "01-heap-m1024.mp4").read_bytes() == b"the old shot"
