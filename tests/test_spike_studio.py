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
    touch(folder, "board/01-heap.png", "board/retakes/01-heap-s7.png", "board/contact-sheet.jpg",
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
