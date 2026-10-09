"""spike-animate new: a local LLM drafts story.toml from a one-line idea. The draft is shaped by a
JSON schema, written as TOML, and linted against the story rules; a draft that breaks them goes
back to the model with the problems listed. Ollama is replaced by a small local HTTP server."""
from __future__ import annotations

import json
import os
import socketserver
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import spike_animate as sa  # noqa: E402
import spike_lane  # noqa: E402

ANIMATE = ROOT / "bin" / "spike-animate"


def draft(**over):
    d = {
        "title": "The Lighthouse Gull",
        "look": "2D cartoon animation, stormy blues, warm lamp light",
        "still_style": "2D cartoon animation still, bold black outlines, flat vibrant colors, stormy sea coast",
        "characters": [{"name": "KEEPER", "description": "an old lighthouse keeper with a white beard and yellow raincoat"},
                       {"name": "GULL", "description": "a small grey seagull with a bent wing"}],
        "narrator": {"voice": "a gentle older woman, warm and slow", "sample": "Out where the land runs out, an old man kept the light burning."},
        "music": {"caption": "gentle sea shanty strings, accordion, building to a storm",
                  "structure": "[Intro - calm sea]\n\n[Climax - storm]\n\n[Outro - dawn]", "key": "A minor"},
        "shots": [
            {"id": "01-light", "still": "wide shot of KEEPER lighting the lamp at dusk", "motion": "static camera, an old man lights a lamp",
             "sfx": "waves, wind, creaking", "narration": "The keeper lit the lamp, as always.", "intensity": 0.2, "action": False},
            {"id": "02-storm", "still": "GULL tumbling through a storm toward the lighthouse", "motion": "a small gull is blown through rain",
             "sfx": "howling wind, thunder", "narration": "", "intensity": 1.0, "action": True},
            {"id": "03-friends", "still": "KEEPER feeding GULL by the warm stove", "motion": "static camera, the old man feeds a gull",
             "sfx": "crackling stove, rain on glass", "narration": "Neither was alone after that.", "intensity": 0.3, "action": False},
        ],
    }
    d.update(over)
    return d


# --- turning a draft into a story -------------------------------------------

def test_a_draft_becomes_a_loadable_story(tmp_path):
    p = tmp_path / "story.toml"
    p.write_text(sa.draft_to_toml(draft(), idea="a lonely lighthouse keeper befriends a seagull"))
    story = sa.load_story(p)
    assert story.title == "The Lighthouse Gull"
    assert story.characters["GULL"] == "a small grey seagull with a bent wing"
    assert [s.id for s in story.shots] == ["01-light", "02-storm", "03-friends"]
    assert story.shots[1].anchor == 0 and story.shots[0].anchor == 1        # action shots move freely
    assert story.shots[1].narration == "" and story.narrator.lead == 0.5
    assert story.music.key == "A minor" and story.music.structure.startswith("[Intro")
    assert "a lonely lighthouse keeper befriends a seagull" in p.read_text()  # the idea is kept as a comment


def test_quotes_and_newlines_survive_the_round_trip(tmp_path):
    d = draft()
    d["shots"][0]["narration"] = 'He said "hold on" \\ and waited.'
    p = tmp_path / "story.toml"
    p.write_text(sa.draft_to_toml(d, idea="x"))
    assert sa.load_story(p).shots[0].narration == 'He said "hold on" \\ and waited.'


# --- story rules ------------------------------------------------------------

def test_a_good_draft_has_no_problems():
    assert sa.lint_draft(draft(), shots=3) == []


def test_lint_names_each_broken_rule():
    d = draft()
    d["shots"][0]["narration"] = "The keeper lit the great brass lamp at the top of the tower, as he had done every night for forty long years."
    d["shots"][1]["intensity"] = 1.4
    d["shots"][2]["id"] = "three"
    d["shots"][2]["still"] = "a warm stove by the window"           # GULL now appears in only one still: fine
    d["characters"].append({"name": "CAT", "description": "a ginger cat"})  # never used
    problems = sa.lint_draft(d, shots=4)
    text = "\n".join(problems)
    assert "4 shots" in text                                  # wrong count
    assert "01-light" in text and "words" in text            # narration too long for a 5 s shot
    assert "02-storm" in text and "intensity" in text
    assert "'three'" in text and "id" in text
    assert "CAT" in text and "never" in text
    assert len(problems) == 5


def test_stills_must_name_characters_not_redescribe_them():
    d = draft()
    d["shots"][0]["still"] = "wide shot of an old lighthouse keeper with a white beard lighting the lamp"
    assert any("KEEPER" in p and "01-light" in p for p in sa.lint_draft(d, shots=3))


def test_writer_prompt_carries_the_idea_the_count_and_the_rules():
    p = sa.writer_prompt("a robot learns to paint", shots=6)
    assert "a robot learns to paint" in p and "6 shots" in p
    assert "UPPERCASE" in p and "5 seconds" in p and "intensity" in p


# --- the command ------------------------------------------------------------

class FakeOllama(BaseHTTPRequestHandler):
    replies: list = []
    requests: list = []
    holders: list = []

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOllama.requests.append(body)
        FakeOllama.holders.append((spike_lane.holder() or {}).get("name"))
        reply = FakeOllama.replies.pop(0)
        out = json.dumps({"response": json.dumps(reply), "done": True}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


class QuickServer(HTTPServer):
    def server_bind(self):  # skip HTTPServer's reverse-DNS lookup (seconds on macOS)
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = "127.0.0.1", self.socket.getsockname()[1]


@pytest.fixture
def ollama(tmp_path, monkeypatch):
    FakeOllama.replies, FakeOllama.requests, FakeOllama.holders = [], [], []
    srv = QuickServer(("127.0.0.1", 0), FakeOllama)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("SPIKE_LANE_DIR", str(tmp_path / "lane"))
    e = os.environ.copy()
    e.update(OLLAMA_HOST=f"http://127.0.0.1:{srv.server_port}", PATH="/usr/bin:/bin")
    yield e
    srv.shutdown()


def new(e, *argv, cwd):
    return subprocess.run([str(ANIMATE), "new", *argv], capture_output=True, text=True, env=e, cwd=cwd)


def test_new_drafts_a_story_with_the_local_model_inside_the_lane(ollama, tmp_path):
    FakeOllama.replies = [draft()]
    r = new(ollama, "gull", "a lonely lighthouse keeper befriends a seagull", "--shots", "3", cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    story = sa.load_story(tmp_path / "gull" / "story.toml")
    assert story.title == "The Lighthouse Gull" and len(story.shots) == 3
    req = FakeOllama.requests[0]
    assert req["model"] == "gemma4:26b" and req["stream"] is False
    assert req["think"] is False                       # gemma4 thinking runs away otherwise
    assert req["format"]["type"] == "object"           # structured output, shaped by the schema
    assert req["keep_alive"] == 0                      # free the 17 GB as soon as the draft is in
    assert "a lonely lighthouse keeper befriends a seagull" in req["prompt"]
    assert FakeOllama.holders == ["writer"]
    assert "board" in r.stdout                          # says what to do next


def test_new_sends_problems_back_and_keeps_the_fixed_draft(ollama, tmp_path):
    bad = draft()
    bad["shots"][1]["intensity"] = 3
    FakeOllama.replies = [bad, draft()]
    r = new(ollama, "gull", "a lighthouse keeper and a gull", "--shots", "3", cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert len(FakeOllama.requests) == 2
    assert "02-storm" in FakeOllama.requests[1]["prompt"] and "intensity" in FakeOllama.requests[1]["prompt"]
    assert '"title": "The Lighthouse Gull"' in FakeOllama.requests[1]["prompt"]   # it edits its own draft
    assert sa.load_story(tmp_path / "gull" / "story.toml").shots[1].intensity == 1.0


def test_new_gives_up_after_three_bad_drafts_and_saves_nothing(ollama, tmp_path):
    bad = draft()
    bad["shots"][1]["intensity"] = 3
    FakeOllama.replies = [bad, bad, bad]
    r = new(ollama, "gull", "x", "--shots", "3", cwd=tmp_path)
    assert r.returncode == 1
    assert "intensity" in r.stderr
    assert not (tmp_path / "gull" / "story.toml").exists()


def test_new_never_overwrites_an_existing_story(ollama, tmp_path):
    (tmp_path / "gull").mkdir()
    (tmp_path / "gull" / "story.toml").write_text("mine")
    r = new(ollama, "gull", "x", cwd=tmp_path)
    assert r.returncode == 2
    assert (tmp_path / "gull" / "story.toml").read_text() == "mine"
    assert FakeOllama.requests == []


# --- reference notes: what the author already knows about the film ---------------------------

def test_reference_notes_go_to_the_writer():
    p = sa.writer_prompt("a robot learns to paint", shots=6, notes="The robot is named PIP. Set in a rainy Paris attic.")
    assert "REFERENCE NOTES" in p and "named PIP" in p
    assert "REFERENCE NOTES" not in sa.writer_prompt("a robot learns to paint", shots=6)


def test_the_story_keeps_its_idea_and_notes(tmp_path):
    p = tmp_path / "story.toml"
    p.write_text(sa.draft_to_toml(draft(), idea='a "lonely" keeper', notes="stormy\nnorth coast"))
    import tomllib
    d = tomllib.loads(p.read_text())
    assert d["idea"] == 'a "lonely" keeper' and d["notes"] == "stormy\nnorth coast"
    sa.load_story(p)


def test_new_passes_notes_to_the_writer(ollama, tmp_path):
    FakeOllama.replies = [draft()]
    r = new(ollama, "gull", "a keeper and a gull", "--shots", "3", "--notes", "the gull is called PIP", cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert "the gull is called PIP" in FakeOllama.requests[0]["prompt"]


# --- the author's own story, and a picture for the style ----------------------------------

def test_the_writer_can_follow_a_whole_story_and_its_dialogue():
    story = 'Pip the robot finds a paintbrush. "What is this?" he asks. He paints the sunrise.'
    p = sa.writer_prompt(story, shots=6)
    assert story in p
    assert "whole story" in p and "word for word" in p


def test_a_style_from_a_picture_replaces_the_default_cartoon_look():
    style = "soft watercolor storybook illustration, muted pastels, loose ink lines, paper texture"
    p = sa.writer_prompt("a robot learns to paint", shots=6, style=style)
    assert style in p and "VISUAL STYLE" in p
    assert "bold black outlines, flat vibrant colors," not in p.split("Example of the format")[0]
    assert "bold black outlines, flat vibrant colors," in sa.writer_prompt("x", shots=6).split("Example of the format")[0]


def test_the_story_remembers_its_style_picture(tmp_path):
    p = tmp_path / "story.toml"
    p.write_text(sa.draft_to_toml(draft(), idea="x", style_image="reference/style.png", style="watercolor"))
    import tomllib
    d = tomllib.loads(p.read_text())
    assert d["style_image"] == "reference/style.png" and d["style"] == "watercolor"
    sa.load_story(p)


def test_new_reads_the_style_from_a_picture_with_the_local_model(ollama, tmp_path):
    pic = tmp_path / "mine.png"
    pic.write_bytes(b"\x89PNG fake picture")
    FakeOllama.replies = [{"style": "soft watercolor, muted pastels"}, draft()]
    r = new(ollama, "gull", "a keeper and a gull", "--shots", "3", "--style-image", str(pic), cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    look, write = FakeOllama.requests
    import base64
    assert look["images"] == [base64.b64encode(pic.read_bytes()).decode()]
    assert look["model"] == "gemma4:26b" and look["think"] is False
    assert FakeOllama.holders == ["writer", "writer"]
    assert "soft watercolor, muted pastels" in write["prompt"]
    import tomllib
    d = tomllib.loads((tmp_path / "gull" / "story.toml").read_text())
    assert d["style"] == "soft watercolor, muted pastels"
    assert (tmp_path / "gull" / d["style_image"]).read_bytes() == pic.read_bytes()
