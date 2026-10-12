"""spike-animate new: a local LLM drafts story.toml from a one-line idea. The draft is shaped by a
JSON schema, written as TOML, and linted against the story rules; a draft that breaks them goes
back to the model with the problems listed. Ollama is replaced by a small local HTTP server."""
from __future__ import annotations

import json
import os
import shutil
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
    assert story.shots[1].narration == "" and story.narrator.lead == sa.DRAFT_LEAD
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
        out = reply if isinstance(reply, bytes) else json.dumps({"response": json.dumps(reply), "done": True}).encode()
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
    assert "bold black outlines, flat vibrant colors," in sa.writer_prompt("x", shots=6, look="flat").split("Example of the format")[0]
    assert sa.KEYART_STILL in sa.writer_prompt("x", shots=6).split("Example of the format")[0]   # the default now
    assert sa.KEYART_STILL not in p.split("Example of the format")[0]                              # a picture still wins


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


# --- optional parts: narrator, dialogue, music --------------------------------------------

def test_a_film_can_have_no_narrator_no_dialogue_and_no_music(tmp_path):
    parts = sa.Parts(narrator=False, dialogue=False, music=False)
    p = sa.writer_prompt("a gull and a keeper", shots=3, parts=parts)
    rules = p.split("Example of the format")[0]
    assert "There is no narrator" in rules and "No one speaks" in rules and "no score" in rules
    schema = sa.draft_schema(parts)
    assert not {"narrator", "music"} & set(schema["required"]) and "narrator" not in schema["properties"]
    d = draft()
    del d["narrator"], d["music"]
    for s in d["shots"]:
        s["narration"] = ""
    path = tmp_path / "story.toml"
    path.write_text(sa.draft_to_toml(d, idea="x"))
    story = sa.load_story(path)
    assert story.narrator is None and story.music is None
    assert "[narrator]" not in path.read_text() and "[music]" not in path.read_text()


def test_lint_holds_the_writer_to_the_parts_the_author_chose():
    d = draft()
    d["shots"][0]["dialogue"] = [{"speaker": "KEEPER", "line": "Hello there.", "emotion": ""}]
    d["characters"][0]["voice"] = "gruff"
    problems = sa.lint_draft(d, 3, sa.Parts(narrator=False, dialogue=False))
    assert any("01-light" in p and "no narrator" in p for p in problems)
    assert any("01-light" in p and "no one speaks" in p for p in problems)
    assert not sa.lint_draft(draft(), 3)                       # everything on, as before


def test_new_can_leave_out_the_narrator_and_music(ollama, tmp_path):
    d = draft()
    del d["narrator"], d["music"]
    for s in d["shots"]:
        s["narration"] = ""
    FakeOllama.replies = [d]
    r = new(ollama, "gull", "a keeper and a gull", "--shots", "3", "--no-narrator", "--no-music", cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    req = FakeOllama.requests[0]
    assert "narrator" not in req["format"]["required"] and "There is no narrator" in req["prompt"]
    story = sa.load_story(tmp_path / "gull" / "story.toml")
    assert story.narrator is None and story.music is None


# --- motion that stays inside the picture ---------------------------------------------------

def test_motion_may_only_move_what_the_still_shows_and_silent_faces_stay_closed():
    rules = sa.writer_prompt("a student studies at a cafe", shots=6).split("Example of the format")[0]
    assert "only what the still shows" in rules            # "as she reads" grew a book out of a laptop edge
    assert "mouth closed" in rules                          # the video model makes idle faces talk


# --- continuity: does a picture match the film bible? ---------------------------------------

BIBLE_STORY = '''title = "Morning"
look = "cel animation"
still_style = "cel animation still"
lighting = "sunrise, warm low pink-gold light from the left"

[characters.ELENA]
look = "a young woman in a cream linen shirt with rolled sleeves"
voice = "soft"

[places]
TERRACE = "a whitewashed cafe terrace on a white sand beach"

[[shot]]
id = "01-sit"
still = "ELENA sitting at a table on the TERRACE"
motion = "she sips her coffee, mouth closed"
'''


def test_check_compares_a_picture_with_the_film_bible(ollama, tmp_path):
    film = tmp_path / "morning"
    (film / "board").mkdir(parents=True)
    (film / "story.toml").write_text(BIBLE_STORY)
    (film / "board" / "01-sit.png").write_bytes(b"\x89PNG a picture")
    FakeOllama.replies = [{"misses": ["ELENA's shirt is blue, not cream linen"]}]
    r = subprocess.run([str(ANIMATE), "check", "story.toml", "--shot", "01-sit"], capture_output=True, text=True,
                       env=ollama, cwd=film)
    assert r.returncode == 0, r.stderr
    req = FakeOllama.requests[0]
    assert req["images"] and req["think"] is False and FakeOllama.holders == ["writer"]
    for fact in ("cream linen shirt", "pink-gold light", "whitewashed cafe terrace"):
        assert fact in req["prompt"]
    saved = json.loads((film / "board" / "01-sit.check.json").read_text())
    assert saved["misses"] == ["ELENA's shirt is blue, not cream linen"]
    assert "blue" in r.stdout
    # the same picture is not checked twice
    r = subprocess.run([str(ANIMATE), "check", "story.toml", "--shot", "01-sit"], capture_output=True, text=True,
                       env=ollama, cwd=film)
    assert r.returncode == 0 and len(FakeOllama.requests) == 1


def test_the_check_asks_only_about_what_the_shot_shows(tmp_path):
    p = tmp_path / "story.toml"
    p.write_text(BIBLE_STORY.replace('lighting = "sunrise, warm low pink-gold light from the left"\n', "")
                 + '\n[[shot]]\nid = "02-sea"\nstill = "the empty TERRACE"\nmotion = "waves"\n')
    story = sa.load_story(p)
    sit, sea = story.shots
    assert "cream linen shirt" in sa.continuity_prompt(story, sit) and "Lighting" not in sa.continuity_prompt(story, sit)
    assert "cream linen shirt" not in sa.continuity_prompt(story, sea)        # ELENA isn't in this shot
    assert "whitewashed cafe terrace" in sa.continuity_prompt(story, sea)


def bible_draft():
    d = draft()
    d["lighting"] = "stormy dusk, cold blue light, the lamp's warm glow from the right"
    d["places"] = [{"name": "TOWER", "description": "a white lighthouse on black rocks above a grey sea"}]
    d["angles"] = [{"name": "WIDE", "frame": "wide shot from the cliff path, the tower on the left"},
                   {"name": "LAMP", "frame": "medium shot inside the lamp room, the window behind"}]
    d["shots"][0]["angle"], d["shots"][1]["angle"], d["shots"][2]["angle"] = "LAMP", "WIDE", "LAMP"
    d["shots"][1]["still"] = "GULL tumbling through a storm toward the TOWER"
    return d


def test_the_writer_drafts_a_film_bible(tmp_path):
    rules = sa.writer_prompt("a keeper and a gull", shots=3).split("Example of the format")[0]
    assert '"lighting"' in rules and '"places"' in rules and '"angles"' in rules
    assert "clothes" in rules                                          # wardrobe goes in the character's look
    schema = sa.draft_schema()
    assert {"lighting", "places", "angles"} <= set(schema["required"])
    p = tmp_path / "story.toml"
    p.write_text(sa.draft_to_toml(bible_draft(), idea="x"))
    story = sa.load_story(p)
    assert story.lighting.startswith("stormy dusk") and story.places["TOWER"].startswith("a white lighthouse")
    assert [s.angle for s in story.shots] == ["LAMP", "WIDE", "LAMP"]
    lamp = [s.seed for s in story.shots if s.angle == "LAMP"]
    assert len(set(lamp)) == 1                                         # one angle, one seed
    assert story.angles["WIDE"].seed != story.angles["LAMP"].seed


def test_lint_catches_an_unknown_angle_or_place_name():
    d = bible_draft()
    d["shots"][0]["angle"] = "CLOSEUP"
    d["places"].append({"name": "Cliff-Top", "description": "a cliff"})
    problems = sa.lint_draft(d, 3)
    assert any("01-light" in p and "CLOSEUP" in p for p in problems)
    assert any("Cliff-Top" in p for p in problems)
    assert not sa.lint_draft(bible_draft(), 3)


# --- voice pace: short films cut to their lines -----------------------------------------------

def test_a_drafted_story_is_paced_by_its_voice_over(tmp_path):
    p = tmp_path / "story.toml"
    p.write_text(sa.draft_to_toml(draft(), idea="a lonely lighthouse keeper befriends a seagull"))
    assert sa.load_story(p).pace == "voice"


def test_the_writer_cuts_to_the_lines_and_keeps_them_short():
    p = sa.writer_prompt("a robot learns to paint", shots=6)
    assert "as long as its line" in p and "15 to 20 seconds" in p
    assert f"at most {sa.MAX_NARRATION_WORDS} words" in p and sa.MAX_NARRATION_WORDS <= 10
    assert "half the shots" not in p                       # the voice-over carries the film now


def test_a_new_film_has_six_shots_unless_asked(tmp_path):
    args = sa.build_parser().parse_args(["new", str(tmp_path), "an idea"])
    assert args.shots == 6


# --- captions and the end card ----------------------------------------------------------------

def test_a_draft_keeps_its_captions_and_end_card(tmp_path):
    d = draft(end_card={"text": "Keep the light burning.", "sub": "lighthouse.org"})
    d["shots"][0]["caption"] = "Lamp lit."
    d["shots"][1]["caption"] = ""
    d["shots"][2]["caption"] = ""                                  # no caption over a narrated shot
    p = tmp_path / "story.toml"
    p.write_text(sa.draft_to_toml(d, idea="a keeper and a gull"))
    story = sa.load_story(p)
    assert [sa.caption_text(story, s) for s in story.shots] == ["Lamp lit.", "", ""]
    assert story.end_card == sa.EndCard("Keep the light burning.", "lighthouse.org")


def test_a_draft_without_an_end_card_has_none(tmp_path):
    p = tmp_path / "story.toml"
    p.write_text(sa.draft_to_toml(draft(end_card={"text": "", "sub": ""}), idea="a keeper and a gull"))
    assert sa.load_story(p).end_card is None


def test_the_writer_asks_for_short_captions_and_an_end_card():
    p = sa.writer_prompt("an ad for a bakery", shots=6)
    assert '"caption"' in p and f"at most {sa.MAX_CAPTION_WORDS} words" in p and '"end_card"' in p
    schema = sa.draft_schema()
    assert "end_card" in schema["required"] and "caption" in schema["properties"]["shots"]["items"]["required"]


def test_lint_keeps_captions_short():
    d = draft()
    d["shots"][0]["caption"] = "one two three four five six seven eight"
    assert any("caption" in x for x in sa.lint_draft(d, 3))


def test_a_drafted_story_renders_in_hd(tmp_path):
    p = tmp_path / "story.toml"
    p.write_text(sa.draft_to_toml(draft(), idea="a keeper and a gull"))
    assert sa.load_story(p).quality == "hd"


# --- the default look: key art, or a flat cartoon ----------------------------------------------

def test_without_a_style_picture_the_writer_asks_for_key_art():
    p = sa.writer_prompt("an ad for a bakery", shots=6)
    assert "anime key art" in p and "flat vibrant colors" not in p
    flat = sa.writer_prompt("an ad for a bakery", shots=6, look="flat")
    assert "flat vibrant colors" in flat and "anime key art" not in flat


def test_the_writer_dresses_each_place():
    assert "set dressing" in sa.writer_prompt("an ad for a bakery", shots=6)


def test_new_can_ask_for_a_flat_cartoon():
    assert sa.build_parser().parse_args(["new", "f", "idea"]).look == "keyart"
    assert sa.build_parser().parse_args(["new", "f", "idea", "--look", "flat"]).look == "flat"


# --- ranking a scene's picture versions ----------------------------------------------------------

def picture(path: Path, prompt: str, tag: str) -> None:
    """A picture with its prompt stored the way mflux stores it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG " + tag.encode() + json.dumps({"mflux_version": "x", "prompt": prompt}).encode())


def test_rank_orders_a_scenes_versions_best_first(ollama, tmp_path):
    film = tmp_path / "morning"
    film.mkdir()
    (film / "story.toml").write_text(BIBLE_STORY)
    story = sa.load_story(film / "story.toml")
    shot = story.shots[0]
    prompt = sa.render_prompt(story, shot)[0]
    picture(film / "board" / "01-sit.png", prompt, "current")
    picture(film / "board" / "retakes" / "01-sit-s7.png", prompt, "seven")
    picture(film / "board" / "retakes" / "01-sit-s9.png", prompt, "nine")
    picture(film / "board" / "retakes" / "01-sit-s5.png", "an older description", "old")    # not this picture any more
    FakeOllama.replies = [{"score": 6, "faults": ["hands are blurred"]}, {"score": 4, "faults": ["no terrace"]},
                          {"score": 9, "faults": []}]
    r = subprocess.run([str(ANIMATE), "rank", "story.toml", "01-sit"], capture_output=True, text=True, env=ollama, cwd=film)
    assert r.returncode == 0, r.stderr
    assert len(FakeOllama.requests) == 3 and all(len(q["images"]) == 1 for q in FakeOllama.requests)
    q = FakeOllama.requests[0]["prompt"]
    assert "ELENA sitting at a table" in q and "cream linen shirt" in q and "score" in q
    ranked = json.loads((film / "board" / "01-sit.rank.json").read_text())
    assert [x["seed"] for x in ranked] == [9, shot.seed, 7]
    assert ranked[0] == {"seed": 9, "score": 9, "faults": []}
    assert "seed 9" in r.stdout
    r = subprocess.run([str(ANIMATE), "rank", "story.toml", "01-sit"], capture_output=True, text=True, env=ollama, cwd=film)
    assert r.returncode == 0 and len(FakeOllama.requests) == 3                      # each picture is judged once


def test_the_ranking_counts_the_people_and_scores_by_a_rubric(tmp_path):
    (tmp_path / "story.toml").write_text(BIBLE_STORY)
    story = sa.load_story(tmp_path / "story.toml")
    q = sa.rank_prompt(story, story.shots[0])
    assert "exactly 1 person (ELENA)" in q and "drawn twice" in q
    assert "1-3" in q and "9-10" in q                              # a broken picture can't score well
    empty = BIBLE_STORY.replace('still = "ELENA sitting at a table on the TERRACE"', 'still = "the empty TERRACE at dawn"')
    (tmp_path / "story.toml").write_text(empty)
    story = sa.load_story(tmp_path / "story.toml")
    assert "no people" in sa.rank_prompt(story, story.shots[0])


# --- motion levels --------------------------------------------------------------------------------

def test_a_draft_keeps_how_each_shot_moves(tmp_path):
    d = draft()
    d["shots"][0].update(tier="still", camera="left")
    d["shots"][1].update(tier="full", camera="push")
    d["shots"][2].update(tier="subtle", camera="push")
    p = tmp_path / "story.toml"
    p.write_text(sa.draft_to_toml(d, idea="a keeper and a gull"))
    story = sa.load_story(p)
    assert [(s.tier, s.camera) for s in story.shots] == [("still", "left"), ("full", "push"), ("subtle", "push")]


def test_the_writer_chooses_how_each_shot_moves():
    p = sa.writer_prompt("an ad for a bakery", shots=6)
    assert '"tier"' in p and '"camera"' in p and f"at most {sa.MAX_FULL_SHOTS}" in p
    shot = sa.draft_schema()["properties"]["shots"]["items"]
    assert {"tier", "camera"} <= set(shot["required"])
    assert shot["properties"]["tier"]["enum"] == list(sa.TIERS)
    assert shot["properties"]["camera"]["enum"] == list(sa.CAMERAS)
    assert all(s["tier"] in sa.TIERS and s["camera"] in sa.CAMERAS for s in sa._example(sa.Parts())["shots"])


def test_lint_keeps_full_motion_for_a_few_shots():
    d = draft()
    d["shots"].append(dict(d["shots"][2], id="04-dawn"))
    for s in d["shots"]:
        s.update(tier="full", camera="push")
    assert len(d["shots"]) == sa.MAX_FULL_SHOTS + 1
    assert any("full" in x for x in sa.lint_draft(d, 4))
    d["shots"][0]["tier"] = "still"
    assert not any("full" in x for x in sa.lint_draft(d, 4))


def test_a_drafted_narrator_comes_in_quickly_after_each_cut(tmp_path):
    p = tmp_path / "story.toml"
    p.write_text(sa.draft_to_toml(draft(), idea="a keeper and a gull"))
    assert sa.load_story(p).narrator.lead == sa.DRAFT_LEAD and sa.DRAFT_LEAD <= 0.25


# --- clipping: things passing through each other ----------------------------------------------------

def clip(path: Path, seconds: float = 2.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    f"testsrc=size=320x176:rate=24", "-t", str(seconds), "-pix_fmt", "yuv420p", str(path)], check=True)


def fresh_clip(film: Path, seconds: float = 2.0) -> None:
    """01-sit's picture and a clip animated from it, recorded as up to date."""
    (film / "board").mkdir(parents=True, exist_ok=True)
    (film / "board" / "01-sit.png").write_bytes(b"\x89PNG a picture")
    clip(film / "shots" / "01-sit.mp4", seconds)
    story = sa.load_story(film / "story.toml")
    sa.Keys(film).record(sa.shot_path(story, story.shots[0]), sa.shot_key(story, story.shots[0]))


def with_ffmpeg(e: dict) -> dict:
    return {**e, "PATH": f"{Path(shutil.which('ffmpeg') or '/opt/homebrew/bin/ffmpeg').parent}:{e['PATH']}"}


def test_the_motion_check_looks_across_a_clip_for_things_passing_through_each_other(ollama, tmp_path):
    film = tmp_path / "morning"
    film.mkdir()
    (film / "story.toml").write_text(BIBLE_STORY)
    fresh_clip(film)
    FakeOllama.replies = [{"faults": ["her right arm disappears as she lowers it"]}]
    e = with_ffmpeg(ollama)
    r = subprocess.run([str(ANIMATE), "check", "story.toml", "--shot", "01-sit", "--motion"], capture_output=True,
                       text=True, env=e, cwd=film)
    assert r.returncode == 0, r.stderr
    [req] = FakeOllama.requests
    assert len(req["images"]) == sa.MOTION_FRAMES and FakeOllama.holders == ["writer"]   # frames across the clip, in order
    assert "she sips her coffee" in req["prompt"] and "through" in req["prompt"] and "disappear" in req["prompt"]
    assert json.loads((film / "shots" / "01-sit.motion.json").read_text()) == {"faults": ["her right arm disappears as she lowers it"]}
    assert "her right arm disappears" in r.stdout
    r = subprocess.run([str(ANIMATE), "check", "story.toml", "--shot", "01-sit", "--motion"], capture_output=True,
                       text=True, env=e, cwd=film)
    assert r.returncode == 0 and len(FakeOllama.requests) == 1                       # a clip is judged once
    fresh_clip(film, 3.0)                                                             # a new take is judged again
    FakeOllama.replies = [{"faults": []}]
    r = subprocess.run([str(ANIMATE), "check", "story.toml", "--shot", "01-sit", "--motion"], capture_output=True,
                       text=True, env=e, cwd=film)
    assert len(FakeOllama.requests) == 2 and "✓" in r.stdout


def motion_check(e: dict, film: Path) -> subprocess.CompletedProcess:
    return subprocess.run([str(ANIMATE), "check", "story.toml", "--shot", "01-sit", "--motion"], capture_output=True,
                          text=True, env=with_ffmpeg(e), cwd=film)


def checked_film(tmp_path: Path) -> Path:
    film = tmp_path / "morning"
    film.mkdir()
    (film / "story.toml").write_text(BIBLE_STORY)
    fresh_clip(film)
    return film


def test_the_motion_check_waits_for_a_clip_animated_from_the_current_picture(ollama, tmp_path):
    film = checked_film(tmp_path)
    (film / "board" / "01-sit.png").write_bytes(b"\x89PNG a new picture")          # redrawn, not animated yet
    r = motion_check(ollama, film)
    assert r.returncode == 0, r.stderr
    assert not FakeOllama.requests and "animate" in r.stdout and not (film / "shots" / "01-sit.motion.json").exists()


def test_another_vision_model_checks_the_motion_again(ollama, tmp_path):
    film = checked_film(tmp_path)
    FakeOllama.replies = [{"faults": []}, {"faults": ["her cup turns into a glass"]}]
    assert motion_check(ollama, film).returncode == 0
    r = motion_check({**ollama, "SPIKE_WRITER_MODEL": "qwen3-vl:32b"}, film)
    assert [q["model"] for q in FakeOllama.requests] == ["gemma4:26b", "qwen3-vl:32b"] and "glass" in r.stdout


def test_a_damaged_motion_verdict_is_checked_again(ollama, tmp_path):
    film = checked_film(tmp_path)
    FakeOllama.replies = [{"faults": []}, {"faults": []}]
    assert motion_check(ollama, film).returncode == 0
    (film / "shots" / "01-sit.motion.json").write_text('{"fau')
    r = motion_check(ollama, film)
    assert r.returncode == 0, r.stderr
    assert len(FakeOllama.requests) == 2 and "✓" in r.stdout


def test_the_motion_check_keeps_no_frames_for_a_later_take_to_reuse(ollama, tmp_path):
    film = checked_film(tmp_path)
    FakeOllama.replies = [{"faults": []}]
    assert motion_check(ollama, film).returncode == 0
    assert len(FakeOllama.requests[0]["images"]) == sa.MOTION_FRAMES
    assert not (film / "shots" / "01-sit.motion").exists()


def test_a_reply_from_ollama_that_is_not_json_is_a_stage_error(ollama, monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", ollama["OLLAMA_HOST"])
    FakeOllama.replies = [b"<html>502 bad gateway</html>"]
    with pytest.raises(sa.StageError):
        sa.ollama_generate("hello", "gemma4:26b")


def test_a_glide_has_no_motion_to_check(ollama, tmp_path):
    film = tmp_path / "morning"
    film.mkdir()
    (film / "story.toml").write_text(BIBLE_STORY.replace('motion = "she sips', 'tier = "still"\nmotion = "she sips'))
    clip(film / "shots" / "01-sit.mp4")
    r = subprocess.run([str(ANIMATE), "check", "story.toml", "--shot", "01-sit", "--motion"], capture_output=True,
                       text=True, env=with_ffmpeg(ollama), cwd=film)
    assert r.returncode == 0 and not FakeOllama.requests


def test_the_ranking_marks_down_things_passing_through_each_other(tmp_path):
    (tmp_path / "story.toml").write_text(BIBLE_STORY)
    q = sa.rank_prompt(sa.load_story(tmp_path / "story.toml"), sa.load_story(tmp_path / "story.toml").shots[0])
    assert "passing through" in q and "sitting on nothing" in q


def test_the_writer_frames_people_at_tables_from_the_waist_up():
    p = sa.writer_prompt("an ad for a bakery", shots=6)
    assert "waist up" in p and "sitting on a chair" in p


# --- variety: light, framing and gestures ------------------------------------------------------------

NIGHT_STORY = '''title = "Night Shift"
look = "cel-shaded anime"
still_style = "cel-shaded anime still"
lighting = "late night, candlelight from the left"

[characters]
LYRA = "a young woman with long black hair"

[places]
ROOM = "a dark bedroom with candles"

[angles.WIDE]
frame = "wide shot of the whole room"

[[shot]]
id = "01-code"
angle = "WIDE"
still = "LYRA typing in the ROOM"
motion = "static camera, she types"

[[shot]]
id = "02-dawn"
still = "LYRA stretching in the ROOM"
motion = "static camera, she stretches"
lighting = "dawn, pale blue light through the window"

[[shot]]
id = "03-laptop"
still = "close-up of an open laptop on a desk, its bright blank white screen"
motion = "static camera"
'''


def night(tmp_path):
    (tmp_path / "story.toml").write_text(NIGHT_STORY)
    return sa.load_story(tmp_path / "story.toml")


def test_a_shot_can_have_its_own_light(tmp_path):
    story = night(tmp_path)
    code, dawn = story.shots[0], story.shots[1]
    assert "late night" in sa.still_prompt(story, code) and "late night" in sa.motion_prompt(story, code)
    for text in (sa.still_prompt(story, dawn), sa.motion_prompt(story, dawn), "\n".join(sa.continuity_facts(story, dawn))):
        assert "dawn, pale blue light" in text and "late night" not in text


def test_the_writer_gives_a_shot_its_own_light_when_the_time_of_day_changes(tmp_path):
    shot = sa.draft_schema()["properties"]["shots"]["items"]
    assert "lighting" in shot["required"] and shot["properties"]["lighting"]["type"] == "string"
    p = sa.writer_prompt("an ad for a bakery", shots=6)
    assert "time of day" in p and 'own "lighting"' in p
    d = draft()
    d["shots"][2]["lighting"] = "dawn, pale gold light"
    d["shots"][0]["lighting"] = ""
    f = tmp_path / "story.toml"
    f.write_text(sa.draft_to_toml(d, idea="a keeper and a gull"))
    story = sa.load_story(f)
    assert [s.lighting for s in story.shots] == ["", "", "dawn, pale gold light"]


def test_a_place_is_seen_from_each_shots_own_camera(tmp_path):
    story = night(tmp_path)
    look = sa.look_path(story, "ROOM")
    look.parent.mkdir(parents=True, exist_ok=True)
    look.write_bytes(b"\x89PNG a room")
    prompt, refs = sa.render_prompt(story, story.shots[0])
    assert refs == [look]
    assert "layout" not in prompt and "this shot's own camera angle" in prompt


def test_a_picture_with_no_one_and_nowhere_named_still_gets_the_films_style(tmp_path):
    story = night(tmp_path)
    assert sa.render_prompt(story, story.shots[2])[0] == sa.still_prompt(story, story.shots[2])   # no looks yet
    look = sa.look_path(story, "LYRA")
    look.parent.mkdir(parents=True, exist_ok=True)
    look.write_bytes(b"\x89PNG LYRA")             # the other scenes copy her drawn look's style
    prompt, refs = sa.render_prompt(story, story.shots[2])
    assert not refs and prompt.startswith(f"an illustration in this style: {story.look}:")


def test_lint_never_cuts_between_two_shots_from_the_same_angle():
    d = draft(angles=[{"name": "WIDE", "frame": "wide"}, {"name": "CLOSE", "frame": "close-up"}])
    for s, a in zip(d["shots"], ("WIDE", "WIDE", "CLOSE")):
        s["angle"] = a
    assert any("01-light and 02-storm" in x and "angle" in x for x in sa.lint_draft(d, 3))
    d["shots"][1]["angle"] = "CLOSE"
    d["shots"][2]["angle"] = "WIDE"
    assert not any("same angle" in x for x in sa.lint_draft(d, 3))


def test_the_writer_gives_people_a_gesture_and_screens_a_blank_page():
    p = sa.writer_prompt("an ad for a bakery", shots=6)
    assert "gesture" in p and "never two shots in a row from the same angle" in p
    assert "blank white screen" in p
    assert sa.MAX_FULL_SHOTS == 3


def test_lint_keeps_words_off_drawn_screens():
    d = draft()
    d["shots"][0]["still"] = "a laptop screen showing the BotGuild.ai website with job posts"
    d["shots"][1]["still"] = "over the shoulder of KEEPER, his monitor displays a 'Gig Won' notice"
    d["shots"][2]["still"] = "KEEPER staring at a glowing monitor"
    bad = [x for x in sa.lint_draft(d, 3) if "blank white screen" in x]
    assert len(bad) == 2 and "01-light" in bad[0] and "02-storm" in bad[1]
    d["shots"][0]["still"] = "close-up of a laptop, its bright blank white screen showing a web page"
    assert not any("01-light" in x and "blank white screen" in x for x in sa.lint_draft(d, 3))


def test_lint_names_whose_hands_and_shoulders_a_shot_shows():
    d = draft()
    d["shots"][1]["still"] = "over the shoulder at a laptop, hands on the keys"
    assert any("02-storm" in x and "whose" in x for x in sa.lint_draft(d, 3))
    d["shots"][1]["still"] = "over KEEPER's shoulder at a laptop, his hands on the keys"
    assert not any("whose" in x for x in sa.lint_draft(d, 3))


def test_lint_keeps_birds_out_of_the_sound():
    d = draft()
    d["shots"][0]["sfx"] = "morning birds chirping, a satisfied exhale"
    d["shots"][2]["sfx"] = "quiet birdsong outside"
    bad = [x for x in sa.lint_draft(d, 3) if "bird" in x]
    assert len(bad) == 2 and "01-light" in bad[0] and "03-" in bad[1]
    assert "bird" not in json.dumps(sa.WRITER_EXAMPLE).lower()     # the worked example never teaches birds
