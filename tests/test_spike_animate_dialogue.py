"""Characters who talk: voices in the story file, dialogue lines on shots, line timing, splitting a
one-pass scene recording into lines, and the writer's dialogue rules."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import spike_animate as sa  # noqa: E402

HEAD = 'title="Robots"\nlook="2D cartoon animation"\nstill_style="cartoon still"\n'
CHARS = '''
[characters]
JUNK = "a pile of scrap"

[characters.TINY]
look = "a tiny rusty robot"
voice = "small, bright, chirpy robot voice"
sample = "Look at all this stuff!"

[characters.NEWBIE]
look = "a big clumsy robot"
voice = "big, slow, deep and gentle"
'''


def story_of(tmp_path, body, head=HEAD + CHARS):
    p = tmp_path / "story.toml"
    p.write_text(head + body)
    return sa.load_story(p)


TALK = '''
[[shot]]
id = "01-hello"
still = "TINY waving at NEWBIE"
motion = "static camera, a small robot waves and talks"
dialogue = [{ speaker = "TINY", line = "Hello? Can you hear me?", emotion = "hopeful" }]

[[shot]]
id = "02-reply"
still = "close-up of NEWBIE blinking"
motion = "static camera, a big robot blinks and talks"
dialogue = [{ speaker = "NEWBIE", line = "I can hear you." }, { speaker = "NEWBIE", line = "Where am I?" }]
'''


# --- story file -------------------------------------------------------------

def test_characters_can_have_a_voice_and_shots_can_have_dialogue(tmp_path):
    s = story_of(tmp_path, TALK)
    assert s.characters == {"JUNK": "a pile of scrap", "TINY": "a tiny rusty robot", "NEWBIE": "a big clumsy robot"}
    assert s.voices["TINY"].voice == "small, bright, chirpy robot voice"
    assert s.voices["TINY"].sample == "Look at all this stuff!"
    assert s.voices["NEWBIE"].sample == sa.DEFAULT_VOICE_SAMPLE      # a sample is optional
    assert "JUNK" not in s.voices                                    # plain-string characters don't talk
    hello, reply = s.shots
    assert hello.dialogue == [sa.Line("TINY", "Hello? Can you hear me?", "hopeful")]
    assert [l.line for l in reply.dialogue] == ["I can hear you.", "Where am I?"] and reply.dialogue[0].emotion == ""
    assert s.dialogue_mode == "clone"
    assert sa.still_prompt(s, hello) == "a tiny rusty robot waving at a big clumsy robot, cartoon still"


def test_dialogue_mistakes_are_named(tmp_path):
    with pytest.raises(sa.StoryError, match="'ROBO'.*not a character"):
        story_of(tmp_path, '[[shot]]\nid="a"\nstill="s"\nmotion="m"\ndialogue=[{speaker="ROBO", line="hi"}]\n')
    with pytest.raises(sa.StoryError, match="JUNK.*no voice"):
        story_of(tmp_path, '[[shot]]\nid="a"\nstill="s"\nmotion="m"\ndialogue=[{speaker="JUNK", line="hi"}]\n')
    with pytest.raises(sa.StoryError, match="mode"):
        story_of(tmp_path, '[dialogue]\nmode="opera"\n' + TALK)
    with pytest.raises(sa.StoryError, match="look"):
        story_of(tmp_path, '[[shot]]\nid="a"\nstill="s"\nmotion="m"\n',
                 head=HEAD + '[characters.TINY]\nvoice="v"\n')


def test_a_shot_can_have_narration_then_dialogue(tmp_path):
    s = story_of(tmp_path, '[narrator]\nvoice="v"\nsample="s"\n[[shot]]\nid="a"\nstill="s"\nmotion="m"\n'
                           'narration="Then he spoke."\ndialogue=[{speaker="TINY", line="hi"}]\n')
    assert s.shots[0].narration == "Then he spoke." and s.shots[0].dialogue[0].line == "hi"
    assert sa.spoken_text(s) == "Then he spoke. hi"


def test_dialogue_waits_for_the_narrator_to_finish():
    assert sa.dialogue_lead(narration_seconds=2.0, narrator_lead=0.5) == pytest.approx(2.8)
    assert sa.dialogue_lead(narration_seconds=None, narrator_lead=0.5) == pytest.approx(0.4)


def test_scene_mode_can_be_chosen(tmp_path):
    assert story_of(tmp_path, '[dialogue]\nmode="scene"\n' + TALK).dialogue_mode == "scene"


# --- timing -----------------------------------------------------------------

def test_lines_in_a_shot_play_one_after_another():
    starts, overflow = sa.place_lines(shot_start=10.0, durations=[1.5, 1.2], lead=0.4, gap=0.3)
    assert starts == pytest.approx([10.4, 12.2])
    assert overflow == pytest.approx(0.0)


def test_a_line_running_past_the_cut_is_reported():
    starts, overflow = sa.place_lines(shot_start=0.0, durations=[3.0, 2.5], lead=0.4, gap=0.3)
    assert starts == pytest.approx([0.4, 3.7])
    assert overflow == pytest.approx(3.7 + 2.5 - sa.shot_seconds())


def test_voice_lines_in_film_order(tmp_path):
    s = story_of(tmp_path, TALK)
    assert [(sid, l.speaker, l.line) for sid, l in sa.dialogue_lines(s)] == [
        ("01-hello", "TINY", "Hello? Can you hear me?"),
        ("02-reply", "NEWBIE", "I can hear you."), ("02-reply", "NEWBIE", "Where am I?")]
    assert sa.spoken_text(s) == "Hello? Can you hear me? I can hear you. Where am I?"


# --- splitting a one-pass scene into lines ---------------------------------

def test_a_scene_recording_is_cut_at_its_longest_pauses():
    silences = [(1.9, 2.4), (3.0, 3.1), (5.0, 5.8), (8.0, 8.3)]   # (start, end) seconds
    cuts = sa.split_at_silences(silences, total=10.0, n=3)
    assert cuts == pytest.approx([(0.0, 2.15), (2.15, 5.4), (5.4, 10.0)])


def test_too_few_pauses_is_an_error():
    with pytest.raises(sa.StageError, match="3 lines.*1 pause"):
        sa.split_at_silences([(2.0, 2.5)], total=6.0, n=3)


# --- the writer -------------------------------------------------------------

def talking_draft():
    shot = {"still": "s", "motion": "static camera, talking", "sfx": "", "narration": "", "intensity": 0.5, "action": False}
    return {
        "title": "Robots", "look": "2D cartoon animation, junkyard", "still_style": "2D cartoon animation still, junkyard",
        "characters": [{"name": "TINY", "description": "a tiny rusty robot", "voice": "small chirpy robot voice"},
                       {"name": "NEWBIE", "description": "a big clumsy robot", "voice": "big deep gentle voice"}],
        "narrator": {"voice": "warm storyteller", "sample": "Once, in a junkyard."},
        "music": {"caption": "whimsical", "structure": "[Intro - calm]", "key": "C major"},
        "shots": [
            dict(shot, id="01-hello", still="TINY waving", dialogue=[{"speaker": "TINY", "line": "Hello? Can you hear me?", "emotion": "hopeful"}]),
            dict(shot, id="02-reply", still="NEWBIE blinking", dialogue=[{"speaker": "NEWBIE", "line": "I can hear you.", "emotion": "confused"}]),
        ],
    }


def test_a_talking_draft_round_trips_with_voices_and_dialogue(tmp_path):
    p = tmp_path / "story.toml"
    p.write_text(sa.draft_to_toml(talking_draft(), idea="two robots meet"))
    s = sa.load_story(p)
    assert s.voices["NEWBIE"].voice == "big deep gentle voice"
    assert s.shots[0].dialogue == [sa.Line("TINY", "Hello? Can you hear me?", "hopeful")]
    assert sa.lint_draft(talking_draft(), shots=2) == []


def test_lint_checks_dialogue():
    d = talking_draft()
    d["shots"][0]["dialogue"][0]["line"] = "Hello there my big new friend, can you hear me at all right now?"
    d["shots"][1]["dialogue"][0]["speaker"] = "GHOST"
    d["shots"][1]["narration"] = "Then, slowly and carefully, the big robot finally spoke up."
    text = "\n".join(sa.lint_draft(d, shots=2))
    assert "01-hello" in text and "words" in text
    assert "GHOST" in text
    assert "02-reply" in text and "narration and dialogue together" in text   # 10 + 4 words is too much for 5 s


def test_writer_prompt_explains_dialogue_staging():
    p = sa.writer_prompt("two robots meet", shots=8)
    assert "dialogue" in p and "one speaker per shot" in p and "voice" in p


def test_the_writer_must_say_how_every_character_sounds():
    assert "voice" in sa.DRAFT_SCHEMA["properties"]["characters"]["items"]["required"]
    # The worked example shows a talking exchange, so the model sees the dialogue shape in use.
    ex = sa.WRITER_EXAMPLE
    assert all("voice" in c for c in ex["characters"]) and any(c["voice"] for c in ex["characters"])
    assert any(s.get("dialogue") for s in ex["shots"])
    assert sa.lint_draft(ex, shots=len(ex["shots"])) == []


def test_a_speaker_without_a_voice_gets_an_actionable_message():
    d = talking_draft()
    d["characters"][1]["voice"] = ""
    assert any("NEWBIE speaks" in p and "give NEWBIE a voice" in p for p in sa.lint_draft(d, shots=2))


def test_character_names_are_one_uppercase_word():
    d = talking_draft()
    d["characters"][0]["name"] = "UNIT-01"
    d["shots"][0]["still"] = "UNIT-01 waving"
    d["shots"][0]["dialogue"][0]["speaker"] = "UNIT-01"
    assert any("UNIT-01" in p and "one UPPERCASE word" in p for p in sa.lint_draft(d, shots=2))


def test_a_retry_edits_the_previous_draft_instead_of_starting_over():
    d = talking_draft()
    p = sa.writer_prompt("two robots meet", shots=2, problems=["shot 02-reply has narration and dialogue"], previous=d)
    assert '"title": "Robots"' in p and "only" in p.lower() and "02-reply" in p


def test_short_narration_before_dialogue_is_fine():
    d = talking_draft()
    d["shots"][1]["narration"] = "Then he spoke."
    assert sa.lint_draft(d, shots=2) == []
