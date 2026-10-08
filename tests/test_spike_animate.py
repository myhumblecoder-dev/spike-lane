"""spike-animate's planning rules: the story file, prompts, timing, caching, mixing and checks.
These are pure functions; the model runs are covered by the CLI tests with stand-in engines."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import spike_animate as sa  # noqa: E402

STORY = """
title = "Fox Hunt"
seed = 1024
look = "2D cartoon animation, storybook colors"
still_style = "2D cartoon animation still, bold black outlines"

[characters]
FOX = "a slender red fox with a white-tipped tail"
RAB = "a plump brown rabbit"

[narrator]
voice = "a deep warm storyteller"
sample = "Deep in the winter woods, every creature knows one rule."
lead = 0.5

[music]
caption = "dark fairy tale chamber orchestra"
structure = "[Intro - hushed]"
key = "D minor"
takes = 4

[[shot]]
id = "01-establish"
still = "wide shot of FOX walking through snow"
motion = "static camera, a red fox walks forward"
sfx = "paws crunching in snow"
narration = "Deep in the winter woods, the fox was hungry."
intensity = 0.2

[[shot]]
id = "02-pounce"
still = "FOX leaping at RAB"
motion = "the fox leaps and pounces"
anchor = 0
intensity = 1.0

[[shot]]
id = "03-stalk"
still = "FOX creeping through snow"
motion = "the fox creeps forward"
intensity = 0.4
narration = "Slowly. Closer."

[shot.reference]
clip = "reference/stalk.mp4"
start = 0.5
"""


@pytest.fixture
def story(tmp_path):
    p = tmp_path / "story.toml"
    p.write_text(STORY)
    return sa.load_story(p)


def write_story(tmp_path, text):
    p = tmp_path / "story.toml"
    p.write_text(text)
    return p


# --- story file -------------------------------------------------------------

def test_loads_shots_in_order_with_defaults(story, tmp_path):
    assert story.title == "Fox Hunt" and story.seed == 1024
    assert [s.id for s in story.shots] == ["01-establish", "02-pounce", "03-stalk"]
    first, pounce, stalk = story.shots
    assert first.anchor == 1 and pounce.anchor == 0      # light anchoring unless the shot says otherwise
    assert first.seed == 1024                             # shots inherit the film seed
    assert pounce.sfx == "" and pounce.narration == ""
    assert first.colormatch and not stalk.colormatch      # real-clip shots skip colour-matching
    assert stalk.reference.clip == tmp_path / "reference" / "stalk.mp4"   # relative to the story file
    assert stalk.reference.start == 0.5
    assert stalk.reference.sigma == 0.88 and stalk.reference.first_strength == 0.4
    assert first.reference is None


def test_story_errors_name_the_problem(tmp_path):
    with pytest.raises(sa.StoryError, match="still"):
        sa.load_story(write_story(tmp_path, 'title="x"\nlook="l"\nstill_style="s"\n[[shot]]\nid="a"\nmotion="m"\n'))
    with pytest.raises(sa.StoryError, match="duplicate shot id 'a'"):
        sa.load_story(write_story(tmp_path, 'title="x"\nlook="l"\nstill_style="s"\n'
                                  '[[shot]]\nid="a"\nstill="s"\nmotion="m"\n[[shot]]\nid="a"\nstill="s"\nmotion="m"\n'))
    with pytest.raises(sa.StoryError, match="no shots"):
        sa.load_story(write_story(tmp_path, 'title="x"\nlook="l"\nstill_style="s"\n'))


def test_narration_needs_a_narrator(tmp_path):
    with pytest.raises(sa.StoryError, match="narrator"):
        sa.load_story(write_story(tmp_path, 'title="x"\nlook="l"\nstill_style="s"\n'
                                  '[[shot]]\nid="a"\nstill="s"\nmotion="m"\nnarration="hello"\n'))


# --- prompts ----------------------------------------------------------------

def test_character_names_expand_to_their_fixed_description(story):
    assert sa.expand("FOX leaping at RAB", story.characters) == \
        "a slender red fox with a white-tipped tail leaping at a plump brown rabbit"
    assert sa.expand("FOXES and FOXY stay", story.characters) == "FOXES and FOXY stay"  # whole words only


def test_still_and_motion_prompts(story):
    pounce = story.shots[1]
    assert sa.still_prompt(story, pounce) == ("a slender red fox with a white-tipped tail leaping at a plump "
                                              "brown rabbit, 2D cartoon animation still, bold black outlines")
    assert sa.motion_prompt(story, pounce) == "2D cartoon animation, storybook colors, the fox leaps and pounces"


def test_a_shot_can_override_the_still_style(tmp_path):
    story = sa.load_story(write_story(tmp_path, 'title="x"\nlook="l"\nstill_style="film style"\n'
                                      '[[shot]]\nid="a"\nstill="s"\nmotion="m"\n'
                                      '[[shot]]\nid="b"\nstill="s"\nmotion="m"\nstill_style="gorier style"\n'))
    assert sa.still_prompt(story, story.shots[0]) == "s, film style"
    assert sa.still_prompt(story, story.shots[1]) == "s, gorier style"


# --- timing -----------------------------------------------------------------

def test_every_shot_is_121_frames_at_24_fps():
    assert sa.SHOT_FRAMES == 121 and sa.FPS == 24
    assert sa.shot_seconds() == pytest.approx(5.041667, abs=1e-5)


def test_score_tempo_puts_every_cut_on_a_bar_line():
    # 8 beats (two bars of 4/4) per 5.04 s shot.
    assert sa.score_bpm(sa.shot_seconds()) == 95


def test_narration_starts_lead_seconds_into_its_shot(story):
    L = sa.shot_seconds()
    assert sa.narration_offsets(story) == [("01-establish", pytest.approx(0.5)),
                                          ("03-stalk", pytest.approx(2 * L + 0.5))]


def test_story_arc_and_hit_point_come_from_shot_intensity(story):
    assert sa.intensity_arc(story) == [0.2, 1.0, 0.4]
    assert sa.hit_time(story) == pytest.approx(sa.shot_seconds())   # start of the first most-intense shot


# --- caching ----------------------------------------------------------------

def test_keys_change_only_when_inputs_change(tmp_path):
    keys = sa.Keys(tmp_path)
    out = tmp_path / "board" / "a.png"
    k = sa.key_for("prompt", 1024)
    assert keys.stale(out, k)                 # never made
    out.parent.mkdir()
    out.write_bytes(b"png")
    keys.record(out, k)
    assert not sa.Keys(tmp_path).stale(out, k)          # persisted across runs
    assert sa.Keys(tmp_path).stale(out, sa.key_for("prompt", 7))
    out.unlink()
    assert sa.Keys(tmp_path).stale(out, k)              # deleted output is stale even with a matching key


def test_file_digest_tracks_content(tmp_path):
    f = tmp_path / "x"
    f.write_bytes(b"one")
    a = sa.file_digest(f)
    f.write_bytes(b"two")
    assert sa.file_digest(f) != a


# --- checks -----------------------------------------------------------------

def test_reads_the_prompt_mflux_stored_in_the_png(tmp_path):
    png = tmp_path / "k.png"
    png.write_bytes(b"\x89PNG....eXIf....ASCII\x00\x00\x00"
                    b'{"mflux_version": "0.21.0", "seed": 1, "prompt": "a red fox, cartoon"}\x00IEND')
    assert sa.png_prompt(png) == "a red fox, cartoon"
    (tmp_path / "bare.png").write_bytes(b"\x89PNG no metadata")
    assert sa.png_prompt(tmp_path / "bare.png") is None


def test_heard_speech_matches_the_line_despite_punctuation_and_homophones():
    assert sa.words_match("Then... a scent, on the wind.", "Then, ascent on the wind.") >= 0.8
    assert sa.words_match("Slowly. Silently. Closer.", "Slowly, silently, closer.") == 1.0
    assert sa.words_match("Slowly. Silently. Closer.", "In the winter woods") < 0.5


# --- mixing -----------------------------------------------------------------

def test_mix_graph_ducks_the_bed_under_the_narrator():
    g = sa.mix_filtergraph(narration_ms=[500, 10583], total=40.3333, sfx_db=-4, music_db=-10)
    assert "[3:a]" in g and "[4:a]" in g and "[5:a]" not in g          # inputs 0-2 are picture, sfx, music
    assert "adelay=500:all=1" in g and "adelay=10583:all=1" in g
    assert "amix=inputs=2:normalize=0,apad,atrim=0:40.3333" in g
    assert "volume=-4dB" in g and "volume=-10dB" in g
    assert "sidechaincompress" in g and g.rstrip(";").endswith("[out]")
    assert "loudnorm=I=-16" in g


def test_mix_graph_without_narration_is_just_the_bed():
    g = sa.mix_filtergraph(narration_ms=[], total=10.0, sfx_db=-4, music_db=-10)
    assert "sidechaincompress" not in g and "[3:a]" not in g and g.endswith("[out]")
