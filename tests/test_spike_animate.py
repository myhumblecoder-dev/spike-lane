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


def test_mix_levels_default_to_quiet_beds_and_half_voices(story, tmp_path):
    assert (story.levels.sound, story.levels.music, story.levels.voices) == (20, 20, 50)
    s = sa.load_story(write_story(tmp_path, 'title="x"\nlook="l"\nstill_style="s"\n'
                                  '[mix]\nsound = 35\nvoices = 80\n[[shot]]\nid="01-a"\nstill="s"\nmotion="m"\n'))
    assert (s.levels.sound, s.levels.music, s.levels.voices) == (35, 20, 80)
    with pytest.raises(sa.StoryError, match="mix"):
        sa.load_story(write_story(tmp_path, 'title="x"\nlook="l"\nstill_style="s"\n'
                                  '[mix]\nmusic = -5\n[[shot]]\nid="01-a"\nstill="s"\nmotion="m"\n'))


def test_mix_graph_sets_each_layer_to_its_level_without_renormalizing():
    # master_db brings the old mix to its old loudness; each level is a share of that.
    g = sa.mix_filtergraph(narration_ms=[500], total=10.0, sfx_db=-4, music_db=-10,
                           levels=sa.Levels(sound=20, music=20, voices=50), master_db=3)
    assert "volume=-14.9794dB" in g          # -4 + 3 + 20·log10(0.20)
    assert "volume=-20.9794dB" in g          # -10 + 3 + 20·log10(0.20)
    assert "[nar]volume=-3.0206dB" in g      # 3 + 20·log10(0.50), after the ducking key is split off
    assert "loudnorm" not in g and "alimiter" in g and g.endswith("[out]")
    bed = sa.mix_filtergraph(narration_ms=[], total=10.0, sfx_db=-4, music_db=-10,
                             levels=sa.Levels(sound=20, music=20, voices=50), master_db=3)
    assert "loudnorm" not in bed and "alimiter" in bed and "sidechaincompress" not in bed


# --- the film bible: lighting, places and camera angles shared by every shot ---------------

BIBLE = '''title = "Morning"
look = "cel animation"
still_style = "cel animation still"
lighting = "sunrise, warm low pink-gold light from the left"

[characters.ELENA]
look = "a young woman in a cream linen shirt with rolled sleeves"
voice = "soft"

[places]
TERRACE = "a whitewashed cafe terrace on a white sand beach"

[angles.ACROSS]
frame = "medium shot across the cafe table, the sea behind"
seed = 3002

[[shot]]
id = "01-sit"
angle = "ACROSS"
still = "ELENA sitting at a table on the TERRACE"
motion = "she sips her coffee, mouth closed"

[[shot]]
id = "02-wide"
still = "wide shot of the TERRACE at dawn"
motion = "static camera, waves roll in"
seed = 7
'''


def test_the_film_bible_goes_into_every_picture_and_motion(tmp_path):
    story = sa.load_story(write_story(tmp_path, BIBLE))
    sit, wide = story.shots
    assert sa.still_prompt(story, sit) == (
        "medium shot across the cafe table, the sea behind, "
        "a young woman in a cream linen shirt with rolled sleeves sitting at a table on "
        "a whitewashed cafe terrace on a white sand beach, "
        "sunrise, warm low pink-gold light from the left, cel animation still")
    assert sa.motion_prompt(story, sit) == "cel animation, sunrise, warm low pink-gold light from the left, she sips her coffee, mouth closed"
    assert sit.seed == 3002 and wide.seed == 7                 # an angle keeps its seed; a shot can still set its own
    assert sa.still_prompt(story, wide).startswith("wide shot of a whitewashed cafe terrace")   # no "the a ..."
    assert sa.expand("ZAP and the ZAP", {"ZAP": "the small robot"}) == "the small robot and the small robot"


def test_stories_without_a_bible_keep_their_prompts(story):
    shot = story.shots[0]
    assert sa.still_prompt(story, shot) == f"{sa.expand(shot.still, story.characters)}, {story.still_style}"
    assert sa.motion_prompt(story, shot) == f"{story.look}, {sa.expand(shot.motion, story.characters)}"


def test_the_bible_names_must_make_sense(tmp_path):
    with pytest.raises(sa.StoryError, match="angle 'CLOSE'"):
        sa.load_story(write_story(tmp_path, BIBLE.replace('angle = "ACROSS"', 'angle = "CLOSE"')))
    with pytest.raises(sa.StoryError, match="ELENA"):
        sa.load_story(write_story(tmp_path, BIBLE.replace("TERRACE =", "ELENA =")))
    with pytest.raises(sa.StoryError, match="frame"):
        sa.load_story(write_story(tmp_path, BIBLE.replace('frame = "medium shot across the cafe table, the sea behind"\n', "")))


def test_names_in_an_angle_frame_are_filled_in_too(tmp_path):
    text = BIBLE.replace('frame = "medium shot across the cafe table, the sea behind"', 'frame = "medium shot across a table on the TERRACE"')
    story = sa.load_story(write_story(tmp_path, text))
    assert sa.still_prompt(story, story.shots[0]).startswith("medium shot across a table on a whitewashed cafe terrace")


# --- pace: shots as long as their lines ------------------------------------------------------

VOICE_PACED = STORY.replace('seed = 1024\n', 'seed = 1024\npace = "voice"\n', 1)


def tone(path: Path, secs: float) -> None:
    import subprocess
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"sine=frequency=300:duration={secs}",
                    "-ac", "1", "-ar", "24000", str(path)], check=True)


def test_a_clip_length_is_a_frame_count_the_video_model_can_make():
    assert sa.wan_frames(3.0) == 73              # 72 frames, up to the next 4n+1
    assert sa.wan_frames(0.5) == 49              # never shorter than about 2 s
    assert sa.wan_frames(9.0) == 121             # never longer than the model's 5 s
    assert all((sa.wan_frames(s) - 1) % 4 == 0 for s in (2.1, 2.6, 3.3, 4.4))


def test_without_a_pace_every_shot_stays_five_seconds(story):
    assert story.pace == "fixed"
    assert [sa.shot_frames(story, s) for s in story.shots] == [121, 121, 121]


def test_a_voice_paced_shot_lasts_its_line_and_a_breath(tmp_path):
    story = sa.load_story(write_story(tmp_path, VOICE_PACED))
    first, silent, last = story.shots
    assert sa.shot_frames(story, first) is None                  # its line isn't recorded yet
    tone(tmp_path / "voice" / "01-establish.wav", 2.0)
    assert sa.shot_frames(story, first) == sa.wan_frames(0.5 + 2.0 + sa.TAIL)   # lead, line, breath
    assert sa.shot_frames(story, silent) == sa.wan_frames(sa.SILENT_SECONDS)


def test_a_voice_paced_shot_can_be_given_its_own_length(tmp_path):
    text = VOICE_PACED.replace('id = "02-pounce"\n', 'id = "02-pounce"\nseconds = 4.2\n', 1)
    story = sa.load_story(write_story(tmp_path, text))
    assert sa.shot_frames(story, story.shots[1]) == sa.wan_frames(4.2)


def test_a_bad_pace_or_length_is_refused(tmp_path):
    with pytest.raises(sa.StoryError, match="pace"):
        sa.load_story(write_story(tmp_path, STORY.replace('seed = 1024\n', 'seed = 1024\npace = "fast"\n', 1)))
    with pytest.raises(sa.StoryError, match="seconds"):
        sa.load_story(write_story(tmp_path, VOICE_PACED.replace('id = "02-pounce"\n', 'id = "02-pounce"\nseconds = 9\n', 1)))


def test_voice_paced_lines_start_where_their_shots_start(tmp_path):
    story = sa.load_story(write_story(tmp_path, VOICE_PACED))
    tone(tmp_path / "voice" / "01-establish.wav", 2.0)
    tone(tmp_path / "voice" / "03-stalk.wav", 1.0)
    lens = [sa.shot_frames(story, s) / sa.FPS for s in story.shots]
    assert sa.shot_starts(story) == pytest.approx([0, lens[0], lens[0] + lens[1]])
    assert sa.narration_offsets(story) == [("01-establish", pytest.approx(0.5)),
                                           ("03-stalk", pytest.approx(lens[0] + lens[1] + 0.5))]
    assert sa.film_length(story) == pytest.approx(sum(lens))


# --- titles: captions and the end card --------------------------------------------------------

def test_captions_are_on_and_show_the_narration_unless_a_shot_has_its_own(tmp_path):
    text = STORY.replace('id = "03-stalk"\n', 'id = "03-stalk"\ncaption = "Closer."\n', 1)
    story = sa.load_story(write_story(tmp_path, text))
    assert story.captions is True
    assert [sa.caption_text(story, s) for s in story.shots] == ["Deep in the winter woods, the fox was hungry.", "", "Closer."]
    hidden = sa.load_story(write_story(tmp_path, text.replace('id = "01-establish"\n', 'id = "01-establish"\ncaption = ""\n', 1)))
    assert sa.caption_text(hidden, hidden.shots[0]) == ""                   # an empty caption hides the line
    off = sa.load_story(write_story(tmp_path, text.replace('seed = 1024\n', 'seed = 1024\ncaptions = false\n', 1)))
    assert [sa.caption_text(off, s) for s in off.shots] == ["", "", ""]


END = '\n[end_card]\ntext = "Put your agent to work."\nsub = "BotGuild.ai"\n'


def test_the_end_card_holds_the_last_frame_and_replaces_the_last_caption(tmp_path):
    story = sa.load_story(write_story(tmp_path, STORY.replace("\n[[shot]]", END + "\n[[shot]]", 1)))
    assert story.end_card == sa.EndCard(text="Put your agent to work.", sub="BotGuild.ai", hold=1.5)
    L = sa.SHOT_FRAMES / sa.FPS
    assert sa.film_length(story) == pytest.approx(3 * L + 1.5)
    plan = sa.title_plan(story)
    assert plan == [
        sa.Title("Deep in the winter woods, the fox was hungry.", "caption", 0.0, pytest.approx(L)),
        sa.Title("Put your agent to work.\nBotGuild.ai", "end", pytest.approx(2 * L + 0.5), pytest.approx(3 * L + 1.5)),
    ]


def test_without_an_end_card_the_last_caption_stays(story):
    plan = sa.title_plan(story)
    L = sa.SHOT_FRAMES / sa.FPS
    assert [(t.text, t.kind) for t in plan] == [("Deep in the winter woods, the fox was hungry.", "caption"),
                                                ("Slowly. Closer.", "caption")]
    assert (plan[1].start, plan[1].end) == (pytest.approx(2 * L), pytest.approx(3 * L))
    assert sa.film_length(story) == pytest.approx(3 * L)


def test_a_film_wide_grade_is_neutral_unless_set(story, tmp_path):
    assert sa.grade_filter(story) == ""
    text = STORY + "\n[grade]\nwarmth = 0.1\nsaturation = 1.2\ncontrast = 1.05\n"
    graded = sa.load_story(write_story(tmp_path, text))
    assert graded.grade == sa.Grade(warmth=0.1, saturation=1.2, contrast=1.05)
    f = sa.grade_filter(graded)
    assert "saturation=1.2" in f and "contrast=1.05" in f and "rm=0.1" in f and "bm=-0.1" in f
    with pytest.raises(sa.StoryError, match="warmth"):
        sa.load_story(write_story(tmp_path, STORY + "\n[grade]\nwarmth = 2\n"))


# --- screens: a real picture laid onto a screen in the shot ------------------------------------

SCREEN = 'screen = { picture = "reference/page.png", corners = [[0.3, 0.2], [0.7, 0.2], [0.7, 0.6], [0.3, 0.6]] }\n'


def test_a_shot_can_show_a_real_picture_on_a_screen(tmp_path):
    (tmp_path / "reference").mkdir()
    (tmp_path / "reference" / "page.png").write_bytes(b"x")
    story = sa.load_story(write_story(tmp_path, STORY.replace('id = "02-pounce"\n', 'id = "02-pounce"\n' + SCREEN, 1)))
    s = story.shots[1].screen
    assert s.picture == (tmp_path / "reference" / "page.png").resolve()
    assert s.corners == [(0.3, 0.2), (0.7, 0.2), (0.7, 0.6), (0.3, 0.6)]
    assert story.shots[0].screen is None


def test_a_screen_needs_its_picture_and_four_corners(tmp_path):
    with pytest.raises(sa.StoryError, match="page.png"):
        sa.load_story(write_story(tmp_path, STORY.replace('id = "02-pounce"\n', 'id = "02-pounce"\n' + SCREEN, 1)))
    (tmp_path / "reference").mkdir()
    (tmp_path / "reference" / "page.png").write_bytes(b"x")
    three = SCREEN.replace(", [0.3, 0.6]]", "]")
    with pytest.raises(sa.StoryError, match="4 corners"):
        sa.load_story(write_story(tmp_path, STORY.replace('id = "02-pounce"\n', 'id = "02-pounce"\n' + three, 1)))


def test_a_point_on_the_still_lands_where_the_finished_film_shows_it():
    # the still is cropped to the clip (cover, centre), and the clip to the finished film the same way
    still, clip, fin = (1248, 720), (832, 448), (1280, 720)
    assert sa.still_to_finish((0.5, 0.5), still, clip, fin) == pytest.approx((640, 360))
    x, y = sa.still_to_finish((0.0, 0.0), still, clip, fin)
    s1 = 832 / 1248                                       # still → clip: widths match, 16 px cropped top and bottom
    y_clip = (0 * 720 * s1 - (720 * s1 - 448) / 2)
    s2 = 720 / 448                                        # clip → film: heights match, sides cropped
    assert (x, y) == pytest.approx((0 - (832 * s2 - 1280) / 2, y_clip * s2))


def test_a_screen_picture_can_skip_its_top(tmp_path):
    (tmp_path / "reference").mkdir()
    (tmp_path / "reference" / "page.png").write_bytes(b"x")
    text = STORY.replace('id = "02-pounce"\n', 'id = "02-pounce"\n' + SCREEN.replace("] }", "], top = 0.05 }"), 1)
    story = sa.load_story(write_story(tmp_path, text))
    assert story.shots[1].screen.top == 0.05
    plain = sa.load_story(write_story(tmp_path, STORY.replace('id = "02-pounce"\n', 'id = "02-pounce"\n' + SCREEN, 1)))
    assert plain.shots[1].screen.top == 0.0
