"""engines/findscreen.py: where a scene's drawn blank screen is, so a real page can be laid onto it.
Runs the real engine in FastVideo's venv (it needs OpenCV) on pictures drawn here."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
VENV = Path.home() / "FastVideo" / ".venv" / "bin" / "python"
pytestmark = pytest.mark.skipif(not VENV.exists(), reason="needs FastVideo's venv (OpenCV)")

# A dark room 1248x720 with a bright tilted laptop screen, a candle flame (small and bright), and a hand
# resting over the screen's bottom edge, as in an over-the-shoulder shot.
DRAW = r'''
import sys, cv2, numpy as np
out, screen, hand = sys.argv[1], sys.argv[2] == "1", sys.argv[3] == "1"
img = np.full((720, 1248, 3), (40, 20, 50), np.uint8)
img = cv2.add(img, np.random.default_rng(1).integers(0, 25, img.shape, dtype=np.uint8))
if screen:
    cv2.fillConvexPoly(img, np.int32([[743, 304], [1083, 328], [1023, 586], [689, 502]]), (250, 250, 250))
cv2.ellipse(img, (1150, 600), (14, 30), 0, 0, 360, (235, 245, 255), -1)          # a candle flame
if hand:
    cv2.ellipse(img, (850, 560), (70, 45), 0, 0, 360, (150, 160, 220), -1)         # skin over the bottom edge
cv2.imwrite(out, img)
'''


def picture(tmp_path, screen=True, hand=True) -> Path:
    out = tmp_path / "still.png"
    subprocess.run([str(VENV), "-c", DRAW, str(out), str(int(screen)), str(int(hand))], check=True)
    return out


def find(path: Path):
    r = subprocess.run([str(VENV), str(ROOT / "engines" / "findscreen.py"), str(path)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def test_it_finds_the_screens_four_corners_even_with_a_hand_over_its_edge(tmp_path):
    got = find(picture(tmp_path))
    want = [[743 / 1248, 304 / 720], [1083 / 1248, 328 / 720], [1023 / 1248, 586 / 720], [689 / 1248, 502 / 720]]
    assert got is not None and len(got) == 4
    for (x, y), (wx, wy) in zip(got, want):     # top-left, top-right, bottom-right, bottom-left
        assert abs(x - wx) < 0.006 and abs(y - wy) < 0.006


def test_it_says_so_when_there_is_no_screen(tmp_path):
    assert find(picture(tmp_path, screen=False)) is None
