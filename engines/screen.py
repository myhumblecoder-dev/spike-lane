"""Warp a real picture (a web page, an app) onto a screen in the film, as a transparent overlay.
usage: screen.py SPEC.json
SPEC: {"size": [W, H], "items": [{"picture", "corners": [[x, y] x4: top-left, top-right, bottom-right,
bottom-left, in the film's pixels], "top": 0-1 of the picture to skip first, "out": PNG}]}
The picture is cropped to the screen's shape from the top (as a browser shows a page), then fitted to the
screen's four corners; its edge is anti-aliased so it sits in the frame.
Needs OpenCV (FastVideo's venv has it)."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

spec = json.loads(Path(sys.argv[1]).read_text())
W, H = spec["size"]
SS = 2   # draw the mask at twice the size, then shrink it: smooth edges

for item in spec["items"]:
    pic = cv2.imread(item["picture"], cv2.IMREAD_COLOR)
    if pic is None:
        sys.exit(f"cannot read {item['picture']}")
    dst = np.float32(item["corners"])
    pic = pic[int(pic.shape[0] * item.get("top", 0.0)):]
    side = lambda a, b: float(np.linalg.norm(dst[a] - dst[b]))   # noqa: E731
    aspect = (side(0, 1) + side(3, 2)) / (side(0, 3) + side(1, 2))   # the screen's width : height
    h, w = pic.shape[:2]
    if w / h < aspect:      # taller than the screen: keep its top
        pic = pic[:max(1, round(w / aspect))]
    else:                   # wider: keep its middle
        x0 = round((w - h * aspect) / 2)
        pic = pic[:, x0:x0 + round(h * aspect)]
    target = 1.5 * max(side(0, 1), side(3, 2))           # shrink a big screenshot smoothly first: crisp small text
    if pic.shape[1] > target:
        pic = cv2.resize(pic, (round(target), max(1, round(pic.shape[0] * target / pic.shape[1]))),
                         interpolation=cv2.INTER_AREA)
    h, w = pic.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    m = cv2.getPerspectiveTransform(src, dst)
    color = cv2.warpPerspective(pic, m, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    big = cv2.warpPerspective(np.full((h, w), 255, np.uint8),
                              cv2.getPerspectiveTransform(src, dst * SS), (W * SS, H * SS), flags=cv2.INTER_LINEAR)
    alpha = cv2.resize(big, (W, H), interpolation=cv2.INTER_AREA)
    cv2.imwrite(item["out"], np.dstack([color, alpha]))
