"""Find the blank screen drawn in a scene's picture, so a real page can be laid onto it.
usage: findscreen.py PICTURE
Prints the screen's corners as JSON, [[x, y] x4: top-left, top-right, bottom-right, bottom-left] as 0-1 of
the picture, or null when there is no screen. The writer asks for "its bright blank white screen", so the
screen is the biggest near-white patch; its outline is closed over hands or a cursor resting on an edge.
Needs OpenCV (FastVideo's venv has it)."""
import json
import sys

import cv2
import numpy as np

MIN_AREA = 0.01      # a screen smaller than 1% of the picture is a highlight, not a screen

img = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
if img is None:
    sys.exit(f"cannot read {sys.argv[1]}")
h, w = img.shape[:2]
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask = ((hsv[:, :, 2] > 235) & (hsv[:, :, 1] < 30)).astype(np.uint8) * 255
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
quad = None
if n > 1:
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[i, cv2.CC_STAT_AREA] >= MIN_AREA * w * h:
        cnt = max(cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
                  key=cv2.contourArea)
        hull = cv2.convexHull(cnt)   # bridges whatever covers part of an edge
        for eps in np.linspace(0.005, 0.1, 60):
            q = cv2.approxPolyDP(hull, eps * cv2.arcLength(hull, True), True)
            if len(q) == 4:
                quad = q[:, 0, :].astype(float)
                break
if quad is None:
    print("null")
else:
    s, d = quad.sum(1), quad[:, 0] - quad[:, 1]
    corners = [quad[s.argmin()], quad[d.argmax()], quad[s.argmax()], quad[d.argmin()]]
    print(json.dumps([[round(x / w, 4), round(y / h, 4)] for x, y in corners]))
