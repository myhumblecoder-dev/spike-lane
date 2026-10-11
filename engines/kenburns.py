"""A still shot: a slow camera move over a picture, rendered as a clip (no video model).
usage: kenburns.py STILL OUT.mp4 WIDTH HEIGHT FRAMES CAMERA [OVERLAY.png]
CAMERA: push (in), pull (out), left or right (a pan, slightly zoomed in so the edges never show).
OVERLAY: a transparent picture the still's size (a screen laid on it) to put on the still first.
Every frame is placed with sub-pixel precision, so the move is smooth, not stepped like ffmpeg's zoompan.
Needs OpenCV (FastVideo's venv has it)."""
import subprocess
import sys

import cv2
import numpy as np

GLIDE = 0.08   # how far the camera moves: 8% closer, or the margin that zoom gives a pan

still, dst, W, H, n, camera = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
img = cv2.imread(still, cv2.IMREAD_COLOR)
if img is None:
    sys.exit(f"cannot read {still}")
if len(sys.argv) > 7:
    over = cv2.imread(sys.argv[7], cv2.IMREAD_UNCHANGED)
    a = over[:, :, 3:4].astype(np.float32) / 255
    img = (over[:, :, :3] * a + img * (1 - a)).round().astype(np.uint8)
h, w = img.shape[:2]
cover = max(W / w, H / h)

enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                        "-r", "24", "-i", "-", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", dst],
                       stdin=subprocess.PIPE)
for i in range(n):
    t = i / max(1, n - 1)
    zoom = {"push": 1 + GLIDE * t, "pull": 1 + GLIDE * (1 - t)}.get(camera, 1 + GLIDE)
    s = cover * zoom
    spare = (w * s - W) / 2 / s                     # how far the view can slide either way, in still pixels
    cx = w / 2 + {"left": spare * (1 - 2 * t), "right": -spare * (1 - 2 * t)}.get(camera, 0.0)
    m = np.float32([[s, 0, W / 2 - s * cx], [0, s, H / 2 - s * h / 2]])
    frame = cv2.warpAffine(img, m, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    enc.stdin.write(frame.tobytes())
enc.stdin.close()
sys.exit(enc.wait())
