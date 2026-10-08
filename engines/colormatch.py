"""Grade every frame of a clip toward a reference still (Reinhard mean/std transfer in LAB).
usage: colormatch.py REF.png IN.mp4 OUT.mp4"""
import subprocess, sys

import cv2
import numpy as np

ref_path, src, dst = sys.argv[1:4]
cap = cv2.VideoCapture(src)
fps = cap.get(cv2.CAP_PROP_FPS)
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Reference cropped exactly like the I2V script crops it (cover, then center-crop).
ref = cv2.imread(ref_path)
s = max(w / ref.shape[1], h / ref.shape[0])
ref = cv2.resize(ref, (round(ref.shape[1] * s), round(ref.shape[0] * s)), interpolation=cv2.INTER_AREA)
y0, x0 = (ref.shape[0] - h) // 2, (ref.shape[1] - w) // 2
ref = cv2.cvtColor(ref[y0:y0 + h, x0:x0 + w], cv2.COLOR_BGR2LAB).astype(np.float32)
r_mean, r_std = ref.reshape(-1, 3).mean(0), ref.reshape(-1, 3).std(0)

enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                        "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", dst],
                       stdin=subprocess.PIPE)
while True:
    ok, frame = cap.read()
    if not ok:
        break
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    f_mean, f_std = lab.reshape(-1, 3).mean(0), lab.reshape(-1, 3).std(0)
    lab = (lab - f_mean) * (r_std / np.maximum(f_std, 1e-3)) + r_mean
    enc.stdin.write(cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).tobytes())
enc.stdin.close()
sys.exit(enc.wait())
