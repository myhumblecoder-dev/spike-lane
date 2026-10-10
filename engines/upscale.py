"""Upscale a clip frame by frame with a Real-ESRGAN model, then fit it to a size (cover, centre crop).
usage: upscale.py MODEL.pth IN.mp4 OUT.mp4 WIDTH HEIGHT
Runs on the GPU (MPS) when there is one. Needs torch + spandrel + OpenCV (~/upscale's venv)."""
import subprocess
import sys

import cv2
import numpy as np
import spandrel
import torch

model_path, src, dst, W, H = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
dev = "mps" if torch.backends.mps.is_available() else "cpu"
model = spandrel.ModelLoader().load_from_file(model_path).model.to(dev).eval()
if dev == "mps":
    model = model.half()

cap = cv2.VideoCapture(src)
fps = cap.get(cv2.CAP_PROP_FPS) or 24
enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                        "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", dst],
                       stdin=subprocess.PIPE)
with torch.inference_mode():
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        x = torch.from_numpy(frame[:, :, ::-1].copy()).permute(2, 0, 1)[None].to(dev)
        x = (x.half() if dev == "mps" else x.float()) / 255
        y = model(x).clamp(0, 1)[0].permute(1, 2, 0).float().cpu().numpy()[:, :, ::-1]
        big = (y * 255).round().astype(np.uint8)
        k = max(W / big.shape[1], H / big.shape[0])
        big = cv2.resize(big, (round(big.shape[1] * k), round(big.shape[0] * k)), interpolation=cv2.INTER_AREA)
        y0, x0 = (big.shape[0] - H) // 2, (big.shape[1] - W) // 2
        enc.stdin.write(np.ascontiguousarray(big[y0:y0 + H, x0:x0 + W]).tobytes())
enc.stdin.close()
sys.exit(enc.wait())
