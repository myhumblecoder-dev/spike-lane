"""Rank score takes by how well they follow the film: per-shot loudness against the story's
intensity arc, a hit at the story's peak, and a tempo that stays locked to the edit.
usage: score_rank.py SPEC.json OUT.json TAKE.wav [TAKE.wav ...]
SPEC: {"shot_len", "arc": [per-shot intensity], "hit": seconds, "bpm"}
Writes OUT.json: [{"take", "score", "arc_r", "hit", "tempo", "shot_db"}], best first.
Needs librosa (MMAudio's venv has it)."""
import json
import sys
from pathlib import Path

import librosa
import numpy as np

spec = json.loads(Path(sys.argv[1]).read_text())
shot, arc, hit_at, bpm = spec["shot_len"], np.array(spec["arc"], float), spec["hit"], spec["bpm"]

rows = []
for path in sys.argv[3:]:
    y, sr = librosa.load(path, sr=22050, mono=True)
    rms = librosa.feature.rms(y=y, hop_length=512)[0]
    t = librosa.times_like(rms, sr=sr, hop_length=512)
    db = np.array([20 * np.log10(np.sqrt(np.mean(rms[(t >= i * shot) & (t < (i + 1) * shot)] ** 2)) + 1e-9)
                   for i in range(len(arc))])
    arc_r = float(np.corrcoef(db, arc)[0, 1]) if arc.std() > 0 else 0.0
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    win = (t >= hit_at - 0.3) & (t <= hit_at + 0.5)
    hit = float(onset[win].max() / (np.median(onset) + 1e-9)) if win.any() else 0.0
    tempo = float(np.atleast_1d(librosa.beat.beat_track(onset_envelope=onset, sr=sr, hop_length=512,
                                                        start_bpm=bpm)[0])[0])
    locked = min(abs(tempo - m) for m in (bpm, bpm / 2, bpm * 2)) < 4
    score = arc_r + 0.05 * min(hit, 10) + (0.2 if locked else 0)
    rows.append({"take": path, "score": round(score, 3), "arc_r": round(arc_r, 3), "hit": round(hit, 1),
                 "tempo": round(tempo, 1), "shot_db": [round(float(d - db.max()), 1) for d in db]})

rows.sort(key=lambda r: r["score"], reverse=True)
Path(sys.argv[2]).write_text(json.dumps(rows, indent=1))
for r in rows:
    print(f"{Path(r['take']).name:14} score {r['score']:5.2f}  arc r={r['arc_r']:+.2f}  hit {r['hit']:4.1f}x  tempo {r['tempo']:5.1f}")
