"""Generate takes of a film score with ACE-Step 1.5 (turbo DiT on MLX, no LM planner).
One model load, one take per seed.   usage: score_gen.py SPEC.json
SPEC: {"out", "duration", "bpm", "key", "caption", "structure", "seeds": [...]}
Writes OUT/take-s<seed>.wav. Run with ACE-Step's venv python from the ACE-Step-1.5 checkout."""
import json
import sys
import time
from pathlib import Path

from acestep.handler import AceStepHandler
from acestep.inference import GenerationConfig, GenerationParams, generate_music
from acestep.llm_inference import LLMHandler

spec = json.loads(Path(sys.argv[1]).read_text())
out = Path(spec["out"])
out.mkdir(parents=True, exist_ok=True)

t0 = time.time()
dit = AceStepHandler()
msg, ok = dit.initialize_service(project_root=str(Path.cwd()), config_path="acestep-v15-turbo", device="mps")
if not ok:
    sys.exit(f"ACE-Step init failed: {msg}")
print(f"loaded in {time.time() - t0:.0f}s", flush=True)

for seed in spec["seeds"]:
    t1 = time.time()
    params = GenerationParams(caption=spec["caption"], lyrics=spec["structure"], instrumental=True,
                              duration=spec["duration"], bpm=spec["bpm"], keyscale=spec.get("key", ""),
                              timesignature="4", seed=seed, thinking=False, fade_out_duration=1.5)
    r = generate_music(dit, LLMHandler(), params,
                       GenerationConfig(batch_size=1, audio_format="wav", use_random_seed=False), save_dir=str(out))
    if not r.success:
        sys.exit(f"seed {seed} failed: {r.error}")
    Path(r.audios[0]["path"]).rename(out / f"take-s{seed}.wav")
    print(f"seed {seed} in {time.time() - t1:.0f}s", flush=True)
