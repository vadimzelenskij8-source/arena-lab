#!/usr/bin/env python3
"""PBT explore: prints worker args from base/hparams.json (best of last gen) with mutations.
usage: hparams.py <base_dir> <worker> <run_number>"""
import json
import random
import sys
from pathlib import Path

base, w, run = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
hp = {"lr": 1e-4, "ent": 0.003, "anchor": 0.02, "cap": 99.0}
if (base / "hparams.json").exists():
    hp.update(json.load(open(base / "hparams.json")))
hist = json.load(open(base / "history.json")) if (base / "history.json").exists() else []
stag = hist[-1].get("stagnation", 0) if hist else 0
rnd = random.Random(run * 1000 + w)
fac = [0.5, 0.7, 1.0, 1.4, 2.0] if stag < 2 else [0.25, 0.5, 2.0, 4.0]  # застой -> сильнее мутации
if w > 0:
    for k in ("lr", "ent", "anchor"):
        hp[k] *= rnd.choice(fac)
    if w % 4 != 1:  # исследование: часть воркеров обязана делать свои ходы
        hp["cap"] = rnd.choice([99.0, 6.0, 5.0, 4.0, 3.5, 3.0])
    if rnd.random() < 0.3:
        hp["anchor"] = rnd.choice([0.0, 0.002, 0.005])
hp["lr"] = min(1e-3, max(1e-5, hp["lr"])); hp["ent"] = min(0.05, max(1e-4, hp["ent"])); hp["anchor"] = min(0.2, max(0.0, hp["anchor"]))
print(f'--lr {hp["lr"]:.3g} --ent {hp["ent"]:.3g} --anchor {hp["anchor"]:.3g} --cap {hp["cap"]:.3g}')
