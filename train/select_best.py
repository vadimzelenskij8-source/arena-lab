#!/usr/bin/env python3
"""Picks the best worker of a generation by its evaluation and updates base/ and history.

usage: select_best.py <workers_dir> <base_dir> <run_id>
"""
import json
import shutil
import sys
import time
from pathlib import Path

wd, base, run = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
base.mkdir(parents=True, exist_ok=True)
hist_p = base / "history.json"
hist = json.load(open(hist_p)) if hist_p.exists() else []
prev = hist[-1]["best"] if hist else None

rows = []
for d in sorted(wd.iterdir()):
    e, p = d / "eval.json", d / "policy.pt"
    if e.exists() and p.exists():
        ev = json.load(open(e))
        cfg = json.load(open(d / "metrics.json")).get("config", {}) if (d / "metrics.json").exists() else {}
        rows.append({"dir": d, "wr": ev.get("winrate") or 0.0, "games": ev.get("games", 0), "ev": ev,
                     "lr": cfg.get("lr"), "ent": cfg.get("ent"), "anchor": cfg.get("anchor"), "seed": ev.get("seed")})
if not rows:
    sys.exit("no finished workers")
rows.sort(key=lambda r: -r["wr"])
b = rows[0]
# прошлый лучший остаётся базой, если новый хуже его проверки (защита от отката)
keep_old = prev is not None and b["wr"] < prev - 0.01
if not keep_old:
    for f in ("policy.pt", "metrics.json", "eval.json"):
        if (b["dir"] / f).exists():
            shutil.copy(b["dir"] / f, base / f)
if not keep_old and b["lr"] is not None:
    json.dump({"lr": b["lr"], "ent": b["ent"], "anchor": b["anchor"]}, open(base / "hparams.json", "w"))
stag = (hist[-1].get("stagnation", 0) + 1) if (hist and (keep_old or b["wr"] <= (prev or 0) + 0.005)) else 0
hist.append({"stagnation": stag, "gen": len(hist) + 1, "run": run, "t": round(time.time()),
             "best": b["wr"] if not keep_old else prev, "gen_best": b["wr"],
             "mean": round(sum(r["wr"] for r in rows) / len(rows), 4),
             "workers": [{"wr": r["wr"], "games": r["games"], "lr": r["lr"], "ent": r["ent"], "anchor": r["anchor"], "seed": r["seed"]} for r in rows],
             "kept_previous": keep_old})
json.dump(hist, open(hist_p, "w"), indent=1)
print(json.dumps(hist[-1], indent=1))
