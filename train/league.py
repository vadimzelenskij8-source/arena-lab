#!/usr/bin/env python3
"""League bookkeeping after a generation: usage league.py <workers_dir> <base_dir> <run_id>

- Elo of every worker from its exam vs the scripted bot (fixed at 1000) and its games vs league
  members (their Elo known): performance rating averaged over opponents.
- If select_best took a new best (history: kept_previous False), the best policy joins the
  league as gen<N>.pt with its Elo; the league keeps its 8 newest members.
Writes base/league.json and base/league/*.pt.
"""
import json
import math
import shutil
import sys
from pathlib import Path

wd, base, run = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
lg_dir = base / "league"; lg_dir.mkdir(parents=True, exist_ok=True)
lj = base / "league.json"
L = json.load(open(lj)) if lj.exists() else {"members": {}, "scripted": 1000.0, "workers": []}
hist = json.load(open(base / "history.json")) if (base / "history.json").exists() else []


def perf(p, r):
    p = min(0.97, max(0.03, p))
    return r + 400 * math.log10(p / (1 - p))


rows = []
for d in sorted(wd.iterdir()):
    e = d / "eval.json"
    if not e.exists():
        continue
    ev = json.load(open(e))
    parts = [(perf(ev.get("winrate") or 0.0, L["scripted"]), ev.get("games", 0))]
    for nm, (w, g) in (ev.get("league") or {}).items():
        if g >= 5 and nm != "self":
            L["members"].setdefault(nm, round(perf(0.47, L["scripted"]), 1))
            parts.append((perf(w / g, L["members"][nm]), g))
    tot = sum(g for _, g in parts)
    elo = sum(r * g for r, g in parts) / max(1, tot)
    rows.append({"worker": d.name, "elo": round(elo, 1), "winrate": ev.get("winrate"), "league": ev.get("league")})
rows.sort(key=lambda r: -r["elo"])
L["workers"] = rows
if hist and not hist[-1].get("kept_previous") and (base / "policy.pt").exists():
    name = f"gen{hist[-1]['gen']:03d}"
    shutil.copy(base / "policy.pt", lg_dir / f"{name}.pt")
    best_wr = hist[-1]["best"]
    L["members"][name] = round(perf(best_wr, L["scripted"]), 1)
for f in sorted(lg_dir.glob("*.pt"))[:-8]:
    L["members"].pop(f.stem, None); f.unlink()
json.dump(L, open(lj, "w"), indent=1, ensure_ascii=False)
print("league", L["members"], "top workers", rows[:3])
