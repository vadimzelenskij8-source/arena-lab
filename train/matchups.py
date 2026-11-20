#!/usr/bin/env python3
"""Matchup table after a generation: usage matchups.py <base_dir> <ceilings_dir>

Per meta deck, for "any deck vs meta" and "X-Bow vs meta": the bot's exam result (base/eval.json, the
current main version), the teacher's result on the same games (the practical ceiling; refreshed when
ceiling parts are given), the history of the bot's winrate, and a status:
  solved  - the bot is within TOL of the teacher;
  paused  - not solved, but no growth over the last 3 generations (a ceiling we cannot pass now);
  active  - room to grow. Paused decks are re-activated every 3rd generation (re-check).
Training weights (any-deck mode) go to the largest gaps; solved and paused decks get little.
Writes base/matchups.json and base/matchup_weights.json.
"""
import json
import sys
from pathlib import Path

TOL = 0.10  # ~20 games per deck: about one standard error
base, cdir = Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv) > 2 else None
mf = base / "matchups.json"
M = json.load(open(mf)) if mf.exists() else {"gen": 0, "any": {}, "xbow": {}}
M["gen"] = M.get("gen", 0) + 1
ev = json.load(open(base / "eval.json")) if (base / "eval.json").exists() else {}

# teacher ceilings from this generation's ceiling jobs (summed over parts)
ceil = {"any": {}, "xbow": {}}
if cdir and cdir.exists():
    for f in cdir.rglob("ceilings.json"):
        c = json.load(open(f))
        for mode in ("any", "xbow"):
            for d, (w, g) in c.get(mode, {}).items():
                t = ceil[mode].setdefault(d, [0, 0]); t[0] += w; t[1] += g
bot = {"any": ev.get("by_deck", {}), "xbow": (ev.get("xbow") or {}).get("by_deck", {})}

for mode in ("any", "xbow"):
    tab = M.setdefault(mode, {})
    for d in set(bot[mode]) | set(ceil[mode]) | set(tab):
        r = tab.setdefault(d, {"hist": [], "status": "active"})
        if d in ceil[mode]:
            r["teacher"] = ceil[mode][d]
        if d in bot[mode]:
            r["bot"] = bot[mode][d]
            w, g = bot[mode][d]
            r["hist"] = (r["hist"] + [round(w / max(1, g), 3)])[-8:]
        bw = r["hist"][-1] if r["hist"] else 0.0
        tw = (r["teacher"][0] / max(1, r["teacher"][1])) if r.get("teacher") else None
        r["gap"] = round(tw - bw, 3) if tw is not None else None
        h = r["hist"]
        if tw is not None and bw >= tw - TOL:
            r["status"] = "solved"
        elif len(h) >= 4 and max(h[-3:]) <= h[-4] + 0.02:
            r["status"] = "paused"
        else:
            r["status"] = "active"
        if r["status"] == "paused" and M["gen"] % 3 == 0:
            r["status"] = "active"  # re-check paused matchups now and then
    M[f"solved_{mode}"] = sum(r["status"] == "solved" for r in tab.values())
    M[f"paused_{mode}"] = sum(r["status"] == "paused" for r in tab.values())
    M[f"decks_{mode}"] = len(tab)

W = {}
for d, r in M["any"].items():
    g = r.get("gap")
    if r["status"] == "solved":
        W[d] = 0.03
    elif r["status"] == "paused":
        W[d] = 0.02
    else:
        W[d] = min(1.0, max(0.05, g if g is not None else 0.3))
json.dump(M, open(mf, "w"), indent=1, ensure_ascii=False)
json.dump(W, open(base / "matchup_weights.json", "w"), ensure_ascii=False)
print("matchups gen", M["gen"], "solved any", M["solved_any"], "/", M["decks_any"], "xbow", M["solved_xbow"], "/", M["decks_xbow"],
      "paused", M["paused_any"], M["paused_xbow"])
