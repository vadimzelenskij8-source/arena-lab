#!/usr/bin/env python3
"""Checks the Java scripted policy against the Python one, then measures the playout teacher's
winrate and speed vs the scripted opponent on the meta-deck evaluation (same seeds as eval)."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from crforge_env import LadderConfig, ScriptedBot, VectorBattle

from jbot import PlayoutTeacher, java_act, make_policy
ROOT = Path(__file__).resolve().parent
ROW = json.load(open(ROOT / "cards.json", encoding="utf-8"))
MY = {"cards": ["x-bow", "tesla", "archers", "knight", "skeletons", "electro-spirit", "fireball", "the-log"],
      "evo": ["tesla", "archers"], "hero": ["knight"]}
META = [d for d in json.load(open(ROOT / "decks.json", encoding="utf-8")) if d.get("src") == "meta"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--install", required=True)
    ap.add_argument("--tables", required=True)
    ap.add_argument("--out", default="out")
    ap.add_argument("--games", type=int, default=320)
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--play-ticks", type=int, default=160)
    ap.add_argument("--top", type=int, default=4)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--margin", type=float, default=0.3)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    N = a.envs
    res = {}
    with VectorBattle(N, a.threads, install=a.install, tables=a.tables, config=LadderConfig(decision_ticks=5)) as vec:
        s = vec.schema; I = s.info
        my_rows = [ROW[k] for k in MY["cards"]]
        my_slots = [(s.slot_evolution if k in MY["evo"] else 0) | (s.slot_hero if k in MY["hero"] else 0) for k in MY["cards"]]
        slots_of = lambda d: [(s.slot_evolution if k in d.get("evo", []) else 0) | (s.slot_hero if k in d.get("hero", []) else 0) for k in d["cards"]]
        bots = [ScriptedBot().bind(vec), ScriptedBot().bind(vec)]
        policy = make_policy(vec, bots[0])
        rng = np.random.default_rng(1)

        def reset(bi):
            ds = [META[(bi * N + i) % len(META)] for i in range(N)]
            seats = np.array([i % 2 for i in range(N)])
            d0 = [my_rows if seats[i] == 0 else [ROW[c] for c in ds[i]["cards"]] for i in range(N)]
            d1 = [[ROW[c] for c in ds[i]["cards"]] if seats[i] == 0 else my_rows for i in range(N)]
            s0 = [my_slots if seats[i] == 0 else slots_of(ds[i]) for i in range(N)]
            s1 = [slots_of(ds[i]) if seats[i] == 0 else my_slots for i in range(N)]
            vec.reset([((10 ** 6 + bi * 1009) * 1000 + i) % 2_000_000_000 for i in range(N)], d0, d1, slots0=s0, slots1=s1)
            return seats

        # 1) equivalence on states of scripted-vs-scripted games
        reset(0)
        same = tot = plays = 0
        t0 = time.time()
        for step in range(400):
            acts = np.zeros((N, 2), np.int32)
            for i in range(N):
                for seat in (0, 1):
                    o, m = vec.observations[i, seat], vec.masks[i, seat]
                    pa = bots[seat](o, m, rng)
                    if i < 16:
                        ja = java_act(vec, policy, o, m)
                        same += pa == ja; tot += 1; plays += pa != 0
                    acts[i, seat] = pa
            vec.step(acts)
        res["equivalence"] = {"agree": round(same / max(1, tot), 4), "checked": tot, "python_plays": plays}
        print("equivalence", res["equivalence"], round(time.time() - t0), "s", flush=True)

        # 2) the playout teacher's own strength and speed
        teacher = PlayoutTeacher(vec, policy, play_ticks=a.play_ticks, top=a.top, reps=a.reps, margin=a.margin,
                                 rng=np.random.default_rng(7))
        ev = {"wins": 0, "losses": 0, "draws": 0, "faults": 0}
        t0 = time.time(); t_teach = 0.0
        for bi in range(max(1, a.games // N)):
            seats = reset(bi)
            done = np.zeros(N, bool)
            for _ in range(1800):
                acts = np.zeros((N, 2), np.int32)
                prop = np.zeros(N, np.int64)
                for i in range(N):
                    me = seats[i]
                    prop[i] = bots[me](vec.observations[i, me], vec.masks[i, me], rng)
                    acts[i, 1 - me] = bots[1 - me](vec.observations[i, 1 - me], vec.masks[i, 1 - me], rng)
                t1 = time.time()
                tact, _ = teacher.refine2(prop, seats, ~done & (prop > 0), [[] for _ in range(N)])
                t_teach += time.time() - t1
                for i in range(N):
                    acts[i, seats[i]] = tact[i]
                vec.step(acts)
                for i in range(N):
                    if vec.dones[i] and not done[i]:
                        done[i] = True
                        w_ = int(vec.infos[i, I["winner"]])
                        k = "faults" if int(vec.infos[i, I["fault"]]) else ("wins" if w_ == seats[i] else ("losses" if w_ in (0, 1) else "draws"))
                        ev[k] += 1
                if done.all():
                    break
            g = ev["wins"] + ev["losses"] + ev["draws"]
            print("batch", bi, "winrate", round(ev["wins"] / max(1, g), 4), "teacher_s", round(t_teach), "total_s", round(time.time() - t0), teacher.stats, flush=True)
        g = ev["wins"] + ev["losses"] + ev["draws"]
        ev["games"] = g; ev["winrate"] = round(ev["wins"] / max(1, g), 4); ev["seconds"] = round(time.time() - t0)
        ev["teacher_seconds"] = round(t_teach); ev["stats"] = teacher.stats; ev["args"] = vars(a)
        res["teacher"] = ev
        json.dump(res, open(out / "jtest.json", "w"))
        print("jtest", json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
