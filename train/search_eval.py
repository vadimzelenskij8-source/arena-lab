#!/usr/bin/env python3
"""Search teacher test: the scripted policy proposes a play, the simulator looks ahead over
alternatives (other tiles, other cards, wait) and the best by a material/tower/elixir value is
played. Measures the winrate vs the scripted opponent on meta decks (same seeds as ppo2 eval).
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
from crforge_env import LadderConfig, ScriptedBot, VectorBattle

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
    ap.add_argument("--games", type=int, default=640)
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--spread", type=int, default=2)
    ap.add_argument("--w-elixir", type=float, default=1.0)
    ap.add_argument("--margin", type=float, default=0.3)
    ap.add_argument("--off", action="store_true", help="no search (baseline)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    N = a.envs
    with VectorBattle(N, a.threads, install=a.install, tables=a.tables, config=LadderConfig(decision_ticks=5)) as vec:
        s = vec.schema; h = s.header; e = s.entity; I = s.info
        per = s.cols * s.rows
        tw = [h[f"{t}{k}"][0] for t in ("own_king", "own_left", "own_right", "enemy_king", "enemy_left", "enemy_right") for k in ("_hp", "_max_hp")]
        EL, CNT, H, EI = h["elixir"][0], h["entity_count"][0], s.header_ints, s.entity_ints
        my_rows = [ROW[k] for k in MY["cards"]]
        my_slots = [(s.slot_evolution if k in MY["evo"] else 0) | (s.slot_hero if k in MY["hero"] else 0) for k in MY["cards"]]
        slots_of = lambda d: [(s.slot_evolution if k in d.get("evo", []) else 0) | (s.slot_hero if k in d.get("hero", []) else 0) for k in d["cards"]]
        bots = [ScriptedBot().bind(vec), ScriptedBot().bind(vec)]
        rng = np.random.default_rng(12345)

        def value(O):
            """Material in elixir units from the seat's view, for a batch of observations."""
            t = O[:, tw].astype(np.float64).reshape(-1, 6, 2)
            towers = (t[:, :3, 0].sum(1) - t[:, 3:, 0].sum(1)) / 300.0
            rows = O[:, H:H + s.max_entities * EI].reshape(len(O), s.max_entities, EI)
            n = np.minimum(O[:, CNT], s.max_entities)
            valid = np.arange(s.max_entities)[None] < n[:, None]
            ch = valid & (rows[:, :, e["kind"]] == 1)
            hp, mh = rows[:, :, e["hp"]].astype(np.float64), rows[:, :, e["max_hp"]].astype(np.float64)
            frac = np.where(hp >= 0, hp / np.maximum(1, mh), 1.0)
            val = np.minimum(8.0, np.maximum(0, mh) / 350.0) * frac
            side = rows[:, :, e["side"]]
            units = (val * (ch & (side == 0))).sum(1) - (val * (ch & (side == 1))).sum(1)
            return towers + units + a.w_elixir * O[:, EL] / 10000.0

        def candidates(act, mask):
            sl, tile = divmod(act - 1, per)
            r, c = divmod(tile, s.cols)
            cand = {act, 0}
            d = a.spread
            for dr in (-d, 0, d):
                for dc in (-d, 0, d):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < s.rows and 0 <= cc < s.cols:
                        cand.add(1 + sl * per + rr * s.cols + cc)
            cand.add(1 + sl * per + r * s.cols + (s.cols - 1 - c))
            for o in range(4):
                cand.add(1 + o * per + tile)
            return [x for x in cand if mask[x]]

        res = {"wins": 0, "losses": 0, "draws": 0, "faults": 0, "changed": 0, "searched": 0, "by_deck": {}}
        t0 = time.time()
        for bi in range(max(1, a.games // N)):
            ds = [META[(bi * N + i) % len(META)] for i in range(N)]
            seats = np.array([i % 2 for i in range(N)])
            d0 = [my_rows if seats[i] == 0 else [ROW[c] for c in ds[i]["cards"]] for i in range(N)]
            d1 = [[ROW[c] for c in ds[i]["cards"]] if seats[i] == 0 else my_rows for i in range(N)]
            s0 = [my_slots if seats[i] == 0 else slots_of(ds[i]) for i in range(N)]
            s1 = [slots_of(ds[i]) if seats[i] == 0 else my_slots for i in range(N)]
            vec.reset([((10 ** 6 + bi * 1009) * 1000 + i) % 2_000_000_000 for i in range(N)], d0, d1, slots0=s0, slots1=s1)
            done = np.zeros(N, bool)
            for _ in range(1800):
                acts = np.zeros((N, 2), np.int32)
                req_i, req_a, owner = [], [], []
                for i in range(N):
                    me = seats[i]
                    acts[i, me] = bots[me](vec.observations[i, me], vec.masks[i, me], rng)
                    acts[i, 1 - me] = bots[1 - me](vec.observations[i, 1 - me], vec.masks[i, 1 - me], rng)
                    if not a.off and not done[i] and 0 < acts[i, me] < 1 + 4 * per:
                        cs = candidates(int(acts[i, me]), vec.masks[i, me])
                        owner.append((i, len(req_i), cs))
                        req_i += [i] * len(cs); req_a += cs
                if req_i:
                    obs, info = vec.lookahead(np.array(req_i), seats[np.array(req_i)], np.array(req_a), a.horizon)
                    v = value(obs)
                    v[info[:, I["fault"]] != 0] = -1e9
                    for i, st, cs in owner:
                        vv = v[st:st + len(cs)]
                        base = vv[cs.index(int(acts[i, seats[i]]))]
                        j = int(np.argmax(vv))
                        res["searched"] += 1
                        if vv[j] > base + a.margin:
                            acts[i, seats[i]] = cs[j]; res["changed"] += 1
                vec.step(acts)
                for i in range(N):
                    if vec.dones[i] and not done[i]:
                        done[i] = True
                        w_ = int(vec.infos[i, I["winner"]])
                        k = "faults" if int(vec.infos[i, I["fault"]]) else ("wins" if w_ == seats[i] else ("losses" if w_ in (0, 1) else "draws"))
                        res[k] += 1
                        bd = res["by_deck"].setdefault(ds[i].get("group", "?"), [0, 0]); bd[0] += k == "wins"; bd[1] += k != "faults"
                if done.all():
                    break
            g = res["wins"] + res["losses"] + res["draws"]
            print("batch", bi, "winrate", round(res["wins"] / max(1, g), 4), "changed", res["changed"], "/", res["searched"], round(time.time() - t0), "s", flush=True)
        g = res["wins"] + res["losses"] + res["draws"]
        res["games"] = g; res["winrate"] = round(res["wins"] / max(1, g), 4); res["seconds"] = round(time.time() - t0)
        res["args"] = vars(a)
        json.dump(res, open(out / "search.json", "w"), ensure_ascii=False)
        print("search", {k: res[k] for k in ("games", "winrate", "changed", "searched", "seconds")}, flush=True)


if __name__ == "__main__":
    main()
