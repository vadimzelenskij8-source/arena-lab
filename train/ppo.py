#!/usr/bin/env python3
"""PPO agent for a card-battle simulator (residual over a scripted policy).

Worker of a population run: resumes --resume, trains until the time budget, then runs a fixed
evaluation (same seeds for every worker) and writes policy.pt, metrics.json, eval.json.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from crforge_env import LadderConfig, ScriptedBot, VectorBattle

ROOT = Path(__file__).resolve().parent
ROW = json.load(open(ROOT / "cards.json", encoding="utf-8"))
MY = {"cards": ["x-bow", "tesla", "archers", "knight", "skeletons", "electro-spirit", "fireball", "the-log"],
      "evo": ["tesla", "archers"], "hero": ["knight"]}
DECKS = [d for d in json.load(open(ROOT / "decks.json", encoding="utf-8"))]
META = [d for d in DECKS if d.get("src") == "meta"]
GX, GY = 9, 16  # сетка признаков (2x2 тайла); действия = полная сетка crforge
NA = None


class Feat:
    def __init__(self, vec):
        s = vec.schema
        self.s = s
        self.ids = {n: i for i, n in enumerate(vec.vocabulary)}
        self.deck_ids = [self.ids[ROW[k]] for k in MY["cards"]]
        h = s.header
        g = lambda n: h.get(n)
        self.hand, self.nxt, self.el, self.eel = g("hand_card"), g("next_card"), g("elixir"), g("enemy_elixir")
        self.tw, self.tms, self.form = g("towers"), g("time_ms"), g("hand_form")
        E = s.entity
        self.E = {k: E[k] for k in ("side", "kind", "x", "y", "hp", "max_hp")}
        self.hi, self.mi = s.header_ints, s.max_entities
        self.ei = s.entity_ints
        self.cnt = h["entity_count"]
        self.cnt = self.cnt[0] if isinstance(self.cnt, (list, tuple)) else self.cnt
        self.dim = 6 * GX * GY + 32 + 8 + 12 + 12 + 4
        self.seat_side = h.get("seat")

    def one(self, o):
        f = np.zeros(self.dim, np.float32)
        n = int(o[self.cnt])
        rows = o[self.hi:self.hi + n * self.ei].reshape(n, self.ei)
        E = self.E
        if n:
            my_side = int(o[self.seat_side[0] if isinstance(self.seat_side, (list, tuple)) else self.seat_side]) if self.seat_side is not None else 0
            mine = rows[:, E["side"]] == my_side
            kind = rows[:, E["kind"]]
            cx = np.clip(rows[:, E["x"]] // 2000, 0, GX - 1)
            cy = np.clip(rows[:, E["y"]] // 2000, 0, GY - 1)
            hp = rows[:, E["hp"]] / 1000.0
            ch = np.where(kind == 2, 4, 0) + np.where(mine, 0, 1)  # 0/1 юниты, 4/5 башни
            np.add.at(f, (ch * GX * GY + cy * GX + cx).astype(np.int64), hp.astype(np.float32))
        b = 6 * GX * GY
        for sl in range(4):
            cid = int(o[self.hand[sl]])
            if cid in self.deck_ids:
                f[b + sl * 8 + self.deck_ids.index(cid)] = 1
        b += 32
        cid = int(o[self.nxt[0] if isinstance(self.nxt, (list, tuple)) else self.nxt])
        if cid in self.deck_ids:
            f[b + self.deck_ids.index(cid)] = 1
        b += 8
        if self.tw is not None:
            t = np.asarray(o[self.tw], np.float32)[:12]
            f[b:b + 12] = t / 5000.0
        b += 12
        b += 12
        sc = lambda k: float(o[k[0] if isinstance(k, (list, tuple)) else k]) if k is not None else 0.0
        f[b] = sc(self.el) / 100000.0
        f[b + 1] = sc(self.eel) / 100000.0
        f[b + 2] = sc(self.tms) / 300000.0
        f[b + 3] = 1.0 if sc(self.tms) > 120000 else 0.0
        return f

    def towers(self, o):
        t = np.asarray(o[self.tw], np.float32)[:12] if self.tw is not None else np.zeros(12)
        return t[0:6:2].sum() / max(1, t[1:6:2].sum()), t[6:12:2].sum() / max(1, t[7:12:2].sum())


def coarse_mask(m, s):
    per = s.cols * s.rows
    out = np.zeros(NA, bool)
    out[0] = True
    for sl in range(4):
        g = m[1 + sl * per:1 + (sl + 1) * per].reshape(s.rows, s.cols)[:GY * 2, :GX * 2]
        c = g.reshape(GY, 2, GX, 2).any(axis=(1, 3))
        out[1 + sl * GX * GY:1 + (sl + 1) * GX * GY] = c.reshape(-1)
    return out


def to_fine(a, m, s):
    if a == 0:
        return 0
    a -= 1
    sl, r = divmod(a, GX * GY)
    cy, cx = divmod(r, GX)
    per = s.cols * s.rows
    g = m[1 + sl * per:1 + (sl + 1) * per].reshape(s.rows, s.cols)
    for dy, dx in ((1, 0), (1, 1), (0, 0), (0, 1)):  # ближе к реке сначала
        row, col = 2 * cy + dy, 2 * cx + dx
        if row < s.rows and col < s.cols and g[row, col]:
            return 1 + sl * per + row * s.cols + col
    return 0


def to_coarse(fine, s):
    if fine <= 0 or fine >= 1 + 4 * s.cols * s.rows:
        return 0
    a = fine - 1
    per = s.cols * s.rows
    sl, r = divmod(a, per)
    row, col = divmod(r, s.cols)
    return 1 + sl * GX * GY + min(GY - 1, row // 2) * GX + min(GX - 1, col // 2)


class Net(nn.Module):
    def __init__(self, d, na):
        super().__init__()
        self.b = nn.Sequential(nn.Linear(d, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU())
        self.pi = nn.Linear(256, na)
        self.v = nn.Linear(256, 1)

    def forward(self, x, mask):
        h = self.b(x)
        lg = self.pi(h).masked_fill(~mask, -1e9)
        return lg, self.v(h).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--install", required=True)
    ap.add_argument("--tables", required=True)
    ap.add_argument("--out", default="out/rl")
    ap.add_argument("--minutes", type=float, default=300)
    ap.add_argument("--envs", type=int, default=32)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--bc-steps", type=int, default=0)
    ap.add_argument("--delegate-bias", type=float, default=10.0)
    ap.add_argument("--resume", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ent", type=float, default=0.002)
    ap.add_argument("--decision-ticks", type=int, default=5)
    ap.add_argument("--eval-games", type=int, default=0)
    ap.add_argument("--eval-minutes", type=float, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t_end = time.time() + a.minutes * 60
    torch.set_num_threads(2)
    rng = np.random.default_rng(a.seed)
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    t_train_end = t_end - a.eval_minutes * 60
    N = a.envs
    metrics = {"iters": [], "bc": [], "config": vars(a)}
    if a.resume and Path(a.resume).exists() and Path(a.resume).with_name("metrics.json").exists():
        metrics = json.load(open(Path(a.resume).with_name("metrics.json")))
        metrics.setdefault("bc", [])

    cfg = LadderConfig(decision_ticks=a.decision_ticks)
    with VectorBattle(N, a.threads, install=a.install, tables=a.tables, config=cfg) as vec:
        s = vec.schema
        F = Feat(vec)
        my_rows = [ROW[k] for k in MY["cards"]]
        my_slots = [(s.slot_evolution if k in MY["evo"] else 0) | (s.slot_hero if k in MY["hero"] else 0) for k in MY["cards"]]
        I = {"tick": 0, "winner": 2, "fault": 5}

        def new_decks(seed, ds=None):
            ds = ds or random.sample(DECKS, N)
            d_rows = [[ROW[k] for k in d["cards"]] for d in ds]
            sl = [[(s.slot_evolution if k in d.get("evo", []) else 0) | (s.slot_hero if k in d.get("hero", []) else 0) for k in d["cards"]] for d in ds]
            seats = [i % 2 for i in range(N)]  # наше место
            d0 = [my_rows if seats[i] == 0 else d_rows[i] for i in range(N)]
            d1 = [d_rows[i] if seats[i] == 0 else my_rows for i in range(N)]
            s0 = [my_slots if seats[i] == 0 else sl[i] for i in range(N)]
            s1 = [sl[i] if seats[i] == 0 else my_slots for i in range(N)]
            vec.reset([seed * 1000 + i for i in range(N)], d0, d1, slots0=s0, slots1=s1)
            return seats

        def safe_decks(seed, ds=None):
            for k in range(10):
                try:
                    return new_decks(seed + k * 7919, ds)
                except Exception as e:  # noqa: BLE001
                    print("reset", e, flush=True)
            raise RuntimeError("no valid decks")

        seats = safe_decks(1 + a.seed * 100003)
        bots = [ScriptedBot().bind(vec), ScriptedBot().bind(vec)]
        NA_ = s.actions
        DEL = NA_  # доп. действие «сыграть как ScriptedBot» (residual: старт = уровень правил, RL только улучшает)
        net = Net(F.dim, NA_ + 1)
        with torch.no_grad():
            net.pi.bias[DEL] = a.delegate_bias
        opt = torch.optim.Adam(net.parameters(), lr=a.lr)
        if a.resume and Path(a.resume).exists():
            net.load_state_dict(torch.load(a.resume))
            print("resumed", a.resume, flush=True)

        def obs_batch(seat_of):
            X = np.stack([F.one(vec.observations[i, seat_of[i]]) for i in range(N)])
            M = np.stack([vec.masks[i, seat_of[i]].astype(bool) for i in range(N)])
            M = np.concatenate([M, np.ones((N, 1), bool)], 1)
            return X, M

        # ---- BC: копируем ScriptedBot нашей колодой ----
        if not a.resume and a.bc_steps > 0:
            bx, bm, by = [], [], []
            steps = 0
            while steps < a.bc_steps:
                X, M = obs_batch(seats)
                acts = np.zeros((N, 2), np.int32)
                for i in range(N):
                    for seat in (0, 1):
                        acts[i, seat] = bots[seat](vec.observations[i, seat], vec.masks[i, seat], rng)
                    c = int(acts[i, seats[i]])
                    if 0 <= c < NA_ and M[i, c] and (c != 0 or rng.random() < 0.08):
                        bx.append(X[i]); bm.append(M[i]); by.append(c)
                vec.step(acts)
                steps += 1
            bx, bm, by = torch.tensor(np.array(bx)), torch.tensor(np.array(bm)), torch.tensor(np.array(by))
            w = None
            nplay = int((by != 0).sum())
            for ep in range(15):
                perm = torch.randperm(len(bx))
                tot = 0.0
                for j in range(0, len(bx), 1024):
                    idx = perm[j:j + 1024]
                    lg, _ = net(bx[idx], bm[idx])
                    loss = nn.functional.cross_entropy(lg, by[idx], weight=w)
                    opt.zero_grad(); loss.backward(); opt.step()
                    tot += loss.item() * len(idx)
                with torch.no_grad():
                    pr = torch.cat([net(bx[j:j + 4096], bm[j:j + 4096])[0].argmax(1) for j in range(0, len(bx), 4096)])
                pl = by != 0
                metrics["bc"].append({"epoch": ep, "loss": tot / len(bx), "samples": len(bx), "plays": nplay,
                                      "acc_play": (pr[pl] == by[pl]).float().mean().item(),
                                      "acc_wait": (pr[~pl] == by[~pl]).float().mean().item(),
                                      "acc_card_slot": ((pr[pl] - 1) // (s.cols * s.rows) == (by[pl] - 1) // (s.cols * s.rows)).float().mean().item()})
                print("bc", metrics["bc"][-1], flush=True)
            seats = safe_decks(2 + a.seed * 100003)

        # ---- PPO ----
        T, gamma, lam, clip = 128, 0.997, 0.95, 0.2
        it = len(metrics["iters"])
        results = []
        prev_tw = np.array([F.towers(vec.observations[i, seats[i]]) for i in range(N)])
        while time.time() < t_train_end:
            buf = {k: [] for k in ("x", "m", "a", "lp", "v", "r", "d")}
            n_del = n_own_play = 0
            for t in range(T):
                X, M = obs_batch(seats)
                with torch.no_grad():
                    lg, v = net(torch.tensor(X), torch.tensor(M))
                    dist = torch.distributions.Categorical(logits=lg)
                    ac = dist.sample()
                    lp = dist.log_prob(ac)
                acts = np.zeros((N, 2), np.int32)
                for i in range(N):
                    me = seats[i]
                    sb = bots[me](vec.observations[i, me], vec.masks[i, me], rng)
                    acts[i, me] = sb if int(ac[i]) == DEL else int(ac[i])
                    n_del += int(ac[i]) == DEL; n_own_play += int(ac[i]) not in (0, DEL)
                    acts[i, 1 - me] = bots[1 - me](vec.observations[i, 1 - me], vec.masks[i, 1 - me], rng)
                vec.step(acts)
                r = np.zeros(N, np.float32)
                d = np.zeros(N, np.float32)
                for i in range(N):
                    me = seats[i]
                    if vec.dones[i]:
                        w_ = int(vec.infos[i, I["winner"]])
                        if not int(vec.infos[i, I["fault"]]):
                            r[i] = 1.0 if w_ == me else (-1.0 if w_ in (0, 1) else 0.0)
                            results.append(r[i])
                        d[i] = 1
                        prev_tw[i] = (1.0, 1.0)
                    else:
                        mt, et = F.towers(vec.observations[i, me])
                        r[i] += 0.1 * ((prev_tw[i][1] - et) - (prev_tw[i][0] - mt))
                        prev_tw[i] = (mt, et)
                for k, val in zip(("x", "m", "a", "lp", "v", "r", "d"), (X, M, ac.numpy(), lp.numpy(), v.numpy(), r, d)):
                    buf[k].append(val)
            X, M = obs_batch(seats)
            with torch.no_grad():
                _, last_v = net(torch.tensor(X), torch.tensor(M))
            V = np.array(buf["v"] + [last_v.numpy()])
            R, D = np.array(buf["r"]), np.array(buf["d"])
            adv = np.zeros_like(R)
            g = 0
            for t in reversed(range(T)):
                delta = R[t] + gamma * V[t + 1] * (1 - D[t]) - V[t]
                g = delta + gamma * lam * (1 - D[t]) * g
                adv[t] = g
            ret = adv + V[:-1]
            bx = torch.tensor(np.array(buf["x"]).reshape(T * N, -1))
            bm = torch.tensor(np.array(buf["m"]).reshape(T * N, -1))
            ba = torch.tensor(np.array(buf["a"]).reshape(-1))
            blp = torch.tensor(np.array(buf["lp"]).reshape(-1))
            badv = torch.tensor(adv.reshape(-1))
            badv = (badv - badv.mean()) / (badv.std() + 1e-8)
            bret = torch.tensor(ret.reshape(-1))
            for ep in range(4):
                perm = torch.randperm(T * N)
                for j in range(0, T * N, 1024):
                    idx = perm[j:j + 1024]
                    lg, v = net(bx[idx], bm[idx])
                    dist = torch.distributions.Categorical(logits=lg)
                    ratio = torch.exp(dist.log_prob(ba[idx]) - blp[idx])
                    l1 = -torch.min(ratio * badv[idx], torch.clamp(ratio, 1 - clip, 1 + clip) * badv[idx]).mean()
                    l2 = 0.5 * ((v - bret[idx]) ** 2).mean()
                    ent = dist.entropy().mean()
                    loss = l1 + l2 - a.ent * ent
                    opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 0.5); opt.step()
            it += 1
            last = results[-200:]
            m = {"iter": it, "t": round(time.time()), "episodes": len(results),
                 "winrate_last200": round(float(np.mean([x > 0 for x in last])), 3) if last else None,
                 "entropy": round(ent.item(), 3), "value_loss": round(l2.item(), 4),
                 "delegate": round(n_del / (T * N), 3), "own_plays_per_100": round(100 * n_own_play / (T * N), 2)}
            metrics["iters"].append(m)
            if it % 5 == 0:
                print(m, flush=True)
                torch.save(net.state_dict(), out / "policy.pt")
                json.dump(metrics, open(out / "metrics.json", "w"))
            if it % 40 == 0:  # новые колоды соперников
                seats = safe_decks(100 + it + a.seed * 100003)
                prev_tw = np.array([F.towers(vec.observations[i, seats[i]]) for i in range(N)])
        torch.save(net.state_dict(), out / "policy.pt")
        json.dump(metrics, open(out / "metrics.json", "w"))
        if a.eval_games > 0:  # одинаковая проверка у всех воркеров: мета-колоды, фиксированные сиды
            ev = {"wins": 0, "losses": 0, "draws": 0, "faults": 0, "by_deck": {}}
            gen = torch.Generator().manual_seed(12345)
            erng = np.random.default_rng(12345)
            for b in range(max(1, a.eval_games // N)):
                ds = [META[(b * N + i) % len(META)] for i in range(N)]
                seats = safe_decks(10 ** 6 + b * 1009, ds)
                done = np.zeros(N, bool)
                for _ in range(2000):
                    X, M = obs_batch(seats)
                    with torch.no_grad():
                        lg, _ = net(torch.tensor(X), torch.tensor(M))
                        ac = torch.multinomial(torch.softmax(lg, 1), 1, generator=gen).squeeze(1)
                    acts = np.zeros((N, 2), np.int32)
                    for i in range(N):
                        me = seats[i]
                        sb = bots[me](vec.observations[i, me], vec.masks[i, me], erng)
                        acts[i, me] = sb if int(ac[i]) == DEL else int(ac[i])
                        acts[i, 1 - me] = bots[1 - me](vec.observations[i, 1 - me], vec.masks[i, 1 - me], erng)
                    vec.step(acts)
                    for i in range(N):
                        if vec.dones[i] and not done[i]:
                            done[i] = True
                            w_ = int(vec.infos[i, I["winner"]])
                            k = "faults" if int(vec.infos[i, I["fault"]]) else ("wins" if w_ == seats[i] else ("losses" if w_ in (0, 1) else "draws"))
                            ev[k] += 1
                            bd = ev["by_deck"].setdefault(ds[i]["name"], [0, 0])
                            bd[0] += k == "wins"; bd[1] += k in ("wins", "losses", "draws")
                    if done.all():
                        break
            g = ev["wins"] + ev["losses"] + ev["draws"]
            ev["games"] = g
            ev["winrate"] = round(ev["wins"] / g, 4) if g else None
            ev["seed"] = a.seed
            json.dump(ev, open(out / "eval.json", "w"), ensure_ascii=False)
            print("eval", {k: ev[k] for k in ("games", "winrate", "wins", "losses", "draws", "faults")}, flush=True)
        print("done", metrics["iters"][-1] if metrics["iters"] else None, flush=True)


if __name__ == "__main__":
    main()
