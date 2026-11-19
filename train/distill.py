#!/usr/bin/env python3
"""DAgger distillation of the search teacher into the recurrent entity network (generation 3).

Rollouts mix the teacher (scripted + lookahead) and the network's own moves (beta anneals);
every visited state is labelled with the teacher's move and the network is trained to predict it
(plus aux heads). The network plays alone at evaluation ("pure": no delegate, no simulator), which
is what can run on the phone. Writes out/policy.pt (best pure mini-eval), metrics.json, eval.json.
"""
import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from crforge_env import LadderConfig, ScriptedBot, VectorBattle

from ppo2 import META, MY, OTHER, ROW, Enc, Net, to_t
from teacher import Teacher


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--install", required=True)
    ap.add_argument("--tables", required=True)
    ap.add_argument("--out", default="out")
    ap.add_argument("--minutes", type=float, default=95)
    ap.add_argument("--eval-minutes", type=float, default=12)
    ap.add_argument("--eval-games", type=int, default=640)
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--resume", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--beta0", type=float, default=0.8)
    ap.add_argument("--beta1", type=float, default=0.3)
    ap.add_argument("--wait-w", type=float, default=0.15)
    ap.add_argument("--horizon", type=int, default=100)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--mini-eval-min", type=float, default=25)
    ap.add_argument("--T", type=int, default=64)
    # accepted for compatibility with hparams.py
    ap.add_argument("--ent", type=float, default=0)
    ap.add_argument("--anchor", type=float, default=0)
    ap.add_argument("--cap", type=float, default=99)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); t_end = t0 + a.minutes * 60; t_train_end = t_end - a.eval_minutes * 60
    torch.set_num_threads(2)
    rng = np.random.default_rng(a.seed); random.seed(a.seed); torch.manual_seed(a.seed)
    N, T = a.envs, a.T
    metrics = {"iters": [], "mini_evals": [], "config": vars(a), "timing": {}}
    with VectorBattle(N, a.threads, install=a.install, tables=a.tables, config=LadderConfig(decision_ticks=5)) as vec:
        s = vec.schema
        enc = Enc(vec)
        NA = s.actions; DEL = NA
        I = s.info
        my_rows = [ROW[k] for k in MY["cards"]]
        my_slots = [(s.slot_evolution if k in MY["evo"] else 0) | (s.slot_hero if k in MY["hero"] else 0) for k in MY["cards"]]
        slots_of = lambda d: [(s.slot_evolution if k in d.get("evo", []) else 0) | (s.slot_hero if k in d.get("hero", []) else 0) for k in d["cards"]]
        bots = [ScriptedBot().bind(vec), ScriptedBot().bind(vec)]
        teacher = Teacher(vec, horizon=a.horizon, margin=a.margin)

        def reset_all(seed, ds):
            seats = np.array([i % 2 for i in range(N)])
            for k in range(10):
                try:
                    d0 = [my_rows if seats[i] == 0 else [ROW[c] for c in ds[i]["cards"]] for i in range(N)]
                    d1 = [[ROW[c] for c in ds[i]["cards"]] if seats[i] == 0 else my_rows for i in range(N)]
                    s0 = [my_slots if seats[i] == 0 else slots_of(ds[i]) for i in range(N)]
                    s1 = [slots_of(ds[i]) if seats[i] == 0 else my_slots for i in range(N)]
                    vec.reset([((seed + k * 7919) * 1000 + i) % 2_000_000_000 for i in range(N)], d0, d1, slots0=s0, slots1=s1)
                    return seats
                except Exception as e:  # noqa: BLE001
                    print("reset", e, flush=True)
                    ds = [random.choice(META) for _ in range(N)]
            raise RuntimeError("no valid decks")

        def train_decks():
            return [random.choice(META) if random.random() < 0.8 else random.choice(OTHER) for _ in range(N)]

        net = Net(enc, NA + 1)
        if a.resume and Path(a.resume).exists():
            try:
                net.load_state_dict(torch.load(a.resume)); print("resumed", flush=True)
            except Exception as e:  # noqa: BLE001
                print("resume skipped", e, flush=True)
        opt = torch.optim.Adam(net.parameters(), lr=a.lr, eps=1e-5)

        def fresh():
            return dict(h=torch.zeros(N, 256), last_play=np.full((N, 8), -1, np.int64), prev_seen=np.full((N, 8), -1, np.int64))

        def batch_obs(seats, st):
            O = np.stack([vec.observations[i, seats[i]] for i in range(N)])
            M = np.stack([vec.masks[i, seats[i]].astype(bool) for i in range(N)])
            M = np.concatenate([M, np.zeros((N, 1), bool)], 1)  # pure: the delegate is never available
            seen = O[:, enc.seen]
            st["last_play"] = np.where(seen != st["prev_seen"], -1, st["last_play"]); st["prev_seen"] = seen
            tick = O[:, enc.tick][:, None]; pc, pt = O[:, enc.pc], O[:, enc.pt]
            match = (seen[:, :, None] == pc[:, None, :]) & (pc[:, None, :] >= 0)
            st["last_play"] = np.maximum(st["last_play"], np.where(match, (tick - pt)[:, None, :], -1).max(-1))
            en = enc.entities(O); g, ids = enc.header(O, st["last_play"])
            hst = np.stack([O[:, enc.hcost] / 10.0, O[:, enc.hform] / 2.0, O[:, enc.hpend]], -1).astype(np.float32)
            return O, M, dict(rid=en["rid"], st=en["st"], kind=en["kind"], side=en["side"], num=en["num"], valid=en["valid"],
                              tile=en["tile"], g=g, hstat=hst, **ids)

        def opp_targets(seats):
            Oo = np.stack([vec.observations[i, 1 - seats[i]] for i in range(N)])
            hand = np.zeros((N, enc.V), np.float32)
            hc = np.clip(Oo[:, enc.hand] + 1, 0, enc.V - 1)
            for j in range(4):
                hand[np.arange(N), hc[:, j]] = 1.0
            hand[:, 0] = 0
            return hand, Oo[:, enc.h["elixir"][0]].astype(np.float32) / 100000.0

        def run_eval(n_games, seed_base):
            ev = {"wins": 0, "losses": 0, "draws": 0, "faults": 0, "by_deck": {}, "by_group": {}, "by_phase": {}}
            erng = np.random.default_rng(12345)
            net.eval()
            for bi in range(max(1, n_games // N)):
                ds = [META[(bi * N + i) % len(META)] for i in range(N)]
                seats = reset_all(seed_base + bi * 1009, ds)
                st = fresh(); done = np.zeros(N, bool); ot = np.zeros(N, bool)
                for _ in range(1800):
                    O, M, b = batch_obs(seats, st)
                    with torch.no_grad():
                        bt = to_t(b); z, tf, hand = net.encode(bt); st["h"] = net.gru(z, st["h"])
                        lg, _ = net.heads(st["h"], tf, hand, bt, torch.from_numpy(M))
                        ac = lg.argmax(1).numpy()
                    acts = np.zeros((N, 2), np.int32)
                    for i in range(N):
                        me = seats[i]
                        acts[i, me] = ac[i]
                        acts[i, 1 - me] = bots[1 - me](vec.observations[i, 1 - me], vec.masks[i, 1 - me], erng)
                        ot[i] |= bool(O[i, enc.h["overtime"][0]])
                    vec.step(acts)
                    for i in range(N):
                        if vec.dones[i] and not done[i]:
                            done[i] = True
                            w_ = int(vec.infos[i, I["winner"]])
                            k = "faults" if int(vec.infos[i, I["fault"]]) else ("wins" if w_ == seats[i] else ("losses" if w_ in (0, 1) else "draws"))
                            ev[k] += 1
                            if k != "faults":
                                for key, name in (("by_deck", ds[i]["name"]), ("by_group", ds[i].get("group", "?")),
                                                  ("by_phase", "overtime" if ot[i] else "regular")):
                                    bd = ev[key].setdefault(name, [0, 0]); bd[0] += k == "wins"; bd[1] += 1
                    if done.all():
                        break
            g = ev["wins"] + ev["losses"] + ev["draws"]
            ev["games"] = g; ev["winrate"] = round(ev["wins"] / g, 4) if g else 0.0
            net.train()
            return ev

        seats = reset_all(1 + a.seed * 7, train_decks()); st = fresh()
        results = []
        best_sd, best_wr, last_mini, it = None, -1.0, time.time(), 0
        tm = defaultdict(float)
        while time.time() < t_train_end:
            frac = min(1.0, (time.time() - t0) / max(1, t_train_end - t0))
            beta = a.beta0 + (a.beta1 - a.beta0) * frac
            buf = defaultdict(list); h0 = st["h"].clone(); n_own = n_t = 0
            for t in range(T):
                t1 = time.time()
                O, M, b = batch_obs(seats, st)
                with torch.no_grad():
                    bt = to_t(b); z, tf, hand = net.encode(bt); hn = net.gru(z, st["h"])
                    lg, _ = net.heads(hn, tf, hand, bt, torch.from_numpy(M))
                    own = torch.distributions.Categorical(logits=lg).sample().numpy()
                tm["net"] += time.time() - t1; t1 = time.time()
                prop = np.zeros(N, np.int64); acts = np.zeros((N, 2), np.int32)
                for i in range(N):
                    me = seats[i]
                    prop[i] = bots[me](vec.observations[i, me], vec.masks[i, me], rng)
                    acts[i, 1 - me] = bots[1 - me](vec.observations[i, 1 - me], vec.masks[i, 1 - me], rng)
                tm["bots"] += time.time() - t1; t1 = time.time()
                tact = teacher.refine(prop, seats, np.ones(N, bool))
                tm["teacher"] += time.time() - t1; t1 = time.time()
                use_t = rng.random(N) < beta
                for i in range(N):
                    acts[i, seats[i]] = tact[i] if use_t[i] else own[i]
                n_own += int((~use_t).sum()); n_t += N
                ah, ae = opp_targets(seats)
                vec.step(acts)
                tm["sim"] += time.time() - t1
                d = np.zeros(N, np.float32)
                for i in range(N):
                    if vec.dones[i]:
                        d[i] = 1
                        if not int(vec.infos[i, I["fault"]]):
                            results.append((int(vec.infos[i, I["winner"]]) == seats[i]))
                        st["last_play"][i] = -1; st["prev_seen"][i] = -1
                st["h"] = hn * torch.from_numpy(1 - d).unsqueeze(1)
                for k, val in (("b", b), ("m", M), ("y", tact), ("d", d), ("ah", ah), ("ae", ae)):
                    buf[k].append(val)
            t1 = time.time()
            keys = buf["b"][0].keys()
            B = {k: np.stack([x[k] for x in buf["b"]]) for k in keys}
            Mm = np.stack(buf["m"]); Y = np.stack(buf["y"]); D = np.stack(buf["d"])
            AH, AE = np.stack(buf["ah"]), np.stack(buf["ae"])
            torch.set_num_threads(4)
            acc_p = []; acc_w = []
            for ep in range(2):
                perm = np.random.permutation(N)
                for j in range(0, N, 16):
                    ix = perm[j:j + 16]; nb = len(ix)
                    flat = {k: torch.from_numpy(np.ascontiguousarray(B[k][:, ix].reshape(T * nb, *B[k].shape[2:]))) for k in keys}
                    z, tf, hand = net.encode(flat); z = z.view(T, nb, -1)
                    h = h0[ix]; hs = []; Dm = torch.from_numpy(D[:, ix])
                    for t in range(T):
                        h = net.gru(z[t], h); hs.append(h); h = h * (1 - Dm[t]).unsqueeze(1)
                    H = torch.stack(hs).view(T * nb, -1)
                    mk = torch.from_numpy(Mm[:, ix].reshape(T * nb, -1))
                    y = torch.from_numpy(Y[:, ix].reshape(-1)).long()
                    mk[torch.arange(len(y)), y] = True  # the teacher's move is always legal for the seat
                    lg, _ = net.heads(H, tf, hand, flat, mk)
                    wts = torch.where(y == 0, a.wait_w, 1.0)
                    lbc = (F.cross_entropy(lg, y, reduction="none") * wts).sum() / wts.sum()
                    la = F.binary_cross_entropy_with_logits(net.aux_hand(H), torch.from_numpy(AH[:, ix].reshape(T * nb, -1)))
                    le = F.mse_loss(net.aux_el(H).squeeze(-1), torch.from_numpy(AE[:, ix].reshape(-1)))
                    loss = lbc + 0.1 * la + 0.1 * le
                    opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
                    with torch.no_grad():
                        pr = lg.argmax(1); pl = y != 0
                        if pl.any():
                            acc_p.append((pr[pl] == y[pl]).float().mean().item())
                        acc_w.append((pr[~pl] == 0).float().mean().item() if (~pl).any() else 1.0)
            torch.set_num_threads(2)
            tm["learn"] += time.time() - t1
            it += 1
            last = results[-300:]
            m = {"iter": it, "t": round(time.time()), "episodes": len(results), "beta": round(beta, 3),
                 "winrate_last200": round(float(np.mean(last)), 3) if last else None,
                 "bc_loss": round(lbc.item(), 4), "acc_play": round(float(np.mean(acc_p)), 3) if acc_p else None,
                 "acc_wait": round(float(np.mean(acc_w)), 3), "aux_hand": round(la.item(), 4),
                 "teacher_changed": round(teacher.stats["changed"] / max(1, teacher.stats["searched"]), 3)}
            metrics["iters"].append(m); metrics["timing"] = {k: round(v_, 1) for k, v_ in tm.items()}
            if it % 10 == 0:
                print(m, metrics["timing"], flush=True)
                json.dump(metrics, open(out / "metrics.json", "w"))
            if time.time() - last_mini > a.mini_eval_min * 60 and time.time() < t_train_end - 300:
                ev = run_eval(N * 2, 777)
                metrics["mini_evals"].append({"iter": it, "winrate": ev["winrate"]})
                print("mini_eval", it, ev["winrate"], flush=True)
                if ev["winrate"] > best_wr:
                    best_wr = ev["winrate"]; best_sd = {k: v_.clone() for k, v_ in net.state_dict().items()}
                last_mini = time.time()
                seats = reset_all(100 + it + a.seed * 7, train_decks()); st = fresh()
            elif it % 40 == 0:
                seats = reset_all(200 + it + a.seed * 7, train_decks()); st = fresh()
        ev_last = run_eval(N * 2, 777)
        metrics["mini_evals"].append({"iter": it, "winrate": ev_last["winrate"], "final": True})
        if best_sd is not None and best_wr > ev_last["winrate"]:
            net.load_state_dict(best_sd)
        torch.save(net.state_dict(), out / "policy.pt")
        json.dump(metrics, open(out / "metrics.json", "w"))
        ev = run_eval(a.eval_games, 10 ** 6); ev["seed"] = a.seed
        json.dump(ev, open(out / "eval.json", "w"), ensure_ascii=False)
        print("eval", {k: ev[k] for k in ("games", "winrate", "wins", "losses", "draws", "faults")}, flush=True)


if __name__ == "__main__":
    main()
