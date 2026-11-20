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
from crforge_env import LadderConfig, ScriptedBot, SideSetup, VectorBattle

from ppo2 import META, MY, OTHER, ROW, Enc, Net, to_t
from teacher import ForkTeacher, Teacher
from jbot import PlayoutTeacher, make_policy, scripted_batch
import decks_gen


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
    ap.add_argument("--horizon", type=int, default=160)
    ap.add_argument("--play-thr", type=float, default=0.5)
    ap.add_argument("--fork", type=int, default=0)
    ap.add_argument("--xbow-share", type=float, default=0.2)
    ap.add_argument("--weak-share", type=float, default=0.15)
    ap.add_argument("--both", type=int, default=1)
    ap.add_argument("--replay-size", type=int, default=32)
    ap.add_argument("--carry", default="")
    ap.add_argument("--final", type=int, default=1)
    ap.add_argument("--p0", type=float, default=0.0)
    ap.add_argument("--p1", type=float, default=1.0)
    ap.add_argument("--league", default="league")
    ap.add_argument("--league-share", type=float, default=0.5)
    ap.add_argument("--vteach", type=float, default=0.0)
    ap.add_argument("--replay-chunks", type=int, default=8)
    ap.add_argument("--play-ticks", type=int, default=240)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--fork-steps", type=int, default=24)
    ap.add_argument("--fork-envs", type=int, default=96)
    ap.add_argument("--top", type=int, default=4)
    ap.add_argument("--soft-w", type=float, default=0.5)
    ap.add_argument("--val-w", type=float, default=0.5)
    ap.add_argument("--noise", type=float, default=0.15)
    ap.add_argument("--levels", type=int, default=1)
    ap.add_argument("--margin", type=float, default=0.3)
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
        jpol = make_policy(vec, bots[0])  # the scripted policy in Java: batched, inside the simulator's threads
        if a.fork == 2:  # Java playouts: both sides keep playing inside the simulator
            teacher = PlayoutTeacher(vec, make_policy(vec, bots[0]), horizon=a.horizon, play_ticks=a.play_ticks, top=a.top,
                                     reps=a.reps, margin=a.margin, rng=np.random.default_rng(a.seed + 99))
        elif a.fork:
            scratch = VectorBattle(a.fork_envs, a.threads, install=a.install, tables=a.tables, config=LadderConfig(decision_ticks=5))
            scratch.set_autoreset(True)  # finished forks restart on their own; results are read at the done step
            scratch.reset(list(range(a.fork_envs)), my_rows, my_rows, slots0=my_slots, slots1=my_slots)
            teacher = ForkTeacher(vec, scratch, [ScriptedBot().bind(scratch), ScriptedBot().bind(scratch)], horizon=a.horizon,
                                  fork_steps=a.fork_steps, top=a.top, margin=a.margin, rng=np.random.default_rng(a.seed + 99))
        else:
            teacher = Teacher(vec, horizon=a.horizon, margin=a.margin)
        TOWERS = ["King_PrincessTowers", "King_CannonTowers", "King_KnifeTowers", "King_ChefTowers"]

        MYD = {"name": "X-Bow 3.0", "group": "X-Bow", "cards": MY["cards"], "evo": MY["evo"], "hero": MY["hero"]}
        ANY = decks_gen.fixed_set(max(a.eval_games, 2 * N), meta=META)  # exam decks training never sees
        ANY_OPP = decks_gen.fixed_set(max(a.eval_games, 2 * N), seed=777, meta=META)  # opponents for the any-vs-any exam
        WEAK = [d for d in META if d.get("group") in ("Triple Elixir", "Mega Knight", "Mirror", "Seasonal Arena I", "Wall Breakers", "Giant")] or META
        grng = random.Random(a.seed * 7919 + 13)

        def reset_all(seed, ds, setups=False, mine=None):
            """ds: the opponents' decks; mine: the agent's decks (None: X-Bow)."""
            mine = mine or [MYD] * N
            seats = np.array([i % 2 for i in range(N)])
            for k in range(10):
                try:
                    st0 = st1 = None
                    if setups and k < 3:  # ladder-like opponents: tower troop and card levels vary
                        opp = [SideSetup(towers=random.choice(TOWERS), card_levels=[random.choice([11, 11, 12, 13]) for _ in range(8)]) for _ in range(N)]
                        st0 = [None if seats[i] == 0 else opp[i] for i in range(N)]
                        st1 = [opp[i] if seats[i] == 0 else None for i in range(N)]
                    rows_ = lambda d: [ROW[c] for c in d["cards"]]
                    d0 = [rows_(mine[i]) if seats[i] == 0 else rows_(ds[i]) for i in range(N)]
                    d1 = [rows_(ds[i]) if seats[i] == 0 else rows_(mine[i]) for i in range(N)]
                    s0 = [slots_of(mine[i]) if seats[i] == 0 else slots_of(ds[i]) for i in range(N)]
                    s1 = [slots_of(ds[i]) if seats[i] == 0 else slots_of(mine[i]) for i in range(N)]
                    vec.reset([((seed + k * 7919) * 1000 + i) % 2_000_000_000 for i in range(N)], d0, d1, slots0=s0, slots1=s1,
                              setup0=st0, setup1=st1)
                    return seats
                except Exception as e:  # noqa: BLE001
                    print("reset", e, flush=True)
                    ds = [random.choice(META) for _ in range(N)]
                    if k >= 2:
                        mine = [random.choice(META) for _ in range(N)]
            raise RuntimeError("no valid decks")

        def train_decks():  # opponents: endless generated decks, real meta decks, and extra weak matchups
            return [grng.choice(WEAK) if grng.random() < a.weak_share else decks_gen.generate(grng, META, p_meta=0.35) for _ in range(N)]

        def train_mine():  # any deck for the agent: a new one every battle, X-Bow now and then
            return [MYD if grng.random() < a.xbow_share else decks_gen.generate(grng, META) for _ in range(N)]

        net = Net(enc, NA + 1)
        if a.resume and Path(a.resume).exists():
            try:
                miss = net.load_state_dict(torch.load(a.resume), strict=False); print("resumed", miss, flush=True)
            except Exception as e:  # noqa: BLE001
                print("resume skipped", e, flush=True)
        opt = torch.optim.Adam(net.parameters(), lr=a.lr, eps=1e-5)
        # league: frozen earlier best versions play the opponent seat (prioritised by how often they beat us)
        league = []
        for f_ in sorted(Path(a.league).glob("*.pt")) if a.league else []:
            try:
                m_ = Net(enc, NA + 1); m_.load_state_dict(torch.load(f_), strict=False); m_.eval()
                league.append((f_.stem, m_))
            except Exception as e:  # noqa: BLE001
                print("league skip", f_, e, flush=True)
        lstats = {nm: [0, 0] for nm, _ in league}
        print("league", [nm for nm, _ in league], flush=True)
        opp_kind = np.full(N, -1)

        def assign_opps():
            opp_kind[:] = -1
            if not league:
                return
            w_ = np.array([(1 - (lstats[nm][0] + 1) / (lstats[nm][1] + 2)) ** 2 + 0.05 for nm, _ in league]); w_ /= w_.sum()
            for i in range(N):
                if rng.random() < a.league_share:
                    opp_kind[i] = int(rng.choice(len(league), p=w_))

        def league_acts(oseats, st_o):
            """Actions of the league opponents (pure network play, as at evaluation)."""
            out_ = {}
            if not league or (opp_kind < 0).all():
                return out_
            O_o, M_o, b_o = batch_obs(oseats, st_o)
            for mi, (_nm, m_) in enumerate(league):
                idx = np.flatnonzero(opp_kind == mi)
                if not len(idx):
                    continue
                sub = {k: v[idx] for k, v in b_o.items()}
                with torch.no_grad():
                    bt = to_t(sub); z, tf, hd = m_.encode(bt); hn_ = m_.gru(z, st_o["h"][idx])
                    lg, _ = m_.heads(hn_, tf, hd, bt, torch.from_numpy(M_o[idx]))
                    pr = torch.softmax(lg, 1); play = pr[:, 1:].clone(); play[:, -1] = 0
                    ac = torch.where(play.sum(1) > a.play_thr, play.argmax(1) + 1, torch.zeros_like(pr[:, 0]).long()).numpy()
                st_o["h"][idx] = hn_
                for j, i in enumerate(idx):
                    out_[int(i)] = int(ac[j])
            return out_

        def vfn(O):
            """The network's value of positions reached by the teacher's playouts (no history: fresh memory)."""
            en = enc.entities(O); g, ids = enc.header(O, np.full((len(O), 8), -1, np.int64))
            hst = np.stack([O[:, enc.hcost] / 10.0, O[:, enc.hform] / 2.0, O[:, enc.hpend]], -1).astype(np.float32)
            dk, dst = enc.deck_feats(O)
            b = dict(rid=en["rid"], st=en["st"], kind=en["kind"], side=en["side"], num=en["num"], valid=en["valid"],
                     tile=en["tile"], g=g, hstat=hst, deck=dk, dstat=dst, **ids)
            with torch.no_grad():
                z, _tf, _hd = net.encode(to_t(b)); h = net.gru(z, torch.zeros(len(O), 256))
                return net.v(h).squeeze(-1).numpy()
        if a.vteach > 0 and a.fork == 2:
            teacher.value_fn, teacher.vw = vfn, a.vteach

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
            dk, dst = enc.deck_feats(O)
            return O, M, dict(rid=en["rid"], st=en["st"], kind=en["kind"], side=en["side"], num=en["num"], valid=en["valid"],
                              tile=en["tile"], g=g, hstat=hst, deck=dk, dstat=dst, **ids)

        def opp_targets(seats):
            Oo = np.stack([vec.observations[i, 1 - seats[i]] for i in range(N)])
            hand = np.zeros((N, enc.V), np.float32)
            hc = np.clip(Oo[:, enc.hand] + 1, 0, enc.V - 1)
            for j in range(4):
                hand[np.arange(N), hc[:, j]] = 1.0
            hand[:, 0] = 0
            return hand, Oo[:, enc.h["elixir"][0]].astype(np.float32) / 100000.0

        def run_eval(n_games, seed_base, any_deck=True, opp_any=False):
            ev = {"wins": 0, "losses": 0, "draws": 0, "faults": 0, "by_deck": {}, "by_group": {}, "by_phase": {}}
            erng = np.random.default_rng(12345)
            net.eval()
            for bi in range(max(1, n_games // N)):
                ds = [(ANY_OPP if opp_any else META)[(bi * N + i) % len(ANY_OPP if opp_any else META)] for i in range(N)]
                mine = [ANY[(bi * N + i) % len(ANY)] for i in range(N)] if any_deck else None
                seats = reset_all(seed_base + bi * 1009, ds, mine=mine)
                st = fresh(); done = np.zeros(N, bool); ot = np.zeros(N, bool)
                for _ in range(1800):
                    O, M, b = batch_obs(seats, st)
                    with torch.no_grad():
                        bt = to_t(b); z, tf, hand = net.encode(bt); st["h"] = net.gru(z, st["h"])
                        lg, _ = net.heads(st["h"], tf, hand, bt, torch.from_numpy(M))
                        pr = torch.softmax(lg, 1)
                        play = pr[:, 1:].clone(); play[:, -1] = 0  # never the delegate
                        ac = torch.where(play.sum(1) > a.play_thr, play.argmax(1) + 1, torch.zeros_like(pr[:, 0]).long()).numpy()
                    acts = np.zeros((N, 2), np.int32)
                    oppa = scripted_batch(vec, jpol, 1 - seats, seed_base + bi * 31 + _)
                    for i in range(N):
                        me = seats[i]
                        acts[i, me] = ac[i]
                        acts[i, 1 - me] = oppa[i]
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

        seats = reset_all(1 + a.seed * 7, train_decks(), a.levels, train_mine()); st = fresh(); st_o = fresh(); assign_opps()
        results = []
        best_sd, best_wr, last_mini, it = None, -1.0, time.time(), 0
        cdir = Path(a.carry) if a.carry else None  # a long generation runs in segments: carry on where the last stopped
        if cdir and (cdir / "metrics.json").exists():
            old = json.load(open(cdir / "metrics.json"))
            metrics["iters"] = old.get("iters", []); metrics["mini_evals"] = old.get("mini_evals", [])
            it = metrics["iters"][-1]["iter"] if metrics["iters"] else 0
            if (cdir / "best.pt").exists() and (cdir / "best.json").exists():
                best_sd = torch.load(cdir / "best.pt"); best_wr = json.load(open(cdir / "best.json"))["winrate"]
            print("carried", it, best_wr, flush=True)
        replay = []
        tm = defaultdict(float)
        st_b = fresh()  # the network's own memory when it plays the opponent's seat (labels for both sides)
        jseed = a.seed * 7919

        def soft_arrays(soft):
            SK = np.full((N, 12), -1, np.int64); SP = np.zeros((N, 12), np.float32)
            for i in range(N):
                if soft[i] is not None:
                    ks, ps = soft[i]
                    ks, ps = list(ks)[:12], np.asarray(ps)[:12]
                    SK[i, :len(ks)] = ks; SP[i, :len(ks)] = ps
            return SK, SP

        def make_pack(bf, h0_, st_, side):
            keys_ = bf["b"][0].keys()
            Bk = {k: np.stack([x[k] for x in bf["b"]]) for k in keys_}
            Bk["num"] = Bk["num"].astype(np.float16)  # half the memory: the archive keeps many batches
            D_ = np.stack(bf["d"])
            O_, M_, b_ = batch_obs(side, st_)
            with torch.no_grad():
                bt_ = to_t(b_); z_, tf_, hd_ = net.encode(bt_); _, lv = net.heads(net.gru(z_, st_["h"]), tf_, hd_, bt_, torch.from_numpy(M_))
            Vv = np.concatenate([np.stack(bf["v"]), lv.numpy()[None]]); Rr = np.stack(bf["r"])
            RET_ = np.zeros((T, N), np.float32); g_ = Vv[-1]
            for t in reversed(range(T)):  # TD(lambda) value targets from game results
                g_ = Rr[t] + 0.997 * (1 - D_[t]) * (0.9 * g_ + 0.1 * Vv[t + 1])
                RET_[t] = g_
            return dict(B=Bk, Mm=np.stack(bf["m"]), Y=np.stack(bf["y"]), D=D_, AH=np.stack(bf["ah"]), AE=np.stack(bf["ae"]),
                        SKa=np.stack(bf["sk"]), SPa=np.stack(bf["sp"]), RET=RET_, h0=h0_)

        while time.time() < t_train_end:
            frac = a.p0 + (a.p1 - a.p0) * min(1.0, (time.time() - t0) / max(1, t_train_end - t0))
            beta = a.beta0 + (a.beta1 - a.beta0) * frac
            buf = defaultdict(list); bufB = defaultdict(list); h0 = st["h"].clone(); h0B = st_b["h"].clone(); n_own = n_t = 0
            for t in range(T):
                t1 = time.time()
                oseat = 1 - seats
                O, M, b = batch_obs(seats, st)
                Ob, Mb, bb = batch_obs(oseat, st_b)
                with torch.no_grad():
                    bt = to_t(b); z, tf, hand = net.encode(bt); hn = net.gru(z, st["h"])
                    lg, vpred = net.heads(hn, tf, hand, bt, torch.from_numpy(M))
                    own = torch.distributions.Categorical(logits=lg).sample().numpy()
                    pl = lg[:, 1:1 + 4 * enc.s.cols * enc.s.rows]
                    gk = (pl - torch.log(-torch.log(torch.rand_like(pl).clamp(1e-9, 1 - 1e-9)))).topk(a.top, 1).indices.numpy() + 1
                    okp = (pl.gather(1, torch.from_numpy(gk - 1)) > -1e8).numpy()
                    btb = to_t(bb); zb, tfb, hdb = net.encode(btb); hnb = net.gru(zb, st_b["h"])
                    _, vpb = net.heads(hnb, tfb, hdb, btb, torch.from_numpy(Mb))
                tm["net"] += time.time() - t1; t1 = time.time()
                acts = np.zeros((N, 2), np.int32)
                jseed += 1
                prop = scripted_batch(vec, jpol, seats, jseed)
                propB = scripted_batch(vec, jpol, oseat, jseed + 7777)
                la_ = league_acts(oseat, st_o)
                tm["bots"] += time.time() - t1; t1 = time.time()
                if a.fork:
                    extra = [[int(x) for x, ok in zip(gk[i], okp[i]) if ok] for i in range(N)]
                    active = np.array([(0 < prop[i] < 2305) or (0 < own[i] < 2305) for i in range(N)])
                    tact, soft = teacher.refine2(prop, seats, active, extra)
                else:
                    tact = teacher.refine(prop, seats, np.ones(N, bool)); soft = teacher.soft
                SK, SP = soft_arrays(soft)
                if a.both:  # the teacher also plays/labels the opponent's seat: twice the labels, a stronger opponent
                    tactB = teacher.refine(propB, oseat, np.ones(N, bool)); SKB, SPB = soft_arrays(teacher.soft)
                else:
                    tactB = propB; SKB, SPB = soft_arrays([None] * N)
                tm["teacher"] += time.time() - t1; t1 = time.time()
                use_t = rng.random(N) < beta
                for i in range(N):
                    acts[i, oseat[i]] = la_[i] if i in la_ else tactB[i]
                    if use_t[i]:
                        acts[i, seats[i]] = tact[i]
                    else:  # the network's own move, with phone-like reaction delay and tap error
                        x = int(own[i]); mk_ = vec.masks[i, seats[i]]
                        if 0 < x < 2305 and rng.random() < a.noise:
                            x = 0
                        elif 0 < x < 2305 and rng.random() < a.noise:
                            sl, tl = divmod(x - 1, 576); r_, c_ = divmod(tl, 18)
                            r2, c2 = min(31, max(0, r_ + int(rng.integers(-1, 2)))), min(17, max(0, c_ + int(rng.integers(-1, 2))))
                            y = 1 + sl * 576 + r2 * 18 + c2
                            x = y if mk_[y] else x
                        acts[i, seats[i]] = x
                n_own += int((~use_t).sum()); n_t += N
                ah, ae = opp_targets(seats)
                ahB, aeB = opp_targets(oseat)
                vec.step(acts)
                tm["sim"] += time.time() - t1
                d = np.zeros(N, np.float32); rr = np.zeros(N, np.float32)
                for i in range(N):
                    if vec.dones[i]:
                        d[i] = 1
                        w_ = int(vec.infos[i, I["winner"]])
                        if not int(vec.infos[i, I["fault"]]):
                            rr[i] = 1.0 if w_ == seats[i] else (-1.0 if w_ in (0, 1) else 0.0)
                            results.append((w_ == seats[i]))
                            if opp_kind[i] >= 0:
                                ls_ = lstats[league[opp_kind[i]][0]]; ls_[0] += int(results[-1]); ls_[1] += 1
                        for s_ in (st_o, st, st_b):
                            s_["last_play"][i] = -1; s_["prev_seen"][i] = -1
                        st_o["h"][i] = 0
                keep = torch.from_numpy(1 - d).unsqueeze(1)
                st["h"] = hn * keep; st_b["h"] = hnb * keep
                for k, val in (("b", b), ("m", M), ("y", tact), ("d", d), ("ah", ah), ("ae", ae), ("sk", SK), ("sp", SP),
                               ("v", vpred.numpy()), ("r", rr)):
                    buf[k].append(val)
                MbL = Mb.copy()
                for k, val in (("b", bb), ("m", MbL), ("y", tactB), ("d", d), ("ah", ahB), ("ae", aeB), ("sk", SKB), ("sp", SPB),
                               ("v", vpb.numpy()), ("r", -rr)):
                    bufB[k].append(val)
            t1 = time.time()
            keys = buf["b"][0].keys()
            packs = [make_pack(buf, h0, st, seats)]
            if a.both:
                packs.append(make_pack(bufB, h0B, st_b, 1 - seats))
            torch.set_num_threads(4)
            acc_p = []; acc_w = []

            def step_chunk(pk, ix):
                nb = len(ix)
                Bk = pk["B"]
                flat = {k: torch.from_numpy(np.ascontiguousarray(Bk[k][:, ix].reshape(T * nb, *Bk[k].shape[2:]))) for k in keys}
                flat["num"] = flat["num"].float()
                z, tf, hand = net.encode(flat); z = z.view(T, nb, -1)
                h = pk["h0"][ix]; hs = []; Dm = torch.from_numpy(pk["D"][:, ix])
                for t in range(T):
                    h = net.gru(z[t], h); hs.append(h); h = h * (1 - Dm[t]).unsqueeze(1)
                H = torch.stack(hs).view(T * nb, -1)
                mk = torch.from_numpy(pk["Mm"][:, ix].reshape(T * nb, -1))
                y = torch.from_numpy(pk["Y"][:, ix].reshape(-1)).long()
                mk[torch.arange(len(y)), y] = True  # the teacher's move is always legal for the seat
                sk = torch.from_numpy(pk["SKa"][:, ix].reshape(T * nb, -1)); sp = torch.from_numpy(pk["SPa"][:, ix].reshape(T * nb, -1))
                rows_ = torch.arange(len(y)).unsqueeze(1).expand_as(sk)
                mk[rows_[sk >= 0], sk[sk >= 0]] = True
                lg, vv_ = net.heads(H, tf, hand, flat, mk)
                wts = torch.where(y == 0, a.wait_w, 1.0)
                lbc = (F.cross_entropy(lg, y, reduction="none") * wts).sum() / wts.sum()
                la = F.binary_cross_entropy_with_logits(net.aux_hand(H), torch.from_numpy(pk["AH"][:, ix].reshape(T * nb, -1)))
                le = F.mse_loss(net.aux_el(H).squeeze(-1), torch.from_numpy(pk["AE"][:, ix].reshape(-1)))
                logp = F.log_softmax(lg, -1)
                has = sk[:, 0] >= 0
                lsoft = -(sp * logp.gather(1, sk.clamp(min=0)) * (sk >= 0)).sum(1)
                lsoft = lsoft[has].mean() if has.any() else torch.zeros(())
                lval = F.mse_loss(vv_, torch.from_numpy(pk["RET"][:, ix].reshape(-1)))
                loss = lbc + a.soft_w * lsoft + a.val_w * lval + 0.1 * la + 0.1 * le
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
                with torch.no_grad():
                    pr = lg.argmax(1); pl = y != 0
                    if pl.any():
                        acc_p.append((pr[pl] == y[pl]).float().mean().item())
                    acc_w.append((pr[~pl] == 0).float().mean().item() if (~pl).any() else 1.0)
                return lbc, lsoft, lval, la

            for pack in packs:
                for ep in range(2):
                    perm = np.random.permutation(N)
                    for j in range(0, N, 16):
                        lbc, lsoft, lval, la = step_chunk(pack, perm[j:j + 16])
            # replay: the teacher's labels are expensive, so recent batches are trained on again
            for _ in range(a.replay_chunks if replay else 0):
                old = replay[rng.integers(len(replay))]
                step_chunk(old, rng.permutation(N)[:16])
            replay.extend(packs); del replay[:-a.replay_size]
            torch.set_num_threads(2)
            tm["learn"] += time.time() - t1
            it += 1
            last = results[-300:]
            m = {"iter": it, "t": round(time.time()), "episodes": len(results), "beta": round(beta, 3),
                 "winrate_last200": round(float(np.mean(last)), 3) if last else None,
                 "bc_loss": round(lbc.item(), 4), "acc_play": round(float(np.mean(acc_p)), 3) if acc_p else None,
                 "acc_wait": round(float(np.mean(acc_w)), 3), "aux_hand": round(la.item(), 4),
                 "teacher_changed": round(teacher.stats["changed"] / max(1, teacher.stats["searched"]), 3),
                 "soft": round(float(lsoft), 4), "value_loss": round(float(lval), 4),
                 "forks": teacher.stats.get("forks", 0), "delayed": teacher.stats.get("delayed", 0),
                 "league": {k: v[:] for k, v in lstats.items()}}
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
                seats = reset_all(100 + it + a.seed * 7, train_decks(), a.levels, train_mine()); st = fresh(); st_o = fresh(); assign_opps()
            elif it % 40 == 0:
                seats = reset_all(200 + it + a.seed * 7, train_decks(), a.levels, train_mine()); st = fresh(); st_o = fresh(); assign_opps()
        if not a.final:  # middle segment: keep the run going, no exam
            torch.save(net.state_dict(), out / "policy.pt")
            if best_sd is not None:
                torch.save(best_sd, out / "best.pt"); json.dump({"winrate": best_wr}, open(out / "best.json", "w"))
            json.dump(metrics, open(out / "metrics.json", "w"))
            print("segment saved", it, best_wr, flush=True)
            return
        ev_last = run_eval(N * 2, 777)
        metrics["mini_evals"].append({"iter": it, "winrate": ev_last["winrate"], "final": True})
        if best_sd is not None and best_wr > ev_last["winrate"]:
            net.load_state_dict(best_sd)
        torch.save(net.state_dict(), out / "policy.pt")
        json.dump(metrics, open(out / "metrics.json", "w"))
        ev = run_eval(a.eval_games, 10 ** 6)  # any deck vs meta decks (selection metric)
        evx = run_eval(a.eval_games // 2, 10 ** 6, any_deck=False)  # X-Bow vs meta, comparable with earlier generations
        eva = run_eval(a.eval_games // 2, 2 * 10 ** 6, opp_any=True)  # any deck vs any deck: both unseen
        ev["xbow"] = {k: evx[k] for k in ("games", "winrate", "wins", "losses", "by_group")}
        ev["any_vs_any"] = {k: eva[k] for k in ("games", "winrate", "wins", "losses", "by_phase")}
        ev["seed"] = a.seed
        ev["league"] = lstats
        json.dump(ev, open(out / "eval.json", "w"), ensure_ascii=False)
        print("eval", {k: ev[k] for k in ("games", "winrate", "wins", "losses", "draws", "faults")}, "xbow", evx["winrate"],
              "any_vs_any", eva["winrate"], flush=True)


if __name__ == "__main__":
    main()
