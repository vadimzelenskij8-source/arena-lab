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
from jbot import PlayoutTeacher, ReplyTeacher, make_policy, scripted_batch
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
    ap.add_argument("--az", type=int, default=0, help="AlphaZero loop: network candidates + value inside the teacher")
    ap.add_argument("--az-vw", type=float, default=0.5)
    ap.add_argument("--strong", type=int, default=0, help="strong opponents: more league, self-play vs the current net")
    ap.add_argument("--matchups", default="", help="json: meta deck name -> training weight (gap to the teacher)")
    ap.add_argument("--meta-share", type=float, default=0.35)
    ap.add_argument("--xbow-games", type=int, default=1060)
    ap.add_argument("--ava-games", type=int, default=640)
    ap.add_argument("--ceilings", default="", help="k/K: only measure the teacher on part k of K of the exams, then exit")
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
    ap.add_argument("--anchor", type=float, default=0, help="KL(anchor policy || net): keeps the human-imitation style while self-play improves it")
    ap.add_argument("--anchor-pt", default="", help="anchor policy (default: the --resume start)")
    ap.add_argument("--cap", type=float, default=99)
    # next generation: D (rewards), exploiters, API decks, imitation, search exam (B)
    ap.add_argument("--pg", type=float, default=0.0, help="D: policy-gradient weight on the network's own moves (BC = anchor)")
    ap.add_argument("--exploit", type=int, default=0, help="exploiter: trains only against the frozen champion")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--api-decks", default="", help="real decks from the official API (json list)")
    ap.add_argument("--api-share", type=float, default=0.35)
    ap.add_argument("--search-games", type=int, default=0, help="B: exam of network + search in battle")
    ap.add_argument("--champ-games", type=int, default=0, help="exam vs the frozen champion (exploiters)")
    ap.add_argument("--ceil-seed", type=int, default=10 ** 6)
    # main loop: the network + search teaches the network (expert iteration)
    ap.add_argument("--selfsearch", type=int, default=0, help="search refines the network's own move; its choice is the label")
    ap.add_argument("--p-old0", type=float, default=0.5, help="soft start: share of moves where the search starts from the old scripted proposal")
    ap.add_argument("--p-old1", type=float, default=0.1)
    ap.add_argument("--sw", type=float, default=0.0, help="weight of the network's position value inside the search")
    ap.add_argument("--v0-w", type=float, default=0.5, help="train the value from a fresh memory too: what the search sees")
    ap.add_argument("--noisy-games", type=int, default=0, help="search exam with phone-like latency and tap error")
    ap.add_argument("--mb", type=int, default=16, help="sequences per training step (bigger = faster on CPU)")
    ap.add_argument("--search-only", type=float, default=0, help="minutes: only measure network+search (normal and phone) on the champion")
    ap.add_argument("--spread", type=int, default=2)
    ap.add_argument("--h1", type=int, default=0, help="two-stage search: screening horizon in ticks (0 = off)")
    ap.add_argument("--k2", type=int, default=3, help="two-stage search: candidates looked at in full")
    ap.add_argument("--skip-conf", type=float, default=0.0, help="no search when the network's move has this probability")
    ap.add_argument("--export-phone", default="", help="folder: check the torch encoder against numpy, export policy.onnx")
    ap.add_argument("--reply", type=int, default=0, help="search with the enemy's answer (our network from its side)")
    ap.add_argument("--reply-ticks", type=int, default=10, help="the enemy answers this many ticks after our move")
    ap.add_argument("--search-seed", type=int, default=0, help="offset of the search-exam games (parallel jobs of one setting)")
    ap.add_argument("--card-w", default="", help="per-card exam results: decks with weak cards are trained more")
    ap.add_argument("--pikl", default="", help="piKL exam only: lam (or 'base' = old margin search); score = Q + lam*log pi(net)")
    ap.add_argument("--pikl-part", default="0/1", help="k/K: this job plays every K-th batch from k")
    ap.add_argument("--pikl-games", type=int, default=256)
    ap.add_argument("--pikl-champ-games", type=int, default=512)
    ap.add_argument("--champ", default="", help="the current champion (frozen): exploiters' opponent and the vs-champion exam")
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
        elif a.reply:
            teacher = ReplyTeacher(vec, reply_ticks=a.reply_ticks, horizon=a.horizon, margin=a.margin, spread=a.spread,
                                   rng=np.random.default_rng(a.seed + 77))
        else:
            teacher = Teacher(vec, horizon=a.horizon, margin=a.margin, spread=a.spread)
        teacher.h1, teacher.k2 = a.h1, a.k2
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

        MW = None
        if a.matchups and Path(a.matchups).exists():  # weights from the gap "teacher minus bot" per meta deck
            mw = json.load(open(a.matchups))
            MW = np.array([max(1e-3, float(mw.get(d["name"], 1.0))) for d in META]); MW /= MW.sum()
            print("matchup weights", round(float(MW.max() * len(META)), 2), "max x mean", flush=True)

        API = []
        if a.api_decks and Path(a.api_decks).exists():
            for d in json.load(open(a.api_decks, encoding="utf-8")):
                if len(d.get("cards", [])) == 8 and all(c in ROW for c in d["cards"]):
                    API.append({"name": d.get("name", "api"), "group": "api", "src": "api", "cards": d["cards"],
                                "evo": d.get("evo", [])[:2], "hero": []})
        API_P = None
        if API:  # card balance: decks of rarely played cards are sampled more often
            from collections import Counter
            cf_ = Counter(c for d in API for c in d["cards"])
            CW = {}
            if a.card_w and Path(a.card_w).exists():
                for c, (wn, gm) in json.load(open(a.card_w)).items():
                    if gm >= 30:
                        CW[c] = 1.0 + 3.0 * max(0.0, 0.5 - wn / gm)  # a 30% card: x1.6
            w_ = np.array([np.mean([cf_[c] ** -0.5 * CW.get(c, 1.0) for c in d["cards"]]) for d in API]); API_P = w_ / w_.sum()
            print("weak cards boosted", len(CW), flush=True)
        print("api decks", len(API), flush=True)
        card_use = defaultdict(int)

        def api_deck():
            return dict(API[int(rng.choice(len(API), p=API_P))])

        def train_decks():  # opponents: endless generated decks, and meta decks where the gap to the teacher is largest
            out_ = []
            for _ in range(N):
                r_ = grng.random()
                if r_ < a.meta_share:
                    out_.append(META[int(rng.choice(len(META), p=MW))] if MW is not None else grng.choice(META))
                elif API and r_ < a.meta_share + a.api_share:
                    out_.append(api_deck())
                elif MW is None and r_ < a.meta_share + a.weak_share:
                    out_.append(grng.choice(WEAK))
                else:
                    out_.append(decks_gen.generate(grng, META, p_meta=0.0))
            return out_

        def train_mine():  # any deck for the agent: a new one every battle, X-Bow now and then
            out_ = []
            for _ in range(N):
                r_ = grng.random()
                d_ = MYD if r_ < a.xbow_share else (api_deck() if API and r_ < a.xbow_share + 0.4 else decks_gen.generate(grng, META))
                for c in d_["cards"]:
                    card_use[c] += 1
                out_.append(d_)
            return out_

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
        anchor_net = None
        if a.anchor > 0:
            apath = a.anchor_pt or a.resume
            anchor_net = Net(enc, NA + 1); print("anchor", apath, anchor_net.load_state_dict(torch.load(apath), strict=False), flush=True)
            anchor_net.eval()
            for p_ in anchor_net.parameters():
                p_.requires_grad_(False)
        champ = None
        cpath = a.champ or a.resume
        if cpath and Path(cpath).exists():
            try:
                champ = Net(enc, NA + 1); champ.load_state_dict(torch.load(cpath), strict=False); champ.eval()
            except Exception as e:  # noqa: BLE001
                champ = None; print("champ skipped", e, flush=True)
        if a.exploit and champ is not None:
            league = [("champ", champ)]; a.league_share = 1.0; a.strong = 0
        lstats = {nm: [0, 0] for nm, _ in league}
        print("league", [nm for nm, _ in league], flush=True)
        opp_kind = np.full(N, -1)

        if a.strong:  # the current network itself joins as an opponent (refreshed at every reset)
            self_net = Net(enc, NA + 1); self_net.eval(); league.append(("self", self_net)); lstats["self"] = [0, 0]
            a.league_share = max(a.league_share, 0.7)

        def assign_opps():
            opp_kind[:] = -1
            if a.strong:
                league[-1][1].load_state_dict(net.state_dict())
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
                z, _tf, _hd = net.encode(to_t(b)); h = net.gru(z, torch.zeros(len(O), 256)); h = h + net.wide(h)
                return net.v(h).squeeze(-1).numpy()
        if a.vteach > 0 and a.fork == 2:
            teacher.value_fn, teacher.vw = vfn, a.vteach
        if a.az and not a.fork:
            teacher.value_fn, teacher.vw = vfn, a.az_vw
        if a.sw > 0:
            teacher.value_fn, teacher.vw = vfn, a.sw

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

        conf_ = {"last": None}

        def net_act(m_, b_, M_, h_):
            with torch.no_grad():
                bt_ = to_t(b_); z_, tf_, hd_ = m_.encode(bt_); hn_ = m_.gru(z_, h_)
                lg_, _ = m_.heads(hn_, tf_, hd_, bt_, torch.from_numpy(M_))
                pr_ = torch.softmax(lg_, 1); play_ = pr_[:, 1:].clone(); play_[:, -1] = 0  # never the delegate
                ac_ = torch.where(play_.sum(1) > a.play_thr, play_.argmax(1) + 1, torch.zeros_like(pr_[:, 0]).long()).numpy()
                # how sure the network is: its chosen play's probability, or the wait's when it waits
                conf_["last"] = torch.where(torch.from_numpy(ac_) > 0, play_.max(1).values, 1.0 - play_.sum(1)).numpy()
                conf_["logp"] = torch.log(torch.cat([(1.0 - play_.sum(1, keepdim=True)).clamp_min(1e-6), play_.clamp_min(1e-6)], 1)).numpy()
                topk_ = (play_[:, :4 * enc.s.cols * enc.s.rows].topk(a.top, 1).indices + 1).numpy()
            return ac_, hn_, topk_

        if a.reply:  # the enemy's answer inside the search: our network from its side, fresh memory
            def reply_fn(Oe, Me):
                en = enc.entities(Oe); g, ids = enc.header(Oe, np.full((len(Oe), 8), -1, np.int64))
                hst = np.stack([Oe[:, enc.hcost] / 10.0, Oe[:, enc.hform] / 2.0, Oe[:, enc.hpend]], -1).astype(np.float32)
                dk, dst = enc.deck_feats(Oe)
                b_ = dict(rid=en["rid"], st=en["st"], kind=en["kind"], side=en["side"], num=en["num"], valid=en["valid"],
                          tile=en["tile"], g=g, hstat=hst, deck=dk, dstat=dst, **ids)
                M_ = np.concatenate([Me.astype(bool), np.zeros((len(Oe), 1), bool)], 1)
                ac_, _h, _t = net_act(net, b_, M_, torch.zeros(len(Oe), 256))
                return ac_
            teacher.reply_fn = reply_fn

        skipped_ = {"n": 0, "all": 0}

        def tap_noise(x, mk_, r_):
            if 0 < x < 2305 and r_.random() < 0.15:
                sl, tl = divmod(x - 1, 576); rr_, cc_ = divmod(tl, 18)
                r2, c2 = min(31, max(0, rr_ + int(r_.integers(-1, 2)))), min(17, max(0, cc_ + int(r_.integers(-1, 2))))
                y_ = 1 + sl * 576 + r2 * 18 + c2
                return y_ if mk_[y_] else x
            return x

        prof_ = {"obs": 0.0, "me": 0.0, "opp": 0.0, "step": 0.0, "n": 0}

        def run_eval(n_games, seed_base, any_deck=True, opp_any=False, player="net", part=None, opp="scripted"):
            """player: "net" (the network alone, as on the phone) or "teacher" (scripted + lookahead: the
            practical ceiling of each matchup). part=(k, K): only every K-th batch, from k (split jobs)."""
            ev = {"wins": 0, "losses": 0, "draws": 0, "faults": 0, "by_deck": {}, "by_group": {}, "by_phase": {}, "by_card": {}}
            erng = np.random.default_rng(12345)
            net.eval()
            for bi in range(max(1, n_games // N)):
                if part is not None and bi % part[1] != part[0]:
                    continue
                ds = [(ANY_OPP if opp_any else META)[(bi * N + i) % len(ANY_OPP if opp_any else META)] for i in range(N)]
                mine = [ANY[(bi * N + i) % len(ANY)] for i in range(N)] if any_deck else None
                seats = reset_all(seed_base + bi * 1009, ds, mine=mine)
                st = fresh(); stc = fresh(); done = np.zeros(N, bool); ot = np.zeros(N, bool)
                pend = np.zeros(N, np.int64); nrng = np.random.default_rng(seed_base + bi)
                for step_ in range(1800):
                    _t = time.perf_counter(); O, M, b = batch_obs(seats, st); prof_["obs"] += time.perf_counter() - _t; _t = time.perf_counter()
                    if player == "teacher":
                        pr_ = scripted_batch(vec, jpol, seats, seed_base + bi * 37 + step_ + 5)
                        ac = teacher.refine(pr_, seats, ~done)
                    elif player == "pikl":  # piKL / Gumbel-style choice: argmax over checked moves of Q(a) + lam*log pi(a)
                        ac0, st["h"], tk = net_act(net, b, M, st["h"])
                        lp_ = conf_["logp"]
                        ac = np.array(teacher.refine(ac0, seats, ~done, extra=[list(tk[i]) for i in range(N)]), np.int64)
                        ac[:] = ac0
                        for i in range(N):
                            sf = teacher.soft[i]
                            if done[i] or sf is None:
                                continue
                            ks_, p_ = sf
                            sc_ = np.log(np.maximum(p_, 1e-12)) + pikl_lam[0] * lp_[i, np.asarray(ks_)]
                            ac[i] = int(ks_[int(np.argmax(sc_))])
                            pikl_st[0] += 1; pikl_st[1] += int(ac[i] != ac0[i])
                    elif player in ("search", "noisy"):  # B: the network proposes, a short lookahead picks (as the phone twin will)
                        ac0, st["h"], tk = net_act(net, b, M, st["h"])
                        act_ = ~done
                        skipped_["all"] += int((~done).sum())
                        if a.skip_conf > 0:  # the network is sure: its move stands without a search
                            act_ = act_ & (conf_["last"] < a.skip_conf)
                            skipped_["n"] += int((~done & ~act_).sum())
                        ac = teacher.refine(ac0, seats, act_, extra=[list(tk[i]) for i in range(N)])
                        if player == "noisy":  # the phone app's latency: ~125 ms on average
                            # one decision = 5 ticks of 1/30 s = 167 ms; a move lands one decision later with p=0.75
                            ac = np.asarray(ac, np.int64).copy(); ac_now = np.zeros(N, np.int64)
                            for i in range(N):
                                mk_ = vec.masks[i, seats[i]]; x_ = int(pend[i]); pend[i] = 0
                                if 0 < x_ and mk_[x_]:
                                    ac_now[i] = x_  # the delayed move from the previous decision
                                elif ac[i] > 0 and nrng.random() < 0.75:
                                    pend[i] = ac[i]
                                else:
                                    ac_now[i] = ac[i]
                            ac = ac_now
                    else:
                        with torch.no_grad():
                            bt = to_t(b); z, tf, hand = net.encode(bt); st["h"] = net.gru(z, st["h"])
                            lg, _ = net.heads(st["h"], tf, hand, bt, torch.from_numpy(M))
                            pr = torch.softmax(lg, 1)
                            play = pr[:, 1:].clone(); play[:, -1] = 0  # never the delegate
                            ac = torch.where(play.sum(1) > a.play_thr, play.argmax(1) + 1, torch.zeros_like(pr[:, 0]).long()).numpy()
                    prof_["me"] += time.perf_counter() - _t; _t = time.perf_counter()
                    acts = np.zeros((N, 2), np.int32)
                    if opp == "champ":
                        _Oc, Mc, bc = batch_obs(1 - seats, stc)
                        oppa, stc["h"], _ = net_act(champ, bc, Mc, stc["h"])
                    else:
                        oppa = scripted_batch(vec, jpol, 1 - seats, seed_base + bi * 31 + step_)
                    for i in range(N):
                        me = seats[i]
                        acts[i, me] = ac[i]
                        acts[i, 1 - me] = oppa[i]
                        ot[i] |= bool(O[i, enc.h["overtime"][0]])
                    prof_["opp"] += time.perf_counter() - _t; _t = time.perf_counter()
                    vec.step(acts)
                    prof_["step"] += time.perf_counter() - _t; prof_["n"] += 1
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
                                for c in (mine[i]["cards"] if mine else MYD["cards"]):
                                    bd = ev["by_card"].setdefault(c, [0, 0]); bd[0] += k == "wins"; bd[1] += 1
                    if done.all():
                        break
            g = ev["wins"] + ev["losses"] + ev["draws"]
            ev["games"] = g; ev["winrate"] = round(ev["wins"] / g, 4) if g else 0.0
            net.train()
            return ev

        if a.ceilings:  # the teacher on the same exam games: the practical ceiling of each matchup
            k_, K_ = (int(x) for x in a.ceilings.split("/"))
            t0c = time.time()
            ca = run_eval(a.eval_games, a.ceil_seed, player="teacher", part=(k_, K_))
            cx = run_eval(a.xbow_games, a.ceil_seed, any_deck=False, player="teacher", part=(k_, K_))
            json.dump({"any": ca["by_deck"], "xbow": cx["by_deck"], "any_wr": ca["winrate"], "xbow_wr": cx["winrate"],
                       "games": [ca["games"], cx["games"]], "seconds": round(time.time() - t0c)},
                      open(out / "ceilings.json", "w"), ensure_ascii=False)
            print("ceilings", k_, K_, ca["winrate"], cx["winrate"], ca["games"], cx["games"], round(time.time() - t0c), flush=True)
            return
        if a.export_phone:  # the phone's policy: one ONNX graph from the raw observation, checked step by step
            from phone import check_and_export
            Path(a.export_phone).mkdir(parents=True, exist_ok=True)
            seats = reset_all(4242, [random.choice(META) for _ in range(N)], False, [decks_gen.generate(grng, META) for _ in range(N)])
            st = fresh(); batches = []
            for k in range(300):
                O, M, b = batch_obs(seats, st)
                with torch.no_grad():
                    bt = to_t(b); z, tf, hd = net.encode(bt); hn = net.gru(z, st["h"])
                    lg, vv = net.heads(hn, tf, hd, bt, torch.from_numpy(M))
                st["h"] = hn
                batches.append((O.astype(np.int32), M[:, :-1].copy(), lg.numpy(), vv.numpy()))
                acts = np.zeros((N, 2), np.int32)
                acts[np.arange(N), seats] = scripted_batch(vec, jpol, seats, 9000 + k)
                acts[np.arange(N), 1 - seats] = scripted_batch(vec, jpol, 1 - seats, 19000 + k)
                vec.step(acts)
                if vec.dones.any():
                    print("an episode ended at step", k, "- the check stops here", flush=True)
                    break
            res = check_and_export(net, enc, batches, a.export_phone)
            # the phone's search (Java PhoneSearch) against train/teacher.py on the same battles and moves
            import jpype
            JPS = jpype.JClass("org.crforge.env.PhoneSearch")(int(a.horizon), int(a.spread), float(a.margin), 1.0)
            JPS.h1 = int(a.h1); JPS.k2 = int(a.k2)
            JI = jpype.JArray(jpype.JInt)
            seats = reset_all(777, [random.choice(META) for _ in range(N)], False, [decks_gen.generate(grng, META) for _ in range(N)])
            st = fresh(); same = tot = changed = 0; t_py = t_j = 0.0
            for k in range(80):
                O, M, b = batch_obs(seats, st)
                ac0, st["h"], tk = net_act(net, b, M, st["h"])
                if k % 4 == 0:
                    t1 = time.time(); py = teacher.refine(ac0, seats, np.ones(N, bool), extra=[list(tk[i]) for i in range(N)]); t_py += time.time() - t1
                    t1 = time.time()
                    jv = np.asarray(JPS.refineVector(vec._env, JI(np.arange(N, dtype=np.int32)), JI(seats.astype(np.int32)),
                                                     JI(np.asarray(ac0, np.int32)), JI(np.asarray(tk, np.int32).reshape(-1)), int(tk.shape[1])))
                    t_j += time.time() - t1
                    same += int((py == jv).sum()); tot += N; changed += int((py != ac0).sum())
                acts = np.zeros((N, 2), np.int32)
                acts[np.arange(N), seats] = scripted_batch(vec, jpol, seats, 5000 + k)
                acts[np.arange(N), 1 - seats] = scripted_batch(vec, jpol, 1 - seats, 15000 + k)
                vec.step(acts)
                if vec.dones.any():
                    break
            res["search_java_vs_python"] = same / max(1, tot); res["search_changed"] = changed / max(1, tot)
            res["search_ms_python"] = round(1000 * t_py / max(1, tot), 2); res["search_ms_java_serial"] = round(1000 * t_j / max(1, tot), 2)
            print("search check", res, flush=True)
            meta = {"seat_ints": int(enc.s.seat_ints), "actions": int(vec.masks.shape[-1]), "play_thr": a.play_thr,
                    "vocabulary": [str(x) for x in vec.vocabulary], "cols": int(enc.s.cols), "rows": int(enc.s.rows),
                    "decks": {nm: {"name": d["name"], "rows": [ROW[c] for c in d["cards"]], "slots": [int(x) for x in slots_of(d)]}
                              for nm, d in (("xbow", MYD), ("opp", META[0]), ("opp2", META[len(META) // 2]))},
                    "horizon": a.horizon, "spread": a.spread, "margin": a.margin, "top": a.top, "h1": a.h1, "k2": a.k2, "skip_conf": a.skip_conf,
                    "check": res}
            json.dump(meta, open(Path(a.export_phone) / "meta.json", "w"))
            return
        pikl_lam = [0.0]; pikl_st = [0, 0]
        if a.pikl:  # piKL exam: network + anchored search vs scripted bots and vs the frozen champion (net alone)
            k_, K_ = (int(x) for x in a.pikl_part.split("/"))
            mode_ = {"base": "search", "net": "net"}.get(a.pikl, "pikl")
            if mode_ == "pikl":
                pikl_lam[0] = float(a.pikl); teacher.margin = 0.0
            t0p = time.time(); res = {"lam": a.pikl, "part": a.pikl_part}
            e1 = run_eval(a.pikl_games, 10 ** 6, player=mode_, part=(k_, K_))
            res["scripted"] = {k: e1[k] for k in ("games", "wins", "losses", "draws", "winrate")}
            print("pikl scripted", res["scripted"], round(time.time() - t0p), flush=True)
            if champ is not None and a.pikl_champ_games:
                e2 = run_eval(a.pikl_champ_games, 3 * 10 ** 6, player=mode_, part=(k_, K_), opp="champ")
                res["champ"] = {k: e2[k] for k in ("games", "wins", "losses", "draws", "winrate")}
                print("pikl champ", res["champ"], round(time.time() - t0p), flush=True)
            res["changed"] = pikl_st[1] / max(1, pikl_st[0]); res["seconds"] = round(time.time() - t0p)
            res["prof_ms_per_decision"] = {k: round(1000 * v / max(1, prof_["n"]), 2) for k, v in prof_.items() if k != "n"}
            res["prof_decisions"] = prof_["n"]; res["prof_search"] = dict(teacher.stats)
            print("PROF", res["prof_ms_per_decision"], prof_["n"], teacher.stats, flush=True)
            json.dump(res, open(out / "pikl.json", "w"))
            return
        if a.search_only:  # the search settings on the same games (paired), until the time budget runs out
            t0s = time.time(); res = {"search": [0, 0], "noisy": [0, 0], "chunks": 0}
            k = 0
            while time.time() - t0s < a.search_only * 60:
                for mode in ("search", "noisy"):
                    e_ = run_eval(N, 10 ** 6 + 7919 * k + a.search_seed, player=mode)
                    res[mode][0] += e_["wins"] + 0.5 * e_["draws"]; res[mode][1] += e_["games"]
                k += 1; res["chunks"] = k
                res.update(top=a.top, horizon=a.horizon, spread=a.spread, sw=a.sw, reply=a.reply, seconds=round(time.time() - t0s),
                           h1=a.h1, k2=a.k2, skip_conf=a.skip_conf,
                           ticks_per_search=round(teacher.stats["ticks"] / max(1, teacher.stats["searched"]), 1),
                           ticks_per_decision=round(teacher.stats["ticks"] / max(1, skipped_["all"]), 1),
                           skipped=round(skipped_["n"] / max(1, skipped_["all"]), 3),
                           ms_per_search=round(1000 * teacher.stats.get("t_reply", 0.0) / max(1, teacher.stats["searched"]), 2),
                           enemy_played=round(teacher.stats.get("reply_plays", 0) / max(1, teacher.stats.get("replies", 0)), 3),
                           search_wr=res["search"][0] / max(1, res["search"][1]), noisy_wr=res["noisy"][0] / max(1, res["noisy"][1]))
                json.dump(res, open(out / "search_variant.json", "w"))
                print("search variant", res, flush=True)
            return
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
                        SKa=np.stack(bf["sk"]), SPa=np.stack(bf["sp"]), RET=RET_, h0=h0_, X=np.stack(bf["x"]), OW=np.stack(bf["o"]))

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
                elif a.selfsearch:  # expert iteration: search around the network's own move (+ its top plays + the old proposal)
                    p_old = a.p_old0 + (a.p_old1 - a.p_old0) * min(1.0, (time.time() - t0) / max(1, t_train_end - t0))
                    nb_ = lg.argmax(1).numpy(); nb_ = np.where(nb_ == lg.shape[1] - 1, prop, nb_)  # delegate -> scripted
                    base_ = np.where(rng.random(N) < p_old, prop, nb_)
                    extra = [[int(x) for x, ok in zip(gk[i], okp[i]) if ok] + ([int(prop[i])] if 0 < prop[i] < 2305 else []) for i in range(N)]
                    tact = teacher.refine(base_, seats, np.ones(N, bool), extra); soft = teacher.soft
                elif a.az:
                    extra = [[int(x) for x, ok in zip(gk[i], okp[i]) if ok] for i in range(N)]
                    tact = teacher.refine(prop, seats, np.ones(N, bool), extra); soft = teacher.soft
                else:
                    tact = teacher.refine(prop, seats, np.ones(N, bool)); soft = teacher.soft
                SK, SP = soft_arrays(soft)
                if a.both:  # the teacher also plays/labels the opponent's seat: twice the labels, a stronger opponent
                    tactB = teacher.refine(propB, oseat, np.ones(N, bool)); SKB, SPB = soft_arrays(teacher.soft)
                else:
                    tactB = propB; SKB, SPB = soft_arrays([None] * N)
                tm["teacher"] += time.time() - t1; t1 = time.time()
                use_t = rng.random(N) < beta
                own_ok = np.zeros(N, np.float32)
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
                        own_ok[i] = float(x == int(own[i]))
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
                               ("v", vpred.numpy()), ("r", rr), ("x", acts[np.arange(N), seats].astype(np.int64)), ("o", own_ok)):
                    buf[k].append(val)
                MbL = Mb.copy()
                for k, val in (("b", bb), ("m", MbL), ("y", tactB), ("d", d), ("ah", ahB), ("ae", aeB), ("sk", SKB), ("sp", SPB),
                               ("v", vpb.numpy()), ("r", -rr), ("x", acts[np.arange(N), 1 - seats].astype(np.int64)),
                               ("o", np.zeros(N, np.float32))):
                    bufB[k].append(val)
            t1 = time.time()
            keys = buf["b"][0].keys()
            packs = [make_pack(buf, h0, st, seats)]
            if a.both:
                packs.append(make_pack(bufB, h0B, st_b, 1 - seats))
            torch.set_num_threads(4)
            acc_p = []; acc_w = []

            pg_l = []; kl_l = []

            def step_chunk(pk, ix, fresh_=False):
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
                if a.v0_w > 0:  # the search asks the value of a position with a fresh memory: train exactly that
                    h0z = net.gru(z.reshape(T * nb, -1), torch.zeros(T * nb, 256)); h0z = h0z + net.wide(h0z)
                    lval = lval + a.v0_w * F.mse_loss(net.v(h0z).squeeze(-1), torch.from_numpy(pk["RET"][:, ix].reshape(-1)))
                loss = lbc + a.soft_w * lsoft + a.val_w * lval + 0.1 * la + 0.1 * le
                if anchor_net is not None:
                    with torch.no_grad():
                        za, tfa, handa = anchor_net.encode(flat); za = za.view(T, nb, -1)
                        ha = torch.zeros(nb, 256); hsa = []
                        for t in range(T):
                            ha = anchor_net.gru(za[t], ha); hsa.append(ha); ha = ha * (1 - Dm[t]).unsqueeze(1)
                        lga, _ = anchor_net.heads(torch.stack(hsa).view(T * nb, -1), tfa, handa, flat, mk)
                        pa = F.softmax(lga, -1)
                    lkl = torch.where(pa > 0, pa * (torch.log(pa.clamp(min=1e-12)) - logp), torch.zeros_like(pa)).sum(1).mean()
                    loss = loss + a.anchor * lkl; kl_l.append(float(lkl.detach()))
                if a.pg > 0 and fresh_ and "OW" in pk:  # D: learn from results on its own moves; BC keeps it near the teacher
                    ow = torch.from_numpy(pk["OW"][:, ix].reshape(-1))
                    if ow.sum() > 8:
                        adv = torch.from_numpy(pk["RET"][:, ix].reshape(-1)) - vv_.detach()
                        adv = (adv / (adv[ow > 0].std() + 1e-6)).clamp(-3, 3)
                        xa = torch.from_numpy(pk["X"][:, ix].reshape(-1)).long()
                        lpg = -(adv * logp.gather(1, xa.unsqueeze(1)).squeeze(1) * ow).sum() / ow.sum()
                        loss = loss + a.pg * lpg; pg_l.append(float(lpg.detach()))
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
                with torch.no_grad():
                    pr = lg.argmax(1); pl = y != 0
                    if pl.any():
                        acc_p.append((pr[pl] == y[pl]).float().mean().item())
                    acc_w.append((pr[~pl] == 0).float().mean().item() if (~pl).any() else 1.0)
                return lbc, lsoft, lval, la

            for pack in packs:
                for ep in range(a.epochs):
                    perm = np.random.permutation(N)
                    for j in range(0, N, a.mb):
                        lbc, lsoft, lval, la = step_chunk(pack, perm[j:j + a.mb], fresh_=(ep == 0))
            # replay: the teacher's labels are expensive, so recent batches are trained on again
            for _ in range(max(1, a.replay_chunks * 16 // a.mb) if replay else 0):  # same number of replayed sequences
                old = replay[rng.integers(len(replay))]
                step_chunk(old, rng.permutation(N)[:a.mb])
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
                 "league": {k: v[:] for k, v in lstats.items()}, "pg": round(float(np.mean(pg_l)), 4) if pg_l else None, "kl": round(float(np.mean(kl_l)), 4) if kl_l else None}
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
        evx = run_eval(a.xbow_games, 10 ** 6, any_deck=False)  # X-Bow vs meta
        eva = run_eval(a.ava_games, 2 * 10 ** 6, opp_any=True)  # any deck vs any deck: both unseen
        ev["xbow"] = {k: evx[k] for k in ("games", "winrate", "wins", "losses", "by_group", "by_deck")}
        ev["any_vs_any"] = {k: eva[k] for k in ("games", "winrate", "wins", "losses", "by_phase")}
        ev["by_card"] = ev.get("by_card", {})
        ev["card_use"] = dict(card_use)
        if a.search_games:  # B: the same network with a short lookahead in battle
            evs = run_eval(a.search_games, 10 ** 6, player="search")
            ev["search"] = {k: evs[k] for k in ("games", "winrate", "wins", "losses")}
            print("search exam", evs["winrate"], evs["games"], flush=True)
        if a.noisy_games:
            evn = run_eval(a.noisy_games, 10 ** 6, player="noisy")
            ev["noisy"] = {k: evn[k] for k in ("games", "winrate", "wins", "losses")}
            print("noisy search exam", evn["winrate"], evn["games"], flush=True)
        if a.champ_games and champ is not None:  # how well this policy beats the current champion
            evc = run_eval(a.champ_games, 3 * 10 ** 6, opp="champ")
            ev["vs_champ"] = {k: evc[k] for k in ("games", "winrate", "wins", "losses")}
            print("vs champion", evc["winrate"], evc["games"], flush=True)
        ev["exploit"] = a.exploit
        ev["seed"] = a.seed
        ev["league"] = lstats
        json.dump(ev, open(out / "eval.json", "w"), ensure_ascii=False)
        print("eval", {k: ev[k] for k in ("games", "winrate", "wins", "losses", "draws", "faults")}, "xbow", evx["winrate"],
              "any_vs_any", eva["winrate"], flush=True)


if __name__ == "__main__":
    main()
