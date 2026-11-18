#!/usr/bin/env python3
"""Recurrent entity-based PPO worker (generation 2+).

Observation: every entity (type embedding, side, position, hp, state, flags, motion, target,
timers) through attention + a spatial grid CNN; header trackers (elixir both sides, rate phase,
overtime, towers, hand/queue/deck, enemy seen cards with dead-card memory and cycle, enemy
evolution progress); GRU memory across decisions. Actions: wait, slot x tile, abilities, and a
"delegate" action that plays the scripted policy's move (anchored, so the agent only leaves it
when that pays). Aux heads predict the enemy hand and the enemy elixir. Shaped rewards: tower
damage, crowns, elixir-value trades, spell value, elixir leak, X-Bow on tower, anti-spam.
Writes out/policy.pt (best mini-eval snapshot), metrics.json, eval.json.
"""
import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from crforge_env import LadderConfig, ScriptedBot, VectorBattle

ROOT = Path(__file__).resolve().parent
ROW = json.load(open(ROOT / "cards.json", encoding="utf-8"))
MY = {"cards": ["x-bow", "tesla", "archers", "knight", "skeletons", "electro-spirit", "fireball", "the-log"],
      "evo": ["tesla", "archers"], "hero": ["knight"]}
DECKS = json.load(open(ROOT / "decks.json", encoding="utf-8"))
META = [d for d in DECKS if d.get("src") == "meta"]
OTHER = [d for d in DECKS if d.get("src") != "meta"]
EM = 48  # entity rows read (towers and characters come first)
ENT_NUM = ["x", "y", "hp", "max_hp", "shield", "dx", "dy", "target", "target_x", "target_y", "deploy_ms",
           "attack_ms", "load_ms", "speed_percent", "hit_speed_percent", "buff_count", "aim_x", "aim_y",
           "time_left_ms", "radius", "level", "ability_ticks", "flags"]


def hv(h, name, k=None):
    idx = h.get(name)
    if idx is None:
        return None
    return idx if k is None else idx[:k]


class Enc:
    """Vectorised observation encoder over a batch of seat observations."""

    def __init__(self, vec):
        s = vec.schema
        self.s = s
        self.V = len(vec.vocabulary) + 1
        self.S = len(s.state_names) + 2
        h = s.header
        self.h = h
        self.H, self.EI = s.header_ints, s.entity_ints
        self.e = s.entity
        self.nflags = len(s.flag_names)
        self.cnt = h["entity_count"][0]
        names = [n for n in ("elixir", "enemy_elixir", "elixir_reserved", "time_ms", "section_left_ms", "full_bar_ms",
                             "overtime", "crowns", "crowns_enemy", "play_open_in_ticks", "pending_count",
                             "enemy_play_count", "section_type") if n in h]
        self.scalar_idx = [h[n][0] for n in names]
        self.scalar_names = names
        self.tw = [h[f"{t}{k}"][0] for t in ("own_king", "own_left", "own_right", "enemy_king", "enemy_left", "enemy_right")
                   for k in ("_hp", "_max_hp")]
        self.hand = list(h["hand_card"]); self.hcost = list(h["hand_cost"]); self.hform = list(h["hand_form"])
        self.hpend = list(h["hand_pending"])
        self.nxt = list(h["next_card"]); self.queue = list(h.get("queue", ()))
        self.seen = list(h["enemy_seen"])
        self.seen_evo = list(h.get("enemy_seen_evolution", ())); self.seen_need = list(h.get("enemy_seen_evolution_need", ()))
        self.devo = list(h.get("deck_evolution", ()))
        self.pc = list(h["enemy_play_card"]); self.pt = list(h["enemy_play_ticks"])
        self.px = list(h["enemy_play_x"]); self.py = list(h["enemy_play_y"])
        self.tick = h["tick"][0]
        self.ab = [h[n][0] for n in ("ability_state", "ability_cooldown_ms", "ability_charges") if n in h]
        self.vocab = vec.vocabulary
        z = np.zeros((1, s.seat_ints), np.int32)
        self.num_dim = self.entities(z)["num"].shape[-1]
        self.g_dim = self.header(z, np.full((1, 8), -1, np.int64))[0].shape[-1]

    def entities(self, O):
        B = O.shape[0]
        rows = O[:, self.H:self.H + EM * self.EI].reshape(B, EM, self.EI)
        n = np.minimum(O[:, self.cnt], EM)
        valid = np.arange(EM)[None, :] < n[:, None]
        e = self.e
        col = lambda k: rows[:, :, e[k]].astype(np.float32) if k in e else np.zeros((B, EM), np.float32)
        rid = np.clip(rows[:, :, e["row"]] + 1, 0, self.V - 1)
        st = np.clip(rows[:, :, e["state"]] + 2, 0, self.S - 1) if "state" in e else np.zeros((B, EM), np.int64)
        kind = np.clip(rows[:, :, e["kind"]], 0, 5)
        side = np.clip(rows[:, :, e["side"]], 0, 3)
        x, y = col("x"), col("y")
        hp, mh, sh = col("hp"), col("max_hp"), col("shield")
        known = (hp >= 0).astype(np.float32)
        tgt = col("target")
        num = [x / 18000.0, y / 32000.0,
               known, np.where(hp >= 0, hp / np.maximum(1, mh), 0), np.log1p(np.maximum(0, mh)) / 9.0,
               np.clip(sh, 0, None) / 1000.0, col("dx") / 1000.0, col("dy") / 1000.0,
               (tgt >= 0).astype(np.float32), np.where(tgt >= 0, col("target_x") / 18000.0, 0),
               np.where(tgt >= 0, col("target_y") / 32000.0, 0), col("deploy_ms") / 2000.0,
               col("attack_ms") / 3000.0, col("load_ms") / 3000.0, col("speed_percent") / 100.0]
        rest = [col("hit_speed_percent") / 100.0, col("buff_count") / 3.0, col("aim_x") / 18000.0, col("aim_y") / 32000.0,
                col("time_left_ms") / 10000.0, col("radius") / 5000.0, col("level") / 16.0,
                (col("ability_ticks") >= 0).astype(np.float32)]
        fl = rows[:, :, e["flags"]] if "flags" in e else np.zeros((B, EM), np.int64)
        bits = [((fl >> b) & 1).astype(np.float32) for b in range(self.nflags)]
        feats = np.stack(num + rest + bits, -1).astype(np.float32)
        feats = np.nan_to_num(np.clip(feats, -5, 5))
        tc = np.clip((x // 1000).astype(np.int64), 0, 17)
        tr = np.clip((y // 1000).astype(np.int64), 0, 31)
        return dict(rid=rid.astype(np.int64), st=st.astype(np.int64), kind=kind.astype(np.int64),
                    side=side.astype(np.int64), num=feats, valid=valid, tile=(tr * 18 + tc), rows=rows, n=n)

    def header(self, O, last_play):
        """last_play: [B, 8] tick each enemy-seen card was last played (memory), -1 never."""
        B = O.shape[0]
        f = []
        sc = O[:, self.scalar_idx].astype(np.float32)
        div = {"elixir": 100000, "enemy_elixir": 100000, "elixir_reserved": 100000, "time_ms": 300000,
               "section_left_ms": 180000, "full_bar_ms": 1, "overtime": 1, "crowns": 3, "crowns_enemy": 3,
               "play_open_in_ticks": 100, "pending_count": 4, "enemy_play_count": 60, "section_type": 8}
        for j, nme in enumerate(self.scalar_names):
            if nme == "full_bar_ms":
                sc[:, j] = np.clip(28000.0 / np.maximum(1.0, sc[:, j]), 0, 4)  # 1x / 2x / 3x
            else:
                sc[:, j] /= div[nme]
        f.append(sc)
        el = O[:, self.h["elixir"][0]]
        f.append(np.stack([el >= 100000, el >= 70000, O[:, self.h["enemy_elixir"][0]] <= 30000,
                           O[:, self.h["enemy_elixir"][0]] >= 90000], 1).astype(np.float32))
        t = O[:, self.tw].astype(np.float32).reshape(B, 6, 2)
        f.append(t[:, :, 0] / np.maximum(1, t[:, :, 1]))
        f.append(O[:, self.hcost].astype(np.float32) / 10.0)
        f.append(O[:, self.hform].astype(np.float32) / 2.0)
        f.append(O[:, self.hpend].astype(np.float32))
        dev = O[:, self.devo].astype(np.float32) if self.devo else np.zeros((B, 8), np.float32)
        f.append(np.clip(dev, -1, 4) / 4.0)
        seen = O[:, self.seen]
        tick = O[:, self.tick].astype(np.float32)
        has = (seen >= 0).astype(np.float32)
        since = np.where(last_play >= 0, (tick[:, None] - last_play) / 1200.0, 2.0)  # 1200 ticks = 60 s
        pc, pt = O[:, self.pc], O[:, self.pt]
        recent4 = ((seen[:, :, None] == pc[:, None, :4]) & (pc[:, None, :4] >= 0)).any(-1).astype(np.float32)
        dead = ((since > 0.5) & (has > 0)).astype(np.float32)  # held > 30 s
        f.append(np.concatenate([has, np.clip(since, 0, 2), recent4, dead], 1))
        if self.seen_evo:
            se, sn = O[:, self.seen_evo].astype(np.float32), O[:, self.seen_need].astype(np.float32)
            f.append(np.concatenate([np.clip(se, -1, 8) / 8, (se >= 0).astype(np.float32), np.where(sn > 0, np.clip(se, 0, None) / np.maximum(1, sn), 0)], 1))
        else:
            f.append(np.zeros((B, 24), np.float32))
        f.append(np.concatenate([np.clip(pt.astype(np.float32), 0, 2400) / 2400.0,
                                 O[:, self.px].astype(np.float32) / 18000.0, O[:, self.py].astype(np.float32) / 32000.0], 1))
        if self.ab:
            a = O[:, self.ab].astype(np.float32)
            f.append(np.concatenate([np.clip(a, -1, 30000) / 30000.0, (a > 0).astype(np.float32)], 1))
        g = np.nan_to_num(np.concatenate(f, 1).astype(np.float32))
        ids = dict(hand=np.clip(O[:, self.hand] + 1, 0, self.V - 1), nxt=np.clip(O[:, self.nxt + self.queue] + 1, 0, self.V - 1),
                   seen=np.clip(seen + 1, 0, self.V - 1), play=np.clip(pc + 1, 0, self.V - 1))
        return g, {k: v.astype(np.int64) for k, v in ids.items()}


class Net(nn.Module):
    def __init__(self, enc, na):
        super().__init__()
        D = 128
        self.na = na
        self.card = nn.Embedding(enc.V, 32)
        self.state = nn.Embedding(enc.S, 8)
        self.kind = nn.Embedding(6, 4)
        self.side = nn.Embedding(4, 4)
        self.ent = nn.Sequential(nn.Linear(32 + 8 + 4 + 4 + enc.num_dim, D), nn.ReLU(), nn.Linear(D, D), nn.ReLU())
        self.att = nn.MultiheadAttention(D, 4, batch_first=True)
        self.ln = nn.LayerNorm(D)
        self.to_grid = nn.Linear(D, 16)
        self.cnn = nn.Sequential(nn.Conv2d(18, 24, 3, padding=1), nn.ReLU(), nn.Conv2d(24, 32, 3, padding=1), nn.ReLU())
        gin = enc.g_dim + 32 * 4 + 32 + 32 + 32
        self.glob = nn.Sequential(nn.Linear(gin, 256), nn.ReLU())
        self.trunk = nn.Sequential(nn.Linear(2 * D + 256 + 32, 256), nn.ReLU())
        self.gru = nn.GRUCell(256, 256)
        self.wait = nn.Linear(256, 1)
        self.slot = nn.Sequential(nn.Linear(256 + 32 + 3, 128), nn.ReLU(), nn.Linear(128, 33))
        self.abil = nn.Linear(256, 2)
        self.dele = nn.Linear(256, 1)
        self.v = nn.Linear(256, 1)
        self.aux_hand = nn.Linear(256, enc.V)
        self.aux_el = nn.Linear(256, 1)
        ys, xs = torch.meshgrid(torch.linspace(0, 1, 32), torch.linspace(0, 1, 18), indexing="ij")
        self.register_buffer("coords", torch.stack([xs, ys])[None])

    def encode(self, b):
        """b: dict of tensors for B samples -> features before the GRU [B,256], tile feats [B,32,576]."""
        rid, valid = b["rid"], b["valid"]
        B = rid.shape[0]
        x = torch.cat([self.card(rid), self.state(b["st"]), self.kind(b["kind"]), self.side(b["side"]), b["num"]], -1)
        x = self.ent(x)
        pad = ~valid
        allpad = pad.all(1)
        pad2 = pad.clone(); pad2[allpad, 0] = False
        a, _ = self.att(x, x, x, key_padding_mask=pad2, need_weights=False)
        x = self.ln(x + a)
        m = valid.unsqueeze(-1).float()
        mean = (x * m).sum(1) / m.sum(1).clamp(min=1)
        mx = x.masked_fill(~valid.unsqueeze(-1), -1e4).max(1).values.clamp(min=-10)
        gv = self.to_grid(x) * m
        grid = torch.zeros(B, 16, 576, device=x.device)
        grid.scatter_add_(2, b["tile"].unsqueeze(1).expand(-1, 16, -1), gv.transpose(1, 2))
        grid = torch.cat([grid.view(B, 16, 32, 18), self.coords.expand(B, -1, -1, -1)], 1)
        tf = self.cnn(grid)  # [B,32,32,18]
        hand = self.card(b["hand"])  # [B,4,32]
        g = torch.cat([b["g"], hand.flatten(1), self.card(b["nxt"]).mean(1), self.card(b["seen"]).mean(1),
                       self.card(b["play"]).mean(1)], -1)
        g = self.glob(g)
        sp = tf.mean((2, 3))
        z = self.trunk(torch.cat([mean, mx, g, sp], -1))
        return z, tf.flatten(2), hand

    def heads(self, h, tf, hand, b, mask):
        B = h.shape[0]
        hs = torch.cat([h.unsqueeze(1).expand(-1, 4, -1), hand, b["hstat"]], -1)
        q = self.slot(hs)  # [B,4,33]
        tiles = torch.einsum("bct,bsc->bst", tf, q[:, :, :32]) + q[:, :, 32:]  # [B,4,576]
        lg = torch.cat([self.wait(h), tiles.reshape(B, -1), self.abil(h), self.dele(h)], -1)
        lg = lg.masked_fill(~mask, -1e9)
        return lg, self.v(h).squeeze(-1)


def to_t(d, keys=None):
    return {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in d.items() if keys is None or k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--install", required=True)
    ap.add_argument("--tables", required=True)
    ap.add_argument("--out", default="out")
    ap.add_argument("--minutes", type=float, default=95)
    ap.add_argument("--eval-minutes", type=float, default=10)
    ap.add_argument("--eval-games", type=int, default=640)
    ap.add_argument("--envs", type=int, default=128)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--resume", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ent", type=float, default=0.003)
    ap.add_argument("--anchor", type=float, default=0.02)
    ap.add_argument("--mini-eval-min", type=float, default=25)
    ap.add_argument("--T", type=int, default=64)
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
        NA = s.actions
        DEL = NA
        per = s.cols * s.rows
        vocab = vec.vocabulary
        types = vec.card_types()
        items = types.items() if isinstance(types, dict) else enumerate(types)
        spell_ids = {int(k) for k, v in items if str(v).lower() == "spell"}
        print("spells", len(spell_ids), flush=True)
        xbow_rows = {i + 1 for i, n in enumerate(vocab) if "xbow" in n.lower().replace("-", "").replace("_", "")}
        my_rows = [ROW[k] for k in MY["cards"]]
        my_slots = [(s.slot_evolution if k in MY["evo"] else 0) | (s.slot_hero if k in MY["hero"] else 0) for k in MY["cards"]]
        I = s.info
        bots = [ScriptedBot().bind(vec), ScriptedBot().bind(vec)]

        def slots_of(d):
            return [(s.slot_evolution if k in d.get("evo", []) else 0) | (s.slot_hero if k in d.get("hero", []) else 0) for k in d["cards"]]

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
            return [random.choice(META) if random.random() < 0.7 else random.choice(OTHER) for _ in range(N)]

        net = Net(enc, NA + 1)
        with torch.no_grad():
            net.dele.bias.fill_(10.0)
        if a.resume and Path(a.resume).exists():
            try:
                net.load_state_dict(torch.load(a.resume))
                print("resumed", flush=True)
            except Exception as e:  # noqa: BLE001  другая архитектура — старт заново
                print("resume skipped", e, flush=True)
        opt = torch.optim.Adam(net.parameters(), lr=a.lr, eps=1e-5)

        # ---- per-env state ----
        def fresh_state():
            return dict(h=torch.zeros(N, 256), last_play=np.full((N, 8), -1, np.int64), prev_seen=np.full((N, 8), -1, np.int64),
                        tw=None, cr=None, ids_me={}, ids_en={}, recent=[[] for _ in range(N)], ot=np.zeros(N, bool))

        def batch_obs(seats, st):
            O = np.stack([vec.observations[i, seats[i]] for i in range(N)])
            M = np.stack([vec.masks[i, seats[i]].astype(bool) for i in range(N)])
            M = np.concatenate([M, np.ones((N, 1), bool)], 1)
            # dead-card memory update
            seen = O[:, enc.seen]
            st["last_play"] = np.where(seen != st["prev_seen"], -1, st["last_play"])
            st["prev_seen"] = seen
            tick = O[:, enc.tick][:, None]
            pc, pt = O[:, enc.pc], O[:, enc.pt]
            match = (seen[:, :, None] == pc[:, None, :]) & (pc[:, None, :] >= 0)
            lp = np.where(match, (tick - pt)[:, None, :], -1).max(-1)
            st["last_play"] = np.maximum(st["last_play"], lp)
            en = enc.entities(O)
            g, ids = enc.header(O, st["last_play"])
            hst = np.stack([O[:, enc.hcost] / 10.0, O[:, enc.hform] / 2.0, O[:, enc.hpend]], -1).astype(np.float32)
            b = dict(rid=en["rid"], st=en["st"], kind=en["kind"], side=en["side"], num=en["num"], valid=en["valid"],
                     tile=en["tile"], g=g, hstat=hst, **ids)
            return O, M, b, en

        def opp_targets(seats):
            Oo = np.stack([vec.observations[i, 1 - seats[i]] for i in range(N)])
            hand = np.zeros((N, enc.V), np.float32)
            hc = np.clip(Oo[:, enc.hand] + 1, 0, enc.V - 1)
            for j in range(4):
                hand[np.arange(N), hc[:, j]] = 1.0
            hand[:, 0] = 0
            return hand, Oo[:, enc.h["elixir"][0]].astype(np.float32) / 100000.0

        def value_of(mh):
            return np.minimum(8.0, np.maximum(0, mh) / 350.0)

        def shaped(i, O, en, act, st):
            """Reward for env i from its new observation (my seat)."""
            r = 0.0
            t = O[enc.tw].astype(np.float32).reshape(6, 2)
            mine = t[:3, 0].sum() / max(1, t[:3, 1].sum()); theirs = t[3:, 0].sum() / max(1, t[3:, 1].sum())
            if st["tw"][i] is not None:
                pm, pt_ = st["tw"][i]
                r += 0.5 * ((pt_ - theirs) - (pm - mine))
            st["tw"][i] = (mine, theirs)
            cr = (int(O[enc.h["crowns"][0]]), int(O[enc.h["crowns_enemy"][0]]))
            if st["cr"][i] is not None:
                r += 0.15 * ((cr[0] - st["cr"][i][0]) - (cr[1] - st["cr"][i][1]))
            st["cr"][i] = cr
            n = int(en["n"][i]); rows = en["rows"][i, :n]
            e = enc.e
            ch = rows[rows[:, e["kind"]] == 1]
            for side, key, sign in ((0, "ids_me", -1.0), (1, "ids_en", 1.0)):
                cur = {int(q[e["id"]]): float(q[e["max_hp"]]) for q in ch[ch[:, e["side"]] == side]}
                prev = st[key].get(i, {})
                gone = sum(value_of(v) for k, v in prev.items() if k not in cur)
                r += sign * 0.03 * gone
                st[key][i] = cur
            if O[enc.h["elixir"][0]] >= 99000 and O[enc.h["play_open_in_ticks"][0]] == 0:
                r -= 0.003
            # X-Bow locked on a tower
            xb = ch[(ch[:, e["side"]] == 0) & np.isin(ch[:, e["row"]] + 1, list(xbow_rows))] if xbow_rows else ch[:0]
            if len(xb) and ((xb[:, e["target"]] >= 0) & (xb[:, e["target_y"]] >= 24500)).any():
                r += 0.004
            if 0 < act < 1 + 4 * per:
                sl, tile = divmod(act - 1, per)
                rec = st["recent"][i]
                cid = int(st["hand_ids"][i][sl])
                if (cid, tile) in rec:
                    r -= 0.02
                rec.append((cid, tile)); del rec[:-3]
                if cid in spell_ids:
                    row_, col_ = divmod(tile, s.cols)
                    px, py = col_ * 1000 + 500, row_ * 1000 + 500
                    en_all = rows[(rows[:, e["side"]] == 1) & (rows[:, e["kind"]] == 1)]
                    d = np.hypot(en_all[:, e["x"]] - px, en_all[:, e["y"]] - py) if len(en_all) else np.zeros(0)
                    k = int((d < 2500).sum())
                    tower_hit = (py > 24000) and abs(px - 9000) > 3000 or py > 28500
                    r += 0.01 * min(k, 5) if (k or tower_hit) else -0.05
            return r

        def act_batch(lg, sample=True, gen=None):
            if sample:
                return torch.multinomial(torch.softmax(lg, 1), 1, generator=gen).squeeze(1)
            return lg.argmax(1)

        def run_eval(n_games, seed_base):
            """Fixed evaluation vs meta decks; returns dict with by_deck / by_group / by_phase."""
            ev = {"wins": 0, "losses": 0, "draws": 0, "faults": 0, "by_deck": {}, "by_group": {}, "by_phase": {}}
            gen = torch.Generator().manual_seed(12345)
            erng = np.random.default_rng(12345)
            net.eval()
            for bi in range(max(1, n_games // N)):
                ds = [META[(bi * N + i) % len(META)] for i in range(N)]
                seats = reset_all(seed_base + bi * 1009, ds)
                st = fresh_state(); st["tw"] = [None] * N; st["cr"] = [None] * N
                done = np.zeros(N, bool)
                for _ in range(1800):
                    O, M, b, en = batch_obs(seats, st)
                    with torch.no_grad():
                        z, tf, hand = net.encode(to_t(b))
                        st["h"] = net.gru(z, st["h"])
                        lg, _ = net.heads(st["h"], tf, hand, to_t(b), torch.from_numpy(M))
                        ac = act_batch(lg, True, gen).numpy()
                    acts = np.zeros((N, 2), np.int32)
                    for i in range(N):
                        me = seats[i]
                        sb = bots[me](vec.observations[i, me], vec.masks[i, me], erng)
                        acts[i, me] = sb if ac[i] == DEL else ac[i]
                        acts[i, 1 - me] = bots[1 - me](vec.observations[i, 1 - me], vec.masks[i, 1 - me], erng)
                        st["ot"][i] |= bool(O[i, enc.h["overtime"][0]])
                    vec.step(acts)
                    for i in range(N):
                        if vec.dones[i] and not done[i]:
                            done[i] = True
                            w_ = int(vec.infos[i, I["winner"]])
                            k = "faults" if int(vec.infos[i, I["fault"]]) else ("wins" if w_ == seats[i] else ("losses" if w_ in (0, 1) else "draws"))
                            ev[k] += 1
                            if k != "faults":
                                for key, name in (("by_deck", ds[i]["name"]), ("by_group", ds[i].get("group", "?")),
                                                  ("by_phase", "overtime" if st["ot"][i] else "regular")):
                                    bd = ev[key].setdefault(name, [0, 0]); bd[0] += k == "wins"; bd[1] += 1
                    if done.all():
                        break
            g = ev["wins"] + ev["losses"] + ev["draws"]
            ev["games"] = g; ev["winrate"] = round(ev["wins"] / g, 4) if g else 0.0
            net.train()
            return ev

        # ---- training ----
        seats = reset_all(1 + a.seed * 7, train_decks())
        st = fresh_state(); st["tw"] = [None] * N; st["cr"] = [None] * N; st["hand_ids"] = np.zeros((N, 4), np.int64)
        ret_rms = [0.0, 1.0, 1e-4]  # mean, var, count of returns (value normalisation)
        results, ep_ret = [], np.zeros(N)
        best_sd, best_wr, last_mini = None, -1.0, time.time()
        it = 0
        tm = defaultdict(float)
        total_iters_est = None
        while time.time() < t_train_end:
            frac = min(1.0, (time.time() - t0) / max(1, t_train_end - t0))
            for gp in opt.param_groups:
                gp["lr"] = a.lr * (1.0 - 0.7 * frac)
            ent_c = a.ent * (1.0 - 0.7 * frac)
            buf = defaultdict(list)
            h0 = st["h"].clone()
            n_del = n_own = 0
            for t in range(T):
                t1 = time.time()
                O, M, b, en = batch_obs(seats, st)
                st["hand_ids"] = O[:, enc.hand]
                tm["obs"] += time.time() - t1; t1 = time.time()
                with torch.no_grad():
                    bt = to_t(b)
                    z, tf, hand = net.encode(bt)
                    hn = net.gru(z, st["h"])
                    lg, v = net.heads(hn, tf, hand, bt, torch.from_numpy(M))
                    dist = torch.distributions.Categorical(logits=lg)
                    ac = dist.sample(); lp = dist.log_prob(ac)
                tm["net"] += time.time() - t1; t1 = time.time()
                ahand, ael = opp_targets(seats)
                acts = np.zeros((N, 2), np.int32)
                acn = ac.numpy()
                for i in range(N):
                    me = seats[i]
                    sb = bots[me](vec.observations[i, me], vec.masks[i, me], rng)
                    acts[i, me] = sb if acn[i] == DEL else acn[i]
                    acts[i, 1 - me] = bots[1 - me](vec.observations[i, 1 - me], vec.masks[i, 1 - me], rng)
                n_del += int((acn == DEL).sum()); n_own += int(((acn != DEL) & (acn != 0)).sum())
                tm["bots"] += time.time() - t1; t1 = time.time()
                vec.step(acts)
                tm["sim"] += time.time() - t1; t1 = time.time()
                r = np.zeros(N, np.float32); d = np.zeros(N, np.float32)
                O2 = None
                for i in range(N):
                    me = seats[i]
                    if vec.dones[i]:
                        w_ = int(vec.infos[i, I["winner"]])
                        if not int(vec.infos[i, I["fault"]]):
                            r[i] = 1.0 if w_ == me else (-1.0 if w_ in (0, 1) else 0.0)
                            results.append(r[i])
                        d[i] = 1
                        st["tw"][i] = None; st["cr"][i] = None
                        st["ids_me"].pop(i, None); st["ids_en"].pop(i, None); st["recent"][i] = []
                        st["last_play"][i] = -1; st["prev_seen"][i] = -1
                    else:
                        if O2 is None:
                            O2 = np.stack([vec.observations[j, seats[j]] for j in range(N)]); en2 = enc.entities(O2)
                        r[i] += shaped(i, O2[i], en2, int(acts[i, me]) if acn[i] != DEL else 0, st)
                tm["reward"] += time.time() - t1
                st["h"] = hn * torch.from_numpy(1 - d).unsqueeze(1)
                for k, val in (("b", b), ("m", M), ("a", acn), ("lp", lp.numpy()), ("v", v.numpy()), ("r", r), ("d", d),
                               ("ah", ahand), ("ae", ael)):
                    buf[k].append(val)
            # bootstrap
            O, M, b, en = batch_obs(seats, st)
            with torch.no_grad():
                bt = to_t(b); z, tf, hand = net.encode(bt); hn = net.gru(z, st["h"])
                _, last_v = net.heads(hn, tf, hand, bt, torch.from_numpy(M))
            std = math.sqrt(ret_rms[1]) + 1e-6
            V = np.array(buf["v"] + [last_v.numpy()]) * std
            R, D = np.array(buf["r"]), np.array(buf["d"])
            adv = np.zeros_like(R); g_ = 0
            for t in reversed(range(T)):
                delta = R[t] + 0.997 * V[t + 1] * (1 - D[t]) - V[t]
                g_ = delta + 0.997 * 0.95 * (1 - D[t]) * g_
                adv[t] = g_
            ret = adv + V[:-1]
            bm, bv, bc = ret.mean(), ret.var(), ret.size  # update running return stats
            tot = ret_rms[2] + bc; dm = bm - ret_rms[0]
            ret_rms[0] += dm * bc / tot
            ret_rms[1] = (ret_rms[1] * ret_rms[2] + bv * bc + dm ** 2 * ret_rms[2] * bc / tot) / tot
            ret_rms[2] = tot
            std = math.sqrt(ret_rms[1]) + 1e-6
            t1 = time.time()
            # stack batch [T, N, ...]
            keys = buf["b"][0].keys()
            B = {k: np.stack([x[k] for x in buf["b"]]) for k in keys}
            Mm = np.stack(buf["m"]); A = np.stack(buf["a"]); LP = np.stack(buf["lp"])
            ADV = (adv - adv.mean()) / (adv.std() + 1e-8); RET = ret / std
            AH, AE = np.stack(buf["ah"]), np.stack(buf["ae"])
            torch.set_num_threads(4)  # the simulator is idle while learning
            for ep in range(1):
                perm = np.random.permutation(N)
                for j in range(0, N, 16):
                    ix = perm[j:j + 16]; nb = len(ix)
                    flat = {k: torch.from_numpy(np.ascontiguousarray(B[k][:, ix].reshape(T * nb, *B[k].shape[2:]))) for k in keys}
                    z, tf, hand = net.encode(flat)
                    z = z.view(T, nb, -1)
                    h = h0[ix]
                    hs = []
                    Dm = torch.from_numpy(D[:, ix])
                    for t in range(T):
                        h = net.gru(z[t], h); hs.append(h); h = h * (1 - Dm[t]).unsqueeze(1)
                    H = torch.stack(hs).view(T * nb, -1)
                    lg, v = net.heads(H, tf, hand, flat, torch.from_numpy(Mm[:, ix].reshape(T * nb, -1)))
                    dist = torch.distributions.Categorical(logits=lg)
                    at = torch.from_numpy(A[:, ix].reshape(-1)).long()
                    ratio = torch.exp(dist.log_prob(at) - torch.from_numpy(LP[:, ix].reshape(-1)))
                    ad = torch.from_numpy(ADV[:, ix].reshape(-1)).float()
                    l1 = -torch.min(ratio * ad, torch.clamp(ratio, 0.8, 1.2) * ad).mean()
                    l2 = 0.5 * ((v - torch.from_numpy(RET[:, ix].reshape(-1)).float()) ** 2).mean()
                    ent = dist.entropy().mean()
                    anchor = -F.log_softmax(lg, -1)[:, DEL].mean()
                    la = F.binary_cross_entropy_with_logits(net.aux_hand(H), torch.from_numpy(AH[:, ix].reshape(T * nb, -1)))
                    le = F.mse_loss(net.aux_el(H).squeeze(-1), torch.from_numpy(AE[:, ix].reshape(-1)))
                    loss = l1 + l2 - ent_c * ent + a.anchor * anchor + 0.1 * la + 0.1 * le
                    opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 0.5); opt.step()
            torch.set_num_threads(2)
            tm["learn"] += time.time() - t1
            it += 1
            last = results[-300:]
            m = {"iter": it, "t": round(time.time()), "episodes": len(results),
                 "winrate_last200": round(float(np.mean([x > 0 for x in last])), 3) if last else None,
                 "entropy": round(ent.item(), 3), "value_loss": round(l2.item(), 4), "aux_hand": round(la.item(), 4),
                 "delegate": round(n_del / (T * N), 3), "own_plays_per_100": round(100 * n_own / (T * N), 2),
                 "anchor": round(anchor.item(), 4)}
            metrics["iters"].append(m)
            metrics["timing"] = {k: round(v_, 1) for k, v_ in tm.items()}
            if it % 10 == 0:
                print(m, metrics["timing"], flush=True)
                json.dump(metrics, open(out / "metrics.json", "w"))
            if time.time() - last_mini > a.mini_eval_min * 60 and time.time() < t_train_end - 300:
                ev = run_eval(N, 777)
                metrics["mini_evals"].append({"iter": it, "winrate": ev["winrate"]})
                print("mini_eval", it, ev["winrate"], flush=True)
                if ev["winrate"] > best_wr:
                    best_wr = ev["winrate"]; best_sd = {k: v_.clone() for k, v_ in net.state_dict().items()}
                last_mini = time.time()
                seats = reset_all(100 + it + a.seed * 7, train_decks())
                st = fresh_state(); st["tw"] = [None] * N; st["cr"] = [None] * N; st["hand_ids"] = np.zeros((N, 4), np.int64)
            if it % 60 == 0:
                seats = reset_all(200 + it + a.seed * 7, train_decks())
                st = fresh_state(); st["tw"] = [None] * N; st["cr"] = [None] * N; st["hand_ids"] = np.zeros((N, 4), np.int64)
        # final: compare last weights with the best snapshot on the mini set, keep the better
        ev_last = run_eval(N, 777)
        metrics["mini_evals"].append({"iter": it, "winrate": ev_last["winrate"], "final": True})
        if best_sd is not None and best_wr > ev_last["winrate"]:
            net.load_state_dict(best_sd)
        torch.save(net.state_dict(), out / "policy.pt")
        json.dump(metrics, open(out / "metrics.json", "w"))
        ev = run_eval(a.eval_games, 10 ** 6)
        ev["seed"] = a.seed
        json.dump(ev, open(out / "eval.json", "w"), ensure_ascii=False)
        print("eval", {k: ev[k] for k in ("games", "winrate", "wins", "losses", "draws", "faults")}, flush=True)


if __name__ == "__main__":
    main()
