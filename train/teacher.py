"""Search teacher: refines the scripted policy's plays with simulator lookahead.

For every battle where the scripted policy wants to play a card, candidates (the same card on
nearby tiles and the mirrored lane, the other hand cards on the same tile, and waiting) are
looked ahead `horizon` ticks; the best by material + towers + elixir replaces the scripted play
when it is better by `margin` elixir.
"""
import numpy as np


class Teacher:
    def __init__(self, vec, horizon=100, margin=1.0, spread=2, w_elixir=1.0):
        s = vec.schema
        self.vec, self.s = vec, s
        h, self.e = s.header, s.entity
        self.per = s.cols * s.rows
        self.tw = [h[f"{t}{k}"][0] for t in ("own_king", "own_left", "own_right", "enemy_king", "enemy_left", "enemy_right")
                   for k in ("_hp", "_max_hp")]
        self.EL, self.CNT = h["elixir"][0], h["entity_count"][0]
        self.H, self.EI, self.ME = s.header_ints, s.entity_ints, s.max_entities
        self.fault = s.info["fault"]
        self.horizon, self.margin, self.spread, self.we = horizon, margin, spread, w_elixir
        self.stats = {"searched": 0, "changed": 0}

    def value(self, O):
        e = self.e
        t = O[:, self.tw].astype(np.float64).reshape(-1, 6, 2)
        towers = (t[:, :3, 0].sum(1) - t[:, 3:, 0].sum(1)) / 300.0
        rows = O[:, self.H:self.H + self.ME * self.EI].reshape(len(O), self.ME, self.EI)
        n = np.minimum(O[:, self.CNT], self.ME)
        valid = np.arange(self.ME)[None] < n[:, None]
        ch = valid & (rows[:, :, e["kind"]] == 1)
        hp, mh = rows[:, :, e["hp"]].astype(np.float64), rows[:, :, e["max_hp"]].astype(np.float64)
        frac = np.where(hp >= 0, hp / np.maximum(1, mh), 1.0)
        val = np.minimum(8.0, np.maximum(0, mh) / 350.0) * frac
        side = rows[:, :, e["side"]]
        units = (val * (ch & (side == 0))).sum(1) - (val * (ch & (side == 1))).sum(1)
        return towers + units + self.we * O[:, self.EL] / 10000.0

    def candidates(self, act, mask):
        s, per = self.s, self.per
        sl, tile = divmod(act - 1, per)
        r, c = divmod(tile, s.cols)
        cand = {act, 0}
        d = self.spread
        for dr in (-d, 0, d):
            for dc in (-d, 0, d):
                rr, cc = r + dr, c + dc
                if 0 <= rr < s.rows and 0 <= cc < s.cols:
                    cand.add(1 + sl * per + rr * s.cols + cc)
        cand.add(1 + sl * per + r * s.cols + (s.cols - 1 - c))
        for o in range(4):
            cand.add(1 + o * per + tile)
        return [x for x in cand if mask[x]]

    def refine(self, proposals, seats, active):
        """proposals: [N] scripted actions for the seat; returns the teacher's actions."""
        vec = self.vec
        out = np.array(proposals, dtype=np.int64)
        req_i, req_a, owner = [], [], []
        for i in range(len(out)):
            a = int(out[i])
            if active[i] and 0 < a < 1 + 4 * self.per:
                cs = self.candidates(a, vec.masks[i, seats[i]])
                if len(cs) > 1:
                    owner.append((i, len(req_a), cs))
                    req_i += [i] * len(cs); req_a += cs
        if not req_i:
            return out
        obs, info = vec.lookahead(np.array(req_i), np.asarray(seats)[np.array(req_i)], np.array(req_a), self.horizon)
        v = self.value(obs)
        v[info[:, self.fault] != 0] = -1e9
        for i, st, cs in owner:
            vv = v[st:st + len(cs)]
            base = vv[cs.index(int(out[i]))]
            j = int(np.argmax(vv))
            self.stats["searched"] += 1
            if vv[j] > base + self.margin:
                out[i] = cs[j]; self.stats["changed"] += 1
        return out
