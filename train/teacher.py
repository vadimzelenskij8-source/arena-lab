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


class ForkTeacher(Teacher):
    """Two-stage teacher (generation 5).

    1) lookahead screens many candidates (scripted proposal, the network's Gumbel-top-k plays,
       nearby tiles, other cards, wait) with both sides waiting;
    2) the best few, plus delayed versions of the best (wait k decisions, then play), are played
       out in forks of the battle as the seat knows it (the enemy's unseen hand re-dealt) where
       BOTH sides keep playing (scripted policies), so the enemy answers the move.
    Returns the chosen action and a soft target over the evaluated candidates.
    """

    def __init__(self, vec, scratch, bots_scratch, horizon=160, fork_steps=24, top=3, delays=(2, 4), temp=1.0,
                 margin=0.3, rng=None, **kw):
        super().__init__(vec, horizon=horizon, margin=margin, **kw)
        self.sc, self.sb = scratch, bots_scratch
        self.S = scratch.count
        self.fork_steps, self.top, self.delays, self.temp = fork_steps, top, delays, temp
        self.rng = rng or np.random.default_rng(0)
        self.win_i = vec.schema.info["winner"]
        self.stats.update({"forks": 0, "delayed": 0})

    def refine2(self, proposals, seats, active, extra):
        """extra: per env list of additional candidate actions (network proposals)."""
        vec = self.vec
        N = len(proposals)
        out = np.array(proposals, dtype=np.int64)
        soft = [None] * N
        req_i, req_a, owner = [], [], []
        for i in range(N):
            if not active[i]:
                continue
            a = int(out[i])
            mask = vec.masks[i, seats[i]]
            cs = set()
            if 0 < a < 1 + 4 * self.per:
                cs.update(self.candidates(a, mask))
            for x in extra[i]:
                x = int(x)
                if 0 < x < 1 + 4 * self.per and mask[x]:
                    cs.update(self.candidates(x, mask))
            if len(cs) <= 1:
                continue
            cs.add(0); cs.add(a)
            cs = sorted(c for c in cs if mask[c])
            owner.append((i, len(req_a), cs))
            req_i += [i] * len(cs); req_a += cs
        if not req_i:
            return out, soft
        obs, info = vec.lookahead(np.array(req_i), np.asarray(seats)[np.array(req_i)], np.array(req_a), self.horizon)
        v = self.value(obs)
        v[info[:, self.fault] != 0] = -1e9
        # stage 2: fork playouts of the best candidates
        plans = []  # (env, plan_index, first_action, delay)
        per_env = {}
        for i, st, cs in owner:
            vv = v[st:st + len(cs)]
            order = [cs[j] for j in np.argsort(-vv)]
            pick = [c for c in order if c != 0][:self.top]
            base = int(out[i])
            if base not in pick:
                pick.append(base)
            if 0 not in pick:
                pick.append(0)
            cand = [(c, 0) for c in pick]
            best_play = next((c for c in order if c != 0), None)
            if best_play is not None:
                cand += [(best_play, d) for d in self.delays]
            per_env[i] = cand
            plans += [(i, c, d) for c, d in cand]
            self.stats["searched"] += 1
        vals = np.full(len(plans), -1e9)
        for c0 in range(0, len(plans), self.S):
            chunk = plans[c0:c0 + self.S]
            self._playout(chunk, seats, vals, c0)
        k = 0
        for i, _st, _cs in owner:
            cand = per_env[i]
            vv = vals[k:k + len(cand)]; k += len(cand)
            base_idx = next(j for j, (c, d) in enumerate(cand) if c == int(out[i]) and d == 0)
            j = int(np.argmax(vv))
            if vv[j] > vv[base_idx] + self.margin:
                c, d = cand[j]
                out[i] = c if d == 0 else 0
                self.stats["changed"] += 1
                self.stats["delayed"] += d > 0
            # soft target over immediate actions (a delayed play counts as "wait now")
            imm = {}
            for (c, d), val in zip(cand, vv):
                key = c if d == 0 else 0
                imm[key] = max(imm.get(key, -1e9), val)
            ks = list(imm); vs = np.array([imm[q] for q in ks])
            p = np.exp((vs - vs.max()) / self.temp); p /= p.sum()
            soft[i] = (ks, p)
        return out, soft

    def _playout(self, chunk, seats, vals, offset):
        sc, vec = self.sc, self.vec
        n = len(chunk)
        targets = list(range(n))
        by_env = {}
        for t, (i, _c, _d) in enumerate(chunk):
            by_env.setdefault(i, []).append(t)
        for i, ts in by_env.items():
            sc.fork_from(vec, i, int(seats[i]), ts, shuffles=[int(self.rng.integers(1 << 30)) for _ in ts],
                         seeds=[int(self.rng.integers(1 << 30)) for _ in ts])
        self.stats["forks"] += n
        done = np.zeros(n, bool)
        res = np.zeros(n)
        for step in range(self.fork_steps):
            acts = np.zeros((self.S, 2), np.int32)
            for t, (i, c, d) in enumerate(chunk):
                if done[t]:
                    continue
                me = int(seats[i])
                if step < d:
                    acts[t, me] = 0
                elif step == d:
                    acts[t, me] = c if sc.masks[t, me][c] else 0
                else:
                    acts[t, me] = self.sb[me](sc.observations[t, me], sc.masks[t, me], self.rng)
                acts[t, 1 - me] = self.sb[1 - me](sc.observations[t, 1 - me], sc.masks[t, 1 - me], self.rng)
            sc.step(acts)
            for t, (i, _c, _d) in enumerate(chunk):
                if not done[t] and sc.dones[t]:
                    done[t] = True
                    w = int(sc.infos[t, self.win_i])
                    res[t] = 50.0 if w == int(seats[i]) else (-50.0 if w in (0, 1) else 0.0)
        O = np.stack([sc.observations[t, int(seats[chunk[t][0]])] for t in range(n)])
        v = self.value(O)
        vals[offset:offset + n] = np.where(done, res, v)
