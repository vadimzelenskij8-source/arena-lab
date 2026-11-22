"""Java scripted policy and the fast playout teacher.

make_policy() hands a bound ScriptedBot's rules to org.crforge.env.ScriptedPolicy (sim_patch/),
so playouts where both sides keep playing run inside the simulator's threads
(VectorEnv.playout) instead of calling the Python bot thousands of times.
"""
import jpype
import numpy as np

from teacher import Teacher


def make_policy(vec, bot):
    s, h, e = vec.schema, vec.schema.header, vec.schema.entity
    L = [0] * 56
    L[0], L[1], L[2], L[3] = h["pending_count"][0], h["entity_count"][0], h["elixir"][0], h["elixir_reserved"][0]
    L[4], L[5], L[6] = h["enemy_left_hp"][0], h["enemy_right_hp"][0], h["enemy_king_hp"][0]
    L[7:11] = list(h["hand_card"]); L[11:15] = list(h["hand_cost"])
    L[15], L[16], L[17] = s.header_ints, s.entity_ints, s.max_entities
    for k, n in zip(range(18, 27), ["kind", "side", "flags", "x", "y", "hp", "shield", "max_hp", "row"]):
        L[k] = e[n]
    L[27], L[28], L[29], L[30], L[31], L[32], L[33] = s.cols, s.rows, s.tile_units, s.width_units, s.height_units, s.unknown, s.wait
    L[34], L[35] = s.kinds["character"], s.kinds["tower"]
    for k, n in zip(range(36, 41), ["alive", "hidden", "underground", "air", "building"]):
        L[k] = s.flags[n]
    L[41] = bot.attack_elixir; L[42], L[43] = bot.lane_cols; L[44], L[45] = bot.bridge_row, bot.defend_row
    L[46], L[47] = bot.building_cols; L[48], L[49] = bot.building_row, bot.back_row
    L[50], L[51], L[52] = bot.spell_targets, int(round(bot.spell_worth * 1000)), bot.area_targets
    L[53], L[54], L[55] = bot.support_elixir, bot.support_gap, bot.support_radius
    V = max([vec.vocabulary_size, *bot.card_types.keys()]) + 2
    code = {"troop": 1, "building": 2, "spell": 3}
    typ = [0] * V
    for c, t in bot.card_types.items():
        typ[c] = code.get(t, 0)
    has, dmg, tw, rad, roll, air, gnd, mx = ([False] * V, [0] * V, [0] * V, [0] * V, [0] * V, [False] * V, [False] * V, [0] * V)
    for c, f in bot._spells.items():
        has[c] = True; dmg[c] = int(f.damage); tw[c] = int(f.tower_damage); rad[c] = int(f.radius); roll[c] = int(f.roll)
        air[c] = bool(f.hits_air); gnd[c] = bool(f.hits_ground); mx[c] = int(f.max_targets)
    att = [False] * V; tank = [False] * V
    for c in bot.attack_cards:
        att[c] = True
    for c in bot.tank_cards:
        tank[c] = True
    au = [bool(x) for x in bot._attack_unit.tolist()]
    JI, JB = jpype.JArray(jpype.JInt), jpype.JArray(jpype.JBoolean)
    P = jpype.JClass("org.crforge.env.ScriptedPolicy")
    return P(JI(L), JI(typ), JB(has), JI(dmg), JI(tw), JI(rad), JI(roll), JB(air), JB(gnd), JI(mx), JB(att), JB(tank), JB(au))


def java_act(vec, policy, obs, mask, seed=1):
    """One action of the Java policy for a seat observation and mask (for the equivalence check)."""
    R = jpype.JClass("java.util.SplittableRandom")(seed)
    o = jpype.JArray(jpype.JInt)(np.asarray(obs, dtype=np.int32))
    m = jpype.JArray(jpype.JByte)(np.asarray(mask, dtype=np.int8))
    return int(policy.act(o, 0, m, 0, R))


def scripted_batch(vec, policy, seats, seed=0, indices=None):
    """Java scripted actions for env i at seat seats[i] (all envs, or the given indices)."""
    idx = np.arange(vec.count) if indices is None else np.asarray(indices)
    if len(idx) == 0:
        return np.zeros(0, np.int64)
    JI = jpype.JArray(jpype.JInt)
    out = vec._env.scriptedActs(JI(idx.astype(np.int32)), JI(np.asarray(seats, np.int32)[idx] if indices is None else np.asarray(seats, np.int32)),
                                int(seed), policy)
    return np.asarray(out, dtype=np.int64)


class PlayoutTeacher(Teacher):
    """Lookahead screens candidates (scripted proposal, the network's Gumbel-top-k plays, nearby
    tiles, other cards, wait); the best few plus delayed versions of the best are then played out
    with BOTH sides following the Java scripted policy (the enemy answers), the enemy's unseen
    hand re-dealt per playout. Each candidate gets `reps` playouts; the value is their mean."""

    def __init__(self, vec, policy, horizon=160, play_ticks=160, top=4, delays=(2, 4), reps=2, temp=1.0, margin=0.3,
                 rng=None, **kw):
        super().__init__(vec, horizon=horizon, margin=margin, **kw)
        self.policy = policy
        self.play_ticks, self.top, self.delays, self.reps, self.temp = play_ticks, top, delays, reps, temp
        self.rng = rng or np.random.default_rng(0)
        self.win_i = vec.schema.info["winner"]
        self.stats.update({"playouts": 0, "delayed": 0})
        self.value_fn, self.vw = None, 0.0  # optional: the network's value of the playout's end, mixed in

    def playout(self, idx, seats, acts, delays):
        s = self.s
        JI, JL = jpype.JArray(jpype.JInt), jpype.JArray(jpype.JLong)
        n = len(idx)
        sh = self.rng.integers(1, 1 << 40, n); sd = self.rng.integers(1, 1 << 40, n)
        out = self.vec._env.playout(JI(np.asarray(idx, np.int32)), JI(np.asarray(seats, np.int32)), JI(np.asarray(acts, np.int32)),
                                    JI(np.asarray(delays, np.int32)), int(self.play_ticks), JL([int(x) for x in sh]),
                                    JL([int(x) for x in sd]), self.policy)
        flat = np.asarray(out, dtype=np.int32).reshape(n, s.seat_ints + s.info_ints)
        obs, info = flat[:, :s.seat_ints], flat[:, s.seat_ints:]
        v = self.value(obs)
        if self.value_fn is not None and self.vw > 0:
            v = v + self.vw * 20.0 * self.value_fn(obs)
        w = info[:, self.win_i]
        seats = np.asarray(seats)
        ended = (w == 0) | (w == 1)
        v = np.where(ended, np.where(w == seats, 50.0, -50.0), v)
        v[info[:, self.fault] != 0] = -1e9
        self.stats["playouts"] += n
        return v

    def refine2(self, proposals, seats, active, extra):
        vec = self.vec
        N = len(proposals)
        out = np.array(proposals, dtype=np.int64)
        soft = [None] * N
        req_i, req_a, owner = [], [], []
        for i in range(N):
            if not active[i]:
                continue
            a = int(out[i]); mask = vec.masks[i, seats[i]]
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
            owner.append((i, len(req_a), cs)); req_i += [i] * len(cs); req_a += cs
        if not req_i:
            return out, soft
        obs, info = vec.lookahead(np.array(req_i), np.asarray(seats)[np.array(req_i)], np.array(req_a), self.horizon)
        v = self.value(obs); v[info[:, self.fault] != 0] = -1e9
        P_i, P_s, P_a, P_d, plan = [], [], [], [], []
        for i, st, cs in owner:
            vv = v[st:st + len(cs)]
            order = [cs[j] for j in np.argsort(-vv)]
            pick = [c for c in order if c != 0][:self.top]
            base = int(out[i])
            for c in (base, 0):
                if c not in pick:
                    pick.append(c)
            cand = [(c, 0) for c in pick]
            bp = next((c for c in order if c != 0), None)
            if bp is not None:
                cand += [(bp, d) for d in self.delays]
            plan.append((i, cand))
            for c, d in cand:
                for _ in range(self.reps):
                    P_i.append(i); P_s.append(int(seats[i])); P_a.append(c); P_d.append(d)
            self.stats["searched"] += 1
        pv = self.playout(P_i, P_s, P_a, P_d)
        k = 0
        for i, cand in plan:
            vv = pv[k:k + len(cand) * self.reps].reshape(len(cand), self.reps).mean(1); k += len(cand) * self.reps
            base_idx = next(j for j, (c, d) in enumerate(cand) if c == int(out[i]) and d == 0)
            j = int(np.argmax(vv))
            if vv[j] > vv[base_idx] + self.margin:
                c, d = cand[j]
                out[i] = c if d == 0 else 0
                self.stats["changed"] += 1; self.stats["delayed"] += d > 0
            imm = {}
            for (c, d), val in zip(cand, vv):
                key = c if d == 0 else 0
                imm[key] = max(imm.get(key, -1e9), val)
            ks = list(imm); vs = np.array([imm[q] for q in ks])
            p = np.exp((vs - vs.max()) / self.temp); p /= p.sum()
            soft[i] = (ks, p)
        return out, soft


class ReplyTeacher(Teacher):
    """Search with the enemy's answer: every candidate move is played in a fork of the battle as the
    seat knows it (the enemy's unseen hand re-dealt), the enemy answers ~reply_ticks later with
    `reply_fn` (our network from the enemy's side), then both wait until the horizon. The same
    candidates and choice rule as Teacher.refine, so it is a drop-in for the search exams."""

    def __init__(self, vec, reply_ticks=10, rng=None, **kw):
        super().__init__(vec, **kw)
        self.reply_ticks = reply_ticks
        self.rng = rng or np.random.default_rng(0)
        self.win_i = vec.schema.info["winner"]
        self.reply_fn = None  # (enemy obs [n, seat_ints], enemy mask [n, actions] bool) -> actions [n]
        self.stats.update({"replies": 0, "reply_plays": 0, "t_reply": 0.0})

    def refine(self, proposals, seats, active, extra=None):
        import time
        vec, s = self.vec, self.s
        out = np.array(proposals, dtype=np.int64)
        self.soft = [None] * len(out)
        req_i, req_a, owner = [], [], []
        for i in range(len(out)):
            a = int(out[i]); mask = vec.masks[i, seats[i]]
            cs = set()
            if active[i] and 0 < a < 1 + 4 * self.per:
                cs.update(self.candidates(a, mask))
            if active[i] and extra is not None:
                for x in extra[i]:
                    x = int(x)
                    if 0 < x < 1 + 4 * self.per and mask[x]:
                        cs.update(self.candidates(x, mask))
            if len(cs) > 1:
                cs.add(0); cs.add(a)
                cs = sorted(c for c in cs if mask[c])
                owner.append((i, len(req_a), cs)); req_i += [i] * len(cs); req_a += cs
        if not req_i:
            return out
        t0 = time.time()
        n = len(req_i); A = vec.masks.shape[-1]
        rs = np.asarray(seats)[np.array(req_i)].astype(np.int32)
        JI, JL = jpype.JArray(jpype.JInt), jpype.JArray(jpype.JLong)
        sh = self.rng.integers(1, 1 << 40, n); sd = self.rng.integers(1, 1 << 40, n)
        raw = self.vec._env.replyStart(JI(np.asarray(req_i, np.int32)), JI(rs), JI(np.asarray(req_a, np.int32)),
                                       int(self.reply_ticks), int(self.horizon), JL([int(x) for x in sh]), JL([int(x) for x in sd]))
        flat = np.asarray(raw, dtype=np.int32).reshape(n, s.seat_ints + A + 1)
        Oe, Me, status = flat[:, :s.seat_ints], flat[:, s.seat_ints:s.seat_ints + A] != 0, flat[:, -1]
        reply = np.zeros(n, np.int32)
        run = np.where(status == 0)[0]
        if len(run) and self.reply_fn is not None:
            for j in range(0, len(run), 512):
                ix = run[j:j + 512]
                reply[ix] = np.asarray(self.reply_fn(Oe[ix], Me[ix]), np.int32)
        self.stats["replies"] += len(run); self.stats["reply_plays"] += int((reply > 0).sum())
        res = self.vec._env.replyFinish(JI(rs), JI(reply))
        flat = np.asarray(res, dtype=np.int32).reshape(n, s.seat_ints + s.info_ints)
        obs, info = flat[:, :s.seat_ints], flat[:, s.seat_ints:]
        v = self.value(obs)
        if self.value_fn is not None and self.vw > 0:
            v = v + self.vw * 20.0 * self.value_fn(obs)
        w = info[:, self.win_i]
        ended = (w == 0) | (w == 1)
        v = np.where(ended, np.where(w == rs, 50.0, -50.0), v)
        v[(info[:, self.fault] != 0) | (status == 2)] = -1e9
        self.stats["t_reply"] += time.time() - t0
        for i, st, cs in owner:
            vv = v[st:st + len(cs)]
            base = vv[cs.index(int(out[i]))]
            j = int(np.argmax(vv))
            self.stats["searched"] += 1
            if vv[j] > base + self.margin:
                out[i] = cs[j]; self.stats["changed"] += 1
            ok = vv > -1e8
            if ok.sum() > 1:
                ks = [c for c, o in zip(cs, ok) if o]; vs = vv[ok]
                p = np.exp((vs - vs.max()) / 1.0); p /= p.sum()
                self.soft[i] = (ks, p)
        return out
