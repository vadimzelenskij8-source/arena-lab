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
