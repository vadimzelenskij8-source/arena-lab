package org.crforge.env;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A Java port of crforge_env.scripted.ScriptedBot (the same rules, the same order), so playouts
 * can run inside the simulator's threads. Layout offsets, card facts and parameters are given by
 * the Python side (arena-lab train/jbot.py), so the class depends on nothing but the arrays.
 */
public final class ScriptedPolicy {
  static final int PENDING = 0, CNT = 1, EL = 2, RES = 3, E_LEFT = 4, E_RIGHT = 5, E_KING = 6, HAND_CARD = 7,
      HAND_COST = 11, H = 15, EI = 16, ME = 17, C_KIND = 18, C_SIDE = 19, C_FLAGS = 20, C_X = 21, C_Y = 22,
      C_HP = 23, C_SHIELD = 24, C_MAXHP = 25, C_ROW = 26, COLS = 27, ROWS = 28, TILE = 29, WIDTH = 30,
      HEIGHT = 31, UNKNOWN = 32, WAIT = 33, K_CHAR = 34, K_TOWER = 35, F_ALIVE = 36, F_HIDDEN = 37,
      F_UNDER = 38, F_AIR = 39, F_BUILDING = 40, ATTACK_EL = 41, LANE_L = 42, LANE_R = 43, BRIDGE = 44,
      DEFEND = 45, BCOL_L = 46, BCOL_R = 47, BROW = 48, BACK = 49, SPELL_TARGETS = 50, SPELL_WORTH_M = 51,
      AREA_TARGETS = 52, SUPPORT_EL = 53, SUPPORT_GAP = 54, SUPPORT_RADIUS = 55;

  private final int[] L;
  private final int[] type; // 0 none, 1 troop, 2 building, 3 spell
  private final int[] spDamage, spTower, spRadius, spRoll, spMax;
  private final boolean[] spHas, spAir, spGround, attack, tank, attackUnit;

  public ScriptedPolicy(int[] layout, int[] type, boolean[] spHas, int[] spDamage, int[] spTower, int[] spRadius,
      int[] spRoll, boolean[] spAir, boolean[] spGround, int[] spMax, boolean[] attack, boolean[] tank,
      boolean[] attackUnit) {
    this.L = layout.clone();
    this.type = type; this.spHas = spHas; this.spDamage = spDamage; this.spTower = spTower;
    this.spRadius = spRadius; this.spRoll = spRoll; this.spAir = spAir; this.spGround = spGround;
    this.spMax = spMax; this.attack = attack; this.tank = tank; this.attackUnit = attackUnit;
  }

  private boolean in(boolean[] a, int i) { return i >= 0 && i < a.length && a[i]; }
  private int typeOf(int card) { return card >= 0 && card < type.length ? type[card] : 0; }
  private int col(int[] o, int at, int e, int c) { return o[at + L[H] + e * L[EI] + L[c]]; }
  private int tiles() { return L[COLS] * L[ROWS]; }
  private int play(int slot, int c, int r) { return 1 + slot * tiles() + r * L[COLS] + c; }
  private boolean slotOpen(byte[] m, int ma, int slot) {
    int first = ma + 1 + slot * tiles();
    for (int k = 0; k < tiles(); k++) if (m[first + k] != 0) return true;
    return false;
  }
  private int clampCol(int x) { return Math.min(Math.max(x / L[TILE], 0), L[COLS] - 1); }
  private int clampRow(int y) { return Math.min(Math.max(y / L[TILE], 0), L[ROWS] - 1); }

  /** The scripted action for a seat's observation {@code o} at {@code oa} and mask {@code m} at {@code ma}. */
  public int act(int[] o, int oa, byte[] m, int ma, SplittableRandom rng) {
    int n = Math.min(o[oa + L[CNT]], L[ME]);
    boolean waiting = o[oa + L[PENDING]] > 0;
    if (!waiting) {
      List<Integer> enemy = new ArrayList<>();
      for (int e = 0; e < n; e++) {
        int fl = col(o, oa, e, C_FLAGS);
        if (col(o, oa, e, C_KIND) == L[K_CHAR] && col(o, oa, e, C_SIDE) == 1 && (fl & L[F_ALIVE]) != 0
            && (fl & (L[F_HIDDEN] | L[F_UNDER])) == 0) enemy.add(e);
      }
      // spells in hand, cheapest first (cost, slot)
      List<int[]> spells = new ArrayList<>();
      for (int s = 0; s < 4; s++) {
        int card = o[oa + L[HAND_CARD + s]];
        if (in(spHas, card) && typeOf(card) == 3) spells.add(new int[] {o[oa + L[HAND_COST + s]], s, card});
      }
      spells.sort((a, b) -> a[0] != b[0] ? Integer.compare(a[0], b[0]) : Integer.compare(a[1], b[1]));
      if (!spells.isEmpty()) {
        boolean[] kept = new boolean[4];
        int a = finish(o, oa, n, spells, m, ma, kept);
        if (a >= 0) return a;
        List<int[]> rest = new ArrayList<>();
        for (int[] sp : spells) if (!kept[sp[1]]) rest.add(sp);
        a = cast(o, oa, enemy, rest, m, ma);
        if (a >= 0) return a;
      }
      int threat = deepestThreat(o, oa, enemy);
      if (threat != Integer.MIN_VALUE) {
        int lane = threat < L[WIDTH] / 2 ? 0 : 1;
        int[] tile = {lane == 0 ? L[LANE_L] : L[LANE_R], L[DEFEND]};
        int[] btile = {lane == 0 ? L[BCOL_L] : L[BCOL_R], L[BROW]};
        return playCheapest(o, oa, m, ma, new int[] {1, 2}, tile, btile, true);
      }
      int sup = support(o, oa, n, m, ma);
      if (sup >= 0) return sup;
    }
    int elixir = o[oa + L[EL]] - o[oa + L[RES]];
    if (elixir >= L[ATTACK_EL]) {
      int left = o[oa + L[E_LEFT]], right = o[oa + L[E_RIGHT]];
      int lane = left == right ? rng.nextInt(2) : (left < right ? 0 : 1);
      int laneCol = lane == 0 ? L[LANE_L] : L[LANE_R];
      int slot = pick(o, oa, m, ma, new int[] {1, 2}, true, false, false);
      if (slot >= 0) {
        int card = o[oa + L[HAND_CARD + slot]];
        int row = in(tank, card) ? L[BACK] : L[BRIDGE];
        return openTile(m, ma, slot, laneCol, row);
      }
      return playCheapest(o, oa, m, ma, new int[] {1}, new int[] {laneCol, L[BRIDGE]}, new int[] {laneCol, L[BRIDGE]}, false);
    }
    return L[WAIT];
  }

  private int finish(int[] o, int oa, int n, List<int[]> spells, byte[] m, int ma, boolean[] kept) {
    int reach = 0;
    for (int[] sp : spells) if (spRoll[sp[2]] == 0) reach = Math.max(reach, spTower[sp[2]]);
    int minStanding = Integer.MAX_VALUE;
    for (int k : new int[] {E_KING, E_LEFT, E_RIGHT}) { int hp = o[oa + L[k]]; if (hp > 0) minStanding = Math.min(minStanding, hp); }
    if (reach <= 0 || minStanding == Integer.MAX_VALUE || reach < minStanding) return -1;
    List<Integer> towers = new ArrayList<>();
    for (int e = 0; e < n; e++)
      if (col(o, oa, e, C_KIND) == L[K_TOWER] && col(o, oa, e, C_SIDE) == 1 && (col(o, oa, e, C_FLAGS) & L[F_ALIVE]) != 0
          && col(o, oa, e, C_HP) > 0) towers.add(e);
    towers.sort((a, b) -> Integer.compare(col(o, oa, a, C_HP), col(o, oa, b, C_HP)));
    for (int[] sp : spells) {
      int card = sp[2];
      if (spRoll[card] != 0 || spTower[card] <= 0) continue;
      for (int t : towers) {
        if (spTower[card] < col(o, oa, t, C_HP) + col(o, oa, t, C_SHIELD)) continue;
        kept[sp[1]] = true;
        int a = play(sp[1], clampCol(col(o, oa, t, C_X)), clampRow(col(o, oa, t, C_Y)));
        if (m[ma + a] != 0) return a;
      }
    }
    return -1;
  }

  private int cast(int[] o, int oa, List<Integer> enemy, List<int[]> spells, byte[] m, int ma) {
    int ne = enemy.size();
    if (spells.isEmpty() || ne < Math.min(L[SPELL_TARGETS], L[AREA_TARGETS])) return -1;
    double need = L[SPELL_WORTH_M] / 1000.0;
    long[] x = new long[ne], y = new long[ne];
    double[] health = new double[ne], whole = new double[ne];
    boolean[] air = new boolean[ne];
    for (int k = 0; k < ne; k++) {
      int e = enemy.get(k);
      x[k] = col(o, oa, e, C_X); y[k] = col(o, oa, e, C_Y);
      boolean shown = col(o, oa, e, C_HP) != L[UNKNOWN];
      int shield = shown ? col(o, oa, e, C_SHIELD) : 0;
      int hp = shown ? col(o, oa, e, C_HP) : col(o, oa, e, C_MAXHP);
      health[k] = hp + shield;
      whole[k] = Math.max(col(o, oa, e, C_MAXHP) + shield, 1);
      air[k] = (col(o, oa, e, C_FLAGS) & L[F_AIR]) != 0;
    }
    int tile = L[TILE];
    double bestMargin = 0; int bestAction = -1;
    for (int[] sp : spells) {
      int slot = sp[1], card = sp[2];
      if (!slotOpen(m, ma, slot)) continue;
      boolean[] hit = new boolean[ne]; int nHit = 0;
      double[] share = new double[ne]; double shareSum = 0;
      int dmg = spDamage[card];
      for (int k = 0; k < ne; k++) {
        hit[k] = air[k] ? spAir[card] : spGround[card];
        if (hit[k]) nHit++;
        if (dmg > 0) {
          double taken = Math.min(health[k], dmg);
          share[k] = taken / Math.max(whole[k], dmg);
          if (hit[k]) shareSum += share[k];
        }
      }
      if (dmg > 0) {
        if (nHit < L[SPELL_TARGETS] || shareSum < need) continue;
      } else if (nHit < L[AREA_TARGETS]) continue;
      java.util.TreeSet<Integer> cands = new java.util.TreeSet<>();
      for (int k = 0; k < ne; k++) if (hit[k]) cands.add(clampRow((int) y[k]) * L[COLS] + clampCol((int) x[k]));
      int rad = spRadius[card], roll = spRoll[card], maxT = spMax[card];
      for (int c : cands) {
        long cx = (long) (c % L[COLS]) * tile + tile / 2, cy = (long) (c / L[COLS]) * tile + tile / 2;
        int count = 0; List<Double> worths = new ArrayList<>();
        for (int k = 0; k < ne; k++) {
          if (!hit[k]) continue;
          long dx = x[k] - cx, dy = y[k] - cy;
          boolean cov = roll != 0 ? (Math.abs(dx) <= rad && dy >= -rad && dy <= roll) : (dx * dx + dy * dy <= (long) rad * rad);
          if (cov) { count++; worths.add(share[k]); }
        }
        boolean good; double margin;
        if (dmg > 0) {
          worths.sort((p, q) -> Double.compare(q, p));
          double worth = 0;
          int lim = maxT > 0 ? Math.min(maxT, worths.size()) : worths.size();
          for (int k = 0; k < lim; k++) worth += worths.get(k);
          if (maxT > 0) count = Math.min(count, maxT);
          good = count >= L[SPELL_TARGETS] && worth >= need;
          margin = worth - need;
        } else {
          good = count >= L[AREA_TARGETS];
          margin = count - L[AREA_TARGETS];
        }
        if (!good) continue;
        int a = play(slot, c % L[COLS], c / L[COLS]);
        if (m[ma + a] == 0) continue;
        if (bestAction < 0 || margin > bestMargin) { bestMargin = margin; bestAction = a; }
      }
    }
    return bestAction;
  }

  private int deepestThreat(int[] o, int oa, List<Integer> enemy) {
    if (enemy.isEmpty()) return Integer.MIN_VALUE;
    int best = -1, by = Integer.MAX_VALUE;
    for (int e : enemy) { int yy = col(o, oa, e, C_Y); if (yy < by) { by = yy; best = e; } }
    if (by >= L[HEIGHT] / 2) return Integer.MIN_VALUE;
    return col(o, oa, best, C_X);
  }

  private int support(int[] o, int oa, int n, byte[] m, int ma) {
    int elixir = o[oa + L[EL]] - o[oa + L[RES]];
    if (elixir < L[SUPPORT_EL]) return -1;
    int lead = -1, ly = Integer.MIN_VALUE;
    List<Integer> own = new ArrayList<>();
    for (int e = 0; e < n; e++)
      if (col(o, oa, e, C_KIND) == L[K_CHAR] && col(o, oa, e, C_SIDE) == 0 && (col(o, oa, e, C_FLAGS) & L[F_ALIVE]) != 0) own.add(e);
    for (int e : own) {
      int row = col(o, oa, e, C_ROW);
      if (in(attackUnit, row) && col(o, oa, e, C_Y) > ly) { ly = col(o, oa, e, C_Y); lead = e; }
    }
    if (lead < 0) return -1;
    long x = col(o, oa, lead, C_X), y = col(o, oa, lead, C_Y);
    long r2 = (long) L[SUPPORT_RADIUS] * L[SUPPORT_RADIUS];
    for (int e : own) {
      if (in(attackUnit, col(o, oa, e, C_ROW)) || (col(o, oa, e, C_FLAGS) & L[F_BUILDING]) != 0) continue;
      long dx = col(o, oa, e, C_X) - x, dy = col(o, oa, e, C_Y) - y;
      if (dx * dx + dy * dy <= r2) return -1;
    }
    int slot = pick(o, oa, m, ma, new int[] {1}, false, true, true);
    if (slot < 0) return -1;
    int c = clampCol((int) x), r = clampRow((int) y);
    return openTile(m, ma, slot, c, Math.max(Math.min(r - L[SUPPORT_GAP], L[BRIDGE]), 0));
  }

  /** onlyAttack: only win conditions; excludeAttack: never win conditions. */
  private int pick(int[] o, int oa, byte[] m, int ma, int[] types, boolean onlyAttack, boolean excludeAttack, boolean dearest) {
    int best = -1, bestKey = 0;
    for (int s = 0; s < 4; s++) {
      int card = o[oa + L[HAND_CARD + s]];
      int t = typeOf(card);
      boolean ok = false;
      for (int tt : types) ok |= tt == t;
      if (!ok) continue;
      if (excludeAttack && in(attack, card)) continue;
      if (onlyAttack && !in(attack, card)) continue;
      if (!slotOpen(m, ma, s)) continue;
      int cost = o[oa + L[HAND_COST + s]];
      int key = dearest ? -cost : cost;
      if (best < 0 || key < bestKey) { best = s; bestKey = key; }
    }
    return best;
  }

  private int playCheapest(int[] o, int oa, byte[] m, int ma, int[] types, int[] tile, int[] btile, boolean excludeAttack) {
    int slot = pick(o, oa, m, ma, types, false, excludeAttack, false);
    if (slot < 0) return L[WAIT];
    int card = o[oa + L[HAND_CARD + slot]];
    int[] t = typeOf(card) == 2 ? btile : tile;
    return openTile(m, ma, slot, t[0], t[1]);
  }

  private int openTile(byte[] m, int ma, int slot, int c, int r) {
    int a = play(slot, c, r);
    if (m[ma + a] != 0) return a;
    int first = 1 + slot * tiles();
    int best = -1; long bd = Long.MAX_VALUE;
    for (int k = 0; k < tiles(); k++) {
      if (m[ma + first + k] == 0) continue;
      int rr = k / L[COLS], cc = k % L[COLS];
      long d = (long) (cc - c) * (cc - c) + (long) (rr - r) * (rr - r);
      if (d < bd) { bd = d; best = first + k; }
    }
    return best < 0 ? L[WAIT] : best;
  }
}
