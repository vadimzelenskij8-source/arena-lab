/*
 * arena-lab: the search of train/teacher.py (Teacher.refine), for the phone's digital twin.
 * Built on crforge (https://github.com/voonhous/crforge, Apache-2.0), commit bff1988.
 */

package org.crforge.env;

import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.Set;
import java.util.stream.IntStream;

/**
 * Picks a move by looking ahead: the candidates around the network's move and its top plays (nearby
 * tiles, the mirrored lane, the same tile with the other hand slots, the wait) are each played in a
 * fork of the battle as the seat knows it, then both seats wait for the horizon; the move whose end
 * position scores best wins, if it beats the network's move by the margin. Exactly
 * train/teacher.py's Teacher.refine and Teacher.value, so the phone plays what the exams measured.
 */
public final class PhoneSearch {

  private final int horizon;
  private final int spread;
  private final double margin;
  private final double elixirWeight;

  /** The last search: candidates and their scores, for the overlay. */
  public int[] lastCandidates = new int[0];

  public double[] lastScores = new double[0];

  /** Forks that failed, and the first failure, for diagnosis on the phone. */
  public volatile int failures = 0;

  public volatile String firstError = null;

  public volatile int forks = 0;

  public PhoneSearch(int horizon, int spread, double margin, double elixirWeight) {
    this.horizon = horizon;
    this.spread = spread;
    this.margin = margin;
    this.elixirWeight = elixirWeight;
  }

  /** Teacher.value: towers, characters' worth by hit points, the seat's elixir. */
  public double value(int[] o, int at) {
    double towers = 0;
    for (int k = 0; k < 3; k++) {
      towers += o[at + Schema.H_TOWERS + 2 * k];
      towers -= o[at + Schema.H_TOWERS + 6 + 2 * k];
    }
    towers /= 300.0;
    int n = Math.min(o[at + Schema.H_ENTITY_COUNT], Schema.MAX_ENTITIES);
    double units = 0;
    for (int i = 0; i < n; i++) {
      int r = at + Schema.HEADER_INTS + i * Schema.ENTITY_INTS;
      if (o[r + Schema.E_KIND] != Schema.KIND_CHARACTER) {
        continue;
      }
      double hp = o[r + Schema.E_HP];
      double mh = o[r + Schema.E_MAX_HP];
      double frac = hp >= 0 ? hp / Math.max(1, mh) : 1.0;
      double val = Math.min(8.0, Math.max(0, mh) / 350.0) * frac;
      int side = o[r + Schema.E_SIDE];
      if (side == 0) {
        units += val;
      } else if (side == 1) {
        units -= val;
      }
    }
    return towers + units + elixirWeight * o[at + Schema.H_ELIXIR] / 10000.0;
  }

  /** Teacher.candidates. */
  public void candidates(int act, byte[] mask, Set<Integer> into) {
    int per = Schema.TILES;
    int sl = (act - 1) / per;
    int tile = (act - 1) % per;
    int r = tile / Schema.COLS;
    int c = tile % Schema.COLS;
    int[] add = new int[16];
    int k = 0;
    add[k++] = act;
    add[k++] = 0;
    int d = spread;
    for (int dr = -d; dr <= d; dr += d) {
      for (int dc = -d; dc <= d; dc += d) {
        int rr = r + dr;
        int cc = c + dc;
        if (rr >= 0 && rr < Schema.ROWS && cc >= 0 && cc < Schema.COLS) {
          add[k++] = 1 + sl * per + rr * Schema.COLS + cc;
        }
        if (d == 0) {
          break;
        }
      }
      if (d == 0) {
        break;
      }
    }
    add[k++] = 1 + sl * per + r * Schema.COLS + (Schema.COLS - 1 - c);
    for (int o = 0; o < Schema.HAND_SLOTS; o++) {
      add[k++] = 1 + o * per + tile;
    }
    for (int j = 0; j < k; j++) {
      if (mask[add[j]] != 0) {
        into.add(add[j]);
      }
    }
  }

  private static boolean isPlay(int a) {
    return a > 0 && a < 1 + Schema.HAND_SLOTS * Schema.TILES;
  }

  /**
   * Teacher.refine for one battle: the seat's move after the search.
   *
   * @param source the battle at a decision (not changed)
   * @param seat the seat
   * @param proposal the network's move
   * @param extra the network's other top plays
   * @param mask the seat's mask
   */
  public int refine(LadderEnv source, int seat, int proposal, int[] extra, byte[] mask) {
    Set<Integer> cs = new LinkedHashSet<>();
    if (isPlay(proposal)) {
      candidates(proposal, mask, cs);
    }
    for (int x : extra) {
      if (isPlay(x) && mask[x] != 0) {
        candidates(x, mask, cs);
      }
    }
    if (cs.size() <= 1) {
      lastCandidates = new int[0];
      lastScores = new double[0];
      return proposal;
    }
    cs.add(0);
    cs.add(proposal);
    int[] cand = cs.stream().filter(a -> a >= 0 && a < Schema.ACTIONS && mask[a] != 0).mapToInt(Integer::intValue).sorted().toArray();
    double[] v = new double[cand.length];
    IntStream.range(0, cand.length)
        .parallel()
        .forEach(
            j -> {
              try {
                LadderEnv fork = source.fork(seat);
                int until = fork.tick() + horizon;
                int a = cand[j];
                fork.step(seat == 0 ? a : Schema.WAIT, seat == 0 ? Schema.WAIT : a);
                while (!fork.done() && fork.tick() < until) {
                  fork.step(Schema.WAIT, Schema.WAIT);
                }
                int[] obs = new int[Schema.SEAT_INTS];
                fork.copyObservation(seat, obs, 0);
                int[] info = new int[Schema.INFO_INTS];
                fork.copyInfo(info, 0);
                v[j] = info[Schema.I_FAULT] != 0 ? -1e9 : value(obs, 0);
              } catch (RuntimeException | AssertionError | StackOverflowError e) {
                v[j] = -1e9;
                failures++;
                if (firstError == null) {
                  java.io.StringWriter w = new java.io.StringWriter();
                  e.printStackTrace(new java.io.PrintWriter(w));
                  String t = w.toString();
                  firstError = t.length() > 3000 ? t.substring(0, 3000) : t;
                }
              }
              forks++;
            });
    lastCandidates = cand;
    lastScores = v;
    int bi = Arrays.binarySearch(cand, proposal);
    double base = bi >= 0 ? v[bi] : -1e9;
    int best = 0;
    for (int j = 1; j < v.length; j++) {
      if (v[j] > v[best]) {
        best = j;
      }
    }
    return v[best] > base + margin ? cand[best] : proposal;
  }

  /** The twin's battle. */
  public int refine(LiveBattle live, int seat, int proposal, int[] extra) {
    return refine(live.ladder(), seat, proposal, extra, live.mask(seat));
  }

  /** For the cross-check against train/teacher.py: the same search on a vector's battles. */
  public int[] refineVector(VectorEnv vec, int[] indices, int[] seats, int[] proposals, int[] extraFlat, int extraPer) {
    int[] out = new int[indices.length];
    for (int r = 0; r < indices.length; r++) {
      int[] ex = Arrays.copyOfRange(extraFlat, r * extraPer, (r + 1) * extraPer);
      LadderEnv env = vec.ladderFor(indices[r]);
      byte[] mask = new byte[Schema.ACTIONS];
      env.copyMask(seats[r], mask, 0);
      out[r] = env.done() ? proposals[r] : refine(env, seats[r], proposals[r], ex, mask);
    }
    return out;
  }
}
