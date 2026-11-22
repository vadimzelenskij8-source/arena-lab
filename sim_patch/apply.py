#!/usr/bin/env python3
"""Adds the Java scripted policy and VectorEnv.playout to a crforge checkout: apply.py <crforge dir>"""
import shutil
import sys
from pathlib import Path

root = Path(sys.argv[1]) / "env/src/main/java/org/crforge/env"
here = Path(__file__).resolve().parent
shutil.copy(here / "ScriptedPolicy.java", root / "ScriptedPolicy.java")

lad = root / "LadderEnv.java"
s = lad.read_text()
if "void copyMask(" not in s:
    anchor = "  /** The info record as last written"
    assert anchor in s
    s = s.replace(anchor, """  /** A seat's mask as last written, {@link Schema#ACTIONS} bytes, into an array. */
  void copyMask(int seat, byte[] into, int at) {
    masks.get(seat * Schema.ACTIONS, into, at, Schema.ACTIONS);
  }

""" + anchor, 1)
    lad.write_text(s)

vec = root / "VectorEnv.java"
s = vec.read_text()
if "public int[] playout(" not in s:
    anchor = "  private void stepEnv(int index) {"
    assert anchor in s
    s = s.replace(anchor, """  /**
   * Playouts with both sides playing (arena-lab): per request a fork of environment
   * {@code indices[r]} as {@code seats[r]} knows it (the enemy's unseen hand shuffled with
   * {@code shuffles[r]}, random source {@code seeds[r]}); the seat waits {@code delays[r]}
   * decisions, then plays {@code actions[r]} (a wait when the mask closes it), then both seats
   * follow {@code policy} until {@code horizonTicks} ticks passed or the battle ended.
   *
   * @return per request the seat's final observation then the fork's info record, as
   *     {@link #lookahead}
   */
  public int[] playout(int[] indices, int[] seats, int[] actions, int[] delays, int horizonTicks,
      long[] shuffles, long[] seeds, ScriptedPolicy policy) {
    int count = indices.length;
    LadderEnv[] sources = new LadderEnv[count];
    for (int r = 0; r < count; r++) {
      if (!(envs[indices[r]] instanceof LadderEnv ladder) || ladder.done()) {
        throw new IllegalArgumentException("environment " + indices[r] + " is not running");
      }
      sources[r] = ladder;
    }
    int stride = Schema.SEAT_INTS + Schema.INFO_INTS;
    int[] out = new int[count * stride];
    run(
        count,
        r -> {
          int seat = seats[r];
          try {
            LadderEnv fork = sources[r].fork(seat);
            SplittableRandom rng = new SplittableRandom(seeds[r] ^ 0x9E3779B97F4A7C15L);
            if (shuffles != null) {
              fork.shuffleUnseenEnemyCards(seat, new SplittableRandom(shuffles[r]));
            }
            fork.reseed((int) seeds[r]);
            int[] obs = new int[2 * Schema.SEAT_INTS];
            byte[] mask = new byte[2 * Schema.ACTIONS];
            int until = fork.tick() + horizonTicks;
            int step = 0;
            while (!fork.done() && fork.tick() < until) {
              fork.copyObservation(0, obs, 0);
              fork.copyObservation(1, obs, Schema.SEAT_INTS);
              fork.copyMask(0, mask, 0);
              fork.copyMask(1, mask, Schema.ACTIONS);
              int[] a = new int[2];
              for (int s = 0; s < 2; s++) {
                if (s == seat && step < delays[r]) {
                  a[s] = Schema.WAIT;
                } else if (s == seat && step == delays[r]) {
                  int act = actions[r];
                  a[s] = act >= 0 && act < Schema.ACTIONS && mask[s * Schema.ACTIONS + act] != 0 ? act : Schema.WAIT;
                } else {
                  a[s] = policy.act(obs, s * Schema.SEAT_INTS, mask, s * Schema.ACTIONS, rng);
                }
              }
              fork.step(a[0], a[1]);
              step++;
            }
            fork.copyObservation(seat, out, r * stride);
            fork.copyInfo(out, r * stride + Schema.SEAT_INTS);
          } catch (RuntimeException | AssertionError | StackOverflowError e) {
            Arrays.fill(out, r * stride, (r + 1) * stride, 0);
            out[r * stride + Schema.SEAT_INTS + Schema.I_FAULT] = EnvFault.of(e, -1).kind().code();
          }
        });
    return out;
  }

""" + anchor, 1)
    vec.write_text(s)

s = vec.read_text()
if "public int[] scriptedActs(" not in s:
    anchor = "  private void stepEnv(int index) {"
    s = s.replace(anchor, """  /** The scripted policy's action for every (environment, seat) pair, computed in parallel (arena-lab). */
  public int[] scriptedActs(int[] indices, int[] seats, long seed, ScriptedPolicy policy) {
    int count = indices.length;
    int[] out = new int[count];
    run(
        count,
        r -> {
          if (!(envs[indices[r]] instanceof LadderEnv ladder) || ladder.done()) {
            out[r] = Schema.WAIT;
            return;
          }
          int[] obs = new int[Schema.SEAT_INTS];
          byte[] mask = new byte[Schema.ACTIONS];
          ladder.copyObservation(seats[r], obs, 0);
          ladder.copyMask(seats[r], mask, 0);
          out[r] = policy.act(obs, 0, mask, 0, new SplittableRandom(seed * 1_000_003L + r));
        });
    return out;
  }

""" + anchor, 1)
    vec.write_text(s)

s = vec.read_text()
if "public int[] replyStart(" not in s:
    anchor = "  private void stepEnv(int index) {"
    s = s.replace(anchor, """  private LadderEnv[] replyForks;
  private int[] replyUntil;

  /**
   * Search with the enemy's answer, part 1 (arena-lab): per request a fork of environment
   * {@code indices[r]} as {@code seats[r]} knows it (the enemy's unseen hand shuffled), the seat
   * plays {@code actions[r]} (a wait when the mask closes it), then both wait until
   * {@code replyTicks} ticks passed. The forks are kept for {@link #replyFinish}.
   *
   * @return per request the ENEMY's observation, the enemy's mask (one int per action) and a
   *     status int (0 running, 1 ended, 2 fault)
   */
  public int[] replyStart(int[] indices, int[] seats, int[] actions, int replyTicks, int horizonTicks,
      long[] shuffles, long[] seeds) {
    int count = indices.length;
    LadderEnv[] sources = new LadderEnv[count];
    for (int r = 0; r < count; r++) {
      if (!(envs[indices[r]] instanceof LadderEnv ladder) || ladder.done()) {
        throw new IllegalArgumentException("environment " + indices[r] + " is not running");
      }
      sources[r] = ladder;
    }
    LadderEnv[] forks = new LadderEnv[count];
    int[] until = new int[count];
    int stride = Schema.SEAT_INTS + Schema.ACTIONS + 1;
    int[] out = new int[count * stride];
    run(
        count,
        r -> {
          int seat = seats[r];
          try {
            LadderEnv fork = sources[r].fork(seat);
            if (shuffles != null) {
              fork.shuffleUnseenEnemyCards(seat, new SplittableRandom(shuffles[r]));
            }
            fork.reseed((int) seeds[r]);
            until[r] = fork.tick() + horizonTicks;
            byte[] mask = new byte[Schema.ACTIONS];
            fork.copyMask(seat, mask, 0);
            int act = actions[r];
            int a = act >= 0 && act < Schema.ACTIONS && mask[act] != 0 ? act : Schema.WAIT;
            int replyAt = fork.tick() + replyTicks;
            boolean first = true;
            while (!fork.done() && (first || fork.tick() < replyAt)) {
              fork.step(seat == 0 && first ? a : Schema.WAIT, seat == 1 && first ? a : Schema.WAIT);
              first = false;
            }
            forks[r] = fork;
            int e = 1 - seat;
            fork.copyObservation(e, out, r * stride);
            byte[] em = new byte[Schema.ACTIONS];
            fork.copyMask(e, em, 0);
            for (int k = 0; k < Schema.ACTIONS; k++) {
              out[r * stride + Schema.SEAT_INTS + k] = em[k];
            }
            out[r * stride + stride - 1] = fork.done() ? 1 : 0;
          } catch (RuntimeException | AssertionError | StackOverflowError ex) {
            forks[r] = null;
            Arrays.fill(out, r * stride, (r + 1) * stride, 0);
            out[r * stride + stride - 1] = 2;
          }
        });
    replyForks = forks;
    replyUntil = until;
    return out;
  }

  /**
   * Search with the enemy's answer, part 2: the enemy plays {@code enemyActions[r]} (a wait when
   * its mask closes it) in each kept fork, then both wait until the horizon of part 1.
   *
   * @return per request the seat's final observation then the fork's info record, as
   *     {@link #lookahead}
   */
  public int[] replyFinish(int[] seats, int[] enemyActions) {
    LadderEnv[] forks = replyForks;
    int[] until = replyUntil;
    int count = forks.length;
    int stride = Schema.SEAT_INTS + Schema.INFO_INTS;
    int[] out = new int[count * stride];
    run(
        count,
        r -> {
          int seat = seats[r];
          try {
            LadderEnv fork = forks[r];
            if (fork == null) {
              throw new IllegalStateException("no fork for request " + r);
            }
            int e = 1 - seat;
            if (!fork.done()) {
              byte[] em = new byte[Schema.ACTIONS];
              fork.copyMask(e, em, 0);
              int act = enemyActions[r];
              int a = act >= 0 && act < Schema.ACTIONS && em[act] != 0 ? act : Schema.WAIT;
              fork.step(e == 0 ? a : Schema.WAIT, e == 1 ? a : Schema.WAIT);
            }
            while (!fork.done() && fork.tick() < until[r]) {
              fork.step(Schema.WAIT, Schema.WAIT);
            }
            fork.copyObservation(seat, out, r * stride);
            fork.copyInfo(out, r * stride + Schema.SEAT_INTS);
          } catch (RuntimeException | AssertionError | StackOverflowError ex) {
            Arrays.fill(out, r * stride, (r + 1) * stride, 0);
            out[r * stride + Schema.SEAT_INTS + Schema.I_FAULT] = EnvFault.of(ex, -1).kind().code();
          }
        });
    replyForks = null;
    replyUntil = null;
    return out;
  }

""" + anchor, 1)
    vec.write_text(s)
print("patched")
