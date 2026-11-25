"""Shard k of K of IL_Replay_1115: RoyaleAPI replays -> crforge battles (ethan-mcgee/clash-ai converter, MIT)
-> rows labelled in OUR env schema (5-tick decisions) -> one compressed npz per replay.

    python il/conv.py --shard 3 --of 16 --limit 2000 --out data/
npz keys: o (rows, seat_ints) int32, m packed mask bits, a label, s seat, k clean-row flag.
"""
import argparse, glob, hashlib, json, os, shutil, time
from pathlib import Path

import numpy as np
from clashai.data import convert as C
from clashai.data.cards import CardNames
from clashai.data.download import dataset_dir
from clashai.data.replays import read_replays
from clashai import paths


def shard_of(tag, K):
    return int(hashlib.sha1(tag.encode()).hexdigest()[:8], 16) % K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--of", type=int, required=True)
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--minutes", type=float, default=90)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tmp = Path("/tmp/conv_docs"); shutil.rmtree(tmp, ignore_errors=True)
    root = dataset_dir()
    reps = [r for r in read_replays(root) if shard_of(r.tag, a.of) == a.shard][:a.limit]
    print("selected", len(reps), flush=True)
    events = C.load_events(root, {r.tag for r in reps})
    paths.ensure_java()
    from crforge_env import LadderConfig, VectorBattle
    from crforge_env.labels import label_replays
    vector = VectorBattle(a.batch, paths.default_threads(), install=paths.install_dir(), tables=paths.tables_dir(),
                          config=LadderConfig(decision_ticks=1, level=C.DEFAULT_LEVEL, tower_level=C.DEFAULT_LEVEL, arena=C.ARENA))
    names = CardNames(vector.vocabulary[i] for i in vector.card_types())
    table = C.SeedTable()
    rng = np.random.default_rng(a.shard)
    t0 = time.time(); stats = dict(converted=0, problems=0, labelled=0, rows=0, plays=0, result_match=0)
    pending = []

    def flush():
        if not pending:
            return
        outs = C.drive(pending, vector, tmp)
        stats["converted"] += len(outs); stats["result_match"] += sum(o.result_match for o in outs)
        pending.clear()
        docs = sorted(glob.glob(str(tmp / "docs" / "*.json")))
        run = label_replays(docs, paths.install_dir(), paths.tables_dir(), decision_ticks=5, threads=4)
        for d, r in zip(docs, run.replays):
            if r.rows < 40:
                continue
            np.savez_compressed(out / (Path(d).stem + ".npz"), o=r.observations, m=np.packbits(r.masks.astype(bool), 1),
                                a=r.actions, s=r.seats, k=(r.flags == 0))
            stats["labelled"] += 1; stats["rows"] += r.rows; stats["plays"] += int((r.actions != 0).sum())
        shutil.rmtree(tmp / "docs", ignore_errors=True)
        print("progress", stats, round(time.time() - t0), "s", flush=True)

    for r in reps:
        if time.time() - t0 > a.minutes * 60:
            print("time budget reached", flush=True); break
        p = C.plan_replay(r, events.get(r.tag, []), names, table, rng, "screen", "team0", 0, 0)
        if p.problem:
            stats["problems"] += 1; continue
        pending.append(p)
        if len(pending) == vector.count:
            flush()
    flush()
    json.dump(stats, open(out / "conv_stats.json", "w"))
    print("DONE", stats, flush=True)


if __name__ == "__main__":
    main()
