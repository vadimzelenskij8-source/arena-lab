"""Probe: convert a few IL_Replay_1115 replays into crforge battles (ethan-mcgee/clash-ai converter, MIT)
and label them with OUR crforge (decision_ticks=5, the schema our network plays). Prints speed, fidelity, sizes."""
import glob, io, json, os, sys, time
import numpy as np
N = int(sys.argv[1]) if len(sys.argv) > 1 else 64
t0 = time.time()
from clashai.data import download, convert
download.main([]) if hasattr(download, "main") else None
print("download s", round(time.time() - t0), flush=True)
t1 = time.time()
convert.main(["--deck", "all", "--limit", str(N), "--out", "/tmp/conv", "--batch", "32"])
print("convert s", round(time.time() - t1), flush=True)
docs = sorted(f for f in glob.glob("/tmp/conv/*.json"))
print("docs", len(docs), flush=True)
from crforge_env.labels import label_replays, flag_names
t2 = time.time()
run = label_replays(docs, os.environ["CLASHAI_CRFORGE_INSTALL"], os.environ["CLASHAI_TABLES"], decision_ticks=5, threads=4, check=True)
print("label s", round(time.time() - t2), flush=True)
rows = nz = 0; fl = {}; sizes = []; repro = 0; keys = None
for r in getattr(run, "labels", getattr(run, "replays", [])):
    rows += r.rows; nz += int((r.actions != 0).sum())
    for f in r.flags[r.flags != 0]:
        for n_ in flag_names(int(f)): fl[n_] = fl.get(n_, 0) + 1
    real = r.masks[:, 1:].any(1)
    b = io.BytesIO(); np.savez_compressed(b, o=r.observations[real], a=r.actions[real]); sizes.append((b.tell(), int(real.sum())))
    if r.drive: repro += bool(r.drive.get("reproduced")); keys = list(r.drive)
print("LABEL replays", len(sizes), "rows", rows, "plays", nz, "flags", fl, "reproduced", repro, keys, flush=True)
if sizes:
    print("SIZE per replay (real-choice rows, compressed) KB", round(np.mean([s for s, _ in sizes]) / 1024), "rows", round(np.mean([n for _, n in sizes])), flush=True)
print("RUN", {k: v for k, v in vars(run).items() if k not in ("labels", "replays")} if hasattr(run, "__dict__") else run, flush=True)
