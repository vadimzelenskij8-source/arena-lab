# Kaggle kernel: pseudo-label the user's real frames with the synthetic model, then fine-tune on synth + real.
import glob, json, os, random, shutil, subprocess, sys
import torch
print("CUDA", torch.cuda.is_available(), flush=True)
if not torch.cuda.is_available():
    raise SystemExit("no GPU")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics>=8.3"], check=True)
from ultralytics import YOLO
W = "/kaggle/working"
root = f"{W}/ds"
os.makedirs(root, exist_ok=True)
w = [p for p in glob.glob("/kaggle/input/**/best.pt", recursive=True) if "train3" in p] or glob.glob("/kaggle/input/**/best.pt", recursive=True)
print("weights", w, flush=True)
classes = json.load(open(glob.glob("/kaggle/input/**/classes.json", recursive=True)[0]))
for k, tar in enumerate(sorted(glob.glob("/kaggle/input/**/*.tar", recursive=True))):
    d = f"{root}/{k}"
    os.makedirs(d, exist_ok=True)
    subprocess.run(["tar", "-xf", tar, "-C", d], check=True)
synth = sorted(p for p in glob.glob(root + "/**/images/*.jpg", recursive=True))
real = sorted(p for p in glob.glob(root + "/**/frames/*.jpg", recursive=True))
print("synth", len(synth), "real", len(real), flush=True)
m = YOLO(w[0])
CONF_KEEP, CONF_UNSURE = 0.5, 0.25
rd = f"{W}/real"
os.makedirs(rd + "/images", exist_ok=True); os.makedirs(rd + "/labels", exist_ok=True)
kept = unsure = 0
stats = {}
for i in range(0, len(real), 64):
    for r in m.predict(real[i:i + 64], imgsz=(832, 384), conf=CONF_UNSURE, verbose=False):
        conf = r.boxes.conf.tolist()
        if any(CONF_UNSURE <= c < CONF_KEEP for c in conf):
            unsure += 1
            continue
        name = os.path.basename(r.path)
        shutil.copy(r.path, f"{rd}/images/{name}")
        with open(f"{rd}/labels/{name[:-4]}.txt", "w") as f:
            for c, b in zip(r.boxes.cls.tolist(), r.boxes.xywhn.tolist()):
                f.write("%d %.5f %.5f %.5f %.5f\n" % (int(c), *b))
                stats[classes[int(c)]] = stats.get(classes[int(c)], 0) + 1
        kept += 1
print("pseudo-labelled frames kept", kept, "skipped (unsure)", unsure, flush=True)
print("classes seen on real frames", sorted(stats.items(), key=lambda x: -x[1])[:60], flush=True)
json.dump(stats, open(f"{W}/real_stats.json", "w"))
reals = sorted(glob.glob(rd + "/images/*.jpg"))
random.seed(0)
train = random.sample(synth, min(len(synth), 20000)) + reals * 3
val = reals[::10]
open(f"{W}/train.txt", "w").write("\n".join(train))
open(f"{W}/val.txt", "w").write("\n".join(val))
open(f"{W}/data.yaml", "w").write("path: /kaggle/working\ntrain: train.txt\nval: val.txt\nnames:\n" + "".join(f"  {i}: '{c}'\n" for i, c in enumerate(classes)))
m.train(data=f"{W}/data.yaml", imgsz=832, rect=True, epochs=int(os.environ.get("EPOCHS", "15")), batch=32, workers=4,
        fliplr=0.0, hsv_h=0.01, hsv_s=0.3, hsv_v=0.3, scale=0.2, translate=0.05, lr0=0.002,
        project=f"{W}/run", name="det", exist_ok=True)
shutil.copy(f"{W}/run/det/weights/best.pt", f"{W}/best.pt")
shutil.copy(glob.glob("/kaggle/input/**/classes.json", recursive=True)[0], f"{W}/classes.json")
# a few pseudo-labelled real frames for review
os.makedirs(f"{W}/review", exist_ok=True)
for r in YOLO(f"{W}/best.pt").predict(reals[::max(1, len(reals) // 24)][:24], imgsz=(832, 384), conf=0.3, verbose=False):
    r.save(filename=f"{W}/review/{os.path.basename(r.path)}")
shutil.rmtree(root, ignore_errors=True); shutil.rmtree(rd, ignore_errors=True)
print("done")
