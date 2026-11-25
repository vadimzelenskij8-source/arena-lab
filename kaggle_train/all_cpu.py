# Kaggle (run interactively with Run All, GPU T4 x2): synth fine-tune -> pseudo-label real frames -> final fine-tune.
# Outputs in /kaggle/working: best.pt, classes.json, real_stats.json, review/*.jpg
import glob, json, os, random, shutil, subprocess, sys, time
import torch
print("CUDA", torch.cuda.is_available(), flush=True)
DEV = 0 if torch.cuda.is_available() else "cpu"
print("device", DEV, "cpus", os.cpu_count(), flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics>=8.3"], check=True)
from ultralytics import YOLO
W = "/kaggle/working"; root = f"{W}/ds"; os.makedirs(root, exist_ok=True)
t0 = time.time()
for k, tar in enumerate(sorted(glob.glob("/kaggle/input/**/*.tar", recursive=True))):
    d = f"{root}/{k}"; os.makedirs(d, exist_ok=True)
    subprocess.run(["tar", "-xf", tar, "-C", d], check=True)
classes = json.load(open(glob.glob("/kaggle/input/**/classes.json", recursive=True)[0]))
subprocess.run("find /kaggle/input -maxdepth 4 | head -40", shell=True)
synth = sorted(glob.glob(root + "/**/images/*.jpg", recursive=True)) or sorted(glob.glob("/kaggle/input/**/images/*.jpg", recursive=True))
real = sorted(glob.glob(root + "/**/frames/*.jpg", recursive=True)) or sorted(glob.glob("/kaggle/input/**/frames/*.jpg", recursive=True))
if synth and synth[0].startswith("/kaggle/input"):  # read-only input: labels sit next to images, YOLO finds them
    pass
print(f"STEP 0 data: synth {len(synth)} real {len(real)} classes {len(classes)} ({time.time()-t0:.0f}s)", flush=True)
names = "".join(f"  {i}: '{c}'\n" for i, c in enumerate(classes))
random.seed(0)


def data(train, val, tag):
    open(f"{W}/{tag}_train.txt", "w").write("\n".join(train))
    open(f"{W}/{tag}_val.txt", "w").write("\n".join(val))
    open(f"{W}/{tag}.yaml", "w").write(f"path: {W}\ntrain: {tag}_train.txt\nval: {tag}_val.txt\nnames:\n" + names)
    return f"{W}/{tag}.yaml"


AUG = dict(imgsz=832, rect=True, batch=16, workers=4, device=DEV, val=False, fliplr=0.0, hsv_h=0.01, hsv_s=0.3, hsv_v=0.3, scale=0.25,
           translate=0.05, project=f"{W}/run", exist_ok=True, plots=False)
# STEP 1: fine-tune det-v1 on correctly scaled synthetic data
sub = random.sample(synth, min(4000, len(synth)))
init = glob.glob("/kaggle/input/**/best.pt", recursive=True)[0]
YOLO(init).train(data=data(sub, sub[::40], "s1"), epochs=5, name="s1", **AUG)
m = YOLO((f"{W}/run/s1/weights/best.pt" if os.path.exists(f"{W}/run/s1/weights/best.pt") else f"{W}/run/s1/weights/last.pt"))
print(f"STEP 1 done ({(time.time()-t0)/60:.0f} min)", flush=True)
# STEP 2: pseudo-label the real frames (keep only frames where every detection is confident)
rd = f"{W}/real"; os.makedirs(rd + "/images", exist_ok=True); os.makedirs(rd + "/labels", exist_ok=True)
kept = unsure = 0; stats = {}
for i in range(0, len(real), 64):
    for r in m.predict(real[i:i + 64], imgsz=(832, 384), conf=0.25, verbose=False):
        conf = r.boxes.conf.tolist()
        if any(0.25 <= c < 0.5 for c in conf):
            unsure += 1; continue
        n = os.path.basename(r.path)
        shutil.copy(r.path, f"{rd}/images/{n}")
        with open(f"{rd}/labels/{n[:-4]}.txt", "w") as f:
            for c, b in zip(r.boxes.cls.tolist(), r.boxes.xywhn.tolist()):
                f.write("%d %.5f %.5f %.5f %.5f\n" % (int(c), *b))
                stats[classes[int(c)]] = stats.get(classes[int(c)], 0) + 1
        kept += 1
json.dump({"kept": kept, "unsure": unsure, "classes": stats}, open(f"{W}/real_stats.json", "w"))
print(f"STEP 2 real frames kept {kept} unsure {unsure}; top classes {sorted(stats.items(), key=lambda x: -x[1])[:25]}", flush=True)
# STEP 3: final fine-tune on synthetic + pseudo-labelled real frames
reals = sorted(glob.glob(rd + "/images/*.jpg"))
mix = random.sample(synth, min(2000, len(synth))) + reals * 2
m.train(data=data(mix, reals[::10] or sub[::40], "s3"), epochs=5, name="s3", lr0=0.002, **AUG)
shutil.copy(f"{W}/run/s3/weights/best.pt" if os.path.exists(f"{W}/run/s3/weights/best.pt") else f"{W}/run/s3/weights/last.pt", f"{W}/best.pt")
shutil.copy(glob.glob("/kaggle/input/**/classes.json", recursive=True)[0], f"{W}/classes.json")
os.makedirs(f"{W}/review", exist_ok=True)
for r in YOLO(f"{W}/best.pt").predict(reals[::max(1, len(reals) // 24)][:24] or real[:24], imgsz=(832, 384), conf=0.3, verbose=False):
    r.save(filename=f"{W}/review/{os.path.basename(r.path)}")
shutil.rmtree(root, ignore_errors=True); shutil.rmtree(rd, ignore_errors=True)

print(f"ALL DONE in {(time.time()-t0)/60:.0f} min. Now press Save Version -> Quick Save.", flush=True)
