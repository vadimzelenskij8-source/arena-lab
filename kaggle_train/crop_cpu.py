# Kaggle CPU: train on the ARENA CROP (y 120..920 of the 540x1200 frame), where units are ~1.5x bigger for the model.
# 1) crop synthetic frames+labels  2) pseudo-label real frames with the previous model on crops (good scale)
# 3) fine-tune on synth crops + confident real crops.  Outputs: best.pt, classes.json, crop.json, real_stats.json
import glob, json, os, random, shutil, subprocess, sys, time
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics>=8.3"], check=True)
import torch
from PIL import Image
from ultralytics import YOLO
W = "/kaggle/working"; t0 = time.time()
CROP = (0, 120, 540, 920)  # x0, y0, x1, y1 on the 540x1200 frame
DEV = 0 if torch.cuda.is_available() else "cpu"
classes = json.load(open(glob.glob("/kaggle/input/**/classes.json", recursive=True)[0]))
synth = sorted(glob.glob("/kaggle/input/**/images/*.jpg", recursive=True))
real = sorted(glob.glob("/kaggle/input/**/frames/*.jpg", recursive=True))
init = [p for p in glob.glob("/kaggle/input/**/best.pt", recursive=True)][0]
print("device", DEV, "synth", len(synth), "real", len(real), "init", init, flush=True)
random.seed(0)
cw, ch = CROP[2] - CROP[0], CROP[3] - CROP[1]


def crop_img(src, dst):
    im = Image.open(src).convert("RGB")
    if im.size != (540, 1200):
        im = im.resize((540, 1200))
    im.crop(CROP).save(dst, quality=92)


def crop_labels(src_txt, dst_txt):
    out = []
    if os.path.exists(src_txt):
        for line in open(src_txt):
            p = line.split()
            if len(p) != 5:
                continue
            c, x, y, w, h = int(p[0]), *map(float, p[1:])
            x0, x1 = (x - w / 2) * 540, (x + w / 2) * 540
            y0, y1 = (y - h / 2) * 1200 - CROP[1], (y + h / 2) * 1200 - CROP[1]
            a = max(0, (x1 - x0) * (y1 - y0))
            cx0, cx1, cy0, cy1 = max(0, x0), min(cw, x1), max(0, y0), min(ch, y1)
            if cx1 <= cx0 or cy1 <= cy0 or (cx1 - cx0) * (cy1 - cy0) < 0.4 * a:
                continue
            out.append("%d %.5f %.5f %.5f %.5f" % (c, (cx0 + cx1) / 2 / cw, (cy0 + cy1) / 2 / ch, (cx1 - cx0) / cw, (cy1 - cy0) / ch))
    open(dst_txt, "w").write("\n".join(out))


S = f"{W}/sc"; os.makedirs(S + "/images", exist_ok=True); os.makedirs(S + "/labels", exist_ok=True)
for p in random.sample(synth, min(3000, len(synth))):
    n = os.path.basename(p)
    crop_img(p, f"{S}/images/{n}")
    crop_labels(p.replace("/images/", "/labels/")[:-4] + ".txt", f"{S}/labels/{n[:-4]}.txt")
print(f"synth crops done ({time.time()-t0:.0f}s)", flush=True)
R = f"{W}/rc"; os.makedirs(R + "/images", exist_ok=True); os.makedirs(R + "/labels", exist_ok=True)
tmp = f"{W}/rtmp"; os.makedirs(tmp, exist_ok=True)
for p in real:
    crop_img(p, f"{tmp}/{os.path.basename(p)}")
m = YOLO(init)
kept = unsure = 0; stats = {}
rc = sorted(glob.glob(tmp + "/*.jpg"))
for i in range(0, len(rc), 32):
    for r in m.predict(rc[i:i + 32], imgsz=832, conf=0.25, device=DEV, verbose=False):
        conf = r.boxes.conf.tolist()
        if any(0.25 <= c < 0.5 for c in conf):
            unsure += 1; continue
        n = os.path.basename(r.path)
        shutil.move(r.path, f"{R}/images/{n}")
        with open(f"{R}/labels/{n[:-4]}.txt", "w") as f:
            for c, b in zip(r.boxes.cls.tolist(), r.boxes.xywhn.tolist()):
                f.write("%d %.5f %.5f %.5f %.5f\n" % (int(c), *b))
                stats[classes[int(c)]] = stats.get(classes[int(c)], 0) + 1
        kept += 1
json.dump({"kept": kept, "unsure": unsure, "classes": stats}, open(f"{W}/real_stats.json", "w"))
print(f"pseudo: kept {kept} unsure {unsure} top {sorted(stats.items(), key=lambda x: -x[1])[:25]} ({(time.time()-t0)/60:.0f} min)", flush=True)
shutil.rmtree(tmp, ignore_errors=True)
reals = sorted(glob.glob(R + "/images/*.jpg"))
train = sorted(glob.glob(S + "/images/*.jpg")) + reals * 2
open(f"{W}/train.txt", "w").write("\n".join(train))
open(f"{W}/val.txt", "w").write("\n".join(reals[::15] or train[:50]))
open(f"{W}/data.yaml", "w").write(f"path: {W}\ntrain: train.txt\nval: val.txt\nnames:\n" + "".join(f"  {i}: '{c}'\n" for i, c in enumerate(classes)))
m.train(data=f"{W}/data.yaml", imgsz=832, rect=True, epochs=6, batch=16, workers=4, device=DEV, fliplr=0.0, scale=0.2,
        hsv_s=0.3, hsv_v=0.3, lr0=0.002, project=f"{W}/run", name="crop", exist_ok=True, plots=False, val=False)
best = f"{W}/run/crop/weights/best.pt"
shutil.copy(best if os.path.exists(best) else f"{W}/run/crop/weights/last.pt", f"{W}/best.pt")
shutil.copy(glob.glob("/kaggle/input/**/classes.json", recursive=True)[0], f"{W}/classes.json")
json.dump({"crop_on_540x1200": CROP}, open(f"{W}/crop.json", "w"))
os.makedirs(f"{W}/review", exist_ok=True)
for r in YOLO(f"{W}/best.pt").predict(reals[::max(1, len(reals) // 24)][:24], imgsz=832, conf=0.3, device=DEV, verbose=False):
    r.save(filename=f"{W}/review/{os.path.basename(r.path)}")
for d in (S, R, f"{W}/run"):
    shutil.rmtree(d, ignore_errors=True)
print(f"ALL DONE {(time.time()-t0)/60:.0f} min", flush=True)
