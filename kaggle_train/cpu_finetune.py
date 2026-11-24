# GitHub Actions CPU fallback: short fine-tune of det-v1 on a synthetic subset (v3 scale).
import glob, json, os, random, shutil, subprocess, sys
from ultralytics import YOLO
root, out, n, ep = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
imgs = sorted(glob.glob(root + "/**/images/*.jpg", recursive=True))
random.seed(1); imgs = random.sample(imgs, min(n, len(imgs)))
classes = json.load(open(glob.glob(root + "/**/classes.json", recursive=True)[0]))
val = imgs[::25]
open("train.txt", "w").write("\n".join(i for i in imgs if i not in set(val)))
open("val.txt", "w").write("\n".join(val))
open("data.yaml", "w").write(f"path: {os.getcwd()}\ntrain: train.txt\nval: val.txt\nnames:\n" + "".join(f"  {i}: '{c}'\n" for i, c in enumerate(classes)))
m = YOLO(glob.glob(root + "/**/best.pt", recursive=True)[0])
m.train(data="data.yaml", imgsz=832, rect=True, epochs=ep, batch=16, workers=4, device="cpu", fliplr=0.0, scale=0.2, lr0=0.003,
        project="run", name="det", exist_ok=True, plots=False)
os.makedirs(out, exist_ok=True)
shutil.copy("run/det/weights/best.pt", f"{out}/best.pt")
print("done")
