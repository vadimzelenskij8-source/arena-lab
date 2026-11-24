# Kaggle kernel: train the battle detector on the synthetic dataset (GPU).
import glob, json, os, shutil, subprocess, sys
import torch
print("CUDA", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-", flush=True)
if not torch.cuda.is_available() and os.environ.get("ALLOW_CPU") != "1":
    raise SystemExit("no GPU in this session")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics>=8.3"], check=True)
subprocess.run("find /kaggle/input -maxdepth 5 | head -60", shell=True)
root = "/kaggle/working/ds"
os.makedirs(root, exist_ok=True)
classes = None
for k, tar in enumerate(sorted(glob.glob("/kaggle/input/**/*.tar", recursive=True))):
    d = f"{root}/{k}"
    os.makedirs(d, exist_ok=True)
    subprocess.run(["tar", "-xf", tar, "-C", d], check=True)
for cj in glob.glob("/kaggle/input/**/classes.json", recursive=True):
    classes = json.load(open(cj))
imgs = sorted(glob.glob(root + "/**/images/*.jpg", recursive=True))
if not imgs:  # Kaggle may have unpacked the archives already: link them in place
    for k, d in enumerate(sorted({os.path.dirname(os.path.dirname(p)) for p in glob.glob("/kaggle/input/**/images/*.jpg", recursive=True)})):
        os.symlink(d, f"{root}/u{k}")
    imgs = sorted(glob.glob(root + "/**/images/*.jpg", recursive=True))
print("images", len(imgs), "classes", len(classes))
import random
random.seed(0)
imgs = random.sample(imgs, min(len(imgs), 20000))
val = imgs[::40]
open("/kaggle/working/train.txt", "w").write("\n".join(i for i in imgs if i not in set(val)))
open("/kaggle/working/val.txt", "w").write("\n".join(val))
yaml = "path: /kaggle/working\ntrain: train.txt\nval: val.txt\nnames:\n" + "".join(f"  {i}: '{c}'\n" for i, c in enumerate(classes))
open("/kaggle/working/data.yaml", "w").write(yaml)
from ultralytics import YOLO
EPOCHS = int(os.environ.get("EPOCHS", "20"))
init = (glob.glob("/kaggle/input/**/best.pt", recursive=True) or ["yolo11n.pt"])[0]
print("init weights", init)
m = YOLO(init)
m.train(data="/kaggle/working/data.yaml", imgsz=832, rect=True, epochs=EPOCHS, batch=32, workers=4, cache=False,
        fliplr=0.0, hsv_h=0.01, hsv_s=0.3, hsv_v=0.3, scale=0.25, translate=0.05,
        patience=100, project="/kaggle/working/run", name="det", exist_ok=True)
best = "/kaggle/working/run/det/weights/best.pt"
shutil.copy(best, "/kaggle/working/best.pt")
try:
    YOLO(best).export(format="onnx", imgsz=832, simplify=True, opset=17)
    shutil.copy(best.replace(".pt", ".onnx"), "/kaggle/working/best.onnx")
except Exception as e:
    print("onnx export failed", e)
shutil.rmtree(root, ignore_errors=True)
print("done")
