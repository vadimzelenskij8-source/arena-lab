# Fine-tune SudhishNeel's phone-frame detector (YOLOv8n v9) to OUR phone, all 187 classes.
# 1) teacher ensemble (v8n + v9n + v10s, weighted box fusion) labels our 3079 real frames on the fitted arena crop
# 2) 10k frames of their synthetic_arena (real cut-outs on real arena backgrounds)
# 3) train v9n (head grown 185 -> 187) on real-pseudo x3 + synth, CPU or GPU, within the time budget
# Outputs: best.pt, crop.json, classes.json, stats.json
import glob, io, json, os, random, shutil, subprocess, sys, time
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics>=8.3", "huggingface_hub", "pyarrow", "ensemble_boxes"], check=False)
import numpy as np, torch, pyarrow.parquet as pq
from PIL import Image
from huggingface_hub import hf_hub_download
from ensemble_boxes import weighted_boxes_fusion
from ultralytics import YOLO
t0 = time.time(); random.seed(3)
W = "/kaggle/working"; D = "/tmp/ds"
DEV = 0 if torch.cuda.is_available() else "cpu"
HOURS = 10.3 if DEV == "cpu" else 8.0
print("device", DEV, flush=True)
CROP = [8, 103, 532, 886]; CS = (568, 896)
RID, MID = "SudhishNeel/clash-royale-detection", "SudhishNeel/clash-royale-detector"
names = open(hf_hub_download(RID, "classes.txt", repo_type="dataset")).read().split()
NC = len(names); print("classes", NC, flush=True)
for s in ("images/train", "images/val", "labels/train", "labels/val"):
    os.makedirs(f"{D}/{s}", exist_ok=True)

# --- 1) teacher ensemble on our frames
teachers = [YOLO(hf_hub_download(MID, f"{v}/best.pt")) for v in ("v8", "v9", "v10")]
real = sorted(glob.glob("/kaggle/input/**/frames/*.jpg", recursive=True))
print("real frames", len(real), flush=True)
kept = 0; nbox = 0; per = {}
val_ids = set(random.sample(range(len(real)), min(150, len(real))))
B = 16
for i in range(0, len(real), B):
    fs = real[i:i + B]
    ims = [Image.open(f).convert("RGB").resize((540, 1200)).crop(CROP).resize(CS, Image.BILINEAR) for f in fs]
    outs = [t.predict(ims, imgsz=800, conf=0.2, verbose=False, device=DEV) for t in teachers]
    for k, f in enumerate(fs):
        bl, sl, ll = [], [], []
        for r in (o[k] for o in outs):
            b = r.boxes
            bl.append((b.xyxyn.cpu().numpy().clip(0, 1)).tolist()); sl.append(b.conf.cpu().numpy().tolist()); ll.append(b.cls.cpu().numpy().astype(int).tolist())
        if not any(bl):
            continue
        bb, ss, cc = weighted_boxes_fusion(bl, sl, ll, weights=[1, 1, 1.2], iou_thr=0.55, skip_box_thr=0.2, conf_type="avg")
        keep = ss >= 0.45  # avg over 3 teachers: a box one model alone found counts at 1/3 of its score
        unsure = ((ss >= 0.25) & (ss < 0.45)).sum()
        if unsure > 2:
            continue  # too many doubtful boxes: an unlabelled unit would teach "background"
        split = "val" if (i + k) in val_ids else "train"
        stem = f"real_{i + k:05d}"
        ims[k].save(f"{D}/images/{split}/{stem}.jpg", quality=92)
        with open(f"{D}/labels/{split}/{stem}.txt", "w") as fo:
            for (x1, y1, x2, y2), c in zip(bb[keep], cc[keep].astype(int)):
                fo.write(f"{c} {(x1 + x2) / 2:.5f} {(y1 + y2) / 2:.5f} {x2 - x1:.5f} {y2 - y1:.5f}\n")
                per[names[c]] = per.get(names[c], 0) + 1; nbox += 1
        if split == "train":
            for rep in (1, 2):  # our phone is the target domain: weigh it x3
                shutil.copy(f"{D}/images/train/{stem}.jpg", f"{D}/images/train/{stem}_r{rep}.jpg")
                shutil.copy(f"{D}/labels/train/{stem}.txt", f"{D}/labels/train/{stem}_r{rep}.txt")
        kept += 1
    if i % 320 == 0:
        print(f"pseudo {i}/{len(real)} kept {kept} boxes {nbox} ({(time.time() - t0) / 60:.0f} min)", flush=True)
print("PSEUDO kept", kept, "of", len(real), "boxes", nbox, "top", sorted(per.items(), key=lambda x: -x[1])[:40], flush=True)
del teachers

# --- 2) synthetic arena frames (real cut-outs), 568x896, bbox xyxy px
NSYN = 6000 if DEV == "cpu" else 30000
shards = [f"synthetic_arena/train-c00{a}{b}-0000{c}-of-00002.parquet" for a in "123" for b in "abcde" for c in "01"]
random.shuffle(shards)
got = 0; syn_cls = {}
for sh in shards:
    if got >= NSYN:
        break
    try:
        p = hf_hub_download(RID, sh, repo_type="dataset")
    except Exception as e:
        print("skip", sh, e); continue
    t = pq.read_table(p, columns=["image", "objects"]).to_pylist()
    for row in random.sample(t, min(len(t), max(0, NSYN - got), 1500)):
        im = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
        w, h = im.size
        stem = f"syn_{got:06d}"
        im.save(f"{D}/images/train/{stem}.jpg", quality=90)
        with open(f"{D}/labels/train/{stem}.txt", "w") as fo:
            for (x1, y1, x2, y2), c in zip(row["objects"]["bbox"], row["objects"]["category"]):
                fo.write(f"{c} {(x1 + x2) / 2 / w:.5f} {(y1 + y2) / 2 / h:.5f} {(x2 - x1) / w:.5f} {(y2 - y1) / h:.5f}\n")
                syn_cls[c] = syn_cls.get(c, 0) + 1
        got += 1
    os.remove(p)
    print(f"synth {got} ({(time.time() - t0) / 60:.0f} min)", flush=True)
print("SYNTH frames", got, "classes with no box", [names[c] for c in range(NC) if c not in syn_cls and names[c] not in per], flush=True)

# --- 3) student: v9n with its 185-class head grown to 187 (keeps what it knows)
open(f"{D}/data.yaml", "w").write(f"path: {D}\ntrain: images/train\nval: images/val\nnames: {json.dumps(names)}\n")
v9 = torch.load(hf_hub_download(MID, "v9/best.pt"), map_location="cpu", weights_only=False)
src = (v9.get("ema") or v9["model"]).float().state_dict()
from ultralytics.nn.tasks import DetectionModel
dm = DetectionModel("yolov8n.yaml", nc=NC, verbose=False)
sd = dm.state_dict(); copied = grown = 0
for k, v in src.items():
    if k not in sd:
        continue
    if sd[k].shape == v.shape:
        sd[k] = v; copied += 1
    elif v.dim() >= 1 and sd[k].shape[1:] == v.shape[1:] and v.shape[0] < sd[k].shape[0]:
        sd[k][:v.shape[0]] = v; grown += 1
dm.load_state_dict(sd); dm.names = {i: n for i, n in enumerate(names)}
torch.save({"model": dm.half(), "train_args": {}, "epoch": -1}, f"{W}/init.pt")
print("init copied", copied, "grown", grown, flush=True)
left = HOURS - (time.time() - t0) / 3600
m = YOLO(f"{W}/init.pt")
m.train(data=f"{D}/data.yaml", imgsz=800, epochs=60, time=max(0.5, left), batch=16 if DEV == "cpu" else 32, workers=4, device=DEV,
        fliplr=0.5, scale=0.3, mosaic=1.0, close_mosaic=2, lr0=0.002, warmup_epochs=0.5, patience=100,
        project=f"{W}/run", name="ft", exist_ok=True, plots=False, verbose=False)
best = f"{W}/run/ft/weights/best.pt"
shutil.copy(best if os.path.exists(best) else f"{W}/run/ft/weights/last.pt", f"{W}/best.pt")
json.dump({"crop_on_540x1200": CROP, "imgsz": 800, "crop_size": list(CS)}, open(f"{W}/crop.json", "w"))
json.dump(names, open(f"{W}/classes.json", "w"))
json.dump({"pseudo_kept": kept, "pseudo_boxes": nbox, "per_class": per, "synth": got}, open(f"{W}/stats.json", "w"))
for f in glob.glob(f"{W}/run/ft/results.csv"):
    print(open(f).read()[-1500:])
shutil.rmtree(f"{W}/run/ft/weights", ignore_errors=True); os.remove(f"{W}/init.pt")
print(f"ALL DONE {(time.time() - t0) / 60:.0f} min", flush=True)
