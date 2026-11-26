# Smoke test of sn_ft.py step 2+3 on a tiny set (CPU, GitHub runner).
import io, json, os, random, torch, pyarrow.parquet as pq
from PIL import Image
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
RID, MID = "SudhishNeel/clash-royale-detection", "SudhishNeel/clash-royale-detector"
names = open(hf_hub_download(RID, "classes.txt", repo_type="dataset")).read().split(); NC = len(names)
D = "/tmp/ds"
for s in ("images/train", "images/val", "labels/train", "labels/val"): os.makedirs(f"{D}/{s}", exist_ok=True)
t = pq.read_table(hf_hub_download(RID, "synthetic_arena/train-c001a-00000-of-00002.parquet", repo_type="dataset"), columns=["image", "objects"]).slice(0, 12).to_pylist()
for i, row in enumerate(t):
    im = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB"); w, h = im.size
    sp = "val" if i < 4 else "train"
    im.save(f"{D}/images/{sp}/s{i}.jpg")
    with open(f"{D}/labels/{sp}/s{i}.txt", "w") as fo:
        for (x1, y1, x2, y2), c in zip(row["objects"]["bbox"], row["objects"]["category"]):
            fo.write(f"{c} {(x1 + x2) / 2 / w:.5f} {(y1 + y2) / 2 / h:.5f} {(x2 - x1) / w:.5f} {(y2 - y1) / h:.5f}\n")
open(f"{D}/data.yaml", "w").write(f"path: {D}\ntrain: images/train\nval: images/val\nnames: {json.dumps(names)}\n")
v9 = torch.load(hf_hub_download(MID, "v9/best.pt"), map_location="cpu", weights_only=False)
src = (v9.get("ema") or v9["model"]).float().state_dict()
dm = DetectionModel("yolov8n.yaml", nc=NC, verbose=False); sd = dm.state_dict(); copied = grown = 0
for k, v in src.items():
    if k not in sd: continue
    if sd[k].shape == v.shape: sd[k] = v; copied += 1
    elif v.dim() >= 1 and sd[k].shape[1:] == v.shape[1:] and v.shape[0] < sd[k].shape[0]: sd[k][:v.shape[0]] = v; grown += 1
dm.load_state_dict(sd); dm.names = {i: n for i, n in enumerate(names)}
torch.save({"model": dm.half(), "train_args": {}, "epoch": -1}, "/tmp/init.pt")
print("SMOKE init copied", copied, "grown", grown, "of", len(src))
m = YOLO("/tmp/init.pt")
r = m.predict(f"{D}/images/val/s0.jpg", imgsz=800, conf=0.3, verbose=False)[0]
print("SMOKE init predicts", [names[int(c)] for c in r.boxes.cls][:15], "truth", [names[c] for c in t[0]["objects"]["category"]][:15])
m.train(data=f"{D}/data.yaml", imgsz=800, epochs=1, batch=4, workers=2, device="cpu", plots=False, project="/tmp/run", name="ft", exist_ok=True, lr0=0.002)
print("SMOKE trained ok", os.listdir("/tmp/run/ft/weights"))
