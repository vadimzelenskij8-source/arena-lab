"""Ground-truth exam for detectors on KataCR's hand-labelled real frames (wty-yy/Clash-Royale-Detection-Dataset,
images/part2, LabelMe json), mapped to SudhishNeel's class names. ultralytics val -> P / R / mAP50 per model."""
import glob, json, os, random, re, shutil, sys
from collections import Counter
from PIL import Image
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
from ultralytics import YOLO
K, N = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1500
RID, MID = "SudhishNeel/clash-royale-detection", "SudhishNeel/clash-royale-detector"
pf = pq.ParquetFile(hf_hub_download(RID, "synthetic_arena/train-c001a-00000-of-00002.parquet", repo_type="dataset"))
names = json.loads(pf.schema_arrow.metadata[b"huggingface"])["info"]["features"]["objects"]["feature"]["category"]["names"]
idx = {n: i for i, n in enumerate(names)}
import hashlib
def held(j):  # the exam folders: never used for training (same rule in sn_ft2.py)
    return int(hashlib.sha1(os.path.dirname(j).encode().split(b"part2/")[-1]).hexdigest()[:6], 16) % 5 == 0
js = sorted(j for j in glob.glob(f"{K}/images/part2/**/*.json", recursive=True) if held(j))
random.Random(0).shuffle(js)
print("json files", len(js)); print("SAMPLE", js[0], open(js[0]).read()[:400].replace("\n", " "))
D = "/tmp/kv"; os.makedirs(f"{D}/images/val", exist_ok=True); os.makedirs(f"{D}/labels/val", exist_ok=True)
miss = Counter(); hit = Counter(); n = 0; sizes = Counter()
for j in js:
    if n >= N: break
    try:
        d = json.load(open(j))
    except Exception:
        continue
    img = j[:-5] + ".jpg"
    if not os.path.exists(img): continue
    W, H = d.get("imageWidth"), d.get("imageHeight")
    if not W:
        W, H = Image.open(img).size
    sizes[(W, H)] += 1
    lines = []; bad = False
    for s in d.get("shapes", []):
        lb = s["label"]; nm = re.sub(r"\d+$", "", re.sub(r"(_[a-z]+)+$", "", lb))  # "_attack/_freeze" states, then side digit
        nm = {"king-tower-bar": None, "tower-bar": None, "bar": None, "bar-level": None, "selected": None, "clock": "clock", "elixir": "elixir",
              "text": None, "emote": None, "evolution-symbol": None}.get(nm, nm)
        if nm is None: continue
        if nm not in idx:
            miss[nm] += 1; bad = True; continue
        (x1, y1), (x2, y2) = s["points"][0], s["points"][-1]
        x1, x2 = sorted((x1, x2)); y1, y2 = sorted((y1, y2))
        lines.append(f"{idx[nm]} {(x1 + x2) / 2 / W:.5f} {(y1 + y2) / 2 / H:.5f} {(x2 - x1) / W:.5f} {(y2 - y1) / H:.5f}")
        hit[nm] += 1
    if bad or not lines: continue  # a frame with a unit we cannot name would count right answers as false positives
    st = f"k{n:05d}"; shutil.copy(img, f"{D}/images/val/{st}.jpg")
    open(f"{D}/labels/val/{st}.txt", "w").write("\n".join(lines) + "\n"); n += 1
print("GT frames", n, "sizes", sizes.most_common(3), "boxes", sum(hit.values()), "classes", len(hit))
print("unmapped names", miss.most_common(30))
open(f"{D}/data.yaml", "w").write(f"path: {D}\ntrain: images/val\nval: images/val\nnames: {json.dumps(names)}\n")
models = {v: hf_hub_download(MID, f"{v}/best.pt") for v in ("v8", "v9", "v10")}
for f in glob.glob("/tmp/extra/*.pt"):
    models[os.path.basename(f)] = f
res = {}
for k, p in models.items():
    m = YOLO(p)
    if len(m.names) < len(names):  # older heads: class ids are a prefix of the full list
        pass
    r = m.val(data=f"{D}/data.yaml", imgsz=800, batch=16, conf=0.001, iou=0.6, device="cpu", plots=False, verbose=False)
    res[k] = dict(P=round(float(r.box.mp), 3), R=round(float(r.box.mr), 3), mAP50=round(float(r.box.map50), 3), mAP=round(float(r.box.map), 3))
    print("RESULT", k, res[k], flush=True)
json.dump(res, open("/tmp/kata_res.json", "w"))
