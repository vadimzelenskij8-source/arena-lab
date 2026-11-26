"""Probe SudhishNeel/clash-royale-detector (YOLOv8n, 568x896 arena crop, imgsz 800) on our 30 eval frames.
Finds the crop of our 540x1200 frame that matches their 18x32-tile arena, then counts detections."""
import glob, json, os, urllib.request
import numpy as np
from PIL import Image
from huggingface_hub import HfApi, hf_hub_download
from ultralytics import YOLO
W = "/tmp/sn"; os.makedirs(W + "/frames", exist_ok=True)
ids = ['79LEx1kfj99LhfOsxqMR', 'pLFr8uEHuY5UdNBKSwaL', 'QNm6GEBGqPTbGHDrtg2p', 'VmuXAoaJxM3qZOTw2AR6', 'ZRVvEPQ8gYlOSWJVzSQ2', 'LbWYv11tdLfwK1LCCVSL', '03agofG7WQvosy92uxN4', '06U38jUQWN0TtWgjv54T', '08rfZf0EO9Ay3IIJuQEc', '0940ijpbbLl1Xr6EXPI8', '09IYML7ALVHwk02kzMul', '0DU9egwL0Wp35QUgzYpv', '0Dti7xFRSgjbl6gKYNKV', '0FMbqBe1RaMr4F0g01oX', '0IzHEpM6RXcuGmTTQjNg', '0KKfBVzQUcl7IHFbG9qH', '0MPHIYrOTaz08DN2sXYx', '0Vr2hpYNLYQYvJyzmG7r', '0WFiKk1iAUSShBwxsXA2', '0XxgMeqyNIUPYc1pLVxN', '0bVqslExai5ZHRkDJ8T9', '0cKSxzBbRPNtyHrQfRf2', '0kipun3NSbXC3oqjqUDZ', '0vva10j0JqBLLU8QQwwR', '10POmYExCEQWLzEJ9bEn', '13fePkwuT4OD4tJWGvp6', '1EI2082msrzcg2P2OXsf', '1Je2BIqZ4TmRLJyXKa6j', '1LzL9u1Y8Xvql1vyXQRf', '1PWAg3MNi7J2gJsYfLYP']
for i in ids:
    try:
        urllib.request.urlretrieve(f"https://source.roboflow.com/03WjEzRyYIPwtXk7A2aqTNAuk4X2/{i}/original.jpg", f"{W}/frames/{i}.jpg")
    except Exception as e:
        print("frame", i, e)
fr = sorted(glob.glob(W + "/frames/*.jpg"))
files = [f for f in HfApi().list_repo_files("SudhishNeel/clash-royale-detector")]
print("FILES", files, flush=True)
pts = sorted(f for f in files if f.endswith(".pt"))
ims = [Image.open(f).convert("RGB").resize((540, 1200)) for f in fr]

def run(m, box, conf=0.25, save=None):
    crops = [im.crop(tuple(int(round(v)) for v in box)).resize((568, 896), Image.BILINEAR) for im in ims]
    rs = m.predict(crops, imgsz=800, conf=conf, verbose=False)
    out = []
    for k, r in enumerate(rs):
        n = [m.names[int(c)] for c in r.boxes.cls]
        out.append((sum(1 for x in n if "tower" not in x), sum(1 for x in n if "tower" in x), float(r.boxes.conf.sum()) if len(n) else 0.0, n))
        if save:
            os.makedirs(save, exist_ok=True); r.save(filename=f"{save}/{k:02d}.jpg")
    return out

# our tile model: x = 8 + tx*29.1, y = 143 + ty*22.6; their crop = 18 x 32 tiles
base = [8, 143, 8 + 18 * 29.1, 143 + 32 * 22.6]
for want in ("v8/", "v9/", "v10/"):
    p = [f for f in pts if f.startswith(want) and f.endswith("best.pt")]
    if not p:
        p = [f for f in pts if want.strip("/") in f]
    if not p:
        continue
    m = YOLO(hf_hub_download("SudhishNeel/clash-royale-detector", p[0]))
    print("MODEL", p[0], "classes", len(m.names), flush=True)
    best = None
    for dy0 in (-40, -20, 0, 20, 40):
        for dy1 in (-40, -20, 0, 20, 40):
            for dx in (-10, 0, 10):
                box = [base[0] - dx, base[1] + dy0, base[2] + dx, base[3] + dy1]
                o = run(m, box, conf=0.4)
                score = sum(t for _, t, _, _ in o) * 10 + sum(c for _, _, c, _ in o)
                if best is None or score > best[0]:
                    best = (score, box)
    print("BEST CROP", [round(v) for v in best[1]], "score", round(best[0], 1), flush=True)
    for conf in (0.25, 0.5):
        o = run(m, best[1], conf=conf, save=f"{W}/eval_{want.strip('/')}_{conf}" if conf == 0.25 else None)
        print(f"{want} conf>={conf} units/frame", [u for u, _, _, _ in o], "mean", round(np.mean([u for u, _, _, _ in o]), 2),
              "towers/frame", round(np.mean([t for _, t, _, _ in o]), 2), flush=True)
    o = run(m, best[1], conf=0.25)
    from collections import Counter
    print(want, "top classes", Counter(x for *_, n in o for x in n if "tower" not in x).most_common(30), flush=True)
    json.dump({"crop_on_540x1200": [round(v, 1) for v in best[1]], "model": p[0]}, open(f"{W}/crop_{want.strip('/')}.json", "w"))
