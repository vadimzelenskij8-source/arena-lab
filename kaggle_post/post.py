# Kaggle kernel: export the trained detector for the phone and check it on real frames.
import glob, json, os, shutil, subprocess, sys, urllib.request
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics>=8.3", "onnx2tf", "tf_keras", "onnx", "onnxruntime", "onnx_graphsurgeon", "sng4onnx", "onnxslim"], check=False)
from ultralytics import YOLO
pt = (glob.glob("/kaggle/input/**/best.pt", recursive=True) or [None])[0]
for f in glob.glob("/kaggle/input/**/results.csv", recursive=True):
    print("RESULTS", open(f).read().splitlines()[-1], flush=True)
print("weights", pt)
W = "/kaggle/working"
ids = ['79LEx1kfj99LhfOsxqMR', 'pLFr8uEHuY5UdNBKSwaL', 'QNm6GEBGqPTbGHDrtg2p', 'VmuXAoaJxM3qZOTw2AR6', 'ZRVvEPQ8gYlOSWJVzSQ2', 'LbWYv11tdLfwK1LCCVSL', '03agofG7WQvosy92uxN4', '06U38jUQWN0TtWgjv54T', '08rfZf0EO9Ay3IIJuQEc', '0940ijpbbLl1Xr6EXPI8', '09IYML7ALVHwk02kzMul', '0DU9egwL0Wp35QUgzYpv', '0Dti7xFRSgjbl6gKYNKV', '0FMbqBe1RaMr4F0g01oX', '0IzHEpM6RXcuGmTTQjNg', '0KKfBVzQUcl7IHFbG9qH', '0MPHIYrOTaz08DN2sXYx', '0Vr2hpYNLYQYvJyzmG7r', '0WFiKk1iAUSShBwxsXA2', '0XxgMeqyNIUPYc1pLVxN', '0bVqslExai5ZHRkDJ8T9', '0cKSxzBbRPNtyHrQfRf2', '0kipun3NSbXC3oqjqUDZ', '0vva10j0JqBLLU8QQwwR', '10POmYExCEQWLzEJ9bEn', '13fePkwuT4OD4tJWGvp6', '1EI2082msrzcg2P2OXsf', '1Je2BIqZ4TmRLJyXKa6j', '1LzL9u1Y8Xvql1vyXQRf', '1PWAg3MNi7J2gJsYfLYP']
os.makedirs(f"{W}/frames", exist_ok=True)
for i in ids:
    try:
        urllib.request.urlretrieve(f"https://source.roboflow.com/03WjEzRyYIPwtXk7A2aqTNAuk4X2/{i}/original.jpg", f"{W}/frames/{i}.jpg")
    except Exception as e:
        print("frame", i, e)
shutil.copy(pt, f"{W}/det.pt")
m = YOLO(f"{W}/det.pt")
names = m.names
json.dump([names[i] for i in range(len(names))], open(f"{W}/classes.json", "w"))
for f in glob.glob("/kaggle/input/**/results.csv", recursive=True):
    print(open(f).read()[-1500:])
os.makedirs(f"{W}/eval", exist_ok=True)
fr = sorted(glob.glob(f"{W}/frames/*.jpg"))
def phone_export():
    # Phone model: full 540x1200 frame in, arena crop + letterbox inside the graph, boxes normalized to the full frame.
    from ultralytics.nn.modules import Detect
    FH, FW = 1200, 540
    cw, chh = C[2] - C[0], C[3] - C[1]
    tw, th = CS if CS else (cw, chh)
    nh, nw, IH, IW = lb(tw, th)
    pl = (IW - nw) // 2
    print("letterbox", nh, nw, IH, IW, pl, flush=True)
    # Same converter ultralytics uses now (LiteRT, torch -> .tflite directly), applied to a torch wrapper:
    # full frame in (NCHW, the app accepts it), arena crop + letterbox inside, boxes normalized to the full frame.
    import inspect
    from ultralytics.nn.modules import Detect
    fmt = "litert" if "litert" in inspect.getsource(Detect) else "tflite"
    net = YOLO(f"{W}/det.pt").model.float().fuse().eval()
    for mod in net.modules():
        if isinstance(mod, Detect):
            mod.export, mod.format, mod.dynamic = True, fmt, False
    nc4 = 4 + len(names)
    sc = torch.ones(1, nc4, 1); of = torch.zeros(1, nc4, 1)
    sc[0, 0, 0] = IW / nw * cw / FW; of[0, 0, 0] = (C[0] - pl / nw * cw) / FW
    sc[0, 2, 0] = IW / nw * cw / FW
    sc[0, 1, 0] = chh / FH; of[0, 1, 0] = C[1] / FH
    sc[0, 3, 0] = chh / FH
    with torch.no_grad():
        yt = net(torch.zeros(1, 3, IH, IW)); yt = yt[0] if isinstance(yt, (list, tuple)) else yt
        norm = float(yt[0, :4].abs().max()) <= 1.5
    print("detect format", fmt, "normalized", norm, flush=True)
    if not norm:  # boxes came in input pixels: normalize them first
        sc[0, 0, 0] /= IW; sc[0, 2, 0] /= IW; sc[0, 1, 0] /= IH; sc[0, 3, 0] /= IH

    class Wrap(nn.Module):
        def __init__(s):
            super().__init__(); s.net = net; s.register_buffer("sc", sc); s.register_buffer("of", of)
        def forward(s, x):
            c = x[:, :, C[1]:C[3], C[0]:C[2]]
            c = F.interpolate(c, size=(nh, nw), mode="bilinear", align_corners=False)
            c = F.pad(c, (pl, IW - nw - pl, 0, IH - nh), value=114 / 255)
            y = s.net(c)
            if isinstance(y, (list, tuple)): y = y[0]
            return y * s.sc + s.of

    wm = Wrap().eval(); dummy = torch.zeros(1, 3, FH, FW)
    with torch.no_grad():
        print("wrap out", tuple(wm(dummy).shape), flush=True)
    try:
        import litert_torch as aet
    except ImportError:
        import ai_edge_torch as aet
    aet.convert(wm, (dummy,)).export(f"{W}/model.tflite")
    with zipfile.ZipFile(f"{W}/model.tflite", "a", zipfile.ZIP_DEFLATED) as z:
        z.writestr("metadata.json", json.dumps({"names": {str(i): names[i] for i in range(len(names))}, "crop_on_540x1200": C}))
    print("tflite", os.path.getsize(f"{W}/model.tflite"), flush=True)
    # Verify exactly like the app: stretch full frame to the input, run, decode, compare with the .pt on the crop.
    import tensorflow as tf
    it = tf.lite.Interpreter(model_path=f"{W}/model.tflite"); it.allocate_tensors()
    ip, op = it.get_input_details()[0], it.get_output_details()[0]
    print("TFL in", ip["shape"], "out", op["shape"], flush=True)
    sh = ip["shape"]; nchw = sh[1] == 3
    ih_, iw_ = (sh[2], sh[3]) if nchw else (sh[1], sh[2])
    def iou(a, b):
        iw = min(a[2], b[2]) - max(a[0], b[0]); ih = min(a[3], b[3]) - max(a[1], b[1])
        if iw <= 0 or ih <= 0: return 0.0
        i = iw * ih; return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)
    tot = hit = extra = 0
    sx, sy = cw / tw, chh / th
    for f, r in zip(fr, m.predict(crops, imgsz=IMG, conf=0.3, verbose=False)):
        a = np.asarray(Image.open(f).convert("RGB").resize((iw_, ih_), Image.BILINEAR), dtype=np.float32) / 255
        a = a.transpose(2, 0, 1)[None] if nchw else a[None]
        it.set_tensor(ip["index"], a); it.invoke(); o = it.get_tensor(op["index"])[0]
        if o.shape[0] > o.shape[1]: o = o.T
        cls = o[4:].argmax(0); sco = o[4:].max(0); k = np.where(sco >= 0.3)[0]
        k = k[np.argsort(-sco[k])][:300]
        td = []
        for j in k:
            cx, cy, w, h = o[:4, j]; b = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            if all(not (c == cls[j] and iou(b, bb) > 0.5) for c, bb, _ in td): td.append((cls[j], b, sco[j]))
        pd = []
        for c, (x1, y1, x2, y2) in zip(r.boxes.cls.tolist(), r.boxes.xyxy.tolist()):
            pd.append((int(c), ((x1 * sx + C[0]) / FW, (y1 * sy + C[1]) / FH, (x2 * sx + C[0]) / FW, (y2 * sy + C[1]) / FH)))
        for c, b in pd:
            tot += 1; hit += any(c == tc and iou(b, tb) > 0.6 for tc, tb, _ in td)
        extra += sum(1 for tc, tb, _ in td if not any(c == tc and iou(b, tb) > 0.6 for c, b in pd))
    print("VERIFY pt-dets", tot, "matched-by-tflite", hit, "tflite-extra", extra, "OK" if tot and hit >= 0.85 * tot else "MISMATCH", flush=True)

cj = (glob.glob("/kaggle/input/**/crop.json", recursive=True) or [None])[0]
if cj:
    import zipfile
    import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
    from PIL import Image
    CFG = json.load(open(cj))
    C = [int(round(v)) for v in CFG["crop_on_540x1200"]]
    IMG = int(CFG.get("imgsz", 832)); CS = CFG.get("crop_size")  # CS: (w, h) the crop is resized to before the model

    def lb(tw, th):
        r_ = min(IMG / th, IMG / tw); nh, nw = round(th * r_), round(tw * r_)
        return nh, nw, nh + (-nh) % 32, nw + (-nw) % 32
    print("CROP", C, flush=True)
    crops = []
    for f in fr:
        im = Image.open(f).convert("RGB").resize((540, 1200))
        d = f"{W}/crops/{os.path.basename(f)}"; os.makedirs(os.path.dirname(d), exist_ok=True)
        cr = im.crop(C)
        if CS: cr = cr.resize(tuple(CS), Image.BILINEAR)
        cr.save(d, quality=95); crops.append(d)
    rr = m.predict(crops, imgsz=IMG, conf=0.05, verbose=False)
    n = [sum(1 for c in r.boxes.cls if "tower" not in names[int(c)]) for r in rr]
    top = sorted(((float(s), names[int(c)]) for r in rr for c, s in zip(r.boxes.cls, r.boxes.conf) if "tower" not in names[int(c)]), reverse=True)[:15]
    print("IMGSZ crop832 non-tower dets per frame", n, "top", top, flush=True)
    for th in (0.25, 0.5):
        print("CROP conf>=%.2f non-tower per frame" % th, [sum(1 for c, s in zip(r.boxes.cls, r.boxes.conf) if s >= th and "tower" not in names[int(c)]) for r in rr], flush=True)
    stats = []
    for r in m.predict(crops, imgsz=IMG, conf=0.25, verbose=False):
        r.save(filename=f"{W}/eval/{os.path.basename(r.path)}")
        stats.append({"frame": os.path.basename(r.path), "dets": [[names[int(c)], round(float(s), 2)] for c, s in zip(r.boxes.cls, r.boxes.conf)]})
    json.dump(stats, open(f"{W}/eval.json", "w"), indent=0)
    try:
        _tw, _th = CS if CS else (C[2] - C[0], C[3] - C[1])
        m.export(format="tflite", imgsz=lb(_tw, _th)[2:])
    except Exception as e:
        print("plain export failed", e, flush=True)
    try:
        phone_export()
    except Exception:
        import traceback; traceback.print_exc(); print("PHONE EXPORT FAILED", flush=True)
    shutil.rmtree(f"{W}/frames", ignore_errors=True); shutil.rmtree(f"{W}/crops", ignore_errors=True)
    print("done")

if not cj:
    for sz in [(832, 384), 832, 1216, 640]:
        rr = m.predict(fr, imgsz=sz, conf=0.05, verbose=False)
        n = [sum(1 for c in r.boxes.cls if "tower" not in names[int(c)]) for r in rr]
        top = sorted(((float(s), names[int(c)]) for r in rr for c, s in zip(r.boxes.cls, r.boxes.conf) if "tower" not in names[int(c)]), reverse=True)[:15]
        print("IMGSZ", sz, "non-tower dets per frame", n, "top", top, flush=True)
    res = m.predict(fr, imgsz=(832, 384), conf=0.1, verbose=False)
    stats = []
    for r in res:
        r.save(filename=f"{W}/eval/{os.path.basename(r.path)}")
        stats.append({"frame": os.path.basename(r.path), "dets": [[names[int(c)], round(float(s), 2)] for c, s in zip(r.boxes.cls, r.boxes.conf)]})
    json.dump(stats, open(f"{W}/eval.json", "w"), indent=0)
    try:
        f = m.export(format="tflite", imgsz=(832, 384))
        t = glob.glob(os.path.join(os.path.dirname(f), "*float16.tflite")) or [f]
        shutil.copy(t[0], f"{W}/model.tflite")
        print("tflite", os.path.getsize(f"{W}/model.tflite"))
    except Exception as e:
        print("tflite export failed", e)
    shutil.rmtree(f"{W}/frames", ignore_errors=True)
    print("done")
