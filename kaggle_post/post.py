# Kaggle kernel: export the trained detector for the phone and check it on real frames.
import glob, json, os, shutil, subprocess, sys, urllib.request
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics>=8.3", "onnx2tf", "tf_keras", "onnx_graphsurgeon", "sng4onnx", "onnxslim"], check=False)
from ultralytics import YOLO
pt = (glob.glob("/kaggle/input/**/best.pt", recursive=True) or [None])[0]
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
