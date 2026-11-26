# Fetch SudhishNeel's phone-frame detector (YOLOv8n v8) + the arena crop fitted to our frames.
import json, shutil, subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"], check=False)
from huggingface_hub import hf_hub_download
shutil.copy(hf_hub_download("SudhishNeel/clash-royale-detector", "v8/best.pt"), "/kaggle/working/best.pt")
json.dump({"crop_on_540x1200": [8, 103, 532, 886], "imgsz": 800, "crop_size": [568, 896]}, open("/kaggle/working/crop.json", "w"))
print("ok")
