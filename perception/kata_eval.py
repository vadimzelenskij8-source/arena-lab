"""Ground-truth exam for detectors: KataCR's hand-labelled real frames (wty-yy/Clash-Royale-Detection-Dataset,
images/part2) mapped to SudhishNeel class names; ultralytics val -> mAP50 / P / R per model."""
import glob, json, os, re, sys, urllib.request
from collections import Counter
from pathlib import Path
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
from ultralytics import YOLO
K = sys.argv[1]  # unpacked dataset root
lab = urllib.request.urlopen("https://raw.githubusercontent.com/wty-yy/KataCR/master/katacr/constants/label_list.py").read().decode()
kn = re.findall(r"'([a-z0-9\-]+)'", lab[lab.index("unit_list"):]) if "unit_list" in lab else re.findall(r"'([a-z0-9\-]+)'", lab)
print("KATA label_list head", lab[:600].replace("\n", " | "))
RID = "SudhishNeel/clash-royale-detection"
pf = pq.ParquetFile(hf_hub_download(RID, "synthetic_arena/train-c001a-00000-of-00002.parquet", repo_type="dataset"))
names = json.loads(pf.schema_arrow.metadata[b"huggingface"])["info"]["features"]["objects"]["feature"]["category"]["names"]
idx = {n: i for i, n in enumerate(names)}
txts = sorted(glob.glob(f"{K}/images/part2/**/*.txt", recursive=True))
print("label files", len(txts)); print("SAMPLE", txts[0], open(txts[0]).read()[:300])
