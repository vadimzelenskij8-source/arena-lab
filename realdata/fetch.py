"""Download the user's real match frames (Roboflow upload batches) for pseudo-labelling."""
import json, os, sys, urllib.request, concurrent.futures as cf
KEY = os.environ["ROBOFLOW_API_KEY"]
PROJECT = "clash-royale-bot-tzfwl"
BATCHES = sys.argv[2].split(",")
out = sys.argv[1]
os.makedirs(out, exist_ok=True)


def call(url, body=None):
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body else None,
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=60))


ws = call(f"https://api.roboflow.com/?api_key={KEY}")["workspace"]
print("workspace ok")
items = []
for b in BATCHES:
    off = 0
    while True:
        r = call(f"https://api.roboflow.com/{ws}/{PROJECT}/search?api_key={KEY}",
                 {"batch_id": b, "limit": 250, "offset": off, "fields": ["id", "name", "owner"]})
        res = r.get("results", [])
        items += [(b, x) for x in res]
        off += len(res)
        if not res or off >= r.get("total", 0):
            break
    print("batch", b, off)


def get(t):
    b, x = t
    url = f"https://source.roboflow.com/{x['owner']}/{x['id']}/original.jpg"
    dst = f"{out}/{b}_{x['name'] if x.get('name') else x['id']}"
    if not dst.endswith(".jpg"):
        dst += ".jpg"
    try:
        urllib.request.urlretrieve(url, dst)
        return 1
    except Exception as e:
        return 0


with cf.ThreadPoolExecutor(16) as ex:
    print("downloaded", sum(ex.map(get, items)), "of", len(items))
