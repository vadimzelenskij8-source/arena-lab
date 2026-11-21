"""Collect real decks from the official Clash Royale API (via the RoyaleAPI proxy).

Key comes only from the CR_API_KEY env var (GitHub secret); it is never printed.
Usage: python crapi.py --mode test|collect --players 1000 --out decks_api.json
"""
import argparse, json, os, re, sys, time, urllib.request, urllib.error, urllib.parse
from collections import Counter
from pathlib import Path

BASE = "https://proxy.royaleapi.dev/v1"
ROOT = Path(__file__).resolve().parent
POOL = {c["key"] for c in json.load(open(ROOT / "card_pool.json", encoding="utf-8"))}


def slug(name):
    s = name.lower().replace(".", "").replace("'", "")
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-")


def get(path, key, tries=4):
    url = BASE + path
    for t in range(tries):
        req = urllib.request.Request(url, headers={"Authorization": "Bearer " + key, "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return r.status, json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="ignore")[:200]
            if e.code == 429 or e.code >= 500:
                time.sleep(1.5 * (t + 1)); continue
            return e.code, {"error": body}
        except Exception as e:
            time.sleep(1.5 * (t + 1)); last = str(e)
    return -1, {"error": "network"}


def ranking(key, n):
    tags = []
    for path in ["/locations/global/pathoflegend/players", "/locations/global/rankings/players"]:
        code, js = get(f"{path}?limit={min(n, 1000)}", key)
        print(f"ranking {path}: http {code}, items {len(js.get('items', [])) if isinstance(js, dict) else 0}")
        if code == 200:
            tags += [p["tag"] for p in js.get("items", [])]
        if len(tags) >= n:
            break
    return list(dict.fromkeys(tags))[:n]


def deck_of(side):
    cards = side.get("cards") or []
    if len(cards) != 8:
        return None
    keys = [slug(c["name"]) for c in cards]
    evo = [k for k, c in zip(keys, cards) if c.get("evolutionLevel", 0) > 0]
    return keys, evo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="test")
    ap.add_argument("--players", type=int, default=1000)
    ap.add_argument("--out", default="decks_api.json")
    ap.add_argument("--merge", default="")
    a = ap.parse_args()
    key = os.environ.get("CR_API_KEY", "").strip()
    if not key:
        print("CR_API_KEY is empty: secret not set"); sys.exit(1)
    code, js = get("/cards", key)
    print(f"cards endpoint: http {code}" + (f", {len(js.get('items', []))} cards" if code == 200 else f", {js}"))
    if code != 200:
        sys.exit(1)
    api_cards = [slug(c["name"]) for c in js["items"]]
    missing = sorted(set(api_cards) - POOL)
    print(f"api cards {len(api_cards)}, in sim pool {len(set(api_cards) & POOL)}, not in sim: {missing}")
    tags = ranking(key, 20 if a.mode == "test" else a.players)
    decks, unsupported, modes = {}, Counter(), Counter()
    if a.merge and Path(a.merge).exists():
        for e in json.load(open(a.merge, encoding="utf-8")):
            if all(k in POOL for k in e["cards"]):
                decks[tuple(sorted(e["cards"]))] = {"cards": e["cards"], "evo": e.get("evo", []), "n": e.get("n", 1), "trophies": e.get("trophies", 0)}
        print(f"merged {len(decks)} decks from previous days")
    battles = 0
    for i, tag in enumerate(tags):
        code, log = get("/players/" + urllib.parse.quote(tag) + "/battlelog", key)
        if code != 200 or not isinstance(log, list):
            continue
        for b in log:
            modes[b.get("type", "?")] += 1
            if len(b.get("team", [])) != 1:  # 1v1 only
                continue
            battles += 1
            for side in (b["team"][0], b["opponent"][0]):
                d = deck_of(side)
                if not d:
                    continue
                keys, evo = d
                bad = [k for k in keys if k not in POOL]
                if bad:
                    unsupported.update(bad); continue
                sig = tuple(sorted(keys))
                e = decks.setdefault(sig, {"cards": keys, "evo": evo, "n": 0, "trophies": 0})
                e["n"] += 1
                e["trophies"] = max(e["trophies"], side.get("startingTrophies") or 0)
        if i % 100 == 0:
            print(f"players {i}/{len(tags)}, battles {battles}, unique decks {len(decks)}", flush=True)
        time.sleep(0.03)
    out = sorted(decks.values(), key=lambda e: -e["n"])
    for k, e in enumerate(out):
        e.update(name=f"api {k}", group="api", src="api", hero=[])
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"DONE players {len(tags)}, 1v1 battles {battles}, unique supported decks {len(out)}")
    print("battle types:", dict(modes.most_common(8)))
    print("unsupported cards (skipped decks):", dict(unsupported.most_common(15)))


if __name__ == "__main__":
    main()
