"""CI: download everything battle-related from the game CDN.

1. fingerprint + csv_logic/csv_client/legacy_vfx (decompressing LZMA tables)
2. catalog.json for the bot's card pool
3. every .sc referenced by the catalog / effects / emitters (+ its .sctx textures), towers, arenas, battle UI
4. every sc3d folder referenced by a PrefabAsset of a pool card (models, animations, textures)
"""
import concurrent.futures as cf
import json
import lzma
import os
import re
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import catalog  # noqa: E402

CDN, SHA = os.environ["CDN"], os.environ["SHA"]
OUT = sys.argv[1] if len(sys.argv) > 1 else "assets"


def get(f):
    dst = os.path.join(OUT, f)
    if os.path.exists(dst):
        return f, os.path.getsize(dst)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    for attempt in range(4):
        try:
            b = urllib.request.urlopen(f"{CDN}/{SHA}/{f}", timeout=120).read()
            open(dst, "wb").write(b)
            return f, len(b)
        except Exception as e:  # noqa: BLE001
            err = e
    return f, "ERR %s" % err


def fetch(files):
    with cf.ThreadPoolExecutor(16) as ex:
        res = list(ex.map(get, sorted(set(files))))
    bad = [r for r in res if isinstance(r[1], str)]
    print("fetched", len(res), "failed", len(bad), bad[:5], flush=True)
    return res


def unlzma(p):
    b = open(p, "rb").read()
    if b[:1] == b"]":
        d = lzma.LZMADecompressor(lzma.FORMAT_ALONE).decompress(b[:9] + b"\x00" * 4 + b[9:])
        open(p, "wb").write(d)


def main():
    fp = json.load(urllib.request.urlopen(f"{CDN}/{SHA}/fingerprint.json", timeout=60))
    files = [f["file"] for f in fp["files"]]
    os.makedirs(OUT, exist_ok=True)
    json.dump(fp, open(os.path.join(OUT, "fingerprint.json"), "w"))
    tables = [f for f in files if f.startswith(("csv_logic/", "csv_client/", "legacy_vfx/"))]
    fetch(tables)
    for f in tables:
        if f.endswith((".csv", ".toml")):
            unlzma(os.path.join(OUT, f))
    here = os.path.dirname(os.path.abspath(__file__))
    cat = catalog.main(OUT, os.path.join(here, "card_pool.json"), os.path.join(OUT, "catalog.json"))

    sc_needed = set()
    effects_needed = set()
    prefabs = set()
    for e in cat.values():
        for part in ("base", "evo", "hero"):
            for v in e.get(part, []):
                for k in ("FileName", "PrestigeSWF", "IndicatorFileName"):
                    if v.get(k):
                        sc_needed.add(v[k])
                effects_needed.update(x for x in v.get("effects", []) if x)
                if v.get("PrefabAsset"):
                    prefabs.add(v["PrefabAsset"])
    # effects -> SWF files + particle emitters -> particle sc files
    eff = json.load(open(os.path.join(OUT, "legacy_vfx/effects.json")))
    emi = json.load(open(os.path.join(OUT, "legacy_vfx/emitters.json")))
    seen = set()
    stack = list(effects_needed)
    while stack:
        n = stack.pop()
        if n in seen or n not in eff:
            continue
        seen.add(n)
        d = eff[n]
        for k in ("EnemyVersion", "PrestigeVersion", "PrestigeVersion2", "PrestigeVersion3"):
            if d.get(k):
                stack.append(d[k])
        for rc in d.get("RenderableComponents", []):
            if rc.get("FileName"):
                sc_needed.add(rc["FileName"])
            if rc.get("Effect"):
                stack.append(rc["Effect"])
            em = emi.get(rc.get("ParticleEmitterName", ""))
            if em:
                res = em.get("ParticleResource")
                for r in (res if isinstance(res, list) else [res]):
                    if r:
                        sc_needed.add(r)
    json.dump(sorted(seen), open(os.path.join(OUT, "effects_used.json"), "w"))
    always = ("sc/effects", "sc/building_tower", "sc/arena", "sc/ui_battle", "sc/battle", "sc/chr_king",
              "sc/chr_princess", "sc/ui_spell", "sc/spell_")
    base = [f for f in files if f.startswith("sc/") and f.endswith(".sc")]
    for f in base:
        if f.startswith(always):
            sc_needed.add(f)
    want = set()
    stems = {re.sub(r"(_dl)?\.sc$", "", f) for f in sc_needed}
    for f in files:
        if f.startswith("sc/"):
            st = re.sub(r"(_dl)?(_\d+)?\.(sc|sctx)$", "", f)
            if st in stems:
                want.add(f)
    for f in sc_needed:
        if f in files:
            want.add(f)
        alt = f.replace(".sc", "_dl.sc")
        if alt in files:
            want.add(alt)
    dirs = {os.path.dirname(p).split("/effect")[0] for p in prefabs}  # sc3d/characters/hero/hero_knight
    for f in files:
        if f.startswith("sc3d/") and any(f.startswith(d + "/") for d in dirs) and not f.endswith(".rmat"):
            want.add(f)
    for f in files:  # shared eye textures etc.
        if f.startswith("sc3d/shared/") and f.endswith(".ktx"):
            want.add(f)
    print("sc needed", len(sc_needed), "files to fetch", len(want), "prefab dirs", sorted(dirs), flush=True)
    missing = sorted(f for f in sc_needed if f not in files and f.replace(".sc", "_dl.sc") not in files)
    print("referenced but absent on CDN:", missing[:40])
    fetch(want)
    json.dump({"sc": sorted(sc_needed), "missing": missing, "prefab_dirs": sorted(dirs)},
              open(os.path.join(OUT, "fetch_report.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
