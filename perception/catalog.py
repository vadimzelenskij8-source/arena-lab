"""Build the card -> visual assets catalog from the game's logic data (decompressed csv/toml).

usage: catalog.py <data_dir with csv_logic/ csv_client/> <card_pool.json> <out.json>

For every card of the simulator's pool (base, evolution, hero form) it walks the game's object graph
(card row -> summoned characters / buildings / projectiles / area effects / actions / buffs, and everything they
reference) and collects every visual: the .sc file plus blue / red / neutral export names, the shadow, prestige
skins and the effect names (deploy, death, hit, trail...).
"""
import csv
import glob
import json
import os
import re
import sys
import tomllib

VISUAL_KEYS = ("FileName", "BlueExportName", "RedExportName", "ExportName", "RedExportName", "NeutralExportName",
               "ShadowExportName", "BlueTopExportName", "RedTopExportName", "PrestigeSWF", "PrestigeExportName",
               "PrestigeExportName2", "PrestigeExportName3", "ExportNameTop", "ExportNameBottom", "CustomExportName",
               "IndicatorFileName", "RedShadowExportName", "BlueShadowExportName", "PrefabAsset")


SKIP_TYPES = ("SUPPORT_RARITIES", "RARITIES", "PREDEFINEDDECK", "CARD_GROUP", "VARIABLE")
SKIP_KEYS = ("EvolvedSpells", "Rarity", "Tribe", "UnlockArena", "TID", "TID_INFO", "IconFile", "AutoChessForm")


def load_csv(p):
    rows = list(csv.reader(open(p, encoding="utf-8")))
    if len(rows) < 2:
        return []
    h = rows[0]
    out = []
    last = None
    for r in rows[2:]:
        d = {h[i]: v for i, v in enumerate(r) if i < len(h) and v != ""}
        if not d.get("Name") and last is not None:  # continuation row (array values)
            for k, v in d.items():
                last.setdefault("_more_" + k, []).append(v)
            continue
        out.append(d)
        last = d
    return out


def slug(s):
    s = s.lower().replace("p.e.k.k.a", "pekka").replace(".", "")
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-")


def flat_strings(v, out):
    if isinstance(v, str):
        for part in v.split(","):
            out.add(part.strip())
    elif isinstance(v, dict):
        for x in v.values():
            flat_strings(x, out)
    elif isinstance(v, list):
        for x in v:
            flat_strings(x, out)
    return out


def main(data, pool_path, out_path):
    logic = os.path.join(data, "csv_logic")
    # ---- object index from toml (typed sections "[TYPE.Name]" and legacy top-level "[Name]")
    objs = {}  # name -> list of (type, file, section)
    for p in sorted(glob.glob(os.path.join(logic, "**", "*.toml"), recursive=True)):
        rel = os.path.relpath(p, logic)
        if rel.startswith(("events", "game_modes", "autochess")):
            continue
        try:
            d = tomllib.loads(open(p, encoding="utf-8").read())
        except Exception:
            continue
        legacy = os.path.basename(p).replace(".toml", "")
        for t, v in d.items():
            if not isinstance(v, dict):
                continue
            if t.isupper() or t in ("EXT",):
                for n, sec in v.items():
                    if isinstance(sec, dict):
                        objs.setdefault(n, []).append((t, rel, sec))
            else:  # legacy table: [Name] in characters_base.toml etc.
                objs.setdefault(t, []).append((legacy.upper(), rel, v))
    # ---- legacy csv object tables (in case the toml lacks some)
    for name in ("characters", "buildings", "projectiles", "area_effect_objects"):
        p = os.path.join(logic, name + ".csv")
        if os.path.exists(p):
            for r in load_csv(p):
                objs.setdefault(r["Name"], []).append((name.upper(), name + ".csv", r))

    texts = {}
    tp = os.path.join(data, "csv_client", "texts.csv")
    for r in csv.reader(open(tp, encoding="utf-8")):
        if len(r) > 1:
            texts[r[0]] = r[1]

    # ---- cards
    cards = {}
    for fn, kind in (("spells_characters", "troop"), ("spells_buildings", "building"), ("spells_other", "spell")):
        for r in load_csv(os.path.join(logic, fn + ".csv")):
            if r.get("NotInUse") == "true" or r.get("NotVisible") == "true":
                continue
            en = texts.get(r.get("TID", ""), r["Name"])
            cards.setdefault(slug(en), (r, kind))
    evolved = {r["Name"]: r for r in load_csv(os.path.join(logic, "spells_evolved.csv"))}
    hero = {r["Name"]: r for r in load_csv(os.path.join(logic, "spells_hero_form.csv"))}

    def walk(roots):
        seen, order = set(), []
        stack = list(roots)
        while stack:
            n = stack.pop()
            if n in seen or n not in objs:
                continue
            seen.add(n)
            order.append(n)
            for _t, _f, sec in objs[n]:
                if _t in SKIP_TYPES:
                    continue
                sec = {k: v for k, v in sec.items() if k not in SKIP_KEYS}
                for s in flat_strings(sec, set()):
                    if s in objs and s not in seen and s != n:
                        stack.append(s)
        return order

    def visuals(names):
        out = []
        for n in names:
            for t, f, sec in objs[n]:
                vis = {k: sec[k] for k in VISUAL_KEYS if isinstance(sec.get(k), str)}
                eff = sorted({v for k, v in sec.items() if isinstance(v, str) and "Effect" in k and k != "Effect"
                              and not k.endswith("ExportName")})
                if (vis or eff) and t not in SKIP_TYPES:
                    out.append({"name": n, "type": t, "src": f, "scale": sec.get("Scale"), **vis, "effects": eff})
        return out

    def card_roots(row):
        roots = set()
        for k, v in row.items():
            if k in ("Name", "TID", "TID_INFO", "IconFile", "Rarity", "Tribe", "UnlockArena", "EvolvedSpells"):
                continue
            flat_strings(v, roots)
        return roots

    pool = json.load(open(pool_path))
    alias = {"pekka": "pekka", "mini-pekka": "mini-pekka", "the-log": "the-log", "x-bow": "x-bow"}
    out = {}
    missing = []
    for c in pool:
        key = c["key"]
        hit = cards.get(alias.get(key, key))
        if hit is None:
            missing.append(key)
            continue
        row, kind = hit
        entry = {"game_name": row["Name"], "kind": kind,
                 "base": visuals(walk(card_roots(row) | {row["Name"]}))}
        evs = (row.get("EvolvedSpells") or "").split(",") + row.get("_more_EvolvedSpells", [])
        for _t, _f, sec in objs.get(row["Name"], []):
            if _t.startswith("SPELL"):
                evs += list(flat_strings(sec.get("EvolvedSpells", []), set()))
        evs = list(dict.fromkeys(e.strip() for e in evs if e.strip()))
        for ev in evs:
            r2 = evolved.get(ev) or hero.get(ev)
            tag = "hero" if ev in hero or "hero" in ev.lower() else "evo"
            roots = card_roots(r2) if r2 else set()
            roots.add(ev)
            entry.setdefault(tag, []).extend(visuals(walk(roots)))
        out[key] = entry
    json.dump({"cards": out, "missing": missing}, open(out_path, "w"), indent=1)
    files = sorted({v["FileName"] for e in out.values() for part in ("base", "evo", "hero") for v in e.get(part, [])
                    if v.get("FileName")})
    print("cards", len(out), "missing", missing)
    print("sc files", len(files))
    return out


if __name__ == "__main__":
    main(*sys.argv[1:4])
