"""Sprite bank: enumerate every renderable visual of every card and render it on demand.

An item is one animation of one object of one card form for one team, e.g.
  {card: knight, form: evo, obj: Knight_EV1, team: red, src: 2d, file: sc/chr_knight_evolution.sc,
   anim: knight_evolution_enemy_run1, dirs: {1: export, ..., 9: export}}
"""
import glob
import json
import os
import re
from functools import lru_cache

import numpy as np

from sc2 import SC2, load_ktx

ANIM_RE = re.compile(r"^(?P<anim>.*?)(?:_(?P<dir>\d{1,2}))?$")


class Bank:
    def __init__(self, assets, catalog_path=None):
        self.assets = assets
        cat = json.load(open(catalog_path or os.path.join(assets, "catalog.json")))
        self.cards = cat["cards"]
        self._sc = {}  # small LRU: decoded textures are big
        vfx = os.path.join(assets, "legacy_vfx")
        self.effects = json.load(open(os.path.join(vfx, "effects.json"))) if os.path.isdir(vfx) else {}
        self.emitters = json.load(open(os.path.join(vfx, "emitters.json"))) if os.path.isdir(vfx) else {}
        self._3d = {}

    # ------------------------------------------------------------------ files
    def sc(self, rel):
        if rel in self._sc:
            self._sc[rel] = self._sc.pop(rel)  # LRU order
        else:
            while len(self._sc) >= 12:
                self._sc.pop(next(iter(self._sc)))
            p = os.path.join(self.assets, rel)
            if not os.path.exists(p):
                alt = p.replace(".sc", "_dl.sc")
                p = alt if os.path.exists(alt) else None
            try:
                self._sc[rel] = SC2(p) if p else None
            except Exception as e:  # noqa: BLE001
                print("SC load failed", rel, e)
                self._sc[rel] = None
        return self._sc[rel]

    # ------------------------------------------------------------------ enumeration
    def items(self, card):
        e = self.cards[card]
        out = []
        seen_prefab = set()
        for form in ("base", "evo", "hero"):
            objs = e.get(form, [])
            if form != "base":  # objects already present in the base form are not evo/hero specific
                base_names = {o["name"] for o in e.get("base", [])}
                objs = [o for o in objs if o["name"] not in base_names]
            for o in objs:
                out += self._items_2d(card, form, o)
                pf = o.get("PrefabAsset") or ""
                if "_battle" in os.path.basename(pf) and pf not in seen_prefab:
                    seen_prefab.add(pf)
                    out += self._items_3d(card, form, o)
        return out

    def _items_2d(self, card, form, o):
        res = []
        f = o.get("FileName")
        if f:
            s = self.sc(f)
            if s is not None:
                blue = o.get("BlueExportName") or o.get("ExportName")
                red = o.get("RedExportName")
                prefixes = []
                if red:
                    prefixes.append(("red", red))
                if blue:
                    prefixes.append(("blue" if red else "any", blue))
                res += self._group(s, f, prefixes, card, form, o)
        if o.get("PrestigeSWF"):
            s = self.sc(o["PrestigeSWF"])
            if s is not None:
                for k in ("PrestigeExportName", "PrestigeExportName2", "PrestigeExportName3"):
                    if o.get(k):
                        res += self._group(s, o["PrestigeSWF"], [("skin", o[k])], card, form, o)
        return res

    def _group(self, s, f, prefixes, card, form, o):
        taken = set()
        res = []
        # longest prefixes first so "Knight_enemy" claims its exports before "Knight"
        for team, pre in sorted(prefixes, key=lambda x: -len(x[1])):
            anims = {}
            for name in s.exports:
                if name in taken:
                    continue
                if name == pre or name.startswith(pre + "_"):
                    rest = name[len(pre):].lstrip("_")
                    if team != "skin" and ("prestige" in rest.lower()):
                        continue
                    m = ANIM_RE.match(rest)
                    anim, d = (m.group("anim") or "main"), m.group("dir")
                    anims.setdefault(anim, {})[int(d) if d else 0] = name
                    taken.add(name)
            for anim, dirs in anims.items():
                res.append({"card": card, "form": form, "obj": o["name"], "type": o["type"], "team": team,
                            "src": "2d", "file": f, "anim": anim, "dirs": dirs, "scale": o.get("scale")})
        return res

    def _items_3d(self, card, form, o):
        pre = o["PrefabAsset"]
        d = os.path.dirname(pre)
        while "/effect" in d:
            d = d.split("/effect")[0]
        root = os.path.join(self.assets, d)
        models = sorted(glob.glob(os.path.join(root, "**", "*_rig.glb"), recursive=True))
        models = [m for m in models if "lod1" in m] or models
        key = os.path.basename(pre).replace("hero_prefab_", "").replace("evo_prefab_", "").replace("chr_prefab_", "").replace("_battle", "")
        best = [m for m in models if key.split("_")[0] in os.path.basename(m)]
        if len(models) > 1 and best:
            # sub-prefabs (turret, decoy, snowman...) have their own rig named after them
            exact = [m for m in models if key.replace("_", "") in os.path.basename(m).replace("_", "")]
            models = exact[:1] or best[:1]
        anims = sorted(a for a in glob.glob(os.path.join(root, "**", "*.glb"), recursive=True)
                       if "/anim" in a and "meta" not in os.path.basename(a))
        res = []
        for m in models:
            rel = os.path.relpath(m, self.assets)
            for a in anims:
                res.append({"card": card, "form": form, "obj": o["name"], "type": o["type"], "team": "any",
                            "src": "3d", "file": rel, "anim_file": os.path.relpath(a, self.assets),
                            "anim": os.path.basename(a)[:-4], "scale": o.get("scale")})
        return res

    def fx_items(self, card, labelled=True):
        """Effect visuals (spell areas, explosions, deploy/death puffs) of a card."""
        e = self.cards[card]
        res = []
        for form in ("base", "evo", "hero"):
            for o in e.get(form, []):
                for name in o.get("effects", []):
                    d = self.effects.get(name)
                    if not d:
                        continue
                    for team, n in (("blue", name), ("red", d.get("EnemyVersion") or name)):
                        comps = self._fx_components(n)
                        if comps:
                            res.append({"card": card, "form": form, "obj": card + ":" + name, "type": "FX",
                                        "team": team, "src": "fx", "anim": name, "comps": comps, "scale": None})
        return res

    def _fx_components(self, name, depth=0):
        d = self.effects.get(name)
        if not d or depth > 3:
            return []
        out = []
        for rc in d.get("RenderableComponents", []):
            t = rc.get("Type")
            if t == "SWF" and rc.get("FileName") and rc.get("ExportName"):
                out.append(("swf", rc["FileName"], rc["ExportName"], 1, 0, 100, 100))
            elif t == "ParticleEmitter":
                em = self.emitters.get(rc.get("ParticleEmitterName", ""))
                if not em:
                    continue
                names = em.get("ParticleExportName") or []
                files = em.get("ParticleResource") or []
                names = names if isinstance(names, list) else [names]
                files = files if isinstance(files, list) else [files]
                for k, nm in enumerate(names):
                    f = files[k] if k < len(files) else (files[0] if files else None)
                    if f:
                        out.append(("pe", f, nm, max(1, int(em.get("ParticleCount", 1)) // max(1, len(names))),
                                    em.get("ParticleStartXYAreaRadius", 0) or 0, em.get("StartScale", 100),
                                    em.get("EndScale", 100)))
            elif t == "Effect" and rc.get("Effect"):
                out += self._fx_components(rc["Effect"], depth + 1)
        return out

    def render_fx(self, item, rng, scale=1.0, progress=None):
        """Compose an effect: SWF layers + scattered particles. Returns (RGBA, origin)."""
        from PIL import Image
        p = rng.random() if progress is None else progress
        layers = []
        for kind, f, nm, cnt, rad, s0, s1 in item["comps"]:
            s = self.sc(f)
            if s is None or nm not in s.exports:
                continue
            clip = s.exports[nm]
            fr = int(p * max(1, len(clip.frames) - 1))
            if kind == "swf":
                img, (x0, y0) = s.render(clip, fr, scale=scale)
                if img is not None:
                    layers.append((img, -x0, -y0))
            else:
                for _ in range(min(cnt, 25)):
                    sc = scale * (s0 + (s1 - s0) * p) / 100.0 * rng.uniform(0.7, 1.2)
                    img, (x0, y0) = s.render(clip, rng.randrange(max(1, len(clip.frames))), scale=max(sc, 0.05))
                    if img is None:
                        continue
                    a = rng.uniform(0, 2 * np.pi)
                    r = rad * scale * np.sqrt(rng.random())
                    layers.append((img, -x0 + r * np.cos(a), -y0 + r * np.sin(a) * 0.75))
        if not layers:
            return None, (0, 0)
        xs0 = min(-ox for _, ox, oy in layers)
        ys0 = min(-oy for _, ox, oy in layers)
        xs1 = max(im.shape[1] - ox for im, ox, oy in layers)
        ys1 = max(im.shape[0] - oy for im, ox, oy in layers)
        W, H = int(xs1 - xs0) + 1, int(ys1 - ys0) + 1
        if W * H > 1500 * 1500:
            return None, (0, 0)
        canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        for im, ox, oy in layers:
            canvas.alpha_composite(Image.fromarray(im), (int(-ox - xs0), int(-oy - ys0)))
        return np.asarray(canvas), (int(-xs0), int(-ys0))

    # ------------------------------------------------------------------ rendering
    def frames(self, item):
        if item["src"] == "2d":
            s = self.sc(item["file"])
            clip = s.exports[next(iter(item["dirs"].values()))]
            return max(1, len(clip.frames))
        return 12

    def render(self, item, frame=0, direction=None, scale=1.0, yaw=None):
        """Returns (RGBA uint8, (ox, oy) = pixel of the object's origin inside the image)."""
        if item["src"] == "2d":
            s = self.sc(item["file"])
            dirs = item["dirs"]
            k = direction if direction in dirs else next(iter(dirs))
            img, (x0, y0) = s.render(s.exports[dirs[k]], frame, scale=scale)
            return img, (-x0, -y0)
        from render3d import camera, render
        model, anim, tex = self._load3d(item["file"], item["anim_file"])
        t = anim.duration * (frame % 12) / 12.0
        img, org = render(model.pose(anim, t), tex, camera(45, yaw if yaw is not None else 0.0),
                          px_per_unit=30.0 * scale)
        return img, org

    def _load3d(self, model_rel, anim_rel):
        from render3d import Anim, Model
        from sc3d import convert
        key = (model_rel, anim_rel)
        if key not in self._3d:
            if len(self._3d) > 6:
                self._3d.clear()
            cache = os.path.join(self.assets, "_conv")
            mp = convert(os.path.join(self.assets, model_rel), os.path.join(cache, model_rel))
            ap = convert(os.path.join(self.assets, anim_rel), os.path.join(cache, anim_rel))
            m = Model(mp)
            tex = {}
            for mi in range(len(m.materials)):
                tp = m.material_texture(mi)
                if tp and os.path.exists(os.path.join(self.assets, tp)):
                    try:
                        tex[mi] = load_ktx(open(os.path.join(self.assets, tp), "rb").read())
                    except Exception as e:  # noqa: BLE001
                        print("ktx", tp, e)
            self._3d[key] = (m, Anim(ap), tex)
        return self._3d[key]
