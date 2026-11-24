"""Synthetic battle screenshots with exact labels, rendered from the game's own assets.

usage: synth.py <assets> <backgrounds dir> <out dir> <n images> <seed> [classes.json]

Output (YOLO format): out/images/*.jpg, out/labels/*.txt, out/classes.json, out/preview_*.jpg
Screen space = the collector's 540x1200 portrait frames (the bot side is always the bottom).
"""
import glob
import json
import math
import os
import random
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bank import Bank  # noqa: E402

W, H = 540, 1200
# arena tile (0..18, 0..32) -> screen, fitted on the towers of the real frames
X0, PX = 8.0, 29.1
Y0, PY = 143.0, 22.6
KING_TOWER_PX = 100.0  # width of the king tower body on a real 540px frame


def tile_to_px(tx, ty):
    return X0 + tx * PX, Y0 + ty * PY


def class_name(it):
    team = it["team"] if it["team"] in ("red", "blue") else ""
    return it["obj"] + ("|" + team if team else "")


class Synth:
    def __init__(self, assets, bg_dir, seed=0):
        self.rng = random.Random(seed)
        self.bank = Bank(assets)
        self.items = []
        for card in self.bank.cards:
            try:
                its = self.bank.items(card)
            except Exception as e:  # noqa: BLE001
                print("items failed", card, e)
                continue
            self.items += [i for i in its if i["team"] != "skin"]
        self.skins = {}
        for card in self.bank.cards:
            for i in self.bank.items(card):
                if i["team"] == "skin":
                    self.skins.setdefault((i["obj"], i["anim"]), []).append(i)
        self.fx = {}
        for card, e in self.bank.cards.items():
            try:
                self.fx[card] = self.bank.fx_items(card)
            except Exception as ex:  # noqa: BLE001
                print("fx failed", card, ex)
        self.spell_cards = [c for c, e in self.bank.cards.items() if e["kind"] == "spell" and self.fx.get(c)]
        self.by_card = {}
        for i in self.items:
            self.by_card.setdefault(i["card"], []).append(i)
        self.classes = sorted({class_name(i) for i in self.items} | {c + "|spell" for c in self.spell_cards} | {
            "king_tower|red", "king_tower|blue", "princess_tower|red", "princess_tower|blue"})
        self.cid = {c: k for k, c in enumerate(self.classes)}
        self.bgs = [Image.open(p).convert("RGB").resize((W, H)) for p in sorted(glob.glob(os.path.join(bg_dir, "*.jpg")))]
        self.k = self._calibrate()
        print("items", len(self.items), "classes", len(self.classes), "bgs", len(self.bgs), "K", round(self.k, 3))

    def _calibrate(self):
        s = self.bank.sc("sc/building_tower.sc")
        try:
            img, _ = s.render(s.exports["KingTower_red"], 0)
            a = img[:, :, 3] > 64
            cols = np.where(a.any(0))[0]
            return KING_TOWER_PX / max(1, cols[-1] - cols[0])
        except Exception as e:  # noqa: BLE001
            print("calibration failed", e)
            return 0.5

    def median_bg(self):
        sub = self.rng.sample(self.bgs, min(len(self.bgs), 9))
        return Image.fromarray(np.median(np.stack([np.asarray(b) for b in sub]), 0).astype(np.uint8))

    def pick(self):
        card = self.rng.choice(list(self.by_card))
        its = self.by_card[card]
        forms = sorted({i["form"] for i in its})
        form = self.rng.choice(forms) if self.rng.random() < 0.5 else "base"
        cand = [i for i in its if i["form"] == form] or its
        return self.rng.choice(cand)

    def sprite(self, it, team_flip=False):
        n = self.bank.frames(it)
        frame = self.rng.randrange(n)
        dirs = sorted(it.get("dirs", {0: 0}))
        d = self.rng.choice(dirs)
        sc = it.get("scale") or 100
        sc = sc[0] if isinstance(sc, list) and sc else sc
        try:
            sc = float(str(sc).split(",")[0]) / 100.0
        except ValueError:
            sc = 1.0
        sc = sc if 0.05 < sc < 5 else 1.0
        scale = self.k * sc * self.rng.uniform(0.9, 1.1)
        yaw = self.rng.uniform(0, 360)
        try:
            img, org = self.bank.render(it, frame, direction=d, scale=scale, yaw=yaw)
        except Exception as e:  # noqa: BLE001
            print("render failed", it["obj"], it.get("anim"), repr(e)[:120])
            self.items.remove(it) if it in self.items else None
            self.by_card[it["card"]] = [i for i in self.by_card[it["card"]] if i is not it] or self.by_card[it["card"]]
            return None
        if img is None or img.size == 0:
            return None
        im = Image.fromarray(img)
        sk = self.skins.get((it["obj"], it["anim"]))
        if sk and it["src"] == "2d" and self.rng.random() < 0.35:
            try:
                ov, oorg = self.bank.render(self.rng.choice(sk), frame, direction=d, scale=scale)
                if ov is not None and ov.size:
                    ovi = Image.fromarray(ov)
                    x0 = min(-org[0], -oorg[0]); y0 = min(-org[1], -oorg[1])
                    x1 = max(im.width - org[0], ovi.width - oorg[0]); y1 = max(im.height - org[1], ovi.height - oorg[1])
                    c = Image.new("RGBA", (int(x1 - x0), int(y1 - y0)), (0, 0, 0, 0))
                    c.alpha_composite(im, (int(-org[0] - x0), int(-org[1] - y0)))
                    c.alpha_composite(ovi, (int(-oorg[0] - x0), int(-oorg[1] - y0)))
                    im, org = c, (int(-x0), int(-y0))
            except Exception:  # noqa: BLE001
                pass
        if it["src"] == "2d" and self.rng.random() < 0.5 and d not in (0,):
            im = im.transpose(Image.FLIP_LEFT_RIGHT)  # the game mirrors directions for the other half
            org = (im.width - org[0], org[1])
        return im, org

    def make(self, n_obj=None):
        bg = self.median_bg() if self.rng.random() < 0.93 else self.rng.choice(self.bgs).copy()
        bg = ImageEnhance.Brightness(bg).enhance(self.rng.uniform(0.55, 1.15))
        bg = ImageEnhance.Color(bg).enhance(self.rng.uniform(0.8, 1.2))
        canvas = bg.convert("RGBA")
        boxes = []
        # towers already in the background
        for name, team, (cx, cy, w, h) in (("king_tower", "red", (270, 205, 110, 120)),
                                            ("princess_tower", "red", (110, 285, 90, 105)),
                                            ("princess_tower", "red", (430, 285, 90, 105)),
                                            ("king_tower", "blue", (270, 810, 110, 120)),
                                            ("princess_tower", "blue", (110, 720, 90, 105)),
                                            ("princess_tower", "blue", (430, 720, 90, 105))):
            boxes.append([self.cid[f"{name}|{team}"], cx, cy, w, h])
        n_obj = n_obj if n_obj is not None else self.rng.choice([1, 3, 5, 8, 12, 16, 24])
        placed = []
        for _ in range(n_obj):
            it = self.pick()
            r = self.sprite(it)
            if r is None:
                continue
            im, (ox, oy) = r
            if im.width > W or im.height > H:
                continue
            tx = self.rng.uniform(0.5, 17.5)
            ty = self.rng.uniform(1.0, 31.0)
            if it["team"] == "red" and self.rng.random() < 0.7:
                ty = self.rng.uniform(1.0, 17.0)
            if it["team"] == "blue" and self.rng.random() < 0.7:
                ty = self.rng.uniform(15.0, 31.0)
            px, py = tile_to_px(tx, ty)
            placed.append((py, it, im, px - ox, py - oy))
        # spell areas / explosions (labelled with the spell card) and unlabelled effect clutter
        for _ in range(self.rng.choice([0, 0, 1, 1, 2, 3])):
            labelled = self.rng.random() < 0.6
            card = self.rng.choice(self.spell_cards if labelled else list(self.fx))
            if not self.fx.get(card):
                continue
            it = self.rng.choice(self.fx[card])
            try:
                img, (ox, oy) = self.bank.render_fx(it, self.rng, scale=self.k)
            except Exception as e:  # noqa: BLE001
                print("fx render failed", it["anim"], repr(e)[:120])
                continue
            if img is None or img.shape[1] > W or img.shape[0] > H:
                continue
            px, py = tile_to_px(self.rng.uniform(1, 17), self.rng.uniform(2, 30))
            fit = dict(it, obj=card, team="spell" if labelled else "clutter")
            placed.append((py - 40, fit, Image.fromarray(img), px - ox, py - oy))
        placed.sort(key=lambda p: p[0])
        sh = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        dsh = ImageDraw.Draw(sh)
        for py, it, im, x, y in placed:
            if it["type"] in ("CHARACTER", "BUILDING", "EXT", "BUILDINGS", "CHARACTERS"):
                cx, w = x + im.width / 2, im.width * 0.5
                dsh.ellipse([cx - w / 2, py - w / 5, cx + w / 2, py + w / 5], fill=(0, 0, 0, 70))
        canvas.alpha_composite(sh.filter(ImageFilter.GaussianBlur(2)))
        for py, it, im, x, y in placed:
            x, y = int(round(x)), int(round(y))
            if x >= W or y >= H or x + im.width <= 0 or y + im.height <= 0:
                continue
            canvas.alpha_composite(im, (max(0, x), max(0, y)), (max(0, -x), max(0, -y)))
            a = np.asarray(im)[:, :, 3] > 40
            if a.sum() < 12:
                continue
            ys, xs = np.where(a)
            bx0, bx1 = max(0, x + xs.min()), min(W, x + xs.max() + 1)
            by0, by1 = max(0, y + ys.min()), min(H, y + ys.max() + 1)
            if bx1 - bx0 < 3 or by1 - by0 < 3:
                continue
            if it["team"] == "clutter":
                continue
            cname = it["obj"] + "|spell" if it["team"] == "spell" else class_name(it)
            boxes.append([self.cid[cname], (bx0 + bx1) / 2, (by0 + by1) / 2, bx1 - bx0, by1 - by0])
            if it["type"] in ("CHARACTER", "BUILDING", "EXT") and self.rng.random() < 0.6:
                self._hpbar(canvas, (bx0 + bx1) / 2, by0 - 6, it["team"])
        img = canvas.convert("RGB")
        if self.rng.random() < 0.3:
            img = img.filter(ImageFilter.GaussianBlur(self.rng.uniform(0.3, 0.9)))
        return img, boxes

    def _hpbar(self, canvas, cx, y, team):
        d = ImageDraw.Draw(canvas)
        col = (220, 50, 60) if team == "red" else (60, 140, 230)
        w = self.rng.choice([22, 28, 34])
        f = self.rng.uniform(0.2, 1.0)
        d.rectangle([cx - w / 2, y - 3, cx + w / 2, y + 1], fill=(30, 30, 30))
        d.rectangle([cx - w / 2 + 1, y - 2, cx - w / 2 + 1 + (w - 2) * f, y], fill=col)
        d.rectangle([cx - w / 2 - 9, y - 5, cx - w / 2 - 1, y + 3], fill=col)


def main():
    assets, bg_dir, out, n, seed = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
    s = Synth(assets, bg_dir, seed)
    os.makedirs(f"{out}/images", exist_ok=True)
    os.makedirs(f"{out}/labels", exist_ok=True)
    json.dump(s.classes, open(f"{out}/classes.json", "w"), indent=0)
    t = time.time()
    for k in range(n):
        try:
            img, boxes = s.make()
        except Exception as e:  # noqa: BLE001
            print("make failed", repr(e)[:160])
            continue
        name = f"s{seed}_{k:06d}"
        img.save(f"{out}/images/{name}.jpg", quality=s.rng.randint(70, 95))
        with open(f"{out}/labels/{name}.txt", "w") as f:
            for c, cx, cy, w, h in boxes:
                f.write(f"{c} {cx / W:.5f} {cy / H:.5f} {w / W:.5f} {h / H:.5f}\n")
        if k < 4:
            pv = img.copy()
            d = ImageDraw.Draw(pv)
            for c, cx, cy, w, h in boxes:
                d.rectangle([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], outline=(255, 255, 0))
                d.text((cx - w / 2, cy - h / 2 - 10), s.classes[c][:18], fill=(255, 255, 255))
            pv.save(f"{out}/preview_{seed}_{k}.jpg")
        if k % 200 == 0:
            print(k, round(time.time() - t, 1), flush=True)


if __name__ == "__main__":
    main()
