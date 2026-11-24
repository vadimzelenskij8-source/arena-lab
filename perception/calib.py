"""Print rendered sizes (scale 1.0) of a few objects to calibrate sprite scale vs real frames."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bank import Bank
b = Bank(sys.argv[1])
for card in ["cannon", "x-bow", "knight", "archers", "electro-wizard", "hog-rider", "giant", "skeletons", "tesla", "mini-pekka"]:
    its = [i for i in b.items(card) if i["src"] == "2d" and i["form"] == "base" and i["team"] in ("red", "any")]
    for it in its[:3]:
        try:
            img, org = b.render(it, 0, direction=sorted(it["dirs"])[len(it["dirs"]) // 2], scale=1.0)
            a = img[:, :, 3] > 64
            ys, xs = np.where(a)
            print(card, it["obj"], it["anim"], "scale", it.get("scale"), "w", xs.max() - xs.min(), "h", ys.max() - ys.min())
        except Exception as e:
            print(card, it["obj"], "fail", e)
s = b.sc("sc/building_tower.sc")
for n in ("KingTower_red", "PrincessTower_red", "princess_tower_red"):
    if n in s.exports:
        img, _ = s.render(s.exports[n], 0); a = img[:, :, 3] > 64; ys, xs = np.where(a)
        print("tower", n, "w", xs.max() - xs.min(), "h", ys.max() - ys.min())
print([k for k in s.exports if "rincess" in k][:20])
