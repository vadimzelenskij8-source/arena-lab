"""Endless deck generator: a new deck for every battle.

mix of: structured decks (1-2 win conditions, 2 spells, the rest troops/buildings, average
elixir 2.6-4.6), fully random decks (8 distinct cards), and real meta decks; at most one champion,
up to two evolutions and one hero (champion and hero share the king slots: not both).
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
POOL = json.load(open(ROOT / "card_pool.json", encoding="utf-8"))
C = {c["key"]: c for c in POOL}
WINS = [c["key"] for c in POOL if c["win"]]
SPELLS = [c["key"] for c in POOL if c["kind"] == "spell"]
UNITS = [c["key"] for c in POOL if c["kind"] != "spell" and not c["win"]]
ALL = [c["key"] for c in POOL]


def _forms(k, rng):
    champs = sum(C[x]["kind"] == "champion" for x in k)
    evo_ok = [x for x in k if C[x]["evo"]]
    evo = rng.sample(evo_ok, min(len(evo_ok), rng.choice([0, 1, 2, 2])))
    hero_ok = [x for x in k if C[x]["hero"] and x not in evo]
    hero = rng.sample(hero_ok, 1) if hero_ok and champs == 0 and rng.random() < 0.6 else []
    return evo, hero


def generate(rng, meta=None, p_meta=0.2, p_random=0.2):
    r = rng.random()
    if meta and r < p_meta:
        return dict(rng.choice(meta))
    for _ in range(200):
        if r < p_meta + p_random:
            k = rng.sample(ALL, 8)
        else:
            k = rng.sample(WINS, rng.choice([1, 1, 2])) + rng.sample(SPELLS, rng.choice([1, 2, 2, 3]))
            k += rng.sample([u for u in UNITS if u not in k], 8 - len(k))
            avg = sum(C[x]["elixir"] for x in k) / 8
            if not 2.6 <= avg <= 4.6:
                continue
        if sum(C[x]["kind"] == "champion" for x in k) > 1:
            continue
        evo, hero = _forms(k, rng)
        return {"name": "gen", "group": "Сгенерированные", "src": "gen", "cards": k, "evo": evo, "hero": hero}
    return dict(rng.choice(meta)) if meta else None


def fixed_set(n, seed=4242, meta=None):
    """Decks for the 'any deck' exam: generated with a seed training never uses."""
    rng = random.Random(seed)
    return [generate(rng, meta, p_meta=0.0, p_random=0.25) for _ in range(n)]
