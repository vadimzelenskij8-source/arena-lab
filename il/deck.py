"""Opponent-deck inference from the replay dataset: given the cards an opponent has played so far,
rank the cards still hidden by how often they appear in recorded decks containing all of them
(falling back to pair co-occurrence when no deck matches). Held-out test: sha1(tag) % 5 == 0."""
import collections, glob, hashlib, itertools, json, sys
import pyarrow.parquet as pq

d = sys.argv[1]; out = sys.argv[2]
base = lambda k: k.split("-ev")[0]  # evolution / hero variants count as the card
decks, order, league, btype = {}, {}, collections.Counter(), collections.Counter()
for f in glob.glob(f"{d}/replays/*.parquet"):
    for r in pq.read_table(f, columns=["replay_tag", "payload_json", "ranked_league", "battle_type"]).to_pylist():
        p = json.loads(r["payload_json"]); b = p["battle"]; league[r["ranked_league"]] += 1; btype[r["battle_type"]] += 1
        for side in ("team", "opponent"):
            try:
                dk = frozenset(base(c["card_key"]) for c in b[side]["players"][0]["deck"])
            except Exception:
                continue
            if len(dk) == 8:
                decks[(r["replay_tag"], side)] = dk
for f in glob.glob(f"{d}/actions/*.parquet"):
    for r in pq.read_table(f, columns=["replay_tag", "side", "kind", "card_key", "replay_tick_20hz"]).to_pylist():
        if r["kind"] == "play_card" and r["card_key"]:
            l = order.setdefault((r["replay_tag"], r["side"]), [])
            c = base(r["card_key"])
            if c not in l:
                l.append(c)
held = lambda t: int(hashlib.sha1(t.encode()).hexdigest(), 16) % 5 == 0
train = [dk for (t, s), dk in decks.items() if not held(t)]
test = [(dk, order.get((t, s), [])) for (t, s), dk in decks.items() if held(t)]
cnt = collections.Counter(train); card = collections.Counter(c for dk in train for c in dk)
pair = collections.Counter(p for dk in train for p in itertools.combinations(sorted(dk), 2))
index = collections.defaultdict(set)
for i, dk in enumerate(cnt):
    for c in dk:
        index[c].add(i)
uniq = list(cnt)

top8 = [c for c, _ in card.most_common(8)]

def predict(rev):
    rev = set(rev); sc = collections.Counter()
    if not rev:
        return top8, len(uniq)
    ids = set.intersection(*(index[c] for c in rev)) if rev else set(range(len(uniq)))
    for i in ids:
        for c in uniq[i] - rev:
            sc[c] += cnt[uniq[i]]
    if len(sc) < 8 - len(rev):  # no exact deck: pair co-occurrence
        for c in card:
            if c not in rev:
                sc[c] += 1e-3 * sum(pair[tuple(sorted((c, r)))] for r in rev) / max(1, card[c]) ** 0.5
    return [c for c, _ in sc.most_common(8 - len(rev))], len(ids)

res = {}
for k in (0, 1, 2, 3, 4):
    hit = tot = exact = 0; n = 0
    for dk, od in test:
        if len(od) < k:
            continue
        rev = od[:k]; pr, _ = predict(rev)
        h = len(set(pr) & (dk - set(rev))); hit += h; tot += 8 - len(set(rev)); exact += int(h == 8 - len(set(rev))); n += 1
        if n >= 5000:
            break
    res[k] = {"decks": n, "hidden_cards_guessed": round(hit / max(1, tot), 4), "whole_deck_right": round(exact / max(1, n), 4)}
    print("REVEALED", k, res[k], flush=True)
summary = {"train_decks": len(train), "unique_decks": len(uniq), "test_decks": len(test), "result": res,
           "league": league.most_common(12), "battle_type": btype.most_common(8)}
print("SUMMARY", json.dumps(summary))
json.dump(summary, open(out, "w"), indent=1)
