"""Behaviour cloning of our network on human top-ladder replays (labelled in our env schema).

Data: <data>/*.npz, one per replay: o (rows, seat_ints) int32, m (rows, actions) uint8, a (rows,) int32,
s (rows,) seat, k (rows,) keep-for-loss (clean label). Rows alternate seat 0 / seat 1 per decision.
Each seat's rows are a sequence: the GRU runs over it as in play. Loss: cross-entropy on the human's
command (waits down-weighted, only rows with a real choice), value toward the game's result.
Starts from our current policy (--init), so the delegate head and the value head keep their training.
"""
import argparse, glob, json, os, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
from crforge_env import LadderConfig, VectorBattle  # noqa: E402
from ppo2 import Enc, Net, to_t  # noqa: E402


def feats(enc, O):
    """Features of one seat's rows in order (memory of enemy plays tracked as in distill.batch_obs)."""
    n = len(O)
    lp = np.full((1, 8), -1, np.int64); prev = np.full((1, 8), -1, np.int64)
    LP = np.zeros((n, 8), np.int64)
    seen_all = O[:, enc.seen]
    for i in range(n):
        seen = seen_all[i:i + 1]
        lp = np.where(seen != prev, -1, lp); prev = seen
        tick = O[i:i + 1, enc.tick][:, None]; pc, pt = O[i:i + 1, enc.pc], O[i:i + 1, enc.pt]
        match = (seen[:, :, None] == pc[:, None, :]) & (pc[:, None, :] >= 0)
        lp = np.maximum(lp, np.where(match, (tick - pt)[:, None, :], -1).max(-1))
        LP[i] = lp[0]
    en = enc.entities(O); g, ids = enc.header(O, LP)
    hst = np.stack([O[:, enc.hcost] / 10.0, O[:, enc.hform] / 2.0, O[:, enc.hpend]], -1).astype(np.float32)
    dk, dst = enc.deck_feats(O)
    return dict(rid=en["rid"], st=en["st"], kind=en["kind"], side=en["side"], num=en["num"], valid=en["valid"],
                tile=en["tile"], g=g, hstat=hst, deck=dk, dstat=dst, **ids)


class Data:
    def __init__(self, files, enc, na):
        self.files, self.enc, self.na, self.cache = files, enc, na, {}

    def seq(self, f, seat):
        key = (f, seat)
        if key not in self.cache:
            if len(self.cache) > 150:
                self.cache.pop(next(iter(self.cache)))
            z = np.load(f)
            sel = z["s"] == seat
            O, A, K = z["o"][sel], z["a"][sel].astype(np.int64), z["k"][sel].astype(bool)
            M = np.unpackbits(z["m"][sel], axis=1, count=self.na).astype(bool)
            if len(O) < 8:
                self.cache[key] = None
                return None
            last = O[-1]
            c, ce = int(last[self.enc.h["crowns"][0]]), int(last[self.enc.h["crowns_enemy"][0]])
            res = 1.0 if c > ce else (-1.0 if c < ce else 0.0)
            real = M[:, 1:].any(1)
            self.cache[key] = (feats(self.enc, O), M, A, K & real, res)
        return self.cache[key]


def chunk(d, f, seat, T, burn, rng):
    s = d.seq(f, seat)
    if s is None:
        return None
    b, M, A, K, res = s
    n = len(A)
    L = burn + T
    plays = np.flatnonzero((A != 0) & K)
    if n <= L:
        st = 0
    elif len(plays) and rng.random() < 0.7:  # most chunks hold a human play in their loss window
        p_ = int(plays[rng.integers(len(plays))])
        st = int(np.clip(p_ - burn - rng.integers(0, T), 0, n - L))
    else:
        st = int(rng.integers(0, n - L + 1))
    ix = np.arange(st, min(n, st + L))
    pad = L - len(ix)
    ix = np.concatenate([ix, np.full(pad, ix[-1])])
    lossw = np.zeros(L, np.float32); lossw[burn:L - pad] = 1
    if st == 0:
        lossw[:burn] = 1  # the sequence's true start: memory is really empty there
    return {k: v[ix] for k, v in b.items()}, M[ix], A[ix], K[ix] & (lossw > 0), np.full(L, res, np.float32), lossw


def batch(d, items, T, burn, rng, NA):
    cs = [c for c in (chunk(d, f, s, T, burn, rng) for f, s in items) if c is not None]
    L = burn + T
    nb = len(cs)
    B = {k: np.stack([c[0][k] for c in cs], 1) for k in cs[0][0]}  # [L, nb, ...]
    M = np.stack([c[1] for c in cs], 1)
    M = np.concatenate([M, np.zeros((L, nb, 1), bool)], 2)  # the delegate is never the human's move
    A, K, R, W = (np.stack([c[i] for c in cs], 1) for i in (2, 3, 4, 5))
    return B, M, A, K, R, W, nb


def run(net, B, M, A, K, R, W, nb, L, wait_w, val_w, train=True):
    flat = {k: torch.from_numpy(np.ascontiguousarray(v.reshape(L * nb, *v.shape[2:]))) for k, v in B.items()}
    flat["num"] = flat["num"].float()
    z, tf, hand = net.encode(flat); z = z.view(L, nb, -1)
    h = torch.zeros(nb, 256); hs = []
    for t in range(L):
        h = net.gru(z[t], h); hs.append(h)
    H = torch.stack(hs).view(L * nb, -1)
    mk = torch.from_numpy(M.reshape(L * nb, -1).copy())
    y = torch.from_numpy(A.reshape(-1)).long()
    mk[torch.arange(len(y)), y] = True
    lg, v = net.heads(H, tf, hand, flat, mk)
    k = torch.from_numpy(K.reshape(-1)).float()
    wts = k * torch.where(y == 0, wait_w, 1.0)
    ce = (F.cross_entropy(lg, y, reduction="none") * wts).sum() / wts.sum().clamp(min=1)
    vw = torch.from_numpy(W.reshape(-1))
    lv = ((v - torch.from_numpy(R.reshape(-1))) ** 2 * vw).sum() / vw.sum().clamp(min=1)
    with torch.no_grad():
        pr = lg.argmax(1); kk = k > 0
        pl = kk & (y != 0)
        stats = dict(acc_play=float((pr[pl] == y[pl]).float().mean()) if pl.any() else 0.0,
                     acc_card=float((((pr[pl] - 1) // 576) == ((y[pl] - 1) // 576)).float().mean()) if pl.any() else 0.0,
                     acc_wait=float((pr[kk & (y == 0)] == 0).float().mean()) if (kk & (y == 0)).any() else 1.0,
                     n_play=int(pl.sum()))
    return ce + val_w * lv, ce, lv, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--install", required=True)
    ap.add_argument("--tables", required=True)
    ap.add_argument("--init", default="")
    ap.add_argument("--out", default="bc_out")
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--T", type=int, default=32)
    ap.add_argument("--burn", type=int, default=16)
    ap.add_argument("--mb", type=int, default=24)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wait-w", type=float, default=0.15)
    ap.add_argument("--val-w", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); random.seed(a.seed)
    torch.set_num_threads(4)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.data, "**", "*.npz"), recursive=True))
    random.Random(1).shuffle(files)
    nval = max(8, len(files) // 50)
    val, tr = files[:nval], files[nval:]
    print("replays", len(files), "train", len(tr), "val", len(val), flush=True)
    vec = VectorBattle(1, 1, install=a.install, tables=a.tables, config=LadderConfig(decision_ticks=5))  # kept open: Enc reads its schema
    enc = Enc(vec); NA = vec.schema.actions
    net = Net(enc, NA + 1)
    if a.init:
        print("init", net.load_state_dict(torch.load(a.init), strict=False), flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    dtr, dval = Data(tr, enc, NA), Data(val, enc, NA)
    L = a.burn + a.T
    vrng = np.random.default_rng(123)
    vitems = [(f, s) for f in val for s in (0, 1)]
    vb = [batch(dval, vitems[i:i + a.mb], a.T, a.burn, vrng, NA) for i in range(0, len(vitems), a.mb)]

    def evaluate():
        net.eval(); agg = dict(ce=0.0, lv=0.0, acc_play=0.0, acc_card=0.0, acc_wait=0.0, n=0)
        with torch.no_grad():
            for B, M, A, K, R, W, nb in vb:
                _, ce, lv, s = run(net, B, M, A, K, R, W, nb, L, a.wait_w, a.val_w)
                agg["ce"] += float(ce); agg["lv"] += float(lv); agg["n"] += 1
                for k_ in ("acc_play", "acc_card", "acc_wait"):
                    agg[k_] += s[k_]
        net.train()
        return {k: round(v / max(1, agg["n"]), 4) for k, v in agg.items() if k != "n"}

    hist = [dict(step=0, t=0, **evaluate())]
    print("VAL", hist[-1], flush=True)
    best = hist[-1]["ce"]; torch.save(net.state_dict(), out / "policy.pt")
    t0 = time.time(); step = 0; last_eval = t0
    while time.time() - t0 < a.minutes * 60:
        items = [(tr[int(rng.integers(len(tr)))], int(rng.integers(2))) for _ in range(a.mb)]
        B, M, A, K, R, W, nb = batch(dtr, items, a.T, a.burn, rng, NA)
        loss, ce, lv, s = run(net, B, M, A, K, R, W, nb, L, a.wait_w, a.val_w)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        step += 1
        if step % 50 == 0:
            print("step", step, "ce", round(float(ce), 3), "v", round(float(lv), 3), s, flush=True)
        if time.time() - last_eval > 600:
            last_eval = time.time()
            hist.append(dict(step=step, t=round(time.time() - t0), **evaluate())); print("VAL", hist[-1], flush=True)
            if hist[-1]["ce"] < best:
                best = hist[-1]["ce"]; torch.save(net.state_dict(), out / "policy.pt")
    hist.append(dict(step=step, t=round(time.time() - t0), **evaluate())); print("VAL", hist[-1], flush=True)
    if hist[-1]["ce"] < best:
        torch.save(net.state_dict(), out / "policy.pt")
    json.dump(hist, open(out / "bc_history.json", "w"), indent=1)


if __name__ == "__main__":
    main()
