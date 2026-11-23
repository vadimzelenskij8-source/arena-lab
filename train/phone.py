"""The policy for the phone: the observation encoder of ppo2.Enc rewritten in torch, so one ONNX graph
takes the simulator's raw seat observation (what the phone's digital twin gives) and returns the
action logits, the value and the new memory. Checked against the numpy path before export."""
import numpy as np
import torch
import torch.nn as nn

from ppo2 import EM


def _L(x):
    return torch.as_tensor(list(x), dtype=torch.long)


class TorchEnc(nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.V, self.S, self.H, self.EI = enc.V, enc.S, enc.H, enc.EI
        e = enc.e
        self.e = {k: int(v) for k, v in e.items()}
        self.nflags = enc.nflags
        self.cnt = int(enc.cnt)
        self.scalar_names = list(enc.scalar_names)
        for name, val in (("scalar_idx", enc.scalar_idx), ("tw", enc.tw), ("hand", enc.hand), ("hcost", enc.hcost),
                          ("hform", enc.hform), ("hpend", enc.hpend), ("nq", list(enc.nxt) + list(enc.queue)),
                          ("seen", enc.seen), ("seen_evo", enc.seen_evo or [0]), ("seen_need", enc.seen_need or [0]),
                          ("devo", enc.devo or [0]), ("pc", enc.pc), ("pt", enc.pt), ("px", enc.px), ("py", enc.py),
                          ("ab", enc.ab or [0]), ("deck", enc.deck or [0]), ("dlev", enc.dlev or [0]), ("dslot", enc.dslot or [0])):
            self.register_buffer(name, _L(val), persistent=False)
        self.has_seen_evo, self.has_devo, self.has_ab = bool(enc.seen_evo), bool(enc.devo), bool(enc.ab)
        self.has_deck = bool(enc.deck)
        self.tick = int(enc.tick)
        self.el = int(enc.h["elixir"][0]); self.eel = int(enc.h["enemy_elixir"][0])
        div = {"elixir": 100000, "enemy_elixir": 100000, "elixir_reserved": 100000, "time_ms": 300000,
               "section_left_ms": 180000, "full_bar_ms": 1, "overtime": 1, "crowns": 3, "crowns_enemy": 3,
               "play_open_in_ticks": 100, "pending_count": 4, "enemy_play_count": 60, "section_type": 8}
        self.register_buffer("sdiv", torch.tensor([float(div[n]) for n in self.scalar_names]), persistent=False)
        self.fullbar = self.scalar_names.index("full_bar_ms") if "full_bar_ms" in self.scalar_names else -1

    @staticmethod
    def _clean(x):
        return torch.where(torch.isnan(x), torch.zeros_like(x), x)

    def col(self, rows, k):
        if k in self.e:
            return rows[:, :, self.e[k]].float()
        return torch.zeros(rows.shape[0], rows.shape[1])

    def forward(self, O, last_play):
        """O: long [B, seat_ints]; last_play: long [B, 8] -> the batch dict of Net.encode/heads."""
        B = O.shape[0]
        V, S = self.V, self.S
        rows = O[:, self.H:self.H + EM * self.EI].reshape(B, EM, self.EI)
        n = torch.clamp(O[:, self.cnt], max=EM)
        valid = torch.arange(EM).unsqueeze(0) < n.unsqueeze(1)
        e = self.e
        rid = torch.clamp(rows[:, :, e["row"]] + 1, 0, V - 1)
        st = torch.clamp(rows[:, :, e["state"]] + 2, 0, S - 1) if "state" in e else torch.zeros(B, EM, dtype=torch.long)
        kind = torch.clamp(rows[:, :, e["kind"]], 0, 5)
        side = torch.clamp(rows[:, :, e["side"]], 0, 3)
        c = lambda k: self.col(rows, k)
        x, y = c("x"), c("y")
        hp, mh, sh = c("hp"), c("max_hp"), c("shield")
        zero = torch.zeros_like(x)
        known = (hp >= 0).float()
        tgt = c("target")
        num = [x / 18000.0, y / 32000.0, known, torch.where(hp >= 0, hp / torch.clamp(mh, min=1), zero),
               torch.log1p(torch.clamp(mh, min=0)) / 9.0, torch.clamp(sh, min=0) / 1000.0, c("dx") / 1000.0, c("dy") / 1000.0,
               (tgt >= 0).float(), torch.where(tgt >= 0, c("target_x") / 18000.0, zero),
               torch.where(tgt >= 0, c("target_y") / 32000.0, zero), c("deploy_ms") / 2000.0,
               c("attack_ms") / 3000.0, c("load_ms") / 3000.0, c("speed_percent") / 100.0]
        rest = [c("hit_speed_percent") / 100.0, c("buff_count") / 3.0, c("aim_x") / 18000.0, c("aim_y") / 32000.0,
                c("time_left_ms") / 10000.0, c("radius") / 5000.0, c("level") / 16.0, (c("ability_ticks") >= 0).float()]
        fl = rows[:, :, e["flags"]] if "flags" in e else torch.zeros(B, EM, dtype=torch.long)
        bits = [(torch.div(fl, 2 ** b, rounding_mode="floor") % 2).float() for b in range(self.nflags)]
        feats = self._clean(torch.clamp(torch.stack(num + rest + bits, -1), -5, 5))
        tc = torch.clamp(torch.floor(x / 1000.0), 0, 17).long()
        tr = torch.clamp(torch.floor(y / 1000.0), 0, 31).long()
        b = dict(rid=rid, st=st, kind=kind, side=side, num=feats, valid=valid, tile=tr * 18 + tc)

        # the seat's own deck
        if self.has_deck:
            b["deck"] = torch.clamp(O[:, self.deck] + 1, 0, V - 1)
            b["dstat"] = torch.cat([O[:, self.dlev].float() / 16.0, torch.clamp(O[:, self.dslot], 0, 3).float() / 3.0], 1)

        # header
        f = []
        sc = O[:, self.scalar_idx].float() / self.sdiv
        if self.fullbar >= 0:
            fb = O[:, self.scalar_idx[self.fullbar]].float()
            sc = torch.cat([sc[:, :self.fullbar], torch.clamp(28000.0 / torch.clamp(fb, min=1.0), 0, 4).unsqueeze(1),
                            sc[:, self.fullbar + 1:]], 1)
        f.append(sc)
        el, eel = O[:, self.el], O[:, self.eel]
        f.append(torch.stack([el >= 100000, el >= 70000, eel <= 30000, eel >= 90000], 1).float())
        t = O[:, self.tw].float().reshape(B, 6, 2)
        f.append(t[:, :, 0] / torch.clamp(t[:, :, 1], min=1))
        f.append(O[:, self.hcost].float() / 10.0)
        f.append(O[:, self.hform].float() / 2.0)
        f.append(O[:, self.hpend].float())
        dev = O[:, self.devo].float() if self.has_devo else torch.zeros(B, 8)
        f.append(torch.clamp(dev, -1, 4) / 4.0)
        seen = O[:, self.seen]
        tick = O[:, self.tick].float()
        has = (seen >= 0).float()
        since = torch.where(last_play >= 0, (tick.unsqueeze(1) - last_play.float()) / 1200.0, torch.full_like(has, 2.0))
        pc, pt = O[:, self.pc], O[:, self.pt]
        pc4 = pc[:, :4]
        recent4 = ((seen.unsqueeze(2) == pc4.unsqueeze(1)) & (pc4.unsqueeze(1) >= 0)).any(-1).float()
        dead = ((since > 0.5) & (has > 0)).float()
        f.append(torch.cat([has, torch.clamp(since, 0, 2), recent4, dead], 1))
        if self.has_seen_evo:
            se, sn = O[:, self.seen_evo].float(), O[:, self.seen_need].float()
            f.append(torch.cat([torch.clamp(se, -1, 8) / 8, (se >= 0).float(),
                                torch.where(sn > 0, torch.clamp(se, min=0) / torch.clamp(sn, min=1), torch.zeros_like(sn))], 1))
        else:
            f.append(torch.zeros(B, 24))
        f.append(torch.cat([torch.clamp(pt.float(), 0, 2400) / 2400.0, O[:, self.px].float() / 18000.0,
                            O[:, self.py].float() / 32000.0], 1))
        if self.has_ab:
            ab = O[:, self.ab].float()
            f.append(torch.cat([torch.clamp(ab, -1, 30000) / 30000.0, (ab > 0).float()], 1))
        b["g"] = self._clean(torch.cat(f, 1))
        b["hand"] = torch.clamp(O[:, self.hand] + 1, 0, V - 1)
        b["nxt"] = torch.clamp(O[:, self.nq] + 1, 0, V - 1)
        b["seen"] = torch.clamp(seen + 1, 0, V - 1)
        b["play"] = torch.clamp(pc + 1, 0, V - 1)
        b["hstat"] = torch.stack([O[:, self.hcost].float() / 10.0, O[:, self.hform].float() / 2.0, O[:, self.hpend].float()], -1)
        return b


class PhoneStep(nn.Module):
    """One decision: (raw obs, mask, memory) -> (logits incl. the closed delegate, value, new memory)."""

    def __init__(self, net, enc):
        super().__init__()
        self.net = net
        self.enc = TorchEnc(enc)

    def encode(self, b):
        """Net.encode, with the all-padding fix written without a boolean index_put (ONNX friendly)."""
        n_ = self.net
        rid, valid = b["rid"], b["valid"]
        B = rid.shape[0]
        x = torch.cat([n_.cv(rid), n_.state(b["st"]), n_.kind(b["kind"]), n_.side(b["side"]), b["num"]], -1)
        x = n_.ent(x)
        pad = ~valid
        allpad = pad.all(1)
        first = torch.arange(pad.shape[1]).unsqueeze(0) == 0
        pad2 = pad & ~(allpad.unsqueeze(1) & first)
        a, _ = n_.att(x, x, x, key_padding_mask=pad2, need_weights=False)
        x = n_.ln(x + a)
        m = valid.unsqueeze(-1).float()
        mean = (x * m).sum(1) / m.sum(1).clamp(min=1)
        mx = x.masked_fill(~valid.unsqueeze(-1), -1e4).max(1).values.clamp(min=-10)
        gv = n_.to_grid(x) * m
        grid = torch.zeros(B, 16, 576)
        grid = grid.scatter_add(2, b["tile"].unsqueeze(1).expand(-1, 16, -1), gv.transpose(1, 2))
        grid = torch.cat([grid.view(B, 16, 32, 18), n_.coords.expand(B, -1, -1, -1)], 1)
        tf = n_.cnn(grid)
        hand = n_.cv(b["hand"])
        g = torch.cat([b["g"], hand.flatten(1), n_.cv(b["nxt"]).mean(1), n_.cv(b["seen"]).mean(1), n_.cv(b["play"]).mean(1)], -1)
        g = n_.glob(g)
        if "deck" in b:
            dv = n_.cv(b["deck"])
            g = g + n_.deck_l(torch.cat([dv.mean(1), dv.max(1).values, b["dstat"]], -1))
        sp = tf.mean((2, 3))
        z = n_.trunk(torch.cat([mean, mx, g, sp], -1))
        return z, tf.flatten(2), hand

    def forward(self, O, mask, h, last_play, prev_seen):
        O = O.long()
        seen = O[:, self.enc.seen]
        diff = (seen != prev_seen).long()  # integer arithmetic: ONNX Runtime has no integer Where on every build
        last_play = last_play * (1 - diff) - diff
        tick = O[:, self.enc.tick].unsqueeze(1)
        pc, pt = O[:, self.enc.pc], O[:, self.enc.pt]
        match = (seen.unsqueeze(2) == pc.unsqueeze(1)) & (pc.unsqueeze(1) >= 0)
        cand = match.long() * ((tick - pt).unsqueeze(1) + 1) - 1
        last_play = torch.maximum(last_play, cand.max(-1).values)
        b = self.enc(O, last_play)
        z, tf, hand = self.encode(b)
        hn = self.net.gru(z, h)
        M = torch.cat([mask.bool(), torch.zeros(mask.shape[0], 1, dtype=torch.bool)], 1)
        lg, v = self.net.heads(hn, tf, hand, b, M)
        return lg, v, hn, last_play, seen


def check_and_export(net, enc, batches, out_dir, onnx_ok=True):
    """batches: list of (O int32 [B,S], M bool [B,A], lg_ref [B,A+1], v_ref [B]) from the numpy path, in time order
    for one set of battles (the memory is carried). Returns the max differences and writes policy.onnx."""
    import json
    net.eval()
    step = PhoneStep(net, enc).eval()
    B = batches[0][0].shape[0]
    h = torch.zeros(B, 256); lp = torch.full((B, 8), -1, dtype=torch.long); ps = torch.full((B, 8), -1, dtype=torch.long)
    worst_lg = worst_v = 0.0; agree = tot = 0
    with torch.no_grad():
        for O, M, lg_ref, v_ref in batches:
            lg, v, h, lp, ps = step(torch.from_numpy(O), torch.from_numpy(M), h, lp, ps)
            ok = M  # only open actions carry meaning
            d = (lg[:, :-1] - torch.from_numpy(lg_ref[:, :-1])).abs()
            worst_lg = max(worst_lg, float(d[torch.from_numpy(ok)].max()) if ok.any() else 0.0)
            worst_v = max(worst_v, float((v - torch.from_numpy(v_ref)).abs().max()))
            agree += int((lg[:, :-1].argmax(1) == torch.from_numpy(lg_ref[:, :-1]).argmax(1)).sum()); tot += B
    res = {"max_logit_diff": worst_lg, "max_value_diff": worst_v, "argmax_agree": agree / max(1, tot)}
    print("torch-vs-numpy", res, flush=True)
    if not onnx_ok:
        return res
    O0, M0 = torch.from_numpy(batches[0][0][:1]), torch.from_numpy(batches[0][1][:1])
    args = (O0, M0, torch.zeros(1, 256), torch.full((1, 8), -1, dtype=torch.long), torch.full((1, 8), -1, dtype=torch.long))
    path = f"{out_dir}/policy.onnx"
    torch.onnx.export(step, args, path, input_names=["obs", "mask", "h", "last_play", "prev_seen"],
                      output_names=["logits", "value", "h_out", "last_play_out", "prev_seen_out"],
                      dynamic_axes={k: {0: "B"} for k in ("obs", "mask", "h", "last_play", "prev_seen",
                                                           "logits", "value", "h_out", "last_play_out", "prev_seen_out")},
                      opset_version=17, dynamo=False)
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    h = np.zeros((B, 256), np.float32); lp = np.full((B, 8), -1, np.int64); ps = np.full((B, 8), -1, np.int64)
    worst = 0.0
    hh = torch.zeros(B, 256); tlp = torch.full((B, 8), -1, dtype=torch.long); tps = torch.full((B, 8), -1, dtype=torch.long)
    with torch.no_grad():
        for O, M, _lg, _v in batches[:40]:
            lg_o, v_o, h, lp, ps = sess.run(None, {"obs": O.astype(np.int32), "mask": M, "h": h, "last_play": lp, "prev_seen": ps})
            lg_t, v_t, hh, tlp, tps = step(torch.from_numpy(O), torch.from_numpy(M), hh, tlp, tps)
            worst = max(worst, float(np.abs(lg_o[:, :-1][M] - lg_t[:, :-1].numpy()[M]).max()) if M.any() else 0.0)
    res["onnx_vs_torch"] = worst
    import os
    res["onnx_mb"] = round(os.path.getsize(path) / 1e6, 2)
    print("onnx", res, flush=True)
    json.dump(res, open(f"{out_dir}/check.json", "w"))
    return res
