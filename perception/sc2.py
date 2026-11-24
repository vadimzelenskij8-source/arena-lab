"""Reader and renderer for Supercell SC2 (version 5/6) 2D asset files and .sctx textures.

Format follows sc-workshop/SupercellFlash + SupercellTexture (flatbuffer schemas).
Needs: zstandard, numpy, pillow, texture2ddecoder (for ASTC/ETC sctx).
"""
import math
import os
import struct

import numpy as np
from PIL import Image

from fb import T

ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


def zstd_decompress(b):
    import zstandard
    return zstandard.ZstdDecompressor().decompressobj().decompress(b)


# ---------------------------------------------------------------- textures (.sctx)

# ScPixel types -> (decoder, block w, block h)
def _pixel_info(t):
    small = [(4, 4), (5, 4), (5, 5), (6, 5), (6, 6)]
    big = [(8, 5), (8, 6), (8, 8), (10, 5), (10, 6), (10, 8), (10, 10), (12, 10), (12, 12)]
    for base_small, base_big in ((186, 192), (204, 210)):  # SRGB, linear
        if base_small <= t < base_small + 5:
            return ("astc",) + small[t - base_small]
        if base_big <= t < base_big + 9:
            return ("astc",) + big[t - base_big]
    if t == 70:
        return ("rgba8", 1, 1)
    if t == 80:
        return ("bgra8", 1, 1)
    if t == 40:
        return ("rgb565", 1, 1)
    if t == 30:
        return ("rg8", 1, 1)
    if t == 10:
        return ("r8", 1, 1)
    if t == 1:
        return ("a8", 1, 1)
    if t == 178 or t == 179:
        return ("etc2a", 4, 4)
    if t == 180 or t == 181:
        return ("etc2", 4, 4)
    raise ValueError("unsupported sctx pixel type %d" % t)


def _decode_pixels(data, w, h, t):
    kind, bw, bh = _pixel_info(t)
    if kind == "astc":
        import texture2ddecoder
        raw = texture2ddecoder.decode_astc(bytes(data), w, h, bw, bh)
        a = np.frombuffer(raw, np.uint8).reshape(h, w, 4)[:, :, [2, 1, 0, 3]]
    elif kind in ("etc2a", "etc2"):
        import texture2ddecoder
        fn = texture2ddecoder.decode_etc2a8 if kind == "etc2a" else texture2ddecoder.decode_etc2
        a = np.frombuffer(fn(bytes(data), w, h), np.uint8).reshape(h, w, 4)[:, :, [2, 1, 0, 3]]
    elif kind == "rgba8":
        a = np.frombuffer(data, np.uint8)[:w * h * 4].reshape(h, w, 4)
    elif kind == "bgra8":
        a = np.frombuffer(data, np.uint8)[:w * h * 4].reshape(h, w, 4)[:, :, [2, 1, 0, 3]]
    elif kind == "rgb565":
        v = np.frombuffer(data, "<u2")[:w * h].reshape(h, w).astype(np.uint32)
        r, g, b = (v >> 11) & 31, (v >> 5) & 63, v & 31
        a = np.stack([r * 255 // 31, g * 255 // 63, b * 255 // 31, np.full_like(v, 255)], -1).astype(np.uint8)
    elif kind == "rg8":
        v = np.frombuffer(data, np.uint8)[:w * h * 2].reshape(h, w, 2)
        a = np.concatenate([v[:, :, :1].repeat(3, 2), v[:, :, 1:]], 2)  # luminance + alpha
    elif kind in ("r8", "a8"):
        v = np.frombuffer(data, np.uint8)[:w * h].reshape(h, w, 1)
        a = np.concatenate([np.full((h, w, 3), 255, np.uint8), v], 2) if kind == "a8" else \
            np.concatenate([v.repeat(3, 2), np.full((h, w, 1), 255, np.uint8)], 2)
    return np.ascontiguousarray(a)


_KTX_ASTC = {0x93B0 + i: d for i, d in enumerate(
    [(4, 4), (5, 4), (5, 5), (6, 5), (6, 6), (8, 5), (8, 6), (8, 8), (10, 5), (10, 6), (10, 8), (10, 10), (12, 10),
     (12, 12)])}
_KTX_ASTC.update({k + 0x20: v for k, v in list(_KTX_ASTC.items())})  # SRGB variants 0x93D0..


_VK_ASTC = {157 + 2 * i: d for i, d in enumerate(
    [(4, 4), (5, 4), (5, 5), (6, 5), (6, 6), (8, 5), (8, 6), (8, 8), (10, 5), (10, 6), (10, 8), (10, 10), (12, 10),
     (12, 12)])}
_VK_ASTC.update({k + 1: v for k, v in list(_VK_ASTC.items())})  # SRGB variants


def load_ktx2(b):
    vk, = struct.unpack_from("<I", b, 12)
    w, h = struct.unpack_from("<II", b, 20)
    scheme, = struct.unpack_from("<I", b, 44)
    off, ln, _ulen = struct.unpack_from("<QQQ", b, 80)
    data = b[off:off + ln]
    if scheme == 2:
        data = zstd_decompress(data)
    import texture2ddecoder
    if vk in _VK_ASTC:
        bw, bh = _VK_ASTC[vk]
        raw = texture2ddecoder.decode_astc(bytes(data), w, h, bw, bh)
    elif vk in (147, 148):
        raw = texture2ddecoder.decode_etc2(bytes(data), w, h)
    elif vk in (151, 152):
        raw = texture2ddecoder.decode_etc2a8(bytes(data), w, h)
    elif vk in (37, 43):
        return np.frombuffer(data, np.uint8)[:w * h * 4].reshape(h, w, 4).copy()
    else:
        raise ValueError("KTX2 vkFormat %d" % vk)
    return np.ascontiguousarray(np.frombuffer(raw, np.uint8).reshape(h, w, 4)[:, :, [2, 1, 0, 3]])


def load_ktx(b):
    """Decode a KTX1/KTX2 (optionally zstd-compressed) texture to RGBA uint8."""
    if b[:4] == ZSTD_MAGIC:
        b = zstd_decompress(b)
    if b[1:7] == b"KTX 20":
        return load_ktx2(b)
    assert b[1:4] == b"KTX", "not KTX: %r" % b[:12]
    glInternal, = struct.unpack_from("<I", b, 28)
    w, h = struct.unpack_from("<II", b, 36)
    kv, = struct.unpack_from("<I", b, 60)
    p = 64 + kv
    size, = struct.unpack_from("<I", b, p)
    data = b[p + 4:p + 4 + size]
    import texture2ddecoder
    if glInternal in _KTX_ASTC:
        bw, bh = _KTX_ASTC[glInternal]
        raw = texture2ddecoder.decode_astc(bytes(data), w, h, bw, bh)
    elif glInternal in (0x9278, 0x9279):
        raw = texture2ddecoder.decode_etc2a8(bytes(data), w, h)
    elif glInternal in (0x9274, 0x9275):
        raw = texture2ddecoder.decode_etc2(bytes(data), w, h)
    elif glInternal == 0x8D64:
        raw = texture2ddecoder.decode_etc1(bytes(data), w, h)
    else:
        raise ValueError("KTX format 0x%x" % glInternal)
    return np.ascontiguousarray(np.frombuffer(raw, np.uint8).reshape(h, w, 4)[:, :, [2, 1, 0, 3]])


def _decode_sc1_pixels(data, w, h, t):
    """Raw SC1 pixel types (0/1 RGBA8, 2 RGBA4, 3 RGB5A1, 4 RGB565, 6 LA8, 10 L8)."""
    n = w * h
    if t in (0, 1):
        return np.frombuffer(data, np.uint8)[:n * 4].reshape(h, w, 4).copy()
    if t in (2, 3, 4):
        v = np.frombuffer(data, "<u2")[:n].reshape(h, w).astype(np.uint32)
        if t == 2:
            ch = [(v >> 12) & 15, (v >> 8) & 15, (v >> 4) & 15, v & 15]
            return (np.stack(ch, -1) * 17).astype(np.uint8)
        if t == 3:
            ch = [((v >> 11) & 31) * 255 // 31, ((v >> 6) & 31) * 255 // 31, ((v >> 1) & 31) * 255 // 31, (v & 1) * 255]
            return np.stack(ch, -1).astype(np.uint8)
        ch = [((v >> 11) & 31) * 255 // 31, ((v >> 5) & 63) * 255 // 63, (v & 31) * 255 // 31, np.full_like(v, 255)]
        return np.stack(ch, -1).astype(np.uint8)
    if t == 6:
        v = np.frombuffer(data, np.uint8)[:n * 2].reshape(h, w, 2)
        return np.concatenate([v[:, :, 1:].repeat(3, 2), v[:, :, :1]], 2)
    if t == 10:
        v = np.frombuffer(data, np.uint8)[:n].reshape(h, w, 1)
        return np.concatenate([v.repeat(3, 2), np.full((h, w, 1), 255, np.uint8)], 2)
    raise ValueError("SC1 pixel type %d" % t)


def load_sctx(path):
    """Decode a .sctx file to an RGBA uint8 array (level 0)."""
    b = open(path, "rb").read()
    n = struct.unpack_from("<I", b, 0)[0]
    hdr = T.root(b[4:4 + n])
    ptype = hdr.scalar(1, "I")
    w, h = hdr.scalar(2, "H"), hdr.scalar(3, "H")
    levels = hdr.scalar(4, "B", 1)
    flags = hdr.scalar(6, "I")
    tex_len = hdr.scalar(7, "i")
    p = 4 + n
    mm_len = struct.unpack_from("<I", b, p)[0]
    p += 4
    data_off = p + mm_len
    lv = []
    q = p
    for _ in range(levels):
        ln = struct.unpack_from("<I", b, q)[0]
        mm = T.root(b[q + 4:q + 4 + ln])
        lv.append((mm.scalar(0, "H"), mm.scalar(1, "H"), mm.scalar(2, "I")))
        q += 4 + ln
    p = data_off
    if flags & 8:  # use_padding
        p = (p + 15) & ~15
    body = b[p:]
    if body[:4] == ZSTD_MAGIC:
        body = zstd_decompress(body)
    body = body[:tex_len] if tex_len else body
    lw, lh, off = lv[0] if lv else (w, h, 0)
    return _decode_pixels(body[off:], lw, lh, ptype)


# ---------------------------------------------------------------- SC2 file

class _MmapList(list):
    """Decoded textures are cached as .npy files and memory-mapped, so repeated loads cost no heap memory."""

    def __init__(self, path):
        super().__init__()
        d = os.environ.get("SC_TEX_CACHE")
        self.prefix = os.path.join(d, os.path.basename(path)) if d else None

    def add(self, decode):
        if self.prefix is None:
            return self.append(decode())
        f = "%s_%d.npy" % (self.prefix, len(self))
        if not os.path.exists(f):
            img = decode()
            if img is None:
                return self.append(None)
            np.save(f + ".tmp.npy", np.ascontiguousarray(img))
            os.replace(f + ".tmp.npy", f)
            del img
        self.append(np.load(f, mmap_mode="r"))


class Shape:
    __slots__ = ("id", "commands")  # commands: list of (texture, xy[n,2], uv[n,2])


class Clip:
    __slots__ = ("id", "name", "fps", "children", "blend", "child_names", "frames", "bank", "labels")


class SC2:
    def __init__(self, path, tex_dir=None, load_textures=True):
        self.path = path
        b = open(path, "rb").read()
        assert b[:2] == b"SC", "not an SC file"
        ver = struct.unpack_from("<I", b, 2)[0]
        p = 6
        if ver == 6:
            p += 2
        assert ver in (5, 6), "SC version %d not supported" % ver
        self.version = ver
        n = struct.unpack_from("<I", b, p)[0]
        hdr = T.root(b[p + 4:p + 4 + n])
        p += 4 + n
        self.tprec = hdr.scalar(0, "I")
        self.sprec = hdr.scalar(1, "I")
        csize = hdr.scalar(11, "I")
        resources_offset = hdr.scalar(8, "I")
        if hdr.scalar(12, "I"):
            raise NotImplementedError("external matrix banks")
        body = b[p:p + csize] if csize else b[p:]
        if body[:4] == ZSTD_MAGIC:
            body = zstd_decompress(body)
        self.body = body
        ds_size = struct.unpack_from("<I", body, 0)[0]
        ds = T.root(body, 4)
        self.strings = ds.strings(0)
        self.elements = ds.array(4, "H")
        self.vertex_buf = ds.bytes_(5)
        self.banks = []
        sm = _prec(self.sprec)
        tm = _prec(self.tprec)
        for bank in ds.tables(6):
            if bank.has(0):
                mats = [tuple(m) for m in bank.structs(0, "6f")]
            elif bank.has(2):
                mats = [(a / sm, bb / sm, c / sm, d / sm, x / tm, y / tm) for a, bb, c, d, x, y in bank.structs(2, "6h")]
            else:
                mats = []
            cols = bank.structs(1, "7B")  # r_mul g_mul b_mul alpha r_add g_add b_add
            self.banks.append((mats, cols))

        # resource chunks
        q = resources_offset
        chunks = []
        for _ in range(6):
            if q + 4 > len(body):
                break
            sz = struct.unpack_from("<I", body, q)[0]
            chunks.append((q + 4, sz))
            q += 4 + sz
        names = {}
        if len(chunks) > 0:
            ex = T.root(body, chunks[0][0])
            for oid, ref in zip(ex.array(0, "H"), ex.array(1, "I")):
                names[oid] = self.strings[ref]
        self.objects = {}
        self.exports = {}
        # shapes
        if len(chunks) > 2:
            sh = T.root(body, chunks[2][0])
            vb = self.vertex_buf
            for s in sh.tables(0):
                shape = Shape()
                shape.id = s.scalar(0, "H")
                shape.commands = []
                for _unk, tex, cnt, off in s.structs(1, "4I"):
                    v = np.frombuffer(vb, np.uint8, cnt * 12, off * 12)
                    xy = np.frombuffer(v.tobytes(), "<f4").reshape(cnt, 3)[:, :2].astype(np.float64)
                    uv = np.frombuffer(v.tobytes(), "<u2").reshape(cnt, 6)[:, 4:6].astype(np.float64) / 65535.0
                    shape.commands.append((tex, xy, uv))
                self.objects[shape.id] = shape
        # movieclips
        if len(chunks) > 3:
            mc = T.root(body, chunks[3][0])
            for m in mc.tables(0):
                c = Clip()
                c.id = m.scalar(0, "H")
                c.name = self.strings[m.scalar(1, "I")] if m.has(1) else names.get(c.id)
                c.fps = m.scalar(2, "B") or 24
                c.children = m.array(5, "H")
                c.child_names = [self.strings[i] for i in m.array(6, "I")] if m.has(6) else []
                c.blend = m.array(7, "B") if m.has(7) else []
                if m.has(8):
                    fr = m.structs(8, "2I")
                    counts = [f[0] for f in fr]
                    c.labels = [self.strings[f[1]] if f[1] < len(self.strings) else "" for f in fr]
                else:
                    counts = [f[0] for f in m.structs(12, "H")]
                    c.labels = [""] * len(counts)
                off = m.scalar(9, "I", 0xFFFFFFFF)
                c.frames = []
                for k in counts:
                    els = []
                    for _ in range(k):
                        els.append((self.elements[off], self.elements[off + 1], self.elements[off + 2]))
                        off += 3
                    c.frames.append(els)
                c.bank = m.scalar(10, "I")
                self.objects[c.id] = c
                if c.name:
                    self.exports[c.name] = c
        for oid, nm in names.items():
            if nm not in self.exports and oid in self.objects:
                self.exports[nm] = self.objects[oid]
        # textures
        self.texture_paths = []
        self.textures = _MmapList(path)
        if len(chunks) > 5:
            tx = T.root(body, chunks[5][0])
            for ts in tx.tables(0):
                hi = ts.table(1)
                ext = hi.string(5) if hi else None
                if ext:
                    self.texture_paths.append(ext)
                else:
                    self.texture_paths.append(None)
                    if load_textures and hi is not None:
                        data = hi.bytes_(4)
                        fmt = hi.scalar(0, "B")
                        w_, h_, pt = hi.scalar(2, "H"), hi.scalar(3, "H"), hi.scalar(1, "B")
                        if fmt & 8 or data[1:4] == b"KTX" or data[:4] == ZSTD_MAGIC:
                            self.textures.add(lambda data=data: load_ktx(data))
                        else:
                            self.textures.add(lambda data=data, w_=w_, h_=h_, pt=pt: _decode_sc1_pixels(data, w_, h_, pt))
                        continue
                if load_textures:
                    d = tex_dir or os.path.dirname(path)
                    tp = os.path.join(d, os.path.basename(ext)) if ext else None
                    self.textures.add(lambda tp=tp: load_sctx(tp) if tp and os.path.exists(tp) else None)

    # ------------------------------------------------------------ rendering

    def _ftex(self, i):
        return self.textures[i] if i < len(self.textures) else None

    def frame_count(self, clip):
        return max(1, len(clip.frames))

    def triangles(self, obj, frame, mat=(1, 0, 0, 1, 0, 0), col=(1, 1, 1, 1, 0, 0, 0), blend=0, out=None, depth=0):
        """Flatten an object at `frame` into a list of (tex, xy[3,2], uv[3,2], colour, blend)."""
        if out is None:
            out = []
        if depth > 32:
            return out
        if isinstance(obj, Shape):
            for tex, xy, uv in obj.commands:
                w = _apply(mat, xy)
                n = len(w)
                for i in range(n - 2):  # triangle strip
                    idx = (i, i + 1, i + 2)
                    out.append((tex, w[list(idx)], uv[list(idx)], col, blend))
            return out
        if not isinstance(obj, Clip) or not obj.frames:
            return out
        els = obj.frames[frame % len(obj.frames)]
        mats, cols = self.banks[obj.bank] if obj.bank < len(self.banks) else ([], [])
        for inst, mi, ci in els:
            if inst >= len(obj.children):
                continue
            child = self.objects.get(obj.children[inst])
            if child is None:
                continue
            m = mats[mi] if mi != 0xFFFF and mi < len(mats) else (1, 0, 0, 1, 0, 0)
            cm = col
            if ci != 0xFFFF and ci < len(cols):
                r, g, b_, a, ra, ga, ba = cols[ci]
                cm = (col[0] * r / 255, col[1] * g / 255, col[2] * b_ / 255, col[3] * a / 255,
                      col[4] + ra / 255, col[5] + ga / 255, col[6] + ba / 255)
            bl = obj.blend[inst] if inst < len(obj.blend) and obj.blend[inst] else blend
            self.triangles(child, frame, _mul(mat, m), cm, bl, out, depth + 1)
        return out

    def render(self, obj, frame=0, scale=1.0, pad=2, bbox=None, skip_add=False):
        """Render to (RGBA uint8 array, (x0, y0) origin in object space * scale)."""
        tris = self.triangles(obj, frame)
        if not tris:
            return None, (0, 0)
        if bbox is None:
            allp = np.concatenate([t[1] for t in tris]) * scale
            x0, y0 = np.floor(allp.min(0)) - pad
            x1, y1 = np.ceil(allp.max(0)) + pad
        else:
            x0, y0, x1, y1 = bbox
        W, H = int(x1 - x0), int(y1 - y0)
        if W <= 0 or H <= 0 or W * H > 4096 * 4096:
            return None, (0, 0)
        canvas = np.zeros((H, W, 4), np.float32)
        for tex, xy, uv, col, blend in tris:
            if skip_add and blend == 8:
                continue
            img = self._ftex(tex)
            if img is None:
                continue
            _raster(canvas, img, (xy * scale) - (x0, y0), uv, col, blend)
        out = np.clip(canvas, 0, 1)
        return (out * 255 + 0.5).astype(np.uint8), (x0, y0)


def _prec(p):
    return {2: 20.0, 3: 1024.0}.get(p, 1.0)


def _mul(p, c):
    """Compose parent p with child c (child applied first)."""
    a1, b1, c1, d1, x1, y1 = p
    a2, b2, c2, d2, x2, y2 = c
    return (a1 * a2 + c1 * b2, b1 * a2 + d1 * b2, a1 * c2 + c1 * d2, b1 * c2 + d1 * d2,
            a1 * x2 + c1 * y2 + x1, b1 * x2 + d1 * y2 + y1)


def _apply(m, xy):
    a, b, c, d, x, y = m
    return np.stack([a * xy[:, 0] + c * xy[:, 1] + x, b * xy[:, 0] + d * xy[:, 1] + y], 1)


def _raster(canvas, img, xy, uv, col, blend):
    """Draw one textured triangle into a float RGBA premultiplied-free canvas (straight alpha)."""
    H, W = canvas.shape[:2]
    th, tw = img.shape[:2]
    x0 = max(int(math.floor(xy[:, 0].min())), 0)
    x1 = min(int(math.ceil(xy[:, 0].max())), W)
    y0 = max(int(math.floor(xy[:, 1].min())), 0)
    y1 = min(int(math.ceil(xy[:, 1].max())), H)
    if x1 <= x0 or y1 <= y0:
        return
    (ax, ay), (bx, by), (cx, cy) = xy
    den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    if abs(den) < 1e-9:
        return
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    px, py = xs + 0.5, ys + 0.5
    l1 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / den
    l2 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / den
    l3 = 1 - l1 - l2
    eps = -1e-6
    inside = (l1 >= eps) & (l2 >= eps) & (l3 >= eps)
    if not inside.any():
        return
    u = (l1 * uv[0, 0] + l2 * uv[1, 0] + l3 * uv[2, 0]) * tw - 0.5
    v = (l1 * uv[0, 1] + l2 * uv[1, 1] + l3 * uv[2, 1]) * th - 0.5
    src = _bilinear(img, u[inside], v[inside])  # [n,4] float 0..1
    r, g, b, a, ra, ga, ba = col
    src[:, 0] = src[:, 0] * r + ra
    src[:, 1] = src[:, 1] * g + ga
    src[:, 2] = src[:, 2] * b + ba
    src[:, 3] = src[:, 3] * a
    src[:, :3] = np.clip(src[:, :3], 0, 1)
    sub = canvas[y0:y1, x0:x1]
    dst = sub[inside]
    sa = src[:, 3:4]
    if blend == 8:  # additive
        dst[:, :3] = dst[:, :3] + src[:, :3] * sa
        dst[:, 3:4] = np.maximum(dst[:, 3:4], sa * src[:, :3].max(1, keepdims=True))
    else:
        da = dst[:, 3:4]
        oa = sa + da * (1 - sa)
        safe = np.where(oa > 0, oa, 1)
        dst[:, :3] = (src[:, :3] * sa + dst[:, :3] * da * (1 - sa)) / safe
        dst[:, 3:4] = oa
    sub[inside] = dst


def _bilinear(img, u, v):
    th, tw = img.shape[:2]
    u = np.clip(u, 0, tw - 1)
    v = np.clip(v, 0, th - 1)
    x0 = np.floor(u).astype(np.int64)
    y0 = np.floor(v).astype(np.int64)
    x1 = np.minimum(x0 + 1, tw - 1)
    y1 = np.minimum(y0 + 1, th - 1)
    fx = (u - x0)[:, None]
    fy = (v - y0)[:, None]
    k = 1.0 / 255.0
    c = (img[y0, x0] * ((1 - fx) * (1 - fy) * k) + img[y0, x1] * (fx * (1 - fy) * k) +
         img[y1, x0] * ((1 - fx) * fy * k) + img[y1, x1] * (fx * fy * k))
    return c.astype(np.float32)
