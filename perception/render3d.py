"""Software renderer for the game's skinned 3D battle models (heroes, new evolutions, some new cards).

Input: glb files already converted from Supercell's flatbuffer glTF ("FLA2") to plain glTF by the
Supercell-Flat-Converter (patched, see sfc.patch). No GPU / OpenGL needed: numpy z-buffer rasterizer.
"""
import json
import math
import struct

import numpy as np

COMP = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


class GLB:
    def __init__(self, path):
        b = open(path, "rb").read()
        L, _ = struct.unpack_from("<II", b, 12)
        self.j = json.loads(b[20:20 + L])
        p = 20 + L
        self.bin = b""
        if p + 8 <= len(b):
            L2, _ = struct.unpack_from("<II", b, p)
            self.bin = b[p + 8:p + 8 + L2]

    def acc(self, i):
        a = self.j["accessors"][i]
        bv = self.j["bufferViews"][a["bufferView"]]
        dt = COMP[a["componentType"]]
        n = NCOMP[a["type"]]
        off = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        stride = bv.get("byteStride")
        isz = np.dtype(dt).itemsize
        if stride and stride != isz * n:
            raw = np.frombuffer(self.bin, np.uint8, stride * (a["count"] - 1) + isz * n, off)
            arr = np.lib.stride_tricks.as_strided(raw.view(dt), (a["count"], n), (stride, isz)).copy()
        else:
            arr = np.frombuffer(self.bin, dt, a["count"] * n, off).reshape(a["count"], n).copy()
        if a.get("normalized"):
            arr = arr.astype(np.float32) / float(np.iinfo(dt).max)
        return arr


def quat_mat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], np.float64)


def trs(t, r, s):
    m = np.eye(4)
    m[:3, :3] = quat_mat(r) * np.asarray(s)[None, :]
    m[:3, 3] = t
    return m


class Model:
    def __init__(self, path):
        g = GLB(path)
        self.g = g
        j = g.j
        self.nodes = j["nodes"]
        self.names = {n.get("name"): i for i, n in enumerate(self.nodes)}
        self.parent = {}
        for i, n in enumerate(self.nodes):
            for c in n.get("children", []):
                self.parent[c] = i
        self.rest = []
        for n in self.nodes:
            if "matrix" in n:
                m = np.array(n["matrix"], np.float64).reshape(4, 4).T
                t = m[:3, 3]
                s = np.linalg.norm(m[:3, :3], axis=0)
                r = _mat_quat(m[:3, :3] / s)
                self.rest.append((t, r, s))
            else:
                self.rest.append((np.array(n.get("translation", [0, 0, 0]), float),
                                  np.array(n.get("rotation", [0, 0, 0, 1]), float),
                                  np.array(n.get("scale", [1, 1, 1]), float)))
        self.skins = []
        for sk in j.get("skins", []):
            ibm = g.acc(sk["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1).astype(np.float64)
            self.skins.append((sk["joints"], ibm))
        self.prims = []  # (node, skin, pos, joints, weights, uv, idx, material)
        for ni, n in enumerate(self.nodes):
            if "mesh" not in n:
                continue
            for pr in j["meshes"][n["mesh"]]["primitives"]:
                at = pr["attributes"]
                pos = g.acc(at["POSITION"]).astype(np.float64)
                uv = g.acc(at["TEXCOORD_0"]).astype(np.float64) if "TEXCOORD_0" in at else np.zeros((len(pos), 2))
                jo = g.acc(at["JOINTS_0"]).astype(np.int64) if "JOINTS_0" in at else None
                we = g.acc(at["WEIGHTS_0"]).astype(np.float64) if "WEIGHTS_0" in at else None
                idx = g.acc(pr["indices"]).reshape(-1).astype(np.int64) if "indices" in pr else np.arange(len(pos))
                self.prims.append((ni, n.get("skin"), pos, jo, we, uv, idx.reshape(-1, 3), pr.get("material")))
        self.materials = j.get("materials", [])

    def material_texture(self, mi, key="diffuseTex"):
        if mi is None or mi >= len(self.materials):
            return None
        tex = self.materials[mi].get("variables", {}).get("textures", {}).get(key)
        return tex.split("#")[0] if tex else None

    def world(self, local):
        out = [None] * len(self.nodes)

        def w(i):
            if out[i] is None:
                m = trs(*local[i])
                out[i] = w(self.parent[i]) @ m if i in self.parent else m
            return out[i]

        for i in range(len(self.nodes)):
            w(i)
        return out

    def pose(self, anim=None, t=0.0):
        local = list(self.rest)
        if anim is not None:
            local = anim.apply(self, local, t)
        W = self.world(local)
        tris = []
        for ni, sk, pos, jo, we, uv, idx, mat in self.prims:
            if sk is not None and jo is not None:
                joints, ibm = self.skins[sk]
                jm = np.stack([W[j] @ ibm[k] for k, j in enumerate(joints)])  # [J,4,4]
                ph = np.concatenate([pos, np.ones((len(pos), 1))], 1)
                acc = np.zeros((len(pos), 4))
                for c in range(jo.shape[1]):
                    wc = we[:, c:c + 1]
                    if not wc.any():
                        continue
                    acc += wc * np.einsum("nij,nj->ni", jm[np.clip(jo[:, c], 0, len(jm) - 1)], ph)
                p = acc[:, :3]
            else:
                ph = np.concatenate([pos, np.ones((len(pos), 1))], 1)
                p = (W[ni] @ ph.T).T[:, :3]
            tris.append((p, uv, idx, mat))
        return tris


def _mat_quat(m):
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        return np.array([(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s])
    i = int(np.argmax(np.diag(m)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k]) * 2
    q = np.zeros(4)
    q[i] = 0.25 * s
    q[j] = (m[j, i] + m[i, j]) / s
    q[k] = (m[k, i] + m[i, k]) / s
    q[3] = (m[k, j] - m[j, k]) / s
    return q


class Anim:
    def __init__(self, path, index=0):
        g = GLB(path)
        j = g.j
        self.node_names = [n.get("name") for n in j["nodes"]]
        anims = j.get("animations") or [{"channels": [], "samplers": []}]
        a = anims[min(index, len(anims) - 1)]
        self.name = a.get("name")
        self.channels = []
        self.duration = 0.0
        for ch in a["channels"]:
            s = a["samplers"][ch["sampler"]]
            tin = g.acc(s["input"]).reshape(-1).astype(np.float64)
            out = g.acc(s["output"]).astype(np.float64)
            path = ch["target"]["path"]
            if path == "rotation":  # some channels come out un-normalized (raw int16)
                out = out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)
            elif np.abs(out).max() > 1e4:  # undecoded raw values: ignore the channel
                continue
            self.duration = max(self.duration, float(tin.max()) if len(tin) else 0)
            self.channels.append((self.node_names[ch["target"]["node"]], ch["target"]["path"], tin, out,
                                  s.get("interpolation", "LINEAR")))

    def apply(self, model, local, t):
        local = [list(x) for x in local]
        for name, path, tin, out, interp in self.channels:
            ni = model.names.get(name)
            if ni is None or not len(tin):
                continue
            v = _sample(tin, out, t, path == "rotation")
            k = {"translation": 0, "rotation": 1, "scale": 2}[path]
            local[ni][k] = v
        return [tuple(x) for x in local]


def _sample(tin, out, t, is_rot):
    if t <= tin[0]:
        return out[0]
    if t >= tin[-1]:
        return out[-1]
    i = int(np.searchsorted(tin, t)) - 1
    f = (t - tin[i]) / max(tin[i + 1] - tin[i], 1e-9)
    a, b = out[i], out[i + 1]
    if is_rot:
        if np.dot(a, b) < 0:
            b = -b
        q = a * (1 - f) + b * f
        return q / max(np.linalg.norm(q), 1e-9)
    return a * (1 - f) + b * f


def camera(pitch_deg=50.0, yaw_deg=0.0):
    """View matrix: model yaw around +Y, then tilt so we look down at pitch_deg from horizontal."""
    y = math.radians(yaw_deg)
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    p = math.radians(pitch_deg)
    rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    zup = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)  # game world is Z-up -> view Y-up
    return rx @ ry @ zup


def render(tris, textures, R, px_per_unit=60.0, size=None, light=(0.3, 0.8, 0.5), ambient=0.55):
    """Orthographic z-buffer render. Returns (RGBA uint8, origin (x0, y0) in screen px of the model origin)."""
    allp = np.concatenate([(R @ p.T).T for p, *_ in tris])
    sx, sy = allp[:, 0] * px_per_unit, -allp[:, 1] * px_per_unit
    x0, y0 = math.floor(sx.min()) - 2, math.floor(sy.min()) - 2
    W, H = (math.ceil(sx.max()) + 2 - x0, math.ceil(sy.max()) + 2 - y0) if size is None else size
    col = np.zeros((H, W, 4), np.float32)
    zb = np.full((H, W), -np.inf, np.float32)
    L = np.asarray(light, float)
    L /= np.linalg.norm(L)
    for p, uv, idx, mat in tris:
        q = (R @ p.T).T
        X, Y, Z = q[:, 0] * px_per_unit - x0, -q[:, 1] * px_per_unit - y0, q[:, 2]
        tex = textures.get(mat)
        for a, b, c in idx:
            _tri(col, zb, (X[a], Y[a], Z[a]), (X[b], Y[b], Z[b]), (X[c], Y[c], Z[c]), uv[a], uv[b], uv[c], tex,
                 _shade(q[a], q[b], q[c], L, ambient))
    return (np.clip(col, 0, 1) * 255 + 0.5).astype(np.uint8), (-x0, -y0)


def _shade(a, b, c, L, ambient):
    n = np.cross(b - a, c - a)
    nn = np.linalg.norm(n)
    if nn < 1e-12:
        return 1.0
    n /= nn
    return ambient + (1 - ambient) * abs(float(n @ L))


def _tri(col, zb, A, B, C, ua, ub, uc, tex, shade):
    H, W = zb.shape
    xs, ys = (A[0], B[0], C[0]), (A[1], B[1], C[1])
    x0, x1 = max(int(min(xs)), 0), min(int(math.ceil(max(xs))) + 1, W)
    y0, y1 = max(int(min(ys)), 0), min(int(math.ceil(max(ys))) + 1, H)
    if x1 <= x0 or y1 <= y0:
        return
    den = (B[1] - C[1]) * (A[0] - C[0]) + (C[0] - B[0]) * (A[1] - C[1])
    if abs(den) < 1e-12:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    px, py = xx + 0.5, yy + 0.5
    l1 = ((B[1] - C[1]) * (px - C[0]) + (C[0] - B[0]) * (py - C[1])) / den
    l2 = ((C[1] - A[1]) * (px - C[0]) + (A[0] - C[0]) * (py - C[1])) / den
    l3 = 1 - l1 - l2
    m = (l1 >= -1e-6) & (l2 >= -1e-6) & (l3 >= -1e-6)
    if not m.any():
        return
    z = l1 * A[2] + l2 * B[2] + l3 * C[2]
    sub = zb[y0:y1, x0:x1]
    m &= z > sub
    if not m.any():
        return
    if tex is not None:
        th, tw = tex.shape[:2]
        u = (l1 * ua[0] + l2 * ub[0] + l3 * uc[0])[m]
        v = (l1 * ua[1] + l2 * ub[1] + l3 * uc[1])[m]
        iu = (np.mod(u, 1.0) * tw).astype(np.int64).clip(0, tw - 1)
        iv = (np.mod(v, 1.0) * th).astype(np.int64).clip(0, th - 1)
        rgba = tex[iv, iu].astype(np.float32) / 255.0
    else:
        rgba = np.tile(np.array([0.8, 0.8, 0.8, 1.0], np.float32), (int(m.sum()), 1))
    rgba[:, :3] *= shade
    rgba[:, 3] = 1.0
    sub[m] = z[m]
    col[y0:y1, x0:x1][m] = rgba
