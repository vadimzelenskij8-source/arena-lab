"""Convert Supercell flatbuffer glTF files (FLA2) to plain glb using the patched Supercell-Flat-Converter."""
import os
import sys

SFC = os.environ.get("SFC_DIR", "/tmp/sfc")


def _setup():
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (here, SFC):
        if p not in sys.path:
            sys.path.insert(0, p)


def convert(src, dst):
    _setup()
    from lib.glTF import glTF
    from lib.odin import SupercellOdinGLTF
    g = glTF()
    g.read(open(src, "rb").read())
    for c in g.chunks:
        c.deserialize_json()
    g = SupercellOdinGLTF(g).process()
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    open(dst, "wb").write(g.write())
    return dst
