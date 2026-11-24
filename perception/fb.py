"""Minimal FlatBuffers table reader (no generated code)."""
import struct


class T:
    """A flatbuffer table at absolute position `pos` in bytes `b`."""
    __slots__ = ("b", "pos", "vt", "vlen")

    def __init__(self, b, pos):
        self.b, self.pos = b, pos
        self.vt = pos - struct.unpack_from("<i", b, pos)[0]
        self.vlen = struct.unpack_from("<H", b, self.vt)[0]

    @staticmethod
    def root(b, off=0):
        return T(b, off + struct.unpack_from("<I", b, off)[0])

    def _o(self, slot):
        vo = 4 + 2 * slot
        if vo >= self.vlen:
            return 0
        return struct.unpack_from("<H", self.b, self.vt + vo)[0]

    def has(self, slot):
        return self._o(slot) != 0

    def scalar(self, slot, fmt, default=0):
        o = self._o(slot)
        return default if not o else struct.unpack_from("<" + fmt, self.b, self.pos + o)[0]

    def _ind(self, slot):
        o = self._o(slot)
        if not o:
            return None
        p = self.pos + o
        return p + struct.unpack_from("<I", self.b, p)[0]

    def table(self, slot):
        p = self._ind(slot)
        return None if p is None else T(self.b, p)

    def string(self, slot):
        p = self._ind(slot)
        if p is None:
            return None
        n = struct.unpack_from("<I", self.b, p)[0]
        return self.b[p + 4:p + 4 + n].decode("utf-8", "replace")

    def vec(self, slot):
        """(start, count) of a vector, or (0, 0)."""
        p = self._ind(slot)
        if p is None:
            return 0, 0
        return p + 4, struct.unpack_from("<I", self.b, p)[0]

    def bytes_(self, slot):
        s, n = self.vec(slot)
        return self.b[s:s + n]

    def tables(self, slot):
        s, n = self.vec(slot)
        out = []
        for i in range(n):
            q = s + 4 * i
            out.append(T(self.b, q + struct.unpack_from("<I", self.b, q)[0]))
        return out

    def strings(self, slot):
        s, n = self.vec(slot)
        out = []
        for i in range(n):
            q = s + 4 * i
            p = q + struct.unpack_from("<I", self.b, q)[0]
            m = struct.unpack_from("<I", self.b, p)[0]
            out.append(self.b[p + 4:p + 4 + m].decode("utf-8", "replace"))
        return out

    def structs(self, slot, fmt):
        s, n = self.vec(slot)
        size = struct.calcsize("<" + fmt)
        return [struct.unpack_from("<" + fmt, self.b, s + i * size) for i in range(n)]

    def array(self, slot, fmt):
        s, n = self.vec(slot)
        return list(struct.unpack_from("<%d%s" % (n, fmt), self.b, s)) if n else []
