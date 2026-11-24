"""Tiny stand-in for the binary-reader package (subset used by Supercell-Flat-Converter)."""
import struct


class Whence:
    BEGIN = 0
    CUR = 1
    END = 2


class BinaryReader:
    def __init__(self, data=b"", big=False, encoding="utf-8"):
        self._b = bytearray(data)
        self._p = 0
        self._enc = encoding

    def buffer(self):
        return bytes(self._b)

    def pos(self):
        return self._p

    def size(self):
        return len(self._b)

    def seek(self, off, whence=0):
        self._p = off if whence == 0 else (self._p + off if whence == 1 else len(self._b) + off)

    def pad(self, n):
        self.write_bytes(b"\x00" * n)

    def _w(self, bs):
        end = self._p + len(bs)
        if end > len(self._b):
            self._b.extend(b"\x00" * (end - len(self._b)))
        self._b[self._p:end] = bs
        self._p = end

    def write_bytes(self, bs):
        self._w(bytes(bs))

    def read_bytes(self, n=1):
        r = bytes(self._b[self._p:self._p + n])
        self._p += n
        return r

    def read_uint32(self, count=None):
        if count is not None:
            return tuple(self.read_uint32() for _ in range(count))
        v, = struct.unpack_from("<I", self._b, self._p)
        self._p += 4
        return v

    def write_uint32(self, v):
        self._w(struct.pack("<I", v))

    def write_float(self, v):
        self._w(struct.pack("<f", v))

    def read_str(self, n, encoding=None):
        return self.read_bytes(n).split(b"\x00")[0].decode(encoding or self._enc, "replace")

    def write_str(self, s, null=False, encoding=None):
        self._w(s.encode(encoding or self._enc) + (b"\x00" if null else b""))

    def write_str_fixed(self, s, n, encoding=None):
        b = s.encode(encoding or self._enc)[:n]
        self._w(b + b"\x00" * (n - len(b)))

    def eof(self):
        return self._p >= len(self._b)

    def read_float(self, count=None):
        if count is not None:
            return tuple(self.read_float() for _ in range(count))
        v, = struct.unpack_from("<f", self._b, self._p)
        self._p += 4
        return v

    def read_uint16(self, count=None):
        if count is not None:
            return tuple(self.read_uint16() for _ in range(count))
        v, = struct.unpack_from("<H", self._b, self._p)
        self._p += 2
        return v

    def read_int16(self, count=None):
        if count is not None:
            return tuple(self.read_int16() for _ in range(count))
        v, = struct.unpack_from("<h", self._b, self._p)
        self._p += 2
        return v

    def read_uint8(self, count=None):
        if count is not None:
            return tuple(self.read_uint8() for _ in range(count))
        v = self._b[self._p]
        self._p += 1
        return v

    def read_int8(self, count=None):
        if count is not None:
            return tuple(self.read_int8() for _ in range(count))
        v, = struct.unpack_from("<b", self._b, self._p)
        self._p += 1
        return v

    def write_uint16(self, v):
        self._w(struct.pack("<H", v))

    def write_int16(self, v):
        self._w(struct.pack("<h", v))

    def write_uint8(self, v):
        self._w(struct.pack("<B", v))
