"""Decoder for WavePlot v407 settings files (``WVPL407.SET``; SETTS.PAS ``SetsType``).

Only the first 1324 bytes of the 4096-byte record are meaningful (the rest is whatever followed
the record in DOS memory). Strings are Turbo Pascal short strings (length byte + characters).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

VIDMODES = ("std16", "vga256", "svga640", "svga800", "svga1024", "svga16_800", "svga16_1024")
SNAPSETS = ("ColMap", "ContMap", "MLine")
DIRNS = ("Horiz", "Vert", "Both")
DUMPSETS = ("Stress", "VVect", "ESS", "ISO")
SNAPFILLS = ("Log", "Quad", "Lin", "Fixed", "Cosn", "DQuad", "SqLog")
COLSCALES = ("blackb", "whiteb", "grey")
SETS_SIZE = 1324


def _pstr(b: bytes, off: int, maxlen: int) -> str:
    n = min(b[off], maxlen)
    return b[off + 1:off + 1 + n].decode("latin-1", errors="replace")


def _enum(names: tuple[str, ...], v: int) -> str:
    return names[v] if 0 <= v < len(names) else f"#{v}"


@dataclass
class LegacySettings:
    col256: bool
    vidmode: str
    drive: str
    directory: str
    fname: str
    freq_wind: int
    low_freq: int
    high_freq: int
    low_taper: int
    high_taper: int
    max_taper: int
    quick: bool
    interp: bool
    quick_max: int
    map_on: bool
    snap_set: str
    dirn: str
    line_col: bool
    same_colour: bool
    dump_set: str
    compress: bool
    leg_max: list[float]
    titles: list[str]
    hz_title: str
    vt_title: str
    hd_title: str
    theta: float
    friction: float
    set3d: bool
    ang3d: int
    mult_snap: bool
    mult_file: bool
    whqty: int
    wvqty: int
    mjump: int
    mfiles: list[list[str]]
    geom_sc_set: bool
    geom_sc: float
    geom_ext_set: bool
    geom_ext: tuple[float, float, float, float]
    vect_set: bool
    vect_sc: float
    max_v: float
    min_v: float
    snap_fill: str
    snap_col_abs: bool
    snap_1lev: float
    snap_nlev: float
    qty_levels: int
    qty_cols: int
    show_titles: bool
    col_scale: str
    extra: dict = field(default_factory=dict)

    @property
    def level_method(self) -> str:
        return {"Log": "log", "Quad": "quad", "Lin": "lin", "Fixed": "fixed", "Cosn": "cos", "DQuad": "dquad",
                "SqLog": "log"}.get(self.snap_fill, "lin")

    @property
    def representation(self) -> str:
        return {"ColMap": "map", "ContMap": "contour", "MLine": "lines"}.get(self.snap_set, "map")


def read_legacy_settings(path: str | Path | bytes) -> LegacySettings:
    b = path if isinstance(path, bytes | bytearray) else Path(path).read_bytes()
    if len(b) < SETS_SIZE:
        raise ValueError(f"settings file too short ({len(b)} bytes; expected ≥ {SETS_SIZE})")
    u16 = lambda off: struct.unpack_from("<H", b, off)[0]  # noqa: E731
    i16 = lambda off: struct.unpack_from("<h", b, off)[0]  # noqa: E731
    f32 = lambda off: struct.unpack_from("<f", b, off)[0]  # noqa: E731
    bol = lambda off: b[off] != 0  # noqa: E731
    mfiles = [[_pstr(b, 877 + (hz * 6 + vt) * 9, 8) for vt in range(6)] for hz in range(6)]
    return LegacySettings(
        col256=bol(0), vidmode=_enum(VIDMODES, b[1]), drive=chr(b[2]), directory=_pstr(b, 3, 255),
        fname=_pstr(b, 259, 255), freq_wind=u16(515), low_freq=u16(517), high_freq=u16(519), low_taper=u16(521),
        high_taper=u16(523), max_taper=u16(525), quick=bol(527), interp=bol(528), quick_max=u16(529),
        map_on=bol(531), snap_set=_enum(SNAPSETS, b[532]), dirn=_enum(DIRNS, b[534]), line_col=bol(535),
        same_colour=bol(536), dump_set=_enum(DUMPSETS, b[537]), compress=bol(538),
        leg_max=[f32(539 + 4 * k) for k in range(8)], titles=[_pstr(b, 571 + 41 * k, 40) for k in range(4)],
        hz_title=_pstr(b, 735, 40), vt_title=_pstr(b, 776, 40), hd_title=_pstr(b, 817, 40), theta=f32(858),
        friction=f32(862), set3d=bol(866), ang3d=i16(867), mult_snap=bol(869), mult_file=bol(870),
        whqty=u16(871), wvqty=u16(873), mjump=u16(875), mfiles=mfiles, geom_sc_set=bol(1273), geom_sc=f32(1274),
        geom_ext_set=bol(1278), geom_ext=(f32(1279), f32(1283), f32(1287), f32(1291)), vect_set=bol(1295),
        vect_sc=f32(1296), max_v=f32(1300), min_v=f32(1304), snap_fill=_enum(SNAPFILLS, b[1308]),
        snap_col_abs=bol(1309), snap_1lev=f32(1310), snap_nlev=f32(1314), qty_levels=u16(1318), qty_cols=u16(1320),
        show_titles=bol(1322), col_scale=_enum(COLSCALES, b[1323]),
    )
