"""Write app/icons (SVG + PNG) showing a small plain-weave pattern. Stdlib only."""
import struct
import zlib
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "app" / "icons"
WARP = (0x5B, 0x3A, 0x29)   # dark brown
WEFT = (0xE0, 0xB4, 0x8F)   # light tan
BG = (0xF6, 0xF1, 0xEA)
N = 6                        # threads across


def cell_colour(x, y, size):
    margin = size // 8
    inner = size - 2 * margin
    if not (margin <= x < size - margin and margin <= y < size - margin):
        return BG
    c = (x - margin) * N // inner
    r = (y - margin) * N // inner
    return WARP if (r + c) % 2 == 0 else WEFT


def png(size):
    raw = b"".join(
        b"\x00" + bytes(v for x in range(size) for v in cell_colour(x, y, size)) for y in range(size)
    )
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9))
            + chunk(b"IEND", b""))


def svg():
    hexc = lambda c: "#%02x%02x%02x" % c
    s, m = 96, 12
    step = (s - 2 * m) / N
    rects = "".join(
        f'<rect x="{m + c * step:.0f}" y="{m + r * step:.0f}" width="{step:.0f}" height="{step:.0f}" '
        f'fill="{hexc(WARP if (r + c) % 2 == 0 else WEFT)}"/>'
        for r in range(N) for c in range(N)
    )
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {s} {s}">'
            f'<rect width="{s}" height="{s}" rx="18" fill="{hexc(BG)}"/>{rects}</svg>\n')


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for size in (192, 512):
        (OUT / f"icon-{size}.png").write_bytes(png(size))
    (OUT / "icon.svg").write_text(svg())
    print("icons written to", OUT)
