"""
Extract curves from a vector (matplotlib) PDF figure with mutool.

    python extract_vector_limit.py fig.pdf                    # calibration and the stroked paths
    python extract_vector_limit.py fig.pdf "R G B" [width]    # data points of the paths with that colour

Used for NuSTAR_1908.09037.dat, NuSTAR_2207.04572.dat, NuSTAR_straylight_2405.17861.dat and
eROSITA_LMC_2603.19109.dat (figures and curve colours in their headers; drop the vertical segments
to the frame at the ends of a curve).

Axes are calibrated from the tick marks (short stroked segments on the frame) and the numeric
tick labels next to them; log axes (labels "10" + exponent, exponent sign taken negative for
the y axis, as in every figure used here) or plain numbers on a log x axis.
"""
import html
import re
import subprocess
import sys
import numpy as np

pdf = sys.argv[1]
trace = subprocess.run(["mutool", "trace", pdf], capture_output=True, text=True).stdout
stext = subprocess.run(["mutool", "draw", "-F", "stext", "-o", "-", pdf], capture_output=True, text=True).stdout


def paths():
    out = []
    for m in re.finditer(r'<(stroke_path|fill_path)([^>]*)>(.*?)</\1>', trace, re.S):
        kind, attrs, body = m.groups()
        a = dict(re.findall(r'(\w+)="([^"]*)"', attrs))
        t = [float(v) for v in a.get("transform", "1 0 0 1 0 0").split()]
        pts = []
        for op in re.finditer(r'<(moveto|lineto|curveto)([^/]*)/>', body):
            v = dict(re.findall(r'(\w+)="([^"]*)"', op.group(2)))
            if op.group(1) == "curveto":
                x, y = float(v["x3"]), float(v["y3"])
            else:
                x, y = float(v["x"]), float(v["y"])
            pts.append((t[0] * x + t[2] * y + t[4], t[1] * x + t[3] * y + t[5]))
        out.append(dict(kind=kind, color=a.get("color", ""), cs=a.get("colorspace", ""),
                        lw=float(a.get("linewidth", "0") or 0), pts=np.array(pts)))
    return out


def labels():
    """(text, x_center, y_center, size) of every text line."""
    out = []
    for m in re.finditer(r'<line bbox="([^"]*)"[^>]*>(.*?)</line>', stext, re.S):
        x0, y0, x1, y1 = map(float, m.group(1).split())
        chars = re.findall(r'<char quad="([^"]*)"[^>]*c="([^"]*)"', m.group(2))
        out.append((html.unescape("".join(c for _, c in chars)), chars, (x0 + x1) / 2, (y0 + y1) / 2, y1 - y0))
    return out


P = paths()
L = labels()

ticks = [p for p in P if p["kind"] == "stroke_path" and len(p["pts"]) == 2]
vt = [p for p in ticks if abs(p["pts"][0, 0] - p["pts"][1, 0]) < 0.05 and 1 < abs(p["pts"][0, 1] - p["pts"][1, 1]) < 15]
ht = [p for p in ticks if abs(p["pts"][0, 1] - p["pts"][1, 1]) < 0.05 and 1 < abs(p["pts"][0, 0] - p["pts"][1, 0]) < 15]


def on_frame(tk, k):
    """Ticks that start on one of the two most common frame lines (coordinate k)."""
    from collections import Counter
    frames = [v for v, _ in Counter(round(p["pts"][0, k], 1) for p in tk).most_common(2)]
    return [p for p in tk if any(abs(p["pts"][0, k] - f) < 0.2 for f in frames)]


vt, ht = on_frame(vt, 1), on_frame(ht, 0)

# x axis: numeric labels (plain numbers) below the lowest horizontal extent of the plot
xl = [(float(t), cx, cy) for t, _, cx, cy, _ in L if re.fullmatch(r"\d+(\.\d+)?", t)]
yl = []
for t, chars, cx, cy, h in L:
    m = re.fullmatch(r"10[\u2212-]?(\d+)", t)
    if m:                                   # "10" followed by the exponent (negative on every y axis used)
        yl.append((-float(m.group(1)), cx, cy))
    elif t == "10":                         # "10" and the exponent as separate lines
        x1 = float(chars[-1][0].split()[2])
        for t2, ch2, cx2, cy2, h2 in L:
            x0 = float(ch2[0][0].split()[0]) if ch2 else 1e9
            if re.fullmatch(r"\d+", t2) and 0 <= x0 - x1 < 12 and 0 < cy - cy2 < 1.2 * h:
                yl.append((-float(t2), cx, cy))
# y labels: rows with "10"+exponent; x labels: plain numbers at the bottom (largest cy)
if xl:
    ybottom = max(c[2] for c in xl)
    xl = [c for c in xl if abs(c[2] - ybottom) < 3]
    # labels 10^n read as "10n" (e.g. 102, 103, 104 keV): all start with "10" and are longer
    if all(str(int(v)).startswith("10") and len(str(int(v))) > 2 for v, _, _ in xl):
        xl = [(10.0 ** int(str(int(v))[2:]), cx, cy) for v, cx, cy in xl]


def calib(lab, tk, axis):
    """Fit log10(value) = a + b * coord from labels matched to the nearest major tick."""
    pos = [p["pts"][0, 0] if axis == "x" else p["pts"][0, 1] for p in tk]
    lens = [np.ptp(p["pts"][:, 1 if axis == "x" else 0]) for p in tk]
    major = max(lens) if lens else 0
    # x: labels sit at their tick, major or minor; y: decade labels, always at major ticks
    pos = np.array(pos if axis == "x" else [q for q, l in zip(pos, lens) if l > 0.75 * major])
    xs, vs = [], []
    for v, cx, cy in lab:
        c = cx if axis == "x" else cy
        if len(pos) == 0:
            continue
        j = np.argmin(np.abs(pos - c))
        if abs(pos[j] - c) < 12:
            xs.append(pos[j]); vs.append(np.log10(v) if axis == "x" else v)
    xs, vs = np.array(xs), np.array(vs)
    b, a = np.polyfit(xs, vs, 1)
    res = vs - (a + b * xs)
    return a, b, xs, vs, np.max(np.abs(res)) if len(res) else np.nan


ax_ = calib(xl, vt, "x")
ay_ = calib(yl, ht, "y")

if len(sys.argv) == 2:
    print("x calib: %d labels, max residual %.2e dex" % (len(ax_[2]), ax_[4]), list(zip(np.round(ax_[2], 2), np.round(10 ** ax_[3], 3))))
    print("y calib: %d labels, max residual %.2e dex" % (len(ay_[2]), ay_[4]), list(zip(np.round(ay_[2], 2), ay_[3])))
    from collections import Counter
    c = Counter((p["kind"], p["color"], round(p["lw"], 2)) for p in P if len(p["pts"]) > 10)
    for (k, col, lw), n in sorted(c.items(), key=lambda t: -t[1]):
        pts = np.vstack([p["pts"] for p in P if (p["kind"], p["color"], round(p["lw"], 2)) == (k, col, lw) and len(p["pts"]) > 10])
        mx = 10 ** (ax_[0] + ax_[1] * pts[:, 0])
        my = 10 ** (ay_[0] + ay_[1] * pts[:, 1])
        print(f"{k:11s} color={col:22s} lw={lw:5.2f} paths={n:3d} pts={len(pts):5d}  x {mx.min():.3g}-{mx.max():.3g}  y {my.min():.2g}-{my.max():.2g}")
else:
    col = sys.argv[2]
    lw = float(sys.argv[3]) if len(sys.argv) > 3 else None
    kind = sys.argv[4] if len(sys.argv) > 4 else None
    sel = [p for p in P if p["color"] == col and len(p["pts"]) > 10 and (lw is None or abs(p["lw"] - lw) < 0.01)
           and (kind is None or p["kind"] == kind)]
    for i, p in enumerate(sel):
        print(f"# path {i}: {p['kind']} {len(p['pts'])} points")
        for x, y in p["pts"]:
            print(f"{10 ** (ax_[0] + ax_[1] * x):.6g} {10 ** (ay_[0] + ay_[1] * y):.6g}")
