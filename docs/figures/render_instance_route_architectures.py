"""Render the v3 instance-route method figures as editable vector SVG.

Geometry and output poses are deterministic schematics, not experiment data.
Use --export to additionally generate 4400x2700 PNG and 220x135 mm vector PDF
with Inkscape. Both SVGs keep their text editable and use no external images.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess

import numpy as np
import svgwrite

W, H = 2200, 1350
C = dict(ink="#243343", muted="#657386", line="#667789", blue="#487CB7",
         teal="#359F92", orange="#E6A252", violet="#8B79BA", rose="#CC758D",
         grid="#D9E1E8", pale="#F5F8FB", gray="#A7B4C0")
SOURCE_COMMIT = "c23f4b81b88d6646e40e49248d2df0b30026d70b"


def mix(color, amount=.7):
    color = C.get(color, color)
    rgb = [int(color[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (255 - x) * amount):02x}" for x in rgb)


class Figure:
    def __init__(self, output, route):
        self.output_path = output
        self.d = svgwrite.Drawing(str(output), size=("220mm", "135mm"),
                                  viewBox=f"0 0 {W} {H}")
        self.d.set_desc(title=f"Route {route.upper()} instance suction architecture, design v3",
                        desc="Deterministic schematic. Not a trained-model result. "
                             f"Architecture source: {SOURCE_COMMIT}.")
        self.d.add(self.d.rect((0, 0), (W, H), fill="white"))
        self.arrows = {}
        self.text_index = 0
        for key in C:
            marker = self.d.marker(id=f"arrow-{key}", insert=(9, 4), size=(10, 8),
                                   orient="auto", markerUnits="userSpaceOnUse")
            marker.add(self.d.path(d="M 0 0 L 9 4 L 0 8 Z", fill=C[key]))
            self.d.defs.add(marker)
            self.arrows[key] = marker.get_funciri()

    def text(self, x, y, value, size=23, color="ink", weight="normal", anchor="start"):
        self.text_index += 1
        item = self.d.text(value, insert=(x, y), id=f"label-{self.text_index}",
                           font_size=size, font_family="DejaVu Sans, Arial, sans-serif",
                           fill=C.get(color, color), font_weight=weight, text_anchor=anchor)
        self.d.add(item)
        return item

    def math(self, x, y, base, sub="", size=28, color="ink", anchor="middle"):
        self.text_index += 1
        item = self.d.text("", insert=(x, y), id=f"label-{self.text_index}",
                           font_size=size, font_family="DejaVu Sans, Arial, sans-serif",
                           fill=C.get(color, color), text_anchor=anchor)
        item.add(self.d.tspan(base, font_style="italic"))
        if sub:
            item.add(self.d.tspan(sub, font_size=size * .67, baseline_shift="sub"))
        self.d.add(item)
        return item

    def line(self, pts, color="line", width=2, arrow=False, dash=None, opacity=1):
        item = self.d.polyline(pts, fill="none", stroke=C.get(color, color),
                               stroke_width=width, stroke_linejoin="round",
                               stroke_linecap="round", opacity=opacity)
        if arrow:
            item["marker-end"] = self.arrows.get(color, self.arrows["line"])
        if dash:
            item.dasharray(dash)
        self.d.add(item)
        return item

    def path(self, value, color="line", width=2, dash=None, opacity=1, fill="none"):
        item = self.d.path(d=value, fill=fill, stroke=C.get(color, color),
                           stroke_width=width, opacity=opacity)
        if dash:
            item.dasharray(dash)
        self.d.add(item)
        return item

    def box(self, x, y, w, h, color="blue", fill=None, dash=None, radius=7):
        item = self.d.rect((x, y), (w, h), rx=radius,
                           fill=fill or mix(color, .92), stroke=C.get(color, color),
                           stroke_width=1.8)
        if dash:
            item.dasharray(dash)
        self.d.add(item)
        return item

    def dot(self, x, y, color="blue", r=3.5, opacity=1, hollow=False):
        self.d.add(self.d.circle((float(x), float(y)), r,
                                fill="white" if hollow else C.get(color, color),
                                stroke=C.get(color, color) if hollow else "none",
                                stroke_width=1.3, opacity=opacity))

    def ellipse(self, cx, cy, rx, ry, color, fill=None, width=2, dash=None):
        item = self.d.ellipse((cx, cy), (rx, ry), fill=fill or "none",
                              stroke=C.get(color, color), stroke_width=width)
        if dash:
            item.dasharray(dash)
        self.d.add(item)
        return item

    def vector(self, x, y, width=115, height=20, color="orange", cells=8):
        for i in range(cells):
            self.d.add(self.d.rect((x + i * width / cells, y), (width / cells - 1, height),
                                   fill=mix(color, .18 + .5 * ((i * 3) % 7) / 6),
                                   stroke=C.get(color, color), stroke_width=.7))

    def prism(self, x, y, w, h, color="blue", depth=11, grid=5):
        col = C.get(color, color)
        self.d.add(self.d.polygon([(x, y), (x + depth, y - depth),
                                   (x + w + depth, y - depth), (x + w, y)],
                                  fill=mix(col, .55), stroke=col, stroke_width=1.5))
        self.d.add(self.d.polygon([(x + w, y), (x + w + depth, y - depth),
                                   (x + w + depth, y + h - depth), (x + w, y + h)],
                                  fill=mix(col, .2), stroke=col, stroke_width=1.5))
        self.d.add(self.d.rect((x, y), (w, h), fill=mix(col, .73),
                               stroke=col, stroke_width=1.5))
        for j in range(1, grid):
            self.line([(x, y + h * j / grid), (x + w, y + h * j / grid)],
                      color, 1, opacity=.42)

    def stack(self, x, y, height=95, color="blue", n=3, width=24, gap=10):
        for i in range(n - 1, -1, -1):
            self.prism(x + i * gap, y - i * 5, width, height, color, depth=7)

    def cloud(self, cx, cy, scale=105, output=False):
        rng = np.random.default_rng(42)
        pts, colors = [], []
        # One rounded package, with a curved top and two visible side faces.
        for face, count in ((0, 270), (1, 95), (2, 95)):
            u, v = rng.uniform(-1, 1, (2, count))
            if face == 0:
                keep = (u / 1.05) ** 6 + (v / .79) ** 6 <= 1
                u, v = u[keep], v[keep]
                z = .48 + .06 * np.cos(2 * u) * np.cos(2 * v)
                arr = np.c_[u, v, z]
            elif face == 1:
                arr = np.c_[u, np.full(count, .72), .26 + .25 * v]
            else:
                arr = np.c_[np.full(count, -.98), .72 * u, .26 + .25 * v]
            for p in arr:
                pts.append(p)
                colors.append("#8EA4B4" if output else
                              C["teal"] if p[0] < .15 else C["orange"])
        pts = np.asarray(pts)
        def project(p):
            return cx + scale * (.85 * p[0] - .43 * p[1]), cy + scale * (.25 * p[0] + .43 * p[1] - p[2])
        for i in np.argsort(pts[:, 1] - .18 * pts[:, 0]):
            px, py = project(pts[i])
            self.dot(px, py, colors[i], r=1.7, opacity=.83)
        if output:
            p = np.array([.12, -.02, .54])
            px, py = project(p)
            self.ellipse(px, py, 18, 7, "teal", fill="white", width=2.6)
            self.line([(px, py), (px - 7, py - 72)], "teal", 2.4, arrow=True)
            self.text(px + 8, py - 66, "n*", 21, "teal")
            self.dot(px, py, "ink", 3.5)
            self.text(px + 22, py + 24, "p*", 20)
        else:
            for p in [(-.3, .1, .53), (.4, -.1, .52)]:
                px, py = project(np.array(p))
                self.line([(px, py), (px - 5, py - 35)], "line", 1.8, arrow=True)

    def image_tile(self, x, y, w, h, variant="rgb", outline=True):
        self.box(x, y, w, h, "grid", fill="#F9FBFD", radius=4)
        # A single capsule-shaped target; no camera image is claimed.
        xx, yy = x + .18 * w, y + .12 * h
        ow, oh = .64 * w, .76 * h
        col = "teal" if variant == "rgb" else "blue"
        self.d.add(self.d.rect((xx, yy), (ow, oh), rx=.2 * ow,
                               fill=mix(col, .55), stroke=C[col], stroke_width=1.2))
        for j in range(1, 10):
            sy = yy + j * oh / 10
            self.line([(xx + 5, sy), (xx + ow - 5, sy)],
                      "orange" if variant == "rgb" and j in (3, 6) else col,
                      2 if j in (3, 6) else .8, opacity=.7)
        if outline:
            self.d.add(self.d.rect((xx - 5, yy - 5), (ow + 10, oh + 10), rx=.22 * ow,
                                   fill="none", stroke=C["gray"], stroke_dasharray="4 4",
                                   stroke_width=1.1))

    def fusion(self, x, y, dim, hidden):
        self.d.add(self.d.circle((x, y), 16, fill="white", stroke=C["ink"], stroke_width=2))
        self.text(x, y + 7, "C", 22, weight="bold", anchor="middle")
        self.line([(x + 17, y), (x + 47, y)], "ink", 2.3, arrow=True)
        self.stack(x + 49, y - 45, height=94, color="orange", n=2, width=23)
        self.math(x + 76, y + 88, "h", "pose", 27, "orange")
        self.text(x + 76, y + 119, f"{dim} → {hidden}", 20, "muted", anchor="middle")

    def headings(self, title, detail):
        self.text(40, 46, title, 32, weight="bold")
        self.text(2155, 43, "DESIGN v3", 19, "muted", anchor="end")
        self.line([(40, 628), (2160, 628)], "grid", 1.2)
        self.line([(1360, 675), (1360, 1298)], "grid", 1.2)
        self.text(40, 681, detail, 31, weight="bold")
        self.text(1400, 681, "(c) Same-pose training and adaptation", 29, weight="bold")
        self.text(40, 1331, "Schematic geometry and poses; production models and performance remain unvalidated.", 18, "muted")
        self.text(2160, 1331, "C: concatenate + MLP", 18, "muted", anchor="end")

    def output(self):
        self.text(2050, 110, "Selected pose", 25, anchor="middle")
        self.cloud(2060, 310, 96, output=True)
        self.text(2050, 408, "g = [R | p*]", 25, anchor="middle")
        self.text(2050, 444, "+Z_tool = -n*", 20, "muted", anchor="middle")
        self.text(2050, 477, "roll: fixed convention", 18, "muted", anchor="middle")
        self.box(1890, 231, 96, 88, "line", fill="white")
        self.text(1938, 267, "rank", 22, anchor="middle")
        self.text(1938, 296, "3D NMS", 18, anchor="middle")
        self.line([(1988, 275), (2015, 275)], "ink", 2.4, arrow=True)
        self.text(1938, 360, "stage-specific", 17, "muted", anchor="middle")

    def heads(self, route):
        # Only q and task enter ranking. Auxiliary heads keep their own inputs.
        hx, hy = 1490, 263
        self.text(1765, 110, "Physics and task", 25, anchor="middle")
        self.box(1700, 171, 146, 65, "blue")
        self.math(1773, 212, "q", "inst", 29, "blue")
        self.box(1700, 285, 146, 65, "rose")
        self.math(1773, 326, "s", "task", 28, "rose")
        self.text(1773, 377, "+ n, n·g" if route == "a" else "task logit", 18, "muted", anchor="middle")
        self.line([(1590, 263), (1650, 263), (1650, 203), (1700, 203)], "blue", 2.2, arrow=True)
        self.line([(1650, 263), (1650, 318), (1700, 318)], "rose", 2.2, arrow=True)
        self.dot(1650, 263, "ink")
        self.line([(1847, 203), (1870, 203), (1870, 253), (1890, 253)], "blue", 2, arrow=True)
        self.line([(1847, 318), (1870, 318), (1870, 291), (1890, 291)], "rose", 2, arrow=True)
        self.text(1858, 306, "σ", 18, "rose", anchor="middle")
        self.text(1871, 278, "or", 15, "muted", anchor="middle")
        for x, name, col in ((1360, "Seal", "orange"), (1520, "Wrench", "blue"), (1700, "log b", "violet")):
            self.box(x, 440, 146, 52, col)
            self.text(x + 73, 474, name, 23, col, anchor="middle")
        self.line([(1650, 318), (1650, 412), (1773, 412), (1773, 440)], "violet", 1.8, arrow=True)
        self.line([(1650, 412), (1593, 412), (1593, 440)], "blue", 1.8, arrow=True)
        if route == "a":
            self.line([(1340, 334), (1340, 412), (1433, 412), (1433, 440)], "orange", 2, arrow=True)
            self.text(1433, 524, "regions + lip (200)", 17, "muted", anchor="middle")
            self.text(1593, 524, "+ G, Δp, n, g", 17, "muted", anchor="middle")
        else:
            self.line([(1593, 412), (1433, 412), (1433, 440)], "orange", 1.8, arrow=True)
            self.text(1600, 524, "five independent heads from h_pose", 19, "muted", anchor="middle")
        self.text(1773, 524, "Laplace scale" if route == "a" else "", 17, "muted", anchor="middle")

    def training(self):
        self.text(1405, 732, "Source physics", 23, "blue", "bold")
        self.image_tile(1410, 765, 110, 76, "rgb")
        self.text(1465, 870, "observation", 18, "muted", anchor="middle")
        self.line([(1522, 802), (1560, 802)], "blue", 2, arrow=True)
        self.box(1560, 763, 175, 78, "blue")
        self.text(1647, 792, "shared scorer θ", 21, "blue", anchor="middle")
        self.text(1647, 821, "canonical (p, n)", 18, "blue", anchor="middle")
        self.line([(1737, 802), (1770, 802)], "blue", 2, arrow=True)
        self.text(1850, 792, "q, S, W, log b", 22, "blue", anchor="middle")
        self.text(1850, 823, "four predictions", 17, "muted", anchor="middle")
        self.line([(1940, 802), (1980, 802)], "blue", 2, arrow=True)
        self.box(1980, 770, 145, 65, "blue", fill="white", radius=24)
        self.math(2052, 810, "L", "phys", 28, "blue")
        self.box(1455, 914, 670, 62, "gray", fill="white", dash="5 4")
        self.text(1790, 939, "Offline teacher: complete target + identical (p, n)", 21, "muted", anchor="middle")
        self.text(1790, 965, "Q = S W (1 - C_self); supervise q / S / W", 20, "blue", anchor="middle")
        self.line([(2052, 912), (2052, 838)], "blue", 1.8, arrow=True, dash="5 4")
        self.text(1920, 888, "labels only", 17, "muted", anchor="middle")
        self.text(1405, 1026, "Target planned-action adaptation", 23, "rose", "bold")
        self.ellipse(1460, 1100, 28, 10, "rose", fill=mix("rose", .85), width=2)
        self.line([(1460, 1100), (1452, 1052)], "rose", 2.3, arrow=True)
        self.text(1460, 1148, "(p_a, n_a)", 21, "rose", anchor="middle")
        self.text(1460, 1177, "planned pose", 17, "muted", anchor="middle")
        self.line([(1500, 1095), (1560, 1095)], "rose", 2, arrow=True)
        self.box(1560, 1056, 175, 78, "rose")
        self.text(1647, 1085, "same scorer θ", 21, "rose", anchor="middle")
        self.text(1647, 1114, "+ observation", 18, "rose", anchor="middle")
        self.line([(1737, 1095), (1775, 1095)], "rose", 2, arrow=True)
        self.math(1840, 1103, "s", "task", 27, "rose")
        self.line([(1900, 1095), (1980, 1095)], "rose", 2, arrow=True)
        self.box(1980, 1061, 145, 67, "rose", fill="white", radius=24)
        self.text(2052, 1103, "Task BCE", 23, "rose", anchor="middle")
        self.text(2052, 1182, "outcome y", 20, "muted", anchor="middle")
        self.line([(2052, 1160), (2052, 1132)], "rose", 1.7, arrow=True, dash="5 4")
        self.line([(2127, 802), (2165, 802), (2165, 1237), (1867, 1237)], "blue", 1.8, arrow=True, dash="6 5")
        self.line([(2127, 1095), (2141, 1095), (2141, 1210), (1852, 1210), (1852, 1223)], "rose", 1.8, arrow=True)
        self.d.add(self.d.circle((1852, 1237), 14, fill="white", stroke=C["line"], stroke_width=1.7))
        self.text(1852, 1245, "+", 24, anchor="middle")
        self.text(2030, 1267, "0.5 × source replay", 18, "blue", anchor="middle")
        self.text(1750, 1300, "L_ft = L_task + 0.5 L_source", 25, anchor="middle")
        # Historical pose bypass: condition the common query port in panel (a).
        self.line([(1430, 1094), (1380, 1094), (1380, 596), (930, 596), (930, 536)],
                  "rose", 1.9, arrow=True, dash="6 5")
        self.text(1160, 588, "recorded pose: bypass proposals", 19, "rose", anchor="middle")

    def save(self):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.d.save(pretty=True)
        text = self.output_path.read_text(encoding="utf-8")
        self.output_path.write_text("\n".join(line.rstrip() for line in text.splitlines()) + "\n", encoding="utf-8")


def route_a(output):
    f = Figure(output, "a")
    f.headings("(a) Route A - single-object sparse 3D + RGB scoring",
               "(b) Three-region contact encoding and lip harmonics")
    for x, label in ((160, "Target instance"), (465, "Sparse ResUNet14"),
                     (860, "Object / query features"), (1215, "Contact encoding")):
        f.text(x, 110, label, 25, anchor="middle")
    # Shared encoding and raw-support bypass.
    f.line([(255, 255), (315, 255)], "ink", 2.5, arrow=True)
    f.cloud(150, 290, 108)
    f.text(160, 356, "RGB-D, mask, K, g", 20, "muted", anchor="middle")
    tower = [(320, 194, 31, 105), (375, 232, 32, 81), (430, 267, 35, 57),
             (488, 232, 32, 81), (543, 194, 31, 105)]
    for i, (x, y, w, h) in enumerate(tower):
        if i < len(tower) - 1:
            nx, ny, _, nh = tower[i + 1]
            f.line([(x + w + 11, y + h * .6), (nx - 3, ny + nh * .6)], "blue", 1.8, arrow=True)
        f.prism(x, y, w, h, "violet" if i == 2 else "blue")
    f.path("M 339 180 Q 451 120 559 180", "blue", 1.5, dash="5 4")
    f.path("M 395 220 Q 448 178 505 220", "blue", 1.5, dash="5 4")
    f.text(462, 356, "2-mm voxels; FPS ≤ 2048", 20, "muted", anchor="middle")
    f.text(462, 383, "12-D point input", 19, "muted", anchor="middle")
    f.line([(585, 250), (632, 250)], "blue", 2.4, arrow=True)
    f.stack(635, 207, 95)
    f.math(665, 344, "F", "point", 26, "blue")
    f.line([(686, 250), (710, 250), (710, 189), (750, 189)], "blue", 2, arrow=True)
    f.line([(710, 250), (710, 286), (750, 286)], "blue", 2, arrow=True)
    f.dot(710, 250)
    f.box(750, 155, 230, 65, "blue")
    f.text(865, 182, "Masked object pool", 21, "blue", anchor="middle")
    f.text(865, 208, "G [64]", 21, "blue", anchor="middle")
    f.box(750, 251, 230, 69, "blue")
    f.text(865, 280, "Query feature readout", 20, "blue", anchor="middle")
    f.text(865, 307, "Fq [256] + valid", 21, "blue", anchor="middle")
    f.line([(982, 188), (1020, 188), (1020, 145), (1490, 145), (1490, 246)], "blue", 2.1, arrow=True)
    f.line([(982, 286), (1020, 286), (1020, 188)], "blue", 2)
    f.text(1250, 137, "query semantics + object context", 19, "blue", anchor="middle")
    f.line([(710, 286), (710, 365), (1080, 365), (1080, 294), (1110, 294)], "blue", 1.9, arrow=True)
    f.text(900, 357, "associated point features", 18, "blue", anchor="middle")
    f.line([(160, 368), (160, 408), (1130, 408), (1130, 353)], "gray", 2, arrow=True)
    f.text(665, 431, "raw support + RGB (before backbone sampling)", 20, "muted", anchor="middle")
    f.line([(160, 408), (160, 500), (310, 500)], "gray", 1.8, arrow=True)
    f.dot(160, 408, "gray")
    f.box(310, 462, 420, 77, "teal")
    f.text(520, 493, "v3 normals + spatial refill", 23, "teal", anchor="middle")
    f.text(520, 521, "raw p, observed n; ≤256 queries", 20, "teal", anchor="middle")
    f.line([(733, 500), (850, 500)], "teal", 2.1, arrow=True)
    f.box(850, 466, 160, 70, "ink", fill="white")
    f.text(930, 509, "(p, n)", 28, anchor="middle")
    f.line([(930, 466), (930, 337), (865, 337), (865, 322)], "teal", 1.7, arrow=True)
    f.line([(1012, 500), (1210, 500), (1210, 373)], "teal", 2, arrow=True)
    f.line([(1010, 522), (1055, 522), (1055, 558), (1490, 558), (1490, 282)], "gray", 1.6, arrow=True)
    f.text(1300, 551, "pose descriptor [10]", 18, "muted", anchor="middle")
    f.box(1110, 210, 240, 162, "grid", fill="white", dash="4 5")
    for i, (name, color) in enumerate((('core', 'blue'), ('inner', 'teal'), ('lip', 'orange'))):
        y = 235 + i * 31
        f.text(1130, y + 15, name, 18, color)
        f.vector(1192, y, 140, 18, color)
    f.text(1230, 349, "+ raw lip summary [8]", 18, "orange", anchor="middle")
    f.line([(1352, 263), (1472, 263)], "orange", 2.4, arrow=True)
    f.text(1410, 246, "192 + 8", 18, "orange", anchor="middle")
    f.fusion(1490, 263, 531, 256)
    f.heads("a")
    f.output()
    contact_a(f)
    f.training()
    f.save()


def contact_a(f):
    rng = np.random.default_rng(17)
    cx, cy, rx, ry = 225, 900, 132, 57
    for _ in range(210):
        x, y = rng.uniform(-1.25, 1.25, 2)
        if x*x + y*y < 1.7:
            f.dot(cx + 147*x, cy + 61*y + 6*np.sin(x*4), "gray", 1.5, .65)
    f.ellipse(cx, cy, rx, ry, "orange", fill=mix("orange", .76))
    f.ellipse(cx, cy, rx*18/22, ry*18/22, "teal", fill=mix("teal", .79))
    f.ellipse(cx, cy, rx*10/22, ry*10/22, "blue", fill=mix("blue", .72))
    for _ in range(100):
        a = rng.uniform(0, 2*np.pi); r = np.sqrt(rng.uniform(0, 1))
        col = "blue" if r < 10/22 else "teal" if r < 18/22 else "orange"
        f.dot(cx + rx*r*np.cos(a), cy + ry*r*np.sin(a), col, 1.9)
    f.line([(cx, cy), (cx + 17, 741)], "ink", 2.3, arrow=True)
    f.text(cx + 32, 750, "n", 25)
    f.dot(cx, cy, "ink", 4)
    f.text(cx - 21, cy + 28, "p", 23)
    f.text(110, 800, "core", 21, "blue", anchor="middle")
    f.line([(126, 810), (194, 870)], "blue", 1.5)
    f.text(362, 823, "inner", 21, "teal", anchor="middle")
    f.line([(347, 832), (300, 865)], "teal", 1.5)
    f.text(384, 940, "lip", 21, "orange", anchor="middle")
    f.line([(365, 930), (347, 922)], "orange", 1.5)
    f.text(225, 1002, "core / inner / lip", 22, "muted", anchor="middle")
    f.text(225, 1032, "0-10 / 10-18 / 18-22 mm", 19, "muted", anchor="middle")
    f.text(225, 1060, "R = 20 mm; |h_i| ≤ 40 mm", 19, "muted", anchor="middle")
    f.line([(389, 885), (430, 885)], "ink", 2.1, arrow=True)
    f.text(472, 737, "region FPS", 20, "muted", anchor="middle")
    f.line([(431, 795), (431, 945)], "line", 1.3)
    for j, (name, col, count) in enumerate((('core', 'blue', 32), ('inner', 'teal', 32), ('lip', 'orange', 64))):
        y = 798 + j*72
        f.line([(431, y), (452, y)], col, 1.6, arrow=True)
        for px, py in rng.normal(size=(19, 2))*[17, 9]:
            f.dot(488 + px, y + py, col, 2)
        f.text(488, y + 35, f"≤{count}", 17, "muted", anchor="middle")
        f.line([(526, y), (601, y)], col, 1.6, arrow=True)
        f.line([(730, y), (765, y)], col, 1.7, arrow=True)
        f.line([(895, y), (937, y)], col, 1.7, arrow=True)
        f.vector(942, y - 10, 137, 21, col)
    f.box(601, 757, 130, 233, "blue")
    f.text(666, 839, "shared", 22, "blue", anchor="middle")
    f.text(666, 872, "point MLP", 23, "blue", "bold", "middle")
    f.text(666, 911, "features", 18, "muted", anchor="middle")
    f.text(666, 939, "+ RGB + geometry", 16, "muted", anchor="middle")
    f.box(765, 757, 130, 233, "orange")
    f.text(830, 840, "masked", 21, "orange", anchor="middle")
    f.text(830, 873, "mean / max", 21, "orange", anchor="middle")
    f.text(830, 911, "+ count / valid", 17, "muted", anchor="middle")
    f.text(1012, 1019, "3 × 64-D regions", 21, "muted", anchor="middle")
    f.path("M 1096 775 L 1111 775 L 1111 960 L 1096 960", "line", 1.7)
    f.line([(1112, 867), (1150, 867)], "orange", 2, arrow=True)
    f.stack(1155, 824, 86, "orange", n=2)
    f.text(1190, 960, "192-D", 22, "orange", anchor="middle")
    # Independent raw-lip path is before regional FPS.
    f.line([(348, 918), (394, 918), (394, 1110), (449, 1110)], "orange", 2, arrow=True)
    f.text(212, 1130, "Raw lip before FPS", 22, "orange", anchor="middle")
    f.text(212, 1162, "source-ID deduplication", 18, "muted", anchor="middle")
    f.text(675, 1087, "Same heights, different angular arrangements", 20, "muted", anchor="middle")
    for index, label in enumerate(("one-sided", "quadrants", "alternating")):
        x, y = 512 + index*145, 1171
        for j in range(32):
            theta = 2*np.pi*(j+.5)/32
            positive = j < 16 if index == 0 else (j//8)%2 == 0 if index == 1 else j%2 == 0
            f.dot(x + 39*np.cos(theta), y + 39*np.sin(theta), "orange" if positive else "blue", 3.8)
        f.text(x, 1234, label, 17, "muted", anchor="middle")
    f.line([(865, 1171), (935, 1171)], "orange", 2, arrow=True)
    f.box(935, 1119, 346, 107, "teal")
    f.text(1108, 1152, "Centered RMS + |A1, A2, A3|", 21, "teal", anchor="middle")
    f.text(1108, 1183, "+ count / gap / validity", 21, "teal", anchor="middle")
    f.text(1108, 1213, "8-D lip summary", 21, "teal", "bold", "middle")
    f.text(680, 1283, "Basis-invariant low-order summary; not a complete angular shape code.", 21, "muted", anchor="middle")


def route_b(output):
    f = Figure(output, "b")
    f.headings("(a) Route B - shared 2D encoding with projected-ring queries",
               "(b) Perspective sampling, support gates and retained raw evidence")
    for x, label in ((145, "Target RGB-D"), (590, "Shared image encoding"),
                     (920, "Object / query features"), (1230, "Projected contact")):
        f.text(x, 110, label, 25, anchor="middle")
    f.image_tile(55, 183, 180, 153, "rgb")
    f.text(145, 368, "mask, K, g", 20, "muted", anchor="middle")
    # Full-instance context and native texture are two once-per-object paths.
    f.line([(237, 212), (270, 212), (270, 188), (310, 188)], "blue", 2.1, arrow=True)
    f.image_tile(310, 144, 145, 104, "depth")
    f.text(382, 278, "13 × 128 × 128", 19, "blue", anchor="middle")
    f.line([(457, 195), (500, 195)], "blue", 2.2, arrow=True)
    for i in range(4):
        x, y, h = 506 + 59*i, 153 + i*17, 104 - i*17
        if i < 3:
            f.line([(x + 40, y + h*.55), (x + 55, y + 17 + (h-17)*.55)], "blue", 1.7, arrow=True)
        f.prism(x, y, 29, h, "blue", depth=9)
    f.text(618, 133, "ResNet-18 encoder / GN", 18, "blue", anchor="middle")
    f.line([(724, 226), (770, 226), (770, 166), (810, 166)], "blue", 1.9, arrow=True)
    f.box(810, 135, 230, 62, "blue")
    f.text(925, 161, "L4 object pooling", 20, "blue", anchor="middle")
    f.text(925, 186, "G [128]", 21, "blue", anchor="middle")
    # Query semantics tap L1 and L2 explicitly, before deeper object pooling.
    f.line([(520, 257), (520, 275), (752, 275), (752, 267), (810, 267)], "blue", 1.9, arrow=True)
    f.line([(579, 257), (579, 275)], "blue", 1.9)
    f.dot(579, 275, "blue", r=2.8)
    f.text(520, 298, "L1", 15, "blue", anchor="middle")
    f.text(579, 298, "L2", 15, "blue", anchor="middle")
    f.box(810, 231, 230, 68, "blue")
    f.text(925, 259, "L1/L2 query sampling", 20, "blue", anchor="middle")
    f.text(925, 286, "Fq [64]", 21, "blue", anchor="middle")
    f.line([(1042, 166), (1080, 166), (1080, 135), (1490, 135), (1490, 246)], "blue", 2.1, arrow=True)
    f.line([(1042, 266), (1080, 266), (1080, 166)], "blue", 1.9)
    f.line([(237, 287), (270, 287), (270, 385), (310, 385)], "orange", 2, arrow=True)
    f.image_tile(310, 343, 145, 102, "rgb")
    f.text(382, 474, "native ROI / 5 ch", 19, "orange", anchor="middle")
    f.line([(458, 392), (514, 392)], "orange", 2.1, arrow=True)
    f.box(514, 353, 190, 82, "orange")
    f.text(609, 386, "Shared texture stem", 20, "orange", anchor="middle")
    f.text(609, 415, "stride 1; computed once", 17, "muted", anchor="middle")
    f.line([(707, 392), (758, 392)], "orange", 2, arrow=True)
    f.stack(762, 350, 85, "orange", n=3)
    f.math(797, 473, "F", "tex", 25, "orange")
    f.line([(817, 392), (1050, 392), (1050, 329), (1120, 329)], "orange", 2.1, arrow=True)
    f.text(939, 416, "native texture [16]", 18, "orange", anchor="middle")
    f.line([(237, 307), (250, 307), (250, 320), (1100, 320), (1100, 284), (1120, 284)], "gray", 1.8, arrow=True)
    f.text(634, 316, "original depth + masks", 18, "muted", anchor="middle")
    # Canonical proposals and historical pose share the same query port.
    f.line([(145, 382), (145, 526), (310, 526)], "gray", 1.8, arrow=True)
    f.box(310, 498, 394, 65, "teal")
    f.text(507, 524, "v3 normals + spatial refill", 22, "teal", anchor="middle")
    f.text(507, 550, "raw p, observed n; ≤256 queries", 18, "teal", anchor="middle")
    f.line([(707, 530), (850, 530)], "teal", 2.1, arrow=True)
    f.box(850, 466, 160, 70, "ink", fill="white")
    f.text(930, 508, "(p, n)", 28, anchor="middle")
    f.line([(930, 466), (930, 301)], "teal", 1.8, arrow=True)
    f.line([(1012, 499), (1210, 499), (1210, 374)], "teal", 2, arrow=True)
    f.line([(1010, 521), (1055, 521), (1055, 558), (1490, 558), (1490, 281)], "gray", 1.6, arrow=True)
    f.text(1280, 551, "query descriptor [12]", 18, "muted", anchor="middle")
    f.box(1120, 207, 230, 165, "grid", fill="white", dash="4 5")
    for i, (label, col) in enumerate((('contact 96', 'orange'), ('support 18', 'teal'), ('raw 12', 'teal'))):
        y = 227 + 38*i
        f.text(1130, y + 15, label, 17, col)
        f.vector(1230, y, 105, 18, col)
    f.text(1235, 356, "read depth; gate; retain raw", 16, "muted", anchor="middle")
    f.line([(1353, 263), (1472, 263)], "orange", 2.4, arrow=True)
    f.text(1410, 245, "96 + 18 + 12", 17, "orange", anchor="middle")
    f.fusion(1490, 263, 330, 128)
    f.heads("b")
    f.output()
    contact_b(f)
    f.training()
    f.save()


def contact_b(f):
    # Cross-section: optical rays are explicitly oblique to the query normal.
    O, p, ideal = (115, 767), (345, 1005), (449, 968.6)
    observed = (509.12, 1004.888)
    f.box(67, 730, 72, 47, "line", fill="white", radius=4)
    f.path("M 139 740 L 164 750 L 139 765 Z", "line", 1.5, fill=mix("gray", .7))
    f.text(106, 713, "camera O", 21, "muted", anchor="middle")
    f.line([O, p], "gray", 1.5, dash="5 4")
    f.line([O, ideal, observed], "blue", 2.1, arrow=True)
    f.text(276, 840, "camera ray", 19, "blue", anchor="middle")
    f.line([(116, 1085), (549, 933.6)], "teal", 1.9, dash="6 4")
    f.text(414, 919, "ideal ring plane", 20, "teal", anchor="middle")
    f.path("M 65 1060 C 180 1050 263 1001 345 1005 C 421 1009 473 984 509.12 1004.888 L 565 1030",
           "orange", 3.2)
    f.text(299, 1113, "observed surface", 22, "orange", anchor="middle")
    f.line([p, (293, 856)], "ink", 2.1, arrow=True)
    f.text(278, 852, "n", 25)
    f.dot(*p, "ink", r=4)
    f.text(328, 1033, "p", 23)
    f.dot(*ideal, "teal", r=5, hollow=True)
    f.text(470, 957, "x_r", 20, "teal")
    f.dot(*observed, "orange", r=5)
    f.text(540, 997, "x_obs", 20, "orange")
    # Orthogonal foot in this 2D schematic, for the observed height.
    e = np.array([1., -.35]); e /= np.linalg.norm(e)
    delta = np.array(observed) - np.array(p)
    foot = np.array(p) + e * np.dot(delta, e)
    f.line([observed, tuple(foot)], "muted", 1.5, dash="3 3")
    f.text(535, 1048, "h_i, ρ", 22, "muted")
    f.text(290, 1186, "Camera ray ≠ suction axis", 23, "muted", anchor="middle")
    f.text(290, 1220, "Actual depth determines h_i and ρ", 19, "muted", anchor="middle")
    # Observed samples split before contact masking; rejected height survives.
    f.line([(568, 1026), (590, 1026), (590, 827), (620, 827)], "orange", 1.9, arrow=True)
    f.box(620, 786, 250, 80, "orange")
    f.text(745, 817, "Observed samples", 22, "orange", anchor="middle")
    f.text(745, 847, "F_tex + depth + masks", 19, "muted", anchor="middle")
    f.line([(872, 826), (930, 826)], "orange", 2.1, arrow=True)
    f.dot(895, 826, "orange")
    f.box(930, 786, 255, 80, "orange")
    f.text(1057, 817, "Radial / angular gate", 22, "orange", anchor="middle")
    f.text(1057, 848, "actual geometry + validity", 18, "muted", anchor="middle")
    f.line([(895, 826), (895, 732), (1280, 732), (1280, 1180), (1250, 1180)],
           "teal", 1.8, arrow=True)
    f.text(1080, 723, "raw h_i, m_obs (before gate)", 18, "teal", anchor="middle")
    f.line([(1057, 868), (1057, 927)], "orange", 2, arrow=True)
    f.text(1110, 905, "valid contact", 17, "orange", anchor="middle")
    f.box(930, 928, 255, 82, "orange")
    f.text(1057, 958, "Circular Conv1d + LN", 22, "orange", anchor="middle")
    f.text(1057, 987, "LN per angle; masked pool", 17, "muted", anchor="middle")
    f.text(1057, 1042, "3 regions → 96-D", 21, "orange", anchor="middle")
    f.line([(975, 868), (975, 890), (750, 890), (750, 940)], "teal", 1.8, arrow=True)
    f.text(750, 922, "72 slots / radius", 18, "teal", anchor="middle")
    # The missing arc wraps across the zero-angle seam in this illustration.
    for j in range(72):
        a = 2*np.pi*(j+.5)/72
        missing = j >= 54 or j < 18
        f.dot(750 + 72*np.cos(a), 1021 + 72*np.sin(a),
              "gray" if missing else "orange", r=2.4, hollow=missing)
    f.ellipse(750, 1021, 48, 48, "grid", width=1)
    f.text(750, 1017, "explicit", 18, "teal", anchor="middle")
    f.text(750, 1041, "missing arc", 18, "teal", anchor="middle")
    f.line([(750, 1096), (750, 1160)], "teal", 1.9, arrow=True)
    f.box(600, 1160, 290, 80, "teal")
    f.text(745, 1193, "Coverage + cyclic gap", 22, "teal", anchor="middle")
    f.text(745, 1222, "9 radii × 2 = 18-D", 20, "teal", anchor="middle")
    f.line([(1187, 826), (1320, 826), (1320, 1100), (1140, 1100), (1140, 1125)],
           "muted", 1.6, arrow=True, dash="4 4")
    f.text(1102, 1090, "rejection mask", 17, "muted", anchor="middle")
    f.box(950, 1125, 300, 115, "teal")
    f.text(1100, 1156, "Raw observation summary", 21, "teal", anchor="middle")
    f.text(1100, 1186, "observed / rejected / ±h_i", 19, "teal", anchor="middle")
    f.text(1100, 1222, "3 regions × 4 = 12-D", 20, "teal", anchor="middle")
    f.text(800, 1283, "Rejected contact support retains observed protrusion / depression evidence.", 20, "muted", anchor="middle")


def validate_design():
    directory = Path(__file__).resolve().parents[1] / "instance_routes" / "reference"
    a = json.loads((directory / "route_a.json").read_text(encoding="utf-8"))
    b = json.loads((directory / "route_b.json").read_text(encoding="utf-8"))
    if a["model"]["query_input_dim"] != 531 or a["model"]["heads"]["seal"][0] != 200:
        raise ValueError("Route A design changed; review the figure before rendering")
    if b["model"]["query_input_dim"] != 330 or b["model"]["ring_summary_dim"] != 30:
        raise ValueError("Route B design changed; review the figure before rendering")
    if a["quality_contract"] != "gt_INST20_pose_v2" or b["quality_contract"] != a["quality_contract"]:
        raise ValueError("The physical-quality contract changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", choices=("a", "b", "both"), default="both")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--export", action="store_true", help="Also export PNG and vector PDF with Inkscape")
    args = parser.parse_args()
    validate_design()
    if args.export and not shutil.which("inkscape"):
        parser.error("--export requires the Inkscape CLI")
    outputs = []
    for route in (("a", "b") if args.route == "both" else (args.route,)):
        path = args.output_dir / f"instance-route-{route}-architecture.svg"
        (route_a if route == "a" else route_b)(path)
        outputs.append(str(path))
        if args.export:
            for ext in ("png", "pdf"):
                cmd = ["inkscape", str(path), f"--export-type={ext}",
                       f"--export-filename={path.with_suffix('.' + ext)}"]
                if ext == "png":
                    cmd += ["--export-width=4400", "--export-height=2700"]
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                outputs.append(str(path.with_suffix('.' + ext)))
    print(json.dumps({"outputs": outputs, "source_design_commit": SOURCE_COMMIT,
                      "canvas_mm": [220, 135], "schematic_only": True}, indent=2))


if __name__ == "__main__":
    main()
