"""Draw a publication-style Suction20Net method figure as editable SVG.

Point clouds are deterministic schematic geometry, not experimental predictions.
Run this script, then use Inkscape to export the SVG to PNG/PDF.
"""
from pathlib import Path
import argparse
import math

import numpy as np
import svgwrite


W, H = 1900, 1000
C = dict(ink="#243343", muted="#657386", line="#667789", blue="#487CB7",
         teal="#359F92", orange="#E6A252", violet="#8B79BA", rose="#CC758D",
         grid="#D9E1E8", pale="#F5F8FB", gray="#A7B4C0")
ZONES = [C["blue"], C["orange"], C["teal"], C["violet"], C["rose"]]


def mix(color, amount=.7):
    rgb = [int(color[i:i+2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x+(255-x)*amount):02x}" for x in rgb)


def draw(output):
    d = svgwrite.Drawing(str(output), size=("190mm", "100mm"), viewBox=f"0 0 {W} {H}")
    d.add(d.rect((0, 0), (W, H), fill="white"))
    arrows = {}
    for key in ["ink", "muted", "line", "blue", "teal", "orange", "violet", "rose", "gray"]:
        marker = d.marker(insert=(9, 4), size=(10, 8), orient="auto", markerUnits="userSpaceOnUse")
        marker.add(d.path(d="M 0 0 L 9 4 L 0 8 Z", fill=C[key]))
        d.defs.add(marker)
        arrows[key] = marker.get_funciri()
        arrows[C[key]] = marker.get_funciri()

    def text(x, y, value, size=23, color="ink", weight="normal", anchor="start", italic=False):
        item = d.text(value, insert=(x, y), font_size=size, font_family="DejaVu Sans, Arial, sans-serif",
                      fill=C.get(color, color), font_weight=weight, text_anchor=anchor,
                      font_style="italic" if italic else "normal")
        d.add(item)
        return item

    def mathlabel(x, y, base, sub="", size=26, color="ink", anchor="middle"):
        item = d.text("", insert=(x, y), font_size=size, font_family="DejaVu Sans, Arial, sans-serif",
                      fill=C.get(color, color), text_anchor=anchor)
        item.add(d.tspan(base, font_style="italic"))
        if sub:
            item.add(d.tspan(sub, font_size=size*.68, baseline_shift="sub"))
        d.add(item)

    def mathseq(x, y, parts, size=23, color="ink", anchor="middle"):
        item = d.text("", insert=(x, y), font_size=size,
                      font_family="DejaVu Sans, Arial, sans-serif",
                      fill=C.get(color, color), text_anchor=anchor)
        for base, sub in parts:
            item.add(d.tspan(base, font_style="italic" if base.strip() in {"p", "n", "n̂", "q", "L"} else "normal"))
            if sub:
                item.add(d.tspan(sub, font_size=size*.68, baseline_shift="sub"))
        d.add(item)

    def line(points, color="line", width=2, dash=None, arrow=False, opacity=1):
        item = d.polyline(points, fill="none", stroke=C.get(color, color), stroke_width=width,
                          stroke_linejoin="round", stroke_linecap="round", opacity=opacity)
        if dash:
            item.dasharray(dash)
        if arrow:
            item["marker-end"] = arrows.get(color, arrows["line"])
        d.add(item)
        return item

    def path(value, color="line", width=2, fill="none", opacity=1, dash=None, arrow=False):
        item = d.path(d=value, fill=fill, stroke=C.get(color, color), stroke_width=width, opacity=opacity)
        if dash:
            item.dasharray(dash)
        if arrow:
            item["marker-end"] = arrows.get(color, arrows["line"])
        d.add(item)

    def box(x, y, w, h, color="blue", fill=None, radius=6, dash=None, width=1.8):
        item = d.rect((x, y), (w, h), rx=radius, fill=fill or mix(C.get(color, color), .91),
                      stroke=C.get(color, color), stroke_width=width)
        if dash:
            item.dasharray(dash)
        d.add(item)

    def dot(x, y, color="blue", r=3.6):
        d.add(d.circle((x, y), r, fill=C.get(color, color)))

    def prism(x, y, w, h, color, depth=11, grid=0):
        color = C.get(color, color)
        d.add(d.polygon([(x, y), (x+depth, y-depth), (x+w+depth, y-depth), (x+w, y)],
                        fill=mix(color, .55), stroke=color, stroke_width=1.5))
        d.add(d.polygon([(x+w, y), (x+w+depth, y-depth), (x+w+depth, y+h-depth), (x+w, y+h)],
                        fill=mix(color, .2), stroke=color, stroke_width=1.5))
        d.add(d.rect((x, y), (w, h), fill=mix(color, .73), stroke=color, stroke_width=1.5))
        for i in range(1, grid):
            line([(x, y+h*i/grid), (x+w, y+h*i/grid)], color, 1, opacity=.45)

    def feature_stack(x, y, h=98, color="blue", n=3, w=24, gap=11):
        for i in range(n-1, -1, -1):
            prism(x+i*gap, y-i*5, w, h, color, depth=7, grid=7)

    def feature_vector(x, y, color, w=104, h=20, cells=8):
        for i in range(cells):
            d.add(d.rect((x+i*w/cells, y), (w/cells-1, h), fill=mix(color, .15+.52*((i*3)%7)/6),
                         stroke=color, stroke_width=.7))

    def cylinder(cx, y_bottom, rx, ry, height, color, alpha=.12):
        color = C.get(color, color)
        yt = y_bottom-height
        body=f"M {cx-rx} {yt} L {cx-rx} {y_bottom} A {rx} {ry} 0 0 0 {cx+rx} {y_bottom} L {cx+rx} {yt} Z"
        path(body, color, 1.6, color, alpha)
        line([(cx-rx, yt), (cx-rx, y_bottom)], color, 1.6, opacity=.7)
        line([(cx+rx, yt), (cx+rx, y_bottom)], color, 1.6, opacity=.7)
        d.add(d.ellipse((cx, yt), (rx, ry), fill=color, fill_opacity=alpha, stroke=color, stroke_width=1.6))
        d.add(d.ellipse((cx, y_bottom), (rx, ry), fill="none", stroke=color, stroke_width=1.3,
                        stroke_dasharray="4 4", opacity=.7))

    def scene(cx, cy, scale=100, output_poses=False):
        rng = np.random.default_rng(42)
        points, colors = [], []
        def add(points_new, color):
            points.extend(points_new)
            colors.extend([color]*len(points_new))
        gx, gy = np.meshgrid(np.linspace(-1.05, 1.02, 29), np.linspace(-.70, .68, 21))
        add(np.c_[gx.ravel(), gy.ravel(), np.zeros(gx.size)], "#C0CAD3")
        # Cuboid: top and two visible side faces.
        for face in range(3):
            u, v = rng.uniform(-.5, .5, (2, 170))
            if face == 0: arr=np.c_[-.43+.65*u, .10+.55*v, np.full(len(u), .40)]
            elif face == 1: arr=np.c_[-.43+.65*u, np.full(len(u), .375), .20+.40*v]
            else: arr=np.c_[np.full(len(u), -.755), .10+.55*u, .20+.40*v]
            add(arr, C["teal"])
        # Cylinder and a rounded object behind it.
        a = rng.uniform(0, 2*np.pi, 260); z = rng.uniform(0, .58, 260)
        add(np.c_[.46+.23*np.cos(a), .06+.23*np.sin(a), z], C["orange"])
        a = rng.uniform(0, 2*np.pi, 160); r = np.sqrt(rng.uniform(0, 1, 160))*.23
        add(np.c_[.46+r*np.cos(a), .06+r*np.sin(a), np.full(len(a), .58)], C["orange"])
        a = rng.uniform(0, 2*np.pi, 210); b = rng.uniform(0, np.pi/2, 210)
        add(np.c_[-.06+.29*np.cos(a)*np.sin(b), -.42+.21*np.sin(a)*np.sin(b), .07+.34*np.cos(b)], C["violet"])
        p=np.array(points)
        def project(q):
            return (cx+scale*(.86*q[0]-.50*q[1]), cy+scale*(.26*q[0]+.46*q[1]-1.12*q[2]))
        for idx in np.argsort(p[:, 1]-.18*p[:, 0]):
            px, py = project(p[idx]); color=colors[idx]
            if output_poses and color!="#C0CAD3": color="#879DAF"
            d.add(d.circle((px, py), 1.15 if color=="#C0CAD3" else 1.45,
                           fill=color, opacity=.60 if color=="#C0CAD3" else .83))
        if output_poses:
            for rank, q, col, tilt in [(1, (.46,.06,.58), C["teal"], -3),
                                       (2, (-.43,.1,.40), C["blue"], 8),
                                       (3, (-.06,-.42,.41), C["orange"], -10)]:
                px,py=project(q)
                d.add(d.ellipse((px,py), (13,5.6), fill="white", fill_opacity=.7, stroke=col, stroke_width=2.8))
                line([(px,py),(px+tilt,py-43)], col, 2.7, arrow=True)
                d.add(d.circle((px+tilt+1,py-60),10,fill="white",stroke=col,stroke_width=1.5))
                text(px+tilt+1,py-56,str(rank),14,col,"bold","middle")
        else:
            for q in [(-.55,.08,.4),(-.25,.16,.4),(.45,.04,.58)]:
                px,py=project(q);line([(px,py),(px-5,py-24)],"line",1.35,arrow=True)

    # (a) Main network. Labels intentionally remain short; full specifications
    # belong in the caption and implementation documentation.
    text(35, 42, "(a)  Geometry- and instance-aware suction scoring", 29, weight="bold")
    text(150, 98, "Observed geometry", 24, anchor="middle")
    text(410, 98, "Sparse U-Net", 24, anchor="middle")
    text(785, 98, "Instance context", 24, anchor="middle")
    text(1170, 98, "Contact encoding", 24, anchor="middle")
    text(1510, 98, "Parallel scores", 24, anchor="middle")
    text(1760, 98, "Ranked poses", 24, anchor="middle")
    box(1075,153,240,170,"grid",fill="white",radius=8,dash="4 5",width=1.3)

    # Background data routes first, so visual modules sit clearly above them.
    line([(240,240),(293,240)],"ink",2.5,arrow=True)
    line([(524,230),(552,230)],"blue",2.5,arrow=True)
    line([(606,222),(632,222),(632,169),(660,169)],"blue",2.2,arrow=True)
    line([(632,222),(632,344),(672,344)],"blue",2.2,arrow=True)
    line([(632,344),(632,447),(1308,447),(1308,324)],"blue",2,arrow=True)
    dot(632,222);dot(632,344)
    line([(606,222),(618,222),(618,128),(823,128),(823,196)],"blue",1.8,arrow=True)
    line([(923,219),(953,219),(953,137),(1355,137),(1355,238)],"teal",2.2,arrow=True)
    line([(901,271),(901,323)],"teal",2.2,arrow=True)
    line([(745,344),(785,344),(785,313),(962,313),(962,333),(978,333)],"blue",2.2,arrow=True)
    line([(929,344),(950,344),(950,354),(978,354)],"teal",2.2,arrow=True)
    line([(1062,344),(1085,344),(1085,324)],"ink",2.2,arrow=True)
    line([(1298,250),(1355,250)],"orange",2.5,arrow=True)
    line([(1381,250),(1404,250)],"ink",2.3,arrow=True)
    line([(1452,249),(1465,249),(1465,197),(1480,197)],"blue",2.2,arrow=True)
    line([(1465,249),(1465,310),(1480,310)],"rose",2.2,arrow=True)
    dot(1465,249,"ink")
    line([(1572,197),(1602,197),(1602,242),(1614,242)],"blue",2.1,arrow=True)
    line([(1572,310),(1602,310),(1602,278),(1614,278)],"rose",2.1,arrow=True)
    line([(1693,260),(1711,260)],"ink",2.4,arrow=True)
    line([(225,306),(280,306),(280,414),(1066,414),(1066,265),(1074,265)],"gray",1.8,arrow=True)
    text(395,403,"raw scene points",19,"muted",anchor="middle")
    text(813,437,"regional point features",19,"blue",anchor="middle")
    text(1138,127,"center feature",18,"teal",anchor="middle")

    scene(150,275,scale=93)
    text(150,350,"xyz, normals, instance IDs",18,"muted",anchor="middle")
    # Encoder-decoder feature towers and skip paths.
    tower=[(295,172,30,103),(344,213,31,79),(394,253,34,55),(445,213,31,79),(493,172,30,103)]
    for i,(x,y,w,h) in enumerate(tower):
        if i<4:
            nx,ny,nw,nh=tower[i+1]
            line([(x+w+10,y+h*.6),(nx-3,ny+nh*.6)],"blue",1.8,arrow=True)
        prism(x,y,w,h,"blue" if i!=2 else "violet",depth=10,grid=5)
    path("M 313 161 Q 408 107 509 161","blue",1.5,dash="5 4",arrow=True)
    path("M 361 201 Q 409 164 460 201","blue",1.5,dash="5 4",arrow=True)
    text(410,334,"2-mm voxels",20,"muted",anchor="middle")
    feature_stack(553,179,105,"blue",3,23,11)
    mathlabel(582,329,"F","local",25,"blue")
    # Instance pooling: colored point features are gathered per identity.
    rng=np.random.default_rng(7)
    for j,col in enumerate([C["teal"],C["orange"],C["violet"]]):
        cy=167+j*48
        for dx,dy in rng.normal(size=(23,2))*[12,8]:
            d.add(d.circle((676+dx,cy+dy),1.9,fill=col))
        line([(696,cy),(717,cy)],"line",1.5,arrow=True)
        feature_vector(726,cy-9,col,w=59,h=17,cells=5)
        line([(788,cy),(801,cy),(801,212),(808,212)],"teal",1.4,arrow=True)
    d.add(d.circle((823,212),14,fill="white",stroke=C["teal"],stroke_width=2))
    text(823,219,"C",19,"teal",weight="bold",anchor="middle")
    line([(837,212),(866,212)],"teal",2,arrow=True)
    feature_stack(867,178,102,"teal",3,23,11)
    text(740,294,"pool / broadcast",18,"muted",anchor="middle")
    mathlabel(897,298,"F","fused",24,"teal")
    # Separate proposal quality and axis prediction, with distinct feature paths.
    box(672,323,73,42,"blue",radius=8)
    text(708,351,"Axis",22,"blue",anchor="middle")
    text(709,392,"+ observed normal",17,"muted",anchor="middle")
    box(873,323,56,42,"teal",radius=8)
    mathlabel(901,350,"q","0",25,"teal")
    box(978,320,84,48,"ink",fill="white",radius=8)
    text(1020,352,"(p, n)",24,"ink",anchor="middle",italic=True)
    text(1020,394,"512 proposals",20,"muted",anchor="middle")
    # Compact pose-specific contact query and encoded region strips.
    cx,base=1127,281
    cylinder(cx,base-33,31,10,68,"rose")
    cylinder(cx,base-9,31,10,24,"violet")
    cylinder(cx,base+2,25,9,11,"teal")
    d.add(d.ellipse((cx,base+1),(27,10),fill=mix(C["orange"],.72),stroke=C["orange"],stroke_width=2))
    d.add(d.ellipse((cx,base+1),(20,7),fill=mix(C["blue"],.6),stroke=C["blue"],stroke_width=1.5))
    dot(cx,base,"ink",3)
    line([(cx,base),(cx-8,base-121)],"ink",1.8,arrow=True)
    text(cx+5,base-118,"n",20,italic=True)
    text(cx+7,base+26,"p",20,italic=True)
    line([(1162,252),(1190,252)],"orange",2,arrow=True)
    for j,col in enumerate(ZONES):
        feature_vector(1199,193+j*25,col,w=97,h=18,cells=7)
    text(1200,342,"five-region features",19,"muted",anchor="middle")
    text(1182,367,"detail in (b)",18,"muted",anchor="middle")
    # Pose fusion. Axis and gravity are explicit geometric conditions.
    d.add(d.circle((1368,250),13,fill="white",stroke=C["ink"],stroke_width=2))
    text(1368,257,"C",19,weight="bold",anchor="middle")
    line([(1368,327),(1368,264)],"gray",1.7,arrow=True)
    text(1368,353,"n, g",22,"muted",anchor="middle",italic=True)
    feature_stack(1404,211,76,"orange",2,20,8)
    mathlabel(1426,324,"h","",25,"orange")
    box(1480,167,92,60,"blue",radius=9)
    mathlabel(1526,204,"q","B",29,"blue")
    text(1526,251,"physical",18,"blue",anchor="middle")
    box(1480,280,92,60,"rose",radius=9)
    mathlabel(1526,317,"s","task",27,"rose")
    text(1526,364,"task",18,"rose",anchor="middle")
    text(1526,391,"+ n, n·g",17,"muted",anchor="middle")
    # Ranking uses the stage-appropriate score, not a product of the two heads.
    box(1614,222,79,76,"line",fill="white",radius=5,width=1.5)
    text(1653,249,"rank",19,"ink",anchor="middle")
    text(1653,277,"NMS",19,"ink",anchor="middle")
    text(1602,264,"or",14,"muted",anchor="middle")
    text(1580,460,"C: concatenate + MLP",18,"muted",anchor="middle")
    scene(1770,271,scale=81,output_poses=True)
    text(1758,391,"(p, n, score)",21,"muted",anchor="middle")

    # Light panel separation; no large text cards.
    line([(35,501),(1870,501)],"grid",1)
    line([(1195,550),(1195,954)],"grid",1)
    text(35,546,"(b)  Five-region contact encoding",28,weight="bold")
    text(1240,546,"(c)  Pretraining and target adaptation",27,weight="bold")

    # (b) Pose-aligned query geometry. Nominal segments: [-2,25], [25,70],
    # [70,150] mm; radii22/28/28 mm. The display compresses axial scale.
    rng=np.random.default_rng(12)
    for _ in range(600):
        x,y=rng.uniform(-1,1,2)
        if x*x+y*y > 1: continue
        px=242+205*x+32*y; py=865+49*y+9*np.sin(x*4)
        d.add(d.circle((px,py),1.35,fill="#B9C6CF",opacity=.72))
    # A neighboring surface indicates that obstacle queries retain scene points.
    for _ in range(65):
        z=rng.uniform(0,1); x=rng.uniform(-1,1)
        d.add(d.circle((310+x*18,850-z*205+x*3),1.55,fill="#AAB7C3",opacity=.6))
    center,base=233,858
    cylinder(center,base-70*1.35,70,22,80*1.35,"rose",.09)
    cylinder(center,base-25*1.35,70,22,45*1.35,"violet",.12)
    cylinder(center,base+2*1.35,55,18,27*1.35,"teal",.14)
    # Contact footprint: core disk and lip annulus.
    d.add(d.ellipse((center,base),(55,18),fill=mix(C["orange"],.35),stroke=C["orange"],stroke_width=2.5))
    d.add(d.ellipse((center,base),(45,14.7),fill=mix(C["blue"],.48),stroke=C["blue"],stroke_width=1.7))
    for _ in range(36):
        a=rng.uniform(0,2*np.pi);r=np.sqrt(rng.uniform(0,1))
        dot(center+42*r*np.cos(a),base+13*r*np.sin(a),"blue",1.8)
    for a in np.linspace(0,2*np.pi,25,endpoint=False):
        dot(center+51*np.cos(a),base+16*np.sin(a),"orange",1.8)
    for y_mid,color in [(base-17,"teal"),(base-65,"violet"),(base-146,"rose")]:
        for _ in range(14):
            x,y=rng.normal(size=2)
            dot(center+42+x*8,y_mid+y*9,color,1.8)
    line([(center,base),(center,620)],"ink",2.1,arrow=True)
    text(center+13,625,"n",25,italic=True)
    dot(center,base,"ink",4.2)
    text(center-18,887,"p",23,italic=True)
    line([(center,base+34),(center+50,base+34)],"ink",1.4)
    line([(center,base+28),(center,base+40)],"ink",1.3)
    line([(center+50,base+28),(center+50,base+40)],"ink",1.3)
    text(center+24,base+62,"R = 20 mm",19,"muted",anchor="middle")
    for value,yy,col in [("approach",704,"rose"),("mount",790,"violet"),("cup",837,"teal")]:
        line([(305,yy-5),(338,yy-5)],col,1.4)
        text(346,yy,value,21,col)
    text(125,854,"core",20,"blue",anchor="end")
    line([(133,850),(183,855)],"blue",1.3)
    text(365,875,"lip",20,"orange")
    line([(289,866),(354,869)],"orange",1.3)
    text(230,970,"Pose-aligned query regions",22,"muted",anchor="middle")
    # Multi-zone features: raw samples -> shared point MLP -> masked pooling.
    line([(419,775),(474,775)],"ink",2.2,arrow=True)
    line([(476,625),(476,845)],"line",1.4)
    text(445,742,"query",18,"muted",anchor="middle")
    text(445,763,"+ FPS",18,"muted",anchor="middle")
    names=["core","lip","cup","mount","approach"]
    for j,(name,col) in enumerate(zip(names,ZONES)):
        yy=625+j*55
        line([(476,yy),(490,yy)],col,1.4,arrow=True)
        text(520,yy-18,name,18,col,anchor="middle")
        for px,py in rng.normal(size=(18,2))*[16,6]:
            dot(520+px,yy+py,col,1.8)
        line([(548,yy),(582,yy)],"line",1.5,arrow=True)
        feature_vector(589,yy-9,C["blue"],w=26,h=18,cells=3)
        feature_vector(620,yy-9,col,w=27,h=18,cells=3)
        line([(654,yy),(686,yy)],"line",1.5,arrow=True)
        line([(778,yy),(817,yy)],"line",1.5,arrow=True)
        feature_vector(930,yy-9,col,w=107,h=18,cells=8)
        line([(899,yy),(923,yy)],"line",1.5,arrow=True)
    box(686,592,92,278,"blue",radius=7)
    box(817,592,82,278,"orange",radius=7)
    text(732,696,"shared",19,"blue",anchor="middle")
    text(732,725,"point",19,"blue",anchor="middle")
    text(732,754,"MLP",23,"blue",weight="bold",anchor="middle")
    text(858,697,"masked",17,"orange",anchor="middle")
    text(858,728,"pool",23,"orange",weight="bold",anchor="middle")
    text(619,903,"local + geometry",17,"muted",anchor="middle")
    # Lip sector statistics form the final 8 components.
    feature_vector(930,903,C["orange"],w=107,h=17,cells=8)
    text(983,948,"lip statistics",18,"muted",anchor="middle")
    path("M 1048 605 L 1062 605 L 1062 920 L 1048 920","line",1.7)
    line([(1063,755),(1091,755)],"orange",2,arrow=True)
    feature_stack(1098,698,112,"orange",2,18,7)
    mathlabel(1121,853,"F","zones",23,"orange")
    text(1121,884,"328-d",19,"muted",anchor="middle")
    text(924,975,"Five region descriptors + lip statistics",19,"muted",anchor="middle")

    # (c) Learning scheme. Recorded action conditions bypass the proposal path.
    # The shared-network block includes the region/instance/local inputs used by
    # auxiliary heads, rather than claiming all auxiliary heads use only h.
    text(1245,594,"Source pretraining",22,"blue",weight="bold")
    # Small stacked source observation icon.
    for i in range(2,-1,-1):
        box(1252+i*5,620-i*5,68,54,"gray",fill="white",radius=3,width=1.1)
    for k in range(21):
        dot(1264+(k%7)*7,634+(k//7)*9,[C["teal"],C["orange"],C["violet"]][k%3],1.7)
    text(1288,700,"source",19,"muted",anchor="middle")
    line([(1335,646),(1370,646)],"blue",2,arrow=True)
    box(1370,616,160,61,"blue",radius=7)
    text(1450,642,"shared network",19,"blue",anchor="middle")
    mathlabel(1450,667,"θ","",23,"blue")
    line([(1531,646),(1560,646)],"blue",2,arrow=True)
    mathseq(1640,638,[("q","0"),(", ",""),("n̂",""),(", ",""),("q","B")],21,"blue")
    text(1640,667,"S / W / C-free",19,"muted",anchor="middle")
    line([(1722,646),(1753,646)],"blue",2,arrow=True)
    box(1756,618,88,57,"blue",fill="white",radius=22)
    mathlabel(1800,654,"L","phys",25,"blue")
    text(1800,710,"physics labels",18,"muted",anchor="middle")
    line([(1800,690),(1800,678)],"blue",1.6,dash="5 4",arrow=True)

    text(1245,765,"Target adaptation",22,"rose",weight="bold")
    # Recorded action icon and target outcome.
    d.add(d.ellipse((1285,837),(25,8),fill=mix(C["rose"],.85),stroke=C["rose"],stroke_width=1.6))
    line([(1285,837),(1278,793)],"rose",2,arrow=True)
    mathseq(1285,876,[("(",""),("p","a"),(", ",""),("n","a"),(")","")],23,"rose")
    text(1285,901,"recorded action",17,"muted",anchor="middle")
    line([(1322,822),(1370,822)],"rose",2,arrow=True)
    box(1370,792,160,61,"rose",radius=7)
    text(1450,817,"shared scorer",19,"rose",anchor="middle")
    text(1450,842,"+ task head",19,"rose",anchor="middle")
    line([(1531,822),(1567,822)],"rose",2,arrow=True)
    mathlabel(1620,830,"s","task",25,"rose")
    line([(1669,822),(1709,822)],"rose",2,arrow=True)
    box(1712,795,87,56,"rose",fill="white",radius=22)
    mathlabel(1755,830,"L","task",25,"rose")
    text(1755,891,"outcome y",19,"muted",anchor="middle")
    line([(1755,870),(1755,855)],"rose",1.6,dash="5 4",arrow=True)
    line([(1450,680),(1450,787)],"gray",1.5,dash="4 5")
    text(1465,742,"shared weights",17,"muted")
    # Source replay is shown as a loss contribution, not a target pseudo-label.
    line([(1848,645),(1866,645),(1866,925),(1574,925)],"blue",1.8,dash="6 5",arrow=True)
    line([(1800,822),(1822,822),(1822,903),(1560,903),(1560,911)],"rose",1.6,arrow=True)
    d.add(d.circle((1560,925),13,fill="white",stroke=C["line"],stroke_width=1.6))
    text(1560,932,"+",22,"ink",anchor="middle")
    line([(1560,939),(1560,951)],"line",1.5,arrow=True)
    text(1742,947,"λ · source replay",18,"blue",anchor="middle")
    mathseq(1560,981,[("L","adapt"),(" = ",""),("L","task"),(" + λ ",""),("L","phys")],24)
    # Route the recorded pose to the common (p,n) action port in panel (a).
    line([(1257,816),(1217,816),(1217,481),(1020,481),(1020,371)],"rose",1.8,dash="6 5",arrow=True)
    text(1121,474,"recorded pose",18,"rose",anchor="middle")

    output.parent.mkdir(parents=True, exist_ok=True)
    d.save(pretty=True)
    output.write_text("\n".join(x.rstrip() for x in output.read_text().splitlines())+"\n",encoding="utf-8")
    print(output)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path(__file__).with_name("paper-architecture.svg"))
    args=parser.parse_args()
    draw(args.output)
