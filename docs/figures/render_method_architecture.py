"""Render the implementation-grounded E5 architecture as SVG, PNG and PDF.

Run from any directory with matplotlib and Noto Sans CJK installed.
The figure describes model and supervision structure, not runtime scheduling.
"""
from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.lines import Line2D


W, H = 1800, 1630
INK = "#17293E"
MUTED = "#58697D"
LINE = "#8A98A9"
BLUE = "#2A65A2"
TEAL = "#117D76"
AMBER = "#A86A11"
ROSE = "#A84466"
VIOLET = "#72559D"
BG = "#FFFFFF"
PALE_BLUE = "#F0F6FC"
PALE_TEAL = "#EEF8F5"
PALE_AMBER = "#FFFAF0"
PALE_ROSE = "#FCF1F5"
PALE_VIOLET = "#F6F3FB"


def render(out_dir, regular, bold):
    reg = font_manager.FontProperties(fname=str(regular))
    strong = font_manager.FontProperties(fname=str(bold))
    plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42,
                         "axes.unicode_minus": False})
    fig = plt.figure(figsize=(18, 16.3), dpi=100, facecolor=BG)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    def text(x, y, value, size=15, color=INK, weight=False, ha="left", va="top", **kw):
        return ax.text(x, y, value, fontsize=size, color=color,
                       fontproperties=strong if weight else reg,
                       ha=ha, va=va, zorder=8, **kw)

    def box(x, y, w, h, color=BLUE, fill=PALE_BLUE, dashed=False, radius=11):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle=f"round,pad=0,rounding_size={radius}",
                     facecolor=fill, edgecolor=color, linewidth=1.25,
                     linestyle=(0, (5, 3)) if dashed else "solid", zorder=5))

    def arrow(points, color=LINE, dashed=False, width=1.8, head=True, z=2):
        xs, ys = zip(*points)
        style = (0, (5, 4)) if dashed else "solid"
        ax.add_line(Line2D(xs, ys, color=color, linewidth=width, linestyle=style,
                          solid_capstyle="round", solid_joinstyle="round", zorder=z))
        if head:
            ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                         mutation_scale=15, color=color, linewidth=width,
                         linestyle=style, shrinkA=0, shrinkB=0, zorder=z))

    def dot(x, y, color):
        ax.add_patch(Circle((x, y), 4, facecolor=color, edgecolor="none", zorder=4))

    def section(y, number, label):
        text(60, y, number, 15, BLUE, True)
        text(101, y, label, 16, INK, True)

    # Header and compact visual key.
    text(60, 38, "Suction20Net：几何与实例感知的吸取评分", 28, INK, True)
    text(60, 95, "公开数据预训练  →  真实尝试微调  →  实际吸取姿态排序", 17, MUTED)
    box(1510, 41, 230, 44, TEAL, PALE_TEAL, radius=8)
    text(1625, 63, "E5 完整模型", 16, TEAL, True, ha="center", va="center")
    arrow([(1130, 117), (1190, 117)], LINE, width=1.5)
    text(1203, 107, "前向信息", 13, MUTED)
    arrow([(1360, 117), (1420, 117)], VIOLET, dashed=True, width=1.5)
    text(1433, 107, "训练监督", 13, MUTED)
    ax.add_line(Line2D([60, 1740], [148, 148], color="#DCE3EC", linewidth=1))

    section(170, "01", "共享场景编码与实例上下文")

    # Scene encoding path.
    box(60, 215, 325, 190)
    text(81, 237, "观测与实例输入", 19, BLUE, True)
    text(81, 285, "深度 → xyz / 观测法向", 15)
    text(81, 319, "实例 ID / 法向有效性 / g", 15)
    text(81, 367, "RGB 仅用于前置实例分割", 13, MUTED)

    box(430, 215, 390, 190)
    text(452, 237, "2 mm 稀疏几何编码", 19, BLUE, True)
    text(452, 285, "8 维体素特征 → ResUNet14", 15)
    text(452, 319, "512 → 256 → 映射回原始点", 15)
    text(452, 365, "局部点特征  F_local：N × 256", 14, BLUE, True)

    box(865, 215, 420, 190, TEAL, PALE_TEAL)
    text(887, 237, "实例上下文融合", 19, TEAL, True)
    text(887, 285, "逐实例 mean / max → 128 维", 15)
    text(887, 319, "局部点 + 实例特征 + 位置 / 尺寸 / g", 13.7)
    text(887, 365, "393 → 256 → F_fused", 15, TEAL, True)

    box(1330, 215, 410, 80, TEAL, PALE_TEAL)
    text(1350, 228, "初始质量 q0", 17, TEAL, True)
    text(1350, 264, "F_fused (256) → 128 → 1", 13.5)
    box(1330, 325, 410, 80, BLUE, PALE_BLUE)
    text(1350, 338, "吸取轴 n_pred", 17, BLUE, True)
    text(1350, 374, "F_local + 观测法向 → 切向残差", 13.5)

    arrow([(385, 310), (430, 310)], BLUE)
    arrow([(820, 310), (865, 310)], BLUE)
    arrow([(1285, 255), (1330, 255)], TEAL)
    # Fused center feature skip into pose fusion; not used by axis prediction.
    arrow([(1285, 382), (1303, 382), (1303, 850), (1330, 850)], TEAL)
    # Local features feed the axis head and zone encoder independently.
    arrow([(625, 405), (625, 440), (1314, 440), (1314, 365), (1330, 365)], BLUE)
    text(847, 415, "F_local → 轴分支", 12.5, BLUE)

    section(468, "02", "姿态条件评分：候选动作与真实执行动作共用同一评分网络")

    # Supporting geometry and two clearly separate action sources.
    box(60, 505, 325, 150, BLUE, PALE_BLUE)
    text(81, 526, "全场景几何上下文", 18, BLUE, True)
    text(81, 574, "原始点 xyz / 实例 ID", 15)
    text(81, 611, "保留台面与邻物障碍", 14, MUTED)
    arrow([(60, 330), (35, 330), (35, 580), (60, 580)], BLUE)

    box(430, 505, 390, 150, BLUE, PALE_BLUE)
    text(452, 526, "区域内点特征", 18, BLUE, True)
    text(452, 574, "F_local：256 → 64 维", 15)
    text(452, 611, "配合每点 7 维姿态几何描述", 14, MUTED)
    arrow([(840, 440), (840, 580), (820, 580)], BLUE)
    dot(840, 440, BLUE)

    box(865, 505, 420, 150, ROSE, PALE_ROSE)
    text(887, 525, "真实执行动作", 18, ROSE, True)
    text(887, 570, "p_exec / n_exec / instance_id", 15)
    text(887, 609, "仅目标微调；直接评分执行姿态", 13.8, ROSE)

    box(1330, 505, 410, 150, TEAL, PALE_TEAL)
    text(1350, 525, "候选动作 · 最多 512", 18, TEAL, True)
    text(1350, 564, "推理：q0 + 实例均衡 + FPS", 13.5)
    text(1350, 594, "训练：标签池 / 提议混合采样", 13.5)
    text(1350, 625, "候选 p + 预测轴 n_pred", 12.5, TEAL)
    arrow([(1740, 255), (1770, 255), (1770, 575), (1740, 575)], TEAL)
    arrow([(1740, 365), (1770, 365)], BLUE, head=False)
    dot(1770, 365, BLUE)

    # Explicit point/axis input controls all zone geometry. Neighbor points only
    # provide features; the executed center is never snapped to another action.
    arrow([(1075, 655), (1075, 735)], ROSE)
    arrow([(1535, 655), (1535, 695), (1215, 695), (1215, 735)], TEAL)
    text(890, 700, "按阶段选择 (p, n)", 13, MUTED)
    arrow([(222, 655), (222, 822), (430, 822)], BLUE)
    text(239, 790, "原始点 / ID", 13, BLUE)
    arrow([(625, 655), (625, 735)], BLUE)

    # Pose-conditioned contact features, with the actual five-zone sampling.
    box(430, 735, 855, 245, AMBER, PALE_AMBER)
    text(452, 754, "R20 五区姿态条件编码", 19, AMBER, True)
    text(452, 793, "原始点查询 → FPS → 点 MLP → masked pooling", 14, MUTED)
    zones = [("中心接触区", "core", 32), ("唇口环带", "lip", 64),
             ("杯体区域", "cup", 32), ("安装座", "mount", 32),
             ("接近通道", "approach", 32)]
    for i, (label, name, count) in enumerate(zones):
        x = 452 + i * 163
        box(x, 834, 146, 80, AMBER, "#FFFFFF", radius=7)
        text(x + 73, 846, label, 14, AMBER, True, ha="center")
        text(x + 73, 879, f"{name} ≤{count} 点", 11.2, MUTED, ha="center")
    text(452, 936, "5 × 64 维区域特征 + 8 维唇口扇区统计 = 328 维", 15, AMBER, True)

    box(1330, 735, 410, 245, AMBER, PALE_AMBER)
    text(1350, 756, "姿态特征融合", 19, AMBER, True)
    text(1350, 810, "F_fused(p) 256 + 五区 328", 14.5)
    text(1350, 850, "吸取轴 n：3 + 重力 g：3", 14.5)
    text(1350, 919, "590 → 256 → 256 = h", 17, AMBER, True)
    arrow([(1285, 895), (1330, 895)], AMBER)

    # Parallel prediction heads. Auxiliary heads are not multiplied into q_B.
    box(60, 1040, 350, 155, BLUE, PALE_BLUE)
    text(81, 1061, "物理质量头 q_B", 19, BLUE, True)
    text(81, 1112, "h → 128 → 1 → sigmoid", 14)
    text(81, 1155, "独立预测连续物理质量", 14, MUTED)

    box(455, 1040, 675, 155, VIOLET, PALE_VIOLET)
    text(477, 1060, "E5 物理辅助头", 19, VIOLET, True)
    text(477, 1108, "Seal", 16, VIOLET, True)
    text(477, 1145, "core / lip", 13.5)
    text(651, 1108, "Wrench", 16, VIOLET, True)
    text(651, 1145, "h + 实例 / 动作", 13.5)
    text(865, 1108, "Collision-free", 16, VIOLET, True)
    text(865, 1145, "障碍区 + 局部 / 轴", 13)

    box(1175, 1040, 565, 155, ROSE, PALE_ROSE, dashed=True)
    text(1197, 1060, "任务成功头（真实成败微调时新增）", 17.5, ROSE, True)
    text(1197, 1110, "[ h 256, n 3, n·g 1 ] → 260 → 128 → 1", 14)
    text(1197, 1155, "task_success logit → sigmoid 任务排序分", 14, ROSE)

    # Shared h reaches independent q/task outputs. Region/instance/local inputs
    # enter the auxiliary-head bundle separately, as named inside the bundle.
    arrow([(1535, 980), (1535, 1010), (235, 1010), (235, 1040)], AMBER)
    arrow([(1535, 1010), (1535, 1040)], AMBER)
    dot(1535, 1010, AMBER)
    arrow([(950, 1010), (950, 1040)], AMBER)
    dot(950, 1010, AMBER)
    arrow([(795, 980), (795, 1040)], VIOLET)
    text(669, 987, "区域特征", 11.5, VIOLET,
         bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})
    text(1235, 986, "共享姿态特征 h", 11.5, AMBER)

    section(1240, "03", "预训练、微调与部署")

    box(60, 1290, 520, 235, BLUE, PALE_BLUE)
    text(82, 1312, "阶段 1 · 公开数据预训练", 19, BLUE, True)
    text(82, 1360, "离线标签：GT-B = S × W × (1 − C)", 15)
    text(82, 1400, "监督 q0 / q_B / axis 与 S-W-C 辅助头", 14.5)
    text(82, 1465, "物理标签用于损失与采样；场景特征来自观测", 13, MUTED)

    box(640, 1290, 520, 235, ROSE, PALE_ROSE)
    text(662, 1312, "阶段 2 · 少量真实场景微调", 19, ROSE, True)
    text(662, 1360, "真实成败 → task BCE；源域回放维持物理头", 13.8)
    text(662, 1400, "每更新：2 帧 target + 2 帧 source", 15)
    text(662, 1438, "L = mean(L_target) + 0.5 mean(L_source)", 12.9, ROSE, True)
    text(662, 1485, "Encoder / BN 冻结；200 步后开放 decoder", 12.7, MUTED)

    box(1220, 1290, 520, 235, TEAL, PALE_TEAL)
    text(1242, 1312, "阶段 3 · 实际部署推理", 19, TEAL, True)
    text(1242, 1360, "预测候选 → 任务分数排序 → NMS", 15)
    text(1242, 1400, "输出：接触点 p、外向轴 n、score", 15)
    text(1242, 1465, "未进行真实成败微调时，按 q_B 排序", 14, MUTED)
    arrow([(580, 1405), (640, 1405)], LINE)
    arrow([(1160, 1405), (1220, 1405)], LINE)

    # Sparse loss arrows illustrate the split between physical and task labels.
    arrow([(320, 1290), (320, 1195)], BLUE, dashed=True, width=1.5)
    arrow([(500, 1290), (500, 1195)], VIOLET, dashed=True, width=1.5)
    arrow([(1050, 1290), (1050, 1220), (1460, 1220), (1460, 1195)], ROSE,
          dashed=True, width=1.5)

    ax.add_line(Line2D([60, 1740], [1550, 1550], color="#DCE3EC", linewidth=1))
    text(60, 1570, "R20 = 半径 20 mm（直径 40 mm） · 8 维输入不含 RGB · q_B 与任务分数独立，不相乘", 13, MUTED)
    text(60, 1602, "实现依据：chichu-lv/suctionnet_modern_20mm @ cfbe7ab  |  2026-11-16", 11.5, MUTED)
    text(1740, 1602, "E3：省略实例融合与物理辅助头", 11.5, MUTED, ha="right")

    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"Title": "Suction20Net method architecture",
                "Author": "Suction20Net project",
                "Subject": "E5 shared geometry encoder, pose scoring, pretraining and real-attempt adaptation"}
    stem = out_dir / "method-architecture"
    fig.savefig(stem.with_suffix(".svg"), format="svg", facecolor=BG,
                metadata={"Title": metadata["Title"], "Description": metadata["Subject"]})
    svg = stem.with_suffix(".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines())
                   + "\n", encoding="utf-8")
    fig.savefig(stem.with_suffix(".png"), format="png", dpi=180, facecolor=BG)
    fig.savefig(stem.with_suffix(".pdf"), format="pdf", facecolor=BG, metadata=metadata)
    plt.close(fig)
    for suffix in ("svg", "png", "pdf"):
        path = stem.with_suffix('.' + suffix)
        print(f"{path}: {path.stat().st_size} bytes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--font-regular", type=Path,
                        default=Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"))
    parser.add_argument("--font-bold", type=Path,
                        default=Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"))
    args = parser.parse_args()
    render(args.output_dir, args.font_regular, args.font_bold)
