"""Analyze the saved CD010 experiment export; never load a model or run inference.

Dependencies: numpy, pandas, matplotlib. The default data root is this file's
parent directory's parent, matching the published experiment directory.
"""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


SOURCE_COMMIT = "33ee3bc18979772c1f4b3aa4385ecdcbf9a88ab1"


def distribution(values):
    values = np.asarray(values, dtype=float)
    return {"count": len(values), "mean": float(values.mean()),
            "median": float(np.median(values)), "p10": float(np.percentile(values, 10)),
            "p90": float(np.percentile(values, 90)), "p95": float(np.percentile(values, 95))}


def numeric_records(frame):
    return json.loads(frame.to_json(orient="records", double_precision=15))


def analyze(root, output):
    output.mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv(root / "pretraining/run_summary.csv").set_index("experiment")
    nodes = pd.read_csv(root / "pretraining/validation_results.csv")
    frames = pd.read_csv(root / "pretraining/validation_frames.csv")
    target = pd.read_csv(root / "finetuning/validation_metrics.csv").set_index("global_step")
    target_frames = pd.read_csv(root / "finetuning/validation_frames.csv")
    updates = pd.read_csv(root / "finetuning/training_updates.csv")
    predictions = pd.read_csv(root / "inference/frame_metrics.csv")
    baseline = predictions[predictions.model == "pretrained"].set_index("sample_id")
    adapted = predictions[predictions.model == "finetuned"].set_index("sample_id").loc[baseline.index]

    def vectors(series):
        return np.stack(series.map(json.loads))

    position_delta = np.linalg.norm(vectors(adapted.top1_point_cam_m) -
                                   vectors(baseline.top1_point_cam_m), axis=1) * 1000
    a, b = vectors(baseline.top1_normal_out_cam), vectors(adapted.top1_normal_out_cam)
    cosine = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    axis_delta = np.degrees(np.arccos(np.clip(cosine, -1, 1)))

    geometry = {}
    for name, step in [("best", 2880), ("final", 4800)]:
        group = frames[(frames.experiment == "E5") & (frames.global_step == step)]
        total = int(group.visible_instance_count.sum())
        selected = int(group.top1_candidate_count.sum())
        good = int(np.rint(group.instance_Top1_Q20_at_0_4 * group.visible_instance_count).sum())
        collision = int(np.rint(group.collision_rate * group.top1_candidate_count).sum())
        geometry[name] = dict(step=step, frame_count=len(group), visible_instances=total,
                             good_quality=good, colliding_top1=collision,
                             collision_free_but_low_quality=selected-collision-good,
                             missing_candidate=total-selected)

    diffs = []
    for newer, older in [("E1", "E0"), ("E2", "E1"), ("E3", "E1"), ("E4", "E3"), ("E5", "E4")]:
        item = {"comparison": f"{newer}-{older}"}
        for key in ["instance_Top1_Q20_at_0_4", "AP_R20", "collision_rate"]:
            item[key+"_pp"] = float((runs.loc[newer, "final_metrics_"+key] -
                                     runs.loc[older, "final_metrics_"+key]) * 100)
        diffs.append(item)

    rows = [json.loads(line) for line in (root / "inference/original_comparisons.jsonl").read_text().splitlines() if line.strip()]
    comparison = pd.DataFrame([dict(model=row["model"], sample_id=row["sample_id"], **pose)
                               for row in rows for pose in row["comparisons"] if pose["status"] == "compared"])
    original_stats = {}
    for model, group in comparison.groupby("model"):
        original_stats[model] = {key: distribution(group[key]) for key in ["distance_mm", "outward_axis_angle_deg"]}
    original_pair = comparison[comparison.model == "pretrained"].set_index(["sample_id", "original_index"])
    original_fine = comparison[comparison.model == "finetuned"].set_index(["sample_id", "original_index"]).loc[original_pair.index]

    group_losses = target_frames.groupby(["global_step", "group_id"]).agg(
        records=("observation_id", "size"), task_BCE=("task_loss", "mean")).reset_index()
    success_updates = updates[updates.updated].copy()
    success_updates["window_end"] = ((success_updates.global_step - 1) // 100 + 1) * 100
    windows = success_updates.groupby("window_end").agg(
        target_BCE=("loss_target", "mean"), weighted_source_loss=("loss_source", "mean"),
        grad_norm=("grad_norm", "mean")).reset_index()
    unique_observations = {value for row in success_updates.target_observations for value in json.loads(row)}
    prevalence = 209 / 399
    timing = {model: {key: distribution([json.loads(value)[key] for value in group.timing])
                     for key in ["load_seconds", "inference_seconds", "total_seconds"]}
              for model, group in predictions.groupby("model")}

    summary = {
        "source_commit": SOURCE_COMMIT, "analysis_date_beijing": "2026-11-17",
        "scope": "Saved numerical records only; no model inference, training, or new evaluation",
        "pretraining": {"run_results": numeric_records(runs.reset_index()),
                        "final_recipe_differences_pp": diffs,
                        "e5_instance_observation_counts": geometry,
                        "e5_best_final_scene_top1": numeric_records(frames[(frames.experiment == "E5") &
                            frames.global_step.isin([2880, 4800])].groupby(["global_step", "scene_id"])
                            .instance_Top1_Q20_at_0_4.mean().reset_index())},
        "finetuning": {"validation_nodes": numeric_records(target.reset_index()),
                       "auroc_gain": float(target.loc[1000, "attempt_AUROC"]-target.loc[0, "attempt_AUROC"]),
                       "pr_auc_gain": float(target.loc[1000, "attempt_PR_AUC"]-target.loc[0, "attempt_PR_AUC"]),
                       "heads_phase_fraction_of_auroc_gain": float((target.loc[200, "attempt_AUROC"]-target.loc[0, "attempt_AUROC"])/
                                                                    (target.loc[1000, "attempt_AUROC"]-target.loc[0, "attempt_AUROC"])),
                       "validation_prevalence": prevalence,
                       "constant_reference_bce": float(-prevalence*np.log(prevalence)-(1-prevalence)*np.log(1-prevalence)),
                       "constant_reference_brier": prevalence*(1-prevalence),
                       "group_losses": numeric_records(group_losses), "training_windows": numeric_records(windows),
                       "successful_target_draws": 2*len(success_updates), "unique_target_observations": len(unique_observations)},
        "inference": {"frames_each": len(baseline),
                      "same_top1_instance": int((baseline.top1_instance_id == adapted.top1_instance_id).sum()),
                      "same_top1_source_pixel": int((baseline.top1_source_index == adapted.top1_source_index).sum()),
                      "top1_position_change_mm": distribution(position_delta),
                      "top1_axis_change_deg": distribution(axis_delta),
                      "positions_within_mm": {str(t): int((position_delta <= t).sum()) for t in [5, 10, 20, 40, 50]},
                      "joint_within_5mm_5deg": int(((position_delta <= 5) & (axis_delta <= 5)).sum()),
                      "original_pose_geometry": original_stats,
                      "position_closer_to_original": int((original_fine.distance_mm < original_pair.distance_mm).sum()),
                      "axis_closer_to_original": int((original_fine.outward_axis_angle_deg < original_pair.outward_axis_angle_deg).sum()),
                      "timing": timing,
                      "interpretation": "Geometric behavior differences, not measured action success or pose accuracy"},
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return runs, target, geometry, position_delta


def plot(output, runs, target, geometry, distances, font_path):
    font_manager.fontManager.addfont(str(font_path))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=str(font_path)).get_name(),
                         "axes.unicode_minus": False, "svg.fonttype": "none", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    blue, teal, rose, amber, ink = "#316DA4", "#11867D", "#BE5B68", "#D8A452", "#25384C"
    fig, axes = plt.subplots(2, 2, figsize=(14, 10.5))
    fig.subplots_adjust(top=.86, bottom=.11, left=.08, right=.97, hspace=.5, wspace=.28)
    fig.suptitle("Suction20Net：预训练与目标微调结果", x=.08, y=.96, ha="left", fontsize=22, color=ink, fontweight="bold")
    fig.text(.08, .91, "六组预训练已完成；E5 best 用于迁移；微调改善固定历史动作的排序", fontsize=13, color="#52667C")

    ax = axes[0, 0]
    x = np.arange(len(runs))
    final = runs.final_metrics_instance_Top1_Q20_at_0_4.to_numpy()*100
    best = runs.best_metrics_instance_Top1_Q20_at_0_4.to_numpy()*100
    ax.bar(x-.19, final, width=.36, color="#BDCCDC", label="最终模型")
    bars = ax.bar(x+.19, best, width=.36, color=blue, label="按主指标选中的 best")
    ax.bar_label(bars, labels=[f"{v:.2f}" for v in best], fontsize=10, padding=3)
    ax.set_xticks(x, runs.index)
    ax.set_ylim(0, 16)
    ax.set_ylabel("实例 Top-1 Q20≥0.4（%）")
    ax.set_title("A  预训练选模：E5 best 领先", loc="left", pad=14, color=ink)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=.15); ax.set_axisbelow(True)

    ax = axes[0, 1]
    counts = geometry["best"]
    labels = ["Q20≥0.4", "碰撞", "无碰撞但 Q20<0.4", "无候选"]
    values = [counts[k] for k in ["good_quality", "colliding_top1", "collision_free_but_low_quality", "missing_candidate"]]
    bars = ax.barh(np.arange(4), values, color=[teal, rose, amber, "#9CA9B7"], height=.58)
    ax.bar_label(bars, labels=[f"{v:,}" for v in values], padding=5, fontsize=11)
    ax.set_yticks(np.arange(4), labels); ax.invert_yaxis(); ax.set_xlim(0, 1750)
    ax.set_xlabel("320 帧中的实例观测次数，共 2,951 次")
    ax.set_title("B  E5 best：碰撞与非碰撞低质量均需关注", loc="left", pad=14, color=ink)
    ax.grid(axis="x", alpha=.15); ax.set_axisbelow(True)

    ax = axes[1, 0]
    ax.axvspan(0, 200, color="#EAF1F8", zorder=0)
    ax.plot(target.index, target.attempt_AUROC, "o-", color=blue, label="AUROC", linewidth=2)
    ax.plot(target.index, target.attempt_PR_AUC, "s--", color=teal, label="PR-AUC", linewidth=1.7, markersize=5)
    ax.set_ylim(.60, .735); ax.set_xlim(-20, 1040)
    ax.set_xlabel("微调成功更新数（step 0 为原物理 q 基线）")
    ax.set_title("C  399 条固定历史动作：排序有效改善", loc="left", pad=14, color=ink)
    ax.text(70, .729, "仅 heads", fontsize=9, color="#52667C")
    ax.annotate("0.7121", (1000, target.loc[1000, "attempt_AUROC"]), xytext=(720, .725),
                arrowprops={"arrowstyle": "-", "color": blue}, color=blue, fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="lower right"); ax.grid(alpha=.15)

    ax = axes[1, 1]
    ordered = np.sort(distances)
    ax.step(ordered, np.arange(1, len(ordered)+1)/len(ordered)*100, where="post", color=teal, linewidth=2)
    median, p90 = np.percentile(distances, [50, 90])
    ax.axvline(median, linestyle="--", color=blue, linewidth=1)
    ax.axvline(p90, linestyle=":", color=amber, linewidth=1.5)
    ax.text(median+2, 20, f"中位数 {median:.1f} mm", color=blue, fontsize=10)
    ax.text(p90+2, 70, f"P90 {p90:.1f} mm", color="#A27322", fontsize=10)
    ax.set_xlim(0, 100); ax.set_ylim(0, 103)
    ax.set_xlabel("微调前后 Top-1 接触点距离（mm）")
    ax.set_ylabel("累计帧比例（%）")
    ax.set_title("D  401 帧新候选：动作选择发生变化", loc="left", pad=14, color=ink)
    ax.grid(alpha=.15)
    ax.text(.02, .95, "400 / 401 帧仍选择同一实例", transform=ax.transAxes, fontsize=10, va="top", color=ink)
    fig.text(.08, .045, "A 为场景等权指标；B 为实例观测合计。C 衡量历史规划动作排序，D 衡量行为变化；均不是新策略机器人成功率。", fontsize=10, color="#52667C")
    fig.text(.08, .02, "来源：GitHub 33ee3bc · 已保存记录的独立分析 · 2026-11-17（北京时间）", fontsize=9, color="#52667C")
    fig.savefig(output / "overview.png", dpi=180, facecolor="white")
    fig.savefig(output / "overview.svg", facecolor="white")
    svg = output / "overview.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines())+"\n")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--font", type=Path, default=Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"))
    args = parser.parse_args()
    values = analyze(args.data_root, args.output_dir)
    plot(args.output_dir, *values, args.font)
    print(f"Wrote analysis summary and figures to {args.output_dir}")
