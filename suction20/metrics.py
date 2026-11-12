"""Deterministic postprocessing and project R20/target diagnostics (NumPy).

AP-R20 is a fixed-denominator mean of precision@1..50, not PR-curve area.
No metric here filters final predictions by cached gt_valid.
"""
from __future__ import annotations

from collections import defaultdict
import numpy as np
from scipy.integrate import trapezoid


def pose_nms(points, normals, scores, instance_ids, proposal_scores=None,
             point_indices=None, distance=.01, angle_deg=15,
             max_per_instance=10, max_per_scene=50):
    """Return ranked original indices before and after the scene cap.

    Suppression requires BOTH distance and directed angle tests. Score ties
    break by q0 then original observation point index; input order is the final
    stable tie break. Instances are input mask identities, not GT identities.
    """
    p, n, q, instance = map(np.asarray, (points, normals, scores, instance_ids))
    count = len(q)
    q0 = np.zeros(count) if proposal_scores is None else np.asarray(proposal_scores)
    raw = np.arange(count) if point_indices is None else np.asarray(point_indices)
    if p.shape != (count, 3) or n.shape != p.shape or any(a.shape != (count,) for a in (q, instance, q0, raw)):
        raise ValueError("NMS expects aligned pose arrays and one-dimensional scores/identities")
    if not all(np.isfinite(a).all() for a in (p, n, q, q0)):
        raise ValueError("NMS inputs must be finite")
    lengths = np.linalg.norm(n, axis=1)
    if np.any(lengths < 1e-9):
        raise ValueError("NMS requires valid nonzero normals (remove padding first)")
    if distance <= 0 or not 0 < angle_deg < 180 or min(max_per_instance, max_per_scene) < 1:
        raise ValueError("invalid NMS limits")
    n = n / lengths[:, None]
    order = np.lexsort((np.arange(count), raw, -q0, -q))
    retained, by_instance = [], defaultdict(list)
    cosine = np.cos(np.deg2rad(angle_deg))
    for i in order:
        kept = by_instance[int(instance[i])]
        if len(kept) >= max_per_instance:
            continue
        if kept:
            close = np.linalg.norm(p[kept] - p[i], axis=1) < distance
            aligned = n[kept] @ n[i] > cosine
            if np.any(close & aligned):
                continue
        kept.append(int(i))
        retained.append(int(i))
    retained = np.asarray(retained, np.int64)
    return {"per_instance_indices": retained, "scene_indices": retained[:max_per_scene]}


def ap_r20(quality, thresholds=(.2, .4, .6, .8), max_k=50):
    """One view's AP-R20; input quality must already be model-score sorted."""
    quality = np.asarray(quality, np.float64).reshape(-1)
    if not np.isfinite(quality).all() or max_k < 1:
        raise ValueError("finite quality and positive max_k required")
    hits = np.zeros((len(thresholds), max_k), np.float64)
    used = min(max_k, len(quality))
    hits[:, :used] = quality[:used][None, :] >= np.asarray(thresholds)[:, None]
    return float(np.mean(np.cumsum(hits, axis=1) / np.arange(1, max_k + 1)))


def evaluate_view_metrics(Q20, collision, gt_instance_id, visible_gt_instance_ids,
                          per_instance_indices, scene_indices):
    """Evaluate GT ownership, retaining all visible GT targets in Top1's denominator.

    IDs are one-based GT instance identities (background=0). NMS indices must
    be score sorted. Top1 is selected before global Top50, including proposals
    from multiple input masks assigned to the same GT target.
    """
    q, c, owner = np.asarray(Q20), np.asarray(collision, bool), np.asarray(gt_instance_id)
    if q.ndim != 1 or c.shape != q.shape or owner.shape != q.shape:
        raise ValueError("quality, collision, and GT identity must align")
    pre, final = np.asarray(per_instance_indices, np.int64), np.asarray(scene_indices, np.int64)
    if any(a.ndim != 1 or np.any((a < 0) | (a >= len(q))) for a in (pre, final)):
        raise ValueError("NMS indices outside proposals")
    visible = np.unique(np.asarray(visible_gt_instance_ids))
    visible = visible[visible > 0]
    # Background predictions remain in the ranked denominator as failures.
    quality = np.where(owner > 0, q, 0.)
    top = []
    for instance in visible:
        candidates = pre[owner[pre] == instance]
        top.append(int(candidates[0]) if len(candidates) else -1)
    chosen = np.asarray([i for i in top if i >= 0], dtype=np.int64)
    success = int(np.count_nonzero(quality[chosen] >= .4))
    return {
        "AP_R20": ap_r20(quality[final]),
        "instance_Top1_Q20_at_0_4": success / len(visible) if len(visible) else None,
        "collision_rate": float(c[chosen].mean()) if len(chosen) else None,
        "no_candidate_rate": 1 - len(chosen) / len(visible) if len(visible) else None,
        "visible_instance_count": int(len(visible)),
        "top1_candidate_count": int(len(chosen)),
        "scene_candidate_count": int(len(final)),
    }


def aggregate_scene_metrics(records):
    """Mean views within each scene, then scenes equally, ignoring undefineds.

    Pass scalar metric dictionaries with a scene_id. Counts are diagnostic
    per-view means too; explicit view_count/scene_count describe aggregation.
    """
    grouped = defaultdict(list)
    for record in records:
        grouped[record["scene_id"]].append(record)
    keys = set().union(*(set(r) for rows in grouped.values() for r in rows)) - {"scene_id", "frame_id"}
    result = {}
    for key in sorted(keys):
        means = []
        for rows in grouped.values():
            values = [r[key] for r in rows if r.get(key) is not None]
            if values:
                means.append(float(np.mean(values)))
        result[key] = float(np.mean(means)) if means else None
    result.update(scene_count=len(grouped), view_count=sum(map(len, grouped.values())))
    return result


def normal_angle_metrics(prediction, target, axis_valid):
    p, t, valid = np.asarray(prediction, float), np.asarray(target, float), np.asarray(axis_valid, bool)
    if p.shape != t.shape or p.shape != (len(valid), 3):
        raise ValueError("normal arrays and axis mask must align")
    p, t = p[valid], t[valid]
    if not len(p):
        return {"normal_median_deg": None, "normal_p90_deg": None, "normal_count": 0}
    if not np.isfinite([p, t]).all() or np.any(np.linalg.norm(p, axis=1) < 1e-9) or np.any(np.linalg.norm(t, axis=1) < 1e-9):
        raise ValueError("invalid normals inside axis_valid")
    dot = np.sum(p * t, axis=1) / (np.linalg.norm(p, axis=1) * np.linalg.norm(t, axis=1))
    angle = np.rad2deg(np.arccos(np.clip(dot, -1, 1)))
    return {"normal_median_deg": float(np.median(angle)), "normal_p90_deg": float(np.percentile(angle, 90)),
            "normal_count": len(angle)}


def masked_quality_metrics(prediction, target, quality_valid, instance_ids=None, beta=.05):
    """Fixed labeled-query MAE/SmoothL1 for one frame, optionally instance equal.

    For dense target validation pass all fixed labeled queries, never a changing
    high-scoring seed subset. Call aggregate_scene_metrics for frame/scene means.
    """
    p, t, valid = np.asarray(prediction), np.asarray(target), np.asarray(quality_valid, bool)
    if p.shape != t.shape or p.shape != valid.shape or p.ndim != 1 or beta <= 0:
        raise ValueError("aligned vectors and positive SmoothL1 beta required")
    ids = np.ones(len(p), np.int64) if instance_ids is None else np.asarray(instance_ids)
    if ids.shape != p.shape:
        raise ValueError("instance IDs must align")
    if instance_ids is not None:
        valid = valid & (ids > 0)
    error = np.abs(p[valid] - t[valid])
    if not np.isfinite(error).all():
        raise ValueError("nonfinite labeled quality")
    mae, smooth = [], []
    for instance in np.unique(ids[valid]):
        e = error[ids[valid] == instance]
        mae.append(e.mean())
        smooth.append(np.where(e < beta, .5 * e ** 2 / beta, e - .5 * beta).mean())
    return {"masked_q_mae": float(np.mean(mae)) if mae else None,
            "masked_q_smooth_l1": float(np.mean(smooth)) if smooth else None,
            "quality_valid_count": int(valid.sum()),
            "quality_coverage": float(valid.mean()) if len(valid) else None}


def proposal_coverage(seed_indices, instance_id_gt, gt_valid, gt_B, seed_valid=None):
    """Fraction of eligible GT instances with a cached positive seed (diagnostic)."""
    owners, valid, quality = np.asarray(instance_id_gt), np.asarray(gt_valid, bool), np.asarray(gt_B)
    if owners.shape != valid.shape or owners.shape != quality.shape or owners.ndim != 1:
        raise ValueError("GT diagnostic arrays must align")
    seeds = np.asarray(seed_indices, np.int64).reshape(-1)
    if seed_valid is not None:
        seeds = seeds[np.asarray(seed_valid, bool).reshape(-1)]
    if np.any((seeds < 0) | (seeds >= len(owners))):
        raise ValueError("remove padded seed indices before coverage")
    good = valid & (quality >= .4) & (owners > 0)
    eligible = np.unique(owners[good])
    covered = np.unique(owners[seeds][good[seeds]])
    return {"proposal_coverage": len(covered) / len(eligible) if len(eligible) else None,
            "proposal_eligible_instances": int(len(eligible)),
            "proposal_covered_instances": int(len(covered))}


def attempt_metrics(scores, outcomes, *, calibrated=False):
    """Distinct-score AUROC and staircase average precision on executed poses.

    Only a trained task-success head may request BCE/Brier. Source physical q
    supports ranking diagnostics only. Single-class AUROC stays undefined.
    """
    scores, y = np.asarray(scores, float), np.asarray(outcomes)
    if scores.ndim != 1 or scores.shape != y.shape or not np.isfinite(scores).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("finite executed-pose scores and binary outcomes required")
    positives, negatives = int(y.sum()), int(len(y) - y.sum())
    out = {"n_success": positives, "n_failure": negatives, "attempt_AUROC": None, "attempt_PR_AUC": None}
    if len(y):
        order = np.argsort(-scores, kind="stable")
        s, labels = scores[order], y[order]
        ends = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]
        tp = np.r_[0, np.cumsum(labels)[ends]]
        fp = np.r_[0, (ends + 1) - np.cumsum(labels)[ends]]
        if positives:
            recall = tp / positives
            precision = tp[1:] / (tp[1:] + fp[1:])
            out["attempt_PR_AUC"] = float(np.sum(np.diff(recall) * precision))
        if positives and negatives:
            out["attempt_AUROC"] = float(trapezoid(tp / positives, fp / negatives))
    if calibrated:
        if np.any((scores < 0) | (scores > 1)):
            raise ValueError("task success scores must be probabilities in [0,1]")
        p = np.clip(scores, np.finfo(float).eps, 1 - np.finfo(float).eps)
        out["attempt_BCE"] = float(np.mean(-y * np.log(p) - (1 - y) * np.log1p(-p))) if len(y) else None
        out["attempt_Brier"] = float(np.mean((scores - y) ** 2)) if len(y) else None
    return out
