"""Safely inspect and compare SuctionNet object-level seal labels.

This module never deletes negative candidates and does not infer labels for unmatched points.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


def read_label(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as file:
        raw = {key: np.asarray(file[key]) for key in file.files}
    # Both released seal_label/{id}_seal.npz and annotation/{id}_labels.npz
    # expose points/normals/scores, but may differ in optional keys.
    for required in ('points', 'normals', 'scores'):
        if required not in raw:
            raise ValueError(f'{path}: missing required key {required!r}; keys={list(raw)}')
    points = np.asarray(raw['points'], dtype=np.float64)
    normals = np.asarray(raw['normals'], dtype=np.float64)
    scores = np.asarray(raw['scores'], dtype=np.float64).reshape(-1)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f'{path}: points must be [N,3], got {points.shape}')
    if normals.shape != points.shape or len(scores) != len(points):
        raise ValueError(f'{path}: points, normals, scores have inconsistent lengths')
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(scores)):
        raise ValueError(f'{path}: non-finite points/scores')
    if np.any((scores < -1e-5) | (scores > 1 + 1e-5)):
        raise ValueError(f'{path}: seal scores outside expected [0,1]')
    output = dict(raw)
    output['points'] = points
    output['normals'] = normals
    output['scores'] = scores
    if 'indices' in raw:
        indices = np.asarray(raw['indices']).reshape(-1)
        if len(indices) != len(scores):
            raise ValueError(f'{path}: indices must have length N')
        if len(np.unique(indices)) != len(indices):
            raise ValueError(f'{path}: duplicate candidate indices')
        output['indices'] = indices.astype(np.int64)
    if 'collision' in raw:
        collision = np.asarray(raw['collision']).reshape(-1)
        if len(collision) != len(scores):
            raise ValueError(f'{path}: collision must have length N')
        output['collision'] = collision.astype(bool)
    return output


def _align(ref: dict[str, np.ndarray], new: dict[str, np.ndarray], tolerance_m: float) -> dict[str, Any]:
    """Find geometrically identical candidates, preferring stable original vertex IDs.

    Return indices only for pairs inside coordinate tolerance. Report unmatched distinctly.
    """
    if 'indices' in ref and 'indices' in new:
        lookup = {int(i): j for j, i in enumerate(new['indices'])}
        candidates = [(i, lookup[int(k)]) for i, k in enumerate(ref['indices']) if int(k) in lookup]
        if candidates:
            ref_ix, new_ix = np.asarray(candidates, dtype=np.int64).T
        else:
            ref_ix = new_ix = np.empty(0, dtype=np.int64)
        mode = 'candidate_index'
    else:
        mode = 'nearest_coordinate'
        if len(new['points']) == 0:
            ref_ix = new_ix = np.empty(0, dtype=np.int64)
        else:
            dist, neigh = cKDTree(new['points']).query(ref['points'], k=1)
            matched = np.isfinite(dist) & (dist <= tolerance_m)
            # Restrict to one-to-one assignments to avoid silently duplicating a candidate.
            matched_i = np.where(matched)[0]
            order = np.argsort(dist[matched_i], kind='stable')
            used = set()
            refs, news = [], []
            for i in matched_i[order]:
                j = int(neigh[i])
                if j not in used:
                    used.add(j); refs.append(int(i)); news.append(j)
            ref_ix = np.asarray(refs, dtype=np.int64)
            new_ix = np.asarray(news, dtype=np.int64)
    d = np.linalg.norm(ref['points'][ref_ix] - new['points'][new_ix], axis=1)
    valid = d <= tolerance_m
    return {'ref_ix': ref_ix[valid], 'new_ix': new_ix[valid], 'distance': d[valid],
            'method': mode, 'common_index_but_far': int(np.sum(~valid))}


def compare_labels(reference: str | Path, candidate: str | Path,
                   max_distance_mm: float = 0.20) -> dict[str, Any]:
    """Compare two seal files with candidate alignment, never by array row alone."""
    ref, new = read_label(reference), read_label(candidate)
    tol = max_distance_mm / 1000
    pairs = _align(ref, new, tol)
    ri, ni = pairs['ref_ix'], pairs['new_ix']
    differences = np.abs(ref['scores'][ri] - new['scores'][ni])
    out: dict[str, Any] = {
        'reference': str(reference), 'candidate': str(candidate),
        'match_method': pairs['method'],
        'reference_count': int(len(ref['points'])), 'candidate_count': int(len(new['points'])),
        'matched_count': int(len(ri)),
        'reference_match_rate': float(len(ri) / max(1, len(ref['points']))),
        'candidate_match_rate': float(len(ri) / max(1, len(new['points']))),
        'common_index_but_far': pairs['common_index_but_far'],
        'distance_max_mm': float(1000 * pairs['distance'].max()) if len(ri) else None,
        'reference_zero_ratio': float(np.mean(ref['scores'] <= 1e-8)) if len(ref['scores']) else None,
        'candidate_zero_ratio': float(np.mean(new['scores'] <= 1e-8)) if len(new['scores']) else None,
        'reference_collision_ratio': (float(ref['collision'].mean()) if 'collision' in ref else None),
        'candidate_collision_ratio': (float(new['collision'].mean()) if 'collision' in new else None),
        'score_abs_mean': float(differences.mean()) if len(differences) else None,
        'score_abs_p95': float(np.percentile(differences, 95)) if len(differences) else None,
        'score_abs_max': float(differences.max()) if len(differences) else None,
        'score_pearson': (float(np.corrcoef(ref['scores'][ri], new['scores'][ni])[0, 1])
                          if len(ri) > 1 and np.std(ref['scores'][ri]) > 0 and np.std(new['scores'][ni]) > 0 else None),
        'ref_positive_count': int(np.sum(ref['scores'] > 0)),
        'new_positive_count': int(np.sum(new['scores'] > 0)),
        'thresholds': {'matching_max_distance_mm': max_distance_mm},
    }
    if 'collision' in ref and 'collision' in new:
        out['collision_disagreement_rate'] = (float(np.mean(ref['collision'][ri] != new['collision'][ni]))
                                             if len(ri) else None)
    return out


def validate_pair(base: str | Path, adapted: str | Path, tol_mm: float = 0.20) -> dict[str, Any]:
    result = compare_labels(base, adapted, max_distance_mm=tol_mm)
    # Radius change legitimately changes scores and collision, not candidate identity.
    result['same_candidate_set'] = (
        result['matched_count'] == result['reference_count'] == result['candidate_count']
    )
    return result


def write_json(path: str | Path, value: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def label_summary(label_path: str | Path) -> dict[str, Any]:
    val = read_label(label_path)
    return {
        'path': str(label_path), 'count': len(val['points']), 'keys': sorted(val.keys()),
        'with_indices': 'indices' in val, 'with_collision': 'collision' in val,
        'zero_score_fraction': float(np.mean(val['scores'] <= 1e-8)) if len(val['scores']) else None,
        'score_p50': float(np.median(val['scores'])) if len(val['scores']) else None,
        'score_p95': float(np.percentile(val['scores'], 95)) if len(val['scores']) else None,
        'score_max': float(np.max(val['scores'])) if len(val['scores']) else None,
    }