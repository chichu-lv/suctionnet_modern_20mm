"""Deterministic CPU geometry for InstanceSuction3D.

All geometric selection uses original float64 camera coordinates. This is an
explicit correctness implementation, not an assertion about GPU latency. No
CAD geometry, target labels, or predicted quality participates in sampling.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

try:
    from . import _fps_cpu
except ImportError:
    _fps_cpu = None


def _fps_axis_order():
    """Match this NumPy build's einsum reduction order for three FP64 squares.

    Each small square rounds away when added to 1 separately. Adding both
    first instead produces the next float above 1. Row i identifies whether
    dimension i is added last. Keep the reference loop for other reductions.
    """
    probe = np.full((3, 3), 1e-8, dtype=np.float64)
    np.fill_diagonal(probe, 1.)
    reference = np.einsum('ij,ij->i', probe, probe)
    squares = probe * probe
    for order in ((0, 1, 2), (0, 2, 1), (1, 2, 0)):
        a, b, c = order
        if np.array_equal(reference, (squares[:, a] + squares[:, b]) + squares[:, c]):
            return order
    return None


_FPS_AXES = _fps_axis_order()

from suction20.instance_geometry import BOUNDARY_ROUNDOFF_M

ZONE_BOUNDS = ((0., .010), (.010, .018), (.018, .022))
ZONE_SAMPLES = (32, 32, 64)
HEIGHT_LIMIT_M = .040


def numpy_array(value, dtype=None):
    """Copy Torch tensors to CPU only at the fixed-geometry boundary."""
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def deterministic_fps(points, source_ids, count):
    """Return unique input rows, breaking every distance tie by source ID."""
    points = np.asarray(points, dtype=np.float64)
    ids = np.asarray(source_ids, dtype=np.int64)
    if points.shape != (len(ids), 3) or not np.isfinite(points).all():
        raise ValueError('FPS expects finite [M,3] points and integer source IDs')
    if len(np.unique(ids)) != len(ids):
        raise ValueError('FPS source IDs must already be unique')
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 0:
        raise ValueError('FPS count must be a nonnegative integer')
    order = np.argsort(ids, kind='stable')
    return order[fps_positions_in_source_order(points[order], count)]


def fps_positions_in_source_order(ordered, count):
    """Greedy FPS on previously validated points in ascending unique source-ID order."""
    limit = min(int(count), len(ordered))
    if _fps_cpu is not None and _FPS_AXES is not None:
        if _FPS_AXES != (0, 1, 2):
            ordered = np.ascontiguousarray(ordered[:, _FPS_AXES])
        positions = _fps_cpu.fps_positions(ordered, limit, 0)
        return np.asarray(positions, dtype=np.intp)
    result = np.empty(limit, dtype=np.int64)
    minimum = np.full(len(ordered), np.inf, dtype=np.float64)
    next_row = 0
    for step in range(limit):
        result[step] = next_row
        delta = ordered - ordered[next_row]
        minimum = np.minimum(minimum, np.einsum('ij,ij->i', delta, delta))
        # All selected rows stay negative, including duplicate XYZ positions.
        minimum[next_row] = -np.inf
        next_row = int(np.argmax(minimum)) if step + 1 < limit else 0
    return result


def voxel_representatives(points, source_ids, centroid, voxel_size=.002):
    """One immutable source point per voxel, nearest centre then source ID."""
    points = np.asarray(points, np.float64)
    ids = np.asarray(source_ids, np.int64)
    centroid = np.asarray(centroid, np.float64)
    if (points.shape != (len(ids), 3) or centroid.shape != (3,)
            or not np.isfinite(points).all() or not np.isfinite(centroid).all()
            or not np.isfinite(voxel_size) or voxel_size <= 0):
        raise ValueError('Invalid voxel representative inputs')
    if not len(points):
        return np.empty(0, np.int64), np.empty((0, 3), np.int64)
    key = np.floor((points - centroid) / voxel_size).astype(np.int64)
    centre = centroid + (key.astype(np.float64) + .5) * voxel_size
    distance2 = np.sum((points - centre) ** 2, axis=1)
    order = np.lexsort((ids, distance2, key[:, 2], key[:, 1], key[:, 0]))
    ordered_keys = key[order]
    keep = np.r_[True, np.any(ordered_keys[1:] != ordered_keys[:-1], axis=1)]
    return order[keep], key[order[keep]]


@dataclass
class NeighborIndex:
    points: np.ndarray
    source_ids: np.ndarray
    components: np.ndarray

    def __post_init__(self):
        self.points = np.asarray(self.points, np.float64)
        self.source_ids = np.asarray(self.source_ids, np.int64)
        self.components = np.asarray(self.components, np.int64)
        if (self.points.shape != (len(self.source_ids), 3)
                or self.components.shape != self.source_ids.shape
                or not np.isfinite(self.points).all()):
            raise ValueError('Invalid component neighbor index')
        self.groups = {}
        self._tied_source_ids = set()
        for comp in np.unique(self.components):
            rows = np.flatnonzero(self.components == comp)
            rows = rows[np.argsort(self.source_ids[rows], kind='stable')]
            ids = self.source_ids[rows]
            if np.any(ids[1:] == ids[:-1]):
                self._tied_source_ids.add(int(comp))
            self.groups[int(comp)] = (rows, cKDTree(self.points[rows]))
        self.all_tree = cKDTree(self.points) if len(self.points) else None

    def nearest_anchor(self, points, radius=.005):
        """Nearest raw point, resolving exact ties without modifying the pose."""
        points = np.asarray(points, np.float64)
        anchors = np.full(len(points), -1, np.int64)
        distance = np.full(len(points), np.inf, np.float64)
        if self.all_tree is None or not len(points):
            return anchors, distance
        d, rows = self.all_tree.query(points, k=min(2, len(self.points)))
        if len(self.points) == 1:
            d, rows = d[:, None], rows[:, None]
        supported = d[:, 0] <= radius + BOUNDARY_ROUNDOFF_M
        active = np.flatnonzero(supported)
        anchors[active] = rows[active, 0]
        # Recompute with the same NumPy reduction as the scalar reference;
        # cKDTree's own distance arithmetic is not the output contract.
        distance[active] = np.linalg.norm(
            self.points[rows[active, 0]] - points[active], axis=1)
        ties = (supported & (d[:, 0] == d[:, 1]) if d.shape[1] > 1
                else np.zeros(len(points), bool))
        for i in np.flatnonzero(ties):
            choices = np.asarray(self.all_tree.query_ball_point(
                points[i], np.nextafter(d[i, 0], np.inf)), np.int64)
            actual = np.linalg.norm(self.points[choices] - points[i], axis=1)
            winner = np.lexsort((self.source_ids[choices], actual))[0]
            anchors[i] = int(choices[winner])
            distance[i] = float(actual[winner])
        return anchors, distance

    def associate(self, points, components, radius=.010):
        """At most 3 same-component neighbors, with exact-point precedence.

        d1/d3 describe the untruncated same-component nearest distances. Missing
        neighbors remain NaN; the valid interpolation may contain 1, 2 or 3.
        """
        points = np.asarray(points, np.float64)
        components = np.asarray(components, np.int64)
        if points.shape != (len(components), 3) or not np.isfinite(points).all():
            raise ValueError('Association queries must be finite [Q,3]')
        indices = np.full((len(points), 3), -1, np.int64)
        weights = np.zeros((len(points), 3), np.float64)
        d1 = np.full(len(points), np.nan, np.float64)
        d3 = np.full(len(points), np.nan, np.float64)
        counts = np.zeros(len(points), np.int64)
        component_missing = np.ones(len(points), bool)
        for component in np.unique(components):
            if int(component) not in self.groups:
                continue
            query_rows = np.flatnonzero(components == component)
            source_rows, tree = self.groups[int(component)]
            component_missing[query_rows] = False
            # One extra neighbor detects an unresolved tie across the 3-cut.
            k = min(4, len(source_rows))
            distances, nearest = tree.query(points[query_rows], k=k)
            if k == 1:
                distances, nearest = distances[:, None], nearest[:, None]
            selected = source_rows[nearest]
            dist = np.linalg.norm(self.points[selected] - points[query_rows, None], axis=2)
            # Sort the short axis in one vectorized operation. This handles
            # ties within the returned neighbors and any last-bit difference
            # between tree and NumPy distances without a Python point loop.
            order = np.lexsort((self.source_ids[selected], dist), axis=1)[:, :3]
            selected = np.take_along_axis(selected, order, axis=1)
            dist = np.take_along_axis(dist, order, axis=1)
            width = selected.shape[1]
            d1[query_rows] = dist[:, 0]
            if width == 3:
                d3[query_rows] = dist[:, 2]
            keep = dist <= radius + BOUNDARY_ROUNDOFF_M
            counts[query_rows] = keep.sum(axis=1)
            indices[query_rows, :width] = np.where(keep, selected, -1)
            reciprocal = np.zeros_like(dist)
            np.divide(1., dist, out=reciprocal, where=keep & (dist != 0))
            denominator = reciprocal.sum(axis=1, keepdims=True)
            normalized = np.zeros_like(reciprocal)
            np.divide(reciprocal, denominator, out=normalized, where=denominator != 0)
            weights[query_rows, :width] = normalized
            exact = (counts[query_rows] > 0) & (dist[:, 0] == 0)
            weights[query_rows[exact]] = 0.
            weights[query_rows[exact], 0] = 1.

            # Only a tie across the three-neighbor cut can hide additional
            # contenders. Re-run the original ball/lexsort path for those rows.
            ties = (np.flatnonzero(distances[:, 2] == distances[:, 3])
                    if k == 4 else np.empty(0, np.int64))
            for local_row in ties:
                query_row = query_rows[local_row]
                selected = source_rows[np.asarray(tree.query_ball_point(
                    points[query_row], np.nextafter(distances[local_row, 2], np.inf)), np.int64)]
                dist = np.linalg.norm(self.points[selected] - points[query_row], axis=1)
                order = np.lexsort((self.source_ids[selected], dist))[:3]
                selected, dist = selected[order], dist[order]
                d1[query_row] = dist[0]
                d3[query_row] = np.nan
                if len(dist) >= 3:
                    d3[query_row] = dist[2]
                keep = dist <= radius + BOUNDARY_ROUNDOFF_M
                selected, dist = selected[keep], dist[keep]
                count = len(selected)
                counts[query_row] = count
                indices[query_row] = -1
                weights[query_row] = 0.
                indices[query_row, :count] = selected
                if not count:
                    continue
                if dist[0] == 0:
                    weights[query_row, 0] = 1.
                else:
                    reciprocal = 1. / dist
                    weights[query_row, :count] = reciprocal / reciprocal.sum()
        return {'indices': indices, 'weights': weights, 'valid': counts > 0,
                'counts': counts, 'd1': d1, 'd3': d3,
                'component_missing': component_missing}

    def support_ball(self, point, component):
        if int(component) not in self.groups:
            return np.empty(0, np.int64)
        rows, tree = self.groups[int(component)]
        bound = math.hypot(.022, HEIGHT_LIMIT_M) + 2 * BOUNDARY_ROUNDOFF_M
        tied = getattr(self, '_tied_source_ids', None)
        if tied is not None and int(component) not in tied:
            # For a compact support cloud, a C distance scan avoids boxing and
            # sorting thousands of Python indices for every contact query.
            if len(rows) <= 32768:
                distance2 = cdist(np.asarray(point)[None, :], tree.data, metric='sqeuclidean')[0]
                radius2 = bound * bound
                tolerance = 16 * np.finfo(np.float64).eps * radius2
                # Preserve the tree's rounding rule on the decision boundary.
                if not np.any(np.abs(distance2 - radius2) <= tolerance):
                    return rows[distance2 < radius2]
            # Tree row positions were assigned in source-ID order at construction.
            return rows[np.asarray(tree.query_ball_point(point, bound, return_sorted=True), np.intp)]
        selected = rows[np.asarray(tree.query_ball_point(point, bound), np.intp)]
        return selected[np.argsort(self.source_ids[selected], kind='stable')]


def distance_summary(values):
    finite = np.asarray(values, np.float64)
    finite = finite[np.isfinite(finite)]
    return {'count': int(len(finite)), 'p50': float(np.median(finite)) if len(finite) else None,
            'p95': float(np.percentile(finite, 95)) if len(finite) else None,
            'max': float(finite.max()) if len(finite) else None}


def association_diagnostics(association):
    count = len(association['valid'])
    mean = lambda a: float(np.mean(a)) if count else 0.
    nearest = distance_summary(association['d1'])
    return {'d1_p50_p95_max': nearest,
            'd3_p50_p95_max': distance_summary(association['d3']),
            'nearest_coverage_radius_max': nearest['max'],
            'within_10mm_feature_valid_fraction': mean(association['valid']),
            'same_component_no_backbone_fraction': mean(association['component_missing']),
            'same_component_fewer_than_three_neighbors_fraction': mean(association['counts'] < 3)}


def contact_coordinates(points, point, normal):
    delta = np.asarray(points, np.float64) - point
    height = delta @ normal
    tangent = delta - height[:, None] * normal
    rho = np.linalg.norm(tangent, axis=1)
    membership_rho = rho.copy()
    for boundary in (0., .010, .018, .022):
        membership_rho[np.abs(membership_rho - boundary) <= BOUNDARY_ROUNDOFF_M] = boundary
    height_valid = np.abs(height) <= HEIGHT_LIMIT_M + BOUNDARY_ROUNDOFF_M
    masks = [(membership_rho >= low) & ((membership_rho <= high) if zone == 2 else (membership_rho < high)) & height_valid
             for zone, (low, high) in enumerate(ZONE_BOUNDS)]
    return rho, height, tangent, masks


def lip_summary(height, tangent, normal, *, basis=None):
    """Eight raw-lip moments before FPS; invariant to tangent-basis rotation."""
    height = np.asarray(height, np.float64)
    tangent = np.asarray(tangent, np.float64)
    result = np.zeros(8, np.float64)
    if not len(height):
        return result, 0
    if basis is None:
        ref = np.eye(3)[int(np.argmin(np.abs(normal)))]
        first = np.cross(ref, normal)
        first /= np.linalg.norm(first)
        second = np.cross(normal, first)
    else:
        first, second = np.asarray(basis, np.float64)[:, :2].T
    angles = np.mod(np.arctan2(tangent @ second, tangent @ first), 2 * np.pi)
    ordered = np.sort(angles)
    gaps = np.diff(np.r_[ordered, ordered[0] + 2 * np.pi])
    angular_support = max(1, int(np.count_nonzero(gaps > 1e-6)))
    valid = angular_support >= 7
    centred = height - height.mean()
    rms = np.sqrt(np.mean(centred ** 2))
    amplitude = np.zeros(3, np.float64)
    if valid:
        argument = angles[:, None] * np.arange(1, 4)[None]
        amplitude = np.hypot(np.mean(centred[:, None] * np.cos(argument), axis=0),
                             np.mean(centred[:, None] * np.sin(argument), axis=0))
    result[:] = [np.log1p(len(height)), gaps.max() / (2 * np.pi), 1.,
                 rms / HEIGHT_LIMIT_M, *(amplitude / HEIGHT_LIMIT_M), float(valid)]
    return result, angular_support
