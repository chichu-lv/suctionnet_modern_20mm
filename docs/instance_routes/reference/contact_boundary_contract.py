#!/usr/bin/env python3
"""NumPy v3 contact-boundary reference; metres throughout.

This module verifies descriptor algebra, support gates and missing-data
reductions. It is not a model, a depth sampler, a geometry teacher, an export
implementation or a latency benchmark. Caller-provided A support must already
belong to the query's instance/component. B observed_valid must already encode
image bounds, target-instance membership and usable observed depth.

The v2 quality definition remains gt_INST20_pose_v2. The earlier reference
modules are imported for their unchanged camera chart and ideal-ring geometry.
"""
from __future__ import annotations

import argparse
import json
import platform

import numpy as np

from geometry_contract import project
from revision_contract import QUALITY_CONTRACT, RADII_M, camera_chart_frame, projected_rings


ANGLES = 72
HEIGHT_SCALE_M = .040
GATE_TOLERANCE_M = .003
ANGLE_MERGE_TOLERANCE_RAD = 1e-6
# Roundoff guard only: snap exact metric boundaries before half-open membership.
# This is one billionth of a millimetre, not a sensor-error allowance.
BOUNDARY_ROUNDOFF_M = 1e-12
LIP_SUMMARY_FIELDS = (
    "log1p_count", "maximum_angle_gap_fraction", "lip_nonempty",
    "height_rms_scaled", "a1_scaled", "a2_scaled", "a3_scaled", "harmonic_valid",
)


def _point(value, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite 3-vector")
    return result


def _frame(normal_out, frame=None):
    n = _point(normal_out, "normal_out")
    length = np.linalg.norm(n)
    if length <= 1e-12:
        raise ValueError("normal_out must be nonzero")
    n = n / length
    if frame is None:
        basis, _ = camera_chart_frame(n)
    else:
        basis = np.asarray(frame, dtype=np.float64)
        if (basis.shape != (3, 3) or not np.isfinite(basis).all()
                or not np.allclose(basis.T @ basis, np.eye(3), atol=1e-10, rtol=0)
                or not np.allclose(basis[:, 2], n, atol=1e-10, rtol=0)
                or not np.isclose(np.linalg.det(basis), 1., atol=1e-10, rtol=0)):
            raise ValueError("frame must be a right-handed orthonormal [e1,e2,n]")
    return n, basis


def _bool_mask(value, shape, name):
    mask = np.asarray(value)
    if mask.shape != shape or mask.dtype != np.bool_:
        raise ValueError(f"{name} must be bool with shape {shape}")
    return mask


def _snap_boundaries(value, boundaries):
    result = np.array(value, dtype=np.float64, copy=True)
    for edge in boundaries:
        result[np.abs(result - edge) <= BOUNDARY_ROUNDOFF_M] = edge
    return result


def lip_harmonic_summary(points_m, source_ids, point_m, normal_out, *,
                         support_valid=None, frame=None):
    """Return A's fixed eight-dimensional raw-lip summary and diagnostics.

    Use full raw support before region FPS. A source ID denotes one immutable
    observed point; duplicates count once, and conflicting duplicate positions
    raise. Nonfinite/unsupported points are removed before deduplication. Only
    actual 18 <= rho <= 22 mm and |h| <= 40 mm support enters the summary.

    Heights are centred with equal point weights. Harmonic amplitudes k=1,2,3
    are invariant to a rotation of the supplied tangent basis, but do not form
    a complete shape invariant. Uneven density and holes still affect these
    observed-sample moments. At least seven independent angular supports are
    required to return amplitudes; this is not a coverage/conditioning test.

    Empty summary is all zero. Nonempty support always retains centred RMS,
    even when harmonic_valid is false. No learnt parameters are present here.
    """
    points = np.asarray(points_m, dtype=np.float64)
    ids = np.asarray(source_ids)
    if points.ndim != 2 or points.shape[1] != 3 or ids.shape != (len(points),):
        raise ValueError("Expected [M,3] points and [M] source IDs")
    if ids.size and ids.dtype.kind not in "iu":
        raise ValueError("source IDs must be integers")
    ids = ids.astype(np.int64, copy=False)
    p = _point(point_m, "point_m")
    n, basis = _frame(normal_out, frame)
    keep = (np.ones(len(points), dtype=bool) if support_valid is None
            else _bool_mask(support_valid, (len(points),), "support_valid"))
    keep = keep & np.isfinite(points).all(axis=1)
    points, ids = points[keep], ids[keep]
    if np.any(ids < 0):
        raise ValueError("Valid support must have nonnegative source IDs")
    if len(ids):
        order = np.argsort(ids, kind="stable")
        points, ids = points[order], ids[order]
        duplicate = ids[1:] == ids[:-1]
        if np.any(np.max(np.abs(points[1:] - points[:-1]), axis=1)[duplicate]
                  > BOUNDARY_ROUNDOFF_M):
            raise ValueError("One source ID cannot identify different observed points")
        first = np.r_[True, ids[1:] != ids[:-1]]
        points, ids = points[first], ids[first]
    delta = points - p
    height = delta @ n
    tangent = delta - height[:, None] * n
    rho = np.linalg.norm(tangent, axis=1)
    rho_region = _snap_boundaries(rho, (.018, .022))
    in_lip = ((rho_region >= .018) & (rho_region <= .022)
              & (np.abs(height) <= HEIGHT_SCALE_M + BOUNDARY_ROUNDOFF_M))
    ids, height, tangent = ids[in_lip], height[in_lip], tangent[in_lip]
    count = len(ids)
    summary = np.zeros(8, dtype=np.float64)
    amplitudes = np.zeros(3, dtype=np.float64)
    rms, support_count, harmonic_valid = 0., 0, False
    if count:
        theta = np.mod(np.arctan2(tangent @ basis[:, 1], tangent @ basis[:, 0]),
                       2 * np.pi)
        ordered = np.sort(theta)
        gaps = np.diff(np.r_[ordered, ordered[0] + 2 * np.pi])
        support_count = max(1, int(np.count_nonzero(gaps > ANGLE_MERGE_TOLERANCE_RAD)))
        harmonic_valid = support_count >= 7
        centred = height - height.mean()
        rms = float(np.sqrt(np.mean(centred * centred)))
        if harmonic_valid:
            angles = theta[:, None] * np.arange(1, 4)[None]
            real = np.mean(centred[:, None] * np.cos(angles), axis=0)
            imaginary = np.mean(centred[:, None] * np.sin(angles), axis=0)
            amplitudes = np.hypot(real, imaginary)
        summary[:] = [np.log1p(count), gaps.max() / (2 * np.pi), 1.,
                      rms / HEIGHT_SCALE_M, *(amplitudes / HEIGHT_SCALE_M),
                      float(harmonic_valid)]
    return {"summary": summary, "fields": LIP_SUMMARY_FIELDS,
            "source_ids": ids, "support_count": count,
            "independent_angular_support": support_count,
            "height_rms_m": rms, "amplitudes_m": amplitudes,
            "harmonic_valid": harmonic_valid}


def projected_ring_gate(observed_points_m, observed_valid, point_m, normal_out, *, phase=0.):
    """Gate B's [9,72] observed slots while retaining raw observed heights.

    Uses the exact chart and phase of revision_contract.projected_rings. This
    helper does not sample a depth image. observed_valid=False may contain NaN;
    observed_valid=True with a nonfinite position is a malformed input.

    Radius zero must have one observation/validity broadcast across 72 slots.
    Its support is evaluated once, without angle/tangent/positive-radial gates.
    Other slots require radial error <=3mm, membership in their nominal zone,
    tangential offset <=3mm and strictly positive signed radial projection.

    Regions are core [0,10), inner [10,18), lip [18,22] mm. Snap rho within
    1e-12 m of a region edge before this half-open membership test, and use the
    same roundoff allowance for inclusive 3mm gates. Raw h is never clipped or
    discarded merely because contact support failed.
    """
    points = np.asarray(observed_points_m, dtype=np.float64)
    if points.shape != (len(RADII_M), ANGLES, 3):
        raise ValueError("observed_points_m must have shape [9,72,3]")
    observed = _bool_mask(observed_valid, (len(RADII_M), ANGLES), "observed_valid")
    if not np.isfinite(points[observed]).all():
        raise ValueError("Observed-valid positions must be finite")
    if not np.all(observed[0] == observed[0, 0]):
        raise ValueError("The centre validity must be read once and broadcast")
    if observed[0, 0] and not np.allclose(points[0], points[0, 0],
                                         atol=BOUNDARY_ROUNDOFF_M, rtol=0):
        raise ValueError("The centre position must be read once and broadcast")
    if not np.isfinite(phase):
        raise ValueError("phase must be finite")
    p = _point(point_m, "point_m")
    n, basis = _frame(normal_out)
    theta = 2 * np.pi * (np.arange(ANGLES) + .5) / ANGLES + phase
    radial = np.cos(theta)[:, None] * basis[:, 0] + np.sin(theta)[:, None] * basis[:, 1]
    angular = -np.sin(theta)[:, None] * basis[:, 0] + np.cos(theta)[:, None] * basis[:, 1]
    # Clear invalid points before any subtract/multiply, so NaN padding cannot
    # leak into raw summaries through expressions such as NaN * False.
    delta = np.where(observed[..., None], points, p) - p
    height = delta @ n
    tangent = delta - height[..., None] * n
    rho = np.linalg.norm(tangent, axis=-1)
    tangential_offset = np.sum(tangent * angular[None], axis=-1)
    signed_radial = np.sum(tangent * radial[None], axis=-1)
    rho_region = _snap_boundaries(rho, (0., .010, .018, .022))
    in_region = np.zeros_like(observed)
    in_region[:3] = (rho_region[:3] >= 0) & (rho_region[:3] < .010)
    in_region[3:6] = (rho_region[3:6] >= .010) & (rho_region[3:6] < .018)
    in_region[6:] = (rho_region[6:] >= .018) & (rho_region[6:] <= .022)
    in_region &= observed
    radial_ok = np.abs(rho - RADII_M[:, None]) <= GATE_TOLERANCE_M + BOUNDARY_ROUNDOFF_M
    tangential_ok = np.abs(tangential_offset) <= GATE_TOLERANCE_M + BOUNDARY_ROUNDOFF_M
    positive_radial = signed_radial > 0
    contact = observed & in_region & radial_ok & tangential_ok & positive_radial
    centre_ok = bool(observed[0, 0] and in_region[0, 0]
                     and rho[0, 0] <= GATE_TOLERANCE_M + BOUNDARY_ROUNDOFF_M)
    contact[0] = centre_ok
    return {"observed_valid": observed.copy(), "contact_valid": contact,
            "height_m": height, "rho_m": rho,
            "radial_error_m": rho - RADII_M[:, None],
            "tangential_offset_m": tangential_offset, "signed_radial_m": signed_radial,
            "in_region": in_region, "radial_ok": radial_ok,
            "tangential_ok": tangential_ok, "positive_radial": positive_radial}


def maximum_circular_invalid_gap(valid):
    """Longest consecutive invalid run / slot count; all empty=1, all valid=0."""
    mask = np.asarray(valid)
    if mask.ndim != 1 or not mask.size or mask.dtype != np.bool_:
        raise ValueError("Expected a nonempty one-dimensional bool mask")
    if not mask.any():
        return 1.
    longest, run = 0, 0
    for missing in np.tile(~mask, 2):
        run = run + 1 if missing else 0
        longest = max(longest, run)
    return float(min(longest, len(mask)) / len(mask))


def ring_support_summary(contact_valid, observed_valid, height_m):
    """Return B's 30d support/raw summary, ring-major then zone-major.

    First 18: [contact_fraction, maximum_invalid_gap] for each of 9 radii.
    Last 12: [observed_fraction, rejection_among_observed, max_positive_h/.04,
    max_negative_h/.04] for core, inner, lip. A nominal zone has 216 logical
    slots; the broadcast centre contributes one radius weight, not 72 unique
    independent observations. Empty observed sets give four zero raw values.

    Raw evidence stays in its nominal template zone even when observed rho is
    outside that zone. These diagnostics are not target-self collision labels.
    """
    shape = (len(RADII_M), ANGLES)
    contact = _bool_mask(contact_valid, shape, "contact_valid")
    observed = _bool_mask(observed_valid, shape, "observed_valid")
    height = np.asarray(height_m, dtype=np.float64)
    if height.shape != shape or not np.isfinite(height[observed]).all():
        raise ValueError("height_m must be [9,72] and finite wherever observed")
    if np.any(contact & ~observed):
        raise ValueError("Contact support must be a subset of observed support")
    if not np.all(observed[0] == observed[0, 0]) or not np.all(contact[0] == contact[0, 0]):
        raise ValueError("Centre observed/contact validity must be broadcast")
    if observed[0, 0] and not np.allclose(height[0], height[0, 0],
                                         atol=BOUNDARY_ROUNDOFF_M, rtol=0):
        raise ValueError("Centre observed height must be broadcast")
    coverage = np.array([[row.mean(), maximum_circular_invalid_gap(row)] for row in contact])
    raw = np.zeros((3, 4), dtype=np.float64)
    for zone in range(3):
        rows = slice(3 * zone, 3 * zone + 3)
        count = int(observed[rows].sum())
        if count:
            selected = height[rows][observed[rows]]
            raw[zone] = [count / (3 * ANGLES),
                         np.count_nonzero(observed[rows] & ~contact[rows]) / count,
                         max(0., float(selected.max())) / HEIGHT_SCALE_M,
                         max(0., float(-selected.min())) / HEIGHT_SCALE_M]
    return {"summary": np.r_[coverage.ravel(), raw.ravel()],
            "coverage_by_radius": coverage, "raw_by_zone": raw}


def per_angle_layer_norm(tokens, valid, *, eps=1e-5, weight=None, bias=None):
    """Numerical reference: normalize channels only, then zero invalid angles.

    Shape [...,angles,channels]. No reduction spans angles or queries. This
    illustrates the normalization contract, not learned Conv1d execution.
    Invalid slots are cleared before arithmetic; affine bias is masked again.
    """
    values = np.asarray(tokens, dtype=np.float64)
    if values.ndim < 2 or values.shape[-1] == 0 or values.shape[-2] == 0:
        raise ValueError("Expected nonempty [...,angles,channels] tokens")
    mask = _bool_mask(valid, values.shape[:-1], "valid")
    if not np.isfinite(values[mask]).all() or not np.isfinite(eps) or eps <= 0:
        raise ValueError("Valid tokens must be finite and eps positive")
    channels = values.shape[-1]
    gamma = np.ones(channels) if weight is None else np.asarray(weight, dtype=np.float64)
    beta = np.zeros(channels) if bias is None else np.asarray(bias, dtype=np.float64)
    if (gamma.shape != (channels,) or beta.shape != (channels,)
            or not np.isfinite(gamma).all() or not np.isfinite(beta).all()):
        raise ValueError("Affine parameters must be finite channel vectors")
    values = np.where(mask[..., None], values, 0.)
    centred = values - values.mean(axis=-1, keepdims=True)
    variance = np.mean(centred * centred, axis=-1, keepdims=True)
    result = centred / np.sqrt(variance + eps) * gamma + beta
    return np.where(mask[..., None], result, 0.)


def masked_mean_max(tokens, valid):
    """Pool over the angle axis (-2); empty mean/max are zero.

    Always pair this reduction with explicit coverage/gap descriptors. Its
    outputs alone do not distinguish a half-observed constant ring from a
    fully observed constant ring. NaN or arbitrary invalid tokens are ignored.
    """
    values = np.asarray(tokens, dtype=np.float64)
    if values.ndim < 2 or values.shape[-2] == 0 or values.shape[-1] == 0:
        raise ValueError("Expected nonempty [...,angles,channels] tokens")
    mask = _bool_mask(valid, values.shape[:-1], "valid")
    if not np.isfinite(values[mask]).all():
        raise ValueError("Valid tokens must be finite")
    count = mask.sum(axis=-1)[..., None]
    total = np.where(mask[..., None], values, 0.).sum(axis=-2)
    mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    maximum = np.max(np.where(mask[..., None], values, -np.inf), axis=-2)
    return mean, np.where(count > 0, maximum, 0.)


def selftest():
    checks = []

    def check(name, condition):
        if not bool(condition):
            raise AssertionError(name)
        checks.append(name)

    def reject(name, operation):
        try:
            operation()
        except ValueError:
            checks.append(name)
        else:
            raise AssertionError(name)

    p = np.array([.01, -.02, .6])
    n, frame = _frame([.2, .3, -.9])
    theta = 2 * np.pi * (np.arange(ANGLES) + .5) / ANGLES
    ids = np.arange(ANGLES, dtype=np.int64)

    def lip_points(angles, heights, radius=.020):
        angles, heights = np.asarray(angles), np.asarray(heights)
        tangent = np.cos(angles)[:, None] * frame[:, 0] + np.sin(angles)[:, None] * frame[:, 1]
        return p + radius * tangent + heights[:, None] * n

    def summary(angles, heights, **kwargs):
        return lip_harmonic_summary(lip_points(angles, heights), np.arange(len(angles)), p, n, **kwargs)

    height = .003 * np.cos(theta) + .002 * np.sin(2 * theta) + .001 * np.cos(3 * theta)
    points = lip_points(theta, height)
    base = lip_harmonic_summary(points, ids, p, n)
    check("a_summary_has_fixed_eight_dimensions", base["summary"].shape == (8,))
    check("a_full_ring_has_72_independent_supports", base["independent_angular_support"] == 72)
    errors = []
    for angle in (.731, np.pi / 2, np.pi + .127, -1.376):
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0.],
                             [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
        rotated = lip_harmonic_summary(points, ids, p, n, frame=frame @ rotation)
        errors.append(float(np.max(np.abs(rotated["summary"] - base["summary"]))))
    check("a_arbitrary_tangent_basis_rotation_preserves_summary", max(errors) < 1e-12)
    irregular = np.array([.01, .04, .11, .27, .48, .51, .63, 1.07, 2.0])
    irregular_height = .003 * np.sin(2 * irregular) + .001
    uneven = summary(irregular, irregular_height)
    rotation = np.array([[np.cos(.81), -np.sin(.81), 0.],
                         [np.sin(.81), np.cos(.81), 0.], [0., 0., 1.]])
    uneven_rotated = summary(irregular, irregular_height, frame=frame @ rotation)
    check("a_gauge_invariance_also_holds_with_uneven_partial_support",
          np.allclose(uneven["summary"], uneven_rotated["summary"], atol=1e-12, rtol=0))
    duplicated = lip_harmonic_summary(np.repeat(points, 3, axis=0), np.repeat(ids, 3), p, n)
    check("a_duplicate_source_ids_do_not_increase_weight_or_count",
          np.array_equal(duplicated["source_ids"], ids)
          and np.allclose(duplicated["summary"], base["summary"], atol=1e-12, rtol=0))
    reject("a_conflicting_duplicate_source_position_is_rejected",
           lambda: lip_harmonic_summary(points[:2], [0, 0], p, n))
    extra = np.vstack([points, p + .017 * frame[:, 0], p + .023 * frame[:, 0],
                       p + .020 * frame[:, 0] + .041 * n, [np.nan] * 3,
                       p + .020 * frame[:, 0]])
    support_valid = np.ones(len(extra), dtype=bool)
    support_valid[-1] = False
    filtered = lip_harmonic_summary(extra, np.arange(len(extra)), p, n, support_valid=support_valid)
    check("a_raw_lip_filter_excludes_wrong_radius_height_nonfinite_and_unsupported",
          np.array_equal(filtered["source_ids"], ids)
          and np.allclose(filtered["summary"], base["summary"], atol=1e-12, rtol=0))
    partial = np.linspace(.1, .7, 12)
    constant = summary(partial, np.full(len(partial), .007))
    check("a_constant_height_partial_arc_has_zero_centred_rms_and_harmonics",
          np.allclose(constant["summary"][3:7], 0., atol=1e-12, rtol=0))
    check("a_validity_is_not_a_full_coverage_guarantee",
          constant["harmonic_valid"] and constant["summary"][1] > .9)
    few = summary(theta[:6], np.linspace(-.003, .003, 6))
    check("a_fewer_than_seven_angles_zero_amplitudes_but_retain_rms",
          not few["harmonic_valid"] and np.array_equal(few["amplitudes_m"], np.zeros(3))
          and few["height_rms_m"] > 0)
    check("a_seven_distinct_angles_meet_minimum_sampling_condition",
          summary(theta[:7], height[:7])["harmonic_valid"])
    single_ray = summary(np.zeros(12), np.linspace(-.003, .003, 12))
    check("a_many_points_on_one_angle_do_not_fake_seven_supports",
          single_ray["independent_angular_support"] == 1 and not single_ray["harmonic_valid"])
    seam = np.r_[-.2e-6, .2e-6, np.arange(1, 7) * .7]
    seam_summary = summary(seam, .003 * np.sin(seam))
    check("a_angular_merge_tolerance_wraps_across_zero",
          seam_summary["independent_angular_support"] == 7)
    empty = lip_harmonic_summary(np.empty((0, 3)), np.empty(0, dtype=int), p, n)
    check("a_empty_support_summary_is_all_zero", np.array_equal(empty["summary"], np.zeros(8)))
    single = summary(theta[:1], height[:1])
    check("a_single_support_has_full_gap_and_zero_rms",
          np.allclose(single["summary"], [np.log(2), 1, 1, 0, 0, 0, 0, 0], atol=1e-12, rtol=0))
    offset = summary(theta, height + .01)
    check("a_centred_descriptor_ignores_common_height_offset",
          np.allclose(offset["summary"], base["summary"], atol=1e-12, rtol=0))
    perm = np.random.default_rng(123).permutation(ANGLES)
    permuted = lip_harmonic_summary(points[perm], ids[perm], p, n)
    check("a_source_order_does_not_change_summary",
          np.array_equal(permuted["summary"], base["summary"]))
    patterns = [np.r_[np.ones(36), -np.ones(36)],
                np.tile(np.r_[np.ones(18), -np.ones(18)], 2),
                np.tile([1., -1.], 36)]
    examples = [summary(theta, .003 * values) for values in patterns]
    examples_mm = np.array([value["amplitudes_m"] * 1000 for value in examples])
    check("a_three_examples_have_identical_height_multisets_and_rms",
          all(np.array_equal(np.sort(values), np.sort(patterns[0])) for values in patterns)
          and all(np.isclose(value["height_rms_m"], .003, atol=1e-12) for value in examples))
    check("a_harmonics_distinguish_the_three_specific_equal_multisets",
          np.allclose(examples_mm, [[1.91046547, 0, .638441465],
                                    [0, 1.91228554, 0], [0, 0, 0]], atol=1e-8, rtol=0))
    k4 = summary(theta, .003 * np.cos(4 * theta))
    check("a_k4_energy_is_missing_from_first_three_amplitudes_but_present_in_rms",
          np.allclose(k4["amplitudes_m"], 0., atol=1e-12, rtol=0)
          and np.isclose(k4["height_rms_m"], .003 / np.sqrt(2), atol=1e-12))
    phase0 = .003 * np.cos(theta) + .0015 * np.cos(2 * theta)
    phase1 = .003 * np.cos(theta) + .0015 * np.cos(2 * theta + np.pi / 2)
    s0, s1 = summary(theta, phase0), summary(theta, phase1)
    # Their nonzero first harmonics have the same phase, fixing any common
    # rotation to zero; the different second-harmonic phase then cannot agree.
    first0, first1 = np.mean(phase0 * np.exp(1j * theta)), np.mean(phase1 * np.exp(1j * theta))
    second0, second1 = np.mean(phase0 * np.exp(2j * theta)), np.mean(phase1 * np.exp(2j * theta))
    check("a_same_amplitudes_and_rms_do_not_recover_relative_harmonic_phase",
          np.allclose(s0["summary"], s1["summary"], atol=1e-12, rtol=0)
          and np.isclose(first0, first1) and not np.isclose(second0, second1))

    K = np.array([[600., 0, 320], [0, 600., 240], [0, 0, 1.]])
    centre = np.array([0., 0., .6])
    axis = np.array([0., 0., -1.])
    phase = -np.pi / ANGLES  # Include exact theta=0,pi for analytic examples.
    rings = projected_rings(centre, axis, K, phase=phase)
    observed = np.ones((9, ANGLES), dtype=bool)
    gate = projected_ring_gate(rings["points"], observed, centre, axis, phase=phase)
    check("b_ideal_ring_interface_and_all_metric_boundaries_are_supported",
          gate["contact_valid"].all() and np.allclose(gate["rho_m"], RADII_M[:, None], atol=1e-12))
    support = ring_support_summary(gate["contact_valid"], observed, gate["height_m"])
    check("b_summary_has_30_dimensions_with_full_coverage_and_zero_raw_heights",
          support["summary"].shape == (30,)
          and np.array_equal(support["coverage_by_radius"], np.tile([1., 0.], (9, 1)))
          and np.array_equal(support["raw_by_zone"], np.tile([1., 0., 0., 0.], (3, 1))))

    tilted = np.array([1., 0., -1.]) / np.sqrt(2)
    tilted_rings = projected_rings(centre, tilted, K, phase=phase)
    nominal = tilted_rings["points"]
    # Camera ray lambda*x intersects the parallel observed plane n.(x-p)=h.
    raw_height = -.006
    scale = (tilted @ centre + raw_height) / (nominal @ tilted)
    actual = scale[..., None] * nominal
    tilted_gate = projected_ring_gate(actual, observed, centre, tilted, phase=phase)
    parallax_mm = tilted_gate["rho_m"][6, [0, 36]] * 1000
    actual_uv, _ = project(actual.reshape(-1, 3), K)
    check("b_parallax_example_points_lie_on_the_same_camera_rays",
          np.allclose(actual_uv, tilted_rings["uv"].reshape(-1, 2), atol=1e-10, rtol=0)
          and np.allclose(tilted_gate["height_m"], raw_height, atol=1e-12, rtol=0))
    check("b_45deg_minus6mm_moves_nominal18mm_to_24p254558_or_12p254558mm",
          np.allclose(parallax_mm, [24.254558441227, 12.254558441227], atol=1e-9, rtol=0))
    check("b_both_parallax_cross_zone_samples_are_rejected",
          not tilted_gate["contact_valid"][6, [0, 36]].any())
    parallax_summary = ring_support_summary(tilted_gate["contact_valid"], observed, tilted_gate["height_m"])
    check("b_rejected_parallax_retains_raw_negative_height",
          np.isclose(parallax_summary["raw_by_zone"][2, 3], .15, atol=1e-12))

    centre_offset = rings["points"].copy()
    centre_offset[0] = centre + [.0025, 0, .001]
    centre_gate0 = projected_ring_gate(centre_offset, observed, centre, axis, phase=phase)
    centre_gate1 = projected_ring_gate(centre_offset, observed, centre, axis, phase=phase + .731)
    check("b_centre_support_is_broadcast_and_independent_of_angular_phase",
          centre_gate0["contact_valid"][0].all() and centre_gate1["contact_valid"][0].all())
    centre_offset[0] = centre + [.0031, 0, 0]
    check("b_centre_support_rejects_radial_offset_beyond_three_mm",
          not projected_ring_gate(centre_offset, observed, centre, axis, phase=phase)["contact_valid"][0].any())
    centre_offset[0, 1] = centre
    reject("b_nonbroadcast_centre_observation_is_rejected",
           lambda: projected_ring_gate(centre_offset, observed, centre, axis, phase=phase))
    opposite = rings["points"].copy()
    opposite[7, 0] = 2 * centre - opposite[7, 0]
    opposite_gate = projected_ring_gate(opposite, observed, centre, axis, phase=phase)
    check("b_opposite_180deg_slot_fails_positive_radial_gate_despite_matching_rho",
          opposite_gate["radial_ok"][7, 0] and opposite_gate["tangential_ok"][7, 0]
          and opposite_gate["in_region"][7, 0] and not opposite_gate["contact_valid"][7, 0])
    off_angle = rings["points"].copy()
    off_angle[7, 0] = centre + [.020 * np.cos(np.pi / 6), .020 * np.sin(np.pi / 6), 0.]
    angle_gate = projected_ring_gate(off_angle, observed, centre, axis, phase=phase)
    check("b_tangential_mismatch_is_rejected_even_with_correct_radius",
          angle_gate["radial_ok"][7, 0] and angle_gate["positive_radial"][7, 0]
          and not angle_gate["contact_valid"][7, 0])
    wrong_zone = rings["points"].copy()
    wrong_zone[2, 0] = centre + [.010, 0, 0]
    wrong_zone[5, 0] = centre + [.018, 0, 0]
    zone_gate = projected_ring_gate(wrong_zone, observed, centre, axis, phase=phase)
    check("b_half_open_zone_edges_reject_core10mm_and_inner18mm",
          zone_gate["radial_ok"][[2, 5], 0].all()
          and not zone_gate["contact_valid"][[2, 5], 0].any())
    threshold = rings["points"].copy()
    threshold[1, 0] = centre + [.008, 0, 0]
    threshold[1, 1] = centre + .0080001 * (rings["points"][1, 1] - centre) / .005
    threshold_gate = projected_ring_gate(threshold, observed, centre, axis, phase=phase)
    check("b_three_mm_gate_is_inclusive_with_roundoff_only",
          threshold_gate["contact_valid"][1, 0] and not threshold_gate["contact_valid"][1, 1])

    no_observations = np.zeros_like(observed)
    no_points = np.full_like(rings["points"], np.nan)
    no_gate = projected_ring_gate(no_points, no_observations, centre, axis, phase=phase)
    no_summary = ring_support_summary(no_gate["contact_valid"], no_observations, no_gate["height_m"])
    check("b_empty_observations_have_zero_coverage_full_gaps_and_zero_raw_summary",
          np.array_equal(no_summary["coverage_by_radius"], np.tile([0., 1.], (9, 1)))
          and np.array_equal(no_summary["raw_by_zone"], np.zeros((3, 4)))
          and np.isfinite(no_summary["summary"]).all())
    reject("b_nan_position_cannot_be_declared_observed_valid",
           lambda: projected_ring_gate(no_points, observed, centre, axis, phase=phase))
    reject("b_contact_support_cannot_exist_without_observation",
           lambda: ring_support_summary(observed, no_observations, np.zeros((9, ANGLES))))
    half = np.r_[np.ones(36, dtype=bool), np.zeros(36, dtype=bool)]
    crossing = np.roll(half, 18)
    alternating = np.tile([True, False], 36)
    check("b_half_circle_gap_includes_wraparound_at_zero",
          maximum_circular_invalid_gap(half) == .5 and maximum_circular_invalid_gap(crossing) == .5)
    check("b_equal_missing_fraction_can_have_different_contiguous_gaps",
          half.mean() == alternating.mean() and maximum_circular_invalid_gap(alternating) == 1 / 72)
    partial_observed = observed.copy()
    partial_observed[1:] = half
    partial_contact = partial_observed.copy()
    partial_contact[1:, 5:10] = False
    partial_height = np.tile(.003 * np.sin(theta), (9, 1))
    partial_height[0] = 0.
    partial_summary = ring_support_summary(partial_contact, partial_observed, partial_height)
    shifted_summary = ring_support_summary(np.roll(partial_contact, 11, axis=1),
                                          np.roll(partial_observed, 11, axis=1),
                                          np.roll(partial_height, 11, axis=1))
    check("b_coverage_and_raw_summaries_are_cyclic_shift_invariant",
          np.allclose(partial_summary["summary"], shifted_summary["summary"], atol=1e-12, rtol=0))
    raw_points, raw_observed = no_points.copy(), no_observations.copy()
    raw_points[7, 0] = centre + [.030, 0, -.060]
    raw_points[7, 36] = centre + [-.030, 0, .070]
    raw_observed[7, [0, 36]] = True
    raw_gate = projected_ring_gate(raw_points, raw_observed, centre, axis, phase=phase)
    raw_summary = ring_support_summary(raw_gate["contact_valid"], raw_observed, raw_gate["height_m"])
    check("b_rejected_raw_protrusions_and_depressions_are_retained_without_height_clip",
          not raw_gate["contact_valid"].any()
          and np.allclose(raw_summary["raw_by_zone"][2], [2 / 216, 1, 1.5, 1.75], atol=1e-12, rtol=0))

    tokens = np.random.default_rng(321).normal(size=(3, ANGLES, 16))
    supported = np.tile(half, (3, 1))
    corrupted = tokens.copy()
    corrupted[~supported] = np.nan
    normal = per_angle_layer_norm(tokens, supported)
    corrupt_normal = per_angle_layer_norm(corrupted, supported)
    check("b_invalid_nan_tokens_do_not_change_valid_per_angle_channel_layernorm",
          np.array_equal(normal, corrupt_normal) and np.isfinite(corrupt_normal).all())
    affine = per_angle_layer_norm(corrupted, supported, weight=np.linspace(.5, 1.5, 16), bias=np.ones(16))
    check("b_affine_bias_cannot_create_valid_features_in_missing_angles",
          np.array_equal(affine[~supported], np.zeros_like(affine[~supported])))
    different = tokens.copy()
    different[1:] *= 1000
    different[0, 1] += np.arange(16) * 100
    different_normal = per_angle_layer_norm(different, supported)
    check("b_layernorm_does_not_mix_other_angles_or_other_queries",
          np.array_equal(normal[0, 0], different_normal[0, 0]))
    full_pool = masked_mean_max(tokens, supported)
    corrupt_pool = masked_mean_max(corrupted, supported)
    check("b_invalid_tokens_do_not_change_masked_mean_or_max",
          all(np.array_equal(left, right) for left, right in zip(full_pool, corrupt_pool)))
    empty_pool = masked_mean_max(np.full_like(tokens, np.nan), np.zeros_like(supported))
    check("b_empty_masked_pool_returns_finite_zero_mean_and_max",
          all(np.array_equal(value, np.zeros((3, 16))) for value in empty_pool))
    negative = -np.ones_like(tokens)
    negative[~supported] = 1e9
    check("b_invalid_zero_padding_does_not_bias_max_of_negative_valid_values",
          np.array_equal(masked_mean_max(negative, supported)[1], -np.ones((3, 16))))
    full_constant = masked_mean_max(np.ones((ANGLES, 2)), np.ones(ANGLES, dtype=bool))
    half_constant = masked_mean_max(np.ones((ANGLES, 2)), half)
    check("b_masked_pool_alone_cannot_replace_explicit_missing_coverage",
          all(np.array_equal(left, right) for left, right in zip(full_constant, half_constant)))

    return {"schema_version": "suction20.contact_boundary.v3", "quality_contract": QUALITY_CONTRACT,
            "scope": "synthetic_contact_algebra_not_model_teacher_export_or_performance_validation",
            "passed": True, "check_count": len(checks), "checks": checks,
            "python": platform.python_version(), "numpy": np.__version__,
            "measurements": {"maximum_basis_rotation_summary_error": max(errors),
                             "same_multiset_harmonic_amplitudes_mm": examples_mm.tolist(),
                             "parallax_rho_mm": parallax_mm.tolist()},
            "limits": ["Three harmonic magnitudes do not recover relative phase or higher modes.",
                       "Seven angular supports do not establish complete coverage or good conditioning.",
                       "Observed-sample descriptors remain sensitive to density, holes and depth error.",
                       "Ring gates preserve raw observations but neither infer hidden surfaces nor compute teacher collisions.",
                       "No model training, real-data quality, depth-image sampler, export or timing was tested."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["self-test"])
    parser.parse_args()
    print(json.dumps(selftest(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
