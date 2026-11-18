"""Native single-instance inference with explicit checkpoint and pose identity.

This module does not implement a public SuctionNet AP evaluator. All geometry
is in camera coordinates and metres. A tool's +Z is the approach direction,
the negative of the directed outward normal used to score the contact.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from .instance_geometry import (NORMAL_SPEC, SAMPLER_SPEC, PRECISION_SPEC,
                                prepare_instance)
from .instance_training import (CHECKPOINT_SCHEMA, QUALITY_CONTRACT,
                                create_instance_model, clean_metadata, file_record,
                                evaluation_mode, minimum_support_points)


PREDICTION_SCHEMA = 'instance_suction_prediction_v2'
INFERENCE_REPORT_SCHEMA = 'instance_suction_inference_v1'
ROLL_POLICY = 'least_aligned_camera_axis_cross_tool_z_xyz_tiebreak'
NMS_DISTANCE_M = .010
NMS_ANGLE_DEGREES = 15.
PUBLIC_EVALUATION = {'status': 'deferred', 'paper_comparison_ready': False}


def _numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def load_instance_checkpoint(path, device='cpu', *, backbone_root=None, query_chunk_size=None):
    """Load this project's own full training checkpoint, with strict weights.

    Training checkpoints contain RNG and optimizer data as well as tensors;
    ``weights_only=False`` is intentional. The caller must supply a trusted
    checkpoint produced by ``train_instance.py``. A relocated A backbone is
    still checked against the wrapper's pinned source blobs when constructed.
    """
    from .instance_config import (apply_model_runtime_overrides, model_execution_options,
                                  semantic_model_config, validate_saved_config_definition)
    path = Path(path)
    state = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(state, dict) or state.get('schema') != CHECKPOINT_SCHEMA:
        raise ValueError('Expected an instance_training_checkpoint_v1 checkpoint')
    config, identity, policy = (state.get(key) for key in ('config', 'identity', 'policy'))
    if not all(isinstance(value, dict) for value in (config, identity, policy)):
        raise ValueError('Checkpoint must include config, identity and selection policy')
    route, stage = config.get('route'), state.get('stage')
    if config.get('schema') != 'instance_runtime_v1' or route not in ('a', 'b'):
        raise ValueError('Checkpoint runtime config or route is invalid')
    if stage not in ('pretrain', 'finetune'):
        raise ValueError('Checkpoint stage must be pretrain or finetune')
    if (config.get('quality_contract') != QUALITY_CONTRACT or
        identity.get('quality_contract') != QUALITY_CONTRACT):
        raise ValueError('Checkpoint INST20 quality contract mismatch')
    if identity.get('route') != route or identity.get('stage') != stage:
        raise ValueError('Checkpoint route/stage identity mismatch')
    validate_saved_config_definition(config, identity)
    source_identity = identity.get('source', {}).get('data_identity', {})
    expected_preprocessing = {'normal_spec': NORMAL_SPEC, 'sampler_spec': SAMPLER_SPEC,
                              'precision_spec': PRECISION_SPEC, 'quality_contract': QUALITY_CONTRACT}
    if any(source_identity.get(key) != value for key, value in expected_preprocessing.items()):
        raise ValueError('Checkpoint source normal/sampler/precision/quality identity differs from this runtime')
    if (config.get('public_evaluation', {}).get('status') != 'deferred' or
        policy.get('public_evaluation_status') != 'deferred' or
        policy.get('paper_comparison_ready') is not False):
        raise ValueError('Public radius-only evaluation is deferred, not a checkpoint default')
    expected_split, expected_role = (('test', 'test_selected') if stage == 'pretrain'
                                     else ('val', 'validation_selected'))
    if (policy.get('selection_split') != expected_split or
        policy.get('selection_role') != expected_role or
        policy.get('train_scene_range') != [0, 99] or
        policy.get('independent_test_estimate') is not False):
        raise ValueError('Checkpoint selection policy does not match the declared source/target protocol')
    pretrain = config.get('pretrain', {})
    if pretrain.get('selection_split') != 'test' or pretrain.get('train_scene_range') != [0, 99]:
        raise ValueError('Checkpoint must declare all 100 source scenes and explicit test selection')
    step, kind = state.get('global_step'), state.get('checkpoint_kind')
    if isinstance(step, bool) or not isinstance(step, int) or step < 1:
        raise ValueError('Inference requires at least one completed optimizer update')
    if kind not in ('last', 'best'):
        raise ValueError('Checkpoint kind must be last or best')
    role = state.get('checkpoint_selection_role')
    expected_checkpoint_role = expected_role if kind == 'best' else 'budget_endpoint_or_progress'
    if role != expected_checkpoint_role:
        raise ValueError('Checkpoint selection role is inconsistent with its kind')
    if not isinstance(state.get('model'), dict):
        raise ValueError('Checkpoint is missing model weights')

    runtime_config = apply_model_runtime_overrides(config, backbone_root=backbone_root,
                                                    query_chunk_size=query_chunk_size)
    model = create_instance_model(runtime_config, device=device)
    model.load_state_dict(state['model'], strict=True)
    model.eval()
    architecture = getattr(model, 'architecture_metadata', None)
    architecture = architecture() if callable(architecture) else architecture
    metadata = {
        'schema': 'instance_loaded_checkpoint_v1',
        'checkpoint_file': file_record(path), 'route': route, 'stage': stage,
        'quality_contract': QUALITY_CONTRACT, 'checkpoint_kind': kind,
        'global_step': step, 'selection_policy': deepcopy(policy),
        'checkpoint_selection_role': role, 'checkpoint_identity': clean_metadata(identity),
        'initialization': clean_metadata(state.get('initialization')),
        'public_evaluation': deepcopy(PUBLIC_EVALUATION),
        'runtime_model_config': runtime_config.get('model', {}),
        'model_compatibility': {
            'semantic_model': semantic_model_config(route, config.get('model', {})),
            'checkpoint_execution': model_execution_options(route, config.get('model', {})),
            'runtime_execution': model_execution_options(route, runtime_config.get('model', {})),
        },
        'architecture': architecture,
    }
    return model, metadata


def resolve_ranking(stage, ranking='auto', alpha=None):
    """Keep calibrated task probability separate from optional mixed ranking."""
    if stage not in ('pretrain', 'finetune'):
        raise ValueError('stage must be pretrain or finetune')
    if ranking == 'auto':
        ranking = 'quality' if stage == 'pretrain' else 'task'
    if ranking not in ('quality', 'task', 'mixed'):
        raise ValueError('ranking must be auto, quality, task or mixed')
    if stage == 'pretrain' and ranking != 'quality':
        raise ValueError('A source checkpoint has no trained task head; task/mixed ranking is forbidden')
    if ranking != 'mixed':
        if alpha is not None:
            raise ValueError('alpha is only meaningful with explicit mixed ranking')
        return ranking, 0.
    alpha = .5 if alpha is None else float(alpha)
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError('Mixed ranking alpha must be finite and nonnegative')
    return ranking, alpha


def _unit_normals(normals):
    normals = np.asarray(normals, np.float64)
    lengths = np.linalg.norm(normals, axis=-1)
    if not np.isfinite(normals).all() or not np.isfinite(lengths).all() or np.any(lengths <= 1e-12):
        raise ValueError('Pose normals must be finite directed nonzero vectors')
    return normals / lengths[..., None]


def contact_transform(point_cam_m, normal_out_cam):
    """Return a proper SE(3) matrix without moving or flipping the scored pose."""
    point = np.asarray(point_cam_m, np.float64)
    normal = np.asarray(normal_out_cam, np.float64)
    if point.shape != (3,) or normal.shape != (3,) or not np.isfinite(point).all():
        raise ValueError('Contact point and normal must be finite three-vectors')
    z = -_unit_normals(normal)
    reference = np.eye(3)[int(np.argmin(np.abs(z)))]
    x = np.cross(reference, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.column_stack((x, y, z))
    transform[:3, 3] = point
    return transform


def nms_pose_candidates(points, normals, scores, query_ids, *,
                        distance_m=NMS_DISTANCE_M, angle_degrees=NMS_ANGLE_DEGREES):
    """Stable directed-axis NMS; both distance and angle tests are strict <.

    ``atan2(||cross||, dot)`` measures the directed angle without the poor
    conditioning of acos near parallel axes. Opposite axes remain distinct.
    Return all retained row indices; Top-K truncation is a separate operation.
    """
    points = np.asarray(points, np.float64)
    normals = np.asarray(normals, np.float64)
    scores, query_ids = np.asarray(scores, np.float64), np.asarray(query_ids)
    count = len(points)
    if points.shape != (count, 3) or normals.shape != points.shape or scores.shape != (count,):
        raise ValueError('NMS requires matching [Q,3] points/normals and [Q] scores')
    if (query_ids.shape != (count,) or query_ids.dtype.kind not in 'iu' or
        len(np.unique(query_ids)) != count or np.any(query_ids < 0)):
        raise ValueError('NMS query_ids must be unique nonnegative integers')
    if not np.isfinite(points).all() or not np.isfinite(scores).all():
        raise ValueError('NMS requires finite points and scores')
    if not math.isfinite(distance_m) or distance_m < 0 or not 0 <= angle_degrees <= 180:
        raise ValueError('Invalid NMS distance or directed angle')
    normals = _unit_normals(normals)
    threshold = math.radians(angle_degrees)
    selected = []
    for i in np.lexsort((query_ids, -scores)):
        if selected:
            previous = np.asarray(selected, np.int64)
            distance = np.linalg.norm(points[previous] - points[i], axis=1)
            dot = np.clip(normals[previous] @ normals[i], -1., 1.)
            sine = np.linalg.norm(np.cross(normals[previous], normals[i]), axis=1)
            angle = np.arctan2(sine, dot)
            if np.any((distance < distance_m) & (angle < threshold)):
                continue
        selected.append(int(i))
    return np.asarray(selected, np.int64)


def _synchronize(model):
    parameter = next(model.parameters(), None)
    if parameter is not None and parameter.device.type == 'cuda':
        torch.cuda.synchronize(parameter.device)


def _aligned_vector(prediction, key, count, *, boolean=False):
    if key not in prediction:
        raise ValueError(f'Model output is missing {key}')
    value = _numpy(prediction[key])
    if value.shape != (count,):
        raise ValueError(f'Model output {key} must have shape [{count}]')
    if boolean:
        if value.dtype.kind != 'b':
            raise ValueError(f'Model output {key} must be boolean')
    elif not np.isfinite(value).all():
        raise ValueError(f'Model output {key} is nonfinite; a failed run is not an empty instance')
    return value


def _support_diagnostics(prediction, row, count):
    fields = ('query_feature_valid', 'region_counts', 'lip_summary', 'independent_angular_support',
              'anchor_source_ids', 'anchor_distance_m')
    ring_fields = ('query_support_valid', 'unique_source_pixels_per_radius',
                   'bilinear_fallback_count', 'radial_gate_rejection_fraction',
                   'tangential_gate_rejection_fraction', 'opposite_angle_gate_rejection_fraction',
                   'native_underresolved_fraction', 'phase', 'support_summary', 'raw_summary')
    result = {}
    for source, names in ((prediction, fields), (prediction.get('diagnostics', {}), ring_fields)):
        for key in names:
            if key not in source:
                continue
            value = _numpy(source[key])
            if value.ndim == 0 or len(value) != count:
                raise ValueError(f'Support diagnostic {key} has no aligned query dimension')
            selected = value[row]
            if np.asarray(selected).dtype.kind in 'fc' and not np.isfinite(selected).all():
                raise ValueError(f'Valid candidate has a nonfinite {key} diagnostic')
            result[key] = selected.tolist()
    return result


@torch.no_grad()
def infer_instance(model, observation, *, stage, ranking='auto', alpha=None, top_k=5):
    """Encode once, score unchanged canonical queries, then serialize native poses.

    ``no_pose`` describes an observed lack of candidates/support. Malformed
    model outputs raise instead of silently becoming a successful empty frame.
    Explicitly ineligible input slots are recorded but never sent to the scorer;
    their point/normal placeholders do not affect eligible poses or their IDs.
    Model train/eval state is restored, allowing this API inside validation.
    """
    ranking, alpha = resolve_ranking(stage, ranking, alpha)
    for key, expected in (('normal_spec', NORMAL_SPEC), ('sampler_spec', SAMPLER_SPEC),
                          ('precision_spec', PRECISION_SPEC)):
        if observation.get(key) != expected:
            raise ValueError(f'Observation {key} differs from the current inference contract')
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 256:
        raise ValueError('top_k must be an integer in [1,256]')
    points = np.asarray(observation['query_points'], np.float64)
    normals = np.asarray(observation['query_normals'], np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or normals.shape != points.shape:
        raise ValueError('Observation queries must have matching [Q,3] arrays')
    count = len(points)
    query_ids = np.asarray(observation.get('query_ids', np.arange(count, dtype=np.int64)))
    source_ids = observation.get('query_source_ids')
    if (query_ids.shape != (count,) or query_ids.dtype.kind not in 'iu' or
        np.any(query_ids < 0) or len(np.unique(query_ids)) != count):
        raise ValueError('Observation query_ids must be unique nonnegative integers')
    if count > 256:
        raise ValueError('Native instance inference supports at most 256 canonical queries')
    if source_ids is not None:
        source_ids = np.asarray(source_ids)
        if source_ids.shape != (count,) or source_ids.dtype.kind not in 'iu' or np.any(source_ids < 0):
            raise ValueError('Query source pixel IDs must be aligned nonnegative integers')
    query_valid = np.asarray(observation.get('query_valid', np.ones(count, bool)))
    if query_valid.shape != (count,) or query_valid.dtype.kind != 'b':
        raise ValueError('Observation query_valid must be an aligned boolean vector')
    scored_rows = np.flatnonzero(query_valid)
    scored_points, scored_normals = points[scored_rows], normals[scored_rows]
    scored_count = len(scored_rows)
    if not np.isfinite(scored_points).all():
        raise ValueError('Eligible observation query points must be finite')
    unit_normals = _unit_normals(scored_normals)
    result = {
        'schema': PREDICTION_SCHEMA, 'instance_uid': str(observation.get('instance_uid', '')),
        'sample_uid': str(observation.get('sample_uid', '')),
        'status': 'no_pose', 'reason': 'no_eligible_queries', 'poses': [],
        'quality_contract': QUALITY_CONTRACT, 'stage': stage,
        'ranking': ranking, 'ranking_alpha': alpha, 'top_k': top_k,
        'coordinate_frame': 'camera', 'position_unit': 'm', 'tool_z': '-normal_out',
        'roll_policy': ROLL_POLICY,
        'normal_spec': observation.get('normal_spec'), 'sampler_spec': observation.get('sampler_spec'),
        'precision_spec': observation.get('precision_spec'),
        'query_identity': observation.get('query_identity'),
        'observation_identity': observation.get('observation_identity'),
        'intrinsics': np.asarray(observation['intrinsics'], np.float64).tolist(),
        'public_evaluation': deepcopy(PUBLIC_EVALUATION),
        'nms': {'distance_m': NMS_DISTANCE_M, 'angle_degrees': NMS_ANGLE_DEGREES,
                'comparison': 'strict_less_than_both', 'directed_normals': True},
        'counts': {'input_queries': count, 'eligible_queries': int(query_valid.sum()),
                   'geometry_valid_queries': 0, 'before_nms': 0, 'after_nms': 0, 'returned': 0},
        'rejected_queries': [], 'rejection_counts': {},
        'timing_ms': {'encode': 0., 'score': 0., 'postprocess': 0., 'total': 0.},
    }
    start = time.perf_counter()
    if len(observation.get('support_points', ())) < minimum_support_points(model):
        result['reason'] = 'insufficient_support'
        result['rejected_queries'] = [
            {'query_id': int(query_ids[i]),
             'reason': 'insufficient_support' if query_valid[i] else 'ineligible_query'}
            for i in range(count)]
        result['rejection_counts'] = dict(Counter(item['reason'] for item in result['rejected_queries']))
        result['timing_ms']['total'] = (time.perf_counter() - start) * 1000.
        return result
    if not scored_count:
        result['rejected_queries'] = [
            {'query_id': int(query_ids[i]), 'reason': 'ineligible_query'}
            for i in range(count)]
        result['rejection_counts'] = {'ineligible_query': count} if count else {}
        result['timing_ms']['total'] = (time.perf_counter() - start) * 1000.
        return result

    with evaluation_mode(model):
        _synchronize(model)
        begin_encode = time.perf_counter()
        encoded = model.encode_instance(observation)
        _synchronize(model)
        begin_score = time.perf_counter()
        prediction = model.score_at_pose(encoded, scored_points, scored_normals)
        _synchronize(model)
        begin_postprocess = time.perf_counter()
    if not isinstance(prediction, dict):
        raise ValueError('Model must return a prediction dictionary')
    valid = _aligned_vector(prediction, 'valid', scored_count, boolean=True)
    vectors = {key: _aligned_vector(prediction, key, scored_count).astype(np.float64)
               for key in ('q_inst', 'seal', 'wrench', 'ell', 'task_logit')}
    for key in ('q_inst', 'seal', 'wrench'):
        if np.any((vectors[key] < 0) | (vectors[key] > 1)):
            raise ValueError(f'Model output {key} lies outside [0,1]')
    if np.any((vectors['ell'] < -6) | (vectors['ell'] > 1)):
        raise ValueError('Model ell must be clamped log Laplace scale in [-6,1]')
    # These return fields are optional, but when supplied they must confirm the
    # pose actually scored, including B's explicit camera-coordinate names.
    for key in ('points', 'points_cam_m'):
        if key in prediction:
            output = _numpy(prediction[key])
            if output.shape != scored_points.shape or not np.array_equal(output, scored_points):
                raise ValueError('Model moved the query contact after scoring')
    for key in ('normal_out', 'normals_out_cam'):
        if key in prediction:
            output = _numpy(prediction[key])
            if output.shape != scored_normals.shape or not np.allclose(output[valid], unit_normals[valid], rtol=0, atol=4e-15):
                raise ValueError('Model changed the directed axis used for scoring')
            unit_normals[valid] = output[valid]
    # Stable sigmoid avoids overflow for otherwise finite extreme logits.
    task_probability = np.exp(-np.logaddexp(0., -vectors['task_logit']))
    if ranking == 'quality':
        scores = vectors['q_inst']
    elif ranking == 'task':
        scores = task_probability
    else:
        scores = task_probability * np.maximum(vectors['q_inst'], 1e-6) ** alpha
    eligible = np.flatnonzero(valid)
    kept_local = nms_pose_candidates(scored_points[eligible], unit_normals[eligible],
                                    scores[eligible], query_ids[scored_rows[eligible]])
    kept = eligible[kept_local]
    reasons = prediction.get('reason')
    if reasons is not None and len(reasons) != scored_count:
        raise ValueError('Model reason list has no aligned query dimension')
    rejected = {int(row): 'ineligible_query' for row in np.flatnonzero(~query_valid)}
    for row in np.flatnonzero(~valid):
        reason = str(reasons[row]) if reasons is not None else 'pose_geometry_invalid'
        if reason == 'ok':
            raise ValueError('Invalid model query cannot have an ok reason')
        rejected[int(scored_rows[row])] = reason
    result['rejected_queries'] = [
        {'query_id': int(query_ids[row]), 'reason': rejected[row]} for row in sorted(rejected)]
    result['rejection_counts'] = dict(Counter(item['reason'] for item in result['rejected_queries']))
    for row in kept[:top_k]:
        original_row = scored_rows[row]
        normal = unit_normals[row]
        transform = contact_transform(scored_points[row], normal)
        pose = {
            'query_id': int(query_ids[original_row]),
            'source_point_id': int(source_ids[original_row]) if source_ids is not None else None,
            'point_cam_m': scored_points[row].tolist(), 'normal_out_cam': normal.tolist(),
            'tool_z_cam': transform[:3, 2].tolist(), 'T_cam_contact': transform.tolist(),
            'pose_id': f"{result['instance_uid']}:q{int(query_ids[original_row])}",
            'q_inst': float(vectors['q_inst'][row]), 'seal': float(vectors['seal'][row]),
            'wrench': float(vectors['wrench'][row]), 'ell': float(vectors['ell'][row]),
            'task_probability': float(task_probability[row]) if stage == 'finetune' else None,
            'task_logit': float(vectors['task_logit'][row]) if stage == 'finetune' else None,
            'ranking_score': float(scores[row]), 'pose_support_valid': True,
            'support': _support_diagnostics(prediction, int(row), scored_count),
        }
        result['poses'].append(pose)
    result['counts'].update(geometry_valid_queries=int(valid.sum()), before_nms=len(eligible),
                            after_nms=len(kept), returned=len(result['poses']))
    if result['poses']:
        result.update(status='ok', reason='ok')
    else:
        result['reason'] = 'no_geometry_supported_queries'
    end = time.perf_counter()
    result['timing_ms'].update(encode=(begin_score-begin_encode)*1000.,
                               score=(begin_postprocess-begin_score)*1000.,
                               postprocess=(end-begin_postprocess)*1000., total=(end-start)*1000.)
    return result


def load_intrinsics(path):
    """Read a JSON 3x3 matrix or {fx,fy,cx,cy}; never infer camera calibration."""
    with Path(path).open() as stream:
        value = json.load(stream)
    if isinstance(value, dict):
        if 'intrinsics' in value or 'K' in value:
            value = value.get('intrinsics', value.get('K'))
        elif all(key in value for key in ('fx', 'fy', 'cx', 'cy')):
            value = [[value['fx'], 0, value['cx']], [0, value['fy'], value['cy']], [0, 0, 1]]
        else:
            raise ValueError('Intrinsics JSON must contain K, intrinsics, or fx/fy/cx/cy')
    matrix = np.asarray(value, np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError('Intrinsics must be a finite 3x3 matrix')
    return matrix


def _load_image_array(path, *, rgb=False):
    path = Path(path)
    if path.suffix.lower() == '.npy':
        return np.load(path, allow_pickle=False)
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB') if rgb else image).copy()


def load_raw_observation(*, rgb_path, depth_path, mask_path, intrinsics_path,
                         depth_scale_m, gravity, instance_id=None, max_queries=256,
                         depth_min_m=.2, depth_max_m=1.2):
    """Prepare one supplied binary/instance mask from native organized RGB-D."""
    if not math.isfinite(depth_scale_m) or depth_scale_m <= 0:
        raise ValueError('depth_scale_m must be finite and positive')
    rgb = _load_image_array(rgb_path, rgb=True)
    depth = _load_image_array(depth_path)
    mask = _load_image_array(mask_path)
    if depth.ndim != 2 or mask.shape != depth.shape:
        raise ValueError('Depth and mask must have matching native HxW shapes')
    if instance_id is not None:
        if isinstance(instance_id, bool) or not isinstance(instance_id, int) or instance_id < 1:
            raise ValueError('instance_id must be a positive integer; zero is background')
        if mask.dtype.kind not in 'biu':
            raise ValueError('An instance-ID mask must contain integer IDs')
        instance_mask = mask == instance_id
    else:
        values = set(np.unique(mask).tolist())
        if not (values <= {0, 1} or values <= {0, 255}):
            raise ValueError('A multi-instance mask requires an explicit instance_id')
        instance_mask = mask != 0
    return prepare_instance(rgb, np.asarray(depth, np.float64) * float(depth_scale_m),
                            instance_mask, load_intrinsics(intrinsics_path), gravity,
                            max_queries=max_queries, depth_min_m=depth_min_m, depth_max_m=depth_max_m)
