"""Training and explicit test-selected checkpoint policy for instance routes.

This module never invokes a public paper evaluator. INST20 selection and
paper AP are different protocols; the radius-only proposal remains deferred.
"""
from __future__ import annotations

from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
import os
from contextlib import contextmanager
from copy import deepcopy
import gzip
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import yaml

from .metrics import attempt_metrics
from .instance_execution import (EXECUTION_SCHEMA, configure_model_execution, execution_math_mode, resolve_execution,
    sample_evaluation_steps, source_instance_budget, source_update_size)

CHECKPOINT_SCHEMA = 'instance_training_checkpoint_v1'
QUALITY_CONTRACT = 'gt_INST20_pose_v2'


RUNTIME_CONTRACT = 'instance_runtime_semantic_checks_v2'


def clean_metadata(value):
    """Discard obsolete content-checksum metadata without reading any asset."""
    if isinstance(value, dict):
        return {k: clean_metadata(v) for k, v in value.items()
                if 'sha256' not in str(k).lower().replace('_', '').replace('-', '')}
    if isinstance(value, (tuple, list)):
        return [clean_metadata(v) for v in value]
    return deepcopy(value)


def file_record(path):
    """Cheap provenance only; size and path do not authenticate file contents."""
    path = Path(path)
    return {'path': str(path.resolve()), 'size_bytes': path.stat().st_size}


@contextmanager
def evaluation_mode(model):
    """Restore every module's exact mixed train/eval flags, also on errors."""
    previous_states = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        with execution_math_mode(getattr(model, '_instance_execution', None), training=False):
            yield
    finally:
        # train(flag) recurses and would overwrite descendants with mixed modes.
        for module, flag in previous_states:
            module.training = flag


def minimum_support_points(model):
    value = getattr(model, 'min_support_points', 100)
    if isinstance(value, bool) or int(value) != value or value < 1:
        raise ValueError('min_support_points must be a positive integer')
    return int(value)


def _positive_update(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def scheduled_evaluation_steps(stage_config):
    """Resolve one unambiguous schedule; the original budget endpoint is mandatory."""
    total = _positive_update(stage_config['steps'], 'steps')
    if stage_config.get('evaluate_every_instances') is not None:
        return sample_evaluation_steps(stage_config)
    explicit = stage_config.get('evaluation_steps')
    every = stage_config.get('evaluate_every')
    if explicit is not None:
        if every is not None:
            raise ValueError('Choose evaluation_steps or evaluate_every, not both')
        if not isinstance(explicit, list):
            raise ValueError('evaluation_steps must be a sorted list of successful update numbers')
        nodes = [_positive_update(step, 'evaluation_steps entry') for step in explicit]
        if nodes != sorted(set(nodes)) or any(step > total for step in nodes):
            raise ValueError('evaluation_steps must be unique, increasing and within the training budget')
    else:
        interval = _positive_update(every, 'evaluate_every')
        nodes = list(range(interval, total + 1, interval))
    if not nodes or nodes[-1] != total:
        nodes.append(total)
    return tuple(nodes)


def checkpoint_interval(stage_config):
    value = stage_config.get('checkpoint_every')
    if value is None:
        value = stage_config.get('evaluate_every') or stage_config['steps']
    return _positive_update(value, 'checkpoint_every')


def load_instance_config(path):
    from .instance_selection import validate_selection_config
    from .instance_config import B_FEATURE_FRAME
    with Path(path).open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or config.get('schema') != 'instance_runtime_v1':
        raise ValueError('Expected an instance_runtime_v1 runtime config, not design_spec')
    config = clean_metadata(config)
    if config.get('route') not in ('a', 'b'):
        raise ValueError('route must be a or b')
    if config['route'] == 'b':
        options = config.setdefault('model', {})
        if not isinstance(options, dict):
            raise ValueError('Instance model configuration must be a mapping')
        options.setdefault('min_support_points', 100)
        options.setdefault('feature_frame', B_FEATURE_FRAME)
    if config.get('quality_contract') != QUALITY_CONTRACT:
        raise ValueError('Instance quality identity mismatch')
    if config.get('public_evaluation', {}).get('status') != 'deferred':
        raise ValueError('Public radius evaluation is deferred; no implicit default is implemented')
    pre = config['pretrain']
    if pre.get('selection_split') != 'test' or pre.get('train_scene_range') != [0, 99]:
        raise ValueError('Current source policy is 100 train scenes and explicit test selection')
    validate_selection_config(pre)
    for stage in ('pretrain', 'finetune'):
        row = config[stage]
        scheduled_evaluation_steps(row)
        checkpoint_interval(row)
        if row.get('evaluation_records', 'best_and_final') not in ('best_and_final', 'all', 'none'):
            raise ValueError('evaluation_records must be best_and_final, all or none')
    if not 1 <= int(pre['queries']) <= 256 or int(pre['accumulate_instances']) < 1:
        raise ValueError('Invalid query or effective batch budget')
    if int(config['finetune']['head_only_steps']) < 0:
        raise ValueError('head_only_steps must be nonnegative')
    execution = resolve_execution(config)
    if execution is not None:
        config['execution'] = execution
    return config


def create_instance_model(config, device='cpu'):
    route = config['route']
    options = deepcopy(config.get('model', {}))
    if route == 'a':
        from models.instance_route_a import InstanceSuction3D
        model = InstanceSuction3D(**options)
    elif route == 'b':
        from .instance_config import B_LEGACY_FEATURE_FRAME
        from models.instance_route_b import InstanceSuction2DFastV3
        # Saved configs predating the chart field retain their original grid.
        # load_instance_config materializes the new default for fresh runs.
        options.setdefault('feature_frame', B_LEGACY_FEATURE_FRAME)
        model = InstanceSuction2DFastV3(**options)
    else:
        raise ValueError(f'Unknown route: {route}')
    model = model.to(device)
    configure_model_execution(model, resolve_execution(config))
    return model


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:
        if not torch.cuda.is_available() or len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('Resume CUDA RNG topology differs from the saved run')
        torch.cuda.set_rng_state_all(state['cuda'])


class InstanceSchedule:
    """Uniform scene -> frame -> instance; state is saved at optimizer boundaries."""
    def __init__(self, records, seed, *, hierarchy=True):
        if not records:
            raise ValueError('Empty training dataset')
        self.rng = np.random.default_rng(seed)
        self.draws = 0
        self.hierarchy = hierarchy
        grouped = defaultdict(lambda: defaultdict(list))
        for i, row in enumerate(records):
            grouped[int(row.get('scene_id', 0))][int(row.get('frame_id', i))].append(i)
        self.groups = {s: dict(f) for s, f in sorted(grouped.items())}
        self.scenes = list(self.groups)
        self.length = len(records)
        # Stable semantic keys, not file contents or a content digest. Save the
        # actual sequence so resumed index draws have the same meaning.
        self.identity = [(int(row.get('scene_id', 0)), int(row.get('frame_id', i)),
                          row.get('instance_row'), str(row.get('sample_id', row.get('instance_uid', i))))
                         for i, row in enumerate(records)]

    def next(self):
        self.draws += 1
        if not self.hierarchy:
            return int(self.rng.integers(self.length))
        scene = self.scenes[int(self.rng.integers(len(self.scenes)))]
        frames = self.groups[scene]
        frame_ids = list(frames)
        frame = frame_ids[int(self.rng.integers(len(frame_ids)))]
        entries = frames[frame]
        return entries[int(self.rng.integers(len(entries)))]

    def state_dict(self):
        return {'rng': deepcopy(self.rng.bit_generator.state), 'draws': self.draws,
                'identity': self.identity, 'hierarchy': self.hierarchy}

    def load_state_dict(self, state):
        if state['hierarchy'] != self.hierarchy:
            raise ValueError('Sampler strategy changed across resume')
        # Earlier checkpoints stored only a checksum. It is ignored; new states
        # compare the concrete semantic order directly.
        if isinstance(state.get('identity'), (list, tuple)) and state['identity'] != self.identity:
            raise ValueError('Sampler data/order/strategy changed across resume')
        self.rng.bit_generator.state = state['rng']
        self.draws = int(state['draws'])


def augment_rgb(observation, rng):
    """One color transform for both model paths; geometry and validity are unchanged."""
    return _apply_rgb(observation, _draw_rgb(rng))


def _draw_rgb(rng):
    # The coordinator samples these in consumption order, including skipped draws.
    factors = [float(rng.uniform(.8, 1.2)) for _ in range(3)]
    order = rng.permutation(3)
    hue = int(round(float(rng.uniform(-.05, .05)) * 255))
    return factors, order, hue, rng.random() < .2


def _apply_rgb(observation, parameters):
    factors, order, hue, grayscale = parameters
    obs = dict(observation)
    rgb = np.asarray(obs['rgb'], dtype=np.float32).copy()
    valid = (np.asarray(obs['rgb_valid'], bool).copy() if 'rgb_valid' in obs
             else np.isfinite(rgb).all(-1))
    valid &= np.asarray(obs['mask'], bool)
    # PIL HSV conversion provides deterministic CPU color jitter without torchvision.
    from PIL import Image, ImageEnhance
    clean = np.nan_to_num(rgb, nan=0., posinf=0., neginf=0.)
    clean[~valid] = 0
    im = Image.fromarray(np.round(np.clip(clean, 0, 1) * 255).astype(np.uint8))
    transforms = [ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color]
    for i in order:
        im = transforms[int(i)](im).enhance(factors[int(i)])
    hsv = np.asarray(im.convert('HSV')).copy()
    hsv[..., 0] = (hsv[..., 0].astype(np.int16) + hue) % 256
    im = Image.fromarray(hsv, mode='HSV').convert('RGB')
    if grayscale:
        im = im.convert('L').convert('RGB')
    result = np.asarray(im, dtype=np.float32) / 255.
    result[~valid] = 0.
    obs['rgb'] = result
    obs['rgb_valid'] = valid
    uv = np.asarray(obs['support_uv'], np.int64)
    obs['support_rgb'] = result[uv[:, 1], uv[:, 0]].copy()
    obs['support_rgb_valid'] = valid[uv[:, 1], uv[:, 0]].copy()
    return obs


def _array(labels, canonical, alias=None):
    key = canonical if canonical in labels else alias
    if key is None or key not in labels:
        raise KeyError(f'Missing label field {canonical}')
    return np.asarray(labels[key])


def source_loss(prediction, labels):
    q = prediction['q_inst'].reshape(-1)
    device = q.device
    valid = torch.as_tensor(_array(labels, 'label_valid', 'valid'), device=device, dtype=torch.bool)
    if valid.shape != q.shape:
        raise ValueError('Prediction/query/label prefix length mismatch')
    pv = prediction.get('valid', prediction.get('query_valid'))
    if pv is not None:
        valid = valid & pv.reshape(-1).bool()
    if not bool(valid.any()):
        return None
    targets = [torch.as_tensor(_array(labels, k, alias), device=device, dtype=q.dtype)
               for k, alias in [('q_inst', 'Q_inst'), ('seal', 'S'), ('wrench', 'W')]]
    values = [q[valid], prediction['seal'].reshape(-1)[valid],
              prediction['wrench'].reshape(-1)[valid]]
    truth = [t[valid] for t in targets]
    if not all(bool(torch.isfinite(t).all()) and bool(((t >= 0) & (t <= 1)).all()) for t in truth):
        raise ValueError('Known physical labels must be finite probabilities')
    ell = prediction.get('ell', prediction.get('log_scale'))
    if ell is None:
        raise ValueError('Laplace ell=log(scale), not log-variance, is required')
    ell = ell.reshape(-1)[valid].clamp(-6., 1.)
    if not all(bool(torch.isfinite(t).all()) for t in values + [ell]):
        raise FloatingPointError('Nonfinite model output on supported queries')
    return ((torch.exp(-ell) * (values[0] - truth[0]).abs() + ell).mean()
            + .5 * F.smooth_l1_loss(values[1], truth[1], beta=.05)
            + .25 * F.smooth_l1_loss(values[2], truth[2], beta=.05))


def batched_source_losses(predictions, labels):
    """FP32 loss per supported instance, with one packed query reduction.

    Each instance has equal weight regardless of its number of valid queries.
    Unknown targets are selected out before arithmetic, never multiplied by 0.
    """
    if len(predictions) != len(labels):
        raise ValueError('Model output count differs from the requested instance batch')
    if not predictions:
        raise ValueError('Batched source loss requires at least one instance')
    device = predictions[0]['q_inst'].device
    values, targets, masks, memberships = [], [], [], []
    for index, (prediction, truth) in enumerate(zip(predictions, labels)):
        q = prediction['q_inst'].reshape(-1)
        mask = torch.as_tensor(_array(truth, 'label_valid', 'valid'), device=device, dtype=torch.bool)
        if mask.shape != q.shape:
            raise ValueError('Prediction/query/label prefix length mismatch')
        pv = prediction.get('valid', prediction.get('query_valid'))
        if pv is not None:
            mask = mask & pv.reshape(-1).bool()
        ell = prediction.get('ell', prediction.get('log_scale'))
        if ell is None:
            raise ValueError('Laplace ell=log(scale), not log-variance, is required')
        values.append(torch.stack((q.float(), prediction['seal'].reshape(-1).float(),
                                    prediction['wrench'].reshape(-1).float(), ell.reshape(-1).float()), -1))
        targets.append(torch.as_tensor(np.stack([_array(truth, key, alias) for key, alias in
                        [('q_inst', 'Q_inst'), ('seal', 'S'), ('wrench', 'W')]], -1),
                        device=device, dtype=torch.float32))
        masks.append(mask)
        memberships.append(torch.full((len(q),), index, device=device, dtype=torch.long))
    valid = torch.cat(masks)
    value, truth = torch.cat(values)[valid], torch.cat(targets)[valid]
    member = torch.cat(memberships)[valid]
    if not bool((torch.isfinite(truth) & (truth >= 0) & (truth <= 1)).all()):
        raise ValueError('Known physical labels must be finite probabilities')
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError('Nonfinite model output on supported queries')
    ell = value[:, 3].clamp(-6., 1.)
    per_query = (torch.exp(-ell) * (value[:, 0] - truth[:, 0]).abs() + ell
        + .5 * F.smooth_l1_loss(value[:, 1], truth[:, 1], beta=.05, reduction='none')
        + .25 * F.smooth_l1_loss(value[:, 2], truth[:, 2], beta=.05, reduction='none'))
    sums = per_query.new_zeros(len(predictions)).scatter_add(0, member, per_query)
    counts = per_query.new_zeros(len(predictions)).scatter_add(0, member, torch.ones_like(per_query))
    return (sums / counts.clamp_min(1))[counts > 0]


def attempt_loss(model, item):
    obs, attempts = item['observation'], item['attempts']
    p = _array(attempts, 'point_cam_m', 'points')
    n = _array(attempts, 'normal_out_cam', 'normal_out')
    encoded = model.encode_instance(obs)
    prediction = model.score_at_pose(encoded, p, n)
    logits = prediction['task_logit'].reshape(-1)
    y = torch.as_tensor(_array(attempts, 'success', 'outcome'), device=logits.device, dtype=logits.dtype)
    valid = torch.as_tensor(attempts.get('label_valid', np.ones(len(p), bool)),
                            device=logits.device, dtype=torch.bool)
    valid &= torch.as_tensor(attempts.get('pose_support_valid', np.ones(len(p), bool)),
                             device=logits.device, dtype=torch.bool)
    valid &= torch.as_tensor(attempts.get('pose_valid', np.ones(len(p), bool)),
                             device=logits.device, dtype=torch.bool)
    if 'valid' in prediction:
        valid &= prediction['valid'].bool().reshape(-1)
    if not bool(valid.any()):
        return None
    if not bool(torch.isfinite(logits[valid]).all()) or not bool(((y[valid] == 0) | (y[valid] == 1)).all()):
        raise ValueError('Invalid task logit or binary action label')
    return F.binary_cross_entropy_with_logits(logits[valid], y[valid])


def _to_numpy(tensor):
    return tensor.detach().cpu().numpy() if torch.is_tensor(tensor) else np.asarray(tensor)


class _InstanceMetrics:
    """Keep only three scalar metrics per instance, never large record dicts."""
    def __init__(self):
        self.frames = defaultdict(list)
        self.instance_count = 0
        self.no_pose_count = 0

    def add(self, row):
        self.frames[(row['scene_id'], row['frame_id'])].append(
            (row['hit'], row['selected_quality'], row['coverage']))
        self.instance_count += 1
        self.no_pose_count += row['selected_query'] is None

    def result(self):
        scenes = defaultdict(list)
        for (scene, _), values in self.frames.items():
            # Use the original per-column reduction order for selection ties.
            scenes[scene].append([float(np.mean([r[k] for r in values])) for k in range(3)])
        if not scenes:
            raise ValueError('No instances available for source selection')
        names = ('InstTop1_INST20_at_0_4', 'selected_Q_macro', 'Coverage_INST20_at_256')
        result = {name: float(np.mean([np.mean([f[k] for f in fs]) for fs in scenes.values()]))
                  for k, name in enumerate(names)}
        result.update(instance_count=self.instance_count, frame_count=len(self.frames),
                      scene_count=len(scenes), no_pose_count=self.no_pose_count)
        return result


def aggregate_instance_rows(rows):
    accumulator = _InstanceMetrics()
    for row in rows:
        accumulator.add(row)
    return accumulator.result()


def _workers(name):
    count = int(os.environ.get(name, '0'))
    if count < 0:
        raise ValueError(f'{name} must be nonnegative')
    return count


def _ordered_map(fn, tickets, workers):
    """At most workers reads in flight; retain input order and propagate errors."""
    tickets = iter(tickets)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = deque(pool.submit(fn, item) for item in islice(tickets, workers))
        while pending:
            result = pending.popleft().result()
            for item in islice(tickets, 1):
                pending.append(pool.submit(fn, item))
            yield result
            del result


def _geometry_options(model):
    options = {'min_support_points': minimum_support_points(model)}
    if hasattr(model, 'backbone_points'):
        options.update(backbone_points=model.backbone_points, query_chunk_size=model.query_chunk_size)
    return options


def _evaluation_items(dataset, prepare_geometry=None, *, model=None, execution=None):
    if execution is not None:
        from .instance_loading import InstancePreparationPool
        route = ('a' if hasattr(model, 'backbone_points') else
                 'b' if hasattr(model, 'feature_frame') else None)
        with InstancePreparationPool(dataset, workers=execution['evaluation_workers'],
                prefetch=execution['prefetch_factor'], route=route,
                geometry_options=_geometry_options(model) if model is not None else None) as pool:
            yield from pool.map((index, None) for index in range(len(dataset)))
        return
    workers = _workers('SUCTION20_INSTANCE_EVAL_WORKERS')
    if not workers:
        yield from dataset
        return
    read = getattr(dataset, 'load_uncached', dataset.__getitem__)
    def prepare(index):
        item = read(index)
        if prepare_geometry is not None:
            item = dict(item, observation=prepare_geometry(item['observation']))
        return item
    yield from _ordered_map(prepare, range(len(dataset)), workers)


def _source_evaluation_predictions(model, dataset):
    execution = getattr(model, '_instance_execution', None)
    threshold = minimum_support_points(model)
    iterator = _evaluation_items(dataset, getattr(model, 'prepare_source_geometry', None),
                                 model=model, execution=execution)
    size = execution['evaluation_microbatch_instances'] if execution is not None else 1
    try:
        while group := list(islice(iterator, size)):
            eligible = [i for i, item in enumerate(group) if len(item['observation']['query_points'])
                        and len(item['observation']['support_points']) >= threshold]
            outputs = {}
            if eligible:
                observations = [group[i]['observation'] for i in eligible]
                predictions = model.forward_instances(observations) if execution is not None else [model(observations[0])]
                if len(predictions) != len(eligible):
                    raise ValueError('Model output count differs from the requested evaluation batch')
                outputs = dict(zip(eligible, predictions))
            for index, item in enumerate(group):
                yield item, outputs.get(index)
    finally:
        iterator.close()


@torch.no_grad()
def evaluate_instance_source(model, dataset, *, collect_records=True, record_sink=None):
    rows, overall, subsets = [], _InstanceMetrics(), {}
    threshold = minimum_support_points(model)
    with evaluation_mode(model):
        for item, pred in _source_evaluation_predictions(model, dataset):
            obs, labels, record = item['observation'], item['labels'], item['record']
            count = len(obs['query_points'])
            if labels is None:
                raise ValueError('Selection requires complete known teacher labels')
            valid_label = _array(labels, 'label_valid', 'valid').astype(bool)
            quality = _array(labels, 'q_inst', 'Q_inst')
            if len(quality) != count or not valid_label.all() or not np.isfinite(quality).all():
                raise ValueError('Incomplete teacher for selection/coverage; no unknown-to-zero fill')
            selected = None
            if pred is not None:
                valid = _to_numpy(pred.get('valid', np.ones(count, bool))).astype(bool)
                scores = _to_numpy(pred['q_inst']).reshape(-1)
                if len(scores) != count or not np.isfinite(scores[valid]).all():
                    raise ValueError('Invalid source inference scores')
                ids = np.flatnonzero(valid)
                if len(ids):
                    query_ids = np.asarray(obs.get('query_ids', np.arange(count)))
                    selected = int(ids[np.lexsort((query_ids[ids], -scores[ids]))[0]])
            selected_q = float(quality[selected]) if selected is not None else 0.
            row = {'scene_id': int(record['scene_id']), 'frame_id': int(record['frame_id']),
                   'instance_uid': str(record.get('instance_uid', record.get('uid', record.get('sample_id', '')))),
                   'test_subset': record.get('test_subset'), 'selected_query': selected,
                   'selected_quality': selected_q, 'hit': float(selected is not None and selected_q >= .4),
                   'coverage': float(np.any(quality >= .4))}
            overall.add(row)
            if row['test_subset'] in ('seen', 'similar', 'novel'):
                subsets.setdefault(row['test_subset'], _InstanceMetrics()).add(row)
            if collect_records:
                rows.append(row)
            if record_sink is not None:
                record_sink(row)
    metrics = overall.result()
    metrics['by_subset'] = {s: a.result() for s, a in subsets.items()}
    metrics['evaluation_protocol'] = 'native_INST20_not_public_AP'
    return metrics, rows


@torch.no_grad()
def evaluate_instance_attempts(model, dataset, *, collect_records=True, record_sink=None):
    scores, outcomes, rows = [], [], []
    total = 0
    with evaluation_mode(model):
        for item in _evaluation_items(dataset, execution=getattr(model, '_instance_execution', None)):
            a = item['attempts']
            p, n = _array(a, 'point_cam_m', 'points'), _array(a, 'normal_out_cam', 'normal_out')
            pred = model.score_at_pose(model.encode_instance(item['observation']), p, n)
            probability = torch.sigmoid(pred['task_logit']).detach().cpu().numpy().reshape(-1)
            valid = np.asarray(a.get('label_valid', np.ones(len(p), bool)), bool).copy()
            valid &= np.asarray(a.get('pose_support_valid', np.ones(len(p), bool)), bool)
            valid &= np.asarray(a.get('pose_valid', np.ones(len(p), bool)), bool)
            valid &= _to_numpy(pred.get('valid', np.ones(len(p), bool))).astype(bool)
            y = _array(a, 'success', 'outcome')
            total += len(p)
            if not np.isfinite(probability[valid]).all():
                raise ValueError('Nonfinite task probability')
            scores.extend(probability[valid].tolist()); outcomes.extend(y[valid].tolist())
            for i in range(len(p)):
                row = {'observation_uid': str(item['record'].get('observation_uid', item['record'].get('sample_id', ''))),
                       'attempt_id': str(a.get('attempt_ids', list(range(len(p))))[i]),
                       'action_index': i, 'supported': bool(valid[i]),
                       'pose_source': a.get('pose_source', 'planned_pipeline_contact'),
                       'pose_valid': bool(a.get('pose_valid', np.ones(len(p), bool))[i]),
                       'pose_support_valid': bool(a.get('pose_support_valid', np.ones(len(p), bool))[i]),
                       'actual_execution_verified': bool(a.get('actual_execution_verified_by_attempt',
                           np.full(len(p), a.get('actual_execution_verified', False)))[i]),
                       'probability': float(probability[i]) if valid[i] else None,
                       'outcome': int(y[i]) if valid[i] else None}
                if collect_records:
                    rows.append(row)
                if record_sink is not None:
                    record_sink(row)
    metrics = attempt_metrics(np.asarray(scores), np.asarray(outcomes), calibrated=True)
    metrics.update(total_attempts=total, supported_attempts=len(scores), unsupported_attempts=total-len(scores))
    return metrics, rows


def check_source_splits(dataset, expected, *, allow_partial=False):
    rows = dataset.records
    expected_set = set(range(0, 100)) if expected == 'train' else set(range(100, 190))
    actual = {int(r['scene_id']) for r in rows}
    if not actual <= expected_set or any(r['split'] != expected for r in rows):
        raise ValueError(f'{expected} manifest contains scenes from another split')
    frames = defaultdict(set)
    for row in rows:
        frame = int(row['frame_id'])
        if frame not in range(256):
            raise ValueError('Frame index outside official 0..255 range')
        frames[int(row['scene_id'])].add(frame)
    with_instances = sum(map(len, frames.values()))
    empty_frames = []
    manifest = getattr(dataset, 'manifest', getattr(dataset, 'manifest_path', None))
    if manifest is not None:
        summary_path = Path(manifest).with_suffix('.summary.json')
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text())
            if summary.get('manifest', Path(manifest).name) != Path(manifest).name:
                raise ValueError('Manifest coverage summary names another manifest')
            # Coverage is checked structurally below; obsolete file checksums
            # are intentionally not read or evaluated.
            for scene, frame in summary.get('empty_gt_frames', []):
                if int(scene) not in expected_set or int(frame) not in range(256):
                    continue  # A combined train/test summary can list both splits.
                if int(frame) in frames[int(scene)]:
                    raise ValueError('A frame cannot contain instances and be declared GT-empty')
                frames[int(scene)].add(int(frame))
                empty_frames.append((int(scene), int(frame)))
    if not allow_partial and (set(frames) != expected_set or any(len(v) != 256 for v in frames.values())):
        raise ValueError('Formal run requires full 100-train/90-test coverage; use --allow-partial-manifests only for smoke')
    return {'split': expected, 'scenes': len(frames), 'frames_with_instances': with_instances,
            'empty_gt_frames': len(empty_frames),
            'partial_manifests_allowed': bool(allow_partial)}


def validate_source_selection_checkpoint(state):
    """Keep a declared fixed test subset consistent across all loading paths.

    Legacy configurations without a subset option retain full-test semantics.
    The explicit ID sequence is inspected metadata, not a content fingerprint.
    """
    if state.get('stage') != 'pretrain':
        return
    from .instance_selection import selection_frame_ids, validate_selection_config
    config = state.get('config') or {}
    count = validate_selection_config(config.get('pretrain', {}))
    identity = state.get('identity') or {}
    selection = identity.get('selection') or {}
    data = selection.get('data_identity') or {}
    descriptor = data.get('evaluation_selection')
    policy = state.get('policy') or {}
    if count == 256:
        if (isinstance(descriptor, dict) and descriptor.get('is_subset')) or policy.get('selection_scope') == 'fixed_test_subset':
            raise ValueError('Checkpoint fixed test selection conflicts with its full-test configuration')
        return
    expected_frames = list(selection_frame_ids(count))
    ids = data.get('selection_instance_ids')
    partial = bool(identity.get('partial_manifests_allowed', False))
    scope = 'partial_manifest_smoke' if partial else 'fixed_test_subset'
    if (not isinstance(descriptor, dict) or descriptor.get('is_subset') is not True
            or descriptor.get('frames_per_scene') != count
            or descriptor.get('frame_ids') != expected_frames
            or descriptor.get('scope') != scope
            or descriptor.get('partial_manifests_allowed') != partial
            or policy.get('selection_scope') != scope
            or policy.get('selection_frames_per_scene') != count
            or policy.get('selection_frame_ids') != expected_frames
            or policy.get('selection_frame_sampler') != descriptor.get('frame_sampler')
            or not isinstance(ids, list) or len(ids) != selection.get('length')
            or descriptor.get('selected_instance_count') != selection.get('length')):
        raise ValueError('Checkpoint fixed test selection definition is inconsistent')
    if not partial and (descriptor.get('complete_requested_selection') is not True
                        or descriptor.get('scene_ids') != list(range(100, 190))
                        or descriptor.get('expected_frame_count') != 90 * count
                        or descriptor.get('selected_frame_count') != 90 * count):
        raise ValueError('Checkpoint fixed test selection has incomplete formal coverage')


def _dataset_identity(dataset):
    path = getattr(dataset, 'manifest_path', getattr(dataset, 'manifest', None))
    return {'manifest_path': str(Path(path).resolve()) if path is not None else None,
            'length': len(dataset), 'data_identity': clean_metadata(getattr(dataset, 'identity', None))}


def check_data_contract(left, right):
    """Compare known semantic preprocessing/physics, never content checksums."""
    for key in ('normal_spec', 'sampler_spec', 'precision_spec', 'quality_contract'):
        if left.get(key) != right.get(key):
            raise ValueError(f'Source preprocessing/quality differs: {key}')
    a, b = left.get('teacher_identity') or {}, right.get('teacher_identity') or {}
    for key in ('physics', 'teacher_spec', 'schema_version', 'quality_contract'):
        if a.get(key) is not None and b.get(key) is not None and a[key] != b[key]:
            raise ValueError(f'Source teacher {key} definitions differ')


def _check_available_definition(previous, current, path):
    """Require every surviving legacy field; unknown teacher provenance stays unknown."""
    if previous is None and path.rsplit('.', 1)[-1] in ('physics', 'teacher_spec'):
        return
    if isinstance(previous, dict):
        if not isinstance(current, dict):
            raise ValueError(f'Resume dataset definition differs: {path}')
        for key, value in previous.items():
            if key not in current:
                raise ValueError(f'Resume dataset definition missing: {path}.{key}')
            _check_available_definition(value, current[key], f'{path}.{key}')
    elif previous != current:
        raise ValueError(f'Resume dataset definition differs: {path}')


def check_attempt_contract(target, selection):
    """Keep task meaning and declared capture ownership aligned across manifests."""
    left, right = getattr(target, 'identity', {}) or {}, getattr(selection, 'identity', {}) or {}
    for key in ('success_definition', 'normal_spec', 'sampler_spec', 'precision_spec', 'pose_source'):
        if left.get(key) != right.get(key):
            raise ValueError(f'Target training/selection definitions differ: {key}')
    a, b = getattr(target, 'observation_owners', {}), getattr(selection, 'observation_owners', {})
    for capture in a.keys() & b.keys():
        if a[capture] != b[capture]:
            raise ValueError('Shared target observation crosses split/group declarations')


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def _atomic_torch_save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


class InstanceTrainer:
    """Legacy accumulation or explicitly configured batched execution.

    A test-selected best is labeled as such in every checkpoint and report.
    Explicit last/budget-end weights are retained independently of that ranking.
    """
    def __init__(self, model, config, source_dataset, selection_dataset, run_dir,
                 *, stage='pretrain', target_dataset=None, allow_partial=False):
        if stage not in ('pretrain', 'finetune'):
            raise ValueError('stage must be pretrain or finetune')
        if stage == 'finetune' and target_dataset is None:
            raise ValueError('Finetuning requires target training observations')
        self.model, self.config = model, clean_metadata(config)
        self.execution = resolve_execution(self.config)
        if self.execution is not None:
            self.config['execution'] = deepcopy(self.execution)
            configure_model_execution(model, self.execution)
        self._preparation_pools = {}
        self._preparation_streams = {}
        self._pending_tickets = {'source': deque(), 'target': deque()}
        self._data_wait_seconds = 0.
        architecture = getattr(model, 'architecture_metadata', None)
        b_architecture = architecture() if config['route'] == 'b' and callable(architecture) else None
        if b_architecture is not None:
            # Programmatic constructors also save their actual chart/gate, so
            # their checkpoints cannot be mistaken for historical defaults.
            options = self.config.setdefault('model', {})
            for key, actual in (('min_support_points', model.min_support_points),
                                ('feature_frame', model.feature_frame)):
                if key in options and options[key] != actual:
                    raise ValueError(f'Route B configuration differs from the constructed model: {key}')
                options[key] = actual
        self.source, self.selection, self.target = source_dataset, selection_dataset, target_dataset
        if stage == 'pretrain':
            from .instance_selection import select_source_evaluation, validate_selection_config
            self.selection = select_source_evaluation(selection_dataset,
                validate_selection_config(self.config['pretrain']), allow_partial=allow_partial)
            source_identity = getattr(self.source, 'identity', {}) or {}
            selection_identity = getattr(self.selection, 'identity', {}) or {}
            check_data_contract(source_identity, selection_identity)
        else:
            check_attempt_contract(self.target, self.selection)
        self.stage, self.run_dir = stage, Path(run_dir)
        scheduled_evaluation_steps(self.config[stage])
        checkpoint_interval(self.config[stage])
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.allow_partial = bool(allow_partial)
        self.step = 0
        self.supervised_source_instances = 0
        self.supervised_target_observations = 0
        self.skipped_no_supervision = 0
        self.best = None
        self.initialization = None
        seed = int(config.get('seed', 42))
        self.source_schedule = InstanceSchedule(self.source.records, seed)
        self.target_schedule = InstanceSchedule(self.target.records, seed+1, hierarchy=False) if self.target else None
        self.augmentation_rng = np.random.default_rng(seed+2)
        self.target_augmentation_rng = np.random.default_rng(seed+3) if self.execution is not None else None
        groups = model.parameter_groups()
        if set(groups) != {'encoder', 'decoder', 'contact', 'heads'}:
            raise ValueError('Model must expose encoder/decoder/contact/heads groups')
        all_parameters = [p for values in groups.values() for p in values]
        if len({id(p) for p in all_parameters}) != len(all_parameters):
            raise ValueError('Parameter groups overlap')
        if {id(p) for p in all_parameters} != {id(p) for p in model.parameters()}:
            raise ValueError('Parameter groups omit model parameters')
        self.groups = groups
        self.optimizer = torch.optim.AdamW(
            [{'params': values, 'name': name, 'lr': 0.} for name, values in groups.items() if values],
            betas=(.9, .999), eps=1e-8, weight_decay=.01)
        self.identity = {
            'route': config['route'], 'stage': stage, 'config_definition': clean_metadata(self.config),
            'source': _dataset_identity(self.source), 'selection': _dataset_identity(self.selection),
            'target': _dataset_identity(self.target) if self.target else None,
            'quality_contract': QUALITY_CONTRACT, 'runtime_contract': RUNTIME_CONTRACT,
            'partial_manifests_allowed': self.allow_partial,
            'torch_version': str(torch.__version__), 'numpy_version': str(np.__version__),
            'device_type': next(model.parameters()).device.type,
        }
        if b_architecture is not None:
            self.identity['model_architecture'] = clean_metadata(b_architecture)
        if self.execution is not None:
            self.identity['execution'] = deepcopy(self.execution)
        self._apply_phase()

    @property
    def policy(self):
        result = {'selection_split': 'test' if self.stage == 'pretrain' else 'val',
                'selection_role': 'test_selected' if self.stage == 'pretrain' else 'validation_selected',
                'independent_test_estimate': False,
                'train_scene_range': [0, 99],
                'evaluation_steps': list(scheduled_evaluation_steps(self.config[self.stage])),
                'selection_scope': 'partial_manifest_smoke' if self.allow_partial else 'complete_manifest',
                'public_evaluation_status': 'deferred',
                'paper_comparison_ready': False}
        if self.stage == 'pretrain':
            selection = self.selection.selection_descriptor
            if selection['is_subset']:
                result.update(selection_scope=selection['scope'],
                    selection_frames_per_scene=selection['frames_per_scene'],
                    selection_frame_ids=list(selection['frame_ids']),
                    selection_frame_sampler=selection['frame_sampler'])
            if 'successful_instance_budget' in self.config['pretrain']:
                result['successful_instance_budget'] = source_instance_budget(self.config['pretrain'])
                result['effective_instances_per_update'] = self.config['pretrain']['accumulate_instances']
                result['final_update_instances'] = (source_instance_budget(self.config['pretrain']) -
                    (self.config['pretrain']['steps'] - 1) * self.config['pretrain']['accumulate_instances'])
            if self.config['pretrain'].get('evaluate_every_instances') is not None:
                result['evaluate_every_instances'] = self.config['pretrain']['evaluate_every_instances']
                result['evaluation_instance_counts'] = [min(source_instance_budget(self.config['pretrain']),
                    step * self.config['pretrain']['accumulate_instances']) for step in result['evaluation_steps']]
        if self.execution is not None:
            result['execution'] = deepcopy(self.execution)
            result['evaluation_precision'] = 'fp32_tf32_disabled'
        return result

    def _learning_rate(self):
        cfg = self.config['pretrain']
        if 'warmup_instances' in cfg:
            next_count = self.supervised_source_instances + source_update_size(cfg, self.supervised_source_instances)
            warmup = int(cfg['warmup_instances'])
            peak, final = float(cfg['peak_lr']), float(cfg['final_lr'])
            if warmup and next_count <= warmup:
                return peak * next_count / warmup
            fraction = min(1., max(0., (next_count - warmup) / max(1, source_instance_budget(cfg) - warmup)))
            return final + .5 * (peak - final) * (1 + math.cos(math.pi * fraction))
        next_step = self.step + 1
        warmup = int(cfg['warmup_steps'])
        peak, final = float(cfg['peak_lr']), float(cfg['final_lr'])
        if warmup > 0 and next_step <= warmup:
            return peak * next_step / warmup
        fraction = min(1., max(0., (next_step-warmup) / max(1, int(cfg['steps'])-warmup)))
        return final + .5 * (peak-final) * (1 + math.cos(math.pi*fraction))

    def _apply_phase(self):
        self.model.train()
        if self.stage == 'pretrain':
            for p in self.model.parameters():
                p.requires_grad_(True)
            rates = {name: self._learning_rate() for name in self.groups}
        else:
            warm = self.step < int(self.config['finetune']['head_only_steps'])
            active = {'heads'} if warm else {'heads', 'decoder', 'contact'}
            rates = {'encoder': 0., 'decoder': 0. if warm else 1e-5,
                     'contact': 0. if warm else 1e-5, 'heads': 3e-4 if warm else 1e-4}
            for name, params in self.groups.items():
                for p in params:
                    p.requires_grad_(name in active)
            for module in self.model.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    if not module.track_running_stats:
                        raise ValueError('Finetuning BN requires tracked running statistics')
                    module.eval()
                    for p in module.parameters():
                        p.requires_grad_(False)
        for group in self.optimizer.param_groups:
            group['lr'] = rates[group['name']]
        if hasattr(self.model, 'set_training_context'):
            self.model.set_training_context(update=self.step, seed=int(self.config.get('seed', 42)))

    def _observation(self, item):
        if self.config.get('rgb_augmentation', True):
            result = dict(item)
            result['observation'] = augment_rgb(item['observation'], self.augmentation_rng)
            return result
        return item

    def _draw_batch(self, dataset, schedule, count):
        if self.execution is not None:
            stream = self._prepared_stream(dataset, schedule)
            key = 'source' if dataset is self.source else 'target'
            for _ in range(count):
                started = time.perf_counter()
                item = next(stream)
                self._data_wait_seconds += time.perf_counter() - started
                self._pending_tickets[key].popleft()
                yield item
            return
        workers = _workers('SUCTION20_INSTANCE_TRAIN_WORKERS')
        if not workers:
            for _ in range(count):
                yield self._observation(dataset[schedule.next()])
            return
        read = getattr(dataset, 'load_uncached', dataset.__getitem__)
        geometry = getattr(self.model, 'prepare_source_geometry', None) if dataset is self.source else None
        def prepare(ticket):
            index, rgb = ticket
            item = read(index)
            if rgb is not None:
                item = dict(item, observation=_apply_rgb(item['observation'], rgb))
            if geometry is not None:
                item = dict(item, observation=geometry(item['observation']))
            return item
        augment = self.config.get('rgb_augmentation', True)
        tickets = ((schedule.next(), _draw_rgb(self.augmentation_rng) if augment else None)
                   for _ in range(count))
        yield from _ordered_map(prepare, tickets, min(workers, count))

    def _prepared_stream(self, dataset, schedule):
        from .instance_loading import InstancePreparationPool
        key = 'source' if dataset is self.source else 'target'
        if key in self._preparation_streams:
            return self._preparation_streams[key]
        pending = self._pending_tickets[key]
        saved_pending = list(pending)
        rng = self.augmentation_rng if key == 'source' else self.target_augmentation_rng
        def tickets():
            # The saved sampler already issued these tickets before the last
            # checkpoint. Replay them without sampling or drawing augmentation.
            yield from saved_pending
            while True:
                rgb = _draw_rgb(rng) if self.config.get('rgb_augmentation', True) else None
                ticket = (schedule.next(), rgb)
                pending.append(ticket)
                yield ticket
        pool = InstancePreparationPool(dataset, workers=self.execution['train_workers'],
            prefetch=self.execution['prefetch_factor'],
            route=self.config['route'], geometry_options=_geometry_options(self.model))
        self._preparation_pools[key] = pool
        self._preparation_streams[key] = pool.map(tickets())
        return self._preparation_streams[key]

    def close(self):
        """Release workers while keeping pending ticket definitions resumable."""
        for pool in self._preparation_pools.values():
            pool.close()
        for stream in self._preparation_streams.values():
            stream.close()
        self._preparation_pools.clear()
        self._preparation_streams.clear()

    def _microbatch_source(self, count, weight, microbatch):
        completed = draws = 0
        limit = max(100, count * int(self.config.get('max_draw_multiplier', 20)))
        while completed < count and draws < limit:
            items = self._draw_batch(self.source, self.source_schedule,
                                     min(count - completed, limit - draws))
            while group := list(islice(items, microbatch)):
                draws += len(group)
                supported = [item for item in group if
                             len(item['observation']['support_points']) >= minimum_support_points(self.model)
                             and len(item['observation']['query_points'])]
                losses = []
                if supported:
                    outputs = self.model.forward_instances([item['observation'] for item in supported])
                    if len(outputs) != len(supported):
                        raise ValueError('Model output count differs from the requested instance batch')
                    if self.execution is not None:
                        losses = list(batched_source_losses(outputs, [item['labels'] for item in supported]).unbind())
                    else:
                        for output, item in zip(outputs, supported):
                            loss = source_loss(output, item['labels'])
                            if loss is not None:
                                losses.append(loss)
                    del outputs
                self.skipped_no_supervision += len(group) - len(losses)
                if losses:
                    # One instance-normalized loss per supervised observation.
                    (torch.stack(losses).sum() * weight).backward()
                    completed += len(losses)
                    yield from torch.stack(losses).detach().float().cpu().tolist()
        if completed < count:
            raise RuntimeError('Insufficient supervised source draws; check masks/teacher/support before training')

    def _update_items(self, dataset, schedule, count, weight, *, target):
        completed = draws = 0
        limit = max(100, count * int(self.config.get('max_draw_multiplier', 20)))
        while draws < limit and completed < count:
            # Consume this update's draws in order. Explicit execution may keep
            # lookahead tickets, saved separately at each successful checkpoint.
            for item in self._draw_batch(dataset, schedule, min(count - completed, limit - draws)):
                draws += 1
                obs = item['observation']
                if target:
                    loss = attempt_loss(self.model, item)
                elif len(obs['support_points']) >= minimum_support_points(self.model) and len(obs['query_points']):
                    loss = source_loss(self.model(obs), item['labels'])
                else:
                    loss = None
                if loss is None:
                    self.skipped_no_supervision += 1
                    continue
                (loss * weight).backward()
                completed += 1
                yield float(loss.detach())
        if completed < count:
            if target:
                raise RuntimeError('Insufficient supported historical actions; no zero-label substitution is permitted')
            raise RuntimeError('Insufficient supervised source draws; check masks/teacher/support before training')

    def _source_update_items(self, count, weight):
        if self.execution is not None:
            size = min(count, self.execution['microbatch_instances'])
            return self._microbatch_source(count, weight, size)
        size = int(os.environ.get('SUCTION20_INSTANCE_B_MICROBATCH', '1'))
        if size < 1:
            raise ValueError('SUCTION20_INSTANCE_B_MICROBATCH must be positive')
        if size > 1 and hasattr(self.model, 'forward_instances'):
            return self._microbatch_source(count, weight, size)
        return self._update_items(self.source, self.source_schedule, count, weight, target=False)

    def _target_update_items(self):
        return self._update_items(self.target, self.target_schedule, 2, .5, target=True)

    def train_one_update(self):
        with execution_math_mode(self.execution, training=True):
            return self._train_one_update()

    def _train_one_update(self):
        started = time.perf_counter()
        self._data_wait_seconds = 0.
        if self.stage == 'pretrain':
            count = source_update_size(self.config['pretrain'], self.supervised_source_instances)
            if count == 0:
                raise ValueError('The declared source instance budget is already complete')
        if self.execution is not None and self.execution['compile_mode'] == 'reduce-overhead':
            mark = getattr(getattr(torch, 'compiler', None), 'cudagraph_mark_step_begin', None)
            if callable(mark) and next(self.model.parameters()).device.type == 'cuda':
                mark()
        self._apply_phase()
        self.optimizer.zero_grad(set_to_none=True)
        if self.stage == 'pretrain':
            losses = list(self._source_update_items(count, 1. / count))
            target_losses = []
        else:
            count = 2
            target_losses = list(self._target_update_items())
            losses = list(self._source_update_items(2, .25))
        gradients = [p for p in self.model.parameters() if p.grad is not None]
        if not gradients:
            self.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError('Absent gradients: no successful step or instance budget credited')
        try:
            # The total norm performs the nonfinite check once instead of one
            # host-device synchronization per parameter. This is the PRE-clip norm.
            norm = nn.utils.clip_grad_norm_(gradients, float(self.config.get('max_grad_norm', 10.)),
                                            error_if_nonfinite=True)
        except RuntimeError as error:
            self.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError('Nonfinite gradients: no successful step or instance budget credited') from error
        self.optimizer.step()
        self.step += 1
        self.supervised_source_instances += count
        self.supervised_target_observations += 2 if self.stage == 'finetune' else 0
        record = {'step': self.step, 'source_loss': float(np.mean(losses)),
                'target_loss': float(np.mean(target_losses)) if target_losses else None,
                'grad_norm': float(norm), 'supervised_source_instances': self.supervised_source_instances,
                'supervised_target_observations': self.supervised_target_observations,
                'source_draws': self.source_schedule.draws - len(self._pending_tickets['source']),
                'target_draws': (self.target_schedule.draws - len(self._pending_tickets['target'])) if self.target_schedule else 0,
                'skipped_no_supervision': self.skipped_no_supervision,
                'learning_rates': {g['name']: g['lr'] for g in self.optimizer.param_groups}}
        if self.execution is not None:
            record['effective_source_instances_this_update'] = count
            record['source_draws_issued'] = self.source_schedule.draws
            record['pending_source_tickets'] = len(self._pending_tickets['source'])
            record['timing'] = {'update_wall_seconds': time.perf_counter() - started,
                                'data_wait_seconds': self._data_wait_seconds}
        return record

    def state_dict(self, *, kind='last'):
        names = {id(p): name for name, p in self.model.named_parameters()}
        state = {'schema': CHECKPOINT_SCHEMA, 'identity': self.identity, 'config': self.config,
                'initialization': deepcopy(self.initialization),
                'policy': self.policy, 'checkpoint_kind': kind,
                'checkpoint_selection_role': self.policy['selection_role'] if kind == 'best' else 'budget_endpoint_or_progress',
                'stage': self.stage, 'global_step': self.step, 'best': self.best,
                'supervised_source_instances': self.supervised_source_instances,
                'supervised_target_observations': self.supervised_target_observations,
                'skipped_no_supervision': self.skipped_no_supervision,
                'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(),
                'source_schedule': self.source_schedule.state_dict(),
                'target_schedule': self.target_schedule.state_dict() if self.target_schedule else None,
                'augmentation_rng': deepcopy(self.augmentation_rng.bit_generator.state), 'rng': rng_state(),
                'parameter_groups': {key: [names[id(p)] for p in values] for key, values in self.groups.items()}}
        if self.execution is not None:
            # Samplers/RNGs include issued work. Save the small unconsumed
            # ticket definitions as well, never prepared arrays or processes.
            state['preparation_state'] = {
                'schema': EXECUTION_SCHEMA,
                'pending_source': deepcopy(list(self._pending_tickets['source'])),
                'pending_target': deepcopy(list(self._pending_tickets['target'])),
                'target_augmentation_rng': deepcopy(self.target_augmentation_rng.bit_generator.state),
            }
        return state

    def save(self, name='last.pt'):
        _atomic_torch_save(self.run_dir/name, self.state_dict(kind='best' if name == 'best.pt' else 'last'))

    def resume(self, path):
        state = torch.load(path, map_location='cpu', weights_only=False)
        if state.get('schema') != CHECKPOINT_SCHEMA:
            raise ValueError('Resume checkpoint schema mismatch')
        if clean_metadata(state.get('config')) != clean_metadata(self.config):
            raise ValueError('Resume configuration differs from the saved run')
        preparation = state.get('preparation_state')
        if self.execution is not None:
            if not isinstance(preparation, dict) or preparation.get('schema') != EXECUTION_SCHEMA:
                raise ValueError('Batched resume requires its saved preparation ticket state')
            for key, dataset in (('source', self.source), ('target', self.target)):
                tickets = preparation.get('pending_' + key)
                if not isinstance(tickets, list):
                    raise ValueError('Batched resume requires a pending ticket list')
                if dataset is None and tickets:
                    raise ValueError('Pending target tickets without a target dataset')
                for ticket in tickets:
                    if (not isinstance(ticket, (list, tuple)) or len(ticket) != 2
                            or isinstance(ticket[0], bool) or not isinstance(ticket[0], (int, np.integer))
                            or not 0 <= ticket[0] < len(dataset)):
                        raise ValueError('Resume pending dataset index is invalid')
            if self.stage == 'pretrain':
                step = state.get('global_step')
                pre = self.config['pretrain']
                if (isinstance(step, bool) or not isinstance(step, int) or not 0 <= step <= pre['steps']
                        or state.get('supervised_source_instances') != min(
                            step * pre['accumulate_instances'], source_instance_budget(pre))):
                    raise ValueError('Resume successful source budget is inconsistent with its step')
        validate_source_selection_checkpoint(state)
        old_identity = state.get('identity', {})
        if 'model_architecture' in self.identity:
            if 'model_architecture' not in old_identity:
                raise ValueError('Resume checkpoint predates the route-B support/chart contract; '
                                 'start a new run with explicit compatible weight initialization')
            if clean_metadata(old_identity['model_architecture']) != self.identity['model_architecture']:
                raise ValueError('Resume model architecture differs from the saved run')
        # A fixed subset must retain its exact ordered instance definition,
        # including when reading an otherwise legacy runtime identity.
        old_selection = old_identity.get('selection') or {}
        new_selection = self.identity.get('selection') or {}
        if (any((entry.get('data_identity') or {}).get('evaluation_selection', {}).get('is_subset')
                for entry in (old_selection, new_selection)) and
                clean_metadata(old_selection) != new_selection):
            raise ValueError('Resume fixed test selection definition differs')
        for key in ('route', 'stage', 'quality_contract', 'partial_manifests_allowed',
                    'torch_version', 'numpy_version', 'device_type'):
            if old_identity.get(key) != self.identity.get(key):
                raise ValueError(f'Resume run identity mismatch: {key}')
        if self.execution is not None and old_identity.get('execution') != self.execution:
            raise ValueError('Resume execution profile differs from the saved run')
        if old_identity.get('runtime_contract') == RUNTIME_CONTRACT:
            for key in ('source', 'selection', 'target'):
                if clean_metadata(old_identity.get(key)) != self.identity.get(key):
                    raise ValueError(f'Resume dataset definition differs: {key}')
        else:
            for key in ('source', 'selection', 'target'):
                if key not in old_identity:
                    raise ValueError(f'Resume dataset definition missing: {key}')
                _check_available_definition(clean_metadata(old_identity[key]), self.identity[key], key)
        self.close()
        self.model.load_state_dict(state['model'], strict=True)
        self.optimizer.load_state_dict(state['optimizer'])
        self.step = int(state['global_step'])
        self.supervised_source_instances = int(state['supervised_source_instances'])
        self.supervised_target_observations = int(state['supervised_target_observations'])
        self.skipped_no_supervision = int(state['skipped_no_supervision'])
        self.best = clean_metadata(state['best'])
        self.initialization = clean_metadata(state.get('initialization'))
        self.source_schedule.load_state_dict(state['source_schedule'])
        if self.target_schedule:
            self.target_schedule.load_state_dict(state['target_schedule'])
        self.augmentation_rng.bit_generator.state = state['augmentation_rng']
        if self.execution is not None:
            self._pending_tickets = {key: deque(deepcopy(preparation['pending_' + key]))
                                     for key in ('source', 'target')}
            self.target_augmentation_rng.bit_generator.state = preparation['target_augmentation_rng']
        restore_rng(state['rng'])
        self._apply_phase()

    def initialize_finetune(self, path):
        from .instance_config import (semantic_model_config, model_execution_options,
                                      validate_saved_config_definition)
        if self.stage != 'finetune' or self.step:
            raise ValueError('Source initialization is only allowed before finetuning step one')
        state = torch.load(path, map_location='cpu', weights_only=False)
        if not isinstance(state, dict) or state.get('schema') != CHECKPOINT_SCHEMA:
            raise ValueError('Finetune initialization requires the same source instance architecture/quality')
        source_config, source_identity = state.get('config'), state.get('identity')
        validate_saved_config_definition(source_config, source_identity)
        validate_source_selection_checkpoint(state)
        route = self.config['route']
        if (state.get('stage') != 'pretrain' or source_config.get('schema') != 'instance_runtime_v1'
                or self.config.get('quality_contract') != QUALITY_CONTRACT
                or source_identity.get('stage') != 'pretrain'
                or source_identity.get('route') != route or source_config.get('route') != route
                or source_identity.get('quality_contract') != QUALITY_CONTRACT
                or source_config.get('quality_contract') != QUALITY_CONTRACT
                or semantic_model_config(route, source_config.get('model', {}))
                   != semantic_model_config(route, self.config.get('model', {}))):
            raise ValueError('Finetune initialization requires the same source instance architecture/quality')
        if (route == 'b' and 'model_architecture' in source_identity and
                clean_metadata(source_identity['model_architecture']) != self.identity.get('model_architecture')):
            raise ValueError('Finetune source model architecture differs from the target runtime')
        old_data = state['identity']['source'].get('data_identity') or {}
        new_data = self.identity['source'].get('data_identity') or {}
        check_data_contract(old_data, new_data)
        self.model.load_state_dict(state['model'], strict=True)
        self.initialization = {
            'kind': 'source_pretrain_checkpoint', 'checkpoint_file_record': file_record(path),
            'checkpoint_file': Path(path).name, 'checkpoint_kind': state['checkpoint_kind'],
            'checkpoint_selection_role': state['checkpoint_selection_role'],
            'global_step': int(state['global_step']), 'selection_policy': deepcopy(state['policy']),
            'source_definition': clean_metadata(state['identity']),
            'model_compatibility': {
                'semantic_model': semantic_model_config(route, self.config.get('model', {})),
                'source_execution': model_execution_options(route, source_config.get('model', {})),
                'target_execution': model_execution_options(route, self.config.get('model', {})),
            },
        }
        self._apply_phase()

    def evaluate(self):
        cfg = self.config[self.stage]
        started = time.perf_counter()
        retention = cfg.get('evaluation_records', 'best_and_final')
        if retention not in ('best_and_final', 'all', 'none'):
            raise ValueError('evaluation_records must be best_and_final, all or none')
        details = self.run_dir/f'evaluation_step{self.step:06d}.records.jsonl.gz'
        temporary = details.with_name(details.name + '.tmp')
        count = 0
        def evaluate_with_sink(sink):
            fn = evaluate_instance_source if self.stage == 'pretrain' else evaluate_instance_attempts
            return fn(self.model, self.selection, collect_records=False, record_sink=sink)[0]
        try:
            if retention == 'none':
                metrics = evaluate_with_sink(None)
            else:
                with gzip.open(temporary, 'wt', encoding='utf-8', compresslevel=1) as stream:
                    def write_record(row):
                        nonlocal count
                        stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')
                        count += 1
                    metrics = evaluate_with_sink(write_record)
            if self.stage == 'pretrain':
                key = [metrics['InstTop1_INST20_at_0_4'], metrics['selected_Q_macro'], -self.step]
            else:
                if metrics['attempt_AUROC'] is None or metrics['attempt_BCE'] is None:
                    raise ValueError('Target validation needs both classes and supported observations for AUROC selection')
                key = [metrics['attempt_AUROC'], -metrics['attempt_BCE'], -self.step]
            improved = self.best is None or tuple(key) > tuple(self.best['key'])
            final_step = self.step == int(cfg['steps'])
            same_best_step = self.best is not None and self.best['step'] == self.step
            keep = retention == 'all' or (retention == 'best_and_final' and (improved or final_step or same_best_step))
            record_info = {'policy': retention, 'retained': keep, 'count': count,
                           'path': details.name if keep else None, 'format': 'jsonl.gz'}
            if keep:
                temporary.replace(details)
            else:
                details.unlink(missing_ok=True)
            report = {'step': self.step, 'policy': self.policy, 'metrics': metrics,
                      'initialization': deepcopy(self.initialization), 'selection_key': key,
                      'identity': self.identity, 'record_details': record_info,
                      'best_updated': improved, 'final_step': final_step,
                      'timing': {'scope': 'selection_forward_metrics_and_records_excludes_checkpoint_save',
                                 'wall_seconds': time.perf_counter() - started,
                                 'observations': len(self.selection)}}
            atomic_json(self.run_dir/f'evaluation_step{self.step:06d}.json', report)
            if improved:
                self.best = {'step': self.step, 'key': key, 'metrics': metrics, 'policy': self.policy}
                self.save('best.pt')
            return report
        finally:
            temporary.unlink(missing_ok=True)

    def run(self, *, stop_after=None):
        try:
            return self._run(stop_after=stop_after)
        finally:
            self.close()

    def _run(self, *, stop_after=None):
        cfg = self.config[self.stage]
        evaluation_steps = frozenset(scheduled_evaluation_steps(cfg))
        save_every = checkpoint_interval(cfg)
        end = int(cfg['steps'])
        if stop_after is not None:
            if int(stop_after) < self.step:
                raise ValueError('stop_after precedes resumed step')
            end = min(end, int(stop_after))
        atomic_json(self.run_dir/'run_identity.json', {'identity': self.identity, 'policy': self.policy,
                    'config': self.config, 'initialization': deepcopy(self.initialization)})
        while self.step < end:
            record = self.train_one_update()
            with (self.run_dir/'training.jsonl').open('a') as stream:
                stream.write(json.dumps(record, allow_nan=False) + '\n')
            print(json.dumps(record, allow_nan=False), flush=True)
            if self.step in evaluation_steps:
                self.evaluate()
                self.save()
            elif self.step % save_every == 0:
                self.save()
        self.save()
        summary = {'completed_steps': self.step, 'planned_steps': int(cfg['steps']),
                   'completed_budget': self.step == int(cfg['steps']), 'best': self.best,
                   'policy': self.policy, 'identity': self.identity,
                   'initialization': deepcopy(self.initialization),
                   'supervised_source_instances': self.supervised_source_instances,
                   'supervised_target_observations': self.supervised_target_observations}
        atomic_json(self.run_dir/'summary.json', summary)
        return summary
