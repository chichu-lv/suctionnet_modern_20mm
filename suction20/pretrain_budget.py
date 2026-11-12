"""Standard-library-only pretraining budget resolution shared with the supervisor.

V3 credits E samples only after a successful optimizer update. Consumed frames
from AMP skips and physical replay remain separate. Legacy configs keep their
original optimizer-step schedule and sampling-epoch behavior.
"""
from __future__ import annotations

import copy

CONTRACT_VERSION = 'suction20.pretrain.samples.v3'
_SAMPLE_FIELDS = {'target_effective_samples': 230400, 'warmup_effective_samples': 11520,
                  'validation_effective_samples': 23040, 'epoch_effective_samples': 5760}


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return value


def resolve_budget(config):
    training = config.get('training', {})
    micro = _positive_int(training.get('microbatch_size', 1), 'microbatch_size')
    accumulation = _positive_int(training.get('gradient_accumulation_steps', 4), 'gradient_accumulation_steps')
    world = _positive_int(training.get('world_size', 1), 'world_size')
    if world != 1:
        raise ValueError('Pretraining budget supports exactly one GPU per experiment')
    effective = _positive_int(training.get('effective_batch_size', micro * accumulation), 'effective_batch_size')
    if effective != micro * accumulation:
        raise ValueError('Effective batch differs from microbatch × accumulation')
    marker = training.get('budget_contract')
    if marker is not None and marker != CONTRACT_VERSION:
        raise ValueError(f'Unsupported training budget contract: {marker}')
    if marker == CONTRACT_VERSION:
        if any('smoke' in key for key in config.get('data', {})):
            raise ValueError('V3 formal sample contract cannot use smoke frames')
        for name, expected in _SAMPLE_FIELDS.items():
            actual = _positive_int(training.get(name, expected), name)
            if actual != expected:
                raise ValueError(f'V3 {name} must remain {expected}')
            if actual % effective:
                raise ValueError(f'{name} must be exactly divisible by effective batch {effective}')
        if training.get('epochs', 40) != 40:
            raise ValueError('V3 formal budget is 40 credited-sample epochs')
        total, warmup, every = (230400 // effective, 11520 // effective, 23040 // effective)
        epoch = 5760
        formal = True
        version = CONTRACT_VERSION
        prefix = (400 + effective - 1) // effective
    else:
        if any(key in training for key in _SAMPLE_FIELDS):
            raise ValueError('Sample budget fields require an explicit budget_contract')
        total = _positive_int(training['max_optimizer_steps'], 'max_optimizer_steps')
        warmup = _positive_int(config.get('scheduler', {}).get('warmup_optimizer_steps', 2880), 'warmup_optimizer_steps')
        every = _positive_int(training.get('validate_every_optimizer_steps', 5760), 'validate_every_optimizer_steps')
        epoch = 90 * config.get('data', {}).get('views_per_train_scene_per_epoch', 64)
        formal = total == 57600 and not config.get('data', {}).get('smoke_frame_keys')
        version = 'suction20.pretrain.updates.legacy'
        prefix = 100
    nodes = list(range(every, total + 1, every))
    if not nodes or nodes[-1] != total:
        nodes.append(total)
    return {'formal': formal, 'contract_version': version, 'target_updates': total,
            'warmup_updates': warmup, 'validation_steps': nodes,
            'validation_interval_updates': every, 'prefix_step': prefix,
            'checkpoint_interval_updates': prefix, 'effective_batch_size': effective,
            'target_samples': total * effective, 'epoch_samples': epoch}


def resolve_config(config):
    """Return a resolved copy; explicit conflicting v3 step fields are errors."""
    resolved = copy.deepcopy(config)
    budget = resolve_budget(resolved)
    if budget['contract_version'] != CONTRACT_VERSION:
        return resolved
    for section, key, value in (
        ('training', 'max_optimizer_steps', budget['target_updates']),
        ('training', 'validate_every_optimizer_steps', budget['validation_interval_updates']),
        ('scheduler', 'warmup_optimizer_steps', budget['warmup_updates']),
        ('scheduler', 'total_optimizer_steps', budget['target_updates']),
    ):
        values = resolved.setdefault(section, {})
        if key in values and (type(values[key]) is not int or values[key] != value):
            raise ValueError(f'Explicit {section}.{key} conflicts with sample budget')
        values[key] = value
    resolved['training'].setdefault('effective_batch_size', budget['effective_batch_size'])
    for key, value in _SAMPLE_FIELDS.items():
        resolved['training'].setdefault(key, value)
    return resolved


def sampling_epoch(budget, global_step, consumed_schedule_epoch):
    """AMP skips do not advance v3's label-guided sampling regime."""
    if budget['contract_version'] == CONTRACT_VERSION:
        return global_step * budget['effective_batch_size'] // budget['epoch_samples']
    return consumed_schedule_epoch


def validate_resume_progress(state, budget):
    """Cross-check v3 counters and validation transactions before continuing."""
    if budget['contract_version'] != CONTRACT_VERSION:
        return
    runtime = state['runtime_state']
    if runtime.get('budget') != budget:
        raise ValueError('Resumed sample budget differs from resolved budget')
    step = state['global_step']
    if type(step) is not int or not 0 <= step <= budget['target_updates']:
        raise ValueError('Invalid resumed successful update count')
    effective = budget['effective_batch_size']
    if runtime.get('effective_samples_completed') != step * effective:
        raise ValueError('Resumed credited sample count differs from successful updates')
    ledger = runtime['ledger']
    for key in ('attempts', 'skipped', 'frames'):
        if type(ledger.get(key)) is not int or ledger[key] < 0:
            raise ValueError(f'Invalid resumed ledger {key}')
    if ledger['attempts'] - ledger['skipped'] != step or ledger['frames'] != ledger['attempts'] * effective:
        raise ValueError('Resumed attempt/skip/frame counters disagree with successful updates')
    pending = runtime.get('pending_validation')
    if pending is not None and (pending.get('step') != step or step not in budget['validation_steps']):
        raise ValueError('Resumed pending validation is not at the current prescribed node')
    completed_steps = [node for node in budget['validation_steps'] if node < step or (node == step and pending is None)]
    refs = runtime.get('validations', {})
    if set(refs) != {str(node) for node in completed_steps}:
        raise ValueError('Resumed validation references do not match completed budget nodes')
    expected = [{'step': node, 'effective_samples': node * effective, 'frame_count': 320} for node in completed_steps]
    if runtime.get('completed_validation_nodes') != expected:
        raise ValueError('Resumed completed validation nodes disagree with validation references')
    last = runtime.get('last_completed_validation')
    if expected:
        if not isinstance(last, dict) or any(last.get(key) != value for key, value in expected[-1].items()) or last.get('result') != refs[str(completed_steps[-1])]:
            raise ValueError('Resumed last completed validation is inconsistent')
        best = state.get('best_metric')
        if not best or best.get('global_step') not in completed_steps:
            raise ValueError('Resumed best checkpoint is not a completed validation node')
    elif last is not None or state.get('best_metric') is not None:
        raise ValueError('Resumed validation/best state precedes the first validation node')
