"""Explicit from-scratch batching and sample-budget policy.

Execution settings contain inspectable values, never asset content digests.
Omitting this section retains the historical single-instance execution path.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import math

import torch


EXECUTION_SCHEMA = 'instance_batched_execution_v1'
DEFAULT_EXECUTION = {
    'schema': EXECUTION_SCHEMA,
    'microbatch_instances': 1,
    'precision': 'fp32',
    'dense_query_chunk': None,
    'train_workers': 0,
    'evaluation_workers': 0,
    'prefetch_factor': 2,
    'evaluation_microbatch_instances': 1,
    'compile_mode': None,
    'allow_tf32': False,
    'main_threads': 1,
}


def positive_int(value, name, *, allow_zero=False):
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def source_instance_budget(pretrain):
    batch = positive_int(pretrain['accumulate_instances'], 'accumulate_instances')
    steps = positive_int(pretrain['steps'], 'steps')
    explicit = pretrain.get('successful_instance_budget')
    if explicit is None:
        return batch * steps
    total = positive_int(explicit, 'successful_instance_budget')
    if math.ceil(total / batch) != steps:
        raise ValueError('steps must equal ceil(successful_instance_budget / accumulate_instances)')
    return total


def source_update_size(pretrain, completed_instances):
    total = source_instance_budget(pretrain)
    if completed_instances < 0 or completed_instances > total:
        raise ValueError('Completed source instance count is outside the declared budget')
    return min(int(pretrain['accumulate_instances']), total - completed_instances)


def sample_evaluation_steps(pretrain):
    """Evaluate at first optimizer boundary reaching each sample milestone."""
    every = positive_int(pretrain['evaluate_every_instances'], 'evaluate_every_instances')
    if pretrain.get('evaluate_every') is not None or pretrain.get('evaluation_steps') is not None:
        raise ValueError('Choose sample evaluation milestones or an update schedule, not both')
    if 'successful_instance_budget' not in pretrain:
        raise ValueError('Sample evaluation milestones require successful_instance_budget')
    total = source_instance_budget(pretrain)
    batch = int(pretrain['accumulate_instances'])
    samples = list(range(every, total, every)) + [total]
    return tuple(sorted(set(math.ceil(value / batch) for value in samples)))


def resolve_execution(config):
    raw = config.get('execution')
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError('execution must be a mapping')
    unknown = set(raw) - set(DEFAULT_EXECUTION)
    if unknown:
        raise ValueError(f'Unknown execution fields: {sorted(unknown)}')
    result = deepcopy(DEFAULT_EXECUTION)
    result.update(raw)
    if result['schema'] != EXECUTION_SCHEMA:
        raise ValueError('Unsupported instance execution schema')
    for key in ('microbatch_instances', 'evaluation_microbatch_instances', 'prefetch_factor', 'main_threads'):
        positive_int(result[key], key)
    for key in ('train_workers', 'evaluation_workers'):
        positive_int(result[key], key, allow_zero=True)
    if result['dense_query_chunk'] is not None:
        positive_int(result['dense_query_chunk'], 'dense_query_chunk')
    if result['precision'] not in ('fp32', 'bf16_dense'):
        raise ValueError('execution.precision must be fp32 or bf16_dense')
    if result['compile_mode'] not in (None, 'default', 'reduce-overhead', 'max-autotune-no-cudagraphs'):
        raise ValueError('Unsupported execution.compile_mode')
    if not isinstance(result['allow_tf32'], bool):
        raise ValueError('allow_tf32 must be boolean')
    pretrain = config['pretrain']
    source_instance_budget(pretrain)
    if result['microbatch_instances'] > pretrain['accumulate_instances']:
        raise ValueError('microbatch_instances cannot exceed the effective instance batch')
    if 'warmup_instances' in pretrain:
        positive_int(pretrain['warmup_instances'], 'warmup_instances', allow_zero=True)
        if pretrain.get('warmup_steps') is not None:
            raise ValueError('Choose warmup_instances or warmup_steps, not both')
        if pretrain['warmup_instances'] >= source_instance_budget(pretrain):
            raise ValueError('warmup_instances must be smaller than the source budget')
    return result


def configure_model_execution(model, execution):
    if execution is None:
        return
    if getattr(model, '_instance_execution', None) == execution:
        return
    configure = getattr(model, 'configure_execution', None)
    if not callable(configure):
        raise ValueError('This model does not implement explicit batched execution')
    if (execution['microbatch_instances'] > 1 or execution['evaluation_microbatch_instances'] > 1) and not callable(
            getattr(model, 'forward_instances', None)):
        raise ValueError('True batching requires model.forward_instances; no hidden serial fallback')
    parameter = next(model.parameters())
    if parameter.device.type == 'cuda' and execution['precision'] == 'bf16_dense' and not torch.cuda.is_bf16_supported():
        raise ValueError('bf16_dense requires a CUDA device with BF16 support')
    torch.set_num_threads(execution['main_threads'])
    configure(precision=execution['precision'], compile_mode=execution['compile_mode'],
              dense_query_chunk=execution['dense_query_chunk'])
    model._instance_execution = deepcopy(execution)


@contextmanager
def execution_math_mode(execution, *, training):
    """Scope TF32 to explicitly opted-in training; evaluation remains FP32."""
    if execution is None:
        yield
        return
    previous = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    try:
        enabled = bool(training and execution['allow_tf32'])
        torch.backends.cuda.matmul.allow_tf32 = enabled
        torch.backends.cudnn.allow_tf32 = enabled
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous
