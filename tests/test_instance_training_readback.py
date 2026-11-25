"""Metric readback boundaries with real fixture losses, backward and AdamW.

CPU spies verify call placement and tensor ownership, not CUDA throughput.
"""
import numpy as np
import pytest
import torch

import suction20.instance_training as training
from test_instance_execution import TinyBatchedModel, execution_config
from test_instance_training import MemoryAttempts, MemorySource, config


@pytest.fixture(scope='module', autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_trainer(path, *, stage='pretrain', batched=True):
    cfg = execution_config(batch=5, total=7) if batched else config()
    if batched:
        cfg['execution']['microbatch_instances'] = 3
    cfg['max_grad_norm'] = .01
    return training.InstanceTrainer(TinyBatchedModel(), cfg, MemorySource(),
        MemorySource('test') if stage == 'pretrain' else MemoryAttempts('val'), path,
        stage=stage, target_dataset=MemoryAttempts() if stage == 'finetune' else None,
        allow_partial=True)


def observe_update(trainer, monkeypatch):
    """Observe production operations while retaining their implementations."""
    observed = {'source': [], 'target': [], 'source_weights': [], 'target_weights': [],
                'events': [], 'transfers': [], 'norms': []}

    def observe_loss(original, category):
        def call(*args, **kwargs):
            loss = original(*args, **kwargs)
            if loss is not None:
                observed[category].append(loss.detach().clone().reshape(-1))
                loss.register_hook(lambda gradient: observed[category + '_weights'].append(
                    gradient.detach().clone().reshape(-1)))
            return loss
        return call

    for name, category in (('source_loss', 'source'), ('batched_source_losses', 'source'),
                           ('attempt_loss', 'target')):
        monkeypatch.setattr(training, name, observe_loss(getattr(training, name), category))

    original_step = trainer.optimizer.step
    def step(*args, **kwargs):
        result = original_step(*args, **kwargs)
        observed['events'].append('optimizer_step')
        return result
    monkeypatch.setattr(trainer.optimizer, 'step', step)

    original_clip = torch.nn.utils.clip_grad_norm_
    def clip(*args, **kwargs):
        assert kwargs.get('error_if_nonfinite') is True
        norm = original_clip(*args, **kwargs)
        observed['norms'].append(norm.detach().clone())
        return norm
    monkeypatch.setattr(torch.nn.utils, 'clip_grad_norm_', clip)

    original_cpu = torch.Tensor.cpu
    def cpu(value, *args, **kwargs):
        observed['events'].append('metrics_cpu')
        observed['transfers'].append((value.detach().clone(), value.requires_grad, value.grad_fn))
        return original_cpu(value, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, 'cpu', cpu)
    return observed


def assert_metrics(record, observed, *, source_count, target_count, source_weight):
    assert observed['events'] == ['optimizer_step', 'metrics_cpu']
    assert len(observed['transfers']) == len(observed['norms']) == 1
    transferred, requires_grad, grad_fn = observed['transfers'][0]
    assert not requires_grad and grad_fn is None
    assert transferred.shape == (source_count + target_count + 1,)
    source = torch.cat(observed['source']).tolist()
    target = torch.cat(observed['target']).tolist() if target_count else []
    # The original implementation converted every instance value to a Python
    # float, then used NumPy's float64 mean in this same consumption order.
    assert len(source) == source_count and len(target) == target_count
    assert record['source_loss'] == float(np.mean(source))
    assert record['target_loss'] == (float(np.mean(target)) if target else None)
    assert transferred.tolist() == [*source, *target, observed['norms'][0].item()]
    assert record['grad_norm'] == observed['norms'][0].item()
    assert record['grad_norm'] > .01  # This is the PRE-clip norm, not the threshold.
    for weights in observed['source_weights']:
        torch.testing.assert_close(weights, torch.full_like(weights, source_weight), rtol=0, atol=0)
    for weights in observed['target_weights']:
        torch.testing.assert_close(weights, torch.full_like(weights, .5), rtol=0, atol=0)


def test_pretrain_readback_preserves_multimicrobatch_and_partial_budget_metrics(tmp_path, monkeypatch):
    torch.manual_seed(102)
    trainer = make_trainer(tmp_path)
    try:
        for step, count in enumerate((5, 2), start=1):
            with monkeypatch.context() as patch:
                observed = observe_update(trainer, patch)
                record = trainer.train_one_update()
            assert_metrics(record, observed, source_count=count, target_count=0, source_weight=1. / count)
            assert record['step'] == step
            assert record['effective_source_instances_this_update'] == count
            assert record['supervised_source_instances'] == (5 if step == 1 else 7)
            assert record['supervised_target_observations'] == 0
        assert trainer.model.batch_sizes == [3, 2, 2]
        assert trainer.step == 2 and trainer.supervised_source_instances == 7
        with pytest.raises(ValueError, match='already complete'):
            trainer.train_one_update()
    finally:
        trainer.close()


@pytest.mark.parametrize('batched', [False, True])
def test_finetune_readback_preserves_target_source_weights_and_means(tmp_path, monkeypatch, batched):
    torch.manual_seed(103)
    trainer = make_trainer(tmp_path, stage='finetune', batched=batched)
    try:
        with monkeypatch.context() as patch:
            observed = observe_update(trainer, patch)
            record = trainer.train_one_update()
        assert_metrics(record, observed, source_count=2, target_count=2, source_weight=.25)
        assert record['step'] == 1
        assert record['supervised_source_instances'] == record['supervised_target_observations'] == 2
        assert trainer.model.batch_sizes == ([2] if batched else [])
        assert trainer.optimizer.state  # AdamW actually updated the active heads.
    finally:
        trainer.close()


def test_nonfinite_gradients_still_reject_update_before_metrics_and_budget_credit(tmp_path, monkeypatch):
    torch.manual_seed(104)
    trainer = make_trainer(tmp_path)
    handle = trainer.model.heads.weight.register_hook(lambda gradient: torch.full_like(gradient, float('nan')))
    try:
        with monkeypatch.context() as patch:
            observed = observe_update(trainer, patch)
            with pytest.raises(FloatingPointError, match='Nonfinite gradients'):
                trainer.train_one_update()
        assert observed['events'] == [] and observed['transfers'] == []
        assert trainer.step == trainer.supervised_source_instances == trainer.supervised_target_observations == 0
        assert trainer.optimizer.state == {}
        assert all(parameter.grad is None for parameter in trainer.model.parameters())
    finally:
        handle.remove()
        trainer.close()
