"""Weight initialization permits execution changes while preserving semantics."""
from copy import deepcopy
import json

import pytest
import torch

from suction20.instance_config import (apply_model_runtime_overrides,
    B_FEATURE_FRAME, B_LEGACY_FEATURE_FRAME, model_execution_options, semantic_model_config)
from suction20.instance_inference import load_instance_checkpoint
import suction20.instance_inference as inference
from suction20.instance_training import InstanceTrainer, load_instance_config
from test_instance_training import (MemorySource, MemoryAttempts, TinyInstanceModel,
                                    config as engine_config, assert_nested_equal)
from test_instance_inference import FixedScorer, checkpoint_state


def _trainer(path, route, options, *, stage='pretrain'):
    config = engine_config()
    config.update(route=route, model=deepcopy(options))
    return InstanceTrainer(TinyInstanceModel(), config, MemorySource(),
        MemorySource('test') if stage == 'pretrain' else MemoryAttempts('val'), path,
        stage=stage, target_dataset=MemoryAttempts() if stage == 'finetune' else None,
        allow_partial=True)


def _source(path, route, options):
    trainer = _trainer(path, route, options)
    trainer.train_one_update()
    trainer.save('best.pt')
    return trainer, path/'best.pt'


@pytest.mark.parametrize('route,source_options,target_options', [
    ('a', {'backbone_root': '/source/backbone', 'query_chunk_size': 32},
          {'backbone_root': '/target/backbone', 'query_chunk_size': 7}),
    ('b', {'query_chunk': 64}, {'query_chunk': 7}),
])
def test_finetune_allows_relocation_and_query_batching_without_changing_weights(
        tmp_path, route, source_options, target_options):
    source, checkpoint = _source(tmp_path/'source', route, source_options)
    target = _trainer(tmp_path/'target', route, target_options, stage='finetune')
    target.initialize_finetune(checkpoint)
    assert_nested_equal(target.model.state_dict(), source.model.state_dict())
    provenance = target.initialization['model_compatibility']
    assert provenance['source_execution'] == model_execution_options(route, source_options)
    assert provenance['target_execution'] == model_execution_options(route, target_options)
    assert provenance['semantic_model'] == semantic_model_config(route, source_options)
    assert target.config['model'] == target_options
    target.train_one_update()
    assert target.step == 1


@pytest.mark.parametrize('route,field,value', [
    ('a', 'backbone_points', 1024), ('a', 'min_support_points', 50),
    ('a', 'backbone_coordinate_fastpath', True), ('a', 'singleton_bn_policy', 'error'),
    ('b', 'phase_seed', 5), ('b', 'training_phase', False),
    ('b', 'min_support_points', 101), ('b', 'feature_frame', B_FEATURE_FRAME),
    ('a', 'future_model_parameter', 17),
])
def test_finetune_still_rejects_model_semantic_changes(tmp_path, route, field, value):
    _, checkpoint = _source(tmp_path/'source', route, {})
    target = _trainer(tmp_path/'target', route, {field: value}, stage='finetune')
    with pytest.raises(ValueError, match='same source instance architecture/quality'):
        target.initialize_finetune(checkpoint)


def test_omitted_and_explicit_constructor_defaults_are_semantically_equal():
    assert semantic_model_config('a', {}) == semantic_model_config('a', {
        'backbone_points': 2048, 'min_support_points': 100,
        'backbone_coordinate_fastpath': False, 'singleton_bn_policy': 'running_stats'})
    assert semantic_model_config('b', {}) == semantic_model_config('b', {
        'phase_seed': 42, 'training_phase': True, 'min_support_points': 100,
        'feature_frame': B_LEGACY_FEATURE_FRAME})


def test_new_b_run_and_historical_b_checkpoint_resolve_different_explicit_charts(tmp_path):
    config = engine_config()
    path = tmp_path/'config.json'
    path.write_text(json.dumps(config))
    fresh = load_instance_config(path)
    assert fresh['model']['feature_frame'] == B_FEATURE_FRAME
    assert fresh['model']['min_support_points'] == 100
    historical = apply_model_runtime_overrides(config, query_chunk_size=7)
    assert historical['model']['feature_frame'] == B_LEGACY_FEATURE_FRAME
    assert historical['model']['min_support_points'] == 100
    assert historical['model']['query_chunk'] == 7
    assert config['model'] == {}  # Loading must not rewrite saved provenance.
    assert apply_model_runtime_overrides(fresh)['model']['feature_frame'] == B_FEATURE_FRAME
    config['model'] = {'feature_frame': B_LEGACY_FEATURE_FRAME, 'min_support_points': 101}
    path.write_text(json.dumps(config))
    assert load_instance_config(path)['model'] == config['model']


class MetadataBModel(TinyInstanceModel):
    """Small engine fixture for the model-identity checkpoint contract."""
    def __init__(self, *, feature_frame=B_FEATURE_FRAME, min_support_points=100):
        super().__init__()
        self.feature_frame = feature_frame
        self.min_support_points = min_support_points

    def architecture_metadata(self):
        return {'architecture_id': 'test_fixture_not_production',
                'feature_frame': self.feature_frame,
                'minimum_support_points': self.min_support_points}


def _metadata_trainer(path, *, options=None, model=None, stage='pretrain'):
    config = engine_config()
    config['model'] = deepcopy(options or {})
    return InstanceTrainer(model or MetadataBModel(), config, MemorySource(),
        MemorySource('test') if stage == 'pretrain' else MemoryAttempts('val'), path,
        stage=stage, target_dataset=MemoryAttempts() if stage == 'finetune' else None,
        allow_partial=True)


def test_b_trainer_records_actual_constructor_semantics_and_resumes(tmp_path):
    source = _metadata_trainer(tmp_path/'source')
    assert source.config['model'] == {'feature_frame': B_FEATURE_FRAME, 'min_support_points': 100}
    assert source.identity['config_definition'] == source.config
    assert source.identity['model_architecture'] == source.model.architecture_metadata()
    source.train_one_update()
    source.save()
    resumed = _metadata_trainer(tmp_path/'resumed')
    resumed.resume(tmp_path/'source/last.pt')
    assert resumed.step == source.step
    assert_nested_equal(resumed.model.state_dict(), source.model.state_dict())
    # Resume applies the next update's scheduled LR; saved Adam moments persist.
    assert_nested_equal(resumed.optimizer.state_dict()['state'], source.optimizer.state_dict()['state'])


@pytest.mark.parametrize('mutation,match', [('missing', 'predates the route-B'),
                                          ('changed', 'model architecture differs')])
def test_b_resume_rejects_missing_or_changed_behavior_contract(tmp_path, mutation, match):
    source = _metadata_trainer(tmp_path/'source')
    source.train_one_update()
    state = source.state_dict()
    if mutation == 'missing':
        state['identity'].pop('model_architecture')
    else:
        state['identity']['model_architecture']['feature_frame'] = B_LEGACY_FEATURE_FRAME
    checkpoint = tmp_path/'checkpoint.pt'
    torch.save(state, checkpoint)
    with pytest.raises(ValueError, match=match):
        _metadata_trainer(tmp_path/'resumed').resume(checkpoint)


@pytest.mark.parametrize('options', [{'feature_frame': B_LEGACY_FEATURE_FRAME},
                                    {'min_support_points': 99}])
def test_b_trainer_rejects_configuration_that_misdescribes_constructed_model(tmp_path, options):
    with pytest.raises(ValueError, match='configuration differs from the constructed model'):
        _metadata_trainer(tmp_path, options=options)


@pytest.mark.parametrize('field,value', [('feature_frame', B_LEGACY_FEATURE_FRAME),
                                       ('minimum_support_points', 101)])
def test_b_finetune_checks_existing_architecture_identity(tmp_path, field, value):
    source = _metadata_trainer(tmp_path/'source')
    source.train_one_update()
    source.save('best.pt')
    checkpoint = tmp_path/'source/best.pt'
    target = _metadata_trainer(tmp_path/'target', stage='finetune')
    target.initialize_finetune(checkpoint)
    assert_nested_equal(target.model.state_dict(), source.model.state_dict())
    state = torch.load(checkpoint, weights_only=False)
    state['identity']['model_architecture'][field] = value
    torch.save(state, checkpoint)
    with pytest.raises(ValueError, match='source model architecture differs'):
        target.initialize_finetune(checkpoint)


def test_historical_b_weights_can_initialize_explicit_legacy_new_run(tmp_path):
    source, checkpoint = _source(tmp_path/'source', 'b', {})
    target = _metadata_trainer(tmp_path/'target', stage='finetune',
                               model=MetadataBModel(feature_frame=B_LEGACY_FEATURE_FRAME))
    target.initialize_finetune(checkpoint)
    assert target.config['model']['feature_frame'] == B_LEGACY_FEATURE_FRAME
    assert target.config['model']['min_support_points'] == 100
    assert_nested_equal(target.model.state_dict(), source.model.state_dict())


def test_b_inference_rejects_inconsistent_declared_architecture(tmp_path, monkeypatch):
    class MetadataScorer(FixedScorer):
        def architecture_metadata(self):
            return {'architecture_id': 'test_fixture', 'feature_frame': B_FEATURE_FRAME}
    model = MetadataScorer()
    state = checkpoint_state(model, route='b')
    state['config']['model'] = {'feature_frame': B_FEATURE_FRAME}
    state['identity']['config_definition'] = deepcopy(state['config'])
    state['identity']['model_architecture'] = model.architecture_metadata()
    checkpoint = tmp_path/'model.pt'
    torch.save(state, checkpoint)
    monkeypatch.setattr(inference, 'create_instance_model', lambda *a, **k: MetadataScorer())
    _, report = load_instance_checkpoint(checkpoint)
    assert report['architecture'] == model.architecture_metadata()
    state['identity']['model_architecture']['feature_frame'] = B_LEGACY_FEATURE_FRAME
    torch.save(state, checkpoint)
    with pytest.raises(ValueError, match='model architecture differs'):
        load_instance_checkpoint(checkpoint)


@pytest.mark.parametrize('mutation,match', [
    ('contradictory_execution', 'config definition'),
    ('quality', 'architecture/quality'),
    ('weights', 'Unexpected key'),
])
def test_finetune_checks_saved_definition_quality_and_strict_weights(tmp_path, mutation, match):
    _, checkpoint = _source(tmp_path/'source', 'a', {'backbone_root': '/original'})
    state = torch.load(checkpoint, weights_only=False)
    if mutation == 'contradictory_execution':
        state['config']['model']['backbone_root'] = '/changed-without-definition'
    elif mutation == 'quality':
        state['config']['quality_contract'] = 'different-quality'
        state['identity']['quality_contract'] = 'different-quality'
        state['identity']['config_definition'] = deepcopy(state['config'])
    else:
        state['model']['unexpected_weight'] = torch.ones(1)
    torch.save(state, checkpoint)
    target = _trainer(tmp_path/'target', 'a', {'backbone_root': '/relocated'}, stage='finetune')
    with pytest.raises((ValueError, RuntimeError), match=match):
        target.initialize_finetune(checkpoint)


def test_same_run_resume_stays_strict_about_execution_configuration(tmp_path):
    _, checkpoint = _source(tmp_path/'source', 'a', {'backbone_root': '/source', 'query_chunk_size': 32})
    resumed = _trainer(tmp_path/'resumed', 'a', {'backbone_root': '/source', 'query_chunk_size': 7})
    with pytest.raises(ValueError, match='Resume configuration differs'):
        resumed.resume(checkpoint)


@pytest.mark.parametrize('route,key', [('a', 'query_chunk_size'), ('b', 'query_chunk')])
def test_inference_overrides_follow_saved_definition_validation(tmp_path, monkeypatch, route, key):
    state = checkpoint_state(FixedScorer(), route=route)
    state['config']['model'] = {key: 64}
    if route == 'a':
        state['config']['model']['backbone_root'] = '/source/backbone'
    state['identity']['config_definition'] = deepcopy(state['config'])
    checkpoint = tmp_path/'best.pt'
    torch.save(state, checkpoint)
    calls = []
    def factory(config, device='cpu'):
        calls.append(deepcopy(config))
        return FixedScorer()
    monkeypatch.setattr(inference, 'create_instance_model', factory)
    kwargs = {'query_chunk_size': 9}
    if route == 'a':
        kwargs['backbone_root'] = tmp_path/'backbone'
    _, metadata = load_instance_checkpoint(checkpoint, **kwargs)
    assert calls[0]['model'][key] == 9
    assert metadata['model_compatibility']['checkpoint_execution'][key] == 64
    assert metadata['model_compatibility']['runtime_execution'][key] == 9
    assert metadata['checkpoint_identity']['config_definition'] == state['config']
    assert torch.load(checkpoint, weights_only=False)['config'] == state['config']
    if route == 'a':
        assert calls[0]['model']['backbone_root'] == str((tmp_path/'backbone').resolve())
    state['config']['model'][key] = 17
    torch.save(state, checkpoint)
    with pytest.raises(ValueError, match='config definition'):
        load_instance_checkpoint(checkpoint, **kwargs)
    assert len(calls) == 1  # A bad saved definition never reaches construction.


@pytest.mark.parametrize('value', [0, -1, True, 2.5])
def test_runtime_chunk_override_requires_a_positive_integer(value):
    with pytest.raises(ValueError, match='positive integer'):
        apply_model_runtime_overrides({'route': 'a', 'model': {}}, query_chunk_size=value)


def test_runtime_override_has_no_implicit_backbone_for_b(tmp_path):
    with pytest.raises(ValueError, match='only valid for route A'):
        apply_model_runtime_overrides({'route': 'b', 'model': {}}, backbone_root=tmp_path)


@pytest.mark.parametrize('route,key', [('a', 'query_chunk_size'), ('b', 'query_chunk')])
def test_training_cli_records_the_resolved_chunk_and_backbone(tmp_path, monkeypatch, route, key):
    from scripts import train_instance as cli
    config = engine_config(); config.update(route=route, model={})
    path = tmp_path/'config.json'; path.write_text(json.dumps(config))
    source = MemorySource()
    calls = []
    monkeypatch.setattr(cli, 'InstanceDataset',
                        lambda *a, **k: source if k.get('split') == 'train' else MemorySource('test'))
    monkeypatch.setattr(cli, 'check_source_splits', lambda *a, **k: {})
    monkeypatch.setattr(cli, 'create_instance_model', lambda *a, **k: TinyInstanceModel())
    class Trainer:
        def __init__(self, model, config, *args, **kwargs):
            calls.append(deepcopy(config))
        def run(self, *, stop_after):
            pass
    monkeypatch.setattr(cli, 'InstanceTrainer', Trainer)
    args = ['--config', str(path), '--source-manifest', 'train.jsonl',
            '--selection-manifest', 'test.jsonl', '--run-dir', str(tmp_path/'run'),
            '--device', 'cpu', '--query-chunk-size', '7', '--allow-partial-manifests']
    if route == 'a':
        args += ['--backbone-root', str(tmp_path/'backbone')]
    cli.main(args)
    assert calls[0]['model'][key] == 7
    if route == 'a':
        assert calls[0]['model']['backbone_root'] == str((tmp_path/'backbone').resolve())


def test_training_cli_reports_missing_a_backbone_before_reading_datasets(tmp_path, monkeypatch):
    from scripts import train_instance as cli
    config = engine_config(); config.update(route='a', model={'backbone_root': None})
    path = tmp_path/'config.json'; path.write_text(json.dumps(config))
    monkeypatch.setattr(cli, 'InstanceDataset', lambda *a, **k: pytest.fail('Read data before missing backbone error'))
    with pytest.raises(SystemExit, match='2'):
        cli.main(['--config', str(path), '--source-manifest', 'train.jsonl',
                  '--selection-manifest', 'test.jsonl', '--run-dir', str(tmp_path/'run')])


@pytest.mark.parametrize('operation', ['infer', 'evaluate'])
def test_inference_and_evaluation_cli_forward_only_explicit_chunk_override(tmp_path, monkeypatch, operation):
    from scripts import infer_instance, evaluate_instance
    cli = infer_instance if operation == 'infer' else evaluate_instance
    calls = []
    def stop_at_loader(*args, **kwargs):
        calls.append(kwargs)
        raise RuntimeError('checkpoint loader reached')
    monkeypatch.setattr(cli, 'load_instance_checkpoint', stop_at_loader)
    args = ['--checkpoint', 'own.pt', '--output', str(tmp_path/'output.json'), '--device', 'cpu']
    args += ['--observation', 'obs.npz'] if operation == 'infer' else ['--manifest', 'manifest.jsonl', '--domain', 'source']
    with pytest.raises(RuntimeError, match='checkpoint loader reached'):
        cli.main(args)
    assert 'query_chunk_size' not in calls[-1]
    with pytest.raises(RuntimeError, match='checkpoint loader reached'):
        cli.main(args+['--query-chunk-size', '11'])
    assert calls[-1]['query_chunk_size'] == 11
