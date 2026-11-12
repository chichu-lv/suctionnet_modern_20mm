import json
import random

import numpy as np
import pytest
import torch

from suction20.training import FrameSchedule, capture_rng, learning_rate, load_config, restore_rng


def test_resume_sequence_and_augmentation_seeds():
    a = FrameSchedule([0, 1], views_per_scene=3)
    consumed = [a.next()[0] for _ in range(6)]
    assert len(set(consumed)) == 6
    saved = a.state_dict()
    expected = [a.next() for _ in range(15)]
    b = FrameSchedule([0, 1], views_per_scene=3)
    b.load_state_dict(saved)
    assert [b.next() for _ in range(15)] == expected
    with pytest.raises(ValueError):
        FrameSchedule([2], views_per_scene=3).load_state_dict(saved)


def test_lr_boundaries_and_last_step():
    assert learning_rate(0) == 1e-5
    assert learning_rate(2879) == pytest.approx(1e-3)
    assert learning_rate(2880) == pytest.approx(1e-3)
    assert learning_rate(57599) == pytest.approx(1e-5)


def test_rng_roundtrip():
    state = capture_rng()
    expected = (random.random(), np.random.rand(), torch.rand(1))
    restore_rng(state)
    assert random.random() == expected[0]
    assert np.random.rand() == expected[1]
    torch.testing.assert_close(torch.rand(1), expected[2])


def test_experiment_conflict_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("experiment: E0\nmodel:\n  quality_target: gt_B\n")
    with pytest.raises(ValueError, match="conflicting"):
        load_config(p)


@pytest.mark.parametrize('suffix', ['.json', '.JSON'])
def test_json_config_preserves_scientific_numbers_for_optimizer(tmp_path, suffix):
    path = tmp_path / ('selected-config' + suffix)
    raw = {'experiment': 'E5', 'optimizer': {'eps': 1e-8, 'lr_peak': 1e-5,
                                          'weight_decay': 1e-4, 'betas': [.9, .999]},
           'scheduler': {'lr_final': 1e-5}, 'seed': 42,
           'training': {'amp': True}, 'label': '1e-08'}
    path.write_text(json.dumps(raw))
    assert '1e-08' in path.read_text()  # Actual selected-profile serialization.
    config = load_config(path)
    assert config['optimizer'] == raw['optimizer']
    assert type(config['optimizer']['eps']) is float
    assert type(config['scheduler']['lr_final']) is float
    assert type(config['seed']) is int and config['training']['amp'] is True
    assert config['label'] == '1e-08'
    assert config['model']['use_aux_heads'] is True
    opt = config['optimizer']
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))],
                                  eps=opt['eps'], lr=opt['lr_peak'],
                                  weight_decay=opt['weight_decay'], betas=tuple(opt['betas']))
    assert optimizer.defaults['eps'] == 1e-8
    assert not optimizer.state  # Construction only; no optimizer update.


def test_yaml_config_preserves_existing_scalar_and_experiment_behavior(tmp_path):
    path = tmp_path / 'existing.yaml'
    path.write_text('experiment: E0\noptimizer:\n  eps: 1.0e-8\n'
                    '  quoted: "1e-08"\n  legacy_bare: 1e-08\ntraining:\n  amp: true\n')
    config = load_config(path)
    assert config['optimizer'] == {'eps': 1e-8, 'quoted': '1e-08', 'legacy_bare': '1e-08'}
    assert config['training']['amp'] is True
    assert config['model']['quality_target'] == 'gt_A'


def test_json_config_refuses_malformed_json_and_nonmapping_payload(tmp_path):
    path = tmp_path / 'selected.json'
    path.write_text('{"optimizer": {"eps": 1e-08},}')
    with pytest.raises(json.JSONDecodeError):
        load_config(path)
    path.write_text('[1e-08]')
    with pytest.raises(ValueError, match='mapping'):
        load_config(path)


def test_resume_allows_relocation_but_rejects_changed_losses_and_geometry():
    from copy import deepcopy
    from suction20.training import validate_resume_config
    original = {"model": {"voxel_size_m": .002, "backbone_root": "/a"},
                "loss": {"beta": .05}, "data": {"source_root": "/data"}}
    moved = deepcopy(original)
    moved["model"]["backbone_root"] = "/b"
    moved["data"]["source_root"] = "/new-data"
    validate_resume_config(original, moved)
    for section, key, value in [("model", "voxel_size_m", .003), ("loss", "beta", .1)]:
        changed = deepcopy(moved)
        changed[section][key] = value
        with pytest.raises(ValueError, match="semantics"):
            validate_resume_config(original, changed)


def test_checkpoint_resumes_optimizer_rng_and_next_frame_exactly(tmp_path):
    from suction20.training import save_checkpoint, load_checkpoint, optimizer_update
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    schedule = FrameSchedule([0], views_per_scene=3)
    config = {"model": {"quality_target": "gt_B"}, "data": {}}
    def update():
        key = schedule.next()
        loss = model(torch.randn(4, 3)).square().mean()
        loss.backward()
        assert optimizer_update(optimizer, scaler, model.parameters())[0]
        return key
    update()
    path = tmp_path / "last.pt"
    save_checkpoint(path, model, optimizer, scaler, schedule, global_step=1,
                    config=config, source_label_config={"r": .02})
    expected_frame = update()
    expected = {k: v.clone() for k, v in model.state_dict().items()}
    load_checkpoint(path, model, optimizer=optimizer, scaler=scaler, schedule=schedule,
                    resume=True, source_label_config={"r": .02})
    assert update() == expected_frame
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)


def test_nonfinite_update_does_not_change_parameters():
    from suction20.training import optimizer_update
    value = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.AdamW([value], lr=.01)
    value.grad = torch.full_like(value, float("inf"))
    updated, norm = optimizer_update(optimizer, torch.amp.GradScaler("cuda", enabled=False), [value])
    assert not updated and np.isinf(norm)
    torch.testing.assert_close(value, torch.ones(2))


def test_supervision_telemetry_exposes_mask_partition_without_changing_loss():
    from suction20.training import source_supervision_counts
    from suction20.losses import source_losses
    q=torch.tensor([[.2],[.3],[.4],[.5]],requires_grad=True)
    prediction={'q0_all':q[:,0],'q':q,'seed_point_index':torch.arange(4)[:,None],
                'seed_valid':torch.tensor([[1],[1],[0],[1]],dtype=torch.bool),
                'normal_out_all':torch.zeros(4,3,requires_grad=True),'seal_aux':q}
    packed={'inputs':{'batch_index':torch.arange(4)},'targets':{
        'instance_id_gt':torch.ones(4,dtype=torch.long),'quality_valid':torch.tensor([1,0,1,0],dtype=torch.bool),
        'gt_B':torch.zeros(4),'axis_valid':torch.tensor([0,1,0,0],dtype=torch.bool),
        'normal_label':torch.tensor([[0.,0.,1.]]*4),'seal_valid':torch.tensor([0,1,0,1],dtype=torch.bool),
        'seal_target':torch.ones(4)}}
    before=source_losses(prediction,packed['targets'],packed['inputs'])['total']
    rng=torch.get_rng_state().clone()
    counts=source_supervision_counts(prediction,packed)
    assert counts=={'q0':2,'q':1,'normal':1,'seal':2}
    after=source_losses(prediction,packed['targets'],packed['inputs'])['total']
    torch.testing.assert_close(before,after,rtol=0,atol=0)
    torch.testing.assert_close(rng,torch.get_rng_state(),rtol=0,atol=0)
