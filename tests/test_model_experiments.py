import pytest
import torch

from models import Suction20Net
from test_model import TinyTestBackbone, model_inputs
from test_contact_grouping import FROZEN


@pytest.mark.parametrize('experiment,instance,contact,aux', [
    ('E0', False, False, False), ('E1', False, False, False),
    ('E2', True, False, False), ('E3', False, True, False),
    ('E4', True, True, False), ('E5', True, True, True)])
def test_full_experiment_matrix_shapes_and_gradient_boundary(experiment, instance, contact, aux):
    torch.manual_seed(42)
    model = Suction20Net({'experiment': experiment, 'seed_count': 6, 'dense_chunk_size': 5,
                         'pose_chunk_size': 2, 'use_task_success_head': True}, FROZEN, TinyTestBackbone())
    inputs = model_inputs()
    output = model(inputs)
    assert output['pose_features'].shape == (2, 6, 256)
    assert output['q'].shape == (2, 6)
    assert model.quality_target == ('gt_A' if experiment == 'E0' else 'gt_B')
    assert (model.instance_module is not None) == instance
    assert (model.contact_grouping is not None) == contact
    assert hasattr(model, 'seal_head') == aux
    quality_keys = ['q', 'task_success_logit'] + (['seal_aux', 'wrench_aux', 'collision_free_logit'] if aux else [])
    sum(output[key].sum() for key in quality_keys).backward()
    assert all(p.grad is None for p in model.axis_head.parameters())
    assert model.backbone.linear.weight.grad.abs().sum() > 0
    groups = model.parameter_groups()
    all_params = [p for params in groups.values() for p in params]
    assert len(all_params) == len({id(p) for p in all_params}) == len(list(model.parameters()))
    decoder_ids = {id(p) for p in groups['decoder']}
    for module in (model.instance_module, model.contact_grouping):
        if module is not None:
            assert {id(p) for p in module.parameters()} <= decoder_ids
    model.eval()
    with torch.no_grad():
        first, second = model(inputs), model(inputs)
    torch.testing.assert_close(first['q'], second['q'], rtol=0, atol=0)


def test_all_explicit_quality_paths_detach_supplied_axis_and_use_actual_pose():
    model = Suction20Net({'experiment': 'E5', 'use_task_success_head': True}, FROZEN, TinyTestBackbone()).eval()
    inputs = model_inputs()
    # Anchor at row0, but put another same-instance point exactly in the lip.
    inputs['xyz_cam_m'][0] = torch.tensor([0., 0., 0.])
    inputs['xyz_cam_m'][1] = torch.tensor([.019, 0., 0.])
    encoded = model.encode_scene(inputs)
    normal = torch.tensor([[0., 0., 2.]], requires_grad=True)
    kwargs = {'instance_id': torch.tensor([1]), 'pose_batch_index': torch.tensor([0]),
              'feature_point_index': torch.tensor([0])}
    origin = model.score_at_pose(encoded, torch.zeros(1, 3), normal, **kwargs)
    moved = model.score_at_pose(encoded, torch.tensor([[.002, 0., 0.]]), normal, **kwargs)
    assert not torch.equal(origin['zone_counts'], moved['zone_counts'])
    score = sum(moved[key].sum() for key in ('q', 'seal_aux', 'wrench_aux', 'collision_free_logit', 'task_success_logit'))
    score.backward()
    assert normal.grad is None
    assert all(p.grad is None for p in model.axis_head.parameters())
    assert model.backbone.linear.weight.grad.abs().sum() > 0


@pytest.mark.parametrize('override', [
    {'experiment': 'E1', 'use_contact_grouping': True},
    {'normal_max_tilt_deg': 45}, {'point_feature_dim': 32},
    {'group_samples': {'core': 1}}, {'voxel_size_m': .005},
    {'experiment': 'E0', 'quality_target': 'gt_B'}, {'seed_count': 0}])
def test_conflicting_structure_is_rejected(override):
    with pytest.raises(ValueError):
        Suction20Net(override, FROZEN, TinyTestBackbone())


def test_e1_has_no_new_state_keys_and_e5_empty_scenes_work():
    e1 = Suction20Net({}, FROZEN, TinyTestBackbone())
    assert not any(key.startswith(('instance_module.', 'contact_grouping.', 'seal_head.', 'wrench_head.', 'collision_head.'))
                   for key in e1.state_dict())
    clone = Suction20Net({}, FROZEN, TinyTestBackbone())
    clone.load_state_dict(e1.state_dict(), strict=True)
    e5 = Suction20Net({'experiment': 'E5', 'seed_count': 4}, FROZEN, TinyTestBackbone())
    inputs = {key: value if key == 'g_cam' else value[:0] for key, value in model_inputs().items()}
    result = e5(inputs)
    assert not result['seed_valid'].any()
    assert result['q'].eq(0).all() and result['seal_aux'].eq(0).all()
    (result['q'].sum() + result['seal_aux'].sum()).backward()
