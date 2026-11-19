"""Opt-in contact-point projection preserves geometry and the full public API."""
import copy
import pickle

import numpy as np
import pytest
import torch

from models.instance_route_a import InstanceSuction3D, compact_prepared_source_item
from test_instance_route_a import InjectedBackbone, observation


@pytest.fixture(scope="module", autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


LEARNED = {"q_inst", "seal", "wrench", "ell", "log_scale", "log_variance", "task_logit", "task_probability"}


def compare_scores(actual, expected, *, atol=1e-7, rtol=2e-5):
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        if torch.is_tensor(value):
            tolerance = dict(atol=atol, rtol=rtol) if key in LEARNED else dict(atol=0, rtol=0)
            torch.testing.assert_close(actual[key], value, equal_nan=True, msg=key, **tolerance)
        else:
            assert actual[key] == value


def loss(outputs):
    return sum(output[key].square().sum() for output in outputs
               for key in ("q_inst", "seal", "wrench", "ell", "task_logit"))


def compare_gradients(actual, expected, *, atol=1e-7, rtol=2e-5):
    for (name, left), (other, right) in zip(actual.named_parameters(), expected.named_parameters()):
        assert name == other
        if right.grad is None:
            assert left.grad is None, name
        else:
            assert left.grad is not None, name
            torch.testing.assert_close(left.grad, right.grad, rtol=rtol, atol=atol, msg=name)
    for (name, left), (other, right) in zip(actual.named_buffers(), expected.named_buffers()):
        assert name == other
        torch.testing.assert_close(left, right, rtol=0, atol=0, msg=name)


def prepared_rows(model, *, empty=False):
    first, second = observation(41), observation(51)
    first["support_rgb"][:] = .2
    second["support_rgb"][:] = .8
    second["query_points"] = second["query_points"][:2]
    second["query_normals"] = second["query_normals"][:2]
    second["query_ids"] = second["query_ids"][:2]
    if empty:
        second["query_points"] = second["query_points"][:0]
        second["query_normals"] = second["query_normals"][:0]
        second["query_ids"] = second["query_ids"][:0]
    return [compact_observation(model, obs) for obs in (first, second)]


def compact_observation(model, obs):
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("SUCTION_A_CONTACT_UNION", "1")
        item = compact_prepared_source_item({"observation": model.prepare_source_geometry(obs)})
    return pickle.loads(pickle.dumps(item))["observation"]


@pytest.mark.parametrize("precision", ["fp32", "bf16_dense"])
def test_union_projection_preserves_predictions_gradients_and_sparse_bn(precision, monkeypatch):
    torch.manual_seed(370)
    reference = InstanceSuction3D(backbone=InjectedBackbone(with_bn=True), backbone_points=128,
                                  query_chunk_size=2).train().configure_execution(
                                      precision=precision, dense_query_chunk=3)
    current = copy.deepcopy(reference)
    rows = prepared_rows(reference)
    old_counts, new_counts = [], []
    old_hook = reference.local_projection.register_forward_pre_hook(lambda _m, args: old_counts.append(len(args[0])))
    new_hook = current.local_projection.register_forward_pre_hook(lambda _m, args: new_counts.append(len(args[0])))
    monkeypatch.setenv("SUCTION_A_CONTACT_UNION", "0")
    expected = reference.forward_instances(rows)
    loss(expected).backward()
    monkeypatch.setenv("SUCTION_A_CONTACT_UNION", "1")
    actual = current.forward_instances(rows)
    loss(actual).backward()
    old_hook.remove()
    new_hook.remove()
    assert old_counts == [sum(len(obs["support_points"]) for obs in rows)]
    assert new_counts == [sum(len(obs["_route_a_contact_feature_rows"]) for obs in rows)]
    assert new_counts[0] < old_counts[0]
    for left, right in zip(actual, expected):
        compare_scores(left, right)
    compare_gradients(current, reference)


def test_public_encode_and_eval_keep_full_reference_features_with_flag_enabled(monkeypatch):
    monkeypatch.setenv("SUCTION_A_CONTACT_UNION", "1")
    model = InstanceSuction3D(backbone=InjectedBackbone(), backbone_points=64, query_chunk_size=2).train()
    rows = prepared_rows(model)
    encoded = model.encode_instance(rows[0])
    assert encoded["contact_feature_rows"] is None
    assert len(encoded["raw_projected_features"]) == len(rows[0]["support_points"])
    assert len(encoded["rgb"]) == len(rows[0]["support_points"])
    model.eval()
    calls = []
    hook = model.local_projection.register_forward_pre_hook(lambda _m, args: calls.append(len(args[0])))
    with torch.no_grad():
        model.forward_instances(rows)
    hook.remove()
    assert calls == [sum(len(row["support_points"]) for row in rows)]


def test_private_partial_features_lazily_restore_arbitrary_pose_and_keep_autograd():
    torch.manual_seed(839)
    reference = InstanceSuction3D(backbone=InjectedBackbone(with_bn=True), backbone_points=128,
                                  query_chunk_size=2).train()
    current = copy.deepcopy(reference)
    obs = observation(61)
    small = compact_observation(reference, obs)
    full = reference.encode_instance(small)
    partial = current.encode_instances([small], _contact_union=True)[0]
    assert len(partial["raw_projected_features"]) < len(full["raw_projected_features"])
    p, n = obs["query_points"], obs["query_normals"]
    expected = [reference.score_at_pose(full, p, n)]
    actual = [current.score_at_pose(partial, p, n)]
    assert partial["contact_feature_rows"] is not None
    new_point, new_normal = np.array([[.025, .025, .6]]), np.array([[0., 0., -1.]])
    expected.append(reference.score_at_pose(full, new_point, new_normal))
    actual.append(current.score_at_pose(partial, new_point, new_normal))
    assert partial["contact_feature_rows"] is None
    assert len(partial["raw_projected_features"]) == len(obs["support_points"])
    loss(expected).backward()
    loss(actual).backward()
    for left, right in zip(actual, expected):
        compare_scores(left, right)
    compare_gradients(current, reference)
    assert current.backbone.projection.weight.grad.abs().sum() > 0
    assert current.local_projection[0].weight.grad.abs().sum() > 0


@pytest.mark.parametrize("unsupported", [False, True])
def test_empty_contact_union_retains_zero_gradients_and_mixed_batch_ownership(unsupported, monkeypatch):
    torch.manual_seed(793)
    reference = InstanceSuction3D(backbone=InjectedBackbone(with_bn=True), backbone_points=64,
                                  query_chunk_size=2).train()
    current = copy.deepcopy(reference)
    if unsupported:
        obs = observation(7)
        rows = [compact_observation(reference, obs)]
        assert len(rows[0]["_route_a_contact_feature_rows"]) == 0
    else:
        rows = prepared_rows(reference, empty=True)
        assert len(rows[1]["_route_a_contact_feature_rows"]) == 0
    monkeypatch.setenv("SUCTION_A_CONTACT_UNION", "0")
    expected = reference.forward_instances(rows)
    loss(expected).backward()
    monkeypatch.setenv("SUCTION_A_CONTACT_UNION", "1")
    actual = current.forward_instances(rows)
    loss(actual).backward()
    for left, right in zip(actual, expected):
        compare_scores(left, right)
    compare_gradients(current, reference)


def test_default_compact_wire_keeps_reference_payload_without_union_hint(monkeypatch):
    monkeypatch.delenv("SUCTION_A_CONTACT_UNION", raising=False)
    model = InstanceSuction3D(backbone=InjectedBackbone(), backbone_points=64, query_chunk_size=2).train()
    prepared = model.prepare_source_geometry(observation())
    small = compact_prepared_source_item({"observation": prepared})["observation"]
    assert "_route_a_contact_feature_rows" not in small
    # A previously prepared ticket is still valid after enabling the consumer.
    monkeypatch.setenv("SUCTION_A_CONTACT_UNION", "1")
    calls = []
    hook = model.local_projection.register_forward_pre_hook(lambda _m, args: calls.append(len(args[0])))
    model.forward_instances([small])
    hook.remove()
    assert calls == [len(small["support_points"])]
