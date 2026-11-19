"""True route-B neural batching and explicit mixed-precision boundaries.

CPU synthetic tests verify arithmetic/gradient contracts, not H20 throughput.
"""
from copy import deepcopy
import pickle

import numpy as np
import pytest
import torch

import models.instance_route_b as route_b
from models.instance_route_b import (InstanceSuction2DFastV3,
    prepare_source_inputs_cpu, prepare_source_phase_cpu)
from test_instance_route_b import plane_observation


@pytest.fixture(scope="module", autouse=True)
def bounded_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def observations():
    rows = [plane_observation(48, 64), plane_observation(52, 60), plane_observation(76, 100)]
    for i, row in enumerate(rows):
        row["sample_uid"] = f"batch/instance-{i}"
    rows[1]["query_points"] = rows[1]["query_points"][:2]
    rows[1]["query_normals"] = rows[1]["query_normals"][:2]
    return rows


def assert_predictions_close(actual, expected, *, rtol=3e-5, atol=2e-6):
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected):
        for key in ("q_inst", "seal", "wrench", "ell", "task_logit", "log_variance"):
            torch.testing.assert_close(a[key], b[key], rtol=rtol, atol=atol)
        for key in ("valid", "query_valid", "points_cam_m", "normals_out_cam"):
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        for key in ("contact_valid", "observed_valid", "height_m", "rho_m", "phase",
                    "support_summary", "raw_summary", "support_count", "source_ids"):
            torch.testing.assert_close(a["diagnostics"][key], b["diagnostics"][key], rtol=0, atol=0)


def test_context_and_neural_queries_batch_while_texture_preserves_exact_roi_buckets():
    model = InstanceSuction2DFastV3(query_chunk=2).eval()
    model.configure_execution(dense_query_chunk=5)
    rows = observations()
    with torch.no_grad():
        expected = [model(row) for row in rows]
    calls = {"encoder": [], "texture": [], "ring": [], "fusion": []}
    hooks = [model.encoder.register_forward_pre_hook(lambda _m, args: calls["encoder"].append(tuple(args[0].shape))),
             model.texture_stem.register_forward_pre_hook(lambda _m, args: calls["texture"].append(tuple(args[0].shape))),
             model.ring_encoders[0].register_forward_pre_hook(lambda _m, args: calls["ring"].append(len(args[0]))),
             model.query_fusion.register_forward_pre_hook(lambda _m, args: calls["fusion"].append(len(args[0])))]
    try:
        with torch.no_grad():
            actual = model.forward_instances(rows)
    finally:
        for hook in hooks:
            hook.remove()
    assert calls["encoder"] == [(3, 13, 128, 128)]
    assert calls["texture"] == [(2, 5, 64, 96), (1, 5, 96, 128)]
    assert calls["ring"] == calls["fusion"] == [5, 3]
    assert_predictions_close(actual, expected)


def _loss(outputs):
    return torch.stack([sum((part[key].square().mean() for key in
                             ("q_inst", "seal", "wrench", "ell", "task_logit")))
                        for part in outputs]).mean()


@pytest.mark.parametrize("dense_chunk", [3, 8])
def test_batched_gradient_matches_instance_normalized_serial_training(dense_chunk):
    torch.manual_seed(52)
    serial = InstanceSuction2DFastV3(query_chunk=2).train()
    batched = deepcopy(serial).configure_execution(dense_query_chunk=dense_chunk)
    serial.set_training_context(13)
    batched.set_training_context(13)
    # Unequal query counts, two exact-size texture buckets, and a short final
    # dense chunk must preserve all parameter gradients through batch splitting.
    rows = observations()
    _loss([serial(row) for row in rows]).backward()
    _loss(batched.forward_instances(rows)).backward()
    for (name, a), (other, b) in zip(serial.named_parameters(), batched.named_parameters()):
        assert name == other
        assert a.grad is not None and b.grad is not None
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-3, atol=2e-5, msg=name)
    assert batched.encoder.stem[0].weight.grad.abs().sum() > 0


def test_bf16_neural_blocks_keep_geometry_masks_and_loss_outputs_full_precision():
    torch.manual_seed(8)
    model = InstanceSuction2DFastV3(query_chunk=2, training_phase=False).train()
    rows = observations()[:2]
    with torch.no_grad():
        reference = model.forward_instances(rows)
    model.configure_execution(precision="bf16_dense", dense_query_chunk=8)
    dtypes = []
    handles = [model.encoder.layer1.register_forward_hook(lambda _m, _a, out: dtypes.append(out.dtype)),
               model.query_fusion.register_forward_hook(lambda _m, _a, out: dtypes.append(out.dtype))]
    try:
        actual = model.forward_instances(rows)
        _loss(actual).backward()
    finally:
        for handle in handles:
            handle.remove()
    assert dtypes == [torch.bfloat16, torch.bfloat16]
    for output, old in zip(actual, reference):
        for key in ("q_inst", "seal", "wrench", "ell", "task_logit"):
            assert output[key].dtype == torch.float32
            assert torch.isfinite(output[key]).all()
            torch.testing.assert_close(output[key], old[key], rtol=.04, atol=.025)
        for key in ("points_cam_m", "normals_out_cam"):
            assert output[key].dtype == torch.float64
        for key in ("height_m", "rho_m", "support_summary", "raw_summary"):
            assert output["diagnostics"][key].dtype == torch.float64
            torch.testing.assert_close(output["diagnostics"][key], old["diagnostics"][key], rtol=0, atol=0)
        assert torch.equal(output["diagnostics"]["contact_valid"], old["diagnostics"]["contact_valid"])
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.encoder.stem[0].weight.grad.abs().sum() > 0


def test_bf16_profile_evaluation_uses_fp32_and_restores_training_choice():
    model = InstanceSuction2DFastV3(training_phase=False).configure_execution(precision="bf16_dense")
    row = observations()[0]
    dtypes = []
    handle = model.query_fusion.register_forward_hook(lambda _m, _a, out: dtypes.append(out.dtype))
    try:
        with torch.no_grad():
            model.eval()(row)
            model.train()(row)
    finally:
        handle.remove()
    assert dtypes == [torch.float32, torch.bfloat16]
    assert model.execution_precision == "bf16_dense"


def test_worker_inputs_are_numpy_pickle_safe_and_model_reuses_them(monkeypatch):
    row = prepare_source_inputs_cpu(observations()[0])
    cached = pickle.loads(pickle.dumps(row))
    assert all(not torch.is_tensor(value) for value in cached["_route_b_inputs"]["encoded"].values())
    assert cached["_route_b_inputs"]["sources"]["rgb"] is cached["rgb"]
    model = InstanceSuction2DFastV3().eval()
    with torch.no_grad():
        expected = model(observations()[0])
    monkeypatch.setattr(route_b, "prepare_shared_images",
                        lambda *a, **k: pytest.fail("Valid worker inputs were rebuilt in the model"))
    with torch.no_grad():
        actual = model(cached)
    assert_predictions_close([actual], [expected], rtol=0, atol=0)


@pytest.mark.parametrize("mutation", ["rgb", "threshold", "spec"])
def test_worker_cache_rejects_replaced_sources_or_wrong_contract(monkeypatch, mutation):
    row = prepare_source_inputs_cpu(observations()[0])
    if mutation == "rgb":
        row["rgb"] = np.zeros_like(row["rgb"])
    elif mutation == "threshold":
        row["_route_b_inputs"]["min_support_points"] = 101
    else:
        row["_route_b_inputs"]["spec"] = "different"
    calls = []
    original = route_b.prepare_shared_images
    def prepare(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(route_b, "prepare_shared_images", prepare)
    model = InstanceSuction2DFastV3().eval()
    with torch.no_grad():
        model(row)
    assert calls == [True]


def test_worker_phase_identity_uses_consumed_update_and_survives_query_reordering(monkeypatch):
    model = InstanceSuction2DFastV3().train()
    row = prepare_source_phase_cpu(observations()[0])
    p = torch.as_tensor(row["query_points"], dtype=torch.float64)
    n = torch.as_tensor(row["query_normals"], dtype=torch.float64)
    encoded = {"sample_uid": row["sample_uid"]}
    for update in (0, 99):
        model.set_training_context(update)
        expected = model._phase(encoded, p, n)
        torch.testing.assert_close(model._canonical_phase(row), expected, rtol=0, atol=0)
        reordered = dict(row, query_points=row["query_points"][[2, 0, 1]],
                         query_normals=row["query_normals"][[2, 0, 1]])
        torch.testing.assert_close(model._canonical_phase(reordered), expected[[2, 0, 1]], rtol=0, atol=0)
    # Canonical NumPy tickets never take the tensor-to-CPU phase path.
    monkeypatch.setattr(model, "_phase", lambda *a: pytest.fail("Unexpected tensor-to-CPU phase fallback"))
    assert model._canonical_phase(row).shape == (3,)


def test_batched_empty_and_unsupported_instances_preserve_output_alignment():
    empty, small, normal = observations()
    empty["query_points"] = np.empty((0, 3), np.float64)
    empty["query_normals"] = np.empty((0, 3), np.float64)
    small["mask"][:] = False
    small["mask"].flat[:10] = True
    model = InstanceSuction2DFastV3().eval().configure_execution(dense_query_chunk=8)
    with torch.no_grad():
        result = model.forward_instances([empty, small, normal])
    assert len(result) == 3
    assert result[0]["q_inst"].shape == (0,)
    assert not result[1]["valid"].any()
    assert not result[1]["q_inst"].any()
    assert result[2]["valid"].all()
    assert model.forward_instances([]) == []


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_dense_query_chunk_is_validated(value):
    with pytest.raises(ValueError, match="dense_query_chunk"):
        InstanceSuction2DFastV3().configure_execution(dense_query_chunk=value)


def test_compile_is_optional_and_does_not_add_checkpoint_keys_or_parameters(monkeypatch):
    model = InstanceSuction2DFastV3().eval()
    names = tuple(model.state_dict())
    params = tuple(id(p) for p in model.parameters())
    calls = []
    def compiler(module, *, mode):
        calls.append((module, mode))
        return lambda *args: module(*args)
    monkeypatch.setattr(torch, "compile", compiler)
    model.configure_execution(compile_mode="reduce-overhead", dense_query_chunk=8)
    assert len(calls) == 10  # encoder, fusion, three ring blocks and five heads
    assert model.texture_stem not in [module for module, _ in calls]
    assert tuple(model.state_dict()) == names
    assert tuple(id(p) for p in model.parameters()) == params
    with torch.no_grad():
        assert model(observations()[0])["valid"].all()
    model.configure_execution(compile_mode=None)
    assert not model._compiled_dense
    assert tuple(model.state_dict()) == names

@pytest.mark.parametrize("mask_kind", ["interior", "empty", "full"])
def test_compact_source_ticket_roundtrips_geometry_scores_and_gradients(mask_kind):
    obs = observations()[0]
    if mask_kind == "interior":
        obs["mask"][:8] = False
        obs["mask"][:, :10] = False
        obs["mask"][-3:] = False
    elif mask_kind == "empty":
        obs["mask"][:] = False
    prepared = prepare_source_inputs_cpu(obs)
    item = {"observation": prepared, "record": {"id": 1}, "labels": {"seal": 2}}
    candidate = dict(item, observation=prepare_source_inputs_cpu(obs, sample_image_channels=True))
    compact = pickle.loads(pickle.dumps(route_b.compact_prepared_source_item(candidate)))
    assert compact["record"] == item["record"] and compact["labels"] == item["labels"]
    model = InstanceSuction2DFastV3().train()
    model.set_training_context(7)
    small = compact["observation"]
    ref = model._prepare_encoded(prepared)
    restored = model._prepare_encoded(small)
    assert set(restored) == set(ref) | {"geometry_image_hw", "geometry_top_left", "bilinear_valid_map"}
    height, width = ref["mask"].shape
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    for key, value in ref.items():
        if key in route_b._GEOMETRY_MAPS:
            logical = route_b._geometry_at_pixels(restored, key, yy, xx)
            torch.testing.assert_close(logical, value, rtol=0, atol=0)
        elif isinstance(value, torch.Tensor):
            torch.testing.assert_close(restored[key], value, rtol=0, atol=0)
        elif key != "observation":
            assert restored[key] == value
    torch.testing.assert_close(model._canonical_phase(small), model._canonical_phase(prepared), rtol=0, atol=0)
    if mask_kind == "empty":
        return
    def score(row):
        model.zero_grad(set_to_none=True)
        result = model(row)
        loss = result["q_inst"].sum()
        loss.backward()
        return result, [p.grad.detach().clone() for p in model.parameters() if p.grad is not None]
    full_pred, full_grad = score(prepared)
    new_pred, new_grad = score(small)
    assert_predictions_close([new_pred], [full_pred], rtol=0, atol=0)
    assert len(new_grad) == len(full_grad) > 0
    for actual, expected in zip(new_grad, full_grad):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
