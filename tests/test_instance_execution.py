"""Executable sample budgets, batching, precision and inference separation.

These CPU contracts do not measure GPU throughput or CUDA compilation speed.
"""
from copy import deepcopy
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

import suction20.instance_inference as inference
from suction20.instance_execution import (DEFAULT_EXECUTION, EXECUTION_SCHEMA,
    configure_model_execution, execution_math_mode, resolve_execution,
    sample_evaluation_steps, source_instance_budget, source_update_size)
from suction20.instance_training import (InstanceTrainer, batched_source_losses,
    create_instance_model, evaluate_instance_source, evaluation_mode,
    load_instance_config, scheduled_evaluation_steps)
from test_instance_inference import FixedScorer, checkpoint_state
from test_instance_training import MemorySource, TinyInstanceModel, config


@pytest.fixture(scope="module", autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def execution_config(*, batch=2, total=5):
    cfg = config()
    pre = cfg["pretrain"]
    pre.pop("warmup_steps")
    pre.pop("evaluate_every")
    pre.update(accumulate_instances=batch, steps=math.ceil(total / batch),
               successful_instance_budget=total, warmup_instances=min(batch, total - 1),
               evaluate_every_instances=batch)
    cfg["execution"] = {
        "microbatch_instances": batch, "evaluation_microbatch_instances": batch,
        "dense_query_chunk": 8, "train_workers": 0, "evaluation_workers": 0,
    }
    return cfg


def test_execution_is_opt_in_and_defaults_are_copied():
    assert resolve_execution(config()) is None
    cfg = config()
    cfg["execution"] = {}
    execution = resolve_execution(cfg)
    assert execution == DEFAULT_EXECUTION
    execution["main_threads"] = 99
    assert DEFAULT_EXECUTION["main_threads"] == 1
    assert cfg["execution"] == {}


@pytest.mark.parametrize("field,value,match", [
    ("schema", "unsupported", "schema"),
    ("unknown", 1, "Unknown execution"),
    ("microbatch_instances", 0, "microbatch_instances"),
    ("microbatch_instances", 3, "effective instance batch"),
    ("evaluation_microbatch_instances", True, "evaluation_microbatch_instances"),
    ("prefetch_factor", 1.5, "prefetch_factor"),
    ("main_threads", 0, "main_threads"),
    ("train_workers", -1, "train_workers"),
    ("evaluation_workers", False, "evaluation_workers"),
    ("dense_query_chunk", 0, "dense_query_chunk"),
    ("dense_query_chunk", True, "dense_query_chunk"),
    ("precision", "fp16", "precision"),
    ("compile_mode", "magic", "compile_mode"),
    ("allow_tf32", 1, "allow_tf32"),
])
def test_execution_rejects_ambiguous_or_invalid_options(field, value, match):
    cfg = execution_config()
    cfg["execution"][field] = value
    with pytest.raises(ValueError, match=match):
        resolve_execution(cfg)


def test_execution_requires_a_mapping():
    cfg = execution_config()
    cfg["execution"] = []
    with pytest.raises(ValueError, match="mapping"):
        resolve_execution(cfg)


@pytest.mark.parametrize("batch,steps", [(32, 4375), (48, 2917), (64, 2188)])
def test_exact_140000_budget_tail_and_ten_sample_milestones(batch, steps):
    cfg = execution_config(batch=batch, total=140000)
    pre = cfg["pretrain"]
    pre.update(evaluate_every_instances=14000, warmup_instances=7008)
    assert source_instance_budget(pre) == 140000
    assert pre["steps"] == steps
    counts = []
    completed = 0
    while remaining := source_update_size(pre, completed):
        counts.append(remaining)
        completed += remaining
    assert len(counts) == steps
    assert counts[:-1] == [batch] * (steps - 1)
    assert counts[-1] == 32
    assert completed == sum(counts) == 140000
    nodes = sample_evaluation_steps(pre)
    assert nodes == tuple(math.ceil(i * 14000 / batch) for i in range(1, 11))
    assert scheduled_evaluation_steps(pre) == nodes
    assert len(nodes) == 10 and nodes[-1] == steps
    for milestone, step in zip(range(14000, 140001, 14000), nodes):
        reached = min(step * batch, completed)
        assert milestone <= reached < milestone + batch
    assert resolve_execution(cfg)["microbatch_instances"] == batch


def test_sample_budget_rejects_wrong_steps_and_schedule_conflicts():
    cfg = execution_config(batch=64, total=140000)
    pre = cfg["pretrain"]
    pre["steps"] -= 1
    with pytest.raises(ValueError, match="ceil"):
        source_instance_budget(pre)
    pre["steps"] += 1
    for completed in (-1, 140001):
        with pytest.raises(ValueError, match="outside"):
            source_update_size(pre, completed)
    for key, value in (("evaluate_every", 100), ("evaluation_steps", [100])):
        with pytest.raises(ValueError, match="not both"):
            sample_evaluation_steps({**pre, key: value})
    pre.pop("successful_instance_budget")
    with pytest.raises(ValueError, match="require successful_instance_budget"):
        sample_evaluation_steps(pre)
    assert source_instance_budget({"steps": 8750, "accumulate_instances": 16}) == 140000


@pytest.mark.parametrize("route", ["a", "b"])
def test_h20_profile_has_exact_budget_and_explicit_deferred_protocol(route):
    path = Path(__file__).resolve().parents[1] / "configs" / f"instance_{route}_h20.yaml"
    if not path.exists():
        pytest.skip(f"Route {route} profile is not part of this branch")
    cfg = load_instance_config(path)
    assert cfg["pretrain"]["steps"] == 4375
    assert source_instance_budget(cfg["pretrain"]) == 140000
    assert len(scheduled_evaluation_steps(cfg["pretrain"])) == 10
    assert cfg["pretrain"]["selection_frames_per_scene"] == 8
    assert cfg["pretrain"]["selection_split"] == "test"
    assert cfg["execution"]["microbatch_instances"] == 32
    assert cfg["execution"]["compile_mode"] is None
    assert cfg["execution"]["precision"] == "bf16_dense"
    assert cfg["execution"]["dense_query_chunk"] == 4096
    assert cfg["public_evaluation"]["status"] == "deferred"


@pytest.mark.parametrize("batch", [32, 48, 64])
def test_learning_rate_uses_consumed_samples_and_hits_final_rate(batch):
    cfg = execution_config(batch=batch, total=140000)
    pre = cfg["pretrain"]
    pre.update(warmup_instances=7008, peak_lr=4.2e-4, final_lr=1.5e-5)
    state = SimpleNamespace(config=cfg, supervised_source_instances=0)
    assert InstanceTrainer._learning_rate(state) == pytest.approx(pre["peak_lr"] * batch / 7008)
    state.supervised_source_instances = (7008 // batch - 1) * batch
    endpoint = state.supervised_source_instances + batch
    assert InstanceTrainer._learning_rate(state) == pytest.approx(pre["peak_lr"] * endpoint / 7008)
    state.supervised_source_instances = (7008 // batch) * batch
    endpoint = state.supervised_source_instances + batch
    cosine_fraction = (endpoint - 7008) / (140000 - 7008)
    expected = pre["final_lr"] + .5 * (pre["peak_lr"] - pre["final_lr"]) * (1 + math.cos(math.pi * cosine_fraction))
    assert InstanceTrainer._learning_rate(state) == pytest.approx(expected)
    state.supervised_source_instances = 139968
    assert InstanceTrainer._learning_rate(state) == pytest.approx(pre["final_lr"])


@pytest.mark.parametrize("extra", [{"warmup_steps": 1}, {"warmup_instances": -1},
                                   {"warmup_instances": 5}, {"warmup_instances": True}])
def test_warmup_policy_rejects_conflicting_or_out_of_budget_values(extra):
    cfg = execution_config()
    cfg["pretrain"].update(extra)
    with pytest.raises(ValueError, match="warmup"):
        resolve_execution(cfg)


def test_execution_math_mode_restores_backend_flags_on_failure():
    previous = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    execution = {**DEFAULT_EXECUTION, "allow_tf32": True}
    try:
        with execution_math_mode(execution, training=True):
            assert torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
            with pytest.raises(RuntimeError, match="intentional"):
                with execution_math_mode(execution, training=False):
                    assert not torch.backends.cuda.matmul.allow_tf32
                    assert not torch.backends.cudnn.allow_tf32
                    raise RuntimeError("intentional")
            assert torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
        assert (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) == previous
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous


def test_explicit_execution_refuses_a_hidden_serial_model_fallback():
    execution = resolve_execution(execution_config())
    with pytest.raises(ValueError, match="implement explicit batched"):
        configure_model_execution(TinyInstanceModel(), execution)
    model = TinyInstanceModel()
    model.configure_execution = lambda **kwargs: None
    with pytest.raises(ValueError, match="True batching"):
        configure_model_execution(model, execution)


def loss_example():
    q_values = ([.25, np.nan, np.nan], [.25, .75, .75], [np.nan, np.nan])
    known = ([True, False, True], [True, True, True], [False, False])
    supported = ([True, True, False], [True, True, True], [True, True])
    truth = ([.5, np.nan, np.nan], [0., 0., 0.], [np.nan, np.nan])
    predictions, labels = [], []
    for q, valid, support, target in zip(q_values, known, supported, truth):
        q = torch.tensor(q, dtype=torch.bfloat16, requires_grad=True)
        aux = torch.where(torch.isfinite(q), torch.full_like(q, .5), torch.full_like(q, float("nan"))).detach()
        ell = torch.where(torch.isfinite(q), torch.zeros_like(q), torch.full_like(q, float("nan"))).detach()
        predictions.append({"q_inst": q, "seal": aux.clone().requires_grad_(),
                            "wrench": aux.clone().requires_grad_(), "ell": ell.requires_grad_(),
                            "valid": torch.tensor(support)})
        target = np.array(target, dtype=np.float64)
        aux_target = np.where(np.isfinite(target), .5, np.nan)
        labels.append({"q_inst": target, "seal": aux_target, "wrench": aux_target.copy(),
                       "label_valid": np.array(valid)})
    return predictions, labels


def test_batch_loss_is_fp32_and_gives_each_instance_equal_weight_not_each_query():
    predictions, labels = loss_example()
    losses = batched_source_losses(predictions, labels)
    assert losses.dtype == torch.float32
    torch.testing.assert_close(losses, torch.tensor([.25, 7 / 12]))
    assert losses.mean().item() == pytest.approx(5 / 12)
    losses.mean().backward()
    torch.testing.assert_close(predictions[0]["q_inst"].grad.float(), torch.tensor([-.5, 0., 0.]))
    torch.testing.assert_close(predictions[1]["q_inst"].grad.float(),
                               torch.full((3,), 1 / 6), rtol=.004, atol=0)
    assert torch.equal(predictions[2]["q_inst"].grad, torch.zeros(2, dtype=torch.bfloat16))
    assert all(torch.isfinite(p[key].grad).all() for p in predictions
               for key in ("q_inst", "seal", "wrench", "ell"))


def test_batch_loss_does_not_hide_nonfinite_known_targets_or_predictions():
    predictions, labels = loss_example()
    labels[0]["q_inst"][0] = np.nan
    with pytest.raises(ValueError, match="Known physical"):
        batched_source_losses(predictions, labels)
    predictions, labels = loss_example()
    with torch.no_grad():
        predictions[0]["q_inst"][0] = float("nan")
    with pytest.raises(FloatingPointError, match="Nonfinite model"):
        batched_source_losses(predictions, labels)
    with pytest.raises(ValueError, match="output count"):
        batched_source_losses(predictions, labels[:-1])


class TinyBatchedModel(TinyInstanceModel):
    """One joined neural call; independent of either production architecture."""

    def __init__(self):
        super().__init__()
        self.dropout = nn.Identity()
        self.batch_sizes = []
        self.execution_options = None

    def configure_execution(self, **options):
        self.execution_options = dict(options)

    def forward_instances(self, observations):
        self.batch_sizes.append(len(observations))
        sizes = [len(obs["query_points"]) for obs in observations]
        joined = self.score_at_pose(None, np.concatenate([obs["query_points"] for obs in observations]),
                                    np.concatenate([obs["query_normals"] for obs in observations]))
        outputs = []
        start = 0
        for obs, size in zip(observations, sizes):
            output = {key: tensor[start:start + size] for key, tensor in joined.items()}
            if not obs.get("fixture_model_supported", True):
                output["valid"] = torch.zeros_like(output["valid"])
            outputs.append(output)
            start += size
        return outputs


class QualitySource(MemorySource):
    def __getitem__(self, index):
        item = super().__getitem__(index)
        item["labels"]["q_inst"][:] = .8
        return item


def test_trainer_preserves_effective_budget_with_skips_and_final_partial_batch(tmp_path, monkeypatch):
    class SometimesUnknown(QualitySource):
        def __getitem__(self, index):
            item = super().__getitem__(index)
            if index == 0:
                item["labels"]["label_valid"][:] = False
                for key in ("q_inst", "seal", "wrench"):
                    item["labels"][key][:] = np.nan
            return item

    cfg = execution_config()
    model = TinyBatchedModel()
    trainer = InstanceTrainer(model, cfg, SometimesUnknown(), QualitySource("test"), tmp_path,
                              allow_partial=True)
    indices = iter([0, 1, 2, 1, 2, 1])

    def next_index():
        trainer.source_schedule.draws += 1
        return next(indices)

    monkeypatch.setattr(trainer.source_schedule, "next", next_index)
    try:
        records = [trainer.train_one_update() for _ in range(3)]
        assert [r["effective_source_instances_this_update"] for r in records] == [2, 2, 1]
        assert [r["supervised_source_instances"] for r in records] == [2, 4, 5]
        assert trainer.step == 3 and trainer.skipped_no_supervision == 1
        assert records[-1]["source_draws"] == 6
        assert model.batch_sizes == [2, 1, 2, 1]
        assert all(math.isfinite(record["source_loss"]) for record in records)
        assert set(records[-1]["learning_rates"].values()) == {cfg["pretrain"]["final_lr"]}
        assert trainer.policy["successful_instance_budget"] == 5
        assert trainer.policy["final_update_instances"] == 1
        assert trainer.policy["evaluation_instance_counts"] == [2, 4, 5]
        state = trainer.state_dict()
        assert state["preparation_state"]["schema"] == EXECUTION_SCHEMA
        assert state["preparation_state"]["pending_source"] == []
        with pytest.raises(ValueError, match="already complete"):
            trainer.train_one_update()
    finally:
        trainer.close()


def test_batched_evaluation_retains_invalid_instances_in_metric_denominators():
    class MixedSelection(QualitySource):
        def __getitem__(self, index):
            item = super().__getitem__(index)
            if index == 0:
                item["observation"]["support_points"] = item["observation"]["support_points"][:20]
            elif index == 1:
                for key in ("query_points", "query_normals", "query_ids"):
                    item["observation"][key] = item["observation"][key][:0]
                item["labels"] = {key: value[:0] for key, value in item["labels"].items()}
            elif index == 2:
                item["observation"]["fixture_model_supported"] = False
            return item

    model = TinyBatchedModel().train()
    model.decoder[1].eval()
    modes = {name: module.training for name, module in model.named_modules()}
    configure_model_execution(model, resolve_execution(execution_config()))
    selection = MixedSelection("test", count=4)
    metrics, rows = evaluate_instance_source(model, selection)
    assert model.batch_sizes == [2]
    assert metrics["instance_count"] == 4 and metrics["no_pose_count"] == 3
    assert metrics["InstTop1_INST20_at_0_4"] == pytest.approx(.25)
    assert metrics["selected_Q_macro"] == pytest.approx(.2)
    assert metrics["Coverage_INST20_at_256"] == pytest.approx(.75)
    assert [row["instance_uid"] for row in rows] == [f"test-{i}" for i in range(4)]
    assert [row["selected_query"] is None for row in rows] == [True, True, True, False]
    assert {name: module.training for name, module in model.named_modules()} == modes


def test_inference_validates_but_does_not_activate_saved_training_execution(tmp_path, monkeypatch):
    state = checkpoint_state(FixedScorer(), route="b")
    cfg = state["config"]
    cfg["pretrain"].update(execution_config()["pretrain"])
    cfg["execution"] = {**resolve_execution(execution_config()), "precision": "bf16_dense",
                        "compile_mode": "reduce-overhead", "train_workers": 24,
                        "evaluation_workers": 16, "main_threads": 7, "allow_tf32": True}
    state["identity"]["config_definition"] = deepcopy(cfg)
    path = tmp_path / "training_profile.pt"
    torch.save(state, path)
    calls = []

    def factory(config, device="cpu"):
        assert "execution" not in config
        calls.append(deepcopy(config))
        return FixedScorer().to(device)

    monkeypatch.setattr(inference, "create_instance_model", factory)
    monkeypatch.setattr(torch, "compile", lambda *a, **kw: pytest.fail("Inference enabled training compilation"))
    monkeypatch.setattr(torch, "set_num_threads", lambda *a, **kw: pytest.fail("Inference changed host thread count"))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **kw: pytest.fail("FP32 inference probed BF16"))
    loaded, metadata = inference.load_instance_checkpoint(path)
    assert len(calls) == 1 and not loaded.training
    assert metadata["checkpoint_training_execution"] == cfg["execution"]
    assert metadata["runtime_inference_execution"] == {
        "precision": "fp32", "compile_mode": None, "preparation_workers": 0, "allow_tf32": False}
    assert loaded._instance_execution["precision"] == "fp32"
    assert loaded._instance_execution["compile_mode"] is None
    assert loaded._instance_execution["train_workers"] == loaded._instance_execution["evaluation_workers"] == 0
    previous = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    try:
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
        with evaluation_mode(loaded):
            assert not torch.backends.cuda.matmul.allow_tf32
            assert not torch.backends.cudnn.allow_tf32
        assert torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous

    # Disabling activation must not turn malformed stored profiles into valid checkpoints.
    cfg["execution"]["precision"] = "invalid"
    state["identity"]["config_definition"] = deepcopy(cfg)
    torch.save(state, path)
    with pytest.raises(ValueError, match="precision"):
        inference.load_instance_checkpoint(path)
    assert len(calls) == 1


def test_actual_route_b_bf16_batch_train_and_fp32_evaluation_smoke(tmp_path):
    if not (Path(__file__).resolve().parents[1] / "models" / "instance_route_b.py").exists():
        pytest.skip("Production route B is not part of this branch")
    from test_instance_route_b import plane_observation

    class Images(QualitySource):
        def __getitem__(self, index):
            item = super().__getitem__(index)
            obs = plane_observation(48 + 2 * index, 64)
            obs["sample_uid"] = item["record"]["instance_uid"]
            obs["support_points"] = obs["xyz_map"][obs["mask"]]
            obs["query_ids"] = np.arange(len(obs["query_points"]))
            item["observation"] = obs
            return item

    cfg = execution_config(total=4)
    cfg["model"] = {"query_chunk": 2}
    cfg["execution"]["precision"] = "bf16_dense"
    model = create_instance_model(cfg)
    trainer = InstanceTrainer(model, cfg, Images(), Images("test"), tmp_path, allow_partial=True)
    dtypes = []
    handle = model.query_fusion.register_forward_hook(lambda _m, _args, out: dtypes.append(out.dtype))
    try:
        record = trainer.train_one_update()
        assert record["step"] == 1 and record["supervised_source_instances"] == 2
        assert dtypes == [torch.bfloat16]
        dtypes.clear()
        metrics, _ = evaluate_instance_source(model, trainer.selection)
        assert metrics["instance_count"] == 3
        assert dtypes == [torch.float32, torch.float32]
        assert model.training and model.execution_precision == "bf16_dense"
    finally:
        handle.remove()
        trainer.close()
