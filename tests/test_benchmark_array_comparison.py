"""Keep bounded benchmark equivalence checks sensitive to actual array values."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def load_script(name):
    path = Path(__file__).resolve().parents[1] / 'scripts' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pipeline = load_script('benchmark_training_pipeline')
batch = load_script('benchmark_batch_cpu')


def test_direct_comparison_preserves_array_contract_and_bit_pattern():
    expected = np.array([0., np.nan, 3.], dtype=np.float64)
    assert pipeline.exact_equal({'x': expected}, {'x': expected.copy()})
    assert not pipeline.exact_equal(expected, expected.astype(np.float32))
    assert not pipeline.exact_equal(expected, expected.reshape(1, -1))
    changed = expected.copy()
    changed[0] = -0.
    assert not pipeline.exact_equal(expected, changed)
    # Descriptive shape/dtype metadata must never decide value equivalence.
    assert pipeline.array_structure(expected) == pipeline.array_structure(changed)
    matrix = np.arange(12).reshape(3, 4)
    assert pipeline.exact_equal(matrix.T, matrix.T.copy())


def write_four_frame_evidence(out):
    data = {'inputs': torch.tensor([[.1, .2, .3]]), 'targets': np.array([1.], np.float32)}
    query = {'geometry': np.array([.1, .2], dtype=np.float32), 'indices': np.array([1, 2])}
    frames = [dict(frame_key=[0, i], augmentation_seed=i,
                   data=pipeline.array_structure(data), loss={'total': .5}) for i in range(4)]
    for phase in ('baseline', 'optimized'):
        for i in range(4):
            torch.save(data, out / f'{phase}-data-{i}.pt')
            torch.save({'prediction': torch.tensor([.5])}, out / f'{phase}-frame-{i}.pt')
        torch.save({'gradients': {'weight': torch.tensor([1.])}, 'buffers': {}, 'rng': {}},
                   out / f'{phase}-state.pt')
        artifact = f'{phase}-query-0.pt'
        torch.save(query, out / artifact)
        report = dict(phase=phase, amp=False, checkpoint_step=10, sampler_after={'position': 4},
                      checkpoint_metadata={'path': '/fixture/checkpoint.pt', 'size_bytes': 100},
                      frames=copy.deepcopy(frames),
                      queries=[{'result': pipeline.array_structure(query), 'artifact': artifact}])
        (out / f'{phase}.json').write_text(json.dumps(report))


@pytest.mark.parametrize('corrupt', ['data', 'query'])
def test_cross_process_evidence_compares_saved_values_not_structure(tmp_path, corrupt):
    write_four_frame_evidence(tmp_path)
    args = SimpleNamespace(out_dir=str(tmp_path))
    assert pipeline.compare(args) == 0
    if corrupt == 'data':
        path = tmp_path / 'optimized-data-2.pt'
        value = torch.load(path, weights_only=False)
        value['inputs'][0, 0] = torch.nextafter(value['inputs'][0, 0], torch.tensor(float('inf')))
        expected_failure = 'frame.2.data_values'
    else:
        path = tmp_path / 'optimized-query-0.pt'
        value = torch.load(path, weights_only=False)
        value['geometry'][0] = np.nextafter(value['geometry'][0], np.float32(np.inf))
        expected_failure = 'query.0.values'
    torch.save(value, path)
    assert pipeline.compare(args) == 2
    assert json.loads((tmp_path / 'comparison.json').read_text())['failures'] == [expected_failure]


def test_disposable_benchmark_detects_parameter_updates_without_a_file_scan():
    model = torch.nn.Linear(2, 1)
    snapshot = batch.parameter_snapshot(model)
    assert batch.parameters_unchanged(model, snapshot)
    with torch.no_grad():
        model.weight[0, 0].add_(1.)
    assert not batch.parameters_unchanged(model, snapshot)
