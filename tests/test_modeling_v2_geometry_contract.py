"""Geometry regressions for the new protocol, independent of training assets."""
import importlib.util
from pathlib import Path


def test_reference_geometry_contract():
    path = Path(__file__).resolve().parents[1] / 'docs/modeling_v2/reference/geometry_contract.py'
    spec = importlib.util.spec_from_file_location('modeling_v2_geometry_contract', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.self_test()
    assert report['status'] == 'passed'
    assert report['scope'] == 'reference_contract_only_not_production_teacher'
