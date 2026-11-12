"""Alignment metadata optimization uses the same raw sparse result as reference."""
import pytest
import torch

from models.backbones import spconv_resunet14 as backbone


def test_shared_storage_skips_mapper_and_gather_but_preserves_feature_gradient(monkeypatch):
    coords = torch.tensor([[0, 1, 2, 3], [0, 2, 1, 3]], dtype=torch.int32)
    alias = coords.view_as(coords)
    features = torch.randn(2, 512, requires_grad=True)
    reference = backbone.align_backbone_features(coords, alias, features)
    def forbidden(*args):
        raise AssertionError('Shared-storage fast path must not read coordinates')
    monkeypatch.setattr(backbone, 'coordinate_rows', forbidden)
    assert backbone.same_coordinate_storage(coords, alias)
    actual = backbone.align_backbone_features(coords, alias, features, fastpath=True)
    assert actual is features
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    cotangent = torch.randn_like(features)
    torch.testing.assert_close(torch.autograd.grad(actual, features, cotangent)[0],
                               torch.autograd.grad(reference, features, cotangent)[0],
                               rtol=0, atol=0)


def test_coordinate_fastpath_falls_back_for_copies_reordering_and_changed_layout():
    coords = torch.tensor([[0, 1, 2, 3], [0, 2, 1, 3]], dtype=torch.int32)
    features = torch.randn(2, 512)
    for indices in (coords.clone(), coords.flip(0)):
        assert not backbone.same_coordinate_storage(coords, indices)
        actual = backbone.align_backbone_features(coords, indices, features, fastpath=True)
        reference = features[backbone.coordinate_rows(coords, indices)]
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    square = torch.arange(16).reshape(4, 4)
    assert square.data_ptr() == square.T.data_ptr()
    assert not backbone.same_coordinate_storage(square, square.T)
    # Same base allocation alone is insufficient when views have different offsets.
    assert not backbone.same_coordinate_storage(coords[:1], coords[1:])
    bad = coords.clone()
    bad[-1, -1] = 9
    with pytest.raises(ValueError, match='missing'):
        backbone.align_backbone_features(coords, bad, features, fastpath=True)
    with pytest.raises(ValueError, match='Duplicate'):
        backbone.align_backbone_features(coords, coords[[0, 0]], features, fastpath=True)
