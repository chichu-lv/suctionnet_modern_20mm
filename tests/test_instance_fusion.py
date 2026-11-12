import torch
from models.instance_fusion import InstanceFusion, instance_blocks


def test_instance_pooling_frame_keys_background_and_geometry():
    torch.manual_seed(42)
    features = torch.randn(5, 256, requires_grad=True)
    xyz = torch.tensor([[1., 1., 1.], [1., 3., 1.], [0., 0., 0.], [5., 5., 5.], [7., 5., 5.]])
    batch = torch.tensor([0, 0, 0, 1, 1])
    instance = torch.tensor([1, 1, 0, 1, 1])
    module = InstanceFusion()
    stats = module(features, xyz, batch, instance, checkpointed=True)
    assert stats['instance_keys'].tolist() == [[0, 1], [1, 1]]
    assert stats['point_instance_index'].tolist() == [1, 1, 0, 2, 2]
    torch.testing.assert_close(stats['instance_centers'], torch.tensor([[0., 0., 0.], [1., 2., 1.], [6., 5., 5.]]))
    torch.testing.assert_close(stats['instance_extents'], torch.tensor([[0., 0., 0.], [0., 2., 0.], [2., 0., 0.]]))
    g, relative, extent = instance_blocks(stats, xyz, stats['point_instance_index'])
    assert g[2].eq(0).all() and relative[2].eq(0).all() and extent[2].eq(0).all()
    # Same numeric instance ID in another frame cannot change frame0 pooling.
    changed = features.detach().clone()
    changed[3:] *= 100
    other = module(changed, xyz, batch, instance)
    torch.testing.assert_close(stats['instance_features'][1], other['instance_features'][1])
    stats['instance_features'].sum().backward()
    assert features.grad[:2].abs().sum() > 0 and features.grad[3:].abs().sum() > 0
    assert features.grad[2].eq(0).all()
