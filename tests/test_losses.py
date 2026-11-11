import torch

from suction20.losses import attempt_loss, hierarchical_mean, source_losses


def test_reduction_gives_frames_equal_weight_and_ignores_empty():
    values = torch.tensor([1., 3., 10., 1000.], requires_grad=True)
    # First frame has two supervised instances, second just one.
    got = hierarchical_mean(values, torch.tensor([1, 1, 1, 0], dtype=torch.bool),
                            torch.tensor([0, 0, 1, 2]), torch.tensor([1, 2, 1, 1]))
    assert got.item() == 6.
    got.backward()
    torch.testing.assert_close(values.grad, torch.tensor([.25, .25, .5, 0.]))


def test_unknown_nan_not_negative_and_known_zero_remains_supervised():
    q0 = torch.tensor([.5, .9], requires_grad=True)
    pred = {"q0_all": q0, "q": q0[None], "normal_out_all": torch.zeros(2, 3, requires_grad=True),
            "seed_point_index": torch.tensor([[0, 1]]), "seed_valid": torch.tensor([[True, False]])}
    target = {"gt_B": torch.tensor([0., float("nan")]), "quality_valid": torch.tensor([True, False]),
              "axis_valid": torch.tensor([False, False]), "instance_id_gt": torch.ones(2, dtype=torch.long)}
    loss = source_losses(pred, target, {"batch_index": torch.zeros(2, dtype=torch.long)})
    assert torch.isfinite(loss["total"])
    loss["total"].backward()
    assert q0.grad[0] > 0
    assert q0.grad[1] == 0


def test_empty_loss_is_differentiable():
    x = torch.tensor([2.], requires_grad=True)
    loss = hierarchical_mean(x, torch.tensor([False]), torch.tensor([0]), torch.tensor([1]))
    loss.backward()
    assert loss.item() == 0 and x.grad.item() == 0


def test_attempts_average_frames_and_skip_unmapped():
    logits = torch.tensor([0., 0., 2., -100.], requires_grad=True)
    a = {"label_valid": torch.ones(4, dtype=torch.bool), "success": torch.ones(4),
         "point_cam_m": torch.zeros(4, 3), "normal_out_cam": torch.tensor([[0., 0., -1.]] * 4),
         "batch_index": torch.tensor([0, 0, 1, 2])}
    loss = attempt_loss(logits, a, torch.tensor([True, True, True, False]))
    expected = (torch.nn.functional.softplus(torch.tensor(0.)) + torch.nn.functional.softplus(torch.tensor(-2.))) / 2
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert logits.grad[3] == 0


def test_empty_frame_padding_is_never_gathered():
    q = torch.ones(1, 512, requires_grad=True)
    empty = torch.empty(0, requires_grad=True)
    pred = {"q0_all": empty, "q": q, "normal_out_all": torch.empty(0, 3, requires_grad=True),
            "seed_point_index": torch.zeros(1, 512, dtype=torch.long),
            "seed_valid": torch.zeros(1, 512, dtype=torch.bool), "seal_aux": q}
    target = {"gt_B": torch.empty(0), "quality_valid": torch.empty(0, dtype=torch.bool),
              "instance_id_gt": torch.empty(0, dtype=torch.long), "seal_target": torch.empty(0)}
    result = source_losses(pred, target, {"batch_index": torch.empty(0, dtype=torch.long)})
    result["total"].backward()
    assert result["total"] == 0 and torch.count_nonzero(q.grad) == 0


def test_target_component_can_exist_without_final_quality_label():
    value = torch.tensor([.5], requires_grad=True)
    pred = {"q0_all": value, "q": value[None], "normal_out_all": torch.zeros(1, 3, requires_grad=True),
            "seed_point_index": torch.tensor([[0]]), "seed_valid": torch.tensor([[True]]),
            "seal_aux": value[None]}
    targets = {"gt_B": torch.tensor([float("nan")]), "quality_valid": torch.tensor([False]),
               "instance_id_gt": torch.tensor([1]), "seal_target": torch.tensor([0.]),
               "seal_valid": torch.tensor([True])}
    result = source_losses(pred, targets, {"batch_index": torch.tensor([0])})
    assert result["q"].item() == 0 and result["q0"].item() == 0
    result["total"].backward()
    assert value.grad.item() == .5
