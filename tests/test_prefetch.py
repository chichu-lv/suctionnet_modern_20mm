"""A small ordered spawn-loader comparison and normal consumed-state resume."""
import copy
import time

import numpy as np
import torch

from suction20.datasets import derive_masks
from suction20.prefetch import close_ticket_loader, consume_tickets, make_ticket_loader
from suction20.training import FrameSchedule


class ToySource:
    keys = [(0, 'realsense', i) for i in range(4)]

    def __len__(self):
        return len(self.keys)

    def get_sample(self, index, augmentation_seed):
        # Normal unequal preparation times; DataLoader still returns ticket order.
        time.sleep(.002 * (3 - index))
        rng = np.random.default_rng(augmentation_seed)
        inputs = dict(xyz_cam_m=rng.normal(size=(12, 3)).astype(np.float32) * .01,
                      normal_obs=np.tile(np.array([0., 0., -1.], np.float32), (12, 1)),
                      normal_obs_valid=np.ones(12, bool), instance_id_input=np.ones(12, np.int64),
                      source_indices=np.arange(12, dtype=np.int64), g_cam=np.array([0., 0., -1.], np.float32))
        targets = dict(gt_A=np.zeros(12, np.float32), gt_valid=np.ones(12, bool),
                       instance_id_gt=np.ones(12, np.int64), normal_label=inputs['normal_obs'].copy())
        derive_masks(inputs, targets)
        return dict(inputs=inputs, targets=targets, meta=dict(sample_uid=self.keys[index],
                    augmentation_seed=augmentation_seed))


def test_spawn_prefetch_matches_serial_and_resumes_consumed_state():
    source = ToySource()
    serial = FrameSchedule([0], seed=31, frame_keys=source.keys)
    consumed = FrameSchedule([0], seed=31, frame_keys=source.keys)
    serial_loader = make_ticket_loader(source, serial, start_sequence_id=0,
                                      loader_config={'num_workers': 0, 'pin_memory': False})
    loader = make_ticket_loader(source, consumed, start_sequence_id=0,
                                loader_config={'num_workers': 2, 'pin_memory': False})
    rng = torch.get_rng_state().clone()
    serial_iter, iterator = iter(serial_loader), iter(loader)
    assert torch.equal(torch.get_rng_state(), rng)
    count = 0
    try:
        for _ in range(3):
            expected, actual = next(serial_iter), next(iterator)
            assert actual['tickets'] == expected['tickets']
            for section in ('inputs', 'targets'):
                for key in actual[section]:
                    assert torch.equal(actual[section][key], expected[section][key])
            count = consume_tickets(consumed, actual['tickets'], count)
            consume_tickets(serial, expected['tickets'], count - len(expected['tickets']))
        assert consumed.state_dict() == serial.state_dict()
        assert consumed.state_dict() != loader.batch_sampler.issued_schedule.state_dict()
        saved = copy.deepcopy(consumed.state_dict())
        expected_next = next(serial_iter)
    finally:
        close_ticket_loader(loader, iterator)
        close_ticket_loader(serial_loader, serial_iter)
    resumed = FrameSchedule([0], seed=31, frame_keys=source.keys)
    resumed.load_state_dict(saved)
    resumed_loader = make_ticket_loader(source, resumed, start_sequence_id=count,
                                        loader_config={'num_workers': 2, 'pin_memory': False})
    iterator = iter(resumed_loader)
    try:
        actual = next(iterator)
        assert actual['tickets'] == expected_next['tickets']
        for section in ('inputs', 'targets'):
            for key in actual[section]:
                assert torch.equal(actual[section][key], expected_next[section][key])
        assert consume_tickets(resumed, actual['tickets'], count) == count + 1
        assert torch.equal(torch.get_rng_state(), rng)
    finally:
        close_ticket_loader(resumed_loader, iterator)
