"""Ordered CPU ticket production with separate issued and consumed schedules."""
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
import os
from time import perf_counter

from .batching import collate_public
from .datasets import frame_key


@dataclass(frozen=True)
class SampleTicket:
    sequence_id: int
    dataset_index: int
    frame_key: tuple
    augmentation_seed: int
    sample_epoch: int


def clone_schedule(state):
    from .training import FrameSchedule
    schedule = FrameSchedule(state['scene_ids'], seed=state['seed'],
                             views_per_scene=state['views_per_scene'], frame_keys=state['frame_keys'])
    schedule.load_state_dict(deepcopy(state))
    return schedule


class TicketDataset:
    def __init__(self, source_dataset):
        self.source = source_dataset

    def __len__(self):
        return len(self.source)

    def __getitem__(self, ticket):
        started = perf_counter()
        sample = self.source.get_sample(ticket.dataset_index, ticket.augmentation_seed)
        return {'sample': sample, 'ticket': ticket, 'prepare_seconds': perf_counter() - started}


def collate_tickets(items, voxel_size_m=.002):
    started = perf_counter()
    packed = collate_public([item['sample'] for item in items], voxel_size_m)
    # This transport is training-only. General get_sample/collate retain all meta.
    packed.pop('meta', None)
    packed['tickets'] = [item['ticket'] for item in items]
    timing = packed.setdefault('production_timing', {})
    timing['producer_total_s'] = sum(item['prepare_seconds'] for item in items) + perf_counter() - started
    packed['worker_pid'] = os.getpid()
    packed['cache_info'] = [item['sample'].get('cache_info', {}) for item in items]
    return packed


class TicketBatchSampler:
    def __init__(self, start_state, start_sequence_id, microbatch, key_to_index, camera='realsense'):
        if type(microbatch) is not int or microbatch < 1:
            raise ValueError('Ticket microbatch must be a positive integer')
        self.start_state = deepcopy(start_state)
        self.start_sequence_id = int(start_sequence_id)
        self.microbatch = microbatch
        self.key_to_index = dict(key_to_index)
        self.camera = camera
        self.issued_schedule = None

    def __iter__(self):
        issued = clone_schedule(self.start_state)
        self.issued_schedule = issued
        sequence = self.start_sequence_id
        while True:
            tickets = []
            for _ in range(self.microbatch):
                key, seed = issued.next()
                key = frame_key(key, self.camera)
                tickets.append(SampleTicket(sequence, self.key_to_index[key], key, seed, issued.epoch))
                sequence += 1
            yield tickets


def worker_init_cpu_only(worker_id):
    import torch
    # The launcher sets these before Python imports NumPy/Torch. Repeat them for
    # independently launched CPU diagnostics; spawned workers never initialize CUDA.
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = '1'
    torch.set_num_threads(1)


def make_ticket_loader(dataset, consumed_schedule, *, start_sequence_id, microbatch=1,
                       voxel_size_m=.002, loader_config=None, seed=20261110):
    import torch
    from torch.utils.data import DataLoader

    settings = dict(num_workers=2, prefetch_factor=2, persistent_workers=True,
                    pin_memory=True, multiprocessing_context='spawn')
    settings.update(loader_config or {})
    if settings['multiprocessing_context'] != 'spawn':
        raise ValueError('CPU training prefetch requires spawn workers')
    count = settings['num_workers']
    if type(count) is not int or count < 0:
        raise ValueError('Loader worker count must be a nonnegative integer')
    if count:
        # Set before spawn imports NumPy/BLAS, not merely inside worker_init_fn.
        # The main process libraries are already initialized; the launcher uses
        # the same values. Its Torch intra-op thread count is left unchanged.
        for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
            os.environ[name] = '1'
    camera = dataset.keys[0][1] if len(dataset.keys[0]) == 3 else 'realsense'
    indices = {frame_key(key, camera): i for i, key in enumerate(dataset.keys)}
    sampler = TicketBatchSampler(consumed_schedule.state_dict(), start_sequence_id,
                                 microbatch, indices, camera=camera)
    generator = torch.Generator(device='cpu')
    generator.manual_seed(seed)
    options = dict(num_workers=count, pin_memory=settings['pin_memory'], generator=generator,
                   worker_init_fn=worker_init_cpu_only)
    if count:
        options.update(prefetch_factor=settings['prefetch_factor'],
                       persistent_workers=settings['persistent_workers'], multiprocessing_context='spawn')
    return DataLoader(TicketDataset(dataset), batch_sampler=sampler,
                      collate_fn=partial(collate_tickets, voxel_size_m=voxel_size_m), **options)


def consume_tickets(schedule, tickets, consumed_count):
    """Call only after this microbatch's backward; save this schedule, not issued."""
    for ticket in tickets:
        key, seed = schedule.next()
        if (ticket.sequence_id != consumed_count or frame_key(key, ticket.frame_key[1]) != ticket.frame_key or
                seed != ticket.augmentation_seed or schedule.epoch != ticket.sample_epoch):
            raise ValueError('Prepared ticket differs from the next consumed frame/augmentation seed')
        consumed_count += 1
    return consumed_count


def close_ticket_loader(loader, iterator=None):
    """Release persistent workers on normal pause/end; prefetched tickets are discarded."""
    active = iterator if iterator is not None else getattr(loader, '_iterator', None)
    if active is not None and hasattr(active, '_shutdown_workers'):
        active._shutdown_workers()
    # Torch 2.5 has no public iterator close; do not persist the issued iterator.
    if hasattr(loader, '_iterator'):
        loader._iterator = None
