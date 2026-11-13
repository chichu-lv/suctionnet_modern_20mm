import json

import pytest

from scripts.benchmark_geometry_overlap import prepare_plan, summarize_rows, summarize_plan


@pytest.mark.parametrize('hardware,experiment,microbatch,limit', [
    ('H20', 'E5', 2, 112), ('L20', 'E3', 1, 36)])
def test_plan_is_one_experiment_two_fresh_candidates(hardware, experiment, microbatch, limit, tmp_path):
    plan = prepare_plan(hardware, tmp_path, cpu_list='16-31', python='/opt/train/bin/python')
    assert plan['experiment'] == experiment
    assert len(plan['variants']) == 2
    assert plan['timed_credited_frames'] == 80
    for variant, depth in zip(plan['variants'], (1, 2)):
        config = json.loads(open(variant['train_config']).read())
        performance = json.loads(open(variant['performance_config']).read())
        assert config['experiment'] == experiment
        assert config['seed'] == 42
        assert config['training']['microbatch_size'] == microbatch
        assert config['training']['gradient_accumulation_steps'] * microbatch == 4
        assert config['training']['geometry_pipeline_depth'] == depth
        assert config['model']['pose_chunk_size'] == 32
        assert config['runtime']['gpu_memory_limit_gib'] == limit
        assert performance['dense_chunk_size_override'] == 65536
        assert performance['geometry_threads'] == 8
        assert performance['loader']['num_workers'] == 4
        assert variant['command'][:3] == ['taskset', '-c', '16-31']
        assert variant['command'][-2:] == ['--pause-at-step', '30']
        assert not (tmp_path / variant['name'] / 'run').exists()


def row(step, ended, *, updated=True, depth=2, effective_batch=4):
    return {'global_step': step, 'updated': updated, 'ended_monotonic': ended,
            'effective_samples_completed': step * effective_batch, 'elapsed_seconds': 2.,
            'loader_wait_s': .01, 'gradients_finite': updated,
            'geometry_pipeline': {'depth': depth, 'geometry_job_wall_s': 1.2,
                                  'geometry_wait_host_s': .2},
            'peak_gpu_bytes': 20 * 1024**3, 'peak_gpu_reserved_bytes': 44 * 1024**3}


def test_summary_counts_wall_skips_and_credited_frames():
    rows = [row(10, 100.)]
    # One skipped attempt costs time without contributing credited frames.
    rows.append(row(10, 102., updated=False))
    rows.extend(row(step, 102. + (step - 10) * 2.) for step in range(11, 31))
    result = summarize_rows(rows)
    assert result['credited_frames'] == 80
    assert result['wall_seconds'] == 42.
    assert result['frames_per_second'] == pytest.approx(80 / 42)
    assert result['successful_updates'] == 20
    assert result['skipped_updates'] == 1
    assert result['peak_reserved_gib'] == 44
    assert result['geometry_future_wait_seconds']['median'] == .2
    assert not result['gradients_finite']


def test_serial_future_is_not_misreported_as_zero_work():
    rows = [row(step, step * 2., depth=1) for step in range(1, 31)]
    result = summarize_rows(rows)
    assert result['geometry_job_wall_seconds'] is None
    assert result['geometry_future_wait_seconds'] is None
    assert summarize_rows(rows[:20])['measurement_status'] == 'incomplete'


def test_unobserved_process_exit_is_not_reported_as_success(tmp_path, monkeypatch):
    plan = prepare_plan('H20', tmp_path)
    monkeypatch.setattr('scripts.benchmark_geometry_overlap.summarize_run',
                        lambda _, **kwargs: {'trainer_phase': 'paused', 'frames_per_second': 1.})
    result = summarize_plan(plan, tmp_path)
    assert all(item['normal_pause'] is None for item in result['results'])


@pytest.mark.parametrize('hardware,batches,boundaries', [
    ('H20', [(2, 4, 8), (4, 2, 8)], [(5, 15), (5, 15)]),
    ('L20', [(1, 4, 4), (2, 2, 4)], [(10, 30), (10, 30)]),
])
def test_memory_suite_uses_equal_frame_windows(hardware, batches, boundaries, tmp_path):
    plan = prepare_plan(hardware, tmp_path, suite='memory', gpu_memory_limit_gib=128 if hardware == 'H20' else 36)
    assert plan['suite'] == 'memory'
    assert plan['total_credited_frames_per_variant'] == 120
    assert len(plan['variants']) == 2
    for variant, batch, (warmup, stop) in zip(plan['variants'], batches, boundaries):
        config = json.loads(open(variant['train_config']).read())
        performance = json.loads(open(variant['performance_config']).read())
        micro, accumulation, effective = batch
        assert config['training']['microbatch_size'] == micro
        assert config['training']['gradient_accumulation_steps'] == accumulation
        assert config['training']['effective_batch_size'] == effective
        assert config['training']['geometry_pipeline_depth'] == 2
        assert config['runtime']['gpu_memory_limit_gib'] == (128 if hardware == 'H20' else 36)
        assert config['training']['target_effective_samples'] == 230400
        assert config['training']['max_optimizer_steps'] == 230400 // effective
        assert config['scheduler']['warmup_optimizer_steps'] == 11520 // effective
        assert variant['warmup_successful_steps'] == warmup
        assert variant['stop_at_successful_step'] == stop
        assert warmup * effective == 40
        assert (stop - warmup) * effective == 80
        assert variant['command'][-2:] == ['--pause-at-step', str(stop)]
        assert performance['dense_chunk_size_override'] == 65536
        assert config['model']['pose_chunk_size'] == 32


def test_memory_summary_uses_variant_boundaries_not_common_update_count(tmp_path):
    plan = prepare_plan('H20', tmp_path, suite='memory')
    for variant in plan['variants']:
        effective = variant['effective_batch_size']
        stop = variant['stop_at_successful_step']
        # The H20 pair uses the same effective batch and 5→15 window, not 10→30.
        rows = [row(step, step * effective / 2, effective_batch=effective) for step in range(1, stop + 1)]
        attempt = tmp_path / variant['name'] / 'run' / 'attempts' / 'one'
        attempt.mkdir(parents=True)
        (attempt / 'updates.jsonl').write_text(''.join(json.dumps(item) + '\n' for item in rows))
    result = summarize_plan(plan, tmp_path)
    assert result['throughput_change_percent'] == pytest.approx(0)
    assert [item['successful_updates'] for item in result['results']] == [10, 10]
    assert all(item['credited_frames'] == 80 and item['frames_per_second'] == 2 for item in result['results'])
