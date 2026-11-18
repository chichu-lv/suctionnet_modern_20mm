"""Read completed route outputs on H20. Export only aggregate fields."""
import json
import sys
from pathlib import Path

route = sys.argv[1]
root = Path('/home/ubuntu/work/instance-routes-20261118')
if route == 'A':
    source = root/'A/runs/source100'
    target = root/'A/runs/finetune'
    full_source = source/'source_full_test.json'
    full_target = target/'target_full_test.json'
    timing = target/'timing'
elif route == 'B':
    source = root/'B/run/source'
    target = root/'B/run/target'
    full_source = source.parent/'source-test.json'
    full_target = target.parent/'target-test.json'
    timing = target.parent/'timing'
else:
    raise ValueError(route)

def read(path):
    with path.open() as stream:
        return json.load(stream)

def stage(folder, steps):
    summary = read(folder/'summary.json')
    if not summary['completed_budget'] or summary['completed_steps'] != steps:
        raise ValueError(f'Budget incomplete: {folder}')
    evaluations = [read(p) for p in sorted(folder.glob('evaluation_step*.json'))]
    winners = [report for report in evaluations if report['best_updated']]
    best = winners[-1]
    return dict(steps=steps, best_step=best['step'], best_selection_metrics=best['metrics'],
                evaluations=[dict(step=r['step'], metrics=r['metrics'], seconds=r['timing']) for r in evaluations],
                supervised_source_instances=summary['supervised_source_instances'],
                supervised_target_observations=summary['supervised_target_observations'],
                checkpoints={k: str((folder/(k+'.pt')).relative_to(root)) for k in ('best','last')})

def main():
    time_result = read(timing/'summary.json')
    report = dict(route=route, source=stage(source,8750), finetune=stage(target,1000),
                  source_test=read(full_source)['metrics'], target_test=read(full_target)['metrics'],
                  source_selection='test_selected/fixed_test_subset', independent_test_estimate=False,
                  timing={k:time_result[k] for k in ('measured_instances','warmup_count','device','dtype','model_ms','e2e_ms','raw_preprocessing_ms','peak_allocated_bytes','no_pose','model_scope','e2e_scope')})
    if report['source_test']['frame_count'] > 23040 or report['target_test']['total_attempts'] != 87:
        raise ValueError('Unexpected reporting scope')
    coverage = {}
    for phase in ('train','selection','test'):
        stamp = read(root/'source-logs'/(phase+'.completed'))
        manifest = read(root/'source-logs'/(phase+'.json'))
        coverage[phase] = {k:manifest[k] for k in ('frame_count','visible_instance_count','status_counts')}
        coverage[phase]['elapsed_s'] = stamp.get('elapsed_s')
    report['cache'] = coverage
    print(json.dumps(report, indent=2, allow_nan=False))

if __name__ == '__main__':
    main()
