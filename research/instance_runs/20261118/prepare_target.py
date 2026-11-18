"""Attach the recorded RGB image for each historical action using the original frame ID."""
import argparse
import json
from collections import Counter
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('release', type=Path)
parser.add_argument('output', type=Path)
args = parser.parse_args()
def rows(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)

frames = {}
for row in rows(args.release / 'prepared/geometry_observations.jsonl'):
    assert row['sample_id'] not in frames
    frames[row['sample_id']] = row
counts = Counter()
groups = {}
out = []
for row in rows(args.release / 'data/manifest_attempt_success.jsonl'):
    frame = frames[row['provenance']['original_sample_id']]
    for field in ('group_id', 'split'):
        assert row[field] == frame[field]
    obs = row['observation']
    for field in ('depth_path', 'instance_mask_path', 'intrinsics', 'g_cam', 'depth_scale_m'):
        assert obs[field] == frame['observation'][field], (field, row['sample_id'])
    obs['rgb_path'] = frame['observation']['rgb_path']
    assert Path(obs['rgb_path']).parent == Path(obs['depth_path']).parent
    assert row['attempt']['pose_source'] == 'planned_pipeline_contact'
    counts[row['split']] += 1
    assert groups.setdefault(row['group_id'], row['split']) == row['split']
    out.append(json.dumps(row, ensure_ascii=False))
assert counts == {'train': 1676, 'val': 399, 'test': 87}, counts
args.output.parent.mkdir(parents=True, exist_ok=True)
tmp = args.output.with_suffix('.tmp')
tmp.write_text('\n'.join(out) + '\n')
tmp.replace(args.output)
print(dict(counts), dict(Counter(groups.values())))
