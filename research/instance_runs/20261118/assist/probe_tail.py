"""Exercise H20's actual cache-reuse path on one complete L20 frame."""
from pathlib import Path
import os
import json
import sys
import tempfile

base=Path('/home/ubuntu/work/instance-routes-20261118')
repo=base/'A/repo'
sys.path.insert(0,str(repo))
from suction20.instance_data import _build_source_instances
cache=Path('/home/ubuntu/data/datasets/pose/instance-routes-20261118/train')
manifest=Path(sys.argv[1])
rows=[json.loads(line) for line in manifest.read_text().splitlines()]
assert rows
scene,frame=rows[0]['scene_id'],rows[0]['frame_id']
original,=(cache/'parts').glob(f'shards/realsense/train/*/*/shards/*/shards/realsense/train/scenes_{scene:04d}/frames_0000-0255')
src=Path('/home/ubuntu/data/datasets/pose/suctionnet-r20-v1/full-20261110')
with tempfile.TemporaryDirectory(dir=base) as dirname:
    folder=Path(dirname)
    for row in rows:
        assert (row['scene_id'],row['frame_id'])==(scene,frame)
        for key in ('observation_npz','label_cache'):
            path=original/row[key]
            assert path.is_symlink() and path.resolve()==(manifest.parent/row[key]).resolve()
            new=folder/row[key]
            new.parent.mkdir(exist_ok=True)
            new.symlink_to(path)
    result,info=_build_source_instances(src/'input_raw',src/'output',folder,split='train',scenes=[scene],frames=[frame])
    print(json.dumps(dict(scene=scene,frame=frame,validated_instances=info['visible_instance_count'],links=len(rows)*2)))
