"""Small synthetic integration tests. Not real SuctionNet G0/G1/G2 acceptance."""
import json
import numpy as np
from pathlib import Path
from PIL import Image
from scipy.io import savemat

from suction20.full_data import (
    DEFAULT_CONFIG, load_seal, match_instance, wrench_scores, compute_scene,
    generate_view, dataset_stats, config_from_file,
)
from suction20.full_cli import range_spec, fit_normalization


def make_fixture(tmp_path):
    dataset = tmp_path / 'data'
    root = dataset / 'scenes' / 'scene_0000'
    cam = root / 'realsense'
    for sub in ('depth','label','meta'):
        (cam / sub).mkdir(parents=True, exist_ok=True)
    (root / 'object_id_list.txt').write_text('0\n')
    K = np.array([[120.,0,5.5],[0,120.,5.5],[0,0,1.]])
    pose = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,1.]])
    for frame in (0,1):
        Image.fromarray(np.full((12,12),1000,dtype=np.uint16)).save(cam/'depth'/f'{frame:04d}.png')
        Image.fromarray(np.ones((12,12),dtype=np.uint8)).save(cam/'label'/f'{frame:04d}.png')
        savemat(cam/'meta'/f'{frame:04d}.mat',{'intrinsic_matrix':K,'factor_depth':np.array([[1000]]),
                                             'cls_indexes':np.array([[1]]),'poses':pose[:,:,None]})
    np.save(cam/'camera_poses.npy',np.repeat(np.eye(4)[None],2,axis=0))
    np.save(cam/'cam0_wrt_table.npy',np.eye(4))
    seal_root=tmp_path/'seal'
    seal_root.mkdir()
    pts=np.array([[-.010,0,0],[0,0,0],[.010,0,0]],np.float32)
    normals=np.tile(np.array([[0,0,-1]],np.float32),(3,1))
    np.savez_compressed(seal_root/'000_labels.npz',points=pts,normals=normals,
                        scores=np.array([.8,0,.5],np.float32), collision=np.array([False,False,False]),
                        indices=np.array([0,1,2],np.int32))
    dense=dataset/'dense_point_clouds';dense.mkdir(parents=True)
    xs=np.linspace(-.06,.06,31);xx,yy=np.meshgrid(xs,xs)
    pp=np.stack([xx,yy,np.zeros_like(xx)],axis=-1).reshape(-1,3)
    np.savez_compressed(dense/'000.npz',points=pp,normals=np.tile([0.,0.,-1.],(len(pp),1)))
    out=tmp_path/'out'
    cfg=dict(DEFAULT_CONFIG)
    cfg['table_width_m']=cfg['table_depth_m']=.04
    cfg['match_radius_m']=.006
    cfg['table_grid_m']=.01
    return dataset, seal_root, out, cfg


def test_stage2_and_view_gt(tmp_path):
    dataset, seal_root, out, cfg=make_fixture(tmp_path)
    result=compute_scene(dataset,seal_root,out,0,cfg)
    assert result['objects'][0]['n']==3
    r=generate_view(dataset,seal_root,out,0,0,cfg)
    assert r['matched']>0
    path=out/'scenes/scene_0000/realsense/graspness_r20_v2/0000.npz'
    with np.load(path) as f:
        v=f['gt_valid'];cid=f['candidate_id']
        assert f['gt_normal'].shape == (len(f['xyz']),3)
        assert f['gt_normal_pca'].shape == f['gt_normal'].shape
        assert f['normals_raw'].shape == f['gt_normal'].shape
        np.testing.assert_allclose(f['gt_normal'][f['gt_valid']],
                                   np.tile([0., 0., -1.], (int(f['gt_valid'].sum()), 1)), atol=1e-6)
        assert np.all(cid[~v]==-1)
        assert np.all(np.isfinite(f['gt_B']))
        assert np.all(f['gt_B'][v] <= f['gt_A'][v]+1e-7)
        assert np.all(f['instance_id']==1)
        assert np.all(f['class_id']==1)
        assert len(set(cid[v]))>=2
        assert np.any(v & (f['gt_seal_raw']==0)), 'confirmed negatives must be preserved'
    stats=dataset_stats(out,cfg,[0],[0])
    assert stats['frame_count']==1 and stats['matched']>0 and not stats['g2_errors']


def test_no_false_zero_gt_for_unmatched(tmp_path):
    dataset, seal_root, out, cfg=make_fixture(tmp_path)
    cfg['match_radius_m']=.00001
    compute_scene(dataset,seal_root,out,0,cfg)
    generate_view(dataset,seal_root,out,0,0,cfg)
    with np.load(out/'scenes/scene_0000/realsense/graspness_r20_v2/0000.npz') as x:
        unknown=~x['gt_valid']
        assert unknown.sum()>0
        assert np.all(x['candidate_id'][unknown]==-1)


def test_normal_consistency_and_no_max_quality_copy():
    cfg=dict(DEFAULT_CONFIG)
    cfg['match_radius_m']=.005
    p=np.array([[0,0,0]],np.float32)
    n=np.array([[0,0,1]],np.float32)
    cp=np.array([[.001,0,0],[.004,0,0]],np.float32)
    cn=np.array([[0,0,1],[0,0,1]],np.float32)
    ids,dist=match_instance(p,n,np.array([True]),cp,cn,cfg)
    assert ids.tolist()==[0], 'nearest must win even when farther candidate might score better'
    assert np.allclose(dist,[.001])
    # Without valid camera normal, should be unknown rather than zero GT.
    ids,_=match_instance(p,n,np.array([False]),cp,cn,cfg)
    assert ids.tolist()==[-1]


def test_wrench_invariance_under_joint_rigid_transform():
    cfg=dict(DEFAULT_CONFIG)
    p=np.array([[.1,0,0],[0,.2,0]])
    n=np.array([[0,0,1],[0,0,1]])
    c=np.array([0,0,0])
    a=wrench_scores(p,n,c,1,cfg)
    # Translations should not change gravitational torque.
    b=wrench_scores(p+[3,2,1],n,c+[3,2,1],1,cfg)
    assert np.allclose(a,b)


def test_range_and_norm_freeze(tmp_path):
    dataset, seal_root, out, cfg=make_fixture(tmp_path)
    assert range_spec('0-2,5', 190)==[0,1,2,5]
    assert range_spec('0:3', 190)==[0,1,2]
    fit=fit_normalization(dataset,seal_root,cfg,out,train_scene_ids=[0])
    assert fit['n_candidates']==3
    assert (out/'config_frozen.json').is_file()
    frozen=config_from_file(out/'config_frozen.json')
    assert frozen['seal_norm_max']>frozen['seal_norm_min']


def test_full_cli_scene_view_stats_smoke(tmp_path):
    import subprocess
    import sys
    dataset, seal_root, out, cfg = make_fixture(tmp_path)
    config = tmp_path / 'cfg.json'
    config.write_text(json.dumps(cfg), encoding='utf-8')
    project_root = Path(__file__).resolve().parents[1]
    for stage in ('scene', 'views', 'stats'):
        cmd = [sys.executable, '-m', 'suction20.full_cli', stage,
               '--dataset', str(dataset), '--output', str(out),
               '--config', str(config), '--scenes', '0', '--frames', '0',
               '--objects', '0', '--seal-root', str(seal_root)]
        subprocess.run(cmd, check=True, cwd=project_root, capture_output=True, text=True)
    report = json.loads((out / 'reports' / 'stage4_g2_audit.json').read_text())
    assert report['frame_count'] == 1
    assert not report['g2_errors']
