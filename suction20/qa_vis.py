"""Produce quality-control PNGs for 10/20mm Seal and per-frame GT.

Requires matplotlib (optional). This script does not train models.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np


def plot_seal(r10_path: str, r20_path: str, output: str, max_points=12000):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .core import read_label
    a, b = read_label(r10_path), read_label(r20_path)
    if len(a['points']) != len(b['points']) or not np.array_equal(a.get('indices'), b.get('indices')):
        raise ValueError('10mm/20mm candidate sets differ; abort visualization')
    ids = np.arange(len(a['points']))
    if len(ids) > max_points:
        ids = np.linspace(0, len(ids)-1, max_points, dtype=int)
    points = a['points'][ids]
    fig = plt.figure(figsize=(12,5), layout='constrained')
    for j,(score,label) in enumerate([(a['scores'][ids],'10mm Seal'),(b['scores'][ids],'20mm Seal')]):
        ax = fig.add_subplot(1,2,j+1,projection='3d')
        sc=ax.scatter(points[:,0],points[:,1],points[:,2],c=score,s=2,vmin=0,vmax=1,cmap='viridis')
        ax.set_title(label);fig.colorbar(sc,ax=ax,shrink=.5)
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=130);plt.close(fig)


def plot_view(gt_path: str, output: str, height: int, width: int):
    from PIL import Image
    with np.load(gt_path,allow_pickle=False) as f:
        idx = f['source_indices'].astype(int)
        valid = f['gt_valid'].astype(bool)
        scores = f['gt_B']
        color = np.zeros((height*width,3),dtype=np.uint8)
        color[:]=[38,39,45]                     # unobserved
        color[idx]=[95,95,100]                 # unmatched or background
        color[idx[valid]] = np.stack([255*(1-scores[valid]),
                                     240*scores[valid],
                                     20*np.ones(valid.sum())],axis=1).astype(np.uint8)
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(color.reshape(height,width,3)).save(output)


def main():
    ap=argparse.ArgumentParser()
    sub=ap.add_subparsers(dest='mode',required=True)
    a=sub.add_parser('seal');a.add_argument('--r10',required=True);a.add_argument('--r20',required=True)
    a.add_argument('--output',required=True)
    b=sub.add_parser('view');b.add_argument('--gt',required=True);b.add_argument('--output',required=True)
    b.add_argument('--height',required=True,type=int);b.add_argument('--width',required=True,type=int)
    args=ap.parse_args()
    if args.mode=='seal':plot_seal(args.r10,args.r20,args.output)
    else:plot_view(args.gt,args.output,args.height,args.width)

if __name__=='__main__':main()
