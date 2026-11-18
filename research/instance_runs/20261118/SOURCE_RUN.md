# Shared RealSense source execution — 2026-11-18

Route A commit 364132a5f892745c9fff05e6ed4a33b173a11ea5 and route B commit f30e06ce6b25a35a5911f1044d66fdceb5f85cf7 have identical source builder, merger, dataset, normal/sampler, teacher and R20 evaluation implementations. H20 runs one writer workflow. Both route trainers consume its published manifests.

On the GPU hosts, raw inputs are under `/home/ubuntu/data/datasets/pose/suctionnet-r20-v1/full-20261110/input_raw` and frozen R20 assets under the sibling `output` directory. There are 88 dense object archives and 88 Poisson R20 label archives. All 190 scene object lists contain unique classes within each scene. Scene and MAT row-order checks remain active at build time.

| Real pilot | Frames | Instances | Label validity | Support range | Depth range | Observation + label bytes |
|---|---:|---:|---:|---:|---:|---:|
| Train, scenes 0/44 at 0/64/128/192 | 8 | 74 | 9,472/9,472 | 1,767–171,930 | 0.207–0.611 m | 159,957,721 |
| Test, scenes 100/160 at 0 | 2 | 17 | 4,352/4,352 | 5,510–107,273 | 0.289–0.706 m | 27,392,488 |

Both sets used two workers each, concurrently. Time from CLI help completion to manifest publication was approximately 173 s for train and 60 s for test. The production loader read every one of these 91 instances including p/n and teacher relations. Seal self/envelope positives were 3,532/4,206 on train and 2,436/2,698 on test. A further sample from scene 6 frame 192 yielded ten visible rows of which one had insufficient support (zero usable depth). Total real pilot coverage is 101 instances. All scoped pilot outputs retain partial identities.

H20 manager and receipts are in `/home/ubuntu/work/instance-routes-20261118`. Run `bash run-source.sh` there for start/recovery. The source Python environment is `/home/ubuntu/work/cd009-pretraining-v3/venv/bin/python`. The manager uses A's worktree as its module search root, writes each invocation under `source-logs/<phase>-shards`, reuses published scene-pair receipts after interruption, and uses the official CLI to merge shards by complete frame identity. Two-scene subprocesses use two workers. Train has up to 50 build workers, selection 12, and the subsequent remaining-test stage 30. Train and selection start together; full test continues after selection publication and includes all selection rows once. The data-server release root is `/data/datasets/pose/instance-routes-20261118`. Consumers wait for `source-logs/<phase>.completed` and read the actual `manifest` field in `source-logs/<phase>.json`. Raw image data and per-instance cache files remain on the data server. The native quality definition, query ordering, support threshold and partial/full split meanings follow the branch configs.

The formal build was launched at 22:21 China time with manager PID 2454714. At the first saved progress inspection, source data had 434 train labels and 66 selection labels. Both pilot summary files, streaming dataset checks and executable recipes are in this directory. Only the real pilot has a final coverage receipt at that inspection.
