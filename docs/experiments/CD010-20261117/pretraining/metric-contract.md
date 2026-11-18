# Research: Final training analysis metric contract

- Query: Establish exact E0–E5 ablations, validation denominators, best-checkpoint rule, label semantics and sample budget for the final training report.
- Scope: internal; frozen local code and existing recipe documentation only. No training, inference, remote commands or tests.
- Date: 2026-11-16 (Asia/Shanghai)
- Source root: the frozen model repository tree `1d4cc490b78999a43dca8863ab043b7e9a1b7d42` (all source-code paths below are repository-relative references, not files bundled in this analysis directory).

## Findings

### Experiment definitions and interpretation

Authoritative configuration expansion: `suction20/training.py:19–46`.

| Experiment | Quality target | Instance feature fusion | Contact grouping | Auxiliary physical heads |
|---|---|---|---|---|
| E0 | gt_A | off | off | off |
| E1 | gt_B | off | off | off |
| E2 | gt_B | on | off | off |
| E3 | gt_B | off | on | off |
| E4 | gt_B | on | on | off |
| E5 | gt_B | on | on | on |

`gt_A = S × C`, `gt_B = S × W × C`, where S is frozen normalized seal quality, W is wrench quality and C is collision-free indicator (`docs/DATA_SCHEMA.md:40–41`, `docs/implementation/DATA_ADAPTER.md:96`). E0 isolates training-target change relative to E1; E1/E2 contrasts instance fusion; E1/E3 contrasts contact grouping; E3/E4 contrasts instance fusion with contact grouping; E4/E5 contrasts auxiliary heads but includes the batch/hardware confound below. E0 output must not be relabeled q_B. These are geometric quality proxies, not calibrated physical-success probabilities (`docs/implementation/TRAINING.md:28,80`).

Recipe: `docs/SELECTED_PRETRAINING_RECIPE.md:30–70`:
- E0–E4 L20, B1 × accumulation12 = effective12, 19,200 successful updates.
- E5 H20, B4 × accumulation12 = effective48, 4,800 successful updates.
- E2 uses BF16 dense / FP32 sparse; others FP16 training. All AMP sparse inference runs in FP32, dense heads retain configured AMP (`models/backbones/spconv_resunet14.py:104–125`).
- All total budgets 230,400 effective frame presentations; independently initialized seed42. Same peak learning rate 1e-3 without linear batch scaling.
- E5 has one-quarter the optimizer update count, a different microbatch, gradient-noise/optimization trajectory and hardware. E2 has a different training precision policy. Attribute differences to the achieved recipes; do not claim fully isolated causal module effects or statistical significance. A single seed per condition provides no seed-variance estimate. No reruns are required or proposed by this note.

### Training loss

`source_losses`, `suction20/losses.py:8–85`:
- q0: dense point quality SmoothL1(beta=.05).
- q: selected seed quality SmoothL1(beta=.05).
- normal: 1 − dot(predicted outward unit axis, label).
- E5 adds seal SmoothL1, wrench SmoothL1 and collision-free BCE-with-logits.
- Weighted total = q0 + q + 2×normal + .5×seal + .25×wrench + .1×collision_free, with absent terms omitted (unless actual config overrides weights).
- Hierarchical reduction averages points within valid foreground instance, instances within frame, then frames; empty/unknown supervision is excluded (`losses.py:12–41`).
- Total loss is not directly comparable between E0 and gt_B experiments, nor E5 and experiments without auxiliary losses. Within-run curves and same-target components are useful. Training loss and validation quality errors use different sampling/targets, so do not compute a literal train–validation gap from mismatched quantities.

### Dataset split, epoch and validation schedule

`datasets.py:40–68`, `training.py:95–143`, `pretrain_budget.py:11–13,35–68,93–97`:
- Train scenes: scene0…99 except IDs ending in9 =90 scenes.
- Each sampling epoch draws64 of256 views per scene without replacement, shuffles all 5,760 frames, then draws a new deterministic subset/order next epoch.
- Thus 40 credited epochs =230,400 effective presentations, NOT 40 complete traversals of all48,640 dataset frames, and not230,400 distinct frames.
- Dev scenes9,19,…99 =10 scenes; fixed views0,8,…248 =32/scene,320 total per validation. Official test scenes100…189 are disjoint and not part of periodic validation.
- A successful optimizer update credits effective batch samples; AMP skipped updates consume frames without crediting the budget. Physical consumption may exceed230,400. Skips also do not advance the credited epoch used for sampling regime.
- Validation every23,040 credited samples (=4 credited epochs),10 nodes including final budget. L20 nodes every1,920 updates; E5 every480. Warmup11,520 credited samples (=960 or240 updates).
- Use effective samples or credited epoch on cross-model plot x-axis, not raw optimizer steps.

### Pose metrics: exact denominators

`validation.py:120–179`, `metrics.py:13–119`, `eval_r20.py:124–171`:
- Inference uses observation inputs only (`validation.py:71–104`), then evaluates actual predicted point AND normal via frozen geometry. It does not substitute nearest cached gt_B and does not filter final predictions by cached gt_valid.
- Q20 = recomputed normalized seal S × wrench W × collision-free C.
- NMS uses both spatial distance<10mm and directed angle<15degrees to suppress within input instance. Retain max10 per input instance and max50 per scene/view. Ranking is descending predicted q, ties by q0 then point index. Input instance masks determine NMS grouping; evaluation ownership uses GT instance at selected observation point.
- **AP_R20 is not conventional precision–recall area.** Per frame, for thresholds .2,.4,.6,.8, compute precision@k for each k1…50 and average the200 values. Missing ranked candidates are failure/zero slots, denominator remains k, background predictions count as failures (`metrics.py:54–62,81–90`). This quantity can be expressed as percent but must retain its project-specific name/formula.
- **instance_Top1_Q20_at_0_4:** find highest-ranked candidate belonging to each visible GT instance BEFORE the scene top50 cap; success if its actual Q20≥.4. Denominator is ALL visible GT instances in the observation, including those with no candidate; missing candidates are failures (`metrics.py:65–97`). This is per-instance Top1, not one global top pick per image.
- **collision_rate:** fraction of selected per-instance Top1 candidates colliding; denominator includes only instances that HAVE a selected candidate. It is not collision among all512 seeds or scene top50. Always show no_candidate_rate alongside it.
- **no_candidate_rate:** (visible GT instances − instances with selected Top1 candidate) / visible GT instances. No visible instances → undefined, not zero.
- **Aggregation:** average defined frame values inside each scene, then average scenes equally (`metrics.py:100–119`). This is not pooled-instance micro averaging. Count fields in aggregate dictionaries are ALSO average per-frame counts, not dataset totals. scene_count/view_count are explicit total counts. Do not sum or label mean count fields as totals without reading frame records.

### Secondary diagnostics and limitations

`validation.py:151–170`, `metrics.py:122–180`:
- masked_seed_q_mae compares q with cached target at model-selected seed points where labels are valid, averaged equally across foreground instances within each frame. Its seed subset changes with model and training, so reduced seed MAE does not by itself demonstrate improved actual-pose performance. E0 target is gt_A; others gt_B.
- masked_dense_q0_mae compares q0 with target at all fixed labeled observation points. More suitable cross-time/cross-gt_B-model regression diagnostic than changing seed MAE, but still not a metric of newly predicted axis quality.
- normal_angle_median_deg is per-frame median directed angle between dense prediction and axis label on axis_valid points, then scene aggregation. Final number is a mean of per-frame medians, NOT pooled global median. Similarly p90 is a mean of per-frame p90 values, not pooled p90. Normal metric uses the valid normal subset, not only Top1 candidates.
- proposal_coverage_at_512: among GT instances having at least one observed cached gt_valid & gt_B≥.4 point, fraction whose512 proposal seeds include at least one such point. It is proposal/cached-label coverage, not success of final predicted poses. Different denominator from instance Top1.
- Distinguish development-set geometric results from held-out official-test performance and physical validation. Reusing these320 frames for checkpoint selection means selected-best validation metrics are selection-biased estimates of generalization.

### Checkpoint selection and retention

`train_pretrain.py:303–319`:
- Best checkpoint selection is lexicographic `(instance_Top1_Q20_at_0_4, AP_R20)` with strict >. Primary objective Top1; AP breaks exact Top1 ties. Exact pair ties retain earlier checkpoint. Best is NOT the AP maximum, lowest loss or latest checkpoint by definition.
- Publish after validation result persisted; validation records include global step and experiment. Distinguish final vs selected best in tables and recommend based on declared primary criterion. Do not mix independent metric maxima into a fictitious single checkpoint.

`suction20/pretrain_runtime.py:193–249`:
- Manifest is authoritative commit; it records current, predecessor, best and retained fixed validation nodes with blob path, size, SHA256 and step.
- last.pt and best.pt are repairable aliases; downloads must dereference them or retain matching checkpoint blobs/manifests. A tar of only symlinks can appear valid but be unusable after extraction.
- Unreferenced ordinary blobs are pruned; all referenced current/predecessor/best/fixed-node blobs pinned. It is inaccurate to promise every periodic ordinary checkpoint still exists.
- Checkpoint full state contains model/config, optimizer, scheduler, scaler, sampler, RNG, source-label config/hash, split, best metric and code identity (`training.py:233–251`). Logs and validation JSON can be downloaded independently without the larger tensor checkpoints.

## Files found

- `suction20/training.py` — E0–E5 contract, frame schedule, full checkpoint schema.
- `suction20/losses.py` — loss definitions, weights and hierarchy.
- `suction20/metrics.py` — AP/Top1/collision/coverage/error formulas and aggregation.
- `suction20/validation.py` — observation-only inference and periodic metric assembly.
- `suction20/eval_r20.py` — actual-pose S/W/collision recomputation.
- `suction20/datasets.py` — scene splits, sampled frame lists and input/target adapter.
- `suction20/pretrain_budget.py` — credited-sample epoch/budget/schedule.
- `suction20/pretrain_runtime.py` — immutable checkpoint publication and retention.
- `scripts/train_pretrain.py` — validation transaction and best promotion.
- `models/backbones/spconv_resunet14.py` — AMP sparse/dense precision boundary.
- `docs/SELECTED_PRETRAINING_RECIPE.md` — selected hardware/runtime/optimizer setup.
- `docs/performance/CD010_SELECTED_TRAINING_START.md` — original launch recipe and output organization.

## External references

None needed: this report defines the implemented local contract, not a literature comparison. spconv2.3.6 behavior is documented in the frozen local compatibility comments; no external claims added.

## Related specs

Read `.trellis/workflow.md`, `.trellis/spec/backend/index.md`, `.trellis/spec/guides/index.md`. Backend specs are unfilled templates, so authoritative behavioral facts were taken from frozen code plus selected recipe. No specs modified.

## Caveats / Not Found

- This bounded research did not independently read current final remote logs/checkpoint manifests or execute final metrics. Numerical final tables and completeness checks belong to the main session's downloaded snapshot.
- Existing old runtime docs may show superseded batch4/8 recipes; use selected recipe plus actual final config/manifest, not historical reports.
- No additional experiment/test/inference requested or performed, and no claim of statistical significance, official-test evaluation, calibrated success probability or physical suction success is established.
