# E2 hardware adaptation and formal startup

The E2 experiment has started on an available H20-class device with the verified P2 implementation. It passed100 successful updates, normal checkpoint saving and final backup verification, then strict resume verification through the first next update. The resume audit observed113 updates; training continues. This is startup acceptance, not convergence or full validation.

The selected configuration is microbatch1, accumulation4,8 CPU threads,2 loader workers, dense chunk65536, dense recomputation disabled, and112 GiB allocator ceiling. The independent E2 scientific definition is unchanged. The budget remains230400 effective frames (40 nominal epochs),57600 successful updates,2880 warmup updates, and320 development frames every5760 successful updates.

| Same16-frame test | B1A4 selected | B2A2 |
| --- | ---: | ---: |
| Measured frames/second | 1.35289 | 1.37169 |
| Peak allocated GiB | 11.49 | 22.33 |
| Peak reserved GiB | 16.89 | 34.13 |

Each disposable trial used4 warmup and12 measured frames with identical frame/augmentation sequences, finite gradients, unchanged parameters and zero optimizer updates. The B2 aggregate throughput advantage was1.39%, with one of three measured blocks slower. This short test did not establish a stable benefit; B1A4 was retained. Both fit memory. Profiling excludes Adam updates, saving and validation; resource samples are sparse observations.

The formal100-update prefix consumed400 effective/physical frames, with0 skipped updates. Median update time was3.091 seconds, total update timing319.598 seconds. Extrapolating this early rate gives about2.1 days of update computation for the whole budget, excluding validation, checkpoint/queue overhead and later sampling-stage changes. It is not a completion-time guarantee. The resumed first update was audited as live, not yet checkpoint-committed, data.

The26 existing budget tests passed; the selected configuration and supplemental manifest hashes were checked and six historical profile files stayed unchanged. Full private evidence, process identities and raw training logs remain outside Git. The running checkout remains frozen.

- [Selected E2 profile](../../configs/runtime_profiles/cd009-v3/H20-E2.json)
- [Supplemental manifest](../../configs/runtime_profiles/cd009-v3/manifest-h20-e2-p2.json)
- [Measurement and startup summary](H20_E2_GPU1_RESULTS.json)
