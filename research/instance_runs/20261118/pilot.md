# 2026-11-18 各路线试运行记录

## 路线 A

Route A ran with the pinned external sparse backbone on H20 GPU0. The real RGB-D gradient probe had 8,188 support points and 102 backbone parameter gradients were finite. Project tests reported 706 passed and 20 skipped. The shared source pilot used 74 train instances across eight frames and 17 test instances across two frames. Two successful training updates with strict resume consumed 32 supervised instances with zero skips. A two-step target smoke exercised head training and the configured module unfreeze with two target and two replay observations per update. The raw RGB-D CLI wrote a native pose prediction. These are partial checkpoints. The end-to-end formal controller uses the original YAML.

## 路线 B

Route B ran on H20 GPU1. The 223 targeted B model, boundary, training, inference, evaluation and no-content-verification tests passed. On a real recorded training action with 8,188 support points, the forward/backward wiring probe produced BCE 0.72910 and gradient norm 1.40189. Two source updates with two supervised instances per update were used to test checkpointing and strict resume, followed by two partial target updates and frozen-checkpoint inference and evaluation. A real sparse instance with zero support returned insufficient_support. Formal training uses the original YAML including 16 source instances per update, minimum 100 support points and two_fixed_z_charts_rodrigues_z05_v1.
