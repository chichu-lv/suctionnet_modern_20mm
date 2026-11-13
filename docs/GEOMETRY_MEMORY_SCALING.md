# 几何重叠后的显存扩展：H20 / L20 下一档 batch

日期：2026-11-13（北京时间）。基于 `c684aae` 的[几何重叠实测](performance/CD010_GEOMETRY_OVERLAP_RESULTS.md)。

**两机都还有扩大工作集的空间；下一轮只比较更大的 microbatch，不继续增大 dense chunk，也不新增六组训练。** H20 需要先明确有效 batch 8 的训练含义，L20 可以保持有效 batch 4。已经提供可生成、可直接执行的 `--suite memory` 短测入口。本文没有新的 GPU 测量结果。

完整训练与协调器保持停止。只有用户选定最终代码和训练设置后才启动完整训练；这里的脚本只运行指定短测，不恢复旧任务或安排额外备份。

## 1. 最新结果说明了什么

| 已测设置 | 有效帧/s | 更新中位数 | allocated / reserved 峰值 GiB | future 等待中位数 |
|---|---:|---:|---:|---:|
| H20/E5，B2A2，depth2 | 1.842032 | 2.1626 s | 45.28 / 83.11 | 0.2645 s |
| L20/E3，B1A4，depth2 | 2.841787 | 1.3808 s | 11.72 / 22.94 | 0.0739 s |

相对各自串行基线，吞吐分别提高19.90%与33.82%。这是已经取得的收益，不再要求重做串行/流水线比较。上述future等待是主线程等待CPU结果的时间，不是精确GPU空闲时间；按中位数比较，大约相当于更新耗时的12%与5%。因此继续优化应关注更大的GPU工作批次及接触头效率，不能假定再隐藏一次几何等待还能获得同样的幅度。

显存容量可支持更大的在途激活，但不保证吞吐随占用增长。当前checkpoint_dense已经关闭，64k→128k dense chunk又只测出0.50%收益，不重复这两项。

## 2. H20 不能只把 depth2 改成 depth4

当前H20是B2A2：两份microbatch、每份两帧，depth2已经包含这次更新的全部任务。更深的队列不会产生第三份任务；跨optimizer更新提前编码会使用旧参数，本实现不这样做。

如果保持有效batch4：

- B2A2/depth2已有最多四帧图同时驻留。
- B4A1只有一份任务，无法使用当前跨microbatch几何重叠。

要实质增加H20的在途工作集，下一档是 **B4A2/depth2、有效batch8**，最多八帧图同时驻留。L20则可从B1A4/depth2的最多两帧增加到 **B2A2/depth2** 的最多四帧，仍保持有效batch4。

按当前allocated粗略线性估算，H20下一档的活跃工作集可能接近90 GiB，L20接近23 GiB。这个估算只用于选择下一档；spconv工作区、反向暂存和分配器会改变真实峰值，**不能把reserved83.11乘二并据此判定OOM，也不能保证下一档一定能放下。** 本轮默认沿用112/36 GiB配置上限，实测决定。

## 3. 只测两个代表实验，H20 对照保持相同有效 batch

| 主机 / 实验 | 对照 | 更大microbatch候选 | 共同有效batch | depth | 理论最多在途帧 |
|---|---|---|---:|---:|---|
| H20 / E5 | **B2A4** | **B4A2** | **8** | 2 | 4 → 8 |
| L20 / E3 | **B1A4** | **B2A2** | **4** | 2 | 2 → 4 |

H20额外显存的收益应在同样的有效batch8下比较，所以对照是B2A4，而不是直接拿有效batch4的B2A2当同条件对照。两者每八帧都只做一次AdamW，避免将“optimizer更新频率减半”当作“更大的GPU microbatch更快”。已测B2A2/有效batch4的1.842032帧/s仍可作为当前方案的历史参考，但不是同窗口配对结果。

两机各只有这两个设置，没有新增实验类型。共同固定：

- dense chunk65,536、pose chunk32。
- geometry8、CPU affinity16、Torch8、workers4。
- depth2、FP16、FP32 geometry、原有segment/共享体素头适用规则。
- frozen数据、20mm半径、2mm体素、512候选，seed42。
- allocator上限：H20 112 GiB、L20 36 GiB。

本轮不同时改pose chunk，以便直接回答更大的microbatch是否值得。pose128已有CLI入口，只有更大batch没有净收益且需要改GPU接触头时才另选这一项，不自动执行。

## 4. H20 有效 batch 8 的训练含义与实现

**有效batch8是新的训练设置，不是仅改变浮点归约顺序。** 与现有有效batch4相比，它降低每样本的optimizer更新频率；microbatch2→4也改变训练期BatchNorm分组。当前只确认性能，不据短窗口loss判断收敛优劣，也不因此增加六组训练或额外完整验证。

配置生成器新增显式`--effective-batch-size`，仍保留230,400个有效样本及按样本定义的warmup/验证间隔：

| 预算 | 有效batch4 | 有效batch8 |
|---|---:|---:|
| 总有效样本 | 230,400 | 230,400 |
| 成功更新数 | 57,600 | 28,800 |
| warmup有效样本 | 11,520 | 11,520 |
| warmup更新数 | 2,880 | 1,440 |
| 验证间隔有效样本 | 23,040 | 23,040 |
| 验证间隔更新数 | 5,760 | 2,880 |
| 验证节点数量 | 10 | 10 |

实现只在显式改变有效batch时，从正式v3样本预算重新派生四个步数字段：`training.max_optimizer_steps`、`training.validate_every_optimizer_steps`、`scheduler.warmup_optimizer_steps`、`scheduler.total_optimizer_steps`。学习率峰值、Adam参数、weight decay、clip和loss不自动缩放。新的profile ID和资源输出包含有效batch，避免配置混淆。

默认不传该参数时保持原有效batch。不能将新effective8配置直接恢复进原effective4 checkpoint；短测均独立初始化。本轮也不自动修改其他五组的最终训练配方。如果最终采用不同有效batch，后续实验解释应记录这项训练差异，不能把全部效果差异都归给模型结构；这不增加当前性能测试的组数。

## 5. 比较窗口按样本对齐

每设置都使用 **40个有效样本预热 + 80个有效样本计时**，共120个有效样本，执行真实forward、loss、backward和AdamW。

| 套件 / 主机 | 成功步边界 | 预热更新 | 计时更新 | 计时有效帧 |
|---|---|---:|---:|---:|
| memory / H20，两个设置均有效batch8 | step5结束 → step15结束 | 5 | 10 | 80 |
| memory / L20，两个设置均有效batch4 | step10结束 → step30结束 | 10 | 20 | 80 |

H20五次预热更新已经包含40帧；B4A2对应10份真实B4 microbatch。第一次比较不额外安排重复轮次。吞吐统一为`80 / 窗口wall`，保留窗口内skip、loader和日志间隙；不包含启动或末尾保存/退出。

报告保留每个设置的microbatch、accumulation、有效batch、具体计时边界、有效帧/s、更新median/p95、allocated/reserved、future等待、loader等待和成功/skip。旧默认`overlap`套件的10→30含义保持不变。

## 6. 直接运行

在独立的新代码checkout中，沿用已有训练Python环境并按[构建入口](FROM_SCRATCH_PERFORMANCE.md#61-构建与生成)构建现有CPU扩展。为两机选择各自实际分配的GPU/16个逻辑CPU；脚本不自动分配设备。

```bash
export PYTHONPATH="$PWD"
export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

taskset --cpu-list "$TRAIN_CPUS" python scripts/benchmark_geometry_overlap.py \
  --suite memory --hardware H20 \
  --out-dir "$BENCH_ROOT/H20-E5-memory" \
  --source-root "$DATA_ROOT/input_raw" \
  --label-root "$DATA_ROOT/output" \
  --backbone-root "$BACKBONE_ROOT" \
  --run
```

L20改为`--hardware L20`与独立的`L20-E3-memory`目录。省略`--run`只生成配置和命令；结束后得到`plan.json`、两份配置和原始训练日志，以及`summary.json`/`summary.md`。没有第31步恢复或完整验证。

如果H20实际因112 GiB配置上限而OOM，且该卡资源安排允许，可以显式使用`--gpu-memory-limit-gib 128`生成新的测试配置。它对该次两个设置使用同一上限；不自动提高上限或搜索batch。不因reserved接近上限就提前认定OOM。

单独生成较大H20配置：

```bash
python scripts/prepare_from_scratch.py \
  --base-config configs/runtime_profiles/cd009-v3/H20-E5.json \
  --hardware-preset H20 \
  --microbatch-size 4 --effective-batch-size 8 --geometry-pipeline-depth 2 \
  --dense-chunk-size 65536 --pose-chunk-size 32 \
  --out-dir "$CONFIG_ROOT/H20-E5-B4A2" \
  --run-dir "$RUN_ROOT/H20-E5-B4A2" \
  --source-root "$DATA_ROOT/input_raw" --label-root "$DATA_ROOT/output" \
  --backbone-root "$BACKBONE_ROOT"
```

H20对照把microbatch改为2，仍显式effective8，得到B2A4。L20候选用L20/E3来源、microbatch2、effective4，并显式geometry8/affinity16。

## 7. 怎样选择结果

- **更大batch占用增加、帧/s也明显提高：** 该档有实际价值，记录为该机的最终版本候选；H20仍要明确选择有效batch8这一训练设置。
- **占用增加但帧/s不变或下降：** 保留较小microbatch；显存容量没有转化为GPU效率。无需为了“用满”继续试更大的档位。
- **显存不足：** 这是该配置/上限下的实际容量限制，保留较小档；H20可按上面的显式上限选项处理一次正常容量调整，不做自动OOM扫描。

如果更大microbatch失败，已有`pose_chunk_size=128`是下一项具体GPU优化：减少每个区域的小分块与重复可微gather调用，而不是继续增dense chunk。它不会占用几十GiB新显存，却可能减少工作开销；是否实施由本轮结果决定。当前不追加CUDA trace、恢复审计、六组消融或完整训练。
