# H20 / L20：最优 microbatch 与梯度累积组合补测

日期：2026-11-13（北京时间）。用户在有效batch8的最终测试后要求补充两台设备的最优batch组合。本轮固定已获胜的`zone gather + pose chunk128`，补齐 **H20 B4A6**、**L20 B1A12** 与新接触头的组合，并比较更长累积窗口和相同有效batch下的microbatch。

**执行入口：`scripts/benchmark_geometry_overlap.py --suite batch-combo`。H20/E5六档、L20/E3六档，一次执行固定列表。此入口只做独立短测和汇总，不调用正式训练的配置选择或启动器，也不自动改动正在运行的六组任务。**

最新已发布的[最终测试报告](performance/CD010_FINAL_OPTIMIZATION_RESULTS.md)记录：H20 B4A2/zone128为2.450753帧/s，L20 B1A8/zone128为3.622360帧/s，六组正式任务已启动并产生成功更新。这是该报告的状态快照。本补测现已完成十二档真实GPU测试，全部正常结束；完整结果见[补测报告](performance/CD010_BATCH_ACCUMULATION_RESULTS.md)。正式配方保持有效batch8。

## 1. 本轮补上的问题

上一轮`final`把有效batch固定为8，因此保留了H20的microbatch4，却没有保留[此前显存扫描](performance/CD010_GEOMETRY_MEMORY_RESULTS.md)采用的完整B4A6组合。旧B4A6/chunk32的2.259727帧/s不能代替B4A6/zone128的测量。

本轮区分两种比较：

1. **固定microbatch B，增加累积次数A。** H20固定B4、L20固定B1，观察较长累积窗口能否摊薄流水线填充、收尾和更新边界开销。
2. **固定有效batch24，改变B/A分组。** H20比较B2/B3/B4，L20比较B1/B2/B3，观察新接触头下较大microbatch是否改变吞吐排名。

`B × A`为有效batch。当前[几何流水线](../suction20/geometry_pipeline.py)以一次累积窗口为边界，depth2最多保持两份计算图在途。增加A不会把同时保留的计算图数量增加到A，但会改变每次参数更新覆盖的样本数。不同有效batch的总体排序用于比较执行方案，不能写成训练过程完全等价的算子加速。

## 2. 固定测试矩阵

### H20 / E5：六档

| 候选名称 | Microbatch × 累积 | 有效batch | 比较目的 |
|---|---|---:|---|
| `b4_a2_e8_zone128` | B4A2 | 8 | 当前正式配置，本轮同机基线 |
| `b4_a4_e16_zone128` | B4A4 | 16 | 累积窗口从2增加到4 |
| **`b4_a6_e24_zone128`** | **B4A6** | **24** | **补齐上一轮完整最优组合与zone128** |
| `b4_a12_e48_zone128` | B4A12 | 48 | 检查较长窗口是否继续获得收益 |
| `b2_a12_e24_zone128` | B2A12 | 24 | 同有效batch下的B2对照 |
| `b3_a8_e24_zone128` | B3A8 | 24 | 同有效batch下的B3对照 |

### L20 / E3：六档

| 候选名称 | Microbatch × 累积 | 有效batch | 比较目的 |
|---|---|---:|---|
| `b1_a8_e8_zone128` | B1A8 | 8 | 当前正式配置，本轮同机基线 |
| **`b1_a12_e12_zone128`** | **B1A12** | **12** | **补齐上一轮完整采用组合与zone128** |
| `b1_a24_e24_zone128` | B1A24 | 24 | 检查较长累积窗口 |
| `b1_a48_e48_zone128` | B1A48 | 48 | 检查窗口继续增长时的收益 |
| `b2_a12_e24_zone128` | B2A12 | 24 | 新接触头下重新比较B2 |
| `b3_a8_e24_zone128` | B3A8 | 24 | 同有效batch下覆盖B3 |

两机均固定以下设置：

| 项目 | 设置 |
|---|---|
| 接触特征索引 / pose chunk | `zone` / `128` |
| 几何流水线 | depth2 |
| Dense chunk | 65,536 |
| 接触几何线程 / Torch线程 | 8 / 8 |
| CPU affinity | 实际分配的16个逻辑CPU |
| 数据worker | 4 |
| 精度 | FP16，FP32原生几何 |
| 优化器 | 既有fused AdamW，峰值学习率与其他超参数保留 |
| 显存 | 无人工分配上限，使用测试卡可用显存 |
| 其他 | 局部compile关闭；原数据、候选数量、点云、区域定义和监督定义保留 |

本轮只运行H20/E5和L20/E3两个代表模型。已发生容量失败的H20 B6/B8、L20 B4不重新加入；不增加pose、dense、worker、CPU或allocator的新扫描。

## 3. 计时：各档相同有效帧数

每档独立进程、seed42新初始化，**48有效帧预热＋240有效帧计时，共288有效帧**。执行真实前向、损失、反向与AdamW更新。吞吐为计时窗口内有效帧数除以墙钟耗时，保留中间skip、等待和日志间隙，排除启动、预热及结束保存/退出。

| 有效batch | 预热成功更新 | 计时成功更新 | 计时边界 | 结束成功步数 |
|---:|---:|---:|---|---:|
| 8 | 6 | 30 | step6结束→step36结束 | 36 |
| 12 | 4 | 20 | step4结束→step24结束 | 24 |
| 16 | 3 | 15 | step3结束→step18结束 | 18 |
| 24 | 2 | 10 | step2结束→step12结束 | 12 |
| 48 | 1 | 5 | step1结束→step6结束 | 6 |

有效batch48只有一次预热更新，但仍完成48帧预热和一次真实AdamW更新；其计时窗口仍处理240帧。比较依据是帧/s，不能用不同有效batch的秒/更新直接排名。不同有效batch会经历不同的参数更新次数，本轮测量不承担训练收敛或数值等价证明。

保留原48＋240帧窗口即可完成一次吞吐筛选，不为更新次数较少追加重复轮次。288个有效样本也尚未到[既有预算规则](../suction20/pretrain_budget.py)定义的约400样本常规保存节点。CUDA OOM按已有机制记录后继续剩余固定候选，其他执行错误沿用停止行为；没有自动调参重试。

十二档全部完成合计**3,456个有效样本**，每台机器1,728个，其中每台1,440个进入吞吐统计。

## 4. 结果怎样汇总

每台机器的第一档就是其当前正式设置，作为本轮同机基线。历史帧/s保留作背景，不替代本轮基线。`summary.json`和`summary.md`同时报告：

| 字段 | 含义 |
|---|---|
| `ranking` / `best_variant` | 所有完整完成候选的总体吞吐排序与最快候选 |
| `rankings_by_effective_batch` | 按有效batch分组后的吞吐排序，键为`"8"`、`"24"`等字符串 |
| `best_variant_by_effective_batch` | 每个有效batch组的最快候选；没有完成结果时为null |
| `results[].throughput_change_from_first_percent` | 相对本轮第一档现行设置的吞吐变化 |
| `target_optimizer_updates` | 当前候选若使用既定总样本预算，对应的总成功更新数 |
| `warmup_optimizer_updates` | 对应完整训练的warmup更新数，不是本次短测预热更新数 |
| `validation_interval_updates` | 对应完整训练的原定验证间隔 |

同时保留microbatch、累积次数、有效batch、显存allocated/reserved峰值、更新及既有阶段耗时、退出/OOM记录。只将完整计时窗口且正常结束的候选纳入排名。

阅读结果时先回答：**完整B4A6/zone128与B1A12/zone128相对本轮现行配置提升多少？** 然后检查固定B增加A是否继续获益，以及有效batch24组中哪个microbatch最快。各候选只测一轮，微小差异不作为统计显著性结论，不因此开启重复扫描。

### 有效batch变化对应的训练预算

配置生成仍遵循230,400个总有效样本、11,520个warmup有效样本、每23,040个有效样本验证一次的[预算合同](../suction20/pretrain_budget.py)。这些字段会被记录，使跨有效batch的比较含义明确；本轮短测只运行上表规定的结束步数。

| 有效batch | 完整训练成功更新 | 完整训练warmup更新 | 完整训练验证间隔 | 验证节点数 |
|---:|---:|---:|---:|---:|
| 8 | 28,800 | 1,440 | 2,880 | 10 |
| 12 | 19,200 | 960 | 1,920 | 10 |
| 16 | 14,400 | 720 | 1,440 | 10 |
| 24 | 9,600 | 480 | 960 | 10 |
| 48 | 4,800 | 240 | 480 | 10 |

总体最快可能来自有效batch24或48。这是可用的另一种训练方案，但排名不会自动重写当前有效batch8的正式配置，也不会将现有checkpoint直接切换到新的有效batch。

## 5. 在实际训练机器上一次执行六档

使用**独立补测checkout**和现有依赖环境，正式任务继续保留其冻结代码。设置实际数据路径、一个新的`$BENCH_ROOT`、明确分配给测试的同型号空闲GPU UUID和16核CPU列表。

最新已发布部署中H20 GPU0运行E5，L20 GPU3运行E3，其他L20也有正式任务。因此不能直接照搬旧命令的GPU0/GPU3绑定、让无限额显存短测与正式任务在同一卡并行。可用空闲同型号卡时直接使用；只有正在训练的目标卡可用时，按下一节的既有正常暂停/恢复流程临时释放该卡。补测入口本身不探测、暂停或调度正式任务。

`$BENCH_GPU_UUID`和`$BENCH_CPU_LIST`由执行环境填入实际分配；`$DATA_ROOT`、`$BACKBONE_ROOT`沿用真实数据与骨干依赖路径。以下命令从补测checkout根目录运行：

### H20 / E5

```bash
PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES="$BENCH_GPU_UUID" \
python scripts/benchmark_geometry_overlap.py \
  --suite batch-combo --hardware H20 \
  --cpu-list "$BENCH_CPU_LIST" \
  --no-gpu-memory-limit \
  --source-root "$DATA_ROOT/input_raw" \
  --label-root "$DATA_ROOT/output" \
  --backbone-root "$BACKBONE_ROOT" \
  --out-dir "$BENCH_ROOT/H20-E5-batch-combo" \
  --run
```

### L20 / E3

```bash
PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES="$BENCH_GPU_UUID" \
python scripts/benchmark_geometry_overlap.py \
  --suite batch-combo --hardware L20 \
  --cpu-list "$BENCH_CPU_LIST" \
  --no-gpu-memory-limit \
  --source-root "$DATA_ROOT/input_raw" \
  --label-root "$DATA_ROOT/output" \
  --backbone-root "$BACKBONE_ROOT" \
  --out-dir "$BENCH_ROOT/L20-E3-batch-combo" \
  --run
```

每台机器一次命令按顺序执行六档，不需要每档回传后再增加参数。入口已经为子进程设置OMP/MKL/OpenBLAS线程限制及代码路径。输出目录必须是新的测试位置，不能指向正式训练的run目录。

省略`--run`只生成计划和逐档配置，不执行GPU计算。若先查看计划、随后决定执行，`--run`使用另一个新的测试输出目录；准备目录中的`plan.json`也保留了逐档命令。读取已完成结果不重跑：

```bash
PYTHONPATH="$PWD" python scripts/benchmark_geometry_overlap.py \
  --hardware H20 \
  --out-dir "$BENCH_ROOT/H20-E5-batch-combo" \
  --summarize-existing
```

L20替换主机名与目录即可。**本轮不传`--start-training`或`--training-root`。** `--start-training`只属于旧`final`入口；`batch-combo`不会调用`select_final_setting()`或`finalize_pretraining()`，不会再启动一批正式任务。

## 6. 仅在需要复用当前训练卡时暂停、补测、恢复

当前部署是trainer直接后台会话，旧supervisor和协调器未运行。使用原生[暂停请求接口](../suction20/pretrain_runtime.py)即可：对占用所需GPU的那个正式run写入`pause.request`，等待trainer在正常边界保存checkpoint、状态变为`paused`并以exit75退出。其他正式任务不需要一起暂停。

```bash
touch "$RUN_DIR/control/pause.request"
```

卡释放后运行本轮固定测试。补测结束后移除该暂停请求，在**原正式checkout和原环境**中重用`$FINAL_ROOT/manifests/$HOST.json`记录的原命令、GPU/CPU绑定、环境变量、train/performance配置，并追加：

```bash
--resume "$RUN_DIR/last.pt"
```

移除请求的命令为`rm "$RUN_DIR/control/pause.request"`。正式trainer的`--resume`会按现有checkpoint manifest恢复原任务；不加载补测checkpoint，不修改正式batch或学习率。原运行代码与配置留在原位置，补测代码放在独立checkout，因而普通恢复无需引入代码迁移。这里沿用[现有trainer](../scripts/train_pretrain.py)的正常保存/恢复，不新增接续审计、备份服务或调度器。

## 7. 本轮交付与结束

每台机器完成六档后，交付其`plan.json`、`summary.json`、`summary.md`及逐候选配置/日志/退出记录。汇总明确列出当前配置、完整历史最优组合与新接触头的结果，以及有效batch24内的microbatch排名。

本轮补测只回答batch与累积组合的性能缺口。使用空闲卡时正式训练照常推进；临时复用训练卡时，固定测试结束后按原配置恢复。任何后续正式训练配方变更都应明确其有效batch与样本预算含义，不能由短测排行榜隐式修改已启动任务。
