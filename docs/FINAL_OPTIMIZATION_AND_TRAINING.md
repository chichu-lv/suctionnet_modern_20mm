# 最后一轮优化与正式训练启动方案

日期：2026-11-13（北京时间）。本方案依据用户最新决定：**这就是最后一轮性能优化；本轮结束后，无论新增优化是否有效，都选择可用配置启动完整训练，不再追加优化轮次或等待再次确认。**

**执行结果（2026-11-13）：** 九档测试已完成，两机均选zone128；H20较最快原路径+17.79%，L20+29.40%。E0–E5新正式训练均已从头启动并观察到成功更新。详见[最终结果与训练说明](performance/CD010_FINAL_OPTIMIZATION_RESULTS.md)。以下保留原执行方案。

当前环境无法直接连接H20/L20训练机，因此本文件与代码提供可直接执行的最后一轮入口，不能把这里的计划生成或CPU/CI检查写成GPU测试完成、正式训练已启动。实际结果以训练机生成的选择记录与训练日志为准。

## 1. 收益评估：值得完成这次短测，但不能先认定会明显提速

已有的batch优化和新增的接触头优化需要分别判断：

| 改动 | 现有证据 | 本次决定 |
|---|---|---|
| H20增加microbatch并保留depth2 | 上轮B4A6为2.259727帧/s，比同轮B2A12快7.59%、B1A24快24.76% | 采用B4作为正式batch候选，同时在最终有效batch8下保留B2参考 |
| L20继续增加microbatch | B2仅比B1快0.17%，活跃显存近乎翻倍；B3更慢 | 固定B1，不再扩batch扫描 |
| 每区域一次gather | 已减少重复的可微索引分支，CPU小模型的普通输出/梯度与双在途图检查通过；尚无GPU速度数据 | 作为最后一次低成本候选 |
| zone gather搭配pose chunk64/128 | 同时减少接触头的小块MLP、LayerNorm、池化调用；尚无GPU速度数据 | 与最佳batch在同一次最终测试中比较 |

以上硬件数字来自[完整显存/batch扫描](performance/CD010_GEOMETRY_MEMORY_RESULTS.md)。它们使用H20有效batch24、L20有效batch12；不能直接当作本轮有效batch8的速度。[接触头实现与原理](CONTACT_HEAD_OPTIMIZATION.md)保留了代码结构、显存规模和上轮测量边界。

### 1.1 为什么索引减少很多，整体收益仍不确定

`ContactGrouping.encode_query`只投影五区选中原始点的并集，得到每点64维特征。新`zone`模式优化的是这份局部特征的索引和梯度累加，不会同时消除稀疏骨干、全点dense特征、实例融合及全部MLP的计算。因此不能将旧版实例广播的kernel热点占比套用到这里。[实际代码](../models/contact_grouping.py)

以512个候选的只读CPU自动微分图计数为例：

| 路径 | IndexBackward节点 | AddmmBackward节点 | LayerNormBackward节点 |
|---|---:|---:|---:|
| 原chunk32 | 81 | 241 | 241 |
| zone32 | 6 | 241 | 241 |
| zone128 | 6 | 61 | 61 |

每种模式都保留一次并集索引和一次local projection，所以区域内部索引实际由80降到5；H20/B4的对应区域索引由320降到5。**zone32没有减少MLP/LN调用；pose128才进一步把这些区域块调用降为原来的四分之一。** 总点MLP算术量基本不变，变化主要是调用粒度和索引反向组织。以上是操作图数量，不是GPU kernel计时或提速倍数。[分块实现](../models/heads.py)

合理判断是：这项实现已经完成、改动集中，值得做完最后一次直接吞吐比较；H20较大的microbatch更可能受益，但现有数据不足以承诺某个百分比。**不把潜在收益作为继续推迟正式训练的理由。**

### 1.2 当前没有必要继续扩大CPU或显存参数测试

上轮采用配置中，H20/L20暴露的几何future等待分别约0.223/0.073秒，约为各自更新中位数的2.1%/1.9%；loader等待均约3毫秒。后台CPU几何仍在运行，只是大部分被GPU计算覆盖。这里的host区间含同步影响，不是GPU空闲率，也不是严格的提速上限。[阶段记录](performance/CD010_GEOMETRY_MEMORY_RESULTS.md#3-已有阶段日志揭示的限制)

本轮不增加worker、CPU线程、allocator参数、dense chunk、BF16替换、局部编译或CUDA几何实现的新测试。已测过的12档batch不重做，正常训练也不以额外trace、历史AMP/RNG对齐或恢复审计为前置条件。

## 2. 现在确定正式有效batch：六组统一为8

**最后一轮直接使用将用于正式训练的有效batch8。** 这样不会在性能测试结束后再从24/12切换到8，留下实际组合尚未运行的问题。H20/B4使用A2，L20/B1使用A8；H20/B2参考使用A4。

H20 B4/depth2至少需要两份microbatch，因而有效batch至少为8。六组使用相同有效batch，便于维持模型实验之间的训练更新频率；实际microbatch仍按硬件选择，不声称不同microbatch下的BatchNorm数值完全一致。

| 每组训练预算 | 正式有效batch8 |
|---|---:|
| 总有效训练样本 | 230,400 |
| 总成功参数更新 | 28,800 |
| warmup有效样本 / 更新 | 11,520 / 1,440 |
| 验证间隔有效样本 / 更新 | 23,040 / 2,880 |
| 原定开发验证节点 | 10个，最后一个为step28,800 |
| 名义样本预算epoch | 40 |

每组仍使用原冻结数据、物理定义、样本总量和按样本计算的训练阶段。学习率峰值保持原值，不随有效batch自动线性放大；AdamW、weight decay、clip和loss定义沿用原设置。六组总有效样本仍为1,382,400。预算由[配置生成器](../suction20/from_scratch.py)和[样本预算合同](../suction20/pretrain_budget.py)换算，不直接照搬旧的57,600次更新。

## 3. 最终测试矩阵：H20五档、L20四档

入口为`benchmark_geometry_overlap.py --suite final`。所有候选均使用有效batch8、depth2；只在H20/E5和L20/E3两个代表实验上运行。

### H20 / E5

| 候选名称 | Microbatch × 累积 | 特征索引 | Pose chunk | 用途 |
|---|---|---|---:|---|
| `b2_chunk32` | B2A4 | 原chunk | 32 | 最终更新频率下的已知小batch参考 |
| `b4_chunk32` | B4A2 | 原chunk | 32 | 检查上轮最佳batch在正式设置中的收益 |
| `b4_zone32` | B4A2 | 每区一次 | 32 | 最佳batch与新gather共同应用 |
| `b4_zone64` | B4A2 | 每区一次 | 64 | 加大接触头分块 |
| `b4_zone128` | B4A2 | 每区一次 | 128 | 较大接触头分块候选 |

加入B2A4参考的理由是：上轮B4A6每次更新有六份microbatch，这次B4A2只有两份，流水线填充/收尾比例会变化。这里一次完成比较，不再新增B3/B5/B6或其他batch搜索。

### L20 / E3

| 候选名称 | Microbatch × 累积 | 特征索引 | Pose chunk |
|---|---|---|---:|
| `b1_chunk32` | B1A8 | 原chunk | 32 |
| `b1_zone32` | B1A8 | 每区一次 | 32 |
| `b1_zone64` | B1A8 | 每区一次 | 64 |
| `b1_zone128` | B1A8 | 每区一次 | 128 |

两机固定dense chunk65,536、geometry8、affinity16、Torch8、workers4、FP16、FP32 geometry、fused AdamW和无限额显存。先只比较实际需要的batch与接触头组合，不改变数据或候选定义。

### 计时窗口

每档48有效帧预热、240有效帧计时，共288有效帧。有效batch8对应**预热6次成功更新，计时30次成功更新，窗口step6结束到step36结束**。执行真实前向、损失、反向及AdamW；保留窗口内skip与等待，不把启动和末尾保存算入吞吐。

若所有候选完成，H20处理1,440个有效样本，L20处理1,152个；进入计时的分别为1,200和960个。各档独立进程，优化候选失败会记录并继续剩余固定候选，不修改参数重试。

## 4. 测完立即选择，不开启下一轮

选择规则在测试前固定：

1. 在同机完整完成的原路径候选中，选择帧/s最高的作为基线。H20比较B2A4与B4A2，L20只有B1A8。
2. 在完整完成的新接触头候选中选择最快的一项。
3. 新头候选的端到端帧/s比上述最快基线**至少提高3%**，才采用新头；否则采用最快原路径基线。
4. 记录原始排名与实际采用配置，随后生成正式训练配置并启动。没有新增收益，也照常启动原路径训练。

**3%是本次工程选型阈值，不是置信区间或统计显著性声明。** 不为0.17%这类微小差异复跑。采用规则比较真实端到端吞吐，不能用索引次数减少、某个阶段单独更快或显存更高替代。

本轮结束后，不追加allocator组合、pose chunk256、更多batch、worker/线程扫描、CUDA trace、图编译或GPU几何重构。剩余性能问题可以记录，但不再阻塞完整训练。

## 5. 结果如何应用到六组正式训练

采用当前已经记录的设备与CPU资源分配；实际映射发生变化时可通过`--resource-map`提供明确的JSON映射。[原资源记录](performance/CD010_RUN_STATUS.md)

| 实验 | GPU | CPU逻辑核 | 正式microbatch × 累积 | depth | workers | 接触几何线程 | 精度 |
|---|---|---|---|---:|---:|---:|---|
| E0 | L20 GPU1 | 8–15 | B1A8 | 1 | 6 | 不适用 | FP16 |
| E1 | L20 GPU0 | 0–7 | B1A8 | 1 | 6 | 不适用 | FP16 |
| E2 | L20 GPU5 | 56–63 | B1A8 | 1 | 4 | 不适用 | BF16 dense / FP32 sparse |
| E3 | L20 GPU3 | 24–39 | B1A8 | 2 | 4 | 8 | FP16 |
| E4 | L20 GPU4 | 40–55 | B1A8 | 2 | 4 | 8 | FP16 |
| E5 | H20 GPU0 | 0–15 | 本轮选择B2A4或B4A2 | 2 | 4 | 8 | FP16 |

E3/E4使用L20最后一轮选出的接触头模式与pose chunk；E5使用H20的选择。E0/E1/E2没有接触分组，不启用无用的接触流水线。E0/E1的6 workers、E2的精度选择沿用已有结果，不为这次收尾重新测试。全部使用Torch8线程、dense64k、无人工显存上限，以及已实现的缓存、split、原生几何和适用的共享体素头/实例广播优化。

每组正式训练从step0、seed42独立初始化。短测模型、优化器和checkpoint不计入正式预算，不恢复此前停止的CD-009/CD-010任务。原定10个开发验证节点照常运行；它们属于完整训练，不是新增的性能验收。

## 6. 一次执行最后测试，并自动进入正式训练

先在两台机器使用包含本次入口的代码及原训练环境，设置实际路径。`$BENCH_ROOT`与`$FINAL_ROOT`使用新的输出位置；`$DATA_ROOT`、`$BACKBONE_ROOT`沿用现有数据和骨干依赖路径。

### H20：测试结束后启动E5

```bash
export PYTHONPATH="$PWD"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

python scripts/benchmark_geometry_overlap.py \
  --suite final --hardware H20 --cpu-list 0-15 \
  --no-gpu-memory-limit \
  --out-dir "$BENCH_ROOT/H20-E5-final" \
  --source-root "$DATA_ROOT/input_raw" \
  --label-root "$DATA_ROOT/output" \
  --backbone-root "$BACKBONE_ROOT" \
  --training-root "$FINAL_ROOT" \
  --run --start-training
```

### L20：测试结束后启动E0–E4

```bash
export PYTHONPATH="$PWD"
export CUDA_VISIBLE_DEVICES=3
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

python scripts/benchmark_geometry_overlap.py \
  --suite final --hardware L20 --cpu-list 24-39 \
  --no-gpu-memory-limit \
  --out-dir "$BENCH_ROOT/L20-E3-final" \
  --source-root "$DATA_ROOT/input_raw" \
  --label-root "$DATA_ROOT/output" \
  --backbone-root "$BACKBONE_ROOT" \
  --training-root "$FINAL_ROOT" \
  --run --start-training
```

`CUDA_VISIBLE_DEVICES`在上述命令中选择本机短测GPU；正式启动时，启动器会为每个实验分别设置该进程的GPU绑定和CPU池，不把L20的五组都放到测试GPU3。两台机器各自完成测试、选择并启动本机训练，不需要在每档之间回传，也不需要测试结束后再等待一次授权。

省略`--run`只生成测试计划。`--run`执行短测；同时提供`--start-training --training-root`才进入本方案要求的测试后启动流程。若使用资源映射文件，将`--resource-map /absolute/path/resources.json`加入命令。

### 资源映射文件

`--resource-map`读取按实验名称组织的JSON对象。只填写需要覆盖的项，其余采用上表已记录的分配。例如H20使用：

```json
{
  "E5": {"gpu": "0", "cpus": "0-15"}
}
```

L20完整映射为：

```json
{
  "E0": {"gpu": "1", "cpus": "8-15"},
  "E1": {"gpu": "0", "cpus": "0-7"},
  "E2": {"gpu": "5", "cpus": "56-63"},
  "E3": {"gpu": "3", "cpus": "24-39"},
  "E4": {"gpu": "4", "cpus": "40-55"}
}
```

### 输出文件与独立收尾入口

测试保留`summary.json`/`summary.md`、逐档日志及退出记录。新的`$FINAL_ROOT`包含：

| 路径 | 用途 |
|---|---|
| `selection/HOST.json` | 本机原始最快基线、新头候选和最终采用结果 |
| `configs/HOST/E/train.json` | 每组正式训练配置与有效样本预算 |
| `configs/HOST/E/performance.json` | 实际采用的执行优化与接触头设置 |
| `manifests/HOST.json` | 命令、环境、GPU/CPU映射、启动记录及可供既有supervisor使用的配置登记项 |
| `logs/HOST/E.log` | 正式训练进程标准输出与错误日志 |
| `runs/E_seed42/` | trainer原生生成的状态、训练日志、checkpoint和原定验证结果 |

`HOST`为H20或L20，`E`为对应实验名。启动器为每组创建独立后台会话，不传入短测暂停/停止参数，也不传入resume/init-checkpoint；它不重启旧协调器或额外备份进程。

如果已经先用`--run`完成最后测试，也可以直接使用同一选择与生成逻辑进入正式训练，无需重跑测试：

```bash
python scripts/finalize_pretraining.py \
  --benchmark-dir "$BENCH_ROOT/H20-E5-final" \
  --training-root "$FINAL_ROOT" \
  --launch
```

L20改用其测试目录。独立入口同样支持`--resource-map`；不带`--launch`只导出正式配置与命令。输出根目录须为本轮新目录，本工具不承担旧任务重入或恢复。

启动记录中的PID与`process_spawned`表示进程已创建，不表示已经完成训练更新。以正式日志和状态文件中的实际进度确认训练在推进；本仓库的方案说明或CPU检查不充当运行状态。

## 7. 本轮结束条件

**两机固定候选执行结束、按预定规则产生采用配置、正式训练进程启动，即结束本次性能优化工作。** 新头有效则带着收益训练；新头无效则使用已验证的原路径训练。之后的工作是完成既定预训练、开发验证和目标场景微调，不再追加一轮性能优化作为开训前置条件。
