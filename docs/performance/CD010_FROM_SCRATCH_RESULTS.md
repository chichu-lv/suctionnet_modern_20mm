# CD-010 从头训练性能实测

记录日期：2026-11-13。本文是有界生产训练测试报告，依据 [FROM_SCRATCH_PERFORMANCE.md](../FROM_SCRATCH_PERFORMANCE.md) 执行。六组新正式训练的全面启动不属于本文已完成结果。

## 1. 当前结论与验收状态

H20/E5 的 B1A4 比较支持 **65,536 dense chunk、8 geometry 线程**：相对 geometry4，窗口吞吐提高 11.87%；再把 chunk 增至 131,072 只提高 0.50%，不足以支持增加资源占用。追加的 B2A2 候选有更高吞吐，但它改变训练期 BatchNorm 和损失归约的数值行为，不能宣称与 B1A4 数学或梯度等价。

原版部分候选虽然完成 30 次有限更新并保存 checkpoint，却在预取 worker 清理阶段失败。它们的窗口数字保留为诊断结果，不作为正常暂停验收通过。修复版 H20/B2A2 已完成第 30 步正常暂停及自身 checkpoint 恢复到第 31 步正常暂停；L20修复版E0（4/6 workers）和E2也均通过相同30→31验收。原失败候选仍保留，不能用修复后的通过状态覆盖。

L20/E3 的原版正常候选中，geometry8/CPU affinity16 相对 geometry4/affinity8 提高 24.63%。这项比较同时改变了两项资源设置，不能把全部收益归因于 geometry 线程。修复版L20/E0的单项workers4→6比较提高15.52%，支持该实验采用6 workers；E2保留4 workers、BF16 dense/FP32 sparse。

## 2. 实验边界与环境

所有候选使用冻结 CD-007 SuctionNet 数据、seed42，以及各自独立的新配置、run 和控制目录；不从 CD-009 checkpoint 迁移。数据 manifest SHA256 为 `37c4336b839b9621fa8317ec6f701846b9e8c073a494c8fa7a66b13cb62ae6a8`。模型保持 20 mm 物理半径、2 mm 网络体素、512 个候选，不使用 RGB；本轮不是物理吸附成功率或收敛测试。

| 项目 | 实测/配置记录 |
|---|---|
| GPU | 每候选单GPU；H20：NVIDIA H20-3e，实测可见容量143,771 MiB（140.40 GiB）；L20：NVIDIA L20，46,068 MiB（44.99 GiB） |
| GPU驱动 | H20：580.159.03；L20：570.133.20 |
| 整机CPU | H20：2×Intel Xeon Platinum 8558P，每插槽48物理核，合计96物理核/192逻辑CPU；L20：2×Intel Xeon Gold 6530，每插槽32物理核，合计64物理核/128逻辑CPU；两机均每核2线程 |
| 整机RAM | 操作系统MemTotal：H20为2,113,371,912 KiB（2,015.47 GiB）；L20为792,182,556 KiB（755.48 GiB） |
| 软件 | PyTorch 2.5.1+cu124，CUDA 12.4，spconv-cu120 2.3.6，cumm-cu120 0.4.11，NumPy 1.26.4，SciPy 1.11.4 |
| 原版代码 | `97aea5f9f3c029c99f1247fa8bc63ab9f2a83d1c`；运行 worktree SHA256 `6743c6ea861e4c072a446bdc6faba6db0123766408ced1b5e90a8378fc72275f` |
| 清理修复版 | `af2fe6f129b48d4ff2eb23875917d6171ce104a2`；H20/L20修复版运行 worktree SHA256 `e0ceb0a795effd30eda60531fc6ecff5ab6778c130ca1266b53c5aebbf239c2e`；仅修复预取器正常关闭并补测试 |
| CPU 资源 | H20 affinity16 / Torch8；L20 默认 affinity8 / Torch8，E3 对照增加至 affinity16 |
| 数据加载 | 默认4 workers，E0另测6 workers；spawn、persistent workers、pin memory、prefetch factor2；场景/物体缓存90/88 |
| 执行路径 | v4 sidecar，compiled FPS，原始点 uniform grid，native CPU contact，FP32 geometry，dense split，segment instance broadcast，未启用 dense-head compile |
| 优化器 | 运行记录为 fused AdamW 生效，FP32 参数 |
| 精度 | E0/E3/E5：FP16 sparse+dense、FP32 loss、GradScaler；E2 候选：BF16 dense、FP32 sparse/loss、关闭 GradScaler |
| 数学开关 | TF32 cuDNN 开、TF32 matmul 关；cuDNN benchmark/deterministic 与 deterministic algorithms 均关 |
| 内存配置上限 | H20 112 GiB，L20 36 GiB；这是配置上限，不是实测占用或物理容量 |

硬件型号、GPU容量/驱动、CPU拓扑和RAM来自本轮独立实测；整机物理资源不等于候选分配，候选仍只使用所分配的单GPU及8或16个逻辑CPU，GPU配置上限也不等于实际占用。本报告未汇总loader子进程RSS；表中仅trainer RSS，不代表整项任务内存。GPU使用率/功率是低频设备级采样，不是独占进程核算。各次运行的原配置、sidecar、实际执行身份及 native 构建摘要均随原日志保留；不同主机编译的二进制摘要不要求相同。

## 3. 测量口径与样本计划

使用生产 `scripts/train_pretrain.py`，每候选从头完成 **30 次成功更新**。前10次预热，计时窗口为**成功 step10 结束至成功 step30 结束**，覆盖11–30这20次成功更新，而不是从step11结束起算。B1A4和B2A2均每次成功更新计4个有效样本，故该窗口为80个有效样本；吞吐为 `80 / 窗口 wall 秒`。

窗口保留期间的 skip、读取等待及同步日志开销；启动/编译和第30步之后的最终 checkpoint/清理耗时不计入该吞吐。`elapsed` 为日志中的单次 attempt 时间，median/p95 由实际行计算，因此均值的20倍不必等于窗口 wall。host forward/backward、数据生产和几何查询时间可重叠或嵌套，不当作 GPU kernel 时间相加。

H20 H0/H1/H2/HB2 原版四组及HB2修复版的有序计划摘要均为 `9c629bf17c56046705daadac1ae4d16e682be6f0cdfcdb15e8cb7dce69658262`。摘要按更新保留 frame/sample ID、augmentation seed、ticket sequence/epoch 和 sampler 边界，不包含 microbatch 分组，因此 B1A4 与 B2A2 不会仅因分组不同被误报为不同计划。该一致性证明读取和增强 ticket 对齐，**不证明 B2A2 的 BN、损失、梯度或最终模型与 B1A4 等价**。缺失日志字段不能靠摘要补证。

本轮是独立初始化下的执行设置比较，不是旧 P3 固定权重、固定四帧的梯度接续验收；没有使用旧 P3 结果作为本轮加速或数值等价证明。

## 4. H20/E5 结果

以下原版候选均30成功、0 skip，日志中的 loss/gradient finite 检查均通过。

| 候选 | batch / accumulation | chunk / geometry | 11–30 wall (s) | 有效帧/s | elapsed median / p95 (s) | loader wait median / p95 (ms) | 结束状态 |
|---|---|---|---:|---:|---:|---:|---|
| H0 | 1 / 4 | 65,536 / 4 | 70.1434 | 1.140521 | 3.5037 / 3.6164 | 1.987 / 2.557 | 正常 pause30，退出75 |
| H1 | 1 / 4 | 65,536 / 8 | 62.7007 | 1.275904 | 3.1340 / 3.2053 | 2.015 / 2.835 | 正常 pause30，退出75 |
| H2 | 1 / 4 | 131,072 / 8 | 62.3902 | 1.282252 | 3.1173 / 3.1897 | 2.032 / 2.338 | 正常 pause30，退出75 |
| HB2 原版 | 2 / 2 | 65,536 / 8 | 52.1242 | 1.534795 | 2.6039 / 2.6810 | 0.898 / 1.130 | 30步有限；worker 清理失败 |
| HB2 修复版 | 2 / 2 | 65,536 / 8 | 51.7680 | 1.545357 | 2.6008 / 2.7069 | 0.917 / 1.089 | pause30退出75；自身恢复pause31退出75 |

H1/H0 的提升为11.87%，H2/H1为0.50%。原版HB2/H1为20.29%，但原版清理失败必须与速度数字一并说明；修复版HB2/H1约21.12%，跨代码版本且单次短窗口，只作为当前观察。B2A2是否用于最终长跑还需结合其训练数值语义和资源要求确定。

| 原版候选 | CUDA max allocated / reserved (GiB) | trainer peak RSS (GiB) | forward / backward host median (s) |
|---|---:|---:|---:|
| H0 | 11.13 / 22.27 | 3.35 | 2.3769 / 0.9454 |
| H1 | 11.13 / 22.26 | 3.38 | 2.0120 / 0.9426 |
| H2 | 11.13 / 23.07 | 3.37 | 1.9963 / 0.9388 |
| HB2 原版 | 21.60 / 44.54 | 4.54 | 1.6121 / 0.8435 |
| HB2 修复版 | 21.60 / 44.54 | 4.54 | 1.6083 / 0.8458 |

五组窗口单帧最大原始点数903,750、最大体素数179,931。H0/H1/H2/HB2分别仅有7/6/6/5个对齐资源样本：trainer平均CPU约2.11/2.72/2.71/2.56核，设备平均功率约148/158/157/163 W。CPU值未合计所有worker，不能据此断言整个数据链CPU空闲。B2A2接近翻倍的CUDA内存峰值是选择时的实测成本。修复版HB2有4个资源样本，设备平均功率165 W。

H0/H1/H2/HB2原版/HB2修复版的窗口平均total loss分别为1.10622/1.10632/1.10646/1.10434/1.10456；数值有限但并不完全相同。该短期差值既不是等价性证明，也不能作为模型质量优劣结论。

## 5. L20 结果

全部候选为B1A4、dense chunk65,536，30次成功更新均0 skip且loss/gradient有限。退出是否正常另列，不能由有限性推断。

| 候选 | geometry / affinity / workers | 11–30 wall (s) | 有效帧/s | elapsed median / p95 (s) | loader wait median / p95 (ms) | 结束状态 |
|---|---|---:|---:|---:|---:|---|
| E0 原版 | 4 / 8 / 4 | 21.5027 | 3.720463 | 1.0304 / 1.3103 | 218.790 / 492.701 | 清理失败，退出1 |
| E0 修复版 | 4 / 8 / 4 | 19.6477 | 4.071732 | 0.9714 / 1.0752 | 196.615 / 313.928 | pause30及resume31均75 |
| E0 修复版w6 | 4 / 8 / 6 | 17.0080 | 4.703657 | 0.8377 / 0.8990 | 1.257 / 1.674 | pause30及resume31均75 |
| E3 原版参照 | 4 / 8 / 4 | 44.9023 | 1.781648 | 2.2170 / 2.3955 | 1.180 / 1.399 | pause30退出75 |
| E3 原版资源对照 | 8 / 16 / 4 | 36.0279 | 2.220498 | 1.7938 / 1.8492 | 1.162 / 1.435 | pause30退出75 |
| E2 原版BF16 | 4 / 8 / 4 | 34.2168 | 2.338032 | 1.6963 / 1.7872 | 1.208 / 1.675 | 清理失败，退出1 |
| E2 修复版BF16 | 4 / 8 / 4 | 34.6600 | 2.308135 | 1.6903 / 1.8210 | 1.338 / 1.785 | pause30及resume31均75 |

E3资源对照提高24.63%，前30步frame、augmentation seed、点数和体素数全部一致。E0的workers比较在同一af2版本下进行，吞吐提高15.52%，等待明显下降。所有L20候选的窗口有序ticket计划摘要与第3节相同。原版到修复版的吞吐变化受顺序运行、缓存和共享负载影响，不能归因于仅在退出时执行的清理修复。

| 候选 | CUDA max allocated / reserved (GiB) | trainer peak RSS (GiB) | forward / backward host median (s) | total loss均值 |
|---|---:|---:|---:|---:|
| E0 原版 | 5.69 / 9.13 | 2.55 | — | 0.84814 |
| E0 修复版w4 | 5.69 / 9.13 | 2.65 | 0.3471 / 0.2914 | 0.84821 |
| E0 修复版w6 | 5.69 / 9.13 | 3.08 | 0.3599 / 0.2887 | 0.84818 |
| E3 原版参照 | 5.83 / 13.44 | 2.80 | — | 0.78012 |
| E3 原版资源对照 | 5.83 / 13.44 | 2.82 | 1.1964 / 0.5153 | 0.78018 |
| E2 原版BF16 | 11.68 / 22.71 | 2.70 | — | 0.82897 |
| E2 修复版BF16 | 11.68 / 22.71 | 2.69 | 0.5507 / 0.9919 | 0.82893 |

表中“—”只表示本表不展开该项，原summary仍保存host时序。E0修复w4/w6、E2修复、E3资源对照分别有4/3/7/7个资源样本，trainer平均CPU约1.40/1.66/1.30/2.53核，设备平均功率144/178/220/115 W；不包含全部worker合计CPU/RSS。各组窗口单帧最大原始点数903,750、最大体素数179,931。E1/E4未单独测速，不填造其性能或完成时间。

E2的实际运行身份明确记录`requested=bf16`、`dense=bf16`、`sparse_backbone=fp32`、`bf16_sparse_policy=fp32_spconv`、`grad_scaling=false`、FP32 loss/参数和fused AdamW生效。这只证明该真实CUDA路径短程更新与自身恢复可用，不证明旧E2故障checkpoint已恢复，也不保证整个新训练预算稳定。

## 6. 预取器退出故障与修复验收

原版故障发生在最终checkpoint之后的worker关闭路径：`_shutdown_workers()`等待worker时，PyTorch报告worker abort。30次loss/gradient有限不等于进程正常结束。现有日志没有worker native回溯，不能把推测的底层原因当作已证明事实。

af2修复先停止项目batch sampler继续发ticket，再通过原DataLoader迭代器排空已发出的批次，最后关闭worker。排空不执行model、backward或optimizer，不推进已消费ticket和有效样本预算，不吞掉实际错误。针对spawn worker、关闭顺序、错误传播、RNG/消费位置的CPU回归测试通过；真实GPU的pause30和自身resume31作为单独验收。

H20修复版已通过上述真实路径：30步对应120有效样本，恢复31步对应124有效样本，sampler连续。正常resume加载并校验原30步checkpoint；随后对predecessor30和current31运行生产exporter，提交前缀均无gap。这里不声称resume之前已独立导出核验原30步文件。其current manifest在resume后为31，故按旧attempt生成的summary可能显示 `normal_pause_at30_observed=false`；判断时必须结合原退出75、pause30事件、predecessor checkpoint及其提交日志前缀，不能把这个布尔值当作新的故障。L20修复版E0-w4、E0-w6、E2也均30→31正常退出75，配置/预算身份不变，ticket120–123及增强seed连续，120→124有效样本，无skip。L20在resume前已用生产exporter核验原30步文件及日志；原文件随后由正常retention回收，保留下来的resume重存30步和31步文件再经exporter核验。原30步和重存30步具有不同文件摘要，不能冒充同一blob。

## 7. 配置、启动与证据留存

以下为可替换根目录的命令模板，不表示本报告已经启动六组新长跑。在所记录的独立源码checkout根目录执行，复用已经验证的Python环境；`$CD010_ROOT`、`$DATA_ROOT`、`$BACKBONE_ROOT`、`$GPU`与`$TRAIN_CPUS`由实际资源安排填写。

```bash
export PYTHONPATH="$PWD"
python setup.py build_ext --inplace
python scripts/prepare_from_scratch.py \
  --base-config configs/runtime_profiles/cd009-v3/L20-E3.json \
  --out-dir "$CD010_ROOT/configs/L20-E3" \
  --run-dir "$CD010_ROOT/runs/E3_seed42" \
  --source-root "$DATA_ROOT/input_raw" \
  --label-root "$DATA_ROOT/output" \
  --backbone-root "$BACKBONE_ROOT" \
  --dense-chunk-size 65536 --geometry-threads 8 --cpu-affinity-count 16

export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
taskset --cpu-list "$TRAIN_CPUS" python scripts/train_pretrain.py \
  --config "$CD010_ROOT/configs/L20-E3/train.json" \
  --performance-config "$CD010_ROOT/configs/L20-E3/performance.json" \
  --run-dir "$CD010_ROOT/runs/E3_seed42" \
  --control-dir "$CD010_ROOT/control/E3-candidate" \
  --attempt-id "$NEW_ATTEMPT_ID" \
  --cpu-threads 8 --gpu-memory-limit-gib 36 --pause-at-step 30
```

生成器只写配置并打印资源请求，不自行启动训练。H20/E5使用`configs/runtime_profiles/cd009-v3/H20-E5.json`，显式覆盖64k/geometry8；E2使用自己的E2 base config加`--precision bf16`。B2A2须在独立候选配置中明确记录batch/accumulation，生成器没有`--microbatch`参数。

自身恢复继续传入**原始**`train.json`、`performance.json`和原run目录，增加`--resume "$RUN_DIR/last.pt"`；恢复31步验收使用新attempt/control目录及`--pause-at-step 31`。不得用`resolved_config.json`替代恢复时的原配置字节，也不得把旧CD-009 checkpoint接入v4。

保留原配置/sidecar及SHA、`resolved_config.json`、`runtime.json`、每attempt的`updates.jsonl`和`events.jsonl`、控制状态/退出结果/资源日志、`checkpoint-manifest.json`及其current/predecessor/best/验证节点引用的文件。保留失败attempt和未提交尾部，不把它们拼接为成功窗口。生产导出命令为：

```bash
python scripts/export_training_logs.py --run-dir "$RUN_DIR" \
  --output-dir "$NEW_EXPORT_DIR" --resource-log "$CONTROL_DIR/resources.jsonl"
```

该exporter校验checkpoint文件摘要及提交日志前缀；私有窗口summary保留原始路径和摘要，公开本文只列去标识化统计。

H20五组候选H0/H1/H2/HB2/HB2-fix及其configs、control、reports和原版/af2源码归档的共享备份已完成逐文件hash核验：150个文件、2,330,595,400字节（约2.33 GB），checkpoint manifest摘要核验通过。随后生成的硬件记录和旧训练状态快照不属于这次备份清单，不能宣称一并备份。

L20七组候选（原版E0/E2、两组E3对照、修复版E0-w4/E0-w6/E2）的独立共享备份已完成hash核验，备份中的checkpoint及提交日志均通过生产exporter；配置与控制证据（含退出结果、资源、stdout及已有导出结果）另有文件摘要/大小核对。备份包含失败候选，不代表其训练退出验收通过。

训练仍按230,400有效样本预算进行：有效batch4对应57,600次成功更新，warmup2,880次，每5,760次更新按原固定验证节点运行。30/31步候选没有完成任何正式验证节点，不宣称验证指标、测试集或收敛结果。独立验证接口使用run中的`resolved_config.json`，具体见[原运行说明](../FROM_SCRATCH_PERFORMANCE.md#63-启动恢复评价)；本报告没有追加验证或官方测试。

## 8. 外推限制与后续

当前实测支持的设置如下。这里的“支持”是已测代表组的性能选择，不是六组运行已经启动；E1/E4迁用设置不产生它们自己的实测数字。

| 已测实验 | 设置依据 | 证据版本及范围 |
|---|---|---|
| H20/E5 | 64k、geometry8、affinity16、workers4；B2A2更快但训练语义改变须明确 | B1A4比较在97；B2A2在af2通过30→31 |
| L20/E0 | 64k、geometry4、affinity8、workers6、B1A4/FP16 | af2同版本w4/w6比较及各自30→31 |
| L20/E2 | 64k、geometry4、affinity8、workers4、B1A4/BF16 dense+FP32 sparse | af2真实30→31 |
| L20/E3 | 64k、geometry8、affinity16、workers4、B1A4/FP16 | 97正常pause30资源对照；本轮没有af2 E3单独复测 |

仅以短窗口恒定吞吐外推230,400样本，H20/H1约50.16小时，修复版HB2约41.41小时，L20/E0-w6约13.61小时、E2修复版约27.73小时、E3 geometry8约28.82小时。这些是**纯训练时间**，不是交付ETA；未包含10个验证节点、启动、保存/备份、暂停、共享主机负载变化、长尾帧、未来skip或数值故障。每设置只有一次短窗口，没有置信区间。

后续由正式运行记录给出六组配置落地、资源分配、恢复/新启动状态及长期观测；本文不把未测E1/E4、未执行的af2 E3复测或长跑验收补写为已完成。旧CD-009数据、日志和checkpoint继续保留。测试通过只支持已测执行路径的短程可用性，不能保证长期数值稳定、模型精度或真实吸附成功。
