# P2 后 H20 的 batch size：可行性、显存与实际限制

**后续进展（2026-11-13）：** [P3 原生接触几何](PERFORMANCE_PHASE3.md) 已实现并记录本地 CPU 对照。本文保留 P2 时的 batch 分析；P3 的真实训练和 batch 收益需要依据其实际运行记录判断。

2026-11-12（北京时间）。依据 `cd009/full-pretraining` 的 [`60e55a2`](https://github.com/chichu-lv/suctionnet_modern_20mm/tree/60e55a263d6dd8a0b753341fd4993848222fbde4) 和 [P2 真实反馈](performance/P2_REAL_RESULTS.json)。实际训练部署仍是本地 `e73d62e`，同树 GitHub 发布 `4e9c9f5`；本文是后续分析，没有执行新的 H20 测试或修改训练配置。

## 1. 结论

**增加 microbatch 仍然可行，值得在 P2 上重新对比一次 H20/E5 的 B1A4 与 B2A2。** 当前没有模型结构或硬件上的“只能 batch 1”限制；P1 时 H20 的 B2、B4 都完成过前反向。P2 尚未重新测量它们，因此不能将 P1 的吞吐结论当成永久限制。

H20 的显存余量确实存在，但更多容量不会自动缩短逐候选 CPU 几何，也不会自动扩大被固定 chunk 拆开的 GPU 算子。当前需要回答的是：**相同四帧的全部工作，能否因重新组批而更高效完成**。

最小行动只有一项：使用已有 P2 代码和 benchmark，先比较 B1A4 与 B2A2。B4A1 仅在 B2 已显示明确收益时再考虑；不扩展 CPU 核数、worker、精度和算子版本矩阵。

## 2. 最新反馈实际改变了什么

| 项目 | 当前事实 |
| --- | --- |
| 原生全点 CPU 网格 | 已启用 |
| 骨干坐标同存储快路径 | 已启用 |
| 无实例 voxel fusion/q0 | 真实 AMP 梯度对照未通过，部署关闭；E5 的实例融合分支原本也不适用整层 voxel 复用 |
| H20/E5 正常窗口中位 update | P1 10.879 s → P2 8.590 s |
| L20/E3 正常窗口中位 update | P1 11.891 s → P2 9.590 s |
| L20/E0 正常窗口中位 update | P1 4.276 s → P2 4.285 s，无明确收益 |
| P2 后重新测 batch | **尚未执行** |

前后是不同自然帧窗口，P1 细计时关闭、P2 开启。P2 的三个 30 步窗口在截取时为 live/uncommitted；这些数字能说明观察到的性能变化，不代表此刻进度或相同帧的严格因果倍率。

P1 后的 H20/E5 对照仍有参考意义：

| 历史 P1 配置 | 帧/秒 | 相对 B1A4 | allocated / reserved 峰值 |
| --- | ---: | ---: | ---: |
| B1A4 | 0.3386 | 基线 | 11.53 / 17.07 GiB |
| B2A2 | 0.3436 | +1.48% | 22.42 / 34.46 GiB |
| B4A1 | 0.3380 | −0.20% | 43.39 / 64.87 GiB |

当时 B2/B4 都能装下，但没有稳定吞吐收益。P2 减少了 CPU 几何及坐标映射成本，执行比例和 batch 扩展代价可能改变，需要更新这条证据。与此同时，“CPU 更快了”也不意味着 B2 的 GPU 部分会自动更高效。来源：[P1_BATCH_CPU_RESULTS.json](performance/P1_BATCH_CPU_RESULTS.json)。

## 3. 为什么 30 多 G / 140 G 不能代表只用了约四分之一算力

显存容量表示能保存多少激活、参数、索引和工作区；算力表示每秒能完成多少运算；显存带宽表示每秒能搬运多少数据。三者相关，但不能按显存占用比例互相推算。少量已分配数据可以被频繁读写，也可以让某个 kernel 的计算或带宽达到上限。

这里的稀疏资源采样也不足以证明某一个 GPU kernel 已经饱和或一直空闲。P2 E5 只有四个利用率样本，不能将其均值或中位数当作连续时间的 GPU 忙闲比例。

### 3.1 最新提交中的显存口径

P2 E5 的 1031–1060 窗口记录如下：

| 指标 | 数值 | 用途 |
| --- | ---: | --- |
| Torch 可见物理容量 | 139.80 GiB | 设备容量 |
| 训练配置 allocator cap | 112 GiB | 当前进程的 PyTorch 分配器上限 |
| allocated 峰值最大值 | 11.91 GiB | PyTorch 统计到的活跃 tensor 峰值 |
| reserved 峰值最大值 | 25.36 GiB | 缓存分配器持有的峰值，包含空闲缓存 |
| NVML 进程内存采样均值 | 24.15 GiB | 3 个有效进程样本，不是峰值或连续均值 |
| NVML 进程内存采样最大值 | 25.87 GiB | 该窗口采到的最大进程占用 |

用户随后看到的 30 多 G 可以属于另一时点或另一口径；这份公开记录不足以解释它的具体来源。无论采用哪一个，都离当前 112 GiB cap 较远。cap 不是预分配目标，程序不会主动申请到 112 GiB 来提高利用率。

不能用“30 G × 4”推算 B4 峰值：模型/优化器有固定部分，激活通常随 microbatch 增长，稀疏索引、工作区和缓存还会改变。P1 的 B4 成功记录提供了容量可行性的历史证据，最终仍读取 P2 实际峰值。

PyTorch 对 allocated、reserved 及 `nvidia-smi` 差异的定义见 [2.5.1 官方源码](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/cuda/memory.py)。

### 3.2 梯度累积为什么不需要同时保留四帧激活

B1A4 依次前向/反向四帧，保留并累加参数梯度，已反向的计算图通常可以释放。B4A1 同时保留四帧的前向激活直到一次反向，所以峰值更高。

二者都处理四帧。更多显存首先让多帧计算图能够同时存在；是否更快，还取决于代码是否能把这批工作组织成更有效的算子和调度。

## 4. 源码中的四个实际限制

### 4.1 一帧已经是很大的 GPU 工作量

P2 E5 窗口每帧平均约 **841,878 个原始点、133,912 个活跃 voxel、512 个候选名额**。spconv 输入特征是 `[活跃 voxel 数, 通道数]`，坐标中的帧号区分不同场景；B1 并不是只给 GPU 一条向量。

因此，B2 可能改善某些较小深层算子、减少固定调用成本，但不能直接套用“小图像 batch 1 一定填不满 GPU”的直觉。行数大也不能反向证明所有层都已饱和。具体 kernel 的 compute、bandwidth 或 launch 占比，现有 host 计时没有给出。

### 4.2 batch 变大，很多 GPU 块的大小没有变

当前 H20 `dense_chunk_size=65536`。`Suction20Net._chunks` 调用 `chunk_apply`，沿点维逐块执行 fusion、q0 和 axis。B2 会把更多点放进这一循环，但每个满块仍是 65,536 行。

接触特征另使用 `pose_chunk_size=32`，五个区域各自按 32 个候选执行 MLP/pooling。局部 selected-point projection 还有自己的默认 chunk。增加帧 batch 不会修改这些参数。

以每帧约 84 万点、512 个有效候选估算，固定每次 update 四帧：

| 配置 | 每次 forward 的 dense 块数，单个 dense 网络 | forward 次数/update | dense 块总数/update | 五区 pose 块总数/update |
| --- | ---: | ---: | ---: | ---: |
| B1A4 | 约 13 | 4 | 约 52 | 约 320 |
| B2A2 | 约 26 | 2 | 约 52 | 约 320 |
| B4A1 | 约 52 | 1 | 约 52 | 约 320 |

实际点数、候选有效数和尾块会有少量变化；表格说明主要结构。**Python 层 forward 次数减半，不等于全部 MLP 调用或 GPU kernel 数量减半。**

真正被合并的部分包括：一个更大的 `SparseConvTensor`、实例 pooled MLP、未分块的最终 pose/q/辅助头，以及部分整次 forward/backward 的固定开销。这些仍可能受益，是重新尝试 B2 的依据。

源码：[suction20net.py](../models/suction20net.py)、[heads.py](../models/heads.py)、[contact_grouping.py](../models/contact_grouping.py)、[H20 profile](../configs/runtime_profiles/cd009-v3/H20-E5.json)。

### 4.3 CPU 几何仍然逐帧、逐候选执行

`CpuGridIndex._build` 用逐帧列表推导建立各帧索引；当前 `query_chunk_size=null`，接触 query 仍逐候选处理四个球和五个区域。原生 grid 和 FPS 的单次调用会释放 GIL，但当前调用者没有因此自动启动候选并行。

B1A4、B2A2、B4A1 每次参数更新仍处理四帧和约 2,048 个候选。改成 B2 后，几何任务被分成两次各两帧，通常仍要完成近似相同总量的 CPU 工作。

P2 E5 同一个 30 步窗口的均值为：

| 区间 | 秒/update |
| --- | ---: |
| 空间索引整体 | 0.184 |
| 整段 query | 4.719 |
| 上述两项合计 | **4.904** |
| query 内的区域筛选 | 2.025 |
| query 内的 FPS | 1.350 |
| query 内的球查询 | 1.104 |

query 内的小段已经包含在 query 总计中，不再次相加。CPU 几何可以因未来的并行/融合而变化，但“batch 从 1 改为 2”本身没有实现这种调度。

源码：[spatial_index.py](../models/spatial_index.py)、[contact_grouping.py](../models/contact_grouping.py)。

### 4.4 有些逐实例扫描会随 batch 放大

E5 的 `InstanceFusion.forward` 对整批每一个实例执行：

```python
rows = torch.where(point_instance == slot)[0]
```

这会为该实例扫描整批点。若每帧约 P 个点、I 个实例，B 帧一次 forward 约有 BI 个实例，每次扫描 BP 个元素，因此**仅这一类 mask 扫描**的规模约为 `B²IP`。固定四帧/update、累积次数为 `4/B` 时，其累计扫描量约为 `4BIP`。

候选采样和分层 loss 中也存在逐帧/实例在全批数组上生成 mask 的类似模式。不能据此声称整个训练耗时按 B² 增长，但可以确认：较大 batch 不只是省调用，它也会放大一部分当前实现中的扫描成本。这是需要与潜在 GPU 合批收益一起考虑的真实代码因素。

若后续确认它影响吞吐，可以用已有帧 offsets 限定扫描范围，并复用 `(frame, instance)` 的分组行号。它不是本次 B2 小对照的前置重构，也没有必要为此重新构建数据集。

源码：[instance_fusion.py](../models/instance_fusion.py)、[candidate_sampler.py](../models/candidate_sampler.py)、[losses.py](../suction20/losses.py)。

## 5. 数量级：增大 batch 的收益受到哪一部分限制

采用 E5 P2 **同一个 30 步窗口的均值，包含首步**：

- 总 elapsed：`T = 9.089 s/update`；
- index + query：`C = 4.904 s/update`；
- 剩余：`R = T-C = 4.185 s/update`。

这是用于估计量级的近似分解。R 包含其他 CPU、GPU、同步和启动工作，不是纯 GPU kernel 时间；少量索引工作可能与此前 GPU 工作重叠。不能把去首步后的总均值 8.661 秒与含首步的几何均值混算，也不能把各段中位数直接相加。

假设 batch 只改善 R，几何量和串行调度保持不变：

`T_new ≈ C + R/s`

| 假设，不是实测 | 推算总时间 | 整体吞吐倍数 |
| --- | ---: | ---: |
| R 快 25%（s=1.25） | 8.252 s | 1.10× |
| R 快一倍（s=2） | 6.996 s | 1.30× |
| R 完全消失 | 4.904 s | 1.85×，仅为此模型中的数学上限 |

batch 翻倍不等于 R 减半，固定 chunk 和逐实例扫描还会限制改善。该表说明为什么“显存还有约 100 G”不能换算成几倍训练速度，也说明一成左右的实际收益仍有价值。

未来若把 CPU/GPU 的微批次阶段交错执行，或真正并行 CPU 几何，这个固定串行模型就需要重算；1.85× 不是整套模型所有未来优化的全局上限。

## 6. microbatch、有效 batch 和训练行为

| 方案 | microbatch | 累积 | 有效 batch | 从头按当前样本预算派生的总更新/预热/验证间隔 |
| --- | ---: | ---: | ---: | --- |
| 当前 B1A4 | 1 | 4 | 4 | 57,600 / 2,880 / 5,760 |
| 候选 B2A2 | 2 | 2 | 4 | 同上 |
| 后续候选 B4A1 | 4 | 1 | 4 | 同上 |
| B8A1，或 B2A4 | 8 或 2 | 1 或 4 | 8 | 28,800 / 1,440 / 2,880 |

B2A2 保持每次更新四帧、总有效样本数及更新次数，适合先回答执行效率问题。B8A1 或只改 B=2 而保留 A=4 会改变有效 batch；同样的 230,400 帧需要更少更新，但每次处理更多帧，不能据此直接推导训练时间减半。

当前优化器本身的 host 区间中位约 0.0123 秒/update，远小于几何与反向。减少 Adam 更新次数不会自动省掉主要训练成本。

### 6.1 BN 变化是可记录的训练配置变化，不是禁止尝试的理由

骨干的 BatchNorm 在每次 forward 的所有活跃 voxel 行上计算统计。B1 的统计来自一帧的许多 voxel；B2 来自两帧，并且每四帧的 BN running-stat 更新次数从四次变成两次。

所以相同有效 batch 并不保证相同训练轨迹；候选随机操作的交错也可能改变。损失仍应按原有效监督集合和实例→帧规则归约，不能为了加速改成全点平均。一般正常帧全部有监督时可继续沿用现有缩放；缺项分项的权重不应被无意改写。

**不能把“B1/B2 的输出或梯度逐项相同”作为 batch 对照的通过条件。** 这是 microbatch 配置选择，不是原先要求数值等价的算法替换。保持正常 loss/梯度有限性与已有训练监控即可，不新增冻结 BN、换归一化或重跑全部消融的前置任务。

BatchNorm 定义：[PyTorch 2.5.1 源码](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/nn/modules/batchnorm.py)。预算：[pretrain_budget.py](../suction20/pretrain_budget.py)。

### 6.2 当前 profile 的限制属于部署记录

正式 profile 固定了 microbatch 和 accumulation，`validate_resume_config` 会比较这些语义字段；当前性能迁移只允许相应代码/performance 身份变化。因此若 B2 值得采用，需要记录新的执行 profile 并显式接续现有 run，不能直接编辑旧 profile 后普通 resume。

这与硬件能否运行 B2 是两回事。保持 E=4 时不用重新训练或重置样本游标；具体接续需要保留现有学习状态，并记录配置切换点。若改为 E=8，旧 `global_step × E` 样本账本不能原值直接套用，应另行处理预算定义，不作为本次首先尝试的路线。

## 7. 最小下一步：只重新比较一次 H20 的 B1A4 与 B2A2

复用已有 [benchmark_batch_cpu.py](../scripts/benchmark_batch_cpu.py)，使用**实际 P2 部署代码**和 `configs/performance/cd009-p2.json`，其余保持当前 H20/E5 配置：

| 控制项 | 两个候选的设置 |
| --- | --- |
| microbatch / accumulation | 1/4 对 2/2 |
| 数据 | 同 16 帧、相同增强 seed；原脚本 4 帧预热 + 12 帧计时 |
| CPU / data workers | 8 / 2 |
| dense / pose chunk | 65,536 / 32 |
| 体素 / 候选 | 2 mm / 每帧 512 |
| AMP / BN | 沿用当前开启 AMP、BN train |
| allocator cap | 112 GiB |
| 输出 | 每四帧 block wall、帧/秒、峰值 allocated/reserved、现有 forward/backward/query/wait |

原脚本只前向/反向，无 optimizer 更新且不加载正式 checkpoint；它可低成本筛选是否有明显吞吐收益，不能直接给出完整训练 ETA。当前代码已经接受性能 sidecar 参数，不需要另写一套 benchmark。

现有脚本要求 GPU 空闲。实际执行时，在 E5 的正常 checkpoint 边界暂停，沿用现有 GPU 锁完成两组对照后，从原 checkpoint 恢复正式训练。

如果 B2 明显领先，例如约一成且不是单个异常块造成，再考虑 B4A1 或把 B2 接入正常训练窗口。若仍只是约 1%–2% 的差别，当前短测不足以支持更改正式配置；保留 B1A4即可。这里的一成是工程取舍示例，不是统计显著性阈值。

只在候选被采用后读取原预算内正常窗口，确认包含 Adam/正常日志后的 wall time。无需新的 H20/L20 大矩阵、B8 扫描、故障注入、额外完整验证或严格逐梯度等价检查。本分析没有连接训练主机，因此没有声称上述新对照已经完成。

## 8. 如果 B2 仍不快，显存余量还可以用在哪里

下面是有针对性的后续方向，不是本次测试的额外清单：

1. **直接扩大算子 chunk。** `dense_chunk_size` 和 `pose_chunk_size` 比帧 batch 更直接控制这些网络的一次计算规模。例如 pose chunk 32→128 可以减少区域编码调用次数；dense chunk 65,536→131,072 可以减少 dense 分块次数。原 dense 块已较大，是否改善仍看实际耗时，不假定加倍就快。选择一个由现有计时或一次必要 trace 支持的改动即可。
2. **减少/并行 CPU 几何。** 继续 P2 中筛选、FPS 衔接与打包的融合。已有网格不是这条路径的终点，当前约 4.9 秒仍值得处理。
3. **在同一个 accumulation 内交错微批次阶段。** 用额外显存暂存少量 B1 前向图，当 CPU 查询一帧时让 GPU处理另一帧，所有梯度准备完成后再执行原来一次 optimizer step。这样可以利用多余显存改善重叠，而不必把 BN 的帧 batch 改大。它需要代码调度改动、保持更新边界和 RNG 顺序；不是单纯修改 batch 参数即可得到的收益，也不安排为当前小对照的前置工程。

如果日后并行多个独立实验，目标是六组总体完成时间，和缩短单组 E5 时间又不同；当前继续按一张卡一组实验的既定安排。

## 9. 来源

- [最新 P2 部署与实测](PERFORMANCE_PHASE2.md#12-实施与真实验收记录2026-11-12)、[P2_REAL_RESULTS.json](performance/P2_REAL_RESULTS.json)。
- [历史 P1 batch/CPU 对照](performance/P1_BATCH_CPU_RESULTS.json)。
- [当前 H20/E5 profile](../configs/runtime_profiles/cd009-v3/H20-E5.json)、[P2 sidecar](../configs/performance/cd009-p2.json)。
- [模型与 chunk 调度](../models/suction20net.py)、[分块实现](../models/heads.py)、[空间索引](../models/spatial_index.py)、[接触分组](../models/contact_grouping.py)。
- [实例 pooling](../models/instance_fusion.py)、[候选采样](../models/candidate_sampler.py)、[分层损失](../suction20/losses.py)。
- [训练配置恢复](../suction20/training.py)、[按样本预算](../suction20/pretrain_budget.py)、[现有 batch benchmark](../scripts/benchmark_batch_cpu.py)。
- [PyTorch 2.5.1 显存统计](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/cuda/memory.py)、[BatchNorm](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/nn/modules/batchnorm.py)。
- [NVIDIA 矩阵乘法性能原理](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)：矩阵大小、算术强度、tile 和并行效率。
- [NVIDIA CUDA 性能指南](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)：计算/访存与并行限制、Amdahl 条件估计；不使用它替代本项目 kernel 实测。
