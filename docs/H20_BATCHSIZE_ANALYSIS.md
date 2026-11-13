# P2 后 H20/L20 的 batch size：显存容量、吞吐与当前选择

2026-11-13 更新。本文依据已经完成的 [H20 / E5 P2 batch 诊断](performance/H20_P2_BATCH_DIAGNOSIS.md)和 [L20 / E5 P2 batch 诊断](performance/L20_P2_BATCH_DIAGNOSIS.md)，替换此前“P2 尚未测 B2，先重新比较”的建议。两份诊断的测量代码为 P2 `e73d62e`，对应发布快照 `4e9c9f5`；不是本次整合的 P3。**按用户最新运行说明，P2 仅测试、尚未实际应用，P3 尚未在训练机测试；报告中的历史部署措辞不作为当前应用状态。**

**当前两卡均保留 B1A4。** L20 的 dense chunk 65,536 已在 P2 外部候选配置中测过，但尚未应用到正式 profile；[P3 整合配置](../configs/performance/cd009-p3-integrated.json)已在仓库提供，训练机尚未测量其整步收益。本地 CPU 证据不能当作正式训练已经提速。

## 1. 最新结论：显存装得下，不代表扩大 batch 更快

| P2 / E5 对照 | B1A4 帧/秒 | B2A2 帧/秒 | B2 相对 B1 | 当前选择 |
| --- | ---: | ---: | ---: | --- |
| H20，dense chunk 65,536 | 0.451160 | 0.461086 | +2.20% | 保留 B1A4 |
| L20，dense chunk 16,384 | 0.352710 | 0.274569 | −22.15% | 保留 B1A4 |
| L20，dense chunk 65,536 | 0.452896 | 0.410654 | −9.33% | 保留 B1A4；采用更大 dense chunk 的方向已有依据 |

H20 的 B2 三个测量块都略快，但只有一次短测、三个块，没有稳定生产收益的证据。L20 将 dense chunk 从 16,384 改为 65,536 后，B1 吞吐提高 **28.40%**，B2 提高 49.56%，然而 B2 仍慢于对应的 B1。因此当前无需继续扩展 B4/B8 扫描，也无需重复已经完成的 P2 B1/B2 对照。

两卡都有容纳更大 microbatch 的记录，没有“模型只能 batch 1”的限制。选择 B1A4 是当前吞吐证据支持的执行配置，不是显存容量强制的结果。显存余量允许保存更多激活；它不会自动减少逐候选 CPU 几何、固定 chunk 的总计算量或当前实现中的反向累加成本。

上述测试均使用相同 16 帧和增强种子，4 帧预热、12 帧测量，fresh 初始化、零 optimizer 更新，不读取正式权重。它们用于比较执行效率，不包含 Adam moments、保存和验证，不能直接换算完整训练 ETA 或收敛结果。来源：[H20 诊断](performance/H20_P2_BATCH_DIAGNOSIS.md)、[L20 诊断](performance/L20_P2_BATCH_DIAGNOSIS.md)。

## 2. 已测试和本次待应用的内容

| 项目 | 状态与含义 |
| --- | --- |
| P2 全点 CPU grid、骨干坐标同存储快路径 | 两份诊断的测试进程已启用；正式训练仍列为待应用 |
| 无实例 voxel fusion/q0 | 真实 AMP 梯度对照未通过，保持关闭；E5 的实例融合分支也不能直接整层复用到 voxel |
| P2 B1A4 / B2A2 | H20、L20 均已测量，结论见上表 |
| L20 dense chunk 65,536 | P2 定向测试已完成；仅外部候选配置改变，正式 profile 未因此修改 |
| P3 原生接触几何 | 已实现，有本地 CPU 对照；本次 H20/L20 诊断没有部署或测量它 |
| P3 `chunk_input_mode=split` | 已实现可选路径并完成小型 CPU 输出/梯度对照；训练机尚未测量 |
| P3 整合 sidecar | 目标为原生几何、P2 grid/坐标快路径、dense override 65,536 和 split；尚不能报告真实训练加速比例 |
| 实例特征广播反向改写 | 下一候选，当前不实施；没有把它计入 P3 整合收益 |

历史 P1 batch 记录和 P2 自然训练窗口仍可查阅，但当前 batch 选择以同帧的两份新诊断为依据。不同自然帧窗口、不同计时开关的 update 数字，不用于替代本次配对证据。来源：[P2 实施记录](PERFORMANCE_PHASE2.md)、[P3 实施记录](PERFORMANCE_PHASE3.md)、[split CPU 结果](performance/P3_SPLIT_CPU_RESULTS.json)。

## 3. 为什么 30 多 G / 140 G 不是“只用了四分之一算力”

显存容量表示能保存多少激活、参数、索引和工作区；算力表示每秒能完成多少运算；显存带宽表示每秒能搬运多少数据。不能按显存占用比例互相推算。少量数据也可能被频繁读写，但只有内存占用本身同样不能证明某个 kernel 已经饱和。

### 3.1 本次 H20 的实际显存口径

| 指标 | B1A4 | B2A2 | 含义 |
| --- | ---: | ---: | --- |
| Torch 可见物理容量 | 139.80 GiB | 139.80 GiB | 设备容量 |
| PyTorch allocator cap | 112 GiB | 112 GiB | 当前进程分配器的上限，不是预分配目标 |
| allocated 峰值 | 11.5317 GiB | 22.4130 GiB | PyTorch 活跃 tensor 的统计峰值 |
| reserved 峰值 | 17.0684 GiB | 34.4648 GiB | 缓存分配器持有的峰值，包含空闲缓存 |

用户看到的 30 多 G 可能是另一时点或另一统计口径；本次 B2 的 reserved 峰值本身就约 34.46 GiB。它离 112 GiB cap 很远，**不是需要增大 batch 的吞吐理由，也不是阻止扩大 batch 的硬限制**。提高 cap 不会消除约 5.14 秒的串行 CPU query，程序也不会主动申请到 cap 来提高利用率。

不能用“30 G × 4”推算 B4 峰值：参数和优化器有固定部分，激活通常随 microbatch 增长，稀疏索引、工作区和缓存也会改变。本次零更新 benchmark 不含完整优化器状态，因此这里的峰值仅描述这次执行。来源：[H20 诊断的显存记录](performance/H20_P2_BATCH_DIAGNOSIS.md)。

L20 虽然 B2 更接近 36 GiB cap，但配对 trace 的 allocation retries、OOM、缓存段释放和 allocator all-stream sync 增量均为零。这次退化已有切片梯度 kernel 增长的直接证据，不能以“触发上限后反复清缓存”解释，也没有必要为这条猜测继续提高 cap。该观察不扩展为所有未来输入都不会接近上限。来源：[L20 诊断](performance/L20_P2_BATCH_DIAGNOSIS.md)。

### 3.2 梯度累积为什么不同时保留四帧激活

B1A4 依次前向/反向四帧，累加参数梯度，已反向的计算图通常可以释放。B4A1 同时保留四帧的前向激活直到一次反向，所以峰值更高。两者每次更新都处理四帧；额外显存首先让更多计算图能够同时存在，是否更快还取决于算子和调度。

## 4. P2 执行路径为什么没有随 batch 明显加速

### 4.1 一帧已有约 84 万原始点和 512 个候选

两份诊断的 12 个测量帧各组都包含 10,089,031 个原始点、1,568,314 个活跃 voxel、6,144 个有效候选，平均每帧约 **84 万原始点、13.1 万活跃 voxel、512 个候选**。候选用于后续接触区域提取，场景编码仍使用完整观测点云。

spconv 输入特征是 `[活跃 voxel 数, 通道数]`，坐标中的帧号区分场景。B1 已经向 GPU 提供大量稀疏行；B2 可能改善部分较小算子或固定调用成本，但不能直接套用“小图像 batch 1 一定填不满 GPU”的直觉。行数很大也不证明所有层已饱和。来源：[H20 诊断](performance/H20_P2_BATCH_DIAGNOSIS.md)、[L20 诊断](performance/L20_P2_BATCH_DIAGNOSIS.md)、[模型编码](../models/suction20net.py)。

### 4.2 microbatch 变大，多数 dense 块并没有变大

H20 测试的 `dense_chunk_size=65536`；L20 的首轮为 16,384，定向复测为 65,536。`Suction20Net._chunks` 沿点维逐块执行 fusion、q0 和 axis。B2 将更多点放进同一个循环，但每个满块仍受同一 chunk 上限约束。

接触特征使用 `pose_chunk_size=32`，五个区域各自按候选分块执行 MLP/pooling；selected-point local projection 还有自己的 chunk。增加帧 batch 不会修改这些参数。

以每帧约 84 万点、512 个有效候选、dense chunk 65,536 估算，每次更新处理四帧：

| 组批方式 | 每次 forward 的 dense 块，单个网络 | forward 次数 | dense 块总数 | 五区 pose 块总数 |
| --- | ---: | ---: | ---: | ---: |
| B1A4 | 约 13 | 4 | 约 52 | 约 320 |
| B2A2 | 约 26 | 2 | 约 52 | 约 320 |
| B4A1，仅作结构说明 | 约 52 | 1 | 约 52 | 约 320 |

实际点数和尾块会改变计数：H20 捕获四帧中 fusion、q0、axis 各有 **54 块**；L20 16,384 的 B1/B2 各有 **208/207 块**。两份诊断的 local projection 都是 16 块，五区编码都是 320 块。**forward 次数减半，不等于全部 MLP 或 GPU kernel 数减半。**

一个更大的 `SparseConvTensor`、实例 pooled MLP、未分块的最终 pose/q/辅助头以及部分固定调用确实会合并；它们可能获益，但两份实测说明总收益没有抵消当前主要成本。来源：[H20 诊断](performance/H20_P2_BATCH_DIAGNOSIS.md)、[L20 诊断](performance/L20_P2_BATCH_DIAGNOSIS.md)、[分块实现](../models/heads.py)、[接触分组](../models/contact_grouping.py)。

### 4.3 CPU 接触 query 的工作量几乎没有减少

P2 的 grid 按帧建立索引，接触查询仍逐候选处理四个球和五个区域。原生球查询和 FPS 释放 GIL，不等于调用者已经把完整候选流程并行化。B1A4 和 B2A2 每四帧仍需处理约 2,048 个候选。

| 未插桩区间，秒/四帧 | B1A4 | B2A2 |
| --- | ---: | ---: |
| H20 CPU query，dense 65,536 | 5.1443 | 5.1388 |
| H20 空间索引 | 0.1857 | 0.1844 |
| L20 CPU query，dense 16,384 | 5.4491 | 5.4304 |
| L20 CPU query，dense 65,536 | 5.4176 | 5.3779 |

H20 的独立 trace 中，query 区间并集为 5.139426 秒，与记录到的 CUDA 工作重叠仅约 0.0000495 秒；L20 配对 trace 也近乎串行。增大 batch 没有缩短这部分总工作。query 和索引包含在 forward 内，不能再与 forward 相加；已经被预取隐藏的数据准备时间也不能再加到 wall time。

本段描述的是**实际测量的 P2 路径**。P3 原生接触流程把区域筛选、FPS 衔接和打包一起纳入编译路径，正是针对这个成本；其训练机收益仍未测量。来源：[两卡诊断中的时间线](performance/H20_P2_BATCH_DIAGNOSIS.md)、[L20 时间线](performance/L20_P2_BATCH_DIAGNOSIS.md)、[P3 实施记录](PERFORMANCE_PHASE3.md)。

### 4.4 新 trace 已把 GPU 代价定位到更具体的反向路径

**L20 的主要 batch 退化来自切片反向。** 已关联到 `SliceBackward0` 的 add 与 zeros/fill，累计从 B1 的 3.180685 秒升到 B2 的 6.364260 秒，增加 3.183575 秒。调用数量几乎不变，但一次 microbatch 的整输入更大。最大的调用经 autograd 序列号关联到模型顶层的前向 `aten::slice`；P2 `chunk_apply` 在每个回调前独立执行 `t[start:end]`。trace 与源码共同支持“逐块切片反复构造整输入梯度并累加”的解释。

上述增量约占**那一对插桩运行** backward host 增量的 95.8%，不能当成未插桩基准的精确因果百分比。该 trace 没有逐 ATen 输入形状或 Python stack，未直接测量每个梯度张量的完整字节数，也没有证明带宽饱和或整个训练严格按 B² 增长。来源：[L20 诊断](performance/L20_P2_BATCH_DIAGNOSIS.md)。

**H20 的突出 GPU 热点是实例融合范围内的 indexing backward。** `indexing_backward_kernel<float, 4>` 累计约 1.513 秒，其中 54 个长调用合计 1.4676 秒，经 External ID 和 autograd 序列号关联到 `encode_scene.fuse`。结合源码，具体表达式推断为 `instance_blocks` 中的 `stats['instance_features'][slots]`；没有 Python stack，所以表达式级归属仍是源码支持的推断。不能仅凭时长断言原子操作、带宽或计算饱和，也不能把它归因于四次 voxel-to-point gather。来源：[H20 诊断](performance/H20_P2_BATCH_DIAGNOSIS.md)、[实例广播实现](../models/instance_fusion.py)。

两张卡的热点比例不同，dense chunk 设置也不同。L20 的实例广播 indexing 时间在 B1/B2 间几乎不变，不能把 H20 的热点比例套到 L20。P3 的 split 改动针对已确认的切片反向机制；实例广播改写保留为下一候选，当前不实施。

### 4.5 逐实例全批扫描是源码因素，不能冒充实测首因

`InstanceFusion.forward` 对每个实例执行：

```python
rows = torch.where(point_instance == slot)[0]
```

若每帧约 P 个点、I 个实例，B 帧一次 forward 约有 BI 个实例，每次扫描 BP 个元素，**仅这类 mask 扫描**的工作量约为 `B²IP`。固定四帧、累积次数 `4/B` 时，累计约为 `4BIP`。候选采样和 loss 中也有在全批数组上按帧/实例生成 mask 的路径。

这能解释较大 batch 为什么不只是节省调用，却不能据此认定本次退化来自这些扫描。新 trace 已给出切片反向等更直接证据，因此当前不把实例分组重构加入实施前置工作。来源：[实例 pooling](../models/instance_fusion.py)、[候选采样](../models/candidate_sampler.py)、[分层损失](../suction20/losses.py)。

## 5. 当前优化选择：固定 B1A4，处理有证据的成本

L20 65,536 的结果已经说明：**直接调整 dense chunk 比扩大帧 batch 更有效**。B1 backward 与 loss 回读从 4.8328 秒降到 2.4779 秒，而 CPU query 基本不变。H20 原来就使用 65,536，本次 override 对它不再构成同样的 chunk 扩大收益。来源：[L20 定向复测](performance/L20_P2_BATCH_DIAGNOSIS.md)。

本次 [P3 整合 sidecar](../configs/performance/cd009-p3-integrated.json)的目标如下：

| 配置字段 | 目标值 | 对应作用 |
| --- | --- | --- |
| 顶层 `dense_chunk_size_override` | `65536` | 保留已冻结正式 profile 原字节，通过显式运行配置采用较大的 dense chunk |
| 顶层 `chunk_input_mode` | `split` | 每个输入一次 `torch.split`，复用多输出视图的反向拼接，避免每个独立 slice 构造整输入梯度 |
| `contact_query_backend` | `native_cpu` | 使用完整原生接触 query 路径处理球查询、五区筛选、FPS 衔接和描述量打包 |
| `spatial_index.backend` | `uniform_grid_cpu` | 保留 P2 全原始点网格索引 |
| `backbone_coordinate_fastpath` | `true` | 纳入已测量、尚待正式应用的骨干坐标快路径 |
| `no_instance_voxel_heads` | `false` | 保持此前未通过真实 AMP 梯度对照的路径关闭 |

split 保持每个满块的大小、点顺序、MLP 参数和 checkpoint 调用方式；它不是把模型算到更少的点上，也没有改变 2 mm 体素、每帧 512 候选或监督定义。小型 CPU 融合子图中，逐块 slice 的 9 个 `SliceBackward`、9 次整输入 zeros 和 8 次整点梯度 add，在 split 路径都消失；FP32/FP64 输出及所有测量梯度相同。这是机制和数值证据，**不是 CUDA/AMP 或完整模型真实损失的验收结果**。来源：[分块实现](../models/heads.py)、[CPU 结果](performance/P3_SPLIT_CPU_RESULTS.json)。

实例特征整次前向广播也有小型 CPU 原型，但一次大归约仍需处理所有点的梯度，且会持有额外全点特征。不能从调用数减少推算 GPU 时长消失；本次不把这项候选接入产品或宣称其收益。

## 6. microbatch、有效 batch 和训练行为

| 方案 | microbatch | 累积 | 有效 batch | 从头按当前样本预算派生的总更新/预热/验证间隔 |
| --- | ---: | ---: | ---: | --- |
| 当前 B1A4 | 1 | 4 | 4 | 57,600 / 2,880 / 5,760 |
| 已测 B2A2 | 2 | 2 | 4 | 同上 |
| B4A1，仅说明预算关系 | 4 | 1 | 4 | 同上 |
| B8A1，或 B2A4 | 8 或 2 | 1 或 4 | 8 | 28,800 / 1,440 / 2,880 |

B2A2 保持每次更新四帧以及总有效样本预算。B8A1 或只改 B=2 而保留 A=4 会改变有效 batch；同样的 230,400 帧虽然更新次数更少，每次却处理更多帧，不能据此推导训练时间减半。当前没有采用这类改变。来源：[样本预算实现](../suction20/pretrain_budget.py)。

### 6.1 相同有效 batch 不保证相同 BN 或训练轨迹

骨干 BatchNorm 在每次 forward 的活跃 voxel 行上计算统计。B1 来自一帧，B2 来自两帧；每四帧的 running-stat 更新次数也从四次变为两次。候选随机操作的交错可能随组批改变。因此两份诊断没有以 B1/B2 输出或梯度逐项相同为通过条件，它们是执行配置对照，不是等价算子替换。

损失仍按原实例→帧→batch 规则归约，不能为了加速改成全点平均。当前保持 B1A4，也没有新增冻结 BN、替换归一化或重跑全部消融的任务。来源：[两卡测试设置](performance/H20_P2_BATCH_DIAGNOSIS.md)、[分层损失](../suction20/losses.py)。

### 6.2 保持正式训练预算与接续状态

正式 profile 的 microbatch、accumulation 和样本预算属于训练语义。现有恢复检查会比较这些字段，性能配置变化应走已有的显式迁移和记录方式，不能直接覆盖旧 profile 后当作原配置普通恢复。

本次仍为 B1A4、230,400 有效帧；P3 目标通过性能 sidecar 声明，不因速度测试另加训练预算、验证节点或更改样本游标。来源：[训练恢复实现](../suction20/training.py)、[P3 实施说明](PERFORMANCE_PHASE3.md)。

## 7. 下一步采用已有 P3 接续流程，不重开 batch 矩阵

已经完成的 P2 B1/B2 对照和 trace 足够支持当前选择：两卡保持 B1A4，L20 的 65,536 作为待应用配置，并使用本次已实现的 split 与原生几何路径。真实整步收益以 [P3 实施说明](PERFORMANCE_PHASE3.md)中的既有接续安排和正常训练日志为准。

本文件不另设一次 profiler、B4/B8 扫描或新的组合矩阵。没有把 P3 本地 CPU 倍率相乘来预测训练速度，也没有声称训练机已部署这些改动。未来若正常日志显示瓶颈明显改变，再依据那时的实际问题决定是否调整 batch；当前无需为这种可能性预先追加工作。

更多显存也不要求立即引入跨微批次流水线、并行独立实验或更大 pose chunk。当前单卡单实验的安排保持，重点是完成已选实现并读正常运行结果。

## 8. 来源

- [H20 / E5：P2 batch 与 CPU/CUDA 瓶颈实测](performance/H20_P2_BATCH_DIAGNOSIS.md)。
- [L20 / E5：P2 batch、切片反向与 65,536 定向复测](performance/L20_P2_BATCH_DIAGNOSIS.md)。
- [P2 实施记录](PERFORMANCE_PHASE2.md)、[P3 实施记录](PERFORMANCE_PHASE3.md)、[split CPU 证据](performance/P3_SPLIT_CPU_RESULTS.json)。
- [P3 整合 sidecar](../configs/performance/cd009-p3-integrated.json)、[H20/E5 正式 profile](../configs/runtime_profiles/cd009-v3/H20-E5.json)。
- [模型编码](../models/suction20net.py)、[chunk 调度](../models/heads.py)、[接触分组](../models/contact_grouping.py)、[空间索引](../models/spatial_index.py)。
- [实例 pooling](../models/instance_fusion.py)、[候选采样](../models/candidate_sampler.py)、[分层损失](../suction20/losses.py)、[预算](../suction20/pretrain_budget.py)、[恢复检查](../suction20/training.py)。
