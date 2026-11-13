# L20 / E5：P2 batch 对照与反向传播瓶颈

2026-11-13。**B2A2 比 B1A4 的测量吞吐低 22.15%，新增耗时集中在 backward。B1/B2 配对 trace 显示主要新增 GPU 成本是切片梯度的清零与累加；两次捕获均没有 allocator 重试、OOM 或缓存释放事件，不支持把本次变慢归因于 36 GiB 上限触发的反复清缓存。**

本报告对应冻结 P2 源码 `e73d62e`，不是之后的 P3。结果用于诊断，没有修改正式训练配置或预算。后续只把 dense chunk 从 16,384 调到 65,536 复测同一 B1/B2 对照：B1 吞吐提高 28.40%，B2 提高 49.56%，但 B2 仍比对应 B1 慢 9.33%；不据此扩大 batch。

## 设置与未插桩对照

| 项目 | 设置 |
| --- | --- |
| GPU | NVIDIA L20；NVML 总容量 46,068 MiB |
| 主机 CPU | 2 × Intel Xeon Gold 6530；每 socket 32 核、每核 2 线程，共 64 物理核 / 128 逻辑核，2 NUMA nodes |
| 主机内存 | MemTotal 792,182,556 KiB，约 755.48 GiB |
| 本次 CPU / loader 配额 | 仅分配 8 个逻辑核，Torch threads=8，2 workers；未使用全部 128 个逻辑核 |
| 模型 | E5，AMP，BN train，2 mm voxel，每帧 512 候选 |
| dense / pose chunk | 16,384 / 32；dense activation checkpoint 关闭 |
| 几何 | P2 全点 CPU grid、编译 FPS、单候选查询路径 |
| 显存上限 | PyTorch allocator 36 GiB；不是预先分配 36 GiB |
| 对照 | B1A4 / B2A2；有效 batch 均为 4 |
| 输入与更新 | 相同 16 帧及增强种子；4 帧预热、12 帧测量；fresh seed42，零 optimizer 更新 |

两组均通过输入计划、初始化与最终参数 hash、有限梯度、零更新及无检查点读写检查。GPU、源码、sidecar、helper、初始化及非 batch 配置身份相同。测量部分各含 10,089,031 个原始点、1,568,314 个 voxel、6,144 个有效候选。组批会改变 BN、归约和 RNG 交错，这不是逐输出等价或训练质量比较。

| 指标 | B1A4 | B2A2 |
| --- | ---: | ---: |
| 帧/秒 | 0.352710 | 0.274569 |
| 三个四帧块，秒 | 11.414 / 11.422 / 11.186 | 14.779 / 14.564 / 14.362 |
| 平均 wall，秒/块 | 11.3407 | 14.5683 |
| forward host，秒/块 | 6.3642 | 6.3095 |
| backward 与 loss 回读，秒/块 | 4.8328 | 8.1228 |
| CPU query，秒/块 | 5.4491 | 5.4304 |
| 暴露的数据等待，毫秒/块 | 1.359 | 0.453 |
| 测量 allocated 峰值，GiB | 11.4982 | 22.3791 |
| 测量 reserved 峰值，GiB | 16.4082 | 34.8965 |

平均 wall 增加 3.2275 秒，backward 增加 3.2900 秒（约 68.1%），其他部分小幅抵消。CPU query 仍然很慢，但没有随 B2 增长；数据等待也没有增长。额外耗时已经定位到反向阶段，单凭这张表还不能分解其中的设备执行、同步和分配器成本。query 包含在 forward 内，不能重复相加。

## 配对 B1/B2 trace：GPU 主要在做什么

两次独立诊断都只捕获预热后的第一个四帧测量块。B1 捕获块耗时 12.2697 秒，对应未插桩块 11.4138 秒；B2 为 15.4793 / 14.7792 秒。约 7.50% / 4.74% 的差异包含 profiler 开销和波动，不用于修正基准结果。详细 CPU 计时只在捕获块开启，诊断覆盖项单独记录；不能仅凭官方 sidecar hash 将该执行称为未插桩基准。

| 时间线量，秒 | B1A4 trace | B2A2 trace |
| --- | ---: | ---: |
| CUDA kernel 区间并集 | 4.863827 | 8.225735 |
| 全部 CUDA 工作区间并集 | 4.896157 | 8.263103 |
| 事件时间跨度 | 12.270888 | 15.480138 |
| backward host 区间 | 5.147831 | 8.470477 |
| backward 与 CUDA 工作重叠 | 4.635158 | 8.000461 |
| CPU query 区间 | 5.481524 | 5.401788 |
| query 与 CUDA 工作重叠 | 0.0000219 | 0.0000134 |

CPU query 仍与 GPU 工作基本串行。B2 内部区域筛选约 2.1118 秒、FPS 1.6133 秒、球查询 1.4013 秒、描述量打包 0.2487 秒。两个最长 CUDA 空档约 2.8325 / 2.5756 秒，几乎完全落在两个 query 区间内。

B2 GPU 上最大的具体 kernel 变体为 FP32 vectorized add（6,819 次，5.1969 秒）、FP32 fill（4,345 次，1.5973 秒）和 FP32 indexing backward（574 次，0.7733 秒）。进一步通过 kernel External ID 与 CPU/autograd 区间关联：

| 已关联的反向路径 | B1 次数 / 秒 | B2 次数 / 秒 |
| --- | ---: | ---: |
| `SliceBackward0` 梯度累加中的 FP32 add 家族 | 844 / 2.445267 | 842 / 4.877021 |
| `SliceBackward0 → slice_backward → zeros → zero_ → fill_` | 1,472 / 0.735418 | 1,465 / 1.487239 |
| 实例特征融合 scope 的 indexing backward | 208 / 0.732862 | 207 / 0.733090 |

前两项从 **3.180685 秒升到 6.364260 秒，增加 3.183575 秒**；约相当于诊断 backward host 增量 3.322646 秒的 **95.8%**。这是一对插桩运行中的时间分解，不是未插桩基准增量的精确因果百分比。调用数量基本不变，主要是单次切片反向成本变大。最大的 add 从约 4.04 增到 8.03 毫秒，fill 从 1.22 增到 2.41 毫秒；实例广播 indexing backward 时间几乎不变。

对 B2 最大的八次 add 与八次 fill 调用做 autograd 序列映射，前向均为模型顶层 scope 内的 `aten::slice`。冻结代码的 `chunk_apply` 在进入每个回调前执行输入切片。**结合配对 trace 和源码，dense 分块反复构造整输入切片梯度并累加，是主要成本来源的具体解释。**

模块计数记录到 B1 每次输入约 82.9–86.6 万点，而 B2 两次分别为 1,699,303 / 1,685,826 点；dense chunk 上限仍为 16,384 点，末块可以更短。没有采集逐 ATen 输入形状或 Python stack，因此没有直接测出每个 slice-gradient 张量的完整形状和读写字节数，表达式级归属仍是源码支持的推断；不能据此宣称已测得带宽饱和或严格的二次复杂度。

B1/B2 实际模型/backbone/query/backward 调用为 4 / 2 次；dense fusion、q0、axis 各 208 / 207 块；local projection 均为 16 块；五区编码各 64 块，合计 320 块。实例广播的 207 次 indexing backward 通过序列号对应到融合 scope 的前向 `aten::index`，具体表达式推断为 `stats['instance_features'][slots]`。它仍值得优化，但不是本次 L20 B2 的第一 GPU 热点。不能直接套用 [H20 诊断](H20_P2_BATCH_DIAGNOSIS.md) 的热点比例；两者硬件与 dense chunk 不同。

## 36 GiB 上限是否触发反复清缓存

在现有同步边界读取 PyTorch allocator 累计计数，开始前不重置，结束后先取样再导出 trace。捕获块的增量为：

| 计数增量 | B1 trace | B2 trace |
| --- | ---: | ---: |
| allocation retries | 0 | 0 |
| OOM | 0 | 0 |
| allocator device / segment allocations | 0 | 7 |
| allocator device / segment frees | 0 | 0 |
| allocator all-stream sync counter | 0 | 0 |

B1 reserved 保持约 13.887 GiB，B2 从约 25.057 GiB 增至 29.928 GiB，两次均没有释放缓存段。**配对捕获不存在“触及 cap → 重试 → 清缓存”的计数证据，而切片梯度 kernel 时间已有明确增长。** 因此不以 cap 为由继续试验更高上限。未插桩 B2 后续块的 reserved 峰值 34.8965 GiB 确实更接近上限，但那几个块没有相同计数观测；不能把结论扩展为所有输入都不受 cap 影响，也没有做解除上限的反事实对照。

这些是 PyTorch allocator 的计数，不覆盖所有外部 CUDA 库分配。B2 trace 内其他 `cudaMalloc` / `cudaFree` API 仍有调用，总 host 时间分别约 0.0076 / 0.0217 秒；不能把它们等同于 PyTorch 缓存清空。profiler 本身可能改变分配行为，累计差值也包含启动 profiler 的开销。此次没有执行 Adam 更新或分配其全部 moments，不能据此保证正式训练所有阶段都满足容量要求。

## 定向复测：dense chunk 改为 65,536

保持同一 GPU、输入计划、初始化、AMP/BN、CPU/loader、有效 batch 和 36 GiB cap，只修改外部候选配置的 dense chunk。每个 chunk 设置内部的 B1/B2 都通过原有严格身份比较。跨设置检查的模型配置仅有 `dense_chunk_size` 及其派生配置 hash 不同，没有放宽原检查器。

| 未插桩指标 | chunk 16,384 B1 | chunk 65,536 B1 | chunk 16,384 B2 | chunk 65,536 B2 |
| --- | ---: | ---: | ---: | ---: |
| 帧/秒 | 0.352710 | 0.452896 | 0.274569 | 0.410654 |
| backward 与 loss 回读，秒/块 | 4.8328 | 2.4779 | 8.1228 | 3.4178 |
| forward host，秒/块 | 6.3642 | 6.2331 | 6.3095 | 6.1838 |
| CPU query，秒/块 | 5.4491 | 5.4176 | 5.4304 | 5.3779 |
| allocated 峰值，GiB | 11.4982 | 11.4843 | 22.3791 | 22.3636 |
| reserved 峰值，GiB | 16.4082 | 16.2676 | 34.8965 | 34.4434 |

65,536 的 B1 三块耗时为 8.880 / 8.895 / 8.721 秒，B2 为 9.832 / 9.815 / 9.575 秒；全部完成、loss/梯度有限、参数 hash 不变，零更新。B1 相对原 chunk 快 28.40%，B2 快 49.56%；CPU query 基本不变，收益主要来自 backward，支持上述分块反向成本诊断。更大 chunk 下 B2 仍比 B1 慢 9.33%，因此没有 batch 扩大结论。

这次修改未触碰源码或正式 profile，没有测量新 chunk 的 trace，也未做完整梯度等价验证。它是有限输入上的定向性能证据，不保证所有输入、正式 Adam 更新或后续版本都具有同样收益；也不能把显存峰值的小差异解释为普遍节省。

## 后续方向与结论边界

优先调查 chunk 切片反向的整输入梯度构造与累加，评估等价的分块/梯度归约实现；随后检查实例特征广播 backward。CPU 区域查询仍是另一个明确优化目标。任何实现都需保持数值和梯度语义，再做真实整步对照；本次没有部署或测量 P3，也没有修改正式 profile。

所有 CUDA 时间按区间并集合并，CPU 嵌套范围和 GPU annotation 镜像不重复计数。host `copy_`、launch 等时长可能包含等待，不能当成独立 CPU 计算再与 GPU 时长相加。每个设置只有三块基准，另有各一块 B1/B2 插桩观测，没有重复试验、置信区间、带宽/算力饱和测量、收敛或物理成功结论。

## 证据 hash

公开记录只保留硬件、配置与聚合结果；原始 JSON 和完整 trace 保留为私有复核证据。

| 证据 | SHA256 |
| --- | --- |
| B1A4 基准 JSON | `b7b8bbc80273f9985f8687d3f6c5682b9ff117dbb3f2aceb43cb7aed0c5d0f63` |
| B2A2 基准 JSON | `0f80f820b19ccff7a86d78036a012f0ab7a38172240d65fc848352f670127a13` |
| chunk 65,536 B1A4 基准 JSON | `26bbcd1132d2b98097645e8bcc8c71731a588f75e72e589c2dc7fb0ec620df2a` |
| chunk 65,536 B2A2 基准 JSON | `0e8d21eb44b816a8a20c14d6183cfc0bafb5d8935507e218469e853db633e896` |
| B1 单块 Chrome trace | `d007a5599daf98816c6c97b61fdb4a6947ca4ca42e3caa72f6867656a2d557eb` |
| B1 profiler 摘要 JSON | `ab44c86a74ea7941faa9412e75313c45e7070411188a5e5913b148a598dbfff0` |
| B2A2 诊断 JSON | `49ea6258eb7bdd0de32e66d10ea3900e85d2445087a5d51a5887937ce631099b` |
| B2 单块 Chrome trace | `326026e162f0e1e4169c9f73a916c786db3ab8fa99b2f4938f4bf1e66b29f752` |
| B2 profiler 摘要 JSON | `11e132b5b936da96424a363b7325f0cc6e3f600d46898a10535c25aff9752818` |
| 外部 profiler wrapper | `dd4cf420e52fa32a49841e294405f5182d84a9c838b67e99d943d5b5d16a05da` |

冻结 benchmark helper SHA256 `2e2406b6d55b82576edf63b33423bc6f7510dc7751fda735f3ef1bf84e79784e`；源码工作树 SHA256 `0c6501ac1b8e374fc37bbba4df3abc30dd4fbf4f726fa23b9c7092c537b3aed8`；P2 sidecar SHA256 `8a104de89df99ea327a759e5471bef357965ea20e7b57461081560ca87216928`。这些标识测量版本，不代表当前远端训练状态或最新代码。
