# batch 扫描后的选择与接触头优化

日期：2026-11-13（北京时间）。依据最新[完整显存/batch扫描报告](performance/CD010_GEOMETRY_MEMORY_RESULTS.md)，该报告对应测试源码 `a07659a`、结果提交 `d6a9328`。

**本轮固定 H20/E5 的 microbatch4/depth2、L20/E3 的 microbatch1/depth2，结束扩大 batch 的扫描。下一项只优化 GPU 接触头：把重复的特征 gather 移到区域级，再比较 pose chunk32/64/128。** 已提供可选实现与 `--suite contact-head` 入口；新增接触头方案尚无真实 GPU 吞吐结果，默认仍为已测的 `chunk` gather 路径。完整训练与协调器继续停止。

## 1. 扫描已经回答的问题

12档均已尝试，9档完成、3档CUDA OOM。当前选择如下；这是性能短测中的配置，还没有确定正式训练的有效batch。

| 主机 / 实验 | 采用的性能对照 | 有效batch | 帧/s | allocated / reserved 峰值 GiB | 选择依据 |
|---|---|---:|---:|---:|---|
| H20 / E5 | B4A6 / depth2 | 24 | 2.259727 | 89.08 / 138.55 | 比同轮B1快24.76%、B2快7.59%、B3快2.33%；本次完成候选中最快 |
| L20 / E3 | B1A12 / depth2 | 12 | 3.038967 | 11.56 / 25.54 | B2只快0.17%，allocated却升至22.43 GiB；B3更慢 |

H20的更大microbatch在几何重叠之后确实有收益，说明原先“加batch不提速”的结论有实现版本边界。但B3→B4只增加2.33%，收益已经趋缓。L20虽然能运行B3，却没有获得更高吞吐；这里的限制不能用“显存还没用满”解释。L20原始数值排名仍保留B2第一，不把0.17%的单次差异写成已经证实的收益。[原始结果](performance/CD010_GEOMETRY_MEMORY_RESULTS.md#2-完整吞吐和容量结果)

两机都继续保留depth2。H20/B8单图只有1.643636帧/s，比B4双图低27.26%；L20/B4单图只有2.107138帧/s，比B1双图低30.66%。这是整组配置的实测差异，因microbatch也不同，不能把全部差异都分配给流水线开关。

## 2. 显存上限已取消，三个 OOM 的含义不同

本轮`gpu_memory_limit_gib=null`，H20/B4的reserved达到138.55 GiB，L20/B2达到43.66 GiB，已没有原112/36 GiB人为配额。allocated、reserved、设备物理容量使用不同口径；尤其不能用各自独立记录的峰值相减，断言某个失败瞬间有多少连续可用空间。[容量记录](performance/CD010_GEOMETRY_MEMORY_RESULTS.md#4-三个oom档位)

| 失败设置 | 已记录的失败点 | 可以得到的结论 |
|---|---|---|
| H20/B8/depth2 | `encode_scene`的dense分块输出`torch.cat`；allocated133.53 GiB，新增请求6.39 GiB | 二者合计约139.92 GiB，已超过当时报错可见的139.80 GiB。仅再次取消配额或假定回收缓存，不能解决这次活跃张量与新增输出的容量需求 |
| H20/B6/depth2 | 反向；allocated127.38 GiB，新增请求4.84 GiB；设备空闲2.36 GiB、reserved中未allocated9.53 GiB | 分配器碎片或不可释放的分块可能参与失败，但快照不能证明唯一原因，更不能证明改一个allocator选项就能完整跑完B6 |
| L20/B4/depth2 | 反向；allocated38.84 GiB，新增请求3.23 GiB；设备空闲1.16 GiB、reserved中未allocated4.02 GiB | 当前实现及分配器状态下无法完成该设置；小一档B3已经没有吞吐收益，因此不把继续挤入B4作为下一优先级 |

H20/B8若要放入两份图，需要减少活跃/临时张量峰值或改变图的生命周期。这里先采用已完成且最快的B4，而不继续添加allocator组合、B5/B6重试或更大batch扫描。后面的接触头改动不宣称解决这些dense/反向OOM。

## 3. 当前暴露的等待已经很小，继续优化前向与反向

| 采用配置 | 更新中位数 | forward host | backward host | 几何future等待 | loader等待 |
|---|---:|---:|---:|---:|---:|
| H20/B4A6/depth2 | 10.625 s | 4.222 s | 5.470 s | 0.223 s | 3.019 ms |
| L20/B1A12/depth2 | 3.920 s | 1.938 s | 1.695 s | 0.073 s | 3.323 ms |

几何future等待与更新中位数之比约为2.1%和1.9%，loader等待只有约0.03%和0.08%。这是用已记录中位数说明量级，**不是GPU空闲率，也不是下一轮加速的严格上限**。这些host区间含同步影响，不能机械相加，也不能据此判定某个CUDA kernel已达到算力或带宽上限。[阶段数据](performance/CD010_GEOMETRY_MEMORY_RESULTS.md#3-已有阶段日志揭示的限制)

CPU几何仍有实际工作量：采用配置的后台job中位数分别为6.110秒和2.661秒，只是大部分与GPU工作重叠。GPU部分变快后，CPU等待可能重新显现，因此后续继续读取已有future等待即可；当前不增加worker、几何线程或CPU亲和性扫描。

L20 B1→B3的反向host减少0.218秒，但前向增加0.200秒、future等待增加0.057秒，解释了为什么容量增加没有转成端到端收益。H20 B1→B4的主要可见变化也包括前向/反向缩短，不能把全部收益只归因于少等CPU。

## 4. 源码里可以直接减少的工作：按区域 gather

[ContactGrouping.encode_query](../models/contact_grouping.py)先投影被五区选中的点集，再按区域、按pose chunk执行点MLP和池化。现有`chunk`路径在每个小块内部执行`projected[rows]`，每次都建立一个可微索引分支，反向把梯度累加回同一个投影点集。

新`zone`路径把这一索引移到每个区域的分块之前，再使用已有`split`模式切分区域特征：

```mermaid
flowchart TB
    P["投影选中点的并集"] --> C["chunk 模式：每区分成小块"]
    C --> G["每块 gather，再做点 MLP 与池化"]
    P --> Z["zone 模式：每区 gather 一次"]
    Z --> S["split 特征，再做相同点 MLP 与池化"]
    G --> O["五区特征与原评分头"]
    S --> O
```

相应执行顺序为：

```python
# 原路径：每个 pose chunk 内索引。
def encode_zone(rows, mask, geometry, count):
    point_features = projected[rows]
    return encode_and_pool(point_features, mask, geometry, count)

# 可选新路径：每区只索引一次，再把可微特征交给 chunk_apply。
zone_features = projected[zone_indices]
features = chunk_apply(
    encode_and_pool, zone_features, mask, geometry, count,
    chunk_size=pose_chunk_size, input_mode="split",
)
```

上面是执行示意，实际MLP、池化、mask和count逻辑保留在模块内。若一份microbatch有`M`个候选，原路径建立`5 × ceil(M / pose_chunk_size)`次区域特征索引；新路径为每区一次，共5次。

| 标准每帧512候选 | 原chunk32 | 原chunk64 | 原chunk128 | 新zone模式，任意这三档chunk |
|---|---:|---:|---:|---:|
| H20，microbatch4，M=2,048 | 320 | 160 | 80 | 5 |
| L20，microbatch1，M=512 | 80 | 40 | 20 | 5 |

这是**源码操作数量的减少，不是已经测到的kernel热点占比或速度倍数**。按区gather减少重复索引及其反向分支；pose chunk再增至64/128，主要进一步减少小MLP、LayerNorm和池化调用次数。已有dense chunk65,536不控制这一循环，因此dense64k→128k收益小，并不否定独立的pose chunk优化。

按标准五区采样数32/64/32/32/32、投影维度64估算，全部区域gather输出元素量为`M × 192 × 64`：H20/B4的FP16规模约48 MiB、L20/B1约12 MiB，FP32为两倍。这只是张量规模估算，不是新增显存峰值或GPU测量结果。

## 5. 配置与实现边界

v4性能sidecar新增可选字段：

```json
{
  "contact_feature_gather_mode": "zone"
}
```

允许值为`chunk`和`zone`。缺省与配置生成器默认仍为`chunk`，保留已测执行方式；新候选通过`--contact-feature-gather-mode zone`显式启用。字段进入模型执行记录，便于对照实际使用的路径。[配置生成器](../scripts/prepare_from_scratch.py) · [v4字段校验](../suction20/performance.py) · [模型接线](../models/suction20net.py)

区域查询、原始点索引、FPS、几何描述、MLP权重、mask、计数、池化与输出定义沿用原实现。改变的是特征索引和梯度累加的组织方式；允许正常浮点归约差异，不增加历史checkpoint的AMP/RNG逐项对齐。代码检查只需要正常样本的输出和梯度检查及已有套件，不追加完整验证或新的恢复流程。

## 6. 下一次只运行每机四档接触头比较

入口：`benchmark_geometry_overlap.py --suite contact-head`。两机固定当前采用的microbatch、累积次数与depth2，测试四个明确的候选：

| 候选 | gather模式 | pose chunk | 回答的问题 |
|---|---|---:|---|
| 原路径对照 | chunk | 32 | 当前已测方式在本次同条件窗口中的速度 |
| 按区域索引 | zone | 32 | 单独减少重复gather是否有净收益 |
| 按区域索引与中等分块 | zone | 64 | 减少接触头小块调用是否继续受益 |
| 按区域索引与较大分块 | zone | 128 | 更大pose chunk的收益与显存代价 |

H20仍为E5/B4A6/有效batch24，L20仍为E3/B1A12/有效batch12。其余保持dense65,536、geometry8、affinity16、Torch8、workers4、FP16与FP32几何、fused AdamW、无人工显存上限，继续原冻结数据和512候选。

每档48有效帧预热、240帧计时；H20窗口step2→12，L20为step4→24。每机只比较四档，共八次独立短测。运行器沿用CUDA OOM记录后继续、其他错误停止、完整窗口且正常退出才进入排名的机制。不会重新展开已完成的12档batch扫描，也不加入完整训练、恢复审计或额外trace。

```bash
export PYTHONPATH="$PWD"
export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

taskset --cpu-list "$TRAIN_CPUS" python scripts/benchmark_geometry_overlap.py \
  --suite contact-head --hardware H20 \
  --no-gpu-memory-limit \
  --out-dir "$BENCH_ROOT/H20-E5-contact-head" \
  --source-root "$DATA_ROOT/input_raw" \
  --label-root "$DATA_ROOT/output" \
  --backbone-root "$BACKBONE_ROOT" \
  --run
```

L20改为`--hardware L20`与新的`L20-E3-contact-head`输出目录。省略`--run`只生成计划；汇总继续输出`summary.json`/`summary.md`、每档配置和日志。以同机本次原路径对照计算收益，不将不同日期、有效batch或其他模型的速度混为一组。

如果zone32不比chunk32快，原路径结果仍保留；再看zone64/128是否因调用粒度获得净收益。若四档没有明确收益，就保持当前已测方式，使用已有前向/反向和future等待定位剩余阶段，不把源码分支数量的减少写成实际提速。

## 7. 性能配置与正式训练配方分开确定

本轮固定有效batch24/12，是为接续同机性能对照条件。最终正式训练仍需采用明确的有效batch，并保留230,400有效样本及按样本计算的warmup/验证节点；生成器会重新派生更新次数，不自动放大学习率。

特别是H20 B4/depth2要求至少两份microbatch，因此有效batch至少8。若最终采用B4A2/有效batch8，每次更新只有两份任务，与本次B4A6的六份任务相比，流水线填充/收尾占比和更新频率会改变。**不能将B4A6的2.259727帧/s直接标为B4A2的实测速度。** 本轮不提前替其他五组选择最终训练配方，也不启动完整训练。
