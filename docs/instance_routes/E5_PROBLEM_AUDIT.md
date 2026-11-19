# 用 E5 的问题记录检查新 A/B 路线

日期：2026-11-19（北京时间）。本轮以旧问题的成因为线索，逐项检查新模型的实际调用链。基线为 A `59e05c0934bf07ea5c6fb47818c61966b3c8bb02`、B `e6d9578f5fb39174cf9a81c769672800c06a9306`。本轮没有移入 E5 的模型、接触区域、执行器或历史 checkpoint 迁移代码。

## 1. 逐项判定

| E5 留下的排查线索 | 新 A/B 的证据与判定 | 本轮处理 |
|---|---|---|
| 数据生产与 GPU 消费串行，主线程等数据 | 两条新路线已有有序 spawn 预取、限制 worker 线程和紧凑 IPC；现有 L20 报告仍有明显数据等待 | 保留这些已有实现；继续检查生产者与消费者各自还在做的工作 |
| 解压和搬运下游不用的数组 | `_read_npz()` 原来读出全部 archive 成员；源域紧凑训练后续没有使用 `depth_m`、`depth_valid`、`eligible_mask`、`component_map` | 仅在既有紧凑 source 训练 worker 路径跳过这四个数组的物化；普通读取、选模、目标域仍走完整读法 |
| 固定 CPU 几何仍占用 GPU 消费线程 | A 的 `encode_instances()` 在 backbone 启动后逐实例计算全 support 距离分位数；`d1` 同一统计计算两次 | 在现有几何 worker 中一次计算诊断并传递；复用 `d1` 统计 |
| 重复可微切片让反向为完整输入分配梯度 | B 的实例特征分发、纹理桶、环带区域、dense query 分块和结果分发均有这种结构；CPU profiler 已确认 | 沿本路线的实例、环带和 query 边界分别 `split`，保留原来的纹理尺寸分桶与 GroupNorm |
| 每个 microbatch 读回 loss 引入同步 | 多 microbatch 路径逐组 `.cpu().tolist()`，单实例及目标路径逐次 `float(loss.detach())`；当前 L20 microbatch=有效 batch=32，通常仅一组，不能据此称为当前主要瓶颈 | 暂存 detached 标量，成功 optimizer 更新后将 source/target loss 和裁剪前 norm 一次读回；仍在 CPU 按原顺序计算均值 |
| 接触区域重复 gather、实例广播反向热点 | A 存在 raw support 特征插值和分块索引，但没有本轮真实 GPU 耗时证据；B 已按实例编码，无 E5 的完整多实例点级广播结构 | 不改 A 的特征计算或引入 E5 的分段广播、五区接触头 |
| worker 退出与待处理任务导致恢复偏移 | 新执行器已使用私有 pipe、异常传播、有限队列，并保存已发未消费 ticket；已有恢复回归 | 保留现有执行器；复跑 worker 失败、关闭和精确恢复测试 |
| 大 batch、dense chunk、精度模式不能自动换来更快训练 | 当前正式配置为有效 batch 32；已有 FP64 几何、BF16 dense/FP32 sparse 等各路线边界 | 本轮不调整有效 batch、microbatch、学习率、训练预算、BN/GN、候选数或几何精度 |

历史依据：[性能历史](../PERFORMANCE_HISTORY.md)的按需 NPZ 读取、切片反向诊断与 P3 split；[从头训练优化](../FROM_SCRATCH_PERFORMANCE.md)的每更新 loss 读回；[接触头优化](../CONTACT_HEAD_OPTIMIZATION.md)的 gather 范围；[几何重叠](../GEOMETRY_GPU_OVERLAP.md)的生产/消费依赖。旧报告中的硬件倍率、字段名、块大小和退役迁移规则不作为新路线的实现要求。

## 2. 本方案中的修改边界

### 2.1 源数据读取

训练 worker 继续读取同一 NPZ、相同 metadata、支持点、RGB、组织点图、法向图和全部当前 query/label 校验需要的字段。保留 observation/label schema、query prefix、原像素坐标与法向对齐、标签身份、有限性等语义检查。metadata 与数组重名的检查使用完整成员表，即使该数组本次没有物化也不能被 metadata 覆盖。

此优化跟随已存在的 `SUCTION_COMPACT_SOURCE_IPC=1`，仅 source 的 train 记录生效；无需重打包缓存、修改 manifest 或更换标签。普通 `load_observation()` 的完整返回保持不变。固定选模的本地 NPY 映射读取也识别相同的排除字段，并保留完整成员表检查。没有增加内容摘要或扫描。真实训练基准同时记录紧凑 IPC 和选模缓存环境变量，便于解释测量条件。

### 2.2 A 的固定统计

诊断值随本次 observation 的 `_route_a_geometry` 一起生成和传递，属于临时准备结果；没有跨增强复用几何。单实例直接调用仍计算一次，预取消费者复用已经计算的值。支撑不足、非标准姿态、输出及梯度回归继续覆盖原行为。

### 2.3 B 的反向分配

逐实例的 `tensor[i:i+1]` 改为对批特征一次 `split(1)`；不同尺寸纹理仍分别编码，避免改变 GroupNorm。接触 token 沿九圈的半径轴分为三个区域；dense chunk 按原块大小分发；最终输出按各实例实际 query 长度拆分。没有扩大块大小或更改投影采样、相位、掩码、模型参数与 checkpoint 键。

### 2.4 更新边界的指标读回

反向调用顺序和 source/target 原损失权重保持不变。保留所有有限性检查及 `clip_grad_norm_(..., error_if_nonfinite=True)`；异常梯度不执行 optimizer、不增加成功步或样本预算。临时保留的是 detached 标量，没有跨 microbatch 持有 loss 计算图。两条当前 L20 配置均为 microbatch 32/有效 batch 32，通常每更新只有一次 source microbatch；减少重复 loss 同步主要作用于更小 microbatch、跳过后补采样以及目标微调路径。当前默认路径把原来分开的 loss 与 norm 读回合为一次，未测其独立GPU收益。

日志的旧 `source_loss_backward_readback_host_seconds` 更名为 `source_loss_backward_host_seconds`，因为读回已移到 `clip_optimizer_readback_host_seconds`。历史 JSON/CSV 不改写。两项是主机墙钟区间，不能当作 CUDA kernel 时间或直接与旧分项相减；完整更新的比较继续使用已同步的端到端基准。

## 3. 本轮证据

本机为 Python 3.12、PyTorch 2.5.1+cpu、NumPy 2.3.5；所有本轮测量使用合成输入，未连接训练 GPU。机器可读结果见 [CPU 审查记录](../../research/instance_runs/20261119/e5-problem-audit-cpu.json)。

- **A 固定统计**：32 个合成关联数组，support 数为 `[6923,32768,65536,143048] × 8`，9 轮交替测量。去除重复 `d1` 后统计耗时中位数由 **161.831 ms 降至 107.406 ms / 32 实例**；预取时消费者不再计算这些分位数。完整统计字段逐项相同，数值和每轮耗时保存在 JSON 中。
- **B 反向临时分配**：8 个合成实例、21 个 query、两个纹理尺寸桶、dense chunk 4096。`slice_backward` 从 261 次降到 155 次；累计输出分配从 49,876,788 B 降到 12,509,568 B。计入新增 split 反向的 `cat` 后，相关累计分配为 20,494,692 B，较原路径减少 58.9%。这是累计分配，不能解释为峰值显存或吞吐增幅。基线与修改版全部预测相同，126 个参数梯度最大绝对差为 0。
- **训练指标与状态**：原分支训练器与当前训练器在 pretrain/finetune、microbatch 1/2/5 共六种合成设置中各运行三次更新。覆盖未知标签跳过和最后不满有效 batch 的更新；指标、模型、优化器、采样器、RNG 与 checkpoint 状态逐项完全相同，计时字段除外。
- **源读取**：使用相同生产格式的合成 NPZ 对比完整/按需成员读取，720×1280、32,000 support、256 query，5轮交替测量，保留原验证的读取中位数由 **151.147 ms 降至 119.008 ms**，每实例少物化 **12,902,400 B**。保留数组与 metadata 全部相同。metadata JSON 只有5,698字符，解码与清理中位0.258 ms；本次证据指向数组解压/物化，不能把旧profile中的组合耗时都解释为JSON解析。此局部测量不代表真实存储系统、并行 worker 或 L20 的每步收益。

完整回归结果和实际命令登记在 [模型验证记录](../MODEL_VALIDATION.md)。新测试覆盖预取后不再重算 A 统计、B 不等 query/多纹理桶/尾 chunk 梯度、训练指标一次读回及非有限拒绝、选择性读取与完整 metadata 冲突检查。继续运行禁止内容校验扫描门禁。

### 可复现入口

分别在对应路线的 Git checkout 中执行。需要本地包含上文基线 commit；脚本只读取历史源码，不切换工作树、不写权重。输出位置由调用者指定。

```bash
# 两条路线均有：相同生产格式合成 NPZ 的完整/按需读取
python research/instance_runs/20261119/benchmark_source_reader.py --output /tmp/source-reader.json

# A 分支：固定邻域统计
python research/instance_runs/20261119/benchmark_route_a_diagnostics.py --output /tmp/route-a-diagnostics.json

# B 分支：反向累计分配和预测/参数梯度对照
python research/instance_runs/20261119/benchmark_route_b_splits.py --output /tmp/route-b-splits.json
```

## 4. 与此前 L20 记录及后续实机验收的关系

本轮基线已经包含此前的紧凑 IPC、A 精确 FPS/接触几何优化和 B 图像通道准备优化。[A 已有 profile](https://github.com/chichu-lv/suctionnet_modern_20mm/blob/59e05c0934bf07ea5c6fb47818c61966b3c8bb02/research/instance_runs/20261119/l20-preparation-profile.json)与 [B 已有 profile](https://github.com/chichu-lv/suctionnet_modern_20mm/blob/e6d9578f5fb39174cf9a81c769672800c06a9306/research/instance_runs/20261119/l20-preparation-profile.json)中的吞吐变化属于前一轮，不能记到本轮修改名下。

本轮没有发布新 L20 加速倍率，也没有切换或重启正式训练。实机验收使用已有 `scripts/benchmark_instance_training.py`，固定各路线原 L20 配置、有效 batch 32、microbatch、worker 数、相同样本次序及增强种子，先测 disposable 更新；预热单列。比较完成有效实例数/同步墙钟、数据等待、host RAM、GPU allocated/reserved 以及 loss/梯度有限性。A 还需真实 SpConv 骨干验证；CPU 注入骨干不能代替它。

只有完成该配对后，才将当前修改用于正常断点续训并报告新窗口。权重与优化器结构未改变；继续按现有身份、pending ticket 和采样/RNG 规则恢复。保持当前正式样本预算和评价定义，无需引入 E5 的旧迁移系统。
