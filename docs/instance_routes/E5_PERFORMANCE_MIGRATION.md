# E5 性能经验在实例 A/B 路线中的定向适配

日期：2026-11-19（北京时间）。本轮接续 A `4264e4d14319fcf230bb43a43d27ef582da94ae8`、B `f7a3ffc83f4993e3543dcc8ece7080b3dc905596`；更早的 L20 正式窗口和第一轮局部优化不计入本轮收益。

**上一轮审查没有覆盖全部性能能力。** [第一轮记录](E5_PROBLEM_AUDIT.md)只完成按需排除四个数组、A 固定统计、B 反向切片及指标读回。本页补齐完整接触计算、静态几何、路线读取协议、ROI 消费者和传输边界的检查，并逐项区分已经实现、可选配对路径与仍待实机验证的事项。

## 1. 历史依据与诊断优先级

E5 早期的 33.6 秒更新中，接触分组约 22 秒，数据准备约 7 秒；最终 CD010 E5 实际启用 compiled FPS、原生 grid/native_cpu 接触、持久 spawn、pin_memory、zone gather、split 和 depth=2。最后主线程数据等待 6.594 ms、几何等待 183.757 ms，相对于 17.377 秒更新合计约 1.095%。后台 11.227 秒几何工作是累计工作，不能与 GPU 时段相加。

依据：[历史分解](../PERFORMANCE_HISTORY.md#42-最初-e5-的约-336-秒更新耗在哪里)、[最终实际配置](../experiments/CD010-20261117/pretraining/run_summary.json)、[最终等待与利用率证据边界](../experiments/CD010-20261117/pretraining/REPORT.md)。显存不能替代 GPU 利用率曲线；完整场景帧和单实例、H20 和 L20 之间也不能直接比较秒/更新。

当前归档 A 20 步窗口仍约 88.2% 时间等数据，B 约 64.8%。A 的优先对象是完整接触准备及供给能力，主干 2,048 点 FPS 与局部区域 FPS 应分开。现有归档 JSON 没有单列当前主干 FPS 的约 8 ms 条目，不能把旧报告的全部 FPS 合计时间当作该子项。

## 2. 完整能力清单

| 能力/问题 | 本轮状态 | 实际边界 |
|---|---|---|
| A 既有 SpConv、编译 FPS | 保留 | 不据此推断整个接触计算已原生化 |
| 持久有序 spawn、线程限制、pending ticket 恢复 | 保留并作集成回归 | 源几何可在 worker 提前完成，不机械复制 E5 的 depth=2 或 8 线程 |
| A 完整接触查询、三区筛选、局部 FPS、计数、描述子和唇口摘要 | 新增 A 专用原生批量入口 | 复用现有 C++ grid、线程、buffer 生命周期；保留 A 自己的 FP64 合同 |
| A 静态几何与动态 RGB 拆分 | 已实现 | 缓存无 RGB、模型特征、权重或 observation 引用；每次重新绑定本次颜色 |
| A 接触并集后按需插值/投影 | 可选训练路径 | `SUCTION_A_CONTACT_UNION=1`；默认关闭，保持公共 encode/任意姿态接口，需真实 GPU 配对 |
| A/B 按消费字段读取 NPZ/NPY | 已实现路线白名单 | 保留原 query/pixel/normal/label 校验和完整 archive 成员冲突检查 |
| 紧凑训练包 | 进一步精简 | A 及早释放验证专用整图；B 不读取/生成无消费者的 support 字段，原 source count 独立保留 |
| B sample-first 与 ROI | 延伸到准备和消费者 | 冷准备先裁有效几何区域；消费者按全局相机像素访问 ROI，不恢复整图零数组 |
| B 跨曝光静态图像几何 | 已实现 | 不含 phase 或 query 投影结果；本次 RGB 在采样位置重新组成网络输入 |
| B 相位随成功更新步变化 | 保留并验证 | worker 只准备姿态身份，实际消费时生成相位 |
| compact tensor / pinned buffer / H2D 重叠 | 已审计，未新增完整链路 | B wire 可接受 CPU tensor，但本轮没有把普通 NumPy 的 non_blocking 上传称为 pinned 链路，也未默认启用未经 GPU 测量的 pinning |
| 全图剩余工作 | 明确保留 | NPZ 几何成员仍需解码、冷路径仍需 mask bbox 扫描、RGB 增强仍输出完整尺寸 RGB；不能声称所有 H×W 成本消失 |
| batch/显存目标/compile | 未重扫 | 先用新分段数据确定瓶颈是否转移 |
| 旧实例融合、环境碰撞五区、E5 权重 | 不适用 | 未引回新方法 |

## 3. A 完整原生接触适配

新增 `models/csrc/instance_route_a_contact.h`，接入现有 `_spatial_index_cpu` 扩展的 `route_a_contact_query`。Python 入口为 `native_contact_geometry()`；原逐候选实现独立保留为 `reference_contact_geometry()`。

保留 A 的 core/inner/lip 三区、10/18/22 mm 半径、±40 mm 高度、FP64、边界容差、source-ID 首点和并列顺序。即使区域点少于预算，也执行 A 的原 FPS 顺序。唇口统计用采样前的全部 lip 点计算 RMS、最大空缺角及谐波。原生 capsule 不进入静态缓存或 IPC，默认一个几何线程，继续由现有实例 worker 并行。

一次对完整候选批调用接触计算，再按网络 query chunk 切分，避免按每个 chunk 重建网格。原生数学与 NumPy 可能有 FP64 末位差异；区域/高度/角支持判定接近舍入敏感阈值的行，回到当前 A 参考实现。采样索引、掩码、计数和角支持必须精确一致；连续描述量采用紧 FP64 容差并报告最大误差。

`SUCTION_A_CONTACT_BACKEND` 支持 `auto`（默认）、`native`、`reference`。`auto` 在扩展缺失或 ABI 不匹配时保留参考路径；实机验收用 `native` 明确要求原生引擎，不允许静默退回参考实现。切换 backend 会区分静态缓存 key。固定工作量的配对不需要改变模型配置或 checkpoint 身份。

另有可选的接触点并集路径：compact 包记录当前 canonical contact 实际使用的原 support 行并集，训练 `forward_instances` 在 `SUCTION_A_CONTACT_UNION=1` 时仅对这些行插值和投影，并重映射 gather。公开 `encode_instance/encode_instances` 和评价默认保留完整支持特征；私有 partial encoded 如果遇到并集外的新姿态，会保留 autograd 地补齐完整特征。空并集也保留零梯度关系，不把原来的零参数梯度改成 None。

固定合成 CPU 配对使用两实例各 65,536 support、128 query 和显式注入的 Linear/BN 骨干：插值/投影 **131,072→24,538 行**。FP32 全部预测精确相同，40 个有梯度参数的最大绝对差 `1.746e-9`；同输入 BF16 对照预测相同、最大梯度差 `5.960e-8`，几何与 BN buffers 精确相同。验收容差 `atol=1e-7, rtol=2e-5`，未放宽默认路径既有精确 compact 回归。CPU 累计正算子分配 2,551,773,112→721,983,800 B，不是峰值显存；注入骨干前反向时间不是 SpConv/CUDA 时间。原始记录：[接触并集 CPU 对照](../../research/instance_runs/20261119/route-a-contact-union-cpu.json)。此开关默认关闭，先做真实 GPU 配对。

## 4. 静态包、有界缓存和恢复

接触并集开关应在创建 preparation workers 之前设置。默认关闭时，worker 也不增加未消费的并集排序或 wire 数组；不带并集提示的已有票据安全使用完整参考特征路径。

A 的 `prepare_source_static_geometry()` 保存固定索引、关联、查询分块和描述量，移除颜色；`prepare_source_geometry(..., static_geometry=...)` 只绑定本次 RGB，复核设置、候选和 RGB validity。静态包不保存原 observation 引用。

B 的 `prepare_source_static_inputs()` 保存独立持有的几何 ROI、原图尺寸/原点、采样像素、几何通道与仿射参数。`prepare_source_inputs_cpu(..., static_inputs=...)` 从本次增强图像采样 RGB 后组成 context/texture；静态包不含 query phase 或网络特征。

两路线使用每 worker 独立 LRU：默认 **64 MiB 计费上限、16 条目上限**；`SUCTION_INSTANCE_STATIC_CACHE_MB=0` 禁用保留。计费包括数组底层 owner 和空间树，采用保守估计，不等同于 RSS。过大条目不驻留。缓存仅在 `SUCTION_COMPACT_SOURCE_IPC=1` 的源准备路径使用，普通公共 reader、source test 和 target 合同保持原样。

key 包含明确版本、路线、几何设置、query prefix、来源身份、路径、设备/inode、文件大小和 mtime；A 还区分是否使用 RGB 增强。读取前后复核 key，期间发生替换或移除的 draw 不入缓存。它不是内容完整性认证，不进行文件内容扫描。

缓存不持有 RNG、不进入 checkpoint。命中、未命中、分配到另一 worker 和冷恢复都可以按同一 ticket 重算。现有采样顺序、增强随机参数和 pending ticket 恢复继续由 coordinator 管理。基准 JSON 新增每个准备池的 hits/misses/evictions/retained_bytes，真实命中率由实际运行决定；合成 warm 数字不能代表随机大数据集的命中率。

## 5. 读取与紧凑协议

`SOURCE_TRAIN_ARRAYS` 分别列出 A/B 消费字段；NPZ 和 NPY/mmap 使用同一 `needed_arrays` 机制。无论是否读取数组，完整成员表都参与 metadata 冲突检查。验证所需的 XYZ/normal map 仍会读取，原 query/pixel/normal 和 label identity 检查没有删掉。

- A 验证后释放 `xyz_map`、`normal_map`，不再让两个整图数组跨越颜色增强和接触准备阶段；无增强时仍保留模型消费的 support RGB。
- B 不读不消费的 support RGB、UV、normal、component、source-ID 等字段，也不生成扁平 support RGB。原 `support_points` 本轮仍会解压，读取其原长度后释放，写成私有 `source_support_count`。没有将训练器原 gate 改成 B 模型的有效图像像素计数。
- 私有 route prefix 使用已有数组的 view，label alias 复用对应数组；普通公共读取/test 保留原 copy 行为。mmap 保持 copy-on-write 映射。
- B compact wire 有明确 `route_b_training_roi_fp32_geometry64_v2` spec、字段、shape/dtype 与坐标范围检查。按原图全局像素访问 ROI，保留 source ID、离图、半像素最近邻、四角深度规则及任意姿态 fallback。

## 6. 本轮 CPU 对照结果

环境为 Python 3.12、Torch 2.5.1+cpu、NumPy 2.3.5，单线程 Torch/BLAS，A 原生扩展用 GCC 构建。以下均为合成 CPU 分段，**没有新的 L20/H20 吞吐或 GPU 利用率结论**。

| 测量 | 基线中位数 | 当前中位数 | 解释 |
|---|---:|---:|---|
| A 完整接触：32,000 support、128 query | 178.411 ms | 115.960 ms | 原生包含构建网格；参考复用其已有树；接触阶段约 1.539× |
| A 整体 prepare：另含 2,048 主干点 | 266.474 ms | 冷准备 215.250 ms | 7 轮轮转对照，约 1.238×；不是训练倍率 |
| A 已有静态包，仅绑定本次 RGB | — | 2.455 ms | 理想命中，不含缓存查找、读取、增强、IPC 或网络 |
| B prepare + compact：720×1280、32,000 有效像素 | 125.235 ms | 冷准备 12.424 ms | 9 轮交替；先裁几何区域再处理 |
| B 已有静态包，当前 RGB 与打包 | — | 3.602 ms | 理想命中，真实训练命中率未测 |
| B CPU consumer 转换 | 10.412 ms | 0.173 ms | 不恢复完整相机几何数组；不是 CUDA H2D 时间 |
| 源读取：上一轮四字段排除 | 117.97 ms | A 120.47 / B 122.57 ms | 本轮白名单未测出额外加速，不能仅凭字节减少宣称提速 |

A 原生接触离散结果全部精确一致；描述子最大绝对误差 `3.33e-16`，唇口摘要 `8.88e-16`，比较容差 `rtol=atol=2e-13`。整体 prepare 的 query/chunk 与本次 RGB 完成对照，warm guard 明确禁止再次调用几何构建。B 每轮动态曝光的所有 encoded 数值与基线精确一致，并完成模型预测及参数梯度对照。

B consumer 几何数值数组从 46,080,000 B 降到 1,600,000 B；wire 数值数组从 4,860,792 B 降到 4,092,792 B。它们是本合成输入的数组字节数，不是峰值显存。A/B 路线白名单相对上一轮再少读取约 1.84/2.11 MB；A 验证后可提前释放 44,236,800 B 的两张整图。

原始逐轮记录：[A 原生接触](../../research/instance_runs/20261119/route-a-native-contact-cpu.json)、[A 静态准备](../../research/instance_runs/20261119/route-a-static-geometry-cpu.json)、[B 静态 ROI](../../research/instance_runs/20261119/route-b-static-roi-cpu.json)、[路线 reader](../../research/instance_runs/20261119/route-reader-cpu.json)。

## 7. 验证和启用边界

专用验证覆盖 native/reference、FP64 边界、source-ID/FPS、线程确定性与 handle 生命周期；缓存字节预算/底层 owner/来源替换；不同增强下的预测与梯度；serial/spawn 命中及冷恢复；source/target pending ticket、采样、RNG、模型与优化器状态；B ROI 的 empty/full/interior/edge、离图和四角规则。继续执行 `tests/test_no_sha256_verification.py`。

完整本地与 GitHub CI 结果记录在 [MODEL_VALIDATION](../MODEL_VALIDATION.md)。正确性和恢复检查不替代性能覆盖；局部 CPU 加速也不替代正式数据与真实 GPU 的端到端检查。

各路线 checkout 中可复现：

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
CC=gcc CXX=g++ python setup.py build_ext --inplace

# A checkout
python research/instance_runs/20261119/benchmark_route_a_native_contacts.py --output /tmp/a-contact.json
python research/instance_runs/20261119/benchmark_route_a_static_geometry.py --output /tmp/a-static.json
python research/instance_runs/20261119/benchmark_route_a_contact_union.py --output /tmp/a-union.json

# B checkout
python research/instance_runs/20261119/benchmark_route_b_static_roi.py --output /tmp/b-static.json

# 任一路线：按相同生产格式构造的合成压缩观测
python research/instance_runs/20261119/benchmark_source_reader.py --output /tmp/route-reader.json
```

实机使用既有 `scripts/benchmark_instance_training.py`，固定当前路线 L20 配置、有效 batch 32、microbatch、worker、样本顺序和增强种子，预热单列，记录完成实例/同步墙钟、数据等待、准备 cache 命中/驻留量、host RAM 和 GPU allocated/reserved。A 明确设置 `SUCTION_A_CONTACT_BACKEND=native`；接触并集路径单独做开/关配对。不能为了填满显存改变模型或训练预算。

本轮不切换正式训练。真实 SpConv/CUDA 数值与原检查点状态验证、实际供给/吞吐配对完成后，再按原采样和恢复契约从 checkpoint 切换。是否补 pinned buffers、H2D 重叠及其优先级，以改动后的实机分段结果决定。
