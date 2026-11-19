# L20 三分钟实测后的优化与执行说明

日期：2026-11-19（北京时间）。本轮接续 A `4809d6b39ecf11dc31c0ae14c08acdb316be2e46` 与 B `d1c600b1b8ee81f49c9b988c51e217a311699e3b`。代码分别在 `optimize/route-a-e5-audit`、`optimize/route-b-e5-audit`；不改变模型、标签、有效 batch、学习率和 checkpoint 数据身份。本页是本轮新增执行入口；[上一轮性能能力矩阵](E5_PERFORMANCE_MIGRATION.md)保留历史证据。

## 1. 本轮依据及证据边界

用户提供两条路线在 L20 GPU2/GPU3 同时测试的 180 秒结果：17:30:08–17:33:08，预热2步，有效batch32，每卡359次GPU采样。

| 项目 | A / 6 workers | B / 8 workers |
|---|---:|---:|
| 完整更新 / 有效实例 | 76 / 2,432 | 166 / 5,312 |
| 实例吞吐 | 13.51/s | 29.51/s |
| 更新均值 / P95 | 2.361 / 3.102 s | 1.079 / 1.247 s |
| 主线程数据等待 | 2.048 s / 86.7% | 0.538 s / 49.9% |
| 主线程前向时段 | 0.170 s | 0.456 s |
| GPU利用率均值 / 峰值 | 7.3% / 92% | 15.5% / 74% |
| worker几何准备 | 210.3 ms/实例 | 23.1 ms/实例 |
| worker zlib + 其余NPZ解析 | 141.6 ms/实例 | 131.4 ms/实例 |

这些值来自用户消息；提供的服务器 JSON/CSV 路径在当前工作区不可访问，未独立读取原始文件、代码commit或缓存命中配置。逐项转录见[用户实测记录](../../research/instance_runs/20261119/l20-window180-user-reported.json)。不把此表标为本轮改动后的GPU复测。旧 A 61.2% 是启动附近10秒均值，应以这次180秒的7.3%描述该稳态窗口。

主线程前向墙钟不等于纯CUDA kernel时间；worker单实例服务时间不能直接叠加到每步主线程墙钟。A首先继续减少冷几何和供数；B同时处理NPZ供数和消费端确定的重复投影几何。未重新扫描batch、显存目标或compile。

## 2. 本轮已经实现的内容

### A 冷几何

- 对不超过32,768点且source-ID有序的紧凑支持云，连续半径扫描直接生成原顺序结果，避免每候选排序；更大或无序输入保留网格路径。
- 局部FPS区域坐标只整理一次，分离距离更新与argmax，最后一轮不更新无人使用的距离；保持FP64运算顺序、source-ID破平局和原始唇口摘要。
- 单组件且行身份完全相同时复用已有cKDTree。
- 原生实现来源版本为 `route_a_contact_v2_contiguous_fps`，ABI为1。可选CPU分段profile默认关闭，不增加训练路径的clock读取。

### B 消费端重复几何

- 静态ROI保存四角像素有效、深度跨度不超过10mm的布尔表，投影槽一次查表；这不依赖训练相位或权重。
- 最近像素的XYZ、mask、normal和normal_valid复用同一ROI索引；同批候选的环方向和三角函数只计算一次。
- 公共可变encoded没有静态表时，继续实时计算四角规则；不让调用者修改深度后使用旧表。
- 静态版本为 `route_b_static_image_geometry64_rgb_indices_quad_v2`，训练wire为 `route_b_training_roi_fp32_geometry64_quad_v3`。

### 两条独立的只读缓存

1. **源数组镜像：** `build_instance_source_cache.py` 将明确选中的观测和标签成员展开为标准NPY。原manifest与路径不变，每次训练仍作原有schema、姿态、标签及有限值检查。训练只读，缺失/过期回退原NPZ，记录实际命中。
2. **静态几何包：** `build_instance_geometry_cache.py` 预计算与颜色、相位、权重无关的结果，保存小JSON布局和单个可mmap的NPY字节arena。首次访问也可复用，不依赖样本再次落到同一worker。A保留任意姿态回退所需的索引输入，只有调用该API时才建树；canonical训练不重建树。

两者使用A/B独立目录与环境变量，不替换训练manifest，不进入checkpoint。key包含来源路径/设备/inode/长度/mtime、查询前缀、几何配置和明确版本；A还区分RGB增强启用位与接触后端。不存在文件内容摘要或首次全文件校验。文件读取/发布期间替换会拒用缓存；损坏布局/非法shape、索引或有限值回退冷准备。

每worker的既有内存LRU仍有64MiB保守计费、16项上限，包含底层arena和惰性树预留。磁盘缓存没有在线增长：预先构建，训练仅打开映射。实际RSS、页缓存占用和磁盘覆盖率必须另外测量。

## 3. 已取得的CPU配对证据

Python3.12、Torch2.5.1+cpu、NumPy2.3.5，Torch/BLAS单线程。计时均为合成CPU分段，不是新的L20更新耗时。

| 范围 | 基线中位数 | 当前中位数 | 原始记录 |
|---|---:|---:|---|
| A原生完整接触，32k支持/128候选 | 120.566 ms | 91.973 ms | [冷几何配对](../../research/instance_runs/20261119/route-a-cold-geometry-cpu.json) |
| A完整冷几何，另含2048主干点 | 224.874 ms | 188.664 ms | 同上 |
| A当前冷准备+颜色绑定+compact，对比每draw重开静态包 | 180.071 ms | 5.003 ms | [A持久包](../../research/instance_runs/20261119/route-a-static-store-cpu.json) |
| B投影槽读取 | 41.403 ms | 27.746 ms | [B消费者](../../research/instance_runs/20261119/route-b-consumer-geometry-cpu.json) |
| B完整contact输入 | 66.909 ms | 51.577 ms | 同上 |
| B当前冷准备+颜色输入+compact，对比每draw重开静态包 | 12.917 ms | 5.061 ms | [B持久包](../../research/instance_runs/20261119/route-b-static-store-cpu.json) |
| A源读取+验证+触达全部选中数组，NPZ→NPY | 128.045 ms | 9.643 ms | [NPY读取](../../research/instance_runs/20261119/route-source-mmap-cpu.json) |
| B源读取+验证+触达全部选中数组，NPZ→NPY | 118.955 ms | 8.334 ms | 同上 |

A原生配对在同编译参数下分别编译旧/新实现，所有接触输出和静态prepared数组逐元素相同。B消费者比较使用旧完整model+preparation，19个contact tensor、纹理梯度，以及8实例完整网络的256个预测/诊断tensor和126个参数梯度逐元素相同。B每个32k像素ROI增加32,000字节布尔表，预计算约0.791ms；wire数值数组由4,092,792增至4,124,792字节，以一次固定准备替代反复读取。

持久包和NPY读取均是**热OS页缓存**；静态包每draw都真实重新打开、解析布局和执行语义验证，未使用内存LRU。A读/验证2.478ms、本次RGB绑定2.431ms；B读/验证1.611ms、本次RGB输入3.415ms。每条packet（含JSON）A6,409,950B、B4,413,216B。不同曝光重新构建static的值相同，逐轮绑定后的RGB不同且与冷路径完全一致，A延迟索引始终未构树。

这些独立基准的输入/轮次/环境波动分别记录，不能把不同表行的时间相减或倍率相乘来预测训练速度。未计入的范围包含RGB增强、真实IPC、GPU网络以及未命中记录的冷磁盘访问。

## 4. 部署前先看磁盘与覆盖率

NPY以减少解码换取更多存储。720×1280/32k支持的合成观测中，压缩NPZ约1.12MB；路线NPY逻辑长度约60MB。显式 `--sparse` 对全零**原始字节**块留洞，保持负零/NaN位模式和标准mmap格式；本例实际分配约31–33MB，仍明显大于NPZ。文件系统不支持稀疏洞时仍按普通NPY工作。

源镜像 `--max-bytes` 以逻辑长度和保守元数据/分配块余量预算，不依赖稀疏文件节省。几何缓存 `--max-bytes` 限制已发布packet的字节数；并行构建时还需为有限个临时packet预留空间。两个构建器都报告实际覆盖，不能把只缓存前若干记录说成全训练命中。

先使用 `--max-records` 做明确的有限试点，查看容量、验证读命中和训练随机采样命中率，再决定覆盖范围。`--max-records`不改变训练采样清单；随机大数据集的命中率可能很低。不要因为局部mmap耗时短就默认把全部数据搬进RAM或展开到没有容量预算的磁盘。

## 5. 构建与启用

以下变量由执行者填写为**原训练的真实路径**，A/B各在自己的checkout、环境和缓存目录操作；示例的128条是试点，不是完整训练覆盖。既有source/selection manifest保持原路径。

### A：GPU2，原生接触、NPY及静态几何

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export SUCTION_COMPACT_SOURCE_IPC=1
export SUCTION_A_CONTACT_BACKEND=native
export SUCTION_A_CONTACT_UNION=0

CC=gcc CXX=g++ python setup.py build_ext --inplace
python -c 'from models import _spatial_index_cpu as m; print(m.ROUTE_A_CONTACT_SOURCE_VERSION); assert m.ROUTE_A_CONTACT_SOURCE_VERSION == "route_a_contact_v2_contiguous_fps"'

# 设置 A_SOURCE_MANIFEST、A_SELECTION_MANIFEST、A_BACKBONE_ROOT、A_CACHE_ROOT、A_BENCH_ROOT。
# A_CACHE_ROOT必须是独立的新缓存根目录，基准输出也必须是新目录。
python scripts/build_instance_source_cache.py \
  --manifest "$A_SOURCE_MANIFEST" --route a \
  --output "$A_CACHE_ROOT/source-npy" --training-queries 128 \
  --max-records 128 --max-bytes 17179869184 --workers 4 --sparse
export SUCTION_INSTANCE_SOURCE_CACHE_A="$A_CACHE_ROOT/source-npy"

python scripts/build_instance_geometry_cache.py \
  --manifest "$A_SOURCE_MANIFEST" --config configs/instance_a_l20.yaml \
  --output "$A_CACHE_ROOT/static-geometry" \
  --max-records 128 --max-bytes 2147483648 --workers 4
export SUCTION_INSTANCE_GEOMETRY_CACHE_A="$A_CACHE_ROOT/static-geometry"

CUDA_VISIBLE_DEVICES=2 python scripts/benchmark_instance_training.py \
  --config configs/instance_a_l20.yaml --backbone-root "$A_BACKBONE_ROOT" \
  --source-manifest "$A_SOURCE_MANIFEST" --selection-manifest "$A_SELECTION_MANIFEST" \
  --device cuda --warmup-updates 2 --measure-seconds 180 \
  --output "$A_BENCH_ROOT/cache-enabled"
```

`native`要求当前原生路径可用，不能静默回退。接触并集保持0以单独评估本轮变化；以后按原约束在创建worker之前设1，另做真实GPU数值及吞吐配对。

### B：GPU3，NPY、静态ROI与消费端查表

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export SUCTION_COMPACT_SOURCE_IPC=1

# 设置 B_SOURCE_MANIFEST、B_SELECTION_MANIFEST、B_CACHE_ROOT、B_BENCH_ROOT。
python scripts/build_instance_source_cache.py \
  --manifest "$B_SOURCE_MANIFEST" --route b \
  --output "$B_CACHE_ROOT/source-npy" --training-queries 128 \
  --max-records 128 --max-bytes 17179869184 --workers 4 --sparse
export SUCTION_INSTANCE_SOURCE_CACHE_B="$B_CACHE_ROOT/source-npy"

python scripts/build_instance_geometry_cache.py \
  --manifest "$B_SOURCE_MANIFEST" --config configs/instance_b_l20.yaml \
  --output "$B_CACHE_ROOT/static-geometry" \
  --max-records 128 --max-bytes 2147483648 --workers 4
export SUCTION_INSTANCE_GEOMETRY_CACHE_B="$B_CACHE_ROOT/static-geometry"

CUDA_VISIBLE_DEVICES=3 python scripts/benchmark_instance_training.py \
  --config configs/instance_b_l20.yaml \
  --source-manifest "$B_SOURCE_MANIFEST" --selection-manifest "$B_SELECTION_MANIFEST" \
  --device cuda --warmup-updates 2 --measure-seconds 180 \
  --output "$B_BENCH_ROOT/cache-enabled"
```

源镜像断点恢复需同一选中范围/格式加 `--resume`；扩大试点范围应使用新输出目录。几何缓存可同目录 `--resume` 扩大覆盖和预算，但不能缩小既有选中范围，不能更换来源manifest/几何版本。构建日志为本地存储来源元数据，不是文件内容完整性证书。

## 6. 180秒验收与控制项

沿用GPU2/GPU3并测场景、相同CPU/存储资源、固定A6/B8 workers、microbatch32/有效batch32、原数据与增强种子。新benchmark按窗口结束前**已同步完成**的完整更新计数；跨线更新完整执行后单列为overrun，不能中断optimizer或计入180秒吞吐。均值/P95按窗口内完整更新计算；吞吐分母为180秒，另报包含overrun的实际墙钟。预算提前耗尽不算完成窗口。

新benchmark仍是独立随机初始化的临时训练测量，不加载/写入正式checkpoint，也不做选模；不能把不同权重状态的运行当作严格同轨迹数值对照。生产checkpoint切换另走原严格resume，已有CPU状态恢复回归不是CUDA逐位一致性的证明。

建议在相同新版代码上依次配对：全部磁盘缓存关闭、只开source NPY、再开static geometry，输出到不同新目录。关闭一个环境变量即可保留当前冷路径；无需改变manifest/batch/学习率。worker内LRU开关也必须相同；若单独测首次磁盘访问，可两组都设 `SUCTION_INSTANCE_STATIC_CACHE_MB=0`。

每份 `benchmark.json` 的 `preparation_caches` 同时报告内存LRU、`reader.source_hits/source_fallbacks/npz_reads` 以及 `geometry_store.hits/missing/stale/invalid`。reader的观测和标签各算一次请求；几何store只在内存LRU未命中时查磁盘。统计是整个trial已返回准备结果的累计量，包含预热、预取和可能的过线更新，不等同于180秒内恰好消费的实例数；不能把两种命中数直接相除。环境记录原生扩展版本/能力、B静态协议和缓存根目录；能力存在不等同于实际调用次数。

缓存构建必须在计时前结束，避免构建进程改变CPU/存储竞争。评价指标优先为固定窗口完整实例吞吐、更新P95、等待和阶段耗时，并记录host RSS、页缓存/磁盘实际占用及GPU allocated/reserved。继续采样GPU利用率，但不设“填满显存”目标。只有新分段显示传输或GPU占主导，再决定pinned buffer/H2D、B同步点、A并集或compile的优先级。

## 7. 当前边界

- 本地最终核心回归A927/B951项通过；完整GitHub CI与更详细验证范围见 [MODEL_VALIDATION](../MODEL_VALIDATION.md)和对应PR checks。
- 本轮未取得改动后的真实L20复测，没有改写用户提供的180秒结果。
- pinned memory／异步H2D完整链路仍未接入；B的CUDA `nonzero` 支撑选择同步点保留，等待实机定位。
- 原NPZ未覆盖记录仍需解压；NPY映射不保证常驻RAM，真实随机读和多worker页缓存压力待测。
- A静态包不含网络特征；B静态包不含本次曝光、训练相位或任意query投影结果。
- 模型输入语义、几何FP64边界、source-ID采样、标签、权重结构和训练预算保持原样。正式训练先保留原运行，实机验证后从原检查点按原采样状态切换。
