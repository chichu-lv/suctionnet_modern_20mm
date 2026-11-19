# L20 三分钟实测后的优化与执行说明

> **最新更新：并行负载与单跑结果。** 用户实机报告中，缓存配置和命中率基本不变，暂停正式训练并将A/B错开后，吞吐由10.49/21.87提升至16.89/35.20实例/秒。新增结果与限制见第1.1节；这是负载条件对照，不是缓存开关收益。以下128条仅为容量/协议试点，不是默认提速部署配置。

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

### 1.1 后续并行与单跑实测：共享读取受负载影响

以下来自用户随后提供的两次报告，原始远端文件仍未在此实现环境独立读取。[逐项转录、分组口径和派生量](../../research/instance_runs/20261119/l20-pipeline-load-comparison-user-reported.json)分别记录18:20:00–18:23:00并行窗口，以及具体开始时刻未提供的后续两次单跑窗口，未补造单跑时间戳或运行commit。

两次均启用各路线独立的源NPY/静态几何目录；用户确认代码、缓存、输入清单、种子和有效batch32相同，源请求命中率约11%–12%，覆盖仍是128/1088。单跑时先暂停两组正式训练，再由A/GPU2、B/GPU3先后各测180秒；结束后用户报告原训练已恢复，A从3434推进到3440，B从3657推进到3666。这是用户提供的恢复进度，不是本环境重新执行或独立校验checkpoint。

| 项目 | A：并行 → 单跑 | B：并行 → 单跑 |
|---|---:|---:|
| 完整更新 | 59 → 95 | 123 → 198 |
| 吞吐，实例/s | 10.49 → 16.89 | 21.87 → 35.20 |
| 平均步时，s | 3.030 → 1.877 | 1.453 → 0.907 |
| P95步时，s | 4.442 → 2.515 | 1.932 → 1.038 |
| 数据等待，s/步 | 2.708 → 1.563 | 1.018 → 0.459 |
| 非等待时段，s/步（总时减等待） | 0.322 → 0.314 | 0.435 → 0.448 |
| GPU平均利用率 | 4.95% → 8.47% | 9.85% → 14.83% |
| 峰值显存，MiB | 5660 → 6732 | 4294 → 4300 |
| NPZ回退实例的压缩成员读取，ms | 147.8 → 10.3 | 118.5 → 10.0 |
| 全部worker实例平均的压缩读取，ms | 130.3 → 9.1 | 105.1 → 8.8 |
| 几何/特征准备含包查找，ms/实例 | 157.0 → 154.4 | 23.0 → 21.9 |

按完整更新计算，吞吐分别提高61.02%和60.98%。A每步减少1.153s，其中数据等待减少1.145s；B每步减少0.546s，数据等待减少0.559s、非等待反而增加0.013s。改善集中在供数，不能归因为这两次之间发生了新的网络优化。

该对照支持“共享原始NPZ读取路径在较高并发负载下变慢”，且不支持把主要原因归为A/B缓存目录混用。它同时改变了正式训练背景负载和benchmark并发，未控制OS页缓存冷热，不能进一步认定是某一个作业、A/B两条benchmark相互竞争、页缓存挤占或物理磁盘带宽单独造成。两次缓存都开启，不能据此断言缓存本身使训练提高61%。也不能直接拿单跑16.89/35.20与17:30的13.51/29.51计算实现优化净收益。

单跑后A等待仍占83.3%，几何154.4ms/实例最突出；B等待50.6%、前向40.2%，其zlib32.9ms加其他数组解析/构造/元数据73.7ms约106.6ms/实例。源NPY查找仅A2.5/B2.3ms，重复目录探测可优化，但当前不再是主要优先项。worker时间不直接加到主线程步时，不据此推算全覆盖训练倍率。

当前应保留训练超参数，分别验证A扩大静态几何覆盖、B减少解压和数组物化的收益，并在稳定负载后验证预期并发场景。全量源NPY仍有存储与页缓存成本，不能因为单跑读取恢复就直接宣布全量展开更快。

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

这两类磁盘缓存是针对A/B的新增方案，不是最终E5已验证的相同部署方式。最终E5实际为按需NPZ读取、每进程90项场景/88项物体辅助数据缓存、数据预取和在线几何重叠，adapter_cache为null；见[历史实际配置](https://github.com/chichu-lv/suctionnet_modern_20mm/blob/0080c36ec52946e3275fc563cf46ebeea359edb0/docs/experiments/CD010-20261117/pretraining/run_summary.json)。

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

先使用 `--max-records` 做明确的有限试点，查看容量、验证读命中和训练随机采样命中率，再决定覆盖范围。`--max-records`不改变训练采样清单；128/1088仅覆盖11.76%，已经与实机低命中相符。试点构建成功与热页缓存CPU收益均不是端到端部署验收。不要因为局部mmap耗时短就默认把全部数据搬进RAM或展开到没有容量预算的磁盘。

扩大覆盖前使用服务器真实构建报告估计容量；128条大小乘8.5仅是1088条的粗估，不是上界，另留样本差异与临时空间。源NPY按reserved/logical预算，稀疏allocated不能替代门限。source根的complete只表示所选前缀完成，需核对selected_records与原训练总数。几何全覆盖需同时满足records_total/records_selected/entries_available均为1088、complete=true、budget_exhausted=false；预算不足可能成功发布部分包。

A可以先扩静态几何，不必同时扩源NPY；同来源/配置/版本可--resume复用128条，或在独立目录构建。B优先评估源镜像对解压和数组构造的收益，其扩范围必须新输出目录。构建与计时分开；已恢复的正式训练不由这份文档自动暂停，不能在共享存储上边构建边把受干扰窗口作为性能基线。

## 5. 构建与启用

以下变量由执行者填写为**原训练的真实路径**，A/B各在自己的checkout、环境和缓存目录操作；示例的128条只用于构建/容量/命中试点，不是完整训练覆盖或已验证的提速部署配置。既有source/selection manifest保持原路径。下方180秒命令是试点诊断入口，应与第6节相同负载下的缓存开关对照一起解释，不能单独作为部署验收。

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

沿用GPU2/GPU3、固定A6/B8 workers、prefetch2、microbatch32/有效batch32、原数据与增强种子。必须分别记录“单路线且正式训练暂停”“仅A/B benchmark并跑且正式训练暂停”“与正式训练等后台作业共存”等负载条件，不混成同一速度趋势。当前16.89/35.20是单跑基线；预期正式并发吞吐仍需独立窗口。

若需进一步区分竞争来源，缺少的控制组是正式训练暂停条件下仅A/B benchmark并跑，与当前单跑比较；应在已安排的测试窗口进行，不自动中断用户已恢复的训练。记录同时运行的作业、原数据/缓存挂载、读取延迟、CPU与内存/页缓存情况，保持并说明两组的测量条件。

新benchmark按窗口结束前**已同步完成**的完整更新计数；跨线更新完整执行后单列为overrun，不能中断optimizer或计入180秒吞吐。均值/P95按窗口内完整更新计算；吞吐分母为180秒，另报包含overrun的实际墙钟。预算提前耗尽不算完成窗口。

新benchmark仍是独立随机初始化的临时训练测量，不加载/写入正式checkpoint，也不做选模；不能把不同权重状态的运行当作严格同轨迹数值对照。生产checkpoint切换另走原严格resume，已有CPU状态恢复回归不是CUDA逐位一致性的证明。

缓存收益要在同一负载条件下配对。A优先固定源读取方式，仅比较静态几何覆盖；B优先固定静态几何配置，仅比较源NPY。若要拆解双缓存交互，再比较全部磁盘缓存关闭、仅静态几何、双缓存，输出到不同新目录。关闭一个环境变量即可保留当前冷路径；无需改变manifest/batch/学习率。worker内LRU开关也必须相同；若单独测首次磁盘访问，可两组都设 `SUCTION_INSTANCE_STATIC_CACHE_MB=0`。

保持现有训练超参数，不把4/4 workers或prefetch1写成已验证最佳值。独立benchmark可用--worker-counts测试并发，但正式resume严格比较config/execution，直接改workers/prefetch会被拒绝。磁盘缓存根和覆盖可在原来源/几何合同下切换；直接搬迁manifest或改变源路径/文件身份会影响数据身份和已有缓存有效性，不能当成无条件可用的续训参数。

每份 `benchmark.json` 的 `preparation_caches` 同时报告内存LRU、`reader.source_hits/source_fallbacks/npz_reads` 以及 `geometry_store.hits/missing/stale/invalid`。reader的观测和标签各算一次请求；几何store只在内存LRU未命中时查磁盘。统计是整个trial已返回准备结果的累计量，包含预热、预取和可能的过线更新，不等同于180秒内恰好消费的实例数；不能把两种命中数直接相除。环境记录原生扩展版本/能力、B静态协议和缓存根目录；能力存在不等同于实际调用次数。

缓存构建必须在计时前结束，避免构建进程改变CPU/存储竞争。评价指标优先为固定窗口完整实例吞吐、更新P95、等待和阶段耗时，并记录host RSS、页缓存/磁盘实际占用及GPU allocated/reserved。继续采样GPU利用率，但不设“填满显存”目标。只有新分段显示传输或GPU占主导，再决定pinned buffer/H2D、B同步点、A并集或compile的优先级。

## 7. 当前边界

- 本地最终核心回归A927/B951项通过；完整GitHub CI与更详细验证范围见 [MODEL_VALIDATION](../MODEL_VALIDATION.md)和对应PR checks。
- 已补入用户提供的双缓存并行和单跑L20结果；实现环境未独立读取远端JSON/CSV、确认实机commit或执行GPU复测。约61%提升来自负载条件对照，缓存净收益及预期正式并发吞吐仍未由该对照分离。
- pinned memory／异步H2D完整链路仍未接入；B的CUDA `nonzero` 支撑选择同步点保留，等待实机定位。
- 约88%的源请求仍回退NPZ；低并发下压缩读取已恢复到约10ms/回退实例。共享读取对负载敏感的证据已增强，但具体竞争来源、全覆盖随机读和多worker页缓存压力仍需区分。
- A静态包不含网络特征；B静态包不含本次曝光、训练相位或任意query投影结果。
- 模型输入语义、几何FP64边界、source-ID采样、标签、权重结构和训练预算保持原样。用户已报告原正式训练恢复并继续推进；本次文档更新不改运行配置、不触碰checkpoint、不启动或停止远端作业。
