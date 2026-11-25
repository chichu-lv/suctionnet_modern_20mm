# 模型验证与当前训练状态

## 2026-11-19 L20三分钟实测后的冷路径与只读缓存优化

本轮代码基线 A `4809d6b39ecf11dc31c0ae14c08acdb316be2e46`、B `d1c600b1b8ee81f49c9b988c51e217a311699e3b`。用户提供17:30:08–17:33:08、L20 GPU2/GPU3、预热2步、batch32的180秒窗口，A13.51/B29.51实例每秒，数据等待86.7%/49.9%。服务器原始文件在当前工作区不可访问；[转录记录](../research/instance_runs/20261119/l20-window180-user-reported.json)明确标为用户提供，未确认测试commit或缓存启用，未将其当作本轮代码的GPU复测。最新命令、缓存预算/覆盖与配对步骤见 [L20执行文档](instance_routes/L20_PIPELINE_OPTIMIZATION.md)。

实现：A有序紧凑支持云免排序查询、连续局部FPS与同序单组件树复用，native来源版本 `route_a_contact_v2_contiguous_fps`；B静态四角有效性表、最近像素ROI索引和环方向复用；A/B独立source NPY/mmap镜像（可选标准稀疏NPY）及单arena持久静态包。缓存均显式离线构建、有磁盘预算、训练只读，保持原manifest、标签和checkpoint身份。不同来源/配置/几何版本失效，数组shape/索引/有限性及读写代际复核保留，无内容摘要。

本地 Python3.12.14/Torch2.5.1+cpu/NumPy2.3.5，Torch/BLAS单线程，gcc/g++原生扩展。最终两路线核心套件执行 `python -m pytest -q --ignore=tests/test_prefetch.py`：**A 927 passed / 5 skipped（124.97s）**、**B 951 passed / 6 skipped（137.48s）**。包括全部新增执行代码、禁止内容校验60项、串行池统计隔离和最后缓存有效性修正；两个工作树 `git diff --check` 通过。旧DataLoader三项在此本地环境曾阻塞，继续交给不排除文件的GitHub完整CI；最终提交结果见 [PR #9](https://github.com/chichu-lv/suctionnet_modern_20mm/pull/9)、[PR #10](https://github.com/chichu-lv/suctionnet_modern_20mm/pull/10) checks，不把本地排除版本记为完整CI。

针对性证据：A native/流式FPS/模型/noSHA 96 passed；B既有模型73 passed、新consumer+noSHA 82 passed；新source镜像最后专项+noSHA70 passed；静态存储/loading/恢复组 A94/B92 passed（另有另一条路线条件skip）；新180秒benchmark+noSHA两路线各94 passed。这些均与最终套件重叠，不累加为独立覆盖。

缓存验证包含：metadata及完整成员冲突、路径碰撞、同长度/mtime源文件原子替换、映射期间替换、布局/形状/索引/有限值拒绝、COW不污染、预算与断点恢复、负零/NaN位模式、A惰性索引任意姿态回退、第一次disk命中与新RGB、serial/spawn，以及pretrain/finetune各3步连续与暂停冷恢复后的模型/AdamW/采样/RNG/pending ticket/source-target trace。180秒假时钟验证截止前完整更新、正好截止、跨线完整执行后排除、CUDA同步计入时间、预算提前耗尽失败与旧20步默认保持。

CPU配对：A旧/新完整接触120.566→91.973ms、冷几何224.874→188.664ms，所有contact/prepared数组逐元素一致。B slot读取41.403→27.746ms、完整contact输入66.909→51.577ms；19个contact tensor/纹理梯度和完整网络256预测诊断tensor/126参数梯度逐元素一致。NPY读取并触达全部选中数组 A128.045→9.643ms、B118.955→8.334ms，所有数组/metadata一致。逐draw重开静态包+语义校验+新RGB+compact A180.071→5.003ms、B12.917→5.061ms，未使用LRU且所有准备值精确一致。后两项为热OS页缓存，不能推定冷随机磁盘或L20端到端收益。

静态包单实例示例 A6.41MB/B4.41MB；源NPY逻辑约60MB/实例，标准稀疏NPY实际分配示例约31–33MB，原NPZ约1.12MB。已记录磁盘权衡，不默认无界展开数据。原始逐轮JSON、存储对照与复现脚本均在[20261119研究目录](../research/instance_runs/20261119/)。本轮未运行真实GPU、切换正式训练、修改模型权重结构/batch/学习率，也未启用未经实机验收的pinned/H2D或A接触并集默认路径。

## 2026-11-19 第二轮：补齐性能能力适配与静态准备

基线：A `4264e4d14319fcf230bb43a43d27ef582da94ae8`、B `f7a3ffc83f4993e3543dcc8ece7080b3dc905596`。上一轮不是完整性能能力继承验收；本轮逐项结果、启用条件、原始逐轮数据和剩余项见 [定向适配报告](instance_routes/E5_PERFORMANCE_MIGRATION.md)。

实现 A 专用原生三区接触（复用现有空间网格/线程/缓冲管理）、完整候选一次准备、静态几何与当前 RGB 分离，以及默认关闭的接触并集按需插值/投影。B 将静态图像几何与当前 RGB 分开，几何 ROI 保留到 consumer，不再恢复完整相机零图。共同 reader 使用路线白名单、验证后释放不用的整图、prefix view/label alias 少拷贝，并保留原 flattened support 训练计数。每 worker 缓存默认64 MiB计费上限/16条目，包含底层owner计费、来源读取前后key复核、冷恢复验证。没有新增文件内容校验。

本地 Python3.12.14、Torch2.5.1+cpu、NumPy2.3.5，gcc/g++构建可选扩展，Torch/BLAS单线程。核心回归：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m pytest -q --ignore=tests/test_prefetch.py
```

结果：**A 889 passed / 5 skipped（101.44s）**，**B 897 passed / 6 skipped（111.76s）**。末轮针对性组另有 **A 90 passed / 2 skipped**、**B 81 passed / 4 skipped**；最后串行/worker共用选定record读取、来源替换和禁止内容校验门禁，两路线各 **63 passed**。这些与核心套件重叠，不累加为独立覆盖。新增 skip 为另一条路线专属用例；未通过跳过来掩盖新增实现失败。新原生头文件也已验证包含在 source distribution 中。

本地仍排除已知在当前执行环境阻塞的旧 `tests/test_prefetch.py` 三项；该文件和旧 DataLoader 实现未修改，前一轮已单独定位 `_try_get_data` 等待。GitHub CI 继续执行不排除文件的完整 `python -m pytest -q`，最终提交的状态与日志以 [A PR #9](https://github.com/chichu-lv/suctionnet_modern_20mm/pull/9)、[B PR #10](https://github.com/chichu-lv/suctionnet_modern_20mm/pull/10) 的 checks 为准。本地结果不记为未排除的完整套件。

关键验证包括：当前A参考 vs native离散采样/边界/FP64量，单/多原生线程确定性、handle不入IPC，A/B不同曝光预测和梯度，缓存owner内存预算、读取期间文件原子替换、serial/spawn与选定record顺序；compact路径下pretrain/finetune各3步连续与暂停冷恢复，模型、优化器、采样、augmentation RNG、pending ticket及source/target trace一致。A并集路径额外验证FP32/BF16、BN buffers、空并集的零梯度关系、公开encode/eval完整特征和任意姿态惰性补齐autograd。

合成CPU分段：A完整接触178.411→115.960ms；整体prepare冷路径266.474→215.250ms，已有static仅绑定当前RGB为2.455ms。B prepare+compact 125.235→12.424ms，理想缓存命中3.602ms，CPU consumer转换10.412→0.173ms。A并集示例插值/投影131072→24538行，FP32/BF16预测误差0，梯度最大绝对差分别1.746e-9/5.960e-8。路线白名单进一步少物化数组，但读取中位数没有超过上一轮四字段方案，未宣称额外读取加速。

所有计时均为明确固定的CPU合成输入；warm数字不代表正式训练命中率，也不含完整I/O/增强/IPC/H2D/网络。真实SpConv/CUDA数值、L20吞吐、实际缓存命中和资源分段仍待同配置配对。A接触并集默认关闭，pinned/H2D完整链路本轮未新增；没有切换正式训练、改变模型参数/标签/batch/学习率或训练预算。


## 2026-11-19 按 E5 历史问题审查新 A/B 路线

检查基线：A `59e05c0934bf07ea5c6fb47818c61966b3c8bb02`，B `e6d9578f5fb39174cf9a81c769672800c06a9306`。审查结果、适用边界、逐项证据和复现命令见 [E5问题检查报告](instance_routes/E5_PROBLEM_AUDIT.md)，原始 CPU 数值见 [机器可读记录](../research/instance_runs/20261119/e5-problem-audit-cpu.json)。

本轮针对新实现确认的重复开销，分别修改紧凑 source worker 的按需数组读取、A 的固定邻域统计位置、B 的可微批特征拆分，以及训练器的更新边界指标读回。现有模型参数、候选/标签、增强/采样、有效 batch、损失权重及 checkpoint 状态定义保持原样。当前 L20 microbatch32/有效batch32下通常只有一组 source；多组读回问题主要作用于更小microbatch、补采样及目标微调，不能称为当前主要瓶颈。

本轮环境 Python3.12.14、PyTorch2.5.1+cpu、NumPy2.3.5，单线程Torch/BLAS，已用gcc/g++构建本仓库可选CPU扩展。分别在两条独立分支运行：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m pytest -q --ignore=tests/test_prefetch.py
```

结果：**A 851 passed / 5 skipped（73.04s）**，**B 861 passed / 4 skipped（82.57s）**。skip为测试原有的真实发布数据/另一条路线条件，不添加新skip。结果包括本轮新增的数组按需读取、完整成员碰撞/损坏query拒绝、A预取诊断、B不等query/多纹理桶/尾块梯度、指标读回及非有限梯度拒绝，以及当前实例执行器真实spawn、pending ticket、预训练/微调恢复测试。两条路线共用的测试不重复累计为独立覆盖。

最初尝试完整 `pytest -q`，两分支均在原有 `tests/test_prefetch.py::test_spawn_prefetch_matches_serial_and_resumes_consumed_state` 停止推进；单独复现的20秒faulthandler栈停在Torch DataLoader的`_try_get_data`等待，45秒后由外部超时终止。该本地阻塞已在此前验证记录出现，本轮未修改旧prefetch实现或测试。完整GitHub CI仍运行未排除任何文件的`python -m pytest -q`；CI结果以对应PR checks为准。没有把排除该文件的本地结果记为完整套件通过。

A的32实例统计合成对照为161.831→107.406ms；B相关反向累计分配49,876,788→20,494,692B，并且256个预测/诊断tensor和126个参数梯度均与原版完全相同；720×1280源观测的保留验证读取中位数151.147→119.008ms、少物化12,902,400B。训练器原版/新版在六种合成设置中各执行三次更新，模型、优化器、采样/RNG、checkpoint及非计时指标全部相同。全部为CPU合成证据，未测新L20/H20吞吐、真实SpConv反向或机器人效果；本轮未切换正式训练。

## 2026-11-18 B模型支撑门槛、坐标图与结构身份复核

已对最新B分支逐项核实并修复模型内部缺失的 `min_support_points=100` 门控及 `architecture_metadata()`。此前共享 `infer_instance` 已能拒绝小碎片；仍存在缺口的是直接模型评分、目标动作loss及评价。现在B以实例内有限且正深度的原始XYZ像素计数，法向有效性和另传扁平点表不改变该计数；低于可配置门槛时所有query无效。结构报告记录真实13/5输入、128+64+96+18+12+12=330维融合、577个独立投影位置、精度、归一化和中心权重语义。

新的 `two_fixed_z_charts_rodrigues_z05_v1` 在n_z>0.5时使用+Z图。-Z参考图分母是1−n_z，奇点在n_z→+1；新规则全域分母下界为0.5，不能误写成1.5。原float64实现的1°例子不是必然NaN，此次扩大数值余量。两坐标图切换仍可产生切向roll跳变，72槽循环移位不变性不等于任意roll重采样不变性；新增测试明确保留这一限制，不声称已消除接缝。

`r=0` 仍将单次中心观测广播为72槽：support前两维为(c,1−c)，raw core覆盖权重为1/3，独立来源像素计数仍为1。逐角度通道LayerNorm及前后清零保持不变，330维融合和权重形状不变。完整中心token可能因逐角重力或切向偏移项不同，不能把所有通道都描述成72份相同值。

新B配置显式保存图版本和支撑门槛，Trainer保存真实 `identity.model_architecture`。缺少图字段的历史checkpoint继续使用原v2图，不能静默换成新采样网格；新门控仍会修复旧低支撑行为。微调初始化比较图/门槛等语义，并核对已存在的结构身份；旧源权重可按显式legacy配置开始新微调。旧B checkpoint缺少新结构身份时不允许精确resume；新B同配置恢复通过。此变更不改变查询p/n与teacher物理定义，不要求仅因此重算物理标签。A的恢复身份没有新增B字段。[实现与兼容规则](instance_routes/IMPLEMENTATION.md#4-暂停恢复与身份检查)

共享固定test子集、无文件内容校验、分片隔离合并、逐模块模式恢复均已在上一轮进入A/B；本轮同步新的共享配置/初始化检查。默认选模继续使用90场景各8固定帧、每875步一次。A/B仍保持独立模型分支，B修复没有把A稀疏模型或12通道骨干覆盖进B分支。

最终本地联合源码回归 **755 passed / 1 skipped（32.34s）**，命令为 `python -m pytest -q --ignore=tests/test_prefetch.py`，包括60项禁止SHA-256执行代码门禁。新增20项真实B边界/加载案例及12项共享配置/兼容案例；B合成官方格式资产的teacher→训练→checkpoint→恢复→推理/评价集成通过。两极、1°、0.5±epsilon、99/100/101点、目标动作拒绝、中心权重和声明矛盾均有覆盖。唯一skip需要真实R20发布资产；本地多进程预取文件由两分支完整GitHub CI执行，不修改测试或加入新skip。环境为Python3.12.14、PyTorch2.5.1+cpu、NumPy2.3.5。未执行完整真实数据训练、GPU延迟或机器人成功率测量。

## 2026-11-18 固定test子集选模与独立全量报告

当前A/B默认 `selection_frames_per_scene: 8`、`evaluate_every: 875`。全部90个test场景各使用固定帧 `0,32,64,96,128,160,192,224`，共720帧；8750个成功更新共评价10次。所选帧全部可见实例、每实例完整最多256个候选以及指标归约保持，少支持/no-pose实例继续计入分母。每次选模帧数为完整test的1/32，10次共7200帧；这一比例是帧遍历预算，不是实测墙钟加速比。小子集排序有抽样波动，不保证等同于全test最优checkpoint。

训练可使用完整test清单自动筛选，或只构建指定720帧的正式选择缓存；两者都要求90场景全部指定帧有实例或明确GT-empty声明，缺帧不回填。正式固定子集不需要 `--allow-partial-manifests`。新 `instance_selection.py` 仅检查清单与小型覆盖元数据，未选NPZ不加载，子集身份保存固定采样规则、帧集合及实际有序实例ID。旧配置缺少帧数字段时仍按256解释，full view保持旧dataset identity，允许按原配置恢复。

best继续标记 `test_selected`，另保存 `selection_scope=fixed_test_subset`。普通resume对固定子集定义严格比较，旧runtime兼容分支也不能放宽子集成员或顺序；checkpoint加载和FT初始化检查保存配置、policy与selection定义自洽，微调来源保留源子集。训练后的独立全量评价入口始终检查90×256帧，报告 `evaluation_scope` 与 `source_pretrain_selection_scope`，不改写或重新选best。子集选模及包含它的全量报告均保留非独立测试身份。

当前实施命令与8/16/256帧配置见[运行说明](instance_routes/IMPLEMENTATION.md)；下方两次完整test选模是上一轮修复历史，已由本节默认方案替代。完整真实数据构建、训练与GPU耗时仍需训练环境验证。

本轮本地A/B联合回归 **723 passed / 1 skipped（29.51s）**，命令为 `python -m pytest -q --ignore=tests/test_prefetch.py`。真实R20发布条件项保留原skip，多进程预取文件由各分支完整GitHub CI验证。新增32项固定子集数据/覆盖测试及16项真实trainer/CLI接线测试：两轮只加载相同的718个实例（720帧中两帧GT-empty）、90场景及三类分母正确；连续与暂停恢复的模型/Adam/best相同；同长度换ID或重排在当前/旧runtime均拒绝；FT来源保留子集；不自洽checkpoint在模型构造前拒绝；正式subset训练CLI无需partial标志；完整评价实际包含非选择帧1。全套包含60项禁止SHA-256的执行代码门禁。这些是合成输入及CPU运行证据，不是完整真实数据精度或GPU提速测量。

## 2026-11-18 全量运行前的评价、分片与初始化边界修复（上一轮）

本轮将A/B默认source选模计划改为第 **4375、8750** 个成功更新节点，每次仍遍历完整90测试场景×256帧；100训练场景、140000成功实例/8750更新、每875步保存last及目标域val选模口径保持原样。显式 `evaluation_steps` 与旧 `evaluate_every` 二选一，最终预算步必评，暂停不新增节点。报告保存实际评价节点、清单范围和 `timing.wall_seconds/observations`；耗时不包含随后的checkpoint保存。best仅从两个评价节点选择，可能错过中间峰值。按此前五次全量评价计算，减少60%的遍历次数；尚未测得完整真实数据的墙钟节省比例。

手工 `--scenes/--frames` 构建现在在同一输出根下生成按camera/split/实际ID范围区分的可读目录，观测、标签、manifest和summary一起隔离。无显式子集的完整构建路径保持不变。非阻塞OS锁限制同目录单writer；不同分片可并发；合并拒绝遗漏旧帧的覆盖发布。两个数据CLI输出实际manifest绝对路径。旧缓存仍可读取、合并，不自动搬迁，也不进行内容摘要计算。[路径与完整命令](instance_routes/IMPLEMENTATION.md#2-构建共同source缓存)

FT初始化只放宽A的 `backbone_root/query_chunk_size` 和B的 `query_chunk`，其余模型字段、质量及数据合同继续比较；源checkpoint内部保存定义先验证自洽，权重严格加载，并记录源/目标执行设置。推理原有骨干路径覆盖保留，三个CLI新增统一 `--query-chunk-size`。同一运行的普通resume仍要求完整配置一致。[配置边界](instance_routes/IMPLEMENTATION.md#4-暂停恢复与身份检查)

推理先过滤无效query，仅对有效行校验点/轴和评分，保留原始query/source ID及拒绝顺序；无效占位的零/NaN不再触发单位化异常。少支撑且无有效query统一返回 `insufficient_support`。正常与异常退出的逐子模块模式恢复继续保留，混合BN状态有实际回归覆盖。

本轮最终本地A/B联合源码回归 **675 passed / 1 skipped（26.36s）**，命令为 `python -m pytest -q --ignore=tests/test_prefetch.py`。跳过项需要真实 `R20_RELEASE_ROOT`；本地已知受环境限制的多进程预取文件交由各分支GitHub完整CI，不修改测试或新增skip。新增针对性用例覆盖实际CLI同根目录串行/并发分片、同分片进程锁、workers与安全合并、路径/分块FT初始化、坏配置拒绝、三CLI接线、无效查询/原始ID映射，以及减少评价节点后的连续训练与暂停恢复状态一致性。全套包含60项禁止SHA-256的执行代码门禁。运行环境沿用Python3.12.14、PyTorch2.5.1+cpu、NumPy2.3.5；没有执行完整真实数据构建/训练、真实SpConv/GPU时延或机器人试验。

## 2026-11-18 CPU 热路径、状态恢复与无内容校验和修复（上一轮）

按项目要求，所有现行执行代码移除 SHA-256 计算与验证，旧元数据只读兼容且不重新输出；永久约束及回归门禁见 [NO_CONTENT_CHECKSUMS.md](NO_CONTENT_CHECKSUMS.md)。共同数据保留真实 schema、p/n、标签公式、物理定义、配置和 split/group/capture 检查。旧 pretrain 只核对仍可得的实际配置/运行字段；旧 target sampler 若仅有不可还原的观察 ID，明确不支持精确续训，可使用权重初始化新运行。

A 的批量邻居关联在 15,000 支撑点、2,048 主干点下由 **187.740 ms 降至 17.473 ms（10.745×）**；最近锚点的256-query批次由1.815 ms降至0.163 ms。结果与冻结旧标量实现逐字节一致。报告含每次采样、1次预热/7次测量及依赖版本，见 [CPU组件基准](performance/INSTANCE_NEIGHBORS_CPU.json)。范围排除索引构建、FPS、网络、I/O和GPU，不能转述为整条推理时延。

评价上下文恢复每个子模块原有 training 标志，正常和异常退出都保持微调冻结BN；5mm支撑判定统一使用1e-12m舍入容差，A最小支撑阈值读模型配置，B当前回退100。source保持100训练场景和完整90测试场景选best，默认1750步一次共5次，checkpoint保存间隔875步，140000成功实例预算不变。每次评价保存汇总，明细流式写gzip，仅best更新/当前best重复评价/最终步保留，临时文件正常或异常清理。

最终本地A/B联合源码回归 **620 passed / 1 skipped（22.39s）**，命令为 `python -m pytest -q --ignore=tests/test_prefetch.py`。唯一skip需要真实 `R20_RELEASE_ROOT`。本地完整运行在多进程预取处超时，因此只将该文件交由各分支GitHub完整CI，未修改其测试或新增跳过。另已通过135项训练/推理/评价与禁用规则测试、23项共同数据/B CLI集成、29项基础几何参考检查；这些集合有重叠，不相加。B集成覆盖builder、暂停/恢复、微调、raw RGB-D推理与评价；两场景 `--workers 2` 构建及10项CD010旧脚本smoke通过。没有在本环境执行真实SpConv/GPU、全量训练或机器人成功率验证。

## 2026-11-18 A/B代码实现与当前执行政策

本轮已实现共同数据/teacher、A和B网络、预训练/两阶段目标微调、严格checkpoint恢复、原始RGB-D/无标签推理与native评价。实际运行入口见[IMPLEMENTATION.md](instance_routes/IMPLEMENTATION.md)。source直接训练scene0000–0099，完整scene0100–0189 test选择best并保存test_selected身份；last另保留。目标1676/399/87划分不变。公开仅半径评价deferred，不是开训门禁。下方较早的90/10→refit方案是修订历史，不覆盖当前规则。

本地环境Python3.12.14、PyTorch2.5.1+cpu、NumPy2.3.5。新测试覆盖真实teacher合成资产、v3法向/嵌套回填、A CPU注入backbone与完整梯度、B实际GN18网络和投影门控、未知标签损失、阶段BN冻结、优化器/采样/RNG恢复、checkpoint和动作来源身份、NMS及精确SE(3)。A没有在本环境执行真实SpConv/CUDA。全部测试用合成或内存数据，不宣称取得真实数据准确率或机器人成功率。

B额外运行实际CLI：source暂停1步→恢复至2步，与连续2步对照；每步8个train查询、test完整256候选、RGB增强开启。模型126个张量、优化器、采样器、Python/NumPy/Torch及增强RNG逐项完全相同；7条build后训练/推理/评价CLI成功。无标签缓存与原始RGB-D的Top5输出逐字段相同，SE(3)合法。best停在step1，last在step2，报告清楚区分test_selected与budget_endpoint_or_progress。小汇总在[B分支验证记录](https://github.com/chichu-lv/suctionnet_modern_20mm/blob/route-b-2d-projected-rings/docs/instance_routes/route_b/CPU_INTEGRATION.json)；脚本级复现也由 `tests/test_instance_route_b_integration.py`覆盖。

本地联合快照回归为 **481 passed / 21 skipped / 1 deselected**（38.99s）；命令排除 `tests/test_prefetch.py`，并跳过 `test_prefetched_trainer_preserves_ordinary_sequence_and_resume`，这些多进程预取路径交由实际GitHub checkout中的完整CI验证。真实release/GPU等条件项仍按测试自身条件skip。原地初次快照缺Git元数据造成的旧测试失败已通过建立本地测试快照解决，不修改旧训练器。本地新代码测试与远端分支CI状态分别登记，不把旧129项NumPy参考或历史E5测试计数冒充新模型运行。完整真实source/target训练、A GPU稀疏骨干、部署时延与TensorRT导出均未执行。该轮旧checkpoint记录过摘要身份；当前运行按上方无内容校验和规则保存实际配置/数据/teacher定义，FT保存源checkpoint路径/大小和test选择来源。

## 较早修订记录


## 2026-11-18 A/B 公开方法成对比较方案

两分支新增[公共对照合同](instance_routes/PUBLIC_METHOD_COMPARISON.md)和[论文数值目录](instance_routes/reference/paper_baselines.json)，各实验页提供本路线分别与S1B、Diffusion-Suction、UISN UOIS/UOAIS、MultiGraspNet vacuum/joint的六组成对表。本方法各格保持待测；MultiGraspNet未报告的Top1保持缺失，其评价物理语义未核实，暂作论文参考。没有新增第三方重训、参考方法互比或虚构A/B成绩。

本次直接核对S1B v2 Table IV、Diffusion v2 Table I、MultiGraspNet v2 Table I，以及固定官方API/基线和UISN代码。UISN表格沿用仓库既有原PDF核验转录，本轮未重新读取全文；引用身份明确区分。公开训练采用90/10开发后100场景source-only固定预算last。推理按无GT工作区的UOIS、UISN工作区的UOIS/UOAIS分为三种profile；后两项GT联合前景范围确实改变候选池，不能扩大授权至其他比较。

已审查原项目R20与官方评价在Seal归一化、力矩范数、碰撞和后处理上的差异，规定sn-paper-radius-v1仅参数化三处半径、保留官方±1mm环与全场景六指标。固定官方物理函数的有限合成边界检查观察到严格相机Z轴法向可产生非有限Wrench/Q，原阈值判定为未命中；该历史行为及空帧扩展分别记录，不能据此声称适配层或真实数据评价已完成。

本轮验收限于文档/机器可读来源的一致性、90个已报告指标数值、18个缺失指标、每分支成对表、链接、输入/输出合同和分支差异。生产模型、公开exporter、半径适配实现、R10等价验收、100场景refit、全测试与GPU性能仍待执行。旧129项为已有数学/schema参考，未改写为本轮新模型测试数；CI状态按本轮提交查看。

## 2026-11-18 A/B 独立分支实施文档

从共同的 [v3 设计及架构图提交 a9f4a84](https://github.com/chichu-lv/suctionnet_modern_20mm/commit/a9f4a8451c6bc42cfd8867dd96c051398644ad71) 分别整理 [A：route-a-3d-rgb](https://github.com/chichu-lv/suctionnet_modern_20mm/blob/route-a-3d-rgb/docs/instance_routes/route_a/README.md) 与 [B：route-b-2d-projected-rings](https://github.com/chichu-lv/suctionnet_modern_20mm/blob/route-b-2d-projected-rings/docs/instance_routes/route_b/README.md)。每条路线提供六篇实施计划及一个总入口，覆盖数据准备、预处理、模型设计、后处理、训练和实验。

本轮文档审查核对了划分/标签身份、输入及头部维数、精确历史姿态、预算与损失归约、冻结/恢复、排序/NMS、指标分母、实验阶段和导出边界。A 的 teacher 步骤明确保留 raw Seal 到冻结归一化 S20 的转换；B 明确支持摘要按每半径两字段交错、raw 摘要按每区域四字段排列。

共同训练页细化两条实施口径：无有效源监督的实例不计成功监督预算；次级 selected-Q 与主指标采用相同三层宏平均、no_pose=0，并另报有输出条件下的微平均。任何所需 teacher 标签未知或非法使相应评价不完整，不伪造质量零。物理质量、标签 schema 和 v3 观测定义未变。

本次验收范围为 Markdown 链接、示例语法、命令所指文件、各分支差异及共同文件一致性。生产代码、四份参考程序和 A/B 设计 JSON 沿用共同基线。此前129项是已有数学/schema合成验证；本轮没有新增真实 teacher、模型训练、GPU性能或机器人试验结果。两分支后续真实验收分别按其阶段计划记录，仓库 CI 按对应提交查看。


## 2026-11-18 单物体路线v3边界复核与合成参考

[两路线工程规范](instance_routes/README.md)已修订为v3，保留物理v2目标自碰合同与B共享编码结构。新增A中心化高度RMS/前三阶谐波、B实际半径/角槽门控与显式缺测/原始高度摘要；共同法向改为同实例3×3平滑加±3差分，候选加入位反转Morton逐轮格内FPS回填。[v3边界复核](instance_routes/BOUNDARY_REVIEW.md)区分成立风险、已有保护和不能过度推广的结论。Python3.12.14、NumPy2.3.5下基础29项、v2 26项、v3法向/采样26项与接触边界48项检查通过，共129项。细长掩码固定样例1,132个合法像素从46个首轮候选补到256个唯一候选。

[历史合成检查记录](instance_routes/reference/VALIDATION.json)明确为synthetic_only。这不是A/B模型实现、真实数据teacher数值、训练收敛、物理成功率或GPU延迟的验证。现有E5历史记录保持原实验范围。

## 2026-11-18 无内容校验和的运行修复

主线同步删除现行训练、数据、checkpoint、监督器、备份、日志导出、原生扩展及基准/归档脚本中的 SHA-256 计算和验证，加入根 AGENTS 约束与 `tests/test_no_sha256_verification.py`。不以另一内容摘要算法替代；实际配置、模型形状、物理定义、长度、ID、预算、原子写入和运行所有权继续检查。历史结果数值与旧元数据档案保持原记录。[当前读取规则](NO_CONTENT_CHECKSUMS.md)及[旧checkpoint兼容范围](CHECKPOINT_POLICY.md)说明了未知来源与旧target sampler的限制。

主线共用旧运行链路的16组回归为 **268 passed / 1 skipped（11.07s）**；覆盖新旧pretrain恢复、optimizer/RNG连续性、微调、备份、导出、截断与配置冲突。唯一skip需要真实 `R20_RELEASE_ROOT`。永久禁用门禁的 **60项测试通过**；原生扩展用GCC成功重编译，数组对照直接比较实际数值。最终各分支完整CI以提交的Actions结果为准，本地合成检查不等于真实GPU或全量数据训练。

A/B实例模型仍分别位于 [A分支](https://github.com/chichu-lv/suctionnet_modern_20mm/tree/route-a-3d-rgb) 和 [B分支](https://github.com/chichu-lv/suctionnet_modern_20mm/tree/route-b-2d-projected-rings)，该次主线尚未合入实例模型；这是 2026-11-18 的历史状态。


## E5完整测试与公开方法对比（2026-11-18）

E0–E5从头预训练和E5目标微调的完成记录见[实验资料](experiments/CD010-20261117/README.md)。本次汇总冻结的 **E5 best step2880**（checkpoint SHA256 `5bc2c14c014fbf23cbef48062a0dac523ea243b6c18e540bb5af022ff7d0ea3a`）的完整公开测试，使用RealSense、GT实例输入、模型排序分数`q`，覆盖scene100–189的全部256视角，三组各7,680帧，共23,040帧。

| 划分 | 官方R10 AP Top50 | 官方R10 AP Top1 | 项目R20 AP Top50 | 项目R20 AP Top1 |
| --- | ---: | ---: | ---: | ---: |
| Seen | 16.33% | 47.60% | 11.35% | 28.50% |
| Similar | 20.22% | 51.49% | 12.49% | 32.54% |
| Novel | 5.69% | 20.22% | 2.18% | 7.31% |

官方评价使用`suctionnetAPI`固定提交`0dac9f7e5b2f7ea801cddf9655cb82a259681976`；R20复用保存的冻结物理评价值。两套结果共用源预测，但后处理和物理协议不同，不属于纯半径消融。六份precision数组重新聚合与原汇总一致。本轮是已有预测的评价/分析，不包含新训练、GPU吞吐测量或机器人动作。

[完整方法对比](SUCTIONNET_METHOD_COMPARISON.md)给出SuctionNet、Diffusion-Suction、MultiGraspNet、UISN两种完整变体及Only Depth消融的六指标表、来源与输入条件；[紧凑数值](experiments/COMPARISON-20261118/README.md)保留论文报告值与本项目计算值的身份。R20阈值0.4下，保留Top50的候选覆盖率为96.35/95.59/26.30%，首选通过率为36.20/41.08/8.44%，用于区分选择缺口与保留集合覆盖限制，不等同于proposal@512或实机成功率。

目标域step1000的AUROC0.7121来自399条参与选模的历史动作验证；不在本轮改称独立测试或新策略成功率。已完成项与尚待执行的正常业务对比同步见[README Done/TODO](../README.md)和[项目路线](ROADMAP.md)。


## P3 整合：真实配对与正常续训（2026-11-13）

根据 [最新 P3 结果](performance/P3_INTEGRATED_RESULTS.md)，本轮实际源 checkpoint 已确认为 P2 `e73d62e`。**H20/E5 的 64k 与 L20/E3 的 16k 组合已通过真实四帧配对；H20/E5、L20/E0、E1、E3 均已迁移续训并取得首 25 次正常更新窗口。** 方案调整与各阶段证据见 [性能优化全程总览](PERFORMANCE_HISTORY.md)，原 [H20](performance/H20_P2_BATCH_DIAGNOSIS.md) / [L20](performance/L20_P2_BATCH_DIAGNOSIS.md) 诊断保留其历史范围。

- H20 使用 [64k sidecar](../configs/performance/cd009-p3-integrated.json)，L20 使用 [16k sidecar](../configs/performance/cd009-p3-l20-16k.json)。两者保留 P2 grid/坐标快路径，启用 split 与原生接触查询；E0/E1 不使用 contact grouping。voxel heads 保持关闭，B1A4、完整观测、2 mm 体素、512 候选和原训练预算不变。
- 原生正负零字节差异已修复。修复后的 H20/E5 64k 与 L20/E3 16k 配对均为 `passed=true`、0 失败；原 AMP 容差 `atol=2e-4, rtol=2e-3`、离散结果和固定输入几何字节标准未放宽。
- L20 原 16k→64k 组合有 148 项失败，除正负零外还有 scaled 梯度失败。用户选择保留 16k；原 64k 失败报告和此前 E5 零更新短测的吞吐 +28.40% 均保留，不能把短测收益写成当前 16k 组合收益。
- 正常窗口分别为 H20/E5 的 8,331–8,355，L20/E0 的 17,076–17,100、E1 的 2,722–2,746、E3 的 7,602–7,626。每组 25 次尝试均成功、无跳步，100 个物理帧及 100 个有效帧，首个慢步计入。具体墙钟、自然帧差异和持久化范围见 [正常窗口记录](performance/P3_INTEGRATED_RESULTS.md)；日志末步不等于当时已保存的 checkpoint 步。
- 这些窗口的训练代码为 `f4f7f0d6de0bd3bd27107d8be09885fed700e48b`，worktree 为 `0d44b123d9b15d312b64b05aafe992c3e454f7f6c1805eb94e4b070915b2cea1`。后续 GitHub HEAD 或独立 `control_repo` 的 supervisor 身份不替换该训练身份。

[P3 文档](PERFORMANCE_PHASE3.md) 保留正常接续流程；已完成的配对与窗口直接引用上述记录，不重复安排。批准的 original/P1/P2 仍可直接迁到目标 v3，沿用历史回执和学习状态。E2 的既有 FP16 投影溢出未在本轮恢复，六组完整训练、最终评估及收敛结论仍未完成。

### 整合发布前的本地检查（历史）

- [split CPU 子图记录](performance/P3_SPLIT_CPU_RESULTS.json)：2,049 点、9 块的真实 fusion MLP，独立 SliceBackward 9→0、整输入梯度清零 9→0、累加 8→0；仅 split 的 FP32/FP64 输出与全部测量梯度相同。额外一次广播仅为探索原型，未纳入应用代码。
- 普通 CPU E5 集成对照覆盖原生几何、不同 dense chunk、split 和全部参数梯度；checkpoint 分块路径也有现有测试覆盖。该 CPU 结果不代替上述真实 AMP 配对。

整合发布前的本地既有 CPU 套件为 **333 passed、1 skipped（14.94 s）**；因当时本地 Unix socket 限制排除 `tests/test_prefetch.py`，没有改动该测试，完整套件由该提交的 GitHub CI 执行。配对工具保留相同姿态下的几何精确比较，并将不同 chunk 导致的实际浮点描述量按既有 AMP 容差比较；实际离散索引仍须相同，未改变实际输出或梯度。这些历史测试数字不改记为后续正零修复或训练机验证的测试数。

## 原 P3 几何实现：本地 CPU 验证（2026-11-13，整合调整前）

基于 GitHub `0bda78b` 实现融合原生接触查询和两个候选线程，模型及训练定义不变。[完整实现与接续说明](PERFORMANCE_PHASE3.md) 和 [CPU 原始结果](performance/P3_CPU_RESULTS.json) 记录本轮范围。

- 普通两帧场景的五区索引、mask、全部几何描述量、计数、lip 扇区和 Torch RNG 后态与 P2 完全一致；包括倾斜轴和正常无效观测分支。
- CPU 小模型的输出精确一致，参数梯度通过原有 `rtol=1e-6, atol=1e-7`；上述定向测试共 **8 passed、无 skip**。
- 同一 840,000 点、512 候选合成场景，两条路径各预热一次、计时两次；完整 query 中位 **0.901812→0.295082 秒，约 3.056×**。六次调用（含预热）的完整输出和 RNG 一致，未减少邻域或 FPS 输入。
- 新增 P3 配置及两条正常 P2→P3 迁移测试 **3 passed**，覆盖从 P2 开始和带旧迁移链的保存、恢复、预算及归档接续。
- 本地既有套件 **324 passed、1 skipped**；另一个既有 `test_prefetch.py` 因环境禁止 Unix socket 而无法完成，未改写该测试，由原 CI 全量执行。跳过项需要真实 release 路径。

历史测量环境：Python 3.12.14、PyTorch 2.5.1+cpu、NumPy 2.3.5、SciPy 1.17.0、GCC 14.2.0。该轮 CPU 验证没有执行真实训练帧或 GPU optimizer 更新，当时也未部署 P3 到训练机；随后确认的 P2 源 checkpoint、P3 真机配对和部署状态见本文首节。CPU 局部加速比不作为 H20/L20 整步提速或完整训练 ETA。

## CD-009 v3：初始部署记录（2026-11-12，性能优化前）

正式训练按 [固定配置和策略](CD009_PRETRAINING.md) 分阶段运行，六组尚未完成。实际部署源码身份为 `98862f4d6a133c99afa082a0755dbb888b41b38d`；GitHub 发布是包含相同 runtime Python 与新增文档/配置的源码快照，发布 SHA 不替换运行版本，也不声称保留原部署 Git 祖先链。

- 17 个隔离 profile trial，共 216 次帧呈现，零 optimizer 更新；只用于选择运行参数，不能作为收敛证据。
- [六份固定 JSON 和哈希清单](../configs/runtime_profiles/cd009-v3/manifest.json) 对应 H20/E5、L20/E0–E4。当前均为 microbatch1 × accumulation4，按实际吞吐选择；H20/L20 的 chunk 和 allocator cap 不同。
- E5、E3、E0 已有预算内 100 步暂停及恢复下一步的身份/采样器证据；其余实验逐级启用。该检查不是完整 320 帧开发验证，也不构成全程稳定性或训练完成证明。
- 当前没有发布六组完整开发指标、收敛排名、官方测试、真实目标域微调或物理吸取结果。必须每组 230400 有效样本、10 次固定 320 帧验证及 checkpoint/日志/备份验收后才完成 TODO1。

以下 CD-008 的 20 更新 smoke 和 129 项测试是历史记录，保持其原始范围；不将它们改写为 v3 完成结果。

## CD-008 模型实现验证（2026-11-11）

代码已通过只读数据适配、真实完整场景训练链路、checkpoint重启、无标签推理和实际姿态R20评价检查。**这次仅有20次成功更新，不构成收敛、模型优劣或实机成功验证。** 运行方式见 [MODEL_USAGE](MODEL_USAGE.md)，数值与源码哈希见 [机器可读摘要](validation/cd008-smoke-summary.json)。

## 数据与执行范围

- 使用独立SuctionNet-only `suctionnet-r20-v1/full-20261110`，48,640帧/190场景/88物体的发布清单已通过；manifest SHA256为 `37c4336b839b9621fa8317ec6f701846b9e8c073a494c8fa7a66b13cb62ae6a8`。
- 真实更新仅使用场景0、1的帧0、8、16、24，共8帧。开发评价只用场景9帧0、场景19帧8；没有访问官方测试划分进行本轮模型评测。
- 旧标签S/W/C/A/B与冻结配置原样读取，候选行号恢复主轴，观测法向来自相机深度。未修改或重制源数据、物理标签及归一化。
- H20 GPU0，Python3.10.12、torch2.5.1+cu124、spconv-cu120 2.3.6；外部主干固定 `SimonHanrath/graspness_modern@dda1d7586d4a9aea53da1636ffe64ae2c362e861`。
- 每个更新microbatch1累计4，2mm体素、512候选，保留完整观测再应用已定义增强。E5训练每帧约75.4万–88.8万点、12.9万–15.9万体素，没有另设随机点数上限。

## 真实训练与重启

| 检查 | E1 | E5 |
|---|---:|---:|
| 成功optimizer更新 | 2 | 18 |
| 第一次更新总loss | 1.195543 | 1.569269 |
| 最后一次更新总loss | 1.127484 | 1.115736 |
| 成功更新梯度有限且非零 | 全部 | 全部 |
| 主干参数实际变化 | 每步非零 | 每步非零 |
| 正式修复后AMP跳过 | 0 | 0 |

E5独立随机初始化，没有导入E1权重。第9步保存后退出进程，再从同一checkpoint恢复至第18步。恢复后的下一批4帧与已保存采样器完全一致；最终Adam状态步数全部为18；重启前后训练源码哈希一致。CPU合成测试另证明两种目标域训练均满足“连续两步”和“一步＋恢复一步”的逐参数、采样器状态完全相等。

最终E5 checkpoint独立加载两次，对相同开发帧进行无标签推理。16个输出数组全部逐位一致，浮点最大绝对误差0，点/实例/像素身份一致。训练使用的17份核心代码/配置与本次交付逐字节一致；哈希列于摘要。训练代码整体快照、原始日志和checkpoint留在运行工作区，没有上传权重或数据。

训练期间峰值Torch张量显存6,432,461,312字节（约5.99GiB），进程峰值RSS3,058,647,040字节（约2.85GiB）；隔离工作目录约6.2GiB，均低于本轮64GiB显存/64GiB内存/30GiB新增磁盘预算。E5的18次更新累计627.87秒，不包括开发评价和独立推理；这不是部署延迟基准。

单帧E5无optimizer检查还验证了86.6万点、13.3万体素的全量前向/反向，15,525,712个梯度元素非零且全部有限。五区域采样前最大计数为4,718/2,151/5,154/10,644/18,619，证明计数没有被32/64点特征采样上限截断。

## 开发指标及其边界

以下是最后checkpoint在两帧开发样本上的几何代理指标，按视角→场景平均。E1/E5步数不同且都极短，不能把本表当公平消融或泛化结论。

| 指标 | E1 step2 | E5 step18 |
|---|---:|---:|
| AP-R20 | 0 | 0.0155714 |
| instance Top-1 Q20@0.4 | 0 | 0 |
| Top-1碰撞率 | 1.000000 | 0.696970 |
| 无候选实例率 | 0 | 0 |
| 已标注seed的q MAE | 0.324239 | 0.229905 |
| dense q0 MAE | 0.447423 | 0.207186 |
| 主轴误差中位数（°） | 19.6892 | 19.6905 |
| 主轴误差p90（°） | 28.0688 | 28.0716 |

E5两帧分别有11、9个可见实例；两帧Top-1 Q20@0.4均为0。该checkpoint可用来检查接口与复现，不作为已验证的抓取模型或部署权重。

## 评价器与代码检查

独立评价器在场景0/1帧8核对32个原候选，覆盖高/低/零分和碰撞，容差1e-5。最大绝对误差：raw Seal `2.22445e-8`、S `2.54898e-9`、W `0`、Q20 `1.14910e-10`；碰撞标志全部一致。另将8个轴旋转10°，确实按新姿态重新计算Seal。CPU检查20.87秒、383MiB峰值RSS。

最终本地测试：**129 passed，1 skipped**。跳过项需要显式真实release路径，其32候选检查已单独执行并通过。独立检查代理审阅全部新增路径；compileall、diff空白检查、五个CLI帮助和全部发布YAML解析通过。仓库未配置lint或静态类型检查器，因此不把编译检查写成TypeCheck通过。

本轮发现并修复的关键问题：

- NumPy列优先体素坐标进入spconv会错误压缩活动集合。实际首次失败没有成功optimizer更新；在生产组批和主干边界都保证连续内存后，完整场景通过。保留回归测试，不以冻结BN掩盖问题。
- 微调初始验证重复传入日志字段会在step0抛错；完整训练循环测试覆盖此路径。
- 目标训练/验证/无标签推理统一自定义法向预处理和checkpoint深度范围，避免入口间点集变化。

未执行完整40轮、E0–E5正式同预算对比、真实目标域微调、官方测试集评测、实机吸取或部署。可选sidecar转换与residual uncertainty工具仍未交付。

## 合并前独立复核（2026-11-12）

复核的模型代码为 [`57bf0f73634d`](https://github.com/chichu-lv/suctionnet_modern_20mm/commit/57bf0f73634d047afbaf0c5f4adaf01ff1b9bb92)。核对旧 `dc4ad6b00535` producer、实施规格和当前数据适配、模型、损失、训练/微调、推理、评价实现后，没有确认到正常使用路径下需要修复的实现缺陷。本次合并前的后续改动只同步说明文档。

独立复跑环境为Python 3.12、PyTorch 2.5.1+cpu；完整测试结果 **129 passed，1 skipped**。跳过的是需要 `R20_RELEASE_ROOT` 的真实release候选检查。本次CPU复核没有重新执行上文的H20训练或真实数据检查。

对两处具体实现疑点补充了CPU合成对照：

| 检查 | 结果 |
|---|---|
| E3/E5关闭与开启梯度checkpoint，跨多个dense/pose chunk | 输出一致；所有参数最大梯度绝对差为0 |
| `dense_proxy`跨第200步切换：连续运行至202 vs 在200中断后恢复至202 | 最大参数差为0；Adam、source/target采样器与Torch RNG状态相同 |
| `attempt_success`同样的阶段边界恢复 | 最大参数差为0；上述状态相同 |
| 冻结行为 | 完成第200步时decoder未变化；至202步发生更新；encoder始终未变化 |

阶段边界对照使用已有合成权重、Adam和随机状态，将成功步数设置为199，再执行到202；它验证阶段切换与恢复，不代表已完成199步真实目标训练。两种模式均保留同一optimizer，阶段切换没有丢失heads的Adam状态。

当前具备继续按既定方案预训练与消融的代码条件。完整40轮、正式同预算比较、真实目标微调及业务效果仍按 [项目路线](ROADMAP.md) 执行；本次复核没有新增数据制作或额外实验前置要求。
