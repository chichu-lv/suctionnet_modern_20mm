# H20 从头训练：真实批处理、分进程准备和精确样本预算

状态：代码和 CPU 回归已落地；真实 H20 / CUDA / SpConv 性能与收敛仍须按本文压测。没有依据 CPU 测试承诺 GPU 利用率、显存水位或加速倍数。

## 1. 决策与适用范围

用户提供的 2026-11-19 11:02（北京时间）观测为：A/B 各独占一张 H20-3e，GPU 平均利用率约 1.41% / 2.91%，每更新约 23.64 / 9.23 秒，每次仅编码一个实例，累积 16 个有效实例。CPU 供数仍串行。该记录是用户的运行快照，并非本次代码修改后实测。

采用新配方解决三类问题：CPU 读取、增强和几何准备移到持久 spawn worker；模型真正跨实例执行神经网络；密集网络训练允许 BF16。选择新的 run 目录，从随机初始化开始；旧配置、旧 checkpoint 和远端运行不自动切换。

两条路线仍独立：

- A 分支 `route-a-3d-rgb` 使用 `configs/instance_a_h20.yaml`。
- B 分支 `route-b-2d-projected-rings` 使用 `configs/instance_b_h20.yaml`。
- 各自有模型、缓存目录、训练目录、优化器和 best/last。共同代码同步维护，不要求共用同一份缓存，更不让 B 依赖 A 的特征缓存。
- 不重新生成 Teacher 标签；仍使用现有 v3 观测与 `gt_INST20_pose_v2` 标签，保持目标自碰、候选顺序和支持门槛。
- 不计算 SHA-256，也不做首次读取内容校验或换用其他全文件摘要。保留数组形状、有限值、姿态对应及标签语义检查。

## 2. 配置与预算

| 项目 | A 新配方 | B 新配方 |
|---|---|---|
| 有效实例预算 | 140000 | 140000 |
| 每更新有效实例数 | 32 | 32 |
| GPU 微批上限 | 32 | 32 |
| 成功更新次数 | 4375 | 4375 |
| 每实例训练 / 评价候选 | 128 / 最多 256 | 同 A |
| 每实例几何 query chunk | 128 | 128 |
| 跨实例密集 query chunk | 4096 | 4096 |
| 训练准备进程 | 24 | 16 |
| 评价准备进程 | 8 | 8 |
| 每 worker 预取观测数 | 2 | 2 |
| 评价微批 | 32 | 32 |
| 学习率 peak / final | 4.2e-4 / 1.5e-5 | 同 A |
| 预热 | 7008 个有效实例，即 batch32 的 219 次更新 | 同 A |
| 衰减 | 随累计成功实例数余弦衰减 | 同 A |
| AdamW | betas=(0.9,0.999), eps=1e-8, wd=0.01 | 同 A |
| 梯度裁剪 | 总范数 10 | 同 A |
| compile | 默认关闭，显式压测可开 | 同 A |

预热保留原 438×16=7008 个实例的量级，不用四舍五入后的新步数反向改变样本预算。学习率平方根缩放只提供候选起点，并非 AdamW 的普适收敛保证。应同时观察新的 loss、梯度和固定子集指标；旧 E5 的 2880/4800 步不能直接换算为新模型的最佳更新次数。

正常均有监督时，一次更新执行一个 32 实例真微批，随后一次 backward 和 optimizer.step。若样本不足支持、无合法查询或没有已知标签，代码会计数并继续取样，直到本次成功监督预算达到 32；这时可能存在补充的小微批。损失先对每实例的有效 query 求均值，再对成功监督实例等权求均值，不能将全批 query 混成一个加权平均。

新字段 `successful_instance_budget` 是硬预算；无监督样本和未成功更新不计入。其他 batch 仅作为另外命名的实验：

| 有效 batch | 成功更新次数 | 最后一次有效实例数 |
|---|---:|---:|
| 16 | 8750 | 16 |
| 32 | 4375 | 32 |
| 48 | 2917 | 32 |
| 64 | 2188 | 32 |

例如 batch64 不能执行 2188 个完整批而得到 140032 个实例。修改有效 batch 时，必须同步修改 `steps=ceil(140000/batch)`，配置不一致会报错。微批不能超过有效 batch；微批小于有效 batch 时仍明确采用梯度累积。A 的训练 BN 会随微批划分改变，微批大小必须作为该实验配方的一部分记录。

## 3. 精度和真实批处理边界

### A：一次稀疏骨干处理一批物体

`encode_instances` 将各实例体素拼接，稀疏坐标 batch 列为 0..B−1，每个微批只调用一次骨干。最多 2048 个骨干点不等于固定 2048 个独占体素；不能拿 B×2048 当作实际卷积工作量保证。

以下工作在 CPU worker 完成：体素代表点、确定性 FPS、支持点到骨干的关联、接触邻域索引和几何摘要。原生 FPS 扩展沿用仓库 C 实现。主进程保留可微插值、神经点编码、融合和输出头。不同物体的中心、组件、邻域、插值索引和池化边界独立，不能建立跨物体近邻。

几何保持 FP64；稀疏骨干禁用 autocast、保持 FP32；密集 MLP 可在训练时用 BF16，输出及损失回到 FP32。原先单实例骨干的 BatchNorm 也会统计活跃体素，不是“batch=1 就没有统计”；真批现在统计跨物体的活跃体素，因此是新训练行为，不承诺与原积累梯度等价。微调冻结 BN 的规则保留。

### B：合并神经网络计算，几何保持实例隔离

全局 context 一次编码 `[B,13,128,128]`。原生纹理只按相同 padded (H,W) 分组；不把不同 ROI 随意补到整批最大尺寸，否则 GroupNorm 统计会改变。

各实例构造投影环和门控后，合并查询进入 ring encoders、fusion 和 heads。每实例 chunk 控制几何构造，`dense_query_chunk` 控制跨实例神经计算。训练32×128=4096个查询可合并；评价32×256=8192查询默认分成两组密集计算。将旧模型的单实例 chunk 写成4096，本身不能产生真实批处理。

CPU worker 在增强后准备 context、texture、原始几何数组及相位 pose identity。投影、深度读取与可靠性门控维持 FP64；特征插值/grid_sample 保持 FP32；训练卷积、环编码、fusion 和 heads 可 BF16；最终概率、log-scale和损失为FP32。

本次保留 B 既有确定性相位算法。worker 仅缓存 pose bytes；模型消费时使用真实 update，避免预取提前绑定错误训练步。CPU 上的小段 legacy 相位循环尚在，已避免 canonical 路径的 GPU→CPU 回传及逐 chunk 重复；不把此项描述为完全 GPU 向量化。

### 评价与推理

训练器评价固定 FP32，显式关闭 TF32，并恢复每个子模块原有 train/eval 状态。不同微批的 FP32 内核仍可能产生浮点末位差异，不承诺跨 CUDA 版本逐位相同。

加载 checkpoint 推理时验证训练 profile，但不激活其 BF16、compile、worker 或主线程设置。默认 FP32 eager；报告分别记录 checkpoint 的训练执行方式与推理执行方式。批量评价可用显式 CLI 参数启用。

compile 只包装神经网络子模块，既不包动态几何，也不改注册参数键。A 不编译稀疏骨干，B 不编译可变纹理 ROI。默认为 null；可变有效实例数、尾批及训练/评价 dtype 切换都可能导致新图或重编译。PyTorch 官方说明 reduce-overhead 的收益有适用条件且可能增加显存，不能预先声称必然加速。[torch.compile 文档](https://docs.pytorch.org/docs/stable/generated/torch.compile)

## 4. 持久 spawn 供数与可恢复预取

`InstancePreparationPool` 使用有界的持久 spawn 进程池。每个 worker 仅收到小型读取配置，以及当前票据所对应的一行 manifest 与 RGB 增强参数；不向每个 worker 序列化已预热 dataset、全部 manifest 或模型。

单条结果交付前补充下一条任务，使 CPU 准备与 GPU 运算重叠。训练准备池跨更新保留，评价使用独立有限流并在退出时回收。进程意外退出和读失败会传播，不能无限等待丢失结果。

worker 启动前限制 OMP、MKL、OpenBLAS 等线程为1，Torch 线程也为1。新 profile 主进程 Torch intra-op 线程为1。进程数不是CPU核数越多越好；每条结果包含真实数组，过多进程可能增加IPC复制和内存开销。spawn 自身不会保证 dataset 不被复制，关键是当前传入的数据规模受限；fork 也不是立即物理复制全部RSS。PyTorch 对多worker访问大型Python对象的内存风险有明确说明。[数据加载文档](https://docs.pytorch.org/docs/stable/data.html)

训练使用父进程确定的样本票据与增强参数，按发出顺序消费。checkpoint 保存：

- 模型、优化器、成功更新数和有效实例累计数；
- 已发票的 sampler 状态及 RNG；
- 尚未消费的 source/target 票据，含对应RGB参数；
- 目标域独立增强 RNG 和完整 execution 配置。

不把 prepared arrays、队列或进程放入 checkpoint。恢复时先关闭旧池，重放未消费票据，然后继续已超前的 sampler，避免预取造成样本跳过或增强变化。worker在更新中途失败时，不另存半次更新为last；从上一个完整成功checkpoint恢复。

原有配置缺少 `execution` 时继续走旧执行路径。新profile不能直接resume旧profile的checkpoint；新profile内部resume也要求配方、微批、精度、进程与数据定义一致。

## 5. I/O 和 RGB：当前实现与明确保留项

RGB增强仍使用既有算法，但全部下沉CPU worker，与GPU计算并行。一次实例增强保持RGB各输入分支一致。没有分别对已归一化context/texture/support_rgb做不同的GPU颜色变换：contrast均值、HSV量化、裁剪顺序和背景处理会改变原增强分布。若供数优化后增强仍是瓶颈，再以独立实验替换成统一的GPU张量实现；本次不宣称已经完成GPU增强。

新增可选非压缩缓存转换：

```bash
python scripts/repack_instance_cache.py \
  --manifest /data/route_a/train/manifest_train.jsonl \
  --output /data/route_a_uncompressed/train --workers 8

python scripts/repack_instance_cache.py \
  --manifest /data/route_a/selection/manifest_test.jsonl \
  --output /data/route_a_uncompressed/selection --workers 8
```

B使用自己的输入/输出目录。中断可对同一目标加 `--resume`。每份manifest使用独立输出目录。脚本读取既有数组和标签，通过现有语义检查，写 `np.savez`/ZIP_STORED，最后原子发布相对路径清单与覆盖摘要。源缓存只读，不重新跑Teacher，不做内容摘要。

非压缩NPZ避开重复deflate解压，仍有文件读取、数组分配和复制；它不等于mmap或全量数据已在内存。可先使用本地SSD和操作系统页缓存。若选择 `/dev/shm`，先实际检查挂载容量与容器限制，不能把1.8TiB可用主存当作同等大小的内存盘。不要复制全部数据仅为提高RSS或显存占用。

### 源域 CPU 准备与传输

设置 `SUCTION_COMPACT_SOURCE_IPC=1` 可在有准备进程的路线 B 源域训练中启用紧凑传输。有效 batch、微批、准备进程数、抽样次序和 checkpoint 配置沿用当前训练。源域 CPU 增强复用全帧 Pillow 的亮度、对比度和无效像素规则。

启用后，worker 为 context 与纹理实际采样到的像素生成特征。投影使用的四张 FP64 几何或布尔图按有效掩码外接框传输，消费端恢复原始分辨率和像素坐标。保留梯度依赖的网络与每步的相位计算。源域冻结标签仍由原始 NPZ 读取并按原规则验证。

每步的 `timing` 包含等待训练数据、发起源域前向、计算损失及反向与读回、裁剪及优化器与读回的主线程墙钟。CUDA 可异步执行，因此这些数值反映主训练线程的占用和等待。

## 6. 真实数据压测

保留既有GPU软件版本，不为了本功能升级整个环境。A仍需固定外部骨干；在主环境中构建一次FPS扩展，然后再启动workers：

```bash
python setup.py build_ext --inplace
python -c 'from models import _fps_cpu; print(_fps_cpu.__file__)'
python -c 'from models.instance_route_a_geometry import _fps_cpu, _FPS_AXES; assert _fps_cpu is not None and _FPS_AXES is not None; print("native FPS reduction order:", _FPS_AXES)'
```

导入或归约顺序探测失败先解决对应环境；否则现有几何代码可能回落较慢的NumPy FPS。worker不自动并发编译。CPU注入骨干测试不能替代实际SpConv验收。

A分支示例，路径替换为自己的挂载位置：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python scripts/benchmark_instance_training.py \
  --config configs/instance_a_h20.yaml \
  --source-manifest /data/route_a_uncompressed/train/manifest_train.jsonl \
  --selection-manifest /data/route_a_uncompressed/selection/manifest_test.jsonl \
  --backbone-root external/graspness_modern \
  --device cuda:0 --output runs/benchmark_a_h20 \
  --warmup-updates 5 --measure-updates 20
```

B分支使用 `configs/instance_b_h20.yaml`、B的manifest和GPU1，去掉backbone参数：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python scripts/benchmark_instance_training.py \
  --config configs/instance_b_h20.yaml \
  --source-manifest /data/route_b_uncompressed/train/manifest_train.jsonl \
  --selection-manifest /data/route_b_uncompressed/selection/manifest_test.jsonl \
  --device cuda:1 --output runs/benchmark_b_h20 \
  --warmup-updates 5 --measure-updates 20
```

若已设置 `CUDA_VISIBLE_DEVICES=1`，进程内该卡编号是 `cuda:0`；不要又传cuda:1。真实大批、native kernel和compile测试应使用实际数据尺寸。

默认只测配置的一档。需要比较执行效率时用一个新的输出目录追加：

```bash
--microbatches 8,16,32 --worker-counts 0,8,16
```

最多16个组合。effective batch、LR和总预算保持不变；每个组合从相同seed重新构建独立模型和优化器。A的BN使这些微批档不等价，性能档选定后还要评估训练质量。若想比较batch48/64，复制成另命名YAML并同步其预算、步数和LR；不要只改变微批字段。

压测记录预热、正式测量、逐更新秒数、实例/秒、p50/p95、CUDA峰值allocated/reserved、父/子进程RSS、loss、裁剪前梯度范数、实际配置。时间包含GPU完成，计时前后同步CUDA。compile启动及worker启动主要在setup/warmup，动态重编译仍可能进入测量，报告不隐去这些成本。不会加载正式checkpoint、写best/last、做选模或推进正式运行预算。失败档记录错误并回收worker；不把OOM或NaN档排为胜者。

随后选中配方用新目录 `--measure-updates 100` 看短程数值稳定性。不能要求日志中的裁剪前norm一定≤10：超过10后会裁剪；应看非有限值、持续异常大梯度、loss及指标趋势。100步稳定也不等于完整收敛质量已经验证。

成功标准是固定有效batch和统计口径下的实测吞吐与可接受质量。GPU利用率和显存用于定位问题，不要求把显存人为堆到40–90GiB，也不规定batch32必须达到70%利用率。若data_wait高，排查worker/解压/IPC；若等待低但模型仍慢，分析具体GPU算子和CPU发射开销，再决定compile/微批/核函数。不能仅凭“GPU计算占比高”就继续加大batch。

## 7. 正式从头开训

只有实际GPU短测通过的配方才进入正式运行。运行目录应是新的；不要将上述压测作为正式训练的前100步。

A：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python scripts/train_instance.py \
  --config configs/instance_a_h20.yaml --stage pretrain \
  --source-manifest /data/route_a_uncompressed/train/manifest_train.jsonl \
  --selection-manifest /data/route_a_uncompressed/selection/manifest_test.jsonl \
  --backbone-root external/graspness_modern \
  --device cuda:0 --run-dir runs/a_h20_batch32_bf16_fresh
```

B：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python scripts/train_instance.py \
  --config configs/instance_b_h20.yaml --stage pretrain \
  --source-manifest /data/route_b_uncompressed/train/manifest_train.jsonl \
  --selection-manifest /data/route_b_uncompressed/selection/manifest_test.jsonl \
  --device cuda:1 --run-dir runs/b_h20_batch32_bf16_fresh
```

无需重打包时将manifest换回各自原路径。保存代码Git revision、环境和压测报告。若调整新YAML，保存最终的实际配置；checkpoint/run_identity已记录展开后的execution。不要用 `--allow-partial-manifests` 将小样本烟测包装为正式结果。

## 8. 选模和最终评价

仍为100训练场景；90测试场景每场景固定8帧，所选帧全部实例进入指标，支持不足/无法向记no_pose。用户快照中的5954实例是当时数据构建的实际数量，代码不硬编码该数量。每实例仍评价最多256候选。best排序保持 InstTop1_INST20@0.4、selected-Q宏平均、较早步数。

每累计14000个成功实例后的首个optimizer边界选模；batch32节点是：

```text
438, 875, 1313, 1750, 2188, 2625, 3063, 3500, 3938, 4375
```

各节点实际累计样本数写入policy；非整除节点最多跨一个更新边界。总共10次，最终步必评。新profile不写“每438步”而产生4375之后的虚假节点。best/最终步保留gzip明细，其他周期只留汇总；原checkpoint_every作为独立进度保存频率。

训练完成后对固定best运行完整test，不在完整test上重新挑best。批量FP32评价示例：

```bash
python scripts/evaluate_instance.py \
  --checkpoint runs/b_h20_batch32_bf16_fresh/best.pt \
  --manifest /data/route_b/test/manifest_test.jsonl \
  --domain source --split test --device cuda:1 \
  --microbatch-instances 32 --preparation-workers 8 --prefetch-factor 2 \
  --output runs/b_h20_batch32_bf16_fresh/full_test.json
```

A换对应checkpoint/manifest并增加backbone参数。test子集已参与选模，完整test包含这些帧，报告继续标记 `test_selected`、非独立测试估计。公开方法论文值对照与“只参数化吸盘半径”的公开评价仍沿用待定状态，本改动不产生新的公开AP成绩。目标域仍是1676/399/87；新执行层不把目标域单动作监督改成32实例合批协议。

## 9. 实现入口和验收边界

| 文件 | 责任 |
|---|---|
| `models/instance_route_a.py` | A CPU准备、稀疏真批、密集AMP/可选compile |
| `models/instance_route_b.py` | B CPU图像准备、同ROI分组、跨实例环编码及AMP |
| `suction20/instance_execution.py` | 显式执行配置、样本预算、里程碑和精度作用域 |
| `suction20/instance_loading.py` | 有界spawn供数、错误传播与回收 |
| `suction20/instance_training.py` | 实例等权loss、真微批更新、预取checkpoint、批评价 |
| `suction20/instance_inference.py` | 训练profile与默认FP32 eager推理隔离 |
| `scripts/benchmark_instance_training.py` | 一档或小矩阵真实数据压测 |
| `scripts/repack_instance_cache.py` | 可选非压缩NPZ转换 |
| `scripts/evaluate_instance.py` | 显式源域批量评价 |

各分支只含对应模型和测试。共同测试覆盖预算、尾批、loss、选模分母、混合模块状态、预取恢复、进程失败、NPZ转换和不引入内容校验。模型测试覆盖对象隔离、真实调用批大小、梯度、分块与dtype。CPU BF16和编译接口测试不代替CUDA BF16、真实compile内核和SpConv运行。

本次联合工作树执行所有 `tests/test_instance*.py` 与项目内容校验禁用门禁，结果 **456 passed**；其中包含真实spawn、源域/目标域双池的精确恢复和半步worker退出。各独立分支的完整测试结果以对应GitHub Actions为准，联合工作树的测试数量不能冒充单分支数量。

PyTorch参考：[AMP分区](https://docs.pytorch.org/docs/stable/amp.html)、[compile适用条件](https://docs.pytorch.org/docs/stable/generated/torch.compile)、[多进程数据加载](https://docs.pytorch.org/docs/stable/data.html)。本仓库沿用固定PyTorch2.5.1环境，引用说明原理，不据stable文档自动升级安装版本。

