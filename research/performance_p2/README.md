# P2 CPU 空间索引研究复现

这是独立研究目录，未接入训练或推理产品路径。原生 CSR 网格只改变空间索引，cell size 固定为 0.02m，保留全部 840000 个原始点、精确半径判断、`radius+1e-8` 和升序 raw id；它不是点云降采样。

## 运行

在已安装本仓库依赖与生产 compiled FPS 扩展的环境，从**仓库根目录**运行：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m research.performance_p2.benchmark_grid \
  --output-dir /tmp/suction20-p2-grid --repeats 2
```

依赖为现有 `numpy`、`torch`、`scipy` 与可用的 C++17 编译器。脚本直接调用 `make_fps_backend('compiled')`，复用仓库已构建的 FPS，不编译另一份 FPS。若环境还未安装生产扩展，可先按仓库既有安装方式运行 `python -m pip install -e '.[model]'`。

`threadpoolctl` 是可选项：安装时将 BLAS 限为 1 线程；缺失时仍可运行，使用上方环境变量约束线程，并在结果中注明没有由 runner 强制控制 BLAS。未新增仓库依赖。

`--output-dir` 必填。临时 `grid_index.so` 和新测量 `geometry_grid_results.json` 全部写入该目录，不写源码目录，不提交动态库。`--cxx` 可覆盖编译器（默认环境变量 `CXX` 或 `c++`）；`--repeats` 仅接受 1 或 2，固定同一份场景，不开展参数矩阵。

```bash
python -m research.performance_p2.benchmark_grid --help
```

参考源码 HEAD 为 `1e28a2faa9991478d706e4d188081aa1a482ff9b`。脚本记录当前 HEAD，不因不同 HEAD 阻塞运行；未来源码变化时按记录解释测量。

## 文件与测量内容

| 文件 | 用途 |
|---|---|
| `fixture.py` | 固定的 840000 点、512 候选、15 物体区域合成输入 |
| `grid_index.cpp` | 单线程 CSR 网格构建和精确球查询原型 |
| `benchmark_grid.py` | KDTree 与原生网格的建索引、2048 球查询及完整 grouping 对照 |
| `results/geometry_grid_results.json` | 初版临时原型的两次网格对照结果，原样保留 |
| `results/geometry_parallel_results.json` | 初版临时原型的候选线程对照结果，原样保留 |
| `voxel_heads_equivalence.py` | 无实例 fusion/q0 与可选 axis 首层分解的 CPU float64 代数/梯度对照 |
| `results/voxel_heads_equivalence.json` | 已完成的单次代数原型数值，未验证正式损失或 CUDA/AMP |

当前复现脚本**直接调用生产** `ContactGrouping.query(..., random_roll=True)`，不复制或动态改写函数源码。每个对照路径前恢复同一个 Torch RNG 状态，随后比较 RNG after-state；同时比较所有球内 raw ids、五区 `indices/mask/geometry/counts` 和 lip sector counts。后续 NumPy 局部数学、区域筛选及 compiled FPS 都调用现有实现。

原型仅支持这里的单帧输入；所有查询串行。C++ handle 内部复用 scratch，调用方立即复制结果，不应将同一个 handle 并发用于多个查询线程。生产多帧/并发接口不是本研究脚本的实现范围。

## 已保存结果的来源与限制

`results/` 内两个 JSON 来自最初的临时调用适配器，未由整理后的脚本重新测量。初版为共享同一份显式 roll，临时读取 `ContactGrouping.query` 并只替换调用边界：把内部 roll 生成改为传入预生成数组，把实例计时字段写入改为返回 stats；**没有改动局部几何数学、物理边界、查询顺序或 FPS**。线程原型还按 64 候选分块并合并输出。

整理后的 grid runner 改为生产公开调用，并通过恢复 RNG state 实现同样的 roll 对照。因而原 JSON 中 `rng_unchanged` 指初版适配器不抽取 RNG；新 runner 的 `rng_after_matches_reference` 指两条生产路径消耗相同 RNG。两个字段不应混为“生产 query 不消耗 RNG”。

原始数据为本地合成 CPU 实测，无 CUDA、无 optimizer update，也没有真实 SuctionNet 帧：

| 原型 | 两次中位结果 |
|---|---|
| KDTree 建索引 + 完整 query | 1.7809 秒 |
| 原生网格建索引 + 完整 query | 0.9671 秒，几何段约 1.84× |
| 当前 KDTree query，1 候选线程 | 1.5212 秒 |
| 当前 KDTree query，2 候选线程 | 1.2767 秒，query 约 1.19× |
| 当前 KDTree query，4 候选线程 | 1.2757 秒，无进一步收益 |

这些数字支持后续实现的优先级，不是 H20/L20 训练加速承诺。独立 2048 球查询为了逐邻域比较暂存全部返回数组；完整 query 按候选释放列表，二者计时口径不同。并行原型各 phase 时长为 worker 工作时长之和，不能相加当作 wall time。

本次整理只执行导入、语法和参数检查，没有重新运行 840000 点基准，也没有增加压力/边界测试。

## voxel heads 代数原型

```bash
python -m research.performance_p2.voxel_heads_equivalence \
  --output-file /tmp/suction20-p2-voxel-heads.json
```

这个原型使用 2117 个点、113 个 voxel、两帧和逐点合成加权平方误差，同时比较无实例 fusion/q0 复用与可选 axis 首层 affine 分解。它不执行正式 `source_losses`、轴残差几何、真实 backbone 或 CUDA/AMP，也不测吞吐。已保存数值支持数学等价和参数形状不变的设计判断；整理后仅作语法/参数检查，未再次运行该对照。
