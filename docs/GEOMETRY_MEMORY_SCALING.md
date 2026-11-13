# 几何重叠后的显存扩展：一次完成 H20 / L20 多档 batch 短测

日期：2026-11-13（北京时间）。基于 `c684aae` 的[几何重叠实测](performance/CD010_GEOMETRY_OVERLAP_RESULTS.md)。

**实测更新（2026-11-13）：** 原接触头默认12档已全部尝试，9档完成、3档OOM；H20/E5最快B4A6/depth2为2.259727帧/s，L20/E3保留B1A12/depth2的3.038967帧/s，详见[完整扫描报告](performance/CD010_GEOMETRY_MEMORY_RESULTS.md)。随后[9档zone/pose比较](performance/CD010_FINAL_OPTIMIZATION_RESULTS.md)已完成，并以有效batch8启动六组新正式训练。用户后续要求补齐[batch/累积组合](BATCH_ACCUMULATION_SUPPLEMENT.md)，新`batch-combo`每机六档，固定zone128并明确包含B4A6、B1A12；不是重复本页原chunk32矩阵。以下保留原扫描设计和运行入口。

**本轮取消人为显存上限，一次运行预先列出的多档 batch，再统一比较吞吐与显存。** H20 只使用 E5，L20 只使用 E3；默认分别 7 档和 5 档，不需要每测一档就人工回传、再生成下一档。入口是 `benchmark_geometry_overlap.py --suite batch-scan`。

多档batch的GPU实测已经完成，结果和失败档位以上方报告为准。扫描实现与CPU/CI检查属于实现记录，不替代硬件结果。当时完整训练与协调器仍停止，原扫描没有启动六组训练；后续启动状态以上方最新报告为准。

## 1. 已测结果与继续增大 batch 的依据

| 已测设置 | 有效帧/s | 相对各自串行基线 | allocated / reserved 峰值 GiB | future 等待中位数 |
|---|---:|---:|---:|---:|
| H20/E5，B2A2，depth2 | 1.842032 | +19.90% | 45.28 / 83.11 | 0.2645 s |
| L20/E3，B1A4，depth2 | 2.841787 | +33.82% | 11.72 / 22.94 | 0.0739 s |

这里 B 是一次 GPU 前向的 microbatch，A 是每次参数更新的梯度累积次数，`B × A` 为有效 batch。`depth2` 允许同一次更新内两份 microbatch 的图驻留，使 CPU 几何工作与 GPU 计算重叠；`depth1` 是串行方案。

CPU/GPU 重叠已经带来净收益，因此值得重新测量更大的 GPU 批次。H20 当前活跃分配峰值约45 GiB、L20约12 GiB，都还有扩大工作集的可能。future等待是主线程等待CPU结果的区间，不是精确GPU空闲时间；不能把所有未来收益都归因于继续消除CPU等待。

容量估算只用于选择候选：按目前活跃分配粗略线性推算，H20 B4/depth2 可能接近90 GiB，B6/depth2接近135 GiB；L20 B3/depth2可能接近35 GiB。真实峰值还受稀疏卷积工作区、反向暂存和分配器影响，所以这些不是实测值，也不保证能放下。**不能将 reserved83.11 GiB 直接乘以 batch 倍数来判断 OOM。**

## 2. 取消的是分配器上限，不要求提前占满显存

新生成的从头训练配置默认使用：

```json
{
  "runtime": {
    "gpu_memory_limit_gib": null
  }
}
```

`null` 表示不施加项目的进程显存配额；运行时不会再按原先 H20 112 GiB、L20 36 GiB 的值限制分配器。可使用该 GPU 实际可用的显存，由真实工作集按需申请，不预先制造无用张量占满显存。

CLI 也支持显式 `--no-gpu-memory-limit`。如果后续需要有限额运行，仍可传 `--gpu-memory-limit-gib 128` 等具体值；两者互斥。已有历史配置里的有限数值不被批量改写，新的扫描配置会明确记录自己的无限额设置。

每个候选在独立训练进程中运行，结束后进程释放其 CUDA 分配，再启动下一档。不是在一个进程中不断增大 batch，也不会让前一档的缓存或失败状态累积到下一档。

## 3. 默认矩阵：H20 七档、L20 五档

### H20 / E5：共同有效 batch 24

| 设置 | Microbatch | 累积次数 | 流水线深度 | 理论最多同时驻留帧数 | 用途 |
|---|---:|---:|---:|---:|---|
| B1A24 / depth2 | 1 | 24 | 2 | 2 | 小批次对照 |
| B2A12 / depth2 | 2 | 12 | 2 | 4 | 对应已测 microbatch2 的新同条件对照 |
| B3A8 / depth2 | 3 | 8 | 2 | 6 | 填补2与4之间的档位 |
| B4A6 / depth2 | 4 | 6 | 2 | 8 | 较大批次 |
| B6A4 / depth2 | 6 | 4 | 2 | 12 | 接近整卡容量的候选 |
| B8A3 / depth2 | 8 | 3 | 2 | 16 | 探测更大批次及真实容量边界 |
| B8A3 / depth1 | 8 | 3 | 1 | 8 | 用单份大批次图比较容量与重叠的取舍 |

### L20 / E3：共同有效 batch 12

| 设置 | Microbatch | 累积次数 | 流水线深度 | 理论最多同时驻留帧数 | 用途 |
|---|---:|---:|---:|---:|---|
| B1A12 / depth2 | 1 | 12 | 2 | 2 | 小批次对照 |
| B2A6 / depth2 | 2 | 6 | 2 | 4 | 较大批次 |
| B3A4 / depth2 | 3 | 4 | 2 | 6 | 接近整卡容量的候选 |
| B4A3 / depth2 | 4 | 3 | 2 | 8 | 探测更大批次及真实容量边界 |
| B4A3 / depth1 | 4 | 3 | 1 | 4 | 用单份大批次图比较容量与重叠的取舍 |

加入3和6这些中间档，是为了避免只测2、4、8时直接从显存富余跨到OOM。最后一档depth1也是有目的的候选：较大microbatch可能无法同时保留两份图，但能保留一份；需要实测GPU批次收益是否足以抵消失去几何重叠的代价。

这里没有增加更深的流水线。现有实现只支持depth1/2；队列内仍有两份图时，增大microbatch已经能提高驻留工作量。也不同时扫描pose chunk、dense chunk、CPU线程或worker数。共同固定dense chunk65,536、pose chunk32、geometry8、affinity16、Torch8、workers4，以及原FP16/FP32 geometry和适用的模型优化。

## 4. 为什么每台机器固定有效 batch

H20的microbatch1、2、3、4、6、8都整除24，并且至少保留两份microbatch；L20的1、2、3、4都整除12，也满足这个条件。这样每台机器内部所有候选都具有相同的optimizer更新频率，不会将“每样本更新次数减少”误认为“更大的GPU批次更快”。

**24和12是这次容量/吞吐扫描的共同条件，不是已经选定的正式训练设置。** 历史H20 B2A2/有效4的1.842032帧/s、L20 B1A4/有效4的2.841787帧/s只作历史参考；本轮排名以同机、同有效batch的新对照为准。microbatch改变也会改变BN分组和损失的数值行为，短测用于性能选择，不用于判断收敛效果。

配置生成器从原v3样本预算派生更新步数，仍保留230,400个总有效样本、11,520个warmup样本和23,040个样本的验证间隔：

| 预算 | 历史有效batch4 | H20扫描有效batch24 | L20扫描有效batch12 |
|---|---:|---:|---:|
| 总成功更新数 | 57,600 | 9,600 | 19,200 |
| warmup更新数 | 2,880 | 480 | 960 |
| 验证间隔更新数 | 5,760 | 960 | 1,920 |

学习率峰值、Adam参数、weight decay、clip和loss不自动缩放。扫描会提前停止，不运行上述完整预算或完整验证；也不会修改其他实验的最终配置。

## 5. 每档48帧预热、240帧计时

所有默认候选各处理 **48个有效样本预热 + 240个有效样本计时 = 288个有效样本**，执行真实forward、loss、backward与AdamW。

| 主机 | 成功步计时边界 | 预热更新数 | 计时更新数 | 计时有效样本 |
|---|---|---:|---:|---:|
| H20，有效batch24 | step2结束 → step12结束 | 2 | 10 | 240 |
| L20，有效batch12 | step4结束 → step24结束 | 4 | 20 | 240 |

H20两次预热更新实际已经包含48帧；在最大microbatch8时也有六次microbatch前反向，并完成optimizer状态初始化。比较统一用帧/s，不能直接比较不同有效batch的秒/更新。窗口包含AMP跳步、数据等待和日志间隙，排除启动与末尾保存/退出。288样本也低于400样本的定期checkpoint节点。

若所有候选都完成，H20共处理2,016个有效样本，L20共1,440个；其中进入计时的分别为1,680和1,200。发生OOM的档位会提前结束，不伪造其完整窗口或吞吐。

## 6. 一次运行，自动记录OOM并继续

`--run`按计划顺序执行全部候选；每档都有独立目录、配置、console日志、训练日志和退出记录。识别到CUDA显存不足时：

1. 将该档标为OOM，保存原始报错与已有记录。
2. 让失败进程退出，释放该进程的GPU资源。
3. 继续计划中的下一档，包括最后的depth1大批次候选。

不会因某个depth2档OOM就跳过后续depth1，也不会修改上限或偷偷缩小batch重试。非OOM错误仍停止扫描并保留已完成汇总，因为路径、数据或实现错误不能当作容量结果。

最终`summary.json`/`summary.md`汇总所有候选，包括完成、OOM及未执行项。吞吐排名仅使用完整有效窗口且正常退出的候选；失败档位的部分日志或报错数字不能当作完整峰值。汇总保留batch、累积次数、有效batch、depth、帧/s、更新median/p95、allocated/reserved、几何future等待、loader等待和成功/skip。

## 7. 运行命令

使用包含本次修改的新代码checkout和已有训练Python环境；按[构建入口](FROM_SCRATCH_PERFORMANCE.md#61-构建与生成)构建CPU扩展。为每台机器设置其实际分配的GPU和CPU，脚本继承这些选择。

```bash
export PYTHONPATH="$PWD"
export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

taskset --cpu-list "$TRAIN_CPUS" python scripts/benchmark_geometry_overlap.py \
  --suite batch-scan --hardware H20 \
  --no-gpu-memory-limit \
  --out-dir "$BENCH_ROOT/H20-E5-batch-scan" \
  --source-root "$DATA_ROOT/input_raw" \
  --label-root "$DATA_ROOT/output" \
  --backbone-root "$BACKBONE_ROOT" \
  --run
```

L20改为`--hardware L20`与独立的`L20-E3-batch-scan`目录，一次运行其五个设置。两台机器分别启动各自命令即可，无需在每档之间人工确认。省略`--run`只生成计划，不启动GPU进程。

其他入口：

- `--batch-sizes 2 3 4 6 8 --effective-batch-size 24`：自定义扫描的microbatch列表与共同有效batch；microbatch必须整除有效batch，depth2需要累积次数至少2。默认会再加最大microbatch的depth1候选。
- `--overlap-only`：只保留depth2候选，去掉最后的depth1容量比较。
- `--summarize-existing`：读取同一输出目录的计划与日志重新汇总，不启动训练。
- `--suite overlap`仍是原串行/重叠两项比较，`--suite memory`仍是上一轮两档比较。它们的原始40+80样本计时含义保留；本轮使用新的`batch-scan`即可。

## 8. 看什么结果，如何做最后选择

**优先选择帧/s最高且能完成的设置，而不是显存占用最高的设置。** 同时对照三类结果：

- 大microbatch/depth2更快：增加GPU工作量与几何重叠共同有效。
- 大microbatch/depth2 OOM，但最后的depth1更快：更大的单份图有价值，保留两份图的容量代价需要取舍。
- 显存明显增加，吞吐不变或下降：容量不是当前速度限制，继续检查GPU接触头/反向阶段，而不是再扩batch。

几何future等待和loader等待可用于判断等待是否重新暴露；它们不能代替CUDA kernel级结论。如果最终需要优化接触头，已有`pose_chunk_size=128`是下一项具体候选，但本轮不增加pose/chunk交叉扫描。也不追加历史checkpoint/AMP/RNG一致性审计、第31步恢复、六组消融或完整训练。
