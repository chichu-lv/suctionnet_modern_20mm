# 最终采用的预训练配置：H20 / L20 从头训练

日期：2026-11-13（北京时间）。依据已完成的[十二档 batch / 累积补测](performance/CD010_BATCH_ACCUMULATION_RESULTS.md)及用户最新指令，**正式采用 H20 的 B4A12、L20 的 B1A12，六组模型全部从 step0 重新初始化训练，不续接已有任务。** 性能选型到此结束，不再追加测试矩阵。

这里确认的是新的正式配方与启动方式。结果提交 `e367df8` 记录的旧部署仍为有效 batch8；本次代码环境没有 H20/L20 执行通道，配置发布不表示新六组已在机器上启动。实际启动及进度以新根目录中的 trainer 日志为准。

## 1. 采用配置与收益

| 主机 / 直接实测模型 | 正式组合 | 有效 batch | 本轮吞吐 | 相对原配置吞吐 | 同样样本量的训练部分时间 |
|---|---|---:|---:|---:|---:|
| **H20 / E5** | **B4A12、zone128、depth2** | **48** | **2.887176 帧/s** | **+17.60%** | **减少14.97%** |
| **L20 / E3** | **B1A12、zone128、depth2** | **12** | **3.700548 帧/s** | **+4.54%** | **减少4.34%** |

B 表示 microbatch，A 表示梯度累积次数。基线分别是同轮 H20 B4A2 的2.454991帧/s、L20 B1A8的3.539799帧/s。数字来自[公开机器可读结果](performance/CD010_BATCH_ACCUMULATION_RESULTS.json)，时间降幅按相同样本数除以帧/s换算，不能与吞吐涨幅直接等同。

### H20：采用 B4A12，不再限制有效 batch 为8

B4A12比B4A6的2.750321帧/s还快4.98%，比有效batch24组中最快的B3A8快3.86%。本轮直接选择已经完成测量的最高吞吐组合，不为统一两台机器的有效batch排除它。

固定B4增加A时，allocated峰值从约88.65变为89.08 GiB，没有随累积次数成倍增长；depth2仍最多保留两份在途计算图。更长窗口可摊薄流水线填充、收尾及更新边界开销。已有host阶段记录支持这一解释，但不把重叠区间相加，也不把收益全部归于AdamW更新。[阶段与显存记录](performance/CD010_BATCH_ACCUMULATION_RESULTS.md#5-已有阶段耗时与资源观测)

### L20：采用 B1A12，不继续增大A或B

B1A12为本轮最高值。B1A24为3.683097帧/s、B1A48为3.667929帧/s，与A12接近；这些微小差异不解释为统计显著性。采用A12保留了当前最高观测吞吐，也比A24/A48具有更频繁的参数更新。B2/B3没有在本轮取得更高端到端吞吐，不再补做worker或kernel诊断。

## 2. 六组正式配置

L20的B1A12作为本机正式配方用于E0–E4；只有E3具有本轮直接吞吐证据。E4继承同机接触执行设置，E0/E1/E2保持各自无接触模型和精度选择，不将E3的4.54%写成所有L20模型的实测收益。

| 实验 | GPU | CPU逻辑核 | Microbatch × 累积 | 有效 batch | depth | workers | 接触几何线程 | 精度 |
|---|---|---|---|---:|---:|---:|---:|---|
| E0 | L20 GPU1 | 8–15 | **1 × 12** | **12** | 1 | 6 | 不适用 | FP16 |
| E1 | L20 GPU0 | 0–7 | **1 × 12** | **12** | 1 | 6 | 不适用 | FP16 |
| E2 | L20 GPU5 | 56–63 | **1 × 12** | **12** | 1 | 4 | 不适用 | BF16 dense / FP32 sparse |
| E3 | L20 GPU3 | 24–39 | **1 × 12** | **12** | 2 | 4 | 8 | FP16 |
| E4 | L20 GPU4 | 40–55 | **1 × 12** | **12** | 2 | 4 | 8 | FP16 |
| E5 | H20 GPU0 | 0–15 | **4 × 12** | **48** | 2 | 4 | 8 | FP16 |

以上资源分配沿用[最新部署记录](performance/CD010_FINAL_OPTIMIZATION_RESULTS.md#4-最终代码和六组配置)。实际GPU/CPU分配若变化，通过启动入口的`--resource-map`指定对应实验的`gpu`和`cpus`，可用GPU UUID，不需要改模型配置。

共同执行设置：

- **Dense chunk65,536、Torch线程8，无人工显存上限。**
- E3/E4/E5使用**按区域一次gather、pose chunk128、FP32原生几何、depth2**。
- 保留已部署的compiled FPS、native CPU query、split、缓存/预取、适用的共享体素头和分段实例广播；局部compile保持关闭。
- 保留E0–E5各自的模型、loss、监督定义；E5仍含原辅助头，E2继续使用已选BF16 dense / FP32 sparse。
- 原冻结数据、20mm半径、2mm网络体素、512候选保持；每组seed42独立初始化。

配置由[现有生成器](../suction20/from_scratch.py)和[正式启动器](../suction20/final_pretraining.py)生成，数据与backbone路径取自本机已完成补测的获选候选配置。

## 3. 从头训练的样本预算与学习率

**每组重新完成230,400个有效训练样本。** 旧任务或短测已经处理的样本不抵扣新任务预算；新运行不加载它们的模型、优化器、scaler、sampler或RNG状态。

| 项目 | H20 / E5 | L20 / E0–E4，各组 |
|---|---:|---:|
| 有效 batch | **48** | **12** |
| 总有效样本 | 230,400 | 230,400 |
| 成功参数更新 | **4,800** | **19,200** |
| Warmup有效样本 | 11,520 | 11,520 |
| Warmup成功更新 | **240** | **960** |
| 验证间隔有效样本 | 23,040 | 23,040 |
| 验证间隔成功更新 | **480** | **1,920** |
| 原定验证节点 | **10** | **10** |

保留原AdamW设置：峰值学习率`1e-3`、weight decay `1e-4`、betas `(0.9, 0.999)`、eps `1e-8`、梯度裁剪范数10；bias及normalization参数仍不做weight decay。学习率从`1e-5`线性warmup到峰值，再按既有cosine调度降到`1e-5`，使用上表换算的成功更新节点。**不因有效batch增大自动放大学习率。**[原profile](../configs/runtime_profiles/cd009-v3/H20-E5.json) · [预算换算](../suction20/pretrain_budget.py)

六组总有效样本仍为1,382,400，正常checkpoint和原定开发验证照常执行；不增加开训前完整验证或性能验收。

后续模型比较应写明：各模型按硬件选定配方，在相同总样本预算下完成预训练。E5的有效batch48与E0–E4的12具有不同优化轨迹，不能把模型间所有指标差异都归因于结构模块；这项说明不要求再增加统一batch的实验。

## 4. 停止旧六组，使用新目录重新启动

用户已经明确选择从头训练。部署顺序为：**让旧任务正常退出并释放指定GPU，准备新的冻结checkout和运行根目录，再使用已有补测结果直接生成并启动新六组。** 不运行续训转换，不恢复旧任务，不重做十二档短测。

### 4.1 旧任务正常退出

在各机设置`$OLD_TRAINING_ROOT`为当前有效batch8任务的根目录。使用既有原生暂停请求让旧trainer在正常边界保存后退出；这里只把它作为停止旧进程的方式，后续不再resume这些任务。

H20：

```bash
touch "$OLD_TRAINING_ROOT/runs/E5_seed42/control/pause.request"
```

L20：

```bash
for experiment in E0 E1 E2 E3 E4; do
  touch "$OLD_TRAINING_ROOT/runs/${experiment}_seed42/control/pause.request"
done
```

对应trainer退出、GPU释放后进入新启动步骤。旧日志和checkpoint保留在原目录即可，不要求额外备份或恢复审计。[原生请求接口](../suction20/pretrain_runtime.py) · [正常退出实现](../scripts/train_pretrain.py)

### 4.2 使用现有补测结果直接启动

在包含本次正式选择功能的新checkout根目录、使用现有依赖环境执行。设置：

- `$BENCH_ROOT`：本机已经完成的`batch-combo`补测根目录，内部保留原`plan.json`、`summary.json`与候选`train.json`。
- `$NEW_TRAINING_ROOT`：本轮新的正式输出根目录，不能指向旧任务根目录。

这里读取完整本机补测目录；公开结果JSON用于记录，不替代带实际数据路径的本机plan及候选配置。

**H20：选择B4A12，启动新的E5。**

```bash
PYTHONPATH="$PWD" python scripts/finalize_pretraining.py \
  --benchmark-dir "$BENCH_ROOT/H20-E5-batch-combo" \
  --training-root "$NEW_TRAINING_ROOT" \
  --launch
```

**L20：选择B1A12，启动新的E0–E4。**

```bash
PYTHONPATH="$PWD" python scripts/finalize_pretraining.py \
  --benchmark-dir "$BENCH_ROOT/L20-E3-batch-combo" \
  --training-root "$NEW_TRAINING_ROOT" \
  --launch
```

每台机器执行一次即可。启动器为各实验单独设置GPU/CPU绑定，在所有配置保存后调用原`train_pretrain.py`创建后台进程；不传`--resume`、`--init-checkpoint`或短测的暂停/结束步数。

如需覆盖资源，将`--resource-map /absolute/path/resources.json`加入对应命令。文件只填写本机需要覆盖的实验，例如：

```json
{
  "E5": {"gpu": "GPU-实际分配的UUID", "cpus": "0-15"}
}
```

`finalize_pretraining.py`现在接受已完成的`batch-combo`结果，在本机正常完成的候选中直接选择最高帧/s，记录`selection_policy="current_completed_max"`，然后使用获选有效batch派生完整预算。旧`final`套件的有效batch8规则仅保留兼容历史；本次输入必须是上面的最新补测目录。

## 5. 新训练产物与运行状态

| 相对`$NEW_TRAINING_ROOT`的路径 | 内容 |
|---|---|
| `selection/H20.json`、`selection/L20.json` | 本机采用的完整组合、基线和吞吐依据 |
| `configs/HOST/E/train.json` | 新正式训练配置，包含对应batch与样本预算 |
| `configs/HOST/E/performance.json` | 实际执行优化设置 |
| `manifests/HOST.json` | 各实验命令、GPU/CPU、配置与启动记录 |
| `logs/HOST/E.log` | 正式trainer标准输出和错误日志 |
| `runs/E_seed42/` | 新模型、优化器、checkpoint、逐次更新与原定验证记录 |

`HOST`为H20或L20，`E`为实验编号。使用新根目录使本次从头训练与旧任务有独立记录；旧任务的进度和模型不会混入新预算。

启动记录中的PID / `process_spawned`表示进程已经创建；真正开始训练以新`runs/E_seed42/control/status.json`和`attempts/<attempt>/updates.jsonl`中出现成功更新为准。无需设置第50/100步的额外验收暂停。

## 6. 训练时间量级

按本轮短测吞吐、从零处理230,400个有效样本计算：

| 直接实测模型 | 原有效batch8配方 | 本次采用配方 | 训练部分节省 |
|---|---:|---:|---:|
| H20 / E5 | 约26.07小时 | **约22.17小时** | **约3.90小时** |
| L20 / E3 | 约18.08小时 | **约17.29小时** | **约47分钟** |

这是短测速度外推的训练部分量级，不是当前剩余ETA，不包含完整验证、常规保存、停机及后期阶段变化；也不能将E3的时间套用到其他L20模型。实际总时长随新训练日志更新即可，不因此再安排性能测试。
