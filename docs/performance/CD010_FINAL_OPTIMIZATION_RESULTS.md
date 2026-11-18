# CD-010 最后一轮优化结果与正式训练启动

日期：2026-11-13（北京时间）。按 [FINAL_OPTIMIZATION_AND_TRAINING.md](../FINAL_OPTIMIZATION_AND_TRAINING.md) 完成最后一次固定矩阵。

**九档均完成。H20选B4A2、zone gather、pose chunk128，2.450753帧/s，比最快原路径提高17.79%；L20选B1A8、zone gather、pose chunk128，3.622360帧/s，提高29.40%。六组新正式训练均已从step0启动并产生成功更新。** 本次性能优化结束，不再追加优化轮次。

## 1. 本轮范围与选择规则

H20/E5五档，L20/E3四档；全部有效batch8、depth2，每档48有效帧预热、240有效帧计时。计时为成功step6结束→step36结束，共30次成功更新；每档独立初始化seed42，只执行一次。全部计时窗口0 skip、日志中的梯度有限，各候选按既定36步终点退出，无OOM或其他失败。

先在同机原路径中选最快基线，再选最快新头；新头端到端帧/s至少提高3%才采用。两机均满足。3%是预先规定的工程阈值，不是置信区间。未追加trace、allocator/CPU/worker扫描、旧checkpoint对齐或恢复验收。

共同固定dense chunk65,536、geometry8、Torch8、workers4、16逻辑CPU、FP16、FP32 geometry、fused AdamW、无限额显存；局部compile关闭。测试设备为H20 GPU0（CPU0–15）、L20 GPU3（CPU24–39）。硬件及依赖沿用同日记录，见[环境信息](CD010_FROM_SCRATCH_RESULTS.md#2-实验边界与环境)。完整原始点、20mm半径、2mm体素、512候选及原数据/监督定义保持。

## 2. 九档实测

涨幅均相对同机**最快原路径基线**，H20为B4A2/chunk32，L20为B1A8/chunk32。窗口wall包含中间读取、同步和日志间隙，排除启动、预热与最终保存/退出。

| 主机 | 候选 | 帧/s | 相对最快基线 | 窗口wall（秒） | 更新中位数 / p95（秒） | allocated / reserved峰值（GiB） |
|---|---|---:|---:|---:|---:|---:|
| H20 | b2_chunk32 | 1.981948 | -4.74% | 121.093 | 4.062 / 4.145 | 45.66 / 99.93 |
| H20 | b4_chunk32 | 2.080659 | 0.00% | 115.348 | 3.848 / 3.976 | 88.65 / 138.80 |
| H20 | b4_zone32 | 2.101516 | 1.00% | 114.203 | 3.798 / 3.898 | 88.65 / 138.80 |
| H20 | b4_zone64 | 2.312511 | 11.14% | 103.783 | 3.446 / 3.556 | 88.65 / 138.80 |
| H20 | b4_zone128 | 2.450753 | 17.79% | 97.929 | 3.245 / 3.366 | 88.65 / 138.65 |
| L20 | b1_chunk32 | 2.799343 | 0.00% | 85.734 | 2.703 / 3.318 | 11.56 / 25.54 |
| L20 | b1_zone32 | 2.890393 | 3.25% | 83.034 | 2.661 / 3.381 | 11.56 / 25.54 |
| L20 | b1_zone64 | 3.454373 | 23.40% | 69.477 | 2.281 / 2.499 | 11.56 / 25.54 |
| L20 | b1_zone128 | 3.622360 | 29.40% | 66.255 | 2.198 / 2.348 | 11.55 / 25.58 |

H20最终有效batch8下，B4原路径比B2原路径快4.98%。单独改zone32，H20仅提升1.00%，未到门槛；L20提升3.25%。继续将zone路径的pose chunk增至64/128后收益扩大，两机最终均选128。**本轮支持的是zone与更大pose chunk的组合收益，不能把17.79%/29.40%全部归因于gather次数减少。** 没有测试原chunk模式下的pose128，也不补做该项。

选中路径的allocated峰值与同batch基线接近：H20约88.65 GiB，L20约11.55 GiB。reserved含缓存，不能当成当前活跃tensor大小或未来固定显存占用。上一轮有效batch24/12的结果只作历史参考，没有作为这轮选型基线。

## 3. 前向、反向与等待

以下为每次更新的中位数/p95，两机本轮均处理8个有效样本/更新。host区间包含同步影响，不是孤立CUDA kernel计时；后台几何与GPU工作可重叠，不能把它们相加当作更新耗时。

| 主机/候选 | forward host（秒） | backward host（秒） | future等待（秒） | loader等待（毫秒） |
|---|---:|---:|---:|---:|
| H20/b2_chunk32 | 1.557 / 1.650 | 1.951 / 2.008 | 0.250 / 0.307 | 1.679 / 1.955 |
| H20/b4_chunk32 | 1.747 / 1.882 | 1.608 / 1.648 | 0.251 / 0.320 | 0.723 / 1.044 |
| H20/b4_zone32 | 1.717 / 1.828 | 1.599 / 1.636 | 0.244 / 0.287 | 0.738 / 0.976 |
| H20/b4_zone64 | 1.443 / 1.635 | 1.545 / 1.575 | 0.228 / 0.309 | 0.704 / 1.330 |
| H20/b4_zone128 | 1.338 / 1.524 | 1.426 / 1.484 | 0.243 / 0.299 | 0.750 / 1.077 |
| L20/b1_chunk32 | 1.313 / 1.500 | 1.081 / 1.194 | 0.071 / 0.294 | 2.053 / 2.623 |
| L20/b1_zone32 | 1.303 / 1.466 | 1.091 / 1.512 | 0.089 / 0.240 | 2.394 / 2.904 |
| L20/b1_zone64 | 0.920 / 1.003 | 0.999 / 1.155 | 0.098 / 0.194 | 2.150 / 2.643 |
| L20/b1_zone128 | 0.832 / 0.894 | 0.996 / 1.033 | 0.149 / 0.313 | 2.342 / 4.405 |

H20最快原路径→zone128，forward host中位数1.747→1.338秒，backward host1.608→1.426秒，future等待基本相当。L20对应forward host1.313→0.832秒、backward host1.081→0.996秒；future等待反而0.071→0.149秒，但总吞吐仍提升29.40%。这说明阶段等待不能单独代替端到端选型。没有新增kernel级诊断，也不再以进一步优化阻塞训练。

## 4. 最终代码和六组配置

- 测试与正式模型源码均对应上游 `bc599ffb243b7b036916973add391ec4a3fc50ae`，tree `e8db7cd72eb652bef7b46c62a1b9f58e7e62428b`。
- 两机实际部署的运输快照commit为 `8d8047dc98a292ae74467525efca55d8b302d8d7`，文件tree与上游完全一致，Git历史不同。测试后未修改模型实现；后续报告提交不会自动改变运行源码。
- 使用新的正式训练根目录，未继承短测模型、优化器、sampler或旧CD-009/CD-010 checkpoint。六组均seed42独立初始化。

| 实验 | GPU | CPU逻辑核 | microbatch×累积 | depth | 接触头 / pose chunk | workers | 精度 |
|---|---|---|---|---:|---|---:|---|
| E0 | L20/1 | 8–15 | 1×8 | 1 | 无接触分组 | 6 | FP16 |
| E1 | L20/0 | 0–7 | 1×8 | 1 | 无接触分组 | 6 | FP16 |
| E2 | L20/5 | 56–63 | 1×8 | 1 | 无接触分组 | 4 | BF16 dense / FP32 sparse |
| E3 | L20/3 | 24–39 | 1×8 | 2 | zone / 128 | 4 | FP16 |
| E4 | L20/4 | 40–55 | 1×8 | 2 | zone / 128 | 4 | FP16 |
| E5 | H20/0 | 0–15 | 4×2 | 2 | zone / 128 | 4 | FP16 |

所有正式组Torch8、dense64k，无人工显存配额；接触组geometry8。E0/E1/E3共享体素头，E2/E4/E5采用实例路径；缓存、split、原生几何、fused AdamW及既有优化随固定版本部署。E4沿用E3选择，没有新增E4性能测试，也不声称获得同样百分比收益。

每组230,400有效样本、28,800次成功更新、名义40轮；warmup1,440次，每2,880次成功更新运行一个原定开发验证节点，共10次。学习率峰值和优化器超参数不自动放大。六组总有效样本1,382,400。实际microbatch按设备不同，不能宣称BatchNorm数值相同。

## 5. 已观察到的正式训练进度

以下是完成本轮记录时的状态快照，非持续更新看板。各组均已产生真实成功更新，不仅是创建PID。

| 实验 | 阶段 | 已观察成功步数 |
|---|---|---:|
| E0 | training | 282 |
| E1 | training | 281 |
| E2 | training | 134 |
| E3 | training | 213 |
| E4 | training | 152 |
| E5 | training | 2 |

本轮直接启动trainer独立后台会话，未重启旧协调器或备份服务。没有人为的第50/100步验收暂停。启动前确认所分配GPU未被其他计算进程使用并绑定设备UUID；旧目录保留原状，不恢复旧任务。以上进度尚未到首个2,880步验证节点，不宣称已有正式验证指标或收敛结论。

## 6. 配置、日志、模型与查看方法

`$FINAL_ROOT`表示对应机器的新正式训练根目录，`$BENCH_ROOT`表示本轮短测输出根目录；`$HOST`取H20/L20，`$EXPERIMENT`取对应E0–E5。

| 路径 | 内容 |
|---|---|
| `$BENCH_ROOT/H20-E5-final/`、`$BENCH_ROOT/L20-E3-final/` | 原始plan、summary、selection和逐候选配置/console/退出记录 |
| `$FINAL_ROOT/selection/$HOST.json` | 基线、新头、3%阈值及最终选择 |
| `$FINAL_ROOT/manifests/$HOST.json` | 每组实际命令、环境、GPU/CPU绑定、配置路径与启动记录 |
| `$FINAL_ROOT/configs/$HOST/$EXPERIMENT/{train,performance}.json` | 正式配置与执行设置 |
| `$FINAL_ROOT/logs/$HOST/$EXPERIMENT.log` | 标准输出/错误，可tail -f |
| `$FINAL_ROOT/runs/${EXPERIMENT}_seed42/control/status.json` | 当前attempt、阶段、heartbeat、成功步数与样本预算 |
| `runs/${EXPERIMENT}_seed42/attempts/<attempt>/updates.jsonl` | 每次更新loss、梯度、学习率、AMP尺度、耗时、显存峰值、样本与frame统计 |
| `runs/${EXPERIMENT}_seed42/attempts/<attempt>/events.jsonl` | 初始化、保存、验证、失败等事件 |
| `runs/${EXPERIMENT}_seed42/{runtime,resolved_config}.json` | 实际源码/数据/依赖身份和解析后配置 |
| `runs/${EXPERIMENT}_seed42/checkpoint-manifest.json` | 已提交current/predecessor/best/nodes模型引用 |
| `runs/${EXPERIMENT}_seed42/checkpoints/` | 模型、AdamW、scaler、RNG、sampler及预算状态 |
| `runs/${EXPERIMENT}_seed42/validations/` | 原定验证节点的指标与逐帧记录 |

表中以 `runs/` 开头的路径均相对 `$FINAL_ROOT`。常规checkpoint每50次成功更新（400个有效样本），或距上次保存达到900秒触发；验证前后也按原流程保存。保留current、predecessor、验证选优best及固定nodes；并非所有周期checkpoint永久保留。manifest为提交依据，`last.pt`/`best.pt`为兼容别名。

每节点固定320帧开发验证；先比较 `instance_Top1_Q20_at_0_4`，相等时比较 `AP_R20`，更优才更新best。首次保存前manifest可能尚未生成，首个验证节点前没有best属正常。没有新增官方测试集评价或物理吸附实验。

查看示例（在实际执行机器上）：

```bash
export HOST=L20 EXPERIMENT=E3
RUN_DIR="$FINAL_ROOT/runs/${EXPERIMENT}_seed42"
tail -f "$FINAL_ROOT/logs/$HOST/$EXPERIMENT.log"
# 独立终端查看状态：
cat "$RUN_DIR/control/status.json"
# 当前attempt ID从status.json读取，再查看细粒度记录：
tail -n 3 "$RUN_DIR/attempts/$ATTEMPT_ID/updates.jsonl"
tail -n 10 "$RUN_DIR/attempts/$ATTEMPT_ID/events.jsonl"
```

正式启动采用方案提供的独立 `finalize_pretraining.py --launch`，输入本轮已完成测试目录和新的训练根目录；不是再次运行短测。精确已执行命令保存在每机manifest中，六组现已运行，不应重复执行启动命令。
