# CD-010 最终选定配方：六组从头训练启动记录

记录日期：2026-11-13（北京时间）。按 [SELECTED_PRETRAINING_RECIPE.md](../SELECTED_PRETRAINING_RECIPE.md) 执行，**新 E0–E5 均已从 step0 独立初始化并产生成功更新**。这是启动记录，完整训练和定期验证尚未完成。

## 1. 实际启动配置与进度

| 实验 | 设备 | CPU逻辑核 | B × A | 有效batch | depth | workers | 训练精度 | 观测成功步数 |
|---|---|---|---|---:|---:|---:|---|---:|
| E0 | L20 GPU1 | 8–15 | 1 × 12 | 12 | 1 | 6 | FP16 | 25 |
| E1 | L20 GPU0 | 0–7 | 1 × 12 | 12 | 1 | 6 | FP16 | 25 |
| E2 | L20 GPU5 | 56–63 | 1 × 12 | 12 | 1 | 4 | BF16 dense / FP32 sparse | 9 |
| E3 | L20 GPU3 | 24–39 | 1 × 12 | 12 | 2 | 4 | FP16 | 20 |
| E4 | L20 GPU4 | 40–55 | 1 × 12 | 12 | 2 | 4 | FP16 | 14 |
| E5 | H20 GPU0 | 0–15 | 4 × 12 | 48 | 2 | 4 | FP16 | 1 |

以上为一次启动后的状态快照，各组phase均为training，读取到的末条更新均updated=true、gradients_finite=true。两机启动器均exit0；成功更新来自新run的updates.jsonl，不以PID创建替代实际开训。各组仅一个新attempt，命令不包含resume、init-checkpoint或短测暂停步数。

共同保持dense chunk65,536、Torch线程8、无人工显存上限。E3/E4/E5使用zone gather、pose chunk128、geometry8；其他组保持原实验结构。模型、AdamW、scaler、sampler及RNG均从seed42重新初始化，旧任务与短测样本不计入新预算。

## 2. 预算、保存与定期验证

| 项目 | H20 / E5 | L20 / E0–E4，各组 |
|---|---:|---:|
| 总有效样本 | 230,400 | 230,400 |
| 成功更新 | 4,800 | 19,200 |
| Warmup成功更新 | 240 | 960 |
| 验证间隔成功更新 | 480 | 1,920 |
| 原定验证节点 | 10 | 10 |
| 常规checkpoint间隔 | 9更新（432有效样本） | 34更新（408有效样本） |

常规checkpoint另有900秒时间触发条件，沿用现有发布及保留规则；不设置第50/100步验收暂停。current/predecessor、best、固定节点及last.pt/best.pt引用由原生checkpoint manifest管理，不额外启动备份daemon。

峰值学习率保持1e-3，warmup起点和cosine末端为1e-5，不随有效batch放大学习率；AdamW weight decay1e-4、betas(0.9,0.999)、eps1e-8、梯度裁剪范数10保留。冻结数据与物理定义不变；两机具有不同参数更新频率，模型指标比较需要注明此训练配方差异。

定期开发验证使用既有固定320帧集合与原协议，结果落盘到validations目录；最佳模型仍按instance_Top1_Q20_at_0_4、再AP_R20选择。当前启动快照尚未到第一个验证节点，不宣称验证通过或模型收敛。

## 3. 停止旧任务与验证兼容修复

切换前旧有效batch8的E0/E1已经在step2880首次定期验证失败并退出；旧E2/E3/E4/E5分别通过原生pause.request正常保存并停止于step1921/2629/2085/1521。旧目录中的日志和checkpoint原地保留，没有额外备份，也不再resume旧任务。

真实旧错误为spconv 2.3.6 implicit GEMM tuner的`can't find suitable algorithm for 0`。代码路径显示，eval分支直接调用ops.implicit_gemm，绕过训练分支的自定义AMP转换；前面的1×1 torch矩阵运算可产生FP16特征，而下一层仍持有FP32权重。

因此在此次冻结源码中加入最小兼容修复：**外层AMP开启且backbone处于eval时，仅稀疏主干关闭autocast并使用FP32；projection和dense头继续原外层精度。** 原FP16训练分支保持不变，E2的BF16 dense/FP32 sparse策略也保持。修复没有改batch、学习率、loss、监督或评估协议。

精度边界CPU回归一次执行：`tests/test_precision.py`，**7 passed**，覆盖BF16训练/评估和FP16评估下的稀疏FP32边界及dense输出精度。没有额外GPU短测、完整验证、旧checkpoint恢复或性能复跑。真正的GPU定期验证仍由新训练按上述预算执行，CPU回归不替代该结果。

## 4. 冻结版本与实际启动入口

- 采用配方：上游`84c0ef3b8d1f9c741f6c2f77c3f57f345432382b`。
- 加入验证兼容修复后的正式源代码：`4a61b8e3f9743f30439f63c98352e9a340f4cfb9`。
- 正式source tree：`1d4cc490b78999a43dca8863ab043b7e9a1b7d42`。
- 两机离线传输root commit：`a96808c475637bbd237bedad32f1ef39772dc07c`，与正式tree相同，提交历史不同。原生CPU扩展在对应机器构建。
- 直接读取已完成的batch-combo目录；selection_policy为current_completed_max，H20选b4_a12_e48_zone128，L20选b1_a12_e12_zone128。没有重复补测。

新部署根目录记作`$TRAIN_ROOT`，冻结checkout位于`$TRAIN_ROOT/repo`；本机补测父目录记作`$BENCH_ROOT`。实际入口如下，资源文件记录各实验原GPU UUID与CPU绑定：

```bash
cd "$TRAIN_ROOT/repo"
PYTHONPATH="$PWD" "$PYTHON" scripts/finalize_pretraining.py \
  --benchmark-dir "$BENCH_ROOT/H20-E5-batch-combo" \
  --training-root "$TRAIN_ROOT" \
  --resource-map "$TRAIN_ROOT/resources.json" --launch
```

L20将benchmark-dir替换为`$BENCH_ROOT/L20-E3-batch-combo`。以上为已执行的一次性启动命令；现有目录不应再次启动。当前为独立后台trainer会话，不启动旧supervisor、协调器或备份服务。

## 5. 日志与模型位置

| 相对新`$TRAIN_ROOT`的路径 | 内容 |
|---|---|
| `selection/HOST.json` | 完整组合及原补测证据 |
| `configs/HOST/E/train.json`、`performance.json` | 实际训练及优化配置 |
| `manifests/HOST.json` | 各组原命令、资源绑定、启动记录 |
| `launch-record.json`、`launcher.log` | 本机旧任务停止及新任务启动记录 |
| `logs/HOST/E.log` | trainer标准输出及错误 |
| `runs/E_seed42/control/status.json` | 当前phase、成功步数、心跳及持久化状态 |
| `runs/E_seed42/attempts/ATTEMPT/updates.jsonl` | 各项loss、学习率、梯度、skip、累计样本、阶段耗时与资源记录 |
| `runs/E_seed42/attempts/ATTEMPT/events.jsonl` | checkpoint与验证等事件 |
| `runs/E_seed42/checkpoint-manifest.json`及checkpoint目录 | 模型、优化器、scaler、sampler、RNG与恢复状态 |
| `runs/E_seed42/last.pt`、`best.pt` | 当前与最佳模型引用；best在验证产生后可用 |
| `runs/E_seed42/validations/` | 原定验证中间结果 |

查看时使用新根目录，例如：

```bash
tail -f "$TRAIN_ROOT/logs/H20/E5.log"
cat "$TRAIN_ROOT/runs/E5_seed42/control/status.json"
```

按已完成补测外推，E5训练计算部分约22.17小时、E3约17.29小时；不含完整验证、常规保存和共享主机波动，也不是当前剩余ETA。其他L20组不套用E3实测速率。启动记录不承诺训练完成或物理吸附效果。
