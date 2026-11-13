# CD-010 六组从头训练运行状态

记录日期：2026-11-13。**最后九档测试已经完成，新的E0–E5正式训练均已从step0启动并产生成功更新。** H20采用B4A2、L20采用B1A8；接触组使用zone gather、pose chunk128，统一有效batch8。当前配置、实际启动进度、日志与模型位置见[最终结果与训练说明](CD010_FINAL_OPTIMIZATION_RESULTS.md)。旧CD-009及早期CD-010任务仍停止，不恢复旧协调器或备份服务。

后续用户要求补充[batch与累积次数组合测试](../BATCH_ACCUMULATION_SUPPLEMENT.md)。新`batch-combo`入口只做独立短测，不自动替换本页正式配置或重启任务；若需复用正在训练的GPU，由执行环境使用原生checkpoint暂停/恢复对应任务。新增十二档尚无实测结果，本页启动快照不是实时监控。

## 已执行的最终方案

已完成`--suite final --run`固定测试，并通过方案提供的独立`finalize_pretraining.py --launch`启动正式训练。先选同机最快已完成原路径，新头至少快3%才采用；两机均达到门槛，选中zone128。3%为工程取舍阈值。

六组正式有效batch均为8：230,400有效样本、28,800次成功更新、warmup1,440次、验证间隔2,880次，共10节点，学习率峰值不变。H20/E5使用B4A2/depth2；L20/E0/E1/E2为B1A8/depth1，E3/E4为B1A8/depth2。E2保持BF16 dense/FP32 sparse，E0/E1 workers6，其余4；E4继承E3所选接触头，不新增E4短测。原生checkpoint、日志与原定验证继续使用，不附加备份daemon或测试后恢复流程。

以下启动配置、前100步吞吐和接续状态均为停止前的历史记录，表中有效batch4与旧CPU/GPU安排不覆盖上述最新决定，也不代表当前仍在训练。

## 1. 停止前的历史状态与优化结论

| 实验 | 分配设备 | 启动接续完成时的状态 |
|---|---|---|
| E0 | L20逻辑GPU1 | 已继续训练，超过step100 |
| E1 | L20逻辑GPU0 | 已继续训练，超过step100 |
| E2 | L20逻辑GPU5 | 已继续训练，超过step100 |
| E3 | L20逻辑GPU3 | 已正常接续至step101 |
| E4 | L20逻辑GPU4 | 已正常接续至step101 |
| E5 | H20逻辑GPU0 | 已正常接续至step101 |

E0–E2此前已有完整100→101证据登记；E3–E5普通接续，不追加完整前缀审计。该状态不证明优化加速，实时进度见第6节。

已有短窗口比较支持：H20/E5几何线程4→8提高11.87%；L20/E3几何线程4→8与CPU affinity8→16组合提高24.63%；L20/E0 workers4→6提高15.52%。H20/E5 B2A2相对B1A4观察到约20%吞吐提升，但改变BN/损失数值行为；修复版相对参照约21.12%，涉及代码版本差异。H20 chunk64k→128k仅提高0.50%，保留64k。

共享体素头、segment广播、FP32几何、静态缓存/去重读取、fused AdamW和按更新读回loss已启用，但这些共同基线优化没有逐项关闭对照，不能给出各自净收益或宣称全部单项有效。E2 BF16 dense/FP32 sparse是数值范围选择，未证明提速。局部compile关闭；CUDA接触查询/FPS及GPU增强未实现。当前不追加测试补齐这些结论。

六组正式运行都从成功step0开始，使用独立模型、优化器、sampler、日志和checkpoint链；不继承30步候选，也不恢复CD-009模型。CD-009的代码、控制账本、模型与日志另行保留，不能把旧进度计入新预算。

## 2. 停止前的六组执行参数（历史）

每个实验独占所分配的单张GPU执行，无跨GPU数据并行。CPU编号为对应机器上的逻辑CPU。

| 实验 | 实验定义 | GPU | CPU池 | microbatch × accumulation | workers | geometry线程 | 精度 |
|---|---|---|---|---|---:|---:|---|
| E0 | gt_A，无实例融合/接触分组/辅助头 | L20/1 | 8–15 | 1×4 | 6 | 4（无接触分组） | FP16 |
| E1 | gt_B，无实例融合/接触分组/辅助头 | L20/0 | 0–7 | 1×4 | 6 | 4（无接触分组） | FP16 |
| E2 | gt_B，实例融合，无接触分组/辅助头 | L20/5 | 56–63 | 1×4 | 4 | 4（无接触分组） | BF16 dense / FP32 sparse |
| E3 | gt_B，接触分组，无实例融合/辅助头 | L20/3 | 24–39 | 1×4 | 4 | 8 | FP16 |
| E4 | gt_B，实例融合+接触分组，无辅助头 | L20/4 | 40–55 | 1×4 | 4 | 8 | FP16 |
| E5 | gt_B，实例融合+接触分组+辅助头 | H20/0 | 0–15 | 2×2 | 4 | 8 | FP16 |

共同设置：dense chunk65,536、`split`输入、Torch8线程、FP32 geometry、compiled FPS、原始点uniform grid、native CPU contact、segment instance broadcast、未启用dense-head compile。E0/E1/E3启用无实例路径的共享体素头；其他三组保留实例路径。数据加载采用spawn、persistent workers、pin memory、prefetch factor2；场景/物体缓存90/88。FP16使用GradScaler；E2的BF16路径关闭GradScaler，loss与参数存储仍为FP32。

E5采用B2A2，保持有效batch4，但改变训练期BatchNorm与损失归约的数值行为，不声称等价于B1A4。E1的workers6来自E0代表组，E4的geometry8/affinity16来自E3代表组；E1/E4现已有正式前缀吞吐观察，但仍未做独立参数对照。E3的参数资源对照在原版97完成，af2正式前缀观察见第8节，不能将其视为补做原参数对照。

## 3. 数据、优化与预算

| 项目 | 六组共同约束 |
|---|---|
| 数据 | 冻结CD-007 SuctionNet，RealSense；manifest SHA256 `37c4336b839b9621fa8317ec6f701846b9e8c073a494c8fa7a66b13cb62ae6a8` |
| 模型输入 | 观测几何、实例掩码、观测法向和重力；不使用RGB；物理半径20 mm、网络体素2 mm、512候选 |
| 随机种子 | 模型seed42，sampling seed20261110 |
| 每组预算 | 230,400有效训练样本，57,600次成功optimizer更新，有效batch4，40个样本预算epoch |
| 样本记账 | 每次成功更新计4个有效样本；skip消耗的物理样本单独记账，不增加成功步数或有效样本数 |
| 优化器 | fused AdamW，betas=(0.9,0.999)，eps=1e-8，weight decay=1e-4；bias/normalization参数不做weight decay；gradient clip norm10 |
| 学习率 | 按成功更新计时：从1e-5线性warmup到1e-3，warmup2,880步；随后cosine到1e-5，目标57,600步 |
| 首100步 | 成功step100对应400有效样本；已有记录保留，六组已继续，不追加前缀审计 |
| 验证 | 每5,760次成功更新，对同一固定320帧验证集评价；共10个节点，最终节点为57,600 |

六组总预算为1,382,400有效样本、345,600次成功更新；各组独立计算，不能相互抵扣。读取等待、验证等待、备份与暂停不改变训练样本预算。短期finite或前缀恢复成功都不是收敛、完整训练或物理吸附成功的证明。

## 4. 训练、控制与发布版本分离

| 用途 | 版本 | 说明 |
|---|---|---|
| 正式训练部署 | `af2fe6f129b48d4ff2eb23875917d6171ce104a2` | 已测loader关闭修复版本；模型与训练源码固定在这一已测版本 |
| 独立控制部署 | `924f51d8fe8c662c2df8d74950f3f1d96ad32507` | 在af2之上增加明确授权的L20 GPU5注册支持；控制代码与训练checkout分开 |
| GitHub发布快照 | `84cae54fdc13dc174d411cfe4beb452178b52526` | 包含上述训练实现、控制改动及公开测试报告；不是新的训练runtime身份 |

发布仓库后续文档或控制提交不会自动覆盖正在运行的训练checkout。实际训练身份以各run的`runtime.json`、attempt identity及checkpoint manifest内的代码/配置摘要为准；不能把GitHub页面的最新HEAD填入旧运行记录。性能schema固定为 **`suction20.performance.v4`**，样本预算合同另为 **`suction20.pretrain.samples.v3`**。

## 5. 日志、模型与配置路径

以下均使用部署根目录变量，不暴露机器账号或内部绝对路径。`$CD010_ROOT`指对应执行机器的CD-010根目录；`$EXPERIMENT`为E0–E5，`$ATTEMPT_ID`取控制器实际分配的attempt，不能按目录修改时间猜测当前attempt。

```bash
RUN_DIR="$CD010_ROOT/runs/${EXPERIMENT}_seed42"
CONTROL_DIR="$CD010_ROOT/control/attempts/$ATTEMPT_ID"
```

| 路径模板 | 内容与用途 |
|---|---|
| `$CD010_ROOT/repo/` | 固定af2训练checkout |
| `$CD010_ROOT/control-repo/` | 独立控制checkout，实际部署路径以控制配置为准 |
| `$RUN_DIR/resolved_config.json` | 实际解析后的训练/模型/runtime配置 |
| `$RUN_DIR/source_config_frozen.json` | 冻结的标签源配置快照 |
| `$RUN_DIR/runtime.json` | 数据/源码/配置身份、依赖、精度和执行设置 |
| `$RUN_DIR/attempts/$ATTEMPT_ID/identity.json` | 本attempt身份、执行设置、父日志链 |
| `$RUN_DIR/attempts/$ATTEMPT_ID/updates.jsonl` | 每次optimizer attempt的loss、grad、LR、尺度、时间、样本与sampler计数 |
| `$RUN_DIR/attempts/$ATTEMPT_ID/events.jsonl` | 启动/恢复、checkpoint、暂停、验证等待/结果、失败事件 |
| `$CONTROL_DIR/status.json` | 当前阶段、heartbeat、成功步数、样本预算与durable checkpoint引用 |
| `$CONTROL_DIR/resources.jsonl` | worker采集的CPU/RAM/GPU资源记录 |
| `$CONTROL_DIR/trainer.stdout.log` | trainer标准输出及错误 |
| `$RUN_DIR/checkpoint-manifest.json` | 唯一checkpoint提交目录，含current/predecessor/best/nodes及摘要 |
| `$RUN_DIR/checkpoints/<step>-<generation>.pt` | manifest引用的实际模型/optimizer/scaler/RNG/sampler状态文件 |
| `$RUN_DIR/last.pt` | 当前已提交checkpoint的兼容别名 |
| `$RUN_DIR/best.pt` | 有有效验证结果后产生的best兼容别名；首个验证节点前不存在属正常 |
| `$RUN_DIR/validations/<step>-<transaction>.json` | 固定验证结果、指标与逐帧记录，带身份及事务绑定 |

原始`train.json`和`performance.json`存放在控制配置的`profiles`与`performance_configs`所指路径，分别保留原字节SHA。恢复必须继续使用这两个原文件；独立评价才使用run中的`resolved_config.json`。**唯一完整全局控制账本位于H20的`$CD010_ROOT/control/coordinator.json`**；L20不保存完整全局ledger。在H20从`assignments`核对实验所在机器与当前token，再到对应机器查看该attempt日志；历史attempt日志持续保留，不按目录mtime猜当前token。

## 6. 只读查看与导出

后台worker与协调器以独立会话（setsid）运行，通过日志和状态文件查看，无需`tmux attach`。以下命令不启动、暂停或恢复训练。

先在**H20**读取唯一全局账本；此处`$CD010_ROOT`必须是H20上的部署根目录：

```bash
python - "$CD010_ROOT/control/coordinator.json" <<'PY'
import json, sys
ledger = json.load(open(sys.argv[1]))
print("experiment host gpu token step state observed_unix")
for ex in ("E0", "E1", "E2", "E3", "E4", "E5"):
    a = ledger.get("assignments", {}).get(ex, {})
    status = (a.get("view") or {}).get("status") or {}
    same_attempt = bool(a.get("token")) and status.get("attempt_id") == a.get("token")
    print(ex, a.get("host"), a.get("gpu_index"), a.get("token"),
          status.get("global_step") if same_attempt else None,
          a.get("state"), a.get("observed_unix") if same_attempt else None)
PY
```

这里的step是协调器最近一次观察值，可能滞后；缺少匹配当前token的状态时显示`None`，不回填旧attempt的步数。按打印的host到对应机器，填写`$EXPERIMENT`和`$ATTEMPT_ID`，并将`$CD010_ROOT`设为该机器的根目录，再计算第5节的`RUN_DIR`与`CONTROL_DIR`。以当地`status.json`的heartbeat核对新鲜度；恢复会换token，需重新查账本。

在对应执行机器使用已验证的Python环境查看：

```bash
# 查看阶段、成功进度和预算，避免只看最后一行stdout。
python - "$CONTROL_DIR/status.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
keys = ("phase", "global_step", "effective_samples_completed", "target_step",
        "target_effective_samples", "consumed_samples", "skipped_samples", "heartbeat_unix")
print(json.dumps({key: s.get(key) for key in keys}, indent=2, ensure_ascii=False))
PY

# 当前attempt的最新更新与事件是live信息，尚不一定已checkpoint提交。
tail -n 3 "$RUN_DIR/attempts/$ATTEMPT_ID/updates.jsonl"
tail -n 10 "$RUN_DIR/attempts/$ATTEMPT_ID/events.jsonl"
tail -n 3 "$CONTROL_DIR/resources.jsonl"

# 不加载大Torch文件，查看已提交模型引用及保留节点。
python - "$RUN_DIR/checkpoint-manifest.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
for role in ("current", "predecessor", "best"):
    ref = m.get(role)
    print(role, None if ref is None else {k: ref.get(k) for k in ("global_step", "path", "sha256")})
print("retained_nodes", sorted(map(int, m.get("nodes", {}))))
PY
```

需要结构化表格时，使用固定训练checkout中的生产exporter，将结果写入新的外部导出目录：

```bash
PYTHONPATH="$CD010_ROOT/repo" python "$CD010_ROOT/repo/scripts/export_training_logs.py" \
  --run-dir "$RUN_DIR" --output-dir "$NEW_EXPORT_DIR" \
  --resource-log "$CONTROL_DIR/resources.jsonl"
```

可重复传入`--resource-log`纳入历史attempt的资源记录。exporter校验所选checkpoint文件SHA及提交日志前缀，输出`updates.csv`、`frames.csv`、`events.csv`、`validation_frames.csv`、`resources.csv`、`summary.json`和`export-manifest.json`；未提交/放弃的尾部单独分类。`--checkpoint-role`可选current/predecessor/best，默认current。`--live-attempt-id`只给尾部加标签，不证明进程存活。不要将多个attempt的raw尾部直接拼接计入正式成功步数。

## 7. 验证、保存与恢复规则

常规checkpoint在距上次保存达到100次成功更新，或达到900秒时触发；验证前、验证后及正常暂停也按原流程保存。manifest是提交点，别名可能晚于manifest更新，审计以manifest引用为准。

保留当前current、前一predecessor、验证选优best和`nodes`固定证据节点（包括正式prefix100与验证节点）。其他不再被引用的checkpoint blob由正常retention回收；并非所有100步周期文件永久保留。checkpoint包括模型、AdamW、scaler、scheduler、sampler、RNG、身份、样本计数与提交日志链，不只是权重。

每5,760步的320帧验证完成后记录结果并保存节点；先比较`instance_Top1_Q20_at_0_4`，相等时再比较`AP_R20`，较优才更新best。验证等待和验证执行在events中区分；验证前后恢复训练RNG。首个验证节点前没有best模型或正式验证指标，不是缺失训练结果。没有新增官方测试集评价，也不能把训练loss代替验证结论。

共享备份按现有控制器授予的备份流程执行，成功checkpoint与日志前缀核验后保留；备份完成与模型更新成功是不同状态，分别记录。候选测试的已完成备份详见性能报告，不能代替这六组新正式运行自己的备份证据。

CD-010普通接续使用同一run当前已提交checkpoint、原始配置和sidecar。后续沿用既定日志、模型保存和验证周期，不为追加检查、评审、对照或复跑暂停。旧CD-009不再恢复或新增备份。

## 8. 正式首100步的实际吞吐与纯训练时间

六组各自首个正式attempt均记录**100次成功更新、0 skip，loss与gradient全部有限**。以下统计使用同一attempt内**成功step10结束至成功step100结束**的90次更新，即360个有效样本；吞吐为`360 / 窗口wall秒`。不混入30步候选或恢复后的attempt，也不把前100步有限性等同于完整暂停/备份/恢复验收。

| 实验 | 设备 | 有效帧/s | 230,400样本纯训练外推（小时） | CUDA allocated峰值（GiB） | CUDA reserved峰值（GiB） |
|---|---|---:|---:|---:|---:|
| E0 | L20 | 4.889423 | 13.09 | 5.71 | 10.00 |
| E1 | L20 | 4.875733 | 13.13 | 5.71 | 10.00 |
| E2 | L20 | 2.372649 | 26.97 | 11.74 | 25.30 |
| E3 | L20 | 2.115520 | 30.25 | 5.86 | 15.17 |
| E4 | L20 | 1.684633 | 37.99 | 11.19 | 25.60 |
| E5 | H20 | 1.535042 | 41.69 | 21.65 | 59.71 |

显存值取11–100窗口内各次更新所记录峰值的最大值，以字节除以`2^30`换算；reserved是PyTorch分配器保留量，不等同于当前有效tensor大小或整张GPU的设备级占用。原摘要分别保留L20五组与H20/E5的源日志摘要、窗口边界和健康计数。

这些是各组在实际共享机器负载下、不同独立时段的描述性观察，**不是参数优化对照**。实验模型、精度、batch分组和并发负载均可能不同，不能用表中差异归因于GPU硬件，或直接推导各消融模块的独立速度成本。E1/E4现在有各自正式前缀实测值，但仍没有独立参数选择对照。

小时数按`230400 / 帧每秒 / 3600`外推整个预算，不是剩余时间或交付ETA。窗口保留两个边界之间实际发生的日志与checkpoint间隙，但不含启动、前10步预热，以及step100结束后的保存、备份、暂停/恢复；这次前缀也没有正式验证。外推未另加入后续验证、周期保存/备份、暂停、维护、长尾帧或吞吐波动开销，不能据此承诺完成时刻。
