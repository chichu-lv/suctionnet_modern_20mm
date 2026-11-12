# CD-009：正在运行的 v3 预训练

2026-11-12。E0–E5 六组独立预训练正在分阶段执行，**尚未完成六组训练、完整开发验证或收敛比较**。E5/H20、E3/L20 和随后启动的 E0/L20 已有正式 100 步暂停、检查和恢复后的下一步证据；其余实验按资源与稳定性门槛逐级启动。这里发布的是实际代码、固定配置和策略，不是已完成训练的权重或质量报告。

## 固定版本与可复现配置

性能优化前已部署的代码固定为 `98862f4d6a133c99afa082a0755dbb888b41b38d`，运行源码 SHA256 为 `d634627b180cee10be2c42949c8d10cec21ac49424aa734f6b5acf2c5f2c30c8`。该版本包含 JSON 数字读取修复：`.json` 使用 JSON parser，避免 `1e-08` 被 YAML 当作字符串。YAML 配置继续按 YAML 规则读取。

H20/E5、L20/E0 与 E3 已切换到性能版本；代码部署和 GitHub 发布的对应关系如下。这里不表示其余 E1/E2/E4 已完成启动或训练。

| 身份 | 已核验值 |
| --- | --- |
| 冻结部署 commit | `996421c664a5a0b84c308aa4ca47c31eaff4035b` |
| 部署源码 SHA256 | `23bde51eb6b8f091a15b2aeb336db8a7085b150f304076af44b3a1d83ba7ee20` |
| GitHub 同树源码发布 | [`f6cb0102ebe4566bff023b251525171f15265fa8`](https://github.com/chichu-lv/suctionnet_modern_20mm/commit/f6cb0102ebe4566bff023b251525171f15265fa8) |
| 发布 CI | [43186974094：通过](https://github.com/chichu-lv/suctionnet_modern_20mm/actions/runs/43186974094) |
| P1 性能 sidecar SHA256 | `08d2247c5bd433d8fdc16325a9602821a8a5c8c52b0c6cf0148a4991e4f9d188` |

部署 commit 与该 GitHub commit 的 Git tree 已核对一致，但发布有独立 SHA，不承诺部署提交的 Git 祖先链。**本次后续文档发布不部署到训练 checkout，也不改变正在运行的身份。** 普通恢复仍要求相同代码与性能身份；旧 `98862f4` 仅通过显式迁移切换。current checkpoint、receipt 和新 attempt 保存实际 commit/源码摘要/sidecar/本机扩展身份，不能直接用后续文档 HEAD 替代。

**2026-11-12 后续提速工作：** `1e28a2f` 发布了 P1 后的 batch/CPU 对照，继续选择 B1A4、CPU8、loader2。[P2 方案](PERFORMANCE_PHASE2.md) 进一步给出原生 CPU 全点网格、E0/E1/E3 voxel fusion/q0、公共坐标快路径与 P1→P2 迁移的实施规格，附 [CPU 原型证据](performance/P2_CPU_PROTOTYPES.json)。这些 P2 产品改动尚未实现或部署；本页的冻结部署身份保持不变。现有迁移工具仅支持原始代码→P1，P1→P2 需按新方案扩展已有入口后使用。

[六份正式配置及可移植哈希清单](../configs/runtime_profiles/cd009-v3/manifest.json) 保留部署配置的原始字节。清单的 `config_path` 相对仓库根目录，`config_sha256` 是文件原始字节 SHA256；每组还记录测量文件名、SHA256、trial ID 和测量代码版本。原始测量日志、训练日志、权重、数据及私有控制配置不放入 Git。配置中的数据路径是占位符；运行时由 supervisor 参数覆盖，无需改写已绑定哈希的配置。

## 硬件选择：按实际吞吐

共完成 **17 个独立 profile trial，216 次物理帧呈现（72 warmup、144 measured），零 optimizer 更新**；按 trial ID 去重计数。每 trial 一个 warmup、两个 measured block，使用独立 seed42 模型和实际训练帧。BN 状态只在丢弃的测量进程内变化；未加载正式 checkpoint。测量代码为 `bd8f65ffe4b577a5dc2a1c7d8623e924a595ac48`；后续训练版本未改变模型和损失算法。

| 正式配置 | 实测帧/秒 | microbatch × accumulation | dense checkpoint / chunk | allocator 上限 |
| --- | ---: | --- | --- | --- |
| [H20 / E5](../configs/runtime_profiles/cd009-v3/H20-E5.json) | 0.11115 | 1 × 4 | off / 65536 | 112 GiB |
| [L20 / E0](../configs/runtime_profiles/cd009-v3/L20-E0.json) | 0.36217 | 1 × 4 | off / 16384 | 36 GiB |
| [L20 / E1](../configs/runtime_profiles/cd009-v3/L20-E1.json) | 0.36236 | 1 × 4 | off / 16384 | 36 GiB |
| [L20 / E2](../configs/runtime_profiles/cd009-v3/L20-E2.json) | 0.33250 | 1 × 4 | off / 16384 | 36 GiB |
| [L20 / E3](../configs/runtime_profiles/cd009-v3/L20-E3.json) | 0.10905 | 1 × 4 | off / 16384 | 36 GiB |
| [L20 / E4](../configs/runtime_profiles/cd009-v3/L20-E4.json) | 0.10638 | 1 × 4 | off / 16384 | 36 GiB |

全部使用 8 Torch CPU 线程、独立且不重叠的 CPU affinity。相同 microbatch 是测量结果，不是为公平而强制统一：L20 的 microbatch2 比相应 microbatch1 慢；H20 E5 的 B2-on、B4-off、B8-off 分别约 0.10651、0.10562、0.09989 帧/秒。B8 占用更多显存仍未提速。E3/E4/E5 记录的接触区域查询区间约 6 秒/帧，是主要耗时区间；其中包含 tensor 转 CPU 及可能等待此前 GPU 工作完成的同步，不能解释为纯 CPU 查询成本或 CPU 利用率。E0/E1 无该模块，其读取、组批和反向耗时更突出。

这些是同步 CUDA 边界的短测量，会抑制重叠；不包括 Adam 更新、保存、完整验证和排队。H20 选中组合同时改变 checkpoint 与 chunk，不能分别归因。B8 测量使用更多帧；并发/独占及跨硬件条件也不同。结果不能作为架构公平成本排名或正式 ETA。选中 H20 profile 的 Torch reserved 峰值约 14.56 GiB、NVML 采样峰值约 15.08 GiB；L20 约 13.73–13.93 / 14.15–14.36 GiB。显存上限约为硬件可用容量的 80%，是 allocator ceiling，不是占用目标或整个 CUDA 进程的硬限制。复杂帧、Adam 状态、验证和保存仍须持续观察。

CPU 合成测试检查 checkpoint 开关的输出、梯度和 RNG；不承诺真实 CUDA/AMP、不同 batch 或 chunk 之间逐位等价。BN、随机操作交错与有效监督帧平均均可能随 batch 分区变化。

## 数据、样本预算与学习策略

只读使用 SuctionNet-only RealSense `full-20261110`：48,640 帧、190 场景、88 物体；manifest SHA256 `37c4336b839b9621fa8317ec6f701846b9e8c073a494c8fa7a66b13cb62ae6a8`。网络输入为观测几何、实例 mask、观测法向和重力，不用 RGB 特征。20 mm 吸盘半径、2 mm 体素、512 候选以及已有冻结 S/W/C/A/B 标签保持不变，不删减完整观测以适配显存。

训练使用官方训练场景 0–99 中编号末位非 9 的 90 场景；每名义轮每场景抽 64 个视角，共 5760 帧。每组 seed42 独立初始化，sampling_seed20261110 共享确定性帧/增强序列；不串联 E0–E5 权重。旧 v2 E5/E3 各 9 次更新的 checkpoint、日志和备份保留，未计入新 v3 预算。

`training.budget_contract = suction20.pretrain.samples.v3`：每组 **230400 有效训练帧，名义 40 轮**，预热 11520 帧，每 23040 帧完整验证一次。有效 batch E=M×A，预算边界必须整除 E；当前 E=4 得到 57600 成功更新、2880 预热更新、5760 验证间隔。若未来另选 E，必须生成新版本而非在恢复时静默改变配置。只有成功 optimizer 更新推进预算、LR 和候选采样阶段；AMP 跳步及故障回放的物理计算单列。

AdamW：峰值 LR 0.001、betas=(0.9,0.999)、eps=1e-8、weight decay=1e-4，bias/normalization 不衰减；梯度裁剪阈值 10。LR 从 1e-5 线性预热至峰值，再余弦下降至 1e-5，不因 batch 自动线性放大。AMP 开启、BN 正常训练；损失按 accumulation 缩放。候选暖启动前 11520 有效帧使用高/低 GT-A 各半，之后 proposal/high/low 为 1/2、1/4、1/4，按实例平衡且不重复采样。

E0 质量目标为 GT-A；E1–E5 为 GT-B。E2 加实例融合，E3 加接触分区，E4 同时使用二者，E5 再加辅助头。`q0/q` 使用 masked SmoothL1（beta=0.05），轴损失为 `1−dot`；E5 seal/wrench 为 SmoothL1，collision-free 为 BCE-with-logits。权重 q0/q/normal/seal/wrench/collision-free 为 1/1/2/0.5/0.25/0.1。有效元素先在实例内平均，再平均有监督实例和帧；空监督不是负样本。保持现有 microbatch reduction，未增加跨 batch 的重新归一化。详细公式见 [模型规格](implementation/MODEL.md)。

## 启动、日志、恢复与验证

从 [公共 supervisor 模板](implementation/examples/cd009-supervisor.json) 配置部署；私有地址、密钥和控制目录只留本地。模板只登记已测量的六个 host/experiment 组合。填写本地路径、机器身份与 GPU UUID，使用清单中的精确配置 SHA256；命令、环境验证、分阶段启动和恢复见 [运行指南](MODEL_USAGE.md#cd-009-persistent-multi-host-pretraining)。不要把 pending 的 `pretrain_e*.yaml` 当作已选正式 profile 直接启动。

```bash
PY=/path/to/runtime/venv/bin/python
SUP=/path/to/frozen/repo/scripts/supervise_pretrain.py
CFG=/path/to/private/supervisor.json
"$PY" "$SUP" --config "$CFG" probe --host H20
"$PY" "$SUP" --config "$CFG" probe --host L20
"$PY" "$SUP" --config "$CFG" plan
"$PY" "$SUP" --config "$CFG" status
# 完成运行指南中的真实 host evidence 后，通过持久进程运行 serve。
```

每次 attempt 有独立身份与追加式 update/event/resource 日志。记录帧和增强 seed、点/体素/候选数、监督覆盖、loss/LR/梯度、AMP scale/跳步、有效/物理样本、分段时间及显存/CPU/RSS/I/O/PSI。CUDA 峰值按 accumulation attempt 重置；CPU host timings 不伪称 GPU kernel 时间。日志导出核验 checkpoint 与已提交日志字节前缀的 SHA256，区分 committed、live/uncommitted 和 abandoned，不把崩溃尾部计入训练进度。

checkpoint 在完整更新边界每 100 成功步或 900 秒保存，并覆盖初始化、暂停、验证前后和最终状态；manifest 是提交点，使用 fsync、checksum、原子发布，保留前代、best、前缀和验证节点。恢复 model/Adam/AMP/RNG/sampler，严格核对数据、代码、预算及 profile。首个 100 步/400 帧是预算内暂停恢复检查，无额外验证；之后正常继续，不每 100 步暂停。监督器维护单一 owner、串行验证/备份和有证据的故障处理。

每次开发验证固定 10 场景 ×32 视角=320 帧，运行在 5760、11520、…、57600 成功步；按实际 `(p,n)` 重新计算冻结协议下的 Seal/Wrench/Collision/Q20。先比较 instance Top-1 Q20@0.4，再比较 AP-R20；完全相同保留更早节点。验证保存 pending 事务，恢复补齐；训练 RNG 在验证前后保持，最终节点不额外重复一次。`best.pt` 的选择仅限该 run；不会自动宣称 E0–E5 的全局赢家。

完成须每组 230400 有效帧、10×320 帧验证、可恢复 best/last、连续规范日志与完整备份，六组都达标才关闭 README TODO1。当前没有发布完整验证结果、收敛/架构胜负、official-test、目标域微调或实机吸取结论。

## 性能优化验收与 rollout 状态

[性能方案](PERFORMANCE_OPTIMIZATION.md) 保留 `9d8ecc016c011d5b8d295e5e4220bd5e89fc5d97` 基线、CPU 原型和设计依据；当前分支已实现 CPython FP64 FPS、core/lip 复用、有界四球查询、selective NPZ、一维体素去重、CPU 几何上下文、有序后台预取，以及既有 checkpoint 的受限迁移。模型参数形状、原损失、完整观测和 v3 预算保持；六份硬件 profile 原始字节未改。

性能参数独立放在 [cd009-p1.json](../configs/performance/cd009-p1.json)。本机编译、`--performance-config`、supervisor 的 `performance_configs[host][experiment]`、迁移命令及备份要求见 [运行指南](MODEL_USAGE.md#性能实现构建与既有运行迁移)。普通 resume 同时绑定 sidecar SHA256 和 FPS 源码/编译器/flags/ABI/二进制身份，不能静默回退到另一后端。

迁移只接受上述 `98862f4` / `d634627b…` 源身份及已正常暂停的 current checkpoint；保留模型、Adam、AMP、RNG、已消费采样游标、成功步、best、验证和日志前缀。`migrations/performance-UUID/` 留存源状态硬链接、原 manifest、sidecar 和 receipt，备份一并验证和复制。迁移后前 50 次成功更新临时开启内部几何计时，之后回到 sidecar 默认关闭值；这 50 次属于原正式预算，不触发额外验证，也不改 sidecar 字节。

本轮本地全套测试 **296 passed / 1 skipped**，上述 GitHub CI 通过。真实四帧检查从 H20/E5 第 380 步 checkpoint 出发，保持原 AMP `atol=2e-4, rtol=2e-3`，没有 optimizer 更新：

- 原独立进程新旧对照失败；旧代码自身重复运行也失败，稀疏骨干输出已有小幅差异，并通过预测轴影响硬边界区域选择。失败记录保留，没有放宽容差或改成“通过”。
- 配对检查每帧仅计算一次相同的真实骨干输出，再执行旧/新下游路径；输入、完整查询索引/描述符与 RNG 精确核对，loss/输出/下游参数梯度/特征边界梯度按原容差比较，并把两份边界梯度分别通过同一真实骨干图回传。**四帧、零失败**；它验证本轮改动和梯度通路，不证明独立 CUDA 运行逐位确定。

H20/E5 已在第 380 步正常暂停并迁移；迁移前后核对的 **874 个张量完全一致**，原源 checkpoint 和证据归档保留，随后正常接续更新。L20/E0 在第 833 步、E3 在第 441 步迁移，分别核对 642 / 770 个张量完全一致；E0 的下一批帧与 RNG 接续通过。E3 初次启动因 GPU0 被其他进程占用而在创建 trainer 前停止，保留失败 worker 证据，经原有恢复流程改用同主机授权空闲 GPU3（CPU 24–31）；硬件 profile 和学习状态不变，没有终止占用进程。

原预算内正常窗口：E5 的 381–410 共 30 次成功更新，中位数 10.927 秒（旧窗口 33.835 秒），墙钟 341.046 秒；E0 的 834–863 共 30 次，中位数 4.208 秒（旧窗口 11.349 秒），墙钟 133.664 秒。观察到的中位数耗时比分别为 3.10 / 2.70，来自不同自然帧及运行时段，不能当作相同帧的因果倍率或跨架构公平排名。E3 的 442–461 共 20 次成功更新，中位数 12.184 秒（旧窗口 34.190 秒），墙钟 250.806 秒，观察到的中位数耗时比为 2.81；该组额外改变了 GPU/CPU affinity 放置。三个窗口均包含首步启动开销，窗口内没有 checkpoint/validation 事件；不能据此估算完整验证和长期备份开销。三项正常检查已完成，详见 [正常窗口表](PERFORMANCE_OPTIMIZATION.md#115-正常训练窗口) 与 [公开证据摘要](performance/P1_REAL_RESULTS.json)。六组完整训练、10 次开发验证、收敛和质量结论仍未完成。
