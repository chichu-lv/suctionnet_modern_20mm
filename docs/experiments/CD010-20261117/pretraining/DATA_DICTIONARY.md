# 数据字典与读取方式

这是从原始日志导出的标量分析表，不是新的训练结果。原始 JSONL/JSON 保留在未上传的完整归档；GitHub仅包含本目录列出的分析文件。训练记录按一次梯度累积后的优化尝试组织；每行不是一个微批，也不是一帧。

| 文件 | 行数（不含表头） | 含义 |
|---|---:|---|
| training_updates/E0.part*.csv … E5.part*.csv | 100827 | 100800 次成功优化与 27 次跳步 |
| validation_results.csv | 60 | 六组各十轮，标量聚合结果 |
| validation_frames.csv | 19200 | 六组×十轮×320帧，重复验证同一批图像 |
| final_scene_metrics.csv | 60 | 六组最终模型×十个开发场景 |
| events.csv | 见文件 | 开始/完成/验证/保存事件 |

## 训练字段

- `experiment`：E0–E5；`host`：记录来源。
- `global_step`：成功更新计数。跳步记录可能重复上一个成功步数，用 `optimizer_attempt` / `updated` 区分。
- `effective_samples_completed`：已计入预算的样本数；跳步不增加。
- `consumed_samples`：实际消耗样本数，含跳步；`skipped_samples` 为累计跳过样本数。
- `microbatch_size`、`gradient_accumulation_steps`、`effective_batch_size`：分别为微批、累积次数和有效批量。E0–E4=1/12/12，E5=4/12/48。
- `updated` / `gradients_finite`：Python 风格布尔字符串 `True` / `False`。
- `loss_total` 等：每次累积归一化后的损失。分项 q0/q/normal/seal/wrench/collision_free 为加权前数值；E0–E4 无辅助头，对应列为空。
- `grad_norm`：取消 AMP scale 后、裁剪前的梯度总范数。跳步可能为 `inf`；做数值统计时必须筛选有限值并说明是否只统计成功更新。
- `lr`：该步学习率。
- `amp_scale_before` / `amp_scale_after`：AMP 动态缩放值；`skip_reason` 非空行记录跳步原因。
- `elapsed_seconds`：一次优化尝试的主机墙钟时长（秒），不含日志写入、检查点和验证；不是累计秒数，也不是纯 GPU kernel 时间。
- `started_unix` / `ended_unix`：Unix 秒。显示北京时间时使用 Asia/Shanghai 或 UTC+08:00。
- `loader_wait_s`：暴露在训练路径上的数据加载等待。
- `timing_h2d_host` / `timing_forward_host` / `timing_loss_host` / `timing_backward_host` / `timing_optimizer_host` / `timing_diagnostics_host`：主机区间耗时，可能包含异步排队和偶发同步，不能当 GPU kernel profile。
- `timing_geometry_wait_host`：训练路径暴露的几何结果等待。
- `geometry_job_wall_s`：后台几何工作的累计时长；`producer_total_s`：预处理/拼批累计工作时间。它们与训练重叠，可能大于优化步耗时，不能与前向/反向相加。
- `peak_gpu_bytes`：该次优化尝试记录的 CUDA allocated 峰值（字节）。
- `peak_gpu_reserved_bytes`：CUDA reserved 峰值（含分配器缓存）。
- `gpu_allocated_bytes` / `gpu_reserved_bytes`：记录时的当前值，不是峰值。
- `peak_rss_bytes`：trainer 进程生命周期 RSS 高水位，不是该步瞬时值，不是全部 worker 总和。
- `points` / `voxels`：该优化尝试所有微批的输入点／体素数累计。

GiB=2^30 字节，MiB=2^20 字节。CPU 亲和范围是逻辑 CPU ID，不应直接称作物理核心数量。未记录连续 GPU/CPU 使用率、功率和温度；不能由以上字段反推这些曲线。

## 验证字段

详细公式与源码定位见 `metric-contract.md`。

- `instance_Top1_Q20_at_0_4`：所有可见 GT 实例中，最高分候选实际位姿 Q20≥0.4 的比例；缺候选算失败。
- `AP_R20`：项目定义的 k=1…50、阈值0.2/0.4/0.6/0.8 的平均 precision@k，不是普通 PR 曲线面积。
- `collision_rate`：有候选实例的 Top-1 碰撞比例。
- `no_candidate_rate`：可见实例中无候选的比例。
- `masked_dense_q0_mae`：固定有效标注点上的 proposal q0 回归误差。
- `masked_seed_q_mae` / `masked_seed_q_smooth_l1`：模型所选种子中有有效标签部分的误差；种子集合随模型变化。
- `normal_angle_median_deg` / `normal_angle_p90_deg`：逐帧法向误差中位数/P90，随后按场景平均，不是全局点集的分位数。
- `proposal_coverage_at_512`：存在高质量缓存标签点的实例中，512 种子至少命中一个高质量点的比例。分母不同于 Top-1。
- `labeled_seed_fraction` / `labeled_point_fraction`：可用于标签监督的覆盖率。
- 比例字段原始范围0–1，报告显示百分数；MAE、角度原值不乘100。
- `scene_count`=10、`view_count`=320 是总数。其他 `*_count` 聚合字段往往是逐帧平均计数；不得按总计理解。
- `validation_frames.csv` 中 `scene_id/frame_id` 确定开发集图像，多个轮次会重复这些键；完整主键为 experiment/global_step/scene_id/frame_id。
- 聚合规则：场景内对有效帧平均，再对场景等权平均。缺失/未定义值保留为空，不自动补0。

## 不在这些日志中的内容

训练日志没有逐点完整预测数组、每一步完整梯度张量或每一帧图像；验证 JSON 没有所有候选的实际位姿和逐候选 Seal/Wrench 细分。模型 checkpoint 含恢复状态，输入图像/深度属于独立冻结数据集。此前四帧推理 NPZ 属于单独案例产物，不在本轮原始训练包。

## 读取示例

```python
import csv
from pathlib import Path

root = Path('.')
with (root / 'training_updates/E5.part01.csv').open('r', newline='') as stream:
    for row in csv.DictReader(stream):
        if row['experiment'] == 'E5' and row['updated'] == 'True':
            step = int(row['global_step'])
            total_loss = float(row['loss_total'])
            # 按需分析；不需要先读入全部原始 JSONL。
```

六组训练CSV分片合计100,827行，每个文件均保留原始44列，所有字段值与原表逐行核对一致。其他四个原始CSV逐字节一致。见 [导出清单](EXPORT_MANIFEST.json)。原始日志提取脚本需要未上传的大日志；不将此GitHub包描述为完整训练恢复包。

也可直接用 pandas 读取：

```python
import pandas as pd
updates = pd.read_csv("training_updates/E5.part01.csv")
# 如需逐字比较原始文本，使用 dtype=str, keep_default_na=False。
```

训练表使用普通CSV分片，没有丢弃行或列；推理候选gzip使用mtime=0和空文件名头。


训练逐步记录最终发布为 `pretraining/training_updates/E0.part01.csv` 等普通CSV分片（每片至多10,000行），共11片，合计100,827行，保留44列。候选表仍为gzip。`pretraining/EXPORT_MANIFEST.json`记录中间导出版本；最终发布路径、大小与哈希以顶层 `FILES.json` 为准。
