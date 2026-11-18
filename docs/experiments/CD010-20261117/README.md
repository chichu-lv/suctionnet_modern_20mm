# CD-010 实验分析资料：预训练、目标微调与推理对比

本目录发布已完成实验的可分析记录（2026-11-17）。数据均来自保存的实验日志和预测，不重新训练、不重跑推理。

| 内容 | 记录规模 | 入口 |
|---|---:|---|
| E0–E5 正式预训练 | 100,800次成功更新，100,827次尝试；每组230,400样本 | [分析报告](pretraining/REPORT.md) · [文件索引](pretraining/README.md) |
| 预训练验证 | 60个节点，19,200条帧记录（重复使用固定320帧） | [聚合结果](pretraining/validation_results.csv) · [逐帧结果](pretraining/validation_frames.csv) |
| E5 目标微调 | 1,000次成功更新，1,001次尝试；11次验证含基线 | [结果和曲线](finetuning/README.md) · [验证指标](finetuning/validation_metrics.csv) |
| 目标监督清单 | 2,162条接受记录，629条排除记录 | [处理说明](finetuning/data/README.md) · [划分统计](finetuning/data/summary.json) |
| 微调前后测试场景推理 | 两模型各401帧、各205,312个候选 | [推理数值资料](inference/README.md) · [逐帧结果](inference/frame_metrics.csv) |

## 读数据

普通CSV可直接读取；逐次更新表使用普通CSV分片，全部候选表使用无损gzip压缩，数字列没有舍入/平滑/降采样。向量与列表存为JSON字符串；空scene_rank表示未进入场景NMS，不表示候选不存在。

```python
from pathlib import Path
import pandas as pd
updates = pd.read_csv('pretraining/training_updates/E5.part01.csv')
validation = pd.read_csv('finetuning/validation_metrics.csv')
candidates = pd.concat([pd.read_csv(p) for p in sorted(Path('inference/finetuned/candidates').glob('*.csv.gz'))], ignore_index=True)
```

JSONL逐行读取；[历史文件清单](FILES.json)保留当时的路径、大小和摘要元数据；当前不执行其摘要校验。大图像、模型checkpoint、点云、重复NPZ及大体积原始trace/归档包未上传，位置与覆盖范围见[DATA_LOCATIONS.md](DATA_LOCATIONS.md)。本目录不是可脱离外部数据集直接启动训练的完整运行镜像。

## 指标解释与限制

- 预训练 [指标定义](pretraining/metric-contract.md) 和 [数据字典](pretraining/DATA_DICTIONARY.md)：AP_R20采用项目定义，不能当作通用PR曲线面积；best验证用于选模型，不能当作独立测试表现。
- 微调验证固定399条历史规划动作的人工结果；成功含义为完成吸取并搬运到目标位置。规划位姿未经独立执行测量，AUROC/PR-AUC衡量这些记录的关联排序，不是模型新动作的真实成功率。
- 401帧是观测测试场景，不等同于401条有成败标签的动作；此轮未报告新策略实际成功率。预训练按q排序，微调按任务评分排序，分数含义不同。
- 原规划/预训练/微调的距离与夹角是几何差异，不是精度或优劣结论。相机几何采用数值复现，重力使用已批准的皮带法向近似。

## 配置与来源

预训练硬件、批量、迭代数与汇总见 `pretraining/`；微调启动/实际配置见 `finetuning/config/`；推理权重标识、排序规则与配置见 `inference/{pretrained,finetuned}/`。数据准备/推理/三方展示源文件见 `scripts/`，训练模型代码沿用本分支已有实现，未在本次发布改动。

机器绝对路径改为 `${TARGET_RELEASE}`、`${SOURCE_RELEASE}`、`${PRETRAIN_RUN}`、`${FINETUNE_RUN}`、`${BACKBONE_ROOT}` 等逻辑根路径；PID/主机名等部署信息移除。保留实验编号、硬件类型、原始数值、权重/源码哈希；生产样本、观测和分组使用匿名ID，输入文件哈希、逐条人工标注和原始溯源元数据不发布。变换后文件的发布哈希见FILES.json；匿名映射表保留本地，不随资料发布。

本资料包为去标识的分析导出。生产原始动作清单、逐条人工标签/备注、录制标识、输入文件哈希与控制台混合元数据保留服务器；不影响分析已发布的训练损失、验证指标、候选数值和聚合统计。


训练逐步记录最终发布为 `pretraining/training_updates/E0.part01.csv` 等普通CSV分片（每片至多10,000行），共11片，合计100,827行，保留44列。候选表仍为gzip。`pretraining/EXPORT_MANIFEST.json`记录中间导出版本；最终发布路径、大小与哈希以顶层 `FILES.json` 为准。
