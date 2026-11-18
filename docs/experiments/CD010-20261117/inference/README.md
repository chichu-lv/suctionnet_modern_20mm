# 同一401帧测试场景：预训练 / 微调 / 原规划对比

这里保存两个模型的所有候选数值（分片CSV.gz）、逐帧Top1/计时、匿名样本与权重身份、原规划位姿和同实例比较。RGB、NPZ重复副本、观测点云与网页依赖图像未上传。未重新推理。

- [逐帧指标](frame_metrics.csv)：每模型401行，q/q0与微调任务评分独立保留。
- [原规划和同实例比较](original_comparisons.jsonl)：两个模型分别与原规划比较；不含大体积点云文件引用。
- [数量清单](inventory.json)、各模型目录下的模型身份/配置/匿名样本几何。
- `pretrained/candidates/*.csv.gz`、`finetuned/candidates/*.csv.gz`：包含每个候选全部数值字段，未只保留Top1/NMS；向量以JSON数组存于单元格。scene_rank空值代表未入场景NMS排序。

```python
from pathlib import Path
import pandas as pd
candidates = pd.concat([pd.read_csv(p) for p in sorted(Path('finetuned/candidates').glob('*.csv.gz'))], ignore_index=True)
frames = pd.read_csv('frame_metrics.csv')
```

预训练step2880按q排序；微调step1000按task_success_score排序。不同分数不直接比较大小，任务评分并非校准成功概率。人工历史状态属于原规划动作，不是新候选结果。382个规划点分布在381帧，20帧无原点；原位姿不是实测接触真值。距离/夹角只描述几何差异。本次不报告测试集新策略成功率。

完整三方RGB/点云网页和原始文件见[数据位置说明](../DATA_LOCATIONS.md)。

生产录制编号已替换为同一套TEST匿名ID，两个模型可直接按ID对齐。移除输入文件哈希、原物体追踪ID和溯源元数据；几何位姿、分数、排序和计时保持原值。

生产录制编号已替换为同一套TEST匿名ID，两个模型可直接按ID对齐。移除输入文件哈希、原物体追踪ID和溯源元数据；几何位姿、分数、排序和计时保持原值。

生产录制编号已替换为同一套TEST匿名ID，两个模型可直接按ID对齐。移除输入文件哈希、原物体追踪ID和溯源元数据；几何位姿、分数、排序和计时保持原值。

发布的逐帧来源记录仅保留匿名样本编号、模型标识、排序规则、随机种子和运行耗时。原始路径、采集时间戳、生产配置标识及相机标定元数据保留服务器；几何对比文件保留点、轴、评分及距离/夹角等分析数值。
