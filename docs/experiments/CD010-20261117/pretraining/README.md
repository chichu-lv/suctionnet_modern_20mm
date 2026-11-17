# CD-010 E0–E5 预训练分析

从2026-11-16冻结分析快照导出；2026-11-17按用户要求准备GitHub发布。

- [完整分析报告](REPORT.md)：结果、曲线、各实验差异、耗时和资源。
- [数据字典](DATA_DICTIONARY.md)、[指标定义](metric-contract.md)。
- [六组摘要CSV](run_summary.csv)、[完整数值与配置摘要](run_summary.json)。
- [聚合验证](validation_results.csv)、[逐帧验证](validation_frames.csv)、[场景结果](final_scene_metrics.csv)、[事件](events.csv)。
- 完整优化尝试：E0 [分片1](training_updates/E0.part01.csv) / [分片2](training_updates/E0.part02.csv)；E1 [分片1](training_updates/E1.part01.csv) / [分片2](training_updates/E1.part02.csv)；E2 [分片1](training_updates/E2.part01.csv) / [分片2](training_updates/E2.part02.csv)；E3 [分片1](training_updates/E3.part01.csv) / [分片2](training_updates/E3.part02.csv)；E4 [分片1](training_updates/E4.part01.csv) / [分片2](training_updates/E4.part02.csv)；E5 [分片1](training_updates/E5.part01.csv)。
- [原快照完整性与模型哈希](integrity.json)、[导出来源与检查](EXPORT_MANIFEST.json)。

训练CSV共100,827条/44列，包含100,800次成功更新和27次AMP跳步。验证CSV包含60轮/19,200条记录，对应固定320张图，不能解释为19,200张独立验证图像。
所有数值原样保留；机器路径以占位符替代，启动命令/PID/GPU UUID移除。H20/L20为硬件平台标签。
模型权重、原始大日志、归档压缩包和案例大图/点云没有上传。八张PNG均小于1MiB；没有重新训练、推理或重算模型指标。

逐步训练表使用普通CSV分片，可用 `pandas.read_csv` 读取后按实验拼接；每片保留表头，行序与原日志一致。


训练逐步记录最终发布为 `pretraining/training_updates/E0.part01.csv` 等普通CSV分片（每片至多10,000行），共11片，合计100,827行，保留44列。候选表仍为gzip。`pretraining/EXPORT_MANIFEST.json`记录中间导出版本；最终发布路径、大小与哈希以顶层 `FILES.json` 为准。
