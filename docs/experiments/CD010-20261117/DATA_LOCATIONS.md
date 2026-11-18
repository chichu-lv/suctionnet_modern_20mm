# 未随Git上传的大文件与覆盖范围

本次按用户要求上传分析记录，排除原始RGB/深度/双目图像、实例掩码、观测点云BIN、模型PT、重复候选NPZ以及大归档包。下列路径以逻辑根表示；部署入口可在项目工作区原始README中找到。

| 逻辑根 | 原数据位置 |
|---|---|
| `${WORKSPACE}` | 当前项目工作区 `pose_model` |
| `${TARGET_RELEASE}` | 数据服务器 `datasets/pose/target-scenes-v1/20261116` |
| `${SOURCE_RELEASE}` | SuctionNet-r20-v1 `full-20261110` 冻结数据发布 |
| `${PRETRAIN_RUN}` | 训练机 `cd010-selected-training` 运行根 |
| `${FINETUNE_RUN}` | 训练机 `cd010-target-finetune-20261116` 运行根 |

## 预训练

完整原始记录/分析/模型本地目录：`${WORKSPACE}/artifacts/cd010-final-analysis-20261116/`。

- `raw/H20`、`raw/L20`：原始逐更新frame/augmentation traces、验证、配置等（原始日志约2.59GB），Git仅保留可分析的完整逐更新标量表及汇总/逐帧验证等。
- `models/HOST/runs/E*_seed42/{best,last}.pt`：已下载的12个best/last文件，校验哈希保留在分析完整性记录。
- `raw-H20.tar.gz`、`raw-L20.tar.gz`、`models-H20.tar`、`models-L20.tar`：下载归档，不随Git重复上传。
- 原训练运行目录为 `${PRETRAIN_RUN}/runs/E*_seed42/`；实际机器/硬件分配见预训练报告。

## 微调

完整训练记录和checkpoint：`${FINETUNE_RUN}/logs/`、`${FINETUNE_RUN}/run/`；包括 `best.pt`、`last.pt`、`step_200.pt`。最佳为第1000步，具体SHA256见 `finetuning/summary.json`。

准备数据/配置也位于 `${TARGET_RELEASE}/finetuning/E5-attempt-20261116/`；该目录不等同于实时checkpoint镜像。本地分析副本位于 `${WORKSPACE}/artifacts/cd010-target-finetune-20261116/`。

## 原场景和推理

`${TARGET_RELEASE}/raw/` 是原数据，`prepared/` 是分割/几何/划分处理结果。

- 预训练预测：`${TARGET_RELEASE}/baseline/E5-best/`。
- 微调预测：`${TARGET_RELEASE}/baseline/E5-best/finetuned/E5-best-step1000/`。
- `browser.html` 为现有局域网三方同时对比页；需要同根下的图像/点云资产。本Git目录保留其源代码与全部候选数值，但不提供缺失图像的伪网页副本。
- Git内候选CSV.gz与完整JSON/NPZ中的候选一一对应；主要省略重复元数据和二进制图像/点云。

## 导出变换

数值记录不重新计算预测；训练大表按实验无损gzip，候选按模型每约20k条分片无损gzip。微调验证JSONL保留匿名逐帧数值记录，另导出CSV方便分析。原文件路径使用逻辑根替换，PID/主机标识移除。生产原始标识、输入哈希、逐条人工标签/备注和混合控制台元数据不发布，匿名映射表保留本地；FILES.json对应本次实际发布字节。
