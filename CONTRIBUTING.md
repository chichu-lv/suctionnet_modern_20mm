# 开发约定

代码统一在 `suctionnet_modern_20mm` 仓库中维护。新增模块应归入 `suction20/`（数据）、`models/`（模型）、`scripts/`（运行脚本）、`tests/`（测试）或 `docs/`（文档）。

对于所有标签计算改动，需说明：计算公式、使用的数据源、是否改变候选索引、有效/无效标签语义，以及与 G0/G1/G2 的关系。避免把测试阶段的假数据误认为真实 SuctionNet 验收结果。

提交前运行：

```bash
pip install -e .
pip install pytest
python -m pytest -q
```

Issue 用于记录标签数据、模型、评测器、实机实验的待办事项。不要将官方数据集、训练生成文件或不必要的校验散件放入版本库。
