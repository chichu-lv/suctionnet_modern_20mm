# 开发约定

代码统一在 `suctionnet_modern_20mm` 仓库中维护。新增模块应归入 `suction20/`（数据）、`models/`（模型）、`scripts/`（运行脚本）、`tests/`（测试）或 `docs/`（文档）。

对于所有标签计算改动，需说明：计算公式、使用的数据源、是否改变候选索引、有效/无效标签语义，以及与 G0/G1/G2 的关系。避免把测试阶段的假数据误认为真实 SuctionNet 验收结果。

本项目禁止新增或恢复 SHA-256 计算与校验，也不使用其它内容校验和替代原有全文件扫描。数据、缓存、checkpoint、备份和脚本均遵守此约束。保留实际字段/维度/有限性、配置、ID、文件长度、原子写入与运行所有权检查；历史摘要字段忽略且不再写入新记录。完整规则与兼容边界见 [数据读取与恢复检查](docs/NO_CONTENT_CHECKSUMS.md) 和 [AGENTS.md](AGENTS.md)。提交前的测试包括 `tests/test_no_sha256_verification.py`。

提交前运行：

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[model]'
python -m pip install pytest
python -m pytest -q
```

上述是CPU测试环境，覆盖数据与模型测试；真实release测试需显式提供对应数据路径。生产CUDA/spconv环境按 [模型运行指南](docs/MODEL_USAGE.md) 安装，无需为了CPU测试另装GPU主干。仅修改说明文档时检查相关命令、链接和状态与现有实现一致即可。

同步维护 [README的Done/TODO](README.md) 和 [项目路线](docs/ROADMAP.md)，分别记录已实现功能、已执行验证与尚待完成的实验；新增验证结果写入 [MODEL_VALIDATION](docs/MODEL_VALIDATION.md)，注明数据、设备和训练步数范围。

Issue 用于记录标签数据、模型、评测器、实机实验的待办事项。不要将官方数据集、训练生成文件或不必要的校验散件放入版本库。
