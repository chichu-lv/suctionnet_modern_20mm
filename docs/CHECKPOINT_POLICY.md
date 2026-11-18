# Checkpoint 兼容性政策（2026-11-18）

## 当前支持的路径

- **同一实验严格续训**：`--resume` 核对训练配置与代码/数据/performance 身份，并保留模型权重、optimizer、scaler、采样顺序、RNG、调度和已提交 checkpoint；不得静默更换 sidecar 或训练语义。当前从头训练使用 v4 performance 配置。
- **权重初始化与目标域微调**：`--init-checkpoint` 和微调时的任务成功头加载仍保留；它们不是 CD-009 性能 checkpoint 迁移。
- **旧数据只读读取**：`legacy_dc4ad6b` 标签/法向适配仍是当前数据接口，不属于废弃的 checkpoint 迁移。

## 不再支持的过渡路径

- 移除了 `scripts/migrate_performance_checkpoint.py`、original/P1/P2 来源批准名单、P1/P2/P3 新迁移 receipt 创建与递归核验、迁移后 50 步诊断窗口以及 supervisor 配置切换例外。
- 当前代码检测到 checkpoint 的 `performance_migration` 标记时，会拒绝该历史续训或管理路径；不应直接删除标记来绕过身份检查。备份时同样拒绝遗漏其源 checkpoint 或档案。
- 若需核验、审计或恢复已产生的 CD-009 迁移档案，请使用清理前的固定源码版本 `fb8a2dacc996b98d86cd0aa52dd7a859f92dd32c`，配合历史 P1/P2/P3 文档及原运行目录中的源 checkpoint、manifest、receipt 和备份。确保使用与历史训练相符的环境和依赖，不要在最新代码中尝试跨版本恢复。

## 清理边界

- 保留历史 P1/P2/P3 sidecar 配置解析，以支持历史实验配置核对；CD-009 实验结果、日志和原文档不重写。
- 不修改模型张量名、权重数值、当前配置语义、现行 v4 训练/推理和目标微调路径。
- 本次代码清理不等于验证历史 checkpoint 在新代码上可复现，也不取消当前同身份续训的完整校验。
