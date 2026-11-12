# 模型实现入口

当前目录已实现 E0–E5 模型及实例融合、接触分组和评分头，支持公开 SuctionNet 预训练与两种少量自有场景微调接口。设计合同见 [docs/implementation](../docs/implementation/README.md)，可执行命令见 [运行指南](../docs/MODEL_USAGE.md)。

当前模块按以下职责组织：

```text
models/
  backbones/spconv_resunet14.py
  instance_fusion.py
  candidate_sampler.py
  contact_grouping.py
  heads.py
  suction20net.py
suction20/
  datasets.py
  target_dataset.py
  losses.py
  eval_r20.py
scripts/
  train_pretrain.py
  finetune.py
  evaluate.py
  infer.py
```

上述模块均已实现。[MODEL.md](../docs/implementation/MODEL.md) 给出输入输出、每层维度、512候选算法、五区几何、损失和 `encode_scene/score_at_pose/forward` API；[TRAINING.md](../docs/implementation/TRAINING.md) 与 [FINETUNING.md](../docs/implementation/FINETUNING.md) 给出训练合同。

CPU 合成测试已覆盖接口和恢复；H20 真实数据已完成 E1 2 次、E5 18 次成功更新及有界评价，详见 [验证记录](../docs/MODEL_VALIDATION.md)。完整公开预训练、E0–E5 同预算对照和真实目标域微调尚未执行。可选 sidecar 与 residual uncertainty 工具仍未交付，不是当前训练入口的前置条件。

数据入口优先支持用户已完成的旧 `dc4ad6b` 缓存：物理分数不改，标签轴按原候选恢复，缺少的观测法向由原RGB-D派生。只有新producer输出的 `gt_normal` 才已直接保证原候选轴语义。不要仅按字段名读取旧PCA法向，也不要用标签法向补输入。[调整细节](../docs/implementation/LEGACY_DATA_ADJUSTMENT.md)

输入观测、监督标签和追踪元信息分别管理；质量损失使用 `quality_valid`，轴损失使用独立 `axis_valid`，推理筛点不能读取GT有效掩码。默认q学习GT-B，真实执行成败使用独立task success头。

