# 模型实现入口

当前目录尚未实现网络。实现规格已经固定在 [docs/implementation](../docs/implementation/README.md)，以公开SuctionNet预训练、少量自有场景微调为目标。

建议按以下明确职责实现，不另建一套重复训练接口：

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
```

上述均是待新增模块。[MODEL.md](../docs/implementation/MODEL.md) 给出输入输出、每层维度、512候选算法、五区几何、损失和 `encode_scene/score_at_pose/forward` API；[TRAINING.md](../docs/implementation/TRAINING.md) 与 [FINETUNING.md](../docs/implementation/FINETUNING.md) 给出训练实现。

数据入口优先支持用户已完成的旧 `dc4ad6b` 缓存：物理分数不改，标签轴按原候选恢复，缺少的观测法向由原RGB-D派生。只有新producer输出的 `gt_normal` 才已直接保证原候选轴语义。不要仅按字段名读取旧PCA法向，也不要用标签法向补输入。[调整细节](../docs/implementation/LEGACY_DATA_ADJUSTMENT.md)

输入观测、监督标签和追踪元信息分别管理；质量损失使用 `quality_valid`，轴损失使用独立 `axis_valid`，推理筛点不能读取GT有效掩码。默认q学习GT-B，真实执行成败使用独立task success头。

