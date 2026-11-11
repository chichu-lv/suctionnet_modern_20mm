# 依据、版本与事实边界

文档整理日期：2026-11-11。以下是本次实际核对的版本或原始来源；网络宽度、采样额度、训练步数、损失系数和微调安排是本项目的起始设计，不能作为文献已证明有效的结论。

## 项目数据实现

| 来源 | 固定定位 | 支持的事实 |
|---|---|---|
| 用户已经使用的旧数据方法 | [`codex-profile@dc4ad6b00535c89ece1feef2751ba86006e5f97e`](https://github.com/chichu-lv/codex-profile/tree/dc4ad6b00535c89ece1feef2751ba86006e5f97e/suction20_data_starter) | 既有缓存字段、PCA覆盖法向、候选/像素索引、冻结配置 |
| 旧逐帧/物理实现 | [`full_data.py`](https://github.com/chichu-lv/codex-profile/blob/dc4ad6b00535c89ece1feef2751ba86006e5f97e/suction20_data_starter/suction20/full_data.py) | 原始候选轴计算S/W/C、最近几何匹配、GT-A/B、旧gt_normal行为 |
| 旧流程及归一化 | [`full_cli.py`](https://github.com/chichu-lv/codex-profile/blob/dc4ad6b00535c89ece1feef2751ba86006e5f97e/suction20_data_starter/suction20/full_cli.py) | 训练池物体候选p1/p99归一化与config_frozen |
| 当前项目核对基线 | [`suctionnet_modern_20mm@b2680fb47fe6f6a99fcb310e3428c77de1f55406`](https://github.com/chichu-lv/suctionnet_modern_20mm/tree/b2680fb47fe6f6a99fcb310e3428c77de1f55406) | 新producer原候选gt_normal、独立PCA字段、保存观测法向、r20-v2.1 |
| 当前物理及视角实现 | [`full_data.py`](https://github.com/chichu-lv/suctionnet_modern_20mm/blob/b2680fb47fe6f6a99fcb310e3428c77de1f55406/suction20/full_data.py) | Wrench L2、同一Collision、scene_frame0、read_frame、schema差异 |
| 当前归一化与上游参数化 | [`full_cli.py`](https://github.com/chichu-lv/suctionnet_modern_20mm/blob/b2680fb47fe6f6a99fcb310e3428c77de1f55406/suction20/full_cli.py)、[`patch_official.py`](https://github.com/chichu-lv/suctionnet_modern_20mm/blob/b2680fb47fe6f6a99fcb310e3428c77de1f55406/suction20/patch_official.py) | 半径/环带修改、72扇区约束、已有归一化流程 |

旧、新相关函数的源代码/AST已做对照：此次数据适配不改变 S/W/C/A/B 算法。用户告知数据已经完整生成；本次没有访问用户训练服务器上的整套产物，所以未声称已对实际缓存逐帧验收。文档中的具体兼容策略以用户已说明的旧提交为来源，不通过新生成器的默认值覆盖它。

## 主干与相关方法

| 来源 | 链接 | 本项目借鉴/纠正 |
|---|---|---|
| graspness_modern | [固定版本 `dda1d7586d4a9aea53da1636ffe64ae2c362e861`](https://github.com/SimonHanrath/graspness_modern/tree/dda1d7586d4a9aea53da1636ffe64ae2c362e861)；[model](https://github.com/SimonHanrath/graspness_modern/blob/dda1d7586d4a9aea53da1636ffe64ae2c362e861/models/graspnet.py) | spconv主干、SPconvUNet14D、默认512维；不是继续按MinkowskiEngine实现描述 |
| MultiGraspNet | [原论文 v1，III-B/III-C](https://arxiv.org/html/2602.06504v1) | 吸盘点+法向的5DoF表示；其吸盘refiner没有独立训练loss；不把BCE名称当硬标签证据 |
| UISN | [官方实现 `4ad50bfb4c29fc47f1e80eb8e4d95e666e8e48f2`](https://github.com/rcao-hk/UISN/tree/4ad50bfb4c29fc47f1e80eb8e4d95e666e8e48f2)；[论文 DOI](https://doi.org/10.1109/LRA.2024.3385609) | 实例条件与误差感知思路；官方MC-dropout路径不等价于几个共享确定性小头 |

本项目的单场景计算共享、I×R实验、残差尺度小头和目标成功头是本方案的组合设计。未做原方法完整复现实验，不能把 E0/E1 命名为完整 MultiGraspNet/UISN 基线。

## SuctionNet 物理标签与评价

| 来源 | 链接 | 用途 |
|---|---|---|
| SuctionNet 原论文 | [arXiv:2103.12311](https://arxiv.org/pdf/2103.12311) | Seal/Wrench物理代理及P@k平均式评价思路 |
| 官方 Seal 标注 | [`seal_annotate.py@ec5c328a48a2caac74dffb565811ec252da62907`](https://github.com/graspnet/suction_annotation/blob/ec5c328a48a2caac74dffb565811ec252da62907/seal_annotate.py) | 72扇区、环顶点、fit/deform、单物体self-collision；实际复用用户制作数据的对应脚本/版本 |
| 官方 API 评分 | [`eval_utils.py@0dac9f7e5b2f7ea801cddf9655cb82a259681976`](https://github.com/graspnet/suctionnetAPI/blob/0dac9f7e5b2f7ea801cddf9655cb82a259681976/suctionnetAPI/utils/eval_utils.py) | 旧API切向分量max写法、预测姿态评分、与本项目R20协议的差异 |
| 官方 API 评价汇总 | [`suctionnet_eval.py`](https://github.com/graspnet/suctionnetAPI/blob/0dac9f7e5b2f7ea801cddf9655cb82a259681976/suctionnetAPI/suctionnet_eval.py) | P@k与场景/视角评价组织 |
| 官方 API 输出对象 | [`suction.py`](https://github.com/graspnet/suctionnetAPI/blob/0dac9f7e5b2f7ea801cddf9655cb82a259681976/suctionnetAPI/suction.py) | 实际数组为score、normal、point顺序 |

本项目固定 R20、已有 p1/p99 Seal 归一化、L2 Wrench、当前三段Collision、共同NMS及候选上限。因此 AP-R20 是项目内部同协议比较，不与论文10mm公开数值直接对比高低；这项命名与协议明确并不要求重新制作已有数据。

## 用户提供的设计记录

《20mm 吸盘抓取网络设计方案》，41页完整AI对话，由用户在本次工作中提供。

- 第1–3页：全场景主干、实例条件、吸盘局部精炼的初始主线。
- 第16–20页：负例、最近几何匹配、gt_valid的修正。
- 第28–30页：物理标签与数据工程讨论。
- 第34–41页：后期模型代码、损失、不确定性、训练和执行设计。

引用此文件仅用于追溯设计演进，未将原私人对话PDF复制到仓库。这里采用经源码核对后的设计，不把AI对话中的文献描述自动当成论文事实。

## 尚待实测的内容

所有超参数是首轮明确可实施的默认值，不是搜索得出的最优值。没有公开预训练、目标微调、目标标签比例或真机性能的虚构数值。图中“当前设计”表示本轮规格选择；“producer已修正”仅描述当前源码，不意味着旧缓存自动改变或模型已经实现。
