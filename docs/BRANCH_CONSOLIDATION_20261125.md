# 2026-11-25 分支收拢记录

## 目的与结果

将 E0–E5 主线、单实例 A/B 共同设计、两条实现及其性能优化收拢到同一基线，再从该基线另开模型化方案 2 开发分支。本次收拢保留两路线模型、各自配置、共同数据工具、实验记录及所有旧提交；新方案的数据口径独立定义。

本地收拢分支为 `integration/consolidate-20261125`，起点是下面的 `main`。远端发布由本次任务最后一步完成；本记录不将本地合并等同于已推送或已删除远端引用。

## 合并前的固定引用

| 分支 | 合并前 head | 收拢路径 |
|---|---|---|
| `main` | `0080c36ec52946e3275fc563cf46ebeea359edb0` | 收拢起点；保留无内容校验和约束与共用运行修复 |
| `docs/instance-routes-design-20261118` | `a9f4a8451c6bc42cfd8867dd96c051398644ad71` | 已是 A/B 优化分支共同祖先 |
| `route-a-3d-rgb` | `59e05c0934bf07ea5c6fb47818c61966b3c8bb02` | 已是 optimize/A 的祖先 |
| `route-b-2d-projected-rings` | `e6d9578f5fb39174cf9a81c769672800c06a9306` | 已是 optimize/B 的祖先 |
| `optimize/route-a-e5-audit` | `47c9345fb7e791e1ac2ddcf2afc4afa7486900fd` | 第一次非快进合并 |
| `optimize/route-b-e5-audit` | `62bc0b99f557ff2619bf1e689eadc62f3d87f4e9` | 第二次非快进合并 |

发布的 merge 提交为 `395680d8fe294c14bbb9126479f0e9803ea89660`（A）与 `540978b89a20063886cd1af75c475ed127f5c40b`（B）。它们分别使用与本地已验证合并 `56f6774`、`fe7ac79` 完全一致的 Git tree，保留相同两侧父链语义。没有 squash、rebase、强制改写或以单侧覆盖代替合并。通过 `git merge-base --is-ancestor <旧head> HEAD` 逐一确认上述六个 head 均可达，发布后再次读回核对。

## 冲突处理

1. 共同 `suction20/instance_*` 数据、缓存、训练、推理和评价源码在两优化分支已一致；同名文件按一致版本合并。A/B 专属模型及测试各自保留，A 原生接触扩展入口和头文件构建依赖保留。主线稀疏骨干的 `in_channels` 参数保留。
2. README、路线入口、验证历史与路线图保留两侧已有内容，将当前 A/B 入口改为相对路径。仍需引用旧分支的历史链接固定到合并前的 commit SHA，后续清理分支名不会使这些链接失效。
3. 2026-11-18 的 `pilot.md` 按 A/B 两节合并。两条路线原始 L20 JSON 因使用相同路径且代表不同试验，分别保留为 `l20-execution-route-a.json`、`l20-execution-route-b.json`、`l20-preparation-profile-route-a.json`、`l20-preparation-profile-route-b.json`；原通用 JSON 路径改为两路线文件索引，不拼接不兼容的数值字段。
4. `l20-formal-step-times.csv` 保留一个表头及全部原始行：A 661 行、B 496 行，合计 1,157 行；已检查 `(route, step)` 唯一。
5. [AGENTS.md](../AGENTS.md) 的禁止 SHA-256/替代内容摘要规则持续有效。保留 schema、有限值、数组和姿态/标签语义验证。

## 验证范围

环境：Python 3.12.14、PyTorch 2.5.1+cpu、NumPy 2.5.5、SciPy 1.17.0、trimesh 5.2.2、pytest 9.2.0，Torch/BLAS 单线程。主 runtime 缺少 Torch/pytest/trimesh，测试依赖安装在仓库外的隔离目录，通过 `PYTHONPATH` 使用，没有修改项目依赖声明。runtime 的默认 clang 不存在；显式设置 `CC=gcc CXX=g++ LDSHARED="gcc -shared"` 后，两原生扩展成功构建。

| 检查 | 结果 |
|---|---|
| `python -m pytest -q --ignore=tests/test_prefetch.py` | **1,027 passed / 1 skipped，177.66 秒** |
| `python -m pytest -q tests/test_prefetch.py -k 'not test_spawn_prefetch_matches_serial_and_resumes_consumed_state'` | **2 passed / 1 deselected**；这两项与上一行不重叠 |
| `python -m pytest -q tests/test_no_sha256_verification.py` | **60 passed**；已包含于联合套件，不重复累计 |
| `python -m compileall -q suction20 models scripts tests` | 通过 |
| `git diff origin/main --check` | 通过 |
| 六个旧 head 的祖先检查 | 全部可达 |
| 本次导航涉及的 192 个相对链接、74 个已跟踪 JSON | 链接存在、JSON 可解析 |
| A/B 专属运行模块与各自优化 head 对照 | A 两模块及原生头、B 两模块原样保留 |

首次确实尝试了不排除文件的完整套件。其旧 `tests/test_prefetch.py::test_spawn_prefetch_matches_serial_and_resumes_consumed_state` 重现既有本地阻塞：30 秒诊断栈停在 Torch DataLoader `_try_get_data`，随后中断该次运行。没有把该次运行计为完整通过，也没有修改/新增测试 skip。联合套件唯一条件 skip 需要真实 `R20_RELEASE_ROOT`。GitHub 工作流仍执行不排除文件的 `python -m pytest -q`；远端最终提交的 CI 结果另行核对。

以上覆盖两路线同时存在时的数据、模型、梯度、缓存、严格恢复和推理等 CPU 回归。本次不启动训练服务器、不修改正在运行的 checkpoint、不重算真实数据；CPU 回归不代表新 GPU 性能或真实吸取效果。

## 远端引用清理条件

只有在合并基线和新开发分支均成功推送、远端读回确认旧 head 全部可达后，才清理已收拢的五个旧分支引用。提交历史仍由基线保留。本记录列出的旧 SHA 可直接用于 `git show`、GitHub commit 链接或重新创建分支。
