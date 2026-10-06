# 二维光声血管图像拼接

当前稳定版使用 **SIFT 点到点初值＋边缘距离场精化**，不是显式点到线版本。本次整理只准备代码上传，不实施后续优化。

## 当前处理流程

1. 读取每个 `.dat` 中 MATLAB 顺序的 500×1000 矩阵，拆分正返两张 500×500 图，校正扫描轨迹和方向。
2. 一个文件对应一个全局位姿。正返内部对齐通过验证后融合，否则选取较好的单向图像。
3. 适度滤除紧凑亮点；配准不回退到未去噪图像。
4. 建立相邻、短间隔和外观候选关系。SIFT＋RANSAC 提供初值，点到点 LM 后联合边缘距离场进行鲁棒 LM 精化。
5. 对满足条件的断开相邻帧尝试增强特征；弱结构帧可尝试严格验证的相位相关平移。无可靠证据时保持断开。
6. 每个连通分量内优化前端相对变换构成的位姿图，再融合局部块。
7. 默认选择至少 50 个文件的局部块，进行块间配准、闭合一致性检查及块级 LM，最终加权融合。

边缘是 Canny 检出的图内亮度边缘，不是图像边框，也不是已确认身份的血管。当前距离场查找最近边缘，尚未验证血管方向、身份和分支对应。Canny 阈值为 35、90。配准视图和预览将固定强度范围 0.02～0.25 映射到 0～255，科学输出保持浮点数。

融合采用边缘渐变权重的加权平均，不取最大值。当前先融合局部块，再变换融合大图，尚未实现最终画布的单次投影。默认没有绝对旋转先验；块间闭合角度检查不等于强行摆正图像。

实验性的局部地图几何、强度后端因子已移除，普通位姿图 LM 保留。显式点到线实验入口已撤回，历史辅助模块仍保留但不进入主流程。

## 环境与安装

需要 Python 3.10 或更新版本，以及 NumPy、SciPy、OpenCV；依赖见 `pyproject.toml`。在仓库根目录执行：

```bash
python -m pip install -e .
python -m unittest discover -s tests -q
```

本机也可直接使用下面的环境，无需重新安装包。换机器时将 Python 路径改成自己的环境，并准备数据目录。

## 运行

### 全量处理与块间拼接

```bash
PYTHONPATH=src /home/gsp/miniconda3/envs/vision/bin/python -m pam_mosaic.cli --data PAM_Skull_FYH_20260929 --output outputs/current_run_full
PYTHONPATH=src /home/gsp/miniconda3/envs/vision/bin/python -m pam_mosaic.tile_stitch --components outputs/current_run_full --output outputs/current_run_stitched --min-frames 50
```

安装包后对应入口为 `pam-mosaic` 和 `pam-stitch-tiles`。每次新实验选择新的输出目录，不覆盖保留结果。多个互不连通的大图会分别保存在 `mosaic_001`、`mosaic_002` 等目录中。

### 小规模运行检查

```bash
PYTHONPATH=src /home/gsp/miniconda3/envs/vision/bin/python -m pam_mosaic.cli --start 9 --stop 18 --output outputs/smoke_full
PYTHONPATH=src /home/gsp/miniconda3/envs/vision/bin/python -m pam_mosaic.tile_stitch --components outputs/smoke_full --output outputs/smoke_stitched --min-frames 2
PYTHONPATH=src /home/gsp/miniconda3/envs/vision/bin/python -m unittest discover -s tests -q
```

降低块大小门槛仅用于运行检查。子集会重新估计校准参数，不可与全量作严格效果比较。

### 参数说明

- `--start/--stop`：选择文件编号范围。
- `--logs` 或重复 `--log`：提供扫描日志。
- 默认启用正返验证、适度去噪和相邻恢复；`--no-frontend-recovery`、`--no-suppress-speckles` 可关闭后两项。
- `--baseline`：历史直接正返平均路径，会关闭当前去噪与相邻恢复；块阶段可配合 `--no-cycle-gating`。不保证逐像素重现历史结果。
- `--validate-rotations`：历史实验选项，默认关闭。
- 已撤回的 `--registration-model`、`--frozen-run` 不能用于当前版本。

## 缓存与复现

`.run_cache/` 每 100 帧提交恢复图像清单，每 50 对提交配准记录，阶段结束时也提交。相同条件下再次运行可复用这些阶段；位姿优化与融合不是逐块断点续跑。

缓存检查原始文件元信息、校准参数、配置、源代码哈希、库版本和 OpenCV 线程数。条件改变时会拒绝复用，应使用新输出目录，不要强行套用历史缓存。校准与数值库差异可能影响临界匹配，算法对比应固定输入和运行环境。

## 代码结构

代码位于 `src/pam_mosaic/`：

| 模块 | 作用 |
|---|---|
| `dataset.py` | 数据索引、扫描编号和日志 |
| `reconstruction.py` | 原始读取、拆帧、扫描轨迹校正 |
| `roundtrip.py` | 正返内部校准、验证、融合或选择 |
| `denoise.py` | 适度亮点处理 |
| `frontend/legacy.py` | 当前 SIFT、距离场及相邻恢复 |
| `frontend/geometry.py`、`tracking.py` | 历史显式点到线辅助模块，当前主流程不调用 |
| `registration.py` | 配准接口兼容导入层 |
| `graph.py`、`poses.py` | 候选关系、连通分量和位姿图 LM |
| `pipeline.py`、`cli.py` | 主流程及命令入口 |
| `checkpoint.py` | 缓存和配置一致性检查 |
| `fusion.py`、`tile_stitch.py` | 融合、块间匹配和大图生成 |
| `audit.py`、`evaluation.py` | 图像与接缝检查，不修改位姿 |
| `comparison.py`、`controlled_view.py`、`frozen_inputs.py` | 历史实验查看与缓存工具，非默认流程 |
| `frame_pipeline.py` | 历史正返独立定位实验，不是当前方案 |

`tests/` 保存测试。`scripts/compare_factor_free.py` 比较本机历史输出，需先具备相应结果目录，不是首次运行必需步骤。`pam-mosaic-frames` 是历史入口，不应代替主入口；`pam-audit-mosaic` 是检查入口。

## 输出

- `mosaic_float64.npy`：浮点强度图。
- `weight_map.npy`：融合权重和有效覆盖。
- `mosaic_preview.png`：固定显示范围的预览。
- `poses_xytheta.npy`：局部块逐帧位姿，平移为像素、角度为弧度。
- `tile_poses_xytheta.npy`：块级位姿。
- `report.json`：文件归属、匹配、处理决策和结果。

小块和孤立帧保存于局部输出，但不一定进入大图。处理全部数据不代表全部进入大图。当前没有可用的物理像素标定，不能把像素距离直接解释为毫米。相关性、边缘距离和接缝风险仅是图像启发式指标，不证明解剖对应正确。

## 保留的结果

当前稳定版：`outputs/frontend_only_stitched/mosaic_001/mosaic_preview.png`。

此前全量运行处理 1301 份文件、5940 对候选关系，相邻成功 1106 条，形成 125 个分量，7 个大块组成包含 1083 份文件的 3050×3010 大图。这是此前全量记录，不是本次整理重新跑出的全量结果。

- `outputs/lm_*`：原始基线。
- `outputs/moderate_denoise_*`：之前的去噪版本。
- `outputs/frontend_only_*`：当前距离场版本。
- 未采用的显式点到线输出 `point_line_full`、`point_line_stitched`，旧实验 `controlled_v2`、`paatam_2d` 以及本次 `release_check_20261006_*` 临时结果已移入系统回收站，不再位于仓库中；需要时可从回收站恢复。
- `outputs/backend_code_archive/`、`outputs/point_line_code_archive/`：撤回代码的本地备份；历史报告中的旧命令只适用于归档代码。

## Git 上传注意事项

本次整理验证（2026-10-06）：20 项自动测试通过，4 个已声明命令入口可导入；真实数据第 9～18 号文件的小规模恢复、29 对候选配准、局部融合和最终输出通过，输出数组有限且有有效覆盖。该子集最终只选择到一个满足测试门槛的局部块，因此不代表重新验证了全量块间配准效果。本次没有重新运行全部 1301 份数据。验证使用 NumPy 2.4.4、SciPy 1.17.1、OpenCV 4.13.0；临时测试输出验证完成后已移入回收站。

`.gitignore` 排除原始数据、`outputs/`、缓存、环境及本地工具状态。新机器需自行准备数据；输出和代码归档不会随 Git 上传，请按需另行备份。

提交时请包含新增的 `frontend/`、`checkpoint.py`、检查模块、测试和需要的脚本，并保留已确认的旧后端删除记录。只提交已跟踪文件的修改会漏掉必要的新模块。

## 后续优化（尚未实施）

依次评估缺少可用血管结构的帧、距离场的有效重叠区域、方向及外观一致性，再独立测试单次投影融合。无可靠血管结构的帧不必强行加入大图。这些优化尚未进入当前代码，上传本版后再开始。
