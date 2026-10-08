# 二维光声血管图像拼接

将一组光声 `.dat` 文件读取或恢复为二维图像，并拼接为一张或多张大图。支持旧双向扫描恢复，以及新 PAM 数据的 MATLAB MAP1 读取方式；两种输入共用逐帧配准、位姿图优化、局部块融合和块间拼接。每个 `.dat` 文件对应一个全局采集位置。

当前默认流程在**帧间和块间**都使用 SIFT 点到点 LM，以及 SIFT 点与 Canny 血管边缘距离场共同约束的第二次 LM；两级处理方式一致。块间精化改动只影响第二阶段，不需要重新恢复 `.dat` 或生成局部块。本数据集的 1301 份文件已完成全量局部块重跑；使用这些局部块重跑新版块间阶段后，仍是 7 个块、1083 份文件组成一张 3050×3010 大图，与旧版视觉效果基本一致。旧版正式结果未被覆盖，路径、数值对比及校验值见 [STABLE_RELEASE.md](STABLE_RELEASE.md)。

## 算法流程

```mermaid
flowchart TD
    A["按采集编号读取 .dat"] --> Q{"输入格式"}
    Q -->|bidirectional500| B["每个文件重建正向/返向 500×500 图像"]
    Q -->|pam512| P["按 MATLAB MAP1 读取 512×513 单帧；温和去亮点"]
    B --> C["校准、验证；可对齐则融合，否则择优；保守去亮点"]
    C --> BG["逐帧估计背景众数并扣除基线；保留负值，可关闭"]
    P --> BG
    BG --> D["候选帧对：相邻、间隔 2/3/5、外观相似"]
    D --> E["每对帧：SIFT + RANSAC 粗变换"]
    E --> F["点到点 LM；再用 SIFT 点 + Canny 边缘距离场做 LM"]
    F --> G["筛选有效帧间变换；必要时保守恢复相邻关系"]
    G --> H["按有效变换划分帧连通组"]
    H --> I["每组：传递变换求初始位姿 → 位姿图 LM"]
    I --> J["按优化位姿加权融合，得到局部块"]
    J --> K["选取达到最小帧数的局部块，两两尝试配准"]
    K --> L["每对块：SIFT + RANSAC → 点到点 LM → SIFT 点 + Canny 边缘距离场 LM"]
    L --> M["筛选有效块间变换，并检查多路径关系的一致性"]
    M --> N["按块间关系划分连通组；每组传递变换求初始位姿 → 位姿图 LM"]
    N --> O["按优化后的块位姿加权融合，输出最终大图"]
```

点到点/边缘距离 LM 优化的是**一对图像间的变换**；位姿图 LM 优化的是**一个连通组内各帧或各块的摆放位姿**，使其预测的相对变换接近前一步测得的变换。没有可信关系连接的组分别输出，不强行拼接。块间边缘距离精化可用 `--no-refine-geometry` 关闭。

在本数据集的同一批局部块上，新旧块间配置的最终图尺寸和入图文件数相同，块位姿最大平移分量差约 0.69 像素，共同覆盖区域的浮点强度平均绝对差约 0.000745。边缘匹配指标略有改善，但没有证据表明最终大图质量明显提升；启用新版默认配置是为了保持两级配准方法一致，而不是宣称它在该数据集上显著优于旧版。新版结果在 `outputs/tile_edge_refinement_20261006/`，旧配置可用 `--no-refine-geometry` 复现。

## 1. 输入数据要求

默认输入格式为 `bidirectional500`，支持原项目的双向采集数据：

- 每个 `.dat` 文件先有 **8 字节头**，随后是 **500×1000 个小端 float64** 数值。
- 数值按 MATLAB 列优先顺序排列；前后各 500 列分别是正向、返向扫描。恢复后每向为 500×500 像素。
- 文件名中要有唯一的整数采集编号，代表采集顺序。默认识别 `1_Col2_tdT_58_scan1.dat` 这样的名称，其中 `scan1` 是扫描批次。
- 可用 `--filename-regex` 适配其他文件名。正则表达式必须包含命名组 `(?P<index>...)`，可选 `(?P<scan>...)`；不提供 `scan` 时所有文件视为同一批次。编号不能重复。
- 扫描日志可选。`--logs` 读取本项目格式的日志目录，或重复使用 `--log` 指定文件。没有日志仍可运行；日志时间戳不参与配准。

如果数据不属于本文支持的两种格式，矩阵尺寸、数值类型、头长度、列顺序或扫描布局不同，单换文件名规则不够，需要添加相应读取方式并重新验证。新数据集应先人工检查几张恢复帧的方向、强度和血管结构。

另支持 `--input-format pam512`：按新数据附带的 `PAMreadtoMsCamPA.m` 独立读取。文件头是两个小端 int32（1024、512），随后是 512×1024 个小端 float64，仍按 MATLAB 列优先顺序排列。严格复现 `MAP(:,512:1024)`，输出 **512×513** 单帧；保留脚本包含的边界列，不转置、不翻转、不做旧数据的余弦轨迹重采样或正返融合。

读取后默认复用 `denoise.py` 的温和去亮点参数：相对局部中值的强度差阈值 0.02、连通面积不超过 4 像素、宽高不超过 4 像素、3×3 中值窗口。随后通过 `background.py` 估计逐帧背景基线：在第 0.5～70 百分位范围建立 256 格直方图，经标准差为 2 格的高斯平滑后取众数，再从整帧扣除该值。校正后的负值继续保留，不截零、不改变单帧内部强度差；这一估计并非物理强度标定，背景较少的数据需要单独验证。去亮点改动数与 `background_baseline` 逐帧写入报告；缓存及最终数值图存的是预处理后的浮点强度，不压成 0～255。

背景校正默认对两种输入格式启用：新格式在读取、去亮点后校正；旧双向格式在轨迹恢复、正返验证/融合或择优、去亮点后校正，再进入帧间配准。旧格式内部的正返验证和校准仍按原流程执行。配准和融合均使用校正后的浮点图；配准/预览窗口为新格式 0～0.06、旧格式 0～0.25。关闭校正时，旧格式恢复原有 0.02～0.25 窗口。两阶段通过报告传递窗口，不对大块重复扣背景。

`--no-background-correction` 关闭背景校正；新格式同时加 `--no-suppress-speckles` 才得到纯 MATLAB 读取结果（转换为 float32）。复现旧稳定版时，第一阶段必须加 `--no-background-correction`，第二阶段使用 `--no-refine-geometry`。后续候选关系、两次局部 LM、位姿图 LM 和加权融合均复用现有算法。改变背景校正配置或源代码后，请选择新输出目录，不能沿用旧运行缓存。

## 2. 环境与安装

需要 Python 3.10 或更新版本，依赖为 NumPy、SciPy、OpenCV。建议在独立 Python 环境中，从仓库根目录执行：

```bash
python -m pip install -e .
python -m unittest discover -s tests -q
```

安装后有两个正式命令：`pam-mosaic`（恢复与局部块）和 `pam-stitch-tiles`（块间拼接）。若不安装，也可在仓库根目录用 `PYTHONPATH=src python -m pam_mosaic.cli` 和 `PYTHONPATH=src python -m pam_mosaic.backend.tile_stitch` 运行。目录重组后，已安装的旧环境需重新执行 `python -m pip install -e .` 以更新第二个入口。

## 3. 用新数据运行

以下以 `/data/new_scan` 为输入举例。输出目录由使用者指定；建议每组数据和每次参数调整都用新的目录。命令会拒绝覆盖已有 `report.json` 的完整结果。

```bash
pam-mosaic \
  --data /data/new_scan \
  --output /data/new_scan_result/components

pam-stitch-tiles \
  --components /data/new_scan_result/components \
  --output /data/new_scan_result/stitched \
  --min-frames 50
```

如果文件名是 `sample_001.dat`、`sample_002.dat`，第一步改为：

```bash
pam-mosaic \
  --data /data/new_scan \
  --filename-regex 'sample_(?P<index>\d+)\.dat' \
  --output /data/new_scan_result/components
```

本机新数据 `/home/gsp/PAM_brain/PAM` 的文件名是 `302.dat`～`870.dat`，使用新格式时自动识别纯数字文件名，无需重命名：

```bash
pam-mosaic \
  --input-format pam512 \
  --data /home/gsp/PAM_brain/PAM \
  --output /home/gsp/PAM_brain/outputs/pam512_full

pam-stitch-tiles \
  --components /home/gsp/PAM_brain/outputs/pam512_full \
  --output /home/gsp/PAM_brain/outputs/pam512_stitched
```

首次验证可在第一步增加 `--start 302 --stop 311`，第二步使用 `--min-frames 2`，并另选输出目录。第二步从第一步报告自动读取输入格式与强度窗口。新格式没有提供扫描批次，默认视为同一批次；如确有独立批次，应在文件名规则中提供 `scan` 组。

常用选项：

| 选项 | 作用 |
|---|---|
| `--input-format bidirectional500 / pam512` | 选择旧双向恢复流程或新 MATLAB MAP1 读取方式，默认旧格式。 |
| `--start N --stop M` | 仅处理编号 N～M 的文件，适合首次小规模检查。 |
| `--logs DIR` / `--log FILE` | 读取本项目格式的扫描日志；`--log` 可重复。 |
| `--no-frontend-recovery` | 关闭对断开相邻关系的保守重试。 |
| `--no-suppress-speckles` | 关闭轻度紧凑亮点处理。 |
| `--no-background-correction` | 关闭两种格式默认启用的逐帧背景基线校正，旧稳定版复现必须加。 |
| `--pam512-speckle-excess 0.04` | 新格式亮点相对局部中值的检测阈值，默认 0.02；0.04 可用于减少过滤的对照实验，默认不变。 |
| `--temporal-window N` | 调整短间隔候选关系范围，默认 5。 |
| `--appearance-neighbors N` | 调整外观候选数，默认 6。 |
| `pam-stitch-tiles --min-frames N` | 选择进入块间拼接的最小局部块文件数，默认 50。小样本检查可临时降到 2。 |
| `pam-stitch-tiles --no-cycle-gating` | 关闭块间闭合一致性筛选；正式流程保持默认开启。 |
| `pam-stitch-tiles --no-refine-geometry` | 关闭块间 Canny 边缘距离精化，复现此前稳定版配置；默认开启。 |

两步命令均提供 `--help`。首次可用 `--start/--stop` 处理约 10 份文件，再用 `--min-frames 2` 检查程序是否跑通。子集会重新估计校准参数，不能用来严格比较全量图像质量。

第一步支持中断后在相同数据、配置、代码和环境下复用 `.run_cache/` 的恢复帧与配准记录。源码、输入文件或数值库变化时，缓存会拒绝复用；请改用新输出目录。第二步会重新做块间配准和融合。报告中的组件路径由当前命令保存为绝对路径。

## 4. 输出与判断

第一步在 `components/` 生成 `component_001/` 等局部块及 `report.json`；小块与孤立帧也会保留。第二步在 `stitched/` 下生成 `mosaic_001/` 等大图及 `report.json`。若块之间没有可靠关系，可能得到多张大图，而不是强行拼成一张。

| 文件 | 含义 |
|---|---|
| `mosaic_float64.npy` | 浮点强度图；后续定量分析从这里读取。 |
| `weight_map.npy` | 融合权重和有效覆盖；权重大于 0 的位置有数据。 |
| `mosaic_preview.png` | 固定强度范围的彩色预览：旧格式校正后 0～0.25，关闭校正则为原 0.02～0.25；`pam512` 为 0～0.06；只用于观看。 |
| `poses_xytheta.npy` | 局部块内逐文件位姿，平移单位为像素，旋转单位为弧度。 |
| `tile_poses_xytheta.npy` | 最终大图中的局部块位姿。 |
| `report.json` | 输入编号、分量、匹配和处理决策。 |
| `.run_cache/` | 第一阶段缓存，可用于条件完全相同时继续未完成的运行。 |

全量处理文件数不等于最终入图文件数。没有可靠对应关系时，程序保留断开的局部块。相关性、边缘距离和接缝检查只能作为图像证据，不证明解剖身份；目前也没有物理像素标定，不能直接把像素换算为毫米。

### 新 PAM 数据全量验证（2026-10-08）

本机 `PAM/302.dat`～`870.dat` 共 569 帧已完成全流程。读取严格遵循 MATLAB 的列优先排列及第 512～1024 列裁剪，每帧为 512×513。当前仅保留最新背景校正结果：`outputs/pam512_background_full_20261008/`（含全部逐帧浮点缓存 `.run_cache/frame_*.npy`）及 `outputs/pam512_background_stitched_20261008/mosaic_001/`（浮点大图、权重与预览）。

该次运行使用 `--pam512-speckle-excess 0.04`，背景校正开启；569 帧全部进入一个块，2287 对候选中 2286 对有效，最终宽 1315、高 1897。逐帧估计背景约为 −0.01176～−0.00775，指定区域的弱血管可见性改善，同时背景稍亮、局部仍有软化。第二阶段只有一个块，无需额外块间配准。25 项自动测试通过，原始采集数据未修改。

按用户要求，2026-10-08 已删除此前 PAM 的试跑、阈值/亮度对照、局部诊断、重复全量结果及临时比较脚本，释放约 1.44 GB；删除清单见本机 `outputs/cleanup_pam_20261008.json`。这些文件未进入回收站，相关历史聊天中的对照图片链接已失效。旧数据的稳定输出保留。

复现该次运行：

```bash
pam-mosaic --input-format pam512 --pam512-speckle-excess 0.04 \
  --data /home/gsp/PAM_brain/PAM \
  --output /home/gsp/PAM_brain/outputs/pam512_background_full_new
pam-stitch-tiles \
  --components /home/gsp/PAM_brain/outputs/pam512_background_full_new \
  --output /home/gsp/PAM_brain/outputs/pam512_background_stitched_new
```

正式融合仍使用浮点加权平均。泊松融合或拉普拉斯多频段融合尚未接入，也未做实际效果对照；前者改变强度重建，后者主要处理不同尺度的接缝，均不能据此保证恢复细血管或修正配准误差。

### 旧双向数据背景校正全量验证（2026-10-08）

背景校正在正返处理和去亮点后执行，保留负残差，然后重新运行候选关系、帧间配准、两级位姿图及融合。旧格式校正后的配准和预览窗口使用 0～0.25；关闭校正可复现原 0.02～0.25 窗口。

- 第一阶段：`outputs/legacy_background_full_20261008/`，含全部 1301 帧校正缓存及 131 个连通块。
- 第二阶段：`outputs/legacy_background_stitched_20261008/mosaic_001/`，8 个大块、1051 帧进入 3088×3006 的最终图。
- 对比：`outputs/legacy_background_comparison_20261008/`，含全图、三处局部图和统计报告。全图保持各自原始几何，局部仅做显示用相似变换对齐。

本次 5924 对候选中 4461 对有效（含 54 条相邻补救），背景估计范围约 0.03709～0.04486，中位数 0.03923。原配置为 125 个局部块，7 个大块、1083 帧最终入图。新配置增加 1014 号帧，但 415～444、448、589、1055 共 33 帧未进入达到 50 帧门槛的大块，净减少 32 帧；它们仍保存在第一阶段局部块中。预览背景更干净，部分弱信号也更暗，尚未证明细血管整体更清楚或拼接更准确，因此原稳定版不被替换。25 项自动测试通过。

```bash
pam-mosaic --input-format bidirectional500 \
  --data /home/gsp/PAM_brain/PAM_Skull_FYH_20260929 \
  --output /home/gsp/PAM_brain/outputs/legacy_background_full_new
pam-stitch-tiles \
  --components /home/gsp/PAM_brain/outputs/legacy_background_full_new \
  --output /home/gsp/PAM_brain/outputs/legacy_background_stitched_new
```

## 5. 仓库结构：每个保留文件的用途

```text
README.md                         当前使用说明、格式与仓库结构
STABLE_RELEASE.md                 本数据集正式结果、已知限制和校验值
pyproject.toml                    依赖、打包配置与两个命令入口
.gitignore                        排除原始数据、输出、缓存及本机状态
src/pam_mosaic/
  __init__.py                      Python 包标记
  cli.py                          第一阶段命令行入口，自动识别读取适配器
  checkpoint.py                    输入/配置/代码绑定的恢复缓存
  pipeline.py                      第一阶段整体流程和报告
  preprocessing/                  数据预处理：读取、恢复、去亮点、背景校正
    __init__.py                   子包标记
    inputs.py                     自动发现输入格式及通用 InputState 接口
    dataset.py                    输入文件编号、扫描批次和日志索引
    reconstruction.py             旧双向 .dat 读取、拆分和扫描轨迹校正
    roundtrip.py                  旧正返内部校准、验证、融合或择优
    denoise.py                    通用紧凑亮点处理
    background.py                 通用背景众数估计、基线扣除及负值保留
    formats/                      每种采集格式一个适配文件，新数据在此添加
      __init__.py                 适配器目录标记
      bidirectional500.py         旧数据读取适配，组织 reconstruction 和 roundtrip
      pam512.py                   新 PAM 格式读取、MATLAB MAP1 裁剪与温和去亮点
  frontend/                       前端：测量一对图像之间的相对变换
    __init__.py                   子包标记
    view.py                       配准视图窗口映射，不修改科学浮点强度
    candidates.py                 时间邻近和外观相似候选帧对
    registration.py               SIFT/RANSAC、点到点/边缘距离 LM、相邻恢复
  backend/                        后端：利用多条关系优化全局摆放并融合
    __init__.py                   子包标记
    graph.py                      由可信关系求帧/块连通组
    poses.py                      SE(2) 初始位姿传递与鲁棒位姿图优化
    fusion.py                     逐帧/块级浮点加权融合及预览
    tile_stitch.py                第二阶段块间关系、闭环筛选、优化、融合和入口
tests/
  test_dataset.py                  文件名适配、编号和路径检查
  test_reconstruction.py           MATLAB 数据读取与轨迹校正检查
  test_pam512.py                   新格式头部、列优先排列、513 列裁剪及强度视图检查
  test_roundtrip.py                正返融合行为检查
  test_denoise.py                  亮点处理检查
  test_background.py               背景偏移校正、强度差及负值保留检查
  test_registration.py             配准与恢复检查
  test_poses.py                    位姿图优化检查
  test_fusion.py                   融合输出检查
  test_tile_stitch.py              块间一致性筛选检查
  test_checkpoint.py               缓存身份与续跑检查
  test_inputs.py                  输入适配器发现、新格式复用通用后续流程检查
docs/
  history/HANDOFF_20261006.md     本数据集历史排查交接记录
  experiments/IMPROVEMENT_PLAN_20261006.md  历史改进计划
```

两种输入的背景校正位置一致：`读取适配器完成单帧读取/恢复和去亮点 → 通用背景校正 → 前端帧间配准 → 后端连通组位姿优化与融合`。旧数据在读取适配器内部先完成正返扫描处理；块级阶段复用前端配准函数测量块间变换，再由后端优化，背景不会重复扣除。根目录只保留入口、流程组织和缓存基础设施，没有重复的算法实现。

2026-10-08 重组验证：28 项自动测试通过；两种数据各 10 帧的两阶段小样本流程均完成，逐帧预处理缓存与重组前对应全量结果逐元素一致，最大绝对差为 0。既有完整大图和原始数据未重新写入。

### 添加一组新采集格式

在 `src/pam_mosaic/preprocessing/formats/` 新建一个 `.py` 文件，参照现有适配器提供以下接口。`inputs.py` 自动发现它，CLI 的 `--input-format` 会自动增加选项，不需要改前端、后端或 `pipeline.py`。

| 成员 | 用途 |
|---|---|
| `FORMAT_NAME` | 命令行格式名称，必须唯一。 |
| `FILENAME_REGEX` | 带 `(?P<index>...)` 的编号规则，可选扫描批次 `scan` 组。 |
| `FILE_GLOB`（可选） | 文件筛选，例如 `*.npy`；默认 `*.dat`。 |
| `DISPLAY_RANGE` | 扣除背景后用于配准和预览的强度窗口。 |
| `UNCORRECTED_DISPLAY_RANGE`（可选） | 关闭背景校正时的窗口；默认与上项一致。 |
| `prepare(paths)` | 准备读取状态，返回 `InputState()`；需校准时通过 `context` 保存状态，通过 `cache_config` 保存可序列化身份信息。 |
| `options(suppress_speckles, speckle_excess)` | 返回读取/去亮点配置字典，参与缓存身份检查；无特殊处理可返回 `{}`。 |
| `load_frame(path, state, configuration)` | 返回二维浮点单帧及处理记录字典，不在这里重复做通用背景校正。 |

最小示例（已恢复为二维 `.npy` 的输入）：

```python
import numpy as np
from ..inputs import InputState

FORMAT_NAME = "npy_frames"
FILENAME_REGEX = r"frame_(?P<index>\d+)\.npy"
FILE_GLOB = "*.npy"
DISPLAY_RANGE = (0.0, 0.06)  # 按实际数据确定，不要盲用示例范围

def prepare(paths):
    return InputState()

def options(suppress_speckles, speckle_excess):
    return {}

def load_frame(path, state, configuration):
    image = np.load(path, allow_pickle=False).astype(np.float32)
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError(f"Invalid frame: {path}")
    return image, {"selected": "npy", "changed_pixels": 0}
```

选择新格式即可：`pam-mosaic --input-format npy_frames --data /data/frames --output /data/result/components`，第二阶段的命令不变。每帧尺寸、方向和强度应一致；不同扫描布局需要适配器先恢复成一致的二维图像。目录重组改变了源码身份，原有完成结果仍能查看及用于第二阶段，但不能直接复用重组前的未完成第一阶段缓存。

原始数据目录和 `outputs/` 被 `.gitignore` 排除，不会随代码上传。`outputs/frontend_only_*` 是本机稳定结果；`outputs/lm_*` 是历史基线；`outputs/archived_experiments_20261006/code/` 保存移出工作代码的比较和实验脚本，仅供本机追溯。换机器时需自行提供原始数据和选择新的输出路径。

2026-10-07 已清理本机输出：保留当前配置与旧稳定版的完整最终数值图、全量局部块，以及 `outputs/stable_rerun_20261006_full/.run_cache/` 中的一份完整恢复缓存；重复重跑大图、旧基线和小规模试验的数值数组已删除，对比图片、报告及归档源码保留。精简后的历史目录不再是完整可续跑结果。具体保留内容见本机 `outputs/README.md`，删除清单见 `outputs/cleanup_20261007.json`。
