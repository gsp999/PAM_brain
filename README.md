# 二维光声血管图像拼接

将光声 `.dat` 数据恢复为二维浮点帧并拼接。当前确认配置为全帧位姿图＋几何、强度、回环因子＋加权融合；算法原理、流程图和优化表达式见 [算法说明](docs/ALGORITHM.md)。

## 安装

需要 Python 3.10+。本机复现环境为 `/home/gsp/miniconda3/envs/vision/bin/python`（Python 3.12.13）。在仓库根目录、已激活的 Python 环境中执行：

```bash
python -m pip install numpy==2.4.4 scipy==1.17.1 opencv-python==4.13.0.92
python -m pip install -e .
python -m unittest discover -s tests -q

export OPENBLAS_NUM_THREADS=1
export OPENCV_FOR_THREADS_NUM=4
```

固定库版本与线程数可复现本机运行配置。安装后提供 `pam-mosaic` 和 `pam-stitch-tiles` 两个命令；未安装时可分别用 `PYTHONPATH=src python -m pam_mosaic.cli`、`PYTHONPATH=src python -m pam_mosaic.backend.tile_stitch` 替代。

## 复现当前配置

每次运行使用新的输出目录；以下命令输出到 `outputs/legacy_new_*` 和 `outputs/pam512_new_*`，不会覆盖已保留结果。

旧双向数据：**关闭背景校正，开启全部后端因子**。

```bash
pam-mosaic --input-format bidirectional500 --data ./PAM_Skull_FYH_20260929 \
  --no-background-correction --backend-factors all --output ./outputs/legacy_new_full
pam-stitch-tiles --components ./outputs/legacy_new_full --output ./outputs/legacy_new_stitched
```

512 数据：**开启背景校正、亮点阈值 0.04、开启全部后端因子**。

```bash
pam-mosaic --input-format pam512 --data ./PAM --pam512-speckle-excess 0.04 \
  --background-correction --backend-factors all --output ./outputs/pam512_new_full
pam-stitch-tiles --components ./outputs/pam512_new_full --output ./outputs/pam512_new_stitched
```

其余参数使用现有默认值。第二阶段从第一阶段报告读取强度窗口，不重复扣背景；只有一个局部块时仍可直接运行第二步。CLI 的 `--backend-factors` 默认是 `none`，**复现当前确认方案必须显式指定 `all`**。

## 内窥镜 throat 数据

按 `throat/rec.m` 恢复 256×256 螺旋扫描图像：小端 double、去掉末尾值、每 142 点用第二点替换首点、第一段扫描做 6 倍线性插值、同一像素保留最后写入值、零填充 2×2 中值滤波。每个文件含两段扫描，遵循原脚本 `for i = 1` 仅使用第一段；处理全部 houlong 编号。显示窗口沿用 MATLAB 的 0–0.1，不额外应用 PAM512 亮点滤波。

```bash
pam-mosaic --input-format throat --data ./throat \
  --background-correction --backend-factors all --output ./outputs/throat_full_20261016
pam-stitch-tiles --components ./outputs/throat_full_20261016 \
  --min-frames 1 --output ./outputs/throat_stitched_20261016
```

## 输入与新数据复用

| 格式 | 数据布局 | 默认文件名 |
|---|---|---|
| `bidirectional500` | 8 字节头＋500×1000 小端 float64，MATLAB 列优先；恢复为 500×500 单帧 | `1_Col2_tdT_58_scan1.dat` |
| `pam512` | 两个小端 int32 头值 (1024, 512)＋512×1024 小端 float64，MATLAB 列优先；按 `MAP(:,512:1024)` 输出 512×513 | `302.dat` |

文件名只负责确定采集顺序，编号必须唯一。其他命名可以通过正则适配，必须包含 `index` 命名组，可选 `scan` 组：

```bash
pam-mosaic --input-format pam512 --data /data/new_scan \
  --filename-regex 'sample_(?P<index>\d+)\.dat' \
  --background-correction --pam512-speckle-excess 0.04 --backend-factors all \
  --output /data/new_scan_result/components
pam-stitch-tiles --components /data/new_scan_result/components --output /data/new_scan_result/stitched
```

若矩阵尺寸、头部、数据类型或扫描布局不同，在 `src/pam_mosaic/preprocessing/formats/` 添加一个适配器；复制现有适配器并实现以下接口，前端和后端无需修改：

| 成员 | 用途 |
|---|---|
| `FORMAT_NAME`、`FILENAME_REGEX` | 格式名称与带编号的文件名规则 |
| `FILE_GLOB`（可选） | 文件筛选，默认 `*.dat` |
| `DISPLAY_RANGE`、`UNCORRECTED_DISPLAY_RANGE`（可选） | 开启、关闭背景校正时的配准与预览窗口 |
| `prepare(paths)` | 返回 `InputState`，保存读取状态和校准信息 |
| `options(suppress_speckles, speckle_excess)` | 返回读取和去亮点配置 |
| `load_frame(path, state, configuration)` | 返回有限的二维浮点帧与处理记录 |

新格式在读取适配器内恢复为方向、尺寸一致的二维图像，通用背景校正在读取后统一执行。先检查少量单帧；可以加 `--start N --stop M`，并在第二阶段用 `--min-frames 2` 做小样本检查。

## 参数与输出

两步命令均提供 `--help`。常用开关：

| 参数 | 用途 |
|---|---|
| `--backend-factors all / none` | 开启全部因子或仅运行原位姿图；也可指定逗号分隔的 `geometry,intensity,loop` 子集 |
| `--background-correction / --no-background-correction` | 开启或关闭逐帧背景归零，CLI 默认开启 |
| `--pam512-speckle-excess`、`--no-suppress-speckles` | 调整 PAM 亮点阈值或关闭去亮点 |
| `--start N --stop M`、`--filename-regex` | 选择编号范围、适配文件名 |
| `--logs DIR / --log FILE` | 可选扫描日志；时间戳不参与配准 |
| `--temporal-window`、`--appearance-neighbors` | 前端候选范围，默认 5、6 |
| `pam-stitch-tiles --min-frames` | 进入块级阶段的最小帧数，默认 50 |

当前开启因子后固定使用世界坐标局部配准，最终将前端边与附加因子共同进行 SE(2) Log 位姿图优化。实验参数 `--backend-style` 已移除，直接使用 `--backend-factors all`；`none` 仍可关闭附加后端。

`full/` 保存 `component_001/` 等局部块；`stitched/` 保存 `mosaic_001/` 等最终图。未可靠连通的组分别保留，因此最终可能不止一张图。

| 文件 | 用途 |
|---|---|
| `mosaic_float64.npy` | 浮点强度图，用于后续分析，数值不是固定 0～255 |
| `weight_map.npy` | 融合权重；大于 0 表示有效覆盖 |
| `mosaic_preview.png`、`preview_window.json` | HOT 预览及显示窗口，预览不替代浮点数据 |
| `poses_xytheta.npy`、`tile_poses_xytheta.npy` | 帧位姿、块位姿；平移为像素，角度为弧度 |
| `poses_baseline_xytheta.npy`、`backend_factors.json` | 加因子前的位姿、附加测量与文件编号映射 |
| `report.json` | 数据、处理配置、配准、连通组和来源报告 |
| `.run_cache/frame_XXXX.npy`、`.run_cache/frames.json` | 每帧实际用于拼接的预处理浮点图与原文件编号映射 |

第一阶段支持在相同代码、输入、配置和环境下续跑。改变任何一项需使用新目录，不能直接复用旧检查点。完整结果可以继续用于第二阶段；程序拒绝覆盖已有 `report.json` 的完成目录。开启背景校正时缓存已扣基线，不能再次扣除。

## 本机保留结果

| 数据 | 局部块与逐帧缓存 | 最终大图 | 入图情况 |
|---|---|---|---|
| 旧数据 | `outputs/legacy_allframe_factors_full_20261010/` | `outputs/legacy_allframe_factors_stitched_20261010/mosaic_001/` | 1301 帧、125 局部块；7 大块、1083 帧，3068×3006 |
| 512（当前） | `outputs/pam512_world_augmented_full_20261019/` | `outputs/pam512_world_augmented_stitched_20261019/mosaic_001/` | 569 帧全部入图，1298×1876 |

当前 512 主流程实测 **257.44 秒（4 分 17.44 秒）**，记录在 `outputs/pam512_world_augmented_full_20261019/comparison/summary.json`。旧数据表中结果仍为此前版本，未用当前后端重跑。历史 512 结果和对照输出保留；`outputs/current_results.json` 为 10 月 12 日的归档清单，不代表本次版本。

原始数据和 `outputs/` 不随 Git 上传，换机器需另行备份。

## 仓库结构

```text
README.md                         安装、复现、输入输出与仓库结构
docs/ALGORITHM.md                 当前算法流程、因子定义与优化目标
pyproject.toml                    依赖、打包配置和两个命令入口
.gitignore                        排除数据、输出、构建及本机缓存
src/pam_mosaic/
  __init__.py                     包标记
  cli.py                          第一阶段命令行入口
  pipeline.py                     预处理、前端和全帧后端的流程组织
  checkpoint.py                   数据/配置/代码绑定的续跑缓存
  preprocessing/
    inputs.py                     输入适配器发现与 InputState 接口
    dataset.py                    文件编号、扫描批次、日志索引
    reconstruction.py             旧双向数据读取与扫描轨迹恢复
    roundtrip.py                  旧正返校准、验证与融合/择优
    denoise.py                    紧凑亮点去除
    background.py                 可选背景归零，保留浮点负值
    formats/
      bidirectional500.py         旧双向读取适配器
      pam512.py                   MATLAB MAP1 读取适配器
      throat.py                   throat 数据读取适配器
  frontend/
    view.py                       配准视图与强度窗口映射
    candidates.py                 时间及外观候选关系
    registration.py               SIFT/RANSAC、两次 LM 与相邻恢复
  backend/
    graph.py                      可信关系的连通组
    poses.py                      SE(2) 位姿传递和鲁棒位姿图 LM
    allframe.py                   全帧附加因子构建、筛选与联合优化
    factors.py                    局部地图几何、多视图 SIFT 测量
    fusion.py                     帧/块加权融合与预览
    tile_stitch.py                第二阶段块配准、优化、融合与入口
tests/
  test_dataset.py                 文件名与输入索引
  test_inputs.py                  适配器发现及通用流程复用
  test_reconstruction.py          旧数据布局与轨迹恢复
  test_pam512.py                  PAM 数据头、布局、513 列裁剪
  test_throat.py                  throat 输入读取
  test_roundtrip.py               正返验证与择优
  test_denoise.py                 亮点处理
  test_background.py             背景基线与负值保留
  test_registration.py           相对配准与失败恢复
  test_poses.py                  位姿图与雅可比
  test_allframe.py               全帧因子、原关系保留与回环筛选
  test_world_factors.py          世界坐标配准与参考帧转换一致性
  test_fusion.py                 权重与无数据区域处理
  test_tile_stitch.py            块级配准和闭合一致性筛选
  test_checkpoint.py             缓存身份与续跑
outputs/                          当前结果、历史对照及归档校验清单
```

各子目录的 `__init__.py` 为包标记。测试覆盖当前支持的功能；本轮实验后端分支已移除，已有结果与对照图保留。
