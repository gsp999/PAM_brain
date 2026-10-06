# 当前选用的稳定版

冻结日期：2026-10-06。

## 正式结果

- 全量局部块：`outputs/frontend_only_full/`
- 最终拼接大图：`outputs/frontend_only_stitched/mosaic_001/`
- 预览：`outputs/frontend_only_stitched/mosaic_001/mosaic_preview.png`
- 科学浮点图：`outputs/frontend_only_stitched/mosaic_001/mosaic_float64.npy`
- 融合权重：`outputs/frontend_only_stitched/mosaic_001/weight_map.npy`
- 逐帧位姿：各 `outputs/frontend_only_full/component_*/poses_xytheta.npy`
- 块级位姿：`outputs/frontend_only_stitched/mosaic_001/tile_poses_xytheta.npy`
- 运行报告：`outputs/frontend_only_full/report.json` 与 `outputs/frontend_only_stitched/report.json`

## 版本结论

稳定版处理 1301 份数据，7 个大块组成 3050×3010 大图，共纳入 1083 份数据。主要血管网络和全局布局可靠；少数跨扫描重叠存在局部双线或不一致，不适合做跨接缝的精细血管宽度及拓扑定量。

### 整理代码后的全量复现（2026-10-06）

- 新运行的局部块：`outputs/stable_rerun_20261006_full/`
- 新运行的大图：`outputs/stable_rerun_20261006_stitched/mosaic_001/`
- 重新处理 1301 份文件、5940 对候选关系，31 条相邻关系恢复，形成 125 个局部分量；7 个块、1083 份文件进入一张 3050×3010 大图。
- 与上方正式结果相比：4570 条有效前端关系、逐文件处理决策、各局部分量的文件归属和块间关系完全一致；最终 `mosaic_float64.npy`、`weight_map.npy`、`tile_poses_xytheta.npy` 逐元素完全相同，最大绝对差为 0，有效像素均为 5,960,196。
- 新运行只是复现验证；原正式结果仍保留在原路径。

已经验证但未采用的方案包括：受限距离场、方向与外观筛选、单次投影融合、异常边删减、整块平移、小块强行补接和局部跨扫描刚体修正。它们均未显示稳定的整体收益。

### 块间边缘精化试验（2026-10-06）

当前代码的块间配准默认也启用与帧间相同的 Canny 边缘距离场精化；旧稳定版行为可用 `pam-stitch-tiles --no-refine-geometry` 复现。只重跑受改动影响的块间阶段，输入为上述全量复现的局部块，试验输出位于 `outputs/tile_edge_refinement_20261006/`；关闭精化的对照输出位于 `outputs/tile_no_edge_control_20261006/`，其最终浮点图、权重图、块位姿与旧稳定版逐元素相同。7 块、1083 文件及 3050×3010 画布不变；相对旧版，块位姿最大平移分量变化约 0.69 像素，最大旋转变化约 0.00139 弧度。共同覆盖区域的浮点强度平均绝对差约 0.000745。11 条有效块间匹配的采样边缘距离均略降，但 SIFT 内点中位误差的整体中位数由约 1.321 升至 1.369 像素。尚无证据表明最终解剖拼接质量稳定提升，因此**试验结果未取代本文件列出的正式稳定版**。

## 使用原则

1. 不覆盖上述正式结果；新配置在独立目录验证后再决定是否升级正式版本。
2. 后续实验必须使用独立输出目录，并同时检查目标区域和留出区域。
3. `outputs/lm_full/` 与 `outputs/lm_stitched/` 仅作为历史基线。
4. `outputs/seam_localization/` 是稳定版的只读接缝诊断。
5. `outputs/archived_experiments_20261006/` 中内容均为诊断或未采用试验，不得误作正式结果。

## 复现

运行命令、环境、缓存规则和输出格式见 `README.md`。复现本文件所列旧稳定版块级处理时需加 `--no-refine-geometry`。已知问题和验证过程见 `docs/history/HANDOFF_20261006.md`。

## 关键文件 SHA-256

用于确认正式结果没有被后续试验覆盖：

| 文件 | SHA-256 |
|---|---|
| `mosaic_float64.npy` | `673063a0a02c767c343c7fb876cf0c3afe5bdbb017f61668d150a7710edebcf9` |
| `weight_map.npy` | `75d2182a76ed86a5e38addefe9a7900a1e8b661faad71d6fa1b626fdbf55a1e4` |
| `tile_poses_xytheta.npy` | `d48d18ddd833c55e8a77b5b250e45cf1d304000a6dba27204aa18a9d9b54a9f1` |
| `frontend_only_stitched/report.json` | `c26e331e227ab43400bd3f9b456014cfaa2cec859c22b8eb852aa3e587544ba5` |
| `frontend_only_full/report.json` | `40fec7fcbca906980619370333b1ad1c86538d38e1b389d4b4d7f965d7bda269` |
