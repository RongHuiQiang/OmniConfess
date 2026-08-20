# RENDER_NOTES.md — fig_mechanism 渲染记录（2026-08-18，零 GPU）

产物：`fig_mechanism.pdf` / `fig_mechanism.png`（300dpi，2×2），脚本
`render_fig.py`（可重跑：`vllm env python render_fig.py`）、补算脚本
`compute_dev_stats.py`、自检脚本 `selfcheck.py`。图内所有数字由脚本从
`a13_fig2_panels.json` / 面板 CSV / 探针 jsonl 程序化读取，无手写数字。

## 渲染时定的两处决策

1. **(b) 可翻转带：做成 inset（方案 1）**。Lorenz 主体只有两条曲线 + 对角线，
   右下留白充足；inset（band% vs λ∈{0.5,1,2,4}，带 95% CI errorbar，
   log2 轴）放下后不遮挡曲线，且直接可视化"审计强度↔覆盖范围"，与 Eq.(4)
   互指。λ=2 值同时标在 inset 图例里（phd 30.4% / cmm 87.5%）。
2. **(a) FP/FN 符号小条：不加**。四条 ECDF 曲线 + 两组 AUROC 标注已占满
   格面；设计稿注明"不要也成立"，Nemotron −0.974 留给正文 §4.4 叙述。

## (d) Qwen2.5 dev 口径的关键发现（偏离设计稿字面、不偏离其意图）

设计稿说"按 dev100_ids.json 过滤全量探针（一行过滤）"。实测**不可行**：
`dev100_ids.json` 的 100 个 id 是 `OmniHalluBench/phd/phd.json` 的 uid 数字
后缀（数据集行号 0–905），而探针文件的 `id` 字段是另一套空间（phd 探针 =
scored 文件的组 id 0–113，与 dev100 只交 17 个；cmm 探针 = shard 内局部
id）。按它过滤得到的是 136 行/17 题（phd）、104 行/13 题（cmm）的残片，
不是 dev100 子样本。

真正的同口径样本已存在：`phd_probe.jsonl`（n=100，gt×correct 四格各 25，
seed 42）与 `cmm_probe.jsonl`（n=87，25/25/24/13），由与 Qwen3（100/98）、
Nemotron（100/100）dev 探针**同一套分层协议脚本**生成；a3 自己就把它引作
"dev100"。故 (d) 的 Qwen2.5 点用它们（phd band 32.0%、median|d| 0.570；
cmm band 86.2%、median|d| 2.332），dev100_ids 交集残片作为被否弃的替代
口径一并写进 json（`cross_backbone_dev.alternative_dev100ids_intersection_rejected`，
其值 phd 33.8%/0.541、cmm 86.5%/2.169，与主口径几乎一致，结论稳健）。

json 追加节点：`panel_a_concentration.cross_backbone_dev`（未覆盖任何既有键）。

## 自检对照表（selfcheck.py 输出；图上数字 vs json/csv 来源）

| 图上项 | 图上值 | 来源 |
|---|---|---|
| (a) phd AUROC | 0.680 [0.638, 0.720] | auroc_cited_from_a3 (a3) |
| (a) phd n wrong/correct | 301/605 | probe_full.jsonl base_correct |
| (a) cmm AUROC | 0.595 [0.553, 0.634] | auroc_cited_from_a3 (a3) |
| (a) cmm n wrong/correct | 246/474 | probe_full.jsonl base_correct |
| (b) phd Gini / top-10% | 0.510 [0.493, 0.528] / 32.9% | lorenz_gini |
| (b) cmm Gini / top-10% | 0.220 [0.208, 0.233] / 17.0% | lorenz_gini |
| (b) inset phd band% λ=.5/1/2/4 | 6.7/18.4/30.4/45.4 % | flippable_band |
| (b) inset cmm band% λ=.5/1/2/4 | 33.5/56.4/87.5/98.6 % | flippable_band |
| (c) phd IoU@20% | 0.011 [0.003, 0.028] | a13_panel_c_iou.csv |
| (c) cmm IoU@20% | 0.121 [0.083, 0.161] | a13_panel_c_iou.csv |
| (d) phd band% Q2.5/Q3/Nemo | 32.0/9.0/6.0 | cross_backbone_dev + cross_backbone |
| (d) phd median\|d\| | 0.570/0.201/0.125 | 同上 |
| (d) phd n | 100/100/100 | 同上 |
| (d) cmm band% Q2.5/Q3/Nemo | 86.2/10.2/15.0 | 同上 |
| (d) cmm median\|d\| | 2.33/0.00365/0.343 | 同上 |
| (d) cmm n | 87/98/100 | 同上 |

对照结论：**全部一致**。与设计稿唯一按预期不同的数字是 (d) 的 Qwen2.5
两点（band 32.0/86.2 vs 设计稿 full 值 30.4/87.5；median 0.570/2.33 vs
0.548/2.258）——这正是"dev 口径替换 full 口径"的目的，full 值仍在 json
`cross_backbone.qwen2.5_full` 供附录稳健性注使用。

## 其他实现注记

- (a) x 轴 log 尺度（|d| 跨两个量级）；|d|=0 的点在 log 轴上不可见，其占比
  由 caption/n 说明（phd 5/906 个 0 值、cmm 0 个）。
- 色盲安全：数据集=颜色（phd #0072B2 蓝、cmm #E69F00 橙，Okabe-Ito）；
  正确/错误=线型（实/虚）；(d) 指标=线型（band 实线、median|d| 点线）。
- (c) 随机期望虚线 y=p 与两线均远分离（正交证据）。
- 重跑顺序：`compute_dev_stats.py`（若 json 尚无 cross_backbone_dev）→
  `render_fig.py` → `selfcheck.py`。
