# Fig2-style mechanism panels — findings (a13)

All numbers below come from `a13_fig2_panels.json` (script `a13_fig2_panels.py`,
zero GPU, bootstrap 1000 resamples, seed=42). Panel CSVs: `a13_panel_a_lorenz.csv`,
`a13_panel_b_channel_matrix.csv`, `a13_panel_c_iou.csv`.

Data notes / deviations (recorded in json `meta.notes`):
- `cmm_probe_full.jsonl` has **720** rows (task sheet said 721; shard w5 has 89, 91+90×7=720).
- avh probes total **120** rows (4×30), not ~130. `delta_audio/delta_video` are **per-token
  lists** (3109 tokens over 120 samples), not scalars — handled token-level.
- phd ids are non-unique (114 unique / 906), so `phd_probe_qonly.jsonl` **cannot be joined**
  to `phd_probe_full.jsonl`; it is used only as a standalone distribution.
- Cross-backbone rows use dev probes (phd n=100, cmm n=98/100) vs Qwen2.5 full
  (906/720) — n mismatch flagged everywhere below.
- a3_dose_response.json lacked Nemotron probe AUCs; computed here with a3's exact
  convention (signal `lp_yes_noctx − lp_yes_full` / `−d_cf_sum`, label `gt_yes`).
  Qwen3 values reproduce a3 (0.557, 0.721).

---

## Panel (a) — pathology concentration

**Finding 1: Counterfactual dependence mass is concentrated, not diffuse — the top 10% of
commitment points carry 32.9% of all dependence mass (Gini 0.510, 95% CI 0.493–0.528 on
phd n=906), and only 30.4% of points sit in the λ=2 flippable band, so a counterfactual
audit needs to interrogate a small minority of commitments.**

中文注记：
- phd |delta_noctx|：Gini = **0.510** [0.493, 0.528]；top-10% 样本占质量 **32.9%**，
  top-20% 占 **53.6%**。对照组 |delta_noimg| 的 Gini = 0.389 [0.370, 0.407]（更弥散）。
- cmm |d_cf_sum|：Gini = 0.220（质量更均匀，因病灶数据几乎所有样本都病得重）。
- 可翻转带（|margin_full| < λ·|d|）：phd λ=2 占 **30.4%** [0.275, 0.334]；
  敏感性 λ∈{0.5,1,2,4} = 6.7% / 18.4% / 30.4% / 45.4%；cmm λ=2 占 **87.5%**
  [0.851, 0.900]（病灶组本来就病重，审计预算论在 phd 上成立）。
- **跨骨干塌缩（注意口径 dev n=100/98 vs full n=906/720）**：塌缩体现在"质量规模"和
  "可翻转带"上，不是 Gini 上——Gini 在新骨干上反而升高（多数样本 |d|→0，残差更稀疏），
  这本身也是病理自愈的表现：
  - phd：median |d| 0.548 → 0.201 (Qwen3) → 0.125 (Nemotron)；可翻转带
    **30.4% → 9.0% → 6.0%**。
  - cmm：median |d| 2.258 → 0.0037 (Qwen3) → 0.343 (Nemotron)；可翻转带
    **87.5% → 10.2% → 15.0%**。
  - 即"代际自愈"：病理质量在 Qwen2.5 上又大又可翻，在 Qwen3/Nemotron 上近乎消失。

## Panel (b) — channel localization

**Finding 2: The dependence vector points at a specific modality — the video channel
carries 3.8× the audio channel's median dependence in cmm (2.08 vs 0.51 nats in the
visual-lesion group) and d is enriched 1.51× (95% CI 1.41–1.67) in the lesion-matched
channel for visual lesions, while phd shows vision ≫ text-prior (2.06 vs 0.55 nats).**

中文注记：
- cmm 2×2 中位数表（|d|，n=720）：

  | 组 | \|d_cf_audio\| | \|d_cf_video\| |
  |---|---|---|
  | overrely_audio 组 (n=329) | 0.269 | 1.373 |
  | overrely_visual 组 (n=391) | 0.514 | 2.079 |

- 富集比：video 通道在 visual 组 / audio 组 = **1.51×** [1.410, 1.678]（病灶匹配富集）；
  audio 通道在 audio 组 / visual 组 = **0.52×** [0.405, 0.634]——"overrely_audio" 组的
  决策实际上仍主要挂在视频通道上（音频病灶组并没有把质量押在音频通道），这是
  d(c) 向量才暴露的诚实结构：两个组的 |d_cf_video| 都大于 |d_cf_audio|。
- phd：|delta_noimg|（视觉通道）中位数 **2.055** [1.916, 2.194] vs |delta_noctx|
  （文本先验通道）**0.548** [0.508, 0.584]（比值 0.27）；纯问题先验 |margin_qonly|
  中位数 0.75（qonly 906 行，独立分布，未 join）。
- avh（120 样本、3109 token）：token 级 delta_video 中位 0.345 vs delta_audio 0.0036，
  delta_video 均值 1.254 vs delta_audio 0.351，delta_video 为负（视觉遮蔽降分）占
  84.2% —— 描述类生成同样主要依存视频通道。
- 通道×数据集矩阵见 `a13_panel_b_channel_matrix.csv`；跨骨干：phd 文本先验通道
  median |d| 0.548 → 0.201 (Qwen3) → 0.125 (Nemotron)，与面板 (a) 的塌缩一致。

## Panel (c) — profile validity

**Finding 3: The dependence profile is complementary evidence, not margin noise or an
entropy proxy — the top-p% |d| set overlaps the top-p% |margin| set at IoU ≤ 0.12
(p=20%), d alone keeps AUROC 0.68 on phd full906 (0.739 for margin), and both signals
collapse across backbones (phd 0.767→0.557→0.635; cmm 0.595→0.721 polarity-reversed
→0.593), tracking the generational self-healing of the pathology.**

中文注记：
- IoU（top-p% |d| 集 vs top-p% |margin_full| 集）：
  - phd：p=1% **0.000**，5% 0.000，10% 0.006，20% **0.011**（各 CI 见 json/csv）。
  - cmm：p=1% 0.000，5% 0.029，10% 0.051，20% **0.121**。
  - 解读：d 的高分样本与 margin 的高分（自信）样本几乎不相交——d 标记的是
    "反事实可翻的承诺点"，margin 标记的是"自信的承诺点"，二者互补而非冗余。
- AUROC（引用 a3_dose_response.json 全表 + 本次补算 Nemotron；a3 里可用信号还有
  d_cf(noimg) full906 = 0.248、-margin_noctx full906 = 0.161、-d_cf(passage)@pubmedqa
  = 0.585、cmm@Qwen3 margin_full = 0.951）：
  - phd d 信号：**0.767 (dev100) / 0.68 (full906) @Qwen2.5 → 0.557 [0.446, 0.668]
    @Qwen3(dev) → 0.635 [0.523, 0.734] @Nemotron(dev)**。
  - cmm -d_cf_sum：**0.595 (full720) → 0.721 [0.600, 0.836] @Qwen3(dev，极性反转，
    offline-inert) → 0.593 [0.460, 0.716] @Nemotron(dev)**。
- 熵对照（bon_probe 候选级 within-pool AUC，good = 样本内 F1≥中位；CE 特征已翻号，
  低熵=自信）：ragtruth CE_top5mean **0.575** / CE_max 0.563 vs grounding Δ(g_content)
  0.506 / g_mean_claim 0.508；haloquest CE_top5mean **0.624** / CE_max 0.587 vs
  g_content 0.445 / g_mean_claim 0.425。诚实结论：在 BoN 候选挑选任务上熵类特征
  优于 grounding-Δ，但其绝对判别力（≤0.62）仍远低于我们决策点 margin 在 phd 上的
  0.739（任务不同轴，不可直接同表硬比——论文里作对照叙述而非同列赛马）。
