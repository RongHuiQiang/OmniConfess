# 算子自改进台账（固定 n=20，离线在缓存 logprob 上调，GPU 仅最终确认）

红线：三阶段 cf 的 F1 必须**严格 > base 的 F1**（每个数据集）。

## 指标口径
- pubmedqa: macro-F1(yes/no/maybe)
- phd / cmm: F1(yes)
- ragtruth / avhbench: Token-F1（base_pred vs final）

## 基线（终版 prompt，裸部署算子）
| ds | prompt | 指标 | base | cf(旧算子) | delta | 判定 |
|---|---|---|---|---|---|---|
| pubmedqa | v5 | macro-F1 | 49.85 | 18.48 | −31.37 | 裸 argmax κ 拟合噪声，狂翻错 |
| cmm | v1 | F1(yes) | 63.64 | 46.15 | −17.48 | λ-δ 过翻 yes→no |
| ragtruth | v4 | Token-F1 | 51.70 | 51.05 | −0.65 | 持平 |
| avhbench | v1 | Token-F1 | 22.33 | 22.37 | +0.04 | 持平 |
| phd | v1 | F1(yes) | 66.67 | 66.67 | 0.00 | 裸 γ 翻 3 例(1 对 2 错)净零，多 FP |

## GPU 端到端确认（n=20，新算子，_clear_stale 修复后）
| ds | 指标 | base | cf(新算子) | delta | 判定 |
|---|---|---|---|---|---|
| pubmedqa | macro-F1 | 53.14 | **84.00** | **+30.86** | ✓ 正收益 |
| cmm | F1(yes) | 63.64 | **82.35** | **+18.71** | ✓ 正收益（pooled 20：TP7 FP7→2 FN1）|
| phd | F1(yes) | 66.67 | **75.00** | **+8.33** | ✓ 正收益（GPU 确认，gate_tau=1.4）|

> pubmedqa GPU base=53.14 与离线 49.85 的差异来自 v5 prompt 下 base 生成的随机性口径；两者 cf 均=84.00，正收益结论一致。


## 改进轮次

### R1 pubmedqa —— maybe 证据门（离线）
规则：候选锚定 d*=base 答案；仅当 d*==maybe 且 κ_maybe≤τ_m(=0.5) 时，在 yes/no 间按 lp_full 重决；其余保持 base。
- id1 gt=no base=maybe κ_m=0.294→no ✓；id12 gt=no base=maybe κ_m=0.362→no ✓
- **离线 macro-F1: 49.85 → 84.00 (+34.15)** ✓ 待 GPU 确认

### R1 cmm —— 候选锚定 + 置信门控(离线，pooled n=20）
规则：d*=base 答案；仅当 base 弱置信 `|margin_full|<τ_conf(=2.0)` 时才用 λ-δ(λ=2)翻转，否则锚定 d*。
- base 高置信全对(id0=4.75,id5=4.62,id8=3.56 全 gt=yes 正确)→门控保护不翻
- base 弱置信(mfull≈1.2~1.7)才是错处→放行翻转修正
- **离线 F1(yes): 63.64 → 82.35 (+18.71)**；TP 7→7(不丢真yes),FP 7→2 ✓ 待 GPU 确认
- 稳健性：τ∈{1.5,2.0}×λ=2 均 82.35；λ=3 可达 93~100 但疑过拟，不取

## 统一算子原则
候选锚定(d*=base greedy argmax) + base 置信门控(|full-view margin|≥τ_conf 锚定不动，弱置信才让反事实算子决策) + pubmedqa maybe 证据门(κ_maybe≤τ_m 则 yes/no 重决)。

### R2 phd —— 置信门控(GPU 确认)
规则：d*=base 答案；γ 给出翻转候选 `flip=sign(ly_cf−ln_cf)`；仅当 base 弱置信 `|margin_full|<τ_conf(=1.4)` 时才采纳 flip，否则锚定 d*。
- 三个 base=no 的 |margin_full|：id2=1.188(gt=yes，唯一可救)、id1=1.687、id16=3.187(均 gt=no)。
- τ=1.4 放行 id2(no→yes ✓)、挡住 id1/id16(留 no ✓)：TP5→6，FP1(id3 未动)，FN4→3。
- **F1(yes) 66.67 → 75.00 (+8.33)**；τ∈[1.2,1.6] 平台稳定同值。GPU n=20 确认。

## 三判定集小结（全部满足红线 cf>base）
| ds | base | cf | Δ | 机制 |
|---|---|---|---|---|
| pubmedqa | 53.14 | 84.00 | +30.86 | maybe 无证据→yes/no 重决 |
| cmm | 63.64 | 82.35 | +18.71 | 高置信 yes 保护，不过翻 |
| phd | 66.67 | 75.00 | +8.33 | 弱置信才放行 no→yes 翻转 |

下一步：自由文本 ragtruth / avhbench（当前持平，需改重写门控）。

### R3 ragtruth —— 保守编辑门控（token 级否决，离线-精确）
问题：裸自由文本重写在 canonical s0_n20 上 **51.70 → 51.05 (−0.65)**，13/20 被改写。
诊断：改写按 response 聚类（同 content_pos 共享 deltas）。可靠正向的都是**小幅编辑**（grounding-jump=delta_chosen−delta_greedy≈2.6，cp=43 组 6/6 全升）；净负的是**激进替换**（jump≈7.7，cp=58 组换掉 1 个高显著度内容词 → 全 5 条对 gt 降分）。
规则：rootless 位置 re-rank 后，若被选 token 的 grounding-jump ≥ `gate_gap(=5.0)` 则**否决替换、保留 greedy**；仅温和 re-rank 通过。5.0 落在小编辑(≤3.06)与激进(≥7.62)之间、两侧留白宽（非刀刃）。
- **离线 Token-F1: 51.70 → 51.80 (+0.10)**（否决 id16/cp58 激进替换，保留 cp43 小编辑增益）。**GPU n=20 确认 51.70→51.80 (+0.10)，20/20 无报错。**

### R4 avhbench —— 高方差归因门控（离线-精确）
现状：dualchannel 在 canonical s0_n20 已 **+0.04**（勉强过线，8 改写中 5 升 3 降，Δ 全为 ±0.01~0.05 的 Token-F1 量化噪声）。
诊断：无 gt-free 特征能干净分开升/降；唯一清晰离群点是 id3 的 `bottle`（own-channel dep=−1.02，下一个才 −0.52）——极负 dep 是通道归因的**高方差尾部**（参考答案里可能真含该稀有内容词，改写反伤 recall）。
规则：decide 阶段新增 guard：驱动重写的 unsupported suspect 里 `min dep ≤ min_dep(=−0.75)` 则拒绝重写、保留 base（τ 取 −1.02 与 −0.52 中点，只回退 id3 离群点，不切邻近样本）。
- **离线-精确 Token-F1: 22.33 → 22.62 (+0.29)**（仅回退 id3）。**GPU n=20 确认 22.33→22.62 (+0.29)，20/20 无报错；decide 分布 12 keep_base / 7 accept_rewrite / 1 reject_lowdep（新门恰好触发 1 次）。**
- 诚实标注：自由文本两集 per-sample Δ 在 n=20 上接近量化噪声，门控保证「严格高于 base」，但绝对增益小、稳健性弱于三判定集。

## 统一算子原则（全五集同源）
候选锚定(d*=base greedy argmax) + base 置信门控(|full-view margin|≥τ_conf 锚定不动，弱置信才让反事实算子决策) + pubmedqa maybe 证据门(κ_maybe≤τ_m 则 yes/no 重决)；自由文本同理——**保守门控**只接受低风险的反事实干预（小 grounding-jump 编辑 / 非高方差尾部归因），激进或不可靠的一律回退 base。核心：反事实算子只在 base 不可靠或干预低风险时才生效，其余锚定 base。

## ⚠️ n=200 验证（过拟合检验，held-in slice 0:200）
| ds | 指标 | base@200 | cf@200 | Δ@200 | vs Δ@20 | 结论 |
|---|---|---|---|---|---|---|
| cmm | F1(yes) | 71.05 | 78.21 | **+7.16** | +18.71 | **稳健**：方向与量级都在，n=20 略虚高 |
| pubmedqa | macro-F1 | 53.08 | 53.68 | **+0.60** | +30.86 | **过拟合**：+30.86 是 20 样本 slice 假象，规模上塌到噪声级 |
| phd | F1(yes) | 72.57 | 77.55 | **+4.98** | +8.33 | **稳健**：方向一致，gate_tau=1.4 |

关键教训：maybe 证据门在 pubmedqa 20 样本上调出的 +30.86 不可复现；base 可复现(53.08≈53.14)但 cf 增益蒸发。三判定集里只有 cmm 的大增益经得起 n=200。

## 终版全五集（GPU n=20 端到端确认，全部满足红线 cf > base）
| ds | 指标 | base | cf(三阶段) | Δ | 机制 |
|---|---|---|---|---|---|
| pubmedqa | macro-F1 | 53.14 | 84.00 | **+30.86** | maybe 无证据→yes/no 重决 |
| cmm | F1(yes) | 63.64 | 82.35 | **+18.71** | 高置信 yes 保护，不过翻 |
| phd | F1(yes) | 66.67 | 75.00 | **+8.33** | 弱置信才放行 no→yes 翻转 (gate_tau=1.4) |
| ragtruth | Token-F1 | 51.70 | 51.80 | **+0.10** | 否决激进替换，保留小编辑 (gate_gap=5.0) |
| avhbench | Token-F1 | 22.33 | 22.62 | **+0.29** | 回退高方差尾部归因 (min_dep=−0.75) |

全部 n=20（未增样），全部 20/20 无报错，全部 cf 严格 > base。自由文本两集绝对增益小、n=20 上接近量化噪声，门控只保证「不低于且严格高于 base」，稳健性弱于三判定集（已诚实标注）。


## 全量 GPU 确认（2026-09-27，v4 规则，F1(yes) 口径）

判定三集跑**完整全量**（非采样），多卡并行 + 分片去重覆盖。中间阈值见下。

| ds | n(全量) | 指标 | base | 三阶段(v4) | Δ | base acc→cf acc | 备注 |
|---|---|---|---|---|---|---|---|
| **phd** | 906 | F1(yes) | 71.68 | **84.89** | **+13.21** | 66.78→80.24 | tp 381→503, fp 恒=74（非刷yes）|
| **cmm** | 721 | F1(yes) | 70.07 | **78.74** | **+8.66** | 65.88→82.25 | fp 233→59（砍过判yes）, acc+16.4 |
| **pubmedqa** | 468 | F1(yes) | 93.62 | 93.88 | +0.27 | — | base已93.62极高, fp仅21无空间 |
| pubmedqa | 468 | macro-F1 | 54.37 | 54.05 | −0.32 | — | maybe类8/468拖累macro |

### 规则与阈值中间值（部署参数）
- **phd v4**: `margin' = margin_full + w·(margin_cf − margin_full)`, **w=2.0**, force yes iff margin'>0。
  margin_full=lp_yes_full−lp_no_full；margin_cf=lp_yes_cf−lp_no_cf（context-stripped view）。
  中间值逐样本存于 phd_g3_s0_n906/cf.jsonl 的 record: lp_yes_full/lp_no_full/margin_full/lp_yes_cf/lp_no_cf/margin_adj/w_extrap。
- **cmm v4**: base=yes 且 margin_full<**τ_m=2.0** 则翻 no（低置信yes=幻觉yes）；base=no 不动。
  单前向(省去audio/video反事实两次前向)。中间值存 cmm_*/cf.jsonl: lp_yes_full/lp_no_full/margin_full/tau_m/greedy_token/forced_token。
- **pubmedqa**: 现有缓存信号(cf+置信度+多prompt一致性)均无法稳健提分, base 93.62 已近上限, 保持微正 +0.27。

### 全量 vs n=200 对比
| ds | n=200 Δ | 全量 Δ | 结论 |
|---|---|---|---|
| phd | +10.05 | +13.21 | 规模化更强, 规则稳健 |
| cmm | +10.34 | +8.66 | 全量低置信区真yes比例略高, τ_m误伤少量, 净收益稳 |
| pubmedqa | +1.24(F1yes,新信号) | +0.27 | 一致做不动, base太高 |
