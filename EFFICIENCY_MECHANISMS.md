# EFFICIENCY_MECHANISMS.md — CF 成本优化机制（三加速点：理论上界 + 实测墙钟）

维护者：CF 加速机制 agent（S5a/b/c）。本文件不改 RESULTS.md；登记见 `outputs/s5a_cost_gate/RESULTS_snippet.md`。
数据来源：轨迹 `outputs/{phd,cmm}_base_model/*_scored.jsonl`（time_sec / token_ids / input_len / output_tokens），
CF 决策记录 `outputs/{phd,cmm}_cf_confess/cf_lam*.jsonl`（`cf.dec_index` / `intervened`），权威成本 `ablation/a2_cost.json`。
所有量标注 **可实测** / **理论上界**。骨干 Qwen2.5-Omni-7B，A800（共享，墙钟含 contention）。

---

## 0. 预注册判据（写在任何门测量之前）

- **VAS 触发门（S5a Part 1）**：门控后 Yes-F1 相对无门控 CF 降幅 **≤0.3**（phd CF 84.60 / cmm CF 76.36 为参照），
  且 CF 额外前向开销降到 **+20% 以内** → 判为 **“成本优化兑现”**。
  tau 在 dev 半区按成本目标中点（触发率 → +17.5% 开销，窗口 15–20%）选定，test 半区 + 全量报告。
- **判断类 early-stop（S5b）**：F1 **完全不变**（判断类承诺点=决策点，early-stop 的解码是完整解码在 greedy 下的**严格前缀** →
  决策 token 逐位相同，`prefix_ok_all=True` 为证），且墙钟**显著降** → 判为 **“early-stop 兑现”**。
- **TTSP 熵预筛（S5a Part 2）**：预筛选中候选的池内 Token-F1 ≥ 全组合版 − 0.1。
- **KV 视图共享（S5c）**：诚实区分——phd（遮 context 保图，前缀可复用）报可复用前缀占比；
  cmm（遮模态丢序列前端 token，位置错位）报不可复用并说明原因。

> **诚实基调**：理论峰值（省的 decode/prefill token 数）与实测墙钟不等——attention 读取与显存搬运常吃掉理论收益。
> 本文件对每条同时给出“省的 token 数（理论上界/可离线实测）”和“墙钟降幅（可实测）”。

---

## 1. 成本现状（A2，可实测，权威 `ablation/a2_cost.json`）

**注意（勘误）**：CF 前向开销 phd 与 cmm 与初始简报口径相反，以 A2 为准：

| 数据集 | base 均值 | CF 均值 | **CF 开销** | 额外视图 | probe 前向/样本 | 干预率 |
|---|---|---|---|---|---|---|
| phd | 1.06 s | 2.09 s | **+97.2%**（中位 +141%） | 1（遮 context 保图） | 1.03 s | 21.3% |
| cmm | 2.44 s | 3.46 s | **+41.8%**（中位 +43.5%） | 2（遮 audio / 遮 video） | 1.02 s（0.51/视图） | 34.8% |

**成本结构诊断**：CF 开销几乎全部来自 **反事实视图的 prefill 前向**（phd 1.03 s ≈ base 1.06 s 的整整一次；
cmm 两视图合 1.02 s）。修正本身 ~0（决策 token 处硬 logit 强制，1 token）。
→ 三个加速点全部瞄准“少做/复用 prefill 前向”与“少 decode”。

---

## 2. 加速点 (a)：VAS 触发门 —— 只在判别力高的样本上做额外前向

**机制**：CF 的额外视图前向只对“模态依赖结构可诊断”的样本有价值。用 layer-14 证据 span 的注意力占比
（VISTA VAS 口径，`s5a_vas_collect_qwen25.py` 单次 eager teacher-forced 前向，每样本在 base 轨迹上测一次）作门。
低 VAS 样本跳过额外视图 → 回退 base 预测。

- **门信号提取成本（可实测）**：每样本 1 次前向 ~0.3–0.4 s（VAS 已全量采集，phd 906 / cmm 721）。
  注意：门信号本身要一次前向，故“净省”= 省下的额外视图前向 − 门信号前向；在**未来在线版**门信号可与 base 轨迹同批免费。
- **样本过滤率 → 开销（可实测，S5a 回放）**：tau 选到触发率 r，则 CF 开销 ≈ 原开销 × r。
  目标 +17.5%（phd 从 +97% 需 r≈0.18；cmm 从 +42% 需 r≈0.42）。

**S5a Part 1 实测结果**（回放既有全量 CF 记录，零额外 GPU；VAS 全量采集 phd 906 / cmm 721，全 ok）：

| 数据集 | 无门 CF F1 | 门后 F1（全量/test 半） | ΔF1 | 无门开销(A2) | 门后开销 | 触发率(需/实) | 判据 |
|---|---|---|---|---|---|---|---|
| phd | 84.60 | 78.00 / 78.79 | **−6.60** | +97% | +57% | 0.18 / 0.59 | **未过** |
| cmm | 76.36 | 71.08 / 71.86 | **−5.27** | +42% | +30% | 0.42 / 0.71 | **未过** |

**判据未过（负结果，照登）**。即便在最宽松触发率 0.8（只门掉最低 VAS 的 20%），phd 仍降 1.78、cmm 降 1.02 F1，
均远超 0.3 容差（敏感性扫描见 summary.json）。

**为什么失败（诊断，关键）**：VAS 与 CF 收益 **反相关**。在 CF 实际改变（且多数纠正）预测的样本上，
VAS 反而**更低**：

| 数据集 | CF 改变预测数（纠正/改错） | VAS 均值(改变样本) | VAS 均值(未变样本) | 结论 |
|---|---|---|---|---|
| phd | 193（154/39） | 0.0278 | 0.0362 | VAS 越低 CF 越有用 → 门反了 |
| cmm | 251（181/70） | 0.0352 | 0.0420 | 同上 |

→ VAS 触发门在本任务上**证伪**：高 VAS = 模型已重视证据 = CF 无需介入；门保留高 VAS 恰好扔掉 CF 的收益样本。
**成本优化未由 VAS 门兑现**。这条负结果本身有价值：它把“何时该省”与“信号方向”钉在一起，
提示正确的门信号应是 **CF 分歧 Δ（delta_cf）本身**而非 VAS（Δ 门是未来工作，需在线记录）。

---

## 3. 加速点 (b)：判断类 early-stop —— 承诺点=决策点，decode 早停

**核心事实（可实测，从 `cf.dec_index` + `output_tokens`）**：判断类的决策 token 就是承诺点。

| 数据集 | dec_index 均值 | dec_index 中位 | 承诺≤1 token 占比 | 说明 |
|---|---|---|---|---|
| phd | 8.93 | **0** | **83.1%** | 直答样本决策=首 token |
| cmm | 0.00 | **0** | **100%** | 每条决策都在首 token |

**两种 decode 形态（诚实区分，这是关键）**：
- **直答型**（phd 753 / cmm 720）：答案在首 token（"Yes"/"No" / "No, I did not see…"），decode ~2 token，
  early-stop 每条省 ~1 token。
- **boxed-CoT 型**（phd 153 条）：先推理后 `\boxed{}`，**决策在末尾**（dec_index 均值 52.9 / output 55.9），
  这类 decode 占 phd 全部 decode 的 **85%**（8546/10078 token），但因决策真在末尾，**不能截断 CoT**——
  否则丢答案。这类 early-stop 只能省 306 token（尾部 EOS/标点）。

**理论上界（decode token，可离线实测，早停到 dec_index+1）**：

| 数据集 | 总 decode token | 可省 decode token | **可省占比** | 其中直答 / boxed-CoT |
|---|---|---|---|---|
| phd | 10 078 | 1 085 | **10.8%** | 779 / 306 |
| cmm | 1 872 | 1 151 | **61.5%** | 1 151 / 0 |

**墙钟折算（理论上界，线性拟合 `time_sec ≈ a + b·out_tok`）**：

| 数据集 | prefill+固定 a | 每 decode token b | 省墙钟/样本 | **占 base 墙钟** |
|---|---|---|---|---|
| phd | 0.615 s | 0.040 s | 0.048 s | **4.5%** |
| cmm | 2.334 s | 0.039 s | 0.063 s | **2.6%** |

> **诚实结论**：decode 早停的**理论峰值可观（cmm 61.5% decode token）**，但**墙钟只降 2.6–4.5%**——
> 因为 prefill（phd 0.62 s / cmm 2.33 s 固定项）主宰单样本时延，decode 每 token 仅 ~0.04 s。
> early-stop 是“对的方向、小的杠杆”：真正省的是 boxed-CoT 那 85% decode，但那部分**决策在末尾不可截**。
> **F1 不变**由构造保证（pred = 决策 token，early-stop 保留之；greedy 前缀性质，S5b `prefix_ok_all` 实证）。

**S5b 实测墙钟**（full max_new=128 vs early-stop 到 dec_index+1+K，同 prompt 同 greedy）：

**S5b 实测墙钟**（`thinker.generate`，CF 同款快解码路径；full vs early-stop 到 dec_index+1+K，同 prompt 同 greedy；40 样本，20 直答+20 boxed）：

| 数据集 | full_cap | t_full 均值 | t_stop 均值 | 墙钟降幅 | prefix_ok_all（F1 不变证明） | 分层 direct / boxed |
|---|---|---|---|---|---|---|
| phd | 128 | 0.802 s | 0.783 s | **2.4%** | **True** | 1.9% / 2.4% |
| cmm | 32 | 0.362 s | 0.190 s | **47.6%** | **True** | 31.6% / 57.8% |

> **两个数字口径不同，诚实说明**：
> - phd（full_cap=128）：多数样本自然早停或 boxed-CoT 决策在末尾，early-stop 只降 2.4%——**与离线线性拟合 4.5% 同量级**。
> - cmm（full_cap=32）：full 强制解到 32 token（cmm base 常 ramble 到 cap），而决策在 token 0，
>   early-stop 到 2 token → 降 **47.6%**。这个数字**依赖 full_cap 假设**：若按 cmm base 实际 output_tokens（均值 2.6）算，
>   离线墙钟降幅仅 **2.6%**（§上表）。
> **统一结论**：`prefix_ok_all=True` 严格证明 **F1 不变**（early-stop 解码是完整解码的 greedy 前缀，决策 token 逐位相同）。
> 墙钟收益 = 省下的“决策后 ramble” × 每 token ~0.04 s；杠杆大小取决于模型会 ramble 多少——
> 在会跑到长度上限的场景（长 CoT / 无 eos）收益大（cmm 演示到 47.6%），在自然早停场景收益小（phd 2.4%）。
> **early-stop 判据兑现**：F1 完全不变 ✅ + 墙钟在 ramble 场景显著降 ✅（诚实：判断类直答场景本就短，杠杆有限）。

---

## 4. 加速点 (c)：CF 视图 KV 前缀共享 —— 复用 clean/masked 的公共 prefix

**机制**：CF 的额外视图前向与 clean 视图共享一段前缀 token，其 KV 可直接复用 → 少 prefill。
两种遮法的前缀共享程度**由实测决定**（先验假设“cmm 媒体前置 → 零复用”被 S5c **推翻**，见下）：
- **phd（遮 context 保图）**：clean=含误导 context 的问句；masked=`q_only`（MARKER 后的问句、context 剥离），
  media（图像 token）与 system prompt 完全相同。二者从头共享到 context 文本发散处。
- **cmm（遮模态）**：masked=丢 audio 的 token span（`clean_a` 视图，video+text 保留）。
  **实测发现媒体 token 并非在最前端**——system+prompt 前导占前 ~600 token，audio span 在其后，
  故丢 audio 前的前缀仍可复用；只有 audio span 起点之后位置错位不可用。

**理论上界**：prefill 成本 ~线性于 token 数，故可省 prefill ≈ 可复用前缀占比 = clean/masked 首个发散 token / input_len。

**S5c 实测可复用前缀占比**（30 样本，结构一致）：

| 数据集 | input_len 均值 | 首发散位置(均/中) | **可复用前缀占比** | prefill 省(上界) | 备注 |
|---|---|---|---|---|---|
| phd | 642.8 | 537.8 / 532 | **83.6%** | ≤83.6% | context 在 image+问句之后，发散晚 → 高复用 |
| cmm | 1027.9 | 627 / 603 | **60.9%** | ≤60.9% | audio span 在前导之后，发散于 ~603 → 中高复用 |

> **诚实修正**：初始设计假设 cmm 不可复用（媒体前置→位置错位），**实测推翻**——audio span 起于 token ~603（非 0），
> 其前 60.9% 前缀可复用。KV 共享是**两个数据集都成立**的实义杠杆（phd 83.6% / cmm 60.9% prefill 上界）。
> **墙钟折扣警告**：这是**理论 prefill token 上界**；实测墙钟收益要打折——复用省的是矩阵乘计算，
> 但读回长 KV（phd ~538 token × 28 层 × bf16）本身有显存搬运成本，且当前 CF 实现未做 prefix-cache，
> 需改 `CFConfessProcessor` 走 `past_key_values` 才能兑现（工程未落地，此处报可达上界）。

---

## 5. TTSP 熵预筛（S5a Part 2，附带件）

BoN 重排只在 top-2 surprisal 候选上打全组合分（screen=CE_top5mean）。判据：选中 F1 ≥ 全组合 −0.1。
成本两套口径：screen 需一次前向 → 约 −28%；screen 免费（生成时记 rollout logp，一行改动）→ −60%（未来池可达）。

<!-- S5A Part2 measured -->
| 池 | 一致率 | 全组合 F1 | 预筛 F1 | ΔF1 | 省(需前向) | 省(免费) | 判据 |
|---|---|---|---|---|---|---|---|
| haloquest | 65.8% | 35.14 | 33.49 | **−1.64** | −27.8% | −50% | **未过**（>0.1） |
| ragtruth | 31.0% | 42.39 | 39.05 | **−3.34** | −27.8% | −50% | **未过**（>0.1） |

> **判据未过（负结果，照登）**：top-2 surprisal 预筛与全组合选择只 66%/31% 一致，选中候选 F1 降 1.64/3.34，
> 超 0.1 容差。TTSP 熵单独不足以做候选粗筛——CE 排序与最终组合分（Δ+VAS 主导）不对齐。
> 成本口径仍如实：省 −28%（screen 需前向）/ −50%（screen 免费，未来池记 rollout logp 可达）。

---

## 6. 卖点总结（回应 nKqU-W1 / mrLS-W3）—— 实测结论，含负结果

“我们方法贵”→ 我们**实测**了四个成本优化点，诚实区分兑现与未兑现：

| 加速点 | 判据 | 实测 | 兑现？ | 一句话 |
|---|---|---|---|---|
| (a) VAS 触发门 | ΔF1≤0.3 且开销→+20% | phd ΔF1 −6.6 / cmm −5.27，均未达 | **否（负结果）** | VAS 与 CF 收益**反相关**，门保留了 CF 无需介入的样本；提示正确门信号是 CF 分歧 Δ 而非 VAS |
| (b) early-stop | F1 完全不变 + 墙钟显著降 | `prefix_ok_all=True`（F1 严格不变）；墙钟降 phd 2.4% / cmm 47.6%(cap依赖)/2.6%(实际) | **部分兑现** | F1 不变已证；墙钟杠杆在 ramble 场景大、直答场景小 |
| (c) KV 前缀共享 | 诚实报可复用占比 | phd **83.6%** / cmm **60.9%** prefill 上界 | **上界兑现（工程待落）** | 两数据集都可复用（cmm 先验假设被推翻）；实测墙钟需改 CF 走 prefix-cache |
| (d) TTSP 熵预筛 | ΔF1≥−0.1 | halo −1.64 / rag −3.34 | **否（负结果）** | CE 单独排序与组合分不对齐 |

**给写作的诚实定位**：这不是“成本全解决了”，而是“我们**测量**了成本优化的可行边界”——
- **最硬的正向证据是 KV 前缀共享的理论上界（phd 83.6% / cmm 60.9% prefill 可省）** 和 **early-stop 的 F1 严格不变**；
- **两个负结果（VAS 门 / TTSP 预筛）本身是贡献**：它们把“何时该省、用什么信号省”钉在机制上——
  VAS 门失败恰好印证了论文主张“机制只在病灶与信号对齐处起效”，用错信号（VAS）做门就反噬。
- 所有数字有轨迹离线上界 + GPU 实测墙钟，负结果照登。attention 读取/显存搬运吃掉理论收益的现象在
  early-stop（decode 理论 61.5% → 墙钟 2.6%）上被直接观测到。

---

---

# 第二轮：攻 prefill 瓶颈（S6，零 GPU 论证 + 需卡实测清单）

> 第一轮结论钉死了真瓶颈：**CF 开销 ≈ 反事实视图的一次完整 prefill**（phd 1.03 s ≈ base 一整次；cmm 0.51×2 s），
> decode 早停"方向对、杠杆小"（墙钟 2.6–4.5%），KV 前缀共享报了上界但"工程未落"。
> 本轮**专攻 prefill**：把 KV 前缀共享从"上界"推进到"可落地设计 + 精确节省重估"，并逐一核实批处理 / 决策早停 / 浅层近似 / 文献四条。
> 纪律照旧：理论峰值 vs 墙钟诚实分离；零 GPU 能算的算死，要卡的进清单。

## 6.0 关键重估：prefill 节省的正确分母（比第一轮的 83.6%/60.9% 强得多）

第一轮 `s5c` 报的 `reusable_frac = first_diff / input_len`（相对 **clean 视图**长度），得 phd 83.6% / cmm 60.9%。
**但要省的成本是"额外那次 masked 视图 prefill"，其分母应是 masked_len，不是 clean 的 input_len。**
用同一份 `{phd,cmm}_kv_prefix.json`（30 样本，结构一致，全量 input_len 已复核：phd 642.3 / cmm 1048.9，与探针 642.8 / 1027.9 吻合）重算：

| 数据集 | masked_len 均 | first_diff 均 | **发散后需重算 token** | 线性模型省 prefill | 二次(attn-only)省 prefill |
|---|---|---|---|---|---|
| phd | 556.0 | 537.8 | **18.2** | **96.7%** | 92.9% |
| cmm | 658.0 | 627.0 | **31.0** | **95.3%** | 90.5% |

**为什么这么高（结构性，非巧合）**：两个遮法的发散点都靠近共享前缀的**末尾**——
- **phd 遮 context**：clean=`system+图+q_full(含误导 context)`，masked=`system+图+q_only`。二者共享到 context 文本处发散，
  发散后 masked 视图**只剩 q_only 尾巴 ~18 token**（误导 context 被剥离，没有别的东西补在后面）。
- **cmm 遮 audio**：clean=`system+前导+视频+文本(含 audio span)`，masked=丢 audio span。发散点 = audio 占位符起点（token ~627），
  发散后 masked 视图**只剩 audio 之后的短尾 ~31 token**。

即：**共享前缀的 KV（phd 前 538 / cmm 前 627 token）可精确复用，masked 视图只需重算发散后 18/31 个 token。**
线性 ≈ 96.7% / 95.3%，attention-二次 ≈ 92.9% / 90.5%（两模型都远高于第一轮口径，因为分母换对了）。

**这是精确复用（数值不变的证明）**：发散前的 token 在两视图里**位置、内容、可见 media 全一致** → 其每层 KV bitwise 相同（Qwen2.5 用 RoPE，位置对齐即 KV 对齐）。
复用它们、只对发散后 token 重跑 attention（query 发散尾 × key 全前缀+尾），得到的 `logits[-1]` 与从头重算**逐位一致**。
→ 判据"实现不改数值"**由构造满足**（这与 early-stop 的 greedy 前缀性质同源，都是"精确等价加速"而非近似）。

## 6.1 方向 1【KV 前缀共享落地设计】——本轮 Top-1

**现状代码病灶**（`cf_confess_phd.py:76-90` / `cf_confess_cmm.py:81-93`）：
`_cf_lp_yes_no` 每次决策 token 处，把 `cf_clean['input_ids']`（整条 masked prompt）拼上生成前缀，
调 `self.model.thinker(**kw)` 做**一次从零的完整前向**，`use_cache` 默认走全量 prefill，**没有任何 past_key_values 复用**。
这就是 A2 里那 1.03 s / 0.51×2 s 的全部来源。

**落地设计（精确复用，不改数值）**：
1. **一次性建共享前缀 KV**：CF 生成开始前，clean 视图本来就要 prefill（`thinker.generate` 的第一步）。
   在同一次 prefill 里，对 masked 视图的**共享前缀段**（phd 前 ~538 / cmm 前 ~627 token，与 clean 逐 token 相同）
   其 KV **就是 clean 前缀的 KV**——无需单独算，直接从 clean 的 `past_key_values` 切片取前 `first_diff` 层-位置。
2. **masked forward 走 past_key_values**：`_cf_lp_yes_no` 改为
   `self.model.thinker(input_ids=masked_tail_ids, past_key_values=shared_prefix_kv, use_cache=True, ...)`，
   其中 `masked_tail_ids` = masked prompt 发散后的 18/31 token（+ 判断类几乎为空的生成前缀）。
   position_ids 需显式设为 `[first_diff, first_diff+1, ...]`（因为不是从 0 起）。attention_mask 拼 `[ones(first_diff), ones(tail)]`。
3. **判断类的甜点**：phd 83% / cmm 100% 决策在 dec_index=0（无生成前缀），masked forward 就是"前缀 KV + 18/31 token 尾"，
   一次极短前向出 `logits[-1]` → lp_yes/lp_no。phd boxed-CoT（153 条，dec_index 均 52.9）需把 CoT 前缀也拼进 masked tail，
   但这 153 条本就是少数，且 CoT 前缀对 masked 视图也是新 token（clean 的 CoT ≠ masked 的 CoT，不可复用 CoT KODE 段）——保守起见 boxed 段按现状全量或部分复用，主收益在 83%/100% 的直答段。

**理论 prefill 节省（判据 >40%，远超）**：masked 视图 prefill 省 **线性 95–97% / 二次 90–93%**。
折算到**整条 CF 样本**（CF = clean prefill + masked prefill + ~0 修正）：
- phd：CF 额外开销 = masked 那 1.03 s 的 ~95% → **单样本 CF 墙钟上界从 +97% 降到 ~+5%**（clean prefill 1.06 s + masked 残 ~0.05 s）。
- cmm：两个 masked 视图（audio-drop / video-drop）各省 ~95%，两视图合 1.02 s → 残 ~0.05 s，**CF 开销从 +42% 降到 ~+2–4%**。

**墙钟折扣警告（诚实，第一轮教训）**：以上是 **prefill-FLOPs / 理论上界**。实测墙钟会打折，因为：
(a) 读回长共享前缀 KV（phd 538×28 层×2(k,v)×bf16 ≈ 每样本几十 MB）有显存搬运成本，短尾前向下这部分占比升高；
(b) HF `past_key_values` 手工切片 + 自定义 position_ids 在 `thinker` 上是否走 SDPA fast path 需实测（可能回落到较慢分支）。
→ **需 GPU 实测墙钟**（清单 G1）。但注意：即便墙钟只兑现理论的一半，CF 开销也从 +97%/+42% 降到 ~+25%/+20% 量级，
是四条里**唯一直接砍 prefill 主宰项**的杠杆，量级最大。

**新意与文献定位**：这在工程上=vLLM/SGLang 的 **prefix caching（radix-tree / paged KV）**，是成熟且数值精确的技术；
学术上没有"用共享前缀 KV 把 masked-view 对比解码摊成近一次前向"的命名方法（文献里 CD 的省法是 DoLa 式单模型层间对比，或 top-k 候选轻量重排，见 6.5）。
→ 我们的定位诚实：**借成熟 infra 思想，落到 CF 反事实视图这个具体 shared-prefix 结构上**，节省由 token 布局精确决定（不是新算法，是"把病灶结构对上现成杠杆"）。

## 6.2 方向 2【批处理多视图】——正交小杠杆，可与 6.1 叠加

**问题**：clean 与 masked 两分支能否 batch 成一次 forward（同 batch 不同 attention mask / 不同长度）？
**分析**：能，但省的是 **kernel launch + 权重读取（HBM→SM 搬 7B 权重）**，**不是计算量**（两视图 token 数照算）。
- 两视图长度不同（phd clean 643 / masked 556），batch 需 padding 到 max，浪费 padding 位置的 FLOPs；
- 收益上限 = 权重读取在单次前向墙钟里的占比。7B bf16 权重 ~15 GB，A800 HBM 带宽 ~2 TB/s → 读一遍权重 ~7.5 ms；
  而一次 ~600 token prefill 墙钟 ~0.5–1 s → 权重读取占比 <2%。**batch 省的权重读取<2%，被 padding 浪费部分抵消**。
- **与 6.1 的关系（关键）**：6.1 把 masked forward 砍成 18/31 token 的短尾。**短尾前向恰恰是 memory-bound（计算少、权重读取占比高）**，
  此时把"clean 长 prefill"与"masked 短尾"或"cmm 两个短尾"batch 成一次，省权重读取的相对收益**放大**。
  但两者长度极不均（643 vs 18），batch 效率低；更实际的是 **cmm 的两个 masked 短尾（各 31 token）互相 batch**（长度接近），省一次权重读取。
**判据（墙钟收益）**：单独用 batch 估 <2%（不值得单独做）；作为 6.1 的附加项，cmm 双短尾 batch 可能再省个位数 %。
→ **低优先级，需 GPU 实测且仅在 6.1 之上叠加才有意义**（清单 G3，选做）。

## 6.3 方向 3【决策 token 早停 prefill】——核实：**已经在做，无剩余空间**

核实代码：`_cf_lp_yes_no` / `_cf_lp_yes` 都是 `self.model.thinker(**kw).logits[0, -1]`——
**单次 teacher-forced 前向，只读最后一个位置的 logits，没有任何 decode 循环。**
判断类决策 token = prefix 末端（cmm 100% dec_index=0，phd 83% dec_index=0），masked 视图前向**本身就是"prefill 到决策位、直接取 logits[-1]"**，
不生成任何后续 token。→ **方向 3 描述的"只算到决策 token 位置就停"CF 探针本来就是这么实现的**。
剩余的 decode 早停（方向属第一轮 §3）只作用于 **clean 视图的生成**（boxed-CoT 的 ramble），与 masked prefill 无关，已在 §3 定论（墙钟 2.4%）。
**结论：方向 3 无新增空间，是"已实现"的核实项**。masked 视图的成本全在 prefill 本身（=方向 1 的靶）。

## 6.4 方向 4【投机/浅层近似 Δ】——理论有戏，但**零 GPU 无法论证，需卡**

**设想**：Δ = lp_yes(clean) − lp_yes(masked) 能否用**早退层**（logit lens / 中层 head）近似，从而 masked 视图只跑到中层？
若浅层 Δ 与完整 Δ 高相关，masked forward 可砍掉后半程层数（省 ~50% 深度）。
**能否用已落盘数据验证？——不能。** 核实全部 CF 探针（`outputs/cf_probe/*.jsonl`）：只记录**末层** lp_yes/lp_no/margin，
**无 per-layer / hidden_states / logit-lens 落盘**（`probe_cf_signal_phd.py` 等只取 `.logits[-1]`）。
→ 浅层-完整 Δ 相关性**必须新跑一个记录逐层 logit-lens 的 probe**（dev100 即可，teacher-forced 于决策 token，
对每层做 `lm_head(norm(h_layer))[YES]−[NO]` 取 Δ_layer，算 Δ_layer 与 Δ_final 的 Spearman 及"用 Δ_layer 定符号"的翻转一致率）。
**风险提示（预判）**：与 6.1 相比，方向 4 是**近似加速（改数值）**，且深度只占 masked forward 的一部分——
在 6.1 已把 masked forward 砍到 18/31 token 后，"再省一半层数"的绝对墙钟收益已经很小（短尾前向本就便宜）。
**优先级：低。仅在 6.1 实测墙钟折扣严重、且想进一步压 masked 残留时才值得。** 进清单 G4（选做，需卡且需新 probe）。

## 6.5 方向 5【文献新招】

2024–2026 检索（speculative / contrastive / batched decoding 加速）：
- **Prefix caching（vLLM / SGLang，radix-tree / paged KV、shared-prefix attention kernel）**：成熟、数值精确、正是 6.1 要落的东西。
  多篇明确"decoder-only + prefix caching 摊掉多次相似 query 的重复 prefill"（与 CF 两视图同构）。这是 6.1 的直接工程蓝本。
- **TyphoonMLA (2025)**、**Batch Speculative Decoding Done Right (2025, arXiv 2510.22876)**：shared-prefix 场景 KV 复用与 batched SD 的正确性（position-id / KV-drift 陷阱）——
  对 6.1 的 position_ids 手工设置是重要警示（DSD 因 position-id 错位从第一个 token 就崩）。
- **DoLa / LACD / Delta-CD（单模型层间对比）**：CD 的"单前向"省法是**层间对比**而非"复用前缀"——与我们不同路，但方向 4（浅层 Δ）的理论支撑（浅层 vs 深层 logit 可分）来自这一支。
- **轻量对比重排（top-k 候选 + 预计算 embedding，无额外前向）**：只适用于"对比信号可写成 token-embedding 线性项"的场景，CF 的 Δ 需真前向 masked 视图，不适用。
- **未发现**："用共享前缀 KV 把 masked-view 反事实对比解码摊成近一次前向"的命名方法 → 6.1 的框架有轻度新意（借 infra、对 CF 结构）。

## 6.6 本轮小结表

| 方向 | 判据 | 零 GPU 结论 | 兑现？ | 需卡？ |
|---|---|---|---|---|
| 1 KV 前缀共享落地 | prefill 省 >40% 且不改数值 | **线性 95–97% / 二次 90–93%**（分母修正后远超 40%）；数值由构造精确 | **理论兑现（上界巨大）**；墙钟待实测 | **是（G1，高优先级）** |
| 2 批处理多视图 | 墙钟收益 | 单独 <2%（memory-bound 权重读取占比小）；作 6.1 附加项 cmm 双短尾可再省个位数% | 小，正交 | 是（G3，选做，叠 6.1 上） |
| 3 决策 token 早停 prefill | 已实现？ | **已实现**（masked forward 本就是单次 teacher-forced 取 logits[-1]，无 decode） | 无剩余空间（核实项） | 否 |
| 4 浅层近似 Δ | 浅层-完整 Δ 高相关 | **零 GPU 无法验证**（探针未落盘逐层）；近似加速、6.1 后绝对收益小 | 未知 | 是（G4，选做，需新 probe） |
| 5 文献 | 有无可借 | prefix caching(vLLM/SGLang) 是 6.1 蓝本；batched-SD position-id 陷阱是警示 | 借力 6.1 | — |

## 6.7 需 GPU 实测清单（现在不占卡，baseline 补齐优先）

- **G1【最高优先】KV 前缀共享墙钟实测**：改 `CFConfessProcessor._cf_lp_yes_no` 走 `past_key_values`（切 clean 前缀 KV + 显式 position_ids + masked 短尾）。
  验收两条：(i) **数值等价**——改造版 lp_yes/lp_no 与现状全量重算**逐位一致**（bitwise 或 <1e-3，判据"不改数值"）；
  (ii) **墙钟**——CF 单样本时延 vs A2 现状（phd 2.09 s / cmm 3.46 s），报实测降幅 + 与理论上界（+5% / +2–4%）的折扣。dev100 分层即可先验证，再全量。
- **G3【选做】cmm 双 masked 短尾 batch**：在 G1 之上把 audio-drop / video-drop 两个 ~31 token 短尾 batch 成一次前向，报增量墙钟收益（预期个位数%）。
- **G4【选做】浅层 Δ probe**：新 probe 记录逐层 logit-lens Δ_layer（dev100，teacher-forced 决策 token），
  算 Δ_layer↔Δ_final 相关性与定符号翻转一致率；若某中层已高相关（如 Spearman>0.9 且翻转一致>95%），再评估砍层的墙钟收益。
- 均为**改 CF 前向路径**的实验，风险在数值等价性（G1 有精确判据兜底）与 HF past_key_values / position_ids 的 fast-path 兼容性。

---

## 附：产物清单
- 门信号全量：`outputs/s5a_cost_gate/{phd,cmm}_vas.jsonl`（phd 906 / cmm 721，全 ok）
- VAS 门 + 预筛回放：`outputs/s5a_cost_gate/summary.json`（含 `diag_vas_vs_cf_benefit` 诊断、`sensitivity_scan`）
- KV 前缀实测：`outputs/s5a_cost_gate/{phd,cmm}_kv_prefix.json`
- early-stop 实测：`outputs/s5a_cost_gate/{phd,cmm}_earlystop.json`（`prefix_ok_all` = F1 不变证明）
- 离线三杠杆上界：`outputs/s5a_cost_gate/efficiency_offline.json`
- 脚本：`gpu_batch/scripts/s5a_vas_collect_qwen25.py`（修 `prepare_for_thinker` 签名）、`s5a_gate_replay.py`（修 phd 开销基准 0.42→0.972、加诊断）、`s5b_earlystop_timing.py`（新，thinker 路径）、`s5c_kv_prefix_probe.py`（新）、`s5_efficiency_offline.py`（新）
- 登记片段：`outputs/s5a_cost_gate/RESULTS_snippet.md`
- **第二轮（S6，零 GPU）**：prefill 节省重估用同一份 `{phd,cmm}_kv_prefix.json`（分母从 input_len 改 masked_len，线性 96.7%/95.3%、二次 92.9%/90.5%）；
  代码病灶定位 `cf_confess_phd.py:76-90` / `cf_confess_cmm.py:81-93`（`_cf_lp_yes_no` 无 past_key_values 复用）；
  全量 input_len 复核 phd 642.3(n=906)/cmm 1048.9(n=721) 与探针吻合；方向 3 核实 masked forward 已是单次 teacher-forced（无 decode）；
  方向 4 核实 `outputs/cf_probe/*.jsonl` 无逐层落盘（需新 probe）。需卡清单 G1(高)/G3/G4 见 §6.7。
