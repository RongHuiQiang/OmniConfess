# OmniConfess 论文算法 ↔ 重构代码 对照

本文件把论文 §4 METHODOLOGY 的三件套(Commitment Anchoring / Evidence
Interrogation / Confession-Guided Correction)与仓库里三阶段重构后的实现逐条
对照,标出**一致处**与**差异/冲突**,供重构版本核对使用。

- 论文方程编号沿用投稿稿(Eq 5–15)。
- 代码指 `confession/cf_confess_{phd,pubmedqa,cmm,qwen3,cmm_qwen3}.py`
  + `confession/gamma_operator.py` + `confession/cf_common.py`。
- 重要前提:**重构是纯结构搬家**,判定逻辑一字未改,已在 phd 10 条上与改前
  逐字节一致。因此下文任何"差异"都是 **论文 与(既有)代码** 之间的差异,
  不是重构引入的。

---

## 0. 三阶段 ↔ 论文三件套 的骨架映射

| 论文 (§4) | 代码里的阶段 | 代码落点 |
|---|---|---|
| §4.2 Commitment Anchoring(承诺锚定) | Stage 1 Generate + 前缀冻结 | 主循环 `model.thinker.generate(do_sample=False)` + `CFConfessProcessor` 记录 greedy 前缀 |
| §4.3 Evidence Interrogation(证据审讯) | Stage 2 Confess | `_confess_lp_yes_no` / `_cf_lp_yes` / `_confess_lps`:把冻结前缀 teacher-force 到"删通道"视图 |
| §4.4 Confession-Guided Correction(坦白引导修正) | Stage 3 Decide | `gamma_operator.py` 里的决策函数 + 硬钉 token |

三阶段命名 Generate / Confess / Decide 就是论文 Anchoring / Interrogation /
Correction 的一一对应。

---

## 1. 一致的部分(代码忠实于论文)

### 1.1 承诺锚定与"冻结"(Eq 5–6, Proposition 1)
- 论文:自回归解码得到候选 `y_cand` 与全视图分数 `r_θ^full`;**候选及其前缀在
  审讯期间被冻结**;在改动证据 `X'` 下,teacher-force 这个冻结候选得到锚定分数
  (Eq 6)。Proposition 1:锚定消除跨证据条件比较里的"轨迹变化"。
- 代码:第一阶段贪心解码(`do_sample=False`, seed=42)得到前缀 `gen`;第二阶段
  `_confess_*` 把 `gen` 原样拼到"删通道"视图后**只前向一次**读判定位 logits。
  这正是 Eq 6 的锚定重打分。
- **结论:完全一致。** 上一轮讨论的"推理前缀在全视图下生成后被冻结复用"就是
  论文的核心机制本身(Proposition 1 的"消除轨迹变化"),是设计意图,不是 bug。

### 1.2 边际、通道贡献、相关集(Eq 7–8)
- `F_i(Z) = log p(cand|·,Z) − log p(alt|·,Z)` → 二分类即 `ly − ln`。代码用 `ly−ln`。✓
- `κ^(m) = F_i(X) − F_i(X^{-m})` → 代码按通道算 `d_cf`(cmm 双通道)/单通道差。✓
- 通道相关性 `r_m(Q,T) ∈ {0,1}`,`R`=相关集 → `relevant_modalities_cmm`
  (问题关键词 + group_key 元字段)实现;phd 的 context 恒为无关 `I`、image 恒为
  相关 `R`;pubmedqa 的 passage 为相关 `R`。✓

### 1.3 判定修正对 phd 的等价性(Eq 11–12)
- 论文判定修正:`z̃_d(J_cand) = z_d(J_cand) + γ(J_cand)`,`γ = E_d^R − F_d(X)`,
  在 `{J_cand, a_d}` 上取 argmax。代入 `F_d(X)=z_d(J_cand)−z_d(a_d)` 可化简为:
  **保留候选当且仅当 `E_d^R ≥ 0`**。
- 代码 `gamma_margin_phd`:`margin' = ly_cf − ln_cf`(= 相关视图 yes−no 边际
  = 二分类的 `E^R`),`margin' > 0` 判 Yes 否则 No。
- **二分类下两者代数等价**(见下 §2.2 的唯一微小例外)。✓

---

## 2. 差异与冲突(需要与论文对齐的地方)

### 2.1 【主要冲突 — 已决策】cmm 部署的是 λ-δ,不是论文的无参 γ

> **决策(2026-09-27,用户拍板):cmm / cmm_qwen3 保留 λ-δ(带手调旋钮 λ=2)。
> 代码无需改动**(部署现状即 λ-δ)。和解走 §2.1 的路子 (a):**论文措辞需明确
> cmm 用 λ 变体、λ 记为任务级超参**,不能声称 cmm 也用无参 γ。这是纯论文写作
> 侧的待办,不影响代码与分数验证。cmm 的验证红线对齐它自己改前 λ-δ 的分数
> (见 baseline 表),不对齐无参 γ。

- **论文**:Eq 11–12 把判定修正写成 **无 λ 的闭式** `γ(J_cand)=E^R−F(X)`,呈现为
  所有判定任务通用、且"training-free / 无超参"。
- **代码**:`cf_confess_cmm.py`(以及 `cf_confess_cmm_qwen3.py`)部署的是
  `margin_adj = margin_full + λ · Σ_{m∈R} w_m · d_cf_m`,默认 **λ=2**。这是一个
  **带可调超参 λ 的算子**,不是闭式 γ。
- 论文口径的 cmm 无参 γ(`gamma_margin_cmm`)确实**存在于 `gamma_operator.py`,
  但在部署脚本里从未被调用**(死代码)。
- 依据(见 `gamma_operator.py` 混合路由注释 + 记忆 `gamma-operator-method-a-decision`):
  cmm 上无参 γ 离线最高 0.714,λ-δ 达 0.813,故 cmm 保留 λ-δ。
- **这是论文与代码最需要对齐的一处。** 论文若声称所有判定任务用无参 γ,而 cmm
  实际用了超参 λ,则"无超参"对 cmm 被高估。三条可选的和解方式:
  (a) 论文明确 cmm 用 λ-δ 变体并把 λ 记为任务超参;
  (b) 代码把 cmm 换回 `gamma_margin_cmm`(接受掉分);
  (c) 论文把 λ-δ 表述为 γ 框架的一个"保留全边际 + 加权反事实"的推广,λ 为任务级常数。

### 2.2 【微小差异】phd 平局约定
- 论文:平局保留候选(argmax 且平局留候选)。
- 代码:`margin' > 0` 才判 Yes,故 `margin'` 恰为 0 时一律 No,与"候选是谁"无关。
- 仅在 `margin' == 0` 这一测度为零的边界上可能不同,实践影响可忽略。记录在案即可。

### 2.3 【推广/需appendix佐证】pubmedqa 三分类 argmax vs 论文二分类修正
- 论文 Eq 12 判定修正定义在二元 `Y={J_cand, a_d}` 上。
- 代码 `gamma_scores_pubmedqa`:三类 {yes,no,maybe} 上 `κ_d = lp_d_full − lp_d_cf`,
  取 `argmax_d κ_d`。这是"保留 passage 支撑的类"的**三分类推广**。
- pubmedqa 删的是**相关通道 passage**,故 Stage 2 读到的是无 passage 的先验 `P^∅`,
  决策= 每类 passage 贡献取最大。方向与论文 `E^R` 框架一致,但机制是全 3 路
  argmax,而非"只调候选类 + 二元 argmax"。论文附录需明确 maybe 类如何纳入 Eq 12,
  否则代码是超出正文写法的推广。

### 2.4 【范围外但需另行核对】自由文本修复(haloquest/avhbench/ragtruth)
- 论文 §4.4 Free-Form Local Repair(Eq 13–15):修订风险
  `ρ_i = [−γ^free_1]+ + γ^free_2 + γ^free_3`;**阈值 τ**;`ρ_i > τ` 的 token 扩展到
  词边界、合并成 span、用原证据+结构化坦白**局部重生成** `Rev_θ`。
- **用户关心的阈值 τ 就在这里(自由文本路径),不在判定路径。** 判定三脚本里没有
  非零 τ(phd/pubmedqa 是 τ=0 的符号/argmax 规则)。
- 本次三阶段判定重构(task #3)**不实现 Eq 13–15**;自由文本是另一条代码线
  (`scripts/avh_confess_dualchannel.py` + LV2 pipeline)。
- 记忆记录:ρ/τ 局部 span 修复在 HaloQuest 上是死路(多轮全负 delta);拿到
  +12.60 的是 Evidence-Grounded 整句重写 + verify + decide(LV2 ArmD),**与论文
  局部 span 重生成不同**。→ **论文与代码在自由文本路径可能有实质冲突**,需单独
  和解;不在本次判定重构范围内,但在此登记。

---

## 3. 跨骨干(qwen3 / cmm_qwen3)
- `cf_confess_qwen3.py` 镜像 phd:走无参 γ(`gamma_margin_phd`),τ=0。映射同 §1.3。
- `cf_confess_cmm_qwen3.py` 镜像 cmm:走 λ-δ(等权),继承 §2.1 的同一冲突。
- 差异仅在骨干加载与 eos 修复,决策规则与 Qwen2.5 版一致。

---

## 4. 核对清单(重构版对照论文时逐条打勾)
- [ ] 承诺锚定:前缀冻结 + teacher-force 改动证据(Eq 6)—— 代码 ✓
- [ ] 边际/κ/相关集(Eq 7–8)—— 代码 ✓
- [ ] phd 判定 = `E^R` 符号规则(Eq 11–12 二分类等价)—— ✓,注意平局约定 §2.2
- [ ] pubmedqa 三分类 argmax κ —— 推广,待附录佐证 §2.3
- [x] **cmm 判定算子:代码保留 λ-δ(λ=2)—— 已决策 §2.1;论文措辞待改,代码不动**
- [ ] 自由文本 ρ/τ 局部 span 修复(Eq 13–15)—— 判定重构未覆盖;部署走 LV2,待和解 §2.4
