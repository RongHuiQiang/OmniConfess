# OmniConfess 三阶段 Confession 统一框架

**日期**：2026-09-26  
**适用数据集**：PhD · CMM · PubMedQA · RAGTruth（+ HaloQuest/LV2 作参考锚点）  
**目的**：为论文提供统一的方法叙事——所有数据集遵循同一三阶段原则，各自有独立的阈值参数。

---

## 一、三阶段原则（抽象定义）

```
Stage 1  Generate      用完整证据生成初始答案（全模态/全上下文）
Stage 2  Confess       计算反事实信号，量化每个通道对答案的贡献
                       —— 找出"无根据声明"：答案依赖了某通道，但那个通道
                          不应是决策依据（无关通道），或根本没有支持该声明
Stage 3  Decide        用 confession 信号修正决策
                       —— 保留相关通道的贡献，剥离无关通道的先验偏置
```

数学骨架（gamma 算子，λ-free 版）：

```
F_d(X)          = 完整输入下类别 d 的判别边际 (e.g. lp_yes - lp_no)
kappa^(m)       = F_d(X) - F_d(X^{-m})         # 通道 m 的贡献
E_d^R           = Σ_{m ∈ R} kappa^(m)           # 相关通道贡献之和
E_d^I           = Σ_{m ∈ I} kappa^(m)           # 无关通道贡献之和
decision        = argmax_d E_d^R                 # 只用相关通道做决策
```

"confession"的含义：在 Stage 2 中，模型被迫承认——某个通道（misleading context /
单一模态 / 参考文献）是否真正支持了它的判断；Stage 3 据此纠正。

---

## 二、四数据集映射总表

| 数据集 | 任务 | 无关/待剥离通道 | 相关/待保留通道 | Stage 2 信号 | Stage 3 决策 | 可调阈值 |
|--------|------|----------------|----------------|-------------|-------------|---------|
| **PhD** | 视觉 yes/no，含误导文本上下文 | context（misleading text） | image + question | margin' = ly_cf − ln_cf（cf = image+Q only） | Yes if margin' > 0 | — (λ-free) |
| **CMM** | AV yes/no，单模态过依赖 | 与问题无关的那个模态 | 问题所指模态（audio 或 video） | E_d^R = Σ_{m∈R} (margin_full − margin_no_m) | Yes if E_d^R > 0 | w_audio, w_video, R 选择规则 |
| **PubMedQA** | 文本 yes/no/maybe，有 passage | — （passage 是相关证据） | passage（文献） | kappa_d^(passage) = lp_d_full − lp_d_cf，cf = no-passage | argmax_d kappa_d^(passage) | — (λ-free gamma)；旧版有 λ |
| **RAGTruth** | 开放式文本生成，有参考段落 | — | passage（参考段落） | Δ_w = lp(w\|full) − lp(w\|no-passage) per token | Δ_w < ε → 重排 top-k，score = lp_full + λ·max(Δ,0) | ε（rootless 阈值）, λ_cf（重排强度）, topk |

---

## 三、逐数据集详解

### 3.1 PhD

**幻觉类型**：误导性文字上下文压制了视觉证据支持的正确答案（anti-yes bias on gt=yes items）。

**三阶段展开**：

- **Stage 1**：完整输入 = image + misleading context + marker + question，greedy decode → 得到初始 Yes/No（step 0 决策 token）。
- **Stage 2**：构造反事实 view = image + question only（移除 context+marker）。在决策 token 处，teacher-forced 前向：
  ```
  margin_full = lp_yes_full − lp_no_full    # 完整输入的边际
  margin' = ly_cf − ln_cf                   # image+Q 的边际（confession: 去掉 context 后答案是什么）
  ```
  confession 语义：模型承认——若只看图像和问题（无 text prior），我会选 margin' 对应的答案。
- **Stage 3**：`decision = Yes if margin' > 0 else No`，强制写入决策 token。

**现有代码**：`gamma_operator.py::gamma_margin_phd` + `cf_confess_phd.py`（旧 λ 版）。  
**推荐使用**：`gamma_operator.py` λ-free 版（已验证 acc=79.5%，greedy base=78.5%）。

**各自阈值**：无 λ（λ-free）。若保留旧 λ 版，λ∈{0.5, 1.0, 2.0} 做 ablation。

---

### 3.2 CMM

**幻觉类型**：模型过度依赖单一模态（audio 或 video），忽略另一模态，导致误判。  
- `overrely_audio_ignore_visual`：问题考察视频，但模型被音频带偏 → 相关通道 R = {video}  
- `overrely_visual_ignore_audio`：问题考察音频，但模型被视频带偏 → 相关通道 R = {audio}

**三阶段展开**：

- **Stage 1**：完整输入 = audio + video + question，greedy decode → 初始 Yes/No。
- **Stage 2**：构造两个反事实 view（no-audio、no-video），各做 teacher-forced 前向，在决策 token 处：
  ```
  kappa_audio = margin_full − margin_noaudio    # 音频对 Yes/No 边际的贡献
  kappa_video = margin_full − margin_novideo    # 视频对 Yes/No 边际的贡献
  R = relevant_modalities_cmm(question, group_key)  # 相关通道集合
  E_d^R = Σ_{m∈R} kappa_m
  ```
  confession 语义：模型承认——对于"你看到 X 了吗"这类视觉问题，如果视频通道的贡献 E_d^{video} > 0，则视频支持 Yes；否则 No。
- **Stage 3**：`decision = Yes if E_d^R > 0 else No`，强制写入决策 token。

  **可选扩展（三阶段完整版）**：  
  若 E_d^R 接近 0（不确定区间），追加一次 channel-specific verify prompt（仅开相关模态）：
  ```
  verify_prompt = "只看视频（屏蔽音频），这个问题的答案是 Yes 还是 No？"
  ```
  verify 结果作为 Stage 3 的 tie-breaking 依据。

**现有代码**：`gamma_operator.py::gamma_margin_cmm` + `cf_confess_cmm.py`（旧 λ 版）。  
**改造点**：旧版用 `delta = w_audio * d_audio + w_video * d_video`（双通道加权），缺乏 R 选择逻辑。  
需将 `relevant_modalities_cmm()` 引入 `cf_confess_cmm.py` 的决策路径，使其只累加 R 内通道的 kappa。

**各自阈值**：w_audio, w_video（默认 1.0）；R 选择规则（question-keyword first, group_key fallback）。

---

### 3.3 PubMedQA

**幻觉类型**：文本模型依赖参数化先验（training-time知识），忽略输入中的 passage 证据，或 passage 与先验冲突时选先验。

**三阶段展开**：

- **Stage 1**：完整输入 = passage + question，greedy decode → 初始 Yes/No/Maybe（step 0）。
- **Stage 2**：构造反事实 view = question only（移除 passage），在决策 token 处：
  ```
  lp_{d,full}  ← 完整输入下 d∈{Yes,No,Maybe} 的 log-prob
  lp_{d,cf}    ← no-passage 下各类的 log-prob（纯先验）
  kappa_d^(passage) = lp_{d,full} − lp_{d,cf}    # passage 对各类的贡献
  ```
  confession 语义：模型承认——我对各类的判断中，有多少是来自 passage，有多少是先验记忆。
- **Stage 3**：`decision = argmax_d kappa_d^(passage)`——选 passage 贡献最大的类。
  这等价于"我的答案必须以 passage 为依据，排除先验偏置"。

**现有代码**：`gamma_operator.py::gamma_margin_pubmedqa` + `cf_confess_pubmedqa.py`（旧 λ 版）。  
旧版只调整 Yes 的 logit（`yes_adj = ly + λ * d_cf`），gamma 版用 argmax kappa 三类联动。

**各自阈值**：无 λ（γ 版）；旧 λ 版 λ∈{0.5, 1.0, 2.0}。

---

### 3.4 RAGTruth

**幻觉类型**：开放式 RAG 生成中，模型在某些 content word 处输出不依赖参考段落的词（参数化先验词），而非 passage 中的事实。

**三阶段展开**（注意：Stage 2/3 嵌入 Stage 1 的 token-by-token 循环）：

- **Stage 1（outer loop）**：用完整 passage+question 输入驱动 KV-cache greedy loop。每步得到 full-view logits。
- **Stage 2（per content token）**：在每个 word-initial content token 处，懒惰同步 no-passage KV cache，计算：
  ```
  Δ_w = log P(w | full) − log P(w | no-passage)
  ```
  confession 语义：若 Δ_w < ε（rootless zone），模型承认——这个词的选择不依赖 passage，是参数先验生成的。
- **Stage 3（per content token）**：对 rootless token 触发重排：
  ```
  score(w') = log P(w' | full) + λ_cf * max(Δ_{w'}, 0)
  ```
  奖励 passage-grounded 候选，惩罚 passage-rootless 候选。选 score 最高者替换。

**现有代码**：`cf_confess_ragtruth.py`（完整实现，两 KV cache，lazy sync）。

**各自阈值**：ε（rootless 阈值，probe 显示膝点约 0.1）、λ_cf（重排强度，grid 0.5/1.0）、topk（候选池大小，默认 10）。

---

## 四、参考锚点：HaloQuest/LV2（ArmD，已确认有效）

HaloQuest 是三阶段的完整实现，且是唯一大幅正向结果（ΔF1=+12.60）：

- Stage 1：图像 captioning/QA，full A+V greedy → base answer
- Stage 2：双通道 delta 定位 suspects；channel-specific verify（仅单模态）→ SUPPORTED/CONTRADICTED/UNSUPPORTED
- Stage 3：confession prompt（"video does NOT show X; audio does NOT contain Y"）→ evidence-grounded rewrite → decide guards

AVHBench 是三阶段的边界案例（ΔF1=+0.01，p=0.0065 方向显著但受指标天花板限制）。

---

## 五、论文叙事要点

**统一的方法叙事**：四个数据集共享同一三阶段骨架。差异在于"relevant channel"的定义：

| 数据集 | 无关（剥离） | 相关（保留） |
|--------|------------|------------|
| PhD | misleading text context | image + question |
| CMM | over-relied modality | question-targeted modality |
| PubMedQA | parametric prior (no-passage) | passage evidence |
| RAGTruth | parametric prior (no-passage) | passage-grounded tokens |

**各自阈值的合理性**：  
- PhD/PubMedQA gamma 版无阈值（λ-free），决策仅取信号符号 → 最简洁  
- CMM 有 w_audio/w_video 权重 → 支持不对称通道贡献  
- RAGTruth 有 ε/λ/topk → 生成任务需要精细控制介入率

**Confession 的共同可解释性**：  
每次干预都能回答：为什么改（Δ/kappa 显示原答案依赖了无关通道）、依据哪个模态（R 集合）、为什么接受（Stage 3 的决策规则）。

---

## 六、代码改造待办

| 数据集 | 状态 | 待做 |
|--------|------|------|
| PhD | ✅ gamma_operator.py 已实现 | 无，可直接用 |
| CMM | ⚠️ cf_confess_cmm.py 缺 R 选择 | 将 `relevant_modalities_cmm()` 引入决策路径，替换双通道平权加法为 E_d^R |
| PubMedQA | ✅ gamma_operator.py 已实现 | 无，可直接用 |
| RAGTruth | ✅ cf_confess_ragtruth.py 已实现 | 无，ε/λ 超参 ablation 可补 |
| HaloQuest | ✅ ArmD 已实现且有+12.60结果 | 无 |
| AVHBench | ✅ 案例分析完成 | 写入论文边界条件章节 |
