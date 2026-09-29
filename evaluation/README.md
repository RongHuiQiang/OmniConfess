# OmniConfess 评测（evaluation/）

本目录收拢了 OmniConfess 六数据集的**官方打分脚本**，并给出从预测结果到指标的复现步骤。打分算法、公式、STOPWORDS 均与仓库主评测保持一致（脚本为原样复制，仅统一放置于此）。

评测分两类：

- **判定类（judgement）**：`phd`、`cmm`、`pubmedqa` —— 输出 yes/no（pubmedqa 另有 maybe），指标为 **Categorical-F1（即 F1(yes)）** 与 **PhD-Index**（yes/no 双向召回的调和平均）。
- **自由文本类（descriptive）**：`ragtruth`、`haloquest`、`avhbench` —— 输出整段文本，指标为 **Token-F1**（内容词级 F1，去 38 词 STOPWORDS，保留 no/not）。

---

## 目录文件

| 文件 | 来源 | 用途 |
|------|------|------|
| `eval_unified.py` | 复制自 `OmniConfess/eval/single_eval.py` | 统一评测器（**推荐**）。判定类 + 自由文本类通吃；判定类含 pubmedqa 的 maybe 处理；按样本落 `*_single_scored.jsonl`。 |
| `eval_unified_legacy.py` | 复制自 `OmniConfess/eval/final_eval.py` | 早期版统一评测器（含 CIDEr、按文件夹聚合）。保留以对齐历史结果；日常用 `eval_unified.py` 即可。 |
| `eval_haloquest.py` | 复制自 `scripts/eval_haloquest_armd_full365.py` | HaloQuest ArmD 专用 Token-F1 scorer，读 `base_pred`/`repaired_pred` 两列做 base vs 修复的 Δ 对比。 |

> 依赖：`eval_unified*.py` 需要 `numpy`、`scikit-learn`、`tqdm`、`pycocoevalcap`、`zhipuai`（见仓库根 `requirements.txt`）。`eval_haloquest.py` 仅用标准库。

---

## 预测结果的两种格式

**格式 A — 统一流水线输出**（`outputs/*_omniconfess_shared_v1/<ds>_omniconfess_full.json[l]`）
每行一个 JSON，字段契约：

| 字段 | 含义 |
|------|------|
| `id` | 样本 id |
| `question` | 问题（含上下文/指令） |
| `ground_truth` | 参考答案（判定类为 `yes`/`no`/`maybe`；自由文本类为整句） |
| `response` | 模型/流水线的预测 |
| `sub_category` | 子类分组键（用于分组报表） |
| `status` | `success` 才计入指标 |

`eval_unified.py` 取 gt 的键顺序为 `gt / answer / label / ground_truth`，取 pred 的键顺序为 `pred / response / output`。**数据集类型由文件名子串自动判定**：文件名须含 `phd` / `cmm` / `pubmedqa` / `halo` / `avh` / `ragtruth` 之一（本仓库产物命名已满足）。

**格式 B — HaloQuest ArmD 输出**（`outputs/lv2_full365/confession_full365_merged.jsonl`）
每行字段：`id`、`gt`、`base_pred`、`repaired_pred`、`edited`。`eval_haloquest.py` 用它对比 base 与修复后的 Token-F1。

---

## Step 1：生成预测

按主仓库流程运行 OmniConfess 流水线，得到各数据集的预测文件（本目录不负责推理，仅打分）：

- 判定类 / 自由文本类统一产物：`outputs/<ds>_omniconfess_shared_v1/<ds>_omniconfess_full.json[l]`
- HaloQuest ArmD 专用产物：`outputs/lv2_full365/confession_full365_merged.jsonl`

## Step 2：打分

以下命令均在**仓库根目录** `web-app/` 下执行。

### 判定类（phd / cmm / pubmedqa）—— Categorical-F1(yes) + PhD-Index

```bash
# PhD
python evaluation/eval_unified.py --target_files outputs/phd_omniconfess_shared_v1/phd_omniconfess_full.json

# CMM
python evaluation/eval_unified.py --target_files outputs/cmm_omniconfess_shared_v1/cmm_omniconfess_full.json

# PubMedQA（maybe 样本按脚本规则排除/记为 miss）
python evaluation/eval_unified.py --target_files outputs/pubmedqa_omniconfess_shared_v1/pubmedqa_omniconfess_full.json
```

输出：终端打印分组指标；同时在 cwd 追加汇总 `judgement_<ds>.csv`（列：File Name / Samples / Categorical-F1 / PhD-Index），并在预测文件旁生成按样本的 `*_single_scored.jsonl`。

### 自由文本类（ragtruth / haloquest / avhbench）—— Token-F1

```bash
# RAGTruth
python evaluation/eval_unified.py --target_files outputs/ragtruth_omniconfess_shared_v1/ragtruth_omniconfess_full.json

# HaloQuest（统一格式，走 Token-F1）
python evaluation/eval_unified.py --target_files outputs/haloquest_omniconfess_shared_v1/haloquest_omniconfess_full.json

# AVHBench
python evaluation/eval_unified.py --target_files outputs/avhbench_omniconfess_shared_v1/avhbench_omniconfess_full.jsonl
```

输出：终端打印 Token-F1；追加汇总 `descriptive_<ds>.csv`（列：File Name / Samples / Token-F1 / GAVIE-Rel / GAVIE-Acc），并生成 `*_single_scored.jsonl`。

> **GAVIE（可选）**：如需 GAVIE-Rel/Acc 打分，设 `export ZHIPU_API_KEY=...`（可选 `ZHIPU_JUDGE_MODEL`，默认 `glm-4`）；不设则跳过，Token-F1 不受影响。

一次性对整个文件夹聚合（自动按文件名识别数据集类型）：

```bash
python evaluation/eval_unified.py --target_folder outputs/haloquest_omniconfess_shared_v1
```

### HaloQuest ArmD 专用（base vs 修复的 Δ Token-F1）

当预测为 **格式 B**（含 `base_pred`/`repaired_pred`/`edited`）时，用专用 scorer：

```bash
python evaluation/eval_haloquest.py outputs/lv2_full365/confession_full365_merged.jsonl
# 逐样本 CSV：
python evaluation/eval_haloquest.py outputs/lv2_full365/confession_full365_merged.jsonl --csv
```

字段契约：`gt`、`base_pred`、`repaired_pred`（缺失则回退 base_pred）、`edited`。输出：base 与 ArmD(修复) 的 Token-F1、Δ、修复样本占比。

---

## 数据集 → scorer → 指标 速查

| 数据集 | 类型 | scorer | 主指标 |
|--------|------|--------|--------|
| phd | 判定 | `eval_unified.py` | Categorical-F1(yes) / PhD-Index |
| cmm | 判定 | `eval_unified.py` | Categorical-F1(yes) / PhD-Index |
| pubmedqa | 判定 | `eval_unified.py` | Categorical-F1(yes) / PhD-Index（maybe 单列处理） |
| ragtruth | 自由文本 | `eval_unified.py` | Token-F1 |
| haloquest | 自由文本 | `eval_unified.py`（统一格式）/ `eval_haloquest.py`（ArmD 格式） | Token-F1 |
| avhbench | 自由文本 | `eval_unified.py` | Token-F1 |
