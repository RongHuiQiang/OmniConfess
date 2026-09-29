# OmniConfess: Counterfactual Channel Attribution for Multimodal Hallucination Correction

> *A model that cannot distinguish what it sees from what it already believed has no principled basis for correction. OmniConfess makes that distinction measurable.*

**OmniConfess** is a training-free framework that unifies hallucination correction across six multimodal benchmarks under a single three-stage counterfactual spine. The central observation is that a generative model's output conflates two incommensurable sources of confidence — input-grounded evidence and parametric prior belief — and that this conflation can be decomposed, per channel, by asking a controlled counterfactual: *how much would the model's commitment change if this channel were withheld?* The answer is the confession signal; the correction follows from it.

---

## The Problem

Language and multimodal models hallucinate not because they are random, but because they are too confident: the same parameters that encode rich world knowledge also supply fluent, plausible completions when the evidence is absent or contradictory. Standard inference gives no handle on which part of a prediction the evidence actually supports. Post-hoc critics and verifiers try to recover this after the fact, but they operate on the output alone and cannot trace a specific claim back to a specific channel.

OmniConfess addresses this at the level of the forward pass. By teacher-forcing the same answer prefix under both the full view and an ablated view, the method reads the channel's contribution directly from the shift in log-probability mass — without a second model, without labeled correction pairs, and without any supervision signal leaking from the ground truth.

---

## The Three-Stage Spine

The three-stage computation is identical across all six datasets. Only the ablated channel and the Stage-3 decision rule are instantiated per task.

| Stage | Operation |
|---|---|
| **1 · Generate** | Greedy decode under the full input view $X$ to commitment: the decision token for judgement tasks, or the per-content-word positions for free-form generation. |
| **2 · Confess** | Teacher-force one or more counterfactual views $X^{-m}$ on the same prefix; read the per-channel confession signal $\kappa_d^{(m)} = F_d(X) - F_d(X^{-m})$ at the commit position. Every sample that reaches the commit point confesses unconditionally. |
| **3 · Decide** | Aggregate the confession signals over the relevant channel set $R$ and map to a decision: $E_d^R = \sum_{m \in R} \kappa_d^{(m)}$, $\hat{d} = \operatorname{argmax}_d\, E_d^R$. Accept a correction only when the evidence supports it. |

The quantity $E_d^R$ is a measurable, per-sample, per-channel attribution. It is not a heuristic score — it is the model's own log-probability accounting, exposed by the counterfactual.

---

## Design Constraints

OmniConfess is built around four constraints that jointly rule out the failure modes common to post-hoc correction:

1. **Training-free.** Inference-only. No weight updates, no fine-tuning, no adapters, no LoRA. The correction signal exists in the pretrained model's forward pass; eliciting it requires no additional supervision.
2. **Ground truth never enters the loop.** Labels are used strictly for offline metric computation after all runs are complete. They do not influence generation, span localization, the accept/reject decision, or any threshold. Contaminating the decision boundary with oracle information produces numbers that do not generalize.
3. **Every correction is channel-traceable.** Each accepted edit has an explicit reason: which channel moved the decision, by how much, and which decision rule triggered. Generic self-refinement — asking the model to improve its own output without a grounded attribution — does not satisfy this constraint and is not used.
4. **Metric and evaluation cohort are fixed before the first run.** No post-hoc metric selection, no subset filtering, no threshold tuning on held-in data. The reported numbers are the numbers from the protocol as written.

---

## Dataset Instantiations

The spine is abstract; the six instantiations below show how a single mechanism covers judgement tasks, free-form generation, text-only and multimodal channels, and both binary and structured decision spaces.

| Dataset | Ablated channel $m$ | Stage-3 rule |
|---|---|---|
| **PHD** | text context passage | $\lambda$-free $\gamma$-extrapolation on the yes/no log-margin |
| **CMM** | audio view · video view (dual) | dual-channel grounding gate with calibrated threshold |
| **PubMedQA** | biomedical passage | candidate-anchored maybe-gate; $\gamma$ selects over $\{$yes, no, maybe$\}$ |
| **RAGTruth** | passage (per content-word position) | temperate re-rank with fluency and length guards |
| **HaloQuest** | image view | $\delta_{\text{img-tok}}$ span localization → bucket-routed directed regen (VC / FP / IC) |
| **AVHBench** | four cross-modal arm views (each arm is a counterfactual of the others) | per-arm generation → cross-arm confession signal aggregation → directed regeneration over inconsistent spans |

---

## Environment

Dependencies and interpreter are wired by [`scripts/env.sh`](scripts/env.sh). The portable dependency spec is pinned in [`requirements.txt`](requirements.txt) (Python 3.10+, `torch==2.6.0`, `transformers>=4.52.0`).

```bash
source scripts/env.sh          # sets $PY, PYTHONPATH, offline flags, CUDA_VISIBLE_DEVICES
```

For an open-source install without the local tmpfs wiring:

```bash
pip install -r requirements.txt
```

Model weights are loaded from `MMHALU_MODEL_ROOT` (default: Qwen2.5-Omni-7B). The model is loaded once per shard and shared across all samples in that shard.

---

## Quick Start

`run_confess.py` is the unified entry point; it dispatches into `omniconfess/` and reads benchmark data from `OmniHalluBench/<ds>/annotations/<ds>.json` by default.

```bash
# PHD — text-context ablation, gamma decision
$PY run_confess.py phd --output_dir outputs/omniconfess_phd

# CMM — audio/video dual-channel grounding gate
$PY run_confess.py cmm --output_dir outputs/omniconfess_cmm

# PubMedQA — passage ablation, three-class maybe-gate
$PY run_confess.py pubmedqa --output_dir outputs/omniconfess_pubmedqa

# RAGTruth — per-word passage ablation with fluency guards
$PY run_confess.py ragtruth --output_dir outputs/omniconfess_ragtruth

# HaloQuest — image ablation, bucket-routed span correction (shardable)
CUDA_VISIBLE_DEVICES=4,5,6,7 \
$PY run_confess.py haloquest \
    --halo_shard 0 --halo_nshards 4 \
    --output_dir outputs/omniconfess_haloquest

# AVHBench — cross-arm confession aggregation → directed regeneration over inconsistent spans
$PY run_confess.py avhbench \
    --arm_paths a0.jsonl,a1.jsonl,a2.jsonl,a3.jsonl \
    --output_dir outputs/omniconfess_avhbench
```

`--dry_run` limits each job to the first few samples for a smoke test:

```bash
$PY run_confess.py pubmedqa --dry_run
```

Full flag reference: `$PY run_confess.py --help`.

---

## Repository Layout

```
OmniConfess/
├── run_confess.py            # unified six-dataset entry point
├── omniconfess/              # core library
│   ├── backbone/             #   model loading, data I/O, prompting utilities
│   ├── spine.py              #   three-stage driver for binary decision tasks
│   ├── spine_freeform.py     #   three-stage driver for free-form generation
│   ├── operators.py          #   Stage-3 decision operators (γ / grounding / rerank / vote)
│   ├── pipeline.py           #   per-dataset drivers and channel ablation definitions
│   ├── common.py             #   shared utilities (normalization, token helpers)
│   └── cli.py                #   argument parsing and subcommand dispatch
├── evaluation/               # scoring scripts and reproduction steps
├── OmniHalluBench/           # benchmark — annotations/<ds>.json + media/
│   └── phd/  cmm/  pubmedqa/  ragtruth/  haloquest/  avhbench/
├── scripts/                  # env.sh, batch runners, media fetch
├── figs/                     # paper figures
├── requirements.txt
└── README.md
```

The `confession/` directory is the original research codebase from which `omniconfess/` was extracted. It is preserved for historical traceability and is not a dependency of the published pipeline.

---

## Evaluation

Scoring scripts, metric definitions, and step-by-step reproduction instructions are in [`evaluation/README.md`](evaluation/README.md). The primary metrics are Categorical-F1(yes) for judgement tasks and Token-F1 (content-word F1 against 38-word stoplist, retaining *no*/*not*) for free-form tasks.

---

## Citation

```bibtex
@article{omniconfess2026,
  title   = {OmniConfess: Counterfactual Channel Attribution for Multimodal Hallucination Correction},
  author  = {Anonymous},
  journal = {Preprint},
  year    = {2026}
}
```
