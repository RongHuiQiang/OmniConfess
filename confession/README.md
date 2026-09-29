# OmniConfess — Unified Runner

Three-stage counterfactual **confession** for hallucination correction, across all
six OmniHalluBench datasets from a single entry point:
[`omniconfess_all.py`](./omniconfess_all.py).

The three-stage spine is identical for every dataset — only the ablated channel
and the Stage-3 decision rule change:

| Stage | What happens |
|---|---|
| **1 Generate** | Greedy decode under the full input view until the model commits to its answer. |
| **2 Confess** | Teacher-force one or more counterfactual (ablated) views on the same prefix; read how much each channel actually moved the answer. |
| **3 Decide** | Map the full-view signal and the confession signal to a decision — force a token (judgement tasks) or emit corrected text (free-form tasks). |

```
κ_d^(m)  =  F_d(X) − F_d(X^{−m})     # how much channel m moves decision d
E_d^R    =  Σ_{m ∈ R} κ_d^(m)        # sum over the relevant channels only
decision =  argmax_d  E_d^R          # decide from relevant evidence alone
```

## Dataset instantiations

| Dataset | Ablated channel | Stage-3 rule |
|---|---|---|
| **phd** | text context | gamma extrapolation on the yes/no margin |
| **cmm** | audio- and video-masked views | dual-channel grounding gate |
| **pubmedqa** | passage | candidate-anchored maybe-gate |
| **ragtruth** | passage (per content word) | temperate re-rank with fluency guards |
| **haloquest** | image | bucket-routed span correction (VC regen / FP pool / IC keep) |
| **avhbench** | four cross-arm views | majority vote-SC |

## Environment

Dependencies and interpreter are wired by [`scripts/env.sh`](../scripts/env.sh)
(pinned in [`requirements.txt`](../requirements.txt)):

```bash
cd <repo-root>
source scripts/env.sh          # sets $PY, PYTHONPATH, offline flags, CUDA
```

## Quick start

All datasets read `OmniHalluBench/<ds>/annotations/<ds>.json` by default. The
wrapper script picks the right arguments per dataset:

```bash
bash confession/run_omniconfess.sh phd                       # judgement task
bash confession/run_omniconfess.sh pubmedqa --dry_run        # 3-sample smoke test
bash confession/run_omniconfess.sh haloquest --halo_shard 0 --halo_nshards 8
```

Or call the runner directly:

```bash
# Judgement tasks (phd / cmm / pubmedqa) and RAGTruth
$PY confession/omniconfess_all.py --datasets phd \
    --data_path OmniHalluBench/phd/annotations/phd.json \
    --output_dir outputs/omniconfess_phd

# HaloQuest — free-text image-ablation arm, shardable across GPUs
$PY confession/omniconfess_all.py --datasets haloquest \
    --halo_shard 0 --halo_nshards 8 \
    --output_dir outputs/omniconfess_haloquest

# AVHBench — Stage-3 vote across four pre-generated arm files
$PY confession/omniconfess_all.py --datasets avhbench \
    --arm_paths a0.jsonl,a1.jsonl,a2.jsonl,a3.jsonl \
    --output_dir outputs/omniconfess_avhbench
```

Every subcommand accepts `--dry_run` (first few samples) for a quick check.
Run `$PY confession/omniconfess_all.py --help` for the full per-dataset flag list.

## Layout

```
confession/
├── omniconfess_all.py        # unified six-dataset entry (this runner)
├── run_omniconfess.sh        # one-command wrapper
├── three_stage.py            # binary (forced-token) three-stage spine
├── three_stage_freeform.py   # free-text three-stage spine
├── cf_confess_haloquest.py   # HaloQuest driver (authoritative pipeline)
├── gamma_operator.py         # Stage-3 gamma / grounding operators
└── cf_common.py              # shared IO / seeding / record helpers
```
