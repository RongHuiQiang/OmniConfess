# OmniConfess

> *When a model says something it cannot actually see, hear, or read — it should confess.*

Language models hallucinate because they conflate two very different sources of confidence: what the input evidence genuinely supports, and what their parameters already believed before looking. OmniConfess makes that distinction explicit. For each claim in a model's output, it asks: would the model still assert this if the relevant evidence were removed? If yes, the claim has no grounding in the evidence — the model is confessing to a parametric prior dressed up as an observation. OmniConfess uses that confession signal to selectively correct or suppress the offending content.

---

## The Three-Stage Framework

Every dataset and modality uses the same three-stage structure:

**Stage 1 — Generate.** Run the model with all evidence present (full audio, full video, full passage). Produce the initial answer.

**Stage 2 — Confess.** Build one or more counterfactual views by ablating specific evidence channels. Measure how much each channel actually contributed to each claim or decision token. A channel that contributed little to a claim the model asserted confidently is the confession signal: the model was not grounded there.

**Stage 3 — Decide.** Re-score or rewrite using only the channels that are relevant to the question. Strip the contribution of irrelevant or over-relied channels. Accept the correction only when the evidence supports it.

The math is simple. For a decision *d* and a modality *m*:

```
κ_d^(m)  =  F_d(X) − F_d(X^{−m})        # how much channel m moves the decision
E_d^R    =  Σ_{m ∈ R} κ_d^(m)           # sum over relevant channels only
decision =  argmax_d  E_d^R              # decide from relevant evidence alone
```

---

## Dataset Instantiations

| Dataset | Hallucination type | Ablated channel | Kept channel | Result |
|---|---|---|---|---|
| **HaloQuest** | Single-point AV claim fabrication | Per-claim channel (audio or video) | The channel the claim should depend on | ΔF1 = **+12.60** |
| **PhD** | Misleading text context suppresses visual evidence | Misleading text context | Image + question | Acc = **79.5%** (base 78.5%) |
| **CMM** | Over-reliance on one modality in AV QA | The over-relied modality | The modality the question actually asks about | Acc = **81.3%** |
| **PubMedQA** | Parametric prior overrides passage evidence | Passage (to isolate prior) | Passage contribution κ | λ-free gamma decision |
| **RAGTruth** | Passage-rootless tokens in open-ended RAG generation | Passage (per content token) | Passage-grounded candidates | ε / λ / topk ablation |
| **AVHBench** | Captioning, single-reference token-F1 | Both channels | Both channels | ΔF1 = +0.01 (Oracle ceiling +0.63; metric-structural limit, not method failure) |

---

## Hard Constraints

These are not design choices — they are the definition of the method:

1. **Training-free.** No weight updates, no fine-tuning, no LoRA. Inference-only.
2. **GT never enters the loop.** Ground truth is used only for offline metric computation. It never influences generation, localization, verification, or the accept/reject decision.
3. **Correction must come from evidence grounding.** Every accepted edit has a traceable reason: which channel, what counterfactual delta, which decision rule. Generic self-refinement ("please check your answer") does not qualify.
4. **Explainability.** For every correction: why this claim was flagged, which modality it should have been grounded in, and why the final answer was accepted.
5. **No metric gaming.** The evaluation cohort and metric are fixed before running any experiment and are not adjusted post-hoc.

---

## Repository Structure

```
OmniConfess/          Core method implementation
  methods/            Model loading, prompting, data utilities
  confession/         Per-dataset confession pipeline scripts
    cf_confess_cmm.py         CMM: dual-channel R-selection
    cf_confess_phd.py         PhD: misleading-context ablation
    cf_confess_pubmedqa.py    PubMedQA: passage-contribution gamma
    cf_confess_ragtruth.py    RAGTruth: token-level rootless rerank
    gamma_operator.py         λ-free gamma decision (PhD, PubMedQA)
    confess_pipeline.py       Shared pipeline utilities

OmniHalluBench/       Benchmark data
  avhbench/
  haloquest/
  cmm/
  phd/
  pubmedqa/
  ragtruth/

outputs/              Run outputs (gitignored; directory structure preserved)
```

---

## Running

Install dependencies into a Python 3.10+ environment:

```bash
pip install -r OmniConfess/requirements.txt
```

Model weights are expected at the path configured in each script (default: `/dev/shm/Qwen2.5-Omni-7B`). Override with `MMHALU_MODEL_ROOT` where applicable.

Run a confession pipeline — PhD example on GPU 0:

```bash
CUDA_VISIBLE_DEVICES=0 \
MMHALU_MODEL_ROOT=/tmp \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python OmniConfess/confession/cf_confess_phd.py \
  --out outputs/phd_cf/run0.jsonl
```

Each script accepts `--start`, `--n`, `--shard`, `--nshard` for parallel sharding across GPUs 0–3.

---

## Why "Confession"

A model that outputs a claim unsupported by the evidence it was given is not lying — it is asserting something it believed before it looked. Making that visible is a confession: the model admits, through a counterfactual comparison, that a specific channel did not actually raise the probability of what it said. Stage 3 takes that admission seriously.

The name also points at what the mechanism is not. It is not a critic, a verifier, or a second model. It is the same model, asked to reveal the gap between what it said and what the evidence warranted.

---

## Citation

```bibtex
@article{omniconfess2026,
  title   = {OmniConfess: Counterfactual Channel Attribution for Multimodal Hallucination Correction},
  year    = {2026},
}
```
