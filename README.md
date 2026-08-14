# OmniConfess

A training-free decoding framework for mitigating cross-modal hallucinations in
omni-modal foundation models, audited at token, layer, and trajectory granularities
through self-confess decoding.

## Repository contents

| Path                          | Description                                                     |
|-------------------------------|-----------------------------------------------------------------|
| `OmniConfess/`                | Core implementation, baselines, and evaluation                   |
| `OmniConfess/methods/`        | Ten decoding methods, each for both backbones                    |
| `OmniConfess/methods/shared/` | Shared model loading, data loading, prompts, and MiniCPM decoding |
| `OmniConfess/eval/`           | Evaluation scripts                                               |
| `OmniHalluBench/`             | Unified 3,540-sample omni-modal hallucination benchmark          |

## The OmniConfess framework

OmniConfess audits and repairs hallucinations at three granularities during
generation:

- **Dist-Confess** (token level) flags prior-driven tokens using the modality
  dependency index (MDI) and re-samples them under a contrastive objective.
- **Layer-Confess** (layer level) measures drift between anchor layers and the
  final layer, producing a hallucination penalty.
- **Path-Confess** (trajectory level) consumes the two signals above to veto and
  re-rank candidate generation paths.

The three modules are **not orthogonal**, and any ablation must say so: Dist emits
the flag that gates Layer's repair and supplies Path's evidence, Layer emits the
penalty that is Path's sole veto input, and Path produces no new observable of its
own. The `--disable_dist`, `--disable_layer` and `--disable_path` help strings
document the exact coupling.

Two backbones are provided:

- `OmniConfess/omniconfess_qwen.py` — Qwen2.5-Omni-7B
- `OmniConfess/omniconfess_minicpm.py` — MiniCPM-o-2.6

## Installation

```bash
pip install -r requirements.txt
```

`transformers >= 4.52` is required: Qwen2.5-Omni support landed in that release and
earlier versions cannot load the checkpoint at all.

Model weights are pulled from the Hub by default (`Qwen/Qwen2.5-Omni-7B`,
`openbmb/MiniCPM-o-2_6`); pass `--model_path` to use a local copy.

## Quick start

Run from the repository root, which is what the built-in defaults assume:

```bash
python OmniConfess/omniconfess_qwen.py \
    --datasets cmm \
    --data_path ./OmniHalluBench/cmm/cmm.json \
    --data_root ./OmniHalluBench/cmm/ \
    --output_dir ./outputs \
    --file_name cmm_run \
    --max_samples 4
```

A baseline is invoked the same way (note that the baselines take `--data_root` at
the benchmark root rather than the dataset subdirectory):

```bash
python OmniConfess/methods/base_cad.py \
    --datasets cmm \
    --data_path ./OmniHalluBench/cmm/cmm.json \
    --data_root ./OmniHalluBench/ \
    --output_dir ./outputs \
    --file_name base_cad_cmm \
    --max_samples 4
```

Runs are resumable: the output file is appended to, and re-running the same command
skips the samples already present.

Every knob is documented in `--help`. The defaults reproduce the shipped
configuration; the flags that change behaviour by default are `--final_tiebreak r5`
and tree voting (disable with `--disable_tree_vote`).

## Baselines

`OmniConfess/methods/` provides ten decoding methods (direct inference plus nine
comparison baselines), each implemented for both backbones (`base_<method>.py` for Qwen, `base_<method>_minicpm.py` for MiniCPM):

| Method       | Kind                     | Notes                                            |
|--------------|--------------------------|--------------------------------------------------|
| `base_model` | direct inference         | reference point                                   |
| `base_sc`    | self-consistency         |                                                   |
| `base_gd`    | guided decoding          |                                                   |
| `base_pd`    | prompt-based decoding    |                                                   |
| `base_tot`   | tree-of-thought          |                                                   |
| `base_dola`  | contrasting layers       | DoLa, ICLR 2024                                   |
| `base_vcd`   | visual contrast          | VCD, CVPR 2024; degenerates to greedy on text     |
| `base_cad`   | context-aware decoding   | CAD, NAACL 2024; stays active on text             |
| `base_avcd`  | audio-visual contrast    | AVCD; needs three modalities                      |
| `base_opera` | over-trust penalty       | OPERA, CVPR 2024; the only beam-search baseline   |

Each file's module docstring records the paper it follows, where every
hyperparameter value comes from, how the implementation differs from the official
release, and how the method degenerates on subsets that do not supply what it
needs. Those degeneration notes matter when reading the numbers: VCD and AVCD
reduce to greedy decoding on the text-only subsets, whereas CAD and OPERA do not.

## OmniHalluBench

| Dataset   | Modality      | Samples |
|-----------|---------------|---------|
| AVHBench  | video + audio | 645     |
| CMM       | video + audio | 721     |
| HaloQuest | image         | 365     |
| PHD       | image         | 906     |
| PubMedQA  | text          | 468     |
| RAGTruth  | text          | 435     |
| **Total** |               | **3,540** |

The unified annotation JSONs are released here. Media files (images, videos, audio)
should be obtained from the corresponding original public sources; the relative
paths in each JSON's media fields indicate the expected layout, and `--data_root`
points at it.

## Evaluation

```bash
# per-sample metrics, writing new files carrying eval_metrics
python OmniConfess/eval/single_eval.py --target_files ./outputs/cmm_run.json

# aggregate report over a folder
python OmniConfess/eval/final_eval.py --target_folder ./outputs
```

Judgement datasets are scored with Yes-class F1; descriptive datasets are scored
with Token-F1 plus the standard captioning metrics.

The LLM-as-a-judge component (GAVIE) reads its key from the `ZHIPU_API_KEY`
environment variable and is skipped when the variable is unset:

```bash
export ZHIPU_API_KEY=...
```

## Prompt templates

The prompt modules (`avh_prompts.py`, `halo_prompts.py`, `halueval_prompts.py`,
`judge_prompts.py`) are **not yet included in this repository**. The code imports
them behind `try/except ImportError` and falls back to built-in defaults when they
are absent, so everything runs without them, but the prompt-version flags
(`--halo_prompt_version`, `--avh_prompt_version`, `--judge_prompt_version`) will
only offer the baseline `v0` until they are added.

## License

MIT License. See `LICENSE`.
