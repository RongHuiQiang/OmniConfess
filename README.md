# OmniConfess

A training-free decoding framework for mitigating cross-modal hallucinations in
omni-modal foundation models, audited at token, layer, and trajectory granularities
through self-confess decoding.

## Repository contents

| Path                          | Description                                                     |
|-------------------------------|-----------------------------------------------------------------|
| `OmniConfess/`                | Core implementation, baselines, and evaluation                   |
| `confess_pipeline.py`         | Unified reproduction entry for the six frozen cells              |
| `configs/`                    | Declarative per-dataset configs (terminal values + provenance)   |
| `OmniConfess/methods/`        | Ten decoding methods, each for both backbones                    |
| `OmniConfess/methods/shared/` | Shared model loading, data loading, prompts, and MiniCPM decoding |
| `OmniConfess/eval/`           | Evaluation scripts                                               |
| `OmniConfess/prompts/`        | Prompt-version registry and its drift check                      |
| `OmniConfess/paths.py`        | Repository-root and model-weight resolution                      |
| `OmniHalluBench/`             | Unified 3,540-sample omni-modal hallucination benchmark          |
| `results_shipped/`            | Per-sample outputs behind every main-table cell                  |
| `scripts/`                    | Manifests mapping each cell to its file, plus the two scorers    |
| `scripts/fetch_media.sh`      | Downloads the benchmark's images / videos / audio (3.24 GB)      |
| `KNOWN_ISSUES.md`             | Measured caveats that the code does not announce                 |

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

Python **3.10** -- every result in this repository was produced on 3.10.18 /
3.10.19, and no other minor version has been tested:

```bash
conda create -n omniconfess python=3.10 && conda activate omniconfess
pip install -r requirements.txt
```

`transformers >= 4.52` is required: Qwen2.5-Omni support landed in that release and
earlier versions cannot load the checkpoint at all.

Model weights are not distributed with this repository. They are pulled from the
Hub by default (`Qwen/Qwen2.5-Omni-7B`, `openbmb/MiniCPM-o-2_6`). To use local
checkpoints without editing any source file, point one environment variable at the
directory holding them:

```bash
export MMHALU_MODEL_ROOT=/srv/hf_models
```

`OmniConfess/paths.py` then maps each Hub id to `$MMHALU_MODEL_ROOT/<name>`, trying
the full id, the bare model name, and a lowercased `_`-normalised variant, and
falling back to the Hub id when nothing matches. `--model_path` still overrides
everything.

No source file in this repository contains an absolute path: dataset, result, and
registry locations are all resolved relative to the repository root.

Not everything needs the full dependency set. Reproducing the main table from the
shipped outputs (below) needs only `scikit-learn`, and the prompt drift check needs
nothing beyond the standard library -- neither imports `torch`. Only actually
running a decoding method requires the rest:

```bash
python OmniConfess/prompts/verify.py    # works in a bare interpreter
```

## Quick start

Run from the repository root, which is what the built-in defaults assume. The
command below reads video and audio, so fetch the media first (see *Media files*
for why reproducing the main table does not need this step):

```bash
bash scripts/fetch_media.sh cmm

python OmniConfess/omniconfess_qwen.py \
    --datasets cmm \
    --data_path ./OmniHalluBench/cmm/cmm.json \
    --data_root ./OmniHalluBench/cmm/ \
    --output_dir ./outputs \
    --file_name cmm_run \
    --max_samples 4
```

A baseline is invoked the same way. `--data_root` only matters when media live
outside the annotation JSON's directory; see the layout note under OmniHalluBench:

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

## Unified reproduction entry: `confess_pipeline.py`

The finished experiments were run by research-stage scripts that are kept in the
repository untouched for provenance. `confess_pipeline.py` is their converged
entry point: one CLI, routing by **task type** (not dataset name), with every
per-dataset knob read from a declarative config whose comments carry the
terminal value and the artifact it came from.

| Task type | Datasets | Operator | Merged from |
|---|---|---|---|
| `judgement` | phd, cmm, pubmedqa | CF-Confess (λ, views, classes per config) | `cf_confess_phd.py`, `cf_confess_cmm.py`, `cf_confess_pubmedqa.py` |
| `descriptive-text` | ragtruth | output-contract layer (prompt version) | `OmniConfess/methods/base_model.py` |
| `descriptive-vqa` | haloquest | four-layer stack (contract × gate × BoN × feature combo) | `answerability_contract_gate.py`, `bon_stacked_haloquest.py`, `bon_stacked_feat_combo.py` |
| `perception-heavy` | avhbench | Vote-SC (multi-arm generation + vote rebuild) | `avhbench_vote_sc_rebuild.py` |

One line per cell (from the repository root; media fetched first for the four
multimodal subsets, `MMHALU_MODEL_ROOT` set if weights are local):

```bash
python confess_pipeline.py --dataset phd                     # CF, λ=2.0, v1  -> 84.60
python confess_pipeline.py --dataset cmm                     # CF, λ=1.0, v1  -> 76.36 / 0.600
python confess_pipeline.py --dataset cmm --prompt_version v6 # prompt-program arm -> 79.15 / 0.641
python confess_pipeline.py --dataset pubmedqa                # CF, λ=0.25, v5 -> 94.58 / 82.96
python confess_pipeline.py --dataset ragtruth                # contract v4    -> Token-F1 51.58
python confess_pipeline.py --dataset haloquest               # four layers    -> Token-F1 40.14
python confess_pipeline.py --dataset avhbench                # Vote-SC 5-arm  -> Token-F1 21.56
```

Useful flags: `--dry-run` (3-item smoke), `--limit N`, `--prompt_version` /
`--lambda_cf` / `--control` overrides, and `--stage` to run one layer at a time
(`route` and `vote` are offline and need neither GPU nor weights; haloquest
layers 1–2 regenerate with `--stage base` / `--stage contract`). Outputs go to
`outputs/pipeline/<dataset>/` and are resume-safe by id.

Two honest notes on equivalence. The judgement and contract layers are greedy
and byte-identical to the research scripts (verified on smoke runs; `time_sec`
is the only volatile field). The haloquest BoN branch and the vote stage
involve sampling / set iteration: the routing stage and the vote rebuild are
byte-identical to their research artifacts, while BoN rollouts are compared
under fixed seeds (greedy-candidate parity plus distribution), and the vote
output's `votes` dict order follows Python set iteration — record contents and
every scored field (`pred`, `local_f1`, per-arm stats) are exact.

The haloquest gate flags and the avhbench cross-generation arm (`q3`,
Qwen3-Omni-30B) are consumed as artifacts produced by their own research
scripts (`answerability_probe.py` + `answerability_gate.py`,
`OmniConfess/methods/base_model_qwen3_omni.py`); `configs/haloquest.yaml` and
`configs/avhbench.yaml` say exactly where each comes from.

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

### Annotation schema

Every subset uses one record layout, so a loader never branches on the dataset
name:

```json
{
  "uid": "cmm_000000",
  "dataset": "cmm",
  "task_type": "judgement",          // judgement | descriptive
  "modality": "visual+audio",        // visual+audio | visual | text
  "answer_space": "yes_no",          // yes_no | yes_no_maybe | free_text
  "question": "Did you see clock in the video?",
  "answer": "yes",                   // the reference text, as released
  "label": "yes",                    // yes | no | maybe | null (free-text subsets)
  "passage": null,                   // retrieval context, text-only subsets
  "media":  {"video": "./...mp4", "audio": "./...wav", "image": null},
  "facets": {"hallucination_type": "overrely_audio_ignore_visual"},
  "source": {}                       // provenance, e.g. the original index
}
```

The three fields above `question` are the only universal taxonomy. Everything a
subset labels for itself lives in `facets` under **its own name**, because the
subsets do not agree on how many levels of category they have and their second
level is not always a child of the first -- HaloQuest's two fields are orthogonal
axes (image source x question type), and CMM's old `granularity` field was a
function of its category rather than a level above it. Flattening them into a
positional `category` / `sub_category` pair is what forced a reader to know which
subset they were holding.

| Subset    | `facets` keys                    | `answer_space` |
|-----------|----------------------------------|----------------|
| CMM       | `hallucination_type`             | `yes_no`       |
| PhD       | `mode`                           | `yes_no`       |
| HaloQuest | `image_source`, `question_type`  | `free_text`    |
| AVHBench  | `task`                           | `free_text`    |
| PubMedQA  | none                             | `yes_no_maybe` |
| RAGTruth  | none                             | `free_text`    |

`answer_space` is not decoration: PubMedQA is genuinely three-way, and its eight
`maybe` rows are preserved in `label` rather than folded into `no`. See
`KNOWN_ISSUES.md` for what that costs downstream.

`OmniConfess/methods/shared/data_loader.py` maps this layout onto the field names
the method scripts already read (`sub_category`, `video_path`, `audio_path`,
`image_path`), choosing one facet per subset as the grouping key -- so the on-disk
schema carries no duplicate spellings and no method script needed changing. Both
layouts load: a record written in the old layout passes through untouched.

`id` is deliberately absent. Only three subsets ever had one, and where it exists
it is an index into the *unfiltered* original release (HaloQuest ships 365 rows
numbered up to 608), so it is provenance, kept under `source`, not a join key.
Cross-file alignment keys on the question text.

### Media files

**You do not need them to reproduce the main table.** The aggregator recomputes
every published number from the generations already committed under
`results_shipped/`; nothing in that path opens an image, a video, or a waveform.
Media are required only to re-run generation, which needs model weights anyway.

When you do want to re-run generation, the media are distributed from this
repository -- exactly the bytes these experiments consumed, not a re-download from
six upstream projects that may have moved on:

```bash
bash scripts/fetch_media.sh              # all four multimodal subsets, 3.24 GB
bash scripts/fetch_media.sh cmm phd      # or only the subsets you need
```

The script pulls one tarball per subset from the `media-v1` release, checks it
against a pinned sha256, unpacks it into the layout below, and counts the files
on disk. Re-running it is cheap: a verified tarball in `.media_cache/` is not
downloaded again. Because this repository is private, the download needs
credentials -- either `gh auth login` once, or `export GITHUB_TOKEN=...`.

| Subset    | Media                          | Files | Size    |
|-----------|--------------------------------|------:|--------:|
| CMM       | 660 mp4 + 660 wav              |  1320 | 1.61 GB |
| AVHBench  | 645 mp4 (video and audio share one file) | 645 | 1.24 GB |
| HaloQuest | png / jpg / webp               |   246 | 0.30 GB |
| PHD       | jpg (COCO val2014)             |   625 | 0.10 GB |
| PubMedQA  | none -- text is inside the JSON |     0 |       - |
| RAGTruth  | none -- text is inside the JSON |     0 |       - |

Only four of the six subsets have media at all. PubMedQA and RAGTruth are
text-only and their annotation JSONs are self-contained.

The tarballs unpack so that each subset's media sit *beside* its annotation JSON.
Path resolution tries the JSON's own directory first, so this arrangement needs no
flags at all:

```
OmniHalluBench/
  cmm/cmm.json             + reorg_raw_files/.../*.mp4, *.wav
  avhbench/avhbench.json   + videos/*.mp4
  haloquest/haloquest.json + images/*.png
  phd/phd.json             + images/val2014/*.jpg
```

If you keep media somewhere else, point `--data_root` at the directory that
*directly contains* those subfolders (e.g. `--data_root /srv/media/cmm`), not at
the benchmark root -- resolution falls back to `data_root/<relative path>` with no
dataset name inserted. Verified against all four subsets.

Two HaloQuest images (of 248 referenced) could not be located and are absent from
the tarball; the 3 affected rows out of 365 are named in `KNOWN_ISSUES.md`, which
also explains why a re-run answers them blind instead of failing.

The media are redistributed here for reproducibility under each original project's
terms; the annotations and code in this repository are ours. Original releases:

| Dataset   | Original release                          | Paper            |
|-----------|-------------------------------------------|------------------|
| AVHBench  | https://github.com/kaist-ami/AVHBench      | arXiv 2410.18325 |
| CMM       | https://github.com/DAMO-NLP-SG/CMM         | arXiv 2410.12787 |
| HaloQuest | https://github.com/google/haloquest        | arXiv 2407.15680 |
| PHD       | https://github.com/jiazhen-code/PhD        | arXiv 2403.11116 |
| PubMedQA  | https://github.com/pubmedqa/pubmedqa       | arXiv 1909.06146 |
| RAGTruth  | https://github.com/ParticleMedia/RAGTruth  | arXiv 2401.00396 |

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

Both arms of every comparison are included, because neither can be reproduced
without the exact prompt bytes:

| Arm                | Module                                  |
|--------------------|-----------------------------------------|
| Baselines          | `OmniConfess/methods/shared/prompts.py` |
| OmniConfess        | `OmniConfess/judge_prompts.py` (cmm / phd / pubmedqa) |
|                    | `OmniConfess/halo_prompts.py` (HaloQuest) |
|                    | `OmniConfess/avh_prompts.py` (AVHBench, 28 versions) |
|                    | `OmniConfess/halueval_prompts.py` |

Two further prompts -- the RAGTruth system prompt and the HalluEval fallback -- are
hardcoded in `omniconfess_qwen.py` and have no version mechanism.

**The baselines and OmniConfess do not share a prompt.** Baseline arms are pinned to
`v0` throughout; the OmniConfess arm uses the modules above. Any comparison that
treats the gap as purely a decoding effect is therefore overstating it, and the
per-dataset prompt version must be reported alongside the score.

`OmniConfess/prompts/` records every version's length, md5, and source module, and
`verify.py` recomputes them from the live code so that a silent reword cannot
detach a published number from its provenance:

```bash
python OmniConfess/prompts/verify.py          # -> 65 prompt versions all match.
python OmniConfess/prompts/verify.py --list   # also print the per-cell verdicts
```

`registry_verdicts.json` is the human-maintained layer that records, per cell,
whether a prompt version helped or hurt and on what strength of evidence. It is not
regenerated from code and must be edited by hand.

## Reproducing the main table

Every main-table cell is backed by a per-sample output file in `results_shipped/`,
and the two manifests in `scripts/` map (dataset, method) to the exact file. This
part needs **no GPU, no model weights, and no API key**:

```bash
# descriptive datasets: Token-F1
python scripts/tokenf1_fullset.py scripts/desc_fullset_manifest.json

# judgement datasets: Yes-class F1 and PhD-Index
python scripts/aggregate_fullset.py --manifest scripts/judge_fullset_manifest_scored.json
```

Three things to know before reading the output:

1. **The judgement manifests come in two stages.** `judge_fullset_manifest.json`
   points at raw generations; `..._scored.json` points at the `single_eval.py`
   output carrying `eval_metrics`, which is what the aggregator reads. Use the
   scored one to get numbers.
2. **PubMedQA uses the `_formatted` variant** for every arm. The unformatted
   generations are prose in which the verdict cannot be parsed, so scoring them
   yields zero for all methods.
3. **GAVIE cannot be reproduced offline.** Token-F1 and the accuracy-family
   metrics are computed locally; the two GAVIE columns require the LLM judge and
   therefore an API key.
4. **PubMedQA's `maybe` label is erased in the scored files.** The scorer now has
   a `maybe` branch and excludes those rows, but the shipped `*_scored.jsonl`
   files predate it and contain no `maybe` ground truth, so the reported numbers
   are unchanged. `KNOWN_ISSUES.md` gives the measurement and the corrected
   comparison.

Both scorers are pure Python: `tokenf1_fullset.py` needs only the standard library
and `aggregate_fullset.py` only `scikit-learn`. Neither loads a model.

Universe sizes differ from the benchmark totals because the main table is computed
on filtered subsets. `OmniHalluBench/main_table_universes/` ships the four whose
item sets are exactly recoverable (`haloquest_608`, `avhbench_645`, `cmm_795`,
`ragtruth_740`). `phd_566` and `pubmedqa_500` are label-balanced subsets that are
not separately materialised; their membership is defined by the shipped result
files, each row of which carries its own question and ground truth.

Cross-file alignment must key on `(question, ground_truth)`, never on `id`.
Different arms wrote `id` against different index spaces -- some the row number of
an already-filtered file, some the original benchmark index -- so aligning by `id`
silently pairs unrelated items. `scripts/judge_fullset_manifest_scored.json`
records the specifics for `phd_566`.

## License

MIT License. See `LICENSE`.
