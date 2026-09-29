# Evaluation

All six benchmarks are scored through a single script: `eval_unified.py`.

## Metrics

| Task type | Datasets | Metric |
|---|---|---|
| Judgment | PHD, CMM, PubMedQA | Yes-F1 (eq. 36), PhD-Index (eq. 37) |
| Free-form | RAGTruth, HaloQuest, AVHBench | Token-F1 (eq. 38), GAVIE-Acc (eq. 39) |
| Localization | span-annotated outputs | AUPRC (eq. 40) |

**Yes-F1** is the standard binary F1 on the positive (yes) class. **PhD-Index** is the harmonic mean of yes-recall and no-recall. **Token-F1** is content-word F1 against a 38-word stoplist (no/not are retained). **GAVIE-Acc** is LLM-judged factual accuracy on a 0–10 scale (requires `ZHIPU_API_KEY`). **AUPRC** is the average precision over candidate spans ranked by predicted hallucination risk, pooled across all samples.

PubMedQA has a three-way label set (yes / no / maybe). Rows whose ground truth is maybe are excluded from the binary metrics; a maybe prediction on a yes/no row is scored as a miss.

## Running the scorer

The dataset type is inferred from the filename. Files must contain one of `phd`, `cmm`, `pubmedqa`, `ragtruth`, `halo`, or `avh` as a substring.

```bash
# Score a single file
python evaluation/eval_unified.py --target_files outputs/omniconfess_phd/phd.jsonl

# Score all files in a folder
python evaluation/eval_unified.py --target_folder outputs/omniconfess_haloquest/
```

Per-sample results are written beside the input file as `*_single_scored.jsonl`. Aggregate results are appended to a CSV in the current directory (`judgement_<ds>.csv` or `descriptive_<ds>.csv`).

## Confession format

OmniConfess outputs for HaloQuest and AVHBench carry `base_pred` and `repaired_pred` fields. The scorer detects this automatically and reports:

- Token-F1 for both base and repaired predictions
- Delta with a 95% bootstrap CI (2000 resamples)
- Edited-subset delta (rows where `edited: true`)

No special flag is needed — the same command works for both baseline and confession outputs.

## GAVIE scoring (optional)

Set `ZHIPU_API_KEY` to enable GAVIE-Rel and GAVIE-Acc scores for free-form tasks. The judge model defaults to `glm-4` and can be overridden with `ZHIPU_JUDGE_MODEL`. Token-F1 is unaffected if the key is absent.

## Dependencies

```
numpy
scikit-learn
tqdm
zhipuai      # optional — only for GAVIE scoring
```
