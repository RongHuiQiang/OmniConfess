# Known issues

Things a reader will hit that the code does not announce. Each entry says what was
measured, and whether any published number depends on it.

## PubMedQA has a third label, and it was being read by accident

PubMedQA's label set is `yes` / `no` / **`maybe`**. Eight of the 468 released
annotations carry `maybe`; in the `pubmedqa_500` main-table universe it is ten
rows.

`single_eval.py::_parse_yesno_answer` had no `maybe` branch. A ground truth
spelled `"Maybe. <long answer>"` therefore fell past every explicit test and
landed on the last-resort heuristic, which scans the final 30 characters and
returns whichever of `yes` / `no` it finds there. All ten rows came out as `no`,
i.e. the label was decided by an accident of the answer's wording rather than by
the annotation. Methods that answered `no` on those rows collected credit for it.

That branch now exists. `_parse_yesno_answer` returns the sentinel `MAYBE = 2`;
`evaluate_judgment_tasks` records the row (`Ground_Truth: 2`, `Undecidable: 1`)
and then excludes it, because a three-way label has no position on the yes/no
axis these metrics measure; and a `maybe` *prediction* against a yes/no ground
truth is scored as a miss rather than silently folded into `no`. That last case is
not hypothetical: on the raw generations in `results_shipped/pubmedqa_500/`, GD
emits `maybe` 58 times, PD 58, ToT 36 and SC 4. `aggregate_fullset.py` skips the
same rows.

Three things this does **not** do:

1. **It does not change the main table.** The manifest
   (`scripts/judge_fullset_manifest_scored.json`) points at the `*_scored.jsonl`
   files, and those contain zero `maybe` ground truths for all eight arms -- the
   label had already been erased when the scored/formatted variant was produced,
   upstream of any scoring. PubMedQA still reports at `n = 500`, OmniConfess
   92.51 against BaseModel 94.62, exactly as before. The skip added to
   `aggregate_fullset.py` is a deliberate no-op on files scored by the old parser.

2. **It does not make the arms comparable if you rescore the raw files.**
   Rescoring `BaseModel.jsonl` with the current `single_eval.py` gives `n = 490`
   and Yes-F1 95.54 (was 500 / 94.62), verified end to end. But the OmniConfess
   raw file stores the short label in its `gt` field, and for those ten rows it
   stores `"no"` -- the `maybe` was collapsed before the file was written, so the
   parser sees a clean `no` and keeps the row. A naive rescore would compare seven
   baselines at `n = 490` against OmniConfess at `n = 500`. Dropping the same ten
   questions from every arm by question text puts OmniConfess at 93.62 against
   BaseModel's 95.54: the gap narrows from -2.11 to -1.91, the ranking is
   unchanged, and OmniConfess still loses this cell. Nothing has been written back
   into `results_shipped/`; those files are left as generated.

3. **It does not reconcile the raw and scored files.** They are different
   variants: predictions match in 0 of 500 rows, and PD scores 73.12 on the raw
   file against 74.53 on the scored one. Quote the scored files -- they are what
   the manifest and the paper use.

## On PubMedQA, the ground truth is not a join key either

The README says to align across files on `(question, ground_truth)` rather than
`id`. On PubMedQA that is not enough: the baseline arms record the dataset's
long-form answer (`"Yes. our data show ..."`) while the OmniConfess arm records
the short label (`"yes"`), so **all 500 rows differ in ground-truth text** while
naming the same 500 questions (intersection 500/500). Align PubMedQA on the
question alone.

## `single_eval.py` needs the captioning metrics even for judgement datasets

`pycocoevalcap` is imported at module top, so `single_eval.py` will not start
without it even when only `cmm` / `phd` / `pubmedqa` are being scored. The two
scorers under `scripts/` have no such dependency, which is why the main-table
reproduction described in the README needs nothing beyond `scikit-learn`.

## Two HaloQuest images are missing from the media release

`haloquest.json` references 248 distinct images; 246 ship in the media tarball.
These two could not be located in any copy of the dataset available to us:

```
images/3_2718_4502888288_e4860b29c3_o.jpg   (referenced by 2 rows)
images/4_3256_2789033013_c2f79a2bec_o.jpg   (referenced by 1 row)
```

They are Flickr-sourced originals. Three rows of 365 reference them -- 0-based
indices 254, 282, 322 in `haloquest.json`. Every published HaloQuest number in
this repository was produced from generations that already exist under
`results_shipped/`, so no reported result changes; only a from-scratch re-run of
those three rows is affected.

Be aware of how that re-run behaves: a missing media path is **not** an error.
`load_image` returns `None`, and the caller prints `[MEDIA ERROR] invalid image
path` and answers the question anyway -- text-only, with no image. So those rows
would be scored on a blind guess rather than skipped. Grep the run log for
`MEDIA ERROR` and drop the affected rows yourself if you re-generate HaloQuest.

