#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""prep_pope_data.py -- Build POPE evaluation files from lmms-lab POPE parquets.

Source: HF dataset lmms-lab-encoder/POPE (mirror of lmms-lab/POPE), split
"Full" (COCO val2014 images embedded). POPE is the standard object-hallucination
benchmark (Li et al., EMNLP 2023, arXiv 2305.10315).

Per setting (random / popular / adversarial): sample N items stratified to
50/50 yes/no, extract the embedded image to images/, and write a unified-schema
JSON compatible with OmniConfess/methods/shared.load_dataset
(id / question / answer / category / image_path).

Usage:
  python scripts/prep_pope_data.py --raw /slow_share/tsq/datasets/pope/raw \
      --out /slow_share/tsq/datasets/pope --per-setting 500 --seed 42
"""
import argparse, hashlib, json, os, random

import pyarrow.parquet as pq

SETTINGS = ['random', 'popular', 'adversarial']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--per-setting', type=int, default=500)
    ap.add_argument('--half-yes', type=int, default=250)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    img_dir = os.path.join(args.out, 'images')
    os.makedirs(img_dir, exist_ok=True)
    rng = random.Random(args.seed)

    for s in SETTINGS:
        pf = os.path.join(args.raw, 'Full', f'{s}-00000-of-00001.parquet')
        t = pq.read_table(pf)
        rows = list(range(t.num_rows))
        answers = [t.column('answer')[i].as_py() for i in rows]
        yes_idx = [i for i in rows if answers[i] == 'yes']
        no_idx = [i for i in rows if answers[i] == 'no']
        rng.shuffle(yes_idx); rng.shuffle(no_idx)
        picked = sorted(yes_idx[:args.half_yes] + no_idx[:args.per_setting - args.half_yes])
        assert len(picked) == args.per_setting, (s, len(picked))

        items = []
        for i in picked:
            img = t.column('image')[i].as_py()
            src = t.column('image_source')[i].as_py()   # e.g. COCO_val2014_000000310196
            h = hashlib.md5(img['bytes']).hexdigest()[:10]
            fname = f"{src}_{h}.jpg"
            fpath = os.path.join(img_dir, fname)
            if not os.path.exists(fpath):
                with open(fpath, 'wb') as f:
                    f.write(img['bytes'])
            items.append({
                'id': f"pope_{s}_{t.column('question_id')[i].as_py()}",
                'question': t.column('question')[i].as_py(),
                'answer': answers[i],
                'category': s,                    # sub-category == POPE setting
                'image_path': f'images/{fname}',  # relative to the JSON's dir
            })
        out_json = os.path.join(args.out, f'pope_{s}.json')
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(items, f, ensure_ascii=False, indent=1)
        n_img = len({it['image_path'] for it in items})
        print(f'{s}: {len(items)} items ({sum(1 for a in answers if a=="yes") and len([1 for it in items if it["answer"]=="yes"])} yes), '
              f'{n_img} unique images -> {out_json}', flush=True)


if __name__ == '__main__':
    main()
