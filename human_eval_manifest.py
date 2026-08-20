# -*- coding: utf-8 -*-
"""
human_eval_manifest.py -- A11 human-evaluation stratified sample (zero GPU).

Draws the frozen 80-item annotation manifest for the human-eval protocol
(HUMAN_EVAL_PROTOCOL.md):

  40 question-ids x 2 arms (base_model vs our final mechanism, blind order)
  = 80 items. Dataset allocation: haloquest 14 / avhbench 16 / ragtruth 10.
  Within each dataset, ids are stratified by the BASE-arm GAVIE-Acc terciles
  with low-score oversampling (low : mid : high = 2 : 1 : 1), seed 42.

  The ragtruth contract-v4 arm has no GAVIE scores (judge sentinel -1 on all
  rows), so stratification uses the base arm for all three datasets (ids are
  shared across arms).

Writes outputs/human_eval_manifest.json (+ prints the sampling table).
"""
import json, os, random
import numpy as np

ROOT = '/home/tsq/rhq/omniconfess/OmniConfess'

ARMS = {
    'haloquest': {
        'base': 'outputs/haloquest_base_model/haloquest_base_model_full_single_scored.jsonl',
        'final': 'outputs/haloquest_bon_stacked/haloquest_bon_stacked_feat_combo_single_scored.jsonl',
        'final_name': '4-layer (contract v2 x gate x BoN x feature-combo, 40.14)',
        'n_ids': 14, 'media': 'OmniHalluBench/haloquest/images',
    },
    'avhbench': {
        'base': 'outputs/avhbench_base_model/avhbench_base_model_full_single_scored.jsonl',
        'final': 'outputs/avhbench_vote_sc/avhbench_vote_sc_full_single_scored.jsonl',
        'final_name': 'Vote-SC 5arm@2 (21.56)',
        'n_ids': 16, 'media': 'OmniHalluBench/avhbench/videos',
    },
    'ragtruth': {
        'base': 'outputs/ragtruth_base_model/ragtruth_base_model_full_single_scored.jsonl',
        'final': 'outputs/ragtruth_contract_v4/ragtruth_contract_v4_full_single_scored.jsonl',
        'final_name': 'contract v4 (51.58)',
        'n_ids': 10, 'media': None,
    },
}


def jl(p):
    return [json.loads(l) for l in open(os.path.join(ROOT, p)) if l.strip()]


def main():
    rng = random.Random(42)
    manifest = {'seed': 42, 'design': '40 ids x 2 arms (base/final, blind), '
                 'strata = base-arm GAVIE-Acc terciles, low:mid:high = 2:1:1',
                'datasets': {}, 'items': []}
    for ds, cfg in ARMS.items():
        base = {str(r['id']): r for r in jl(cfg['base'])}
        fin = {str(r['id']): r for r in jl(cfg['final'])}
        ids = [i for i in base if i in fin
               and (base[i].get('eval_metrics') or {}).get('GAVIE-Acc', -1) >= 0]
        accs = np.array([base[i]['eval_metrics']['GAVIE-Acc'] for i in ids])
        q1, q2 = np.quantile(accs, [1 / 3, 2 / 3])
        strata = {'low': [], 'mid': [], 'high': []}
        if float((accs == accs.max()).mean()) > 0.5:
            # degenerate (point-mass) distribution, e.g. haloquest GAVIE-Acc is
            # bimodal {0, 10}: use score bands low=0 / mid=(0,10) / high=10
            q1 = q2 = None
            for i in ids:
                a = base[i]['eval_metrics']['GAVIE-Acc']
                strata['low' if a == 0 else 'high' if a == accs.max() else 'mid'].append(i)
        else:
            for i in ids:
                a = base[i]['eval_metrics']['GAVIE-Acc']
                strata['low' if a <= q1 else 'mid' if a <= q2 else 'high'].append(i)
        n = cfg['n_ids']
        alloc = {'low': n // 2, 'mid': n // 4, 'high': n - n // 2 - n // 4}
        picked = []
        for s, k in alloc.items():
            pool = sorted(strata[s])
            k = min(k, len(pool))
            picked += [(s, i) for i in rng.sample(pool, k)]
        manifest['datasets'][ds] = dict(
            n_ids=len(picked), alloc=alloc,
            terciles=[float(q1), float(q2)] if q1 is not None else 'bands 0 / (0,10) / 10',
            stratum_pool={s: len(v) for s, v in strata.items()},
            final_arm=cfg['final_name'])
        for s, i in picked:
            for arm, src in (('A', base[i]), ('B', fin[i])):
                manifest['items'].append(dict(
                    item_id=f'{ds[:3]}-{i}-{arm}', dataset=ds, sample_id=i,
                    arm='base' if arm == 'A' else 'final', stratum=s,
                    question=src.get('question'), gt=src.get('gt', src.get('ground_truth')),
                    pred=src.get('pred', src.get('response')),
                    group=src.get('group_key', src.get('sub_category')),
                    gavie_rel=(src.get('eval_metrics') or {}).get('GAVIE-Rel'),
                    gavie_acc=(src.get('eval_metrics') or {}).get('GAVIE-Acc'),
                    token_f1=(src.get('eval_metrics') or {}).get('Token-F1')))
    out = os.path.join(ROOT, 'outputs/human_eval_manifest.json')
    with open(out, 'w') as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
    # summary
    import collections
    tab = collections.Counter((it['dataset'], it['stratum']) for it in manifest['items'])
    print('items:', len(manifest['items']))
    for ds, cfg in ARMS.items():
        info = manifest['datasets'][ds]
        ter = info['terciles']
        ter_s = ter if isinstance(ter, str) else '%.2f / %.2f' % tuple(ter)
        print(f"{ds:10s} ids={info['n_ids']} alloc={info['alloc']} terciles={ter_s}")
    print('written', out)


if __name__ == '__main__':
    main()
