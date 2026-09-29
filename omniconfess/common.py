# -*- coding: utf-8 -*-
"""cf_common.py -- shared spine for the Qwen-Omni CF-Confess judgement family.

The five judgement entrypoints (cf_confess_phd.py, cf_confess_pubmedqa.py,
cf_confess_cmm.py, cf_confess_cmm_qwen3.py, cf_confess_qwen3.py) all share the
same scaffolding around a per-dataset CFConfessProcessor + generate() call:
seeding, per-token logprob readout, resume-by-id, the record_process JSONL
attach block, and the time file. Those were byte-identical copies in every
file; they live here now as the single source of truth.

DISCIPLINE -- every function below is a verbatim lift of the inlined version it
replaces (same math, same dict key order, same rounding, same format strings),
so the JSONL rows stay byte-for-byte identical to the research outputs. The
per-dataset essence (view construction, the Stage-2/3 decision math in each
CFConfessProcessor, the generate() call, the method label) stays explicit in
each entrypoint.

Out of scope on purpose: the Nemotron arm (different HF backbone, generate-only
offline-equivalent path, its own token_logprobs from base_model_nemotron_hf)
and the RAGTruth free-generation arm (token-by-token loop, different record
schema) keep their own self-contained helpers -- see cf_confess_nemotron_hf.py
and cf_confess_ragtruth.py. Stage-3 decision rules live in gamma_operator.py.
"""
import os
import json
import time
import random

import numpy as np
import torch

# The misleading-context marker peeled off the phd/qwen3 question to build the
# context-removed counterfactual view (Stage 2). Kept here so the phd (Qwen2.5)
# and qwen3 arms share one definition.
MARKER = ("In case there is an inconsistency between the context and the image "
          "content, you should follow the image.")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_question(q):
    """(full, question-only): peel the misleading context off at the marker."""
    idx = q.find(MARKER)
    if idx < 0:
        return q, q
    return q, q[idx + len(MARKER):].lstrip()


def token_logprobs(gen_ids, scores):
    """Per-token logprob of the realised sequence. NaN at the forced decision
    step (its logit is -inf under the hard intervention)."""
    out = []
    for tid, logits in zip(gen_ids.tolist(), scores):
        logits = logits.float()
        if logits.dim() > 1:
            logits = logits[0]
        if not torch.isfinite(logits[tid]):
            lp = float('nan')      # decision step under intervention
        else:
            lp = torch.log_softmax(logits, dim=-1)[tid]
        out.append(round(lp.item(), 4))
    return out


def load_done_ids(output_path):
    """Resume-by-id: ids already present in the output jsonl."""
    done_ids = set()
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line.strip())['id'])
                except Exception:
                    pass
        print(f'[resume] skipping {len(done_ids)} already-done samples', flush=True)
    return done_ids


def attach_record(result, rec_extra, gen_ids, logp_list, input_len,
                  record_process):
    """Attach the CF record + trajectory logprobs to a result row, in the exact
    key order the research outputs use (JSONL is byte-compared). No-op fields
    when record_process is off or the sample errored (gen_ids is None)."""
    if record_process:
        result['record_process'] = True
        result['cf'] = rec_extra
        if gen_ids is not None:
            cum = float(np.nansum(logp_list))
            result['token_ids'] = gen_ids.tolist()
            result['logp_list'] = logp_list
            result['cumulative_logprob'] = round(cum, 4)
            result['norm_logp'] = round(cum / (len(gen_ids) + 1e-8), 4)
            result['input_len'] = int(input_len)
            result['output_tokens'] = int(len(gen_ids))
    return result


def write_time(time_path, file_name, t_start, total_tokens):
    """Write the per-shard timing sidecar (total_time only volatile field)."""
    with open(os.path.join(time_path, file_name + '.txt'), 'w') as f:
        f.write(f'total_time: {time.time()-t_start:.2f}s\n'
                f'total_output_tokens: {total_tokens}\n')
