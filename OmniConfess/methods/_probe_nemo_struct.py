# -*- coding: utf-8 -*-
"""Probe Nemotron LLM structure: layer list, norm, lm_head, forward logits access."""
import os, sys, torch
REPO = os.environ.get('NEMO_REPO', '/home/tsq/rhq/omniconfess/OmniConfess')
sys.path.insert(0, os.path.join(REPO, 'OmniConfess', 'methods'))
from base_model_nemotron_hf_pv import _repair_buffers, DEFAULT_MODEL
from transformers import AutoModel, AutoProcessor

mp = os.environ.get('NEMOTRON_MODEL_PATH', DEFAULT_MODEL)
DEVICE = 'cuda:0'
print('[load]', mp, flush=True)
processor = AutoProcessor.from_pretrained(mp, trust_remote_code=True)
model = AutoModel.from_pretrained(mp, trust_remote_code=True, torch_dtype=torch.bfloat16,
                                  low_cpu_mem_usage=True, attn_implementation='eager')
model.to(DEVICE).eval()
_repair_buffers(model, mp, DEVICE)
tk = processor.tokenizer

lm = model.language_model
print('language_model type:', type(lm).__name__)
print('lm children:', [n for n,_ in lm.named_children()])
bb = lm.backbone
print('backbone type:', type(bb).__name__)
print('backbone children:', [n for n,_ in bb.named_children()])
# find layers container
for cand in ['layers','blocks','h']:
    if hasattr(bb, cand):
        layers = getattr(bb, cand)
        print(f'backbone.{cand}: len={len(layers)}')
        # classify layer types
        types = [type(l).__name__ for l in layers]
        print('layer types (first 8):', types[:8])
        from collections import Counter
        print('type counts:', Counter(types))
        # which are attention layers? look for attributes
        for i,l in enumerate(layers[:6]):
            print(f'  L{i} {type(l).__name__} children={[n for n,_ in l.named_children()]}')
        break
# final norm?
for cand in ['norm','norm_f','final_layernorm','ln_f']:
    if hasattr(bb, cand):
        print('final norm attr:', cand, type(getattr(bb,cand)).__name__)
print('lm_head:', type(lm.lm_head).__name__ if hasattr(lm,'lm_head') else 'NONE')

# Try a text-only forward to see logits shape & output type
msgs=[{'role':'user','content':'Answer yes or no: is the sky blue?'}]
text=tk.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
inp=processor(text=[text], return_tensors='pt')
inp={k:(v.to(DEVICE) if torch.is_tensor(v) else v) for k,v in inp.items()}
for mk in ('num_patches','num_tokens','imgs_sizes'): inp.pop(mk,None)
print('input keys:', list(inp.keys()))
with torch.inference_mode():
    out = model(**{k:v for k,v in inp.items()}, use_cache=True, output_hidden_states=True)
print('forward out type:', type(out).__name__)
print('has logits:', hasattr(out,'logits'), 'logits shape:', tuple(out.logits.shape) if hasattr(out,'logits') else None)
print('has past_key_values:', hasattr(out,'past_key_values') and out.past_key_values is not None)
print('has hidden_states:', hasattr(out,'hidden_states') and out.hidden_states is not None)
if hasattr(out,'hidden_states') and out.hidden_states is not None:
    print('num hidden_states:', len(out.hidden_states), 'each shape:', tuple(out.hidden_states[0].shape))
# test lm_head on a hidden state through final norm
hs = out.hidden_states[10]  # mid layer
try:
    fn = getattr(bb, 'norm', None) or getattr(bb,'norm_f',None)
    normed = fn(hs)
    logits_e = lm.lm_head(normed)
    print('early-exit projection OK, shape', tuple(logits_e.shape))
except Exception as e:
    print('early-exit projection FAIL:', repr(e))
print('[done]')
