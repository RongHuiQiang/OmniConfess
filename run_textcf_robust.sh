#!/bin/bash
# P6: text-channel CF neutralization-scheme robustness runs (card 7, serial)
set -u
export CUDA_VISIBLE_DEVICES=7
cd /home/tsq/rhq/omniconfess/OmniConfess
run () {  # script scheme out
  echo "=== [$(date +%F\ %T)] $1 --scheme $2 -> $3 ==="
  /ultrafast_share/tsq/venvs/omniconfess/bin/python "$1" --scheme "$2" --out "$3" >> "outputs/cf_probe/${4}.log" 2>&1
  echo "=== [$(date +%F\ %T)] exit=$? : $3 ==="
  nvidia-smi --query-gpu=index,memory.used --format=csv -i 7
}
run probe_textcf_haloquest.py  shuffle outputs/cf_probe/haloquest_textcf_shuffle.jsonl haloquest_textcf_shuffle
run probe_textcf_haloquest.py  empty   outputs/cf_probe/haloquest_textcf_empty.jsonl   haloquest_textcf_empty
run probe_cot_cf_cmm_text.py   shuffle outputs/cf_probe/cmm_textcf_shuffle.jsonl        cmm_textcf_shuffle
run probe_cot_cf_cmm_text.py   empty   outputs/cf_probe/cmm_textcf_empty.jsonl          cmm_textcf_empty
run probe_cot_cf_phd_text.py   shuffle outputs/cf_probe/phd_textcf_shuffle.jsonl        phd_textcf_shuffle
run probe_cot_cf_phd_text.py   empty   outputs/cf_probe/phd_textcf_empty.jsonl          phd_textcf_empty
run probe_cf_avhbench_text.py  shuffle outputs/cf_probe/avh_probe_text_shuffle.jsonl    avh_probe_text_shuffle
run probe_cf_avhbench_text.py  empty   outputs/cf_probe/avh_probe_text_empty.jsonl      avh_probe_text_empty
echo "=== ALL DONE $(date +%F\ %T) ==="
