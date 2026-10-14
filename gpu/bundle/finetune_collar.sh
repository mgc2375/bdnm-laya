#!/usr/bin/env bash
# Second pass after run_h200.sh: continue from adapter/ on the collar chains only, at a low learning rate,
# to win back trap rejection (ProofWiki has no traps). ~10 min on an H200.
#   bash finetune_collar.sh        -> adapter_ft/ (keep whichever of adapter/ and adapter_ft/ scores better)
set -euo pipefail
export HF_HUB_OFFLINE=0
cd "$(dirname "$0")"
mkdir -p data_collar
python - <<'PY'
import json
rows = [l for l in open("data/pairs_train.jsonl") if json.loads(l).get("domain") == "collar"]
open("data_collar/pairs_train.jsonl", "w").writelines(rows)
print(len(rows), "collar training pairs")
PY
cp data/pairs_val.jsonl data_collar/pairs_val.jsonl
python train.py --data data_collar --init adapter --out adapter_ft --amp --bs 32 --lr "${LR:-1e-4}" --epochs "${EPOCHS:-3}" \
    2>&1 | grep --line-buffered -v -i "warn" | tee finetune.log
echo "compare: adapter/meta.json vs adapter_ft/meta.json (val_auc_collar, val_auc_vs_trap, blind_bd_auc)"
