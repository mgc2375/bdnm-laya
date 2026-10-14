#!/usr/bin/env bash
# Lightning AI Studio on an H200 (or H100): bigger data, bigger batch, bf16.
#   bash run_h200.sh        -> adapter/laya_bdnm.safetensors + adapter/meta.json, log in train.log
cd "$(dirname "$0")"
MAX_NP="${MAX_NP:-20000}" EPOCHS="${EPOCHS:-3}" BS="${BS:-64}" LR="${LR:-5e-4}" EVAL_EVERY="${EVAL_EVERY:-3000}" \
  bash run.sh
