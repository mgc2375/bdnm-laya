#!/usr/bin/env bash
# Train the bdnm Laya adapter on any CUDA machine (RunPod, Lambda, Colab, vast.ai...).
#   bash run.sh            -> adapter/laya_bdnm.safetensors + adapter/meta.json
# Knobs (env vars): MAX_NP=8000 COLLAR_REPEAT=3 EPOCHS=2 BS=32 TOP=28 LR=5e-4 EVAL_EVERY=2000
set -euo pipefail
export HF_HUB_OFFLINE=0          # train.py defaults to offline (right on the Mac); the GPU box must download Laya
cd "$(dirname "$0")"
case "$PWD" in /teamspace/lightning_storage*) echo "run from ~ (the Studio home), not lightning_storage: it cannot write the weights file"; exit 1;; esac
pip install -q "laya==0.3.20" torch "transformers>=4.50,<5" peft safetensors numpy
python prep_naturalproofs.py --out data/chains_naturalproofs.jsonl --max "${MAX_NP:-8000}"
python build_data.py --data data --collar-repeat "${COLLAR_REPEAT:-3}"
python train.py --data data --out "${OUT:-adapter}" --amp --bs "${BS:-16}" --top "${TOP:-28}" --lr "${LR:-5e-4}" \
    --epochs "${EPOCHS:-2}" --eval-every "${EVAL_EVERY:-2000}" 2>&1 | grep --line-buffered -v -i "warn" | tee train.log
echo "done: copy adapter/ back to ~/bdnm-laya/adapter/ on the Mac"
