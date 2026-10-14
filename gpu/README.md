# Train the bdnm Laya judge on a cloud GPU

Everything the run needs is in `bundle/` (208 KB): the training code, your 85 collar chains, and the
blind BD test. NaturalProofs (117 MB, ProofWiki proofs) downloads on the GPU side, not your Mac.

What it trains: a LoRA adapter on all 28 of Laya's encoder layers plus the head. It answers "can 'to'
follow immediately from 'from'?" on ~8,000 ProofWiki proofs plus your collar chains (repeated 3x so
they aren't drowned out). It picks the best checkpoint on held-out *collar* chains; the BD (HA-T)
problem is never trained on or selected on, so `blind_bd_auc` stays a genuine blind test.

## Option A: Modal (API, pay per second)
```
pip install modal
modal setup                       # once: logs in through the browser
cd ~/bdnm-laya/gpu
modal run modal_app.py            # A10G, 8000 proofs, 2 epochs
```
Bigger run: `BDNM_GPU=L40S modal run modal_app.py --max-np 20000 --epochs 3`.
The adapter lands in `gpu/out/`. Rough cost: A10G is about $1.10/h, and the default run should be about 1.5-2 h.

## Option B: any CUDA box (RunPod, Lambda, vast.ai, Colab)
Upload `bdnm_gpu_bundle.zip`, then:
```
unzip bdnm_gpu_bundle.zip && cd bundle && bash run.sh
```
Knobs: `MAX_NP=20000 EPOCHS=3 BS=64 bash run.sh`. The adapter lands in `bundle/adapter/`.

## Use the result on the Mac
```
cp out/laya_bdnm.safetensors out/meta.json ~/bdnm-laya/adapter/      # Modal
```
`bdnm.py` loads it automatically for problems created with `init --technical`.

## Reading the log
Each evaluation line shows AUC (0.5 = guessing, 1.0 = perfect):
- `val_auc_collar`: unseen collar chains. The local Mac run reached 0.666.
- `val_auc_general`: unseen ProofWiki proofs.
- `val_auc_vs_reverse`: can it tell step order? It was stuck at 0.51 locally.
- `blind_bd_auc`: the (HA-T) problem, never seen. It was 0.56 locally.

`meta.json` records the base and best scores.
