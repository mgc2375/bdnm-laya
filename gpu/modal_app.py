"""Train the bdnm Laya adapter on a cloud GPU through Modal.

    pip install modal && modal setup          # once: creates your Modal token in the browser
    modal run modal_app.py                    # defaults: T4, 8000 NaturalProofs chains, 2 epochs
    BDNM_GPU=L40S modal run modal_app.py --max-np 20000 --epochs 3

The adapter comes back into ./out/ (and stays in the Modal volume "bdnm-laya-out").
Copy out/laya_bdnm.safetensors + out/meta.json into ~/bdnm-laya/adapter/ to use it locally.
"""
import os, subprocess
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
app = modal.App("bdnm-laya-train")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("laya==0.3.20", "torch", "transformers>=4.50,<5", "peft", "safetensors", "numpy")
         .add_local_dir(HERE / "bundle", remote_path="/root/bdnm"))
vol = modal.Volume.from_name("bdnm-laya-out", create_if_missing=True)
GPU = os.environ.get("BDNM_GPU", "T4")        # T4 (cheapest, fp16), A10G, L4, L40S, A100


@app.function(gpu=GPU, image=image, timeout=8 * 3600, volumes={"/out": vol})
def train(max_np, collar_repeat, epochs, bs, top, lr, eval_every):
    def sh(*cmd):
        print("$", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd="/root/bdnm", check=True)

    sh("python", "prep_naturalproofs.py", "--out", "data/chains_naturalproofs.jsonl", "--max", str(max_np))
    sh("python", "build_data.py", "--data", "data", "--collar-repeat", str(collar_repeat))
    sh("python", "train.py", "--data", "data", "--out", "/out/adapter", "--amp", "--bs", str(bs), "--top", str(top),
       "--lr", str(lr), "--epochs", str(epochs), "--eval-every", str(eval_every))
    vol.commit()
    return {f.name: f.read_bytes() for f in Path("/out/adapter").iterdir()}


@app.local_entrypoint()
def main(max_np: int = 8000, collar_repeat: int = 3, epochs: int = 2, bs: int = 16, top: int = 28,
         lr: float = 5e-4, eval_every: int = 2000):
    print(f"training on {GPU}", flush=True)
    files = train.remote(max_np, collar_repeat, epochs, bs, top, lr, eval_every)
    out = HERE / "out"
    out.mkdir(exist_ok=True)
    for name, data in files.items():
        (out / name).write_bytes(data)
    print(f"adapter saved to {out}: {sorted(files)}")
    print("use it: cp out/laya_bdnm.safetensors out/meta.json ~/bdnm-laya/adapter/")
