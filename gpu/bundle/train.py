"""Fine-tune Laya's "can 'to' follow immediately from 'from'?" judgement with LoRA.

    .venv/bin/python train.py [--epochs 3] [--bs 8] [--lr 1e-3] [--rank 16] [--top 8]
    python train.py --data data --out adapter --amp --bs 32 --top 28     # on a CUDA GPU (see gpu/README.md)

Uses Laya's own forward path (the same tokenisation, markers and yes/no logits as predict()), LoRA on
the ModernBERT encoder (Wqkv, Wo, Wi) and the decision head (linear1, linear2), and trains the scorer fully.
Writes only the trainable weights to adapter/laya_bdnm.safetensors (+ adapter/meta.json).
Evaluates on held-out chains and, blind, on work/bd-esa (Claude's judged links as the answer key).
"""
import argparse, json, math, os, random, time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
ADAPTER = ROOT / "adapter"
AMP = False                   # mixed precision on CUDA (--amp): bf16 where supported, else fp16 + grad scaling (T4)


def amp_dtype():
    return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
TOP = 8                       # LoRA only on the top TOP of 28 encoder layers: backward stops there (~3x faster)


def targets(top=TOP):
    first = 28 - top
    layers = "|".join(str(i) for i in range(first, 28))
    return rf"^(encoder\.layers\.({layers})\.(attn\.(Wqkv|Wo)|mlp\.(Wi|Wo))|head\.layers\.\d+\.(linear1|linear2))$"
QUESTION = {"type": "choice",
            "instructions": "Once 'from' is done or true, can 'to' be done or made true immediately, "
                            "with no other step needed in between?",
            "criteria": {"yes": "yes, 'to' follows immediately from 'from'",
                         "no": "no, other steps are needed first, or 'from' does not help with 'to'"}}


def load_agent(adapter=True):
    """Laya with the bdnm LoRA adapter applied (if trained). Returns (agent, temperature)."""
    import laya
    from peft import LoraConfig, inject_adapter_in_model
    agent = laya.load("convaiinnovations/laya")
    t = agent.cfg["temperature_by_options"]["choice:2"]
    f = ADAPTER / "laya_bdnm.safetensors"
    if adapter and f.exists():
        from safetensors.torch import load_file
        meta = json.loads((ADAPTER / "meta.json").read_text())
        inject_adapter_in_model(LoraConfig(r=meta["rank"], lora_alpha=2 * meta["rank"],
                                           target_modules=targets(meta.get("top", 28)),
                                           lora_dropout=0.0), agent.model)
        missing = agent.model.load_state_dict(load_file(str(f)), strict=False)
        assert not missing.unexpected_keys, missing.unexpected_keys
        agent.model.to(agent.device).eval()
        t = meta["temperature"]
    return agent, t


def batch_logits(agent, pairs):
    """yes-minus-no logit (before temperature) for [(from, to), ...], with gradients if enabled."""
    from laya.common import collate_items
    internal = {"next": agent._to_internal(QUESTION)}
    items = [agent._encode_state({"from": a, "to": b}, ["next"], internal) for a, b in pairs]
    b = collate_items(items, agent.tok.pad_token_id)
    with torch.autocast("cuda", dtype=amp_dtype(), enabled=AMP and agent.device.type == "cuda"):
        logits, _ = agent.model(*(b[k].to(agent.device) for k in
                                  ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")))
    return logits[:, 0].float() - logits[:, 1].float()


@torch.no_grad()
def scores(agent, pairs, bs=32):
    agent.model.eval()
    out = []
    for i in range(0, len(pairs), bs):
        out.extend(batch_logits(agent, pairs[i:i + bs]).cpu().tolist())
    return np.array(out)


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not len(pos) or not len(neg):
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def read(f):
    return [json.loads(l) for l in open(f) if l.strip()]


def blind_set():
    """work/bd-esa: nodes + Claude's judged links (p >= 0.5 good, p <= 0.2 bad)."""
    g = json.loads((ROOT / "work/bd-esa/graph.json").read_text())
    j = json.loads((ROOT / "work/bd-esa/judge_r1.json").read_text())
    T = {k: n["text"] for k, n in g["nodes"].items()}
    good = [(T[k.split("->")[0]], T[k.split("->")[1]]) for k, v in j.items() if v["p"] >= 0.5]
    bad = [(T[k.split("->")[0]], T[k.split("->")[1]]) for k, v in j.items() if v["p"] <= 0.2]
    return good, bad


def evaluate(agent, val, tag, bs=32):
    sv = scores(agent, [(r["from"], r["to"]) for r in val], bs=bs)
    y = np.array([r["y"] for r in val])
    kinds = np.array([r["kind"] for r in val])
    dom = np.array([r.get("domain", "collar") for r in val])
    res = {"val_auc": auc(sv[y >= 0.5], sv[y == 0.0])}
    for d in ("collar", "general"):
        if (dom == d).any():
            res[f"val_auc_{d}"] = auc(sv[(y >= 0.5) & (dom == d)], sv[(y == 0.0) & (dom == d)])
    for k in ("start-goal", "trap", "reverse", "skip2+", "cross"):
        res[f"val_auc_vs_{k}"] = auc(sv[y >= 0.5], sv[kinds == k])
    good, bad = blind_set()
    res["blind_bd_auc"] = auc(scores(agent, good), scores(agent, bad))
    print(f"  [{tag}] " + "  ".join(f"{k}={v:.3f}" for k, v in res.items()), flush=True)
    return res, sv, y


def fit_temperature(sv, y):
    """1-D search for the temperature that minimises validation log-loss on the yes/no logit."""
    best = (1e9, 1.0)
    for t in np.exp(np.linspace(np.log(0.03), np.log(10), 120)):
        p = 1 / (1 + np.exp(-sv / t))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        ll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        best = min(best, (ll, float(t)))
    return best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--top", type=int, default=TOP)
    ap.add_argument("--skip-base", action="store_true")
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ADAPTER))
    ap.add_argument("--amp", action="store_true", help="bf16 autocast (CUDA)")
    ap.add_argument("--eval-every", type=int, default=0, help="also evaluate every N steps (big datasets)")
    ap.add_argument("--init", help="continue from a trained adapter dir (its rank/top are reused)")
    a = ap.parse_args()
    global AMP
    AMP = a.amp
    out = Path(a.out)
    random.seed(0); torch.manual_seed(0)
    from peft import LoraConfig, inject_adapter_in_model
    from safetensors.torch import save_file

    train = read(Path(a.data) / "pairs_train.jsonl")
    val = read(Path(a.data) / "pairs_val.jsonl")
    print(f"{len(train)} train pairs, {len(val)} val pairs", flush=True)
    agent, t0 = load_agent(adapter=False)
    agent.amp_enabled = False
    ebs = 4 * a.bs
    base = {} if a.skip_base else evaluate(agent, val, "base Laya", ebs)[0]

    if a.init:
        meta0 = json.loads((Path(a.init) / "meta.json").read_text())
        a.rank, a.top = meta0["rank"], meta0.get("top", 28)
    inject_adapter_in_model(LoraConfig(r=a.rank, lora_alpha=2 * a.rank, target_modules=targets(a.top), lora_dropout=0.0),
                            agent.model)
    if a.init:
        from safetensors.torch import load_file
        r = agent.model.load_state_dict(load_file(str(Path(a.init) / "laya_bdnm.safetensors")), strict=False)
        assert not r.unexpected_keys, r.unexpected_keys
        print(f"  continuing from {a.init} (rank {a.rank}, top {a.top})", flush=True)
    for n, p in agent.model.named_parameters():
        p.requires_grad = ("lora_" in n) or n.startswith("scorer.")
    agent.model.to(agent.device)
    params = [p for p in agent.model.parameters() if p.requires_grad]
    print(f"  trainable {sum(p.numel() for p in params) / 1e6:.2f}M params", flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    steps = a.epochs * math.ceil(len(train) / a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.06)
    scaler = torch.amp.GradScaler("cuda", enabled=AMP and agent.device.type == "cuda" and amp_dtype() == torch.float16)
    print(f"  device {agent.device}, mixed precision: {amp_dtype() if AMP and agent.device.type == 'cuda' else 'off'}",
          flush=True)
    npos = sum(r["y"] >= 0.5 for r in train)
    pos_w = (len(train) - npos) / max(npos, 1)          # balance positives against negatives

    best, step = None, 0
    out.mkdir(parents=True, exist_ok=True)
    key = "val_auc_collar" if any(r.get("domain") == "general" for r in val) else "val_auc"

    def checkpoint(tag):
        nonlocal best
        res, sv, yv = evaluate(agent, val, tag, ebs)
        if best is None or res[key] > best[key]:      # select on held-out chains; the blind test stays blind
            best = res
            temp = fit_temperature(sv, yv)
            state = {n: p.detach().float().cpu().contiguous() for n, p in agent.model.named_parameters() if p.requires_grad}
            save_file(state, str(out / "laya_bdnm.safetensors"))
            (out / "meta.json").write_text(json.dumps({"rank": a.rank, "top": a.top, "temperature": temp, "at": tag,
                                                      "base": base, "result": res}, indent=1))
            print(f"  saved adapter ({tag}, temperature {temp:.2f})", flush=True)

    for ep in range(a.epochs):
        random.shuffle(train)
        agent.model.eval()          # no dropout: with this little data its noise swamped the signal
        t = time.time(); run = 0.0
        for i in range(0, len(train), a.bs):
            b = train[i:i + a.bs]
            z = batch_logits(agent, [(r["from"], r["to"]) for r in b]) / t0
            y = torch.tensor([r["y"] for r in b], device=z.device)
            w = torch.where(y >= 0.5, torch.full_like(y, pos_w), torch.ones_like(y))
            loss = (w * torch.nn.functional.binary_cross_entropy_with_logits(z, y, reduction="none")).mean()
            opt.zero_grad(); scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update(); sched.step(); step += 1
            run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
            if step % (10 if steps < 5000 else 100) == 0:
                print(f"  epoch {ep + 1} step {step}/{steps} loss {run:.3f} ({(time.time() - t) / (i // a.bs + 1):.2f}s/step)",
                      flush=True)
            if a.eval_every and step % a.eval_every == 0:
                checkpoint(f"step {step}")
        checkpoint(f"epoch {ep + 1}")
    print("base :", json.dumps(base))
    print("best :", json.dumps(best))


if __name__ == "__main__":
    main()
