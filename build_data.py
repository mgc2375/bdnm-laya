"""Chains (DATA/chains_*.jsonl) -> labelled step pairs (DATA/pairs_train.jsonl, DATA/pairs_val.jsonl).

    python build_data.py [--data data] [--collar-repeat 1]

Label = P('to' follows immediately from 'from'):
  consecutive step            1.0
  skip one step               0.15   (right direction, one step missing)
  skip two or more / S->goal  0.0    (the "anything leads to the goal" bias)
  reversed order              0.0
  trap after its step         0.0    (tempting wrong move), trap -> next step 0.0
  step from another chain     0.0    (same domain: hard negative)
Split is by chain and per domain (1/7 of collar chains, 5% of others), so validation chains are unseen.
--collar-repeat k repeats collar training pairs k times so a big general corpus doesn't drown them.
"""
import argparse, glob, json, random
from pathlib import Path


def load_chains(data):
    chains = []
    for f in sorted(glob.glob(str(Path(data) / "chains_*.jsonl"))):
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(c.get("steps"), list) and len(c["steps"]) >= 3:
                c["file"] = Path(f).stem
                c["collar"] = not c["file"].endswith("naturalproofs")
                chains.append(c)
    return chains


def pairs_for(c, pool, rng):
    s = c["steps"]
    out = []
    dom = "collar" if c["collar"] else "general"
    add = lambda a, b, y, kind: out.append({"from": a, "to": b, "y": y, "kind": kind, "chain": c.get("id", ""), "domain": dom})
    for k in range(len(s) - 1):
        add(s[k], s[k + 1], 1.0, "next")
        add(s[k + 1], s[k], 0.0, "reverse")
        if k + 2 < len(s):
            add(s[k], s[k + 2], 0.15, "skip1")
        far = list(range(k + 3, len(s)))
        for j in rng.sample(far, min(2, len(far))):
            add(s[k], s[j], 0.0, "skip2+")
    if len(s) > 3:
        add(s[0], s[-1], 0.0, "start-goal")
    for t in c.get("traps", []):
        i = t.get("after")
        if not isinstance(i, int) or not (0 <= i < len(s)) or not t.get("text"):
            continue
        add(s[i], t["text"], 0.0, "trap")
        if i + 1 < len(s):
            add(t["text"], s[i + 1], 0.0, "trap-next")
        add(t["text"], s[-1], 0.0, "trap-goal")
    for k in range(len(s) - 1):
        o = pool[rng.randrange(len(pool))]
        if o is c:
            continue
        add(s[k], rng.choice(o["steps"]), 0.0, "cross")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path(__file__).resolve().parent / "data"))
    ap.add_argument("--collar-repeat", type=int, default=1)
    a = ap.parse_args()
    rng = random.Random(0)
    chains = load_chains(a.data)
    rng.shuffle(chains)
    val = set()
    for collar, frac in ((True, 1 / 7), (False, 0.05)):
        grp = [c for c in chains if c["collar"] == collar]
        val |= {id(c) for c in grp[:max(1, int(len(grp) * frac))] if grp}
    train, valp = [], []
    for c in chains:
        is_val = id(c) in val
        pool = [o for o in chains if (id(o) in val) == is_val and o["collar"] == c["collar"]] or chains
        ps = pairs_for(c, pool, rng)
        if is_val:
            valp += ps
        else:
            train += ps * (a.collar_repeat if c["collar"] else 1)
    for name, rows in (("train", train), ("val", valp)):
        with open(Path(a.data) / f"pairs_{name}.jsonl", "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
    n_col = sum(c["collar"] for c in chains)
    print(f"{len(chains)} chains ({n_col} collar, {len(chains) - n_col} general; {len(val)} held out); "
          f"{len(train)} train pairs, {len(valp)} val pairs")


if __name__ == "__main__":
    main()
