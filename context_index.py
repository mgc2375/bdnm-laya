"""Build the maths context index that `bdnm.py context` searches.

    .venv/bin/python context_index.py [--naturalproofs PATH_TO_naturalproofs_proofwiki.json]

Sources, all embedded with GTR-T5 (the space the nodes and vec2text live in):
  context/library.jsonl   theorems with their hypotheses (each hypothesis is its own entry, so a gap can land on it)
  data/chains_*.jsonl     collar record steps, and traps with the reason they fail
  NaturalProofs           ProofWiki theorem and definition statements (optional; the raw file isn't kept)
Writes context/index.npz (float16 vectors) and context/texts.jsonl.
"""
import argparse, glob, json, re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CTX = ROOT / "context"


def library():
    for line in open(CTX / "library.jsonl"):
        if not line.strip():
            continue
        e = json.loads(line)
        yield {"text": f"{e['name']}: {e['statement']}", "kind": "theorem", "src": "library"}
        for r in e.get("requires", []):
            yield {"text": f"{e['name']} requires {r}", "kind": "requirement", "src": "library"}


def collar():
    for f in sorted(glob.glob(str(ROOT / "data" / "chains_*.jsonl"))):
        if f.endswith("naturalproofs.jsonl"):
            continue
        for line in open(f):
            try:
                c = json.loads(line)
            except json.JSONDecodeError:
                continue
            for s in c.get("steps", []):
                yield {"text": s, "kind": "step", "src": c.get("id", "")}
            for t in c.get("traps", []):
                if t.get("text"):
                    yield {"text": f"{t['text']} -- fails: {t.get('why', '')}", "kind": "trap", "src": c.get("id", "")}


def naturalproofs(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("pn", ROOT / "gpu" / "bundle" / "prep_naturalproofs.py")
    pn = importlib.util.module_from_spec(spec); spec.loader.exec_module(pn)
    ds = json.loads(Path(path).read_text())["dataset"]
    for kind, items in (("theorem", ds.get("theorems", [])), ("definition", ds.get("definitions", []))):
        for it in items:
            s = pn.clean(" ".join(it.get("contents", [])))
            if 6 <= len(s.split()) <= 90:
                yield {"text": f"{it.get('title', '')}: {s}", "kind": kind, "src": "proofwiki"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--naturalproofs")
    a = ap.parse_args()
    import bdnm
    rows = list(library()) + list(collar()) + (list(naturalproofs(a.naturalproofs)) if a.naturalproofs else [])
    seen, uniq = set(), []
    for r in rows:
        if r["text"] not in seen:
            seen.add(r["text"]); uniq.append(r)
    print(f"embedding {len(uniq)} statements", flush=True)
    vecs = []
    for i in range(0, len(uniq), 256):
        vecs.append(bdnm.embed([r["text"] for r in uniq[i:i + 256]]))
        if i % 5120 == 0:
            print(f"  {i}/{len(uniq)}", flush=True)
    V = np.vstack(vecs)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    np.savez(CTX / "index.npz", V=V.astype(np.float16))
    with open(CTX / "texts.jsonl", "w") as fh:
        for r in uniq:
            fh.write(json.dumps(r) + "\n")
    kinds = {}
    for r in uniq:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    print(f"wrote context/index.npz ({V.shape[0]} x {V.shape[1]}), kinds: {kinds}")


if __name__ == "__main__":
    main()
