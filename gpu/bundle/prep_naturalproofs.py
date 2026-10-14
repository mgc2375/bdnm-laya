"""NaturalProofs (ProofWiki, CC BY-SA) -> proof chains in the same format as the collar chains.

    python prep_naturalproofs.py --out data/chains_naturalproofs.jsonl [--max 8000]

Downloads naturalproofs_proofwiki.json (117 MB, Zenodo record 4902289) if it isn't already here.
Each chain = [theorem statement, proof line 1, proof line 2, ...]: consecutive proof lines are the
"follows immediately" positives. ProofWiki wiki markup is stripped; trivial lines ({{qed}}, eqn
braces, "Hence the result") are dropped. No traps: negatives come from skips, reversals and other proofs.
"""
import argparse, json, random, re, urllib.request
from pathlib import Path

URL = "https://zenodo.org/records/4902289/files/naturalproofs_proofwiki.json?download=1"
TRIVIAL = re.compile(r"^(\{\{(qed|begin-eqn|end-eqn|begin-axiom|end-axiom|explain|questionable|proofread)[^}]*\}\}|"
                     r"(hence|thus|so) the result( follows)?\.?|the result follows\.?|$)", re.I)


def clean(line):
    s = line
    s = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", s)                 # [[Page|text]] -> text
    s = re.sub(r"\{\{eqn\s*\|\s*(l|ll)\s*=", "", s)                          # {{eqn | l = .. | o = = | r = .. | c = ..}}
    s = re.sub(r"\{\{eqn\s*\|", "", s)
    s = re.sub(r"\|\s*c\s*=", " ; ", s)                                     # the justification comment
    s = re.sub(r"\|\s*(l|r|o|ll|rr|n)\s*=", " ", s)
    s = s.replace("}}", "")
    s = re.sub(r"\{\{(?:[A-Za-z]+)\|([^}]*)\}\}", r"\1", s)                  # {{Defof|X}} -> X
    s = re.sub(r"\{\{[^}]*\}\}", "", s)
    s = s.replace("$", "").replace("\\ds", "").replace("\\displaystyle", "")
    s = re.sub(r"'''?", "", s)
    s = re.sub(r"\s+", " ", s).strip(" :;")
    return s


def chains(data, max_chains, seed=0):
    out = []
    for thm in data["dataset"]["theorems"]:
        stmt = clean(" ".join(thm.get("contents", [])))
        if not (8 <= len(stmt.split()) <= 120):
            continue
        for k, pf in enumerate(thm.get("proofs", [])):
            lines = [clean(l) for l in pf.get("contents", [])]
            lines = [l for l in lines if not TRIVIAL.match(l) and 3 <= len(l.split()) <= 80]
            if len(lines) < 3:
                continue
            steps = [f"{thm.get('title', '')}: {stmt}"] + lines[:11]
            out.append({"id": f"np-{thm['id']}-{k}", "source": "NaturalProofs/ProofWiki",
                        "domain": "proofwiki", "steps": steps, "traps": []})
    random.Random(seed).shuffle(out)
    return out[:max_chains]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/chains_naturalproofs.jsonl")
    ap.add_argument("--raw", default="naturalproofs_proofwiki.json")
    ap.add_argument("--max", type=int, default=8000)
    a = ap.parse_args()
    raw = Path(a.raw)
    if not raw.exists():
        print(f"downloading {URL}", flush=True)
        urllib.request.urlretrieve(URL, raw)
    cs = chains(json.loads(raw.read_text()), a.max)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as fh:
        for c in cs:
            fh.write(json.dumps(c) + "\n")
    print(f"{len(cs)} NaturalProofs chains -> {a.out} "
          f"({sum(len(c['steps']) - 1 for c in cs)} consecutive-step pairs)")


if __name__ == "__main__":
    main()
