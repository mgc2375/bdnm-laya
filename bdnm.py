"""Bidirectional node mapping with Laya.

Claude (in chat) is the generator: it writes forward/backward branches as JSON into work/<name>/.
This tool vectorises nodes (GTR-T5), proposes forward<->backward links, has Laya score every edge,
solves the probability matrix for the best S->G path, and decodes hard links into waypoint steps
(vec2text inverts interpolated vectors between the two nodes; Claude rewrites the drafts; this tool checks them).

    bdnm.py init NAME --start "..." --goal "..." [--problem "..."]
    bdnm.py add NAME branches.json        [{"from": "S", "side": "fwd", "text": "..."}, ...]
    bdnm.py score NAME                    embed, propose links, Laya-score new edges, print frontier
    bdnm.py candidates NAME [--k 3]       shortlist edges for Claude to judge -> work/NAME/judge_todo.json
    bdnm.py judge NAME scores.json        {"A->B": {"p": 0.8, "gap": "..."}}: Claude's p replaces Laya's
    bdnm.py solve NAME                    best paths, node flow, hard links -> report.md + map.html
    bdnm.py bridge NAME A B [--n 1]       vec2text drafts for waypoints between A and B
    bdnm.py check NAME A B steps.json     verify Claude's waypoint steps, splice them in if they hold
    bdnm.py context NAME A B [--k 6]      real maths statements near the A->B gap (context_index.py builds the index)
    bdnm.py require NAME [--cos 0.7]      hypotheses of theorems used on the winning path become mandatory waypoints
"""
import argparse, hashlib, html, json, math, os, re, sys, time
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).resolve().parent
WORK = ROOT / "work"

MIN_P = 0.50      # calibrated p at or below chance: not a usable edge
HARD_P = 0.70     # a usable edge below this is a hard link that needs decoding
HARD_COS = 0.50   # ...and so is one whose two nodes are far apart in vector space (a big jump, even if Laya likes it)
BATCH = 64
CENTRE_TECHNICAL = False  # the tuned judge scores better raw (0.73 vs 0.66 against Claude's BD labels); centring was
                          # a patch for stock Laya's "sounds like the goal" bias, which the adapter no longer has
TECH_QUANTILE = 0.5       # technical: usable edge = forward-consistent and in the top half of forward edges
TECH_HARD_P = 0.5         # technical: a step the tuned judge thinks more likely "no" than "yes" is hard

# Tested on a labelled pi-wifi edge set (AUC, good vs bad edges): raw Laya 0.74 -> 0.89 once the
# matrix is double-centred (removes "this target sounds good after anything" and "this source leads
# anywhere" bias). The problem text is left out of the state: including it lowered AUC.
EDGE_Q = {"next": {"type": "choice",
                   "instructions": "Once 'from' is done or true, can 'to' be done or made true immediately, "
                                   "with no other step needed in between?",
                   "criteria": {"yes": "yes, 'to' follows immediately from 'from'",
                                "no": "no, other steps are needed first, or 'from' does not help with 'to'"}}}


# ----------------------------------------------------------------------------- storage
def gpath(name):
    return WORK / name / "graph.json"


def load(name):
    return json.loads(gpath(name).read_text())


def save(name, g):
    gpath(name).write_text(json.dumps(g, indent=1))


def ekey(a, b):
    return f"{a}->{b}"


# ----------------------------------------------------------------------------- models (lazy)
_M = {}


def laya_agent():
    """Base Laya, or Laya + the bdnm LoRA adapter (train.py) for technical problems."""
    if "laya" not in _M:
        import train
        t = time.time()
        _M["laya"], _M["temp"] = train.load_agent(adapter=_M.get("technical", False))
        print(f"  Laya{' + bdnm adapter' if _M.get('technical') else ''} loaded in {time.time() - t:.0f}s", flush=True)
    return _M["laya"]


def judge_tag():
    """Cache key for the judge, so a retrained adapter never reuses old scores."""
    f = ROOT / "adapter" / "laya_bdnm.safetensors"
    return f"adapter-{int(f.stat().st_mtime)}" if _M.get("technical") and f.exists() else "base"


def laya_logits(pairs):
    """Logit that 'to' follows immediately from 'from' (temperature-scaled), for [(from_text, to_text), ...]."""
    import train
    agent = laya_agent()
    return [float(z) / _M["temp"] for z in train.scores(agent, pairs, bs=BATCH)] if pairs else []


def sig(z):
    return 1 / (1 + math.exp(-z))


def gtr():
    if "gtr" not in _M:
        import torch
        from transformers import T5EncoderModel, AutoTokenizer
        _M["gtr"] = (T5EncoderModel.from_pretrained("sentence-transformers/gtr-t5-base").eval(),
                     AutoTokenizer.from_pretrained("sentence-transformers/gtr-t5-base"))
    return _M["gtr"]


def embed(texts):
    """GTR-T5 mean-pooled embeddings: the same space vec2text inverts."""
    import torch
    enc, tok = gtr()
    b = tok(list(texts), return_tensors="pt", max_length=128, truncation=True, padding=True)
    with torch.no_grad():
        h = enc(input_ids=b.input_ids, attention_mask=b.attention_mask).last_hidden_state
    m = b.attention_mask.unsqueeze(-1).float()
    return ((h * m).sum(1) / m.sum(1)).numpy()


def node_vecs(name, g):
    """Embeddings cached by text hash in work/<name>/emb.npz."""
    f = WORK / name / "emb.npz"
    cache = dict(np.load(f)) if f.exists() else {}
    h = {k: hashlib.sha1(n["text"].encode()).hexdigest() for k, n in g["nodes"].items()}
    todo = [k for k in g["nodes"] if h[k] not in cache]
    if todo:
        for k, v in zip(todo, embed([g["nodes"][k]["text"] for k in todo])):
            cache[h[k]] = v
        np.savez(f, **cache)
    return {k: cache[h[k]] for k in g["nodes"]}


def invert(vecs, steps=20, beam=4):
    import torch, vec2text
    if "v2t" not in _M:
        _M["v2t"] = vec2text.load_pretrained_corrector("gtr-base")
    cor = _M["v2t"]
    dev = next(cor.model.parameters()).device
    out = vec2text.invert_embeddings(torch.tensor(np.stack(vecs), dtype=torch.float32).to(dev), cor,
                                     num_steps=steps, sequence_beam_width=beam)
    return [re.sub(r"\s+", " ", s).strip() for s in out]


def cos(u, v):
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))


# ----------------------------------------------------------------------------- commands
def cmd_init(a):
    d = WORK / a.name
    if gpath(a.name).exists():
        sys.exit(f"{a.name} already exists")
    d.mkdir(parents=True, exist_ok=True)
    g = {"problem": a.problem or f"Get from: {a.start}  To: {a.goal}",
         "nodes": {"S": {"text": a.start, "side": "start", "depth": 0},
                   "G": {"text": a.goal, "side": "goal", "depth": 0}},
         "edges": {}, "bridges": {}, "technical": bool(a.technical)}
    save(a.name, g)
    print(f"created {d}")


def cmd_add(a):
    g = load(a.name)
    items = json.loads(Path(a.file).read_text())
    n_f = sum(1 for k in g["nodes"] if k.startswith("F"))
    n_b = sum(1 for k in g["nodes"] if k.startswith("B"))
    for it in items:
        par = it["from"]
        if par not in g["nodes"]:
            sys.exit(f"unknown parent {par}")
        side = it.get("side") or ("fwd" if g["nodes"][par]["side"] in ("fwd", "start") else "bwd")
        if side == "fwd":
            n_f += 1; k = f"F{n_f}"
            g["edges"].setdefault(ekey(par, k), {"kind": "tree"})
        else:
            n_b += 1; k = f"B{n_b}"
            g["edges"].setdefault(ekey(k, par), {"kind": "tree"})
        g["nodes"][k] = {"text": it["text"], "side": side, "parent": par,
                         "depth": g["nodes"][par].get("depth", 0) + 1}
        print(f"  {k:5s} <- {par:4s} {it['text']}")
    save(a.name, g)


def laya_matrix(name, g):
    """Full N x N raw-logit matrix over all nodes, cached in work/<name>/laya.json; only new pairs are scored."""
    f = WORK / name / "laya.json"
    c = json.loads(f.read_text()) if f.exists() else {}
    if c.get("_judge") != judge_tag():
        c = {"_judge": judge_tag()}
    ks = list(g["nodes"])
    todo = [(s, t) for s in ks for t in ks if s != t and ekey(s, t) not in c]
    if todo:
        print(f"  Laya scoring {len(todo)} new pairs of the {len(ks)}x{len(ks)} matrix", flush=True)
        for (s, t), z in zip(todo, laya_logits([(g["nodes"][s]["text"], g["nodes"][t]["text"]) for s, t in todo])):
            c[ekey(s, t)] = round(z, 4)
        f.write_text(json.dumps(c))
    L = np.full((len(ks), len(ks)), np.nan)
    for i, s in enumerate(ks):
        for j, t in enumerate(ks):
            if s != t:
                L[i, j] = c[ekey(s, t)]
    return ks, L


def centre(L):
    """Double-centring: z = raw - column mean - row mean + grand mean."""
    return L - np.nanmean(L, 0, keepdims=True) - np.nanmean(L, 1, keepdims=True) + np.nanmean(L)


def cmd_score(a):
    g = load(a.name)
    V = node_vecs(a.name, g)
    ks, L = laya_matrix(a.name, g)
    Z = centre(L) if (CENTRE_TECHNICAL or not g.get("technical")) else L
    min_p = MIN_P
    if g.get("technical"):       # relative threshold: the tuned judge's scores average ~0.27, so a flat 0.5 drops real steps
        fwd = [sig(Z[i, j]) for i in range(len(ks)) for j in range(len(ks)) if i != j and Z[i, j] > Z[j, i]]
        min_p = float(np.quantile(fwd, TECH_QUANTILE)) if fwd else MIN_P
    g["min_p"] = round(min_p, 4)
    tree = {k for k, e in g["edges"].items() if e.get("kind") == "tree"}
    judged = {k: {f: e[f] for f in ("claude_p", "gap") if f in e} for k, e in g["edges"].items() if "claude_p" in e}
    keep = {k: e for k, e in g["edges"].items() if e.get("kind") in ("waypoint",) or e.get("bridged")}
    for i, s in enumerate(ks):
        for j, t in enumerate(ks):
            if s == t or t == "S" or s == "G" or g["nodes"][s]["side"] == "waypoint" or g["nodes"][t]["side"] == "waypoint":
                continue
            k = ekey(s, t)
            if k in keep:
                continue
            p = sig(Z[i, j])
            if g.get("technical") and Z[i, j] <= Z[j, i]:
                p = 0.0         # the tuned judge knows step order: keep A->B only if it beats B->A
            if p >= min_p or k in tree:
                keep[k] = {"kind": "tree" if k in tree else "link", "p": round(p, 4), "raw": round(sig(L[i, j]), 4),
                           "cos": round(cos(V[s], V[t]), 4)}
    for k, extra in judged.items():     # Claude's judgements survive a rescore
        keep.setdefault(k, {"kind": "link", "p": 0.0}).update(extra)
    for k in keep:      # a bridged edge keeps its bridge; refresh its score
        if keep[k].get("bridged"):
            s, t = k.split("->"); keep[k]["p"] = round(sig(Z[ks.index(s), ks.index(t)]), 4)
    g["edges"] = keep
    g["centre"] = {"col": dict(zip(ks, np.nanmean(L, 0).round(4).tolist())),
                   "row": dict(zip(ks, np.nanmean(L, 1).round(4).tolist())), "grand": round(float(np.nanmean(L)), 4)}
    save(a.name, g)
    usable = sum(1 for e in keep.values() if e["p"] >= min_p)
    print(f"  {usable} usable edges (p >= {min_p:.3f}) out of {len(ks) * (len(ks) - 1)} pairs")
    print("  weak tree edges (a branch Laya doubts):")
    for k in sorted(tree, key=lambda k: keep[k]["p"]):
        if keep[k]["p"] < min_p:
            print(f"    {keep[k]['p']:.3f}  {k}")
    ix = {k: i for i, k in enumerate(ks)}
    flips = [(k, Z[ix[k.split("->")[1]], ix[k.split("->")[0]]] - Z[ix[k.split("->")[0]], ix[k.split("->")[1]]])
             for k in tree if g["nodes"][k.split("->")[0]]["side"] == "bwd"]
    flips = [(k, d) for k, d in flips if d > 1.0]
    if flips:
        print("  backward branches that look reversed (the opposite direction scores much higher):")
        for k, d in sorted(flips, key=lambda x: -x[1]):
            print(f"    {k}  (reverse ahead by {d:.1f} logits)")
    frontier(g)


def frontier(g):
    has_child = {g["nodes"][k].get("parent") for k in g["nodes"]}
    print("  frontier (leaves to expand next):")
    for k, n in g["nodes"].items():
        if k not in has_child and n["side"] in ("fwd", "bwd"):
            print(f"    {k:5s} {n['side']} d{n['depth']}  {n['text']}")


def edge_p(g, e):
    """Claude-judged problems use only judged edges (Laya shortlists); otherwise Laya's calibrated p."""
    if g.get("judge") == "claude":
        return e.get("claude_p", 0) if e["kind"] != "waypoint" else e["p"]
    return e.get("p", 0)


def build_graph(g):
    """Usable edges; a bridged edge is replaced by its verified waypoint chain."""
    import networkx as nx
    D = nx.DiGraph()
    for k, e in g["edges"].items():
        p = edge_p(g, e)
        if p < (MIN_P if g.get("judge") == "claude" else g.get("min_p", MIN_P)) or p <= 0 or e.get("bridged"):
            continue
        s, t = k.split("->")
        D.add_edge(s, t, p=p, w=-math.log(p))
    # A node that uses a theorem can only be entered through that theorem's hypotheses (chained, all must hold),
    # and a hypothesis node only leads on to the step that needs it.
    for u, n in g["nodes"].items():
        reqs = [r for r in n.get("requires", []) if r in g["nodes"]]
        if not reqs or u not in D:
            continue
        for v in list(D.predecessors(u)):
            D.remove_edge(v, u)
        chain = reqs + [u]
        for r in reqs:
            for w in list(D.successors(r)) if r in D else []:
                D.remove_edge(r, w)
        for x, y in zip(chain, chain[1:]):
            D.add_edge(x, y, p=1.0, w=0.0)
    return D


def paths(g, D, n=5):
    """Rank by weakest link first (a chain is only as good as its worst step), product second."""
    import networkx as nx
    out = []
    if "S" in D and "G" in D and nx.has_path(D, "S", "G"):
        levels = sorted({d["p"] for _, _, d in D.edges(data=True)}, reverse=True)
        for tau in levels:          # highest threshold that still connects S to G
            H = D.edge_subgraph([(u, v) for u, v, d in D.edges(data=True) if d["p"] >= tau])
            if "S" in H and "G" in H and nx.has_path(H, "S", "G"):
                break
        g["bottleneck"] = tau
        for pth in nx.shortest_simple_paths(H, "S", "G", weight="w"):
            ps = [D[u][v]["p"] for u, v in zip(pth, pth[1:])]
            out.append((float(np.prod(ps)), pth, ps))
            if len(out) >= n:
                break
    return out


def node_flow(D):
    """For each node: probability of the best S->G path forced through it, divided by the best path overall."""
    import networkx as nx
    dS = nx.single_source_dijkstra_path_length(D, "S", weight="w")
    dG = nx.single_source_dijkstra_path_length(D.reverse(copy=False), "G", weight="w")
    best = dS.get("G", math.inf)
    return {k: math.exp(-(dS[k] + dG[k] - best)) for k in D.nodes
            if k not in ("S", "G") and k in dS and k in dG}


def cmd_solve(a):
    g = load(a.name)
    D = build_graph(g)
    best = paths(g, D)
    d = WORK / a.name
    L = [f"# {a.name}", "", f"**Problem:** {g['problem']}", "",
         f"- S: {g['nodes']['S']['text']}", f"- G: {g['nodes']['G']['text']}", "",
         f"{len(g['nodes'])} nodes, {len(g['edges'])} edges scored, {D.number_of_edges()} usable "
         f"(p >= {g.get('min_p', MIN_P) if g.get('judge') != 'claude' else MIN_P:.3f}); judge: "
         + ("Claude (Laya shortlists)" if g.get("judge") == "claude"
            else "tuned Laya: raw scores, forward-consistent edges, top half usable" if g.get("technical")
            else "Laya matrix, double-centred so 0.5 = chance"), ""]
    if not best:
        L.append("**No complete S->G path yet.** Expand the frontier and score again.")
        print("\n".join(L)); (d / "report.md").write_text("\n".join(L)); write_html(a.name, g, [], [])
        return
    L += ["## Best paths (product of edge probabilities)", ""]
    for i, (pp, pth, ps) in enumerate(best, 1):
        L.append(f"{i}. **{pp:.4f}**  " + " -> ".join(pth))
    top = best[0]
    L += ["", "## Winning path, step by step", ""]
    hard = []
    for (u, v), p in zip(zip(top[1], top[1][1:]), top[2]):
        c = g["edges"].get(ekey(u, v), {}).get("cos", 1.0)
        hard_p = TECH_HARD_P if g.get("technical") and g.get("judge") != "claude" else HARD_P
        why = [w for w, bad in (("low p", p < hard_p),
                                (f"big jump, cos {c:.2f}", c < HARD_COS and not g.get("technical"))) if bad]
        if why:
            hard.append((u, v))
        L.append(f"- `{u}` {g['nodes'][u]['text']}")
        L.append(f"  - p={p:.3f}" + (f" **HARD** ({', '.join(why)})" if why else ""))
        if g["edges"].get(ekey(u, v), {}).get("gap"):
            L.append(f"  - gap: {g['edges'][ekey(u, v)]['gap']}")
    L.append(f"- `G` {g['nodes']['G']['text']}")
    flow = node_flow(D)
    L += ["", "## Strongest alternative waypoints (best path through node / best path)", ""]
    for k, f in sorted(flow.items(), key=lambda x: -x[1])[:8]:
        L.append(f"- {f:.2f}  `{k}` {g['nodes'][k]['text']}")
    if hard:
        L += ["", "## Hard links to decode", ""] + [f"- `bdnm.py bridge {a.name} {u} {v}`" for u, v in hard]
    (d / "report.md").write_text("\n".join(L))
    write_html(a.name, g, top[1], hard, [b[1] for b in best])
    print("\n".join(L))
    print(f"\n  wrote {d/'report.md'} and {d/'map.html'}")


def cmd_candidates(a):
    """Tree edges + S->G + each node's top-k targets by Laya and by vector proximity; unjudged ones only."""
    g = load(a.name)
    V = node_vecs(a.name, g)
    ks, L = laya_matrix(a.name, g)
    Z = centre(L)
    ok = lambda s, t: s != t and t != "S" and s != "G" and "waypoint" not in (g["nodes"][s]["side"], g["nodes"][t]["side"])
    want = {k for k, e in g["edges"].items() if e["kind"] == "tree"} | {"S->G"}
    for i, s in enumerate(ks):
        ts = [(j, t) for j, t in enumerate(ks) if ok(s, t)]
        want |= {ekey(s, t) for j, t in sorted(ts, key=lambda x: -Z[i, x[0]])[:a.k]}
        want |= {ekey(s, t) for j, t in sorted(ts, key=lambda x: -cos(V[s], V[x[1]]))[:a.k]}
    todo = sorted(k for k in want if "claude_p" not in g["edges"].get(k, {}))
    f = WORK / a.name / "judge_todo.json"
    f.write_text(json.dumps({"problem": g["problem"],
                             "nodes": {k: n["text"] for k, n in g["nodes"].items() if n["side"] != "waypoint"},
                             "edges": todo}, indent=1))
    print(f"  {len(todo)} edges to judge -> {f}")


def cmd_judge(a):
    g = load(a.name)
    sc = json.loads(Path(a.file).read_text())
    for k, v in sc.items():
        s, t = k.split("->")
        e = g["edges"].setdefault(k, {"kind": "link", "p": 0.0})
        e["claude_p"], e["gap"] = float(v["p"]), v.get("gap", "")
    g["judge"] = "claude"
    save(a.name, g)
    print(f"  {len(sc)} judged edges stored; solve now uses Claude's p (Laya only shortlists)")


def cmd_bridge(a):
    """Draft waypoints: invert points evenly spaced on the line from A's vector to B's vector."""
    g = load(a.name)
    V = node_vecs(a.name, g)
    A, B = V[a.a], V[a.b]
    ts = [(i + 1) / (a.n + 1) for i in range(a.n)]
    drafts = invert([A + t * (B - A) for t in ts])
    rec = {"from": a.a, "to": a.b, "delta_norm": round(float(np.linalg.norm(B - A)), 3),
           "cos": round(cos(A, B), 3), "drafts": [{"t": t, "text": s} for t, s in zip(ts, drafts)]}
    g["bridges"][ekey(a.a, a.b)] = rec
    save(a.name, g)
    print(f"  {a.a}: {g['nodes'][a.a]['text']}")
    for dr in rec["drafts"]:
        print(f"  t={dr['t']:.2f} draft: {dr['text']}")
    print(f"  {a.b}: {g['nodes'][a.b]['text']}")
    print(f"  |B-A| = {rec['delta_norm']}, cos = {rec['cos']}")


def chain_p(g, chain):
    """Calibrated p for consecutive links of new texts: centre each against the current nodes, like the matrix."""
    others = [n["text"] for n in g["nodes"].values() if n["side"] != "waypoint"]
    raw = laya_logits(list(zip(chain, chain[1:])))
    cols = laya_logits([(o, y) for y in chain[1:] for o in others])
    rows = laya_logits([(x, o) for x in chain[:-1] for o in others])
    n = len(others)
    out = []
    for i, z in enumerate(raw):
        if g.get("technical") and not CENTRE_TECHNICAL:
            out.append(sig(z))       # the tuned judge is used raw, as in the matrix
            continue
        z -= np.mean(cols[i * n:(i + 1) * n]) + np.mean(rows[i * n:(i + 1) * n]) - g["centre"]["grand"]
        out.append(sig(z))
    return out


def cmd_context(a):
    """Look up real statements along the A->B vector instead of decoding it: theorems, their hypotheses,
    collar steps and recorded traps. Hits between A and B are candidate missing steps; requirements
    near A are what the step A relies on."""
    g = load(a.name)
    V = node_vecs(a.name, g)
    idx = np.load(ROOT / "context" / "index.npz")["V"].astype(np.float32)
    rows = [json.loads(l) for l in open(ROOT / "context" / "texts.jsonl")]
    A, B = (V[a.a] / np.linalg.norm(V[a.a])), (V[a.b] / np.linalg.norm(V[a.b]))
    d = B - A
    pts = [A + t * d for t in (0.25, 0.5, 0.75)]
    pts = np.stack([p_ / np.linalg.norm(p_) for p_ in pts])
    near = (idx @ pts.T).max(1)
    t = (idx - A) @ d / (d @ d)
    between = np.where((t > 0.05) & (t < 0.95))[0]
    known = {n["text"] for n in g["nodes"].values()}
    print(f"  {a.a}: {g['nodes'][a.a]['text']}")
    print(f"  {a.b}: {g['nodes'][a.b]['text']}")
    print(f"\n  statements lying between them (candidate missing steps):")
    shown = 0
    for i in between[np.argsort(-near[between])]:
        if rows[i]["text"] in known:
            continue
        print(f"    [{rows[i]['kind']:<11s} cos {near[i]:.2f} t={t[i]:.2f}] {rows[i]['text'][:220]}")
        shown += 1
        if shown >= a.k:
            break
    reqs = [i for i, r in enumerate(rows) if r["kind"] in ("requirement", "trap")]
    ca = idx[reqs] @ A
    print(f"\n  what {a.a} relies on (hypotheses and recorded traps closest to it):")
    for j in np.argsort(-ca)[:a.k]:
        r = rows[reqs[j]]
        print(f"    [{r['kind']:<11s} cos {ca[j]:.2f}] {r['text'][:220]}")


def cmd_require(a):
    """Hypotheses of the theorems a winning-path step relies on become nodes that must hold before that step."""
    g = load(a.name)
    D = build_graph(g)
    best = paths(g, D, 1)
    if not best:
        sys.exit("no S->G path yet")
    V = node_vecs(a.name, g)
    idx = np.load(ROOT / "context" / "index.npz")["V"].astype(np.float32)
    rows = [json.loads(l) for l in open(ROOT / "context" / "texts.jsonl")]
    req = [i for i, r in enumerate(rows) if r["kind"] == "requirement"]
    have = {n["text"] for n in g["nodes"].values()}
    n_r = sum(1 for k in g["nodes"] if k.startswith("R"))
    added = 0
    for u in best[0][1][1:-1]:
        if g["nodes"][u].get("requires"):
            continue
        A = V[u] / np.linalg.norm(V[u])
        ca = idx[req] @ A
        for j in np.argsort(-ca)[:3]:
            if ca[j] < a.cos:
                break
            name, cond = rows[req[j]]["text"].split(" requires ", 1)
            text = f"{cond[0].upper() + cond[1:]} hold here (hypothesis of {name})"
            if text in have:
                continue
            n_r += 1; k = f"R{n_r}"
            g["nodes"][k] = {"text": text, "side": "req", "for": u, "cos": round(float(ca[j]), 3)}
            g["nodes"][u].setdefault("requires", []).append(k)
            have.add(text); added += 1
            print(f"  {k} must hold before {u}: {text}  (cos {ca[j]:.2f})")
    save(a.name, g)
    print(f"  {added} hypothesis nodes added; run score (Laya rates what could establish them) then solve")


def cmd_check(a):
    """Claude's waypoint steps must move forward along B-A, each be a smaller jump than A->B, and each be usable."""
    g = load(a.name)
    steps = json.loads(Path(a.file).read_text())
    V = node_vecs(a.name, g)
    A, B = V[a.a], V[a.b]
    d = B - A
    E = embed(steps)
    ts = [float((e - A) @ d / (d @ d)) for e in E]
    off = [float(np.linalg.norm((e - A) - t * d) / np.linalg.norm(d)) for e, t in zip(E, ts)]
    chain = [g["nodes"][a.a]["text"]] + steps + [g["nodes"][a.b]["text"]]
    ps = chain_p(g, chain)
    orig = g["edges"][ekey(a.a, a.b)]["p"]
    CE = np.vstack([A, E, B])
    jumps = [cos(x, y) for x, y in zip(CE, CE[1:])]
    c0 = cos(A, B)
    mono = all(x < y + 0.1 for x, y in zip([0] + ts, ts + [1]))        # forward along B-A (0.1 slack for noise)
    small = all(j > c0 for j in jumps)                                  # each step is a smaller jump than A->B
    usable = min(ps) >= MIN_P and (orig >= HARD_P or min(ps) > orig)   # every step usable; beats a low-p original
    ok = mono and small and usable
    print(f"  original {a.a}->{a.b} p={orig:.3f} cos={c0:.2f}")
    for s_, t, o in zip(steps, ts, off):
        print(f"    t={t:+.2f} off-line={o:.2f}  {s_}")
    print("  sub-link p:   " + ", ".join(f"{p:.3f}" for p in ps))
    print("  sub-link cos: " + ", ".join(f"{j:.2f}" for j in jumps))
    print(f"  forward along the vector: {mono}; every step a smaller jump: {small}; every step usable: {usable}")
    rec = g["bridges"].setdefault(ekey(a.a, a.b), {"from": a.a, "to": a.b})
    rec["check"] = {"steps": steps, "t": ts, "off": off, "p": ps, "cos": jumps, "ok": ok}
    if ok:
        prev = a.a
        n_w = sum(1 for k in g["nodes"] if k.startswith("W"))
        for i, (s, p) in enumerate(zip(steps, ps)):
            n_w += 1; k = f"W{n_w}"
            g["nodes"][k] = {"text": s, "side": "waypoint", "bridge": ekey(a.a, a.b)}
            g["edges"][ekey(prev, k)] = {"kind": "waypoint", "p": round(p, 4)}
            prev = k
        g["edges"][ekey(prev, a.b)] = {"kind": "waypoint", "p": round(ps[-1], 4)}
        g["edges"][ekey(a.a, a.b)]["bridged"] = True
        print("  ACCEPTED: waypoints spliced in; run solve again")
    else:
        print("  REJECTED: rewrite the steps (or split the weakest sub-link with another bridge)")
    save(a.name, g)


# ----------------------------------------------------------------------------- viewer
def write_html(name, g, best, hard, alts=()):
    """Draw the branch tree, the top paths and decoded waypoints only (the full matrix is unreadable)."""
    on = set(zip(best, best[1:]))
    shown = on | {(u, v) for pth in alts for u, v in zip(pth, pth[1:])}
    hard = set(hard)
    lines = ["flowchart LR"]
    for k, n in g["nodes"].items():
        t = html.escape(n["text"][:70] + ("…" if len(n["text"]) > 70 else "")).replace('"', "'")
        shape = ('(["', '"])') if k in ("S", "G") else (('[/"', '"/]') if n["side"] == "waypoint" else ('["', '"]'))
        lines.append(f'  {k}{shape[0]}<b>{k}</b> {t}{shape[1]}')
    styles, i = [], 0
    for key, e in g["edges"].items():
        u, v = key.split("->")
        if e.get("p", 0) < MIN_P or not ((u, v) in shown or e["kind"] in ("tree", "waypoint") or e.get("bridged")):
            continue
        arrow = "-.->" if e.get("bridged") else "-->"
        lines.append(f"  {u} {arrow}|{e['p']:.2f}| {v}")
        if (u, v) in hard and not e.get("bridged"):
            styles.append(f"  linkStyle {i} stroke:#d64545,stroke-width:3px,stroke-dasharray:6 4")
        elif (u, v) in on:
            styles.append(f"  linkStyle {i} stroke:#2f7de1,stroke-width:4px")
        else:
            styles.append(f"  linkStyle {i} stroke:#999,stroke-width:1px")
        i += 1
    lines += styles
    lines += ["  classDef s fill:#e8f0fe,stroke:#2f7de1", "  classDef f fill:#eef7ee,stroke:#4a9a4a",
              "  classDef b fill:#fdf1e6,stroke:#d9822b", "  classDef w fill:#f5eefc,stroke:#8a5cd1",
              "  class S,G s"]
    for c, side in (("f", "fwd"), ("b", "bwd"), ("w", "waypoint")):
        ks = [k for k, n in g["nodes"].items() if n["side"] == side]
        if ks:
            lines.append(f"  class {','.join(ks)} {c}")
    mer = "\n".join(lines)
    rep = (WORK / name / "report.md").read_text() if (WORK / name / "report.md").exists() else ""
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>BDNM {html.escape(name)}</title>
<style>body{{font:14px system-ui;margin:16px;background:#fff;color:#222}}pre.r{{white-space:pre-wrap;background:#f6f6f6;padding:12px}}
.k span{{margin-right:14px}}</style></head><body>
<h2>{html.escape(g['problem'])}</h2>
<div class="k"><span style="color:#2f7de1">━ winning path</span><span style="color:#d64545">┅ hard link</span>
<span style="color:#4a9a4a">■ forward</span><span style="color:#d9822b">■ backward</span><span style="color:#8a5cd1">■ decoded waypoint</span></div>
<pre class="mermaid">{mer}</pre><pre class="r">{html.escape(rep)}</pre>
<script type="module">import m from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";
m.initialize({{startOnLoad:true,maxTextSize:500000,flowchart:{{htmlLabels:true}}}});</script></body></html>"""
    (WORK / name / "map.html").write_text(page)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("init"); p.add_argument("name"); p.add_argument("--start", required=True)
    p.add_argument("--goal", required=True); p.add_argument("--problem")
    p.add_argument("--technical", action="store_true", help="maths/technical: use the trained adapter as judge")
    p = sp.add_parser("add"); p.add_argument("name"); p.add_argument("file")
    p = sp.add_parser("score"); p.add_argument("name")
    p = sp.add_parser("solve"); p.add_argument("name")
    p = sp.add_parser("candidates"); p.add_argument("name"); p.add_argument("--k", type=int, default=3)
    p = sp.add_parser("judge"); p.add_argument("name"); p.add_argument("file")
    p = sp.add_parser("bridge"); p.add_argument("name"); p.add_argument("a"); p.add_argument("b")
    p.add_argument("--n", type=int, default=1)
    p = sp.add_parser("check"); p.add_argument("name"); p.add_argument("a"); p.add_argument("b"); p.add_argument("file")
    p = sp.add_parser("require"); p.add_argument("name"); p.add_argument("--cos", type=float, default=0.7)
    p = sp.add_parser("context"); p.add_argument("name"); p.add_argument("a"); p.add_argument("b")
    p.add_argument("--k", type=int, default=6)
    a = ap.parse_args()
    if a.cmd != "init" and gpath(a.name).exists():
        _M["technical"] = bool(load(a.name).get("technical"))
    {"init": cmd_init, "add": cmd_add, "candidates": cmd_candidates, "judge": cmd_judge, "score": cmd_score, "solve": cmd_solve,
     "bridge": cmd_bridge, "check": cmd_check, "context": cmd_context, "require": cmd_require}[a.cmd](a)


if __name__ == "__main__":
    main()
