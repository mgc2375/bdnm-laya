# bdnm-laya

Bidirectional node mapping: Claude (in chat) branches, Laya judges, vectors decode the hard links.

    .venv/bin/python bdnm.py init NAME --start "..." --goal "..." [--problem "..."]
    .venv/bin/python bdnm.py add NAME work/NAME/r1.json     # [{"from":"S","text":"..."}, {"from":"G","side":"bwd","text":"..."}]
    .venv/bin/python bdnm.py score NAME                     # Laya fills the N x N matrix (new pairs only), prints frontier
    .venv/bin/python bdnm.py solve NAME                     # best paths, hard links -> work/NAME/report.md + map.html
    .venv/bin/python bdnm.py bridge NAME A B --n 3          # vec2text drafts at 1/4, 1/2, 3/4 along A->B
    .venv/bin/python bdnm.py check NAME A B steps.json      # verify Claude's waypoint steps; splice in if accepted

Loop: add a round of branches -> score -> solve -> bridge each hard link on the winner -> check -> solve.

## How it scores
- Nodes are embedded with GTR-T5-base (the space vec2text inverts). Cache: work/NAME/emb.npz.
- Laya answers "once 'from' is done, can 'to' be done immediately?" for every ordered pair (predict_batch).
  The raw logit matrix is double-centred (minus column mean, minus row mean, plus grand mean), then sigmoid:
  0.5 = chance. On a labelled test set this took AUC from 0.74 (raw) to 0.89. Putting the problem
  text into the state made it worse, so it is left out. Cache: work/NAME/laya.json.
- Usable edge: calibrated p >= 0.5. Path score = product of p (shortest path on -log p).
- Hard link: p < 0.70 OR cosine between the two nodes < 0.50 (a big jump Laya may be overrating).
- Waypoints are accepted when they move forward along B-A (projection, 0.1 slack), every step is a
  smaller jump than A->B, and every step is usable (and beats a low-p original).

## Notes
- vec2text needs transformers < 5 (nested from_pretrained breaks on 5.x); Laya 0.3.20 runs fine on 4.57.
- Laya's warning about invalid temperatures for choice questions with 11+ options doesn't apply (2 options here).
- Offline: HF_HUB_OFFLINE=1 is set; models are in ~/.cache/huggingface (t5-base, gtr-t5-base, jxm/gtr__nq__32*).
- Independent of ~/laya-render and ~/autonomous-finance (own venv, own Laya instance, no :8000).

## Claude as judge (for technical problems)
Laya fails on maths: on the Bullough-Dodd (HA-T) test it ranked "start -> goal" at 0.93 and the KLMN form-sum
trap second, and scored the real N9 truncation proof 0.09. It matches wording, not reasoning.
So for technical problems Laya only shortlists, and Claude (in chat) judges the links:

    .venv/bin/python bdnm.py candidates NAME     # tree edges + S->G + each node's top-3 by Laya and by vector
    .venv/bin/python bdnm.py judge NAME scores.json   # {"A->B": {"p": 0.8, "gap": "what is missing"}}

Once judged, solve uses only Claude's p. The gap notes print under each step of the winning path.
(The headless `claude -p` route needs the CLI logged in; it wasn't on 2026-09-28.)
