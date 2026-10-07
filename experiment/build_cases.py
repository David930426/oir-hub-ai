"""Sample the cases to label, and write the offline labeling page.

    python experiment/build_cases.py     # -> data/cases_oir.jsonl, labeling/annotate_oir.html

Already done: re-running it reproduces the same 200 cases (fixed seed), but only re-run it if you
mean to start the labeling over. After it, fetch_files.py turns each case into an input file.

Why pools and not a plain random sample: roughly 9 in 10 OIR posts are NEW, so 200 random
posts would give a handful of UPDATEs and no hard negatives. Three pools instead:

    title    an older post has a near-identical normalised title  (recurring programs: true
             UPDATEs, re-posts, and same-template-different-program traps)
    semantic no title match, but the nearest older post is very close in e5 space (reworded
             successors and template traps)
    random   uniform over all posts, so the natural case mix is represented too

Pool membership is saved with each case but hidden from the annotator (cases are shuffled),
and results are reported per pool as well as overall. The pools are defined by the same kind
of signal the baselines use; that bias is a threat to validity and is written up as one.
"""

import difflib
import json
import random

import numpy as np

import corpus as C

SEED = 13
POOL_SIZES = {"title": 90, "semantic": 60, "random": 50}
MIN_DATE = "2015-01-01"   # earlier posts have too little history behind them to be interesting
N_CANDIDATES = 5          # per retrieval method, for the labeling view


def bulletin_query_vectors(lib):
    """Each bulletin embedded as a query, for comparing it with the ones before it."""
    return C.query_vectors([C.doc_text(d) for d in lib.docs])


def pool_features(lib, qvecs):
    """Cheap per-bulletin features: best title similarity and best cosine against older ones."""
    titles = [C.norm_title(d["title"]) for d in lib.docs]
    feats = []
    for i, d in enumerate(lib.docs):
        kb = lib.before(d["date"])
        if not kb:
            feats.append(None)
            continue
        best_t = 0.0
        if titles[i]:
            sm = difflib.SequenceMatcher(None, "", titles[i], autojunk=False)
            for j in kb:
                if titles[j]:
                    sm.set_seq1(titles[j])
                    if sm.real_quick_ratio() > best_t and sm.quick_ratio() > best_t:
                        best_t = max(best_t, sm.ratio())
        feats.append({"title": best_t, "cos": float((lib.vectors[kb] @ qvecs[i]).max())})
    return feats


def sample(lib, qvecs):
    feats = pool_features(lib, qvecs)
    elig = [i for i, d in enumerate(lib.docs) if feats[i] and d["date"] >= MIN_DATE]
    cos_hi = np.quantile([feats[i]["cos"] for i in elig], 0.8)
    pools = {
        "title": [i for i in elig if feats[i]["title"] >= 0.9],
        "semantic": [i for i in elig if feats[i]["title"] < 0.9 and feats[i]["cos"] >= cos_hi],
    }
    rng = random.Random(SEED)
    chosen, taken = [], set()
    for name in ["title", "semantic"]:
        pick = rng.sample(pools[name], min(POOL_SIZES[name], len(pools[name])))
        chosen += [(i, name) for i in pick]
        taken |= set(pick)
    rest = [i for i in elig if i not in taken]
    # The random pool absorbs any shortfall (the semantic pool is small: most near-duplicates
    # also share a title), so the total stays at sum(POOL_SIZES).
    n_random = sum(POOL_SIZES.values()) - len(chosen)
    chosen += [(i, "random") for i in rng.sample(rest, n_random)]
    print({k: len(v) for k, v in pools.items()}, "eligible:", len(elig), f"cos p80={cos_hi:.3f}")
    return chosen


def candidates(lib, qvecs, i):
    """What the annotator sees: the union of several retrievers' top hits, ordered by date
    (newest first) rather than by any one method's score, so the view favours no method."""
    docs = lib.docs
    kb = lib.before(docs[i]["date"])
    cos = lib.vectors[kb] @ qvecs[i]
    bm = lib.bm25.scores(C.tokens(C.doc_text(docs[i], 4000)))[kb]
    nt = C.norm_title(docs[i]["title"])
    tsim = np.array([difflib.SequenceMatcher(None, nt, C.norm_title(docs[j]["title"])).ratio() if nt else 0
                     for j in kb])
    pick = set()
    for s in (cos, bm, tsim):
        pick |= {kb[k] for k in np.argsort(-s)[:N_CANDIDATES]}
    return sorted(pick, key=lambda j: docs[j]["date"], reverse=True)


def build():
    lib = C.Library.load()
    qvecs = bulletin_query_vectors(lib)
    chosen = sample(lib, qvecs)
    random.Random(SEED + 1).shuffle(chosen)  # annotators must not see pool order
    return lib, [{
        "case_id": f"oir-{n:03d}", "corpus": "oir", "doc_id": lib.docs[i]["doc_id"],
        "date": lib.docs[i]["date"], "pool": pool,
        "candidates": [lib.docs[j]["doc_id"] for j in candidates(lib, qvecs, i)],
    } for n, (i, pool) in enumerate(chosen, 1)]


def main():
    lib, cases = build()
    out = C.DATA / "cases_oir.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # The page carries every bulletin, so the annotator can search the whole history when the
    # right target is not among the suggested candidates.
    payload = {
        "corpus": "oir",
        "docs": {d["doc_id"]: [d["title"], d["date"], d["text"], d.get("url", "")] for d in lib.docs},
        "cases": [{k: c[k] for k in ("case_id", "doc_id", "date", "candidates")} for c in cases],
        # The second annotator's subset: 40 random cases from the first 150, which is how
        # far the main annotator is expected to get.
        "kappa": sorted(random.Random(SEED + 2).sample([c["case_id"] for c in cases[:150]],
                                                        min(40, len(cases[:150])))),
    }
    tpl = (C.EXP / "labeling" / "annotate_template.html").read_text(encoding="utf-8")
    page = C.EXP / "labeling" / "annotate_oir.html"
    page.write_text(tpl.replace("/*DATA*/null", json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")),
                    encoding="utf-8")
    print(f"{len(cases)} cases -> {out.name}, {page.relative_to(C.EXP)}")


if __name__ == "__main__":
    main()
