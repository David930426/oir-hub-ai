"""The conditions under comparison. Each maps a case to (label, target doc_id, info).

A case is an input document (case["query"]: title, text, date) and its signals against the
library (see `signals`). The same code serves the experiment (run.py) and daily use (check_doc.py).

Baselines have free thresholds; `tune(dev)` picks them by grid search on the dev split ONLY,
then `predict` is applied unchanged to test. LLM methods have nothing to tune except the
few-shot exemplars, which also come from dev.

    B0  majority class
    B1  metadata: normalised-title match + body identity
    B2  e5 cosine of the nearest KB doc, two thresholds
    B3  BM25 (normalised) of the nearest KB doc, two thresholds
    B4  bge-reranker-v2-m3 probability over the top-10 cosine candidates, two thresholds
    M1  LLM, input document only (no library)
    M2  LLM + top-5 cosine candidates   <- what the OIR Hub router does
    M3  M2 + one dev exemplar per class
    M4  cascade: confident deterministic rule, otherwise M2
"""

import difflib
import hashlib
import json
import pickle
import time
from collections import Counter

import numpy as np
import requests
from pydantic import BaseModel, Field
from typing import Literal

import corpus as C
from metrics import LABELS, macro_f1

U, N, R = "UPDATE", "NEW", "REDUNDANT"
TOP_RERANK = 10
TOP_LLM = 5


# ---------------------------------------------------------------- per-case signals
#
# A case is an input document - case["query"] = {"title", "text", "date"} - to be compared with
# the library documents dated before it. Everything a method needs is computed here once and
# stored by doc_id, so the methods never look anything up in the library themselves.

def _ratio(a, b):
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def compute_signals(query, lib, qvec):
    docs = lib.docs
    # Documents already replaced by a newer one (check_doc --apply) are not the current version,
    # so they are not offered as candidates. The experiment's library has none.
    kb = [i for i in lib.before(query["date"]) if not docs[i].get("replaced_by")]
    if not kb:
        return {"kb_size": 0}
    ids = [docs[i]["doc_id"] for i in kb]

    cos = lib.vectors[kb] @ qvec
    bm = lib.bm25.scores(C.tokens(f"{query['title']}\n{query['text']}"[:4000]))[kb]
    by_cos = [ids[k] for k in np.argsort(-cos)[:50]]

    nt = C.norm_title(query["title"])
    tsim = np.array([_ratio(nt, C.norm_title(docs[i]["title"])) if nt else 0.0 for i in kb])
    title_sim = {ids[k]: float(tsim[k]) for k in np.flatnonzero(tsim >= 0.5)}

    nb = C.norm_body(query["text"])[:3000]
    identical = [ids[k] for k, i in enumerate(kb) if nb and C.norm_body(docs[i]["text"])[:3000] == nb]

    # Body similarity only for documents a method could name as the target.
    shortlist = set(by_cos[:TOP_RERANK]) | {d for d, t in title_sim.items() if t >= 0.7}
    body_sim = {d: _ratio(nb, C.norm_body(docs[lib.index[d]]["text"])[:3000]) for d in shortlist}

    keep = set(by_cos) | set(title_sim) | set(identical)
    snippet = lambda d: {"title": d["title"], "date": d["date"], "url": d.get("url", ""),
                         "text": d["text"][:CAND_CHARS]}
    return {
        "kb_size": len(kb),
        "cos": dict(zip(ids, cos.tolist())), "bm25": dict(zip(ids, bm.tolist())),
        "by_cos": by_cos, "title_sim": title_sim, "identical": identical, "body_sim": body_sim,
        "docs": {d: snippet(docs[lib.index[d]]) for d in keep},
    }


def add_rerank(query, sig, lib, reranker):
    cands = sig["by_cos"][:TOP_RERANK]
    q = f"{query['title']}\n{query['text']}"[:1500]
    probs = reranker.predict([(q, C.doc_text(lib.docs[lib.index[d]], 1500)) for d in cands],
                             activation_fn=__import__("torch").nn.Sigmoid(), batch_size=8)
    sig["rerank"] = dict(zip(cands, map(float, probs)))


def signals(cases, lib, rerank=True, cache_name="signals_files"):
    """Signals for many cases. Cached on disk, keyed by the input itself, so editing a file or
    its date recomputes that case and nothing else. cache_name=None disables the cache."""
    def key(c):
        q = c["query"]
        return hashlib.md5(f"{q['date']}|{q['title']}|{q['text']}".encode()).hexdigest()

    path = C.CACHE / f"{cache_name}.pkl"
    cache = pickle.loads(path.read_bytes()) if (cache_name and path.exists()) else {}
    todo = [c for c in cases if key(c) not in cache]
    if todo:
        qvecs = C.query_vectors([f"{c['query']['title']}\n{c['query']['text']}" for c in todo])
        for c, v in zip(todo, qvecs):
            cache[key(c)] = compute_signals(c["query"], lib, v)
    need = [c for c in cases if cache[key(c)].get("kb_size") and "rerank" not in cache[key(c)]]
    if rerank and need:
        import torch
        from sentence_transformers import CrossEncoder
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = CrossEncoder("BAAI/bge-reranker-v2-m3", device=dev, max_length=512)
        for c in need:
            add_rerank(c["query"], cache[key(c)], lib, model)
        del model
        if dev == "cuda":
            torch.cuda.empty_cache()  # hand the VRAM back before Ollama needs it
    if cache_name and (todo or need):
        C.CACHE.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(cache))
    return {c["case_id"]: cache[key(c)] for c in cases}


# ---------------------------------------------------------------- baselines

class Method:
    name = "?"
    uses_llm = False

    def tune(self, dev, gold, sig):
        pass

    def predict(self, case, sig):
        raise NotImplementedError


class Majority(Method):
    name = "B0 majority"

    def tune(self, dev, gold, sig):
        self.label = Counter(gold[c["case_id"]]["label"] for c in dev).most_common(1)[0][0]

    def predict(self, case, sig):
        return self.label, None, {}


def _grid(values, n=25):
    v = np.asarray([x for x in values if x is not None])
    return sorted(set(np.quantile(v, np.linspace(0, 1, n)).round(4))) if len(v) else [0.5]


class Threshold(Method):
    """score >= t_red -> REDUNDANT, score >= t_upd -> UPDATE, else NEW; target = top doc.
    The shared shape of B2/B3/B4 - only the score differs."""
    key = None

    def top(self, sig):
        s = sig.get(self.key) or {}
        if not s:
            return None, None
        j = max(s, key=s.get)
        return j, s[j]

    def decide(self, sig, t_upd, t_red):
        j, s = self.top(sig)
        if j is None or s < t_upd:
            return N, None
        return (R if s >= t_red else U), j

    def tune(self, dev, gold, sig):
        grid = _grid([self.top(sig[c["case_id"]])[1] for c in dev])
        best = (-1, None)
        for t_upd in grid:
            for t_red in [g for g in grid if g >= t_upd] + [9e9]:  # 9e9 = never REDUNDANT
                preds = [self.decide(sig[c["case_id"]], t_upd, t_red)[0] for c in dev]
                f = macro_f1([gold[c["case_id"]]["label"] for c in dev], preds)
                if f > best[0]:
                    best = (f, (t_upd, t_red))
        self.t_upd, self.t_red = best[1]

    def predict(self, case, sig):
        label, d = self.decide(sig, self.t_upd, self.t_red)
        return label, d, {"t_upd": self.t_upd, "t_red": self.t_red}


class Cosine(Threshold):
    name, key = "B2 cosine (e5)", "cos"


class BM25(Threshold):
    name, key = "B3 BM25", "bm25"


class Reranker(Threshold):
    name, key = "B4 reranker", "rerank"


class Metadata(Method):
    """Same normalised title as an older doc -> that program again. Identical body -> REDUNDANT.
    Thresholds: how fuzzy a title match may be, and how similar a body must be to count as a
    re-post rather than an update."""
    name = "B1 metadata/title"

    def decide(self, case, sig, t_title, t_body):
        date = lambda d: sig["docs"][d]["date"]
        if sig.get("identical"):
            return R, max(sig["identical"], key=date)
        matches = [d for d, s in (sig.get("title_sim") or {}).items() if s >= t_title]
        if not matches:
            return N, None
        d = max(matches, key=lambda d: (date(d), sig["title_sim"][d]))
        return (R if sig["body_sim"].get(d, 0.0) >= t_body else U), d

    def tune(self, dev, gold, sig):
        best = (-1, None)
        for t_title in [0.7, 0.8, 0.9, 0.95, 1.0]:
            for t_body in [0.8, 0.9, 0.95, 0.98, 1.01]:
                preds = [self.decide(c, sig[c["case_id"]], t_title, t_body)[0] for c in dev]
                f = macro_f1([gold[c["case_id"]]["label"] for c in dev], preds)
                if f > best[0]:
                    best = (f, (t_title, t_body))
        self.t_title, self.t_body = best[1]

    def predict(self, case, sig):
        label, d = self.decide(case, sig, self.t_title, self.t_body)
        return label, d, {"t_title": self.t_title, "t_body": self.t_body}


# ---------------------------------------------------------------- LLM judge

OLLAMA = "http://localhost:11434/api/chat"
LLM_MODEL = "qwen3:4b"
# ~4.6k prompt tokens with 5 candidates; few-shot adds three more such blocks, hence num_ctx 16k.
IN_CHARS, CAND_CHARS = 2000, 700

SYSTEM = """You maintain the knowledge base of a Taiwanese university's Office of International Relations.
A new announcement has arrived. Decide what the ingestion pipeline must do with it.

UPDATE    - it replaces information in an existing document: the same program, scholarship, event or rule,
            but something a student acts on changed (new cycle/semester/year, deadline, fee, quota,
            eligibility, required documents, procedure, link). A new yearly edition is an UPDATE.
            A deadline extension or correction notice is an UPDATE.
NEW       - no existing document covers the same program/event/rule. Two different scholarships or schools
            that merely share a template are different programs. Result lists and info sessions are NEW.
REDUNDANT - an existing document already says the same thing for the same cycle (re-post, typo fix,
            or the English/Chinese translation of the same announcement).

If several candidates are editions of the same program, the target is the most recent one.
Write the rationale first (one or two sentences, English), then the label.
target must be the candidate id (C1-C5) that is replaced or duplicated for UPDATE and REDUNDANT, and NONE for NEW."""

SYSTEM_NO_KB = SYSTEM.split("If several")[0] + """You cannot see the knowledge base. Judge from the announcement alone.
Write the rationale first (one or two sentences, English), then the label. target must be null."""


class Verdict(BaseModel):
    rationale: str
    label: Literal["UPDATE", "NEW", "REDUNDANT"]
    # Required, with an explicit NONE: when it was Optional the model often answered UPDATE
    # and left target null, which is useless to the pipeline.
    target: Literal["C1", "C2", "C3", "C4", "C5", "NONE"]
    confidence: float = Field(ge=0, le=1)


class VerdictNoKB(BaseModel):
    rationale: str
    label: Literal["UPDATE", "NEW", "REDUNDANT"]
    confidence: float = Field(ge=0, le=1)


def _render_doc(d, limit):
    return f"title: {d['title']}\ndate: {d['date']}\n{d['text'][:limit]}"


def render_user(case, sig, with_kb=True, in_chars=IN_CHARS, cand_chars=CAND_CHARS):
    msg = f"INCOMING ANNOUNCEMENT\n{_render_doc(case['query'], in_chars)}"
    if not with_kb:
        return msg, []
    cands = (sig.get("by_cos") or [])[:TOP_LLM]
    blocks = [f"[C{k + 1}]\n{_render_doc(sig['docs'][d], cand_chars)}" for k, d in enumerate(cands)]
    msg += "\n\nEXISTING DOCUMENTS (most similar first)\n" + ("\n\n".join(blocks) if blocks else "(none)")
    return msg, cands


LLM_CACHE = C.CACHE / "llm_calls.jsonl"
_llm_cache = None


def call_llm(messages, schema, timeout=300):
    """Cached by the exact request, so re-running an analysis, or M4 escalating to M2's
    identical prompt, costs nothing - and the measured latency of the original call is kept."""
    global _llm_cache
    import hashlib
    key = hashlib.sha256(json.dumps([LLM_MODEL, messages, schema.__name__], ensure_ascii=False).encode()).hexdigest()
    if _llm_cache is None:
        _llm_cache = {}
        if LLM_CACHE.exists():
            for line in LLM_CACHE.open(encoding="utf-8"):
                row = json.loads(line)
                _llm_cache[row["key"]] = row
    if key in _llm_cache:
        row = _llm_cache[key]
        return schema.model_validate_json(row["content"]), row["cost"]
    content, cost = _call_llm(messages, schema, timeout)
    C.CACHE.mkdir(parents=True, exist_ok=True)
    row = {"key": key, "content": content, "cost": cost}
    with LLM_CACHE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    _llm_cache[key] = row
    return schema.model_validate_json(content), cost


def _call_llm(messages, schema, timeout):
    t0 = time.time()
    r = requests.post(OLLAMA, timeout=timeout, json={
        "model": LLM_MODEL, "stream": False,
        "think": False,  # qwen3 thinking = minutes per call on this GPU; see the notebook
        "messages": messages, "format": schema.model_json_schema(),
        "options": {"temperature": 0, "seed": 0, "num_ctx": 16384},
    })
    r.raise_for_status()
    body = r.json()
    return body["message"]["content"], {
        "latency_s": round(time.time() - t0, 2),
        "prompt_tokens": body.get("prompt_eval_count"),
        "output_tokens": body.get("eval_count"),
    }


FLAGS = {"unsure", "template_trap", "reworded", "translation", "old_predecessor"}


def _reason(notes):
    """The annotator's one-sentence reason, without the flag words the labeling tool prepends.
    A dev case without a written reason is never used as an exemplar: a placeholder rationale
    ("This is UPDATE.") taught the model to stop reasoning altogether in the pilot."""
    parts = [p.strip() for p in (notes or "").split(" | ")]
    text = " | ".join(p for p in parts if p and not set(p.split()) <= FLAGS)
    return text if len(text) >= 15 else ""


class LLM(Method):
    uses_llm = True

    def __init__(self, name, with_kb=True, shots=0):
        self.name, self.with_kb, self.shots = name, with_kb, shots
        self.examples = []

    def tune(self, dev, gold, sig):
        """Few-shot exemplars: the first dev case of each class, in a fixed order. Picked
        from dev only, so they never leak into test."""
        if not self.shots:
            return
        for label in LABELS:
            for c in dev:
                g = gold[c["case_id"]]
                why = _reason(g.get("notes"))
                if g["label"] == label and why and sig[c["case_id"]].get("kb_size"):
                    # Shorter than a real prompt: three full-length examples pushed the pilot
                    # to 14k tokens, and the 4B model started copying labels instead of reading.
                    user, cands = render_user(c, sig[c["case_id"]], self.with_kb, 600, 250)
                    tgt = next((f"C{k + 1}" for k, d in enumerate(cands) if d in str(g["target"]).split("|")), None)
                    if label != N and tgt is None:
                        continue  # gold target not among the candidates - a confusing example
                    ans = {"rationale": why, "label": label,
                           "target": tgt if label != N else "NONE", "confidence": 0.9}
                    self.examples += [{"role": "user", "content": user},
                                      {"role": "assistant", "content": json.dumps(ans, ensure_ascii=False)}]
                    break
        if len(self.examples) < 2 * len(LABELS):
            print(f"WARNING {self.name}: only {len(self.examples) // 2} exemplars - write a reason in the "
                  "notes of more dev cases (UPDATE/REDUNDANT ones need their target in the top-5)")

    def predict(self, case, sig):
        user, cands = render_user(case, sig, self.with_kb)
        schema = Verdict if self.with_kb else VerdictNoKB
        system = SYSTEM if self.with_kb else SYSTEM_NO_KB
        msgs = [{"role": "system", "content": system}, *self.examples, {"role": "user", "content": user}]
        try:
            v, cost = call_llm(msgs, schema)
        except Exception as e:  # count a failed call as NEW - the pipeline's safe default
            return N, None, {"error": repr(e)}
        target = None
        tgt = getattr(v, "target", None)
        if v.label != N and tgt and tgt != "NONE" and int(tgt[1:]) <= len(cands):
            target = cands[int(tgt[1:]) - 1]
        return v.label, target, {**cost, "confidence": v.confidence, "rationale": v.rationale}


class Cascade(Method):
    """FrugalGPT-style: decide by rule where the rule is reliable on dev, else ask the LLM.
    Rules are only kept if their dev precision is at least P_MIN."""
    name = "M4 cascade"
    uses_llm = True
    P_MIN = 0.9

    def __init__(self, llm):
        self.llm = llm

    def rule(self, case, sig):
        if not sig.get("kb_size"):
            return N, None
        if sig.get("identical"):
            return R, max(sig["identical"], key=lambda d: sig["docs"][d]["date"])
        top = max(sig["cos"].values())
        if self.t_new is not None and top < self.t_new:
            return N, None
        return None

    def tune(self, dev, gold, sig):
        self.llm.tune(dev, gold, sig)
        # Largest cosine cut-off below which "NEW" is still >= P_MIN precise on dev.
        tops = sorted((max(sig[c["case_id"]]["cos"].values()), gold[c["case_id"]]["label"])
                      for c in dev if sig[c["case_id"]].get("kb_size"))
        self.t_new = None
        for k in range(1, len(tops) + 1):
            below = [l for _, l in tops[:k]]
            if below.count(N) / len(below) >= self.P_MIN:
                self.t_new = tops[k - 1][0] + 1e-9
        # (no rule at all if even the single lowest-similarity dev case is not NEW)

    def predict(self, case, sig):
        r = self.rule(case, sig)
        if r is not None:
            return r[0], r[1], {"escalated": False}
        label, target, info = self.llm.predict(case, sig)
        return label, target, {**info, "escalated": True}


def all_methods():
    m2 = LLM("M2 LLM + candidates")
    return [Majority(), Metadata(), Cosine(), BM25(), Reranker(),
            LLM("M1 LLM zero-shot", with_kb=False), m2,
            LLM("M3 LLM + few-shot", shots=1), Cascade(LLM("M2 LLM + candidates"))]
