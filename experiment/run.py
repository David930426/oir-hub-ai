"""Run every condition on the labeled cases and write the results.

    python experiment/run.py oir data/labels_oir_DG.csv                 # the real experiment
    python experiment/run.py oir data/labels_oir_DG.csv --no-llm        # baselines only (seconds)
    python experiment/run.py local data/labels_local_DG.csv --smoke     # pilot: no split, NOT a result
    python experiment/run.py kappa data/labels_oir_DG.csv data/labels_oir_XY.csv

Output goes to experiment/runs/<corpus>_<timestamp>/: results.md, predictions.csv, errors.csv.
"""

import argparse
import csv
import datetime as dt
import json
import random
from collections import Counter, defaultdict

import numpy as np

import corpus as C
from methods import all_methods, signals
from metrics import (LABELS, bootstrap_ci, confusion, correct_vector, holm, kappa, macro_f1,
                     mcnemar_exact, paired_bootstrap_diff, per_class, target_accuracy, target_hit)

SEED = 13
DEV_FRACTION = 0.4


def read_labels(path):
    with open(path, encoding="utf-8-sig") as f:
        rows = {r["case_id"]: r for r in csv.DictReader(f) if r.get("gold_label")}
    for r in rows.values():
        r["label"], r["target"] = r["gold_label"].strip().upper(), r.get("gold_target", "").strip()
        assert r["label"] in LABELS, f"{r['case_id']}: bad label {r['label']!r}"
    return rows


def split(cases, gold):
    """Stratified by gold label, fixed seed: the same labels always give the same split."""
    rng = random.Random(SEED)
    dev, test = [], []
    for label in LABELS:
        group = [c for c in cases if gold[c["case_id"]]["label"] == label]
        rng.shuffle(group)
        k = round(len(group) * DEV_FRACTION)
        dev += group[:k]
        test += group[k:]
    return dev, test


def fmt(x, d=3):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def evaluate(args):
    cases = [json.loads(l) for l in (C.DATA / f"cases_{args.corpus}.jsonl").open(encoding="utf-8")]
    gold = read_labels(args.labels)
    cases = [c for c in cases if c["case_id"] in gold]
    sig = signals(cases, rerank=True)
    cases = [c for c in cases if sig[c["case_id"]].get("kb_size")]
    dev, test = (cases, cases) if args.smoke else split(cases, gold)
    print(f"{len(cases)} labeled cases: dev {len(dev)}, test {len(test)}")

    methods = [m for m in all_methods() if not (args.no_llm and m.uses_llm)]
    if args.only:
        methods = [m for m in methods if m.name.split()[0] in args.only.split(",")]

    preds = {}
    for m in methods:
        m.tune(dev, gold, sig)
        out = {}
        for n, c in enumerate(test, 1):
            out[c["case_id"]] = m.predict(c, sig[c["case_id"]])
            if m.uses_llm and n % 10 == 0:
                print(f"  {m.name}: {n}/{len(test)}", flush=True)
        preds[m.name] = out
        f = macro_f1([gold[c["case_id"]]["label"] for c in test], [out[c["case_id"]][0] for c in test])
        print(f"{m.name:24s} macro-F1 {f:.3f}")

    run = C.EXP / "runs" / f"{args.corpus}_{dt.datetime.now():%Y%m%d_%H%M%S}"
    run.mkdir(parents=True)
    report(run, args, cases, dev, test, gold, sig, methods, preds)
    print(f"-> {run}")


def report(run, args, cases, dev, test, gold, sig, methods, preds):
    docs, idx = {}, {}
    for corpus in {c["corpus"] for c in cases}:
        docs[corpus], idx[corpus] = C.load(corpus), C.index(corpus)
    ids = [c["case_id"] for c in test]
    y = [gold[i]["label"] for i in ids]
    yt = [gold[i]["target"] for i in ids]
    L = []
    w = L.append

    w(f"# Results — {args.corpus}  ({dt.datetime.now():%Y-%m-%d %H:%M})\n")
    if args.smoke:
        w("> **SMOKE RUN: dev = test = all cases. Thresholds are fit on the very cases they are scored on. "
          "These numbers check that the code runs; they are not results.**\n")
    w(f"Labels: `{args.labels}` · cases: {len(cases)} · dev {len(dev)} / test {len(test)} · seed {SEED}\n")
    w("## Class distribution\n")
    w("| split | " + " | ".join(LABELS) + " |\n|---|" + "---|" * len(LABELS))
    for name, part in [("dev", dev), ("test", test)]:
        cnt = Counter(gold[c["case_id"]]["label"] for c in part)
        w(f"| {name} | " + " | ".join(str(cnt[l]) for l in LABELS) + " |")
    by_pool = defaultdict(Counter)
    for c in test:
        by_pool[c["pool"]][gold[c["case_id"]]["label"]] += 1
    w("\nTest cases by sampling pool: " + ", ".join(f"{p} {dict(v)}" for p, v in sorted(by_pool.items())) + "\n")

    # Retrieval ceiling for M2: is the gold target even among the 5 candidates the LLM sees?
    upd = [c for c in test if gold[c["case_id"]]["label"] != "NEW" and gold[c["case_id"]]["target"]]
    if upd:
        def rank_hit(c, k):
            cands = [docs[c["corpus"]][j]["doc_id"] for j in sig[c["case_id"]]["by_cos"][:k]]
            return any(target_hit(d, gold[c["case_id"]]["target"]) for d in cands)
        w(f"Gold target (UPDATE/REDUNDANT, n={len(upd)}) in e5 top-1: {np.mean([rank_hit(c, 1) for c in upd]):.2f}, "
          f"top-5: {np.mean([rank_hit(c, 5) for c in upd]):.2f}, top-10: {np.mean([rank_hit(c, 10) for c in upd]):.2f} "
          "— M2 cannot name a target that is not in its top-5.\n")

    w("## Main table (test split)\n")
    w("| condition | macro-F1 [95% CI] | acc | F1 UPD | F1 NEW | F1 RED | target cond. | target joint | s/decision | tokens/decision | LLM calls |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    rows = {}
    for m in methods:
        p = preds[m.name]
        yp = [p[i][0] for i in ids]
        tp = [p[i][1] for i in ids]
        info = [p[i][2] for i in ids]
        pc = per_class(y, yp)
        ta = target_accuracy(y, yt, yp, tp)
        lo, hi = bootstrap_ci(y, yp, n=args.boot)
        lat = [x["latency_s"] for x in info if "latency_s" in x]
        tok = [(x.get("prompt_tokens") or 0) + (x.get("output_tokens") or 0) for x in info if "latency_s" in x]
        calls = len(lat)
        rows[m.name] = {"yp": yp, "tp": tp, "f1": macro_f1(y, yp)}
        w(f"| {m.name} | {macro_f1(y, yp):.3f} [{lo:.2f}, {hi:.2f}] | {np.mean(np.array(y) == np.array(yp)):.3f} | "
          f"{pc['UPDATE']['f1']:.2f} | {pc['NEW']['f1']:.2f} | {pc['REDUNDANT']['f1']:.2f} | "
          f"{fmt(ta['conditional'], 2)} | {fmt(ta['joint'], 2)} | "
          f"{fmt(sum(lat) / len(ids), 2) if m.uses_llm else '≈0'} | {fmt(sum(tok) / len(ids), 0) if m.uses_llm else '0'} | "
          f"{f'{calls}/{len(ids)}' if m.uses_llm else '–'} |")
    w("\n*s/decision and tokens/decision are averaged over all test cases, so the cascade's cost includes the "
      "cases it settled without the LLM. Latency is wall-clock on the local GPU and includes model load for "
      "the first call.*\n")

    # Significance: every LLM condition against the best baseline, chosen on DEV macro-F1 so the
    # comparison is not picked after seeing test.
    base = [m for m in methods if not m.uses_llm and not m.name.startswith("B0")]
    llms = [m for m in methods if m.uses_llm]
    if base and llms:
        yd = [gold[c["case_id"]]["label"] for c in dev]
        best = max(base, key=lambda m: macro_f1(yd, [m.predict(c, sig[c["case_id"]])[0] for c in dev]))
        w(f"## Significance vs best baseline ({best.name}, chosen on dev)\n")
        w("Correct = right label, and for UPDATE also the right target. Exact McNemar, Holm-adjusted across "
          "the comparisons. The macro-F1 difference uses a paired bootstrap.\n")
        w("| condition | only baseline right | only condition right | McNemar p | Holm p | ΔmacroF1 [95% CI] |")
        w("|---|---|---|---|---|---|")
        ref = correct_vector(y, yt, rows[best.name]["yp"], rows[best.name]["tp"])
        tests = []
        for m in llms:
            cv = correct_vector(y, yt, rows[m.name]["yp"], rows[m.name]["tp"])
            tests.append((m, mcnemar_exact(ref, cv),
                          paired_bootstrap_diff(y, rows[best.name]["yp"], rows[m.name]["yp"], n=args.boot)))
        adj = holm([t[1]["p"] for t in tests])
        for (m, mc, d), p_adj in zip(tests, adj):
            w(f"| {m.name} | {mc['a_only']} | {mc['b_only']} | {mc['p']:.4f} | {p_adj:.4f} | "
              f"{d['diff']:+.3f} [{d['ci'][0]:+.2f}, {d['ci'][1]:+.2f}] |")
        w("")

    w("## Tuned parameters (from dev)\n")
    for m in methods:
        params = {k: v for k, v in vars(m).items() if k.startswith("t_") or k == "label"}
        if params:
            w(f"- {m.name}: " + ", ".join(f"{k}={v}" for k, v in params.items()))
    w("")

    w("## Macro-F1 by sampling pool (test)\n")
    pools = sorted({c["pool"] for c in test})
    w("| condition | " + " | ".join(pools) + " |\n|---|" + "---|" * len(pools))
    for m in methods:
        cells = []
        for pool in pools:
            sub = [k for k, c in enumerate(test) if c["pool"] == pool]
            cells.append(f"{macro_f1([y[k] for k in sub], [rows[m.name]['yp'][k] for k in sub]):.2f} (n={len(sub)})")
        w(f"| {m.name} | " + " | ".join(cells) + " |")
    w("")

    w("## Confusion matrices (rows = gold, cols = predicted: UPD / NEW / RED)\n")
    for m in methods:
        cm = confusion(y, rows[m.name]["yp"])
        w(f"**{m.name}**\n```\n" + "\n".join(f"{l[:3]}  " + " ".join(f"{v:4d}" for v in cm[k])
                                             for k, l in enumerate(LABELS)) + "\n```")

    # H1 error analysis: false-UPDATE (called UPDATE, gold NEW) and missed-UPDATE (gold UPDATE, called
    # something else), per condition, with the flags the annotator attached.
    w("\n## H1 error counts (test)\n")
    w("| condition | false-UPDATE (gold NEW) | missed-UPDATE | wrong target | of which flagged template_trap / reworded |")
    w("|---|---|---|---|---|")
    err_rows = []
    for m in methods:
        fu = mu = wt = trap = rew = 0
        for k, i in enumerate(ids):
            g, pl, pt = y[k], rows[m.name]["yp"][k], rows[m.name]["tp"][k]
            notes = gold[i].get("notes", "")
            kind = None
            if pl == "UPDATE" and g == "NEW":
                kind, fu = "false_update", fu + 1
                trap += "template_trap" in notes
            elif g == "UPDATE" and pl != "UPDATE":
                kind, mu = "missed_update", mu + 1
                rew += "reworded" in notes
            elif g == "UPDATE" and not target_hit(pt, yt[k]):
                kind, wt = "wrong_target", wt + 1
            elif g != pl:
                kind = f"{g.lower()}_as_{pl.lower()}"
            if kind:
                c = test[k]
                d = docs[c["corpus"]][idx[c["corpus"]][c["doc_id"]]]
                err_rows.append({"condition": m.name, "case_id": i, "error": kind, "gold": g, "gold_target": yt[k],
                                 "pred": pl, "pred_target": pt or "", "pool": c["pool"], "title": d["title"],
                                 "notes": notes, "rationale": preds[m.name][i][2].get("rationale", "")})
        w(f"| {m.name} | {fu} | {mu} | {wt} | {trap} / {rew} |")

    (run / "results.md").write_text("\n".join(L), encoding="utf-8")
    with (run / "errors.csv").open("w", encoding="utf-8-sig", newline="") as f:
        if err_rows:
            wr = csv.DictWriter(f, fieldnames=list(err_rows[0]))
            wr.writeheader()
            wr.writerows(err_rows)
    with (run / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["case_id", "split", "gold", "gold_target", "condition", "pred", "pred_target", "info"])
        split_of = {c["case_id"]: "dev" for c in dev}
        split_of.update({c["case_id"]: "test" for c in test})
        for m in methods:
            for i in ids:
                p = preds[m.name][i]
                wr.writerow([i, split_of[i], gold[i]["label"], gold[i]["target"], m.name, p[0], p[1] or "",
                             json.dumps(p[2], ensure_ascii=False)])


def agreement(args):
    a, b = read_labels(args.labels), read_labels(args.other)
    shared = sorted(set(a) & set(b))
    shared = [i for i in shared if "unsure" not in a[i].get("notes", "") + b[i].get("notes", "")]
    la, lb = [a[i]["label"] for i in shared], [b[i]["label"] for i in shared]
    k = kappa(la, lb)
    raw = np.mean(np.array(la) == np.array(lb))
    print(f"{len(shared)} shared cases (unsure excluded): raw agreement {raw:.3f}, Cohen's kappa {k:.3f}")
    print("-> " + ("proceed (kappa >= 0.6)" if k >= 0.6 else "REVISE THE RUBRIC (kappa < 0.6)"))
    for i in shared:
        if a[i]["label"] != b[i]["label"]:
            print(f"  {i}: {a[i]['label']} vs {b[i]['label']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", help="oir | local | kappa")
    ap.add_argument("labels")
    ap.add_argument("other", nargs="?", help="second annotator's CSV (kappa mode)")
    ap.add_argument("--smoke", action="store_true", help="no dev/test split; pipeline check only")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--only", help="comma-separated condition codes, e.g. B2,M2")
    ap.add_argument("--boot", type=int, default=10_000)
    args = ap.parse_args()
    agreement(args) if args.corpus == "kappa" else evaluate(args)
