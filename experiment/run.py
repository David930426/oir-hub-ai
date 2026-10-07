"""Run every condition on the labeled cases and write the results.

Each case's input is a FILE (data/files/, made by fetch_files.py): the bulletin's PDF/Word
attachment, or the bulletin written out as .md when it has none. Each method reads the file,
compares it with the bulletins posted before it, and answers UPDATE / NEW / REDUNDANT.

    python experiment/run.py experiment/data/labels_oir_DG.csv              # the real experiment
    python experiment/run.py experiment/data/labels_oir_DG.csv --no-llm     # baselines only (seconds)
    python experiment/run.py kappa experiment/data/labels_oir_A.csv experiment/data/labels_oir_B.csv

Output goes to experiment/runs/<timestamp>/:
    results.md       a plain-language summary (% correct, why wrong, conclusion), then every table
    predictions.csv  every answer of every method, each marked CORRECT or WRONG with the reason
    errors.csv       only the wrong answers, with the AI's own rationale
"""

import argparse
import csv
import datetime as dt
import json
import random
import shutil
from collections import Counter, defaultdict

import numpy as np

from core import corpus as C
from core.methods import all_methods, signals
from core.metrics import (LABELS, bootstrap_ci, confusion, correct_vector, holm, kappa, macro_f1,
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


# Every answer is one of these. "Correct" means the right label and, for UPDATE, the right old
# bulletin too - retiring the wrong document is still a wrong answer for the office.
OUTCOMES = {
    "correct_update": "correct UPDATE (right old bulletin)",
    "correct_new": "correct NEW",
    "correct_redundant": "correct REDUNDANT",
    "false_update": "false UPDATE (said UPDATE, really NEW)",
    "missed_update": "missed UPDATE (really UPDATE, said NEW/REDUNDANT)",
    "wrong_target": "wrong target (UPDATE, but wrong old bulletin)",
    "other": "other (REDUNDANT mixed up with NEW/UPDATE)",
}


def outcome(gold_label, gold_target, label, target):
    if label == gold_label:
        if label != "UPDATE":
            return f"correct_{label.lower()}"
        return "correct_update" if target_hit(target, gold_target) else "wrong_target"
    if label == "UPDATE" and gold_label == "NEW":
        return "false_update"
    if gold_label == "UPDATE":
        return "missed_update"
    return "other"


def explain(gold_label, gold_target, label, target):
    """One line a person can read: what the method said, what the answer key says, verdict."""
    said = f"{label} {target or ''}".strip()
    key = f"{gold_label} {gold_target or ''}".strip()
    o = outcome(gold_label, gold_target, label, target)
    return ("CORRECT" if o.startswith("correct") else f"WRONG ({OUTCOMES[o].split(' (')[0]})") + \
        f": said {said}; answer key {key}"


def pct(n, total):
    return f"{100 * n / total:.1f}% ({n})" if total else "–"


def summary(methods, rows, y, yt, sig_results, labels_path):
    """The plain-language part at the top of results.md: percentages, averages, conclusion."""
    n = len(y)
    out = []
    w = out.append
    w("## Summary in plain words\n")
    if "draft" in str(labels_path):
        w("> These scores use the **draft** answer key, so treat them as practice numbers.\n")
    w(f"Every method answered the same **{n} test questions**. An answer counts as **correct** when the label "
      "matches the answer key and, for UPDATE, it also names the right old bulletin. Percentages are out of "
      f"all {n} questions. \"Right label\" ignores which old bulletin was named.\n")

    counts = {}
    for m in methods:
        counts[m.name] = Counter(outcome(g, gt, l, t) for g, gt, l, t in
                                 zip(y, yt, rows[m.name]["yp"], rows[m.name]["tp"]))

    w("### How often each method was right\n")
    w("| method | ✅ correct | ❌ wrong | right label (ignoring the old bulletin) | correct UPDATE | correct NEW | correct REDUNDANT |")
    w("|---|---|---|---|---|---|---|")
    groups = {"rules (B1–B4)": [], "AI (M1–M4)": []}
    for m in methods:
        c = counts[m.name]
        right = sum(v for k, v in c.items() if k.startswith("correct"))
        label_right = sum(g == l for g, l in zip(y, rows[m.name]["yp"]))
        w(f"| {m.name} | **{pct(right, n)}** | {pct(n - right, n)} | {pct(label_right, n)} | {pct(c['correct_update'], n)} | "
          f"{pct(c['correct_new'], n)} | {pct(c['correct_redundant'], n)} |")
        if m.name[:2] in ("B1", "B2", "B3", "B4"):
            groups["rules (B1–B4)"].append(right)
        elif m.name.startswith("M"):
            groups["AI (M1–M4)"].append(right)
    for g, vals in groups.items():
        if vals:
            avg = sum(vals) / len(vals)
            w(f"| *average of {g}* | *{100 * avg / n:.1f}%* | *{100 * (n - avg) / n:.1f}%* | | | | |")
    w("")

    w("### Why the wrong answers were wrong\n")
    w("Each column is the share of all questions that went wrong in that way; the last column is the most "
      "common mistake as a share of that method's mistakes.\n")
    kinds = ["false_update", "missed_update", "wrong_target", "other"]
    w("| method | " + " | ".join(OUTCOMES[k].split(" (")[0] for k in kinds) + " | most common mistake |")
    w("|---|" + "---|" * (len(kinds) + 1))
    for m in methods:
        c = counts[m.name]
        wrong = sum(c[k] for k in kinds)
        top = max(kinds, key=lambda k: c[k])
        top_txt = f"{OUTCOMES[top].split(' (')[0]}: {100 * c[top] / wrong:.0f}% of its mistakes" if wrong else "–"
        w(f"| {m.name} | " + " | ".join(pct(c[k], n) for k in kinds) + f" | {top_txt} |")
    w("")
    w("*false UPDATE* = said UPDATE but it is really NEW (fooled by a look-alike) · *missed UPDATE* = it really "
      "replaces an old bulletin but the method said NEW/REDUNDANT · *wrong target* = said UPDATE correctly but "
      "named the wrong old bulletin · *other* = REDUNDANT confused with NEW or UPDATE.\n")

    # conclusion, generated from the numbers above
    right = {m.name: sum(v for k, v in counts[m.name].items() if k.startswith("correct")) for m in methods}
    f1 = {m.name: rows[m.name]["f1"] for m in methods}
    best_acc = max(right, key=right.get)
    best_f1 = max(f1, key=f1.get)
    real = [k for k in right if not k.startswith("B0")] or list(right)
    worst = min(real, key=right.get)
    w("### Conclusion\n")
    if any(m.name.startswith("B0") for m in methods):
        b0 = next(m.name for m in methods if m.name.startswith("B0"))
        b0_label = sum(g == l for g, l in zip(y, rows[b0]["yp"]))
        w(f"- **The bar to beat:** always answering UPDATE ({b0}) already gets the label right {pct(b0_label, n)} "
          "of the time, because most test questions are UPDATEs - but it never names which old bulletin, so it is "
          "fully correct 0% of the time. A method is only useful if it clearly beats this; macro-F1 is the fairer "
          "score because it also rewards getting NEW and REDUNDANT right.")
    w(f"- **Most often right:** {best_acc}, {pct(right[best_acc], n)} correct.")
    w(f"- **Best on the main score (macro-F1):** {best_f1}, {f1[best_f1]:.3f}." +
      (" Same method as above." if best_f1 == best_acc else ""))
    w(f"- **Least often right (not counting B0):** {worst}, {pct(right[worst], n)} correct.")
    for m in methods:
        c = counts[m.name]
        wrong = sum(c[k] for k in kinds)
        if m.name in (best_acc, best_f1) and wrong:
            top = max(kinds, key=lambda k: c[k])
            w(f"- **Main weakness of {m.name}:** {OUTCOMES[top]} - {100 * c[top] / wrong:.0f}% of its mistakes.")
    if sig_results:
        base = sig_results[0][1]
        w(f"- **AI vs rules** - each AI setup is compared with **{base}**, the rule that scored best on the "
          "*practice* questions (chosen there so the comparison is not picked after seeing the test). "
          "\"Significant\" means p < 0.05 after correcting for the several comparisons:")
    for name, base, p_adj, diff, mc in sig_results:
        ai, rule = right[name], right[base]
        if p_adj < 0.05:
            verdict = "significantly **more** often correct" if mc["b_only"] > mc["a_only"] \
                else "significantly **less** often correct"
        else:
            verdict = "**not significantly different** in how often it is correct"
        w(f"  - {name}: {verdict} ({pct(ai, n)} vs {pct(rule, n)}, p = {p_adj:.3f}). "
          f"On macro-F1 the difference is {diff:+.3f}.")
    w("")
    return out


def fmt(x, d=3):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def load_cases(gold):
    """The labeled cases, each with its input file read in as case["query"]."""
    if not C.MANIFEST.exists():
        raise SystemExit("No input files yet - run: python experiment/fetch_files.py")
    files = {json.loads(l)["case_id"]: json.loads(l) for l in C.MANIFEST.open(encoding="utf-8")}
    cases = []
    for line in (C.DATA / "cases_oir.jsonl").open(encoding="utf-8"):
        c = json.loads(line)
        if c["case_id"] not in gold:
            continue
        f = files[c["case_id"]]
        title, text = C.read_file(C.DATA / f["file"])
        c["query"] = {"title": title, "text": text, "date": c["date"]}
        c["kind"] = "attachment" if f["kind"].startswith("attachment") else "bulletin-md"
        c["file"] = f["file"]
        cases.append(c)
    return cases


def evaluate(args):
    gold = read_labels(args.labels)
    cases = load_cases(gold)
    sig = signals(cases, C.Library.load(), rerank=True)
    cases = [c for c in cases if sig[c["case_id"]].get("kb_size")]
    dev, test = split(cases, gold)
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

    run = C.EXP / "runs" / f"{dt.datetime.now():%Y%m%d_%H%M%S}"
    run.mkdir(parents=True)
    report(run, args, cases, dev, test, gold, sig, methods, preds)
    # The newest results are always also in runs/latest/, so there is one obvious place to look.
    latest = run.parent / "latest"
    shutil.rmtree(latest, ignore_errors=True)
    shutil.copytree(run, latest)
    print(f"-> {run}")
    print(f"   newest results also in: {latest / 'results.md'}  (open it, then Ctrl+Shift+V for the preview)")


def report(run, args, cases, dev, test, gold, sig, methods, preds):
    ids = [c["case_id"] for c in test]
    y = [gold[i]["label"] for i in ids]
    yt = [gold[i]["target"] for i in ids]
    L = []
    w = L.append

    w(f"# Results  ({dt.datetime.now():%Y-%m-%d %H:%M})\n")
    kinds = Counter(c["kind"] for c in test)
    w(f"Input files (test): {kinds['attachment']} real attachments (PDF/Word), "
      f"{kinds['bulletin-md']} bulletins written as .md (no readable attachment).\n")
    w(f"Labels: `{args.labels}` · cases: {len(cases)} · dev {len(dev)} / test {len(test)} · seed {SEED}\n")
    summary_at = len(L)
    sig_results = []
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
            cands = sig[c["case_id"]]["by_cos"][:k]
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
            sig_results.append((m.name, best.name, p_adj, d["diff"], mc))
        w("")

    w("## Tuned parameters (from dev)\n")
    for m in methods:
        params = {k: v for k, v in vars(m).items() if k.startswith("t_") or k == "label"}
        if params:
            w(f"- {m.name}: " + ", ".join(f"{k}={v}" for k, v in params.items()))
    w("")

    w("## Macro-F1 by input file type (test)\n")
    w("| condition | attachment (PDF/Word) | bulletin-md |\n|---|---|---|")
    for m in methods:
        cells = []
        for kind in ("attachment", "bulletin-md"):
            sub = [k for k, c in enumerate(test) if c["kind"] == kind]
            cells.append(f"{macro_f1([y[k] for k in sub], [rows[m.name]['yp'][k] for k in sub]):.2f} (n={len(sub)})"
                         if sub else "–")
        w(f"| {m.name} | " + " | ".join(cells) + " |")
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
                err_rows.append({"condition": m.name, "case_id": i, "error": kind, "gold": g, "gold_target": yt[k],
                                 "pred": pl, "pred_target": pt or "", "pool": c["pool"], "input": c["kind"],
                                 "file": c["file"], "title": c["query"]["title"],
                                 "notes": notes, "rationale": preds[m.name][i][2].get("rationale", "")})
        w(f"| {m.name} | {fu} | {mu} | {wt} | {trap} / {rew} |")

    L[summary_at:summary_at] = summary(methods, rows, y, yt, sig_results, args.labels)
    (run / "results.md").write_text("\n".join(L), encoding="utf-8")
    with (run / "errors.csv").open("w", encoding="utf-8-sig", newline="") as f:
        if err_rows:
            wr = csv.DictWriter(f, fieldnames=list(err_rows[0]))
            wr.writeheader()
            wr.writerows(err_rows)
    with (run / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["case_id", "split", "condition", "result", "gold", "gold_target", "pred", "pred_target",
                     "title", "info"])
        title_of = {c["case_id"]: c["query"]["title"] for c in test}
        split_of = {c["case_id"]: "dev" for c in dev}
        split_of.update({c["case_id"]: "test" for c in test})
        for m in methods:
            for i in ids:
                p = preds[m.name][i]
                wr.writerow([i, split_of[i], m.name, explain(gold[i]["label"], gold[i]["target"], p[0], p[1]),
                             gold[i]["label"], gold[i]["target"], p[0], p[1] or "", title_of[i],
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
    ap.add_argument("labels", help="answer key CSV, or the word kappa to compare two annotators")
    ap.add_argument("other", nargs="*", help="kappa mode: the two annotators' CSVs")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--only", help="comma-separated condition codes, e.g. B2,M2")
    ap.add_argument("--boot", type=int, default=10_000)
    args = ap.parse_args()
    if args.labels == "kappa":
        args.labels, args.other = args.other
        agreement(args)
    else:
        evaluate(args)
