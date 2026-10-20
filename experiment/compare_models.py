"""Put several models' runs side by side (RQ1, RQ2) and test whether they differ.

Every run.py run is one model; this reads their predictions.csv files and writes one table per
LLM condition, plus pairwise exact McNemar tests (Holm-corrected) between the models.

    python experiment/compare_models.py                       # newest run of every model in runs/
    python experiment/compare_models.py runs/A runs/B ...     # exactly these run folders

Output: experiment/runs/model_comparison.md
Only runs made with the same answer key are comparable; the script stops if they differ.
"""

import csv
import itertools
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.metrics import holm, macro_f1, mcnemar_exact

RUNS = Path(__file__).resolve().parent / "runs"
LLM_CONDITIONS = ["M1", "M2", "M3", "M4"]


def run_info(folder):
    head = (folder / "results.md").read_text(encoding="utf-8")
    model = re.search(r"LLM: \*\*(.+?)\*\*", head)
    labels = re.search(r"Labels: `(.+?)`", head)
    # Runs from before run.py had --model were all made with qwen3:4b.
    return {"folder": folder, "model": model.group(1) if model else "qwen3:4b",
            "free_json": "JSON: free" in head, "labels": labels.group(1) if labels else "?",
            "conditions": sum(f"| {c} " in head for c in LLM_CONDITIONS)}


def newest_runs():
    """For every model: the most complete schema-constrained run (most LLM conditions, then newest),
    and likewise the most complete free-JSON run."""
    best = {}
    for f in sorted(RUNS.iterdir()):
        if f.is_dir() and f.name not in ("latest", "old") and (f / "predictions.csv").exists():
            info = run_info(f)
            if not info["conditions"]:
                continue  # rules-only run
            key = (info["model"], info["free_json"])
            if key not in best or info["conditions"] >= best[key]["conditions"]:
                best[key] = info  # sorted by name = by time, so ties go to the newest
    return list(best.values())


def load(info):
    rows = defaultdict(dict)  # condition code -> case_id -> row
    with (info["folder"] / "predictions.csv").open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["split"] == "test":
                r["info"] = json.loads(r["info"] or "{}")
                rows[r["condition"].split()[0]][r["case_id"]] = r
    return rows


def pct(n, total):
    return f"{100 * n / total:.1f}%" if total else "–"


def main(folders):
    infos = [run_info(Path(f)) for f in folders] if folders else newest_runs()
    if not infos:
        sys.exit("No model runs found in experiment/runs/.")
    if len({i["labels"] for i in infos}) > 1:
        sys.exit("These runs used different answer keys and cannot be compared:\n" +
                 "\n".join(f"  {i['folder'].name}: {i['labels']}" for i in infos))
    data = {(i["model"], i["free_json"]): load(i) for i in infos}
    constrained = [i for i in infos if not i["free_json"]]
    L = ["# Model comparison\n", f"Answer key: `{infos[0]['labels']}` · runs: " +
         ", ".join(f"`{i['folder'].name}`" for i in infos) + "\n"]

    for cond in LLM_CONDITIONS:
        present = [i for i in constrained if cond in data[(i["model"], False)]]
        if not present:
            continue
        L += [f"## {cond} (schema-constrained JSON)\n",
              "| model | fully correct | label correct | macro-F1 | mean s | median s | slowest 10% s | tokens/question |",
              "|---|---|---|---|---|---|---|---|"]
        vectors = {}
        for i in present:
            rows = data[(i["model"], False)][cond]
            ids = sorted(rows)
            y = [rows[k]["gold"] for k in ids]
            p = [rows[k]["pred"] for k in ids]
            ok = np.array([rows[k]["result"].startswith("CORRECT") for k in ids], dtype=int)
            vectors[i["model"]] = dict(zip(ids, ok))
            lat = sorted(rows[k]["info"]["latency_s"] for k in ids if "latency_s" in rows[k]["info"])
            tok = [(rows[k]["info"].get("prompt_tokens") or 0) + (rows[k]["info"].get("output_tokens") or 0)
                   for k in ids if "latency_s" in rows[k]["info"]]
            L.append(f"| {i['model']} | {pct(ok.sum(), len(ids))} ({ok.sum()}) | "
                     f"{pct(sum(a == b for a, b in zip(y, p)), len(ids))} | {macro_f1(y, p):.3f} | "
                     + (f"{np.mean(lat):.2f} | {np.median(lat):.2f} | {np.quantile(lat, 0.9):.2f} | {np.mean(tok):,.0f} |"
                        if lat else "– | – | – | – |"))
        pairs = list(itertools.combinations(sorted(vectors), 2))
        if pairs:
            tests = []
            for a, b in pairs:
                ids = sorted(set(vectors[a]) & set(vectors[b]))
                tests.append((a, b, mcnemar_exact([vectors[a][k] for k in ids], [vectors[b][k] for k in ids])))
            adj = holm([t[2]["p"] for t in tests])
            L += ["", "Pairwise exact McNemar on fully-correct answers (Holm-corrected):\n",
                  "| model A | model B | only A right | only B right | p | Holm p | significant (p < 0.05) |",
                  "|---|---|---|---|---|---|---|"]
            for (a, b, t), pa in zip(tests, adj):
                L.append(f"| {a} | {b} | {t['a_only']} | {t['b_only']} | {t['p']:.4f} | {pa:.4f} | "
                         f"{'**yes**' if pa < 0.05 else 'no'} |")
        L.append("")

    # RQ2: JSON stability without the schema constraint
    free = [i for i in infos if i["free_json"]]
    L += ["## JSON stability (free format: the model writes the JSON itself)\n"]
    if free:
        L += ["| model | condition | calls | valid | recovered | invalid | request failed |",
              "|---|---|---|---|---|---|---|"]
        for i in free:
            for cond, rows in sorted(data[(i["model"], True)].items()):
                if cond not in LLM_CONDITIONS:
                    continue
                q = defaultdict(int)
                for r in rows.values():
                    q[r["info"].get("json") or r["info"].get("error_kind") or "?"] += 1
                n = len(rows)
                L.append(f"| {i['model']} | {cond} | {n} | {pct(q['valid'], n)} | {pct(q['recovered'], n)} | "
                         f"{pct(q['invalid_json'], n)} | {pct(q['request_failed'], n)} |")
    else:
        L.append("No free-format runs yet: `run.py <labels> --model <m> --free-json --only M2`.")
    out = RUNS / "model_comparison.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
