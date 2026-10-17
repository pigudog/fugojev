"""Five-fold cross-validation over the 293 non-test patients, with input ablations.

    python -m fugojev.cv train --backbone models/Qwen3.5-0.8B-Base --out runs/cv
    python -m fugojev.cv score --out runs/cv --epoch 3
    python -m fugojev.cv report --out runs/cv

`train` fits one model per fold (train on the other four folds; `--epochs` fixed in
advance, not picked on the held-out fold). `score` runs the held-out fold of every
model through `evaluate` and the rule baseline through `baseline`, under each
ablation, writing per-question rows. `report` pools the rows over all folds and
gives, per question and ablation: accuracy and macro-F1 with a 95% patient-level
bootstrap interval, the paired model-minus-rule difference with its interval, and
an exact McNemar p-value on the discordant pairs.

The test split is never touched. Folds come from `split.fold_of`.
"""

import argparse
import json
import math
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from fugojev.evaluate import ABLATIONS, COMPOSED_QIDS, classification_stats, format_report
from fugojev.questions import QUESTIONS
from fugojev.split import FOLDS

SYSTEMS = ("fugojev", "rule")
# Direct heads first, then the derived 2023 stages (reported, not counted in micro).
REPORTED_QIDS = tuple(QUESTIONS) + tuple(COMPOSED_QIDS)
BOOTSTRAP = 2000


def _run(command):
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def train(args):
    for fold in range(args.folds):
        out = Path(args.out) / f"fold-{fold}"
        if (out / f"epoch-{args.epochs}" / "head.pt").exists():
            print(f"fold {fold}: already trained", flush=True)
            continue
        _run([sys.executable, "-m", "fugojev.train", "--records", args.records, "--backbone", args.backbone,
              "--out", str(out), "--epochs", str(args.epochs), "--lora", str(args.lora),
              "--max-state", str(args.max_state), "--fold", str(fold), "--folds", str(args.folds)])


def score(args):
    ablations = [name for name in args.ablations.split(",") if name]
    for fold in range(args.folds):
        run = Path(args.out) / f"fold-{fold}" / f"epoch-{args.epoch}"
        for ablation in ablations:
            rows_dir = Path(args.out) / "rows"
            model_rows = rows_dir / f"fugojev-fold{fold}-{ablation}.json"
            if not model_rows.exists():
                _run([sys.executable, "-m", "fugojev.evaluate", "--records", args.records, "--run", str(run),
                      "--fold", str(fold), "--folds", str(args.folds), "--dtype", args.dtype,
                      "--ablation", ablation, "--rows", str(model_rows),
                      "--out", str(rows_dir / f"fugojev-fold{fold}-{ablation}-report.json")])
            rule_rows = rows_dir / f"rule-fold{fold}-{ablation}.json"
            if not rule_rows.exists():
                _run([sys.executable, "-m", "fugojev.baseline", "--records", args.records,
                      "--fold", str(fold), "--folds", str(args.folds), "--ablation", ablation,
                      "--rows", str(rule_rows), "--out", str(rows_dir / f"rule-fold{fold}-{ablation}-report.json")])


# ---- pooling ----------------------------------------------------------------


def load_rows(out, system, ablation, folds):
    pooled = []
    for fold in range(folds):
        path = Path(out) / "rows" / f"{system}-fold{fold}-{ablation}.json"
        if not path.exists():
            return None
        for row in json.loads(path.read_text(encoding="utf-8"))["rows"]:
            pooled.append({**row, "patient": (fold, row["index"])})
    return pooled


def _by_patient(rows):
    table = defaultdict(dict)
    for row in rows:
        table[row["patient"]][row["qid"]] = row
    return table


def _stats_for(rows, qid, patients):
    golds, preds = [], []
    for patient in patients:
        row = rows.get(patient, {}).get(qid)
        if row is not None:
            golds.append(row["gold"])
            preds.append(row["pred"])
    return classification_stats(golds, preds)


def _interval(values):
    ordered = sorted(value for value in values if value is not None)
    if not ordered:
        return (None, None)
    low = ordered[int(0.025 * (len(ordered) - 1))]
    high = ordered[int(0.975 * (len(ordered) - 1))]
    return (low, high)


def mcnemar(model_rows, rule_rows, qid):
    """Exact two-sided McNemar on the discordant pairs."""
    b = c = 0
    for patient, questions in model_rows.items():
        if qid not in questions or qid not in rule_rows.get(patient, {}):
            continue
        model_hit = questions[qid]["pred"] == questions[qid]["gold"]
        rule_hit = rule_rows[patient][qid]["pred"] == rule_rows[patient][qid]["gold"]
        if model_hit and not rule_hit:
            b += 1
        elif rule_hit and not model_hit:
            c += 1
    n = b + c
    if n == 0:
        return {"model_only_correct": 0, "rule_only_correct": 0, "p": 1.0}
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return {"model_only_correct": b, "rule_only_correct": c, "p": min(1.0, 2 * tail)}


def pooled_report(out, ablation, folds, seed=0):
    model = load_rows(out, "fugojev", ablation, folds)
    rule = load_rows(out, "rule", ablation, folds)
    if model is None or rule is None:
        return None
    model_rows, rule_rows = _by_patient(model), _by_patient(rule)
    patients = sorted(model_rows)
    rng = random.Random(seed)
    samples = [[rng.choice(patients) for _ in patients] for _ in range(BOOTSTRAP)]
    report = {"ablation": ablation, "n_patients": len(patients), "questions": {}}
    for qid in REPORTED_QIDS:
        point = {system: _stats_for(rows, qid, patients) for system, rows in (("fugojev", model_rows), ("rule", rule_rows))}
        if point["fugojev"]["n"] == 0 or point["rule"]["n"] == 0:
            continue
        boot = {system: {"accuracy": [], "macro_f1": []} for system in SYSTEMS}
        diff = []
        for sample in samples:
            stats = {system: _stats_for(rows, qid, sample) for system, rows in (("fugojev", model_rows), ("rule", rule_rows))}
            for system in SYSTEMS:
                boot[system]["accuracy"].append(stats[system]["accuracy"])
                boot[system]["macro_f1"].append(stats[system]["macro_f1"])
            if stats["fugojev"]["accuracy"] is not None and stats["rule"]["accuracy"] is not None:
                diff.append(stats["fugojev"]["accuracy"] - stats["rule"]["accuracy"])
        entry = {"n": point["fugojev"]["n"], "majority_accuracy": point["fugojev"]["majority_accuracy"]}
        for system in SYSTEMS:
            entry[system] = {
                "accuracy": point[system]["accuracy"], "accuracy_ci": _interval(boot[system]["accuracy"]),
                "macro_f1": point[system]["macro_f1"], "macro_f1_ci": _interval(boot[system]["macro_f1"]),
            }
        entry["difference"] = {
            "accuracy": point["fugojev"]["accuracy"] - point["rule"]["accuracy"],
            "accuracy_ci": _interval(diff),
            "mcnemar": mcnemar(model_rows, rule_rows, qid),
        }
        report["questions"][qid] = entry
    for system, rows in (("fugojev", model_rows), ("rule", rule_rows)):
        correct = total = 0
        for questions in rows.values():
            for qid, row in questions.items():
                if qid in COMPOSED_QIDS:
                    continue
                total += 1
                correct += row["pred"] == row["gold"]
        report[f"micro_{system}"] = correct / total if total else None
    return report


def _fmt(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def markdown(reports) -> str:
    lines = []
    for report in reports:
        lines.append(f"### ablation: {report['ablation']}  ({report['n_patients']} patients, 5 folds pooled)\n")
        lines.append("| question | fugojev acc [95% CI] | rule acc [95% CI] | Δ acc [95% CI] | McNemar p | fugojev F1 | rule F1 | majority |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for qid, entry in report["questions"].items():
            f, r, d = entry["fugojev"], entry["rule"], entry["difference"]
            lines.append(
                f"| {qid} (n={entry['n']}) | {_fmt(f['accuracy'])} [{_fmt(f['accuracy_ci'][0])}, {_fmt(f['accuracy_ci'][1])}] "
                f"| {_fmt(r['accuracy'])} [{_fmt(r['accuracy_ci'][0])}, {_fmt(r['accuracy_ci'][1])}] "
                f"| {d['accuracy']:+.3f} [{_fmt(d['accuracy_ci'][0])}, {_fmt(d['accuracy_ci'][1])}] "
                f"| {d['mcnemar']['p']:.3f} ({d['mcnemar']['model_only_correct']}/{d['mcnemar']['rule_only_correct']}) "
                f"| {_fmt(f['macro_f1'])} | {_fmt(r['macro_f1'])} | {_fmt(entry['majority_accuracy'])} |"
            )
        lines.append(f"| micro | {_fmt(report['micro_fugojev'])} | {_fmt(report['micro_rule'])} | | | | | |\n")
    return "\n".join(lines)


def report(args):
    ablations = [name for name in args.ablations.split(",") if name]
    reports = [item for item in (pooled_report(args.out, ablation, args.folds) for ablation in ablations) if item]
    if not reports:
        raise SystemExit("no pooled rows found; run `score` first")
    table = markdown(reports)
    print(table)
    out = Path(args.out)
    (out / "cv-summary.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "cv-summary.md").write_text(table + "\n", encoding="utf-8")
    return reports


def main(argv=None):
    parser = argparse.ArgumentParser(description="Five-fold cross-validation with ablations")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "score", "report"):
        p = sub.add_parser(name)
        p.add_argument("--records", default="data/ec/records.jsonl")
        p.add_argument("--out", default="runs/cv")
        p.add_argument("--folds", type=int, default=FOLDS)
        if name == "train":
            p.add_argument("--backbone", required=True)
            p.add_argument("--epochs", type=int, default=3)
            p.add_argument("--lora", type=int, default=8)
            p.add_argument("--max-state", type=int, default=4096)
        if name == "score":
            p.add_argument("--epoch", type=int, default=3)
            p.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
        if name in ("score", "report"):
            p.add_argument("--ablations", default=",".join(ABLATIONS))
    args = parser.parse_args(argv)
    {"train": train, "score": score, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
