"""Per-question temperature + class-prior calibration of the pointer-head probabilities.

    python -m fugojev.evaluate --run weights/ec-v1 --split dev --rows runs/cal/dev-rows.json
    python -m fugojev.calibrate --rows runs/cal/dev-rows.json --out runs/cal/calibration.json

For every question the probability vector is rescaled as
`q ∝ p^(1/T) * w`, with one temperature `T` and a class-weight vector `w` that
mixes the predicted class mass toward the empirical prior. Parameters are chosen
by negative log-likelihood (a proper scoring rule, so a tie in accuracy cannot
pick an overconfident setting). Because the only unlocked split is a 36-patient
dev set, the report shows leave-one-out numbers: each record is scored with
parameters fitted on the other records. The parameters written to `--out` are
fitted on all rows and can be passed to `evaluate` / `predict` as `--calibration`.
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

TEMPERATURES = [round(0.5 + 0.1 * step, 1) for step in range(26)]
MIXES = (0.0, 0.25, 0.5, 0.75, 1.0)
EPS = 1e-6


def apply_calibration(probs, keys, params) -> list[float]:
    """`params` is one question's entry from calibration.json, or None."""
    if not params:
        return list(probs)
    temperature = float(params.get("temperature", 1.0))
    weights = params.get("weights") or {}
    scaled = [max(p, EPS) ** (1.0 / temperature) * float(weights.get(key, 1.0)) for p, key in zip(probs, keys)]
    total = sum(scaled)
    return [value / total for value in scaled]


def _weights(rows, keys, mix) -> dict:
    """Empirical prior over predicted mass, Laplace-smoothed so a class absent from
    a small fitting set is not driven to zero probability."""
    n = len(rows)
    k = len(keys)
    prior = {key: (sum(row["gold"] == key for row in rows) + 1) / (n + k) for key in keys}
    predicted = {key: (sum(row["probs"][index] for row in rows) + 1) / (n + k) for index, key in enumerate(keys)}
    return {key: (prior[key] / predicted[key]) ** mix for key in keys}


def nll(rows, params) -> float:
    total = 0.0
    for row in rows:
        probs = apply_calibration(row["probs"], row["keys"], params)
        total -= math.log(max(probs[row["keys"].index(row["gold"])], EPS))
    return total / len(rows)


def fit(rows) -> dict:
    """Grid search over temperature and prior mix for one question's rows."""
    keys = rows[0]["keys"]
    best = None
    for temperature in TEMPERATURES:
        for mix in MIXES:
            params = {"temperature": temperature, "weights": _weights(rows, keys, mix), "mix": mix}
            loss = nll(rows, params)
            if best is None or loss < best[0] - 1e-9:
                best = (loss, params)
    return best[1]


def reliability(rows, params=None) -> dict:
    """Accuracy, NLL, Brier and expected calibration error (10 confidence bins)."""
    correct = 0
    brier = 0.0
    loss = 0.0
    bins = defaultdict(lambda: [0, 0.0, 0.0])  # count, confidence sum, correct
    for row in rows:
        probs = apply_calibration(row["probs"], row["keys"], params)
        top = max(range(len(probs)), key=probs.__getitem__)
        gold = row["keys"].index(row["gold"])
        hit = int(top == gold)
        correct += hit
        loss -= math.log(max(probs[gold], EPS))
        brier += sum((p - (index == gold)) ** 2 for index, p in enumerate(probs))
        bucket = min(int(probs[top] * 10), 9)
        bins[bucket][0] += 1
        bins[bucket][1] += probs[top]
        bins[bucket][2] += hit
    n = len(rows)
    ece = sum(count / n * abs(conf / count - hits / count) for count, conf, hits in bins.values())
    return {"n": n, "accuracy": correct / n, "nll": loss / n, "brier": brier / n, "ece": ece}


def leave_one_out(rows) -> dict:
    """Each row scored with parameters fitted on the other rows of its question."""
    if len(rows) < 3:
        return reliability(rows)
    held = []
    for index, row in enumerate(rows):
        params = fit(rows[:index] + rows[index + 1:])
        probs = apply_calibration(row["probs"], row["keys"], params)
        held.append({**row, "probs": probs})
    return reliability(held)


def calibrate(rows) -> tuple[dict, dict]:
    by_question = defaultdict(list)
    for row in rows:
        by_question[row["qid"]].append(row)
    params = {}
    report = {}
    for qid, items in sorted(by_question.items()):
        params[qid] = fit(items)
        report[qid] = {
            "raw": reliability(items),
            "leave_one_out": leave_one_out(items),
            "temperature": params[qid]["temperature"],
            "mix": params[qid]["mix"],
        }
    return params, report


def format_report(report) -> str:
    lines = []
    for qid, entry in report.items():
        raw, loo = entry["raw"], entry["leave_one_out"]
        lines.append(
            f"{qid}: n={raw['n']} raw acc={raw['accuracy']:.3f} nll={raw['nll']:.3f} brier={raw['brier']:.3f} "
            f"ece={raw['ece']:.3f} | loo acc={loo['accuracy']:.3f} nll={loo['nll']:.3f} "
            f"brier={loo['brier']:.3f} ece={loo['ece']:.3f} | T={entry['temperature']} mix={entry['mix']}"
        )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fit per-question temperature and class prior")
    parser.add_argument("--rows", required=True, help="rows file written by evaluate --rows")
    parser.add_argument("--out", default="", help="write calibration.json here")
    args = parser.parse_args(argv)
    payload = json.loads(Path(args.rows).read_text(encoding="utf-8"))
    params, report = calibrate(payload["rows"])
    print(format_report(report))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"fitted_on": payload.get("split"), "run": payload.get("run"), **params},
                                  ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return params, report


if __name__ == "__main__":
    main()
