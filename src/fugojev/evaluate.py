"""Score a checkpoint on one split. The test split stays locked unless --allow-test is passed."""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch

from fugojev.checkpoint import load_checkpoint
from fugojev.model import MAX_STATE
from fugojev.figo2023 import STAGES_2023, molecular_from_stage, stage_from_predictions
from fugojev.questions import consistent_predictions, with_all_components
from fugojev.records import load_records, materialize
from fugojev.split import FOLDS, canonical_split, fold_records, split_of

COMPOSED_QIDS = ("figo2023_composed", "figo2023_composed_given")

# Ablation names only; `fugojev.ablate` imports from `baseline`, which imports this module.
ABLATIONS = ("none", "paraphrase", "strip-template", "pathology-only")


def classification_stats(golds, preds) -> dict:
    if not golds:
        return {"n": 0, "accuracy": None, "macro_f1": None, "majority_accuracy": None}
    correct = sum(gold == pred for gold, pred in zip(golds, preds))
    majority = max(set(golds), key=golds.count)
    majority_correct = sum(gold == majority for gold in golds)
    f1_scores = []
    for label in sorted(set(golds)):
        tp = sum(gold == label and pred == label for gold, pred in zip(golds, preds))
        fp = sum(gold != label and pred == label for gold, pred in zip(golds, preds))
        fn = sum(gold == label and pred != label for gold, pred in zip(golds, preds))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1_scores.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return {
        "n": len(golds),
        "accuracy": correct / len(golds),
        "macro_f1": sum(f1_scores) / len(f1_scores),
        "majority_accuracy": majority_correct / len(golds),
    }


def report_from_buckets(buckets, n_records, runtime=None, system="fugojev") -> dict:
    """The one report shape every system writes: `buckets` maps qid -> {golds, preds}.
    `compare.py` reads this shape back, so rule baseline, zero-shot probe and the
    fine-tuned model all land in the same table."""
    per_question = {qid: classification_stats(pair["golds"], pair["preds"]) for qid, pair in sorted(buckets.items())}
    # Derived answers (the composed 2023 stage) are reported per question but do not
    # enter micro accuracy, which counts each head once.
    heads = {qid: pair for qid, pair in buckets.items() if qid not in COMPOSED_QIDS}
    correct = sum(gold == pred for pair in heads.values() for gold, pred in zip(pair["golds"], pair["preds"]))
    total = sum(len(pair["golds"]) for pair in heads.values())
    micro = {"n": total, "accuracy": (correct / total) if total else None}
    report = {"system": system, "n_records": n_records, "micro": micro, "questions": per_question}
    if runtime is not None:
        report["runtime"] = runtime
    return report


def add_composed_stages(golds: dict, predicted: dict, record_rows: dict, index: int) -> None:
    """Two derived FIGO 2023 answers next to the direct head: the stage composed from
    the component heads with the molecular class read from the notes, and the same
    with the molecular class supplied as structured input (as a clinic with an
    outside sequencing report would). Both are scored against the gold 2023 stage."""
    gold = golds["figo2023"]
    notes_only = stage_from_predictions(predicted) or "none"
    given = stage_from_predictions(predicted, molecular_from_stage(gold) or "NSMP") or "none"
    for qid, value in zip(COMPOSED_QIDS, (notes_only, given)):
        golds[qid] = gold
        predicted[qid] = value
        record_rows[qid] = {"index": index, "qid": qid, "keys": list(STAGES_2023), "gold": gold, "probs": None}


def score_requests(model, tokenizer, requests, max_state=MAX_STATE, calibration=None, rows=None) -> dict:
    """Per-question gold and prediction names. Does not shuffle option order.

    Runs the serving path: the state is encoded once per record and every question
    branch reads its cache, so the latency reported is the real per-record cost.
    `calibration` is the mapping written by `calibrate.py`; `rows`, when a list, is
    filled with one `{index, qid, keys, gold, probs}` entry per question so the
    probabilities can be calibrated afterwards.
    """
    from fugojev.calibrate import apply_calibration

    was_training = model.training
    model.eval()
    buckets = defaultdict(lambda: {"golds": [], "preds": []})
    on_cuda = str(model.device).startswith("cuda")
    if on_cuda:
        torch.cuda.reset_peak_memory_stats()
    latencies = []
    try:
        with torch.no_grad():
            for index, request in enumerate(requests):
                placeholders = set()
                if "figo2023" in request["questions"]:
                    request, placeholders = with_all_components(request)
                encoded = model.encode(tokenizer, materialize(request), max_state=max_state)
                if on_cuda:
                    torch.cuda.synchronize()
                started = time.perf_counter()
                probabilities, _ = model.probs_and_prefix(encoded)
                if on_cuda:
                    torch.cuda.synchronize()
                latencies.append((time.perf_counter() - started) * 1000)
                golds = {}
                predicted = {}
                record_rows = {}
                for prob, qid, label, keys in zip(
                    probabilities, encoded["qids"], encoded["labels"], encoded["keys"]
                ):
                    values = [float(p) for p in prob]
                    if calibration:
                        values = apply_calibration(values, keys, calibration.get(qid))
                    record_rows[qid] = {"index": index, "qid": qid, "keys": list(keys),
                                        "gold": keys[label], "probs": values}
                    golds[qid] = keys[label]
                    predicted[qid] = keys[max(range(len(values)), key=values.__getitem__)]
                predicted = consistent_predictions(predicted)
                if "figo2023" in golds:
                    add_composed_stages(golds, predicted, record_rows, index)
                for qid, gold in golds.items():
                    if qid in placeholders:
                        continue
                    buckets[qid]["golds"].append(gold)
                    buckets[qid]["preds"].append(predicted[qid])
                    if rows is not None:
                        rows.append({**record_rows[qid], "pred": predicted[qid]})
    finally:
        model.train(was_training)
    latencies.sort()
    runtime = {
        "latency_ms_per_record_median": latencies[len(latencies) // 2] if latencies else None,
        "latency_ms_per_record_mean": (sum(latencies) / len(latencies)) if latencies else None,
        "peak_memory_gb": (torch.cuda.max_memory_allocated() / 2**30) if on_cuda else None,
    }
    return report_from_buckets(buckets, len(requests), runtime)


def records_for_split(records, split):
    return [row for row in records if split_of(str(row["hospital_id"])) == split]


def select_records(args):
    """Records named by `--split` or `--fold`, with the test lock and the ablation applied.
    Returns (records, split name used in reports)."""
    from fugojev.ablate import ablate_records

    all_records = load_records(args.records)
    if getattr(args, "fold", None) is not None:
        records = fold_records(all_records, args.fold, args.folds, held_out=True)
        split = f"fold-{args.fold}/{args.folds}"
    else:
        split = canonical_split(args.split)
        if split == "test" and not args.allow_test:
            raise SystemExit("test split is locked; pass --allow-test")
        records = records_for_split(all_records, split)
    if not records:
        raise SystemExit(f"no records in {split}")
    ablation = getattr(args, "ablation", "none")
    if ablation != "none":
        records = ablate_records(records, ablation)
    return records, split


def format_report(report) -> str:
    accuracy = report["micro"]["accuracy"]
    shown = "n/a" if accuracy is None else f"{accuracy:.3f}"
    lines = [f"records {report['n_records']}  questions {report['micro']['n']}  micro_acc {shown}"]
    for qid, stats in report["questions"].items():
        lines.append(
            f"{qid}: n={stats['n']} acc={stats['accuracy']:.3f} "
            f"macro_f1={stats['macro_f1']:.3f} majority={stats['majority_accuracy']:.3f}"
        )
    runtime = report.get("runtime") or {}
    median = runtime.get("latency_ms_per_record_median")
    if median is not None:
        memory = runtime.get("peak_memory_gb")
        shown_memory = "n/a" if memory is None else f"{memory:.1f} GB"
        lines.append(f"latency_ms_per_record median={median:.1f} mean={runtime['latency_ms_per_record_mean']:.1f}  peak_memory={shown_memory}")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Score a fugojev checkpoint")
    parser.add_argument("--run", required=True)
    parser.add_argument("--records", default="data/ec/records.jsonl")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--fold", type=int, default=None, help="score the held-out records of this CV fold instead of --split")
    parser.add_argument("--folds", type=int, default=FOLDS)
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--ablation", choices=ABLATIONS, default="none", help="input ablation applied before scoring")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--base", default=None, help="override the backbone path stored in the checkpoint")
    parser.add_argument("--calibration", default="", help="calibration.json from fugojev.calibrate")
    parser.add_argument("--rows", default="", help="write per-question probability rows here (for calibrate.py)")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    records, split = select_records(args)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    tokenizer, model = load_checkpoint(args.run, device, dtype=dtype, base=args.base)
    calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8")) if args.calibration else None
    rows = [] if args.rows else None
    report = score_requests(model, tokenizer, records, max_state=model.max_state, calibration=calibration, rows=rows)
    report["split"] = split
    report["ablation"] = args.ablation
    report["run"] = str(args.run)
    if calibration:
        report["calibration"] = str(args.calibration)
    print(format_report(report))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if rows is not None:
        out = Path(args.rows)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"split": split, "run": str(args.run), "rows": rows}, ensure_ascii=False) + "\n",
                       encoding="utf-8")
    return report


if __name__ == "__main__":
    main()
