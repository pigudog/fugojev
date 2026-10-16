"""Score a checkpoint on one split. The test split stays locked unless --allow-test is passed."""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch

from fugojev.checkpoint import load_checkpoint
from fugojev.model import MAX_STATE
from fugojev.questions import consistent_predictions
from fugojev.records import load_records, materialize
from fugojev.split import canonical_split, split_of


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


def score_requests(model, tokenizer, requests, max_state=MAX_STATE) -> dict:
    """Per-question gold and prediction names. Does not shuffle option order.

    Runs the serving path: the state is encoded once per record and every question
    branch reads its cache, so the latency reported is the real per-record cost.
    """
    was_training = model.training
    model.eval()
    buckets = defaultdict(lambda: {"golds": [], "preds": []})
    on_cuda = str(model.device).startswith("cuda")
    if on_cuda:
        torch.cuda.reset_peak_memory_stats()
    latencies = []
    try:
        with torch.no_grad():
            for request in requests:
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
                for prob, qid, label, keys in zip(
                    probabilities, encoded["qids"], encoded["labels"], encoded["keys"]
                ):
                    golds[qid] = keys[label]
                    predicted[qid] = keys[int(prob.argmax())]
                predicted = consistent_predictions(predicted)
                for qid, gold in golds.items():
                    buckets[qid]["golds"].append(gold)
                    buckets[qid]["preds"].append(predicted[qid])
    finally:
        model.train(was_training)
    per_question = {qid: classification_stats(pair["golds"], pair["preds"]) for qid, pair in sorted(buckets.items())}
    correct = sum(gold == pred for pair in buckets.values() for gold, pred in zip(pair["golds"], pair["preds"]))
    total = sum(len(pair["golds"]) for pair in buckets.values())
    micro = {"n": total, "accuracy": (correct / total) if total else None}
    latencies.sort()
    runtime = {
        "latency_ms_per_record_median": latencies[len(latencies) // 2] if latencies else None,
        "latency_ms_per_record_mean": (sum(latencies) / len(latencies)) if latencies else None,
        "peak_memory_gb": (torch.cuda.max_memory_allocated() / 2**30) if on_cuda else None,
    }
    return {"n_records": len(requests), "micro": micro, "questions": per_question, "runtime": runtime}


def records_for_split(records, split):
    return [row for row in records if split_of(str(row["hospital_id"])) == split]


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
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--base", default=None, help="override the backbone path stored in the checkpoint")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    split = canonical_split(args.split)
    if split == "test" and not args.allow_test:
        raise SystemExit("test split is locked; pass --allow-test")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    records = records_for_split(load_records(args.records), split)
    if not records:
        raise SystemExit(f"no records in split {split}")
    tokenizer, model = load_checkpoint(args.run, device, dtype=dtype, base=args.base)
    report = score_requests(model, tokenizer, records, max_state=model.max_state)
    report["split"] = split
    print(format_report(report))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    main()
