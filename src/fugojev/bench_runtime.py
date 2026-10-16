"""Serving cost of the fine-tuned model: per-record latency and memory.

    python -m fugojev.bench_runtime --run weights/ec-v1 --split dev
    python -m fugojev.bench_runtime --run weights/ec-v1 --split dev --device cpu --limit 10

Times the serving path (`probs_and_prefix`: state encoded once, every question
branch reads its cache) for the full five-question request and for each question
alone, so the fixed cost of reading the notes separates from the marginal cost of
one more question. Also buckets latency by state length. The first records are
warm-up and are not counted.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from fugojev.checkpoint import load_checkpoint
from fugojev.evaluate import records_for_split
from fugojev.questions import QUESTIONS
from fugojev.records import load_records, materialize
from fugojev.split import canonical_split

WARMUP = 3


def _subset(record, qids):
    questions = {qid: question for qid, question in record["questions"].items() if qid in qids}
    return {**record, "questions": questions} if questions else None


def _summary(values) -> dict:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {
        "n": len(values),
        "median_ms": statistics.median(ordered),
        "mean_ms": statistics.fmean(ordered),
        "p95_ms": ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))],
    }


@torch.no_grad()
def time_requests(model, tokenizer, records, qids, device):
    on_cuda = str(device).startswith("cuda")
    latencies, lengths = [], []
    for index, record in enumerate(records):
        subset = _subset(record, qids)
        if subset is None:
            continue
        encoded = model.encode(tokenizer, materialize(subset))
        if on_cuda:
            torch.cuda.synchronize()
        started = time.perf_counter()
        model.probs_and_prefix(encoded)
        if on_cuda:
            torch.cuda.synchronize()
        if index >= WARMUP:
            latencies.append((time.perf_counter() - started) * 1000)
            lengths.append(encoded["seg"].count(0))
    return latencies, lengths


def main(argv=None):
    parser = argparse.ArgumentParser(description="Serving latency and memory of a checkpoint")
    parser.add_argument("--run", required=True)
    parser.add_argument("--records", default="data/ec/records.jsonl")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--base", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    split = canonical_split(args.split)
    if split == "test" and not args.allow_test:
        raise SystemExit("test split is locked; pass --allow-test")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    on_cuda = str(device).startswith("cuda")
    dtype = torch.bfloat16 if args.dtype == "bf16" and on_cuda else torch.float32
    records = records_for_split(load_records(args.records), split)
    if args.limit:
        records = records[: args.limit]
    tokenizer, model = load_checkpoint(args.run, device, dtype=dtype, base=args.base)
    model.eval()
    if on_cuda:
        torch.cuda.reset_peak_memory_stats()

    result = {"run": str(args.run), "split": split, "device": str(device), "dtype": str(dtype).replace("torch.", ""),
              "records_timed": max(0, len(records) - WARMUP), "by_request": {}, "by_state_tokens": {}}
    all_latencies, lengths = time_requests(model, tokenizer, records, set(QUESTIONS), device)
    result["by_request"]["all_five"] = _summary(all_latencies)
    for qid in QUESTIONS:
        latencies, _ = time_requests(model, tokenizer, records, {qid}, device)
        result["by_request"][qid] = _summary(latencies)
    buckets = {}
    for latency, length in zip(all_latencies, lengths):
        if length is None:
            continue
        key = f"{(length // 1024) * 1024}-{(length // 1024 + 1) * 1024}"
        buckets.setdefault(key, []).append(latency)
    result["by_state_tokens"] = {key: _summary(values) for key, values in sorted(buckets.items())}
    if on_cuda:
        result["peak_memory_gb"] = torch.cuda.max_memory_allocated() / 2**30
        result["gpu"] = torch.cuda.get_device_name(0)

    print(f"{result['device']} {result['dtype']}  records timed {result['records_timed']}")
    for name, stats in result["by_request"].items():
        if stats.get("n"):
            print(f"  {name:16s} median {stats['median_ms']:8.1f} ms  p95 {stats['p95_ms']:8.1f} ms")
    for key, stats in result["by_state_tokens"].items():
        print(f"  state {key:12s} median {stats['median_ms']:8.1f} ms  (n={stats['n']})")
    if on_cuda:
        print(f"  peak memory {result['peak_memory_gb']:.1f} GB on {result['gpu']}")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    main()
