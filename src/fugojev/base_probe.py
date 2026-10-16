"""Zero-shot baseline: the *base* backbone with no adapter and no pointer head.

    python -m fugojev.base_probe --model models/Qwen3.5-0.8B-Base --split dev
    python -m fugojev.base_probe --model models/Qwen2.5-0.5B --split dev --out runs/probe/q25-dev.json

Each question becomes a completion prompt `<state>\\n<instructions>\\nA. …\\nB. …\\n答案：`
and the answer is the softmax over the next-token logits of the option letters.
This is what the fine-tuned adapter and pointer head have to beat, so it answers
"how much of the score is the backbone, how much is the training".
The state is the same text the fine-tuned model reads, tail-kept to `--max-state`.
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from fugojev.evaluate import format_report, records_for_split, report_from_buckets
from fugojev.model import MAX_STATE, take_state_tail
from fugojev.questions import consistent_predictions
from fugojev.records import load_records, materialize
from fugojev.split import canonical_split

LETTERS = "ABCDEFGHIJKLMNOP"


def build_prompt(state: str, question: dict) -> str:
    body = "\n".join(f"{LETTERS[index]}. {option}" for index, option in enumerate(question["options"]))
    return f"{state}\n\n{question['instr']}\n{body}\n答案："


def letter_ids(tokenizer) -> list[int]:
    ids = []
    for letter in LETTERS:
        encoded = tokenizer.encode(letter, add_special_tokens=False)
        ids.append(encoded[0])
    if len(set(ids)) != len(ids):
        raise ValueError("option letters do not map to distinct tokens")
    return ids


@torch.no_grad()
def probe(model, tokenizer, records, device, max_state=MAX_STATE, apply_rule=False, rows=None):
    ids = letter_ids(tokenizer)
    buckets = defaultdict(lambda: {"golds": [], "preds": []})
    on_cuda = str(device).startswith("cuda")
    if on_cuda:
        torch.cuda.reset_peak_memory_stats()
    latencies = []
    for index, record in enumerate(records):
        materialized = materialize(record)
        state_ids = tokenizer(materialized["state"], add_special_tokens=False)["input_ids"]
        state_ids, _ = take_state_tail(state_ids, max_state)
        state = tokenizer.decode(state_ids)
        golds, predicted = {}, {}
        if on_cuda:
            torch.cuda.synchronize()
        started = time.perf_counter()
        for question in materialized["questions"]:
            prompt = build_prompt(state, question)
            encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
            hidden = model.model(**encoded).last_hidden_state[0, -1]
            logits = model.lm_head(hidden).float()
            k = len(question["options"])
            probs = torch.softmax(logits[ids[:k]], -1).tolist()
            keys = question["keys"]
            golds[question["qid"]] = keys[question["label"]]
            predicted[question["qid"]] = keys[max(range(k), key=probs.__getitem__)]
            if rows is not None:
                rows.append({"index": index, "qid": question["qid"], "keys": list(keys),
                             "gold": keys[question["label"]], "probs": probs})
        if on_cuda:
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - started) * 1000)
        if apply_rule:
            predicted = consistent_predictions(predicted)
        for qid, gold in golds.items():
            buckets[qid]["golds"].append(gold)
            buckets[qid]["preds"].append(predicted[qid])
    latencies.sort()
    runtime = {
        "latency_ms_per_record_median": latencies[len(latencies) // 2] if latencies else None,
        "latency_ms_per_record_mean": (sum(latencies) / len(latencies)) if latencies else None,
        "peak_memory_gb": (torch.cuda.max_memory_allocated() / 2**30) if on_cuda else None,
    }
    return report_from_buckets(buckets, len(records), runtime, system="zero-shot letter logits")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Zero-shot letter-logit probe of a base model")
    parser.add_argument("--model", required=True, help="base model path or hub id")
    parser.add_argument("--records", default="data/ec/records.jsonl")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--max-state", type=int, default=MAX_STATE)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--consistency", action="store_true", help="apply the muscle→FIGO rule, as evaluate does")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--rows", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    split = canonical_split(args.split)
    if split == "test" and not args.allow_test:
        raise SystemExit("test split is locked; pass --allow-test")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.dtype == "bf16" and str(device).startswith("cuda") else torch.float32
    records = records_for_split(load_records(args.records), split)
    if args.limit:
        records = records[: args.limit]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, attn_implementation="sdpa").to(device).eval()
    rows = [] if args.rows else None
    report = probe(model, tokenizer, records, device, args.max_state, args.consistency, rows)
    report.update({"split": split, "model": str(args.model), "consistency_rule": bool(args.consistency)})
    print(format_report(report))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if rows is not None:
        out = Path(args.rows)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"split": split, "run": str(args.model), "rows": rows}, ensure_ascii=False) + "\n",
                       encoding="utf-8")
    return report


if __name__ == "__main__":
    main()
