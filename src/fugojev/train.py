"""Train the pointer head and an optional LoRA adapter.

This module is the training entry. It is not launched by `python -m fugojev`.
Dev records are the only ones scored during training. Test records are never read.
"""

import argparse
import contextlib
import random
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from fugojev.evaluate import records_for_split, score_requests
from fugojev.model import BRANCH_SLACK, DEFAULT_BACKBONE, DecisionModel, load_tokenizer
from fugojev.records import load_records, materialize, permute_choice_options
from fugojev.checkpoint import save_checkpoint
from fugojev.split import FOLDS, fold_records


def label_name(question):
    """Stable class key: option name, true/false, or the score level index."""
    if question["type"] == "noul":
        return "true" if question["label"] else "false"
    if question["type"] == "score":
        return str(int(question["label"]))
    return str(question["label"])


def class_weights(requests, power=0.5):
    """Frequency weights with mean 1.

    power=1 is inverse frequency. On this set it chases rare FIGO stages that the
    notes barely state. power=0.5 is the milder square-root balance.
    """
    counts = defaultdict(Counter)
    for request in requests:
        for qid, question in request["questions"].items():
            counts[qid][label_name(question)] += 1
    table = {}
    for qid, observed in counts.items():
        total = sum(observed.values())
        raw = {key: (total / count) ** power for key, count in observed.items()}
        mean = sum(raw[key] * observed[key] for key in observed) / total
        table[qid] = {key: weight / mean for key, weight in raw.items()}
    return table


def oversample_requests(requests, question_ids):
    """Repeat records whose named questions carry a label below 40% of that question.

    FIGO is left at its natural frequency. Repeating every rare stage made the
    head chase labels the notes do not actually state.
    """
    wanted = {qid for qid in question_ids if qid}
    if not wanted:
        return list(requests)
    counts = defaultdict(Counter)
    for request in requests:
        for qid, question in request["questions"].items():
            if qid in wanted:
                counts[qid][label_name(question)] += 1
    expanded = []
    for request in requests:
        copies = 1
        for qid, question in request["questions"].items():
            if qid not in counts:
                continue
            observed = counts[qid]
            share = observed[label_name(question)] / sum(observed.values())
            if share < 0.4:
                copies = 3
                break
        expanded.extend([request] * copies)
    return expanded


def example_weight(qid, label, qtype, keys, table):
    if not table or qid not in table:
        return 1.0
    if qtype == "noul":
        key = "true" if label else "false"
    elif qtype == "score":
        key = str(label)
    else:
        key = keys[label]
    return table[qid].get(key, 1.0)


def question_loss(logits, label, qtype, ord_w, weight=1.0):
    """Cross-entropy, plus ranked probability score for ordered `score` questions."""
    target = torch.tensor([label], device=logits.device)
    loss = F.cross_entropy(logits[None], target)
    if qtype == "score" and ord_w > 0:
        prob = F.softmax(logits, -1)
        observed = (torch.arange(prob.numel() - 1, device=logits.device) >= label).to(prob.dtype)
        loss = loss + ord_w * (prob.cumsum(-1)[:-1] - observed).square().mean()
    return loss * weight


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train fugojev on endometrial records")
    parser.add_argument("--records", default="data/ec/records.jsonl")
    parser.add_argument("--out", default="runs/ec")
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lora", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--ord-w", type=float, default=0.3)
    parser.add_argument("--class-weight", choices=["none", "sqrt", "balanced"], default="none")
    parser.add_argument(
        "--oversample", default="",
        help="comma-separated question ids; repeat a record when that question's label is under 40%",
    )
    parser.add_argument("--accum", type=int, default=4)
    parser.add_argument("--max-state", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--dtype", choices=["auto", "fp32", "bf16"], default="auto",
        help="bf16 = fp32 master weights under bf16 autocast (CUDA only); auto picks bf16 on CUDA",
    )
    parser.add_argument("--checkpointing", type=int, choices=[0, 1], default=1,
                        help="gradient checkpointing on the backbone; the row form needs it on 24 GB")
    parser.add_argument("--limit", type=int, default=0, help="if >0, use this many training records")
    parser.add_argument("--fold", type=int, default=None,
                        help="cross-validation: train on the non-test records outside this fold, log the fold")
    parser.add_argument("--folds", type=int, default=FOLDS)
    args = parser.parse_args(argv)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    on_cuda = str(device).startswith("cuda")
    use_bf16 = args.dtype == "bf16" or (args.dtype == "auto" and on_cuda)
    if use_bf16 and not on_cuda:
        raise SystemExit("--dtype bf16 requires CUDA")
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else contextlib.nullcontext()
    out_dir = Path(args.out)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite an existing run: {args.out}")
    rows = load_records(args.records)
    if args.fold is None:
        train_rows = records_for_split(rows, "train")
        dev_rows = records_for_split(rows, "dev")
    else:
        train_rows = fold_records(rows, args.fold, args.folds, held_out=False)
        dev_rows = fold_records(rows, args.fold, args.folds, held_out=True)
    if args.limit:
        train_rows = train_rows[:args.limit]
    if not train_rows:
        raise SystemExit("no training records")

    tokenizer = load_tokenizer(args.backbone)
    model = DecisionModel(
        args.backbone, tokenizer, device, lora=args.lora, head_dim=args.head_dim,
        dtype=torch.float32, max_state=args.max_state, max_branch=args.max_state + BRANCH_SLACK,
    )
    if args.checkpointing:
        model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.lm.config.use_cache = False
    if on_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print(
        f"backbone {args.backbone} hybrid={model.hybrid} special={model.special} "
        f"trainable={sum(p.numel() for p in model.trainable_parameters()) / 1e6:.1f}M "
        f"train={len(train_rows)} dev={len(dev_rows)} max_state={args.max_state} "
        f"bf16_autocast={use_bf16} checkpointing={bool(args.checkpointing)}",
        flush=True,
    )
    groups = []
    backbone = [parameter for parameter in model.lm.parameters() if parameter.requires_grad]
    if backbone:
        groups.append({"params": backbone, "lr": args.lr})
    groups.append({"params": list(model.head.parameters()), "lr": args.lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    power = {"sqrt": 0.5, "balanced": 1.0}.get(args.class_weight)
    weights = class_weights(train_rows, power=power) if power else None
    train_rows = oversample_requests(train_rows, args.oversample.split(","))
    rng = random.Random(args.seed)

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        order = list(train_rows)
        rng.shuffle(order)
        total = 0.0
        micro = 0
        for request in order:
            shuffled = permute_choice_options(request, rng)
            encoded = model.encode(tokenizer, materialize(shuffled))
            with autocast:
                logits_list = model.forward(encoded)
            losses = [
                question_loss(
                    logits, label, qtype, args.ord_w,
                    example_weight(qid, label, qtype, keys, weights),
                )
                for logits, label, qtype, qid, keys in zip(
                    logits_list, encoded["labels"], encoded["qtypes"], encoded["qids"], encoded["keys"]
                )
            ]
            loss = torch.stack(losses).mean()
            (loss / args.accum).backward()
            total += float(loss.detach())
            micro += 1
            if micro == args.accum:
                optimizer.step()
                optimizer.zero_grad()
                micro = 0
        if micro:
            optimizer.step()
            optimizer.zero_grad()
        message = f"epoch {epoch + 1} train_loss {total / len(order):.4f}"
        if dev_rows:
            with autocast:
                report = score_requests(model, tokenizer, dev_rows, max_state=model.max_state)
            accuracy = report["micro"]["accuracy"]
            message += f" dev_acc {accuracy:.3f}" if accuracy is not None else " dev_acc n/a"
            for qid in ("figo", "muscle_invasion"):
                stats = report["questions"].get(qid)
                if stats and stats["accuracy"] is not None:
                    message += f" {qid} {stats['accuracy']:.3f} f1 {stats['macro_f1']:.3f}"
        if on_cuda:
            message += f" peak_mem_gb {torch.cuda.max_memory_allocated() / 2**30:.1f}"
        print(message, flush=True)
        save_checkpoint(model, tokenizer, f"{args.out}/epoch-{epoch + 1}", extra={"args": vars(args)})
    return model


if __name__ == "__main__":
    main()
