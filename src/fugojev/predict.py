"""Answer the five endometrial questions for one note bundle.

    python -m fugojev.predict --run runs/ec-q35/epoch-3 --state note.txt

`--state` is a text file (or `-` for stdin) laid out like the training states:
`[门诊]`, `[入院]`, `[影像]`, `[病理]`, `[手术]` blocks, each keeping its conclusion
at the end. Output is one line per question with the chosen option and the full
probability vector over the allowed options. Nothing is generated.
"""

import argparse
import json
import sys

import torch

from fugojev.checkpoint import load_checkpoint
from fugojev.questions import QUESTIONS, consistent_predictions
from fugojev.records import materialize

_PLACEHOLDER = {"noul": False, "choice": None, "score": 0}


def request_for(state: str, question_ids=None) -> dict:
    """A labelled-request shell for unlabelled text. Labels are placeholders and
    are never read on this path."""
    questions = {}
    for qid, spec in QUESTIONS.items():
        if question_ids and qid not in question_ids:
            continue
        criteria = spec["criteria"]
        label = _PLACEHOLDER[spec["type"]]
        if spec["type"] == "choice":
            label = next(iter(criteria))
        questions[qid] = {
            "type": spec["type"],
            "instructions": spec["instructions"],
            "criteria": dict(criteria) if isinstance(criteria, dict) else list(criteria),
            "label": label,
        }
    return {"state": state, "questions": questions}


@torch.no_grad()
def predict(model, tokenizer, state: str, question_ids=None) -> dict:
    encoded = model.encode(tokenizer, materialize(request_for(state, question_ids)))
    probabilities, _ = model.probs_and_prefix(encoded)
    answers = {}
    for prob, qid, keys in zip(probabilities, encoded["qids"], encoded["keys"]):
        values = [float(p) for p in prob]
        answers[qid] = {
            "answer": keys[int(prob.argmax())],
            "probabilities": dict(zip(keys, values)),
        }
    adjusted = consistent_predictions({qid: item["answer"] for qid, item in answers.items()})
    for qid, item in answers.items():
        item["answer"] = adjusted[qid]
    answers["_state_truncated"] = bool(encoded["state_truncated"])
    return answers


def main(argv=None):
    parser = argparse.ArgumentParser(description="Answer the endometrial questions for one state")
    parser.add_argument("--run", required=True, help="checkpoint directory, e.g. runs/ec-q35/epoch-3")
    parser.add_argument("--state", required=True, help="path to a state text file, or - for stdin")
    parser.add_argument("--questions", default="", help="comma-separated subset of question ids")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--base", default=None, help="override the backbone path stored in the checkpoint")
    parser.add_argument("--json", action="store_true", help="print one JSON object instead of lines")
    args = parser.parse_args(argv)
    state = sys.stdin.read() if args.state == "-" else open(args.state, encoding="utf-8").read()
    if not state.strip():
        raise SystemExit("empty state")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.dtype == "bf16" and str(device).startswith("cuda") else torch.float32
    tokenizer, model = load_checkpoint(args.run, device, dtype=dtype, base=args.base)
    wanted = [qid for qid in args.questions.split(",") if qid] or None
    answers = predict(model, tokenizer, state, wanted)
    if args.json:
        print(json.dumps(answers, ensure_ascii=False, indent=2))
        return answers
    for qid, item in answers.items():
        if qid.startswith("_"):
            continue
        shown = "  ".join(f"{key} {value:.2f}" for key, value in item["probabilities"].items())
        print(f"{qid}: {item['answer']}    [{shown}]")
    if answers["_state_truncated"]:
        print("note: state exceeded the token window; its head was dropped", file=sys.stderr)
    return answers


if __name__ == "__main__":
    main()
