"""Answer the five endometrial questions for one note bundle.

    python -m fugojev.predict --run weights/ec-v1 --state note.txt
    python -m fugojev.predict --run weights/ec-v1 --request request.json

`--state` is a text file (or `-` for stdin) laid out like the training states:
`[门诊]`, `[入院]`, `[影像]`, `[病理]`, `[手术]` blocks, each keeping its conclusion
at the end. `--request` is the JSON shape in `fugojev.api` (five sections, optional
question subset) and is rendered exactly like the training data. Output is one
line per question with the chosen option, a confidence, and the full probability
vector over the allowed options. Nothing is generated.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

from fugojev.api import Request, to_answers, to_record
from fugojev.calibrate import apply_calibration
from fugojev.checkpoint import load_checkpoint
from fugojev.records import materialize


def request_for(state: str, question_ids=None) -> dict:
    """Kept for callers that build the labelled-request shell from a state string."""
    return to_record(Request(state=state, questions=list(question_ids) if question_ids else None))


@torch.no_grad()
def predict(model, tokenizer, state_or_request, question_ids=None, calibration=None) -> dict:
    request = state_or_request if isinstance(state_or_request, Request) else Request(
        state=str(state_or_request), questions=list(question_ids) if question_ids else None)
    encoded = model.encode(tokenizer, materialize(to_record(request)))
    probabilities, _ = model.probs_and_prefix(encoded)
    vectors = []
    for prob, qid, keys in zip(probabilities, encoded["qids"], encoded["keys"]):
        values = [float(p) for p in prob]
        if calibration:
            values = apply_calibration(values, keys, calibration.get(qid))
        vectors.append(values)
    answers = to_answers(vectors, encoded["qids"], encoded["keys"])
    answers["_state_truncated"] = bool(encoded["state_truncated"])
    return answers


def main(argv=None):
    parser = argparse.ArgumentParser(description="Answer the endometrial questions for one record")
    parser.add_argument("--run", required=True, help="checkpoint directory, e.g. weights/ec-v1")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--state", help="path to a state text file, or - for stdin")
    source.add_argument("--request", help="path to a request JSON (see fugojev.api), or - for stdin")
    parser.add_argument("--questions", default="", help="comma-separated subset of question ids")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--base", default=None, help="override the backbone path stored in the checkpoint")
    parser.add_argument("--calibration", default="", help="calibration.json from fugojev.calibrate")
    parser.add_argument("--json", action="store_true", help="print one JSON object instead of lines")
    args = parser.parse_args(argv)
    wanted = [qid for qid in args.questions.split(",") if qid] or None
    if args.state is not None:
        text = sys.stdin.read() if args.state == "-" else Path(args.state).read_text(encoding="utf-8")
        if not text.strip():
            raise SystemExit("empty state")
        request = Request(state=text, questions=wanted)
    else:
        text = sys.stdin.read() if args.request == "-" else Path(args.request).read_text(encoding="utf-8")
        payload = json.loads(text)
        if wanted:
            payload["questions"] = wanted
        request = Request.from_dict(payload)
    request.validate()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.dtype == "bf16" and str(device).startswith("cuda") else torch.float32
    tokenizer, model = load_checkpoint(args.run, device, dtype=dtype, base=args.base)
    calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8")) if args.calibration else None
    answers = predict(model, tokenizer, request, calibration=calibration)
    if args.json:
        print(json.dumps(answers, ensure_ascii=False, indent=2))
        return answers
    for qid, item in answers.items():
        if qid.startswith("_"):
            continue
        shown = "  ".join(f"{key} {value:.2f}" for key, value in item["probabilities"].items())
        print(f"{qid}: {item['answer']}  (confidence {item['confidence']:.2f})    [{shown}]")
    if answers["_state_truncated"]:
        print("note: state exceeded the token window; its head was dropped", file=sys.stderr)
    return answers


if __name__ == "__main__":
    main()
