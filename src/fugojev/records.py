"""Labelled requests on disk, and the internal record `encode()` consumes.

Choice labels are option names, noul labels are booleans, score labels are
0-based level indexes. A question that has no label is omitted before it gets here.
Option order for `choice` is shuffled only in training.
"""

import json
import random
from pathlib import Path


def option_text(name, description) -> str:
    text = "" if description is None else str(description).strip()
    return name if not text else f"{name}: {text}"


def permute_choice_options(req, rng: random.Random):
    """Shuffle every choice question. Score levels stay in order."""
    questions = {}
    for qid, question in req["questions"].items():
        if question["type"] != "choice":
            questions[qid] = question
            continue
        keys = list(question["criteria"])
        rng.shuffle(keys)
        questions[qid] = {**question, "criteria": {key: question["criteria"][key] for key in keys}}
    return {**req, "questions": questions}


def materialize(req):
    """Labelled request -> {state, questions: [{instr, options, label, qtype, qid, keys}]}."""
    if not isinstance(req.get("state"), str) or not req["state"].strip():
        raise ValueError("record needs a non-empty state string")
    questions = []
    for qid, question in req["questions"].items():
        qtype = question["type"]
        criteria = question.get("criteria")
        if qtype == "noul":
            criteria = criteria or {}
            options = [option_text("no", criteria.get("false")), option_text("yes", criteria.get("true"))]
            keys = ["false", "true"]
            if not isinstance(question["label"], bool):
                raise TypeError(f"{qid}: noul label must be bool")
            label = int(question["label"])
        elif qtype == "choice":
            if not isinstance(criteria, dict) or not 2 <= len(criteria) <= 255:
                raise ValueError(f"{qid}: choice needs 2..255 options")
            options = [option_text(key, value) for key, value in criteria.items()]
            keys = list(criteria)
            if question["label"] not in criteria:
                raise ValueError(f"{qid}: label {question['label']!r} is not an option")
            label = keys.index(question["label"])
        elif qtype == "score":
            if not isinstance(criteria, list) or len(criteria) < 2:
                raise ValueError(f"{qid}: score needs at least two levels")
            options = [str(item) for item in criteria]
            keys = options
            label = int(question["label"])
            if not 0 <= label < len(options):
                raise ValueError(f"{qid}: score label {label} outside 0..{len(options) - 1}")
        else:
            raise ValueError(f"{qid}: unknown type {qtype!r}")
        questions.append({
            "instr": str(question["instructions"]),
            "options": options,
            "label": label,
            "qtype": qtype,
            "qid": qid,
            "keys": keys,
        })
    if not questions:
        raise ValueError("record has no questions")
    return {"state": req["state"], "questions": questions}


def _dump_line(row) -> str:
    line = json.dumps(row, ensure_ascii=False)
    return line.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def dump_jsonl(path, rows) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            if not row.get("state", "").strip() or not row.get("questions"):
                continue
            if "hospital_id" not in row:
                raise ValueError("record is missing hospital_id")
            handle.write(_dump_line(row) + "\n")
            written += 1
    return written


def load_records(path):
    """One JSON object per line. Splits on '\\n' only, so a note cannot break the file."""
    records = []
    text = Path(path).read_text(encoding="utf-8")
    for number, line in enumerate(text.split("\n"), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if "hospital_id" not in row:
            raise ValueError(f"{path}:{number}: missing hospital_id")
        if not isinstance(row.get("state"), str) or not row["state"].strip():
            raise ValueError(f"{path}:{number}: missing state")
        if not isinstance(row.get("questions"), dict) or not row["questions"]:
            raise ValueError(f"{path}:{number}: missing questions")
        for qid, question in row["questions"].items():
            if "label" not in question:
                raise ValueError(f"{path}:{number}: question {qid} has no label")
        records.append(row)
    if not records:
        raise ValueError(f"{path}: no records")
    return records
