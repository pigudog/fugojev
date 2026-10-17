"""Request and answer shapes for serving, and their conversion to the model's record.

Adapted from MedJev's `api.py` (itself from kev, Apache-2.0; see NOTICE), without the
pydantic dependency. A request carries either a ready `state` string or the five
note sections, which are rendered with the same `render_state` the training data
went through, so the model is served byte-identical text to what it was trained on.

    {
      "sections": {"门诊": ["..."], "入院": "...", "影像": ["..."], "病理": ["..."], "手术": ["..."],
                   "免疫组化": ["..."], "分子病理": ["..."]},
      "questions": ["figo", "muscle_invasion"],         # optional subset
      "molecular": "POLEmut"                            # optional: POLEmut / MMRd / p53abn / NSMP
    }

Answers carry the chosen option, the probability over options, and a confidence
in [0, 1] that is a function of the distribution, not a measured accuracy rate.
When the request asks the FIGO 2023 components, the answers also carry
`figo2023_composed`: the 2023 stage composed from the component answers, using the
structured `molecular` class if supplied, otherwise the p53 / MMR / POLE heads.
"""

from dataclasses import dataclass, field

from fugojev.figo2023 import MOLECULAR_CLASSES, stage_from_predictions
from fugojev.questions import COMPONENT_QIDS, QUESTIONS, consistent_predictions
from fugojev.state import render_state

SECTION_ORDER = ("门诊", "入院", "影像", "病理", "免疫组化", "分子病理", "手术")
_PLACEHOLDER = {"noul": False, "choice": None, "score": 0}


@dataclass
class Request:
    state: str = ""
    sections: dict = field(default_factory=dict)
    questions: list | None = None
    molecular: str | None = None

    @classmethod
    def from_dict(cls, payload: dict) -> "Request":
        if not isinstance(payload, dict):
            raise TypeError("request must be a JSON object")
        request = cls(
            state=str(payload.get("state") or ""),
            sections=dict(payload.get("sections") or {}),
            questions=list(payload["questions"]) if payload.get("questions") else None,
            molecular=payload.get("molecular") or None,
        )
        request.validate()
        return request

    def validate(self):
        if not self.state.strip() and not any(_texts(value) for value in self.sections.values()):
            raise ValueError("request needs a non-empty state or at least one section")
        unknown = set(self.sections) - set(SECTION_ORDER)
        if unknown:
            raise ValueError(f"unknown sections {sorted(unknown)}; expected {list(SECTION_ORDER)}")
        if self.questions is not None:
            missing = [qid for qid in self.questions if qid not in QUESTIONS]
            if missing:
                raise ValueError(f"unknown questions {missing}; expected {list(QUESTIONS)}")
            if not self.questions:
                raise ValueError("questions must not be an empty list")
        if self.molecular is not None and self.molecular not in MOLECULAR_CLASSES:
            raise ValueError(f"unknown molecular class {self.molecular!r}; expected one of {list(MOLECULAR_CLASSES)}")

    def rendered_state(self) -> str:
        """Sections in clinical reading order, tail-kept per section, pathology first in budget."""
        if self.state.strip():
            return self.state
        blocks = []
        for title in SECTION_ORDER:
            for text in _texts(self.sections.get(title)):
                blocks.append((title, text))
        return render_state(blocks)


def _texts(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(item) for item in value if str(item).strip()]


def to_record(request: Request) -> dict:
    """Labelled-request shell for unlabelled text. Labels are placeholders and are
    never read on the serving path."""
    questions = {}
    for qid, spec in QUESTIONS.items():
        if request.questions and qid not in request.questions:
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
    return {"state": request.rendered_state(), "questions": questions}


def choice_confidence(probs) -> float:
    """(p_max - 1/K) / (1 - 1/K): 0 at uniform, 1 at certainty."""
    k = len(probs)
    return 1.0 if k == 1 else (max(probs) - 1 / k) / (1 - 1 / k)


def score_confidence(probs) -> float:
    """1 - E|level - mode| / (L - 1): how concentrated the mass is around the modal level."""
    levels = len(probs)
    mode = max(range(levels), key=probs.__getitem__)
    return 1.0 - sum(p * abs(index - mode) for index, p in enumerate(probs)) / (levels - 1)


def to_answers(probabilities, qids, keys_per_question, apply_rule=True, molecular=None) -> dict:
    """Probability vectors -> answer objects keyed by question id. Adds the composed
    FIGO 2023 stage when every component question was asked."""
    answers = {}
    for probs, qid, keys in zip(probabilities, qids, keys_per_question):
        values = [float(p) for p in probs]
        top = max(range(len(values)), key=values.__getitem__)
        qtype = QUESTIONS[qid]["type"]
        confidence = score_confidence(values) if qtype == "score" else choice_confidence(values)
        answers[qid] = {
            "type": qtype,
            "answer": keys[top],
            "confidence": confidence,
            "probabilities": dict(zip(keys, values)),
        }
        if qtype == "score":
            answers[qid]["expected_level"] = sum(index * p for index, p in enumerate(values))
    if apply_rule:
        adjusted = consistent_predictions({qid: item["answer"] for qid, item in answers.items()})
        for qid, item in answers.items():
            if adjusted[qid] != item["answer"]:
                item["rule_adjusted_from"] = item["answer"]
                item["answer"] = adjusted[qid]
    if all(qid in answers for qid in COMPONENT_QIDS):
        chosen = {qid: item["answer"] for qid, item in answers.items()}
        answers["figo2023_composed"] = {
            "type": "derived",
            "answer": stage_from_predictions(chosen, molecular),
            "molecular_source": "request" if molecular else "notes",
        }
    return answers
