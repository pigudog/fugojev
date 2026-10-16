"""Endometrial-cancer questions. Labels come from the 333-case sheet, not from the note text."""

SRC = "ec-gold"

FIGO_LEVELS = ("IA", "IB", "II", "IIIA", "IIIB", "IIIC1", "IIIC2", "IVA", "IVB")

# FIGO 2009. The sheet's IIIC1/IIIC2 split matches this edition, not a free-text stage name.
FIGO_CRITERIA = {
    "IA": "肿瘤局限于子宫体，无肌层浸润或浸润深度小于1/2",
    "IB": "肿瘤局限于子宫体，肌层浸润深度达到或超过1/2",
    "II": "侵犯宫颈间质，但还没有超出子宫",
    "IIIA": "侵犯子宫浆膜或附件",
    "IIIB": "阴道或宫旁受累",
    "IIIC1": "盆腔淋巴结转移",
    "IIIC2": "腹主动脉旁淋巴结转移，伴或不伴盆腔淋巴结转移",
    "IVA": "侵犯膀胱或直肠粘膜",
    "IVB": "远处转移，包括腹股沟淋巴结、腹腔内转移或远处器官",
}

GRADE_CRITERIA = ["I级", "II级", "III级"]
MUSCLE_CRITERIA = ["无肌层浸润", "浸润深度小于1/2肌层", "浸润深度达到或超过1/2肌层"]
NODE_CRITERIA = {
    "none": "无淋巴结转移",
    "pelvic": "盆腔淋巴结转移",
    "pelvic_and_paraaortic": "盆腔和腹主动脉旁淋巴结都有转移",
    "paraaortic": "只有腹主动脉旁淋巴结转移",
}

_NODE_CODES = {
    "0": "none",
    "1": "pelvic",
    "2": "pelvic_and_paraaortic",
    "3": "paraaortic",
}

QUESTIONS = {
    "figo": {
        "type": "choice",
        "instructions": "根据病理、影像、门诊和手术记录，这位子宫内膜癌患者的FIGO 2009分期是哪一档？只依据文书里写明的侵犯范围。",
        "criteria": FIGO_CRITERIA,
    },
    "muscle_invasion": {
        "type": "score",
        "instructions": "手术标本里，肿瘤浸润肌层有多深？",
        "criteria": MUSCLE_CRITERIA,
    },
    "lvsi": {
        "type": "noul",
        "instructions": "病理是否写明脉管侵犯？",
        "criteria": {"true": "有脉管侵犯", "false": "未见脉管侵犯"},
    },
    "lymph_node": {
        "type": "choice",
        "instructions": "淋巴结转移累及哪个范围？",
        "criteria": NODE_CRITERIA,
    },
    "grade": {
        "type": "score",
        "instructions": "手术标本的组织学分级是哪一档？",
        "criteria": GRADE_CRITERIA,
    },
}


def _blank(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _missing(value) -> bool:
    text = _blank(value)
    return text.lower() in {"", "na", "n/a", "null", "none", "-", "nan"}


def normalize_figo(value) -> str | None:
    """Map sheet spellings onto FIGO_LEVELS. Bare 'IV' is dropped: it does not say IVA or IVB."""
    if _missing(value):
        return None
    text = _blank(value).replace(" ", "").replace("期", "").upper()
    if text == "IV" or text not in FIGO_LEVELS:
        return None
    return text


def normalize_grade(value) -> int | None:
    if _missing(value):
        return None
    text = _blank(value).upper().replace("级", "").replace(" ", "")
    return {"1": 0, "I": 0, "Ⅰ": 0, "2": 1, "II": 1, "Ⅱ": 1, "3": 2, "III": 2, "Ⅲ": 2}.get(text)


def normalize_muscle(value) -> int | None:
    if _missing(value):
        return None
    text = _blank(value)
    if "无" in text and "肌层" in text:
        return 0
    if any(mark in text for mark in ("＜", "<", "小于", "浅")):
        return 1
    if any(mark in text for mark in ("≥", ">=", "＞", ">", "大于", "深")):
        return 2
    return None


def normalize_lvsi(value) -> bool | None:
    if _missing(value):
        return None
    text = _blank(value)
    for mark in ("（", "）", "(", ")", " "):
        text = text.replace(mark, "")
    if text in {"-", "－", "—", "阴性", "无", "neg", "negative"}:
        return False
    if text in {"+", "＋", "阳性", "有", "pos", "positive"}:
        return True
    return None


def normalize_nodes(value) -> str | None:
    """The coded 0/1/2/3 column, not the free-text positive-node counts such as '0/24'."""
    if _missing(value):
        return None
    return _NODE_CODES.get(_blank(value))


def _question(qid: str, label):
    spec = QUESTIONS[qid]
    criteria = spec["criteria"]
    copied = dict(criteria) if isinstance(criteria, dict) else list(criteria)
    return {
        "type": spec["type"],
        "instructions": spec["instructions"],
        "criteria": copied,
        "label": label,
        "src": SRC,
    }


def consistent_predictions(predicted: dict) -> dict:
    """FIGO 2009: invasion of half the myometrium or more is not IA.

    The FIGO head still prefers the majority stage. The muscle head, once it can
    see the depth sentence, is the one that knows this distinction.
    """
    adjusted = dict(predicted)
    if adjusted.get("figo") == "IA" and adjusted.get("muscle_invasion") == MUSCLE_CRITERIA[2]:
        adjusted["figo"] = "IB"
    return adjusted


def questions_from_raw(*, figo, grade, muscle, lvsi, nodes) -> dict:
    """Drop any question whose sheet cell cannot be normalized. Do not invent a label."""
    found = {
        "figo": normalize_figo(figo),
        "muscle_invasion": normalize_muscle(muscle),
        "lvsi": normalize_lvsi(lvsi),
        "lymph_node": normalize_nodes(nodes),
        "grade": normalize_grade(grade),
    }
    return {qid: _question(qid, label) for qid, label in found.items() if label is not None}
