"""Endometrial-cancer questions. Labels come from the 333-case sheet, not from the note text.

v1 asked FIGO 2009 and four components. v1.3 adds the components that FIGO 2023
needs (histotype, cervical stroma, extrauterine extent, p53 and MMR
immunohistochemistry, POLE) and the 2023 stage itself; `figo2023.compose` turns the
component answers into a stage, and the direct 2023 head is kept for comparison.
"""

from .figo2023 import (
    EXTENT_CRITERIA,
    HISTOTYPE_CRITERIA,
    STAGE_2023_CRITERIA,
    normalize_histotype,
    normalize_mmr,
    normalize_p53,
)

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
    # v1.3: FIGO 2023 components and the 2023 stage.
    "histotype": {
        "type": "choice",
        "instructions": "手术标本的病理类型是哪一种？",
        "criteria": HISTOTYPE_CRITERIA,
    },
    "cervical_stroma": {
        "type": "noul",
        "instructions": "病理是否写明肿瘤侵犯宫颈间质？",
        "criteria": {"true": "宫颈间质受累", "false": "宫颈间质未受累"},
    },
    "extent": {
        "type": "choice",
        "instructions": "除淋巴结外，肿瘤超出子宫的最远部位是哪里？只依据病理和手术记录写明的受累部位。",
        "criteria": EXTENT_CRITERIA,
    },
    "p53": {
        "type": "noul",
        "instructions": "免疫组化中 p53 的表达是异常型（弥漫强阳性或完全缺失）还是野生型？",
        "criteria": {"true": "p53 异常表达", "false": "p53 野生型表达"},
    },
    "mmr": {
        "type": "noul",
        "instructions": "免疫组化中错配修复蛋白（MLH1、PMS2、MSH2、MSH6）是否有缺失？",
        "criteria": {"true": "有错配修复蛋白缺失（dMMR）", "false": "四个蛋白都表达（pMMR）"},
    },
    "pole": {
        "type": "noul",
        "instructions": "分子病理是否检出 POLE 致病性突变（核酸外切酶结构域热点，如 P286R、V411L、S459F）？",
        "criteria": {"true": "POLE 致病性突变", "false": "未检出 POLE 致病性突变"},
    },
    "figo2023": {
        "type": "choice",
        "instructions": "这位子宫内膜癌患者的 FIGO 2023 分期是哪一档？分子分型只在文书写明时使用。",
        "criteria": STAGE_2023_CRITERIA,
    },
}

COMPONENT_QIDS = ("histotype", "grade", "muscle_invasion", "lvsi", "cervical_stroma", "extent", "lymph_node", "p53", "mmr", "pole")


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


def placeholder_label(qid):
    """A syntactically valid label for asking a question whose gold cell is missing."""
    spec = QUESTIONS[qid]
    if spec["type"] == "noul":
        return False
    if spec["type"] == "score":
        return 0
    return next(iter(spec["criteria"]))


def with_all_components(record: dict) -> tuple[dict, set]:
    """Ask every FIGO 2023 component even when its gold is missing, so the composed
    stage can be formed; returns the widened record and the qids whose label is a
    placeholder and must not be scored."""
    questions = dict(record["questions"])
    placeholders = set()
    for qid in COMPONENT_QIDS:
        if qid not in questions:
            questions[qid] = _question(qid, placeholder_label(qid))
            placeholders.add(qid)
    return {**record, "questions": questions}, placeholders


def consistent_predictions(predicted: dict) -> dict:
    """FIGO 2009: invasion of half the myometrium or more is not IA.

    The FIGO head still prefers the majority stage. The muscle head, once it can
    see the depth sentence, is the one that knows this distinction.
    """
    adjusted = dict(predicted)
    if adjusted.get("figo") == "IA" and adjusted.get("muscle_invasion") == MUSCLE_CRITERIA[2]:
        adjusted["figo"] = "IB"
    return adjusted


def questions_from_raw(*, figo, grade, muscle, lvsi, nodes, histotype=None, p53=None, mmr=None,
                       pole=None, cervical_stroma=None, extent=None, figo2023=None) -> dict:
    """Drop any question whose sheet cell cannot be normalized. Do not invent a label.

    `pole`, `cervical_stroma`, `extent` and `figo2023` arrive already normalized
    (`corpus` derives them with `figo2023.gold_components`); None drops the question.
    """
    found = {
        "figo": normalize_figo(figo),
        "muscle_invasion": normalize_muscle(muscle),
        "lvsi": normalize_lvsi(lvsi),
        "lymph_node": normalize_nodes(nodes),
        "grade": normalize_grade(grade),
        "histotype": normalize_histotype(histotype),
        "p53": normalize_p53(p53),
        "mmr": normalize_mmr(mmr),
        "pole": pole,
        "cervical_stroma": cervical_stroma,
        "extent": extent,
        "figo2023": figo2023,
    }
    return {qid: _question(qid, label) for qid, label in found.items() if label is not None}
