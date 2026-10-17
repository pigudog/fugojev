"""FIGO 2023 staging of endometrial cancer: label set, composition rule, and the
derivation of the gold stage from the case sheet plus the pathology template.

The 2023 system is a decision tree over components (histotype aggressiveness,
myometrial depth, LVSI, cervical stroma, extrauterine extent, nodes, molecular
class). The model answers the components; `compose` turns them into a stage. The
same function builds the gold stage from the sheet columns, so the model is scored
against a label that was produced by one explicit, inspectable rule.

Local conventions, fixed with the clinicians before any model was trained:

- "aggressive" = non-endometrioid (serous, clear cell, undifferentiated or
  dedifferentiated, carcinosarcoma, mixed) or endometrioid grade 3.
- This centre's pathology reports LVSI as present/absent and never counts vessels,
  so any LVSI in a non-aggressive uterus-confined tumour is staged IIB (the
  conservative reading). The alternative, never staging IIB, is one flag away.
- Node micro/macro-metastasis (IIIC1i/ii, IIIC2i/ii) is not reported; IIIC1 and
  IIIC2 are used without the suffix.
- Molecular: POLEmut in stage I-II -> IAmPOLEmut; p53abn with any myometrial
  invasion in stage I-II -> IICmp53abn. MMRd and NSMP do not change the stage.
"""

import re

HISTOTYPES = ("endometrioid", "serous", "clear_cell", "carcinosarcoma", "undifferentiated", "mixed")
HISTOTYPE_CRITERIA = {
    "endometrioid": "子宫内膜样癌（内膜样腺癌）",
    "serous": "浆液性癌",
    "clear_cell": "透明细胞癌",
    "carcinosarcoma": "癌肉瘤（恶性中胚叶混合瘤）",
    "undifferentiated": "未分化癌或去分化癌",
    "mixed": "混合性癌（含浆液性、透明细胞等成分）",
}

# Highest extrauterine site, ordered by stage.
EXTENTS = (
    "none", "adnexa", "serosa", "vagina_parametrium", "pelvic_peritoneum",
    "bladder_bowel_mucosa", "abdominal_peritoneum", "distant",
)
EXTENT_CRITERIA = {
    "none": "没有超出子宫体和宫颈",
    "adnexa": "卵巢或输卵管受累",
    "serosa": "子宫浆膜或浆膜下受累（附件未受累）",
    "vagina_parametrium": "阴道或宫旁受累",
    "pelvic_peritoneum": "盆腔腹膜转移（膀胱反折、直肠前壁、盆底腹膜等）",
    "bladder_bowel_mucosa": "侵犯膀胱或肠道粘膜",
    "abdominal_peritoneum": "盆腔以外的腹腔腹膜转移（大网膜、结肠旁沟、肠系膜、膈肌、肝表面等）",
    "distant": "远处转移（肺、肝实质、骨、脑，或腹股沟、肾血管以上的淋巴结）",
}
_EXTENT_STAGE = {
    "adnexa": "IIIA1", "serosa": "IIIA2", "vagina_parametrium": "IIIB1",
    "pelvic_peritoneum": "IIIB2", "bladder_bowel_mucosa": "IVA",
    "abdominal_peritoneum": "IVB", "distant": "IVC",
}

STAGES_2023 = (
    "IA1", "IA2", "IA3", "IB", "IC", "IIA", "IIB", "IIC",
    "IIIA1", "IIIA2", "IIIB1", "IIIB2", "IIIC1", "IIIC2",
    "IVA", "IVB", "IVC", "IAmPOLEmut", "IICmp53abn",
)
STAGE_2023_CRITERIA = {
    "IA1": "非侵袭性类型，局限于内膜或息肉，无肌层浸润",
    "IA2": "非侵袭性类型，肌层浸润小于1/2，无或仅局灶脉管侵犯",
    "IA3": "低级别内膜样癌局限于子宫，同时卵巢有低级别内膜样癌",
    "IB": "非侵袭性类型，肌层浸润达到或超过1/2，无或仅局灶脉管侵犯",
    "IC": "侵袭性类型，局限于内膜或息肉，无肌层浸润",
    "IIA": "非侵袭性类型，侵犯宫颈间质",
    "IIB": "非侵袭性类型，广泛脉管侵犯",
    "IIC": "侵袭性类型，有肌层浸润",
    "IIIA1": "卵巢或输卵管受累",
    "IIIA2": "子宫浆膜或浆膜下受累",
    "IIIB1": "阴道或宫旁受累",
    "IIIB2": "盆腔腹膜转移",
    "IIIC1": "盆腔淋巴结转移",
    "IIIC2": "腹主动脉旁淋巴结转移",
    "IVA": "侵犯膀胱或肠道粘膜",
    "IVB": "盆腔以外的腹腔腹膜转移",
    "IVC": "远处转移，含腹股沟或肾血管以上淋巴结",
    "IAmPOLEmut": "I-II期且POLE致病性突变",
    "IICmp53abn": "I-II期且p53异常，伴肌层浸润",
}

STAGE_2009_TO_GROUP = {"IA": 1, "IB": 1, "II": 2, "IIIA": 3, "IIIB": 3, "IIIC1": 3, "IIIC2": 3, "IVA": 4, "IVB": 4}

AGGRESSIVE_HISTOTYPES = frozenset(HISTOTYPES) - {"endometrioid"}


def normalize_histotype(value) -> str | None:
    text = "" if value is None else str(value).strip()
    if not text or text.lower() in {"nan", "none", "na"}:
        return None
    if "癌肉瘤" in text or "中胚叶" in text or "MMMT" in text.upper():
        return "carcinosarcoma"
    if "混合" in text:
        return "mixed"
    if "浆液" in text:
        return "serous"
    if "透明" in text:
        return "clear_cell"
    if "未分化" in text or "去分化" in text:
        return "undifferentiated"
    if "内膜样" in text or text.upper() in {"EA", "EEC"}:
        return "endometrioid"
    return None


def is_aggressive(histotype, grade) -> bool | None:
    """grade is the 0/1/2 index of GRADE_CRITERIA. None when it cannot be decided."""
    if histotype is None:
        return None
    if histotype in AGGRESSIVE_HISTOTYPES:
        return True
    if grade is None:
        return None
    return grade == 2


def compose(*, aggressive, muscle, lvsi, cervical_stroma, extent, nodes,
            pole=False, p53abn=False, lvsi_is_substantial=True) -> str | None:
    """FIGO 2023 stage from components.

    muscle: 0 none / 1 < 1/2 / 2 >= 1/2 (index of MUSCLE_CRITERIA). nodes: key of
    NODE_CRITERIA. extent: key of EXTENTS. Returns None when a component that the
    decision needs is missing.
    """
    if nodes in ("pelvic_and_paraaortic", "paraaortic"):
        node_stage = "IIIC2"
    elif nodes == "pelvic":
        node_stage = "IIIC1"
    elif nodes == "none":
        node_stage = None
    else:
        node_stage = None
        if extent in (None, "none") and nodes is None:
            return None
    extent_stage = _EXTENT_STAGE.get(extent)
    candidates = [stage for stage in (node_stage, extent_stage) if stage]
    if candidates:
        return max(candidates, key=STAGES_2023.index)
    if extent is None or muscle is None or aggressive is None:
        return None
    invaded = muscle > 0 or bool(cervical_stroma)
    if pole:
        return "IAmPOLEmut"
    if p53abn and invaded:
        return "IICmp53abn"
    if aggressive:
        return "IIC" if invaded else "IC"
    if lvsi is None and lvsi_is_substantial:
        return None
    if lvsi and lvsi_is_substantial:
        return "IIB"
    if cervical_stroma:
        return "IIA"
    if cervical_stroma is None and muscle == 0:
        return None
    return ("IA1", "IA2", "IB")[muscle]


# ---------------------------------------------------------------- gold from the sheet

_TEMPLATE_FIELD = r"(?<![\u4e00-\u9fff]){name}\s*[:：]\s*([^\n]{{0,24}})"
_PLUS = re.compile(r"（\s*\+\s*）|\(\s*\+\s*\)")
_MINUS = re.compile(r"未见|阴性|（\s*[-－]\s*）|\(\s*[-－]\s*\)")
_WORDS = re.compile(r"见癌|见肿瘤|癌累及|肿瘤累及|转移")


def cell_value(cell: str) -> bool | None:
    head = re.split(r"\s{2,}|[，,；;]", cell.strip())[0]
    if not head or head.startswith("/"):
        return None
    if _PLUS.search(head):
        return True
    if _MINUS.search(head):
        return False
    if _WORDS.search(head):
        return True
    return None


def template_value(text: str, name: str) -> bool | None:
    """Read `name:（+）` style cells from the pathology template. None when absent or '/'."""
    found = None
    for match in re.finditer(_TEMPLATE_FIELD.format(name=re.escape(name)), text):
        value = cell_value(match.group(1))
        if value:
            return True
        if value is False:
            found = False
    return found


def _any_true(text, names):
    seen = False
    for name in names:
        value = template_value(text, name)
        if value:
            return True
        seen = seen or value is not None
    return False if seen else None


def cervical_from_text(text: str) -> bool | None:
    stroma = template_value(text, "宫颈管间质")
    if stroma is not None:
        return stroma
    whole = template_value(text, "宫颈")
    return whole


def extent_from_text(pathology: str, surgery: str = "") -> str | None:
    """Highest extrauterine site that the pathology template states. Peritoneal and
    distant sites are not in the template; they come from the stage override."""
    vag = _any_true(pathology, ("阴道", "左宫旁", "右宫旁", "宫旁"))
    if vag:
        return "vagina_parametrium"
    adnexa = _any_true(pathology, ("左卵巢", "右卵巢", "左输卵管", "右输卵管", "双附件"))
    if adnexa:
        return "adnexa"
    if re.search(r"浆膜[面下]?[^。\n]{0,8}(见癌|见肿瘤|累及|受累)", pathology):
        return "serosa"
    if vag is None and adnexa is None:
        return None
    return "none"


def normalize_p53(value) -> bool | None:
    """Sheet IHC string -> abnormal (True) / wild-type (False). Ambiguous patterns ('+',
    '个别+') are None: the study resolved them by sequencing, the text cannot."""
    text = "" if value is None else str(value).strip()
    if not text or text.lower() == "nan":
        return None
    body = re.sub(r"^[Pp]53", "", text).replace("（", "(").replace("）", ")").replace(" ", "")
    if re.search(r"\(-\)|弥漫|\+\+\+|>90|＞90|完全缺失|突变型", body):
        return True
    if re.search(r"少\+|部分|散在|弱\+|灶\+|个别细胞|中等|野生", body):
        return False
    return None


def normalize_mmr(value) -> bool | None:
    """Sheet MMR string -> deficient (True) when any protein is lost."""
    text = "" if value is None else str(value).strip()
    if not text or text.lower() == "nan":
        return None
    body = text.replace("（", "(").replace("）", ")").replace(" ", "")
    if re.search(r"\(-\)|缺失|\(少量-\)", body):
        return True
    if re.search(r"\(\+\)|部分\+|少\+", body):
        return False
    return None


def normalize_molecular(value) -> str | None:
    text = "" if value is None else str(value).strip().lower().replace(" ", "")
    if not text or text == "nan":
        return None
    if "pole" in text:
        return "POLEmut"
    if "mmrd" in text or "msi" in text:
        return "MMRd"
    if "p53" in text:
        return "p53abn"
    if "nsmp" in text or "低拷贝" in text:
        return "NSMP"
    return None


def gold_components(*, figo2009, muscle, lvsi, nodes, histotype, grade, molecular,
                    pathology_text="", surgery_text="", override=None) -> dict:
    """Components for the 2023 stage. The sheet decides what it states; the pathology
    template fills cervical stroma and the extrauterine site for stage III-IV, and an
    explicit per-case override (reviewed by eye) settles peritoneal vs distant for
    2009 stage IVB, which the 2009 system did not separate."""
    override = override or {}
    group = STAGE_2009_TO_GROUP.get(figo2009)
    if group is None:
        cervical = None
    elif group == 1:
        cervical = False
    elif group == 2:
        cervical = True
    else:
        cervical = cervical_from_text(pathology_text)
    if "cervical_stroma" in override:
        cervical = override["cervical_stroma"]

    if group in (1, 2):
        extent = "none"
    elif figo2009 == "IIIA":
        extent = extent_from_text(pathology_text, surgery_text)
        extent = extent if extent in ("adnexa", "serosa") else "adnexa"
    elif figo2009 == "IIIB":
        extent = extent_from_text(pathology_text, surgery_text)
        extent = "vagina_parametrium" if extent == "vagina_parametrium" else "pelvic_peritoneum"
    elif figo2009 in ("IIIC1", "IIIC2"):
        extent = extent_from_text(pathology_text, surgery_text) or "none"
    elif figo2009 == "IVA":
        extent = "bladder_bowel_mucosa"
    elif figo2009 == "IVB":
        extent = None  # must be settled by override
    else:
        extent = None
    if "extent" in override:
        extent = override["extent"]
    # The sheet staged these below IIIC, so an unassessed node column means none.
    if nodes is None and figo2009 in ("IA", "IB", "II", "IIIA", "IIIB", "IVA", "IVB"):
        nodes = "none"

    return {
        "aggressive": is_aggressive(histotype, grade),
        "muscle": muscle,
        "lvsi": lvsi,
        "cervical_stroma": cervical,
        "extent": extent,
        "nodes": nodes,
        "pole": molecular == "POLEmut",
        "p53abn": molecular == "p53abn",
    }


def gold_stage(components: dict) -> str | None:
    return compose(**components)


# ------------------------------------------------------------ stage from predictions

MOLECULAR_CLASSES = ("POLEmut", "MMRd", "p53abn", "NSMP")


def components_from_predictions(predicted: dict, molecular: str | None = None,
                                lvsi_is_substantial: bool = True) -> dict:
    """Map predicted option keys (as `evaluate` names them) onto `compose` arguments.

    `molecular`, when given, is a structured molecular class that overrides what the
    notes say; otherwise POLE and p53 come from their heads with the usual surrogate
    precedence (POLE, then MMR, then p53).
    """
    from .questions import GRADE_CRITERIA, MUSCLE_CRITERIA

    def flag(qid):
        value = predicted.get(qid)
        return None if value is None else value == "true"

    grade = predicted.get("grade")
    grade = GRADE_CRITERIA.index(grade) if grade in GRADE_CRITERIA else None
    muscle = predicted.get("muscle_invasion")
    muscle = MUSCLE_CRITERIA.index(muscle) if muscle in MUSCLE_CRITERIA else None
    if molecular is not None:
        pole = molecular == "POLEmut"
        p53abn = molecular == "p53abn"
    else:
        pole = bool(flag("pole"))
        mmrd = bool(flag("mmr"))
        p53abn = bool(flag("p53")) and not pole and not mmrd
    return {
        "aggressive": is_aggressive(predicted.get("histotype"), grade),
        "muscle": muscle,
        "lvsi": flag("lvsi"),
        "cervical_stroma": flag("cervical_stroma"),
        "extent": predicted.get("extent"),
        "nodes": predicted.get("lymph_node"),
        "pole": pole,
        "p53abn": p53abn,
        "lvsi_is_substantial": lvsi_is_substantial,
    }


def stage_from_predictions(predicted: dict, molecular: str | None = None) -> str | None:
    return compose(**components_from_predictions(predicted, molecular))


def molecular_from_stage(stage: str | None) -> str | None:
    """The structured molecular input that would reproduce a gold 2023 stage's suffix.
    Used to simulate a clinic that supplies the molecular class alongside the notes."""
    if stage == "IAmPOLEmut":
        return "POLEmut"
    if stage == "IICmp53abn":
        return "p53abn"
    return None
