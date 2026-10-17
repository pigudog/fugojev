"""Input ablations applied at evaluation time, to both the model and the rule baseline.

    paraphrase       rewrite every templated `字段:值` cell of the pathology report as a
                     prose sentence that says the same thing in different words
                     (`肌层浸润:<1/2肌层` -> `癌组织侵及浅肌层，深度不足肌壁的一半`;
                     `左盆腔（1/5）` -> `左盆腔淋巴结5枚中1枚见转移`). The evidence stays,
                     the surface form the systems were built on goes. Asks whether a
                     system reads or matches the template.
    strip-template   drop the template cells and keep only narrative text. At this
                     centre the conclusion *is* the template, so this removes the
                     evidence itself (see `coverage()`); it is the floor for "no
                     pathology conclusion", not a reading test.
    pathology-only   keep the pathology-department sections alone ([病理], [免疫组化],
                     [分子病理]). Asks what imaging, surgery and history add.
    none             the state as built.

`coverage()` reports how often the evidence words survive an ablation, so a drop in
score can be read as "the evidence was removed" versus "the system could not read it".
"""

import hashlib
import re

from fugojev.baseline import _BREAKDOWN, _FIELD, _FRACTION, classify_grade, classify_muscle, is_positive
from fugojev.evaluate import ABLATIONS  # noqa: F401  (single source of the names)

# Template cells of this centre's pathology report. Anything else (大体所见 prose, 报告
# conclusions, 其他/备注/补充报告 free text) is kept.
_TEMPLATE_FIELD = re.compile(
    r"^(?:肿瘤大小|肿瘤位置|肿瘤大体类型|病理类型|肌层浸润|宫颈管腺体|宫颈管间质|阴道切缘|"
    r"淋巴结转移情况|左宫旁|右宫旁|组织学分化程度|脉管侵犯|阴道|肿瘤周围子宫内膜|左卵巢|右卵巢|"
    r"左输卵管|右输卵管|宫体|宫颈|双附件|平滑肌瘤|.*淋巴结.*)$"
)

EVIDENCE_WORDS = {"muscle": "肌层", "lvsi": "脉管", "nodes": "淋巴结", "cervix": "宫颈", "grade": "级"}

# `pathology-only` keeps every pathology-department report: the surgical pathology
# template, the immunohistochemistry panel and the molecular report. Paraphrase and
# strip-template only touch the [病理] template; the IHC and molecular sections are
# not templated the same way and stay as written.
PATHOLOGY_FAMILY = ("[病理]", "[免疫组化]", "[分子病理]")


def strip_template_block(text: str) -> str:
    """Remove template cells from one pathology block. A cell runs from its field name
    to the next field on the same line or the end of the line, so prose on following
    lines survives."""
    found = list(_FIELD.finditer(text))
    cut = []
    for index, match in enumerate(found):
        if not _TEMPLATE_FIELD.match(match.group(1)):
            continue
        start = match.start()
        line_end = text.find("\n", match.end())
        line_end = len(text) if line_end < 0 else line_end
        next_field = found[index + 1].start() if index + 1 < len(found) else len(text)
        cut.append((start, min(line_end, next_field)))
    if not cut:
        return text
    pieces = []
    cursor = 0
    for start, end in cut:
        pieces.append(text[cursor:start])
        cursor = max(cursor, end)
    pieces.append(text[cursor:])
    stripped = "".join(pieces)
    lines = [" ".join(line.split()) for line in stripped.splitlines()]
    return "\n".join(line for line in lines if line and line not in {"[病理]", "项目：", "所见：", "报告："})


# ---- paraphrase -------------------------------------------------------------

_MUSCLE_PROSE = {
    0: ["肿瘤局限于子宫内膜，镜下未见肌层受侵", "癌组织未侵入肌层"],
    1: ["癌组织侵及浅肌层，深度不足肌壁的一半", "肿瘤浸润肌层，但未达肌壁厚度的二分之一"],
    2: ["癌组织浸润深肌层，超过肌壁厚度的一半", "肿瘤侵及肌壁外半，浸润深度达到或超过二分之一"],
}
_LVSI_PROSE = {
    True: ["脉管内可见癌栓", "镜下见到脉管受侵"],
    False: ["脉管内未见癌栓", "未发现脉管受侵"],
}
_GRADE_PROSE = {
    0: ["组织学分级为一级，高分化", "分化程度：高分化（G1）"],
    1: ["组织学分级为二级，中分化", "分化程度：中分化（G2）"],
    2: ["组织学分级为三级，低分化", "分化程度：低分化（G3）"],
}
_SITE_NAMES = {
    "宫颈管间质": "宫颈间质", "宫颈管腺体": "宫颈管腺体", "左宫旁": "左侧宫旁组织", "右宫旁": "右侧宫旁组织",
    "阴道": "阴道壁", "阴道切缘": "阴道切缘", "左卵巢": "左侧卵巢", "右卵巢": "右侧卵巢",
    "左输卵管": "左侧输卵管", "右输卵管": "右侧输卵管", "双附件": "双侧附件", "宫体": "宫体", "宫颈": "宫颈",
}
_SITE_PROSE = {
    True: ["{site}可见癌累及", "癌组织累及{site}"],
    False: ["{site}未见癌累及", "{site}未受累"],
}


def _pick(variants, seed: str):
    digest = hashlib.sha256(seed.encode()).digest()
    return variants[digest[0] % len(variants)]


def _nodes_prose(value: str, seed: str) -> str:
    parts = []
    total = _FRACTION.search(value[:40])
    if total:
        hit, examined = int(total.group(1)), int(total.group(2))
        if hit == 0:
            parts.append(_pick([f"共检出淋巴结{examined}枚，均未见癌转移", f"{examined}枚淋巴结内均未找到转移癌"], seed))
        else:
            parts.append(_pick([f"共检出淋巴结{examined}枚，其中{hit}枚见癌转移", f"{examined}枚淋巴结中{hit}枚有转移癌"], seed))
    stations = _stations_prose(value)
    if stations:
        parts.append(stations)
    rest = _BREAKDOWN.sub("", _FRACTION.sub("", value, count=1) if total else value)
    rest = re.sub(r"总数|[（）()，,。;；\s]+", " ", rest).strip()
    if rest and not parts:
        parts.append(f"淋巴结转移情况：{rest}")
    return "；".join(parts)


def _station_sentence(match) -> str:
    name, hit, examined = match.group(1), int(match.group(2)), int(match.group(3))
    if name.endswith("总数"):
        return f"共{examined}枚淋巴结，{hit}枚见转移" if hit else f"共{examined}枚淋巴结，均未见转移"
    site = name if name.endswith("淋巴结") else f"{name}淋巴结"
    return f"{site}{examined}枚中{hit}枚见转移" if hit else f"{site}{examined}枚均阴性"


def _stations_prose(text: str) -> str:
    return "，".join(_station_sentence(match) for match in _BREAKDOWN.finditer(text)
                    if not match.group(1).endswith("总数"))


def paraphrase_cell(name: str, value: str, seed: str) -> str | None:
    """One template cell -> a prose sentence, or None to drop an empty cell."""
    head = value.strip()
    if not head or head.startswith("/"):
        return None
    if name == "肌层浸润":
        level = classify_muscle(head)
        return _pick(_MUSCLE_PROSE[level], seed) if level is not None else f"肌层浸润情况为{head}"
    if name == "脉管侵犯":
        if re.search(r"\+|阳性|癌栓|广泛", head[:30]):
            return _pick(_LVSI_PROSE[True], seed)
        if re.search(r"-|未见|阴性|不确定|难以确定|需行|正在", head[:30]):
            return _pick(_LVSI_PROSE[False], seed)
        return f"脉管侵犯情况为{head}"
    if name == "组织学分化程度":
        level = classify_grade(head)
        return _pick(_GRADE_PROSE[level], seed) if level is not None else f"分化程度记录为{head}"
    if name == "淋巴结转移情况":
        return _nodes_prose(head, seed) or None
    if name in _SITE_NAMES:
        positive = is_positive(head)
        if positive is None:
            return f"{_SITE_NAMES[name]}：{head}"
        return _pick(_SITE_PROSE[positive], seed).format(site=_SITE_NAMES[name])
    if name == "淋巴结":
        return head
    if "淋巴结" in name:
        return f"{name}：{head}"
    return f"{name}为{head}"


def paraphrase_block(text: str, seed: str) -> str:
    found = list(_FIELD.finditer(text))
    if not found:
        return text
    pieces = [text[: found[0].start()]]
    for index, match in enumerate(found):
        line_end = text.find("\n", match.end())
        line_end = len(text) if line_end < 0 else line_end
        next_field = found[index + 1].start() if index + 1 < len(found) else len(text)
        end = min(line_end, next_field)
        name, value = match.group(1), text[match.end():end]
        if _TEMPLATE_FIELD.match(name):
            sentence = paraphrase_cell(name, value, seed + name)
            pieces.append(("" if sentence is None else sentence + "。") + ("\n" if end == line_end and end < len(text) else " "))
            tail_start = end
        else:
            tail_start = match.start()
        pieces.append(text[tail_start:next_field] if index + 1 < len(found) else text[tail_start:])
    out = "".join(pieces)
    # Station breakdowns that sit on their own line, outside any cell: `左盆腔（1/5），…`
    out = re.sub(_BREAKDOWN.pattern + r"\s*[，,]?", lambda match: _station_sentence(match) + "，", out)
    lines = [" ".join(line.split()).rstrip("，,") for line in out.splitlines()]
    return "\n".join(line for line in lines if line)


def apply_ablation(name: str, state: str) -> str:
    if name == "none":
        return state
    blocks = state.split("\n\n")
    if name == "pathology-only":
        kept = [block for block in blocks if block.startswith(PATHOLOGY_FAMILY)]
        return "\n\n".join(kept) if kept else state
    if name in {"strip-template", "paraphrase"}:
        seed = hashlib.sha256(state.encode()).hexdigest()[:16]
        out = []
        for block in blocks:
            if block.startswith("[病理]"):
                body = block[len("[病理]"):]
                body = strip_template_block(body) if name == "strip-template" else paraphrase_block(body, seed)
                out.append("[病理]\n" + body.strip() if body.strip() else "[病理]\n（无）")
            else:
                out.append(block)
        return "\n\n".join(out)
    raise ValueError(f"unknown ablation {name!r}; use one of {ABLATIONS}")


def coverage(records, name: str) -> dict:
    """Share of records whose pathology text still carries each evidence word."""
    total = len(records)
    hits = {key: 0 for key in EVIDENCE_WORDS}
    for record in records:
        state = apply_ablation(name, record["state"])
        pathology = "\n".join(block for block in state.split("\n\n") if block.startswith("[病理]"))
        for key, word in EVIDENCE_WORDS.items():
            if word in pathology:
                hits[key] += 1
    return {key: (count / total if total else None) for key, count in hits.items()}


def ablate_records(records, name: str):
    return [{**record, "state": apply_ablation(name, record["state"])} for record in records]
