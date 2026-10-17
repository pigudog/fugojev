"""Rule baseline: hand-written patterns over the same state text the model reads.

    python -m fugojev.baseline --split dev
    python -m fugojev.baseline --split dev --out runs/baseline/dev.json

Pathology reports at this centre mostly follow a template (`肌层浸润:`, `脉管侵犯:（-）`,
`宫颈管间质:（+）`, `淋巴结转移情况:总数（0/24）` …), so the rules read those fields
first and fall back to free-text patterns. A question whose rules do not fire
gets the majority label of the *train* split, never of the split being scored.
FIGO is composed from the component findings, highest stage first.

This is the floor the fine-tuned model has to clear. Patterns were tuned on train
only; dev was looked at once, to report.
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from fugojev.evaluate import (ABLATIONS, COMPOSED_QIDS, format_report, records_for_split, report_from_buckets,
                              select_records)
from fugojev.figo2023 import (
    cervical_from_text,
    extent_from_text,
    molecular_from_stage,
    normalize_histotype,
    normalize_p53,
    stage_from_predictions,
)
from fugojev.questions import COMPONENT_QIDS, GRADE_CRITERIA, MUSCLE_CRITERIA
from fugojev.records import load_records
from fugojev.split import FOLDS, fold_records

# A template field is `名称:` or `名称：` at the start of a line or after whitespace.
_FIELD = re.compile(r"(?:(?<=[\s:：])|^)([\u4e00-\u9fff（）()\w]{2,14})[:：]")
_NEGATED = re.compile(r"未见|未累及|未受|未侵|未达|未发现|未找到|未及|无|阴性|不伴|排除|不考虑|（-）|\(-\)|：-")
_POSITIVE_MARK = re.compile(r"（\+）|\(\+\)|^\s*\+|阳性|(?<!未)见癌|(?<!未见)癌累及|(?<!未见)癌转移|(?<!未)见肿瘤|(?<!未)受累")


def sections(state: str, title: str) -> str:
    return "\n".join(block for block in state.split("\n\n") if block.startswith(f"[{title}]"))


def fields(text: str) -> list[tuple[str, str]]:
    """(field name, value) pairs in document order. Value runs to the next field."""
    found = list(_FIELD.finditer(text))
    pairs = []
    for index, match in enumerate(found):
        end = found[index + 1].start() if index + 1 < len(found) else len(text)
        pairs.append((match.group(1), text[match.end():end].strip()))
    return pairs


def values_of(pairs, name_pattern: str) -> list[str]:
    pattern = re.compile(name_pattern)
    return [value for name, value in pairs if pattern.search(name)]


def is_positive(value: str) -> bool | None:
    """Template cell semantics: （+） positive, （-）/未见 negative, `/` unknown."""
    head = value[:40]
    if not head or head.startswith("/"):
        return None
    if _POSITIVE_MARK.search(head):
        return True
    if _NEGATED.search(head):
        return False
    return None


def clauses(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"[。；;\n]", text) if part.strip()]


def positive_clauses(text: str, pattern: str) -> bool:
    """Any clause matching `pattern` that carries no negation marker."""
    regex = re.compile(pattern)
    return any(regex.search(clause) and not _NEGATED.search(clause) for clause in clauses(text))


# ---- components -------------------------------------------------------------

_DEEP = re.compile(r"≥|>=|＞|>|大于|超过|深肌层|全层|浆膜|4/5|3/4|2/3|外1/2|外侧1/2|达1/2")
_SHALLOW = re.compile(r"＜|<|小于|不足|未达|浅|内1/2|1/3|1/4|1/5|小灶")
_NONE_MUSCLE = re.compile(r"^(?:无|未见|未见肯定|未见明确|局限于内膜|-|（-）)")


def classify_muscle(value: str) -> int | None:
    head = value[:60]
    if _NONE_MUSCLE.search(head):
        return 0
    if _DEEP.search(head):
        return 2
    if _SHALLOW.search(head):
        return 1
    if re.search(r"约?1/2", head):
        return 2
    return None


def rule_muscle(pathology: str) -> int | None:
    for value in reversed(values_of(fields(pathology), r"^肌层浸润$")):
        level = classify_muscle(value)
        if level is not None:
            return level
    for clause in reversed(clauses(pathology)):
        if "肌层" not in clause:
            continue
        if re.search(r"未见(?:明确|明显|肯定)?肌层(?:浸润|侵犯)|局限于内膜", clause):
            return 0
        if re.search(r"浸润|侵及|侵犯|累及", clause):
            if _DEEP.search(clause):
                return 2
            if _SHALLOW.search(clause):
                return 1
    return None


def rule_lvsi(pathology: str) -> bool | None:
    for value in reversed(values_of(fields(pathology), r"^脉管侵犯$")):
        head = value[:30]
        if re.search(r"\+|阳性|癌栓|广泛", head):
            return True
        if re.search(r"-|未见|阴性|不确定|难以确定|需行|正在", head):
            return False
    if positive_clauses(pathology, r"脉管内?(?:见|可见)?癌栓|脉管(?:侵犯|浸润)\s*[（(]?\+"):
        return True
    if re.search(r"未见(?:明确|肯定)?(?:神经、|神经及)?脉管", pathology):
        return False
    return None


_GRADE_WORDS = ((r"III|Ⅲ|3|低分化", 2), (r"II|Ⅱ|2|中分化", 1), (r"I|Ⅰ|1|高分化", 0))


def classify_grade(value: str) -> int | None:
    head = value[:30]
    for pattern, level in _GRADE_WORDS:
        if re.search(pattern, head):
            return level
    return None


def rule_grade(pathology: str) -> int | None:
    for value in reversed(values_of(fields(pathology), r"^组织学分化程度$")):
        level = classify_grade(value)
        if level is not None:
            return level
    for match in reversed(list(re.finditer(r"(?:癌|腺癌)\s*[,，]?\s*(?:\(|（)?(I{1,3}|Ⅰ|Ⅱ|Ⅲ|[123])\s*(?:[-~～](?:I{1,3}|Ⅰ|Ⅱ|Ⅲ|[123]))?\s*级", pathology))):
        return classify_grade(match.group(1))
    for match in reversed(list(re.finditer(r"(高|中|低)分化", pathology))):
        return {"高": 0, "中": 1, "低": 2}[match.group(1)]
    return None


_FRACTION = re.compile(r"(\d+)\s*/\s*(\d+)")
_BREAKDOWN = re.compile(r"([\u4e00-\u9fff]{2,16})\s*[（(]\s*(\d+)\s*/\s*(\d+)\s*[）)]")


def _station_positive(value: str) -> bool | None:
    head = value[:80]
    match = _FRACTION.search(head)
    if match:
        return int(match.group(1)) > 0
    if re.search(r"(?<!未)见癌(?:转移|累及)|见肿瘤转移|转移性癌", head):
        return True
    if re.search(r"未见(?:癌)?(?:转移|累及)|（-）|\(-\)", head):
        return False
    return None


def rule_nodes(pathology: str) -> str | None:
    pairs = fields(pathology)
    pelvic = paraaortic = False
    seen = False
    for name, value in pairs:
        if "淋巴结" not in name or name == "淋巴结转移情况":
            continue
        positive = _station_positive(value)
        if positive is None:
            continue
        seen = True
        if positive:
            if "腹主动脉旁" in name or "腹动脉旁" in name:
                paraaortic = True
            elif "腹股沟" not in name:
                pelvic = True
    totals = values_of(pairs, r"^淋巴结转移情况$")
    any_total = None
    for value in totals:
        positive = _station_positive(value)
        if positive is not None:
            any_total = positive
        # Breakdown inside the summary cell: `左盆腔（1/5），腹主动脉旁（2/4）`.
        for name, hit, _ in _BREAKDOWN.findall(value):
            if int(hit) <= 0:
                continue
            seen = True
            if "腹主动脉旁" in name or "腹动脉旁" in name or "肾静脉" in name:
                paraaortic = True
            elif "腹股沟" not in name:
                pelvic = True
    if pelvic and paraaortic:
        return "pelvic_and_paraaortic"
    if paraaortic:
        return "paraaortic"
    if pelvic:
        return "pelvic"
    if any_total:
        return "pelvic"
    if seen or any_total is False:
        return "none"
    if re.search(r"淋巴结.{0,12}未见(?:癌)?转移", pathology):
        return "none"
    return None


# ---- FIGO composition --------------------------------------------------------


def _any_field_positive(pairs, name_pattern: str) -> bool:
    return any(is_positive(value) for value in values_of(pairs, name_pattern))


def rule_figo(state: str, muscle: int | None, nodes: str | None) -> str | None:
    pathology = sections(state, "病理")
    pairs = fields(pathology)
    imaging_surgery = sections(state, "影像") + "\n" + sections(state, "手术")
    distant = (
        _any_field_positive(pairs, r"腹股沟淋巴结")
        or any(is_positive(value) and re.search(r"大网膜|腹膜|肠壁|肠段|肠系膜", value[:60])
               for value in values_of(pairs, r"^其他$|^另送"))
        or positive_clauses(pathology, r"(?:大网膜|腹膜|肠壁|肠系膜|膀胱壁).{0,12}(?:见癌|癌累及|转移|肿瘤结节)")
        or positive_clauses(pathology + "\n" + imaging_surgery, r"(?:肺|肝|骨|腹股沟|远处|腹腔内?|大网膜|腹膜)[^。;；]{0,12}转移")
    )
    if distant:
        return "IVB"
    if positive_clauses(pathology, r"(?:膀胱|直肠)[^，,。;；]{0,6}(?:粘|黏)膜"):
        return "IVA"
    if nodes == "pelvic_and_paraaortic" or nodes == "paraaortic":
        return "IIIC2"
    if nodes == "pelvic":
        return "IIIC1"
    if _any_field_positive(pairs, r"^(?:左|右)?宫旁$|^阴道$") or positive_clauses(
        pathology, r"(?:阴道壁?|宫旁)[^，,。;；]{0,6}(?:受累|浸润|侵犯|见癌|累及)"
    ):
        return "IIIB"
    if _any_field_positive(pairs, r"卵巢|输卵管|双附件|浆膜") or positive_clauses(
        pathology, r"(?:浆膜|附件|卵巢|输卵管)[^，,。;；]{0,6}(?:受累|浸润|侵犯|见癌|累及)|(?:侵及|达|穿透)浆膜"
    ):
        return "IIIA"
    if _any_field_positive(pairs, r"宫颈管?间质") or positive_clauses(
        pathology, r"(?:侵犯|累及|浸润)宫颈管?间质|宫颈管?间质[^，,。;；]{0,6}(?:浸润|受累|侵犯|见癌)"
    ):
        return "II"
    if muscle is None:
        return None
    return "IB" if muscle == 2 else "IA"


# ---- FIGO 2023 components (v1.3) ---------------------------------------------


def rule_histotype(pathology: str) -> str | None:
    """The template's `病理类型:` cell first, then the diagnosis line."""
    for value in values_of(fields(pathology), r"^病理类型$|^组织学类型$"):
        found = normalize_histotype(value)
        if found:
            return found
    for line in pathology.splitlines()[::-1]:
        if re.search(r"癌肉瘤|浆液性癌|透明细胞癌|未分化癌|去分化癌|混合性?癌|内膜样腺?癌", line):
            return normalize_histotype(line)
    return None


def rule_cervical_stroma(pathology: str) -> bool | None:
    value = cervical_from_text(pathology)
    if value is not None:
        return value
    if positive_clauses(pathology, r"(?:侵犯|累及|浸润)宫颈管?间质|宫颈管?间质[^，,。;；]{0,6}(?:浸润|受累|侵犯|见癌)"):
        return True
    return None


def rule_extent(state: str) -> str | None:
    """Highest extrauterine site. Peritoneal and distant sites come from free text, the
    template cells give adnexa, serosa, vagina and parametrium."""
    pathology = sections(state, "病理")
    pairs = fields(pathology)
    imaging_surgery = sections(state, "影像") + "\n" + sections(state, "手术")
    if (
        _any_field_positive(pairs, r"腹股沟淋巴结")
        or positive_clauses(pathology + "\n" + imaging_surgery,
                            r"(?:肺|肝实质|骨|脑|腹股沟|锁骨上|膈脚后|纵隔)[^。;；]{0,12}转移")
    ):
        return "distant"
    if (
        any(is_positive(value) and re.search(r"大网膜|结肠旁沟|肠系膜|膈肌|肝表面|肝膈面|小肠|腹壁|肝肾隐窝|脾", value[:80])
            for value in values_of(pairs, r"^其他$|^另送"))
        or positive_clauses(pathology, r"(?:大网膜|结肠旁沟|肠系膜|膈肌|肝表面|肝膈面|肝肾隐窝|腹壁)[^。;；]{0,16}(?:见癌|癌累及|肿瘤累及|转移|见肉瘤)")
    ):
        return "abdominal_peritoneum"
    if positive_clauses(pathology, r"(?:膀胱|直肠)[^，,。;；]{0,6}(?:粘|黏)膜[^，,。;；]{0,6}(?:见癌|累及|浸润)"):
        return "bladder_bowel_mucosa"
    if (
        any(is_positive(value) and re.search(r"腹膜|直肠前壁|膀胱反折|膀胱腹膜|盆底|道格拉斯|子宫直肠", value[:60])
            for value in values_of(pairs, r"^其他$|^另送"))
        or positive_clauses(pathology, r"(?:盆底腹膜|膀胱反折腹膜|膀胱腹膜|直肠前壁|盆腔腹膜)[^。;；]{0,12}(?:见癌|癌累及|肿瘤累及|转移)")
    ):
        return "pelvic_peritoneum"
    template = extent_from_text(pathology)
    if template in ("vagina_parametrium", "adnexa", "serosa"):
        return template
    if positive_clauses(pathology, r"(?:阴道壁?|宫旁)[^，,。;；]{0,6}(?:受累|浸润|侵犯|见癌|累及)"):
        return "vagina_parametrium"
    if positive_clauses(pathology, r"(?:附件|卵巢|输卵管)[^，,。;；]{0,6}(?:受累|浸润|侵犯|见癌|累及)"):
        return "adnexa"
    if positive_clauses(pathology, r"(?:侵及|达|穿透|累及)浆膜"):
        return "serosa"
    return template


_P53 = re.compile(r"[Pp]53\s*[（(]([^）)]{1,12})[）)]")
_MMR = re.compile(r"(MLH1|PMS2|MSH2|MSH6)\s*(?:\([^)]{0,6}\))?\s*[（(]([^）)]{1,10})[）)]")


def rule_p53(ihc: str) -> bool | None:
    hits = _P53.findall(ihc)
    if not hits:
        return None
    for cell in hits:
        found = normalize_p53(f"P53（{cell}）")
        if found is not None:
            return found
    return None


def rule_mmr(ihc: str) -> bool | None:
    hits = _MMR.findall(ihc)
    if not hits:
        return None
    if any(re.fullmatch(r"\s*[-－]\s*|少量-|缺失", cell) for _, cell in hits):
        return True
    return False


_POLE_HOTSPOT = re.compile(
    r"POLE[^。；;]{0,60}(?:P286R|V411L|S459F|A456P|S297F|F367S|L424[IV]|M295R|P436R|M444K|A428T"
    r"|Pro286Arg|Val411Leu|Ser459Phe|Ala456Pro|Ser297Phe|Phe367Ser|Leu424(?:Ile|Val)|Met295Arg|Pro436Arg|Met444Lys|Ala428Thr)"
)
_POLE_DETECTED = re.compile(r"检测到POLE基因[^。；;]{0,40}(?:已报道的)?(?:可能)?致病")
_POLE_UNCERTAIN = re.compile(r"POLE[^。；;]{0,60}(?:意义尚不明确|意义未明|VUS)")


def rule_pole(molecular: str) -> bool | None:
    """Pathogenic exonuclease-domain hotspot stated for POLE; a 'detected ... pathogenic'
    sentence counts too. Negations ('未检测到') and variants of uncertain significance
    do not, and a report that lists no POLE variant is False. None without a report."""
    if not molecular.strip():
        return None
    if _POLE_HOTSPOT.search(molecular):
        return True
    if _POLE_DETECTED.search(molecular) and not _POLE_UNCERTAIN.search(molecular):
        return True
    return False


# ---- driver -----------------------------------------------------------------


def predict_rules(state: str) -> dict:
    """Rule answers in option-name form; None where no rule fired."""
    pathology = sections(state, "病理")
    ihc = sections(state, "免疫组化") + "\n" + pathology
    molecular = sections(state, "分子病理")
    muscle = rule_muscle(pathology)
    nodes = rule_nodes(pathology)
    lvsi = rule_lvsi(pathology)
    grade = rule_grade(pathology)
    histotype = rule_histotype(pathology)
    cervical = rule_cervical_stroma(pathology)
    extent = rule_extent(state)
    p53 = rule_p53(ihc)
    mmr = rule_mmr(ihc)
    pole = rule_pole(molecular)
    answers = {
        "muscle_invasion": None if muscle is None else MUSCLE_CRITERIA[muscle],
        "lvsi": None if lvsi is None else str(lvsi).lower(),
        "lymph_node": nodes,
        "grade": None if grade is None else GRADE_CRITERIA[grade],
        "figo": rule_figo(state, muscle, nodes),
        "histotype": histotype,
        "cervical_stroma": None if cervical is None else str(cervical).lower(),
        "extent": extent,
        "p53": None if p53 is None else str(p53).lower(),
        "mmr": None if mmr is None else str(mmr).lower(),
        "pole": None if pole is None else str(pole).lower(),
    }
    return answers


def composed_rule_stages(answers: dict, defaults: dict, gold_stage: str | None) -> dict:
    """The 2023 stage from the component rules, after the same defaults that score
    the components fill the blanks (no extrauterine site, no POLE, train majority
    elsewhere). `figo2023` reads the molecular class from the notes;
    `figo2023_composed_given` takes it as structured input, like `evaluate` does."""
    filled = dict(answers)
    for qid in COMPONENT_QIDS:
        if filled.get(qid) is None:
            filled[qid] = {"extent": "none", "pole": "false"}.get(qid, defaults.get(qid))
    notes_only = stage_from_predictions(filled)
    # The rule system has no direct stage head, so its `figo2023` *is* the composed
    # answer; it is written under both names so the CV tables line up with the model.
    return {
        "figo2023": notes_only,
        "figo2023_composed": notes_only,
        "figo2023_composed_given": stage_from_predictions(filled, molecular_from_stage(gold_stage) or "NSMP"),
    }


def gold_name(question: dict) -> str:
    label = question["label"]
    if question["type"] == "noul":
        return str(bool(label)).lower()
    if question["type"] == "score":
        return str(question["criteria"][int(label)])
    return str(label)


def train_majority(records) -> dict:
    counts = defaultdict(Counter)
    for record in records:
        for qid, question in record["questions"].items():
            counts[qid][gold_name(question)] += 1
    return {qid: counter.most_common(1)[0][0] for qid, counter in counts.items()}


def score_rules(records, defaults, rows=None) -> tuple[dict, dict]:
    buckets = defaultdict(lambda: {"golds": [], "preds": []})
    fired = Counter()
    asked = Counter()
    for index, record in enumerate(records):
        answers = predict_rules(record["state"])
        questions = dict(record["questions"])
        if "figo2023" in questions:
            gold_stage = questions["figo2023"]["label"]
            answers.update(composed_rule_stages(answers, defaults, gold_stage))
            for qid in COMPOSED_QIDS:
                questions[qid] = questions["figo2023"]
        for qid, question in questions.items():
            asked[qid] += 1
            answer = answers.get(qid)
            if answer is not None:
                fired[qid] += 1
            else:
                answer = defaults.get(qid, "none")
            buckets[qid]["golds"].append(gold_name(question))
            buckets[qid]["preds"].append(answer)
            if rows is not None:
                rows.append({"index": index, "qid": qid, "gold": gold_name(question), "pred": answer,
                             "fired": answers.get(qid) is not None})
    coverage = {qid: fired[qid] / asked[qid] for qid in sorted(asked)}
    return buckets, coverage


def main(argv=None):
    parser = argparse.ArgumentParser(description="Rule baseline over the state text")
    parser.add_argument("--records", default="data/ec/records.jsonl")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--fold", type=int, default=None, help="score the held-out records of this CV fold")
    parser.add_argument("--folds", type=int, default=FOLDS)
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--ablation", choices=ABLATIONS, default="none")
    parser.add_argument("--rows", default="", help="write per-question gold/pred rows here")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    scored, split = select_records(args)
    records = load_records(args.records)
    if args.fold is None:
        fitting = records_for_split(records, "train")
    else:
        fitting = fold_records(records, args.fold, args.folds, held_out=False)
    defaults = train_majority(fitting)
    rows = [] if args.rows else None
    buckets, coverage = score_rules(scored, defaults, rows)
    if rows is not None:
        out = Path(args.rows)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"split": split, "ablation": args.ablation, "rows": rows}, ensure_ascii=False) + "\n",
                       encoding="utf-8")
    report = report_from_buckets(buckets, len(scored), system="rule baseline")
    report["split"] = split
    report["ablation"] = args.ablation
    report["defaults"] = defaults
    report["rule_coverage"] = coverage
    print(format_report(report))
    print("rule fired: " + "  ".join(f"{qid} {value:.2f}" for qid, value in coverage.items()))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    main()
