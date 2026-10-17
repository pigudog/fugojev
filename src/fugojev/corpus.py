"""Join the endometrial gold sheet to admission, surgery, imaging, and clinic notes.

National-id numbers and names are used only to link rows in memory. They are not
written to the JSONL. Outpatient TNM columns and the discharge summary are not read.
"""

import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from fugojev.questions import (
    FIGO_LEVELS,
    normalize_figo,
    normalize_grade,
    normalize_lvsi,
    normalize_muscle,
    normalize_nodes,
    questions_from_raw,
)
from fugojev.figo2023 import gold_components, gold_stage, normalize_histotype, normalize_molecular
from fugojev.split import split_of
from fugojev.state import (
    WINDOWS,
    Doc,
    admission_body,
    assemble_sections,
    clean_text,
    clinic_body,
    in_window,
    is_imaging,
    pathology_kind,
    render_state,
    report_body,
    surgery_body,
)

# Data root and gold-sheet name. Set FUGOJEV_DATA / FUGOJEV_GOLD, or pass `--root`.
UTERUS = Path(os.environ.get("FUGOJEV_DATA") or "data/raw/uterus")
GOLD_NAME = os.environ.get("FUGOJEV_GOLD") or "gold.xlsx"
# Per-case FIGO 2023 decisions that the sheet cannot make (2009 IVB split into
# IIIB2 / IVB / IVC, reviewed by eye). JSON {hospital_id: {"extent": ..., "note": ...}}.
# Lives under data/, which is git-ignored.
OVERRIDES_2023 = Path(os.environ.get("FUGOJEV_OVERRIDES") or "data/ec/gold2023-overrides.json")

_BLANK = {"", "na", "n/a", "null", "none", "-", "nan"}


@dataclass
class Case:
    hospital_id: str
    split: str
    state: str
    questions: dict
    links: dict
    used: dict


def _log(message):
    print(message, file=sys.stderr, flush=True)


def norm_id(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    text = str(value).strip()
    if text.lower() in _BLANK or text.lower() == "<na>":
        return ""
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text


def parse_date(value) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    text = str(value).strip()
    if not text or text.lower() in _BLANK or text.lower() == "<na>":
        return None
    day = text[:10].replace("/", "-").replace(".", "-")
    try:
        return datetime.strptime(day, "%Y-%m-%d").date()
    except ValueError:
        pass
    try:
        serial = float(text)
    except ValueError:
        return None
    if 20000 < serial < 80000:
        return (datetime(1899, 12, 30) + timedelta(days=int(serial))).date()
    return None


def _blank(value) -> bool:
    return clean_text(value).lower() in _BLANK


def read_excel(path, columns=None):
    path = Path(path)
    kwargs = {"dtype": str, "sheet_name": 0}
    if columns:
        kwargs["usecols"] = list(columns)
    try:
        frame = pd.read_excel(path, engine="calamine", **kwargs)
    except Exception as first:
        try:
            frame = pd.read_excel(path, engine="openpyxl", **kwargs)
        except Exception as second:
            raise SystemExit(f"could not read {path.name}: {first}; {second}") from second
    frame.columns = [str(column).strip() for column in frame.columns]
    missing = [name for name in (columns or []) if name not in frame.columns]
    if missing:
        raise SystemExit(f"{path.name} missing columns {missing}")
    return frame


def _node_column(columns) -> str:
    for name in columns:
        if str(name).startswith("淋巴结转移 0"):
            return str(name)
    raise SystemExit("gold sheet is missing the coded 淋巴结转移 column")


def _choose_admission(options, anchor):
    usable = [item for item in options if item[1].strip()]
    if not usable:
        return None
    dated = [item for item in usable if item[0] is not None]
    if anchor is not None and dated:
        dated.sort(key=lambda item: abs((item[0] - anchor).days))
        return dated[0]
    return max(usable, key=lambda item: len(item[1]))


def _docs_by_card(frame, date_column, body_fn, cards):
    grouped = defaultdict(list)
    counts = Counter()
    for row in frame.to_dict("records"):
        card = norm_id(row.get("身份证号"))
        if card not in cards:
            continue
        counts[card] += 1
        body = body_fn(row)
        when = parse_date(row.get(date_column))
        if body and when is not None:
            hint = clean_text(row.get("项目小类"))
            grouped[card].append(Doc(when, body, hint))
    return grouped, counts


def load_cases(root=None):
    root = Path(root) if root is not None else UTERUS
    gold_path = root / GOLD_NAME
    if not gold_path.exists():
        raise SystemExit(
            f"gold sheet not found: {gold_path}. Set FUGOJEV_DATA to the directory holding the "
            "four note workbooks and FUGOJEV_GOLD to the gold sheet path relative to it, or pass --root."
        )
    _log(f"reading {gold_path.name}")
    gold = read_excel(gold_path)
    node_column = _node_column(gold.columns)
    for name in ("住院号", "手术日期", "FIGO分期", "组织学分级", "肌层浸润", "脉管侵犯"):
        if name not in gold.columns:
            raise SystemExit(f"{gold_path.name} missing column {name}")

    _log("reading 入院记录.xlsx")
    admissions = read_excel(root / "入院记录.xlsx", ["身份证号", "住院号", "入院日期", "主诉", "现病史"])
    by_hospital = defaultdict(list)
    for row in admissions.to_dict("records"):
        hospital_id = norm_id(row.get("住院号"))
        if not hospital_id:
            continue
        by_hospital[hospital_id].append((
            parse_date(row.get("入院日期")),
            admission_body(row),
            norm_id(row.get("身份证号")),
        ))

    pending = []
    cards = set()
    skipped = 0
    for row in gold.to_dict("records"):
        hospital_id = norm_id(row.get("住院号"))
        if not hospital_id:
            skipped += 1
            continue
        anchor = parse_date(row.get("手术日期"))
        chosen = _choose_admission(by_hospital.get(hospital_id, []), anchor)
        card = chosen[2] if chosen and chosen[2] else ""
        if not card:
            for _, _, other in by_hospital.get(hospital_id, []):
                if other:
                    card = other
                    break
        if card:
            cards.add(card)
        pending.append((hospital_id, anchor, card, chosen, row))

    _log("reading 手术记录.xlsx")
    surgery = read_excel(root / "手术记录.xlsx", ["身份证号", "文书手术名称", "手术时间", "手术经过", "病灶描述"])
    surgery_docs, surgery_counts = _docs_by_card(surgery, "手术时间", surgery_body, cards)

    _log("reading 检查报告.xlsx")
    reports = read_excel(
        root / "检查报告.xlsx",
        ["身份证号", "项目大类", "项目小类", "所见", "报告内容", "报告日期"],
    )
    pathology_docs, pathology_counts = defaultdict(list), Counter()
    ihc_docs, molecular_docs = defaultdict(list), defaultdict(list)
    imaging_docs, imaging_counts = defaultdict(list), Counter()
    categories = Counter()
    for row in reports.to_dict("records"):
        card = norm_id(row.get("身份证号"))
        if card not in cards:
            continue
        category, subtype = row.get("项目大类"), row.get("项目小类")
        categories[clean_text(category) or "(空)"] += 1
        body = report_body(row)
        when = parse_date(row.get("报告日期"))
        kind = pathology_kind(category, subtype)
        if kind is not None:
            pathology_counts[card] += 1
            if body and when is not None:
                bucket = {"pathology": pathology_docs, "ihc": ihc_docs, "molecular": molecular_docs}[kind]
                bucket[card].append(Doc(when, body, clean_text(subtype)))
        elif is_imaging(category, subtype):
            imaging_counts[card] += 1
            if body and when is not None:
                imaging_docs[card].append(Doc(when, body, clean_text(subtype)))

    _log("reading 门急诊病史.xlsx")
    clinic = read_excel(root / "门急诊病史.xlsx", ["身份证号", "主诉", "病史描述", "就诊日期"])
    clinic_docs, clinic_counts = _docs_by_card(clinic, "就诊日期", clinic_body, cards)

    overrides = {}
    if OVERRIDES_2023.exists():
        with open(OVERRIDES_2023, encoding="utf-8") as handle:
            overrides = {norm_id(key): value for key, value in json.load(handle).items()}
        _log(f"read {len(overrides)} FIGO 2023 overrides from {OVERRIDES_2023}")

    cases = []
    summary = Counter()
    summary["gold_rows"] = len(gold)
    summary["skipped_no_hospital_id"] = skipped
    summary["figo2023"] = Counter()
    summary["figo"] = Counter()
    summary["grade"] = Counter()
    summary["muscle"] = Counter()
    summary["lvsi"] = Counter()
    summary["nodes"] = Counter()
    fields = (
        ("figo", "FIGO分期", normalize_figo),
        ("grade", "组织学分级", normalize_grade),
        ("muscle", "肌层浸润", normalize_muscle),
        ("lvsi", "脉管侵犯", normalize_lvsi),
        ("nodes", node_column, normalize_nodes),
    )
    for hospital_id, gold_anchor, card, chosen, row in pending:
        for key, column, normalizer in fields:
            raw = row.get(column)
            if _blank(raw):
                summary[f"{key}_missing"] += 1
                continue
            label = normalizer(raw)
            if label is None:
                summary[f"{key}_unrecognized"] += 1
            else:
                summary[key][label] += 1
        anchor = gold_anchor or (chosen[0] if chosen and chosen[0] else None)
        if gold_anchor is not None:
            summary["anchor_from_surgery"] += 1
        admission_text = chosen[1] if chosen else ""
        sections = assemble_sections(
            admission_text=admission_text,
            clinics=clinic_docs.get(card, []),
            imaging=imaging_docs.get(card, []),
            pathology=pathology_docs.get(card, []),
            surgeries=surgery_docs.get(card, []),
            anchor=anchor,
            ihc=ihc_docs.get(card, []),
            molecular=molecular_docs.get(card, []),
        )
        used = Counter(title for title, _ in sections)
        links = {
            "入院": len(by_hospital.get(hospital_id, [])),
            "手术": surgery_counts[card] if card else 0,
            "病理": pathology_counts[card] if card else 0,
            "影像": imaging_counts[card] if card else 0,
            "门诊": clinic_counts[card] if card else 0,
            "免疫组化": len(ihc_docs.get(card, [])) if card else 0,
            "分子病理": len(molecular_docs.get(card, [])) if card else 0,
        }
        state = render_state(sections)
        # FIGO 2023 components: the sheet first, the pathology template for what the
        # sheet does not state, the reviewed override for 2009 IVB.
        pathology_text = "\n".join(
            doc.text for doc in pathology_docs.get(card, [])
            if in_window(doc.when, anchor, *WINDOWS["pathology"])
        )
        surgery_text = "\n".join(
            doc.text for doc in surgery_docs.get(card, [])
            if in_window(doc.when, anchor, *WINDOWS["surgery"])
        )
        molecular = normalize_molecular(row.get("分子分型"))
        components = gold_components(
            figo2009=normalize_figo(row.get("FIGO分期")),
            muscle=normalize_muscle(row.get("肌层浸润")),
            lvsi=normalize_lvsi(row.get("脉管侵犯")),
            nodes=normalize_nodes(row.get(node_column)),
            histotype=normalize_histotype(row.get("病理类型")),
            grade=normalize_grade(row.get("组织学分级")),
            molecular=molecular,
            pathology_text=pathology_text,
            surgery_text=surgery_text,
            override=overrides.get(hospital_id),
        )
        stage_2023 = gold_stage(components)
        if normalize_figo(row.get("FIGO分期")) == "IVB" and hospital_id not in overrides:
            summary["ivb_without_override"] += 1
        # POLE can only be read when a molecular report made it into the state; a
        # report that lists no POLE variant is the "not detected" answer.
        pole_label = None
        if molecular is not None and "[分子病理]" in state:
            pole_label = components["pole"]
        questions = questions_from_raw(
            figo=row.get("FIGO分期"),
            grade=row.get("组织学分级"),
            muscle=row.get("肌层浸润"),
            lvsi=row.get("脉管侵犯"),
            nodes=row.get(node_column),
            histotype=row.get("病理类型"),
            p53=row.get("P53免疫组化"),
            mmr=row.get("MLH1/PMS2/MSH2/MSH6免疫组化"),
            pole=pole_label,
            cervical_stroma=components["cervical_stroma"],
            extent=components["extent"],
            figo2023=stage_2023,
        )
        if stage_2023 is not None:
            summary["figo2023"][stage_2023] += 1
        else:
            summary["figo2023_missing"] += 1
        case = Case(
            hospital_id=hospital_id,
            split=split_of(hospital_id),
            state=state,
            questions=questions,
            links=links,
            used=dict(used),
        )
        cases.append(case)
        summary[f"split_{case.split}"] += 1
        for kind in ("入院", "手术", "病理", "影像", "门诊", "免疫组化", "分子病理"):
            if links.get(kind):
                summary[f"linked_{kind}"] += 1
            if used.get(kind):
                summary[f"used_{kind}"] += 1
        if state.strip():
            summary["nonempty_state"] += 1
        if questions:
            summary["with_questions"] += 1
        if state.strip() and questions:
            summary["writable"] += 1
            summary[f"writable_{case.split}"] += 1
    summary["with_hospital_id"] = len(cases)
    summary["report_categories"] = categories
    return cases, summary


def format_report(summary) -> str:
    def show(counter, labels):
        return [f"  {name} {counter[key]}" for name, key in labels]

    lines = [
        "子宫内膜癌语料计数",
        f"金标准行数 {summary['gold_rows']}",
        f"有住院号 {summary['with_hospital_id']}",
        f"无住院号已跳过 {summary['skipped_no_hospital_id']}",
        f"手术日期可解析 {summary['anchor_from_surgery']}",
        "FIGO",
        *show(summary["figo"], [(level, level) for level in FIGO_LEVELS]),
        f"  缺失 {summary['figo_missing']}",
        f"  无法规范化 {summary['figo_unrecognized']}",
        "组织学分级",
        *show(summary["grade"], [("I", 0), ("II", 1), ("III", 2)]),
        f"  缺失 {summary['grade_missing']}",
        f"  无法规范化 {summary['grade_unrecognized']}",
        "肌层浸润",
        *show(summary["muscle"], [("无", 0), ("<1/2", 1), (">=1/2", 2)]),
        f"  缺失 {summary['muscle_missing']}",
        f"  无法规范化 {summary['muscle_unrecognized']}",
        "脉管侵犯",
        *show(summary["lvsi"], [("无", False), ("有", True)]),
        f"  缺失 {summary['lvsi_missing']}",
        f"  无法规范化 {summary['lvsi_unrecognized']}",
        "淋巴结（编码列）",
        *show(summary["nodes"], [("无", "none"), ("盆腔", "pelvic"),
                                 ("盆腔+腹主动脉旁", "pelvic_and_paraaortic"), ("仅腹主动脉旁", "paraaortic")]),
        f"  缺失 {summary['nodes_missing']}",
        f"  无法规范化 {summary['nodes_unrecognized']}",
        "能连上文书的病例",
    ]
    for kind in ("入院", "手术", "病理", "影像", "门诊", "免疫组化", "分子病理"):
        lines.append(f"  {kind} 有记录 {summary[f'linked_{kind}']}  写入state {summary[f'used_{kind}']}")
    lines.append("FIGO 2023（由表格列+病理模板+复核推出）")
    lines.extend(f"  {stage} {count}" for stage, count in sorted(summary["figo2023"].items(), key=lambda kv: -kv[1]))
    lines.append(f"  无法推出 {summary['figo2023_missing']}")
    lines.append(f"  2009 IVB 无复核记录 {summary['ivb_without_override']}")
    lines.extend([
        "已关联病例的检查项目大类",
        *[f"  {name} {count}" for name, count in summary["report_categories"].most_common(12)],
        f"state非空 {summary['nonempty_state']}",
        f"至少一道题 {summary['with_questions']}",
        f"可写出记录 {summary['writable']}",
        "可写出记录的划分",
        f"  train {summary['writable_train']}",
        f"  dev {summary['writable_dev']}",
        f"  test {summary['writable_test']}",
    ])
    return "\n".join(lines)


def writable_rows(cases):
    for case in cases:
        if not case.state.strip() or not case.questions:
            continue
        yield {
            "hospital_id": case.hospital_id,
            "split": case.split,
            "state": case.state,
            "questions": case.questions,
        }
