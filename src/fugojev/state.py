"""Turn linked documents into one state string.

Each report keeps its ending, where the conclusion usually sits. A tight character
budget is spent on pathology first, then imaging, surgery, admission, and clinic,
so a long operative note cannot push the pathology conclusion out. Gold columns
and outpatient TNM never have a field to be read from.
"""

from dataclasses import dataclass
from datetime import date

SECTION_CHARS = 800
MAX_STATE_CHARS = 3200
# When the budget runs out, earlier titles in this list keep their conclusion.
SECTION_PRIORITY = ("病理", "影像", "手术", "入院", "门诊")

# Days before the anchor, days after. Staging pathology is the surgical specimen,
# so it is allowed to arrive after the operation. Clinic notes stop at the anchor.
WINDOWS = {
    "pathology": (30, 120),
    "imaging": (180, 30),
    "clinic": (365, 0),
    "surgery": (7, 3),
}
LIMITS = {"pathology": 3, "imaging": 3, "clinic": 2, "surgery": 2}
IMAGING_HINTS = ("盆", "子宫", "内膜", "MRI", "磁共振", "CT", "超声", "核磁")
PATHOLOGY_HINTS = ("子宫", "内膜", "手术", "全切")

_IMAGING_MARKS = ("放射", "超声", "影像", "核医", "ct", "mr", "磁", "pet", "x线", "钼靶", "造影")


@dataclass(frozen=True)
class Doc:
    when: date
    text: str
    hint: str = ""


def clean_text(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "nat", "<na>"}:
        return ""
    return text


def _tidy(text: str) -> str:
    lines = [" ".join(line.split()) for line in str(text).splitlines()]
    return "\n".join(line for line in lines if line)


def clip_tail(text: str, limit: int) -> str:
    text = _tidy(text)
    if len(text) <= limit:
        return text
    return "…" + text[-(limit - 1):]


def render_state(sections, *, section_chars=SECTION_CHARS, max_chars=MAX_STATE_CHARS) -> str:
    """Keep each section's conclusion, and spend a tight budget on pathology first.

    Reading order stays clinical (clinic, admission, imaging, pathology, surgery).
    A long operative note no longer pushes the pathology conclusion out of the window.
    """
    blocks = []
    for title, body in sections:
        body = _tidy(body)
        if body:
            blocks.append((title, body))
    rank = {title: index for index, title in enumerate(SECTION_PRIORITY)}
    claim_order = sorted(range(len(blocks)), key=lambda index: rank.get(blocks[index][0], len(rank)))
    chosen = {}
    remaining = max_chars
    for index in claim_order:
        if remaining < 24:
            break
        title, body = blocks[index]
        room = min(section_chars, remaining - len(title) - 4)
        if room < 12:
            break
        block = f"[{title}]\n{clip_tail(body, room)}"
        chosen[index] = block
        remaining -= len(block) + 2
    return "\n\n".join(chosen[index] for index in range(len(blocks)) if index in chosen)


def clinic_body(row: dict) -> str:
    parts = []
    chief = clean_text(row.get("主诉"))
    history = clean_text(row.get("病史描述"))
    if chief:
        parts.append(f"主诉：{chief}")
    if history:
        parts.append(f"病史：{history}")
    return "\n".join(parts)


def admission_body(row: dict) -> str:
    parts = []
    chief = clean_text(row.get("主诉"))
    illness = clean_text(row.get("现病史"))
    if chief:
        parts.append(f"主诉：{chief}")
    if illness:
        parts.append(f"现病史：{illness}")
    return "\n".join(parts)


def surgery_body(row: dict) -> str:
    parts = []
    for label, key in (("手术", "文书手术名称"), ("经过", "手术经过"), ("病灶", "病灶描述")):
        value = clean_text(row.get(key))
        if value:
            parts.append(f"{label}：{value}")
    return "\n".join(parts)


def report_body(row: dict) -> str:
    parts = []
    for label, key in (("项目", "项目小类"), ("所见", "所见"), ("报告", "报告内容")):
        value = clean_text(row.get(key))
        if value:
            parts.append(f"{label}：{value}")
    return "\n".join(parts)


def is_pathology(category, subtype) -> bool:
    return "病理" in f"{clean_text(category)}{clean_text(subtype)}"


def is_imaging(category, subtype) -> bool:
    if is_pathology(category, subtype):
        return False
    blob = f"{clean_text(category)} {clean_text(subtype)}".lower()
    return any(mark in blob for mark in _IMAGING_MARKS)


def in_window(when: date | None, anchor: date | None, before_days: int, after_days: int) -> bool:
    if when is None or anchor is None:
        return False
    delta = (when - anchor).days
    return -before_days <= delta <= after_days


def pick_docs(docs, anchor, before_days, after_days, limit, prefer=(), text_marks=()):
    """Nearest documents in the window.

    A missing hint costs 45 days. A missing `text_marks` hit costs more than the
    window, so a report that states muscle depth beats a nearer report that does not.
    """
    if anchor is None or limit <= 0:
        return []
    ranked = []
    for doc in docs:
        if not doc.text.strip() or not in_window(doc.when, anchor, before_days, after_days):
            continue
        days = abs((doc.when - anchor).days)
        penalty = 0 if not prefer or any(mark in doc.hint for mark in prefer) else 45
        if text_marks and not any(mark in doc.text for mark in text_marks):
            penalty += 200
        ranked.append((days + penalty, days, doc))
    ranked.sort(key=lambda item: (item[0], item[1]))
    chosen = [doc for _, _, doc in ranked[:limit]]
    chosen.sort(key=lambda doc: doc.when)
    return chosen


def assemble_sections(*, admission_text, clinics, imaging, pathology, surgeries, anchor):
    """Ordered so a tail trim keeps pathology and the operative note."""
    sections = []
    for doc in pick_docs(clinics, anchor, *WINDOWS["clinic"], LIMITS["clinic"]):
        sections.append(("门诊", doc.text))
    if clean_text(admission_text):
        sections.append(("入院", admission_text))
    for doc in pick_docs(imaging, anchor, *WINDOWS["imaging"], LIMITS["imaging"], IMAGING_HINTS):
        sections.append(("影像", doc.text))
    for doc in pick_docs(
        pathology, anchor, *WINDOWS["pathology"], LIMITS["pathology"], PATHOLOGY_HINTS,
        text_marks=("肌层",),
    ):
        sections.append(("病理", doc.text))
    for doc in pick_docs(surgeries, anchor, *WINDOWS["surgery"], LIMITS["surgery"]):
        sections.append(("手术", doc.text))
    return sections
