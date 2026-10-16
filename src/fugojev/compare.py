"""Put the systems side by side on one split, as a markdown table.

    python -m fugojev.compare rule=runs/baseline/dev.json zero-shot=runs/probe/q35-dev.json \\
        fugojev=reports/dev/fugojev-0.8b.json

Every input is a report in the shape `evaluate`, `baseline` and `base_probe` write
(`report_from_buckets`), so the table needs no per-system adapters. Cells are
accuracy, with macro-F1 in parentheses; the last column is the majority-class
floor of the split, which every system must clear to be saying anything.
"""

import argparse
import json
from pathlib import Path

from fugojev.questions import QUESTIONS


def load_reports(specs) -> dict:
    reports = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"expected name=path, got {spec!r}")
        name, path = spec.split("=", 1)
        reports[name] = json.loads(Path(path).read_text(encoding="utf-8"))
    splits = {report.get("split") for report in reports.values()}
    if len(splits) > 1:
        raise SystemExit(f"reports come from different splits: {sorted(map(str, splits))}")
    return reports


def _cell(stats) -> str:
    if not stats or stats.get("accuracy") is None:
        return "—"
    f1 = stats.get("macro_f1")
    return f"{stats['accuracy']:.3f}" + (f" ({f1:.3f})" if f1 is not None else "")


def markdown(reports: dict) -> str:
    names = list(reports)
    header = "| question | " + " | ".join(names) + " | majority |"
    rule = "|---|" + "---|" * (len(names) + 1)
    lines = [header, rule]
    qids = [qid for qid in QUESTIONS if any(qid in report["questions"] for report in reports.values())]
    for qid in qids:
        cells = [_cell(report["questions"].get(qid)) for report in reports.values()]
        floors = [report["questions"][qid]["majority_accuracy"] for report in reports.values() if qid in report["questions"]]
        n = next(report["questions"][qid]["n"] for report in reports.values() if qid in report["questions"])
        lines.append(f"| {qid} (n={n}) | " + " | ".join(cells) + f" | {max(floors):.3f} |")
    micro = [f"{report['micro']['accuracy']:.3f}" if report["micro"]["accuracy"] is not None else "—" for report in reports.values()]
    lines.append("| micro | " + " | ".join(micro) + " | |")
    latency = []
    for report in reports.values():
        median = (report.get("runtime") or {}).get("latency_ms_per_record_median")
        latency.append("—" if median is None else f"{median:.0f} ms")
    if any(cell != "—" for cell in latency):
        lines.append("| latency / record | " + " | ".join(latency) + " | |")
    split = next(iter(reports.values())).get("split", "?")
    records = next(iter(reports.values())).get("n_records", "?")
    return f"split {split}, {records} records; accuracy (macro-F1)\n\n" + "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare report files on one split")
    parser.add_argument("reports", nargs="+", help="name=path pairs")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    table = markdown(load_reports(args.reports))
    print(table)
    if args.out:
        Path(args.out).write_text(table + "\n", encoding="utf-8")
    return table


if __name__ == "__main__":
    main()
