"""`python -m fugojev inventory` prints counts. `build` writes gitignored JSONL."""

import argparse

from fugojev.corpus import format_report, load_cases, writable_rows
from fugojev.records import dump_jsonl


def main(argv=None):
    parser = argparse.ArgumentParser(prog="fugojev")
    sub = parser.add_subparsers(dest="cmd", required=True)
    inventory = sub.add_parser("inventory", help="print endometrial linkage counts")
    inventory.add_argument("--root", default=None, help="data root (default: $FUGOJEV_DATA)")
    build = sub.add_parser("build", help="write data/ec/records.jsonl")
    build.add_argument("--root", default=None)
    build.add_argument("--out", default="data/ec/records.jsonl")
    args = parser.parse_args(argv)
    cases, summary = load_cases(args.root)
    print(format_report(summary))
    if args.cmd == "build":
        written = dump_jsonl(args.out, writable_rows(cases))
        print(f"wrote {written} records to {args.out}")
    return summary


if __name__ == "__main__":
    main()
