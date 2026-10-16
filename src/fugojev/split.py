"""Patient-level splits. The test bucket stays locked unless a script is passed --allow-test."""

import hashlib

SPLIT_SALT = "fugojev-ec-v1"


def split_of(hospital_id: str) -> str:
    """Stable 8:1:1 split from the hospital id. Bucket 0 is test, 1 is dev, the rest train."""
    digest = hashlib.sha256(f"{SPLIT_SALT}:{hospital_id}".encode()).digest()
    bucket = int.from_bytes(digest[:4], "big") % 10
    if bucket == 0:
        return "test"
    if bucket == 1:
        return "dev"
    return "train"


FOLD_SALT = "fugojev-ec-cv-v1"
FOLDS = 5


def fold_of(hospital_id: str, folds: int = FOLDS) -> int:
    """Cross-validation fold of a non-test patient. Independent of `split_of`, so the
    dev patients are spread across folds; test patients must be excluded by the caller."""
    digest = hashlib.sha256(f"{FOLD_SALT}:{hospital_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big") % folds


def fold_records(records, fold: int, folds: int = FOLDS, held_out: bool = True):
    """Non-test records in (`held_out=True`) or out of fold `fold`."""
    if not 0 <= fold < folds:
        raise ValueError(f"fold must be in 0..{folds - 1}")
    chosen = []
    for row in records:
        hospital_id = str(row["hospital_id"])
        if split_of(hospital_id) == "test":
            continue
        if (fold_of(hospital_id, folds) == fold) == held_out:
            chosen.append(row)
    return chosen


def canonical_split(name: str) -> str:
    aliases = {"development": "dev", "val": "dev", "validation": "dev", "training": "train"}
    split = aliases.get(name, name)
    if split not in {"train", "dev", "test"}:
        raise SystemExit(f"unknown split {name!r}; use train, dev, or test")
    return split
