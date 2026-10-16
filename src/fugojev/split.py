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


def canonical_split(name: str) -> str:
    aliases = {"development": "dev", "val": "dev", "validation": "dev", "training": "train"}
    split = aliases.get(name, name)
    if split not in {"train", "dev", "test"}:
        raise SystemExit(f"unknown split {name!r}; use train, dev, or test")
    return split
