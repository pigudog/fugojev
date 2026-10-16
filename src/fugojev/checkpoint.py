"""Save and load a pointer head plus, when LoRA was trained, its adapter.

The base weights stay in the original backbone. A run stores `head.pt` (head
weights, base id, delimiter group, rank, LoRA targets, token budgets), the
tokenizer, and the peft adapter.
"""

import os
from pathlib import Path

import torch

from fugojev.model import MAX_BRANCH, MAX_STATE, DecisionModel, load_tokenizer


def save_checkpoint(model, tokenizer, out, extra=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    base = model.base_name
    if os.path.isdir(base):
        base = os.path.abspath(base)
    torch.save({
        "head": model.head.state_dict(),
        "base": base,
        "lora": model.lora_rank,
        "lora_targets": list(model.lora_targets),
        "head_dim": model.head_dim,
        "max_state": model.max_state,
        "max_branch": model.max_branch,
        "special": list(model.special),
        "hybrid": bool(model.hybrid),
        "weights_dtype": "fp32",
        **(extra or {}),
    }, out / "head.pt")
    tokenizer.save_pretrained(out)
    if model.lora_rank:
        model.lm.save_pretrained(out)


def load_checkpoint(run, device, dtype=torch.float32, base=None):
    """Rebuild the model from `head.pt`. The adapter is not merged; the backbone is
    cast to `dtype` after the adapter is attached. `base` overrides the backbone
    path recorded in the checkpoint (e.g. a local copy of the same hub model)."""
    run = Path(run)
    meta = torch.load(run / "head.pt", map_location="cpu", weights_only=False)
    base = base or meta["base"]
    tokenizer = load_tokenizer(str(run) if (run / "tokenizer_config.json").exists() else base)
    model = DecisionModel(
        base, tokenizer, device, lora=0, head_dim=meta.get("head_dim", 256),
        dtype=torch.float32,
        max_state=meta.get("max_state") or MAX_STATE,
        max_branch=meta.get("max_branch") or MAX_BRANCH,
        special=meta.get("special"),
    )
    if meta.get("hybrid") is not None and bool(meta["hybrid"]) != model.hybrid:
        raise RuntimeError(f"{run}: checkpoint was trained on hybrid={meta['hybrid']} but the base is hybrid={model.hybrid}")
    if meta.get("lora"):
        from peft import PeftModel
        model.lm = PeftModel.from_pretrained(model.lm, str(run)).to(device)
        model.lora_rank = meta["lora"]
        model.lora_targets = meta.get("lora_targets", model.lora_targets)
    if dtype != torch.float32:
        model.lm = model.lm.to(dtype)
    model.head.load_state_dict(meta["head"])
    model.eval()
    return tokenizer, model
