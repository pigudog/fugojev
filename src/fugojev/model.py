"""Pointer-head decision model over a frozen causal LM with a LoRA adapter.

Adapted from MedJev / kev (Apache-2.0); see NOTICE. Two forward forms, picked by
the backbone:

* **packed** (attention-only, e.g. Qwen2.5): the state and every question branch
  sit in one sequence under a block-causal mask, so a token reads the state and
  its own question, never a sibling question.
* **rows** (hybrid, e.g. Qwen3.5 with Gated DeltaNet layers): the recurrent layers
  ignore attention masks, so each question becomes its own causal row with the
  state repeated. Isolation is exact by construction.

For serving, the state is encoded once and the branches read its cache
(`probs_and_prefix`), so answering every question of a note costs about one pass.

State tokens keep the TAIL, because the state builder already put each report's
conclusion at the end.
"""

import copy
import math
import re
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Rare Qwen specials reused as delimiters (state, question, option, end-of-option,
# decide), so no embedding rows are added. The first group that the tokenizer
# round-trips is used; Qwen2.5 and Qwen3.5 both carry the first group.
SPECIAL_SETS = [
    ["<|object_ref_start|>", "<|object_ref_end|>", "<|box_start|>", "<|box_end|>", "<|quad_start|>"],
    ["<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"],
]
SPECIAL = SPECIAL_SETS[0]
MAX_STATE, MAX_BRANCH = 4096, 4864
BRANCH_SLACK = 768
DEFAULT_BACKBONE = "Qwen/Qwen2.5-0.5B"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
# Gated DeltaNet projections (transformers 5 names), added on hybrid backbones.
HYBRID_LORA_TARGETS = ["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"]

_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


def load_tokenizer(name):
    return AutoTokenizer.from_pretrained(name)


def _has_token(tok, token) -> bool:
    token_id = tok.convert_tokens_to_ids(token)
    if not isinstance(token_id, int) or token_id < 0:
        return False
    unk = getattr(tok, "unk_token_id", None)
    if unk is not None and token_id == unk:
        return False
    return tok.convert_ids_to_tokens(token_id) == token


def resolve_special(tok):
    """The first delimiter group this tokenizer carries in full."""
    for group in SPECIAL_SETS:
        if all(_has_token(tok, token) for token in group):
            return list(group)
    raise ValueError(f"tokenizer carries none of the delimiter groups {SPECIAL_SETS}")


def user_tokens(tok, text):
    """Tokenize note text so it cannot emit a delimiter token."""
    safe = _SPECIAL_RE.sub(r"<¦\1¦>", text or "")
    return tok(safe, add_special_tokens=False).input_ids


def take_state_tail(token_ids, max_state):
    """Keep the last tokens. One slot is reserved for the state-start delimiter."""
    room = max_state - 1
    if room < 1:
        raise ValueError("max_state must be at least 2")
    if len(token_ids) <= room:
        return list(token_ids), False
    return list(token_ids[-room:]), True


def encode(tok, rec, max_state=MAX_STATE, max_branch=MAX_BRANCH, special=None):
    """Pack `[state]` and, per question, `[instruction option ... decide]`.

    `max_branch` is the state plus one question, not the concatenated record.
    Branch positions continue from the state length, so a row made of
    `state + branch_k` carries exactly the positions the packed form gives it.
    """
    special = list(special) if special else SPECIAL
    state_tokens, truncated = take_state_tail(user_tokens(tok, rec["state"]), max_state)
    start, q_id, o_id, c_id, d_id = (tok.convert_tokens_to_ids(token) for token in special)
    ids = [start, *state_tokens]
    seg = [0] * len(ids)
    pos = list(range(len(ids)))
    state_len = len(ids)
    decide_idx, opt_idx = [], []
    labels, qtypes, qids, keys = [], [], [], []
    for index, question in enumerate(rec["questions"], start=1):
        instruction = [q_id, *user_tokens(tok, question["instr"])]
        spans = [[o_id, *user_tokens(tok, option), c_id] for option in question["options"]]
        branch = instruction + [token for span in spans for token in span] + [d_id]
        if len(branch) > max_branch - state_len:
            raise ValueError(f"branch too long: {len(branch)} tokens, state is {state_len}")
        base = len(ids)
        ids.extend(branch)
        seg.extend([index] * len(branch))
        pos.extend(range(state_len, state_len + len(branch)))
        cursor = len(instruction)
        ends = []
        for span in spans:
            cursor += len(span)
            ends.append(cursor - 1)
        decide_idx.append(base + len(branch) - 1)
        opt_idx.append([base + end for end in ends])
        labels.append(question["label"])
        qtypes.append(question["qtype"])
        qids.append(question["qid"])
        keys.append(list(question["keys"]))
    return {
        "ids": ids,
        "seg": seg,
        "pos": pos,
        "decide_idx": decide_idx,
        "opt_idx": opt_idx,
        "labels": labels,
        "qtypes": qtypes,
        "qids": qids,
        "keys": keys,
        "state_truncated": truncated,
    }


def rows_of(enc):
    """Split a packed encoding into the state and one row per question.

    Returns (state_ids, state_pos, rows); rows[k] has the branch `ids`, their
    state-continuing `pos`, and the `decide` / `opts` offsets inside the branch.
    Running `state + rows[k]` as one causal row equals the packed block-causal
    form for question k on any architecture.
    """
    seg = enc["seg"]
    n_state = seg.count(0)
    rows = []
    start = n_state
    for index, (decide, options) in enumerate(zip(enc["decide_idx"], enc["opt_idx"]), start=1):
        end = decide + 1
        if seg[start] != index or seg[end - 1] != index:
            raise ValueError("branch layout mismatch")
        rows.append({
            "ids": enc["ids"][start:end],
            "pos": enc["pos"][start:end],
            "decide": decide - start,
            "opts": [option - start for option in options],
        })
        start = end
    return enc["ids"][:n_state], enc["pos"][:n_state], rows


def branch_mask_batch(segs, device, dtype=torch.float32):
    """Additive mask [B, 1, L, L]. A token sees itself, earlier tokens in its
    question, and the state. It does not see another question."""
    length = max(len(seg) for seg in segs)
    segments = torch.full((len(segs), length), -1, device=device)
    for batch, seg in enumerate(segs):
        segments[batch, :len(seg)] = torch.tensor(seg, device=device)
    causal = torch.tril(torch.ones(length, length, dtype=torch.bool, device=device))
    same = (segments[:, None, :] == segments[:, :, None]) | (segments[:, None, :] == 0)
    valid_key = (segments != -1)[:, None, :]
    allow = causal[None] & same & valid_key
    allow = allow | torch.eye(length, dtype=torch.bool, device=device)[None]
    blocked = torch.finfo(dtype).min
    return torch.zeros(len(segs), length, length, dtype=dtype, device=device).masked_fill(~allow, blocked)[:, None]


def _new_cache(config):
    from transformers import DynamicCache
    try:
        return DynamicCache(config=config)
    except TypeError:
        return DynamicCache()


class PointerHead(nn.Module):
    """Score each option's end hidden state against the question's decide state."""

    def __init__(self, hidden_size, head_dim=256):
        super().__init__()
        self.q = nn.Linear(hidden_size, head_dim)
        self.k = nn.Linear(hidden_size, head_dim)
        self.scale = 1 / math.sqrt(head_dim)

    def forward(self, hidden_decide, hidden_options):
        return (self.k(hidden_options) @ self.q(hidden_decide)) * self.scale


def _use_torch_linear_attention(lm):
    """On CPU, transformers still dispatches the Gated DeltaNet to the Triton kernels
    when `flash-linear-attention` is importable, which fails on CPU tensors. Swap the
    decorated functions in the backbone's modeling module back to their PyTorch
    reference implementations (kept by functools.wraps as `__wrapped__`)."""
    module = sys.modules.get(type(lm).__module__)
    if module is None:
        return
    for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule",
                 "causal_conv1d_fn", "causal_conv1d_update"):
        fn = getattr(module, name, None)
        while fn is not None and hasattr(fn, "__wrapped__"):
            fn = fn.__wrapped__
        if fn is not None:
            setattr(module, name, fn)


class DecisionModel(nn.Module):
    def __init__(self, name, tok, device, lora=0, head_dim=256, dtype=torch.float32,
                 max_state=MAX_STATE, max_branch=MAX_BRANCH, special=None, backbone=None):
        super().__init__()
        self.base_name = name
        self.lora_rank = lora
        self.head_dim = head_dim
        self.max_state = max_state
        self.max_branch = max_branch
        self.device = device
        self.special = list(special) if special else resolve_special(tok)
        for token in self.special:
            if not _has_token(tok, token):
                raise ValueError(f"{name}: tokenizer is missing delimiter token {token}")
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
        if backbone is not None:
            self.lm = backbone
        else:
            # backbone only, no vocabulary head: nothing is ever generated
            self.lm = AutoModelForCausalLM.from_pretrained(
                name, dtype=dtype, attn_implementation="sdpa").model
        layer_types = set(getattr(self.lm.config, "layer_types", None) or [])
        self.hybrid = "linear_attention" in layer_types
        self.lora_targets = LORA_TARGETS + (HYBRID_LORA_TARGETS if self.hybrid else [])
        if self.hybrid and torch.device(device).type == "cpu":
            _use_torch_linear_attention(self.lm)
        hidden_size = self.lm.config.hidden_size
        if lora:
            from peft import LoraConfig, get_peft_model
            config = LoraConfig(
                task_type="FEATURE_EXTRACTION",
                r=lora,
                lora_alpha=2 * lora,
                lora_dropout=0.05,
                target_modules=self.lora_targets,
            )
            self.lm = get_peft_model(self.lm, config)
        else:
            for parameter in self.lm.parameters():
                parameter.requires_grad = False
        self.head = PointerHead(hidden_size, head_dim)
        self.to(device)

    # --- encoding -------------------------------------------------------------

    def encode(self, tok, rec, **kwargs):
        kwargs.setdefault("max_state", self.max_state)
        kwargs.setdefault("max_branch", self.max_branch)
        kwargs.setdefault("special", self.special)
        return encode(tok, rec, **kwargs)

    # --- packed form (attention backbones) ---------------------------------------

    def hidden(self, enc):
        ids = torch.tensor([enc["ids"]], device=self.device)
        pos = torch.tensor([enc["pos"]], device=self.device)
        dtype = next(self.lm.parameters()).dtype
        mask = branch_mask_batch([enc["seg"]], self.device, dtype)
        output = self.lm(input_ids=ids, position_ids=pos, attention_mask=mask, use_cache=False)
        return output.last_hidden_state[0].float()

    def _readout(self, hidden, enc):
        logits = []
        for decide, options in zip(enc["decide_idx"], enc["opt_idx"]):
            option_index = torch.tensor(options, device=self.device)
            logits.append(self.head(hidden[decide], hidden[option_index]))
        return logits

    # --- row form (hybrid backbones) ------------------------------------------------

    def forward_rows_batch(self, encs):
        """Every question of every record is one causal row: state tokens then its
        branch, right-padded into a single batch. Returns per-record lists of logits."""
        rows, owners = [], []
        for batch, enc in enumerate(encs):
            state_ids, state_pos, branches = rows_of(enc)
            for row in branches:
                rows.append((
                    state_ids + row["ids"],
                    state_pos + row["pos"],
                    len(state_ids) + row["decide"],
                    [len(state_ids) + option for option in row["opts"]],
                ))
                owners.append(batch)
        length = max(len(ids) for ids, *_ in rows)
        ids = torch.full((len(rows), length), self.pad_id, device=self.device)
        pos = torch.zeros((len(rows), length), dtype=torch.long, device=self.device)
        attention = torch.zeros((len(rows), length), dtype=torch.long, device=self.device)
        for index, (row_ids, row_pos, _, _) in enumerate(rows):
            ids[index, :len(row_ids)] = torch.tensor(row_ids, device=self.device)
            pos[index, :len(row_pos)] = torch.tensor(row_pos, device=self.device)
            attention[index, :len(row_ids)] = 1
        hidden = self.lm(input_ids=ids, position_ids=pos, attention_mask=attention,
                         use_cache=False).last_hidden_state.float()
        out = [[] for _ in encs]
        for index, (owner, (_, _, decide, options)) in enumerate(zip(owners, rows)):
            option_index = torch.tensor(options, device=self.device)
            out[owner].append(self.head(hidden[index, decide], hidden[index, option_index]))
        return out

    # --- common entry points ----------------------------------------------------------

    def forward(self, enc):
        """List of logits tensors, one per question."""
        if self.hybrid:
            return self.forward_rows_batch([enc])[0]
        return self._readout(self.hidden(enc), enc)

    @torch.no_grad()
    def probs(self, enc):
        return [F.softmax(logits, -1).cpu() for logits in self.forward(enc)]

    # --- state-prefix reuse (serving) ------------------------------------------------
    # Exact by construction: branch tokens never attend across questions, and the
    # state never sees the branches, so the state's hidden states and cache are the
    # same with or without the branches present.

    @torch.no_grad()
    def prefix(self, enc):
        """Run the state tokens only -> (n_state, cache, state hidden states)."""
        n_state = enc["seg"].count(0)
        ids = torch.tensor([enc["ids"][:n_state]], device=self.device)
        pos = torch.tensor([enc["pos"][:n_state]], device=self.device)
        out = self.lm(input_ids=ids, position_ids=pos, past_key_values=_new_cache(self.lm.config),
                      use_cache=True)
        return n_state, out.past_key_values, out.last_hidden_state[0].float()

    def _branch_rows_from_prefix(self, enc, cache):
        """Hybrid serving: replicate the cached state once per question and run the
        branches as causal rows. The cache passed in is consumed."""
        state_ids, _, branches = rows_of(enc)
        n_q = len(branches)
        cache.reorder_cache(torch.zeros(n_q, dtype=torch.long, device=self.device))
        width = max(len(row["ids"]) for row in branches)
        ids = torch.full((n_q, width), self.pad_id, device=self.device)
        pos = torch.zeros((n_q, width), dtype=torch.long, device=self.device)
        attention = torch.zeros((n_q, len(state_ids) + width), dtype=torch.long, device=self.device)
        for index, row in enumerate(branches):
            ids[index, :len(row["ids"])] = torch.tensor(row["ids"], device=self.device)
            pos[index, :len(row["pos"])] = torch.tensor(row["pos"], device=self.device)
            attention[index, :len(state_ids) + len(row["ids"])] = 1
        hidden = self.lm(input_ids=ids, position_ids=pos, attention_mask=attention,
                         past_key_values=cache, use_cache=True).last_hidden_state.float()
        out = []
        for index, row in enumerate(branches):
            option_index = torch.tensor(row["opts"], device=self.device)
            out.append(F.softmax(self.head(hidden[index, row["decide"]], hidden[index, option_index]), -1).cpu())
        return out

    @torch.no_grad()
    def probs_and_prefix(self, enc):
        """One pass that also returns the reusable state prefix. This is the serving
        and evaluation path: the state is encoded once per record."""
        n_state = enc["seg"].count(0)
        if self.hybrid:
            n_state, cache, h_state = self.prefix(enc)
            return self._branch_rows_from_prefix(enc, copy.deepcopy(cache)), (n_state, cache, h_state)
        ids = torch.tensor([enc["ids"]], device=self.device)
        pos = torch.tensor([enc["pos"]], device=self.device)
        dtype = next(self.lm.parameters()).dtype
        mask = branch_mask_batch([enc["seg"]], self.device, dtype)
        out = self.lm(input_ids=ids, position_ids=pos, attention_mask=mask,
                      past_key_values=_new_cache(self.lm.config), use_cache=True)
        hidden = out.last_hidden_state[0].float()
        # negative crop removes that many trailing tokens on every transformers we support
        out.past_key_values.crop(-(len(enc["ids"]) - n_state))
        probs = [F.softmax(logits, -1).cpu() for logits in self._readout(hidden, enc)]
        return probs, (n_state, out.past_key_values, hidden[:n_state].clone())

    @torch.no_grad()
    def probs_with_prefix(self, enc, prefix):
        """`probs()` for a record whose state tokens equal the cached prefix's; only
        the branches run. The stored prefix is left reusable."""
        n_state, cache, h_state = prefix
        if enc["seg"].count(0) != n_state:
            raise ValueError("prefix does not match this record's state")
        if self.hybrid:
            return self._branch_rows_from_prefix(enc, copy.deepcopy(cache))
        ids = torch.tensor([enc["ids"][n_state:]], device=self.device)
        pos = torch.tensor([enc["pos"][n_state:]], device=self.device)
        dtype = next(self.lm.parameters()).dtype
        mask = branch_mask_batch([enc["seg"]], self.device, dtype)[:, :, n_state:, :]
        try:
            out = self.lm(input_ids=ids, position_ids=pos, past_key_values=cache,
                          attention_mask=mask, use_cache=True)
            hidden = torch.cat([h_state, out.last_hidden_state[0].float()], 0)
        finally:
            cache.crop(-(len(enc["ids"]) - n_state))
        return [F.softmax(logits, -1).cpu() for logits in self._readout(hidden, enc)]

    def trainable_parameters(self):
        return [parameter for parameter in self.parameters() if parameter.requires_grad]
