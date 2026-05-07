"""
lig_utility_nlp.py - Shared utilities for IG / IDGI / Guided IG / LIG on NLP classification.

All methods produce per-token signed attributions of shape (L,) via dot product:
    attr_k[i] = grad_k[i, :] @ step_k[i, :]    (sum over embedding dim d)
which is the "d_k" quantity in the LIG paper, computed per token.

Special tokens (CLS, SEP, PAD) are held fixed at the input embedding throughout the
path (no interpolation), matching the convention used in pace_gradients for QA.
"""
from __future__ import annotations
import inspect
import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Any
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ---------------------------------------------------------------------------
# Caches
# ---------------------------------------------------------------------------
_MODEL_CACHE: Dict[str, Dict[str, Any]] = {}


def get_model_tokenizer_cls(model_name: str, device: str):
    if model_name not in _MODEL_CACHE:
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        mdl = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        mdl.eval()
        _MODEL_CACHE[model_name] = {"model": mdl, "tokenizer": tok}
    return _MODEL_CACHE[model_name]["model"], _MODEL_CACHE[model_name]["tokenizer"]


def get_helper(model_name: str):
    if "distilbert" in model_name:
        from distilbert_helper import get_inputs, nn_forward_func
    elif "roberta" in model_name:
        from roberta_helper import get_inputs, nn_forward_func
    elif "bert" in model_name:
        from bert_helper import get_inputs, nn_forward_func
    else:
        raise NotImplementedError(f"Model {model_name} not implemented")
    return get_inputs, nn_forward_func


# ---------------------------------------------------------------------------
# Baseline embedding factory (same semantics as pace_gradients)
# ---------------------------------------------------------------------------
def get_baseline_embedding(
    baseline: str,
    embed: torch.nn.Embedding,
    tokenizer,
    X: torch.Tensor,   # (1, L, d)
    device: str,
) -> torch.Tensor:
    """Return a baseline embedding of shape (1, L, d), detached."""
    L, d = X.shape[1], X.shape[2]

    if baseline == "mask":
        token_id = tokenizer.mask_token_id or tokenizer.pad_token_id
        with torch.no_grad():
            e = embed(torch.tensor([[token_id]], device=device))
        return e.expand(1, L, d).clone()
    if baseline == "pad":
        token_id = tokenizer.pad_token_id
        with torch.no_grad():
            e = embed(torch.tensor([[token_id]], device=device))
        return e.expand(1, L, d).clone()
    if baseline == "zero":
        return torch.zeros(1, L, d, device=device, dtype=X.dtype)
    if baseline == "mean":
        with torch.no_grad():
            mean_vec = embed.weight.mean(dim=0)
        return mean_vec.view(1, 1, d).expand(1, L, d).clone()
    if baseline == "random":
        vocab_size = embed.weight.shape[0]
        rid = torch.randint(0, vocab_size, (1,), device=device)
        with torch.no_grad():
            e = embed(rid.unsqueeze(0))
        return e.expand(1, L, d).clone()
    raise ValueError(f"Unknown baseline '{baseline}'")


# ---------------------------------------------------------------------------
# Tokenization / encoding
# ---------------------------------------------------------------------------
def encode_sentence(tokenizer, sentence: str, device: str):
    enc = tokenizer(sentence, return_tensors="pt", truncation=True,
                    return_special_tokens_mask=True)
    enc = {k: v.to(device) for k, v in enc.items()}
    return enc


def fixed_token_mask(tokenizer, input_ids: torch.Tensor,
                     attention_mask: torch.Tensor,
                     special_tokens_mask: torch.Tensor) -> torch.Tensor:
    """
    Boolean mask of length L: True for tokens that should be FIXED at the input
    embedding throughout the path (CLS, SEP, PAD, special tokens).
    """
    L = input_ids.shape[1]
    device = input_ids.device
    ids = input_ids[0]
    cls_id, sep_id = tokenizer.cls_token_id, tokenizer.sep_token_id
    is_special = special_tokens_mask[0].bool() if special_tokens_mask is not None \
                 else torch.zeros(L, dtype=torch.bool, device=device)
    is_pad = (attention_mask[0] == 0)
    is_cls = (ids == cls_id) if cls_id is not None else torch.zeros(L, dtype=torch.bool, device=device)
    is_sep = (ids == sep_id) if sep_id is not None else torch.zeros(L, dtype=torch.bool, device=device)
    return is_special | is_pad | is_cls | is_sep


def build_extra_kwargs(model, token_type_ids):
    fwd = inspect.signature(model.forward).parameters
    extra = {}
    if "token_type_ids" in fwd and token_type_ids is not None:
        extra["token_type_ids"] = token_type_ids
    return extra


# ---------------------------------------------------------------------------
# Path / gradient utilities
# ---------------------------------------------------------------------------
def make_straight_path(X: torch.Tensor, X_baseline: torch.Tensor,
                       fixed: torch.Tensor, N: int) -> torch.Tensor:
    """
    Build a straight-line path of N+1 points from X_baseline -> X over non-fixed
    tokens; fixed tokens stay at X for all k.

    Returns gamma of shape (N+1, L, d).
    """
    L, d = X.shape[1], X.shape[2]
    device = X.device
    t = torch.linspace(0.0, 1.0, N + 1, device=device, dtype=X.dtype)  # (N+1,)
    # coef[k, i] in [0,1] for non-fixed tokens, 1.0 for fixed tokens
    coef = t.view(N + 1, 1).expand(N + 1, L).clone()
    coef[:, fixed] = 1.0
    coef = coef.unsqueeze(-1)  # (N+1, L, 1)
    gamma = X.squeeze(0).unsqueeze(0) * coef + X_baseline.squeeze(0).unsqueeze(0) * (1 - coef)
    return gamma  # (N+1, L, d)


@torch.no_grad()
def forward_logits_batch(model, gamma: torch.Tensor, attention_mask: torch.Tensor,
                         extra_kwargs: dict, target_id: int) -> torch.Tensor:
    """Forward (N+1, L, d) -> logits at target class, shape (N+1,)."""
    K = gamma.shape[0]
    am = attention_mask.expand(K, -1)
    ek = {k: v.expand(K, -1) for k, v in extra_kwargs.items()}
    out = model(inputs_embeds=gamma, attention_mask=am, **ek)
    return out.logits[:, target_id]  # (K,)


def gradient_batch_at_points(model, points: torch.Tensor,
                             attention_mask: torch.Tensor, extra_kwargs: dict,
                             target_id: int) -> torch.Tensor:
    """
    Compute gradient of logits[target] wrt points at each of the K input rows.
    points: (K, L, d) - will be made a leaf with requires_grad.
    Returns: (K, L, d) gradient.
    """
    K = points.shape[0]
    pts = points.detach().clone().requires_grad_(True)
    am = attention_mask.expand(K, -1)
    ek = {k: v.expand(K, -1) for k, v in extra_kwargs.items()}
    out = model(inputs_embeds=pts, attention_mask=am, **ek)
    scores = out.logits[:, target_id]  # (K,)
    (grad,) = torch.autograd.grad(scores.sum(), pts)
    return grad.detach()  # (K, L, d)


# ---------------------------------------------------------------------------
# Common: predict label and target embeddings
# ---------------------------------------------------------------------------
def get_pred_and_X(model, tokenizer, sentence: str, baseline: str, device: str):
    """
    Return everything needed to run any IG variant:
      X (1,L,d), X_baseline (1,L,d), input_ids, attention_mask,
      token_type_ids, special_tokens_mask, fixed mask, pred_id, extra_kwargs.
    """
    enc = encode_sentence(tokenizer, sentence, device)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    token_type_ids = enc.get("token_type_ids", None)
    special_tokens_mask = enc.get("special_tokens_mask", torch.zeros_like(input_ids))

    extra_kwargs = build_extra_kwargs(model, token_type_ids)

    embed = model.get_input_embeddings()
    with torch.no_grad():
        X = embed(input_ids)  # (1, L, d)
        logits0 = model(inputs_embeds=X, attention_mask=attention_mask, **extra_kwargs).logits[0]
    pred_id = int(logits0.argmax().item())

    X_baseline = get_baseline_embedding(baseline, embed, tokenizer, X, device)
    fixed = fixed_token_mask(tokenizer, input_ids, attention_mask, special_tokens_mask)

    return {
        "X": X,
        "X_baseline": X_baseline,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "special_tokens_mask": special_tokens_mask,
        "fixed": fixed,
        "pred_id": pred_id,
        "extra_kwargs": extra_kwargs,
        "embed": embed,
    }


# ---------------------------------------------------------------------------
# Output filtering & packaging
# ---------------------------------------------------------------------------
def filter_special(tokens, attr_full: torch.Tensor, input_ids: torch.Tensor,
                   tokenizer, show_special: bool):
    if show_special:
        return list(tokens), attr_full
    special_ids = set(tokenizer.all_special_ids)
    keep = [i for i, tid in enumerate(input_ids[0].tolist()) if tid not in special_ids]
    return [tokens[i] for i in keep], attr_full[keep]


def pack_classification_result(tokens_full, attr_full, input_ids, tokenizer,
                               show_special, elapsed, pred_id, model,
                               nn_forward_func, X, attention_mask,
                               position_embed, type_embed):
    tokens, attr = filter_special(tokens_full, attr_full, input_ids, tokenizer, show_special)
    return {
        "tokens": tokens,
        "attributions": attr.detach().cpu(),
        "time": elapsed,
        "predicted_label": pred_id,
        "model": model,
        "nn_forward_func": nn_forward_func,
        "input_embed": X,
        "attention_mask": attention_mask,
        "position_embed": position_embed,
        "type_embed": type_embed,
        "attr_full": attr_full.detach(),
    }
