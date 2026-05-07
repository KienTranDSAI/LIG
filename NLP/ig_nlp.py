"""
ig_nlp.py - Standard Integrated Gradients (Sundararajan et al., 2017) for NLP classification.

Per-token signed attribution via dot product:
    attr[i] = sum_k g_{k,i} . (gamma_{k+1,i} - gamma_{k,i})  (sum over embedding dim d)
Equivalently with a straight-line path and uniform mu = 1/N over the N gradient points:
    attr[i] = (1/N) * sum_{k=0}^{N-1} g_{k,i} . Delta_x_i
The gradient is taken at the N midpoints / sample points along the path; here we follow
the canonical N-point Riemann form: g_k evaluated at gamma_k, k=0..N-1, step Delta_x/N.

Special tokens (CLS/SEP/PAD) are held fixed at the input embedding.
"""
from __future__ import annotations
import time
import torch
from typing import Dict, Any

from lig_utility_nlp import (
    get_model_tokenizer_cls, get_helper, get_pred_and_X,
    make_straight_path, gradient_batch_at_points,
    pack_classification_result,
)


def ig_classification(
    sentence: str,
    a: float = 0.0,                # kept for API compat (unused; path is [0,1])
    b: float = 1.0,
    steps: int = 100,
    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    show_special_tokens: bool = False,
    baseline: str = "mask",
) -> Dict[str, Any]:
    N = steps
    model, tokenizer = get_model_tokenizer_cls(model_name, device)
    get_inputs, nn_forward_func = get_helper(model_name)

    info = get_pred_and_X(model, tokenizer, sentence, baseline, device)
    X            = info["X"]
    X_baseline   = info["X_baseline"]
    input_ids    = info["input_ids"]
    attention_mask = info["attention_mask"]
    fixed        = info["fixed"]
    pred_id      = info["pred_id"]
    extra_kwargs = info["extra_kwargs"]

    # Build the (N+1)-point straight-line path; we only need the N gradient points
    gamma = make_straight_path(X, X_baseline, fixed, N)  # (N+1, L, d)
    gamma_pts = gamma[:N]                                # (N, L, d) - eval points
    steps_vec = gamma[1:] - gamma[:N]                    # (N, L, d) - per-step displacement

    t0 = time.perf_counter()

    grads = gradient_batch_at_points(
        model, gamma_pts, attention_mask, extra_kwargs, pred_id
    )  # (N, L, d)

    # Per-token, per-step contribution: sum over embedding dim
    # d_k[i] = grads[k, i, :] . steps_vec[k, i, :]
    per_step = (grads * steps_vec).sum(dim=-1)  # (N, L)

    # Uniform measure mu_k = 1, since steps already contain Delta_x/N (no extra 1/N needed).
    # Here gamma_{k+1}-gamma_k = Delta_x / N already, so summing per_step over k
    # gives the IG attribution directly.
    attr_full = per_step.sum(dim=0)              # (L,)

    elapsed = time.perf_counter() - t0

    # position/type embeddings for caller's metric calls
    inp = get_inputs(model, tokenizer, sentence, device)
    _, _, _, _, position_embed, _, type_embed, _, _ = inp

    tokens_full = tokenizer.convert_ids_to_tokens(input_ids[0])

    return pack_classification_result(
        tokens_full, attr_full, input_ids, tokenizer, show_special_tokens,
        elapsed, pred_id, model, nn_forward_func, X, attention_mask,
        position_embed, type_embed,
    )
