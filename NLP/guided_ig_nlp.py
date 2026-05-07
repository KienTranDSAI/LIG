"""
guided_ig_nlp.py - Guided IG (Kapishnikov et al., 2021) for NLP classification.

Direct port of the canonical Guided IG algorithm to embedding space, treating
each (token, embedding_dim) coordinate as a "feature". The algorithm:

  - Operates in per-feature alpha-space:  alpha_i = (x_i - x_b_i) / (x_in_i - x_b_i)
  - At outer step t, target alpha = (t+1)/N, with allowed window
        [alpha - max_dist, alpha + max_dist]
  - Features lagging behind alpha_min are pulled up to x(alpha_min).
  - An inner gamma-loop selects the bottom `fraction` quantile of |grad| features
    and advances them toward x(alpha_max), consuming exactly the L1 distance
    needed to bring d_current down to d_target.
  - Saturated features (already at x_max) are excluded by setting their |grad| to inf.

Per-feature attribution accumulates  attr += (x - x_old) * grad_actual  inside the
inner loop. To obtain per-TOKEN attributions we sum over the embedding dim (signed
dot-product aggregation).

NO rescale: completeness is enforced by construction (the path reaches x exactly).

For NLP, fixed tokens (CLS/SEP/PAD) have x_in == x_b, so x_in - x_b = 0. The
_translate_x_to_alpha function returns NaN for those coords, which we replace by
alpha_max so they're never selected as "behind" and never selected for the move
step. They naturally stay put.
"""
from __future__ import annotations
import math
import time
import torch
from typing import Dict, Any

from lig_utility_nlp import (
    get_model_tokenizer_cls, get_helper, get_pred_and_X,
    gradient_batch_at_points, forward_logits_batch,
    pack_classification_result,
)


EPSILON = 1e-9


def _translate_alpha_to_x(alpha: float, x_input: torch.Tensor,
                          x_baseline: torch.Tensor) -> torch.Tensor:
    """x(alpha) along straight line from baseline to input."""
    return x_baseline + (x_input - x_baseline) * alpha


def _translate_x_to_alpha(x: torch.Tensor, x_input: torch.Tensor,
                          x_baseline: torch.Tensor) -> torch.Tensor:
    """
    Per-feature alpha for current x in [baseline, input]. NaN where x_in == x_b
    (fixed coordinates: their alpha is undefined, treated as already-saturated).
    """
    diff = x_input - x_baseline
    out = torch.full_like(x, float("nan"))
    nz = diff != 0
    out[nz] = (x[nz] - x_baseline[nz]) / diff[nz]
    return out


def _grad_at(model, point: torch.Tensor, attention_mask, extra_kwargs, target_id):
    """Gradient at a single point (1, L, d). Returns (1, L, d)."""
    return gradient_batch_at_points(
        model, point, attention_mask, extra_kwargs, target_id
    )


def _forward_scalar_pt(model, point, attention_mask, extra_kwargs, target_id) -> float:
    with torch.no_grad():
        out = model(inputs_embeds=point, attention_mask=attention_mask, **extra_kwargs)
    return out.logits[0, target_id].item()


def guided_ig_classification(
    sentence: str,
    a: float = 0.0,
    b: float = 1.0,
    steps: int = 50,
    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    show_special_tokens: bool = False,
    baseline: str = "mask",
    fraction: float = 0.25,
    max_dist: float = 0.02,
) -> Dict[str, Any]:
    N = int(steps)
    if N <= 0:
        raise ValueError(f"N must be > 0, got {N}")
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    if not (0.0 <= max_dist <= 1.0):
        raise ValueError(f"max_dist must be in [0, 1], got {max_dist}")

    model, tokenizer = get_model_tokenizer_cls(model_name, device)
    get_inputs, nn_forward_func = get_helper(model_name)

    info = get_pred_and_X(model, tokenizer, sentence, baseline, device)
    X            = info["X"]                  # (1, L, d)
    X_baseline   = info["X_baseline"]         # (1, L, d)
    input_ids    = info["input_ids"]
    attention_mask = info["attention_mask"]
    fixed        = info["fixed"]
    pred_id      = info["pred_id"]
    extra_kwargs = info["extra_kwargs"]

    # Effective baseline: fixed tokens pinned at X (their diff = 0 -> NaN-alpha)
    x_baseline_eff = X_baseline.clone()
    x_baseline_eff[0, fixed, :] = X[0, fixed, :]

    t0 = time.perf_counter()

    x_input = X
    x_b     = x_baseline_eff
    x       = x_b.clone()                          # current path point (1, L, d)
    total_diff = x_input - x_b
    l1_total = float(total_diff.abs().sum())

    attr = torch.zeros_like(x_input)               # (1, L, d) per-feature accumulator

    if l1_total <= EPSILON:
        # Degenerate: input == baseline (or every token is fixed).
        attr_full = torch.zeros(X.shape[1], device=device)
        elapsed = time.perf_counter() - t0
        inp = get_inputs(model, tokenizer, sentence, device)
        _, _, _, _, position_embed, _, type_embed, _, _ = inp
        tokens_full = tokenizer.convert_ids_to_tokens(input_ids[0])
        return pack_classification_result(
            tokens_full, attr_full, input_ids, tokenizer, show_special_tokens,
            elapsed, pred_id, model, nn_forward_func, X, attention_mask,
            position_embed, type_embed,
        )

    for step in range(N):
        # Gradient at current x (kept untouched for accumulation)
        grad_actual = _grad_at(model, x, attention_mask, extra_kwargs, pred_id)
        grad = grad_actual.clone()

        alpha     = (step + 1.0) / N
        alpha_min = max(alpha - max_dist, 0.0)
        alpha_max = min(alpha + max_dist, 1.0)

        x_min = _translate_alpha_to_x(alpha_min, x_input, x_b)
        x_max = _translate_alpha_to_x(alpha_max, x_input, x_b)

        l1_target = l1_total * (1.0 - (step + 1.0) / N)

        gamma = float("inf")
        # Inner loop: keep moving features until l1_current ~ l1_target
        # Safety cap on inner iterations (rare edge case)
        inner_safety = 0
        while gamma > 1.0:
            inner_safety += 1
            if inner_safety > 1000:
                # Safety: shouldn't happen in normal use
                break

            x_old = x.clone()

            x_alpha = _translate_x_to_alpha(x, x_input, x_b)
            x_alpha = torch.where(torch.isnan(x_alpha),
                                  torch.full_like(x_alpha, alpha_max),
                                  x_alpha)

            # Pull-behind: features that have lagged are fast-forwarded to x_min
            behind = x_alpha < alpha_min
            x = torch.where(behind, x_min, x)

            l1_current = float((x - x_input).abs().sum())
            if math.isclose(l1_target, l1_current,
                            rel_tol=EPSILON, abs_tol=EPSILON):
                attr += (x - x_old) * grad_actual
                break

            # Saturated-at-x_max coords: exclude from selection
            grad[x == x_max] = float("inf")

            abs_grad_flat = grad.abs().reshape(-1)
            # Quantile threshold on |grad|. `interpolation="lower"` matches canonical.
            threshold = torch.quantile(abs_grad_flat, fraction,
                                       interpolation="lower")
            select = (grad.abs() <= threshold) & torch.isfinite(grad)

            # L1 budget reachable using selected features advancing to x_max
            l1_s = float(((x - x_max).abs() * select).sum())
            if l1_s > 0.0:
                gamma = (l1_current - l1_target) / l1_s
            else:
                gamma = float("inf")

            if gamma > 1.0:
                # Move selected features fully to x_max
                x = torch.where(select, x_max, x)
            else:
                if gamma <= 0.0:
                    # Numerically degenerate: stop the inner loop
                    break
                # Partial move: x <- x + gamma * (x_max - x) on selected coords
                x = torch.where(select, x + (x_max - x) * gamma, x)

            attr += (x - x_old) * grad_actual

    # Aggregate per-(token, dim) attribution to per-token (signed sum over dim)
    attr_full = attr[0].sum(dim=-1)                # (L,)

    elapsed = time.perf_counter() - t0

    inp = get_inputs(model, tokenizer, sentence, device)
    _, _, _, _, position_embed, _, type_embed, _, _ = inp
    tokens_full = tokenizer.convert_ids_to_tokens(input_ids[0])

    return pack_classification_result(
        tokens_full, attr_full, input_ids, tokenizer, show_special_tokens,
        elapsed, pred_id, model, nn_forward_func, X, attention_mask,
        position_embed, type_embed,
    )
