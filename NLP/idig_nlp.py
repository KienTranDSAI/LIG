"""
idig_nlp.py - IDGI (Sikdar et al., 2021) for NLP classification.

The image-domain IDGI per-feature update is:

    A_i  +=  g_{k,i}^2  *  Delta_f_k  /  ||g_k||^2

where feature index i ranges over scalar coordinates (channel, height, width)
and the resulting per-pixel attribution is signed because the per-pixel quantity
g_i^2 * d / ||g||^2 is a *projection* g_i * (g_i * d / ||g||^2) of the gradient
onto a single scalar coordinate. Aggregating to a region is just summation.

For NLP the natural "feature" is a (token, embedding_dim) pair. The per-token
analog of IDGI must therefore project the per-step output change Delta_f_k onto
the SIGNED per-token directional derivative along the displacement direction:

    s_{k, i}  :=  < g_{k, i, :},  X_i - X_baseline_i >          (scalar per token)
              =  d / d(coef_i)  f(gamma_k)

This is the same quantity PACE-Gradient differentiates against, and the LIG
paper (App. A.2) writes IDGI's per-feature update as g_i * (gamma_kp,i - gamma_k,i)
-- a signed projection, not a magnitude. Naively replacing s_{k,i} by
||g_{k, i, :}||^2 would discard the sign and force every token's attribution to
match sign(Delta_f_k) at each step, which is fatal for NLP where positive and
negative tokens must oppose each other within a single sentence.

The IDGI-NLP per-step contribution is therefore

    attr_{k, i}  =  ( s_{k, i} / sum_j s_{k, j} )  *  Delta_f_k                (signed share)

This is exactly PACE-Gradient on the straight-line path with special tokens
(CLS / SEP / PAD) pinned at coef = 1. Completeness is exact:
    sum_i attr_{k, i} = Delta_f_k    =>    sum_i attr_i = f(x) - f(x_baseline).

Implementation
--------------
We obtain s_{k, i} efficiently by introducing a per-step, per-token scalar
coefficient `coef_{k, i}` that interpolates token i along the straight line:

    X_inter[k, i, :] = coef_{k, i} * X[i, :]  +  (1 - coef_{k, i}) * X_baseline[i, :]

Differentiating the batched logits w.r.t. `coef` gives a (steps, L) gradient
tensor whose entries are exactly s_{k, i}. This avoids materialising the full
(N, L, d) gradient and matches the PACE implementation closely.

Special tokens are held at coef = 1 throughout the path (their gradient is
zeroed out of the share) so they cannot bleed signal in or out.
"""
from __future__ import annotations
import time
import inspect
import torch
from typing import Dict, Any

from lig_utility_nlp import (
    get_model_tokenizer_cls, get_helper, get_baseline_embedding,
    encode_sentence, fixed_token_mask, build_extra_kwargs,
    pack_classification_result,
)


def idig_classification(
    sentence: str,
    a: float = 0.0,
    b: float = 1.0,
    steps: int = 50,
    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    show_special_tokens: bool = False,
    baseline: str = "mask",
) -> Dict[str, Any]:
    N = int(steps)
    model, tokenizer = get_model_tokenizer_cls(model_name, device)
    get_inputs, nn_forward_func = get_helper(model_name)

    # ---- Encode and prepare ----
    enc = encode_sentence(tokenizer, sentence, device)
    input_ids           = enc["input_ids"]
    attention_mask      = enc["attention_mask"]
    token_type_ids      = enc.get("token_type_ids", None)
    special_tokens_mask = enc.get("special_tokens_mask", torch.zeros_like(input_ids))
    extra_kwargs = build_extra_kwargs(model, token_type_ids)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(device)

    embed = model.get_input_embeddings()
    with torch.no_grad():
        X = embed(input_ids)                                       # (1, L, d)
        logits0 = model(inputs_embeds=X, attention_mask=attention_mask,
                        **extra_kwargs).logits[0]
    pred_id = int(logits0.argmax().item())

    L, d = X.shape[1], X.shape[2]
    X_baseline = get_baseline_embedding(baseline, embed, tokenizer, X, device)
    fixed = fixed_token_mask(tokenizer, input_ids, attention_mask, special_tokens_mask)

    # ---- Build per-step, per-token coefficient grid ----
    # Path:  coef_{k, i} from a -> b uniformly; fixed tokens pinned at 1.
    t_vals     = torch.linspace(a, b, N, device=device, dtype=X.dtype)  # (N,)
    coefs_base = t_vals.unsqueeze(1).expand(N, L).clone()
    coefs_base[:, fixed] = 1.0
    coefs = coefs_base.detach().requires_grad_(True)                    # (N, L)

    # Interpolated embeddings (N, L, d):
    coefs_exp = coefs.unsqueeze(-1)                                     # (N, L, 1)
    X_inter   = X.squeeze(0) * coefs_exp + X_baseline.squeeze(0) * (1 - coefs_exp)

    attn_batch  = attention_mask.expand(N, -1)
    extra_batch = {k: v.expand(N, -1) for k, v in extra_kwargs.items()}

    t0 = time.perf_counter()

    # ---- Single batched forward + backward ----
    out          = model(inputs_embeds=X_inter, attention_mask=attn_batch, **extra_batch)
    logits_batch = out.logits[:, pred_id]                                # (N,)

    # Per-step output change Delta_f_k.
    # Use central-difference style:  d_k = f(gamma_k) - f(gamma_{k-1});  d_0 = 0.
    # This matches PACE / canonical IDGI on the straight-line path.
    delta    = logits_batch - torch.cat([logits_batch[:1], logits_batch[:-1]])
    delta    = delta.clone()
    delta[0] = 0.0

    # s_{k, i} = d/d(coef_{k, i}) f(gamma_k)
    (grad_coefs,) = torch.autograd.grad(logits_batch.sum(), coefs)       # (N, L)

    elapsed = time.perf_counter() - t0

    # ---- Signed share aggregation (the PACE / NLP-IDGI rule) ----
    # share_{k, i} = s_{k, i} / sum_j s_{k, j}      (signed; can be negative)
    # attr_{k, i}  = share_{k, i} * Delta_f_k
    #   sum_i attr_{k, i} = Delta_f_k  (completeness per step)
    denom = grad_coefs.sum(dim=1, keepdim=True)                          # (N, 1)
    # Numerical safety: in steps where denom is tiny, the per-step contribution
    # would explode; skip them (they also have Delta_f_k ~ 0 in practice).
    safe_mask  = denom.abs() > 1e-10
    safe_denom = torch.where(safe_mask, denom, torch.ones_like(denom))
    grad_share = grad_coefs / safe_denom                                 # (N, L)
    grad_share = grad_share * safe_mask.to(grad_share.dtype)

    attr_per_step = grad_share * delta.unsqueeze(1)                      # (N, L)
    attr_full     = attr_per_step.sum(dim=0)                             # (L,)

    # ---- Position / type embeddings for downstream metrics ----
    inp = get_inputs(model, tokenizer, sentence, device)
    _, _, _, _, position_embed, _, type_embed, _, _ = inp

    tokens_full = tokenizer.convert_ids_to_tokens(input_ids[0])

    return pack_classification_result(
        tokens_full, attr_full, input_ids, tokenizer, show_special_tokens,
        elapsed, pred_id, model, nn_forward_func, X, attention_mask,
        position_embed, type_embed,
    )