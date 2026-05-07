"""
lig_nlp.py - Least-action Integrated Gradients (LIG) for NLP classification.

Joint optimisation of (gamma, mu) under the discrete signal-harvesting objective
from the LIG paper (Eq. 16). NLP-specific design choices below.

Discrete objective
------------------
    min   Var_nu(phi)  -  lam * sum_k  mu_k  |d_k|
                       +  (tau / 2) ||mu||^2_2
    gamma in Gamma_N,  mu in Simplex_N

where, with the per-token, per-step SIGNED dot-product aggregation used across
this codebase (matching the reference image-domain LIG):

    s_{k, i}    = < grad_k[i, :],  X_i - X_baseline_i >          (token-i directional
                                                                  derivative at gamma_k)
    d_{k, i}    = (c_{k+1, i} - c_{k, i}) * s_{k, i}              (per-token, per-step
                                                                  signed contribution)
    d_k         = sum_i d_{k, i}                                  (per-step total)
    Delta_f_k   = f(gamma_{k+1}) - f(gamma_k)                     (pre-softmax logit change)
    phi_k       = d_k / Delta_f_k                                  (step fidelity)
    nu_k        = mu_k Delta_f_k^2 / sum_j mu_j Delta_f_j^2

The signal-harvesting term uses |d_k| as per the paper (Eq. 16). Per-token
attributions remain signed via s_{k, i} (which carries sign), so individual
tokens (e.g. negation words) can still receive negative attribution; |d_k|
only governs how the measure concentrates across steps.

Path parameterisation
---------------------
We parameterise the path by per-token cumulative fractions
    c[k, i] in [0, 1],   c[0, i] = 0,  c[N, i] = 1,  c[:, fixed] = 1,
monotone non-decreasing in k. Then
    gamma[k, i, :]  =  X_baseline[i, :]  +  c[k, i] * (X[i, :] - X_baseline[i, :]).

Path init options
-----------------
init_path = "uniform"   -> straight-line path c[k, i] = k/N (default)
init_path = "guided_ig" -> warm-start c from a Guided IG run on the same input;
                           Guided IG moves each (token, dim) along the same line
                           but at different rates, so we recover per-token
                           cumulative fractions by averaging the projected fraction
                           across the embedding dim of each token, then project
                           onto the monotone-in-k feasible set.

Optimisation strategy
---------------------
Alternating minimisation, two passes by default. See module docstring of the
previous version for the per-phase derivation; only the signal term has changed
(|d_k| in place of d_k * sign(Delta_f_k)) and gradients reflect that.

Final attribution
-----------------
With best (gamma, mu) state:
    attr_i  =  sum_k  mu_k  *  (c_{k+1, i} - c_{k, i})  *  s_{k, i}
            =  sum_k  mu_k  *  d_{k, i}
Then rescale to enforce sum_i attr_i = f(X) - f(X_baseline_eff).

Debug
-----
Pass debug=True to lig_classification to print per-iteration optimisation
diagnostics: objective decomposition, mu sparsity, path movement, completeness,
and final per-token attribution top-k.
"""
from __future__ import annotations
import time
import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple

from lig_utility_nlp import (
    get_model_tokenizer_cls, get_helper, get_pred_and_X,
    pack_classification_result,
)


# =============================================================================
# Debug helpers
# =============================================================================
def _dbg(enabled: bool, *args, **kwargs):
    """Print only if debug enabled. Compact tag prefix."""
    if enabled:
        print(*args, **kwargs, flush=True)


def _summarise_vec(name: str, v: torch.Tensor, k_top: int = 5) -> str:
    """One-line summary of a 1-D tensor: min/max/mean/std + top-k indices."""
    v = v.detach()
    if v.numel() == 0:
        return f"{name}: <empty>"
    top_vals, top_idx = v.abs().topk(min(k_top, v.numel()))
    top_signed = v[top_idx]
    top_str = ", ".join(f"{int(i)}:{float(s):+.3f}"
                        for i, s in zip(top_idx, top_signed))
    return (f"{name}: n={v.numel()} "
            f"min={float(v.min()):+.4f} max={float(v.max()):+.4f} "
            f"mean={float(v.mean()):+.4f} std={float(v.std()):.4f} | "
            f"top-{len(top_idx)}: [{top_str}]")


def _mu_sparsity(mu: torch.Tensor) -> Tuple[float, int, float]:
    """Effective support size of mu via entropy; also # of >1/(2N) entries and max."""
    mu = mu.detach()
    eps = 1e-12
    p = mu.clamp_min(eps)
    p = p / p.sum()
    H = -(p * p.log()).sum()
    eff = float(H.exp())
    N = mu.numel()
    nz = int((mu > 0.5 / N).sum())
    return eff, nz, float(mu.max())


# =============================================================================
# Path utilities
# =============================================================================
def _path_from_c(c: torch.Tensor, X: torch.Tensor, X_baseline: torch.Tensor) -> torch.Tensor:
    """
    Build gamma of shape (N+1, L, d) from cumulative fractions c of shape (N+1, L).
    gamma[k, i, :] = X_b[i, :] + c[k, i] * (X[i, :] - X_b[i, :]).
    """
    return X_baseline.squeeze(0).unsqueeze(0) + c.unsqueeze(-1) * (X - X_baseline).squeeze(0).unsqueeze(0)


def _project_c(c: torch.Tensor, fixed: torch.Tensor) -> torch.Tensor:
    """
    Project c (N+1, L) onto the feasible set:
      - c[0, :] = 0 (movable tokens), c[N, :] = 1
      - fixed tokens: c[:, fixed] = 1 throughout
      - monotone non-decreasing along k for movable tokens, values in [0, 1]
    """
    Np1, L = c.shape
    c = c.clamp(0.0, 1.0)
    c, _ = torch.cummax(c, dim=0)
    c[0, :] = 0.0
    c[Np1 - 1, :] = 1.0
    c[:, fixed] = 1.0
    return c


def _init_c_uniform(N: int, L: int, fixed: torch.Tensor, device, dtype) -> torch.Tensor:
    """Straight-line cumulative fractions: c[k, i] = k/N for movable, 1 for fixed."""
    t = torch.linspace(0.0, 1.0, N + 1, device=device, dtype=dtype)
    c = t.view(N + 1, 1).expand(N + 1, L).clone()
    c[:, fixed] = 1.0
    return c


def _init_c_from_guided_ig(
    sentence: str,
    N: int,
    fixed: torch.Tensor,
    X: torch.Tensor,
    X_baseline_eff: torch.Tensor,
    model_name: str,
    device: str,
    baseline: str,
    guided_ig_kwargs: Optional[dict] = None,
) -> torch.Tensor:
    """
    Run Guided IG on the same input and convert its discovered path to per-token
    cumulative fractions c of shape (N+1, L).

    Mechanism. Guided IG advances each (token, embedding-dim) coordinate along
    the line from X_baseline to X but at desynchronised rates. For the LIG path
    parameterisation we need ONE fraction per (step, token), so we project each
    token's path point onto its displacement direction and average across the
    embedding dim (equivalent to a least-squares scalar fit, since all
    coordinates lie on the same line).

    The result is then projected onto LIG's monotone feasible set (cummax + clip)
    so it satisfies c[0]=0, c[N]=1, c[:, fixed]=1, and is non-decreasing in k.
    """
    # Lazy import to avoid circular dependency at module load
    from guided_ig_nlp import guided_ig_classification

    # Run Guided IG with N steps so its trajectory has N+1 points (k=0..N)
    # We need to instrument it to return the path; the public function doesn't,
    # so we re-create the path-construction loop here using its building blocks.
    kwargs = guided_ig_kwargs or {}
    fraction = float(kwargs.get("fraction", 0.25))
    max_dist = float(kwargs.get("max_dist", 0.02))

    # Run a private path-only variant of Guided IG. We import its helpers and
    # mirror its loop, recording x at each outer step.
    import math
    from guided_ig_nlp import (
        _translate_alpha_to_x, _translate_x_to_alpha, _grad_at, EPSILON,
    )
    from lig_utility_nlp import get_pred_and_X

    model, tokenizer = get_model_tokenizer_cls(model_name, device)
    info = get_pred_and_X(model, tokenizer, sentence, baseline, device)
    input_ids      = info["input_ids"]
    attention_mask = info["attention_mask"]
    extra_kwargs   = info["extra_kwargs"]
    pred_id        = info["pred_id"]

    x_input = X
    x_b     = X_baseline_eff
    x       = x_b.clone()                                     # (1, L, d)
    total_diff = x_input - x_b
    l1_total = float(total_diff.abs().sum())

    L = X.shape[1]
    dtype = X.dtype

    if l1_total <= EPSILON:
        # Degenerate: input == baseline. Fall back to uniform c.
        return _init_c_uniform(N, L, fixed, device, dtype)

    # We record x at the END of each outer step => N points after k=0,
    # giving the full N+1 trajectory.
    traj = [x.clone()]

    for step in range(N):
        grad_actual = _grad_at(model, x, attention_mask, extra_kwargs, pred_id)
        grad = grad_actual.clone()

        alpha     = (step + 1.0) / N
        alpha_min = max(alpha - max_dist, 0.0)
        alpha_max = min(alpha + max_dist, 1.0)
        x_min = _translate_alpha_to_x(alpha_min, x_input, x_b)
        x_max = _translate_alpha_to_x(alpha_max, x_input, x_b)
        l1_target = l1_total * (1.0 - (step + 1.0) / N)

        gamma = float("inf")
        inner_safety = 0
        while gamma > 1.0:
            inner_safety += 1
            if inner_safety > 1000:
                break

            x_alpha = _translate_x_to_alpha(x, x_input, x_b)
            x_alpha = torch.where(torch.isnan(x_alpha),
                                  torch.full_like(x_alpha, alpha_max),
                                  x_alpha)
            behind = x_alpha < alpha_min
            x = torch.where(behind, x_min, x)

            l1_current = float((x - x_input).abs().sum())
            if math.isclose(l1_target, l1_current,
                            rel_tol=EPSILON, abs_tol=EPSILON):
                break

            grad[x == x_max] = float("inf")
            abs_grad_flat = grad.abs().reshape(-1)
            threshold = torch.quantile(abs_grad_flat, fraction,
                                       interpolation="lower")
            select = (grad.abs() <= threshold) & torch.isfinite(grad)

            l1_s = float(((x - x_max).abs() * select).sum())
            if l1_s > 0.0:
                gamma = (l1_current - l1_target) / l1_s
            else:
                gamma = float("inf")

            if gamma > 1.0:
                x = torch.where(select, x_max, x)
            else:
                if gamma <= 0.0:
                    break
                x = torch.where(select, x + (x_max - x) * gamma, x)

        traj.append(x.clone())

    # traj is a list of N+1 tensors of shape (1, L, d).
    # Convert to per-token cumulative fractions c[k, i].
    # Each (token i, dim c) coordinate sits on the line from X_b to X, so
    #     gamma[k, i, c] = X_b[i, c] + frac[k, i, c] * (X[i, c] - X_b[i, c]).
    # The Guided IG construction makes frac[k, i, c] consistent across c for a
    # given token only approximately (different dims advance at different times
    # within a token). We average across c to get a single per-token fraction.
    #
    # Numerically robust formula (handles dims where displacement ~ 0):
    #     c[k, i] = sum_c frac_num[k, i, c] / sum_c |X[i, c] - X_b[i, c]|^2
    # where frac_num[k, i, c] = (gamma[k,i,c] - X_b[i,c]) * (X[i,c] - X_b[i,c]).
    # This is a least-squares projection of gamma[k, i, :] onto the line
    # X_b[i, :] + t * (X[i, :] - X_b[i, :]).
    Np1 = N + 1
    gamma_stack = torch.cat(traj, dim=0)                              # (N+1, L, d)
    diff = (X - X_baseline_eff).squeeze(0)                            # (L, d)
    diff_sq = (diff * diff).sum(dim=-1)                               # (L,)
    # numerator = sum_c (gamma_kic - Xb_ic) * (X_ic - Xb_ic)
    num = ((gamma_stack - X_baseline_eff.squeeze(0).unsqueeze(0)) *
           diff.unsqueeze(0)).sum(dim=-1)                             # (N+1, L)

    # Avoid divide-by-zero on tokens whose displacement is zero (fixed tokens).
    safe_diff_sq = torch.where(diff_sq > 1e-12, diff_sq,
                               torch.ones_like(diff_sq))
    c = num / safe_diff_sq.unsqueeze(0)                               # (N+1, L)

    # Project to feasible set: clip, monotonise, re-impose boundaries / fixed.
    c = _project_c(c, fixed)
    return c


# =============================================================================
# Path evaluation: s_{k,i}, Delta_f_k via single batched coef-gradient pass
# =============================================================================
def _evaluate_path(model, c: torch.Tensor, X: torch.Tensor, X_baseline: torch.Tensor,
                   attention_mask, extra_kwargs, target_id: int):
    """
    Compute s_{k,i} = d/d(coef_{k,i}) f(gamma_k) at the N gradient points
    gamma_0..gamma_{N-1}, plus Delta_f_k = f(gamma_{k+1}) - f(gamma_k) at all N steps.

    Returns:
        s     : (N, L)   per-token directional derivative at each k
        df    : (N,)     pre-softmax logit change Delta_f_k
        d_per : (N, L)   per-token, per-step signed contribution d_{k,i}
        d_k   : (N,)     per-step total d_k = sum_i d_{k,i}
    """
    Np1, L = c.shape
    N = Np1 - 1

    coefs = c[:N].detach().clone().requires_grad_(True)               # (N, L)
    X_inter = X_baseline.squeeze(0) + coefs.unsqueeze(-1) * (X - X_baseline).squeeze(0)

    with torch.no_grad():
        c_full = c.detach()
        gamma_full = _path_from_c(c_full, X, X_baseline)              # (N+1, L, d)
        am_full = attention_mask.expand(Np1, -1)
        ek_full = {k: v.expand(Np1, -1) for k, v in extra_kwargs.items()}
        f_full = model(inputs_embeds=gamma_full, attention_mask=am_full,
                       **ek_full).logits[:, target_id]                # (N+1,)
    df = f_full[1:] - f_full[:N]                                       # (N,)

    am_grad = attention_mask.expand(N, -1)
    ek_grad = {k: v.expand(N, -1) for k, v in extra_kwargs.items()}
    out = model(inputs_embeds=X_inter, attention_mask=am_grad, **ek_grad)
    logits_batch = out.logits[:, target_id]                            # (N,)
    (s,) = torch.autograd.grad(logits_batch.sum(), coefs)              # (N, L)
    s = s.detach()

    delta_c = (c[1:] - c[:N]).detach()                                 # (N, L)
    d_per = delta_c * s                                                # (N, L)
    d_k = d_per.sum(dim=1)                                             # (N,)

    return s, df.detach(), d_per, d_k


# =============================================================================
# Objective (paper Eq. 16: |d_k| signal term)
# =============================================================================
def _objective(d_k: torch.Tensor, df: torch.Tensor, mu: torch.Tensor,
               lam: float, tau: float):
    """
    Discrete signal-harvesting objective with |d_k| harvesting term (paper Eq. 16).
    Returns (obj, var, sig, energy) all as scalar tensors.
    """
    eps = 1e-12
    df_sq = df.pow(2)
    denom = (mu * df_sq).sum().clamp_min(eps)
    nu = (mu * df_sq) / denom

    safe_df = torch.where(df.abs() > eps, df, torch.ones_like(df))
    phi = d_k / safe_df
    phi_bar = (nu * phi).sum()
    var = (nu * (phi - phi_bar).pow(2)).sum()

    # Paper Eq. 16: harvesting term is sum_k mu_k * |d_k|.
    sig = (mu * d_k.abs()).sum()

    energy = 0.5 * tau * mu.pow(2).sum()
    obj = var - lam * sig + energy
    return obj, var, sig, energy


# =============================================================================
# Phase 1: optimise mu on the simplex
# =============================================================================
def _project_simplex(v: torch.Tensor) -> torch.Tensor:
    """Euclidean projection onto the probability simplex."""
    n = v.numel()
    u, _ = torch.sort(v, descending=True)
    cssv = torch.cumsum(u, dim=0) - 1.0
    rho_idx = torch.arange(1, n + 1, device=v.device, dtype=v.dtype)
    cond = u - cssv / rho_idx > 0
    if cond.any():
        rho = int(cond.nonzero().max().item()) + 1
    else:
        rho = 1
    theta = cssv[rho - 1] / rho
    return torch.clamp(v - theta, min=0.0)


def _optimise_mu(d_k: torch.Tensor, df: torch.Tensor,
                 lam: float, tau: float,
                 n_iter: int = 200, lr: float = 0.05,
                 debug: bool = False) -> torch.Tensor:
    """Projected Adam on the simplex for mu, minimising the LIG objective."""
    N = d_k.shape[0]
    device = d_k.device
    dtype  = d_k.dtype

    mu = torch.full((N,), 1.0 / N, device=device, dtype=dtype, requires_grad=True)
    opt = torch.optim.Adam([mu], lr=lr)

    if debug:
        obj0, var0, sig0, en0 = _objective(d_k, df, mu.detach(), lam, tau)
        _dbg(True, f"      [mu-opt] init: obj={float(obj0):+.4e} "
                   f"var={float(var0):.4e} sig={float(sig0):.4e} "
                   f"en={float(en0):.4e}")

    best_obj = float("inf")
    best_mu = mu.detach().clone()
    log_iters = {0, n_iter // 4, n_iter // 2, 3 * n_iter // 4, n_iter - 1}

    for it in range(n_iter):
        opt.zero_grad()
        obj, _, _, _ = _objective(d_k, df, mu, lam, tau)
        obj.backward()
        opt.step()
        with torch.no_grad():
            mu.copy_(_project_simplex(mu.detach()))
            cur = float(_objective(d_k, df, mu.detach(), lam, tau)[0])
            if cur < best_obj:
                best_obj = cur
                best_mu = mu.detach().clone()

        if debug and it in log_iters:
            obj_t, var_t, sig_t, en_t = _objective(d_k, df, mu.detach(), lam, tau)
            eff, nz, mu_max = _mu_sparsity(mu.detach())
            _dbg(True, f"      [mu-opt] it={it:3d} obj={float(obj_t):+.4e} "
                       f"var={float(var_t):.4e} sig={float(sig_t):.4e} "
                       f"| mu_max={mu_max:.3f} eff_supp={eff:.1f}/{N} (>1/2N: {nz})")

    if debug:
        eff, nz, mu_max = _mu_sparsity(best_mu)
        _dbg(True, f"      [mu-opt] best: obj={best_obj:+.4e} "
                   f"mu_max={mu_max:.3f} eff_supp={eff:.1f}/{N}")

    return best_mu


# =============================================================================
# Phase 2: optimise path c
# =============================================================================
def _optimise_path(model, c: torch.Tensor, X, X_baseline, fixed,
                   attention_mask, extra_kwargs, target_id, mu: torch.Tensor,
                   lam: float, tau: float, n_iter: int = 8, lr: float = 0.1,
                   debug: bool = False):
    """
    Optimise c (N+1, L) to minimise the LIG objective with |d_k| signal term.

    Surrogate gradient (for fixed s, mu, sgn(d_k), nu):
      sig term: -lam * mu_k * |d_k|, where d_k = sum_i (c_{k+1,i} - c_{k,i}) * s_{k,i}
        => d/d c_{k+1,i} of -lam*sig contribution =  -lam * mu_k * sgn(d_k) * s_{k,i}
           d/d c_{k,i}                            = +lam * mu_{k-1} * sgn(d_{k-1}) * s_{k-1,i}
                                                    -lam * mu_k * sgn(d_k) * s_{k,i}        (if 0<k<N)
      var term: gradient via chain rule through phi_k = d_k / df_k.
    """
    Np1, L = c.shape
    N = Np1 - 1

    s, df, d_per, d_k = _evaluate_path(
        model, c, X, X_baseline, attention_mask, extra_kwargs, target_id
    )
    cur_obj = float(_objective(d_k, df, mu, lam, tau)[0])
    c_init = c.clone()

    if debug:
        obj0, var0, sig0, en0 = _objective(d_k, df, mu, lam, tau)
        _dbg(True, f"      [path-opt] init: obj={float(obj0):+.4e} "
                   f"var={float(var0):.4e} sig={float(sig0):.4e}")

    eps = 1e-12
    accepted = 0
    rejected = 0

    for it in range(n_iter):
        df_sq = df.pow(2)
        denom = (mu * df_sq).sum().clamp_min(eps)
        nu = (mu * df_sq) / denom                                  # (N,)
        safe_df = torch.where(df.abs() > eps, df, torch.ones_like(df))
        phi = d_k / safe_df                                        # (N,)
        phi_bar = (nu * phi).sum()
        d_var_d_phi = 2.0 * nu * (phi - phi_bar)                   # (N,)
        var_grad_per_step_per_tok = (d_var_d_phi / safe_df).unsqueeze(-1) * s  # (N, L)

        # Signal term: -lam * sum_k mu_k * |d_k|, derivative wrt d_k is sgn(d_k).
        sgn_d = torch.sign(d_k)                                    # (N,)
        sig_per_step_per_tok = -lam * (mu * sgn_d).unsqueeze(-1) * s          # (N, L)

        per_step_grad = var_grad_per_step_per_tok + sig_per_step_per_tok       # (N, L)

        # Map to gradient w.r.t. c[k, i]:
        # contribution from step k uses (c_{k+1} - c_{k}); appears at index k+1
        # with sign +1 and at index k with sign -1.
        grad_c = torch.zeros_like(c)                               # (N+1, L)
        grad_c[1:] += per_step_grad
        grad_c[:N] -= per_step_grad

        grad_c[:, fixed] = 0.0
        grad_c[0, :] = 0.0
        grad_c[N, :] = 0.0

        c_new = c - lr * grad_c
        c_new = _project_c(c_new, fixed)

        s_new, df_new, _, d_k_new = _evaluate_path(
            model, c_new, X, X_baseline, attention_mask, extra_kwargs, target_id
        )
        new_obj = float(_objective(d_k_new, df_new, mu, lam, tau)[0])

        if debug:
            grad_norm = float(grad_c.norm())
            move = float((c_new - c).abs().sum())
            obj_t, var_t, sig_t, en_t = _objective(d_k_new, df_new, mu, lam, tau)
            tag = "ACC" if new_obj < cur_obj - 1e-8 else "REJ"
            _dbg(True, f"      [path-opt] it={it:2d} lr={lr:.4f} "
                       f"|grad_c|={grad_norm:.3e} L1_move={move:.3e} | "
                       f"obj={float(obj_t):+.4e} var={float(var_t):.4e} "
                       f"sig={float(sig_t):.4e} [{tag}]")

        if new_obj < cur_obj - 1e-8:
            c = c_new
            s, df, d_k = s_new, df_new, d_k_new
            cur_obj = new_obj
            accepted += 1
        else:
            lr = lr * 0.5
            rejected += 1
            if lr < 1e-4:
                if debug:
                    _dbg(True, f"      [path-opt] lr collapsed at it={it}, stopping")
                break

    if debug:
        c_drift = float((c - c_init).abs().mean())
        _dbg(True, f"      [path-opt] done: accepted={accepted} rejected={rejected} "
                   f"final_obj={cur_obj:+.4e} mean|Δc|={c_drift:.4e}")

    return c, s, df, d_k


# =============================================================================
# Main entry point
# =============================================================================
def lig_classification(
    sentence: str,
    a: float = 0.0,
    b: float = 1.0,
    steps: int = 50,
    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    show_special_tokens: bool = False,
    baseline: str = "mask",
    lam: float = 1.0,
    tau: float = 0.01,
    n_alternating: int = 3,
    mu_iter: int = 200,
    path_iter: int = 8,
    path_lr: float = 0.1,
    init_path: str = "uniform",
    guided_ig_kwargs: Optional[dict] = None,
    debug: bool = False,
    return_path: bool = False,
) -> Dict[str, Any]:
    """
    Args
    ----
    init_path : str
        "uniform"   - straight-line path c[k, i] = k/N (default; matches paper).
        "guided_ig" - warm-start c from a Guided IG run on the same input/baseline.
                      The Guided IG trajectory is converted to per-token cumulative
                      fractions via least-squares projection onto each token's
                      displacement direction, then projected onto LIG's monotone
                      simplex. The subsequent alternating minimisation then refines
                      both c and mu under the LIG objective; the Guided IG init is
                      a starting point only, not a constraint.
    guided_ig_kwargs : dict | None
        Optional kwargs forwarded to the Guided IG warm-start (currently:
        'fraction' and 'max_dist'). Ignored if init_path != 'guided_ig'.
    debug : bool
        If True, print a structured trace of the alternating optimisation:
        per-iter mu/path state, objective decomposition (var / sig / energy),
        completeness deviation, mu sparsity, and final attribution top-k.
    """
    if init_path not in ("uniform", "guided_ig"):
        raise ValueError(f"init_path must be 'uniform' or 'guided_ig', got {init_path!r}")

    N = int(steps)
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

    L = X.shape[1]
    dtype = X.dtype
    n_movable = int((~fixed).sum())

    # Effective baseline: fixed tokens pinned at X (their displacement is 0)
    X_baseline_eff = X_baseline.clone()
    X_baseline_eff[0, fixed, :] = X[0, fixed, :]

    with torch.no_grad():
        f_x  = model(inputs_embeds=X, attention_mask=attention_mask,
                     **extra_kwargs).logits[0, pred_id].item()
        f_bl = model(inputs_embeds=X_baseline_eff, attention_mask=attention_mask,
                     **extra_kwargs).logits[0, pred_id].item()
    target_change = f_x - f_bl

    if debug:
        tokens_full_dbg = tokenizer.convert_ids_to_tokens(input_ids[0])
        _dbg(True, "=" * 78)
        _dbg(True, f"[LIG-debug] sentence: {sentence!r}")
        _dbg(True, f"[LIG-debug] tokens (L={L}, movable={n_movable}): {tokens_full_dbg}")
        _dbg(True, f"[LIG-debug] pred_id={pred_id}  f(X)={f_x:+.4f}  "
                   f"f(X_baseline)={f_bl:+.4f}  Δf={target_change:+.4f}")
        _dbg(True, f"[LIG-debug] hyperparams: N={N} lam={lam} tau={tau} "
                   f"n_alt={n_alternating} mu_iter={mu_iter} path_iter={path_iter} "
                   f"path_lr={path_lr} init={init_path}")

    t0 = time.perf_counter()

    # ---- Init path ----
    if init_path == "guided_ig":
        if debug:
            _dbg(True, "[LIG-debug] running Guided IG to seed c...")
        c = _init_c_from_guided_ig(
            sentence, N, fixed, X, X_baseline_eff,
            model_name, device, baseline, guided_ig_kwargs,
        )
        if debug:
            # How non-uniform did the warm-start make things?
            c_uniform = _init_c_uniform(N, L, fixed, device, dtype)
            drift = float((c - c_uniform)[:, ~fixed].abs().mean())
            _dbg(True, f"[LIG-debug] Guided-IG warm-start: mean|c - c_uniform|={drift:.4f}")
    else:
        c = _init_c_uniform(N, L, fixed, device, dtype)

    mu = torch.full((N,), 1.0 / N, device=device, dtype=dtype)

    # ---- Track best state ----
    s, df, d_per, d_k = _evaluate_path(
        model, c, X, X_baseline_eff, attention_mask, extra_kwargs, pred_id
    )
    best_obj = float(_objective(d_k, df, mu, lam, tau)[0])
    best_c, best_mu = c.clone(), mu.clone()
    best_s, best_df = s.clone(), df.clone()

    if debug:
        obj_init, var_init, sig_init, en_init = _objective(d_k, df, mu, lam, tau)
        completeness = float(d_k.sum())
        _dbg(True, f"[LIG-debug] init eval: obj={float(obj_init):+.4e} "
                   f"var={float(var_init):.4e} sig={float(sig_init):.4e} "
                   f"en={float(en_init):.4e}")
        _dbg(True, f"[LIG-debug] init Σd_k={completeness:+.4f} (target Δf={target_change:+.4f})  "
                   f"|d_k| max={float(d_k.abs().max()):.4e}  "
                   f"|Δf_k| max={float(df.abs().max()):.4e}")

    # ---- Alternating minimisation ----
    for outer in range(n_alternating):
        if debug:
            _dbg(True, f"--- alt iteration {outer + 1}/{n_alternating} ---")
            _dbg(True, "    [Phase 1: mu update]")
        # Phase 1: mu update
        mu = _optimise_mu(d_k, df, lam=lam, tau=tau, n_iter=mu_iter, debug=debug)
        cur_obj = float(_objective(d_k, df, mu, lam, tau)[0])
        if debug:
            obj_t, var_t, sig_t, en_t = _objective(d_k, df, mu, lam, tau)
            eff, nz, mu_max = _mu_sparsity(mu)
            _dbg(True, f"    [Phase 1 done] obj={float(obj_t):+.4e} "
                       f"var={float(var_t):.4e} sig={float(sig_t):.4e} "
                       f"en={float(en_t):.4e} | mu_max={mu_max:.3f} "
                       f"eff_supp={eff:.1f}/{N}  ({'NEW BEST' if cur_obj < best_obj else 'no improvement'})")

        if cur_obj < best_obj:
            best_obj = cur_obj
            best_c, best_mu = c.clone(), mu.clone()
            best_s, best_df = s.clone(), df.clone()

        # Phase 2: path update (skip after the last mu update)
        if outer < n_alternating - 1:
            if debug:
                _dbg(True, "    [Phase 2: path update]")
            c_new, s_new, df_new, d_k_new = _optimise_path(
                model, c, X, X_baseline_eff, fixed, attention_mask, extra_kwargs,
                pred_id, mu, lam=lam, tau=tau, n_iter=path_iter, lr=path_lr,
                debug=debug,
            )
            new_obj = float(_objective(d_k_new, df_new, mu, lam, tau)[0])
            improved = new_obj < best_obj
            if debug:
                _dbg(True, f"    [Phase 2 done] obj={new_obj:+.4e}  "
                           f"({'NEW BEST' if improved else 'rejected — keep best so far'})")
            if improved:
                best_obj = new_obj
                best_c, best_mu = c_new.clone(), mu.clone()
                best_s, best_df = s_new.clone(), df_new.clone()
                c, s, df, d_k = c_new, s_new, df_new, d_k_new
            # else: keep current c, regression-guarded

    elapsed = time.perf_counter() - t0

    # ---- Final attribution ----
    delta_c = (best_c[1:] - best_c[:N])                             # (N, L)
    attr_per_step = best_mu.view(N, 1) * delta_c * best_s            # (N, L)
    attr_full = attr_per_step.sum(dim=0)                             # (L,)

    if debug:
        # Pre-rescale completeness check
        attr_sum_pre = float(attr_full.sum())
        # Compute Q on best state
        eps = 1e-12
        df_sq = best_df.pow(2)
        denom = (best_mu * df_sq).sum().clamp_min(eps)
        nu = (best_mu * df_sq) / denom
        safe_df = torch.where(best_df.abs() > eps, best_df, torch.ones_like(best_df))
        d_k_best = (delta_c * best_s).sum(dim=1)
        phi = d_k_best / safe_df
        phi_bar = (nu * phi).sum()
        var_best = (nu * (phi - phi_bar).pow(2)).sum()
        cv2 = float(var_best / (phi_bar.pow(2) + eps))
        Q = 1.0 / (1.0 + cv2)
        _dbg(True, "=" * 78)
        _dbg(True, f"[LIG-debug] FINAL: best_obj={best_obj:+.4e}  Q={Q:.4f}  "
                   f"Var_ν(φ)={float(var_best):.4e}")
        _dbg(True, f"[LIG-debug] completeness pre-rescale: Σattr={attr_sum_pre:+.4f} "
                   f"(target {target_change:+.4f}, ratio={attr_sum_pre/target_change if abs(target_change)>1e-12 else float('nan'):+.4f})")

    s_sum = attr_full.sum().item()
    if abs(s_sum) > 1e-12:
        attr_full = attr_full * (target_change / s_sum)

    inp = get_inputs(model, tokenizer, sentence, device)
    _, _, _, _, position_embed, _, type_embed, _, _ = inp
    tokens_full = tokenizer.convert_ids_to_tokens(input_ids[0])

    if debug:
        # Show top-k tokens by |attribution|
        k_top = min(10, L)
        top_vals, top_idx = attr_full.abs().topk(k_top)
        top_signed = attr_full[top_idx]
        _dbg(True, f"[LIG-debug] top-{k_top} tokens by |attr|:")
        for rank, (idx, val) in enumerate(zip(top_idx.tolist(), top_signed.tolist())):
            tok = tokens_full[idx]
            mark = "  (FIXED)" if bool(fixed[idx]) else ""
            _dbg(True, f"    {rank + 1:2d}. [{idx:3d}] {tok!r:>20s}  "
                       f"attr={val:+.4f}{mark}")
        _dbg(True, f"[LIG-debug] elapsed: {elapsed:.2f}s")
        _dbg(True, "=" * 78)

    result = pack_classification_result(
        tokens_full, attr_full, input_ids, tokenizer, show_special_tokens,
        elapsed, pred_id, model, nn_forward_func, X, attention_mask,
        position_embed, type_embed,
    )
    if return_path:
        # Best optimised path/measure for downstream Q computation.
        result["lig_c"]              = best_c.detach()           # (N+1, L)
        result["lig_mu"]             = best_mu.detach()          # (N,)
        result["lig_X_baseline_eff"] = X_baseline_eff.detach()   # (1, L, d)
        result["lig_fixed"]          = fixed.detach()            # (L,) bool
    return result