"""
eval_xai_metrics.py - Faithfulness benchmark for IG / IDGI / Guided IG / LIG (NLP).

Supports running across {distilbert, bert, roberta} backbones x {sst2, imdb,
rotten} datasets via the (--backbone, --dataset) shortcut. Falls back to the
explicit --model HF identifier when --backbone is not given.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import time
import statistics
from typing import Dict, List, Tuple, Callable, Any, Optional

import torch
import torch.nn.functional as F


# =============================================================================
# Backbone x Dataset -> HF model name (mirrors run_pace.py)
# =============================================================================
_MODEL_TABLE: Dict[Tuple[str, str], str] = {
    ("distilbert", "sst2"):   "distilbert-base-uncased-finetuned-sst-2-english",
    ("distilbert", "imdb"):   "textattack/distilbert-base-uncased-imdb",
    ("distilbert", "rotten"): "textattack/distilbert-base-uncased-rotten-tomatoes",
    ("bert",       "sst2"):   "textattack/bert-base-uncased-SST-2",
    ("bert",       "imdb"):   "textattack/bert-base-uncased-imdb",
    ("bert",       "rotten"): "textattack/bert-base-uncased-rotten-tomatoes",
    ("roberta",    "sst2"):   "textattack/roberta-base-SST-2",
    ("roberta",    "imdb"):   "textattack/roberta-base-imdb",
    ("roberta",    "rotten"): "textattack/roberta-base-rotten-tomatoes",
}


def resolve_model_name(backbone: str, dataset: str) -> str:
    key = (backbone, dataset)
    if key not in _MODEL_TABLE:
        raise ValueError(
            f"No model registered for backbone={backbone!r}, dataset={dataset!r}. "
            f"Valid combos: {sorted(_MODEL_TABLE)}"
        )
    return _MODEL_TABLE[key]


# =============================================================================
# Sentence sources
# =============================================================================
_FALLBACK_SENTENCES = [
  "The movie was unhilariously funny, I mean it was bad",
  "The movie was visually stunning, but the plot was predictable",
  "it is not a mass-market entertainment but an uncompromising attempt by one artist to think about another.",
  "it 's also heavy-handed and devotes too much time to bigoted views",
  "The food tasted awful and the place was dirty",
  "It is summer, but the weather is bad",
  "The movie was an emotional masterpiece — the storytelling was powerful, the cinematography was breathtaking, and the music added so much depth to every scene",
 "i can go from feeling so hopeless to so damned hopeful just from being around someone who cares and is awake"
]


# Per-dataset reasonable word-count windows. IMDB reviews are long; SST-2 and
# Rotten Tomatoes excerpts are short.
_LEN_BOUNDS: Dict[str, Tuple[int, int]] = {
    "sst2":   (3, 30),
    "rotten": (3, 60),
    "imdb":   (10, 200),
}


def _load_dataset_sentences(dataset: str, n: int,
                            min_words: Optional[int] = None,
                            max_words: Optional[int] = None,
                            seed: int = 0) -> List[str]:
    """Load up to n sentences from a HF dataset, filtered by length."""
    from datasets import load_dataset
    import random as _rnd

    lo, hi = _LEN_BOUNDS.get(dataset, (3, 200))
    if min_words is not None: lo = min_words
    if max_words is not None: hi = max_words

    if dataset == "sst2":
        ds = load_dataset("glue", "sst2", split="validation")
        texts = [r["sentence"].strip() for r in ds]
    elif dataset == "rotten":
        ds = load_dataset("rotten_tomatoes", split="test")
        texts = [r["text"].strip() for r in ds]
    elif dataset == "imdb":
        ds = load_dataset("imdb", split="test")
        texts = [r["text"].strip() for r in ds]
        # IMDB test set is ordered by label; shuffle so we don't get all-negative.
        _rnd.Random(seed).shuffle(texts)
    else:
        raise ValueError(f"Unknown dataset {dataset!r}")

    sents = [t for t in texts if lo <= len(t.split()) <= hi]
    return sents[:n]


def load_sentences(n: int,
                   source: str = "auto",
                   dataset: Optional[str] = None,
                   dump_path: Optional[str] = None,
                   min_words: Optional[int] = None,
                   max_words: Optional[int] = None,
                   seed: int = 0) -> List[str]:
    """Load up to n sentences.

    source semantics:
      - 'fallback' : always use the built-in list
      - 'sst2'     : force GLUE/SST-2 validation (legacy, kept for back-compat)
      - 'auto'     : if `dataset` is given, use that dataset; else try SST-2
                     then fallback; failures fall through to the built-in list

    If `dump_path` is given AND a real dataset contributed sentences, those
    sentences are written one-per-line to that file.
    """
    used_dataset: Optional[str] = None
    sents: List[str] = []

    target_dataset: Optional[str] = None
    if source == "fallback":
        target_dataset = None
    elif source == "sst2":
        target_dataset = "sst2"
    elif source == "auto":
        target_dataset = dataset or "sst2"

    if target_dataset is not None:
        try:
            sents = _load_dataset_sentences(
                target_dataset, n,
                min_words=min_words, max_words=max_words, seed=seed,
            )
            if len(sents) >= n:
                sents = sents[:n]
                used_dataset = target_dataset
            else:
                print(f"  [warn] only {len(sents)} {target_dataset} sentences "
                      f"after filter; padding from fallback")
                sents = (sents + _FALLBACK_SENTENCES)[:n]
                used_dataset = target_dataset
        except Exception as e:
            if source == "sst2":
                raise
            print(f"  [warn] could not load {target_dataset} "
                  f"({type(e).__name__}); using fallback")
            sents = (_FALLBACK_SENTENCES * ((n // len(_FALLBACK_SENTENCES)) + 1))[:n]
    else:
        sents = (_FALLBACK_SENTENCES * ((n // len(_FALLBACK_SENTENCES)) + 1))[:n]

    if dump_path and used_dataset:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(dump_path)) or ".",
                        exist_ok=True)
            with open(dump_path, "w", encoding="utf-8") as f:
                for s in sents:
                    f.write(s + "\n")
            print(f"  Wrote {len(sents)} sampled {used_dataset} sentences "
                  f"to {dump_path}")
        except Exception as e:
            print(f"  [warn] could not write sample dump to {dump_path}: "
                  f"{type(e).__name__}: {e}")

    return sents


# =============================================================================
# Method registry
# =============================================================================
def _get_method_fn(name: str) -> Callable:
    """Return a callable (sentence, **kw) -> result_dict for the named method."""
    if name == "ig":
        from ig_nlp import ig_classification
        return ig_classification
    if name == "idig":
        from idig_nlp import idig_classification
        return idig_classification
    if name == "guided_ig":
        from guided_ig_nlp import guided_ig_classification
        return guided_ig_classification
    if name == "lig":
        from lig_nlp import lig_classification
        def _fn(sentence, **kw):
            return lig_classification(sentence, init_path="uniform", **kw)
        return _fn
    if name == "lig_gig":
        from lig_nlp import lig_classification
        def _fn(sentence, **kw):
            return lig_classification(sentence, init_path="guided_ig", **kw)
        return _fn
    raise ValueError(f"unknown method {name!r}")


METHOD_LABELS = {
    "ig": "IG",
    "idig": "IDGI",
    "guided_ig": "Guided IG",
    "lig": "LIG (uniform init)",
    "lig_gig": "LIG (guided-IG init)",
}


# =============================================================================
# Perturbation utilities
# =============================================================================
def _mask_embedding(model, tokenizer, device) -> torch.Tensor:
    embed = model.get_input_embeddings()
    mid = tokenizer.mask_token_id
    if mid is None:
        mid = tokenizer.pad_token_id
    with torch.no_grad():
        return embed(torch.tensor([[mid]], device=device))


def _logits_from_embeds(model, X: torch.Tensor, attention_mask: torch.Tensor):
    with torch.no_grad():
        out = model(inputs_embeds=X, attention_mask=attention_mask)
    return out.logits


def _select_movable(input_ids: torch.Tensor, attention_mask: torch.Tensor,
                    tokenizer) -> torch.Tensor:
    L = input_ids.shape[1]
    device = input_ids.device
    ids = input_ids[0]
    is_pad = (attention_mask[0] == 0)
    special = set(tokenizer.all_special_ids)
    is_special = torch.tensor([int(t.item()) in special for t in ids],
                              dtype=torch.bool, device=device)
    return ~(is_pad | is_special)


def _topk_indices(attr: torch.Tensor, movable: torch.Tensor, k: int) -> torch.Tensor:
    masked = attr.abs().clone()
    masked[~movable] = float("-inf")
    vals, idx = masked.topk(min(k, int(movable.sum())))
    return idx


# =============================================================================
# Faithfulness metrics
# =============================================================================
def _comp_suff_at_k(model, X: torch.Tensor, attention_mask: torch.Tensor,
                    pred_id: int, attr: torch.Tensor, movable: torch.Tensor,
                    mask_emb: torch.Tensor, k: int) -> Dict[str, float]:
    L = X.shape[1]

    logits0 = _logits_from_embeds(model, X, attention_mask)
    p0 = F.softmax(logits0, dim=-1)[0, pred_id].item()
    logp0 = math.log(max(p0, 1e-12))

    if k <= 0 or int(movable.sum()) == 0:
        return {"comp": 0.0, "suff": 0.0, "log_odds": 0.0}

    top_idx = _topk_indices(attr, movable, k)

    X_rem = X.clone()
    X_rem[0, top_idx, :] = mask_emb[0, 0, :]
    p_rem = F.softmax(_logits_from_embeds(model, X_rem, attention_mask),
                      dim=-1)[0, pred_id].item()

    X_keep = X.clone()
    keep_mask = torch.zeros(L, dtype=torch.bool, device=X.device)
    keep_mask[top_idx] = True
    to_mask = movable & (~keep_mask)
    X_keep[0, to_mask, :] = mask_emb[0, 0, :]
    p_keep = F.softmax(_logits_from_embeds(model, X_keep, attention_mask),
                       dim=-1)[0, pred_id].item()

    log_odds = math.log(max(p_rem, 1e-12)) - logp0

    return {
        "comp": p0 - p_rem,
        "suff": p0 - p_keep,
        "log_odds": log_odds,
    }


def _aopc(model, X, attention_mask, pred_id, attr, movable, mask_emb,
          fractions=(0.1, 0.2, 0.3, 0.4, 0.5)) -> Dict[str, float]:
    n_movable = int(movable.sum())
    if n_movable == 0:
        return {"aopc_comp": 0.0, "aopc_suff": 0.0, "aopc_log_odds": 0.0}
    accs = {"comp": [], "suff": [], "log_odds": []}
    for f in fractions:
        k = max(1, int(round(f * n_movable)))
        m = _comp_suff_at_k(model, X, attention_mask, pred_id, attr,
                            movable, mask_emb, k)
        for key in accs:
            accs[key].append(m[key])
    return {
        "aopc_comp": sum(accs["comp"]) / len(accs["comp"]),
        "aopc_suff": sum(accs["suff"]) / len(accs["suff"]),
        "aopc_log_odds": sum(accs["log_odds"]) / len(accs["log_odds"]),
    }


def _Q_from_path(model, gamma: torch.Tensor, attention_mask: torch.Tensor,
                 pred_id: int, mu: Optional[torch.Tensor] = None) -> float:
    Np1 = gamma.shape[0]
    N   = Np1 - 1
    device = gamma.device

    if mu is None:
        mu = torch.full((N,), 1.0 / N, device=device, dtype=gamma.dtype)

    am = attention_mask.expand(Np1, -1)
    with torch.no_grad():
        out = model(inputs_embeds=gamma, attention_mask=am)
        f_full = out.logits[:, pred_id]
    df = (f_full[1:] - f_full[:N]).detach().cpu()

    pts = gamma[:N].detach().clone().requires_grad_(True)
    am2 = attention_mask.expand(N, -1)
    out2 = model(inputs_embeds=pts, attention_mask=am2)
    scores = out2.logits[:, pred_id]
    (g,) = torch.autograd.grad(scores.sum(), pts)
    steps_vec = (gamma[1:] - gamma[:N]).detach()
    d_k = (g * steps_vec).sum(dim=(1, 2)).detach().cpu()
    mu_cpu = mu.detach().cpu().to(d_k.dtype)

    eps = 1e-12
    df_sq = df.pow(2)
    if float((mu_cpu * df_sq).sum()) < eps:
        return 0.0
    nu = (mu_cpu * df_sq) / (mu_cpu * df_sq).sum()
    safe_df = torch.where(df.abs() > eps, df, torch.ones_like(df))
    phi = d_k / safe_df
    phi_bar = (nu * phi).sum()
    if float(phi_bar.abs()) < eps:
        return 0.0
    var = (nu * (phi - phi_bar).pow(2)).sum()
    cv2 = float(var / (phi_bar.pow(2) + eps))
    return float(1.0 / (1.0 + cv2))


def _build_straightline_path(X: torch.Tensor, X_baseline: torch.Tensor,
                             n_steps: int) -> torch.Tensor:
    device = X.device
    t = torch.linspace(0.0, 1.0, n_steps + 1, device=device, dtype=X.dtype)
    return X_baseline.squeeze(0).unsqueeze(0) + \
        t.view(-1, 1, 1) * (X - X_baseline).squeeze(0).unsqueeze(0)


def _build_lig_path_from_c(c: torch.Tensor, X: torch.Tensor,
                           X_baseline: torch.Tensor) -> torch.Tensor:
    return X_baseline.squeeze(0).unsqueeze(0) + \
        c.unsqueeze(-1) * (X - X_baseline).squeeze(0).unsqueeze(0)


def _build_guided_ig_path(model, tokenizer, sentence: str,
                          X: torch.Tensor, X_baseline_eff: torch.Tensor,
                          fixed: torch.Tensor, attention_mask, extra_kwargs,
                          pred_id: int, n_steps: int,
                          fraction: float = 0.25, max_dist: float = 0.02,
                          ) -> torch.Tensor:
    import math as _math
    from guided_ig_nlp import (
        _translate_alpha_to_x, _translate_x_to_alpha, _grad_at, EPSILON,
    )

    x_input = X
    x_b     = X_baseline_eff
    x       = x_b.clone()
    total_diff = x_input - x_b
    l1_total = float(total_diff.abs().sum())

    if l1_total <= EPSILON:
        return _build_straightline_path(X, X_baseline_eff, n_steps)

    traj = [x.clone()]
    for step in range(n_steps):
        grad_actual = _grad_at(model, x, attention_mask, extra_kwargs, pred_id)
        grad = grad_actual.clone()

        alpha     = (step + 1.0) / n_steps
        alpha_min = max(alpha - max_dist, 0.0)
        alpha_max = min(alpha + max_dist, 1.0)
        x_min = _translate_alpha_to_x(alpha_min, x_input, x_b)
        x_max = _translate_alpha_to_x(alpha_max, x_input, x_b)
        l1_target = l1_total * (1.0 - (step + 1.0) / n_steps)

        gamma_step = float("inf")
        inner_safety = 0
        while gamma_step > 1.0:
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
            if _math.isclose(l1_target, l1_current, rel_tol=EPSILON, abs_tol=EPSILON):
                break
            grad[x == x_max] = float("inf")
            abs_grad_flat = grad.abs().reshape(-1)
            threshold = torch.quantile(abs_grad_flat, fraction, interpolation="lower")
            select = (grad.abs() <= threshold) & torch.isfinite(grad)
            l1_s = float(((x - x_max).abs() * select).sum())
            gamma_step = (l1_current - l1_target) / l1_s if l1_s > 0.0 else float("inf")
            if gamma_step > 1.0:
                x = torch.where(select, x_max, x)
            else:
                if gamma_step <= 0.0:
                    break
                x = torch.where(select, x + (x_max - x) * gamma_step, x)
        traj.append(x.clone())

    return torch.cat(traj, dim=0)


def _conservation_Q_for_method(method: str, model, tokenizer, sentence: str,
                               X, attention_mask, pred_id, movable, mask_emb,
                               result: Dict[str, Any], n_steps: int,
                               model_name: str, device: str,
                               baseline: str) -> float:
    from lig_utility_nlp import (
        get_pred_and_X, fixed_token_mask, encode_sentence, build_extra_kwargs,
    )

    enc = encode_sentence(tokenizer, sentence, device)
    input_ids = enc["input_ids"]
    am = enc["attention_mask"]
    stm = enc.get("special_tokens_mask", None)
    fixed = fixed_token_mask(tokenizer, input_ids, am, stm)

    embed = model.get_input_embeddings()
    from lig_utility_nlp import get_baseline_embedding
    X_baseline = get_baseline_embedding(baseline, embed, tokenizer, X, device)
    X_baseline_eff = X_baseline.clone()
    X_baseline_eff[0, fixed, :] = X[0, fixed, :]

    if method in ("ig", "idig"):
        gamma = _build_straightline_path(X, X_baseline_eff, n_steps)
        return _Q_from_path(model, gamma, attention_mask, pred_id, mu=None)

    if method == "guided_ig":
        token_type_ids = enc.get("token_type_ids", None)
        extra_kwargs = build_extra_kwargs(model, token_type_ids)
        gamma = _build_guided_ig_path(
            model, tokenizer, sentence, X, X_baseline_eff, fixed,
            attention_mask, extra_kwargs, pred_id, n_steps,
        )
        return _Q_from_path(model, gamma, attention_mask, pred_id, mu=None)

    if method in ("lig", "lig_gig"):
        c  = result.get("lig_c")
        mu = result.get("lig_mu")
        Xb = result.get("lig_X_baseline_eff")
        if c is None or mu is None or Xb is None:
            return float("nan")
        gamma = _build_lig_path_from_c(c, X, Xb)
        return _Q_from_path(model, gamma, attention_mask, pred_id, mu=mu)

    return float("nan")


# =============================================================================
# Per-sentence runner
# =============================================================================
def evaluate_one(method_name: str, method_fn: Callable, sentence: str, *,
                 steps: int, model_name: str, device: str, baseline: str,
                 compute_Q: bool = True) -> Dict[str, float]:
    t0 = time.perf_counter()
    method_kwargs = dict(
        steps=steps,
        model_name=model_name,
        device=device,
        baseline=baseline,
        show_special_tokens=True,
    )
    if method_name in ("lig", "lig_gig"):
        method_kwargs["return_path"] = True

    res = method_fn(sentence, **method_kwargs)
    elapsed = time.perf_counter() - t0

    model      = res["model"]
    X          = res["input_embed"]
    attn       = res["attention_mask"]
    pred_id    = res["predicted_label"]
    attr_full  = res["attr_full"]

    from lig_utility_nlp import get_model_tokenizer_cls, encode_sentence
    _, tokenizer = get_model_tokenizer_cls(model_name, device)
    enc = encode_sentence(tokenizer, sentence, device)
    input_ids = enc["input_ids"]

    movable = _select_movable(input_ids, attn, tokenizer)
    mask_emb = _mask_embedding(model, tokenizer, device)

    L = X.shape[1]
    if attr_full.shape[0] != L:
        raise RuntimeError(f"attr len {attr_full.shape[0]} != L {L}")

    n_movable = int(movable.sum())
    k20 = max(1, int(round(0.20 * n_movable)))
    m_single = _comp_suff_at_k(model, X, attn, pred_id, attr_full,
                               movable, mask_emb, k20)
    m_aopc = _aopc(model, X, attn, pred_id, attr_full, movable, mask_emb)

    if compute_Q:
        try:
            Q = _conservation_Q_for_method(
                method_name, model, tokenizer, sentence, X, attn, pred_id,
                movable, mask_emb, res, n_steps=steps,
                model_name=model_name, device=device, baseline=baseline,
            )
        except Exception as e:
            print(f"      [Q-error] {type(e).__name__}: {e}")
            Q = float("nan")
    else:
        Q = float("nan")

    return {
        "comp_20":       m_single["comp"],
        "suff_20":       m_single["suff"],
        "log_odds_20":   m_single["log_odds"],
        "aopc_comp":     m_aopc["aopc_comp"],
        "aopc_suff":     m_aopc["aopc_suff"],
        "aopc_log_odds": m_aopc["aopc_log_odds"],
        "Q":             Q,
        "time_s":        elapsed,
        "n_movable":     n_movable,
    }


# =============================================================================
# Aggregation & display
# =============================================================================
def _mean_std(xs: List[float]) -> Tuple[float, float]:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return float("nan"), float("nan")
    if len(xs) == 1:
        return xs[0], 0.0
    return statistics.mean(xs), statistics.stdev(xs)


def _paired_t(xs: List[float], ys: List[float]) -> Tuple[float, float]:
    diffs = [x - y for x, y in zip(xs, ys)
             if x is not None and y is not None
             and not math.isnan(x) and not math.isnan(y)]
    if len(diffs) < 2:
        return 0.0, 0.0
    m = statistics.mean(diffs)
    sd = statistics.stdev(diffs)
    if sd < 1e-12:
        return m, 0.0
    return m, m / (sd / math.sqrt(len(diffs)))


def print_table(rows: Dict[str, List[Dict[str, float]]]):
    methods = list(rows.keys())

    metric_specs = [
        ("comp_20",       "Comp@20 ↑",  False),
        ("suff_20",       "Suff@20 ↓",  True),
        ("log_odds_20",   "LO@20 ↓",    True),
        ("aopc_comp",     "AOPC-C ↑",   False),
        ("aopc_suff",     "AOPC-S ↓",   True),
        ("aopc_log_odds", "AOPC-LO ↓",  True),
        ("Q",             "Q ↑",        False),
        ("time_s",        "Time(s)",    True),
    ]

    header = f"{'Method':<22s} | " + " | ".join(f"{lbl:>13s}" for _, lbl, _ in metric_specs)
    sep = "-" * len(header)
    print()
    print("=" * len(header))
    print("RESULTS (mean ± std)")
    print("=" * len(header))
    print(header)
    print(sep)

    for m in methods:
        cells = []
        for key, _, _ in metric_specs:
            xs = [r[key] for r in rows[m]]
            mu, sd = _mean_std(xs)
            cells.append(f"{mu:+.3f}±{sd:.3f}")
        line = f"{METHOD_LABELS.get(m, m):<22s} | " + " | ".join(f"{c:>13s}" for c in cells)
        print(line)
    print(sep)

    if "ig" in rows and len(methods) > 1:
        print()
        print("Paired t-test vs IG (mean diff [other - IG], t-stat):")
        print("-" * 70)
        for m in methods:
            if m == "ig":
                continue
            line = f"  {METHOD_LABELS.get(m, m):<22s}"
            for key, lbl, lower_better in metric_specs:
                if key == "time_s":
                    continue
                diff, t = _paired_t([r[key] for r in rows[m]],
                                    [r[key] for r in rows["ig"]])
                wins = (diff < 0) if lower_better else (diff > 0)
                star = "*" if wins and abs(t) >= 1.96 else (" " if wins else " ")
                line += f"  {lbl}: {diff:+.3f} (t={t:+.2f}){star}"
                if key in ("comp_20", "aopc_comp", "Q"):
                    line += "\n  " + " " * 22
            print(line)
        print("(* = wins direction with |t| >= 1.96)")
    print()


# =============================================================================
# Main
# =============================================================================
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=20,
                   help="number of sentences (default 20)")
    p.add_argument("--source", choices=("auto", "sst2", "fallback"), default="auto",
                   help="sentence source. 'auto' uses --dataset if given, else SST-2.")
    p.add_argument("--methods", default="ig,idig,guided_ig,lig,lig_gig",
                   help="comma-separated method names")
    p.add_argument("--steps", type=int, default=30,
                   help="path steps for IG-family methods (default 30)")

    # ---- Model selection: either --backbone+--dataset OR --model ----
    p.add_argument("--backbone", choices=("distilbert", "bert", "roberta"),
                   default=None,
                   help="model backbone; combined with --dataset to resolve "
                        "the HF model name. Overrides --model when given.")
    p.add_argument("--dataset", choices=("sst2", "imdb", "rotten"),
                   default=None,
                   help="dataset to evaluate on. Drives both model resolution "
                        "(with --backbone) and sentence loading (with "
                        "--source auto).")
    p.add_argument("--model", default="distilbert-base-uncased-finetuned-sst-2-english",
                   help="explicit HF model name (used when --backbone is omitted)")

    p.add_argument("--baseline", default="mask",
                   help="embedding baseline: mask, pad, zero, mean, random")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--csv", default=None,
                   help="write per-sentence results to this CSV path")
    p.add_argument("--sample-dump", default="sample_sentences.txt",
                   help="when sentences come from a real dataset, write them "
                        "here one per line (default: sample_sentences.txt)")
    p.add_argument("--min-words", type=int, default=None,
                   help="override min sentence word count for dataset filter")
    p.add_argument("--max-words", type=int, default=None,
                   help="override max sentence word count for dataset filter")
    p.add_argument("--no-Q", action="store_true",
                   help="skip the Q metric (saves model evals)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in methods:
        if m not in METHOD_LABELS:
            raise SystemExit(f"unknown method {m!r}; valid: {list(METHOD_LABELS)}")

    # --- Resolve model name ---
    if args.backbone is not None:
        if args.dataset is None:
            raise SystemExit("--backbone requires --dataset")
        model_name = resolve_model_name(args.backbone, args.dataset)
        print(f"Resolved backbone={args.backbone} + dataset={args.dataset} "
              f"-> {model_name}")
    else:
        model_name = args.model

    # --- Sentence loading ---
    dump_path = (args.sample_dump
                 if args.source in ("sst2", "auto") else None)
    sentences = load_sentences(
        args.n,
        source=args.source,
        dataset=args.dataset,
        dump_path=dump_path,
        min_words=args.min_words,
        max_words=args.max_words,
        seed=args.seed,
    )

    print(f"Loaded {len(sentences)} sentences "
          f"(source={args.source!r}, dataset={args.dataset!r})")
    print(f"Model: {model_name}")
    print(f"Device: {args.device}")
    print(f"Baseline: {args.baseline}, steps: {args.steps}")
    print(f"Methods: {methods}")
    print()

    all_rows: Dict[str, List[Dict[str, float]]] = {m: [] for m in methods}
    for m in methods:
        method_fn = _get_method_fn(m)
        print(f"--- {METHOD_LABELS[m]} ---")
        t_method = time.perf_counter()
        for i, sent in enumerate(sentences):
            try:
                row = evaluate_one(
                    m, method_fn, sent,
                    steps=args.steps,
                    model_name=model_name,
                    device=args.device,
                    baseline=args.baseline,
                    compute_Q=not args.no_Q,
                )
                row["sentence_idx"] = i
                row["sentence"] = sent
                all_rows[m].append(row)
            except Exception as e:
                print(f"  [error] sent {i}: {type(e).__name__}: {e}")
                continue
            if (i + 1) % max(1, len(sentences) // 5) == 0:
                print(f"  {i+1}/{len(sentences)} done")
        print(f"  total: {time.perf_counter() - t_method:.1f}s, "
              f"{len(all_rows[m])}/{len(sentences)} succeeded")

    print_table(all_rows)

    if args.csv:
        import csv
        all_keys = ["method", "sentence_idx", "sentence",
                    "comp_20", "suff_20", "log_odds_20",
                    "aopc_comp", "aopc_suff", "aopc_log_odds",
                    "Q", "time_s", "n_movable"]
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_keys)
            w.writeheader()
            for m, rows in all_rows.items():
                for r in rows:
                    w.writerow({**{"method": m}, **{k: r.get(k, "") for k in all_keys[1:]}})
        print(f"Wrote per-sentence results to {args.csv}")


if __name__ == "__main__":
    main()