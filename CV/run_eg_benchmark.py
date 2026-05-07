"""
run_eg_benchmark.py - Compare Expected Gradients (EG) vs EG + LIG.

EG     = mean of IG attributions over baseline pool D  (MC form of EG with α-grid)
EG+LIG = mean of LIG attributions over baseline pool D

Baseline pool D = {black, white, noise, blur, mean_corners}
Insertion/deletion reference baseline = black (same for both methods).
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from compare_methods import load_model
from ig import compute_ig
from lig import compute_lig
from utility import (
    ClassLogitModel,
    _forward_scalar,
    compute_insertion_deletion,
    get_device,
    load_image_batch,
    make_baseline,
    set_seed,
)


BASELINES = ["black", "white", "noise", "blur", "mean_corners"]

LIG_PARAMS = {
    "lam": 1.0,
    "tau": 0.01,
    "G": 16,
    "patch_size": 14,
    "n_alternating": 2,
    "mu_iter": 300,
    "path_iter": 10,
}


def eval_attr(model, x, x_ref_baseline, attr, delta_f_ref):
    scores = compute_insertion_deletion(
        model, x, x_ref_baseline, attr, n_steps=100, batch_size=16
    )
    attr_sum = float(attr.sum())
    Q = attr_sum / delta_f_ref if abs(delta_f_ref) > 1e-9 else float("nan")
    return {
        "Q": Q,
        "insertion_auc": float(scores.insertion_auc),
        "deletion_auc": float(scores.deletion_auc),
        "ins_del": float(scores.insertion_auc - scores.deletion_auc),
    }


def run_one_image(backbone, x, target_class, N, seed):
    model = ClassLogitModel(backbone, target_class)

    x_black = make_baseline(x, "black", seed=seed)
    with torch.no_grad():
        f_x = float(_forward_scalar(model, x))
        f_black = float(_forward_scalar(model, x_black))
    delta_f_black = f_x - f_black

    ig_attrs, lig_attrs = [], []
    ig_time = 0.0
    lig_time = 0.0

    for bl in BASELINES:
        baseline = make_baseline(x, bl, seed=seed)

        t = time.time()
        ig_res = compute_ig(model, x, {"baseline": baseline, "N": N})
        ig_time += time.time() - t
        ig_attrs.append(ig_res.attributions)

        t = time.time()
        lig_res = compute_lig(model, x, {**LIG_PARAMS, "baseline": baseline, "N": N})
        lig_time += time.time() - t
        lig_attrs.append(lig_res.attributions)

    eg_ig = torch.stack(ig_attrs).mean(dim=0)
    eg_lig = torch.stack(lig_attrs).mean(dim=0)

    eg_metrics = eval_attr(model, x, x_black, eg_ig, delta_f_black)
    eg_lig_metrics = eval_attr(model, x, x_black, eg_lig, delta_f_black)
    eg_metrics["time"] = ig_time
    eg_lig_metrics["time"] = lig_time

    return {"eg": eg_metrics, "eg_lig": eg_lig_metrics}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet50")
    ap.add_argument("--n-test", type=int, default=50)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--min-conf", type=float, default=0.70)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None, choices=["cpu", "cuda", "mps", None])
    ap.add_argument("--image-dir", default="benchmark_50")
    ap.add_argument("--json", default="results/eg_vs_eglig/resnet50.json")
    ap.add_argument("--markdown", default="benchmark_eg_vs_eglig.md")
    args = ap.parse_args()

    set_seed(args.seed)
    dev = get_device(force=args.device)
    backbone = load_model(args.model, dev)

    print(f"\n{'='*70}")
    print(f"EG vs EG+LIG: {args.n_test} images, N={args.steps}, "
          f"pool={len(BASELINES)} baselines")
    print(f"Pool: {BASELINES}")
    print(f"Ins/del reference: black")
    print(f"{'='*70}\n")

    images = load_image_batch(
        backbone, dev, n=args.n_test, min_conf=args.min_conf,
        image_dir=args.image_dir, model_name=args.model,
    )[: args.n_test]

    keys = ["Q", "insertion_auc", "deletion_auc", "ins_del", "time"]
    all_results = {m: {k: [] for k in keys} for m in ["eg", "eg_lig"]}

    t_total = time.time()
    for i, (x, tc, conf, src, cls_name) in enumerate(images):
        t0 = time.time()
        res = run_one_image(backbone, x, tc, args.steps, args.seed)
        dt = time.time() - t0
        for m in ("eg", "eg_lig"):
            for k, v in res[m].items():
                all_results[m][k].append(v)
        print(f"[{i+1}/{len(images)}] "
              f"EG Q={res['eg']['Q']:.3f} ins={res['eg']['insertion_auc']:.3f} "
              f"del={res['eg']['deletion_auc']:.3f} | "
              f"EG+LIG Q={res['eg_lig']['Q']:.3f} "
              f"ins={res['eg_lig']['insertion_auc']:.3f} "
              f"del={res['eg_lig']['deletion_auc']:.3f} "
              f"| {dt:.1f}s")

    print(f"\nTotal wall-clock: {time.time()-t_total:.1f}s\n")

    stats = {}
    for m in ("eg", "eg_lig"):
        stats[m] = {}
        for k, vals in all_results[m].items():
            a = np.array(vals)
            stats[m][k] = {"mean": float(a.mean()), "std": float(a.std())}

    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    out = {
        "config": {
            "model": args.model, "n_test": args.n_test, "N": args.steps,
            "min_conf": args.min_conf, "seed": args.seed,
            "image_dir": args.image_dir, "baselines": BASELINES,
            "ins_del_reference": "black",
        },
        "statistics": stats,
    }
    with open(args.json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {args.json}")

    md = [
        f"# EG vs EG+LIG — {args.model} (n={args.n_test}, N={args.steps})",
        "",
        f"- Baseline pool (size {len(BASELINES)}): `{', '.join(BASELINES)}`",
        "- **EG**: mean over IG attributions across 5 baselines",
        "- **EG+LIG**: mean over LIG attributions across 5 baselines",
        "- Insertion/deletion reference baseline: `black` (same for both)",
        "- Q = sum(attr) / (f(x) − f(black))",
        "",
        "| Metric | EG | EG+LIG |",
        "|---|---|---|",
    ]
    fmt = lambda s: f"{s['mean']:.4f}±{s['std']:.4f}"
    for k, arrow in [("Q", "↑"), ("insertion_auc", "↑"),
                     ("deletion_auc", "↓"), ("ins_del", "↑")]:
        md.append(f"| {k} {arrow} | {fmt(stats['eg'][k])} | {fmt(stats['eg_lig'][k])} |")
    md.append(f"| time/image (s) | {stats['eg']['time']['mean']:.2f} "
              f"| {stats['eg_lig']['time']['mean']:.2f} |")
    md.append("")
    with open(args.markdown, "w") as f:
        f.write("\n".join(md))
    print(f"Saved {args.markdown}")


if __name__ == "__main__":
    main()
