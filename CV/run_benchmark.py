"""
run_benchmark.py - Run attribution benchmark across all backbones
=================================================================

Runs compare_methods_batch for each backbone model and saves per-model
results to JSON files under a timestamped output directory.

Usage:
    python run_benchmark.py
    python run_benchmark.py --models resnet50 vgg16 --n-test 10 --steps 30
    python run_benchmark.py --outdir results/my_run --methods ig idgi guided_ig
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime

from compare_methods import compare_methods_batch
from utility import BASELINE_MODES


ALL_MODELS = [
    "resnet50",
    "vgg16",
    "densenet121",
    "vit_b_16",
    "inception_v3",
    "swin_b",
    "convnext_base",
    "efficientnet_b0",
    "mobilenet_v2",
]

DEFAULT_METHODS = ["ig", "idgi", "guided_ig", "lig"]


def run_benchmark(
    models: list[str],
    methods: list[str],
    metrics: list[str],
    n_test: int,
    N: int,
    min_conf: float,
    device: str | None,
    seed: int,
    outdir: str,
    image_dir: str | None = None,
    baseline_mode: str | None = None,
):
    os.makedirs(outdir, exist_ok=True)

    summary = {
        "config": {
            "models": models,
            "methods": methods,
            "metrics": metrics,
            "n_test": n_test,
            "N": N,
            "min_conf": min_conf,
            "seed": seed,
            "image_dir": image_dir,
            "baseline": baseline_mode if baseline_mode is not None else "zeros",
        },
        "results": {},
    }

    for i, model_name in enumerate(models, 1):
        print(f"\n{'#'*70}")
        print(f"# [{i}/{len(models)}] Benchmark: {model_name}")
        print(f"{'#'*70}")

        json_path = os.path.join(outdir, f"{model_name}.json")

        # Resume: skip if result already exists
        if os.path.exists(json_path):
            try:
                with open(json_path) as f:
                    existing = json.load(f)
                if "statistics" in existing:
                    print(f">>> {model_name} already done, skipping (resume)")
                    summary["results"][model_name] = {
                        "status": "ok (resumed)",
                        "json": json_path,
                        "stats": existing["statistics"],
                    }
                    continue
            except Exception:
                pass  # corrupted file, re-run

        t0 = time.time()

        try:
            stats = compare_methods_batch(
                model_name=model_name,
                methods=methods,
                metrics=metrics,
                N=N,
                n_test=n_test,
                min_conf=min_conf,
                device=device,
                seed=seed,
                json_path=json_path,
                image_dir=image_dir,
                baseline_mode=baseline_mode,
            )
            elapsed = time.time() - t0
            summary["results"][model_name] = {
                "status": "ok",
                "elapsed_s": round(elapsed, 2),
                "json": json_path,
                "stats": stats,
            }
            print(f"\n>>> {model_name} done in {elapsed:.1f}s -> {json_path}")

        except Exception as e:
            elapsed = time.time() - t0
            summary["results"][model_name] = {
                "status": "failed",
                "error": str(e),
                "elapsed_s": round(elapsed, 2),
            }
            print(f"\n>>> {model_name} FAILED after {elapsed:.1f}s: {e}")

    # Save summary
    summary_path = os.path.join(outdir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n{'='*70}")
    print(f"Benchmark complete. Summary saved to {summary_path}")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="Run attribution benchmark across backbone models"
    )
    parser.add_argument(
        "--models", type=str, nargs="+", default=ALL_MODELS,
        choices=ALL_MODELS,
        help=f"Models to benchmark (default: all {len(ALL_MODELS)})",
    )
    parser.add_argument(
        "--methods", type=str, nargs="+", default=DEFAULT_METHODS,
        choices=["ig", "idgi", "guided_ig", "blurig", "lig", "lig_idgi"],
        help="Attribution methods (default: ig idgi guided_ig lig). "
             "Add `blurig` to include Blur Integrated Gradients.",
    )
    parser.add_argument(
        "--metrics", type=str, nargs="+",
        default=["insertion", "deletion", "ins-del"],
        choices=["insertion", "deletion", "ins-del"],
        help="Evaluation metrics",
    )
    parser.add_argument("--n-test", type=int, default=30,
                        help="Number of test images per model (default: 30)")
    parser.add_argument("--steps", type=int, default=50,
                        help="Number of integration steps N (default: 50)")
    parser.add_argument("--min-conf", type=float, default=0.70,
                        help="Minimum classification confidence (default: 0.70)")
    parser.add_argument("--device", type=str, default=None,
                        choices=["cpu", "cuda", "mps"],
                        help="Device (default: auto)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--outdir", type=str, default=None,
                        help="Output directory (default: results/<timestamp>)")
    parser.add_argument("--image-dir", type=str, default=None,
                        help="Fixed image directory (all models use same images)")
    parser.add_argument("--baseline", type=str, default=None,
                        choices=list(BASELINE_MODES),
                        help="Baseline mode. Omit to use zeros_like.")

    args = parser.parse_args()

    if args.outdir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.outdir = os.path.join("results", timestamp)

    run_benchmark(
        models=args.models,
        methods=args.methods,
        metrics=args.metrics,
        n_test=args.n_test,
        N=args.steps,
        min_conf=args.min_conf,
        device=args.device,
        seed=args.seed,
        outdir=args.outdir,
        image_dir=args.image_dir,
        baseline_mode=args.baseline,
    )


if __name__ == "__main__":
    main()
