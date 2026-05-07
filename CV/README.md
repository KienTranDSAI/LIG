# LIG — Least-action Integrated Gradients

A unified variational framework that reveals all Integrated Gradients (IG) variants as special cases of a single **signal-harvesting action**: the path and measure must jointly minimize linearization distortion while maximizing attribution concentrated on the output-transition region.

This framework is the attribution analogue of an **optimal signal-harvesting optical instrument**—combining Fermat's principle (bending light to avoid high-curvature regions) with detector optimization (concentrating measurement on the bright transition region).

## The Core Principle

Standard IG integrates gradients along a straight line from a baseline to the input. In practice, this path traverses regions where the model's output is flat (wasting interpolation budget with no signal) and regions of sharp nonlinearity (where the linear approximation breaks down). Recent methods address this differently:

- **IDGI** reweights *which steps to trust* (the measure μ) by setting μ_k ∝ |Δf_k|
- **Guided IG** changes *where to evaluate gradients* (the path γ) using a low-gradient-first heuristic

**LIG shows both are approximate solutions to the same variational principle**—minimizing the signal-harvesting action:

```
min_{γ,μ}  Var_ν(φ) − λ Σ_k μ_k |d_k| + (τ/2) ||μ||²_2
           └────────┘   └──────────────┘   └─────────┘
           distortion   signal harvested   L² admissibility
```

where:
- **φ_k = d_k / Δf_k** is step fidelity (ratio of gradient-predicted to actual output change)
- **Var_ν(φ)** measures linearization distortion under the effective measure ν_k ∝ μ_k Δf²_k
- **−λ Σ_k μ_k |d_k|** rewards concentrating μ on steps with large output change
- **τ/2 ||μ||²_2** prevents μ from collapsing to a Dirac spike

## Unified Framework: All Methods as Special Cases

All existing IG variants are special cases parameterized by (λ, τ, γ):

| Method | Path γ | λ | τ | Measure μ* | What it optimizes |
|--------|--------|---|---|-----------|-------------------|
| **Standard IG** | straight line | 0 | ∞ | uniform | nothing (baseline) |
| **IDGI** | straight line | >0 | →0 | μ_k ∝ \|Δf_k\| | μ only (exact KKT stationary) |
| **Guided IG** | heuristic | >0 | — | uniform | γ only (approximate stationary) |
| **LIG** (ours) | **optimized** | **>0** | **>0** | **joint optimal** | **γ + μ jointly** |

**Key Insight**: IDGI's measure μ_k ∝ |Δf_k| is not a heuristic—it's the exact stationary point of the signal-harvesting objective in the limit τ → 0⁺. Similarly, Guided IG's path construction approximates the forced Euler-Lagrange equation for the signal-harvesting action.

## Results

**Average over 9 backbones** (ResNet-50, VGG-16, DenseNet-121, ViT-B/16, Inception v3, Swin-B, ConvNeXt-Base, EfficientNet-B0, MobileNet v2) on 50 ImageNet validation images, N=50 steps, zero baseline (mean ± std):

| Method | Q ↑ | Var_ν ↓ | Ins AUC ↑ | Del AUC ↓ | Ins-Del ↑ | Time (s) |
|--------|-----|---------|-----------|-----------|-----------|----------|
| IG | 0.7150 ± 0.116 | 0.4470 ± 0.382 | 0.6257 ± 0.091 | 0.3829 ± 0.180 | 0.2428 ± 0.135 | **0.21 ± 0.17** |
| IDGI | 0.6171 ± 0.284 | 0.3197 ± 0.323 | 0.6395 ± 0.098 | 0.3626 ± 0.180 | 0.2769 ± 0.167 | 2.23 ± 1.38 |
| Guided IG | 0.4881 ± 0.143 | 0.8194 ± 0.396 | 0.6660 ± 0.104 | 0.3417 ± 0.166 | 0.3243 ± 0.098 | 2.42 ± 1.39 |
| **LIG** (ours) | **0.9584 ± 0.039** | **0.0427 ± 0.073** | **0.6992 ± 0.107** | **0.3413 ± 0.156** | **0.3580 ± 0.088** | 39.05 ± 4.80 |

LIG attains the best score on **every** metric simultaneously: Q = 0.9584 with Var_ν = 0.0427 — a ~10× reduction in conservation violation versus Standard IG and ~19× versus Guided IG, while also leading on Insertion AUC, Deletion AUC, and the Ins-Del gap.

The unified framework **eliminates the conservation-faithfulness trade-off**: by jointly optimizing path and measure, LIG inherits the strengths of both IDGI (optimal measure) and Guided IG (improved path) while avoiding their respective weaknesses.

## Quality Metrics

Three related metrics derived from step fidelity φ_k = d_k / Δf_k:

- **Var_ν(φ)** — weighted variance of fidelity. The distortion term in the signal-harvesting action.
- **CV²(φ)** = Var_ν(φ) / φ̄² — scale-free coefficient of variation. Used in μ-optimization to prevent degeneracy.
- **Q** = 1/(1 + CV²) — quality score in [0, 1]. Q = 1 means perfect step-fidelity constancy across all active steps.

**Faithfulness metrics** (Petsiuk et al., 2018):
- **Insertion AUC** (↑): How quickly the output rises when features are added in order of attribution magnitude.
- **Deletion AUC** (↓): How quickly the output drops when features are removed in order of attribution magnitude.
- **Ins-Del** (↑): Insertion AUC − Deletion AUC. Single summary metric; higher is better.

## Files

```
utility.py          Common utilities, metrics, optimization functions
ig.py               Standard IG: compute_ig(model, input, params)
idgi.py             IDGI: compute_idgi(model, input, params)
guided_ig.py        Guided IG: compute_guided_ig(model, input, params)
lig_idgi.py         μ-Optimizer ablation: compute_lig_idgi(model, input, params)
lig.py              LIG (Joint*): compute_lig(model, input, params)
compare_methods.py  Per-image / batch evaluation framework
run_benchmark.py    Multi-model benchmark runner (writes JSON per model)
example_usage.py    Minimal example demonstrating each method
scripts/            Bash entry-points for full benchmarks
benchmark_50/       (download from Drive — see Benchmark Data below)
```



## Benchmark Data

The 50 fixed ImageNet validation images used in the paper, plus a metadata
file (`benchmark_50_images.json` — per-model confidences and predicted
class indices), are **not committed to this repo**. Download them from
Google Drive:

> https://drive.google.com/drive/folders/1npOETd-bZ9W8MwFg0rVH4jZrU8s1i5Z3?usp=sharing

After downloading, place everything inside `benchmark_50/`:

```
LIG-paper/
└── benchmark_50/                # download from Drive
    ├── n01491361_tiger_shark.JPEG
    ├── ...                      # 50 .JPEG images
    └── benchmark_50_images.json # metadata (reference only)
```

The benchmark scripts (`scripts/benchmark_*.sh`) read images from
`./benchmark_50/` via the `--image-dir` flag.

## Quick Start

### Using Code (Recommended)

```python
from utility import ClassLogitModel, get_device
from ig import compute_ig
from idgi import compute_idgi
from guided_ig import compute_guided_ig
from lig import compute_lig

# Setup
device = get_device()
model = ClassLogitModel(backbone, target_class)
baseline = torch.zeros_like(x)

# Run Standard IG
result_ig = compute_ig(model, x, {'baseline': baseline, 'N': 50})

# Run IDGI
result_idgi = compute_idgi(model, x, {'baseline': baseline, 'N': 50})

# Run Guided IG
result_guided = compute_guided_ig(model, x, {'baseline': baseline, 'N': 50})

# Run LIG (Joint*)
result_lig = compute_lig(model, x, {
    'baseline': baseline,
    'N': 50,
    'lam': 1.0,      # Signal-harvesting strength
    'tau': 0.01,     # L² admissibility
    'G': 16,         # Spatial groups
    'patch_size': 14,
    'n_alternating': 2,
    'mu_iter': 300,
    'path_iter': 10,
})

print(f"LIG: Q={result_lig.Q:.4f}, Ins-Del={result_lig.insdel.insertion_auc - result_lig.insdel.deletion_auc:.4f}")
```

### Using Compare Framework

**Single image mode:**
```bash
# Compare methods with ResNet50 on a single image
python compare_methods.py --model resnet50 --image path/to/image.jpg

# Use specific methods
python compare_methods.py --methods ig idgi guided_ig lig --steps 50 --image path/to/image.jpg

# Try different models
python compare_methods.py --model vgg16 --image path/to/image.jpg
python compare_methods.py --model densenet121 --image path/to/image.jpg
python compare_methods.py --model vit_b_16 --image path/to/image.jpg
```

**Batch testing mode (recommended for benchmarking):**
```bash
# Test on the 50 fixed paper images and save results to JSON
python compare_methods.py --model resnet50 --n-test 50 \
    --image-dir benchmark_50 --json results.json

# Full benchmark with all options
python compare_methods.py --model resnet50 --n-test 50 \
    --methods ig idgi guided_ig lig --steps 50 \
    --image-dir benchmark_50 --json benchmark_results.json
```

### Reproducing Paper Benchmarks

End-to-end runs are wrapped in `scripts/`:

```bash
bash scripts/benchmark_resnet50.sh   # ~15 min on A100 — ResNet50 only
bash scripts/benchmark_all.sh        # ~3 h  on A100 — 9 backbones
```

Both write per-model JSON to `results/<run-name>/`. Override the device with `DEVICE=cuda|mps|cpu`. See `scripts/README.md` for details.


## The Signal-Harvesting Action

The framework derives from a physical analogy:

**Classical mechanics**: Trajectories extremize the action S[γ] = ∫ L(γ, γ', t) dt

**Attribution**: Paths and measures jointly extremize the signal-harvesting action:

```
S[γ, μ] = ∫₀¹ ρ(t)² dt  −  λ ∫₀¹ |∇f·γ'| μ(t) dt
          └──────────┘      └───────────────────┘
          Fermat/Snell      Signal harvested
```

subject to γ(0) = x', γ(1) = x, and L² admissibility on μ.

**Stationary conditions** (Euler-Lagrange):
1. **Over μ**: Yields μ* ∝ |∇f·γ'| ≈ |Δf_k| (exactly IDGI's measure)
2. **Over γ**: Forcing term pushes γ toward regions where H_f γ' is large in direction of ∇f (the output-transition region, as in Guided IG)

The discrete objective replaces the intractable Hessian term with its proxy Var_ν(φ), yielding the practical optimization problem.

## Physical Analogy: Optical Signal Harvesting

| Concept | Optics | Attribution (LIG) |
|---------|--------|-------------------|
| **System** | Signal-harvesting optical instrument | Attribution path + measure |
| **Path** | Light ray γ(t) | Interpolation path γ(t) |
| **Measure** | Detector sensitivity μ(t) | Attribution weight μ(t) |
| **Endpoints** | Source, detector | Baseline x', input x |
| **Conservation** | Fermat/Snell: n sin θ = const | Step fidelity: ρ(t) = const |
| **Signal term** | Photon collection: ∫ I(t) μ(t) dt | Output change: ∫ \|∇f·γ'\| μ dt |
| **Trade-off** | Optical path length vs signal | Linearization error vs concentration |

LIG is an **optimal signal-harvesting detector**—it bends the path to avoid high-curvature regions (where gradients are unreliable) and concentrates its detection window on the output-transition region (where the model actually changes). Neither optimization alone is sufficient; both are necessary.

## Key Findings from the Paper

1. **Unification**: All IG variants (Standard IG, IDGI, Guided IG) are special cases of the signal-harvesting objective parameterized by (λ, τ).

2. **IDGI is exact**: IDGI's measure μ_k ∝ |Δf_k| is the exact KKT stationary point in the limit τ → 0⁺, λ > 0, not a heuristic.

3. **Guided IG is approximate**: Guided IG's path construction approximates the forced Euler-Lagrange equation for the signal-harvesting action.

4. **Trade-off elimination**: LIG resolves the conservation-faithfulness trade-off by jointly optimizing both degrees of freedom—best Q *and* best Insertion/Deletion AUC simultaneously across all 9 backbones.

5. **Robustness**: Path optimization is essential on challenging images where measure-only optimization fails (Q drops to 0.085 and 0.269 on e.g. Tibetan terrier, flat-coated retriever), but LIG recovers to Q > 0.999.

## Installation

Python ≥ 3.9 is required.

```bash
# (Recommended) create an isolated environment
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

For CUDA users, install the PyTorch build matching your CUDA version per
the [official selector](https://pytorch.org/get-started/locally/) before
running `pip install -r requirements.txt`.

### Dependencies

| Package | Used for |
|---|---|
| `torch >= 2.0` | model evaluation, autograd |
| `torchvision` | pretrained backbones, image transforms |
| `numpy` | numerical helpers in metrics / I/O |
| `Pillow` | image loading |

## References

**Primary Paper**:
- Anonymous. "Least Action Integrated Gradients." Under review, 2026.

**Related Work**:
- Sundararajan, Taly, Yan. "Axiomatic Attribution for Deep Networks." ICML 2017.
- Sikdar, Bhatt, Heese. "Integrated Directional Gradients." ACL 2021.
- Kapishnikov, Bolukbasi, Viégas, Terry. "Guided Integrated Gradients." CVPR 2021.
- Petsiuk, Das, Saenko. "RISE: Randomized Input Sampling for Explanation." BMVC 2018.
- Friedman. "Paths and Consistency in Additive Cost Sharing." Int. J. Game Theory 2004.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{lig2026,
  title={Least Action Integrated Gradients},
  author={Anonymous},
  journal={Under review},
  year={2026}
}
```
