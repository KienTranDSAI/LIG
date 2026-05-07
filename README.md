# LIG — Least-action Integrated Gradients

Reference implementation accompanying *"Least Action Integrated Gradients"* (under review, 2026). LIG unifies the IG family (Standard IG, IDGI, Guided IG, BlurIG) as special cases of a single **signal-harvesting action** that jointly optimizes the path γ and the measure μ.

This repository ships two self-contained reference implementations — vision and language — sharing the same theoretical core.

## Repository layout

```
LIG-paper/
├── CV/      Image attribution: ResNet/VGG/DenseNet/ViT/… on ImageNet.
│            Faithfulness via insertion / deletion AUC and step-fidelity Q.
└── NLP/     Text attribution: BERT / DistilBERT / RoBERTa on
             SST-2 / IMDb / Rotten Tomatoes. Faithfulness via log-odds,
             comprehensiveness and sufficiency (DeYoung et al.).
```

Both parts ship with their own entry-point scripts and can be run independently — pick the one matching your modality.

## CV — `CV/`

Image attribution on pretrained torchvision backbones, evaluated on 50 fixed ImageNet validation images.

- Methods: `ig`, `idgi`, `guided_ig`, `blurig`, `lig`, `lig_idgi`
- Backbones: ResNet-50, VGG-16, DenseNet-121, ViT-B/16, Inception v3, Swin-B, ConvNeXt-Base, EfficientNet-B0, MobileNet v2
- Quick start: `cd CV && bash scripts/benchmark_resnet50.sh`
- Full docs and result tables: [`CV/README.md`](CV/README.md)
- Benchmark images: download from Drive (link in `CV/README.md`)

## NLP — `NLP/`

Text attribution on pretrained transformer classifiers from the TextAttack model hub.

- Methods: `ig`, `idig`, `guided_ig`, `lig` (uniform init), `lig_gig` (Guided-IG warm-start)
- Backbones × Datasets: `{distilbert, bert, roberta} × {sst2, imdb, rotten}` (9 fine-tuned HF checkpoints)
- Embedding-space baselines: `mask`, `pad`, `zero`, `mean`, `random`
- Metrics: log-odds, comprehensiveness, sufficiency, AOPC variants
- Quick start (single-sentence debug): `cd NLP && python db_lig.py`
- Sweep: `cd NLP && bash run_eval_xai.sh`
- CLI examples: `NLP/cmd`

## Dependencies

Python ≥ 3.9. Each subproject has its own runtime stack — install fresh in a venv per side.

| Subproject | Core stack |
|---|---|
| CV | `torch`, `torchvision`, `numpy`, `Pillow` |
| NLP | `torch`, `transformers`, `datasets`, `captum`, `numpy`, `tqdm` |

## Citation

```bibtex
@article{lig2026,
  title={Least Action Integrated Gradients},
  author={Anonymous},
  journal={Under review},
  year={2026}
}
```
