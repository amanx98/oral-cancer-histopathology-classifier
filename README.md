# 🔬 Oral Cancer Histopathology Classifier

A CNN that classifies H&E-stained oral tissue biopsy images as **Normal** or **OSCC** (Oral Squamous Cell Carcinoma), with Grad-CAM explainability and out-of-distribution detection built in.

**🔗 Live demo:** [oral-cancer-histopathology-classifier.streamlit.app](https://oral-cancer-histopathology-classifier-hng2phn3silmz4emxvhzwd.streamlit.app/)

assets/Screenshot 2026-08-12 011310.png


---

## Overview

Oral cancer is often diagnosed late, and histopathology remains the diagnostic gold standard. This project explores whether a transfer-learning CNN can distinguish normal oral mucosa from OSCC tissue in H&E-stained biopsy images — built as a hands-on deep learning project to (1) practice the full transfer learning → fine-tuning → evaluation → deployment pipeline, and (2) build a portfolio piece with genuine explainability and robustness features, not just a bare accuracy number.

**⚠️ This is a research/portfolio demo, not a diagnostic tool.** It has not been clinically validated and should never be used for actual medical decision-making.

## Dataset

[Histopathological Imaging Database for Oral Cancer](https://www.kaggle.com/datasets/ashenafifasilkebede/dataset) (Kaggle), a binary-labeled set of H&E-stained oral tissue images:

| Split | Normal | OSCC | Total |
|---|---|---|---|
| Train | 2,435 | 2,511 | 4,946 |
| Val | 28 | 92 | 120 |
| Test | 31 | 95 | 126 |

## Approach

**Architecture:** EfficientNet-B0 (ImageNet-pretrained backbone), classifier head replaced for binary output.

**Two-stage transfer learning:**
1. **Stage 1 — Head training:** backbone fully frozen, only a new classification head trained (~82.5% val accuracy).
2. **Stage 2 — Fine-tuning:** last 3 backbone blocks unfrozen and fine-tuned with a differential learning rate (`1e-5` backbone / `1e-4` head) — lets the network adapt its highest-level features to tissue-specific patterns without destroying pretrained low-level features.

**Augmentation:** random flips (horizontal + vertical) and rotation, since histopathology tissue has no inherent "up" — unlike natural images, a rotated slide is still a valid slide.

## Results

| Metric | Score |
|---|---|
| Test accuracy | **88.0%** |
| ROC-AUC | **0.937** |
| OSCC recall | 0.94 |
| Normal precision | 0.79 |

**Confusion matrix (test set, n=126):**

|              | Pred: Normal | Pred: OSCC |
|---|---|---|
| **Actual: Normal** | 22 | 9 |
| **Actual: OSCC** | 6 | 89 |

The model catches 89/95 real OSCC cases (94% recall) — the false-negative case (missed cancer) is the clinically costlier error, so this was prioritized over overall accuracy when interpreting results. The 9 false positives (Normal flagged as OSCC) are less dangerous but would mean unnecessary follow-up in a real clinical setting.

## Features

- **Grad-CAM explainability** — every prediction is accompanied by a heatmap showing which regions of the tissue image the model actually attended to, not just a bare label.
- **Out-of-distribution detection** — a two-stage gate runs before classification:
  1. A cheap color-histogram heuristic checks whether the image's palette resembles H&E staining at all.
  2. A Mahalanobis distance check compares the image's backbone feature embedding against the training set's feature distribution.

  Images that fail both are flagged as "not recognized as a histopathology slide" instead of being forced into a Normal/OSCC label — with a manual override toggle available if you want to see the raw prediction anyway.
- **Interactive demo UI** — built in Streamlit, with a toggle between a circular "microscope eyepiece" view and a full flat view of the uploaded image and its Grad-CAM overlay.

## Known limitations

- **Small test set** (126 images) — the 88% accuracy figure has real statistical noise; a handful of flipped predictions would shift it noticeably.
- **Grad-CAM occasionally attends to non-tissue regions** on misclassified Normal samples (e.g. image corners/edges) rather than cellular structures — a real finding worth further investigation, not hidden in this write-up.
- **Best results require high-resolution, minimally compressed images** — heavy downscaling or JPEG compression removes the fine cellular detail the model relies on.
- **OOD detection is heuristic**, not a formally validated method — it catches obviously wrong inputs well but isn't a rigorous guarantee.

## Tech stack

`PyTorch` · `torchvision` (EfficientNet-B0) · `scikit-learn` (Mahalanobis/covariance estimation) · `OpenCV` · `Streamlit` · trained on Kaggle (T4 GPU)

## Repo structure

```
├── app/
│   ├── app.py                  # Streamlit demo app
│   ├── requirements.txt
│   ├── model.pth                # fine-tuned model weights
│   ├── feature_mean.npy         # OOD detection stats
│   ├── feature_cov_inv.npy
│   └── ood_threshold.npy
├── notebook/
│   └── training.ipynb           # full training pipeline (Kaggle)
└── README.md
```

## Running locally

```bash
git clone https://github.com/amanx98/oral-cancer-histopathology-classifier.git
cd oral-cancer-histopathology-classifier/app
pip install -r requirements.txt
streamlit run app.py
```

## Future work

- Address the OSCC/Normal recall imbalance with class-weighted loss
- Try additional backbones (ResNet50, EfficientNetB3) for comparison
- Cross-validation for a more robust accuracy estimate given the small test set
- Investigate the Grad-CAM edge-attention finding further

## Acknowledgments

Dataset: [Ashenafi Fasil Kebede — Histopathological Imaging Database for Oral Cancer](https://www.kaggle.com/datasets/ashenafifasilkebede/dataset) (Kaggle).

## License

MIT
