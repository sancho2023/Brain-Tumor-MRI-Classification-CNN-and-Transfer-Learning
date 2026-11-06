# Brain-Tumor-MRI-Classification-CNN-and-Transfer-Learning

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%E2%89%A52.16-orange)
![Jupyter](https://img.shields.io/badge/notebook-Jupyter-F37626)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<your-user>/brain-tumor-mri-classification/blob/main/brain_tumor_mri_classification.ipynb)

A single, reproducible Jupyter notebook that classifies brain MRI slices into four classes
(**glioma, meningioma, no tumor, pituitary**) with a custom CNN and ImageNet transfer learning
(EfficientNetB0, optionally ResNet50V2 and DenseNet121), and reports a complete set of metrics.

**Dataset:** [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (Masoud Nickparvar, Kaggle).
The images are **not** included in this repository; the notebook downloads them with `kagglehub`.
Please check the dataset's terms on Kaggle before redistributing anything derived from it.

## Contents

- [What the notebook does](#what-the-notebook-does)
- [Results](#results-from-the-run-stored-in-the-notebook)
- [Quick start](#quick-start)
- [Repository layout](#repository-layout)
- [Roadmap](#roadmap)
- [References](#references)

## What the notebook does

- Downloads the data, builds an inventory and runs an exploratory analysis (class balance, image sizes, intensity, examples).
- **Audits duplicates and train/test leakage** (MD5 for exact copies, 64-bit dHash for near-duplicates) and optionally removes them.
- Trains a **custom CNN** and **transfer-learning models** (frozen base, then fine-tuning of the top layers).
- Selects the best model by **validation MCC** and reports the **test set once**.
- Metrics: accuracy, AUROC, sensitivity, specificity, F1, MCC, PPV/NPV, average precision, Brier score and ECE,
  with **95% bootstrap CIs** (global) and **Wilson CIs** (per class).
- Figures: confusion matrices (counts and normalised), ROC and precision-recall curves, calibration, training history.
- **Exact McNemar** test between models (Bonferroni-corrected), error analysis and **Grad-CAM**.
- Exports CSV/Excel tables, 300-dpi figures, model weights and per-image predictions to `outputs/`.

## Results from the run stored in the notebook

Light **CPU profile** (128×128 images, up to 15 epochs for the CNN), duplicate/leakage removal enabled, seed 42.
Test set: 1,600 images (400 per class). Values are `estimate [95% bootstrap CI]`.

| Model | Accuracy | AUROC (macro) | Sensitivity (macro) | Specificity (macro) | F1 (macro) | MCC |
|---|---|---|---|---|---|---|
| **CNN** (best on validation MCC) | 0.788 [0.768–0.808] | 0.942 [0.934–0.950] | 0.788 [0.770–0.805] | 0.929 [0.923–0.936] | 0.777 [0.756–0.797] | 0.729 [0.704–0.753] |
| EfficientNetB0 (transfer learning) | 0.749 [0.728–0.769] | 0.940 [0.932–0.947] | 0.749 [0.731–0.766] | 0.916 [0.910–0.923] | 0.731 [0.710–0.753] | 0.683 [0.659–0.707] |

Per-class results for the CNN:

| Class | Sensitivity | Specificity |
|---|---|---|
| glioma | 0.535 | 0.997 |
| meningioma | 0.648 | 0.910 |
| no tumor | 0.978 | 0.880 |
| pituitary | 0.993 | 0.931 |

Training took about 31 min (CNN) and 15 min (EfficientNetB0) on a laptop CPU (Windows 10, Python 3.14,
TensorFlow 2.22.0-rc0, Keras 3.15.1, no GPU).

### Figures

| Confusion matrix (CNN) | ROC and PR curves (CNN) |
|---|---|
| ![Confusion matrix](outputs/figures/cm_CNN.png) | ![ROC and PR](outputs/figures/roc_pr_CNN.png) |

![Model comparison with 95% CIs](outputs/figures/metric_comparison.png)

| Grad-CAM, correct predictions | Grad-CAM, most confident errors |
|---|---|
| ![Grad-CAM correct](outputs/figures/gradcam_correct.png) | ![Grad-CAM errors](outputs/figures/gradcam_errors.png) |

### Please read before comparing these numbers with other work

- These are modest results from a deliberately light configuration, not a tuned benchmark. Use `PROFILE = "gpu"` for larger images and longer training.
- The leakage filter removed **1,733 of the 5,600 Training images (about 31%)**, very unevenly: the *no tumor* class dropped from 1,400 to 460 images.
  The notebook flags near-duplicates automatically and does not check each flagged pair by eye, so some removals may not be true duplicates.
  Set `REMOVE_TRAIN_TEST_LEAKAGE = False` and compare the two runs as a sensitivity analysis.
- Validation accuracy (up to about 0.91 for the CNN) is clearly higher than test accuracy (0.79), so the validation split is not a faithful proxy for the test set here.
- The dataset has no patient identifiers, so slices from the same patient may be spread across splits.
- Single run and single seed; no external validation.

## Quick start

```bash
git clone https://github.com/<your-user>/brain-tumor-mri-classification.git
cd brain-tumor-mri-classification
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab brain_tumor_mri_classification.ipynb
```

Then run the notebook top to bottom (Kernel → Restart & Run All). The first cells explain the setup in more detail.

- **Smoke test first:** set `QUICK_RUN = True` in the configuration cell to check the whole pipeline in a few minutes.
- **Profiles:** `PROFILE = "cpu"` (default: 128×128, CNN + EfficientNetB0) or `"gpu"` (224×224, four models, longer training).
- **GPU:** native Windows has no TensorFlow GPU support; use WSL2, Google Colab or Kaggle Notebooks.
- **Memory:** lower `BATCH` (for example 16) if you run out of RAM.

Main settings (first code cell):

| Setting | Meaning |
|---|---|
| `PROFILE` | `"cpu"` (light) or `"gpu"` (full) |
| `QUICK_RUN` | few epochs and few bootstrap resamples, for testing the pipeline |
| `RUN_MODELS` | which models to train, for example `["EfficientNetB0"]` |
| `REMOVE_TRAIN_TEST_LEAKAGE`, `DROP_INTERNAL_DUPLICATES`, `HAMMING_THR` | duplicate/leakage handling |
| `N_BOOT` | number of bootstrap resamples for the 95% CIs |

## Repository layout

```
.
├── brain_tumor_mri_classification.ipynb   # the full pipeline, with outputs from one run
├── requirements.txt
├── scripts/
│   └── scrub_local_paths.py               # removes personal paths from a notebook before publishing
└── outputs/                               # created when you run the notebook (tables, figures, weights)
```

## Roadmap

- [ ] Show flagged near-duplicate pairs side by side and tune the dHash threshold
- [ ] Report results with and without leakage removal (sensitivity analysis)
- [ ] Run the `gpu` profile (224×224, four models) and add the results
- [ ] Repeat with several seeds / stratified cross-validation
- [ ] External validation on another MRI dataset

## References

- Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*, ICCV 2017.
- Tan & Le, *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*, ICML 2019.
- Matthews, *Comparison of the predicted and observed secondary structure of T4 phage lysozyme*, Biochim. Biophys. Acta, 1975 (Matthews correlation coefficient).
- Mongan et al., *Checklist for Artificial Intelligence in Medical Imaging (CLAIM)*, Radiology: Artificial Intelligence, 2020.
- Dataset: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset), Masoud Nickparvar (Kaggle). Check the dataset page for the citation and terms of use requested by the author.

## Disclaimer

This is a training project. The models are not validated for clinical use and must not be used for diagnosis or treatment decisions.
