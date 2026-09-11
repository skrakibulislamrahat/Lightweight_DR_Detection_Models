# Lightweight Deep Learning for Referable Diabetic Retinopathy

This repository accompanies a research project evaluating compact convolutional neural networks for **referable diabetic retinopathy (DR) classification** from retinal fundus images, with emphasis on reproducibility and deployment-oriented model comparison.

## Study design

The current experimental protocol evaluates three lightweight architectures:

- **EfficientNet-B0**
- **MobileNetV2 (1.0 width)**
- **SqueezeNet 1.0**

APTOS 2019 labels are converted to a binary screening task:

- grades 0–1 → **non-referable DR**
- grades 2–4 → **referable DR**

The major-revision experiment uses **5-fold stratified cross-validation**, ImageNet initialization, data augmentation, early stopping, and out-of-fold predictions. Metrics include accuracy, precision, sensitivity, specificity, F1-score, and AUROC. Pairwise McNemar tests are used to compare prediction disagreements between architectures.

## Cross-validation results

| Model | Accuracy | Sensitivity | Specificity | F1 | AUROC |
|---|---:|---:|---:|---:|---:|
| EfficientNet-B0 | 0.9394 ± 0.0137 | 0.9482 ± 0.0219 | 0.9333 ± 0.0139 | 0.9270 ± 0.0167 | 0.9839 ± 0.0054 |
| MobileNetV2 | 0.9358 ± 0.0113 | 0.9354 ± 0.0203 | 0.9361 ± 0.0141 | 0.9221 ± 0.0138 | 0.9847 ± 0.0048 |
| SqueezeNet 1.0 | 0.9342 ± 0.0137 | 0.9327 ± 0.0290 | 0.9352 ± 0.0110 | 0.9199 ± 0.0174 | 0.9813 ± 0.0069 |

Values are mean ± sample standard deviation across five folds.

The pairwise McNemar comparisons in the completed experiment were not statistically significant at α = 0.05, indicating that the three models had similar error-disagreement patterns despite differences in architecture and computational footprint.

## Repository contents

```text
.
├── Lightweight_DR_Model_Colab.ipynb   # Original end-to-end Colab implementation
├── Results/                            # Saved visual/result artifacts currently tracked
├── model_comparison_bar_chart.png      # Model comparison visualization
├── EXPERIMENT_PROTOCOL.md              # Current five-fold evaluation protocol
├── RESULTS.md                          # Validated revision results and statistical comparisons
├── requirements.txt                    # Core Python dependencies
└── README.md
```

The repository intentionally does **not** claim directories or scripts that are not actually tracked.

## Reproducibility notes

The newer research workflow was developed in Google Colab using APTOS 2019 and a separate Messidor-2 workspace for external-validation work. Raw medical-image datasets and trained checkpoints are not distributed here. See [`EXPERIMENT_PROTOCOL.md`](EXPERIMENT_PROTOCOL.md) for the protocol and [`RESULTS.md`](RESULTS.md) for the validated summary.

## Why this project matters

For low-resource or mobile screening systems, predictive performance alone is insufficient. A useful model must also have a manageable computational footprint and stable behavior across resampling. This project therefore treats architecture efficiency, cross-validation stability, sensitivity/specificity balance, and statistical model comparison as first-class evaluation targets.

## Research status

This repository documents an active research project. A formal paper citation/DOI will be added only when stable publication metadata is available.

## Responsible use

The code and results are for research and reproducibility. They are **not a clinical diagnostic system** and should not be used for patient-care decisions without appropriate external validation, governance, and regulatory review.
