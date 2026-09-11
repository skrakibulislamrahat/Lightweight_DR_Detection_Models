# Lightweight Deep Learning for Referable Diabetic Retinopathy

This repository accompanies a research project evaluating compact convolutional neural networks for **referable diabetic retinopathy (DR) classification** from retinal fundus images, with emphasis on reproducibility, cross-validation stability, and deployment-oriented model comparison.

## Study design

The validated experimental protocol compares:

- **EfficientNet-B0**
- **MobileNetV2 (1.0 width)**
- **SqueezeNet 1.0**

APTOS 2019 labels are converted to a binary screening task:

- grades 0–1 → **non-referable DR**
- grades 2–4 → **referable DR**

The protocol uses **5-fold stratified cross-validation**, ImageNet initialization, augmentation, early stopping, and out-of-fold predictions. Metrics include accuracy, precision, sensitivity, specificity, F1-score, and AUROC. Pairwise McNemar tests compare model error-disagreement patterns.

## Cross-validation results

| Model | Accuracy | Sensitivity | Specificity | F1 | AUROC |
|---|---:|---:|---:|---:|---:|
| EfficientNet-B0 | 0.9394 ± 0.0137 | 0.9482 ± 0.0219 | 0.9333 ± 0.0139 | 0.9270 ± 0.0167 | 0.9839 ± 0.0054 |
| MobileNetV2 | 0.9358 ± 0.0113 | 0.9354 ± 0.0203 | 0.9361 ± 0.0141 | 0.9221 ± 0.0138 | 0.9847 ± 0.0048 |
| SqueezeNet 1.0 | 0.9342 ± 0.0137 | 0.9327 ± 0.0290 | 0.9352 ± 0.0110 | 0.9199 ± 0.0174 | 0.9813 ± 0.0069 |

Values are mean ± sample standard deviation across five folds. Pairwise McNemar comparisons in the completed experiment were not statistically significant at α = 0.05.

## Clean implementation

The primary readable implementation is now under `src/`:

```text
src/
├── train_cv.py          # five-fold training, early stopping, OOF predictions, metrics
└── analyze_results.py   # mean±SD tables, McNemar tests, ROC and confusion matrices
```

The older `Lightweight_DR_Model_Colab.ipynb` is retained as historical Colab provenance. It is **not** the recommended entry point because the research workflow evolved during revision.

### Run the five-fold experiment

```bash
pip install -r requirements.txt

python src/train_cv.py \
  --aptos-csv /path/to/train.csv \
  --image-dir /path/to/train_images \
  --output-dir ./run_outputs
```

The default configuration reproduces the documented protocol: seed 42, five folds, 224×224 images, 30 maximum epochs, patience 5, learning rate `1e-4`, weight decay `1e-5`, and all three architectures.

### Aggregate results and statistical comparisons

```bash
python src/analyze_results.py --results-dir ./run_outputs
```

This writes mean±SD summary tables, a manuscript-ready table, pairwise McNemar results, an out-of-fold ROC plot, and per-model confusion matrices.

## Repository contents

```text
.
├── src/
│   ├── train_cv.py
│   └── analyze_results.py
├── Lightweight_DR_Model_Colab.ipynb   # historical Colab implementation
├── Results/                            # saved visual/result artifacts currently tracked
├── model_comparison_bar_chart.png
├── EXPERIMENT_PROTOCOL.md
├── RESULTS.md
├── CITATION.cff
├── requirements.txt
└── README.md
```

## Reproducibility notes

Raw medical-image datasets and trained checkpoints are not distributed here. The scripts require the user to provide local paths rather than depending on a private Google Drive layout. Out-of-fold prediction files store image identifiers and predictions, not private filesystem paths.

See [`EXPERIMENT_PROTOCOL.md`](EXPERIMENT_PROTOCOL.md) for the experimental protocol and [`RESULTS.md`](RESULTS.md) for the validated result summary.

## Why this project matters

For low-resource or mobile screening systems, predictive performance alone is insufficient. A useful model must also have a manageable computational footprint and stable behavior across resampling. This project therefore treats architecture efficiency, cross-validation stability, sensitivity/specificity balance, and statistical model comparison as first-class evaluation targets.

## Research status

This repository documents an active research project. A formal paper citation/DOI will be added only when stable publication metadata is available. GitHub citation metadata are provided in [`CITATION.cff`](CITATION.cff).

## Responsible use

The code and results are for research and reproducibility. They are **not a clinical diagnostic system** and should not be used for patient-care decisions without appropriate external validation, governance, and regulatory review.
