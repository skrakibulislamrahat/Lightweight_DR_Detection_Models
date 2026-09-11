# Validated Cross-Validation Results

The completed major-revision workflow evaluated all three architectures across five stratified folds.

| Model | Accuracy | Precision | Sensitivity | Specificity | F1-score | AUROC |
|---|---:|---:|---:|---:|---:|---:|
| EfficientNet-B0 | 0.9394 ± 0.0137 | 0.9069 ± 0.0181 | 0.9482 ± 0.0219 | 0.9333 ± 0.0139 | 0.9270 ± 0.0167 | 0.9839 ± 0.0054 |
| MobileNetV2 | 0.9358 ± 0.0113 | 0.9094 ± 0.0178 | 0.9354 ± 0.0203 | 0.9361 ± 0.0141 | 0.9221 ± 0.0138 | 0.9847 ± 0.0048 |
| SqueezeNet 1.0 | 0.9342 ± 0.0137 | 0.9078 ± 0.0147 | 0.9327 ± 0.0290 | 0.9352 ± 0.0110 | 0.9199 ± 0.0174 | 0.9813 ± 0.0069 |

Values are mean ± sample standard deviation across five folds.

## Pairwise McNemar tests

| Model 1 | Model 2 | Statistic | p-value |
|---|---|---:|---:|
| EfficientNet-B0 | MobileNetV2 | 1.0827 | 0.2981 |
| EfficientNet-B0 | SqueezeNet 1.0 | 1.9401 | 0.1637 |
| MobileNetV2 | SqueezeNet 1.0 | 0.1420 | 0.7063 |

At α = 0.05, none of these pairwise comparisons is statistically significant.

## Interpretation

EfficientNet-B0 achieved the highest mean accuracy and F1-score, while MobileNetV2 had the highest mean AUROC by a small margin. The overlapping fold variability and non-significant McNemar tests argue against overstating a single architecture as decisively superior.

These results should be interpreted as internal cross-validation performance. External validation is a separate question and should not be inferred from these numbers alone.
