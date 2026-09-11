# Experiment Protocol

## Task

Binary classification of referable diabetic retinopathy using APTOS 2019 fundus images.

- Original grades 0–1: non-referable
- Original grades 2–4: referable

## Models

- EfficientNet-B0
- MobileNetV2 (1.0 width)
- SqueezeNet 1.0

All models use ImageNet-pretrained initialization before adapting the final classifier to two classes.

## Evaluation design

- 5-fold stratified cross-validation
- fixed base random seed (`42`), with fold-specific reproducibility controls
- input resolution: 224 × 224
- batch size: 32
- Adam-style optimization in the Colab workflow
- initial learning rate: `1e-4`
- weight decay: `1e-5`
- maximum 30 epochs
- early-stopping patience: 5 epochs
- training augmentation: resize, horizontal flip, small rotation, and mild color jitter
- validation preprocessing: deterministic resize and ImageNet normalization

## Reported metrics

Each fold records:

- accuracy
- precision
- sensitivity/recall
- specificity
- F1-score
- AUROC
- validation loss

Final model summaries report mean ± sample standard deviation across five folds.

## Statistical comparison

Out-of-fold predictions are retained so that models can be compared on the same samples. Pairwise McNemar tests evaluate whether two classifiers have significantly different error-disagreement patterns.

## Reproducibility boundaries

Raw APTOS and Messidor-2 images are not committed. Large checkpoints and local Drive paths should remain outside Git. Record package versions, GPU/runtime information, dataset source, and any configuration change when reproducing the experiment.
