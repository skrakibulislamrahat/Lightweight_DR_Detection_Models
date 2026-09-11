#!/usr/bin/env python3
"""Five-fold cross-validation for lightweight referable-DR classifiers.

Expected APTOS CSV columns are `id_code` and `diagnosis`. If those names are
absent, the first two columns are interpreted as image ID and ordinal label.

Binary target:
    grades 0-1 -> non-referable (0)
    grades 2-4 -> referable (1)
"""
from __future__ import annotations

import argparse
import gc
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold


MODEL_NAMES = ("efficientnet_b0", "mobilenetv2_100", "squeezenet1_0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aptos-csv", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-checkpoints", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_aptos(csv_path: Path, image_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path).copy()
    if {"id_code", "diagnosis"}.issubset(df.columns):
        df = df.rename(columns={"id_code": "image_id", "diagnosis": "label"})
    else:
        df = df.rename(columns={df.columns[0]: "image_id", df.columns[1]: "label"})

    df["label"] = df["label"].astype(int)
    df["binary_label"] = (df["label"] >= 2).astype(int)
    df["image_path"] = df["image_id"].astype(str).map(lambda x: image_dir / f"{x}.png")
    df = df[df["image_path"].map(Path.exists)].reset_index(drop=True)

    if len(df) == 0:
        raise RuntimeError("No images from the CSV were found in --image-dir.")
    if df["binary_label"].nunique() != 2:
        raise RuntimeError("Both binary classes are required for stratified evaluation.")
    return df


class APTOSBinaryDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform=None) -> None:
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["binary_label"]), str(row["image_id"])


def build_transforms(image_size: int):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform, val_transform


def build_model(model_name: str, num_classes: int = 2) -> nn.Module:
    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    if model_name == "mobilenetv2_100":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    if model_name == "squeezenet1_0":
        model = models.squeezenet1_0(weights=models.SqueezeNet1_0_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
        model.num_classes = num_classes
        return model
    raise ValueError(f"Unsupported model: {model_name}")


def binary_metrics(y_true, y_pred, y_prob) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_prob)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    labels_all, preds_all = [], []
    for images, labels, _ in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        labels_all.extend(labels.detach().cpu().numpy())
        preds_all.extend(preds.detach().cpu().numpy())

    return total_loss / len(loader.dataset), accuracy_score(labels_all, preds_all)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    ids, y_true, y_pred, y_prob = [], [], [], []

    for images, labels, image_ids in tqdm(loader, desc="valid", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)

        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)
        total_loss += loss.item() * images.size(0)

        ids.extend(image_ids)
        y_true.extend(labels.detach().cpu().numpy())
        y_pred.extend(preds.detach().cpu().numpy())
        y_prob.extend(probs.detach().cpu().numpy())

    metrics = binary_metrics(y_true, y_pred, y_prob)
    metrics["loss"] = float(total_loss / len(loader.dataset))
    predictions = pd.DataFrame(
        {"image_id": ids, "y_true": y_true, "y_pred": y_pred, "y_prob": y_prob}
    )
    return metrics, predictions


def fit_fold(
    model_name: str,
    fold: int,
    train_idx,
    val_idx,
    df: pd.DataFrame,
    args: argparse.Namespace,
    device: torch.device,
    train_transform,
    val_transform,
):
    set_seed(args.seed + fold)
    fold_dir = args.output_dir / model_name / f"fold_{fold}"
    metrics_dir = fold_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "splits").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "fold_predictions").mkdir(parents=True, exist_ok=True)

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    train_df[["image_id", "label", "binary_label"]].to_csv(
        args.output_dir / "splits" / f"{model_name}_fold_{fold}_train.csv", index=False
    )
    val_df[["image_id", "label", "binary_label"]].to_csv(
        args.output_dir / "splits" / f"{model_name}_fold_{fold}_val.csv", index=False
    )

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        APTOSBinaryDataset(train_df, train_transform),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        APTOSBinaryDataset(val_df, val_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = build_model(model_name).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = GradScaler("cuda", enabled=device.type == "cuda")

    best_loss = float("inf")
    best_metrics = None
    best_predictions = None
    patience = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        val_metrics, predictions = evaluate(model, val_loader, criterion, device)
        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_accuracy": float(train_acc),
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })

        print(
            f"{model_name} fold={fold} epoch={epoch:02d} "
            f"train_loss={train_loss:.4f} val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} val_auc={val_metrics['auc']:.4f}"
        )

        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            best_metrics = val_metrics.copy()
            best_predictions = predictions.copy()
            patience = 0
            if args.save_checkpoints:
                torch.save(model.state_dict(), fold_dir / "best_model.pth")
        else:
            patience += 1
            if patience >= args.patience:
                break

    if best_metrics is None or best_predictions is None:
        raise RuntimeError("Training finished without a valid evaluation result.")

    with open(metrics_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(metrics_dir / "best_metrics.json", "w", encoding="utf-8") as f:
        json.dump(best_metrics, f, indent=2)
    best_predictions.to_csv(
        args.output_dir / "fold_predictions" / f"{model_name}_fold_{fold}_predictions.csv",
        index=False,
    )

    del model, train_loader, val_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_metrics


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics").mkdir(exist_ok=True)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load_aptos(args.aptos_csv, args.image_dir)
    train_transform, val_transform = build_transforms(args.image_size)

    dataset_summary = {
        "total_images": int(len(df)),
        "non_referable_count": int((df["binary_label"] == 0).sum()),
        "referable_count": int((df["binary_label"] == 1).sum()),
        "seed": args.seed,
        "folds": args.folds,
    }
    with open(args.output_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(dataset_summary, f, indent=2)

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    results = []

    for model_name in args.models:
        for fold, (train_idx, val_idx) in enumerate(
            skf.split(df, df["binary_label"]), start=1
        ):
            metrics = fit_fold(
                model_name,
                fold,
                train_idx,
                val_idx,
                df,
                args,
                device,
                train_transform,
                val_transform,
            )
            results.append({"model": model_name, "fold": fold, **metrics})

    results_df = pd.DataFrame(results).sort_values(["model", "fold"])
    results_df.to_csv(args.output_dir / "metrics" / "cv_fold_metrics.csv", index=False)
    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
