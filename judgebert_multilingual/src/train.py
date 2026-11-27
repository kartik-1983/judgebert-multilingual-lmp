"""
Professional training script for the multilingual JUDGEBERT-inspired LMP regression model.

Improvements over basic version:
- Optional data augmentation with identical / unrelated pairs (JUDGEBERT-DA style)
- Layer-wise learning rates (higher LR for the regression head)
- Early stopping on validation RMSE
- Gradient accumulation
- Mixed precision (optional)
- Clear logging of metrics per epoch
- Reproducible splits stratified by language + quality
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit

from model import JudgeBERTRegressor, get_tokenizer, prepare_pair
from utils import load_dataset, compute_metrics, print_metrics, save_results


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LMPDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 256):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        enc = prepare_pair(
            self.tokenizer,
            str(row["original"]),
            str(row["simplified"]),
            max_length=self.max_length,
        )
        label = torch.tensor(float(row["lmp_score"]), dtype=torch.float)
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": label,
        }


def augment_with_sanity_pairs(df: pd.DataFrame, n_identical: int = 40, n_unrelated: int = 40) -> pd.DataFrame:
    """
    JUDGEBERT-DA style augmentation: add extra identical (score=10) and
    unrelated (score=1) pairs so the model learns the two critical sanity checks.
    """
    records = []
    # Sample from existing originals to create new identical pairs
    pool = df[df["quality"].isin(["good", "bad"])].sample(
        min(len(df), n_identical + n_unrelated), random_state=42
    )

    for i, (_, row) in enumerate(pool.iterrows()):
        if i < n_identical:
            records.append({
                "id": f"aug_id_{i}",
                "language": row["language"],
                "domain": "sanity_identical",
                "original": row["original"],
                "simplified": row["original"],
                "lmp_score": 10.0,
                "decision": "Preserved",
                "error_types": [],
                "explanation": "Identical (augmented).",
                "quality": "identical",
            })
        else:
            # Create unrelated by pairing with a different language / different text
            other = df[df["language"] != row["language"]].sample(1, random_state=i).iloc[0]
            records.append({
                "id": f"aug_un_{i}",
                "language": row["language"],
                "domain": "sanity_unrelated",
                "original": row["original"],
                "simplified": other["simplified"],
                "lmp_score": 1.0,
                "decision": "Not Preserved",
                "error_types": ["unrelated"],
                "explanation": "Unrelated (augmented).",
                "quality": "unrelated",
            })

    aug_df = pd.DataFrame(records)
    combined = pd.concat([df, aug_df], ignore_index=True)
    print(f"Augmented dataset: {len(df)} → {len(combined)} pairs "
          f"(+{n_identical} identical, +{n_unrelated} unrelated)")
    return combined


def get_optimizer(model, lr: float, weight_decay: float = 0.01, head_lr_mult: float = 10.0):
    """
    Layer-wise learning rates: higher LR for the randomly initialized regression head,
    lower LR for the pre-trained encoder (standard fine-tuning practice).
    """
    encoder_params = list(model.encoder.parameters())
    head_params = list(model.regressor.parameters())

    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": lr},
        {"params": head_params, "lr": lr * head_lr_mult},
    ], weight_decay=weight_decay)
    return optimizer


def train_epoch(model, loader, optimizer, scheduler, device, grad_accum: int = 1, use_amp: bool = False):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for step, batch in enumerate(tqdm(loader, desc="Train", leave=False)):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.cuda.amp.autocast(enabled=use_amp):
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out["loss"] / grad_accum

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += out["loss"].item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    total_loss = 0.0
    for batch in tqdm(loader, desc="Eval", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        total_loss += out["loss"].item()
        preds.extend(out["predictions"].cpu().numpy().tolist())
        trues.extend(labels.cpu().numpy().tolist())

    metrics = compute_metrics(np.array(trues), np.array(preds))
    metrics["loss"] = total_loss / max(len(loader), 1)
    return metrics, preds, trues


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Args: {vars(args)}")

    df = load_dataset(args.data)
    provenance = (
        set(df["label_provenance"].dropna().astype(str).str.lower())
        if "label_provenance" in df.columns else set()
    )
    is_synthetic = any("synthetic" in value or "simulated" in value for value in provenance)
    if is_synthetic and not args.allow_synthetic_training:
        raise SystemExit(
            "Dataset provenance marks these labels as synthetic/simulated. "
            "Pass --allow-synthetic-training only for exploratory development; "
            "do not use resulting metrics as human-validated research evidence."
        )
    print(f"Loaded {len(df)} pairs")
    if is_synthetic:
        print("WARNING: TRAINING ON SYNTHETIC LABELS — exploratory development only.")

    # Group by source clause so variants of the same original cannot leak across
    # train/validation/test. Split BEFORE augmentation; synthetic sanity examples
    # are added to training only.
    groups = (
        df["split_group"].astype(str)
        if "split_group" in df.columns
        else df["language"].astype(str) + "::" + df["original"].astype(str).str.strip()
    )
    first = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=args.seed)
    train_idx, temp_idx = next(first.split(df, groups=groups))
    train_df = df.iloc[train_idx].copy()
    temp_df = df.iloc[temp_idx].copy()
    second = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=args.seed + 1)
    temp_groups = (
        temp_df["split_group"].astype(str)
        if "split_group" in temp_df.columns
        else temp_df["language"].astype(str) + "::" + temp_df["original"].astype(str).str.strip()
    )
    val_rel, test_rel = next(second.split(temp_df, groups=temp_groups))
    val_df = temp_df.iloc[val_rel].copy()
    test_df = temp_df.iloc[test_rel].copy()

    if args.augment:
        train_df = augment_with_sanity_pairs(
            train_df, n_identical=args.n_aug_identical, n_unrelated=args.n_aug_unrelated
        )
    print(f"Group split → Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print("Leakage control: grouped by language + original clause; augmentation restricted to training.")

    tokenizer = get_tokenizer(args.model_name)
    model = JudgeBERTRegressor(
        model_name=args.model_name,
        dropout=args.dropout,
        freeze_base=args.freeze_base,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    train_loader = DataLoader(
        LMPDataset(train_df, tokenizer, args.max_length),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        LMPDataset(val_df, tokenizer, args.max_length),
        batch_size=args.batch_size * 2,
        shuffle=False,
    )
    test_loader = DataLoader(
        LMPDataset(test_df, tokenizer, args.max_length),
        batch_size=args.batch_size * 2,
        shuffle=False,
    )

    optimizer = get_optimizer(model, lr=args.lr, weight_decay=args.weight_decay, head_lr_mult=args.head_lr_mult)
    total_steps = (len(train_loader) // args.grad_accum) * args.epochs
    warmup_steps = int(args.warmup_ratio * total_steps)

    if args.scheduler == "cosine":
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    else:
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_val_rmse = float("inf")
    patience_counter = 0
    history = []
    ckpt_dir = Path(args.output_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("Starting fine-tuning")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        print(f"\n----- Epoch {epoch}/{args.epochs} -----")
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device,
            grad_accum=args.grad_accum, use_amp=args.fp16 and torch.cuda.is_available(),
        )
        val_metrics, _, _ = evaluate(model, val_loader, device)

        print(f"Train loss: {train_loss:.4f}")
        print_metrics(val_metrics, f"Validation")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "lr": scheduler.get_last_lr()[0],
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })

        # Early stopping + checkpointing
        if val_metrics["rmse"] < best_val_rmse - args.min_delta:
            best_val_rmse = val_metrics["rmse"]
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_name": args.model_name,
                "epoch": epoch,
                "val_metrics": val_metrics,
                "args": vars(args),
            }, ckpt_dir / "best_model.pt")
            print(f"  ★ New best model saved (val RMSE = {best_val_rmse:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{args.patience})")
            if patience_counter >= args.patience:
                print("Early stopping triggered.")
                break

    # Final test evaluation
    print("\n" + "=" * 60)
    print("Final Test Evaluation (best checkpoint)")
    print("=" * 60)
    ckpt = torch.load(ckpt_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics, test_preds, test_trues = evaluate(model, test_loader, device)
    print_metrics(test_metrics, "Test Set")

    # Per-language breakdown on test
    test_df = test_df.copy()
    test_df["judgebert_score"] = test_preds
    for lang in ["en", "hi"]:
        mask = test_df["language"] == lang
        if mask.sum() == 0:
            continue
        m = compute_metrics(
            test_df.loc[mask, "lmp_score"].values,
            test_df.loc[mask, "judgebert_score"].values,
        )
        print_metrics(m, f"Test – {lang.upper()}")

    test_df.to_csv(ckpt_dir / "test_predictions.csv", index=False)

    results = {
        "history": history,
        "best_val_rmse": best_val_rmse,
        "test_metrics": test_metrics,
        "args": vars(args),
    }
    save_results(results, str(ckpt_dir / "training_results.json"))
    print(f"\nAll artifacts saved under {ckpt_dir}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune JUDGEBERT-style LMP model")
    parser.add_argument("--data", default="data/lmp_dataset.jsonl")
    parser.add_argument(
        "--allow-synthetic-training",
        action="store_true",
        help="Explicitly allow training on synthetic/simulated labels for development only.",
    )
    parser.add_argument("--model_name", default="xlm-roberta-base",
                        help="Base model (xlm-roberta-base, microsoft/mdeberta-v3-base, etc.)")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--head_lr_mult", type=float, default=10.0,
                        help="Multiplier for regression-head learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--scheduler", choices=["linear", "cosine"], default="linear")
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision")
    parser.add_argument("--freeze_base", action="store_true", help="Freeze encoder (train head only)")
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True,
                        help="Add identical/unrelated pairs to training only (use --no-augment to disable)")
    parser.add_argument("--n_aug_identical", type=int, default=40)
    parser.add_argument("--n_aug_unrelated", type=int, default=40)
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--min_delta", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="models/judgebert_xlmr")
    args = parser.parse_args()
    main(args)
