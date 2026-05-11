"""
train.py
--------
Training loop for the VolleyballTransformer.

Usage:
    python train.py                          # default config
    python train.py --epochs 100 --lr 3e-4  # custom

Outputs:
    data/features.parquet   – pre-computed features (cached)
    models/best_model.pt    – best checkpoint (by val accuracy)
    models/training_log.csv – loss/metric history
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

# Project imports
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from utils.feature_engineering import build_features
from utils.dataset import make_dataloaders
from models.model import VolleyballTransformer, VolleyballLoss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RESULT_LABELS = ["3-0", "3-1", "3-2", "0-3", "1-3", "2-3"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return (preds == targets).float().mean().item()


def wins_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = (logits > 0).long()
    return (preds == targets.long()).float().mean().item()


def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = result_loss = wins_loss = 0.0
    all_result_preds, all_result_true = [], []
    all_wins_preds, all_wins_true = [], []

    with torch.no_grad():
        for batch in loader:
            numeric = batch["numeric"].to(device)
            local_idx = batch["local_idx"].to(device)
            visit_idx = batch["visit_idx"].to(device)
            targets = {
                "match_result": batch["match_result"].to(device),
                "local_wins": batch["local_wins"].to(device),
            }

            outputs = model(numeric, local_idx, visit_idx)
            losses = loss_fn(outputs, targets)

            total_loss += losses["total"].item()
            result_loss += losses["result"].item()
            wins_loss += losses["wins"].item()

            all_result_preds.append(outputs["result_logits"].argmax(-1).cpu())
            all_result_true.append(targets["match_result"].cpu())
            all_wins_preds.append((outputs["wins_logit"] > 0).long().cpu())
            all_wins_true.append(targets["local_wins"].long().cpu())

    n = len(loader)
    result_preds = torch.cat(all_result_preds)
    result_true = torch.cat(all_result_true)
    wins_preds = torch.cat(all_wins_preds)
    wins_true = torch.cat(all_wins_true)

    result_acc = (result_preds == result_true).float().mean().item()
    wins_acc = (wins_preds == wins_true).float().mean().item()

    return {
        "total_loss": total_loss / n,
        "result_loss": result_loss / n,
        "wins_loss": wins_loss / n,
        "result_acc": result_acc,
        "wins_acc": wins_acc,
        "result_preds": result_preds.numpy(),
        "result_true": result_true.numpy(),
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    print(f"Device: {DEVICE}")

    # --- Features ---
    features_path = ROOT / "data" / "features.parquet"
    if features_path.exists() and not args.rebuild_features:
        print("Loading cached features...")
        df = pd.read_parquet(features_path)
    else:
        print("Building features from scratch...")
        df = build_features(form_n=args.form_n)
        df.to_parquet(features_path, index=False)

    # Drop rows with missing targets
    df = df.dropna(subset=["match_result", "local_wins"]).reset_index(drop=True)

    # --- Data loaders ---
    train_loader, val_loader, test_loader, team_indexer, scaler, feature_cols = \
        make_dataloaders(
            df,
            batch_size=args.batch_size,
            val_season=args.val_season,
            test_season=args.test_season,
        )

    n_features = len(feature_cols)
    n_teams = team_indexer.n_teams

    # --- Model ---
    model = VolleyballTransformer(
        n_numeric_features=n_features,
        n_teams=n_teams,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {total_params:,}")
    print(f"Numeric features: {n_features}, Teams: {n_teams}")

    loss_fn = VolleyballLoss(alpha=args.loss_alpha, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # --- Training ---
    best_val_acc = 0.0
    best_epoch = 0
    log_rows = []
    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)

    print(f"\n{'Epoch':>6}  {'TrainLoss':>10}  {'ValLoss':>8}  {'ResAcc':>8}  {'WinsAcc':>8}  {'LR':>8}")
    print("-" * 65)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for batch in train_loader:
            numeric = batch["numeric"].to(DEVICE)
            local_idx = batch["local_idx"].to(DEVICE)
            visit_idx = batch["visit_idx"].to(DEVICE)
            targets = {
                "match_result": batch["match_result"].to(DEVICE),
                "local_wins": batch["local_wins"].to(DEVICE),
            }

            optimizer.zero_grad()
            outputs = model(numeric, local_idx, visit_idx)
            losses = loss_fn(outputs, targets)
            losses["total"].backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += losses["total"].item()

        scheduler.step()
        train_loss /= len(train_loader)

        # Validation
        val_metrics = evaluate(model, val_loader, loss_fn, DEVICE)
        lr_now = scheduler.get_last_lr()[0]

        print(f"{epoch:>6}  {train_loss:>10.4f}  {val_metrics['total_loss']:>8.4f}  "
              f"{val_metrics['result_acc']:>8.3f}  {val_metrics['wins_acc']:>8.3f}  "
              f"{lr_now:>8.2e}")

        log_rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_total_loss": val_metrics["total_loss"],
            "val_result_loss": val_metrics["result_loss"],
            "val_wins_loss": val_metrics["wins_loss"],
            "val_result_acc": val_metrics["result_acc"],
            "val_wins_acc": val_metrics["wins_acc"],
            "lr": lr_now,
        })

        # Checkpoint
        if val_metrics["result_acc"] > best_val_acc:
            best_val_acc = val_metrics["result_acc"]
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_result_acc": best_val_acc,
                "feature_cols": feature_cols,
                "n_teams": n_teams,
                "n_features": n_features,
                "args": vars(args),
            }, models_dir / "best_model.pt")

    # --- Test evaluation ---
    print(f"\nBest val result_acc: {best_val_acc:.3f} (epoch {best_epoch})")
    print("\nLoading best model for test evaluation...")

    checkpoint = torch.load(models_dir / "best_model.pt", map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate(model, test_loader, loss_fn, DEVICE)
    print(f"\nTest result_acc:  {test_metrics['result_acc']:.3f}")
    print(f"Test wins_acc:    {test_metrics['wins_acc']:.3f}")
    print(f"Test total_loss:  {test_metrics['total_loss']:.4f}")

    print("\nClassification Report (match result):")
    # Only report labels that appear in true
    present = sorted(set(test_metrics["result_true"].tolist()))
    present_labels = [RESULT_LABELS[i] for i in present]
    print(classification_report(
        test_metrics["result_true"],
        test_metrics["result_preds"],
        labels=present,
        target_names=present_labels,
        zero_division=0,
    ))

    print("Confusion Matrix (match result):")
    cm = confusion_matrix(test_metrics["result_true"], test_metrics["result_preds"],
                          labels=present)
    cm_df = pd.DataFrame(cm, index=present_labels, columns=present_labels)
    print(cm_df)

    # Save log
    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(models_dir / "training_log.csv", index=False)
    print(f"\nTraining log saved to {models_dir / 'training_log.csv'}")
    print(f"Best model saved to  {models_dir / 'best_model.pt'}")

    return model, test_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train VolleyballTransformer")

    # Data
    p.add_argument("--val_season", default="2024/2025")
    p.add_argument("--test_season", default="2025/2026")
    p.add_argument("--form_n", type=int, default=5,
                   help="Number of recent matches for form features")
    p.add_argument("--rebuild_features", action="store_true",
                   help="Force re-computation of features (ignores cache)")

    # Model
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--d_ff", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)

    # Training
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--loss_alpha", type=float, default=0.7,
                   help="Weight of result loss vs wins loss")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
