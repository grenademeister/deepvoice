#!/usr/bin/env python3
"""Fine-tune SONICS on offline-prepared HTDemucs accompaniment stems.

This script never mutates model/sonics/pytorch_model.bin. It writes a standalone
checkpoint and metrics into --run-dir, so adoption by script.py is explicit.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "model"))
from sonics_infer import SonicsClassifier, preprocess_window


class PreparedStemDataset(torch.utils.data.Dataset):
    """Read immutable prepared stem manifests; no separator is invoked here."""
    def __init__(self, manifest: Path, max_len: int = 80000):
        with Path(manifest).open() as f:
            self.rows = list(csv.DictReader(f))
        if not self.rows:
            raise ValueError(f"Empty manifest: {manifest}")
        for row in self.rows:
            if not Path(row["filepath"]).is_file():
                raise FileNotFoundError(row["filepath"])
            if int(row["target"]) not in (0, 1):
                raise ValueError(f"Invalid target: {row['target']}")
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        audio, sr = torchaudio.load(row["filepath"])
        audio = audio.mean(0).numpy()
        if sr != 16000:
            audio = torchaudio.functional.resample(torch.from_numpy(audio)[None], sr, 16000)[0].numpy()
        audio = preprocess_window(audio, 16000, self.max_len)
        return torch.from_numpy(audio), torch.tensor([float(row["target"])], dtype=torch.float32)


def eer(y_true: np.ndarray, score: np.ndarray) -> float:
    """EER by exact threshold sweep; fake is the positive class."""
    y_true = np.asarray(y_true, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    if len(np.unique(y_true)) != 2:
        return float("nan")
    thresholds = np.r_[np.inf, np.sort(np.unique(score))[::-1], -np.inf]
    best = 1.0
    for threshold in thresholds:
        pred = score >= threshold
        fpr = np.mean(pred[y_true == 0])
        fnr = np.mean(~pred[y_true == 1])
        best = min(best, float((fpr + fnr) / 2))
    return best


def scores(model, loader, device):
    model.eval()
    all_score, all_target = [], []
    with torch.inference_mode():
        for audio, target in loader:
            logits = model(audio.to(device))
            all_score.extend(torch.sigmoid(logits).reshape(-1).cpu().numpy())
            all_target.extend(target.reshape(-1).numpy())
    return np.asarray(all_target, dtype=np.int64), np.asarray(all_score, dtype=np.float64)


def scores_with_loss(model, loader, device):
    """Compute validation EER inputs and mean BCE loss in one model pass."""
    model.eval()
    all_score, all_target, losses = [], [], []
    with torch.inference_mode():
        for audio, target in loader:
            target = target.to(device)
            logits = model(audio.to(device))
            losses.append(float(F.binary_cross_entropy_with_logits(logits, target).item()))
            all_score.extend(torch.sigmoid(logits).reshape(-1).cpu().numpy())
            all_target.extend(target.reshape(-1).cpu().numpy())
    return (np.asarray(all_target, dtype=np.int64), np.asarray(all_score, dtype=np.float64),
            float(np.mean(losses)))


def configure_trainable(
    model, unfreeze_last_block: bool, unfreeze_all: bool = False
) -> list[torch.nn.Parameter]:
    if unfreeze_all:
        for parameter in model.parameters():
            parameter.requires_grad = True
        return list(model.parameters())
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    if unfreeze_last_block:
        for parameter in model.encoder.transformer.blocks[-1].parameters():
            parameter.requires_grad = True
    return [p for p in model.parameters() if p.requires_grad]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--valid-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT / "model" / "sonics" / "pytorch_model.bin")
    parser.add_argument("--config", type=Path, default=PROJECT / "model" / "sonics" / "config.json")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    fine_tuning = parser.add_mutually_exclusive_group()
    fine_tuning.add_argument("--unfreeze-last-block", action="store_true")
    fine_tuning.add_argument("--unfreeze-all", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    datasets = {name: PreparedStemDataset(getattr(args, f"{name}_manifest")) for name in ("train", "valid", "test")}
    loaders = {
        name: torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=name == "train", num_workers=0)
        for name, ds in datasets.items()
    }
    with args.config.open() as f:
        config = json.load(f)
    model = SonicsClassifier(config)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True), strict=True)
    model = model.to(device)
    parameters = configure_trainable(
        model, args.unfreeze_last_block, args.unfreeze_all
    )
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    best_eer, best_epoch, stale = float("inf"), 0, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = []
        for audio, target in loaders["train"]:
            logits = model(audio.to(device))
            loss = F.binary_cross_entropy_with_logits(logits, target.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()
            train_loss.append(float(loss.item()))
        valid_target, valid_score, valid_loss = scores_with_loss(model, loaders["valid"], device)
        valid_eer = eer(valid_target, valid_score)
        record = {"epoch": epoch, "train_loss": float(np.mean(train_loss)), "valid_loss": valid_loss, "valid_eer": valid_eer}
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if valid_eer < best_eer:
            best_eer, best_epoch, stale = valid_eer, epoch, 0
            torch.save({"state_dict": model.state_dict(), "config": config, "source_checkpoint": str(args.checkpoint),
                        "seed": args.seed, "unfreeze_last_block": args.unfreeze_last_block,
                        "unfreeze_all": args.unfreeze_all}, args.run_dir / "best.pt")
        else:
            stale += 1
            if stale >= args.patience:
                break
    saved = torch.load(args.run_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(saved["state_dict"], strict=True)
    test_target, test_score = scores(model.to(device), loaders["test"], device)
    metrics = {
        "best_epoch": best_epoch, "validation_eer": best_eer, "test_eer": eer(test_target, test_score),
        "test_real_mean": float(test_score[test_target == 0].mean()), "test_fake_mean": float(test_score[test_target == 1].mean()),
        "counts": {name: {str(label): sum(int(r["target"]) == label for r in ds.rows) for label in (0, 1)} for name, ds in datasets.items()},
        "trainable_parameters": sum(p.numel() for p in parameters), "unfreeze_last_block": args.unfreeze_last_block,
        "unfreeze_all": args.unfreeze_all,
    }
    (args.run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    epochs = [x["epoch"] for x in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [x["train_loss"] for x in history], marker="o", label="train BCE")
    axes[0].plot(epochs, [x["valid_loss"] for x in history], marker="o", label="validation BCE")
    axes[0].set(xlabel="epoch", ylabel="BCE loss", title="Optimization loss")
    axes[0].legend(); axes[0].grid(alpha=.3)
    axes[1].plot(epochs, [x["valid_eer"] for x in history], marker="o", color="tab:red")
    axes[1].set(xlabel="epoch", ylabel="EER", title="Validation EER")
    axes[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(args.run_dir / "learning_curves.png", dpi=160); plt.close(fig)
    (args.run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    with (args.run_dir / "test_scores.csv").open("w", newline="") as f:
        writer = csv.writer(f); writer.writerow(["target", "fake_probability"]); writer.writerows(zip(test_target, test_score))
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
