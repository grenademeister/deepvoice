#!/usr/bin/env python3
"""Train source-grouped OOF V4 three-output fusion heads.

Input feature CSV must contain `sample_id` plus all V4 frozen detector columns:
`df_raw, df_voice, sonics_stem, artifact_raw, artifact_stem, voice_present,
music_present`. Optional numeric columns are intentionally ignored: any future
feature requires an explicit contract update in `model/v4_fusion.py`.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold

from model.v4_fusion import (
    OUTPUT_NAMES,
    RAW_FEATURES,
    V4FusionHead,
    V4FusionMetadata,
    build_v4_features,
    checkpoint_payload,
    predict_v4_fusion,
)


def number(value: str | None) -> float:
    return 0.0 if value is None or value.strip() == "" else float(value)


def read_csv_by_id(path: Path, id_column: str) -> dict[str, dict[str, str]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or id_column not in rows[0]:
        raise ValueError(f"{path} must contain {id_column}")
    result = {row[id_column]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate {id_column} values in {path}")
    return result


def join_rows(manifest_path: Path, features_path: Path, split: str) -> list[dict[str, str]]:
    manifest = read_csv_by_id(manifest_path, "sample_id")
    features = read_csv_by_id(features_path, "sample_id")
    missing = sorted(set(manifest).intersection(features))
    rows = []
    for sample_id in missing:
        row = {**manifest[sample_id], **features[sample_id]}
        if row.get("split") == split:
            rows.append(row)
    if not rows:
        raise ValueError(f"No joined rows for split={split}; check IDs and manifest")
    absent = [name for name in RAW_FEATURES if name not in rows[0]]
    if absent:
        raise ValueError(f"Features missing required columns: {absent}")
    return sorted(rows, key=lambda r: r["sample_id"])


def arrays(rows: list[dict[str, str]]):
    values = {name: np.array([number(r.get(name)) for r in rows], dtype=np.float32) for name in RAW_FEATURES}
    x = build_v4_features(values)
    y = np.stack(
        [
            np.array([number(r.get("expected_file_fake")) for r in rows], dtype=np.float32),
            np.array([number(r.get("expected_voice_fake")) for r in rows], dtype=np.float32),
            np.array([number(r.get("expected_music_fake")) for r in rows], dtype=np.float32),
        ],
        axis=1,
    )
    component_mask = np.stack(
        [
            np.ones(len(rows), dtype=np.float32),
            np.array([number(r.get("expected_voice_present")) for r in rows], dtype=np.float32),
            np.array([number(r.get("expected_music_present")) for r in rows], dtype=np.float32),
        ],
        axis=1,
    )
    groups = np.array([r.get("split_group") or r.get("voice_source_id") or r["sample_id"] for r in rows])
    return x, y, component_mask, groups


def masked_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    per = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    # All files supervise direct file fake; components are supervised only when present.
    return (per * mask).sum() / mask.sum().clamp_min(1.0)


def fit_model(
    x_train: np.ndarray, y_train: np.ndarray, m_train: np.ndarray,
    x_val: np.ndarray, y_val: np.ndarray, m_val: np.ndarray,
    *, hidden_dim: int, epochs: int, batch_size: int, lr: float, seed: int, device: str,
):
    torch.manual_seed(seed)
    mean = x_train.mean(axis=0).astype(np.float32)
    std = x_train.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    model = V4FusionHead(x_train.shape[1], hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    rng = np.random.default_rng(seed)
    best_state = None
    best_val = float("inf")
    patience = max(20, epochs // 8)
    stale = 0
    history = []
    x_val_t = torch.from_numpy(x_val).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)
    m_val_t = torch.from_numpy(m_val).to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(x_train))
        losses = []
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            x = torch.from_numpy(x_train[idx]).to(device)
            y = torch.from_numpy(y_train[idx]).to(device)
            m = torch.from_numpy(m_train[idx]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = masked_loss(model(x), y, m)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.inference_mode():
            val_loss = float(masked_loss(model(x_val_t), y_val_t, m_val_t).cpu())
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss})
        scheduler.step()
        if val_loss < best_val - 1e-6:
            best_val, stale = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, mean, std, history


def write_predictions(path: Path, rows: Iterable[dict[str, str]], probabilities: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["sample_id", "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row, p in zip(rows, probabilities, strict=True):
            writer.writerow({"sample_id": row["sample_id"], "FILE_FAKE_PROB": f"{p[0]:.10f}", "VOICE_FAKE_PROB": f"{p[1]:.10f}", "MUSIC_FAKE_PROB": f"{p[2]:.10f}"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--hidden-dim", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=20260902)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if args.folds < 2:
        raise ValueError("--folds must be at least two")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    rows = join_rows(args.manifest, args.features, args.train_split)
    x, y, m, groups = arrays(rows)
    unique_groups = np.unique(groups)
    if len(unique_groups) < args.folds:
        raise ValueError(f"Only {len(unique_groups)} groups for {args.folds} folds")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "feature_contract.json").open("w") as f:
        json.dump({"raw_features": list(RAW_FEATURES), "feature_dim": int(x.shape[1]), "outputs": list(OUTPUT_NAMES), "group_column": "split_group", "component_loss_masks": ["all", "expected_voice_present", "expected_music_present"]}, f, indent=2)
    oof = np.full((len(rows), len(OUTPUT_NAMES)), np.nan, dtype=np.float32)
    histories = {}
    splitter = GroupKFold(n_splits=args.folds)
    for fold, (tr, va) in enumerate(splitter.split(x, y[:, 0], groups), start=1):
        if set(groups[tr]).intersection(groups[va]):
            raise RuntimeError("Split-group leakage detected")
        model, mean, std, history = fit_model(x[tr], y[tr], m[tr], x[va], y[va], m[va], hidden_dim=args.hidden_dim, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed + fold, device=args.device)
        metadata = V4FusionMetadata.create(x.shape[1], args.hidden_dim)
        payload = checkpoint_payload(model, mean, std, metadata, {"fold": fold, "train_samples": int(len(tr)), "validation_samples": int(len(va)), "train_groups": int(len(set(groups[tr]))), "validation_groups": int(len(set(groups[va]))), "history": history})
        torch.save(payload, args.output_dir / f"fold_{fold}.pt")
        values = {name: np.array([number(rows[i].get(name)) for i in va], dtype=np.float32) for name in RAW_FEATURES}
        oof[va] = predict_v4_fusion(model, payload, values, args.device)
        histories[f"fold_{fold}"] = history
    if not np.isfinite(oof).all():
        raise RuntimeError("OOF predictions are incomplete")
    write_predictions(args.output_dir / "oof_predictions.csv", rows, oof)
    with (args.output_dir / "training_history.json").open("w") as f:
        json.dump(histories, f, indent=2)
    print(json.dumps({"output_dir": str(args.output_dir), "samples": len(rows), "groups": int(len(unique_groups)), "folds": args.folds, "oof_predictions": str(args.output_dir / "oof_predictions.csv")}, indent=2))

if __name__ == "__main__":
    main()
