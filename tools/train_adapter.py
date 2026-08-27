#!/usr/bin/env python3
"""Train adapter heads and fusion calibrators on the evalset.

This script:
  1. Runs the existing zero-shot pipeline on the training split to collect
     component-level scores and backbone features (separate modes).
  2. Trains a voice_fake MLP adapter on DF-Arena voice-stem embeddings.
  3. Trains a music_fake adapter on SONICS music-stem features (optional).
  4. Trains a fusion calibrator (MLP) from 4 component scores → 3 targets.

Outputs are saved to model/fusion_weights.pt (PyTorch state dict)
and model/presence_weights.pt for modified inference.

Usage:
  cd ~/deepvoice-evalset
  .venv311/bin/python3 ~/deepvoice/tools/train_adapter.py \
    --mode voice_adapter --evalset .
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Model definitions (lightweight, meant to be saved and loaded in script.py)
# ---------------------------------------------------------------------------

class VoiceFakeAdapter(nn.Module):
    """Lightweight adapter on top of frozen DF-Arena Wav2Vec2 features.

    Input: 768-dim XLS-R frame-level mean-pooled embeddings (from backbone).
    Output: scalar fake probability.
    """
    def __init__(self, input_dim=768, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze(-1)


class FusionCalibrator(nn.Module):
    """Calibrated fusion from 4 component scores → 3 targets.

    Input: [voice_fake, music_fake, voice_present, music_present] (4-dim)
    Output: [file_fake, voice_fake, music_fake] (3-dim, sigmoided)

    Keeps the component-level predictions as-is but learns nonlinear
    interactions between them (e.g., how much to trust music_fake
    when music_present is low).
    """
    def __init__(self, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, x):
        logits = self.net(x)
        return torch.sigmoid(logits)


class PresenceCalibrator(nn.Module):
    """Calibrated presence probability from PANNs-derived scores.

    Input: raw voice probability, raw music probability (2-dim)
    Output: calibrated voice_present, music_present (2-dim, sigmoided)
    """
    def __init__(self, hidden_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x):
        logits = self.net(x)
        return torch.sigmoid(logits)


# ---------------------------------------------------------------------------
# Dataset utilities
# ---------------------------------------------------------------------------

def parse_bool(v):
    v = v.strip()
    if v == "":
        return 0.0
    return float(v)


def filter_split(manifest_rows, split_name):
    return [r for r in manifest_rows if r.get("split") == split_name]


def load_manifest(manifest_path):
    with open(manifest_path) as f:
        reader = csv.DictReader(f)
        return list(reader)


def build_label_tensor(rows):
    """Return dict of numpy arrays."""
    n = len(rows)
    result = {
        "file_fake": np.zeros(n, dtype=np.float64),
        "voice_fake": np.zeros(n, dtype=np.float64),
        "music_fake": np.zeros(n, dtype=np.float64),
        "voice_present": np.zeros(n, dtype=np.float64),
        "music_present": np.zeros(n, dtype=np.float64),
    }
    for i, r in enumerate(rows):
        result["file_fake"][i] = parse_bool(r.get("expected_file_fake", "0"))
        result["voice_fake"][i] = parse_bool(r.get("expected_voice_fake", "0"))
        result["music_fake"][i] = parse_bool(r.get("expected_music_fake", "0"))
        result["voice_present"][i] = parse_bool(r.get("expected_voice_present", "0"))
        result["music_present"][i] = parse_bool(r.get("expected_music_present", "0"))
    return result


# ---------------------------------------------------------------------------
# Feature extraction from backbone (requires GPU)
# ---------------------------------------------------------------------------

def build_fusion_features_from_pipeline(project_root, evalset, python_exe,
                                         train_rows, val_rows, device="cuda"):
    """Run the existing pipeline on flat symlink dir and cache predictions.
    
    Returns dicts of training & validation features [voice_fake, music_fake,
    voice_present, music_present] and corresponding labels.
    """
    outdir = Path("/tmp/deepvoice_fusion_features")
    outdir.mkdir(parents=True, exist_ok=True)

    # Create flat symlink dirs
    def make_flat_dir(rows, name):
        d = outdir / f"{name}_audio"
        d.mkdir(parents=True, exist_ok=True)
        csv_path = outdir / f"{name}_submission.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
                              "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB"])
            for r in rows:
                sid = r["sample_id"]
                src = evalset / r["local_path"]
                dst = d / f"{sid}.wav"
                if not dst.exists() and src.exists():
                    os.symlink(os.path.abspath(src), dst)
                writer.writerow([sid, 0.0, 0.0, 0.0, 0.0, 0.0])
        return d, csv_path

    for name, rows in [("train", train_rows), ("val", val_rows)]:
        pred_path = outdir / f"predictions_{name}.csv"
        if pred_path.exists():
            print(f"  Using cached predictions: {pred_path}")
            continue

        flat_dir, sub_path = make_flat_dir(rows, name)
        print(f"  Running pipeline on {name} split ({len(rows)} files)...")
        t0 = time.time()
        subprocess.run(
            [python_exe, str(project_root / "script.py"),
             "--test-dir", str(flat_dir),
             "--sample-submission", str(sub_path),
             "--output", str(pred_path),
             "--device", device],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=14400,
        )
        elapsed = time.time() - t0
        print(f"  Completed in {elapsed/60:.1f} min")

    # Load predictions
    result = {}
    for name in ["train", "val"]:
        pred_path = outdir / f"predictions_{name}.csv"
        rows = train_rows if name == "train" else val_rows
        with open(pred_path) as f:
            reader = csv.DictReader(f)
            preds = list(reader)

        features = []
        labels_list = {"file_fake": [], "voice_fake": [], "music_fake": [],
                       "voice_present": [], "music_present": []}
        for p, r in zip(preds, rows):
            features.append([
                float(p.get("VOICE_FAKE_PROB", 0)),
                float(p.get("MUSIC_FAKE_PROB", 0)),
                float(p.get("VOICE_PRESENT_PROB", 0)),
                float(p.get("MUSIC_PRESENT_PROB", 0)),
            ])
            for k in labels_list:
                labels_list[k].append(parse_bool(r.get(f"expected_{k}", "0")))

        result[name] = {
            "features": np.array(features, dtype=np.float32),
            "labels": {k: np.array(v, dtype=np.float32) for k, v in labels_list.items()},
        }
        print(f"  {name}: {len(features)} samples")

    return result


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def train_binary_classifier(model, X, y, epochs=200, lr=1e-3, weight_decay=1e-4,
                             val_X=None, val_y=None, verbose=True):
    """Train a binary classifier with binary cross-entropy."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    X_t = torch.from_numpy(X).to(device)
    y_t = torch.from_numpy(y).to(device).float()

    if val_X is not None and val_y is not None:
        X_v = torch.from_numpy(val_X).to(device)
        y_v = torch.from_numpy(val_y).to(device).float()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float("inf")
    best_state = None
    patience = 30
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        pred = model(X_t)
        loss = F.binary_cross_entropy(pred, y_t)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if val_X is not None:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_v)
                val_loss = F.binary_cross_entropy(val_pred, y_v).item()
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if epoch % 50 == 0 and verbose:
                print(f"  Epoch {epoch:3d} | train_loss={loss.item():.6f} | val_loss={val_loss:.6f}")
            if no_improve >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch}")
                break
        else:
            if epoch % 100 == 0 and verbose:
                print(f"  Epoch {epoch:3d} | loss={loss.item():.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
        if verbose:
            print(f"  Restored best model (val_loss={best_loss:.6f})")

    return model.cpu()


def train_multioutput_classifier(model, X, Y, epochs=200, lr=1e-3,
                                  val_X=None, val_Y=None, verbose=True):
    """Train a multi-output model where Y is a dict of label arrays."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    X_t = torch.from_numpy(X).to(device)

    # Build single target matrix [file_fake, voice_fake, music_fake]
    target_keys = ["file_fake", "voice_fake", "music_fake"]
    y_t = torch.stack([torch.from_numpy(Y[k]).float() for k in target_keys], dim=1).to(device)

    if val_X is not None and val_Y is not None:
        X_v = torch.from_numpy(val_X).to(device)
        y_v = torch.stack([torch.from_numpy(val_Y[k]).float() for k in target_keys], dim=1).to(device)
    else:
        X_v, y_v = None, None

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float("inf")
    best_state = None
    patience = 30
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        pred = model(X_t)
        loss = F.binary_cross_entropy(pred, y_t)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if X_v is not None:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_v)
                val_loss = F.binary_cross_entropy(val_pred, y_v).item()
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if epoch % 50 == 0 and verbose:
                print(f"  Epoch {epoch:3d} | train_loss={loss.item():.6f} | val_loss={val_loss:.6f}")
            if no_improve >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch}")
                break
        else:
            if epoch % 100 == 0 and verbose:
                print(f"  Epoch {epoch:3d} | loss={loss.item():.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
        if verbose:
            print(f"  Restored best model (val_loss={best_loss:.6f})")

    return model.cpu()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_on_val(model, X_val, labels_val, model_type="fusion"):
    """Compute EER and score metrics for a trained model on validation features."""
    from sklearn.metrics import roc_curve, roc_auc_score

    device = torch.device("cpu")
    model = model.to(device).eval()

    with torch.no_grad():
        X_t = torch.from_numpy(X_val).float()
        pred = model(X_t).numpy()

    if model_type == "fusion":
        # pred: (N, 3) — [file_fake, voice_fake, music_fake]
        p_file = pred[:, 0]
        p_voice_fake = pred[:, 1]
        p_music_fake = pred[:, 2]
    else:
        # binary classifier: just voice_fake
        p_file = np.zeros_like(pred)
        p_voice_fake = pred
        p_music_fake = np.zeros_like(pred)

    y_file = labels_val["file_fake"]
    y_voice_fake = labels_val["voice_fake"]
    y_music_fake = labels_val["music_fake"]
    y_voice_present = labels_val["voice_present"]
    y_music_present = labels_val["music_present"]

    # File EER
    fpr, tpr, _ = roc_curve(y_file, p_file, pos_label=1, drop_intermediate=False)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    file_eer = float((fpr[idx] + fnr[idx]) / 2)

    # Voice EER
    voice_mask = y_voice_present > 0.5
    if voice_mask.sum() > 0:
        fpr_v, tpr_v, _ = roc_curve(y_voice_fake[voice_mask], p_voice_fake[voice_mask],
                                      pos_label=1, drop_intermediate=False)
        fnr_v = 1 - tpr_v
        idx_v = np.argmin(np.abs(fpr_v - fnr_v))
        voice_eer = float((fpr_v[idx_v] + fnr_v[idx_v]) / 2)
    else:
        voice_eer = 0.0

    # Music EER
    music_mask = y_music_present > 0.5
    if music_mask.sum() > 0:
        fpr_m, tpr_m, _ = roc_curve(y_music_fake[music_mask], p_music_fake[music_mask],
                                      pos_label=1, drop_intermediate=False)
        fnr_m = 1 - tpr_m
        idx_m = np.argmin(np.abs(fpr_m - fnr_m))
        music_eer = float((fpr_m[idx_m] + fnr_m[idx_m]) / 2)
    else:
        music_eer = 0.0

    ads = 0.5 * (1 - file_eer) + 0.2 * (1 - voice_eer) + 0.3 * (1 - music_eer)
    voice_auc = float(roc_auc_score(y_voice_present, p_voice_fake.clip(0, 1)))
    music_auc = float(roc_auc_score(y_music_present, p_music_fake.clip(0, 1)))
    cps = 0.5 * voice_auc + 0.5 * music_auc
    score = 0.9 * ads + 0.1 * cps

    return {
        "file_eer": file_eer, "voice_eer": voice_eer, "music_eer": music_eer,
        "ads": ads, "voice_auc": voice_auc, "music_auc": music_auc,
        "cps": cps, "score": score,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

MODELS_DIR = Path("model")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="fusion",
                        choices=["fusion", "voice_adapter", "presence", "all"])
    parser.add_argument("--evalset", default="/root/deepvoice-evalset")
    parser.add_argument("--project", default="/root/deepvoice")
    parser.add_argument("--venv-python", default=".venv311/bin/python3")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    evalset = Path(args.evalset).resolve()
    project_root = Path(args.project).resolve()
    python_exe = os.path.abspath(str(project_root / args.venv_python))
    models_dir = project_root / MODELS_DIR

    # Load manifest
    manifest = load_manifest(evalset / "manifests" / "manifest.csv")
    train_rows = filter_split(manifest, "train")
    val_rows = filter_split(manifest, "validation")
    print(f"Train: {len(train_rows)} files, Val: {len(val_rows)} files")

    # Build labels
    train_labels = build_label_tensor(train_rows)
    val_labels = build_label_tensor(val_rows)

    if args.mode in ("fusion", "all"):
        print(f"\n{'='*60}")
        print("Mode: FUSION CALIBRATOR — train MLP on 4 component scores")
        print(f"{'='*60}")

        # We need to first run the pipeline on train/val splits to get component scores
        # This is expensive, so cache is critical
        fusion_data = build_fusion_features_from_pipeline(
            project_root, evalset, python_exe,
            train_rows, val_rows, args.device,
        )

        # Train fusion model
        print("  Training fusion calibrator...")
        model = FusionCalibrator(hidden_dim=16)
        trained = train_multioutput_classifier(
            model,
            fusion_data["train"]["features"],
            fusion_data["train"]["labels"],
            val_X=fusion_data["val"]["features"],
            val_Y=fusion_data["val"]["labels"],
            epochs=500, lr=1e-3,
        )

        # Evaluate
        metrics = evaluate_on_val(trained,
                                    fusion_data["val"]["features"],
                                    val_labels, model_type="fusion")
        print(f"\n  Fusion Calibrator Validation:")
        print(f"    File EER:  {metrics['file_eer']:.6f}")
        print(f"    Voice EER: {metrics['voice_eer']:.6f}")
        print(f"    Music EER: {metrics['music_eer']:.6f}")
        print(f"    ADS:       {metrics['ads']:.6f}")
        print(f"    CPS:       {metrics['cps']:.6f}")
        print(f"    SCORE:     {metrics['score']:.6f}")

        # Save
        save_path = models_dir / "fusion_calibrator.pt"
        torch.save(trained.state_dict(), save_path)
        print(f"  Saved to {save_path}")

    if args.mode in ("voice_adapter", "all"):
        print(f"\n{'='*60}")
        print("Mode: VOICE ADAPTER — train MLP on DF-Arena embeddings")
        print(f"{'='*60}")
        print("  (Not yet implemented — requires backbone forward pass)")
        # TODO: extract XLS-R embeddings from DF-Arena backbone
        # train VoiceFakeAdapter
        pass

    if args.mode in ("presence", "all"):
        print(f"\n{'='*60}")
        print("Mode: PRESENCE CALIBRATOR — train MLP on PANNs scores")
        print(f"{'='*60}")
        # Use same fusion features: voice_present, music_present
        # Train calibration on (presence features) → (calibrated presence)
        print("  (Requires fusion feature pass first; run --mode fusion first)")


if __name__ == "__main__":
    main()