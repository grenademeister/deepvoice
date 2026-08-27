#!/usr/bin/env python3
"""
End-to-end training pipeline for DeepVoice.

Phases:
  1. Run script.py on training split → collect 4-dim component features
  2. Train fusion calibrator (4→3 MLP)
  3. Evaluate on validation split
  4. Save model/fusion_calibrator.pt

Usage:
  cd ~/deepvoice
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
    .venv311/bin/python3 tools/train_full.py \\
      --evalset ~/deepvoice-evalset \\
      --device cuda
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_curve, roc_auc_score


# ---------------------------------------------------------------------------
# Fusion Calibrator
# ---------------------------------------------------------------------------

class FusionCalibrator(nn.Module):
    def __init__(self, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_eer(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1, drop_intermediate=False)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2)


def compute_score(preds, labels):
    yf = np.array([r["file_true"] for r in labels])
    yvf = np.array([r["voice_fake_true"] for r in labels])
    ymf = np.array([r["music_fake_true"] for r in labels])
    yvp = np.array([r["voice_present_true"] for r in labels])
    ymp = np.array([r["music_present_true"] for r in labels])

    pf = np.array([r["FILE_FAKE_PROB"] for r in preds])
    pvf = np.array([r["VOICE_FAKE_PROB"] for r in preds])
    pmf = np.array([r["MUSIC_FAKE_PROB"] for r in preds])
    pvp = np.array([r["VOICE_PRESENT_PROB"] for r in preds])
    pmp = np.array([r["MUSIC_PRESENT_PROB"] for r in preds])

    file_eer = compute_eer(yf, pf)
    voice_mask = yvp > 0.5
    voice_eer = compute_eer(yvf[voice_mask], pvf[voice_mask]) if voice_mask.sum() > 0 else 0.0
    music_mask = ymp > 0.5
    music_eer = compute_eer(ymf[music_mask], pmf[music_mask]) if music_mask.sum() > 0 else 0.0

    ads = 0.5 * (1 - file_eer) + 0.2 * (1 - voice_eer) + 0.3 * (1 - music_eer)
    voice_auc = float(roc_auc_score(yvp, pvp))
    music_auc = float(roc_auc_score(ymp, pmp))
    cps = 0.5 * voice_auc + 0.5 * music_auc
    score = 0.9 * ads + 0.1 * cps

    return {"file_eer": file_eer, "voice_eer": voice_eer, "music_eer": music_eer,
            "ads": ads, "voice_auc": voice_auc, "music_auc": music_auc,
            "cps": cps, "score": score,
            "n_file": len(yf), "n_voice": int(voice_mask.sum()), "n_music": int(music_mask.sum())}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_fusion(model, X_train, Y_train, X_val, Y_val, epochs=500, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    Xt = torch.from_numpy(X_train).to(device)
    yt = torch.stack([torch.from_numpy(Y_train[k]).float() for k in ["file_fake", "voice_fake", "music_fake"]], dim=1).to(device)
    Xv = torch.from_numpy(X_val).to(device)
    yv = torch.stack([torch.from_numpy(Y_val[k]).float() for k in ["file_fake", "voice_fake", "music_fake"]], dim=1).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_loss = float("inf")
    best_state = None
    patience = 50
    no_improv = 0

    for epoch in range(epochs):
        model.train()
        loss = F.binary_cross_entropy(model(Xt), yt)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            vl = F.binary_cross_entropy(model(Xv), yv).item()
        if vl < best_loss:
            best_loss = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improv = 0
        else:
            no_improv += 1
        if epoch % 100 == 0:
            print(f"  Epoch {epoch:4d} | train_loss={loss.item():.6f} | val_loss={vl:.6f}")
        if no_improv >= patience:
            print(f"  Early stopping epoch {epoch}")
            break

    model.load_state_dict(best_state)
    return model.cpu()


def parse_float(v):
    v = v.strip()
    return 0.0 if v == "" else float(v)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evalset", default="/root/deepvoice-evalset")
    parser.add_argument("--project", default="/root/deepvoice")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--venv-python", default=".venv311/bin/python3")
    args = parser.parse_args()

    evalset = Path(args.evalset).resolve()
    project = Path(args.project).resolve()
    python_exe = os.path.abspath(str(project / args.venv_python))
    models_dir = project / "model"
    outdir = project / "eval_results"
    outdir.mkdir(parents=True, exist_ok=True)

    # Load manifest
    with open(evalset / "manifests" / "manifest.csv") as f:
        all_rows = list(csv.DictReader(f))
    train_rows = [r for r in all_rows if r["split"] == "train"]
    val_rows = [r for r in all_rows if r["split"] == "validation"]
    print(f"Train: {len(train_rows)}, Val: {len(val_rows)}")

    # ==================================================================
    # PHASE 1: Run pipeline on train split → collect component features
    # ==================================================================
    print("\n" + "=" * 60)
    print("PHASE 1: Pipeline feature extraction on training split")
    print("=" * 60)

    # Create flat symlink dir + submission CSV
    def prepare_split(rows, name):
        d = Path("/tmp/deepvoice_train") / f"{name}_audio"
        d.mkdir(parents=True, exist_ok=True)
        csv_path = Path("/tmp/deepvoice_train") / f"{name}_submission.csv"

        # Only write if not already done
        if csv_path.exists():
            return d, csv_path

        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ID", "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
                        "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB"])
            for r in rows:
                sid = r["sample_id"]
                src = evalset / r["local_path"]
                dst = d / f"{sid}.wav"
                if not dst.exists() and src.exists():
                    os.symlink(os.path.abspath(src), dst)
                w.writerow([sid, 0.0, 0.0, 0.0, 0.0, 0.0])
        return d, csv_path

    train_audio_dir, train_sub_path = prepare_split(train_rows, "train")
    val_audio_dir, val_sub_path = prepare_split(val_rows, "val")

    # Run pipeline
    for name, rows, audio_dir, sub_path in [
        ("train", train_rows, train_audio_dir, train_sub_path),
        ("val", val_rows, val_audio_dir, val_sub_path),
    ]:
        pred_path = Path("/tmp/deepvoice_train") / f"predictions_{name}.csv"
        if pred_path.exists():
            print(f"  Cached predictions exist: {pred_path}")
            continue

        print(f"  Running pipeline on {name} ({len(rows)} files)...")
        t0 = time.time()
        result = subprocess.run(
            [python_exe, str(project / "script.py"),
             "--test-dir", str(audio_dir),
             "--sample-submission", str(sub_path),
             "--output", str(pred_path),
             "--device", args.device],
            capture_output=True, text=True, timeout=7200,
        )
        elapsed = time.time() - t0
        print(f"  Completed in {elapsed/60:.1f} min, exit={result.returncode}")
        if result.returncode != 0:
            # Print last 20 lines of stderr
            for line in result.stderr.strip().split("\n")[-20:]:
                print(f"  ERR: {line}")
            raise RuntimeError(f"Pipeline failed on {name} split")

    # ==================================================================
    # PHASE 2: Load predictions + train fusion calibrator
    # ==================================================================
    print("\n" + "=" * 60)
    print("PHASE 2: Train fusion calibrator")
    print("=" * 60)

    def load_predictions_and_labels(pred_path, rows):
        with open(pred_path) as f:
            preds = list(csv.DictReader(f))
        features = []
        labels = {"file_fake": [], "voice_fake": [], "music_fake": [],
                  "voice_present": [], "music_present": []}
        for p, r in zip(preds, rows):
            features.append([float(p["VOICE_FAKE_PROB"]), float(p["MUSIC_FAKE_PROB"]),
                             float(p["VOICE_PRESENT_PROB"]), float(p["MUSIC_PRESENT_PROB"])])
            for k in labels:
                labels[k].append(parse_float(r.get(f"expected_{k}", "0")))
        return np.array(features, dtype=np.float32), {k: np.array(v, dtype=np.float32) for k, v in labels.items()}

    X_train, Y_train = load_predictions_and_labels(
        Path("/tmp/deepvoice_train") / "predictions_train.csv", train_rows)
    X_val, Y_val = load_predictions_and_labels(
        Path("/tmp/deepvoice_train") / "predictions_val.csv", val_rows)

    print(f"  Train features: {X_train.shape}, Val features: {X_val.shape}")

    model = FusionCalibrator(hidden_dim=16)
    train_fusion(model, X_train, Y_train, X_val, Y_val)

    # ==================================================================
    # PHASE 3: Evaluate calibrator on validation set
    # ==================================================================
    print("\n" + "=" * 60)
    print("PHASE 3: Evaluate fusion calibrator")
    print("=" * 60)

    # Apply calibrator to original validation predictions
    with torch.no_grad():
        Xv_t = torch.from_numpy(X_val)
        cal_preds = model(Xv_t).numpy()

    # Build calibrated prediction dicts for scoring
    cal_rows = []
    with open(Path("/tmp/deepvoice_train") / "predictions_val.csv") as f:
        orig = list(csv.DictReader(f))
    for i, o in enumerate(orig):
        cal_rows.append({
            "FILE_FAKE_PROB": float(cal_preds[i, 0]),
            "VOICE_FAKE_PROB": X_val[i, 0],  # keep original voice_fake
            "MUSIC_FAKE_PROB": X_val[i, 1],  # keep original music_fake
            "VOICE_PRESENT_PROB": X_val[i, 2],
            "MUSIC_PRESENT_PROB": X_val[i, 3],
        })

    # Build label dicts
    val_labels = []
    for i, r in enumerate(val_rows):
        val_labels.append({
            "file_true": parse_float(r.get("expected_file_fake", "0")),
            "voice_fake_true": parse_float(r.get("expected_voice_fake", "0")),
            "music_fake_true": parse_float(r.get("expected_music_fake", "0")),
            "voice_present_true": parse_float(r.get("expected_voice_present", "0")),
            "music_present_true": parse_float(r.get("expected_music_present", "0")),
        })

    # Baseline (original pipeline)
    base_preds = []
    with open(Path("/tmp/deepvoice_train") / "predictions_val.csv") as f:
        for r in csv.DictReader(f):
            base_preds.append({
                "FILE_FAKE_PROB": float(r["FILE_FAKE_PROB"]),
                "VOICE_FAKE_PROB": float(r["VOICE_FAKE_PROB"]),
                "MUSIC_FAKE_PROB": float(r["MUSIC_FAKE_PROB"]),
                "VOICE_PRESENT_PROB": float(r["VOICE_PRESENT_PROB"]),
                "MUSIC_PRESENT_PROB": float(r["MUSIC_PRESENT_PROB"]),
            })
    base_metrics = compute_score(base_preds, val_labels)

    # Calibrated
    cal_metrics = compute_score(cal_rows, val_labels)

    print(f"\n  {'Metric':<15} {'Baseline':>10} {'Calibrated':>12} {'Change':>10}")
    print("-" * 50)
    for k in ["score", "ads", "cps", "file_eer", "voice_eer", "music_eer"]:
        b = base_metrics[k]
        c = cal_metrics[k]
        change = c - b
        print(f"  {k:<15} {b:>10.6f} {c:>12.6f} {change:>+10.6f}")

    # ==================================================================
    # PHASE 4: Save calibrator weights
    # ==================================================================
    save_path = models_dir / "fusion_calibrator.pt"
    torch.save(model.state_dict(), save_path)
    print(f"\n  Calibrator saved to {save_path}")
    print(f"  Size: {save_path.stat().st_size / 1024:.1f} KB")

    # Print the threshold-based comparison too
    base_fp = sum(1 for r, l in zip(base_preds, val_labels)
                  if r["FILE_FAKE_PROB"] >= 0.5 and l["file_true"] < 0.5)
    base_fn = sum(1 for r, l in zip(base_preds, val_labels)
                  if r["FILE_FAKE_PROB"] < 0.5 and l["file_true"] >= 0.5)
    cal_fp = sum(1 for r, l in zip(cal_rows, val_labels)
                 if r["FILE_FAKE_PROB"] >= 0.5 and l["file_true"] < 0.5)
    cal_fn = sum(1 for r, l in zip(cal_rows, val_labels)
                 if r["FILE_FAKE_PROB"] < 0.5 and l["file_true"] >= 0.5)

    print(f"\n  Threshold (0.5) classification:")
    print(f"  {'':15} {'Baseline':>10} {'Calibrated':>12}")
    print(f"  {'FP':>15} {base_fp:>10} {cal_fp:>12}")
    print(f"  {'FN':>15} {base_fn:>10} {cal_fn:>12}")
    print(f"  {'Accuracy':>15} {(144-base_fp-base_fn)/144*100:>9.1f}% {(144-cal_fp-cal_fn)/144*100:>11.1f}%")


if __name__ == "__main__":
    main()
