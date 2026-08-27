#!/usr/bin/env python3
"""Fine-tune SONICS SpecTTTra on evalset music stems to fix OOD gap on GTZAN.

Usage:
  cd ~/deepvoice
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
    .venv311/bin/python3 tools/finetune_sonics.py \\
      --evalset ~/deepvoice-evalset \\
      --device cuda
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram

# Add model directory to path for sonics_infer
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model"))

from sonics_infer import SonicsClassifier, FeatureExtractor, SpecTTTra


# ---------------------------------------------------------------------------
# SEPARATION: extract music stems via HTDemucs
# ---------------------------------------------------------------------------

def separate_music_stem(audio_path, htdemucs_model, device, sr=16000):
    """Run HTDemucs and return only the music stem (non-vocal sum)."""
    from demucs.apply import apply_model
    from demucs.separate import load_track

    waveform = load_track(str(audio_path), htdemucs_model.audio_channels, htdemucs_model.samplerate).float()
    mono = waveform.mean(0)
    mean = mono.mean()
    std = mono.std()

    if float(std) < 1e-8:
        return np.zeros(0, dtype=np.float32)

    norm = (waveform - mean) / std
    with torch.inference_mode():
        sources = apply_model(htdemucs_model, norm[None], device=device,
                              shifts=0, split=True, overlap=0.25, progress=False)[0]
    sources = sources * std + mean

    music_idxs = [i for i, name in enumerate(htdemucs_model.sources) if name != "vocals"]
    music = torch.stack([sources[i] for i in music_idxs]).sum(0).mean(0, keepdim=True)
    music = torchaudio.functional.resample(music, htdemucs_model.samplerate, sr)[0]
    return music.cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MusicStemDataset(torch.utils.data.Dataset):
    """Dataset of music stems with fake/real labels."""
    def __init__(self, rows, evalset_root, htdemucs_model, device, sr=16000, max_len=80000):
        self.items = []
        self.sr = sr
        self.max_len = max_len

        for r in rows:
            src = evalset_root / r["local_path"]
            label = float(r.get("expected_music_fake", "0").strip() or 0)
            self.items.append((str(src), label))

        self.model = htdemucs_model
        self.device = device
        self.cache = {}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        if path in self.cache:
            audio = self.cache[path]
        else:
            audio = separate_music_stem(path, self.model, self.device, self.sr)
            self.cache[path] = audio

        # Standard SONICS preprocessing
        n = audio.shape[0]
        if n >= self.max_len:
            idx_crop = int((n - self.max_len) / 4 * 3)
            audio = audio[idx_crop : idx_crop + self.max_len]
        else:
            audio = np.pad(audio, (0, self.max_len - n), mode="constant")
        audio = audio / max(float(np.std(audio)), 1e-6)
        audio = audio.astype(np.float32)

        return torch.from_numpy(audio), torch.tensor([label], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evalset", default="/root/deepvoice-evalset")
    parser.add_argument("--project", default="/root/deepvoice")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    evalset = Path(args.evalset).resolve()
    project = Path(args.project).resolve()
    models_dir = project / "model"
    sonics_dir = models_dir / "sonics"
    device = torch.device(args.device)

    # ==================================================================
    # PHASE 1: Load manifest + filter music files
    # ==================================================================
    print("=" * 60)
    print("PHASE 1: Dataset preparation")
    print("=" * 60)

    with open(evalset / "manifests" / "manifest.csv") as f:
        all_rows = list(csv.DictReader(f))

    def parse_float(v):
        v = v.strip()
        return 0.0 if v == "" else float(v)

    # Music-containing files: only where expected_music_present >= 0.5
    music_rows = [r for r in all_rows
                  if parse_float(r.get("expected_music_present", "0")) >= 0.5]
    print(f"Music files: {len(music_rows)}")

    # Filter by split
    train_music = [r for r in music_rows if r["split"] == "train"]
    val_music = [r for r in music_rows if r["split"] == "validation"]
    print(f"Train music: {len(train_music)}, Val music: {len(val_music)}")

    # Load HTDemucs
    print("Loading HTDemucs for stem extraction...")
    from demucs.pretrained import get_model
    original_load = torch.load
    def trusted_load(*a, **kw):
        kw.setdefault("weights_only", False)
        return original_load(*a, **kw)
    torch.load = trusted_load
    try:
        htdemucs = get_model("htdemucs", repo=models_dir / "htdemucs")
    finally:
        torch.load = original_load
    htdemucs = htdemucs.cpu().eval()

    # ==================================================================
    # PHASE 2: Extract music stems (this is the slow part)
    # ==================================================================
    print("\n" + "=" * 60)
    print("PHASE 2: Music stem extraction + training")
    print("=" * 60)

    train_dataset = MusicStemDataset(train_music, evalset, htdemucs, device)
    val_dataset = MusicStemDataset(val_music, evalset, htdemucs, device)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=False)

    # ==================================================================
    # PHASE 3: Load SONICS + fine-tune
    # ==================================================================
    print("\n" + "=" * 60)
    print("PHASE 3: Fine-tuning SONICS SpecTTTra")
    print("=" * 60)

    with open(sonics_dir / "config.json") as f:
        cfg = json.load(f)

    model = SonicsClassifier(cfg)
    state = torch.load(sonics_dir / "pytorch_model.bin", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model = model.to(device).train()

    # Full fine-tuning: all layers trainable (370 music files is enough)
    for param in model.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Full fine-tune: {trainable:,} / {total:,} params ({trainable/total*100:.1f}%)")

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.lr, weight_decay=0.05)
    best_val = float("inf")
    patience = 5
    no_improv = 0

    for epoch in range(args.epochs):
        # Train
        model.train()
        train_losses = []
        t0 = time.time()
        for audio, label in train_loader:
            audio, label = audio.to(device), label.to(device)
            logit = model(audio)
            loss = F.binary_cross_entropy_with_logits(logit, label)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            train_losses.append(loss.item())

        # Validate
        model.eval()
        val_losses = []
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for audio, label in val_loader:
                audio, label = audio.to(device), label.to(device)
                logit = model(audio)
                loss = F.binary_cross_entropy_with_logits(logit, label)
                val_losses.append(loss.item())
                val_preds.extend(torch.sigmoid(logit).cpu().numpy().flatten())
                val_labels.extend(label.cpu().numpy().flatten())

        t1 = time.time()
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        # Compute val accuracy at 0.5 threshold
        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels)
        correct = ((val_preds >= 0.5) == (val_labels >= 0.5)).mean()

        print(f"Epoch {epoch+1:2d}/{args.epochs} | "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_acc={correct:.4f} time={t1-t0:.0f}s")

        if val_loss < best_val:
            best_val = val_loss
            no_improv = 0
            # Save best checkpoint
            torch.save(model.state_dict(), models_dir / "sonics_finetuned.pt")
            print(f"  -> Saved best checkpoint (val_loss={val_loss:.4f})")
        else:
            no_improv += 1
            if no_improv >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # ==================================================================
    # PHASE 4: Update SONICS weights and evaluate
    # ==================================================================
    print("\n" + "=" * 60)
    print("PHASE 4: Restore best checkpoint + evaluate")
    print("=" * 60)

    # Copy the fine-tuned weights to the SONICS model directory
    # This way the regular script.py loads them
    import shutil
    finetuned_path = models_dir / "sonics_finetuned.pt"
    original_path = sonics_dir / "pytorch_model.bin"

    # Save original for rollback
    backup_path = sonics_dir / "pytorch_model_original.bin"
    if not backup_path.exists():
        shutil.copy2(original_path, backup_path)
        print(f"Backed up original to {backup_path}")

    # Overwrite with fine-tuned
    shutil.copy2(finetuned_path, original_path)
    print(f"Updated {original_path} with fine-tuned weights")

    # Run pipeline on validation split and evaluate
    print("\nRunning evaluation with fine-tuned SONICS...")
    import subprocess
    out_csv = project / "eval_results" / "predictions_sonics_ft.csv"
    result = subprocess.run(
        [str(project / ".venv311/bin/python3"), str(project / "script.py"),
         "--test-dir", str(project / "eval_results" / "validation_audio"),
         "--sample-submission", str(project / "eval_results" / "validation_submission.csv"),
         "--output", str(out_csv),
         "--device", "cuda"],
        capture_output=True, text=True, timeout=7200,
    )
    print(result.stdout[-200:] if len(result.stdout) > 200 else result.stdout)

    # Compute metrics (using same logic as tools/compute_metrics.py)
    from sklearn.metrics import roc_curve, roc_auc_score

    def compute_eer(y_true, y_score):
        fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1, drop_intermediate=False)
        fnr = 1 - tpr
        idx = np.argmin(np.abs(fpr - fnr))
        return float((fpr[idx] + fnr[idx]) / 2)

    def compute_metrics(preds, labels_dict):
        def to_float(v):
            try: return float(v)
            except: return 0.0
        yf = np.array([to_float(l["file_true"]) for l in labels_dict])
        yvf = np.array([to_float(l["voice_fake_true"]) for l in labels_dict])
        ymf = np.array([to_float(l["music_fake_true"]) for l in labels_dict])
        yvp = np.array([to_float(l["voice_present_true"]) for l in labels_dict])
        ymp = np.array([to_float(l["music_present_true"]) for l in labels_dict])
        pf = np.array([to_float(p["FILE_FAKE_PROB"]) for p in preds])
        pvf = np.array([to_float(p["VOICE_FAKE_PROB"]) for p in preds])
        pmf = np.array([to_float(p["MUSIC_FAKE_PROB"]) for p in preds])
        pvp = np.array([to_float(p["VOICE_PRESENT_PROB"]) for p in preds])
        pmp = np.array([to_float(p["MUSIC_PRESENT_PROB"]) for p in preds])
        file_eer = compute_eer(yf, pf)
        vm = yvp > 0.5
        veer = compute_eer(yvf[vm], pvf[vm]) if vm.sum() > 0 else 0.0
        mm = ymp > 0.5
        meer = compute_eer(ymf[mm], pmf[mm]) if mm.sum() > 0 else 0.0
        ads = 0.5*(1-file_eer) + 0.2*(1-veer) + 0.3*(1-meer)
        vauc = float(roc_auc_score(yvp, pvp))
        mauc = float(roc_auc_score(ymp, pmp))
        cps = 0.5*vauc + 0.5*mauc
        score = 0.9*ads + 0.1*cps
        return {"file_eer": file_eer, "voice_eer": veer, "music_eer": meer,
                "ads": ads, "voice_auc": vauc, "music_auc": mauc,
                "cps": cps, "score": score,
                "n_file": len(yf), "n_voice": int(vm.sum()), "n_music": int(mm.sum())}

    with open(out_csv) as f:
        preds = list(csv.DictReader(f))
    with open(evalset / "manifests" / "manifest.csv") as f:
        manifest_rows = {r["sample_id"]: r for r in csv.DictReader(f)}

    labels = []
    for p in preds:
        r = manifest_rows[p["ID"]]
        labels.append({
            "file_true": parse_float(r.get("expected_file_fake", "0")),
            "voice_fake_true": parse_float(r.get("expected_voice_fake", "0")),
            "music_fake_true": parse_float(r.get("expected_music_fake", "0")),
            "voice_present_true": parse_float(r.get("expected_voice_present", "0")),
            "music_present_true": parse_float(r.get("expected_music_present", "0")),
        })

    metrics = compute_metrics(preds, labels)
    metrics_path = project / "eval_results" / "sonics_ft_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSONICS Fine-tuned Results:")
    print(f"  Score:   {metrics['score']:.6f}")
    print(f"  FileEER: {metrics['file_eer']:.6f}")
    print(f"  MusicEER: {metrics['music_eer']:.6f}")
    print(f"  VoiceEER: {metrics['voice_eer']:.6f}")
    print(f"  ADS:     {metrics['ads']:.6f}")
    print(f"  CPS:     {metrics['cps']:.6f}")
    print(f"\nResults saved to {metrics_path}")

    # Compare with baseline
    baseline_metrics_path = project / "eval_results" / "summary.json"
    if baseline_metrics_path.exists():
        with open(baseline_metrics_path) as f:
            baseline = json.load(f).get("baseline", {})
        if "file_eer" in baseline:
            print(f"\n  Comparison with baseline:")
            print(f"  {'':>10} {'Baseline':>10} {'SonFT':>10} {'Change':>10}")
            for k in ["file_eer", "music_eer", "voice_eer", "score"]:
                b = baseline.get(k, 0)
                c = metrics.get(k, 0)
                print(f"  {k:>10} {b:>10.6f} {c:>10.6f} {c-b:>+10.6f}")


if __name__ == "__main__":
    main()