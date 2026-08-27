#!/usr/bin/env python3
"""Evaluate 3 model variants on the validation split of the evalset.

Variants:
  baseline     — DF-Arena for voice + music, max temporal agg
  v1_original  — DF-Arena for voice, SONICS for music, max temporal agg
  v1_q90       — DF-Arena for voice, SONICS for music, q90 temporal agg

Usage:
  cd ~/deepvoice
  .venv311/bin/python3 tools/evaluate_validation.py \
    --evalset ~/deepvoice-evalset \
    --project . \
    --outdir eval_results \
    --venv-python .venv311/bin/python3
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score


# ---------------------------------------------------------------------------
# Metrics — exact competition definitions
# ---------------------------------------------------------------------------

def compute_eer(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1, drop_intermediate=False)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    eer = float((fpr[idx] + fnr[idx]) / 2)
    return eer


def compute_score(preds, labels):
    y_file = np.array([r["file_true"] for r in labels], dtype=np.float64)
    y_voice_fake = np.array([r["voice_fake_true"] for r in labels], dtype=np.float64)
    y_music_fake = np.array([r["music_fake_true"] for r in labels], dtype=np.float64)
    y_voice_present = np.array([r["voice_present_true"] for r in labels], dtype=np.float64)
    y_music_present = np.array([r["music_present_true"] for r in labels], dtype=np.float64)

    p_file = np.array([r["FILE_FAKE_PROB"] for r in preds], dtype=np.float64)
    p_voice_fake = np.array([r["VOICE_FAKE_PROB"] for r in preds], dtype=np.float64)
    p_music_fake = np.array([r["MUSIC_FAKE_PROB"] for r in preds], dtype=np.float64)
    p_voice_present = np.array([r["VOICE_PRESENT_PROB"] for r in preds], dtype=np.float64)
    p_music_present = np.array([r["MUSIC_PRESENT_PROB"] for r in preds], dtype=np.float64)

    # File EER
    file_eer = compute_eer(y_file, p_file)

    # Voice EER — only among samples where voice is present
    voice_mask = y_voice_present > 0.5
    if voice_mask.sum() > 0:
        voice_eer = compute_eer(y_voice_fake[voice_mask], p_voice_fake[voice_mask])
    else:
        voice_eer = 0.0

    # Music EER — only among samples where music is present
    music_mask = y_music_present > 0.5
    if music_mask.sum() > 0:
        music_eer = compute_eer(y_music_fake[music_mask], p_music_fake[music_mask])
    else:
        music_eer = 0.0

    # ADS
    ads = 0.5 * (1 - file_eer) + 0.2 * (1 - voice_eer) + 0.3 * (1 - music_eer)

    # CPS
    voice_auc = roc_auc_score(y_voice_present, p_voice_present)
    music_auc = roc_auc_score(y_music_present, p_music_present)
    cps = 0.5 * voice_auc + 0.5 * music_auc

    score = 0.9 * ads + 0.1 * cps

    return {
        "file_eer": float(file_eer),
        "voice_eer": float(voice_eer),
        "music_eer": float(music_eer),
        "ads": float(ads),
        "voice_auc": float(voice_auc),
        "music_auc": float(music_auc),
        "cps": float(cps),
        "score": float(score),
        "n_file": len(y_file),
        "n_voice": int(voice_mask.sum()),
        "n_music": int(music_mask.sum()),
    }


# ---------------------------------------------------------------------------
# File-based variant management — no git dependency
# ---------------------------------------------------------------------------

ORIGINAL_SUFFIX = ".eval_original"
BACKUP_FILES = [
    "script.py",
    "model/temporal_aggregation.py",
    "model/sonics_infer.py",
]


def backup_originals(project_root):
    """Save original files with .eval_original suffix."""
    for rel_path in BACKUP_FILES:
        src = project_root / rel_path
        dst = project_root / (rel_path + ORIGINAL_SUFFIX)
        if dst.exists():
            dst.unlink()
        shutil.copy2(src, dst)
        print(f"  Backed up: {rel_path}")


def restore_originals(project_root):
    """Restore original files from .eval_original suffix (keep backup)."""
    for rel_path in BACKUP_FILES:
        src = project_root / (rel_path + ORIGINAL_SUFFIX)
        dst = project_root / rel_path
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  Restored: {rel_path}")


def write_file_content(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


# ---------------------------------------------------------------------------
# Variant configurations — exact file contents
# ---------------------------------------------------------------------------

def apply_variant(project_root, variant):
    """Write variant-specific file content to the project."""
    if variant == "baseline":
        # temporal_aggregation.py: max agg
        write_file_content(
            project_root / "model/temporal_aggregation.py",
            '''"""Temporal aggregation utilities for detector window scores."""

import numpy as np


def aggregate_temporal_scores(scores, quantile=0.90, default=0.0):
    """Aggregate window probabilities with a robust upper quantile."""
    values = np.asarray(scores, dtype=np.float64)
    if values.size == 0:
        return float(default)
    return float(np.max(values))
''',
        )

        # sonics_infer.py: max agg
        content = (project_root / "model/sonics_infer.py").read_text()
        content = content.replace(
            "    return aggregate_temporal_scores(scores, default=0.0)",
            "    return max(scores) if scores else 0.0",
        )
        (project_root / "model/sonics_infer.py").write_text(content)

        # script.py: no SONICS, DF-Arena for music, max agg for both
        content = (project_root / "script.py").read_text()
        # Remove SONICS import
        content = content.replace(
            "# v1: 로컬 SONICS 추론 모듈 (awsaf49/sonics-spectttra-alpha-5s).\nfrom sonics_infer import load_sonics_model, predict_fake as predict_sonics_fake\nfrom temporal_aggregation import aggregate_temporal_scores\n",
            "# no-op: baseline has no SONICS\n",
        )
        # Remove SONICS_DIR
        content = content.replace(
            "SONICS_DIR = LOCAL_MODEL_DIR / \"sonics\"\n",
            "# baseline: no SONICS_DIR\n",
        )
        # Change predict_fake to use max instead of aggregate
        content = content.replace(
            "    return aggregate_temporal_scores(segment_scores)",
            "    return max(segment_scores) if segment_scores else 0.0",
        )
        # Remove SONICS model loading
        content = content.replace(
            "    # v1: 음악 성분 전용 AI 생성 음악 탐지기 (SONICS) 를 추가로 로드한다.\n    sonics_model = load_sonics_model(SONICS_DIR, device)",
            "    sonics_model = None  # baseline: no SONICS",
        )
        # Change music_fake to use DF-Arena
        content = content.replace(
            "        music_fake = predict_sonics_fake(sonics_model, music_audio, AUDIO_SAMPLE_RATE, device=device)",
            "        music_fake = predict_fake(\n            df_arena_model, fake_label_index, music_audio, device\n        )",
        )
        (project_root / "script.py").write_text(content)
        print("  Applied: baseline variant")

    elif variant == "v1_original":
        # temporal_aggregation.py: max agg
        write_file_content(
            project_root / "model/temporal_aggregation.py",
            '''"""Temporal aggregation utilities for detector window scores."""

import numpy as np


def aggregate_temporal_scores(scores, quantile=0.90, default=0.0):
    """Aggregate window probabilities with a robust upper quantile."""
    values = np.asarray(scores, dtype=np.float64)
    if values.size == 0:
        return float(default)
    return float(np.max(values))
''',
        )

        # sonics_infer.py: max agg
        content = (project_root / "model/sonics_infer.py").read_text()
        content = content.replace(
            "    return aggregate_temporal_scores(scores, default=0.0)",
            "    return max(scores) if scores else 0.0",
        )
        (project_root / "model/sonics_infer.py").write_text(content)
        print("  Applied: v1_original variant")

    elif variant == "v1_q90":
        # q90 is the current state — nothing to change
        print("  Applied: v1_q90 variant (no changes needed)")
    else:
        raise ValueError(f"Unknown variant: {variant}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evalset", default=".", help="Path to deepvoice-evalset root")
    parser.add_argument("--project", default=".", help="Path to deepvoice project root")
    parser.add_argument("--outdir", default="eval_results", help="Output directory")
    parser.add_argument("--venv-python", default="python3",
                        help="Python interpreter (e.g. .venv311/bin/python3)")
    parser.add_argument("--variants", nargs="+",
                        default=["baseline", "v1_original", "v1_q90"],
                        choices=["baseline", "v1_original", "v1_q90"])
    args = parser.parse_args()

    evalset = Path(args.evalset).resolve()
    project_root = Path(args.project).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    python_exe = os.path.abspath(str(project_root / args.venv_python))

    # Read manifest
    manifest_path = evalset / "manifests" / "manifest.csv"
    with manifest_path.open() as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    val_rows = [r for r in all_rows if r["split"] == "validation"]
    print(f"Validation samples: {len(val_rows)}")

    # Build label dicts
    labels = []
    for r in val_rows:
        def parse_bool(v):
            v = v.strip()
            if v == "":
                return 0.0
            return float(v)

        labels.append({
            "sample_id": r["sample_id"],
            "file_true": parse_bool(r.get("expected_file_fake", "0")),
            "voice_fake_true": parse_bool(r.get("expected_voice_fake", "0")),
            "music_fake_true": parse_bool(r.get("expected_music_fake", "0")),
            "voice_present_true": parse_bool(r.get("expected_voice_present", "0")),
            "music_present_true": parse_bool(r.get("expected_music_present", "0")),
        })

    # Create flat symlink test directory and sample submission
    flat_dir = outdir / "validation_audio"
    flat_dir.mkdir(parents=True, exist_ok=True)

    sample_submission_path = outdir / "validation_submission.csv"
    with sample_submission_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
                          "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB"])
        for r in val_rows:
            sid = r["sample_id"]
            src = evalset / r["local_path"]
            if not src.exists():
                print(f"  WARNING: file not found: {src}")
            dst = flat_dir / f"{sid}.wav"
            if not dst.exists():
                os.symlink(os.path.abspath(src), dst)
            writer.writerow([sid, 0.0, 0.0, 0.0, 0.0, 0.0])

    print(f"Flat test dir: {flat_dir} ({len(val_rows)} files)")
    print(f"Sample submission: {sample_submission_path}")

    # Backup originals
    print("\nBacking up original files...")
    backup_originals(project_root)

    try:
        results = {}
        for variant in args.variants:
            print(f"\n{'='*60}")
            print(f"Variant: {variant}")
            print(f"{'='*60}")

            # Restore originals first, then apply variant
            print("  Restoring originals...")
            restore_originals(project_root)

            print("  Applying variant patches...")
            apply_variant(project_root, variant)

            # Verify the key difference
            if variant == "baseline":
                with open(project_root / "script.py") as f:
                    c = f.read()
                uses_max = "max(segment_scores)" in c
                uses_sonics = "predict_sonics_fake" in c
                print(f"  Verified: script.py uses max()={uses_max}, has SONICS={uses_sonics}")

            # Run inference
            out_csv = outdir / f"predictions_{variant}.csv"
            print("  Running inference (this will take a while)...")
            result = subprocess.run(
                [
                    python_exe, str(project_root / "script.py"),
                    "--test-dir", str(flat_dir),
                    "--sample-submission", str(sample_submission_path),
                    "--output", str(out_csv),
                    "--device", "cuda",
                ],
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=7200,
            )
            print(result.stdout.strip())
            if result.stderr:
                stderr_lines = result.stderr.strip().split("\n")
                for line in stderr_lines[-20:]:
                    print(f"  STDERR: {line}")

            if result.returncode != 0:
                print(f"  FAILED with exit code {result.returncode}")
                error_detail = result.stderr[-500:] if result.stderr else "no stderr"
                results[variant] = {"error": error_detail}
                continue

            # Read predictions
            with out_csv.open() as f:
                pred_reader = csv.DictReader(f)
                preds = list(pred_reader)

            print(f"  Predictions: {len(preds)} rows")

            # Compute metrics
            metrics = compute_score(preds, labels)
            results[variant] = metrics

            print(f"  File EER:      {metrics['file_eer']:.6f}")
            print(f"  Voice EER:     {metrics['voice_eer']:.6f}")
            print(f"  Music EER:     {metrics['music_eer']:.6f}")
            print(f"  ADS:           {metrics['ads']:.6f}")
            print(f"  Voice AUC:     {metrics['voice_auc']:.6f}")
            print(f"  Music AUC:     {metrics['music_auc']:.6f}")
            print(f"  CPS:           {metrics['cps']:.6f}")
            print(f"  Score:         {metrics['score']:.6f}")

    finally:
        # Always restore originals
        print("\nRestoring original files...")
        restore_originals(project_root)

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Variant':<16} {'Score':>8} {'ADS':>8} {'CPS':>8} {'FileEER':>8} {'VoiceEER':>8} {'MusicEER':>8}")
    print("-" * 64)
    for variant in args.variants:
        r = results.get(variant, {})
        if "error" in r:
            print(f"{variant:<16} ERROR: {r['error'][:40]}")
        else:
            print(f"{variant:<16} {r['score']:>8.6f} {r['ads']:>8.6f} {r['cps']:>8.6f} "
                  f"{r['file_eer']:>8.6f} {r['voice_eer']:>8.6f} {r['music_eer']:>8.6f}")

    # Save full results
    summary = {variant: results.get(variant, {}) for variant in args.variants}
    with (outdir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFull results saved to {outdir / 'summary.json'}")


if __name__ == "__main__":
    main()
