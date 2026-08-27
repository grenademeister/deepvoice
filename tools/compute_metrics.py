#!/usr/bin/env python3
"""Compute competition metrics from script.py predictions + ground truth manifest."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score


def compute_eer(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1, drop_intermediate=False)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2)


def compute_score(preds, labels):
    """preds: list of dicts with FILE_FAKE_PROB, VOICE_FAKE_PROB, etc.
       labels: list of dicts with file_true, voice_fake_true, etc."""
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

    file_eer = compute_eer(y_file, p_file)
    voice_mask = y_voice_present > 0.5
    voice_eer = compute_eer(y_voice_fake[voice_mask], p_voice_fake[voice_mask]) if voice_mask.sum() > 0 else 0.0
    music_mask = y_music_present > 0.5
    music_eer = compute_eer(y_music_fake[music_mask], p_music_fake[music_mask]) if music_mask.sum() > 0 else 0.0

    ads = 0.5 * (1 - file_eer) + 0.2 * (1 - voice_eer) + 0.3 * (1 - music_eer)
    voice_auc = float(roc_auc_score(y_voice_present, p_voice_present))
    music_auc = float(roc_auc_score(y_music_present, p_music_present))
    cps = 0.5 * voice_auc + 0.5 * music_auc
    score = 0.9 * ads + 0.1 * cps

    return {
        "file_eer": file_eer, "voice_eer": voice_eer, "music_eer": music_eer,
        "ads": ads, "voice_auc": voice_auc, "music_auc": music_auc,
        "cps": cps, "score": score,
        "n_file": len(y_file), "n_voice": int(voice_mask.sum()), "n_music": int(music_mask.sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    with open(args.predictions) as f:
        preds = list(csv.DictReader(f))

    with open(args.manifest) as f:
        rows = {r["sample_id"]: r for r in csv.DictReader(f)}

    def parse_float(v):
        v = v.strip()
        return 0.0 if v == "" else float(v)

    file_ids = [p["ID"] for p in preds]
    labels = []
    for pid in file_ids:
        r = rows[pid]
        labels.append({
            "file_true": parse_float(r.get("expected_file_fake", "0")),
            "voice_fake_true": parse_float(r.get("expected_voice_fake", "0")),
            "music_fake_true": parse_float(r.get("expected_music_fake", "0")),
            "voice_present_true": parse_float(r.get("expected_voice_present", "0")),
            "music_present_true": parse_float(r.get("expected_music_present", "0")),
        })

    pred_dicts = []
    for p in preds:
        pred_dicts.append({
            "FILE_FAKE_PROB": float(p["FILE_FAKE_PROB"]),
            "VOICE_FAKE_PROB": float(p["VOICE_FAKE_PROB"]),
            "MUSIC_FAKE_PROB": float(p["MUSIC_FAKE_PROB"]),
            "VOICE_PRESENT_PROB": float(p["VOICE_PRESENT_PROB"]),
            "MUSIC_PRESENT_PROB": float(p["MUSIC_PRESENT_PROB"]),
        })

    result = compute_score(pred_dicts, labels)
    print(json.dumps(result, indent=2))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()