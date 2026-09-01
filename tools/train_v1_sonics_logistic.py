#!/usr/bin/env python3
"""Train a file-level logistic fusion calibrator for V1+head1 SONICS."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EPS = 1e-5
C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)


def value(row, key):
    return float(row.get(key) or 0.0)


def logit(p):
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def eer(y, score):
    fpr, tpr, _ = roc_curve(y, score, pos_label=1, drop_intermediate=False)
    fnr = 1.0 - tpr
    return float(((fpr + fnr) / 2.0)[np.argmin(np.abs(fpr - fnr))])


def score_metrics(rows, details, file_score):
    y_file = np.array([value(r, "expected_file_fake") for r in rows])
    y_voice = np.array([value(r, "expected_voice_fake") for r in rows])
    y_music = np.array([value(r, "expected_music_fake") for r in rows])
    y_vp = np.array([value(r, "expected_voice_present") for r in rows])
    y_mp = np.array([value(r, "expected_music_present") for r in rows])
    voice = np.array([value(d, "voice") for d in details])
    music = np.array([value(d, "adapted_music") for d in details])
    vp = np.array([value(d, "voice_present") for d in details])
    mp = np.array([value(d, "music_present") for d in details])
    file_eer = eer(y_file, file_score)
    voice_eer = eer(y_voice[y_vp > .5], voice[y_vp > .5])
    music_eer = eer(y_music[y_mp > .5], music[y_mp > .5])
    cps = .5 * roc_auc_score(y_vp, vp) + .5 * roc_auc_score(y_mp, mp)
    ads = .5 * (1 - file_eer) + .2 * (1 - voice_eer) + .3 * (1 - music_eer)
    mixed = np.array([r["audio_domain"] == "mixed" for r in rows])
    return {"n": len(rows), "score": .9 * ads + .1 * cps, "ads": ads, "cps": cps,
            "file_eer": file_eer, "file_eer_mixed": eer(y_file[mixed], file_score[mixed]),
            "voice_eer": voice_eer, "music_eer": music_eer}


def load_split(evalset, score_dir, split):
    rows = [r for r in csv.DictReader((evalset / "manifests/manifest_balanced.csv").open()) if r["split_balanced"] == split]
    details_by_id = {r["ID"]: r for r in csv.DictReader((score_dir / f"v1_sonics_{split}_scores.csv").open())}
    if {r["sample_id"] for r in rows} != set(details_by_id):
        raise ValueError(f"Mismatched manifest/scores IDs for {split}")
    details = [details_by_id[r["sample_id"]] for r in rows]
    event_voice = np.array([value(d, "voice") * value(d, "voice_present") for d in details])
    event_music = np.array([value(d, "adapted_music") * value(d, "music_present") for d in details])
    X = np.column_stack((logit(event_voice), logit(event_music)))
    y = np.array([value(r, "expected_file_fake") for r in rows], dtype=int)
    return rows, details, X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evalset", type=Path, default=Path("/root/deepvoice-evalset"))
    parser.add_argument("--train-scores", type=Path, required=True)
    parser.add_argument("--heldout-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train_rows, train_details, x_train, y_train = load_split(args.evalset, args.train_scores, "train")
    val_rows, val_details, x_val, _ = load_split(args.evalset, args.heldout_scores, "validation")
    test_rows, test_details, x_test, _ = load_split(args.evalset, args.heldout_scores, "test")
    candidates = []
    for c in C_GRID:
        model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000, random_state=20260830))
        model.fit(x_train, y_train)
        val_prob = model.predict_proba(x_val)[:, 1]
        candidates.append((score_metrics(val_rows, val_details, val_prob)["score"], c, model))
    _, c, model = max(candidates, key=lambda x: x[0])
    result = {"features": ["logit(vp*df_voice)", "logit(mp*sonics_head_epoch1)"], "C_grid": list(C_GRID), "selected_C_by_validation_score": c,
              "train": score_metrics(train_rows, train_details, model.predict_proba(x_train)[:, 1]),
              "validation": score_metrics(val_rows, val_details, model.predict_proba(x_val)[:, 1]),
              "test": score_metrics(test_rows, test_details, model.predict_proba(x_test)[:, 1]),
              "coefficients_standardized": model[-1].coef_[0].tolist(), "intercept": float(model[-1].intercept_[0])}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
