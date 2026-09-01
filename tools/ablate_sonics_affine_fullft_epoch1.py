#!/usr/bin/env python3
"""Grid-search affine SONICS score transforms using cached V1 scores only."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np


def load_helpers(path):
    spec = importlib.util.spec_from_file_location("v1_logistic", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_split(evalset, score_dir, split):
    rows = [r for r in csv.DictReader((evalset / "manifests/manifest_balanced.csv").open()) if r["split_balanced"] == split]
    by_id = {r["ID"]: r for r in csv.DictReader((score_dir / f"v1_sonics_{split}_scores.csv").open())}
    details = [by_id[r["sample_id"]] for r in rows]
    return rows, details


def file_score(details, slope, intercept):
    voice = np.array([float(d["voice"]) * float(d["voice_present"]) for d in details])
    music_raw = np.array([float(d["adapted_music"]) for d in details])
    mp = np.array([float(d["music_present"]) for d in details])
    music = mp * np.clip(slope * music_raw + intercept, 0.0, 1.0)
    return np.maximum(voice, music)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evalset", type=Path, default=Path("/root/deepvoice-evalset"))
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--train-scores", type=Path, required=True)
    parser.add_argument("--helpers", type=Path, default=Path("tools/train_v1_sonics_logistic.py"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    h = load_helpers(args.helpers)
    train_rows, train_details = load_split(args.evalset, args.train_scores, "train")
    val_rows, val_details = load_split(args.evalset, args.scores, "validation")
    test_rows, test_details = load_split(args.evalset, args.scores, "test")
    slopes = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    intercepts = [-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
    records = []
    for slope in slopes:
        for intercept in intercepts:
            train = h.score_metrics(train_rows, train_details, file_score(train_details, slope, intercept))
            val = h.score_metrics(val_rows, val_details, file_score(val_details, slope, intercept))
            test = h.score_metrics(test_rows, test_details, file_score(test_details, slope, intercept))
            records.append({"slope": slope, "intercept": intercept, "train": train, "validation": val, "test": test})
    selected = max(records, key=lambda r: r["train"]["score"])
    baseline = next(r for r in records if r["slope"] == 1.0 and r["intercept"] == 0.0)
    out = {"formula": "max(vp*df_voice, mp*clip(slope*sonics_fullft_epoch1 + intercept, 0, 1))", "selected_by_train_score": selected, "identity_baseline": baseline, "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"selected": selected, "identity": baseline}, indent=2))

if __name__ == "__main__":
    main()
