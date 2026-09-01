#!/usr/bin/env python3
"""Cached-score ablation of strongly regularized logistic and ridge fusion."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def load_helpers(path: Path):
    spec = importlib.util.spec_from_file_location("v1_logistic", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate(model, method, rows, details, x, helpers):
    if method == "logistic":
        file_score = model.predict_proba(x)[:, 1]
    else:
        file_score = model.predict(x)
    return helpers.score_metrics(rows, details, file_score)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evalset", type=Path, default=Path("/root/deepvoice-evalset"))
    parser.add_argument("--train-scores", type=Path, required=True)
    parser.add_argument("--heldout-scores", type=Path, required=True)
    parser.add_argument("--helpers", type=Path, default=Path("tools/train_v1_sonics_logistic.py"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    h = load_helpers(args.helpers)
    train_rows, train_details, x_train, y_train = h.load_split(args.evalset, args.train_scores, "train")
    val_rows, val_details, x_val, _ = h.load_split(args.evalset, args.heldout_scores, "validation")
    test_rows, test_details, x_test, _ = h.load_split(args.evalset, args.heldout_scores, "test")
    specs = [("logistic", float(c)) for c in (1e-6, 1e-5, 1e-4, 1e-3, 3e-3, 1e-2)]
    specs += [("ridge", float(a)) for a in (0.01, 0.1, 1, 10, 100, 1000, 10000)]
    report = []
    for method, strength in specs:
        if method == "logistic":
            model = make_pipeline(StandardScaler(), LogisticRegression(C=strength, max_iter=2000, random_state=20260830))
        else:
            model = make_pipeline(StandardScaler(), Ridge(alpha=strength))
        model.fit(x_train, y_train)
        validation = evaluate(model, method, val_rows, val_details, x_val, h)
        test = evaluate(model, method, test_rows, test_details, x_test, h)
        report.append({"method": method, "strength": strength, "validation": validation, "test": test})
    selected = {}
    for method in ("logistic", "ridge"):
        selected[method] = max((r for r in report if r["method"] == method), key=lambda r: r["validation"]["score"])
    output = {"all_variants": report, "selected_by_validation_score": selected}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()
