#!/usr/bin/env python3
"""V4 robustness report from prediction and immutable manifest files.

Reports deployment-oriented aggregate metrics, requested domain/channel/
generator partitions where those attributes exist, detector rank correlations,
and pairwise file-score error overlap. Groups with one class only are retained
with unavailable EER rather than silently discarded.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def num(value): return 0.0 if value is None or str(value).strip() == "" else float(value)


def eer(y, p):
    y, p = np.asarray(y), np.asarray(p)
    if len(y) < 2 or len(np.unique(y)) < 2: return None
    fpr, tpr, _ = roc_curve(y, p, pos_label=1, drop_intermediate=False)
    fnr = 1.0 - tpr
    i = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[i] + fnr[i]) / 2.0)


def auc(y, p):
    return None if len(np.unique(y)) < 2 else float(roc_auc_score(y, p))


def metrics(rows):
    y_file = np.array([num(r["expected_file_fake"]) for r in rows])
    y_voice = np.array([num(r["expected_voice_fake"]) for r in rows])
    y_music = np.array([num(r["expected_music_fake"]) for r in rows])
    vp = np.array([num(r["expected_voice_present"]) for r in rows])
    mp = np.array([num(r["expected_music_present"]) for r in rows])
    pf = np.array([num(r["FILE_FAKE_PROB"]) for r in rows])
    pv = np.array([num(r["VOICE_FAKE_PROB"]) for r in rows])
    pm = np.array([num(r["MUSIC_FAKE_PROB"]) for r in rows])
    pvp = np.array([num(r.get("VOICE_PRESENT_PROB")) for r in rows])
    pmp = np.array([num(r.get("MUSIC_PRESENT_PROB")) for r in rows])
    file_eer, voice_eer, music_eer = eer(y_file, pf), eer(y_voice[vp > .5], pv[vp > .5]), eer(y_music[mp > .5], pm[mp > .5])
    voice_auc, music_auc = auc(vp, pvp), auc(mp, pmp)
    ads = None if None in (file_eer, voice_eer, music_eer) else .5 * (1-file_eer) + .2 * (1-voice_eer) + .3 * (1-music_eer)
    cps = None if None in (voice_auc, music_auc) else .5 * (voice_auc + music_auc)
    score = None if None in (ads, cps) else .9 * ads + .1 * cps
    return {"n": len(rows), "file_positive": int(y_file.sum()), "voice_present": int(vp.sum()), "music_present": int(mp.sum()), "file_eer": file_eer, "voice_eer": voice_eer, "music_eer": music_eer, "ads": ads, "cps": cps, "score": score}


def group_report(rows, key, min_n):
    report = {}
    for group in sorted({r.get(key, "") or "<missing>" for r in rows}):
        selected = [r for r in rows if (r.get(key, "") or "<missing>") == group]
        if len(selected) >= min_n: report[group] = metrics(selected)
    return report


def tag_report(rows, min_n):
    tags = Counter()
    row_tags = []
    for r in rows:
        current = tuple(t.strip() for t in (r.get("augmentation", "") or "none").split("|") if t.strip()) or ("none",)
        row_tags.append(current); tags.update(current)
    result = {}
    for tag, count in sorted(tags.items()):
        if count >= min_n: result[tag] = metrics([r for r, current in zip(rows, row_tags) if tag in current])
    return result


def correlations(rows):
    candidates = ["df_raw", "df_voice", "sonics_stem", "artifact_raw", "artifact_stem", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB", "FILE_FAKE_PROB"]
    columns = [name for name in candidates if all(name in r and str(r[name]).strip() != "" for r in rows)]
    if len(columns) < 2: return {}
    x = np.stack([[num(r[name]) for r in rows] for name in columns])
    matrix = np.corrcoef(x)
    return {a: {b: float(matrix[i, j]) for j, b in enumerate(columns)} for i, a in enumerate(columns)}


def error_overlap(rows):
    y = np.array([num(r["expected_file_fake"]) for r in rows])
    candidates = ["df_raw", "df_voice", "sonics_stem", "artifact_raw", "artifact_stem", "FILE_FAKE_PROB"]
    available = [name for name in candidates if all(name in r and str(r[name]).strip() != "" for r in rows)]
    if not available or len(np.unique(y)) < 2: return {}
    errors = {}
    for name in available:
        p = np.array([num(r[name]) for r in rows])
        threshold = .5
        errors[name] = p >= threshold
        errors[name] = errors[name] != y.astype(bool)
    return {a: {b: float(np.mean(errors[a] & errors[b])) for b in available} for a in available}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--split", default=None)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--min-group-size", type=int, default=20)
    args = ap.parse_args()
    with args.manifest.open(newline="") as f: manifest = {r["sample_id"]: r for r in csv.DictReader(f)}
    with args.predictions.open(newline="") as f: predictions = list(csv.DictReader(f))
    joined = []
    for p in predictions:
        sample_id = p.get("sample_id") or p.get("ID")
        if sample_id not in manifest: raise KeyError(f"Prediction ID not in manifest: {sample_id}")
        row = {**manifest[sample_id], **p}; row["sample_id"] = sample_id
        if args.split is None or row.get("split") == args.split: joined.append(row)
    if not joined: raise ValueError("No rows after manifest/prediction join and split filter")
    domains = group_report(joined, "audio_domain", args.min_group_size)
    families = group_report(joined, "voice_family", args.min_group_size)
    report = {"n": len(joined), "split": args.split, "aggregate": metrics(joined), "by_audio_domain": domains, "by_voice_family_or_generator": families, "by_augmentation": tag_report(joined, args.min_group_size), "score_rank_correlation": correlations(joined), "threshold_0_5_file_error_overlap": error_overlap(joined)}
    mixed = [r for r in joined if r.get("audio_domain") == "mixed"]
    report["mixed"] = metrics(mixed) if mixed else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f: json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps(report["aggregate"], indent=2))

if __name__ == "__main__": main()
