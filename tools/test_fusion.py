#!/usr/bin/env python3
"""Test different fusion strategies on existing predictions."""
import csv, json, sys
import numpy as np
from sklearn.metrics import roc_curve

PREDS = sys.argv[1]  # predictions CSV
MANIFEST = sys.argv[2]  # evalset manifest

with open(PREDS) as f:
    preds = list(csv.DictReader(f))

with open(MANIFEST) as f:
    rows = {r["sample_id"]: r for r in csv.DictReader(f) if r["split"] == "validation"}

def pf(v):
    v=v.strip(); return 0.0 if v=="" else float(v)

# True labels
yf = np.array([pf(rows[p["ID"]].get("expected_file_fake","0")) for p in preds])
yvp = np.array([pf(rows[p["ID"]].get("expected_voice_present","0")) for p in preds])
ymp = np.array([pf(rows[p["ID"]].get("expected_music_present","0")) for p in preds])
yvf = np.array([pf(rows[p["ID"]].get("expected_voice_fake","0")) for p in preds])
ymf = np.array([pf(rows[p["ID"]].get("expected_music_fake","0")) for p in preds])

# Predictions
vf = np.array([float(p["VOICE_FAKE_PROB"]) for p in preds])
mf = np.array([float(p["MUSIC_FAKE_PROB"]) for p in preds])
vp = np.array([float(p["VOICE_PRESENT_PROB"]) for p in preds])
mp = np.array([float(p["MUSIC_PRESENT_PROB"]) for p in preds])

def compute_eer(yt, yp):
    fpr, tpr, _ = roc_curve(yt, yp, pos_label=1, drop_intermediate=False)
    fnr = 1-tpr
    idx = np.argmin(np.abs(fpr-fnr))
    return (fpr[idx]+fnr[idx])/2

def score(file_eer, voice_eer, music_eer, voice_auc, music_auc):
    ads = 0.5*(1-file_eer) + 0.2*(1-voice_eer) + 0.3*(1-music_eer)
    cps = 0.5*voice_auc + 0.5*music_auc
    return 0.9*ads + 0.1*cps, ads, cps

# Reference metrics (assuming voice/music EER/AUC unchanged by fusion)
vm = yvp > 0.5
voice_eer = compute_eer(yvf[vm], vf[vm]) if vm.sum()>0 else 0.0
mm = ymp > 0.5
music_eer = compute_eer(ymf[mm], mf[mm]) if mm.sum()>0 else 0.0
from sklearn.metrics import roc_auc_score
voice_auc = float(roc_auc_score(yvp, vp))
music_auc = float(roc_auc_score(ymp, mp))

strategies = {
    "max(vp*vf, mp*mf)": np.maximum(vp*vf, mp*mf),
    "max(vf, mf)": np.maximum(vf, mf),
    "mean(vf, mf)": (vf+mf)/2,
    "max(vp, mp)*0.7": np.maximum(vp, mp)*0.7,
    "vp*vf + mp*mf": vp*vf + mp*mf,
    "noisy_or": vp*vf + mp*mf - vp*vf*mp*mf,
    "max_smart": np.where(vp>mp, vp*vf, mp*mf),
    "vp*vf": vp*vf,
    "mp*mf": mp*mf,
    "vf": vf,
    "mf": mf,
}

print(f"{'Strategy':<25} {'FileEER':>8} {'Score':>8}")
print("-"*45)
for name, p_file in strategies.items():
    eer = compute_eer(yf, p_file)
    s, ads, cps = score(eer, voice_eer, music_eer, voice_auc, music_auc)
    print(f"{name:<25} {eer:>8.4f} {s:>8.4f}")