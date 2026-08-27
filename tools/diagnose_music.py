#!/usr/bin/env python3
"""Analyze music detection errors by domain and source dataset."""
import csv
import json
import sys
from collections import defaultdict
import numpy as np

PREDS_FILE = sys.argv[1]  # predictions CSV
MANIFEST_FILE = sys.argv[2]  # manifest CSV

# Load predictions
with open(PREDS_FILE) as f:
    preds = {p["ID"]: p for p in csv.DictReader(f)}

# Load manifest
with open(MANIFEST_FILE) as f:
    rows = [r for r in csv.DictReader(f) if r["split"] == "validation"]

print(f"Validation rows: {len(rows)}")
print(f"Predictions loaded: {len(preds)}")

# Analyze per domain
domains = defaultdict(list)
sources = defaultdict(list)
vocal_modes = defaultdict(list)

for r in rows:
    sid = r["sample_id"]
    p = preds.get(sid)
    if p is None:
        continue

    y_file = float(r.get("expected_file_fake", "0") or 0)
    y_voice = float(r.get("expected_voice_fake", "0") or 0)
    y_music = float(r.get("expected_music_fake", "0") or 0)
    y_voice_pre = float(r.get("expected_voice_present", "0") or 0)
    y_music_pre = float(r.get("expected_music_present", "0") or 0)

    p_file = float(p["FILE_FAKE_PROB"])
    p_voice = float(p["VOICE_FAKE_PROB"])
    p_music = float(p["MUSIC_FAKE_PROB"])

    domain = r.get("audio_domain", "unknown")
    source = r.get("source_dataset", "unknown")
    vmode = r.get("vocal_mode", "unknown")

    err = abs(p_file - y_file)
    music_err = abs(p_music - y_music) if y_music_pre > 0.5 else None

    item = {
        "sid": sid, "domain": domain, "source": source, "vmode": vmode,
        "y_file": y_file, "y_music": y_music, "y_voice": y_voice,
        "y_music_pre": y_music_pre, "y_voice_pre": y_voice_pre,
        "p_file": p_file, "p_music": p_music, "p_voice": p_voice,
        "music_err": music_err, "file_err": err,
    }
    domains[domain].append(item)
    sources[source].append(item)
    vocal_modes[vmode].append(item)

print("\n=== BY DOMAIN ===")
print(f"{'Domain':<20} {'N':>5} {'FileEER':>10} {'MusicEER':>10} {'FileFP':>8} {'FileFN':>8} {'MusicFP':>8} {'MusicFN':>8}")
print("-" * 80)

for domain, items in sorted(domains.items()):
    n = len(items)
    # File-level: threshold at 0.5
    fp = sum(1 for x in items if x["p_file"] >= 0.5 and x["y_file"] < 0.5)
    fn = sum(1 for x in items if x["p_file"] < 0.5 and x["y_file"] >= 0.5)
    # Music-level (only where music present)
    music_items = [x for x in items if x["y_music_pre"] >= 0.5]
    music_fp = sum(1 for x in music_items if x["p_music"] >= 0.5 and x["y_music"] < 0.5) if music_items else -1
    music_fn = sum(1 for x in music_items if x["p_music"] < 0.5 and x["y_music"] >= 0.5) if music_items else -1

    print(f"{domain:<20} {n:>5} {'-':>10} {'-':>10} {fp:>8} {fn:>8} {str(music_fp):>8} {str(music_fn):>8}")

print("\n=== BY SOURCE DATASET (music_present files only) ===")
print(f"{'Source':<35} {'N':>5} {'MusicFP':>8} {'MusicFN':>8} {'AvgMusicPred(real)':>18} {'AvgMusicPred(fake)':>18}")
print("-" * 90)

for source, items in sorted(sources.items()):
    music_items = [x for x in items if x["y_music_pre"] >= 0.5]
    n_music = len(music_items)
    if n_music == 0:
        continue
    music_fp = sum(1 for x in music_items if x["p_music"] >= 0.5 and x["y_music"] < 0.5)
    music_fn = sum(1 for x in music_items if x["p_music"] < 0.5 and x["y_music"] >= 0.5)

    real_items = [x for x in music_items if x["y_music"] < 0.5]
    fake_items = [x for x in music_items if x["y_music"] >= 0.5]
    avg_real = np.mean([x["p_music"] for x in real_items]) if real_items else -1
    avg_fake = np.mean([x["p_music"] for x in fake_items]) if fake_items else -1

    print(f"{source:<35} {n_music:>5} {music_fp:>8} {music_fn:>8} {avg_real:>18.4f} {avg_fake:>18.4f}")

# Also check the cross-contamination hypothesis:
# For speech-only files, what does p_music look like (should be 0)?
speech_only = [x for x in domains.get("speech", []) if x["y_music_pre"] < 0.5]
if speech_only:
    avg_music_pred_on_speech = np.mean([x["p_music"] for x in speech_only])
    max_music_pred_on_speech = max(x["p_music"] for x in speech_only)
    print(f"\n=== CROSS-CONTAMINATION CHECK ===")
    print(f"Speech-only files with music pred: N={len(speech_only)}")
    print(f"  Mean MUSIC_FAKE_PROB on speech-only: {avg_music_pred_on_speech:.4f}")
    print(f"  Max MUSIC_FAKE_PROB on speech-only: {max_music_pred_on_speech:.4f}")

# For music-only files, what does p_voice look like?
music_only = [x for x in domains.get("music", []) if x["y_voice_pre"] < 0.5]
if music_only:
    avg_voice_pred_on_music = np.mean([x["p_voice"] for x in music_only])
    max_voice_pred_on_music = max(x["p_voice"] for x in music_only)
    print(f"Music-only files with voice pred: N={len(music_only)}")
    print(f"  Mean VOICE_FAKE_PROB on music-only: {avg_voice_pred_on_music:.4f}")
    print(f"  Max VOICE_FAKE_PROB on music-only: {max_voice_pred_on_music:.4f}")

# Show individual misclassified music files
print("\n=== MUSIC MISCLASSIFICATIONS (FileEER errors) ===")
errors = [x for x in rows if x["sample_id"] in preds]
for r in errors:
    sid = r["sample_id"]
    p = preds[sid]
    yf = float(r.get("expected_file_fake", "0") or 0)
    pf = float(p["FILE_FAKE_PROB"])
    if abs(pf - yf) > 0.5:
        print(f"  {sid:40s} true={yf:.0f} pred={pf:.4f} "
              f"src={r.get('source_dataset','?'):20s} dom={r.get('audio_domain','?'):15s} "
              f"mf={r.get('expected_music_fake','?'):>4s} pm={p['MUSIC_FAKE_PROB']:>6s} "
              f"vf={r.get('expected_voice_fake','?'):>4s} pv={p['VOICE_FAKE_PROB']:>6s}")