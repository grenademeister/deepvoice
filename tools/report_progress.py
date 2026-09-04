#!/usr/bin/env python3
"""Generate plain-language progress + loss graph for V5."""
import json, re, csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUN = Path("/root/deepvoice/runs/v5_20260904")
LOG = RUN / "run.log"
OUT_IMG = Path("/tmp/v5_loss.png")

# parse log
batches = []  # list of (batch_idx, loss, cache)
epochs = []
if LOG.exists():
    for line in LOG.read_text().splitlines():
        line=line.strip()
        if not line: continue
        try:
            j=json.loads(line)
        except: continue
        if j.get("phase")=="train_batch":
            batches.append((j["batch"], j["loss"], j.get("cache_files",0)))
        elif j.get("phase")=="epoch_start":
            epochs.append(j["epoch"])
        elif "train_loss" in j and "file_eer" in j:
            epochs.append(j)

cache_files = 0
if (RUN/"sonics_cache").exists():
    cache_files = len(list((RUN/"sonics_cache").glob("*.npz")))

# GPU
gpu = "unknown"
try:
    import subprocess
    out=subprocess.check_output(["nvidia-smi","--query-gpu=memory.used,memory.total,utilization.gpu","--format=csv,noheader"], text=True)
    gpu=out.strip()
except: pass

# plot
if batches:
    xs = [b[0] for b in batches]
    ys = [b[1] for b in batches]
    plt.figure(figsize=(10,4))
    plt.plot(xs, ys, linewidth=1.2)
    # rolling mean
    if len(ys)>20:
        rm=np.convolve(ys, np.ones(20)/20, mode='valid')
        plt.plot(xs[19:], rm, linewidth=1.5, alpha=0.6, label="rolling20")
        plt.legend()
    plt.xlabel("Batch (1 = 16 samples)")
    plt.ylabel("Loss")
    plt.title(f"V5 Training Loss — {len(batches)} batches, cache {cache_files} files, GPU {gpu}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_IMG, dpi=140)
    plt.close()
    print(f"PLOT {OUT_IMG} {OUT_IMG.stat().st_size} bytes")
else:
    # still create placeholder
    plt.figure(figsize=(8,3))
    plt.text(0.5,0.5,"No batch logs yet — still in epoch 1", ha='center', va='center')
    plt.axis('off')
    plt.savefig(OUT_IMG, dpi=120)
    plt.close()

# plain language summary
if batches:
    last = batches[-1]
    first = batches[0]
    trend = "decreasing" if last[1] < first[1] else "increasing"
    summary = f"V5 Training — {len(epochs)} epoch(s) started, {len(batches)} batches logged. Last batch {last[0]} loss {last[1]:.4f} ({trend} from {first[1]:.4f}). Cache {cache_files}/80000 sonics files on disk ({cache_files/800:.1f}%). GPU {gpu}. {'Validation metrics not yet (still epoch 1)' if not any('file_eer' in str(e) for e in epochs) else 'See metrics.json for validation.'}"
else:
    summary = f"V5 Training started — no batch loss yet (still processing first batches). Cache {cache_files} files. GPU {gpu}."

print(summary)
# also dump last 5 losses for cron
for b in batches[-5:]:
    print(f"batch {b[0]} loss {b[1]:.4f} cache {b[2]}")
