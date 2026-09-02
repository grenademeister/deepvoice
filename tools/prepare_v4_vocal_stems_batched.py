#!/usr/bin/env python3
"""Cache HTDemucs vocal stems for V4 DF-Arena feature extraction.

This reproduces the deployed separator preprocessing: mono -> duplicate model
channels -> per-file mean/std normalization -> HTDemucs split inference ->
vocal source -> mono 16 kHz PCM. Existing valid outputs are reused.

Every batch emits JSONL telemetry: elapsed time, throughput, stem statistics,
CUDA allocation/reservation, label/domain counts, and failures.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from demucs.apply import apply_model

from prepare_sonics_htdemucs_music_stems import load_audio_16k, load_htdemucs, to_model_channels


def valid_cached(path: Path, expected_samples: int) -> bool:
    try:
        audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
        return int(sr) == 16000 and np.asarray(audio).reshape(-1).shape[0] == expected_samples and bool(np.isfinite(audio).all())
    except Exception:
        return False


def cuda_stats(device: str) -> dict[str, int]:
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return {"cuda_allocated_bytes": 0, "cuda_reserved_bytes": 0, "cuda_max_allocated_bytes": 0}
    return {
        "cuda_allocated_bytes": int(torch.cuda.memory_allocated()),
        "cuda_reserved_bytes": int(torch.cuda.memory_reserved()),
        "cuda_max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    rows = [r for r in csv.DictReader(args.manifest.open()) if int(r.get("expected_voice_present") or 0)]
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        raise ValueError("No voice-present manifest rows")
    args.output.mkdir(parents=True, exist_ok=True)
    log_path = args.output / "prepare_vocal_batched.jsonl"
    expected = int(round(16000 * args.duration))
    started = time.time()
    failures: list[dict[str, str]] = []
    complete: list[dict[str, str]] = []
    torch.cuda.reset_peak_memory_stats() if str(args.device).startswith("cuda") and torch.cuda.is_available() else None
    model = load_htdemucs(Path("/root/deepvoice/model/htdemucs")).to(args.device).eval()
    vocal_index = model.sources.index("vocals")
    with log_path.open("a", buffering=1) as log:
        def emit(record: dict) -> None:
            record["timestamp_unix"] = time.time()
            record["elapsed_seconds"] = record["timestamp_unix"] - started
            record.update(cuda_stats(args.device))
            log.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps(record, sort_keys=True), flush=True)

        emit({"event": "start", "total": len(rows), "batch_size": args.batch_size, "duration_seconds": args.duration, "separator_sources": list(model.sources), "target": "vocals"})
        for start in range(0, len(rows), args.batch_size):
            batch_started = time.time()
            batch = rows[start:start + args.batch_size]
            reusable, pending = [], []
            for row in batch:
                target = args.output / "audio" / row["split"] / f"{row['sample_id']}.wav"
                if valid_cached(target, expected):
                    item = dict(row); item["filepath"] = str(target.resolve()); reusable.append(item)
                else:
                    pending.append(row)
            complete.extend(reusable)
            output_stats = {"rms_mean": None, "peak_max": None}
            try:
                if pending:
                    tensors = []
                    for row in pending:
                        audio = load_audio_16k(Path(row["local_path"]), model.samplerate, args.duration)
                        tensors.append(to_model_channels(torch.from_numpy(audio), model.audio_channels))
                    waveform = torch.stack(tensors).to(args.device)
                    mean = waveform.mean((1, 2), keepdim=True)
                    std = waveform.std((1, 2), keepdim=True).clamp_min(1e-8)
                    normalized = (waveform - mean) / std
                    with torch.inference_mode():
                        sources = apply_model(model, normalized, device=args.device, shifts=0, split=True, overlap=0.25, progress=False)
                    sources = sources * std[:, None] + mean[:, None]
                    vocals = sources[:, vocal_index].mean(1)
                    vocals = torchaudio.functional.resample(vocals, model.samplerate, 16000).detach().cpu().numpy().astype(np.float32)
                    vocals = vocals[:, :expected]
                    if vocals.shape != (len(pending), expected) or not np.isfinite(vocals).all():
                        raise RuntimeError(f"Invalid vocal batch shape/finiteness: {vocals.shape}")
                    output_stats = {"rms_mean": float(np.sqrt(np.mean(vocals.astype(np.float64) ** 2))), "peak_max": float(np.max(np.abs(vocals)))}
                    for row, audio in zip(pending, vocals, strict=True):
                        target = args.output / "audio" / row["split"] / f"{row['sample_id']}.wav"
                        target.parent.mkdir(parents=True, exist_ok=True)
                        sf.write(str(target), audio, 16000, subtype="PCM_16")
                        item = dict(row); item["filepath"] = str(target.resolve()); complete.append(item)
            except Exception as exc:
                failures.extend({"sample_id": row["sample_id"], "error": repr(exc)} for row in pending)
            done = min(start + len(batch), len(rows))
            domain_counts = Counter(row["audio_domain"] for row in batch)
            label_counts = Counter(int(row.get("expected_voice_fake") or 0) for row in batch)
            elapsed = time.time() - batch_started
            emit({"event": "batch", "done": done, "total": len(rows), "complete": len(complete), "failed": len(failures), "batch_rows": len(batch), "reused": len(reusable), "inferred": len(pending), "batch_seconds": elapsed, "batch_items_per_second": len(batch) / elapsed if elapsed else None, "overall_items_per_second": done / (time.time() - started), "batch_domain_counts": dict(domain_counts), "batch_voice_fake_counts": dict(label_counts), **output_stats})
    manifests = args.output / "manifests"; manifests.mkdir(parents=True, exist_ok=True)
    fields = list(complete[0]) if complete else []
    for split in ("train", "validation", "test"):
        with (manifests / f"{split}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
            writer.writerows(row for row in complete if row["split"] == split)
    report = {"requested": len(rows), "complete": len(complete), "failures": failures, "batch_size": args.batch_size, "duration_seconds": args.duration, "target": "vocals", "counts": {split: dict(Counter(int(r.get("expected_voice_fake") or 0) for r in complete if r["split"] == split)) for split in ("train", "validation", "test")}}
    (args.output / "integrity.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    with log_path.open("a", buffering=1) as log:
        final = {"event": "complete", **report, "timestamp_unix": time.time(), "elapsed_seconds": time.time() - started, **cuda_stats(args.device)}
        log.write(json.dumps(final, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if failures or len(complete) != len(rows):
        raise SystemExit(1)

if __name__ == "__main__":
    main()
