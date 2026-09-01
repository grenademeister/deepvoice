#!/usr/bin/env python3
"""Prepare leakage-safe HTDemucs accompaniment stems for SONICS adaptation.

The generated waveforms reproduce DeepVoice's separator contract: HTDemucs
non-vocal source sum (drums+bass+other), mono, 16 kHz. Separation is offline;
SONICS training only reads prepared files.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
SPLITS = ("train", "validation", "test")


def stable_int(*parts: object) -> int:
    digest = hashlib.sha256("\0".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def split_rows(rows: list[dict], seed: int) -> list[dict]:
    """Assign each music parent once, stratified by class, to 80/10/10 splits."""
    result = []
    for target in (0, 1):
        groups: dict[str, list[dict]] = {}
        for row in rows:
            if int(row["target"]) != target:
                continue
            groups.setdefault(str(row["music_parent_id"]), []).append(dict(row))
        grouped_rows = list(groups.values())
        random.Random(stable_int(seed, target)).shuffle(grouped_rows)
        n_groups = len(grouped_rows)
        n_train = int(n_groups * 0.8)
        n_validation = int(n_groups * 0.1)
        for i, group in enumerate(grouped_rows):
            split = "train" if i < n_train else "validation" if i < n_train + n_validation else "test"
            for row in group:
                row["split"] = split
                result.append(row)
    return result


def build_source_rows(real_dir: Path, fake_records_path: Path, seed: int, max_per_class: int) -> list[dict]:
    """Build balanced real/fake parent rows with parent-level split assignment."""
    real_files = sorted(p for p in real_dir.rglob("*") if p.suffix.lower() in AUDIO_SUFFIXES)
    if not real_files:
        raise ValueError(f"No audio found under {real_dir}")
    fake_records = [json.loads(line) for line in fake_records_path.read_text().splitlines() if line.strip()]
    fake_rows = []
    for record in fake_records:
        if int(record.get("expected_music_fake", 0) or 0) != 1:
            continue
        path = Path(record["local_path"])
        fake_rows.append({
            "id": str(record["sample_id"]), "filepath": str(path), "target": 1,
            "music_parent_id": str(record.get("parent_id") or record["sample_id"]),
            "generator_family": str(record.get("generator_family") or "unknown"),
        })
    if not fake_rows:
        raise ValueError(f"No fake records in {fake_records_path}")

    random.Random(stable_int(seed, "real")).shuffle(real_files)
    random.Random(stable_int(seed, "fake")).shuffle(fake_rows)
    n = min(len(real_files), len(fake_rows), max_per_class) if max_per_class > 0 else min(len(real_files), len(fake_rows))
    if n < 10:
        raise ValueError(f"Need at least 10 parents per class; found {n}")
    rows = [
        {"id": f"real_{p.stem}", "filepath": str(p), "target": 0,
         "music_parent_id": f"real:{p.stem}", "generator_family": "real"}
        for p in real_files[:n]
    ] + fake_rows[:n]
    return split_rows(rows, seed)


def expand_mixtures(rows: list[dict], donors: list[str], variants: int, seed: int) -> list[dict]:
    """Create deterministic voice-donor variants without moving a music parent across splits."""
    if not donors:
        raise ValueError("No voice donors available")
    expanded = []
    for row in rows:
        count = variants if row["split"] == "train" else min(2, variants)
        for variant in range(count):
            item = dict(row)
            item["variant"] = variant
            item["id"] = f"{row['id']}__v{variant:02d}"
            item["voice_donor"] = donors[stable_int(seed, row["music_parent_id"], variant) % len(donors)]
            # Independent, deterministic mixing gains. target remains music-parent-derived.
            item["music_gain"] = 0.30 + 0.10 * (stable_int(seed, "music", item["id"]) % 5)
            item["voice_gain"] = 1.0
            expanded.append(item)
    return expanded


def index_reusable_stems(prepared_root: Path) -> dict[str, Path]:
    """Index existing stems by ID so corrected manifests need not re-separate them."""
    result = {}
    for path in (prepared_root / "audio").glob("*/*.wav"):
        if path.stem in result:
            raise ValueError(f"Duplicate reusable stem ID: {path.stem}")
        result[path.stem] = path.resolve()
    return result


def model_input_sample_rate(separator_sample_rate: int) -> int:
    """HTDemucs must receive audio at its native sample rate before output resampling."""
    if int(separator_sample_rate) <= 0:
        raise ValueError(f"Invalid separator sample rate: {separator_sample_rate}")
    return int(separator_sample_rate)


def load_audio_16k(path: Path, target_sr: int, duration_s: float) -> np.ndarray:
    import torchaudio
    audio, sr = torchaudio.load(str(path))
    audio = audio.mean(0, keepdim=True)
    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, sr, target_sr)
    expected = int(target_sr * duration_s)
    if audio.shape[1] < expected:
        repeats = (expected + audio.shape[1] - 1) // max(audio.shape[1], 1)
        audio = audio.repeat(1, repeats)
    return audio[:, :expected].numpy().astype(np.float32)


def load_htdemucs(repo: Path):
    import torch
    from demucs.pretrained import get_model
    original_load = torch.load
    def trusted_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)
    torch.load = trusted_load
    try:
        return get_model("htdemucs", repo=repo).eval()
    finally:
        torch.load = original_load


def to_model_channels(waveform, audio_channels: int):
    """Duplicate a mono mixture for the channel count required by HTDemucs."""
    if waveform.shape[0] == audio_channels:
        return waveform
    if waveform.shape[0] != 1:
        raise ValueError(f"Cannot map {waveform.shape[0]} channels to {audio_channels}")
    return waveform.repeat(audio_channels, 1)


def extract_accompaniment(mixture: np.ndarray, model, device: str, target_sr: int) -> np.ndarray:
    """Exact DeepVoice-style normalization, HTDemucs call, source sum, resample."""
    import torch
    import torchaudio
    from demucs.apply import apply_model
    waveform = to_model_channels(torch.from_numpy(mixture), model.audio_channels)
    mean, std = waveform.mean(), waveform.std()
    if float(std) < 1e-8:
        return np.zeros((1, 0), dtype=np.float32)
    normalized = (waveform - mean) / std
    with torch.inference_mode():
        sources = apply_model(model, normalized[None], device=device, shifts=0, split=True, overlap=0.25, progress=False)[0]
    sources = sources * std + mean
    music_indices = [i for i, name in enumerate(model.sources) if name != "vocals"]
    accompaniment = torch.stack([sources[i] for i in music_indices]).sum(0).mean(0, keepdim=True)
    accompaniment = torchaudio.functional.resample(accompaniment, model.samplerate, target_sr)
    return accompaniment.cpu().numpy().astype(np.float32)


def write_manifest(rows: list[dict], path: Path) -> None:
    fields = ["id", "filepath", "target", "split", "music_parent_id", "generator_family", "voice_donor", "variant", "music_gain", "voice_gain"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--fake-records", type=Path, required=True)
    parser.add_argument("--fake-root", type=Path, default=Path("/root/deepvoice-evalset"))
    parser.add_argument("--voice-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reuse-root", type=Path, default=None,
                        help="Reuse matching stem IDs from a prior prepared root; prepare only missing IDs.")
    parser.add_argument("--htdemucs-repo", type=Path, default=Path("/root/deepvoice/model/htdemucs"))
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--max-per-class", type=int, default=500)
    parser.add_argument("--train-variants", type=int, default=4)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    raw_fake = [json.loads(line) for line in args.fake_records.read_text().splitlines() if line.strip()]
    resolved_records = args.output_root / "fake_records_resolved.jsonl"
    resolved_records.parent.mkdir(parents=True, exist_ok=True)
    for record in raw_fake:
        local = Path(record["local_path"])
        if not local.is_absolute():
            record["local_path"] = str(args.fake_root / local)
    resolved_records.write_text("\n".join(json.dumps(r, sort_keys=True) for r in raw_fake) + "\n")

    parents = build_source_rows(args.real_dir, resolved_records, args.seed, args.max_per_class)
    donors = sorted(str(p) for p in args.voice_dir.rglob("*") if p.suffix.lower() in AUDIO_SUFFIXES)
    examples = expand_mixtures(parents, donors, args.train_variants, args.seed)
    if args.limit:
        examples = examples[:args.limit]
    model = load_htdemucs(args.htdemucs_repo)
    reusable = index_reusable_stems(args.reuse_root) if args.reuse_root else {}
    import soundfile as sf
    complete = []
    failures = []
    for index, row in enumerate(examples, 1):
        out = args.output_root / "audio" / row["split"] / f"{row['id']}.wav"
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            if row["id"] in reusable:
                row["filepath"] = str(reusable[row["id"]])
                complete.append(row)
                continue
            if not out.exists():
                separator_sr = model_input_sample_rate(model.samplerate)
                music = load_audio_16k(Path(row["filepath"]), separator_sr, args.duration)
                voice = load_audio_16k(Path(row["voice_donor"]), separator_sr, args.duration)
                mixture = row["voice_gain"] * voice + row["music_gain"] * music
                stem = extract_accompaniment(mixture, model, args.device, 16000)
                if stem.shape[1] != int(16000 * args.duration) or not np.isfinite(stem).all():
                    raise RuntimeError(f"invalid stem shape={stem.shape}")
                sf.write(out, stem[0], 16000, subtype="PCM_16")
            row["filepath"] = str(out.resolve())
            complete.append(row)
        except Exception as exc:
            failures.append({"id": row["id"], "error": repr(exc)})
        if index % 25 == 0 or index == len(examples):
            print(json.dumps({"processed": index, "complete": len(complete), "failed": len(failures)}, sort_keys=True), flush=True)
    manifests = args.output_root / "manifests"
    manifests.mkdir(exist_ok=True)
    for split in SPLITS:
        write_manifest([r for r in complete if r["split"] == split], manifests / f"{split}.csv")
    report = {
        "seed": args.seed, "target_sr": 16000, "duration": args.duration,
        "htdemucs_repo": str(args.htdemucs_repo), "sources": list(getattr(model, "sources", [])),
        "counts": {split: dict(Counter(int(r["target"]) for r in complete if r["split"] == split)) for split in SPLITS},
        "parents": len({r["music_parent_id"] for r in complete}), "examples": len(complete),
        "failures": failures,
    }
    (args.output_root / "integrity.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if failures:
        raise RuntimeError(f"Preparation failed for {len(failures)} examples; see {args.output_root / 'integrity.json'}")


if __name__ == "__main__":
    main()
