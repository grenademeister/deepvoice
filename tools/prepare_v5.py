#!/usr/bin/env python3
"""Prepare resumable HTDemucs, DF-Arena, and SONICS features for V5."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from demucs.apply import apply_model

from model.v5_fusion import SONICS_DIM


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"sample_id", "local_path", "split", "expected_file_fake", "expected_voice_fake",
                "expected_music_fake", "expected_voice_present", "expected_music_present"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Invalid manifest; required columns: {sorted(required)}")
    ids = [row["sample_id"].strip() for row in rows]
    if any(not item for item in ids) or len(ids) != len(set(ids)):
        raise ValueError("Manifest sample_id values must be non-empty and unique")
    for row in rows:
        source = Path(row["local_path"])
        row["local_path"] = str((source if source.is_absolute() else ROOT / source).resolve())
    return rows


def contract(path: Path) -> dict:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "schema": 1, "manifest_sha256": digest, "preprocessing": "v5-expensive-features-v1",
        "sample_rate": 16000, "htdemucs": "955717e8-8726e21a.th",
        "df_arena": "Speech-Arena-2025/DF_Arena_500M_V_1", "sonics": "sonics-spectttra-alpha-5s",
    }


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=".tmp-", delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_npz(path: Path, **values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".tmp-", suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **values)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class Cache:
    directories = {"stems": "stems_cache", "df": "df_cache", "sonics": "sonics_cache"}

    def __init__(self, root: Path):
        self.root = root.resolve()

    def path(self, kind: str, sample_id: str) -> Path:
        if not sample_id or any(char in sample_id for char in "/\\"):
            raise ValueError(f"Unsafe sample_id: {sample_id!r}")
        return self.root / self.directories[kind] / f"{sample_id}.npz"

    @staticmethod
    def _read(path: Path):
        try:
            with np.load(path, allow_pickle=False) as saved:
                return {name: saved[name].copy() for name in saved.files}
        except (OSError, ValueError, KeyError):
            return None

    def stems(self, sample_id: str, length: int | None = None):
        values = self._read(self.path("stems", sample_id))
        if values is None or set(values) != {"voice", "acc"}:
            return None
        voice, music = values["voice"], values["acc"]
        valid = voice.dtype == music.dtype == np.float32 and voice.ndim == 1 and music.shape == voice.shape
        valid &= (length is None or len(voice) == length) and np.isfinite(voice).all() and np.isfinite(music).all()
        return values if valid else None

    def df(self, sample_id: str):
        values = self._read(self.path("df", sample_id))
        if values is None or set(values) != {"probability"}:
            return None
        value = values["probability"]
        return float(value[0]) if value.shape == (1,) and value.dtype == np.float32 and np.isfinite(value[0]) and 0 <= value[0] <= 1 else None

    def sonics(self, sample_id: str):
        values = self._read(self.path("sonics", sample_id))
        if values is None or set(values) != {"raw", "stem"}:
            return None
        valid = all(value.shape == (SONICS_DIM,) and value.dtype == np.float32 and np.isfinite(value).all() for value in values.values())
        return values if valid else None

    def write_stems(self, sample_id: str, voice: np.ndarray, music: np.ndarray):
        atomic_npz(self.path("stems", sample_id), voice=np.asarray(voice, np.float32), acc=np.asarray(music, np.float32))

    def write_df(self, sample_id: str, probability: float):
        atomic_npz(self.path("df", sample_id), probability=np.array([probability], np.float32))

    def write_sonics(self, sample_id: str, raw: np.ndarray, stem: np.ndarray):
        atomic_npz(self.path("sonics", sample_id), raw=np.asarray(raw, np.float32), stem=np.asarray(stem, np.float32))

    def validate_contract(self, expected: dict, adopt: bool = False):
        path = self.root / "cache_metadata.json"
        if path.exists():
            if json.loads(path.read_text()) != expected:
                raise ValueError("Cache metadata does not match this manifest/model contract")
        elif any((self.root / name).exists() for name in self.directories.values()) and not adopt:
            raise ValueError("Unversioned cache found; run adopt-cache first")
        else:
            atomic_json(path, expected)

    def summary(self, rows: list[dict[str, str]]) -> dict[str, int]:
        result = {"samples": len(rows), "stems": 0, "df": 0, "sonics": 0, "complete": 0}
        for row in rows:
            stem = self.stems(row["sample_id"], samples(row)) is not None
            df = self.df(row["sample_id"]) is not None
            sonics = self.sonics(row["sample_id"]) is not None
            result["stems"] += stem; result["df"] += df; result["sonics"] += sonics
            result["complete"] += stem and df and sonics
        return result


def samples(row: dict[str, str]) -> int | None:
    duration = float(row.get("duration") or 0)
    return round(duration * 16000) if duration else None


def log(cache: Cache, event: str, **fields):
    record = {"event": event, "time": time.time(), **fields}
    cache.root.mkdir(parents=True, exist_ok=True)
    with (cache.root / "events.jsonl").open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(json.dumps(record, sort_keys=True), flush=True)


def release(model, device: torch.device):
    target = model if hasattr(model, "to") else getattr(model, "model", None)
    if target is not None and hasattr(target, "to"):
        target.to("cpu")
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def separate(rows: list[dict[str, str]], model, device: torch.device):
    from demucs.separate import load_track
    prepared, active = [], []
    for row in rows:
        waveform = load_track(row["local_path"], model.audio_channels, model.samplerate).float()
        mean, std = waveform.mean(0).mean(), waveform.mean(0).std()
        item = None if float(std) < 1e-8 else ((waveform - mean) / std, mean, std, waveform.shape[-1])
        prepared.append(item)
        if item is not None:
            active.append(item[0])
    outputs = []
    if active:
        length = max(audio.shape[-1] for audio in active)
        batch = torch.stack([F.pad(audio, (0, length - audio.shape[-1])) for audio in active]).to(device)
        with torch.inference_mode():
            outputs = list(apply_model(model, batch, device=device, shifts=0, split=True, overlap=.25, progress=False))
    result, index = [], 0
    vocal = model.sources.index("vocals")
    for row, item in zip(rows, prepared, strict=True):
        length = samples(row)
        if item is None:
            if length is None:
                raise ValueError("Silence requires manifest duration")
            result.append((np.zeros(length, np.float32), np.zeros(length, np.float32)))
            continue
        _, mean, std, native_length = item
        sources = outputs[index][..., :native_length] * std + mean
        index += 1
        voice = sources[vocal].mean(0)
        music = torch.stack([sources[i] for i, name in enumerate(model.sources) if name != "vocals"]).sum(0).mean(0)
        length = length or round(native_length * 16000 / model.samplerate)
        converted = []
        for audio in (voice, music):
            audio = torchaudio.functional.resample(audio[None], model.samplerate, 16000)[0]
            converted.append(F.pad(audio, (0, max(0, length - len(audio))))[:length].cpu().numpy().astype(np.float32))
        result.append(tuple(converted))
    return result


def prepare_stems(rows, cache: Cache, device, batch_size):
    pending = [row for row in rows if cache.stems(row["sample_id"], samples(row)) is None]
    log(cache, "start", stage="stems", pending=len(pending), reused=len(rows) - len(pending))
    if not pending:
        return
    from script import load_htdemucs_model
    model = load_htdemucs_model()
    try:
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            for row, (voice, music) in zip(batch, separate(batch, model, device), strict=True):
                cache.write_stems(row["sample_id"], voice, music)
            log(cache, "batch", stage="stems", done=start + len(batch), total=len(pending))
    finally:
        release(model, device)


def prepare_df(rows, cache: Cache, device, batch_size):
    pending = [row for row in rows if cache.df(row["sample_id"]) is None]
    log(cache, "start", stage="df", pending=len(pending), reused=len(rows) - len(pending))
    if not pending:
        return
    from script import load_df_model
    model, fake = load_df_model(device)
    try:
        for start in range(0, len(pending), batch_size):
            groups = defaultdict(list)
            for row in pending[start:start + batch_size]:
                stem = cache.stems(row["sample_id"], samples(row))
                if stem is None:
                    raise RuntimeError(f"Missing stems for {row['sample_id']}")
                groups[len(stem["voice"])].append((row, stem["voice"]))
            for group in groups.values():
                values = torch.from_numpy(np.stack([voice for _, voice in group])).to(device)
                with torch.inference_mode():
                    output = model(input_values=values)
                    output = output["logits"] if isinstance(output, dict) else output.logits
                    probabilities = torch.softmax(output.float(), -1)[:, fake].cpu().numpy()
                for (row, _), probability in zip(group, probabilities, strict=True):
                    cache.write_df(row["sample_id"], float(probability))
            log(cache, "batch", stage="df", done=min(start + batch_size, len(pending)), total=len(pending))
    finally:
        release(model, device)


def embeddings(model, waveforms, device):
    from sonics_infer import preprocess_window
    values = torch.from_numpy(np.stack([preprocess_window(audio, 16000, 80000) for audio in waveforms])).to(device)
    with torch.inference_mode():
        spectrogram = F.interpolate(model.ft_extractor(values).unsqueeze(1), size=model.input_shape, mode="bilinear")
        return model.encoder(spectrogram).mean(1).cpu().numpy()


def prepare_sonics(rows, cache: Cache, device, batch_size):
    pending = [row for row in rows if cache.sonics(row["sample_id"]) is None]
    log(cache, "start", stage="sonics", pending=len(pending), reused=len(rows) - len(pending))
    if not pending:
        return
    from script import MODEL_DIR, load_audio
    from sonics_infer import load_sonics_model
    model = load_sonics_model(MODEL_DIR / "sonics", device)
    try:
        for start in range(0, len(pending), batch_size):
            batch, raw, music = pending[start:start + batch_size], [], []
            for row in batch:
                stem = cache.stems(row["sample_id"], samples(row))
                if stem is None:
                    raise RuntimeError(f"Missing stems for {row['sample_id']}")
                raw.append(load_audio(row["local_path"])); music.append(stem["acc"])
            for row, raw_embedding, stem_embedding in zip(batch, embeddings(model, raw, device), embeddings(model, music, device), strict=True):
                cache.write_sonics(row["sample_id"], raw_embedding, stem_embedding)
            log(cache, "batch", stage="sonics", done=start + len(batch), total=len(pending))
    finally:
        release(model, device)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("adopt-cache", "prepare", "status"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=("train", "validation"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--stage", choices=("all", "stems", "df", "sonics"), default="all")
    parser.add_argument("--separator-batch-size", type=int, default=2)
    parser.add_argument("--df-batch-size", type=int, default=4)
    parser.add_argument("--sonics-batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = arguments()
    rows = read_manifest(args.manifest.resolve())
    rows = [row for row in rows if row["split"] in args.splits]
    rows = rows[:args.limit] if args.limit else rows
    cache, expected = Cache(args.cache_dir), contract(args.manifest.resolve())
    cache.validate_contract(expected, adopt=args.command == "adopt-cache")
    if args.command in {"adopt-cache", "status"}:
        print(json.dumps(cache.summary(rows), indent=2))
        return
    device = torch.device(args.device)
    stages = ("stems", "df", "sonics") if args.stage == "all" else (args.stage,)
    functions = {"stems": (prepare_stems, args.separator_batch_size), "df": (prepare_df, args.df_batch_size),
                 "sonics": (prepare_sonics, args.sonics_batch_size)}
    for stage in stages:
        function, batch_size = functions[stage]
        if batch_size < 1:
            raise ValueError("Batch sizes must be positive")
        function(rows, cache, device, batch_size)
    print(json.dumps(cache.summary(rows), indent=2))


if __name__ == "__main__":
    main()
