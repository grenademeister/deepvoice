#!/usr/bin/env python3
"""Run the deployable V5 DeepVoice pipeline."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import shutil
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
sys.path.insert(0, str(MODEL_DIR))

import librosa
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from demucs.apply import apply_model
from demucs.pretrained import get_model
from demucs.separate import load_track
from artifactnet_infer import ARTIFACTNET_SAMPLE_RATE, load_artifactnet_model, predict_artifactnet_raw_and_stem
from sonics_infer import load_sonics_model, preprocess_window
from v5_fusion import load as load_fusion, logits

AUDIO_RATE = 16_000
PANNS_RATE = 32_000
PANNS_WINDOW = 64_600
OUTPUTS = ("FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB", "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB")
EXTENSIONS = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav", ".wma"}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, default=Path("data/test"))
    parser.add_argument("--sample-submission", type=Path, default=Path("data/sample_submission.csv"))
    parser.add_argument("--output", type=Path, default=Path("output/submission.csv"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--df-batch-size", type=int, default=4)
    parser.add_argument("--sonics-batch-size", type=int, default=16)
    parser.add_argument("--fusion-batch-size", type=int, default=256)
    return parser.parse_args()


def load_audio(path: Path | str, sample_rate: int = AUDIO_RATE) -> np.ndarray:
    audio, _ = librosa.load(path, sr=sample_rate, mono=True, dtype=np.float32)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError(f"Invalid audio: {path}")
    return audio


def inputs(directory: Path, template: Path):
    audio = {path.stem: path for path in directory.iterdir() if path.suffix.lower() in EXTENSIONS}
    with template.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle); fields, rows = reader.fieldnames, list(reader)
    if fields is None or not rows or any(name not in fields for name in ("ID", *OUTPUTS)):
        raise ValueError("Invalid sample submission")
    ids = [row["ID"].strip() for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(audio):
        raise ValueError("Audio files and submission IDs must match exactly")
    return fields, rows, [audio[sample_id] for sample_id in ids]


def release(model, device: torch.device):
    target = model if hasattr(model, "to") else getattr(model, "model", None)
    if target is not None and hasattr(target, "to"):
        target.to("cpu")
    del model; gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def load_panns_model(device: torch.device):
    source = MODEL_DIR / "panns/class_labels_indices.csv"
    target = Path.home() / "panns_data/class_labels_indices.csv"
    target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
    from panns_inference import AudioTagging, labels
    model = AudioTagging(checkpoint_path=str(MODEL_DIR / "panns/Cnn14_mAP=0.431.pth"), device=device.type)
    groups = json.loads((MODEL_DIR / "panns/component_labels.json").read_text())
    indices = {label: index for index, label in enumerate(labels)}
    return model, [indices[label] for label in groups["voice"]], [indices[label] for label in groups["music"]]


def predict_presence(model, voice_indices, music_indices, audio: np.ndarray):
    starts = list(range(0, max(1, len(audio) - PANNS_WINDOW + 1), PANNS_WINDOW))
    last = max(0, len(audio) - PANNS_WINDOW)
    starts = sorted(set((*starts, last)))
    segments = []
    for start in starts:
        segment = audio[start:start + PANNS_WINDOW]
        if len(segment) < PANNS_WINDOW:
            segment = np.resize(segment, PANNS_WINDOW)
        segments.append(librosa.resample(segment, orig_sr=AUDIO_RATE, target_sr=PANNS_RATE, res_type="soxr_hq"))
    predictions, _ = model.inference(np.asarray(segments, np.float32))
    return float(predictions[:, voice_indices].max()), float(predictions[:, music_indices].max())


def load_htdemucs_model():
    original = torch.load
    def trusted(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)
    torch.load = trusted
    try:
        return get_model("htdemucs", repo=MODEL_DIR / "htdemucs").cpu().eval()
    finally:
        torch.load = original


def separate_voice_and_music(path: Path | str, model, device: torch.device):
    waveform = load_track(path, model.audio_channels, model.samplerate).float()
    mono = waveform.mean(0); mean, std = mono.mean(), mono.std()
    length = max(1, round(waveform.shape[-1] * AUDIO_RATE / model.samplerate))
    if float(std) < 1e-8:
        silence = np.zeros(length, np.float32)
        return silence, silence.copy()
    with torch.inference_mode():
        sources = apply_model(model, ((waveform - mean) / std)[None], device=device, shifts=0,
                              split=True, overlap=.25, progress=False)[0] * std + mean
    vocal = model.sources.index("vocals")
    voice = sources[vocal].mean(0, keepdim=True)
    music = torch.stack([sources[i] for i, name in enumerate(model.sources) if name != "vocals"]).sum(0).mean(0, keepdim=True)
    voice = torchaudio.functional.resample(voice, model.samplerate, AUDIO_RATE)[0, :length]
    music = torchaudio.functional.resample(music, model.samplerate, AUDIO_RATE)[0, :length]
    return voice.cpu().numpy().astype(np.float32), music.cpu().numpy().astype(np.float32)


def load_df_model(device: torch.device):
    from df_arena_500m.modeling_antispoofing import DF_Arena_500M_Antispoofing
    directory = MODEL_DIR / "df_arena_500m"
    model = DF_Arena_500M_Antispoofing.from_pretrained(directory, local_files_only=True).to(device).eval()
    return model, int(model.config.label2id["spoof"])


def df_scores(model, fake_index: int, stems: list[np.ndarray], batch_size: int, device: torch.device):
    scores = np.empty(len(stems), np.float32)
    groups = {}
    for index, stem in enumerate(stems):
        groups.setdefault(len(stem), []).append(index)
    with torch.inference_mode():
        for indices in groups.values():
            for start in range(0, len(indices), batch_size):
                batch = indices[start:start + batch_size]
                output = model(input_values=torch.from_numpy(np.stack([stems[i] for i in batch])).to(device))
                output = output["logits"] if isinstance(output, dict) else output.logits
                scores[batch] = torch.softmax(output.float(), -1)[:, fake_index].cpu().numpy()
    return scores


def sonics_embeddings(model, audio: list[np.ndarray], batch_size: int, device: torch.device):
    result = []
    with torch.inference_mode():
        for start in range(0, len(audio), batch_size):
            batch = np.stack([preprocess_window(value, AUDIO_RATE, 80_000) for value in audio[start:start + batch_size]])
            spectrogram = model.ft_extractor(torch.from_numpy(batch).to(device)).unsqueeze(1)
            spectrogram = F.interpolate(spectrogram, size=model.input_shape, mode="bilinear")
            result.append(model.encoder(spectrogram).mean(1).cpu().numpy())
    return np.concatenate(result)


def infer(paths: list[Path], device: torch.device, df_batch: int, sonics_batch: int, fusion_batch: int):
    raw = [load_audio(path) for path in paths]
    panns, voice_indices, music_indices = load_panns_model(device)
    presence = np.array([predict_presence(panns, voice_indices, music_indices, audio) for audio in raw], np.float32)
    release(panns, device)
    del panns

    separator = load_htdemucs_model()
    separated = [separate_voice_and_music(path, separator, device) for path in paths]
    release(separator, device)
    del separator
    voice, music = map(list, zip(*separated, strict=True))

    df, fake_index = load_df_model(device)
    df_probability = df_scores(df, fake_index, voice, df_batch, device)
    release(df, device)
    del df

    sonics = load_sonics_model(MODEL_DIR / "sonics", device)
    raw_embeddings = sonics_embeddings(sonics, raw, sonics_batch, device)
    stem_embeddings = sonics_embeddings(sonics, music, sonics_batch, device)
    release(sonics, device)
    del sonics

    artifact = load_artifactnet_model(MODEL_DIR / "artifactnet")
    artifact_scores = np.array([
        predict_artifactnet_raw_and_stem(artifact, raw_audio=load_audio(path, ARTIFACTNET_SAMPLE_RATE),
                                         raw_sample_rate=ARTIFACTNET_SAMPLE_RATE, music_stem=stem, stem_sample_rate=AUDIO_RATE)
        for path, stem in zip(paths, music, strict=True)
    ], np.float32)
    scalars = np.column_stack((df_probability, artifact_scores, presence)).astype(np.float32)

    fusion, _ = load_fusion(MODEL_DIR / "v5_fusion.pt", str(device))
    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(paths), fusion_batch):
            end = start + fusion_batch
            output = fusion(
                torch.from_numpy(logits(scalars[start:end])).to(device),
                torch.from_numpy(raw_embeddings[start:end]).to(device),
                torch.from_numpy(stem_embeddings[start:end]).to(device),
            )
            probabilities.append(torch.sigmoid(output).cpu().numpy())
    return np.concatenate(probabilities), presence


def main():
    args = arguments()
    if min(args.df_batch_size, args.sonics_batch_size, args.fusion_batch_size) < 1:
        raise ValueError("Batch sizes must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    fields, rows, paths = inputs(args.test_dir, args.sample_submission)
    fake, presence = infer(paths, torch.device(args.device), args.df_batch_size, args.sonics_batch_size, args.fusion_batch_size)
    for row, scores, found_presence in zip(rows, fake, presence, strict=True):
        for name, value in zip(OUTPUTS, (*scores, *found_presence), strict=True):
            row[name] = f"{value:.10f}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    print(f"Saved {len(rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
