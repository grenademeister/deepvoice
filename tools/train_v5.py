#!/usr/bin/env python3
"""Train or resume V5 fusion from prepared expensive features."""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve

from model.v5_fusion import Fusion, logits, payload
from tools.prepare_v5 import Cache, atomic_json, contract, read_manifest, release, samples


def number(value: str | None) -> float:
    return 0.0 if value is None or str(value).strip() == "" else float(value)


@dataclass(frozen=True)
class Config:
    epochs: int = 3
    batch_size: int = 16
    projection: int = 64
    hidden: int = 128
    learning_rate: float = 3e-4
    seed: int = 20260904


@dataclass
class Dataset:
    rows: list[dict[str, str]]
    scalars: np.ndarray
    raw: np.ndarray
    stem: np.ndarray
    labels: np.ndarray
    masks: np.ndarray
    presence: np.ndarray

    def split(self, name: str):
        indices = np.array([i for i, row in enumerate(self.rows) if row["split"] == name])
        if not len(indices):
            raise ValueError(f"No rows for split={name}")
        return Dataset([self.rows[i] for i in indices], self.scalars[indices], self.raw[indices],
                       self.stem[indices], self.labels[indices], self.masks[indices], self.presence[indices])


def event(path: Path, name: str, **fields):
    record = {"event": name, "time": time.time(), **fields}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(json.dumps(record, sort_keys=True), flush=True)


def assemble(rows: list[dict[str, str]], cache: Cache, device: torch.device, events: Path) -> Dataset:
    expensive = []
    for row in rows:
        if cache.stems(row["sample_id"], samples(row)) is None:
            raise RuntimeError(f"Missing stems for {row['sample_id']}")
        df, sonics = cache.df(row["sample_id"]), cache.sonics(row["sample_id"])
        if df is None or sonics is None:
            raise RuntimeError(f"Incomplete feature cache for {row['sample_id']}")
        expensive.append((df, sonics))

    from script import (ARTIFACTNET_SAMPLE_RATE, MODEL_DIR, load_artifactnet_model, load_audio,
                        load_panns_model, predict_artifactnet_raw_and_stem, predict_presence)
    event(events, "transient_start", stage="panns", total=len(rows))
    model, voice_indices, music_indices = load_panns_model(device)
    presence = []
    try:
        for i, row in enumerate(rows, 1):
            presence.append(predict_presence(model, voice_indices, music_indices, load_audio(row["local_path"])))
            if i % 100 == 0 or i == len(rows):
                event(events, "transient_batch", stage="panns", done=i, total=len(rows))
    finally:
        release(model, device)
    del model

    event(events, "transient_start", stage="artifactnet", total=len(rows))
    model = load_artifactnet_model(MODEL_DIR / "artifactnet")
    artifacts = []
    try:
        for i, row in enumerate(rows, 1):
            stem = cache.stems(row["sample_id"], samples(row))["acc"]
            artifacts.append(predict_artifactnet_raw_and_stem(
                model, raw_audio=load_audio(row["local_path"], ARTIFACTNET_SAMPLE_RATE),
                raw_sample_rate=ARTIFACTNET_SAMPLE_RATE, music_stem=stem, stem_sample_rate=16000,
            ))
            if i % 100 == 0 or i == len(rows):
                event(events, "transient_batch", stage="artifactnet", done=i, total=len(rows))
    finally:
        release(model, device)
    del model

    scalar_values, raw, stem, labels, masks = [], [], [], [], []
    for row, (df, sonics), found_presence, artifact in zip(rows, expensive, presence, artifacts, strict=True):
        scalar_values.append((df, *artifact, *found_presence))
        raw.append(sonics["raw"]); stem.append(sonics["stem"])
        labels.append((number(row["expected_file_fake"]), number(row["expected_voice_fake"]), number(row["expected_music_fake"])))
        masks.append((1, number(row["expected_voice_present"]), number(row["expected_music_present"])))
    return Dataset(rows, logits(np.asarray(scalar_values, np.float32)), np.asarray(raw, np.float32),
                   np.asarray(stem, np.float32), np.asarray(labels, np.float32), np.asarray(masks, np.float32),
                   np.asarray(presence, np.float32))


def masked_loss(output, labels, masks):
    losses = F.binary_cross_entropy_with_logits(output, labels, reduction="none")
    return (losses * masks).sum() / masks.sum().clamp_min(1)


def predict(model: Fusion, data: Dataset, batch_size: int, device: torch.device):
    model.eval(); predictions, total_loss, count = [], 0.0, 0.0
    with torch.inference_mode():
        for start in range(0, len(data.rows), batch_size):
            end = start + batch_size
            tensors = [torch.from_numpy(value[start:end]).to(device) for value in (data.scalars, data.raw, data.stem, data.labels, data.masks)]
            output = model(*tensors[:3]); weight = float(tensors[4].sum())
            total_loss += float(masked_loss(output, tensors[3], tensors[4]).cpu()) * weight; count += weight
            predictions.append(torch.sigmoid(output).cpu().numpy())
    return np.concatenate(predictions), total_loss / count


def eer(labels, scores):
    if len(np.unique(labels)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1, drop_intermediate=False)
    fnr = 1 - tpr; index = np.argmin(np.abs(fpr - fnr))
    return float((fpr[index] + fnr[index]) / 2)


def metrics(data: Dataset, predictions: np.ndarray):
    present = np.array([[number(row["expected_voice_present"]), number(row["expected_music_present"])] for row in data.rows])
    file_eer = eer(data.labels[:, 0], predictions[:, 0])
    voice_eer = eer(data.labels[present[:, 0] > .5, 1], predictions[present[:, 0] > .5, 1])
    music_eer = eer(data.labels[present[:, 1] > .5, 2], predictions[present[:, 1] > .5, 2])
    aucs = [float(roc_auc_score(present[:, i], data.presence[:, i])) if len(np.unique(present[:, i])) > 1 else float("nan") for i in range(2)]
    ads = .5 * (1 - file_eer) + .2 * (1 - voice_eer) + .3 * (1 - music_eer)
    cps = sum(aucs) / 2
    return {"file_eer": file_eer, "voice_eer": voice_eer, "music_eer": music_eer, "ads": ads,
            "voice_auc": aucs[0], "music_auc": aucs[1], "cps": cps, "score": .9 * ads + .1 * cps}


def atomic_save(value: object, path: Path):
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".tmp-", suffix=".pt", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(value, temporary); os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def training_state(model, optimizer, scheduler, config, cache_contract, epoch, offset, step,
                   history, best_score, loss_sum, loss_count):
    return {
        "format": "deepvoice-v5-training-state", "config": asdict(config), "cache_contract": cache_contract,
        "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "epoch": epoch, "offset": offset, "step": step, "history": history, "best_score": best_score,
        "loss_sum": loss_sum, "loss_count": loss_count, "rng": rng_state(),
    }


def write_predictions(path: Path, data: Dataset, predictions: np.ndarray):
    fields = ("ID", "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB", "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB")
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(fields)
        for row, scores, presence in zip(data.rows, predictions, data.presence, strict=True):
            writer.writerow((row["sample_id"], *[f"{value:.10f}" for value in scores], *[f"{value:.10f}" for value in presence]))


def train(train_data, validation, cache_contract, run_dir: Path, config: Config, device,
          checkpoint_every: int, resume: bool, max_steps: int):
    run_dir.mkdir(parents=True, exist_ok=True); events = run_dir / "events.jsonl"; last = run_dir / "last.pt"
    random.seed(config.seed); np.random.seed(config.seed); torch.manual_seed(config.seed)
    if device.type == "cuda": torch.cuda.manual_seed_all(config.seed)
    model = Fusion(config.projection, config.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    epoch, offset, step, history, best_score, loss_sum, loss_count = 1, 0, 0, [], -float("inf"), 0.0, 0
    if resume:
        saved = torch.load(last, map_location=device, weights_only=False)
        if saved.get("format") != "deepvoice-v5-training-state" or saved["config"] != asdict(config) or saved["cache_contract"] != cache_contract:
            raise ValueError("Incompatible resume checkpoint")
        model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"]); scheduler.load_state_dict(saved["scheduler"])
        epoch, offset, step, history = saved["epoch"], saved["offset"], saved["step"], saved["history"]
        best_score = saved["best_score"]
        loss_sum, loss_count = saved["loss_sum"], saved["loss_count"]; restore_rng(saved["rng"])
        event(events, "resume", epoch=epoch, offset=offset, step=step)
    elif last.exists():
        raise FileExistsError("Run already exists; pass --resume or choose another --run-dir")

    invocation_steps = 0
    while epoch <= config.epochs:
        order = np.random.default_rng(config.seed + epoch).permutation(len(train_data.rows)); model.train()
        while offset < len(order):
            indices = order[offset:offset + config.batch_size]
            tensors = [torch.from_numpy(value[indices]).to(device) for value in (train_data.scalars, train_data.raw, train_data.stem, train_data.labels, train_data.masks)]
            optimizer.zero_grad(set_to_none=True); loss = masked_loss(model(*tensors[:3]), tensors[3], tensors[4])
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1); optimizer.step()
            loss_sum += float(loss.detach().cpu()); loss_count += 1; offset += len(indices); step += 1; invocation_steps += 1
            save = step % checkpoint_every == 0 or (max_steps and invocation_steps >= max_steps)
            if save:
                atomic_save(training_state(model, optimizer, scheduler, config, cache_contract, epoch, offset, step,
                                           history, best_score, loss_sum, loss_count), last)
            if max_steps and invocation_steps >= max_steps:
                event(events, "paused", epoch=epoch, offset=offset, step=step); return

        predictions, validation_loss = predict(model, validation, config.batch_size, device)
        record = {"epoch": epoch, "train_loss": loss_sum / loss_count, "validation_loss": validation_loss, **metrics(validation, predictions)}
        history.append(record); write_predictions(run_dir / f"validation_{epoch:02d}.csv", validation, predictions)
        atomic_json(run_dir / f"metrics_{epoch:02d}.json", record)
        checkpoint = payload(model, {"config": asdict(config), "history": history})
        atomic_save(checkpoint, run_dir / f"epoch_{epoch:02d}.pt")
        if record["score"] > best_score:
            best_score = record["score"]; atomic_save(checkpoint, run_dir / "best.pt")
        scheduler.step(); epoch += 1; offset = 0; loss_sum = 0.; loss_count = 0
        atomic_save(training_state(model, optimizer, scheduler, config, cache_contract, epoch, offset, step,
                                   history, best_score, loss_sum, loss_count), last)
        atomic_json(run_dir / "history.json", history); event(events, "epoch", **record, step=step)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True); parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="validation"); parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16); parser.add_argument("--projection", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128); parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20260904); parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--resume", action="store_true"); parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = arguments()
    if min(args.epochs, args.batch_size, args.checkpoint_every) < 1:
        raise ValueError("Epochs, batch size, and checkpoint interval must be positive")
    manifest = args.manifest.resolve(); cache_contract = contract(manifest); cache = Cache(args.cache_dir)
    cache.validate_contract(cache_contract)
    rows = [row for row in read_manifest(manifest) if row["split"] in {args.train_split, args.validation_split}]
    data = assemble(rows, cache, torch.device(args.device), args.run_dir.resolve() / "events.jsonl")
    config = Config(args.epochs, args.batch_size, args.projection, args.hidden, args.lr, args.seed)
    train(data.split(args.train_split), data.split(args.validation_split), cache_contract, args.run_dir.resolve(),
          config, torch.device(args.device), args.checkpoint_every, args.resume, args.max_steps)


if __name__ == "__main__":
    main()
