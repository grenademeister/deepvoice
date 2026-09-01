#!/usr/bin/env python3
"""Evaluate original and fine-tuned SONICS inside the V1 fusion contract.

PANNs and DF-Arena component scores are read from a completed full V2 run on
identical files. HTDemucs is rerun here to obtain the music stem for SONICS.
No deployed checkpoint is modified.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "model"))
from sonics_infer import SonicsClassifier, predict_fake
from script import load_htdemucs_model, separate_voice_and_music


def v1_file_score(*, voice_fake: float, music_fake: float, voice_present: float, music_present: float) -> float:
    return max(voice_present * voice_fake, music_present * music_fake)


def number(value: str | None) -> float:
    return 0.0 if value in (None, "") else float(value)


def eer(y, score):
    y = np.asarray(y, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    if len(np.unique(y)) != 2:
        return None
    fpr, tpr, _ = roc_curve(y, score, pos_label=1, drop_intermediate=False)
    fnr = 1.0 - tpr
    return float(((fpr + fnr) / 2.0)[np.argmin(np.abs(fpr - fnr))])


def metrics(rows, predictions):
    y_file = np.array([number(r["expected_file_fake"]) for r in rows])
    y_voice = np.array([number(r["expected_voice_fake"]) for r in rows])
    y_music = np.array([number(r["expected_music_fake"]) for r in rows])
    y_vp = np.array([number(r["expected_voice_present"]) for r in rows])
    y_mp = np.array([number(r["expected_music_present"]) for r in rows])
    p_file = np.array([p["file"] for p in predictions])
    p_voice = np.array([p["voice"] for p in predictions])
    p_music = np.array([p["music"] for p in predictions])
    p_vp = np.array([p["voice_present"] for p in predictions])
    p_mp = np.array([p["music_present"] for p in predictions])
    file_eer = eer(y_file, p_file)
    voice_eer = eer(y_voice[y_vp > .5], p_voice[y_vp > .5])
    music_eer = eer(y_music[y_mp > .5], p_music[y_mp > .5])
    ads = .5 * (1 - file_eer) + .2 * (1 - voice_eer) + .3 * (1 - music_eer)
    cps = .5 * roc_auc_score(y_vp, p_vp) + .5 * roc_auc_score(y_mp, p_mp)
    mixed = np.array([r["audio_domain"] == "mixed" for r in rows])
    return {
        "n": len(rows), "score": .9 * ads + .1 * cps, "ads": ads, "cps": cps,
        "file_eer": file_eer, "file_eer_mixed": eer(y_file[mixed], p_file[mixed]),
        "voice_eer": voice_eer, "music_eer": music_eer, "mixed_n": int(mixed.sum()),
    }


def load_sonics(config_path: Path, checkpoint: Path, state_key: str | None, device):
    config = json.loads(config_path.read_text())
    state = torch.load(checkpoint, map_location="cpu", weights_only=state_key is None)
    if state_key:
        state = state[state_key]
    model = SonicsClassifier(config)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evalset", type=Path, default=Path("/root/deepvoice-evalset"))
    parser.add_argument("--component-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--adapted-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--splits", nargs="+", default=["validation", "test"],
        choices=["train", "validation", "test"],
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = list(csv.DictReader((args.evalset / "manifests/manifest_balanced.csv").open()))
    models = {
        "original": load_sonics(PROJECT / "model/sonics/config.json", PROJECT / "model/sonics/pytorch_model.bin", None, device),
        "adapted": load_sonics(PROJECT / "model/sonics/config.json", args.adapted_checkpoint, "state_dict", device),
    }
    htdemucs = load_htdemucs_model()
    all_metrics = {}
    for split in args.splits:
        rows = [r for r in manifest if r["split_balanced"] == split]
        components = {r["ID"]: r for r in csv.DictReader((args.component_run / f"predictions_{split}.csv").open())}
        if set(components) != {r["sample_id"] for r in rows}:
            raise ValueError(f"Component prediction IDs do not match {split} manifest")
        pred_sets = {name: [] for name in models}
        detail_path = args.output_dir / f"v1_sonics_{split}_scores.csv"
        with detail_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["ID", "original_music", "original_file", "adapted_music", "adapted_file", "voice", "voice_present", "music_present"])
            writer.writeheader()
            for index, row in enumerate(rows, 1):
                component = components[row["sample_id"]]
                source = args.evalset / row["local_path"]
                _, music, _ = separate_voice_and_music(source, htdemucs, device)
                voice = number(component["VOICE_FAKE_PROB"])
                vp = number(component["VOICE_PRESENT_PROB"])
                mp = number(component["MUSIC_PRESENT_PROB"])
                detail = {"ID": row["sample_id"], "voice": voice, "voice_present": vp, "music_present": mp}
                for name, model in models.items():
                    music_score = predict_fake(model, music, 16000, device=device)
                    file_score = v1_file_score(voice_fake=voice, music_fake=music_score, voice_present=vp, music_present=mp)
                    pred_sets[name].append({"file": file_score, "voice": voice, "music": music_score, "voice_present": vp, "music_present": mp})
                    detail[f"{name}_music"] = music_score
                    detail[f"{name}_file"] = file_score
                writer.writerow(detail)
                if index % 25 == 0 or index == len(rows):
                    print(json.dumps({"split": split, "processed": index, "total": len(rows)}), flush=True)
        all_metrics[split] = {name: metrics(rows, pred) for name, pred in pred_sets.items()}
    (args.output_dir / "metrics.json").write_text(json.dumps(all_metrics, indent=2) + "\n")
    print(json.dumps(all_metrics, indent=2))


if __name__ == "__main__":
    main()
