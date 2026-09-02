"""Serializable V4 three-output fusion contract.

The base-detector probabilities remain frozen inputs.  This module defines the
only feature transform used by both training and deployment; the checkpoint
persists the raw feature order and normalisation statistics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn

RAW_FEATURES: tuple[str, ...] = (
    "df_raw",
    "df_voice",
    "sonics_stem",
    "artifact_raw",
    "artifact_stem",
    "voice_present",
    "music_present",
)
OUTPUT_NAMES: tuple[str, ...] = ("file_fake", "voice_fake", "music_fake")
EPSILON = 1e-5


def _probability_column(values: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    if name not in values:
        raise KeyError(f"Missing required frozen score: {name}")
    x = np.asarray(values[name], dtype=np.float32)
    if x.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got {x.shape}")
    if not np.isfinite(x).all() or (x < 0).any() or (x > 1).any():
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return x


def probability_logit(x: np.ndarray, epsilon: float = EPSILON) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float32), epsilon, 1.0 - epsilon)
    return np.log(x) - np.log1p(-x)


def build_v4_features(values: Mapping[str, np.ndarray]) -> np.ndarray:
    """Build fixed-order deployment features from detector probabilities.

    Features are detector logits followed by bounded interaction evidence. No
    manifest labels or source identifiers enter this representation.
    """
    raw = {name: _probability_column(values, name) for name in RAW_FEATURES}
    n = len(raw[RAW_FEATURES[0]])
    if any(len(v) != n for v in raw.values()):
        raise ValueError("All feature columns must have identical length")
    df_raw, df_voice, sonics, artifact_raw, artifact_stem, vp, mp = (
        raw[name] for name in RAW_FEATURES
    )
    artifact_max = np.maximum(artifact_raw, artifact_stem)
    music_consensus = np.maximum(sonics, artifact_max)
    return np.stack(
        [
            *(probability_logit(raw[name]) for name in RAW_FEATURES),
            probability_logit(artifact_max),
            probability_logit(music_consensus),
            df_raw - df_voice,
            artifact_raw - artifact_stem,
            sonics - artifact_max,
            vp * df_voice,
            mp * sonics,
            mp * artifact_max,
            np.abs(sonics - artifact_max),
        ],
        axis=1,
    ).astype(np.float32, copy=False)


@dataclass(frozen=True)
class V4FusionMetadata:
    version: str
    raw_features: tuple[str, ...]
    output_names: tuple[str, ...]
    input_dim: int
    hidden_dim: int
    epsilon: float

    @classmethod
    def create(cls, input_dim: int, hidden_dim: int) -> "V4FusionMetadata":
        return cls("v4.0", RAW_FEATURES, OUTPUT_NAMES, input_dim, hidden_dim, EPSILON)


class V4FusionHead(nn.Module):
    """Small shared-trunk fusion head with direct component/file supervision."""

    def __init__(self, input_dim: int, hidden_dim: int = 32, dropout: float = 0.10):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, len(OUTPUT_NAMES)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def checkpoint_payload(
    model: V4FusionHead,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    metadata: V4FusionMetadata,
    training: Mapping[str, object],
) -> dict:
    return {
        "format": "deepvoice-v4-fusion",
        "metadata": {
            "version": metadata.version,
            "raw_features": list(metadata.raw_features),
            "output_names": list(metadata.output_names),
            "input_dim": metadata.input_dim,
            "hidden_dim": metadata.hidden_dim,
            "epsilon": metadata.epsilon,
        },
        "model_state_dict": model.state_dict(),
        "feature_mean": np.asarray(feature_mean, dtype=np.float32),
        "feature_std": np.asarray(feature_std, dtype=np.float32),
        "training": dict(training),
    }


def load_v4_fusion(path: str | bytes | "os.PathLike[str]", device: str = "cpu"):
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("format") != "deepvoice-v4-fusion":
        raise ValueError("Not a DeepVoice V4 fusion checkpoint")
    meta = payload["metadata"]
    if tuple(meta["raw_features"]) != RAW_FEATURES or tuple(meta["output_names"]) != OUTPUT_NAMES:
        raise ValueError("Unsupported V4 fusion feature/output contract")
    model = V4FusionHead(meta["input_dim"], meta["hidden_dim"])
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.to(device).eval(), payload


def predict_v4_fusion(model: V4FusionHead, payload: Mapping[str, object], values: Mapping[str, np.ndarray], device: str = "cpu") -> np.ndarray:
    x = build_v4_features(values)
    mean = np.asarray(payload["feature_mean"], dtype=np.float32)
    std = np.asarray(payload["feature_std"], dtype=np.float32)
    if x.shape[1] != len(mean) or mean.shape != std.shape:
        raise ValueError("Checkpoint normalisation dimensions do not match feature contract")
    x = (x - mean) / np.maximum(std, 1e-6)
    with torch.inference_mode():
        logits = model(torch.from_numpy(x).to(device))
        return torch.sigmoid(logits).cpu().numpy()
