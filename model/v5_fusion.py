"""V5 fusion model and checkpoint contract."""
from __future__ import annotations

from os import PathLike

import numpy as np
import torch
from torch import nn

SONICS_DIM = 384
SCALARS = ("df_voice", "artifact_raw", "artifact_stem", "voice_present", "music_present")
OUTPUTS = ("file_fake", "voice_fake", "music_fake")


def logits(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.clip(np.asarray(probabilities, np.float32), 1e-5, 1 - 1e-5)
    return np.log(probabilities) - np.log1p(-probabilities)


class Fusion(nn.Module):
    def __init__(self, projection: int = 64, hidden: int = 128, dropout: float = 0.15):
        super().__init__()
        self.projection, self.hidden = projection, hidden
        self.raw_projection = nn.Sequential(nn.LayerNorm(SONICS_DIM), nn.Linear(SONICS_DIM, projection), nn.GELU())
        self.stem_projection = nn.Sequential(nn.LayerNorm(SONICS_DIM), nn.Linear(SONICS_DIM, projection), nn.GELU())
        self.network = nn.Sequential(
            nn.Linear(len(SCALARS) + 4 * projection, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, len(OUTPUTS)),
        )

    def forward(self, scalars: torch.Tensor, raw: torch.Tensor, stem: torch.Tensor) -> torch.Tensor:
        raw, stem = self.raw_projection(raw), self.stem_projection(stem)
        return self.network(torch.cat((scalars, raw, stem, (raw - stem).abs(), raw * stem), dim=1))


def payload(model: Fusion, training: dict) -> dict:
    return {
        "format": "deepvoice-v5-fusion",
        "metadata": {
            "version": "v5.1", "projection": model.projection, "hidden": model.hidden,
            "scalars": list(SCALARS), "outputs": list(OUTPUTS), "sonics_dim": SONICS_DIM,
        },
        "state_dict": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
        "training": training,
    }


def load(path: str | bytes | PathLike[str], device: str = "cpu"):
    saved = torch.load(path, map_location=device, weights_only=False)
    if saved.get("format") != "deepvoice-v5-fusion":
        raise ValueError("Not a V5 fusion checkpoint")
    metadata = saved["metadata"]
    if tuple(metadata["scalars"]) != SCALARS or tuple(metadata["outputs"]) != OUTPUTS:
        raise ValueError("Unsupported V5 feature contract")
    model = Fusion(metadata["projection"], metadata["hidden"])
    model.load_state_dict(saved["state_dict"], strict=True)
    return model.to(device).eval(), saved


def predict(model: Fusion, scalars: np.ndarray, raw: np.ndarray, stem: np.ndarray, device: str = "cpu") -> np.ndarray:
    scalars, raw, stem = map(lambda value: np.asarray(value, np.float32), (scalars, raw, stem))
    count = len(scalars)
    if scalars.shape != (count, len(SCALARS)) or raw.shape != (count, SONICS_DIM) or stem.shape != raw.shape:
        raise ValueError("Invalid V5 input shapes")
    if not all(np.isfinite(value).all() for value in (scalars, raw, stem)):
        raise ValueError("V5 inputs must be finite")
    if (scalars < 0).any() or (scalars > 1).any():
        raise ValueError("Scalar probabilities must be in [0, 1]")
    with torch.inference_mode():
        output = model(
            torch.from_numpy(logits(scalars)).to(device),
            torch.from_numpy(raw).to(device), torch.from_numpy(stem).to(device),
        )
    return torch.sigmoid(output).cpu().numpy()
