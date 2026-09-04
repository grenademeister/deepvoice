"""V5 fusion — direct SONICS embeddings + scalar detectors."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn

SONICS_DIM = 384
SCALARS = ("df_voice", "artifact_raw", "artifact_stem", "voice_present", "music_present")
OUTPUTS = ("file_fake", "voice_fake", "music_fake")
EPS = 1e-5


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, np.float32), EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


@dataclass(frozen=True)
class V5Meta:
    version: str
    proj_dim: int
    hidden_dim: int
    df_variant: str = "500m"

    @classmethod
    def create(cls, proj_dim: int, hidden_dim: int, df_variant: str = "500m") -> "V5Meta":
        return cls("v5.0", int(proj_dim), int(hidden_dim), str(df_variant))


class V5Fusion(nn.Module):
    def __init__(self, sonics_dim: int = SONICS_DIM, proj_dim: int = 64, hidden_dim: int = 128, dropout: float = 0.15):
        super().__init__()
        self.proj_dim = proj_dim
        self.raw_proj = nn.Sequential(nn.LayerNorm(sonics_dim), nn.Linear(sonics_dim, proj_dim), nn.GELU())
        self.stem_proj = nn.Sequential(nn.LayerNorm(sonics_dim), nn.Linear(sonics_dim, proj_dim), nn.GELU())
        in_dim = 5 + proj_dim * 4
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, len(OUTPUTS)),
        )

    def forward(self, scalars: torch.Tensor, raw_emb: torch.Tensor, stem_emb: torch.Tensor) -> torch.Tensor:
        pr = self.raw_proj(raw_emb)
        ps = self.stem_proj(stem_emb)
        return self.net(torch.cat([scalars, pr, ps, (pr - ps).abs(), pr * ps], dim=1))


def payload(model: V5Fusion, meta: V5Meta, s_mean: np.ndarray, s_std: np.ndarray, history: list) -> dict:
    return {
        "format": "deepvoice-v5-fusion",
        "metadata": {"version": meta.version, "proj_dim": meta.proj_dim, "hidden_dim": meta.hidden_dim,
                     "df_variant": meta.df_variant, "scalars": list(SCALARS), "sonics_dim": SONICS_DIM,
                     "outputs": list(OUTPUTS), "epsilon": EPS},
        "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "scalar_mean": np.asarray(s_mean, np.float32),
        "scalar_std": np.asarray(s_std, np.float32),
        "history": list(history),
    }
