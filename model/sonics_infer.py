#!/usr/bin/env python3
"""Offline inference for the SONICS SpecTTTra detector (awsaf49/sonics-spectttra-alpha-5s).

This is a self-contained reimplementation that reconstructs the exact model
architecture from the SONICS config.json and loads the official
pytorch_model.bin weights. It depends only on torch / torchaudio / numpy so it
runs on the competition image without the `sonics` or `timm` packages.

Output of the model is a single logit; the fake probability is sigmoid(logit).
"""

import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram

class MeanStdNorm(nn.Module):
    def forward(self, x):
        mean = x.mean((1, 2), keepdim=True)
        std = x.reshape(x.size(0), -1).std(1, keepdim=True).unsqueeze(-1)
        return (x - mean) / (std + 1e-6)


class FeatureExtractor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.audio2melspec = MelSpectrogram(
            n_fft=cfg["melspec"]["n_fft"],
            hop_length=cfg["melspec"]["hop_length"],
            win_length=cfg["melspec"]["win_length"],
            n_mels=cfg["melspec"]["n_mels"],
            sample_rate=cfg["audio"]["sample_rate"],
            f_min=cfg["melspec"]["f_min"],
            f_max=cfg["melspec"]["f_max"],
            power=cfg["melspec"]["power"],
        )
        self.amplitude_to_db = AmplitudeToDB(top_db=cfg["melspec"]["top_db"])
        self.normalizer = MeanStdNorm()

    def forward(self, x):
        melspec = self.audio2melspec(x.float())
        melspec = self.amplitude_to_db(melspec)
        melspec = self.normalizer(melspec)
        return melspec


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, token_dim, num_tokens):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, num_tokens, token_dim) * 0.02)

    def forward(self, x):
        return x + self.pe


class Tokenizer1D(nn.Module):
    def __init__(self, input_dim, token_dim, clip_size, num_clips):
        super().__init__()
        self.conv1d = nn.Conv1d(
            input_dim, token_dim, clip_size, stride=clip_size, bias=False
        )
        self.act = nn.GELU()
        self.pos_encoder = LearnedPositionalEncoding(token_dim, num_clips)
        self.norm_pre = nn.LayerNorm(token_dim, eps=1e-6)

    def forward(self, x):
        x = self.act(self.conv1d(x))
        x = x.transpose(1, 2)
        x = self.pos_encoder(x)
        x = self.norm_pre(x)
        return x


class STTokenizer(nn.Module):
    def __init__(self, input_spec_dim, input_temp_dim, t_clip, f_clip, embed_dim):
        super().__init__()
        num_temporal_tokens = math.floor((input_temp_dim - t_clip) / t_clip + 1)
        num_spectral_tokens = math.floor((input_spec_dim - f_clip) / f_clip + 1)
        self.temporal_tokenizer = Tokenizer1D(
            input_spec_dim, embed_dim, t_clip, num_temporal_tokens
        )
        self.spectral_tokenizer = Tokenizer1D(
            input_temp_dim, embed_dim, f_clip, num_spectral_tokens
        )

    def forward(self, x):
        temporal_tokens = self.temporal_tokenizer(x)
        spectral_tokens = self.spectral_tokenizer(x.permute(0, 2, 1))
        return torch.cat((temporal_tokens, spectral_tokens), dim=1)


class Attention(nn.Module):
    def __init__(self, dim, num_heads, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(self.proj(x))
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return self.drop(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, proj_drop, attn_drop):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=proj_drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self, dim, num_heads, num_layers, mlp_ratio=4.0, proj_drop=0.0, attn_drop=0.0
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(dim, num_heads, mlp_ratio, proj_drop, attn_drop)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class SpecTTTra(nn.Module):
    def __init__(self, input_spec_dim, input_temp_dim, embed_dim, t_clip, f_clip,
                 num_heads, num_layers, mlp_ratio=4.0, pos_drop_rate=0.0,
                 proj_drop_rate=0.0, attn_drop_rate=0.0):
        super().__init__()
        self.st_tokenizer = STTokenizer(
            input_spec_dim, input_temp_dim, t_clip, f_clip, embed_dim
        )
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        self.transformer = Transformer(
            embed_dim,
            num_heads,
            num_layers,
            mlp_ratio=mlp_ratio,
            proj_drop=proj_drop_rate,
            attn_drop=attn_drop_rate,
        )

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(1)
        x = self.st_tokenizer(x)
        x = self.pos_drop(x)
        return self.transformer(x)


class SonicsClassifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        model_cfg = cfg["model"]
        self.input_shape = tuple(model_cfg["input_shape"])
        self.ft_extractor = FeatureExtractor(cfg)
        self.encoder = SpecTTTra(
            input_spec_dim=model_cfg["input_shape"][0],
            input_temp_dim=model_cfg["input_shape"][1],
            embed_dim=model_cfg["embed_dim"],
            t_clip=model_cfg["t_clip"],
            f_clip=model_cfg["f_clip"],
            num_heads=model_cfg["num_heads"],
            num_layers=model_cfg["num_layers"],
            mlp_ratio=model_cfg["mlp_ratio"],
            pos_drop_rate=model_cfg["pos_drop_rate"],
            proj_drop_rate=model_cfg["proj_drop_rate"],
            attn_drop_rate=model_cfg["attn_drop_rate"],
        )
        self.classifier = nn.Linear(model_cfg["embed_dim"], cfg["num_classes"])

    def forward(self, audio):
        spec = self.ft_extractor(audio)
        spec = spec.unsqueeze(1)
        spec = F.interpolate(spec, size=self.input_shape, mode="bilinear")
        features = self.encoder(spec)
        embeds = features.mean(dim=1)
        return self.classifier(embeds)


def load_sonics_model(model_dir, device):
    with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    model = SonicsClassifier(cfg)
    state_dict = torch.load(
        os.path.join(model_dir, "pytorch_model.bin"),
        map_location="cpu",
        weights_only=True,
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"SONICS state dict mismatch. missing={missing} unexpected={unexpected}"
        )
    return model.to(device).eval()


def preprocess_window(audio, sample_rate, max_len):
    """Replicate the SONICS validation preprocessing for a single clip."""
    n = audio.shape[0]
    if n >= max_len:
        idx = int((n - max_len) / 4 * 3)
        audio = audio[idx : idx + max_len]
    else:
        audio = np.pad(audio, (0, max_len - n), mode="constant")
    audio = audio / max(float(np.std(audio)), 1e-6)
    return audio.astype(np.float32)
