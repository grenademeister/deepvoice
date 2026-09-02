from pathlib import Path

import torch
import torchaudio
from demucs.apply import apply_model
from demucs.pretrained import get_model


def load(model_dir, device):
    original = torch.load
    def compatible_load(*args, **kwargs):
        kwargs.setdefault('weights_only', False)
        return original(*args, **kwargs)
    torch.load = compatible_load
    try:
        model = get_model('htdemucs', repo=Path(model_dir)).to(device).eval()
    finally:
        torch.load = original
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def separate(model, mixture, sample_rate=16000):
    # mixture [B,T], returns [B,T] vocal/accompaniment at 16 kHz.
    x = mixture[:, None].repeat(1, model.audio_channels, 1)
    x = torchaudio.functional.resample(x, sample_rate, model.samplerate)
    mean, std = x.mean((1, 2), keepdim=True), x.std((1, 2), keepdim=True).clamp_min(1e-6)
    with torch.no_grad():
        stems = apply_model(model, (x - mean) / std, device=x.device, shifts=0, split=True, overlap=.25)
    stems = stems * std[:, None] + mean[:, None]
    vocals = stems[:, model.sources.index('vocals')].mean(1)
    music = stems[:, [i for i, name in enumerate(model.sources) if name != 'vocals']].sum(1).mean(1)
    return (torchaudio.functional.resample(vocals, model.samplerate, sample_rate),
            torchaudio.functional.resample(music, model.samplerate, sample_rate))
