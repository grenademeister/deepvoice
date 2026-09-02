from pathlib import Path
import sys

import torch


def load(model_dir, device, train_head=False):
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / 'model'))
    from sonics_infer import load_sonics_model
    model = load_sonics_model(model_dir, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.classifier.parameters():
        parameter.requires_grad_(train_head)
    return model


def logits(model, audio):
    audio = audio / audio.std(dim=-1, keepdim=True).clamp_min(1e-6)
    return model(audio).reshape(-1)
