import csv
import importlib.util
from pathlib import Path

import soundfile as sf
import numpy as np


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("finetune_sonics_htdemucs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepared_stem_dataset_uses_manifest_paths_and_parent_labels(tmp_path):
    module = load_module(Path("/root/deepvoice/tools/finetune_sonics_htdemucs.py"))
    wav = tmp_path / "stem.wav"
    sf.write(wav, np.linspace(-0.1, 0.1, 16000, dtype=np.float32), 16000)
    manifest = tmp_path / "train.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "filepath", "target", "music_parent_id"])
        writer.writeheader()
        writer.writerow({"id": "sample", "filepath": wav, "target": 1, "music_parent_id": "fake-parent"})
    dataset = module.PreparedStemDataset(manifest, max_len=80000)
    audio, target = dataset[0]
    assert tuple(audio.shape) == (80000,)
    assert str(audio.dtype) == "torch.float32"
    assert target.item() == 1.0
    assert dataset.rows[0]["music_parent_id"] == "fake-parent"
import torch
import torch.nn as nn


def test_configure_trainable_unfreezes_every_parameter_for_full_fine_tuning():
    module = load_module(Path("/root/deepvoice/tools/finetune_sonics_htdemucs.py"))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.classifier = nn.Linear(2, 1)
            self.encoder = nn.Linear(2, 2)

    model = Model()
    parameters = module.configure_trainable(model, unfreeze_last_block=False, unfreeze_all=True)
    assert len(parameters) == len(list(model.parameters()))
    assert all(parameter.requires_grad for parameter in model.parameters())
