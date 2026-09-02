from pathlib import Path
import json
import shutil

import numpy as np


def prepare_labels(model_dir):
    source = Path(model_dir) / 'class_labels_indices.csv'
    if not source.is_file():
        raise FileNotFoundError(f'PANNs class labels missing: {source}')
    target = Path.home() / 'panns_data' / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def load(model_dir, device):
    from panns_inference import AudioTagging, labels
    model_dir = Path(model_dir)
    prepare_labels(model_dir)
    groups = json.loads((model_dir / 'component_labels.json').read_text())
    index = {name: i for i, name in enumerate(labels)}
    model = AudioTagging(checkpoint_path=str(model_dir / 'Cnn14_mAP=0.431.pth'), device=device.type)
    return model, [index[x] for x in groups['voice']], [index[x] for x in groups['music']]


def score(model, voice, music, audio):
    import librosa
    x = librosa.resample(audio, orig_sr=16000, target_sr=32000, res_type='soxr_hq')
    probabilities, _ = model.inference(x[None].astype(np.float32))
    return float(probabilities[:, voice].max()), float(probabilities[:, music].max())
