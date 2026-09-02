from pathlib import Path
import sys

import numpy as np
import torch

WINDOW = 64600


def load(model_dir, device):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'model'))
    from df_arena_1b.modeling_antispoofing import DF_Arena_1B_Antispoofing
    model = DF_Arena_1B_Antispoofing.from_pretrained(str(model_dir), local_files_only=True).to(device).eval()
    return model, int(model.config.label2id['spoof'])


def score(model, fake_index, audio, device):
    if np.sqrt(np.mean(np.square(audio, dtype=np.float64))) < 1e-5:
        return 0.0
    starts = list(range(0, max(1, len(audio) - WINDOW + 1), WINDOW))
    starts.append(max(0, len(audio) - WINDOW))
    values = []
    with torch.inference_mode():
        for start in sorted(set(starts)):
            x = audio[start:start + WINDOW]
            x = np.pad(x, (0, max(0, WINDOW - len(x))))
            values.append(torch.softmax(model(input_values=torch.from_numpy(x).to(device))["logits"], -1)[0, fake_index].item())
    return float(np.quantile(values, .9))
