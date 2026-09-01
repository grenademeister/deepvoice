import importlib.util
from pathlib import Path

import numpy as np


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("artifactnet_infer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_retry_scale_caps_peak_without_amplifying_safe_audio():
    module = load_module(Path("/root/deepvoice/model/artifactnet_infer.py"))
    loud = np.array([-1.32, 1.45], dtype=np.float32)
    safe = np.array([-0.05, 0.08], dtype=np.float32)
    assert np.isclose(module.artifactnet_retry_scale(loud, peak_limit=0.1), 0.1 / 1.45)
    assert module.artifactnet_retry_scale(safe, peak_limit=0.1) == 1.0
