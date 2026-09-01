import importlib.util
from pathlib import Path


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("evaluate_v1_sonics_balanced", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v1_file_score_uses_gated_component_maximum():
    module = load_module(Path("/root/deepvoice/tools/evaluate_v1_sonics_balanced.py"))
    assert module.v1_file_score(voice_fake=0.8, music_fake=0.9, voice_present=0.5, music_present=0.2) == 0.4
    assert abs(module.v1_file_score(voice_fake=0.1, music_fake=0.9, voice_present=0.5, music_present=0.8) - 0.72) < 1e-12
