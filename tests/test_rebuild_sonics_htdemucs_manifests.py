import importlib.util
from pathlib import Path


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("rebuild_sonics_htdemucs_manifests", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_index_stems_finds_ids_across_previous_split_directories(tmp_path):
    module = load_module(Path("/root/deepvoice/tools/rebuild_sonics_htdemucs_manifests.py"))
    stem = tmp_path / "previous" / "audio" / "validation" / "fake_x__v00.wav"
    stem.parent.mkdir(parents=True)
    stem.write_bytes(b"wave")
    assert module.index_stems(tmp_path / "previous") == {"fake_x__v00": stem.resolve()}
