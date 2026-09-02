from pathlib import Path


def test_clean_v4_has_only_online_mixing_contract():
    root = Path(__file__).resolve().parents[1]
    script = root / "script.py"
    assert len(script.read_text().splitlines()) < 300
    source = script.read_text()
    assert "class OnlineMixtureDataset" in source
    assert "df_raw" not in source
    assert "v4_fast" not in source
    for name in ("dfarena.py", "sonics.py", "artifactnet.py", "panns.py", "separator.py"):
        assert (root / "models" / name).is_file()
