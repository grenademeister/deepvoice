import importlib.util
import json
from pathlib import Path


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("prepare_sonics_htdemucs_music_stems", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_source_rows_is_parent_disjoint_and_balanced(tmp_path):
    module = load_module(Path("/root/deepvoice/tools/prepare_sonics_htdemucs_music_stems.py"))
    real_dir = tmp_path / "real"
    fake_dir = tmp_path / "fake"
    real_dir.mkdir()
    fake_dir.mkdir()
    for i in range(12):
        (real_dir / f"real_{i}.wav").write_bytes(b"real")
    fake_records = []
    for i in range(12):
        path = fake_dir / f"fake_{i}.wav"
        path.write_bytes(b"fake")
        fake_records.append({
            "sample_id": f"fake_{i}",
            "parent_id": f"fake-parent-{i}",
            "local_path": str(path),
            "expected_music_fake": 1,
        })
    record_path = tmp_path / "fake.jsonl"
    record_path.write_text("\n".join(json.dumps(row) for row in fake_records) + "\n")

    rows = module.build_source_rows(
        real_dir=real_dir,
        fake_records_path=record_path,
        seed=7,
        max_per_class=12,
    )
    by_split = {split: [r for r in rows if r["split"] == split] for split in ("train", "validation", "test")}
    assert {r["target"] for r in rows} == {0, 1}
    assert len([r for r in rows if r["target"] == 0]) == 12
    assert len([r for r in rows if r["target"] == 1]) == 12
    assert all({r["target"] for r in split_rows} == {0, 1} for split_rows in by_split.values())
    parent_splits = {}
    for row in rows:
        parent_splits.setdefault(row["music_parent_id"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in parent_splits.values())


def test_index_reusable_stems_finds_existing_ids(tmp_path):
    module = load_module(Path("/root/deepvoice/tools/prepare_sonics_htdemucs_music_stems.py"))
    stem = tmp_path / "audio" / "train" / "already_prepared.wav"
    stem.parent.mkdir(parents=True)
    stem.write_bytes(b"wave")
    assert module.index_reusable_stems(tmp_path) == {"already_prepared": stem.resolve()}


def test_split_rows_keeps_duplicate_fake_parent_in_one_split():
    module = load_module(Path("/root/deepvoice/tools/prepare_sonics_htdemucs_music_stems.py"))
    rows = [
        {"id": "real-1", "target": 0, "music_parent_id": "real-1"},
        {"id": "real-2", "target": 0, "music_parent_id": "real-2"},
        {"id": "real-3", "target": 0, "music_parent_id": "real-3"},
        {"id": "real-4", "target": 0, "music_parent_id": "real-4"},
        *[{"id": f"fake-a-{i}", "target": 1, "music_parent_id": "fake-a"} for i in range(5)],
        *[{"id": f"fake-b-{i}", "target": 1, "music_parent_id": "fake-b"} for i in range(5)],
    ]
    split = module.split_rows(rows, seed=7)
    fake_a_splits = {r["split"] for r in split if r["music_parent_id"] == "fake-a"}
    fake_b_splits = {r["split"] for r in split if r["music_parent_id"] == "fake-b"}
    assert len(fake_a_splits) == 1
    assert len(fake_b_splits) == 1


def test_model_input_sample_rate_matches_separator_rate():
    module = load_module(Path("/root/deepvoice/tools/prepare_sonics_htdemucs_music_stems.py"))
    assert module.model_input_sample_rate(44100) == 44100


def test_to_model_channels_duplicates_mono_for_htdemucs():
    module = load_module(Path("/root/deepvoice/tools/prepare_sonics_htdemucs_music_stems.py"))
    import torch
    mono = torch.tensor([[1.0, -1.0]])
    stereo = module.to_model_channels(mono, 2)
    assert tuple(stereo.shape) == (2, 2)
    assert torch.equal(stereo[0], mono[0])
    assert torch.equal(stereo[1], mono[0])


def test_expand_mixtures_preserves_parent_split_and_label(tmp_path):
    module = load_module(Path("/root/deepvoice/tools/prepare_sonics_htdemucs_music_stems.py"))
    rows = [
        {"id": "real-a", "filepath": "/music/real-a.wav", "target": 0,
         "music_parent_id": "real-a", "split": "train"},
        {"id": "fake-b", "filepath": "/music/fake-b.wav", "target": 1,
         "music_parent_id": "fake-b", "split": "validation"},
    ]
    donors = ["/voice/a.wav", "/voice/b.wav"]
    expanded = module.expand_mixtures(rows, donors, variants=3, seed=11)
    assert len(expanded) == 5  # train has 3 variants; validation is capped at 2.
    assert {(r["music_parent_id"], r["split"], r["target"]) for r in expanded} == {
        ("real-a", "train", 0), ("fake-b", "validation", 1)}
    assert len({r["id"] for r in expanded}) == len(expanded)
    assert {r["voice_donor"] for r in expanded}.issubset(set(donors))
