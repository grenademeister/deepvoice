import json
from pathlib import Path

import numpy as np
import pytest
import torch

from model.v5_fusion import SONICS_DIM, Fusion, load, payload, predict
from tools.prepare_v5 import Cache
from tools.train_v5 import Config, Dataset, masked_loss, train


def row(sample_id: str, split: str, index: int) -> dict[str, str]:
    label = index % 2
    return {
        "sample_id": sample_id, "split": split,
        "expected_file_fake": str(label), "expected_voice_fake": str(label),
        "expected_music_fake": str(label), "expected_voice_present": str((index // 2) % 2),
        "expected_music_present": str(((index + 1) // 2) % 2),
    }


def dataset(split: str, count: int) -> Dataset:
    rng = np.random.default_rng(41 if split == "train" else 42)
    rows = [row(f"{split}_{index}", split, index) for index in range(count)]
    return Dataset(
        rows=rows,
        scalars=rng.normal(size=(count, 5)).astype(np.float32),
        raw=rng.normal(size=(count, SONICS_DIM)).astype(np.float32),
        stem=rng.normal(size=(count, SONICS_DIM)).astype(np.float32),
        labels=np.array([[index % 2] * 3 for index in range(count)], np.float32),
        masks=np.ones((count, 3), np.float32),
        presence=np.array([[index % 2, (index + 1) % 2] for index in range(count)], np.float32),
    )


def test_cache_validates_shapes_and_contract(tmp_path: Path):
    cache = Cache(tmp_path)
    contract = {"manifest_sha256": "abc"}
    cache.validate_contract(contract)
    cache.validate_contract(contract)
    with pytest.raises(ValueError):
        cache.validate_contract({"manifest_sha256": "different"})
    cache.write_stems("sample", np.zeros(160, np.float32), np.ones(160, np.float32))
    cache.write_df("sample", 0.25)
    cache.write_sonics("sample", np.zeros(SONICS_DIM, np.float32), np.ones(SONICS_DIM, np.float32))
    assert cache.stems("sample", 160) is not None
    assert cache.stems("sample", 159) is None
    assert cache.df("sample") == pytest.approx(0.25)
    assert cache.sonics("sample") is not None


def test_unversioned_cache_requires_explicit_adoption(tmp_path: Path):
    cache = Cache(tmp_path)
    cache.write_df("sample", 0.5)
    with pytest.raises(ValueError, match="adopt-cache"):
        cache.validate_contract({"schema": 1, "manifest_sha256": "abc"})
    cache.validate_contract({"schema": 1, "manifest_sha256": "abc"}, adopt=True)
    assert json.loads((tmp_path / "cache_metadata.json").read_text())["schema"] == 1


def test_masked_loss_ignores_absent_components():
    logits = torch.tensor([[0.0, 100.0, -100.0]])
    labels = torch.zeros_like(logits)
    masks = torch.tensor([[1.0, 0.0, 0.0]])
    assert float(masked_loss(logits, labels, masks)) == pytest.approx(np.log(2), rel=1e-5)


def test_deployment_checkpoint_round_trip(tmp_path: Path):
    model = Fusion(projection=8, hidden=16).eval()
    saved = payload(model, {"test": True})
    path = tmp_path / "model.pt"
    torch.save(saved, path)
    loaded, _ = load(path)
    probabilities = np.full((3, 5), 0.5, np.float32)
    embeddings = np.zeros((3, SONICS_DIM), np.float32)
    output = predict(loaded, probabilities, embeddings, embeddings)
    assert output.shape == (3, 3)
    assert np.all((0 <= output) & (output <= 1))


def test_batch_checkpoint_resume_matches_uninterrupted(tmp_path: Path):
    training, validation = dataset("train", 6), dataset("validation", 4)
    config = Config(epochs=1, batch_size=2, projection=8, hidden=16, seed=7)
    contract = {"manifest_sha256": "test"}
    uninterrupted = tmp_path / "uninterrupted"
    resumed = tmp_path / "resumed"
    train(training, validation, contract, uninterrupted, config, torch.device("cpu"), 1, False, 0)
    train(training, validation, contract, resumed, config, torch.device("cpu"), 1, False, 1)
    train(training, validation, contract, resumed, config, torch.device("cpu"), 1, True, 0)
    first = torch.load(uninterrupted / "last.pt", map_location="cpu", weights_only=False)
    second = torch.load(resumed / "last.pt", map_location="cpu", weights_only=False)
    assert first["history"] == second["history"]
    for name, value in first["model"].items():
        torch.testing.assert_close(value, second["model"][name], rtol=0, atol=0)
