import numpy as np
import torch

from model.v4_fusion import RAW_FEATURES, V4FusionHead, V4FusionMetadata, build_v4_features, checkpoint_payload, predict_v4_fusion


def values(n=4):
    return {name: np.linspace(.1, .9, n, dtype=np.float32) for name in RAW_FEATURES}


def test_feature_contract_is_finite_and_fixed_width():
    x = build_v4_features(values())
    assert x.shape == (4, 16)
    assert np.isfinite(x).all()


def test_serialized_fusion_prediction_shape(tmp_path):
    x = build_v4_features(values())
    model = V4FusionHead(x.shape[1], hidden_dim=8).eval()
    payload = checkpoint_payload(model, np.zeros(x.shape[1], np.float32), np.ones(x.shape[1], np.float32), V4FusionMetadata.create(x.shape[1], 8), {"test": True})
    path = tmp_path / "fusion.pt"
    torch.save(payload, path)
    probabilities = predict_v4_fusion(model, payload, values(), "cpu")
    assert probabilities.shape == (4, 3)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
