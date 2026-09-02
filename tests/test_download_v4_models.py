from pathlib import Path


def test_model_bundle_validation_requires_all_runtime_weights(tmp_path):
    from tools.download_v4_models import validate
    (tmp_path / 'df_arena_1b').mkdir()
    try:
        validate(tmp_path)
    except FileNotFoundError as error:
        assert 'df_arena_1b/pytorch_model.bin' in str(error)
    else:
        raise AssertionError('incomplete model bundle accepted')
    for name in ('df_arena_1b/pytorch_model.bin', 'sonics/config.json', 'sonics/pytorch_model.bin',
                 'artifactnet/artifactnet_v94_full.onnx', 'artifactnet/artifactnet_v94_full.onnx.data',
                 'panns/Cnn14_mAP=0.431.pth', 'htdemucs/htdemucs.yaml', 'htdemucs/955717e8-8726e21a.th'):
        path = tmp_path / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'x')
    validate(tmp_path)
