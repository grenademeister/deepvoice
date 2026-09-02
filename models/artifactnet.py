from pathlib import Path
import sys


def load(model_dir):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'model'))
    from artifactnet_infer import load_artifactnet_model
    return load_artifactnet_model(Path(model_dir))


def score(session, audio, sample_rate):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'model'))
    from artifactnet_infer import predict_artifactnet
    return predict_artifactnet(session, audio, sample_rate)
