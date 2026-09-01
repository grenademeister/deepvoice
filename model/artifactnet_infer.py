"""Offline ArtifactNet v9.4 ONNX inference utilities."""

from __future__ import annotations

from math import gcd
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np
from scipy.signal import resample_poly

ARTIFACTNET_SAMPLE_RATE = 44_100
ARTIFACTNET_CHUNK_SECONDS = 4
ARTIFACTNET_CHUNK_SAMPLES = ARTIFACTNET_SAMPLE_RATE * ARTIFACTNET_CHUNK_SECONDS
ARTIFACTNET_ZERO_EPS = np.float32(1e-12)


class ArtifactNetSession(Protocol):
    def get_inputs(self) -> Sequence[Any]: ...

    def run(self, output_names: Any, inputs: dict[str, np.ndarray]) -> Sequence[np.ndarray]: ...


def _select_execution_providers(available: Sequence[str]) -> list[str]:
    providers = []
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    if "CPUExecutionProvider" in available:
        providers.append("CPUExecutionProvider")
    if not providers:
        raise RuntimeError("ArtifactNet requires CUDA or CPU ONNX Runtime provider")
    return providers


def _preload_cuda_dependencies(ort: Any, providers: Sequence[str]) -> None:
    if "CUDAExecutionProvider" in providers:
        ort.preload_dlls()


def load_artifactnet_model(
    model_dir: Path,
    providers: list[str] | None = None,
) -> ArtifactNetSession:
    """Load the pinned ArtifactNet ONNX graph for offline inference."""
    import onnxruntime as ort

    model_path = Path(model_dir) / "artifactnet_v94_full.onnx"
    data_path = Path(f"{model_path}.data")
    if not model_path.is_file() or not data_path.is_file():
        raise FileNotFoundError(f"ArtifactNet model files not found under {model_dir}")

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    selected_providers = providers or _select_execution_providers(ort.get_available_providers())
    _preload_cuda_dependencies(ort, selected_providers)
    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=selected_providers,
    )
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or inputs[0].name != "audio":
        raise RuntimeError("Unexpected ArtifactNet input contract")
    if inputs[0].type != "tensor(float)" or inputs[0].shape != ["batch", ARTIFACTNET_CHUNK_SAMPLES]:
        raise RuntimeError(f"Unexpected ArtifactNet input shape/type: {inputs[0].shape}, {inputs[0].type}")
    if len(outputs) != 1 or outputs[0].type != "tensor(float)":
        raise RuntimeError("Unexpected ArtifactNet output contract")
    return session


def make_artifactnet_chunks(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Convert mono audio into ArtifactNet's fixed-size 4-second chunks."""
    values = np.asarray(audio, dtype=np.float32)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("ArtifactNet audio must be a non-empty mono waveform")
    if not np.isfinite(values).all():
        raise ValueError("ArtifactNet audio contains NaN or Inf")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if sample_rate != ARTIFACTNET_SAMPLE_RATE:
        divisor = gcd(sample_rate, ARTIFACTNET_SAMPLE_RATE)
        values = resample_poly(
            values,
            ARTIFACTNET_SAMPLE_RATE // divisor,
            sample_rate // divisor,
        ).astype(np.float32, copy=False)
    if values.size <= ARTIFACTNET_CHUNK_SAMPLES:
        return np.pad(
            values,
            (0, ARTIFACTNET_CHUNK_SAMPLES - values.size),
        )[None, :].astype(np.float32, copy=False)

    starts = list(range(0, values.size - ARTIFACTNET_CHUNK_SAMPLES + 1, ARTIFACTNET_CHUNK_SAMPLES))
    final_start = values.size - ARTIFACTNET_CHUNK_SAMPLES
    if starts[-1] != final_start:
        starts.append(final_start)
    return np.stack([values[start:start + ARTIFACTNET_CHUNK_SAMPLES] for start in starts])


def artifactnet_retry_scale(audio: np.ndarray, peak_limit: float = 0.1) -> float:
    """Return a non-amplifying scale that bounds retry input amplitude."""
    peak = float(np.max(np.abs(audio)))
    if not np.isfinite(peak) or peak <= 0.0:
        return 1.0
    return min(1.0, peak_limit / peak)


def predict_artifactnet(
    session: ArtifactNetSession,
    audio: np.ndarray,
    sample_rate: int,
) -> float:
    """Return the median ArtifactNet probability over all 4-second chunks."""
    chunks = make_artifactnet_chunks(audio, sample_rate)
    if float(np.max(np.abs(chunks))) <= 1e-7:
        return 0.0

    input_name = session.get_inputs()[0].name
    probabilities = []
    for chunk in chunks:
        model_chunk = chunk.copy()
        model_chunk[model_chunk == 0.0] = ARTIFACTNET_ZERO_EPS
        outputs = session.run(None, {input_name: model_chunk[None, :]})
        probability = float(np.asarray(outputs[0]).reshape(-1)[0])
        if not np.isfinite(probability):
            retry_scale = artifactnet_retry_scale(model_chunk)
            model_chunk = (model_chunk * retry_scale).astype(np.float32)
            outputs = session.run(None, {input_name: model_chunk[None, :]})
            probability = float(np.asarray(outputs[0]).reshape(-1)[0])
        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(f"Invalid ArtifactNet probability after retry: {probability}")
        probabilities.append(probability)
    return float(np.median(np.asarray(probabilities, dtype=np.float64)))


def predict_artifactnet_raw_and_stem(
    session: ArtifactNetSession,
    *,
    raw_audio: np.ndarray,
    raw_sample_rate: int,
    music_stem: np.ndarray,
    stem_sample_rate: int,
) -> tuple[float, float]:
    """Run ArtifactNet independently on raw audio and the HTDemucs music stem."""
    raw_probability = predict_artifactnet(session, raw_audio, raw_sample_rate)
    stem_probability = predict_artifactnet(session, music_stem, stem_sample_rate)
    return raw_probability, stem_probability
