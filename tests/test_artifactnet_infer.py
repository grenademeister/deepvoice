import numpy as np
import pytest
from pathlib import Path

from model.artifactnet_infer import (
    ARTIFACTNET_CHUNK_SAMPLES,
    _preload_cuda_dependencies,
    _select_execution_providers,
    load_artifactnet_model,
    make_artifactnet_chunks,
    predict_artifactnet,
    predict_artifactnet_raw_and_stem,
)


def test_prefers_cuda_then_keeps_cpu_fallback():
    providers = _select_execution_providers(
        ["AzureExecutionProvider", "CPUExecutionProvider", "CUDAExecutionProvider"]
    )

    assert providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_uses_cpu_when_cuda_is_unavailable():
    assert _select_execution_providers(["CPUExecutionProvider"]) == ["CPUExecutionProvider"]


def test_preloads_cuda_dependencies_only_for_cuda_provider():
    class FakeOrt:
        calls = 0

        @classmethod
        def preload_dlls(cls):
            cls.calls += 1

    _preload_cuda_dependencies(FakeOrt, ["CUDAExecutionProvider", "CPUExecutionProvider"])
    assert FakeOrt.calls == 1

    _preload_cuda_dependencies(FakeOrt, ["CPUExecutionProvider"])
    assert FakeOrt.calls == 1


def test_exact_four_second_clip_creates_one_chunk():
    audio = np.linspace(-1.0, 1.0, ARTIFACTNET_CHUNK_SAMPLES, dtype=np.float32)

    chunks = make_artifactnet_chunks(audio, sample_rate=44_100)

    assert chunks.shape == (1, ARTIFACTNET_CHUNK_SAMPLES)
    assert chunks.dtype == np.float32
    np.testing.assert_array_equal(chunks[0], audio)


def test_short_clip_is_zero_padded():
    audio = np.arange(100, dtype=np.float32)

    chunks = make_artifactnet_chunks(audio, sample_rate=44_100)

    assert chunks.shape == (1, ARTIFACTNET_CHUNK_SAMPLES)
    np.testing.assert_array_equal(chunks[0, :100], audio)
    np.testing.assert_array_equal(chunks[0, 100:], 0.0)


def test_remainder_adds_end_aligned_chunk():
    audio = np.arange(ARTIFACTNET_CHUNK_SAMPLES * 2 + 100, dtype=np.float32)

    chunks = make_artifactnet_chunks(audio, sample_rate=44_100)

    assert chunks.shape == (3, ARTIFACTNET_CHUNK_SAMPLES)
    np.testing.assert_array_equal(chunks[0], audio[:ARTIFACTNET_CHUNK_SAMPLES])
    np.testing.assert_array_equal(
        chunks[1], audio[ARTIFACTNET_CHUNK_SAMPLES:2 * ARTIFACTNET_CHUNK_SAMPLES]
    )
    np.testing.assert_array_equal(chunks[2], audio[-ARTIFACTNET_CHUNK_SAMPLES:])


def test_resamples_to_44100_hz():
    audio = np.ones(16_000, dtype=np.float32)

    chunks = make_artifactnet_chunks(audio, sample_rate=16_000)

    assert chunks.shape == (1, ARTIFACTNET_CHUNK_SAMPLES)
    assert np.count_nonzero(chunks[0]) >= 44_000


@pytest.mark.parametrize("audio", [np.array([], dtype=np.float32), np.array([np.nan], dtype=np.float32)])
def test_rejects_invalid_audio(audio):
    with pytest.raises(ValueError):
        make_artifactnet_chunks(audio, sample_rate=44_100)


class FakeSession:
    def __init__(self):
        self.inputs = []

    class Input:
        name = "audio"

    def get_inputs(self):
        return [self.Input()]

    def run(self, output_names, inputs):
        assert output_names is None
        chunk = inputs["audio"]
        self.inputs.append(chunk.copy())
        return [np.array([len(self.inputs) / 10], dtype=np.float32)]


def test_predicts_each_chunk_and_returns_median():
    session = FakeSession()
    audio = np.ones(ARTIFACTNET_CHUNK_SAMPLES * 3, dtype=np.float32)

    probability = predict_artifactnet(session, audio, sample_rate=44_100)

    assert probability == pytest.approx(0.2)
    assert len(session.inputs) == 3
    assert all(value.shape == (1, ARTIFACTNET_CHUNK_SAMPLES) for value in session.inputs)
    assert all(value.dtype == np.float32 for value in session.inputs)


def test_silence_returns_zero_without_running_onnx():
    session = FakeSession()

    probability = predict_artifactnet(
        session,
        np.zeros(ARTIFACTNET_CHUNK_SAMPLES, dtype=np.float32),
        sample_rate=44_100,
    )

    assert probability == 0.0
    assert session.inputs == []


def test_non_silent_chunks_replace_exact_zeros_before_onnx():
    session = FakeSession()
    audio = np.zeros(ARTIFACTNET_CHUNK_SAMPLES, dtype=np.float32)
    audio[: ARTIFACTNET_CHUNK_SAMPLES // 2] = 0.1

    predict_artifactnet(session, audio, sample_rate=44_100)

    assert np.count_nonzero(session.inputs[0]) == ARTIFACTNET_CHUNK_SAMPLES
    assert np.max(np.abs(session.inputs[0][0, ARTIFACTNET_CHUNK_SAMPLES // 2 :])) <= 1e-9


def test_retries_nonfinite_output_with_rms_normalization():
    class LoudnessSensitiveSession(FakeSession):
        def run(self, output_names, inputs):
            chunk = inputs["audio"]
            self.inputs.append(chunk.copy())
            rms = float(np.sqrt(np.mean(np.square(chunk, dtype=np.float64))))
            value = np.nan if rms > 0.11 else 0.25
            return [np.array([value], dtype=np.float32)]

    session = LoudnessSensitiveSession()
    probability = predict_artifactnet(
        session,
        np.full(ARTIFACTNET_CHUNK_SAMPLES, 0.5, dtype=np.float32),
        sample_rate=44_100,
    )

    assert probability == pytest.approx(0.25)
    assert len(session.inputs) == 2
    retry_rms = float(np.sqrt(np.mean(np.square(session.inputs[1], dtype=np.float64))))
    assert retry_rms == pytest.approx(0.1, abs=1e-6)


def test_real_model_contract_and_probability():
    model_dir = Path(__file__).resolve().parents[1] / "model" / "artifactnet"
    if not (model_dir / "artifactnet_v94_full.onnx").is_file():
        pytest.skip("ArtifactNet model is not installed")

    session = load_artifactnet_model(model_dir, providers=["CPUExecutionProvider"])
    probability = predict_artifactnet(
        session,
        np.random.default_rng(0).normal(
            0.0, 0.1, ARTIFACTNET_CHUNK_SAMPLES
        ).astype(np.float32),
        sample_rate=44_100,
    )

    assert session.get_inputs()[0].name == "audio"
    assert session.get_inputs()[0].shape == ["batch", ARTIFACTNET_CHUNK_SAMPLES]
    assert 0.0 <= probability <= 1.0


def test_runs_artifactnet_on_both_raw_audio_and_music_stem():
    session = FakeSession()
    raw_audio = np.ones(ARTIFACTNET_CHUNK_SAMPLES, dtype=np.float32)
    music_stem = np.ones(ARTIFACTNET_CHUNK_SAMPLES, dtype=np.float32) * 0.5

    raw_probability, stem_probability = predict_artifactnet_raw_and_stem(
        session,
        raw_audio=raw_audio,
        raw_sample_rate=44_100,
        music_stem=music_stem,
        stem_sample_rate=44_100,
    )

    assert raw_probability == pytest.approx(0.1)
    assert stem_probability == pytest.approx(0.2)
    assert len(session.inputs) == 2
    np.testing.assert_array_equal(session.inputs[0][0], raw_audio)
    np.testing.assert_array_equal(session.inputs[1][0], music_stem)
