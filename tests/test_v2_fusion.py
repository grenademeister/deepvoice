import pytest

from model.v2_fusion import fuse_v2_scores


def test_fusion_uses_both_artifactnet_views_for_music():
    result = fuse_v2_scores(
        df_raw=0.4,
        df_voice=0.3,
        artifact_raw=0.7,
        artifact_stem=0.8,
        music_present=0.5,
        voice_present=0.9,
    )

    assert result.music_fake == pytest.approx(0.8)


def test_file_score_gates_artifactnet_with_music_presence():
    result = fuse_v2_scores(
        df_raw=0.4,
        df_voice=0.3,
        artifact_raw=0.7,
        artifact_stem=0.8,
        music_present=0.5,
        voice_present=0.9,
    )

    assert result.file_fake == pytest.approx(0.4)


def test_file_score_keeps_strong_music_artifact_when_music_is_present():
    result = fuse_v2_scores(
        df_raw=0.2,
        df_voice=0.1,
        artifact_raw=0.7,
        artifact_stem=0.8,
        music_present=0.9,
        voice_present=0.9,
    )

    assert result.file_fake == pytest.approx(0.72)


def test_file_score_gates_df_voice_with_voice_presence_h1():
    # gtzan music-only: DF hallucination 0.84 but vp low -> should be suppressed
    result = fuse_v2_scores(
        df_raw=0.05,
        df_voice=0.846,
        artifact_raw=0.0005,
        artifact_stem=0.0005,
        music_present=0.93,
        voice_present=0.15,  # PANNs correctly low on music-only
    )

    assert result.file_fake == pytest.approx(0.1269, rel=1e-3)
    # Without gate, file would be 0.846 — H1 prevents FP
