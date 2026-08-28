import pytest

from model.v2_fusion import fuse_v2_scores


def test_fusion_uses_both_artifactnet_views_for_music():
    result = fuse_v2_scores(
        df_raw=0.4,
        df_voice=0.3,
        artifact_raw=0.7,
        artifact_stem=0.8,
        music_present=0.5,
    )

    assert result.music_fake == pytest.approx(0.8)


def test_file_score_gates_artifactnet_with_music_presence():
    result = fuse_v2_scores(
        df_raw=0.4,
        df_voice=0.3,
        artifact_raw=0.7,
        artifact_stem=0.8,
        music_present=0.5,
    )

    assert result.file_fake == pytest.approx(0.4)


def test_file_score_keeps_strong_music_artifact_when_music_is_present():
    result = fuse_v2_scores(
        df_raw=0.2,
        df_voice=0.1,
        artifact_raw=0.7,
        artifact_stem=0.8,
        music_present=0.9,
    )

    assert result.file_fake == pytest.approx(0.72)
