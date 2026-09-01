"""Deterministic score fusion for DeepVoice v2."""

from dataclasses import dataclass


@dataclass(frozen=True)
class V2Scores:
    file_fake: float
    music_fake: float


def fuse_v2_scores(
    *,
    df_raw: float,
    df_voice: float,
    artifact_raw: float,
    artifact_stem: float,
    music_present: float,
    voice_present: float,
) -> V2Scores:
    values = (df_raw, df_voice, artifact_raw, artifact_stem, music_present, voice_present)
    if any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("All fusion inputs must lie in [0, 1]")
    music_fake = max(artifact_raw, artifact_stem)
    file_fake = max(df_raw, voice_present * df_voice, music_present * music_fake)
    return V2Scores(file_fake=file_fake, music_fake=music_fake)
