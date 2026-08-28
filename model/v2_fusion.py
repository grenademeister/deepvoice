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
    """Fuse DF-Arena with raw/stem ArtifactNet evidence.

    ArtifactNet's strongest view drives the music score. Its contribution to the
    file score is gated by PANNs music presence to protect non-music inputs.
    DF-Arena voice evidence is gated by voice presence to suppress
    HTDemucs hallucination on pure music (gtzan mean VOICE_FAKE 0.862 on
    music-only). H1 validated 2026-08-28: FileEER 0.138->0.118, Score 0.870->0.879.
    """
    values = (df_raw, df_voice, artifact_raw, artifact_stem, music_present, voice_present)
    if any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("All fusion inputs must lie in [0, 1]")
    music_fake = max(artifact_raw, artifact_stem)
    file_fake = max(df_raw, voice_present * df_voice, music_present * music_fake)
    return V2Scores(file_fake=file_fake, music_fake=music_fake)
