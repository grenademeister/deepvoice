# DeepVoice Improvement Plan

**Current Score (baseline max):** 0.8072  
**Target:** Top 15 on private leaderboard

---

## Step 1: Domain-specific routing (hierarchical inference)

**Status:** Abandoned

HTDemucs separation is necessary for SONICS to work correctly (raw audio drops SONICS confidence 10×). Routing provided no EER improvement. Not worth pursuing.

## Step 2: Trained fusion calibrator

**Status:** Abandoned

4→3 MLP trained on pipeline features made things worse (0.770 → 0.748).
Features from q90 pipeline are too correlated (cross-contamination).

## Step 3: SONICS fine-tuning on evalset music

**Status:** Abandoned

Full fine-tuning of 16.8M params on 370 files → severe overfitting (train_loss=0.066, val_loss=1.374). MusicEER worsened from 0.298 → 0.474.

## Step 4: DF-Arena on raw audio (best result)

**Status:** Implemented ✓

**Winner:** DF-Arena on raw audio for FILE_FAKE, DF-Arena on music stem for MUSIC_FAKE.

| Metric | Before | After | Change |
|---|---|---|---|
| **Score** | **0.8072** | **0.8384** | **+3.9%** |
| **FileEER** | **0.2363** | **0.1671** | **-29% (1.41×)** |
| MusicEER | 0.2981 | 0.2981 | same |
| VoiceEER | 0.0301 | 0.0301 | same |

**Key insight:** FILE_FAKE = max(DF-Arena on raw audio, DF-Arena on voice stem) — no SONICS, no presence gating. The 4.3B Wav2Vec2 backbone on raw audio is the most reliable single signal.

---

## Progress tracking

| Step | Started | Completed | Score gain |
|------|---------|-----------|------------|
| 1 | | | |
| 2 | | | |
| 3 | | | |
| 4 | | | |

## Step 5: ArtifactNet v2

**Status:** Implemented and validated 2026-08-28 (GTX 1080, 144 validation q90, `split=validation`)

ArtifactNet v9.4 runs independently on raw 44.1 kHz audio and the native-rate HTDemucs music stem. `MUSIC_FAKE_PROB` uses the maximum of the two median-over-chunk probabilities. File-level ArtifactNet evidence is gated by PANNs music presence and fused with the existing DF-Arena raw/voice score. Validated on same 144 validation as v0 q90; wall 407s→408s (+0.2%), VRAM 5.20G→6.25G.

| Metric | v0 q90 (2026-08-28) | v2 | Change |
|---|---:|---:|---:|
| **Score** | 0.82582 | **0.87028** | **+4.45 pts** |
| ADS | 0.80706 | **0.85646** | +4.94 pts |
| CPS | 0.99466 | 0.99466 | — |
| FileEER | 0.15274 | **0.13838** | -9.4% |
| VoiceEER | 0.03006 | 0.03006 | unchanged |
| MusicEER | 0.36851 | **0.22777** | **-38.2%** |
| Wall 144 (GTX) | 407s | 408s | +0.2% |
| Wall 1200 extrap GTX | 56.5 min | 56.7 min | <60 min |
| Wall 1200 extrap L4 22.4G | ~32 min | ~33 min | <60 min |
