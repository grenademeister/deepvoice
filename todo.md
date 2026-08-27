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