# DeepVoice Improvement Plan

**Current best validated score:** 0.87939 (`v2` H1)

**Target:** Breakthrough v3 with robust cross-generator generalization

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
| 1 — routing | 2026-08 | Abandoned | none |
| 2 — fusion calibrator | 2026-08 | Abandoned | regression |
| 3 — SONICS fine-tuning | 2026-08 | Abandoned | regression |
| 4 — DF-Arena raw | 2026-08 | 2026-08-28 | 0.8072 → 0.8384 |
| 5 — ArtifactNet v2 H1 | 2026-08-28 | 2026-08-28 | 0.8384 → 0.87939 |
| 6 — v3 bottleneck audit | 2026-08-28 | Planned | prerequisite |
| 7 — v3 representation replacement | 2026-08-28 | Planned | target 0.94–0.95 |

## Step 5: ArtifactNet v2 — H1 `vp*df_voice`

**Status:** Implemented and validated 2026-08-28 (GTX 1080, 144 validation q90, `split=validation`) — **H1 promoted to v2**

ArtifactNet v9.4 runs independently on raw 44.1 kHz audio and the native-rate HTDemucs music stem. `MUSIC_FAKE_PROB` uses the maximum of the two median-over-chunk probabilities. File-level: `FILE = max(df_raw, vp*df_voice, mp*max(a_raw,a_stem))` — H1 gates DF voice by `voice_present` to suppress HTDemucs hallucination (`VOICE_FAKE 0.862` avg on music-only). H3 `median→q90` regressed to `0.84482` (`MusicEER 0.298`), H2 `avg` to `0.85393` — not shipped.

| Metric | v0 q90 (2026-08-28) | v2 | v2 H1 | Change H1 vs v2 |
|---|---:|---:|---:|---:|
| **Score** | 0.82582 | 0.87028 | **0.87939** | **+0.91 pts** |
| ADS | 0.80706 | 0.85646 | **0.86658** | +1.01 pts |
| CPS | 0.99466 | 0.99466 | 0.99466 | — |
| FileEER | 0.15274 | 0.13838 | **0.11814** | **-14.6%** |
| VoiceEER | 0.03006 | 0.03006 | 0.03006 | unchanged |
| MusicEER | 0.36851 | 0.22777 | 0.22777 | — |
| Wall 144 (GTX) | 407s | 408s | 409s | +0.2% |
| Wall 1200 extrap GTX | 56.5 min | 56.7 min | 56.8 min | <60 min |
| Wall 1200 extrap L4 22.4G | ~32 min | ~33 min | ~33 min | <60 min |

### v2 completion state

**Algorithm and implementation:** complete. `v2` and `origin/v2` point to `ef9cd18`. H1 was validated on 144 files and the previous run passed 24 tests.

**Release qualification:** incomplete. The existing `deepvoice_v2_with_artifactnet.zip` contains ArtifactNet but excludes the DF-Arena, HTDemucs, and PANNs weights. Those weights exist locally, but a complete weight-inclusive archive still requires clean-environment installation, cold-start inference, checksum verification, L4 timing, and confirmation that ArtifactNet's CC BY-NC 4.0 terms are acceptable for the competition.

---

## Step 6: v3 bottleneck audit

**Status:** Planned — mandatory before implementing or training v3

The measurable bottleneck is music authenticity, not voice detection or score calibration:

| Target | EER | Score weight | Weighted score penalty |
|---|---:|---:|---:|
| Music | **0.22777** | 0.27 | **0.06150** |
| File | 0.11814 | 0.45 | 0.05316 |
| Voice | 0.03006 | 0.18 | 0.00541 |

The architectural bottleneck is representation coverage. ArtifactNet captures local forensic residuals but does not directly model long-range musical organization, vocal–instrument consistency, production effects, or deviation from the bona fide music manifold. Previous q90, averaging, routing, and score-level fusion experiments rearranged the same insufficient evidence and regressed.

The validation set cannot safely support further direct architecture search:

- 144 rows collapse to only **40 split groups** and **71 parents**;
- **8/13 file sources** are label-pure;
- **7/8 music sources** are label-pure;
- row-level confidence intervals and natural-split gains can therefore reward source recognition or correlated transformations.

Before any v3 model work:

1. Reproduce H1 exactly and cache every window-level constituent score.
2. Compute source-balanced metrics and parent/split-group cluster-bootstrap intervals.
3. Measure offline temporal-aggregation and fusion oracle bounds.
4. Run raw-versus-stem leakage counterfactuals.
5. Build a matched audit subset with overlapping real/fake source, domain, and codec support.
6. Decide whether the ceiling is representation, fusion, separation, or data confounding.

Decision gate:

- **High oracle EER:** proceed to representation replacement.
- **Low oracle EER:** solve routing/reliability before adding another backbone.

---

## Step 7: v3 representation replacement

**Status:** Planned — implementation blocked on Step 6

### Candidate A: music-intrinsic multi-axis model

Primary architecture:

```text
Sofia-VAG structural experts:
  Wav2Vec2 + RawNet2 + Fx-Encoder++ + MuQ + MERT
                                  ┐
MusicDET real-manifold flow ──────┼─> reliability-gated segment MoE
                                  │   -> MUSIC_FAKE_PROB
ArtifactNet forensic residuals ───┘

DF-Arena -> VOICE_FAKE_PROB
PANNs + separator diagnostics -> presence
Constrained noisy-OR -> FILE_FAKE_PROB
```

Evidence classes:

1. music-intrinsic vocal, production, and global structure;
2. generator-agnostic deviation from the real-music manifold;
3. local physical synthesis residuals.

### Candidate B: unified all-type detector

Parallel challenger:

```text
XLS-R-300M or BEATs
  + wavelet forensic prompts
  + high-frequency/phase residual branch
  + frame, segment, voice, music, and file heads
```

Train with generator-domain meta-learning, codec/channel consistency, and counterfactual neural-codec re-synthesis. Promote only if it matches the Sofia branch on MusicEER, improves FileEER, preserves VoiceEER, and yields a better runtime/robustness Pareto point.

### Quantitative gates

| Stage | Promotion requirement |
|---|---|
| Released Sofia probe | MusicEER ≤ 0.12 or ≥0.07 absolute reduction |
| Real-only MusicDET | MusicEER ≤ 0.15 and complementary errors |
| Final reliability MoE | MusicEER ≤ 0.08, FileEER ≤ 0.07, VoiceEER ≤ 0.03 |
| Deployment | ≤50 min/1,200 files on L4, ≤20 GiB VRAM, offline cold start |

Projected score is approximately 0.94–0.95 if MusicEER reaches 0.08 and FileEER reaches 0.07 while preserving VoiceEER. This is a target, not a validated result.

### Experiment order

1. Complete the instrumented H1 audit.
2. Benchmark released Sofia and the all-type wavelet-prompt challenger.
3. Train MusicDET only if it adds source-held-out complementary evidence.
4. Train reliability-gated fusion on external generator/source-held-out splits, not the 57 music validation examples.
5. Add segment MIL and constrained noisy-OR composition only after the representation gain is established.
6. Profile, package, and cold-start test the selected architecture.

### Full plan

The complete implementation plan, ablations, promotion gates, kill criteria, runtime strategy, and research sources are recorded in:

`.hermes/plans/2026-08-28_194105-deepvoice-v3-breakthrough.md`
