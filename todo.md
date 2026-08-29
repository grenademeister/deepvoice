# DeepVoice Improvement Plan

**Current best validated score:** 0.87939 (`v2` H1, local 144-file validation)

**Current best official submission score:** 0.6984253545 (`DV_v2_q90.zip`, submitted 2026-08-28 23:15:44)

The local and official scores are not interchangeable. Actual submission history:

| Submission | Submitted | Official score |
|---|---|---:|
| `DV_v2_q90.zip` | 2026-08-28 23:15:44 | **0.6984253545** |
| `DV_v2_maxpool.zip` | 2026-08-28 22:54:35 | 0.6957896402 |
| `DV_v2_q90_rawfile.zip` | 2026-08-28 23:29:07 | 0.6939253545 |
| `DV_v1_betterpooling.zip` | 2026-08-27 19:47:27 | 0.6922539259 |
| `DV_v1_baseline.zip` | 2026-08-27 17:54:26 | 0.6900110688 |

Official evidence reverses the local aggregation conclusion: q90 improves over max pooling in both submitted generations (`+0.0022428571` for v1 better-pooling over baseline; `+0.0026357143` for v2 q90 over maxpool). Treat q90 as the deployment-favored temporal aggregator unless a stronger official or source-held-out test overturns it.

The user confirms that the artifact actually submitted as `DV_v2_q90.zip` used only the HTDemucs music stem for music-fake inference. This supersedes the earlier architectural attribution based on the current local file bearing that name. The local `DV_v2_q90.zip` (SHA-256 `5e61c09e...`) is a dual-view rebuild: it explicitly computes ArtifactNet on raw audio and the music stem and sets `MUSIC_FAKE=max(a_raw,a_stem)`. Because its modification time precedes the recorded submission time but no platform-side archive hash is available here, there is an unresolved artifact-identity/provenance mismatch; do not infer the uploaded architecture from the current local ZIP.

The official `DV_v2_q90_rawfile.zip` regression should be interpreted against the user's confirmed stem-only winner: replacing the separated/component-based file pathway with direct raw-file inference reduced official score by exactly `0.0045`. If only FileEER changed, this corresponds to `+0.01` absolute FileEER. It does not establish that raw ArtifactNet music scoring is superior on the hidden set; that conclusion is only from the source-confounded 57-file local music-only audit.

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

### v2 full archive generated

**Status:** Package generated and structurally verified locally.

- Artifact: `deepvoice_v2_full.zip`
- Archive size: 4,584,258,514 bytes
- Extracted member size: 5,020,634,529 bytes
- SHA-256: `cd6f01c0b3531569ad9fc8b1cd40a485babfe44a60f6604b1cf45e7a99f6bee7`
- Top-level roots: exactly `model/`, `script.py`, and `requirements.txt`
- Required weights included: DF-Arena 1B, HTDemucs, PANNs Cnn14, and ArtifactNet v9.4 ONNX plus external data
- ArtifactNet checksum manifest: passed
- ZIP CRC verification: passed
- Python syntax compilation of the packaged script and local model modules: passed
- Full `pytest` execution: blocked in the current system environment because `pytest` is not installed

Remaining release gates are clean-environment dependency installation, cold-start inference, L4 timing, and ArtifactNet license review.

### ArtifactNet q90 variant

**Status:** Generated as a replacement archive on the requested q90 aggregation.

- Artifact: `DV_v2_q90.zip`
- ArtifactNet chunk aggregation changed from median to `np.quantile(probabilities, 0.90)`.
- Archive size: 4,584,258,522 bytes
- Extracted member size: 5,020,634,534 bytes
- SHA-256: `5e61c09ee4ac66fc7fffde8f0ca31528a3fa879fa48e40df3e1af10249d1262b`
- ZIP structure, required weights, CRC verification, ArtifactNet checksums, and Python syntax compilation: passed
- This is an unvalidated experimental aggregation variant; the previously recorded v2 H1 score used ArtifactNet median aggregation.

### Direct raw-audio file-score variant

**Status:** Generated as an unvalidated comparison archive.

- Artifact: `DV_v2_q90_rawfile.zip`
- `FILE_FAKE_PROB` is directly assigned from q90 DF-Arena on the pre-separation raw 16 kHz audio (`raw_fake`).
- Voice and music probabilities remain computed by the existing separated-stem pipeline; ArtifactNet remains q90 over chunks and max across raw/stem views.
- Archive size: 4,584,258,556 bytes
- Extracted member size: 5,020,634,604 bytes
- SHA-256: `9ebaaa230dd88ce6c8001bb78227a75a72bf045da4142565626fe6cf3749d681`
- Archive structure, required weights, CRC verification, ArtifactNet checksums, and Python syntax compilation: passed
- This variant corresponds to the earlier raw-audio DF-Arena file-evidence experiment; it has not been re-evaluated on the validation split.

#### Raw-file q90 validation result

**Status:** Evaluated on the fixed validation split; not promoted.

- Evaluation environment: GTX 1080 remote server, CUDA inference
- Validation size: 144 files (`n_voice=68`, `n_music=57`)
- Components phase: approximately 393 seconds
- Score: **0.844821**
- ADS: **0.828172**
- CPS: **0.994662**
- FileEER: **0.152742**
- VoiceEER: **0.030062**
- MusicEER: **0.298148**
- Voice AUC: **0.990132**
- Music AUC: **0.999193**
- Compared with validated v2 H1 (`Score 0.87939`, `FileEER 0.11814`): score **-0.034569**, FileEER **+0.034602**.
- Conclusion: directly assigning raw pre-separation DF-Arena q90 to `FILE_FAKE_PROB` regresses substantially versus H1 fusion, so this variant is not promoted.
- Prediction artifact: `analysis_outputs/rawfile_q90/predictions.csv`
- Metric artifact: `analysis_outputs/rawfile_q90/metrics.json`

### Mixed-audio coverage and sample inference

The evalset contains 56 files with both `VOICE_PRESENT=1` and `MUSIC_PRESENT=1`: 40 `derived:voice_music_mix` files and 16 DeepVoice stress-suite files. All 56 belong to stress splits; none are in the 144-file validation split. The published H1 score therefore does **not** measure mixed voice+music performance. v2 is zero-shot and does not train on these files; they are used only for stress evaluation.

A fresh v2 H1 inference run on redistributable FakeMusicCaps sample `fakemusiccaps_MusicGen_medium_44bc8bbaa7b1cca8` produced:

| Output | Probability |
|---|---:|
| FILE_FAKE | 0.9411401391 |
| VOICE_FAKE | 0.8720627785 |
| MUSIC_FAKE | 0.9999839067 |
| VOICE_PRESENT | 0.0240431540 |
| MUSIC_PRESENT | 0.9254109859 |

The file and music decisions are correct for this MusicGen sample. The high `VOICE_FAKE` value occurs on a near-absent vocal stem and demonstrates HTDemucs/DF-Arena leakage. H1 suppresses it to `VOICE_PRESENT × VOICE_FAKE = 0.02096714`. Music contributes `MUSIC_PRESENT × MUSIC_FAKE = 0.92539609`; the final `FILE_FAKE = 0.94114014` is driven by DF-Arena on raw audio.

### H1 hard-error audit

A fresh complete H1 run reproduced the promoted metrics exactly and persisted `analysis_outputs/h1/h1_predictions.csv`, `h1_metrics.json`, and `hard_samples_scores.json`. The principal error modes are:

1. **Unseen/newer music generators are ranked below real-music false positives.** Stable Audio Open sample `fakemusiccaps_stable_audio_open_77a9f35afd4432e8` has `MUSIC_FAKE=0.57217`; raw ArtifactNet gives 0.57217 and the HTDemucs music stem only 0.07653. Udio sample `sonics_03_89a7ac826d2cb25e` has `MUSIC_FAKE=0.76725`; the music-stem score collapses to `1.25e-6`.
2. **Bona fide production/transform artifacts look synthetic.** DeepVoice stress-suite real music has mean `MUSIC_FAKE=0.98914` after transformations, while GTZAN real music has mean 0.53214 and maximum 0.96873. These high real scores force the validation Music-EER operating threshold to 0.99664 and create `MusicEER=0.22778`.
3. **Some unseen speech generators evade DF-Arena.** WaveFake WF7 sample `wavefake_6eefebe983b32ef5` has `df_raw=0.30989`, `df_voice=0.40165`, and final `FILE_FAKE=0.32795` despite being fake. Presence is correct (`VOICE_PRESENT=0.81650`), so this is a detector-generalization failure rather than a presence-gating failure.
4. **Stem evidence can be anti-informative.** For both hard music false negatives, ArtifactNet's stem score is much lower than its raw score. HTDemucs can erase or transform the forensic residuals that ArtifactNet needs; `max(raw, stem)` prevents a worse result but does not solve the missing representation.

#### Music-only raw-versus-stem counterfactual

The separator-shift hypothesis was tested on all 57 validation files with music present and voice absent (30 fake, 27 real), holding ArtifactNet weights and median aggregation fixed:

| ArtifactNet input | ROC-AUC | EER |
|---|---:|---:|
| Original audio only | **0.78272** | **0.22778** |
| HTDemucs music stem only | 0.63889 | 0.33333 |
| `max(original, stem)` | 0.78025 | **0.22778** |

HTDemucs induces a large distribution shift. The stem score decreased for all 30 fake files, with mean `stem - raw = -0.63815` and median `-0.81189`; at threshold 0.5, original audio correctly detects all 30 fake files, while stem-only detects only 9. The shift also lowers many real scores (26/27), so it sometimes removes real-audio false positives: at threshold 0.5, stem-only correctly rejects 20/27 real files versus 11/27 for raw. It is therefore not a simple global calibration shift; separation destroys class-discriminative evidence and compresses both classes toward low scores.

Conclusion: the user's hypothesis is confirmed as a major failure of the stem path, but not as the sole cause of H1's MusicEER. Original-only ArtifactNet already gives the same `0.22778` EER as current max fusion because raw real/fake score distributions overlap. `max(raw, stem)` protects against most stem damage and does not improve EER; it slightly lowers ROC-AUC because one real telephone-transformed sample has a higher stem score. Persisted artifacts: `analysis_outputs/h1/music_only_raw_stem.csv` and `music_only_raw_stem_summary.json`.

The severe real-music false positives are non-redistributable and are documented numerically rather than exported. The attached qualitative examples use redistributable hard false negatives.

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
