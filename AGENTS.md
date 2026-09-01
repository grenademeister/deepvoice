# DeepVoice — AGENTS.md

Project: **DACON 236749 AI Deepvoice Detection Challenge**  
Deadline: **September 29, 2026**  
Primary zero-shot benchmark: `manifest_balanced.csv`, `split_balanced=validation/test`
Locked full-pipeline max baseline: validation Score **0.73085**, test Score **0.70723**
Primary bottleneck: **mixed MusicEER = 0.4241 / 0.3870** (validation/test)

---

## 1. Repository layout

```
deepvoice/
├── script.py                    # Inference pipeline (entry point)
├── requirements.txt             # Competition submission deps
├── model/
│   ├── df_arena_1b/             # Voice fake detector (Wav2Vec2-XLS-R 1B)
│   │   └── pytorch_model.bin    # 4.3 GB — excluded from git
│   ├── sonics/                  # Music fake detector (SpecTTTra)
│   │   ├── config.json
│   │   └── pytorch_model.bin    # 65 MB
│   ├── sonics_infer.py          # Self-contained SONICS inference
│   ├── temporal_aggregation.py  # Window score aggregation (max / q90)
│   ├── htdemucs/                # Source separation (Demucs)
│   │   ├── htdemucs.yaml
│   │   └── 955717e8-8726e21a.th  # 81 MB
│   └── panns/                   # Voice/music presence detector
│       ├── Cnn14_mAP=0.431.pth   # 313 MB
│       ├── class_labels_indices.csv
│       └── component_labels.json
├── tools/
│   ├── evaluate_validation.py   # Run 3 variants + compute metrics
│   ├── compute_metrics.py       # Standalone metric computation
│   ├── train_adapter.py         # Fusion calibrator / voice adapter training
│   └── diagnose_music.py        # Error analysis by domain/source
├── .venv311/                    # Python 3.11 venv with all deps
└── data/
    └── test/                    # 3 dummy WAVs (format validation only)
```

**GPU server:** `ssh -p 30266 root@147.46.92.88` — GTX 1080 8GB, CUDA 12.4  
**Server workdir:** `~/deepvoice/`, venv: `.venv311/bin/python3`  
**Evalset:** `~/deepvoice-evalset/` contains `manifests/manifest_balanced.csv`; use `split_balanced`, not legacy `split`, for all benchmark results. The group-disjoint benchmark partitions are train 698, validation 150, test 149; validation/test are 66.7% mixed voice+music with balanced authenticity quadrants.

---

## 2. Inference pipeline (`script.py`)

```
Raw audio → [PANNs] → presence (voice, music)
          → [HTDemucs] → source separation → voice stem, music stem
          → [DF-Arena 1B] on voice stem → VOICE_FAKE_PROB
          → [SONICS SpecTTTra] on music stem → MUSIC_FAKE_PROB
          → Fusion: FILE_FAKE = max(presence_voice × voice_fake, 
                                     presence_music × music_fake)
```

**All 3 models are loaded simultaneously on GPU.** DF-Arena (4.3B) alone fills most of the 8GB VRAM.

---

## 3. Locked zero-shot benchmark (balanced protocol)

**Mandatory protocol:** Run the full deployed pipeline on `manifest_balanced.csv`, selecting `split_balanced=validation` only for hypotheses and evaluating `split_balanced=test` once after locking a choice. Never report the legacy `split=validation` result as a primary benchmark.

| Metric | Validation (150) | Test (149) |
|---|---:|---:|
| **Score** | **0.73085** | **0.70723** |
| ADS | 0.70250 | 0.67722 |
| CPS | 0.98592 | 0.97726 |
| FileEER | 0.30537 | 0.35530 |
| FileEER, mixed only | 0.32667 | 0.40270 |
| VoiceEER | 0.08798 | 0.14516 |
| MusicEER | 0.42405 | 0.38698 |

These figures are the reproducibility-verified no-scale `max(artifact_raw, artifact_stem)` baseline. The selected val-oriented submission fusion is stem-only (`music_fake = artifact_stem`); see `eval_results_balanced/final_ablation.txt` for the locked val/test trade-off.

---

## 4. Critical empirical findings (from `diagnose_music.py`)

### Finding A: HTDemucs separation is leaky
```
Speech-only files: mean MUSIC_FAKE_PROB = 0.856  (should be ~0)
Music-only files: mean VOICE_FAKE_PROB = 0.893  (should be ~0)
```
Both detectors fire on leaked content from the other domain. The only thing preventing total collapse is PANNs presence gating (`vp × vf`, `mp × mf`).

### Finding B: SONICS OOD on real music
```
GTZAN real music:   mean MUSIC_FAKE = 0.688  (10/11 → FP)
FakeMusicCaps TTM:  mean MUSIC_FAKE = 0.773  (lower confidence)
SONICS native:      mean MUSIC_FAKE = 0.907  (good)
```
SONICS has no concept of "real instrumental music" — assigns high fake probabilities to any music it wasn't trained on.

### Finding C: Presence gating failures
```
Envsdd fake environmental: MUSIC_FAKE ~0.88, MUSIC_PRESENT ~0.02
  → FILE_FAKE = 0.02 × 0.88 = 0.02  (misses the true=1 file)
```
Low presence probabilities suppress correct fake detections when the fake is in an unexpected domain.

### Finding D: q90 hurts more than helps
```
max aggregation:  MusicEER = 0.298, Score = 0.807
q90 aggregation:  MusicEER = 0.404, Score = 0.770
```
Max preserves the detector's highest-confidence signal. q90 discards the top 10% which kills recall on borderline fakes.

---

## 5. Improvement strategy (ordered by ROI)

```
Priority 1: Domain-specific routing
  - Use PANNs to skip HTDemucs for single-domain files
  - Speech-only: DF-Arena on raw audio, MUSIC_FAKE = 0
  - Music-only: SONICS on raw audio, VOICE_FAKE = 0
  - Mixed only: full separation
  - Benefit: eliminates cross-contamination, saves ~70% runtime

Priority 2: Trained fusion calibrator
  - Replace naive max(vp×vf, mp×mf) with 4→3 MLP
  - Trained on pipeline outputs from training split (1075 files)
  - Learns cross-domain interaction patterns

Priority 3: SONICS fine-tuning on evalset music
  - Fine-tune SpecTTTra head on music stems (525 music files)
  - Addresses OOD gap on GTZAN, FakeMusicCaps, instrumental
  - Need to first separate music stems from training split

Priority 4: Presence calibration
  - Train a calibrator on PANNs presence scores
  - Fixes edge cases where low presence suppresses correct detection
```

---

## 6. Running inference on GPU server

```bash
# Single file inference (for testing)
cd ~/deepvoice
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv311/bin/python3 script.py \
  --test-dir <dir> \
  --sample-submission <csv> \
  --output <csv> \
  --device cuda

# Evaluate on validation set (~5 min)
.venv311/bin/python3 tools/compute_metrics.py \
  --predictions eval_results/predictions_v1_q90.csv \
  --manifest ~/deepvoice-evalset/manifests/manifest.csv

# Fusion calibrator training (~30 min for feature extraction + 10s training)
.venv311/bin/python3 tools/train_adapter.py \
  --mode fusion \
  --evalset ~/deepvoice-evalset \
  --project . \
  --venv-python .venv311/bin/python3 \
  --device cuda
```

---

## 7. Competition constraints

| Constraint | Value |
|---|---|
| Max zip size | 10 GB |
| Max extracted | 32 GB |
| Install time | 10 min |
| Inference time | 60 min (1200 files) |
| GPU | NVIDIA L4, 22.4 GiB |
| CUDA | 12.8 |
| Python | 3.11.15 |
| Internet (inference) | Disabled |
| Package preinstalled | torch 2.7.1, torchaudio 2.7.1, demucs, panns-inference |
| Submissions/day | 3 |

**Submission structure:**
```
submit.zip
├── model/         # weights + inference module
├── script.py      # inference script
└── requirements.txt
```

---

## 8. Important gotchas

- `model/temporal_aggregation.py` is NOT tracked in git (created after branching). Always re-create via `git show HEAD:model/temporal_aggregation.py` or the local copy.
- GPU OOM: GTX 1080 can barely hold all 3 models. Use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Kill all GPU processes between runs.
- SONICS SpecTTTra expects 16kHz mono audio, 5s windows (80k samples). Uses a trailing 75% crop `(n - max_len) / 4 * 3`.
- DF-Arena expects 16kHz, 64,600-sample segments. Uses `aggregate_temporal_scores()` with q90=0.90 by default.
- PanNS expects 32kHz audio. `script.py` resamples before inference.
- The evalset manifest uses **comma** delimiter (not pipe despite historical notes saying otherwise).
- Empty fields in manifest (e.g., `expected_music_fake` for speech-only files) must be parsed as 0.0, not empty string.

## V3 experiment discipline

- Treat V3 SONICS/fusion adaptations as experimental until evaluated through the full `manifest_balanced.csv` / `split_balanced` pipeline.
- The synthetic adaptation corpus is source-parent-disjoint and balanced: 12,000 mixtures, 20% music-only, 20% voice-only, 60% mixed with equal real/fake quadrants in every split.
- V3 freezes DF-Arena 1B and PANNs. Do not promote a checkpoint from synthetic component EER alone; the deployed FileEER and total score are the release gates.
- Current raw-mixture SONICS one-epoch result is rejected for promotion: target validation Score declines 0.80084→0.79573 despite a held-out test Score increase 0.73210→0.74506.
