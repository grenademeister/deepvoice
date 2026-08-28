# DeepVoice — AGENTS.md

Project: **DACON 236749 AI Deepvoice Detection Challenge**  
Deadline: **September 29, 2026**  
Current Score (v2 H1, 144 validation q90, 2026-08-28): **0.87939** — v2 **0.87028** → H1 `vp*df_voice`  
Primary bottleneck (v0): **MusicEER = 0.3685** → v2 **0.2277** (-38%) — VoiceEER 0.030 stable — H1 `FileEER 0.138→0.118` via HTDemucs gate

---

## 1. Repository layout

```
deepvoice/
├── script.py                    # Inference pipeline (entry point) — v2: DF-Arena + ArtifactNet
├── requirements.txt             # Competition submission deps (+ onnxruntime-gpu 1.22.0)
├── model/
│   ├── df_arena_1b/             # Voice fake detector (Wav2Vec2-XLS-R 1B)
│   │   └── pytorch_model.bin    # 4.3 GB — excluded from git
│   ├── artifactnet/             # Music fake detector (ArtifactNet v9.4 ONNX, 17M)
│   │   ├── artifactnet_v94_full.onnx
│   │   └── artifactnet_v94_full.onnx.data
│   ├── artifactnet_infer.py     # ArtifactNet wrapper (44.1kHz, 4s median, ORT CUDA/CPU)
│   ├── v2_fusion.py             # FILE_FAKE = max(df_raw, df_voice, mp*max(a_raw,a_stem))
│   ├── temporal_aggregation.py  # DF-Arena q90 aggregation (quantile 0.90)
│   ├── htdemucs/                # Source separation (Demucs)
│   │   ├── htdemucs.yaml
│   │   └── 955717e8-8726e21a.th  # 81 MB
│   └── panns/                   # Voice/music presence detector
│       ├── Cnn14_mAP=0.431.pth   # 313 MB
│       ├── class_labels_indices.csv
│       └── component_labels.json
├── tests/
│   ├── test_artifactnet_infer.py
│   ├── test_v2_fusion.py
│   └── test_temporal_aggregation.py
└── data/
    └── test/                    # 3 dummy WAVs (format validation only)
```

**GPU server:** `ssh -p 30266 root@147.46.92.88` — GTX 1080 8GB, CUDA 12.4  
**Server workdir:** `~/deepvoice/`, venv: `.venv311/bin/python3`  
**Evalset:** `~/deepvoice-evalset/` (1578 files, 22 sources, 4.78h)

---

## 2. Inference pipeline (`script.py`) — v2 H1

```
Raw audio → [PANNs] → presence (voice, music)
          → [HTDemucs] → voice stem (16k) + music stem (native 44.1k preserved)
          → [DF-Arena 1B q90] on voice stem → VOICE_FAKE_PROB (gated by vp for file)
          → [DF-Arena 1B q90] on raw 16k → FILE evidence
          → [ArtifactNet v9.4 ORT 44.1kHz, 4s median] on raw + music stem → MUSIC_FAKE = max(a_raw, a_stem)
          → Fusion: FILE_FAKE = max(df_raw, vp×df_voice, mp × music_fake)  // v2_fusion.py H1
```

**Validated 2026-08-28 on GTX 1080 8GB (144 validation, mean 11.58s):** v0 407s / 5.2G VRAM → v2 408s / 6.25G → H1 409s / 6.25G. DF-Arena q90 + AN median + vp gate. `FileEER 0.138→0.118` via suppressing `VOICE_FAKE 0.862` hallucination on music-only. H3 `median→q90` regressed to `0.84482`, H2 `avg` to `0.85393` — not shipped.

---

## 3. Current performance (144 validation, q90, 2026-08-28 GTX 1080) — H1

| Metric | v0 q90 | v2 ArtifactNet | v2 H1 `vp*df_voice` | Δ H1 vs v2 |
|---|---|---|---|---|
| **Score** | **0.82582** | **0.87028** | **0.87939** | **+0.91 pts** |
| ADS | 0.80706 | 0.85646 | **0.86658** | +1.01 pts |
| CPS | 0.99466 | 0.99466 | 0.99466 | — |
| FileEER | 0.15274 | 0.13838 | **0.11814** | **-14.6%** |
| VoiceEER | **0.03006** | **0.03006** | **0.03006** | — |
| MusicEER | 0.36851 | **0.22777** | **0.22777** | — |
| Wall (144) | 407s | 408s | 409s | +0.2% |
| VRAM peak | 5.20G | 6.25G | 6.25G | — |
| Extrap 1200 (GTX) | 56.5 min | 56.7 min | 56.8 min | <60 min |
| Extrap 1200 (L4 22.4G) | ~32 min | ~33 min | ~33 min | <60 min |

**H1 `file=max(df_raw, vp*df_voice, mp*music)` validated 409s/144** — suppresses `VOICE_FAKE 0.862` on music-only (HTDemucs hallucination). H3 `median→q90` regressed to `Score 0.84482 MusicEER 0.298`, H1+H2 `avg` to `0.85393` — not shipped. **v2 validated on same split** (`split=validation`, n_file=144, n_voice=68, n_music=57).

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