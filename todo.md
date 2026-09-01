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
|---|---:|---:|---:|
| **Score** | **0.8072** | **0.8384** | **+3.9%** |
| **FileEER** | **0.2363** | **0.1671** | **-29% (1.41×)** |
| MusicEER | 0.2981 | 0.2981 | same |
| VoiceEER | 0.0301 | 0.0301 | same |

**Key insight:** FILE_FAKE = max(DF-Arena on raw audio, DF-Arena on voice stem) — no SONICS, no presence gating. The 4.3B Wav2Vec2 backbone on raw audio is the most reliable single signal.

## Step 5: SONICS HTDemucs-stem head adaptation (FakeMusicCaps)

**Status:** Controlled experiment complete; **do not deploy**

- Dataset: 500 real MusicCaps files and 500 FakeMusicCaps generator outputs; deterministic voice+music mixtures; DeepVoice-equivalent HTDemucs `drums+bass+other` accompaniment, 10 s mono 16 kHz.
- Split repair: all variants of each underlying MusicCaps parent are confined to one split. Final manifests: train 1,600/1,600, validation 100/100, test 100/100 real/fake stem examples; cross-split parent groups = 0; all paths resolve.
- Adaptation: freeze SONICS encoder, train only the 385-parameter classifier head; checkpoint selected at epoch 8 by validation EER (0.415).
- Held-out stem test: original SONICS EER = 0.380; adapted head EER = 0.370. The 0.010 absolute gain is not sufficient evidence for deployment, particularly because fake examples share only 10 underlying MusicCaps parent groups in the test split.
- Artifacts: `runs/sonics_htdemucs_music/20260830_fmc500_v4/prepared_strict/`, `sonics_head/best.pt`, `sonics_head/metrics.json`, `sonics_original_baseline.json`.
- Safety: `model/sonics/pytorch_model.bin` remains the original deployed checkpoint; no inference code or production weights were changed.

---

## Progress tracking

| Step | Started | Completed | Score gain |
|------|---------|-----------|------------|
| 1 | | | |
| 2 | | | |
| 3 | | | |
| 4 | | | |
| 5 | 2026-08-30 | 2026-08-30 | Not deployment-worthy: stem EER -0.010 |
| 6 | 2026-08-30 | 2026-08-30 | Balanced full-pipeline max reproduced exactly; no improvement |

## Step 6: Balanced full-pipeline evaluation protocol

**Status:** Implemented and locked

- Primary benchmark: `~/deepvoice-evalset/manifests/manifest_balanced.csv`, selecting the `split_balanced` column; legacy `split` is not a valid primary result.
- Reproduced full v2 pipeline (`PANNs → HTDemucs → DF-Arena + ArtifactNet → fusion`) with no-scale max fusion. Validation Score/FileEER/MusicEER: `0.73085 / 0.30537 / 0.42405`; locked test: `0.70723 / 0.35530 / 0.38698`.
- These exactly reproduce the prior zero-shot max results. No improvement was obtained.
- Added a bounded ArtifactNet retry normalization for amplified inputs that produce NaN; it only activates after a non-finite model output. The previously failing hard-clip file now gives finite score `0.02555`.
- The val-oriented stem-only choice and the mild harmonic p-norm compromise remain as documented in `eval_results_balanced/final_ablation.txt`; do not claim calibration gains without a locked-test improvement.

## Step 7: SONICS separator-matched adaptation

**Status:** Full fine-tuning rejected; epoch-1 head-only adaptation is the only candidate

- Evaluated the actual V1 composition (`PANNs → HTDemucs → DF-Arena voice + SONICS music`, `max(vp×voice, mp×music)`) on `manifest_balanced.csv` / `split_balanced` for all 299 files.
- Original SONICS: validation/test Score `0.76690 / 0.77865`, FileEER `0.25408 / 0.26908`, MusicEER `0.37602 / 0.26615`.
- Head-only epoch-1 adaptation (385 parameters) improved both: Score `0.78288 / 0.79907`, FileEER `0.23777 / 0.23340`, MusicEER `0.34401 / 0.25000`. The epoch-2 run retained the identical epoch-1 checkpoint, because its adaptation-validation EER worsened.
- Full fine-tuning (16,828,045 parameters) overfits rapidly. At epoch 1: validation/test Score `0.78930 / 0.76121`, and test MusicEER `0.33073`; at epoch 2: validation/test Score `0.79362 / 0.72774`, and test MusicEER `0.38698`. The selected epoch-19 model is worse: validation/test Score `0.75888 / 0.68543`; test MusicEER `0.46771`.
- Promotion candidate: only the head-only **epoch-1** checkpoint, pending independent rerun/package validation. Do not use the epoch-8 or full-finetuned checkpoints.

## Step 8: V1 head-only SONICS logistic file fusion

**Status:** Rejected

- Logistic fusion was fit only on `split_balanced=train` (698 files) using `[logit(vp×DF-Arena_voice), logit(mp×SONICS_head_epoch1_music)]` and `expected_file_fake`; `C ∈ {0.01, 0.1, 1, 10, 100}` was selected by validation Score.
- Selected `C=0.01` gave validation/test Score `0.77554 / 0.74421`, FileEER `0.25408 / 0.35530`, and mixed FileEER `0.24000 / 0.39595`.
- This is worse than the fixed V1 head-only epoch-1 max fusion on validation/test: Score `0.78288 / 0.79907`, FileEER `0.23777 / 0.23340`, mixed FileEER `0.24000 / 0.24162`.
- Stronger L2 logistic regularization (`C=10^-6…10^-2`) and ridge linear regression (`alpha=10^-2…10^4`) were evaluated entirely from the cached 698 train / 150 validation / 149 test component-and-SONICS scores, with no additional audio inference. Every variant produced the same validation/test ranking and metrics as the rejected logistic fusion; regularization cannot repair this feature-space source shift.
- The supervised linear fusion does not survive source-disjoint transfer; retain the original V1 `max(vp×df_voice, mp×sonics_head1)` composition.

## Step 9: Affine SONICS score transform before V1 max fusion

**Status:** Validation-selected transform rejected; post-hoc candidate is not promotable

- Cached-score grid tested `file=max(vp×df_voice, mp×clip(a×sonics_head1+b,0,1))` for `a∈{0.25,0.5,0.75,1,1.25,1.5,2}` and `b∈{-0.4,…,0.3}`; no audio inference was repeated.
- Validation selection chose `a=0.5,b=-0.1`: validation Score/FileEER `0.79442 / 0.21213`, but test regressed to `0.76250 / 0.31467`; reject as overfit.
- Test/mean oracle selects `a=1.5,b=-0.3`, with `0.78498 / 0.80130` Score and `0.23310 / 0.22844` FileEER on validation/test. It is a post-hoc test-selected result and cannot be promoted without an untouched source-disjoint holdout.




## Step 10: External SONICS + ASVspoof mixed-corpus adaptation

**Status:** Data acquisition complete; deterministic corpus construction in progress (2026-08-30).

- Acquired and extracted SONICS at /root/datasets/sonics: 49,074 fake-song audio files. Real music inputs use the existing 4,281-file MusicCaps-derived corpus at /root/datasets/musiccaps_real.
- Acquired ASVspoof 2019 LA at /root/datasets/asvspoof2019_la: 121,461 labelled utterances, 12,483 bona fide and 108,978 spoof. Acquired LJSpeech at /root/datasets/ljspeech: 13,100 bona-fide utterances.
- Added tools/build_synthetic_av_dataset.py on the GPU server. Its 200-example smoke test passed exact 20% music-only, 20% voice-only, 60% mixed allocation, with 30 examples in each mixed authenticity quadrant.
- Full 12,000-example build is active under /root/deepvoice/runs/synthetic_av_20260830 (build.pid and build.log). It produces 10 s, 16 kHz PCM mixtures with exact 9,600/1,200/1,200 train/validation/test allocation. The manifest records component labels and source identifiers; DF-Arena remains frozen.

## Step 10: Joint SONICS + learned fusion checkpoint

**Status:** Rejected on the balanced benchmark

- The separately trained one-epoch joint checkpoint (`SONICS + 4→16→1 fusion MLP`; DF-Arena/PANNs frozen) reported FileEER/MusicEER `0.11914 / 0.08542` on its own 1,200-row validation split and `0.11419 / 0.08333` on its own 1,200-row test split.
- On the required `manifest_balanced.csv` / `split_balanced` evaluation, it regressed: validation FileEER/mixed FileEER/MusicEER `0.25875 / 0.16000 / 0.36002`; test `0.29435 / 0.28189 / 0.36302`.
- This is worse than V1 head-only epoch-1 fixed-max: validation `0.23777 / 0.24000 / 0.34401`; test `0.23340 / 0.24162 / 0.25000`.
- Do not package or promote the joint checkpoint. Its internal 1,200-row split is not representative of the balanced target protocol.

## V3: Source-disjoint synthetic adaptation experiments (2026-09-01)

**Protocol:** The synthetic corpus has 12,000 10-second mixtures with immutable source-parent splits (train/validation/test = 9,600/1,200/1,200). Both voice and music parent cross-split leakage are zero. Each split is 20% music-only, 20% voice-only, and 60% mixed; every mixed real/fake quadrant is exactly balanced. DF-Arena 1B and PANNs are frozen in every V3 adaptation.

### V3-A: Joint SONICS + learned file fusion — rejected

- Trainable: all SONICS parameters plus a 4→16→1 file-fusion head over frozen DF-Arena voice score, PANNs presence scores, and SONICS logit.
- One-epoch internal synthetic result: validation/test FileEER 0.11914/0.11419 and MusicEER 0.08542/0.08333.
- The internal synthetic result is not a deployment metric. The checkpoint must not be promoted based on it; its target-balanced evaluation regressed, as recorded above.

### V3-B: Raw-mixture SONICS, one full epoch — rejected for promotion

- Intervention: `MUSIC_FAKE_PROB` is SONICS on the original 16 kHz mixture, with no HTDemucs music input. The frozen DF-Arena vocal-stem pathway, PANNs presence scores, and V1 `max(vp×voice, mp×music)` file fusion are unchanged.
- Raw synthetic music subset (7,680/960/960, class-balanced): original→adapted validation EER 0.16042→0.05833; test EER 0.16042→0.06667. Epoch-1 train/validation/test BCE: 0.21765/0.15987/0.15653.
- Target balanced pipeline (`split_balanced`): validation Score 0.80084→0.79573, FileEER 0.20745→0.22843, MusicEER 0.32800→0.31199; test Score 0.73210→0.74506, FileEER 0.30972→0.31467, MusicEER 0.37083→0.31458.
- Conclusion: direct raw SONICS improves held-out MusicEER and test Score relative to its matched raw baseline, but worsens FileEER and validation Score. It is an experimental archive only; do not submit without a validation-consistent file-fusion intervention.
- Reproducibility: `tools/finetune_sonics_raw_mixture.py`, `tools/evaluate_v1_sonics_raw_balanced.py`, `runs/sonics_raw_mixture_full1_20260901/`, and `runs/v1_sonics_raw_mixture_full1_balanced_20260901/`.
