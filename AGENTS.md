# DeepVoice V5

## Goal

Train and validate the V5 fusion model for DACON 236749. The official zero-shot
benchmark is `manifest_balanced.csv` with `split_balanced=validation/test`.
Never report the legacy `split` column as the primary benchmark.

## V5 pipeline

```text
audio -> HTDemucs voice/music stems
      -> DF-Arena 500M voice probability
      -> SONICS raw/stem embeddings
      -> PANNs presence + ArtifactNet scores
      -> V5 three-output fusion
```

HTDemucs stems, DF-Arena probability, and SONICS embeddings are the only cached
features. PANNs and ArtifactNet outputs are transient.

## Commands

```bash
.venv/bin/python3 tools/prepare_v5.py prepare \
  --manifest /path/to/manifest.csv --cache-dir /path/to/cache --device cuda

.venv/bin/python3 tools/train_v5.py \
  --manifest /path/to/manifest.csv --cache-dir /path/to/cache --run-dir /path/to/run --device cuda

.venv/bin/python3 tools/train_v5.py \
  --manifest /path/to/manifest.csv --cache-dir /path/to/cache --run-dir /path/to/run --device cuda --resume
```

Use `.venv/bin/python3` on the GPU server. The older `.venv311` environment is
incomplete. Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for GPU runs.

## Constraints

- GPU server: `ssh -p 30266 root@147.46.92.88`, workdir `~/deepvoice`
- Competition inference: Python 3.11, NVIDIA L4, no internet
- Limits: 10 GB archive, 32 GB extracted, 10 minute install, 60 minute inference
- Submission contains only `model/`, `script.py`, and `requirements.txt`
- SONICS: 16 kHz mono, 5-second windows
- DF-Arena: 16 kHz
- PANNs: 32 kHz
- Manifest CSV fields that are empty represent `0.0`

## Verification

Run `python -m pytest -q` after changes. Keep caches and checkpoints atomic and
resumable; never infer cache validity from file existence alone.
