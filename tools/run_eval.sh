#!/bin/bash
# Run all 3 model variants on validation set
cd ~/deepvoice && .venv311/bin/python3 tools/evaluate_validation.py \
  --evalset ~/deepvoice-evalset --project . \
  --outdir eval_results --venv-python .venv311/bin/python3 \
  --variants baseline v1_original v1_q90
