#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/a7_p2_common.sh"

test -d "${A7_P2_DATA_ROOT}/sequences/00/image_2"
test -f "${A7_P2_DATA_ROOT}/out_csv/00_dct.txt"
test -d "${A7_P2_DEPTH_WEIGHTS}"
test -f scripts/train_deepdct_vo.py
test -f scripts/evaluate_deepdct_vo.py
test -f scripts/audit_a7_rotation_attribution.py

python -m py_compile \
  deepdct/training/train_one_epoch.py \
  deepdct/training/validate_one_epoch.py \
  scripts/train_deepdct_vo.py \
  scripts/evaluate_deepdct_vo.py

python scripts/train_deepdct_vo.py --help | grep -- '--head-conv-l2'
python scripts/train_deepdct_vo.py --help | grep -- '--head-dense-l2'

pytest -q \
  tests/models/test_deepdct_forward.py \
  tests/integration/test_single_batch_training_step.py

echo "A7-P2 preflight: PASS"
