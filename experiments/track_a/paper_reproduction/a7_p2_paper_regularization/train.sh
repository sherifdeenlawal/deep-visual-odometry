#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/a7_p2_common.sh"

if [[ -e "${A7_P2_CHECKPOINT}" ]]; then
  echo "Refusing to overwrite existing A7-P2 checkpoint: ${A7_P2_CHECKPOINT}" >&2
  echo "A7-P2 must start from a fresh initialization; move the existing directory first." >&2
  exit 2
fi

mkdir -p "${A7_P2_CHECKPOINT_DIR}"

python scripts/train_deepdct_vo.py \
  --data-root "${A7_P2_DATA_ROOT}" \
  --train-sequences 00 01 02 03 04 05 06 07 08 \
  --no-validation \
  --epochs 30 \
  --batch-size 1 \
  --num-workers 0 \
  --learning-rate 1e-4 \
  --weight-decay 0.0 \
  --head-conv-l2 "${A7_P2_HEAD_CONV_L2}" \
  --head-dense-l2 "${A7_P2_HEAD_DENSE_L2}" \
  --pose-loss-type mae \
  --rotation-normalization-scale 0.175 \
  --translation-decoder dense \
  --use-ground-truth-rotation \
  --use-semantic-cues \
  --pretrained-semantic \
  --freeze-semantic \
  --semantic-provider segformer \
  --semantic-model-name "${A7_P2_SEMANTIC_MODEL}" \
  --semantic-feed-size 512 512 \
  --semantic-map-mode foreground_probability \
  --use-depth-cues \
  --freeze-depth \
  --depth-provider monodepth2 \
  --depth-checkpoint-dir "${A7_P2_DEPTH_WEIGHTS}" \
  --depth-model-name mono_640x192 \
  --depth-output-mode normalized_depth \
  --depth-normalization-meters 80.0 \
  --checkpoint-dir "${A7_P2_CHECKPOINT_DIR}" \
  --experiment-name track_a_a7_p2_paper_regularization \
  --source-experiment track_a_a7_p1_epoch30_control
