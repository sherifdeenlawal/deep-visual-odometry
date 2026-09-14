#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/a7_p2_common.sh"

for sequence in 09 10; do
  python scripts/audit_a7_rotation_attribution.py \
    --predictions "${A7_P2_OUTPUT_ROOT}/evaluation_sequence_${sequence}/frame_predictions.csv" \
    --output-dir "${A7_P2_OUTPUT_ROOT}/rotation_attribution_sequence_${sequence}" \
    --sequence "${sequence}" \
    --euler-order xyz \
    --translation-scale 1.0
done
