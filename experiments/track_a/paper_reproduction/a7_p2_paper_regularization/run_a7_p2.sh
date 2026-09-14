#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${script_dir}/preflight.sh"
"${script_dir}/train.sh"
"${script_dir}/evaluate.sh"
"${script_dir}/evaluate_gt_rotation.sh"
"${script_dir}/audit_rotation_attribution.sh"
