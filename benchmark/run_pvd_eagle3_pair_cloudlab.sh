#!/usr/bin/env bash
set -euo pipefail
source /users/Yizhzhu/.sglang-v100-pvd-env.sh
export CUDA_VISIBLE_DEVICES=1
checkout="$SGLANG_PVD_ROOT/validation/pvd-joint-draft-q-20260930"
evidence="$SGLANG_PVD_ROOT/validation/eagle3-pair-20261001"
mkdir -p "$evidence"
trap 'printf "%s\n" "$?" > "$evidence/run.exit"' EXIT
cd "$checkout"
python benchmark/pvd_eagle3_pair_probe.py \
  --target-model "$SGLANG_PVD_ROOT/models/Qwen2.5-7B-Instruct" \
  --draft-model "$SGLANG_PVD_ROOT/models/Qwen2.5-0.5B-Instruct" \
  --old-checkpoint "$SGLANG_PVD_ROOT/validation/draft-q-decode-20260930/joint.pt" \
  --questions "$checkout/benchmark/results/pvd_draft_q_decode_cloudlab_20260930/questions.json" \
  --eagle-source "$evidence/EAGLE" --output-dir "$evidence" --repeats 3
