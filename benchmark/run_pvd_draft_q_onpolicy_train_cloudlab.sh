#!/usr/bin/env bash
set -euo pipefail
source /users/Yizhzhu/.sglang-v100-pvd-env.sh
checkout="$SGLANG_PVD_ROOT/validation/pvd-joint-draft-q-20260930"
evidence="$SGLANG_PVD_ROOT/validation/draft-q-onpolicy-train-20260930"
mkdir -p "$evidence"
cd "$checkout"
trap 'printf "%s\n" "$?" > "$evidence/run.exit"' EXIT
python benchmark/pvd_draft_q_onpolicy_train.py \
  --questions "$checkout/benchmark/results/pvd_draft_q_decode_cloudlab_20260930/questions.json" \
  --trajectories "$evidence/trajectories.jsonl" \
  --calibration-captures "$SGLANG_PVD_ROOT/validation/draft-q-decode-20260930/captures.pt" \
  --checkpoint "$SGLANG_PVD_ROOT/validation/draft-q-decode-20260930/joint.pt" \
  --target-model "$SGLANG_PVD_ROOT/models/Qwen2.5-7B-Instruct" \
  --draft-model "$SGLANG_PVD_ROOT/models/Qwen2.5-0.5B-Instruct" \
  --output-dir "$evidence" --steps 4000
