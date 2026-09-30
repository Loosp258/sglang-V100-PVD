#!/usr/bin/env bash
set -euo pipefail
source /users/Yizhzhu/.sglang-v100-pvd-env.sh
checkout="$SGLANG_PVD_ROOT/validation/pvd-joint-draft-q-20260930"
evidence="$SGLANG_PVD_ROOT/validation/draft-q-onpolicy-20260930"
mkdir -p "$evidence"
cd "$checkout"
trap 'printf "%s\n" "$?" > "$evidence/collect.exit"' EXIT
questions="$checkout/benchmark/results/pvd_draft_q_decode_cloudlab_20260930/questions.json"
python benchmark/pvd_draft_q_onpolicy_collect.py prepare --questions "$questions" \
  --dataset "$evidence/dataset.json"
python benchmark/pvd_joint_draft_q_online_probe.py --arm joint --warmup-only \
  --tokenizer /users/Yizhzhu/pvd-models/Qwen2.5-7B-Instruct \
  --source-root . --output "$evidence/warmup.jsonl"
python benchmark/pvd_draft_q_onpolicy_collect.py collect --questions "$questions" \
  --dataset "$evidence/dataset.json" --output "$evidence/trajectories.jsonl"
