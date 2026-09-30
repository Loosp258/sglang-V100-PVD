#!/usr/bin/env bash
set -euo pipefail
source /users/Yizhzhu/.sglang-v100-pvd-env.sh
checkout="$SGLANG_PVD_ROOT/validation/pvd-joint-draft-q-20260930"
evidence="$SGLANG_PVD_ROOT/validation/draft-q-decode-20260930"
cd "$checkout"
trap 'printf "%s\n" "$?" > "$evidence/run.exit"' EXIT
python benchmark/pvd_draft_q_decode_train.py run \
  --dataset "$evidence/questions.json" \
  --checkpoint "$SGLANG_PVD_ROOT/validation/draft-q-multitask-ceqkl-20260929/trained.pt" \
  --target-model "$SGLANG_PVD_ROOT/models/Qwen2.5-7B-Instruct" \
  --draft-model "$SGLANG_PVD_ROOT/models/Qwen2.5-0.5B-Instruct" \
  --output-dir "$evidence" --steps 1200
