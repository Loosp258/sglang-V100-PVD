#!/usr/bin/env bash
# Run on V/Gateway node1 after all four services pass health checks.
set -euo pipefail
arm="${1:?full, target or joint required}"
[[ "$arm" =~ ^(full|target|joint)$ ]] || exit 2
source /users/Yizhzhu/.sglang-v100-pvd-env.sh
checkout="$SGLANG_PVD_ROOT/validation/pvd-joint-draft-q-20260930"
evidence="${PVD_QUALITY_EVIDENCE_DIR:-$SGLANG_PVD_ROOT/validation/output-quality-20260930}"
resume_args=()
if [[ "${2:-}" == resume ]]; then
  resume_args=(--resume)
  cp "$evidence/warmup-$arm-final.jsonl" "$evidence/warmup-$arm-before-resume.jsonl"
fi
cd "$checkout"
trap 'printf "%s\n" "$?" > "$evidence/$arm.exit"' EXIT
# The previous transport smoke helper labels a non-joint warmup as target;
# actual full/target mode is verified from the serving logs, not that label.
warmup_arm=target
[[ "$arm" == joint ]] && warmup_arm=joint
python benchmark/pvd_joint_draft_q_online_probe.py --arm "$warmup_arm" \
  --warmup-only --tokenizer /users/Yizhzhu/pvd-models/Qwen2.5-7B-Instruct \
  --source-root . --output "$evidence/warmup-$arm-final.jsonl"
python benchmark/pvd_output_quality.py collect --arm "$arm" \
  --dataset "$evidence/dataset.json" --output "$evidence/$arm.jsonl" "${resume_args[@]}"
