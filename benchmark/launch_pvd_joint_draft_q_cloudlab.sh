#!/usr/bin/env bash
# Same resources and graph predictor for the exact-Q and learned-Q arms.
set -euo pipefail
role="${1:?role p/v/d/gateway required}"
arm="${2:?arm full/target/joint required}"
source /users/Yizhzhu/.sglang-v100-pvd-env.sh
export PVD_CHECKOUT="${PVD_CHECKOUT:-$SGLANG_PVD_ROOT/validation/pvd-joint-draft-q-20260930}"
export PVD_EXPECTED_COMMIT="${PVD_EXPECTED_COMMIT:-$(git -C "$PVD_CHECKOUT" rev-parse HEAD)}"
export PVD_RUN_TAG="${PVD_RUN_TAG:-jointq-$arm-20260930}"
export PVD_CHUNKED_CAGRA_UPLOAD=1 PVD_CAGRA_EXTEND25=1
export PVD_CAGRA_GROUP_HEADS=4 PVD_CAGRA_EXACT_HEAD_SEED=1 PVD_CAGRA_ITOPK_SIZE=2048
export PVD_DIRECT_PD_BOOTSTRAP=0 PVD_GATE_INITIAL_FANIN_ON_INDEX=1
export PVD_SPLIT_POLICY_FILE="$PVD_CHECKOUT/benchmark/results/pvd_split_parametric_profile_cloudlab_20260930.json"
export PVD_PREFILL_CHUNK_TOKENS=512 PVD_P_TP_SIZE=1 PVD_DISABLE_RADIX_CACHE=1
export PVD_CONTEXT_TOKENS=2304 PVD_V_TOTAL_PAGES="${PVD_V_TOTAL_PAGES:-16384}"
export PVD_DRAFT_PREDICT_TOKENS=8 PVD_REFRESH_INTERVAL=4
export PVD_RETRIEVAL_TOP_K=16 PVD_RETRIEVAL_UNION_TOKENS=128
export PVD_DRAFT_SCRATCH_BYTES=2147483648 PVD_PROBE_SCRATCH_BYTES=536870912
export PVD_COOPERATIVE_PREDICTION=0 PVD_CONCURRENT_PREDICTION=0
export PVD_SEED_PROBE_FROM_PROMPT_KV=0 PVD_PRECOMPILE_QWEN_KERNELS=0 PVD_PROBE_SIDECAR=0
case "$arm" in
  full) export PVD_MODE=full; unset PVD_JOINT_DRAFT_Q_CHECKPOINT ;;
  target) unset PVD_JOINT_DRAFT_Q_CHECKPOINT ;;
  joint)
    if [[ "$role" == d ]]; then
      export PVD_JOINT_DRAFT_Q_CHECKPOINT="${PVD_JOINT_DRAFT_Q_CHECKPOINT:-$SGLANG_PVD_ROOT/validation/joint-draft-q-checkpoint-20260930/trained.pt}"
    fi
    ;;
  *) echo 'arm must be full, target or joint' >&2; exit 2 ;;
esac
bash "$PVD_CHECKOUT/test/registered/disaggregation/cloudlab_pvd_new_lease.sh" "$role"
