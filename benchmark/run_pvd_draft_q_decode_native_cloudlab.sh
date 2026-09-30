#!/usr/bin/env bash
# Native serving-shaped retrieval validation; no model or quality tuning here.
set -euo pipefail
source /users/Yizhzhu/.sglang-v100-pvd-env.sh
checkout="${PVD_CHECKOUT:-$SGLANG_PVD_ROOT/validation/pvd-joint-draft-q-20260930}"
evidence="${PVD_NATIVE_EVIDENCE_DIR:-$SGLANG_PVD_ROOT/validation/draft-q-decode-native-20260930}"
cuvs_site="$SGLANG_PVD_ROOT/deps/pvd-cagra-extend25-venv/lib/python3.12/site-packages"
nvidia_site="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia"
export PYTHONPATH="$checkout/python:$checkout/test/registered/disaggregation:$SGLANG_PVD_ROOT/validation:$cuvs_site"
export LD_LIBRARY_PATH="$nvidia_site/cublas/lib:$nvidia_site/cusolver/lib:$nvidia_site/cusparse/lib:$nvidia_site/nvjitlink/lib:$nvidia_site/cuda_runtime/lib:$cuvs_site/libcuvs/lib64:$cuvs_site/libraft/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=0
cd "$checkout"
set +e
"$CONDA_PREFIX/bin/python" benchmark/pvd_draft_q_decode_native.py native \
  --fixture "$evidence/native-fixtures.pt" --output "$evidence/native.json"
status=$?
printf '%s\n' "$status" > "$evidence/native.exit"
exit "$status"
