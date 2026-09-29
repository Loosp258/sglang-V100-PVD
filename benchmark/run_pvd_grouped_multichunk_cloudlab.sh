#!/usr/bin/env bash
# Run the real-Q grouped multi-extend quality probe on an idle CloudLab V100S.
set -euo pipefail

source /users/Yizhzhu/.sglang-v100-pvd-env.sh
root="$SGLANG_PVD_ROOT"
checkout="${PVD_CHECKOUT:-$root/validation/pvd-stream-online-20260929}"
probe="${PVD_PROBE_SCRIPT:-$checkout/test/registered/disaggregation/run_pvd_qwen_grouped_cagra_gpu.py}"
cuvs_site="$root/deps/pvd-cagra-extend25-venv/lib/python3.12/site-packages"
nvidia_site="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia"
export PYTHONPATH="$checkout/python:$checkout/test/registered/disaggregation:$root/validation:$cuvs_site"
export LD_LIBRARY_PATH="$nvidia_site/cublas/lib:$nvidia_site/cusolver/lib:$nvidia_site/cusparse/lib:$nvidia_site/nvjitlink/lib:$nvidia_site/cuda_runtime/lib:$cuvs_site/libcuvs/lib64:$cuvs_site/libraft/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PVD_CAGRA_GROUP_FULL_SHARD=1
export PVD_CAGRA_GROUP_SIZES="${PVD_CAGRA_GROUP_SIZES:-1,2}"
export PVD_CAGRA_GROUP_ITOPK="${PVD_CAGRA_GROUP_ITOPK:-256}"
export PVD_CAGRA_GROUP_PREFIX="${PVD_CAGRA_GROUP_PREFIX:-512}"
export PVD_CAGRA_GROUP_CHUNK_ROWS="${PVD_CAGRA_GROUP_CHUNK_ROWS:-512}"
export PVD_CAGRA_PREFILL_CHUNK_ROWS="${PVD_CAGRA_PREFILL_CHUNK_ROWS:-512}"
export PVD_CAGRA_RECALL_ROWS="${PVD_CAGRA_RECALL_ROWS:-2155}"

"$CONDA_PREFIX/bin/python" "$probe" \
  --architecture qwen2 \
  --model-path "$root/models/Qwen2.5-7B-Instruct" \
  --context-length "${PVD_CAGRA_CONTEXT_LENGTH:-2304}" \
  --max-total-tokens "${PVD_CAGRA_MAX_TOTAL_TOKENS:-4096}"
