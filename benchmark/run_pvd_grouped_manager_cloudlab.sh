#!/usr/bin/env bash
set -euo pipefail
source /users/Yizhzhu/.sglang-v100-pvd-env.sh
root="$SGLANG_PVD_ROOT"
checkout="${PVD_CHECKOUT:-$root/validation/pvd-group-online-20260929}"
site="$root/deps/pvd-cagra-extend25-venv/lib/python3.12/site-packages"
nvidia="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia"
export PYTHONPATH="$checkout/python:$site"
export LD_LIBRARY_PATH="$nvidia/cublas/lib:$nvidia/cusolver/lib:$nvidia/cusparse/lib:$nvidia/nvjitlink/lib:$nvidia/cuda_runtime/lib:$site/libcuvs/lib64:$site/libraft/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=0
"$CONDA_PREFIX/bin/python" "$checkout/test/registered/disaggregation/run_pvd_chunked_index_gpu.py"
