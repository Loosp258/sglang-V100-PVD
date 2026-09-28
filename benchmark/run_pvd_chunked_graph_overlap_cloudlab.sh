#!/usr/bin/env bash
set -euo pipefail

source /users/Yizhzhu/.sglang-v100-pvd-env.sh
root="$SGLANG_PVD_ROOT"
checkout="$root/validation/pvd-concurrent-b35752ffb"
scratch="$root/validation/prefix-cagra-spike-20260929"
cuvs_site="$root/deps/pvd-cagra25-venv/lib/python3.12/site-packages"
nvidia_site="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia"

test -f "$checkout/python/sglang/srt/disaggregation/pvd/cagra_backend.py"
test -f "$scratch/benchmark/pvd_chunked_graph_overlap_bench.py"
test -d "$cuvs_site/cuvs"
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo 'GPU already has compute processes; refusing benchmark' >&2
  exit 2
fi

export PYTHONPATH="$scratch:$checkout/python:$cuvs_site${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$nvidia_site/cublas/lib:$nvidia_site/cusolver/lib:$nvidia_site/cusparse/lib:$nvidia_site/nvjitlink/lib:$nvidia_site/cuda_runtime/lib:$cuvs_site/libcuvs/lib64:$cuvs_site/libraft/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python -m benchmark.pvd_chunked_graph_overlap_bench "$@"
