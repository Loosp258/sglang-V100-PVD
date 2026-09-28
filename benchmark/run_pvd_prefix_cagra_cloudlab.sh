#!/usr/bin/env bash
# Run the isolated prefix-graph microbenchmark on the idle V CloudLab node.
set -euo pipefail

source /users/Yizhzhu/.sglang-v100-pvd-env.sh
root="$SGLANG_PVD_ROOT"
checkout="$root/validation/pvd-concurrent-b35752ffb"
scratch="$root/validation/prefix-cagra-spike-20260929"
cuvs_site="$root/deps/pvd-cagra25-venv/lib/python3.12/site-packages"
nvidia_site="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia"

test -f "$checkout/python/sglang/srt/disaggregation/pvd/cagra_backend.py"
test -d "$cuvs_site/cuvs"
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo 'GPU already has compute processes; refusing benchmark' >&2
  exit 2
fi

export PYTHONPATH="$scratch:$checkout/python:$cuvs_site${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$nvidia_site/cublas/lib:$nvidia_site/cusolver/lib:$nvidia_site/cusparse/lib:$nvidia_site/nvjitlink/lib:$nvidia_site/cuda_runtime/lib:$cuvs_site/libcuvs/lib64:$cuvs_site/libraft/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
echo "checkout=$(git -C "$checkout" rev-parse HEAD)"
echo "python=$(command -v python)"
module=benchmark.pvd_prefix_cagra_cloudlab_bench
if [[ "${1:-}" == --search-only ]]; then
  module=benchmark.pvd_prefix_cagra_search_cloudlab_bench
  shift
fi
python -m "$module" "$@"
