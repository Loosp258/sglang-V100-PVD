#!/usr/bin/env bash
# Explicit experiment roles; does not stop or reconfigure any serving process.
set -euo pipefail
role="${1:?prepare/serve/decode required}"
source /users/Yizhzhu/.sglang-v100-pvd-env.sh
checkout="$SGLANG_PVD_ROOT/validation/pvd-oasiskv-20261001"
evidence="$SGLANG_PVD_ROOT/validation/oasiskv-20261001"
export CUDA_VISIBLE_DEVICES="${OASIS_GPU:-1}"
export PYTHONPATH="$checkout/python"
mkdir -p "$evidence"
cd "$checkout"
trap 'printf "%s\n" "$?" > "$evidence/$role.exit"' EXIT
case "$role" in
  prepare)
    python benchmark/pvd_oasis_experiment.py prepare \
      --target-model "$SGLANG_PVD_ROOT/models/Qwen2.5-7B-Instruct" \
      --questions "$checkout/benchmark/results/pvd_draft_q_decode_cloudlab_20260930/questions.json" \
      --output "$evidence/fixtures" --steps 16 --include-2155
    ;;
  serve)
    site="$SGLANG_PVD_ROOT/deps/pvd-cagra-extend25-venv/lib/python3.12/site-packages"
    nvidia="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia"
    export PYTHONPATH="$checkout/python:$site"
    export LD_LIBRARY_PATH="$nvidia/cublas/lib:$nvidia/cusolver/lib:$nvidia/cusparse/lib:$nvidia/nvjitlink/lib:$nvidia/cuda_runtime/lib:$site/libcuvs/lib64:$site/libraft/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    # Import cuVS before Torch for this isolated wheel environment.
    python -c 'from cuvs.neighbors import cagra; import runpy, sys; sys.argv=["benchmark/pvd_oasis_experiment.py", "serve", "--fixtures", sys.argv[1], "--bind", "10.10.1.2", "--port", "38931"]; runpy.run_path(sys.argv[0], run_name="__main__")' "$evidence/fixtures"
    ;;
  decode)
    python benchmark/pvd_oasis_experiment.py decode \
      --target-model "$SGLANG_PVD_ROOT/models/Qwen2.5-7B-Instruct" \
      --eagle-source "$evidence/EAGLE" --eagle-checkpoint "$evidence/checkpoint" \
      --v-url http://10.10.1.2:38931 --output "$evidence/results" \
      --fixture gsm8k_1876.pt --fixture hotpotqa_5a72224755429971e9dc92be.pt \
      --fixture case40-2155.pt --steps 16 --capacity 128 --max-new 16 --top-k 16
    ;;
  *) echo 'unknown experiment role' >&2; exit 2 ;;
esac
