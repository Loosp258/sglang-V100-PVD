#!/usr/bin/env bash
# Isolated three-node smoke launcher. The final shell keeps tmux logs visible
# after a model server exits, so a crash cannot erase the diagnostic pane.
set -u

if [ "$#" -ne 8 ]; then
  printf 'usage: %s ROLE REPO PYTHON DEPS HOST V_URL MODEL BACKEND\n' "$0" >&2
  exit 2
fi

pvd_role="$1"
pvd_repo="$2"
pvd_python="$3"
pvd_deps="$4"
pvd_host="$5"
pvd_vector_url="$6"
pvd_model="$7"
pvd_backend="$8"
case "$pvd_role" in
  prefill) pvd_port=30000 ;;
  decode) pvd_port=30001 ;;
  *) printf 'invalid PVD role: %s\n' "$pvd_role" >&2; exit 2 ;;
esac

export PYTHONPATH="$pvd_repo/python:$pvd_deps"
export MC_DISABLE_METACACHE=1
export MOONCAKE_PROTOCOL=rdma
export SGLANG_HOST_IP="$pvd_host"
export CUDA_VISIBLE_DEVICES=0,1

"$pvd_python" -u -m sglang.launch_server \
  --model-path "$pvd_model" \
  --dtype float16 \
  --tp-size 2 \
  --page-size 16 \
  --attention-backend "$pvd_backend" \
  --host "$pvd_host" \
  --port "$pvd_port" \
  --disaggregation-mode "$pvd_role" \
  --disaggregation-topology pvd \
  --pvd-vector-coordinator-url "$pvd_vector_url" \
  --pvd-model-instance-id qwen25-7b-pvd \
  --pvd-rank-rails mlx5_0,mlx5_0 \
  --disaggregation-transfer-backend mooncake \
  --pvd-transfer-staging-budget-bytes 67108864 \
  --pvd-transfer-max-inflight 16 \
  --log-level info
pvd_exit_status="$?"
printf 'PVD_VALIDATION_SERVER_EXIT role=%s code=%s\n' "$pvd_role" "$pvd_exit_status" >&2
exec /bin/bash
