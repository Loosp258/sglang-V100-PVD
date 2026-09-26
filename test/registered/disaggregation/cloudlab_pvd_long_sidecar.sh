#!/usr/bin/env bash
# Start isolated CloudLab long-context acceptance sidecars.
# The short-context 9000/8000/30000/30001 ports are never used here.
set -euo pipefail

role="${1:-}"
export MC_DISABLE_METACACHE=1
log_tag="${PVD_LONG_RUN_TAG:-large}"
if [[ ! "$log_tag" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo 'PVD_LONG_RUN_TAG must be a filename-safe label' >&2
  exit 2
fi

case "$role" in
  p)
    root=/mnt/sglang-data/yiliu124-node-0-sglang-pvd
    work="$root/pvd-long-acceptance-20260925"
    checkout="${PVD_P_CHECKOUT:-$root/src/sglang-PVD-validate-8a96123b0-long}"
    export PYTHONPATH="$checkout/python:$root/deps/pvd-validation-mooncake"
    export SGLANG_HOST_IP=10.0.1.1
    if pgrep -f 'sglang.launch_server.*--port 30002' >/dev/null; then
      echo 'Refusing to start: isolated P server 30002 already exists' >&2
      exit 1
    fi
    nohup setsid "$root/conda-envs/sglang-v100/bin/python" -m sglang.launch_server \
      --model-path /proj/edgecut-PG0/models/Qwen2.5-7B-Instruct \
      --dtype float16 --tp-size 1 --base-gpu-id 1 \
      --page-size 1 --attention-backend torch_native \
      --host 10.0.1.1 --port 30002 \
      --disaggregation-mode prefill --disaggregation-topology pvd \
      --pvd-vector-coordinator-url http://10.0.1.2:9100 \
      --pvd-model-instance-id qwen25-7b-pvd \
      --pvd-rank-rails mlx5_0 --disaggregation-transfer-backend mooncake \
      --pvd-transfer-staging-budget-bytes 67108864 \
      --pvd-transfer-max-inflight 16 \
      --mem-fraction-static 0.5 --context-length 2304 \
      --max-total-tokens 2304 --max-running-requests 4 \
      --max-prefill-tokens 2304 \
      --disable-cuda-graph --disable-overlap-schedule \
      --log-level info >"$work/p-$log_tag.log" 2>&1 </dev/null &
    ;;
  v)
    root=/mnt/sglang-data/yiliu124-node-1-sglang-pvd
    work="$root/pvd-long-acceptance-20260925"
    checkout="${PVD_V_CHECKOUT:-$root/src/sglang-PVD-validate-91c284b12}"
    export PYTHONPATH="$checkout/python:$root/deps/pvd-validation-mooncake"
    cuda_lib="$root/conda-envs/sglang-v100/lib/python3.12/site-packages/nvidia"
    cuvs_lib="$root/deps/pvd-cagra25-venv/lib/python3.12/site-packages"
    export LD_LIBRARY_PATH="$cuda_lib/cublas/lib:$cuda_lib/cusolver/lib:$cuda_lib/cusparse/lib:$cuda_lib/nvjitlink/lib:$cuda_lib/cuda_runtime/lib:$cuvs_lib/libcuvs/lib64:$cuvs_lib/libraft/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    if pgrep -f 'sglang.srt.disaggregation.pvd.server.*--coordinator-port 9100' >/dev/null; then
      echo 'Refusing to start: isolated V coordinator 9100 already exists' >&2
      exit 1
    fi
    triton_sparse_args=()
    case "${PVD_TRITON_SPARSE_PACKING:-0}" in
      0) ;;
      1) triton_sparse_args+=(--experimental-triton-sparse-packing) ;;
      *) echo 'PVD_TRITON_SPARSE_PACKING must be 0 or 1' >&2; exit 2 ;;
    esac
    exact_index_args=()
    if [[ -n "${PVD_LONG_EXACT_MAX_ROWS:-}" ]]; then
      if [[ ! "$PVD_LONG_EXACT_MAX_ROWS" =~ ^[1-9][0-9]{0,3}$ ]] ||
         (( PVD_LONG_EXACT_MAX_ROWS < 16 || PVD_LONG_EXACT_MAX_ROWS > 2304 )); then
        echo 'PVD_LONG_EXACT_MAX_ROWS must be an integer in 16..2304' >&2
        exit 2
      fi
      exact_index_args+=(--prompt-index-exact-max-rows "$PVD_LONG_EXACT_MAX_ROWS")
    fi
    nohup setsid "$root/deps/pvd-cagra25-venv/bin/python" -m sglang.srt.disaggregation.pvd.server \
      --world-size 2 --host 0.0.0.0 --advertise-host 10.0.1.2 \
      --coordinator-port 9100 --shard-port-base 9300 \
      --pvd-rank-devices 0,1 --pvd-rank-rails mlx5_0,mlx5_0 \
      --transfer-backend mooncake --strict-rdma-preflight \
      --total-pages 8192 --page-bytes 57344 \
      --transfer-staging-budget-bytes 67108864 --transfer-max-inflight 16 \
      --prompt-index-vector-space qwen25-7b-pvd \
      --prompt-index-budget-bytes 1073741824 \
      --prompt-index-backend cagra-auto \
      --prompt-index-cagra-native-bytes 536870912 \
      --prompt-index-cagra-global-native-bytes 671088640 \
      --prompt-index-cagra-graph-degree 8 \
      --prompt-index-cagra-intermediate-degree 16 \
      --prompt-index-exact-max-rows 64 \
      --prompt-index-cagra-itopk-size 64 \
      "${exact_index_args[@]}" \
      --experimental-cuda-sparse-packing \
      "${triton_sparse_args[@]}" \
      --full-kv-fanin-max-slices 262144 \
      --full-kv-fanin-max-inflight 2 \
      --full-kv-fanin-max-records 1024 \
      --full-kv-fanin-native-batch >"$work/v-$log_tag.log" 2>&1 </dev/null &
    ;;
  d)
    root=/mnt/sglang-data/yiliu124-node-2-sglang-pvd
    work="$root/pvd-long-acceptance-20260925"
    checkout="${PVD_D_CHECKOUT:-$root/src/sglang-PVD-validate-8a96123b0-long}"
    predictive_args=()
    case "${PVD_LONG_MODE:-predictive}" in
      predictive)
        serving_config="${PVD_LONG_LIMITS_PATH:-$work/limits.json}"
        if [[ "$serving_config" != /* || ! -f "$serving_config" ]]; then
          echo 'PVD_LONG_LIMITS_PATH must name an existing absolute config file' >&2
          exit 2
        fi
        retrieval_top_k="${PVD_LONG_TOP_K:-4}"
        retrieval_union_cap="${PVD_LONG_UNION_CAP:-32}"
        for value in "$retrieval_top_k" "$retrieval_union_cap"; do
          if [[ ! "$value" =~ ^[1-9][0-9]{0,3}$ ]] || (( value > 512 )); then
            echo 'PVD_LONG_TOP_K and PVD_LONG_UNION_CAP must be integers in 1..512' >&2
            exit 2
          fi
        done
        if (( retrieval_union_cap < retrieval_top_k )); then
          echo 'PVD_LONG_UNION_CAP must be at least PVD_LONG_TOP_K' >&2
          exit 2
        fi
        predictive_args=(
          --pvd-draft-model-path "$root/models/Qwen2.5-0.5B-Instruct"
          --pvd-draft-revision 7ae557604adf67be50417f59c2c2f167def9a775
          --pvd-draft-device cuda:1 --pvd-draft-mem-fraction-static 0.1
          --pvd-draft-scratch-budget-bytes 268435456
          --pvd-draft-persistent-budget-bytes 2147483648
          --pvd-draft-predict-tokens 2 --pvd-predictive-retrieval-config
          --pvd-cuda-predictive-serving --pvd-cuda-serving-config "$serving_config"
          --pvd-retrieval-vector-space qwen25-7b-pvd
          --pvd-retrieval-top-k "$retrieval_top_k"
          --pvd-retrieval-max-union-tokens "$retrieval_union_cap"
          --pvd-retrieval-bank-budget-bytes 268435456
          --pvd-retrieval-scratch-budget-bytes 536870912
        )
        ;;
      full) ;;
      *) echo 'PVD_LONG_MODE must be predictive or full' >&2; exit 2 ;;
    esac
    # Long online target forwards can exceed the search client's HTTP timeout.
    # Keep V search I/O advancing while the scheduler executes that forward.
    export PVD_SEARCH_BACKGROUND_IO="${PVD_SEARCH_BACKGROUND_IO:-1}"
    export PYTHONPATH="$checkout/python:$root/deps/pvd-validation-mooncake"
    if [[ -n "${PVD_STACK_SIGNAL_DIR:-}" ]]; then
      export PYTHONPATH="$PVD_STACK_SIGNAL_DIR:$PYTHONPATH"
    fi
    if pgrep -f 'sglang.launch_server.*--port 30003' >/dev/null; then
      echo 'Refusing to start: isolated D server 30003 already exists' >&2
      exit 1
    fi
    rank_packed_args=()
    if [[ "${PVD_RANK_PACKED_FANIN:-0}" == 1 ]]; then
      rank_packed_args+=(--pvd-full-kv-fanin-rank-packed)
    fi
    nohup setsid "$root/conda-envs/sglang-v100/bin/python" -m sglang.launch_server \
      --model-path /proj/edgecut-PG0/models/Qwen2.5-7B-Instruct \
      --device cuda --dtype float16 --tp-size 1 --base-gpu-id 1 \
      --page-size 1 --attention-backend torch_native \
      --host 10.0.1.3 --port 30003 \
      --disaggregation-mode decode --disaggregation-topology pvd \
      --pvd-vector-coordinator-url http://10.0.1.2:9100 \
      --pvd-model-instance-id qwen25-7b-pvd \
      --pvd-rank-rails mlx5_0 --pvd-d-receive-rails mlx5_0 \
      --disaggregation-transfer-backend mooncake \
      --pvd-transfer-staging-budget-bytes 268435456 \
      --pvd-transfer-max-inflight 16 --pvd-waiting-queue-bootstrap \
      --pvd-full-kv-fanin-max-slices 262144 \
      --pvd-full-kv-fanin-response-bytes 67108864 \
      "${rank_packed_args[@]}" \
      --pvd-kv-refresh-interval 4 --num-reserved-decode-tokens 16 \
      "${predictive_args[@]}" \
      --mem-fraction-static 0.5 --context-length 2304 \
      --max-total-tokens 2304 --max-running-requests 4 \
      --max-prefill-tokens 2304 \
      --disable-cuda-graph --disable-overlap-schedule \
      --log-level info >"$work/d-$log_tag.log" 2>&1 </dev/null &
    ;;
  gateway)
    root=/mnt/sglang-data/yiliu124-node-1-sglang-pvd
    work="$root/pvd-long-acceptance-20260925"
    gateway="${PVD_GATEWAY_BIN:-$root/deps/pvd-gateway-target-fbd77287b/release/smg}"
    if pgrep -f 'smg launch.*--port 8001' >/dev/null; then
      echo 'Refusing to start: isolated Gateway 8001 already exists' >&2
      exit 1
    fi
    nohup setsid "$gateway" launch --host 10.0.1.2 --port 8001 \
      --prometheus-port 29001 --pvd-disaggregation \
      --pvd-vector-coordinator-url http://10.0.1.2:9100 \
      --prefill http://10.0.1.1:30002 none \
      --decode http://10.0.1.3:30003 --log-level info \
      >"$work/gateway-$log_tag.log" 2>&1 </dev/null &
    ;;
  *)
    echo 'Usage: cloudlab_pvd_long_sidecar.sh {p|v|d|gateway}' >&2
    exit 2
    ;;
esac

echo "Started isolated $role launcher PID $!"
