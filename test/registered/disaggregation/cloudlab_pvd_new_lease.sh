#!/usr/bin/env bash
# Isolated three-node V100S PVD acceptance launcher for the 10.10.1.x lease.
# Run on node0 (p), node1 (v/gateway), or node2 (d). Override the IPs and
# checkout through environment variables when CloudLab changes the topology.
set -euo pipefail

role="${1:-}"
source /users/Yizhzhu/.sglang-v100-pvd-env.sh

root="$SGLANG_PVD_ROOT"
checkout="${PVD_CHECKOUT:-$root/validation/pvd-05d7a2afe}"
expected="${PVD_EXPECTED_COMMIT:-05d7a2afe98494bc75de92f7e7644d1072ca7f25}"
p_ip="${PVD_P_IP:-10.10.1.1}"
v_ip="${PVD_V_IP:-10.10.1.2}"
d_ip="${PVD_D_IP:-10.10.1.3}"
rail="${PVD_RAIL:-mlx5_0}"
tag="${PVD_RUN_TAG:-acceptance}"
log_dir="$root/validation/logs"
# Gateway groups P/D by model_path and loads that tokenizer on V. Each node's
# link has the same path and pinned tokenizer bytes, while P/D links include
# their own local weight shards.
model="/users/Yizhzhu/pvd-models/Qwen2.5-7B-Instruct"
draft="$root/models/Qwen2.5-0.5B-Instruct"
python="$CONDA_PREFIX/bin/python"

if [[ ! "$tag" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo 'PVD_RUN_TAG must be filename-safe' >&2
  exit 2
fi
if [[ ! -d "$checkout/python/sglang/srt/disaggregation/pvd" ]] ||
   [[ "$(git -C "$checkout" rev-parse HEAD)" != "$expected" ]]; then
  echo 'PVD checkout is absent or not the expected commit' >&2
  exit 2
fi
if [[ ! -r "/sys/class/infiniband/$rail/ports/1/state" ]] ||
   [[ "$(<"/sys/class/infiniband/$rail/ports/1/state")" != *ACTIVE* ]]; then
  echo "RDMA rail $rail is not ACTIVE" >&2
  exit 2
fi
export MC_DISABLE_METACACHE=1
export PYTHONPATH="$checkout/python${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$log_dir"

require_free_port() {
  local port="$1"
  if ss -H -ltn "sport = :$port" | grep -q .; then
    echo "port $port already has a listener" >&2
    exit 1
  fi
}

require_model() {
  local directory="$1"
  if [[ ! -s "$directory/config.json" ]] ||
     [[ ! -s "$directory/tokenizer.json" ]]; then
    echo "incomplete model directory: $directory" >&2
    exit 2
  fi
}

case "$role" in
  p)
    require_free_port 30002
    require_model "$model"
    export SGLANG_HOST_IP="$p_ip"
    nohup setsid "$python" -m sglang.launch_server \
      --model-path "$model" --dtype float16 --tp-size 1 --base-gpu-id 1 \
      --page-size 1 --attention-backend torch_native \
      --host "$p_ip" --port 30002 \
      --disaggregation-mode prefill --disaggregation-topology pvd \
      --pvd-vector-coordinator-url "http://$v_ip:9100" \
      --pvd-model-instance-id qwen25-7b-pvd \
      --pvd-rank-rails "$rail" --disaggregation-transfer-backend mooncake \
      --pvd-transfer-staging-budget-bytes 67108864 \
      --pvd-transfer-max-inflight 16 \
      --mem-fraction-static 0.5 --context-length 2304 \
      --max-total-tokens 2304 --max-running-requests 4 \
      --max-prefill-tokens 2304 --disable-cuda-graph \
      --disable-overlap-schedule --log-level info \
      >"$log_dir/p-$tag.log" 2>&1 </dev/null &
    ;;
  v)
    require_free_port 9100
    exact_max_rows="${PVD_PROMPT_INDEX_EXACT_MAX_ROWS:-512}"
    index_budget="${PVD_PROMPT_INDEX_BUDGET_BYTES:-1073741824}"
    if [[ ! "$exact_max_rows" =~ ^[1-9][0-9]{0,3}$ ]] ||
       (( exact_max_rows < 16 || exact_max_rows > 2304 )); then
      echo 'PVD_PROMPT_INDEX_EXACT_MAX_ROWS must be an integer in [16, 2304]' >&2
      exit 2
    fi
    if [[ ! "$index_budget" =~ ^[1-9][0-9]{0,9}$ ]] ||
       (( index_budget < 1073741824 || index_budget > 4294967296 )); then
      echo 'PVD_PROMPT_INDEX_BUDGET_BYTES must be in [1 GiB, 4 GiB]' >&2
      exit 2
    fi
    cuvs_site="$root/deps/pvd-cagra25-venv/lib/python3.12/site-packages"
    if [[ ! -d "$cuvs_site/cuvs" ]]; then
      echo 'isolated cuVS 25.02 environment is absent' >&2
      exit 2
    fi
    export PYTHONPATH="$checkout/python:$cuvs_site"
    nvidia_site="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia"
    export LD_LIBRARY_PATH="$nvidia_site/cublas/lib:$nvidia_site/cusolver/lib:$nvidia_site/cusparse/lib:$nvidia_site/nvjitlink/lib:$nvidia_site/cuda_runtime/lib:$cuvs_site/libcuvs/lib64:$cuvs_site/libraft/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    pack_args=()
    if [[ "${PVD_TRITON_SPARSE_PACKING:-0}" == 1 ]]; then
      pack_args+=(--experimental-triton-sparse-packing)
    fi
    nohup setsid "$python" -m sglang.srt.disaggregation.pvd.server \
      --world-size 2 --host 0.0.0.0 --advertise-host "$v_ip" \
      --coordinator-port 9100 --shard-port-base 9300 \
      --entry-ttl-secs 300 --pvd-rank-devices 0,1 \
      --pvd-rank-rails "$rail,$rail" \
      --transfer-backend mooncake --strict-rdma-preflight \
      --total-pages 8192 --page-bytes 57344 \
      --transfer-staging-budget-bytes 67108864 --transfer-max-inflight 16 \
      --prompt-index-vector-space qwen25-7b-pvd \
      --prompt-index-budget-bytes "$index_budget" \
      --prompt-index-backend cagra-auto \
      --prompt-index-cagra-native-bytes 536870912 \
      --prompt-index-cagra-global-native-bytes 671088640 \
      --prompt-index-cagra-graph-degree 8 \
      --prompt-index-cagra-intermediate-degree 16 \
      --prompt-index-exact-max-rows "$exact_max_rows" \
      --prompt-index-cagra-itopk-size 64 \
      --experimental-cuda-sparse-packing "${pack_args[@]}" \
      --full-kv-fanin-max-slices 262144 \
      --full-kv-fanin-max-inflight 2 \
      --full-kv-fanin-max-records 1024 \
      --full-kv-fanin-native-batch \
      >"$log_dir/v-$tag.log" 2>&1 </dev/null &
    ;;
  d)
    require_free_port 30003
    require_model "$model"
    refresh_interval="${PVD_REFRESH_INTERVAL:-4}"
    draft_predict_tokens="${PVD_DRAFT_PREDICT_TOKENS:-2}"
    if [[ ! "$refresh_interval" =~ ^[1-9][0-9]?$ ]] ||
       (( refresh_interval > 32 )) ||
       [[ ! "$draft_predict_tokens" =~ ^[1-9][0-9]?$ ]] ||
       (( draft_predict_tokens > 32 )); then
      echo 'PVD_REFRESH_INTERVAL and PVD_DRAFT_PREDICT_TOKENS must be in [1, 32]' >&2
      exit 2
    fi
    export PVD_SEARCH_BACKGROUND_IO=1
    export PVD_CONTIGUOUS_SPARSE_BANK_COPY="${PVD_CONTIGUOUS_SPARSE_BANK_COPY:-0}"
    export SGLANG_HOST_IP="$d_ip"
    rank_packed_args=()
    if [[ "${PVD_RANK_PACKED_FANIN:-1}" == 1 ]]; then
      rank_packed_args+=(--pvd-full-kv-fanin-rank-packed)
      if [[ "${PVD_TRITON_FANIN_SCATTER:-0}" == 1 ]]; then
        rank_packed_args+=(--pvd-full-kv-fanin-triton-scatter)
      fi
    fi
    predictive_args=()
    case "${PVD_MODE:-predictive}" in
      predictive)
        require_model "$draft"
        limits="${PVD_LIMITS_PATH:-$checkout/test/registered/disaggregation/pvd_qwen_v100s_serving_limits_triton_long.json}"
        if [[ ! -s "$limits" ]]; then
          echo "missing predictive serving config: $limits" >&2
          exit 2
        fi
        predictive_args=(
          --pvd-draft-model-path "$draft"
          --pvd-draft-revision 7ae557604adf67be50417f59c2c2f167def9a775
          --pvd-draft-device cuda:1 --pvd-draft-mem-fraction-static 0.1
          --pvd-draft-scratch-budget-bytes 268435456
          --pvd-draft-persistent-budget-bytes 2147483648
          --pvd-draft-predict-tokens "$draft_predict_tokens"
          --pvd-predictive-retrieval-config
          --pvd-cuda-predictive-serving --pvd-cuda-serving-config "$limits"
          --pvd-retrieval-vector-space qwen25-7b-pvd
          --pvd-retrieval-top-k 4 --pvd-retrieval-max-union-tokens 32
          --pvd-retrieval-bank-budget-bytes 268435456
          --pvd-retrieval-scratch-budget-bytes 536870912
        )
        ;;
      full) ;;
      *) echo 'PVD_MODE must be predictive or full' >&2; exit 2 ;;
    esac
    nohup setsid "$python" -m sglang.launch_server \
      --model-path "$model" --device cuda --dtype float16 \
      --tp-size 1 --base-gpu-id 1 --page-size 1 \
      --attention-backend torch_native --host "$d_ip" --port 30003 \
      --disaggregation-mode decode --disaggregation-topology pvd \
      --pvd-vector-coordinator-url "http://$v_ip:9100" \
      --pvd-model-instance-id qwen25-7b-pvd \
      --pvd-rank-rails "$rail" --pvd-d-receive-rails "$rail" \
      --disaggregation-transfer-backend mooncake \
      --pvd-transfer-staging-budget-bytes 268435456 \
      --pvd-transfer-max-inflight 16 --pvd-waiting-queue-bootstrap \
      --pvd-full-kv-fanin-max-slices 262144 \
      --pvd-full-kv-fanin-response-bytes 67108864 \
      "${rank_packed_args[@]}" \
      --pvd-kv-refresh-interval "$refresh_interval" \
      --num-reserved-decode-tokens 16 \
      "${predictive_args[@]}" \
      --mem-fraction-static 0.5 --context-length 2304 \
      --max-total-tokens 2304 --max-running-requests 4 \
      --max-prefill-tokens 2304 --disable-cuda-graph \
      --disable-overlap-schedule --log-level info \
      >"$log_dir/d-$tag.log" 2>&1 </dev/null &
    ;;
  gateway)
    require_free_port 8001
    require_model "$model"
    gateway_bin="${PVD_GATEWAY_BIN:-$CARGO_TARGET_DIR/release/smg}"
    if [[ ! -x "$gateway_bin" ]]; then
      echo "missing PVD Gateway binary: $gateway_bin" >&2
      exit 2
    fi
    nohup setsid "$gateway_bin" launch --host "$v_ip" --port 8001 \
      --prometheus-port 29001 --pvd-disaggregation \
      --pvd-vector-coordinator-url "http://$v_ip:9100" \
      --prefill "http://$p_ip:30002" none \
      --decode "http://$d_ip:30003" --log-level info \
      >"$log_dir/gateway-$tag.log" 2>&1 </dev/null &
    ;;
  *)
    echo 'Usage: cloudlab_pvd_new_lease.sh {p|v|d|gateway}' >&2
    exit 2
    ;;
esac

echo "Started $role launcher PID $!, log: $log_dir/$role-$tag.log"
