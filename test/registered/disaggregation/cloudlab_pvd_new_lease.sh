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
context_tokens="${PVD_CONTEXT_TOKENS:-2304}"
fanin_max_slices="${PVD_FANIN_MAX_SLICES:-262144}"
d_staging_bytes="${PVD_D_STAGING_BYTES:-268435456}"
v_total_pages="${PVD_V_TOTAL_PAGES:-8192}"
probe_scratch_bytes="${PVD_PROBE_SCRATCH_BYTES:-536870912}"
draft_scratch_bytes="${PVD_DRAFT_SCRATCH_BYTES:-268435456}"
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
if [[ ! "$context_tokens" =~ ^[1-9][0-9]{3,4}$ ]] ||
   (( context_tokens < 2304 || context_tokens > 20480 )) ||
   [[ ! "$fanin_max_slices" =~ ^[1-9][0-9]{5,7}$ ]] ||
   (( fanin_max_slices < 262144 || fanin_max_slices > 2097152 )) ||
   [[ ! "$d_staging_bytes" =~ ^[1-9][0-9]{8,9}$ ]] ||
   (( d_staging_bytes < 268435456 || d_staging_bytes > 4294967296 )) ||
   [[ ! "$v_total_pages" =~ ^[1-9][0-9]{3,4}$ ]] ||
   (( v_total_pages < context_tokens || v_total_pages > 32768 )) ||
   [[ ! "$probe_scratch_bytes" =~ ^[1-9][0-9]{8,9}$ ]] ||
   (( probe_scratch_bytes < 536870912 || probe_scratch_bytes > 4294967296 )) ||
   [[ ! "$draft_scratch_bytes" =~ ^[1-9][0-9]{8,9}$ ]] ||
   (( draft_scratch_bytes < 268435456 || draft_scratch_bytes > 2147483648 )); then
  echo 'invalid bounded long-context capacity settings' >&2
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
      --mem-fraction-static 0.5 --context-length "$context_tokens" \
      --max-total-tokens "$context_tokens" --max-running-requests 4 \
      --max-prefill-tokens "$context_tokens" --disable-cuda-graph \
      --disable-overlap-schedule --log-level info \
      >"$log_dir/p-$tag.log" 2>&1 </dev/null &
    ;;
  v)
    require_free_port 9100
    # This lease caps model context at 2304. A cold native CAGRA build for a
    # 2095-row, one-shot Entry took >20 s end-to-end; exact search returned
    # the same answer in <4 s. Keep CAGRA selectable via an explicit lower
    # threshold for its separate recall/long-index acceptance experiments.
    exact_max_rows="${PVD_PROMPT_INDEX_EXACT_MAX_ROWS:-2304}"
    index_budget="${PVD_PROMPT_INDEX_BUDGET_BYTES:-2147483648}"
    if [[ ! "$exact_max_rows" =~ ^[1-9][0-9]{0,4}$ ]] ||
       (( exact_max_rows < 16 || exact_max_rows > context_tokens )); then
      echo 'PVD_PROMPT_INDEX_EXACT_MAX_ROWS must be in [16, context]' >&2
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
      --total-pages "$v_total_pages" --page-bytes 57344 \
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
      --full-kv-fanin-max-slices "$fanin_max_slices" \
      --full-kv-fanin-max-inflight 2 \
      --full-kv-fanin-max-records 1024 \
      --full-kv-fanin-native-batch \
      >"$log_dir/v-$tag.log" 2>&1 </dev/null &
    ;;
  d)
    require_free_port 30003
    require_model "$model"
    d_base_gpu=1
    d_draft_device=cuda:1
    case "${PVD_PROBE_SIDECAR:-0}" in
      0) ;;
      1)
        # Expose only physical GPU 1 to the committed D Scheduler. Its child
        # sidecar is separately restricted to physical GPU 0 by the strict
        # probe_sidecar config; both processes then address their GPU as 0.
        export CUDA_VISIBLE_DEVICES=1
        d_base_gpu=0
        d_draft_device=cuda:0
        ;;
      *) echo 'PVD_PROBE_SIDECAR must be 0 or 1' >&2; exit 2 ;;
    esac
    refresh_interval="${PVD_REFRESH_INTERVAL:-4}"
    draft_predict_tokens="${PVD_DRAFT_PREDICT_TOKENS:-2}"
    retrieval_top_k="${PVD_RETRIEVAL_TOP_K:-4}"
    retrieval_union_tokens="${PVD_RETRIEVAL_UNION_TOKENS:-32}"
    reserved_tokens="${PVD_RESERVED_DECODE_TOKENS:-16}"
    if [[ ! "$refresh_interval" =~ ^[1-9][0-9]?$ ]] ||
       (( refresh_interval > 64 )) ||
       [[ ! "$draft_predict_tokens" =~ ^[1-9][0-9]?$ ]] ||
       (( draft_predict_tokens > 32 )) ||
       [[ ! "$reserved_tokens" =~ ^[1-9][0-9]?$ ]] ||
       (( reserved_tokens > 64 )); then
      echo 'invalid bounded refresh, prediction or reserved-token setting' >&2
      exit 2
    fi
    if [[ ! "$retrieval_top_k" =~ ^[1-9][0-9]?$ ]] ||
       (( retrieval_top_k > 16 )) ||
       [[ ! "$retrieval_union_tokens" =~ ^[1-9][0-9]{0,2}$ ]] ||
       (( retrieval_union_tokens < retrieval_top_k || retrieval_union_tokens > 128 )); then
      echo 'PVD_RETRIEVAL_TOP_K must be in [1, 16] and union in [top_k, 128]' >&2
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
        if [[ "${PVD_PROBE_SIDECAR:-0}" == 1 ]] &&
           ! jq -e '.probe_sidecar != null' "$limits" >/dev/null; then
          echo 'PVD_PROBE_SIDECAR requires a probe_sidecar limits object' >&2
          exit 2
        fi
        predictive_args=(
          --pvd-draft-model-path "$draft"
          --pvd-draft-revision 7ae557604adf67be50417f59c2c2f167def9a775
          --pvd-draft-device "$d_draft_device" --pvd-draft-mem-fraction-static 0.1
          --pvd-draft-scratch-budget-bytes "$draft_scratch_bytes"
          --pvd-draft-persistent-budget-bytes 2147483648
          --pvd-draft-predict-tokens "$draft_predict_tokens"
          --pvd-predictive-retrieval-config
          --pvd-cuda-predictive-serving --pvd-cuda-serving-config "$limits"
          --pvd-retrieval-vector-space qwen25-7b-pvd
          --pvd-retrieval-top-k "$retrieval_top_k"
          --pvd-retrieval-max-union-tokens "$retrieval_union_tokens"
          --pvd-retrieval-bank-budget-bytes 268435456
          --pvd-retrieval-scratch-budget-bytes "$probe_scratch_bytes"
        )
        ;;
      full) ;;
      *) echo 'PVD_MODE must be predictive or full' >&2; exit 2 ;;
    esac
    nohup setsid "$python" -m sglang.launch_server \
      --model-path "$model" --device cuda --dtype float16 \
      --tp-size 1 --base-gpu-id "$d_base_gpu" --page-size 1 \
      --attention-backend torch_native --host "$d_ip" --port 30003 \
      --disaggregation-mode decode --disaggregation-topology pvd \
      --pvd-vector-coordinator-url "http://$v_ip:9100" \
      --pvd-model-instance-id qwen25-7b-pvd \
      --pvd-rank-rails "$rail" --pvd-d-receive-rails "$rail" \
      --disaggregation-transfer-backend mooncake \
      --pvd-transfer-staging-budget-bytes "$d_staging_bytes" \
      --pvd-transfer-max-inflight 16 --pvd-waiting-queue-bootstrap \
      --pvd-full-kv-fanin-max-slices "$fanin_max_slices" \
      --pvd-full-kv-fanin-response-bytes 67108864 \
      "${rank_packed_args[@]}" \
      --pvd-kv-refresh-interval "$refresh_interval" \
      --num-reserved-decode-tokens "$reserved_tokens" \
      "${predictive_args[@]}" \
      --mem-fraction-static 0.5 --context-length "$context_tokens" \
      --max-total-tokens "$context_tokens" --max-running-requests 4 \
      --max-prefill-tokens "$context_tokens" --disable-cuda-graph \
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
