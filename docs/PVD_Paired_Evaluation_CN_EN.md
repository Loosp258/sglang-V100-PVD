# PVD 配对评估 / Paired evaluation

`test/registered/disaggregation/run_pvd_paired_eval.py` 是不依赖 GPU 的
顺序请求采集与比较工具；实际推理由已启动的三机服务完成。它不会启动、
切换或验证 D 的模式，也不会从 token 一致率推断语义质量。

`run_pvd_paired_eval.py` is a CPU-only sequential collection/comparison
tool. The running three-node service performs inference. The script neither
starts nor verifies D's mode, and token agreement is not semantic quality.

准备 UTF-8 JSONL，每行仅需唯一 `id` 和 `text`。使用固定数据集、模型版本、
采样参数、GPU 资源与 P/V/Gateway；**只切换 D 模式**。分别采集：

Prepare UTF-8 JSONL with a unique `id` and `text` on each line. Keep the
dataset, model revisions, sampling settings, GPU resources and P/V/Gateway
fixed; **switch only D's mode**. Collect both reports:

仓库附带的 `test/registered/disaggregation/pvd_eval_smoke.jsonl` 只有 3 条
短/中长度 Prompt，仅用于验证采集链路，不是正式质量集。
The checked-in `test/registered/disaggregation/pvd_eval_smoke.jsonl` has only
three short/mid-length Prompts for validating the collection path; it is not
a representative quality dataset.

```bash
python test/registered/disaggregation/run_pvd_paired_eval.py collect \
  --gateway-url http://10.0.1.2:8000 --dataset eval.jsonl \
  --mode full --config-id qwen25-7b-full-v1 \
  --max-new-tokens 20 --output full.json

python test/registered/disaggregation/run_pvd_paired_eval.py collect \
  --gateway-url http://10.0.1.2:8000 --dataset eval.jsonl \
  --mode predictive --config-id qwen25-7b-predictive-v1 \
  --max-new-tokens 20 --output predictive.json

python test/registered/disaggregation/run_pvd_paired_eval.py compare \
  --full full.json --predictive predictive.json
```

报告只保存输入数据集 SHA-256、请求 ID、输出 token ID、Prompt/输出 token
数和客户端耗时，不重复保存 Prompt 或模型回复文本。采集失败时非零退出；
输出文件若已存在则拒绝覆盖。比较前要求数据集哈希、生成长度、请求 ID
和顺序一致。比较项为输出完全一致比例、平均共同前缀长度、客户端延迟
中位数及 nearest-rank p95；样本少时 p95 不稳定。模式标签完全由操作者
声明，`mode_verified_by_script=false`，必须另行保存 D 启动命令/日志。

Reports contain the dataset SHA-256, request IDs, output token IDs, token
counts and client latency, not copies of prompts or response text. Collection
fails nonzero on an incomplete reply and never overwrites an existing output
file. Comparison requires identical dataset hash, output length, request IDs
and order. It reports exact token agreement, mean common-prefix length,
median latency and nearest-rank p95; p95 is unstable for small samples. The
operator supplies each mode label (`mode_verified_by_script=false`), so retain
the D startup command and logs as independent evidence.

正式验收仍需语义/任务质量评估、真实 Q 的 CAGRA Top-K 召回率、长上下文与并发负载、
TPOT/吞吐/刷新边界等待、GPU 显存峰值和 RDMA/生命周期检查。顺序客户端耗时
不等于服务器吞吐，也不能证明网络等待已经隐藏。

D 的 CUDA 刷新驱动还会在安全安装时记录
`PVD boundary installed: ... observed_to_install_seconds=...`。
该值从 Scheduler **首次观察到** committed 计数抵达边界开始，至安装
成功为止；包含轮询和安装开销，是观测到的边界等待，**不是** token
真实生成时刻到安装的精确延迟，也不能单独归因于网络。若请求失败或
取消而没有安全安装，不产生“成功安装”日志。

Final acceptance additionally requires task/semantic quality, actual-Q CAGRA
Top-K recall, longer contexts and concurrent load, TPOT/throughput/boundary
wait, peak GPU memory and RDMA/lifecycle evidence. Sequential client latency
is not server throughput and does not prove that network waits are hidden.

On safe installation, D's CUDA refresh driver also logs
`PVD boundary installed: ... observed_to_install_seconds=...`. It measures
from the Scheduler's **first observation** of the committed boundary until
successful installation. It includes polling and installation overhead: it
is neither the exact interval since token generation nor network-only time.
A failed or cancelled refresh emits no successful-install log.

## 流式时间线 / Streaming timeline

可对固定 Prompt 用 `run_pvd_stream_probe.py` 记录 TTFT 和每个 SSE token
的到达时间；它拒绝非 SSE、倒退或缺失的 token 计数以及不完整流。
若一个 SSE 事件一次增加多个 token，它会报告 `coalesced_tokens>0`
并将 `true_tpot_observable=false`，不把同一事件的时间戳误作真实逐
token TPOT。即使每 token 都有事件，客户端间隔仍包含网络和路由开销。

Use `run_pvd_stream_probe.py` on a fixed Prompt to observe TTFT and each
SSE token's arrival time. It rejects non-SSE responses, regressed/missing
token counts and incomplete streams. If an event carries several tokens,
`coalesced_tokens>0` and `true_tpot_observable=false`; duplicated event
timestamps are not claimed as true per-token TPOT. Even one event per token
still includes client/network/router overhead.

```bash
python test/registered/disaggregation/run_pvd_stream_probe.py \
  --gateway-url http://10.0.1.2:8000 \
  --text 'PVD_TPOT_001: Explain GPU RDMA in one sentence.' \
  --max-new-tokens 20
```

## 冷启动与就绪 / Cold start and readiness

PVD 必须跳过 SGLang 通用 PD HTTP warmup：该请求没有 Gateway 选定的
P/V/D 身份。D 的 `/health=200` 因而不代表首个端到端请求已经预热。
在 Qwen2 target **且** Qwen2 draft 的当前 V100S 配置中，可在启动 D
之前设置 `PVD_PRECOMPILE_QWEN_KERNELS=1`，把四类已知 Decode JIT
特化（int64 位置、RoPE、SiLU、KV-store）的编译放到服务就绪之前。
它不运行模型 forward，不修改请求/KV 状态；非 Qwen2 draft 不要开启。
`PVD_PROFILE_COLD_STAGES=1` 是插入 CUDA 同步点的诊断开关，正常
benchmark 应保持关闭。

PVD skips SGLang's generic PD HTTP warmup because that request lacks the
Gateway-selected P/V/D identities. Therefore D `/health=200` does not imply
that the first end-to-end request is warm. With a Qwen2 target **and** Qwen2
draft on the current V100S setup, set `PVD_PRECOMPILE_QWEN_KERNELS=1`
before starting D to compile the four known Decode JIT specializations
(int64 positions, RoPE, SiLU and KV-store) before readiness. It performs
no model forward and changes no request/KV state. Leave it off for a
non-Qwen2 draft. `PVD_PROFILE_COLD_STAGES=1` adds CUDA fences for diagnosis;
keep it off for normal benchmarks.

三机 Gateway 与 P/V/D 均健康后，先通过 Gateway 运行上面的流式探针。
首次请求成功、20/20 token 完整且各端无错误之后，再采正式性能样本；
首次请求的冷态耗时应**单独报告**，不要静默丢弃。短 Prompt 预热不会
代替长 Prompt 的 CAGRA 建图，正式测试长度与阈值也必须覆盖。
当前实机的预编译首请求约 4.20 秒、最大间隔约 0.416 秒，但刷新
边界仍有可观测等待，不能把“消除 JIT 冷停顿”表述为“网络已隐藏”。

After all three machines and the Gateway report healthy, run the streaming
probe above through the Gateway. Require a complete 20/20-token response
and no node-side errors before collecting steady measurements, but report
the first cold request separately instead of silently discarding it. A
short-Prompt warmup does not build the longer-Prompt CAGRA graphs; cover
the actual benchmark lengths and exact/CAGRA threshold. On the measured
V100S setup, the precompiled first request took about 4.20 s with a
0.416 s maximum token gap. Refresh boundaries still waited, so removing
cold JIT delay is not evidence that network/search latency is hidden.
