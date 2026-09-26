# PVD 新租期三机验收 / Three-node CloudLab acceptance, new lease

日期 / Date: 2026-09-27 (Asia/Taipei; node logs use UTC). Code: `05d7a2afe98494bc75de92f7e7644d1072ca7f25` plus the isolated launcher `test/registered/disaggregation/cloudlab_pvd_new_lease.sh`. These are bounded experiments, **not** proof of production readiness or an end-to-end speedup.

## 环境与可复现性 / Environment and reproducibility

| Role | CloudLab host | Private IP | GPU use |
| --- | --- | --- | --- |
| P | `clgpu020.clemson.cloudlab.us` | `10.10.1.1` | V100S GPU 1, TP1 |
| V + Gateway | `clgpu021.clemson.cloudlab.us` | `10.10.1.2` | V100S GPUs 0+1, one V process/two ranks |
| D | `clgpu019.clemson.cloudlab.us` | `10.10.1.3` | V100S GPU 1, TP1; draft on GPU 1 |

All nodes have two Tesla V100S-PCIE-32GB GPUs, driver 580.178.04, Python 3.12.14, PyTorch 2.9.1+cu128, Triton 3.5.1, and `mooncake-transfer-engine==0.3.13.post1`. `mlx5_0` is ACTIVE (25 GbE RoCE); `mlx5_1` is DOWN. These tests therefore used the explicitly labelled **single-rail debug mode**. Local GPU MR registration/transfer passed for P GPU0, V GPU0/GPU1 and D GPU0; byte-exact cross-node Mooncake WRITE passed P GPU0→V GPU0 and V GPU1→D GPU0. Serving used the GPU mappings in the table, whose startup GDR preflight also passed. The separate preflight is not a dual-rail test.

Target: `Qwen/Qwen2.5-7B-Instruct` revision `a09a35458c702b33eeacc393d103063234e8bc28`; draft: `Qwen/Qwen2.5-0.5B-Instruct` revision `7ae557604adf67be50417f59c2c2f167def9a775`. P/D have local target weights; V has matching tokenizer/config only. Their identical visible model path is `/users/Yizhzhu/pvd-models/Qwen2.5-7B-Instruct`, because Gateway groups P and D by `model_path` and loads the tokenizer locally. The three tokenizer/config files were SHA256-matched. The repos' original directories were not overwritten; detached validation worktrees are under each node's `/mnt/sglang-data/Yizhzhu-node*-sglang-pvd/validation/pvd-05d7a2afe`.

V's base Conda environment lacked cuVS. An isolated V-side venv supplies `cuvs-cu12==25.2.0`, `libcuvs-cu12==25.2.1`, `rmm-cu12==25.2.0`, and `cupy-cuda12x==13.3.0`; the launcher exposes only its package/library paths to V. The NVIDIA-index wheels were used because the configured base mirror lacked this older version. A synthetic native CAGRA smoke passed on both V GPUs (recall@10 0.94375/0.93125 against exact on that synthetic set); the project backend smoke on V GPU0 built two 4096×128 indexes and found 32/32 self-hits. This **does not** establish retrieval quality for real Qwen queries. A compatible `smg` Gateway was built from this checkout on V; installing Ubuntu's matching `libprotobuf-dev` was needed for `google/protobuf/*.proto` during compilation.

Run roles with `bash test/registered/disaggregation/cloudlab_pvd_new_lease.sh {v|p|d|gateway}` from each role's validation worktree. Start V, P, D, then Gateway, checking `http://10.10.1.2:9100/health`, P/D `/health`, and Gateway `/v1/models`. On D, `PVD_MODE=predictive` is the default; `PVD_MODE=full` is the full-KV control. On V, `PVD_PROMPT_INDEX_EXACT_MAX_ROWS` defaults to 512 for this V100S launcher and can be set to 64 to reproduce the first CAGRA experiment. The launcher refuses a mismatched checkout HEAD, inactive rail, occupied port, or incomplete local model. Treat its P/D/V/Gateway process groups as experiment-owned; inspect the exact PID/PGID before stopping them. Logs are under each node's `validation/logs/` and are not in Git.

## 实测 / Observations

All requests below were sent through the PVD Gateway with `run_pvd_live_load.py`, greedy decoding, `ignore_eos`, and 20 output tokens unless stated otherwise. The replay seed fixes request text across modes; timings are wall-clock SSE observations, not isolated kernel timings. Modes were switched by restarting D, and V was restarted for the exact-threshold comparison. Warmth/Entry retention therefore limit causal claims.

| Prompt / workload | V index and D mode | Observed result |
| --- | --- | --- |
| 50 tokens, 8 output, first ever request | exact, predictive | Complete P→V→D, refresh and `triton_grouped`; 67.12 s with cold compilation, not a steady-state figure |
| 374 tokens, 1 client × 2 rounds | CAGRA (threshold 64), predictive | 18.26 / 17.66 s; max token gap 15.38 / 14.79 s; same output hash in both rounds |
| Same input | exact (threshold 512), predictive | 3.58 / 2.91 s; same output hashes as the CAGRA prediction run; V's per-rank index total ~0.013 s |
| Same input | full Prompt KV, D restarted | 4.04 / 5.40 s; restart-adjacent and **not a warmed fair superiority claim**; output hash differs from sparse prediction |
| 374 tokens, 2 clients × 2 rounds | exact (threshold 512), predictive | 5.92 / 6.12 s wall per round; all 4 requests complete |
| Same concurrent input | full Prompt KV | 1.85 / 1.70 s wall per round; all 4 requests complete; this warmed comparison still favors full KV |
| 915 tokens, 1 client | native CAGRA (threshold 512), predictive | 22.67 s, max token gap 18.63 s; both V ranks logged `path=cagra` |
| Same 915-token input | full Prompt KV | 2.18 s; output hash matched the predictive run for this one input only |
| 374 tokens, 3 sequential requests | exact, predictive, 512 MiB private target-probe prefix cache | 3.68 / 2.99 / 3.21 s; all complete, no D quarantine or cleanup exception. No measured cache speedup claim. |

For a 915-token Entry, V logged 56 head indexes per rank. Building them took 10.68 s on rank 0 and 7.69 s on rank 1 in one run; a subsequent full-KV request also triggered a native build because V indexing is configured independently of D mode. At 374 tokens with threshold 512, each V rank built the exact index in ~0.013 s. This is strong evidence that **per-Entry, per-head native CAGRA construction is the dominant first-refresh stall** at these sizes. It is not evidence that Mooncake/RDMA transfer is slow: initial full-KV fan-in of 374 tokens completed in roughly 16 ms per V rank in the measured logs.

The V health endpoint reported both shards ready, transport healthy, no unknown in-flight transfers, no quarantine, and a healthy maintenance reaper after these requests. Its stored Entry/page counts were nonzero due to the configured 300-second TTL; a snapshot alone cannot distinguish useful reuse from pressure. The D cache-on run removed the earlier *second-request cleanup crash* as an observed failure on this workload, but does not prove all cancellation, eviction or concurrent cache cases.

### 374 行 grouped-exact 复验 / 374-row grouped-exact retest

`PVD_GROUPED_EXACT_SEARCH=1` 在原代码里仍把超过 128 行的索引退回逐项检索；
374-token 请求的 V 日志实际为 `path=individual`，24/32 项批次的
`manager_total` 常见约 60–140 ms。将已存在的有预算 grouped kernel 上限
扩至 512 行后，V100S 上 374/512 行 Top-K 与逐项路径一致，GPU 实际
PyTorch 峰值未超过声明的 scratch 上限；相关索引/CAGRA-auto 回归
**120 passed, 1 skipped**。在线日志确认 `path=grouped_exact`，多次
`manager_total` 约 9–50 ms，V 双 rank 健康、无隔离/未知传输。

相同 replay 输入下，扩展前的分组开关（实际 individual）单客户端两轮
**2.81/3.03 s**，扩展后 **3.12/2.91 s**；两客户端每轮墙钟扩展前
**4.97/5.25 s**，扩展后 **5.54/5.13 s**。四个并发输出哈希及 SSE
完整性均保持。样本太少且受重启与竞争影响，**不能宣称端到端提速**；
这说明只缩短 V 算子没有解决 D 的 probe、HTTP、交付和边界等待。
开关仍默认关闭；没有把 512 行分组路径宣称为生产默认值。

The existing `PVD_GROUPED_EXACT_SEARCH=1` still fell back to individual
search above 128 rows: the 374-token logs actually said `path=individual`.
Extending the budgeted grouped kernel to 512 rows passed V100S Top-K and
PyTorch peak-allocation checks at 374/512 rows, plus **120 passing and one
skipped** index/CAGRA-auto tests. Online logs then showed
`path=grouped_exact`, with many V manager batches at roughly 9–50 ms
instead of the prior 60–140 ms. Both V ranks remained healthy.

With matching replay inputs, one-client rounds changed from **2.81/3.03 s**
to **3.12/2.91 s**; two-client wall rounds changed from **4.97/5.25 s** to
**5.54/5.13 s**. All four concurrent requests kept complete SSE streams and
their previous output hashes. These small, restart-separated samples do
**not** show an end-to-end speedup. Grouped search remains opt-in while D
probe, HTTP, delivery and boundary waits dominate the full path.

### M=8 / lead=6 探索 / Longer prefetch-window experiment

独立的 `pvd_qwen_v100s_serving_limits_triton_m8_lead6_long.json` 只将
Triton 长上下文配置的 `lead_tokens` 从 2 改为 6，并须配合
`PVD_REFRESH_INTERVAL=8`、`PVD_DRAFT_PREDICT_TOKENS=6`。配置加载测试
验证 M=4 会被拒绝；同一 P/V/Gateway 和 374-token replay 下，
单请求两轮耗时 **2.76/2.23 s**（M=4 为 **3.12/2.91 s**），
两并发每轮墙钟 **3.85/3.68 s**（M=4 为 **5.54/5.13 s**）。
20/20 SSE 均完整；单请求输出哈希与 M=4 相同，但两并发输出哈希
**不同**。完整 KV 的两并发对照仍为 **1.85/1.70 s**。
较少刷新可解释一部分总时长改善，但六步预测的近似误差和不同输出
未做质量验收，不能把 M=8 配置设为默认或宣称达到最终目标。

An isolated Triton long-context fixture changes only `lead_tokens` from
2 to 6 and requires `PVD_REFRESH_INTERVAL=8` with
`PVD_DRAFT_PREDICT_TOKENS=6`; its loader test rejects the M=4 pairing.
With the same P/V/Gateway and 374-token replay, one-client rounds took
**2.76/2.23 s** versus **3.12/2.91 s** at M=4. Two-client wall rounds
took **3.85/3.68 s** versus **5.54/5.13 s** at M=4. Every stream had all
20 SSE events. The single-client output hashes matched M=4, but the
concurrent output hashes **changed**. Full KV still took **1.85/1.70 s**
for the concurrent workload. Fewer refreshes plausibly reduce overhead;
quality and approximate-retrieval error are not established, so M=8 is
not a new default or evidence of the final performance target.

### 915 行 exact 阈值与压力对照 / 915-row exact threshold and pressure control

保持 P、V、Gateway、Qwen2.5-7B 及请求生成方式不变，将 V 的实验性
`PVD_PROMPT_INDEX_EXACT_MAX_ROWS` 从 512 提至 2048。915 行 Entry 的
56 个 head index / rank 改走 exact 路径，每个 rank 的构建约
**0.02–0.05 s**，避开先前 native CAGRA 的 **7.69–10.68 s** 冷构建。
这只是此规模下的阈值实验：exact 搜索随行数增长，不能据此全局替代
CAGRA。

同一 915-token、20 输出 replay，M=4 预测单请求 **4.54/4.01 s**，
输出哈希与先前 CAGRA M=4 和完整 KV 的该输入相同；先前 native CAGRA
为 **22.67 s**。M=8/lead=6 的两轮为 **3.13/3.01 s**，但哈希不同，
因此仍是未经质量验收的探索。M=4 两客户端一轮为 **7.32 s**；随后
连续三轮相同的两客户端压力测试每轮 **7.71/7.58/7.40 s**，六个请求
均完整。V 的 `coordinator_pressure_evictions=5`，两 rank 健康、无未知
传输或隔离，说明该有限负载下回收路径运行，但不是长期稳定性证明。

随后只将 D 切换为完整 Prompt KV，三轮相同两客户端、相同 seed 的
墙钟为 **2.093/1.955/1.938 s**。在这些重复性 prompt 上，输入及
输出 SHA256 与预测 exact 路径匹配；预测路径仍约慢 **3.7 倍**。
比较跨 D 重启，样本量小，不能推广到其他负载或证明通用输出质量。
它足以说明：消掉 CAGRA 冷构建后，现有 D probe、检索请求、交付和
刷新边界仍未达到“网络如本地”的目标。开关默认值保持 512。

With V's experimental exact threshold raised from 512 to 2048, each
rank built the 915-row Entry's 56 head indexes in roughly **0.02–0.05 s**,
avoiding the earlier **7.69–10.68 s** native CAGRA cold build. At M=4,
one-client 20-token runs took **4.54/4.01 s** with the same output hash as
the earlier CAGRA M=4 and full-KV run on that input. M=8/lead=6 took
**3.13/3.01 s** but changed the output hash and remains exploratory.
Three two-client M=4 pressure rounds took **7.71/7.58/7.40 s**; all six
streams completed. V reported five pressure evictions and no unknown
transfers or quarantine. Switching only D to full KV gave
**2.093/1.955/1.938 s** for the matching three rounds, with matching
input/output hashes on these repetitive prompts. Sparse PVD was still
about **3.7× slower**. Restart effects, small sample size and repetitive
input limit this comparison; it neither establishes general quality nor
justifies changing the default threshold. Exact search also scales with
Entry length, so this is not a general replacement for CAGRA.

### 真实目标 Q 的 CAGRA 召回 / Real target-Q CAGRA recall

在 D 节点空闲的 V100S GPU0 上，使用独立的 cuVS 25.02 虚拟环境和已有的
Qwen2.5-7B-Instruct FP16 checkpoint，运行更新后的
`run_pvd_qwen_cagra_recall_gpu.py`。测试先从本地 tokenizer 编码的
自然语言段落构造 1024/2048-token Prompt，经目标模型真实 forward
生成 post-RoPE K；使用该 forward 的 greedy 下一 token，通过独立目标
probe 捕获位置 1024/2048 的 post-RoPE Q。覆盖 layer 0/16、Q head
0/1/7/8（对应 KV head 0/1），每种长度 8 个查询，以 GPU exact
点积 Top-10 为 oracle。两种长度均为 **8/8 查询 recall@10=1.0**，
最大分数绝对误差 **0.000244140625**，私有 probe 预算归零。
2048 行第一次被脚本原来的 256 MiB 测试预算正确拒绝；将这个独立
测试预算增至 512 MiB 后通过。V/D 正在运行的服务预算未变。

On idle D GPU0, an isolated pinned cuVS 25.02 environment loaded the
existing FP16 Qwen2.5-7B checkpoint. The updated acceptance script used
tokenized natural-language passages, actual target Prompt K, the target
model's greedy next token, and a separate post-RoPE target-Q probe. For both
1024 and 2048 rows, all eight sampled layer/GQA-head queries had
**recall@10=1.0** against GPU exact search; maximum score error was
**0.000244140625**, with the private probe budget refunded. The 2048-row
run initially hit the script's own 256 MiB budget; it passed after raising
that independent test bound to 512 MiB. This is a **small, repeated-text
single-prefix sample** on one GPU, not an end-to-end sparse-attention quality
distribution, a CAGRA recall guarantee on diverse prompts, or a performance
result. The script's `performance_validated` flag remains false.

```bash
# On the D node's isolated validation checkout, after sourcing the normal
# Conda environment and exposing the pinned cuVS venv's package/library paths:
CUDA_VISIBLE_DEVICES=0 PVD_CAGRA_RECALL_ROWS=2048 \
  python test/registered/disaggregation/run_pvd_qwen_cagra_recall_gpu.py \
  --architecture qwen2 \
  --model-path /users/Yizhzhu/pvd-models/Qwen2.5-7B-Instruct \
  --context-length 2304 --max-total-tokens 2304
```

### 916 行 grouped exact / 916-row grouped exact

进一步将仍为 opt-in 的 grouped exact 上限从 512 行扩至 2048 行；
完整堆叠 K、分数和排序 scratch 在运行前按实际行数计入预算，超过 2048
行仍回退到独立的分块 exact/CAGRA 路径。V100S 上针对 915/2048 行、
32 个索引和 7 个查询行的 Top-K/峰值预算回归通过；索引/CAGRA-auto
套件 **124 passed, 1 skipped**。在线 V 双 rank 的 916 行日志实际为
`path=grouped_exact`，常见 manager 阶段约 **5–48 ms**，无传输隔离。

固定 `--replay-seed grouped2048a`、默认负载句子重复 100 次、
2 客户端 × 2 轮、每请求 20 输出，墙钟 **7.34/8.06 s**，四个 SSE
完整。该 seed 未在旧逐项路径上重复，故这些数字**不是严格 A/B**；
先前相近长度的逐项路径约 **7.4–7.7 s**。D 日志中每次刷新目标
Q capture 仍约 **0.39 s**，search 总阶段约 **0.39–0.82 s**，其中
单个 V HTTP 批次约 **24–133 ms**、V manager 通常远低于总 search。
因此更大的 grouped kernel 没有建立端到端收益；开关继续默认关闭。

The opt-in grouped exact cap now admits up to 2048 rows and precharges
the entire stacked-K, score and sort footprint; larger indexes retain the
separate bounded path. V100S Top-K/peak-budget checks at 915/2048 rows passed
with **124 passing, one skipped** index/CAGRA-auto tests. Online V logs
confirmed `path=grouped_exact` for a 916-row Entry, generally spending
**5–48 ms** in the manager stage. Two clients over two `grouped2048a`
replay rounds completed all SSE streams in **7.34/8.06 s** wall-clock.
Because the previous individual-path runs used a different seed, this is
**not a controlled A/B** and cannot establish an end-to-end speedup. D's
target-Q capture remained about **0.39 s** per refresh and the whole search
stage **0.39–0.82 s**. The grouped switch remains off by default.

## 未完成 / Remaining work

1. **Performance:** predictive sparse refresh still loses to warmed full KV under two-client load even after avoiding native CAGRA cold build. Determine a measured admission strategy (exact first, background CAGRA promotion only when the Entry is likely to be reused) and preserve index/version/retirement fencing before implementing it. The 2048-row exact threshold is an experiment, not a new default.
2. **Quality:** the new real-Qwen target-Q test covers only one repeated-text prefix, two layers and eight GQA-head queries per length. It is not a broad recall distribution or generated-answer quality evidence. Compare generated outputs on varied, non-repetitive prompts and more query positions/layers before claiming quality.
3. **Scale:** this experiment is P TP1, V 2 ranks, D TP1, single rail, one or two clients, max 2304 sequence tokens. TP asymmetry, dual-rail, long-running load, TTL pressure, multi-D routing, and full GPU-memory safety have not been established here.
4. **Cache:** compare private probe prefix cache on/off with matched warmups, unique prompts and budget snapshots; exercise concurrent close, cancellation and retraction. The three-request run establishes only the narrow cleanup regression.
