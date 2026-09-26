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

## 未完成 / Remaining work

1. **Performance:** predictive sparse refresh still loses to warmed full KV under two-client load; native CAGRA cold build is much slower. Determine a measured admission strategy (exact first, background CAGRA promotion only when the Entry is likely to be reused) and preserve index/version/retirement fencing before implementing it. A larger exact threshold may be appropriate within the 2304-token experiment context, but must be measured rather than assumed.
2. **Quality:** synthetic CAGRA recall and output hashes on repetitive prompts are not real-query recall or task-quality evidence. Measure Qwen query recall against exact, and compare generated outputs under varied prompts.
3. **Scale:** this experiment is P TP1, V 2 ranks, D TP1, single rail, one or two clients, max 2304 sequence tokens. TP asymmetry, dual-rail, long-running load, TTL pressure, multi-D routing, and full GPU-memory safety have not been established here.
4. **Cache:** compare private probe prefix cache on/off with matched warmups, unique prompts and budget snapshots; exercise concurrent close, cancellation and retraction. The three-request run establishes only the narrow cleanup regression.
