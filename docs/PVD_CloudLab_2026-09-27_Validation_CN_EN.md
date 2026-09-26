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

The later real-Qwen recall gate installed the same pinned cuVS wheels only in
an isolated D-side venv, leaving the serving Conda environment unchanged.
After the experiments, all experiment-owned P/V/D/Gateway process groups
were stopped. A final three-node check found no remaining GPU compute
processes or PVD listeners on ports 30002/30003/9100/9300/9301/8001;
model files, isolated environments, validation worktrees and logs were kept.

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

### 私有目标-probe 前缀缓存 A/B / Private target-probe prefix-cache A/B

在上述 `grouped2048a` 的 P/V/Gateway、M=4、Top-4、2 客户端 × 2 轮
负载中，仅将 D 的 `probe_prefix_cache_bytes` 从 0 改为 512 MiB。
新配置文件为
`pvd_qwen_v100s_serving_limits_triton_m4_probe_cache_long.json`；
加载测试确认它与原 Triton 长配置只差这个显式预算（2 个定向测试通过）。
使用同一 replay seed，**输入和四个输出的 SHA256 均完全一致**，所有
SSE 完整。无缓存每轮墙钟 **7.34/8.06 s**；有缓存为 **6.00/7.09 s**。
D 日志中典型预测 capture 从约 **0.39 s** 降到 **0.215 s**，但也有
0.42–0.57 s 的冷/未命中阶段；search 和 delivery 仍波动。V 健康。
跨 D 重启、样本仅两轮，不能把此差值当成稳定吞吐收益，也不能
默认为开启；完整 KV 对照仍约 **2 s** 每两客户端轮，差距显著。

For the same `grouped2048a` replay, only D's private target-probe
`probe_prefix_cache_bytes` changed from zero to **512 MiB**. All four
input/output hashes matched the no-cache run and every SSE stream completed.
Two-client wall rounds moved from **7.34/8.06 s** to **6.00/7.09 s**.
Typical D capture stages fell from roughly **0.39 s** to **0.215 s**, with
some cold/miss stages at 0.42–0.57 s. The change crossed a D restart and
only two rounds, so it is evidence of a useful mechanism, not a stable
throughput estimate or a default-on decision. The full-KV control remains
around **2 s** per two-client round on this workload.

### 批量检索背压 / Search-batch backpressure

尝试把单 V rank 的有版本 pin 检索批次从 32 提到 64 项，以便将 Qwen
每 rank 56 个 layer/KV-head 搜索压成一个 HTTP 批次。单元和 V100S
峰值测试覆盖 56 个索引、915/2048 行及有界 64 项协议。在默认 1 GiB
V 索引预算下直接启用 64 项**失败**：第二轮出现大量 HTTP 507
`index_capacity`，原客户端原样重试导致 180 秒超时。原因可由预算
构成解释：CAGRA-auto 的全局 native 额度先占约 640 MiB，余量须同时
容纳多个 Entry 的索引副本与 56 项整批 scratch；并发时可能不足。

新增容量拒绝时的有界、顺序拆批回退：仅对 `index_capacity` 拆半，
必要时单项调用原接口；每个子批仍检查原 Entry 与版本 pin，全部成功
前不发布部分结果。包含单项回退的测试及相关回归在 V100S 上
**377 passed, 1 skipped**。1 GiB/64 项在线复测四个 SSE 完整，但
仍发生 **44 次 507**，两轮 **6.05/6.32 s**，所以调度默认值恢复
**32 项**，64 项需要显式 `PVD_SEARCH_BATCH_MAX_ITEMS=64`。

V 索引预算改为有界可配置，启动脚本默认仍 **1 GiB**；在这两张 V100S
当前可用显存约 31 GiB/卡的条件下，显式
`PVD_PROMPT_INDEX_BUDGET_BYTES=2147483648` 与 64 项配合，
同一 `grouped2048a`、M=4、512 MiB 私有 probe 缓存的两轮为
**6.08/5.35 s**，V 日志确认 55/56 项 grouped 请求，**0 次 507**，
双 rank 健康。四个输出哈希均与 32 项和完整 KV 路径一致。
只切换 D 为完整 KV、保持 P/V/Gateway 的同 seed 对照为
**2.06/1.96 s**。64 项 + 2 GiB 在第二轮较快、第一轮不快；样本太少，
不能宣称稳定收益，更没有达到完整 KV 性能。2 GiB 不设为默认。

The attempt to send all 56 Qwen layer/KV-head searches per V rank in one
version-pinned HTTP batch exposed a real budget interaction. At the default
**1 GiB** index budget, 64-item batches incurred repeated HTTP 507
`index_capacity`; the first unmodified retry loop timed out at 180 seconds.
The CAGRA-auto shared native reservation consumes roughly 640 MiB before
Entry copies and batch scratch are charged. A new bounded, sequential split
on capacity refusal preserves all version pins and publishes no partial
result. Related tests passed **377/377** with one skip. The 1 GiB online
retry completed four streams but logged **44 capacity refusals** and took
**6.05/6.32 s** for two-client rounds. The scheduler therefore defaults to
**32 items**; 64 is explicit opt-in only.

The isolated launcher now permits a bounded 1–4 GiB V index budget while
defaulting to 1 GiB. With an explicitly selected **2 GiB** budget and
`PVD_SEARCH_BATCH_MAX_ITEMS=64`, the same `grouped2048a` replay completed
in **6.08/5.35 s**, with real 55/56-item grouped batches, **zero 507s**,
healthy V ranks and unchanged hashes. The matching full-KV D control took
**2.06/1.96 s**. One round was faster than the 32-item cached run and one
was not; this is neither a stable throughput claim nor the final latency
target. Neither 64 items nor 2 GiB becomes a default.

### 首批检索版本发现 A/B / First-batch version discovery A/B

本轮在本地 commit `40f77b5ca` 与 `657568e97` 增加 V 侧同一读租约内的
无版本 pin 批量检索，并让 D 的首次刷新直接对每个 V shard 并行发整批查询。
V 回包的所有结果必须属于同一 index/mapping 版本；混合版本、混合 pin、
取消与错误回包均拒绝发布。后续轮次继续显式 pin。CloudLab 隔离环境的
搜索/索引回归 **255 passed**，Ruff 通过。两次提交均仅在本地，未推送。

固定 `--replay-seed firstbatch-ab-20260927`、921-token prompt、2 客户端
× 3 轮、每请求 20 token，保持 P/V/Gateway 运行，只重启 D 并替换隔离
验证目录的 `search_client.py`、`probe_search.py`。V 使用 2048 行 exact
阈值、2 GiB index 预算与 grouped exact，D 使用 M=4、512 MiB 私有
target-probe 缓存与 64 项 batch。旧单条 seed 再批量的每轮墙钟为
**6.34/5.06/5.12 s**；新无 pin 首批整批为 **5.63/5.06/5.00 s**。
六个请求的输入和输出哈希逐一相同，120/120 SSE token 完整。D 日志确认
draft、目标 Q probe 和刷新边界安装；V 日志确认 HTTP batch 200。
只有第一轮有约 0.71 s 的观察差值；暖轮差异很小，样本量也不足以作
稳定性能结论，更没有超过完整 KV 基线。实验的 `mode_verified_by_script`
字段为 false，模式由 D 启动参数和日志另行核对。

The local commits `40f77b5ca` and `657568e97` let an initial unpinned batch
discover one V index/mapping version under a single reader lease, removing the
separate seed-search round trip. D sends first batches to different V shards
concurrently and continues to pin subsequent rounds. Mixed pins or versions,
cancellation and malformed replies fail closed. Search/index tests passed
**255/255** on CloudLab; Ruff passed. Neither commit was pushed to GitHub.

With the same fixed seed, 921-token prompts, two clients, three rounds and
20 tokens/request, P/V/Gateway stayed up while D alone was restarted. The
old path took **6.34/5.06/5.12 s** wall time per round; the new path took
**5.63/5.06/5.00 s**. All six input/output hashes matched and all 120 SSE
tokens arrived. D confirmed draft/probe/boundary activity; V returned batch
HTTP 200. Only the first round shows a notable observed improvement (~0.71 s);
warm rounds are essentially unchanged and this is not evidence of a stable
speedup over full-KV Decode.

### M=4 预取提前量 2→3 / M=4 prefetch lead 2→3

为检验一个额外生成 token 是否能遮蔽搜索和 RDMA，新增独立实验配置
`pvd_qwen_v100s_serving_limits_triton_m4_lead3_probe_cache_long.json`：
与上述 lead=2 配置只差 `lead_tokens=3`，启动时配套
`PVD_DRAFT_PREDICT_TOKENS=3`；配置差异与约束测试 **2 passed**。
同一新搜索代码、P/V/Gateway、固定输入和 2 客户端 × 3 轮，lead=3
墙钟为 **5.75/5.44/5.11 s**，对照 lead=2 的 **5.63/5.06/5.00 s**。
六个输出哈希均相同、120/120 SSE 完整。此负结果不支持把 lead=3
设为默认；它不证明所有 prompt、并发度或网络条件下 lead=2 最优。

The lead=3 configuration changes only the prefetch lead from the cached
M=4/lead=2 setup and pairs it with three draft tokens; its two targeted
configuration tests passed. Under the same fixed seed and live P/V/Gateway,
three two-client rounds took **5.75/5.44/5.11 s** versus
**5.63/5.06/5.00 s** with lead=2. All six output hashes and 120 SSE tokens
matched. This narrow negative result does not justify a default change.

## 未完成 / Remaining work

1. **Performance:** predictive sparse refresh still loses to warmed full KV under two-client load even after avoiding native CAGRA cold build. Determine a measured admission strategy (exact first, background CAGRA promotion only when the Entry is likely to be reused) and preserve index/version/retirement fencing before implementing it. The 2048-row exact threshold is an experiment, not a new default.
2. **Quality:** the new real-Qwen target-Q test covers only one repeated-text prefix, two layers and eight GQA-head queries per length. It is not a broad recall distribution or generated-answer quality evidence. Compare generated outputs on varied, non-repetitive prompts and more query positions/layers before claiming quality.
3. **Scale:** this experiment is P TP1, V 2 ranks, D TP1, single rail, one or two clients, max 2304 sequence tokens. TP asymmetry, dual-rail, long-running load, TTL pressure, multi-D routing, and full GPU-memory safety have not been established here.
4. **Cache:** the two-round same-seed A/B indicates lower capture time without changing these four outputs; repeat with matched warmups, more varied prompts and budget snapshots, then exercise concurrent close, cancellation and retraction. The earlier three-request run established only the narrow cleanup regression.
