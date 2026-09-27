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

Run roles with `bash test/registered/disaggregation/cloudlab_pvd_new_lease.sh {v|p|d|gateway}` from each role's validation worktree. Start V, P, D, then Gateway, checking `http://10.10.1.2:9100/health`, P/D `/health`, and Gateway `/v1/models`. On D, `PVD_MODE=predictive` is the default; `PVD_MODE=full` is the full-KV control. On V, `PVD_PROMPT_INDEX_EXACT_MAX_ROWS` now defaults to 2304 for this **2304-context V100S launcher only**; set it to 64 to reproduce the first CAGRA experiment, or 2048 to exercise CAGRA on a 2095-row Entry. The launcher refuses a mismatched checkout HEAD, inactive rail, occupied port, or incomplete local model. Treat its P/D/V/Gateway process groups as experiment-owned; inspect the exact PID/PGID before stopping them. Logs are under each node's `validation/logs/` and are not in Git.

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

### 当前完整 KV 控制与刷新关键路径 / Current full-KV control and refresh critical path

上述 `firstbatch-ab-20260927` 的相同固定 seed、921-token prompt、双客户端
× 3 轮，保持 P/V/Gateway 不变并仅把 D 切为完整 KV，墙钟为
**2.11/2.00/2.01 s**；6 个输入/输出哈希与预测 PVD 均相同，120/120
SSE 完整。新预测路径 lead=2 的 **5.63/5.06/5.00 s** 仍约为完整 KV
的 2.5 倍；当前未证明端到端收益。

用新预测路径打开 D 时间线，仅观察一轮同输入负载：8 条已记录的
`PVD refresh ready` 中，典型同步 draft+目标 Q 捕获约 **0.216 s**，
搜索等待约 **0.20–0.44 s**，稀疏交付约 **0.07–0.10 s**；最长总刷新
约 **0.91 s**。D 的 56 项 batch 各含 392 行 Q，JSON 请求约
**0.7 MiB/shard**，部分底层 HTTP 为约 0.07–0.12 s；这些阶段的
并发/排队使它们不能简单相加成纯网络 RTT。完整 KV 控制的最大 SSE
token 间隔约 0.17 s，而预测路径约 0.73 s。优先工作应是避免每轮
同步捕获阻塞正式 Decode，并将 V 检索、选择、打包和发送合成更短的
关键路径；不能以单纯少一个首次 seed RPC 代替这两项。

For the exact same seed and six 921-token/twenty-output requests, a
full-KV D control with P/V/Gateway unchanged took **2.11/2.00/2.01 s** per
two-client round, versus **5.63/5.06/5.00 s** for the new lead=2 predictive
path. All six input/output hashes matched and all 120 SSE tokens arrived.
Predictive PVD is still about 2.5x slower. One instrumented predictive round
showed typical synchronous draft+target-Q capture near **0.216 s**, search
wait **0.20–0.44 s**, and sparse delivery **0.07–0.10 s** per refresh; the
longest recorded total refresh was **0.91 s**. A 56-item search batch held
392 Q rows and about **0.7 MiB** of JSON per shard. Queueing and concurrency
make those numbers different from pure network RTT. The next critical-path
work is to keep Q capture off committed Decode's execution path and collapse
V search/selection/pack/send, not only remove first-window discovery RPCs.

### V grouped exact Top-4 归约试验 / V grouped exact Top-4 reduction experiment

新增显式 opt-in `PVD_GROUPED_EXACT_TOPK_REDUCE=1`：在已有 grouped GEMM
的分数张量上，对 Top-K≤8 重复取最大值并屏蔽获胜 token，不再稳定全排序
全部 Prompt 行；`torch.max(dim=-1)` 在同分时选最低行号，Top-K>8
及默认路径仍使用稳定排序。CPU 的随机/全同分测试、V100S 的
56×915×7×128 数值与峰值显存测试均通过。独立 V100S 微基准
（20 次交替顺序、每次 CUDA 同步）在 915 行、Top-4 下稳定排序
中位 **2.47 ms**，归约 **1.08 ms**，结果逐元素相同。
这只节省约 **1.39 ms/批**；D 端单次搜索等待常为数百毫秒，不能
据此推断端到端收益或开启默认开关。尚需三机对照与更多同分/模型负载。

The opt-in `PVD_GROUPED_EXACT_TOPK_REDUCE=1` repeatedly reduces only the
requested Top-K winners from the existing grouped score tensor. The first
row wins exact ties; Top-K>8 and the default retain stable full sort.
CPU random/all-tie tests and a V100S 56×915×7×128 equality/peak-memory test
passed. A synchronized, alternating-order 20-run V100S microbenchmark at
Top-4 measured **2.47 ms** median for stable sort versus **1.08 ms** for
the reduction, with identical outputs. Saving about **1.39 ms/batch** is
not an end-to-end speedup claim; the switch remains off by default.

### V 查询数组校验与零拷贝 CPU tensor / V query validation and zero-copy CPU tensor

V 控制面把有界 JSON Q 行先转成 NumPy float32 数组、整批检查有限值，
工作线程再用 `torch.from_numpy` 建立 CPU tensor 视图，代替旧的逐标量
`math.isfinite`/界限检查及第二次 `torch.tensor` 复制。bool、字符串、
嵌套列表、None、NaN、Inf 和极大整数仍在后端前拒绝；HTTP/索引回归
**159 passed**，Ruff 通过。同一 V100S 配置、56 项/392 Q 行的真实
V 日志中，请求校验由约 **19 ms** 降至 **6.5–6.8 ms**，tensor 构造
由约 **3–4 ms** 降至 **0.1–0.2 ms**。保持 Top-K 归约 opt-in 开启，
同 seed 三机每轮墙钟由 **6.12/5.25/5.14 s** 至
**6.05/5.14/5.13 s**，六个哈希与 120 SSE 均一致。三轮小样本不能
证明端到端稳定收益，仍远慢于完整 KV。

V now converts bounded JSON Q rows to one float32 NumPy array, checks
finiteness in bulk, and creates a zero-copy CPU tensor view in the worker.
Malformed booleans, strings, nested rows, nulls, NaN/Inf and enormous ints
still fail before backend execution. Search/index HTTP tests passed
**159/159** and Ruff passed. On real V100S 56-item/392-row batches,
request validation fell from about **19 ms** to **6.5–6.8 ms** and tensor
construction from **3–4 ms** to **0.1–0.2 ms**. With the Top-K opt-in still
enabled, three same-seed two-client rounds were **6.05/5.14/5.13 s**
versus **6.12/5.25/5.14 s** before this change, with all hashes/SSE intact.
This is a measured V control-path improvement, not an established end-to-end
win or evidence of the final latency target.

### 有界 float32 Q 批量传输 / Bounded packed-float32 Q batches

新增默认关闭的 `PVD_PACKED_QUERY_BATCH=1`，仅把 D→V search-batch 中的
Q 行改成 little-endian float32 + base64；身份、版本 pin、逻辑 token/page
结果与后续 Mooncake KV 传输不变。V 在索引调用前验证行数、维度、
编码长度、实际字节数与有限值，并拒绝同时出现 packed/JSON Q。
旧 JSON 客户端仍可用。搜索/索引回归 **245 passed**，Ruff E/F/I 通过。

同一 P/V/Gateway，固定 `firstbatch-ab-20260927`、921-token Prompt、
双客户端 × 3 轮、每请求 20 token，仅重启 D 切换 opt-in：JSON 墙钟
**6.06/5.18/5.12 s**，packed **5.67/4.99/4.96 s**。
24/32 项 V-shard batch 的编码请求从约 **311–419 kB** 降到
**126–168 kB**（均按日志中的十进制字节计）。
六个输入/输出哈希逐一相同，120/120 SSE 完整。D 端打包准备约多
1–2 ms，而编码约少 0.7–0.9 ms；网络/排队时间波动明显。不能将三轮
差值归因为稳定链路提速，更不能宣称整体超过约 2 秒的完整 KV 控制。

The opt-in `PVD_PACKED_QUERY_BATCH=1` changes only the search-batch Q rows
to little-endian float32 encoded as base64. V validates shape, encoded and
decoded lengths, finiteness, and exclusive representation before index use.
Identity, version fencing, logical selection and Mooncake KV delivery are
unchanged; legacy JSON requests remain accepted. Search/index regressions
passed **245/245**, with Ruff E/F/I clean. On the same live P/V/Gateway and
fixed two-client, three-round, 921-prompt-token replay, JSON took
**6.06/5.18/5.12 s** and packed Q **5.67/4.99/4.96 s** per round. Encoded
24/32-item shard requests shrank from roughly **311–419 kB** to
**126–168 kB**. All six hashes and 120 SSE events matched. The small,
restart-separated sample and queueing variance do not establish a stable
end-to-end win; full-KV Decode remains much faster.

### 事实检索型 Prompt 的答案与延迟对照 / Fact-recall prompt A/B

新增 `run_pvd_fact_recall.py`：固定 seed 生成 48 条带唯一五位代码的事实，
每个 case 查询随机一条；只记录输入/输出哈希和生成文本中**第一个**五位
代码是否正确，不保存原始生成文本。生成器和汇总的 5 项本地逻辑测试通过，
Ruff E/F/I 通过。它是结构化事实检索负载，不能代表开放式回答质量。

三机保持相同 P/V/Gateway、Qwen2.5-7B、`quality_20260927` seed，
6 个不同 case、约 1413–1425 token Prompt、20 token 输出；D 分别以
M=4/Top-4 packed-Q 稀疏预测和完整 KV 启动。两模式的第一个五位代码都
**6/6 正确**；两模式各重跑一次，模式内六个输出哈希逐一稳定，模式间
**0/6 完整输出哈希相同**。因此短重复输入的哈希一致不能推广到本负载；
这也不等于目标代码答案错误。稀疏两次中位请求耗时约 **6.14/6.10 s**，
完整 KV 两次约 **2.37/2.13 s**；脚本每轮最多两个并发请求，但没有
同步起跑屏障，因此这些数值是有限样本的诊断，不是严格吞吐基准。
当前稀疏路径仍显著慢于完整 KV。

The deterministic fact-recall probe asks for one of 48 unique five-digit
record codes in each of six distinct ~1413–1425-token prompts, with 20
generated tokens. It checks the **first** five-digit code in the answer,
records hashes but not raw generated text, and has five passing logic tests
plus Ruff E/F/I. With P/V/Gateway fixed, sparse predictive M4/Top-4 and
full-KV D each answered **6/6** codes correctly. Two runs per mode had
identical hashes within that mode, but **none of the six full generated
outputs matched across modes**. This shows that matching hashes on repeated
short prompts are not a general quality guarantee; the requested fact still
matched in this narrow task. Median request latencies were about
**6.14/6.10 s** for sparse versus **2.37/2.13 s** for full KV. The probe
uses at most two concurrent clients without a start barrier, so these are
diagnostic timings, not a throughput claim.

### Top-K 质量—成本实验 / Top-K quality-cost experiment

CloudLab D 启动器新增有界环境参数 `PVD_RETRIEVAL_TOP_K`（1–16）和
`PVD_RETRIEVAL_UNION_TOKENS`（Top-K–128），默认仍为 4/32；
`bash -n` 通过。保持同一 P/V/Gateway、事实型 seed、packed Q、M=4
与其余参数，只把 D 设为 Top-8/union-64。两轮均为第一个代码 **6/6
正确**，模式内输出哈希稳定，但相对完整 KV **0/6** 哈希相同，
相对 Top-4 也 **0/6** 相同；中位请求耗时约 **6.28/6.28 s**，
Top-4 约 **6.14/6.10 s**，完整 KV 约 **2.37/2.13 s**。
这个小实验不支持把 Top-8 设为默认，也表明仅增加检索 token
不能保证完整生成文本接近完整 KV。

The launcher now accepts bounded `PVD_RETRIEVAL_TOP_K` (1–16) and
`PVD_RETRIEVAL_UNION_TOKENS` (Top-K–128), defaulting to 4/32; `bash -n`
passed. Changing only D to Top-8/union-64 for the same packed-Q M4 fact
replay answered the first code **6/6** in both runs, with stable hashes
within Top-8. Yet **0/6** full outputs matched full KV or Top-4.
Median request latency was about **6.28/6.28 s**, versus **6.14/6.10 s**
for Top-4 and **2.37/2.13 s** for full KV. This narrow negative result
does not justify a default change; adding retrieved tokens alone does not
guarantee full-output equivalence.

### M=8 + probe 缓存事实负载 / M8 with probe cache on fact prompts

独立配置 `pvd_qwen_v100s_serving_limits_triton_m8_lead6_probe_cache_long.json`
与 M4 缓存配置只差 `lead_tokens=6`，启动时配套刷新间隔 8、draft
预测 6 token；加载约束回归 **55 passed**，Ruff E/F/I 通过。
保持 P/V/Gateway、Top-4、packed Q、seed 与事实负载不变，仅更换 D。
两轮第一个代码仍 **6/6 正确**，模式内输出哈希稳定；与完整 KV 的
完整输出 **1/6** 哈希相同。两轮请求中位耗时约 **5.10/5.16 s**，
比 M4 缓存 **6.14/6.10 s** 低，但仍约为完整 KV **2.37/2.13 s**
的两倍以上。两轮日志的 `PVD refresh ready` 计数从 M4 的 48
降至 M8 的 24（每模式均 12 个请求）。这说明减少刷新次数对该
负载有效，不能证明 M8 的近似质量足以默认启用，也未达到最终性能目标。

The isolated M8/lead-6 fixture differs from cached M4 only in lead tokens;
it pairs with interval 8 and six draft tokens. Its loader regression passed
**55/55** with Ruff E/F/I clean. Holding P/V/Gateway, Top-4, packed Q and
the six fact prompts fixed, both M8 runs answered the first code **6/6**
and produced stable within-mode hashes. **1/6** complete outputs matched
full KV. Median request latency was **5.10/5.16 s**, down from cached M4's
**6.14/6.10 s**, but still over twice full KV's **2.37/2.13 s**.
The two runs logged 24 refreshes in M8 versus 48 in M4 (12 requests each).
Fewer refreshes helped this workload; neither broader answer quality nor
the final latency target is established.

### D 侧前向读租约复用 / D forward bank-reader reuse

新增默认关闭的 `PVD_REUSE_FORWARD_BANK_LEASE=1`：整段模型前向已经为
每个 request 持有 Prompt bank 读租约时，稀疏 attention 每层复用**同一**
live bank/group 对象，不再嵌套进入第二个读租约。仍保留 workspace
每层 CUDA 完成栅栏、前向结束栅栏和 bank 最终读者栅栏；无 live
读者、非当前 group、错误刷新边界一律拒绝。CPU 夹具验证嵌套 bank
栅栏从每层两次读者作用域降为整段前向一次，数值/故障回归
**120 passed**，Ruff E/F/I 通过。

V100S 上相同 M8/Top-4/packed Q 事实负载，两轮首代码均 6/6 正确，
六个输出哈希与未开启复用的 M8 逐一相同，且无观察到 RDMA 错误或
quarantine。但中位请求耗时约 **5.17/5.17 s**，未开启时约
**5.10/5.16 s**；未证明端到端改善，因此不默认开启。
仅删去已无待执行 GPU 工作的嵌套读者栅栏，关键的**每层 workspace
同步**仍在；若要进一步合并同步，必须先让每层的指针表、行索引、
预算、输出与 Prompt 读租约都延迟持有到整段前向的最终 CUDA 栅栏。

The opt-in `PVD_REUSE_FORWARD_BANK_LEASE=1` borrows the exact live Prompt
bank reader already owned by the whole model forward, rather than entering
a nested bank reader per layer. The workspace still synchronizes each layer,
and the forward and bank readers still synchronize on retirement. Invalid
or stale borrowed groups fail closed. CPU numerical/failure tests passed
**120/120** and Ruff E/F/I passed. On V100S, two identical M8 fact runs
answered 6/6 codes with hashes equal to the prior M8 path and no observed
RDMA/quarantine errors. Median latency was **5.17/5.17 s** versus
**5.10/5.16 s** without the opt-in: no established speedup, so it remains
off by default. Removing the more costly per-layer workspace sync requires
forward-scoped ownership of all pending tables, rows, budgets and outputs;
simply deleting that fence would be unsafe.

### D 侧整段 forward 延迟完成栅栏 / Forward-scoped CUDA completion

新增默认关闭的 `PVD_DEFER_LAYER_FENCES=1`，且只允许与
`PVD_REUSE_FORWARD_BANK_LEASE=1`、`triton_grouped` 同时使用。
每层的 Prompt 指针表、生成 token 行索引、输入/输出资源 pin 和预算
保留至整段 forward 的 CUDA 完成栅栏之后，再依次释放；如果栅栏失败，
资源进入 quarantine，不重新发放可能仍被 GPU 使用的地址。独立实验
配置将 scratch 最大预留槽数扩大到 256；旧配置与默认行为不变。
CPU 生命周期/配置测试分别 **123/56 passed**，Ruff E/F/I 通过。

CloudLab V100S 上复用相同 P/V/Gateway、Qwen2.5-7B、M8/Top-4、
packed Q、六个事实 Prompt 和两个客户端；D 确认加载两个 opt-in 环境
变量并实际执行稀疏 attention。两轮首代码均 **6/6 正确**，每条完整
输出哈希与上一版 M8 复用读租约路径一致，D 日志未见 quarantine、
Traceback 或 RDMA remote-access 错误。两轮中位耗时约 **5.09/5.08 s**；
先前仅复用读租约的两轮约 **5.17/5.17 s**，完整 KV 约 **2.37/2.13 s**。
这是小样本、未同步起跑的诊断，不能将约 0.08 秒差异归因于本改动，
更不能称其达到端到端目标；因此仍默认关闭。下一步需 profile 前向
各阶段及重复 target-Q capture，再决定是否扩大使用范围。

The opt-in `PVD_DEFER_LAYER_FENCES=1` is permitted only with the borrowed
forward Prompt reader and grouped Triton attention. Each layer's pointer
tables, generated-row indices, input/output pins and budget charge remain
owned until one forward-scoped CUDA completion fence; an uncertain fence
quarantines the owners instead of recycling their addresses. The isolated
fixture raises scratch reservation slots to 256; defaults are unchanged.
CPU lifecycle/config tests passed **123/56** respectively, with Ruff E/F/I
clean. On V100S, two replays of the same six fact prompts answered **6/6**
first codes and reproduced every complete-output hash from the reader-only
M8 path. The active D process had both opt-ins set and logged real sparse
attention; its log showed no traceback, quarantine or RDMA access error.
Median request latency was **5.09/5.08 s**, versus **5.17/5.17 s** for
reader-only M8 and **2.37/2.13 s** for full KV. This tiny, unsynchronized
sample does not establish a causal speedup or the final end-to-end target,
so deferred fences stay disabled by default.

### 跳过不可达的末尾刷新 / Skip unreachable terminal refresh

实测 M8/20-token 请求中，每个请求只安装边界 8、16 的两轮工作集，
却仍额外启动边界 24 的 draft 预测和目标模型 Q probe。前一版本两轮
共 12 个请求产生 **36** 次预测阶段日志，但只有 **24** 次刷新完成。
P 的首 token 不推进 D 时钟，因此 `max_new_tokens <= next_boundary`
时，最终 D 时钟至多为 `max_new_tokens - 1`，该工作集绝不被 Decode
读取。新调度门槛只对明确的整数生成上限跳过这类刷新；缺失或未知
上限保留原行为，上限在到达边界前变化仍可重新安排。相关驱动测试
**36 passed**，Ruff E/F/I 通过。

相同三机、M8/Top-4、两个客户端、六个事实 Prompt 的两轮中，
每轮首代码 **6/6 正确**，完整输出哈希与改动前逐一一致；两轮共
**24/24** 次预测阶段均对应完成的刷新，不再有 12 次末尾无用
预测。中位请求耗时约 **4.47/4.45 s**，此前延迟栅栏配置约
**5.09/5.08 s**，完整 KV 约 **2.37/2.13 s**。这仍是有限的
非屏障负载，尚未达到端到端性能目标。另以单个 28-token 请求
验证边界 8、16、24 均正常刷新并安装，首代码正确，无观察到
quarantine 或 RDMA 错误。

With a 20-token generation cap, D only consumes refresh boundaries 8 and
16, yet the old scheduler also launched draft and target-Q work for boundary
24. Across 12 requests, it logged **36** prediction stages but only **24**
completed refreshes. Because P's first token never advances D's clock,
`max_new_tokens <= next_boundary` proves no later D token can read that
workset. The new check skips only explicit integer caps; unknown limits
retain the old path, and a changed cap can be reconsidered before the
boundary. Driver tests passed **36/36**, Ruff E/F/I clean. Two same-seed
six-fact V100S replays retained **6/6** correct first codes and identical
complete-output hashes; all **24/24** prediction stages now corresponded to
completed refreshes. Median latency was **4.47/4.45 s**, down from
**5.09/5.08 s** with deferred fences alone, but still slower than full KV's
**2.37/2.13 s**. A separate 28-token request confirmed boundaries 8, 16,
and 24 all refreshed and installed correctly. This is not yet a broad
quality or end-to-end performance acceptance.

### M=16 的延迟与质量取舍 / M16 latency-quality tradeoff

在跳过不可达末尾刷新之后，仅把 D 的刷新间隔从 8 改为 16，仍保留
lead=6、Top-4、packed Q、forward 延迟栅栏以及相同 P/V/Gateway、
模型和六个事实请求。两轮首代码均 **6/6 正确**，各自完整输出哈希
稳定，中位耗时约 **3.64/3.67 s**；M8 约 **4.47/4.45 s**，完整
KV 约 **2.37/2.13 s**。M16 与 M8 的完整输出只有 **1/6** 哈希
相同。现有事实任务不能证明广义答案质量，故此配置仅作隔离实验，
不改默认刷新间隔。专用 M16 配置与 M8 延迟栅栏配置的资源界限完全
相同；选择 interval=16 必须在启动命令中显式指定。

After terminal-refresh skipping, changing only D's interval from 8 to 16
(lead 6, Top-4 and the same packed-Q/deferred-forward settings) yielded
**6/6** correct first codes in each same-seed replay, stable within-mode
complete-output hashes and **3.64/3.67 s** median request latency. M8
measured **4.47/4.45 s** and full KV **2.37/2.13 s**. Only **1/6**
complete outputs matched M8, so this is a latency/approximation tradeoff,
not evidence for a new default or general answer quality. The named M16
fixture has exactly the same resource bounds as the M8 deferred fixture;
`--pvd-kv-refresh-interval 16` remains an explicit launch choice.

### 2095 行冷 CAGRA 与 exact 准入 / Cold CAGRA versus exact admission

在相同 P/D/Gateway、Qwen2.5-7B、M16/lead6/Top4、单 rail 下，
72 条记录的 Prompt 为 2095 token（部分 case 为 2113），超过 V
当前 2048 行 exact 阈值。首次原生 CAGRA 路径的一条请求首代码
正确，但耗时 **26.60 s**；D 日志显示等待检索约 **23.44 s**，
V 两个 rank 分别为 56 个 head 建图、每 rank 合计约 **10.46 s**，
就绪前搜索收到可重试的 400。该时间不是一次 GPU search 的耗时。

只将 V 的 `PVD_PROMPT_INDEX_EXACT_MAX_ROWS=2304`（显式实验覆盖）
并保持 2 GiB index 预算、其他组件和请求不变，2095-token 单请求
耗时 **3.91 s**，完整输出哈希与冷 CAGRA 路径相同；V 两 rank 的
exact 索引构建合计分别约 **0.053/0.035 s**。随后六个不同
2095–2113-token 事实请求、两个客户端、两轮均为首代码 **6/6**，
模式内完整输出哈希稳定，中位约 **4.51/4.52 s**。仅切换 D 至
完整 KV，对照两轮也是 **6/6**，中位约 **2.26/2.29 s**；
exact 稀疏 M16 与完整 KV 的完整输出 **3/6** 哈希相同。
这些小样本说明：短寿命 Entry 强制冷建 CAGRA 是严重尾延迟，
但 exact 规避冷建图后仍未证明稀疏路径更快。静态 2304 阈值
不解决更长索引的准入，也可能增加多 Entry 常驻内存；只更新了
此 2304-context CloudLab 实验启动器的默认阈值和 index 预算
（512→2304、1→2 GiB），没有修改通用 V 服务端默认值。

For a 2095-token prompt just above V's current 2048-row exact threshold,
the first native CAGRA run answered correctly but took **26.60 s**;
D spent about **23.44 s** waiting for search while V built 56 graphs per
rank, about **10.46 s** per rank in total. Raising only V's explicit
experimental exact threshold to 2304 gave **3.91 s** with the identical
complete-output hash; V's two exact builds took about **0.053/0.035 s**.
Two replays of six distinct 2095–2113-token facts with two clients were
**6/6** correct, stable within mode and **4.51/4.52 s** median. Full-KV D
under the same P/V/Gateway answered **6/6** in **2.26/2.29 s** median;
only **3/6** complete outputs matched exact sparse M16. Exact-first
admission is clearly preferable for these short-lived Entries, but the
static threshold does not solve longer indexes or prove sustainable memory
pressure. Only this 2304-context CloudLab launcher's defaults were changed
(exact threshold 512→2304, index budget 1→2 GiB); the general V server
defaults and explicit native-CAGRA overrides were not changed.

更新后的隔离 V 启动器通过 `bash -n`，在不提供阈值/预算环境变量时，
实测进程参数为 `--prompt-index-exact-max-rows 2304`、
`--prompt-index-budget-bytes 2147483648`；新的 2095-token Entry 在
两 rank 均报告 `path=exact`、建索引约 0.053/0.035 秒。此时 D
切到完整 KV 做启动器冒烟验证，一条请求首代码正确、耗时约 2.19 秒；
这不是稀疏模式的额外性能样本。该默认仅适用于这份 CloudLab V100S
启动脚本，用户仍可显式降低阈值验证原生 CAGRA。

The updated isolated V launcher passed `bash -n`. Without threshold or
budget environment overrides, the live process used exact-max-rows 2304
and index-budget-bytes 2147483648; a new 2095-token Entry reported
`path=exact` on both ranks with ~0.053/0.035-second builds. D was in
full-KV mode for this launcher smoke (one correct request, ~2.19 s), so it
is not another sparse-performance sample. Explicit lower thresholds remain
available for native CAGRA acceptance.

## 未完成 / Remaining work

1. **Performance:** predictive sparse refresh still loses to warmed full KV under two-client load even after avoiding native CAGRA cold build. Determine a measured admission strategy (exact first, background CAGRA promotion only when the Entry is likely to be reused) and preserve index/version/retirement fencing before implementing it. The 2048-row exact threshold is an experiment, not a new default.
2. **Quality:** the real-Qwen target-Q recall test still covers one repeated-text prefix, two layers and eight GQA-head queries per length. The six new fact prompts answer their first code correctly in both modes, but all six full output hashes differ across sparse/full KV. Expand to a broad recall distribution, diverse natural prompts and answer-level metrics before claiming quality equivalence.
3. **Scale:** this experiment is P TP1, V 2 ranks, D TP1, single rail, one or two clients, max 2304 sequence tokens. TP asymmetry, dual-rail, long-running load, TTL pressure, multi-D routing, and full GPU-memory safety have not been established here.
4. **Cache:** the two-round same-seed A/B indicates lower capture time without changing these four outputs; repeat with matched warmups, more varied prompts and budget snapshots, then exercise concurrent close, cancellation and retraction. The earlier three-request run established only the narrow cleanup regression.
