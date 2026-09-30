# Jointly trained Draft-Q in the full P/V/D path — CloudLab, 2026-09-30

## Result

The latest six-layer CE+Q+score-KL checkpoint now runs in D's real predictive
serving path. Four paired requests completed with 16 output tokens each. P did
not send KV to D: P uploaded to V; D waited for both V graph indexes, pulled
initial full KV from V, then used native V search and sparse V→D delivery for
three refreshes. All 12 formal learned-Q refreshes used predicted queries and
logged zero target-model forwards; no committed-Q fallback occurred.

The two repetitive-text pairs had identical final output hashes and saved
1.314/3.243 seconds (25.4%/39.1%). **Both natural-text pairs changed output.**
Their lower latency is not evidence of acceleration at equal quality. This is
an opt-in integration experiment; the real target-Q probe remains the default.

## Configuration and attribution

- Serving code: `4ff33838636ca2cea49678a1491e7514de45a52a`; integration commits
  `b78751449`, `066af62c4`, `4ff338386`. Latest V split predictor was imported
  in `300fcccdd` and `ad0a9ba53` before these runs.
- Three CloudLab hosts with V100S: node0 P TP1/GPU0; node1 V ranks 0/1 on
  GPU0/1 plus Gateway; node2 D TP1/GPU1. Native Mooncake RDMA, Qwen2.5-7B
  FP16, context 2304, page size 1, 512-token chunked Prefill, P radix cache off.
- V: centered four-head CAGRA, exact degree-16 initial KNN seeds imported
  using native cuVS 25.10 `from_graph`, 14 graphs/rank, `itopk_size=2048`.
  Native extension remains available when the split predictor chooses a prefix.
- Latest graph timing predictor:
  `pvd_split_parametric_profile_cloudlab_20260930.json`, per-rank arrival,
  build and extension fits with a 0.795243-second uncertainty guard. It chose
  **prefix 0 on all five requests per arm**, including warmup. Predicted best
  split gains were only 0.08585–0.09359 seconds. Therefore these runs built once
  after complete KV arrival; they do not measure streaming/extension overlap.
- `PVD_DIRECT_PD_BOOTSTRAP=0`, `PVD_GATE_INITIAL_FANIN_ON_INDEX=1`. The explicit
  graph barrier was identical in both arms. Normal P→D first-token/metadata
  signaling is still required; it carries no direct bootstrap KV.
- Refresh every four Decode tokens; Draft horizon eight; query lead two.
  The serving window searches **one selected future position per refresh**.
  Top-16 per query head, seven GQA query heads per KV head, union cap 128 IDs.
  This does not validate an eight-position Top-16 union or the earlier offline
  Top-80 budget. Native search, union, packing, transfer and bank installation
  are all included in the online refresh timings below.
- Joint arm: first six trained Qwen2.5-0.5B blocks, learned six-anchor fusion
  and rank-896 all-layer target-Q readout; FP32 resident parameters, FP16
  readout autocast, target absolute RoPE with theta 1e6. Checkpoint SHA256:
  `4b3fa87316c6b70d513fb32c3cdb79baf586e95046efbbfe3a51939cc8974215`.
  Resident parameters occupy 1,281,770,656 bytes (1.194 GiB), charged to a
  separate 2 GiB persistent budget; private eager Draft scratch reserves 2 GiB.
- Target arm: existing full 24-layer 0.5B Draft generates tokens, then the
  real 7B target probe computes their Q. Same placement, scratch declarations,
  V graphs, retrieval budgets and initial KV barrier.

## Paired client measurements

The order was joint arm then target arm. Each arm had fresh P/V/D/Gateway
processes, a separate excluded warmup, and the same four sequential Prompts.
V restarted between arms to reset stored entries and graph capacity. Prompt
hashes, input/output lengths and sampling settings match. All responses were
HTTP 200 with no client error. Sampling was greedy, `ignore_eos=true`.

Each case has **one measurement per arm**. There is no reversed-order repeat,
confidence interval, concurrent-load result or TP2 result. The table gives
observed paired differences, not a general performance guarantee.

| Prompt | Tokens | Target-Q TTFT | Joint-Q TTFT | Target-Q total | Joint-Q total | Saved | Output hash matches |
|---|---:|---:|---:|---:|---:|---:|---|
| Repetitive short | 1007 | 1.399 s | 1.408 s | 5.174 s | 3.860 s | 1.314 s / 25.4% | Yes |
| Repetitive long | 2157 | 2.464 s | 2.323 s | 8.287 s | 5.043 s | 3.243 s / 39.1% | Yes |
| Natural short | 1009 | 1.486 s | 1.337 s | 5.235 s | 3.766 s | 1.470 s / 28.1% | **No** |
| Natural long | 2155 | 2.450 s | 2.297 s | 8.461 s | 5.220 s | 3.241 s / 38.3% | **No** |

TTFT is elapsed client time to the first streamed event; total ends after the
stream closes. The median across these four different cases is 6.761 s for
target-Q and 4.452 s for joint-Q. This is a descriptive aggregate, not a
repeated-request median.

Warmup is retained in raw evidence: joint TTFT/total 24.286/54.036 s; target
2.955/20.130 s. The first joint run included cold P CUDA compilation and D
attention compilation; target ran later and had a cold Draft kernel path.
These cold runs had different compilation-cache states and are not a fair
cold-start comparison. Their costs are excluded from the formal table.

## Initial graph and V→D KV path

All times below are seconds. Graph build duration is logged inside V's actual
manager final step, including its extraction/native graph work; it is not
the earlier isolated CAGRA microbenchmark. The ranks build concurrently.

| Joint-Q Prompt | V rank0/rank1 build | Request→both READY | D graph gate wait | D initial fan-in, including gate | Fan-in minus gate |
|---|---:|---:|---:|---:|---:|
| Repetitive short | 0.892 / 0.888 | 1.300 | 0.859 | 0.931 | 0.072 |
| Repetitive long | 1.349 / 1.353 | 2.172 | 1.357 | 1.457 | 0.101 |
| Natural short | 0.876 / 0.877 | 1.234 | 0.871 | 0.944 | 0.073 |
| Natural long | 1.319 / 1.321 | 2.146 | 1.316 | 1.417 | 0.101 |

The fan-in remainder includes control work, native transfer, polling and GPU
installation; it is not a pure network latency measurement. For the joint
2155-token request, each V rank delivered 61,788,160 bytes in one rank-packed
native write; terminal elapsed times were 0.048781/0.048769 s. Both V writer
terminals reported success. D then logged `initial KV installed from V`.

The initial wait remains: the learned predictor runs during later refresh,
so it does not remove the V graph barrier or explain the small TTFT differences.
For the long repetitive pair, V build time itself differed by roughly 0.15 s
between arms, while total latency differed by 3.24 s. Most of the measured
gain is in the three subsequent Q capture stages.

## First refresh, including native V retrieval and delivery

The following is the 2155-token natural-text pair. Its outputs differ, so this
stage comparison demonstrates the execution path and cost, not equal quality.

| Stage | Existing Draft + target-Q probe | Joint six-layer Draft-Q |
|---|---:|---:|
| Draft forward / joint prefix prefill | 342.1 ms | 55.6 ms |
| Target Q probe / eight-token rollout plus learned Q | 749.7 ms | 50.5 ms |
| Complete capture, including scope and query preparation | 1112.8 ms | 129.4 ms |
| V graph search | 375.3 ms | 378.2 ms |
| Candidate union | 3.0 ms | 3.0 ms |
| Sparse KV delivery and installation | 396.7 ms | 348.1 ms |
| **Complete first refresh** | **1887.8 ms** | **858.7 ms** |

The two prediction rows are substeps of capture, not additional durations.
Search is native CAGRA across both V ranks. The log's `ranks=1` refers to D's
single TP rank, not to V's GPU count. Delivery includes packing/control/native
transfer/completion/installation under the existing serving refresh driver.

For the **identical-output 2157-token repetitive pair**, first refresh dropped
from 1794.1 ms to 793.0 ms: capture 1104.2→137.4 ms, search 366.0→367.3 ms,
union 3.1→2.8 ms, delivery 320.8→285.5 ms. The next two refreshes were
1810.8/1835.6 ms versus 780.2/770.9 ms. This supports the attribution to faster
Q production rather than a graph-search speedup.

Across the joint arm's 12 warmed refreshes, long-prefix prefill was
55.6–63.5 ms and eight-token rollout/Q 50.5–52.2 ms. Short-prefix prefill was
21.5–26.1 ms and rollout/Q 50.2–54.9 ms. Full capture adds about 21–23 ms for
scope and query preparation. This first integration recomputes the committed
Draft prefix at each refresh; it does not retain a cross-refresh private cache.

## Output quality and remaining gates

The natural-text source is a held-out tail of the repository's HotpotQA JSONL
rendered as completion text. This is a continuation check, not a scored QA
evaluation. For example, the natural-long target arm continued:

> ` and 81 episodes, until May 26, 199`

The joint arm continued:

> ` and 81 episodes until 1999. The Simpsons is an`

Neither is an authoritative dense-model answer. Output divergence shows that
changing Q affects generation; it is not, by itself, a measured exact-neighbor
recall value. Earlier offline combined token/Q recall was about 0.546 under
its own test protocol. This online run does not establish better recall or
quality equivalence. A dense-target reference, longer held-out generations,
on-policy retrieval recall and bounded candidate improvements remain necessary
before considering default enablement.

The adapter keeps target-Q committed-position recovery, request/prefix identity,
absolute future positions, private Draft KV, and bounded scratch ownership.
It supports the validated TP1 Qwen2.5-7B geometry and at most 2304 tokens only.
Concurrent/cooperative prediction, sidecar, target Prompt-KV seeding and eager
precompile combinations are rejected for this experimental path. Unknown CUDA
completion retains private owners and prevents further Draft use.

TP2, cancellation/unknown-operation races, pressure testing, live multi-request
contention and a measured memory peak are not validated here. Resident weight
accounting above is not a peak-memory measurement.

## Validation and reproduction

The CloudLab targeted regression run passed **14 tests** covering the learned-Q
request/layer/head/position mapping, absolute RoPE, incompatible geometry,
existing CUDA prediction startup and the graph split policy. Windows syntax
compilation passed; runtime tests ran on Linux/CloudLab. Pytest was installed
only into an isolated validation dependency directory, not the serving env.

```bash
PYTHONPATH=python:/proj/llm-course-PG0/Yizhzhu-node0-sglang-pvd/validation/joint-draft-q-testdeps-20260930 \
python -m pytest -q \
  test/registered/disaggregation/test_pvd_joint_draft_q.py \
  test/registered/disaggregation/test_pvd_cuda_prediction_startup.py \
  test/registered/disaggregation/test_pvd_split_upload_policy.py

# On each role's host, with isolated validation checkout and model paths:
bash benchmark/launch_pvd_joint_draft_q_cloudlab.sh p joint
bash benchmark/launch_pvd_joint_draft_q_cloudlab.sh v joint
bash benchmark/launch_pvd_joint_draft_q_cloudlab.sh d joint
bash benchmark/launch_pvd_joint_draft_q_cloudlab.sh gateway joint
# Wait for health on P, D, V and Gateway before sending requests.
python benchmark/pvd_joint_draft_q_online_probe.py --arm joint \
  --tokenizer /users/Yizhzhu/pvd-models/Qwen2.5-7B-Instruct --source-root . \
  --output /absolute/path/jointq-joint-20260930.jsonl
# Stop these experiment-owned process groups, restart all roles with `target`,
# and run the same probe with --arm target and its own output file.

python benchmark/analyze_pvd_joint_draft_q_online.py \
  benchmark/results/pvd_joint_draft_q_online_cloudlab_20260930 \
  --output benchmark/results/pvd_joint_draft_q_online_detail_cloudlab_20260930.json
```

The analyzer rejects mismatched Prompt hashes/token lengths, missing V ranks,
missing bootstrap/refresh stages, failed responses, unexpected split choices,
committed fallbacks, and joint refreshes with target forwards. Exact stages,
request/Entry IDs, output hashes and raw-file SHA256 values are saved in the
detail JSON. Both client JSONLs and P/V/D/Gateway logs are retained in the
adjacent evidence directory.

All experiment-owned service groups were stopped after the paired run. Each
of the six GPUs returned to 0 MiB used and 0% utilization. Existing serving
defaults and the original CloudLab checkouts were left unchanged.
