# Draft-Q candidate budgets: CloudLab V100S

## Setup

Use the best six-layer, jointly trained Draft and rank-896 Q readout from
`pvd_draft_q_multitask_cloudlab_20260930.md`. Re-roll the trained Draft over
80 training Prompts, capture target Q and real Prompt K on those trajectories,
and calibrate a fixed per-head candidate policy. Hold out eight validation
Prompts. Each validation request uses two future query positions, 28 layers,
28 Q heads and 112 layer/KV-head transfer groups across both V ranks.

The static policy gives one Q head K16, two heads K4 and four heads K8 within
each seven-head GQA group: 56 candidates per position, equal to uniform K8.
Its chosen heads maximize training Top4 coverage among this constrained
family. Calibrate from the **trained** Draft trajectories; the earlier
pretraining trajectory calibration is saved separately as a stale control.

On CloudLab node1 V100S, build the same centered per-layer/KV-head native
cuVS 25.10 CAGRA graph for every policy (IVF-PQ builder, degree 16,
intermediate degree 32, `itopk_size=256`). Search the predicted Q once for
Top16 and score prefixes of that actual native result against the target
model's exact Prompt-K Top4 on the Draft trajectory. Also compare against
the target-model greedy trajectory to expose token-branch error. The true-Q
native search on the Draft trajectory checks graph approximation error.

## Held-out native CAGRA result

| Candidate policy | True Top4 coverage on Draft branch | True Top4 coverage on target branch | Fresh K+V per request, estimated |
| --- | ---: | ---: | ---: |
| Uniform K4 | 0.647 | 0.533 | 1.015 MiB |
| Uniform K8 | 0.809 | 0.682 | 1.907 MiB |
| Static per-head budget, equal candidate count to K8 | 0.781 | 0.657 | 2.058 MiB |
| Uniform K16 | 0.896 | 0.790 | 3.455 MiB |

True target Q searched through the same CAGRA graphs obtains **0.9996**
Top4 recall on the Draft branch. Thus the large remaining gap is Q/token
prediction, not CAGRA approximation in this fixture. Static per-head
allocation loses 0.028 coverage against K8 and consumes more transfer bytes
because its selected IDs overlap less. It loses on each of the eight held-out
Prompts. Recalibration on post-training Draft trajectories does not fix that
result (exact-K validation coverage 0.781 versus K8 0.809).

A normalized predicted-score margin threshold fitted to the bottom 20% of
training requests flags two of eight validation requests. Substituting true
target Q for *whole flagged requests* raises the static policy's Draft-branch
coverage from 0.781 to 0.835, below K16's 0.896. It misses the worst
validation request and omits the cost of computing/communicating true Q.
This margin is not a reliable fallback rule.

The native predicted-Q Top16 search median is roughly 1.72–1.87 ms **per
graph** across the eight requests when performed sequentially. The policy
comparison reuses this same result, so these data do not establish a search
latency difference among K4, K8 and K16. Isolated graph construction averages
21.64 s per request for 112 graphs and is excluded from query-path timing.

The K+V figures count unique token IDs per layer/KV-head group at 512 bytes
per token (128 FP16 K plus 128 FP16 V). They are payload estimates, **not
measured V→D transfer times**. Network, packing, control messages, cache
reuse, GPU copy and two-rank contention have not been measured for these
policies. Since the equal-budget adaptive candidate failed the recall gate,
it was not integrated into PVD for a transport timing run.

## Long Prompt capacity check

With two 2155-token Prompts, exact Prompt-K scoring of the trained Draft Q
gives the following two-position results. This is a capacity and candidate
quality probe, not native CAGRA search.

| Prompt | K8 Top4 coverage / max group union | K16 Top4 coverage / max group union | K16 groups above 128 |
| --- | ---: | ---: | ---: |
| Case40-like repetition | 0.835 / 56 | 0.923 / 91 | 0 / 112 |
| Hotpot last-20 text | 0.815 / 70 | 0.893 / 123 | 0 / 112 |

The implemented two-position cap policy begins with K16 and reduces selected
heads to K8 until each seven-head GQA group's union is at most 128 IDs. It
made no reductions on these two Prompts. K8 has a worst-case union of 112 for
two positions; K16 can reach 224 and therefore requires the cap. Earlier
eight-position experiments exceeded the 128-ID limit; the two-position
result must not be extrapolated to a longer refresh batch.

## Decision and remaining gate

Reject the tested fixed per-head policy and margin fallback. Uniform K16 is
a candidate for a **two-position** online experiment: it improves native
coverage by 0.088 over K8 on the Draft branch, but costs about 1.55 MiB more
fresh K+V per request in this fixture and still reaches only 0.790 coverage
against the true target trajectory. Keep the current target-Q serving path.
Before changing it, measure actual two-rank V→D transfer, search batching,
refresh latency and answer quality with a capacity-safe K16 policy. The
current experiment does not claim a Decode latency gain.

Scripts: `pvd_draft_q_budget_refresh.py`, `pvd_draft_q_budget_prepare.py`,
`pvd_draft_q_budget_native.py`, `pvd_draft_q_long_union_probe.py`. Raw data:
`pvd_draft_q_budget_stale_cloudlab_20260930.json`,
`pvd_draft_q_budget_refreshed_cloudlab_20260930.json`,
`pvd_draft_q_budget_native_cloudlab_20260930.json`, and
`pvd_draft_q_long_union_cloudlab_20260930.json`.
