# Causal-adapted six-layer Draft-Q: full-path quality — CloudLab, 2026-09-30

## Experiment

The new checkpoint was selected solely on the separate causal calibration
split before this output benchmark. Training excludes all 40 benchmark
questions and the repository ReAct question source. The architecture stays
six Qwen2.5-0.5B blocks plus rank-896 target-Q readout; the new checkpoint
SHA256 is `75a0c48a95d80ef8ccf6025e4c92e2ea2a028d9cf394e327fb69bec1da007efe`.
Training and native recall details are in
`pvd_draft_q_decode_cloudlab_20260930.md`.

The frozen dataset SHA256 is
`5591e964da8fd457f0e113d82330c84c94e6074b1ad80d1ee414dca751d893c3`:
16 GSM8K test and 24 HotpotQA dev distractor questions, in the original mixed
order. Every HotpotQA context has all ten passages. Greedy generation stops
naturally at EOS, with the same limits of 512 math / 128 reading tokens.
Primary numeric math scoring and strict `FINAL:` reading EM/F1 are unchanged.
The previously declared `FINAL ANSWER:` sensitivity rule is reported separately.
No answer judge or benchmark-guided checkpoint tuning is used.

## Matched configuration and reference reuse

- Fresh P, V, D and Gateway processes; one excluded six-token warmup.
- Node0 P TP1/GPU0, node1 V rank0/GPU0 and rank1/GPU1 plus Gateway,
  node2 D TP1/GPU1. Qwen2.5-7B FP16, context 2304, page size 1,
  Prefill chunks 512, radix cache disabled, native Mooncake RDMA.
- **P→D initial KV disabled**; V delivers initial complete KV only after
  both rank indexes READY. P first-token/metadata signaling remains.
- Latest graph timing predictor; centered four-head CAGRA, exact degree16,
  cuVS 25.10, itopk2048, 14 graphs/rank and 32768 V pages.
- Predictive refresh every four tokens, eight-token Draft horizon, query
  lead two, one selected future position, Top16 per Q head and 128-token
  per layer/KV-head union cap. FP32 Draft/readout resident parameters and
  FP16 readout autocast; each refresh recomputes its private committed prefix.
- Experimental retained fan-in control-record cap 4096, serving default
  still 1024. No prediction fallback or candidate-budget tuning is intended.

Full-KV, real-target-Q and previous six-layer results are reused from the
earlier completed 40-question experiment. Those exact JSONLs and reference
logs are copied unchanged and hashed here. They ran before the new checkpoint,
in separate processes; this is not a randomized interleaved timing trial.
Old full-KV first 39 requests used record cap1024 and its last resumed request
used4096; both old predictive references and every new request use4096.
The model, prompt, generation and retrieval settings match. The reference
exception and excluded attempts remain documented in
`pvd_output_quality_cloudlab_20260930.md`.

Old serving commit is `839eefd64`; new P/D checkout is `f647b3797`, V/Gateway
checkout is `3df89fb6e`. Diff under `python/sglang/srt` is empty between
the old serving revision and this experiment. New commits add benchmark
capture, launcher overrides, validation and evidence only. D startup logs
prove the selected new checkpoint SHA256 and unchanged six-layer/rank896
configuration; the experimental SHA override does not alter the default.

## Excluded startup failure

The first Gateway launch preceded V readiness and exited during coordinator
health validation. The attempted warmup failed with connection refused;
there were no formal answer rows. After V health200, Gateway was restarted,
health checked, and a fresh excluded warmup completed. Original failure logs
are retained; model weights, serving parameters and dataset did not change.

## Results

All **40 new requests** completed with confirmed Entry release and collector
exit0. Existing reference rows also pass the same matched dataset/order,
Prompt token/hash, gold, generation-limit and successful-finish checks.

| Method | GSM8K numeric accuracy | HotpotQA strict EM | HotpotQA strict token F1 |
|---|---:|---:|---:|
| Full KV reference | 15/16 — 93.75% | 17/24 — 70.83% | 80.10% |
| Existing Draft + real target Q | 15/16 — 93.75% | 15/24 — 62.50% | 71.07% |
| Previous six-layer joint | 3/16 — 18.75% | 13/24 — 54.17% | 66.30% |
| New causal-adapted six-layer joint | **8/16 — 50.00%** | **14/24 — 58.33%** | **67.29%** |

Math gains five correct answers with **zero previously correct answers
lost**: IDs 288, 1009, 1213, 1289, 449. Case 449 now uses the correct packet counts.
Case 288 ends at the correct answer 3 but still invents a 750-pound truck
capacity in part of its explanation; final-answer accuracy alone does not
establish faithful reasoning. Case 137 remains wrong at 27.80 versus gold 29.
Reading primary scoring gains two
and loses one; the already declared format-sensitive variant resolves that
to one semantic gain and zero losses. Variant-tolerant reading EM/F1:

| Method | Reading variant EM | Reading variant F1 |
|---|---:|---:|
| Full KV | 17/24 — 70.83% | 80.10% |
| Real target Q | 17/24 — 70.83% | 79.40% |
| Previous joint | 15/24 — 62.50% | 74.63% |
| New joint | **16/24 — 66.67%** | **75.62%** |

This is a measurable recovery, **not restored reference quality**: math is
still seven correct answers below either reference. The fixed subset is small,
with one greedy generation per method; no full-benchmark guarantee is implied.
Both old and new joint arms have two 512-token length-limited math responses.
New IDs 1193 and 186 are included; no reading answer is truncated. New math has
13/16 missing strict `FINAL:`, strict numeric FINAL accuracy 2/16, and primary
numeric accuracy 8/16 under the frozen FINAL-then-last-number rule. The answer
format problem therefore remains, alongside incorrect arithmetic/reasoning.

New completions total 4910 math and 178 reading tokens, versus 5013 and 183 old.
Math median client duration is 51.408 s versus 56.219 s old; reading 2.791 s versus
2.897 s. These are different generated trajectories in separate runs, not
equal-quality or repeated-request speedup measurements.

## Serving evidence and remaining gate

Matched P/D logs prove V initial KV and the dual-rank READY barrier for all 40
requests. P chose prefix 0, a complete build, for all 40. Every math request installs
sparse banks, totaling 1215 boundaries and 1215 predicted refreshes. Reading
installs banks for 15/24 requests, totaling 21 boundaries and 21 predicted
refreshes. All 1236 successful refreshes use predicted Q, with zero committed-Q
fallback. All 1259 logged formal Draft-Q predictions match their request IDs
and show target-forward-count 0. Extra predictions can be canceled at natural
completion; refresh failure records are teardown `CancelledError` only.

The formal per-prediction logged prefill+rollout median is 59.927 ms over 1259
predictions with 87–2053 committed-prefix tokens. These varying-prefix serving
samples differ from the idle 2155+8 microbenchmark, and exclude native search
and delivery. They do not measure a paired end-to-end latency gain.

Nine short reading answers finish before any sparse bank is installed; some
installed banks may affect EOS rather than answer-content tokens. Math's many
actual sparse boundaries provide the stronger full-path quality signal.
Preserve the existing real target-Q serving default: the new checkpoint still
fails the output-quality equivalence gate.

The current adaptation uses one HF teacher capture round with teacher-committed
prefixes; it does not cover every erroneous sparse-D trajectory. The teacher
uses repetition penalty 1.0; serving retains its existing target generation
configuration and Draft penalty. More on-policy token/Prompt-K ranking
training and teacher/runtime alignment are the next experiments. Raising
candidate budgets would be a separate controlled experiment.

## Reproduction and artifacts

`pvd_output_quality_decode_cloudlab_20260930/` retains frozen prompts, new and
old answers, collection status, warmup, startup failure and compressed logs.
The summary, checkpoint-paired changes and serving evidence JSON reports are
adjacent to this report. Their extraction/scoring rules are unchanged from
the previous experiment. All six GPUs returned to 0 MiB / 0% after owned services
stopped; the temporary private-network checkpoint-copy server was closed.
