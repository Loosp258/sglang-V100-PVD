# Latest jointly trained Draft-Q inference latency

Measured on 2026-09-30 (Asia/Taipei), CloudLab node0, otherwise idle Tesla
V100S-PCIE-32GB GPU1. Use the best six-layer token Draft plus learned
rank-896 target-Q readout from the joint CE + Q MSE + score-KL 0.1 experiment.
The checkpoint is
`validation/draft-q-multitask-ceqkl-20260929/trained.pt`, SHA256
`4b3fa87316c6b70d513fb32c3cdb79baf586e95046efbbfe3a51939cc8974215`.

The student uses FP32 parameters and computation, matching the latest
held-out quality evaluator. The readout uses FP32 parameters with FP16
autocast, followed by target RoPE. The model is loaded with HF eager
attention and torch 2.9.1+cu128. Each emitted position contains Q for all
28 target layers and 28 target Q heads, each of dimension 128.

Each shape uses a repeated Case40 text Prompt, two unmeasured warmups and
eight single-request repetitions. CUDA synchronization brackets each timed
stage. Model load, tokenization, teacher inference, graph construction or
readiness waits, CAGRA retrieval, KV transport and serving contention are
excluded. These are warm experiment timings, not online D latency.

## Existing evaluator: generate, then replay the full sequence

This measures the actual functions used by the latest quality evaluator:
`generate_greedy()` generates the future tokens; `predicted_q()` then
replays the entire Prompt plus future sequence, copies captured features
to CPU and back, and computes target Q for all positions before selecting
the future positions. This path pays for another full Prompt forward.

| Prompt tokens | Future positions | Generate median | Full-sequence replay + Q median | Total median | Total p90 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 1 | 13.7 ms | 23.5 ms | 37.2 ms | 38.1 ms |
| 512 | 8 | 53.7 ms | 23.3 ms | 77.0 ms | 79.6 ms |
| 2155 | 1 | 57.5 ms | 132.4 ms | 190.1 ms | 191.9 ms |
| 2155 | 8 | 98.1 ms | 133.0 ms | **231.0 ms** | 232.0 ms |

## Cached forward: emit Q while consuming generated tokens

This timing harness retains the student's KV and captures the six anchor
states directly on GPU after each future-token forward. It runs the same
readout and target RoPE immediately; it does not replay the Prompt. Prefill
requests only the last-position LM logits. Token selection reproduces the
existing evaluator's greedy generation and inherited repetition penalty
1.1, with EOS termination disabled.

| Prompt tokens | Future positions | Private-KV prefill median | Rollout + all-layer Q median | Total median | Total p90 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 1 | 12.9 ms | 6.7 ms | 19.6 ms | 19.8 ms |
| 512 | 8 | 10.8 ms | 50.9 ms | 61.6 ms | 64.4 ms |
| 2155 | 1 | 56.6 ms | **6.7 ms** | 63.3 ms | 64.1 ms |
| 2155 | 8 | 55.8 ms | **51.1 ms** | **106.9 ms** | 107.9 ms |

The total medians are computed from paired samples, so they need not equal
the sum of the individual stage medians. The 512-token single-position
shape runs first, and its prefill is somewhat slower than the later
eight-position shape despite identical Prompt length; retained warm state
and device clocks are not independently controlled. This small difference
does not establish a horizon effect on prefill.

All emitted Q tensors are finite. The cached and replay paths produce
identical future token IDs in all four shapes. Their minimum per-position
Q cosine similarities are 1.0000000, 0.99999988, 0.99999994 and 0.99999994.
These checks validate the timing harness on this fixture; they do not
replace broader retrieval-quality validation or an online serving trial.

For a fresh 2155-token request, the cached harness reduces measured total
from 231.0 to 106.9 ms by avoiding full-sequence feature capture and replay.
If the current student KV already covers the committed prefix, the measured
eight-position rollout stage is about 51.1 ms; preparing or updating that
prefix KV is additional work. A single future position costs about 6.7 ms
for the six student blocks plus Q readout and RoPE, rather than for the
readout alone. The cached route is implemented only in this benchmark.

The earlier 91.8 ms result used a frozen student in FP16 and a bare argmax
rollout. It is not a controlled comparison with this trained FP32 run and
should not be interpreted as a training-induced latency regression.
The existing target-Q serving path has not been replaced.

Reproduce with `benchmark/pvd_draft_q_trained_latency_probe.py`, passing the
model directory and joint checkpoint, `--dtype fp32 --warmup 2 --repeats 8`.
Raw synchronized samples and metadata are in
`pvd_draft_q_trained_latency_fp32_cloudlab_20260930.json`.
