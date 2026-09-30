# Draft-Q output quality in the full P/V/D path — CloudLab, 2026-09-30

## Result

All **120 formal requests** completed with confirmed Entry release: 40 per
arm, comprising 16 GSM8K and 24 HotpotQA questions. The latest joint six-layer
Draft-Q has a substantial answer-quality regression on this fixed subset.
The existing real target-Q probe preserves the full-KV math accuracy; the
joint predictor does not. Keep the experimental predictor opt-in.

### Frozen primary metrics

| Method | GSM8K numeric accuracy | HotpotQA strict answer EM | HotpotQA token F1 |
|---|---:|---:|---:|
| Full KV | 15/16 — **93.75%** | 17/24 — **70.83%** | **80.10%** |
| Existing Draft + real target Q | 15/16 — **93.75%** | 15/24 — **62.50%** | **71.07%** |
| Joint six-layer Draft-Q | 3/16 — **18.75%** | 13/24 — **54.17%** | **66.30%** |

### Format sensitivity, reported separately

Both predictive arms have two correct reading answers written as
`FINAL ANSWER:` rather than the requested `FINAL:`. Accepting that declared
variant changes their reading scores as follows; full KV is unchanged:

| Method | HotpotQA variant-tolerant EM | HotpotQA variant-tolerant F1 |
|---|---:|---:|
| Full KV | 17/24 — **70.83%** | **80.10%** |
| Existing Draft + real target Q | 17/24 — **70.83%** | **79.40%** |
| Joint six-layer Draft-Q | 15/24 — **62.50%** | **74.63%** |

Under variant-tolerant scoring, the old probe and full KV have the same set
of correct math and reading answers. Joint loses **12 previously correct
math answers and two previously correct reading answers**, with no gains.
Its math accuracy is 75 percentage points below either reference. This
remains a semantic regression after accounting for format noncompliance.

| Method | Math missing strict FINAL | Math strict FINAL accuracy | Length-limited math / reading |
|---|---:|---:|---:|
| Full KV | 8/16 | 7/16 | 0 / 0 |
| Existing Draft + real target Q | 13/16 | 3/16 | 0 / 0 |
| Joint six-layer Draft-Q | 9/16 | 0/16 | 2 / 0 |

Only joint requests `gsm8k:591` and `gsm8k:186` reached the common 512-token
limit. They are included in the score. Even if both were corrected by a
longer limit, joint would reach at most 5/16 on these questions, so truncation
does not explain the full gap. All other formal requests ended naturally.

### Observed errors

| Question ID / key condition | Gold | Full KV | Real target Q | Joint Q |
|---|---:|---:|---:|---:|
| GSM8K 288: 80 flagstones × 75 pounds, truck capacity 2000 | 3 | 3 | 3 | 1 |
| GSM8K 137: meal price, delivery fee and tip | 29 | 29 | 29 | 28.40 |
| GSM8K 449: six packs of eight and four packs of sixteen | 112 | 112 | 112 | 160 |

Joint's explanation in case 288 changes the input to seven pounds and ten
flagstones. In case 449 it substitutes ten packs of eight plus five packs
of sixteen. These are failures to retain the question's numeric conditions,
not simply a different writing style. This suggests lost useful Prompt
information, but does not isolate learned-Q error from Draft token error
or every possible sparse-attention interaction.

## Evidence that the complete serving path was exercised

Matched D logs prove graph-gated initial KV from V for all 120 requests.
P logs show the latest graph predictor chose **prefix 0 for every formal
request**, so V built after full KV arrival. These runs do not measure native
extension overlap. Sparse-bank installation counts are:

| Arm / task | Requests with installed sparse bank | Installed boundaries / predicted refreshes | Committed-Q refreshes |
|---|---:|---:|---:|
| Target / GSM8K | 16/16 | 1008 / 1008 | 0 |
| Joint / GSM8K | 16/16 | 1239 / 1239 | 0 |
| Target / HotpotQA | 15/24 | 25 / 25 | 0 |
| Joint / HotpotQA | 15/24 | 24 / 24 | 0 |

Full KV has zero sparse-bank installs. All joint prediction records log zero
target-model forwards and match their request IDs. Every formal successful
refresh is predicted, with no committed-Q fallback. Joint refresh-failure
records are `CancelledError` during request teardown.

Nine reading responses per predictive arm finish before a sparse bank is
installed. Some of the other short answers can install a bank immediately
before EOS; installation alone does not prove that many answer-content tokens
used it. Thus reading scores are partly diluted by the shared initial full-KV
path. All sixteen long math completions use many actual sparse boundaries,
making the large math regression a stronger full-path quality signal.

The joint math completions average 313.3 tokens, versus 255.7 for target and
258.0 for full KV. Their median client duration is 56.219 s, versus 55.591 s
for target; faster Q production does not guarantee shorter completion when
generation length changes. These are medians over different questions,
not repeated-request latency estimates. No equal-quality speedup is claimed.

## Benchmark and scoring

This is a fixed **40-question subset**, not a full public benchmark run.
Sixteen questions come from the official [GSM8K test set](https://github.com/openai/grade-school-math).
Twenty-four come from [HotpotQA dev distractor](https://hotpotqa.github.io/),
with all ten supplied passages intact. Answer EM and token F1 use the
[official HotpotQA scoring rules](https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py).
Supporting-fact metrics are not evaluated. No model judge is used.

Selection uses seed 20260930, a shuffled source order, and a fixed context
limit. It excludes the questions in the repository's ReAct HotpotQA source,
which appeared in the predictor's training material. This removes known
question overlap, not every possible training or model-pretraining overlap.
HotpotQA prompts must contain at least 600 tokens. Prompt, generation bound
and eight-token horizon must fit 2304 tokens. Selection does not use model
answers or correctness. The same mixed question order is used for each arm.

GSM8K prompts contain 84–169 tokens; HotpotQA prompts contain 760–2046.
The Qwen chat template asks for brief reasoning and a final answer line.
All arms use greedy sampling, natural EOS, and the same limits: 512 generated
tokens for GSM8K, 128 for HotpotQA. All passages and exact prompts are retained
in the frozen dataset. This is not a long-context or concurrent-load result.

- GSM8K primary accuracy: extract a numeric `FINAL:` answer when possible,
  otherwise the last number in the completion, retaining signs and decimals.
  Strict `FINAL:` accuracy and missing-format counts are also reported.
- HotpotQA primary EM/F1: extract the last `FINAL:` answer line and apply
  official normalization. Missing `FINAL:` receives zero. A declared
  **post-hoc format sensitivity check** additionally accepts `FINAL ANSWER:`
  with the answer on the same or following line. It never consults the gold
  answer or guesses from an unrestricted explanation. Primary scores remain
  unchanged. This check was added after observing two correct joint-arm
  answers using that alternative spelling.
- Generation failures and aborted streams are rejected, rather than scored
  as wrong model answers. Length-limited valid generations are scored and
  counted separately. An Entry is released only after its completed request
  and consumer/delivery gates permit release.

## Matched serving configuration

The three arms are `full` (complete Prompt KV during Decode), `target`
(existing 24-layer 0.5B Draft token rollout followed by the real 7B Q probe),
and `joint` (latest jointly trained six-layer Draft and learned target-Q
readout). Each starts fresh P/V/D/Gateway processes and runs an excluded
six-token warmup before the fixed questions. The order is full, joint, target.
There is one greedy generation per question per arm.
The legacy transport warmup helper labels a non-joint warmup `target`,
including the full arm; actual full mode is established from D serving logs,
not from that warmup-only label. Formal answer rows use the correct arm.

- Runtime checkout on all three nodes: `839eefd64`. Subsequent local commits
  add only evidence analysis, UTF-8 I/O, and format sensitivity scoring.
- CloudLab V100S: node0 P TP1/GPU0; node1 V ranks 0/1 on GPU0/1 plus Gateway;
  node2 D TP1/GPU1. Target Qwen2.5-7B FP16, native Mooncake RDMA, page size 1,
  context 2304, 512-token Prefill chunks, P radix cache disabled.
- **No P→D bootstrap KV**: direct P→D flag is zero. All arms require both V
  indexes READY before initial V→D full-KV fan-in. P's first-token/metadata
  signaling remains part of the protocol.
- V uses the latest parametric graph timing predictor, centered four-head
  CAGRA, native cuVS 25.10, exact degree-16 initial seeds, 14 graphs/rank,
  and `itopk_size=2048`. V allocation is 32768 pages in this experiment.
- Both predictive arms refresh every four Decode tokens, use an eight-token
  Draft horizon and query lead two, and search one selected future position
  per refresh. Top-16 per Q head and a 128-ID per layer/KV-head union cap are
  identical. This is not an eight-position union or a Top-80 experiment.
- Joint checkpoint SHA256:
  `4b3fa87316c6b70d513fb32c3cdb79baf586e95046efbbfe3a51939cc8974215`.
  Six trained Qwen2.5-0.5B blocks, learned six-anchor fusion, rank-896 target-Q
  readout, FP32 resident parameters, FP16 readout autocast, target absolute
  RoPE theta 1e6. Each refresh recomputes its private committed prefix.

The `full` arm retains the current full-KV refresh every four tokens. It is
an answer-quality reference, not a unified dense-inference latency reference.
Different methods generate different lengths. Client durations therefore do
not establish acceleration at equal output quality.

The old `target` arm probes real 7B Q on **Draft-predicted future tokens**,
not on an oracle's true future trajectory. Its difference from full KV can
include token prediction error, finite retrieval budget, approximate search,
and sparse attention. Joint versus target also changes token rollout and Q
production together; this is not a Q-only error isolation experiment.

## Collection repairs and excluded attempts

An initial full-KV pilot used a 256-token GSM8K limit and did not release
completed V Entries. After graph capacity accumulated, graph-readiness waits
became very long. That pilot is excluded from every formal score. Its 26
captured rows and original dataset are retained separately. The formal limit
was raised uniformly to 512 before either predictive arm was evaluated.

The formal full-KV run completed 39 valid questions, then its full-KV refresh
traffic filled the default 1024 retained fan-in control records. The last
request and an attempted resume aborted. They are not scored as answers.
The collector now rejects abort finish reasons and refuses to clear active
leases. The experiment launcher exposes a bounded record-cap override;
normal serving still defaults to 1024. After restarting all services with
4096 records, only the last uncompleted question was resumed successfully.
Thus the first 39 full-KV answers use cap 1024, the last uses 4096; model,
prompts, generation limits, full-KV content and retrieval configuration did
not change. Joint and target use cap 4096 throughout. This is a control
capacity difference that must be retained in provenance, not a model-quality
tuning change. The long-run record-capacity issue remains a serving concern.

## Evidence and reproduction

Frozen dataset SHA256:
`5591e964da8fd457f0e113d82330c84c94e6074b1ad80d1ee414dca751d893c3`.

Official source artifacts:

- [GSM8K test JSONL](https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl):
  `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`.
- [HotpotQA maintainer's dev-distractor Parquet mirror](https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/distractor/validation-00000-of-00001.parquet):
  `c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6`.
  The hash matches the published LFS object. Its title/sentences struct is
  converted into the official ten-paragraph representation.
- Excluded repository ReAct source:
  `41d0a75d73dd207a3fa3adb1440441d5a41442ed94eb19b2e38c3a00cc4519e9`.

Raw answers, immutable selected prompts, warmups, client logs and compressed
P/V/D logs are in `pvd_output_quality_cloudlab_20260930/`. Primary metrics and
actual V-bootstrap / sparse-bank evidence are separate JSON artifacts.

```bash
# Per node: p on node0, v/gateway on node1, d on node2; then health checks.
PVD_RUN_TAG=quality-ARM-20260930 PVD_FANIN_MAX_RECORDS=4096 \
PVD_V_TOTAL_PAGES=32768 \
bash benchmark/launch_pvd_joint_draft_q_cloudlab.sh ROLE ARM
# On node1; ARM is full, joint or target. Each arm requires fresh services.
bash benchmark/run_pvd_output_quality_cloudlab.sh ARM
python benchmark/pvd_output_quality.py compare \
  --folder benchmark/results/pvd_output_quality_cloudlab_20260930 \
  --output benchmark/results/pvd_output_quality_summary_cloudlab_20260930.json
python benchmark/analyze_pvd_quality_serving.py \
  benchmark/results/pvd_output_quality_cloudlab_20260930 \
  --output benchmark/results/pvd_output_quality_serving_cloudlab_20260930.json
```
