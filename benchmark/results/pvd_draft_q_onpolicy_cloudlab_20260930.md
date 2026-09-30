# Decode-focused Draft-Q with observed sparse-D prefixes — CloudLab, 2026-09-30

## Completed trajectory collection

Starting checkpoint is causal-adapted six-layer SHA256
`75a0c48a95d80ef8ccf6025e4c92e2ea2a028d9cf394e327fb69bec1da007efe`, whose
previous fixed output benchmark achieved math8/16 and strict reading14/24.
The six-layer rank896 architecture and Top16/128-token retrieval budget remain
the experimental serving configuration. The real target-Q probe remains default.

The first eight math and four reading **training** questions from the previous
frozen split are collected without correctness-based selection. Neither the
eight calibration questions nor the40 output questions enter this collection.
Dataset SHA256:
`77e00e3f77a215e44c26ba4acce9a62a2765b0f0677fd71bb44fe14e796d0f7f`.
All12 requests completed with confirmed Entry release, producing1808 exact
server output token IDs. The final `output_ids` length must equal the reported
completion token count; IDs are not reconstructed from text. Trajectory SHA256:
`5edb761bc763105260bcafb57ee39fa19eb482b059a127aec32c216547f49232`.

P→D initial KV is disabled. D starts with V initial KV after both rank graphs
READY. P selected complete-build/prefix0 on all12 requests. All eight math
requests and three of four reading requests install actual sparse banks.
Matched logs show442 successful predicted refreshes, no committed-Q fallback,
and zero target-model forwards for learned Q. The fourth short reading request
ends before bank installation. Six GPUs returned to idle after capture services
stopped. Capture serving/collector revision: `e7d02fc7c`.

The capture uses identical graph predictor, exact-degree16 centered four-head
CAGRA, cuVS25.10, itopk2048, graph-gated V bootstrap and generation settings
as the previous quality experiment. Only the collection bounds differ: math384,
reading128. These are training trajectories, not output-quality scores. Empty
gold-answer fields are never used as supervision.

## Controlled training protocol

Full-attention FP16 HF7B teacher labels **original Prompt K** and causal future
Q on two prefix sources: teacher-generated continuations on all48 training
questions, and observed sparse-D continuations on the12 collected questions.
Teacher continuation is EOS-bounded, up to384 math /128 reading tokens, with
repetition penalty1.05 matching target serving. The actual EOS token is retained;
no subsequent padded continuation is treated as a real teacher answer.
Boundaries0,4,32,64,128,256 cover early and later Decode. On observed prefixes,
the teacher provides a fresh counterfactual continuation, its full-attention Q,
and next-token correction labels. This is not sparse-D Q supervision or proof
that a teacher can repair every erroneous answer.

Each training record contains teacher, current six-layer Draft, and (on
observed prefixes) actual D future branches. Prompt K is prefetched at its
fixed original length and must remain equal across branches. Committed Decode
tokens are causal context and are excluded from the retrieval dataset.

Two arms start from identical75a0 weights, captures, seed, learning rates,
head/record/branch sample order, fresh AdamW and4000 steps:

- `all_positions`: token CE on spaced Prompt positions and future Decode.
- `decode_only`: token CE only on positions predicting the next real Decode
  token; exactly one label per horizon position.

Both retain normalized pre-RoPE Q MSE +0.1×teacher-K score KL/Top10 CE
+2×token CE, readout LR5e-5, student LR2e-6, decay0.01 and separate gradient
clipping1.0. Student parameters/body remain FP32 and readout/token projection
use FP16 autocast. Token logits use the student's actual serving repetition
processor; unique seen-token IDs keep identical forward results while avoiding
duplicate scatter gradients. Teacher token labels apply target penalty1.05.

Selection uses the **same25 calibration prefixes and old teacher references**
as the previous round, to keep recall comparisons meaningful. That calibration
teacher used penalty1.0; the reference is deliberately frozen rather than
redefined after training. No40-question answer is used for selection.

## Results

Teacher capture and training are running. Completed calibration, native,
latency and fixed output-quality results will be recorded here with raw evidence.
