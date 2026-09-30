# Six-layer Draft-Q on real Decode prefixes — CloudLab, 2026-09-30

## Calibration result

The six-layer architecture and rank-896 readout are unchanged. Adaptation
uses new causal question/answer trajectories rather than raw 512-token text
windows. On **eight disjoint calibration questions, 25 committed prefixes,
144,256 layer/head/position cases**, joint adaptation improves combined
exact Prompt-K Top-10 recall from **0.4606 to 0.5836**. This is a comparison
within the new protocol; it must not be compared directly with the earlier
eight-text-window 0.546 result.

| Method | True tokens + learned Q | Draft tokens + real target Q | Draft tokens + learned Q | Token agreement |
|---|---:|---:|---:|---:|
| Previous joint checkpoint | 0.6542 | 0.4452 | 0.4606 | 0.0% |
| Freeze Draft, train Q readout | 0.6942 | 0.4452 | 0.4452 | 0.0% |
| Joint causal token/Q adaptation | 0.6825 | 0.5759 | **0.5836** | **14.1%** |

The reference true-token + real-target-Q arm is 1.000. At the selected future
position 2, exact Top-16 candidates cover **70.94% / 69.43% / 80.77%** of the
reference Top-4 for baseline / frozen / joint. That coverage is averaged over
the 25 prefixes; Top-10 above is weighted by their actual query counts.
Worst-layer mean combined Top-10 improves from 0.2389 to 0.4497 with joint
adaptation. These are exact-score offline results, not native CAGRA recall or
answer-quality equivalence.

The frozen control improves Q accuracy when true tokens are given but fails
to improve the actual generated-token path. Old Draft samples on math and QA
chat prompts often emit Markdown image syntax such as `![](https://miro...`.
The teacher instead emits math reasoning or a short `FINAL:` answer. This
exposes a severe instruction/trajectory mismatch in the previous student,
whose joint training used raw code/document text windows. The joint arm's
token-only retrieval improves substantially, while 14.1% token agreement
remains low. Both token and Q errors remain open.

The joint checkpoint is selected using **only this calibration result**.
Its SHA256 is
`75a0c48a95d80ef8ccf6025e4c92e2ea2a028d9cf394e327fb69bec1da007efe`.
The old checkpoint remains unchanged at SHA256
`4b3fa87316c6b70d513fb32c3cdb79baf586e95046efbbfe3a51939cc8974215`.
Frozen-control SHA256:
`a830b8f66dc0964c7f39725646a37b3bab3758bcbccb7d964cac8fad17f6a744`.

## Data and causal labels

The fixed split contains 32 GSM8K train + 16 other HotpotQA dev questions for
training, and four + four separate calibration questions. Seed 202609301.
Every question in the frozen 40-question output benchmark and repository
ReAct source is excluded by question text. Calibration questions never enter
the optimizer. Selection checks context length and complete passages, without
using answers or model correctness. This excludes known question overlap,
not all pretraining contamination.

Training Prompt lengths are 78–153 GSM8K and 685–1529 HotpotQA. Calibration
lengths are 82–141 and 1330–1506. Math prompts and passage prompts use the same
chat template/instructions as the previous output benchmark. Dataset SHA256:
`f2547066bf57e930d8b558488f7359a72ece3b7e5dba07ad08543382066b3c27`.
Exact prompts, token IDs and source hashes are retained in `questions.json`.
GSM8K source is the [official train JSONL](https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl).
HotpotQA uses the same maintainer Parquet source described in the output
quality report, with different questions.

The FP16 7B teacher generates greedily with repetition penalty 1.0 and stops
at EOS or 192 output tokens. The experiment uses only actual generated
tokens, never a padded continuation after EOS. It takes boundaries
0/4/32/64/128 when at least three real future tokens remain; horizons are
three to eight. Short reading answers often have only a boundary-zero fixture.
There are **181 training records and 25 calibration records**.

For each boundary, committed input is original Prompt plus real teacher
Decode tokens before that boundary. Labels cover both the teacher continuation
and the old trained student's independent continuation on that same prefix.
Queries use target absolute RoPE. The search dataset contains **only original
Prompt K**; committed generated tokens are not added to it. The teacher's
next-token argmax also labels spaced Prompt positions and the future branch.
These are HF teacher trajectories, not instrumented online D trajectories.

Teacher prefill is computed at its original fixed Prompt length, and that
cache is used for subsequent Decode labels. An initial full-sequence replay
changed FP16 Prompt K rounding with sequence length and failed the consistency
gate. A second attempt used an obsolete DynamicCache indexing API. Both
attempts were rejected before training and are preserved as failure logs.
The final capture uses the inspected `cache.layers[layer].keys` API, validates
the `(28, Prompt length, 4, 128)` layout, and checks finite Q/K/logits and
unchanged Prompt K across all labeled branches.

## Controlled training

Both arms start from the same old joint checkpoint, use identical saved
records, random record/branch/head order, 1200 steps and fresh AdamW state.
Token trunk and resident readout remain FP32, as in serving; readout and
selected token logits use FP16 autocast. Frozen training caches the actual
FP32 six-block features. Joint training updates all six-block student
parameters while keeping the architecture unchanged.

Loss is normalized pre-RoPE Q MSE + 0.1 × retrieval loss + 2 × teacher-token
CE (CE is absent in the frozen control). Retrieval loss samples 16 layer/head
pairs per step over their **real Prompt K**. It combines teacher/student
listwise score KL and uniform cross-entropy on teacher Top-10 IDs, with a
teacher score standard-deviation scale. It retains Q MSE; this differs from
the earlier unsuccessful standalone pairwise loss. Readout LR is 5e-5,
student LR 2e-6, weight decay 0.01, separate gradient clipping 1.0.

Training took 33.8 s frozen and 99.1 s joint, excluding capture, loading and
evaluation. Teacher used node0 GPU0 and student GPU1. Validation rolls each
current student afresh through the same cached emission function used by the
serving adapter. The records are one capture/adaptation round; they are not
iterative on-policy retraining, and teacher-committed prefixes do not cover
every erroneous sparse-D trajectory.

## Native and serving validation

Native retrieval, cached inference timing and the unchanged 40-question
output test are recorded after completion in the follow-up evidence.

## Artifacts and reproduction

Scripts: `pvd_draft_q_decode_train.py`, `run_pvd_draft_q_decode_cloudlab.sh`,
`pvd_draft_q_decode_inspect.py`, `pvd_draft_q_decode_native.py`.
Raw calibration report, training log, token samples and failed captures are
in `pvd_draft_q_decode_cloudlab_20260930/`. Large captures (approximately
2 GiB) and checkpoints remain in node0's
`validation/draft-q-decode-20260930/`; serving copy is separate on node2.
Runtime training code is `99449165e`; native export code is `5efaa3f4a`.
No default serving mode or retrieval budget is changed.
