# Joint token and target-Q training: CloudLab V100S

## Controlled setup

The previous error decomposition showed that both token and Q prediction
matter. This step truncates Qwen2.5-0.5B-Instruct to its first six blocks,
**retains its LM head**, and jointly updates those blocks and a rank-896
target-Q readout. It uses the true Qwen2.5-7B-Instruct as teacher. The
starting readout is the previous diverse-corpus rank-896 checkpoint; all
training arms start from identical student/readout weights.

Eighty training Prompt windows contain 16 project-code windows and 64
HotpotQA/article/document/example windows. For each Prompt, the 7B and
initial six-block Draft independently generate eight greedy future tokens.
The teacher is then replayed on both branches. The saved data contain teacher
Q at the future positions, actual Prompt K, and the teacher's next-token
argmax at spaced Prompt positions and all future positions. The four arms
reuse **the same initial trajectories**, record/branch sampling order, 800
steps, optimizer and learning rates. The Draft is not rolled out again during
training; this is one round of on-policy capture followed by offline
training, not iterative DAgger.

The common loss is teacher next-token cross-entropy plus four times
normalized pre-RoPE target-Q MSE. The additional score loss is listwise KL
between teacher and student distributions over the **real Prompt K** at
eight sampled layer/Q-head pairs per step (temperature 2). This directly
supervises K ranking but does not replace the Q-MSE term. The tested score
weights are 0, 0.1, 0.5 and 2.0. The earlier, harmful pairwise ranking
experiment used a different loss and only the small original training set.

All validation is on eight held-out Prompts and eight greedy future positions
per Prompt. The model is rolled out after training, then the four-arm exact
Top-10 decomposition is recomputed on 50,176 layer/head/position cases.
There is no native CAGRA or PVD serving in this measurement.

## Results

| Six-layer student | Teacher-token agreement | Teacher tokens + predicted Q | Student tokens + real target Q | Student tokens + predicted Q |
| --- | ---: | ---: | ---: | ---: |
| Initial, before any joint update | 0.0% | 0.595 | 0.441 | 0.480 |
| Token CE + Q MSE | 46.9% | 0.655 | 0.589 | 0.536 |
| Token CE + Q MSE + score KL 0.1 | **50.0%** | **0.655** | **0.611** | **0.546** |
| Token CE + Q MSE + score KL 0.5 | 45.3% | 0.647 | 0.601 | 0.536 |
| Token CE + Q MSE + score KL 2.0 | 43.8% | 0.629 | 0.605 | 0.526 |

The target-token + real-target-Q reference is 1.000 in every row by
definition. Low-weight score distillation gave the best observed combined
recall, only **0.010** above the CE+Q control. The larger weights reduced
held-out combined recall. With only eight validation Prompts and one seed,
the 0.010 difference is a candidate signal, not a robust gain. The useful
large change comes from training token generation: the token-only arm rose
from 0.441 to 0.611 in the best run. Its predicted-Q arm remained much lower
at 0.546, so both error sources still require work.

The 800-step training times were 48.7, 54.0 and 57.6 seconds for score
weights 0, 0.1 and 2.0; these exclude capture, model load and final
evaluation. Peak allocated GPU memory in the runs was about 19.9 GiB with
the 7B teacher resident on the same V100S. These are experiment timings,
not Decode costs. The six-block model is still much less accurate than the
full 7B greedy token path.

## Decision

Keep the real target-Q probe in serving. The jointly trained six-block Draft
now generates meaningful token trajectories, but 0.546 combined exact-K
recall does not justify serving integration. The next step tests whether a
**causally available** target-model hidden state at the committed prefix
improves the readout on these same branches. That state is used only as a
conditioning input and contains no future-token information. Check whether
shuffling the state across Prompts removes any gain, and account for how D
would receive or retain the state before considering an online path.

The script is `benchmark/pvd_draft_q_multitask_probe.py`; raw reports are
`pvd_draft_q_multitask_ceq_cloudlab_20260929.json`,
`pvd_draft_q_multitask_ceqkl_cloudlab_20260929.json`,
`pvd_draft_q_multitask_ceqkl05_cloudlab_20260929.json`, and
`pvd_draft_q_multitask_ceqkl2_cloudlab_20260929.json`. The 1.6 GiB capture
tensor and model checkpoints remain under the CloudLab project's
`validation/draft-q-multitask-*` directories. The best tested checkpoint is
`validation/draft-q-multitask-ceqkl-20260929/trained.pt`.
