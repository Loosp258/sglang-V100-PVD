# Causal target-prefix state for Draft-Q: CloudLab probe

## Method

The best step-2 six-layer, token-generating Draft and rank-896 target-Q
readout are frozen. A zero-initialized rank-128 residual adapter sees two
inputs: the last *committed Prompt* target hidden state and the future-token
Draft's last block feature. It predicts a Q correction for all 28 layers.
The adapter is trained for 800 steps on the same 80 Prompt/teacher-trajectory
captures as step 2, with normalized Q MSE and score KL weight 0.1. The eight
held-out Prompts are generated once from the frozen Draft, so all evaluation
arms share exactly the same tokens, true Prompt K and true target Q.

Two causal target states were tried: the hidden state after layer 12, and the
final normalized hidden state. The latter was captured with SDPA: an FP16
eager full-Prompt forward produced nonfinite final hidden values on six of
the eight validation Prompts, while a standalone SDPA check was finite and
the full SDPA capture passed finiteness checks for all 80 training and eight
validation Prompts. This numerical issue affected the hidden feature only;
the earlier Q/K captures were finite. The zero-initialized adapter is required
to reproduce the frozen Draft's known 0.546 combined Top-10 baseline before
training; both runs passed that gate.

The control shuffles complete target hidden vectors between validation
Prompts without changing tokens, Prompt K, the trained adapter or Draft
features. A zero-hidden control checks whether the adapter relies on any
target signal. All numbers are offline exact Prompt-K Top-10 overlap with the
target-model greedy trajectory; no native CAGRA or online P/V/D timing.

## Results

| Target state supplied to adapter | Target tokens + predicted Q | Draft tokens + predicted Q |
| --- | ---: | ---: |
| Zero-initialized adapter / frozen Draft baseline | 0.655 | 0.546 |
| Layer 12, correct Prompt after training | 0.660 | 0.549 |
| Layer 12, shuffled Prompt after training | 0.659 | **0.550** |
| Layer 12, zero target state after training | 0.658 | 0.547 |
| Final layer, correct Prompt after training | 0.656 | 0.547 |
| Final layer, shuffled Prompt after training | 0.656 | 0.548 |
| Final layer, zero target state after training | 0.655 | 0.546 |

The exact-target-Q arm on Draft tokens remains 0.611 in every row, because
the Draft token path is frozen. Both correct target-state variants gain less
than 0.003 combined recall, and shuffled states perform at least as well.
The small change is therefore not evidence that the adapter extracted useful
Prompt-specific target information. An additional Draft-feature mapping can
explain it. This result only tests one lightweight additive residual and a
single committed-prefix state; it does not rule out a differently trained
EAGLE-style model with multi-layer features and many more trajectories.

## Decision

Do not add hidden-state transfer or target-state retention to serving for
this adapter. D does not currently receive P's final Prefill hidden state;
later D refreshes may have a recently computed target state, but exploiting
it would need an explicit ownership/lifetime path. With no demonstrated
quality gain, that implementation cost is not justified yet. Proceed to a
bounded candidate-allocation experiment using the validated step-2 model.

The script is `benchmark/pvd_draft_q_prefix_state_probe.py`; raw reports are
`pvd_draft_q_prefix_state_l12_cloudlab_20260930.json` and
`pvd_draft_q_prefix_state_final_cloudlab_20260930.json`. The large validation
capture and small adapters remain under CloudLab
`validation/draft-q-prefix-state-20260930/` and
`validation/draft-q-final-state-20260930/`.
