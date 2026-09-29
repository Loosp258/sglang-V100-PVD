# Draft-token versus Draft-Q error on CloudLab

## Question and method

Which error currently matters more for predicting future Prompt-K retrieval:
the small Draft's generated token path or the six-layer target-Q readout?

On an isolated V100S, the real Qwen2.5-7B-Instruct and full 24-layer
Qwen2.5-0.5B-Instruct independently generate eight greedy future tokens from
each identical Prompt. The frozen first-six-layer Draft readout is the
previously trained rank-896 model. Both future-token branches are replayed
through the real 7B teacher to capture its actual target-space Q. The readout
also predicts Q on both branches. All four Q streams search **the same real
Prompt K** with masked exact dot products and Top-10. The target-token plus
target-Q arm defines the reference token IDs and reference neighbor set.

The eight held-out Prompts are the four earlier project/random fixtures and
four disjoint natural-text windows. One fixture has 1,024 Prompt tokens; the
others have 512. Each arm has 50,176 query/layer/head/position cases: eight
future positions, all 28 layers and all 28 Q heads on eight Prompts. This
experiment uses greedy generation and teacher-forced branch replay, not the
PVD serving scheduler or approximate CAGRA. These cases are correlated and
should not be read as a production confidence interval.

## Results

| Future-token branch | Retrieval Q | Mean overlap with reference Top-10 |
| --- | --- | ---: |
| 7B target tokens | Real 7B Q | 1.000 (definition) |
| 7B target tokens | Six-layer Draft readout Q | 0.595 |
| 0.5B Draft tokens | Real 7B Q replayed on Draft branch | 0.462 |
| 0.5B Draft tokens | Six-layer Draft readout Q | 0.463 |

The full 0.5B Draft matched the 7B's greedy token at only 2/64 sampled future
positions. With the *same Draft-token branch* as the reference instead, the
readout Q's overlap with real target Q is 0.621. Thus the readout has a
substantial error of its own, but using real target Q on the wrong generated
branch does not recover the target-token reference neighbors. The average
token-branch arm falls from 0.595 at future position 0 to 0.363 at position
7; this is consistent with autoregressive branch divergence. The readout-Q
arm on target tokens ranges 0.573–0.636 across these positions. This
decomposition compares neighbor identity to the 7B greedy trajectory; it
does not measure answer quality or prove which misses affect attention most.

The previous in-Prompt rank-896 readout result was 0.652/0.663 on original/
natural fixtures. The 0.595 value here is lower because these are unseen
post-Prompt positions on newly generated trajectories, not the prior
in-Prompt teacher-forced positions. Do not combine those numbers as a single
quality estimate.

## Consequence for the next step

Train and validate a token-generating Draft jointly with target-Q retrieval
supervision on actual Draft trajectories. Keep all four arms above as the
evaluation gate, plus token agreement, native CAGRA recall and eventual
Decode output. Improving Q alone cannot repair a different future-token
trajectory. Training must preserve the token head; the earlier Q-only trunk
adaptation did not test this.

The reproducible script is `benchmark/pvd_draft_q_error_decomposition.py`.
The full per-record, layer, head and future-position data are in
`pvd_draft_q_error_decomposition_cloudlab_20260929.json`. Readout weights
remain on CloudLab under `validation/draft-q-corpus-r896-20260929/readout.pt`.
