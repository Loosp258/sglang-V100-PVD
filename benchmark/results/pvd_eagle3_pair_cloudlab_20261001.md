# One-GPU Qwen2.5-7B / dedicated EAGLE3 probe, 2026-10-01

## Result

The pinned dedicated EAGLE3 checkpoint runs in FP16 with its target on one
32 GB V100S. Its greedy eight-token lookahead improves aggregate token
agreement over the current six-layer checkpoint on this bounded calibration
sample, but the gain comes from math; reading agreement regresses sharply.
Do not replace the serving predictor on this evidence.

| Token-only measure | Existing six-layer Draft | Dedicated EAGLE3 |
| --- | ---: | ---: |
| All scored positions | 32/161 = 19.88% | 52/161 = 32.30% |
| Math positions | 28/128 = 21.88% | 51/128 = 39.84% |
| Reading positions | 4/33 = 12.12% | 1/33 = 3.03% |
| First predicted position | 14/23 = 60.87% | 17/23 = 73.91% |
| Mean consecutive correct prefix, maximum 8 | 1.174 tokens | 1.957 tokens |

These are exact greedy token agreements, not answer accuracy, CAGRA recall or
speculative acceptance rates. Aggregate results are position weighted, with
128 math and only 33 reading positions. The known target-supplied root token
is excluded. Four calibration questions per task produce 23 eligible prefixes
at Decode boundaries 0, 4, 32 and 64; EOS removes unavailable later boundaries
and shortens the scoring horizon. The prefix lengths are 82–1510 tokens.

## Single-GPU resources and identities

Only node0 GPU 1 was exposed (`CUDA_VISIBLE_DEVICES=1`, exactly one device
reported by PyTorch). GPU 0 was initially idle but acquired an unrelated P
service during preparation, so the experiment moved to GPU 1 without stopping
that service. Both GPUs were idle at the beginning of the final measured run;
GPU 1 returned to zero allocated process memory after the probe exited.

- Hardware: Tesla V100S-PCIE-32GB; Torch 2.9.1+cu128.
- Target: existing local Qwen2.5-7B-Instruct, FP16, HF SDPA.
- Dedicated Draft: `thoughtworks/Qwen2.5-7B-Instruct-Eagle3`, revision
  `ff17dda64a036cf5bd7bc56c0ab728325f1c0d0b`.
- Dedicated weight SHA256:
  `f77665495c620c17d05d6672444094844866e2a27e545d6837e8b107465ddfd8`.
- Actual dedicated config: one layer, hidden 3584, 28 Q heads, **7 KV heads**,
  intermediate 14336, full vocab 152064, draft vocab 32000. The publisher's
  model card describes 4 KV heads; tensors and config both establish 7.
- The downloaded weights contain 358,758,400 BF16 floating parameters;
  inference converts them to FP16 and shares the target embedding omitted
  from the checkpoint. Every required tensor loads, with only the expected
  embedding omission; converted weights and produced logits are finite.
- Author inference source: SafeAILab/EAGLE commit
  `cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b`, unchanged eager `cnets.Model`.
- Old Draft: trained six-block token trunk from checkpoint
  `75a0c48a95d80ef8ccf6025e4c92e2ea2a028d9cf394e327fb69bec1da007efe`;
  FP32 eager trunk as in current serving, with no Q readout executed.
- Shared greedy repetition penalty: 1.05 for target and both Draft arms.
  This controlled token comparison differs from the old Draft's usual 1.1
  policy, so prior calibration agreement figures are not directly comparable.
- Frozen calibration question SHA256:
  `f2547066bf57e930d8b558488f7359a72ece3b7e5dba07ad08543382066b3c27`.
  No checkpoint is trained or selected on the 40 output-benchmark answers.

PyTorch peak allocated inference memory, including all three resident models
and the long shape, was **18,369.8 MiB (17.94 GiB)**; peak reserved was
19,324 MiB (18.87 GiB). This comfortably fits one 32 GB card. Model loading
was completed before resetting the reported inference peak.

## Alignment and compatibility gates

The published post-block target features from blocks `{1,13,24}` map to HF
hidden-state slots `{2,14,25}`. For a prefix of N tokens, the Draft receives
N target feature rows paired with `prefix[1:] + known_root`. It predicts the
token after that root. The six-layer control receives the same complete
committed prefix including the root and predicts that same next position.

The checkpoint's `d2t + arange(32000)` mapping exactly equals the nonzero
`t2d` IDs. An independent call to the author's width-one, eight-step tree API
returns the same eight greedy tokens as the probe, plus its explicit root.
That check uses repetition penalty 1.0 because the author tree API does not
apply a repetition processor. See `alignment.json` for both token lists.

The first loading attempt failed because Transformers 5 normalized missing
`rope_scaling` to a dictionary the older author code could not interpret.
The adapter restores `None` and theta 1e6 directly from the pinned checkpoint
config; it does not change the attention algorithm or checkpoint weights.
The failed log and exit code are retained. The corrected final run exits 0.

## Warm token-only timing

Each calibration prefix has one excluded warmup per arm and three measured
repeats, with alternating arm order and CUDA synchronization at boundaries.
Across the different calibration lengths, median total Draft time is 18.23 ms
for EAGLE3 and 46.16 ms for the old trunk. Those pooled timings are not a
fixed-length serving latency.

The additional 2155-token synthetic Case 40 fixture is the same repeated
`EEFTRITON` shape used in earlier latency probes. It supplies one known root
and measures eight genuinely predicted tokens; it is not a quality question.
Five warmed, alternating-order samples give:

| Component, median | Six-layer trunk | Dedicated EAGLE3 |
| --- | ---: | ---: |
| Draft prefix processing | 56.07 ms | 21.28 ms |
| Eight-token prediction stage | 39.99 ms | 15.67 ms |
| Total Draft computation | 96.28 ms | 36.89 ms |

**These timings assume the target features already exist.** EAGLE3 consumes
privileged target features; the control consumes token IDs. They therefore
measure conditional computation for the two architectures, not two equal
standalone inputs. The warm target Prefill that produced the 2155 feature
rows took 737.38 ms in this probe, plus 0.595 ms for fusion/checking. Re-running
that target Prefill for each refresh would overwhelm the Draft saving.

The float16 feature payload for a 2155-token initial prefix is
`2155 × 3 × 3584 × 2 = 46,341,120` bytes. Subsequent committed features could
be appended to a retained Draft prefix cache, but neither that cache lifecycle
nor P→D feature handoff is implemented here. The experiment has no new P→D KV
transfer and makes no transfer-latency claim.

Both arms exclude all-layer Q readout, CAGRA search, KV transport, target
verification and serving contention. The old 107.7 ms figure included Q
readout and is not the denominator for this token-only comparison.

## Next gate

Investigate the reading token regression before a full-path replacement.
Then evaluate Draft-generated tokens with the existing real target-Q probe
to isolate token effects on retrieval. Reuse actual target features instead
of re-running its Prefill, and account for initial feature transport and
sparse-D feature drift. All-layer predicted Q remains a separate task.

Raw evidence is in `pvd_eagle3_pair_cloudlab_20261001/`: final report and rows,
2155 timing samples, alignment proof, pinned checkpoint config/card, failed
compatibility logs and an excluded pilot. The 717.8 MB weights remain on
CloudLab and are not checked into Git. The final probe source SHA256 is
`a0b9008b0736433395eb7d1a92c50d4209a5e2bfe7c618533d0ee1a8ca67aea1`.
