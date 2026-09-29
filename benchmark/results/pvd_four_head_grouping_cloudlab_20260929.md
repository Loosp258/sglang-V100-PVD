# Four-head grouped CAGRA trial on CloudLab (2026-09-29)

## Setup

The experimental `--prompt-index-group-heads 4` mode groups two adjacent
layers and both local KV heads into one native CAGRA graph. Each head is
centered separately; a native bitset filters search to its rows. The current
Qwen2.5-7B-Instruct layout has 28 layers and two local KV heads per rank, so
each rank builds 14 graphs rather than 28 with two-head grouping. The option
is disabled by default and requires chunked CAGRA upload.

Both online arms used the same P/V/D/Gateway nodes, V100S GPUs, model weights,
Mooncake rail, cuVS 25.10, native graph degree 8/16, `itopk_size=2048`,
512-token Prefill chunks, and 2156-token Gateway prompts. P and V restarted
for each grouping arm; D in full-KV mode and Gateway remained up. Each request
generated two greedy tokens with matching input and output hashes across arms.
The order was two-head then four-head for Cases 40/41, followed by four-head
then two-head for Cases 42/43. The last two four-head requests used the same
P/V instance, so this is a small paired feasibility measurement rather than a
latency distribution.

## Online client completion

| Case | Two-head (s) | Four-head (s) | Reduction (s) |
| --- | ---: | ---: | ---: |
| 40 | 10.524 | 6.186 | 4.338 |
| 41 | 10.083 | 5.822 | 4.261 |
| 42 | 10.769 | 4.901 | 5.868 |
| 43 | 10.378 | 5.678 | 4.700 |

Median completion fell from 10.451 to 5.750 s, a 4.701 s (45.0%) reduction.
The slowest rank's initial 512-token graph build was 8.78–10.00 s for two-head
grouping and 4.31–5.39 s for four-head grouping. Both ranks performed native
`build(512)` then `extend(1644)` and reached READY on every request, with no
fallback. As in earlier full-KV experiments, the client completed roughly
0.6–0.8 s before the two graphs were READY. This client benefit measures less
contention with V-to-D full-KV delivery; full-KV D did not use sparse search.

## Real K/Q recall on the same Case 40 prompt

The offline replay loaded the real Qwen weights, copied the actual per-layer K,
used two real Q positions per head, and ran `build(512)` plus four native
`extend` calls for 512/512/512/512/108 rows. The Top-10 oracle used exact
float64 scores for each head. The operation order was reversed between ranks.

| Rank | Grouping | Native build (s) | Four extends (s) | Mean Top-10 recall | Worst head | Invalid IDs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2 | 8.779 | 2.221 | 1.0000 | 1.00 | 0 |
| 0 | 4 | 4.305 | 1.017 | 0.9982 | 0.95 | 0 |
| 1 | 2 | 9.039 | 2.059 | 0.9991 | 0.95 | 0 |
| 1 | 4 | 4.313 | 1.222 | 0.9946 | 0.90 | 0 |

Four-head grouping saves approximately half the isolated build+extend time,
but loses exact neighbors on some heads. This is a quality regression even
though the mean recall remains high. The online full-KV test cannot establish
the effect on predictive Decode answers. Keep the mode opt-in pending a
predictive workload and a retrieval-quality threshold decision.

The recall misses in this one Prompt were concentrated in layer pairs
12/13 and 20/21 on rank 0, and 14/15, 16/17, 22/23, and 24/25 on rank 1.
An adaptive layout could keep four-head graphs for pairs with adequate recall
and use two-head graphs for difficult pairs. That would build 16 and 18
graphs on the respective ranks for this Prompt. These pairs are observations
from one Prompt, not a fixed production policy; more Q/K samples are needed.

CPU tests covered both ranks, incremental publish and full-build fallback.
The V100S native manager test covered build, two extends, filtered search for
all four heads, `search_many`, and disposal with zero retained native bytes.

CloudLab raw logs: node1 `validation/logs/v-group{2a,4a,2b}_20260929.log`
and `client-group{2a,4a,4b,2b}_20260929.jsonl`; node0
`validation/logs/group4-case40-rank{0,1}-20260929.log`.
