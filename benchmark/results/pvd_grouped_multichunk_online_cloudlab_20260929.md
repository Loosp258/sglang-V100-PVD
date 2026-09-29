# Two-head CAGRA with real Prefill chunks and dual-rank online timing

CloudLab V100S, 2026-09-29. This is a bounded experiment with a default-off
`--prompt-index-group-heads 2` V option. It does not establish production
retrieval quality or predictive D latency.

## Real-K/Q quality with repeated native extend

The Qwen2.5-7B-Instruct fixture performed actual 512-token Prefill calls for
the same 2155-token text sent by the online Gateway probe. It copied K after
each call (512×4 + 107), then replayed the five chunks to native CAGRA as
`build(512)` followed by four distinct `extend` calls. Each head used its
first 512 K rows to fix its mean; every later chunk used that same mean. The
two-head graph pairs the same layer's adjacent KV heads. A head-specific cuVS
bitset filters search, and all IDs are mapped back to original Prompt tokens.
Both arms use the same K, post-RoPE Q, float64 exact inner-product Top-10
oracle, graph degree 8, intermediate degree 16, and `itopk_size=2048`.
Each hypothetical V rank covers 28 layers × 2 local KV heads, with two Q
vectors per head (112 queries per rank).

| Gateway text / V heads | One-head mean / worst Top-10 recall | Two-head mean / worst | Invalid IDs | Isolated one-head build+extend | Two-head build+extend |
| --- | ---: | ---: | ---: | ---: | ---: |
| Case 0 / rank0 heads 0,1 | 1.000 / 1.000 | 1.000 / 1.000 | 0 / 0 | 10.728 s | 5.274 s |
| Case 1 / rank0 heads 0,1 | 1.000 / 1.000 | 1.000 / 1.000 | 0 / 0 | 17.818 s | 9.889 s |
| Case 0 / rank1 heads 2,3 | 1.000 / 1.000 | 0.998 / 0.950 | 0 / 0 | 18.342 s | 8.742 s |

The latter two fixtures were run concurrently on the two node0 GPUs, so
their absolute construction times include competition and are descriptive.
The isolated Case 0 rank0 median search time per two-query head call was
1.164 ms for one-head indexes and 1.253 ms for two-head indexes. Width 256
lost substantial recall after repeated extend; width 1024 still had a rank0
head at 0.75–0.80 against a 1.00 independent baseline. Width 2048 is the
tested quality setting for the online timing below. The fixture creates real
Prefill chunks but replays them after Prefill; its graph topology has the
same chunk order and native call boundaries, without P→V transport overlap.

Raw reports: `pvd_grouped_multichunk2155_rank0_itopk2048_cloudlab_20260929.json`,
`pvd_grouped_multichunk2155_rank0_trial1_itopk2048_cloudlab_20260929.json`,
`pvd_grouped_multichunk2155_rank1_itopk2048_cloudlab_20260929.json`.

## Two V ranks during live P→V chunk transfer

P=node0 GPU0/TP1; V=node1 GPUs0+1 in one process; D=node2 GPU1/TP1;
Gateway=node1. Both arms used the same model weights, V100S, `mlx5_0`,
cuVS 25.10, Mooncake, 512-token Prefill chunks, native CAGRA degree 8/16,
`itopk_size=2048`, 2155-token Prompt and two greedy output tokens. D ran in
full-KV mode for both arms, so client time measures construction interference
with V→D delivery; the client did not depend on sparse index search. P and V
were restarted for each arm, while D and Gateway stayed up. Order was
one-head→two-head for Case 0, then two-head→one-head for Case 1. Each Case's
prompt and output hashes matched across arms. No cached P tokens were used.

| Case | One-head graphs/rank | Two-head graphs/rank | One-head client | Two-head client | Two-head reduction | Concurrent first-512 build, slowest rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 56 | 28 | 19.807 s | 10.376 s | 9.431 s (47.6%) | 18.969 → 9.727 s |
| 1 | 56 | 28 | 19.459 s | 10.580 s | 8.879 s (45.6%) | 18.778 → 9.888 s |

Both V ranks logged a 512-row provisional build and eventual READY in every
arm; no graph fallback or invalid completion occurred. In all four requests,
the client finished roughly 0.5–0.8 s before both graphs became READY. The
client improvement is therefore associated with less V build interference
with full-KV fan-in, not an earlier sparse query. The real Prefill chunks
arrived while the first native build ran. The online manager then coalesced
the remaining available KV into one `extend(1643)` per graph. It did **not**
execute four separate online extends; that exact operation sequence was
tested by the quality replay above. This distinction matters for future
scheduling claims.

The paired timings are two descriptive requests, not a confidence interval.
Predictive D, V search through the grouped production path, sustained load,
peak memory, more prompts and cancellation remain unverified. The feature
stays opt-in. The V100S manager acceptance did build+two extends, filtered
search for both heads, score restoration, and disposal with zero retained
native bytes. All experiment-owned P/V/D/Gateway processes were stopped;
no PVD listeners or GPU compute processes remained on the three nodes.
The compact paired data, including input/output hashes and per-rank native
build times, is in `pvd_grouped_multichunk_online_cloudlab_20260929.json`.

Sources: `test/registered/disaggregation/run_pvd_qwen_grouped_cagra_gpu.py`,
`test/registered/disaggregation/run_pvd_chunked_index_gpu.py`,
`benchmark/pvd_chunked_online_probe.py`. Raw online V logs and client JSONL
are preserved as `pvd_v_group{1a,2a,2b,1b}_online_cloudlab_20260929.log`
and `pvd_client_group{1a,2a,2b,1b}_online_cloudlab_20260929.jsonl`.
