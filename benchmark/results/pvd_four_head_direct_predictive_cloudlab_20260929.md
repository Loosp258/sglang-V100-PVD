# Four-head graph with direct P→D bootstrap and predictive Decode

CloudLab, 2026-09-29. P=node0 GPU0, V=node1 GPUs0/1, D=node2 GPU1,
Gateway=node1. The request uses Qwen2.5-7B-Instruct, 2156 Prompt tokens,
512-token Prefill chunks, six greedy output tokens, P→D direct initial KV,
and native cuVS 25.10 CAGRA with graph degree 8/16 and `itopk_size=2048`.
Both V ranks build from 512 rows and extend to 2156. Only V's grouped-head
setting changes between arms: 28 two-head graphs or 14 four-head graphs per
rank. P and V restart for each arm; D and Gateway stay up. Case 50 warms
each P/V instance and is excluded. Cases 51–54 run four-head then two-head;
Cases 55–58 run two-head then four-head. All cases have zero cached Prefill
tokens and matching Prompt/output hashes between arms.

`D first search` is the `search_seconds` field of `PVD refresh ready` in D's
log. It starts after Q capture and ends after all routed V shard searches
return. It **includes the retryable index-not-ready responses and the wait for
both V ranks to reach READY**, plus actual search and HTTP time. `D refresh`
adds Q capture, selection union and KV delivery. `Client complete` is the
Gateway SSE response through the sixth token.

| Case | D first search, 2 heads (s) | D first search, 4 heads (s) | D refresh, 2 heads (s) | D refresh, 4 heads (s) | Client complete, 2 heads (s) | Client complete, 4 heads (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 51 | 9.583 | 4.808 | 10.732 | 5.931 | 11.798 | 7.023 |
| 52 | 8.820 | 4.288 | 9.934 | 5.433 | 10.999 | 6.487 |
| 53 | 8.124 | 3.695 | 9.236 | 4.824 | 10.316 | 5.904 |
| 54 | 8.173 | 3.679 | 9.295 | 4.827 | 10.354 | 5.883 |
| 55 | 8.409 | 4.779 | 9.569 | 5.903 | 10.656 | 6.976 |
| 56 | 8.910 | 4.133 | 10.035 | 5.250 | 11.102 | 6.286 |
| 57 | 9.278 | 3.739 | 10.400 | 4.869 | 11.469 | 5.929 |
| 58 | 8.378 | 4.094 | 9.535 | 5.229 | 10.606 | 6.285 |

Across these eight pairs, the median D first search falls from 8.615 to
4.114 s, a 4.501 s (52.2%) reduction. The median full D refresh falls from
9.752 to 5.240 s. Median client completion falls from 10.828 to 6.285 s.
The first streamed event is 1.530 s with two-head and 1.552 s with four-head
grouping, consistent with direct initial KV allowing Decode to begin before
V's graph is READY. Four-head first-search time ranges from 3.679 to 4.808 s.

Every measured request had HTTP 200, six SSE events, six output tokens,
matching final-text hashes, a direct P→D terminal proof, both V ranks READY,
an initial V `index_not_ready` refusal, and successful subsequent search.
There were no graph fallbacks. The earlier real-K/Q trial found a worst-head
Top-10 recall of 0.90 for four-head grouping versus 0.95 for two-head on
rank 1, so the speedup does not settle retrieval quality. These are eight
bounded sequential requests, not a throughput or tail-latency study.

Raw logs on CloudLab: node0 `validation/logs/p-g{4direct,2direct,4direct_reverse}_20260929.log`;
node1 `validation/logs/v-g{4direct,2direct,4direct_reverse}_20260929.log`
and `client-g{4direct-measured,2direct-measured,2direct-reverse,4direct-reverse}_20260929.jsonl`;
node2 `validation/logs/d-g4direct_20260929.log`. The probe is
`benchmark/pvd_direct_stream_probe.py`.
