# Chunked P→V KV / CAGRA online probe — CloudLab V100S, 2026-09-29

## Scope and controls

This is an experimental two-pair online check, not a production speedup claim.
P was node0 GPU0, V was node1 GPUs0/1, D was node2 GPU1, and the Gateway was
node1. Both arms used Qwen2.5-7B-Instruct from
`/proj/llm-course-PG0/Yizhzhu-node0-sglang-pvd/models/Qwen2.5-7B-Instruct`,
TP1 P/D, the same model files,
V100S hardware, `mlx5_0` Mooncake path, two V shards, cuVS 25.10 from the
isolated environment, native CAGRA with graph degree 8 / intermediate degree
16, a 512-token Prefill chunk size, 2155-token Gateway prompts and two output
tokens. Each arm restarted P and V. D and Gateway stayed on the same processes;
the trial order was complete→chunked for prompt 0, then chunked→complete for
prompt 1. The only P/V feature difference within a pair was the chunked upload
flag. The chunked arm included an event-driven index kick in the shard HTTP
handler, but group-mode coordinator commits bypassed that handler in these
paired trials; the follow-up below corrects this. CAGRA itself was unchanged.

The request inputs were fixed per trial and deliberately differed near the
start to avoid whole-Prompt cache reuse. Gateway metadata confirmed 2155
Prompt and two completion tokens in all four requests. Both arms produced
the same output SHA-256 for each matched request. P logged five Prefill
compute chunks (512×4 + 107) and zero cached tokens. The graph READY time is
the later of the two V-rank log timestamps, since both shards must be indexed.

| Trial | Mode | Client wall | Last V shard commit → both READY | Client start → both READY |
| --- | --- | ---: | ---: | ---: |
| 0 | complete | 1.755 s | 23.192 s | 24.730 s |
| 0 | chunked | 5.891 s | 19.580 s | 22.658 s |
| 1 | chunked | 7.278 s | 21.080 s | 24.828 s |
| 1 | complete | 1.815 s | 22.241 s | 23.837 s |

The chunked arm shortened the time **after the final V commit** by 3.612 s
and 1.161 s in these two trials. Measured from client request start, the
index was 2.072 s earlier in trial 0 but 0.991 s later in trial 1. Client
completion was 4.136 s and 5.463 s slower. Therefore the measured version
has not earned default enablement. These are two descriptive requests, not
confidence intervals. D can complete via full KV before the index is READY,
so client completion is not an index-ready measurement.

## What actually overlapped

In trial 0, rank0 committed a 512-page prefix at 19:20:44.795 UTC, finished
its 56-head provisional build at 19:20:55.860, then extended to 2155 rows
and became READY at 19:21:07.127. The complete Entry had committed at
19:20:47.547, so its prefix build overlapped arrival of later KV. Rank1's
event task did not start before the final chunk and built the full 2155 rows.

Trial 1 still built a rank0 prefix (1536 rows), extended the remaining 619 rows,
and became READY at 19:25:26.996. Rank1 again started with all 2155 rows.
These trials exposed a missed trigger: in one-process group mode the coordinator
uses `LocalShardClient`, so chunk commits bypass the shard HTTP handler's
event-driven index kick. The group reaper progresses rank0 and rank1 in order;
it happened to start rank0 first. This explains why these logs show early
native work on only one rank. Final shard commit remained independent of a
running build: both final commits in trial 1 returned at 19:25:05.916 while
rank0's prefix build ended at 19:25:15.788.

## Follow-up: trigger both local ranks

`LocalShardClient` now schedules the same background progress after each
successful nonfinal chunk and final shard commit. A focused test confirms
both local shards receive an early kick. One CloudLab diagnostic request used
the same 2155-token prompt, P/V topology, cuVS 25.10, and 512-token chunk
size, but D was relaunched in full mode; it is not another matched AB pair.
The request output hash matched trial 0 and the client wall time was 19.448 s.

The first chunk committed on ranks 0/1 at 06:29:10.597/10.609 UTC. Both
56-head provisional builds ran concurrently, each for about 18.55 s, and
finished at 06:29:29.153. Both final shard commits returned at 06:29:15.122;
both graphs extended to 2155 rows and were READY by 06:29:30.212. Thus both
ranks can use the incremental method. However, V→D fan-in preflight did not
return until 06:29:29.156, immediately after both builds ended, and client
completion was delayed. The logs show a serious same-process service
interference; they do not identify whether the cause is the Python GIL,
native CPU threads, GPU work, or another shared resource. Keep this path
opt-in until construction can be scheduled without stalling fan-in.

The 2048+107-token setting was also tried. Its approximately 0.6-second
first/final arrival gap was shorter than the original 1-second maintenance
interval, so without an event kick V usually built only after all KV arrived.
On a later run, the reaper happened to start one rank's 2048-row prefix before
the final commit, but that prefix build took about 14 seconds and did not give
a reliable end-to-end advantage. This motivated the 512-token matched
comparison above.

## Correctness and remaining gates

The P→V batch PUTs, independent chunk identities, exact byte/terminal checks,
complete `STORED` commits, native extension on one V rank and eventual
searchable graphs on both ranks all ran through the real Gateway/P/V/D/Mooncake stack. The
isolated native V100S manager acceptance additionally built 512 rows,
extended 512 rows, searched and disposed with zero retained native bytes.
Unit integration verified component-major reconstruction and that provisional
graphs remain unsearchable until complete KV is readable.

The online probe is TP1 only, with one request at a time, two output tokens,
one model and one prompt length. It does not establish recall equivalence:
native CAGRA full and extended graphs can return different approximate
neighbors. TP2, overlapping requests, cancellation during a native build,
unknown Mooncake completion under load and long-run memory pressure still
need an acceptance pass. The feature remains off by default.

CloudLab raw files under each node's `$SGLANG_PVD_ROOT/validation/logs/`:
`base512-probe0.jsonl`, `stream512-probe0.jsonl`,
`stream512b-probe1.jsonl`, `base512b-probe1.jsonl` on V; matching
`v-{base512,stream512,stream512b,base512b}.log` on V and
`p-{base512,stream512,stream512b,base512b}.log` on P. The follow-up has
`stream512c-probe0.jsonl`, `v-stream512c.log`, `p-stream512c.log`, and
`d-stream512c.log` on the respective nodes. The experiments used
isolated `validation/pvd-stream-online-20260929` worktrees and all four role
process groups were stopped after measurement.
