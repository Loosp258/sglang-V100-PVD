# Codex handoff — PVD disaggregation (2026-09-27)

This is the starting point for a **new Codex session**. Read this file, then
inspect the current checkout; do not infer success from an old conversation or
from historical sections of other documents. The final performance goal is
**not achieved**. Do not push to GitHub unless the user explicitly changes the
current instruction: make a **local commit after each verified step; no push**
until the final goal is met and the user permits it.
Sections 1–5 preserve the `ecf1fc247` handoff snapshot; section 6 records
later verified checkpoints and remaining work.

## 1. Repository and preservation rules

- Local workspace: `D:\code\sglang-V100-PVD`; branch `pvd-disaggregation`.
- The last code commit before this handoff is `4fc27b27b`; the handoff was
  committed locally as `35350660d`. `origin/pvd-disaggregation=05d7a2afe`;
  **57 local commits are ahead** after this accuracy correction, none pushed
  in this work. `origin` is
  `https://github.com/Loosp258/sglang-V100-PVD.git` (the user's fork).
- Recheck `git status`/`git log` at the start of a new session; this document
  was committed separately from the code. Do not stage unrelated work.
- Pre-existing, unrelated dirty tracked files belong to the user and must be
  preserved: `pvd/client.py`, `pvd/transfer_authorization.py`,
  `pvd/transfer_lifecycle.py`, `entrypoints/openai/protocol.py`,
  `mem_cache/allocator/token.py`, `model_executor/model_runner.py`, and
  `test_pvd_core.py`. They were not staged by this run. Numerous untracked
  `.pvd-validation-*.bundle` files and `Claude outputs/` also exist; preserve
  them. `.pvd-validation-88c72ed79-sidecar.bundle` was created in this run.
- On Windows, `.git/index.lock` needs an escalated Git operation; stage only
  explicit paths with `git -c core.autocrlf=input add -- <paths>` to avoid
  CRLF-only changes. Use `apply_patch` for file edits.

## 2. Product goal and non-negotiable design

PVD separates Prefill (P), vector/KV service (V), and Decode (D). The Router
chooses P/V/D and conveys the selected V identity to P and D. P writes the
complete Prompt KV to V by RDMA; D first obtains the full Prompt KV at final
waiting-queue admission. Thereafter each request refreshes on its **own**
M-token clock. A newly admitted request does **not** force older batch members
to refresh or reset their clocks. D-generated token KV remains local to D.

The intended advanced path predicts a bounded continuation with an independent
small draft, captures **target-model post-RoPE Q**, searches V's target-K
index, selects a sparse Prompt-KV working set (GQA Q heads sharing one KV head
use a bounded deduplicated union), transfers that set to D, and installs it
before the request's refresh boundary. Late prediction must never be accepted
as current truth: at the boundary, wait for the applicable result or use a
query from the current committed prefix under the explicit fallback policy.
Approximate retrieval is allowed, but its quality must be measured. Native
SGLang speculative generation remains prohibited under PVD; draft output is
prediction-only and never committed as user-visible tokens. V's full CAGRA
search and long-context/concurrency behavior must be validated on V100S.

The final target is **safe, numerically/quality-acceptable, real three-node
serving with a demonstrable end-to-end gain over the full-Prompt-KV baseline**,
including long contexts and concurrency. “The network looks local” is a
performance objective, not a property already proven. P/D node and GPU counts
remain an architectural scalability objective; the tested predictive CUDA
path here is TP1 P and TP1 D, with V two rank-sharded GPUs. Never generalize
this result to arbitrary P:D TP ratios or non-V100S hardware.

Related specifications/evidence: `docs/PVD_Current_Readiness_CN_EN.md`,
`docs/PVD_CloudLab_2026-09-27_Validation_CN_EN.md`,
`docs/PVD_Independent_Probe_Lane_CN_EN.md`, and
`docs/PVD_CAGRA_Acceptance_CN_EN.md`. The **opening status line** of the
independent-lane document is stale (“not integrated with serving”); its later
checkpoint paragraphs describe the newer opt-in serving integration. Update
that inconsistency in a future documentation step.

## 3. Current implementation — follow the request path

1. Existing PVD Router/Coordinator and P/V/D transfer/Entry/Delivery flow
   remain. V owns rank-sharded stored Prompt KV and exact/CAGRA-capable index;
   selections are logical token/page IDs, not exposed GPU addresses. Sparse
   delivery uses the existing bounded V→D RDMA/Mooncake path. Mooncake
   0.3.13.post1 and `MC_DISABLE_METACACHE=1` are still part of the validated
   transfer policy; no double-rail validation is claimed on this lease.
2. `probe_lane_protocol.py`, `probe_lane_wire.py`, `probe_lane_unix.py`, and
   `probe_lane_identity.py` implement bounded immutable ticket/reply framing,
   nonce/replay/deadline/Entry/prefix/model checks, checkpoint-content hashes,
   owner-private Unix socket, exact Linux peer PID/UID, and host-reply budgets.
   The sidecar reply contains CPU FP32 Q only; it never sends a Req, GPU pool,
   MR address or rkey.
3. `probe_lane_model.py` runs the private target+draft CUDA pipeline and
   returns only the requested post-RoPE Q rows. `probe_lane_routing.py`
   derives a rectangular layer × Q-head ticket from trusted V search routes.
   `probe_search.py` then submits verified Q to the existing V search/selection
   path; `cpu_prefetch_request.py`/`decode_refresh.py` carry it through sparse
   delivery and installation.
4. `cuda_refresh_driver.py` can release the formal target-forward arbiter while
   awaiting off-owner sidecar Q, without granting the sidecar ownership of the
   live target runner. `cuda_waiting_admission.py` carries the optional lane
   binding through atomic admission. Without the option, the old synchronous
   path stays selected.
5. `probe_lane_sidecar_process.py` makes the D Scheduler the process owner:
   it creates a private directory, starts a fresh sidecar, checks ready PID,
   socket, GPU label and checkpoint hashes, creates an exact-PID client, and
   terminates/cleans the child on close or failed startup.
6. `cuda_serving_limits.py` accepts a strict optional `probe_sidecar` object;
   `cuda_serving_startup.py` starts/binds the sidecar only when opted in,
   requires distinct single physical GPUs, passes the client/checkpoint at
   waiting-queue admission, and retains the process owner until drained close.
   **Current opt-in still also loads a local draft on formal D GPU1**. It has
   not yet removed that duplicate resident memory/initialization. The
   experiment-specific script is `run_pvd_qwen_sidecar_gpu.py` under `test/`;
   productionizing its loader/packaging remains work.

Important active files in the 55-commit local range (68 changed files total;
use `git diff --name-only origin/pvd-disaggregation..HEAD` for the complete
machine-readable inventory):

- Serving/state: `python/sglang/srt/disaggregation/pvd/{cuda_serving_limits.py,
  cuda_serving_startup.py,cuda_waiting_admission.py,cuda_refresh_driver.py,
  cuda_request_admission.py,cpu_prefetch_request.py,probe_search.py,
  cuda_probe_search.py,target_probe.py}`.
- New private lane: `python/sglang/srt/disaggregation/pvd/probe_lane_{protocol,
  wire,unix,identity,model,routing,sidecar_process}.py`.
- V/attention/performance: `pvd/{control_server.py,index_search.py,
  prompt_index.py,search_client.py,search_routing.py,search_wire.py,
  cuda_model_attention.py,cuda_sparse_attention.py}`.
- Experiments and regressions: `test/registered/disaggregation/
  {cloudlab_pvd_new_lease.sh,run_pvd_qwen_sidecar_gpu.py,
  run_pvd_qwen_sidecar_supervised_gpu.py,run_pvd_cuda_probe_smoke.py,
  pvd_qwen_v100s_serving_limits_triton_sidecar_cloudlab.json}` and
  `test_pvd_probe_lane_*.py`, `test_pvd_cuda_serving_*.py`, plus V/index/
  attention/fact-recall tests. The four named docs above were also changed.

## 4. Commits and verification

The exact history is `git log --oneline 05d7a2afe..HEAD` (55 commits before
this document). Major stages and their commit IDs, in chronological order:

- Earlier performance/quality work in this range: `de78d61f1` (new-lease
  validation), `4430b9ed3`/`8f52c6d51`/`e33e4c7c6` (bounded grouped exact
  index), `0ce56f8d7` (M8 prefetch), `d8c7e8099` (real-Q CAGRA recall),
  `2789c35fb` (M16 tradeoff), `657568e97`/`88ae9d882`/`c932aeadb`
  (refresh/search/probe experiments), `8934ab9bb` (seven-refresh result).
- Private lane protocol/transport: `0c7b5f544`, `07d7e503e`, `7758e574c`,
  `229a2eba2`, `04a65286f`, `b31ffdbbc`.
- Model/Q/V/driver binding: `c921db17e`, `1c2fe2ccb`, `5d37f9d73`,
  `0fe0cc15d`, `22030514f`, `19cfbd7e8`, `5531608af`, `fe5f4d450`,
  `6f4609e45`.
- This last run: `1aaf0d8b4` (real GPU sidecar readiness), `70a84e709`
  (child-process owner), `b86c5a6e6` (real supervised Q), `88c72ed79`
  (explicit D serving opt-in), `4fc27b27b` (concurrent reply capacity and
  CloudLab launcher/config). All are **local only**.

Verified facts, not just unit-test intent:

- On V's isolated validation environment, Ruff E/F/I and formatting passed
  for each newly edited private-lane/startup/test file. Focused startup,
  strict-config and real Linux child-process tests: **83 passed** before the
  concurrency patch; the later startup+Unix suite: **26 passed**. These are
  different overlapping subsets, not a summed test count. The entire PVD CPU
  suite has **not** been rerun after the latest 55 commits; older full-suite
  counts in readiness docs are historical.
- On D GPU0, real Qwen2.5-7B-Instruct + 0.5B target/draft startup, content
  hashing and cross-process Q matched the direct target probe for all 28
  layers. Content hashes: weights/config
  `5725e17b4bd31a7d1f723215ebc2402028658615a1b9fcab0d11f4445dedd5f2`,
  tokenizer `fc79977ab8ac1b4fedc4b83e5eb86414438030dd7b2c31c701310efb58f9c7b8`.
  The supervised one-request smoke had ~39.32 s readiness, ~0.34 s Q reply,
  refunded host budget and cleaned its socket/process. These are **not**
  serving latency numbers.
- Opt-in three-node P/V/D/Gateway actually started on V100S with P GPU1,
  V GPUs0+1, formal D GPU1, D sidecar GPU0, `mlx5_0` ACTIVE and `mlx5_1`
  DOWN (single-rail debug). One cold 104-Prompt/12-output request completed
  P→V→D and two sparse refreshes in **65.88 s**; the first Q capture was
  **14.89 s** due to cold compilation. Three later single-client requests,
  ~103 Prompt/20 output, took **2.67/3.07/2.88 s**. No same-input full-KV
  control was run in this final sidecar experiment.
- Before `4fc27b27b`, the first two-client attempt aborted one stream after
  5/20 tokens. The experiment sidecar had `max_connections=2` and a reply
  budget permitting only **one** in-flight reservation. That is a plausible
  capacity cause, **not confirmed as the unique cause** because the sidecar
  swallowed refusal details. `4fc27b27b` aligns the server's connection and
  reply-reservation bounds to the explicit D request limit. A real Unix test
  holds two distinct reservations concurrently. On a restarted D, identical
  two-client/three-round `sidecarfair1` load completed **6/6** streams with
  20/20 events; wall times **5.13/5.20/5.12 s**. That is a narrow regression
  pass, not proof of all concurrency/cancellation or a speedup.
- Earlier **full-KV** controls in the CloudLab validation document were often
  faster (for example ~1.85/1.70 s for a different 374-token two-client
  load; M16 128-output full KV ~6.17–6.36 s versus sparse ~9.54–9.75 s).
  Do not compare these different inputs directly with `sidecarfair1`.

## 5. CloudLab environment and remote state at the original handoff

SSH (authorized by the user; key contents must never be printed):

```text
ssh -o BatchMode=yes -i "C:\Users\34155\.ssh\cloudlab_pub_wsl" Yizhzhu@clgpu020.clemson.cloudlab.us  # P, 10.10.1.1
ssh -o BatchMode=yes -i "C:\Users\34155\.ssh\cloudlab_pub_wsl" Yizhzhu@clgpu021.clemson.cloudlab.us  # V + Gateway, 10.10.1.2
ssh -o BatchMode=yes -i "C:\Users\34155\.ssh\cloudlab_pub_wsl" Yizhzhu@clgpu019.clemson.cloudlab.us  # D, 10.10.1.3
```

Each node sources `/users/Yizhzhu/.sglang-v100-pvd-env.sh`; use
`$SGLANG_PVD_ROOT` rather than guessing mount aliases. The pinned local
models are `$SGLANG_PVD_ROOT/models/Qwen2.5-{7B,0.5B}-Instruct`; Gateway sees
the common model symlink `/users/Yizhzhu/pvd-models/Qwen2.5-7B-Instruct`.
Use `PYTHONPATH="$checkout/python:$checkout/test/registered/disaggregation"`
for standalone GPU/test scripts. Without it an editable install may import
the old source checkout; the sidecar script deliberately refuses that mix.

All three nodes have a fresh isolated Git worktree at
`$SGLANG_PVD_ROOT/validation/pvd-sidecar-88c72ed79`, **Git HEAD 88c72ed79**.
The D worktree also has manually copied `4fc27b27b` versions of
`cuda_serving_startup.py` and `run_pvd_qwen_sidecar_gpu.py`, and the updated
launcher/config. P/V worktrees remain at `88c72ed79`. This is an experimental
overlay, **not an exact synchronized Git checkout**. The initial bundle is
`$SGLANG_PVD_ROOT/validation/staging/.pvd-validation-88c72ed79-sidecar.bundle`
on each node. Avoid treating those remote worktrees as a clean commit or
overwriting the older `pvd-05d7a2afe` worktrees.

All experiment-owned services were intentionally stopped after evidence was
captured: P PGID 66486; V PGID 143664; Gateway PGID 144154; D PGID 118211
(the earlier D PGID 116243 was stopped before the fix). A final three-node
check showed **0 MiB GPU memory on all six cards**, no PVD listener on
30002/30003/9100/9300/9301/8001, and no leftover sidecar. Logs remain:
`$SGLANG_PVD_ROOT/validation/logs/{p,v,d,gateway}-sidecar-gate.log`, plus
`d-sidecar-gate2.log` for the passing two-client rerun. Do not blindly send
signals to those historical PIDs; check ownership and PGID afresh.

The launcher is `test/registered/disaggregation/cloudlab_pvd_new_lease.sh`.
For this checkout, pass `PVD_CHECKOUT=<isolated worktree>`,
`PVD_EXPECTED_COMMIT=<that worktree's actual HEAD>`, and a unique
`PVD_RUN_TAG`. D additionally used `PVD_PROBE_SIDECAR=1` and
`PVD_LIMITS_PATH=<worktree>/test/registered/disaggregation/
pvd_qwen_v100s_serving_limits_triton_sidecar_cloudlab.json`. This fixture
contains the D node's absolute sidecar script path in the current CloudLab
lease; update that path when creating a new worktree. Start V, P, D, then
Gateway; check V `/health`, P/D `/health`, Gateway `/v1/models` and actual
GPU/sidecar PID placement. The new D startup takes ~1–2 min. The current
fixture selects bounded exact index for prompts ≤2304; native CAGRA must be
opted into/verified separately.

## 6. Outstanding work — exact next steps

1. **Completed: normalize a reproducible deployment.** The handoff was
   committed as `35350660d` and corrected in `ecf1fc247`. The incremental
   bundle `.pvd-validation-ecf1fc247-sidecar.bundle` contains precisely
   `88c72ed79..ecf1fc247`; SHA-256 is
   `12c4c72670759d15ec030372ddb472123c1fcf84935a0ac2ab7765611df07d8a`.
   It was copied, hash-checked, verified and fetched on P/V/D. All three new
   isolated worktrees are `$SGLANG_PVD_ROOT/validation/pvd-sidecar-ecf1fc247`
   at exact HEAD `ecf1fc247e459c1be1ca0aa37e29a5f7d26df2a0`, clean at
   creation. Older worktrees and user data were preserved. D uses a separate
   validation-only JSON at
   `$SGLANG_PVD_ROOT/validation/config/pvd_qwen_v100s_serving_limits_triton_sidecar_ecf1fc247.json`, whose
   `probe_sidecar.script_path` points into the new D worktree. P and D target
   weights/tokenizer content hashes match section 4; V's tokenizer-only mirror
   matches the same tokenizer hash. All six V100S showed 0 MiB usage,
   `mlx5_0` was ACTIVE, `mlx5_1` DOWN, and ports 30002/30003/9100/9300/9301/
   8001 had no listener. The focused startup/Unix/process/config suite on V's
   pytest-equipped environment passed **96 tests** against this worktree. D's
   serving Conda Python lacks pytest. No services were started for this
   normalization step, no new serving-latency claim was made, and nothing was
   pushed to GitHub.
2. **Prove and diagnose the lane selection under failures.** Add an explicit
   `probe_source=private_lane|inline` field to D refresh timings and bounded
   sidecar rejection/lifecycle diagnostics; the current sidecar supervisor
   drains/discards ordinary child stdout after readiness, and the Unix server
   suppresses expected capacity/protocol errors. Build a deterministic test
   of two concurrent live tickets, one capacity refusal, timeout, sidecar
   restart, cancellation/retraction and the required committed-prefix
   boundary fallback. Confirm no stale Q, no leaked reply budget and no
   abandoned RDMA destination/MR. The earlier one-stream abort must become
   attributable rather than merely disappearing after the bound increase.
   **2026-09-27 fault gate verified; historical abort cause remains unknown:**
   `d042ec364`/`dc9ccf227` add
   `probe_source=private_lane|inline` to D refresh timings, bounded categorical
   sidecar rejection forwarding and lifecycle logs. `1befeef2e`/`610b9fdb7`
   test two live replies plus a capacity refusal, boundary committed-prefix
   Q, cancellation/retraction and cancellation during an in-flight fake RDMA
   write; the destination is retained until the write is safe, then the
   registry, pending rounds and reply budgets drain. `582163751` logs the
   request ID, source and exception type on refresh failure and tests a
   capacity refusal through the controller. The focused V suite passed
   **61 tests** with Ruff E/F/I and formatting clean at `582163751`.
   A real three-node opt-in smoke at `dc9ccf227` returned HTTP 200 for
   42 Prompt/20 output tokens in 69.24 s; D logged four private-lane sparse
   refreshes at boundaries 4/8/12/16. The first capture took 14.85 s cold;
   later captures were ~0.30 s. This is one request, not a baseline or gain.
   A controlled real sidecar stop at `582163751` returned HTTP 503
   `decode_unavailable` at waiting admission in 1.95 s, before any refresh
   could install stale Q. Repeated exit logs in that run motivated
   `5ab196426`, which reports one exit per owner; its real-child test passed
   **10 tests** with Ruff E/F/I clean. At final `826fe46c7`, a warm 20-output
   request completed, then a new 128-output request reached
   `refresh_scheduled` and an established sidecar Unix connection. Killing
   only that sidecar PID produced D's request-scoped
   `probe_source=private_lane error_type=ProbeLaneProtocolError` failure before
   boundary 4; there was no `refresh_ready` or installed boundary for that
   request. Gateway returned 503 after D errors/retries. The exit was logged
   once. V's two shards each reported zero transfer reservations, in-flight
   and unknown transfers afterward; Entry records remained under the 300 s
   TTL, so their mere presence is not evidence of a leak. The old aborted
   stream cannot be assigned a unique historical cause because its rejection
   detail was swallowed. The same capacity condition now reproduces with an
   explicit `reply_capacity` category in deterministic tests. All three real
   experiments were stopped; GPUs and target ports returned to idle, and
   only their own sidecar socket paths were removed. Two older empty
   `/tmp/pvd-probe-*` directories were preserved. No same-input baseline or
   performance gain is claimed.
3. **Run controlled A/B on the same exact inputs**: full Prompt KV, existing
   inline predictive path, and sidecar predictive path; 20 and 128 output
   tokens; short, ~1k and ~2k Prompt; 1/2/4 clients; repeated warmed trials
   with randomized/interleaved mode order. Log completion, output hashes,
   wall/request median/p95, TTFT, observed per-token gaps, stage timing,
   sidecar GPU0/GPU1 peak memory/utilization and V/RDMA timing. Include cold
   CAGRA build separately from warm exact/CAGRA. Do not claim improvement if
   same-input full KV is still faster or if quality/correctness fails.
4. **Remove the duplicate D-GPU1 draft only with a proven interface**. The
   current serving startup still invokes `build_cuda_prediction_startup` in
   the formal D process, keeping a local draft and target probe even when the
   sidecar is active. Split off-owner pipeline metadata/health from the
   resident runner so `CUDARefreshDriver` can safely function without these
   duplicate weights/KV pools. Prove request ownership, cleanup, fallback,
   memory savings, and no formal-forward lock held while Q waits. Avoid
   assuming ModelRunner is reentrant.
5. **Finish V/CAGRA and long-context acceptance**, including real target-Q
   recall/quality, cold index construction, budget/eviction/reuse, M cadence
   and throughput under multi-client load. CAGRA and sparse Select–Pack–
   FanIn–Install/attention fusion still have measured or design gaps. Update
   documentation when each verified step is committed.

Every subsequent step: preserve the unrelated dirty worktree, run focused
tests and relevant real GPU/RDMA checks, commit locally, do **not** push.
If the final goal is still unproven, report the exact remaining blocker and
evidence rather than calling the project complete.

## 7. 2026-09-27 Decode-overlap checkpoint (later than section 6)

The next work after the earlier fault gate used an isolated checkout on P/V/D;
no code was pushed. `6d6c29e75` moved sidecar Unix I/O to a control loop,
`31303afe7` moved V delivery HTTP I/O there, `2ebc53974` added a bounded
target-Q prefix cache, and `426095fba` added a separate, incarnation-scoped
draft prefix cache. `b8a6f7d0a` enabled child stage diagnostics. The opt-in
cache budgets in the D validation JSON are 144 MiB target and 64 MiB draft.
The sidecar GPU0 and formal D GPU1 run concurrently; the caches retain only
request-owned prompt prefix rows, and each speculative branch frees its own
suffix. `c418e7a4d` raised the explicitly bounded CUDA prediction horizon to
32 tokens; `ac3169962` lets the fact-recall validation script request at most
256 output tokens so a second M64 boundary can be measured. The V Linux
focused startup/draft suite at `c418e7a4d` passed **233 tests** (one unrelated
pytest config warning); Ruff E/F/I and format checks passed. Earlier dual-cache
and Unix/control suites passed **234** and **30** tests, respectively.

Same-input one-client V100S checks used seed `overlap485d`, 48 facts, 1413
Prompt tokens, first expected code `20900`, and P/V/Gateway held constant
except for the required Gateway restart after each D restart. These are
individual observations, not randomized performance distributions:

| D path | 20 output | 128 output | Boundary evidence |
| --- | ---: | ---: | --- |
| Full Prompt KV (`485d725f9`) | 1.31 s warm (1.50 s cold) | 5.36 s | No sparse refresh |
| Sidecar M16/lead8, no prefix cache (`485d725f9`) | 2.23 s warm | 11.37 s | Refresh repeated full-prefix work |
| Sidecar M16/lead8, target-Q cache only (`2ebc53974`) | 3.04 s | 9.33–9.41 s | Warm capture about 0.50 s |
| Sidecar M16/lead8, both caches (`20183ea4d`/`b8a6f7d0a`) | 3.04 s | 9.13 s (diagnostic rerun 9.87 s) | Draft append about 0.156 s; warm refresh about 0.75 s; boundary wait about 0.30 s |
| Sidecar M32/lead16/predict16, both caches (`b8a6f7d0a`) | 1.50 s | 7.75 s, then 7.05 s | Warm refresh 0.97–1.00 s; boundary wait 0.19–0.22 s |
| Sidecar M64/lead32/predict32, both caches (`c418e7a4d`) | 1.50 s | 6.82 s, then 6.37 s | Cold refresh 1.87 s; boundary wait 0.39 s. A 192-output run finished in 9.07 s; its later cached refresh took 1.50 s and boundary wait 0.09 s. |

All fact probes above returned the expected first code. D logs explicitly
reported `probe_source=private_lane`, cache `prefill` then `append`, and
installed sparse boundaries. The script's `mode_verified_by_script` field is
always false; the process arguments and D logs establish the mode. The
128-token output hashes differ between full KV and sparse modes, so first-code
success is only a narrow quality check. At M64 the 20-output request has no
refresh and is near full-KV cold latency; 128-output latency remains about
19% above the earlier full-KV observation even on the faster repeat. The
later M64 refresh overlaps most of formal Decode, but its observed boundary
wait is still 0.09 s, and each new request incurs cold-prefix work. Do **not**
claim full overlap, quality parity, or a speedup over full KV.

Next: prewarm both sidecar prefixes during the first formal Decode window or
otherwise remove the first-refresh cold stall without changing committed
state; reduce V search/delivery latency enough to cover the remaining warm
boundary gap. Then run interleaved, repeated full-KV/sidecar A/B with output
quality and true-Q recall, multiple Prompt lengths and clients, plus the
original phase-4 duplicate D-GPU1 draft removal and phase-5 V/CAGRA work.
The 32-token horizon remains an opt-in experimental bound; later-position
draft errors and budget pressure need broader real-GPU tests before it can be
called production-ready.

After measurement, the exact experiment process groups on P/V/D and Gateway
were stopped. Ports 30002/30003/9100/9300/9301/8001 had no listeners and
all three nodes reported no GPU compute processes. The isolated V checkout at
`ac3169962` and D checkout at `c418e7a4d` were clean. The seven pre-existing
dirty local files and all validation bundles remain untouched; no push occurred.

## 8. 2026-09-27 long-context and sidecar-prewarm checkpoint

This section supersedes the prior "next" instruction for the requested long-
context experiment. The user explicitly authorized bundling local commits to
the named CloudLab validation directories and using both 32 GiB GPUs on each
node. P ran Qwen2.5-7B-Instruct as TP2 on GPUs 0/1 with 512-token chunked
prefill. V used GPUs 0/1, 32,768 pages per shard, and `mlx5_0` on both ranks.
D used formal Decode on GPU1 and the target+draft sidecar on GPU0. Only
`mlx5_0` was active; these results make no dual-rail claim. The V
`cagra-auto` exact threshold was **explicitly** 8192 rows for 4k/8k tests
and 16384 rows for 16k tests. A V restart that omitted this override caused
20-second native CAGRA builds and 47–49-second requests; those runs are
configuration mistakes and are excluded from the matched table below.

All table rows use seed `longpvd0927`, one fact-recall case, 128 greedy output
tokens, and SSE timing. The same input SHA-256 was used for each mode at each
length; all returned the expected first five-digit code. Each cell is an
individual observation, not a median over repeated randomized trials. Only
the 8k prewarm cell has two repeats.

| Actual Prompt | Full Prompt KV | PVD sidecar, no prewarm | PVD sidecar, prewarm at n=0 | Prewarm first boundary wait |
| ---: | ---: | ---: | ---: | ---: |
| 3,997 | 6.57 s | 10.13 s (earlier stack observation) | 8.09 s | 0.15 s |
| 7,948 | 10.44 s | 16.74 s | 13.95 / 13.85 s | 2.74 s |
| 15,875 | 20.49 s | 36.41 s | 32.49 s | 11.87 s |

The 16k matched PVD A/B used the same commit, P/V services and budgets; only
`sidecar_prefix_prewarm` changed. It saved 3.92 s and reduced the first M64
boundary wait from 15.82 to 11.87 s, but remained 12.00 s slower than full
KV. Its initial sidecar draft/target prefix stages took 5.19/13.86 s; the
later cached append took 0.67/0.33 s. V exact search was 0.61 s. At 8k,
prewarm cut the first boundary wait from 4.95 to 2.74 s. At 4k, the
observed 0.15 s boundary wait was nearly covered by formal Decode, while
PVD's ordinary token spacing remained higher than full KV's. For 16k the
PVD/full-KV observable inter-token p50 was 0.138/0.084 s. Sidecar GPU0 used
about 29.4/32 GiB and formal D GPU1 about 21.7/32 GiB after the 16k run.
Output hashes differed between full KV and PVD at each length, so first-code
success is only a narrow quality check. Prewarm-on and prewarm-off PVD output
hashes matched at each tested length.

Implementation/verification since section 7: `950f7b1f0`/`8576d919b`
added long-context launcher and fact profiles; `e50cce0b0` used both P
GPUs; `7bc027db6` fixed the quadratic P-prefill Q allocation via compact
chunk attention; `8fcc5c1e8` added opt-in sidecar prewarm; `6d0bf83ac`
raised the bounded D retrieval-bank setting for long Prompt KV;
`1dd0f2ed3` exposed bounded sidecar handler errors; `a58238eef` chunked
cached target probe prefix forwards; `9578935e4` started cache idle TTL at
probe completion; `974736f47` chunked cached and uncached draft prefix
forwards. Focused test/fixture fixes are `25cd5c409`, `b52f46fdd`,
`e68aa49e8`, `a354e6f06`, and `a0005f0d4`. The V Linux focused target
probe/compact suite passed 39 tests, handler/target suite 25, and final
draft/compact suite 110 (each with one unrelated pytest config warning).
Ruff E/F/I, formatting, bytecode compilation and `git diff --check` passed
for the newly changed source and tests. Initial 8k target prefix and 16k
draft prefix each caused a real sidecar CUDA OOM before their respective
chunking fixes; both lengths subsequently completed.

The experimental D JSON files are under the D node's isolated
`$SGLANG_PVD_ROOT/validation/config/` directory. For 16k,
`pvd_qwen_v100s_sidecar_long18432_m64_{prewarm,}.json` set a 1.5 GiB target
prefix budget, 384 MiB draft prefix budget and 18,432-token maximum. The
launcher used 2 GiB D staging, 4 GiB probe scratch, 1 GiB draft scratch,
1.5 GiB retrieval bank, M64/lead32/predict32. The 16k full-KV control used
the same 18,432-token P/V context and 2 GiB D staging. These lease-specific
JSON files contain an absolute sidecar script path and are intentionally not
committed to this repository.

Next priority: start valid sidecar prewarm earlier than formal Decode n=0,
overlapping the 8–9 s long-Prompt Prefill/transfer window, while preserving
exact Req incarnation/version checks and never installing speculative state.
The current `CUDARefreshDriver.register` requires P's first output token and
the imported Prompt, so merely moving its current n=0 call earlier would
break its admission contract. The earliest potential overlap is D Req
creation/arrival, before waiting for V: its prompt IDs, transfer/delivery IDs
and D worker epoch are already present. `PVDKVReceiver._send_metadata()` in
`conn.py` is a later seam where the Entry/manifest and P first-token metadata
are present before initial receive completes. A prompt-only sidecar ticket
at Req arrival could overlap P Prefill, but requires a
new provisional authorization record keyed to the exact Req object/rid,
D worker epoch and Entry transfer ID. Route discovery presently scans only
the final waiting queue with `bootstrap_runnable`, and the existing
`InitialPromptReceipt` proof is minted after full receive ACK/TP agreement.
An early cache-only ticket needs its own bounded constructor using the
private sidecar's configured layer/head coverage instead of pretending final
V routes exist. It must reconcile with that later receipt, use a distinct
prompt-only prefix version, discard Q, and have explicit cancellation/cache
retirement. Do not bypass those checks by calling the current n=0 method
from the receiver. Also reduce the 16k target-Q prefix
compute (13.86 s) and sparse formal Decode token cost before claiming a
long-context speedup. Full-KV Prompt transfer to D and the duplicate D-GPU1
draft still remain. Complete randomized multi-request throughput and quality
acceptance before claiming final PVD advantage.

The experiment process groups were stopped after these measurements:
P 88217, V 189390, D 173489, Gateway 191700. A final three-node check
showed **0 MiB on all six GPUs**, no GPU compute process, and no listener on
30002/30003/9100/9300/9301/8001. Logs and validation JSON files remain.
The clean isolated checkouts at `$SGLANG_PVD_ROOT/validation/pvd-sidecar-d042ec364`
are P `7bc027db6ce702fb57050c5e8b53cf72c51526db`, V and D
`a0005f0d4cdf2efe47e8464051c8980ec74ce351`. The seven pre-existing
dirty local files and all untracked validation bundles remain untouched.
No Git push occurred.

## 9. 2026-09-27 16k Decode-segment and early-ticket validation

Section 8's next-priority text predates the new D Req-arrival implementation.
Commit `c70692a21` adds a bounded prompt-only sidecar ticket keyed to the
exact D Req, delivery, Entry transfer ID, and worker epoch. It warms only the
private draft/target prefix caches; the Q reply is discarded and later
admission reconciles against the completed initial Prompt receipt. Commit
`632dd7eed` adds early/late 32-gap timing to the SSE fact benchmark. The
new early-ticket path passed **101 focused tests on V Linux**. The isolated
P checkout stayed at `7bc027db6`; V and D used clean `c70692a21` checkouts.

All following observations used the same 15,875-token Qwen2.5-7B input
within each output-length pair, P TP2, V two GPUs, D formal GPU1 plus
sidecar GPU0 for PVD, 18,432-token context, V exact index threshold 16,384,
M64/lead32/predict32, and one client. The P process was restarted before
the 128-token full-KV control and before the deferred-fence PVD diagnostic
to clear its prompt cache. The 128-token input SHA-256 was
`4a8877d0d093bc6a38c813bb49ce3cadc817e78c7540d46359a04f96464e2fa5`;
the 256-token input SHA-256 was
`11306f421ea8400d445404cd80b0c4e8e0cfaf49c34ee0e889683adb7eeaa3a9`.
These are single runs, not distributions. The output hashes differ between
PVD and full KV; first-code correctness is a narrow check only.

| Mode | Output | Wall | TTFT | First 32 gap p50 | Last 32 gap p50 | Max gap | First-code correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| PVD, Req-arrival early ticket | 128 | 33.59 s | 9.59 s | 0.138 s | 0.043 s | 12.05 s | yes |
| Full KV | 128 | 20.55 s | 9.30 s | 0.084 s | 0.085 s | 0.46 s | yes |
| PVD, early ticket plus borrowed Prompt reader/deferred layer fences | 128 | 34.13 s | 10.30 s | 0.133 s | 0.042 s | 12.27 s | yes; PVD output hash unchanged |
| PVD, Req-arrival early ticket | 256 | 39.25 s | 8.48 s | 0.138 s | 0.045 s | 11.94 s | no |
| Full KV | 256 | 31.67 s | 8.96 s | 0.084 s | 0.085 s | 0.46 s | no |

The first 64 formal Decode tokens still use the imported **full Prompt bank**
through the custom PVD sparse-attention adapter. They pay host metadata
`.tolist()` and per-layer pointer/row planning plus CUDA completion fences;
they do not yet benefit from a small retrieved bank. The first M64 refresh
then installs that sparse bank. At 16k, PVD's first 32 gaps were slower than
full KV's by about 54 ms/token, while the last 32 gaps were faster by about
42 ms/token. These segment timings support the adapter/full-bank path as a
contributor to early Decode cost; no per-kernel profile yet allocates the
exact 54 ms among metadata, fences, attention and other scheduling work.
The roughly 12-second first boundary stall dominates the 128-token
end-to-end gap. On the 256-token run, two later cached refreshes took
2.17/2.23 seconds each end-to-end and largely overlapped Decode, but PVD
still finished 7.58 seconds behind full KV. Neither 256-token run returned
the expected first code; do not infer quality parity from these samples.

The new early ticket did **not** demonstrate an end-to-end improvement over
the earlier n=0-only 16k PVD observation (32.49 seconds in section 8).
In the 128-token early-ticket run, P's last prefill batch was logged at
12:49:01 and the sidecar's `draft prefix cache action=prefill` at 12:49:08.
In the deferred-fence repeat, those log seconds were 13:05:53 and 13:06:00.
The draft action log is emitted **after** draft prefix preparation/generation,
so it is not a sidecar-start marker; the draft stage itself took about
5.2 seconds and target probe about 13.9 seconds. Gateway sends P and D POSTs
concurrently. Existing logs cannot distinguish D Req creation, owner-loop
queueing, sidecar ticket dispatch, and GPU compute start; do not attribute
the seven-second log separation to Gateway sequencing. Timing stamps at the
early-ticket scheduling/execution seam are the next diagnostic.

The borrowed-reader/deferred-fence 16k diagnostic set
`PVD_REUSE_FORWARD_BANK_LEASE=1` and `PVD_DEFER_LAYER_FENCES=1` on D.
The first startup intentionally failed its guard because the existing
scratch-reservation limit was 64 versus the required minimum of 113 for a
four-request/28-layer bound. A separate validation JSON raised the bound to
256; the successful run showed no observed traceback or quarantine and had
the same PVD output hash, but its total time did not improve. Do not enable
these opt-ins by default based on this sample.

Next: timestamp the early ticket from D Req creation through sidecar accept
and prefix compute, then move or speed the first target prefix work so it
actually overlaps P Prefill and first formal Decode. Profile the first
64-token PVD full-bank attention path and test avoiding its repeated host
metadata/synchronization overhead. Re-evaluate output quality, true-Q recall,
and repeated randomized 16k+ throughput before claiming a PVD advantage.

## 10. 2026-09-27 early-ticket correction and same-GPU design checkpoint

Section 9 records the first early-ticket trials. A later timing probe
(`2436a30af`) found that the early ticket was not actually scheduled: D's
`Req.origin_input_ids` is an `array("q")`, while the ticket constructor in
`c70692a21` accepted only lists and tuples. Commit `63f5eb2e0` accepts this
production type. Thirteen focused Linux tests passed after the fix. The
older section 9 conclusion about the ticket's effect applies only to the
pre-fix trials; it does not measure an active early ticket.

A fresh 16k single-request PVD repeat with the same 15,875-token input
and 128-token output reduced wall time from 33.2246 to 24.1776 seconds and the
maximum token gap from 11.8927 to 2.6252 seconds. TTFT was 9.3155 vs
9.5864 seconds. First-32 gap p50 was 0.13883 vs 0.13763 seconds, and
last-32 gap p50 was 0.04307 vs 0.04330 seconds (pre-fix vs post-fix).
The output SHA-256 was unchanged, but the first-code fact was incorrect
in both runs. D logged the ticket request start 1.373 ms after scheduling,
19.20 seconds of sidecar work, and P's last Prefill about 9 seconds after
the early ticket started. The first boundary wait fell from about 11.85
to 2.58 seconds. This demonstrates early sidecar work overlapping P Prefill
and initial formal Decode on **separate D GPUs**. It does not establish a
PVD speed advantage over full KV or validate output quality.

The requested architecture uses the **same GPU** for formal Decode and
prediction, while other requests continue producing tokens during that
prediction. The current inline D path does not meet this requirement:
`cuda_refresh_driver.py` acquires `TargetExecutionArbiter` for synchronous
capture, and `cuda_scheduler_binding.py::ready_to_prepare` stops Decode
batch preparation while it is busy. The tested sidecar path avoids that
global wait by using another GPU, so its timing cannot be used as evidence
for same-GPU coexistence.

SGLang's native `STANDALONE` speculation shares the Scheduler's batch
contract: `EAGLEWorker.forward_batch_generation` runs draft, installs
`batch.spec_info`, runs target verification, and returns accepted tokens
for formal request processing. `StandaloneWorker` constructs its draft
worker with the target worker's request/allocator interface. Thus the
whole native worker is not a prediction-only API whose tokens can merely
be ignored. PVD already reuses lower-level SGLang model forwards in
`draft_forward_adapter.py` and `draft_runner_sglang.py`. A same-GPU
implementation should give prediction a private Req/KV branch, advance
that branch in bounded Scheduler turns while normal request Decode
continues, send only predicted query vectors to V, then leave V search
and delivery asynchronous while the original request formally decodes.
Its candidate tokens must never enter formal output IDs, sequence length,
or KV mappings. No same-GPU implementation or benchmark is validated yet.

## 11. 2026-09-27 same-GPU cooperative prediction checkpoint

The design in section 10 has now been implemented as an opt-in path. Commits
`72c277531` through `e2f0fbba2` add a private prediction Req/KV branch,
bounded native SGLang model forwards, and `PVD_COOPERATIVE_PREDICTION=1` in
the CUDA refresh driver. The branch produces query vectors for V; its draft
tokens are never accepted into the formal Req or its KV mapping. An optional
`PVD_SEED_PROBE_FROM_PROMPT_KV=1` copies the committed Prompt KV to the private
target prefix cache. The refresh epoch opens before the query future finishes,
so advancing formal Decode cannot make the prediction's prefix snapshot
regress the committed-token clock. Cancellation closes the private iterator.

The D Scheduler retains the canonical running batch, selects ready rows for
each formal forward, and synchronizes their tokens and sampling state back to
the canonical requests. A request waiting for V refresh therefore does not
hold up ready peers. Prediction work runs on the **same physical GPU** as D,
in bounded steps between formal forwards; this is Scheduler interleaving, not
simultaneous CUDA kernel execution. It reuses SGLang's lower-level model
forwards and KV pools, not its high-level `STANDALONE` speculation worker,
which would formally accept draft tokens into the whole batch.

Validation on CloudLab clgpu019 D / clgpu020 P / clgpu021 V used target and
draft on D GPU 1, with GPU 0 unused by D, `PVD_REFRESH_INTERVAL=4`,
`PVD_DRAFT_PREDICT_TOKENS=2`, a 2304-token context limit, and
`PVD_PROFILE_REFRESH_TIMELINE=1`. The first concurrent pair returned two
HTTP 200 responses, each with a 255-token prompt and 16 completion tokens,
in 3.833 seconds. The unequal-prompt pair returned two HTTP 200 responses,
with 755/130 prompt tokens and 20 completion tokens each, in 4.455 seconds.
These are client wall times for functional checks, not matched full-KV
performance comparisons or quality validation.

The D trace for the unequal pair directly shows interleaving. Request
`44f4c3c0...` had a prediction step at monotonic `101373.033236`; the
other request `ae355c71...` formally committed token 1 at `101373.085665`;
the first request's prediction did not complete until `101373.129298`.
The first request also formally advanced from token 2 to 3 while its own
private prediction was active. Later, when the first request waited at
refresh boundary 4, the ready-only view committed the second request's
tokens 2, 3, and 4. V search HTTP from `101375.137881` to `101375.268463`
overlapped a formal commit of the second request's token 11 at
`101375.219397`. Trace file on D:
`validation/logs/d-samegpu_e2f0.log`. The matching client runs were launched
from V's `validation/samegpu_pair_client.py`.

Focused Linux tests on the early-epoch implementation: 101 passed; the
larger ready-batch/draft suite on the preceding implementation: 137 passed.
Ruff E/F/I and format checks passed for the touched PVD modules. The final
trace-only commit adds bounded diagnostic logs and a format correction; its
live two-request run passed. The paired client now supports unequal prompt
lengths and bounded staggering to make overlap observable.

The same-GPU path still has no controlled throughput comparison against full
KV, no repeated randomized concurrent-load trial, and no 16k+ validation.
The short-context trace proves progress of formal requests during private
prediction and V search; it does not prove a long-context speed advantage or
that every Scheduler poll produces a formal token. Measure those separately
before enabling the opt-in by default. Preserve the seven pre-existing dirty
files and local `.pvd-validation-*.bundle` artifacts when continuing work.

## 12. 2026-09-28 same-GPU long-context and concurrent control

Matched live controls now exist for the same-GPU path. On the 10.10.1.x
CloudLab lease, P used Qwen2.5-7B TP2 and 512-token Prefill chunks on GPUs
0/1; V used both GPUs, 32,768 pages per shard and an explicit 16,384-row
exact-search threshold; D ran target and draft on **GPU1**, with GPU0 idle.
The P/V services and Gateway route were kept fixed across each D-mode switch.
D used context/max-total 18,432, 2 GiB staging, `mlx5_0`, and 128 greedy
output tokens. PVD used M64/lead32/predict32, cooperative prediction, Prompt
KV seeding and the checked-in
`pvd_qwen_v100s_serving_limits_triton_m64_samegpu_16k.json` profile. The
three-case 16k PVD run preceded the profile's cache-budget increase; it had
a 1.5 GiB target-prefix budget. The successful two-request 8k run used the
checked-in 3 GiB budget. The full-KV D control used the same D checkout base,
model, context, staging and source/route, with predictive serving disabled.

| Workload and matched input set | Same-GPU PVD | Full KV | First-code check |
| --- | ---: | ---: | --- |
| 3 sequential 15,875–16,019-token prompts, median request wall | 27.63 s | 19.42 s | 3/3 in each mode |
| Same 16k set, median TTFT / Decode duration | 9.28 / 18.35 s | 8.18 / 11.24 s | 128 SSE events per request |
| 4 roughly 8k prompts, two synchronized requests per wave, wave walls | 23.21 / 22.59 s | 16.28 / 16.24 s | 4/4 in each mode |
| Same 8k set, total time for two waves | 45.79 s | 32.52 s | 128 SSE events per request |

The input-set hashes matched between modes: 16k
`3aa5529da7b91a18d13545ed58b299a55b95b8ee14e65492d66a156219d2973e`,
8k `41c623d9d464abc27f9e8cb906cd9a7536a9202d6c5c0351550b1ce5b0d342e6`.
Output hashes **differed** between modes; the first-code check is narrow and
does not establish generation-quality parity. These are three sequential 16k
cases and two 8k waves, not a broad throughput sweep. The 16k single-request
PVD trace had early-32 inter-token p50 about 0.139 s and late-32 about
0.043 s; full KV stayed near 0.084 s in both segments. The first PVD refresh
waited about 1.1 s at M64. Same-GPU prediction work consumes time between
formal forwards, while the pre-refresh PVD full-bank adapter remains costly;
the measurements do not isolate those two costs per kernel. At the end of
the successful paired run, D GPU1 used 24,670 MiB and GPU0 had no compute
process. The current configuration has no speed advantage over full KV.

The first paired 8k PVD attempt exposed a real multi-request bug: request B
reached lead position 32 while A owned the sole private draft branch, and B
was aborted with `CUDA prediction is active or quarantined`. Commit
`514a88bc5` defers B's refresh with a deadline while B remains eligible for
formal Decode. The next attempt exposed a separate capacity issue: the
1.5 GiB target-prefix reservation held A's cache and B was aborted with
`cooperative target capture requires reserved private prefix cache`.
Commit `faf92fa73` evicts idle private target caches under pressure and
raises this long-context profile's budget to 3 GiB. The final paired run
returned four complete HTTP 200 streams with no D abort. Its trace directly
shows B's prediction deferred at committed token 32 while A predicted at
token 54, followed by a two-row formal target batch that committed A token
55 and B token 33. Thus the second request continues formal Decode while
waiting for the private branch; it may still wait at its own refresh
boundary if its result is not ready.

On V's independent Linux test checkout, the driver and target-probe suites
passed **65 tests** after the behavior and fixture changes. Ruff E/F/I and
format checks passed for the four touched Python files at `e1a942c90`.
One unrelated pytest configuration warning (`asyncio_mode`) remained.
The final style-only commit followed the 65-test run. Client JSON reports
are `validation/logs/{samegpu_long16_3cases,full_long16_3cases,full_8k_pair4,samegpu_8k_cachefix_pair4}.json`
on V. D timeline logs are `validation/logs/d-samegpu_long16.log` and
`validation/logs/d-samegpu_8k_cachefix.log`. The final D serving checkout
was `faf92fa73`; the local branch includes later fixture/style commits.

All experiment service groups were stopped (P 104275, V 213228, D 201835,
Gateway 218395). A final check found no listeners on the test ports and no
GPU compute processes on any node. No Git push occurred. Next, profile the
initial full-Prompt PVD attention adapter and its host metadata/fences, then
measure a shorter same-GPU prediction horizon and a refreshed M value under
matched 16k/8k workloads. Recheck retrieval recall and full output quality
before treating first-code success as acceptance. Preserve the seven
pre-existing dirty files and validation bundles.

## 13. 2026-09-28 same-GPU Decode cost diagnosis

The 16k case-0 request from section 12 was rerun on the same CloudLab lease
with P TP2, V TP2, Gateway, and the 15,875-token input fixed. D target and
draft shared physical GPU1. All three successful runs returned 128 SSE events
and the expected first code. The two PVD runs used M64/lead32/predict32,
cooperative prediction, the 3 GiB private target-prefix cache, a 2 GiB
retrieval-bank budget, and a 4 GiB probe scratch bound. Only
`PVD_CACHE_DECODE_METADATA` changed between PVD runs. The full-KV control
restarted D with predictive serving disabled while P/V stayed running.

| Mode | Wall | TTFT | Decode | Early-32 / late-32 token-gap p50 |
| --- | ---: | ---: | ---: | ---: |
| PVD metadata cache off | 27.667 s | 8.541 s | 19.125 s | 0.1409 / 0.0436 s |
| PVD metadata cache on | 26.923 s | 8.522 s | 18.400 s | 0.1337 / 0.0402 s |
| Full KV, same input | 19.410 s | 8.167 s | 11.243 s | 0.0843 / 0.0846 s |

The input SHA-256 was identical in all three. The two PVD output hashes were
identical; the full-KV output hash differed. First-code success therefore
does not establish full output parity. This is one controlled case, not a
confidence interval; section 12's three-case control showed the same overall
direction. Before the successful runs, two D starts with undersized scratch
or retrieval-bank bounds failed preflight/admission and produced no valid
measurement. A direct predict-8 trial with lead32 was rejected at startup by
`lead_tokens must not exceed predict_tokens`; it handled no requests.

The opt-in `PVD_CACHE_DECODE_METADATA=1` in
`cuda_model_attention.py` reads four GPU metadata vectors once per whole
model `bind()` and reuses the host lists across layers. It requires the same
`ForwardBatch` and tensor objects, validates their shape/device/type on
reuse, and clears the cache when the bind ends. It does **not** cache the
generated Req-to-KV rows, which remain checked per layer. The synchronous
Scheduler runs one bound target forward before processing results or the next
batch. The opt-in remains default-off while broader concurrency and output
validation are incomplete. Its single-case improvement was 0.725 s of Decode
and 0.744 s wall, with identical PVD output hash.

The remote-only timeline aggregate for cache-on explains the remaining gap.
Across 127 formal target batches, measured batch durations summed to 10.709 s,
while the first-to-last formal batch spanned 18.398 s. Thus 7.689 s lay
between measured target batches. This timing scope includes the PVD executor
bind, `run_batch`, and result processing; it is not a pure model-forward or
CUDA-kernel measurement. The first 64 batches cost 8.598 s, with per-batch
p50 0.1268 s for indices 0–31 and 0.1272 s for 32–63. After installing the
sparse bank, the last 63 batches cost 2.112 s with p50 about 0.033–0.034 s.
The private prediction-step trace spanned 9.945 s while formal Decode
continued to interleave. Four V search HTTP calls overlapped in a 0.512 s
wall window. The final pre-refresh formal batch ended 1.045 s before the
refresh became ready. Prediction/search spans are **not additive** to the
7.689 s because work overlaps. The evidence shows both slower pre-refresh
PVD formal batches and substantial same-GPU prediction/scheduling time; the
faster sparse tail nearly offsets the pre-refresh formal-batch cost, but
cannot offset that additional time. It does not isolate each CUDA kernel.

The full-bank PVD adapter uses its own per-Q-head Triton attention and
per-layer generated-row checks instead of the native full-KV Torch SDPA path.
It also has per-layer completion fences unless the separate deferred-fence
opt-in is enabled. These code paths plausibly explain the pre-refresh
forward gap, but no attention-only A/B has yet measured their individual
contributions. A native full-bank fallback needs explicit proof that the
model pool still holds the complete Prompt and must preserve consumer
validation/ownership; do not substitute it solely from bank completeness.

The new `analyze_pvd_decode_timeline.py` emits aggregate numbers on D without
copying request logs. D logs are
`validation/logs/d-metacache16{off3,on,full}.log`; V reports are
`validation/logs/metacache16{off3,on,full}_case1.json`. On V's isolated test
checkout, 53 model-attention tests passed, including two new cache-scope and
tensor-replacement tests; Ruff E/F/I and format checks passed for all three
changed Python files. No Git push occurred. Next compare a coupled shorter
lead/prediction profile and test full-bank fast-path eligibility, while
checking full-output parity and retrieval recall. Preserve the seven
pre-existing dirty tracked files.

## 14. 2026-09-28 matched PVD versus full-KV Decode steps

Reran section 13's same 15,875-token input and 128-output case with the
same P TP2 and V TP2 services. PVD is the metadata-cache-on run above;
the full-KV control was restarted with opt-in
`PVD_PROFILE_FULL_KV_BATCH=1`. Both had 127 logged formal Decode batches.
The full-KV client report is wall 19.448 s, TTFT 8.192 s, Decode 11.256 s;
the matched PVD client report is 26.923 s, 8.522 s, and 18.400 s.
Both returned the expected first code; full output hashes differ. All times
below are seconds. A segment's gap includes the idle time *before* its first
batch (except batch 0); it is not time within `run_batch`.

| Decode part, batch indices | PVD formal batches | Full-KV formal batches | PVD gaps | Full-KV gaps |
| --- | ---: | ---: | ---: | ---: |
| 0–31 | 4.5248 | 2.8172 | 0.2097 | 0.0330 |
| 32–63 | 4.0729 | 2.6686 | 6.0050 | 0.0326 |
| 64–95 | 1.0669 | 2.6728 | 1.2719 | 0.4091 |
| 96–126 | 1.0447 | 2.5911 | 0.2021 | 0.0317 |
| Total | 10.7094 | 10.7497 | 7.6887 | 0.5064 |

The total formal-batch time differs by only -0.0403 s for PVD. The
7.1823 s excess batch-gap time accounts for the approximately 7.145 s
client Decode disadvantage within run-to-run and timestamp differences.
Before the first refresh, PVD formal batches sum to 8.5977 s versus
5.4858 s for full KV (+3.1119 s). After refresh they sum to 2.1116 s
versus 5.2639 s (-3.1523 s). PVD batch p50 is about 0.127 s before
refresh and 0.033–0.034 s after it; full KV remains about 0.083 s.
The PVD full-bank adapter uses a separate per-Q-head Triton path, generated
row checks, and completion fences, but those contributions have not been
individually measured. The measured PVD batch also includes executor bind;
full KV measures `run_batch` plus result processing. Neither number is a
pure attention-kernel duration.

The main 32–63 gap is 6.005 s for PVD versus 0.033 s for full KV. During
this window the private prediction-step trace spans 9.945 s across 65
steps, interleaved with formal Decode. Its span overlaps the measured formal
batches and must not be added to them. The four V search HTTP calls overlap
in 0.512 s wall time; their individual durations sum to 0.972 s, also
non-additive. At the M64 boundary PVD's largest single gap is 1.065 s
before formal batch 64; the preceding formal batch ended about 1.045 s
before `refresh_ready`. Full KV's analogous largest gap is 0.377 s and
includes its full-bank refresh. Thus PVD's V search is reasonably short,
but same-GPU prediction/scheduling and an unhidden first-refresh wait make
the overall Decode slower.

TTFT is a client-observed composite: PVD 8.522 s, full KV 8.192 s,
delta +0.330 s. It includes Gateway, P Prefill, V index/fan-in, D import,
and first output. This run did not log separate P Prefill or D import
durations, so no substage attribution is justified. V logged initial
full-Prompt fan-in around 0.329–0.336 s per rank for both modes and index
build around 0.159–0.175 s per rank. These V operations can overlap other
stages and cannot be added to TTFT. The PVD wall disadvantage is 7.475 s:
0.330 s in TTFT plus 7.145 s in Decode.

`decode.py` now has a default-off full-KV formal-batch timer. The analyzer
accepts `--batch-event full_kv_batch` and reports batch/gap sums by ordinal
batch index so the 127 batches align despite different committed-token
labels. Logs remain on the remote lease: PVD D
`validation/logs/d-metacache16on.log`, full-KV D
`validation/logs/d-stepfull16v2.log`, and full-KV V client report
`validation/logs/stepfull16v2_case1.json`. The detailed log was analyzed
remotely; only aggregates were copied into this handoff. Next isolate the
PVD full-bank adapter overhead and reduce prediction-induced gaps, then
repeat multiple cases and verify exact output parity and retrieval recall.
