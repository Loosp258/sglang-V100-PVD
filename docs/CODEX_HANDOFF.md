# Codex handoff — PVD disaggregation (2026-09-27)

This is the starting point for a **new Codex session**. Read this file, then
inspect the current checkout; do not infer success from an old conversation or
from historical sections of other documents. The final performance goal is
**not achieved**. Do not push to GitHub unless the user explicitly changes the
current instruction: make a **local commit after each verified step; no push**
until the final goal is met and the user permits it.

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

## 5. CloudLab environment and precise current remote state

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
