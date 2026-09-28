# 16k PVD prediction-horizon sweep (2026-09-28)

## Setup

The CloudLab V100S sweep kept P (clgpu020, TP2), V (clgpu021, two GPU
ranks), Gateway, and their model and index settings running while D
(clgpu019, TP1 target and 0.5B draft on GPU1) was restarted for each profile.
D used commit `ae7e8c709`, TorchNative target attention, the concurrent
predictor, the 384 MiB draft-prefix cache, the 3 GiB target-Q prefix cache,
M64 refreshes, and the same memory budgets. The only profile changes were
`--pvd-draft-predict-tokens=N` and `lead_tokens=N` in the D limits JSON.
The code requires `predict_tokens >= lead_tokens`; this sweep therefore
measures the combined shorter-prediction/shorter-lead policy, not a
single-variable change to the draft length.

Each D profile received one excluded 16k/256-output warmup. The measured
request used seed `compare-cached-full-16k-measured`, 15,875 actual prompt
tokens, input SHA-256
`489223019dc08934ff78b586cc86f53e9c1fb1297ff8d0757b97f5a6de9046d4`,
temperature 0, `ignore_eos=true`, and 256 streamed output tokens. Each row is
one measured request, so subsecond differences are not a stable ranking.

| Predict / lead | Client wall (s) | TTFT (s) | Decode (s) | First capture (s) | Later captures (s) | M64 / M128 / M192 boundary gaps (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 / 32 | 74.927 | 8.475 | 66.452 | 37.091 | 6.483 / 6.501 | not logged in this repeat |
| 16 / 16 | 68.164 | 8.670 | 59.493 | 35.056 | 3.771 / 3.706 | 0.056 / 0.285 / 0.272 |
| 8 / 8 | 63.373 | 8.457 | 54.915 | 32.608 | 2.159 / 2.163 | 0.403 / 0.550 / 0.595 |
| 4 / 4 | **61.650** | 8.468 | **53.181** | 31.827 | 1.481 / 1.452 | 0.591 / 0.746 / 0.719 |
| 2 / 2 | 62.190 | 8.474 | 53.716 | 32.289 | 1.206 / 1.205 | 0.847 / 0.937 / 0.915 |

The same-day no-refresh full-KV control from
`PVD_CACHED_VS_FULL_KV_20260928.md` took 29.742 s wall and 21.596 s Decode.
It was not rerun as part of this sweep; the 32-token row was rerun with the
same P/V/Gateway services as the shorter profiles. Its output SHA-256 matched
the prior 32-token baseline. The prior 32-token timeline measured boundary
gaps of 0.023 / 0.010 / 0.018 s, but those are a separate request and are
not inserted into the sweep table.

## Where the time moved

The 16/8/4/2 runs logged 255 formal batches each. Their formal batch sums
were 14.801 / 14.920 / 14.807 / 14.868 s. Their time between batches was
44.690 / 39.993 / 38.372 / 38.846 s, respectively. For predict4, A paused
after the lead position for 31.682 / 1.431 / 1.460 s at M64/M128/M192. The
corresponding V search plus delivery took 0.987 / 0.884 / 0.899 s; with only
four formal tokens left, the later boundary gaps rose to 0.746 / 0.719 s.
For predict2, the later captures saved about 0.25 s each relative to predict4,
but the later boundary gaps grew by about 0.19 s each and the first capture
varied upward by 0.46 s. Its 0.54 s wall regression is too small to establish
a reliable optimum from one observation.

Reducing the horizon from 32 to 4 saved 13.277 s in this measured request,
but predict4 remained 31.908 s slower than the earlier no-refresh full-KV
control. The first private long-prefix capture still took about 32 s. Shorter
prediction alone therefore does not solve the cold target-Q prefix work.

## Quality and limits

All five measured requests returned 256 tokens and the expected first fact
code. Their full output hashes differed across profiles (predict16 and
predict2 happened to match each other), so first-code success is not full
output parity. A separate predict4 check used the earlier `longpvd0927`
three-case, approximately 16k-prompt/128-output input set: all three first
codes were correct, matching the earlier full-KV control's 3/3. The input-set
SHA-256 matched (`3aa5529da7b91a18d13545ed58b299a55b95b8ee14e65492d66a156219d2973e`),
but the three complete output hashes differed from full KV. This is a narrow
fact-recall check, not a retrieval-recall or general quality validation.

The observed latency knee is near predict4/lead4 for this single request.
Before changing the default, repeat 4 and 8 with multiple seeds and
concurrent requests, measure actual retrieval recall and answer quality, and
separate first-prefix capture from the per-refresh draft and target-Q stages.

## Evidence

Client reports are `validation/logs/predsweep{32,16,8,4,2}-measured.json`
on clgpu021. D logs are `validation/logs/d-predsweep32.log`,
`d-predsweep16timed.log`, and `d-predsweep{8,4,2}.log` on clgpu019. The
predict4 three-case quality report is
`validation/logs/predsweep4-quality3.json` on clgpu021. Copies of the sweep
reports and D logs are in the local excluded `.pvd-compare-artifacts/` folder.
Run `python scripts/pvd_compare_decode_logs.py LOG --event target_batch
--run-index 1` for the 16/8/4/2 measured-request timeline. The 32 repeat
omitted `PVD_PROFILE_REFRESH_TIMELINE=1`, so its D log contains refresh-stage
timing but not batch/gap events.

After the sweep, this session terminated its P/V/D/Gateway process groups.
The corresponding service ports and `nvidia-smi` compute-process lists were
empty on all three nodes.
