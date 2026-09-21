# Rank installation control / 跨进程 rank 安装控制

## Contract / 契约

`rank_install_wire.py` defines `pvd-rank-install-v1`: PREPARED, PARKED,
APPLIED, INSTALL, RESUME, RESUMED and FAILED. Frames are UTF-8 JSON bytes, at most 16 KiB;
individual strings are bounded to 1024 UTF-8 bytes and counts to signed-int64
nonnegative values. Duplicate/unknown/missing fields, nonfinite numbers, boolean
counts, wrong message direction and ambiguous schemas are refused.

Every frame names the full request/Entry/incarnation/operation/round/boundary,
rank, layout and staging identity plus the bound worker incarnation. Peer epochs
must be established from a trusted channel, not accepted from an arbitrary
sender's self-declaration. This is not authentication or a network transport.

`RankInstallExchange` connects these frames to the existing logical coordinator.
It issues INSTALL only after every rank is prepared AND parked, and RESUME only
after every rank acknowledges application. Partial application, missing ranks,
stale workers, unknown staging and old rounds cannot authorize resume. Membership
is explicit, not fixed to TP2, and is immutable for this request incarnation.

消息绑定完整请求、Entry、轮次、边界、rank、staging 与 worker epoch。协调端只能在
全部 rank 准备并暂停后发送 INSTALL；全部应用确认后才能发送 RESUME。消息有大小、
字段和数值边界；重复 JSON 字段、旧身份、错方向或缺失字段均拒绝。epoch 绑定必须
来自可信通道，不能把对端自报的 epoch 当成身份认证。

## CPU participant / CPU rank 本地参与者

`CPURankInstallParticipant` owns one caller-provided CPU bank. Initial admission
requires the full Prompt at boundary zero; later epochs follow the configured
request-local interval. Existing synchronous readers must drain before PARKED.
Once parked, no new reader may enter; local installation emits APPLIED but
remains unreadable until the exact RESUME. Duplicate INSTALL/RESUME cannot swap
or advance the clock twice. A prior round's RESUME cannot clear a newer bank.

The participant is owner-thread-only. It must exclusively control its bank:
callers must not bypass it with raw bank reads or mutations. Cancellation closes
the read gate but does not release active readers. Cleanup is explicit; a failed
local installation permanently closes this request's read gate. No rollback is
attempted. Generated Decode KV is not managed or evicted by this Prompt bank.

Each participant owns only its rank's bank. No all-rank tensor view is exposed.
The adapter supports CPU FP32 only. Reader scopes are NOT CUDA events or RDMA
fences. Future GPU participants need their own proven visibility and ownership.

每个参与者只拥有本 rank 的 CPU 工作集；旧读者未结束不能报告 PARKED。即使本地已
安装新 KV，也必须等待协调端的全体安装确认后 RESUME 才能读取。取消、部分安装异常
都不能提前恢复读，也不能强制释放仍被读者持有的资源。D 生成的 KV 不归此模块回收。

## Failure boundary / 失败边界

No dynamic membership, retry rollback or cross-restart recovery is implemented.
On peer loss, the transport/scheduler owner must fail or cancel the entire request
and drain/close all peers. Cancellation messages can name only received prepared
receipts: a missing receipt may be a lost reply, NOT proof that the peer owns no
resources. Timeouts and process loss are never GPU/MR completion evidence.

## Evidence / 验证

56 unit tests cover strict wire framing, incarnation/channel binding, independent
membership, both barriers, duplicate/reordered/stale messages, reader draining,
partial install failure, retained charges and terminal cleanup. These are CPU
contract tests. Multiprocess acceptance is implemented below; production TP
collectives, GPU attention, native RDMA and Scheduler integration remain open.

56 个 CPU 单测已覆盖协议边界和本地读门控；单测本身不是生产 TP/GPU 集成。
后续独立进程验收见下节，每个进程持有各自 bank，通过有界字节消息验证这些约束。

Full regression: Windows 1549 passed / 14 skipped; WSL 1554 passed / 9 skipped.
New modules/tests pass Ruff. Skipped hardware cases are not treated as passed.

## Independent-process acceptance / 独立进程验收

```bash
python test/registered/disaggregation/run_pvd_rank_install_cpu_smoke.py --ranks 2
python test/registered/disaggregation/run_pvd_rank_install_cpu_smoke.py --ranks 4
python test/registered/disaggregation/run_pvd_rank_install_cpu_smoke.py --ranks 2 --fault install
python test/registered/disaggregation/run_pvd_rank_install_cpu_smoke.py --ranks 2 --fault exit
```

This executable uses `multiprocessing` **spawn**, not forked copies of the parent's
banks. Every child constructs its own real CPU K/V tensors and participant. The
parent owns the coordinator only, verifies distinct PIDs, and exchanges bounded
JSON bytes over local pipes. No tensor, pointer, pickled model or bank travels
through the control channel. Fixture setup creates deterministic K/V inside each
worker; it is not V-to-D payload delivery. Bootstrap bypasses package initializers
only; the actual PVD coordinator, message codec, participant and CPU bank execute.

The successful scenarios install the full Prompt at boundary zero, hold an old
reader while preparing the next bank, refuse early park/installation, then install
the selected token subset at boundary four. Exact paired K/V values are checked
in each rank before and after. Duplicate commands do not install twice; an old
RESUME cannot clear the next round. APPLIED is deliberately read-blocked until
all ranks apply and RESUME arrives.

The failure scenarios apply on one rank, then either raise after another rank's
local swap or terminate that CPU fixture process. The coordinator closes the
request without advancing its committed installation boundary. No global RESUME
can be generated and the already-applied surviving rank remains unreadable.
All live ranks close with zero bank charges. For a killed rank, cleanup is
explicitly **not proven**; process death is not generalized to GPU/MR safety.

新增独立可执行验收：使用 spawn 启动 2/4 个 CPU 进程，各自构造并持有真实 K/V；
父进程只做协调，pipe 中只传有界 JSON 控制字节。验证首轮完整 Prompt、刷新子集、
读者排空、旧消息拒绝、重复消息幂等和全体安装后才恢复读。故障场景在部分 rank
安装后注入换 bank 异常或进程退出，协调端不能推进边界或发送 RESUME。
存活 rank 的预算全部归零；被终止进程不宣称已证明资源清理。

Four subprocess regression tests invoke these exact commands. This is **real
multi-process CPU control**, but still **not model tensor parallelism**, NCCL/
Gloo integration, cross-node control, CUDA reader completion, RDMA, CAGRA or a
production Scheduler. Do not relabel `--ranks 4` as a working GPU TP4 deployment.

Final full suite with all four real-process scenarios: Windows **1553 passed /
14 skipped**, WSL **1558 passed / 9 skipped**. Both environments execute the
process scenarios; these are not hardware skips. New source/tests pass Ruff.

## Resume receipt gate / 恢复回执门控

Sending RESUME is not evidence that a peer received it. Wire admission now uses
`RankInstallExchange.can_decode`, not the logical coordinator's `can_decode`.
Every peer sends RESUMED with its exact staging identity after reopening its
local gate. No next round or global dispatch is allowed until every expected
RESUMED arrives. A lost reply is retried by reissuing the same RESUME; the peer
replies idempotently, without another bank swap or clock advance. Resending
commands does not clear ACK progress. Stale/wrong worker, epoch or staging cannot
open the gate; cancellation overrides collected ACKs. Update both ends together:
older peers without RESUMED leave admission closed, not silently compatible.

发送成功不代表对端收到 RESUME。线协议用户必须使用 Exchange 的准入门控；只有
全部 rank 返回匹配的 RESUMED 后才允许全局执行或下一轮。回执丢失可重发同一
RESUME；重复回执不会重复安装或推进时钟。旧端没有此回执时保持关闭，不能静默
混用。该回执不是 GPU/RDMA 完成证明，也不替代 Scheduler 的正式 forward 调度。

Five added cases and updated independent-process scenarios verify the added
gate, including withholding the final ACK and idempotent retry. Windows full
suite: 1558 passed / 14 skipped; WSL: 1563 passed / 9 skipped. GPU/TP serving
remains unimplemented.

Follow-up: [owner-polled rank runtime](PVD_Rank_Runtime_CN_EN.md) supplies bounded
callback queues, fixed round deadlines, automatic barrier progression and
request-scoped failure notifications. It uses the RESUMED gate above and has
real CPU process tests for lost PREPARED/RESUMED and rank failure. It still needs
production Scheduler, transport and resource-ownership integration.
