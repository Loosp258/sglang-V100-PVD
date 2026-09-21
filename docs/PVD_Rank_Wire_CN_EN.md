# Rank installation control / 跨进程 rank 安装控制

## Contract / 契约

`rank_install_wire.py` defines `pvd-rank-install-v1`: PREPARED, PARKED,
APPLIED, INSTALL, RESUME and FAILED. Frames are UTF-8 JSON bytes, at most 16 KiB;
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
contract tests. A separate multiprocess acceptance is the next step; production
TP collectives, GPU attention, native RDMA and Scheduler integration remain open.

56 个 CPU 单测已覆盖协议边界和本地读门控；本阶段尚不是实际多进程验收，更不是
生产 TP/GPU 集成。下一步用独立进程持有各自 bank，通过有界字节消息验证这些约束。

Full regression: Windows 1549 passed / 14 skipped; WSL 1554 passed / 9 skipped.
New modules/tests pass Ruff. Skipped hardware cases are not treated as passed.
