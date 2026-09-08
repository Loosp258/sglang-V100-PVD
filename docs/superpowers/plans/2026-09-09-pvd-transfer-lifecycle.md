# PVD Transfer Lifecycle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 P→V、V→D 的超时、取消和失败路径中，阻止尚在途的 RDMA 使用已释放或已复用的内存，同时限制保留资源的数量。

**Architecture:** 保持业务结果和原生传输状态独立，由每个共享 engine 的生命周期管理器持有句柄、张量及资源引用。发送端报告真实终态，接收端关闭授权并确认终态后回收。不能查询终态的传输进入有界隔离并停止新传输，不修改 Mooncake 原生依赖。

**Tech Stack:** Python 3.12、pytest、PyTorch CPU/CUDA、aiohttp、Mooncake transfer engine 0.3.13.post1。

**Spec:** `docs/superpowers/specs/2026-09-09-pvd-transfer-lifecycle-design.md`，用户于 2026-09-09 确认首版方案。

## Global Constraints

- 保留 fresh metadata 的版本锁定与启动前配置。
- 底层改为异步不意味着允许 D 用未完成的 KV 做 forward；现有 Decode 刷新屏障保持。
- 本轮保持 Router 选择 P/V/D、完整 Prompt KV、每 M token 刷新、D 生成 KV 常驻、Entry/EntryShard/Delivery 分离，以及现有 TP/rail 配置。
- 本轮不修改外部 Mooncake 仓库，也不自动升级依赖。
- 原生提交失败丢失句柄时，采用隔离和协调重启；不声称能够可靠自动回收。
- `--pvd-transfer-staging-budget-bytes` 和 `--pvd-transfer-max-inflight` 使用显式正整数配置；普通 PD 不要求这些参数。
- 每步记录实际测试命令、失败原因和通过结果；CPU 测试不替代真实 RDMA 验收。
- 工作分支为用户指定的 `pvd-disaggregation`。执行前使用 using-git-worktrees 检查隔离环境；不擅自丢弃工作区改动或改名该分支。

## 文件边界与执行约定

新增三个小模块，避免把所有状态塞进现有大文件：

- `python/sglang/srt/disaggregation/pvd/transfer_lifecycle.py`：原生状态、资源 pin、容量和共享生命周期所有权；不依赖 HTTP 或 Coordinator。
- `python/sglang/srt/disaggregation/pvd/transfer_authorization.py`：目标写入授权、身份和 fence 状态；不直接调用 Mooncake。
- `python/sglang/srt/disaggregation/pvd/transfer_progress.py`：有界后台 poll/控制同步；不执行模型 forward。

现有 `transfer_engine.py`/`mooncake_engine.py` 接入管理器；`vector_store.py` 保护页分配；`client.py`/`control_server.py`/`coordinator.py` 承载授权协议；`runtime.py`/`conn.py`/`decode_refresh.py` 负责角色集成。配置仅修改 `server_args.py`、`arg_groups/pvd_disaggregation_hook.py` 和 `pvd/server.py`。

以下命令均在仓库根目录运行，使用 `.venv/Scripts/python.exe`，不使用系统 Python。测试 runner 绕过 GPU 初始化，但不替代真实实现对象。新增测试全部通过 `test/registered/disaggregation/run_pvd_cpu_tests.py` 执行。

提交是阶段检查点，不表示阶段间代码可部署。任务 1–7 完成之前不得把半接入的异步路径部署到实验机器；当前同步消费者把 PENDING 当作失败，必须全部迁移后再发布。

## Task 1: 独立传输状态、资源保留和容量原语

**Files:** 创建 `python/sglang/srt/disaggregation/pvd/transfer_lifecycle.py`；修改 `python/sglang/srt/disaggregation/pvd/transfer_engine.py`；创建 `test/registered/disaggregation/test_pvd_transfer_lifecycle.py`。

**Interfaces:** 新增 `TransportState`（NOT_SUBMITTED、IN_FLIGHT、DRAINING、TERMINAL_SUCCESS、TERMINAL_FAILED、UNKNOWN）和 `TransferHandle.transport_state`，默认 NOT_SUBMITTED。`ResourceGuard(value, release)` 强引用 value；`pin(owner: str)`、`unpin(owner: str)`、`request_release()` 使用幂等 owner 集合及锁。只在请求回收且集合为空时调用 release；回调异常保留 value 并允许显式重试。新增 `TransferBudget(staging_bytes: int, max_inflight: int)`，`reserve(owner: str, byte_count: int, slots: int) -> None`、`release(owner: str) -> None`、`snapshot() -> dict`，超额抛 `TransferCapacityError`，同 owner 同参数幂等，不同参数拒绝。

- [ ] 写失败测试，包括以下真实资源回收探针；补充两个 owner、回调失败、重复释放以及 UNKNOWN 不归还额度。

```python
def test_guard_retains_source_until_transport_unpins():
    released = []
    source = object()
    guard = ResourceGuard(source, lambda: released.append(source))
    guard.pin("write-1")
    guard.request_release()
    assert released == []
    guard.unpin("write-1")
    guard.unpin("write-1")
    assert released == [source]

def test_budget_rejects_before_allocation():
    budget = TransferBudget(staging_bytes=64, max_inflight=1)
    budget.reserve("a", 64, 1)
    with pytest.raises(TransferCapacityError):
        budget.reserve("b", 1, 1)
    budget.release("a")
    budget.reserve("b", 64, 1)
```

- [ ] 运行 `& .venv/Scripts/python.exe test/registered/disaggregation/run_pvd_cpu_tests.py test/registered/disaggregation/test_pvd_transfer_lifecycle.py -q`，确认新增接口缺失导致失败，而非环境导入失败。
- [ ] 实现上述接口；`request_release()` 的核心条件为 `release_requested and not owners and not released`。回调不能在业务锁中运行；用独立 releasing 标记阻止并发重复释放。状态判定只允许 NOT_SUBMITTED 和两个 TERMINAL 状态作为本地不再在途的依据，UNKNOWN 永远不算安全。
- [ ] 重跑同一命令，全部通过；再跑 `test_pvd_core.py`，确认既有 FakeTransferEngine 同步行为不变。
- [ ] 检查改动范围并提交：`feat(disaggregation): add PVD transfer lifetime primitives`。

## Task 2: 原生异步句柄与共享管理器

**Files:** 修改 `python/sglang/srt/disaggregation/pvd/mooncake_engine.py`、`transfer_engine.py`、`transfer_lifecycle.py`；创建 `test/registered/disaggregation/test_pvd_mooncake_lifecycle.py`；更新现有 `test_pvd_mooncake_metadata.py` 中依赖同步调用的断言。

**Interfaces:** 新增 `TransferLifecycleManager`，构造接收 Task 1 的 budget；`attach(handle, source_guard, byte_count)`、`mark_unknown(handle, reason)`、`complete(handle, success: bool)`、`request_cancel(handle)`、`snapshot()`。manager 放在共享 Mooncake wrapper 的 PVD 私有属性，通过模块锁只创建一次。`TransferEngine.poll()` 保持返回业务 `TransferStatus`；原生状态从 handle 读取。取消后晚成功不能把 CANCELLED 改成 SUCCESS。

- [ ] 创建有调用计数的 native stub：`transfer_submit_write` 返回预设 ID；`transfer_check_status` 从列表弹出值。测试提交 0/异常、轮询 0/-2/1/-1、重复 poll、不同 adapter 共享 engine、unregister 失败仍保留张量。

```python
def test_native_terminal_is_polled_only_once():
    native = NativeStub(submit_result=7, statuses=[0, -2, 1])
    # make_adapter_with_source 在本测试文件定义，沿用 metadata 测试的
    # CUDA buffer/native wrapper stub；仅替换外部 GPU 和 Mooncake 边界。
    adapter, local, remote = make_adapter_with_source(native)
    handle = adapter.submit_put(local, remote)
    adapter.abort(handle)
    for _ in range(4):
        adapter.poll(handle)
    assert native.check_calls == [7, 7, 7]
    assert handle.status == TransferStatus.CANCELLED
    assert handle.transport_state == TransportState.TERMINAL_SUCCESS
```

- [ ] 运行新增测试文件，先确认旧 adapter 不满足异步序列和取消保留语义。
- [ ] 改提交/轮询的分支如下；原生调用按句柄串行化，绝不在终态后重复检查。

```python
# 提交前完成 bounds/rail/CUDA 同步以及 source pin。
native_id = native.transfer_submit_write(peer, source_address, target_address, size)
if native_id == 0:
    manager.mark_unknown(handle, "native submit returned no trackable handle")
else:
    handle.backend_handle = native_id
    handle.transport_state = TransportState.IN_FLIGHT

# 已在 handle 锁内，且 handle 尚未终结时：
result = native.transfer_check_status(handle.backend_handle)
if result in (1, -1):
    manager.complete(handle, success=result == 1)
elif result == -2:
    handle.transport_state = TransportState.DRAINING
elif result != 0:
    manager.mark_unknown(handle, f"unexpected native status {result}")
```

进入原生 submit 后抛异常走 UNKNOWN；原生调用前校验失败保持 NOT_SUBMITTED。manager 不拥有远端张量，只能通过后续授权协议要求接收方保留。启动检查异步 API 与已有 metadata policy；不能改成同步 fallback。
- [ ] 运行生命周期、metadata 两组测试与 Ruff 关键错误检查；确认原有 metadata 启动失败策略仍生效。
- [ ] 提交：`fix(disaggregation): retain native PVD write handles until terminal`。

## Task 3: 写入授权与身份完整的 fence

**Files:** 创建 `python/sglang/srt/disaggregation/pvd/transfer_authorization.py`；修改 `protocol.py`、`client.py`、`control_server.py`、`coordinator.py`；创建 `test/registered/disaggregation/test_pvd_transfer_authorization.py`。

**Interfaces:** 新增不可变 `WriteIdentity`：`protocol: str` 固定 `pvd_transfer_lifecycle_v1`、`sender_epoch: str`、`receiver_epoch: str`、`transfer_id: str`、`region_id: str`、`generation: str`、`shard_rank: int`、`key: KVEntryKey`。实现 `to_dict()/from_dict()` 严格校验类型和空值。`WriteAuthorization(identity, guard)` 在创建时 pin；`begin(identity)` 原子检查未关闭且未开始；`close()` 阻止以后 begin；`observe_terminal(identity, state)` 只接受 NOT_SUBMITTED/两个 TERMINAL；`fence(identity) -> dict` 返回原身份及 `fenced`，未关闭或未终结为 false。

- [ ] 写身份篡改、旧协议、关闭后迟到 begin、双重 begin、UNKNOWN fence 的失败测试。

```python
def test_closed_authorization_rejects_late_submission():
    identity = make_identity()  # 本文件用所有明确字段构造 WriteIdentity。
    guard = ResourceGuard(object(), lambda: None)
    authorization = WriteAuthorization(identity, guard)
    authorization.close()
    with pytest.raises(ValueError):
        authorization.begin(identity)
    assert authorization.fence(identity)["fenced"] is False
```

- [ ] 跑该测试文件，确认缺失授权约束导致失败。
- [ ] 实现身份序列化和授权状态；begin/close 同锁，NOT_SUBMITTED 也必须来自已关闭的发送端授权确认。目标描述符只在 guard pin 后返回。相同 ID 不同描述符拒绝，旧 epoch 不继承新 epoch 的安全结论。
- [ ] 将 `fence_retrieval` 扩展为 `fence_retrieval(delivery_id: str, identities: list[dict]) -> dict`，HTTP 和 local shard adapter 均传完整身份。聚合条件为 `all(reply.identity == expected and reply.fenced is True)`，缺少任一 shard 不成功；不能只比 delivery ID。HTTP 用现有可信 V 地址，不接收任意用户回调 URL。
- [ ] 跑序列化和真实 aiohttp 控制接口测试，验证旧 `{"fenced": true}` 回复被拒绝；提交 `feat(disaggregation): fence PVD writes by authorization identity`。

## Task 4: V Entry 页与发送 staging 保护

**Files:** 修改 `python/sglang/srt/disaggregation/pvd/vector_store.py`、`coordinator.py`；创建 `test/registered/disaggregation/test_pvd_vector_lifecycle.py`。

**Interfaces:** EntryShardRecord 持有 allocation 的 `ResourceGuard`；DeliveryShardRecord 持有 handle、source guard 和 Task 3 授权。新增 `VectorKVStore.progress_transfers() -> None`，短锁获取待处理列表，锁外 poll，再短锁提交状态；所有 cancel/reap/close 路径只申请释放。业务 active 计数和 guard owner 集合分开。

- [ ] 在测试文件创建 `DelayedTransferEngine(FakeTransferEngine)`：提交仅记录源/目标并返回 IN_FLIGHT；测试显式 `finish(handle, success)` 时才执行 copy 或设置原生失败。继承并使用真实注册区域映射，不替换 allocator/store/fence。
- [ ] 添加以下场景并验证旧行为失败：同构在途时 cancel Entry 不归还页；异构 TP 的 staging 在晚完成前不注销；一个 Entry 的两个 Delivery 只有一个完成时不回收；TTL/close 不越过 upload pin。

```python
# 使用现有 test_pvd3.make_vector/make_ready_entry 构造真实 Entry，
# 将 store 的外部 engine 换为上述可控延迟 engine 后执行：
before = store.allocator.available_pages
store.start_delivery(key, delivery_id)
store.cancel_entry(key, "injected timeout")
assert store.allocator.available_pages == before
assert store.fence_delivery(key, delivery_id, identities)["fenced"] is False
engine.finish(handle, success=False)
store.progress_transfers()
assert store.fence_delivery(key, delivery_id, identities)["fenced"] is True
```

- [ ] 实现 source pin 在 submit 之前取得、终态后解除；staging 的 finally 只 request_release，不真正提前 unregister。移除持有 store 锁执行 native submit/poll 的等待；使用授权 begin 与提交保留状态阻止关闭竞态。
- [ ] 将 Coordinator 的 start/retrieve 改为可返回 transferring；调用方等待终态通过状态查询，不长期占据 entry/retrieval 锁。新增 client/HTTP `poll_delivery(delivery_id: str) -> dict`，不得因首轮 PENDING 取消。
- [ ] 跑新增测试、test_pvd_core.py、test_pvd3.py；提交 `fix(disaggregation): pin V allocations across in-flight transfers`。

## Task 5: P 上传终态确认

**Files:** 修改 `python/sglang/srt/disaggregation/pvd/runtime.py`、`conn.py`、`coordinator.py`、`client.py`、`control_server.py`、`vector_store.py`；创建 `test/registered/disaggregation/test_pvd_upload_lifecycle.py`。

**Interfaces:** `PVDEntryLease` 增加每 shard 的上传身份；创建 Entry 时先 pin V 目标页。新增 `PVDCoordinatorClient.sync_upload(identity: dict, state: str, closed: bool) -> dict` 与 `/uploads/sync` 路由，返回 `identity/close_requested/terminal_ack`。V 需要关闭上传时记录 close_requested，由 P 的有界后台同步取得；不需要在 P 上新增公网回调服务。P 停止新提交并在原生终态后报告 closed=true；失联则 V 保留 allocation。

- [ ] 写测试：上传 PENDING 不发布 STORED；P 业务取消但 WRITE 晚到；V 先取消而 P 迟到提交；终态通知丢失重试；P epoch 改变不能释放旧上传。

```python
# upload_pair 是本测试文件用真实 runtime/coordinator/store 和延迟
# transport 构造的 fixture，提供 tick() 驱动一次双方状态同步。
pair = await upload_pair()
await pair.cancel_on_v()
await pair.tick()
assert pair.v_pages_reusable is False
pair.finish_native(success=False)
await pair.tick()
assert pair.v_pages_reusable is True
assert pair.entry_is_stored is False
```

- [ ] 运行新增文件，确认旧 cancel_entry/finally 路径无法保证上述顺序。
- [ ] P publish 使用 retained handle 等待而不是 poll 一次；业务 deadline 只记录失败并转移到后台 drain。P 本地上传授权的 close 与 submit 同步；本地校验失败也向 V 关闭已发出的目标授权。成功 commit_shard 必须伴随匹配上传终态，失败后晚成功只回收。
- [ ] 同步记录归 manager 所有，不能随着 sender.clear()/abort() 丢失。P 原始模型 KV 在 packing 完成且 staging 独立后可依旧调度，但发送 staging 必须保留。运行取消/通知重试/现有 Prefill adapter 测试。
- [ ] 提交 `fix(disaggregation): retain upload buffers until P to V drain`。

## Task 6: D 接收回收与连续批处理屏障

**Files:** 修改 `python/sglang/srt/disaggregation/pvd/decode_refresh.py`、`runtime.py`、`conn.py`、`retrieval.py`；创建 `test/registered/disaggregation/test_pvd_decode_lifecycle.py`；更新 `test_pvd3.py`。

**Interfaces:** 每次 refresh 的目的 region generation 独立，session 持有 Task 3 的 identities 和 receive guard；`prepare()` 先 pin 后返回 descriptor。新增 `PVDDecodeSession.progress_close() -> bool`：驱动一次 fence，未排空返回 false，不无限新建协程；返回 true 才解除本轮目标 pin。`PVDDecodeRuntime.deliver()` 同样使用 Task 4 的 poll 和 Task 3 fence，不能保留绕过保护的旧入口。

- [ ] 用现有 CPU session/scheduler fixtures 测试正确 ID 错 epoch、错误 generation、漏 shard、旧版 fence、HTTP 超时与晚成功；验证所有情况在终态确认前都不注销目标。

```python
assert await session.progress_close() is False  # fence pending
assert session.registration is not None
assert native.unregister_calls == []
client.reply = matching_terminal_fence
assert await session.progress_close() is True
assert native.unregister_calls == [destination_address]
```

- [ ] 验证旧实现会接受弱身份 fence 或缺少单步清理接口；再实现 pending 查询与有界后台清理交接。
- [ ] 成功路径先等所有 shard 传输终态，再 CUDA 同步/回填，再 ACK 和 clock.complete；下一 refresh 才复用 staging。错误路径所有 TP rank 达成一致，不执行 forward，移交 receive guard 到后台记录。保留尾页 generated KV 测试。
- [ ] 测试首次刷新、每 M token 刷新、多序列、一个 rank 错误的 collective 次数一致；提交 `fix(disaggregation): fence decode destinations before reuse`。

## Task 7: 启动配置、有界进度与故障状态

**Files:** 创建 `python/sglang/srt/disaggregation/pvd/transfer_progress.py`；修改 `python/sglang/srt/server_args.py`、`python/sglang/srt/arg_groups/pvd_disaggregation_hook.py`、`python/sglang/srt/disaggregation/pvd/server.py`、`conn.py`、`metrics.py` 及任务 1 的预算接入处；创建 `test/registered/disaggregation/test_pvd_transfer_admission.py`。

**Interfaces:** `TransferProgress` 持有 manager 的有界资源记录，`tick() -> None` 推进 native poll，`async tick_control() -> None` 推进 upload sync/receive fence，`snapshot() -> dict` 暴露 outstanding 和 isolated。每 worker 一个进度驱动，不为失败请求无限创建任务；用有限数量并发控制请求，HTTP 重试有退避。启动能力为 `pvd_transfer_lifecycle_v1`，epoch 使用进程启动 UUID。

- [ ] 写严格正整数配置测试（None/0/负数/bool 拒绝，普通 PD 不变）、超额时 torch.empty 尚未调用、连续超时预算不增长、双 adapter 共用上限、UNKNOWN 熔断；所有 rank 先 prepare 再汇总结果。

```python
budget.reserve("isolated", 64, 1)
for i in range(100):
    with pytest.raises(TransferCapacityError):
        budget.reserve(f"rejected-{i}", 64, 1)
assert budget.snapshot()["staging_bytes"] == 64
assert budget.snapshot()["inflight_slots"] == 1
```

- [ ] 运行测试确认缺少 admission/共享上限导致失败。
- [ ] 配置默认 None，但 PVD 启动必须显式指定两个正整数；配置值同步到 V group 每个子进程和 P/D 每 rank。预算覆盖 P packing、V 重排、D 接收；先按布局计算字节，reserve 后再分配。已有 V pool 容量覆盖目标 allocation，但授权条目还计入 slots。session 复用不重复收取 staging 字节；活动/隔离都不退费。
- [ ] 限制已关闭授权记录的增长：同一 sender/receiver epoch 下使用关闭的授权域（session/Entry generation）拒绝所有旧请求后，才压缩域内 tombstone；仍可接收新请求的域不按 TTL 删除。域数量受 admission 限制，无法安全压缩时拒绝新域，不删安全记录腾位置。为迟到 start 添加压缩后回归测试。
- [ ] health 显示 readiness、native metadata policy、协议能力、epoch、inflight/draining/unknown、字节预算/峰值/拒绝数。shutdown 停止 admission 并 drain，超过关闭期限报告未排空并保留资源，不返回假成功。真实原生进程退出需协调停掉相关发送者，不能在退出 hook 中强制释放在途内存。
- [ ] 运行真实 aiohttp 多 shard 测试、TP 错误传播测试及完整 CPU 回归；提交 `feat(disaggregation): bound PVD transfer admission and quarantine`。

## Task 8: 验收文档与发布检查

**Files:** 修改 `python/sglang/srt/disaggregation/pvd/README.md`，新增 `docs/superpowers/verification/2026-09-09-pvd-transfer-lifecycle.md`。

**Interfaces:** 不新增运行时代码；文档使用前述 health 字段及 CLI 参数，写清版本锁、协调重启和未完成的硬件验证。

- [ ] 执行全部现有四组及七组新增测试：

```powershell
& .venv/Scripts/python.exe test/registered/disaggregation/run_pvd_cpu_tests.py test/registered/disaggregation/test_pvd_core.py test/registered/disaggregation/test_pvd3.py test/registered/disaggregation/test_pvd_rails.py test/registered/disaggregation/test_pvd_mooncake_metadata.py test/registered/disaggregation/test_pvd_transfer_lifecycle.py test/registered/disaggregation/test_pvd_mooncake_lifecycle.py test/registered/disaggregation/test_pvd_transfer_authorization.py test/registered/disaggregation/test_pvd_vector_lifecycle.py test/registered/disaggregation/test_pvd_upload_lifecycle.py test/registered/disaggregation/test_pvd_decode_lifecycle.py test/registered/disaggregation/test_pvd_transfer_admission.py -q --tb=short
```

- [ ] 用 `.venv/Scripts/python.exe -m ruff check` 检查新增文件和修改文件的 E9/F63/F7/F82；对新增文件完整 lint；运行 `git diff --check`。检查所有 submit_put/poll/abort/release_memory 调用点，不能遗留“业务 FAILED 等于 drained”的分支。
- [ ] 在验证文档写入实际计数、CPU 无 RDMA 限制、修改覆盖清单。真实机器操作先正常请求、再并发/顺序刷新、再获准的故障注入；每项检查传输 ID、终态日志、pin 归零、内存不增长以及生成 token 正确性。没有远端实验权限时该项记为未运行，不编造通过结论。
- [ ] 使用 requesting-code-review 技能审查终态/并发/容量/协议兼容；修复发现并重跑相关测试。依 verification-before-completion 核对最终 diff 与证据。
- [ ] 文档提交 `docs(disaggregation): document PVD drain verification and recovery`。仅在用户要求时推送，不自动创建 PR；最终交付明确哪些代码测试完成、哪些硬件实验尚未验证。

## 计划自审与执行记录

覆盖映射：设计 §3–4 → 任务 1–2；§5 两段协议 → 任务 3–6；§6–7 容量/重启/协议 → 任务 7；§8 验证 → 每任务红绿测试及任务 8。记录中不把设计或计划完成写成修复已完成。

执行时按任务填勾，不跨过失败的验证步骤。实现遇到无法保持接口或安全不变量的原生行为时，先报告证据并修订计划；不得用释放隔离资源或接受旧 fence 的方式使测试通过。
