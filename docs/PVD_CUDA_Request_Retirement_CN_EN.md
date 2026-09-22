# CUDA 请求原始池回收 / CUDA request pool retirement

`CUDARequestRelease` 将明确注册的 CUDA 请求接入原 `release_kv_cache` 入口。
没有 owner 的请求保持原路径；CPU/CUDA owner 不可竞争同一个 Req。
结束回调只记录 release intent 并停止 controller，不同步等待 HTTP 或远端 WRITE。

The explicit CUDA release owner connects registered requests to the original
release_kv_cache entrypoint. Unbound requests retain their existing path; CPU
and CUDA owners cannot compete for one Req. The callback records release intent
and stops the controller without synchronously waiting for HTTP or remote WRITE.

## 完成顺序 / Completion order

1. 等待同一个 controller 的 aclose 真正成功，且 shared target arbiter 空闲。
2. 取得 shared arbiter、target RLock 和实际池 owner，确认设备完成状态。
3. 原 cache 函数在 deferred-free view 上消费 Req bookkeeping，暂不公布 free rows。
4. 检查释放计划恰好覆盖本请求 allocated KV extent，拒绝重复/非法行。
5. 实际 allocator.free 读取映射中的行号，然后同步；此时映射仍未清零。
6. 清零该请求映射，再同步；此时 host request slot 仍未归还。
7. 归还 host slot，释放 pool pin 和执行许可，最后移除 driver registration。

Wait for successful controller close and an idle target, acquire shared execution
ownership, then let the original cache routine consume bookkeeping through a
deferred-free view. Validate the exact owned KV extent before publishing frees.
Fence allocator reads before clearing the mapping, fence the clear before
returning the host slot, and only then retire the registration and pool pin.

第一版只接受 exact ChunkCache、ReqToTokenPool、page-1 TokenToKVPoolAllocator。
不能在 free_group 中运行，因为其中保存的 mapping view 可能尚未被 allocator
消费。原生 speculative、paged/SWA/Mamba 等其他 allocator 不由此扩展支持。

The baseline accepts only exact ChunkCache, ReqToTokenPool and page-1
TokenToKVPoolAllocator. Retirement inside free_group is refused because that
group may still hold unconsumed mapping views. This does not add support for
paged/SWA/Mamba allocators or native speculative generation.

## UNKNOWN / Failure policy

设备完成或 allocator/slot 更新失败时，保留真实 plan、tensor view、pool pin、
异常和 shared target lease；两个池标记为 poisoned，alloc/free 都拒绝继续。
即使 free list 已部分更新，也不能将对应行重新发给下一请求。不会通过再次
同步、clear()、timeout 或取消来宣称修复；应停止该 worker 并重新启动新实例。

An ambiguous device, allocator or host-slot update retains the actual plan,
views, pool pin, exception and target lease. Both pools are poisoned: alloc/free
refuse reuse even if a free list was partially updated. A later synchronize,
clear, timeout or cancellation is not recovery; stop the worker and start a new
instance. No automatic cleanup retry is performed after an ambiguous mutation.

成功回收的 tombstone 不再持有整个 driver/worker/pool，仅保留拒绝重复释放所需
的身份信息。重复 release callback 不会清零已经被后继请求使用的 slot。

A successful tombstone drops its driver/worker/pool references. A duplicate
release callback never clears a slot that may now belong to a successor.

## 证据边界 / Evidence limits

CPU 故障测试执行仓库真实 allocator/cache 方法源码；WSL 额外使用真实 Req、
ReqToTokenPool、TokenToKVPoolAllocator 和 ChunkCache 类。CUDA placement 与
completion calls 被替代，因此不代表 GPU/RDMA 验收。真实 Req 测试最初因 fixture
缺少全局服务参数失败；已补齐，并加强 post-free fault 断言以避免被无关异常掩盖。

CPU fault tests execute the checkout's allocator/cache method bodies. WSL also
uses actual Req and pool/cache classes. CUDA placement/completion are substituted,
not GPU/RDMA evidence. An initial missing global-server-args fixture was fixed;
the post-free fault now verifies the exact fence count/error rather than accepting
any quarantine as proof that the intended fault executed.

该入口和 driver retirement hook 已实现，但生产 startup/admission factory 尚须
安装 owner，不能因为文件存在就认为所有服务请求已启用此生命周期。
The entrypoint and driver hook exist; the production startup/admission factory
must still install the owner. Their presence alone does not activate this
lifecycle for all serving requests.

本步验证 / Step evidence: Windows **2192 passed / 28 skipped**;
focused WSL **45 passed** (three existing CPU-platform warnings); strict v5
real-model CPU matrix **all four scenarios passed** after the pool/common-release
changes. Thirteen new tests include two actual-class cases skipped on Windows
and executed in WSL. No GPU/RDMA test was run.
