# CPU batch 执行与真实多请求验证 / CPU batch execution and real multi-request validation

2026-09-21。上一步生命周期已提交 `81bfb64b4`，未推送；本轮后续修改尚未提交。

## 本轮一次完成的内容 / Delivered together

1. `cpu_batch_dispatch.py`：一个 batch 共用一个 target lease，固定 dispatch 成员、
   request ID/incarnation/每请求唯一票据；结果按身份匹配，不按后来 batch 的位置猜测。
2. `cpu_batch_forward.py`：可复用真实 CPU executor，以真实 `ForwardBatch` 调用
   `ModelRunner`，整个前向持有所有请求/分片的 Prompt 工作集 reader，返回所有 logits 行。
3. `pvd_batch_decode_smoke.py`：同一真实 tiny Llama 上的多请求生成、HTTP 检索、
   边界安装、成员加入/重排/取消及实际模型错误注入。

The batch dispatcher prevalidates every member before acquiring one lease or
assigning any permit. If any selected member must wait, **the whole selected
batch waits**. It does not silently filter out waiting requests. Cancelling a
member during execution cannot release the shared lease; only batch completion
or failure retires all permits. Individual lifecycle completion is refused for
a batch-owned permit.

## 身份、存储和输出约束 / Identity, storage and output contracts

- 必须返回完整的成员结果集；缺失、重复、外来、旧票据或非法 token/finish 标记，
  在任何 token 追加之前拒绝。拒绝后保留所有权，由驱动显式 `fail()` 收尾。
- 正常完成时，仅仍在运行的成员提交；取消/超时成员输出丢弃，其他成员可继续。
- 整个模型前向失败时，所有本批成员终止且不提交本次输出；已写入的 KV 不回滚。
- 相同 dispatch 不可重复执行/提交。失败调用只能在实际同步执行和 reader scope
  已退出后发出，不可把提前发 fail 当作 GPU/native 完成证明。
- 执行器要求预先注册 lifecycle→request slot。目的地址顺序可以重排，但槽位不得
  串给其他请求；本批 slot/KV row 必须唯一。调用者负责实际槽位/KV 分配和 mapping。
  注册并不是分配器授权证明；终止且无活跃 Decode 之后解除绑定，再由驱动回收槽位。
- 使用完整 `[batch_size, vocabulary]` logits。原 draft adapter 的 `forward()` 仅返回
  最后一行，不能直接用于多请求提交；这里只复用它的 `build_forward_batch()`。
- `batch_results_from_logits` 是离线 greedy 测试适配，不改变正式 sampler、EOS 判断
  或 `req.output_ids` 提交点。finish flags 必须由驱动显式提供。

No tensor/ForwardBatch history is retained. The executor stores a last operation
ID for replay rejection and a caller-managed request-slot registry. The latter
must be explicitly unbound during teardown; it does not free pool rows itself.
Attention temporaries and CPU ledger allocations are still not production-budgeted.

## 实验和证据 / Executed evidence

```bash
PYTHONPATH=python /home/loosp/torch311-env/bin/python \
  test/registered/disaggregation/run_pvd_draft_cpu_smoke.py \
  --probe --search --sparse-decode --controlled-decode --batch-decode
```

- batch sizes：`[1,1,1,2,2,2,1,1,1,2]`，包含 Prompt 长度不同、绝对位置不同的请求。
- 旧请求 3 token 时预取，等待 HTTP 期间新请求加入，同批前进到旧请求边界 4；
  整批暂停，安装后恢复。新请求加入不改变旧 epoch/时钟。
- 下一批交换成员顺序，并把结果倒序交回 dispatcher，仍提交到正确请求。
- 在一个真实 batch 前向完成但输出未提交时模拟取消新请求，其输出被丢弃；
  旧请求继续至边界 8，完成正式前缀补查，再生成第 9 个 D token。
- 第三个请求加入后，真实 attention 内注入错误：本批所有输出均不提交。
  最终 committed D counts：old=9、new=2、third=0。
- 19 次逐层独立 softmax 对照，最大绝对误差 `2.384185791015625e-7`。
  生成 KV 历史不变；原 Prompt 池/映射毒化后仍正确消费工作集；预算和池容量恢复。
- 执行器重放与跨请求 slot 交换在真实 forward 前拒绝。
- 新增 39 个 CPU 契约测试：整批预校验、等待、共享 lease、身份/乱序/重复/取消/
  EOS/异线程、完整 logits 行与目标 slot/row 校验。

全量回归：Windows **1331 passed / 11 skipped**；WSL **1336 passed / 6 skipped**。
上述完整严格 smoke 在可复用 executor 和槽位绑定接入后再次通过。Ruff check、
format check、`git diff --check` 通过。跳过项不作为原生/GPU 验收证据。

The first real multi-request run exposed a fixture integration error:
`PrivatePoolAllocator` owns exactly one request slot per instance. The fixture
now uses one allocator handle per request over the shared runner pools; it does
not weaken that existing ownership check. Strict real-model validation was
rerun after the correction. This is separate from mock-based contract tests.

## 距离最终目标 / Remaining work

This is a reusable **offline CPU** execution path, not a running production
Scheduler, actual distributed TP, GPU sparse attention, or RDMA pipeline.
The selected batch's wait-all policy is established; fairness and selection
across batches are still the serving Scheduler's responsibility.

下一步应成组推进：实际 `ScheduleBatch`/结果处理接点、请求 retraction/finished
生命周期、正式模型执行互斥，以及稀疏模式的显式能力开关和失败拒绝。不能把当前
CPU 工作集指针直接给 GPU backend，也不能让镜像输出账本重复追加正式 Req 输出。

还缺生产稀疏交付（源 lease、目标授权、MR/GPU fence、TP 一致安装）、GPU attention、
可配置真实 draft 的组合验证、V100S CAGRA、真实召回/质量和 TPOT/吞吐/显存实验。
这些不只是等待硬件的测试项，仍有实际实现缺口；不能声称最终 PVD 目标已完成。
