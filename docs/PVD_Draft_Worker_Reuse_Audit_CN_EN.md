# PVD prediction-only draft reuse: audit and plan / PVD 仅预测 draft 复用：审计与方案

## 2026-09-22: actual branch-local allocator ownership / 分支分配器所有权

Two failures were reproduced: a single `PrivatePoolAllocator` stored one mutable
request handle and was reused by every factory handle, so the second live
branch failed allocation despite independent handle/admission metadata. Also,
cleanup did not hold the model execution lock and could interleave pool frees
with another branch's allocation/forward.

`SlotAllocator.fork_for_branch()` now explicitly mints branch ownership metadata
without device allocation. The real adapter returns a new request handle over
the SAME private request/KV pool storage, not new pools or copied weights.
Factories refuse allocators missing this contract. Provider retirement uses the
same execution lock as prediction; failed cleanup keeps quarantine/admission
semantics. More than one branch may own rows, but model execution remains serial.
The test allocator double uses a per-index ledger and explicitly opts into this
contract; it must not be mistaken for the real single-request adapter.

New tests cover two live handles, mapping survival after peer retirement,
continuation after peer release, missing-contract refusal and two concurrently
admitted threads. One variant uses actual CPU `ReqToTokenPool` and
`TokenToKVPoolAllocator` in WSL; it still uses a model executor double and is
not GPU/kernel concurrency evidence. The separate real-model gate covers the
standard prediction/Decode path.

Verification: new ownership tests Windows **4 passed / 1 skipped**, WSL **5
passed**, including real CPU pools. Full regression Windows **1823 passed / 15
skipped**, WSL **1829 passed / 9 skipped**. Strict v5 real CPU four-case matrix
passed. Touched files pass formatting; pre-existing Ruff findings are retained.

已复现并修复“多个 handle 共享同一个请求 owner”和“释放未持执行锁”两个问题。
现在每个分支只新建请求所有权元数据，仍共享原私有池和模型权重；前向/分配/释放
串行化，多个分支可以同时持有各自槽位。真实 CPU 分配器验证不等于 GPU 并行证明。

## 2026-09-22: shared-budget accounting / 共享预算计费修复

Eleven regression cases reproduced before the fix: two independent 1024-byte
providers sharing a budget were charged only 1024 bytes because both used the
same idempotent reservation owner; a second provider could bypass capacity.
Missing local persistent settings skipped charging even an explicitly supplied
budget. Size coercion, scratch/persistent aliasing and ignoring a per-provider
bound when an external budget was supplied also bypassed the stated contract.
Each provider now has a unique owner, requires a nonnegative integer footprint,
requires an explicit budget for positive retained bytes, and enforces separate
scratch/persistent accounting plus both local and aggregate bounds. This is
post-load accounting, NOT proof that model loading stayed within that budget.

Factory diagnostics previously retained every opened branch id forever. They
now retain the latest 64 ids plus a locked total counter; returned history is an
immutable snapshot. Persistent charges are deliberately NOT refunded at branch
exit or object collection: externally owned model/pool storage can still be live.
No model teardown or GPU reclamation is inferred from Python reference counts.

Verification: 92 focused draft tests; full Windows **1819 passed / 14 skipped**,
WSL **1824 passed / 9 skipped**; strict v5 real CPU four-case matrix passed.
New tests pass Ruff and touched files pass formatting. Existing Ruff findings
in `draft_sglang.py` / `draft_runner_sglang.py` remain (31 / 19 versus HEAD
31 / 20); unrelated mass typing/style rewrites were excluded.

修复前 11 个测试失败，覆盖 owner 冲突导致漏计、缺失预算/外部预算绕过、类型
强转、预算混用及诊断历史无界增长。现在独立 provider 独立计费，历史最多 64 条。
这不是加载前显存上限保证，也不在每轮预测结束时假装释放仍存活的模型和私有池。

Follow-up / 后续：[target-Q CPU probe](PVD_Target_Q_CPU_Probe_CN_EN.md) 已实现
CPU/TP1 Llama 的离线真实 Q 捕获；本页旧阶段的“真实 target-Q 未实现”不再适用于
该参考子集。生产/GPU probe 与并发服务接线仍待完成。

Latest / 最新：2026-09-21 的 [真实 CPU forward 记录](PVD_Draft_CPU_Execution_CN_EN.md)
补充了 33 次真实 ModelRunner 前向；覆盖下文旧阶段“从未执行”的历史陈述。
GPU/RDMA、真实目标 Q 与生产服务接线仍未验证/未完成。

Audited against HEAD `3b1b4e75b` on 2026-09-21. This records why SGLang's
speculative *generation* path cannot be called for PVD prediction, what is
reused instead, and what remains architecture-dependent.

审计基于 HEAD `3b1b4e75b`（2026-09-21）。记录为何不能直接调用 SGLang 的推测
**生成**路径，改为复用什么，以及哪些部分仍依赖具体模型架构。

---

## 1. What `draft()` actually does / `draft()` 的真实行为

`StandaloneWorker` extends `EAGLEWorker`. One call to `draft()` on a live
batch mutates committed state in at least six ways
(`speculative/eagle_worker.py::_draft_preprocess_decode`):

| # | Mutation | Restored? |
| --- | --- | --- |
| 1 | `req.decode_batch_idx += 1` for every request in the batch | no |
| 2 | `sampling_info.penalizer_orchestrator.cumulate_output_tokens(...)` | no |
| 3 | `batch.maybe_evict_swa()` evicts from the shared cache | no |
| 4 | `assign_draft_cache_locs` writes into `req_to_token_pool.req_to_token` | **no** |
| 5 | `batch.out_cache_loc`, `batch.seq_lens_sum`, `batch.return_hidden_states`, `spec_info.positions` | no |
| 6 | Allocation from the shared KV allocator | yes — `backup_state=True` / `restore_state` |

Only the allocator is rolled back. The request-to-slot mapping, the counters,
the sampler penalties and the batch fields are not.

只有分配器会回滚。请求到槽位的映射、计数器、采样惩罚状态和 batch 字段都不会。

Two further facts make this structural rather than incidental:

* `speculative/standalone_worker.py` takes both pools **from the target**:
  `self.req_to_token_pool, self.token_to_kv_pool_allocator = target_worker.get_memory_pool()`,
  under the comment *"Share the allocator with a target worker."*
* `EAGLEWorker.clear_cache_pool()` is a deliberate `pass`, commented
  *"allocator and kv cache pool are shared with target worker"*.

And the return value is an `EagleVerifyInput` — an object whose purpose is to
be verified and committed, which is the path this project must never enter.

返回值是 `EagleVerifyInput`，其存在目的就是被验证并提交——正是本项目绝不能进入的路径。

**Conclusion: `draft()` is not the reuse point.** 结论：`draft()` 不是复用点。

---

## 2. Reuse plan / 复用方案

| Layer | Decision |
| --- | --- |
| Model loading, config resolution, weight loading, device placement, attention backend | **Reused unchanged** via `TpModelWorker(..., is_draft_worker=True)` — the same construction `StandaloneWorker` performs |
| Memory pools | **Isolated.** `TpModelWorker` accepts both pools as `Optional`; passing neither makes `ModelRunner` allocate private ones. `require_private_pools()` refuses a worker that shares either |
| `draft`, `draft_extend`, `verify`, `forward_batch_generation`, `forward_target_extend`, `capture_for_decode`, `on_verify_complete_cpu` | **Refused.** `PredictionOnlyWorker` raises on *attribute access*, so a future edit cannot quietly reintroduce them |
| Prefix → forward pass | **Architecture-dependent, not implemented.** Behind the `DraftRunner` protocol |

Pool isolation was chosen over sharing-with-reserved-slots so that "prediction
does not mutate committed state" is a structural property rather than a
bookkeeping discipline. The cost is extra device memory, which becomes an
explicitly budgeted quantity (`--pvd-draft-scratch-budget-bytes`).

选择私有内存池而非共享+预留槽位：使"预测不修改已提交状态"成为结构性属性，
而不是依赖簿记正确性。代价是额外显存，该代价被显式预算化。

---

## 3. Configuration / 配置

PVD-owned flags only. **The startup prohibition in
`arg_groups/pvd_disaggregation_hook.py` is unchanged and unconditional**:

```
--pvd-draft-model-path          no default, no hard-coded model
--pvd-draft-revision            optional, recorded for reproducibility
--pvd-draft-device
--pvd-draft-predict-tokens
--pvd-draft-scratch-budget-bytes   required with the model path, no default
```

None of these reads or writes `speculative_algorithm`, so SGLang's speculative
generation loop remains unreachable under PVD. Setting a draft model does not
create an exemption — a regression test asserts the prohibition still fires
with the draft flags set.

这些参数不读写 `speculative_algorithm`，PVD 下仍无法进入 SGLang 的推测生成循环。

---

## 3b. Ownership, cleanup and the prefix / 所有权、清理与前缀

**Branch-owned handles.** Each branch gets a `DraftExecutionHandle` minted by
a factory; the handle owns that branch's request slot, KV rows and scratch,
and is the only thing whose `release` frees them. Weights and pools are **not**
per branch: one model, one worker, one pool set, shared by every handle.

**Execution is serialized.** Separate resource ownership is not a claim that
`ModelRunner` and its attention backend are reentrant, so one lock guards
execution until concurrent use is positively established. The branch limit
bounds how many handles exist, not how many forwards run at once.

每个分支拥有自己的执行句柄（请求槽、KV 行、scratch），但**不**复制模型权重。
资源独立拥有不等于可并发执行，因此执行串行化，直到并发安全被正面证实。

**Release ordering.** Budget and admission capacity are given back only
*after* the resources they stand for. A failed `release` refunds nothing and
returns no admission slot — the rows may still be live, and reissuing them
would hand the same memory to a second owner. The branch is quarantined and
the provider reports itself degraded.

预算与准入名额只在资源真正释放之后归还。释放失败则两者都不归还（行可能仍然存活，
重新发放等于把同一块内存交给第二个所有者），该分支被隔离，provider 标记为降级。

**Two budgets.** `scratch_budget_bytes` bounds per-branch execution scratch;
`persistent_budget_bytes` bounds the model weights and private pools, charged
once. Different lifetimes, different budgets.

**Prefix: recomputed every call.** `prepare_prefix` runs one EXTEND forward
over the whole snapshot into branch-owned rows, and `release` frees them.
Nothing survives a branch, so retained-KV ownership, budgeting and
invalidation do not arise. **This is a correctness baseline, not the latency
answer**: a full prefill per round costs O(prefix), and whether it fits has to
be measured against the prefetch window before anyone calls it sufficient.

Persistent prefix caching would replace `prepare_prefix` and would first need:
a named owner for the retained KV; a persistent budget distinct from
per-branch scratch; incremental extension as the committed prefix grows; and
invalidation on prefix replacement, token retraction, Entry replacement and
request close. None of that is implemented, and none is assumed.

前缀每次调用重算，位于显式的前缀准备接口之后。这是正确性基线而非最终的延迟方案，
其 prefill 开销后续必须与可用预取窗口对比测量。持久前缀缓存需先定义：保留 KV 的
所有者、独立于 per-branch scratch 的持久预算、前缀增长时的增量扩展，以及在前缀替换、
token 回撤、Entry 替换和请求关闭时的失效规则。

---

## 3c. The execution path / 执行路径

`draft_runner_sglang.py` produces `DraftForwardInputs` — ids, positions,
sequence lengths, request-pool indices, KV write locations — and hands them to
a `ModelExecutor`. `ForwardBatch.init_new` takes a `ScheduleBatch`, i.e. the
live scheduler object with tree cache, sampling info and the committed request
list attached; building on it would reintroduce exactly the coupling this
audit exists to avoid. Mapping `DraftForwardInputs` onto a `ForwardBatch` is
architecture- and backend-specific and is the one piece not implemented here.

Capability subset, stated positively: standard autoregressive causal LMs
(`LlamaForCausalLM`, `Qwen2ForCausalLM`), full-attention backends
(`flashinfer`, `triton`, `torch_native`), bounded prefix and prediction
lengths. Anything else is refused before allocation. `require_model` is
checked against what the loaded model reports; `require_shape` against the
request. The two are separate so neither can be satisfied by comparing a
declaration with itself.

---

## 3d. The ForwardBatch mapping / ForwardBatch 映射

`draft_forward_adapter.py` closes the seam `3c` left open, in two halves that
are deliberately not confused with each other:

* `forward_fields()` returns the mapping as **plain data** -- ids, positions,
  sequence lengths, request indices, KV locations, extend bookkeeping.
* `build_forward_batch()` constructs the real `ForwardBatch` from it, through
  an injectable factory.

The split exists because importing `forward_batch_info` pulls in triton,
torchvision and the HTTP stack, which the PVD CPU suite does not require (and
which currently fails against torch 2.14 here). So the tests check two
different things and claim only what each supports: the **values** against a
recording double, and the **field names** by parsing `forward_batch_info.py`
as source -- every key must be a real `ForwardBatch` field, and every field
without a default must be supplied, so an upstream rename fails the suite.

拆成两半是因为导入真实 `ForwardBatch` 会拉入 triton、torchvision 与 HTTP 栈。
因此测试分别验证：**取值**（用记录型替身）与**字段名**（解析上游源码），
两者都不冒充对方。

`ForwardBatch` carries no pools and no attention backend -- the `ModelRunner`
supplies those -- which is what makes direct construction viable, and also
what makes the private-pool decision load-bearing: the runner handed to the
adapter must be the draft runner.

### The private request map / 私有请求映射

Found while writing the adapter: the attention backend locates a request's KV
through `req_to_token_pool.req_to_token[req_index, :seq_len]`, and the runner
was allocating rows without ever recording them. A forward would have read
whatever the row happened to contain. `SlotAllocator` now carries
`write_mapping` / `clear_mapping`; the prefix is mapped before the forward
that reads it, each step extends the map by exactly one position, and
`release` clears the row **before** returning the slot so a later branch
cannot inherit rows it does not own. Every write goes to a row this branch
allocated, in the private pool.

注意力后端通过 `req_to_token[req_index, :seq_len]` 定位 KV，而此前 runner 只分配行、
从不登记。现已补齐：前缀在读取它的前向之前完成映射，每步恰好扩展一个位置，
释放时先清空该行再归还槽位。所有写入都发生在本分支分配的私有行上。

---

## 3e. Contract audit: six interfaces got wrong by assumption / 接口契约审计

Every item below was written from a plausible-looking API instead of a read
one, and every one is wrong against this checkout. They are recorded because
"it compiles and the doubles agree" is exactly how this class of defect
survives — the doubles agreed because they had been written from the same
assumption.

| # | Assumed | Actually |
| --- | --- | --- |
| 1 | `req_pool.alloc(1)` / `free(index)` | `alloc(reqs: list[Req]) -> Optional[List[int]]` assigns `r.req_pool_idx` in place; `free(req: Req)` asserts it is set and clears it. Slot 0 is a padding row, never issued |
| 2 | `output.next_token_logits` | `ModelRunnerOutput.logits_output.next_token_logits`, `[#seq, vocab]`, **Optional**; `logits_output` may be `PPProxyTensors` |
| 3 | capture disabled with `None` | `CaptureHiddenMode.NULL = 0`; the logits processor calls `.need_capture()`, so `None` is an `AttributeError` waiting to happen |
| 4 | `extend_start_loc = (0,)`, no `extend_num_tokens` | `init_new` sets `extend_num_tokens` and derives `extend_start_loc` from `compute_position`; both are required on the extend path |
| 5 | release indices as CPU int64 | `free()` does `torch.cat((free_pages, free_index))`; `free_pages` is int64 **on the allocator's device** |
| 6 | token-wise `alloc(1)` everywhere | `PagedTokenToKVPoolAllocator.alloc` asserts `need_size % page_size == 0` and returns whole pages |

修复要点：为分支自有的请求对象（`DraftRequestHandle`，仅携带内存池会读写的三个属性，
绝不借用已提交请求）；按真实结构解包 `ModelRunnerOutput`，并显式拒绝流水线并行输出与
`None` logits；用 `CaptureHiddenMode.NULL` 而非 `None`，并允许该禁用值、拒绝真实捕获模式；
补齐 `extend_num_tokens` 与按前缀和计算的 `extend_start_loc`；释放张量使用分配器自身的
device 与 dtype；`page_size != 1` 在分配之前直接拒绝。

A seventh, unrelated to the interfaces: the adapter retained every
`ForwardBatch` in `self.batches`, pinning their device tensors for its whole
life. Diagnostics are now bounded — the most recent forward's shapes and
counts, and no tensors — and a regression measures the adapter's own state
after 1 and after 21 forwards and requires it not to grow.

---

## 3f. What now runs against the real classes / 已对真实类验证的部分

The earlier claim that the real imports were blocked was too coarse. The
actual chain, resolved: torchvision from PyPI is built against a different
torch (`RuntimeError: operator torchvision::nms does not exist`) and needs
the CPU wheel; `transformers` must be the pinned `5.8.1` (`5.17` raises
`'qwen3_asr' is already used by a Transformers config`); then `openai`,
`partial_json_parser`, `dill`, `sentencepiece`, `einops`,
`compressed_tensors`, `gguf`.

With those installed, these now use the **real** classes, not doubles:

* `ForwardBatch`, `ForwardMode`, `CaptureHiddenMode` — real objects
  constructed for both EXTEND and DECODE, `capture_hidden_mode` is
  `CaptureHiddenMode.NULL` and `need_capture()` returns `False` on it.
* `ReqToTokenPool` — a real pool allocates and frees through the
  branch-owned request object; slot 0 is confirmed never issued.
* `TokenToKVPoolAllocator` — real `alloc`/`free`, with the release tensor
  built on its own device and dtype.

These tests skip where the imports are unavailable, and the skip reason
carries the exact exception text rather than a guess.

**Construction is still not execution.** No model forward has been run.

以上为真实类的**构造**与内存池**分配/释放**，不是执行；从未运行过任何模型前向。

---

## 4. What is NOT established / 未建立的结论

* No production checkpoint has been loaded. The new smoke loads a random tiny
  fixture through ModelRunner directly. `build_prediction_only_worker` takes an
  injectable worker factory so its *arguments* are tested; its default
  TpModelWorker construction path remains unverified.
* **Production forward validation is still missing.** Real tiny-Llama CPU
  ModelRunner forwards now pass through TorchNativeAttnBackend (latest record
  above); selected checkpoints, GPU backends and TpModelWorker construction
  remain unverified.
* The real-class tests depend on a specific dependency set assembled by hand
  (see 3f). A checkout without it skips them; the CPU suite still passes
  because the source-contract and double-based tests do not need it.
* No GPU was used: nothing is established about CUDA RNG isolation, GPU
  memory safety, or device-side scratch behaviour. Greedy selection is used
  precisely because sampling would need an RNG whose isolation is established.
* Serialized execution is a *conservative default*, not a measurement: no
  evidence was gathered about whether concurrent `ModelRunner` use is safe.
* No target architecture has been selected, so real target-Q extraction is
  **not** implemented; the probe remains an explicit adapter boundary.
* Nothing about prediction quality, speed, or compute overlap between a
  co-located draft and target. Co-location does not by itself overlap compute.

已加载离线随机小模型作为 CPU 回归夹具，未验证正式 checkpoint；未使用 GPU；
未选定目标架构，真实 target-Q 提取**未实现**；
未验证预测质量、速度，也未验证 draft 与 target 共置能否重叠计算。
