# 真实模型稀疏 CPU Decode / Real-model sparse CPU Decode

2026-09-21。本轮未 commit/push；保留之前所有未提交工作。

后续：[跨 rank 安装契约](PVD_Rank_Install_Contract_CN_EN.md) 已通过逻辑/CPU 驱动测试，
但未与本文的模型 backend 组合为线上多 rank pipeline；下文 Next 为该变更前的记录。

最新：[同一真实 CPU Decode 闭环](PVD_Controlled_CPU_Decode_CN_EN.md) 已接通 HTTP
检索结果到实际模型消费，并以统一 group 视图门控整个前向的读取；覆盖下文“检索与
模型消费仍分开”的历史限制。它依然是离线 CPU 驱动，不是线上多 rank pipeline。

## 本轮推进 / Implemented

第三步从数学 reference 推进到真正的 SGLang 模型 attention 调用：

- `sparse_cpu_backend.py` 的 `CPUSparseDecodeConsumer` 消费按 layer/KV-head 分组的
  Prompt bank，并从原 KV pool 读取此前 D 生成的 KV、追加当前 token 的 K/V。
- `make_offline_sparse_backend` 显式构造继承 `TorchNativeAttnBackend` 的离线适配器。
  不注册 CLI/backend registry、不修改原后端默认行为；必须显式创建与绑定才可运行。
- 一个 `bind()` 作用域覆盖完整模型 forward，始终持有所有请求 bank 的 reader。
  作用域结束才允许 bank 被安装/释放；拒绝错 request/incarnation、重复 slot/请求实例、
  错位置、缺层/重复层、非 Decode 与不支持的 attention 变体。
- 在写入当前层 K/V 前检查整批的 generated-row 映射、地址范围、重复与跨请求 alias。
  检索子集各自保留 post-RoPE 绝对位置；不以压缩后的索引重算 RoPE。
  单 token Decode 已限定所有 Prompt 和 generated 位置不晚于 query，无需三角索引 mask。

The opt-in adapter executes through real `ModelRunner → Llama → RadixAttention`.
Prompt KV comes only from the installed per-layer/KV-head bank. Generated KV
stays in the existing pool and is addressed through the generated suffix of the
request map. No Prompt pool rows or Prompt map entries are read by this consumer.
The per-layer scaling factor is preserved; GQA Q heads share their KV-head bank.

Scope: offline CPU FP32, exact Llama, TP1/PP1/CP1, ordinary full causal attention,
page size 1, no overlap scheduling, no speculative decoding/quantization/SWA.
Caller must own the request slots/rows and keep the runner quiescent. Main-thread
checks are not a proof of online concurrency safety. Multi-request/reordered
batches are unit-tested at the pool-consumer level, not via a real Scheduler.

## 实际发现并修复 / Reproduced integration defect

首次真实前向失败：`RadixAttention.attn_type` 是 `AttentionType` 枚举，不是字符串。
原校验错误拒绝了正常 Decode；修正为比较枚举值，unit fixture 也改为枚举，随后重跑
实际模型通过。保留其余能力限制，没有用默认回退绕过失败。

The first real run caught an enum-vs-string assumption invisible to the original
double. The regression fixture now uses the enum contract. This is why both
lightweight tests and real backend execution remain required.

## 严格验收 / Strict evidence

全量回归：WSL **1191 passed / 6 skipped**；Windows **1186 passed / 11 skipped**。
新增 20 个 consumer 测试，包括拒绝、资源作用域、跨请求别名检查，以及不同位置请求
重排后与逐请求独立执行的输出一致。跳过的硬件项仍未验证。

运行原生完整 KV、全选 Prompt bank、子集 Prompt bank 三条路径；使用相同真实模型、
完整前缀和固定 continuation tokens。独立的逐层 hook 用手写 matmul/softmax 核对输出。

- 14 次真实模型 forward 调用，包含一次在实际 sparse attention 后注入的失败。
- 全选 bank 与原后端的最终 logits 最大误差：**0**。
- 13 次 attention 输出对照（12 次成功路径，加一次故障前对照），最大误差：
  **1.1920928955078125e-7**。
- 子集与全量 logits 最大差异 **2.2379446029663086**，确认夹具能检测忽略子集的错误；
  这个差异不是质量改进或退化的测量。
- 原 Prompt pool 写 NaN、Prompt map 改成无效值后，bank 路径仍产生正确有限结果。
- D 先前生成的 K/V 逐值不变；当前 K/V 正常追加。
- 故障后恢复原 backend、释放 bank reader/预算及本次请求的池资源。

On failure, a partially executed model may already have written current-token
KV in earlier layers. There is NO transactional rollback. The fixture aborts
the request and frees its owned rows; production wiring must invalidate/discard
incomplete output before retry. Reader cleanup alone does not make retry safe.

```bash
PYTHONPATH=python /home/loosp/torch311-env/bin/python \
  test/registered/disaggregation/run_pvd_draft_cpu_smoke.py \
  --probe --search --sparse-decode
```

Tiny random Llama is a test fixture, not the user's model choice. No download.
The real HTTP retrieval gate and the real sparse model-consumption gate both
run, but remain separate fixtures: search results are NOT yet fed into an online
model forward through an actual transfer/scheduler pipeline.

## 没有完成 / Not established

- 无生产后端注册/CLI 开关、无在线 Scheduler 接线、无实际多进程 TP 安装。
  后续已有[CPU 安装协议](PVD_Rank_Install_Contract_CN_EN.md)与
  [受控检索安装闭环](PVD_Controlled_Request_Loop_CN_EN.md)，但尚未门控此模型的读取。
- 无 GPU、RDMA、CAGRA 或延迟隐藏验证；未实现生成 KV 压缩或清除。
- attention 临时拼接/计算 workspace 尚无生产预算；CPU bank 预算只覆盖其持有副本。
- 测试保留原完整 Prompt 的池分配（只将内容/映射毒化），**不是实际显存回收证据**。
- 失败处置、source lease、异步传输授权/fence、TP head 分布仍不能由 CPU adapter 推导。

Next: connect the now-tested controlled CPU search/install loop to this real
model-consumption path, with group-gated reads and actual generated-prefix
snapshots. A request may resume only after all expected ranks acknowledge the
same incarnation/operation/boundary. The CPU protocol is not distributed TP
evidence. Preserve generated KV and independent request clocks; new requests
must not cancel or reset existing prefetches.
