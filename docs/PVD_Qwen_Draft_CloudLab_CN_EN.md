# CloudLab 独立 draft 模型 / Independent draft checkpoint

2026-09-24 复测：在 D 节点新建隔离检出 `ecf94fb07` 的 `sglang` 代码上，
复用原实验目录的只读双模型 smoke 脚本，同卡目标 7B + draft 0.5B
再次通过。预测 token 为 `[13, 2585]`，28 层目标 post-RoPE Q 的位置为
9、10；已知同卡峰值 CUDA allocated **16328797184** 字节、reserved
**16435380224** 字节。测试结束后两张 GPU 均显示 0 MiB 使用量。
脚本来自原实验目录，导入的 `sglang` 来自新检出；不是生产服务、RDMA、
检索或延迟隐藏验收。

2026-09-24 rerun: using `sglang` from the new isolated D checkout at
`ecf94fb07`, the read-only dual-model smoke script from the previous
experimental directory passed again with target 7B and draft 0.5B on one
GPU. It predicted `[13, 2585]` and captured target post-RoPE Q at positions
9 and 10 in all 28 layers. Observed same-GPU CUDA peaks were **16328797184**
allocated and **16435380224** reserved bytes; both GPUs reported 0 MiB used
after exit. This does not validate production serving, RDMA, retrieval or
latency hiding.

2026-09-24 在隔离的 D 节点工作区下载官方
[Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)，
固定 revision `7ae557604adf67be50417f59c2c2f167def9a775`，位置为
`/mnt/sglang-data/yiliu124-node-2-sglang-pvd/models/Qwen2.5-0.5B-Instruct`。
它只是本次硬件实验的**可更换 draft**；目标模型仍是三节点已有的
`/proj/edgecut-PG0/models/Qwen2.5-7B-Instruct`。代码不得硬编码这些路径。

下载后检查：draft 占用约 954 MiB，权重文件约 943 MiB；两个模型的
`tokenizer.json` SHA-256 同为
`c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`。
通过项目的 `VocabularySignature.from_tokenizer()` 实际加载两侧 tokenizer，
得到相同的词表大小 **151643**、EOS **151645** 和 probe 指纹
`acfab8ea50c649a2ee8dd929c4099117870bb53dd4a5589476460d34ce4b7f73`。
两侧均为 `Qwen2ForCausalLM`。模型配置中的 `vocab_size` 却不同：
目标 **152064**、draft **151936**；这不是 tokenizer 签名差异。
预测入口仍必须逐次检查实际 token ID 是否属于共享 tokenizer 的精确 ID 集，
超出时拒绝该分支，不能截断、重映射或假定嵌入矩阵尺寸相同。

随后在 D GPU0 用现有 `run_pvd_cuda_draft_smoke.py` 加载该真实 checkpoint：
Qwen2 架构、FP16、context 64、模型 token 容量 128。独立
prediction-only runner 对 4-token 输入做两步贪心预测，本次得到
`[16, 15]`，与逐步重算完整前缀的对照一致；两次 forward 后私有
Req/KV 池的可用容量恢复原值，进程退出后 D GPU 无残留计算进程。
报告的已知常驻张量为 **998041680 字节**，不是峰值显存。

上述第一轮 smoke 在**单独进程**中运行。随后新增的
`run_pvd_qwen_dual_draft_gpu.py` 在同一 D GPU0 进程同时驻留目标 7B 和
draft 0.5B，执行两步 draft 贪心预测 `[13, 2585]`，并由目标模型在
28 层各捕获位置 9、10 的 post-RoPE Q（每层形状 `[2, 28, 128]`）。
目标权重/KV/Req 映射的 canary 与 CPU/CUDA RNG 均未变化，draft 私有
Req/KV 容量从 `(1, 4096)` 恢复到相同值，scratch 与 probe 预算归还。
已知常驻 draft 张量为 **1046839328 字节**，同卡峰值 CUDA allocated
**16629784576 字节**、reserved **16670261248 字节**。这些是一次输入的
观察值，**不是**生产显存上界或并发/吞吐数据；V 检索、RDMA 和生产
Scheduler 仍未由该测试执行。测试进程退出后 GPU 无残留计算进程。

真实运行发现并修复两个 draft 启动问题：`ModelRunner` 的设备配置必须是
平台类型 `cuda`（实际 GPU 仍由 `gpu_id` 指定，显式 `cuda:N` 先校验与之
一致）；关闭 disaggregation 必须使用字面值 `"null"`，不能用 `None`，
否则会错误初始化 Mooncake。回归测试覆盖这两个语义。

后续复测已修正 token 边界：`tokenizer.vocab_size=151643` 是基础词表大小，
EOS ID 为 151645；真实 tokenizer 的 `get_vocab()` 给出 **151665 个**
有效 ID。两侧完整 token→ID 映射 SHA-256 均为
`a6edb125c5dc615dad0179399ac171ea6315addbe5607c72e19097fcb8fc3597`。
draft 前缀/输出、目标 probe 均按精确 ID 集检查，模型 padding 行不因此
视为合法。回归测试覆盖 added EOS、ID 空洞和未探测 token 的映射差异。
同卡真实 gate 在此修复后再次通过；它实际生成的仍是 `[13, 2585]`，
并未通过真实模型输出路径触发 EOS。

On 2026-09-24, the official
[Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
was downloaded to the isolated D-node worktree at
`/mnt/sglang-data/yiliu124-node-2-sglang-pvd/models/Qwen2.5-0.5B-Instruct`,
pinned to revision `7ae557604adf67be50417f59c2c2f167def9a775`.
It is a **replaceable experimental draft**, not a hard-coded requirement;
the target remains the existing Qwen2.5-7B-Instruct checkpoint on all nodes.

The draft directory occupies about 954 MiB, including a 943 MiB weight file.
Both `tokenizer.json` files have SHA-256
`c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`.
The project's loaded `VocabularySignature` values match exactly: tokenizer
size **151643**, EOS **151645**, and encoded-probe fingerprint
`acfab8ea50c649a2ee8dd929c4099117870bb53dd4a5589476460d34ce4b7f73`.
Both architectures are `Qwen2ForCausalLM`. Their model-config `vocab_size`
values differ (**152064** target, **151936** draft), so every actual token
crossing the model boundary must still be checked against the shared
tokenizer's exact ID set; no truncation, remapping, or equal embedding-size assumption
is justified.

The existing `run_pvd_cuda_draft_smoke.py` subsequently loaded this real
checkpoint on D GPU0 (Qwen2, FP16, context 64, model-token capacity 128).
Its isolated prediction-only runner greedily predicted **[16, 15]** from a
four-token prefix, matching independent full-prefix recomputation at both
steps. The private Req/KV pool capacity returned to its starting value, and
no D GPU compute process remained after exit. Known retained tensors totaled
**998041680 bytes**; that is **not** a peak-VRAM measurement.

The first smoke above was a **standalone process**. A subsequent
`run_pvd_qwen_dual_draft_gpu.py` gate kept the 7B target and 0.5B draft
resident together on D GPU0. Two greedy draft tokens **[13, 2585]** drove
target-model post-RoPE Q capture at positions 9 and 10 in all 28 layers
(shape `[2, 28, 128]` per layer). Target weight/KV/request-map canaries and
CPU/CUDA RNG were unchanged; the draft's private Req/KV capacity returned
from `(1, 4096)` to the same value, and scratch/probe budgets were refunded.
Known draft retained tensors were **1046839328 bytes**. The observed
same-GPU CUDA peak was **16629784576 allocated** and **16670261248 reserved**
bytes. This is one input, **not** a production VRAM bound or concurrency/
throughput result. V retrieval, RDMA and production Scheduler activation did
not run. The process exited without a residual GPU compute process.

The real run exposed two draft-startup defects that are now fixed: SGLang's
`ModelRunner` needs the platform type `cuda` in its configuration (the actual
GPU remains selected by `gpu_id`, after validating an explicit `cuda:N`), and
disaggregation must be disabled with the literal `"null"`, not `None`, or a
second Mooncake engine is initialized. Regression tests cover both semantics.

A later rerun corrected token-ID validation: `tokenizer.vocab_size=151643` is
the base vocabulary, while EOS has ID 151645; the real `get_vocab()` mapping
contains **151665 valid IDs**. Both complete token→ID maps have SHA-256
`a6edb125c5dc615dad0179399ac171ea6315addbe5607c72e19097fcb8fc3597`.
Draft prefix/output and target probe now check exact ID membership, not an
embedding/padding bound. Regressions cover added EOS, ID holes and changed
unprobed token mappings. The same-GPU real gate passed again, though its
actual generated IDs remained `[13, 2585]`; it did not emit EOS.
