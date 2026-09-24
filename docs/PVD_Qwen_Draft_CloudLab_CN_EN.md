# CloudLab 独立 draft 模型 / Independent draft checkpoint

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
预测入口仍必须逐次检查实际 token ID 是否落在共享 tokenizer 范围，
超出时拒绝该分支，不能截断、重映射或假定嵌入矩阵尺寸相同。

随后在 D GPU0 用现有 `run_pvd_cuda_draft_smoke.py` 加载该真实 checkpoint：
Qwen2 架构、FP16、context 64、模型 token 容量 128。独立
prediction-only runner 对 4-token 输入做两步贪心预测，本次得到
`[16, 15]`，与逐步重算完整前缀的对照一致；两次 forward 后私有
Req/KV 池的可用容量恢复原值，进程退出后 D GPU 无残留计算进程。
报告的已知常驻张量为 **998041680 字节**，不是峰值显存。

该 smoke 在**单独进程**中运行，目标 7B 未同时驻留，亦未运行目标 Q
probe、V 检索或生产 Scheduler；不能据此声称 draft 与正式 Decode
共存、并行或有预测质量/延迟收益。下一硬件门槛需在同一 D 进程同时
加载两模型，核对私有池与正式 Req/模型池隔离，并执行目标 Q probe。

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
tokenizer range; no truncation, remapping, or equal embedding-size assumption
is justified.

The existing `run_pvd_cuda_draft_smoke.py` subsequently loaded this real
checkpoint on D GPU0 (Qwen2, FP16, context 64, model-token capacity 128).
Its isolated prediction-only runner greedily predicted **[16, 15]** from a
four-token prefix, matching independent full-prefix recomputation at both
steps. The private Req/KV pool capacity returned to its starting value, and
no D GPU compute process remained after exit. Known retained tensors totaled
**998041680 bytes**; that is **not** a peak-VRAM measurement.

This was a **standalone process**: the target 7B was not resident at the same
time, and neither target-Q probing, V retrieval nor the production Scheduler
ran. It does not establish model coexistence, concurrent execution,
predictive quality or latency benefit. The next gate must load both models in
one D process, verify private-pool/committed-state isolation and run target-Q
probing with the draft tokens.
