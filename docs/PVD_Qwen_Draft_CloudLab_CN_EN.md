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

本步骤**未**加载 draft 权重到 GPU、运行 draft forward、连接生产
Scheduler、测量显存/延迟或证明预测质量。下一硬件门槛是用该 checkpoint
执行独立 prediction-only worker 和目标模型 Q probe，并核对私有池、
预算、正式 Req/模型池不变，以及失败后的所有权回收。

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

No draft weights were loaded on GPU, no draft forward ran, and this step does
not establish production Scheduler activation, VRAM/latency, or predictive
quality. The next hardware gate must run this checkpoint through the
prediction-only worker and target-Q probe while checking private pools,
budgets, committed-state isolation, and failure cleanup.
