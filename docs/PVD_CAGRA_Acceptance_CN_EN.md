# CAGRA 验收边界 / Acceptance gate

## 2026-09-24 稀疏交付与真实 Decode 的安全交错 / Safe interleaving of sparse delivery and real Decode

在隔离的三节点 Qwen2.5-7B 实验中，P 提交真实 Prefill first token **198**，
V 两 rank 建立原生 CAGRA 索引，D 在正式生成计数 3 发起两路稀疏交付。
验证程序确认两个**本地**接收记录已开始发布且交付任务仍未完成，随后执行
第 4 次目标模型 Decode 前向；前向结束后等待两路 Mooncake/RDMA 终态及
写入 fence，才在计数 4 安装工作集。第 5 次目标前向消费新稀疏 KV；
112 组安装值与独立 Prompt KV 逐字节一致，Entry 和预算均回收。
初始/刷新查询的最小并集召回分别为 **0.9772727273/0.98**。

这里的本地 `published` 标志在首次 HTTP await **之前**设置，不能证明
V 已接收请求或 RDMA WRITE 已与 GPU 前向同时进行。交付任务在前向开始时
pending 也不是网络/计算重叠的时间测量；本门槛仅证明该执行顺序的安全性，
**不证明延迟隐藏、吞吐收益或生产 Scheduler 自动调度**。

In the isolated three-node real-Qwen2.5-7B gate, P committed Prefill token
**198**, V built native CAGRA indexes on both ranks, and D started two sparse
deliveries after three committed target Decode forwards. Both **local**
receiver publications had started and the delivery task was pending when D
ran its fourth target forward. Only afterward did D await both terminal
Mooncake/RDMA write proofs and install the new bank at count four. The fifth
forward consumed that sparse KV; all 112 installed groups matched independent
Prompt KV byte-for-byte, and Entry/budgets were retired. Initial/refresh
minimum union recalls were **0.9772727273/0.98** in this run.

The local `published` flag is set **before** the first HTTP await, so it does
not establish V acceptance or that RDMA WRITE overlapped GPU computation.
A pending task at forward start is not a latency measurement. This gate
establishes safe ordering only, **not latency hiding, throughput gain, or
automatic production Scheduler integration**.

## 2026-09-24 全 112 组 GQA 并集召回 / All-112-group GQA union recall

对上述真实 Qwen2.5-7B 的确定性 1024-token Prompt，D 使用独立模型
Prompt K 和每组 7 个目标 Q 计算精确 Top-10 并集，作为 V 原生 CAGRA
并集的对照。全部 **28 层 × 4 KV heads = 112 组**都参与测量；召回定义
为“CAGRA 并集与精确并集交集大小 / 精确并集大小”。首次查询最低召回
**0.9772727273**、平均 **0.9994345149**，112 组中 109 组完全一致；
正式前缀位置 1027 的刷新查询最低 **0.98**、平均 **0.9998214286**，
111 组完全一致。随后同一次运行仍完成 RDMA 稀疏安装、第 5 次模型
前向以及 Entry/预算回收。这是**一个 Prompt、一次刷新**的经验值，
不是跨请求/长度/随机种子的召回保证，也不是延迟或吞吐测试。

For the deterministic 1024-token real-Qwen2.5-7B Prompt, D computed an
exact Top-10 token union from its independently generated Prompt K and all
seven target Q heads per group. All **112 layer/KV-head groups** were compared
with V native CAGRA, using intersection-over-exact-union as recall. Initial
queries had minimum **0.9772727273**, mean **0.9994345149** and 109 perfect
groups; actual-prefix refresh queries at position 1027 had minimum **0.98**,
mean **0.9998214286** and 111 perfect groups. The same run completed RDMA
sparse installation, a fifth target forward and resource retirement. These
figures cover **one prompt and one refresh**, not a quality distribution or
a latency/throughput guarantee.

## 2026-09-24 P 真实 first token 到 D / Real P first token reaches D

最新三节点验收取消了测试种子 `42` 和协议占位 first token。P 对真实
Qwen2.5-7B 1024-token Prefill logits 贪心取样，本次输出 token ID **198**；
随 rank0 的 Prompt KV commit 将其作为 `FirstTokenMetadata` 保存于选定
Entry。D 根据明确的 Entry key 调用 Coordinator `select`，逐项核对
STORED 状态、Entry key、first-token 类型及词表范围，再使用从该 Entry
读取的 **198** 执行自身原生对照、目标 Q 捕获、完整 bank Decode 和后续
4-token 稀疏刷新。D 报告 `p_first_token_id: 198`；首次完整 bank 前向
相对原生 dense 对照的最大 logits 绝对误差 **0.01171875**，生成 K/V
最大误差 **0.03125**，贪心输出相同；正式计数 4 的稀疏 bank 被下一次
目标前向消费。P/V/D 资源正常回收。这里使用的是确定性人工 Prompt token
序列和贪心采样，不代表面向用户的生产请求链路已启用。

The latest three-node gate removed both the test seed `42` and placeholder
first-token metadata. P greedily sampled token ID **198** from actual
Qwen2.5-7B Prefill logits and committed it with the selected Entry's Prompt
KV. D used the explicit Entry key to call Coordinator `select`, validated
the stored Entry identity and first-token type/range, and then used that same
**198** for its dense oracle, target-Q probe, full-bank Decode and subsequent
four-token sparse refresh. D reported `p_first_token_id: 198`; the initial
full-bank forward differed from native dense Decode by at most
**0.01171875** in logits and **0.03125** in generated KV, with the same
greedy output. A later real forward consumed the sparse bank after count
four. P/V/D owners drained. The prompt token sequence remains a deterministic
test input, and this does not activate a production user-request path.

The following older 2026-09-24 sections record earlier, narrower gates;
their `42` seed and one-Q-head limitations are superseded by this result.

## 2026-09-24 真实 Decode 计数驱动的稀疏刷新 / Generated-token-driven sparse refresh

`run_pvd_qwen_native_bank_gpu.py --model-forward --generated-refresh` 在
CloudLab 三节点真实 Qwen2.5-7B 上通过。D 在初始完整 Prompt bank 上执行
**4 次真实目标模型 Decode 前向**，逐次将生成 K/V 写入并保留在 D 模型池，
由这些正式前向的贪心输出驱动下一 token。第 3 次之后，用当前正式 token
前缀在位置 **1027** 做目标模型 Q 补查；每 KV head 的全部 7 个 Q heads
检索 V 的 Prompt 索引，按同组有界并集取 K/V。第 4 次前向仍读取旧 bank；
随后在正式计数 4 完成双 V RDMA 接收、安装、ACK，并让第 **5 次**
目标模型前向读取新的稀疏 bank。D 的工作集/接收预算和真实生成行最终
回收，V 两 rank 的 Entry/索引清空。验证脚本输出 `passed`。

该离线验证以 token ID **42** 作为 D 的初始输入；它不是 P 对本次输入
真实采样的 first token。目标 Q 使用正式前缀的**迟到补查**，并非独立
draft 小模型的提前预测；检索/网络传输先于边界前向等待完成，因此**没有
证明计算与网络重叠或延迟隐藏**。第五次输出仅校验有限值与完整执行，
未作质量/召回分布评估，也未接入生产 Scheduler。

The real Qwen2.5-7B three-node gate passed with `--generated-refresh`.
D performed **four actual target-model Decode forwards** over the complete
Prompt bank, retained their generated KV locally and used greedy target
outputs to drive the next input. After step three, it captured target Q at
position **1027** from the current official token prefix (late fallback),
searched all seven GQA query heads per KV head and fetched bounded Prompt
unions from both V shards. Step four still read the old bank; at committed
count four D installed/ACKed the new sparse bank, and a **fifth real target
forward consumed it**. D owners drained and both V Entry/index records were
empty afterward.

The gate seeds D with token ID **42**, not a token sampled by P for this
prompt. The refresh Q is actual-prefix fallback, not independent draft
prediction. Delivery completed before the boundary forward, so this test
does **not** establish compute/network overlap or latency hiding. It checks
finite execution of the fifth forward, not retrieval-quality distributions,
and production Scheduler integration remains open.

## 2026-09-24 完整 GQA Q-head 并集 / Full GQA Q-head union

真实 Qwen2.5-7B 三节点测试现对每个 KV head 使用对应的 **7 个**目标模型
post-RoPE Q heads，而不再只选一个代表 head。V 对每路 Q 各取 Top-10，
在同一层/KV head 内合并 token、去重，并限制为最多 **70 token/组**；
不跨层或 KV heads 合并分数。全部 112 组的实际并集大小为 **17–65**。
D 在模拟边界 4 从两个 V rank 接收并安装这些并集的 K/V，值与独立
目标模型 Prompt KV 逐字节相同；初始完整 bank 仍完成真实模型 Decode
前向，输出 token 与 dense 对照相同。V 的 Entry/索引记录与 D 的接收、
聚合、工作集预算均回收。模拟边界 4 的稀疏 bank 尚未运行后续模型前向。

The real three-node Qwen2.5-7B gate now searches all **seven** target-model
post-RoPE Q heads mapped to each KV head. V takes Top-10 per query and
deduplicates the token union **within** each layer/KV-head group, bounded by
**70 tokens per group**; no cross-layer/head score merge occurs. The observed
union sizes across all 112 groups were **17–65**. D received and installed
these K/V unions at synthetic boundary four; all values matched its
independent target-model Prompt KV bit-for-bit. A real Qwen Decode forward
still consumed the complete initial bank with matching greedy output.
The sparse bank has not yet been consumed by a subsequent model forward.

## 2026-09-24 真实 Qwen Decode 消费远端 Prompt bank / Real Qwen Decode consumes remote Prompt bank

在上一节完整工作集安装测试上增加 `--model-forward`，D 用同一真实 FP16
Qwen2.5-7B checkpoint、同一 1024-token Prompt 和位置 1024 的输入 token，
先执行原生 dense Decode 作为对照；然后在远端完整 Prompt 工作集安装后，
通过 PVD CUDA sparse attention 后端执行**真实目标模型 Decode 前向**。
这条路径实际读取 D CUDA bank 的全部 28 层、4 KV heads，并把新生成的
K/V 保存在 D 模型池。两种注意力的 FP16 求和顺序不同，因此不要求
新生成 K/V 位级相等；实测最大 logits 绝对误差 **0.01611328125**、
新 K/V 最大绝对误差 **0.0234375**，贪心 top-1 token 相同，均小于脚本的
0.15 上限。之后的边界 4 稀疏刷新仍是模拟边界，尚未由真实 token
序列推进，也未在稀疏刷新后的 bank 上再执行模型前向。

With `--model-forward`, D first ran the real Qwen2.5-7B native dense Decode
as an oracle, then ran a real target-model Decode at position 1024 through
the PVD CUDA sparse-attention backend over the remotely installed complete
Prompt bank. All 28 layers and four KV heads were consumed and generated KV
remained in D's model pool. FP16 reductions need not be bit-identical across
these backends: the measured maximum absolute logit difference was
**0.01611328125**, maximum generated-KV difference **0.0234375**, and the
greedy top-1 token matched (the gate bounds each error by 0.15). The later
sparse refresh at boundary four remains synthetic and has **not** been
consumed by a subsequent target-model forward. No production Scheduler,
draft overlap, multi-request serving, throughput or broad quality claim follows.

Run the full-bank command below with `--model-forward` and a **fresh** retained
P Entry; a failed Delivery can close its V index even while the Entry allocation
remains, so do not reuse an unsearchable transfer ID. After success, both V
index snapshots had empty `entries`, D budgets drained, and the bounded V
service released ports and GPU memory.

## 2026-09-24 真实 Qwen 全层 D 工作集安装 / Real Qwen full-layer D bank installation

在 CloudLab 三节点隔离验证目录中，P GPU0 对现有 FP16
`Qwen2.5-7B-Instruct` 执行真实 1024-token Prefill，经单 rail `mlx5_0`
上传全部 Prompt K/V 到 V 的两个 GPU 分片。D GPU0 独立加载同一模型，
为全部 **28 层 × 4 KV heads = 112 组**生成目标模型 post-RoPE Q，
各组从其所属 V rank 的原生 CAGRA 索引检索。D 在边界 0 从两个 V rank
接收并安装完整 Prompt 工作集；在**模拟**的边界 4 再接收并安装每组
Top-10 的稀疏工作集。两轮均走 Mooncake/RDMA、远端完成及本地 CUDA
排序、PREPARED→PARKED→APPLIED→RESUMED 和每源 ACK；安装的所有
K/V 值与 D 独立目标模型前向产生的值逐组逐字节相同。接收、聚合、
工作集预算回零；V 两 rank 的 Entry/索引记录为空，只保留各自的
671088640 字节共享 CAGRA 根预留。验证程序输出 `passed`。

The isolated three-node CloudLab gate loaded the existing FP16
Qwen2.5-7B-Instruct checkpoint on P and D. P uploaded all 1024-token Prompt
KV to two V GPUs over single-rail `mlx5_0` RDMA. D generated real post-RoPE
target Q for all **28 layers × 4 KV heads = 112 groups** and searched each
group's native CAGRA index on its selected V shard. At boundary zero it
received and installed the complete Prompt bank; at a **synthetic** boundary
four it installed a Top-10 sparse bank for every group. Both rounds exercised
Mooncake/RDMA completion, CUDA ordering, the
PREPARED→PARKED→APPLIED→RESUMED install protocol and per-source ACK.
All installed K/V values matched an independent D target-model forward
bit-for-bit. D budgets were refunded and V Entry/index records were empty,
leaving only each rank's 671088640-byte shared CAGRA root reservation.

This is an offline protocol/byte-correctness gate. The boundary-four counter
does **not** represent four real generated tokens. It uses one representative
Q head per KV head, not the final GQA union of all seven Q heads. No Decode
attention consumed this remotely installed bank, and no draft/Scheduler
overlap, quality distribution, throughput or production path was validated.

```bash
# After run_pvd_qwen_native_upload_gpu.py retains the new Entry, on D/node-2:
python test/registered/disaggregation/run_pvd_qwen_native_bank_gpu.py \
  --decode-host 10.0.1.3 --coordinator-url http://10.0.1.2:19100 \
  --vector-base-url http://10.0.1.2 --shard-port-base 19200 \
  --transfer-id <P-output-transfer-id> \
  --layout-fingerprint <P-output-layout-fingerprint> \
  --rail mlx5_0 --expected-gpu V100S --architecture qwen2 \
  --model-path /proj/edgecut-PG0/models/Qwen2.5-7B-Instruct \
  --dtype float16 --context-length 1056 --max-total-tokens 4096
```

## 2026-09-24 真实 Qwen 三节点检索与稀疏回写 / Real Qwen three-node search and sparse WRITE

沿用真实 Qwen2.5-7B P→V Entry，node-2 D GPU0 加载同一 FP16 checkpoint，
对相同的 1024-token 输入独立执行目标模型 forward 并生成位置 1024 的
post-RoPE Q。第 0 层分别以 Q head 0→V0 KV head 0、Q head 14→V1 KV head 2
检索。两路原生 CAGRA Top-10 均与 D 本地精确点积 **10/10 重合**。
V0/V1 各将选中的 10 个 token 的 K/V 经单 rail `mlx5_0` Mooncake/RDMA
写入 D 的独立 CUDA 目的缓冲区，各 **5120 字节**；完成证明及本地可见性同步
之后，与 D 独立目标模型生成的对应 K/V **逐字节相同**。D 安全关闭目的 MR，
释放 Entry 后 V0/V1 的索引记录为空、只剩各一次 671088640 字节共享根预留；
V 测试服务退出并释放端口/GPU 显存。

The retained real Qwen2.5-7B P-to-V Entry was queried from node-2 D GPU0,
which loaded the same FP16 checkpoint and independently computed target
post-RoPE Q at position 1024. At layer 0, Q head 0 searched V0/KV head 0
and Q head 14 searched V1/KV head 2. Both native CAGRA Top-10 sets overlapped
D's exact local dot-product Top-10 by **10/10**. Each V rank RDMA-wrote the
selected K/V (10 tokens, **5120 bytes**) to a private D CUDA destination;
after exact remote completion/fence/extent proof and local CUDA ordering,
both payloads matched D's independent target-model K/V **bit-for-bit**.
Receive MRs were closed safely, both V indexes released to root-only budget,
and the bounded V service released its ports and GPU memory.

This checks two selected heads in one layer and one prompt. It does **not**
prove a general recall distribution, install a full 28-layer D working set,
run generated-token attention over the RDMA payload, or enable production
predictive Scheduler/pipeline overlap.

```bash
# After run_pvd_qwen_native_upload_gpu.py retained its Entry, on D/node-2:
python test/registered/disaggregation/run_pvd_qwen_native_search_receive_gpu.py \
  --decode-host 10.0.1.3 --coordinator-url http://10.0.1.2:19100 \
  --vector-base-url http://10.0.1.2 --shard-port-base 19200 \
  --transfer-id <P-output-transfer-id> \
  --layout-fingerprint <P-output-layout-fingerprint> \
  --rail mlx5_0 --expected-gpu V100S --architecture qwen2 \
  --model-path /proj/edgecut-PG0/models/Qwen2.5-7B-Instruct \
  --dtype float16 --context-length 1056 --max-total-tokens 4096
```

## 2026-09-24 真实 Qwen Prompt KV 的 P→V 上传 / Real Qwen P-to-V upload

CloudLab node-0 的 V100S GPU0 加载现有 FP16 Qwen2.5-7B-Instruct 权重，
以确定性 1024-token 输入实际执行 Prefill，提取全部 28 层、4 个 KV heads
的 Prompt K/V。按 V 的 TP2 存储布局拆成两个 shard，单 rail `mlx5_0`
通过 Mooncake/RDMA 送往 node-1 的 V GPU0/GPU1。V 每 rank commit 完整
28-layer/2-head 分片，建成 **56 个**可搜索索引；每页 114688 字节。
`run_pvd_qwen_native_upload_gpu.py` 输出 `passed`，Entry 暂留供 D 的下一步
验收。测试用的 first-token 元数据只是协议占位值，不声称 P 已按该输入
正式采样出那个 token。

Node-0 V100S GPU0 loaded the existing FP16 Qwen2.5-7B-Instruct checkpoint
and actually prefetched deterministic 1024-token input. All 28 layers and
four KV heads were split across two V storage ranks and uploaded over native
Mooncake/RDMA `mlx5_0` to node-1 GPUs 0/1. Each rank committed its full
28-layer/two-head shard and built **56** searchable indexes (114688 bytes per
page). The gate passed and retained the Entry for D-side validation. The
first-token metadata in this gate is a protocol placeholder, not a sampled
output claim.

This proves real-checkpoint KV upload/index creation, not real Q retrieval,
D installation, generated-token attention, or production scheduling.

```bash
# Start V cagra-auto with --page-bytes 114688, --total-pages 512,
# --prompt-index-vector-space qwen2.5-7b-real-target and a shared native cap.
# On P/node-0, with pinned Mooncake and PYTHONPATH=python:
python test/registered/disaggregation/run_pvd_qwen_native_upload_gpu.py \
  --prefill-host 10.0.1.1 --coordinator-url http://10.0.1.2:19100 \
  --vector-base-url http://10.0.1.2 --shard-port-base 19200 \
  --rail mlx5_0 --page-bytes 114688 --expected-gpu V100S \
  --architecture qwen2 \
  --model-path /proj/edgecut-PG0/models/Qwen2.5-7B-Instruct \
  --dtype float16 --context-length 1056 --max-total-tokens 4096
```

## 2026-09-24 真实 Qwen2.5-7B K/Q 原生 CAGRA / Real Qwen2.5-7B K/Q CAGRA

CloudLab node-1 的 V100S GPU0 在隔离的 cuVS 25.02 环境加载现有 FP16
`Qwen2.5-7B-Instruct` checkpoint。目标模型实际 forward 生成 1024 个 Prompt
token 的第 0 层、第 0 KV head 的 post-RoPE K；同一目标模型的独立 probe 在
位置 1024 捕获第 0 Q head 的 post-RoPE Q。原生 CAGRA 对这些 K 建图并搜索，
与精确 GPU 点积 Top-10 对照：本次 **10/10 命中、recall@10=1.0**，返回分数
最大绝对误差 **0.000244140625**，probe 预算已退还。脚本状态 `passed`。

On node-1 V100S GPU0, an isolated cuVS 25.02 environment loaded the existing
FP16 Qwen2.5-7B-Instruct checkpoint. A real target forward produced 1024
post-RoPE Prompt K rows for layer 0/KV head 0. A separate target probe captured
the matching post-RoPE Q at position 1024, query head 0. Native CAGRA's
Top-10 overlapped the exact GPU dot-product Top-10 by **10/10** in this one
query; maximum returned-score error was **0.000244140625**. Probe budget was
refunded and the gate passed.

This is one query and one layer/head, not a quality distribution or a recall
guarantee. The Prompt tokens are deterministic synthetic IDs processed by real
weights. The gate runs on one node: it does not send this model KV over RDMA,
install it on D, run generated-token attention, or measure pipeline latency.

```bash
# On the isolated V cuVS-25.02 candidate, with PYTHONPATH=python and the
# existing local model checkpoint available; this script imports cuVS first.
python test/registered/disaggregation/run_pvd_qwen_cagra_recall_gpu.py \
  --architecture qwen2 \
  --model-path /proj/edgecut-PG0/models/Qwen2.5-7B-Instruct \
  --dtype float16 --context-length 1056 --max-total-tokens 4096
```

## 2026-09-24 三节点 D 工作集安装与 ACK / Three-node D bank install and ACK

在上一项跨节点稀疏交付基础上，新增
`run_pvd_native_sparse_install_gpu.py`。CloudLab node-2 的 D GPU0
以 TP1 计算布局将 V 两个存储 rank 的 K/V 汇入一个 CUDA Prompt 工作集：
第一次从 V 拉完整 1024-token Prompt KV，安装于边界 0；第二次按两 V
rank 各自的原生 CAGRA 逻辑选择拉稀疏 KV，安装于边界 4。两轮均完成
PREPARED→PARKED→APPLIED→RESUMED 协议、V Delivery ACK，并对工作集
中的四个 layer/head 组逐字节核对。脚本输出
`installed_boundaries=[0,4]`、`released_entry=true`、`passed`；D 的接收、
聚合、工作集预算均归零，V 两 rank 的 Entry/索引清空、只留根预算。

The new `run_pvd_native_sparse_install_gpu.py` gate combines both V storage
ranks into one TP1 D CUDA Prompt bank. It installs the complete 1024-token
Prompt KV at boundary 0, then installs the sparse CAGRA-selected refresh at
boundary 4. Both rounds passed PREPARED→PARKED→APPLIED→RESUMED, ACKed the V
Deliveries, and produced bit-exact values for all four layer/head groups.
The D receive, aggregation and bank budgets refunded to zero; both V Entry
indexes were released. Final report: `installed_boundaries=[0,4]`, `passed`.

This is still a synthetic protocol/installation gate. No target-model forward
produced the intervening four D tokens; no actual target Q, attention output,
recall or network/compute pipeline latency was measured. It does not activate
production predictive retrieval in SGLang's Scheduler.

```bash
# After the same P --retain-entry upload below, run on D/node-2:
python test/registered/disaggregation/run_pvd_native_sparse_install_gpu.py \
  --decode-host 10.0.1.3 --coordinator-url http://10.0.1.2:19100 \
  --vector-base-url http://10.0.1.2 --shard-port-base 19200 \
  --transfer-id <P-output-transfer-id> \
  --layout-fingerprint <P-output-layout-fingerprint> \
  --rail mlx5_0 --expected-gpu V100S
```

## 2026-09-24 三节点原生稀疏交付 / Three-node native sparse Delivery

CloudLab node-0 的 P GPU0 通过 `--retain-entry` 上传完整合成 Prompt KV 到
node-1 的 V GPU0/GPU1；V 各建两个 CAGRA 图。node-2 的 D GPU0 取 V 的逻辑
检索结果和版本，分别给两个 V rank 注册私有 CUDA 接收缓冲区，走 Mooncake
`mlx5_0` RDMA WRITE。D 仅在 V 返回精确的 terminal-success、写入 fence 和
字节数证明后执行 CUDA 接收同步，再逐字节核对每个 rank 的两个 layer/KV-head
K/V 组。两 rank 各收到 512 字节；关闭 Delivery 和 MR 后，D 接收预算归零；
释放 Entry 后，V 两 rank 索引均无 Entry，且各只保留 671088640 字节根预算。
`run_pvd_native_sparse_receive_gpu.py` 最终输出 `passed`。首次试运行的 V
分量不一致是验收脚本随机数重建顺序错误，修正后复跑通过。

P GPU0 on node-0 uploaded synthetic full Prompt KV with `--retain-entry` to
the two V100S ranks on node-1. D GPU0 on node-2 took the V search results and
versions, registered a private CUDA destination per V rank, and received sparse
K/V by Mooncake RDMA WRITE over `mlx5_0`. D read nothing until exact remote
terminal-success, fence and byte-count proof plus local CUDA receive ordering.
Both layers' K/V were bit-exact for each rank (512 bytes/rank). Closing the
Deliveries refunded D's receive budget; releasing the Entry cleared both V
indexes, leaving only their 671088640-byte root reservations. The first run
exposed an incorrect synthetic RNG reconstruction order in the gate itself;
the corrected run passed.

The D gate only inspects a private receive buffer. It does **not** install a
Decode working set, ACK a completed install, run model attention, use real
post-RoPE target Q, measure CAGRA recall, or validate latency hiding. The
successful unacknowledged Deliveries are closed with an explicit V fence.

```bash
# Start isolated V with --experimental-cuda-sparse-packing, cagra-auto,
# shared native cap, Mooncake and private ports 19100/19200/19201.
# On P/node-0, record entry_key.transfer_id and layout_fingerprint:
python test/registered/disaggregation/run_pvd_native_upload_index_gpu.py \
  --prefill-host 10.0.1.1 --coordinator-url http://10.0.1.2:19100 \
  --vector-base-url http://10.0.1.2 --shard-port-base 19200 \
  --rail mlx5_0 --page-bytes 1024 --expected-gpu V100S --retain-entry
# On D/node-2, use those exact two values:
python test/registered/disaggregation/run_pvd_native_sparse_receive_gpu.py \
  --decode-host 10.0.1.3 --coordinator-url http://10.0.1.2:19100 \
  --vector-base-url http://10.0.1.2 --shard-port-base 19200 \
  --transfer-id <P-output-transfer-id> \
  --layout-fingerprint <P-output-layout-fingerprint> \
  --rail mlx5_0 --expected-gpu V100S
```

## 2026-09-24 跨节点 P→V 原生上传与检索 / Native cross-node P-to-V gate

CloudLab node-0（P GPU0）向 node-1（V GPU0/GPU1）用 Mooncake/RDMA
`mlx5_0` 发送合成 1024-token、2-layer、2-KV-head 的完整 Prompt KV。V 每 rank
接收一个 head，commit 后各建两个原生 CAGRA 图；HTTP 搜索各查询四个合成 K 行，
两 rank 均 4/4 self-hit。释放 Entry 后两侧索引记录均清空，预算只剩各自
671088640-byte 共享根预留，未隔离。最终脚本输出 `passed`、`committed_shards=2`、
`indexed_heads_per_rank=[2,2]`、`root_only_after_release=true`。

这次实测首先发现 V HTTP query 在 CPU、CAGRA 索引在 CUDA，原本会以
`CAGRA requires contiguous float32 matrices on its declared device` 拒绝。
修复后，管理器先验证请求身份并预留搜索预算，再将 query 放到后端设备，
完成检索后同步并退还临时预算。node-2 定向回归 **226 passed / 3 skipped**。
测试用 K 行充当 query，仅验证跨节点数据/索引/搜索链路，**不证明真实目标
模型 Q 的 recall、稀疏 KV 交付给 D 或生产预测 Scheduler**。

Node-0 P GPU0 sent synthetic 1024-token, two-layer/two-KV-head complete
Prompt KV over Mooncake/RDMA `mlx5_0` to node-1 V GPUs 0 and 1. Each V rank
committed one head and built two native CAGRA graphs. Four synthetic K-row
queries per rank achieved 4/4 self-hits. Releasing the Entry cleared both
indexes and refunded all per-Entry bytes, leaving only each rank's
671088640-byte shared-root reservation. The final gate reported `passed`.

The first real HTTP search exposed a device-placement bug: HTTP built CPU Q
while native CAGRA required CUDA Q. The manager now checks query identity,
reserves its search footprint, moves Q to the backend device, searches,
synchronizes and refunds. Focused node-2 regressions: **226 passed / 3 skipped**.
Synthetic stored K rows are only plumbing queries, not target-model Q or
recall evidence. This gate does not exercise D delivery or predictive serving.

```bash
# With the isolated V service listening privately at 10.0.1.2:19100/19200/19201,
# --page-bytes 1024 and cagra-auto/shared-native budget configured:
PYTHONPATH=<pinned-mooncake-target>:python python \
  test/registered/disaggregation/run_pvd_native_upload_index_gpu.py \
  --prefill-host 10.0.1.1 --coordinator-url http://10.0.1.2:19100 \
  --vector-base-url http://10.0.1.2 --shard-port-base 19200 \
  --rail mlx5_0 --page-bytes 1024 --expected-gpu V100S
```

## 2026-09-24 原生 V 双 rank 服务门控 / Native dual-rank V service gate

新增 `run_pvd_cagra_v_service_gpu.py`：在指定的空闲隔离端口上使用
`pvd_cagra_server` 启动单进程双 V rank，等待 coordinator 健康，检查 V0/V1
Mooncake 0.3.13.post1/RDMA、单 rail `mlx5_0`、GPU MR 注册与本地传输预检、
两份 `cagra_auto` 索引状态和各一次 640 MiB 根预算预留，然后向子进程发送
TERM 并等待正常退出。CloudLab node-1 两张 V100S 上输出 `passed`，服务退出码
0，测试端口释放。没有创建 Entry，也没有检索真实模型 Q 或运行 D。

The self-cleaning `run_pvd_cagra_v_service_gpu.py` gate launches the
one-process/two-rank V group on explicit unused ports, checks healthy
coordinator/V0/V1, Mooncake 0.3.13.post1 over active single-rail `mlx5_0`,
GPU registration/local transfer preflight, and one 640 MiB native-root
reservation per `cagra_auto` rank. It then terminates its own process and
waits for clean exit. The node-1 dual-V100S run passed with exit code 0 and
released its test ports. No Entry, real-model Q search, or D was involved.

```bash
PYTHONPATH=<pinned-mooncake-target>:python python \
  test/registered/disaggregation/run_pvd_cagra_v_service_gpu.py \
  --advertise-host 10.0.1.2 --rail mlx5_0 \
  --coordinator-port 19100 --shard-port-base 19200
```

## 2026-09-24 管理器与真实原生图联合门控 / Manager/native integration gate

node-1 的 GPU 0 和 GPU 1 都在 V100S/cuVS 25.02 隔离候选环境运行
`run_pvd_cagra_shared_manager_gpu.py`：先通过真实 V CLI 解析、校验和
`_build_prompt_index` 工厂构建管理器，然后用 Prompt KV packer 将合成
1024-token、2-layer、rank-local 1-KV-head 数据打包；`PromptIndexManager`
提取向量，为两个 Entry 共构建四个原生 CAGRA 图，并通过管理器完成一次检索。
共享 RMM 根上限 671088640 bytes、每图子上限 536870912 bytes。管理器启动时
已计费 671088640；四图存活时原生根占用 524288，预算占用 671612928
（根上限加 524288 向量副本）；关闭两个 Entry 后原生根占用归零，预算仍保留
671088640 的根预留。状态 `passed`。这证明了合成数据组件接线，不代表真实 V
worker、真实目标 Q、56 图容量、并发服务、召回或吞吐验收。

The isolated node-1 V100S/cuVS 25.02 component gate passed separately on
GPU 0 and GPU 1. It used the real V CLI parser, argument validator and
`_build_prompt_index` factory, then packed synthetic
1024-token Prompt KV, extracted rank-local K through `PromptIndexManager`,
built four native CAGRA graphs across two Entries, and searched through the
manager. The root cap was 671088640 bytes and each child cap 536870912.
Initial charge was exactly the root cap; four live graphs used 524288 native
root bytes, and budget charge was root plus 524288 vector-copy bytes. Closing
both Entries returned native root usage to zero while retaining the root
budget charge. Status: `passed`. No real V service, real-model Q, 56-graph
capacity, concurrent serving, recall or throughput claim follows.

```bash
PYTHONPATH=python python test/registered/disaggregation/run_pvd_cagra_shared_manager_gpu.py \
  --device 0 --expected-gpu V100S --expect-cuvs-version 25.02.00
PYTHONPATH=python python test/registered/disaggregation/run_pvd_cagra_shared_manager_gpu.py \
  --device 1 --expected-gpu V100S --expect-cuvs-version 25.02.00
```

## 2026-09-24 共享原生上限能力探针 / Shared native-cap capability gate

`CagraNativeRuntime` 新增可选父级 RMM limiter：每个图仍有
自己的 per-index limiter，但其上游可指向同一个全局 limiter。node-1 V100S、
cuVS 25.02 的隔离探针先同时构建并检索两个 1024×32 原生索引；两个子 limiter
各保留 131072 bytes，父 limiter 精确记录总计 262144 bytes。随后从两个
不同 cuVS Resources 各申请 360 MiB：第一笔通过，第二笔在共享 640 MiB
上限处被拒，尽管它未超过每图 512 MiB 上限。第一笔安全释放、两个索引
逆序销毁后父级计数回到零；门控输出 `passed`。

这只证明 cuVS C API 与嵌套 RMM resource 在这套实测环境中共享计数和限额。
随后增加的 `--prompt-index-cagra-global-native-bytes N` 已把父级 cap 纳入
`PromptIndexManager` 的一次性预算预留：`N` 至少覆盖一个图 cap，不得超过总索引
预算；每图 child limiter 保留，向量和其他内存另计。该接线的 node-2 焦点回归
**219 passed / 3 skipped**，但尚未在真实 V 服务中验证，亦不能据此断言
56 图或多个 Entry 的容量。

`CagraNativeRuntime` now has an optional
parent RMM limiter shared by the per-index child limiters. The isolated
node-1 V100S/cuVS 25.02 gate built and searched two 1024×32 indexes. The
root recorded their combined 262144 retained bytes. A 360 MiB cuVS C-API
allocation from one index succeeded; a second 360 MiB request from the other
was rejected by the shared 640 MiB root even though each child allowed up to
512 MiB. After safe release and reverse disposal, root allocation returned
to zero. The later `--prompt-index-cagra-global-native-bytes N` integration
reserves the parent cap once in `PromptIndexManager`, provided it covers a
child cap and fits the total index budget. Focused node-2 regressions passed
**219 / 3 skipped**. The combined serving path and safe 56-graph admission
remain unverified.

```bash
PYTHONPATH=python python test/registered/disaggregation/run_pvd_cagra_shared_cap_gpu.py \
  --expected-gpu V100S --expect-cuvs-version 25.02.00
```

## 2026-09-24 短 Prompt 自动回退 / Explicit short-Prompt fallback

新增显式 `--prompt-index-backend cagra-auto`，纯 `cagra` 模式保持原行为。
短 Prompt（行数不超过 `intermediate_degree`）采用同一 V GPU 上的有界精确
后端；长 Prompt 仍用原生 cuVS CAGRA。索引管理器按实际选择的后端分别预留
保留内存与 build/search scratch，native UNKNOWN 会阻止整个组合后端继续运行。
node-0 V100S 上索引/管理器相关回归 **213 passed / 3 skipped**，其中 GPU
精确回退真的在 CUDA tensor 上运行；跳过项不是此回退用例。

node-1 隔离 cuVS 25.02 候选环境的原生门控同时构建 16×32 精确 GPU 索引与
1024×32 CAGRA 索引，各检索 4 条 query、Top-4，短/长自向量命中均 4/4，
最大分数误差分别为 0 与约 3.81e-6；两个索引均正常销毁。短索引保留
2048 bytes，长索引构建后保留 131072 bytes；长索引仍需 512 MiB 原生
构建上限，不能把保留量当作峰值。没有真实模型 Q、56 图、多个 Entry、
生产 Scheduler 或端到端性能证据。

The explicit `cagra-auto` mode uses bounded exact search on the same V GPU
for rows at or below `intermediate_degree`, and native cuVS CAGRA above it;
pure `cagra` is unchanged. The manager charges each actual path separately,
and native UNKNOWN poisons the combined mode. Node-0 V100S index regressions
passed **213 / 3 skipped**, including a real CUDA exact-fallback case.
In the isolated node-1 cuVS 25.02 candidate, a 16×32 exact GPU index and a
1024×32 native index coexisted, each searched four Top-4 queries with 4/4
self-hits, then both disposed. Score errors were 0 and about 3.81e-6.
Retained bytes after build (2048 and 131072) are **not** peak-build bounds;
the native test still used a 512 MiB cap. Real-model Q, 56 graphs,
multi-Entry admission, production serving and latency remain unvalidated.

在隔离候选环境从仓库根目录复验 / Reproduce from the repository root in the
isolated candidate environment:

```bash
PYTHONPATH=python python test/registered/disaggregation/run_pvd_cagra_auto_gpu.py \
  --device 0 --expected-gpu V100S --expect-cuvs-version 25.02.00 \
  --native-cap-bytes 536870912
```

## 2026-09-23 V100S 候选环境实测 / Candidate V100S execution

在 node-1 的独立 `pvd-cagra25-venv` 中安装了 `cuvs-cu12==25.2.0`
（实际导入版本字符串 `25.02.00`）、`libcuvs-cu12==25.2.1`、
`rmm-cu12==25.2.0` 和 `cupy-cuda12x==13.3.0`。原 Conda 环境和运行中的
V worker 未修改。仓库 `check_cagra.py` 在 V100S/SM70、CUDA runtime 12.6
上完成真实 4096×128 index build 和 32-query Top-10 search；合成 recall@10
为 0.953125，分数最大绝对误差约 6.4e-6。

新增的 `run_pvd_cagra_backend_gpu.py` 还实际执行了**本项目**的
`CagraIndexBackend`：原生 RMM 限额/共享注册表探针、IVF-PQ build、16-query
search、分数核对和 dispose 全部成功。512 MiB 每索引原生上限的本次合成
配置通过，64/128/256 MiB 上限均在真实 RMM 限额处拒绝构建；这些数值不是
对不同数据规模的通用上限。仅加载模块或官方平台兼容表均不能替代这些实测。
同一 backend 上再运行 `--index-count 2`，两个索引同时存活，各自检索 16 个
query，32/32 个自向量命中，逆序释放 2/2 个索引；这仍不是完整 56 图服务验收。
两个索引构建后各只保留 524288 bytes 的 RMM 分配，远低于本配置构建阶段
必须允许的 >256 MiB 峰值。这说明当前“每图终生保留完整 512 MiB cap”是
保守而昂贵的**预算策略**，不是该合成图的真实静态占用；不可据此直接把 cap
降到 512 KiB，因为构建会失败。

The isolated cuVS 25.02 candidate on node-1 V100S completed a real 4096×128
CAGRA build and 32-query Top-10 search (synthetic recall@10 0.953125). PVD's
own `CagraIndexBackend` also passed its native RMM bridge/cap probe, IVF-PQ
build, 16-query search, score check and dispose with a 512 MiB per-index cap.
The same synthetic build was refused at 64, 128 and 256 MiB by the native
limiter. This is an artifact-specific execution result, not a general memory
bound, model-query recall, production serving or performance acceptance.
With `--index-count 2`, both native indexes coexisted, searched independently
and disposed in reverse order (32/32 self-neighbor hits, 2/2 disposals).
Each retained 524,288 RMM bytes after build. The peak allowance required by
build is much larger than the retained graph, so a future shared transient
budget could improve capacity only after preserving concurrent build/search
admission and native-completion safety.

当前实现为每个 `(layer, KV head)` 索引在**整个生命周期**保留完整 native cap。
Qwen2.5-7B TP2 的每个 V rank 有 28×2=56 个此类索引；若都用 512 MiB，
保留预算就是 28 GiB/rank，尚未计入 KV pool、向量副本和其他显存。
因此不能因为单图通过就直接在 32 GiB V100S 上打开全部生产索引；需要
新的跨索引共享/峰值预算设计，或实测更小的安全 cap，并通过真实 Prompt/Q
与并发 Entry 验收。The current whole-lifetime per-index cap would reserve
28 GiB per rank for 56 Qwen2.5-7B TP2 graphs at 512 MiB each, before the
KV pool and other users. Do not enable all graphs from this synthetic result.

## Compatibility is artifact-specific / 兼容性不能凭标签判断

The current [cuVS installation guide](https://docs.nvidia.com/cuvs/installation)
lists Ampere or newer for current source builds. The broader
[RAPIDS platform table](https://docs.nvidia.com/datascience/platform-support/)
still lists Volta for CUDA 12 combinations. These different scopes are not
proof that any particular cuVS wheel or source revision runs on V100S. Record
the actual artifact/revision, CUDA and GPU, then execute build/search on V100S.
Do not silently upgrade the environment, switch GPUs or replace CAGRA.

官方通用平台表与 cuVS 自身安装页的范围不同。不能从 RAPIDS/Volta 标签、导入成功
或版本字符串推导具体 cuVS 包适配 V100S；也不能据此断言全部历史版本都不支持。
具体软件组合必须实测。默认 inventory 仅收集环境；没有 GPU 时仍可推进通用代码。

## Score verification / 分数数值验收

The [cuVS 25.06 CAGRA Python implementation](https://github.com/rapidsai/cuvs/blob/branch-25.06/python/cuvs/cuvs/neighbors/cagra/cagra.pyx)
defines inner-product and squared-Euclidean metrics and returns scores together
with neighbor IDs. The PVD probe now compares each returned score against its
actual selected row under the requested metric, using an independent float64
CPU calculation. Higher dot products and lower squared distances must not be
confused. This inspection is an API reference, not a required-version choice.

Previously finite but incorrect scores could pass if neighbor recall was good.
Now wrong signs, wrong metrics, ID/score mispairing, duplicate/out-of-range IDs,
nonfinite scores and low recall fail. Exact kth-score ties are interchangeable
for recall, without widening the cutoff using numerical tolerance.

原先只检查有限分数与邻居召回，会漏掉正确 ID 搭配错误分数的问题。现在增加独立
CPU 数值验证；分数容差为 rtol/atol 各 1e-3，仅用于数值检查，不扩大 recall 的
Top-K cutoff。新增 10 个 CPU oracle 测试通过；后续独立的真实 V100S 探针见本页顶部。

```bash
# No GPU import/build/search, no dependency changes
python scripts/pvd/check_cagra.py --mode inventory
# Only on the candidate experimental environment
python scripts/pvd/check_cagra.py --mode smoke --expected-gpu V100S
```

The GPU smoke is synthetic, not real-query recall or generation quality. No
CAGRA serving backend has been enabled. The bounded native adapter probe
establishes one version/device/configuration, not safe multi-Entry admission,
all 56 graphs, real-model queries, RDMA integration or performance. CPU doubles
remain useful for lifecycle edge cases but cannot establish those GPU properties.
