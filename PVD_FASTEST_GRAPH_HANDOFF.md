# PVD 最快建图路径：跨 session 交接

整理日期：2026-10-02。最新实测：2026-10-01。仓库：`D:\code\sglang-V100-PVD`。

## 1. 接手时先记住这些

- 当前目标是缩短 **P→V→D 路径中，最终 KV 提交后到 V 双 rank 图 READY 的等待**。用户明确不考虑 P→D 初始 KV 直送；D 等双 rank READY，再从 V 拉取初始 KV。
- 用户选定精确 degree 16、四合一。每 rank 有 56 个 local KV heads，合为 14 张图；V 的两张 GPU 并发处理两个 rank。
- 当前最快且有在线 READY 收益证据的组合：**Prefill chunk 256 + 首图 2048 + 最后 111 tokens 更新 + 批量准备/提取/中心化 + 预分配/固定视图/提前捕获 + 相似度复用 + 最后 chunk 提前触发**。
- 在 2159-token Prompt 上，该组合两轮正式测量的“最终内部提交→双 READY”中位数分别为 **5.607 ms、5.086 ms**。可以概括为约 **5.1–5.6 ms**，不是任意请求的上界，也不是完整建图或完整 D 等待时间。
- 当前快路径是 **自定义精确 KV 邻接维护**，已经不走原生 `cagra.extend`。原生 cuVS CAGRA 仍持有索引并执行过滤检索。日志或函数中的 extend 不应一律解释为原生插入。
- 新 Top-16 有局部 GPU 收益，stream event 缩小了同步范围，但两者尚无普通在线 READY 收益；推荐组合中都关闭。
- 上述实验选项保持默认关闭；完整 Prompt 上传仍为默认路径。没有证明任意 Prompt 的最优切点、TP2 或多请求负载下的收益。

## 2. 实际数据流和图生命周期

```text
请求开始：P 已知 Prompt 长度，选首段切点
  → P chunked prefill，算出前 2048 tokens 的 KV
  → P 上传首段到 V 两个 rank
  → 每 rank 验证原生 PUT 成功终态、准确字节数与授权关闭
  → V 提取首段 K、固定每 head 的中心化均值、建 provisional 图
    同时：P 继续 prefill 最后 111 tokens，并上传剩余 KV
  → 最后 chunk 通过完成证明，立即触发该 rank 的尾段更新
    同时：完整 Entry 的 aggregate commit / STORED 继续执行
  → 尾段 GPU 更新完成，完整行数、ID mapping 和 STORED 均满足
  → 各 rank 原子发布 READY
  → D 经图门确认双 rank READY，从 V 拉取并安装初始 KV，然后 Decode
```

**Prefill chunk 256 是 P 的计算粒度；本轮 KV 上传只有两份：2048 和 111。**
不是每计算 256 tokens 都向 V 插入一次。两 rank 各一次首图、一次最终更新。
如果首图未完成而后续 KV 已到达，manager 可以合并待处理范围；其他实验的多次插入回放不能当作这轮在线到达序列。

最后 chunk 可以在 STORED 前更新 provisional 图，但在完整 STORED 前不能搜索、不能发布 READY。
完成未知时必须保留 batch、native owner、GPU 张量、allocation pin 和预算；不能因为等待超时就释放或发布。
packed KV 是 component-major；页段在每个 K/V layer component 中各有独立目标区间，不能按整块 buffer 尾部 append。

## 3. 为什么这条路径快

### 基础图算法

1. 利用 Prompt K 到达后不可变的特点，首图固定每 head 均值，后续用同一个均值中心化。不要在每次插入时重新计算均值并移动旧点。
2. 按最终容量预留 K、图边、邻居缓存和工作区，保留稳定指针；56 个 heads 批量做 FP32 cuBLAS 点积和邻居计算。
3. 缓存旧节点的精确 Top-16 分数和 ID。更新只计算旧×新和新×全部可读节点，合并旧缓存；不重新计算旧×旧。
4. 维护完整精确 Top-16 缓存，但供 CAGRA 搜索的 degree-16 图为 **14 条近邻边 + 2 条同 head token-ring routing edge**。纯 16 条近邻边此前有召回损失；不要把 ring2 去掉后仍称同一配置。
5. 每图只做一次 native import，后续维护同一个 owner 和 ID。四合一保留每 head 的映射及过滤搜索，`itopk_size=2048`。

### 压缩尾段固定开销

| 优化 | 做法 |
|---|---|
| small-tail | 阈值 512；扩大旧节点行批次，融合旧邻居合并、缓存写回及 ID/ring/图边写回 |
| fused prepare | 将 14 组有限值检查合为一次，连续相邻 storage 的 K/ID 准备合成批量 kernel |
| batched K extraction | 从 component-major packed K 批量提取所有 head，转为连续 FP32；分组直接用相邻视图 |
| fused K centering | 尾段提取与固定均值中心化融合，复用首图验证过的分组计划 |
| planned tail | 首图阶段预分配尾段、创建描述/最终映射，固定 native views 到最终容量，提前捕获尾段 CUDA Graph |
| reuse scores | ≤128 行尾段只做一次 new×all，将 new×old 转置用于旧行合并，避免第二次 GEMM |
| early final update | 最后 chunk 一经终态与准确字节证明就启动更新，和 aggregate commit/STORED 重叠 |

提前捕获绑定当前 Entry 自有缓冲区；不使用尚未证明完成的 KV。录制不执行更新；实际到达范围必须匹配才重放，否则正确回退。
预分配、提前捕获的成本仍计入首图，只是本轮被后续 Prefill/传输覆盖。

## 4. 正式时间结果和计时口径

两轮都为相同四个 2159-token Prompt、temperature 0、输出 6 tokens；每配置 8 次正式请求。
ABBA 正反顺序，每 arm 重启 P/V/Gateway，D 共用；每 arm 相同三次暖机，暖机独立保存。

| 正式对照 | 基线 | 候选 | 结论 |
|---|---:|---:|---|
| planned-tail + reuse-scores，最终内部提交→双 READY | 7.890 ms | 5.933 ms | 组合有 READY 收益 |
| 再测 early-final-update，同一指标 | 7.671 ms | **5.607 ms** | 提前触发缩短 2.064 ms，26.9% |
| 同快配置再测 stream-completion，同一指标 | **5.086 ms** | 5.281 ms | 没有加速；保留整卡 fence |

各行是各自同轮配对结果，不能跨轮相减或叠加收益。5.086 ms 来自同步范围实验的 device 基线，其配置包含 early-final-update。

最新 device 基线的其他指标：

| 指标，中位数 | 时间 | 包含什么 |
|---|---:|---|
| 首图完整处理 | 32.178 ms | K 提取、准备、构图、native import、提前捕获和完成同步 |
| 首图完成→最终 HTTP commit 余量 | 58.000 ms | 首图已被覆盖；最小余量 55 ms |
| GPU 图计算 event 区间 | 2.332 ms | 相似度、邻居维护、写边；可能含提交空隙 |
| GPU 输入/图更新 event 区间 | 3.077 ms | 额外包含后端输入准备；不是额外再加 2.332 ms |
| 完成 fence 的 host 等待 | 2.130 ms | 等待尚未完成 GPU 工作；不是纯同步 API 固有开销 |
| 完整尾段，发布前 | 5.333 ms | 提取、准备、GPU 等待和 manager 簿记 |
| 最终内部成功 chunk commit→双 rank READY | **5.086 ms** | 两 rank 的较晚 final commit 到较晚 READY |
| 请求开始→双 READY | 950.306 ms | 包含 Prefill、上传和图工作 |
| D 图门调用等待 | 1.574 ms | 从 D 发起图门调用计时，起点更晚 |
| D 从 V 安装初始 KV | 99.904 ms | fan-in；并非上述图等待 |
| 客户端首个流事件 / 完成 | 1117.500 / 2743.000 ms | 本轮真实服务指标 |

**最终内部 commit 是可读性推进的软件锚点，不是 NIC 收到最后一个 byte 的硬件时间。**
旧 HTTP 200 指标分辨率为 1 ms、起点更晚；不能与 monotonic ns 指标混用。
提前触发允许一 rank 在另一 rank final commit 前开始，所以完整尾段可能长于全局 final commit→双 READY。
各行独立取中位数，包含关系和跨 rank 起点不同，不能相加构造一次请求账单。

### 之前 5.607 ms 的可加分解

这是 early-final-update 轮排序后中间两个请求（5.487254、5.726634 ms）的分解均值，**不是每列独立取中位数**。
每个请求选最后 READY 的 rank，从全局较晚 final commit 开始：

| 连续区间 | 时间 |
|---|---:|
| 最终 commit→该 rank 尾段 manager 开始 | 0.537 ms |
| 尾段内 GPU 输入/图更新 event 区间 | 3.356 ms |
| 尾段 wall time 中 event 区间之外 | 1.343 ms |
| 尾段结束→READY | 0.371 ms |
| 合计 | **5.607 ms** |

第三行仍含 host 提交/等待/簿记，不等于纯 CPU 执行时间。
STORED 约在锚点后 1.968 ms 完成，与尾段重叠，不能另加到合计。
这张表说明历史 5.607 ms，不能当成最新 5.086 ms 的逐阶段账单。

## 5. 推荐实验配置

在已验证的 CloudLab 环境使用 `test/registered/disaggregation/cloudlab_pvd_new_lease.sh`，相应环境变量如下。
这是复现已测快配置，不是改变生产默认值。

```bash
# P/V/D 的共同上传与交付选择
export PVD_CAGRA_EXTEND25=1 PVD_CHUNKED_CAGRA_UPLOAD=1
export PVD_CAGRA_GROUP_HEADS=4 PVD_CAGRA_EXACT_HEAD_SEED=1
export PVD_CAGRA_ITOPK_SIZE=2048
export PVD_GATE_INITIAL_FANIN_ON_INDEX=1 PVD_DIRECT_PD_BOOTSTRAP=0
export PVD_PREFILL_CHUNK_TOKENS=256 PVD_MODE=predictive

# V 的快图后端
export PVD_CAGRA_KV_EDGE_UPDATE=1 PVD_CAGRA_KV_ROUTING_EDGES=2
export PVD_CAGRA_SMALL_TAIL_MAX_ROWS=512 PVD_CAGRA_FUSED_PREPARE=1
export PVD_BATCHED_K_EXTRACTION=1 PVD_FUSED_K_CENTERING=1
export PVD_PLANNED_TAIL=1 PVD_REUSE_SCORES=1 PVD_EARLY_FINAL_UPDATE=1
export PVD_CAGRA_EXTEND_CONCURRENCY=1 PVD_CAGRA_NOGIL_EXTEND=1

# 本轮未证明在线收益的候选关闭
export PVD_NEW_TOP16=0 PVD_FUSED_EDGE_WRITE=0 PVD_STREAM_COMPLETION=0

# 公平复测时与基线相同的日志/计时设置
export PVD_PROFILE_GPU=1 PVD_PROFILE_CHUNK_STAGES=1
```

P 指定 `PVD_SPLIT_POLICY_FILE` 为下面的固定策略文件：

```json
{"schema":"pvd-exact16-split-policy-v1","choices":{"2159":{"prefix":2048}}}
```

这是隔离图优化的固定实验策略，**不是预测器已经自动选出 2048**。
未列入的长度走完整图回退；不要把该文件直接理解为任意长度的最优预测器。
原参数化 profile 仍按旧成本/20 ms 重叠 margin 对 2159 选择 1792；后续需要按当前完整 manager 成本重新校准。

预测目标应包括到达时间、首图是否覆盖和尾段：

```text
full_READY  = max_rank(arrival(N) + build(N))
split_READY = max_rank(max(arrival(p) + build(p) + margin, arrival(N))
                      + tail_stage(N, p))
```

候选 p 受 scheduler 可达边界和页对齐约束。首图覆盖后应尝试后移 p，但还要满足波动余量，并比较请求开始→READY。
旧 profile 标定域约 509–2159 tokens，不能用历史原生 extend 曲线预测当前小尾段路径，也不能宣称已解决任意长度最优切分。

## 6. 环境、代码和版本

| 角色 | CloudLab 主机 | GPU / IP |
|---|---|---|
| P | clgpu020.clemson.cloudlab.us | GPU0，10.10.1.1 |
| V / Gateway | clgpu021.clemson.cloudlab.us | GPU0+GPU1，10.10.1.2；V 单进程双 rank |
| D | clgpu019.clemson.cloudlab.us | GPU1，10.10.1.3 |

以上是最后测量的租约，不保证接手时主机仍可用。P/D TP1；V 双 rank 不等于已通过 TP2 模型服务验证。
全部为 V100S。环境脚本 `/users/Yizhzhu/.sglang-v100-pvd-env.sh`。
模型路径 `/users/Yizhzhu/pvd-models/Qwen2.5-7B-Instruct`：P/D 链接含本地权重，V 只需要 tokenizer；不能在 V 的同名路径假定权重存在。

- V checkout：`/proj/llm-course-PG0/Yizhzhu-node1-sglang-pvd/validation/pvd-joint-graph-20260930`。
- P/D checkout：`$SGLANG_PVD_ROOT/validation/pvd-direct-20260929`。
- 隔离 cuVS 25.10：`$SGLANG_PVD_ROOT/deps/pvd-cagra-extend25-venv/lib/python3.12/site-packages`。
- native view adapter：`$SGLANG_PVD_ROOT/deps/pvd-cagra-joint25`；旧 nogil adapter 为 `deps/pvd-cagra-nogil25`。launcher 管理加载路径。
- 实测 Torch 2.9.1+cu128、Triton 3.5.1。V index budget 2 GiB/rank；不能把 reservation 或 Torch peak 当整机物理显存峰值。

| 仓库代码（相对路径） | 作用 |
|---|---|
| `python/sglang/srt/disaggregation/pvd/cagra_kv_update.py` | `KVGraphBuffers`、`CagraKVUpdateBackend`；批量构图/更新、缓存、capture、fence、native owner |
| 同目录 `cagra_kv_small_tail.py`、`cagra_kv_prepare.py` | 小尾段合并/写边、批量 finite/K/ID 准备 |
| 同目录 `prompt_k_batch.py`、`prompt_vectors.py` | packed K 提取、连续 head 视图、固定均值中心化 |
| 同目录 `prompt_index.py` | provisional 生命周期、planned-tail、映射/预算、STORED/READY gate 和计时 |
| 同目录 `coordinator.py`、`control_server.py`、`vector_store.py` | chunk 完成证明、final kick、aggregate commit 和进度触发 |
| 同目录 `server.py` | CLI 依赖校验和 backend/manager 构造 |
| 同目录 `split_upload_policy.py`、`conn.py` | 请求开始选择切点、P 两段发送 |
| `test/registered/disaggregation/cloudlab_pvd_new_lease.sh` | P/V/D/Gateway 启动与各实验开关 |

本地 HEAD 为 `2a43e1c6949064b7f2e36baf525d4fb06ad03250`，仅包含之前的小尾段阶段；**最快组合有大量后续未提交修改/新文件，单独 checkout 此 commit 不够**。
最新实测源码快照位于 `benchmark/results/pvd_cagra_stream_completion_cloudlab_20261001/sources/`，其中的 measured/source hashes 记录实际上传版本。
线上 driver 的 checkout commit gate 与上传源码 hash 是两层校验；保留 launcher 的 expected-commit 校验，并核对实际 argv。
AGENTS.md 与 split policy 等已有修改应保留，不要为复现清空工作区。此交接文档未改变默认值、未提交代码、未启动远程服务。

## 7. 已验证的正确性和仍需验证的范围

- 最新 stream 范围轮有 **216 项测试通过**，包括 event 失败后的未知完成保留、跨 GPU/stream、最后 partial page、STORED 前不可发布等。
- 实测 native K、图边和固定均值 hash 在该同步对照的三个到达形状中一致。2048→2159 的采样真实 Q Top-10：rank0 mean **0.998214**，rank1 **0.999107**，worst head 均 **0.95**，无效 ID 为 0。
- native graph 是 14 邻居 + ring2；这些 Q 是真实模型的有限采样，不代表所有未来 Decode Q 的召回保证。
- planned-tail 的离线 Torch peak per rank 曾增加约 77.33 MiB；这不是完整 native/cuVS/服务显存峰值。并发 Entry 的持久 workspace 和 O(N²) 总点对成本仍需测量。
- 正式请求的 Prompt/token/输出 hash 一致，暖机分开，机器×请求监控确认只有测试服务 GPU PID；最后测量后 P/V/D 每张 GPU 均为 0 MiB。
- 尚缺更长 Prompt、多请求并发、TP2、真实压力下取消/失败竞态和物理显存峰值的新增在线证据。

## 8. 接手后避免重复的尝试

| 尝试 | 已有结果 / 下一步取舍 |
|---|---|
| 缩小 chunk 到 128/64 | 固定 1792 不缩小尾段；更晚切点可缩小，但 Prefill/提交前增加约 0.5–1 s，抵消微秒/毫秒尾段收益。2159-token 当前保留 chunk256/prefix2048 |
| 点积、Top-K、写边 mega kernel | 已有 `fused_core_block` 实验没有加速；保留 FP32 cuBLAS 路径，不默认启用 |
| 到尾段才捕获 CUDA Graph | 捕获成本计入请求后没有收益；使用首图阶段提前捕获的 planned-tail |
| 合并旧 Top-16 时选择性写边 | 111-token 写边仅约 0.047 ms；融合省约 0.039 ms 却增加约 0.091 ms 合并成本，未在线启用 |
| 新节点分块 Top-16 | 阶段 0.840→0.475 ms、在线 GPU 2.258→1.870 ms；但同轮 READY 6.147→6.555 ms。代码保留，`PVD_NEW_TOP16=0` |
| stream event 替代整卡 fence | 受控实验不再等无关 stream；普通在线 fence 2.130→2.168 ms、READY 5.086→5.281 ms，`PVD_STREAM_COMPLETION=0` |

下一步优先按同一请求时间轴定位提交→尾段开始、输入准备、host 提交与发布成本，或按当前后端重标定切点。新增 GPU kernel 的收益必须落到完整尾段和双 READY，再看客户端。
不要把 fence 等待当可全部删除的 CPU 开销，也不要在 GPU 未完成前发布 READY。

## 9. 证据与复现入口

以下都是仓库内相对链接；对应同名目录保存 raw JSON、日志、配置、argv、hash、测试结果和驱动快照。

1. [联合图算法与最初端到端收益](benchmark/results/pvd_cagra_joint_kv_optimization_cloudlab_20261001.md)：原生大尾段后端约 0.48 s 降至约 24 ms；早期客户端收益约 0.48–0.57 s，不能当作后续每项都带来的收益。
2. [后移首图切点](benchmark/results/pvd_cagra_late_cuts_cloudlab_20261001.md)：1792→2048 缩小尾段，chunk256 优于更小 chunk 的整体等待。
3. [预分配、固定视图、提前捕获、相似度复用](benchmark/results/pvd_cagra_planned_tail_cloudlab_20261001.md)。
4. [最后 chunk 提前更新，5.607 ms](benchmark/results/pvd_cagra_final_overlap_cloudlab_20261001.md)。
5. [stream 同步范围对照，快配置基线 5.086 ms](benchmark/results/pvd_cagra_stream_completion_cloudlab_20261001.md)。
6. [新 Top-16 的局部收益与在线结果](benchmark/results/pvd_cagra_new_top16_cloudlab_20261001.md)。
7. [现有预测器及旧成本模型](benchmark/results/pvd_kv_edge_predictor_cloudlab_20261001.md)。

最新完整归档入口：`benchmark/results/pvd_cagra_stream_completion_cloudlab_20261001/`。
重点看 `online.json` 的 `base_a/base_b`（上述推荐配置），`raw/*_v.command`、`raw/source_hashes.json`、`sources/`、`drivers/`、`fixture_sha256.txt`、`artifact_hashes.json`。
`batch_a/batch_b` 是启用 stream event 的候选，不能误选为推荐快配置。

已有工作区在线驱动 `.pvd-compare-artifacts/run_stream_completion_online.py` 会运行完整 ABBA 并上传源码、重启服务；分析脚本为 `.pvd-compare-artifacts/analyze_stream_completion_online.py`。
复用前检查租约、GPU 占用、checkout/模型/adapter，并使用新输出目录避免覆盖证据。该驱动复测同步两臂；若只复现快配置，明确选 device/base 臂。
无需为阅读本交接重新跑 GPU 测试。今后修改比较时保持模型、Prompt、chunk、切点、资源、暖机、输出长度和日志配置相同，所有正式样本保留。
