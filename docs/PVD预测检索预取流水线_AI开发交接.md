# PVD 预测检索预取流水线：AI 开发交接文档

更新日期：2026-09-21。

最新：[rank 消息与 CPU 参与者](PVD_Rank_Wire_CN_EN.md)：有界消息绑定 worker
epoch 和精确 staging；本地 APPLIED 后仍等全局 RESUME 才能读取。56 个新单测，
全量 Windows 1549/14 skipped、WSL 1554/9 skipped。后续实际 spawn 独立进程验收
已通过：2/4 个 CPU rank 各自持有 bank，pipe 只传有界 JSON 控制消息；覆盖换 bank
后异常和进程退出。最新全量 Windows 1553/14 skipped、WSL 1558/9 skipped。
协议提交 `8d18e76c0`；尚未连接真实模型 TP、跨节点传输、CUDA 完成证明或 Scheduler。

最新：[稀疏交付 Step 7](PVD_Sparse_Delivery_CN_EN.md) 修复 reserve 未抵达时的
取消恢复。仅已知 Entry、当前 V epoch、原子建立并保留完整身份关闭标记后，才能
证明迟到 reserve/start 不会写向 D。单纯未找到记录、缺少证明、旧 epoch、原生
UNKNOWN 或容量不足仍不能释放。新增 17 个 HTTP/并发测试。直接打包步骤已作为
`2220a1d6e` 提交并推送。
最新完整回归：Windows 1493 passed / 14 skipped；WSL 1498 passed / 9 skipped。

最新：此前未能推送的 `d44dd9ebf` 已成功推送。V 稀疏打包改为直接写入持有生命周期
与预算的最终 staging，消除逐组临时副本。底层提供显式 CUDA copy 入口，但其返回
不是 GPU 完成证明，不管理注册或资源释放；生产 CUDA sparse 仍拒绝启动此路径。
详见[稀疏交付 Step 6](PVD_Sparse_Delivery_CN_EN.md)。最新 Windows 1476 passed /
14 skipped，WSL 1481 passed / 9 skipped；新增 3 个真实 CUDA 测试跳过，尚未验收。
以下固定设计目标不变。

当前权威更新（覆盖下面历史“未推送”等状态）：见[稀疏 Delivery 推进](PVD_Sparse_Delivery_CN_EN.md)。
历史本地 commit 与交付步骤至 `309e464be` 均已推送 GitHub `pvd-disaggregation`；
用户现在要求每一步 commit 后 push。V 在 Entry/index lease 下打包所选 K/V；D
持有目标缓冲并验证 fence，全部 rank 安装后 ACK；请求级驱动已接该 HTTP 路径。
`--wire-sparse-loop` 真实双 CPU 小模型通过：4 次交付、1600 bytes、21 次 attention
对照，最大误差约 3.58e-7，没有本地打包回调。Windows suite 1436/11 skipped，
WSL 1441/6 skipped。HTTP 控制为真实本地网络，数据拷贝仍为 fake transport。
生产 GPU 打包/attention、原生 RDMA、实际 TP/Scheduler 激活与 CAGRA 仍有缺口，
不能开启生产 sparse 模式或宣称最终目标已完成。

最新：[请求级 CPU 刷新驱动](PVD_CPU_Refresh_Driver_CN_EN.md)：根据各请求正式 token
时钟选择边界末 token 的目标 Q，每次 poll 最多启动一次 capture，保留迟到任务，
仅边界安装。真实独立 draft 闭环已使用；9 个 HTTP/CPU 新测试通过。
此前真实 draft 阶段提交 `92ae30c23`，未推送。生产 Scheduler loop/cleanup、GPU 稀疏
attention、授权 sparse MR 交付、TP 激活与 CAGRA 仍有代码缺口，不能说只缺 benchmark。

最新：[独立真实 SGLang draft CPU 闭环](PVD_Real_Draft_CPU_Loop_CN_EN.md)。可选验收已
不依赖固定候选，两个独立随机小 Llama 跑通 draft → target Q → V HTTP → 稀疏 Decode
→ 原 Req 提交；验证目标状态/RNG 不变与私有池，修复 backing KV alias 检查。
无质量/生产加载/GPU 结论。此前 ScheduleBatch 接点已提交 `bef8a59c4`，未推送。

最新状态（覆盖下方历史记录）：[真实 ScheduleBatch 结果接点](PVD_CPU_Schedule_Result_Bridge_CN_EN.md)。
原结果处理器保持 Req 唯一输出写入点；显式 CPU bridge 校验 dispatch 身份、观察提交、
拒绝重放、跳过撤回成员。29 个新测试；WSL 全量 1365 passed / 6 skipped；真实 Req/
ScheduleBatch/模型前向通过，21 次 attention 对照。metrics/stream/cache-release
回调仍为测试 spy，尚非完整 serving 验收。前一阶段 batch 已提交 `1553d036c`，未推送。
下一步连接独立真实 draft 与请求级调度；GPU/TP/MR/CAGRA 仍有实际实现缺口。

最新组合推进：[batch 执行器与真实多请求验收](PVD_CPU_Batch_Execution_CN_EN.md)：
共享 batch lease、完整结果身份校验/提交、固定请求槽位与可复用 CPU forward，
已在真实多请求不同长度/成员变化下完成 HTTP 刷新和安装。39 个新单测、19 次真实
attention 对照，最大误差约 `2.38e-7`；wait-all、取消和整批失败均通过。
此前生命周期已提交 `81bfb64b4`，未推送；本后续修改尚未提交。
仍有正式 ScheduleBatch、GPU/RDMA/CAGRA 的实际实现缺口，非仅缺硬件测试。

最新：[CPU 生命周期接点](PVD_CPU_Decode_Lifecycle_CN_EN.md)已驱动真实 CPU smoke，
覆盖准入、唯一 Decode 票据、正式 token 提交、共享目标执行、超时/EOS/取消/drain；
25 个新增契约测试通过。此前真实 Decode 消费提交为 `8f482e631`，未推送；
此后续修改尚未提交。正式 Scheduler 只核对未修改。下一步实现 batch 级共享执行
lease 与按身份匹配的结果提交，不能给 batch 每个成员申请一个独占 lease，
也不能与服务的 `req.output_ids` 追加形成重复提交。

最新验收：[同一真实 CPU Decode 的检索消费](PVD_Controlled_CPU_Decode_CN_EN.md)
已将真实输出快照、HTTP 检索、组门控安装与同一个模型生成序列连通。9 个 D token，
边界 4/8，18 次独立 attention 对照；迟到等待、正式前缀补查、真实前向失败终止
均通过。覆盖下文历史“仍为分开的 fixture”限制，但不代表 Scheduler/GPU/TP/RDMA/
CAGRA 已通过。此前工作提交为 `3c4b0c479`，未推送；此后续修改尚未提交。
下一步明确并测试 Scheduler 独占的生命周期/dispatch 接口，再接在线执行。

最新推进：[受控请求刷新闭环](PVD_Controlled_Request_Loop_CN_EN.md)，统一控制端
epoch/prefix snapshot 从一次捕获贯穿分片 HTTP、GQA 并集、打包与
[全部 rank 安装门控](PVD_Rank_Install_Contract_CN_EN.md)，不修改旧结果身份。
边界补查策略已确认：到边界才首次发起时，暂停 D，用正式前缀的目标模型 Q，
不调用 draft；已提前发起但迟到的结果仍等待原操作。两条路径都通过真实 CPU Llama
和本地 HTTP 验证。下一步将结果交给同一个真实 CPU Decode 序列消费，目前仍分开
验收；不代表已接入 Scheduler、实际分布式 TP、GPU/RDMA 或 CAGRA。

最新推进：[离线真实 target-Q probe](PVD_Target_Q_CPU_Probe_CN_EN.md)
已复用目标权重与独立 CPU 池，实现明确限定 CPU/TP1 Llama 的 post-RoPE Q 捕获，
通过真实模型数值、状态隔离与异常检查。
这覆盖历史“完全没有真实 target-Q 路径”的描述，不代表已接入在线 Decode 或 GPU。

最新进度见[顺序推进计划](PVD_Implementation_Roadmap_CN_EN.md)与
[第 1–3 步验收记录](PVD_Roadmap_Steps_1_3_CN_EN.md)。真实 K/Q 已通过 V 本地 HTTP
精确检索、稀疏配对 K/V 打包与分片 CPU 工作集安装。最新
[真实模型稀疏 CPU Decode](PVD_Sparse_CPU_Decode_CN_EN.md) 已执行真实 Llama attention：
全选 logits 与原后端误差为 0，子集通过逐层独立 softmax、Prompt 池毒化与异常清理验证。
WSL 回归 **1191 passed / 6 skipped**，Windows **1186 passed / 11 skipped**。
GQA 已确认：同层同 KV head 的 Q heads 检索
token 取并集去重；显式上限，超限拒绝更新，不静默截断。首轮完整 Prompt 不变。
正式 D 稀疏 attention、跨 rank 安装、在线预取调度、GPU/RDMA 与 CAGRA 仍未完成或未验证。

最新执行证据覆盖下文历史进展中“尚无前向执行”的描述：
[真实 CPU draft 前向与剩余限制](PVD_Draft_CPU_Execution_CN_EN.md)。
现已通过 33 次真实 tiny-Llama CPU forward；不代表 GPU/RDMA 或正式权重验证。
非 KV 临时内存上限改为显式声明，未知上限拒绝 provider 准入。
WSL 全量回归为 1104 passed / 6 skipped。

本文供另一位 AI 在没有历史对话的情况下接手。请先完整阅读，再查看代码。
本文整合用户当前要求，不需要通过历史对话猜测设计。
如用户后续给出新要求，以用户最新明确要求为准，并同步维护本文。

本次已实现的异步首轮交付约束及限制见
[最终等待队列首轮 KV 中英双语目标](PVD_Waiting_Queue_Bootstrap_CN_EN.md)。

## 1. 一句话目标

在现有 SGLang PVD 框架中，使用用户可配置的独立小模型预测未来 token，
再由目标模型的独立 probe 分支生成检索 Q；V 提前执行 CAGRA 搜索并发送相关 Prompt KV，
使检索与网络传输尽可能和 D 的正式 Decode 重叠。

预测 token 不作为正式输出。每个请求独立按 M 个正式 Decode token 刷新。
新请求加入不触发旧请求额外更新，不重置旧请求时钟，不取消旧请求在途预取。

新请求的首轮采用最终等待队列触发的拉取：D 等到调度器把请求放入最终等待队列
（`scheduler.waiting_queue`）后，才发起完整 Prompt KV 的交付，目标是已注册的 staging 缓冲区，D 再把它 unpack 到该请求已预分配的最终 KV 页。
传输仍由 V 执行获授权的 RDMA WRITE；“拉取”指由 D 发起，不是改变传输方向，也不是 RDMA READ。
请求在等待队列中标记为 not-runnable，安装校验与 ACK 通过后才进入运行 batch。
该路径已在 `--pvd-waiting-queue-bootstrap` 下实现，真实硬件验收待完成。

最终要验证的是端到端刷新等待、TPOT、吞吐、显存与质量的权衡，
不是只证明一个 CAGRA search 或 RDMA WRITE 能运行。

## 2. 必须保持一致的设计结论

| 项目 | 用户已确认的要求 |
| --- | --- |
| 正式生成 | D 的目标模型产生正式 token |
| 预测模型 | 独立小模型；名称或本地路径由用户配置，不写死 |
| 模型 revision | 可选；实际解析到的版本用于复现记录，不是开发门槛 |
| query 来源 | 小模型预测 token → 目标模型独立 probe → 目标模型空间的 Q |
| 近似 | 允许稀疏检索近似，但必须测量质量退化 |
| 刷新周期 | 每个请求独立按正式 D token 数计算 |
| 首轮初始化 | D 等请求进入最终等待队列后发起拉取，先写入已注册的 staging 缓冲区，再 unpack 到已预分配的最终 KV 页；V 执行获授权的 WRITE；请求在等待队列内为 not-runnable，安装完成后才入运行 batch |
| 新请求加入 | 就绪后接纳，不重复首轮传输；旧请求的周期与预取保持不变 |
| 周期等待策略 | 运行 batch 内到期请求仍可同步等待；未就绪的新请求留在 batch 外，不要求旧请求陪等其初始化 |
| V 硬件目标 | V100S；当前本地没有该实验硬件 |
| 本地开发 | 不因无 V100S、cuVS 或具体模型权重而停止通用实现和 CPU 测试 |
| 检索范围 | 当前请求自己的 Prompt KV，优先在选定的 V group 内完成 |
| 新生成的 KV | 暂时留在 D，不持续回写 V |
| 数据安全 | 保留已有 Mooncake/PVD 原生句柄、fence、epoch/generation 和资源生命周期保护 |

“自定义模型”不等于“所有架构无条件兼容”。真正加载或启用时检查 tokenizer、
目标架构的 Q 捕获能力、设备、dtype 和预算；不支持时明确说明，不隐式改语义。

## 3. 项目位置与版本状态

- 项目：SGLang V100 PVD 分离框架。
- 原工作区：`D:\code\sglang-V100-PVD`；迁移到 Linux/WSL 时以实际仓库根目录为准。
- 目标分支：`pvd-disaggregation`。
- 本交接创建时 HEAD：`7beb1cc58c7ddfed59cbf139588e3cd8364ae23e`。
- 该 HEAD 之外有尚未提交的预取基础设施、测试与文档，不能只看 GitHub HEAD 就假定没有这些修改。
- 用户的未跟踪目录 `Claude outputs/` 不属于本任务改动，不覆盖、不删除、不擅自提交。

接手先检查 `git status --short`、当前分支和 HEAD，阅读适用的 AGENTS.md。
不要假定本文件记录的 commit 永远是最新代码。
未经明确要求，不自动 commit、push、换分支或重置工作区。
不要为了试装 CAGRA 升级整个 SGLang/CUDA/Mooncake 环境。

## 4. P、V、D、Router 分别做什么

- Router 选择 P、V worker group、D，并给 P 和 D 一致的请求与 V 关联信息。
- P 计算完整 Prompt KV，按现有布局上传到选定 V。
- V 保存完整 Prompt KV；目标版本中增加请求内索引和 CAGRA 检索。
- D 正式 Decode，保留生成 KV，管理当前 Prompt 工作集与下一轮预取。

V 节点、V worker group、V rank/shard 不是同一概念。
一个 V group 可以服务多个 D。Router 已选择 V，不意味着 V 与某个 D 永久一对一绑定。
没有必要仅因为使用 CAGRA 就引入跨 V group 分片。

源/目的布局按 layer、KV-head、token/page 的所属关系匹配，不假定 rank 编号相等。
现有文档描述 P TP1/TP2、V 两个存储 shard、D TP2/TP4 的若干路径；
具体支持范围必须核对最新代码，不宣称任意 M→N 或 GQA/MQA 组合已经支持。

## 5. 当前真正实现到了哪里

### 5.1 既有服务路径

- V 保存完整 Prompt KV，Entry 可以供多个 Delivery 使用。
- D 在首次 forward 前及每请求到期时从 V 获取完整 Prompt KV。
- 每个请求有独立 `RefreshClock`。
- 正式 D token 数不包含 P 采样的第一个输出 token。
- 刷新成功、安装及确认后才推进该请求时钟。
- 目前 `full_prompt` 才是已接入的检索模式。
- 当前刷新有同步等待，可能阻塞整个当前 batch 的下一次 forward。
- 现有路径不等于稀疏检索，也不等于已经隐藏网络延迟。

### 5.2 最近增加、但尚未接入服务的基础设施

1. `prefetch.py`：模型无关的 `PrefetchClock` 与 `PrefetchTicket`。
   - 固定每请求的目标安装边界。
   - 可提前开始，但不能提前安装或推进周期。
   - 单请求只有一个 pending ticket。
   - 拒绝错误身份、旧轮次和正式计数越界。
   - `close()` 保留 pending 身份，不代表释放/终止 RDMA。
2. `bootstrap.py`：模型无关的 `BootstrapGate` 与 `BootstrapTicket`，用于等待队列触发的首轮拉取。
   - 只有进入最终等待队列才是触发点，更早阶段不得发布授权。
   - 等待队列触发与 KV_STORED 相互独立，任意顺序到达均可。
   - 每请求只有一次授权；重复控制请求复用首个 ticket，不产生第二次 WRITE。
   - RECEIVED 不等于 RUNNABLE；AUTHORIZED 不能直接跳到 INSTALLED。
   - `handoff()` 只能调用一次并返回正式计数 0，round 0 不会被重复获取。
   - `close()` 保留 pending 身份并拒绝迟到完成；不释放、不 fence、不排空。
3. CAGRA 检测脚本：默认 inventory，显式 smoke 才执行 GPU 操作。
4. `--pvd-waiting-queue-bootstrap`（默认关闭）下的等待队列首轮接入：
   - `PVDKVManager` 的 `open_bootstrap_gate/bootstrap_runnable/enter_waiting_queue/close_bootstrap_gate`。
   - `PVDKVReceiver` 入队时开 gate，Entry 校验通过时上报 KV_STORED。
   - `decode.py` 每轮推进完整的最终等待队列，包括没有新请求到达的轮次；
     batch 构建改为统计已接纳请求数而非队列下标，跳过 not-runnable 请求且不占用 batch 名额。
     关闭该开关时不会跳过任何请求，计数与原先的下标比较完全一致。
   - 首轮交付与 ACK 使用异步轮询的控制 future，TP 协调和 unpack 仍在调度线程。
     周期刷新保留同一个 `full_prompt` 协议的同步驱动；延期请求无需等新到达就会重试。
   - 各 rank 共同确认源就绪和 staging 余量后选择 wave。wave 失败可能影响同组新请求，
     但不包含已在运行 batch 的旧请求。
5. `prediction.py`：draft/probe 接口，含 fake，不加载任何模型。
   - `DraftConfig` 接受模型名或本地路径、可选 revision、device、dtype 与 token 预算；
     不设默认模型；缺少 revision 时记为 `local/unknown`，不伪造。
   - `CommittedPrefix` / `snapshot_committed` 向预测分支交付不可变且已拷贝的正式状态，
     使其无法触碰正式 id、位置或 KV。
   - `run_isolated` fork torch RNG，采样型 draft 模型不会改变正式采样器的后续输出。
   - `QueryVectors` 显式携带向量空间、版本、层、head 范围、位置与有效长度；
     `PredictionPipeline` 拒绝空间不等于目标模型的 query，draft 空间 Q 永远无法用于检索 target K。
   - 同时拒绝：属于其他请求的预测、基于过期 prefix 的预测、超预算预测、
     probe 返回未请求或重复的层。
   - `FakeDraftProvider` / `FakeTargetProbe` 支持无权重开发。
6. `draft_hf.py`：第一个具体 `DraftProvider`，基于 Hugging Face causal LM。
   - `transformers` 在 loader 内惰性导入，不成为新的硬依赖；加载步骤可注入，因此无需权重即可测试。
   - `VocabularySignature` 比较词表大小、BOS/EOS 以及固定探针编码的指纹。
     与目标模型 tokenizer 不一致的 draft 会被拒绝：预测出的 id 对 probe 而言意味着不同的文本，
     而目前不存在转换步骤。
   - device 与 dtype 按模型实际加载结果校验，而非假定。
   - token 预算按模型返回值强制执行，不只按请求值，因此 generate() 超额返回仍会被截断。
   - 记录 loader 解析出的 revision；本地路径无 revision 时记为 `local/unknown`。
   - 目前没有任何代码构造它；opt-in，未接入服务。
7. `index_lifecycle.py`：V 侧检索索引状态机，不含任何向量。
   - ABSENT、BUILDING、READY、FAILED 可区分：「未就绪」不等于「失败」。
   - 只有在完整 Prompt KV 已存储且可见之后才允许构建索引。
   - `deliverable` 刻意与索引状态无关，首轮完整交付不会产生 INDEX_READY 依赖。
   - 已建成的 Prompt 索引不可变，且就绪状态不会被一次检索消耗，可服务多轮 Delivery。
   - `authorize_search` 校验 query 向量空间与调用方的 id 映射版本；
     索引版本、映射版本与向量空间保持为彼此独立的身份。
   - 失败的构建可在上限内重试，超出后拒绝；`close()` 保留 descriptor，
     不释放任何向量、图存储或映射。
   - 未接入 V 控制服务；目前没有任何代码构建或检索索引。
8. `index_search.py`：检索后端契约、精确参考实现、逻辑选择与合并策略。CAGRA 成为可替换项而非重写。
   - `IndexBackend` 是接缝：`build` 与 `search`。`BruteForceIndexBackend` 精确、纯 CPU、
     不需要 cuVS，因此它同时是将来衡量 CAGRA recall 的基准，而不是一次性桩。
   - `select` 通过带版本的 `IdMapping` 返回**逻辑** token/page id，绝不返回原始地址；
     由负责 gather KV 的一方在自己的边界检查下解析地址。
   - 被多个 query 选中的同一 token 只选取一次，保留其最佳分数。
   - `merge_selections` 必须显式指定策略（`per_layer`、`union`、`intersection`），其余一律拒绝。
     不同层的分数永远不互相排序，因为全局 Top-K 等于一个未声明的建模假设。
   - 并列时按更小的行号确定性排序，测试不依赖 kernel 顺序。
9. `prompt_vectors.py`：从真实已存储 EntryShard 中提取 Prompt K。
   - 输入是 `kv_packer` 打包出的字节缓冲区，加上经校验的存储 `KVLayoutSignature`
     与 `KVShardManifest`，而不是已经提取好的张量。
   - 只取 K，跳过 V 分量；按 `last_page_valid_tokens` 排除末页 padding，
     不会用 prompt 中不存在的 token 构造向量。
   - 层与 KV head 保持分离，不做任何 head 平均或拼接。输出按（全局层，全局 KV head）
     分组，全局 head = `manifest.rank * kv_heads_per_rank + 本地下标`；
     过滤器使用全局 id，请求对端 shard 的 head 会报错而非返回空结果。
   - dtype、分量偏移、形状与 head 归属全部来自 layout 元数据，并与缓冲区实际大小交叉校验。
   - **位置编码**：存储的 K 是 post-RoPE（模型在注意力层写入 k 之前完成旋转，
     见 `models/llama.py`）。提取不做任何变换，因此不会二次旋转；
     `require_compatible_query` 拒绝声明了不同编码的 Q。该值必须显式传入，绝不从元数据推断。
   - `QueryHeadMapping` 定义 MHA/GQA/MQA 的 query-head → KV-head 分组，拒绝不整除的布局；
     query head 数量作为参数传入，因为 `KVLayoutSignature` 并不携带它。
   - 向量**自有副本**：绝不修改已存储 KV，索引也不借用 Entry 注册区内的内存。
     传入 budget 时对副本计费，服务路径现在一定会传（见第 11 条）。
   - 这份副本同时也是**设备迁移**发生的地方。`device=` 指定副本落在哪里，
     默认跟随已存储分片所在设备；manager 传入其 backend 声明的设备。
     绝不为了迁就检索后端而移动权威 KV 池。
   - `Selection` 新增可选 `kv_head`，head 身份在选择与合并中得以保留；原有按层调用不受影响。
   - 未接入 V 控制服务。
10. `prompt_index.py` 与 `VectorKVStore` 接线：第一个真正驱动 `IndexGate` 的实现。
    - store 新增可选 `prompt_index`，**默认 None**，不传时行为与之前完全一致，功能整体关闭。
    - `_publish_stored_locked` 标记 gate 为 KV 可读——这是唯一允许开始构建索引的时点；
      锁内不做其他事情。
    - `progress_prompt_indexes()` 是一次有界、由调用方驱动的步骤，与 upload/decode close
      的推进方式一致。它为每个候选 pin `allocation_guard` 再拷贝，因此已开始释放的 Entry
      会被跳过而不是被读取（release 一旦请求，`pin` 即拒绝）；提取在 store 锁之外进行，
      无论成败都会 unpin。
    - 构建失败记录在 gate 上并如实返回，绝不抛出：无法建索引的 Entry 仍是 STORED 且可交付；
      重试在 gate 的上限处停止。
    - `_free_allocation` 在页面归还分配器之前关闭 gate，只丢弃索引自己的副本；
      页面、注册与 MR 仍由原有所有者释放，未作改动。
    - `PromptIndexManager.search` 接受 **`SearchRequestIdentity`**：调用方自己声明
      vector space、位置编码语义、Entry、层与 KV head，并可选地钉住 index 版本与
      id 映射版本。所有比较都针对这份身份，**缺失的字段绝不从索引自身配置补齐**。
      返回 `SearchResult`，其 `validated` 元组精确列出实际做过哪些比较，
      未钉版本的检索会如实显示为未钉，而不是笼统地称为"已校验"。
      所有身份检查都在调用后端之前完成。
    - **每一份保留的副本都在其存在之前被计费。** 提取的副本与后端自身的存储分别在
      两个按尝试隔离的 owner 下预留，检索的有界临时空间在第三个 owner 下按需预留并
      在结束时退还。`IndexBackend` 新增 `device`、`build_footprint(rows, dim)` 与
      `search_footprint(rows, dim, num_queries, top_k)`，让所有者在分配之前预留，
      而不是事后才发现开销。
    - 按尝试隔离的 owner 意味着：在 Entry 关闭并重建之后才完成的旧构建，
      只会退还它自己的额度，绝不会退还新尝试的额度。
    - 只要仍有检索持有这些张量，就不退还额度：`close()` 立即摘除记录，
      由最后一个离开的检索来真正回收。在 close 处退还，会把仍在被读取的内存报告为空闲。
    - **容量不足是背压，不是失败。** 构建过程中的 `TransferCapacityError` 会调用新增的
      `IndexGate.abandon_build`，把 gate 恢复到构建前状态并归还该次尝试，
      因此短暂的压力不会消耗掉一个 Entry 仅有的几次永久尝试。
      出于同样的理由，`progress_prompt_indexes()` 把 `deferred` 与 `failed` 分开统计。
    - 关闭时与构建失败时都会退还向量副本的预算；在 Entry 已关闭之后才完成的构建会被丢弃
      而不是安装。注册表加锁，因为 `close()` 可能由 guard 释放回调在另一线程触发。
    - 交付路径完全不查询这些状态，因此没有引入 INDEX_READY 依赖。
11. V 侧检索服务接线，位于 `--prompt-index-vector-space` 之后（默认不设置，
    此时 V 不构建任何索引，行为与之前完全一致）。
    - `pvd/server.py`：`_build_prompt_index(args)` 返回 manager 或 `None`；
      reaper 循环每个间隔驱动一次 `progress_prompt_indexes()`，单 rank 与 group 启动器都覆盖。
    - **`--prompt-index-budget-bytes` 与 `--prompt-index-vector-space` 必须同时给出**，
      不猜任何默认值。它创建的 `TransferBudget` 与 `--transfer-staging-budget-bytes`
      是**两个独立对象**：索引副本绝不能侵占传输已经据以准入的余量。
      只给 vector space 而不给预算的启动会被拒绝，而不是悄悄地不计费地服务。
    - shard 路由：`POST /internal/v1/indexes/progress`、`POST /internal/v1/indexes/search`、
      `GET /internal/v1/indexes`；未配置索引时前两者返回 `{"enabled": false}` 或拒绝。
    - 检索返回逻辑 token/page id，并带层、KV head、度量、id 映射版本、index 版本
      以及 `validated` 列表，绝不返回地址。
      请求有界：最多 64 条 query，`top_k` 不超过 512，query 各行长度必须一致。
    - **请求自带身份。** `vector_space` 与 `positional_encoding` 是必填字段；
      `expected_index_version` 与 `expected_id_mapping_version` 是可选的版本钉，
      缺失就保持缺失，绝不由 shard 代填。来自另一个模型、形状恰好一致的 query
      会被 400 拒绝。query 张量的构造与检索本身都在事件循环之外执行。
    - **设备策略（2026-09-20 决定）。** `BruteForceIndexBackend` 按声明固定在 CPU：
      在 V worker 上 GPU 持有权威 KV 池，而每个被索引 prompt 的精确 float32 镜像
      会与之争抢正是池所需要的那部分显存；同时 query 以 JSON 到达，天然产生在主机侧。
      后端显式地把两侧都放到它声明的设备上——dtype 转换不移动设备，
      因此绝不从转换推断设备。设备驻留型后端（CAGRA）声明自己的设备，
      同一套预留与放置机制随之生效。
    - 交付路由未作改动，也完全不查询索引。
12. CPU 测试：时钟、首轮门控、等待队列触发、decode.py 调度钩子、draft/probe 接口、
   新请求隔离、现有 refresher 选择范围、检测脚本行为。
13. 完整目标和阶段记录文档。
14. 独立单 shard HTTP 检索客户端（`search_client.py`），包含调用者身份、版本钉、
    单次请求关联、回复大小限制和逻辑选择校验。已用合成 query 测试真实 V shard
    路由，但正式 D 服务尚未调用。最新修复、**917 passed / 6 skipped** 证据和
    后续边界见[中英检索复核记录](PVD_Shard_Search_Review_CN_EN.md)；下方旧测试数字
    保留为历史记录，不代表最新总数。

### 5.3 尚未实现

最新基础层进展（2026-09-21，第五轮）：针对真实接口做契约审计，发现并修复适配层的**七个缺陷**。
**1091 passed / 6 skipped**；15 处变异验证修复有效。

上一轮把剩余阻塞称为"环境问题"，这个判断是错的：适配层并未匹配它所依赖的接口，
而测试替身之所以"同意"，是因为它们基于同一套假设写成。实际情况：

| 假设 | 实际 |
| --- | --- |
| `req_pool.alloc(1)` / `free(index)` | `alloc(reqs: list[Req])` 原地赋值 `r.req_pool_idx`；`free(req: Req)` 断言其非空并清空；槽位 0 是 padding 行 |
| `output.next_token_logits` | `ModelRunnerOutput.logits_output.next_token_logits`，形状 `[#seq, vocab]`，且为 Optional；也可能是 `PPProxyTensors` |
| 用 `None` 关闭捕获 | 应为 `CaptureHiddenMode.NULL`；logits processor 会对其调用 `.need_capture()` |
| `extend_start_loc = (0,)`、无 `extend_num_tokens` | extend 路径两者都必需，`extend_start_loc` 为前缀和 |
| 释放索引用 CPU int64 | `free()` 会与 `free_pages` 拼接，后者是**分配器自身设备**上的 int64 |
| 到处按 token 调 `alloc(1)` | 分页分配器断言页对齐并返回整页 |

另有一个保留缺陷：每个 `ForwardBatch` 都被存入 `self.batches`，在适配器整个生命周期内
钉住其设备张量。现已改为有界诊断（不持有任何张量），并加入回归测试：
比较 1 次与 21 次前向之后适配器自身状态，要求不增长。

**真实导入已不再受阻。** 准确链路为：PyPI 的 torchvision 构建自不同的 torch
（`operator torchvision::nms does not exist`，需装 CPU wheel），`transformers` 需锁定
`5.8.1`（`5.17` 会报 `'qwen3_asr' is already used by a Transformers config`），
随后依次为 `openai`、`partial_json_parser`、`dill`、`sentencepiece`、`einops`、
`compressed_tensors`、`gguf`。安装后，真实的 `ForwardBatch`、`ForwardMode`、
`CaptureHiddenMode` 已能在两种模式下构造，真实的 `ReqToTokenPool` 与
`TokenToKVPoolAllocator` 已能分配与释放。这些测试在其他环境会跳过，跳过原因携带精确异常文本。

**构造不等于执行：从未运行过任何模型前向。** 详见
[中英复用审计](PVD_Draft_Worker_Reuse_Audit_CN_EN.md)。

最新基础层进展（2026-09-21，第四轮）：ForwardBatch 接缝已补齐，并修复了它暴露出的一个真实缺口。
**1062 passed / 6 skipped**；新增 13 处变异验证有效。

`draft_forward_adapter.py` 将 `DraftForwardInputs` 映射为 `ForwardBatch` 并在 draft
`ModelRunner` 上执行。映射由 `forward_fields()` 以纯数据形式产出，构造经由可注入工厂完成，
因为导入 `forward_batch_info` 会拉入 triton、torchvision 与 HTTP 栈，而 PVD CPU 测试套件
并不需要它们。因此测试分别验证两件事，且都不越界声称：**取值**用记录型替身验证，
**字段名**通过解析 `forward_batch_info.py` 源码验证——每个 key 必须是真实字段，
每个无默认值的字段必须被提供，上游改名会直接让测试失败。
`PrivatePoolAllocator` 驱动 draft 的 `ReqToTokenPool` 与 KV 分配器，缺少私有池时直接拒绝存在。

编写适配器时暴露出 runner 的一个真实缺口：注意力后端通过
`req_to_token_pool.req_to_token[req_index, :seq_len]` 定位 KV，而此前 runner 只分配行、
从不登记，前向会读到该行中恰好残留的内容。现已为 `SlotAllocator` 增加
`write_mapping`/`clear_mapping`：前缀在读取它的前向之前完成映射，每步恰好扩展一个位置，
释放时**先**清空该行再归还槽位，使被复用的槽位不会继承不属于它的行。

**从未构造过真实 `ForwardBatch`，也从未执行过任何前向。** 详见
[中英复用审计](PVD_Draft_Worker_Reuse_Audit_CN_EN.md)。

最新基础层进展（2026-09-21，第三轮）：修复了 draft 适配层的九个正确性缺口，
并实现了仅预测的执行路径。**1035 passed / 6 skipped**；15 处变异验证修复有效。

修复前已复现：`release()` 尚未执行时 scratch 就被退还；清理完成前准入名额已重新开放；
多个分支共用一个 runner，其 `release()` 不带参数、无法说明释放的是谁的行；
在 `branch()` 之外调用 `predict()` 可以成功，且没有预留也没有清理；
worker 包装是黑名单，任何未列出的方法都能穿透；
以及两个指向同一缓冲区的不同 pool **对象**被当作"私有内存池"接受。

现在：每个分支拥有自己的 `DraftExecutionHandle`（请求槽、KV 行、scratch）；
权重与内存池共享，并**一次性**计入独立的持久预算；预算与准入名额只在句柄释放**之后**
归还，释放失败则隔离该分支，而不是把可能仍然存活的内存重新发放；
`predict()` 在其分支之外（或跨线程）被拒绝；worker 表面改为**白名单**
（`get_memory_pool`、`model_config`、`device`），上游明天新增的方法在被审查前不可达；
内存池检查比较底层存储，并如实报告 `storage_verified`，不会在未验证时声称"私有"；
tokenizer 兼容性复用 `VocabularySignature`（size、特殊 id **以及**编码指纹），
并同时校验前缀与返回的 token id；`build_draft_server_args` 生成私有深拷贝，
将 `--pvd-draft-*` 映射到 `model_path`/`tokenizer_path`/`revision`/`device`，
并在副本内关闭推测与 disaggregation 字段，目标配置完全不被修改。

`draft_runner_sglang.py` 是执行路径：私有请求槽、私有 KV 行、从零开始的绝对位置、
每步一行、有界续写，成功/失败/取消都会清理。**前缀每次调用重算**，位于显式的
`prepare_prefix` 接口之后——这是正确性基线，其 prefill 开销后续必须与预取窗口对比测量，
并非最终的延迟方案。执行**串行化**：资源独立拥有并不等于 `ModelRunner` 可重入。

`ForwardBatch.init_new` 需要 `ScheduleBatch`，因此 runner 产出 `DraftForwardInputs`
并交给 `ModelExecutor`；把它们映射为真实 `ForwardBatch` 依赖具体架构与后端，
**尚未实现**，也从未执行过任何一次前向。详见
[中英复用审计](PVD_Draft_Worker_Reuse_Audit_CN_EN.md)。

最新基础层进展（2026-09-21，第二轮）：`draft_sglang.py` 将 SGLang 自身的 draft worker
构造适配到本项目的 `DraftProvider` 契约，作为**仅预测**路径。
**1016 passed / 6 skipped**，其中本轮新增 48 项测试；11 处变异验证新增拒绝逻辑有效。
详见[中英复用审计](PVD_Draft_Worker_Reuse_Audit_CN_EN.md)。

审计结论：`StandaloneWorker.draft()` **不是**安全的复用点。一次调用会修改
`req.decode_batch_idx`、已提交采样器的 penalizer、共享缓存（`maybe_evict_swa`）、
活跃的 `req_to_token` 映射（经 `assign_draft_cache_locs`）以及四个 `batch` 字段，
而只有分配器会回滚。`StandaloneWorker` 的两个内存池都取自 target worker，
`clear_cache_pool()` 正因如此被刻意写成空操作。因此复用的是其下一层：
以 `is_draft_worker=True` 构造 `TpModelWorker`，并使用**私有**内存池
（两个池都传 `None` 时 `ModelRunner` 自行分配），同时由 `PredictionOnlyWorker`
在属性访问层面拒绝 `draft`、`draft_extend`、`verify`、`forward_batch_generation`、
`forward_target_extend`、`capture_for_decode`、`on_verify_complete_cpu`。

配置使用 PVD 自有参数（`--pvd-draft-model-path`、`--pvd-draft-revision`、
`--pvd-draft-device`、`--pvd-draft-predict-tokens`、`--pvd-draft-scratch-budget-bytes`）。
**启动时对 `speculative_algorithm` 的禁止保持原样且无条件生效**；
有回归测试断言：即使设置了 draft 参数，该禁止依然触发。

未加载任何模型、未使用 GPU；由于尚未选定目标架构，
"前缀 → 前向计算"这一步位于 `DraftRunner` 协议之后。

上一轮（2026-09-21）：`probe_search.py` 已将带清理作用域的 CPU 测试 probe
接入真实单 shard HTTP 客户端，增加请求/窗口失效保护和同版本多路线结果校验。
**967 passed / 6 skipped**，其中该轮新增 50 项测试。详见
[中英 probe/search 交接记录](PVD_Probe_Search_Foundation_CN_EN.md)。未执行真实模型、
未安装 KV、未推进时钟、未接入正式 D 服务。

- draft provider 的真实权重运行与服务接线：`draft_hf.py` 与 `draft_sglang.py` 都实现了
  `DraftProvider` 契约（含加载、词表、放置与预算校验），但都未对真实权重运行过，
  因此速度、显存与预测质量均无结论。`build_prediction_only_worker` 写出了 SGLang 的
  构造路径以供审阅，CPU 测试刻意不会走到它。
- **正式模型执行验证**。真实 tiny-Llama CPU `ModelRunner` 前向已通过，详见最新记录。
  用户选定 checkpoint、`TpModelWorker` 构造入口、CUDA/V100S、TP>1 与生产峰值内存
  仍待验证。CPU 随机权重夹具不是用户实验模型的默认选择。
- **采样**。当前为贪心选择，因为采样需要已被确立的 RNG 隔离，而这尚未确立。
- **前缀重算方案的实测**。它正确且自洽，但其每轮 O(prefix) 的开销从未与预取窗口
  对比测量过。持久缓存刻意未实现；审计文档列出了它必须先定义的内容。
- **并发执行安全性的证据**。执行串行化只是保守默认，并非测量结论。
- 在线目标模型 probe 与其他架构支持：离线 CPU/TP1 Llama 已捕获真实 post-RoPE Q，
  但不能将其并发挂接到运行中的 Decode。
- CAGRA 服务端索引生命周期和真实请求搜索。
- **D 侧在线 query。** 独立单 shard 客户端已用合成及离线真实模型 Q 调用检索路由，
  但正式 D 服务尚无真实 probe 执行或客户端接线。调用者必须提供
  `SearchRequestIdentity` 和可信 `SearchScope`，不能从 V 回复反推身份与边界。
- **检索路径的任何 GPU 执行。** 设备策略已决定并已强制执行，CPU 测试用一个声明了
  非 CPU 设备的后端覆盖了放置链路，但从未从 CUDA 池构建过索引，也从未在真实硬件上
  跨设备做过 query。相关 CUDA 测试在每一次纯 CPU 运行中都会 skip。
- **索引预算该设多大，没有任何测量依据。** `--prompt-index-budget-bytes` 已强制要求，
  但尚无针对真实模型的实测数值，运营方目前没有定量依据。预算过小时的行为是背压，
  正确但未在真实负载上验证。
- 稀疏 KV 的服务端选择、打包、D 安装及 attention。
- 实际 active/next GPU 缓冲、预取 Scheduler 接入。
- 已实现的异步首轮拉取尚待硬件验证和性能测量：网络及 ACK 等待会让出调度循环，TP 协调和 GPU 安装仍有开销。
-（已放弃，不再是待办。）直写最终页不再是目标，见第 16 节。拉取复用现有 `full_prompt` 的 staging + unpack 路径是决定的结果，不是遗漏。
- 真实模型、V100S、TP 多 GPU、RDMA、质量与性能验收。

不存在可直接启用完整预测流水线的新启动参数。
禁止把纯逻辑 `PrefetchClock` 换到现有路径，就宣称流水线已完成。

## 6. 精确的请求级时间线

定义：

- `n`：D 目标模型已正式生成的 token 数，不含 P 的首 token。
- `M`：这个请求的刷新间隔。
- `r`：提前预取步数，可配置；现有基础时钟要求 `0 <= r < M`。
- `boundary`：下一轮 KV 必须安装的正式 token 位置。
- `round`：请求自己的刷新轮次，不是 batch 轮次。

首轮在 `n=0` 完成初始化。以 `M=16, r=4` 为例：

```text
n=0：安装初始 KV
    ↓ 正式生成
n=12：小模型预测 → 目标模型 probe Q → V 搜索和传输
    ↓ D 继续正式生成 13～16，仍使用 active KV
n=16：检查 next KV；就绪则安装，否则等待/按失败策略处理
    ↓ 安装和必要确认完成，推进该请求刷新轮次
下一周期：n=28 预取，n=32 安装
```

“已发起”“已传输”“可安全使用”“已安装”必须分别表示。
提前到达不能提前切换，网络耗时不能改变按正式 token 计数的周期。
预测 token 不推进 `n`，也不允许正式 forward 越过尚未满足的刷新边界。

### 新请求等待接纳示例（目标行为）

| 请求 | 距上次刷新已生成的 D token | 本轮动作 |
| --- | ---: | --- |
| A | 10 | 不刷新 |
| B | 16 | 周期到期，安装/等待自己的 KV |
| C | 3 | 不刷新 |
| 新请求 D，尚未进入运行 batch | 未初始化 | 在 batch 外预约空间并接收首轮完整 KV |

本轮只有 B 的周期刷新与新请求 D 的首轮准备需要 KV 工作；它们不是一个共同完成屏障。
A、C 的计数、round、在途预取不变。现有运行 batch 可以因 B 到期而等待，
但不能仅因新请求 D 还未初始化而等待。D 安装完成后才可接纳。
已在运行 batch 内的到期请求异步放行仍是另一项未来优化；本次只明确增加新请求的异步首轮准备。

### 首轮等待队列拉取（已实现，需开关启用；待硬件验收）

```text
Router 选择 P、V、D
  ├─ P 计算并上传完整 Prompt KV → V：KV_STORED
  └─ D：prealloc 队列 → transfer 队列 → 最终等待队列
                         ↓ 进入 scheduler.waiting_queue 即为触发点
D 注册 staging 缓冲区并对其发布授权，发起交付请求
                         ↓ 此时 V 必须已 KV_STORED，否则 D 在此等待
V 执行获授权的 RDMA WRITE，写入该 staging 缓冲区
                         ↓ D 将其 unpack 到已预分配的最终 KV 页
                         ↓
D：原生完成/身份检查 → GPU 同步及安装 → RUNNABLE → 接纳到运行 batch
```

- 触发点是进入最终等待队列，而不是进入 prealloc 或 transfer 队列；更早的阶段不做任何 KV 传输工作。
- 由 D 发起，由 V 写入：复用现有的授权目标 WRITE 路径、写身份、epoch/generation 与 fence，不引入 RDMA READ，也不新增传输方向。
- 仍然是有接收许可的写入，不是 V 未经授权向 D 地址写入。D 不必等请求进入运行 batch 才请求传输。
- 目标是已注册的 staging 缓冲区，发布 descriptor 前先 pin；随后 D 把它 unpack 到 `DecodePreallocQueue` 已分配的 KV 页。直写这些最终页**不是**当前目标，见第 16 节。
- staging 字节计入本 worker 的 `--pvd-transfer-staging-budget-bytes`，因此首轮拉取与运行中请求的刷新争用容量。容量不足的请求留在等待队列、后续轮次重试；staging 额度用尽是背压，不是失败，绝不能因此中止排队中的新请求。
- 首版首轮使用完整 Prompt KV，不需要 bootstrap query，也不需要先完成 CAGRA 搜索。
- V 索引构建可在 KV 安全可读后并行进行；首轮完整 KV 推送不应额外依赖 INDEX_READY。
- 后续周期检索仍须索引就绪，并使用 draft → target probe → CAGRA 路径；失败策略另行明确。
- 限制同时处于拉取中的请求数量；字节受两道约束：prealloc 准入限定最终页，staging 预算限定在途接收缓冲区。
- 无法完成 prealloc 的请求根本到不了等待队列，其 KV 留在 V，不会提前占用 D 显存。
- 预算包含 staging 缓冲区、最终 KV 页和在途传输额度。拉取不能耗尽运行请求必需的显存或造成容量死锁。
- 数据到达不等于就绪：先记为 RECEIVED；必须完成原生终态、身份检查、GPU 可见性与所需 TP 一致后才是 RUNNABLE。
- 首轮已确定使用 staging + unpack；不能默认有地址就能安全安装或立即调度。
- 首轮只执行一次：入 batch 时不得重复获取 round 0；接纳前完成初始化时钟，再按正式 D token 计数。
- 取消/超时关闭后续提交并安全排空原生 WRITE；逻辑移出队列不代表 descriptor 可立即回收。
- 完整首轮需要完整 Prompt 的目标显存预算，不能宣称支持 D 从未容纳得下的 Prompt。
- 拉取期间请求位于 `scheduler.waiting_queue` 内且为 not-runnable：batch 构建必须跳过它，且不得把它计入 batch token 预算；它不是任何运行中请求的完成屏障。
- 触发点比先前的预推送方案更晚，与排队重叠的传输时间相应减少。这是为了只对已进入最终等待队列的请求占用 D 显存而接受的取舍，不承诺通信全部隐藏。

## 7. draft 和 probe 接口的设计要求

不要继续要求用户先选一个固定模型才能开发。
先实现可配置接口、fake provider 和隔离测试，再在实验时加载具体模型。

配置应能表达模型名称/本地路径、可选 revision、部署位置、dtype、预算、预测长度。
这些是接口要求，不是当前已经存在的 CLI 参数；命名与实际接入需按仓库习惯实现。

正确的数据路径：

```text
正式前缀快照
    → 独立 draft 预测 token
    → 目标模型独立 probe 计算对应位置的 Q
    → 带 layer/head/位置语义的 query
    → V 搜索目标模型 Prompt K
```

不能直接用 draft 的 Q 搜索目标模型 K，不能只发 token ID 却称作向量检索。
tokenizer 不兼容时不能把一套词表的 token ID 直接交给另一套模型。

预测与 probe 不得改变正式输出列表、正式位置、正式 KV、采样器/RNG 状态。
预测分支使用自己的临时状态与预算，结束后按生命周期清理。
目标模型 probe 使用当前稀疏 KV 得到的深层 Q 仍可能是近似值，需要记录和评估。

实际实验记录加载的模型标识及可获得的解析后 revision。
本地模型没有 revision 时记录本地来源或未知，不伪造；不为了记录而要求下载/哈希全部权重。
模型 revision、索引版本、query 版本、内存 generation 是不同概念，不能混用。

## 8. V 的搜索和交付

- 完整 Prompt KV 上传完成且 GPU 可安全读取后才构建索引。
- 区分 KV_STORED、INDEX_BUILDING、INDEX_READY、INDEX_FAILED。
- 分别管理原始 KV、检索向量、图和 ID 映射。
- 映射能定位 Entry、layer、KV-head、原始 token/page 和实际存储。
- 一个请求的不可变 Prompt 索引可供多轮 Delivery 复用。
- 不同层/head 的分数不能无定义地合并成一个全局 Top-K。
- raw dot product、cosine、page 代表向量是不同策略，不自动互换。
- 搜索返回逻辑 token/page 选择和 payload 描述，不只返回裸地址。
- gather/pack 之后才通过已授权的目标区域交付给 D。
- 搜索完成、HTTP 成功、WRITE 提交、原生传输完成、D 安装完成不是同一状态。

必须有返回字节数上限和背压。结果大于授权目标区域时不得越界写入。
批量合并可以降低控制开销，但不能改变每请求身份和周期。

## 9. D 的 sparse KV 与传输安全

只让 V 少发数据而 D 仍按完整 Prompt 读取是错误实现。
必须同步支持：

- 原始 token/page 到本地 slot 的映射。
- 正确的位置语义、有效长度、attention mask 和 payload 布局。
- Prompt 选中 KV 与本地正式生成 KV 的组合。
- 缺失页、尾页、不同 layer/head 选择和 GQA/MQA 的适配/拒绝。
- 不破坏正式生成 KV，包括 Prompt 最后一页共享空间的情况。

active 和 next 必须逻辑隔离；不可向 GPU 正在读取的位置做并发覆盖写。
只有身份、传输终态、GPU 可见性及必要 TP 协调满足后才能安装 next。

保留现有 MR/元数据一致性防护与原生句柄生命周期：

- 发布接收 descriptor 前分配并固定空间。
- 身份绑定 Entry、请求 incarnation、round、query/index 版本、rank、epoch、generation、范围。
- 应用层 generation 不会自动让 RNIC 拒绝旧 WRITE。
- 请求/预测取消不表示 RDMA 停止。
- 关闭后续提交、已提交 WRITE 终态明确、GPU 使用结束后才能复用空间。
- 无法证明安全时保留 draining/quarantine，不把超时当作释放许可。
- Entry、索引、Delivery、预测分支、buffer 分开管理引用和生命周期。

预算包括模型、draft/probe 临时 KV、V 索引构建/搜索空间、active/next、
pack/staging、生成 KV 和等待排空的资源。
Prompt 工作集有界不意味着生成 KV 不增长，仍需要正常接纳和容量限制。

## 10. 检测流程：无硬件不阻塞通用开发

### A. 本地 CPU/fake 验证

继续实现接口、协议、状态机、映射和隔离。
没有模型权重、V100S、CuPy/cuVS 时也可以推进这些工作。
但 fake 不能证明实际 GPU 或 RDMA 正确。

### B. 环境采集——默认命令

在仓库根目录运行：

```bash
python scripts/pvd/check_cagra.py
```

默认 `--mode inventory`：不导入 CuPy/cuVS、不执行 GPU kernel、不安装依赖。
输出 `status=collected, cagra_test=not_run`；退出码 0 仅表示采集完成。

### C. 有实验机器后显式实测

```bash
python scripts/pvd/check_cagra.py --mode smoke
```

默认检查 V100S 并执行真实 CAGRA build/search，失败退出非零。
实际 cuVS 版本按 API、GPU 架构、dtype 和距离度量能力验证，再记录已测环境。
不凭版本号或 import 成功就宣布支持 V100S，也不预先要求唯一版本才能开发。
这不授权删除已有 Mooncake 的版本/安全约束。

脚本默认合成 recall 阈值 0.90 仅用于小规模烟测，不是生成质量验收标准。
显式选择另一 GPU 做开发实验，不能替代 V100S 验收。

## 11. 当前验证事实

截至本交接创建前最近一轮：

- 2026-09-20，在 Windows 检出上用 Linux/WSL venv 运行 23 个 PVD CPU 测试文件：
  **837 passed, 3 skipped in 7.4s**（799 加 38 条回归测试，对应下述三个集成缺陷，
  每一个都先复现再修复）。3 条 skip 是仅限 CUDA 的设备测试，
  在纯 CPU 机器上**不构成任何证据**。十处变异全部被捕获：
  - *索引内存计费*。复现：`_build_prompt_index()` 返回的 manager `budget is None`，
    服务路径上的索引分配完全不计费；即使传入预算，也只对提取的副本计费，
    而 `BruteForceIndexBackend.build()` 的克隆保留了等量的第二份副本
    （测试分片上为 `提取=1536 后端=1536 实际保留=3072 已计费=1536 未计费=1536`）。
    修复过程中又暴露第三个问题：五轮容量吃紧就烧掉了一个 Entry 全部三次永久构建尝试，
    因为短暂压力而永久失去索引。变异：去掉启动器预算失败 2 条；后端副本不计费失败 6 条；
    把容量拒绝计为失败失败 8 条；检索仍持有张量时就退还失败 1 条；
    多次尝试共用同一个预算 owner 失败 1 条。
  - *调用方 query 身份*。复现：`search()` 把 `self.vector_space` 和记录自身的映射版本
    传给 `authorize_search`，因此一个用另一模型 K 构造、形状合法的 query 会被应答而非拒绝。
    变异：改回用 manager 自己的 vector space 授权失败 3 条；
    版本钉由索引自身补齐失败 3 条；HTTP 路由代填缺失的 `vector_space` 失败 1 条。
  - *设备一致性*。通过审阅与放置复现，而非通过崩溃：除非 `--allow-cpu-for-tests`，
    池创建在 `cuda:{local_rank}`；提取与后端都用 `.to(torch.float32)` 拷贝，而它保持设备不变；
    HTTP 路由始终构造 CPU query——因此在 GPU worker 上索引在 CUDA 而 query 在 CPU。
    变异：从 dtype 转换推断设备失败 1 条；让提取保持池的设备失败 1 条；
    两者能被捕获，仅仅因为测试使用了一个声明非 CPU 设备的后端。
  **CUDA 设备不匹配这一失败本身从未被执行过：没有可用 GPU。**
  在 CPU 上得到验证的是导致它的放置链路，以及防止它的策略。
- 2026-09-20，该次工作之前：23 个 PVD CPU 测试文件，
  799 passed in 7.2s（786 加 13 条 V 服务接线测试，走真实 aiohttp shard 路由与启动器）。
  五处变异验证有效：去掉 query 数量上限失败 1 条；去掉 `top_k` 上限失败 1 条；
  未配置索引仍提供检索失败 1 条；启动时无条件构建索引失败 1 条；接受长度不一致的 query 失败 1 条。
  其中三条最初未导致失败，因为测试只断言 400，而下游错误本来也会产生 400，
  且从未真正执行 `_build_prompt_index`；现在改为断言具体拒绝文本，
  并把启动器的索引构造提取成可测试的 helper。
- 2026-09-20 中间结果：23 个 PVD CPU 测试文件，786 passed in 7.4s（776 加 10 条回归测试，对应复查中发现并先复现再修复的三个缺陷：
  关闭时预算泄漏、构建失败时预算泄漏、非二维 query 报出无意义的 `head_dim -1`）。
  四处变异再次确认修复有效：关闭时不退还失败 3 条；构建失败时不退还失败 2 条；
  安装关闭后才完成的构建失败 1 条；重新接受非二维 query 失败 1 条。
- 2026-09-20 中间结果：23 个 PVD CPU 测试文件，776 passed in 9.0s（754 加 22 条 store/索引集成测试，驱动真实 `VectorKVStore`
  走完创建、写入、提交、构建、检索、释放）。五处变异验证有效：
  读取已开始释放的 entry 失败 1 条；对已离开 STORED 的 entry 构建失败 1 条；
  让构建失败抛出而非记录失败 2 条；释放后仍提供索引失败 1 条；跳过检索授权失败 1 条。
  entry 状态那条最初未导致失败，因为 gate 只在 STORED 之后才存在；
  补充了一条在 gate 已打开后再改变状态的测试。
- 2026-09-20 中间结果：22 个 PVD CPU 测试文件，754 passed in 6.1s（691 加 63 条 Prompt K 提取测试，含一条从真实打包缓冲区经精确索引
  回到原始 token/page 的往返测试）。六处变异验证有效：包含 padding token 失败 7 条；
  连 V 分量一起提取失败 6 条；把本地 head 下标当作全局 head 失败 4 条；
  接受不匹配的位置编码失败 1 条；对不整除的 GQA 布局取整而非拒绝失败 3 条；
  借用存储而非拷贝失败 1 条。最后一条最初未导致失败，因为默认的 float16→float32
  转换本身就会拷贝；补充了一条按存储 dtype 提取的测试，那才是会产生别名视图的情形。
  **这些测试中的 query 是从提取向量里取出的合成行，只能证明映射与身份正确，
  不能证明真实模型的检索质量。**
- 2026-09-20 中间结果：21 个 PVD CPU 测试文件，691 passed in 18.3s（633 加 58 条索引后端与选择测试）。五处变异验证有效：
  允许未命名的合并策略失败 5 条；并列排序不稳定失败 9 条；跳过映射覆盖检查失败 1 条；
  跨不同 id 映射合并失败 1 条；别名化调用方向量失败 1 条。最后一条最初未导致失败，
  因为原本的别名测试仍会选出同一个赢家；已改为断言存储下来的分数。
- 2026-09-20 中间结果：20 个 PVD CPU 测试文件，633 passed in 16.4s（异步首轮拉取后的 584，加 49 条 V 侧索引生命周期测试）。
  五处变异验证新测试有效：让交付等待 INDEX_READY 失败 5 条；KV 可读前就构建失败 1 条；
  去掉向量空间检查失败 1 条；重试次数无上限失败 1 条；把失败的构建当作 absent 失败 1 条。
- 2026-09-19 中间结果：18 个 PVD CPU 测试文件，
  552 passed in 6.2s（516 加 36 条 Hugging Face draft provider 测试）。四处变异验证有效：
  跳过词表检查失败 4 条；跳过 device/dtype 校验失败 2 条；不按预算截断失败 1 条；
  去掉越界 prefix 守卫失败 1 条。预算变异最初未导致失败，因为第一个 fake 模型遵守
  `max_new_tokens`；补充了一个忽略该参数的 fake，该守卫才真正被覆盖。
- 2026-09-19 中间结果：17 个文件，516 passed in 5.8s（510 加 6 条 staging 背压测试）。三处变异验证有效：
  忽略 staging 余量失败 3 条；一轮内不递减余量失败 2 条；把延后当作失败上报失败 2 条。
- 2026-09-19 中间结果：510 passed in 5.4s（445 加 65 条 draft/probe 接口测试）。两处变异验证有效：
  去掉向量空间检查失败 1 条；去掉 RNG fork 失败 2 条。
  第三处变异（让快照别名化 token 序列）未导致失败，因为 `CommittedPrefix` 本身就拒绝非 tuple，
  该拷贝测试与此校验重复。
- 2026-09-19 中间结果：16 个 PVD CPU 测试文件，445 passed in 6.6s（基线 377，首轮门控 +38，等待队列接线 +15，decode.py 调度钩子 +15）。
  调度钩子测试用 `ast` 从实际源码中抽出 `get_new_prebuilt_batch` 与 `_pvd_enter_waiting_queue`
  并对 fake 协作者执行，decode.py 的两处改动现已覆盖。注入三处变异验证有效：
  按下标计数而非按接纳计数失败 1 条；去掉 not-runnable 守卫失败 4 条；
  失败的拉取仍留在等待队列失败 1 条。
  所有改动文件通过 `ruff check --select E9,F401,F821,I` 与 `ruff format --check`。
  `decode.py` 在 HEAD 上已经无法通过 `I001` 与 `ruff format --check`，属既有问题，本次未引入也未修复。
- 2026-09-19 中间结果：14 个文件，415 passed in 5.75s
  （新增 `bootstrap.py` 前为 377，新增首轮门控测试 38 条）。
  注入两处变异验证新测试有效：把 RECEIVED 当作 runnable 失败 2 条；
  允许已关闭的 gate 接受迟到完成失败 1 条。
- 历史记录（本次改动前）：13 个 PVD CPU 测试文件，377 passed in 3.38s。
- 相关修改通过 Ruff format 和 E9/F401/F821/I 检查。
- 本机 inventory 成功，CAGRA 标记 not_run。
- 本机 smoke 因缺 CuPy 在 import_cupy 阶段失败，退出码 1。
- 当前开发机为 Windows / RTX 4060 Laptop，不是 V100S。
- 没有 V100S build/search 成功证据，没有实际模型和 RDMA 性能结论。

复现 CPU 回归（PowerShell，仓库根目录）：

```powershell
$pvdTestFiles = @(rg --files test/registered/disaggregation -g 'test_pvd*.py')
& .venv/Scripts/python.exe test/registered/disaggregation/run_pvd_cpu_tests.py @pvdTestFiles -q --tb=short
```

这些是历史验证结果，接手修改后必须重新执行相关测试，不能复用旧数字宣称通过。

## 12. 实施顺序及每阶段交付

### 阶段 0：核对与通用接口

- 读取实际代码和未提交 diff。
- 维护请求级时钟与设计状态机。
- 实现可配置 draft provider、目标 probe 接口和 fake 测试。
- 明确快照、query、selection、Delivery 的身份与容量协议。
- 环境采集与实机测试分开记录。

不等用户选定模型、不等本地出现 V100S 才开始通用实现。

### 阶段 1：影子预测/检索

- 增加独立首轮拉取子任务：等待队列进入作为触发点、对已注册 staging 缓冲区的目标授权、V readiness、原生传输确认、D 安装、RUNNABLE 接纳；可先用现有 full_prompt 和 fake transport 验证，不依赖 CAGRA。
- 正式生成仍用完整 KV 基线。
- 实现 draft、probe、索引、搜索的可替换接口，逐步接入真实后端。
- 选择结果暂不改变 attention。
- 测量与真实 query/精确检索的差异及附加开销。

无硬件时先完成代码与 fake 验证，真实模型运行单列待验收。
影子模式不是最终目标，不能宣称降低 D 的完整 Prompt KV 占用。

### 阶段 2：稀疏交付与 attention

- 接通逻辑选择、pack、传输、D 安装和 attention。
- 先采用同步刷新隔离正确性问题。
- 验证源/目的 K/V 对应关系和正式生成 KV 完整性。
- 实机评估质量与显存。

### 阶段 3：请求独立预取流水线

- 每请求在 M-r 时触发预测预取，在 M 时安装/等待。
- 接通真正的 active/next、传输安全与 Scheduler。
- 加入新请求、退出、EOS、取消、迟到和故障测试。
- 周期刷新保留运行 batch 的同步边界，不增加全员刷新；新请求的首轮等待发生在运行 batch 外。

### 阶段 4：端到端优化

- 基于数据调整 M、r、检索预算、并发和打包策略。
- 分析 draft/probe 与正式 D 的 GPU 竞争。
- 优化提交和进度推进，不删除安全保护。
- 不在未经确认时引入异步放行、其他搜索算法或新硬件需求。

各阶段区分“代码已实现”“CPU/fake 已验证”“真实模型/GPU 已验收”。
有硬件依赖的验收可以等待，不阻塞独立开发，但不能默认启用未验证的生产路径。

## 13. 接口最小语义

以下不是已存在的 API 或最终命名，而是实现必须保留的信息：

- 预测输入：只读正式前缀快照、请求身份、正式 token 位置、预测长度与预算。
- probe 输入：预测 token、对应正式前缀、目标模型标识与 layer/head/位置选择。
- query：明确向量空间、位置语义、版本及有效长度。
- 预取请求：Entry、round、目标安装边界、query/index 身份、返回预算、各 D rank grant。
- 首轮拉取授权：Entry/请求 incarnation、首轮 Delivery 身份、各 rank staging 区域的授权、容量和有效性/取消状态；只有请求已进入最终等待队列且页面完成注册与 pin 后才发布授权，且布局与实际容量校验通过才可提交 WRITE。
- selection：原始 token/page IDs、layer/head 范围、布局和实际字节数。
- delivery：请求身份、状态、写入身份和原生完成证明的关联。

TP ranks 需要对公共请求集合/round/错误一致，但各 rank 的局部 query、head 和地址可以不同。
重试必须幂等；重复控制请求不能造成无界重复写入。
批次成员变化不是请求级快照的自动失效条件。

## 14. 验收清单

### 正确性与生命周期

- 预测不改变正式输出、计数、RNG 和 KV。
- 新请求不影响旧请求时钟和在途预取。
- 等待队列触发和 KV_STORED 任意顺序到达都可推进；不必等进入运行 batch 才开始传输。
- 未就绪的新请求不阻塞已有运行 batch，除共享资源争用外不建立额外完成屏障。
- 未发布授权则无 WRITE；多请求排队时 staging/最终页/在途总预算有界。
- RECEIVED 不等于 RUNNABLE；接纳前完成安装，入 batch 后不重复首轮交付。
- 等待队列内取消、迟到 WRITE、租约过期/续租和授权重试不会造成提前释放或重复写入。
- 提前完成不提前安装；迟到不越过边界读取旧/半完成数据。
- 旧轮次、错 Entry、错 head、错位置、重复请求被正确处理。
- TP collective 次序及分支一致。
- 取消/超时不提前释放，尾页和生成 KV 不被损坏。
- 内存和在途任务有界。
- 原 full_prompt 基线仍可运行。

### 质量

记录任务质量、必要时困惑度、真实 query 下的召回、预取有效率和补取率。
token 预测匹配率不等于 KV 检索质量。
允许近似不意味着任何退化都自动合格；实验前请用户确认可接受指标/阈值。

### 性能

至少比较：
1. 完整 KV 同步刷新。
2. 稀疏检索、无提前预取。
3. 最近真实 Q 驱动的提前预取。
4. 独立 draft + 目标 probe 驱动的提前预取。

记录 TTFT、TPOT p50/p95/p99、吞吐、刷新等待、V 排队、搜索、pack、传输、安装、
draft/probe 开销、网络字节、无用预取、峰值显存。
测试稳定 batch 和持续加入新请求的负载；额外记录首轮排队/传输重叠时间、就绪后等待时间、
接纳时仍暴露的等待、旧请求 TPOT 变化及预备队列峰值占用。资源争用仍可能影响旧请求，不宣称零影响。
不能用少做规定刷新或隐瞒质量损失来制造加速结果。

## 15. 明确的非目标与禁止事项

- 不恢复“新请求加入，整个 batch 所有请求都 update”。
- 不引入 batch 共用 M 计数器。
- 不把预测 token 输出给用户或当成已提交生成 KV。
- 不强制标准投机采样接受/拒绝流程。
- 不写死某个 draft model 或要求预先固定 revision 才允许开发。
- 不跨请求替换 KV、不自动引入跨 V group 搜索。
- 不宣称任意 TP 布局或模型架构已经支持。
- 不默认将 D 生成 KV 回写 V。
- 不顺便产品化 Host/NVMe 分层存储、替换整个通信栈。
- 不默默使用其他算法替代 CAGRA。
- 不删除原有 MR、元数据缓存、fence 和原生句柄保护。
- 不因硬件缺失而停做无硬件依赖的任务，也不伪造硬件验收。

## 16. 尚需确认，但不应阻塞所有工作的事项

以下决策应提出方案和影响，向用户确认；同时继续不依赖该决策的工作：

- 具体架构的 probe 实现，以及预测位置如何构成下一窗口 query。
- 索引是 token/page 级、哪些层/head 单独或共享选择。
- 首轮方案已确定为最终等待队列触发的完整 KV 拉取（D 发起、V 写入、staging + unpack），不重复询问是否采用；等待队列内 not-runnable 门控与并发拉取请求之间的公平接纳策略需设计。
- **直写最终页已于 2026-09-19 决定：保留 staging。** `unpack_full_prompt_kv` 把连续的打包缓冲区按页索引散布到各层 K/V 分量，而分配器并不保证这些页连续；
  一次 RDMA WRITE 只落在一段连续地址。直写最终页要么要求整个 prompt 连续分配（约束分配器，碎片化时会失败），
  要么每个（分量 × 连续页段）一次 WRITE（成倍增加传输槽位与授权区域，改变预算模型）。两者当前都不值得。
  首轮沿用现有 staging + unpack 路径，去掉这次拷贝不是当前实现目标；没有证明该拷贝确实影响性能的实测之前不要重开此议题。
- 周期检索的索引未就绪、预测偏差和失败策略。
- 初始/最近 token 保留规则与检索容量。
- 正式质量验收阈值。

具体模型名称、revision、部署设备、M/r/预算等由用户配置，
不是要求用户现在给出固定值才能设计接口。
提供的默认值应标明依据和可配置性。

## 17. 接手先看哪些文件

以下均相对实际仓库根目录：

| 文件/目录 | 用途 |
| --- | --- |
| `python/sglang/srt/disaggregation/pvd/README.md` | 现有服务流程、支持范围与开发状态 |
| `python/sglang/srt/disaggregation/pvd/retrieval.py` | 当前 full_prompt 协议与 RefreshClock |
| `python/sglang/srt/disaggregation/pvd/prefetch.py` | 未接入服务的请求级预取逻辑 |
| `python/sglang/srt/disaggregation/pvd/bootstrap.py` | 请求级首轮拉取门控；已在 `--pvd-waiting-queue-bootstrap` 下接入 `conn.py` 与 `decode.py` |
| `python/sglang/srt/disaggregation/decode.py` | Decode 队列链：prealloc → transfer → `scheduler.waiting_queue` → 运行 batch；最终等待队列即首轮触发点 |
| `python/sglang/srt/disaggregation/pvd/decode_refresh.py` | 接收、到期筛选、等待、unpack、ACK |
| `python/sglang/srt/disaggregation/pvd/conn.py` | PVD 与 P/D runtime 接口 |
| `python/sglang/srt/disaggregation/pvd/runtime.py` | 上传与传输生命周期 |
| `python/sglang/srt/disaggregation/pvd/coordinator.py` | Entry/Delivery 协调 |
| `python/sglang/srt/disaggregation/pvd/vector_store.py` | V 存储与交付 |
| `python/sglang/srt/disaggregation/pvd/selector.py` | 当前身份查询；还不是完整向量检索接口 |
| `python/sglang/srt/disaggregation/pvd/kv_packer.py` | KV 布局和打包 |
| `scripts/pvd/check_cagra.py` | inventory/smoke 分层检测 |
| `test/registered/disaggregation/test_pvd*.py` | 当前 CPU 回归与生命周期测试 |
| `docs/superpowers/specs/2026-09-20-pvd-prefetch-design.md` | 完整设计约束 |
| `docs/superpowers/reports/2026-09-20-pvd-prefetch-foundation.md` | 基础设施变更与验证记录 |

还必须沿调用链阅读实际 Scheduler、Decode 队列、模型 attention 后端及 Router，
不能仅凭这张文件表修改推理行为。

## 18. 持续交接规范

每批修改保存：当前 commit、未提交文件、实现阶段、确认过的决策、测试命令和结果、
实际模型/库版本、未验证事项、阻断点及下一项可执行任务。

文档和旧对话不能代替代码证据；代码现状也不能悄悄覆盖用户最终目标。
不确定的实质性设计先询问；已确认的设计不要反复要求用户重新决定。

最终自检：

> 可配置独立小模型只负责预测；目标模型 probe 提供 Q；目标模型正式生成；
> V100S 目标环境中的 V 执行请求内检索；每请求独立按 M 刷新；
> 新请求不重置旧请求、不取消旧预取；本地无硬件仍可继续通用开发；
> Decode 把请求放入最终等待队列后，在已发布授权下把完整首轮 KV 拉取到该请求已预分配的页上，安装就绪才入运行 batch；
> 安全、可测地隐藏检索与传输等待，不混淆计划、实现与验收。
