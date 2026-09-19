# PVD 预测检索预取流水线：AI 开发交接文档

更新日期：2026-09-20。

本文供另一位 AI 在没有历史对话的情况下接手。请先完整阅读，再查看代码。
本文整合用户当前要求，不需要通过历史对话猜测设计。
如用户后续给出新要求，以用户最新明确要求为准，并同步维护本文。

## 1. 一句话目标

在现有 SGLang PVD 框架中，使用用户可配置的独立小模型预测未来 token，
再由目标模型的独立 probe 分支生成检索 Q；V 提前执行 CAGRA 搜索并发送相关 Prompt KV，
使检索与网络传输尽可能和 D 的正式 Decode 重叠。

预测 token 不作为正式输出。每个请求独立按 M 个正式 Decode token 刷新。
新请求加入不触发旧请求额外更新，不重置旧请求时钟，不取消旧请求在途预取。

新请求的首轮采用最终等待队列触发的拉取：D 等到调度器把请求放入最终等待队列
（`scheduler.waiting_queue`）后，才发起完整 Prompt KV 的交付，目标是该请求已预分配的最终 KV 页。
传输仍由 V 执行获授权的 RDMA WRITE；“拉取”指由 D 发起，不是改变传输方向，也不是 RDMA READ。
请求在等待队列中标记为 not-runnable，安装校验通过后才进入运行 batch。该方案已确认，尚未实现。

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
| 首轮初始化 | D 等请求进入最终等待队列后发起拉取，直接写入已预分配的最终 KV 页；V 执行获授权的 WRITE；请求在等待队列内为 not-runnable，安装完成后才入运行 batch |
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
   - `decode.py` 在 `waiting_queue.extend(transferred_reqs)` 之后触发拉取；
     batch 构建改为统计已接纳请求数而非队列下标，跳过 not-runnable 请求且不占用 batch 名额。
     关闭该开关时不会跳过任何请求，计数与原先的下标比较完全一致。
   - 拉取本身仍是现有的同步 `full_prompt` refresher。
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
6. CPU 测试：时钟、首轮门控、等待队列触发、decode.py 调度钩子、draft/probe 接口、
   新请求隔离、现有 refresher 选择范围、检测脚本行为。
7. 完整目标和阶段记录文档。

### 5.3 尚未实现

- 自定义 draft 模型的真实加载配置与 provider。
- 目标模型 probe、预测前缀重对齐、Q 捕获。
- CAGRA 服务端索引生命周期和真实请求搜索。
- 稀疏 KV 的服务端选择、打包、D 安装及 attention。
- 实际 active/next GPU 缓冲、预取 Scheduler 接入。
- 异步首轮拉取。等待队列触发、门控与接纳已接入（见 5.2），但拉取本身仍在调度线程上同步执行，只是把等待挪了位置，尚未隐藏。重叠属于预取流水线工作。
- 直接把已预分配的最终页注册/pin 为 RDMA 目标。当前拉取复用现有 `full_prompt` 刷新路径（staging + unpack）；直写最终页是既定目标但尚未实现。
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

### 首轮等待队列拉取（已确认，待实现）

```text
Router 选择 P、V、D
  ├─ P 计算并上传完整 Prompt KV → V：KV_STORED
  └─ D：prealloc 队列 → transfer 队列 → 最终等待队列
                         ↓ 进入 scheduler.waiting_queue 即为触发点
D 对已预分配的最终 KV 页发布授权，并发起交付请求
                         ↓ 此时 V 必须已 KV_STORED，否则 D 在此等待
V 执行获授权的 RDMA WRITE，写入这些最终页
                         ↓
D：原生完成/身份检查 → GPU 同步及安装 → RUNNABLE → 接纳到运行 batch
```

- 触发点是进入最终等待队列，而不是进入 prealloc 或 transfer 队列；更早的阶段不做任何 KV 传输工作。
- 由 D 发起，由 V 写入：复用现有的授权目标 WRITE 路径、写身份、epoch/generation 与 fence，不引入 RDMA READ，也不新增传输方向。
- 仍然是有接收许可的写入，不是 V 未经授权向 D 地址写入。D 不必等请求进入运行 batch 才请求传输。
- 目标是该请求已预分配的最终 KV 页，发布 descriptor 前先注册并 pin。首轮路径没有 staging 拷贝，因此也不需要单独的预备字节额度：`DecodePreallocQueue` 的准入已经限定了这部分显存。
- 首版首轮使用完整 Prompt KV，不需要 bootstrap query，也不需要先完成 CAGRA 搜索。
- V 索引构建可在 KV 安全可读后并行进行；首轮完整 KV 推送不应额外依赖 INDEX_READY。
- 后续周期检索仍须索引就绪，并使用 draft → target probe → CAGRA 路径；失败策略另行明确。
- 限制同时处于拉取中的请求数量；字节上限由 prealloc 准入继承，因为目标是最终页而不是额外的 staging 池。
- 无法完成 prealloc 的请求根本到不了等待队列，其 KV 留在 V，不会提前占用 D 显存。
- 预算包含最终 KV 页和在途传输额度。拉取不能耗尽运行请求必需的显存或造成容量死锁。
- 数据到达不等于就绪：先记为 RECEIVED；必须完成原生终态、身份检查、GPU 可见性与所需 TP 一致后才是 RUNNABLE。
- 首轮已确定直接写最终 KV 页；不能默认有地址就能安全安装或立即调度。
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

- 2026-09-19，在 Windows 检出上用 Linux/WSL venv 运行 17 个 PVD CPU 测试文件：
  510 passed in 5.4s（445 加 65 条 draft/probe 接口测试）。两处变异验证有效：
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

- 增加独立首轮拉取子任务：等待队列进入作为触发点、对已预分配最终页的目标授权、V readiness、原生传输确认、D 安装、RUNNABLE 接纳；可先用现有 full_prompt 和 fake transport 验证，不依赖 CAGRA。
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
- 首轮拉取授权：Entry/请求 incarnation、首轮 Delivery 身份、各 rank 最终页的授权区域、容量和有效性/取消状态；只有请求已进入最终等待队列且页面完成注册与 pin 后才发布授权，且布局与实际容量校验通过才可提交 WRITE。
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
- 未发布授权则无 WRITE；多请求排队时最终页/在途总预算有界。
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
- 首轮方案已确定为最终等待队列触发的完整 KV 拉取（D 发起、V 写入、直写已预分配最终页），不重复询问是否采用；等待队列内 not-runnable 门控与并发拉取请求之间的公平接纳策略需设计。
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
