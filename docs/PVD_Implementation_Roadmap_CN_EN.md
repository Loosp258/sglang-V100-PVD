# PVD 最终目标推进步骤 / Implementation roadmap

当前统一状态：[实现与验收边界](PVD_Current_Readiness_CN_EN.md)。下面保留历史推进
记录；不得从 CPU 回归推导“生产实现或全部非硬件缺口已完成”。

最新修复：[draft 分支私有请求 owner](PVD_Draft_Worker_Reuse_Audit_CN_EN.md) 避免
多个 runner handle 共用一个可变请求对象，且释放加入执行锁。仍只有一份模型
和私有池；并发所有权不代表 GPU 前向可重入。

最新修复：[draft 常驻预算](PVD_Draft_Worker_Reuse_Audit_CN_EN.md) 使用独立 owner
计费并同时检查局部/共享容量，拒绝缺失/混用预算和错误字节数，诊断历史有界。
11 个回归用例修复前失败；模型加载峰值与卸载能力没有扩大。

最新修复：回收入口绑定校验失败按 owner 隔离，其他请求继续排空，关闭先停止
全部生命周期。失败所有权保留并退避，不猜测释放；46 个定向回归通过。

最新子步：[实模自动回收](PVD_Rank_Model_Binding_CN_EN.md) 已替换正常手动 owner
清理，接入实际 Scheduler polling 方法，验证取消后自动回收、真实 allocator
槽位复用及关闭排空。严格证据升级 v5；仍不等于完整生产 Scheduler/GPU/RDMA。

最新子步：[CPU owner 回收驱动](PVD_Rank_Model_Binding_CN_EN.md) 已增加有界注册、
排空轮询、退避和关闭协议，普通 Decode 循环在暂停/空闲/结果处理后都轮询
显式绑定的 driver；待回收资源阻止 idle 泄漏检查/休眠。
14 个定向测试通过；下一步替换实模 fixture 手动清理。
这不是完整生产 sparse Scheduler 装配，不放开 GPU/RDMA 能力门禁。

更新 / Updated: 2026-09-22. 按以下顺序推进；每一步记录实现和证据，
不把接口、CPU 通过或硬件预检当成生产端到端验收。当前用户要求每阶段验证后 commit，
继续推进。当前用户已明确要求每步 commit 后推送 GitHub `pvd-disaggregation`。

### 当前权威状态 / Current authoritative status

本轮：[实际缓存释放边界](PVD_Rank_Model_Binding_CN_EN.md) 接入显式 CPU request
owner：真实结束/取消回调先挂起释放，排空后调用原 ChunkCache/分配器逻辑。
未绑定请求保持原样；清理失败保留、部分释放失败隔离。严格报告 v4 要求真实
finish、waiting abort、cache release 证据。**Scheduler 主循环的 owner 自动
注册/轮询/关闭仍未接入**，GPU/RDMA 能力不变，不能宣布 Step 4 完成。
新增 23 个用例；Windows 1763 passed / 14 skipped，WSL 1768 passed / 9 skipped。
四场景真实 CPU 模型矩阵通过；三个完整场景运行真实结束/等待队列取消与缓存释放。

本轮后续：[刷新驱动异步回收](PVD_Rank_Model_Binding_CN_EN.md) 拒绝重复并发 remove，
await 后核验同一注册对象；清理失败/取消保留注册供重试。close 先关闭准入并停止
全部请求，再异步排空，避免其他请求在清理窗口继续发起刷新。五个新增回归测试；
这是 CPU 调度接点加固，正式 Scheduler 回收与硬件 gates 仍未完成。
全量回归：Windows 1740 passed / 14 skipped；WSL 1745 passed / 9 skipped。

本轮：[请求回收与实际槽位复用](PVD_Rank_Model_Binding_CN_EN.md) 补上 CPU permit
清理与 rank 结果作用域退出之间的解绑保护。CPU 实模验收增加容量压力下的真实
slot/KV rows 复用、旧回调拒绝和重复清理检查；严格报告升级 v3。
资源回收测试不等于正式 Scheduler finish/cache-release 服务已接入。
新增 13 个测试；Windows 1735 passed / 14 skipped，WSL 1740 passed / 9 skipped。
四场景 CPU 实模矩阵通过；三个完整场景实际执行了槽位复用，部分安装失败场景
提前结束，不声称完成复用验证。

最新：[模型故障验收及迟到 RESUMED 修复](PVD_Rank_Model_Binding_CN_EN.md) 增加
丢回执、真实 CPU bank 切换后异常、结果提交前清理/排空后重试三个场景。
复现并修复生命周期和自动刷新驱动混用“本轮 ready 边界/下一边界”的卡住问题；
没有跳过 RESUMED、重置时钟或放宽硬件能力门控。严格报告 v2 按所选故障核验证据。
四场景实模矩阵已在 WSL 通过；新增 20 个测试。全量 Windows 1722 passed /
14 skipped；WSL 1727 passed / 9 skipped。

最新：[严格 rank/model 验收入口](PVD_Rank_Model_Binding_CN_EN.md) 实际启动完整
CPU 双模型分支并校验本次运行证据；缺失/旧报告、错误计数、关闭断言或未知开关
不能误报通过。56 个工具测试；严格入口已在 WSL 实际执行。硬件结论仍明确为 false。
全量 Windows 1702 passed / 14 skipped；WSL 1707 passed / 9 skipped。

最新：[rank/model/Req 绑定](PVD_Rank_Model_Binding_CN_EN.md) 已把此前独立的
rank 运行时和 CPU 模型路径连接到同一 coordinator/bank/epoch。原 Req 结果处理器
仍是唯一正式写入者。真实双 tiny Llama + HTTP 稀疏交付 + rank 控制闭环已通过：
21 次 attention 对照，最大误差约 3.58e-7；4 次交付、1600 bytes。新增 15 个测试。
这是 CPU TP1、本进程 rank 控制和 fake payload copy，不是生产 Scheduler/GPU/RDMA。
最终全量：Windows 1646 passed / 14 skipped；WSL 1651 passed / 9 skipped。

最新：[批量 wait-all 与结果处理作用域](PVD_Rank_Runtime_CN_EN.md) 为 rank 运行时
增加整批准入、独立请求时钟、执行票据和共享目标锁；结果处理结束前不允许
INSTALL。单请求失败丢弃对应行，整批 forward 失败丢弃全部行；不新增 token
写入者。实际 CPU bank 测试使用同一 coordinator/epoch 验证读者排空与切换，
但尚未连接生产 Scheduler 或原有 CPU 模型执行路径，不能称为端到端模型验收。
新增 27 个测试；完整回归 Windows 1631 passed / 14 skipped，WSL 1636 passed /
9 skipped。原有十个独立 CPU 子进程场景仍通过；本步新 bank 联动为进程内验证。

最新：[在途 forward 票据](PVD_Rank_Runtime_CN_EN.md)。rank 运行时新增单请求执行
所有权，票据在途禁止 INSTALL；取消/超时不能自动退还，执行方确认 drain 后才
决定接受或丢弃输出。13 个新单测，实际 CPU 子进程增加取消在途 forward 场景。
这是准入元数据接点，不持有 tensor/MR，不写正式 token，仍不是生产 Scheduler 接入。
最新全量：Windows 1604 passed / 14 skipped；WSL 1609 passed / 9 skipped。

最新：[rank 运行时驱动](PVD_Rank_Runtime_CN_EN.md) 已提供线程安全有界队列、owner
轮询、固定整轮截止时间、RESUMED 准入和请求级失败通知。新增 27 个单测及 5 个
真实 CPU 子进程场景；不是已接入正式 Scheduler 或原生传输。前一门控提交
`c9942b637` 已推送。GPU/MR 的释放仍需各自真实完成证明，停止通知不能替代。
最新全量：Windows 1590 passed / 14 skipped，WSL 1595 passed / 9 skipped。

最新补充：[RESUMED 准入门控](PVD_Rank_Wire_CN_EN.md)。发送 RESUME 不再等于对端
已经恢复；全部匹配回执后才允许全局执行/下一轮。更新了实际 CPU 子进程验收。
完整回归 Windows 1558 passed / 14 skipped；WSL 1563 passed / 9 skipped。

最新：[rank 消息与本地读门控](PVD_Rank_Wire_CN_EN.md) 已实现：严格有界消息、
worker epoch/通道绑定、全体 APPLIED 后才 RESUME、单 rank CPU bank 门控。
56 个新单测；全量 Windows 1549 passed / 14 skipped，WSL 1554 passed / 9 skipped。
后续已完成 2/4 个独立 CPU 进程的真实字节控制消息验收：各 rank 独立持有 bank，
覆盖首轮/刷新、reader 排空、重复/旧消息、部分换 bank 失败和进程退出。最新全量
Windows 1553 passed / 14 skipped、WSL 1558 passed / 9 skipped。
尚未连接生产 TP collective/Scheduler；这不是 GPU TP4 或跨节点 RDMA 验收。

本节覆盖下面的历史未推送/未接交付记录。此前本地 commit 已推送；本轮已逐步推送
`45c47a788`（清单与 index lease）、`46856081b`（V 稀疏 Delivery）、`de935b931`
（D-owned receive/fence/install/ACK）、`309e464be`（请求级 HTTP Delivery 闭环）。
详见[稀疏交付推进与证据](PVD_Sparse_Delivery_CN_EN.md)。

真实 CPU 双模型严格验收也已通过新的交付路径：4 次交付、1600 bytes、21 次
attention 对照，最大误差约 3.58e-7；不再使用本地 pack_source 回调。Windows 完整
suite 1436 passed / 11 skipped；WSL 1441 passed / 6 skipped。

后续新增[真实双模型 Delivery 严格验收](PVD_Sparse_Delivery_CN_EN.md)（`b74614cf0`）与
[CAGRA 分数验收修复](PVD_CAGRA_Acceptance_CN_EN.md)（`751637275`），均已推送。
再后续[异构 TP 重打包资源修复](PVD_Repack_Ownership_CN_EN.md)复现并修复 3 项原路径
缺陷，新增 5 个测试。最新完整回归：Windows 1451 passed / 11 skipped；
WSL 1456 passed / 6 skipped。该修复 `d44dd9ebf` 已重试推送成功。
后续已实现[直接稀疏打包](PVD_Sparse_Delivery_CN_EN.md)：去掉 group 临时副本，
复用最终 staging 的预算与生命周期；提供显式 CUDA copy 原语，但尚不开放 CUDA
serving。最新完整回归为 Windows 1476 passed / 14 skipped、WSL 1481 passed /
9 skipped；新增 3 个真实 CUDA 测试在本地跳过，不能当作硬件验收。

再后续：[缺失 reserve 的安全取消恢复](PVD_Sparse_Delivery_CN_EN.md)（Step 7）
为当前 V epoch 中的已知 Entry 建立有界、不淘汰的完整身份关闭标记；只有关闭
迟到 reserve/start 的入口后才允许 D 释放。已提交/UNKNOWN 传输保持原有 fence。
17 个新测试覆盖 HTTP 丢包、并发先后顺序、容量上限、重试与过期；不涉及 GPU 验收。
最新完整回归：Windows 1493 passed / 14 skipped；WSL 1498 passed / 9 skipped。

These are CPU/local-HTTP gates. Payload transfer remains a fake in-process byte
copy. Production GPU packing/attention, native Mooncake, real TP activation,
Scheduler loop/cleanup and V100S CAGRA are still implementation/validation gaps.
The final goal is **not complete**. The local lack of hardware does not block
further protocol, lifecycle, reference, fault-injection or tooling development.
Do not mistake the delivery substeps numbered in that note for completion of
the larger production gates 1–7 below.

## 固定目标 / Invariants

独立可配置 draft 只预测；目标模型独立 probe 产生 post-RoPE Q；最终 token 仍由 D
目标模型生成。Router 选择 P/V/D。首轮完整 KV 在 D 最终 waiting queue 触发交付，
安装与 ACK 后才运行。旧请求各自按 M 个 committed token 刷新；提前预测/检索/传输，
边界才切换；新请求不重置旧时钟、不取消旧预取。D 生成的 KV 常驻 D。
不启用原生 speculative generation，不擅自引入跨 V group 分片或跨层全局 Top-K。

## 执行顺序 / Ordered gates

| 步骤 / Step | 交付物 / Deliverable | 进入下一步前的验收 / Gate |
|---|---|---|
| 1. 真实 Q → V 精确检索 | 真实 Prompt K 打包存入 V；真实目标 Q 经现有 D 客户端/HTTP 搜索 | 与独立 Q·K oracle 的 token/page/score 一致；GQA、层、版本和填充正确；坏身份拒绝 |
| 2. 选择结果 → 稀疏 KV 数据契约 | 显式身份/版本、layer/KV-head/token 坐标，配对 K/V 打包与可验证载荷 | 字节与原完整 KV 对应；越界、重复、错版本、错头拒绝；不猜测合并策略 |
| 3. D 安装和 attention 消费 | 有界 current/next 工作集、绝对位置、mask、生成 KV 保留、边界切换 | 全选与完整 KV 数值一致；子集与独立稀疏参考一致；旧 forward 不读被覆盖缓冲 |
| 4. 请求级在线调度 | draft/probe/search/交付接入 Scheduler，明确共享目标执行边界 | 请求时钟独立；新请求不影响旧预取；迟到结果/取消/EOS/超时安全；未就绪按既定等待策略处理 |
| 5. 真实 GPU/Mooncake 交付 | 稀疏 payload 接现有授权、MR 生命周期、完成/ACK/fence 与预算 | GPU 目标在未 fence 前不释放/重用；TP 各 rank 一致；故障注入无旧写污染 |
| 6. V100S CAGRA | 依赖兼容验证、索引生命周期、显存预算、真实 Q 的近似检索 | 对精确基准测召回/质量/内存/搜索延迟；未验证后端不得声称支持 |
| 7. 端到端收益 | prefix 复用/流水线调优与可复现实验 | TPOT、吞吐、尾延迟、显存、刷新等待、输出质量；证明确实隐藏等待而非只增加负载 |

第 3–5 步先完成可独立 CPU 验证的契约与状态机；GPU/RDMA 的 gate 必须由真实硬件
验证，不能伪造为通过。对未确认且会改变 attention/模型语义的选择，暂停该分支询问用户，
继续其他已明确工作。CAGRA 的兼容性调研可提前，但不得为安装它擅自升级整个环境。

For every gate: reproduce failures, implement narrowly, run unit and integration
checks, update evidence and limitations, then advance. Do not infer online safety
from the offline CPU probe: its ForwardContext is process-global. Existing
full-prompt serving remains the baseline until sparse mode is explicitly gated.

## GQA 已确认语义 / Confirmed GQA semantics

同一层、同一 KV head 对应的 Q heads，各自检索后对 token 取并集并去重，
共享一份 KV 工作集。不跨层、不跨 KV heads 合并分数。并集上限必须显式提供；
超限拒绝本轮更新，不静默截断。首轮仍传输完整 Prompt，不受刷新子集上限裁剪。

Within each (layer, KV head), union and deduplicate the selections of all its
Q heads. Share that one KV working set. Require an explicit union bound; refuse
an over-capacity refresh, never silently truncate or rank scores across groups.
The initial complete Prompt is not truncated by the later refresh union bound.

## 进度 / Progress

- 基础：PVD 完整 KV 传输、V 精确后端/HTTP、离线 CPU draft 与 target probe 已有。
- Step 1：CPU/本地 HTTP 验收已通过。真实模型 Prompt K 与 post-RoPE Q，
  两个 V shard、8 组 layer/Q-head 结果、14 个拒绝检查。修复 probe 的
  `post_rope` 标签与 V 的 `rope_applied` 协议不一致的问题。
- Step 2：离线稀疏 K/V 载荷契约已通过；8 份载荷与原始 K/V 精确一致。
  后续已接源 Entry/index lease、异步授权 Delivery、D fence 与安装/ACK；
  目前为 CPU/local-HTTP/fake transport，GPU 目标与原生 RDMA 尚未验收。
- Step 3：CPU reference 已实现并验证：GQA 并集、初始完整 Prompt、current/next
  预算、边界切换、reader 释放前禁止切换、生成 KV 不变和绝对位置 attention。
  本地 HTTP 验收接入 4 组并集的 CPU 分片工作集安装。
  后续已新增显式离线 CPU backend，14 次真实 Llama 前向执行（含一次注入故障），
  全选 logits 与原后端误差为 0，子集 attention 与独立 softmax 一致。
  详见[稀疏 CPU Decode 验收](PVD_Sparse_CPU_Decode_CN_EN.md)。
  后续新增[跨 rank 安装协议与 CPU bank 驱动](PVD_Rank_Install_Contract_CN_EN.md)：
  全部 prepared/parked/applied 后才提交请求时钟，部分失败后禁止恢复读取。
  该验证是单进程 CPU，不是实际多进程 TP 或 GPU fence。
  **生产 D/GPU attention 与多 rank 原子安装仍未实现，因此第 3 步整体未完成。**
- Step 4–7：未完成；没有宣称在线稀疏 attention、CAGRA 或网络隐藏已可用。

Steps 1–2 pass their offline CPU/local-HTTP gates. Step 3 now passes a real-model
CPU attention-consumption gate as well as the working-set reference. It is an
opt-in offline adapter, not a registered production D/GPU backend.
The strict smoke uses a tiny randomly initialized Llama and fixed draft token
candidates; it is not an application-quality test or a loaded-draft-to-serving
end-to-end experiment. P→V is a local byte copy, not RDMA.

后续已接通[受控 CPU 请求刷新闭环](PVD_Controlled_Request_Loop_CN_EN.md)：从控制端
创建共同 epoch，一次 draft/probe 后按 shard 搜索，经并集/打包/安装门控推进请求。
用户已确认边界首次发起时用正式前缀目标 Q 补查；已提前发起但迟到时等待原结果。
两条路径均已通过真实 CPU Llama + 本地 HTTP 验证，补查 Q 独立 oracle 误差为 0。

后续已完成[同一真实 CPU Decode 的检索消费](PVD_Controlled_CPU_Decode_CN_EN.md)：
真实输出快照 → 查询 → 安装 → 模型消费；9 个 D token、边界 4/8、18 次 attention
检查，最大误差约 `3.58e-7`，已覆盖延迟结果、边界补查及实际前向失败。
单进程 group 视图持有所有 shard reader，整个模型前向期间不能切换 bank。

新增[CPU 生命周期接点](PVD_CPU_Decode_Lifecycle_CN_EN.md)：准入、唯一执行票据、正式
token 提交、共享目标互斥、超时/EOS/取消/drain，已驱动真实 CPU smoke。
新增[batch 执行器和真实多请求验收](PVD_CPU_Batch_Execution_CN_EN.md)：共享一次执行
lease、完整 logits、按身份提交、wait-all、请求槽位绑定；真实 batch 加入/重排/取消/
故障均通过，19 次 attention 对照最大误差约 `2.38e-7`。
新增[真实 ScheduleBatch 结果接点](PVD_CPU_Schedule_Result_Bridge_CN_EN.md)：
原结果处理器是 Req 唯一输出写入者，PVD 观察正式提交；真实请求撤回/停止条件、
错配拒绝和重放保护通过。尚非完整 Scheduler 服务或生产缓存释放验收。
新增[独立真实 draft CPU 闭环](PVD_Real_Draft_CPU_Loop_CN_EN.md)：真实较小 ModelRunner
预测 → target Q → V HTTP → 稀疏安装 → target Decode → 原 Req 提交；私有池、
目标状态/RNG 不变已验证，仍为随机 toy 模型而非质量/生产加载/性能证据。
新增[请求级 CPU 自动刷新驱动](PVD_CPU_Refresh_Driver_CN_EN.md)：由正式 token 时钟
决定 capture/HTTP/边界安装与实际 Q fallback；真实双模型闭环使用该驱动通过。
Next: the production GPU sparse-attention / authorized delivery / TP activation
gate. The production Scheduler event loop and resource cleanup remain unwired;
CPU callbacks and local rank mirrors cannot be substituted for GPU/MR fences.
The CPU driver is still a standalone smoke, not Scheduler or GPU/TP evidence.
CPU banks are not GPU fences or production allocators. No new request may reset
an old request's clock. See the linked note for current limits.

真实 CPU Decode 消费 `8f482e631`、此前协议 `3c4b0c479`、CPU 生命周期接点
`81bfb64b4`、batch 执行器 `1553d036c` 已随历史本地 commit 一并推送。

## 当前接续点 / Current handoff gate (2026-09-21)

以下为先前阶段的历史记录（现已推送；最新状态见文首）：

| Commit | 完成的阶段 / Completed stage |
|---|---|
| `1553d036c` | CPU batch dispatch / real multi-request execution |
| `bef8a59c4` | Actual Req/ScheduleBatch result observation, one authoritative output writer |
| `92ae30c23` | Real independent draft closed loop; backing-KV storage isolation fix |
| `1e3c8ab83` | Request-local CPU refresh driver, boundary installation / actual-Q fallback |

该历史阶段完整 suite：Windows 1373 passed / 11 skipped；WSL 1378 passed / 6 skipped。
全部严格 CPU 组合验收通过，包括真实双模型、V 本地 HTTP、稀疏 attention 和正式 Req
结果处理。**最终生产目标尚未完成。** 对真实 TP/GPU/RDMA/CAGRA 的未验证项目，仍有
实际实现缺口，不能只运行一个 benchmark 就宣布完成。

只读环境核对：WSL 可通过 nvidia-smi 看到 RTX 4060 Laptop（8188 MiB），不是 V100S。
现用 `/home/loosp/torch311-env` 是 `torch 2.14.0+cpu`，`torch.version.cuda=None`，
`torch.cuda.is_available()=False`；`/sys/class/infiniband` 无设备；未装 Mooncake、CuPy、
cuVS。`check_cagra.py --mode inventory` 结果为 collected / not_run，绝不是 smoke pass。
本轮没有为推进而更改依赖或把 CPU backend 注册成生产 GPU backend。

下一阶段顺序：

1. 明确可用 Linux GPU 实验环境（目标验收为 V100S + RDMA 节点）；使用隔离环境，
   获取已有软件栈与模型/tokenizer 路径。模型保持用户可配置，不预设 checkpoint。
   本地 RTX 可用于部分 CUDA 开发，但不能替代 V100S 或多节点 RDMA 验收。
2. 实现并逐层数值验证 GPU sparse attention/current-next bank/stream 生命周期。
   先全选对照原 attention，再子集对照独立参考，再故障/取消与读写重叠验证。
3. 稀疏 payload 接入既有 Entry/source lease、Delivery 授权、目标 MR 注册与原生
   submit/poll/fence/ACK；不得在尚有远端 WRITE 时释放/重用目标缓冲。
4. 实际 TP 多 rank 激活与 Scheduler 队列/资源释放接通，替换目前单进程镜像与
   测试 callback；初始完整 Prompt 的真实交付/ACK 必须保留。
5. 在目标软件栈验证真实 CAGRA，接入 IndexBackend 的预算与生命周期，测真实目标 Q
   的 recall/质量；最后测 TPOT/吞吐/尾延迟/显存/刷新等待，验证是否隐藏通信。

Real GPU/RDMA integration needs an accessible experimental environment and
model/tokenizer configuration. The local CPU suite cannot establish native
write completion, stream safety, distributed installation or CAGRA support.
Do not enable production sparse mode or declare the final goal achieved from
the above CPU passes. Preserve the existing full-Prompt serving default.
