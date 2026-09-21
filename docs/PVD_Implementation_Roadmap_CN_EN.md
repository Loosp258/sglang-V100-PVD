# PVD 最终目标推进步骤 / Implementation roadmap

更新 / Updated: 2026-09-21. 按以下顺序推进；每一步记录实现和证据，
不把接口、CPU 通过或硬件预检当成生产端到端验收。未明确要求时不自动 commit/push。

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
  尚未接入异步交付、源 Entry lease、GPU 目标授权和 MR fence。
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

Next: define/test Scheduler-owned dispatch, completion, install and abort
lifecycles, including target-execution arbitration, real committed counts,
EOS/cancel/timeout and batch changes; then wire supported online execution.
The CPU driver is still a standalone smoke, not Scheduler or GPU/TP evidence.
CPU banks are not GPU fences or production allocators. No new request may reset
an old request's clock. See the linked note for current limits.

安装协议、受控闭环与边界补查已提交为 `3c4b0c479`，未推送。
后续真实 CPU Decode 消费接线尚未提交。
