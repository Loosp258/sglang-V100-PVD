# D 四 head 批量安装：本地实现与真实 KV 回放

2026-10-04；`codex/pvd-oasiskv`；实现提交 `d6ac2fb61`。
CloudLab 租约已到期，用户确认当前没有 GPU。本轮没有连接旧节点；新选项默认关闭。

## 修改与目标

在 D 接收到 V 的稀疏 KV 并保存进 CPU 历史缓存后，将四个 KV head 的安装合并：

- 驻留交集用两个全 head gather、两个 scatter 搬到新 bank；避免逐 head 重复提交。
- 不在当前 GPU bank 的选中 token，仍从已验证的 CPU 缓存读取；把原来最多四次
  KV H2D 合为一次，再用两次 scatter 安装 K 和 V。
- 保持原候选集合、token 顺序、padding/mask、capacity32、max_new16、Top4、
  Q 数量、逐层流水线、V/CAGRA 和快建图参数；不扩展为完整块，也不多传 KV。
- 不移除 CPU 历史缓存。曾收到但已不驻留的 token 仍会有本地 H2D，且没有新的
  V→D payload；两类字节分别记录。
- 显式聚合的 int64 索引元数据另外计数。相同 KV 字节不等于已经证明相同索引
  传输或相同 CUDA kernel 数量；原 advanced indexing 的隐式元数据未测量。

配置 `batched_bank_install=false` 为默认。配置为 true 时，拒绝与 GPU backup、
staged transport 或 attention workspace 混用；超过请求 scratch 的容量也拒绝。
capacity32 的保守张量上界是每任务268416B、两 worker536832B，覆盖输出、mask、
普通及 pinned host 缓冲、GPU miss/hit 缓冲及索引；allocator reserved memory
和既有 CPU cache/resident bank 不属于这个临时张量计数。既有请求预算仍包含它们。

每个临时所有者在后续操作前进入 caller 的 retained 列表。沿用 worker stream、
完成事件和同步证明；复制失败且无法证明 CUDA drain 时仍进入原 quarantine，
保留源、目的及请求预算。安装时间记录到 completion fence 成功，包含 H2D、
gather/scatter 和同步等待，不等于 V 搜索时间。

## 已完成验证

最终 CPU gate：**133 passed, 14 skipped, 1 warning in 6.94s**。
Python3.14.5、Torch2.14.0+cpu。包 bootstrap 仅绕过 SGLang 前端的 Linux 初始化；
Torch 与 PVD 实现未被 runner 替换。现有 HTTP/生命周期测试中有明确的 CPU policy
和 transport doubles，不构成 native RDMA 证据。warning 是既有 asyncio_mode 配置。

覆盖跨 head 扁平索引、不同 bank 宽度、候选重排、空 head、全命中、全 miss、
忽略 resident padding、无 CPU 数据时的 GPU resident 语义、后续 head 非法输入
在分配前拒绝、四个 scatter 位置的部分失败所有者保留、预算拒绝和真实 profile gate。
14个真实 CUDA 案例跳过：4个批量安装/nondefault-stream、4个部分失败、6个既有
稀疏 packing 案例。CPU 成功不能代替这些 CUDA 条件。

用两份保存的真实 Qwen Decode 轨迹执行原逐 head 安装与实际批量 helper。
每份2159-token Prompt、16输出、15个已消费 forward；bootstrap 为首个28层 bank，
稳态为后14步392层 bank。缓存仅由捕获到的真实行构造，并逐次验证 Prompt KV 不变。
两个模式在**840个不同 bank**上，K/V、mask 和 ID 顺序与捕获值逐位一致。
没有运行新的目标 forward、检索、质量 benchmark 或最后未消费的 prefetch。

| 稳态指标，原安装→批量安装 | case99401 | case99402 |
|---|---:|---:|
| 已消费 bank | 392→392 | 392→392 |
| GPU resident 行 | 35254→35254 | 35476→35476 |
| 本地 CPU→GPU KV 行 | 4145→4145 | 4123→4123 |
| 本地 KV H2D bytes | 2122240→2122240 | 2110976→2110976 |
| KV H2D 调用 | 1189→387 | 1183→388 |
| resident gather 调用 | 3136→784 | 3136→784 |
| resident scatter 调用 | 3136→784 | 3136→784 |
| CPU miss scatter 调用 | 2378→774 | 2366→776 |

首轮28个 bootstrap bank，每份 KV H2D 调用112→28，miss scatter224→56，
各自610816B／616960B保持一致。全请求 V→D 稀疏 payload 为2564096B／2560512B，
与之前的连续 packing 回放相同。以上调用计数来自执行路径与真实 IDs，
描述 Torch 层操作机会；**不能换算为 GPU 时间、D wait 或 TPOT 提升比例**。

## 性能范围与下一步

此前 `pvd_oasis_stages_cloudlab_20261003.md` 的另一组配置测到平均安装5.119ms/层，
说明这个阶段值得测量；与本轮代码、并发配置不同，不能直接减进已有端到端时间。
本修改也不消除检索 RPC、注册、网络交付和队列等待，不保证覆盖整个 KV 等待窗口。

新增 `d-batch-install` ABBA 入口，尚未运行。基线和优化均用同一源码、模型、请求、
warmup、两 worker、快图和预算，只切换该选项；V packing 的实际 health 必须为
Torch，排序、直送、GPU backup、stages/workspace 均固定关闭。检查实际 fenced
安装 profile、source hash、各层行数与字节及输出一致，并报告 D wait、TPOT、
队列、安装与清理。P→D 初始直送关闭，初始完整 KV 交付与 Prompt pass 仍计入时间。
恢复新 GPU 资源及 checkout 配置后，先完成 CUDA/native gate，再执行该对照。

同名 JSON 保存 CPU gate、轨迹哈希、计数与源码证明；完整原始日志、哈希及回放结果
位于 `artifacts/d_batch_install_20261004/evidence.tar.gz`。记录源码与实现提交的
Git blob 采用 LF 归一化核对。计划、实现、验证报告分步本地提交，未 push。

复现（输出必须新建并位于本 worktree artifacts）：

```powershell
& C:/Python314/python.exe -X utf8 -B benchmark/run_pvd_cpu_gate.py --output artifacts/d_batch_fresh/gate test/registered/disaggregation/test_pvd_oasis_bank_install.py test/registered/disaggregation/test_pvd_oasis_transport_io.py test/registered/disaggregation/test_pvd_oasis_serving.py test/registered/disaggregation/test_pvd_oasis_pipeline.py test/registered/disaggregation/test_pvd_oasis_request.py test/registered/disaggregation/test_pvd_contiguous_sparse.py test/registered/disaggregation/test_pvd_sparse_copy.py
& C:/Python314/python.exe -X utf8 -B benchmark/verify_pvd_batched_real_banks.py --capture artifacts/oasis_workspace_replay01/capture --output artifacts/d_batch_fresh/real_banks.json
```
