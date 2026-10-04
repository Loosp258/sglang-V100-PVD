# D CPU 历史缓存批量安装：实现、真实 KV 回放及本机计时

2026-10-04；`codex/pvd-oasiskv`；实现提交 `1b581757d`。
CloudLab 租约到期，用户确认目前没有 GPU；没有连接旧节点。选项默认关闭。

## 修改

在既有 receive ordering、整包 D2H 和 stream 同步完成后，把每个 head 的稀疏
payload `[2,N,128]` 转为 `[N,2,128]` 的 view，批量写进既有 CPU 历史缓存。
每组一次 `index_copy_` 写 KV、一次 `index_fill_` 更新 valid；替代逐 token 的
clone、复制、valid 写入。没有额外 KV clone 或完整 Prompt 副本。

所有组在第一次写入前验证：目标 dtype/shape/device/extent、token 范围、重复、
源/缓存别名、同一 layer 和 head 唯一性。valid 在本组同步 KV 写入成功后才更新；
部分失败不足以证明接收完成，ACK 仍要求整个 record 的 installed 状态。
GPU 接收区、pinned D2H 缓冲、source views 及元数据沿用 caller owner 保留；
D2H 未知保留注册、接收区和预算，禁止 ACK/复用。没有减少 ordering fence。

开关是 D 配置 `batched_cache_install=false`。试验拒绝与 batched-bank、GPU backup、
stages 或 attention workspace 混用，避免一次修改多个阶段。四 head、capacity32
的额外索引与有效性检查张量上界1156B/任务、两 worker2312B，在原 request scratch
预算内检查；它不包含既有 cache、GPU receive 或 D2H 分配，也不代表 allocator
reserved memory。历史缓存、resident bank 与 native 字节预算仍由原路径计入。

V/CAGRA、快建图、Q 数量、Top4、capacity32、max_new16、候选排序、GPU bank
安装、逐层配对 Decode 和初始完整 P→V→D 路径一致。P→D 初始直送关闭。

## CPU 与生命周期验证

最终：**173 passed, 20 skipped, 1 warning in 5.25s**。
Python3.14.5、Torch2.14.0+cpu；package bootstrap 仅绕过 Linux 前端初始化。
CPU policy/transport doubles 用于 HTTP、ordering 及 UNKNOWN 单元测试，
不是真实 RDMA。20项 CUDA 跳过包含6个本次 D2H/nondefault-stream 案例、
8个既有批量 bank 安装/部分失败、6个既有 sparse packing。真实 native gate 未做。

覆盖不同源顺序、跨 head、cache 内容与 mask、无 clone、覆盖源后缓存仍独立、
后续组无效时不改第一组、部分复制不发布 valid、失败 ACK 拒绝、D2H UNKNOWN
保留目的地/预算、模式与预算拒绝。也验证原 CPU receiver 的小 shape、多 dtype
合同继续可用。warning 是既有 asyncio_mode 配置缺少本地插件。

首轮 gate 的 ACK 错误文本断言写成 `installed`，实际正确拒绝消息是
`all ranks must install before delivery ACK`；修正测试断言后通过。
失败日志留在 gate01，未计入最终通过数。

## 保存的真实 KV 回放

两份2159-token Prompt、16输出、15个已消费 forward 的真实 Qwen KV 轨迹。
按相同单调缓存 miss 构造真实 manifest/wire，执行 actual CPU receive record
的原方法与批量方法。readiness 是明确的本地 fixture，不制造 native terminal
证据；源来自捕获的已知选中行，不是 V 原 Entry 或真实 PUT。
每次安装后覆盖 wire，再从 cache 安装 bank，与捕获值检查。

两模式在**840个不同逐层 bank**上 K/V、顺序和 mask 逐位相同，复用的 Prompt
行不变；wire+manifest 哈希、delivery/head-group 数及本地 KV H2D 预算相同。
最后未消费的 prefetch 不计入；没有目标 forward、CAGRA 搜索或新质量 benchmark。

| 稳态指标：原缓存安装→批量安装 | case99401 | case99402 |
|---|---:|---:|
| 已消费 bank | 392→392 | 392→392 |
| remote-rank 交付 | 714→714 | 716→716 |
| 缓存安装 KV 行 | 3815→3815 | 3796→3796 |
| V→D payload bytes | 1953280→1953280 | 1943552→1943552 |
| 逐 token KV clone | 3815→0 | 3796→0 |
| KV 复制调用 | 3815→1170 | 3796→1169 |
| valid 写入调用 | 3815→1170 | 3796→1169 |
| 新增 CPU 索引 bytes，累积 | 0→30520 | 0→30368 |

首轮每份56次交付、112个 head group：clone1193/1205→0，KV复制及valid写入
1193/1205→112；payload610816B／616960B保持一致。全请求稀疏 payload 为
2564096B／2560512B；本地非 resident KV H2D bytes 为2733056B／2727936B，
均与原路径相同。索引是 CPU 本地元数据，没有加入网络/GPU KV 传输。
上述调用次数描述 Torch 操作，不是 CUDA kernel 数或端到端收益比例。

## 本机 CPU 安装阶段计时

同一批真实 miss 包、缓存容量、数据与代码，先对两模式各 warm 一遍，
再运行五轮 base/opt/opt/base。准备 payload、构造 record 及清零 validity 不计时；
每包计实际 `OasisCPUReceiveRecord.copy_to_cache()`，两模式都有相同的完成 profile。
这是可重复的孤立 CPU 安装测试，不包括 D2H、GPUDirect、GPU/网络竞争或排队。

| 各轮 mean 的均值，µs／rank 交付：原→批量 | case99401 | case99402 |
|---|---:|---:|
| bootstrap | 808.8→158.5 | 665.0→148.5 |
| steady | 210.9→113.3 | 205.5→108.5 |

稳态本机 CPU 部分每交付约节省0.098ms，不能直接换算成每 token 或 CloudLab
等待改善。历史端到端 KV wait 还包含逐层 RPC、注册、D2H、GPU 竞争和队列。
新代码在 GPU 上仍可能表现不同；本轮没有 D wait、TPOT 或完整生成质量结论。

## 后续与证据

准备 `d-cache-install` 公平 ABBA 入口，尚未运行。只切换缓存安装开关；两 worker、
per-job HTTP、原 per-delivery 注册、Torch packing、原 per-head GPU bank 安装
均固定。验证实际 D2H fence、installed profile、clone/copy/valid 次数、源哈希、
V 两 rank health、各层字节和输出；报告 cache-copy time、D wait、TPOT 与清理。
恢复新 GPU 及 checkout 配置后先验证真实 CUDA/native，再测完整路径。

同名 JSON 保存 gate、轨迹 SHA-256、操作数及 CPU 计时汇总。
原始 gate01/02/03 日志、回放、全部 ABBA CPU 轮次、计时脚本和源码证明归档在
`artifacts/d_cache_install_20261004/evidence.tar.gz`；源码与实现提交的 Git blob
以 LF 归一化核对。所有临时输出在 worktree artifacts；分步本地提交，未 push。

复现（使用新的 artifacts 输出路径）：

```powershell
& C:/Python314/python.exe -X utf8 -B benchmark/run_pvd_cpu_gate.py --output artifacts/d_cache_fresh/gate test/registered/disaggregation/test_pvd_oasis_cache_install.py test/registered/disaggregation/test_pvd_oasis_receive_slot_records.py test/registered/disaggregation/test_pvd_oasis_transport_io.py test/registered/disaggregation/test_pvd_oasis_serving.py test/registered/disaggregation/test_pvd_oasis_bank_install.py test/registered/disaggregation/test_pvd_oasis_pipeline.py test/registered/disaggregation/test_pvd_oasis_request.py test/registered/disaggregation/test_pvd_contiguous_sparse.py test/registered/disaggregation/test_pvd_sparse_copy.py
& C:/Python314/python.exe -X utf8 -B benchmark/verify_pvd_cache_real_banks.py --capture artifacts/oasis_workspace_replay01/capture --output artifacts/d_cache_fresh/real_banks.json
```
