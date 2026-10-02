# Oasis 请求级 HTTP 复用：完整路径小规模对照

CloudLab 2026-10-02，`codex/pvd-oasiskv`。计划 `b9ef29c96`，实现 `955d3199a`，
单变量基线校正 `d42c479ce`。正式 tag `oasis_io_reuse_abba01`。

## 结论

本轮客户端中位数 **10.096→9.809 s，缩短 2.85%**；D 每步累计 KV 等待 **419.218→409.987 ms，缩短 2.20%**。
收益仍是毫秒级的局部减少，尚未降低等待的时间量级。两个 worker 占用仍约 99.8%。
两条 Prompt、每配置四次请求的小样本存在顺序漂移，不能宣布稳定生产收益；`reuse_io` 继续默认关闭。
下一项稀疏打包对照固定 `reuse_io=false`，分别测量，避免把本轮小幅收益叠加成更大的宣称。

## 公平性与实际复用

ABBA：base_a→opt_a→opt_b→base_b；每臂重启全部角色，排除两次相同 warmup，
两条 2159-token Prompt、greedy、ignore_eos、16 输出 tokens。全部八次实际输出 IDs 和文本一致，cached_tokens=0。
P=node0 GPU0，V=node1 GPU0+1，D=node2 GPU1；同一 Qwen2.5-7B＋专用 EAGLE3，TP1，Oasis 配对逐层 Decode。
初始 KV 仍经图门 P→V→D，private Prompt seed 计入客户端；没有 P→D 直送。
V 固定已验证的 pooled host-candidate 检索，GPU finite-Q proof；两臂都关闭没有在线收益的 CPU Q 校验。
四合一快图、degree16=KNN14+ring2、prefix2048+tail111、itopk2048、Top4、capacity32、max_new16、workers2均相同。
唯一切换为 D JSON `reuse_io`；没有减少 Q、候选或预算。每请求 420 层 job、840 双 rank 搜索 RPC。
每模式 3136 稳态 V profiles/D搜索RPC，均 two items/14 Q rows，完整计数匹配，无回退/重试。

| 每请求资源计数 | 原方式 | 请求级复用 |
|---|---:|---:|
| native worker job / worker loop | 420 / 420 | 420 / 420 |
| search client / HTTP session | 840 / 840 | 2 / 2 |
| control client / HTTP session | 840 / 770或772 | 2 / 2 |

计数来自实际创建对象，不代表实测 TCP 连接数。control 原来已在单层 reserve/start/poll/ack 内复用；
现在两 rank 各一组客户端驻留于现有 manager I/O loop，跨层复用。每 job 的线程相关接收 registry、stream 与 native proof 仍独立。
所有正式请求 trace 均 closed=true；优化臂共享 close future 已提交并完成。

## 各阶段时间

| 指标（独立中位数） | 原方式 | 请求级复用 | 变化 |
|---|---:|---:|---:|
| V batch 每 rank/层 | 7.437 ms | 7.676 ms | +3.21% |
| D search_many（含编码/HTTP/验证） | 12.042 ms | 11.509 ms | -4.43% |
| D search HTTP范围 | 11.613 ms | 11.073 ms | -4.65% |
| D 整层检索/交付RPC范围 | 34.591 ms | 33.904 ms | -1.99% |
| 层 worker 完整 service | 38.041 ms | 37.475 ms | -1.49% |
| 每步累计 KV 等待 | 419.218 ms | 409.988 ms | -2.20% |
| 后续 Decode forward | 521.159 ms | 502.395 ms | -3.60% |
| 稳态 draft | 3.909 ms | 6.198 ms | +58.58% |
| 首个客户端事件 | 2.588 s | 2.579 s | -0.36% |
| 客户端完成 | 10.096 s | 9.809 s | -2.85% |

中位数不能相加重建请求账单。V没有修改，V时间变化属于这轮测量环境；
D HTTP范围也含 V 排队与处理，不能把差值全部解释为TCP握手。D整层RPC包含KV交付和安装，不能叫纯检索时间。
worker service 还包括 finally 中的逐 job 清理；service-minus-RPC 包含缓存安装、CPU/GPU 复制、stream 和 session 关闭，不是纯 D 计算。
请求共享 HTTP clients 的最终退休可能晚于最后一个流事件；客户端完成时间不保证包含最终 HTTP close，安全退休在 trace 和退出 gate 中另行核对。

| 执行顺序 | case99401 | case99402 |
|---|---:|---:|
| base_a | 10.140s | 9.801s |
| opt_a | 9.802s | 9.816s |
| opt_b | 9.862s | 9.514s |
| base_b | 10.053s | 10.466s |

初始逻辑全KV均 123805696 B/request；稀疏逻辑payload中位数 2562304/2563072 B。
稀疏流量未冻结，原生 CAGRA 候选可以抖动；输出一致不等于已证明任意 Prompt 的召回/质量等价。

## 生命周期和验证

本地真实 aiohttp、多线程与独立 background loop **16 passed**；CloudLab 完整相关回归 **314 passed, 2 skipped**。
两个 skipped 为独立 native opt-in 测试，本轮完整双 rank GPU/RDMA serving 另实际完成。
本地最初 rail fixture 缺字段，以及旧 pytest 临时目录权限错误，原始历史/完整成功日志一同保留。
单元 gate 覆盖多 worker 复用、严格 JSON bool、I/O loop 归属、取消/关闭与未知完成保留。
lookahead join、worker native 完成和 receive/cache drain 后才能关闭共享 clients；
close future 只提交一次，超时/失败/未知结果保留 clients、cache 和 request 预算。
现有 serving owner 一旦记录 retirement error 不自动解除，即使 future 后来完成，仍保留至隔离进程退出。
长 Decode、多并发、TP2、GPU/RDMA 真实故障注入及完整峰值内存尚未验证。

## 原始证据与复现

同名目录保留 raw.tar.gz（全部四臂命令、日志、warmup与正式输出、source hashes）、
local.tar.gz（16项测试及失败历史）、native.tar.gz（314项回归）、精简 summary、RPC/V 分解和便携 manifest。
所有所属服务停止：owned={}、cleanup_errors=[]，三节点六 GPU 均 0 MiB。

```text
wsl -d Ubuntu -- python3 /mnt/d/code/sglang-V100-PVD-oasiskv/benchmark/run_pvd_oasis_serving_cloudlab.py --tag fresh_io --comparison v-io --arms base_a,opt_a,opt_b,base_b --cases 99401,99402 --tokens 16
wsl -d Ubuntu -- python3 /mnt/d/code/sglang-V100-PVD-oasiskv/benchmark/analyze_pvd_oasis_serving.py /mnt/d/code/sglang-V100-PVD-oasiskv/artifacts/fresh_io --comparison v-io
wsl -d Ubuntu -- python3 /mnt/d/code/sglang-V100-PVD-oasiskv/benchmark/analyze_pvd_oasis_v_search.py /mnt/d/code/sglang-V100-PVD-oasiskv/artifacts/fresh_io
wsl -d Ubuntu -- python3 /mnt/d/code/sglang-V100-PVD-oasiskv/benchmark/analyze_pvd_oasis_rpc.py /mnt/d/code/sglang-V100-PVD-oasiskv/artifacts/fresh_io
```
