# Oasis KV 交付：合并 reserve＋start公平对照

CloudLab 2026-10-02；完整快图＋V/CAGRA＋Oasis 配对逐层 Decode。
计划 `7d5d262fd`，合并提交实现 `6f5cab2ca`，共同接收区实现及验证源码 `9b8b5dc0c`；正式 tag `oasis_delivery_combine_abba02`。

## 结果

客户端完成中位数 **10.076→9.887 s（-1.87%）**；
每请求后续Decode累计 KV 等待 **6040.889→5780.557 ms**。
每模式四个正式请求、只覆盖两个Prompt；3136个层/rank profile不视作独立请求样本。两个独立优化的结果不能相加。新选项仍默认关闭。

## 公平性

base_a→opt_a→opt_b→base_b，各臂重启全角色，排除相同两次 warmup；两个2159-token Prompt、16输出token、greedy/ignore_eos。
四合一快图、degree16、prefix2048＋tail111、itopk2048、Top4、capacity32、max_new16、workers2保持一致。
V固定 pool/host candidates/GPU finite-Q proof；Triton打包关闭；D显式 reuse_io=false。初始KV仍图门后P→V→D，private Prompt seed计入客户端。
唯一变量为D combine_reserve_start；所有臂 reuse_receive_slots=false。
八次Prompt、实际输出ID和文本一致，cached_tokens=0；每请求420jobs/840搜索RPC，每模式3136稳态查询profile，均2items/14Qrows。
部署 bundle 中55个文件的实际哈希与V/D serving source gate一致；源码身份使用记录的部署包，没有读取后来编辑的工作树。
所有接收写入保留完整身份、精确native终态字节证明、安装后ACK及UNKNOWN保留；最终 owned={}、cleanup_errors=[]、六张GPU归零。

## 完整路径时间

| 指标（独立中位数） | 独立 reserve/start | 合并 reserve/start | 相对变化 |
|---|---:|---:|---:|
| V查询wall/rank/层 | 7.531 ms | 7.300 ms | -3.07% |
| D search_many | 12.122 ms | 11.875 ms | -2.03% |
| D整层检索/交付RPC | 34.931 ms | 33.924 ms | -2.88% |
| 层worker完整service | 38.690 ms | 37.439 ms | -3.23% |
| 每请求后续Decode累计KV等待 | 6040.889 ms | 5780.557 ms | -4.31% |
| 每步逐层等待和中位数 | 426.381 ms | 414.268 ms | -2.84% |
| 后续Decode执行 | 522.492 ms | 519.345 ms | -0.60% |
| 首个客户端事件 | 2.567 s | 2.541 s | -1.05% |
| 客户端完成 | 10.076 s | 9.887 s | -1.87% |

每请求累计等待先对该请求step>0的各次forward等待求和，再对每模式四个正式请求取中位数；
每步逐层等待和另按forward取中位数。两者范围不同，不能将每步约数百毫秒当作整个请求的累计等待。

## 实际交付子阶段与调用

以下时间为已完成的稳态missing-rank交付，排除初始28层bank priming；注册池首次注册仍计入客户端初始化。
控制提交时间先对每次交付的reserve/start/combined求和，再计算中位数。各阶段独立中位数不能相加重建请求。

| 子阶段 | 独立 reserve/start | 合并 reserve/start |
|---|---:|---:|
| 接收区准备 | 1.064 ms | 1.068 ms |
| 物理分配 | 0.017 ms | 0.017 ms |
| 物理注册 | 0.909 ms | 0.911 ms |
| poll RPC | 0.000 ms | 0.000 ms |
| 安装后ACK | 1.332 ms | 1.324 ms |
| 接收区close | 0.412 ms | 0.407 ms |
| GPU→CPU缓存copy | 0.283 ms | 0.288 ms |
| reserve/start控制提交 | 16.802 ms | 16.275 ms |

实际总调用计数覆盖每模式四个正式请求，含priming及最终物理池退休；poll为实测次数，start直接返回READY时可为零。

| 总计数 | 独立 reserve/start | 合并 reserve/start |
|---|---:|---:|
| missing-rank交付 | 3084 | 3084 |
| 物理register | 3084 | 3084 |
| 物理unregister | 3084 | 3084 |
| reserve RPC | 3084 | 0 |
| start RPC | 3084 | 0 |
| combined RPC | 0 | 3084 |
| poll RPC | 1 | 7 |
| ACK RPC | 3084 | 3084 |

reserve/start联合提交中位数缩短 **0.527 ms（3.14%）**。每模式四个请求的控制RPC总数从9253降至6175，减少3078次；
这里计入reserve/start/combined、实际poll和ACK，搜索RPC另外计数。
联合提交仍需16.275 ms，其范围包含V端source打包、注册、native提交与完成进展。当前profile没有将这些服务工作与HTTP往返分别计时，不能把联合阶段全算作网络固定开销。

## 四臂与流量

| 执行順序 | case 99401 | case 99402 |
|---|---:|---:|
| base_a | 10.032s | 10.434s |
| opt_a | 9.851s | 10.408s |
| opt_b | 9.922s | 9.713s |
| base_b | 10.110s | 10.041s |

前后同配置arm的客户端中位数变化：baseline -1.54%，optimized -3.08%；保留顺序漂移，不据八个请求宣称统计显著性。
本轮客户端1.87%的改善与这些顺序变化处于相近范围；调用减少已被计数证明，客户端收益仍需更多请求验证。
初始逻辑全KV为123805696/123805696 B/request；稀疏payload中位数2563072/2562816 B/request。
CAGRA候选允许原生抖动；没有冻结missing集合。查询、D RPC和交付时间包含不同范围，不把差值当作纯网络或纯native传输时间。

## 验证与范围

CPU gate实际结果：`512 passed, 4 skipped, 1 warning in 8.13s`；完整输出见gate.tar.gz与gate_count_record.json。
原生本地Mooncake gate通过48个精确字节案例，两个执行器复用四个物理MR并安全注销；本地session gate与线上跨节点RDMA对照是独立观测。
接收record集成重复gate：`9 passed, 1 warning in 2.14s`；这九项已包含在完整CPU gate中，不相加为新的独立测试。
单元异常测试采用CPU policy double；没有注入真实RDMA/CUDA故障。长Decode、多请求并发、TP2、更多Prompt质量与服务显存峰值仍开放。
客户端完成与最终安全退休分别验证；本次子阶段profile不能单独证明整个流水线的重叠收益。

## 证据与复现

同名目录保存完整raw.tar.gz、gate.tar.gz、紧凑summary/实际IO计数、V/RPC分解、部署来源、GPU归零与LF便携manifest。
前置失败尝试另存failed_prelaunch.tar.gz：D client.py source gate mismatch stopped before any service or request; sources aligned for a fresh ABBA.

```text
wsl -d Ubuntu -- python3 /mnt/d/code/sglang-V100-PVD-oasiskv/benchmark/run_pvd_oasis_serving_cloudlab.py --tag fresh_combine --comparison v-combine --arms base_a,opt_a,opt_b,base_b --cases 99401,99402 --tokens 16
wsl -d Ubuntu -- python3 /mnt/d/code/sglang-V100-PVD-oasiskv/benchmark/analyze_pvd_oasis_serving.py /mnt/d/code/sglang-V100-PVD-oasiskv/artifacts/fresh_combine --comparison v-combine
wsl -d Ubuntu -- python3 /mnt/d/code/sglang-V100-PVD-oasiskv/benchmark/analyze_pvd_oasis_v_search.py /mnt/d/code/sglang-V100-PVD-oasiskv/artifacts/fresh_combine
wsl -d Ubuntu -- python3 /mnt/d/code/sglang-V100-PVD-oasiskv/benchmark/analyze_pvd_oasis_rpc.py /mnt/d/code/sglang-V100-PVD-oasiskv/artifacts/fresh_combine
```
