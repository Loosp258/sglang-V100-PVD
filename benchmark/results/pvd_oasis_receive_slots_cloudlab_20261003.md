# Oasis KV 交付：复用接收区物理注册公平对照

CloudLab 2026-10-03；完整快图＋V/CAGRA＋Oasis 配对逐层 Decode。
日期按首个正式请求started_unix换算UTC+8：2026-10-03T00:01:31+08:00。
计划 `7d5d262fd`，合并接口实现 `6f5cab2ca`，接收区实现及冻结服务源码 `9b8b5dc0c`；正式 tag `oasis_delivery_slots_abba01`。
本轮与合并提交对照使用同一份55文件部署源码，bundle SHA256为 `e004d72d10f4816d50ced8ded13755d17ed5ae30822f0b4dfa9dcd2479527e51`；复用10月2日保存的CPU及原生gate。

## 结果

客户端完成中位数 **10.202→10.110 s（-0.91%）**；
每请求后续Decode累计 KV 等待 **6133.970→6159.638 ms**。
每模式四个正式请求、只覆盖两个Prompt；3136个层/rank profile不视作独立请求样本。两个独立优化的结果不能相加。新选项仍默认关闭。
**物理注册和接收区准备的局部收益已被计数及阶段计时证明；D累计等待没有改善，本轮尚未证明稳定端到端收益。**

## 公平性

base_a→opt_a→opt_b→base_b，各臂重启全角色，排除相同两次 warmup；两个2159-token Prompt、16输出token、greedy/ignore_eos。
四合一快图、degree16、prefix2048＋tail111、itopk2048、Top4、capacity32、max_new16、workers2保持一致。
V固定 pool/host candidates/GPU finite-Q proof；Triton打包关闭；D显式 reuse_io=false。初始KV仍图门后P→V→D，private Prompt seed计入客户端。
唯一变量为D reuse_receive_slots；所有臂 combine_reserve_start=false。
八次Prompt、实际输出ID和文本一致，cached_tokens=0；每请求420jobs/840搜索RPC，每模式3136稳态查询profile，均2items/14Qrows。
部署 bundle 中55个文件的实际哈希与V/D serving source gate一致；源码身份使用记录的部署包，没有读取后来编辑的工作树。
所有接收写入保留完整身份、精确native终态字节证明、安装后ACK及UNKNOWN保留；最终 owned={}、cleanup_errors=[]、六张GPU归零。

## 完整路径时间

| 指标（独立中位数） | 每次交付注册 | 请求内复用 slots | 相对变化 |
|---|---:|---:|---:|
| V查询wall/rank/层 | 7.658 ms | 7.778 ms | +1.57% |
| D search_many | 12.332 ms | 11.870 ms | -3.74% |
| D整层检索/交付RPC | 35.262 ms | 34.749 ms | -1.45% |
| 层worker完整service | 39.118 ms | 37.908 ms | -3.09% |
| 每请求后续Decode累计KV等待 | 6133.970 ms | 6159.638 ms | +0.42% |
| 每步逐层等待和中位数 | 423.600 ms | 424.259 ms | +0.16% |
| 后续Decode执行 | 528.746 ms | 523.334 ms | -1.02% |
| 首个客户端事件 | 2.557 s | 2.580 s | +0.89% |
| 客户端完成 | 10.202 s | 10.110 s | -0.91% |

每请求累计等待先对该请求step>0的各次forward等待求和，再对每模式四个正式请求取中位数；
每步逐层等待和另按forward取中位数。两者范围不同，不能将每步约数百毫秒当作整个请求的累计等待。

## 实际交付子阶段与调用

以下时间为已完成的稳态missing-rank交付，排除初始28层bank priming；注册池首次注册仍计入客户端初始化。
控制提交时间先对每次交付的reserve/start/combined求和，再计算中位数。各阶段独立中位数不能相加重建请求。

| 子阶段 | 每次交付注册 | 请求内复用 slots |
|---|---:|---:|
| 接收区准备 | 1.072 ms | 0.111 ms |
| 物理分配 | 0.018 ms | 0.000 ms |
| 物理注册 | 0.913 ms | 0.000 ms |
| poll RPC | 0.000 ms | 0.000 ms |
| 安装后ACK | 1.395 ms | 1.212 ms |
| 接收区close | 0.410 ms | 0.015 ms |
| 每次交付prepare＋close联合中位数 | 1.471 ms | 0.125 ms |
| GPU→CPU缓存copy | 0.284 ms | 0.296 ms |
| reserve/start控制提交 | 16.889 ms | 18.569 ms |

prepare＋close联合值先对每次交付求和，再取2860个稳态missing-rank交付的中位数，缩短1.346 ms；没有相加两个独立中位数。
复用模式中的close负责返还逻辑lease；四个物理MR在请求最终退休时注销，完整注销计数另由最终pool inventory证明，退休时间未单独计时。

实际总调用计数覆盖每模式四个正式请求，含priming及最终物理池退休；poll为实测次数，start直接返回READY时可为零。

| 总计数 | 每次交付注册 | 请求内复用 slots |
|---|---:|---:|
| missing-rank交付 | 3084 | 3084 |
| 物理register | 3084 | 16 |
| 物理unregister | 3084 | 16 |
| reserve RPC | 3084 | 3084 |
| start RPC | 3084 | 3084 |
| combined RPC | 0 | 0 |
| poll RPC | 3 | 6 |
| ACK RPC | 3084 | 3084 |

物理register/unregister各从3084次降至16次，减少 **99.48%**；四个优化请求每个各注册、注销四个MR。
每请求两个rank各有两个32768B物理接收槽，持久接收容量131072B；所有逻辑lease返还后，最终物理槽、字节和UNKNOWN计数均为零。
控制RPC仍为reserve＋start＋ACK，合计9255→9258次，三次差异来自实测poll。
reserve/start联合提交反而增加1.680 ms（9.95%）。这一阶段包含V源端打包、注册、native提交及完成进展，当前profile不能进一步区分具体阻塞点；局部省下的接收端工作没有转化为D等待改善。

## 四臂与流量

| 执行順序 | case 99401 | case 99402 |
|---|---:|---:|
| base_a | 9.998s | 10.408s |
| opt_a | 10.141s | 9.957s |
| opt_b | 10.405s | 10.079s |
| base_b | 10.042s | 10.363s |

前后同配置arm的客户端中位数变化：baseline -0.01%，optimized +1.92%；保留顺序漂移，不据八个请求宣称统计显著性。
客户端0.91%的改善小于优化组两个arm间1.92%的变化；同时每请求D累计等待增加0.42%，首个事件增加0.89%。保持选项默认关闭，扩大请求样本后再判断端到端收益。
按同一Prompt的两次运行分别取中位数，case99401客户端时间10.020→10.273s（+2.52%），case99402为10.385→10.018s（-3.54%），变化方向相反。
本次复用发生在D接收端；V源端仍在每次交付时分配staging、复制所选KV并注册发送区，相关开销仍包含在提交阶段。
初始逻辑全KV为123805696/123805696 B/request；稀疏payload中位数2562816/2562816 B/request。
CAGRA候选允许原生抖动；没有冻结missing集合。查询、D RPC和交付时间包含不同范围，不把差值当作纯网络或纯native传输时间。

## 验证与范围

CPU gate实际结果：`512 passed, 4 skipped, 1 warning in 8.13s`；完整输出见gate.tar.gz与gate_count_record.json。
原生本地Mooncake gate通过48个精确字节案例，两个执行器复用四个物理MR并安全注销；本地session gate与线上跨节点RDMA对照是独立观测。
接收record集成重复gate：`9 passed, 1 warning in 2.14s`；这九项已包含在完整CPU gate中，不相加为新的独立测试。
单元异常测试采用CPU policy double；没有注入真实RDMA/CUDA故障。长Decode、多请求并发、TP2、更多Prompt质量与服务显存峰值仍开放。
客户端完成与最终安全退休分别验证；本次子阶段profile不能单独证明整个流水线的重叠收益。

## 证据与复现

同名目录保存完整raw.tar.gz、gate.tar.gz、紧凑summary/实际IO计数、V/RPC分解、部署来源、GPU归零与LF便携manifest。

```text
wsl -d Ubuntu -- python3 /mnt/d/code/sglang-V100-PVD-oasiskv/benchmark/run_pvd_oasis_serving_cloudlab.py --tag fresh_slots --comparison v-slots --arms base_a,opt_a,opt_b,base_b --cases 99401,99402 --tokens 16
wsl -d Ubuntu -- python3 /mnt/d/code/sglang-V100-PVD-oasiskv/benchmark/analyze_pvd_oasis_serving.py /mnt/d/code/sglang-V100-PVD-oasiskv/artifacts/fresh_slots --comparison v-slots
wsl -d Ubuntu -- python3 /mnt/d/code/sglang-V100-PVD-oasiskv/benchmark/analyze_pvd_oasis_v_search.py /mnt/d/code/sglang-V100-PVD-oasiskv/artifacts/fresh_slots
wsl -d Ubuntu -- python3 /mnt/d/code/sglang-V100-PVD-oasiskv/benchmark/analyze_pvd_oasis_rpc.py /mnt/d/code/sglang-V100-PVD-oasiskv/artifacts/fresh_slots
```
