# Oasis KV 交付：有界搜索／交付／安装阶段公平对照

CloudLab 2026-10-03；完整快图＋V/CAGRA＋Oasis 配对逐层 Decode。
日期按首个正式请求started_unix换算UTC+8：2026-10-03T05:28:41+08:00。

## 结果

客户端完成中位数 **9.894→9.998 s（+1.05%）**；
每请求后续Decode累计 KV 等待 **5835.922→6225.588 ms**。
每模式四个正式请求、只覆盖两个Prompt；3136个层/rank profile不视作独立请求样本。两个独立优化的结果不能相加。新选项仍默认关闭。

## 公平性

base_a→opt_a→opt_b→base_b，各臂重启全角色，排除相同两次 warmup；两个2159-token Prompt、16输出token、greedy/ignore_eos。
四合一快图、degree16、prefix2048＋tail111、itopk2048、Top4、capacity32、max_new16、workers2保持一致。
V固定 pool/host candidates/GPU finite-Q proof；Triton打包关闭；D显式 reuse_io=false。初始KV仍图门后P→V→D，private Prompt seed计入客户端。
唯一变量为D staged_transport；opt使用2搜索＋2交付＋1安装线程，持久线程相关loop/client/stream/registry。固定硬件，额外CPU线程明确计入；旧manager共享reuse_io开关仍false，stage自己的HTTP客户端跨job复用。GPU直接安装关闭，CPU历史缓存政策、每次交付注册、独立reserve/start保持一致。
八次Prompt、实际输出ID和文本一致，cached_tokens=0；每请求420jobs/840搜索RPC，每模式3136稳态查询profile，均2items/14Qrows。
部署 bundle 中329个文件的实际哈希与V/D serving source gate一致；源码身份使用记录的部署包，没有读取后来编辑的工作树。
所有接收写入保留完整身份、精确native终态字节证明、安装后ACK及UNKNOWN保留；最终 owned={}、cleanup_errors=[]、六张GPU归零。

## 完整路径时间

| 指标（独立中位数） | 两个完整回调worker | 2搜索＋2交付＋1安装线程 | 相对变化 |
|---|---:|---:|---:|
| V查询wall/rank/层 | 7.179 ms | 14.372 ms | +100.19% |
| D search_many | 11.811 ms | 19.380 ms | +64.09% |
| D整层检索/交付RPC | 33.769 ms | 62.766 ms | +85.87% |
| 完整链service（含阶段间queue） | 37.407 ms | 527.535 ms | +1310.27% |
| 每请求后续Decode累计KV等待 | 5835.922 ms | 6225.588 ms | +6.68% |
| 每步逐层等待和中位数 | 412.445 ms | 448.497 ms | +8.74% |
| 后续Decode执行 | 514.120 ms | 523.142 ms | +1.75% |
| 客户端TPOT均值（15个流式间隔） | 494.631 ms | 492.505 ms | -0.43% |
| D每token KV等待均值（后14步） | 419.790 ms | 446.477 ms | +6.36% |
| 首个客户端事件 | 2.522 s | 2.515 s | -0.28% |
| 客户端完成 | 9.894 s | 9.998 s | +1.05% |

每请求累计等待先对该请求step>0的各次forward等待求和，再对每模式四个正式请求取中位数；
每步逐层等待和另按forward取中位数。两者范围不同，不能将每步约数百毫秒当作整个请求的累计等待。

## 实际交付子阶段与调用

以下时间为已完成的稳态missing-rank交付，排除初始28层bank priming；注册池首次注册仍计入客户端初始化。
控制提交时间先对每次交付的reserve/start/combined求和，再计算中位数。各阶段独立中位数不能相加重建请求。

| 子阶段 | 两个完整回调worker | 2搜索＋2交付＋1安装线程 |
|---|---:|---:|
| 接收区准备 | 1.078 ms | 0.980 ms |
| 物理分配 | 0.018 ms | 0.041 ms |
| 物理注册 | 0.919 ms | 0.855 ms |
| poll RPC | 0.000 ms | 0.000 ms |
| 安装后ACK | 1.398 ms | 1.704 ms |
| 接收区close | 0.411 ms | 0.405 ms |
| GPU→CPU缓存copy | 0.283 ms | 0.428 ms |
| reserve/start控制提交 | 16.076 ms | 31.550 ms |

实际总调用计数覆盖每模式四个正式请求，含priming及最终物理池退休；poll为实测次数，start直接返回READY时可为零。

| 总计数 | 两个完整回调worker | 2搜索＋2交付＋1安装线程 |
|---|---:|---:|
| missing-rank交付 | 3084 | 3084 |
| 物理register | 3084 | 3084 |
| 物理unregister | 3084 | 3084 |
| reserve RPC | 3084 | 3084 |
| start RPC | 3084 | 3084 |
| combined RPC | 0 | 0 |
| poll RPC | 4 | 2 |
| ACK RPC | 3084 | 3084 |

## 四臂与流量

| 执行順序 | case 99401 | case 99402 |
|---|---:|---:|
| base_a | 10.087s | 10.415s |
| opt_a | 10.120s | 10.203s |
| opt_b | 9.875s | 9.561s |
| base_b | 9.596s | 9.701s |

前后同配置arm的客户端中位数变化：baseline -5.88%，optimized -4.37%；保留顺序漂移，不据八个请求宣称统计显著性。
初始逻辑全KV为123805696/123805696 B/request；稀疏payload中位数2563072/2563072 B/request。
CAGRA候选允许原生抖动；没有冻结missing集合。查询、D RPC和交付时间包含不同范围，不把差值当作纯网络或纯native传输时间。

## 验证与范围

CPU gate实际结果：`116 passed, 1 warning in 10.39s`；完整输出见gate.tar.gz与gate_count_record.json。
原生本地Mooncake gate通过48个精确字节案例，由真实有界stage执行器调度48次native scatter，交付线程峰值2，持久CUDA资源在创建线程退休；本地search/install为调度／proof检查，未在此gate运行CAGRA或serving bank安装，完整阶段另由线上trace验证；本地session gate与线上跨节点RDMA对照是独立观测。
单元异常测试采用CPU policy double；没有注入真实RDMA/CUDA故障。长Decode、多请求并发、TP2、更多Prompt质量与服务显存峰值仍开放。
客户端完成与最终安全退休分别验证；本次子阶段profile不能单独证明整个流水线的重叠收益。

## 证据与复现

同名目录保存完整raw.tar.gz、gate.tar.gz、紧凑summary/实际IO计数、V/RPC分解、部署来源、GPU归零与LF便携manifest。

## 实际阶段与持续完成间隔

成功请求的每个stage均按发布时刻的60秒截止时间验证；下一阶段持有独立job context，未转移native registry的线程归属。
每请求420个已完成job、392个实际消费的后续bank；stage关闭时队列、原生记录和持久owner均退休。
| 算术平均指标 | 原回调 | 阶段流水线 |
|---|---:|---:|
| 客户端TPOT | 494.631 ms | 492.505 ms |
| D每token KV等待 | 419.790 ms | 446.477 ms |
| 持续bank完成间隔 | 18.842 ms | 18.642 ms |
| 完整链service（含阶段间队列） | 37.725 ms | 516.995 ms |
| 发布→实际消费 | 525.394 ms | 533.480 ms |

| 优化臂阶段 | 线程数／实际峰值 | 平均service | 平均queue |
|---|---:|---:|---:|
| search | 2/2 | 27.299 ms | 15.587 ms |
| delivery | 2/2 | 37.541 ms | 445.461 ms |
| install | 1/1 | 5.119 ms | 1.573 ms |

持续间隔按392个实际READY时刻的跨度／391计算，含前台发布节奏；不是纯GPU吞吐上限。
完整链中可以同时存在多个排队job，不能用完整链service／2解释stage线程占用。rpc_summary将不适用的worker容量字段置null。
额外线程、stage持久HTTP资源和排队均属于这一架构变量；不叠加此前独立连接／注册优化的结果。
stage_timing_summary.json保存逐请求实测阶段、完成间隔与最小READY截止余量。仍需长Decode、TP2、负载与原生故障注入，默认关闭。

```text
wsl -d Ubuntu -- python3 /mnt/d/code/sglang-V100-PVD-oasiskv/benchmark/run_pvd_oasis_serving_cloudlab.py --tag fresh_stages --comparison d-stages --arms base_a,opt_a,opt_b,base_b --cases 99401,99402 --tokens 16
wsl -d Ubuntu -- python3 /mnt/d/code/sglang-V100-PVD-oasiskv/benchmark/analyze_pvd_oasis_serving.py /mnt/d/code/sglang-V100-PVD-oasiskv/artifacts/fresh_stages --comparison d-stages
wsl -d Ubuntu -- python3 /mnt/d/code/sglang-V100-PVD-oasiskv/benchmark/analyze_pvd_oasis_v_search.py /mnt/d/code/sglang-V100-PVD-oasiskv/artifacts/fresh_stages
wsl -d Ubuntu -- python3 /mnt/d/code/sglang-V100-PVD-oasiskv/benchmark/analyze_pvd_oasis_rpc.py /mnt/d/code/sglang-V100-PVD-oasiskv/artifacts/fresh_stages
```

## 采用决定与前置检查

没有观察到稳定客户端收益，staged_transport继续默认关闭。
完整链的平均service从37.725变为516.995 ms，后者包含交付队列，不能解释成单个线程执行变慢到517 ms。
实际阶段均值为search27.299、delivery37.541、install5.119 ms；交付queue均值445.461 ms。
持续bank完成间隔18.842→18.642 ms，距离约1.031 ms/layer的READY-KV前台诊断预算仍很大。
这是实测完成节奏比较，不是可达到延迟的预测。
V batch wall7.180→14.372 ms；更多同时活跃的搜索／交付使host wall范围增加，具体锁、CUDA同步及网络贡献尚未独立隔离。
单独CAGRA kernel并不能解释完整交付，也不把两者差值当作网络时长。
本地CPU复核另存local_cpu.tar.gz：首次因project-local basetemp父目录未建立产生4个fixture errors，未涉及服务；修正路径后53 passed。
这53项已包含在CloudLab116项gate中，不相加为额外独立测试。实际native失败注入仍未执行。
