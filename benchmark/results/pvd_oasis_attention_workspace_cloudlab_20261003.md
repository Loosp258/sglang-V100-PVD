# Oasis KV 交付：有界注意力工作区公平对照

CloudLab 2026-10-03；完整快图＋V/CAGRA＋Oasis 配对逐层 Decode。
日期按首个正式请求started_unix换算UTC+8：2026-10-03T06:17:08+08:00。

## 结果

客户端完成中位数 **9.983→9.870 s（-1.13%）**；
每请求后续Decode累计 KV 等待 **5921.626→5770.256 ms**。
每模式四个正式请求、只覆盖两个Prompt；3136个层/rank profile不视作独立请求样本。两个独立优化的结果不能相加。新选项仍默认关闭。

## 公平性

base_a→opt_a→opt_b→base_b，各臂重启全角色，排除相同两次 warmup；两个2159-token Prompt、16输出token、greedy/ignore_eos。
四合一快图、degree16、prefix2048＋tail111、itopk2048、Top4、capacity32、max_new16、workers2保持一致。
V固定 pool/host candidates/GPU finite-Q proof；Triton打包关闭；D显式 reuse_io=false。初始KV仍图门后P→V→D，private Prompt seed计入客户端。
唯一变量为D attention_workspace；保持原变长span、GQA展开、mask和SDPA语义。工作区独立持有在已计费的32MiB request scratch内，关闭时先同步再清空。完整线上ABBA不捕获CUDA Graph；纯SDPA图另由同进程真实轨迹诊断验证。stage/GPU备份、HTTP复用、合并reserve/start和接收槽均关闭。
八次Prompt、实际输出ID和文本一致，cached_tokens=0；每请求420jobs/840搜索RPC，每模式3136稳态查询profile，均2items/14Qrows。
部署 bundle 中331个文件的实际哈希与V/D serving source gate一致；源码身份使用记录的部署包，没有读取后来编辑的工作树。
所有接收写入保留完整身份、精确native终态字节证明、安装后ACK及UNKNOWN保留；最终 owned={}、cleanup_errors=[]、六张GPU归零。

## 完整路径时间

| 指标（独立中位数） | 普通逐层分配 | 预分配工作区 | 相对变化 |
|---|---:|---:|---:|
| V查询wall/rank/层 | 7.167 ms | 6.838 ms | -4.58% |
| D search_many | 11.796 ms | 11.482 ms | -2.66% |
| D整层检索/交付RPC | 33.831 ms | 33.007 ms | -2.44% |
| 层worker完整service | 37.565 ms | 36.800 ms | -2.04% |
| 每请求后续Decode累计KV等待 | 5921.626 ms | 5770.256 ms | -2.56% |
| 每步逐层等待和中位数 | 416.421 ms | 398.550 ms | -4.29% |
| 后续Decode执行 | 514.636 ms | 504.757 ms | -1.92% |
| 客户端TPOT均值（15个流式间隔） | 494.448 ms | 487.835 ms | -1.34% |
| D每token KV等待均值（后14步） | 420.369 ms | 411.835 ms | -2.03% |
| 首个客户端事件 | 2.503 s | 2.537 s | +1.35% |
| 客户端完成 | 9.983 s | 9.870 s | -1.13% |

每请求累计等待先对该请求step>0的各次forward等待求和，再对每模式四个正式请求取中位数；
每步逐层等待和另按forward取中位数。两者范围不同，不能将每步约数百毫秒当作整个请求的累计等待。

## 实际 token 间隔与后台任务容量

客户端TPOT按每请求实际16个token事件的(last-first)/15计算，排除TTFT；每模式4请求，共60个输出间隔。
首token来自P Prefill；首个D forward使用已初始化bank，KV等待为零。另报告其后14个间隔，避免短Decode的首步降低均值。
后续KV等待按step>0累计等待/56计算算术平均；以下请求TPOT中位数、单步等待中位数与算术平均分别标明。

| 指标 | 普通逐层分配 | 预分配工作区 | 相对变化 |
|---|---:|---:|---:|
| 客户端TPOT算术平均 | 494.448 ms | 487.835 ms | -1.34% |
| 客户端TPOT请求中位数 | 496.294 ms | 488.977 ms | -1.47% |
| 后续14间隔TPOT算术平均 | 525.502 ms | 518.912 ms | -1.25% |
| 后续14间隔TPOT请求中位数 | 527.102 ms | 520.027 ms | -1.34% |
| 后续每token KV等待算术平均 | 420.369 ms | 411.835 ms | -2.03% |
| 后续每token KV等待单步中位数 | 416.421 ms | 398.550 ms | -4.29% |
| 层任务service算术平均 | 37.706 ms | 37.201 ms | -1.34% |
| 层任务queue算术平均 | 487.531 ms | 480.913 ms | -1.36% |
| 实际service总和/workers/56 | 527.888 ms | 520.820 ms | -1.34% |

实际callback峰值并发：baseline [2, 2, 2, 2]，optimized [2, 2, 2, 2]；每请求392个已消费任务，每模式1568个。
按实际start/ready时间戳重建的worker占用率为99.725%/99.722%；
消费者需要前已经READY的比例为3.125%/4.082%。
占用率=sum(service)/(workers×已消费callback时间窗)，包含RPC阻塞，不能解释成CPU或GPU利用率。
service总和/workers/56仅描述本次观察到的任务容量，未假定增加并发后service保持不变，也不是端到端时延下限或未来加速预测。
客户端事件、28层等待、worker callback与V单rank查询包含不同范围；独立中位数不能相加解释客户端TPOT。

## 实际交付子阶段与调用

以下时间为已完成的稳态missing-rank交付，排除初始28层bank priming；注册池首次注册仍计入客户端初始化。
控制提交时间先对每次交付的reserve/start/combined求和，再计算中位数。各阶段独立中位数不能相加重建请求。

| 子阶段 | 普通逐层分配 | 预分配工作区 |
|---|---:|---:|
| 接收区准备 | 1.074 ms | 1.070 ms |
| 物理分配 | 0.018 ms | 0.018 ms |
| 物理注册 | 0.914 ms | 0.909 ms |
| poll RPC | 0.000 ms | 0.000 ms |
| 安装后ACK | 1.400 ms | 1.363 ms |
| 接收区close | 0.411 ms | 0.408 ms |
| GPU→CPU缓存copy | 0.285 ms | 0.286 ms |
| reserve/start控制提交 | 16.136 ms | 15.560 ms |

实际总调用计数覆盖每模式四个正式请求，含priming及最终物理池退休；poll为实测次数，start直接返回READY时可为零。

| 总计数 | 普通逐层分配 | 预分配工作区 |
|---|---:|---:|
| missing-rank交付 | 3084 | 3084 |
| 物理register | 3084 | 3084 |
| 物理unregister | 3084 | 3084 |
| reserve RPC | 3084 | 3084 |
| start RPC | 3084 | 3084 |
| combined RPC | 0 | 0 |
| poll RPC | 7 | 7 |
| ACK RPC | 3084 | 3084 |

## 四臂与流量

| 执行順序 | case 99401 | case 99402 |
|---|---:|---:|
| base_a | 9.473s | 10.258s |
| opt_a | 9.669s | 10.296s |
| opt_b | 9.379s | 10.071s |
| base_b | 10.075s | 9.891s |

前后同配置arm的客户端中位数变化：baseline +1.19%，optimized -2.58%；保留顺序漂移，不据八个请求宣称统计显著性。
初始逻辑全KV为123805696/123805696 B/request；稀疏payload中位数2562816/2563072 B/request。
CAGRA候选允许原生抖动；没有冻结missing集合。查询、D RPC和交付时间包含不同范围，不把差值当作纯网络或纯native传输时间。

## 验证与范围

CPU gate实际结果：`121 passed, 1 warning in 10.43s`；完整输出见gate.tar.gz与gate_count_record.json。
原生本地Mooncake gate通过48个精确字节案例，真实收到的字节经serving bank安装后与独立oracle精确比较；原注意力与工作区在0/3/14历史长度上CUDA逐位一致。此处Q为代理随机张量，真实模型logits/token由独立轨迹实验验证；本地session gate与线上跨节点RDMA对照是独立观测。
单元异常测试采用CPU policy double；没有注入真实RDMA/CUDA故障。长Decode、多请求并发、TP2、更多Prompt质量与服务显存峰值仍开放。
客户端完成与最终安全退休分别验证；本次子阶段profile不能单独证明整个流水线的重叠收益。

## 证据与复现

同名目录保存完整raw.tar.gz、gate.tar.gz、紧凑summary/实际IO计数、V/RPC分解、部署来源、GPU归零与LF便携manifest。
前置失败尝试另存failed_prelaunch.tar.gz：首个native夹具未归还接收lease，四个案例完成后下一轮acquire屏障中断；未产生线上正式请求，未纳入资格与性能统计。修复lease退休并覆盖64-row夹具后重新冻结gate02，通过121项CPU与48项native。

```text
wsl -d Ubuntu -- python3 /mnt/d/code/sglang-V100-PVD-oasiskv/benchmark/run_pvd_oasis_serving_cloudlab.py --tag fresh_workspace --comparison d-workspace --arms base_a,opt_a,opt_b,base_b --cases 99401,99402 --tokens 16
wsl -d Ubuntu -- python3 /mnt/d/code/sglang-V100-PVD-oasiskv/benchmark/analyze_pvd_oasis_serving.py /mnt/d/code/sglang-V100-PVD-oasiskv/artifacts/fresh_workspace --comparison d-workspace
wsl -d Ubuntu -- python3 /mnt/d/code/sglang-V100-PVD-oasiskv/benchmark/analyze_pvd_oasis_v_search.py /mnt/d/code/sglang-V100-PVD-oasiskv/artifacts/fresh_workspace
wsl -d Ubuntu -- python3 /mnt/d/code/sglang-V100-PVD-oasiskv/benchmark/analyze_pvd_oasis_rpc.py /mnt/d/code/sglang-V100-PVD-oasiskv/artifacts/fresh_workspace
```

## READY-KV 真实轨迹与 CUDA Graph

同一加载的SGLang runner、真实EAGLE预测闭包、普通采样器和两份实际逐层KV轨迹，分别运行original/workspace/sdpa_graph。
按case与repetition交替反序，每模式两次排除warmup、三次wall trial；event另测。
logits、features、实际KV、预测／实际token、positions与420次formal KV写入均逐位一致，所有消费前callback已READY。
网络／V／接收／CPU备份竞争不在此诊断中；不是从线上耗时减去等待得到的纯计算数字。

| READY-KV 模式 | D前台算术平均ms/token | 每trial准备／捕获ms，另计 |
|---|---:|---:|
| 原路径 | 28.836 | 0.001 |
| 有界工作区 | 28.604 | 0.103 |
| 纯SDPA CUDA Graph | 28.543 | 20.043 |

工作区占820160 B；39个实际span的graph分配增量560128 B，32MiB显式上限检查和同步后退休通过。
CUDA Graph仅捕获SDPA；future等待、Q发布、EAGLE、采样与formal写入均在图外。
这次短Decode后14步的graph稳态节省约4.10ms，小于每次20.04ms准备成本；完整线上ABBA未使用graph。
完整轨迹、replay源码、各次计时和哈希验证见[pvd_oasis_attention_replay_cloudlab_20261003.md](pvd_oasis_attention_replay_cloudlab_20261003.md)及同名目录。

## 采用决定与本轮结论

本轮完整路径平均KV等待420.369→411.835ms/token（-2.03%），平均客户端TPOT494.448→487.835ms（-1.34%），完成中位数9.983→9.870s（-1.13%）。
两个Prompt、四请求／模式不足以证明稳定生产收益；opt_a两请求的完成时间高于base_a，opt_b降低，顺序漂移已分别保留。
工作区的READY-KV改善只有约0.23ms/token；没有把完整KV交付降到D约1ms/layer的前台预算内。继续保持默认关闭，没有上线CUDA Graph。
V原生CAGRA提交／完成仍约1.9ms/rank/layer，V整体查询约7ms；完整callback平均约37ms，还包含检索、交付、控制与安装。
这些是不同范围的实测，不把独立中位数相减解释成纯网络成本，也不把service/workers当作端到端延迟下限。
更长Decode、更多Prompt、TP2、负载和真实native故障注入仍未覆盖；本轮完成的是已授权的四步有界实验序列。

本地24项CPU与11项诊断hook重复结果另存local_cpu_repeats.tar.gz；不与CloudLab121项相加。
代码、结果与失败尝试在codex/pvd-oasiskv本地提交；不push GitHub。
