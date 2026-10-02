# Oasis KV 交付：D GPU接收直接安装与异步CPU备份公平对照

CloudLab 2026-10-03；完整快图＋V/CAGRA＋Oasis 配对逐层 Decode。
日期按首个正式请求started_unix换算UTC+8：2026-10-03T05:00:25+08:00。

## 结果

客户端完成中位数 **10.147→10.310 s（+1.60%）**；
每请求后续Decode累计 KV 等待 **5990.404→6125.198 ms**。
每模式四个正式请求、只覆盖两个Prompt；3136个层/rank profile不视作独立请求样本。两个独立优化的结果不能相加。新选项仍默认关闭。

## 公平性

base_a→opt_a→opt_b→base_b，各臂重启全角色，排除相同两次 warmup；两个2159-token Prompt、16输出token、greedy/ignore_eos。
四合一快图、degree16、prefix2048＋tail111、itopk2048、Top4、capacity32、max_new16、workers2保持一致。
V固定 pool/host candidates/GPU finite-Q proof；Triton打包关闭；D显式 reuse_io=false。初始KV仍图门后P→V→D，private Prompt seed计入客户端。
唯一变量为D gpu_receive_to_bank；V两臂均采用staging packed PUT，直接scatter关闭。GPU接收先完成私有clone，再ACK/退休MR；下一bank直接从clone安装，CPU历史备份独立持有引用和预算。
八次Prompt、实际输出ID和文本一致，cached_tokens=0；每请求420jobs/840搜索RPC，每模式3136稳态查询profile，均2items/14Qrows。
部署 bundle 中327个文件的实际哈希与V/D serving source gate一致；源码身份使用记录的部署包，没有读取后来编辑的工作树。
所有接收写入保留完整身份、精确native终态字节证明、安装后ACK及UNKNOWN保留；最终 owned={}、cleanup_errors=[]、六张GPU归零。

## 完整路径时间

| 指标（独立中位数） | 同步CPU缓存往返 | GPU直接安装＋owned异步备份 | 相对变化 |
|---|---:|---:|---:|
| V查询wall/rank/层 | 7.366 ms | 7.722 ms | +4.83% |
| D search_many | 12.035 ms | 12.421 ms | +3.22% |
| D整层检索/交付RPC | 34.608 ms | 35.871 ms | +3.65% |
| 层worker完整service | 38.318 ms | 39.183 ms | +2.26% |
| 每请求后续Decode累计KV等待 | 5990.404 ms | 6125.198 ms | +2.25% |
| 每步逐层等待和中位数 | 417.995 ms | 427.100 ms | +2.18% |
| 后续Decode执行 | 523.087 ms | 534.037 ms | +2.09% |
| 客户端TPOT均值（15个流式间隔） | 502.199 ms | 518.285 ms | +3.20% |
| D每token KV等待均值（后14步） | 425.105 ms | 440.957 ms | +3.73% |
| 首个客户端事件 | 2.561 s | 2.609 s | +1.89% |
| 客户端完成 | 10.147 s | 10.310 s | +1.60% |

每请求累计等待先对该请求step>0的各次forward等待求和，再对每模式四个正式请求取中位数；
每步逐层等待和另按forward取中位数。两者范围不同，不能将每步约数百毫秒当作整个请求的累计等待。

## 实际 token 间隔与后台任务容量

客户端TPOT按每请求实际16个token事件的(last-first)/15计算，排除TTFT；每模式4请求，共60个输出间隔。
首token来自P Prefill；首个D forward使用已初始化bank，KV等待为零。另报告其后14个间隔，避免短Decode的首步降低均值。
后续KV等待按step>0累计等待/56计算算术平均；以下请求TPOT中位数、单步等待中位数与算术平均分别标明。

| 指标 | 同步CPU缓存往返 | GPU直接安装＋owned异步备份 | 相对变化 |
|---|---:|---:|---:|
| 客户端TPOT算术平均 | 502.199 ms | 518.285 ms | +3.20% |
| 客户端TPOT请求中位数 | 505.759 ms | 514.445 ms | +1.72% |
| 后续14间隔TPOT算术平均 | 534.015 ms | 551.024 ms | +3.19% |
| 后续14间隔TPOT请求中位数 | 537.956 ms | 546.967 ms | +1.68% |
| 后续每token KV等待算术平均 | 425.105 ms | 440.957 ms | +3.73% |
| 后续每token KV等待单步中位数 | 417.995 ms | 427.100 ms | +2.18% |
| 层任务service算术平均 | 38.303 ms | 39.523 ms | +3.19% |
| 层任务queue算术平均 | 494.953 ms | 511.146 ms | +3.27% |
| 实际service总和/workers/56 | 536.241 ms | 553.323 ms | +3.19% |

实际callback峰值并发：baseline [2, 2, 2, 2]，optimized [2, 2, 2, 2]；每请求392个已消费任务，每模式1568个。
按实际start/ready时间戳重建的worker占用率为99.743%/99.717%；
消费者需要前已经READY的比例为3.890%/5.102%。
占用率=sum(service)/(workers×已消费callback时间窗)，包含RPC阻塞，不能解释成CPU或GPU利用率。
service总和/workers/56仅描述本次观察到的任务容量，未假定增加并发后service保持不变，也不是端到端时延下限或未来加速预测。
客户端事件、28层等待、worker callback与V单rank查询包含不同范围；独立中位数不能相加解释客户端TPOT。

## 实际交付子阶段与调用

以下时间为已完成的稳态missing-rank交付，排除初始28层bank priming；注册池首次注册仍计入客户端初始化。
控制提交时间先对每次交付的reserve/start/combined求和，再计算中位数。各阶段独立中位数不能相加重建请求。

| 子阶段 | 同步CPU缓存往返 | GPU直接安装＋owned异步备份 |
|---|---:|---:|
| 接收区准备 | 1.081 ms | 1.078 ms |
| 物理分配 | 0.018 ms | 0.017 ms |
| 物理注册 | 0.920 ms | 0.916 ms |
| poll RPC | 0.000 ms | 0.000 ms |
| 安装后ACK | 1.334 ms | 1.379 ms |
| 接收区close | 0.411 ms | 0.418 ms |
| 接收安装／GPU clone与备份提交 | 0.293 ms | 0.316 ms |
| reserve/start控制提交 | 16.567 ms | 17.365 ms |

实际总调用计数覆盖每模式四个正式请求，含priming及最终物理池退休；poll为实测次数，start直接返回READY时可为零。

| 总计数 | 同步CPU缓存往返 | GPU直接安装＋owned异步备份 |
|---|---:|---:|
| missing-rank交付 | 3084 | 3084 |
| 物理register | 3084 | 3084 |
| 物理unregister | 3084 | 3084 |
| reserve RPC | 3084 | 3084 |
| start RPC | 3084 | 3084 |
| combined RPC | 0 | 0 |
| poll RPC | 4 | 1 |
| ACK RPC | 3084 | 3084 |

## 四臂与流量

| 执行順序 | case 99401 | case 99402 |
|---|---:|---:|
| base_a | 10.068s | 10.480s |
| opt_a | 10.146s | 10.473s |
| opt_b | 10.130s | 10.741s |
| base_b | 9.641s | 10.226s |

前后同配置arm的客户端中位数变化：baseline -3.31%，optimized +1.22%；保留顺序漂移，不据八个请求宣称统计显著性。
初始逻辑全KV为123805696/123805696 B/request；稀疏payload中位数2562816/2563328 B/request。
CAGRA候选允许原生抖动；没有冻结missing集合。查询、D RPC和交付时间包含不同范围，不把差值当作纯网络或纯native传输时间。

## 验证与范围

CPU gate实际结果：`110 passed, 1 warning in 10.06s`；完整输出见gate.tar.gz与gate_count_record.json。
原生本地Mooncake gate通过48个精确字节案例，故意阻塞CPU缓存发布，并在原接收逻辑lease退休后覆盖接收区；GPU bank与历史CPU备份均保持精确，独立引用和预算完整退休。固定物理MR在全部案例完成后注销；本地session gate与线上跨节点RDMA对照是独立观测。
单元异常测试采用CPU policy double；没有注入真实RDMA/CUDA故障。长Decode、多请求并发、TP2、更多Prompt质量与服务显存峰值仍开放。
客户端完成与最终安全退休分别验证；本次子阶段profile不能单独证明整个流水线的重叠收益。

## 证据与复现

同名目录保存完整raw.tar.gz、gate.tar.gz、紧凑summary/实际IO计数、V/RPC分解、部署来源、GPU归零与LF便携manifest。
前置失败尝试另存failed_prelaunch.tar.gz：强化最后unpin与并发预算退款的释放顺序，SIGINT中断P启动；未产生正式请求，finally完整释放全部记录服务，六张GPU归零。该目录另保存此前gate02完整通过记录，主资格仅采用gate03。

## 异步备份与采用范围

opt在同一台D上添加两个有界CPU备份线程；完整回调workers仍为2。它保持每次交付物理注册，不把CPU缓存发布作为新GPU bank就绪的依赖。
原生强制延迟实验验证逻辑lease可先退休、接收区可先覆盖；实际跨节点路径仍逐次物理注销，IO快照记录了全部注册/注销/ACK与备份完成数。
CPU历史行仅在复制完成后标记valid；pending行可从独立GPU owner取出。GPU reader和CPU备份各持一份pin，借用行引用先清空，再退还存储预算。
opt的cache_copy_seconds计量私有GPU clone与备份提交，异步CPU发布时间另在gpu_backup.background_seconds记录；它不是完整D2H复制时长。
没有根据本次八个短请求默认启用；TPOT、均值等待和顺序漂移分别保留，不能相加或用局部copy节省推断客户端收益。
先行原型和首次gate另存previous_pilot.tar.gz/previous_gate.tar.gz。复查后修正临时行别名与预算退款次序，重新冻结与重测；主统计只来自当前一轮ABBA。

```text
wsl -d Ubuntu -- python3 /mnt/d/code/sglang-V100-PVD-oasiskv/benchmark/run_pvd_oasis_serving_cloudlab.py --tag fresh_gpu_bank --comparison d-gpu-bank --arms base_a,opt_a,opt_b,base_b --cases 99401,99402 --tokens 16
wsl -d Ubuntu -- python3 /mnt/d/code/sglang-V100-PVD-oasiskv/benchmark/analyze_pvd_oasis_serving.py /mnt/d/code/sglang-V100-PVD-oasiskv/artifacts/fresh_gpu_bank --comparison d-gpu-bank
wsl -d Ubuntu -- python3 /mnt/d/code/sglang-V100-PVD-oasiskv/benchmark/analyze_pvd_oasis_v_search.py /mnt/d/code/sglang-V100-PVD-oasiskv/artifacts/fresh_gpu_bank
wsl -d Ubuntu -- python3 /mnt/d/code/sglang-V100-PVD-oasiskv/benchmark/analyze_pvd_oasis_rpc.py /mnt/d/code/sglang-V100-PVD-oasiskv/artifacts/fresh_gpu_bank
```
