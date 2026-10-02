# Oasis KV 交付：预取 workers 2→4公平对照

CloudLab 2026-10-03；完整快图＋V/CAGRA＋Oasis 配对逐层 Decode。
日期按首个正式请求started_unix换算UTC+8：2026-10-03T00:51:45+08:00。

## 结果

**本轮没有证明端到端收益，保持默认 workers=2。** 后续每token KV等待算术平均427.322→412.722 ms（-3.42%），但标准客户端TPOT算术平均499.343→509.332 ms（+2.00%）。

客户端完成中位数 **9.987→10.211 s（+2.24%）**；
每请求后续Decode累计 KV 等待 **5901.056→5758.343 ms**。
每模式四个正式请求、只覆盖两个Prompt；3136个层/rank profile不视作独立请求样本。本实验与之前的合并RPC、接收注册复用实验不能叠加计算收益；默认workers仍为2。

## 公平性

base_a→opt_a→opt_b→base_b，各臂重启全角色，排除相同两次 warmup；两个2159-token Prompt、16输出token、greedy/ignore_eos。
四合一快图、degree16、prefix2048＋tail111、itopk2048、Top4、capacity32、max_new16保持一致；唯一变量为每请求预取workers=2/4。
V固定 pool/host candidates/GPU finite-Q proof；Triton打包关闭；D显式 reuse_io=false。初始KV仍图门后P→V→D，private Prompt seed计入客户端。
所有臂 combine_reserve_start=false、reuse_receive_slots=false、reuse_io=false；原生serving实现相同，增加worker无需改动交付协议。
八次Prompt、实际输出ID和文本一致，cached_tokens=0；每请求420jobs/840搜索RPC，每模式3136稳态查询profile，均2items/14Qrows。
部署 bundle 中55个文件的实际哈希与V/D serving source gate一致；源码身份使用记录的部署包，没有读取后来编辑的工作树。
计划commit `8aa78b903`、runner commit `d229f8bc3`、分析/证据helper commit `d56c38dda`；serving冻结实现为`9b8b5dc0c`。
共享部署包SHA256：`e004d72d10f4816d50ced8ded13755d17ed5ae30822f0b4dfa9dcd2479527e51`。远端checkout HEAD另行保留，运行实现以部署包和V/D实际source hashes对齐为准。
所有接收写入保留完整身份、精确native终态字节证明、安装后ACK及UNKNOWN保留；最终 owned={}、cleanup_errors=[]、六张GPU归零。

## 完整路径时间

| 指标（独立中位数） | workers=2 | workers=4 | 相对变化 |
|---|---:|---:|---:|
| V查询wall/rank/层 | 7.387 ms | 12.312 ms | +66.68% |
| D search_many | 11.932 ms | 19.349 ms | +62.16% |
| D整层检索/交付RPC | 34.159 ms | 70.381 ms | +106.04% |
| 层worker完整service | 38.009 ms | 77.313 ms | +103.41% |
| 每请求后续Decode累计KV等待 | 5901.056 ms | 5758.343 ms | -2.42% |
| 每步逐层等待和中位数 | 417.095 ms | 408.046 ms | -2.17% |
| 后续Decode执行 | 519.457 ms | 530.351 ms | +2.10% |
| 首个客户端事件 | 2.544 s | 2.620 s | +3.01% |
| 客户端完成 | 9.987 s | 10.211 s | +2.24% |

每请求累计等待先对该请求step>0的各次forward等待求和，再对每模式四个正式请求取中位数；
每步逐层等待和另按forward取中位数。两者范围不同，不能将每步约数百毫秒当作整个请求的累计等待。
V查询与D search_many为每模式3136个稳态rank/层调用的pooled中位数；层worker/RPC为1568个稳态层任务的pooled中位数。客户端完成/首事件为四个请求的中位数；后续Decode执行与单步等待为56次forward的pooled中位数。
这里V查询7.387→12.3125 ms来自pooled稳态调用；独立审计中的“请求中位数再取中位数”7.59125→12.509 ms是另一统计量，未用于该表。

## 实际 token 间隔与后台任务容量

客户端TPOT按每请求实际16个token事件的(last-first)/15计算，排除TTFT；每模式4请求，共60个输出间隔。
首token来自P Prefill；首个D forward使用已初始化bank，KV等待为零。另报告其后14个间隔，避免短Decode的首步降低均值。
后续KV等待按step>0累计等待/56计算算术平均；以下请求TPOT中位数、单步等待中位数与算术平均分别标明。

| 指标 | workers=2 | workers=4 | 相对变化 |
|---|---:|---:|---:|
| 客户端TPOT算术平均 | 499.343 ms | 509.332 ms | +2.00% |
| 客户端TPOT请求中位数 | 496.238 ms | 506.533 ms | +2.07% |
| 后续14间隔TPOT算术平均 | 531.238 ms | 541.488 ms | +1.93% |
| 后续14间隔TPOT请求中位数 | 527.759 ms | 538.444 ms | +2.02% |
| 后续每token KV等待算术平均 | 427.322 ms | 412.722 ms | -3.42% |
| 后续每token KV等待单步中位数 | 417.095 ms | 408.046 ms | -2.17% |
| 层任务service算术平均 | 38.091 ms | 77.652 ms | +103.86% |
| 层任务queue算术平均 | 492.415 ms | 458.177 ms | -6.95% |
| 实际service总和/workers/56 | 533.278 ms | 543.563 ms | +1.93% |

实际callback峰值并发：baseline [2, 2, 2, 2]，optimized [4, 4, 4, 4]；每请求392个已消费任务，每模式1568个。
按实际start/ready时间戳重建的worker占用率为99.750%/99.691%；
消费者需要前已经READY的比例为3.763%/32.015%。
占用率=sum(service)/(workers×已消费callback时间窗)，包含RPC阻塞，不能解释成CPU或GPU利用率。
service总和/workers/56仅描述本次观察到的任务容量，未假定增加并发后service保持不变，也不是端到端时延下限或未来加速预测。
客户端事件、28层等待、worker callback与V单rank查询包含不同范围；独立中位数不能相加解释客户端TPOT。

### 为什么等待略少，TPOT反而增加

同一份56次后续forward日志，逐次计算`total_ms-wait_ms`再取算术平均：前台非等待部分101.919→125.673 ms。等待减少14.600 ms，但非等待部分增加23.754 ms，forward总时间529.241→538.395 ms，增加9.154 ms。这是同一观测的分解，没有相减独立中位数。

实际callback峰值达到了4，单层service算术平均也从38.091升到77.652 ms；实际service总和/workers/56从533.278升到543.563 ms/token。增加worker后，单任务处理时间没有保持不变，整体任务容量没有改善。

独立只读审计在相同稳态范围观察到V锁等待与候选下载变长。以下为每请求784个rank查询、每模式3136次调用的call-weighted算术平均，排除每请求最先的56次初始bank调用：

| V阶段算术平均 | workers=2 | workers=4 |
|---|---:|---:|
| 查询batch wall | 7.682 ms | 13.190 ms |
| manager锁等待 | 0.092 ms | 1.738 ms |
| candidate download | 1.424 ms | 2.959 ms |
| native submit | 1.002 ms | 1.021 ms |
| native completion | 0.999 ms | 1.017 ms |

这些数据提供了并发竞争增加的线索，不能证明某个CUDA全设备同步是唯一原因，也不能把阶段均值相加重建客户端时延。原生CAGRA submit/completion没有出现与完整service相当的翻倍。
可重算脚本与完整JSON分别保存在`raw.tar.gz:oasis_workers_abba01/independent_worker_audit.py`和`independent_worker_audit.json`；JSON记录24份输入LF hashes、parser hash、每arm/每请求统计及full840/steady784的独立范围。

## 实际交付子阶段与调用

以下时间为已完成的稳态missing-rank交付，排除初始28层bank priming；注册池首次注册仍计入客户端初始化。
控制提交时间先对每次交付的reserve/start/combined求和，再计算中位数。各阶段独立中位数不能相加重建请求。

| 子阶段 | workers=2 | workers=4 |
|---|---:|---:|
| 接收区准备 | 1.063 ms | 1.094 ms |
| 物理分配 | 0.018 ms | 0.019 ms |
| 物理注册 | 0.908 ms | 0.921 ms |
| poll RPC | 0.000 ms | 0.000 ms |
| 安装后ACK | 1.361 ms | 1.970 ms |
| 接收区close | 0.413 ms | 0.415 ms |
| GPU→CPU缓存copy | 0.284 ms | 0.627 ms |
| reserve/start控制提交 | 16.246 ms | 41.366 ms |

实际总调用计数覆盖每模式四个正式请求，含priming及最终物理池退休；poll为实测次数，start直接返回READY时可为零。

| 总计数 | workers=2 | workers=4 |
|---|---:|---:|
| missing-rank交付 | 3084 | 3084 |
| 物理register | 3084 | 3084 |
| 物理unregister | 3084 | 3084 |
| reserve RPC | 3084 | 3084 |
| start RPC | 3084 | 3084 |
| combined RPC | 0 | 0 |
| poll RPC | 8 | 5 |
| ACK RPC | 3084 | 3084 |

## 四臂与流量

| 执行順序 | case 99401 | case 99402 |
|---|---:|---:|
| base_a | 10.038s | 10.419s |
| opt_a | 9.802s | 9.829s |
| opt_b | 11.035s | 10.594s |
| base_b | 9.937s | 9.731s |

前后同配置arm的客户端中位数变化：baseline -3.86%，optimized +10.18%；保留顺序漂移，不据八个请求宣称统计显著性。
顺序漂移大于本轮汇总2.24%的客户端差异，不能把等待减少3.42%称为稳定收益。按同Prompt两次请求的客户端完成算术平均，case99401变长4.32%，case99402变长1.35%；质量结论只覆盖本轮两个Prompt的16-token输出一致性。
初始逻辑全KV为123805696/123805696 B/request；稀疏payload中位数2562304/2562304 B/request。
CAGRA候选允许原生抖动；没有冻结missing集合。查询、D RPC和交付时间包含不同范围，不把差值当作纯网络或纯native传输时间。
独立审计发现同case实际payload存在微小差异：99401最多差1024 B/2行（0.039920%），99402差512 B/1行（0.019996%）。保持一致的是预算、查询次数、Prompt和输出；没有声称各臂稀疏payload逐byte相同。各次成功交付仍由现有协议要求精确native终态字节证明；审计profile仅能另行核对manifest bytes/rows、ACK/close与注销计数，未单独记录NIC终态。

## 验证与范围

CPU gate实际结果：`512 passed, 4 skipped, 1 warning in 8.13s`；完整输出见gate.tar.gz与gate_count_record.json。
原生本地Mooncake gate通过48个精确字节案例，两个执行器复用四个物理MR并安全注销；本地session gate与线上跨节点RDMA对照是独立观测。
接收record集成重复gate：`9 passed, 1 warning in 2.14s`；这九项已包含在完整CPU gate中，不相加为新的独立测试。
worker资格/生命周期CPU复核：`48 passed, 1 warning in 11.29s`；9个相关serving文件LF哈希与冻结部署包一致，完整证据保存在worker_cpu_gate.tar.gz。
这48项是已有CPU测试的重复复核，不与共享512项CPU gate或48个原生字节案例相加为新的独立测试数；它不证明原生四worker故障场景。线上callback实际并发另由start/ready时间戳验证。
单元异常测试采用CPU policy double；没有注入真实RDMA/CUDA故障。长Decode、多请求并发、TP2、更多Prompt质量与服务显存峰值仍开放。
客户端完成与最终安全退休分别验证；本次子阶段profile不能单独证明整个流水线的重叠收益。

## 证据与复现

同名目录保存完整raw.tar.gz、gate.tar.gz、紧凑summary/实际IO计数、V/RPC分解、部署来源、GPU归零与LF便携manifest。
raw同时保存离线helper证据检查和独立worker审计；worker_cpu_gate.tar.gz保存已有CPU资格/生命周期复核。它们分别验证证据与CPU生命周期，不计为新的GPU性能样本。

```text
wsl -d Ubuntu -- python3 /mnt/d/code/sglang-V100-PVD-oasiskv/benchmark/run_pvd_oasis_serving_cloudlab.py --tag fresh_workers --comparison v-workers --arms base_a,opt_a,opt_b,base_b --cases 99401,99402 --tokens 16
wsl -d Ubuntu -- python3 /mnt/d/code/sglang-V100-PVD-oasiskv/benchmark/analyze_pvd_oasis_serving.py /mnt/d/code/sglang-V100-PVD-oasiskv/artifacts/fresh_workers --comparison v-workers
wsl -d Ubuntu -- python3 /mnt/d/code/sglang-V100-PVD-oasiskv/benchmark/analyze_pvd_oasis_v_search.py /mnt/d/code/sglang-V100-PVD-oasiskv/artifacts/fresh_workers
wsl -d Ubuntu -- python3 /mnt/d/code/sglang-V100-PVD-oasiskv/benchmark/analyze_pvd_oasis_rpc.py /mnt/d/code/sglang-V100-PVD-oasiskv/artifacts/fresh_workers
```
