# Oasis KV 交付：原始注册区直接 sparse batch PUT公平对照

CloudLab 2026-10-03；完整快图＋V/CAGRA＋Oasis 配对逐层 Decode。
日期按首个正式请求started_unix换算UTC+8：2026-10-03T03:48:17+08:00。

## 结果

客户端完成中位数 **9.938→11.607 s（+16.79%）**；
每请求后续Decode累计 KV 等待 **5815.576→7176.120 ms**。
每模式四个正式请求、只覆盖两个Prompt；3136个层/rank profile不视作独立请求样本。两个独立优化的结果不能相加。新选项仍默认关闭。

## 公平性

base_a→opt_a→opt_b→base_b，各臂重启全角色，排除相同两次 warmup；两个2159-token Prompt、16输出token、greedy/ignore_eos。
四合一快图、degree16、prefix2048＋tail111、itopk2048、Top4、capacity32、max_new16、workers2保持一致。
V固定 pool/host candidates/GPU finite-Q proof；Triton打包关闭；D显式 reuse_io=false。初始KV仍图门后P→V→D，private Prompt seed计入客户端。
唯一变量为V direct_sparse_batch_put；D配置完全相同。保留device readiness同步，opt从原pool MR按component-major精确地址直接batch发送，至多128片段。
八次Prompt、实际输出ID和文本一致，cached_tokens=0；每请求420jobs/840搜索RPC，每模式3136稳态查询profile，均2items/14Qrows。
部署 bundle 中324个文件的实际哈希与V/D serving source gate一致；源码身份使用记录的部署包，没有读取后来编辑的工作树。
所有接收写入保留完整身份、精确native终态字节证明、安装后ACK及UNKNOWN保留；最终 owned={}、cleanup_errors=[]、六张GPU归零。

## 完整路径时间

| 指标（独立中位数） | staging packed PUT | 原始 Entry scatter PUT | 相对变化 |
|---|---:|---:|---:|
| V查询wall/rank/层 | 7.067 ms | 7.544 ms | +6.76% |
| D search_many | 11.658 ms | 11.696 ms | +0.33% |
| D整层检索/交付RPC | 33.558 ms | 38.248 ms | +13.98% |
| 层worker完整service | 37.195 ms | 42.446 ms | +14.12% |
| 每请求后续Decode累计KV等待 | 5815.576 ms | 7176.120 ms | +23.39% |
| 每步逐层等待和中位数 | 407.032 ms | 488.911 ms | +20.12% |
| 后续Decode执行 | 512.782 ms | 587.763 ms | +14.62% |
| 客户端TPOT均值（15个流式间隔） | 489.666 ms | 573.654 ms | +17.15% |
| D每token KV等待均值（后14步） | 413.390 ms | 515.459 ms | +24.69% |
| 首个客户端事件 | 2.573 s | 3.039 s | +18.09% |
| 客户端完成 | 9.938 s | 11.607 s | +16.79% |

每请求累计等待先对该请求step>0的各次forward等待求和，再对每模式四个正式请求取中位数；
每步逐层等待和另按forward取中位数。两者范围不同，不能将每步约数百毫秒当作整个请求的累计等待。

## 实际 token 间隔与后台任务容量

客户端TPOT按每请求实际16个token事件的(last-first)/15计算，排除TTFT；每模式4请求，共60个输出间隔。
首token来自P Prefill；首个D forward使用已初始化bank，KV等待为零。另报告其后14个间隔，避免短Decode的首步降低均值。
后续KV等待按step>0累计等待/56计算算术平均；以下请求TPOT中位数、单步等待中位数与算术平均分别标明。

| 指标 | workers=2 | workers=4 | 相对变化 |
|---|---:|---:|---:|
| 客户端TPOT算术平均 | 489.666 ms | 573.654 ms | +17.15% |
| 客户端TPOT请求中位数 | 491.767 ms | 571.163 ms | +16.14% |
| 后续14间隔TPOT算术平均 | 520.952 ms | 611.530 ms | +17.39% |
| 后续14间隔TPOT请求中位数 | 523.549 ms | 608.772 ms | +16.28% |
| 后续每token KV等待算术平均 | 413.390 ms | 515.459 ms | +24.69% |
| 后续每token KV等待单步中位数 | 407.032 ms | 488.911 ms | +20.12% |
| 层任务service算术平均 | 37.344 ms | 43.758 ms | +17.18% |
| 层任务queue算术平均 | 482.918 ms | 567.295 ms | +17.47% |
| 实际service总和/workers/56 | 522.822 ms | 612.618 ms | +17.18% |

实际callback峰值并发：baseline [2, 2, 2, 2]，optimized [2, 2, 2, 2]；每请求392个已消费任务，每模式1568个。
按实际start/ready时间戳重建的worker占用率为99.726%/99.743%；
消费者需要前已经READY的比例为3.189%/10.778%。
占用率=sum(service)/(workers×已消费callback时间窗)，包含RPC阻塞，不能解释成CPU或GPU利用率。
service总和/workers/56仅描述本次观察到的任务容量，未假定增加并发后service保持不变，也不是端到端时延下限或未来加速预测。
客户端事件、28层等待、worker callback与V单rank查询包含不同范围；独立中位数不能相加解释客户端TPOT。

## 实际交付子阶段与调用

以下时间为已完成的稳态missing-rank交付，排除初始28层bank priming；注册池首次注册仍计入客户端初始化。
控制提交时间先对每次交付的reserve/start/combined求和，再计算中位数。各阶段独立中位数不能相加重建请求。

| 子阶段 | staging packed PUT | 原始 Entry scatter PUT |
|---|---:|---:|
| 接收区准备 | 1.075 ms | 1.076 ms |
| 物理分配 | 0.018 ms | 0.017 ms |
| 物理注册 | 0.913 ms | 0.917 ms |
| poll RPC | 0.000 ms | 1.910 ms |
| 安装后ACK | 1.361 ms | 0.787 ms |
| 接收区close | 0.412 ms | 0.415 ms |
| GPU→CPU缓存copy | 0.286 ms | 0.312 ms |
| reserve/start控制提交 | 15.839 ms | 16.403 ms |

实际总调用计数覆盖每模式四个正式请求，含priming及最终物理池退休；poll为实测次数，start直接返回READY时可为零。

| 总计数 | staging packed PUT | 原始 Entry scatter PUT |
|---|---:|---:|
| missing-rank交付 | 3084 | 3084 |
| 物理register | 3084 | 3084 |
| 物理unregister | 3084 | 3084 |
| reserve RPC | 3084 | 3084 |
| start RPC | 3084 | 3084 |
| combined RPC | 0 | 0 |
| poll RPC | 6 | 3085 |
| ACK RPC | 3084 | 3084 |

## 四臂与流量

| 执行順序 | case 99401 | case 99402 |
|---|---:|---:|
| base_a | 9.988s | 10.185s |
| opt_a | 11.718s | 11.411s |
| opt_b | 11.955s | 11.495s |
| base_b | 9.601s | 9.888s |

前后同配置arm的客户端中位数变化：baseline -3.39%，optimized +1.39%；保留顺序漂移，不据八个请求宣称统计显著性。
初始逻辑全KV为123805696/123805696 B/request；稀疏payload中位数2562560/2563072 B/request。
CAGRA候选允许原生抖动；没有冻结missing集合。查询、D RPC和交付时间包含不同范围，不把差值当作纯网络或纯native传输时间。

## 验证与范围

CPU gate实际结果：`176 passed, 6 skipped, 1 warning in 8.50s`；完整输出见gate.tar.gz与gate_count_record.json。
原生本地Mooncake gate通过48个精确字节案例，两GPU原始pool MR、多层、多head、非连续token、非零Entry偏移与末页scatter，0 staging注册；caller初态GPU0，发送使用显式source设备上下文；本地session gate与线上跨节点RDMA对照是独立观测。
单元异常测试采用CPU policy double；没有注入真实RDMA/CUDA故障。长Decode、多请求并发、TP2、更多Prompt质量与服务显存峰值仍开放。
客户端完成与最终安全退休分别验证；本次子阶段profile不能单独证明整个流水线的重叠收益。

## 证据与复现

同名目录保存完整raw.tar.gz、gate.tar.gz、紧凑summary/实际IO计数、V/RPC分解、部署来源、GPU归零与LF便携manifest。
前置失败尝试另存failed_prelaunch.tar.gz：两次源码门禁脚本错误都在服务启动前发生，未产生正式请求。修正P检查范围，并以第一步记录的独立P源码版本固定四臂；未重新部署或修改P。

## 采用决定

这次直接scatter没有收益，保持默认关闭；下一步D GPU直接安装对照继续使用原staging packed PUT。
TPOT均值489.666→573.654 ms/token；KV等待均值413.390→515.459 ms/token。
opt虽然取消V staging复制/注册，但每个缺失head/token变成独立256B源切片。当前绑定下，small-write batch提交成本与额外poll使完整交付变慢；此解释来自实际start/poll和native计数，不能把这些含同步/控制的wall时间当成纯网络延迟。
direct_sparse_summary.json逐arm/rank核对warmup后新增batch/slice数与D真实交付行数，保留adapter native计时（含初始full-KV fan-in）、唯一原pool MR及退休证明。
gate.tar.gz也保存本地CPU初次fixture错误、缺Triton环境失败与后续修正后的原始输出。CloudLab正式CPU和native gate才作为本次资格。

```text
wsl -d Ubuntu -- python3 /mnt/d/code/sglang-V100-PVD-oasiskv/benchmark/run_pvd_oasis_serving_cloudlab.py --tag fresh_direct_sparse --comparison v-direct-sparse --arms base_a,opt_a,opt_b,base_b --cases 99401,99402 --tokens 16
wsl -d Ubuntu -- python3 /mnt/d/code/sglang-V100-PVD-oasiskv/benchmark/analyze_pvd_oasis_serving.py /mnt/d/code/sglang-V100-PVD-oasiskv/artifacts/fresh_direct_sparse --comparison v-direct-sparse
wsl -d Ubuntu -- python3 /mnt/d/code/sglang-V100-PVD-oasiskv/benchmark/analyze_pvd_oasis_v_search.py /mnt/d/code/sglang-V100-PVD-oasiskv/artifacts/fresh_direct_sparse
wsl -d Ubuntu -- python3 /mnt/d/code/sglang-V100-PVD-oasiskv/benchmark/analyze_pvd_oasis_rpc.py /mnt/d/code/sglang-V100-PVD-oasiskv/artifacts/fresh_direct_sparse
```
