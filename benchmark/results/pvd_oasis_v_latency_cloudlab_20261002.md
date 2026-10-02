# Oasis 逐层 Decode：继续压低 V 检索固定开销

CloudLab，2026-10-02；分支 `codex/pvd-oasiskv`。
计划 commit `087e6c643`，阶段计时 `dcbcd1374`，实测实现 `c9512e4c0`。
正式实验 tag `oasis_v_latency_abba01`，诊断 tag `oasis_v_latency_profile01`。

## 结论

基线是上一轮已优化的两-head缓存搜索。在相同快图、原生 CAGRA、查询及 KV
预算、Oasis 配对逐层 Decode 下，V 单 rank 每层 batch 中位数
**14.374 → 7.526 ms**，缩短
**47.64%**，约
**1.91 倍加速**。客户端完成
**11.955 → 10.194 秒**，减少
**1.761 秒、14.73%**。
两个顺序、两条 Prompt 都有收益；八条正式请求的实际 token IDs／文本一致。

ID 映射从 5.119 ms 降到 62 μs，约
82.6 倍。这是局部阶段跨到几十微秒；完整候选处理仍需
1.318 ms，完整 V batch 仍需 7.526 ms。
**本轮没有实现整体检索十倍加速或亚毫秒服务时间。**

## 改动与诊断

一次完整路径诊断定位到候选 GPU 映射约5.214 ms、原生提交2.554 ms、finite-Q
校验1.121 ms；manager锁等待约0.001 ms。此前怀疑的跨-rank大锁不是本次测得
的主要耗时。单 GPU真实K/Q诊断约2.38 ms，在线七行GQA Q约14.77 ms；二者
工作量和并发环境不同，不能把它们的比值直接称作优化收益。

1. **有界 RMM pool。** 每 rank 的私有 PoolMemoryResource置于现有全局与每图
   limiter之下，initial_pool_size=0、maximum_pool_size=原先640 MiB原生预留。
   原生 CAGRA复用临时 CUDA内存，scope结束恢复此前 RMM资源。搜索参数、图
   邻接与过滤器不变。没有用无界资源绕过 limiter或修改进程默认 allocator。
2. **批量取回小候选集合，在 CPU 映射。** 两个已到达 head仍各自提交一次原生
   CAGRA，共享现有完成 fence。分数仍按原来的GPU float32 `scores + Q @ mean`
   恢复，然后批量复制有限候选；CPU执行chunk/head原生ID→Prompt ID映射，
   复用原来的有效范围、重复候选、有限分数校验与GQA union／稳定tie排序。
   不等待其他层，不额外生成Q。
3. **保留完成与生命周期证明。** Entry reader、查询、缓存输出、恢复分数和预算
   保留到现有全设备完成 fence之后。close等reader排空；未知完成隔离并保留
   owner／预算。结果处理的保守设备 fence仍存在。

默认关闭的开关：`--prompt-index-cagra-native-pool`（launcher `PVD_NATIVE_POOL=1`），
`--prompt-index-host-candidate-processing`（`PVD_HOST_CANDIDATES=1`）。后者要求
已有batched group search；本试验也开启既有partial group search。

RMM池复用与上限来自当前25.10的原生API；参考
[RMM PoolMemoryResource定义](https://github.com/rapidsai/rmm/blob/branch-25.10/python/rmm/rmm/pylibrmm/memory_resource.pyx)。

## 正式完整路径计时

以下各行独立取中位数，不能相加重建某个请求。缩短列负值代表增加。
原生提交＋完成和候选完整处理先在每条V观测内合并阶段，再取中位数。

| 指标 | 两-head缓存基线 | RMM pool＋CPU候选 | 缩短 |
|---|---:|---:|---:|
| V 单 rank 每层 batch 处理 | 14.374 ms | 7.526 ms | 47.64% |
| V manager，含锁等待 | 12.954 ms | 6.287 ms | 51.46% |
| 候选完整处理（逐条合并后取中位数） | 6.143 ms | 1.318 ms | 78.55% |
| 其中 ID 映射 | 5.119 ms | 0.062 ms | 98.79% |
| 原生提交＋完成（逐条合并后取中位数） | 2.729 ms | 1.992 ms | 26.99% |
| GPU finite-Q 校验 | 1.069 ms | 1.048 ms | 1.87% |
| D 层检索／交付 RPC 区间 | 43.891 ms | 35.295 ms | 19.58% |
| D 层 job service | 46.985 ms | 39.057 ms | 16.87% |
| 后续 Decode 每步累计层 KV 等待 | 556.487 ms | 422.846 ms | 24.02% |
| 后续 Decode 执行区间 | 646.570 ms | 528.868 ms | 18.20% |
| 执行区间扣除层等待 | 91.316 ms | 106.758 ms | -16.91% |
| EAGLE 后续提议 | 3.304 ms | 4.810 ms | -45.56% |
| 首个 Decode 执行 | 41.993 ms | 50.675 ms | -20.67% |
| D 初始化：Prompt seed＋初始 sparse banks | 1.611 s | 1.488 s | 7.61% |
| 客户端首个流事件 | 2.689 s | 2.570 s | 4.44% |
| 首个事件到完成 | 9.252 s | 7.624 s | 17.59% |
| 客户端完成 | 11.955 s | 10.194 s | 14.73% |

池化会把原先CUDA释放隐含的等待移动到显式native完成fence。单独的
native_submit从2.563→0.984 ms，而native_completion从0.115→1.000 ms；应该读
逐条合并的 **2.729→1.993 ms**，不能把提交下降61.6%当作原生GPU搜索快61.6%。
完整候选处理包含映射、GPU分数恢复与取回；新路径的恢复成本记在
candidate_download中，不能把score_restore日志变为0解释成计算消失。

V计时是服务端host wall：含锁、放置、原生调用、结果处理等；batch_total不含
单列的JSON解析，也不是纯CAGRA kernel或完整网络往返。D RPC区间包含检索、
resident handoff、交付控制和缓存安装，不能当作网络传输时间。
Decode执行包括EAGLE、配对目标前向、实际KV写回及普通采样，EAGLE是其子阶段。
扣除层等待仍包含拷贝、发布、同步等，不能叫纯Q生成时间。首个流事件包含P
提供的root token，不能叫首次D生成。

优化后D每步仍有 **422.8/528.9 ms** 层等待，约
**80.0%**。D的非等待区间和EAGLE提议没有改善。
两条Python后台worker、逐层RPC／交付控制及保守GPUDirect全设备同步仍保留。

### 顺序与单请求结果

| 顺序 | case99401 | case99402 |
|---|---:|---:|
| base_a | 11.913 s | 11.717 s |
| opt_a | 10.113 s | 10.552 s |
| opt_b | 9.981 s | 10.275 s |
| base_b | 11.997 s | 12.222 s |

正式结果代表服务预热后的请求；每臂两个相同warmups排除，原始记录全部保存。
每个请求的图门、初始全KV传输与private Prompt seed仍计入客户端时间。服务器／
模型冷启动及首次池扩容不在正式请求表内；没有证明冷启动同等收益。

## 公平性与完整路径证据

- P=node0 GPU0，V=node1 GPU0＋1，D=node2 GPU1，均V100S32GB；Qwen2.5-7B
  与同一专用EAGLE3，TP1、单请求、greedy、ignore_eos、cached_tokens=0。
- base_a→opt_a→opt_b→base_b；每臂重启P/V/D/Gateway，相同两条2159-token
  Prompt、16输出tokens，各配置四正式请求。实际输出IDs／文本hash四臂一致。
- P/D配置完全相同且overlap=true；唯一V处理差异为上述两个新开关。没有冻结
  Q、检索IDs、KV字节或teacher生成轨迹。查询与预算均保持原样。
- 相同快图：每rank14图、四合一、degree16=KNN14＋ring2，固定中心化；
  chunk256、prefix2048＋tail111，批量准备／提取、planned tail、固定视图、
  ahead capture、reuse scores、early final update；新Top16／fused edge write／
  stream completion关闭。保持自定义不可变KV边维护及native filtered CAGRA。
- itopk2048、Top4、capacity32/head、max_new16/head、D workers2、timeout60相同。
  P→D直送关闭，初始全KV仍经图门后P→V→D，并保留一次private Prompt seed。
- 每请求15次真实Decode、392个consumed layer futures、28初始＋392后续job，
  共840次双-rank搜索RPC（另有交付控制RPC）；没有减少Q或搜索次数。
  每配置3136条正式稳态V profile，全部2 items／14 Q rows。
  基线路径全部`grouped_cagra_partial_batched`，新路径全部带`_host`，没有回退。
- sparse逻辑payload/request中位数2,563,072/2,562,560 B，含初始化；初始全量
  逻辑payload两边均123,805,696 B。实际选择产生小量传输差异，没有冻结流量。
- 实际部署LF源hash匹配本地实现；argv、配置、原始输出／D/V日志和source hashes
  保留。运行退出0，cleanup_errors=[]、owned={}，三节点全部GPU最终0 MiB。

## 原生语义、召回与内存

**196 tests passed，2个显式原生GPU opt-in测试跳过。** 原生验证另走真实V100S／
cuVS25.10探针和上述完整服务。新增检查覆盖两个rank、完整／反序head子集、
缓存与预算归还，非法跨head／sentinel／重复ID／非有限分数及未知manager完成。
扩展旧backend测试时先暴露过过时的Resources(stream=...)测试double；补齐
实际stream_set边界后通过。这个失败没有改动生产stream绑定。

同一真实K/Q、不可变快图上，对全部28层、每rank两个head，冻结**一次原生
候选输出**分别走旧／新结果处理：token、page、float32分数和排序全部完全一致。
两rank `same_native_candidates_semantics_identical=true`；图hash与此前快图相同。

fixture每head只有**两行真实Q**（[56,2,128]），没有补造七行；在线则每KV head
七行GQA Q。池内ABBA、两次warmup＋三次正式repeat：

| 有pool、同图两-head probe | rank0 | rank1 |
|---|---:|---:|
| GPU候选处理 | 1.935 ms | 1.929 ms |
| CPU候选处理 | 1.466 ms | 1.462 ms |
| 原 exact Top4 GQA-union平均覆盖 | 0.999575 | 1.000000 |
| 新 exact Top4 GQA-union平均覆盖 | 0.999150 | 1.000000 |
| 最差单次覆盖，两路径 | 0.857143 | 1.000000 |

rank0在独立原生搜索之间仍有候选抖动：原路径一次、新路径两次观察到miss；
保留这些负面记录，不声称候选完全相同或总体质量等价。此前同一layer14重复
100次/模式已在旧／缓存路径都复现这种抖动。上述固定候选验证只证明结果处理
语义；两条合成Prompt输出相同也不是自然任务质量benchmark。

本轮原生probe每rank pool保留 **335,544,320 B（320 MiB）**，native最大值
671,088,640 B（640 MiB）受既有预留约束。关闭所有Entry后native live bytes为0，
但池仍保留物理内存到runtime退役／服务退出；不能把live=0叫物理显存已归还。
缓存输出仍为在线73,024 B/rank（离线两行Q64,064 B/rank）。这里不是整机显存
峰值测量。更广Prompt／长Decode／并发／TP2／压力与取消仍是后续gate。
两个新开关与Oasis实验模式均保持默认关闭。

## 下一步降低量级的判断

当前新路径仍有约1.99 ms原生提交／完成、1.05 ms GPU有限性校验、约1.25 ms
分数恢复／候选下载，以及RPC／HTTP工作。manager锁等待很小。
继续压低可先验证CPU不可变Q快照的有限性证明与放置、合并原生提交和候选
下载，或减少双GPU Python线程调度；每项都需要单独的原生API和质量／生命周期
验证。本次没有把其中任何一个推测记作已实现收益。

完整查询亚毫秒和D等待的数量级下降仍无实测支持；尤其D每层RPC区间约35 ms，
明显大于V单rank batch约7.5 ms。压低CAGRA本身后，跨节点完整路径仍有控制、
缓存安装、同步与排队成本，不能用单GPU检索时间代替Q→可用KV延迟。

## 复现与证据

```text
wsl -d Ubuntu -- python3 /mnt/d/code/sglang-V100-PVD-oasiskv/benchmark/run_pvd_oasis_serving_cloudlab.py --tag FRESH_TAG --comparison v-latency --arms base_a,opt_a,opt_b,base_b --cases 99401,99402 --tokens 16
wsl -d Ubuntu -- python3 /mnt/d/code/sglang-V100-PVD-oasiskv/benchmark/analyze_pvd_oasis_serving.py /mnt/d/code/sglang-V100-PVD-oasiskv/artifacts/FRESH_TAG --comparison v-latency
wsl -d Ubuntu -- python3 /mnt/d/code/sglang-V100-PVD-oasiskv/benchmark/analyze_pvd_oasis_v_search.py /mnt/d/code/sglang-V100-PVD-oasiskv/artifacts/FRESH_TAG
```

同名目录：`raw.tar.gz`完整正式／warmup输出和日志；`diagnostic.tar.gz`前置诊断；
`native.tar.gz`所有原生试验（含中间版本和负面结果）；精简summary、阶段数据、
source hashes、checkout身份、implementation与unit输出、最终GPU状态。
`host_pool_final`是最终纯CPU映射版本的原生gate；`host_nopool`／`host_pool`是
此前Torch CPU映射中间版本，不能把它们当作最终版本的严格pool-only在线消融。
正式ABBA比较的是两个新开关的组合，未做在线单开关归因。
