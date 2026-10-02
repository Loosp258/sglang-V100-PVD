# Oasis 逐层 Decode：V 检索缓存的完整路径收益

CloudLab，2026-10-02；分支 `codex/pvd-oasiskv`。
实测实现 commit：`005f874cc`；实验 tag：`oasis_v_search_abba01`。

## 结论

在保留当前快建图和 Oasis 配对 Decode 的条件下，把 **已经到达的两个
head** 接入 CAGRA 缓存搜索。八条正式请求中，客户端完成中位数
**13.597 → 11.995 秒**，减少 **1.602 秒、11.78%**。
两个比较顺序、两条 Prompt 均观察到收益；全部输出 IDs 和文本相同。

这个收益来自 V 搜索缓存；两边的 Decode 都是逐层重叠模式，不能据此声称
“Oasis 重叠相对串行配对”的收益已重新验证。查询数量、搜索宽度和 KV 预算
保持相同，搜索仍使用 native CAGRA。

## 改动

现有四合一搜索缓存要求四个 head 的 Q 全部到达。Oasis 每层每 rank 只有
两个 head，原来始终回退到逐项搜索。现在可直接处理已到达的 head 子集：

- 每 Entry、每图缓存四个不可变 head filter、SearchParams、输出缓冲；相邻
  两层交替使用同一个缓存。每个 head 的 Q 对应自己的原生过滤器与输出。
- 只提交当前两个 head 的原生 CAGRA 搜索，没有为其他 head 生成占位 Q，
  也不等待下一层。两次调用共享一次显式原生完成 fence。
- 保留原 head/token mapping、中心化分数恢复、GQA union、版本检查和
  结果处理完成 fence。缓存/输出消费串行保护，close 等 reader 排空。
- 缓存显存按 Entry 记账；shape 改变先关闭旧缓存并退预算。完成未知时保留
  reader、原生 owner、Q、缓存和 scratch，并隔离 manager。

新增开关默认关闭：`--prompt-index-partial-group-search`。
本轮 launcher 已有 `PVD_BATCHED_GROUP_SEARCH=1` 和
`PVD_GROUPED_EXACT_SEARCH=1`（让 HTTP 走原子 `search_many` 入口）；新增
`PVD_PARTIAL_GROUP_SEARCH=1` 才启用子集缓存。搜索仍走 native CAGRA。

## 完整路径实测

各列独立取中位数，不能相加构造单个请求的耗时。

| 指标 | 原搜索 | 两 head 缓存搜索 | 缩短 |
|---|---:|---:|---:|
| V 单 rank 每层 batch 处理阶段 | 20.039 ms | 13.995 ms | 30.16% |
| 其中 manager 处理，含锁等待 | 18.710 ms | 12.646 ms | 32.41% |
| D 层 job 的检索/交付 RPC 区间 | 50.567 ms | 43.590 ms | 13.80% |
| D 层后台 job service | 54.307 ms | 46.724 ms | 13.96% |
| 后续 Decode 步累计前台层等待 | 687.889 ms | 554.700 ms | 19.36% |
| 后续 Decode 步执行区间 | 764.654 ms | 650.540 ms | 14.92% |
| 执行区间扣除层等待 | 79.304 ms | 91.655 ms | 增加 |
| EAGLE 后续提议 | 2.809 ms | 3.195 ms | 增加 |
| D 初始化：Prompt seed＋初始 sparse banks | 1.717 s | 1.584 s | 7.71% |
| 客户端首个流事件 | 2.802 s | 2.671 s | 4.68% |
| 首个流事件到完成 | 10.803 s | 9.220 s | 14.65% |
| 客户端完成 | **13.597 s** | **11.995 s** | **11.78%** |

V 计时来自服务端 host wall 日志，包含锁等待、放置、原生调用、结果处理等，
不是纯 CAGRA GPU kernel 时间，也不含完整网络往返。D RPC 区间还包含
选择/驻留 bank handoff、交付控制和缓存安装等，不能解释成纯网络时间。

Decode 执行区间是 Scheduler 的 `PVD Oasis forward` 计时：包含 EAGLE 提议、
配对目标前向、真实 KV 写回和采样；EAGLE 单独计时是其子阶段，不应再相加。
扣除层等待的值包含多种计算、拷贝和同步开销，不是 Q 投影时间。本轮这部分
和 EAGLE 均没有加速。首个流事件包含 P 提供的首 token，不能当作首次 D 生成。

当前主要等待仍是逐层 KV：优化后约 **555 ms / 651 ms**，约占 85%。
本轮只优化 V，不拆分/重写 D 的后台流水线，也保留 GPUDirect 全设备 fence。

### 顺序与单请求结果

| 顺序 | case 99401 | case 99402 |
|---|---:|---:|
| 原搜索 A | 13.480 s | 13.419 s |
| 优化 A | 12.151 s | 12.246 s |
| 优化 B | 11.839 s | 11.433 s |
| 原搜索 B | 13.713 s | 14.067 s |

优化 A 比原搜索 A 分别少 1.329/1.173 秒；反序中优化 B 比原搜索 B 分别少
1.874/2.634 秒。存在运行间漂移，不能把 11.78% 当作所有请求的固定收益。
两种顺序的方向一致；这是两个合成 Prompt 上有限规模的端到端结果。

## 公平比较与路径证据

- P=node0，V=node1 GPU0＋GPU1，D=node2 GPU1，均为 V100S；P/D TP1，
  每次一个请求。每臂重启 P/V/D/Gateway，使用相同两条预热请求，均输出16 tokens。
- 正式 Prompt 为相同2159 tokens、case99401/99402，greedy、ignore_eos、
  cached_tokens=0。四臂每臂两请求，每配置四请求，共八请求。
- 两边都是当前 EAGLE3＋真实/预测 token 配对目标前向、逐层提交 Q 和等待 KV。
  所有 D config 完全相同且 `overlap=true`，请求预算与 timeout 完全相同。
- 相同快图：每 rank14图、四合一、degree16=14KNN＋ring2、固定中心化、
  chunk256/prefix2048/tail111、批量准备/提取/中心化、planned tail、固定视图、
  ahead capture、reuse scores、early final update。自定义不可变 KV 边维护，
  不走 native `cagra.extend`；native CAGRA 仍执行过滤搜索。
- 相同 `itopk_size=2048`、Top4、bank容量32/head、每步最多换入16/head、
  两个 D 后台 worker；P→D direct KV 关闭，初始 KV 仍经图门后 P→V→D。
- 两配置的唯一 V 运行开关差异是 partial-group-search。V 在新隔离 checkout
  `validation/pvd-oasis-v-search-20261002`，没有修改旧的搜索实验 checkout。
- 每请求15次实际 Decode 执行、392个 consumed lookahead layer futures，
  28个初始 KV bank job＋392个后续 job，共840次双-rank搜索 RPC。没有减少查询。
  每配置有3136条正式稳态 V batch，每条2 items/14 Q rows，完整计数验证通过。
  原路径均为 `grouped_cagra`，新路径均为 `grouped_cagra_partial_batched`。
- 没有冻结检索 IDs 或 Q/KV 字节；真实在线选择直接决定传输和 Decode。
  sparse payload/request中位数2,563,072/2,562,816 B（含初始化），初始完整
  KV逻辑 payload相同123,805,696 B，未减去初始全量传输成本。
- 实际源文件 hash、argv/config、预热、原始事件、D/V日志均保存。运行退出0；
  cleanup_errors为空、owned为空，三节点全部 GPU 最终均0 MiB。

## 召回、测试和限制

**150 tests passed**：配对因果隔离/真实提交、layer futures、native terminal/ACK
缓存协议、Prompt index/chunks，以及新增的子集/反序/跨相邻层缓存复用、版本
拒绝、shape替换、close/search交叉、完成未知下的预算与 owner 保留。

同一份真实模型 K/Q、同一不可变快图上的原生探针，分别测两 rank 全部28层的
两个 head。fixture 每 head 只有**两行真实 Q**，没有补造七行 Q；在线则为七行
GQA Q。因此离线探针用于核对子集语义与候选质量，不替代在线的七行测量。

| 离线 native manager，两 head | rank0 | rank1 |
|---|---:|---:|
| 原搜索中位数 | 3.142 ms | 3.114 ms |
| 缓存搜索中位数 | 2.364 ms | 2.341 ms |
| 原搜索 exact Top4 GQA-union 覆盖率均值 | 1.0000 | 1.0000 |
| 缓存搜索 exact Top4 GQA-union 覆盖率均值 | 0.9996 | 1.0000 |

rank0 在一次原生调用中有候选差异；保留了该负面记录，没有声称候选完全相同。
针对 layer14 再做每模式100次测量，同一图和相同 Q：原搜索/优化的两 head
平均 union覆盖率0.9893/0.9879，最差单次0.8571/0.8571，抖动在两边均出现。
rank1仍全部1.0。该结果没有明确系统性下降证据，也不构成质量等价证明或
生产召回保证。这里测的是 exact Top4 union覆盖率，不是 Top10逐查询召回。

完整路径的两条 Prompt 在四臂的 token IDs、文本 hash完全一致；这不是自然
任务质量 benchmark。更广 Prompt、长 Decode、并发、TP2、取消/未知完成压力
和物理显存峰值仍待验证。实验选项保持默认关闭。

缓存正常 shape 的声明 retained bytes：在线七行Q为73,024 B/rank，离线两行
Q为64,064 B/rank；缓存遵循 reader/Entry 退役，不是额外无界全局缓存。
这不是整机物理显存峰值。初始全量 KV 和一次 private Prompt seed pass仍存在；
本次没有验证论文的稀疏启动或显存容量收益。

## 复现与证据

```text
wsl -d Ubuntu -- python3 /mnt/d/code/sglang-V100-PVD-oasiskv/benchmark/run_pvd_oasis_serving_cloudlab.py --tag FRESH_TAG --comparison v-search --arms base_a,opt_a,opt_b,base_b --cases 99401,99402 --tokens 16
wsl -d Ubuntu -- python3 /mnt/d/code/sglang-V100-PVD-oasiskv/benchmark/analyze_pvd_oasis_serving.py /mnt/d/code/sglang-V100-PVD-oasiskv/artifacts/FRESH_TAG --comparison v-search
wsl -d Ubuntu -- python3 /mnt/d/code/sglang-V100-PVD-oasiskv/benchmark/analyze_pvd_oasis_v_search.py /mnt/d/code/sglang-V100-PVD-oasiskv/artifacts/FRESH_TAG
```

同名结果目录保存：`raw.tar.gz`完整在线记录、`native.tar.gz`原生质量/抖动重测、
精简summary、V处理summary、源hash、checkout heads、测试输出和最终GPU状态。
remote HEAD 是启动门的身份；部署文件 hash才是实际源码证据。
