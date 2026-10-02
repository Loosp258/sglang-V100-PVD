# V 私有 CPU Q 校验：原生验证通过，在线没有收益

CloudLab，2026-10-02，`codex/pvd-oasiskv`。计划`1c6c97707`，实现`9869bfbbd`。
正式 tag `oasis_v_host_query_abba01`；基线为已验证的 RMM pool＋CPU候选处理`c9512e4c0`。

## 结论

**保留默认关闭，不把这项叠加到后续已获益路径。** GPU有限性检查单阶段变快，
但在线V完整查询和客户端没有收益：V7.268→7.482 ms，客户端10.158→10.324 s。
离线两行真实Q约1.48→1.42 ms不能用来推断在线七行Q/双GPU服务收益。

## 实现与生命周期

`FilteredSearchWorkspace.search_host()`先检查CPU float32形状/连续性/heads/owner，
detach并clone私有快照，再用NumPy检查同一份快照的有限性；随后同步复制到backend设备。
接口不接受调用者的unchecked布尔值或外部proof。GPU调用保留原有限性检查。
原生参数、head过滤、GPU float32均值恢复、逻辑映射和GQA合并不变。
CPU源、设备Q、workspace、reader与scratch在完成前保留；复制/提交/完成异常统一drain，
未知完成隔离并保留owners。经理为额外CPU快照显式预留空间，在线7行Q为7168B/batch。
最终保守设备fence保留。拷贝使用默认non_blocking=False；[PyTorch接口](https://docs.pytorch.org/docs/2.8/generated/torch.Tensor.to.html)。

默认关闭开关：`--prompt-index-host-query-validation`／`PVD_HOST_QUERY_VALIDATION=1`。

## 完整路径公平对照

base_a→opt_a→opt_b→base_b，每臂重启全部服务；各两次相同warmup排除，
两条2159-token Prompt、16输出tokens，各配置4次正式请求。全部8次实际token IDs和文本一致。
P=node0 GPU0、V=node1 GPU0+1、D=node2 GPU1；Qwen2.5-7B＋同一专用EAGLE3、TP1、greedy、ignore_eos。
P/D配置完全相同，Oasis实际/预测token配对逐层前向、workers2；无P→D直送。
初始全KV仍图门后P→V→D，private Prompt seed计入客户端；冷启动不在正式表内。
同一快图：每rank14图、四合一、degree16=KNN14+ring2、固定均值，prefix2048+tail111；
同一native filtered CAGRA itopk2048、Top4、capacity32、max_new16，查询数量不变。
每请求15次实际Decode、28初始+392后续job，840双rank搜索RPC；每模式3136稳态V profile，均2items/14Qrows，无回退/重试。

| 指标（独立中位数） | 原GPU校验 | CPU快照校验 | 后者变化 |
|---|---:|---:|---:|
| V每rank每层batch | 7.268 ms | 7.482 ms | +2.94% |
| V manager | 6.117 ms | 6.105 ms | -0.19% |
| 有限性检查子阶段 | 1.074 ms | 0.034 ms | -96.83% |
| 原生提交+完成（逐条合并） | 1.975 ms | 1.975 ms | -0.03% |
| 候选完整处理（逐条合并） | 1.208 ms | 1.412 ms | +16.84% |
| D search_many总范围 | 11.862 ms | 12.151 ms | +2.44% |
| D层检索/交付范围 | 34.075 ms | 35.004 ms | +2.73% |
| 每步累计层KV等待 | 413.382 ms | 436.584 ms | +5.61% |
| 后续Decode执行 | 515.070 ms | 529.229 ms | +2.75% |
| 客户端完成 | 10.158 s | 10.324 s | +1.64% |

中位数不能相加重建请求。有限性子阶段变快不是整个检索变快；CPU快照、放置、
候选恢复与线程调度仍有成本。本实验不对哪个未分解成本导致总时间差作因果断言。

| 顺序 | case99401 | case99402 |
|---|---:|---:|
| base_a | 9.925s | 10.390s |
| opt_a | 9.630s | 10.487s |
| opt_b | 10.161s | 10.563s |
| base_b | 9.527s | 10.410s |

两case全部cached_tokens=0。初始逻辑KV123805696B/request相同；sparse逻辑payload中位数
2563072/2562816B，来自实际选择，未冻结流量。

## 原生质量与测试

**226 passed，2个显式原生opt-in测试跳过；另做真实V100S/cuVS25.10验证。**
覆盖非finite、shape/dtype/device、反序heads、调用者alias修改、梯度detach、复制失败、
unknown双源/reader/预算保留与普通GPU验证。两rank、28层实际设备Q逐值等于已证明CPU快照；
固定一次原生候选的ID/page/分数/排序处理完全一致，图hash与上一轮相同。
fixture真实形状[56,2,128]；没有伪造七行Q。ABBA暖机两轮、正式三轮；每轮均含完整四臂顺序：

| 原生离线 | rank0基线/新 | rank1基线/新 |
|---|---:|---:|
| 两head ms | 1.472/1.417 | 1.486/1.426 |
| exact Top4 union平均覆盖 | 0.999575/0.999150 | 1.0/1.0 |
| 最差观察覆盖 | 0.857143/0.857143 | 1.0/1.0 |

rank0独立原生调用仍抖动，基线一次/新路径两次miss；固定候选语义不等于所有搜索候选/质量等价。
Pool保留320MiB/rank，原640MiB预留内；Entry close原生live=0而pool继续保留到runtime退出。
完整线上显存峰值、更多自然Prompt、长Decode、TP2、并发/压力/取消仍未验证。

## 搜索之后的瓶颈与API边界

上一轮完整日志重算确认：D search_many约12.151ms、V约7.526ms；D整层约35.295ms。
两个D worker占用约99.7%，后续Decode接近两worker服务吞吐下限。
无remote miss层约12.429ms，有miss层约35.364ms；两组分布不同，不能严格相减当作因果KV传输时间。
每job会关闭HTTP sessions，下一job失去跨层keepalive；控制连接在单层内已经复用。
后续请求级连接复用需要单独在线对照，不能归因所有剩余时间到TCP握手。

cuVS25.10的Python虽接受通用Prefilter，但CAGRA的C API和公开C++ dispatcher仅支持none/bitset。
所以当前依赖下不能直接把两个head用per-query bitmap合成一次调用。
[C API分派](https://github.com/rapidsai/cuvs/blob/branch-25.10/cpp/src/neighbors/cagra_c.cpp#L213)、
[C++分派](https://github.com/rapidsai/cuvs/blob/branch-25.10/cpp/src/neighbors/cagra.cuh#L323)。
本轮未改原生算法/依赖，也未进行确定不支持的bitmap GPU试错。

## 证据与复现

同名目录保留raw.tar.gz（全部正式/warmup输出/日志/配置/argv/source hashes）、native.tar.gz、
完整unit、精简summary、逐阶段/RPC汇总和实现身份。服务停止，owned={}、cleanup_errors=[]，三节点6GPU0MiB。

```text
wsl -d Ubuntu -- python3 /mnt/d/code/sglang-V100-PVD-oasiskv/benchmark/run_pvd_oasis_serving_cloudlab.py --tag fresh_tag --comparison v-host-query --arms base_a,opt_a,opt_b,base_b --cases 99401,99402 --tokens 16
wsl -d Ubuntu -- python3 /mnt/d/code/sglang-V100-PVD-oasiskv/benchmark/analyze_pvd_oasis_serving.py /mnt/d/code/sglang-V100-PVD-oasiskv/artifacts/fresh_tag --comparison v-host-query
wsl -d Ubuntu -- python3 /mnt/d/code/sglang-V100-PVD-oasiskv/benchmark/analyze_pvd_oasis_v_search.py /mnt/d/code/sglang-V100-PVD-oasiskv/artifacts/fresh_tag
wsl -d Ubuntu -- python3 /mnt/d/code/sglang-V100-PVD-oasiskv/benchmark/analyze_pvd_oasis_rpc.py /mnt/d/code/sglang-V100-PVD-oasiskv/artifacts/fresh_tag
```

证据清单对JSON/TXT采用LF规范化hash；tar.gz按原始bytes校验，兼容Git跨平台换行。
