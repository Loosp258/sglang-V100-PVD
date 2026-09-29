# 只凭 Prompt 长度决定 P→V 切分：CloudLab 2026-09-30

## 结论

P 现在可在知道 Prompt token 数 `N` 时，直接计算所有实际可触发的单次切点，
预测双 V rank 的图 READY 时间，并决定是否使用“首图＋一次 extend”。请求期间
不逐切点试运行；耗时曲线来自事先完成的 CloudLab 校准。

当前 V100S、512-token Prefill chunk、每图四个 head、精确 degree 16 的校准
范围是 **509–2159 token**。模型在 `N=2159` 时给出的名义最优切点是
`1024`，预测双 rank READY 比完整图早 **85 ms**。但本轮同配置的在线实测
中，切分收益的最大配对预测误差是 **745 ms**。采用额外 50 ms 的保护量后，
需要预测收益超过 **795 ms** 才实际切分，因此目前测试长度均选完整图。
这个结论是“当前没有可靠到足以启用切分的收益”，不是证明所有长度的
最优切点都是零。

旧版按长度查表的策略只比较了各长度的一个在线切点。它不能推断未测
长度，也不能证明已测长度的切点最优；本次用参数模型取代了这项决策。

## 决策计算

每个 rank 离线拟合三条曲线：到达时刻 `A(n)`、首图或完整图耗时
`B(n)`、尾段原生 `extend` 耗时 `E(n)`。`A` 用线性式，`B/E` 用二次式，
自变量均为 `n/1024`。在请求入口只读取 `N` 与固定的硬件配置：

```
T_full = max_r [ A_r(N) + B_r(N) ]
T_split(p) = max_r [ max(A_r(p) + B_r(p), A_r(N)) + E_r(N-p) ]
p ∈ {512, 1024, ...}, p < N, N-p ≥ 64
```

选择预测 `T_split` 最小的 `p`；仅当 `T_full-T_split(p)` 大于误差保护量
才启用切分。`A(N)` 是完整上传图开始时间的拟合值，作为切分时尾段已到达
的近似；切分后最终步骤的开始时间会受到首图完成的阻塞，**没有**用它
来拟合 `A(N)`。这一近似只在校准范围内使用。超出范围、配置不匹配或
无法触发切点时回退完整图。首段按原生 CAGRA `from_graph` 建图，尾段
调用原生 `extend`；D 在双 rank 图 READY 后从 V 拉 KV。无 P→D 直送。

可复现拟合：

```
python benchmark/fit_pvd_split_profile.py \
  benchmark/results/pvd_split_online_calibration_detail_cloudlab_20260929.json \
  benchmark/results/pvd_split_sweep_training_cloudlab_20260930.json \
  --holdout benchmark/results/pvd_split_sweep_holdout_cloudlab_20260930.json \
  --output benchmark/results/pvd_split_parametric_profile_cloudlab_20260930.json
```

## 在线校准和留出验证

CloudLab node0=P、node1=V/Gateway、node2=D，V 两张 V100S。每个方案
使用相同模型、Prompt 文本、512-token Prefill chunk、输出长度 6、同一个
V 进程；P/D 在方案之间重启。每个 1509/2159-token 方案有两个相同
case 的请求，另有暖机请求。Prompt 与输出 SHA-256 逐 case 一致，均为
HTTP 200、无应用错误。下表为两次的中位数，从客户端请求起算；
`1536` 是未参与系数拟合及误差阈值计算的留出切点。

| Prompt token | P→V 前缀 | 双 rank 图 READY | 首个 SSE | 客户端完成 |
| ---: | ---: | ---: | ---: | ---: |
| 1509 | 完整图 | 1.577 s | 1.720 s | 2.908 s |
| 1509 | 512 | 1.927 s | 2.069 s | 3.276 s |
| 1509 | 1024 | 1.739 s | 1.883 s | 3.091 s |
| 2159 | 完整图 | 2.180 s | 2.351 s | 3.969 s |
| 2159 | 512 | 2.146 s | 2.320 s | 3.940 s |
| 2159 | 1024 | **2.124 s** | **2.292 s** | **3.926 s** |
| 2159 | 1536（留出） | 2.173 s | 2.345 s | 3.946 s |

`2159→1024` 相比完整图，图 READY 早约 **56 ms**，客户端完成早约
**43 ms**。这是这两个样本的微小收益，不能据此启用切分：1509-token
的两个切点都更慢，2159-token 的收益小于测得波动。留出切点两次的
图 READY 收益预测误差分别为 14/64 ms；训练集中最大误差来自
`1509→512` 的一次样本，达到 745 ms。拟合曲线对图阶段的平均绝对
误差为 rank0 74 ms、rank1 95 ms。此前一天的 2048 前缀数据也参与
拟合，但未列入同进程的本轮对照表。

使用新 profile 的完整 P/V/D/Gateway 路径另跑了 2159 和 1509 token
各一次，均返回 HTTP 200、6 个输出 token；P 日志分别记录
`best_prefix=1024/512`、`prefix=0`。该集成检查含冷启动，不与上述
暖机时延比较。

原始[完整图请求](pvd_split_sweep_full_cloudlab_20260930.jsonl)、
[512 请求](pvd_split_sweep_p512_cloudlab_20260930.jsonl)、
[1024 请求](pvd_split_sweep_p1024_cloudlab_20260930.jsonl)、
[留出请求](pvd_split_sweep_p1536_cloudlab_20260930.jsonl)、
[V 图事件](pvd_split_sweep_graph_events_cloudlab_20260930.log)、
[训练明细](pvd_split_sweep_training_cloudlab_20260930.json)、
[留出明细](pvd_split_sweep_holdout_cloudlab_20260930.json)和
[部署 profile](pvd_split_parametric_profile_cloudlab_20260930.json)。

## 适用边界

目前模型只在上述硬件、精确 degree 16、四合一、page size 1、512-token
chunk 和 509–2159 token 内校准。它根据长度计算切点，不查逐长度策略表；
但不能仅凭这组短 Prompt 数据声称对任意长 Prompt 找到全局最优。要扩大
范围，需要针对更长长度和并发负载做一次**离线**校准及独立验证，随后
更新 profile；线上请求仍只输入 `N`。本轮未重测 CAGRA 检索召回率。
