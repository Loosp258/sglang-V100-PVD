# 在线切分预测器与 V→D 等图策略：CloudLab 2026-09-29

## 结论

在本次 V100S、Qwen2.5-7B、精确 degree 16、四合一 CAGRA、双 V rank 的
四个 Prompt 长度上，在线校准后的预测器均选择 **完整 KV 一次上传、完整图一次建造**。
让 D 在双 rank 图 READY 后才从 V 拉取初始 KV，未缩短客户端等待；与
“完整 KV 到 V 后立即拉取、首次检索按现有协议等待图”相比，首 token
增加 0.16–1.20 秒，六个输出 token 的完成时间增加 0.16–1.21 秒。

本报告只比较两种 V→D 初始 KV 交付时机，不把 P→D 直送作为对照。

## 实验设置与公平性

- CloudLab node0=P、node1=V/Gateway、node2=D；V 两张 V100S、P/D 各一张。
- 两臂使用同一个隔离 checkout、模型、Prompt 文本、`temperature=0`、
  `max_new_tokens=6`、512-token Prefill chunk、同一 Gateway 和 V 进程。
- V 使用 14 图/rank、每图四个 head、精确 per-head top-16 邻接和
  cuVS `cagra.from_graph`，扩展使用原生 `cagra.extend`。
- 每个长度两次，按 2159、509、1509、1009、1009、1509、509、2159
  token 的顺序运行。两臂分别重启 P/D；P 的日志显示全部请求
  `#cached-token: 0`。逐 case 校验 Prompt SHA-256、完成 token 数和
  最终文本 SHA-256 一致；全部 16 个请求返回 HTTP 200 且无应用错误。
- “立即拉取”指 D 在完整 KV 到 V 后启动已有 V→D fan-in；并不要求图
  READY。对照臂在同一 fan-in 前显式查询双 rank 图状态，全部 READY 才拉取。

| Prompt token | 无等图：首 token | 预测器＋等图：首 token | 增加 | 无等图：完成 | 预测器＋等图：完成 | 增加 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 509 | 0.360 s | 1.251 s | 0.891 s | 1.439 s | 1.946 s | 0.507 s |
| 1009 | 0.533 s | 1.404 s | 0.871 s | 1.738 s | 2.295 s | 0.558 s |
| 1509 | 1.688 s | 1.851 s | 0.164 s | 2.853 s | 3.015 s | 0.162 s |
| 2159 | 1.321 s | 2.519 s | 1.199 s | 3.194 s | 4.407 s | 1.213 s |

均为两次的中位数；“增加”为每个相同 case 配对差值的中位数。
原始[无等图请求](pvd_split_vfanin_nogate_cloudlab_20260929.jsonl)、
[预测器＋等图请求](pvd_split_vfanin_predict_gate_cloudlab_20260929.jsonl)、
[逐 case 对照](pvd_split_vfanin_paired_cloudlab_20260929.json)。

## 预测器为何选择不切分

原离线模型只覆盖单独 GPU 工作，其图时延显著低于本次双 rank 在线服务。
本次先用相同 Prompt 的完整图和晚前缀切分各两次做在线校准；从 V 日志中
取图步骤开始时刻作为该段 KV 已到齐的上界，并对两个 rank 分别取中位数。
预测器使用在线的完整图/首图/extend 耗时，计算双 rank 最晚 READY：

`T_split[r] = max(t_prefix[r] + B[r], t_full_split[r]) + E[r]`

`T_full[r] = t_full_baseline[r] + B_full[r]`

| Prompt token | 实测候选前缀 | 预测完整图双 rank READY | 预测切分双 rank READY | 切分差值 |
| ---: | ---: | ---: | ---: | ---: |
| 1009 | 512 | 1.305 s | 1.446 s | +0.140 s |
| 1509 | 1024 | 1.712 s | 1.795 s | +0.084 s |
| 2159 | 2048 | 2.178 s | 2.409 s | +0.231 s |

509-token Prompt 无有效 512-token 前缀，走完整图。上述时间相对客户端请求
起点。首图执行会使 V 的尾段进入最终图步骤变晚：例如 2159-token
校准中，完整上传图步骤约在 0.80–0.83 秒开始，而 2048+111 切分的
最终步骤约在 2.11–2.23 秒开始。因此短尾段的 `extend` 虽只约
0.16–0.18 秒，整体 READY 仍变晚。在线到达估计是图步骤开始时间，
不等同于网络收包完成时间；它作为保守上界用于本次决策。

[完整图校准请求](pvd_split_gate_full_pilot_cloudlab_20260929.jsonl)、
[晚前缀校准请求](pvd_split_gate_late_pilot_cloudlab_20260929.jsonl)、
[V 图阶段日志](pvd_split_graph_events_cloudlab_20260929.log)、
[逐请求校准映射](pvd_split_online_calibration_detail_cloudlab_20260929.json)、
[在线到达和建图校准](pvd_split_online_arrivals_cloudlab_20260929.json)、
[编译出的策略](pvd_split_predicted_policy_cloudlab_20260929.json)。正式请求
的 P 日志逐条确认 509、1009、1509、2159 token 都用了 `prefix=0`。

## 实现与边界

P 可加载预测器编译的长度→前缀策略；未知长度安全回退完整图。D 的
双图门控由显式开关控制，并在初次 V→D fan-in 前执行。图状态绑定
所选 Entry 和两个 V rank，等待有时限。默认测试配置选择精确 degree 16、
四合一 CAGRA。

本次每个长度只有两次正式配对和两次在线切分校准；只测 509–2159
token、单请求、6 个输出 token。结果足以说明这些配置下没有观察到
等待 D 的收益，不能外推到更长 Prompt、其他负载或网卡。在线实验未重测
召回率；预测器的时延决策不代替图质量验证。
