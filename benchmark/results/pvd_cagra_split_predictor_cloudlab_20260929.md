# Exact-16 四合一 Prompt 图时间预测器：CloudLab 实测

## 范围与测量方法

在 CloudLab `clgpu020` 的两张 V100S 32 GB 上运行真实
Qwen2.5-7B-Instruct Prefill，提取每个 V rank 的 56 个 K head。
每四个 head 共用一张图，共 14 张图；每个 head 的首图邻接由精确
Top-16 KNN 生成，经 cuVS 25.10 `cagra.from_graph` 导入。
尾段使用一次原生 `cagra.extend`。计时覆盖 14 张图并逐次 CUDA
同步；K 的均值、居中和拼接另计，不在下面的建图/插入时间内。
每个网格点测两遍，第二遍反序执行。训练输入是随机 token Prompt，
留出输入包含自然语言 Prompt 和另一个 V rank 的 K head。
这一离线测试没有 P/V/D 并发、真实传输、检索或召回率测量。

探针：[run_pvd_qwen_grouped_cagra_gpu.py](../../test/registered/disaggregation/run_pvd_qwen_grouped_cagra_gpu.py)；
预测器：[pvd_cagra_split_predictor.py](../pvd_cagra_split_predictor.py)。
训练原始数据：
[短中长度网格](pvd_cagra_grid_train_rank0_20260929.json)、
[短尾段网格](pvd_cagra_grid_short_tail_rank0_20260929.json)、
[长长度网格](pvd_cagra_grid_long_rank0_20260929.json)。
拟合后的[模型](pvd_cagra_split_model_rank0_20260929.json)。

## 代表性实测：14 张图的总时间

| Prompt `N` | 首图前缀 `p` | 尾段 `m` | 首图 B | 一次 extend E |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 全量 | 0 | 0.074 s | — |
| 1024 | 全量 | 0 | 0.134 s | — |
| 2048 | 全量 | 0 | 0.323 s | — |
| 2304 | 全量 | 0 | 0.419 s | — |
| 3072 | 全量 | 0 | 0.708 s | — |
| 4096 | 全量 | 0 | 1.128 s | — |
| 2156 | 1792 | 364 | 0.275 s | 0.244 s |
| 2156 | 1920 | 236 | 0.304 s | 0.202 s |
| 2156 | 2048 | 108 | 0.333 s | 0.188 s |
| 4096 | 2048 | 2048 | 0.306 s | 0.674 s |
| 4096 | 3584 | 512 | 0.878 s | 0.305 s |
| 4096 | 3968 | 128 | 1.065 s | 0.214 s |

数值为两遍均值，rank 0。首图近似随 `p²` 增长；`extend` 需要同时
输入 `p` 和 `m`。同一 `m` 下的耗时会随 `p` 变化，数据内容和运行次序
也会引入几十毫秒波动。此前 Case 40 的 2048+108 复测为
0.242/0.230 s（rank 0/1），本轮随机 Prompt rank 0 为 0.188 s；
不能把单次短尾段测量当作固定常数。

## 拟合与独立验证

预测器对完整/首段建图拟合 `a + b p + c p²`，对一次 `extend`
拟合 `a + b p + c m + d p m + e m² + f p²`；变量 `p,m` 以 1024
token 为单位。只在已测范围插值：`256 ≤ N ≤ 4096`，切分时
`256 ≤ p ≤ 3968`、`64 ≤ m ≤ 2048`。更长 Prompt 或范围外切分
返回“未校准”，建议完整图路径。训练覆盖多个 `N,p,m`，不是
对固定尾段单变量外推。

| 未参与拟合的集合 | 图 B MAE | extend E MAE | B+E MAE | B+E 95% 绝对误差 |
| --- | ---: | ---: | ---: | ---: |
| rank 0，自然语言，640–2176 token | 4.1 ms | 17.0 ms | 18.6 ms | 33.9 ms |
| rank 1，随机 token，640–2176 token | 5.3 ms | 20.0 ms | 22.0 ms | 43.9 ms |
| rank 0，自然语言，2816–3840 token | 16.0 ms | 20.5 ms | 30.2 ms | 61.6 ms |
| rank 1，自然语言，3328/3840 token | 9.5 ms | 19.6 ms | 24.9 ms | 66.6 ms |
| 合计，74 次 B、48 次 E | 8.6 ms | 19.3 ms | 24.0 ms | 52.9 ms |

合计中 `B+E` 最大观测绝对误差为 82.8 ms。留出原始数据：
[短 rank 0](pvd_cagra_grid_holdout_rank0_language_20260929.json)、
[短 rank 1](pvd_cagra_grid_holdout_rank1_random_20260929.json)、
[长 rank 0](pvd_cagra_grid_long_holdout_rank0_language_20260929.json)、
[长 rank 1](pvd_cagra_grid_long_holdout_rank1_language_20260929.json)。
这只是离线 GPU 时间预测误差，不能视为在线服务时的误差界。

## 用于切分的时间轴

对每个 V rank `r`，输入同一时钟下的：

- `t_prefix[r,p]`：`p` 个 K 到达且可读；
- `t_full_split[r]`：采用切分上传时全部 K 到达；
- `t_full_baseline[r]`：采用完整上传时全部 K 到达。

则预测图 READY：

`T_split[r,p] = max(t_prefix[r,p] + B(p), t_full_split[r]) + E(p,N-p)`

`T_full[r] = t_full_baseline[r] + B(N)`

两个 rank 取 `max_r T`。脚本在两个 rank 都有到达时间的候选前缀
中选择最早 READY 的值；默认只有预测双 rank 收益 **超过 125 ms**
才推荐切分，否则保留完整图。这一余量约为最差留出集合 95% 的
两倍，用来避免为几十毫秒噪声改变策略；不是概率保证。

调用方式：

```bash
python benchmark/pvd_cagra_split_predictor.py optimize \
  benchmark/results/pvd_cagra_split_model_rank0_20260929.json \
  2156 arrivals.json --uncertainty-ms 125
```

`arrivals.json` 的格式：

```json
{
  "ranks": [
    {
      "baseline_full_seconds": 10.0,
      "split_full_seconds": 10.1,
      "prefix_seconds": {"1792": 9.6, "1920": 9.7, "2048": 9.8}
    },
    {
      "baseline_full_seconds": 10.1,
      "split_full_seconds": 10.2,
      "prefix_seconds": {"1792": 9.7, "1920": 9.8, "2048": 9.9}
    }
  ]
}
```

上述时间只示意格式，不是 CloudLab 到达实测。`optimize` 输出每个
候选的双 rank 图 READY 时间、相对完整图收益和推荐前缀。
`evaluate` 命令可在上述留出 JSON 上复现误差表。

以 2156-token 的模型预测为例，完整图约 0.361 s，2048+108 的
`B≈0.332 s`、`E≈0.190 s`。在两种上传的最终到达时间相同时，
2048 前缀须早到约 0.161 s 才刚好胜过完整图；须早到约
0.332 s 才能完全隐藏首图。完全隐藏时，图 READY 的理论最大
收益约 0.171 s。要超过默认 125 ms 的决策余量，早到量约需
0.286 s。真实上传可能合并晚到前缀；必须提供实际或经过验证的
到达时间，不能只代入 25 Gb/s 物理带宽。

## 使用边界

- 这个模型预测 V 的图 READY，不包含 P 端 Prefill、打包、上传、
  V→D KV 传输、D 首次检索或客户端完成时间。到达曲线需要来自
  在线采样或单独校准的 Prefill/上传模型。
- 当前只覆盖一次 `extend`、degree 16、精确每-head 首图、四合一、
  cuVS 25.10、V100S 32 GB。多次 `extend`、不同 GPU/算法需重测。
- 未在本轮重测召回率。此前 2156-token 的晚前缀方案有 rank 0
  最差 head 召回率 0.85；预测器不能绕过质量门槛，尚未接入生产。
