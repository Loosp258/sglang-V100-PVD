# Oasis 真实轨迹：注意力工作区与纯 SDPA CUDA Graph

两个2159-token Prompt、16实际输出。同一进程／SGLang runner／真实 EAGLE 与普通采样器；每个模式的logits、features、actual KV和420次formal KV写入均逐位一致。

每模式两次排除warmup、三次wall trial；按case和repetition交替反序。GPU-event另测。所有KV已在GPU，后台仍保留两worker、query clone/event、ticket与handoff；本诊断不含V／网络／receive／CPU备份竞争。

| 模式 | D稳态ms/token | 每trial准备ms（不含在稳态） |
|---|---:|---:|
| original | 28.836 | 0.001 |
| workspace | 28.604 | 0.103 |
| sdpa_graph | 28.543 | 20.043 |

CUDA Graph仅捕获SDPA，保持原变长span和mask，按实际span预先捕获；future等待、Q发布、EAGLE、采样和formal写入均在图外。每次trial新建工作区；显式32MiB上限检查实际graph pool显存增量，关闭先同步再释放。

准备／捕获时间计在表中；初始KV与private Prompt seed不在稳态，但完整线上实验收费。本表不是客户端TPOT收益，也不能与其他独立优化相加。默认工作区关闭；没有上线CUDA Graph。

完整轨迹、逐层时间戳、launch/config/source和清理证据在raw.tar.gz。预算、原生请求与private formal rows全部退休，六GPU归零。TP1、两个合成文本Prompt、短greedy Decode；更多质量、长Decode、负载、TP2仍开放。

CPU验证：`python benchmark/preserve_pvd_oasis_attention_replay.py benchmark/results/pvd_oasis_attention_replay_cloudlab_20261003 --verify`。
