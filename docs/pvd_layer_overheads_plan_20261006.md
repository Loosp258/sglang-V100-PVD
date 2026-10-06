# 逐层交付固定开销：五步实施

2026-10-06。用户要求按顺序实施，每步验证后本地 commit，不上传 GitHub。
工作树 `sglang-V100-PVD-oasiskv`，分支 `codex/pvd-oasiskv`。
保留快图、V/CAGRA、Q/TopK/驻留预算与 actual-only 写回。
CloudLab 已到期，无 GPU；明确区分 CPU 验证与 CUDA/RDMA/TPOT 实测。
临时输出只放在本工作树 `artifacts/layer_overheads_20261006/`。

1. V 融合搜索直接消费结果对象，最终响应单次编码；保留所有请求与大小校验。
2. 两个 rank 的 owned ACK/close 并行推进，仍尝试并 join 全部清理，失败可见。
3. ACK/fence 复用请求级 channel，绑定完整 WriteIdentity，断线使用原 HTTP fence
   恢复；清理不挤占原两项搜索预算，通道总容量明确有界。
4. 本地 Q D2H、KV D2H、bank H2D 事件完成以异步等待推进，保持原 RDMA ordering。
   owner、临时内存与接收槽只在实际完成证明后退休；GPU 正确性待有 GPU 验证。
5. 精确缓存按版本传增量，绑定请求/Entry/layer/head；ACK 延迟不阻塞查询。
   丢版本/重排不得造成假命中；可重新发送完整快照恢复，不改变选择与 missing 集。

新模式默认关闭。每步记录代码、测试、资源/失败边界及 commit；有 GPU 后逐项
ABBA，只改变一项，检查精确输出、流量、峰值内存、D wait 与 TPOT。
