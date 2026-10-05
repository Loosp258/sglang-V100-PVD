# 继续减少逐层 Q→KV 开销

2026-10-05。用户要求按顺序完成，每步本地 commit，不上传 GitHub。

共同边界：保留快图、V/CAGRA、Oasis 配对逐层前向、立即发布 Q、actual-only
写回及 Top4/capacity32/max_new16。新模式默认关闭。临时产物位于本工作树
`artifacts/delivery_followup_20261005/`。没有 GPU；真实 CPU HTTP/字节/生命周期
验证与待执行 CUDA/RDMA/完整路径性能测试分开报告。

1. **二进制 Q＋融合交付。** 嵌套搜索的 float32 Q 使用有界原始字节载荷。
   同一冻结 Q/选择快照绑定授权摘要；两端核对原身份与动态 manifest。验证
   双逻辑 rank、miss/hit、错误长度、非 finite、丢响应与取消。
2. **融合 D 接收槽。** 将 allocation-only 授权接入既有有界物理槽；每次
   lease 使用独立 generation，实际交付仍为精确 prefix。零 miss 必须确认
   没有潜在 writer 后返槽；UNKNOWN 保留。验证复用、延迟旧写和关闭。
3. **缓存快照压缩。** 稀疏整数载荷与 bitset 择小；精确恢复相同缓存集合。
   Prompt 长度、head 顺序、padding 和载荷上界必须验证。保持完整快照，先
   避免引入需要跨请求 ACK 的增量状态。验证选择、缺失字节与边界等价。
4. **固定 pinned scratch＋事件衔接。** 使用计费、有界、请求拥有的缓冲区，
   在实际 GPU 完成前保留每个 lease。研究仅对本地 bank 安装采用事件依赖，
   远程写完成与 GPUDirect receive ordering 保持原证明。CPU 生命周期测试
   不替代 CUDA 测试；不能在没有完成证明时释放或覆盖缓冲区。
5. **请求级二进制控制通道。** 将同一融合业务处理器接到有界持久通道，
   使用明确序号、身份及响应上界；保留既有 KV/Mooncake、ACK/fence 路径。
   验证真实本地网络、并发、断线、取消、close drain 和旧模式兼容。

每步提交代码、必要的回归测试和报告后进入下一步。性能采用与前一步相同
配置的独立对照，不累加历史阶段中位数，不宣称未实测的 D wait/TPOT 收益。

## 本地完成与资源恢复后的对照

1. `ecd35a47a`：二进制 Q＋融合；126 passed/1 CUDA skipped。
2. `e0e51f02c`：融合接收槽；87 passed/1 CUDA skipped。
3. `26f330ff4`：缓存快照；107 passed/1 CUDA skipped。
4. `e7c3188fc`：pinned scratch＋事件 bank；93 passed/2 CUDA skipped。
5. 二进制持久控制通道：166 passed/2 CUDA skipped；最终跨步 gate
   436 passed/3 CUDA skipped。各次测试有重叠，数量不相加。

所有代码与 gate 说明见 `benchmark/results/pvd_delivery_followup_local_20261005.md`。
最终配置解析已验证每个组合，但组合 CUDA/native/full-path 未执行。

| 下一轮公平实验 | 两臂共同设置 | 单一变量 |
|---|---|---|
| 1 | fused=true，原固定预算 | binary_queries false→true |
| 2 | fused/binary=true | reuse_receive_slots false→true |
| 3 | fused/binary/receive_slots=true | compact_cache_snapshots false→true |
| 4a | 上一步设置 | reuse_pinned_scratch false→true；event_bank_ready=false |
| 4b | 上一步＋pinned=true | event_bank_ready false→true |
| 5 | 上一步全部设置；reuse_io=false | binary_control_channel false→true |

示例开关仅应合并到已验证的完整 D 配置；全部默认关闭：

```json
{
  "fused_search_delivery": true,
  "binary_queries": true,
  "reuse_receive_slots": true,
  "compact_cache_snapshots": true,
  "reuse_pinned_scratch": true,
  "event_bank_ready": true,
  "binary_control_channel": true
}
```

有 GPU 后按相同 Prompt/输出数/暖机/启动源码/资源做 ABBA，记录 per-rank
查询、控制与 native 交付、callback READY、实际 GPU 事件依赖、D wait、
客户端 TPOT、精确输出和峰值预算。尤其不能把事件 READY 提前的时间当作
真实 Decode 加速，也不能将历史独立注册池与当前融合通道收益相加。
