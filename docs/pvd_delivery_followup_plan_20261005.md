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
