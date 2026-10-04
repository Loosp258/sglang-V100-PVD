# 发送源 CUDA 完成范围

顺序优化第 3 步：`--experimental-scoped-sparse-source-completion`，默认关闭。
仅用于 immutable Prompt 上普通 Torch staging，不能混用 source slots、其他
打包实验或 direct batch PUT。launcher 设置
`PVD_SCOPED_SPARSE_SOURCE_COMPLETION=1`；准备独立 `v-scoped-completion` 对照。

每次打包记录实际 producer stream 及其末尾 event，CPU 等待 event 后才退休
Entry/index 读者。打包部分失败也经过同一完成证明；event 未知保留 staging、
index lease 及 Entry owner。成功后外层复用这次完成证明。RDMA adapter 仍
独立检查 exact physical registration/buffer/slice/device/stream/event capability，
再 CPU 等待该 event，然后才 submit native PUT。没有证明的其他调用保留
原来的全设备 fence；native metadata policy、终态字节及 cleanup gate 不变。

本地 gate：128 passed、15 actual CUDA skipped。包括 scope mismatch 拒绝、
证明先于 native submit、取消/部分复制/未知 event owner 保留，以及旧路径
生命周期回归。测试 CUDA/native policy doubles 为显式替身。
原始日志/hash：`artifacts/ordered_delivery_20261005/step3/gate01/`。

GPU 非默认 producer stream 测试已加入但无法执行。此完成范围尚需实际
Mooncake、并行建图/查询、两 rank 及 full-path 验证；不宣称当前 D wait/TPOT
已有收益，也不自动启用。
