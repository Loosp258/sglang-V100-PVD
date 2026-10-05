# 逐层交付继续优化：本地验证

工作树 `sglang-V100-PVD-oasiskv`；四步各自本地提交，不上传 GitHub。
没有 GPU。这里的时间顺序、网络与生命周期验证不证明 RDMA/CUDA 或 TPOT 收益。
共同设置：保留快图、V/CAGRA、逐层 Q 与实际历史写回，检索预算不变。

## 1. 融合交付 READY 与清理分离

- 复用默认关闭的 `ready_before_cleanup`，允许与融合 binary Q、接收槽、
  compact snapshot、pinned/event bank 和持久 channel 组合。
- 正确 native terminal 与 CPU 缓存复制仍在 bank 安装前；随后发布独立 bank，
  原线程才 ACK/close。事件模式仍由 consumer wait_event 依赖真实 GPU 完成。
- 延迟 ACK 不提前释放接收槽；请求关闭 join 原 owner；ACK 异常可见。
- `step1/gate02`：41 passed、1 actual-CUDA skipped。包含真实 CPU HTTP
  融合交付与精确 FP16 字节、延迟 ACK/失败、完整 job 的发布顺序和配置组合。
  CUDA API 是显式 CPU policy doubles。既有 asyncio_mode 配置 warning。
- 第一次 gate 因 Windows 不提供 Linux `resource` 而未收集成功；修正测试
  runner 使用项目既有 namespace bootstrap 后重跑，失败日志原样保留。
- 日志/源码 hash：`artifacts/async_delivery_20261005/step1/gate02/`。
  本步仍由两个原 owner 执行清理，不宣称已消除 worker 占用。

## 2. 零 miss 的 absent-write 证明

- 新增默认关闭 D 配置 `fused_zero_miss_proof`，要求融合模式。V 使用同一
  store 锁核对原 WriteIdentity 没有 delivery 并安装已有有界 tombstone；
  搜索响应返回完整身份、generation 与 fenced=true。
- D 严格验证后，仅关闭本地 lease；不发送额外 fence 或虚构一次 ACK。
  原先/关闭模式响应不变。丢失/错误/缺少证明仍走身份绑定的 HTTP fence。
- 不允许把已有 writer 当作零 miss：拒绝并保留 writer，不取消其 native PUT。
- `step2/gate02`：82 passed、1 actual-CUDA skipped。覆盖原两 rank 的
  miss/mixed/hit、HTTP JSON/binary/channel、精确字节、MR generation 复用、
  丢响应、坏证明和晚 reserve 被 tombstone 拒绝。有效 proof 路径断言 fence
  调用为零。第一次 gate 的旧手工 transport fixture 缺新字段已修正。
- 日志/源码 hash：`artifacts/async_delivery_20261005/step2/gate02/`。
  缓存命中比例与完整路径收益尚未测量。
