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

## 3. 冻结二进制 Q 只编码一次

- D 一次编码保留不可变 wire bytes，供摘要和最终 envelope 使用；Q 为其
  readonly view。metadata 字节单独冻结，producer 后续 mutation 不改变发送。
- V 从原始载荷重组 canonical search metadata 与 Q bytes 计算相同摘要，
  不再把 ndarray 编码为 Q bytes。V 保留原 writable owned array，供 Torch/
  native 消费，避免把只读 numpy 内存作为可写 tensor。
- 协议/字节/旧摘要完全一致。计数测试确认 D freeze、摘要、发送、V 解析、
  V 摘要合计只调用一次 query pack。旧 D prepare→digest→send 原有四次。
- `step3/gate03`：69 passed、1 actual-CUDA skipped。真实双 rank HTTP/
  channel、miss/mixed/hit、授权 proof、FP16 字节与旧 binary oracle 等价；
  签名零、producer mutation、NaN/长度/shape 错误继续拒绝。
- 日志/源码 hash：`artifacts/async_delivery_20261005/step3/gate03/`。
  CPU pack 调用数减少不等价于已实测 TPOT 降幅。

## 4. 请求级有界异步 owner

- 新增默认关闭 `async_layer_jobs`，要求 fused binary channel 与
  `ready_before_cleanup`。一个请求 owner 线程运行异步循环，最多两个有效
  查询/交付任务，另保留两个退休任务；总 admitted 上限 56。
- 网络 await 不占独立层线程；bank 本地完成后让出查询名额，原 owner
  继续 ACK/close。每个 job 的 registry/stream/control client 保持独立，
  创建、CPU copy 与退休均在同一线程。CUDA context 不跨 await。
- local GPU 事件、RDMA ordering 和 exact terminal proof 保留；事件 READY
  不等于 GPU 完成。GPU copy 同步仍可能阻塞 owner loop，需要实际 GPU 测试。
- receive pool 每 rank 从 2 到 4 个物理槽，支持 2 active+2 cleanup。capacity32
  时最坏物理容量从 128 KiB 增到 256 KiB，实际创建仍逐个计费，有界 UNKNOWN
  不退槽。pinned scratch 仍两个 slot。queued Q/四 live job 的额外 tensor
  allowance 是 1,875,968 bytes，必须由 request_scratch_bytes 覆盖。
- 完整 CPU-policy job 与真实双 V CPU HTTP/channel 证明：阻塞前两层 ACK
  时，后两层仍安装 bank；FP16 K/V 与 immutable source 逐字节一致。随后
  全缓存命中，无新增 PUT；8 个 MR 全部释放，budget/inflight 回到零。
  两种 pinned/event 组合均通过。搜索/backend/native completion 是明确的
  CPU exact/fake doubles，不作为 CAGRA/RDMA/真实模型质量或性能证据。
- public cancellation 不取消真实任务；队列过期不创建 owner；清理失败
  会 latch 并保留 loop；close 超时保留原 drain future，重试 join 原任务。
- 中间 gate 暴露 Python 3.14 gather(all-done) 不 yield 的关闭忙循环，已
  修复，失败/终止记录保留。最终跨步收集还发现旧测试重新绑定 pipeline
  module，测试装配需使用消费者当前的 exact LayerReply class；仅修正
  fixture，生产 ticket/类型检查保持严格。

## 最终验证与后续实测

- `final/gate03`：**517 passed、3 actual-CUDA skipped**，一个既有
  asyncio_mode warning。涵盖之前五步、当前四步和 store/receiver 失败恢复。
  这些测试与分步 gate 有重叠，数量不相加。
- 154 个 PVD Python 模块全部 AST parse 通过。
- 日志/执行命令/测试前后源码 hash：
  `artifacts/async_delivery_20261005/final/gate03/`。阶段源文件与真实 commit
  对照保存为该产物树内的 `evidence_commit.json`。
- 有 GPU 后依次对照：fused READY flag；zero-miss proof flag；旧/新 binary
  snapshot（相同开关、不同对应源码）；async_layer_jobs flag。最后一项明确
  记录 owner 线程数和新增接收槽，检索/native 并发保持相同。使用相同 Prompt、
  warmup、输出数与快图配置，报告 D wait/TPOT/TTFT、精确输出与峰值内存。
- 尚未测量最新真实 D wait/TPOT，不能把历史收益相加或宣称 GPU 加速。
