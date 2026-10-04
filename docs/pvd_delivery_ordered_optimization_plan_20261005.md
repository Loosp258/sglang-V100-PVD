# 按顺序减少 Q→可用 KV 交付开销

2026-10-05，用户授权在 `codex/pvd-oasiskv` 上按顺序实施，每完成一步提交并
上传 GitHub。既有工作已同步至 `ce344c711`；大文件迁移与旧提交映射见
`pvd_github_publication_20261005.md`。

## 共同边界

- 保留当前快建图、V/CAGRA、每层立即发布 Q、Oasis actual/lookahead 配对前向、
  actual-only 写回及原 Top4/capacity32/max_new16 预算。
- 每步单独默认关闭，以相同源码、请求、模式、行数/字节/输出预算做对照。
  不累加独立实验的收益；不把减少调用次数直接称为 D wait 或 TPOT 改善。
- 每个实验产物放在 `artifacts/ordered_delivery_20261005/stepN/`；源码、测试、
 计划及报告使用正常仓库目录。创建前检查路径。
- CloudLab 已过期且无 GPU。可完成真实 CPU bytes/HTTP/lifecycle 和捕获回放；
  真实 CUDA、原生 RDMA、双物理 rank、完整路径时延/质量属于未执行的独立 gate。
- 收发 owners、generation、授权、精确 native 终态字节、GPU 本地完成证明及
  UNKNOWN 保留规则不能通过 timeout/HTTP cancel 推断或删除。

## 顺序与每步交付

1. **V 发送 staging/MR 复用。** 增加 rank/store 拥有的有界物理槽，每个交付
   获得独立逻辑 lease 和 exact-length staging view，发送使用真实 physical
   RegisteredMemory。槽在原 pack/outer/native fences 与 native cleanup 完成后
   返回；取消或 UNKNOWN 保留。分配/注册/注销失败保留物理 owner 和 charge。
   重复 start 不重新借槽，容量不足在写入前拒绝；关闭等待全部 lease 退休。
   检查真实 CPU PUT 字节、物理注册次数、并发 lease、预算、取消/失败/关闭。
   提交实现及证据、推送，再进入步骤 2。
2. **D READY 与 owned cleanup 分离。** 仅在 native KV 完整证明、独立安装和
   本地读写完成后发布 bank；ACK/接收退休保留独立所有者及有界清理队列。
   发布不释放接收槽，request close 必须 join 清理。清理失败传播并隔离，
   UNKNOWN 不复用。验证慢 ACK 不挡已完成 bank、取消/超时/退休和容量边界。
3. **缩小 CUDA source 完成范围。** 为本次打包建立明确 stream/event producer
   关系，保留异常路径完成证明与全部 owner。adapter 的 RDMA source readiness
   仍独立验证，不能只删除全设备同步。CPU 政策测试与真实 CUDA 测试分开。
4. **二进制 Q 传输。** 以有界 float32 二进制载荷替代 Python float JSON
   数组，元数据保持明确身份/shape/版本；逐层 Q 到达后立即发送，不等待后层。
   检查精确浮点值、布局/大小/finite、错误 envelope、HTTP 兼容性与完整 lifecycle。
5. **融合检索与交付请求。** 携带已验证缓存/驻留快照和有界接收授权；V 保持
   相同选择策略/候选预算后交付缺失 KV。动态 manifest 和授权必须先有可核对
   的 request/step/layer/entry/head/region/generation/byte 上界，不能先写再猜身份。
   验证旧/新选择与实际 wire 字节、无 miss、部分失败、取消、重放和未知响应。

每步报告记录实际执行的测试和失败修复、源码/input hash、注册或 API 计数，
以及尚未执行的 GPU/native/live gate。恢复 GPU 后按以上顺序做独立 full-path
ABBA，并以 D wait、TPOT、实际输出和 cleanup 判断采用，不能默认启用全部选项。

## 完成记录

- 第 1 步实现 `58a147863` 已推送；CPU gate 334 passed/32 CUDA skipped。
  真实捕获 KV 回放 840 bank 字节一致，注册次数从每份 770/772 降至 2。
  GPU/native/full-path 尚未执行，保持默认关闭。见
  `benchmark/results/pvd_v_source_slots_local_20261005.md` 及同名 JSON。
- 第 2 步：实现独立 READY future 与同线程 owned cleanup，64 个 CPU 测试
  通过；实际 CUDA/native/full-path 待测。详见
  `benchmark/results/pvd_owned_ready_cleanup_local_20261005.md`。
- 第 3 步：scoped producer event 与 adapter 本地 capability 核对；128 CPU
  测试通过、15 CUDA 测试跳过。见
  `benchmark/results/pvd_scoped_source_completion_local_20261005.md`。
- 第 4 步：直接 little-endian float32 envelope 与逐层 ndarray 发送，CPU
  HTTP/精确 bytes 对照通过，默认关闭；详见
  `benchmark/results/pvd_binary_queries_local_20261005.md`。
- 第 5 步：预授权动态接收前缀，融合原搜索/选择/reserve/start，并接入 D
  逐层 `_select_and_fetch`。CPU 双逻辑 rank/HTTP/字节/重放/失败 gate 通过；
  默认关闭，GPU/native/full-path 待测。详见
  `benchmark/results/pvd_fused_search_delivery_local_20261005.md`。
