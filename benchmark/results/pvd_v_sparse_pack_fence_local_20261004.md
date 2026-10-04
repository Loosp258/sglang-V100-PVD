# V 稀疏交付：阶段计时与打包同步复用

2026-10-04，分支 `codex/pvd-oasiskv`，仅本地提交。CloudLab 租约已过期，
本机 Torch 2.14.0+cpu；本报告没有新增 GPU、原生 RDMA 或线上 Decode 实验。

## 已完成的步骤

| 步骤 | 本地提交 | 完成内容 |
|---|---|---|
| 计划 | `56e689006` | 明确同步的保护对象、失败处理和公平比较条件 |
| 阶段计时 | `13c8f38ae` | 每次交付记录 V 源准备的七个子阶段，D 保存有效诊断 |
| 同步复用 | `05dec10f2` | 默认关闭的实验选项、生命周期验证、真实 CUDA 测试及 ABBA 入口 |

保留当前四 head 快建图、V/CAGRA、Oasis 配对 Decode、逐层 Q 发布、Top4、
capacity32 和两个回调 worker。此次优化对象是选定 KV 的交付准备。

## 为什么可以省掉一次同步

普通 Torch packed PUT 路径依次执行：

1. V 拷贝选定 K/V；在索引与 Entry 租约仍有效时完成设备同步。
2. V 释放打包索引租约，注册已经完成打包的 staging；随后再做一次外层设备同步。
3. Mooncake 提交前再次同步，保证 GPUDirect 源就绪，再提交原生 PUT。

第 2 次同步原本也保护取消或部分拷贝失败后的 Entry 读取。新选项只在本次
打包已完成第 1 次同步、没有 UNKNOWN、采用普通 Torch staging 时复用该证明。
私有证明在每次真正进入准备路径时重置；远端诊断不能设置它。注册没有成功
完成或同步失败时仍执行外层路径并保留 UNKNOWN 的所有资源。

| 正常路径的调用次数 | 原路径 | 实验路径 | 验证范围 |
|---|---:|---:|---|
| V store 设备同步 | 2 | 1 | 显式 CPU CUDA 策略模拟 |
| Mooncake 提交前同步 | 1 | 1 | 原适配器代码，CUDA/native 边界模拟 |
| 代码路径合计 | 3 | 2 | 不是 GPU 同步耗时实测 |

部分拷贝抛错但第一次同步成功时，可以安全复用其完成证明，随后关闭
NOT_SUBMITTED 授权并释放 staging。打包同步失败、注册不确定、原生提交不确定
和 metadata UNKNOWN 均保持原有隔离与保留规则；成功的后续同步不修复 UNKNOWN。

入口为 `--experimental-reuse-sparse-pack-fence`，launcher 对应
`PVD_REUSE_SPARSE_PACK_FENCE=1`。默认关闭，首轮不与 Triton、direct batch 或
contiguous packing 混用。完整 KV、直接 scatter 和 fan-in 路径不使用这项复用。

## 新增计时能说明什么

`source_profile` 固定记录 allocate、pin_index、pack、pack_fence、register、
outer_fence、submit 的调用数、成功数和单调时钟 wall time。它只含固定字段，
随既有响应返回，不新增 RPC。D 在身份、精确字节数和原生成功终态校验之后
复制这些诊断；缺失或无效诊断不能替代接收证明。

pack 是 Torch 提交拷贝的 wall time；pack_fence 才包含等待完成。
submit 包含适配器自己的 CUDA 同步和原生提交。各阶段独立中位数不能相加
重建客户端耗时；这些字段没有覆盖整个 HTTP 请求、后续 native poll 或源退休。
计时本身的线上成本也需要在恢复 GPU 后检查。

另外重读了既有 `oasis_direct_sparse_abba03` 两个 baseline arm、两个 V rank 的
warmed→after 健康计数。按调用数加权的适配器均值为：

| 历史子阶段 | 加权均值 |
|---|---:|
| Mooncake 提交前 CUDA 同步 | 0.183 ms/调用 |
| 原生单 PUT 提交 | 1.880 ms/调用 |

这是历史 formal 窗口的进程累计计数差，包含初始 fan-in，**不是当前选项收益，
也不是 steady 每层或每 token 时间**。它没有测到 V 内部前两次同步，不能由此
认定所有同步都很便宜，也不能拿这些均值与另一实验约 16 ms 的控制阶段中位数
做减法。新计时用于确定打包、外层同步、V 注册和提交各自的实际份额。

## 验证与证据

- 最终 gate：**233 passed，22 个真实 CUDA 用例跳过**，1 个已有 asyncio_mode
  配置警告。测试涉及实际 CPU 张量、HTTP、适配器和生命周期代码；CUDA/native
  的模拟边界在测试名称与说明中明确标注。
- 两 rank 的打包 KV 与独立未打包张量核对精确相等；覆盖不同 head、乱序 ID、
  layer 与索引映射。候选、字节数、原生终态、ACK 条件和预算保持既有语义。
- 验证部分拷贝失败、打包中取消、同步失败、注册及提交 UNKNOWN、重复 start、
  过期私有证明、诊断不能伪造终态，以及计时对象分配失败后的授权关闭。
- 八个真实 V CUDA 用例覆盖原/复用模式、默认/非默认 stream 和部分拷贝失败。
  由于无 GPU 全部跳过；使用受控传输，未来通过也不能代替原生 RDMA gate。
- 三个修改的 benchmark Python 入口完成语法解析；launcher 完成 bash `-n`。
- 初期失败日志保留：profile_gate01 的夹具缺少 V terminal 轮询，profile_gate02
  的异常类型匹配错误，reuse_gate01 的启动拒绝错误文字匹配错误；修正后通过。

原始日志、逐文件 Git/测试源码证明和历史计数来源保存在
`artifacts/v_pack_fence_20261004/evidence.tar.gz`。最终测试源码与提交
`05dec10f2` 按 LF 规范化逐文件核对；用户已有的 AGENTS.md 修改没有加入提交。
结构化摘要见同名 JSON。

## 后续执行顺序

1. 恢复 GPU 后先更新过期的节点连接信息及模型路径，跑真实 CUDA 与原生 RDMA
   gate，再运行 `v-pack-fence` ABBA：
   base_a、opt_a、opt_b、base_b；两个相同 Prompt、max_new16、相同 warmup。
   只切换 V 复用选项，D 的配置以及所有其他优化选项完全相同。
2. 对照脚本要求实际两 rank 模式、每次交付成功的 pack fence、outer fence
   次数、未改变的候选/传输预算、输出及清理；同时保存适配器累计提交计数。
   报告 V 阶段、D 等待和客户端 TPOT，不用 CPU 次数推算线上节省。
3. 若 V register 成为主要开销，再实施有界 V staging/MR 复用；若 pack_fence
   等待占大头，再评估精确 stream/event 完成证明；若整个 RPC 仍慢，则继续
   测 native poll、源退休和控制调度。每项独立验证、独立本地提交。

当前可以确认减少了一次冗余同步调用；**尚不能确认减少多少毫秒，或已经缩短
约 420 ms/token 的等待**。保留默认关闭，等待线上证据决定是否采用。
