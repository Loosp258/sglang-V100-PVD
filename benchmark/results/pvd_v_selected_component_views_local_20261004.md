# V 稀疏交付：只准备所选层的 K/V 视图

2026-10-04，分支 `codex/pvd-oasiskv`，仅本地提交。CloudLab 租约已过期，
本机 Torch 2.14.0+cpu；本报告没有新增 GPU、原生 RDMA 或线上 Decode 测量。

## 已完成的步骤

| 步骤 | 本地提交 | 完成内容 |
|---|---|---|
| 计划 | `e80d93f1d` | 核对历史轮询，限定所选层视图的优化范围与验证条件 |
| 实现 | `dfa4c8f9e` | 默认关闭的 V 选项、真实 KV 字节回放、CPU 对照与公平线上入口 |
| 证据 | 本报告与同名 JSON | 冻结测试、输入和已提交源码，记录收益范围及待测项目 |

### 为什么选择这一项

先核对历史 `oasis_direct_sparse_abba03` 的 base_a/base_b、两个 Prompt，
每个请求排除初始 28 个 layer job。2860 次 steady rank 交付中只有 6 次额外
轮询，合计 6 个 poll。全部交付平均 poll wall time 为 0.00508 ms，发生轮询
的交付中位数为 2.395 ms。该夹具没有显示频繁轮询的优化空间，因此此次没有
实施 start 内等待或轮询合并；这个结论只适用于该历史样本。

V 的 Entry 采用 component-major 布局：28 层 K 和 28 层 V。
原 `sparse_payload._views()` 每次交付都建立全部 56 个 Torch 类型视图；
Oasis 逐层 job 只读取该层 K/V，两个 local head 可共享同一对视图。
新路径每次只准备所选层的两个视图。多层请求按唯一层去重，仍正确映射 K/V。

所有 component 的 dtype、shape、bytes 元数据仍在每次调用检查，
包括没有被查询的层；没有缓存可变 layout 字典。保持 Entry/index 身份、
layer/head/token 边界、partial-page padding、目标非别名及对齐校验、
wire 顺序、Torch 拷贝、原有两个 V store 同步与 native adapter 同步。
不更改 CAGRA、候选预算、注册、ACK 或 UNKNOWN 的资源保留。

入口：`--experimental-selected-sparse-component-views`，launcher 对应
`PVD_SELECTED_SPARSE_COMPONENT_VIEWS=1`。默认关闭，首次实验要求普通
Torch CUDA staging，与 Triton、direct PUT、contiguous packing、同步复用互斥。

## 本地 CPU 打包时间

使用与上一轮 cache-install 报告**相同 SHA256 的真实捕获轨迹**：
`artifacts/oasis_workspace_replay01/capture/{99401,99402}/trajectory.pt`。
固定 Torch 一个 CPU 线程，两种模式均预热；5 轮 ABBA，每轮
base_a、opt_a、opt_b、base_b，复用相同预分配的 source、manifest 和 destination。

以下为实际 CPU `copy_sparse_kv_into` 加回放循环的均值，单位为
**ms/rank 交付**。排除 source/manifest/destination 准备、CUDA、注册、
原生提交、网络和 D 等待；不同线程调度下的线上收益尚待测量。

| Case | 阶段 | 原路径 | 所选层视图 | CPU 降幅 |
|---|---|---:|---:|---:|
| 99401 | bootstrap | 1.602 | 0.999 | 37.7% |
| 99401 | steady | 1.063 | 0.487 | 54.1% |
| 99402 | bootstrap | 1.615 | 1.032 | 36.1% |
| 99402 | steady | 1.054 | 0.480 | 54.5% |

这是可确认的本地 CPU 固定开销改善，**不能据此推算每 token 约 420 ms 的
KV 等待缩短多少，也不能与此前 CPU 或 GPU 实验的节省相加**。

## 真实 KV 字节与工作量核对

捕获数据只含实际被消费的选定 KV，未包含完整 Prompt KV。
回放为两 rank 重建 component-major CPU source，总计 123805696 bytes，
并声明 page_size=1 的本地布局和符号身份。所有未捕获行填为 0xA5，
只查询已捕获行；相同 Prompt 行的重复捕获必须 uint8 字节相等。
这不是原生 Entry、完整 Prompt 夹具或新的 CAGRA 召回实验。

每个 Case 有 15×28=420 个消费 bank，两个 Case 共 840 个。按相同单调 CPU
cache 缺失顺序构造每个 rank 的交付；直接从捕获的 K/V 构建独立字节 oracle。
原/新路径 payload 与 manifest 哈希相等，完整 source 前后哈希不变。
此次验证选定 KV 的打包，没有重新执行目标模型 forward 或实际 D bank 安装。

| Case | 已捕获 / 未捕获 head-token 行 | steady rank 交付 | steady 行 | steady bytes | source 视图总数：原→新 |
|---|---:|---:|---:|---:|---:|
| 99401 | 5008 / 236800 | 714 | 3815 | 1953280 | 39984 → 1428 |
| 99402 | 5001 / 236807 | 716 | 3796 | 1943552 | 40096 → 1432 |

bootstrap 分别为 1193/1205 行、610816/616960 bytes、各 56 次交付；
视图总数均从 3136 降到 112。原/新路径的行数、字节数和选择顺序保持一致。
视图不是 KV 副本；省掉的是重复 slice/view/reshape 对象准备。

最初还回放了 `oasis_ready_kv02/capture`。虽然 Case ID 相同，其输入 SHA256
与上述轨迹不同，属于另一份捕获。该次独立字节检查和 CPU 时间保存在
`real_kv01.json`，不混入主对照，不将不同轨迹当成同一请求比较。

## 验证与源码证据

- 最终 gate：**333 passed，34 个真实 CUDA 用例跳过**，1 个已有
  asyncio_mode 配置警告。实际 CPU 张量、store、HTTP、接收与 native 适配器代码
  参与测试；CUDA/native 的策略模拟仅验证控制和生命周期。
- 两 rank、float16/bfloat16/float32、单层/多层/非零 layer_start、物理页乱序、
  final partial page，均与独立 uint8 oracle 核对。实际源 storage 的 reshape
  调用计数验证单层 56→2、多层按唯一层数建立视图，没有增加 gather/KV 分配。
- 未选中层的元数据在一次成功调用后被修改，下一次仍在写入前拒绝。
  后续 group 的 layer/head/padding、目标别名和对齐错误同样在写入前拒绝。
- 部分 copy 抛错不会返回成功计数；同步与注册 UNKNOWN 保留原有 staging、
  Entry/index 租约和预算，正常路径保持原有两个 store 完成同步。
- 六个新增真实 CUDA 用例覆盖两 rank、三种 dtype 和非默认 stream；没有 GPU，
  全部跳过。未来通过也不能代替原生 RDMA 或线上服务 gate。
- 四个 benchmark Python 文件通过 AST 解析，launcher 通过 bash `-n`。
  最终 gate 和主回放的源码逐文件按 LF 规范化与提交 `dfa4c8f9e` 核对。
  主输入哈希与此前 CPU-cache 报告一致；用户原有 AGENTS.md 修改未提交。

结构化报告见同名 JSON；原始日志、每阶段 20 次试验的五轮对照、输入哈希、
源码证明、保存脚本和历史轮询来源保存在
`artifacts/v_selected_views_20261004/evidence.tar.gz`。

## 下一步的顺序

1. 新 GPU 可用后更新过期的节点连接信息和模型路径，先跑真实 CUDA 和
   native RDMA gate，再跑独立 `v-selected-views` ABBA 全路径比较。
2. D 配置完全一致，保留当前快建图、V/CAGRA、Oasis 配对 Decode、
   Top4、capacity32、两个 worker、max_new16、相同 Prompt 和 warmup。
   只切换 V 的所选层视图选项，不合并此前实验。
3. 对照入口要求实际两 rank 模式、每次交付实际视图数 56/2、原有成功同步、
   同一候选/传输/安装预算、输出和清理，并保存 native 提交计数。
   同时报告 V pack 与 pack_fence、注册、提交、D 等待和客户端 TPOT。
4. 根据线上阶段占比选择下一项：register 占主要时间时评估有界 V staging/MR
   复用；pack_fence 主要在等待其他设备工作时评估更精确的 stream/event
   完成证明；控制阶段仍慢时定位源退休和线程排队。每项独立实现与本地提交。

当前已完成 CPU 可验证的实现和收益对照；GPU/原生交付收益等待新资源验证，
默认保持关闭。
