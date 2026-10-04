# V 稀疏 KV 行批量打包：CPU 实测与待验证范围

2026-10-05 整理，分支 `codex/pvd-oasiskv`，仅本地提交。实验产物目录沿用
`v_manifest_rows_20261004`。CloudLab 租约已过期，本机 Torch 2.14.0+cpu；
本报告测量 CPU 打包路径，尚未测量 CUDA、原生 RDMA、D 等待或 TPOT 收益。

## 已完成步骤

| 步骤 | 本地提交 | 内容 |
|---|---|---|
| 计划 | `df4df2a83` | 比较 manifest 与逐行打包机会，限定预算、归属和公平对照 |
| 实现 | `bcab18ac6` | 默认关闭的索引批量打包、生命周期验证及线上对照入口 |
| 证据 | 本报告与同名 JSON | 冻结提交源码、真实 KV 输入、字节证明与五轮 CPU 对照 |

## 机会与实现

V 为每层交付少量 KV 时，原路径对每个 token 的 K、V 分别调用 `copy_`。
所选层视图已经减少了 component 视图创建，但多行交付仍有逐行 Python
调度和 Torch API 调用。此次针对这项固定开销进行批量打包。

先比较了两个 CPU 探针。manifest 指纹减少中间 tuple 复制仅节省约 2–3 μs，
没有实现该改动。索引 gather 原型让实际校验及打包从 244–248 μs 降至
183–184 μs；加入完整预算、归属与索引检查后的正式收益见下表，原型的
收益不能作为最终结果。

每次交付准备一个独立 `SparseRowIndexWorkspace`，将逻辑 token 与本地 head
映射为 `token * kv_heads_per_rank + local_head`。多行 group 对 K、V 各调用
一次 `torch.index_select(..., out=...)`，直接写入既有 staging；单行 group
保留 `copy_`。全部 group 都是单行时不创建 workspace。

group 次序、token 次序、原始候选、完整布局校验和传输字节保持一致。
index 元数据在分配前计入现有预算；CPU/CUDA 索引和视图由交付所有者保留，
沿用原 pack fence、outer fence 与 native adapter fence。构造或拷贝失败时
先取得完成证明；UNKNOWN 保留索引、Entry/index、staging/MR 及预算。

`out` 尺寸不匹配可能触发调整和存储重分配，因此拷贝前额外检查索引长度、
类型、设备和存储归属，输出继续使用已有精确形状校验。
[PyTorch 2.14 index_select 文档](https://docs.pytorch.org/docs/2.14/generated/torch.index_select.html)
说明了这项 API 行为。CPU 测试也核对输出指针与字节，真实 CUDA 对应测试尚未执行。

启用要求 V 同时采用 ordinary CUDA staging 和所选层视图，并使用 Mooncake。
该实验与 Triton、direct sparse PUT、连续区间打包和 fence reuse 互斥；
`--experimental-indexed-sparse-packing` / `PVD_INDEXED_SPARSE_PACKING=1` 默认关闭。
当前快建图、V/CAGRA、Q、Oasis Decode 及 P→D bootstrap 配置没有改动。

## 公平 CPU 对照

两组都使用所选层视图，唯一差异为索引批量打包。使用相同两份真实 KV
捕获、预分配 source/manifest/destination、Torch 单 CPU 线程，执行 5 轮
ABBA：base_a、opt_a、opt_b、base_b，每个阶段共 20 个 arm 样本。

计时包含实际 CPU helper、回放调度、完整校验，以及每次交付的索引预算
申请、Python 元数据、host tensor、group 视图与释放；没有跨交付缓存索引。
计时排除 source/manifest/destination 准备、staging 分配和注册、index lease、
CUDA/H2D、原生提交、网络及 D 等待。

单位：**μs / rank 交付**。

| Case | 阶段 | 逐行打包 | 索引批量打包 | 降低 |
|---|---|---:|---:|---:|
| 99401 | bootstrap | 606.39 | 219.19 | 63.9% |
| 99402 | bootstrap | 638.55 | 225.79 | 64.6% |
| 99401 | steady | 235.70 | 217.49 | 7.7% |
| 99402 | steady | 246.76 | 217.91 | 11.7% |

bootstrap 的 group 更大，能合并更多逐行调用；后续交付较小且包含单行，
新增索引准备和归属检查抵消了一部分收益。正式 steady 降低约 8%–12%，
没有沿用原型约 25% 的数字。

同一实验两组可以比较；不同轮次机器负载和绝对时间不同，不能与之前
所选层视图或布局序列化实验相减、累加。这里的 helper 时间也不能直接乘以
层数推算 D 等待，因为真实流水线还有 GPU、网络、并发与依赖关系。

## 真实 KV 字节与调用核对

输入沿用以下捕获，SHA256 与前两份报告一致：

| Case | 输入 SHA256 |
|---|---|
| 99401 | `4983da4f677cf94a3a270981af9294be1b7da66235da8e1672de881b153ceff4` |
| 99402 | `9e3daabce259caada81ebfcf43dbd4dc8f210b938604c425eed838acc67a55e9` |

每份捕获包含 420 个消费 bank，共 840 个。它们只包含已选 KV，不能重建完整
Prompt 内容。回放声明 CPU component-major source，page_size=1，使用本地
符号身份；未捕获行填充 0xA5 且从不读取，不冒充原生 Entry 或完整 Prompt。
已知/未知行分别为 5008/236800 和 5001/236807。

独立保存的 KV oracle 与两组实际 staging 的 uint8 字节逐一相同。源字节
哈希不变；bootstrap、steady 的 wire/manifest 哈希、行数、字节和视图数也
与所选层视图及布局序列化报告相同。

| Case | 阶段 | 交付数 | KV 行数 | wire bytes | 原 copy_ 调用 | 新 copy_ + index_select 调用 | CPU index bytes 合计 |
|---|---|---:|---:|---:|---:|---:|---:|
| 99401 | bootstrap | 56 | 1193 | 610816 | 2386 | 0 + 224 | 9544 |
| 99402 | bootstrap | 56 | 1205 | 616960 | 2410 | 0 + 224 | 9640 |
| 99401 | steady | 714 | 3815 | 1953280 | 7630 | 614 + 1726 | 28064 |
| 99402 | steady | 716 | 3796 | 1943552 | 7592 | 606 + 1732 | 27944 |

每次 rank 交付的 source component 视图数仍为 2。上表是实际 Torch API
调用次数，不能称为测量到的 CUDA kernel 数。CPU int64 索引 bytes 是额外
元数据量，不计入网络 KV payload，也不是测量到的 GPU H2D bytes。

scratch 保守预算按 `64 * index_count + 512 * group_count` 计费；本次
bootstrap 每次交付峰值 3520 bytes，steady 峰值 2368 bytes。回放每次释放后
预算归零。真正 CUDA 路径还要创建 device 索引和上传，可能抵消 GPU 收益。

## 验证与修复记录

最终受影响 gate：**476 passed，46 actual CUDA skipped，1 个已有配置警告**。
runner 仅引导 package frontend；实际执行 Torch/PVD/HTTP 代码，CUDA/native
生命周期政策场景使用显式标注的 CPU 测试替身，不计作 GPU 资格验证。

- 实际 CPU tensor 覆盖双逻辑 rank、三种 dtype、混合/单行 group、多层、
  非零 layer 起点、物理页重排、最后部分页、输出 storage 与精确 uint8 字节。
- 每组在写入前检查，拒绝错误布局、旧 workspace、索引尺寸或存储归属。
  预算不足在 scratch 分配前失败；部分拷贝失败不会记录成功计数。
- 既有 store 和 native fences、取消、注册、提交、metadata UNKNOWN 保留
  流程继续核对；构造阶段的 CUDA upload/sync 失败用显式 CPU policy double
  验证隔离与预算保留，不声称实际 CUDA 行为通过。
- 新增 12 个实际 CUDA 用例：两个逻辑 rank × 三种 dtype × 成功/部分失败，
  使用非默认 stream，全部因本机无 GPU 跳过。即使在一张 GPU 上执行双逻辑
  rank，也不等于两张 GPU 的 TP2 原生验证。
- 首次 gate 为 3 failed、154 passed、24 skipped：indexed workspace 被
  同时传入旧 fused 参数。修复为仅在 fused flag 启用时转交；第二次 gate
  为 157 passed、24 skipped，随后扩大为最终 gate。原失败日志保留。

4 个 benchmark Python 文件 AST、Git Bash launcher 语法和 `git diff --check`
通过。171 个源码/测试/对照文件的 normalized-LF 哈希与实现提交 Git blob
及最终 gate、回放记录一致；冻结输入也与前两份报告一致。

## GPU 恢复后的下一步

已准备独立 `v-indexed-pack` full-path ABBA。所有 arm 都开启所选层视图，
D 配置、快建图、candidate/byte budgets 与原始 fences 一致，仅候选 arm
开启索引打包。核对两 rank 实际 health、部署源码、warm 后 native counters、
逐交付 row/index API 计数，以及相同输出、候选、网络字节和安装预算。

这项线上对照尚未执行，现有 launcher 的过期节点需要先换成新租约连接和
模型路径。执行顺序：实际 CUDA bytes/storage/失败用例 → 两物理 rank 的
native 生命周期 → V pack/index upload 和排队时间 → D wait 与 TPOT 公平对照。
根据实际 GPU 成本决定是否启用；继续保持该选项默认关闭。

后续优化优先依据 V source profile 的 allocate、pin_index、pack、fence、
register、submit 成本选择。GPU 恢复前可以完善生命周期与 CPU 证据，无法
确认检索时间量级、D 等待降低或输出质量变化。

## 产物

- 汇总：`benchmark/results/pvd_indexed_sparse_pack_local_20261004.json`。
- 正式原始回放：`artifacts/v_manifest_rows_20261004/real_kv01.json`。
- gate：`artifacts/v_manifest_rows_20261004/gate{01,02,03}/{unit.txt,status.json}`。
- 探索性探针：`artifacts/v_manifest_rows_20261004/probe.{py,json}`。
- 提交源码核对：`artifacts/v_manifest_rows_20261004/source_proof.json`。
- 归档：`artifacts/v_manifest_rows_20261004/evidence.tar.gz`，包含失败与通过日志、
  探针、正式回放、核对脚本/证明及当前和两份前序 JSON；捕获输入通过哈希
  关联，未重复打包大体积 `.pt`。
