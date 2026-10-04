# KV 布局序列化：删除被覆盖的递归复制

2026-10-04，分支 `codex/pvd-oasiskv`，仅本地提交。CloudLab 租约已过期，
本机 Torch 2.14.0+cpu；以下不是 GPU、原生 RDMA 或 Decode 性能结果。

## 已完成的步骤

| 步骤 | 本地提交 | 内容 |
|---|---|---|
| 计划 | `8334dc410` | 基于实际 CPU 打包探针，限定等价序列化及验证范围 |
| 实现 | `3da9d9ed8` | 去掉被覆盖的元数据复制，增加协议兼容性和真实 KV 对照 |
| 证据 | 本报告与同名 JSON | 冻结源码、基线方法、输入、字节及五轮 CPU 对照 |

## 机会与实现

`KVLayoutSignature.fingerprint` 每次都调用 `to_dict()`，再执行相同的
canonical JSON 和 SHA256。原 `to_dict()` 先执行 `dataclasses.asdict(self)`，
递归复制 56 个 K/V component 的 dtype、shape、bytes 元数据，随后立即
用 `dict(self.extra)` 覆盖刚复制的 extra。对逐层小量 KV 交付，这是一项
重复的 Python 固定开销，与图点数无关。

新实现对顶层普通标量字段直接形成同一个有序字典，再保留既有的 shallow
extra 拷贝。遇到非普通标量字段，例如 nested dataclass、custom object 或
带复杂字段的 subclass，继续使用原 `asdict` 转换与深拷贝路径。

每次仍读取当前 extra 并重新计算 JSON/指纹；不缓存可变布局，不减少任何
component 校验。字典字段及顺序、JSON、SHA256、协议版本、selected/unselected
布局检查、Torch 拷贝、完成同步、MR、授权、ACK 和 UNKNOWN 保留规则保持一致。

这是一项直接采用的等价 CPU 序列化实现，没有增加服务模式或设备选项。
上一轮所选层视图选项仍默认关闭；当前快建图、V/CAGRA 和 Oasis Decode 配置
没有改变。

## 相同输入的 CPU 对照

主输入为此前所选层视图和 CPU-cache 报告相同 SHA256 的两个真实捕获：
`artifacts/oasis_workspace_replay01/capture/{99401,99402}/trajectory.pt`。
5 轮 ABBA，每轮 base_a、opt_a、opt_b、base_b；Torch 固定一个 CPU 线程。
两组均使用同一份预分配 source、manifest、destination，且均采用所选层视图。

对照只临时替换本地 `KVLayoutSignature.to_dict`。参考方法的 AST 与基线提交
`3df6cc9d6` 完全一致，基线 protocol Git blob 的 SHA256 也已冻结。
实际打包、完整元数据校验和指纹计算均执行；每个 arm 结束恢复方法。
这不是在运行中的 V 服务上 monkeypatch，也不是线上 ABBA。

单位：**μs/调用**。bootstrap/steady 是 CPU 打包 helper 加回放循环，
fingerprint 为单独 2000 次调用的均值。source、manifest、destination 准备、
CUDA、注册、原生提交、网络及 D 等待均不在计时范围内。

| Case | 项目 | 原序列化 | 新序列化 | 变化 |
|---|---|---:|---:|---|
| 99401 | steady 打包 | 363.36 | 312.04 | 降低 14.1% |
| 99402 | steady 打包 | 377.01 | 312.04 | 降低 17.2% |
| 99401 | fingerprint | 82.06 | 37.30 | 降低 54.5% |
| 99402 | fingerprint | 81.02 | 37.93 | 降低 53.2% |
| 99401 | bootstrap 打包 | 741.50 | 661.28 | 降低 10.8% |
| 99402 | bootstrap 打包 | 727.61 | 740.96 | **增加 1.8%** |

保留 bootstrap 回退结果，不声称所有阶段都有收益。单独的 fingerprint
已包含在打包 helper 内，不能把两项节省再相加。此次基线均值与上一轮实验
不同，因此也不能用不同轮次的绝对时间相减或累加优化收益。

探索性 cProfile 用于定位递归调用，原始结果已归档。它会改变 Python 执行
成本，没有用于上述计时，也没有用其 cumulative 比例预测线上收益。

## 真实 KV 与协议核对

捕获只覆盖选定 KV，不包含完整 Prompt。为每个 Case 声明重建两 rank
component-major CPU source，总计 123805696 bytes；未捕获行以 0xA5 填充，
只读取已捕获行。使用 page_size=1 的本地布局与符号身份，不冒充原生 Entry。
独立 oracle 直接来自捕获 K/V；原/新序列化均调用实际 Torch 打包 helper。

两个 Case 共 840 个消费 bank；每个 Case bootstrap 均有 56 次 rank 交付，
分别为 1193/1205 行、610816/616960 bytes。

| Case | 已捕获 / 未捕获 head-token 行 | steady rank 交付 | steady 行 | steady bytes |
|---|---:|---:|---:|---:|
| 99401 | 5008 / 236800 | 714 | 3815 | 1953280 |
| 99402 | 5001 / 236807 | 716 | 3796 | 1943552 |

两组 layout JSON、manifest、wire payload 哈希完全相等，行数、字节数、
source 视图数与上一轮报告一致；完整 source 前后哈希不变。
没有重新运行目标模型 forward、D bank 安装或 CAGRA 召回测试。

## 验证与证据

- 最终 gate：**612 passed，5 个 subtests passed，16 个真实 CUDA 用例跳过**，
  1 个已有 asyncio_mode 配置警告。
- 对照旧序列化的 dictionary、字段顺序、普通/规范化 JSON、hash 和 wire
  roundtrip；覆盖 56-component 元数据、嵌套 list/tuple、UserDict、非 ASCII
  文本、复杂 model_revision、scalar/complex subclass 字段。
- 保留 extra 顶层副本与嵌套共享语义；嵌套修改立即改变 fingerprint。
  普通布局不会执行被覆盖的 asdict；复杂字段保留原有转换和深拷贝。
- 普通/所选层打包均在写入前拒绝未选中 component 的非法 metadata，以及
  已改变但未重新匹配指纹的 tag。原有部分失败与 UNKNOWN 生命周期 gate 通过。
- 最终 gate 覆盖 core/HTTP、Prompt 分页、chunk upload、授权、fan-in、sparse
  delivery/receiver、native 适配器策略模拟、source diagnostics 和完成同步。
- gate01 为 93 passed/6 CUDA skipped。gate02 在收集 direct-bootstrap 测试时
  因 Windows 缺少 Linux `resource` 模块中止，没有执行测试；保留日志。
  gate03 排除此单个模块后通过，Linux direct-bootstrap 集成仍待验证。
- 新 benchmark 通过 AST 解析；最终 gate、主回放和已提交源码按 LF 逐文件核对。
  基线方法 AST/blob、主输入和上一轮 wire 哈希均核对一致。

完整 trial 和机器可读结果见同名 JSON。原始 gate 日志、cProfile、保存脚本、
参考方法证据和逐文件源码证明归档于
`artifacts/v_layout_serialization_20261004/evidence.tar.gz`。
用户原有 AGENTS.md 修改未加入提交，没有上传 GitHub。

## 后续顺序

1. 继续用本地可复现探针检查 sparse manifest 的 hash/wire 构造和逐行 copy
   调度，只有确认重复工作及兼容边界后再实施独立优化；保留各阶段本地提交。
2. 新 GPU 可用后先更新节点/模型信息，完成 Linux direct-bootstrap、真实 CUDA
   和 native RDMA gate，再用固定 baseline/candidate revision、相同启动配置、
   Prompt、warmup 和候选/传输预算比较全路径。部署时仅替换此次 protocol 实现。
3. 测量 V 的 pack、pack_fence、register、submit 和源退休，D 的 KV 等待、
   客户端 TPOT 及并发吞吐；这些实验完成前，不能声称 CPU 节省已经缩短
   当前约 420 ms/token 的 KV 等待。
