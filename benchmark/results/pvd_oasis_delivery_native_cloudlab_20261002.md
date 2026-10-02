# 合并提交与接收区复用：CloudLab 原生门槛通过

2026-10-02，分支 `codex/pvd-oasiskv`，tag `oasis_delivery_slots_gate01`。
计划 `7d5d262fd`；合并 reserve＋start `6f5cab2ca`；接收区复用 `9b8b5dc0c`；证据工具 `f28b0f5ea`。

## 已验证的结果

| 验证 | 结果 |
|---|---|
| CloudLab CPU 生命周期回归 | **512 passed，4 skipped，1 warning**，8.13 s |
| 接收记录子集重复执行 | **9 passed**，2.14 s；已包含在上面的 512 项中 |
| 原生 Mooncake 本地 session PUT | **48 次精确字节检查全部通过** |
| 物理接收区注册／注销 | **4／4**；48 个独立交付 generation |
| 物理接收区保留容量 | 4 × 32768 B = **131072 B**，退休后 0 B |
| 最终注册区、native handles、预算 | 全部归零，隔离状态为空 |

本报告完成正确性和生命周期门槛，不给出在线 Decode 或客户端加速结论。
合并 RPC 与注册复用的耗时收益需要分别在完整 P/V/D 路径做公平对照。

## 验证范围

合并 RPC 保留原 reserve 校验和 start 路径；lost reply 仍可能已经提交 native WRITE，
接收区必须继续持有并使用完整 WriteIdentity fence。接收槽为请求所有，每 rank 最多两个，
bootstrap 和 Decode 的两个独立 2-worker executor 复用同一组物理 MR。
每次交付仍有独立 UUID generation、精确 byte extent、manifest 和 worker 内接收记录；
复用不会共享交付身份或取消 terminal proof。

原生 gate 在 **D=node2 GPU1** 上执行，调用方当前设备为 **GPU0**；
两 V rank 对应接收路由都使用实际部署的 `mlx5_0`／同一 D session。
行数 1、2、8、16、32、64 × 两 rank × 两 worker × 两 executor，共 48 次。
每行是一个 head 的 K/V 对，FP16、head_dim=128，因此每行 512 B。
每次实际 native PUT 都确认 terminal_success、精确 transferred_bytes 和本地 cleanup，
再做保守 GPUDirect receive ordering、同步消费及字节比较；全部 generation 不重复。
退休仅传原始 physical RegisteredMemory 给 engine，调用方 CUDA 当前设备保持 GPU0。

CPU 测试包括 busy／超容量／错误身份的发布前拒绝，lost reply／ACK／迟到写入 fence，
registration、RDMA、CUDA ordering／D2H 和注销 UNKNOWN 保留，以及阻止复用未知槽。
其中 CUDA／native 失败路径使用 CPU double；本轮未对真实 GPU/RDMA 注入 UNKNOWN。
CUDA receive ordering 失败即保留 UNKNOWN，后续 stream drain 成功也不能清除它。
4 项跳过记录与 pytest 配置 warning 保存在完整日志中。

这里的原生 PUT 使用**本地 session**，不能代替 V→D 网络、并发压力、长 Decode、
TP2 或生产故障恢复测试。正式服务 D GPU1 的网络收益由独立线上对照验证。
本门槛执行前后 V/D 两节点共四张 GPU 均为 0 MiB，完整查询输出已保存。

## 可复核证据

同名目录保存 `gate.tar.gz`（全部部署 bundle、argv、逐步状态、日志、source hashes、
原生完整 48 条 observations）、独立 `native.json`、完整 `unit.txt`、
`records_unit.txt`、源身份和计数记录。
归档内容逐项与原始 artifact 比较；部署 bundle 的每个文件 hash 与 local/V/D 的记录一致。
`manifest.json` 对文本采用 LF 规范化，对 `.tar.gz` 使用原始字节 SHA-256；完整 gzip CRC 已校验。

原生语义可在项目根目录独立复核：

```powershell
python -B -c "import json,sys; from pathlib import Path; sys.path.insert(0,'benchmark'); from verify_pvd_oasis_latency_evidence import verify_native_slots; verify_native_slots(json.loads(Path('benchmark/results/pvd_oasis_delivery_native_cloudlab_20261002/native.json').read_text(encoding='utf-8'))); print('native semantic gate passed')"
```

复现需要 CloudLab 的既有隔离依赖和空闲 GPU；完整原始命令在归档的 `native.command`
与 `unit.command` 中，所有日志、缓存和临时目录都指定为远端 checkout 的 `artifacts/`。
