# P→V→D 原生 GPU 缓冲区接力验收 / Native GPU-buffer relay acceptance

`run_pvd_native_relay.py` 是独立的一次性硬件验收工具，不是 PVD 生产服务。
P、V、D 各运行一个进程；P 的 GPU 源区经 Mooncake WRITE 到 V 的 GPU 中间区，
V 在确认首段安全终态及字节一致后，**复用同一份已注册的 GPU 中间区**作为第二段
WRITE 的源区，把字节送到 D 的 GPU 目标区。TCP 只传 descriptor、终态与校验回执，
不承载 payload。

`run_pvd_native_relay.py` is a one-shot hardware acceptance tool, not the
production PVD service. P writes GPU bytes to V, and V reuses the very same
registered GPU allocation as the source of its WRITE to D. TCP carries only
descriptors and terminal/verification acknowledgements, never the payload.

## 安全边界 / Safety boundary

- 两段 WRITE 均需 Mooncake 的本地安全释放证明；单独的 `FAILED` 状态不够。
  未知完成状态时保留源/目标 GPU MR 至进程退出，不猜测完成、不复用地址。
- V 必须在首段终态及 GPU 字节校验之后，才可提交第二段 WRITE；第二段完成前
  不注销中间区。D 校验 P 原始 pattern，三端正常退出要求零遗留 MR/句柄。
- 每个 role 使用独立进程和私网地址。建议先在 `mlx5_0` 单 rail 下分别验收
  GPU 0、GPU 1；两次独立小样本不等于真实 TP 同步。

- Both WRITEs require a locally safe-to-release Mooncake proof; `FAILED` alone
  is insufficient. An unknown completion retains the GPU MR until process exit.
- V does not submit the second WRITE before the first has terminal proof and
  a GPU-byte check. It unregisters its intermediate region only after the
  second WRITE is safe. D checks the original P pattern; healthy exit requires
  no live MR or transfer handle on any node.
- Separate processes use private addresses. Independent GPU-0 and GPU-1
  samples do not establish synchronized model TP.

## 三节点运行 / Three-node run

在三台节点各自的相同代码版本及已隔离安装的
`mooncake-transfer-engine==0.3.13.post1` 环境中，先 D、后 V、最后 P。下例使用
CloudLab 私网地址，`--gpu-id` 在三端必须相同。等待 D 和 V 输出 `ready` 后再启动
下一端；每端最终都必须输出 `status=passed`。

Use the same code revision and isolated, pinned Mooncake dependency on all
three nodes. Start D, wait for `ready`; then V, wait for `ready`; then P. Run
the command from the repository root with the environment's Python and
`PYTHONPATH=python:<isolated Mooncake target>` (order may be reversed if the
target is prepended). The concrete command on each node is:

```bash
python test/registered/disaggregation/run_pvd_native_relay.py D \
  --p-ip 10.0.1.1 --v-ip 10.0.1.2 --d-ip 10.0.1.3 \
  --rail mlx5_0 --gpu-id 0
```

把 `D` 依次替换为 `V`、`P`，分别在 node-1、node-0 运行。rank1 另开一轮，
三端都设置 `--gpu-id 1`。默认端口为 28175/28176，长度 4096 bytes；并发运行
两轮时必须给它们配置不同端口。若端口、地址或对端不符，验收失败。

Replace `D` with `V` on node-1 and `P` on node-0. Repeat separately with
`--gpu-id 1` on all three nodes. Defaults are ports 28175/28176 and 4096
bytes. Concurrent rank runs need distinct ports. A wrong address, rail,
length, port or peer is a failure, not a skip.

## 证据范围 / Evidence boundary

单次通过只验证原生 Mooncake 对一个请求级 GPU 缓冲区的连续两跳小样本、终态和
MR 生命周期。不验证完整 Prompt KV 布局、生产 Scheduler、检索/CAGRA、并发压力、
多 rail、模型输出或 GPUDirect 零拷贝性能。

A pass validates only a small same-buffer two-hop native Mooncake sample and
its terminal/MR lifetime. It does not validate full Prompt KV layout,
production Scheduler, retrieval/CAGRA, concurrency, multi-rail, model output
or GPUDirect zero-copy performance.
