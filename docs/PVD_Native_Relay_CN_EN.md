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
  不注销中间区。D 校验 P 原始 payload，三端正常退出要求零遗留 MR/句柄。
- 每个 role 使用独立进程和私网地址。建议先在 `mlx5_0` 单 rail 下分别验收
  GPU 0、GPU 1；两次独立小样本不等于真实 TP 同步。

- Both WRITEs require a locally safe-to-release Mooncake proof; `FAILED` alone
  is insufficient. An unknown completion retains the GPU MR until process exit.
- V does not submit the second WRITE before the first has terminal proof and
  a GPU-byte check. It unregisters its intermediate region only after the
  second WRITE is safe. D checks the original P payload; healthy exit requires
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

添加 `--payload-kind packed-kv` 可使用本项目 `kv_packer` 生成/解包合成的
2-layer、2-KV-head、FP16 Prompt KV。默认 4096 bytes 对应 32 个槽位，
其中 30 个有效 token、最后一页 2 个 padding token；D 同时核对四个 K/V
分量及 padding 未被覆盖。此模式要求长度至少 1024 bytes 且为 512 的倍数。

Add `--payload-kind packed-kv` on **all three roles** to use the project's
actual `kv_packer` on synthetic two-layer, two-KV-head FP16 Prompt KV. The
default 4096 bytes represent 32 slots, 30 valid tokens and two padded
positions in the final page. D verifies all four unpacked K/V components and
that padding is untouched. This mode requires at least 1024 bytes in
512-byte multiples.

## 证据范围 / Evidence boundary

单次通过只验证原生 Mooncake 对一个请求级 GPU 缓冲区的连续两跳小样本、终态和
MR 生命周期。不验证完整 Prompt KV 布局、生产 Scheduler、检索/CAGRA、并发压力、
多 rail、模型输出或 GPUDirect 零拷贝性能。

A pass validates only a small same-buffer two-hop native Mooncake sample and
its terminal/MR lifetime. It does not validate full Prompt KV layout,
production Scheduler, retrieval/CAGRA, concurrency, multi-rail, model output
or GPUDirect zero-copy performance.

## 2026-09-23 CloudLab 验收 / CloudLab acceptance

在提交 `db13c5b9c` 的三个独立 worktree 中，node-0/P `10.0.1.1`、node-1/V
`10.0.1.2`、node-2/D `10.0.1.3` 分别运行同一脚本。三端均使用
`mooncake-transfer-engine==0.3.13.post1`、V100S-PCIE-32GB、`mlx5_0`，
先后对 GPU 0 和 GPU 1 各做一轮 4096-byte 接力。**两轮的 P/V/D 报告均为
`status=passed`**；V 把首段接收注册区作为第二段源，D 的 GPU 字节与 P pattern
一致，三端 health 检查均无遗留 MR/传输句柄。结束后三台节点无接力 GPU 进程
或端口 28175/28176 监听。Linux 同提交的 8 个 CPU 契约用例也通过。

At commit `db13c5b9c`, independent worktrees on node-0/P (`10.0.1.1`),
node-1/V (`10.0.1.2`) and node-2/D (`10.0.1.3`) ran the same tool with
`mooncake-transfer-engine==0.3.13.post1`, V100S-PCIE-32GB and `mlx5_0`.
Both separate 4096-byte runs (GPU 0 and GPU 1) returned `status=passed` on
all three roles. V reused its receiving registration for the second WRITE;
D's GPU bytes matched the P pattern; each role's health check found no live
MR or transfer handle. No relay GPU process or control listener remained.
Eight CPU contract tests also passed on Linux at the same revision.

These two successful rank samples do **not** establish simultaneous TP2 model
execution, Prompt-KV tensor layout, throughput, loss/retry behaviour, native
sparse Delivery, or production request scheduling.

随后在同一提交的三个 worktree 上，同时启动 GPU 0 和 GPU 1 的两套 P/V/D
进程；两套分别使用控制端口 `28175/28176` 与 `28177/28178`，共用
`mlx5_0`。六份最终报告均为 `passed`，且三台节点结束后没有遗留接力进程。
这验证的是同一 rail 上两个独立 GPU 会话可并存；**没有**运行 TP2 模型
collective、请求级并发压力或真实 KV tensor。

A second run started the GPU-0 and GPU-1 P/V/D processes concurrently on
the same three worktrees, with distinct control-port pairs `28175/28176` and
`28177/28178` and shared `mlx5_0`. All six final reports passed, and no
relay process remained on any node. This establishes coexistence of two
independent GPU relay sessions on one rail, **not** TP2 model collective
execution, request-level stress or real KV tensors.

随后用 `--payload-kind packed-kv` 分别在 GPU 0、GPU 1 运行三节点接力：
两轮的 P/V/D 报告均为 `passed`，D 两次均报告
`d_prompt_kv_unpacked=true`。Linux 的 9 个 CPU 契约用例通过，包含故意
篡改打包字节后必须检出不一致；Windows 为 8 passed / 1 skipped（SGLang
serving 依赖 POSIX `resource`）。这证明合成 Prompt-KV 的打包字节可以
经原生两跳传输并正确解包；**不是**真实模型生成的 KV、TP collective、
CAGRA/稀疏刷新或生产服务。

Subsequent separate GPU-0 and GPU-1 `--payload-kind packed-kv` runs passed
on all P/V/D roles; D reported `d_prompt_kv_unpacked=true` in both. Nine
Linux CPU contract cases passed, including a corrupted-byte detection case.
Windows ran eight and skipped the POSIX-only real-packer case. This validates
native two-hop transport and unpack of **synthetic** Prompt-KV bytes, not
model-generated KV, TP collective, CAGRA/sparse refresh or production serving.
