# CUDA 组件严格验收 / Strict CUDA component acceptance

本入口只验证已实现的独立 CUDA packing、bank、rank participant、tiled attention。
**不验证原生 RDMA、真实模型 forward、CAGRA、生产 Scheduler、实际多 GPU TP 或性能。**
V 交付测试仍使用 fake transport，不能把成功结果当作网卡或 GPUDirect 验收。

This gate validates standalone CUDA components, not production serving, model
forward, CAGRA, native RDMA, multi-GPU model TP or latency. V delivery uses a fake
transport even when its source allocation/copies are real CUDA.

## 操作 / Run

在有 CUDA 版 torch 与现有测试依赖的隔离环境，仓库根目录运行：
Use an isolated environment with CUDA-enabled torch and the project's test
dependencies. From the repository root:

```bash
PYTHONPATH=python python test/registered/disaggregation/run_pvd_cuda_acceptance.py \
  --expected-gpu V100S --timeout-seconds 300
```

检查 `cuda:0`（按 `CUDA_VISIBLE_DEVICES` 映射）；不安装或改写驱动、依赖、模型。
`--expected-gpu` 可省略，省略后不会假装测试卡是 V100S；报告记录实际设备名、计算
能力、显存、Python、torch 和 CUDA build。此参数不改变应用中 HCA/rail 配置。

The gate uses logical `cuda:0`, honors `CUDA_VISIBLE_DEVICES`, and changes no
drivers, dependencies or model files. The optional GPU-name assertion does not
impose a model/software version or alter HCA/rail routing.

## 成功条件 / Success criteria

- 必须精确执行 9 个固定真实 CUDA 用例，包括实际 driver SYNC_MEMOPS 设置/回读、失败后 packing 回收、非默认 stream
  bank、rank 握手、FP16/FP32 attention 数值和两轮切换消费。
  Exactly nine explicitly named CUDA tests must run and pass; missing,
  duplicate, extra, skipped, failed or errored cases are rejected.
- 使用独立 pytest 子进程和临时 JUnit 报告，不依赖解析“passed”字样。禁止
  `PYTEST_ADDOPTS` 隐式筛选测试；子进程关闭自动第三方插件加载和 Python 优化模式。
  A fresh subprocess/report prevents reuse of stale evidence. No skipped test
  is promoted to a pass, even if pytest itself exits zero.
- exit 0 / `status=passed` 仅表示上述组件用例通过。
  exit 1 / `status=failed` 表示测试、超时、报告或执行出错。
  exit 2 / `status=blocked` 表示无 CUDA、依赖/运行时预检失败或设备不符。
  Assertion-disabled mode / invalid arguments are refused by the CLI parser.

CPU 本地运行已实际报告 `blocked` / CUDA unavailable。17 个 CPU parser/runner
回归覆盖缺测、skip、XML 错误、重复、超时、非零退出和证据范围标志。它们不是
CUDA 测试通过记录。首轮测试抓到 XML 空元素的布尔值导致 skip/failure 被忽略的
问题，现按 tag 检查；该缺陷未进入提交。

Local execution with CPU-only torch reports blocked. Seventeen CPU harness tests
exercise refusal and evidence handling, not device computation. Initial tests
caught empty XML elements being falsey; tag-based validation now refuses skips
and failures. That defect was fixed before committing.

入口加入后 Windows 全量 **1974 passed / 23 skipped**，WSL 入口定向 **17 passed**。
After adding the gate, Windows full regression is 1974/23; WSL harness regression
is 17 passed. These counts do not change the blocked local CUDA acceptance result.

## CloudLab V100S 实测 / CloudLab V100S execution

2026-09-23，在 `clgpu020` 的独立 `82de6a548` 验证 worktree 中，以
`Tesla V100S-PCIE-32GB`、PyTorch `2.9.1+cu128`、CUDA 12.8 工具链运行上述严格入口，
报告 `status=passed`，固定的 **9/9** CUDA 用例全部通过，无 skip。
测试用 `pytest==8.4.2` 安装到数据盘上的独立 `pvd-test-pydeps` 目录；原推理环境
未安装测试依赖，首次运行因此报告缺少 `pytest`，不属于组件失败。

On 2026-09-23, the isolated `82de6a548` worktree on CloudLab `clgpu020`
returned `status=passed`: all **9/9** fixed CUDA cases ran on a
`Tesla V100S-PCIE-32GB` with PyTorch `2.9.1+cu128` and the CUDA 12.8 toolkit,
without skips. `pytest==8.4.2` was supplied through a separate data-disk
test-dependency directory; the first attempt lacked pytest and did not execute
the CUDA tests.

This is component-level evidence only. The report explicitly keeps
`production_gpu_rdma_validated=false`, `cagra_validated=false`,
`model_forward_validated=false` and `performance_validated=false`. It must not be
used to claim native Mooncake delivery, production Scheduler activation or
multi-rank model TP acceptance.
