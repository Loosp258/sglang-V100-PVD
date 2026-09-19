# PVD 请求独立预取：基础设施实施记录

## 基线和用户决定

- 分支：`pvd-disaggregation`。
- 基线 commit：`7beb1cc58c7ddfed59cbf139588e3cd8364ae23e`。
- 用户要求新请求只初始化自己，不触发旧请求额外刷新、不重置旧请求时钟、不取消旧请求预取。
- 独立 draft 小模型仅预测 token；目标模型独立 probe 生成对应 Q；预测不进入正式输出。
- 目标/draft 模型尚未选择，probe 的具体模型接入尚未实现。
- 后续用户明确要求：draft model 名称/路径自定义、revision 可选；不要求先选模型才能开发。按能力校验加载，实际版本用于实验复现记录。
- 用户确认目前本地无 V100S 硬件。开发机为 Windows、RTX 4060 Laptop，未安装 CuPy/cuVS。
- 完整目标及非目标保存在 [设计文档](../specs/2026-09-20-pvd-prefetch-design.md)。

## 已实现，但尚未接入服务

### 请求级逻辑时钟

`python/sglang/srt/disaggregation/pvd/prefetch.py`：

- `PrefetchClock` 与线上 `RefreshClock` 分开，线上行为不变。
- `PrefetchTicket` 固定 delivery 身份、round、正式前缀计数、目标安装边界。
- 首次安装位于 D count 0；后续窗口按 M 推进。
- 在 `boundary - r` 发起预取，只在 `boundary` 安装；`r=0` 为同步对照。
- `IDLE -> IN_FLIGHT -> READY -> IDLE`，完成安装后才能推进轮次。
- READY 不等于已安装；提前到达不移动周期，迟到不允许正式计数越过边界。
- 每请求最多一个 pending ticket；错误轮次/身份不能推进状态。
- 新请求不操作旧请求对象，没有 batch 计时器。
- `close()` 关闭逻辑对象并保留 pending 身份，不负责取消原生传输或释放任何资源。

这是纯逻辑状态机，不管理 GPU buffer，不是 MR/fence 证明。
将来需要在 scheduler 线程串行推进；后台通知排队回 scheduler。
调用者必须使用进程/请求 incarnation 唯一的 delivery prefix。
完整 query/index/Entry/epoch/generation wire 绑定尚未实现。

### V100S CAGRA 预检工具

`scripts/pvd/check_cagra.py` 不导入 SGLang，不安装依赖，不修改运行环境。
默认在任何开发环境收集信息（不导入 CuPy/cuVS、不执行 GPU kernel）：

```bash
python scripts/pvd/check_cagra.py
```

- 默认 `--mode inventory`，返回 `status=collected, cagra_test=not_run`。缺 GPU 或缺 GPU 库不阻止采集，更不阻止通用开发。
- 有候选 Linux/V100S 环境后，显式运行 `python scripts/pvd/check_cagra.py --mode smoke`。
- smoke 默认要求所选 GPU 名称包含 `V100S`；报告实际 GPU、驱动、CUDA、Python 和相关包版本。
- smoke 尝试真实 `cagra.build` / `cagra.search`，包含设备同步以发现异步错误。
- 默认 synthetic 配置：4096 rows、32 queries、128 dimensions、FP32、inner product、NN-descent、Top-10。
- 使用非归一化独立 query，与 CPU 精确结果比较 recall；默认 0.90 仅为合成烟测阈值，不是模型质量阈值。
- 输入规模设上限；不能据此推断生产建图的峰值显存。
- JSON schema 更新为 `pvd_cagra_probe_v2`；采集成功不等于实测通过。显式 smoke 失败退出码为 1，不进行距离度量或后端自动回退。
- 可显式指定 `--build-algo ivf_pq` 等候选配置；一个组合失败不等于所有 CAGRA 版本都不支持 V100S。

工具按 [CAGRA API](https://docs.nvidia.com/cuvs/api-reference/python-api-neighbors-cagra) 编写能力探测。
[当前安装要求](https://docs.nvidia.com/cuvs/installation) 不能替代历史版本/V100S 验证；本次未锁定可用版本。

首次单模式脚本的本机实测结果（现在对应 `--mode smoke`）：`status=failed, stage=import_cupy, ModuleNotFoundError`，退出码 1。
这证明缺依赖被正确报告，不证明 CAGRA 或 V100S 兼容性。
没有安装 cuVS/CuPy，也没有替换现有 torch/Mooncake。

## 验证记录

- 修改前，原 11 个 PVD CPU 测试文件：324 passed。
- 新增时钟和预检 CLI 测试：49 passed。
- 新增测试中包含真实 `PVDDecodeRefresher.refresh` 的选择回归：
  A=10、B=16、C=3、新请求=0，只获取 B/新请求；旧时钟不重置。
  该测试使用 fake session 数据操作、单 rank collective stub 和 fake client，
  不是实际多 rank/GPU/RDMA 测试。
- 初次新增测试发现 `--min-recall nan` 的失败报告无法 JSON 序列化，已修复并加入回归覆盖。
- 完整 13 文件 PVD CPU 回归：373 passed in 3.50s。
- 4 个新增 Python 文件通过 Ruff format 检查和 E9/F401/F821/I 检查；`git diff --check` 通过。

### 分层检测修订后的复核

- 新增 4 项测试：默认/显式 inventory 不调用 GPU probe，缺失 nvidia-smi 仍可采集，显式 smoke 才调用 probe。
- 完整 CPU 回归：377 passed in 3.38s；修改的 Python 文件通过格式和 E9/F401/F821/I 检查。
- 本机默认命令：退出码 0，`status=collected, cagra_test=not_run`。
- 本机 `--mode smoke`：退出码 1，`cagra_test=failed, stage=import_cupy`，未掩盖缺依赖。
- 未安装依赖、未加载/下载模型、未改变线上服务；自定义 draft 加载接口仍属于后续实现，当前只是明确其配置约束，不存在可用的新模型启动参数。

复现全部 PVD CPU 测试（PowerShell）：

```powershell
$pvdTestFiles = @(rg --files test/registered/disaggregation -g 'test_pvd*.py')
& .venv/Scripts/python.exe test/registered/disaggregation/run_pvd_cpu_tests.py @pvdTestFiles -q --tb=short
```

## 尚未实现/验证

- 真实模型与 V100S CAGRA 尚未验证；这是实机验收待办，不再作为通用开发的阻断点。
- 无目标模型 probe、draft 前缀重对齐、query 捕获和模型显存预算。
- 无 CAGRA 服务端索引生命周期、真实请求检索、稀疏打包与 attention。
- 无实际 active/next GPU 双缓冲、Scheduler 预取接入和 RDMA/compute overlap。
- 无生成质量或端到端性能结论。
- 没有启用任何新启动参数；不能通过本次改动启动完整预测流水线。
- 没有改动现有 Mooncake 生命周期、传输协议、full_prompt 或 TP 支持范围。
- 未 commit/push；未修改用户的 `Claude outputs/`。

## 下一步

1. 不等用户选模型：继续设计/实现可自定义模型的 provider 配置、目标模型 probe 接口及 RNG/KV 隔离测试。
2. 默认 inventory 只收集信息；候选 Linux/V100S 环境可用后再执行 smoke，记录经过验证的依赖组合。
3. 明确索引粒度与失败策略，继续影子预测/检索的实现；用 fake 与 CPU 测试推进，真实实验单列待验收。
4. 首轮初始化方案已于 2026-09-19 变更为「最终等待队列触发的拉取」：
   D 等请求进入 `scheduler.waiting_queue` 后发起，V 写入其已预分配的最终 KV 页，
   等待队列内 not-runnable，安装后才接纳进运行 batch。相关交接文档已同步更新。
5. 不将纯逻辑 `PrefetchClock` 直接替换线上时钟；实际接入必须有 next buffer、传输保护和功能开关，不默认启用未验证路径。
