# NSA 连续 KV 打包：本地实现与验证

2026-10-04；`codex/pvd-oasiskv`；实现提交`28e223be6`。
用户确认CloudLab租约到期且当前没有GPU，因此本轮只报告CPU与保存轨迹验证。
新选项默认关闭；尚无CUDA、原生交付或线上加速结论。

## 改动

借用[NSA §3.3.2/3.4](https://arxiv.org/html/2502.11089v1#S3)的连续块访问思想。
没有加入论文的学习压缩分支、门控或原生稀疏训练。

- V保留现有连续staging发送缓冲区，把同一head的相邻token从strided source view
  成段复制。不会把交错head的原Entry地址错误地合并成连续RDMA区间。
- D只排序实际缺失token的wire顺序；CAGRA候选评分、选中集合、resident bank顺序、
  Top4、capacity32、max_new16、Q数量和逐层配对Decode保持原样。
- 不复制块内未选中的token，不增加payload字节；保留既有预算、版本租约、
  CUDA fence、native terminal/byte proof、安装后ACK及UNKNOWN保留逻辑。
- V选项`--experimental-contiguous-sparse-packing`，D配置`sort_missing_tokens`。
  V拒绝与Triton packing或原Entry direct batch PUT混用；所有新选项默认为false。
- 增加`v-contiguous`公平ABBA入口，实际V kernel模式与D排序开关都需要证据，
  验证source hashes、预算、warmup、调用计数及端到端指标。入口已准备，没有启动旧节点。

## 本地资格验证

最终结果：**105 passed, 6 skipped, 1 warning in 3.89s**。
Python3.14.5／Torch2.14.0+cpu；包bootstrap只绕过SGLang前端的Linux依赖，实际PVD、
Torch和HTTP代码均来自工作树；生命周期测试中使用明确的CPU policy/transport doubles。
这不是原生GPU/RDMA验证。六项真实CUDA/nondefault-stream案例因无GPU跳过；
warning是现有pytest配置的`asyncio_mode`缺少本地插件。

覆盖多dtype、多head/layer、原顺序和倒序、跨页连续段、最终有效token、独立源/目标、
无额外gather分配、非法后续组写入前拒绝、部分复制失败、真实store预算及ACK路径，
并验证排序wire不改变候选顺序与CPU缓存内容。

前一次扩展gate中的一个测试夹具错误地使用token9，而该小store的Prompt不足10行。
store正确地在授权/写入前拒绝。修复测试为有效0..5后重新通过；失败日志保留在
`artifacts/nsa_blocks_20261004/gate03/`，没有纳入最终通过计数。

## 保存的真实模型 KV 回放

使用2026-10-03保存的两个2159-token、16输出token轨迹。
仅根据捕获到的真实Prompt KV行重建interleaved source，未捕获行置为占位且从不选取。
逐次执行实际manifest、pack、单调CPU缓存安装，再按原selected order重建bank。
没有运行新的目标模型forward，也没有进行检索质量或GPU性能测试。

| 指标 | case99401：原路径→连续打包 | case99402：原路径→连续打包 |
|---|---:|---:|
| 已消费逐层bank检查数／模式 | 420→420 | 420→420 |
| 缺失token检查数／模式 | 5008→5008 | 5001→5001 |
| missing-rank交付次数 | 770→770 | 772→772 |
| 逻辑payload bytes | 2564096→2564096 | 2560512→2560512 |
| pack的copy调用数 | 10016→8652 | 10002→8628 |

两个模式在**840个不同的实际逐层bank**上，K、V、valid mask及token顺序全部逐位一致。
copy调用数由实际wire ranges推导；上述约13.6%–13.7%的全请求调用减少包含bootstrap。
分开计数时，bootstrap约减少35%，后14步仅约减少7%；见
[布局分析](pvd_nsa_block_layout_cloudlab_20261004.md)。这些比例不是耗时收益。

## 来源、复现与后续条件

同名JSON记录最终CPU gate、两个输入轨迹SHA-256、真实bank结果及源码身份。
149个被记录源文件与实现提交的Git blob一致，使用LF归一化比较。
原始布局明细、失败/最终gate输出和源哈希归档在项目内
`artifacts/nsa_blocks_20261004/evidence.tar.gz`；临时文件和日志均在artifacts内。

```powershell
& C:/Python314/python.exe -X utf8 -B benchmark/run_pvd_cpu_gate.py --output artifacts/nsa_cpu_fresh test/registered/disaggregation/test_pvd_contiguous_sparse.py test/registered/disaggregation/test_pvd_sparse_copy.py test/registered/disaggregation/test_pvd_sparse_store_delivery.py test/registered/disaggregation/test_pvd_oasis_transport_io.py test/registered/disaggregation/test_pvd_oasis_serving.py
& C:/Python314/python.exe -X utf8 -B benchmark/verify_pvd_nsa_real_banks.py --capture artifacts/oasis_workspace_replay01/capture --output artifacts/nsa_real_banks_fresh/result.json
```

CPU gate需要pytest及已有项目测试依赖；helper可复用现有项目内依赖目录。
待新GPU节点、连接及验证checkout配置完成，先验证真实CUDA复制和native交付，
再执行同模型、快图、预算、warmup及资源的base/opt/opt/base完整P/V/D对照。
P→D仍关闭，初始完整P→V→D与private Prompt pass仍计入客户端时间。

**目前完成的是本地实现和精确字节/bank验证。** 稳态连续性有限，这项改动保留为
默认关闭的实验，不能据此声称D等待或TPOT下降。整块扩展在约79%的实际bank上
会超过32-token预算，因此没有实施。后续块选择需要在相同预算下单独验证质量。
