# V 融合稀疏 KV 打包：完整快图＋Oasis Decode 对照

CloudLab 2026-10-02，`codex/pvd-oasiskv`。计划 `81db9b537`，实现/异常门 `b61c671a4`，原生证据 `9f140226f`。
正式 tag `oasis_v_sparse_pack_abba01`。

## 结果

客户端中位数 **10.011→10.668 s（+6.56%）**；每步累计KV等待 **415.997→463.596 ms**。
本轮没有收益，继续关闭该选项。下面给出完整四臂，不把微基准大payload结果直接推成线上收益。

## 公平性

base_a→opt_a→opt_b→base_b，各臂重启全角色，排除相同两次warmup；两个2159-token Prompt，16输出tokens，greedy/ignore_eos。
P=node0 GPU0，V=node1 GPU0+1，D=node2 GPU1；Qwen2.5-7B＋同一专用EAGLE3，TP1，Oasis配对逐层前向。
初始KV仍图门后P→V→D，private Prompt seed计入客户端，没有P→D直送。
四合一快图、degree16=KNN14+ring2、prefix2048+tail111、itopk2048、Top4、capacity32、max_new16、workers2相同。
两臂V固定RMM pool/host candidates/GPU finite-Q proof；D所有配置完全相同，显式reuse_io=false。
唯一变量为V `PVD_TRITON_SPARSE_PACKING=0/1`；没有改变Q、选择预算或图。两臂均含相同metadata-UNKNOWN lease修复。
15个V文件和D文件逐file本地/远端LF hash相符；各臂两个实际rank健康响应确认kernel=torch/triton及cuda:0/1。
八次实际输出IDs、文本、Prompt完全一致，cached_tokens=0。每请求420jobs/840搜索RPC；每模式3136稳态profile，均2items/14Qrows，无回退/重试。
D全部保持840search sessions；所有native job/loop与安全关闭计数通过，owned={}、cleanup_errors=[]、三节点6GPU最终0MiB。

## 完整阶段时间

| 指标（独立中位数） | Torch打包 | Triton打包 | 变化 |
|---|---:|---:|---:|
| V search batch 每rank/层 | 7.308 ms | 7.932 ms | +8.54% |
| D search_many | 11.858 ms | 12.869 ms | +8.52% |
| D整层检索/交付RPC | 34.059 ms | 37.797 ms | +10.97% |
| 层worker完整service | 37.806 ms | 41.626 ms | +10.11% |
| 每步累计KV等待 | 415.997 ms | 463.596 ms | +11.44% |
| 后续Decode执行 | 515.785 ms | 565.743 ms | +9.69% |
| 首个客户端事件 | 2.532 s | 2.570 s | +1.51% |
| 客户端完成 | 10.011 s | 10.668 s | +6.56% |

打包位于V search-batch timer之后。因此V查询时间变化不能归因于打包kernel变快；
同一GPU上的打包与搜索可相互影响，当前数据没有逐操作分解这一干扰成本。
D RPC范围同时包含搜索、控制往返、打包、注册、PUT、轮询和缓存安装，不是纯网络或纯CAGRA时间。
各列中位数不能相加重建请求。客户端完成不保证覆盖最终共享资源清理，退休另由trace和退出门证明。

| 执行顺序 | case99401 | case99402 |
|---|---:|---:|
| base_a | 10.063s | 10.140s |
| opt_a | 10.519s | 10.817s |
| opt_b | 10.504s | 10.841s |
| base_b | 9.959s | 9.708s |

初始逻辑全KV均123805696B/request；稀疏逻辑payload中位数2563072/2563072B。
原生CAGRA候选可抖动；流量未冻结，输出相同并不保证任意Prompt检索质量。

## 实际小payload与原生测试

143项回归＋24个已有CUDA smoke＋两rank/page1实际形状与page2 padding字节gate全部通过。
源码保留component-major布局，Triton按uint8 gather，无浮点计算；source/destination/index及metadata预算保留至fence。
原生微基准含新metadata分配/上传/释放和两次device fence，8行payload约0.89→1.20ms，60行约2.67→1.23ms。
大选择节省逐行copy提交，小选择受到metadata固定准备成本；不能只选大选择汇报。
完整原生报告：`pvd_oasis_v_sparse_pack_native_cloudlab_20261002.md`。

| 稳态job aggregate remote_rows（双rank合计） | Torch | Triton |
|---|---:|---:|
| mean | 9.712 | 9.711 |
| median | 8 | 8 |
| max | 35 | 35 |
| zero / total jobs | 22 / 1568 | 22 / 1568 |

trace只记录job双rank总rows；不把它当作精确per-rank分布。
微基准为synthetic bytes，不测召回；单元UNKNOWN gate为CPU policy fault double，不宣称真实CUDA/RDMA故障注入。
本轮同时修复metadata上传UNKNOWN后因后来同步成功而释放index lease的路径；
UNKNOWN现在持续保留index/Entry/staging/metadata与预算，禁止register/PUT。该修复在两个对照臂相同。
长Decode、多请求并发、TP2、真实故障和完整服务显存峰值仍需另测。

## 下一项优先级

当前每个missing rank在search之后分别reserve、start、poll、ACK，并分配/注册接收区。
先测合并reserve+start，再研究请求级预注册接收slots；前者每次missing-rank少一次必需串行RPC。
必须保留写入身份、幂等性、lost-reply fence、unknown保留和cache安装后的ACK。
当前没有poll次数与这些子阶段的时间，不据调用次数承诺毫秒收益。

## 证据与复现

同名目录保留四臂完整raw.tar.gz、native.tar.gz、summary/IO计数、V/RPC分解、实际health响应、sourcehash与LF便携manifest。
IO复用的2.85%结果来自另一轮单变量试验，不能和本轮相加。已验证V pooledhost baseline继续保留。

```text
wsl -d Ubuntu -- python3 /mnt/d/code/sglang-V100-PVD-oasiskv/benchmark/run_pvd_oasis_serving_cloudlab.py --tag fresh_pack --comparison v-pack --arms base_a,opt_a,opt_b,base_b --cases 99401,99402 --tokens 16
wsl -d Ubuntu -- python3 /mnt/d/code/sglang-V100-PVD-oasiskv/benchmark/analyze_pvd_oasis_serving.py /mnt/d/code/sglang-V100-PVD-oasiskv/artifacts/fresh_pack --comparison v-pack
wsl -d Ubuntu -- python3 /mnt/d/code/sglang-V100-PVD-oasiskv/benchmark/analyze_pvd_oasis_v_search.py /mnt/d/code/sglang-V100-PVD-oasiskv/artifacts/fresh_pack
wsl -d Ubuntu -- python3 /mnt/d/code/sglang-V100-PVD-oasiskv/benchmark/analyze_pvd_oasis_rpc.py /mnt/d/code/sglang-V100-PVD-oasiskv/artifacts/fresh_pack
```
