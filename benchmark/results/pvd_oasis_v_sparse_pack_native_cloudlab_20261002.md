# V 融合稀疏 KV 打包：原生字节与时间检查

CloudLab V100S 两 GPU，2026-10-02；计划 `81db9b537`、实现/异常修复 `b61c671a4`。

## 正确性

**143 passed**；已有 standalone smoke 每 GPU 12 cases，共24个，覆盖 FP16/BF16/FP32、多块、跨当前设备。
当前形状：28层、2 heads/rank、dim128、FP16、2159有效tokens；page1源61902848B/rank，
额外page2源61931520B/rank含一个padding token。每种页面尺寸都在两个GPU上运行，并将另一GPU设为current。
每 fixture 比较首/末layer/head的乱序1/3/28/32选择，共32768目标字节；所有位相同。
原始源为随机bytes，包含FP16特殊编码；这里只验证字节布局，不涉及实际模型Q/K或召回。
非法2159 ID在首/末组分别被两种copy拒绝，sentinel未变；page1为越界，page2为padding。
所有metadata/staging预算回收、caller设备恢复，Torch live allocated结束为0；process退出后两卡0MiB。

## 元数据、copy和双fence的微基准

两臂均包含验证、copy和两次同设备fence；Triton包含每次新metadata的分配/上传/释放。
源/目标已预分配、event创建、注册和RDMA不在wall范围。ABBA，每block两次warmup、10次正式copy。
GPU event span含host pacing，不能作为纯kernel时间。下表为真实page1的wall中位数。

| rank | 每组选择行数 | Torch | Triton | metadata |
|---|---|---:|---:|---:|
| 0 | [1, 3] | 0.770ms | 1.185ms | 112B |
| 0 | [4, 4] | 0.897ms | 1.201ms | 144B |
| 0 | [28, 32] | 2.687ms | 1.239ms | 560B |
| 1 | [1, 3] | 0.767ms | 1.197ms | 112B |
| 1 | [4, 4] | 0.885ms | 1.205ms | 144B |
| 1 | [28, 32] | 2.666ms | 1.232ms | 560B |

60行容量边界case变快；4行和8行case变慢。metadata的固定准备成本影响小payload，
不能只选大case预测Decode收益。bootstrap GQA Top4×7 union理论<=28/head，steady max_new16；
28×32微基准是边界检查，不是本次典型稳态流量。完整在线结果另做ABBA，默认关闭。

## UNKNOWN路径修复

检查发现已判定metadata upload UNKNOWN后，finally的后来成功全设备sync仍释放index lease。
现在此路径保留index reader，模块隔离metadata workspace，Entry/staging/预算也保持占用，拒绝register/PUT。
强化CPU故障替身测试确认close后仍保留、授权不能fence；普通success/copy异常原逻辑保留。
此故障gate使用CPU policy doubles，不宣称真实GPU/RDMA故障注入。

## 证据与复现

同名目录raw.tar.gz保留部署bundle、逐fileLF sourcehash、unit/smoke/probe完整命令/输出/status、initial/finalGPU。
summary包含完整ABBA观测、Torch峰值/allocator缓存与预算；Torch统计不包含Mooncake/cuVS/完整服务内存。

```text
wsl -d Ubuntu -- python3 /mnt/d/code/sglang-V100-PVD-oasiskv/benchmark/deploy_pvd_oasis_pack.py --tag fresh_pack_native
```
