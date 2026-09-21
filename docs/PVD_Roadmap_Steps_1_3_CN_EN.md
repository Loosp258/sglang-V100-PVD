# 第 1–3 步离线验收 / Offline gates 1–3

2026-09-21；基线提交 `6dba510dba4129c8c94266085366968f7811f37a`。
本轮修改未提交、未推送；未触碰 `Claude outputs/`。

后续更新：[真实模型稀疏 CPU Decode](PVD_Sparse_CPU_Decode_CN_EN.md) 已通过，
覆盖下文“尚缺真实后端消费”的历史描述；多 rank / GPU / 在线 Scheduler 仍待完成。

## 实现 / Implementation

1. `target_probe.py` 修复编码标签：使用 V 现有 `ROPE_APPLIED` 常量
   (`rope_applied`)，不放宽 V 身份校验。回归先复现原 `post_rope` 标签不兼容。
2. `pvd_search_roundtrip.py` 经真实 V HTTP route/client 验证两分片、两层、GQA
   查询，独立 Python 点积/top-k oracle 核对 token/page/score；拒绝未就绪索引、
   错模型/编码/分片、旧 index/mapping 与同 Entry 重建后的旧版本。
3. `sparse_payload.py` 验证请求/Entry/index/mapping/layout 身份及全局层/head/token，
   配对复制 K/V、排除 page padding、先预留预算，退出作用域释放。
   调用方仍须持有源 Entry；版本字符串不是 lease，也不是 RDMA 写授权。
4. `sparse_union.py` 实现用户选择：同一层同一 KV head 的所有 Q heads 取 token
   并集去重。要求完整 Q-head 集合与一致版本/上下文；不合并分数，超显式上限拒绝。
5. `sparse_working_set.py` CPU current/next 双工作集：首轮完整 Prompt、后台候选
   独立持有副本与预算、边界精确匹配才安装、读者未退出拒绝替换/释放、失败退款。
   生成 KV 由调用者单独持有；attention reference 将它与选定 Prompt KV 合并，
   使用原始绝对位置 causal mask，逐 Q head 映射到 KV head。

The GQA union is per (layer, KV head), not a cross-head/layer score ranking.
An explicit cap is mandatory; overflow refuses the refresh. Initial Prompt stays
complete. The CPU bank is single-owner/synchronous: read scopes are not CUDA
events, tensor aliases must not outlive their scope, and it is not a serving pool.

## 验证 / Evidence

- Windows 全套：**1166 passed / 11 skipped**。
- WSL 全套：**1171 passed / 6 skipped**；硬件检查仍跳过，不能计作验证通过。
- 新第三步定向测试：24 passed，包括并集、拒绝、预算/生命周期、独立 softmax
  数值对照、完整/子集 attention、绝对位置掩码、生成 KV 不变。
- Strict CPU smoke `--probe --search`：34 次真实 CPU forward；draft forward
  对照最大误差 `8.344650268554688e-7`；Q 与实际 post-RoPE 捕获误差 0。
- 真实 K/Q 经两个 V shard 的 HTTP 搜索：8 组 layer/Q-head 结果、14 个拒绝检查，
  最大 score 误差 `1.4965603156724683e-7`；8 份配对 K/V 载荷逐值一致；
  4 组 GQA 并集安装入 CPU 分片 bank。
- Ruff 检查/格式化覆盖本轮新增模块与测试。

The strict fixture is a tiny randomly initialized Llama, CPU FP32,
`TorchNativeAttnBackend`, TP1/PP1. It is not a chosen production checkpoint.
The search pipeline uses fixed draft candidate tokens; its K and Q are real
target-model outputs, but this is not a loaded-draft-to-online-Decode run.
P→V is a direct local byte copy using fake registration, NOT RDMA. Each V shard's
CPU bank is checked separately; this is NOT multi-rank atomic installation.

## 复现 / Reproduce

WSL, repository root (existing environment):

```bash
PYTHONPATH=python /home/loosp/torch311-env/bin/python \
  test/registered/disaggregation/run_pvd_cpu_tests.py \
  test/registered/disaggregation/test_pvd*.py -q --tb=short -p no:cacheprovider

PYTHONPATH=python /home/loosp/torch311-env/bin/python \
  test/registered/disaggregation/run_pvd_draft_cpu_smoke.py --probe --search
```

Dependency/fixture failure in the strict smoke fails rather than silently skipping.
No model download or GPU is needed for this fixture.

## 下一步与未完成 / Next and remaining gaps

第 3 步尚缺真正的 D attention backend 消费及多 rank 安装协议，不应跳过并宣称完成。
下一项：明确后端如何按 layer/KV-head 消费不同 token 集，如何合并本地生成 KV、保留
绝对位置，并使同请求所有 rank 在同一边界完成安全切换；先用可控 CPU 集成测试验证。
之后才接请求级在线调度，处理迟到/取消/EOS、每请求时钟、失败等待和新请求隔离。

Still absent: actual serving sparse attention, source/index leases spanning async
packing, authorized sparse transport and GPU lifetime/fences, online target-probe
execution arbitration, rank-coordinated install, CAGRA, V100S/RDMA validation,
quality/recall and performance evidence. Overflow refusal does not silently choose
a serving fallback; any online retry/failure policy must be specified when wired.
Initial full Prompt must still fit D; this does not solve unbounded initial context.
