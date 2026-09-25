# PVD 配对评估 / Paired evaluation

`test/registered/disaggregation/run_pvd_paired_eval.py` 是不依赖 GPU 的
顺序请求采集与比较工具；实际推理由已启动的三机服务完成。它不会启动、
切换或验证 D 的模式，也不会从 token 一致率推断语义质量。

`run_pvd_paired_eval.py` is a CPU-only sequential collection/comparison
tool. The running three-node service performs inference. The script neither
starts nor verifies D's mode, and token agreement is not semantic quality.

准备 UTF-8 JSONL，每行仅需唯一 `id` 和 `text`。使用固定数据集、模型版本、
采样参数、GPU 资源与 P/V/Gateway；**只切换 D 模式**。分别采集：

Prepare UTF-8 JSONL with a unique `id` and `text` on each line. Keep the
dataset, model revisions, sampling settings, GPU resources and P/V/Gateway
fixed; **switch only D's mode**. Collect both reports:

仓库附带的 `test/registered/disaggregation/pvd_eval_smoke.jsonl` 只有 3 条
短/中长度 Prompt，仅用于验证采集链路，不是正式质量集。
The checked-in `test/registered/disaggregation/pvd_eval_smoke.jsonl` has only
three short/mid-length Prompts for validating the collection path; it is not
a representative quality dataset.

```bash
python test/registered/disaggregation/run_pvd_paired_eval.py collect \
  --gateway-url http://10.0.1.2:8000 --dataset eval.jsonl \
  --mode full --config-id qwen25-7b-full-v1 \
  --max-new-tokens 20 --output full.json

python test/registered/disaggregation/run_pvd_paired_eval.py collect \
  --gateway-url http://10.0.1.2:8000 --dataset eval.jsonl \
  --mode predictive --config-id qwen25-7b-predictive-v1 \
  --max-new-tokens 20 --output predictive.json

python test/registered/disaggregation/run_pvd_paired_eval.py compare \
  --full full.json --predictive predictive.json
```

报告只保存输入数据集 SHA-256、请求 ID、输出 token ID、Prompt/输出 token
数和客户端耗时，不重复保存 Prompt 或模型回复文本。采集失败时非零退出；
输出文件若已存在则拒绝覆盖。比较前要求数据集哈希、生成长度、请求 ID
和顺序一致。比较项为输出完全一致比例、平均共同前缀长度、客户端延迟
中位数及 nearest-rank p95；样本少时 p95 不稳定。模式标签完全由操作者
声明，`mode_verified_by_script=false`，必须另行保存 D 启动命令/日志。

Reports contain the dataset SHA-256, request IDs, output token IDs, token
counts and client latency, not copies of prompts or response text. Collection
fails nonzero on an incomplete reply and never overwrites an existing output
file. Comparison requires identical dataset hash, output length, request IDs
and order. It reports exact token agreement, mean common-prefix length,
median latency and nearest-rank p95; p95 is unstable for small samples. The
operator supplies each mode label (`mode_verified_by_script=false`), so retain
the D startup command and logs as independent evidence.

正式验收仍需语义/任务质量评估、真实 Q 的 CAGRA Top-K 召回率、长上下文与并发负载、
TPOT/吞吐/刷新边界等待、GPU 显存峰值和 RDMA/生命周期检查。顺序客户端耗时
不等于服务器吞吐，也不能证明网络等待已经隐藏。

Final acceptance additionally requires task/semantic quality, actual-Q CAGRA
Top-K recall, longer contexts and concurrent load, TPOT/throughput/boundary
wait, peak GPU memory and RDMA/lifecycle evidence. Sequential client latency
is not server throughput and does not prove that network waits are hidden.
