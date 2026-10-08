# 缓存后处理与 Q 冻结：继续优化

2026-10-08，用户授权查找优化点、按顺序实施、每步本地 commit，不上传 GitHub。
分支 `codex/pvd-oasiskv`；工作树 `sglang-V100-PVD-oasiskv`。
无 GPU，继续真实 CPU/HTTP/字节/生命周期验证，不将它们称为 native/TPOT 证据。
所有临时输出在本工作树 `artifacts/cache_compute_20261008/`。

CPU 固定候选发现：2159/8192/32768 Prompt、每隔一个 token 已缓存，
两 head 的旧 choose_wire 约227/581/1994 us；V 同版本 delta prepare
约48/105/254 us。这些是后处理函数时间，不是完整 V 搜索或 D 等待。

1. **精确缓存直接 membership。** 默认关闭 `PVD_DIRECT_CACHE_MEMBERSHIP=1`
   （D/V 同时设置）。保留原快图/CAGRA、Q、Top4、32-token bank、max_new16。
   严格校验 bitmap/sorted sparse/list 编码，然后只检查 chosen IDs；避免展开
   全部 bitmap ID 和创建整份 Python set。旧摘要、选择顺序与 missing manifest
   必须精确相同。覆盖 padding/非规范 base64/越界/重复/坏类型与双 rank HTTP。
2. **V 已确认缓存版本复用。** 默认关闭 `PVD_CACHE_DELTA_REUSE=1`（V）。
   仅原 channel、同 layer/head、相同版本与 digest、空增量才能复用已验证
   canonical 快照；不复用未知或更旧版本。返回独立 metadata，保留满量 resync、
   budget/ownership、transactional commit。证明不扫描 bitmap、不跨请求混用。
3. **Q 直接冻结。** 默认关闭 `PVD_DIRECT_BINARY_FREEZE=1`（D）。
   合法 f32 host 数组直接生成 owned bytes，有限值检查针对那些真实发送字节。
   不保留 mutable producer alias；非连续/非 f32 输入仍正确转换。证明原 wire、
   摘要、signed zero 和 lifetime 完全一致，不改变模型/Q 数量/检索参数。

每步先语义/失败验证，再 CPU ABBA：相同 shapes/candidates/cache、暖身、逆序、
原始样本和源 hash。微基准包括 validation/materialization，不只计小操作。
没收益的结果也保留，不推断 D/TPOT 降幅，不相加历史实验收益。每步 commit 后
再下一步；最后组合回归与 commit blob 对照，默认关闭，GPU gates 待恢复。
