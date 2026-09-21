# 异构 TP 完整 KV 重打包修复 / Heterogeneous full-KV repack ownership

Three defects were reproduced before modification in the existing full-Prompt
V-to-D layout-conversion path, independently of the new sparse mode:

1. A packing allocation failure retained a 32-byte reservation with no owner
   capable of retiring it.
2. A registration exception was reported as NOT_SUBMITTED without retaining a
   staging guard, although native registration might already have occurred.
3. Two Entries using the same Delivery ID shared one budget owner; completion
   of one could refund the other's still-live allocation.

修改前已用 3 个失败测试复现：打包分配失败泄漏预算、注册异常误判为未提交且未持有
staging、不同 Entry 的相同 Delivery ID 共用预算 owner。属于原完整 Prompt 的
异构 TP 交付路径，不能因为稀疏路径已经修好而忽略。

The fix preallocates only the final byte buffer and copies directly from component
views, eliminating separate contiguous chunks and concatenation. Budget owners
use each Delivery record's unique owner. The backing tensor is attached to a
ResourceGuard before any copy or registration. A failed CPU allocation refunds;
failed copies retire only after the existing packing synchronization. An unknown
registration or CUDA synchronization failure quarantines rather than freeing.
Unregister must succeed before budget refund. Source Entry ownership remains
independent and protected by the existing write authorization.

现在先分配最终连续缓冲，直接按 component view 拷贝，避免 chunks + cat 的双份峰值。
每个 Delivery 使用独立 owner，copy/register 前就挂上 guard；不明注册结果隔离，
不能因业务失败退还仍被原生层使用的内存。既有 CUDA 同步失败时也保留资源。

Five new CPU tests cover the reproduced failures, copy failure after ownership
attachment, and successful exact-byte delivery with a budget fitting only the
final buffer and with concatenation/materialization forbidden. Existing delayed
transport tests still cover late completion and unregister retry. These tests
do not establish CUDA stream or real Mooncake correctness; hardware verification
of the unchanged synchronization boundary remains outstanding.

Final regression: Windows **1451 passed / 11 skipped**; WSL **1456 passed /
6 skipped**. The five new tests passed; the first three failed against the
pre-fix implementation with the exact symptoms described above.
