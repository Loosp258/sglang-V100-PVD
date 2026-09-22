# V 索引完成与资源隔离 / V index completion and quarantine

## 已实现 / Implemented

索引构建、搜索、销毁现在具有显式完成屏障。释放 extraction/index/search
预算前，必须证明相关操作已完成；CPU reference 保持同步语义，CUDA backend
必须提供完成契约，opaque native index 必须提供明确销毁契约。

Index build, search and disposal now have explicit completion boundaries. The
manager must establish completion before refunding extraction, index or search
reservations. The CPU reference remains synchronous; CUDA backends require a
completion contract and opaque native indexes require an explicit disposal contract.

构建或 extraction 完成未知时，VectorKVStore 保留真正的 Entry allocation pin、
packed source、已产生的副本、异常和预算 owner。搜索未知时保留 reader 与 scratch；
销毁未知时保留 detached record。UNKNOWN 是粘性隔离：后续不自动重试 fence，
也不继续建立索引、搜索或发放 selection lease。完整 Prompt 交付不依赖索引就绪。

An unknown build/extraction retains the actual Entry allocation pin, packed
source, produced copies, exception and budget owners. An unknown search retains
its reader and scratch; unknown disposal retains the detached record. UNKNOWN
is sticky quarantine, not a retry signal: new builds, searches and selection
leases are refused. Full-Prompt delivery does not depend on index readiness.

普通失败在完成与部分索引销毁得到确认后退款；容量压力仍是 backpressure；
KeyboardInterrupt 等控制异常清理后重新抛出，不转为成功或普通构建失败。

Ordinary failures refund only after completion and partial-index disposal are
proved. Capacity pressure remains backpressure. Control exceptions such as
KeyboardInterrupt are cleaned up and re-raised, never converted into success.

## 验证与边界 / Evidence and limits

- 新增 7 个 CPU fault-injection 用例；全量 Windows **2149 passed / 24 skipped**。
- WSL index/source/sparse Delivery 定向 **108 passed / 1 skipped**。
- 测试覆盖源 pin、预算、异常持有的 native-buffer double、部分构建销毁、搜索
  reader 及 UNKNOWN 后禁止自动重试；没有真实 CUDA/native destructor 执行证据。

Seven new CPU fault-injection cases pass. The full Windows suite reports
2149 passed / 24 skipped; focused WSL coverage reports 108 passed / 1 skipped.
These establish ownership policies, not actual CUDA completion or native
destructor behavior. No production cuVS/CAGRA backend is implemented by this
change; that backend must enforce its own allocation limits and native lifetime
contract rather than treating a Python reference drop as proof of safe release.
