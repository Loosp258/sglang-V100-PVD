# OasisKV 分支首次完整上传

2026-10-05，用户授权将 `codex/pvd-oasiskv` 的既有本地工作上传 GitHub，
随后按顺序优化、每完成一步提交并推送。此前的仅本地提交约束已由本次
指令覆盖，历史实验记录保留原文。

## 大文件迁移与历史对应

上传前发现 283 个远端尚无的提交和两份超过 GitHub 普通 Git 100 MiB
限制的原始归档。只将这两份归档迁移到 Git LFS；源码和实验数据不删除。
[GitHub 文件限制](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)

| 归档 | 原始 bytes | SHA256 |
|---|---:|---|
| `benchmark/results/pvd_oasis_attention_replay_cloudlab_20261003/raw.tar.gz` | 154178083 | `a8a038e1e157c3f7fa77f6ed717333bbf732f9308a1ecfe86684a007216aa69c` |
| `benchmark/results/pvd_oasis_ready_kv_cloudlab_20261003/raw.tar.gz` | 152804595 | `8ea8d3f12c91eb79b22719ec3d90de7b076ccfc5a24de1adda3078b87960f7d0` |

- 原 HEAD：`58078a717b19a60e2d9c0ba88fe7e01a181d6be7`。
- 迁移后对应 HEAD：`7e062be7ecbb2f7e4b6fdd7e088496f7584046b6`。
- 原提交历史保存在本地 `codex/pvd-oasiskv-before-lfs-20261005`。
- `pvd_github_lfs_commit_map_20261005.csv` 保存 283 条旧/新完整提交号对应。
  旧报告、测试 source proof 中的旧提交号保持原记录，可通过此表查询上传版。
- Git LFS 工具同时更新了若干共享祖先的本地 refs；这些 refs 已按迁移前
  记录恢复，并逐一核对其他本地分支 tip 不变。只上传目标 OasisKV 分支。
- HEAD tree 比较只差 `.gitattributes` 与两个 LFS pointer，其余所有 tree
  entries 的模式和 Git blob ID 完全相同。工作树中的两份完整归档与迁移前
  size/SHA256 相同，`git lfs fsck` 通过。
- 原始检查在 `artifacts/github_publish_20261005/{before,migration_proof}.json`。
  LFS 本地缓存使用本 worktree 的 `artifacts/github_publish_20261005/lfs`，
  由 worktree 专用 Git 配置指定。

该远端分支此前不存在；推送采用普通新分支创建，不强制覆盖远端历史。
本次迁移没有修改服务算法、模式、默认值或测试结果。

## 后续取回归档

克隆分支后安装 Git LFS 并运行 `git lfs pull`。普通源码、测试、报告及小型
证据仍使用常规 Git；上述两份大归档必须取回 LFS 内容后再按 SHA256 核对。
所有新的本地产物继续放在项目/worktree 内，临时输出使用 `artifacts/`。
