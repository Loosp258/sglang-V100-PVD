# NSA 连续块思想：现有真实 Decode 轨迹的布局机会

分析日期2026-10-04；输入为2026-10-03保存的两个实际Qwen2.5-7B/EAGLE3
Decode轨迹，每个2159-token Prompt、16输出token、15次实际D forward。
本次分析在本地CPU完成，没有运行新的CloudLab GPU实验。

## 固定条件与计数范围

保留V/CAGRA、四合一快图、Top4、capacity32、max_new16、两worker及实际
selected bank IDs。根据D单调CPU缓存重建缺失token；bootstrap与后14步分开。
只计已消费bank，最后一次未消费预取不在保存的bank轨迹中。K/V各复制一次，
每token逻辑payload为512字节。复制调用数是布局推导，不是实测CUDA launch时间。

| 指标 | case99401 | case99402 |
|---|---:|---:|
| bootstrap缺失token | 1193 | 1205 |
| bootstrap逐行复制调用 | 2386 | 2410 |
| bootstrap排序后连续段复制调用 | 1556 | 1574 |
| bootstrap调用减少 | 34.79% | 34.69% |
| 后14步缺失token | 3815 | 3796 |
| 后14步逐行复制调用 | 7630 | 7592 |
| 后14步排序后连续段复制调用 | 7096 | 7054 |
| 后14步调用减少 | 7.00% | 7.09% |
| 后14步missing-rank交付次数 | 714 | 716 |
| 后14步逻辑payload bytes | 1953280 | 1943552 |

排序只改变缺失token的线序；候选、驻留bank顺序、注意力语义、payload字节及
交付次数不变。原Entry每token交错两个KV head，单head相邻token并非连续字节；
需要从strided view打包进入现有连续发送缓冲区，不能直接合并原Entry RDMA地址。

## 整块扩展的容量代价

下表把实际已选token扩展成包含它们的全部块，截断最后不完整Prompt块。
每case覆盖1680个已消费layer/head bank。没有实现该选择策略，也没有测其召回。

| 块大小 | case99401超过32-token预算 | case99402超过预算 | 总驻留token相对原选择 |
|---|---:|---:|---:|
| 4 | 1321/1680（78.63%） | 1334/1680（79.40%） | 2.404/2.410倍 |
| 8 | 1492/1680（88.81%） | 1494/1680（88.93%） | 3.601/3.610倍 |
| 16 | 1621/1680（96.49%） | 1633/1680（97.20%） | 5.680/5.685倍 |

整块扩展会改变预算，不能用它直接替换当前baseline并宣称公平收益。
本轮先验证精确token连续打包；稳态布局收益上限较小，不预期仅此实现数量级改善。

## 来源与复现

原始轨迹及回放范围见[pvd_oasis_attention_replay_cloudlab_20261003.md](pvd_oasis_attention_replay_cloudlab_20261003.md)。
紧凑统计及输入SHA-256保存于同名JSON。逐缺失组明细在项目内
`artifacts/nsa_blocks_20261004/layout.json`，没有写入项目外目录。

```powershell
& C:/Python314/python.exe -X utf8 -B benchmark/analyze_pvd_nsa_blocks.py --capture artifacts/oasis_workspace_replay01/capture --output artifacts/nsa_blocks_fresh/layout.json
```

用户随后确认CloudLab租约到期且当前无GPU；没有新CUDA/native资格验证、
线上TPOT或D等待结果。后续本地实现和测试记录于
[连续KV打包报告](pvd_nsa_contiguous_kv_local_20261004.md)。
