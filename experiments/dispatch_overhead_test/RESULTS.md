# Dispatch Overhead 實驗結果

## 實驗目的

驗證假設：對於執行時間很短的 kernel，rocprofv2 的測量可能包含較大比例的 dispatch overhead，而 MDR @TIMESTAMP 可以測量更精確的「純執行時間」。

## 測試環境

- GPU: AMD MI300 (gfx950)
- 測試大小: 64 elements
- 每個 kernel 只運行一次

## 測量結果

| Kernel | rocprofv2 (ticks) | MDR @TIMESTAMP (ticks) | 差異 | 比例 |
|--------|-------------------|------------------------|------|------|
| **scalar_ops** | 1,680 | 856 | 824 | 1.96x |
| **memory_ops** | 1,960 | 1,584 | 376 | 1.24x |
| **conditional** | 1,680 | 1,384 | 296 | 1.21x |
| **loop** | 1,920 | 608 | **1,312** | **3.16x** |

## 分析

### 1. 所有 kernel 的 MDR @TIMESTAMP 都比 rocprofv2 短

這是因為測量範圍不同：
- **rocprofv2**: 測量從 kernel dispatch 到 completion signal 的完整時間
- **MDR @TIMESTAMP**: 測量從第一條指令到 `s_endpgm` 前的時間

### 2. loop kernel 差異最大（3.16x）

`loop` kernel 是最簡單的 kernel（只寫入一個常數），執行時間最短（608 ticks）。
在這種情況下，dispatch overhead 的比例最大。

### 3. 「額外時間」分析

假設「額外時間 = rocprofv2 - MDR @TIMESTAMP」代表 dispatch 相關的 overhead：

| Kernel | 額外時間 (ticks) | 占 rocprofv2 比例 |
|--------|------------------|-------------------|
| scalar_ops | 824 | 49% |
| memory_ops | 376 | 19% |
| conditional | 296 | 18% |
| **loop** | **1,312** | **68%** |

## 重要發現

### ⚠️ 假設部分驗證

1. **確實存在差異**：MDR @TIMESTAMP 測量的時間比 rocprofv2 短
2. **但差異不是「dispatch overhead」**：額外時間在不同 kernel 間變化很大

### 可能的原因

差異可能來自：
1. **Kernel 啟動時間**：wavefront 調度到第一條指令執行的時間
2. **Kernel 結束時間**：最後一條指令到 completion signal 的時間
3. **測量點位置**：@TIMESTAMP_START 放在 kernel 內部，不是真正的「開始」

### 結論

| 用途 | 推薦工具 |
|------|---------|
| 比較不同 kernel 的整體效能 | rocprofv2（包含完整 dispatch 週期）|
| 測量 kernel **內部**的純執行時間 | MDR @TIMESTAMP |
| 分析短 kernel 的效能瓶頸 | 兩者都需要，比較差異 |

## 下一步建議

1. 把 @TIMESTAMP_START 放在 kernel 的真正開頭（第一條指令之前）
2. 多次運行取平均值，減少測量誤差
3. 測試更多不同類型的 kernel
