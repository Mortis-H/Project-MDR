# Kernel Profiling 穩定性測試結果

## 測試條件

- 測量次數: **100 次/kernel**
- 統計方法: 去掉最大最小值後計算
- GPU: AMD gfx950
- 測試大小: 64 elements

---

## 三方比較：中位數

| Kernel | MDR @TIMESTAMP | rocprofv2 | rocprofv3 |
|--------|----------------|-----------|-----------|
| **scalar_ops** | 788 | 1,620 | 2,920 |
| **memory_ops** | 1,596 | 1,800 | 2,080 |
| **conditional** | 1,682 | 1,820 | 2,040 |
| **loop** | 760 | 1,680 | 2,000 |

### 相對比例（以 MDR 為基準 = 1.00x）

| Kernel | MDR @TIMESTAMP | rocprofv2 | rocprofv3 |
|--------|----------------|-----------|-----------|
| **scalar_ops** | 1.00x | 2.06x | **3.71x** |
| **memory_ops** | 1.00x | 1.13x | 1.30x |
| **conditional** | 1.00x | 1.08x | 1.21x |
| **loop** | 1.00x | 2.21x | **2.63x** |

---

## 三方比較：穩定性（變異係數）

| Kernel | MDR CV | rocprofv2 CV | rocprofv3 CV | 最穩定 |
|--------|--------|--------------|--------------|--------|
| scalar_ops | 15.2% | 15.9% | 18.3% | MDR |
| memory_ops | 13.4% | **7.3%** | 30.2% | rocprofv2 |
| conditional | 12.3% | **7.9%** | 13.1% | rocprofv2 |
| loop | **15.6%** | 29.0% | 17.0% | MDR |

**觀察**：
- rocprofv2 在 `memory_ops` 和 `conditional` 表現最穩定
- MDR 在 `scalar_ops` 和 `loop` 表現最穩定
- rocprofv3 在 `memory_ops` 有異常高的變異（30.2%）

---

## 詳細數據

### MDR @TIMESTAMP

| Kernel | 中位數 | 平均值 | 標準差 | 變異係數 | 範圍 |
|--------|--------|--------|--------|----------|------|
| scalar_ops | 788 | 786.2 | 119.5 | 15.2% | 476 |
| memory_ops | 1,596 | 1,610.7 | 215.2 | 13.4% | 760 |
| conditional | 1,682 | 1,727.3 | 212.3 | 12.3% | 932 |
| loop | 760 | 746.0 | 116.5 | 15.6% | 456 |

### rocprofv2

| Kernel | 中位數 | 平均值 | 標準差 | 變異係數 | 範圍 |
|--------|--------|--------|--------|----------|------|
| scalar_ops | 1,620 | 1,647.8 | 262.7 | 15.9% | 1,800 |
| memory_ops | 1,800 | 1,818.5 | 131.9 | 7.3% | 520 |
| conditional | 1,820 | 1,823.5 | 143.7 | 7.9% | 641 |
| loop | 1,680 | 1,899.2 | 550.4 | 29.0% | 2,000 |

### rocprofv3

| Kernel | 中位數 | 平均值 | 標準差 | 變異係數 | 範圍 |
|--------|--------|--------|--------|----------|------|
| scalar_ops | 2,920 | 2,754.7 | 505.2 | 18.3% | 1,800 |
| memory_ops | 2,080 | 2,265.8 | 684.8 | 30.2% | 5,281 |
| conditional | 2,040 | 2,002.2 | 263.0 | 13.1% | 1,041 |
| loop | 2,000 | 1,892.7 | 322.4 | 17.0% | 1,120 |

---

## 分析與結論

### 1. 測量時間排序

對於所有 kernel，測量時間都是：

```
MDR @TIMESTAMP < rocprofv2 < rocprofv3
```

這反映了三種工具測量的「範圍」不同：
- **MDR @TIMESTAMP**: 測量 kernel 內部純執行時間
- **rocprofv2**: 測量 kernel dispatch 週期（較短）
- **rocprofv3**: 測量 kernel dispatch 週期（較長，可能包含更多 metadata 處理）

### 2. 短 kernel vs 長 kernel

| Kernel | 類型 | rocprofv2/MDR | rocprofv3/MDR |
|--------|------|---------------|---------------|
| loop | 最短 | 2.21x | 2.63x |
| scalar_ops | 短 | 2.06x | **3.71x** |
| memory_ops | 較長 | 1.13x | 1.30x |
| conditional | 較長 | 1.08x | 1.21x |

**結論**: 對於執行時間越短的 kernel，rocprof 測量的「額外時間」比例越高。

### 3. 穩定性總結

| 工具 | 最穩定的 kernel | 最不穩定的 kernel |
|------|-----------------|-------------------|
| MDR | conditional (12.3%) | loop (15.6%) |
| rocprofv2 | memory_ops (7.3%) | loop (29.0%) |
| rocprofv3 | conditional (13.1%) | memory_ops (30.2%) |

### 4. 建議使用場景

| 場景 | 推薦工具 | 原因 |
|------|---------|------|
| 測量 kernel **內部純執行時間** | **MDR @TIMESTAMP** | 最接近實際指令執行時間 |
| 測量 **memory-bound kernel** | **rocprofv2** | 最穩定 (CV 7.3%) |
| 測量 **短 kernel** | **MDR @TIMESTAMP** | 避免 dispatch overhead |
| 測量 kernel **內部區段** | **MDR @TIMESTAMP** | 唯一選擇 |
| 需要與標準 profiling 比較 | **rocprofv2** | 業界標準 |

---

## 實驗日期

2026-01-29
