# Kernel Profiling 穩定性測試結果

## 測試條件

- 測量次數: **100 次/kernel**
- 統計方法: 去掉最大最小值後計算
- GPU: AMD gfx950
- 測試大小: 64 elements

---

## 三方比較：中位數 (ticks)

| Kernel | MDR @TIMESTAMP | rocprofv2 | rocprofv3 |
|--------|----------------|-----------|-----------|
| **vectoradd** | 1,688 | 2,000 | 1,760 |
| **scalar_ops** | 788 | 1,620 | 2,920 |
| **memory_ops** | 1,596 | 1,800 | 2,080 |
| **conditional** | 1,682 | 1,820 | 2,040 |
| **loop** | 760 | 1,680 | 2,000 |

### 相對比例（以 MDR 為基準 = 1.00x）

| Kernel | MDR @TIMESTAMP | rocprofv2 | rocprofv3 |
|--------|----------------|-----------|-----------|
| **vectoradd** | 1.00x | 1.18x | 1.04x |
| **scalar_ops** | 1.00x | 2.06x | **3.71x** |
| **memory_ops** | 1.00x | 1.13x | 1.30x |
| **conditional** | 1.00x | 1.08x | 1.21x |
| **loop** | 1.00x | 2.21x | **2.63x** |

---

## 三方比較：穩定性（變異係數）

| Kernel | MDR CV | rocprofv2 CV | rocprofv3 CV | 最穩定 |
|--------|--------|--------------|--------------|--------|
| **vectoradd** | **9.1%** | 10.6% | **7.6%** | rocprofv3 |
| scalar_ops | 15.2% | 15.9% | 18.3% | MDR |
| memory_ops | 13.4% | **7.3%** | 30.2% | rocprofv2 |
| conditional | 12.3% | **7.9%** | 13.1% | rocprofv2 |
| loop | **15.6%** | 29.0% | 17.0% | MDR |

---

## 關鍵發現：vectoradd 與其他 kernel 的差異

### vectoradd 是特殊案例

| 比較項目 | vectoradd | 其他 kernel (平均) |
|----------|-----------|-------------------|
| rocprofv2/MDR | **1.18x** | 1.62x |
| rocprofv3/MDR | **1.04x** | 2.21x |

**vectoradd 的三種工具測量結果非常接近**，這與其他 kernel（特別是 `scalar_ops` 和 `loop`）形成鮮明對比。

### 可能原因

1. **vectoradd 是 memory-bound kernel**
   - 主要操作：`global_load` + `global_store`
   - 實際執行時間較長，dispatch overhead 占比較小

2. **scalar_ops 和 loop 是 compute-bound kernel**
   - 計算密集但 memory 操作少
   - 純計算時間極短（~760 ticks）
   - dispatch overhead 比例顯著（rocprofv3 是 MDR 的 2.6-3.7 倍）

---

## 詳細數據

### MDR @TIMESTAMP

| Kernel | 中位數 | 平均值 | 標準差 | 變異係數 | 範圍 |
|--------|--------|--------|--------|----------|------|
| **vectoradd** | 1,688 | 1,693.5 | 153.6 | 9.1% | 704 |
| scalar_ops | 788 | 786.2 | 119.5 | 15.2% | 476 |
| memory_ops | 1,596 | 1,610.7 | 215.2 | 13.4% | 760 |
| conditional | 1,682 | 1,727.3 | 212.3 | 12.3% | 932 |
| loop | 760 | 746.0 | 116.5 | 15.6% | 456 |

### rocprofv2

| Kernel | 中位數 | 平均值 | 標準差 | 變異係數 | 範圍 |
|--------|--------|--------|--------|----------|------|
| **vectoradd** | 2,000 | 1,991.6 | 211.1 | 10.6% | 1,400 |
| scalar_ops | 1,620 | 1,647.8 | 262.7 | 15.9% | 1,800 |
| memory_ops | 1,800 | 1,818.5 | 131.9 | 7.3% | 520 |
| conditional | 1,820 | 1,823.5 | 143.7 | 7.9% | 641 |
| loop | 1,680 | 1,899.2 | 550.4 | 29.0% | 2,000 |

### rocprofv3

| Kernel | 中位數 | 平均值 | 標準差 | 變異係數 | 範圍 |
|--------|--------|--------|--------|----------|------|
| **vectoradd** | 1,760 | 1,758.0 | 134.0 | 7.6% | 640 |
| scalar_ops | 2,920 | 2,754.7 | 505.2 | 18.3% | 1,800 |
| memory_ops | 2,080 | 2,265.8 | 684.8 | 30.2% | 5,281 |
| conditional | 2,040 | 2,002.2 | 263.0 | 13.1% | 1,041 |
| loop | 2,000 | 1,892.7 | 322.4 | 17.0% | 1,120 |

---

## 分析與結論

### 1. Kernel 類型影響測量差異

| Kernel 類型 | 代表 | MDR vs rocprof 差異 | 說明 |
|-------------|------|---------------------|------|
| **Memory-bound** | vectoradd, memory_ops | 小 (1.04-1.30x) | memory 等待時間長，dispatch 占比小 |
| **Compute-bound** | scalar_ops, loop | 大 (2.06-3.71x) | 計算快，dispatch 占比大 |
| **Mixed** | conditional | 中 (1.08-1.21x) | 有分支，執行時間較長 |

### 2. 工具特性總結

| 工具 | 優點 | 缺點 | 最佳使用場景 |
|------|------|------|--------------|
| **MDR @TIMESTAMP** | 最接近純執行時間 | 需要注入程式碼 | 短 kernel、內部區段分析 |
| **rocprofv2** | memory-bound kernel 最穩定 | 包含 dispatch overhead | 標準 profiling、長 kernel |
| **rocprofv3** | vectoradd 最穩定 | memory_ops 異常不穩定 | 需要更多評估 |

### 3. 建議

- **短 kernel (< 1000 ticks)**：使用 MDR @TIMESTAMP
- **Memory-bound kernel**：rocprofv2 最穩定
- **需要精確純執行時間**：MDR @TIMESTAMP
- **標準 profiling 需求**：rocprofv2（業界標準）

---

## 實驗日期

2026-01-29
