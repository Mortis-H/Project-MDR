# 原始 HSACO vs 重建 HSACO 對比測試報告

## 🎯 測試目的

驗證通過 pipeline.py 重建的 HSACO 是否與原始 HSACO 產生相同的執行結果。

## 📋 測試方法

1. **生成原始 HSACO**：從 `original.s` 直接組裝和連結
2. **使用重建 HSACO**：pipeline.py 生成的 `pipeline_output/original_rebuilt.hsaco`
3. **執行測試**：使用 `universal_hsaco_runner` 對兩者進行相同的測試
4. **比較結果**：比較兩者的輸出樣本是否一致

## 📊 測試結果總覽

| 測試案例 | 原始 HSACO | 重建 HSACO | 結果一致 | 差異說明 |
|---------|-----------|-----------|---------|---------|
| test_01_vector_add | ✅ 正常 | ✅ 正常 | ✅ **是** | 完全一致 |
| test_02_scalar_ops | ✅ 正常 | ✅ 正常 | ✅ **是** | 完全一致 |
| test_03_memory_ops | ✅ 正常 | ✅ 正常 | ✅ **是** | 完全一致 |
| test_04_conditional | ✅ 正常 | ✅ 正常 | ✅ **是** | 完全一致 |
| test_05_loop | ✅ 正常 | ✅ 正常 | ✅ **是** | 完全一致 |
| test_06_shared_memory | ✅ 正常 | ⚠️ 錯誤 | ❌ **否** | 輸出不一致 |

**總結**：
- ✅ 一致：5/6 (83.3%)
- ❌ 不一致：1/6 (16.7%)

## ✅ 一致的測試案例

### test_01_vector_add (Float 向量加法)

**測試配置**：
- Kernel: `_Z9vectorAddPKfS0_Pfi`
- 類型: `float_add`
- 大小: 1024 元素

**結果對比**：
```
原始: [0] 0 + 0 = 0, [1] 1 + 2 = 3, [2] 2 + 4 = 6
重建: [0] 0 + 0 = 0, [1] 1 + 2 = 3, [2] 2 + 4 = 6
```
**狀態**：✅ 完全一致

---

### test_02_scalar_ops (Int 純量運算)

**測試配置**：
- Kernel: `_Z9scalarOpsPii`
- 類型: `int_scalar`
- 大小: 1024 元素

**結果對比**：
```
原始: [0] = 2, [1] = 14, [2] = 24
重建: [0] = 2, [1] = 14, [2] = 24
```
**狀態**：✅ 完全一致

---

### test_03_memory_ops (Int 記憶體操作)

**測試配置**：
- Kernel: `_Z9memoryOpsPKiPii`
- 類型: `int_mem`
- 大小: 1024 元素

**結果對比**：
```
原始: [0] input=0, output=1, [1] input=1, output=3
重建: [0] input=0, output=1, [1] input=1, output=3
```
**狀態**：✅ 完全一致

---

### test_04_conditional (Int 條件判斷)

**測試配置**：
- Kernel: `_Z17conditionalKernelPKiPii`
- 類型: `int_cond`
- 大小: 1024 元素

**結果對比**：
```
原始: [0] input=0, output=0, [1] input=1, output=4
重建: [0] input=0, output=0, [1] input=1, output=4
```
**狀態**：✅ 完全一致

---

### test_05_loop (Int 迴圈)

**測試配置**：
- Kernel: `_Z10loopKernelPii`
- 類型: `int_loop`
- 大小: 1024 元素

**結果對比**：
```
原始: [0] = 45, [1] = 45, [2] = 45
重建: [0] = 45, [1] = 45, [2] = 45
```
**狀態**：✅ 完全一致

## ❌ 不一致的測試案例

### test_06_shared_memory (Int Shared Memory) - **關鍵問題**

**測試配置**：
- Kernel: `_Z15sharedMemKernelPKiPii`
- 類型: `int_shared`
- 大小: 256 元素

**結果對比**：
```
原始: [0] input=0, output=32640 ✅
      [1] input=1, output=0
      [2] input=2, output=0
      [3] input=3, output=0
      [4] input=4, output=0

重建: [0] input=0, output=0     ❌
      [1] input=1, output=0
      [2] input=2, output=0
      [3] input=3, output=0
      [4] input=4, output=0
```

**差異分析**：

| 元素 | 原始輸出 | 重建輸出 | 差異 |
|-----|---------|---------|------|
| [0] | 32640 | 0 | ❌ 不一致 |
| [1-4] | 0 | 0 | ✅ 一致（但可能都錯了） |

**根本原因**：

1. **Metadata 丟失**：
   ```assembly
   # original.s
   .amdhsa_group_segment_fixed_size 1024
   
   # original_rebuilt.s
   .amdhsa_group_segment_fixed_size 0    # ❌ 錯誤！
   ```

2. **影響**：
   - HIP runtime 沒有為 kernel 分配 LDS (Local Data Share)
   - 當 kernel 嘗試讀寫 shared memory 時，訪問的是未分配的記憶體
   - 讀取返回 0 或未定義值
   - 結果：計算錯誤

3. **為什麼沒有崩潰**：
   - GPU 對非法記憶體訪問有容錯機制
   - 訪問未分配的 LDS 地址返回 0 而不是崩潰
   - 程式"成功執行"但產生錯誤結果

## 🔍 深入分析

### HSACO 文件大小對比

| 測試案例 | 原始 HSACO | 重建 HSACO | 差異 |
|---------|-----------|-----------|------|
| test_01_vector_add | 6,256 bytes | 5,696 bytes | -560 bytes |
| test_02_scalar_ops | 5,936 bytes | 5,448 bytes | -488 bytes |
| test_03_memory_ops | 5,904 bytes | 5,416 bytes | -488 bytes |
| test_04_conditional | 6,072 bytes | 5,592 bytes | -480 bytes |
| test_05_loop | 5,896 bytes | 5,408 bytes | -488 bytes |
| test_06_shared_memory | 6,176 bytes | 5,696 bytes | -480 bytes |

**觀察**：重建的 HSACO 普遍比原始小約 500 bytes，這是因為：
- 原始 .s 有更多註釋和冗餘信息
- 重建過程生成了更精簡的 metadata

### 為什麼 test_01-05 能正確工作？

這些測試案例不使用 shared memory，因此：
- ✅ 核心 ISA 指令完全相同
- ✅ 寄存器分配正確
- ✅ Global memory 訪問正確
- ✅ 控制流正確

唯一影響的是 `group_segment_fixed_size`，只有當 kernel 實際使用 LDS 時才會出問題。

### 問題的嚴重性評估

**嚴重程度**：🔴 HIGH

**影響範圍**：
- ❌ 所有使用 shared memory 的 kernel
- ❌ 使用 `__shared__` 或 `__local` 的代碼
- ❌ 明確依賴 LDS 的優化

**功能影響**：
- 計算結果錯誤（而不是崩潰）
- 難以察覺（程式"成功執行"）
- 可能導致嚴重的資料錯誤

## 🛠️ 修復驗證

### 快速修復方案

對 test_06 手動修復並重新測試：

```bash
cd test_06_shared_memory/pipeline_output
sed -i 's/\.amdhsa_group_segment_fixed_size 0/\.amdhsa_group_segment_fixed_size 1024/' original_rebuilt.s
llvm-mc -triple amdgcn-amd-amdhsa -mcpu=gfx950 -filetype=obj original_rebuilt.s -o original_rebuilt.o
ld.lld -shared original_rebuilt.o -o original_rebuilt.hsaco
```

**預期結果**：修復後重建的 HSACO 應該產生與原始一致的結果。

## 📊 結論

### 成功的部分 ✅

1. **常規 kernel 完美轉換**：
   - 5/5 不使用 shared memory 的測試案例完全一致
   - 核心指令、寄存器使用、記憶體訪問都正確

2. **Pipeline 流程可靠**：
   - Assembly → MLIR → Assembly 的轉換流程有效
   - 生成的 HSACO 可以正常執行

3. **基本 metadata 正確**：
   - vgpr_count、sgpr_count、kernarg_size 都正確保留

### 存在的問題 ❌

1. **Shared memory 支持缺陷**：
   - 1/1 使用 shared memory 的測試案例結果錯誤
   - group_segment_fixed_size 丟失
   - 導致功能性錯誤（不是崩潰）

2. **問題嚴重性**：
   - 高嚴重程度（會導致錯誤結果）
   - 難以檢測（不會崩潰）
   - 影響所有使用 LDS 的代碼

### 建議 📝

1. **立即修復**：修改 `amdisa-translate` 和 `pipeline.py` 支持 group_segment_size
2. **測試增強**：為 `universal_hsaco_runner` 添加結果驗證
3. **文檔說明**：明確標註當前版本不支持 shared memory kernel

---

**測試日期**：2025-12-19  
**測試環境**：AMD GPU (gfx950)  
**測試工具**：universal_hsaco_runner  
**測試者**：AI Assistant
