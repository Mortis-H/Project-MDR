# Test 07 Multi-Kernel 測試結果

## 測試日期
2025-12-19

## 🎯 測試目標
測試 pipeline.py 處理包含多個 kernel 的單一 HSACO 文件的能力。

## 📊 測試結果

### ✅ 成功的部分

1. **多 Kernel 解析成功**：
   - ✅ amdisa-translate 成功解析包含 3 個 kernel 的 .s 文件
   - ✅ 生成了 AMDISA MLIR
   - ✅ 轉換為 GPU MLIR 成功

2. **Kernel 識別**：
   - ✅ 所有 3 個 kernel 都被正確識別
   - ✅ 每個 kernel 的代碼都被提取

### ❌ 發現的問題

#### 問題 1: 標籤衝突 (Label Conflict) - **關鍵發現**

**錯誤信息**：
```
pipeline_output/original_rebuilt.s:153:2: error: symbol '.LBB0_2' is already defined
pipeline_output/original_rebuilt.s:219:2: error: symbol '.LBB0_2' is already defined
```

**問題描述**：

當多個 kernel 被組合在同一個 .s 文件中時，它們使用了相同的局部標籤名稱（例如 `.LBB0_2`, `.LBB0_1` 等）。在組裝階段，assembler 檢測到標籤重複定義，導致組裝失敗。

**根本原因**：

1. **原始 .s 文件**：每個 kernel 是獨立生成的，使用獨立的標籤命名空間
   ```assembly
   ; Kernel 1
   _Z9vectorAddPKfS0_Pfi:
     ...
     .LBB0_2:
     s_endpgm
   
   ; Kernel 2
   _Z9scalarOpsPii:
     ...
     .LBB0_2:    # ← 衝突！與 Kernel 1 的標籤重複
     s_endpgm
   ```

2. **轉換過程**：
   - amdisa-translate 將每個標籤轉換為 `llvm.inline_asm` 指令
   - 標籤被當作字符串保留，沒有重命名
   - 在重建 .s 時，標籤原樣輸出
   - Assembler 發現全局命名空間中有重複標籤

3. **為什麼原始 .s 沒問題**：
   原始 .s 文件是由編譯器生成的，編譯器確保了：
   - 每個 kernel 有唯一的標籤前綴（例如 `.LBB0_`, `.LBB1_`, `.LBB2_`）
   - 或使用函數作用域標籤

**影響範圍**：

- ❌ **所有**包含多個 kernel 的 .s 文件
- ❌ 任何有分支/循環的 kernel（會產生局部標籤）
- ✅ 單一 kernel 的文件不受影響（這就是為什麼 test_01-06 都通過了）

**問題嚴重性**：🔴 **HIGH**

這不是一個小問題，而是多 kernel 支持的根本性障礙。

## 🔍 詳細分析

### 轉換流程

```
original.s (3 kernels, 獨立標籤)
    ↓ [amdisa-translate -x s -emit=mlir]
AMDISA MLIR (3 kernels, 標籤作為字符串)
    ↓ [amdisa-translate -x mlir -emit=gpu]
GPU MLIR (3 kernels, inline_asm)
    ↓ [mlir-opt + passes]
original_rebuilt.s (3 kernels, 標籤衝突❌)
    ↓ [llvm-mc - 組裝失敗]
❌ 錯誤: symbol '.LBB0_2' is already defined
```

### 重現步驟

1. 創建包含多個 kernel 的 .s 文件
2. 運行 pipeline.py
3. 在組裝階段失敗，報告標籤重複定義

### 衝突的標籤

檢查 `pipeline_output/original_rebuilt.s`：

```bash
grep "\.LBB0_2:" pipeline_output/original_rebuilt.s
# 輸出：
# .LBB0_2:  (第 153 行 - Kernel 1)
# .LBB0_2:  (第 219 行 - Kernel 2)
```

## 💡 可能的解決方案

### 方案 1: 在 amdisa-translate 中重命名標籤（推薦）

在轉換階段為每個 kernel 的標籤添加唯一前綴：

```assembly
; Kernel 1
_Z9vectorAddPKfS0_Pfi:
  ...
  .LBB_vectorAdd_0_2:    # ← 添加 kernel 特定前綴
  s_endpgm

; Kernel 2
_Z9scalarOpsPii:
  ...
  .LBB_scalarOps_0_2:    # ← 不同的前綴
  s_endpgm
```

**優點**：
- 根本性解決問題
- 保持標籤語義
- 不影響其他轉換

**缺點**：
- 需要修改 amdisa-translate 源碼
- 需要追蹤每個 kernel 的標籤

### 方案 2: 使用數字後綴區分

為每個 kernel 添加序號後綴：

```assembly
.LBB0_2_K0:  # Kernel 0
.LBB0_2_K1:  # Kernel 1
.LBB0_2_K2:  # Kernel 2
```

**優點**：
- 實現簡單
- 容易自動化

**缺點**：
- 改變了標籤語義
- 可能影響調試信息

### 方案 3: 分離編譯每個 Kernel（臨時方案）

將多 kernel .s 文件拆分為多個單 kernel 文件，分別處理後再合併 HSACO。

**優點**：
- 可以立即使用
- 不需要修改工具

**缺點**：
- 只是繞過問題，不是真正解決
- 增加處理複雜度
- 失去了多 kernel 測試的意義

### 方案 4: 使用函數局部作用域

如果 assembler 支持函數作用域標籤，可以使用：

```assembly
_Z9vectorAddPKfS0_Pfi:
  ...
1:    # 局部標籤
  s_endpgm
```

但這需要檢查 AMDGPU assembler 的支持情況。

## 🎓 學到的教訓

1. **單元測試的重要性**：
   - test_01-06 都是單 kernel，所以都通過了
   - test_07 multi-kernel 立即暴露了問題
   - 證明了多樣化測試案例的價值

2. **標籤管理的複雜性**：
   - 標籤不僅僅是字符串，它們有作用域
   - 在多 kernel 環境中需要特別處理

3. **往返轉換的挑戰**：
   - 不是所有信息都能簡單地來回轉換
   - 需要考慮上下文（例如哪個 kernel 的標籤）

## 📊 測試狀態總結

| 階段 | 狀態 | 備註 |
|------|------|------|
| .s 文件創建 | ✅ 成功 | 包含 3 個 kernel |
| AMDISA MLIR 生成 | ✅ 成功 | 所有 kernel 都被識別 |
| GPU MLIR 生成 | ✅ 成功 | 轉換正確 |
| ISA 重建 | ✅ 成功 | .s 文件生成 |
| 組裝 (llvm-mc) | ❌ **失敗** | 標籤衝突 |
| HSACO 生成 | ⏭️  跳過 | 由於組裝失敗 |
| 執行測試 | ⏭️  跳過 | 由於 HSACO 未生成 |

**測試結果**：❌ **失敗** - 發現關鍵問題

## 🔄 下一步行動

1. **立即**：
   - ✅ 記錄這個問題（本文檔）
   - ✅ 創建 issue 追蹤
   - ⬜ 通知開發團隊

2. **短期**：
   - ⬜ 實現方案 2（數字後綴）作為臨時解決方案
   - ⬜ 驗證修復是否有效

3. **長期**：
   - ⬜ 實現方案 1（在 amdisa-translate 中正確處理）
   - ⬜ 添加更多多 kernel 測試案例
   - ⬜ 改進錯誤報告機制

## 📝 建議的修復優先級

1. 🔴 **P0 - Critical**：標籤衝突問題
   - 阻止所有多 kernel 文件的處理
   - 需要立即解決

2. 🟡 **P1 - High**：Metadata 保留
   - 注意到警告："No amdisa attributes found in GPU MLIR"
   - 可能影響某些 kernel 的正確性

3. 🟢 **P2 - Medium**：測試覆蓋
   - 添加更多邊界案例
   - 測試更複雜的 kernel 組合

## 🎯 結論

這個測試成功地揭示了一個**關鍵問題**：

> **Pipeline.py 當前不支持包含多個 kernel 的 HSACO 文件，因為它無法正確處理標籤命名衝突。**

雖然測試"失敗"了，但這正是我們需要的結果：
- ✅ 證明了測試案例的有效性
- ✅ 發現了實際問題
- ✅ 為修復提供了清晰的方向

這是一個**有價值的失敗**，因為它幫助我們理解了系統的限制。

---

**測試者**：AI Assistant  
**狀態**：問題已識別，等待修復  
**優先級**：🔴 Critical

