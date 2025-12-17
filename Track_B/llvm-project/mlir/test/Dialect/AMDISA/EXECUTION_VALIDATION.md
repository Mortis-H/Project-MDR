# HSACO 執行驗證說明

## 概述

這個文件說明如何使用 `hsaco_runner` 來驗證 Track B 中 MLIR 轉換後的 GPU kernel 是否能正確執行。

## 架構

```
┌─────────────┐
│ .hip 檔案   │
└──────┬──────┘
       │ hipcc
       ↓
┌─────────────┐
│ .s (bundle) │
└──────┬──────┘
       │ extract
       ↓
┌─────────────┐     ┌──────────────────┐
│ original.s  │────→│ MLIR 轉換流程    │
└─────────────┘     │ (amdisa→gpu)     │
                    └────────┬─────────┘
                             │
                             ↓
                    ┌─────────────────┐
                    │ rebuilt.s       │
                    └────────┬────────┘
                             │
            ┌────────────────┴────────────────┐
            │                                 │
            ↓                                 ↓
    ┌──────────────┐                 ┌──────────────┐
    │ original.o   │                 │ rebuilt.o    │
    └──────┬───────┘                 └──────┬───────┘
           │ ld.lld                         │ ld.lld
           ↓                                 ↓
    ┌──────────────┐                 ┌──────────────┐
    │ original.out │                 │ rebuilt.out  │
    └──────┬───────┘                 └──────┬───────┘
           │ bundler                        │ bundler
           ↓                                 ↓
    ┌──────────────┐                 ┌──────────────┐
    │original.hsaco│                 │rebuilt.hsaco │
    └──────┬───────┘                 └──────┬───────┘
           │                                 │
           │ hsaco_runner                   │ hsaco_runner
           ↓                                 ↓
    ┌──────────────┐                 ┌──────────────┐
    │執行結果 A    │                 │執行結果 B    │
    └──────┬───────┘                 └──────┬───────┘
           │                                 │
           └────────────┬────────────────────┘
                        │
                        ↓
                   ┌─────────┐
                   │比較結果 │
                   └─────────┘
```

## 工具：hsaco_runner

### 編譯

```bash
cd /home/morhuang/Project-MDR/Track_B/llvm-project/mlir/test/Dialect/AMDISA
make hsaco_runner
```

### 使用方式

```bash
./hsaco_runner <hsaco_path> <kernel_name> <test_size>
```

**參數說明**：
- `hsaco_path`: .hsaco 檔案路徑
- `kernel_name`: kernel 函數名稱（如 `vectorAdd`）
- `test_size`: 測試數據大小（元素數量）

**範例**：

```bash
./hsaco_runner test_results/test_01_vector_add/original.hsaco vectorAdd 1024
```

### 輸出

runner 會：
1. 載入 .hsaco 模組
2. 取得 kernel 函數
3. 準備測試數據（A[i] = i, B[i] = i*2）
4. 執行 kernel（計算 C = A + B）
5. 驗證結果是否正確
6. 顯示執行摘要和樣本結果

**成功輸出範例**：

```
========================================
HSACO Runner
========================================
HSACO:  original.hsaco
Kernel: vectorAdd
Size:   1024
========================================
✓ Module loaded successfully
✓ Function found: vectorAdd
✓ Kernel launched (grid=4, block=256)
✓ Kernel execution completed
========================================
✅ PASS: All 1024 elements correct
Sample results:
  [0] 0.000 + 0.000 = 0.000
  [1] 1.000 + 2.000 = 3.000
  [2] 2.000 + 4.000 = 6.000
  [3] 3.000 + 6.000 = 9.000
  [4] 4.000 + 8.000 = 12.000
========================================
```

## 整合到測試腳本

`test_all_kernels_v2.sh` 已經整合了執行驗證：

### 新增步驟

**Step 12**: 執行 original.hsaco
- 載入並執行原始的 .hsaco 檔案
- 記錄執行結果到 `original_run.log`

**Step 13**: 執行 rebuilt.hsaco
- 載入並執行重建的 .hsaco 檔案
- 記錄執行結果到 `rebuilt_run.log`
- 比較兩次執行的輸出結果

### 執行測試

```bash
cd /home/morhuang/Project-MDR/Track_B/llvm-project/mlir/test/Dialect/AMDISA
./test_all_kernels_v2.sh
```

## 驗證層級

測試腳本現在提供**五層驗證**：

1. **語法驗證** (clang 組譯)
   - 檢查 assembly 語法是否正確
   - 工具：`clang -x assembler`

2. **連結驗證** (ld.lld)
   - 檢查符號解析和重定位
   - 工具：`ld.lld`

3. **封裝驗證** (clang-offload-bundler)
   - 確保 HIP 可執行格式正確
   - 工具：`clang-offload-bundler`

4. **大小比較**
   - 比較各階段檔案大小（.o, .out, .hsaco）
   - 相同大小通常表示二進制相同

5. **✨ 執行驗證** (hsaco_runner) - 最終驗證
   - 實際在 GPU 上執行
   - 驗證計算結果正確性
   - 比較 original 和 rebuilt 的執行結果

## 測試報告

測試完成後，每個 kernel 的測試目錄會包含：

### 執行日誌
- `original_run.log` - Original HSACO 執行日誌
- `rebuilt_run.log` - Rebuilt HSACO 執行日誌

### TEST_REPORT.md

報告會包含：

```markdown
## 實際執行驗證

### Original HSACO 執行結果
✅ PASS: All 1024 elements correct

### Rebuilt HSACO 執行結果
✅ PASS: All 1024 elements correct

### 執行結果比較
✅ **執行結果完全一致**
```

## 注意事項

### Kernel 名稱偵測

腳本會自動從 .hip 檔案中提取 kernel 函數名稱：

```bash
kernel_function=$(grep -oP '__global__\s+\w+\s+\K\w+(?=\s*\()' "$hip_file" | head -1)
```

如果偵測失敗，會使用預設值 `vectorAdd`。

### 測試數據

- 預設測試大小：1024 個元素
- 可以通過修改 `TEST_SIZE` 變數來調整

### 支援的 Kernel 類型

目前 `hsaco_runner` 針對向量運算 kernel 設計，假設 kernel 簽名為：

```cpp
__global__ void kernel_name(const float* a, const float* b, float* c, int n)
```

對於其他類型的 kernel，可能需要修改 `hsaco_runner.cpp` 來適應不同的參數結構。

## 疑難排解

### 執行失敗

如果看到執行失敗：

```
❌ FAIL: 128 errors found
```

請檢查：
1. `original_run.log` 和 `rebuilt_run.log` 中的詳細錯誤信息
2. Kernel 名稱是否正確偵測
3. HSACO 檔案是否正確生成

### 結果不一致

如果執行結果不一致：

```
⚠️ 執行結果有差異
```

這可能表示 MLIR 轉換過程中**改變了語義**，需要：
1. 檢查 AMDISA → GPU MLIR 的降級過程
2. 比較 `original.s` 和 `rebuilt.s` 的差異
3. 檢查 inline assembly 的正確性

## 總結

加入執行驗證後，Track B 的測試流程現在可以：

✅ 驗證語法正確性（組譯）
✅ 驗證連結正確性（連結）
✅ 驗證格式正確性（封裝）
✅ 驗證大小一致性（比較）
✅ **驗證語義正確性（執行）** ← 最重要！

這確保了 MLIR 轉換過程不僅在語法上正確，而且在**功能上也完全保留了原始 kernel 的行為**。

