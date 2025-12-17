# AMDISA 測試目錄重組總結

## 重組日期
2024-12-17

## 問題診斷

### 原始問題
您發現原本的 `universal_hsaco_runner.cpp` 是從原始的 HIP 程式使用 `hipcc` 編譯生成 `.hsaco`，而不是從 MLIR 轉換後的 `.s` 檔案生成。這無法驗證「從 `.s` 組裝檔案轉換後是否能執行」的核心需求。

### 錯誤的流程
```
HIP 源碼 → hipcc --genco → .hsaco → universal_hsaco_runner
```
**問題**：
- ❌ 跳過了從 `.s` 檔案的轉換驗證
- ❌ 無法證明重建的組裝檔案是否正確
- ❌ 測試的是原始編譯流程，不是 MLIR 轉換流程

## 解決方案

### 正確的流程
```
重建的 .s 檔案 (stage3_rebuilt.s)
    ↓ llvm-mc (組裝)
.o 物件檔
    ↓ ld.lld -shared (連結)
.hsaco 執行檔
    ↓ universal_hsaco_runner (執行)
GPU 驗證 ✅
```

**關鍵工具**：
1. `llvm-mc` - 將 `.s` 組裝成 `.o`
2. `ld.lld -shared` - 將 `.o` 連結成 `.hsaco`
3. `universal_hsaco_runner` - 在 GPU 上執行並驗證

## 重組內容

### 新增的測試腳本

1. **`test_asm_to_hsaco.sh`** ✅
   - 從 `.s` 檔案到 GPU 執行的完整流程
   - 單一檔案測試工具
   - 包含詳細的進度輸出

2. **`test_all_rebuilt_s.sh`** ✅
   - 批次測試所有 6 個 kernel
   - 自動化測試報告
   - 統計通過/失敗數量

3. **`test_mlir_pipeline.sh`** ⚠️ (待 AMDISA dialect 實作)
   - 完整的 MLIR 轉換管道測試
   - 包含 5 個階段的驗證
   - 需要 MLIR AMDISA dialect 工具支援

4. **`run_all_tests.sh`** ⚠️ (待 AMDISA dialect 實作)
   - 執行所有完整管道測試
   - 需要 MLIR 工具支援

### 新增的文檔

1. **`README.md`** ✅
   - 完整的測試套件說明
   - 使用範例和命令
   - 工具需求和環境配置

2. **`TEST_REPORT.md`** ✅
   - 詳細的測試結果報告
   - 每個 kernel 的執行細節
   - 檔案大小和性能數據

3. **`REORGANIZATION_SUMMARY.md`** ✅ (本文件)
   - 重組原因和過程
   - 問題診斷和解決方案

### 保留的檔案

1. **`universal_hsaco_runner.cpp`** ✅ (保持不變)
   - 通用 HSACO 執行器
   - 支援多種 kernel 類型
   - 已經正確實作

2. **`Makefile`** ✅ (保持不變)
   - 編譯 runner 的 Makefile

3. **`hip_kernels/`** ✅ (保持不變)
   - 原始 HIP kernel 源碼
   - 用於參考和比對

4. **`test_results/`** ✅ (保持不變)
   - 所有測試結果和中間檔案
   - 包含 `stage3_rebuilt.s` 等重要檔案

### 移動到 archive 的檔案

以下舊檔案已移動到 `archive_old_tests/`：

- `test_all_kernels_v2.sh`
- `test_execution_v3.sh`
- `test_full_gpu_execution.sh`
- `test_gpu_execution.sh`
- `extract_device_asm.sh`
- `hsaco_runner.cpp` (舊版本)
- 各種 `.log` 檔案
- 舊的 README 和報告文件

## 測試結果

### ✅ 完全通過 (6/6)

所有 6 個 kernel 的重建組裝檔案都能：

| Kernel | 組裝 | 連結 | 執行 | 驗證 |
|--------|------|------|------|------|
| test_01_vector_add | ✅ | ✅ | ✅ | ✅ |
| test_02_scalar_ops | ✅ | ✅ | ✅ | ✅ |
| test_03_memory_ops | ✅ | ✅ | ✅ | ✅ |
| test_04_conditional | ✅ | ✅ | ✅ | ✅ |
| test_05_loop | ✅ | ✅ | ✅ | ✅ |
| test_06_shared_memory | ✅ | ✅ | ✅ | ✅ |

### 關鍵驗證點

1. ✅ **組裝正確**：`llvm-mc` 成功將 `.s` 轉換為 `.o`
2. ✅ **連結正確**：`ld.lld -shared` 成功生成 `.hsaco`
3. ✅ **載入正確**：HIP runtime 能載入 `.hsaco` 模組
4. ✅ **執行正確**：Kernel 在 GPU 上成功執行
5. ✅ **結果正確**：計算結果通過驗證

## 目錄結構 (重組後)

```
AMDISA/
├── README.md                      # 主要說明文件 ✨ 新增
├── TEST_REPORT.md                 # 測試報告 ✨ 新增
├── REORGANIZATION_SUMMARY.md      # 重組總結 ✨ 新增
├── Makefile                       # 編譯 runner
├── universal_hsaco_runner.cpp     # 通用執行器
├── universal_hsaco_runner         # 編譯後的執行器
│
├── test_asm_to_hsaco.sh          # 單一檔案測試 ✨ 新增
├── test_all_rebuilt_s.sh         # 批次測試 ✨ 新增
├── test_mlir_pipeline.sh         # 完整管道測試 ✨ 新增
├── run_all_tests.sh              # 執行所有測試 ✨ 新增
│
├── hip_kernels/                  # 原始 HIP 源碼
├── test_results/                 # 測試結果
│   ├── test_01_vector_add/
│   │   ├── stage3_rebuilt.s          # 重建的組裝檔案
│   │   ├── stage3_rebuilt_test.o     # 組裝後的物件檔 ✨ 新生成
│   │   ├── stage3_rebuilt_test.hsaco # 連結後的執行檔 ✨ 新生成
│   │   └── rebuilt_test.log          # 測試日誌 ✨ 新生成
│   └── ...
│
└── archive_old_tests/            # 舊檔案存檔 ✨ 新增
    ├── test_full_gpu_execution.sh
    ├── hsaco_runner.cpp
    └── ...
```

## 使用方式

### 測試單一 Kernel

```bash
cd /home/morhuang/Project-MDR/Track_B/llvm-project/mlir/test/Dialect/AMDISA

./test_asm_to_hsaco.sh \
    test_results/test_01_vector_add/stage3_rebuilt.s \
    _Z9vectorAddPKfS0_Pfi \
    float_add \
    1024
```

### 測試所有 Kernel

```bash
./test_all_rebuilt_s.sh
```

## 技術細節

### 組裝命令

```bash
llvm-mc \
    -triple amdgcn-amd-amdhsa \
    -mcpu=gfx950 \
    -filetype=obj \
    input.s \
    -o output.o
```

### 連結命令

```bash
ld.lld \
    -shared \
    input.o \
    -o output.hsaco
```

這個連結命令是關鍵，參考來源：
- Track_A/e2e_test/Makefile
- Track_B/llvm-project/my_test/pipeline.py
- LLVM ROCDL Target 實作

### 執行命令

```bash
universal_hsaco_runner \
    kernel.hsaco \
    _Z9vectorAddPKfS0_Pfi \
    float_add \
    1024
```

## 重要發現

### 1. ld.lld 的使用

最初不清楚如何從 `.o` 生成 `.hsaco`，經過搜尋 codebase 發現：
- Track_A 的 Makefile 使用 `ld.lld -shared`
- MLIR ROCDL Target 的 C++ 實作也使用相同方法
- 這是標準的 AMD GPU code object 生成方式

### 2. MLIR 工具尚未支援 AMDISA

執行測試時發現：
- `mlir-translate` 沒有 `--import-amdisa` 選項
- `mlir-opt` 沒有 `--convert-amdisa-to-gpu` pass
- AMDISA dialect 可能還在開發中

因此，測試腳本 `test_mlir_pipeline.sh` 和 `run_all_tests.sh` 暫時無法使用。

### 3. 測試策略調整

由於 MLIR 工具限制，我們調整為：
- ✅ **可執行的測試**：`test_asm_to_hsaco.sh` 和 `test_all_rebuilt_s.sh`
- ⚠️ **未來的測試**：`test_mlir_pipeline.sh` 和 `run_all_tests.sh`

這樣至少驗證了執行階段的正確性，為 MLIR 工具的開發提供了測試基礎。

## 成果總結

### ✅ 已完成

1. ✅ 重新整理測試目錄結構
2. ✅ 創建正確的 `.s` → `.hsaco` 測試流程
3. ✅ 實作單一和批次測試腳本
4. ✅ 驗證所有 6 個 kernel 都能正確執行
5. ✅ 編寫完整的文檔和報告
6. ✅ 保留舊檔案到 archive 目錄

### ⚠️ 待完成 (需要 MLIR 支援)

1. ⚠️ MLIR AMDISA dialect 的實作
2. ⚠️ `mlir-translate --import-amdisa` 工具
3. ⚠️ `mlir-opt --convert-amdisa-to-gpu` pass
4. ⚠️ `mlir-translate --mlir-to-amdisa` 工具
5. ⚠️ 完整的端到端 MLIR 管道測試

## 結論

### 問題解決 ✅

原本的問題「驗證 .s 檔案是否能執行」已經完全解決：

1. ✅ 找到正確的工具鏈：`llvm-mc` + `ld.lld`
2. ✅ 實作完整的測試流程
3. ✅ 驗證所有 kernel 都能正確執行
4. ✅ 提供清晰的文檔和範例

### 測試基礎就緒 ✅

當 AMDISA dialect 的 MLIR 工具完成後，可以立即：
1. 使用 `test_mlir_pipeline.sh` 進行端到端測試
2. 執行 `run_all_tests.sh` 批次驗證
3. 確信執行階段是正確的（已驗證）

### 關鍵成就 🎯

- 🎯 證明了重建的 `.s` 檔案是完全可執行的
- 🎯 找到了正確的 `.o` → `.hsaco` 連結方式
- 🎯 建立了可重複的測試流程
- 🎯 為 MLIR 開發提供了驗證基礎

---

**重組執行者**: AI Assistant  
**完成日期**: 2024-12-17  
**測試狀態**: ✅ 全部通過 (6/6)  
**下一步**: 等待 AMDISA dialect MLIR 工具實作

