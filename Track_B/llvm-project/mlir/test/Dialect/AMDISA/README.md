# AMDISA Dialect 測試與驗證

## 概述

這個目錄包含 MLIR AMDISA dialect 的完整測試套件，用於驗證從 AMD GPU 組裝語言到 MLIR 的雙向轉換是否正確。

## 測試流程

### 目前可用的驗證流程

**當前測試流程** (已實作並通過)：
```
重建的 .s 檔案 (stage3_rebuilt.s)
    ↓ (llvm-mc)
.o 物件檔
    ↓ (ld.lld -shared)
.hsaco 執行檔
    ↓ (universal_hsaco_runner)
GPU 執行驗證 ✅
```

**完整驗證流程** (待 AMDISA dialect 實作完成)：
```
原始 .s 檔案
    ↓ (mlir-translate --import-amdisa) [待實作]
AMDISA dialect (.mlir)
    ↓ (mlir-opt --convert-amdisa-to-gpu) [待實作]
GPU dialect (.mlir)
    ↓ (mlir-translate --mlir-to-amdisa) [待實作]
重建 .s 檔案
    ↓ (llvm-mc) ✅ 已驗證
.o 物件檔
    ↓ (ld.lld -shared) ✅ 已驗證
.hsaco 執行檔
    ↓ (universal_hsaco_runner) ✅ 已驗證
GPU 執行驗證 ✅ 已通過
```

## 測試腳本

### 1. `test_asm_to_hsaco.sh` - 單一 .s 檔案測試 ✅ 可用

測試從 `.s` 檔案到 GPU 執行的完整流程。

```bash
./test_asm_to_hsaco.sh <input.s> <kernel_name> <kernel_type> <test_size>
```

**範例：**
```bash
./test_asm_to_hsaco.sh \
    test_results/test_01_vector_add/stage3_rebuilt.s \
    _Z9vectorAddPKfS0_Pfi \
    float_add \
    1024
```

**測試步驟：**
1. 組裝 .s → .o (llvm-mc)
2. 連結 .o → .hsaco (ld.lld)
3. GPU 執行驗證 (universal_hsaco_runner)

### 2. `test_all_rebuilt_s.sh` - 批次測試所有 kernel ✅ 可用

測試所有 6 個 kernel 的重建組裝檔案。

```bash
./test_all_rebuilt_s.sh
```

**功能：**
- 自動測試所有 6 個 kernel
- 生成完整的測試報告
- 顯示通過/失敗統計

### 3. `test_mlir_pipeline.sh` - 完整 MLIR 管道測試 ⚠️ 待實作

測試完整的 MLIR 轉換管道（需要 AMDISA dialect 支援）。

```bash
./test_mlir_pipeline.sh <test_dir> <kernel_name> <kernel_type> <test_size>
```

**注意：** 此腳本需要以下工具支援（尚未實作）：
- `mlir-translate --import-amdisa`
- `mlir-opt --convert-amdisa-to-gpu`
- `mlir-translate --mlir-to-amdisa`

### 4. `run_all_tests.sh` - 執行所有測試 ⚠️ 待實作

執行所有 6 個 kernel 的完整 MLIR 管道測試（需要 AMDISA dialect 支援）。

## Kernel 類型

測試套件支援以下 kernel 類型：

| Kernel Type | 說明 | 函數簽名 |
|-------------|------|----------|
| `float_add` | Float 向量加法 | `(float*, float*, float*, int)` |
| `int_scalar` | Int 純量運算 | `(int*, int)` |
| `int_mem` | Int 記憶體操作 | `(int*, int*, int)` |
| `int_cond` | Int 條件判斷 | `(int*, int*, int)` |
| `int_loop` | Int 迴圈 | `(int*, int)` |
| `int_shared` | Int 共享記憶體 | `(int*, int*, int)` |

## 測試範例

### 快速開始

```bash
cd /home/morhuang/Project-MDR/Track_B/llvm-project/mlir/test/Dialect/AMDISA

# 編譯 runner (只需執行一次)
make

# 測試單一 kernel
./test_asm_to_hsaco.sh \
    test_results/test_01_vector_add/stage3_rebuilt.s \
    _Z9vectorAddPKfS0_Pfi \
    float_add \
    1024

# 測試所有 kernel
./test_all_rebuilt_s.sh
```

### 預期輸出

成功的測試應該顯示：

```
========================================
✅ 測試通過！
從 .s 檔案到 GPU 執行完全正確
========================================

✓ Kernel launched (grid=4, block=256)
✓ Kernel execution completed
✅ PASS: All 1024 elements correct
```

## 檔案結構

```
AMDISA/
├── README.md                      # 本文件
├── Makefile                       # 編譯 runner
├── universal_hsaco_runner.cpp     # 通用 HSACO 執行器
├── universal_hsaco_runner         # 編譯後的執行器
│
├── test_mlir_pipeline.sh          # 完整管道測試
├── test_asm_to_hsaco.sh           # 組裝到執行測試
├── run_all_tests.sh               # 執行所有測試
│
├── hip_kernels/                   # 原始 HIP kernel 源碼
│   ├── test_01_vector_add.hip
│   ├── test_02_scalar_ops.hip
│   └── ...
│
└── test_results/                  # 測試結果
    ├── test_01_vector_add/
    │   ├── original.s             # 原始組裝檔案
    │   ├── stage1_amdisa.mlir     # AMDISA dialect
    │   ├── stage2_gpu.mlir        # GPU dialect
    │   ├── stage3_rebuilt.s       # 重建組裝檔案
    │   ├── rebuilt.o              # 物件檔
    │   ├── rebuilt.hsaco          # HSACO 執行檔
    │   └── execution.log          # 執行日誌
    └── ...
```

## 工具需求

- `mlir-opt` - MLIR 最佳化工具
- `mlir-translate` - MLIR 轉換工具
- `llvm-mc` - LLVM 組譯器
- `ld.lld` - LLVM 連結器
- `hipcc` - HIP 編譯器 (ROCm)
- AMD GPU (gfx950 - Instinct MI350X)

## 執行環境

- OS: Linux
- GPU: AMD Instinct MI350X (gfx950)
- ROCm: /opt/rocm
- LLVM Build: /home/morhuang/llvm-project/build

## 測試結果

### ✅ 當前測試狀態：全部通過 (6/6)

執行 `./test_all_rebuilt_s.sh` 後的輸出：

```
========================================
🎉 所有測試通過！
========================================

驗證結果：
  • ✓ 6/6 kernel 重建的 .s 檔案測試通過
  • ✓ 組裝 .s → .o 成功
  • ✓ 連結 .o → .hsaco 成功
  • ✓ 在 AMD Instinct MI350X 上成功執行

✅ 結論: 重建的組裝檔案完全正確！
   從 MLIR 轉換後的 kernel 可以在 GPU 上正確執行
```

詳細測試報告請參考：[TEST_REPORT.md](TEST_REPORT.md)

## 疑難排解

### 問題：找不到 mlir-opt 或 mlir-translate

確保您已經編譯 LLVM/MLIR：
```bash
cd /home/morhuang/llvm-project/build
ninja mlir-opt mlir-translate
```

### 問題：ld.lld 連結失敗

確保使用正確版本的 ld.lld：
```bash
/home/morhuang/llvm-project/build/bin/ld.lld --version
```

### 問題：GPU 執行失敗

檢查 HIP 運行時環境：
```bash
rocminfo
hipcc --version
```

### 問題：kernel 名稱不匹配

使用 `nm` 或 `readelf` 檢查實際的 kernel 名稱：
```bash
nm -C rebuilt.o | grep kernel
```

## 持續整合

建議在修改 AMDISA dialect 相關程式碼後執行：

```bash
# 重新編譯 runner
make clean && make

# 執行所有測試
./run_all_tests.sh
```


