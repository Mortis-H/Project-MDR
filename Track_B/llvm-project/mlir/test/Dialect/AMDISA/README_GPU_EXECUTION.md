# GPU 執行驗證指南

## 🎯 快速開始

### 1. 編譯 Runner

```bash
cd /home/morhuang/Project-MDR/Track_B/llvm-project/mlir/test/Dialect/AMDISA
make all
```

這會編譯：
- `hsaco_runner` - 原始的 float 向量加法 runner
- `universal_hsaco_runner` - 支援所有 kernel 類型的通用 runner

### 2. 執行完整測試

```bash
./test_full_gpu_execution.sh
```

這會：
- 為所有 6 個 kernel 生成 HSACO
- 在 GPU 上執行每個 kernel
- 驗證執行結果
- 生成完整報告

**預期輸出**: ✅ 所有測試通過！

---

## 📋 可用的測試腳本

### test_all_kernels_v2.sh
**完整的 MLIR 轉換和工具鏈驗證**

```bash
./test_all_kernels_v2.sh
```

驗證內容：
- MLIR 轉換 (Assembly → AMDISA → GPU → Assembly)
- 組譯驗證 (clang)
- 連結驗證 (ld.lld)
- Binary 一致性 (Object 檔案比較)

**不包含**: GPU 執行

### test_full_gpu_execution.sh
**GPU 執行驗證**

```bash
./test_full_gpu_execution.sh
```

驗證內容：
- 從 HIP 源碼生成 HSACO
- 在 GPU 上執行所有 6 個 kernel
- 驗證執行結果

**專注於**: 實際 GPU 執行

### 組合使用（推薦）

```bash
# 先運行完整測試
./test_all_kernels_v2.sh

# 然後運行 GPU 執行測試
./test_full_gpu_execution.sh
```

這樣可以獲得最完整的驗證！

---

## 🔧 Universal HSACO Runner 使用方式

### 基本用法

```bash
./universal_hsaco_runner <hsaco_path> <kernel_name> <kernel_type> <test_size>
```

### Kernel 類型

| 類型 | 說明 | 參數簽名 |
|------|------|----------|
| `float_add` | Float 向量加法 | (float*, float*, float*, int) |
| `int_scalar` | Int 標量運算 | (int*, int) |
| `int_mem` | Int 記憶體操作 | (int*, int*, int) |
| `int_cond` | Int 條件分支 | (int*, int*, int) |
| `int_loop` | Int 迴圈 | (int*, int) |
| `int_shared` | Int 共享記憶體 | (int*, int*, int) |

### 範例

```bash
# Float 向量加法
./universal_hsaco_runner \
    test_results/test_01_vector_add/exec.hsaco \
    "_Z9vectorAddPKfS0_Pfi" \
    float_add \
    1024

# Int 標量運算
./universal_hsaco_runner \
    test_results/test_02_scalar_ops/exec.hsaco \
    "_Z9scalarOpsPii" \
    int_scalar \
    1024

# Int 共享記憶體
./universal_hsaco_runner \
    test_results/test_06_shared_memory/exec.hsaco \
    "_Z15sharedMemKernelPKiPii" \
    int_shared \
    1024
```

---

## 📊 Kernel 配置表

| Kernel | Mangled Name | 類型 | 測試大小 |
|--------|--------------|------|---------|
| test_01_vector_add | `_Z9vectorAddPKfS0_Pfi` | float_add | 1024 |
| test_02_scalar_ops | `_Z9scalarOpsPii` | int_scalar | 1024 |
| test_03_memory_ops | `_Z9memoryOpsPKiPii` | int_mem | 1024 |
| test_04_conditional | `_Z17conditionalKernelPKiPii` | int_cond | 1024 |
| test_05_loop | `_Z10loopKernelPii` | int_loop | 1024 |
| test_06_shared_memory | `_Z15sharedMemKernelPKiPii` | int_shared | 1024 |

---

## 🔍 手動測試流程

### Step 1: 生成 HSACO

```bash
/opt/rocm/bin/hipcc --genco --offload-arch=gfx950 \
    hip_kernels/test_01_vector_add.hip \
    -o test.hsaco
```

### Step 2: 執行 Kernel

```bash
./universal_hsaco_runner \
    test.hsaco \
    "_Z9vectorAddPKfS0_Pfi" \
    float_add \
    1024
```

### Step 3: 查看結果

成功輸出範例：
```
========================================
Universal HSACO Runner
========================================
HSACO:  test.hsaco
Kernel: _Z9vectorAddPKfS0_Pfi
Type:   float_add
Size:   1024
========================================
✓ Module loaded successfully
✓ Function found: _Z9vectorAddPKfS0_Pfi
========================================
✓ Kernel launched (grid=4, block=256)
✓ Kernel execution completed
✅ PASS: All 1024 elements correct
Sample results:
  [0] 0 + 0 = 0
  [1] 1 + 2 = 3
  [2] 2 + 4 = 6
========================================
```

---

## 📁 測試結果位置

### 完整測試結果
```
test_results/
├── SUMMARY.md                    # 總體報告
├── test_01_vector_add/
│   ├── TEST_REPORT.md           # 詳細報告
│   ├── exec.hsaco               # 可執行的 HSACO
│   ├── gpu_exec.log             # GPU 執行日誌
│   ├── original.o               # Original object 檔案
│   ├── rebuilt.o                # Rebuilt object 檔案
│   └── ...
├── test_02_scalar_ops/
│   └── ...
└── ...
```

### 重要檔案
- `gpu_exec.log` - GPU 執行的完整輸出
- `TEST_REPORT.md` - 包含所有驗證結果
- `exec.hsaco` - 可以直接執行的 GPU code object

---

## 🐛 故障排除

### 問題: Kernel 名稱找不到

```
Failed to get function 'myKernel': named symbol not found
```

**解決方案**: 使用 mangled 名稱

```bash
# 提取實際的符號名稱
/opt/rocm/llvm/bin/clang-offload-bundler \
    -type=o \
    -targets=hipv4-amdgcn-amd-amdhsa--gfx950 \
    -input=test.hsaco \
    -output=temp.o \
    -unbundle

/opt/rocm/llvm/bin/llvm-readelf -s temp.o | grep FUNC
```

### 問題: GPU 不可用

```
HIP error: no device is available
```

**檢查步驟**:
1. 確認 GPU 存在: `rocminfo | grep gfx`
2. 檢查權限: `groups` (應包含 `render`)
3. 測試 HIP: `hipconfig --check`

### 問題: HSACO 載入失敗

```
Failed to load module: no kernel image is available
```

**可能原因**:
- GPU 架構不匹配
- HSACO 檔案損壞
- 需要重新生成 HSACO

**解決方案**:
```bash
# 確認 GPU 架構
rocminfo | grep "Name:" | grep gfx

# 使用正確的架構重新生成
/opt/rocm/bin/hipcc --genco --offload-arch=gfx950 source.hip -o output.hsaco
```

---

## 📚 相關文檔

- `COMPLETE_VALIDATION_REPORT.md` - 完整驗證報告
- `GPU_EXECUTION_SUMMARY.md` - GPU 執行總結
- `EXECUTION_VALIDATION.md` - 執行驗證說明
- `README_TEST_RESULTS.md` - 測試結果說明

---

## 💡 提示與技巧

### 快速驗證單個 Kernel

```bash
# 1. 生成 HSACO
hipcc --genco --offload-arch=gfx950 hip_kernels/test_01_vector_add.hip -o test.hsaco

# 2. 執行
./universal_hsaco_runner test.hsaco "_Z9vectorAddPKfS0_Pfi" float_add 1024
```

### 自動提取 Kernel 名稱

```bash
# 從 HSACO 提取所有函數
clang-offload-bundler -type=o -targets=hipv4-amdgcn-amd-amdhsa--gfx950 \
    -input=test.hsaco -output=temp.o -unbundle
llvm-readelf -s temp.o | grep FUNC | grep -v UND
```

### 批量測試

```bash
# 測試所有 kernel
for hip in hip_kernels/*.hip; do
    name=$(basename "$hip" .hip)
    echo "Testing $name..."
    hipcc --genco --offload-arch=gfx950 "$hip" -o "${name}.hsaco"
done
```

---

## ✅ 驗證檢查清單

執行驗證前，確保：

- [ ] GPU 硬體可用 (`rocminfo` 顯示 GPU)
- [ ] ROCm 已安裝 (`hipconfig --check` 成功)
- [ ] 編譯 runner (`make all` 成功)
- [ ] 測試數據目錄存在 (`test_results/` 目錄)

執行後檢查：

- [ ] 所有 kernel 編譯成功 (6/6 HSACO 生成)
- [ ] 所有 kernel 執行成功 (6/6 顯示 ✅ PASS)
- [ ] Object 檔案一致 (6/6 完全一致)
- [ ] 查看完整報告 (`COMPLETE_VALIDATION_REPORT.md`)

---

**文檔版本**: 1.0  
**最後更新**: 2025-12-17  
**維護者**: Track B Team

