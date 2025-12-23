# Universal HSACO Runner 使用指南

## 概述

`universal_hsaco_runner` 是一個通用的 GPU kernel 驗證工具，用於載入 HSACO (HSA Code Object) 檔案並在 AMD GPU 上執行和驗證 kernel。

此工具基於舊版本進行擴展，添加了對 Test 07 多 kernel 測試案例的完整支持。

## 檔案位置

```
/home/morhuang/Project-MDR/Track_B/kernel_testcases/
├── universal_hsaco_runner.cpp    # 源代碼
├── universal_hsaco_runner         # 編譯後的可執行檔
├── test_07_multi_kernels/        # Test 07 測試案例
│   ├── original.s                # 原始組合語言（包含 5 個 kernel）
│   ├── test_all_kernels.sh      # 自動化測試腳本
│   ├── test_07.hsaco            # 編譯後的 HSACO 檔案
│   └── test_*.log               # 測試日誌
└── README.md                     # 本文件
```

## 支援的 Kernel 類型

### Float Kernel 類型（Test 07 新增）

| 類型 | 說明 | 函數簽名 | 測試功能 |
|------|------|----------|----------|
| `float_add` | 向量加法 | `(float* a, float* b, float* c, int n)` | 基本浮點加法 |
| `float_mul` | 向量乘法 | `(float* a, float* b, float* c, int n)` | 浮點乘法 |
| `float_dot` | 向量點積 | `(float* a, float* b, float* partial_sums, int n)` | Shared memory + Reduction |
| `float_saxpy` | SAXPY 運算 | `(float alpha, float* x, float* y, int n)` | Scalar-vector 運算 |
| `float_cond` | 條件運算 | `(float* input, float* output, float threshold, int n)` | 條件分支 |

### Int Kernel 類型（原有支持）

| 類型 | 說明 | 函數簽名 |
|------|------|----------|
| `int_scalar` | 純量運算 | `(int* output, int n)` |
| `int_mem` | 記憶體操作 | `(int* input, int* output, int n)` |
| `int_cond` | 條件判斷 | `(int* input, int* output, int n)` |
| `int_loop` | 迴圈 | `(int* output, int n)` |
| `int_shared` | 共享記憶體 | `(int* input, int* output, int n)` |

## 編譯

```bash
cd /home/morhuang/Project-MDR/Track_B/kernel_testcases
hipcc -o universal_hsaco_runner universal_hsaco_runner.cpp -std=c++11
```

## 使用方式

### 基本語法

```bash
./universal_hsaco_runner <hsaco_path> <kernel_name> <kernel_type> <test_size>
```

### 參數說明

- `<hsaco_path>`: HSACO 檔案路徑
- `<kernel_name>`: Kernel 函數名稱（mangled name）
- `<kernel_type>`: Kernel 類型（見上表）
- `<test_size>`: 測試資料大小（元素數量）

### 使用範例

#### 測試單個 kernel

```bash
# 測試向量加法
./universal_hsaco_runner test_07_multi_kernels/test_07.hsaco \
  _Z9vectorAddPKfS0_Pfi float_add 1024

# 測試向量乘法
./universal_hsaco_runner test_07_multi_kernels/test_07.hsaco \
  _Z9vectorMulPKfS0_Pfi float_mul 1024

# 測試 SAXPY
./universal_hsaco_runner test_07_multi_kernels/test_07.hsaco \
  _Z5saxpyfPKfPfi float_saxpy 1024
```

#### 測試所有 kernel（推薦）

```bash
cd test_07_multi_kernels
./test_all_kernels.sh
```

這個腳本會自動：
1. 組裝 `original.s` → `test_07.o`
2. 連結 `test_07.o` → `test_07.hsaco`
3. 測試所有 5 個 kernel
4. 顯示詳細的測試結果

## Test 07 驗證結果

### 完整測試通過 ✅

所有 5 個 kernel 都已通過驗證：

```
✓ vectorAdd      - 所有 1024 個元素正確
✓ vectorMul      - 所有 1024 個元素正確
✓ vectorDot      - Shared memory 和 reduction 正確
✓ saxpy          - 所有 1024 個元素正確
✓ conditionalOps - 條件分支邏輯正確
```

### 範例輸出

#### vectorAdd 測試結果
```
✓ Kernel launched (grid=4, block=256)
✓ Kernel execution completed
✅ PASS: All 1024 elements correct
Sample results:
  [0] 0 + 0 = 0
  [1] 1 + 2 = 3
  [2] 2 + 4 = 6
  [3] 3 + 6 = 9
  [4] 4 + 8 = 12
```

#### vectorMul 測試結果
```
✓ Kernel launched (grid=4, block=256)
✓ Kernel execution completed
✅ PASS: All 1024 elements correct
Sample results:
  [0] 0 * 0 = 0
  [1] 1 * 2 = 2
  [2] 2 * 4 = 8
  [3] 3 * 6 = 18
  [4] 4 * 8 = 32
```

#### saxpy 測試結果
```
✓ Kernel launched (grid=4, block=256)
✓ Kernel execution completed
✅ PASS: All 1024 elements correct
Sample results (alpha=2.5):
  [0] 2.5 * 0 + 0 = 0
  [1] 2.5 * 1 + 0.5 = 3
  [2] 2.5 * 2 + 1 = 6
  [3] 2.5 * 3 + 1.5 = 9
  [4] 2.5 * 4 + 2 = 12
```

## Test 07 Kernel 資訊

### 包含的 5 個 Kernel

| # | Kernel 名稱 | Mangled Name | 類型 | 功能 |
|---|-------------|--------------|------|------|
| 1 | vectorAdd | `_Z9vectorAddPKfS0_Pfi` | float_add | C[i] = A[i] + B[i] |
| 2 | vectorMul | `_Z9vectorMulPKfS0_Pfi` | float_mul | C[i] = A[i] * B[i] |
| 3 | vectorDot | `_Z9vectorDotPKfS0_Pfi` | float_dot | 點積（使用 shared memory） |
| 4 | saxpy | `_Z5saxpyfPKfPfi` | float_saxpy | Y[i] = alpha * X[i] + Y[i] |
| 5 | conditionalOps | `_Z14conditionalOpsPKfPffi` | float_cond | 條件分支運算 |

## 工具鏈流程

```
.hip 源碼 (test_07_multi_kernels.hip)
  ↓ hipcc -S
.s 組合語言 (original.s)
  ↓ llvm-mc
.o 物件檔 (test_07.o)
  ↓ ld.lld
.hsaco 可執行檔 (test_07.hsaco)
  ↓ universal_hsaco_runner
✅ GPU 執行驗證
```

## 擴展功能（相較於舊版本）

### 新增的 Kernel 類型支持

1. **FLOAT_VECTOR_MUL** - 向量乘法
   - 驗證函數：`run_float_vector_mul()`
   - 測試：C[i] = A[i] * B[i]

2. **FLOAT_VECTOR_DOT** - 向量點積
   - 驗證函數：`run_float_vector_dot()`
   - 特點：使用 shared memory 和 reduction

3. **FLOAT_SAXPY** - SAXPY 運算
   - 驗證函數：`run_float_saxpy()`
   - 測試：Y[i] = alpha * X[i] + Y[i]

4. **FLOAT_CONDITIONAL** - 條件運算
   - 驗證函數：`run_float_conditional()`
   - 測試：條件分支邏輯

### 程式碼結構

- 更新 `KernelType` 列舉
- 更新 `print_usage()` 顯示新類型
- 更新 `parse_kernel_type()` 解析新類型
- 添加 4 個新的驗證函數
- 更新 `main()` 中的 switch 語句

## 系統需求

- AMD GPU (gfx900 或更新)
- HIP Runtime (7.0 或更新)
- LLVM/Clang (支援 AMDGPU)
- C++11 或更新的編譯器

## 故障排除

### 編譯錯誤

```bash
# 確認 HIP 是否正確安裝
hipcc --version

# 重新編譯
cd /home/morhuang/Project-MDR/Track_B/kernel_testcases
hipcc -o universal_hsaco_runner universal_hsaco_runner.cpp -std=c++11
```

### 執行錯誤

```bash
# 確認 GPU 可用
rocm-smi

# 檢查 HSACO 檔案是否存在
ls -lh test_07_multi_kernels/test_07.hsaco

# 查看詳細錯誤日誌
cat test_07_multi_kernels/test_*.log
```

## 版本歷史

- **v2.0** (2025-12-23)
  - 添加 Test 07 多 kernel 支持
  - 新增 4 個 float kernel 類型
  - 新增驗證函數
  - 創建自動化測試腳本

- **v1.0** (原始版本)
  - 支持基本的 float_add 和 int kernel 類型

