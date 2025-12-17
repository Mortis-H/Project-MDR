# Track B AMDISA Dialect

## Universal HSACO Runner

### 工具說明

`universal_hsaco_runner` 是一個通用的 AMD GPU kernel 執行器，可以載入 `.hsaco` 檔案並在 GPU 上執行驗證。


**源碼**：
```
/Project-MDR/Track_B/llvm-project/mlir/test/Dialect/AMDISA/universal_hsaco_runner.cpp
```

### 編譯方式

```bash
cd /Project-MDR/Track_B/llvm-project/mlir/test/Dialect/AMDISA
make
```

或手動編譯：

```bash
/opt/rocm/bin/hipcc -std=c++11 -O2 universal_hsaco_runner.cpp -o universal_hsaco_runner
```

### 使用方式

#### 基本語法

```bash
universal_hsaco_runner <hsaco_path> <kernel_name> <kernel_type> <test_size>
```

#### 參數說明

1. **hsaco_path** - HSACO 檔案的路徑
2. **kernel_name** - Kernel 的名稱（通常是 mangled name）
3. **kernel_type** - Kernel 類型（見下表）
4. **test_size** - 測試資料大小

#### 支援的 Kernel 類型

| Kernel Type | 說明 | 函數簽名 |
|-------------|------|----------|
| `float_add` | Float 向量加法 | `(float* a, float* b, float* c, int n)` |
| `int_scalar` | Int 純量運算 | `(int* output, int n)` |
| `int_mem` | Int 記憶體操作 | `(int* input, int* output, int n)` |
| `int_cond` | Int 條件判斷 | `(int* input, int* output, int n)` |
| `int_loop` | Int 迴圈 | `(int* output, int n)` |
| `int_shared` | Int 共享記憶體 | `(int* input, int* output, int n)` |

### 使用範例

#### 範例 1: Float 向量加法

```bash
/Project-MDR/Track_B/llvm-project/mlir/test/Dialect/AMDISA/universal_hsaco_runner \
    kernel.hsaco \
    _Z9vectorAddPKfS0_Pfi \
    float_add \
    1024
```

#### 範例 2: Int 純量運算

```bash
/Project-MDR/Track_B/llvm-project/mlir/test/Dialect/AMDISA/universal_hsaco_runner \
    kernel.hsaco \
    _Z9scalarOpsPii \
    int_scalar \
    1024
```

#### 範例 3: Int 記憶體操作

```bash
/Project-MDR/Track_B/llvm-project/mlir/test/Dialect/AMDISA/universal_hsaco_runner \
    kernel.hsaco \
    _Z9memoryOpsPKiPii \
    int_mem \
    1024
```

### 輸出說明

#### 成功執行的輸出

```
========================================
Universal HSACO Runner
========================================
HSACO:  kernel.hsaco
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
  [3] 3 + 6 = 9
  [4] 4 + 8 = 12
========================================
```

#### 錯誤處理

- **找不到檔案**: 顯示檔案載入錯誤
- **Kernel 不存在**: 顯示找不到指定的 kernel 函數
- **執行失敗**: 顯示 HIP 錯誤訊息
- **結果錯誤**: 顯示不匹配的元素位置和值

## 完整測試流程

### 從 .s 檔案到 GPU 執行

```bash
# 1. 組裝 .s → .o
llvm-mc -triple amdgcn-amd-amdhsa -mcpu=gfx950 -filetype=obj \
    input.s -o output.o

# 2. 連結 .o → .hsaco
ld.lld -shared output.o -o output.hsaco

# 3. 執行驗證
universal_hsaco_runner output.hsaco kernel_name kernel_type 1024
```

### 使用測試腳本

更簡單的方式是使用提供的測試腳本：

```bash
cd /Project-MDR/Track_B/llvm-project/mlir/test/Dialect/AMDISA

# 測試單一檔案
./test_asm_to_hsaco.sh \
    test_results/test_01_vector_add/stage3_rebuilt.s \
    _Z9vectorAddPKfS0_Pfi \
    float_add \
    1024

# 測試所有 kernel
./test_all_rebuilt_s.sh
```


## 測試案例

測試目錄包含 6 個預先準備的測試案例：

| 測試 | 路徑 | Kernel | 類型 |
|------|------|--------|------|
| test_01 | `test_results/test_01_vector_add/` | `_Z9vectorAddPKfS0_Pfi` | `float_add` |
| test_02 | `test_results/test_02_scalar_ops/` | `_Z9scalarOpsPii` | `int_scalar` |
| test_03 | `test_results/test_03_memory_ops/` | `_Z9memoryOpsPKiPii` | `int_mem` |
| test_04 | `test_results/test_04_conditional/` | `_Z17conditionalKernelPKiPii` | `int_cond` |
| test_05 | `test_results/test_05_loop/` | `_Z10loopKernelPii` | `int_loop` |
| test_06 | `test_results/test_06_shared_memory/` | `_Z15sharedMemKernelPKiPii` | `int_shared` |

每個測試目錄包含：
- `stage3_rebuilt.s` - 重建的組裝檔案
- `original.s` - 原始組裝檔案
- 其他中間檔案和日誌

## 協作者快速開始

### 步驟 1: 設定環境變數（可選）

```bash
export LLVM_BUILD=/home/morhuang/llvm-project/build
export PATH=$LLVM_BUILD/bin:$PATH
```

### 步驟 2: 編譯 Runner

```bash
cd /Project-MDR/Track_B/llvm-project/mlir/test/Dialect/AMDISA
make
```

### 步驟 3: 執行測試

```bash
# 快速測試所有案例
./test_all_rebuilt_s.sh

# 或測試單一案例
./test_asm_to_hsaco.sh \
    test_results/test_01_vector_add/stage3_rebuilt.s \
    _Z9vectorAddPKfS0_Pfi \
    float_add \
    1024
```

### 步驟 4: 測試自己的 HSACO

```bash
# 如果您有自己的 .hsaco 檔案
/Project-MDR/Track_B/llvm-project/mlir/test/Dialect/AMDISA/universal_hsaco_runner \
    your_kernel.hsaco \
    your_kernel_name \
    kernel_type \
    test_size
```
