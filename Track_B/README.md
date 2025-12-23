# Track B: AMDISA Dialect 工具使用指南

本目錄包含兩個主要工具，用於 AMD GPU kernel 的開發、測試和驗證。

## 目錄

- [1. amdisa-translate Pipeline 工具](#1-amdisa-translate-pipeline-工具)
- [2. Universal HSACO Runner 工具](#2-universal-hsaco-runner-工具)

---

## 1. amdisa-translate Pipeline 工具

### 概述

`pipeline.py` 是一個完整的 AMD ISA 到 MLIR 轉換工具，支援從 `.s` 組合語言檔案或 `.mlir` 檔案生成可執行的 HSACO。

**工具位置**: `/home/morhuang/Project-MDR/Track_B/llvm-project/my_test/pipeline.py`

### Pipeline 流程圖

```
輸入 .s (AMD ISA Assembly)
    ↓
[Stage 1] amdisa-translate -emit=mlir
    ↓
.amdisamlir (AMDISA Dialect MLIR)
    ↓
[Stage 2] amdisa-translate -emit=gpu
    ↓
.gpumlir (GPU Dialect MLIR)
    ↓
[Stage 3] mlir-opt (優化 pipeline)
    - gpu-kernel-outlining
    - rocdl-attach-target
    - convert-gpu-to-rocdl
    - gpu-to-llvm
    - gpu-module-to-binary{format=isa/llvm}
    ↓
.s (重建的 ISA) / .bc (LLVM Bitcode)
    ↓
[Stage 4] llvm-mc (組裝)
    ↓
.o (Object File)
    ↓
[Stage 5] ld.lld (連結)
    ↓
.hsaco (可執行的 HSA Code Object)
```

### 基本使用

#### 語法

```bash
python3 pipeline.py [選項] <輸入檔案>
```

#### 輸入檔案類型

- **`.s`**: AMD ISA 組合語言檔案（執行完整 pipeline）
- **`.mlir` / `.gpumlir`**: GPU MLIR 檔案（從 Stage 3 開始）

### 命令選項

| 選項 | 說明 | 預設值 |
|------|------|--------|
| `--chip <架構>` | 目標 GPU 架構 | `gfx950` |
| `--workdir <目錄>` | 中間檔案輸出目錄 | `pipeline_output` |
| `--emit-isa` | 生成 ISA 和 HSACO | 啟用 |
| `--no-emit-isa` | 不生成 ISA 和 HSACO | - |
| `--emit-llvm-ir` | 同時生成 LLVM IR（用於調試） | 未啟用 |
| `--output-prefix <前綴>` | 輸出檔案名稱前綴 | 自動生成 |

### 使用範例

#### 範例 1: 從 .s 檔案生成 HSACO（基本用法）

```bash
cd /home/morhuang/Project-MDR/Track_B/llvm-project/my_test
python3 pipeline.py ../../../kernel_testcases/test_01_vector_add/original.s
```

**輸出檔案** (在 `pipeline_output/` 目錄中):
- `original_rebuilt.amdisamlir` - AMDISA MLIR
- `original_rebuilt.gpumlir` - GPU MLIR
- `original_rebuilt.s` - 重建的 ISA
- `original_rebuilt.o` - Object 檔案
- `original_rebuilt.hsaco` - 可執行檔案 ✓

#### 範例 2: 指定輸出目錄和架構

```bash
python3 pipeline.py \
  --chip gfx900 \
  --workdir test01_output \
  ../../../kernel_testcases/test_01_vector_add/original.s
```

#### 範例 3: 生成 LLVM IR（用於調試）

```bash
python3 pipeline.py \
  --emit-llvm-ir \
  ../../../kernel_testcases/test_02_scalar_ops/original.s
```

**額外輸出**:
- `original_rebuilt_llvm.bc` - LLVM Bitcode
- `original_rebuilt_llvm.ll` - 人類可讀的 LLVM IR

#### 範例 4: 自定義輸出前綴

```bash
python3 pipeline.py \
  --output-prefix my_kernel \
  --workdir my_test \
  ../../../kernel_testcases/test_03_memory_ops/original.s
```

**輸出檔案**:
- `my_kernel.amdisamlir`
- `my_kernel.gpumlir`
- `my_kernel.hsaco` ✓

#### 範例 5: 從 GPU MLIR 開始（跳過 Stage 1-2）

```bash
# 先從 .s 生成 GPU MLIR
python3 pipeline.py --no-emit-isa test.s

# 再從 GPU MLIR 生成 HSACO
python3 pipeline.py pipeline_output/test_rebuilt.gpumlir
```

### 進階功能

#### Metadata 自動修復

Pipeline 會自動：
1. 從原始 ISA 提取 `group_segment_fixed_size`（LDS/Shared Memory 大小）
2. 注入到 GPU MLIR 的 module attributes
3. 修復重建後的 ISA metadata（VGPR/SGPR counts、kernarg size 等）

這確保了重建的 HSACO 與原始版本功能一致。

#### 處理多個測試案例（批次處理）

```bash
#!/bin/bash
# 批次處理所有測試案例

cd /home/morhuang/Project-MDR/Track_B/llvm-project/my_test

for test_dir in ../../../kernel_testcases/test_*/; do
    test_name=$(basename "$test_dir")
    echo "Processing $test_name..."
    
    python3 pipeline.py \
      --workdir "../../../kernel_testcases/$test_name/pipeline_output" \
      --output-prefix "$test_name" \
      "$test_dir/original.s"
done
```

### 常見問題排除

#### 問題 1: 找不到工具

```bash
# 確認所有必要工具在 PATH 中
which amdisa-translate  # 應該指向你的 LLVM build
which mlir-opt
which llvm-mc
which ld.lld
```

**解決方法**: 將 LLVM build/bin 目錄加入 PATH:
```bash
export PATH="/home/morhuang/llvm-project/build/bin:$PATH"
```

#### 問題 2: Metadata 不正確

Pipeline 會自動修復 metadata，但如果仍有問題：

1. 檢查原始 `.s` 檔案是否包含完整的 `.amdhsa_*` 指令
2. 查看 console 輸出中的 `[Info] Fixed ISA metadata` 訊息
3. 使用 `--emit-llvm-ir` 生成 LLVM IR 進行調試

#### 問題 3: HSACO 無法執行

```bash
# 檢查生成的 HSACO
objdump -h pipeline_output/original_rebuilt.hsaco

# 或使用 universal_hsaco_runner 測試（見下一節）
```

---

## 2. Universal HSACO Runner 工具

### 概述

`universal_hsaco_runner` 是一個通用的 GPU kernel 驗證工具，用於載入 HSACO (HSA Code Object) 檔案並在 AMD GPU 上執行和驗證 kernel。

**工具位置**: `/home/morhuang/Project-MDR/Track_B/kernel_testcases/universal_hsaco_runner`

### 基本使用

#### 語法

```bash
./universal_hsaco_runner <hsaco_path> <kernel_name> <kernel_type> <test_size>
```

#### 參數說明

- `<hsaco_path>`: HSACO 檔案路徑
- `<kernel_name>`: Kernel 函數名稱（C++ mangled name）
- `<kernel_type>`: Kernel 類型（見下表）
- `<test_size>`: 測試資料大小（元素數量）

### 支援的 Kernel 類型

#### Float Kernel 類型

| 類型 | 說明 | 函數簽名 | 測試內容 |
|------|------|----------|----------|
| `float_add` | 向量加法 | `(float* a, float* b, float* c, int n)` | 驗證 C[i] = A[i] + B[i] |
| `float_mul` | 向量乘法 | `(float* a, float* b, float* c, int n)` | 驗證 C[i] = A[i] * B[i] |
| `float_dot` | 向量點積 | `(float* a, float* b, float* partial_sums, int n)` | Shared memory + Reduction |
| `float_saxpy` | SAXPY 運算 | `(float alpha, float* x, float* y, int n)` | 驗證 Y[i] = alpha*X[i] + Y[i] |
| `float_cond` | 條件運算 | `(float* input, float* output, float threshold, int n)` | 條件分支測試 |

#### Int Kernel 類型

| 類型 | 說明 | 函數簽名 |
|------|------|----------|
| `int_scalar` | 純量運算 | `(int* output, int n)` |
| `int_mem` | 記憶體操作 | `(int* input, int* output, int n)` |
| `int_cond` | 條件判斷 | `(int* input, int* output, int n)` |
| `int_loop` | 迴圈 | `(int* output, int n)` |
| `int_shared` | 共享記憶體 | `(int* input, int* output, int n)` |

### Test 01-07 使用範例

#### Test 01: Vector Add（向量加法）

```bash
cd /home/morhuang/Project-MDR/Track_B/kernel_testcases/test_01_vector_add

# 使用 pipeline.py 生成 HSACO（如果還沒有）
python3 ../../llvm-project/my_test/pipeline.py \
  --workdir pipeline_output \
  original.s

# 測試執行
../universal_hsaco_runner \
  pipeline_output/original_rebuilt.hsaco \
  _Z9vectorAddPKfS0_Pfi \
  float_add \
  1024
```

**預期輸出**:
```
✅ PASS: All 1024 elements correct
Sample results:
  [0] 0 + 0 = 0
  [1] 1 + 2 = 3
  [2] 2 + 4 = 6
```

#### Test 02: Scalar Ops（純量運算）

```bash
cd /home/morhuang/Project-MDR/Track_B/kernel_testcases/test_02_scalar_ops

# 生成 HSACO
python3 ../../llvm-project/my_test/pipeline.py original.s

# 測試執行
../universal_hsaco_runner \
  pipeline_output/original_rebuilt.hsaco \
  _Z9scalarOpsPlll \
  int_scalar \
  256
```

#### Test 03: Memory Ops（記憶體操作）

```bash
cd /home/morhuang/Project-MDR/Track_B/kernel_testcases/test_03_memory_ops

# 生成 HSACO
python3 ../../llvm-project/my_test/pipeline.py original.s

# 測試執行
../universal_hsaco_runner \
  pipeline_output/original_rebuilt.hsaco \
  _Z9memoryOpsPKlPlll \
  int_mem \
  512
```

#### Test 04: Conditional（條件判斷）

```bash
cd /home/morhuang/Project-MDR/Track_B/kernel_testcases/test_04_conditional

# 生成 HSACO
python3 ../../llvm-project/my_test/pipeline.py original.s

# 測試執行
../universal_hsaco_runner \
  pipeline_output/original_rebuilt.hsaco \
  _Z11conditionalPKlPlll \
  int_cond \
  512
```

#### Test 05: Loop（迴圈）

```bash
cd /home/morhuang/Project-MDR/Track_B/kernel_testcases/test_05_loop

# 生成 HSACO
python3 ../../llvm-project/my_test/pipeline.py original.s

# 測試執行
../universal_hsaco_runner \
  pipeline_output/original_rebuilt.hsaco \
  _Z4loopPlll \
  int_loop \
  256
```

#### Test 06: Shared Memory（共享記憶體）

```bash
cd /home/morhuang/Project-MDR/Track_B/kernel_testcases/test_06_shared_memory

# 生成 HSACO
python3 ../../llvm-project/my_test/pipeline.py original.s

# 測試執行
../universal_hsaco_runner \
  pipeline_output/original_rebuilt.hsaco \
  _Z12sharedMemoryPKlPlll \
  int_shared \
  512
```

#### Test 07: Multi-Kernels（多 Kernel）

Test 07 包含 5 個不同的 kernel，可以使用自動化測試腳本：

```bash
cd /home/morhuang/Project-MDR/Track_B/kernel_testcases/test_07_multi_kernels

# 自動測試所有 5 個 kernel（推薦）
./test_all_kernels.sh
```

**或者手動測試個別 kernel**:

```bash
# 生成 HSACO（如果還沒有）
python3 ../../llvm-project/my_test/pipeline.py original.s

# 測試 vectorAdd
../universal_hsaco_runner \
  pipeline_output/original_rebuilt.hsaco \
  _Z9vectorAddPKfS0_Pfi \
  float_add \
  1024

# 測試 vectorMul
../universal_hsaco_runner \
  pipeline_output/original_rebuilt.hsaco \
  _Z9vectorMulPKfS0_Pfi \
  float_mul \
  1024

# 測試 vectorDot（使用 shared memory）
../universal_hsaco_runner \
  pipeline_output/original_rebuilt.hsaco \
  _Z9vectorDotPKfS0_Pfi \
  float_dot \
  1024

# 測試 saxpy
../universal_hsaco_runner \
  pipeline_output/original_rebuilt.hsaco \
  _Z5saxpyfPKfPfi \
  float_saxpy \
  1024

# 測試 conditionalOps
../universal_hsaco_runner \
  pipeline_output/original_rebuilt.hsaco \
  _Z14conditionalOpsPKfPffi \
  float_cond \
  1024
```

### 一鍵測試腳本

創建一個測試腳本來測試所有案例：

```bash
#!/bin/bash
# test_all.sh - 測試所有 test cases

cd /home/morhuang/Project-MDR/Track_B/kernel_testcases

# Test cases 配置（kernel_name|kernel_type|size）
declare -A tests=(
    ["test_01_vector_add"]="_Z9vectorAddPKfS0_Pfi|float_add|1024"
    ["test_02_scalar_ops"]="_Z9scalarOpsPlll|int_scalar|256"
    ["test_03_memory_ops"]="_Z9memoryOpsPKlPlll|int_mem|512"
    ["test_04_conditional"]="_Z11conditionalPKlPlll|int_cond|512"
    ["test_05_loop"]="_Z4loopPlll|int_loop|256"
    ["test_06_shared_memory"]="_Z12sharedMemoryPKlPlll|int_shared|512"
)

for test_name in "${!tests[@]}"; do
    echo "=== Testing $test_name ==="
    
    IFS='|' read -r kernel_name kernel_type size <<< "${tests[$test_name]}"
    
    ./universal_hsaco_runner \
        "$test_name/pipeline_output/original_rebuilt.hsaco" \
        "$kernel_name" \
        "$kernel_type" \
        "$size"
    
    echo ""
done

# Test 07 使用專用腳本
echo "=== Testing test_07_multi_kernels ==="
cd test_07_multi_kernels
./test_all_kernels.sh
```

### 查看詳細輸出

```bash
# 查看完整的 kernel 執行資訊
./universal_hsaco_runner \
  test.hsaco \
  kernel_name \
  float_add \
  1024 2>&1 | tee test_output.log
```

### 常見問題排除

#### 問題 1: Failed to load module

```
HIP error: hipErrorInvalidImage
```

**可能原因**:
- HSACO 檔案損壞
- 架構不匹配（gfx900 vs gfx950）

**解決方法**:
```bash
# 重新生成 HSACO，確認使用正確的 --chip
python3 pipeline.py --chip gfx950 original.s

# 檢查系統 GPU 架構
rocminfo | grep "Name:"
```

#### 問題 2: Failed to get function

```
Failed to get function '_Z9vectorAddPKfS0_Pfi'
```

**解決方法**:
```bash
# 查看 HSACO 中包含的符號
readelf -s test.hsaco | grep _Z

# 或使用 objdump
objdump -t test.hsaco
```

#### 問題 3: 執行結果不正確

```
❌ FAIL: 512 errors found
```

**調試步驟**:
1. 使用 `--emit-llvm-ir` 生成 LLVM IR 檢查邏輯
2. 比較原始 `.s` 和重建的 `.s` 的 metadata
3. 檢查 console 輸出中的 metadata 修復訊息

---

## 完整工作流程範例

### 從零開始：開發 → 測試 → 驗證

```bash
# 1. 編寫 HIP kernel（假設已有 test.hip）
hipcc -S -o test.s test.hip --offload-arch=gfx950

# 2. 使用 pipeline.py 處理
cd /home/morhuang/Project-MDR/Track_B/llvm-project/my_test
python3 pipeline.py \
  --emit-llvm-ir \
  --workdir ../../my_kernel_output \
  test.s

# 3. 驗證生成的 HSACO
cd ../../kernel_testcases
./universal_hsaco_runner \
  ../my_kernel_output/test_rebuilt.hsaco \
  _Z10myKernelPfi \
  float_add \
  1024

# 4. 如果測試失敗，檢查 LLVM IR
cat ../my_kernel_output/test_rebuilt_llvm.ll
```

### 批次處理和驗證

```bash
#!/bin/bash
# 完整測試流程

TESTS_DIR="/home/morhuang/Project-MDR/Track_B/kernel_testcases"
PIPELINE="/home/morhuang/Project-MDR/Track_B/llvm-project/my_test/pipeline.py"

cd $TESTS_DIR

for test_dir in test_*/; do
    test_name=$(basename "$test_dir")
    echo "========================================
    echo "Processing: $test_name"
    echo "========================================"
    
    # 生成 HSACO
    python3 $PIPELINE \
      --workdir "$test_dir/pipeline_output" \
      --output-prefix "rebuilt" \
      "$test_dir/original.s"
    
    # 根據測試類型選擇驗證方式
    if [ "$test_name" == "test_07_multi_kernels" ]; then
        cd "$test_dir"
        ./test_all_kernels.sh
        cd ..
    fi
    
    echo ""
done
```

---

## 延伸閱讀

- **完整 pipeline.py 文檔**: `/home/morhuang/Project-MDR/Track_B/llvm-project/my_test/pipeline.py`
- **universal_hsaco_runner 文檔**: `/home/morhuang/Project-MDR/Track_B/kernel_testcases/README.md`
- **Test 07 詳細報告**: `/home/morhuang/Project-MDR/Track_B/kernel_testcases/test_07_multi_kernels/TEST_REPORT.md`

---

## 系統需求

- **AMD GPU**: gfx900 或更新
- **ROCm**: 5.0 或更新
- **LLVM/MLIR**: 自定義 build（包含 AMDISA dialect）
- **Python**: 3.7+
- **工具**: hipcc, mlir-opt, llvm-mc, ld.lld, llvm-dis

---

**最後更新**: 2025-12-23
