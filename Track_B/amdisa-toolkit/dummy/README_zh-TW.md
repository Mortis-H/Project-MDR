# AMDISA Toolkit

AMDGPU ISA 到 MLIR 的轉換工具鏈（Out-of-tree 專案）

## 專案簡介

AMDISA Toolkit 是一個獨立的 MLIR 工具鏈，用於處理 AMD GPU ISA (Instruction Set Architecture)。本專案提供：

- **AMDISA Dialect**：用於表示 AMD GPU ISA 的 MLIR dialect
- **amdisa-translate**：將 AMD ISA assembly (.s) 轉換為 MLIR，並可 lower 為 GPU dialect
- **pipeline.py**：端到端的轉換工具，從 .s 文件生成 HSACO
- **test_pipeline_correctness.py**：用於驗證轉換正確性的測試腳本

---

## 快速開始

### 前置需求

- CMake 3.20+
- Ninja 或 Make
- C++17 編譯器 (GCC 11+ 或 Clang 14+)
- Python 3.8+
- 已安裝的 LLVM/MLIR (包含 AMDGPU 後端)

### 1. 準備 LLVM/MLIR

您需要一個已構建或安裝的 LLVM/MLIR。如果還沒有，可以：

```bash
# 選項 A：使用系統已安裝的 LLVM
# (如果已安裝，跳到步驟 2)

# 選項 B：從源碼構建新的 LLVM
git clone --depth 1 https://github.com/llvm/llvm-project.git

cd llvm-project
cmake -S llvm -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS="mlir" \
  -DLLVM_TARGETS_TO_BUILD="AMDGPU;X86" \
  -DCMAKE_INSTALL_PREFIX=/tmp/llvm-install

ninja -C build install
```

### 2. 構建 AMDISA Toolkit

```bash
cd amdisa-toolkit

# 方式 1：使用 cmake.sh 腳本
./cmake.sh

# 方式 2：手動配置
cmake -B build -G Ninja \
  -DMLIR_DIR=/path/to/llvm-install/lib/cmake/mlir \
  -DLLVM_DIR=/path/to/llvm-install/lib/cmake/llvm \
  -DCMAKE_BUILD_TYPE=Release

# 構建
ninja -C build
```

---

## 使用方法

### amdisa-translate 工具

`amdisa-translate` 是核心轉換工具，支援兩種輸出模式：

#### 模式 1：輸出 AMDISA MLIR

```bash
# 將 AMD ISA assembly 轉換為 AMDISA dialect MLIR
./build/bin/amdisa-translate examples/sample_isa/kernel_isa.s -emit=mlir 
```

#### 模式 2：輸出 GPU MLIR（自動 Lowering）

```bash
# 直接生成 GPU dialect MLIR（內置 lowering pass）
./build/bin/amdisa-translate examples/sample_isa/kernel_isa.s -emit=gpu
```

---

## 範例與測試

examples 目錄包含三個主要工具和範例文件：

### 1. pipeline.py - 端到端轉換工具

**功能：** 完整的 AMD ISA → HSACO 轉換 pipeline

```bash
cd examples

# 基本用法：從 .s 文件生成 HSACO
python3 pipeline.py sample_isa/kernel_isa.s
```

**工作流程：**

```
.s (AMD ISA assembly)
  ↓ amdisa-translate -emit=mlir
.amdisamlir (AMDISA MLIR)
  ↓ amdisa-translate -emit=gpu
.gpumlir (GPU MLIR)
  ↓ mlir-opt + mlir-translate
.ll (LLVM IR)
  ↓ llvm-mc
.o (目標文件)
  ↓ ld.lld
.hsaco (HSA Code Object)
```

**輸出文件：**
- `*.amdisamlir` - AMDISA dialect MLIR
- `*.gpumlir` - GPU dialect MLIR
- `*.ll` - LLVM IR
- `*.o` - 目標文件
- `*.hsaco` - 最終的 HSACO 文件

### 2. test_pipeline_correctness.py - 正確性測試

**功能：** 自動化測試 pipeline 的正確性

```bash
cd examples

# 完整編譯: 預設使用 Track_A 中的 main.cpp 和 vec_add_kernel.hip
python3 test_pipeline_correctness.py

# 從 .s 開始組譯
python3 test_pipeline_correctness.py --use-mdr-isa sample_isa/kernel_isa.s
```

**測試內容：**
- ✅ ISA → AMDISA MLIR 轉換
- ✅ AMDISA MLIR → GPU MLIR 轉換
- ✅ GPU MLIR → LLVM IR 轉換
- ✅ LLVM IR → HSACO 生成
- ✅ 中間文件格式驗證

### 3. test_universal_mode.sh - 通用模式測試

**功能：** 測試不同類型的 hip kernel

```bash
cd examples

# 運行所有測試模式
bash test_universal_mode.sh
```
---

**最後更新：** 2024-12-23
**版本：** 1.0.0

