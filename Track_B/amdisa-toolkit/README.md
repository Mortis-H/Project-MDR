# AMDISA Toolkit

AMDGPU ISA to MLIR Conversion Toolchain (Out-of-tree Project)

## Overview

AMDISA Toolkit is a standalone MLIR toolchain for processing AMD GPU ISA (Instruction Set Architecture). This project provides:

- **AMDISA Dialect**: An MLIR dialect for representing AMD GPU ISA
- **amdisa-translate**: Converts AMD ISA assembly (.s) to MLIR and lowers to GPU dialect
- **pipeline.py**: End-to-end conversion tool from .s files to HSACO
- **test_pipeline_correctness.py**: Testing script for verifying conversion correctness

---

## Quick Start

### Prerequisites

- CMake 3.20+
- Ninja or Make
- C++17 compiler (GCC 11+ or Clang 14+)
- Python 3.8+
- Installed LLVM/MLIR (with AMDGPU backend)

### 1. Prepare LLVM/MLIR

You need a built or installed LLVM/MLIR. If you don't have one:

```bash
# Option A: Use system-installed LLVM
# (Skip to step 2 if already installed)

# Option B: Build LLVM from source
git clone --depth 1 https://github.com/llvm/llvm-project.git

cd llvm-project
cmake -S llvm -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS="mlir;lld" \
  -DLLVM_TARGETS_TO_BUILD="AMDGPU;X86" \
  -DCMAKE_INSTALL_PREFIX=/tmp/llvm-install

ninja -C build install

# Make LLVM/MLIR tools available in PATH
export PATH=/tmp/llvm-install/bin:$PATH
```

### 2. Build AMDISA Toolkit

```bash
cd amdisa-toolkit

# Method 1: Use cmake.sh script
./cmake.sh

# Method 2: Manual configuration
cmake -B build -G Ninja \
  -DMLIR_DIR=/path/to/llvm-install/lib/cmake/mlir \
  -DLLVM_DIR=/path/to/llvm-install/lib/cmake/llvm \
  -DCMAKE_BUILD_TYPE=Release

# Build
ninja -C build 

# (Optional) Build tools explicitly
ninja -C build amdisa-translate

# Make amdisa-translate available in PATH
export PATH=$(pwd)/build/bin:$PATH
```

---

## Usage

### amdisa-translate Tool

`amdisa-translate` is the core conversion tool with two output modes:

#### Mode 1: Output AMDISA MLIR

```bash
# Convert AMD ISA assembly to AMDISA dialect MLIR
./build/bin/amdisa-translate examples/sample_isa/kernel_isa.s -emit=mlir 
```

#### Mode 2: Output GPU MLIR (Auto Lowering)

```bash
# Generate GPU dialect MLIR directly (with built-in lowering pass)
./build/bin/amdisa-translate examples/sample_isa/kernel_isa.s -emit=gpu
```

---

## Examples and Testing

The examples directory contains three main tools and sample files:

### 1. pipeline.py - End-to-End Conversion Tool

**Function:** Complete AMD ISA → HSACO conversion pipeline

```bash
cd examples

# Basic usage: Generate HSACO from .s file
python3 pipeline.py sample_isa/kernel_isa.s --chip=gfx950
```

**Workflow:**

```
.s (AMD ISA assembly)
  ↓ amdisa-translate -emit=mlir
.amdisamlir (AMDISA MLIR)
  ↓ amdisa-translate -emit=gpu
.gpumlir (GPU MLIR)
  ↓ mlir-opt + mlir-translate
.ll (LLVM IR)
  ↓ llvm-mc
.o (Object file)
  ↓ ld.lld
.hsaco (HSA Code Object)
```

**Output Files:**
- `*.amdisamlir` - AMDISA dialect MLIR
- `*.gpumlir` - GPU dialect MLIR
- `*.ll` - LLVM IR
- `*.o` - Object file
- `*.hsaco` - Final HSACO file

### 2. test_pipeline_correctness.py - Correctness Testing

**Function:** Automated pipeline correctness testing

```bash
cd examples

# Full compilation: Uses main.cpp and vec_add_kernel.hip from Track_A by default
python3 test_pipeline_correctness.py

# Assembly from .s file
python3 test_pipeline_correctness.py --use-mdr-isa sample_isa/kernel_isa.s
```

**Test Coverage:**
- ISA → AMDISA MLIR conversion
- AMDISA MLIR → GPU MLIR conversion
- GPU MLIR → LLVM IR conversion
- LLVM IR → HSACO generation
- Intermediate file format validation

### 3. test_universal_mode.sh - Universal Mode Testing

**Function:** Test different types of HIP kernels

```bash
cd examples

# Run all test modes
bash test_universal_mode.sh
```
---

**Last Updated:** 2024-12-23
**Version:** 1.0.0
