Pipeline for ISA ↔ MLIR round-trip.
====================================

## Part A: Using pipeline.py to translate ISA assembly

### 1. Translate ISA to MLIR and rebuild HSACO.

```bash
# Input: .s (AMD ISA assembly)
# Output: .amdisamlir → .gpumlir → rebuilt .s → rebuilt .hsaco
./pipeline.py source/kernel_isa.s --chip=gfx950 --emit-isa
```

Generated files in `kernel_isa/` directory:
- `kernel_isa_rebuilt.amdisamlir` - AMDISA MLIR
- `kernel_isa_rebuilt.gpumlir` - GPU MLIR
- `kernel_isa_rebuilt_binary_isa.mlir` - MLIR with gpu.binary attribute
- `kernel_isa_rebuilt.s` - Rebuilt ISA assembly
- `kernel_isa_rebuilt.o` - Object file
- `kernel_isa_rebuilt.hsaco` - Final code object

### 2. Use custom output prefix.

```bash
# Specify custom output prefix
./pipeline.py source/kernel_isa.s \
    --chip=gfx950 \
    --emit-isa \
    --output-prefix my_kernel \
    --workdir output/
```

---

## Part B: End-to-end correctness testing

`test_pipeline_correctness.py` validates that the rebuilt kernel produces identical output.

### 1. Run basic correctness test.

```bash
# Uses Track_A/e2e_test/vec_add_kernel.hip and main.cpp by default
./test_pipeline_correctness.py
```

Default parameters:
- Kernel: `Track_A/e2e_test/vec_add_kernel.hip`
- Host: `Track_A/e2e_test/main.cpp`
- Architecture: `gfx950`
- Workdir: `e2e_test_output/`

### 2. Test with custom kernel and host.

```bash
./test_pipeline_correctness.py \
    --kernel /path/to/your_kernel.hip \
    --host /path/to/your_host.cpp \
    --arch gfx942
```

### 3. What the test does (under the hood):

```
┌─────────────────────────────────────────────────────────┐
│ Step 1: 編譯原始 kernel                                  │
│   - 用 hipcc 編譯 vec_add_kernel.hip → .hsaco          │
│   - 提取 ISA assembly (.s 文件)                         │
│   - 編譯 host 程式 (main.cpp)                           │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ Step 2: 執行原始版本                                     │
│   - 執行 host 程式 + 原始 .hsaco                        │
│   - 記錄輸出 (結果 A)                                   │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ Step 3: 用 pipeline 重建 kernel                         │
│   - 輸入: 原始 .s (ISA)                                 │
│   - amdisa-translate: .s → .amdisamlir → .gpumlir      │
│   - mlir-opt: .gpumlir → 新的 .s → 新的 .hsaco         │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ Step 4: 執行重建版本                                     │
│   - 執行 host 程式 + 重建的 .hsaco                      │
│   - 記錄輸出 (結果 B)                                   │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ Step 5: 比較結果                                        │
│   - 比較 A == B ?                                       │
│   - 如果相同 → ✓ 測試通過                               │
│   - 如果不同 → ✗ 測試失敗 (顯示差異)                    │
└─────────────────────────────────────────────────────────┘
```
