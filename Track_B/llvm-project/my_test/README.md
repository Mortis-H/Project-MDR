Pipeline for ISA ↔ MLIR round-trip
====================================

A complete toolchain for translating AMD GPU ISA assembly to MLIR and back, with correctness validation.

**Key Features:**
- Raising ISA to MLIR using custom AMDISA dialect
- End-to-end correctness testing (validates rebuilt kernels produce identical results)
- LLVM IR generation for debugging
- Support for MDR-optimized ISA

---

## Part A: Using pipeline.py to translate ISA assembly

**Prerequisites:** Navigate to the test directory first.

```bash
cd Project-MDR/Track_B/llvm-project/my_test
```

### 1. Translate ISA to MLIR and rebuild HSACO (default behavior).

```bash
# Input: .s (AMD ISA assembly)
# Output: .amdisamlir → .gpumlir → rebuilt .s → rebuilt .hsaco
./pipeline.py source/kernel_isa.s
```

**Note:** `--emit-isa` is enabled by default. Use `--no-emit-isa` to disable HSACO generation.

Generated files in `pipeline_output/` directory:
- `kernel_isa_rebuilt.amdisamlir` - AMDISA MLIR (ISA parsed into AMDISA dialect)
- `kernel_isa_rebuilt.gpumlir` - GPU MLIR (lowered to GPU dialect with inline asm)
- `kernel_isa_rebuilt_binary_isa.mlir` - MLIR with gpu.binary attribute
- `kernel_isa_rebuilt.s` - Rebuilt ISA assembly (metadata fixed)
- `kernel_isa_rebuilt.o` - Object file (assembled)
- `kernel_isa_rebuilt.hsaco` - Final code object (linked)

### 2. Generate LLVM IR for debugging.

```bash
# Add --emit-llvm-ir to generate LLVM bitcode and human-readable IR
./pipeline.py source/kernel_isa.s --emit-llvm-ir
```

Additional files generated:
- `kernel_isa_rebuilt_binary_llvm.mlir` - MLIR with gpu.binary attribute (LLVM format)
- `kernel_isa_rebuilt_llvm.bc` - LLVM bitcode (binary format)
- `kernel_isa_rebuilt_llvm.ll` - LLVM IR (human-readable text)

**Use case:** Debug the MLIR → LLVM lowering process, inspect inline assembly blocks, or analyze register allocation before final ISA generation.

### 3. Use custom output prefix and directory.

```bash
# Specify custom output prefix and working directory
./pipeline.py source/kernel_isa.s \
    --chip=gfx950 \
    --emit-llvm-ir \
    --output-prefix my_kernel \
    --workdir my_output/
```

### 4.Pipeline architecture.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MLIR Pipeline Architecture                        │
└─────────────────────────────────────────────────────────────────────┘

Input: kernel.s (AMD ISA assembly)
  │
  ├─> [Stage 1] amdisa-translate --to-amdisa
  │   └─> kernel.amdisamlir (AMDISA dialect)
  │       - Custom attributes: amdisa.sgpr_count, amdisa.kernarg_segment_size
  │       - Contains: amdisa.func with inline ISA instructions
  │
  ├─> [Stage 2] amdisa-translate --to-gpu (with -amdisa-lower-to-gpu-inline-asm)
  │   └─> kernel.gpumlir (GPU dialect with inline asm)
  │       - Lowered: amdisa.func → gpu.func
  │       - Metadata transferred: amdisa.* attributes copied to gpu.func
  │       - Contains: llvm.inline_asm (ISA as opaque string)
  │
  ├─> [Stage 3] mlir-opt optimization pipeline
  │   │
  │   ├─> builtin.module(gpu-kernel-outlining)
  │   │   └─> Extracts gpu.func into separate gpu.module
  │   │
  │   ├─> builtin.module(rocdl-attach-target{chip=gfx950})
  │   │   └─> Attaches target information for ROCm compilation
  │   │
  │   ├─> gpu.module(strip-debuginfo, convert-gpu-to-rocdl{...})
  │   │   └─> Converts GPU dialect to ROCDL dialect (LLVM for AMD GPUs)
  │   │       ⚠️  WARNING: Custom amdisa.* attributes are LOST here
  │   │
  │   ├─> gpu-to-llvm, reconcile-unrealized-casts
  │   │   └─> Lowers remaining GPU ops to LLVM
  │   │
  │   └─> gpu-module-to-binary{format=isa}
  │       └─> kernel_binary_isa.mlir (with gpu.binary attribute)
  │           - Contains: ISA text embedded in MLIR
  │           - Problem: Metadata is incorrect (attributes were lost)
  │
  ├─> [Stage 3.5] fix_isa_metadata() - Python post-processing
  │   │
  │   ├─> Read correct metadata from kernel.gpumlir
  │   │   └─> Extract: amdisa.sgpr_count, amdisa.vgpr_count, etc.
  │   │
  │   ├─> Parse YAML metadata in generated ISA
  │   │
  │   ├─> Overwrite incorrect values with correct ones
  │   │   ├─> .kernarg_segment_size: 24 → 288
  │   │   ├─> .sgpr_count: 14 → 20
  │   │   ├─> .vgpr_count: 4 → 8
  │   │   └─> .args: [... add all hidden parameters ...]
  │   │
  │   └─> Fix .amdhsa_* assembler directives
  │       ├─> .amdhsa_kernarg_size 24 → 288
  │       ├─> .amdhsa_next_free_sgpr 14 → 20
  │       └─> .amdhsa_next_free_vgpr 4 → 8
  │
  └─> Output: kernel_rebuilt.s (corrected ISA)
      │
      ├─> [Stage 4] llvm-mc -filetype=obj
      │   └─> kernel_rebuilt.o (ELF object file)
      │
      └─> [Stage 5] ld.lld -shared
          └─> kernel_rebuilt.hsaco (final code object)

┌─────────────────────────────────────────────────────────────────────┐
│                   Optional: LLVM IR Generation                       │
└─────────────────────────────────────────────────────────────────────┘

Input: kernel.gpumlir
  │
  └─> mlir-opt ... -gpu-module-to-binary{format=llvm}
      └─> kernel_binary_llvm.mlir (with gpu.binary containing LLVM bitcode)
          │
          └─> Extract bitcode → kernel_llvm.bc
              │
              └─> llvm-dis → kernel_llvm.ll (human-readable LLVM IR)
                  - Shows: inline assembly blocks
                  - Debug: MLIR → LLVM lowering
```

---

## Part B: End-to-end correctness testing

`test_pipeline_correctness.py` validates that the rebuilt kernel produces identical output.

**Prerequisites:** Navigate to the test directory first.

```bash
cd Project-MDR/Track_B/llvm-project/my_test
```

### 1. Run basic correctness test (standard compilation).

```bash
# Uses Track_A/e2e_test/vec_add_kernel.hip and main.cpp by default
# Compiles kernel with hipcc, extracts ISA, then rebuilds with pipeline
./test_pipeline_correctness.py
```

**Process:**
- Compiles `.hip` kernel with `hipcc` → generates `.hsaco` and `.s` (ISA)
- Rebuilds `.s` through MLIR pipeline → generates new `.hsaco`
- Compares execution results

### 2. Run with MDR-optimized ISA (assemble mode).

```bash
# Use pre-optimized ISA from MDR (skips hipcc compilation)
./test_pipeline_correctness.py --use-mdr-isa mdr_debugging.s
```

**MDR Assemble mode:**
- Uses `clang -cc1as` to assemble ISA
- Uses `ld.lld` to link code object
- Uses `clang-offload-bundler` to create fat binary
- Mirrors Track_A/e2e_test/Makefile's `assemble` target

### 3. Test with custom kernel and host.

```bash
./test_pipeline_correctness.py \
    --kernel /path/to/your_kernel.hip \
    --host /path/to/your_host.cpp \
    --arch gfx942
```

### 4. What the test does (under the hood):

**Test workflow (5 stages):**

```
┌─────────────────────────────────────────────────────────┐
│ Stage 1: Compile original kernel                        │
│   - hipcc: vec_add_kernel.hip → .hsaco                 │
│   - Extract ISA assembly (.s file)                      │
│   - Compile host program (main.cpp)                     │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ Stage 2: Run original version                           │
│   - Execute: host program + original .hsaco             │
│   - Capture output → Result A                           │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ Stage 3: Rebuild kernel with MLIR pipeline              │
│   - Input: original .s (ISA)                            │
│   - amdisa-translate: .s → .amdisamlir → .gpumlir      │
│   - mlir-opt + fix_isa_metadata: .gpumlir → new .s     │
│   - llvm-mc + ld.lld: new .s → new .hsaco              │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ Stage 4: Run rebuilt version                            │
│   - Execute: host program + rebuilt .hsaco              │
│   - Capture output → Result B                           │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ Stage 5: Compare results                                │
│   - Compare A == B ?                                    │
│   - If identical → ✓ Test passed                        │
│   - If different → ✗ Test failed (show diff)            │
└─────────────────────────────────────────────────────────┘
```

### 5. Output directory structure.

After running the test, the output directory contains:

```
output/
├── [Original files from hipcc]
│   ├── vec_add_kernel-hip-amdgcn-amd-amdhsa-gfx950.out
│   │   └─→ Original code object generated by hipcc
│   ├── vec_add_kernel-hip-amdgcn-amd-amdhsa-gfx950.s
│   │   └─→ Original ISA assembly (extracted from hipcc --save-temps)
│   ├── vec_add_kernel-hip-amdgcn-amd-amdhsa-gfx950.bc
│   │   └─→ LLVM bitcode (intermediate file from hipcc)
│   ├── vec_add_kernel-hip-amdgcn-amd-amdhsa-gfx950.o
│   │   └─→ Object file (intermediate file from hipcc)
│   └── vec_add_kernel-hip-amdgcn-amd-amdhsa-gfx950
│       └─→ Host executable (compiled from main.cpp)
│
├── [Symlink for host program]
│   └── vec_add_kernel.hsaco -> vec_add_kernel-hip-amdgcn-amd-amdhsa-gfx950.out
│       └─→ Symlink that host program loads (switches between original/rebuilt)
│
└── pipeline_output/
    └── [Rebuilt files from pipeline]
        ├── vec_add_kernel-gfx950_rebuilt.amdisamlir
        │   └─→ AMDISA MLIR (Stage 1: ISA → AMDISA dialect)
        ├── vec_add_kernel-gfx950_rebuilt.gpumlir
        │   └─→ GPU MLIR (Stage 2: AMDISA → GPU dialect with inline asm)
        ├── vec_add_kernel-gfx950_rebuilt_binary_isa.mlir
        │   └─→ MLIR with gpu.binary (Stage 3: after optimization pipeline)
        ├── vec_add_kernel-gfx950_rebuilt.s
        │   └─→ Rebuilt ISA assembly (Stage 3: extracted and metadata-fixed)
        ├── vec_add_kernel-gfx950_rebuilt.o
        │   └─→ Rebuilt object file (Stage 4: assembled by llvm-mc)
        ├── vec_add_kernel-gfx950_rebuilt.hsaco
        │   └─→ Rebuilt code object (Stage 5: linked by ld.lld)
        │
        └── [Optional: LLVM IR files (with --emit-llvm-ir)]
            ├── vec_add_kernel-gfx950_rebuilt_binary_llvm.mlir
            │   └─→ MLIR with gpu.binary (LLVM bitcode format)
            ├── vec_add_kernel-gfx950_rebuilt_llvm.bc
            │   └─→ LLVM bitcode (binary, for llvm-dis)
            └── vec_add_kernel-gfx950_rebuilt_llvm.ll
                └─→ LLVM IR (human-readable, shows inline asm blocks)
```

**Key Points:**
- All actual files contain full architecture information in their names
- The symlink `vec_add_kernel.hsaco` provides a stable interface for the host program
- Original and rebuilt files are kept separate (no overwriting)
- Rebuilt files have `_rebuilt` suffix to distinguish from originals
- LLVM IR files (`.bc`, `.ll`) are only generated with `--emit-llvm-ir` flag

<!-- ## Part C: Technical Details

### 1. Metadata preservation in the pipeline

**Challenge:** Standard MLIR passes (`convert-gpu-to-rocdl`, `gpu-module-to-binary`) don't recognize custom `amdisa.*` attributes, leading to incorrect metadata in the generated ISA.

**Solution (3-stage fix):**

1. **AMDISAAsmParser.cpp** - Initial metadata capture
   - Parses ISA YAML metadata and stores as `amdisa.*` module attributes
   - Includes **all** kernel arguments (including `hidden_*` parameters)
   - Stores `sgpr_count`, `vgpr_count`, `agpr_count`, `kernarg_segment_size`

2. **LowerToGPUInlineAsm.cpp** - Metadata propagation
   - Transfers `amdisa.*` attributes from module to `gpu.func` operation
   - Makes metadata accessible for later passes

3. **pipeline.py:fix_isa_metadata()** - Post-processing correction
   - Reads correct metadata from GPU MLIR (which preserves `amdisa.*` attributes)
   - Overwrites incorrect values in generated ISA's YAML metadata section
   - Fixes `.amdhsa_kernarg_size`, `.amdhsa_next_free_vgpr`, `.amdhsa_next_free_sgpr` directives

**Why this is critical:**
- Incorrect `.kernarg_segment_size` → `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`
- Incorrect `.sgpr_count` / `.vgpr_count` → register allocation errors
- Missing hidden parameters → runtime cannot pass required metadata to kernel

### 2. Hidden parameters and ABI compliance

**AMD GPU ABI requires hidden parameters:**
```yaml
.args:
  - .offset: 0    .size: 8    .value_kind: global_buffer  # Your arg: A
  - .offset: 8    .size: 8    .value_kind: global_buffer  # Your arg: B
  - .offset: 16   .size: 8    .value_kind: global_buffer  # Your arg: C
  # Hidden parameters (automatically passed by HIP runtime):
  - .offset: 24   .size: 8    .value_kind: hidden_global_offset_x
  - .offset: 32   .size: 8    .value_kind: hidden_global_offset_y
  - .offset: 40   .size: 8    .value_kind: hidden_global_offset_z
  - .offset: 48   .size: 8    .value_kind: hidden_printf_buffer
  - .offset: 56   .size: 8    .value_kind: hidden_none
  - .offset: 64   .size: 8    .value_kind: hidden_none
  - .offset: 72   .size: 8    .value_kind: hidden_multigrid_sync_arg
  - .offset: 80   .size: 8    .value_kind: hidden_block_count_x
  - .offset: 88   .size: 8    .value_kind: hidden_block_count_y
  - .offset: 96   .size: 8    .value_kind: hidden_block_count_z
  - .offset: 104  .size: 8    .value_kind: hidden_group_size_x
  - .offset: 112  .size: 8    .value_kind: hidden_hostcall_buffer
```

**Result:**
- `.kernarg_segment_size: 288` (not 24!)
- Runtime needs this space to pass all parameters
- Kernel code may access these hidden parameters (e.g., at offset 0x70 for `hidden_hostcall_buffer`)

### 3. SGPR allocation: `.sgpr_count` vs `.amdhsa_user_sgpr_count`

| Field | Meaning | Example Value |
|-------|---------|---------------|
| `.sgpr_count` | **Total SGPRs used** by kernel (User + System) | 20 |
| `.amdhsa_user_sgpr_count` | **User SGPRs** initialized by Command Processor | 2 |

**Breakdown:**
- **User SGPRs (s[0:1])**: Kernel-wide data (e.g., `kernarg_segment_ptr`)
  - Initialized by Command Processor (CP) before kernel launch
  - Count specified in `.amdhsa_user_sgpr_count`
- **System SGPRs (s[2:?])**: Wavefront-specific data (e.g., `workgroup_id_x`)
  - Initialized by Asynchronous Compute Engine (ACE) / Shader Processor Input (SPI)
  - Not counted in `.amdhsa_user_sgpr_count`
- **Temp SGPRs**: Scratch registers for kernel logic
  - Allocated dynamically during execution

**Why both are needed:**
- `.sgpr_count` → tells hardware how many SGPRs to allocate
- `.amdhsa_user_sgpr_count` → tells CP how many SGPRs to initialize

### 4. Debugging with LLVM IR

Use `--emit-llvm-ir` to inspect the MLIR → LLVM lowering:

```llvm
; Example: vec_add_kernel-gfx950_rebuilt_llvm.ll

define protected amdgpu_kernel void @vec_add_kernel(
    ptr addrspace(1) %arg0,           ; A
    ptr addrspace(1) %arg1,           ; B
    ptr addrspace(1) %arg2            ; C
) {
  ; Inline assembly block (black box for LLVM optimizer)
  call void asm sideeffect "
    s_load_dwordx2 s[0:1], s[0:1], 0x0
    v_mov_b32_e32 v0, s0
    ...
  ", ""()
  
  ret void
}
```

**Key observations:**
- Inline assembly is treated as opaque by LLVM optimizer
- Register allocation happens inside the asm block
- LLVM doesn't see or modify the actual GPU instructions
- This is why metadata must be preserved manually

---

## Part D: Troubleshooting

### Common Issues

**1. `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`**
- **Cause:** Incorrect `.kernarg_segment_size` in ISA metadata
- **Fix:** Ensure all hidden parameters are included in `.args` list
- **Verification:** Check rebuilt `.s` file for correct YAML metadata

**2. Segmentation fault (SIGSEGV)**
- **Cause:** Incorrect register counts (`.sgpr_count`, `.vgpr_count`)
- **Fix:** Verify `fix_isa_metadata()` correctly overwrites `.amdhsa_*` directives
- **Verification:** Compare original and rebuilt `.s` files

**3. `clang-offload-bundler: Too many levels of symbolic links`**
- **Cause:** Symlink loop (`.hsaco` pointing to itself)
- **Fix:** Already handled in `test_pipeline_correctness.py`
- **Prevention:** Clean output directory before each run

**4. `Required tool 'xxx' not found in PATH`**
- **Cause:** LLVM tools not in PATH
- **Fix:** `source ~/.bashrc` or manually add LLVM bin directory to PATH
- **Required tools:** `amdisa-translate`, `mlir-opt`, `llvm-mc`, `ld.lld`, `clang-offload-bundler`, `llvm-dis` -->
