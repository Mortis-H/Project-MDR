# File Index - my_test Directory

## 📁 目錄結構（清理後）

```
my_test/
├── 📄 核心工具
│   ├── pipeline.py                                (主 Pipeline)
│   └── add_register_clobber_v2.py                 (Register Clobber 工具)
│
├── 📄 驗證與測試
│   ├── validate_with_clobber.sh                   (Register Clobber 驗證腳本)
│   ├── test_pipeline_correctness.py               (Pipeline 正確性測試)
│   └── test_universal_mode.sh                     (通用模式測試)
│
├── 📄 文檔
│   ├── README.md                                  (Pipeline 使用說明)
│   ├── REGISTER_CLOBBER_VALIDATION_REPORT.md     (Register Clobber 驗證報告)
│   └── FILE_INDEX.md                              (本文件)
│
├── 📁 source/                                      (源代碼示例)
├── 📁 hip_kernels/                                 (HIP kernel 示例)
└── 📄 mdr_debugging.s                              (調試用 ISA 文件)
```

## 📚 文件說明

### 🔧 核心工具

#### `pipeline.py`
**用途**：AMDISA Dialect Pipeline 主程序
- ISA → AMDISA MLIR → GPU MLIR → ISA/HSACO 轉換
- 支持 `--trust-llvm-resources` 參數（信任 LLVM 計算資源需求）
- 支持 `--emit-isa` 和 `--emit-llvm-ir` 輸出模式

**用法**：
```bash
# 從 ISA 轉換
python3 pipeline.py original.s --workdir output

# 從 GPU MLIR 轉換（信任 LLVM 資源計算）
python3 pipeline.py kernel.gpumlir --trust-llvm-resources
```

#### `add_register_clobber_v2.py`
**用途**：自動為 GPU MLIR 添加 Register Clobber（Reserve + Release）
- 分析 `llvm.inline_asm` 中的暫存器使用
- 生成 Reserve 聲明（`gpu.func` 開頭）
- 生成 Release 聲明（`gpu.return` 之前）

**用法**：
```bash
python3 add_register_clobber_v2.py input.gpumlir -o output.gpumlir

# Dry-run 模式（只分析，不修改）
python3 add_register_clobber_v2.py input.gpumlir --dry-run
```

**背景**：當 GPU MLIR 包含原始 ISA 的 inline assembly 時，需要告訴 LLVM 哪些暫存器已被使用，這樣 LLVM 才能正確計算資源需求並為新插入的 DSL 分配適當的暫存器。

### ✅ 驗證與測試

#### `validate_with_clobber.sh`
**用途**：完整的 Register Clobber 驗證流程
- 對 `test_01` 到 `test_06` 進行 ISA round-trip 測試
- 自動添加 register clobber
- 比較原始和重建的 metadata
- 執行 HSACO 並驗證結果

**用法**：
```bash
./validate_with_clobber.sh
```

**驗證流程**：
```
ISA → MLIR → +Clobber → ISA (LLVM 計算) → 執行驗證
```

#### `test_pipeline_correctness.py`
**用途**：原有的 Pipeline 正確性測試

#### `test_universal_mode.sh`
**用途**：原有的通用模式測試

### 📖 文檔

#### `README.md`
- Pipeline 的完整使用說明
- 支持的輸入格式（`.s`, `.mlir`, `.gpumlir`）
- 輸出選項說明
- 示例用法

#### `REGISTER_CLOBBER_VALIDATION_REPORT.md`
- **Register Clobber 驗證報告**（最新）
- 實驗背景與動機
- 技術實現細節
- 6 個測試案例的完整驗證結果
- 對比分析（原始方法 vs 新方法）
- 未來方向（DSL 插入）

#### `FILE_INDEX.md`
- 本文件，提供目錄結構和文件說明

## 🎯 核心成果

### Register Clobber 機制

通過 Register Clobber（Reserve + Release），我們證明了：

✅ **LLVM 能夠正確計算 GPU kernel 的資源需求**
- 無需手動修復 metadata
- 為 DSL 插入鋪平道路

### 驗證結果

| 測試 | VGPR | SGPR | Kernarg | 執行 |
|------|------|------|---------|------|
| test_01_vector_add | 8 ✅ | 14 ✅ | 288 ✅ | PASS ✅ |
| test_02_scalar_ops | 6 ✅ | 11 ✅ | 272 ✅ | PASS ✅ |
| test_03_memory_ops | 4 ✅ | 11 ✅ | 280 ✅ | PASS ✅ |
| test_04_conditional | 6 ✅ | 12 ✅ | 280 ✅ | PASS ✅ |
| test_05_loop | 3 ✅ | 11 ✅ | 272 ✅ | PASS ✅ |
| test_06_shared_memory | 6 ✅ | 15 ✅ | 280 ✅ | PASS ✅ |

## 🚀 快速開始

### 1. ISA Round-trip（不帶 DSL）

```bash
# 使用 fix_isa_metadata（傳統方法）
python3 pipeline.py original.s

# 使用 register clobber + 信任 LLVM（新方法）
python3 pipeline.py original.s --no-emit-isa  # 生成 GPU MLIR
python3 add_register_clobber_v2.py output.gpumlir -o output_clobber.gpumlir
python3 pipeline.py output_clobber.gpumlir --trust-llvm-resources
```

### 2. 驗證所有測試

```bash
./validate_with_clobber.sh
```

### 3. 查看驗證報告

```bash
cat REGISTER_CLOBBER_VALIDATION_REPORT.md
```

## 📌 相關分支

- **Current Branch**: `feature/register-clobber-trust-llvm`
- **Commit**: `9b7d5b807`

---

**Last Updated**: 2025-12-23  
**Status**: ✅ Production Ready

