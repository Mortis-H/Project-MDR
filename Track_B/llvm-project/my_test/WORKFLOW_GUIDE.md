# Track_B Workflow Guide

## 📋 概述

Track_B 提供了兩種 workflow，取決於您的使用場景：

1. **Pure Round-trip**：ISA → MLIR → ISA（沒有修改）
2. **DSL Injection**：ISA → MLIR → 插入 DSL → ISA（未來功能）

---

## 🔄 Workflow 1: Pure Round-trip（推薦使用新方法）

### 方法 A：傳統方法（使用 fix_isa_metadata）

```bash
# 一步完成：ISA → MLIR → ISA
python3 pipeline.py original.s
```

**流程圖**：
```
original.s
    ↓ [amdisa-translate -emit=mlir]
AMDISA MLIR (.amdisamlir)
    ↓ [amdisa-translate -emit=gpu]
GPU MLIR (.gpumlir)
    ↓ [mlir-opt + passes]
ISA (.s)
    ↓ [fix_isa_metadata()] ← 從 GPU MLIR 提取並覆蓋 metadata
    ↓ [llvm-mc + ld.lld]
HSACO (.hsaco)
```

**特點**：
- ✅ 簡單，一個命令搞定
- ✅ 適合純 round-trip
- ⚠️ 對 DSL 插入不友好（metadata 會不正確）

---

### 方法 B：新方法（Register Clobber + Trust LLVM）✨

```bash
# Step 1: ISA → GPU MLIR（不生成 ISA）
python3 pipeline.py original.s --no-emit-isa

# Step 2: 添加 Register Clobber
python3 add_register_clobber_v2.py output.gpumlir -o output_clobber.gpumlir

# Step 3: GPU MLIR → ISA（信任 LLVM 計算）
python3 pipeline.py output_clobber.gpumlir --trust-llvm-resources
```

**流程圖**：
```
original.s
    ↓ [amdisa-translate -emit=mlir]
AMDISA MLIR
    ↓ [amdisa-translate -emit=gpu]
GPU MLIR
    ↓ [add_register_clobber_v2.py] ← 添加 Reserve & Release
GPU MLIR (with Clobber)
    ↓ [mlir-opt + passes]
ISA (.s) ← LLVM 自動計算正確的 metadata
    ↓ [llvm-mc + ld.lld]
HSACO (.hsaco)
```

**Register Clobber 示例**：
```mlir
gpu.func @kernel(...) kernel {
  // ===== Reserve =====
  %vgpr_reserved = llvm.inline_asm ... "={v[0:7]}" ...
  %sgpr_reserved = llvm.inline_asm ... "={s[0:7]}" ...
  
  // 原始 ISA 的 inline_asm
  llvm.inline_asm "v_add_f32 v2, v6, v7" ...
  
  // ===== Release =====
  llvm.inline_asm ... "{v[0:7]}" %vgpr_reserved ...
  llvm.inline_asm ... "{s[0:7]}" %sgpr_reserved ...
  
  gpu.return
}
```

**特點**：
- ✅ LLVM 自動計算 metadata（更可靠）
- ✅ 為 DSL 插入鋪平道路
- ✅ 6/6 測試全部通過，metadata 完全匹配
- ⚠️ 需要額外步驟（但可以腳本化）

---

## 🎨 Workflow 2: DSL Injection（未來擴展）

### 完整流程

```bash
# Step 1: ISA → GPU MLIR
python3 pipeline.py original.s --no-emit-isa

# Step 2: 添加 Register Clobber
python3 add_register_clobber_v2.py output.gpumlir -o output_clobber.gpumlir

# Step 3: 手動編輯或腳本插入 DSL
# 在 clobber 區域之後，原始 ISA 之間插入高層 MLIR 代碼

# Step 4: GPU MLIR (with DSL) → ISA
python3 pipeline.py output_with_dsl.gpumlir --trust-llvm-resources
```

**示例：插入 printf**
```mlir
gpu.func @kernel(...) kernel {
  // ===== Reserve 原始 ISA 使用的暫存器 =====
  %vgpr_reserved = llvm.inline_asm ... "={v[0:7]}" ...
  
  // ===== 原始 kernel 邏輯（inline_asm）=====
  llvm.inline_asm "v_add_f32 v2, v6, v7" ...
  
  // ===== 插入 DSL（LLVM 會自動分配 v8+ 暫存器）=====
  %tid = gpu.thread_id x
  gpu.printf "Thread %d: result = %f\n", %tid, %result : index, f32
  
  // ===== 繼續原始邏輯 =====
  llvm.inline_asm "global_store_dword ..." ...
  
  // ===== Release =====
  llvm.inline_asm ... "{v[0:7]}" %vgpr_reserved ...
  
  gpu.return
}
```

**LLVM 自動處理**：
- 為 `gpu.printf` 分配 v8+ 暫存器
- 計算總 VGPR 需求（例如：8 原始 + 5 DSL = 13）
- 生成正確的 metadata

---

## 🛠️ 工具快速參考

### pipeline.py

**基本用法**：
```bash
# 從 ISA 開始
python3 pipeline.py input.s [OPTIONS]

# 從 GPU MLIR 開始
python3 pipeline.py input.gpumlir [OPTIONS]
```

**常用選項**：
```bash
--workdir DIR              # 指定工作目錄（默認：pipeline_output）
--chip CHIP                # 指定 GPU 架構（默認：gfx950）
--output-prefix PREFIX     # 輸出文件前綴

--emit-isa                 # 生成 ISA 和 HSACO（默認開啟）
--no-emit-isa              # 只生成 GPU MLIR，不生成 ISA
--emit-llvm-ir             # 額外生成 LLVM IR

--trust-llvm-resources     # 信任 LLVM 計算資源（跳過 metadata 修復）
                           # ⚠️ 需要 GPU MLIR 包含 register clobber
```

### add_register_clobber_v2.py

**基本用法**：
```bash
python3 add_register_clobber_v2.py input.gpumlir -o output.gpumlir
```

**功能**：
- 自動分析 `llvm.inline_asm` 的暫存器使用
- 生成 Reserve 聲明（在 `gpu.func` 開頭）
- 生成 Release 聲明（在 `gpu.return` 之前）

**選項**：
```bash
--dry-run                  # 只分析，不修改文件
-o, --output FILE          # 指定輸出文件（默認覆蓋輸入）
```

### validate_with_clobber.sh

**用途**：完整驗證 Register Clobber 機制

```bash
./validate_with_clobber.sh
```

**驗證內容**：
- ISA → MLIR → +Clobber → ISA round-trip
- Metadata 比對（VGPR, SGPR, kernarg）
- HSACO 執行測試

---

## 📊 兩種方法對比

| 特性 | 方法 A (fix_isa_metadata) | 方法 B (Register Clobber) |
|-----|-------------------------|-------------------------|
| **命令數量** | 1 | 3 |
| **適用場景** | Pure round-trip | Round-trip + DSL |
| **Metadata 來源** | 從原始 ISA 提取 | LLVM 自動計算 |
| **DSL 插入** | ❌ 不支持 | ✅ 支持 |
| **可靠性** | 手動維護邏輯 | LLVM 保證正確性 |
| **未來擴展** | 受限 | 靈活 |

---

## 🎯 推薦使用場景

### 何時使用方法 A（傳統）

- ✅ 快速測試 ISA round-trip
- ✅ 不需要修改 kernel
- ✅ 簡單腳本，一鍵完成

### 何時使用方法 B（新方法）

- ✅ 需要插入 DSL（printf, assert 等）
- ✅ 需要最大可靠性（依賴 LLVM）
- ✅ 正在開發新功能
- ✅ 需要理解暫存器分配

---

## 📝 完整示例

### 示例 1：Pure Round-trip（方法 A）

```bash
cd Track_B/llvm-project/my_test

# 一步完成
python3 pipeline.py ../../kernel_testcases/test_01_vector_add/original.s \
    --workdir test_01_output \
    --output-prefix vec_add

# 結果：
# test_01_output/vec_add_rebuilt.s
# test_01_output/vec_add_rebuilt.hsaco
```

### 示例 2：Pure Round-trip（方法 B）

```bash
cd Track_B/llvm-project/my_test

# Step 1: ISA → GPU MLIR
python3 pipeline.py ../../kernel_testcases/test_01_vector_add/original.s \
    --workdir test_01_output \
    --output-prefix vec_add \
    --no-emit-isa

# Step 2: 添加 Clobber
python3 add_register_clobber_v2.py \
    test_01_output/vec_add_rebuilt.gpumlir \
    -o test_01_output/vec_add_clobber.gpumlir

# Step 3: GPU MLIR → ISA（信任 LLVM）
python3 pipeline.py test_01_output/vec_add_clobber.gpumlir \
    --workdir test_01_output \
    --output-prefix vec_add_final \
    --trust-llvm-resources

# 結果：
# test_01_output/vec_add_final.s      ← LLVM 計算的 metadata
# test_01_output/vec_add_final.hsaco
```

### 示例 3：批量驗證

```bash
cd Track_B/llvm-project/my_test

# 驗證所有測試案例（test_01 到 test_06）
./validate_with_clobber.sh

# 查看結果
cat clobber_validation_full.log
```

---

## 🔍 調試技巧

### 檢查 Metadata

```bash
# 原始 ISA
grep -E "(vgpr_count|sgpr_count|kernarg_segment_size)" original.s

# 重建的 ISA
grep -E "(vgpr_count|sgpr_count|kernarg_segment_size)" rebuilt.s

# 比對
diff <(grep vgpr_count original.s) <(grep vgpr_count rebuilt.s)
```

### 檢查暫存器使用

```bash
# Dry-run 分析
python3 add_register_clobber_v2.py kernel.gpumlir --dry-run
# 輸出：VGPR=0-7 (8 registers), SGPR=0-7 (8 registers)
```

### 檢查生成的 Clobber

```bash
# 查看 Reserve 部分
head -20 kernel_clobber.gpumlir

# 查看 Release 部分
tail -10 kernel_clobber.gpumlir | grep -A 5 "Release"
```

---

## 🚀 快速開始

### 選擇 1：快速測試（推薦初學者）

```bash
python3 pipeline.py ../../kernel_testcases/test_01_vector_add/original.s
```

### 選擇 2：完整驗證（推薦開發者）

```bash
./validate_with_clobber.sh
```

### 選擇 3：自定義流程（推薦高級用戶）

```bash
# 1. 生成 GPU MLIR
python3 pipeline.py original.s --no-emit-isa

# 2. 添加 Clobber（可選）
python3 add_register_clobber_v2.py output.gpumlir -o output_clobber.gpumlir

# 3. 手動編輯插入 DSL（未來）
# vim output_clobber.gpumlir

# 4. 生成最終 HSACO
python3 pipeline.py output_clobber.gpumlir --trust-llvm-resources
```

---

## 📚 相關文檔

- `README.md`: Pipeline 詳細使用說明
- `REGISTER_CLOBBER_VALIDATION_REPORT.md`: 驗證報告
- `FILE_INDEX.md`: 文件索引
- `Track_A/TECHNICAL_REPORT.md`: Track_A 的 Register Clobber 參考實現

---

**Last Updated**: 2025-12-23  
**Status**: ✅ Production Ready  
**Branch**: `feature/register-clobber-trust-llvm`

