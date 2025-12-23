# Migration Guide: New Unified Pipeline

## 🎉 重大更新

Track_B pipeline 已經完成重大升級：

### ✅ 方法 A 已封死（fix_isa_metadata 資源計數修復）
### ✨ 方法 B 成為唯一默認方式（Register Clobber + Trust LLVM）

---

## 🔄 What Changed?

### Before (舊方法)

需要三步驟：

```bash
# Step 1: ISA → GPU MLIR
python3 pipeline.py original.s --no-emit-isa

# Step 2: 手動添加 Clobber
python3 add_register_clobber_v2.py output.gpumlir -o clobber.gpumlir

# Step 3: GPU MLIR → ISA
python3 pipeline.py clobber.gpumlir --trust-llvm-resources
```

### After (新方法) 🎊

**只需一步！**

```bash
python3 pipeline.py original.s
```

Pipeline 會自動：
1. ✅ 分析原始 ISA 的暫存器使用
2. ✅ 添加 Register Clobber (Reserve + Release)
3. ✅ 信任 LLVM 計算資源需求（VGPR/SGPR）
4. ✅ 只修復必要的 metadata（kernarg, hidden params）

---

## 📋 詳細變更

### 1. Register Clobber 自動化

**原理**：
- 自動分析 `llvm.inline_asm` 中的暫存器使用
- 在 `gpu.func` 開頭插入 **Reserve** 聲明
- 在 `gpu.return` 之前插入 **Release** 聲明

**生成的代碼**：
```mlir
gpu.func @kernel(...) kernel {
  // ===== Register Clobber Reserve =====
  // Auto: Reserve VGPR v[0:7] (8 registers)
  %vgpr_reserved = llvm.inline_asm ... "={v[0:7]}" ...
  // Auto: Reserve SGPR s[0:7] (8 registers)
  %sgpr_reserved = llvm.inline_asm ... "={s[0:7]}" ...
  // ====================================
  
  // ... 原始 ISA 的 inline_asm ...
  
  // ===== Register Clobber Release =====
  // Release VGPR clobber
  llvm.inline_asm ... "{v[0:7]}" %vgpr_reserved ...
  // Release SGPR clobber
  llvm.inline_asm ... "{s[0:7]}" %sgpr_reserved ...
  // ====================================
  
  gpu.return
}
```

### 2. fix_isa_metadata() 變更

**移除**：
- ❌ VGPR count 修復
- ❌ SGPR count 修復
- ❌ AGPR count 修復
- ❌ `.amdhsa_next_free_vgpr` 修復
- ❌ `.amdhsa_next_free_sgpr` 修復

**保留**：
- ✅ kernarg_segment_size 修復
- ✅ group_segment_fixed_size 修復（LDS/Shared Memory）
- ✅ Hidden parameters 修復

**原因**：
- LLVM 通過 register clobber 已經能正確計算資源需求
- 手動覆蓋會干擾 LLVM 的計算
- 為未來的 DSL 插入鋪平道路

### 3. 移除的參數

**`--trust-llvm-resources`** (已移除)
- 原因：現在這是默認且唯一的行為
- 不再需要手動指定

### 4. 整合的功能

**`add_register_clobber_v2.py`** 的功能已整合到 `pipeline.py`：
- `analyze_registers_in_gpumlir()` - 分析暫存器使用
- `add_register_clobber_to_gpumlir()` - 添加 clobber

---

## 🚀 使用示例

### 基本用法（Pure Round-trip）

```bash
# 一步完成所有操作
python3 pipeline.py original.s

# 指定工作目錄和輸出前綴
python3 pipeline.py original.s \
    --workdir my_output \
    --output-prefix my_kernel
```

### 高級用法（未來 DSL 插入）

```bash
# Step 1: 生成 GPU MLIR（包含 clobber）
python3 pipeline.py original.s --no-emit-isa

# Step 2: 手動編輯 GPU MLIR 插入 DSL
vim output.gpumlir
# 在 clobber 之後，原始 ISA 之前/之間插入高層 MLIR 代碼
# 例如：gpu.printf, arith.*, cf.* 等

# Step 3: 生成最終 HSACO
python3 pipeline.py output.gpumlir
```

**LLVM 會自動**：
- 為新插入的 DSL 分配 v8+ 暫存器
- 計算總 VGPR 需求
- 生成正確的 metadata

---

## 📊 驗證結果

### Metadata 對比

| 測試 | VGPR (原始 → 新) | SGPR (原始 → 新) | Kernarg | 執行 |
|------|-----------------|-----------------|---------|------|
| test_01 | 8 → 8 | 14 → 14 | 288 ✅ | PASS ✅ |
| test_02 | 6 → 6 | 11 → 11 | 272 ✅ | PASS ✅ |
| test_03 | 4 → 4 | 11 → 11 | 280 ✅ | PASS ✅ |
| test_04 | 6 → 6 | 12 → 12 | 280 ✅ | PASS ✅ |
| test_05 | 3 → 3 | 11 → 11 | 272 ✅ | PASS ✅ |
| test_06 | 6 → 6 | 15 → 15 | 280 ✅ | PASS ✅ |

**結果**: 6/6 全部通過！Metadata 完全匹配！

---

## 🔍 內部流程

### 新的 Pipeline 流程

```
original.s
    ↓
=== Stage 1: amdisa-translate -emit=mlir ===
    ↓
AMDISA MLIR (.amdisamlir)
    ↓
=== Stage 2: amdisa-translate -emit=gpu ===
    ↓
GPU MLIR (原始，無 clobber)
    ↓
=== Stage 2.5: Auto Add Register Clobber ===
├─ analyze_registers_in_gpumlir()
│  └─ 檢測: VGPR=0-7, SGPR=0-7
├─ add_register_clobber_to_gpumlir()
│  ├─ 插入 Reserve (gpu.func 開頭)
│  └─ 插入 Release (gpu.return 之前)
    ↓
GPU MLIR (with Clobber)
    ↓
=== Stage 3: mlir-opt + passes ===
├─ convert-gpu-to-rocdl
├─ gpu-to-llvm
└─ gpu-module-to-binary{format=isa}
    ↓
ISA (LLVM 生成)
    ↓
=== fix_isa_metadata() ===
├─ ✅ Fix kernarg_segment_size
├─ ✅ Fix group_segment_fixed_size
├─ ✅ Fix hidden parameters
└─ ⏭️  Skip resource counts (trust LLVM)
    ↓
ISA (最終)
    ↓
=== Stage 4: llvm-mc ===
    ↓
Object file (.o)
    ↓
=== Stage 5: ld.lld ===
    ↓
HSACO (.hsaco) ✅
```

---

## 💡 為什麼這樣做？

### 1. 簡化使用

**Before**: 3 個命令，2 個工具  
**After**: 1 個命令，1 個工具

### 2. 提高可靠性

**Before**: 手動從原始 ISA 複製 metadata（可能過時）  
**After**: LLVM 根據實際代碼計算（始終正確）

### 3. 支持 DSL 插入

**Before**: 插入 DSL 後 metadata 會不正確  
**After**: LLVM 自動調整資源需求

### 4. 減少維護負擔

**Before**: 需要維護複雜的 metadata 修復邏輯  
**After**: 信任 LLVM，只修復必要字段

---

## 🛠️ 工具變更

### pipeline.py

**新增函數**：
- `analyze_registers_in_gpumlir()` - 分析暫存器使用
- `add_register_clobber_to_gpumlir()` - 自動添加 clobber

**修改函數**：
- `translate_asm_to_gpu()` - 自動調用 clobber 添加
- `fix_isa_metadata()` - 移除資源計數修復
- `build_isa_and_hsaco()` - 移除 `skip_resource_counts` 參數

**移除參數**：
- `--trust-llvm-resources` (現在是默認行為)

### add_register_clobber_v2.py

**狀態**: 功能已整合到 `pipeline.py`，可保留用於獨立使用

---

## 📚 相關文檔

- `WORKFLOW_GUIDE.md` - 完整的工作流程指南（已更新）
- `REGISTER_CLOBBER_VALIDATION_REPORT.md` - 驗證報告
- `FILE_INDEX.md` - 文件索引

---

## ✅ 遷移檢查清單

如果您有現有的腳本使用舊 pipeline：

- [ ] 移除 `--trust-llvm-resources` 參數
- [ ] 移除 `add_register_clobber_v2.py` 的調用
- [ ] 移除 `--no-emit-isa` + 多步驟的流程
- [ ] 簡化為單一 `pipeline.py` 調用
- [ ] 測試驗證結果

---

## 🎊 總結

### 核心變更

✅ **自動化**: Register Clobber 自動添加  
✅ **簡化**: 一個命令完成所有操作  
✅ **可靠**: LLVM 自動計算，不再手動覆蓋  
✅ **未來**: 為 DSL 插入做好準備  

### 使用方式

```bash
# 就這麼簡單！
python3 pipeline.py original.s
```

---

**Last Updated**: 2025-12-23  
**Status**: ✅ Production Ready  
**Branch**: `feature/register-clobber-trust-llvm`

