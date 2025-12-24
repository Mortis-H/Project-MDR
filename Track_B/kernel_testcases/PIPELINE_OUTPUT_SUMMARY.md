# Pipeline 輸出總結報告

## 概述

已為 test_01 到 test_06 生成完整的 Pipeline 輸出，包含自動 Register Clobber。

## 處理結果

### ✅ test_01_vector_add
- **檢測到的暫存器使用**: VGPR=0-7 (8), SGPR=0-7 (8)
- **Metadata 驗證**: 
  - VGPR: 8 → 8 ✅
  - SGPR: 14 → 14 ✅
  - AGPR: 0 → 0 ✅
- **輸出**: 所有中間產物和最終 HSACO

### ✅ test_02_scalar_ops
- **檢測到的暫存器使用**: VGPR=0-5 (6), SGPR=0-4 (5)
- **Metadata 驗證**: 
  - VGPR: 6 → 6 ✅
  - SGPR: 11 → 11 ✅
  - AGPR: 0 → 0 ✅
- **輸出**: 所有中間產物和最終 HSACO

### ✅ test_03_memory_ops
- **檢測到的暫存器使用**: VGPR=0-3 (4), SGPR=0-4 (5)
- **Metadata 驗證**: 
  - VGPR: 4 → 4 ✅
  - SGPR: 11 → 11 ✅
  - AGPR: 0 → 0 ✅
- **輸出**: 所有中間產物和最終 HSACO

### ✅ test_04_conditional
- **檢測到的暫存器使用**: VGPR=0-5 (6), SGPR=0-5 (6)
- **Metadata 驗證**: 
  - VGPR: 6 → 6 ✅
  - SGPR: 12 → 12 ✅
  - AGPR: 0 → 0 ✅
- **輸出**: 所有中間產物和最終 HSACO

### ✅ test_05_loop
- **檢測到的暫存器使用**: VGPR=0-2 (3), SGPR=0-4 (5)
- **Metadata 驗證**: 
  - VGPR: 3 → 3 ✅
  - SGPR: 11 → 11 ✅
  - AGPR: 0 → 0 ✅
- **輸出**: 所有中間產物和最終 HSACO

### ✅ test_06_shared_memory
- **檢測到的暫存器使用**: VGPR=0-5 (6), SGPR=0-8 (9)
- **Metadata 驗證**: 
  - VGPR: 6 → 6 ✅
  - SGPR: 15 → 15 ✅
  - AGPR: 0 → 0 ✅
- **輸出**: 所有中間產物和最終 HSACO

## 生成的檔案

每個測試的 `pipeline_output/` 目錄包含：

1. **`original_rebuilt.amdisamlir`**: AMDISA MLIR（ISA 的 MLIR 表示）
2. **`original_rebuilt_binary_isa.mlir`**: Binary ISA MLIR
3. **`original_rebuilt.gpumlir`**: GPU MLIR（包含 Register Clobber）
4. **`original_rebuilt.s`**: 重建的 ISA 組合語言
5. **`original_rebuilt.o`**: 目標檔案
6. **`original_rebuilt.hsaco`**: 可執行的 HSACO 二進制檔案

## Register Clobber 範例

### test_01_vector_add (original_rebuilt.gpumlir)

```mlir
// Auto: Reserve VGPR v[0:7] (8 registers)
%vgpr_reserved = llvm.inline_asm has_side_effects asm_dialect = att "", 
    "={v[0:7]}" : () -> vector<8xi32>

// Auto: Reserve SGPR s[0:7] (8 registers)
%sgpr_reserved = llvm.inline_asm has_side_effects asm_dialect = att "", 
    "={s[0:7]}" : () -> vector<8xi32>

// ... [原始的 kernel 邏輯在 inline_asm 中] ...

// Release VGPR clobber
llvm.inline_asm has_side_effects asm_dialect = att "", 
    "{v[0:7]}" %vgpr_reserved : (vector<8xi32>) -> ()

// Release SGPR clobber
llvm.inline_asm has_side_effects asm_dialect = att "", 
    "{s[0:8]}" %sgpr_reserved : (vector<8xi32>) -> ()
```

## 核心技術

### 自動暫存器檢測
Pipeline 自動分析原始 ISA 中使用的暫存器：
- 掃描 `llvm.inline_asm` 中的暫存器引用
- 支援單一暫存器 (`v0`, `s1`) 和範圍 (`v[0:7]`, `s[0:4]`)
- 計算每種類型的最大暫存器編號

### Register Clobber 生成
自動插入 reserve 和 release：
- **Reserve**: 在 `gpu.func` 開始處聲明暫存器使用
- **Release**: 在 `gpu.return` 前釋放暫存器

### LLVM 自動計算
LLVM 的 register allocator 根據 clobber 聲明：
- 自動計算需要的暫存器數量
- 正確設置 `.vgpr_count`, `.sgpr_count`, `.agpr_count`
- 無需手動修復 metadata

## 驗證狀態

| 測試案例 | VGPR | SGPR | AGPR | 狀態 |
|---------|------|------|------|------|
| test_01 | ✅   | ✅   | ✅   | 通過 |
| test_02 | ✅   | ✅   | ✅   | 通過 |
| test_03 | ✅   | ✅   | ✅   | 通過 |
| test_04 | ✅   | ✅   | ✅   | 通過 |
| test_05 | ✅   | ✅   | ✅   | 通過 |
| test_06 | ✅   | ✅   | ✅   | 通過 |

**100% 通過率！**

## 使用方式

```bash
# 對任何測試案例運行 Pipeline
cd /home/morhuang/Project-MDR/Track_B/kernel_testcases/test_XX_xxxxx

python3 ../../llvm-project/my_test/pipeline.py original.s \
  --workdir pipeline_output \
  --output-prefix original_rebuilt

# 查看生成的 GPU MLIR (包含 clobber)
cat pipeline_output/original_rebuilt.gpumlir

# 查看重建的 ISA
cat pipeline_output/original_rebuilt.s

# 比較 metadata
diff <(grep -E "\.vgpr_count:|\.sgpr_count:" original.s) \
     <(grep -E "\.vgpr_count:|\.sgpr_count:" pipeline_output/original_rebuilt.s)
```

## 技術價值

這些輸出檔案展示了：

1. **完整的 ISA ↔ MLIR Pipeline**: 
   - ISA → AMDISA MLIR → GPU MLIR → ISA
   
2. **自動 Register Management**:
   - 無需手動指定暫存器使用
   - LLVM 自動計算資源需求
   
3. **為 DSL 插入做好準備**:
   - 原始邏輯已被 clobber 保護
   - 可以安全地在 GPU MLIR 中插入高階操作（如 `gpu.printf`）
   - LLVM 會自動為新增的 DSL 分配額外的暫存器

## 下一步

這些帶有 clobber 的 GPU MLIR 檔案現在可以：
- ✅ 作為 DSL 插入的基礎
- ✅ 用於測試不同的 MLIR 轉換
- ✅ 驗證 metadata 的正確性
- ✅ 作為 Track_A + Track_B 整合的起點

---

生成時間: $(date)
Pipeline 版本: feature/register-clobber-trust-llvm
