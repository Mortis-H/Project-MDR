# Register Clobber Validation Report

## 執行摘要

本實驗成功驗證了核心假設：**通過 Register Clobber（Reserve + Release）機制，LLVM 能夠正確計算 GPU kernel 的資源需求（VGPR/SGPR counts），無需依賴後處理的 metadata 修復。**

## 背景與動機

### 問題描述

Track_B 原本的 `amdisa-translate pipeline` 使用三階段策略來處理 ISA metadata：

1. 從原始 ISA 提取 metadata 並存為 `amdisa.*` attributes
2. 通過 MLIR 優化 passes 時，這些 attributes 會丟失
3. 使用 `fix_isa_metadata()` 函數後處理，將 metadata 強制寫回生成的 ISA

**這種方法的局限性**：當插入新的 DSL（如 `gpu.printf`）時，原始的資源計數不再準確，但 `fix_isa_metadata()` 仍會覆蓋 LLVM 計算的正確值。

### 解決方案

借鑒 Track_A 的經驗，使用 **Register Clobber** 機制：

1. **Reserve（開頭）**：使用 `llvm.inline_asm` 聲明原始 ISA 使用的暫存器範圍
2. **Release（結尾）**：告訴 LLVM 這些暫存器的生命週期結束

這樣 LLVM 的 register allocator 就能正確理解暫存器使用情況，並計算出準確的資源需求。

## 技術實現

### 1. Register Clobber 代碼生成

創建了 `add_register_clobber_v2.py` 工具，能自動：

- **分析** GPU MLIR 中所有 `llvm.inline_asm` 的暫存器使用
- **生成** Reserve 聲明（在 `gpu.func` 開頭）
- **生成** Release 聲明（在 `gpu.return` 之前）

#### Reserve 示例（開頭）

```mlir
// ===== Reserve Register Clobber =====
// Reserve VGPR: v[0:7] (8 registers)
%vgpr_reserved = llvm.inline_asm has_side_effects asm_dialect = att "", "={v[0:7]}" : () -> vector<8xi32>
// Reserve SGPR: s[0:7] (8 registers)
%sgpr_reserved = llvm.inline_asm has_side_effects asm_dialect = att "", "={s[0:7]}" : () -> vector<8xi32>
// =====================================
```

#### Release 示例（結尾）

```mlir
// ===== Release Register Clobber =====
// Release VGPR clobber: v[0:7]
llvm.inline_asm has_side_effects asm_dialect = att "", "{v[0:7]}" %vgpr_reserved : (vector<8xi32>) -> ()
// Release SGPR clobber: s[0:7]
llvm.inline_asm has_side_effects asm_dialect = att "", "{s[0:7]}" %sgpr_reserved : (vector<8xi32>) -> ()
// =====================================
```

### 2. Pipeline 集成

在 `pipeline.py` 中添加 `--trust-llvm-resources` 參數：

- **功能**：跳過 `fix_isa_metadata()` 中的 VGPR/SGPR counts 修復
- **適用場景**：當 GPU MLIR 包含 register clobber 聲明時
- **實現**：在 `fix_isa_metadata()` 中添加 `skip_resource_counts` 參數

```python
def fix_isa_metadata(isa_text: str, gpumlir_file: pathlib.Path, skip_resource_counts: bool = False) -> str:
    ...
    if not skip_resource_counts:
        # 修復 VGPR/SGPR counts
        if 'vgpr_count' in attrs:
            kernel['.vgpr_count'] = attrs['vgpr_count']
        ...
    else:
        print("[Info] Skipping resource count fixes (trusting LLVM)")
    ...
```

## 驗證結果

### 測試覆蓋

對 `Track_B/kernel_testcases` 中的 6 個測試案例進行了完整驗證：

| 測試案例 | 描述 | VGPR (原始→最終) | SGPR (原始→最終) | Kernarg | 執行結果 |
|---------|------|-----------------|-----------------|---------|---------|
| test_01_vector_add | Vector addition | 8 → 8 ✅ | 14 → 14 ✅ | 288 ✅ | PASS ✅ |
| test_02_scalar_ops | Scalar operations | 6 → 6 ✅ | 11 → 11 ✅ | 272 ✅ | PASS ✅ |
| test_03_memory_ops | Memory operations | 4 → 4 ✅ | 11 → 11 ✅ | 280 ✅ | PASS ✅ |
| test_04_conditional | Conditional branching | 6 → 6 ✅ | 12 → 12 ✅ | 280 ✅ | PASS ✅ |
| test_05_loop | Loop operations | 3 → 3 ✅ | 11 → 11 ✅ | 272 ✅ | PASS ✅ |
| test_06_shared_memory | Shared memory | 6 → 6 ✅ | 15 → 15 ✅ | 280 ✅ | PASS ✅ |

### 驗證流程

```bash
ISA (原始)
  ↓ [amdisa-translate -emit=mlir]
AMDISA MLIR
  ↓ [amdisa-translate -emit=gpu]
GPU MLIR
  ↓ [add_register_clobber_v2.py]
GPU MLIR (with Reserve + Release)
  ↓ [mlir-opt + gpu-module-to-binary]
ISA (重建) ← LLVM 自動計算 metadata
  ↓ [llvm-mc + ld.lld]
HSACO
  ↓ [universal_hsaco_runner]
✅ 執行成功 + 結果正確
```

### 關鍵發現

1. ✅ **Metadata 完全匹配**：所有測試的 VGPR、SGPR、kernarg size 與原始版本完全一致
2. ✅ **功能等價性**：所有重建的 HSACO 都能正確執行並通過驗證
3. ✅ **自動化**：整個流程完全自動化，無需手動干預
4. ✅ **可擴展性**：為未來插入 DSL（如 `gpu.printf`）鋪平道路

## 對比分析

### 方法對比

| 特性 | 原始方法 (fix_isa_metadata) | 新方法 (Register Clobber) |
|-----|--------------------------|-------------------------|
| **Metadata 來源** | 從原始 ISA 提取並覆蓋 | LLVM 自動計算 |
| **適用場景** | Pure round-trip (ISA → MLIR → ISA) | Round-trip + DSL injection |
| **DSL 插入** | ❌ Metadata 會不正確 | ✅ LLVM 自動調整資源 |
| **維護成本** | 需維護複雜的後處理邏輯 | 一次性實現 clobber 生成 |
| **正確性保證** | 手動驗證 | LLVM register allocator 保證 |

### 未來擴展：DSL 插入

使用 Register Clobber 後，可以安全地插入 DSL：

```mlir
// Reserve 原始 ISA 使用的暫存器
%vgpr_reserved = llvm.inline_asm ... "={v[0:7]}" ...

// 原始 ISA 的 inline_asm
llvm.inline_asm "v_add_f32 v6, v6, v7" ...

// 插入 DSL（LLVM 會分配 v8+ 暫存器）
%tid = gpu.thread_id x
gpu.printf "Thread %d\n", %tid : index

// Release
llvm.inline_asm ... "{v[0:7]}" %vgpr_reserved ...
```

LLVM 會自動：
- 為 `gpu.printf` 分配 v8+ 暫存器
- 計算總 VGPR 需求 (例如 8 + 額外需求)
- 生成正確的 metadata

## 工具鏈

### 新增工具

1. **`add_register_clobber_v2.py`**
   - 自動分析暫存器使用
   - 生成 Reserve + Release 代碼
   - 用法：`python3 add_register_clobber_v2.py input.gpumlir -o output.gpumlir`

2. **`validate_with_clobber.sh`**
   - 自動化完整驗證流程
   - 對比 metadata 和執行結果
   - 用法：`./validate_with_clobber.sh`

### 增強功能

- **`pipeline.py`**: 添加 `--trust-llvm-resources` 參數
- **`fix_isa_metadata()`**: 添加 `skip_resource_counts` 參數

## 結論

### 核心成果

✅ **驗證成功**：Register Clobber（Reserve + Release）+ 信任 LLVM = 正確的 Metadata

### 技術意義

1. **解決了關鍵瓶頸**：為 DSL 插入功能鋪平道路
2. **提升了可靠性**：依賴 LLVM 的 register allocator 而非手動計算
3. **簡化了維護**：減少了後處理邏輯的複雜度

### 下一步

1. ✅ 完成 Register Clobber 機制驗證
2. 🔄 整合 Track_A 和 Track_B 的功能
3. 📋 實現完整的 ISA → MLIR → DSL injection → ISA pipeline
4. 🚀 支持更複雜的 DSL（print, assert, profiling 等）

## 附錄

### 實驗環境

- **平台**：AMD gfx950
- **LLVM 版本**：MLIR + ROCDL
- **工具鏈**：amdisa-translate, mlir-opt, llvm-mc, ld.lld
- **驗證工具**：universal_hsaco_runner

### 相關文件

- `Track_A/kernel_template.mlir`: Register Clobber 參考實現
- `Track_B/llvm-project/my_test/pipeline.py`: AMDISA Pipeline
- `Track_B/kernel_testcases/`: 測試案例
- `Track_B/llvm-project/my_test/add_register_clobber_v2.py`: Clobber 生成工具
- `Track_B/llvm-project/my_test/validate_with_clobber.sh`: 驗證腳本

### 驗證日誌

完整的驗證結果保存在：
- `clobber_validation_full.log`: 完整驗證日誌
- `clobber_*/`: 每個測試的中間文件

---

**Report Date**: 2025-12-23  
**Status**: ✅ All Tests Passed (6/6)  
**Conclusion**: Register Clobber 機制完全可行，可以取代 fix_isa_metadata 進行資源計算

