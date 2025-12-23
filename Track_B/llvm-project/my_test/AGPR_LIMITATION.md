# AGPR (Accumulation GPR) 支持限制說明

## 發現時間
2025-12-23

## 問題描述

在為 Pipeline 添加 Register Clobber 支持時，發現 **LLVM AMDGPU Backend 不支持 AGPR 的 inline asm 約束**。

## 技術細節

### AGPR 是什麼？

AGPR (Accumulation GPR) 是 AMD CDNA 架構（如 MI100, MI200, MI300）中的特殊暫存器：
- 專門用於矩陣運算加速 (Matrix Core / MFMA)
- 與 VGPR 分離，有獨立的暫存器檔案
- 通常用於 `v_mfma_*` 和 `v_accvgpr_*` 指令

### 當前實現的限制

#### ✅ VGPR 和 SGPR Clobber 支持

```mlir
// VGPR Reserve
%vgpr_reserved = llvm.inline_asm has_side_effects asm_dialect = att "", 
    "={v[0:63]}" : () -> vector<64xi32>

// SGPR Reserve  
%sgpr_reserved = llvm.inline_asm has_side_effects asm_dialect = att "", 
    "={s[0:78]}" : () -> vector<79xi32>
```

**狀態**: ✅ 完全支持，LLVM 可以正確處理

#### ❌ AGPR Clobber 不支持

```mlir
// AGPR Reserve (不支持!)
%agpr_reserved = llvm.inline_asm has_side_effects asm_dialect = att "", 
    "={a[0:255]}" : () -> vector<256xi32>
```

**錯誤**:
```
mlir-opt: llvm/include/llvm/CodeGen/ValueTypes.h:317: 
llvm::MVT llvm::EVT::getSimpleVT() const: 
Assertion `isSimple() && "Expected a SimpleValueType!"' failed.
```

**原因**: LLVM AMDGPU backend 的 `ParseConstraints()` 不識別 `a` 約束字符。

## 影響範圍

### 不受影響的情況

1. **不使用 AGPR 的 kernel**
   - 絕大多數標準 GPGPU kernel
   - test_01 到 test_06 全部通過 ✅
   - 這些 kernel 只使用 VGPR 和 SGPR

2. **AGPR metadata 已正確的 kernel**
   - 如果原始 ISA 的 `.agpr_count` 已經正確
   - Pipeline 會保留這個值（通過 `fix_isa_metadata`）

### 受影響的情況

1. **需要動態計算 AGPR 需求的場景**
   - 當我們插入使用 AGPR 的 DSL (如 MFMA 操作)
   - LLVM 無法通過 clobber 得知 AGPR 使用情況
   - 需要手動計算並設定 `.agpr_count`

2. **Round-trip 精確性**
   - 對於使用 AGPR 的 kernel，無法保證 round-trip 後 `.agpr_count` 完全匹配
   - LLVM 可能生成不正確的 AGPR count

## 當前的解決方案

### Pipeline.py 的處理方式

```python
def add_register_clobber_to_gpumlir(mlir_content: str) -> str:
    max_vgpr, max_sgpr, max_agpr = analyze_registers_in_gpumlir(mlir_content)
    
    # VGPR 和 SGPR: 正常添加 clobber
    if max_vgpr >= 0:
        # 添加 VGPR reserve/release
        ...
    
    if max_sgpr >= 0:
        # 添加 SGPR reserve/release
        ...
    
    # AGPR: 只添加註釋警告
    if max_agpr >= 0:
        reserve_lines.append(f'    // ⚠️  Detected AGPR usage: a[0:{max_agpr}] ({agpr_count} registers)')
        reserve_lines.append(f'    // Note: LLVM does not support AGPR clobber constraints')
        reserve_lines.append(f'    // AGPR count may need manual adjustment in metadata')
```

**行為**:
- ✅ 檢測 AGPR 使用
- ✅ 在 GPU MLIR 中添加警告註釋
- ❌ 不嘗試添加 AGPR clobber（會導致編譯失敗）
- ⚠️  依賴 `fix_isa_metadata` 從原始 ISA 複製 `.agpr_count`

### fix_isa_metadata 的處理

```python
def fix_isa_metadata(isa_text: str, gpumlir_file: pathlib.Path) -> str:
    # 從 GPU MLIR 屬性中提取 metadata
    attrs = extract_metadata_from_gpumlir(gpumlir_file)
    
    # agpr_count: 從原始 ISA 中提取並保留
    if 'agpr_count' in attrs:
        # 使用原始 ISA 的值
        isa_text = re.sub(r'\.agpr_count:\s*\d+', 
                         f'.agpr_count:     {attrs["agpr_count"]}', 
                         isa_text)
```

## 測試案例：test_08_HK_kernels

### 原始 Metadata

```asm
.agpr_count:     256
.sgpr_count:     85
.vgpr_count:     512
```

### 實際使用情況

- VGPR: v0-v63 (僅 64 個)
- SGPR: s0-s78 (79 個)
- AGPR: a0-a255 (256 個，全部使用)

### 問題

這不是一個好的測試案例，因為：
1. `.vgpr_count: 512` 與實際使用 (64) 不符
2. Kernel 可能使用了特殊編譯選項預留所有暫存器
3. 無法驗證我們的 clobber 機制是否正確

## 建議的測試策略

### 1. 尋找更好的 AGPR 測試案例

需要一個：
- 實際使用少量 AGPR 的 kernel（如 a0-a31）
- Metadata 與實際使用匹配的 kernel
- 可以驗證 round-trip 的案例

### 2. 創建合成測試案例

手動編寫一個簡單的 MFMA kernel：
```asm
v_mfma_f32_16x16x16f16 a[0:3], v[0:1], v[2:3], a[0:3]
```

這樣只使用少量 AGPR，方便驗證。

## 未來可能的改進

### 選項 1: 向 LLVM 貢獻 AGPR 約束支持

需要在 LLVM AMDGPU backend 中：
1. 添加 `a` 約束字符的解析
2. 實現 AGPR clobber 的寄存器分配邏輯
3. 提交 patch 到 upstream

**難度**: 高，需要深入理解 LLVM backend

### 選項 2: 手動計算 AGPR 需求

在 DSL 插入階段：
1. 分析插入的 DSL 是否使用 AGPR
2. 手動計算總 AGPR 需求
3. 通過 GPU MLIR 屬性傳遞給後續階段
4. 在 `fix_isa_metadata` 中設定正確的值

**難度**: 中，需要準確追蹤 AGPR 使用

### 選項 3: 限制 DSL 不使用 AGPR

最保守的方案：
- 禁止 DSL 使用需要 AGPR 的操作
- 只允許使用 VGPR 和 SGPR
- 保持當前的 clobber 機制

**難度**: 低，但限制了功能

## 當前建議

### 短期方案 (已實現)

1. ✅ 檢測 AGPR 使用並發出警告
2. ✅ 對 VGPR 和 SGPR 使用 clobber
3. ✅ 通過 `fix_isa_metadata` 保留原始 AGPR count
4. ⚠️  接受 AGPR 相關的 round-trip 可能不精確

### 中期方案 (推薦)

1. 找到或創建更好的 AGPR 測試案例
2. 實現手動 AGPR 計算機制
3. 在 DSL 插入時追蹤 AGPR 使用
4. 更新文檔說明 AGPR 限制

### 長期方案 (可選)

1. 向 LLVM 提交 AGPR 約束支持
2. 或者尋找替代方案繞過限制

## 相關文件

- `pipeline.py`: 當前的實現
- `MIGRATION_GUIDE.md`: 使用指南
- `VALIDATION_REPORT_INTEGRATED.md`: 驗證報告（test_01-06，無 AGPR）

## 總結

**AGPR 是 AMDGPU 的高級功能，主要用於 MFMA/矩陣運算：**

✅ **目前支持**: VGPR 和 SGPR 的完整 clobber 機制  
⚠️  **部分支持**: AGPR 檢測和警告  
❌ **不支持**: AGPR clobber（LLVM 限制）  
🔧 **解決方案**: 保留原始 metadata + 未來手動計算

**對於絕大多數 kernel（不使用 AGPR）**：當前的實現完全可靠 ✅

**對於使用 AGPR 的 kernel**：需要額外注意 metadata 的正確性 ⚠️

---

**Created**: 2025-12-23  
**Status**: Documented Limitation  
**Priority**: Low (影響範圍有限)

