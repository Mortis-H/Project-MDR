# Test 07 Multi-Kernels 測試報告

## 測試日期
2025-12-23

## 測試目的
驗證 AMDISA Dialect 對多 kernel 文件的支持

## 原始文件分析

### 包含的 Kernels
原始 `original.s` 文件包含 **5 個獨立的 GPU kernels**：

1. `_Z9vectorAddPKfS0_Pfi` - 向量加法
2. `_Z9vectorMulPKfS0_Pfi` - 向量乘法
3. `_Z9vectorDotPKfS0_Pfi` - 向量點積（使用 shared memory）
4. `_Z5saxpyfPKfPfi` - SAXPY 運算
5. `_Z14conditionalOpsPKfPffi` - 條件運算

### 文件大小
- `original.s`: 32 KB
- `test_07_multi_kernels.hip`: 5.6 KB

## 測試結果

### ✅ 第一部分：.s 轉換

**狀態：部分成功**

| 階段 | 結果 | 說明 |
|------|------|------|
| `.s` → AMDISA MLIR | ⚠️ 部分成功 | 所有指令被解析，但未分離為多個函數 |
| AMDISA MLIR → GPU MLIR | ⚠️ 部分成功 | 只生成第一個 kernel 的 GPU MLIR |
| GPU MLIR → ISA + HSACO | ✅ 成功 | 第一個 kernel 成功生成 HSACO |

### ⚠️ 發現的問題

#### 問題 1: 多 Kernel 識別
**現象：**
- 原始 .s 文件: 5 個 kernels
- AMDISA MLIR: 所有指令在一個扁平的 module 中
- GPU MLIR: 只有 1 個 `gpu.func`
- HSACO: 只包含第一個 kernel

**原因分析：**
`amdisa-translate` 目前的實現將整個 .s 文件視為線性指令序列，沒有：
1. 識別函數邊界（`.globl`, `.type`, `s_endpgm`）
2. 為每個 kernel 創建獨立的 MLIR 函數
3. 保留每個 kernel 的獨立 metadata

**技術細節：**
```
AMDISA MLIR 結構:
module attributes {
  amdisa.kernel_name = "_Z9vectorAddPKfS0_Pfi"  // 只有第一個 kernel 的名稱
  ...
} {
  amdisa.inst {...}  // 所有 5 個 kernels 的指令混在一起
  amdisa.inst {...}
  amdisa.label {name = "_Z9vectorMulPKfS0_Pfi:"}  // 其他 kernel 只是 label
  ...
}
```

應該是：
```
module {
  amdisa.func @_Z9vectorAddPKfS0_Pfi(...) {
    amdisa.inst {...}
    ...
  }
  amdisa.func @_Z9vectorMulPKfS0_Pfi(...) {
    amdisa.inst {...}
    ...
  }
  ...
}
```

### ✅ 第二部分：HSACO 執行驗證

**第一個 Kernel 測試：**

| Kernel | 類型 | 測試大小 | 結果 | 狀態 |
|--------|------|---------|------|------|
| `_Z9vectorAddPKfS0_Pfi` | float_add | 1024 | 所有 1024 元素正確 | ✅ PASS |

**其他 Kernels：**
- ❌ 無法測試（未包含在 HSACO 中）

## 生成的文件

```
pipeline_output/
├── original_rebuilt.amdisamlir   ✅ (包含所有指令，但未分離函數)
├── original_rebuilt.gpumlir      ⚠️ (只有第一個 kernel)
├── original_rebuilt.s            ⚠️ (只有第一個 kernel)
└── original_rebuilt.hsaco        ⚠️ (只有第一個 kernel)
```

## 當前限制

### 1. 多 Kernel 支持不完整
- ✅ 可以解析包含多個 kernel 的 .s 文件
- ❌ 無法將多個 kernel 分離為獨立的函數
- ❌ 只有第一個 kernel 會被轉換為 GPU MLIR
- ❌ 最終 HSACO 只包含第一個 kernel

### 2. Metadata 處理
- ✅ 第一個 kernel 的 metadata 正確保留
- ❌ 其他 kernel 的 metadata 丟失

### 3. 函數邊界識別
需要改進的地方：
- 識別 `.globl` 和 `.type` 指令標記函數開始
- 識別 `s_endpgm` 指令標記函數結束
- 為每個函數創建獨立的 MLIR 結構

## 建議的改進方向

### 短期方案：分離多個 Kernel
如果需要處理多 kernel 文件，可以：
1. 手動將 `original.s` 分離為多個單獨的 .s 文件
2. 分別處理每個 kernel
3. 使用工具合併多個 HSACO（如果需要）

### 長期方案：增強 amdisa-translate
需要在 `amdisa-translate` 中實現：

1. **函數邊界檢測**
   ```cpp
   // 偽代碼
   if (directive == ".globl" && next_directive == ".type") {
       startNewFunction(symbol_name);
   }
   if (instruction == "s_endpgm") {
       endCurrentFunction();
   }
   ```

2. **多函數 MLIR 生成**
   ```mlir
   module {
     amdisa.func @kernel1(...) attributes {...} { ... }
     amdisa.func @kernel2(...) attributes {...} { ... }
     ...
   }
   ```

3. **Metadata 分離**
   - 為每個函數解析獨立的 `.amdhsa_*` 指令
   - 為每個函數保留獨立的 kernarg 信息

## 總結

### ✅ 成功的部分
1. ✅ 可以解析包含多個 kernel 的 .s 文件
2. ✅ 第一個 kernel 完整轉換成功
3. ✅ 第一個 kernel 的 HSACO 可以正確執行
4. ✅ Pipeline 工具運行穩定

### ⚠️ 限制
1. ⚠️ 只支持單 kernel 轉換
2. ⚠️ 多 kernel 文件只會處理第一個
3. ⚠️ 其他 kernel 的代碼和 metadata 會丟失

### 📝 建議
- **當前使用**：對於多 kernel 文件，建議先分離為單獨的 .s 文件
- **未來改進**：需要增強 amdisa-translate 的多函數識別能力

## 測試評級

| 項目 | 評級 | 說明 |
|------|------|------|
| 單 Kernel 支持 | ⭐⭐⭐⭐⭐ | 完全支持 |
| 多 Kernel 支持 | ⭐⭐ | 僅部分支持（只處理第一個）|
| 功能正確性 | ⭐⭐⭐⭐⭐ | 處理的 kernel 完全正確 |
| 穩定性 | ⭐⭐⭐⭐⭐ | 無崩潰或錯誤 |

**整體評級：⭐⭐⭐⭐ (4/5)**

對於單 kernel 使用場景，工具鏈完全可用且穩定。
對於多 kernel 場景，需要額外的預處理步驟。

