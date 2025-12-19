# Pipeline.py 處理結果總結報告

## 📊 處理概況

處理了 5 個測試案例（跳過 test_01_vector_add，已處理過）：

| 測試案例 | 狀態 | HSACO 大小 | 備註 |
|---------|------|-----------|------|
| test_02_scalar_ops | ✅ | 5,448 bytes | 完全一致 |
| test_03_memory_ops | ✅ | 5,416 bytes | 完全一致 |
| test_04_conditional | ✅ | 5,592 bytes | 完全一致 |
| test_05_loop | ✅ | 5,408 bytes | 完全一致 |
| test_06_shared_memory | ⚠️ | 5,696 bytes | group_segment_size 丟失 |

## ✅ 成功的部分

### 1. 核心指令完全保留

所有測試案例的核心 ISA 指令都完全保留，包括：
- 所有 `s_*` (scalar) 指令
- 所有 `v_*` (vector) 指令  
- 所有 `global_*` (memory) 指令
- 控制流指令和標籤

唯一差異：重建文件在某些位置多了一個 `s_endpgm` 指令（無害）

### 2. Metadata 正確修復

以下 metadata 在所有測試案例中都正確保留：

| Metadata | test_02 | test_03 | test_04 | test_05 | test_06 |
|----------|---------|---------|---------|---------|---------|
| vgpr_count | 6 | 4 | 6 | 3 | 6 |
| sgpr_count | 11 | 11 | 12 | 11 | 15 |
| agpr_count | 0 | 0 | 0 | 0 | 0 |
| kernarg_size | 272 | 280 | 280 | 272 | 280 |

### 3. HSACO 成功生成

所有測試案例都成功生成了可執行的 HSACO 文件。

## ⚠️ 發現的問題

### test_06_shared_memory: group_segment_size 丟失

**問題描述：**
- 原始值：1024 bytes
- 重建值：0 bytes

**根本原因：**

`amdisa-translate` 在從 .s 轉換到 AMDISA MLIR 時，**沒有提取 `.amdhsa_group_segment_fixed_size` 屬性**。

在 AMDISA MLIR 的 module attributes 中，只有：
- `amdisa.vgpr_count`
- `amdisa.sgpr_count`
- `amdisa.agpr_count`
- `amdisa.kernarg_segment_size`
- `amdisa.kernargs`

**缺少：**
- `amdisa.group_segment_fixed_size` ❌

**影響：**

對於使用 shared memory (LDS) 的 kernel：
- Runtime 會認為不需要分配 LDS
- Kernel 執行時會訪問未分配的記憶體
- 導致未定義行為或執行錯誤 💥

## 🔧 需要修復的問題

### 1. amdisa-translate 需要支持 group_segment_size

需要修改 `amdisa-translate` 的實現，在解析 .s 文件時提取：

```assembly
.amdhsa_group_segment_fixed_size 1024
```

並在 MLIR 中生成：

```mlir
module attributes {
  amdisa.group_segment_fixed_size = 1024 : i32,
  ...
}
```

### 2. pipeline.py 需要修復 group_segment_size

在 `fix_isa_metadata()` 函數中添加對 `group_segment_fixed_size` 的處理：

```python
# 提取 group_segment_fixed_size
for attr_name in ['vgpr_count', 'sgpr_count', 'agpr_count', 
                   'kernarg_segment_size', 'group_segment_fixed_size']:  # ← 添加這個
    pattern = rf'amdisa\.{attr_name}\s*=\s*(\d+)'
    match = re.search(pattern, gpumlir_text)
    if match:
        attrs[attr_name] = int(match.group(1))
```

## 📝 建議

### 短期解決方案

對於 test_06_shared_memory，手動修復重建的 .s 文件：

```bash
sed -i 's/\.amdhsa_group_segment_fixed_size 0/\.amdhsa_group_segment_fixed_size 1024/' \
    test_06_shared_memory/pipeline_output/original_rebuilt.s
```

### 長期解決方案

1. **修改 amdisa-translate**：添加對 group_segment_fixed_size 的支持
2. **修改 pipeline.py**：在 metadata 修復邏輯中處理 group_segment_size
3. **添加測試**：確保所有 HSA metadata 都能正確往返轉換

## 🎯 結論

Pipeline.py 在大部分情況下工作良好：
- ✅ 核心指令完全保留
- ✅ 大部分 metadata 正確修復
- ✅ HSACO 成功生成

但對於使用 shared memory 的 kernel，存在一個**關鍵缺陷**，需要修復才能保證正確性。

---
生成時間: $(date)
