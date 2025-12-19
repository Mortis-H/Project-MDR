# Test 07: Multi-Kernel HSACO

## 📋 測試目的

測試 `amdisa-translate` 和 `pipeline.py` 處理包含**多個 kernel** 的單一 HSACO 文件的能力。

這是一個關鍵測試，因為實際應用中，一個 HSACO 文件通常包含多個 kernel函數。

## 🎯 測試重點

1. **多 Kernel 解析**：amdisa-translate 能否正確解析包含多個 kernel 的 .s 文件
2. **MLIR 表示**：每個 kernel 是否都能正確轉換為 GPU MLIR
3. **Metadata 保留**：每個 kernel 的 metadata 是否都能正確保留
4. **獨立執行**：重建後的 HSACO 中每個 kernel 是否都能獨立執行
5. **結果一致性**：每個 kernel 的執行結果是否與原始 HSACO 一致

## 📦 文件結構

```
test_07_multi_kernel/
├── README.md              # 本文件
├── test_config.json       # 測試配置
├── original.s             # 包含 3 個 kernel 的組合語言文件
├── test_multi_kernel.sh   # 測試腳本
└── pipeline_output/       # pipeline.py 的輸出（執行後生成）
    ├── original_rebuilt.amdisamlir
    ├── original_rebuilt.gpumlir
    ├── original_rebuilt.s
    └── original_rebuilt.hsaco
```

## 🔬 包含的 Kernel

### Kernel 1: vectorAdd
- **函數名**：`_Z9vectorAddPKfS0_Pfi`
- **功能**：Float 向量加法
- **簽名**：`void vectorAdd(const float* a, const float* b, float* c, int n)`
- **測試類型**：`float_add`
- **測試大小**：1024 元素

### Kernel 2: scalarOps
- **函數名**：`_Z9scalarOpsPii`
- **功能**：Int 純量運算（位元和算術運算）
- **簽名**：`void scalarOps(int* output, int n)`
- **測試類型**：`int_scalar`
- **測試大小**：1024 元素

### Kernel 3: memoryOps
- **函數名**：`_Z9memoryOpsPKiPii`
- **功能**：Int 記憶體操作
- **簽名**：`void memoryOps(const int* input, int* output, int n)`
- **測試類型**：`int_mem`
- **測試大小**：1024 元素

## 🚀 使用方法

### 1. 使用 pipeline.py 處理

```bash
cd /home/morhuang/Project-MDR/Track_B/2025_12_19/01_check_latest_amdisa-translate/test_results/test_07_multi_kernel

python3 /home/morhuang/Project-MDR/Track_B/llvm-project/my_test/pipeline.py \
    --chip gfx950 \
    --workdir pipeline_output \
    original.s
```

### 2. 測試所有 kernel

```bash
./test_multi_kernel.sh
```

或手動測試每個 kernel：

```bash
# 編譯和連結
llvm-mc -triple amdgcn-amd-amdhsa -mcpu=gfx950 -filetype=obj original.s -o original.o
ld.lld -shared original.o -o original.hsaco

# 測試 kernel 1: vectorAdd
/home/morhuang/Project-MDR/Track_B/llvm-project/mlir/test/Dialect/AMDISA/universal_hsaco_runner \
    original.hsaco _Z9vectorAddPKfS0_Pfi float_add 1024

# 測試 kernel 2: scalarOps
/home/morhuang/Project-MDR/Track_B/llvm-project/mlir/test/Dialect/AMDISA/universal_hsaco_runner \
    original.hsaco _Z9scalarOpsPii int_scalar 1024

# 測試 kernel 3: memoryOps
/home/morhuang/Project-MDR/Track_B/llvm-project/mlir/test/Dialect/AMDISA/universal_hsaco_runner \
    original.hsaco _Z9memoryOpsPKiPii int_mem 1024
```

## ✅ 預期結果

### 成功條件

1. **轉換成功**：
   - ✅ amdisa-translate 成功解析 original.s
   - ✅ 生成包含 3 個 kernel 的 MLIR
   - ✅ 成功重建 .s 文件

2. **HSACO 生成**：
   - ✅ 成功生成 original_rebuilt.hsaco
   - ✅ HSACO 包含所有 3 個 kernel

3. **執行測試**：
   - ✅ 每個 kernel 都能獨立執行
   - ✅ 每個 kernel 的結果與原始 HSACO 一致

4. **Metadata 檢查**：
   - ✅ 每個 kernel 的 vgpr_count, sgpr_count 正確
   - ✅ 每個 kernel 的 kernarg_size 正確

### 可能的問題

1. **Kernel 混淆**：
   - 不同 kernel 的代碼或 metadata 互相混淆
   - 標籤衝突（例如 .LBB0_1 在多個 kernel 中重複）

2. **Metadata 丟失**：
   - 某些 kernel 的 metadata 未被保留
   - group_segment_size 等特殊屬性丟失

3. **執行錯誤**：
   - 某個 kernel 執行失敗
   - 執行結果不一致

## 📊 測試指標

### 轉換指標
- Kernel 識別率：3/3 (100%)
- Metadata 保留率：應為 100%
- 代碼完整性：所有 ISA 指令保留

### 執行指標
- 執行成功率：3/3 (100%)
- 結果一致性：3/3 (100%)
- 獨立性：每個 kernel 可獨立調用

## 🔍 檢查重點

### 1. 檢查 MLIR 輸出

```bash
# 查看 AMDISA MLIR
cat pipeline_output/original_rebuilt.amdisamlir | grep "amdisa.kernel_name"

# 應該看到 3 個不同的 kernel name
```

### 2. 檢查重建的 .s 文件

```bash
# 查看 kernel 列表
grep "^\.globl" pipeline_output/original_rebuilt.s

# 應該看到 3 個 .globl 聲明
```

### 3. 檢查 HSACO

```bash
# 查看 HSACO 中的 kernel
readelf -s pipeline_output/original_rebuilt.hsaco | grep FUNC

# 應該看到 3 個函數符號
```

## 📝 測試報告模板

測試完成後，記錄以下信息：

```
測試日期：____
測試者：____

轉換結果：
  □ original.s → AMDISA MLIR: [成功/失敗]
  □ AMDISA MLIR → GPU MLIR: [成功/失敗]
  □ GPU MLIR → rebuilt.s: [成功/失敗]
  □ rebuilt.s → HSACO: [成功/失敗]

Kernel 檢測：
  □ Kernel 1 (vectorAdd): [檢測到/未檢測到]
  □ Kernel 2 (scalarOps): [檢測到/未檢測到]
  □ Kernel 3 (memoryOps): [檢測到/未檢測到]

執行測試：
  □ Kernel 1 執行: [成功/失敗] - 結果一致: [是/否]
  □ Kernel 2 執行: [成功/失敗] - 結果一致: [是/否]
  □ Kernel 3 執行: [成功/失敗] - 結果一致: [是/否]

發現的問題：
  1. ____
  2. ____

總體評價：[通過/失敗]
```

## 🛠️ 故障排除

### 問題：只檢測到 1 個 kernel

**可能原因**：
- amdisa-translate 可能只處理第一個 kernel
- 標籤衝突導致後續 kernel 被忽略

**排查方法**：
```bash
# 檢查 AMDISA MLIR
cat pipeline_output/original_rebuilt.amdisamlir | grep -c "amdisa.kernel_name"
```

### 問題：某個 kernel 執行失敗

**可能原因**：
- 該 kernel 的 metadata 錯誤
- 寄存器分配衝突

**排查方法**：
```bash
# 比較原始和重建的 metadata
diff -u <(grep -A 50 "kernel_name_here" original.s) \
        <(grep -A 50 "kernel_name_here" pipeline_output/original_rebuilt.s)
```

## 📚 參考資料

- [Track_B README](../../README.md)
- [universal_hsaco_runner 文檔](../../../llvm-project/mlir/test/Dialect/AMDISA/README.md)
- [pipeline.py 使用說明](../../../llvm-project/my_test/pipeline.py)

---

**創建日期**：2025-12-19  
**維護者**：AI Assistant  
**狀態**：準備測試

