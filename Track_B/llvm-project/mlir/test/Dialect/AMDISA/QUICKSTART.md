# AMDISA 測試快速開始指南

## 5 分鐘快速測試

### 步驟 1: 編譯執行器 (只需一次)

```bash
cd /home/morhuang/Project-MDR/Track_B/llvm-project/mlir/test/Dialect/AMDISA
make
```

**預期輸出**：
```
/opt/rocm/bin/hipcc -std=c++11 -O2 universal_hsaco_runner.cpp -o universal_hsaco_runner
```

### 步驟 2: 測試單一 Kernel

```bash
./test_asm_to_hsaco.sh \
    test_results/test_01_vector_add/stage3_rebuilt.s \
    _Z9vectorAddPKfS0_Pfi \
    float_add \
    1024
```

**預期輸出**：
```
========================================
從組裝檔案到 GPU 執行驗證
========================================
[1/3] 組裝 .s → .o
  ✓ 組裝成功: stage3_rebuilt_test.o (4936 bytes)
[2/3] 連結 .o → .hsaco
  ✓ 連結成功: stage3_rebuilt_test.hsaco (6256 bytes)
[3/3] GPU 執行驗證
  ✓ GPU 執行成功並通過驗證
✅ PASS: All 1024 elements correct
```

### 步驟 3: 測試所有 Kernel

```bash
./test_all_rebuilt_s.sh
```

**預期輸出**：
```
========================================
🎉 所有測試通過！
========================================
驗證結果：
  • ✓ 6/6 kernel 重建的 .s 檔案測試通過
  • ✓ 組裝 .s → .o 成功
  • ✓ 連結 .o → .hsaco 成功
  • ✓ 在 AMD Instinct MI350X 上成功執行
```

## 完成！

恭喜！您已經成功驗證了從 `.s` 組裝檔案到 GPU 執行的完整流程。

## 測試其他 Kernel

### 可用的測試案例

| 測試 | .s 檔案路徑 | Kernel 名稱 | 類型 |
|------|------------|-------------|------|
| test_01 | `test_results/test_01_vector_add/stage3_rebuilt.s` | `_Z9vectorAddPKfS0_Pfi` | `float_add` |
| test_02 | `test_results/test_02_scalar_ops/stage3_rebuilt.s` | `_Z9scalarOpsPii` | `int_scalar` |
| test_03 | `test_results/test_03_memory_ops/stage3_rebuilt.s` | `_Z9memoryOpsPKiPii` | `int_mem` |
| test_04 | `test_results/test_04_conditional/stage3_rebuilt.s` | `_Z17conditionalKernelPKiPii` | `int_cond` |
| test_05 | `test_results/test_05_loop/stage3_rebuilt.s` | `_Z10loopKernelPii` | `int_loop` |
| test_06 | `test_results/test_06_shared_memory/stage3_rebuilt.s` | `_Z15sharedMemKernelPKiPii` | `int_shared` |

### 範例：測試 Shared Memory Kernel

```bash
./test_asm_to_hsaco.sh \
    test_results/test_06_shared_memory/stage3_rebuilt.s \
    _Z15sharedMemKernelPKiPii \
    int_shared \
    1024
```

## 測試自己的 .s 檔案

如果您有自己的 AMD GPU 組裝檔案：

```bash
./test_asm_to_hsaco.sh \
    /path/to/your/kernel.s \
    your_kernel_name \
    kernel_type \
    test_size
```

**Kernel 類型**：
- `float_add` - float 向量運算
- `int_scalar` - int 純量運算 (output, n)
- `int_mem` - int 記憶體操作 (input, output, n)
- `int_cond` - int 條件判斷 (input, output, n)
- `int_loop` - int 迴圈 (output, n)
- `int_shared` - int 共享記憶體 (input, output, n)

## 檢視詳細報告

```bash
# 檢視完整測試報告
cat TEST_REPORT.md

# 檢視重組說明
cat REORGANIZATION_SUMMARY.md

# 檢視完整文檔
cat README.md
```

## 疑難排解

### 問題：找不到 llvm-mc

```bash
export PATH=/home/morhuang/llvm-project/build/bin:$PATH
```

### 問題：找不到 ld.lld

```bash
export PATH=/home/morhuang/llvm-project/build/bin:$PATH
```

### 問題：HIP 錯誤

```bash
# 檢查 ROCm 安裝
rocminfo

# 檢查 HIP
hipcc --version
```

### 問題：GPU 不可用

```bash
# 列出可用的 GPU
rocm-smi
```

## 下一步

- 閱讀 [README.md](README.md) 了解完整功能
- 查看 [TEST_REPORT.md](TEST_REPORT.md) 了解詳細測試結果
- 參考 [REORGANIZATION_SUMMARY.md](REORGANIZATION_SUMMARY.md) 了解架構

## 技術支援

如有問題，請檢查：
1. GPU 是否正常運作 (`rocm-smi`)
2. LLVM 工具是否正確編譯 (`llvm-mc --version`)
3. HIP 環境是否正確配置 (`hipcc --version`)

---

**快速開始指南** | 更新日期：2024-12-17

