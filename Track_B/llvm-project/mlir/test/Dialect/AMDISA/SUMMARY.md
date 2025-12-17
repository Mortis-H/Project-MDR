# AMDISA 測試套件 - 總覽

## 📋 快速導航

- 🚀 **快速開始**: [QUICKSTART.md](QUICKSTART.md) - 5 分鐘快速測試指南
- 📖 **完整文檔**: [README.md](README.md) - 詳細說明和使用方式
- 📊 **測試報告**: [TEST_REPORT.md](TEST_REPORT.md) - 完整測試結果
- 🔄 **重組說明**: [REORGANIZATION_SUMMARY.md](REORGANIZATION_SUMMARY.md) - 重組過程和原因

## ✅ 測試狀態

**當前狀態**: 全部通過 ✅ (6/6 kernels)

```
test_01_vector_add      ✅ PASS
test_02_scalar_ops      ✅ PASS
test_03_memory_ops      ✅ PASS
test_04_conditional     ✅ PASS
test_05_loop            ✅ PASS
test_06_shared_memory   ✅ PASS
```

## 🎯 核心目標

驗證從 `.s` 組裝檔案到 GPU 執行的完整流程：

```
.s 組裝檔案 → llvm-mc → .o 物件檔 → ld.lld → .hsaco 執行檔 → GPU 驗證 ✅
```

## 📁 檔案結構

### 主要檔案

```
AMDISA/
├── 📚 文檔
│   ├── QUICKSTART.md              快速開始指南
│   ├── README.md                  完整文檔
│   ├── TEST_REPORT.md             測試報告
│   ├── REORGANIZATION_SUMMARY.md  重組說明
│   └── SUMMARY.md                 本文件
│
├── 🧪 測試腳本
│   ├── test_asm_to_hsaco.sh      單一檔案測試 ✅
│   ├── test_all_rebuilt_s.sh     批次測試 ✅
│   ├── test_mlir_pipeline.sh     完整管道測試 ⚠️
│   └── run_all_tests.sh          執行所有測試 ⚠️
│
├── 🔧 工具
│   ├── universal_hsaco_runner.cpp  HSACO 執行器源碼
│   ├── universal_hsaco_runner      編譯後的執行器
│   └── Makefile                    編譯配置
│
├── 📦 測試資料
│   ├── hip_kernels/              原始 HIP 源碼
│   ├── test_results/             測試結果和中間檔案
│   └── archive_old_tests/        舊檔案存檔
```

### 測試結果目錄結構

```
test_results/
└── test_01_vector_add/
    ├── stage3_rebuilt.s              重建的組裝檔案 (輸入)
    ├── stage3_rebuilt_test.o         組裝後的物件檔
    ├── stage3_rebuilt_test.hsaco     連結後的執行檔
    └── rebuilt_test.log              測試日誌
```

## 🚀 快速使用

### 一行命令測試

```bash
cd /home/morhuang/Project-MDR/Track_B/llvm-project/mlir/test/Dialect/AMDISA && ./test_all_rebuilt_s.sh
```

### 測試單一 Kernel

```bash
./test_asm_to_hsaco.sh \
    test_results/test_01_vector_add/stage3_rebuilt.s \
    _Z9vectorAddPKfS0_Pfi \
    float_add \
    1024
```

## 📊 測試覆蓋

### Kernel 類型

| 類型 | 說明 | 測試案例 | 狀態 |
|------|------|----------|------|
| Float 向量運算 | 向量加法 | test_01 | ✅ |
| Int 純量運算 | 純量計算 | test_02 | ✅ |
| Int 記憶體操作 | 記憶體讀寫 | test_03 | ✅ |
| Int 條件判斷 | 分支控制 | test_04 | ✅ |
| Int 迴圈 | 迴圈結構 | test_05 | ✅ |
| Int 共享記憶體 | LDS 操作 | test_06 | ✅ |

### 測試階段

| 階段 | 工具 | 輸入 | 輸出 | 狀態 |
|------|------|------|------|------|
| 組裝 | llvm-mc | .s | .o | ✅ |
| 連結 | ld.lld | .o | .hsaco | ✅ |
| 執行 | universal_hsaco_runner | .hsaco | 結果 | ✅ |

## 🔑 關鍵技術

### 1. 組裝命令

```bash
llvm-mc -triple amdgcn-amd-amdhsa -mcpu=gfx950 -filetype=obj input.s -o output.o
```

### 2. 連結命令

```bash
ld.lld -shared input.o -o output.hsaco
```

### 3. 執行命令

```bash
universal_hsaco_runner output.hsaco kernel_name kernel_type test_size
```

## 🎓 學習資源

### 新手入門

1. 閱讀 [QUICKSTART.md](QUICKSTART.md)
2. 執行 `./test_all_rebuilt_s.sh`
3. 查看 [TEST_REPORT.md](TEST_REPORT.md) 了解結果

### 進階使用

1. 閱讀 [README.md](README.md) 完整文檔
2. 了解 [REORGANIZATION_SUMMARY.md](REORGANIZATION_SUMMARY.md) 架構設計
3. 測試自己的 `.s` 檔案

## 🔧 環境需求

- **GPU**: AMD Instinct MI350X (gfx950)
- **OS**: Linux
- **ROCm**: /opt/rocm
- **LLVM**: /home/morhuang/llvm-project/build
- **工具**: llvm-mc, ld.lld, hipcc

## 📈 測試統計

- **總測試數**: 6
- **通過**: 6 (100%)
- **失敗**: 0 (0%)
- **平均測試時間**: ~3 秒/kernel
- **總測試時間**: ~20 秒

## 🏆 主要成就

1. ✅ 找到正確的 `.s` → `.hsaco` 流程
2. ✅ 驗證所有 6 個 kernel 都能執行
3. ✅ 建立完整的測試基礎設施
4. ✅ 提供詳細的文檔和範例
5. ✅ 為 MLIR 開發提供驗證基礎

## 🔮 未來計劃

### 待實作 (需要 MLIR 支援)

- ⚠️ AMDISA dialect 的 MLIR 工具
- ⚠️ `mlir-translate --import-amdisa`
- ⚠️ `mlir-opt --convert-amdisa-to-gpu`
- ⚠️ `mlir-translate --mlir-to-amdisa`
- ⚠️ 完整的端到端測試

### 可能的擴展

- 支援更多 GPU 架構 (gfx90a, gfx940, etc.)
- 添加性能基準測試
- 集成到 CI/CD 流程
- 支援更複雜的 kernel 類型

## 📞 技術支援

### 常見問題

1. **找不到工具**: 確認 LLVM build 路徑
2. **GPU 錯誤**: 檢查 `rocm-smi` 和 `rocminfo`
3. **執行失敗**: 查看 `rebuilt_test.log` 日誌

### 檢查清單

```bash
# 1. 檢查 GPU
rocm-smi

# 2. 檢查 LLVM 工具
llvm-mc --version
ld.lld --version

# 3. 檢查 HIP
hipcc --version

# 4. 執行測試
./test_all_rebuilt_s.sh
```

## 📝 更新日誌

### 2024-12-17
- ✅ 完成目錄重組
- ✅ 實作 `test_asm_to_hsaco.sh`
- ✅ 實作 `test_all_rebuilt_s.sh`
- ✅ 所有測試通過 (6/6)
- ✅ 完成文檔編寫

## 🎯 結論

測試套件已經完全重組並驗證通過。核心流程（從 `.s` 檔案到 GPU 執行）已經確認正確，為 AMDISA dialect 的 MLIR 工具開發提供了可靠的驗證基礎。

---

**文檔版本**: 1.0  
**最後更新**: 2024-12-17  
**維護者**: AMDISA Test Team  
**狀態**: ✅ 生產就緒

