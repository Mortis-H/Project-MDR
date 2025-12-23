# amdisa-opt 工具整合說明

## 概述

`amdisa-opt` 是一個自定義的 MLIR 優化工具，實現了 **方案 A：構建自定義的 mlir-opt**（詳見 `DIALECT_REGISTRATION_EXPLAINED.md`）。

## 為什麼需要 amdisa-opt？

### 問題背景

在 out-of-tree 專案中：
- ❌ 標準的 `mlir-opt` 不認識 AMDISA dialect
- ❌ 無法通過命令行使用 AMDISA passes
- ❌ 不便於調試和實驗 AMDISA transformations

### 解決方案

創建 `amdisa-opt`，它：
- ✅ 包含所有標準 MLIR dialects 和 passes
- ✅ 額外註冊了 AMDISA dialect
- ✅ 額外註冊了 AMDISA passes
- ✅ 提供與 `mlir-opt` 相同的命令行接口

## 工具架構

```
amdisa-opt
    │
    ├─ 註冊所有標準 MLIR dialects
    │   └─ mlir::registerAllDialects()
    │
    ├─ 註冊所有標準 MLIR passes
    │   └─ mlir::registerAllPasses()
    │
    ├─ 註冊 AMDISA dialect
    │   └─ registry.insert<mlir::amdisa::AMDISADialect>()
    │
    ├─ 註冊 AMDISA passes
    │   └─ mlir::amdisa::registerAMDISAPasses()
    │
    └─ 運行 mlir-opt 主邏輯
        └─ mlir::MlirOptMain()
```

## 目錄結構

```
amdisa-toolkit/
└── tools/
    ├── amdisa-translate/        # 專用的 ISA → MLIR 轉換工具
    │   ├── amdisa-translate.cpp
    │   └── CMakeLists.txt
    │
    └── amdisa-opt/              # 通用的 MLIR 優化工具 ⭐ 新增
        ├── amdisa-opt.cpp
        ├── CMakeLists.txt
        └── README.md
```

## 實現細節

### 源文件：amdisa-opt.cpp

```cpp
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "mlir/Dialect/AMDISA/IR/AMDISAOps.h"
#include "mlir/Dialect/AMDISA/Passes.h"

int main(int argc, char **argv) {
  // 1. 註冊所有標準 passes
  mlir::registerAllPasses();
  
  // 2. 註冊 AMDISA passes
  mlir::amdisa::registerAMDISAPasses();
  
  // 3. 創建 dialect registry
  mlir::DialectRegistry registry;
  
  // 4. 註冊所有標準 dialects
  mlir::registerAllDialects(registry);
  
  // 5. 註冊 AMDISA dialect
  registry.insert<mlir::amdisa::AMDISADialect>();
  
  // 6. 運行 mlir-opt 主邏輯
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "AMDISA optimizer driver\n", registry));
}
```

### CMake 配置：CMakeLists.txt

```cmake
# 收集所有已註冊的 dialect 庫
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

# 定義工具
add_mlir_tool(amdisa-opt
  amdisa-opt.cpp
  
  DEPENDS
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}
  MLIRAMDISAIncGen
  MLIRAMDISAPassIncGen
)

# 鏈接庫
target_link_libraries(amdisa-opt
  PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}
  
  MLIRAMDISA
  MLIRAMDISATransforms
  
  MLIROptLib
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIRTransforms
)
```

## 與其他工具的對比

| 工具 | 用途 | 輸入 | 輸出 | AMDISA 支援 | 靈活性 |
|------|------|------|------|------------|--------|
| **amdisa-translate** | ISA → MLIR 轉換 | .s, .mlir | .mlir | ✅ 內建 | ⭐ 固定 pipeline |
| **amdisa-opt** | MLIR 優化與轉換 | .mlir | .mlir | ✅ 內建 | ⭐⭐⭐ 任意 pass 組合 |
| **mlir-opt** | MLIR 優化（標準） | .mlir | .mlir | ❌ 不支援 | ⭐⭐⭐ 標準 passes |
| **pipeline.py** | 端到端工具鏈 | .s | .hsaco | ✅ 內建 | ⭐⭐ 預設流程 |

## 使用場景

### 場景 1：調試 AMDISA lowering pass

```bash
# 使用 amdisa-translate（固定 pipeline）
amdisa-translate -x s -emit=gpu kernel.s

# 使用 amdisa-opt（靈活控制）
amdisa-translate -x s -emit=mlir kernel.s | \
  amdisa-opt --mlir-print-ir-before-all --mlir-print-ir-after-all \
    --amdisa-lower-to-gpu-inline-asm
```

**優勢：** 可以看到 pass 執行前後的 IR 變化，便於調試。

### 場景 2：實驗新的 pass 組合

```bash
# 嘗試不同的 pass 順序
amdisa-opt --pass-pipeline="builtin.module(
  amdisa-lower-to-gpu-inline-asm,
  gpu-kernel-outlining,
  inline,
  cse
)" input.mlir
```

**優勢：** 快速實驗不同的優化策略。

### 場景 3：與其他 MLIR dialects 整合

```bash
# AMDISA → GPU → LLVM
amdisa-opt --pass-pipeline="builtin.module(
  amdisa-lower-to-gpu-inline-asm,
  gpu-kernel-outlining,
  convert-gpu-to-llvm
)" input.mlir
```

**優勢：** 靈活整合到更大的 MLIR 工作流程。

### 場景 4：驗證 MLIR 的正確性

```bash
# 只驗證，不執行任何 pass
amdisa-opt --verify-diagnostics input.mlir

# 或
amdisa-opt --verify-each --pass-pipeline="..." input.mlir
```

**優勢：** 確保每個轉換步驟的 IR 都是有效的。

## 構建與使用

### 構建

```bash
cd /home/andycha/workspaces/Project-MDR/Track_B/amdisa-toolkit

cmake -B build \
  -DMLIR_DIR=/path/to/llvm/lib/cmake/mlir \
  -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm \
  -G Ninja

ninja -C build amdisa-opt
```

### 驗證

```bash
# 檢查可執行文件
ls -lh build/bin/amdisa-opt

# 顯示幫助
./build/bin/amdisa-opt --help

# 列出所有 passes（應該能看到 amdisa-lower-to-gpu-inline-asm）
./build/bin/amdisa-opt --help-list-hidden | grep -i amdisa
```

### 基本使用

```bash
# 查看 MLIR 文件（不執行任何轉換）
./build/bin/amdisa-opt input.mlir

# 執行 AMDISA lowering pass
./build/bin/amdisa-opt --amdisa-lower-to-gpu-inline-asm input.mlir

# 使用 pass pipeline 語法（更精確）
./build/bin/amdisa-opt \
  --pass-pipeline="builtin.module(amdisa-lower-to-gpu-inline-asm)" \
  input.mlir
```

## 工作流程範例

### 完整的轉換流程

```bash
# 步驟 1: AMD ISA (.s) → AMDISA MLIR
amdisa-translate -x s -emit=mlir kernel.s -o step1_amdisa.mlir

# 步驟 2: 檢查生成的 AMDISA MLIR
amdisa-opt --verify-diagnostics step1_amdisa.mlir

# 步驟 3: AMDISA MLIR → GPU MLIR
amdisa-opt --amdisa-lower-to-gpu-inline-asm \
  step1_amdisa.mlir -o step2_gpu.mlir

# 步驟 4: 繼續使用標準 MLIR 工具
mlir-opt --gpu-kernel-outlining step2_gpu.mlir -o step3_outlined.mlir

# 步驟 5: GPU MLIR → LLVM Dialect
mlir-opt --convert-gpu-to-llvm step3_outlined.mlir -o step4_llvm.mlir

# 步驟 6: LLVM Dialect → LLVM IR
mlir-translate --mlir-to-llvmir step4_llvm.mlir -o step5.ll
```

### 或者使用 pipeline 一次完成多步

```bash
amdisa-opt --pass-pipeline="builtin.module(
  amdisa-lower-to-gpu-inline-asm,
  gpu-kernel-outlining,
  convert-gpu-to-llvm
)" step1_amdisa.mlir -o step4_llvm.mlir
```

## 與 pipeline.py 的關係

`amdisa-opt` **不會替代** `pipeline.py`，它們是互補的：

- **pipeline.py**: 
  - 端到端自動化工具鏈
  - 適合生產環境
  - 固定流程，易於使用
  
- **amdisa-opt**:
  - 靈活的 MLIR 優化工具
  - 適合開發和調試
  - 可自定義 pass 組合

**典型用法：**
1. 使用 `amdisa-opt` 開發和調試新的 passes
2. 驗證通過後，集成到 `pipeline.py` 中供用戶使用

## 擴展 amdisa-opt

### 添加新的 AMDISA Pass

1. **定義 Pass**（在 `Passes.td`）：
```tablegen
def MyNewAMDISAPass : Pass<"my-new-amdisa-pass", "mlir::ModuleOp"> {
  let summary = "My new AMDISA transformation";
  let constructor = "mlir::amdisa::createMyNewPass()";
}
```

2. **實現 Pass**（在 `lib/Dialect/AMDISA/Transforms/MyNewPass.cpp`）：
```cpp
namespace mlir {
namespace amdisa {
std::unique_ptr<Pass> createMyNewPass() {
  return std::make_unique<MyNewPassImpl>();
}
} // namespace amdisa
} // namespace mlir
```

3. **重新構建**：
```bash
ninja -C build amdisa-opt
```

4. **使用新 Pass**：
```bash
amdisa-opt --my-new-amdisa-pass input.mlir
```

**無需修改 `amdisa-opt.cpp`！** Pass 註冊是自動的。

## 性能注意事項

### 啟動時間

```bash
# 測量啟動時間
time amdisa-opt --help > /dev/null
# 通常 < 0.5 秒
```

由於需要加載所有 MLIR dialects，啟動時間比 `amdisa-translate` 稍長，但對於調試和開發來說完全可接受。

### 批量處理

```bash
# 不推薦：多次調用
for file in *.mlir; do
  amdisa-opt --pass-pipeline="..." $file -o ${file%.mlir}_opt.mlir
done

# 推薦：使用腳本或 pipeline
python batch_optimize.py
```

## 疑難排解

### 問題 1: Pass 未註冊

**症狀：**
```
error: 'amdisa-lower-to-gpu-inline-asm' does not refer to a registered pass
```

**原因：** Pass 未正確註冊或名稱錯誤。

**檢查：**
```bash
# 列出所有已註冊的 passes
amdisa-opt --help-list-hidden | grep -i amdisa

# 檢查 Passes.td 中的 pass 名稱
cat include/Dialect/AMDISA/Passes.td
```

**解決：** 確保 `Passes.td` 中的 pass 定義正確，並重新構建。

### 問題 2: Dialect 未加載

**症狀：**
```
error: Dialect `amdisa` not found
```

**原因：** AMDISA dialect 未正確註冊。

**檢查：**
```cpp
// 確認 amdisa-opt.cpp 中有：
registry.insert<mlir::amdisa::AMDISADialect>();
```

**解決：** 確保 dialect 註冊代碼存在，並重新構建。

### 問題 3: 鏈接錯誤

**症狀：**
```
undefined reference to `mlir::amdisa::registerAMDISAPasses()'
```

**原因：** CMakeLists.txt 中未鏈接 `MLIRAMDISATransforms`。

**檢查：**
```cmake
# 確認 CMakeLists.txt 中有：
target_link_libraries(amdisa-opt
  PRIVATE
  MLIRAMDISATransforms  # ← 必須
)
```

**解決：** 添加缺失的庫依賴，並重新配置和構建。

## 總結

### ✅ 已實現的功能

- ✅ 完整的 `amdisa-opt` 工具實現
- ✅ 支援所有標準 MLIR dialects 和 passes
- ✅ 支援 AMDISA dialect 和 passes
- ✅ 與標準 `mlir-opt` 完全兼容的命令行接口
- ✅ 完整的 CMake 配置
- ✅ 使用說明和範例

### 🎯 使用建議

1. **日常開發**：使用 `amdisa-translate` 進行基本轉換
2. **調試和實驗**：使用 `amdisa-opt` 進行詳細分析
3. **生產環境**：使用 `pipeline.py` 進行自動化構建
4. **學習 MLIR**：使用 `amdisa-opt` 探索不同的 passes

### 📚 相關文檔

- `DIALECT_REGISTRATION_EXPLAINED.md` - Dialect 註冊機制詳解
- `tools/amdisa-opt/README.md` - amdisa-opt 詳細使用指南
- `OUTOFTREE_MIGRATION_GUIDE.md` - Out-of-tree 遷移指南
- `TESTING_OUTOFTREE_BUILD.md` - 構建測試指南

---

🎉 **現在您有了一個完整的 AMDISA MLIR 工具鏈！**

