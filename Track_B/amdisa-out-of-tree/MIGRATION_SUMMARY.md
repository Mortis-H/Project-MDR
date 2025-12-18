# AMDISA 從 In-Tree 到 Out-of-Tree 遷移總結

## 📊 遷移對比表

| 項目 | In-Tree | Out-of-Tree |
|------|---------|-------------|
| **專案位置** | `llvm-project/mlir/` 內 | 獨立專案目錄 |
| **建置方式** | 整體 LLVM 建置 | 獨立 CMake 專案 |
| **Include 路徑** | `mlir/Dialect/AMDISA/...` | `AMDISA/...` |
| **CMake 配置** | 內部變數 | `find_package(MLIR)` |
| **建置時間** | 完整 LLVM（數小時） | 僅 AMDISA（數分鐘） |
| **依賴關係** | 自動依賴所有 MLIR | 明確指定所需庫 |

## 🗂️ 檔案對應關係

### Dialect 定義

| In-Tree 路徑 | Out-of-Tree 路徑 |
|-------------|-----------------|
| `mlir/include/mlir/Dialect/AMDISA/IR/AMDISAOps.td` | `include/AMDISA/IR/AMDISAOps.td` |
| `mlir/include/mlir/Dialect/AMDISA/IR/AMDISAOps.h` | `include/AMDISA/IR/AMDISAOps.h` |
| `mlir/include/mlir/Dialect/AMDISA/Passes.h` | `include/AMDISA/Passes.h` |
| `mlir/include/mlir/Dialect/AMDISA/Passes.td` | `include/AMDISA/Passes.td` |

### Dialect 實作

| In-Tree 路徑 | Out-of-Tree 路徑 |
|-------------|-----------------|
| `mlir/lib/Dialect/AMDISA/IR/AMDISAOps.cpp` | `lib/AMDISA/IR/AMDISAOps.cpp` |
| `mlir/lib/Dialect/AMDISA/Transforms/LowerToGPUInlineAsm.cpp` | `lib/AMDISA/Transforms/LowerToGPUInlineAsm.cpp` |

### 工具

| In-Tree 路徑 | Out-of-Tree 路徑 |
|-------------|-----------------|
| `mlir/tools/amdisa-translate/*` | `tools/amdisa-translate/*` |

## 🔧 程式碼修改

### 1. Include 路徑變更

**修改前 (In-Tree)**:
```cpp
#include "mlir/Dialect/AMDISA/IR/AMDISAOps.h"
#include "mlir/Dialect/AMDISA/Passes.h"
```

**修改後 (Out-of-Tree)**:
```cpp
#include "AMDISA/IR/AMDISAOps.h"
#include "AMDISA/Passes.h"
```

### 2. CMakeLists.txt 變更

**In-Tree CMakeLists.txt**:
```cmake
add_mlir_dialect_library(MLIRAMDISA
  AMDISAOps.cpp
  DEPENDS
  MLIRAMDISAIncGen
  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSideEffectInterfaces
)
```

**Out-of-Tree CMakeLists.txt**:
```cmake
add_mlir_dialect_library(MLIRAMDISA
  AMDISAOps.cpp
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/AMDISA
  DEPENDS
  MLIRAMDISAOpsIncGen
  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSideEffectInterfaces
)
```

### 3. 主 CMakeLists.txt

**Out-of-Tree 新增**:
```cmake
cmake_minimum_required(VERSION 3.20.0)
project(amdisa-dialect LANGUAGES CXX C)

# 尋找已安裝的 MLIR
find_package(MLIR REQUIRED CONFIG)

# 設定 include 和 link 目錄
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

# 包含必要的 CMake 模組
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)
```

## 🚀 建置流程變更

### In-Tree 建置流程

```bash
# 1. 配置整個 LLVM/MLIR
cd llvm-project
mkdir build && cd build
cmake -G Ninja ../llvm -DLLVM_ENABLE_PROJECTS="mlir"

# 2. 建置（需要很長時間）
ninja

# 3. 工具位於建置目錄中
./bin/amdisa-translate
```

### Out-of-Tree 建置流程

```bash
# 1. 先建置並安裝 LLVM/MLIR（一次性）
cd llvm-project/build
cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DCMAKE_INSTALL_PREFIX=/path/to/install
ninja install

# 2. 建置 AMDISA 專案（快速）
cd amdisa-out-of-tree
mkdir build && cd build
cmake -G Ninja .. -DMLIR_DIR=/path/to/install/lib/cmake/mlir
ninja

# 3. 工具位於專案建置目錄中
./bin/amdisa-translate
```

## ✅ 遷移檢查清單

- [x] 創建獨立專案目錄結構
- [x] 複製所有原始碼檔案
- [x] 更新 include 路徑
- [x] 創建 out-of-tree CMake 配置
- [x] 設置建置腳本
- [ ] 建置 LLVM/MLIR（如果尚未完成）
- [ ] 測試建置 AMDISA out-of-tree 專案
- [ ] 執行功能測試
- [ ] 更新文檔和註解

## 📝 已完成的自動化腳本

本專案提供了以下自動化腳本：

1. **copy_sources.sh**: 從 in-tree 位置複製所有原始碼
2. **fix_includes.sh**: 自動修正所有 include 路徑
3. **build.sh**: 自動化建置腳本

## 🎯 關鍵優勢總結

### 為什麼要使用 Out-of-Tree？

1. **開發效率**
   - 僅需建置您的專案（秒～分鐘）
   - 不需要重新建置整個 LLVM（小時級別）

2. **版本管理**
   - 獨立的 Git 倉庫
   - 清晰的版本控制
   - 容易追蹤變更

3. **分發與協作**
   - 更容易與團隊分享
   - 可以發佈為獨立套件
   - 不需要完整的 LLVM 原始碼

4. **彈性**
   - 可以支援多個 LLVM 版本
   - 獨立的發布週期
   - 更容易整合到其他專案

5. **清晰的依賴**
   - 明確的 API 邊界
   - 只連結需要的庫
   - 減少建置時間

## ⚠️ 注意事項

### 1. LLVM/MLIR 版本相容性

- 確保使用相容的 LLVM/MLIR 版本
- 建議固定特定版本或提供版本範圍
- API 可能在不同版本間有所變化

### 2. ABI 相容性

- Debug/Release 建置類型必須一致
- 編譯器版本建議保持一致
- C++ 標準版本（C++17）必須匹配

### 3. 連結庫

確保連結了所有必要的 MLIR 庫：
- `MLIRAMDISA`: 您的 Dialect
- `MLIRAMDISATransforms`: 您的 Passes
- `MLIRIR`: MLIR 核心
- `MLIRParser`: MLIR 解析器
- `MLIRPass`: Pass 基礎設施
- `MLIRGPUDialect`: GPU Dialect
- `MLIRLLVMDialect`: LLVM Dialect

## 🔗 相關資源

- **MLIR Standalone Example**: 官方 out-of-tree 範例
- **專案位置**: `/home/morhuang/Project-MDR/Track_B/amdisa-out-of-tree`
- **原始 In-Tree 位置**: `/home/morhuang/Project-MDR/Track_B/llvm-project/mlir`

## 📞 下一步

1. **完成 LLVM/MLIR 建置**（如果尚未完成）
   ```bash
   cd /home/morhuang/Project-MDR/Track_B/llvm-project
   mkdir build && cd build
   cmake -G Ninja ../llvm \
       -DLLVM_ENABLE_PROJECTS="mlir" \
       -DLLVM_TARGETS_TO_BUILD="X86;AMDGPU" \
       -DCMAKE_INSTALL_PREFIX=../install \
       -DLLVM_INSTALL_UTILS=ON
   ninja install
   ```

2. **更新建置腳本路徑**
   ```bash
   nano /home/morhuang/Project-MDR/Track_B/amdisa-out-of-tree/build.sh
   # 修改 MLIR_DIR 和 LLVM_DIR 變數
   ```

3. **建置 AMDISA 專案**
   ```bash
   cd /home/morhuang/Project-MDR/Track_B/amdisa-out-of-tree
   ./build.sh
   ```

4. **測試工具**
   ```bash
   ./build/bin/amdisa-translate --help
   ./build/bin/amdisa-translate -x s tools/amdisa-translate/source/test.s -emit mlir
   ```

---

**遷移完成！** 🎉

您現在擁有一個完全獨立的 AMDISA out-of-tree 專案，可以獨立開發、建置和分發。

