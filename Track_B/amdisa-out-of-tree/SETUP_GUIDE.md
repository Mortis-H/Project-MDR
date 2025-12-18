# AMDISA Out-of-Tree 專案設置指南

本文檔詳細說明如何設置和建置 AMDISA out-of-tree 專案。

## 📋 目錄結構

專案已經按照 MLIR out-of-tree 專案的標準結構組織：

```
amdisa-out-of-tree/
├── CMakeLists.txt              # 主 CMake 配置
├── README.md                   # 專案說明
├── SETUP_GUIDE.md              # 本檔案
├── build.sh                    # 自動建置腳本
├── copy_sources.sh             # 原始碼複製腳本
├── fix_includes.sh             # Include 路徑修正腳本
│
├── include/AMDISA/             # 公開標頭檔
│   ├── IR/
│   │   ├── AMDISAOps.h        # Dialect C++ 介面
│   │   ├── AMDISAOps.td       # TableGen 定義
│   │   └── CMakeLists.txt
│   ├── Passes.h                # Pass 介面
│   ├── Passes.td               # Pass TableGen 定義
│   └── CMakeLists.txt
│
├── lib/AMDISA/                 # 實作
│   ├── IR/
│   │   ├── AMDISAOps.cpp      # Dialect 實作
│   │   └── CMakeLists.txt
│   ├── Transforms/
│   │   ├── LowerToGPUInlineAsm.cpp  # Pass 實作
│   │   └── CMakeLists.txt
│   └── CMakeLists.txt
│
└── tools/amdisa-translate/     # 轉換工具
    ├── amdisa-translate.cpp
    ├── AMDISAAsmParser.cpp
    ├── AMDISAAsmParser.h
    ├── AMDGCNAssembly.cpp
    ├── AMDGCNAssembly.h
    ├── AMDGPUMetadata.cpp
    ├── AMDGPUMetadata.h
    ├── parse_utils.cpp
    ├── parse_utils.h
    ├── ParsedProgram.h
    ├── CMakeLists.txt
    └── source/                 # 測試用 .s 檔案
```

## 🔧 前置需求

### 1. 建置並安裝 LLVM/MLIR

Out-of-tree 專案需要先有一個已建置（最好是已安裝）的 LLVM/MLIR。

#### 選項 A: 建置並安裝 LLVM/MLIR（推薦）

```bash
cd /home/morhuang/Project-MDR/Track_B/llvm-project

# 創建建置目錄
mkdir -p build
cd build

# 配置 CMake（包含必要的選項）
cmake -G Ninja ../llvm \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_TARGETS_TO_BUILD="X86;AMDGPU" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_INSTALL_PREFIX=/home/morhuang/Project-MDR/Track_B/llvm-install \
    -DLLVM_INSTALL_UTILS=ON

# 建置（這可能需要一些時間）
ninja

# 安裝到指定目錄
ninja install
```

#### 選項 B: 使用現有的建置目錄（不安裝）

如果不想安裝，也可以直接使用建置目錄：

```bash
# 確保 LLVM/MLIR 已經建置
cd /home/morhuang/Project-MDR/Track_B/llvm-project/build
ninja
```

## 🚀 建置 AMDISA Out-of-Tree 專案

### 方法 1: 使用自動建置腳本

修改 `build.sh` 中的路徑設置：

```bash
# 編輯 build.sh
nano /home/morhuang/Project-MDR/Track_B/amdisa-out-of-tree/build.sh

# 根據您的安裝方式，更新以下變數：

# 如果使用選項 A（已安裝）：
LLVM_INSTALL_DIR="/home/morhuang/Project-MDR/Track_B/llvm-install"
MLIR_DIR="$LLVM_INSTALL_DIR/lib/cmake/mlir"
LLVM_DIR="$LLVM_INSTALL_DIR/lib/cmake/llvm"
LLVM_EXTERNAL_LIT="$LLVM_INSTALL_DIR/bin/llvm-lit"

# 如果使用選項 B（使用建置目錄）：
LLVM_BUILD_DIR="/home/morhuang/Project-MDR/Track_B/llvm-project/build"
MLIR_DIR="$LLVM_BUILD_DIR/lib/cmake/mlir"
LLVM_DIR="$LLVM_BUILD_DIR/lib/cmake/llvm"
LLVM_EXTERNAL_LIT="$LLVM_BUILD_DIR/bin/llvm-lit"
```

然後執行：

```bash
cd /home/morhuang/Project-MDR/Track_B/amdisa-out-of-tree
./build.sh
```

### 方法 2: 手動建置

```bash
cd /home/morhuang/Project-MDR/Track_B/amdisa-out-of-tree

# 創建並進入建置目錄
mkdir -p build
cd build

# 配置（使用已安裝的 LLVM/MLIR）
cmake -G Ninja .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DMLIR_DIR=/home/morhuang/Project-MDR/Track_B/llvm-install/lib/cmake/mlir \
    -DLLVM_DIR=/home/morhuang/Project-MDR/Track_B/llvm-install/lib/cmake/llvm \
    -DLLVM_EXTERNAL_LIT=/home/morhuang/Project-MDR/Track_B/llvm-install/bin/llvm-lit

# 或者配置（使用建置目錄）
cmake -G Ninja .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DMLIR_DIR=/home/morhuang/Project-MDR/Track_B/llvm-project/build/lib/cmake/mlir \
    -DLLVM_DIR=/home/morhuang/Project-MDR/Track_B/llvm-project/build/lib/cmake/llvm \
    -DLLVM_EXTERNAL_LIT=/home/morhuang/Project-MDR/Track_B/llvm-project/build/bin/llvm-lit

# 建置
cmake --build . -j$(nproc)
```

## 📝 建置成功後

建置成功後，您會得到：

```
build/
├── bin/
│   └── amdisa-translate    # 可執行檔
└── lib/
    ├── libMLIRAMDISA.a
    └── libMLIRAMDISATransforms.a
```

## 🧪 測試工具

```bash
# 查看幫助訊息
./build/bin/amdisa-translate --help

# 將 .s 檔案轉換為 MLIR
./build/bin/amdisa-translate -x s tools/amdisa-translate/source/test.s -emit mlir

# 降低為 GPU inline assembly
./build/bin/amdisa-translate -x s tools/amdisa-translate/source/test.s -emit gpu
```

## 🔄 與 In-Tree 版本的主要差異

### 1. **Include 路徑**

- **In-Tree**: `#include "mlir/Dialect/AMDISA/IR/AMDISAOps.h"`
- **Out-of-Tree**: `#include "AMDISA/IR/AMDISAOps.h"`

### 2. **CMake 配置**

- **In-Tree**: 使用 LLVM 內部的 CMake 變數和函數
- **Out-of-Tree**: 使用 `find_package(MLIR)` 來尋找已安裝的 MLIR

### 3. **建置方式**

- **In-Tree**: 作為 LLVM/MLIR 專案的一部分建置
- **Out-of-Tree**: 獨立專案，可以單獨建置和發佈

### 4. **依賴管理**

- **In-Tree**: 自動依賴於所有 MLIR dialect
- **Out-of-Tree**: 明確指定所需的 MLIR 庫

## 🎯 優勢

使用 out-of-tree 建置的優勢：

1. **獨立開發**: 不需要修改 LLVM/MLIR 原始碼
2. **版本控制**: 可以獨立管理專案版本
3. **分發**: 更容易與他人分享和分發
4. **快速迭代**: 只需重新建置您的專案，不需要重新建置整個 LLVM
5. **多版本**: 可以針對不同版本的 LLVM/MLIR 進行建置

## ❗ 常見問題

### Q1: CMake 找不到 MLIR

**A**: 確認 `MLIRConfig.cmake` 的位置：

```bash
find /home/morhuang/Project-MDR/Track_B/llvm-project -name "MLIRConfig.cmake" 2>/dev/null
```

然後將路徑設定為包含該檔案的目錄。

### Q2: 連結錯誤

**A**: 確保 LLVM/MLIR 和您的專案使用相同的建置類型（Debug/Release）：

```bash
# 在兩個專案中都使用 Release
cmake .. -DCMAKE_BUILD_TYPE=Release ...
```

### Q3: 找不到 AMDGPU 相關的符號

**A**: 確保 LLVM 建置時包含了 AMDGPU target：

```bash
cmake ... -DLLVM_TARGETS_TO_BUILD="X86;AMDGPU"
```

### Q4: TableGen 錯誤

**A**: 確保 LLVM 建置時啟用了 `LLVM_INSTALL_UTILS`：

```bash
cmake ... -DLLVM_INSTALL_UTILS=ON
```

## 📚 參考資料

- [MLIR Standalone Example](https://github.com/llvm/llvm-project/tree/main/mlir/examples/standalone)
- [MLIR Documentation](https://mlir.llvm.org/)
- [LLVM CMake Documentation](https://llvm.org/docs/CMake.html)

## 🔧 後續步驟

1. **擴展 Dialect**: 在 `include/AMDISA/IR/AMDISAOps.td` 中新增操作
2. **新增 Pass**: 在 `lib/AMDISA/Transforms/` 中實作新的轉換
3. **測試**: 在 `test/` 目錄中新增測試案例
4. **文檔**: 使用 `mlir-tblgen` 生成文檔

---

如有問題，請檢查：
- LLVM/MLIR 是否已正確建置
- CMake 版本是否 >= 3.20
- 路徑設置是否正確

