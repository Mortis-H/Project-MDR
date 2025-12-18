# AMDISA Dialect - Out-of-Tree MLIR Project

這是一個獨立的 (out-of-tree) MLIR Dialect 專案，包含 AMDISA Dialect 以及 `amdisa-translate` 工具。

## 專案說明

AMDISA Dialect 提供了一個通用的 wrapper 來表示從 AMD GPU 組合語言 (`.s` 檔案) 解析出來的指令。此專案包含：

- **AMDISA Dialect**: 自定義的 MLIR dialect，包含 `amdisa.inst` 和 `amdisa.label` 操作
- **amdisa-translate**: 轉換工具，可以將 `.s` 檔案轉換為 AMDISA Dialect，並可進一步降低為 GPU inline assembly

## 建置需求

- LLVM/MLIR 已安裝的版本
- CMake 3.20 或更新版本
- Ninja (建議) 或其他建置系統
- C++17 編譯器

## 建置步驟 - Component Build (推薦)

假設您已經將 LLVM 和 MLIR 建置在 `$BUILD_DIR` 並安裝到 `$PREFIX`：

```bash
# 1. 創建建置目錄
mkdir build && cd build

# 2. 配置 CMake
cmake -G Ninja .. \
    -DMLIR_DIR=$PREFIX/lib/cmake/mlir \
    -DLLVM_EXTERNAL_LIT=$BUILD_DIR/bin/llvm-lit

# 3. 建置專案
cmake --build .

# 4. 執行工具
./bin/amdisa-translate --help
```

### 範例使用 LLVM/MLIR 安裝路徑

如果您的 LLVM/MLIR 安裝在標準位置：

```bash
# 使用系統安裝的 LLVM/MLIR
cmake -G Ninja .. \
    -DMLIR_DIR=/usr/local/lib/cmake/mlir \
    -DLLVM_DIR=/usr/local/lib/cmake/llvm

# 或者，如果從 llvm-project 建置目錄安裝
cmake -G Ninja .. \
    -DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir \
    -DLLVM_DIR=/path/to/llvm-project/build/lib/cmake/llvm \
    -DLLVM_EXTERNAL_LIT=/path/to/llvm-project/build/bin/llvm-lit
```

## 使用 amdisa-translate

將 AMD 組合語言轉換為 MLIR：

```bash
# 解析 .s 檔案並輸出 AMDISA Dialect
./bin/amdisa-translate -x s input.s -emit mlir

# 降低為 GPU inline assembly
./bin/amdisa-translate -x s input.s -emit gpu
```

## 專案結構

```
amdisa-out-of-tree/
├── CMakeLists.txt              # 主 CMake 配置
├── README.md                   # 本檔案
├── include/AMDISA/             # Public headers
│   ├── IR/                     # Dialect 定義
│   └── Passes.h                # Pass 介面
├── lib/AMDISA/                 # 實作
│   ├── IR/                     # Dialect 實作
│   └── Transforms/             # Pass 實作
└── tools/amdisa-translate/     # 轉換工具
```

## 開發注意事項

### 與 In-Tree 版本的差異

1. **Include 路徑**: 從 `mlir/Dialect/AMDISA/...` 改為 `AMDISA/...`
2. **獨立建置**: 不再依賴 LLVM/MLIR 原始碼樹
3. **CMake 配置**: 使用 `find_package(MLIR)` 而非內部 CMake 變數
4. **連結庫**: 明確指定所需的 MLIR 和 LLVM 庫

### 更新 Dialect

修改 `include/AMDISA/IR/AMDISAOps.td` 後，需要重新建置以生成新的 C++ 程式碼。

## 疑難排解

### CMake 找不到 MLIR

確保正確設定 `MLIR_DIR` 指向包含 `MLIRConfig.cmake` 的目錄：

```bash
find / -name "MLIRConfig.cmake" 2>/dev/null
```

### 連結錯誤

確保 LLVM/MLIR 是以相同的建置類型 (Debug/Release) 建置的：

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DMLIR_DIR=...
```

## 授權

本專案基於 LLVM 專案授權。詳見 LLVM LICENSE.TXT。

