# 🚀 AMDISA Out-of-Tree 快速開始

## ⚡ 快速總覽

此專案已經從 **in-tree** 完整遷移為 **out-of-tree** 結構！所有檔案都已正確複製和配置。

## 📦 專案已包含

✅ 完整的專案結構  
✅ 所有原始碼檔案（從 in-tree 複製）  
✅ Include 路徑已修正  
✅ CMake 配置檔案  
✅ 自動化建置腳本  
✅ 詳細文檔  

## 🎯 三步驟建置

### 步驟 1: 建置 LLVM/MLIR（如果尚未完成）

```bash
cd /home/morhuang/Project-MDR/Track_B/llvm-project

# 創建建置目錄
mkdir -p build install
cd build

# 配置
cmake -G Ninja ../llvm \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_TARGETS_TO_BUILD="X86;AMDGPU" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_INSTALL_PREFIX=/home/morhuang/Project-MDR/Track_B/llvm-project/install \
    -DLLVM_INSTALL_UTILS=ON

# 建置並安裝（需要一些時間）
ninja -j$(nproc)
ninja install
```

### 步驟 2: 更新建置腳本

```bash
# 編輯 build.sh，將路徑指向您的 LLVM 安裝目錄
nano /home/morhuang/Project-MDR/Track_B/amdisa-out-of-tree/build.sh

# 更新這些行：
# LLVM_BUILD_DIR="/home/morhuang/Project-MDR/Track_B/llvm-project/install"
# MLIR_DIR="$LLVM_BUILD_DIR/lib/cmake/mlir"
# LLVM_DIR="$LLVM_BUILD_DIR/lib/cmake/llvm"
# LLVM_EXTERNAL_LIT="$LLVM_BUILD_DIR/bin/llvm-lit"
```

### 步驟 3: 建置 AMDISA 專案

```bash
cd /home/morhuang/Project-MDR/Track_B/amdisa-out-of-tree

# 執行建置腳本
./build.sh

# 或手動建置
mkdir build && cd build
cmake -G Ninja .. \
    -DMLIR_DIR=/home/morhuang/Project-MDR/Track_B/llvm-project/install/lib/cmake/mlir \
    -DLLVM_DIR=/home/morhuang/Project-MDR/Track_B/llvm-project/install/lib/cmake/llvm
ninja
```

## 🧪 測試

```bash
cd /home/morhuang/Project-MDR/Track_B/amdisa-out-of-tree/build

# 執行工具
./bin/amdisa-translate --help

# 測試範例
./bin/amdisa-translate -x s ../tools/amdisa-translate/source/test.s -emit mlir
```

## 📚 文檔

- **README.md**: 專案概述和基本使用
- **SETUP_GUIDE.md**: 詳細的設置指南和問題排解
- **MIGRATION_SUMMARY.md**: In-tree 到 Out-of-tree 的遷移對比
- **QUICKSTART.md**: 本檔案，快速開始指南

## 🗂️ 專案結構

```
amdisa-out-of-tree/
├── 📄 CMakeLists.txt           # 主 CMake 配置
├── 📄 README.md                # 專案說明
├── 📄 SETUP_GUIDE.md           # 詳細設置指南
├── 📄 MIGRATION_SUMMARY.md     # 遷移對比
├── 📄 QUICKSTART.md            # 本檔案
│
├── 🔧 build.sh                 # 自動建置腳本
├── 🔧 copy_sources.sh          # 原始碼複製腳本
├── 🔧 fix_includes.sh          # Include 路徑修正腳本
│
├── 📁 include/AMDISA/          # 公開標頭檔
│   ├── IR/                     # Dialect 定義
│   └── Passes.h                # Pass 介面
│
├── 📁 lib/AMDISA/              # 實作
│   ├── IR/                     # Dialect 實作
│   └── Transforms/             # Pass 實作
│
└── 📁 tools/amdisa-translate/  # 轉換工具
    ├── amdisa-translate.cpp
    ├── AMDISAAsmParser.*
    ├── AMDGCNAssembly.*
    └── source/                 # 測試 .s 檔案
```

## 🎯 關鍵變更

### In-Tree → Out-of-Tree

| 方面 | In-Tree | Out-of-Tree |
|------|---------|-------------|
| 位置 | `llvm-project/mlir/` | 獨立目錄 |
| Include | `mlir/Dialect/AMDISA/...` | `AMDISA/...` |
| 建置 | 隨 LLVM 一起 | 獨立建置 |
| 時間 | 數小時 | 數分鐘 |

## ⚠️ 如果遇到問題

### CMake 找不到 MLIR

```bash
# 尋找 MLIRConfig.cmake
find /home/morhuang/Project-MDR/Track_B/llvm-project -name "MLIRConfig.cmake"

# 使用找到的路徑更新 build.sh
```

### 建置錯誤

1. 確認 LLVM/MLIR 已正確建置和安裝
2. 確認建置類型一致（都使用 Release 或 Debug）
3. 檢查 LLVM 是否包含 AMDGPU target
4. 查看 SETUP_GUIDE.md 中的疑難排解章節

### 連結錯誤

確保 LLVM 建置時包含：
- AMDGPU target: `-DLLVM_TARGETS_TO_BUILD="X86;AMDGPU"`
- Install utils: `-DLLVM_INSTALL_UTILS=ON`

## 💡 下一步

1. ✅ 專案結構已建立
2. ⏳ 建置 LLVM/MLIR（如需要）
3. ⏳ 建置 AMDISA 專案
4. ⏳ 執行測試
5. 🎉 開始開發！

## 📞 需要幫助？

查看詳細文檔：
- 設置問題 → `SETUP_GUIDE.md`
- 遷移細節 → `MIGRATION_SUMMARY.md`
- 一般使用 → `README.md`

---

**專案已準備就緒！** 只需完成 LLVM/MLIR 建置，然後執行 `./build.sh` 即可。🚀

