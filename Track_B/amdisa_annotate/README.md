# AMD GCN ISA Parser

AMD GCN 組合語言解析工具，提供完整的 in-memory API 供其他專案使用。

## 專案概述

本專案提供兩個主要工具：

1. **amdisa_annotate** - 組合語言標註工具，可解析並標註 `.s` 檔案
2. **api_usage_demo** - API 使用範例，展示如何在程式中使用解析 API

## 快速開始

### 編譯

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

編譯完成後會生成：
- `build/amdisa_annotate` - 標註工具
- `build/api_usage_demo` - API 示範程式

### 基本使用

#### 1. 標註組合語言檔案

```bash
./build/amdisa_annotate input.s > output.txt
```

輸出範例：
```
[Kernel Name]        .globl  vec_add
[Label]              vec_add:
[Instruction]        s_load_dwordx2 s[50:51], s[0:1], 0x0
[Instruction]        s_load_dwordx4 s[20:23], s[0:1], 0x10
[Comment]            ; %bb.0:
[Directive]          .text
```

#### 2. 顯示 Metadata 資訊

```bash
./build/amdisa_annotate --show-metadata input.s 2>&1
```

顯示 AMD GPU Metadata，包含 kernel 資訊、資源使用量、參數定義等。

#### 3. 使用 API Demo

```bash
./build/api_usage_demo input.s
```

展示如何使用 API 查詢：
- Metadata 資訊（kernel 名稱、SGPR/VGPR 數量、參數列表）
- Labels 列表
- Instructions 列表
- Label Blocks（Label 和 Instructions 的對應關係）

## 在其他專案中使用

### 整合方式

若要在其他專案中使用此 parser，需要以下檔案：

**核心檔案：**
```
src/ParsedProgram.h       - 數據結構定義
src/AMDGPUMetadata.h      - Metadata 數據結構
src/AMDGPUMetadata.cpp    - Metadata 解析實作
src/AMDGCNAssembly.h      - 統一 API 接口
src/AMDGCNAssembly.cpp    - API 實作
src/parse_utils.h         - 解析工具函數
src/parse_utils.cpp       - 解析實作
```

### API 使用範例

```cpp
#include "parse_utils.h"
#include "AMDGCNAssembly.h"

// 1. 解析檔案
AMDGCNAssembly assembly = parseAMDGCNAssembly("input.s");

// 2. 查詢 Metadata
if (assembly.hasMetadata()) {
  const AMDGPUMetadata &meta = assembly.getMetadata();
  for (const auto &kernel : meta.kernels) {
    std::cout << "Kernel: " << kernel.name << "\n";
    std::cout << "VGPR: " << kernel.vgprCount << "\n";
    std::cout << "SGPR: " << kernel.sgprCount << "\n";
  }
}

// 3. 查詢 Labels
const auto &labels = assembly.getAllLabels();
for (const auto &label : labels) {
  std::cout << "Label: " << label.name << " at line " << label.lineNumber << "\n";
}

// 4. 查詢 Instructions
const auto &insts = assembly.getAllInstructions();
for (const auto &inst : insts) {
  std::cout << inst.opcode << " ";
  for (const auto &op : inst.operands) {
    std::cout << op.text << " ";
  }
  std::cout << "\n";
}

// 5. 查詢 Label Blocks（Label 與 Instructions 的對應）
const auto &blocks = assembly.getLabelBlocks();
for (const auto &block : blocks) {
  std::cout << "Label: " << block.labelName << "\n";
  std::cout << "  Range: Line " << block.startLine << "-" << block.endLine << "\n";
  std::cout << "  Instructions: " << block.instructions.size() << "\n";
}
```

完整 API 文件請參考 `docs/API_DOCUMENTATION.md`。

## 工具選項

### amdisa_annotate

```bash
# 顯示操作數類型
./build/amdisa_annotate --show-types input.s

# 隱藏註解行
./build/amdisa_annotate --hide-comments input.s

# 只顯示 Metadata
./build/amdisa_annotate --metadata-only input.s

# 顯示統計資訊
./build/amdisa_annotate --summary input.s 2>&1

# 指定目標架構
./build/amdisa_annotate --triple=amdgcn-amd-amdhsa --mcpu=gfx950 input.s
```

## 專案結構

```
amdisa_annotate/
├── src/                        # 核心源碼
│   ├── main.cpp                # 主程式 (amdisa_annotate)
│   ├── parse_utils.h/cpp       # 可重用的解析函數
│   ├── ParsedProgram.h         # 數據結構定義
│   ├── AMDGPUMetadata.h/cpp    # Metadata 解析器
│   └── AMDGCNAssembly.h/cpp    # 統一 API
├── examples/                   # 範例程式
│   └── api_usage_demo.cpp      # API 使用示範
├── docs/                       # 文件
│   ├── API_DOCUMENTATION.md    # 完整 API 文件
│   └── HOW_TO_USE_API.md       # API 使用指南
├── CMakeLists.txt              # 構建配置
└── README.md                   # 本文件
```

## 技術細節

### 使用的 LLVM 組件

- **MCParser** - 組合語言解析器
- **MCStreamer** - 解析事件處理
- **MCContext** - 符號和區段管理
- **MCInstrInfo** - 指令資訊
- **MCRegisterInfo** - 暫存器資訊
- **MCSubtargetInfo** - 子架構資訊

### 解析流程

1. **預處理** - 移除不支援的 AMD 特定 directives
2. **Metadata 提取** - 提取 `.amdgpu_metadata` 區塊內容
3. **LLVM MC 解析** - 使用 LLVM MC Parser 解析組合語言
4. **Metadata 解析** - 解析 YAML 格式的 metadata
5. **行分類** - 分類每一行為 Label/Instruction/Directive/Comment
6. **Label Block 建構** - 建立 Label 與 Instructions 的對應關係

### Metadata 功能

解析器能夠動態解析 AMD GPU Metadata 中的所有欄位，包括：

**Kernel 資訊：**
- 名稱、符號
- AGPR/SGPR/VGPR 數量
- Spill 計數

**記憶體配置：**
- Group/Private/Kernarg segment 大小
- 對齊要求

**參數資訊：**
- 按原始 YAML 順序保存所有屬性
- 動態處理未來可能新增的欄位

**工作群組配置：**
- Workgroup size
- Wavefront size

## 系統需求

- **LLVM** - LLVM 開發庫（建議 LLVM 16+）
- **CMake** - 3.20 或更高版本
- **C++** - C++17 標準
- **編譯器** - GCC 或 Clang

## CMake 整合

在其他專案的 `CMakeLists.txt` 中整合：

```cmake
# 設定 LLVM
find_package(LLVM REQUIRED CONFIG)
include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

# 包含 parser 源碼目錄
include_directories(/path/to/amdisa_annotate/src)

# 添加源檔案到你的專案
add_executable(your_project
  main.cpp
  /path/to/amdisa_annotate/src/parse_utils.cpp
  /path/to/amdisa_annotate/src/AMDGPUMetadata.cpp
  /path/to/amdisa_annotate/src/AMDGCNAssembly.cpp
)

# LLVM 組件
llvm_map_components_to_libnames(LLVM_LIBS
  Support Option Target TargetParser MC MCParser
  AMDGPUInfo AMDGPUDesc AMDGPUAsmParser AMDGPUCodeGen
)

target_link_libraries(your_project PRIVATE ${LLVM_LIBS})
```

## 故障排除

### 找不到 LLVM

```bash
cmake -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm ..
```

### 執行檔過大

確認使用 Release 模式，編譯後會自動執行 strip：

```bash
cmake -DCMAKE_BUILD_TYPE=Release ..
```

### 解析失敗

常見原因：
1. 目標架構不匹配（檢查 `--triple` 和 `--mcpu` 參數）
2. LLVM 版本不相容
3. 組合語言語法錯誤

## 參考文件

- `docs/API_DOCUMENTATION.md` - 完整 API 參考文件
- `docs/HOW_TO_USE_API.md` - API 使用指南
- `examples/api_usage_demo.cpp` - 實際範例程式


