# 如何使用 AMDGCNAssembly API - 完整指南

## 概述

這份文件說明如何在你的 C++ 程式中使用我們建立的工具和 API。

## 方法一：在 main.cpp 中直接使用（最簡單）

在 `main.cpp` 的最後，我們已經構建了完整的 `AMDGCNAssembly` 對象，你可以直接在那裡添加你的代碼：

### 位置

在 `main.cpp` 的第 651 行（`// 11. 構建統一的 AMDGCNAssembly API 對象` 之後）

### 範例

```cpp
// 在 main.cpp 中，assembly 對象已經構建完成
// 你可以直接使用：

// 步驟 1: 取得 Metadata
if (assembly.hasMetadata()) {
  const auto &meta = assembly.getMetadata();
  
  // 輸出到你的文件或做其他處理
  std::ofstream metaFile("metadata_output.txt");
  metaFile << "Target: " << meta.target << "\n";
  
  for (const auto &kernel : meta.kernels) {
    metaFile << "Kernel: " << kernel.name << "\n";
    metaFile << "  AGPR: " << kernel.agprCount << "\n";
    metaFile << "  SGPR: " << kernel.sgprCount << "\n";
    metaFile << "  VGPR: " << kernel.vgprCount << "\n";
  }
  metaFile.close();
}

// 步驟 2: 取得 Instructions 和 Labels
std::ofstream instFile("instructions_output.txt");

// 取得所有指令
const auto &insts = assembly.getAllInstructions();
instFile << "總共 " << insts.size() << " 個指令\n\n";

for (const auto &inst : insts) {
  instFile << "Line " << inst.lineNumber << ": " 
           << inst.opcode << " ";
  for (const auto &op : inst.operands) {
    instFile << op.text << " ";
  }
  instFile << "\n";
}

// 取得所有標籤
const auto &labels = assembly.getAllLabels();
instFile << "\n總共 " << labels.size() << " 個標籤\n\n";

for (const auto &label : labels) {
  instFile << "Line " << label.lineNumber << ": " 
           << label.name << "\n";
}

instFile.close();
```

## 方法二：創建可重用的解析函數

### 步驟 1: 創建 parse_helper.h

```cpp
#ifndef PARSE_HELPER_H
#define PARSE_HELPER_H

#include "AMDGCNAssembly.h"
#include <string>

// 完整解析組合語言檔案，返回 AMDGCNAssembly 對象
AMDGCNAssembly parseAssemblyFileComplete(const std::string &filename,
                                          const std::string &triple = "amdgcn-amd-amdhsa",
                                          const std::string &mcpu = "gfx950");

#endif
```

### 步驟 2: 創建 parse_helper.cpp

將 `main.cpp` 中的解析邏輯（從第 283 行到第 650 行）提取到這個函數中：

```cpp
#include "parse_helper.h"
#include "llvm/Support/InitLLVM.h"
// ... 其他 includes ...

AMDGCNAssembly parseAssemblyFileComplete(const std::string &filename,
                                          const std::string &triple,
                                          const std::string &mcpu) {
  // 初始化 LLVM
  static bool initialized = false;
  if (!initialized) {
    InitializeAllTargetInfos();
    InitializeAllTargets();
    InitializeAllTargetMCs();
    InitializeAllAsmParsers();
    initialized = true;
  }
  
  // [從 main.cpp 複製解析邏輯]
  // 1. 設置 target
  // 2. 讀取檔案
  // 3. 提取 metadata
  // 4. 使用 LLVM MC Parser 解析
  // 5. 分類行類型
  // 6. 構建 AMDGCNAssembly 對象
  
  AMDGCNAssembly assembly;
  // ... 填充所有資料 ...
  
  return assembly;
}
```

### 步驟 3: 在你的程式中使用

```cpp
#include "parse_helper.h"
#include <iostream>

int main(int argc, char **argv) {
  std::string filename = argv[1];
  
  // 解析檔案（所有資料在記憶體中）
  AMDGCNAssembly assembly = parseAssemblyFileComplete(filename);
  
  // 步驟 1: 取得 Metadata
  if (assembly.hasMetadata()) {
    const auto &meta = assembly.getMetadata();
    std::cout << "Target: " << meta.target << "\n";
    
    for (const auto &kernel : meta.kernels) {
      std::cout << "\nKernel: " << kernel.name << "\n";
      std::cout << "  AGPR: " << kernel.agprCount << "\n";
      std::cout << "  SGPR: " << kernel.sgprCount << "\n";
      std::cout << "  VGPR: " << kernel.vgprCount << "\n";
      std::cout << "  Workgroup Size: " << kernel.maxFlatWorkgroupSize << "\n";
      std::cout << "  Wavefront Size: " << kernel.wavefrontSize << "\n";
    }
  }
  
  // 步驟 2: 取得 Instructions 和 Labels
  const auto &insts = assembly.getAllInstructions();
  const auto &labels = assembly.getAllLabels();
  
  std::cout << "\n總共 " << insts.size() << " 個指令\n";
  std::cout << "總共 " << labels.size() << " 個標籤\n\n";
  
  // 顯示前 20 個指令
  std::cout << "前 20 個指令:\n";
  std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
  for (size_t i = 0; i < std::min(size_t(20), insts.size()); ++i) {
    const auto &inst = insts[i];
    printf("Line %5zu: %-20s ", inst.lineNumber, inst.opcode.c_str());
    for (const auto &op : inst.operands) {
      std::cout << op.text << " ";
    }
    std::cout << "\n";
  }
  
  // 顯示所有標籤
  std::cout << "\n所有標籤:\n";
  std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
  for (const auto &label : labels) {
    printf("Line %5zu: %s\n", label.lineNumber, label.name.c_str());
  }
  
  return 0;
}
```

## 方法三：查看示範程式

我們提供了一個完整的示範程式：

```bash
./build/api_usage_demo input.s
```

這個程式展示了所有主要 API 的調用方式，包括：
- 如何檢查和取得 metadata
- 如何遍歷 kernels
- 如何查詢指令
- 如何查詢標籤
- 如何取得統計資訊

## 完整的 API 調用範例

### Metadata 查詢

```cpp
// 1. 檢查是否有 metadata
if (assembly.hasMetadata()) {
  
  // 2. 取得 metadata 對象
  const AMDGPUMetadata &meta = assembly.getMetadata();
  
  // 3. 取得目標架構
  std::string target = assembly.getTargetTriple();
  std::cout << "Target: " << target << "\n";
  
  // 4. 遍歷所有 kernels
  for (const auto &kernel : meta.kernels) {
    std::cout << "Kernel: " << kernel.name << "\n";
    std::cout << "  AGPR: " << kernel.agprCount << "\n";
    std::cout << "  SGPR: " << kernel.sgprCount << "\n";
    std::cout << "  VGPR: " << kernel.vgprCount << "\n";
    std::cout << "  Max Workgroup: " << kernel.maxFlatWorkgroupSize << "\n";
    std::cout << "  Kernarg Size: " << kernel.kernargSegmentSize << " bytes\n";
    
    // 訪問參數
    for (const auto &arg : kernel.args) {
      std::cout << "    Arg offset=" << arg.offset 
                << " size=" << arg.size 
                << " kind=" << arg.valueKind << "\n";
    }
  }
  
  // 5. 根據名稱查找 kernel
  const KernelInfo *k = assembly.getMetadataForKernel("kernel_name");
  if (k) {
    std::cout << "Found: " << k->name << "\n";
  }
}
```

### Instructions 查詢

```cpp
// 1. 取得指令數量
size_t count = assembly.getInstructionCount();
std::cout << "Total instructions: " << count << "\n";

// 2. 取得所有指令
const auto &allInsts = assembly.getAllInstructions();
for (const auto &inst : allInsts) {
  std::cout << inst.opcode << " at line " << inst.lineNumber << "\n";
  
  // 訪問操作數
  for (const auto &op : inst.operands) {
    std::cout << "  Operand: " << op.text;
    
    // 取得操作數類型
    switch (op.type) {
      case OperandType::Register:
        std::cout << " (register)";
        break;
      case OperandType::Immediate:
        std::cout << " (immediate)";
        break;
      case OperandType::Label:
        std::cout << " (label)";
        break;
      default:
        break;
    }
    std::cout << "\n";
  }
}

// 3. 查找特定 opcode 的指令（精確匹配）
auto loads = assembly.findInstructionsByOpcode("s_load_dwordx2");
std::cout << "Found " << loads.size() << " s_load_dwordx2 instructions\n";

// 4. 查找特定前綴的指令
auto allLoads = assembly.findInstructionsByOpcodePrefix("s_load_");
std::cout << "Found " << allLoads.size() << " s_load_* instructions\n";

for (const auto *inst : allLoads) {
  std::cout << "  " << inst->opcode << " at line " << inst->lineNumber << "\n";
}

// 5. 取得特定行的指令
const ParsedInstruction *inst = assembly.getInstructionAtLine(100);
if (inst) {
  std::cout << "Line 100: " << inst->opcode << "\n";
}
```

### Labels 查詢

```cpp
// 1. 取得所有標籤
const auto &labels = assembly.getAllLabels();
std::cout << "Total labels: " << labels.size() << "\n";

for (const auto &label : labels) {
  std::cout << "Label: " << label.name 
            << " at line " << label.lineNumber << "\n";
}

// 2. 查找特定標籤
const ParsedLabel *label = assembly.findLabel("BB0_1");
if (label) {
  std::cout << "Found label BB0_1 at line " << label->lineNumber << "\n";
}
```

### 統計查詢

```cpp
// 取得統計資訊
auto stats = assembly.getStatistics();

std::cout << "統計資訊:\n";
std::cout << "  總行數: " << stats.totalLines << "\n";
std::cout << "  指令數: " << stats.instructionCount << "\n";
std::cout << "  標籤數: " << stats.labelCount << "\n";
std::cout << "  註解數: " << stats.commentCount << "\n";
std::cout << "\n";

std::cout << "指令分類:\n";
std::cout << "  Scalar (s_*): " << stats.scalarInstructions 
          << " (" << (stats.scalarInstructions * 100 / stats.instructionCount) << "%)\n";
std::cout << "  Vector (v_*): " << stats.vectorInstructions
          << " (" << (stats.vectorInstructions * 100 / stats.instructionCount) << "%)\n";
std::cout << "  Memory: " << stats.memoryInstructions
          << " (" << (stats.memoryInstructions * 100 / stats.instructionCount) << "%)\n";
std::cout << "  Branch: " << stats.branchInstructions << "\n";
```

### 打印報告

```cpp
// 打印摘要報告
assembly.printSummary(llvm::outs());

// 打印詳細報告
assembly.printDetailed(llvm::outs());
```

## 方法二：查看示範程式

我們提供了 `api_usage_demo.cpp`，它展示了所有 API 的調用方式：

```bash
# 查看源代碼
cat api_usage_demo.cpp

# 編譯並執行
cd build
make api_usage_demo
./api_usage_demo input.s
```

輸出會展示：
- 每個 API 的調用代碼
- 每個 API 的返回結果
- 實際的使用範例

## 方法三：查看 main.cpp 的實際使用

在 `main.cpp` 中，我們已經實際使用了這些 API：

### API Demo 模式

```bash
./build/amdisa_annotate --api-demo input.s 2>&1 >/dev/null
```

這會運行一系列 API 調用並顯示結果。

### Detailed Analysis 模式

```bash
./build/amdisa_annotate --detailed-analysis input.s 2>&1 >/dev/null
```

這會使用 API 生成詳細的分析報告。

## 完整的 API 列表

### AMDGCNAssembly 類

#### 基本查詢
- `getLineCount()` - 總行數
- `getLine(lineNum)` - 取得特定行
- `getLines(start, end)` - 取得行範圍
- `isEmpty()` - 檢查是否為空

#### 指令查詢
- `getInstructionCount()` - 指令總數
- `getAllInstructions()` - 所有指令
- `getInstructionAtLine(lineNum)` - 特定行的指令
- `findInstructionsByOpcode(opcode)` - 按名稱查找
- `findInstructionsByOpcodePrefix(prefix)` - 按前綴查找
- `getInstructionLines()` - 所有指令行

#### Kernel 查詢
- `getKernelCount()` - Kernel 數量
- `getAllKernels()` - 所有 kernels
- `findKernel(name)` - 按名稱查找
- `findKernelByLine(lineNum)` - 按行號查找

#### Metadata 查詢
- `hasMetadata()` - 檢查是否有 metadata
- `getMetadata()` - 取得 metadata 對象
- `getTargetTriple()` - 取得目標架構
- `getMetadataForKernel(name)` - 取得 kernel 的 metadata

#### 標籤查詢
- `getAllLabels()` - 所有標籤
- `findLabel(name)` - 按名稱查找
- `getLabelLines()` - 所有標籤行

#### 統計和報告
- `getStatistics()` - 取得統計資訊
- `printSummary(OS)` - 打印摘要
- `printDetailed(OS)` - 打印詳細報告

#### 其他
- `setFilename(name)` / `getFilename()` - 設置/取得檔案名
- `setParsedProgram(program)` - 設置解析結果
- `setMetadata(metadata)` - 設置 metadata
- `addLine(lineInfo)` - 添加行資訊
- `getParsedProgram()` - 取得原始 ParsedProgram

## 實用範例

### 範例 1: 快速查看 Kernel 資源

```cpp
AMDGCNAssembly assembly = parseAssemblyFileComplete("kernel.s");

if (assembly.hasMetadata()) {
  for (const auto &kernel : assembly.getMetadata().kernels) {
    std::cout << kernel.name << ":\n";
    std::cout << "  AGPR=" << kernel.agprCount 
              << " SGPR=" << kernel.sgprCount 
              << " VGPR=" << kernel.vgprCount << "\n";
  }
}
```

### 範例 2: 分析 Memory Access

```cpp
AMDGCNAssembly assembly = parseAssemblyFileComplete("kernel.s");

auto dsInsts = assembly.findInstructionsByOpcodePrefix("ds_");
auto globalInsts = assembly.findInstructionsByOpcodePrefix("global_");

std::cout << "LDS access: " << dsInsts.size() << "\n";
std::cout << "Global memory access: " << globalInsts.size() << "\n";

if (globalInsts.size() > dsInsts.size() * 2) {
  std::cout << "建議: 考慮使用更多 LDS 來減少 global memory 訪問\n";
}
```

### 範例 3: 輸出到檔案

```cpp
AMDGCNAssembly assembly = parseAssemblyFileComplete("kernel.s");

// 輸出 metadata 到檔案
std::ofstream metaOut("metadata.txt");
if (assembly.hasMetadata()) {
  const auto &meta = assembly.getMetadata();
  metaOut << "Target: " << meta.target << "\n";
  for (const auto &k : meta.kernels) {
    metaOut << "Kernel: " << k.name << "\n";
    metaOut << "  Resources: AGPR=" << k.agprCount 
            << " SGPR=" << k.sgprCount 
            << " VGPR=" << k.vgprCount << "\n";
  }
}

// 輸出 instructions 到檔案
std::ofstream instOut("instructions.txt");
for (const auto &inst : assembly.getAllInstructions()) {
  instOut << inst.lineNumber << ": " << inst.opcode << " ";
  for (const auto &op : inst.operands) {
    instOut << op.text << " ";
  }
  instOut << "\n";
}
```

## 相關文檔

- `API_DOCUMENTATION.md` - 完整 API 參考
- `API_EXAMPLE.cpp` - 6個實用範例
- `api_usage_demo.cpp` - API 調用展示
- `main.cpp` - 完整的實現範例

## 快速參考

```cpp
// 基本流程
AMDGCNAssembly assembly = parseAssemblyFileComplete("input.s");

// Metadata
if (assembly.hasMetadata()) {
  const auto &meta = assembly.getMetadata();
  // 使用 meta.kernels, meta.target, meta.version
}

// Instructions
auto insts = assembly.getAllInstructions();
auto loads = assembly.findInstructionsByOpcodePrefix("s_load_");

// Labels
auto labels = assembly.getAllLabels();

// Statistics
auto stats = assembly.getStatistics();

// Reports
assembly.printSummary(llvm::outs());
```

所有資料都在記憶體中，查詢快速高效！

