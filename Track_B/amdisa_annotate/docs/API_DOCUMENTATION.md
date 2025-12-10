# AMDGCNAssembly API 文檔

## 概述

`AMDGCNAssembly` 類提供了一個統一的 in-memory API，整合了AMD GCN 組合語言檔案的所有解析資訊，包括：
- 行級別資訊（類型、內容）
- 指令詳細資訊（opcode、operands、類型）
- Metadata（kernel 配置、資源使用）
- 標籤和 kernel 區域
- 統計和分析功能

## 核心類別

### 1. AMDGCNAssembly

主要的容器類，儲存完整的組合語言檔案資訊。

```cpp
class AMDGCNAssembly {
public:
  // 設置/取得檔案名稱
  void setFilename(const std::string &filename);
  const std::string& getFilename() const;
  
  // 設置解析結果
  void setParsedProgram(const ParsedProgram &program);
  void setMetadata(const AMDGPUMetadata &metadata);
  void addLine(const LineInfo &line);
  void addKernelRegion(const KernelRegion &region);
  
  // ... 查詢接口 ...
};
```

### 2. LineInfo

單行的完整資訊。

```cpp
struct LineInfo {
  size_t lineNumber;                        // 行號（1-based）
  std::string text;                         // 原始文本
  LineKind kind;                            // 行類型
  std::optional<ParsedInstruction> instruction;  // 指令資訊（如果是指令）
  std::string labelName;                    // 標籤名稱（如果是標籤）
  std::string kernelName;                   // Kernel 名稱（如果是 kernel）
  
  // 輔助方法
  bool isInstruction() const;
  bool isLabel() const;
  bool isDirective() const;
  bool isComment() const;
  bool isMetadata() const;
  bool isKernelName() const;
};
```

### 3. KernelRegion

Kernel 的完整資訊（代碼 + metadata）。

```cpp
struct KernelRegion {
  std::string name;                         // Kernel 名稱
  size_t startLine, endLine;                // 行範圍
  std::vector<ParsedInstruction> instructions;  // 所有指令
  std::vector<ParsedLabel> labels;          // 所有標籤
  const KernelInfo *metadata;               // Metadata（如果有）
  
  // 輔助方法
  size_t getInstructionCount() const;
  size_t getLabelCount() const;
  bool hasMetadata() const;
  size_t getLineCount() const;
};
```

### 4. LineKind

行類型枚舉。

```cpp
enum class LineKind {
  Unknown,
  Label,
  Instruction,
  Directive,
  Comment,
  Metadata,
  KernelName,
};
```

## API 參考

### 查詢接口 - 按行號

#### getLineCount()
取得總行數。

```cpp
size_t getLineCount() const;
```

**回傳值**：總行數

**範例**：
```cpp
AMDGCNAssembly assembly;
// ... 填充數據 ...
size_t total = assembly.getLineCount();
llvm::outs() << "Total lines: " << total << "\n";
```

#### getLine()
取得指定行的資訊。

```cpp
const LineInfo* getLine(size_t lineNumber) const;
```

**參數**：
- `lineNumber`：行號（1-based）

**回傳值**：行資訊指針，如果不存在則為 `nullptr`

**範例**：
```cpp
if (const LineInfo *line = assembly.getLine(100)) {
  llvm::outs() << "Line 100: " << line->text << "\n";
  if (line->isInstruction()) {
    llvm::outs() << "  Opcode: " << line->instruction->opcode << "\n";
  }
}
```

#### getLines()
取得指定範圍的行。

```cpp
std::vector<const LineInfo*> getLines(size_t startLine, size_t endLine) const;
```

**參數**：
- `startLine`：起始行號（含）
- `endLine`：結束行號（含）

**回傳值**：行資訊指針的向量

**範例**：
```cpp
auto lines = assembly.getLines(1, 100);
for (const auto *line : lines) {
  llvm::outs() << line->lineNumber << ": " << line->text << "\n";
}
```

#### getInstructionLines()
取得所有指令行。

```cpp
std::vector<const LineInfo*> getInstructionLines() const;
```

**範例**：
```cpp
auto instLines = assembly.getInstructionLines();
llvm::outs() << "Found " << instLines.size() << " instruction lines\n";
```

#### getLabelLines()
取得所有標籤行。

```cpp
std::vector<const LineInfo*> getLabelLines() const;
```

### 查詢接口 - 指令

#### getAllInstructions()
取得所有指令。

```cpp
const std::vector<ParsedInstruction>& getAllInstructions() const;
```

**範例**：
```cpp
const auto &insts = assembly.getAllInstructions();
for (const auto &inst : insts) {
  llvm::outs() << inst.opcode << " (line " << inst.lineNumber << ")\n";
}
```

#### getInstructionAtLine()
取得指定行的指令。

```cpp
const ParsedInstruction* getInstructionAtLine(size_t lineNumber) const;
```

**參數**：
- `lineNumber`：行號（1-based）

**回傳值**：指令指針，如果該行不是指令則為 `nullptr`

**範例**：
```cpp
if (const auto *inst = assembly.getInstructionAtLine(50)) {
  llvm::outs() << "Instruction at line 50: " << inst->opcode << "\n";
  for (const auto &op : inst->operands) {
    llvm::outs() << "  Operand: " << op.text << "\n";
  }
}
```

#### findInstructionsByOpcode()
按 opcode 過濾指令。

```cpp
std::vector<const ParsedInstruction*> 
findInstructionsByOpcode(const std::string &opcode) const;
```

**參數**：
- `opcode`：指令名稱（例如 "s_load_dwordx2"）

**回傳值**：匹配的指令指針向量

**範例**：
```cpp
auto loads = assembly.findInstructionsByOpcode("s_load_dwordx2");
llvm::outs() << "Found " << loads.size() << " s_load_dwordx2 instructions\n";
```

#### findInstructionsByOpcodePrefix()
按 opcode 前綴過濾指令。

```cpp
std::vector<const ParsedInstruction*> 
findInstructionsByOpcodePrefix(const std::string &prefix) const;
```

**參數**：
- `prefix`：指令前綴（例如 "s_" 找出所有 scalar 指令）

**回傳值**：匹配的指令指針向量

**範例**：
```cpp
// 找出所有 scalar 指令
auto scalarInsts = assembly.findInstructionsByOpcodePrefix("s_");
llvm::outs() << "Scalar instructions: " << scalarInsts.size() << "\n";

// 找出所有 load 指令
auto loadInsts = assembly.findInstructionsByOpcodePrefix("s_load_");
llvm::outs() << "Load instructions: " << loadInsts.size() << "\n";
```

#### getInstructionCount()
取得指令總數。

```cpp
size_t getInstructionCount() const;
```

### 查詢接口 - Kernel

#### getKernelCount()
取得 kernel 數量。

```cpp
size_t getKernelCount() const;
```

#### getAllKernels()
取得所有 kernels。

```cpp
const std::vector<KernelRegion>& getAllKernels() const;
```

**範例**：
```cpp
for (const auto &kernel : assembly.getAllKernels()) {
  llvm::outs() << "Kernel: " << kernel.name << "\n";
  llvm::outs() << "  Lines: " << kernel.startLine << "-" << kernel.endLine << "\n";
  llvm::outs() << "  Instructions: " << kernel.getInstructionCount() << "\n";
}
```

#### findKernel()
根據名稱查找 kernel。

```cpp
const KernelRegion* findKernel(const std::string &name) const;
```

**參數**：
- `name`：Kernel 名稱

**回傳值**：Kernel 資訊指針，如果不存在則為 `nullptr`

**範例**：
```cpp
if (const auto *kernel = assembly.findKernel("my_kernel")) {
  llvm::outs() << "Found kernel: " << kernel->name << "\n";
  llvm::outs() << "Instructions: " << kernel->instructions.size() << "\n";
}
```

#### findKernelByLine()
查找包含指定行號的 kernel。

```cpp
const KernelRegion* findKernelByLine(size_t lineNumber) const;
```

**參數**：
- `lineNumber`：行號

**範例**：
```cpp
if (const auto *kernel = assembly.findKernelByLine(1000)) {
  llvm::outs() << "Line 1000 belongs to kernel: " << kernel->name << "\n";
}
```

### 查詢接口 - Metadata

#### getMetadata()
取得 metadata。

```cpp
const AMDGPUMetadata& getMetadata() const;
```

**範例**：
```cpp
const auto &meta = assembly.getMetadata();
llvm::outs() << "Target: " << meta.target << "\n";
for (const auto &kernel : meta.kernels) {
  llvm::outs() << "Kernel: " << kernel.name << "\n";
  llvm::outs() << "  AGPR: " << kernel.agprCount << "\n";
  llvm::outs() << "  SGPR: " << kernel.sgprCount << "\n";
  llvm::outs() << "  VGPR: " << kernel.vgprCount << "\n";
}
```

#### hasMetadata()
檢查是否有 metadata。

```cpp
bool hasMetadata() const;
```

#### getTargetTriple()
取得目標架構。

```cpp
std::string getTargetTriple() const;
```

**範例**：
```cpp
if (assembly.hasMetadata()) {
  llvm::outs() << "Target: " << assembly.getTargetTriple() << "\n";
}
```

#### getMetadataForKernel()
取得指定 kernel 的 metadata。

```cpp
const KernelInfo* getMetadataForKernel(const std::string &name) const;
```

**參數**：
- `name`：Kernel 名稱

**回傳值**：Kernel metadata 指針，如果不存在則為 `nullptr`

**範例**：
```cpp
if (const auto *meta = assembly.getMetadataForKernel("my_kernel")) {
  llvm::outs() << "Kernel metadata:\n";
  llvm::outs() << "  AGPR: " << meta->agprCount << "\n";
  llvm::outs() << "  SGPR: " << meta->sgprCount << "\n";
  llvm::outs() << "  Max workgroup: " << meta->maxFlatWorkgroupSize << "\n";
}
```

### 查詢接口 - 標籤

#### getAllLabels()
取得所有標籤。

```cpp
const std::vector<ParsedLabel>& getAllLabels() const;
```

**範例**：
```cpp
for (const auto &label : assembly.getAllLabels()) {
  llvm::outs() << "Label: " << label.name << " at line " << label.lineNumber << "\n";
}
```

#### findLabel()
查找標籤。

```cpp
const ParsedLabel* findLabel(const std::string &name) const;
```

**參數**：
- `name`：標籤名稱

**回傳值**：標籤資訊指針，如果不存在則為 `nullptr`

**範例**：
```cpp
if (const auto *label = assembly.findLabel("BB0_1")) {
  llvm::outs() << "Label BB0_1 found at line " << label->lineNumber << "\n";
}
```

### 統計接口

#### getStatistics()
取得統計資訊。

```cpp
Statistics getStatistics() const;
```

**回傳值**：`Statistics` 結構，包含：
- `totalLines`：總行數
- `instructionCount`：指令數
- `labelCount`：標籤數
- `directiveCount`：指示數
- `commentCount`：註解數
- `metadataLineCount`：Metadata 行數
- `kernelCount`：Kernel 數
- `scalarInstructions`：Scalar 指令數
- `vectorInstructions`：Vector 指令數
- `memoryInstructions`：Memory 指令數
- `branchInstructions`：Branch 指令數

**範例**：
```cpp
auto stats = assembly.getStatistics();
llvm::outs() << "Statistics:\n";
llvm::outs() << "  Total lines: " << stats.totalLines << "\n";
llvm::outs() << "  Instructions: " << stats.instructionCount << "\n";
llvm::outs() << "  Scalar: " << stats.scalarInstructions << " ("
             << (stats.scalarInstructions * 100 / stats.instructionCount) << "%)\n";
llvm::outs() << "  Vector: " << stats.vectorInstructions << " ("
             << (stats.vectorInstructions * 100 / stats.instructionCount) << "%)\n";
```

#### printSummary()
打印摘要報告。

```cpp
void printSummary(llvm::raw_ostream &OS) const;
```

**參數**：
- `OS`：輸出串流

**範例**：
```cpp
assembly.printSummary(llvm::outs());
```

#### printDetailed()
打印詳細報告。

```cpp
void printDetailed(llvm::raw_ostream &OS) const;
```

**參數**：
- `OS`：輸出串流

**範例**：
```cpp
assembly.printDetailed(llvm::errs());
```

### 輔助接口

#### isEmpty()
檢查檔案是否為空。

```cpp
bool isEmpty() const;
```

#### getParsedProgram()
取得原始的 ParsedProgram。

```cpp
const ParsedProgram& getParsedProgram() const;
```

## 完整使用範例

```cpp
#include "AMDGCNAssembly.h"
#include "llvm/Support/raw_ostream.h"

int main() {
  AMDGCNAssembly assembly;
  
  // 1. 基本查詢
  llvm::outs() << "Total lines: " << assembly.getLineCount() << "\n";
  llvm::outs() << "Instructions: " << assembly.getInstructionCount() << "\n";
  
  // 2. 查詢特定行
  if (const auto *line = assembly.getLine(100)) {
    llvm::outs() << "Line 100: " << line->text << "\n";
  }
  
  // 3. 查找指令
  auto loadInsts = assembly.findInstructionsByOpcodePrefix("s_load_");
  llvm::outs() << "Found " << loadInsts.size() << " load instructions\n";
  
  // 4. Metadata 查詢
  if (assembly.hasMetadata()) {
    const auto &meta = assembly.getMetadata();
    llvm::outs() << "Target: " << meta.target << "\n";
    for (const auto &kernel : meta.kernels) {
      llvm::outs() << "Kernel: " << kernel.name << "\n";
      llvm::outs() << "  AGPR: " << kernel.agprCount << "\n";
    }
  }
  
  // 5. 統計分析
  auto stats = assembly.getStatistics();
  llvm::outs() << "Scalar instructions: " << stats.scalarInstructions << "\n";
  llvm::outs() << "Vector instructions: " << stats.vectorInstructions << "\n";
  
  // 6. 打印報告
  assembly.printSummary(llvm::outs());
  
  return 0;
}
```

## 注意事項

1. **指針有效性**：所有返回的指針都是指向內部數據的，不要在 `AMDGCNAssembly` 對象銷毀後使用。

2. **線程安全**：`AMDGCNAssembly` 不是線程安全的，如果需要在多線程環境中使用，請添加適當的同步機制。

3. **記憶體使用**：`AMDGCNAssembly` 會在記憶體中保存完整的檔案資訊，對於大型檔案可能需要較多記憶體。

4. **1-based 行號**：所有行號都是 1-based，與大多數編輯器一致。

## 更多範例

查看以下文件獲取更多範例：
- `API_EXAMPLE.cpp` - 完整的 API 使用範例
- `main.cpp` - amdisa_annotate 工具的實現
- `USAGE_EXAMPLES.txt` - 命令行使用範例

