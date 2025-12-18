#ifndef PARSED_PROGRAM_H
#define PARSED_PROGRAM_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <string>
#include <vector>

// -----------------------------------------------------------------------------
// 操作數類型
// -----------------------------------------------------------------------------
enum class OperandType {
  Register,      // 寄存器，例如 s[0:1], v0, a0
  Immediate,     // 立即數，例如 0x0, 15, -1
  Label,         // 標籤/符號，例如 src_shared_base
  Expression,    // 表達式，例如 lgkmcnt(0), bitop3:0x6c
  Unknown
};

// -----------------------------------------------------------------------------
// 操作數資訊
// -----------------------------------------------------------------------------
struct Operand {
  std::string text;           // 原始文本
  OperandType type;           // 操作數類型
  
  Operand(const std::string &t, OperandType ty = OperandType::Unknown)
      : text(t), type(ty) {}
  
  // 判斷是否為寄存器
  bool isRegister() const { return type == OperandType::Register; }
  bool isImmediate() const { return type == OperandType::Immediate; }
  bool isLabel() const { return type == OperandType::Label; }
  bool isExpression() const { return type == OperandType::Expression; }
};

// -----------------------------------------------------------------------------
// 指令資訊（增強版）
// -----------------------------------------------------------------------------
struct ParsedInstruction {
  std::string opcode;                      // 操作碼
  std::vector<Operand> operands;           // 操作數列表（帶類型）
  size_t lineNumber;                       // 原始行號（1-based）
  std::string originalText;                // 原始文本
  
  ParsedInstruction() : lineNumber(0) {}
  
  size_t getNumOperands() const { return operands.size(); }
  
  const Operand& getOperand(size_t idx) const { return operands[idx]; }
  
  bool isEmpty() const { return opcode.empty(); }
};

// -----------------------------------------------------------------------------
// 標籤資訊
// -----------------------------------------------------------------------------
struct ParsedLabel {
  std::string name;                        // 標籤名稱
  size_t lineNumber;                       // 原始行號
  std::string originalText;                // 原始文本
  
  ParsedLabel() : lineNumber(0) {}
};

// -----------------------------------------------------------------------------
// Kernel 資訊
// -----------------------------------------------------------------------------
struct ParsedKernel {
  std::string name;                        // Kernel 名稱
  size_t declLineNumber;                   // 宣告行號（.globl）
  size_t defLineNumber;                    // 定義行號（標籤）
  std::vector<ParsedInstruction> instructions; // 該 kernel 的所有指令
  
  ParsedKernel() : declLineNumber(0), defLineNumber(0) {}
};

// -----------------------------------------------------------------------------
// 完整的程序表示
// -----------------------------------------------------------------------------
class ParsedProgram {
public:
  // Kernel 列表
  std::vector<ParsedKernel> kernels;
  
  // 所有指令（按行號順序）
  std::vector<ParsedInstruction> allInstructions;
  
  // 所有標籤
  std::vector<ParsedLabel> labels;
  
  // 原始文件信息
  std::string filename;
  size_t totalLines;
  
  ParsedProgram() : totalLines(0) {}
  
  // API: 遍歷所有 kernels
  const std::vector<ParsedKernel>& getKernels() const { return kernels; }
  
  // API: 遍歷所有指令
  const std::vector<ParsedInstruction>& getInstructions() const { 
    return allInstructions; 
  }
  
  // API: 根據行號查找指令
  const ParsedInstruction* getInstructionAtLine(size_t line) const {
    for (const auto &inst : allInstructions) {
      if (inst.lineNumber == line)
        return &inst;
    }
    return nullptr;
  }
  
  // API: 獲取指令數量
  size_t getInstructionCount() const { return allInstructions.size(); }
  
  // API: 獲取 kernel 數量
  size_t getKernelCount() const { return kernels.size(); }
  
  // 統計信息
  void printSummary(llvm::raw_ostream &OS) const {
    OS << "=== Parsed Program Summary ===\n";
    OS << "Total lines: " << totalLines << "\n";
    OS << "Total kernels: " << kernels.size() << "\n";
    OS << "Total instructions: " << allInstructions.size() << "\n";
    OS << "Total labels: " << labels.size() << "\n";
  }
};

// -----------------------------------------------------------------------------
// 操作數類型檢測輔助函數
// -----------------------------------------------------------------------------
inline OperandType detectOperandType(llvm::StringRef text) {
  text = text.trim();
  
  if (text.empty())
    return OperandType::Unknown;
  
  // 檢查是否為寄存器：s[...], v[...], a[...], s0, v0 等
  if (text.starts_with("s[") || text.starts_with("v[") || 
      text.starts_with("a[") || text.starts_with("ttmp[")) {
    return OperandType::Register;
  }
  
  if ((text.starts_with("s") || text.starts_with("v") || 
       text.starts_with("a")) && text.size() > 1) {
    // 檢查後面是否全是數字
    bool allDigits = true;
    for (size_t i = 1; i < text.size(); ++i) {
      if (text[i] < '0' || text[i] > '9') {
        allDigits = false;
        break;
      }
    }
    if (allDigits)
      return OperandType::Register;
  }
  
  // 檢查是否為立即數：0x..., 數字, 負數
  if (text.starts_with("0x") || text.starts_with("-0x"))
    return OperandType::Immediate;
  
  bool isNumber = true;
  size_t start = 0;
  if (text[0] == '-')
    start = 1;
  
  for (size_t i = start; i < text.size(); ++i) {
    if (text[i] < '0' || text[i] > '9') {
      isNumber = false;
      break;
    }
  }
  if (isNumber && text.size() > start)
    return OperandType::Immediate;
  
  // 檢查是否為表達式（包含括號或冒號）
  if (text.contains('(') || text.contains(':'))
    return OperandType::Expression;
  
  // 其他情況視為標籤/符號
  return OperandType::Label;
}

#endif // PARSED_PROGRAM_H

