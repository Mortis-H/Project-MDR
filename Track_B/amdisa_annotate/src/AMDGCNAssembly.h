#ifndef AMDGCN_ASSEMBLY_H
#define AMDGCN_ASSEMBLY_H

#include "ParsedProgram.h"
#include "AMDGPUMetadata.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <vector>
#include <map>
#include <optional>

// ============================================================================
// 行資訊 - 整合所有行級別的資訊
// ============================================================================

enum class LineKind {
  Unknown,
  Label,
  Instruction,
  Directive,
  Comment,
  Metadata,
  KernelName,
};

/// 單行的完整資訊
struct LineInfo {
  size_t lineNumber;              // 行號（1-based）
  std::string text;               // 原始文本
  LineKind kind;                  // 行類型
  
  // 如果是指令，包含詳細資訊
  std::optional<ParsedInstruction> instruction;
  
  // 如果是標籤，包含標籤名稱
  std::string labelName;
  
  // 如果是 kernel 名稱
  std::string kernelName;
  
  LineInfo() : lineNumber(0), kind(LineKind::Unknown) {}
  
  bool isInstruction() const { return kind == LineKind::Instruction; }
  bool isLabel() const { return kind == LineKind::Label; }
  bool isDirective() const { return kind == LineKind::Directive; }
  bool isComment() const { return kind == LineKind::Comment; }
  bool isMetadata() const { return kind == LineKind::Metadata; }
  bool isKernelName() const { return kind == LineKind::KernelName; }
};

// ============================================================================
// Kernel 區域資訊 - 整合 metadata 和代碼區域
// ============================================================================

/// Kernel 的完整資訊（代碼 + metadata）
struct KernelRegion {
  std::string name;               // Kernel 名稱
  size_t startLine;               // 開始行號
  size_t endLine;                 // 結束行號
  
  // 代碼資訊
  std::vector<ParsedInstruction> instructions;  // 所有指令
  std::vector<ParsedLabel> labels;              // 所有標籤
  
  // Metadata 資訊（如果有）
  const KernelInfo *metadata;     // 指向 AMDGPUMetadata 中的 KernelInfo
  
  KernelRegion() : startLine(0), endLine(0), metadata(nullptr) {}
  
  /// 取得指令數量
  size_t getInstructionCount() const { return instructions.size(); }
  
  /// 取得標籤數量
  size_t getLabelCount() const { return labels.size(); }
  
  /// 檢查是否有 metadata
  bool hasMetadata() const { return metadata != nullptr; }
  
  /// 取得 kernel 的行數範圍
  size_t getLineCount() const {
    return (endLine >= startLine) ? (endLine - startLine + 1) : 0;
  }
};

// ============================================================================
// Label Block 資訊 - Label 和其後的指令區塊
// ============================================================================

/// Label Block（一個 label 到下一個 label 之間的區域）
struct LabelBlock {
  std::string labelName;          // Label 名稱
  size_t labelLine;               // Label 所在行號
  size_t startLine;               // Block 開始行號（label 行）
  size_t endLine;                 // Block 結束行號（下一個 label 前一行或檔案結尾）
  
  // 該 block 內的所有指令
  std::vector<const ParsedInstruction*> instructions;
  
  LabelBlock() : labelLine(0), startLine(0), endLine(0) {}
  
  /// 取得指令數量
  size_t getInstructionCount() const { return instructions.size(); }
  
  /// 取得 block 的行數範圍
  size_t getLineCount() const {
    return (endLine >= startLine) ? (endLine - startLine + 1) : 0;
  }
  
  /// 檢查某行是否在這個 block 內
  bool containsLine(size_t lineNumber) const {
    return lineNumber >= startLine && lineNumber <= endLine;
  }
};

// ============================================================================
// 統一的組合語言檔案表示
// ============================================================================

/// AMD GCN 組合語言檔案的完整表示
class AMDGCNAssembly {
public:
  AMDGCNAssembly() = default;
  
  /// 設置檔案名稱
  void setFilename(const std::string &filename) { 
    filename_ = filename; 
  }
  
  /// 取得檔案名稱
  const std::string& getFilename() const { return filename_; }
  
  /// 添加行資訊
  void addLine(const LineInfo &line) {
    lines_.push_back(line);
    lineIndex_[line.lineNumber] = lines_.size() - 1;
  }
  
  /// 設置 ParsedProgram
  void setParsedProgram(const ParsedProgram &program) {
    parsedProgram_ = program;
  }
  
  /// 設置 Metadata
  void setMetadata(const AMDGPUMetadata &metadata) {
    metadata_ = metadata;
  }
  
  /// 添加 Kernel 區域
  void addKernelRegion(const KernelRegion &region) {
    kernels_.push_back(region);
    kernelIndex_[region.name] = kernels_.size() - 1;
  }
  
  // ------------------------------------------------------------------------
  // 查詢接口 - 按行號
  // ------------------------------------------------------------------------
  
  /// 取得總行數
  size_t getLineCount() const { return lines_.size(); }
  
  /// 取得指定行的資訊
  const LineInfo* getLine(size_t lineNumber) const {
    auto it = lineIndex_.find(lineNumber);
    if (it != lineIndex_.end()) {
      return &lines_[it->second];
    }
    return nullptr;
  }
  
  /// 取得指定範圍的行
  std::vector<const LineInfo*> getLines(size_t startLine, size_t endLine) const {
    std::vector<const LineInfo*> result;
    for (size_t i = startLine; i <= endLine && i <= lines_.size(); ++i) {
      if (const LineInfo *line = getLine(i)) {
        result.push_back(line);
      }
    }
    return result;
  }
  
  /// 取得所有指令行
  std::vector<const LineInfo*> getInstructionLines() const {
    std::vector<const LineInfo*> result;
    for (const auto &line : lines_) {
      if (line.isInstruction()) {
        result.push_back(&line);
      }
    }
    return result;
  }
  
  /// 取得所有標籤行
  std::vector<const LineInfo*> getLabelLines() const {
    std::vector<const LineInfo*> result;
    for (const auto &line : lines_) {
      if (line.isLabel()) {
        result.push_back(&line);
      }
    }
    return result;
  }
  
  // ------------------------------------------------------------------------
  // 查詢接口 - 指令
  // ------------------------------------------------------------------------
  
  /// 取得所有指令
  const std::vector<ParsedInstruction>& getAllInstructions() const {
    return parsedProgram_.allInstructions;
  }
  
  /// 取得指定行的指令
  const ParsedInstruction* getInstructionAtLine(size_t lineNumber) const {
    return parsedProgram_.getInstructionAtLine(lineNumber);
  }
  
  /// 按 opcode 過濾指令
  std::vector<const ParsedInstruction*> findInstructionsByOpcode(
      const std::string &opcode) const {
    std::vector<const ParsedInstruction*> result;
    for (const auto &inst : parsedProgram_.allInstructions) {
      if (inst.opcode == opcode) {
        result.push_back(&inst);
      }
    }
    return result;
  }
  
  /// 按 opcode 前綴過濾指令（例如 "s_" 可以找到所有 scalar 指令）
  std::vector<const ParsedInstruction*> findInstructionsByOpcodePrefix(
      const std::string &prefix) const {
    std::vector<const ParsedInstruction*> result;
    for (const auto &inst : parsedProgram_.allInstructions) {
      if (inst.opcode.find(prefix) == 0) {
        result.push_back(&inst);
      }
    }
    return result;
  }
  
  /// 統計指令數量
  size_t getInstructionCount() const {
    return parsedProgram_.allInstructions.size();
  }
  
  // ------------------------------------------------------------------------
  // 查詢接口 - Kernel
  // ------------------------------------------------------------------------
  
  /// 取得 Kernel 數量
  size_t getKernelCount() const { return kernels_.size(); }
  
  /// 取得所有 Kernels
  const std::vector<KernelRegion>& getAllKernels() const { return kernels_; }
  
  /// 根據名稱查找 Kernel
  const KernelRegion* findKernel(const std::string &name) const {
    auto it = kernelIndex_.find(name);
    if (it != kernelIndex_.end()) {
      return &kernels_[it->second];
    }
    return nullptr;
  }
  
  /// 查找包含指定行號的 Kernel
  const KernelRegion* findKernelByLine(size_t lineNumber) const {
    for (const auto &kernel : kernels_) {
      if (lineNumber >= kernel.startLine && lineNumber <= kernel.endLine) {
        return &kernel;
      }
    }
    return nullptr;
  }
  
  // ------------------------------------------------------------------------
  // 查詢接口 - Metadata
  // ------------------------------------------------------------------------
  
  /// 取得 Metadata
  const AMDGPUMetadata& getMetadata() const { return metadata_; }
  
  /// 檢查是否有 Metadata
  bool hasMetadata() const { return !metadata_.isEmpty(); }
  
  /// 取得目標架構
  std::string getTargetTriple() const { return metadata_.target; }
  
  /// 取得 Metadata 中的 Kernel 資訊
  const KernelInfo* getMetadataForKernel(const std::string &name) const {
    return metadata_.findKernel(name);
  }
  
  // ------------------------------------------------------------------------
  // 查詢接口 - 標籤
  // ------------------------------------------------------------------------
  
  /// 取得所有標籤
  const std::vector<ParsedLabel>& getAllLabels() const {
    return parsedProgram_.labels;
  }
  
  /// 查找標籤
  const ParsedLabel* findLabel(const std::string &name) const {
    for (const auto &label : parsedProgram_.labels) {
      if (label.name == name) {
        return &label;
      }
    }
    return nullptr;
  }
  
  // ------------------------------------------------------------------------
  // 查詢接口 - Label Blocks（Label 和指令的關係）
  // ------------------------------------------------------------------------
  
  /// 取得所有 Label Blocks
  /// 每個 Label Block 包含一個 label 和其後到下一個 label 之間的所有指令
  const std::vector<LabelBlock>& getLabelBlocks() const {
    return labelBlocks_;
  }
  
  /// 根據 label 名稱查找 Label Block
  const LabelBlock* findLabelBlock(const std::string &labelName) const {
    auto it = labelBlockIndex_.find(labelName);
    if (it != labelBlockIndex_.end()) {
      return &labelBlocks_[it->second];
    }
    return nullptr;
  }
  
  /// 根據行號查找所屬的 Label Block
  /// 返回包含該行的 Label Block，如果沒找到則返回 nullptr
  const LabelBlock* findLabelBlockForLine(size_t lineNumber) const {
    for (const auto &block : labelBlocks_) {
      if (block.containsLine(lineNumber)) {
        return &block;
      }
    }
    return nullptr;
  }
  
  /// 取得某個 Label Block 內的所有指令
  std::vector<const ParsedInstruction*> getInstructionsInLabelBlock(
      const std::string &labelName) const {
    const LabelBlock *block = findLabelBlock(labelName);
    if (block) {
      return block->instructions;
    }
    return {};
  }
  
  /// 根據指令行號查找其所屬的 Label
  /// 返回該指令所在 Label Block 的 label 名稱
  const ParsedLabel* findLabelForInstruction(size_t instructionLine) const {
    const LabelBlock *block = findLabelBlockForLine(instructionLine);
    if (block) {
      return findLabel(block->labelName);
    }
    return nullptr;
  }
  
  /// 構建 Label Blocks（需要在設置完數據後調用）
  void buildLabelBlocks();
  
  // ------------------------------------------------------------------------
  // 統計接口
  // ------------------------------------------------------------------------
  
  /// 統計各類型行數
  struct Statistics {
    size_t totalLines;
    size_t instructionCount;
    size_t labelCount;
    size_t directiveCount;
    size_t commentCount;
    size_t metadataLineCount;
    size_t kernelCount;
    
    // 指令類型統計
    size_t scalarInstructions;   // s_* 指令
    size_t vectorInstructions;   // v_* 指令
    size_t memoryInstructions;   // ds_*, global_*, flat_* 指令
    size_t branchInstructions;   // s_branch, s_cbranch 等
    
    Statistics() : totalLines(0), instructionCount(0), labelCount(0),
                   directiveCount(0), commentCount(0), metadataLineCount(0),
                   kernelCount(0), scalarInstructions(0), vectorInstructions(0),
                   memoryInstructions(0), branchInstructions(0) {}
  };
  
  /// 取得統計資訊
  Statistics getStatistics() const;
  
  /// 打印摘要
  void printSummary(llvm::raw_ostream &OS) const;
  
  /// 打印詳細資訊
  void printDetailed(llvm::raw_ostream &OS) const;
  
  // ------------------------------------------------------------------------
  // 輔助接口
  // ------------------------------------------------------------------------
  
  /// 檢查檔案是否為空
  bool isEmpty() const { return lines_.empty(); }
  
  /// 取得原始的 ParsedProgram
  const ParsedProgram& getParsedProgram() const { return parsedProgram_; }

private:
  std::string filename_;
  std::vector<LineInfo> lines_;
  std::map<size_t, size_t> lineIndex_;  // lineNumber -> index in lines_
  
  ParsedProgram parsedProgram_;
  AMDGPUMetadata metadata_;
  
  std::vector<KernelRegion> kernels_;
  std::map<std::string, size_t> kernelIndex_;  // kernelName -> index in kernels_
  
  std::vector<LabelBlock> labelBlocks_;
  std::map<std::string, size_t> labelBlockIndex_;  // labelName -> index in labelBlocks_
};

#endif // AMDGCN_ASSEMBLY_H

