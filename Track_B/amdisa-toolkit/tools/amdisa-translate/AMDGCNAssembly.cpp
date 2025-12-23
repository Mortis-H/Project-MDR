#include "AMDGCNAssembly.h"
#include <algorithm>

// ============================================================================
// 統計實現
// ============================================================================

AMDGCNAssembly::Statistics AMDGCNAssembly::getStatistics() const {
  Statistics stats;
  
  stats.totalLines = lines_.size();
  stats.kernelCount = kernels_.size();
  stats.labelCount = parsedProgram_.labels.size();
  stats.instructionCount = parsedProgram_.allInstructions.size();
  
  // 統計各類型行數
  for (const auto &line : lines_) {
    switch (line.kind) {
      case LineKind::Directive:
        stats.directiveCount++;
        break;
      case LineKind::Comment:
        stats.commentCount++;
        break;
      case LineKind::Metadata:
        stats.metadataLineCount++;
        break;
      default:
        break;
    }
  }
  
  // 統計指令類型
  for (const auto &inst : parsedProgram_.allInstructions) {
    // Scalar 指令
    if (inst.opcode.find("s_") == 0) {
      stats.scalarInstructions++;
      
      // Branch 指令
      if (inst.opcode.find("s_branch") == 0 || 
          inst.opcode.find("s_cbranch") == 0 ||
          inst.opcode.find("s_setpc") == 0) {
        stats.branchInstructions++;
      }
    }
    // Vector 指令
    else if (inst.opcode.find("v_") == 0) {
      stats.vectorInstructions++;
    }
    // Memory 指令
    else if (inst.opcode.find("ds_") == 0 ||
             inst.opcode.find("global_") == 0 ||
             inst.opcode.find("flat_") == 0 ||
             inst.opcode.find("buffer_") == 0 ||
             inst.opcode.find("scratch_") == 0) {
      stats.memoryInstructions++;
    }
  }
  
  return stats;
}

// ============================================================================
// 打印實現
// ============================================================================

void AMDGCNAssembly::printSummary(llvm::raw_ostream &OS) const {
  Statistics stats = getStatistics();
  
  OS << "╔═══════════════════════════════════════════════════════════════╗\n";
  OS << "║           AMD GCN Assembly Analysis Summary                  ║\n";
  OS << "╚═══════════════════════════════════════════════════════════════╝\n";
  OS << "\n";
  
  // 檔案資訊
  OS << "File: " << filename_ << "\n";
  OS << "\n";
  
  // 基本統計
  OS << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
  OS << "Basic Statistics\n";
  OS << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
  OS << "Total Lines:        " << stats.totalLines << "\n";
  OS << "Instructions:       " << stats.instructionCount << "\n";
  OS << "Labels:             " << stats.labelCount << "\n";
  OS << "Directives:         " << stats.directiveCount << "\n";
  OS << "Comments:           " << stats.commentCount << "\n";
  OS << "Metadata Lines:     " << stats.metadataLineCount << "\n";
  OS << "Kernels:            " << stats.kernelCount << "\n";
  OS << "\n";
  
  // 指令分類
  OS << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
  OS << "Instruction Breakdown\n";
  OS << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
  OS << "Scalar (s_*):       " << stats.scalarInstructions 
     << " (" << (stats.instructionCount > 0 ? 
                (stats.scalarInstructions * 100 / stats.instructionCount) : 0) 
     << "%)\n";
  OS << "Vector (v_*):       " << stats.vectorInstructions
     << " (" << (stats.instructionCount > 0 ?
                (stats.vectorInstructions * 100 / stats.instructionCount) : 0)
     << "%)\n";
  OS << "Memory Access:      " << stats.memoryInstructions
     << " (" << (stats.instructionCount > 0 ?
                (stats.memoryInstructions * 100 / stats.instructionCount) : 0)
     << "%)\n";
  OS << "Branch:             " << stats.branchInstructions
     << " (" << (stats.instructionCount > 0 ?
                (stats.branchInstructions * 100 / stats.instructionCount) : 0)
     << "%)\n";
  OS << "\n";
  
  // Metadata 資訊
  if (hasMetadata()) {
    OS << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    OS << "Metadata Summary\n";
    OS << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    OS << "Target:             " << metadata_.target << "\n";
    OS << "Version:            ";
    for (size_t i = 0; i < metadata_.version.size(); ++i) {
      OS << metadata_.version[i];
      if (i < metadata_.version.size() - 1) OS << ".";
    }
    OS << "\n";
    OS << "Kernels in Metadata: " << metadata_.kernels.size() << "\n";
    OS << "\n";
  }
  
  // Kernel 資訊
  if (!kernels_.empty()) {
    OS << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    OS << "Kernel Regions\n";
    OS << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    for (const auto &kernel : kernels_) {
      OS << "• " << kernel.name << "\n";
      OS << "  Lines:        " << kernel.startLine << "-" << kernel.endLine
         << " (" << kernel.getLineCount() << " lines)\n";
      OS << "  Instructions: " << kernel.instructions.size() << "\n";
      OS << "  Labels:       " << kernel.labels.size() << "\n";
      
      if (kernel.hasMetadata()) {
        const KernelInfo *meta = kernel.metadata;
        OS << "  Resources:\n";
        OS << "    AGPR: " << meta->agprCount << "\n";
        OS << "    SGPR: " << meta->sgprCount << "\n";
        OS << "    VGPR: " << meta->vgprCount << "\n";
        OS << "  Workgroup Size: " << meta->maxFlatWorkgroupSize << "\n";
      }
      OS << "\n";
    }
  }
}

void AMDGCNAssembly::printDetailed(llvm::raw_ostream &OS) const {
  // 先打印摘要
  printSummary(OS);
  
  // 詳細的 Kernel 資訊
  if (!kernels_.empty()) {
    OS << "╔═══════════════════════════════════════════════════════════════╗\n";
    OS << "║                    Detailed Kernel Analysis                  ║\n";
    OS << "╚═══════════════════════════════════════════════════════════════╝\n";
    OS << "\n";
    
    for (const auto &kernel : kernels_) {
      OS << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
      OS << "Kernel: " << kernel.name << "\n";
      OS << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
      OS << "\n";
      
      // Metadata
      if (kernel.hasMetadata()) {
        kernel.metadata->print(OS);
        OS << "\n";
      }
      
      // 指令統計
      std::map<std::string, int> opcodeCount;
      for (const auto &inst : kernel.instructions) {
        opcodeCount[inst.opcode]++;
      }
      
      OS << "Instruction Distribution (Top 10):\n";
      std::vector<std::pair<std::string, int>> sortedOpcodes(
          opcodeCount.begin(), opcodeCount.end());
      std::sort(sortedOpcodes.begin(), sortedOpcodes.end(),
                [](const auto &a, const auto &b) { return a.second > b.second; });
      
      int count = 0;
      for (const auto &pair : sortedOpcodes) {
        OS << "  " << pair.first << ": " << pair.second << "\n";
        if (++count >= 10) break;
      }
      OS << "\n";
      
      // 標籤列表
      if (!kernel.labels.empty()) {
        OS << "Labels:\n";
        for (const auto &label : kernel.labels) {
          OS << "  " << label.name << " (line " << label.lineNumber << ")\n";
        }
        OS << "\n";
      }
    }
  }
  
  // 完整的 Metadata
  if (hasMetadata()) {
    OS << "╔═══════════════════════════════════════════════════════════════╗\n";
    OS << "║                    Full Metadata Details                     ║\n";
    OS << "╚═══════════════════════════════════════════════════════════════╝\n";
    OS << "\n";
    metadata_.printSummary(OS);
  }
}

// ============================================================================
// Label Blocks 構建
// ============================================================================

void AMDGCNAssembly::buildLabelBlocks() {
  labelBlocks_.clear();
  labelBlockIndex_.clear();
  
  if (parsedProgram_.labels.empty()) {
    return;  // 沒有 labels，直接返回
  }
  
  // 將所有 labels 按行號排序
  std::vector<const ParsedLabel*> sortedLabels;
  for (const auto &label : parsedProgram_.labels) {
    sortedLabels.push_back(&label);
  }
  std::sort(sortedLabels.begin(), sortedLabels.end(),
            [](const ParsedLabel *a, const ParsedLabel *b) {
              return a->lineNumber < b->lineNumber;
            });
  
  // 為每個 label 創建一個 LabelBlock
  for (size_t i = 0; i < sortedLabels.size(); ++i) {
    const ParsedLabel *currentLabel = sortedLabels[i];
    
    LabelBlock block;
    block.labelName = currentLabel->name;
    block.labelLine = currentLabel->lineNumber;
    block.startLine = currentLabel->lineNumber;
    
    // 確定 block 的結束行號
    if (i + 1 < sortedLabels.size()) {
      // 到下一個 label 的前一行
      block.endLine = sortedLabels[i + 1]->lineNumber - 1;
    } else {
      // 最後一個 label，到檔案結尾
      block.endLine = lines_.size();
    }
    
    // 收集這個 block 內的所有指令
    for (const auto &inst : parsedProgram_.allInstructions) {
      if (inst.lineNumber > block.startLine && inst.lineNumber <= block.endLine) {
        block.instructions.push_back(&inst);
      }
    }
    
    // 添加到 labelBlocks_
    labelBlockIndex_[block.labelName] = labelBlocks_.size();
    labelBlocks_.push_back(block);
  }
}

