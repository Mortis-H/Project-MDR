/**
 * api_usage_demo.cpp
 * 
 * 示範如何使用 AMDGCNAssembly API
 * 
 * 這個程式展示：
 * 1. 如何調用解析函數取得 AMDGCNAssembly 對象
 * 2. 如何使用 API 查詢 metadata 資訊
 * 3. 如何使用 API 查詢 labels 資訊
 * 4. 如何使用 API 查詢 instructions 資訊
 * 
 * 編譯：make api_usage_demo
 * 使用：./build/api_usage_demo input.s
 */

#include "parse_utils.h"
#include "AMDGCNAssembly.h"
#include "AMDGPUMetadata.h"

#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"

#include <iostream>

using namespace llvm;

int main(int argc, char **argv) {
  InitLLVM X(argc, argv);

  if (argc != 2) {
    errs() << "使用方式: " << argv[0] << " <input.s>\n";
    errs() << "\n";
    errs() << "這個程式展示如何使用 AMDGCNAssembly API 查詢解析後的資訊。\n";
    return 1;
  }

  std::string filename = argv[1];

  outs() << "AMDGCNAssembly API 使用示範\n";
  outs() << "\n";
  outs() << "檔案: " << filename << "\n";
  outs() << "\n";

  // ========================================================================
  // 步驟 1: 調用解析函數，取得 AMDGCNAssembly 對象
  // ========================================================================
  // API 調用：parseAMDGCNAssembly
  // 這個函數會完整解析 .s 檔案，並返回一個包含所有資訊的 AMDGCNAssembly 對象
  
  AMDGCNAssembly assembly = parseAMDGCNAssembly(filename);
  // AMDGCNAssembly assembly = parseAMDGCNAssembly(filename, "amdgcn-amd-amdhsa", "gfx950");

  // ========================================================================
  // 步驟 2: 使用 API 查詢 Metadata 資訊
  // ========================================================================
  outs() << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
  outs() << "1. Metadata 資訊\n";
  outs() << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
  outs() << "\n";

  // API 調用：assembly.hasMetadata()
  if (assembly.hasMetadata()) {
    // API 調用：assembly.getMetadata()
    const AMDGPUMetadata &meta = assembly.getMetadata();
    
    // 遍歷所有 kernels
    for (const auto &kernel : meta.kernels) {
      outs() << "Kernel Symbol: " << kernel.symbol << "\n";
      outs() << "Kernel Name:   " << kernel.name << "\n";
      outs() << "SGPR Count:    " << kernel.sgprCount << "\n";
      outs() << "AGPR Count:    " << kernel.agprCount << "\n";
      outs() << "VGPR Count:    " << kernel.vgprCount << "\n";
      outs() << "\n";
      
      // 顯示參數（按 YAML 原始順序）
      outs() << "Arguments (" << kernel.args.size() << " 個):\n";
      for (size_t i = 0; i < kernel.args.size(); ++i) {
        const auto &arg = kernel.args[i];
        outs() << "  [" << i << "] ";
        
        // 按插入順序顯示所有屬性
        const auto &props = arg.getAllProperties();
        for (size_t j = 0; j < props.size(); ++j) {
          if (j > 0) outs() << " ";
          outs() << props[j].first << "=" << props[j].second << ",";
        }
        outs() << "\n";
      }
      outs() << "\n";
    }
  } else {
    outs() << "沒有找到 metadata\n";
    outs() << "\n";
  }

  // ========================================================================
  // 步驟 3: 使用 API 查詢 Labels 資訊
  // ========================================================================
  outs() << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
  outs() << "2. Labels\n";
  outs() << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
  outs() << "\n";

  // API 調用：assembly.getAllLabels()
  const auto &labels = assembly.getAllLabels();
  outs() << "總共有 " << labels.size() << " 個 Labels:\n";
  outs() << "\n";

  // 遍歷所有 labels
  for (size_t i = 0; i < labels.size(); ++i) {
    const auto &label = labels[i];
    // 提取 label 名稱（去掉註解部分）
    std::string labelName = label.name;
    size_t commentPos = labelName.find(';');
    if (commentPos != std::string::npos) {
      labelName = labelName.substr(0, commentPos);
    }
    // 去掉尾部空白
    while (!labelName.empty() && (labelName.back() == ' ' || labelName.back() == '\t')) {
      labelName.pop_back();
    }
    // 去掉尾部的冒號
    if (!labelName.empty() && labelName.back() == ':') {
      labelName.pop_back();
    }
    outs() << "  [" << i << "] Line " << label.lineNumber << ": " << labelName << "\n";
  }
  outs() << "\n";

  // ========================================================================
  // 步驟 4: 使用 API 查詢 Instructions 資訊（前 30 個）
  // ========================================================================
  outs() << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
  outs() << "3. 前 30 行的 Instructions\n";
  outs() << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
  outs() << "\n";

  // API 調用：assembly.getAllInstructions()
  const auto &insts = assembly.getAllInstructions();
  outs() << "總共有 " << insts.size() << " 個指令\n";
  outs() << "顯示前 " << std::min(size_t(30), insts.size()) << " 個:\n";
  outs() << "\n";

  // 遍歷前 30 個指令
  for (size_t i = 0; i < std::min(size_t(30), insts.size()); ++i) {
    const auto &inst = insts[i];
    
    outs() << "[" << i << "] Line " << inst.lineNumber << ":\n";
    outs() << "  指令: " << inst.opcode << "\n";
    outs() << "  操作數 (" << inst.operands.size() << " 個): ";
    
    // 顯示所有操作數
    for (size_t j = 0; j < inst.operands.size(); ++j) {
      if (j > 0) outs() << ", ";
      outs() << inst.operands[j].text;
    }
    outs() << "\n";
  }
  outs() << "\n";

  // ========================================================================
  // 步驟 5: 使用 API 查詢 Label Blocks（Label 和 Instruction 的關係）
  // ========================================================================
  outs() << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
  outs() << "4. Label Blocks（Label 和 Instructions 的關係）\n";
  outs() << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
  outs() << "\n";

  // API 調用：assembly.getLabelBlocks()
  const auto &labelBlocks = assembly.getLabelBlocks();
  outs() << "總共有 " << labelBlocks.size() << " 個 Label Blocks\n";
  outs() << "\n";

  // 展示每個 Label Block 的資訊
  for (size_t i = 0; i < labelBlocks.size(); ++i) {
    const auto &block = labelBlocks[i];
    
    // 清理 label 名稱（去掉註解）
    std::string cleanName = block.labelName;
    size_t commentPos = cleanName.find(';');
    if (commentPos != std::string::npos) {
      cleanName = cleanName.substr(0, commentPos);
    }
    while (!cleanName.empty() && (cleanName.back() == ' ' || cleanName.back() == '\t' || cleanName.back() == ':')) {
      cleanName.pop_back();
    }
    
    outs() << "Label Block [" << i << "]: " << cleanName << "\n";
    outs() << "  範圍: Line " << block.startLine << " - " << block.endLine 
           << " (" << block.getLineCount() << " 行)\n";
    outs() << "  指令數: " << block.instructions.size() << "\n";
    
    // 顯示前 3 個指令
    if (!block.instructions.empty()) {
      outs() << "  前 " << std::min(size_t(3), block.instructions.size()) << " 個指令:\n";
      for (size_t j = 0; j < std::min(size_t(3), block.instructions.size()); ++j) {
        const auto *inst = block.instructions[j];
        outs() << "    Line " << inst->lineNumber << ": " << inst->opcode << "\n";
      }
    }
    outs() << "\n";
  }

  return 0;
}
