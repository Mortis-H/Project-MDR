//===- AMDISAToAssembly.cpp - Reconstruct full .s from GPU MLIR ----------===//
//
// This file implements the reconstruction of a complete, compilable AMD GCN
// assembly file from GPU dialect MLIR containing inline assembly.
//
// It extracts:
//   - Module attributes (target triple, code object version, kernel name, args)
//   - Inline assembly instructions from llvm.inline_asm
//   - Reconstructs proper .s file structure with headers and AMDHSA metadata
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;

namespace {

/// Helper to extract string attribute value
std::string getStringAttr(ModuleOp module, StringRef attrName, 
                           StringRef defaultVal = "") {
  if (auto attr = module->getAttrOfType<StringAttr>(attrName)) {
    std::string val = attr.getValue().str();
    // Remove surrounding quotes if present
    if (val.size() >= 2 && val.front() == '"' && val.back() == '"') {
      return val.substr(1, val.size() - 2);
    }
    return val;
  }
  return defaultVal.str();
}

/// Helper to extract integer attribute value
int64_t getIntAttr(ModuleOp module, StringRef attrName, int64_t defaultVal = 0) {
  if (auto attr = module->getAttrOfType<IntegerAttr>(attrName)) {
    return attr.getInt();
  }
  return defaultVal;
}

/// Extract kernel arguments array from module attributes
std::vector<std::map<std::string, std::string>> 
extractKernelArgs(ModuleOp module) {
  std::vector<std::map<std::string, std::string>> args;
  
  auto kernargsAttr = module->getAttrOfType<ArrayAttr>("amdisa.kernargs");
  if (!kernargsAttr) return args;
  
  for (auto elemAttr : kernargsAttr) {
    auto dictAttr = mlir::dyn_cast<DictionaryAttr>(elemAttr);
    if (!dictAttr) continue;
    
    std::map<std::string, std::string> argInfo;
    for (auto namedAttr : dictAttr) {
      std::string name = namedAttr.getName().str();
      if (auto strAttr = mlir::dyn_cast<StringAttr>(namedAttr.getValue())) {
        argInfo[name] = strAttr.getValue().str();
      } else if (auto intAttr = mlir::dyn_cast<IntegerAttr>(namedAttr.getValue())) {
        argInfo[name] = std::to_string(intAttr.getInt());
      }
    }
    args.push_back(argInfo);
  }
  
  return args;
}

/// Generate AMDHSA kernel metadata using preserved directives from original .s
void emitAMDHSAMetadata(llvm::raw_ostream &os,
                        StringRef kernelName,
                        const std::map<std::string, std::string> &directives) {
  // Note: .section and .p2align directives before .amdhsa_kernel
  // are preserved from the original file ([O] lines), so we don't emit them here
  os << "\t.amdhsa_kernel " << kernelName << "\n";
  
  // List of all expected directives in the correct order
  // This ensures we emit them exactly as the original compiler did
  std::vector<std::string> directiveOrder = {
    ".amdhsa_group_segment_fixed_size",
    ".amdhsa_private_segment_fixed_size",
    ".amdhsa_kernarg_size",
    ".amdhsa_user_sgpr_count",
    ".amdhsa_user_sgpr_dispatch_ptr",
    ".amdhsa_user_sgpr_queue_ptr",
    ".amdhsa_user_sgpr_kernarg_segment_ptr",
    ".amdhsa_user_sgpr_dispatch_id",
    ".amdhsa_user_sgpr_kernarg_preload_length",
    ".amdhsa_user_sgpr_kernarg_preload_offset",
    ".amdhsa_user_sgpr_private_segment_size",
    ".amdhsa_uses_dynamic_stack",
    ".amdhsa_enable_private_segment",
    ".amdhsa_system_sgpr_workgroup_id_x",
    ".amdhsa_system_sgpr_workgroup_id_y",
    ".amdhsa_system_sgpr_workgroup_id_z",
    ".amdhsa_system_sgpr_workgroup_info",
    ".amdhsa_system_vgpr_workitem_id",
    ".amdhsa_next_free_vgpr",
    ".amdhsa_next_free_sgpr",
    ".amdhsa_accum_offset",
    ".amdhsa_reserve_vcc",
    ".amdhsa_float_round_mode_32",
    ".amdhsa_float_round_mode_16_64",
    ".amdhsa_float_denorm_mode_32",
    ".amdhsa_float_denorm_mode_16_64",
    ".amdhsa_dx10_clamp",
    ".amdhsa_ieee_mode",
    ".amdhsa_fp16_overflow",
    ".amdhsa_tg_split",
    ".amdhsa_exception_fp_ieee_invalid_op",
    ".amdhsa_exception_fp_denorm_src",
    ".amdhsa_exception_fp_ieee_div_zero",
    ".amdhsa_exception_fp_ieee_overflow",
    ".amdhsa_exception_fp_ieee_underflow",
    ".amdhsa_exception_fp_ieee_inexact",
    ".amdhsa_exception_int_div_zero"
  };
  
  // Emit directives in the correct order, using preserved values
  for (const auto &directiveName : directiveOrder) {
    auto it = directives.find(directiveName);
    if (it != directives.end()) {
      os << "\t\t" << directiveName << " " << it->second << "\n";
    }
  }
  
  os << "\t.end_amdhsa_kernel\n";
}

} // anonymous namespace

namespace mlir {
namespace amdisa {

/// Reconstruct a complete AMD GCN assembly file from GPU MLIR
int reconstructAssemblyFromGPU(ModuleOp module, llvm::raw_ostream &os) {
  // Check if we have the complete original file with structure
  std::string fullOriginal = getStringAttr(module, "amdisa.full_original", "");
  
  if (!fullOriginal.empty()) {
    // Use the complete original approach: output original lines but replace instructions
    
    // First, collect new instructions from inline_asm
    std::string newInstructions;
    llvm::raw_string_ostream instOS(newInstructions);
    bool foundInlineAsm = false;
    
    module->walk([&](LLVM::InlineAsmOp inlineAsmOp) {
      foundInlineAsm = true;
      std::string asmStr = inlineAsmOp.getAsmString().str();
      
      llvm::SmallVector<llvm::StringRef, 128> lines;
      llvm::StringRef(asmStr).split(lines, '\n');
      
      for (auto line : lines) {
        if (!line.empty()) {
          instOS << line << "\n";
        }
      }
    });
    
    if (!foundInlineAsm) {
      llvm::errs() << "Warning: No inline_asm found. Using complete original.\n";
    }
    
    //  Extract .amdhsa_kernel directives
    std::map<std::string, std::string> amdhdaDirectives;
    if (auto directivesAttr = module->getAttrOfType<mlir::ArrayAttr>("amdisa.amdhsa_directives")) {
      for (auto attr : directivesAttr) {
        if (auto dictAttr = mlir::dyn_cast<mlir::DictionaryAttr>(attr)) {
          std::string name, value;
          if (auto nameAttr = dictAttr.get("name")) {
            if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(nameAttr)) {
              name = strAttr.getValue().str();
            }
          }
          if (auto valueAttr = dictAttr.get("value")) {
            if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(valueAttr)) {
              value = strAttr.getValue().str();
            }
          }
          if (!name.empty() && !value.empty()) {
            amdhdaDirectives[name] = value;
          }
        }
      }
    }
    
    // Now output original lines, using closed-block awareness
    // Key insight: .amdhsa_kernel and .amdgpu_metadata are closed blocks
    // that should be replaced entirely
    llvm::SmallVector<llvm::StringRef, 0> origLines;
    llvm::StringRef(fullOriginal).split(origLines, '\n');
    
    bool emittedInstructions = false;
    bool insideAmdhsaBlock = false;
    bool insideMetadataBlock = false;
    bool insideInstructionRegion = false;
    std::string kernelName = getStringAttr(module, "amdisa.kernel_name", "unknown_kernel");
    
    for (llvm::StringRef line : origLines) {
      if (line.empty()) continue;
      
      // Parse line type: [X]content
      char type = 'O';
      llvm::StringRef content = line;
      if (line.starts_with("[") && line.size() > 2 && line[2] == ']') {
        type = line[1];
        content = line.substr(3);
      }
      
      // Track instruction region (from first [I] to first [K] or [M])
      // IMPORTANT: This must come BEFORE handling [K] and [M] blocks
      if (type == 'I' && !insideInstructionRegion) {
        insideInstructionRegion = true;
      }
      if ((type == 'K' || type == 'M') && insideInstructionRegion) {
        insideInstructionRegion = false;
      }
      
      // Handle .amdhsa_kernel closed block
      if (type == 'K') {
        if (content.starts_with("\t.amdhsa_kernel") || content.starts_with(".amdhsa_kernel")) {
          // Start of .amdhsa_kernel block: emit regenerated version
          emitAMDHSAMetadata(os, kernelName, amdhdaDirectives);
          insideAmdhsaBlock = true;
        } else if (content.starts_with("\t.end_amdhsa_kernel") || content.starts_with(".end_amdhsa_kernel")) {
          // End of .amdhsa_kernel block
          insideAmdhsaBlock = false;
        }
        // Skip all lines in .amdhsa_kernel block (already regenerated)
        continue;
      }
      
      // Handle .amdgpu_metadata closed block
      if (type == 'M') {
        if (content.starts_with("\t.amdgpu_metadata") || content.starts_with(".amdgpu_metadata")) {
          // Start of .amdgpu_metadata block: emit preserved version
          std::string rawMetadata = getStringAttr(module, "amdisa.raw_metadata", "");
          if (!rawMetadata.empty()) {
            os << "\n\t.amdgpu_metadata\n";
            os << rawMetadata;
            if (rawMetadata.empty() || rawMetadata.back() != '\n') {
              os << "\n";
            }
            os << "\t.end_amdgpu_metadata\n";
          }
          insideMetadataBlock = true;
        } else if (content.starts_with("\t.end_amdgpu_metadata") || content.starts_with(".end_amdgpu_metadata")) {
          // End of .amdgpu_metadata block
          insideMetadataBlock = false;
        }
        // Skip all lines in .amdgpu_metadata block (already regenerated)
        continue;
      }
      
      // Handle instructions: emit new ones once, skip originals
      if (type == 'I') {
        if (!emittedInstructions && foundInlineAsm) {
          os << newInstructions;
          emittedInstructions = true;
        }
        // Skip all original instruction lines
        continue;
      }
      
      // Handle labels
      if (type == 'L') {
        // Kernel label: always output (it's the function entry point)
        if (content.starts_with(kernelName)) {
          os << content << "\n";
          continue;
        }
        
        // Extract label name (without trailing comments)
        llvm::StringRef labelName = content;
        size_t colonPos = labelName.find(':');
        if (colonPos != llvm::StringRef::npos) {
          labelName = labelName.substr(0, colonPos + 1);  // Include the ':'
        }
        labelName = labelName.trim();
        
        // Check if this label is already in the new instructions (from inline_asm)
        if (newInstructions.find(labelName.str()) != std::string::npos) {
          // Label already in inline_asm: skip
          continue;
        }
        
        // Other labels outside instruction region: output
        if (!insideInstructionRegion) {
          os << content << "\n";
        }
        // Labels inside instruction region are skipped
        continue;
      }
      
      // Handle other lines (directives, comments, .set, etc.)
      if (type == 'O') {
        if (content.starts_with(";")) {
          // Skip ALL comments in the kernel
          // Comments are purely for readability and not required for compilation.
          // This includes basic block markers ("; %bb.X:"), kernel info, and other annotations.
          continue;
        }
        // All other [O] lines: output as-is
        os << content << "\n";
      }
    }
    
    return 0;
  }
  
  // Fallback: use old method
  // Extract metadata from module attributes
  std::string targetTriple = getStringAttr(module, "llvm.target_triple", 
                                           "amdgcn-amd-amdhsa--gfx950");
  int64_t codeObjectVersion = getIntAttr(module, "amdgpu.code_object_version", 6);
  std::string kernelName = getStringAttr(module, "amdisa.kernel_name", "unknown_kernel");
  auto kernelArgs = extractKernelArgs(module);
  
  // Check if we have raw metadata saved from parsing
  std::string rawMetadata = getStringAttr(module, "amdisa.raw_metadata", "");
  
  // Emit default prologue
  os << "\t.amdgcn_target \"" << targetTriple << "\"\n";
  os << "\t.amdhsa_code_object_version " << codeObjectVersion << "\n";
  os << "\t.text\n";
  os << "\t.globl\t" << kernelName << "\n";
  os << "\t.p2align\t8\n";
  os << "\t.type\t" << kernelName << ",@function\n";
  os << kernelName << ":\n";
  
  // Extract instructions from inline_asm in gpu.func
  bool foundInlineAsm = false;
  module->walk([&](LLVM::InlineAsmOp inlineAsmOp) {
    foundInlineAsm = true;
    std::string asmStr = inlineAsmOp.getAsmString().str();
    
    // The inline asm string contains instructions separated by \n
    // We need to output them with proper formatting
    llvm::SmallVector<llvm::StringRef, 128> lines;
    llvm::StringRef(asmStr).split(lines, '\n');
    
    for (auto line : lines) {
      if (!line.empty()) {
        // Instructions are already formatted with \t prefix from AMDISA
        os << line << "\n";
      }
    }
  });
  
  if (!foundInlineAsm) {
    llvm::errs() << "Warning: No inline_asm found in GPU module. "
                 << "Did you run -emit=gpuinlineasm first?\n";
    return 1;
  }
  
  // Extract preserved .amdhsa_kernel directives from module attributes
  std::map<std::string, std::string> amdhdaDirectives;
  
  if (auto directivesAttr = module->getAttrOfType<mlir::ArrayAttr>("amdisa.amdhsa_directives")) {
    for (auto attr : directivesAttr) {
      if (auto dictAttr = mlir::dyn_cast<mlir::DictionaryAttr>(attr)) {
        std::string name, value;
        if (auto nameAttr = dictAttr.get("name")) {
          if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(nameAttr)) {
            name = strAttr.getValue().str();
          }
        }
        if (auto valueAttr = dictAttr.get("value")) {
          if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(valueAttr)) {
            value = strAttr.getValue().str();
          }
        }
        if (!name.empty() && !value.empty()) {
          amdhdaDirectives[name] = value;
        }
      }
    }
  }
  
  // Emit AMDHSA metadata using preserved directives from original .s
  // This avoids cross-contamination between directives and metadata
  emitAMDHSAMetadata(os, kernelName, amdhdaDirectives);
  
  // Try to use epilogue if available, otherwise generate minimal epilogue
  std::string epilogueContent = getStringAttr(module, "amdisa.epilogue", "");
  if (!epilogueContent.empty()) {
    os << epilogueContent;
  } else {
    // Fallback: emit minimal epilogue
    os << "\t.text\n";
  }
  os << "\n";
  
  // If we have raw metadata from the original .s file, emit it
  if (!rawMetadata.empty()) {
    os << "\n";
    os << "\t.amdgpu_metadata\n";
    os << rawMetadata;
    if (rawMetadata.empty() || rawMetadata.back() != '\n') {
      os << "\n";
    }
    os << "\t.end_amdgpu_metadata\n";
  }
  
  return 0;
}

} // namespace amdisa
} // namespace mlir

