//===- AMDISAAsmParser.cpp - AMD ISA Assembly Parser ---------------------===//
//
// Combined header + implementation into one file
//
// This parser reads AMD GCN ISA (.s) assembly lines and creates
// amdisa.inst ops inside an MLIR Module.
//
// Expected input:
//   v_add_f32 v1, v2, v3
//   s_waitcnt vmcnt(0)
//
// Output ops:
//   amdisa.inst { mnemonic = "...", operands = [...], raw_text = "..." }
//
// This merged version allows debugging compilation issues caused
// by header/include separation.
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

// If your AMDISA dialect ops are defined here:
#include "mlir/Dialect/AMDISA/IR/AMDISAOps.h"

#include "parse_utils.h"
#include "AMDGCNAssembly.h"
#include "AMDGPUMetadata.h"
#include "ParsedProgram.h"

static int parseIntOr(llvm::StringRef s, int def = 0) {
  s = s.trim();
  int v = def;
  if (!s.getAsInteger(10, v)) return v;
  return def;
}

template <typename ArgT>
static mlir::DictionaryAttr
propsToDictAttr(mlir::OpBuilder &b, const ArgT &arg) {
  llvm::SmallVector<mlir::NamedAttribute> kvs;

  for (auto &p : arg.getAllProperties()) {
    llvm::StringRef k = p.first;
    llvm::StringRef v = p.second;

    if (k == "size" || k == "offset" || k == "align" ||
        k == "pointee_align") {
      int iv = 0;
      if (!v.getAsInteger(10, iv))
        kvs.emplace_back(b.getStringAttr(k),
                          b.getI32IntegerAttr(iv));
    } else {
      kvs.emplace_back(b.getStringAttr(k),
                        b.getStringAttr(v));
    }
  }

  return mlir::DictionaryAttr::get(b.getContext(), kvs);
}


namespace mlir {
class MLIRContext;
} // namespace mlir

namespace mlir::amdisa {

//===----------------------------------------------------------------------===//
// AMDISAAsmParser Class Definition
//===----------------------------------------------------------------------===//

class AMDISAAsmParser {
public:
  explicit AMDISAAsmParser(StringRef filename);

  mlir::OwningOpRef<mlir::ModuleOp>
  parseModule(mlir::MLIRContext &context);

private:
  llvm::StringRef filename_;
};


//===----------------------------------------------------------------------===//
// Parse entire buffer into a ModuleOp
//===----------------------------------------------------------------------===//

mlir::OwningOpRef<mlir::ModuleOp>
AMDISAAsmParser::parseModule(mlir::MLIRContext &context) {

  mlir::OpBuilder builder(&context);
  auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToEnd(module.getBody());

  AMDGCNAssembly assembly = parseAMDGCNAssembly(filename_.str());

  // Track whether we're inside the kernel function
  // Start when we see the kernel label, end when we see a .section directive
  bool insideKernelFunction = false;
  std::string kernelLabelName;
  
  for (size_t lineNum = 1; lineNum <= assembly.getLineCount(); ++lineNum) {
    const LineInfo *line = assembly.getLine(lineNum);
    if (!line) continue;

    auto loc = builder.getUnknownLoc();

    switch (line->kind) {

      case LineKind::Label: {
        std::string labelName = line->labelName;
        
        // 提取純粹的 label 名稱（移除註釋部分）
        // 例如："vec_add: ; @vec_add" → "vec_add"
        llvm::StringRef labelRef(labelName);
        size_t colonPos = labelRef.find(':');
        if (colonPos != llvm::StringRef::npos) {
          labelRef = labelRef.substr(0, colonPos).trim();
        }
        std::string pureLabelName = labelRef.str();
        
        // Get kernel name (might have been set earlier)
        llvm::StringRef kname;
        if (auto a = module->getAttrOfType<mlir::StringAttr>("amdisa.kernel_name"))
          kname = a.getValue();
        
        // Check if this is the kernel label (start of kernel function)
        if (!kname.empty() && pureLabelName == kname.str()) {
          insideKernelFunction = true;
          kernelLabelName = pureLabelName;
          break;  // Skip kernel label itself (will be created by gpu.func)
        }
        
        // 1. 跳過 .Lfunc_end 開頭的 label（函數結束標記，會由外層生成）
        if (pureLabelName.rfind(".Lfunc_end", 0) == 0) {
          insideKernelFunction = false;  // End of kernel function
          break;
        }
        
        // 2. Only create LabelOp if inside kernel function
        if (insideKernelFunction) {
          auto nameAttr = builder.getStringAttr(labelName);
          builder.create<LabelOp>(loc, nameAttr);
        }
        break;
      }

    case LineKind::Instruction: {
      // Only create InstOp if inside kernel function
      if (!insideKernelFunction) {
        break;
      }
      
      const ParsedInstruction &inst = *line->instruction;

      auto mnemonicAttr = builder.getStringAttr(inst.opcode);

      llvm::SmallVector<mlir::Attribute> operandAttrs;
      operandAttrs.reserve(inst.operands.size());
      for (const auto &op : inst.operands) {
        operandAttrs.push_back(builder.getStringAttr(op.text));
      }

      auto opsAttr = builder.getArrayAttr(operandAttrs);
      auto rawAttr = builder.getStringAttr(inst.originalText);

      builder.create<InstOp>(loc, mnemonicAttr, opsAttr, rawAttr);
      break;
    }

    case LineKind::KernelName: {
      // .globl amdisa_kernel

      if (!module->hasAttr("amdisa.kernel_name")) {
        auto nameAttr = builder.getStringAttr(line->kernelName);
        module->setAttr("amdisa.kernel_name", nameAttr);
      }
      break;
    }

    case LineKind::AmdgcnTarget: {
      // .amdgcn_target "amdgcn-amd-amdhsa--gfx950"

      if (!module->hasAttr("llvm.target_triple")) {
        auto tripleAttr = builder.getStringAttr(line->amdgcnTarget);
        module->setAttr("llvm.target_triple", tripleAttr);
      }
      break;
    }

    case LineKind::AmdhsaCodeObjectVersion: {
      if (!module->hasAttr("amdgpu.code_object_version")) {
        int version = 0;

        // line->amdhsaCodeObjectVersion (std::string)

        if (!line->amdhsaCodeObjectVersion.empty()) {
          version = std::stoi(line->amdhsaCodeObjectVersion);
        }

        auto verAttr = builder.getI32IntegerAttr(version);
        module->setAttr("amdgpu.code_object_version", verAttr);
      }
      break;
    }

    case LineKind::Directive:
      break;

    case LineKind::Comment:
      break;

    case LineKind::Metadata:
      break;

    case LineKind::Unknown:
    default:
      break;
    }
  }

  // Ultimate simple approach: save complete original file content
  // and mark which lines are instructions
  std::map<std::string, std::string> amdhdaDirectives;
  llvm::SmallVector<std::string> allLines;
  llvm::SmallVector<bool> isInstructionLine;
  llvm::SmallVector<bool> isLabelLine;
  llvm::SmallVector<bool> isAmdhsaKernelSection;
  llvm::SmallVector<bool> isMetadataSection;
  bool inAmdhdaKernel = false;
  bool inMetadata = false;
  
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr = 
      llvm::MemoryBuffer::getFileOrSTDIN(filename_);
  if (fileOrErr && fileOrErr.get()) {
    llvm::StringRef fileContent = fileOrErr.get()->getBuffer();
    llvm::SmallVector<llvm::StringRef, 0> lines;
    fileContent.split(lines, '\n');
    
    // Analyze each line and classify it
    // Key insight: .amdhsa_kernel and .amdgpu_metadata are closed blocks
    for (llvm::StringRef lineRef : lines) {
      llvm::StringRef trimmed = lineRef.trim();
      std::string lineStr = lineRef.str();
      
      // Track .amdhsa_kernel closed block
      if (trimmed.starts_with(".amdhsa_kernel")) {
        inAmdhdaKernel = true;
        // Mark this line as start of kernel block
        allLines.push_back(lineStr);
        isInstructionLine.push_back(false);
        isLabelLine.push_back(false);
        isAmdhsaKernelSection.push_back(true);
        isMetadataSection.push_back(false);
        continue;
      } else if (trimmed.starts_with(".end_amdhsa_kernel")) {
        // Mark this line as end of kernel block
        allLines.push_back(lineStr);
        isInstructionLine.push_back(false);
        isLabelLine.push_back(false);
        isAmdhsaKernelSection.push_back(true);
        isMetadataSection.push_back(false);
        inAmdhdaKernel = false;
        continue;
      } else if (inAmdhdaKernel) {
        // Inside .amdhsa_kernel block: extract directives and mark as [K]
        if (trimmed.starts_with(".amdhsa_")) {
          auto splitPair = trimmed.split(' ');
          if (splitPair.second.empty()) {
            splitPair = trimmed.split('\t');
          }
          if (!splitPair.second.empty()) {
            std::string directiveName = splitPair.first.str();
            std::string value = splitPair.second.trim().str();
            if (!value.empty()) {
              amdhdaDirectives[directiveName] = value;
            }
          }
        }
        allLines.push_back(lineStr);
        isInstructionLine.push_back(false);
        isLabelLine.push_back(false);
        isAmdhsaKernelSection.push_back(true);
        isMetadataSection.push_back(false);
        continue;
      }
      
      // Track .amdgpu_metadata closed block
      if (trimmed.starts_with(".amdgpu_metadata")) {
        inMetadata = true;
        allLines.push_back(lineStr);
        isInstructionLine.push_back(false);
        isLabelLine.push_back(false);
        isAmdhsaKernelSection.push_back(false);
        isMetadataSection.push_back(true);
        continue;
      } else if (trimmed.starts_with(".end_amdgpu_metadata")) {
        allLines.push_back(lineStr);
        isInstructionLine.push_back(false);
        isLabelLine.push_back(false);
        isAmdhsaKernelSection.push_back(false);
        isMetadataSection.push_back(true);
        inMetadata = false;
        continue;
      } else if (inMetadata) {
        // Inside .amdgpu_metadata block: mark as [M]
        allLines.push_back(lineStr);
        isInstructionLine.push_back(false);
        isLabelLine.push_back(false);
        isAmdhsaKernelSection.push_back(false);
        isMetadataSection.push_back(true);
        continue;
      }
      
      // Normal line classification (outside closed blocks)
      bool isInst = false;
      bool isLbl = false;
      
      if (!trimmed.empty()) {
        // Comments (starting with ';') are never labels or instructions
        if (trimmed.starts_with(";")) {
          // Keep as [O] (Other)
        }
        // Check if it's a label (ends with ':' or contains ':' followed by whitespace/comment)
        // This includes local labels like .Lfunc_end0: and .LBB0_2:
        else if (trimmed.ends_with(":") || trimmed.contains(": ") || trimmed.contains(":\t")) {
          isLbl = true;
        }
        // Check if it's an instruction (not starting with '.' and contains instruction patterns)
        else if (!trimmed.starts_with(".")) {
          if (trimmed.contains("s_") || trimmed.contains("v_") || 
              trimmed.contains("global_") || trimmed.contains("buffer_") ||
              trimmed.contains("_e32") || trimmed.contains("_e64")) {
            isInst = true;
          }
        }
      }
      
      // Save line with classification
      allLines.push_back(lineStr);
      isInstructionLine.push_back(isInst);
      isLabelLine.push_back(isLbl);
      isAmdhsaKernelSection.push_back(false);
      isMetadataSection.push_back(false);
    }
    
    // Save the complete original content as a JSON-like structure
    std::string structuredContent;
    for (size_t i = 0; i < allLines.size(); i++) {
      // Format: [type]line_content
      // type: I=instruction, L=label, K=amdhsa_kernel, M=metadata, O=other
      char type = 'O';
      if (isInstructionLine[i]) type = 'I';
      else if (isLabelLine[i]) type = 'L';
      else if (isAmdhsaKernelSection[i]) type = 'K';
      else if (isMetadataSection[i]) type = 'M';
      
      structuredContent += std::string("[") + type + "]" + allLines[i] + "\n";
    }
    
    module->setAttr("amdisa.full_original", builder.getStringAttr(structuredContent));
  }
  
  // Save .amdhsa_kernel directives as a single attribute
  if (!amdhdaDirectives.empty()) {
    llvm::SmallVector<mlir::Attribute> directiveAttrs;
    for (const auto &[key, value] : amdhdaDirectives) {
      llvm::SmallVector<mlir::NamedAttribute> props;
      props.push_back(builder.getNamedAttr("name", builder.getStringAttr(key)));
      props.push_back(builder.getNamedAttr("value", builder.getStringAttr(value)));
      directiveAttrs.push_back(builder.getDictionaryAttr(props));
    }
    module->setAttr("amdisa.amdhsa_directives", builder.getArrayAttr(directiveAttrs));
  }

  if (assembly.hasMetadata()) {
    const AMDGPUMetadata &meta = assembly.getMetadata();
    
    // 保存原始 metadata YAML 以便重建完整的 .s
    if (!meta.rawYAML.empty()) {
      module->setAttr("amdisa.raw_metadata", 
                      builder.getStringAttr(meta.rawYAML));
    }

    llvm::StringRef kname;
    if (auto a = module->getAttrOfType<mlir::StringAttr>("amdisa.kernel_name"))
      kname = a.getValue();

    llvm::SmallVector<mlir::Attribute> argDicts;

    for (const auto &k : meta.kernels) {
      if (!kname.empty() && (k.symbol == kname.str() || k.name == kname.str())) {
        argDicts.reserve(k.args.size());
        for (const auto &arg : k.args) {
          // 跳過 hidden 參數（由 runtime 自動管理）
          bool isHidden = false;
          for (const auto &prop : arg.getAllProperties()) {
            if (prop.first == "value_kind") {
              llvm::StringRef valueKind(prop.second);
              if (valueKind.starts_with("hidden")) {
                isHidden = true;
                break;
              }
            }
          }
          
          if (!isHidden) {
            argDicts.push_back(propsToDictAttr(builder, arg));
          }
        }

        // module->setAttr("amdisa.sgpr_count", builder.getI32IntegerAttr(k.sgprCount));
        // module->setAttr("amdisa.vgpr_count", builder.getI32IntegerAttr(k.vgprCount));
        // module->setAttr("amdisa.agpr_count", builder.getI32IntegerAttr(k.agprCount));
        break;
      }
    }

    if (!argDicts.empty())
      module->setAttr("amdisa.kernargs", builder.getArrayAttr(argDicts));
  }

  return module;
}


} // namespace mlir::amdisa
