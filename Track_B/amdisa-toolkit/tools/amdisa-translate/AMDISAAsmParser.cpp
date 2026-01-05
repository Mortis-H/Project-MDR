//===- AMDISAAsmParser.cpp - AMD ISA Assembly Parser Implementation ------===//
//
// Implementation of AMD ISA Assembly Parser
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
//===----------------------------------------------------------------------===//

#include "AMDISAAsmParser.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

// AMDISA dialect operations
#include "Dialect/AMDISA/IR/AMDISAOps.h"

// Assembly parsing utilities
#include "parse_utils.h"
#include "AMDGCNAssembly.h"
#include "AMDGPUMetadata.h"
#include "ParsedProgram.h"

#include <iostream>

//===----------------------------------------------------------------------===//
// Forward Declarations
//===----------------------------------------------------------------------===//

namespace mlir::amdisa {

void processSingleKernelAsFunc(mlir::OpBuilder &builder, 
                                const AMDGCNAssembly &assembly,
                                const KernelRegion &kernel);

void processKernelRegion(mlir::ModuleOp &module, 
                          mlir::OpBuilder &builder,
                          const AMDGCNAssembly &assembly, 
                          const KernelRegion &kernel);

void processLegacySingleKernel(mlir::ModuleOp &module, 
                                mlir::OpBuilder &builder,
                                const AMDGCNAssembly &assembly);

} // namespace mlir::amdisa

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Convert argument properties to MLIR DictionaryAttr
/// This is used to encode kernel argument metadata from AMD GPU metadata
template <typename ArgT>
static mlir::DictionaryAttr
propsToDictAttr(mlir::OpBuilder &b, const ArgT &arg) {
  llvm::SmallVector<mlir::NamedAttribute> kvs;

  for (auto &p : arg.getAllProperties()) {
    llvm::StringRef k = p.first;
    llvm::StringRef v = p.second;

    // Convert numeric properties to integers
    if (k == "size" || k == "offset" || k == "align" ||
        k == "pointee_align") {
      int iv = 0;
      if (!v.getAsInteger(10, iv))
        kvs.emplace_back(b.getStringAttr(k),
                          b.getI32IntegerAttr(iv));
    } else {
      // Keep string properties as-is
      kvs.emplace_back(b.getStringAttr(k),
                        b.getStringAttr(v));
    }
  }

  return mlir::DictionaryAttr::get(b.getContext(), kvs);
}

namespace mlir::amdisa {

//===----------------------------------------------------------------------===//
// AMDISAAsmParser Implementation
//===----------------------------------------------------------------------===//

AMDISAAsmParser::AMDISAAsmParser(llvm::StringRef filename)
    : filename(filename) {}

mlir::OwningOpRef<mlir::ModuleOp>
AMDISAAsmParser::parseModule(mlir::MLIRContext &context) {

  mlir::OpBuilder builder(&context);
  auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToEnd(module.getBody());

  // Parse the AMD GCN assembly file
  AMDGCNAssembly assembly = parseAMDGCNAssembly(filename.str());

  // Set module-level attributes (target triple, code object version, etc.)
  for (size_t lineNum = 1; lineNum <= assembly.getLineCount(); ++lineNum) {
    const LineInfo *line = assembly.getLine(lineNum);
    if (!line) continue;

    if (line->kind == LineKind::AmdgcnTarget) {
      // .amdgcn_target "amdgcn-amd-amdhsa--gfx950"
      if (!module->hasAttr("llvm.target_triple")) {
        auto tripleAttr = builder.getStringAttr(line->amdgcnTarget);
        module->setAttr("llvm.target_triple", tripleAttr);
      }
    }

    if (line->kind == LineKind::AmdhsaCodeObjectVersion) {
      if (!module->hasAttr("amdgpu.code_object_version")) {
        int version = 0;
        if (!line->amdhsaCodeObjectVersion.empty()) {
          version = std::stoi(line->amdhsaCodeObjectVersion);
        }
        auto verAttr = builder.getI32IntegerAttr(version);
        module->setAttr("amdgpu.code_object_version", verAttr);
      }
    }
  }

  // Process each kernel separately
  const std::vector<KernelRegion> &kernels = assembly.getAllKernels();
  
  // 如果沒有找到 kernel 區域，使用舊的單 kernel 模式（向後兼容）
  if (kernels.empty()) {
    // 舊版本單 kernel 處理邏輯
    processLegacySingleKernel(module, builder, assembly);
  } else {
    // 新版本多 kernel 處理邏輯
    for (const auto &kernel : kernels) {
      processKernelRegion(module, builder, assembly, kernel);
    }
  }

  return module;
}

/// Process a single kernel region and create an amdisa.func op
void processSingleKernelAsFunc(mlir::OpBuilder &builder, 
                                const AMDGCNAssembly &assembly,
                                const KernelRegion &kernel) {
  auto loc = builder.getUnknownLoc();
  
  // Create amdisa.func op
  auto funcOp = builder.create<FuncOp>(loc, builder.getStringAttr(kernel.name));
  
  // Set insertion point to function body
  mlir::Block *funcBody = new mlir::Block();
  funcOp.getBody().push_back(funcBody);
  builder.setInsertionPointToEnd(funcBody);
  
  // Process labels and instructions in this kernel
  for (size_t lineNum = kernel.startLine; lineNum <= kernel.endLine; ++lineNum) {
    const LineInfo *line = assembly.getLine(lineNum);
    if (!line) continue;
    
    switch (line->kind) {
      case LineKind::Label: {
        std::string labelName = line->labelName;
        
        // Extract pure label name (remove comment part)
        llvm::StringRef labelRef(labelName);
        size_t colonPos = labelRef.find(':');
        if (colonPos != llvm::StringRef::npos) {
          labelRef = labelRef.substr(0, colonPos).trim();
        }
        std::string pureLabelName = labelRef.str();
        
        // Skip .Lfunc_end labels (function end markers)
        if (pureLabelName.rfind(".Lfunc_end", 0) == 0) {
          break;
        }
        
        // Skip labels with same name as kernel (function entry)
        if (pureLabelName == kernel.name) {
          break;
        }
        
        // Keep all other labels (including .LBB0_X basic block labels)
        auto nameAttr = builder.getStringAttr(labelName);
        builder.create<LabelOp>(loc, nameAttr);
        break;
      }
      
      case LineKind::Instruction: {
        if (!line->instruction) {
          // Skip instructions without data
          break;
        }
        
        const ParsedInstruction &inst = *line->instruction;
        
        auto mnemonicAttr = builder.getStringAttr(inst.opcode);
        
        // Convert operands to array of string attributes
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
      
      default:
        // Skip other line types inside kernel body
        break;
    }
  }
  
  // Add terminator for the function body
  builder.create<ReturnOp>(loc);
  
  // Add kernel metadata as attributes
  if (kernel.hasMetadata() && kernel.metadata != nullptr) {
    const KernelInfo *meta = kernel.metadata;
    
    // Build kernel arguments metadata
    llvm::SmallVector<mlir::Attribute> argDicts;
    argDicts.reserve(meta->args.size());
    for (const auto &arg : meta->args) {
      argDicts.push_back(propsToDictAttr(builder, arg));
    }
    
    // Set function attributes
    funcOp->setAttr("amdisa.sgpr_count", builder.getI32IntegerAttr(meta->sgprCount));
    funcOp->setAttr("amdisa.vgpr_count", builder.getI32IntegerAttr(meta->vgprCount));
    funcOp->setAttr("amdisa.agpr_count", builder.getI32IntegerAttr(meta->agprCount));
    funcOp->setAttr("amdisa.kernarg_segment_size", builder.getI32IntegerAttr(meta->kernargSegmentSize));
    
    if (!argDicts.empty()) {
      funcOp->setAttr("amdisa.kernargs", builder.getArrayAttr(argDicts));
    }
  }
}

/// Process kernel region and add to module
void processKernelRegion(mlir::ModuleOp &module, 
                          mlir::OpBuilder &builder,
                          const AMDGCNAssembly &assembly, 
                          const KernelRegion &kernel) {
  // Set insertion point to module body
  builder.setInsertionPointToEnd(module.getBody());
  
  // Create and process the kernel function
  processSingleKernelAsFunc(builder, assembly, kernel);
}

/// Legacy single-kernel processing (backward compatibility)
void processLegacySingleKernel(mlir::ModuleOp &module, 
                                mlir::OpBuilder &builder,
                                const AMDGCNAssembly &assembly) {
  // Process each line of the assembly (old behavior)
  for (size_t lineNum = 1; lineNum <= assembly.getLineCount(); ++lineNum) {
    const LineInfo *line = assembly.getLine(lineNum);
    if (!line) continue;

    auto loc = builder.getUnknownLoc();

    switch (line->kind) {

      case LineKind::Label: {
        std::string labelName = line->labelName;
        
        // Extract pure label name (remove comment part)
        llvm::StringRef labelRef(labelName);
        size_t colonPos = labelRef.find(':');
        if (colonPos != llvm::StringRef::npos) {
          labelRef = labelRef.substr(0, colonPos).trim();
        }
        std::string pureLabelName = labelRef.str();
        
        // Skip .Lfunc_end labels
        if (pureLabelName.rfind(".Lfunc_end", 0) == 0) {
          break;
        }
        
        // Skip labels with same name as kernel
        llvm::StringRef kname;
        if (auto a = module->getAttrOfType<mlir::StringAttr>("amdisa.kernel_name"))
          kname = a.getValue();
        if (!kname.empty() && pureLabelName == kname.str()) {
          break;
        }
        
        // Keep all other labels
        auto nameAttr = builder.getStringAttr(labelName);
        builder.create<LabelOp>(loc, nameAttr);
        break;
      }

    case LineKind::Instruction: {
      if (!line->instruction) {
        // Skip instructions without data
        break;
      }
      
      const ParsedInstruction &inst = *line->instruction;

      auto mnemonicAttr = builder.getStringAttr(inst.opcode);

      // Convert operands to array of string attributes
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

    case LineKind::Directive:
    case LineKind::Comment:
    case LineKind::Metadata:
    case LineKind::Unknown:
    default:
      // Skip
      break;
    }
  }

  // Process AMD GPU metadata (kernel arguments, register usage, etc.)
  if (assembly.hasMetadata()) {
    const AMDGPUMetadata &meta = assembly.getMetadata();

    llvm::StringRef kname;
    if (auto a = module->getAttrOfType<mlir::StringAttr>("amdisa.kernel_name"))
      kname = a.getValue();

    llvm::SmallVector<mlir::Attribute> argDicts;

    for (const auto &k : meta.kernels) {
      if (!kname.empty() && (k.symbol == kname.str() || k.name == kname.str())) {
        argDicts.reserve(k.args.size());
        for (const auto &arg : k.args) {
          argDicts.push_back(propsToDictAttr(builder, arg));
        }

        // Store register counts as module attributes
        module->setAttr("amdisa.sgpr_count", builder.getI32IntegerAttr(k.sgprCount));
        module->setAttr("amdisa.vgpr_count", builder.getI32IntegerAttr(k.vgprCount));
        module->setAttr("amdisa.agpr_count", builder.getI32IntegerAttr(k.agprCount));
        module->setAttr("amdisa.kernarg_segment_size", builder.getI32IntegerAttr(k.kernargSegmentSize));
        
        break;
      }
    }

    // Store kernel arguments metadata
    if (!argDicts.empty())
      module->setAttr("amdisa.kernargs", builder.getArrayAttr(argDicts));
  }
}

} // namespace mlir::amdisa

