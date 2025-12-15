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
        
        // 1. 跳過 .Lfunc_end 開頭的 label（函數結束標記，會由外層生成）
        if (pureLabelName.rfind(".Lfunc_end", 0) == 0) {
          break;
        }
        
        // 2. 跳過與 kernel 同名的 label（函數入口，會由 gpu.func 生成）
        llvm::StringRef kname;
        if (auto a = module->getAttrOfType<mlir::StringAttr>("amdisa.kernel_name"))
          kname = a.getValue();
        if (!kname.empty() && pureLabelName == kname.str()) {
          break;
        }
        
        // 3. 保留其他所有 label（包括 .LBB0_X 基本塊 label）
        auto nameAttr = builder.getStringAttr(labelName);
        builder.create<LabelOp>(loc, nameAttr);
        break;
      }

    case LineKind::Instruction: {
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
