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
      auto nameAttr = builder.getStringAttr(line->labelName);
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

    // 找到對應的 kernel（依你的資料，通常用 symbol 或 name 對到 amdisa.kernel_name）
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
