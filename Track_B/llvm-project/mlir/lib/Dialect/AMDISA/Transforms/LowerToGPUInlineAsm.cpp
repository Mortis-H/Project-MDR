//===- LowerToGPUInlineAsm.cpp - AMDISA → GPU Inline ASM Pass -------------===//
//
// This file lowers amdisa.label / amdisa.inst into a gpu.module + gpu.func
// containing a single llvm.inline_asm operation.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/AMDISA/Passes.h"

#include "mlir/Dialect/AMDISA/IR/AMDISAOps.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/Casting.h"

static llvm::StringRef stripOuterQuotes(llvm::StringRef s) {
  s = s.trim();
  if (s.size() >= 2 && s.front() == '"' && s.back() == '"')
    return s.drop_front().drop_back();
  return s;
}

static mlir::Type
typeFromKernargDict(mlir::OpBuilder &b, mlir::DictionaryAttr d) {
  int size = 0;
  if (auto s = d.getAs<mlir::IntegerAttr>("size"))
    size = s.getInt();
  
  // 獲取 value_kind 用於驗證（可選）
  auto vkAttr = d.getAs<mlir::StringAttr>("value_kind");
  llvm::StringRef vk = vkAttr ? vkAttr.getValue() : "";

  switch (size) {
  case 8: {
    // 在 debug 模式下驗證 value_kind
    #ifndef NDEBUG
    if (!vk.empty() && !vk.contains("buffer") && !vk.contains("pointer")) {
      llvm::errs() << "Warning: size=8 but value_kind=" << vk 
                   << " (expected pointer type)\n";
    }
    #endif
    return mlir::LLVM::LLVMPointerType::get(b.getContext());
  }
    
  case 4:
    return b.getIndexType();
    
  case 2:
    return b.getI16Type();
    
  case 1:
    return b.getI8Type();
    
  default:
    if (size > 8 && size <= 4096) {
      // 大型結構體參數（如 attn_bwd_combined_globals）
      // 使用字節數組類型：!llvm.array<N x i8>
      llvm::errs() << "Info: large kernel argument (size=" << size 
                   << " bytes), using !llvm.array<" << size << " x i8>\n";
      mlir::Type i8Type = b.getI8Type();
      return mlir::LLVM::LLVMArrayType::get(i8Type, size);
      
    } else if (size > 4096) {
      llvm::errs() << "Error: kernel argument too large (size=" << size 
                   << " bytes), max 4096 expected. Using index as fallback.\n";
      return b.getIndexType();
      
    } else {
      llvm::errs() << "Error: invalid kernel argument size=" << size 
                   << ". Using index as fallback.\n";
      return b.getIndexType();
    }
  }
}

namespace mlir {
namespace amdisa {

#define GEN_PASS_DEF_LOWERAMDISATOGPUINLINEASM
#include "mlir/Dialect/AMDISA/Passes.h.inc"

} // namespace amdisa
} // namespace mlir

using namespace mlir;
using namespace mlir::amdisa;

namespace {

/// Lower all AMDISA instructions into a gpu.func + inline asm.
class LowerAMDISAToGPUInlineAsmPass
    : public amdisa::impl::LowerAMDISAToGPUInlineAsmBase<
          LowerAMDISAToGPUInlineAsmPass> {

public:
  using Base = amdisa::impl::LowerAMDISAToGPUInlineAsmBase<
      LowerAMDISAToGPUInlineAsmPass>;
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();
    OpBuilder builder(ctx);

    //--------------------------------------------------------------------------
    // (1) Collect all AMDISA operations (preserving order)
    //--------------------------------------------------------------------------
    SmallVector<Operation *, 32> amdisaOps;
    
    module.walk([&](Operation *op) {
      if (isa<amdisa::LabelOp>(op) || isa<amdisa::InstOp>(op)) {
        amdisaOps.push_back(op);
      }
    });

    if (amdisaOps.empty()) {
      module.emitError() << "No AMDISA ops found for lowering\n";
      signalPassFailure();
      return;
    }

    // --------------------------------------------------------------------------
    // (2) Determine kernel name (prefer module attr, fallback to pass option)
    // --------------------------------------------------------------------------
    std::string kernel;

    // Prefer module attribute
    if (auto a = module->getAttrOfType<StringAttr>("amdisa.kernel_name")) {
      kernel = a.getValue().str(); // NOTE: getValue() not .str() on attr itself
    }

    // Fallback to pass option
    if (kernel.empty())
      kernel = kernelName;

    // Final default
    if (kernel.empty())
      kernel = "amdisa_kernel";

    // --------------------------------------------------------------------------
    // (3) Apply / normalize target triple + code object version
    //     Prefer existing module attrs, else use pass options.
    //     Also strip outer quotes on llvm.target_triple to avoid \22.
    // --------------------------------------------------------------------------

    // llvm.target_triple
    if (auto a = module->getAttrOfType<StringAttr>("llvm.target_triple")) {
      llvm::StringRef v = stripOuterQuotes(a.getValue());
      // If normalization changed it, rewrite attribute
      if (v != a.getValue())
        module->setAttr("llvm.target_triple", builder.getStringAttr(v));
    } else if (!targetTriple.empty()) {
      llvm::StringRef v = stripOuterQuotes(targetTriple);
      module->setAttr("llvm.target_triple", builder.getStringAttr(v));
    }

    // amdgpu.code_object_version
    if (!module->getAttr("amdgpu.code_object_version")) {
      // Only set if pass option provides a meaningful value
      // (codeObjectVersion is typically an int option; if you used std::string,
      // parse it before calling getI32IntegerAttr).
      module->setAttr("amdgpu.code_object_version",
                      builder.getI32IntegerAttr(codeObjectVersion));
    }

    Location loc = module.getLoc();


    //--------------------------------------------------------------------------
    // (4) Create gpu.module @amdisa_kernels
    //--------------------------------------------------------------------------
    builder.setInsertionPointToStart(module.getBody());
    auto gpuModule = builder.create<gpu::GPUModuleOp>(
        loc, builder.getStringAttr("amdisa_kernels"));

    //--------------------------------------------------------------------------
    // (5) Create gpu.func @<kernel> () kernel
    //--------------------------------------------------------------------------
    builder.setInsertionPointToStart(gpuModule.getBody());

    // auto funcType = builder.getFunctionType(/*inputs=*/TypeRange{},
    //                                         /*results=*/TypeRange{});

    // auto gpuFunc = builder.create<gpu::GPUFuncOp>(
    //     loc, kernel, funcType);

    SmallVector<Type> inputTypes;

    if (auto a = module->getAttrOfType<ArrayAttr>("amdisa.kernargs")) {
      inputTypes.reserve(a.size());
      for (mlir::Attribute elt : a) {
        if (auto d = llvm::dyn_cast<mlir::DictionaryAttr>(elt)) {
          inputTypes.push_back(typeFromKernargDict(builder, d));
        }
      }
    }

    auto funcType = builder.getFunctionType(/*inputs=*/inputTypes,
                                            /*results=*/TypeRange{});

    auto gpuFunc = builder.create<gpu::GPUFuncOp>(loc, kernel, funcType);
    gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                    builder.getUnitAttr());


    // gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
    //              builder.getUnitAttr());

    Block *entry = &gpuFunc.getBody().front();
    builder.setInsertionPointToStart(entry);

    //--------------------------------------------------------------------------
    // (6) Insert separate llvm.inline_asm for each AMDISA instruction
    //--------------------------------------------------------------------------
    StringRef constraintsRef = "";
    LLVM::AsmDialectAttr dialectAttr;      // default-constructed = null attr
    ArrayAttr operandAttrs;                // default-constructed = null attr
    TypeRange resultTypes;
    ValueRange operands;
    auto tailKind = mlir::LLVM::tailcallkind::TailCallKind::None;

    for (Operation *op : amdisaOps) {
      std::string asmStr;
      
      if (auto label = dyn_cast<amdisa::LabelOp>(op)) {
        // 將 label 作為獨立的 inline asm（label:）
        asmStr = label.getName().str() + ":";
        
      } else if (auto inst = dyn_cast<amdisa::InstOp>(op)) {
        // 每個指令作為獨立的 inline asm
        if (auto raw = inst.getRawText()) {
          asmStr = raw->str();
        } else {
          continue;  // 跳過沒有 raw text 的指令
        }
      }
      
      if (!asmStr.empty()) {
        StringRef asmStrRef(asmStr);
        LLVM::InlineAsmOp::create(
            builder,
            loc,
            /*resultTypes=*/resultTypes,
            /*operands=*/operands,
            /*asm_string=*/asmStrRef,
            /*constraints=*/constraintsRef,
            /*has_side_effects=*/true,
            /*is_align_stack=*/false,
            /*tail_call_kind=*/tailKind,
            /*asm_dialect=*/dialectAttr,
            /*operand_attrs=*/operandAttrs);
      }
    }

    // Add terminator
    builder.create<gpu::ReturnOp>(loc);

    //--------------------------------------------------------------------------
    // (7) Remove original AMDISA ops
    //--------------------------------------------------------------------------
    SmallVector<Operation *, 32> eraseList;
    module.walk([&](amdisa::LabelOp op) { eraseList.push_back(op); });
    module.walk([&](amdisa::InstOp op) { eraseList.push_back(op); });

    for (Operation *op : eraseList)
      op->erase();
  }
};

} // namespace

//----------------------------------------------------------------------------//
// Pass Creation
//----------------------------------------------------------------------------//

std::unique_ptr<Pass> mlir::amdisa::createLowerAMDISAToGPUInlineAsmPass() {
  return std::make_unique<LowerAMDISAToGPUInlineAsmPass>();
}
