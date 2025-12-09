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

using namespace mlir;
using namespace mlir::amdisa;

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
    // (1) Collect all raw assembly from amdisa.label + amdisa.inst
    //--------------------------------------------------------------------------
    std::string asmText;

    module.walk([&](Operation *op) {
      if (auto label = dyn_cast<amdisa::LabelOp>(op)) {
        // label.getName() = StringRef → OK
        asmText += label.getName().str();
        asmText += ":\n";

      } else if (auto inst = dyn_cast<amdisa::InstOp>(op)) {
        if (auto raw = inst.getRawText()) {
          // raw = optional<StringRef> → use *raw
          asmText += *raw;
          asmText += "\n";
        }
      }
    });


    if (asmText.empty()) {
      module.emitError() << "No AMDISA ops found for lowering\n";
      signalPassFailure();
      return;
    }

    //--------------------------------------------------------------------------
    // (2) Determine kernel name
    //--------------------------------------------------------------------------
    std::string kernel = kernelName; // From Pass option

    if (auto attr =
            module->getAttrOfType<StringAttr>("amdisa.kernel_name")) {
      kernel = attr.str();
    }

    if (kernel.empty())
      kernel = "amdisa_kernel";

    //--------------------------------------------------------------------------
    // (3) Apply target triple + code object version if not already present
    //--------------------------------------------------------------------------
    if (!module->getAttr("llvm.target_triple"))
      module->setAttr("llvm.target_triple",
                      builder.getStringAttr(targetTriple));

    if (!module->getAttr("amdgpu.code_object_version"))
      module->setAttr("amdgpu.code_object_version",
                      builder.getI32IntegerAttr(codeObjectVersion));

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

    auto funcType = builder.getFunctionType(/*inputs=*/TypeRange{},
                                            /*results=*/TypeRange{});

    auto gpuFunc = builder.create<gpu::GPUFuncOp>(
        loc, kernel, funcType);

    gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                 builder.getUnitAttr());

    Block *entry = &gpuFunc.getBody().front();
    builder.setInsertionPointToStart(entry);

    //--------------------------------------------------------------------------
    // (6) Insert llvm.inline_asm containing the full AMD ISA
    //--------------------------------------------------------------------------
    StringRef asmStrRef(asmText);
    StringRef constraintsRef = "";

    LLVM::AsmDialectAttr dialectAttr;      // default-constructed = null attr
    ArrayAttr operandAttrs;                // default-constructed = null attr

    TypeRange resultTypes;
    ValueRange operands;

    auto tailKind = mlir::LLVM::tailcallkind::TailCallKind::None;

    auto inlineAsm = LLVM::InlineAsmOp::create(
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
