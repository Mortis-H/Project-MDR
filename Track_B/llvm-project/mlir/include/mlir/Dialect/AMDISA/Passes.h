//===- AMDISAPasses.h - Pass Entrypoints ------------------------*- C++ -*-===//
//
// This file defines the entry points for AMDISA dialect passes.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_AMDISA_PASSES_H
#define MLIR_DIALECT_AMDISA_PASSES_H

#include "mlir/Pass/Pass.h"
#include <string>

namespace mlir {
namespace amdisa {

//----------------------------------------------------------------------------//
// Pass Declarations (from TableGen)
//----------------------------------------------------------------------------//

#define GEN_PASS_DECL
#include "mlir/Dialect/AMDISA/Passes.h.inc"

//----------------------------------------------------------------------------//
// Factory Functions
//----------------------------------------------------------------------------//

/// Lower AMDISA inst + label ops into:
///   gpu.module + gpu.func + llvm.inline_asm
std::unique_ptr<Pass> createLowerAMDISAToGPUInlineAsmPass();

//----------------------------------------------------------------------------//
// Pass Registration
//----------------------------------------------------------------------------//

#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/AMDISA/Passes.h.inc"

} // namespace amdisa
} // namespace mlir

#endif // MLIR_DIALECT_AMDISA_PASSES_H
