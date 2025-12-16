//===- AMDISAToAssembly.h - Reconstruct .s from GPU MLIR -------*- C++ -*-===//
//
// Header for assembly reconstruction functionality.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_AMDISATRANSLATE_AMDISATOASSEMBLY_H
#define MLIR_TOOLS_AMDISATRANSLATE_AMDISATOASSEMBLY_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace amdisa {

/// Reconstruct a complete AMD GCN assembly file from GPU MLIR module.
/// The module is expected to have been lowered to GPU dialect with inline_asm.
int reconstructAssemblyFromGPU(ModuleOp module, llvm::raw_ostream &os);

} // namespace amdisa
} // namespace mlir

#endif // MLIR_TOOLS_AMDISATRANSLATE_AMDISATOASSEMBLY_H

