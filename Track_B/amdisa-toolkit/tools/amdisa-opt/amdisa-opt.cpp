//===- amdisa-opt.cpp - AMDISA Optimizer Driver --------------------------===//
//
// This file implements the 'amdisa-opt' tool, which is similar to mlir-opt
// but includes the AMDISA dialect and related passes.
//
// It can be used to test and debug AMDISA transformations using the standard
// MLIR optimization pipeline infrastructure.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

// AMDISA Dialect and Passes
#include "Dialect/AMDISA/IR/AMDISAOps.h"
#include "Dialect/AMDISA/Passes.h"

int main(int argc, char **argv) {
  // Initialize LLVM
  llvm::InitLLVM y(argc, argv);

  // Register all MLIR passes
  mlir::registerAllPasses();

  // Register AMDISA passes
  mlir::amdisa::registerAMDISAPasses();

  // Create a dialect registry
  mlir::DialectRegistry registry;

  // Register all standard MLIR dialects
  mlir::registerAllDialects(registry);

  // Register AMDISA dialect
  registry.insert<mlir::amdisa::AMDISADialect>();

  // Run the main mlir-opt logic
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "AMDISA optimizer driver\n", registry));
}

