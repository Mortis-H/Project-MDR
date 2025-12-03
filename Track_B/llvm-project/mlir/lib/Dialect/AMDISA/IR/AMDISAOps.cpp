//===- AMDISA.cpp - AMDISA dialect -------------------------------*- C++ -*-===//
//
//  C++ definitions for the AMDISA dialect.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/AMDISA/IR/AMDISAOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Builders.h"

using namespace mlir;
using namespace mlir::amdisa;

// Include dialect initialization logic (auto-generated)
#include "mlir/Dialect/AMDISA/IR/AMDISAOpsDialect.cpp.inc"

void AMDISADialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/AMDISA/IR/AMDISAOps.cpp.inc"
      >();
}

// Include op class method definitions (auto-generated)
#define GET_OP_CLASSES
#include "mlir/Dialect/AMDISA/IR/AMDISAOps.cpp.inc"
