//===- AMDISA.cpp - AMDISA dialect -------------------------------*- C++ -*-===//
//
//  C++ definitions for the AMDISA dialect.
//===----------------------------------------------------------------------===//

#include "Dialect/AMDISA/IR/AMDISAOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Builders.h"

using namespace mlir;
using namespace mlir::amdisa;

// Include dialect initialization logic (auto-generated)
#include "Dialect/AMDISA/IR/AMDISADialect.cpp.inc"

void AMDISADialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/AMDISA/IR/AMDISAOps.cpp.inc"
      >();
}

// Include op class method definitions (auto-generated)
#define GET_OP_CLASSES
#include "Dialect/AMDISA/IR/AMDISAOps.cpp.inc"
