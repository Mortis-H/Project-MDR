//===- AMDISA.h - AMDISA dialect ------------------------*- C++ -*-===//
//
//  C++ declarations for the AMDISA dialect.
//===----------------------------------------------------------------------===//

// #ifndef MLIR_DIALECT_AMDISA_IR_AMDISA_H
// #define MLIR_DIALECT_AMDISA_IR_AMDISA_H

// #include "mlir/IR/BuiltinOps.h"
// #include "mlir/IR/Dialect.h"
// #include "mlir/IR/OpImplementation.h"

// #include "mlir/Dialect/Affine/IR/AffineOpsDialect.h.inc"

// #define GET_OP_CLASSES
// #include "mlir/Dialect/AMDISA/IR/AMDISAOps.h.inc"

// #endif // MLIR_DIALECT_AMDISA_IR_AMDISA_H

#ifndef MLIR_DIALECT_AMDISA_IR_AMDISAOPS_H
#define MLIR_DIALECT_AMDISA_IR_AMDISAOPS_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/Dialect/AMDISA/IR/AMDISAOpsDialect.h.inc"

// Include the auto-generated operation declarations.
#define GET_OP_CLASSES
#include "mlir/Dialect/AMDISA/IR/AMDISAOps.h.inc"

#endif // MLIR_DIALECT_AMDISA_IR_AMDISAOPS_H
