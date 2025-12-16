//===- amdisa-translate.h - AMD ISA Parser Interface ----------------------===//
//
// Header file for AMD ISA Assembly Parser
//
// This header declares the AMDISAAsmParser class which reads AMD GCN ISA 
// (.s) assembly and creates amdisa.inst ops inside an MLIR Module.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_AMDISA_TRANSLATE_H
#define MLIR_TOOLS_AMDISA_TRANSLATE_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
class MLIRContext;
} // namespace mlir

namespace mlir::amdisa {

//===----------------------------------------------------------------------===//
// AMDISAAsmParser - Parses AMD GCN assembly into AMDISA dialect
//===----------------------------------------------------------------------===//

/// Parser for AMD GCN assembly (.s files)
/// 
/// This parser reads AMD ISA assembly instructions and converts them into
/// MLIR operations in the AMDISA dialect. It handles:
/// - Instructions (converted to amdisa.inst ops)
/// - Labels (converted to amdisa.label ops)
/// - Kernel metadata (stored as module attributes)
/// - AMD GPU metadata (parsed from YAML sections)
class AMDISAAsmParser {
public:
  /// Construct a parser for the given assembly file
  explicit AMDISAAsmParser(llvm::StringRef filename);

  /// Parse the assembly file and return an MLIR ModuleOp
  /// containing AMDISA dialect operations
  mlir::OwningOpRef<mlir::ModuleOp>
  parseModule(mlir::MLIRContext &context);

private:
  llvm::StringRef filename;
};

} // namespace mlir::amdisa

#endif // MLIR_TOOLS_AMDISA_TRANSLATE_H
