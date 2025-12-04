//===- AMDISAParser.cpp - AMD ISA Parser & Dumper ------------------------===//
//
// This tool implements:
//     .s  -> AMDISA Dialect  -> .s
//
// It parses AMD GCN assembly into custom MLIR dialect ops (amdisa.inst),
// and can dump the MLIR or reconstruct a .s output.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/IR/AsmState.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/AMDISA/IR/AMDISAOps.h"
#include "mdr.h"


using namespace mlir::amdisa;
namespace cl = llvm::cl;

//===----------------------------------------------------------------------===//
// Command-line options
//===----------------------------------------------------------------------===//

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

enum InputType { Asm, MLIR };
static cl::opt<enum InputType> inputType(
    "x", cl::desc("Input type"),
    cl::values(
        clEnumValN(Asm, "s", "Load input as AMD .s assembly"),
        clEnumValN(MLIR, "mlir", "Load input as MLIR file")));

enum Action { None, DumpMLIR, DumpASM };
static cl::opt<enum Action> emitAction(
    "emit", cl::desc("Output type"),
    cl::values(
        clEnumValN(DumpMLIR, "mlir", "Dump the MLIR module"),
        clEnumValN(DumpASM,  "s",    "Dump reconstructed .s assembly")));

//===----------------------------------------------------------------------===//
// Parsing pipeline
//===----------------------------------------------------------------------===//

/// Parse .s assembly → return ModuleOp containing amdisa.inst ops.
static mlir::OwningOpRef<mlir::ModuleOp>
parseAsmToAMDISA(llvm::StringRef filename, mlir::MLIRContext &context) {

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(filename);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return nullptr;
  }

  auto buffer = fileOrErr.get()->getBuffer();
  AMDISAAsmParser parser(buffer);

  return parser.parseModule(context);
}

// /// Reconstruct .s from AMDISA Dialect
// static void dumpAsAssembly(mlir::ModuleOp module) {
//   for (Operation &op : module.getBody()->getOperations()) {
//     if (auto inst = dyn_cast<InstOp>(&op)) {
//       llvm::outs() << inst.getRawText() << "\n";
//       continue;
//     }
//     // (Optional) handle labels, directives, etc.
//   }
// }

//===----------------------------------------------------------------------===//
// MLIR loading
//===----------------------------------------------------------------------===//

static mlir::OwningOpRef<mlir::ModuleOp>
parseMLIRFile(llvm::StringRef filename, mlir::MLIRContext &context) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(filename);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return nullptr;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  return mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
}

//===----------------------------------------------------------------------===//
// Action drivers
//===----------------------------------------------------------------------===//

static int runDumpMLIR() {
  mlir::MLIRContext context;
  context.getOrLoadDialect<AMDISADialect>();

  mlir::OwningOpRef<mlir::ModuleOp> module;

  if (inputType == InputType::Asm)
    module = parseAsmToAMDISA(inputFilename, context);
  else
    module = parseMLIRFile(inputFilename, context);

  if (!module) {
    llvm::errs() << "Failed to load input.\n";
    return 1;
  }

  module->dump();
  return 0;
}

// static int runDumpASM() {
//   mlir::MLIRContext context;
//   context.getOrLoadDialect<AMDISADialect>();

//   OwningOpRef<ModuleOp> module;

//   if (inputType == InputType::Asm)
//     module = parseAsmToAMDISA(inputFilename, context);
//   else
//     module = parseMLIRFile(inputFilename, context);

//   if (!module) {
//     llvm::errs() << "Failed to load input.\n";
//     return 1;
//   }

//   dumpAsAssembly(*module);
//   return 0;
// }

//===----------------------------------------------------------------------===//
// Main
//===----------------------------------------------------------------------===//

int main(int argc, char **argv) {
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  cl::ParseCommandLineOptions(argc, argv, "AMDISA Parser\n");

  switch (emitAction) {
    case DumpMLIR: return runDumpMLIR();
    case DumpASM:  return 0; //runDumpASM(), not done yet;
    default:
      llvm::errs() << "No action specified. Use -emit=mlir or -emit=s\n";
      return 2;
  }
}
