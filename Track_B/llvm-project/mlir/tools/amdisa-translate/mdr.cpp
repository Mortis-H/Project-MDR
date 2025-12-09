//===- AMDISAParser.cpp - AMD ISA Parser & Dumper ------------------------===//
//
// This tool implements:
//     .s  -> AMDISA Dialect  ->  (MLIR or GPU inline asm)
//
// It parses AMD GCN assembly into custom MLIR dialect ops (amdisa.inst),
// and can:
//   * dump MLIR
//   * dump reconstructed .s
//   * run LowerToGPUInlineAsm pass and emit gpu.func + llvm.inline_asm
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/AMDISA/IR/AMDISAOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "mlir/Dialect/AMDISA/Passes.h"
#include "mdr.h"

using namespace mlir;

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
        clEnumValN(Asm,  "s",    "Load input as AMD .s assembly"),
        clEnumValN(MLIR, "mlir", "Load input as MLIR file")));

enum Action { None, DumpMLIR, DumpASM, DumpGPUInlineASM };
static cl::opt<enum Action> emitAction(
    "emit", cl::desc("Output type"),
    cl::values(
        clEnumValN(DumpMLIR,        "mlir",         "Dump the MLIR module"),
        clEnumValN(DumpASM,         "s",            "Dump reconstructed .s"),
        clEnumValN(DumpGPUInlineASM,"gpuinlineasm", "Lower to gpu.func + llvm.inline_asm")));

//===----------------------------------------------------------------------===//
// Parsing pipeline
//===----------------------------------------------------------------------===//

static OwningOpRef<ModuleOp>
parseAsmToAMDISA(StringRef filename, MLIRContext &context) {

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

static OwningOpRef<ModuleOp>
parseMLIRFile(StringRef filename, MLIRContext &context) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(filename);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return nullptr;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  return parseSourceFile<ModuleOp>(sourceMgr, &context);
}

//===----------------------------------------------------------------------===//
// Action: dump MLIR
//===----------------------------------------------------------------------===//

static int runDumpMLIR() {
  MLIRContext context;
  context.getOrLoadDialect<AMDISADialect>();

  OwningOpRef<ModuleOp> module =
      (inputType == Asm)
          ? parseAsmToAMDISA(inputFilename, context)
          : parseMLIRFile(inputFilename, context);

  if (!module) {
    llvm::errs() << "Failed to load input.\n";
    return 1;
  }

  module->dump();
  return 0;
}

//===----------------------------------------------------------------------===//
// Action: Lower AMDISA → GPU Inline ASM
//===----------------------------------------------------------------------===//

static int runDumpGPUInlineASM() {
  MLIRContext context;

  // Load dialects required by the lowering pass
  context.getOrLoadDialect<AMDISADialect>();
  context.getOrLoadDialect<gpu::GPUDialect>();
  context.getOrLoadDialect<LLVM::LLVMDialect>();

  OwningOpRef<ModuleOp> module =
      (inputType == Asm)
          ? parseAsmToAMDISA(inputFilename, context)
          : parseMLIRFile(inputFilename, context);

  if (!module) {
    llvm::errs() << "Failed to load input.\n";
    return 1;
  }

  PassManager pm(&context);

  // Add your lowering pass
  pm.addPass(createLowerAMDISAToGPUInlineAsmPass());

  if (failed(pm.run(*module))) {
    llvm::errs() << "Lowering to GPU inline asm failed.\n";
    return 1;
  }

  module->dump();
  return 0;
}

//===----------------------------------------------------------------------===//
// Main
//===----------------------------------------------------------------------===//

int main(int argc, char **argv) {
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  cl::ParseCommandLineOptions(argc, argv, "AMDISA Parser\n");

  switch (emitAction) {
    case DumpMLIR:        return runDumpMLIR();
    case DumpGPUInlineASM:return runDumpGPUInlineASM();
    case DumpASM:         return 0;
    default:
      llvm::errs() << "No action specified. Use -emit=mlir, -emit=s, or -emit=gpuinlineasm\n";
      return 2;
  }
}
