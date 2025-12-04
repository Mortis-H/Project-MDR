//===- AMDISAAsmParser.cpp - AMD ISA Assembly Parser ---------------------===//
//
// This file implements a minimal parser that reads AMD GCN ISA assembly
// (.s files) and lowers each line into an AMDISAInstOp inside an MLIR module.
//
// It expects "clean" assembly:
//     v_mbcnt_lo_u32_b32 v0, -1, 0
//     v_mbcnt_hi_u32_b32 v10, -1, v0
//     v_mov_b64_e32 v[4:5], 0
//     v_readfirstlane_b32 s2, v10
//
// Output:
//   amdisa.inst {
//      mnemonic = "...",
//      operands = [...],
//      raw_text = "..."
//   }
//
//===----------------------------------------------------------------------===//

#include "AMDISAAsmParser.h"
#include "mlir/Dialect/AMDISA/IR/AMDISAOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace amdisa;

//===----------------------------------------------------------------------===//
// Constructor
//===----------------------------------------------------------------------===//

AMDISAAsmParser::AMDISAAsmParser(llvm::StringRef buffer)
    : buffer(buffer) {}

//===----------------------------------------------------------------------===//
// Utility: remove whitespace, CRLF, comments (future extension)
//===----------------------------------------------------------------------===//

llvm::StringRef AMDISAAsmParser::cleanLine(llvm::StringRef line) const {
  // Trim spaces and tabs
  line = line.trim();

  // Skip empty lines
  if (line.empty())
    return line;

  // TODO (future): strip comments (# or ;)
  return line;
}

//===----------------------------------------------------------------------===//
// Parse a single instruction line
//   "mnemonic op1, op2, op3"
//===----------------------------------------------------------------------===//

bool AMDISAAsmParser::parseLine(llvm::StringRef line,
                                std::string &mnemonic,
                                llvm::SmallVectorImpl<std::string> &operands,
                                std::string &rawText) const {

  rawText = line.str(); // keep exactly the original text

  // Split first token = mnemonic
  auto firstSpace = line.find(' ');
  if (firstSpace == llvm::StringRef::npos) {
    // Line with no operands => treat whole thing as mnemonic
    mnemonic = line.str();
    return true;
  }

  mnemonic = line.take_front(firstSpace).str();
  llvm::StringRef operandStr = line.drop_front(firstSpace).trim();

  // Split operands by ',' and trim each one
  operands.clear();

  while (!operandStr.empty()) {
    auto comma = operandStr.find(',');
    llvm::StringRef token;

    if (comma == llvm::StringRef::npos) {
      token = operandStr.trim();
      operandStr = "";
    } else {
      token = operandStr.take_front(comma).trim();
      operandStr = operandStr.drop_front(comma + 1).trim();
    }

    if (!token.empty())
      operands.push_back(token.str());
  }

  return true;
}

//===----------------------------------------------------------------------===//
// Parse whole buffer into ModuleOp containing AMDISAInstOps
//===----------------------------------------------------------------------===//

mlir::OwningOpRef<mlir::ModuleOp>
AMDISAAsmParser::parseModule(mlir::MLIRContext &context) {

  OpBuilder builder(&context);
  auto module = ModuleOp::create(builder.getUnknownLoc());

  llvm::StringRef buf = buffer;
  while (!buf.empty()) {
    llvm::StringRef line;

    std::tie(line, buf) = buf.split('\n');
    line = cleanLine(line);

    if (line.empty())
      continue; // skip blank lines

    std::string mnemonic, raw;
    llvm::SmallVector<std::string, 8> ops;

    if (!parseLine(line, mnemonic, ops, raw)) {
      // Skip invalid lines
      llvm::errs() << "Warning: failed to parse line: " << line << "\n";
      continue;
    }

    // Construct MLIR attributes
    auto loc = builder.getUnknownLoc();
    auto mnemonicAttr = builder.getStringAttr(mnemonic);

    // Convert vector<string> → ArrayAttr<StringAttr>
    llvm::SmallVector<Attribute, 8> operandAttrs;
    for (auto &s : ops)
      operandAttrs.push_back(builder.getStringAttr(s));

    auto operandsAttr = builder.getArrayAttr(operandAttrs);
    auto rawTextAttr = builder.getStringAttr(raw);

    // Create the AMDISAInstOp
    builder.setInsertionPointToEnd(module.getBody());

    builder.create<AMDISAInstOp>(loc,
                                 mnemonicAttr,
                                 operandsAttr,
                                 rawTextAttr);
  }

  return module;
}
