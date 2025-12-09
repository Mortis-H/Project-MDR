//===- AMDISAAsmParser.cpp - AMD ISA Assembly Parser ---------------------===//
//
// Combined header + implementation into one file
//
// This parser reads AMD GCN ISA (.s) assembly lines and creates
// amdisa.inst ops inside an MLIR Module.
//
// Expected input:
//   v_add_f32 v1, v2, v3
//   s_waitcnt vmcnt(0)
//
// Output ops:
//   amdisa.inst { mnemonic = "...", operands = [...], raw_text = "..." }
//
// This merged version allows debugging compilation issues caused
// by header/include separation.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

// If your AMDISA dialect ops are defined here:
#include "mlir/Dialect/AMDISA/IR/AMDISAOps.h"

namespace mlir {
class MLIRContext;
} // namespace mlir

namespace mlir::amdisa {

//===----------------------------------------------------------------------===//
// AMDISAAsmParser Class Definition
//===----------------------------------------------------------------------===//

class AMDISAAsmParser {
public:
  explicit AMDISAAsmParser(llvm::StringRef buffer)
      : buffer(buffer) {}

  mlir::OwningOpRef<mlir::ModuleOp>
  parseModule(mlir::MLIRContext &context);

private:
  llvm::StringRef buffer;

  llvm::StringRef cleanLine(llvm::StringRef line) const;

  bool parseLine(llvm::StringRef line,
                 std::string &mnemonic,
                 llvm::SmallVectorImpl<std::string> &operands,
                 std::string &rawText) const;
};

static bool isLabel(llvm::StringRef line) {
  line = line.trim();
  if (line.empty())
    return false;
  if (!line.ends_with(":"))
    return false;
  llvm::StringRef name = line.drop_back().trim();
  return !name.empty();
}


//===----------------------------------------------------------------------===//
// Utility: Trim + clean line
//===----------------------------------------------------------------------===//

llvm::StringRef AMDISAAsmParser::cleanLine(llvm::StringRef line) const {
  line = line.trim();
  if (line.empty())
    return line;
  return line; // TODO: remove comments
}

//===----------------------------------------------------------------------===//
// Parse one ISA instruction line
//===----------------------------------------------------------------------===//

bool AMDISAAsmParser::parseLine(llvm::StringRef line,
                                std::string &mnemonic,
                                llvm::SmallVectorImpl<std::string> &operands,
                                std::string &rawText) const {

  rawText = line.str();

  auto firstSpace = line.find(' ');
  if (firstSpace == llvm::StringRef::npos) {
    mnemonic = line.str();
    return true;
  }

  mnemonic = line.take_front(firstSpace).str();
  llvm::StringRef operandStr = line.drop_front(firstSpace).trim();

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
// Parse entire buffer into a ModuleOp
//===----------------------------------------------------------------------===//

mlir::OwningOpRef<mlir::ModuleOp>
AMDISAAsmParser::parseModule(mlir::MLIRContext &context) {

  mlir::OpBuilder builder(&context);
  auto module = mlir::ModuleOp::create(builder.getUnknownLoc());

  llvm::StringRef buf = buffer;
  while (!buf.empty()) {
    llvm::StringRef line;
    std::tie(line, buf) = buf.split('\n');

    line = cleanLine(line);
    if (line.empty())
      continue;

    if (isLabel(line)) {
      StringRef labelName = line.drop_back().trim();  // remove ':'
      auto loc = builder.getUnknownLoc();
      auto nameAttr = builder.getStringAttr(labelName);

      builder.setInsertionPointToEnd(module.getBody());
      builder.create<LabelOp>(loc, nameAttr);
      continue;
    }

    std::string mnemonic, raw;
    llvm::SmallVector<std::string, 8> ops;

    if (!parseLine(line, mnemonic, ops, raw)) {
      llvm::errs() << "Warning: failed to parse line: " << line << "\n";
      continue;
    }

    auto loc = builder.getUnknownLoc();
    auto mnemonicAttr = builder.getStringAttr(mnemonic);

    llvm::SmallVector<mlir::Attribute, 8> operandAttrs;
    for (auto &s : ops)
      operandAttrs.push_back(builder.getStringAttr(s));

    auto operandsAttr = builder.getArrayAttr(operandAttrs);
    auto rawTextAttr = builder.getStringAttr(raw);

    builder.setInsertionPointToEnd(module.getBody());

    builder.create<InstOp>(
        loc, mnemonicAttr, operandsAttr, rawTextAttr);
  }

  return module;
}

} // namespace mlir::amdisa
