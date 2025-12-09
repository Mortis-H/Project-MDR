#include "llvm/Config/llvm-config.h"

#include "llvm/TargetParser/Triple.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <string>

#include "ParsedProgram.h"

using namespace llvm;

// -----------------------------------------------------------------------------
// CLI options
// -----------------------------------------------------------------------------
static cl::opt<std::string>
    TripleStr("triple",
              cl::desc("Target triple for assembly parsing"),
              cl::init("amdgcn-amd-amdhsa"));

static cl::opt<std::string>
    CPU("mcpu",
        cl::desc("Target CPU for assembly parsing"),
        cl::init("gfx950"));

static cl::opt<std::string>
    InputFilename(cl::Positional,
                  cl::desc("<input asm file>"),
                  cl::init("-"));

static cl::opt<bool>
    ShowSummary("summary",
                cl::desc("Show parsing summary"),
                cl::init(false));

static cl::opt<bool>
    ShowOperandTypes("show-types",
                     cl::desc("Show operand types in output"),
                     cl::init(false));

static cl::opt<bool>
    HideComments("hide-comments",
                 cl::desc("Hide comment lines from output"),
                 cl::init(false));

// -----------------------------------------------------------------------------
// Line kind
// -----------------------------------------------------------------------------
enum class LineKind {
  Unknown,
  Label,
  Instruction,
  Directive,
  Comment,
  Metadata,
  KernelName,
};

static StringRef kindToPrefix(LineKind K) {
  switch (K) {
  case LineKind::Label:
    return "[Label]";
  case LineKind::Instruction:
    return "[Instruction]";
  case LineKind::Directive:
    return "[Directive]";
  case LineKind::Comment:
    return "[Comment]";
  case LineKind::Metadata:
    return "[Metadata]";
  case LineKind::KernelName:
    return "[Kernel Name]";
  case LineKind::Unknown:
  default:
    return "";
  }
}

// -----------------------------------------------------------------------------
// Instruction information structure
// -----------------------------------------------------------------------------
struct InstructionInfo {
  std::string opcode;                  // 操作碼，例如 "s_load_dwordx2"
  SmallVector<std::string, 4> operands; // 操作數列表，例如 ["s[50:51]", "s[0:1]", "0x0"]
  
  InstructionInfo() = default;
  
  bool isEmpty() const {
    return opcode.empty();
  }
  
  void clear() {
    opcode.clear();
    operands.clear();
  }
};

// 解析指令行，提取 opcode 和 operands（舊版，用於輸出）
static InstructionInfo parseInstruction(StringRef line) {
  InstructionInfo info;
  
  // 去除前後空白
  line = line.trim();
  if (line.empty())
    return info;
  
  // 找到第一個空白字符，分離 opcode 和 operands
  size_t firstSpace = line.find_first_of(" \t");
  if (firstSpace == StringRef::npos) {
    // 只有 opcode，沒有 operands
    info.opcode = line.str();
    return info;
  }
  
  // 提取 opcode
  info.opcode = line.substr(0, firstSpace).str();
  
  // 提取 operands 部分
  StringRef operandsPart = line.substr(firstSpace).trim();
  if (operandsPart.empty())
    return info;
  
  // 分割 operands（以逗號分隔）
  SmallVector<StringRef, 4> operandRefs;
  operandsPart.split(operandRefs, ',', -1, false);
  
  for (StringRef op : operandRefs) {
    op = op.trim();
    if (!op.empty()) {
      info.operands.push_back(op.str());
    }
  }
  
  return info;
}

// 解析指令行，創建 ParsedInstruction（新版，帶類型信息）
static ParsedInstruction parseInstructionEnhanced(StringRef line, 
                                                   size_t lineNum,
                                                   StringRef originalText) {
  ParsedInstruction inst;
  inst.lineNumber = lineNum;
  inst.originalText = originalText.str();
  
  // 去除前後空白
  line = line.trim();
  if (line.empty())
    return inst;
  
  // 找到第一個空白字符，分離 opcode 和 operands
  size_t firstSpace = line.find_first_of(" \t");
  if (firstSpace == StringRef::npos) {
    // 只有 opcode，沒有 operands
    inst.opcode = line.str();
    return inst;
  }
  
  // 提取 opcode
  inst.opcode = line.substr(0, firstSpace).str();
  
  // 提取 operands 部分
  StringRef operandsPart = line.substr(firstSpace).trim();
  if (operandsPart.empty())
    return inst;
  
  // 分割 operands（以逗號分隔）
  SmallVector<StringRef, 4> operandRefs;
  operandsPart.split(operandRefs, ',', -1, false);
  
  for (StringRef op : operandRefs) {
    op = op.trim();
    if (!op.empty()) {
      OperandType type = detectOperandType(op);
      inst.operands.emplace_back(op.str(), type);
    }
  }
  
  return inst;
}

// -----------------------------------------------------------------------------
// AnnotatingStreamer: MCStreamer subclass to tag lines
// -----------------------------------------------------------------------------
class AnnotatingStreamer : public MCStreamer {
  SourceMgr &SM;
  SmallVectorImpl<LineKind> &Kinds;

  void mark(SMLoc Loc, LineKind K) {
    if (!Loc.isValid())
      return;

    unsigned BufID = SM.FindBufferContainingLoc(Loc);
    if (!BufID)
      return;

    auto LC = SM.getLineAndColumn(Loc, BufID);
    unsigned Line = LC.first; // 1-based

    if (Line == 0 || Line > Kinds.size())
      return;

    LineKind &Slot = Kinds[Line - 1];
    if (Slot == LineKind::Unknown)
      Slot = K;
  }

public:
  AnnotatingStreamer(MCContext &Ctx,
                     SourceMgr &SM,
                     SmallVectorImpl<LineKind> &Kinds)
      : MCStreamer(Ctx), SM(SM), Kinds(Kinds) {}

  void emitInstruction(const MCInst &Inst,
                       const MCSubtargetInfo &STI) override {
    mark(Inst.getLoc(), LineKind::Instruction);
    MCStreamer::emitInstruction(Inst, STI);
  }

  void emitLabel(MCSymbol *Symbol, SMLoc Loc = SMLoc()) override {
    mark(Loc, LineKind::Label);
    MCStreamer::emitLabel(Symbol, Loc);
  }

  // pure virtuals: no-op
  bool emitSymbolAttribute(MCSymbol *Symbol,
                           MCSymbolAttr Attribute) override {
    (void)Symbol;
    (void)Attribute;
    return true;
  }

  void emitCommonSymbol(MCSymbol *Symbol,
                        uint64_t Size,
                        Align ByteAlignment) override {
    (void)Symbol;
    (void)Size;
    (void)ByteAlignment;
  }
};

static bool isCommentLine(StringRef L) {
  L = L.ltrim();
  if (L.empty())
    return false;
  if (L.starts_with("#") || L.starts_with("//") || L.starts_with(";"))
    return true;
  return false;
}

int main(int argc, char **argv) {
  InitLLVM X(argc, argv);

  cl::ParseCommandLineOptions(argc, argv,
                              "AMD GCN ISA annotator (MCStreamer-based)\n");

  // 1. Initialize targets / MC
  InitializeAllTargetInfos();
  InitializeAllTargets();
  InitializeAllTargetMCs();
  InitializeAllAsmParsers();

  std::string TT = Triple::normalize(TripleStr);
  std::string Error;
  const Target *TheTarget = TargetRegistry::lookupTarget(TT, Error);
  if (!TheTarget) {
    WithColor::error(errs(), "amdisa_annotate")
        << "Failed to lookup target for triple '" << TT << "': "
        << Error << "\n";
    return 1;
  }

  // 2. Read original file
  ErrorOr<std::unique_ptr<MemoryBuffer>> BufferOrErr =
      MemoryBuffer::getFileOrSTDIN(InputFilename);
  if (!BufferOrErr) {
    WithColor::error(errs(), "amdisa_annotate")
        << "Could not open input file '" << InputFilename
        << "': " << BufferOrErr.getError().message() << "\n";
    return 1;
  }

  std::unique_ptr<MemoryBuffer> &OrigBuf = BufferOrErr.get();
  StringRef OrigRef = OrigBuf->getBuffer();

  // Split original content into lines (for final output)
  SmallVector<StringRef, 0> Lines;
  OrigRef.split(Lines, '\n', -1, /*KeepEmpty=*/true);

  SmallVector<LineKind, 0> Kinds;
  Kinds.assign(Lines.size(), LineKind::Unknown);

  // Track which lines are inside metadata blocks
  SmallVector<bool, 0> InMetadata;
  InMetadata.assign(Lines.size(), false);

  // Store instruction information for each line
  SmallVector<InstructionInfo, 0> InstructionInfos;
  InstructionInfos.resize(Lines.size());

  // 3. Build a "sanitized" text for MCAsmParser:
  //    - Same number of lines
  //    - But certain problematic directives are replaced by comment lines
  std::string ParsedText;
  ParsedText.reserve(OrigRef.size());

  bool inMetadataBlock = false;
  size_t lineIdx = 0;

  for (auto L : Lines) {
    StringRef Line = L;
    if (!Line.empty() && Line.back() == '\r')
      Line = Line.drop_back();
    StringRef Trimmed = Line.ltrim();

    // Track if we're inside a .amdgpu_metadata block
    if (Trimmed.starts_with(".amdgpu_metadata")) {
      inMetadataBlock = true;
    }

    // Mark lines that are inside metadata blocks
    if (inMetadataBlock) {
      InMetadata[lineIdx] = true;
    }

    // Strip out directives that may cause parsing issues
    // - Lines inside metadata blocks
    // - Directives containing "amdgcn", "amdhsa", or "amdgpu"
    if (inMetadataBlock || 
        (Trimmed.starts_with(".") && 
         (Trimmed.contains("amdgcn") || Trimmed.contains("amdhsa") || 
          Trimmed.contains("amdgpu")))) {
      ParsedText += "; stripped: ";
      ParsedText += Line.str();
    } else {
      ParsedText += Line.str();
    }
    ParsedText.push_back('\n');

    // Check if we're exiting a metadata block
    if (Trimmed.starts_with(".end_amdgpu_metadata")) {
      inMetadataBlock = false;
    }

    lineIdx++;
  }

  // Create MemoryBuffer from ParsedText for SourceMgr / MCAsmParser
  auto ParseBuf = MemoryBuffer::getMemBufferCopy(ParsedText, InputFilename);

  SourceMgr SrcMgr;
  SrcMgr.AddNewSourceBuffer(std::move(ParseBuf), SMLoc());

  // 4. MC setup
  MCTargetOptions MCOpts;
  Triple TheTriple(TT);

  std::unique_ptr<MCRegisterInfo> MRI(
      TheTarget->createMCRegInfo(TT));
  if (!MRI) {
    WithColor::error(errs(), "amdisa_annotate")
        << "Target does not support MCRegisterInfo for '" << TT << "'.\n";
    return 1;
  }

  std::unique_ptr<MCAsmInfo> MAI(
      TheTarget->createMCAsmInfo(*MRI, TheTriple, MCOpts));
  if (!MAI) {
    WithColor::error(errs(), "amdisa_annotate")
        << "Failed to create MCAsmInfo.\n";
    return 1;
  }

  std::unique_ptr<MCInstrInfo> MCII(TheTarget->createMCInstrInfo());
  if (!MCII) {
    WithColor::error(errs(), "amdisa_annotate")
        << "Failed to create MCInstrInfo.\n";
    return 1;
  }

  std::unique_ptr<MCSubtargetInfo> STI(
      TheTarget->createMCSubtargetInfo(TheTriple, CPU, /*FeaturesStr=*/""));
  if (!STI) {
    WithColor::error(errs(), "amdisa_annotate")
        << "Failed to create MCSubtargetInfo.\n";
    return 1;
  }

  MCContext Ctx(TheTriple, MAI.get(), MRI.get(), STI.get(),
                &SrcMgr, &MCOpts);

  // MCObjectFileInfo: 安全起見還是建立，但我們不走 initSections
  std::unique_ptr<MCObjectFileInfo> MOFI(
      TheTarget->createMCObjectFileInfo(Ctx, /*PIC=*/false));
  if (!MOFI) {
    WithColor::error(errs(), "amdisa_annotate")
        << "Failed to create MCObjectFileInfo.\n";
    return 1;
  }

  auto Streamer =
      std::make_unique<AnnotatingStreamer>(Ctx, SrcMgr, Kinds);

  std::unique_ptr<MCAsmParser> Parser(
      createMCAsmParser(SrcMgr, Ctx, *Streamer, *MAI));
  if (!Parser) {
    WithColor::error(errs(), "amdisa_annotate")
        << "Failed to create MCAsmParser.\n";
    return 1;
  }

  std::unique_ptr<MCTargetAsmParser> TAP(
      TheTarget->createMCAsmParser(*STI, *Parser, *MCII, MCOpts));
  if (!TAP) {
    WithColor::error(errs(), "amdisa_annotate")
        << "Target does not support AsmParser for this triple/CPU.\n";
    return 1;
  }

  Parser->setTargetParser(*TAP);

  // 5. Run parser
  //    使用 NoInitialTextSection = true，避免 initSections 的 object-emission 路徑
  if (Parser->Run(/*NoInitialTextSection=*/true)) {
    WithColor::error(errs(), "amdisa_annotate")
        << "Parsing of input assembly failed.\n";
    return 1;
  }

  // 6. second pass: fill Directive / Comment / Metadata for Unknown lines
  //    and parse instructions
  //    同時構建結構化的 ParsedProgram
  ParsedProgram program;
  program.filename = InputFilename;
  program.totalLines = Lines.size();
  
  for (size_t I = 0, E = Lines.size(); I != E; ++I) {
    StringRef Line = Lines[I];
    if (!Line.empty() && Line.back() == '\r')
      Line = Line.drop_back();

    StringRef Trimmed = Line.ltrim();

    // Lines inside metadata blocks should always be marked as Metadata
    // This overrides any other classification
    if (InMetadata[I]) {
      Kinds[I] = LineKind::Metadata;
      continue;
    }

    // Parse instruction information for lines already marked as Instruction
    if (Kinds[I] == LineKind::Instruction) {
      InstructionInfos[I] = parseInstruction(Trimmed);
      
      // 同時創建增強版的 ParsedInstruction
      ParsedInstruction parsedInst = parseInstructionEnhanced(
          Trimmed, I + 1, Line);
      program.allInstructions.push_back(parsedInst);
      continue;
    }

    // 收集標籤信息
    if (Kinds[I] == LineKind::Label) {
      ParsedLabel label;
      label.name = Trimmed.rtrim(':').str();
      label.lineNumber = I + 1;
      label.originalText = Line.str();
      program.labels.push_back(label);
      continue;
    }

    if (Kinds[I] != LineKind::Unknown)
      continue;

    if (Trimmed.empty())
      continue;

    if (isCommentLine(Trimmed)) {
      Kinds[I] = LineKind::Comment;
      continue;
    }

    if (Trimmed.starts_with("---") || Trimmed.starts_with("..."))
      continue;

    // Check for .globl directive (kernel name declaration)
    if (Trimmed.starts_with(".globl") || Trimmed.starts_with(".global")) {
      Kinds[I] = LineKind::KernelName;
      continue;
    }

    if (Trimmed.starts_with("."))
      Kinds[I] = LineKind::Directive;
  }

  // 7. output: prefix + original line content
  //    (可選：顯示操作數類型)
  for (size_t I = 0, E = Lines.size(); I != E; ++I) {
    StringRef Line = Lines[I];
    if (!Line.empty() && Line.back() == '\r')
      Line = Line.drop_back();

    StringRef Trimmed = Line.ltrim();
    LineKind K = Kinds[I];

    // Skip comments if requested
    if (HideComments && K == LineKind::Comment) {
      continue;
    }

    // Special handling for Metadata lines: keep original formatting
    if (K == LineKind::Metadata) {
      outs() << Line << '\n';
      continue;
    }

    StringRef Prefix = kindToPrefix(K);

    if (Trimmed.empty() || Prefix.empty()) {
      outs() << Line << '\n';
    } else if (K == LineKind::Instruction && !InstructionInfos[I].isEmpty()) {
      // Special formatting for instructions with parsed information
      const InstructionInfo &info = InstructionInfos[I];
      outs() << Prefix << " " << info.opcode;
      
      if (!info.operands.empty()) {
        outs() << " [";
        for (size_t i = 0; i < info.operands.size(); ++i) {
          if (i > 0) outs() << ", ";
          outs() << "\"" << info.operands[i] << "\"";
          
          // 可選：顯示操作數類型
          if (ShowOperandTypes) {
            // 從 ParsedProgram 中查找對應的指令
            const ParsedInstruction *parsedInst = 
                program.getInstructionAtLine(I + 1);
            if (parsedInst && i < parsedInst->operands.size()) {
              const Operand &op = parsedInst->operands[i];
              const char *typeStr = "";
              switch (op.type) {
                case OperandType::Register: typeStr = ":reg"; break;
                case OperandType::Immediate: typeStr = ":imm"; break;
                case OperandType::Label: typeStr = ":label"; break;
                case OperandType::Expression: typeStr = ":expr"; break;
                default: typeStr = ":?"; break;
              }
              outs() << typeStr;
            }
          }
        }
        outs() << "]";
      }
      outs() << '\n';
    } else {
      outs() << Prefix << " " << Trimmed << '\n';
    }
  }

  // 8. 可選：顯示解析摘要
  if (ShowSummary) {
    errs() << "\n";
    program.printSummary(errs());
  }

  return 0;
}
