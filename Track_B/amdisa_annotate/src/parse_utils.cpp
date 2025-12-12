/**
 * parse_utils.cpp
 * 
 * 從 main.cpp 提取的可重用解析函數
 */

#include "parse_utils.h"
#include "AMDGPUMetadata.h"
#include "ParsedProgram.h"

#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/TargetSelect.h"

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
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/TargetRegistry.h"

#include "llvm/TargetParser/Triple.h"
#include "llvm/MC/MCInstPrinter.h"

#include <map>

using namespace llvm;

// ============================================================================
// Helper structures and functions (從 main.cpp 複製)
// ============================================================================

struct InstructionInfo {
  std::string opcode;
  SmallVector<std::string, 4> operands;
  
  InstructionInfo() = default;
  
  bool isEmpty() const {
    return opcode.empty();
  }
  
  void clear() {
    opcode.clear();
    operands.clear();
  }
};

// 解析指令（簡化版）
static InstructionInfo parseInstruction(StringRef line) {
  InstructionInfo info;
  StringRef trimmed = line.ltrim();
  
  size_t firstSpace = trimmed.find_first_of(" \t");
  if (firstSpace == StringRef::npos) {
    info.opcode = trimmed.str();
    return info;
  }
  
  info.opcode = trimmed.substr(0, firstSpace).str();
  StringRef operandsPart = trimmed.substr(firstSpace).ltrim();
  
  SmallVector<StringRef, 8> operandRefs;
  operandsPart.split(operandRefs, ',', -1, false);
  
  for (auto opRef : operandRefs) {
    StringRef opTrimmed = opRef.trim();
    if (!opTrimmed.empty()) {
      info.operands.push_back(opTrimmed.str());
    }
  }
  
  return info;
}

// 從 MCInst 提取 operands（使用 LLVM MC API）
static std::vector<Operand> extractOperandsFromMCInst(
    const MCInst &Inst,
    const MCInstrInfo &MCII,
    const MCRegisterInfo &MRI,
    const MCSubtargetInfo &STI,
    MCInstPrinter *InstPrinter) {
  
  std::vector<Operand> operands;
  
  if (!InstPrinter)
    return operands;
  
  // 使用 MCInstPrinter 打印完整指令，然後解析
  std::string InstStr;
  raw_string_ostream OS(InstStr);
  InstPrinter->printInst(&Inst, 0, "", STI, OS);
  OS.flush();
  
  // DEBUG: 打印看看實際內容（僅針對 global_load_dwordx2）
  // if (InstStr.find("global_load_dwordx2") != std::string::npos) {
  //   errs() << "DEBUG: MCInstPrinter output: \"" << InstStr << "\"\n";
  // }
  
  // 解析打印出來的指令字串
  // MCInstPrinter 的格式: "\topcode operand1, operand2, ..."
  // 注意開頭可能有 tab 或空格
  StringRef InstRef = StringRef(InstStr).ltrim();
  
  // 跳過 opcode（找到第一個空格或 tab）
  size_t firstSpace = InstRef.find_first_of(" \t");
  if (firstSpace == StringRef::npos)
    return operands;
  
  StringRef operandsPart = InstRef.substr(firstSpace).ltrim();
  
  // 如果沒有操作數，直接返回
  if (operandsPart.empty())
    return operands;
  
  // 使用逗號分割操作數
  // 但是要注意，操作數內部可能包含空格（例如 "s[2:3] offset:16"）
  // 所以只在逗號處分割
  SmallVector<StringRef, 8> operandRefs;
  operandsPart.split(operandRefs, ',', -1, false);
  
  for (auto opRef : operandRefs) {
    StringRef opTrimmed = opRef.trim();
    if (!opTrimmed.empty()) {
      OperandType opType = detectOperandType(opTrimmed);
      operands.push_back(Operand(opTrimmed.str(), opType));
    }
  }
  
  return operands;
}

// 解析指令（增強版，返回 ParsedInstruction）
// 如果有 MCInst 可用，使用它；否則回退到文本解析
static ParsedInstruction parseInstructionEnhanced(
    StringRef line, 
    size_t lineNumber,
    StringRef originalLine,
    const MCInst *Inst = nullptr,
    const MCInstrInfo *MCII = nullptr,
    const MCRegisterInfo *MRI = nullptr,
    const MCSubtargetInfo *STI = nullptr,
    MCInstPrinter *InstPrinter = nullptr) {
  
  ParsedInstruction inst;
  inst.lineNumber = lineNumber;
  inst.originalText = originalLine.str();
  
  StringRef trimmed = line.ltrim();
  size_t firstSpace = trimmed.find_first_of(" \t");
  
  if (firstSpace == StringRef::npos) {
    inst.opcode = trimmed.str();
    return inst;
  }
  
  inst.opcode = trimmed.substr(0, firstSpace).str();
  
  // 如果有 MCInst，使用它來提取 operands
  if (Inst && MCII && MRI && STI && InstPrinter) {
    inst.operands = extractOperandsFromMCInst(*Inst, *MCII, *MRI, *STI, InstPrinter);
  } else {
    // 回退到文本解析
    StringRef operandsPart = trimmed.substr(firstSpace).ltrim();
    
    SmallVector<StringRef, 8> operandRefs;
    operandsPart.split(operandRefs, ',', -1, false);
    
    for (auto opRef : operandRefs) {
      StringRef opTrimmed = opRef.trim();
      if (!opTrimmed.empty()) {
        OperandType opType = detectOperandType(opTrimmed);
        inst.operands.push_back(Operand(opTrimmed.str(), opType));
      }
    }
  }
  
  return inst;
}

// 檢查是否為註解行
static bool isCommentLine(StringRef L) {
  if (L.empty())
    return false;
  if (L.starts_with("#") || L.starts_with("//") || L.starts_with(";"))
    return true;
  return false;
}

// ============================================================================
// AnnotatingStreamer (從 main.cpp 複製，增強版：捕獲 MCInst)
// ============================================================================

class AnnotatingStreamer : public MCStreamer {
  SourceMgr &SM;
  SmallVectorImpl<LineKind> &Kinds;
  const MCInstrInfo &MCII;
  const MCRegisterInfo &MRI;
  std::map<unsigned, MCInst> &InstMap;  // 儲存每行的 MCInst

public:
  AnnotatingStreamer(MCContext &Ctx,
                     SourceMgr &SM,
                     SmallVectorImpl<LineKind> &Kinds,
                     const MCInstrInfo &MCII,
                     const MCRegisterInfo &MRI,
                     std::map<unsigned, MCInst> &InstMap)
      : MCStreamer(Ctx), SM(SM), Kinds(Kinds), MCII(MCII), MRI(MRI), InstMap(InstMap) {}

  void emitLabel(MCSymbol *Symbol, SMLoc Loc) override {
    if (!Loc.isValid())
      return;
    unsigned LineNo = SM.FindLineNumber(Loc);
    if (LineNo > 0 && LineNo <= Kinds.size())
      Kinds[LineNo - 1] = LineKind::Label;
  }

  void emitInstruction(const MCInst &Inst,
                       const MCSubtargetInfo &STI) override {
    SMLoc Loc = Inst.getLoc();
    if (!Loc.isValid())
      return;
    unsigned LineNo = SM.FindLineNumber(Loc);
    if (LineNo > 0 && LineNo <= Kinds.size()) {
      Kinds[LineNo - 1] = LineKind::Instruction;
      // 儲存 MCInst 供後續使用
      InstMap[LineNo] = Inst;
    }
  }

  bool emitSymbolAttribute(MCSymbol *Symbol,
                           MCSymbolAttr Attribute) override {
    return true;
  }

  void emitCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                        Align ByteAlignment) override {}
  void emitZerofill(MCSection *Section, MCSymbol *Symbol, uint64_t Size,
                    Align ByteAlignment, SMLoc Loc) override {}
};

// ============================================================================
// 主要解析函數
// ============================================================================

AMDGCNAssembly parseAMDGCNAssembly(const std::string &filename,
                                    const std::string &triple,
                                    const std::string &mcpu) {
  // 初始化 LLVM targets
  static bool initialized = false;
  if (!initialized) {
    InitializeAllTargetInfos();
    InitializeAllTargets();
    InitializeAllTargetMCs();
    InitializeAllAsmParsers();
    initialized = true;
  }

  std::string TT = Triple::normalize(triple);
  std::string Error;
  const Target *TheTarget = TargetRegistry::lookupTarget(TT, Error);
  if (!TheTarget) {
    errs() << "Error: Failed to lookup target for triple '" << TT << "': "
           << Error << "\n";
    return AMDGCNAssembly();
  }

  // 讀取檔案
  ErrorOr<std::unique_ptr<MemoryBuffer>> BufferOrErr =
      MemoryBuffer::getFileOrSTDIN(filename);
  if (!BufferOrErr) {
    errs() << "Error: Could not open input file '" << filename
           << "': " << BufferOrErr.getError().message() << "\n";
    return AMDGCNAssembly();
  }

  std::unique_ptr<MemoryBuffer> &OrigBuf = BufferOrErr.get();
  StringRef OrigRef = OrigBuf->getBuffer();

  // 分割成行
  SmallVector<StringRef, 0> Lines;
  OrigRef.split(Lines, '\n', -1, true);

  SmallVector<LineKind, 0> Kinds;
  Kinds.assign(Lines.size(), LineKind::Unknown);

  SmallVector<bool, 0> InMetadata;
  InMetadata.assign(Lines.size(), false);

  SmallVector<InstructionInfo, 0> InstructionInfos;
  InstructionInfos.resize(Lines.size());

  AMDGPUMetadata metadata;

  // 構建 sanitized text 並提取 metadata
  std::string ParsedText;
  ParsedText.reserve(OrigRef.size());

  bool inMetadataBlock = false;
  size_t lineIdx = 0;
  size_t metadataStartLine = 0;
  std::string metadataYAML;

  for (auto L : Lines) {
    StringRef Line = L;
    if (!Line.empty() && Line.back() == '\r')
      Line = Line.drop_back();
    StringRef Trimmed = Line.ltrim();

    // 追蹤 metadata block
    if (Trimmed.starts_with(".amdgpu_metadata")) {
      inMetadataBlock = true;
      metadataStartLine = lineIdx;
      metadataYAML.clear();
    }

    if (inMetadataBlock) {
      InMetadata[lineIdx] = true;
      
      if (!Trimmed.starts_with(".amdgpu_metadata") && 
          !Trimmed.starts_with(".end_amdgpu_metadata")) {
        metadataYAML += Line.str();
        metadataYAML += "\n";
      }
    }

    // Strip 掉有問題的 directives
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

    if (Trimmed.starts_with(".end_amdgpu_metadata")) {
      inMetadataBlock = false;
      
      if (!metadataYAML.empty()) {
        MetadataParser::parse(metadataYAML, metadataStartLine, lineIdx, metadata);
      }
    }

    lineIdx++;
  }

  // 創建 SourceMgr
  auto ParseBuf = MemoryBuffer::getMemBufferCopy(ParsedText, filename);
  SourceMgr SrcMgr;
  SrcMgr.AddNewSourceBuffer(std::move(ParseBuf), SMLoc());

  // MC setup
  MCTargetOptions MCOpts;
  Triple TheTriple(TT);

  std::unique_ptr<MCRegisterInfo> MRI(
      TheTarget->createMCRegInfo(TT));
  if (!MRI) {
    errs() << "Error: Target does not support MCRegisterInfo\n";
    return AMDGCNAssembly();
  }

  std::unique_ptr<MCAsmInfo> MAI(
      TheTarget->createMCAsmInfo(*MRI, TheTriple, MCOpts));
  if (!MAI) {
    errs() << "Error: Failed to create MCAsmInfo\n";
    return AMDGCNAssembly();
  }

  std::unique_ptr<MCInstrInfo> MCII(TheTarget->createMCInstrInfo());
  if (!MCII) {
    errs() << "Error: Failed to create MCInstrInfo\n";
    return AMDGCNAssembly();
  }

  std::unique_ptr<MCSubtargetInfo> STI(
      TheTarget->createMCSubtargetInfo(TheTriple, mcpu, ""));
  if (!STI) {
    errs() << "Error: Failed to create MCSubtargetInfo\n";
    return AMDGCNAssembly();
  }

  MCContext Ctx(TheTriple, MAI.get(), MRI.get(), STI.get(),
                &SrcMgr, &MCOpts);

  std::unique_ptr<MCObjectFileInfo> MOFI(
      TheTarget->createMCObjectFileInfo(Ctx, false));
  if (!MOFI) {
    errs() << "Error: Failed to create MCObjectFileInfo\n";
    return AMDGCNAssembly();
  }

  // 創建 map 來儲存每行的 MCInst
  std::map<unsigned, MCInst> InstMap;
  
  auto Streamer =
      std::make_unique<AnnotatingStreamer>(Ctx, SrcMgr, Kinds, *MCII, *MRI, InstMap);

  std::unique_ptr<MCAsmParser> Parser(
      createMCAsmParser(SrcMgr, Ctx, *Streamer, *MAI));
  if (!Parser) {
    errs() << "Error: Failed to create MCAsmParser\n";
    return AMDGCNAssembly();
  }

  std::unique_ptr<MCTargetAsmParser> TAP(
      TheTarget->createMCAsmParser(*STI, *Parser, *MCII, MCOpts));
  if (!TAP) {
    errs() << "Error: Target does not support AsmParser\n";
    return AMDGCNAssembly();
  }

  Parser->setTargetParser(*TAP);

  // 執行解析
  if (Parser->Run(true)) {
    // 解析失敗，但繼續處理（可能只是部分失敗）
  }

  // 創建 MCInstPrinter 用於打印 operands
  std::unique_ptr<MCInstPrinter> InstPrinter(
      TheTarget->createMCInstPrinter(Triple(TT), 0, *MAI, *MCII, *MRI));

  // 第二遍：分類行類型並解析指令
  ParsedProgram program;
  program.filename = filename;
  program.totalLines = Lines.size();

  for (size_t I = 0, E = Lines.size(); I != E; ++I) {
    StringRef Line = Lines[I];
    if (!Line.empty() && Line.back() == '\r')
      Line = Line.drop_back();

    StringRef Trimmed = Line.ltrim();

    // Metadata 行
    if (InMetadata[I]) {
      Kinds[I] = LineKind::Metadata;
      continue;
    }

    // 解析指令
    if (Kinds[I] == LineKind::Instruction) {
      InstructionInfos[I] = parseInstruction(Trimmed);
      
      // 從 InstMap 中獲取 MCInst（如果有）
      unsigned LineNo = I + 1;
      const MCInst *Inst = nullptr;
      auto it = InstMap.find(LineNo);
      if (it != InstMap.end()) {
        Inst = &it->second;
      }
      
      ParsedInstruction parsedInst = parseInstructionEnhanced(
          Trimmed, I + 1, Line, Inst, MCII.get(), MRI.get(), STI.get(), InstPrinter.get());
      program.allInstructions.push_back(parsedInst);
      continue;
    }

    // 收集標籤
    if (Kinds[I] == LineKind::Label) {
      ParsedLabel label;
      // 去掉尾部的冒號
      StringRef labelName = Trimmed;
      if (labelName.ends_with(":")) {
        labelName = labelName.drop_back();
      }
      label.name = labelName.str();
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

    // .amdgcn_target directive
    if (Trimmed.starts_with(".amdgcn_target")) {
      Kinds[I] = LineKind::AmdgcnTarget;
      continue;
    }

    // .amdhsa_code_object_version directive
    if (Trimmed.starts_with(".amdhsa_code_object_version")) {
      Kinds[I] = LineKind::AmdhsaCodeObjectVersion;
      continue;
    }

    // .globl directive (kernel name)
    if (Trimmed.starts_with(".globl") || Trimmed.starts_with(".global")) {
      Kinds[I] = LineKind::KernelName;
      continue;
    }

    if (Trimmed.starts_with("."))
      Kinds[I] = LineKind::Directive;
  }

  // 構建 AMDGCNAssembly 對象
  AMDGCNAssembly assembly;
  assembly.setFilename(filename);
  assembly.setParsedProgram(program);
  assembly.setMetadata(metadata);

  // 填充行資訊
  for (size_t I = 0; I < Lines.size(); ++I) {
    LineInfo lineInfo;
    lineInfo.lineNumber = I + 1;
    lineInfo.text = Lines[I].str();
    lineInfo.kind = Kinds[I];
    
    // 如果是指令，添加詳細資訊
    if (lineInfo.kind == LineKind::Instruction && !InstructionInfos[I].isEmpty()) {
      ParsedInstruction inst;
      inst.opcode = InstructionInfos[I].opcode;
      inst.lineNumber = I + 1;
      inst.originalText = Lines[I].str();
      
      const ParsedInstruction *programInst = program.getInstructionAtLine(I + 1);
      if (programInst) {
        inst.operands = programInst->operands;
      } else {
        for (const auto &op : InstructionInfos[I].operands) {
          Operand operand(op, detectOperandType(op));
          inst.operands.push_back(operand);
        }
      }
      
      lineInfo.instruction = inst;
    }
    
    // 如果是標籤，填充標籤名稱
    if (lineInfo.kind == LineKind::Label) {
      // 從 program.labels 中查找對應行號的標籤
      const ParsedLabel *foundLabel = nullptr;
      for (const auto &label : program.labels) {
        if (label.lineNumber == I + 1) {
          foundLabel = &label;
          break;
        }
      }
      
      if (foundLabel) {
        lineInfo.labelName = foundLabel->name;
      } else {
        // 備用方案：從文本中提取標籤名
        StringRef trimmed = Lines[I].ltrim();
        size_t colonPos = trimmed.find(':');
        if (colonPos != StringRef::npos) {
          lineInfo.labelName = trimmed.substr(0, colonPos).str();
        }
      }
    }
    
    // 如果是 kernel 名稱，填充 kernel 名稱
    if (lineInfo.kind == LineKind::KernelName) {
      StringRef trimmed = Lines[I].ltrim();
      // kernel 名稱行通常是 ".globl <name>" 或 "<name>:"
      if (trimmed.starts_with(".globl")) {
        StringRef kernelPart = trimmed.substr(6).ltrim();
        lineInfo.kernelName = kernelPart.str();
      }
    }
    
    // 如果是 amdgcn_target，填充內容
    if (lineInfo.kind == LineKind::AmdgcnTarget) {
      StringRef trimmed = Lines[I].ltrim();
      // .amdgcn_target "amdgcn-amd-amdhsa--gfx950"
      if (trimmed.starts_with(".amdgcn_target")) {
        StringRef content = trimmed.substr(14).ltrim();
        lineInfo.amdgcnTarget = content.str();
      }
    }
    
    // 如果是 amdhsa_code_object_version，填充內容
    if (lineInfo.kind == LineKind::AmdhsaCodeObjectVersion) {
      StringRef trimmed = Lines[I].ltrim();
      // .amdhsa_code_object_version 6
      if (trimmed.starts_with(".amdhsa_code_object_version")) {
        StringRef content = trimmed.substr(27).ltrim();
        lineInfo.amdhsaCodeObjectVersion = content.str();
      }
    }
    
    // 如果是 directive，拆分為 name 和 content
    if (lineInfo.kind == LineKind::Directive) {
      StringRef trimmed = Lines[I].ltrim();
      // 找第一個空白字符
      size_t spacePos = trimmed.find_first_of(" \t");
      if (spacePos != StringRef::npos) {
        lineInfo.directiveName = trimmed.substr(0, spacePos).str();
        lineInfo.directiveContent = trimmed.substr(spacePos).ltrim().str();
      } else {
        // 沒有空白，整個都是 directive name
        lineInfo.directiveName = trimmed.str();
        lineInfo.directiveContent = "";
      }
    }
    
    assembly.addLine(lineInfo);
  }

  // 構建 Label Blocks（建立 label 和指令的關係）
  assembly.buildLabelBlocks();

  return assembly;
}

