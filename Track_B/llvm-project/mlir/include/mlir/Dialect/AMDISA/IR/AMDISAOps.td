//===- AMDISA.td - AMDISA dialect ------------------------------*- tablegen -*-===//
//
//  A minimal dialect that wraps AMD ISA instructions from .s
//
//===----------------------------------------------------------------------===//

#ifndef AMDISA_OPS
#define AMDISA_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def AMDISA_Dialect : Dialect {
  let name = "amdisa";                        // textual dialect name: "amdisa"
  let cppNamespace = "::mlir::amdisa";        // C++ namespace
  let summary = "AMDISA wrapper dialect";
  let description = [{
    This dialect provides a single generic operation `amdisa.inst`
    that wraps AMD GPU ISA instructions parsed from .s files.
  }];
}

// Convenience base class for all AMDISA ops.
class AMDISA_Op<string mnemonic, list<Trait> traits = []>
    : Op<AMDISA_Dialect, mnemonic, traits>;

// ---- InstOp ----

def AMDISA_InstOp : AMDISA_Op<"inst", [Pure]> {
  let summary = "Generic AMD ISA instruction wrapper";
  let description = [{
    This op represents a single AMD ISA instruction parsed from a .s file.
    It does not enforce any typing or register constraints.

    Example:

      amdisA.inst
        { mnemonic = "s_add_i32",
          operands = ["s25", "s5", "0x100"],
          raw_text = "s_add_i32 s25, s5, 0x100" }

  }];

  let arguments = (ins
    StrAttr:$mnemonic,
    ArrayAttr:$operands,
    OptionalAttr<StrAttr>:$raw_text
  );

  let results = (outs);

  // 先用自動格式就好：會印成
  //   amdisA.inst {mnemonic = "...", operands = [...]} 
  let assemblyFormat = "attr-dict";
}

#endif // AMDISA_OPS