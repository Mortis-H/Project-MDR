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

// ---- LabelOp ----
//
// Represents a label from the .s file, such as:
//   L1:
//   .Ltmp3:
//   .LBB0_2:
//
// This op acts as a placeholder marker, but has no semantics.
//
def AMDISA_LabelOp : AMDISA_Op<"label", [Pure]> {
  let summary = ".s label wrapper";
  let description = [{
    A label defined in the .s file (e.g., `L1:` or `.LBB0_3:`).

    Example:
      amdisa.label { name = "L1" }
  }];

  let arguments = (ins
    StrAttr:$name
  );

  let results = (outs);

  // Just print attributes, no operands.
  let assemblyFormat = "attr-dict";
}

// ---- FuncOp ----
//
// Represents a kernel or function from the .s file.
// Contains amdisa.inst and amdisa.label ops as body.
// Each kernel has its own metadata (arguments, register usage, etc.)
//
def AMDISA_FuncOp : AMDISA_Op<"func"> {
  let summary = "AMD ISA kernel/function";
  let description = [{
    Represents a kernel or function parsed from AMD GCN assembly.
    Contains the body with instructions and labels, along with metadata
    such as kernel arguments, register usage, and other properties.

    Example:
      amdisa.func @vec_add(%arg0: !amdisa.buffer, %arg1: !amdisa.buffer) 
        attributes {
          amdisa.sgpr_count = 32,
          amdisa.vgpr_count = 16,
          amdisa.kernargs = [...]
        } {
        amdisa.label { name = ".LBB0_1" }
        amdisa.inst { mnemonic = "s_load_dwordx2", operands = [...], raw_text = "..." }
        // ... more instructions ...
        amdisa.return
      }
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name
  );

  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    $sym_name attr-dict-with-keyword $body
  }];

  let hasVerifier = 0;
}

// ---- ReturnOp ----
//
// Terminator for amdisa.func body
//
def AMDISA_ReturnOp : AMDISA_Op<"return", [Pure, Terminator]> {
  let summary = "Return operation for amdisa.func";
  let description = [{
    Terminates the body of an amdisa.func operation.
    
    Example:
      amdisa.func @kernel {
        // ... instructions ...
        amdisa.return
      }
  }];

  let arguments = (ins);
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

#endif // AMDISA_OPS