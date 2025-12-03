// RUN: mlir-opt %s -split-input-file

module {
  amdisa.inst { mnemonic = "s_add_u32", operands = ["s1","s2"], raw_text = "s_add_u32 s1, s2" }
}
