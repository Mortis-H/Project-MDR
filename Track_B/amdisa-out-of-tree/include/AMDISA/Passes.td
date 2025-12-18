//===-- Passes.td - AMDISA pass definition file ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file contains definitions for passes under the AMDISA dialect.
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_AMDISA_PASSES
#define MLIR_DIALECT_AMDISA_PASSES

include "mlir/Pass/PassBase.td"

def LowerAMDISAToGPUInlineAsm :
    Pass<"amdisa-lower-to-gpu-inline-asm", "mlir::ModuleOp"> {

  let summary = "Lower AMDISA ops into gpu.module + gpu.func + llvm.inline_asm";
  let description = [{
    This pass collects all amdisa.label and amdisa.inst ops and lowers them
    into a gpu.module containing a gpu.func. All AMDISA raw_text is embedded
    into llvm.inline_asm for further lowering to LLVM IR and HSACO.
  }];

  let dependentDialects = [
    "gpu::GPUDialect",
    "LLVM::LLVMDialect"
  ];

  let options = [
    Option<"kernelName", "kernel-name", "std::string",
          "\"amdisa_kernel\"",
          "Name of the generated GPU kernel function">,

    Option<"targetTriple", "target-triple", "std::string",
          "\"amdgcn-amd-amdhsa--gfx950\"",
          "Target triple to attach to the module">,

    Option<"codeObjectVersion", "code-object-version", "int",
          "6",
          "AMDGPU code object version number">
  ];


  let constructor = "mlir::amdisa::createLowerAMDISAToGPUInlineAsmPass()";
}

#endif // MLIR_DIALECT_AMDISA_PASSES
