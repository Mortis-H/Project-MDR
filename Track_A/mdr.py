#!/usr/bin/env python3
import argparse
import pathlib
import re
import shutil
import subprocess
import sys
import textwrap


# =========================
# Part A: generate MLIR module
# =========================

def generate_mlir_module() -> str:
    """
    Generate an MLIR module with:
      - gpu.module @kernels containing a single kernel @print_hex(i32)
      - host-side func.func @main that launches the kernel with gpu.launch_func

    The kernel prints its single i32 argument as a hexadecimal number.
    """
    mlir = textwrap.dedent(
        r"""
        module attributes {gpu.container_module} {
          gpu.module @kernels {
            gpu.func @print_hex(%arg0: i32) kernel {
              llvm.inline_asm has_side_effects asm_dialect = att "blah:", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15},~{v16},~{v17},~{v18},~{v19},~{v20},~{v21},~{v22},~{v23},~{v24},~{v25},~{v26},~{v27},~{v28},~{v29},~{v30},~{31}": () -> ()
              llvm.inline_asm has_side_effects asm_dialect = att "v_mov_b32_e32 v31, 31", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15},~{v16},~{v17},~{v18},~{v19},~{v20},~{v21},~{v22},~{v23},~{v24},~{v25},~{v26},~{v27},~{v28},~{v29},~{v30},~{31}": () -> ()
              %val31 = llvm.inline_asm has_side_effects asm_dialect = att "v_mov_b32 $0, v31", "=v,~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15},~{v16},~{v17},~{v18},~{v19},~{v20},~{v21},~{v22},~{v23},~{v24},~{v25},~{v26},~{v27},~{v28},~{v29},~{v30},~{31}": () -> i32
              gpu.printf "vgpr = 0x%x\n", %val31 : i32

              //%a = arith.addi %arg0, %arg0: i32
              //%b = arith.addi %a, %arg0: i32

              // Plain inline asm works
              //llvm.inline_asm has_side_effects asm_dialect = att "s_nop 0", "" : () -> ()

              // Test register clobbing
              //llvm.inline_asm has_side_effects asm_dialect = att "v_mov_b32_e32 v0, 0", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15}": () -> ()
              //llvm.inline_asm has_side_effects asm_dialect = att "v_mov_b32_e32 v1, 1", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15}": () -> ()
              //llvm.inline_asm has_side_effects asm_dialect = att "v_mov_b32_e32 v2, 2", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15}": () -> ()
              //llvm.inline_asm has_side_effects asm_dialect = att "v_mov_b32_e32 v3, 3", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15}": () -> ()
              //llvm.inline_asm has_side_effects asm_dialect = att "v_mov_b32_e32 v4, 4", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15}": () -> ()
              //llvm.inline_asm has_side_effects asm_dialect = att "v_mov_b32_e32 v5, 5", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15}": () -> ()
              //llvm.inline_asm has_side_effects asm_dialect = att "v_mov_b32_e32 v6, 6", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15}": () -> ()
              //llvm.inline_asm has_side_effects asm_dialect = att "v_mov_b32_e32 v7, 7", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15}" : () -> ()
              //llvm.inline_asm has_side_effects asm_dialect = att "v_mov_b32_e32 v8, 8", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15}": () -> ()
              //llvm.inline_asm has_side_effects asm_dialect = att "v_mov_b32_e32 v9, 9", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15}" : () -> ()
              //llvm.inline_asm has_side_effects asm_dialect = att "v_mov_b32_e32 v10, 10", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15}": () -> ()
              //llvm.inline_asm has_side_effects asm_dialect = att "v_mov_b32_e32 v11, 11", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15}" : () -> ()
              //llvm.inline_asm has_side_effects asm_dialect = att "v_mov_b32_e32 v12, 12", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15}": () -> ()
              //llvm.inline_asm has_side_effects asm_dialect = att "v_mov_b32_e32 v13, 13", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15}": () -> ()
              //llvm.inline_asm has_side_effects asm_dialect = att "v_mov_b32_e32 v14, 14", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15}": () -> ()
              //llvm.inline_asm has_side_effects asm_dialect = att "v_mov_b32_e32 v15, 15", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15}": () -> ()

              // bring VGPR to a value
              //%val0 = llvm.inline_asm has_side_effects asm_dialect = att "v_mov_b32 $0, v0", "=v,~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15}" : () -> i32
              //%val1 = llvm.inline_asm has_side_effects asm_dialect = att "v_mov_b32 $0, v1", "=v,~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15}" : () -> i32
              //%val2 = llvm.inline_asm has_side_effects asm_dialect = att "v_mov_b32 $0, v2", "=v,~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15}" : () -> i32
              //%val3 = llvm.inline_asm has_side_effects asm_dialect = att "v_mov_b32 $0, v3", "=v,~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15}" : () -> i32
              //%val4 = llvm.inline_asm has_side_effects asm_dialect = att "v_mov_b32 $0, v4", "=v,~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15}" : () -> i32
              //%val5 = llvm.inline_asm has_side_effects asm_dialect = att "v_mov_b32 $0, v5", "=v,~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15}" : () -> i32
              //%val6 = llvm.inline_asm has_side_effects asm_dialect = att "v_mov_b32 $0, v6", "=v,~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15}" : () -> i32
              //%val7 = llvm.inline_asm has_side_effects asm_dialect = att "v_mov_b32 $0, v7", "=v,~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15}" : () -> i32
              //%val8 = llvm.inline_asm has_side_effects asm_dialect = att "v_mov_b32 $0, v8", "=v,~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15}" : () -> i32
              //%val9 = llvm.inline_asm has_side_effects asm_dialect = att "v_mov_b32 $0, v9", "=v,~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15}" : () -> i32
              //%val10 = llvm.inline_asm has_side_effects asm_dialect = att "v_mov_b32 $0, v10", "=v,~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15}" : () -> i32
              //%val11 = llvm.inline_asm has_side_effects asm_dialect = att "v_mov_b32 $0, v11", "=v,~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15}" : () -> i32
              //%val12 = llvm.inline_asm has_side_effects asm_dialect = att "v_mov_b32 $0, v12", "=v,~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15}" : () -> i32
              //%val13 = llvm.inline_asm has_side_effects asm_dialect = att "v_mov_b32 $0, v13", "=v,~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15}" : () -> i32
              //%val14 = llvm.inline_asm has_side_effects asm_dialect = att "v_mov_b32 $0, v14", "=v,~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15}" : () -> i32
              //%val15 = llvm.inline_asm has_side_effects asm_dialect = att "v_mov_b32 $0, v15", "=v,~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15}" : () -> i32

              //llvm.inline_asm has_side_effects asm_dialect = att "", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15}" : () -> () 
              //gpu.printf "vgpr = 0x%x\n", %val15 : i32
              //llvm.inline_asm has_side_effects asm_dialect = att "", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15}" : () -> () 
              //gpu.printf "vgpr = 0x%x\n", %val14 : i32
              //llvm.inline_asm has_side_effects asm_dialect = att "", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15}" : () -> () 
              //gpu.printf "vgpr = 0x%x\n", %val13 : i32
              //llvm.inline_asm has_side_effects asm_dialect = att "", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15}" : () -> () 
              //gpu.printf "vgpr = 0x%x\n", %val12 : i32
              //llvm.inline_asm has_side_effects asm_dialect = att "", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15}" : () -> () 
              //gpu.printf "vgpr = 0x%x\n", %val11 : i32
              //llvm.inline_asm has_side_effects asm_dialect = att "", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15}" : () -> () 
              //gpu.printf "vgpr = 0x%x\n", %val10 : i32
              //llvm.inline_asm has_side_effects asm_dialect = att "", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15}" : () -> () 
              //gpu.printf "vgpr = 0x%x\n", %val9 : i32
              //llvm.inline_asm has_side_effects asm_dialect = att "", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15}" : () -> () 
              //gpu.printf "vgpr = 0x%x\n", %val8 : i32
              //llvm.inline_asm has_side_effects asm_dialect = att "", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15}" : () -> () 
              //gpu.printf "vgpr = 0x%x\n", %val7 : i32
              //llvm.inline_asm has_side_effects asm_dialect = att "", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15}" : () -> () 
              //gpu.printf "vgpr = 0x%x\n", %val6 : i32
              //llvm.inline_asm has_side_effects asm_dialect = att "", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15}" : () -> () 
              //gpu.printf "vgpr = 0x%x\n", %val5 : i32
              //llvm.inline_asm has_side_effects asm_dialect = att "", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15}" : () -> () 
              //gpu.printf "vgpr = 0x%x\n", %val4 : i32
              //llvm.inline_asm has_side_effects asm_dialect = att "", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15}" : () -> () 
              //gpu.printf "vgpr = 0x%x\n", %val3 : i32
              //llvm.inline_asm has_side_effects asm_dialect = att "", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15}" : () -> () 
              //gpu.printf "vgpr = 0x%x\n", %val2 : i32
              //llvm.inline_asm has_side_effects asm_dialect = att "", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15}" : () -> () 
              //gpu.printf "vgpr = 0x%x\n", %val1 : i32
              //llvm.inline_asm has_side_effects asm_dialect = att "", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15}" : () -> () 
              //gpu.printf "vgpr = 0x%x\n", %val0 : i32
              //llvm.inline_asm has_side_effects asm_dialect = att "", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15}" : () -> () 

              // store kernel argument into a VGPR
              //llvm.inline_asm has_side_effects asm_dialect = att "v_mov_b32_e32 v0, $0", "v,~{v0}" %arg0 : (i32) -> ()
              // retrieve kernarg via VGPR
              //%kernarg = llvm.inline_asm has_side_effects asm_dialect = att "v_mov_b32 $0, v0", "=v,~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15}" : () -> i32
              //gpu.printf "kernarg = 0x%x\n", %kernarg : i32

              // printf with kernel argument works
              gpu.printf "arg = 0x%x\n", %arg0 : i32

              // printf with computed values work
              //%a = arith.addi %arg0, %arg0: i32
              //%b = arith.addi %a, %arg0: i32
              //gpu.printf "arg = 0x%x\n", %a : i32
              //gpu.printf "arg = 0x%x\n", %b : i32

              gpu.return
            }
          }

          func.func @main() {
            %c1 = arith.constant 1 : index
            %i1 = arith.constant 65536 : i32
            gpu.launch_func @kernels::@print_hex
              blocks in (%c1, %c1, %c1)
              threads in (%c1, %c1, %c1)
              args(%i1 : i32)
              return
          }
        }
        """
    ).lstrip()
    return mlir


# ===============================
# Helpers: tools & subprocess
# ===============================

def ensure_tool(name: str):
    if shutil.which(name) is None:
        raise RuntimeError(f"Required tool '{name}' not found in PATH")


def run_cmd(cmd, cwd=None):
    print("[$]", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, cwd=cwd)


# ======================================
# String decoding for gpu.binary assembly (ISA) – same as before
# ======================================

ASM_ATTR_RE = re.compile(
    r'gpu\.binary\b.*?assembly\s*=\s*"((?:[^"\\]|\\.)*)"',
    re.DOTALL,
)


def decode_mlir_string(raw: str) -> str:
    """
    Decode an MLIR string literal into a Python str.

    MLIR for assembly often uses:
      \09 -> 0x09 (tab)
      \22 -> 0x22 (double quote)
      \0A -> 0x0A (newline)
    which is "backslash + two hex digits".
    """

    def _hex_repl(m: re.Match) -> str:
        hex_val = m.group(1)
        try:
            return chr(int(hex_val, 16))
        except ValueError:
            return "\\" + hex_val

    # First, convert \XX (two hex digits) to the corresponding character
    s = re.sub(r"\\([0-9A-Fa-f]{2})", _hex_repl, raw)
    # Then let unicode_escape handle \n, \t, \", \\, etc.
    return bytes(s, "utf-8").decode("unicode_escape")


def extract_assembly_strings(mlir_text: str):
    out = []
    for raw in ASM_ATTR_RE.findall(mlir_text):
        out.append(decode_mlir_string(raw))
    return out


# ======================================
# String decoding for offload/offloading bitcode (format=llvm)
# ======================================

BITCODE_ATTR_RE = re.compile(
    r'(?:offload|offloading)\s*=\s*"((?:[^"\\]|\\.)*)"',
    re.DOTALL,
)


def decode_mlir_bytes(raw: str) -> bytes:
    """
    Decode an MLIR string literal into raw bytes.

    Example:
      "BC\\C0\\DE5\\14\\00\\00..."  (backslash + two hex digits)
    """

    out = bytearray()
    i = 0
    n = len(raw)

    while i < n:
        ch = raw[i]
        if ch == "\\" and i + 1 < n:
            nxt = raw[i + 1]
            # \XX where X are hex digits
            if i + 2 < n and nxt in "0123456789abcdefABCDEF" and raw[i + 2] in "0123456789abcdefABCDEF":
                val = int(raw[i + 1 : i + 3], 16)
                out.append(val)
                i += 3
                continue
            # Some common escapes, just in case
            if nxt == "n":
                out.append(ord("\n"))
                i += 2
                continue
            if nxt == "t":
                out.append(ord("\t"))
                i += 2
                continue
            if nxt == "r":
                out.append(ord("\r"))
                i += 2
                continue
            if nxt == '"':
                out.append(ord('"'))
                i += 2
                continue
            if nxt == "\\":
                out.append(ord("\\"))
                i += 2
                continue
            # Fallback: keep the backslash as-is
            out.append(ord(ch))
            i += 1
        else:
            out.append(ord(ch))
            i += 1

    return bytes(out)


def extract_bitcode_bytes(mlir_text: str):
    out = []
    for raw in BITCODE_ATTR_RE.findall(mlir_text):
        out.append(decode_mlir_bytes(raw))
    return out


# =====================================
# Part B-1: device-only pipeline – ISA & HSACO (format=isa)
# =====================================

def build_isa_and_hsaco(chip: str, workdir: pathlib.Path):
    for tool in ["mlir-opt", "llvm-mc", "ld.lld"]:
        ensure_tool(tool)

    kernel_mlir = workdir / "kernel.mlir"
    kernel_binary_mlir = workdir / "kernel_binary_isa.mlir"
    kernel_isa_s = workdir / "kernel_isa.s"
    kernel_o = workdir / "kernel.o"
    kernel_hsaco = workdir / "kernel.hsaco"

    pipeline = (
        f"builtin.module("
        f"gpu-kernel-outlining,"
        f"rocdl-attach-target{{chip={chip}}},"
        f"gpu.module(convert-gpu-to-rocdl{{index-bitwidth=32 runtime=HIP}}),"
        f"gpu-to-llvm,"
        f"gpu-module-to-binary{{format=isa}}"
        f")"
    )

    mlir_opt_cmd = [
        "mlir-opt",
        str(kernel_mlir),
        f"--pass-pipeline={pipeline}",
        "-o",
        str(kernel_binary_mlir),
    ]
    run_cmd(mlir_opt_cmd)

    binary_text = kernel_binary_mlir.read_text()
    isa_list = extract_assembly_strings(binary_text)

    if not isa_list:
        raise RuntimeError("No gpu.binary assembly attribute (ISA) found in MLIR output")

    if len(isa_list) > 1:
        print(f"[!] Found {len(isa_list)} gpu.binary entries (ISA), using the first one.")

    isa = isa_list[0]
    kernel_isa_s.write_text(isa)
    print(f"Wrote ISA assembly to {kernel_isa_s}")

    llvm_mc_cmd = [
        "llvm-mc",
        "-triple", "amdgcn-amd-amdhsa",
        f"-mcpu={chip}",
        "-filetype=obj",
        str(kernel_isa_s),
        "-o",
        str(kernel_o),
    ]
    run_cmd(llvm_mc_cmd)

    ld_cmd = [
        "ld.lld",
        "-shared",
        str(kernel_o),
        "-o",
        str(kernel_hsaco),
    ]
    run_cmd(ld_cmd)

    print(f"Generated HSACO: {kernel_hsaco}")


# =====================================
# Part B-2: device-only pipeline – LLVM IR (format=llvm)
# =====================================

def build_llvm_ir_via_binary(chip: str, workdir: pathlib.Path):
    ensure_tool("mlir-opt")
    ensure_tool("llvm-dis")

    kernel_mlir = workdir / "kernel.mlir"
    kernel_binary_mlir = workdir / "kernel_binary_llvm.mlir"
    kernel_llvm_bc = workdir / "kernel_llvm.bc"
    kernel_llvm_ll = workdir / "kernel_llvm.ll"

    pipeline = (
        f"builtin.module("
        f"gpu-kernel-outlining,"
        f"rocdl-attach-target{{chip={chip}}},"
        f"gpu.module(convert-gpu-to-rocdl{{index-bitwidth=32 runtime=HIP}}),"
        f"gpu-to-llvm,"
        f"gpu-module-to-binary{{format=llvm}}"
        f")"
    )

    mlir_opt_cmd = [
        "mlir-opt",
        str(kernel_mlir),
        f"--pass-pipeline={pipeline}",
        "-o",
        str(kernel_binary_mlir),
    ]
    run_cmd(mlir_opt_cmd)

    binary_text = kernel_binary_mlir.read_text()
    bc_list = extract_bitcode_bytes(binary_text)

    if not bc_list:
        raise RuntimeError("No offload/offloading bitcode attribute found in MLIR output")

    if len(bc_list) > 1:
        print(f"[!] Found {len(bc_list)} offload entries (bitcode), using the first one.")

    bitcode = bc_list[0]
    kernel_llvm_bc.write_bytes(bitcode)
    print(f"Wrote LLVM bitcode to {kernel_llvm_bc}")

    llvm_dis_cmd = [
        "llvm-dis",
        str(kernel_llvm_bc),
        "-o",
        str(kernel_llvm_ll),
    ]
    run_cmd(llvm_dis_cmd)

    print(f"Wrote human-readable LLVM IR to {kernel_llvm_ll}")


# =====================================
# Part B-3: host + device – end-to-end run via mlir-runner
# =====================================

def auto_detect_mlir_libs() -> tuple[pathlib.Path, pathlib.Path] | tuple[None, None]:
    """
    Try to auto-detect libmlir_rocm_runtime.so and libmlir_runner_utils.so
    based on the location of mlir-runner (or mlir-opt) in PATH.

    Assumes a typical LLVM build layout: build/bin, build/lib.
    """
    runner_path = shutil.which("mlir-runner") or shutil.which("mlir-opt")
    if not runner_path:
        return None, None

    bin_dir = pathlib.Path(runner_path).parent
    lib_dir = bin_dir.parent / "lib"

    rocm_rt = lib_dir / "libmlir_rocm_runtime.so"
    runner_utils = lib_dir / "libmlir_runner_utils.so"

    if rocm_rt.exists() and runner_utils.exists():
        return rocm_rt, runner_utils

    return None, None


def build_and_run_host(chip: str,
                       workdir: pathlib.Path,
                       rocm_runtime_lib: str | None,
                       runner_utils_lib: str | None):
    """
    Lower host + device and run with mlir-runner.

    Pipeline (split in two mlir-opt invocations, then mlir-runner):

      1) mlir-opt -pass-pipeline="
           builtin.module(
             gpu.module(strip-debuginfo,
                        convert-gpu-to-rocdl{index-bitwidth=32 runtime=HIP}),
             rocdl-attach-target{chip=<chip>}
           )"

      2) mlir-opt -gpu-to-llvm -reconcile-unrealized-casts -gpu-module-to-binary

      3) mlir-runner --shared-libs=... \
                     --shared-libs=... \
                     --entry-point-result=void
    """
    for tool in ["mlir-opt", "mlir-runner"]:
        ensure_tool(tool)

    # Try auto-detect libs if not provided
    auto_rocm_rt, auto_runner_utils = auto_detect_mlir_libs()

    rocm_runtime_lib_path = pathlib.Path(
        rocm_runtime_lib if rocm_runtime_lib is not None else (
            str(auto_rocm_rt) if auto_rocm_rt is not None else ""
        )
    )
    runner_utils_lib_path = pathlib.Path(
        runner_utils_lib if runner_utils_lib is not None else (
            str(auto_runner_utils) if auto_runner_utils is not None else ""
        )
    )

    if not rocm_runtime_lib_path.is_file() or not runner_utils_lib_path.is_file():
        raise RuntimeError(
            "Could not locate libmlir_rocm_runtime.so or libmlir_runner_utils.so.\n"
            "Please pass them explicitly via:\n"
            "  --rocm-runtime-lib /path/to/libmlir_rocm_runtime.so\n"
            "  --runner-utils-lib /path/to/libmlir_runner_utils.so\n"
            "or ensure a standard LLVM build layout so they can be auto-detected."
        )

    print(f"Using ROCm runtime lib: {rocm_runtime_lib_path}")
    print(f"Using runner utils lib: {runner_utils_lib_path}")

    kernel_mlir = workdir / "kernel.mlir"
    host_step1_mlir = workdir / "host_lowered_step1.mlir"
    host_final_mlir = workdir / "host_lowered_final.mlir"

    # Step 1: gpu.module(strip-debuginfo, convert-gpu-to-rocdl{...}), rocdl-attach-target
    pipeline1 = (
        f"builtin.module("
        f"gpu.module(strip-debuginfo,convert-gpu-to-rocdl{{index-bitwidth=32 runtime=HIP}}),"
        f"rocdl-attach-target{{chip={chip}}}"
        f")"
    )

    mlir_opt_cmd1 = [
        "mlir-opt",
        str(kernel_mlir),
        f"--pass-pipeline={pipeline1}",
        "-o",
        str(host_step1_mlir),
    ]
    run_cmd(mlir_opt_cmd1)

    # Step 2: -gpu-to-llvm -reconcile-unrealized-casts -gpu-module-to-binary
    mlir_opt_cmd2 = [
        "mlir-opt",
        str(host_step1_mlir),
        "-gpu-to-llvm",
        "-reconcile-unrealized-casts",
        "-gpu-module-to-binary",
        "-o",
        str(host_final_mlir),
    ]
    run_cmd(mlir_opt_cmd2)

    # Step 3: mlir-runner
    mlir_runner_cmd = [
        "mlir-runner",
        str(host_final_mlir),
        f"--shared-libs={rocm_runtime_lib_path}",
        f"--shared-libs={runner_utils_lib_path}",
        "--entry-point-result=void",
    ]
    run_cmd(mlir_runner_cmd)


# ==========
#   main
# ==========

def main():
    ap = argparse.ArgumentParser(
        description="Generate a simple AMDGPU MLIR kernel and build HSACO / LLVM IR / run host code (device-only pipeline)."
    )
    ap.add_argument(
        "--chip",
        default="gfx950",
        help="AMDGPU chip (for rocdl-attach-target & llvm-mc -mcpu) [default: gfx950]",
    )
    ap.add_argument(
        "--workdir",
        default="build_mdr",
        help="directory to put intermediate files [default: build_mdr]",
    )
    ap.add_argument(
        "--emit-isa",
        action="store_true",
        default=True,
        help="run device-only pipeline with gpu-module-to-binary{format=isa} and build HSACO",
    )
    ap.add_argument(
        "--no-emit-isa",
        dest="emit_isa",
        action="store_false",
        help="disable ISA / HSACO generation",
    )
    ap.add_argument(
        "--emit-llvm-ir",
        action="store_true",
        help="also run device-only pipeline with gpu-module-to-binary{format=llvm} and dump LLVM IR (via llvm-dis)",
    )
    ap.add_argument(
        "--run-host",
        action="store_true",
        help="lower host+device and run @main via mlir-runner",
    )
    ap.add_argument(
        "--rocm-runtime-lib",
        help="path to libmlir_rocm_runtime.so (optional; will be auto-detected if possible)",
    )
    ap.add_argument(
        "--runner-utils-lib",
        help="path to libmlir_runner_utils.so (optional; will be auto-detected if possible)",
    )
    args = ap.parse_args()

    workdir = pathlib.Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    # Part A: write MLIR source
    kernel_mlir = workdir / "kernel.mlir"
    mlir_text = generate_mlir_module()
    kernel_mlir.write_text(mlir_text)
    print(f"Wrote MLIR to {kernel_mlir}")

    # Part B: device-only pipelines
    if args.emit_isa:
        build_isa_and_hsaco(args.chip, workdir)

    if args.emit_llvm_ir:
        build_llvm_ir_via_binary(args.chip, workdir)

    # Part C: host + device end-to-end run
    if args.run_host:
        build_and_run_host(
            chip=args.chip,
            workdir=workdir,
            rocm_runtime_lib=args.rocm_runtime_lib,
            runner_utils_lib=args.runner_utils_lib,
        )

    if not args.emit_isa and not args.emit_llvm_ir and not args.run_host:
        print("Nothing to do: ISA, LLVM IR emission, and host run are all disabled.")


if __name__ == "__main__":
    main()

