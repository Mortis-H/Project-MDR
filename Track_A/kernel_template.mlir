module attributes {gpu.container_module} {
  gpu.module @kernels {

    // Kernel signature, generated after analyzing user logic.
    gpu.func @vec_add(%A : !llvm.ptr, %B : !llvm.ptr, %C : !llvm.ptr, %N : index) kernel {

      // User logic in assembly, raised to LLVM MLIR dialect
      llvm.inline_asm has_side_effects asm_dialect = att "        s_load_dword s3, s[0:1], 0x2c", "" : () -> ()
      llvm.inline_asm has_side_effects asm_dialect = att "        s_load_dword s4, s[0:1], 0x18", "" : () -> ()
      llvm.inline_asm has_side_effects asm_dialect = att "        s_waitcnt lgkmcnt(0)", "" : () -> ()
      llvm.inline_asm has_side_effects asm_dialect = att "        s_and_b32 s3, s3, 0xffff", "" : () -> ()
      llvm.inline_asm has_side_effects asm_dialect = att "        s_mul_i32 s2, s2, s3", "" : () -> ()
      llvm.inline_asm has_side_effects asm_dialect = att "        v_add_u32_e32 v0, s2, v0", "" : () -> ()
      llvm.inline_asm has_side_effects asm_dialect = att "        v_cmp_gt_i32_e32 vcc, s4, v0", "" : () -> ()
      llvm.inline_asm has_side_effects asm_dialect = att "        s_and_saveexec_b64 s[2:3], vcc", "" : () -> ()
      llvm.inline_asm has_side_effects asm_dialect = att "        s_cbranch_execz .LBB0_2", "" : () -> ()
      llvm.inline_asm has_side_effects asm_dialect = att "        s_load_dwordx4 s[4:7], s[0:1], 0x0", "" : () -> ()
      llvm.inline_asm has_side_effects asm_dialect = att "        s_load_dwordx2 s[2:3], s[0:1], 0x10", "" : () -> ()
      llvm.inline_asm has_side_effects asm_dialect = att "        v_ashrrev_i32_e32 v1, 31, v0", "" : () -> ()
      llvm.inline_asm has_side_effects asm_dialect = att "        v_lshlrev_b64 v[0:1], 2, v[0:1]", "" : () -> ()
      llvm.inline_asm has_side_effects asm_dialect = att "        s_waitcnt lgkmcnt(0)", "" : () -> ()
      llvm.inline_asm has_side_effects asm_dialect = att "        v_lshl_add_u64 v[4:5], s[4:5], 0, v[0:1]", "" : () -> ()
      llvm.inline_asm has_side_effects asm_dialect = att "        v_lshl_add_u64 v[2:3], s[6:7], 0, v[0:1]", "" : () -> ()
      llvm.inline_asm has_side_effects asm_dialect = att "        global_load_dword v6, v[4:5], off", "" : () -> ()
      llvm.inline_asm has_side_effects asm_dialect = att "        global_load_dword v7, v[2:3], off", "" : () -> ()
      llvm.inline_asm has_side_effects asm_dialect = att "        v_lshl_add_u64 v[0:1], s[2:3], 0, v[0:1]", "" : () -> ()
      llvm.inline_asm has_side_effects asm_dialect = att "        s_waitcnt vmcnt(0)", "" : () -> ()

      // ===========================

      //////////////////////////////
      // Register clobbing after studying user logic
      //////////////////////////////
      %reserved = llvm.inline_asm has_side_effects asm_dialect = att "", "={v[0:31]}": () -> vector<32xi32>

      //////////////////////////////
      // Value binding to read the register in the user logic
      //////////////////////////////

      // Bind A
      %val_A = llvm.inline_asm has_side_effects asm_dialect = att "v_mov_b32 $0, v6", "=v": () -> f32

      // Bind B
      %val_B = llvm.inline_asm has_side_effects asm_dialect = att "v_mov_b32 $0, v7", "=v": () -> f32

      //////////////////////////////
      // DSL section
      //////////////////////////////

      // Demonstrate:
      // - indexing with GPU dialect
      // - very simple control flow using CF dialect
      // - printf() with formatted strings

      %tid = gpu.thread_id x
      %flag = arith.constant 3 : index
      %is_positive = arith.cmpi eq, %tid, %flag : index
      gpu.printf "TID = %d, FLAG = %d, condition = eq, is_positive = %d\n", %tid, %flag, %is_positive : index, index, i1

      cf.cond_br %is_positive, ^bbPOSITIVE_1, ^bbMERGE_1
      ^bbPOSITIVE_1:
          gpu.printf "A[%d] printed inside kernel = %4.3f\n", %tid, %val_A : index, f32
          gpu.printf "B[%d] printed inside kernel = %4.3f\n", %tid, %val_B : index, f32
          cf.br ^bbMERGE_1

      ^bbMERGE_1:

      // ===========================

      // Original kernel logic continue
      llvm.inline_asm has_side_effects asm_dialect = att "        v_add_f32_e32 v2, v6, v7", "" : () -> ()
      llvm.inline_asm has_side_effects asm_dialect = att "        global_store_dword v[0:1], v2, off", "" : () -> ()
      llvm.inline_asm has_side_effects asm_dialect = att ".LBB0_2:", "" : () -> ()

      // ===========================

      //////////////////////////////
      // Value binding to read the register in the user logic
      //////////////////////////////

      // Bind C
      %val_C = llvm.inline_asm has_side_effects asm_dialect = att "v_mov_b32 $0, v2", "=v": () -> f32

      //////////////////////////////
      // DSL section
      //////////////////////////////

      // Demonstrate utility values used by DSL should be better be created from scratch
      %tid_2 = gpu.thread_id x
      %flag_2 = arith.constant 2 : index
      %is_positive_2 = arith.cmpi eq, %tid_2, %flag_2 : index
      gpu.printf "TID = %d, FLAG = %d, condition = eq, is_positive = %d\n", %tid_2, %flag_2, %is_positive_2 : index, index, i1

      cf.cond_br %is_positive_2, ^bbPOSITIVE_2, ^bbMERGE_2
      ^bbPOSITIVE_2:
          gpu.printf "C[%d] printed inside kernel = %4.3f\n", %tid_2, %val_C : index, f32
          cf.br ^bbMERGE_2

      ^bbMERGE_2:

      //////////////////////////////
      // DSL section
      //////////////////////////////

      // Demonstrate different comparison condition
      %flag_3 = arith.constant 4 : index
      %is_positive_3 = arith.cmpi slt, %tid_2, %flag_3 : index
      gpu.printf "TID = %d, FLAG = %d, condition = slt, is_positive = %d\n", %tid_2, %flag_3, %is_positive_3 : index, index, i1

      cf.cond_br %is_positive_3, ^bbPOSITIVE_3, ^bbMERGE_3
      ^bbPOSITIVE_3:
          gpu.printf "A[%d] printed inside kernel = %4.3f\n", %tid_2, %val_A : index, f32
          gpu.printf "B[%d] printed inside kernel = %4.3f\n", %tid_2, %val_B : index, f32
          gpu.printf "C[%d] printed inside kernel = %4.3f\n", %tid_2, %val_C : index, f32
          cf.br ^bbMERGE_3

      ^bbMERGE_3:

      //////////////////////////////
      // DSL section
      //////////////////////////////

      // Demonstrate simple math operations
      %val_B2 = arith.mulf %val_B, %val_B : f32
      %val_AC = arith.mulf %val_A, %val_C : f32
      %c4 = arith.constant 4.0 : f32
      %val_4AC = arith.mulf %c4, %val_AC : f32
      %val_B2_4AC = arith.subf %val_B2, %val_4AC : f32
      gpu.printf "B[%d]^2-4A[%d]C[%d] = %4.3f\n", %tid_2, %tid_2, %tid_2, %val_B2_4AC : index, index, index, f32

      // Register clobbing end
      llvm.inline_asm has_side_effects asm_dialect = att "", "{v[0:31]}" %reserved : (vector<32xi32>)-> ()

      gpu.return
    }
  }
}

