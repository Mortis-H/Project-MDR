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
      llvm.inline_asm has_side_effects asm_dialect = att "        v_add_f32_e32 v2, v6, v7", "" : () -> ()
      llvm.inline_asm has_side_effects asm_dialect = att "        global_store_dword v[0:1], v2, off", "" : () -> ()
      llvm.inline_asm has_side_effects asm_dialect = att ".LBB0_2:", "" : () -> ()

      // Value binding to read the register in the user logic
      %val_user = llvm.inline_asm has_side_effects asm_dialect = att "v_mov_b32 $0, v2", "=v,~{v[0:31]}": () -> f32

      // Register clobbing after studying user logic
      %reserved = llvm.inline_asm has_side_effects asm_dialect = att "", "={v[0:31]}": () -> vector<32xi32>

      // DSL section
      // Only support printf() for now
      gpu.printf "C printed inside kernel = %4.3f\n", %val_user : f32

      // Register clobbing end
      llvm.inline_asm has_side_effects asm_dialect = att "", "{v[0:31]}" %reserved : (vector<32xi32>)-> ()

      gpu.return
    }
  }
}
