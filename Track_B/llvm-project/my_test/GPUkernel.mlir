module attributes {amdgpu.code_object_version = 6 : i32, llvm.target_triple = "amdgcn-amd-amdhsa--gfx950"} {
  gpu.module @amdisa_kernels {
    gpu.func @amdisa_kernel() kernel {
      llvm.inline_asm has_side_effects ".LBB0:\0Av_mbcnt_lo_u32_b32 v0, -1, 0\0Av_mbcnt_hi_u32_b32 v10, -1, v0\0Av_mov_b64_e32 v[4:5], 0\0Av_readfirstlane_b32 s2, v10\0A", ""  : () -> ()
      gpu.return
    }
  }
}
