	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text
	.globl	batched_gemm
	.p2align	8
	.type	batched_gemm,@function
batched_gemm:
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
		s_load_dword s5, s[0:1], 0x48
	;;#ASMEND
	;;#ASMSTART
		s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
		s_cmp_ge_i32 s4, s5
	;;#ASMEND
	;;#ASMSTART
		s_cbranch_scc1 .LBB0_11
	;;#ASMEND
	;;#ASMSTART
		s_load_dwordx4 s[12:15], s[0:1], 0x0
	;;#ASMEND
	;;#ASMSTART
		s_load_dwordx2 s[20:21], s[0:1], 0x10
	;;#ASMEND
	;;#ASMSTART
		s_load_dwordx4 s[8:11], s[0:1], 0x18
	;;#ASMEND
	;;#ASMSTART
		s_load_dwordx2 s[6:7], s[0:1], 0x28
	;;#ASMEND
	;;#ASMSTART
		s_load_dwordx4 s[16:19], s[0:1], 0x30
	;;#ASMEND
	;;#ASMSTART
		s_load_dwordx2 s[22:23], s[0:1], 0x40
	;;#ASMEND
	;;#ASMSTART
		v_and_b32_e32 v2, 0x3ff, v0
	;;#ASMEND
	;;#ASMSTART
		v_bfe_u32 v3, v0, 10, 10
	;;#ASMEND
	;;#ASMSTART
		s_lshl_b32 s2, s2, 4
	;;#ASMEND
	;;#ASMSTART
		v_lshl_add_u32 v1, s3, 4, v3
	;;#ASMEND
	;;#ASMSTART
		v_add_u32_e32 v0, s2, v2
	;;#ASMEND
	;;#ASMSTART
		s_ashr_i32 s5, s4, 31
	;;#ASMEND
	;;#ASMSTART
		s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
		s_cmp_lt_i32 s10, 1
	;;#ASMEND
	;;#ASMSTART
		v_cmp_gt_i32_e32 vcc, s8, v1
	;;#ASMEND
	;;#ASMSTART
		v_cmp_gt_i32_e64 s[0:1], s9, v0
	;;#ASMEND
	;;#ASMSTART
		s_cbranch_scc1 .LBB0_8
	;;#ASMEND
	;;#ASMSTART
		s_mul_i32 s24, s16, s4
	;;#ASMEND
	;;#ASMSTART
		s_mul_hi_u32 s26, s16, s4
	;;#ASMEND
	;;#ASMSTART
		s_mul_i32 s27, s17, s4
	;;#ASMEND
	;;#ASMSTART
		s_add_i32 s25, s26, s27
	;;#ASMEND
	;;#ASMSTART
		s_mov_b32 s30, 4
	;;#ASMEND
	;;#ASMSTART
		s_mul_i32 s28, s24, s30
	;;#ASMEND
	;;#ASMSTART
		s_mul_hi_u32 s32, s24, s30
	;;#ASMEND
	;;#ASMSTART
		s_mul_i32 s33, s25, s30
	;;#ASMEND
	;;#ASMSTART
		s_add_i32 s29, s32, s33
	;;#ASMEND
	;;#ASMSTART
		s_add_u32 s12, s12, s28
	;;#ASMEND
	;;#ASMSTART
		s_addc_u32 s13, s13, s29
	;;#ASMEND
	;;#ASMSTART
		s_mul_i32 s40, s18, s4
	;;#ASMEND
	;;#ASMSTART
		s_mul_hi_u32 s42, s18, s4
	;;#ASMEND
	;;#ASMSTART
		s_mul_i32 s43, s19, s4
	;;#ASMEND
	;;#ASMSTART
		s_add_i32 s41, s42, s43
	;;#ASMEND
	;;#ASMSTART
		s_mov_b32 s46, 4
	;;#ASMEND
	;;#ASMSTART
		s_mul_i32 s44, s40, s46
	;;#ASMEND
	;;#ASMSTART
		s_mul_hi_u32 s48, s40, s46
	;;#ASMEND
	;;#ASMSTART
		s_mul_i32 s49, s41, s46
	;;#ASMEND
	;;#ASMSTART
		s_add_i32 s45, s48, s49
	;;#ASMEND
	;;#ASMSTART
		s_add_u32 s14, s14, s44
	;;#ASMEND
	;;#ASMSTART
		s_addc_u32 s15, s15, s45
	;;#ASMEND
	;;#ASMSTART
		v_lshlrev_b32_e32 v4, 2, v2
	;;#ASMEND
	;;#ASMSTART
		v_lshlrev_b32_e32 v8, 6, v3
	;;#ASMEND
	;;#ASMSTART
		v_add_u32_e32 v9, v8, v4
	;;#ASMEND
	;;#ASMSTART
		v_add_u32_e32 v10, 0x400, v4
	;;#ASMEND
	;;#ASMSTART
		v_mad_u64_u32 v[4:5], s[16:17], s11, v1, v[2:3]
	;;#ASMEND
	;;#ASMSTART
		v_mul_lo_u32 v5, v3, s6
	;;#ASMEND
	;;#ASMSTART
		v_add_u32_e32 v11, v10, v8
	;;#ASMEND
	;;#ASMSTART
		v_add3_u32 v6, v2, v5, s2
	;;#ASMEND
	;;#ASMSTART
		s_lshl_b32 s6, s6, 4
	;;#ASMEND
	;;#ASMSTART
		s_mov_b32 s11, 0
	;;#ASMEND
	;;#ASMSTART
		v_mov_b32_e32 v5, 0
	;;#ASMEND
	;;#ASMSTART
		s_branch .LBB0_4
	;;#ASMEND
	;;#ASMSTART
	.LBB0_3:                                ;   in Loop: Header=BB0_4 Depth=1:
	;;#ASMEND
	;;#ASMSTART
		s_or_b64 exec, exec, s[2:3]
	;;#ASMEND
	;;#ASMSTART
		s_waitcnt vmcnt(0)
	;;#ASMEND
	;;#ASMSTART
		ds_write_b32 v11, v7
	;;#ASMEND
	;;#ASMSTART
		s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
		s_barrier
	;;#ASMEND
	;;#ASMSTART
		ds_read2_b32 v[28:29], v10 offset1:16
	;;#ASMEND
	;;#ASMSTART
		ds_read_b128 v[12:15], v8
	;;#ASMEND
	;;#ASMSTART
		ds_read_b128 v[16:19], v8 offset:16
	;;#ASMEND
	;;#ASMSTART
		ds_read2_b32 v[30:31], v10 offset0:32 offset1:48
	;;#ASMEND
	;;#ASMSTART
		ds_read_b128 v[20:23], v8 offset:32
	;;#ASMEND
	;;#ASMSTART
		ds_read_b128 v[24:27], v8 offset:48
	;;#ASMEND
	;;#ASMSTART
		ds_read2_b32 v[32:33], v10 offset0:64 offset1:80
	;;#ASMEND
	;;#ASMSTART
		s_waitcnt lgkmcnt(5)
	;;#ASMEND
	;;#ASMSTART
		v_fmac_f32_e32 v5, v12, v28
	;;#ASMEND
	;;#ASMSTART
		v_fmac_f32_e32 v5, v13, v29
	;;#ASMEND
	;;#ASMSTART
		s_waitcnt lgkmcnt(3)
	;;#ASMEND
	;;#ASMSTART
		v_fmac_f32_e32 v5, v14, v30
	;;#ASMEND
	;;#ASMSTART
		v_fmac_f32_e32 v5, v15, v31
	;;#ASMEND
	;;#ASMSTART
		ds_read2_b32 v[12:13], v10 offset0:96 offset1:112
	;;#ASMEND
	;;#ASMSTART
		s_waitcnt lgkmcnt(1)
	;;#ASMEND
	;;#ASMSTART
		v_pk_mul_f32 v[14:15], v[16:17], v[32:33]
	;;#ASMEND
	;;#ASMSTART
		ds_read2_b32 v[16:17], v10 offset0:224 offset1:240
	;;#ASMEND
	;;#ASMSTART
		v_add_f32_e32 v5, v5, v14
	;;#ASMEND
	;;#ASMSTART
		v_add_f32_e32 v5, v5, v15
	;;#ASMEND
	;;#ASMSTART
		ds_read2_b32 v[14:15], v10 offset0:128 offset1:144
	;;#ASMEND
	;;#ASMSTART
		s_waitcnt lgkmcnt(2)
	;;#ASMEND
	;;#ASMSTART
		v_pk_mul_f32 v[12:13], v[18:19], v[12:13]
	;;#ASMEND
	;;#ASMSTART
		s_add_i32 s11, s11, 16
	;;#ASMEND
	;;#ASMSTART
		v_add_f32_e32 v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
		v_add_f32_e32 v5, v5, v13
	;;#ASMEND
	;;#ASMSTART
		ds_read2_b32 v[12:13], v10 offset0:160 offset1:176
	;;#ASMEND
	;;#ASMSTART
		s_waitcnt lgkmcnt(1)
	;;#ASMEND
	;;#ASMSTART
		v_pk_mul_f32 v[14:15], v[20:21], v[14:15]
	;;#ASMEND
	;;#ASMSTART
		s_cmp_ge_i32 s11, s10
	;;#ASMEND
	;;#ASMSTART
		v_add_f32_e32 v5, v5, v14
	;;#ASMEND
	;;#ASMSTART
		v_add_f32_e32 v5, v5, v15
	;;#ASMEND
	;;#ASMSTART
		ds_read2_b32 v[14:15], v10 offset0:192 offset1:208
	;;#ASMEND
	;;#ASMSTART
		s_waitcnt lgkmcnt(1)
	;;#ASMEND
	;;#ASMSTART
		v_pk_mul_f32 v[12:13], v[22:23], v[12:13]
	;;#ASMEND
	;;#ASMSTART
		v_add_u32_e32 v6, s6, v6
	;;#ASMEND
	;;#ASMSTART
		v_add_f32_e32 v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
		v_add_f32_e32 v5, v5, v13
	;;#ASMEND
	;;#ASMSTART
		s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
		v_pk_mul_f32 v[12:13], v[24:25], v[14:15]
	;;#ASMEND
	;;#ASMSTART
		s_nop 0
	;;#ASMEND
	;;#ASMSTART
		v_add_f32_e32 v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
		v_add_f32_e32 v5, v5, v13
	;;#ASMEND
	;;#ASMSTART
		v_pk_mul_f32 v[12:13], v[26:27], v[16:17]
	;;#ASMEND
	;;#ASMSTART
		s_barrier
	;;#ASMEND
	;;#ASMSTART
		v_add_f32_e32 v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
		v_add_f32_e32 v5, v5, v13
	;;#ASMEND
	;;#ASMSTART
		s_cbranch_scc1 .LBB0_9
	;;#ASMEND
	;;#ASMSTART
	.LBB0_4:                                ; =>This Inner Loop Header: Depth=1:
	;;#ASMEND
	;;#ASMSTART
		v_add_u32_e32 v7, s11, v2
	;;#ASMEND
	;;#ASMSTART
		v_cmp_gt_i32_e64 s[2:3], s10, v7
	;;#ASMEND
	;;#ASMSTART
		s_and_b64 s[16:17], vcc, s[2:3]
	;;#ASMEND
	;;#ASMSTART
		v_mov_b32_e32 v7, 0
	;;#ASMEND
	;;#ASMSTART
		s_and_saveexec_b64 s[2:3], s[16:17]
	;;#ASMEND
	;;#ASMSTART
		s_cbranch_execz .LBB0_6
	;;#ASMEND
	;;#ASMSTART
		v_add_u32_e32 v12, s11, v4
	;;#ASMEND
	;;#ASMSTART
		v_ashrrev_i32_e32 v13, 31, v12
	;;#ASMEND
	;;#ASMSTART
		v_lshl_add_u64 v[12:13], v[12:13], 2, s[12:13]
	;;#ASMEND
	;;#ASMSTART
		global_load_dword v7, v[12:13], off
	;;#ASMEND
	;;#ASMSTART
	.LBB0_6:                                ;   in Loop: Header=BB0_4 Depth=1:
	;;#ASMEND
	;;#ASMSTART
		s_or_b64 exec, exec, s[2:3]
	;;#ASMEND
	;;#ASMSTART
		v_add_u32_e32 v12, s11, v3
	;;#ASMEND
	;;#ASMSTART
		v_cmp_gt_i32_e64 s[2:3], s10, v12
	;;#ASMEND
	;;#ASMSTART
		s_waitcnt vmcnt(0)
	;;#ASMEND
	;;#ASMSTART
		ds_write_b32 v9, v7
	;;#ASMEND
	;;#ASMSTART
		s_and_b64 s[16:17], s[0:1], s[2:3]
	;;#ASMEND
	;;#ASMSTART
		v_mov_b32_e32 v7, 0
	;;#ASMEND
	;;#ASMSTART
		s_and_saveexec_b64 s[2:3], s[16:17]
	;;#ASMEND
	;;#ASMSTART
		s_cbranch_execz .LBB0_3
	;;#ASMEND
	;;#ASMSTART
		v_ashrrev_i32_e32 v7, 31, v6
	;;#ASMEND
	;;#ASMSTART
		v_lshl_add_u64 v[12:13], v[6:7], 2, s[14:15]
	;;#ASMEND
	;;#ASMSTART
		global_load_dword v7, v[12:13], off
	;;#ASMEND
	;;#ASMSTART
		s_branch .LBB0_3
	;;#ASMEND
	;;#ASMSTART
	.LBB0_8:
	;;#ASMEND
	;;#ASMSTART
		v_mov_b32_e32 v5, 0
	;;#ASMEND
	;;#ASMSTART
	.LBB0_9:                                ; %Flow140:
	;;#ASMEND
	;;#ASMSTART
		v_cmp_gt_i32_e32 vcc, s8, v1
	;;#ASMEND
	;;#ASMSTART
		v_cmp_gt_i32_e64 s[0:1], s9, v0
	;;#ASMEND
	;;#ASMSTART
		s_and_b64 s[0:1], vcc, s[0:1]
	;;#ASMEND
	;;#ASMSTART
		s_and_saveexec_b64 s[2:3], s[0:1]
	;;#ASMEND
	;;#ASMSTART
		s_cbranch_execz .LBB0_11
	;;#ASMEND
	;;#ASMSTART
		s_mul_i32 s0, s22, s5
	;;#ASMEND
	;;#ASMSTART
		s_mul_hi_u32 s1, s22, s4
	;;#ASMEND
	;;#ASMSTART
		s_add_i32 s0, s1, s0
	;;#ASMEND
	;;#ASMSTART
		s_mul_i32 s1, s23, s4
	;;#ASMEND
	;;#ASMSTART
		s_add_i32 s1, s0, s1
	;;#ASMEND
	;;#ASMSTART
		s_mul_i32 s0, s22, s4
	;;#ASMEND
	;;#ASMSTART
		s_lshl_b64 s[0:1], s[0:1], 2
	;;#ASMEND
	;;#ASMSTART
		s_add_u32 s0, s20, s0
	;;#ASMEND
	;;#ASMSTART
		v_mad_u64_u32 v[0:1], s[2:3], s7, v1, v[0:1]
	;;#ASMEND
	;;#ASMSTART
		s_addc_u32 s1, s21, s1
	;;#ASMEND
	;;#ASMSTART
		v_ashrrev_i32_e32 v1, 31, v0
	;;#ASMEND
	;;#ASMSTART
		v_lshl_add_u64 v[0:1], v[0:1], 2, s[0:1]
	;;#ASMEND
	;;#ASMSTART
		global_store_dword v[0:1], v5, off
	;;#ASMEND
	;;#ASMSTART
	.LBB0_11:
	;;#ASMEND
	;;#ASMSTART
		s_endpgm
	;;#ASMEND
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel batched_gemm
		.amdhsa_group_segment_fixed_size 2048
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 76
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 1
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 1
		.amdhsa_next_free_vgpr 34
		.amdhsa_next_free_sgpr 56
		.amdhsa_accum_offset 36
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end0:
	.size	batched_gemm, .Lfunc_end0-batched_gemm

	.set batched_gemm.num_vgpr, 34
	.set batched_gemm.num_agpr, 0
	.set batched_gemm.numbered_sgpr, 56
	.set batched_gemm.num_named_barrier, 0
	.set batched_gemm.private_seg_size, 0
	.set batched_gemm.uses_vcc, 0
	.set batched_gemm.uses_flat_scratch, 0
	.set batched_gemm.has_dyn_sized_stack, 0
	.set batched_gemm.has_recursion, 0
	.set batched_gemm.has_indirect_call, 0
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.text
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
- .agpr_count: 0
  .args:
  - .address_space: global
    .offset: 0
    .size: 8
    .value_kind: global_buffer
  - .address_space: global
    .offset: 8
    .size: 8
    .value_kind: global_buffer
  - .address_space: global
    .offset: 16
    .size: 8
    .value_kind: global_buffer
  - .offset: 24
    .size: 4
    .value_kind: by_value
  - .offset: 28
    .size: 4
    .value_kind: by_value
  - .offset: 32
    .size: 4
    .value_kind: by_value
  - .offset: 36
    .size: 4
    .value_kind: by_value
  - .offset: 40
    .size: 4
    .value_kind: by_value
  - .offset: 44
    .size: 4
    .value_kind: by_value
  - .offset: 48
    .size: 8
    .value_kind: by_value
  - .offset: 56
    .size: 8
    .value_kind: by_value
  - .offset: 64
    .size: 8
    .value_kind: by_value
  - .offset: 72
    .size: 4
    .value_kind: by_value
  .group_segment_fixed_size: 2048
  .kernarg_segment_align: 8
  .kernarg_segment_size: 76
  .max_flat_workgroup_size: 1024
  .name: batched_gemm
  .private_segment_fixed_size: 0
  .sgpr_count: 56
  .sgpr_spill_count: 0
  .symbol: batched_gemm.kd
  .uniform_work_group_size: 1
  .uses_dynamic_stack: false
  .vgpr_count: 34
  .vgpr_spill_count: 0
  .wavefront_size: 64
  .language: OpenCL C
  .language_version:
  - 2
  - 0
amdhsa.target: amdgcn-amd-amdhsa--gfx942
amdhsa.version:
- 1
- 2
...

	.end_amdgpu_metadata
