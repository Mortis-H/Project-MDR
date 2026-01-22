	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.text
	.protected	matmul_tile             ; -- Begin function matmul_tile
	.globl	matmul_tile
	.p2align	8
	.type	matmul_tile,@function
matmul_tile:                            ; @matmul_tile
; %bb.0:
	s_load_dwordx4 s[8:11], s[0:1], 0x0
	s_load_dwordx2 s[12:13], s[0:1], 0x10
	s_load_dwordx4 s[4:7], s[0:1], 0x18
	v_and_b32_e32 v2, 0x3ff, v0
	v_bfe_u32 v3, v0, 10, 10
	s_lshl_b32 s2, s2, 4
	v_lshl_add_u32 v1, s3, 4, v3
	v_add_u32_e32 v0, s2, v2
	s_waitcnt lgkmcnt(0)
	s_cmp_lt_i32 s6, 1
	v_cmp_gt_i32_e32 vcc, s4, v1
	v_cmp_gt_i32_e64 s[0:1], s5, v0
	s_cbranch_scc1 .LBB0_7
; %bb.1:                                ; %.lr.ph
	v_lshlrev_b32_e32 v4, 2, v2
	v_lshlrev_b32_e32 v8, 6, v3
	v_add_u32_e32 v9, v8, v4
	v_add_u32_e32 v10, 0x400, v4
	v_mad_u64_u32 v[4:5], s[14:15], s6, v1, v[2:3]
	v_mul_lo_u32 v5, v3, s5
	v_add_u32_e32 v11, v10, v8
	v_add3_u32 v6, v2, v5, s2
	s_lshl_b32 s7, s5, 4
	s_mov_b32 s14, 0
	v_mov_b32_e32 v5, 0
	s_branch .LBB0_3
.LBB0_2:                                ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[2:3]
	s_waitcnt vmcnt(0)
	ds_write_b32 v11, v7
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read2_b32 v[28:29], v10 offset1:16
	ds_read_b128 v[12:15], v8
	ds_read_b128 v[16:19], v8 offset:16
	ds_read2_b32 v[30:31], v10 offset0:32 offset1:48
	ds_read_b128 v[20:23], v8 offset:32
	ds_read_b128 v[24:27], v8 offset:48
	ds_read2_b32 v[32:33], v10 offset0:64 offset1:80
	s_waitcnt lgkmcnt(5)
	v_fmac_f32_e32 v5, v12, v28
	v_fmac_f32_e32 v5, v13, v29
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v5, v14, v30
	v_fmac_f32_e32 v5, v15, v31
	ds_read2_b32 v[12:13], v10 offset0:96 offset1:112
	s_waitcnt lgkmcnt(1)
	v_pk_mul_f32 v[14:15], v[16:17], v[32:33]
	ds_read2_b32 v[16:17], v10 offset0:224 offset1:240
	v_add_f32_e32 v5, v5, v14
	v_add_f32_e32 v5, v5, v15
	ds_read2_b32 v[14:15], v10 offset0:128 offset1:144
	s_waitcnt lgkmcnt(2)
	v_pk_mul_f32 v[12:13], v[18:19], v[12:13]
	s_add_i32 s14, s14, 16
	v_add_f32_e32 v5, v5, v12
	v_add_f32_e32 v5, v5, v13
	ds_read2_b32 v[12:13], v10 offset0:160 offset1:176
	s_waitcnt lgkmcnt(1)
	v_pk_mul_f32 v[14:15], v[20:21], v[14:15]
	s_cmp_ge_i32 s14, s6
	v_add_f32_e32 v5, v5, v14
	v_add_f32_e32 v5, v5, v15
	ds_read2_b32 v[14:15], v10 offset0:192 offset1:208
	s_waitcnt lgkmcnt(1)
	v_pk_mul_f32 v[12:13], v[22:23], v[12:13]
; @CAPTURE expr="s7+v6" dst=v6 type=i32
	v_add_f32_e32 v5, v5, v12
	v_add_f32_e32 v5, v5, v13
	s_waitcnt lgkmcnt(0)
	v_pk_mul_f32 v[12:13], v[24:25], v[14:15]
	s_nop 0
	v_add_f32_e32 v5, v5, v12
	v_add_f32_e32 v5, v5, v13
	v_pk_mul_f32 v[12:13], v[26:27], v[16:17]
	s_barrier
	v_add_f32_e32 v5, v5, v12
	v_add_f32_e32 v5, v5, v13
	s_cbranch_scc1 .LBB0_8
.LBB0_3:                                ; =>This Inner Loop Header: Depth=1
; @CAPTURE expr="s14+v2" dst=v7 type=i32
	v_cmp_gt_i32_e64 s[2:3], s6, v7
	s_and_b64 s[16:17], vcc, s[2:3]
	v_mov_b32_e32 v7, 0
	s_and_saveexec_b64 s[2:3], s[16:17]
	s_cbranch_execz .LBB0_5
; %bb.4:                                ;   in Loop: Header=BB0_3 Depth=1
; @CAPTURE expr="s14+v4" dst=v12 type=i32
	v_ashrrev_i32_e32 v13, 31, v12
	v_lshl_add_u64 v[12:13], v[12:13], 2, s[8:9]
	global_load_dword v7, v[12:13], off
.LBB0_5:                                ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[2:3]
; @CAPTURE expr="s14+v3" dst=v12 type=i32
	v_cmp_gt_i32_e64 s[2:3], s6, v12
	s_waitcnt vmcnt(0)
	ds_write_b32 v9, v7
	s_and_b64 s[16:17], s[0:1], s[2:3]
	v_mov_b32_e32 v7, 0
	s_and_saveexec_b64 s[2:3], s[16:17]
	s_cbranch_execz .LBB0_2
; %bb.6:                                ;   in Loop: Header=BB0_3 Depth=1
	v_ashrrev_i32_e32 v7, 31, v6
	v_lshl_add_u64 v[12:13], v[6:7], 2, s[10:11]
	global_load_dword v7, v[12:13], off
	s_branch .LBB0_2
.LBB0_7:
	v_mov_b32_e32 v5, 0
.LBB0_8:                                ; %Flow109
	v_cmp_gt_i32_e32 vcc, s4, v1
	v_cmp_gt_i32_e64 s[0:1], s5, v0
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_cbranch_execz .LBB0_10
; %bb.9:
	v_mad_u64_u32 v[0:1], s[0:1], s5, v1, v[0:1]
	v_mov_b32_e32 v2, s12
	v_mov_b32_e32 v3, s13
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshl_add_u64 v[0:1], v[0:1], 2, v[2:3]
	global_store_dword v[0:1], v5, off
.LBB0_10:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel matmul_tile
		.amdhsa_group_segment_fixed_size 2048
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 36
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
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 1
		.amdhsa_next_free_vgpr 34
		.amdhsa_next_free_sgpr 18
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
	.size	matmul_tile, .Lfunc_end0-matmul_tile
                                        ; -- End function
	.set matmul_tile.num_vgpr, 34
	.set matmul_tile.num_agpr, 0
	.set matmul_tile.numbered_sgpr, 18
	.set matmul_tile.private_seg_size, 0
	.set matmul_tile.uses_vcc, 1
	.set matmul_tile.uses_flat_scratch, 0
	.set matmul_tile.has_dyn_sized_stack, 0
	.set matmul_tile.has_recursion, 0
	.set matmul_tile.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 616
; TotalNumSgprs: 24
; NumVgprs: 34
; NumAgprs: 0
; TotalNumVgprs: 34
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 2048 bytes/workgroup (compile time only)
; SGPRBlocks: 2
; VGPRBlocks: 4
; NumSGPRsForWavesPerEU: 24
; NumVGPRsForWavesPerEU: 34
; AccumOffset: 36
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 1
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 8
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.type	__hip_cuid_77c4299c44ade7ed,@object ; @__hip_cuid_77c4299c44ade7ed
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_77c4299c44ade7ed
__hip_cuid_77c4299c44ade7ed:
	.byte	0                               ; 0x0
	.size	__hip_cuid_77c4299c44ade7ed, 1

	.ident	"AMD clang version 20.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.0.1 25314 f4087f6b428f0e6f575ebac8a8a724dab123d06e)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_77c4299c44ade7ed
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .offset:         24
        .size:           4
        .value_kind:     by_value
      - .offset:         28
        .size:           4
        .value_kind:     by_value
      - .offset:         32
        .size:           4
        .value_kind:     by_value
    .group_segment_fixed_size: 2048
    .kernarg_segment_align: 8
    .kernarg_segment_size: 36
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           matmul_tile
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .sgpr_spill_count: 0
    .symbol:         matmul_tile.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     34
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
