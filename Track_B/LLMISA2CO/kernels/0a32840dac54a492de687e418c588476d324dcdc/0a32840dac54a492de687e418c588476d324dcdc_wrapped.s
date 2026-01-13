	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.text
	.protected	_Z6matDetPdS_           ; -- Begin function _Z6matDetPdS_
	.globl	_Z6matDetPdS_
	.p2align	8
	.type	_Z6matDetPdS_,@function

_Z6matDetPdS_:                          ; @_Z6matDetPdS_
; %bb.0:
	s_load_dword s0, s[4:5], 0x1c
	s_add_u32 s8, s4, 16
	s_addc_u32 s9, s5, 0
	v_and_b32_e32 v1, 0x3ff, v0
	v_bfe_u32 v0, v0, 10, 10
	s_waitcnt lgkmcnt(0)
	s_and_b32 s7, s0, 0xffff
	v_mad_u32_u24 v16, v0, s7, v1
	v_cmp_gt_u32_e32 vcc, 64, v16
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_5
; %bb.1:
	s_load_dword s10, s[8:9], 0xc
	s_load_dwordx4 s[0:3], s[4:5], 0x0
	s_mul_i32 s4, s6, s7
	s_waitcnt lgkmcnt(0)
	s_lshr_b32 s5, s10, 16
	s_mul_i32 s4, s4, s5
	v_add_u32_e32 v0, s4, v16
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshlrev_b64 v[0:1], 3, v[0:1]
	v_mov_b32_e32 v2, s1
	v_add_co_u32_e32 v0, vcc, s0, v0
	v_addc_co_u32_e32 v1, vcc, v2, v1, vcc
	global_load_dwordx2 v[2:3], v[0:1], off
	v_lshlrev_b32_e32 v0, 3, v16
	v_cmp_gt_u32_e32 vcc, 16, v16
	s_waitcnt vmcnt(0)
	ds_write_b64 v0, v[2:3]
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_3
; %bb.2:
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v3, v2
	ds_write_b64 v0, v[2:3] offset:4096
.LBB0_3:
	s_or_b64 exec, exec, s[0:1]
	v_cmp_gt_u32_e32 vcc, 4, v16
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_5
; %bb.4:
	s_movk_i32 s0, 0x78
	v_mad_u64_u32 v[30:31], s[0:1], v16, s0, v[0:1]
	ds_read_b128 v[4:7], v30 offset:32
	ds_read2_b64 v[0:3], v30 offset0:6 offset1:15
	ds_read2_b64 v[18:21], v30 offset0:9 offset1:10
	ds_read2_b64 v[12:15], v30 offset0:11 offset1:12
	ds_read_b128 v[22:25], v30
	ds_read_b128 v[8:11], v30 offset:16
	ds_read2_b64 v[26:29], v30 offset0:7 offset1:8
	ds_read2_b64 v[30:33], v30 offset0:13 offset1:14
	s_waitcnt lgkmcnt(4)
	v_mul_f64 v[44:45], v[0:1], v[12:13]
	v_mul_f64 v[42:43], v[6:7], v[20:21]
	v_lshlrev_b32_e32 v17, 5, v16
	ds_read_b128 v[34:37], v17 offset:4096
	ds_read_b128 v[38:41], v17 offset:4112
	s_waitcnt lgkmcnt(2)
	v_mul_f64 v[46:47], v[44:45], v[30:31]
	v_fmac_f64_e32 v[46:47], v[42:43], v[2:3]
	v_mul_f64 v[52:53], v[26:27], v[18:19]
	v_mul_f64 v[50:51], v[20:21], v[4:5]
	v_fmac_f64_e32 v[46:47], v[52:53], v[32:33]
	v_mul_f64 v[20:21], v[20:21], -v[26:27]
	v_fmac_f64_e32 v[46:47], v[30:31], v[20:21]
	v_mul_f64 v[54:55], v[6:7], v[12:13]
	v_mul_f64 v[48:49], v[18:19], -v[0:1]
	v_fma_f64 v[46:47], -v[54:55], v[32:33], v[46:47]
	v_mul_f64 v[44:45], v[44:45], v[14:15]
	v_fmac_f64_e32 v[46:47], v[2:3], v[48:49]
	v_mul_f64 v[12:13], v[12:13], -v[4:5]
	v_mul_f64 v[4:5], v[18:19], v[4:5]
	v_mul_f64 v[18:19], v[54:55], v[14:15]
	v_fmac_f64_e32 v[44:45], v[2:3], v[50:51]
	s_waitcnt lgkmcnt(1)
	v_fmac_f64_e32 v[34:35], v[22:23], v[46:47]
	v_mul_f64 v[22:23], v[26:27], v[28:29]
	v_fmac_f64_e32 v[18:19], v[2:3], v[4:5]
	v_fmac_f64_e32 v[44:45], v[32:33], v[22:23]
	v_fmac_f64_e32 v[18:19], v[30:31], v[22:23]
	v_fmac_f64_e32 v[44:45], v[20:21], v[14:15]
	v_fma_f64 v[18:19], -v[52:53], v[14:15], v[18:19]
	v_fmac_f64_e32 v[44:45], v[32:33], v[12:13]
	v_mul_f64 v[0:1], v[0:1], v[28:29]
	v_fmac_f64_e32 v[18:19], v[30:31], v[12:13]
	v_mul_f64 v[6:7], v[28:29], -v[6:7]
	v_fma_f64 v[20:21], -v[2:3], v[0:1], v[44:45]
	v_fmac_f64_e32 v[18:19], v[2:3], v[6:7]
	v_mul_f64 v[2:3], v[42:43], v[14:15]
	v_fmac_f64_e32 v[2:3], v[32:33], v[4:5]
	v_fmac_f64_e32 v[2:3], v[30:31], v[0:1]
	v_fmac_f64_e32 v[2:3], v[48:49], v[14:15]
	v_fma_f64 v[0:1], -v[30:31], v[50:51], v[2:3]
	v_fma_f64 v[36:37], -v[24:25], v[20:21], v[36:37]
	v_fmac_f64_e32 v[0:1], v[32:33], v[6:7]
	v_lshl_or_b32 v2, s6, 2, v16
	v_mov_b32_e32 v3, 0
	s_waitcnt lgkmcnt(0)
	v_fmac_f64_e32 v[38:39], v[8:9], v[18:19]
	v_fma_f64 v[40:41], -v[10:11], v[0:1], v[40:41]
	v_add_f64 v[0:1], v[34:35], v[36:37]
	v_lshlrev_b64 v[2:3], 3, v[2:3]
	v_add_f64 v[0:1], v[38:39], v[0:1]
	v_mov_b32_e32 v4, s3
	v_add_co_u32_e32 v2, vcc, s2, v2
	v_add_f64 v[0:1], v[40:41], v[0:1]
	v_addc_co_u32_e32 v3, vcc, v4, v3, vcc
	ds_write_b128 v17, v[34:37] offset:4096
	ds_write_b128 v17, v[38:41] offset:4112
	global_store_dwordx2 v[2:3], v[0:1], off
.LBB0_5:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z6matDetPdS_
		.amdhsa_group_segment_fixed_size 4224
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 272
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
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 1
		.amdhsa_next_free_vgpr 56
		.amdhsa_next_free_sgpr 11
		.amdhsa_accum_offset 56
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
	.size	_Z6matDetPdS_, .Lfunc_end0-_Z6matDetPdS_
                                        ; -- End function
	.set _Z6matDetPdS_.num_vgpr, 56
	.set _Z6matDetPdS_.num_agpr, 0
	.set _Z6matDetPdS_.numbered_sgpr, 11
	.set _Z6matDetPdS_.private_seg_size, 0
	.set _Z6matDetPdS_.uses_vcc, 1
	.set _Z6matDetPdS_.uses_flat_scratch, 0
	.set _Z6matDetPdS_.has_dyn_sized_stack, 0
	.set _Z6matDetPdS_.has_recursion, 0
	.set _Z6matDetPdS_.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 664
; TotalNumSgprs: 17
; NumVgprs: 56
; NumAgprs: 0
; TotalNumVgprs: 56
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 4224 bytes/workgroup (compile time only)
; SGPRBlocks: 3
; VGPRBlocks: 14
; NumSGPRsForWavesPerEU: 17
; NumVGPRsForWavesPerEU: 56
; AccumOffset: 56
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 1
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 13
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text