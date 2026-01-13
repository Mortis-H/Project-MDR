	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.text
	.protected	_Z6matDetPdS_           ; -- Begin function _Z6matDetPdS_
	.globl	_Z6matDetPdS_
	.p2align	8
	.type	_Z6matDetPdS_,@function
_Z6matDetPdS_:                          ; @_Z6matDetPdS_
; %bb.0:
	s_load_dword s3, s[0:1], 0x1c
	s_add_u32 s8, s0, 16
	s_addc_u32 s9, s1, 0
	v_and_b32_e32 v1, 0x3ff, v0
	v_bfe_u32 v0, v0, 10, 10
	s_waitcnt lgkmcnt(0)
	s_and_b32 s3, s3, 0xffff
	v_mad_u32_u24 v10, v0, s3, v1
	v_cmp_gt_u32_e32 vcc, 64, v10
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_5
; %bb.1:
	s_load_dwordx4 s[4:7], s[0:1], 0x0
	s_load_dword s10, s[8:9], 0xc
	s_mul_i32 s0, s2, s3
	v_cmp_gt_u32_e32 vcc, 16, v10
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v0, s4
	s_lshr_b32 s1, s10, 16
	s_mul_i32 s0, s0, s1
	v_add_u32_e32 v2, s0, v10
	v_mov_b32_e32 v1, s5
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[0:1], v[2:3], 3, v[0:1]
	global_load_dwordx2 v[2:3], v[0:1], off
	v_lshlrev_b32_e32 v0, 3, v10
	s_waitcnt vmcnt(0)
	ds_write_b64 v0, v[2:3]
	s_and_saveexec_b64 s[0:1], vcc
; %bb.2:
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v3, v2
	ds_write_b64 v0, v[2:3] offset:4096
; %bb.3:
	s_or_b64 exec, exec, s[0:1]
	v_cmp_gt_u32_e32 vcc, 4, v10
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_5
; %bb.4:
	s_movk_i32 s0, 0x78
	v_mad_u64_u32 v[32:33], s[0:1], v10, s0, v[0:1]
	ds_read_b128 v[0:3], v32 offset:32
	ds_read2_b64 v[4:7], v32 offset0:6 offset1:15
	ds_read2_b64 v[12:15], v32 offset0:9 offset1:10
	ds_read2_b64 v[16:19], v32 offset0:11 offset1:12
	ds_read_b128 v[20:23], v32
	ds_read_b128 v[24:27], v32 offset:16
	ds_read2_b64 v[28:31], v32 offset0:7 offset1:8
	ds_read2_b64 v[32:35], v32 offset0:13 offset1:14
	s_waitcnt lgkmcnt(4)
	v_mul_f64 v[42:43], v[4:5], v[16:17]
	v_mul_f64 v[40:41], v[2:3], v[14:15]
	v_lshlrev_b32_e32 v11, 5, v10
	ds_read_b128 v[36:39], v11 offset:4096
	s_waitcnt lgkmcnt(1)
	v_mul_f64 v[44:45], v[42:43], v[32:33]
	v_fmac_f64_e32 v[44:45], v[40:41], v[6:7]
	v_mul_f64 v[52:53], v[28:29], v[12:13]
	v_mul_f64 v[50:51], v[14:15], v[0:1]
	v_fmac_f64_e32 v[44:45], v[52:53], v[34:35]
	v_mul_f64 v[14:15], v[14:15], v[28:29]
	v_mul_f64 v[46:47], v[2:3], v[16:17]
	v_fma_f64 v[44:45], -v[32:33], v[14:15], v[44:45]
	v_mul_f64 v[48:49], v[4:5], v[12:13]
	v_fma_f64 v[44:45], -v[46:47], v[34:35], v[44:45]
	v_mul_f64 v[42:43], v[42:43], v[18:19]
	v_fma_f64 v[44:45], -v[6:7], v[48:49], v[44:45]
	v_fmac_f64_e32 v[42:43], v[6:7], v[50:51]
	s_waitcnt lgkmcnt(0)
	v_fmac_f64_e32 v[36:37], v[20:21], v[44:45]
	v_mul_f64 v[20:21], v[28:29], v[30:31]
	v_fmac_f64_e32 v[42:43], v[34:35], v[20:21]
	v_fma_f64 v[14:15], -v[14:15], v[18:19], v[42:43]
	v_mul_f64 v[28:29], v[16:17], v[0:1]
	v_fma_f64 v[14:15], -v[34:35], v[28:29], v[14:15]
	v_mul_f64 v[4:5], v[4:5], v[30:31]
	v_fma_f64 v[14:15], -v[6:7], v[4:5], v[14:15]
	v_mul_f64 v[0:1], v[12:13], v[0:1]
	v_mul_f64 v[12:13], v[46:47], v[18:19]
	v_fma_f64 v[38:39], -v[22:23], v[14:15], v[38:39]
	ds_read_b128 v[14:17], v11 offset:4112
	v_fmac_f64_e32 v[12:13], v[6:7], v[0:1]
	v_fmac_f64_e32 v[12:13], v[32:33], v[20:21]
	v_fma_f64 v[12:13], -v[52:53], v[18:19], v[12:13]
	v_fma_f64 v[12:13], -v[32:33], v[28:29], v[12:13]
	v_mul_f64 v[2:3], v[2:3], v[30:31]
	v_fma_f64 v[6:7], -v[6:7], v[2:3], v[12:13]
	s_waitcnt lgkmcnt(0)
	v_fmac_f64_e32 v[14:15], v[24:25], v[6:7]
	v_mul_f64 v[6:7], v[40:41], v[18:19]
	v_fmac_f64_e32 v[6:7], v[34:35], v[0:1]
	v_fmac_f64_e32 v[6:7], v[32:33], v[4:5]
	v_fma_f64 v[0:1], -v[48:49], v[18:19], v[6:7]
	v_fma_f64 v[0:1], -v[32:33], v[50:51], v[0:1]
	v_fma_f64 v[0:1], -v[34:35], v[2:3], v[0:1]
	v_fma_f64 v[16:17], -v[26:27], v[0:1], v[16:17]
	v_add_f64 v[0:1], v[36:37], v[38:39]
	v_mov_b32_e32 v8, s6
	v_mov_b32_e32 v9, s7
	v_add_f64 v[0:1], v[14:15], v[0:1]
	v_lshl_or_b32 v2, s2, 2, v10
	v_mov_b32_e32 v3, 0
	v_add_f64 v[0:1], v[16:17], v[0:1]
	v_lshl_add_u64 v[2:3], v[2:3], 3, v[8:9]
	ds_write_b128 v11, v[36:39] offset:4096
	ds_write_b128 v11, v[14:17] offset:4112
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
		.amdhsa_next_free_vgpr 54
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
	.set _Z6matDetPdS_.num_vgpr, 54
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
; NumVgprs: 54
; NumAgprs: 0
; TotalNumVgprs: 54
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 4224 bytes/workgroup (compile time only)
; SGPRBlocks: 2
; VGPRBlocks: 6
; NumSGPRsForWavesPerEU: 17
; NumVGPRsForWavesPerEU: 54
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
	.protected	_Z8vecMult2Pdmi         ; -- Begin function _Z8vecMult2Pdmi
	.globl	_Z8vecMult2Pdmi
	.p2align	8
	.type	_Z8vecMult2Pdmi,@function
_Z8vecMult2Pdmi:                        ; @_Z8vecMult2Pdmi
; %bb.0:
	s_load_dword s33, s[0:1], 0x10
	s_mov_b32 s8, 0
	s_waitcnt lgkmcnt(0)
	s_cmp_eq_u32 s33, 0
	s_cbranch_scc1 .LBB1_369
; %bb.1:                                ; %.lr.ph
	s_load_dword s3, s[0:1], 0x24
	s_load_dwordx4 s[12:15], s[0:1], 0x0
	s_add_u32 s16, s0, 24
	s_addc_u32 s17, s1, 0
	v_mov_b32_e32 v3, 0
	s_waitcnt lgkmcnt(0)
	s_and_b32 s0, s3, 0xffff
	s_mul_i32 s0, s2, s0
	s_getpc_b64 s[4:5]
	s_add_u32 s4, s4, .str@rel32@lo+4
	s_addc_u32 s5, s5, .str@rel32@hi+12
	v_add_u32_e32 v38, s0, v0
	s_cmp_lg_u64 s[4:5], 0
	s_cselect_b64 s[18:19], -1, 0
	v_mov_b32_e32 v2, v38
	s_getpc_b64 s[4:5]
	s_add_u32 s4, s4, .str.1@rel32@lo+4
	s_addc_u32 s5, s5, .str.1@rel32@hi+12
	v_lshlrev_b32_e32 v48, 3, v0
	v_ashrrev_i32_e32 v39, 31, v38
	v_cmp_gt_u32_e64 s[0:1], 16, v0
	v_mov_b64_e32 v[42:43], v[2:3]
	v_lshl_add_u32 v2, s2, 4, v0
	s_cmp_lg_u64 s[4:5], 0
	v_mbcnt_lo_u32_b32 v0, -1, 0
	v_lshl_add_u64 v[40:41], v[38:39], 3, s[12:13]
	v_lshl_add_u64 v[44:45], v[2:3], 3, s[12:13]
	v_cmp_eq_u32_e64 s[2:3], 0, v38
	s_mov_b32 s38, 1
	s_cselect_b64 s[12:13], -1, 0
	s_movk_i32 s39, 0xff1f
	s_movk_i32 s40, 0xff1d
	v_mbcnt_hi_u32_b32 v49, -1, v0
	v_mov_b32_e32 v6, 2
	v_mov_b32_e32 v7, 1
	v_mov_b32_e32 v0, 33
	s_branch .LBB1_3
.LBB1_2:                                ; %Flow646
                                        ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[6:7]
	s_add_i32 s38, s38, 1
	s_cmp_le_u32 s38, s33
	s_cbranch_scc0 .LBB1_369
.LBB1_3:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB1_12 Depth 2
                                        ;     Child Loop BB1_20 Depth 2
                                        ;     Child Loop BB1_29 Depth 2
                                        ;     Child Loop BB1_34 Depth 2
                                        ;     Child Loop BB1_125 Depth 2
                                        ;     Child Loop BB1_133 Depth 2
                                        ;     Child Loop BB1_142 Depth 2
                                        ;     Child Loop BB1_147 Depth 2
                                        ;     Child Loop BB1_38 Depth 2
                                        ;       Child Loop BB1_41 Depth 3
                                        ;       Child Loop BB1_47 Depth 3
                                        ;       Child Loop BB1_56 Depth 3
                                        ;       Child Loop BB1_64 Depth 3
                                        ;       Child Loop BB1_72 Depth 3
                                        ;       Child Loop BB1_80 Depth 3
                                        ;       Child Loop BB1_88 Depth 3
                                        ;       Child Loop BB1_96 Depth 3
                                        ;       Child Loop BB1_104 Depth 3
                                        ;       Child Loop BB1_113 Depth 3
                                        ;       Child Loop BB1_118 Depth 3
                                        ;     Child Loop BB1_152 Depth 2
                                        ;     Child Loop BB1_160 Depth 2
                                        ;     Child Loop BB1_169 Depth 2
                                        ;     Child Loop BB1_174 Depth 2
                                        ;     Child Loop BB1_178 Depth 2
                                        ;     Child Loop BB1_186 Depth 2
                                        ;     Child Loop BB1_195 Depth 2
                                        ;     Child Loop BB1_200 Depth 2
                                        ;     Child Loop BB1_204 Depth 2
                                        ;     Child Loop BB1_212 Depth 2
                                        ;     Child Loop BB1_221 Depth 2
                                        ;     Child Loop BB1_226 Depth 2
                                        ;     Child Loop BB1_234 Depth 2
                                        ;     Child Loop BB1_242 Depth 2
                                        ;     Child Loop BB1_251 Depth 2
                                        ;     Child Loop BB1_256 Depth 2
                                        ;     Child Loop BB1_346 Depth 2
                                        ;     Child Loop BB1_354 Depth 2
                                        ;     Child Loop BB1_363 Depth 2
                                        ;     Child Loop BB1_368 Depth 2
                                        ;     Child Loop BB1_260 Depth 2
                                        ;       Child Loop BB1_263 Depth 3
                                        ;       Child Loop BB1_269 Depth 3
                                        ;       Child Loop BB1_278 Depth 3
                                        ;       Child Loop BB1_286 Depth 3
                                        ;       Child Loop BB1_294 Depth 3
                                        ;       Child Loop BB1_302 Depth 3
                                        ;       Child Loop BB1_310 Depth 3
                                        ;       Child Loop BB1_318 Depth 3
                                        ;       Child Loop BB1_326 Depth 3
                                        ;       Child Loop BB1_335 Depth 3
                                        ;       Child Loop BB1_340 Depth 3
	s_add_i32 s4, s38, -1
	s_lshl_b32 s6, 1, s4
	s_ashr_i32 s7, s6, 31
	s_or_b64 s[4:5], s[14:15], s[6:7]
	s_mov_b32 s9, s5
	s_cmp_lg_u64 s[8:9], 0
	s_cbranch_scc0 .LBB1_119
; %bb.4:                                ;   in Loop: Header=BB1_3 Depth=1
	v_cvt_f32_u32_e32 v1, s6
	v_cvt_f32_u32_e32 v2, s7
	s_sub_u32 s4, 0, s6
	s_subb_u32 s5, 0, s7
	v_fmac_f32_e32 v1, 0x4f800000, v2
	v_rcp_f32_e32 v1, v1
	s_nop 0
	v_mul_f32_e32 v1, 0x5f7ffffc, v1
	v_mul_f32_e32 v2, 0x2f800000, v1
	v_trunc_f32_e32 v2, v2
	v_fmac_f32_e32 v1, 0xcf800000, v2
	v_cvt_u32_f32_e32 v2, v2
	v_cvt_u32_f32_e32 v1, v1
	v_readfirstlane_b32 s9, v2
	v_readfirstlane_b32 s10, v1
	s_mul_i32 s11, s4, s9
	s_mul_hi_u32 s21, s4, s10
	s_mul_i32 s20, s5, s10
	s_add_i32 s11, s21, s11
	s_mul_i32 s22, s4, s10
	s_add_i32 s11, s11, s20
	s_mul_hi_u32 s20, s10, s11
	s_mul_i32 s21, s10, s11
	s_mul_hi_u32 s10, s10, s22
	s_add_u32 s10, s10, s21
	s_addc_u32 s20, 0, s20
	s_mul_hi_u32 s23, s9, s22
	s_mul_i32 s22, s9, s22
	s_add_u32 s10, s10, s22
	s_mul_hi_u32 s21, s9, s11
	s_addc_u32 s10, s20, s23
	s_addc_u32 s20, s21, 0
	s_mul_i32 s11, s9, s11
	s_add_u32 s10, s10, s11
	s_addc_u32 s11, 0, s20
	v_add_co_u32_e32 v1, vcc, s10, v1
	s_cmp_lg_u64 vcc, 0
	s_addc_u32 s9, s9, s11
	v_readfirstlane_b32 s11, v1
	s_mul_i32 s10, s4, s9
	s_mul_hi_u32 s20, s4, s11
	s_add_i32 s10, s20, s10
	s_mul_i32 s5, s5, s11
	s_add_i32 s10, s10, s5
	s_mul_i32 s4, s4, s11
	s_mul_hi_u32 s20, s9, s4
	s_mul_i32 s21, s9, s4
	s_mul_i32 s23, s11, s10
	s_mul_hi_u32 s4, s11, s4
	s_mul_hi_u32 s22, s11, s10
	s_add_u32 s4, s4, s23
	s_addc_u32 s11, 0, s22
	s_add_u32 s4, s4, s21
	s_mul_hi_u32 s5, s9, s10
	s_addc_u32 s4, s11, s20
	s_addc_u32 s5, s5, 0
	s_mul_i32 s10, s9, s10
	s_add_u32 s4, s4, s10
	s_addc_u32 s5, 0, s5
	v_add_co_u32_e32 v1, vcc, s4, v1
	s_cmp_lg_u64 vcc, 0
	s_addc_u32 s4, s9, s5
	v_readfirstlane_b32 s10, v1
	s_mul_i32 s9, s14, s4
	s_mul_hi_u32 s11, s14, s10
	s_mul_hi_u32 s5, s14, s4
	s_add_u32 s9, s11, s9
	s_addc_u32 s5, 0, s5
	s_mul_hi_u32 s20, s15, s10
	s_mul_i32 s10, s15, s10
	s_add_u32 s9, s9, s10
	s_mul_hi_u32 s11, s15, s4
	s_addc_u32 s5, s5, s20
	s_addc_u32 s9, s11, 0
	s_mul_i32 s4, s15, s4
	s_add_u32 s10, s5, s4
	s_addc_u32 s9, 0, s9
	s_mul_i32 s4, s6, s9
	s_mul_hi_u32 s5, s6, s10
	s_add_i32 s4, s5, s4
	s_mul_i32 s5, s7, s10
	s_add_i32 s11, s4, s5
	s_mul_i32 s5, s6, s10
	v_mov_b32_e32 v1, s5
	s_sub_i32 s4, s15, s11
	v_sub_co_u32_e32 v1, vcc, s14, v1
	s_cmp_lg_u64 vcc, 0
	s_subb_u32 s20, s4, s7
	v_subrev_co_u32_e64 v2, s[4:5], s6, v1
	s_cmp_lg_u64 s[4:5], 0
	s_subb_u32 s20, s20, 0
	s_cmp_ge_u32 s20, s7
	s_cselect_b32 s21, -1, 0
	v_cmp_le_u32_e64 s[4:5], s6, v2
	s_cmp_eq_u32 s20, s7
	v_mov_b32_e32 v4, s21
	v_cndmask_b32_e64 v2, 0, -1, s[4:5]
	s_cselect_b64 s[4:5], -1, 0
	v_cndmask_b32_e64 v2, v4, v2, s[4:5]
	s_add_u32 s4, s10, 1
	s_addc_u32 s20, s9, 0
	s_add_u32 s5, s10, 2
	s_addc_u32 s21, s9, 0
	v_mov_b32_e32 v4, s4
	v_mov_b32_e32 v5, s5
	v_cmp_ne_u32_e64 s[4:5], 0, v2
	s_cmp_lg_u64 vcc, 0
	v_cmp_le_u32_e32 vcc, s6, v1
	v_cndmask_b32_e64 v2, v4, v5, s[4:5]
	v_mov_b32_e32 v4, s20
	v_mov_b32_e32 v5, s21
	v_cndmask_b32_e64 v4, v4, v5, s[4:5]
	s_subb_u32 s4, s15, s11
	s_cmp_ge_u32 s4, s7
	s_cselect_b32 s5, -1, 0
	s_cmp_eq_u32 s4, s7
	v_cndmask_b32_e64 v1, 0, -1, vcc
	v_mov_b32_e32 v5, s5
	s_cselect_b64 vcc, -1, 0
	v_cndmask_b32_e32 v1, v5, v1, vcc
	v_mov_b32_e32 v5, s9
	v_cmp_ne_u32_e32 vcc, 0, v1
	v_mov_b32_e32 v1, s10
	s_nop 0
	v_cndmask_b32_e32 v5, v5, v4, vcc
	v_cndmask_b32_e32 v4, v1, v2, vcc
	s_cbranch_execnz .LBB1_6
.LBB1_5:                                ;   in Loop: Header=BB1_3 Depth=1
	v_cvt_f32_u32_e32 v1, s6
	s_sub_i32 s4, 0, s6
	v_rcp_iflag_f32_e32 v1, v1
	s_nop 0
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	s_nop 0
	v_readfirstlane_b32 s5, v1
	s_mul_i32 s4, s4, s5
	s_mul_hi_u32 s4, s5, s4
	s_add_i32 s5, s5, s4
	s_mul_hi_u32 s4, s14, s5
	s_mul_i32 s7, s4, s6
	s_sub_i32 s7, s14, s7
	s_add_i32 s5, s4, 1
	s_sub_i32 s9, s7, s6
	s_cmp_ge_u32 s7, s6
	s_cselect_b32 s4, s5, s4
	s_cselect_b32 s7, s9, s7
	s_add_i32 s5, s4, 1
	s_cmp_ge_u32 s7, s6
	s_cselect_b32 s4, s5, s4
	s_mov_b32 s5, s8
	v_mov_b64_e32 v[4:5], s[4:5]
.LBB1_6:                                ;   in Loop: Header=BB1_3 Depth=1
	v_cmp_gt_u64_e64 s[4:5], v[4:5], v[38:39]
	s_mov_b64 s[6:7], 0
	s_and_saveexec_b64 s[10:11], s[4:5]
	s_cbranch_execz .LBB1_8
; %bb.7:                                ;   in Loop: Header=BB1_3 Depth=1
	global_load_dwordx2 v[4:5], v[40:41], off
	s_and_b64 s[6:7], s[0:1], exec
	s_waitcnt vmcnt(0)
	ds_write_b64 v48, v[4:5]
.LBB1_8:                                ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[10:11]
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_and_saveexec_b64 s[20:21], s[6:7]
	s_cbranch_execz .LBB1_228
; %bb.9:                                ;   in Loop: Header=BB1_3 Depth=1
	s_waitcnt vmcnt(0)
	ds_read2_b64 v[8:11], v48 offset1:16
	s_load_dwordx2 s[22:23], s[16:17], 0x50
	v_readfirstlane_b32 s6, v49
	v_mov_b64_e32 v[4:5], 0
	s_waitcnt lgkmcnt(0)
	v_mov_b64_e32 v[14:15], v[8:9]
	v_cmp_eq_u32_e64 s[6:7], s6, v49
	s_and_saveexec_b64 s[10:11], s[6:7]
	s_cbranch_execz .LBB1_15
; %bb.10:                               ;   in Loop: Header=BB1_3 Depth=1
	global_load_dwordx2 v[18:19], v3, s[22:23] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[4:5], v3, s[22:23] offset:40
	global_load_dwordx2 v[8:9], v3, s[22:23]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v1, v4, v18
	v_and_b32_e32 v2, v5, v19
	v_mul_lo_u32 v2, v2, 24
	v_mul_hi_u32 v4, v1, 24
	v_add_u32_e32 v5, v4, v2
	v_mul_lo_u32 v4, v1, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[4:5], v[8:9], 0, v[4:5]
	global_load_dwordx2 v[16:17], v[4:5], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[4:5], v3, v[16:19], s[22:23] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[4:5], v[18:19]
	s_and_saveexec_b64 s[24:25], vcc
	s_cbranch_execz .LBB1_14
; %bb.11:                               ; %.preheader3.i.i.i.preheader
                                        ;   in Loop: Header=BB1_3 Depth=1
	s_mov_b64 s[26:27], 0
.LBB1_12:                               ; %.preheader3.i.i.i
                                        ;   Parent Loop BB1_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_load_dwordx2 v[8:9], v3, s[22:23] offset:40
	global_load_dwordx2 v[12:13], v3, s[22:23]
	v_mov_b64_e32 v[18:19], v[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v8, v18
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[4:5], s[28:29], v2, 24, v[12:13]
	v_and_b32_e32 v1, v9, v19
	v_mov_b32_e32 v2, v5
	v_mad_u64_u32 v[8:9], s[28:29], v1, 24, v[2:3]
	v_mov_b32_e32 v5, v8
	global_load_dwordx2 v[16:17], v[4:5], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[4:5], v3, v[16:19], s[22:23] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[4:5], v[18:19]
	s_or_b64 s[26:27], vcc, s[26:27]
	s_andn2_b64 exec, exec, s[26:27]
	s_cbranch_execnz .LBB1_12
; %bb.13:                               ; %Flow768
                                        ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[26:27]
.LBB1_14:                               ; %Flow770
                                        ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[24:25]
.LBB1_15:                               ; %.loopexit4.i.i.i
                                        ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[10:11]
	global_load_dwordx2 v[8:9], v3, s[22:23] offset:40
	global_load_dwordx4 v[16:19], v3, s[22:23]
	v_readfirstlane_b32 s24, v4
	v_readfirstlane_b32 s25, v5
	s_mov_b64 s[10:11], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s26, v8
	v_readfirstlane_b32 s27, v9
	s_and_b64 s[26:27], s[24:25], s[26:27]
	s_mul_i32 s9, s27, 24
	s_mul_hi_u32 s28, s26, 24
	s_add_i32 s29, s28, s9
	s_mul_i32 s28, s26, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[8:9], v[16:17], 0, s[28:29]
	s_and_saveexec_b64 s[28:29], s[6:7]
	s_cbranch_execz .LBB1_17
; %bb.16:                               ;   in Loop: Header=BB1_3 Depth=1
	v_mov_b64_e32 v[4:5], s[10:11]
	global_store_dwordx4 v[8:9], v[4:7], off offset:8
.LBB1_17:                               ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[28:29]
	s_lshl_b64 s[10:11], s[26:27], 12
	v_lshl_add_u64 v[4:5], v[18:19], 0, s[10:11]
	s_mov_b32 s10, s8
	s_mov_b32 s11, s8
	s_mov_b32 s9, s8
	v_mov_b64_e32 v[20:21], s[10:11]
	v_lshlrev_b32_e32 v46, 6, v49
	v_mov_b32_e32 v1, v3
	v_mov_b32_e32 v2, v3
	v_readfirstlane_b32 s26, v4
	v_readfirstlane_b32 s27, v5
	v_mov_b64_e32 v[18:19], s[8:9]
	s_nop 3
	global_store_dwordx4 v46, v[0:3], s[26:27]
	global_store_dwordx4 v46, v[18:21], s[26:27] offset:16
	global_store_dwordx4 v46, v[18:21], s[26:27] offset:32
	global_store_dwordx4 v46, v[18:21], s[26:27] offset:48
	s_and_saveexec_b64 s[10:11], s[6:7]
	s_cbranch_execz .LBB1_25
; %bb.18:                               ;   in Loop: Header=BB1_3 Depth=1
	global_load_dwordx2 v[20:21], v3, s[22:23] offset:32 sc0 sc1
	global_load_dwordx2 v[12:13], v3, s[22:23] offset:40
	v_mov_b32_e32 v18, s24
	v_mov_b32_e32 v19, s25
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v1, s24, v12
	v_and_b32_e32 v2, s25, v13
	v_mul_lo_u32 v2, v2, 24
	v_mul_hi_u32 v13, v1, 24
	v_mul_lo_u32 v12, v1, 24
	v_add_u32_e32 v13, v13, v2
	v_lshl_add_u64 v[12:13], v[16:17], 0, v[12:13]
	global_store_dwordx2 v[12:13], v[20:21], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[18:19], v3, v[18:21], s[22:23] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[18:19], v[20:21]
	s_and_saveexec_b64 s[26:27], vcc
	s_cbranch_execz .LBB1_21
; %bb.19:                               ; %.preheader1.i.i.i.preheader
                                        ;   in Loop: Header=BB1_3 Depth=1
	s_mov_b64 s[28:29], 0
.LBB1_20:                               ; %.preheader1.i.i.i
                                        ;   Parent Loop BB1_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[12:13], v[18:19], off
	v_mov_b32_e32 v16, s24
	v_mov_b32_e32 v17, s25
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[16:17], v3, v[16:19], s[22:23] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[16:17], v[18:19]
	s_or_b64 s[28:29], vcc, s[28:29]
	v_mov_b64_e32 v[18:19], v[16:17]
	s_andn2_b64 exec, exec, s[28:29]
	s_cbranch_execnz .LBB1_20
.LBB1_21:                               ; %Flow766
                                        ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[26:27]
	global_load_dwordx2 v[12:13], v3, s[22:23] offset:16
	s_mov_b64 s[28:29], exec
	v_mbcnt_lo_u32_b32 v1, s28, 0
	v_mbcnt_hi_u32_b32 v1, s29, v1
	v_cmp_eq_u32_e32 vcc, 0, v1
	s_and_saveexec_b64 s[26:27], vcc
	s_cbranch_execz .LBB1_23
; %bb.22:                               ;   in Loop: Header=BB1_3 Depth=1
	s_bcnt1_i32_b64 s9, s[28:29]
	v_mov_b32_e32 v2, s9
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[12:13], v[2:3], off offset:8 sc1
.LBB1_23:                               ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[26:27]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[16:17], v[12:13], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[16:17]
	s_cbranch_vccnz .LBB1_25
; %bb.24:                               ;   in Loop: Header=BB1_3 Depth=1
	global_load_dword v2, v[12:13], off offset:24
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v1, 0xffffff, v2
	s_nop 0
	v_readfirstlane_b32 s9, v1
	s_mov_b32 m0, s9
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[16:17], v[2:3], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB1_25:                               ; %Flow767
                                        ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[10:11]
	v_mov_b32_e32 v47, v3
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[46:47]
	s_branch .LBB1_29
.LBB1_26:                               ;   in Loop: Header=BB1_29 Depth=2
	s_or_b64 exec, exec, s[10:11]
	v_readfirstlane_b32 s9, v1
	s_cmp_eq_u32 s9, 0
	s_cbranch_scc1 .LBB1_28
; %bb.27:                               ;   in Loop: Header=BB1_29 Depth=2
	s_sleep 1
	s_cbranch_execnz .LBB1_29
	s_branch .LBB1_31
.LBB1_28:                               ;   in Loop: Header=BB1_3 Depth=1
	s_branch .LBB1_31
.LBB1_29:                               ;   Parent Loop BB1_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v1, 1
	s_and_saveexec_b64 s[10:11], s[6:7]
	s_cbranch_execz .LBB1_26
; %bb.30:                               ;   in Loop: Header=BB1_29 Depth=2
	global_load_dword v1, v[8:9], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v1, 1, v1
	s_branch .LBB1_26
.LBB1_31:                               ;   in Loop: Header=BB1_3 Depth=1
	global_load_dwordx2 v[16:17], v[4:5], off
	s_and_saveexec_b64 s[10:11], s[6:7]
	s_cbranch_execz .LBB1_35
; %bb.32:                               ;   in Loop: Header=BB1_3 Depth=1
	global_load_dwordx2 v[4:5], v3, s[22:23] offset:40
	global_load_dwordx2 v[8:9], v3, s[22:23] offset:24 sc0 sc1
	global_load_dwordx2 v[12:13], v3, s[22:23]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[18:19], v[4:5], 0, 1
	v_lshl_add_u64 v[22:23], v[18:19], 0, s[24:25]
	v_cmp_eq_u64_e32 vcc, 0, v[22:23]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v20, v8
	v_mov_b32_e32 v21, v9
	v_cndmask_b32_e32 v19, v23, v19, vcc
	v_cndmask_b32_e32 v18, v22, v18, vcc
	v_and_b32_e32 v1, v19, v5
	v_and_b32_e32 v2, v18, v4
	v_mul_lo_u32 v1, v1, 24
	v_mul_hi_u32 v5, v2, 24
	v_mul_lo_u32 v4, v2, 24
	v_add_u32_e32 v5, v5, v1
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[4:5], v[12:13], 0, v[4:5]
	global_store_dwordx2 v[4:5], v[8:9], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[20:21], v3, v[18:21], s[22:23] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[20:21], v[8:9]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB1_35
; %bb.33:                               ; %.preheader.i.i.i.preheader
                                        ;   in Loop: Header=BB1_3 Depth=1
	s_mov_b64 s[6:7], 0
.LBB1_34:                               ; %.preheader.i.i.i
                                        ;   Parent Loop BB1_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[4:5], v[20:21], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[8:9], v3, v[18:21], s[22:23] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[8:9], v[20:21]
	s_or_b64 s[6:7], vcc, s[6:7]
	v_mov_b64_e32 v[20:21], v[8:9]
	s_andn2_b64 exec, exec, s[6:7]
	s_cbranch_execnz .LBB1_34
.LBB1_35:                               ; %__ockl_printf_begin.exit
                                        ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[10:11]
	s_and_b64 vcc, exec, s[18:19]
	s_cbranch_vccz .LBB1_121
; %bb.36:                               ;   in Loop: Header=BB1_3 Depth=1
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v8, 2, v16
	v_and_b32_e32 v18, -3, v16
	v_mov_b32_e32 v19, v17
	s_mov_b64 s[24:25], 27
	s_getpc_b64 s[10:11]
	s_add_u32 s10, s10, .str@rel32@lo+4
	s_addc_u32 s11, s11, .str@rel32@hi+12
	s_branch .LBB1_38
.LBB1_37:                               ; %__ockl_hostcall_preview.exit19.i
                                        ;   in Loop: Header=BB1_38 Depth=2
	s_or_b64 exec, exec, s[30:31]
	s_sub_u32 s24, s24, s26
	s_subb_u32 s25, s25, s27
	s_add_u32 s10, s10, s26
	s_addc_u32 s11, s11, s27
	s_cmp_lg_u64 s[24:25], 0
	s_cbranch_scc0 .LBB1_120
.LBB1_38:                               ;   Parent Loop BB1_3 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB1_41 Depth 3
                                        ;       Child Loop BB1_47 Depth 3
                                        ;       Child Loop BB1_56 Depth 3
                                        ;       Child Loop BB1_64 Depth 3
                                        ;       Child Loop BB1_72 Depth 3
                                        ;       Child Loop BB1_80 Depth 3
                                        ;       Child Loop BB1_88 Depth 3
                                        ;       Child Loop BB1_96 Depth 3
                                        ;       Child Loop BB1_104 Depth 3
                                        ;       Child Loop BB1_113 Depth 3
                                        ;       Child Loop BB1_118 Depth 3
	v_cmp_lt_u64_e64 s[6:7], s[24:25], 56
	s_and_b64 s[6:7], s[6:7], exec
	v_cmp_gt_u64_e64 s[6:7], s[24:25], 7
	s_cselect_b32 s27, s25, 0
	s_cselect_b32 s26, s24, 56
	s_and_b64 vcc, exec, s[6:7]
	s_cbranch_vccnz .LBB1_48
; %bb.39:                               ;   in Loop: Header=BB1_38 Depth=2
	s_mov_b64 s[6:7], 0
	s_cmp_eq_u64 s[24:25], 0
	v_mov_b64_e32 v[20:21], 0
	s_cbranch_scc1 .LBB1_42
; %bb.40:                               ; %.preheader30.i.preheader
                                        ;   in Loop: Header=BB1_38 Depth=2
	s_lshl_b64 s[28:29], s[26:27], 3
	s_mov_b64 s[30:31], 0
	v_mov_b64_e32 v[20:21], 0
	s_mov_b64 s[34:35], s[10:11]
.LBB1_41:                               ; %.preheader30.i
                                        ;   Parent Loop BB1_3 Depth=1
                                        ;     Parent Loop BB1_38 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	global_load_ubyte v1, v3, s[34:35]
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v2, 0xffff, v1
	v_lshlrev_b64 v[4:5], s30, v[2:3]
	s_add_u32 s30, s30, 8
	s_addc_u32 s31, s31, 0
	s_add_u32 s34, s34, 1
	s_addc_u32 s35, s35, 0
	v_or_b32_e32 v20, v4, v20
	s_cmp_lg_u32 s28, s30
	v_or_b32_e32 v21, v5, v21
	s_cbranch_scc1 .LBB1_41
.LBB1_42:                               ; %Flow737
                                        ;   in Loop: Header=BB1_38 Depth=2
	s_mov_b32 s9, 0
	s_andn2_b64 vcc, exec, s[6:7]
	s_mov_b64 s[6:7], s[10:11]
	s_cbranch_vccnz .LBB1_44
.LBB1_43:                               ;   in Loop: Header=BB1_38 Depth=2
	global_load_dwordx2 v[20:21], v3, s[10:11]
	s_add_i32 s9, s26, -8
	s_add_u32 s6, s10, 8
	s_addc_u32 s7, s11, 0
.LBB1_44:                               ; %.loopexit31.i
                                        ;   in Loop: Header=BB1_38 Depth=2
	s_cmp_gt_u32 s9, 7
	s_cbranch_scc1 .LBB1_49
; %bb.45:                               ;   in Loop: Header=BB1_38 Depth=2
	s_cmp_eq_u32 s9, 0
	s_cbranch_scc1 .LBB1_50
; %bb.46:                               ; %.preheader28.i.preheader
                                        ;   in Loop: Header=BB1_38 Depth=2
	s_mov_b64 s[28:29], 0
	v_mov_b64_e32 v[22:23], 0
	s_mov_b64 s[30:31], 0
.LBB1_47:                               ; %.preheader28.i
                                        ;   Parent Loop BB1_3 Depth=1
                                        ;     Parent Loop BB1_38 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_add_u32 s34, s6, s30
	s_addc_u32 s35, s7, s31
	global_load_ubyte v1, v3, s[34:35]
	s_add_u32 s30, s30, 1
	s_addc_u32 s31, s31, 0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v2, 0xffff, v1
	v_lshlrev_b64 v[4:5], s28, v[2:3]
	s_add_u32 s28, s28, 8
	s_addc_u32 s29, s29, 0
	v_or_b32_e32 v22, v4, v22
	s_cmp_lg_u32 s9, s30
	v_or_b32_e32 v23, v5, v23
	s_cbranch_scc1 .LBB1_47
	s_branch .LBB1_51
.LBB1_48:                               ;   in Loop: Header=BB1_38 Depth=2
	s_mov_b32 s9, 0
	s_mov_b64 s[6:7], s[10:11]
	s_branch .LBB1_43
.LBB1_49:                               ;   in Loop: Header=BB1_38 Depth=2
                                        ; implicit-def: $vgpr22_vgpr23
	s_mov_b32 s34, 0
	s_branch .LBB1_52
.LBB1_50:                               ;   in Loop: Header=BB1_38 Depth=2
	v_mov_b64_e32 v[22:23], 0
.LBB1_51:                               ; %Flow730
                                        ;   in Loop: Header=BB1_38 Depth=2
	s_mov_b32 s34, 0
	s_cbranch_execnz .LBB1_53
.LBB1_52:                               ;   in Loop: Header=BB1_38 Depth=2
	global_load_dwordx2 v[22:23], v3, s[6:7]
	s_add_i32 s34, s9, -8
	s_add_u32 s6, s6, 8
	s_addc_u32 s7, s7, 0
.LBB1_53:                               ; %.loopexit29.i
                                        ;   in Loop: Header=BB1_38 Depth=2
	s_cmp_gt_u32 s34, 7
	s_cbranch_scc1 .LBB1_57
; %bb.54:                               ;   in Loop: Header=BB1_38 Depth=2
	s_cmp_eq_u32 s34, 0
	s_cbranch_scc1 .LBB1_58
; %bb.55:                               ; %.preheader26.i.preheader
                                        ;   in Loop: Header=BB1_38 Depth=2
	s_mov_b64 s[28:29], 0
	v_mov_b64_e32 v[24:25], 0
	s_mov_b64 s[30:31], 0
.LBB1_56:                               ; %.preheader26.i
                                        ;   Parent Loop BB1_3 Depth=1
                                        ;     Parent Loop BB1_38 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_add_u32 s36, s6, s30
	s_addc_u32 s37, s7, s31
	global_load_ubyte v1, v3, s[36:37]
	s_add_u32 s30, s30, 1
	s_addc_u32 s31, s31, 0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v2, 0xffff, v1
	v_lshlrev_b64 v[4:5], s28, v[2:3]
	s_add_u32 s28, s28, 8
	s_addc_u32 s29, s29, 0
	v_or_b32_e32 v24, v4, v24
	s_cmp_lg_u32 s34, s30
	v_or_b32_e32 v25, v5, v25
	s_cbranch_scc1 .LBB1_56
	s_branch .LBB1_59
.LBB1_57:                               ;   in Loop: Header=BB1_38 Depth=2
	s_mov_b32 s9, 0
	s_branch .LBB1_60
.LBB1_58:                               ;   in Loop: Header=BB1_38 Depth=2
	v_mov_b64_e32 v[24:25], 0
.LBB1_59:                               ; %Flow725
                                        ;   in Loop: Header=BB1_38 Depth=2
	s_mov_b32 s9, 0
	s_cbranch_execnz .LBB1_61
.LBB1_60:                               ;   in Loop: Header=BB1_38 Depth=2
	global_load_dwordx2 v[24:25], v3, s[6:7]
	s_add_i32 s9, s34, -8
	s_add_u32 s6, s6, 8
	s_addc_u32 s7, s7, 0
.LBB1_61:                               ; %.loopexit27.i
                                        ;   in Loop: Header=BB1_38 Depth=2
	s_cmp_gt_u32 s9, 7
	s_cbranch_scc1 .LBB1_65
; %bb.62:                               ;   in Loop: Header=BB1_38 Depth=2
	s_cmp_eq_u32 s9, 0
	s_cbranch_scc1 .LBB1_66
; %bb.63:                               ; %.preheader24.i.preheader
                                        ;   in Loop: Header=BB1_38 Depth=2
	s_mov_b64 s[28:29], 0
	v_mov_b64_e32 v[26:27], 0
	s_mov_b64 s[30:31], 0
.LBB1_64:                               ; %.preheader24.i
                                        ;   Parent Loop BB1_3 Depth=1
                                        ;     Parent Loop BB1_38 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_add_u32 s34, s6, s30
	s_addc_u32 s35, s7, s31
	global_load_ubyte v1, v3, s[34:35]
	s_add_u32 s30, s30, 1
	s_addc_u32 s31, s31, 0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v2, 0xffff, v1
	v_lshlrev_b64 v[4:5], s28, v[2:3]
	s_add_u32 s28, s28, 8
	s_addc_u32 s29, s29, 0
	v_or_b32_e32 v26, v4, v26
	s_cmp_lg_u32 s9, s30
	v_or_b32_e32 v27, v5, v27
	s_cbranch_scc1 .LBB1_64
	s_branch .LBB1_67
.LBB1_65:                               ;   in Loop: Header=BB1_38 Depth=2
                                        ; implicit-def: $vgpr26_vgpr27
	s_mov_b32 s34, 0
	s_branch .LBB1_68
.LBB1_66:                               ;   in Loop: Header=BB1_38 Depth=2
	v_mov_b64_e32 v[26:27], 0
.LBB1_67:                               ; %Flow720
                                        ;   in Loop: Header=BB1_38 Depth=2
	s_mov_b32 s34, 0
	s_cbranch_execnz .LBB1_69
.LBB1_68:                               ;   in Loop: Header=BB1_38 Depth=2
	global_load_dwordx2 v[26:27], v3, s[6:7]
	s_add_i32 s34, s9, -8
	s_add_u32 s6, s6, 8
	s_addc_u32 s7, s7, 0
.LBB1_69:                               ; %.loopexit25.i
                                        ;   in Loop: Header=BB1_38 Depth=2
	s_cmp_gt_u32 s34, 7
	s_cbranch_scc1 .LBB1_73
; %bb.70:                               ;   in Loop: Header=BB1_38 Depth=2
	s_cmp_eq_u32 s34, 0
	s_cbranch_scc1 .LBB1_74
; %bb.71:                               ; %.preheader22.i.preheader
                                        ;   in Loop: Header=BB1_38 Depth=2
	s_mov_b64 s[28:29], 0
	v_mov_b64_e32 v[28:29], 0
	s_mov_b64 s[30:31], 0
.LBB1_72:                               ; %.preheader22.i
                                        ;   Parent Loop BB1_3 Depth=1
                                        ;     Parent Loop BB1_38 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_add_u32 s36, s6, s30
	s_addc_u32 s37, s7, s31
	global_load_ubyte v1, v3, s[36:37]
	s_add_u32 s30, s30, 1
	s_addc_u32 s31, s31, 0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v2, 0xffff, v1
	v_lshlrev_b64 v[4:5], s28, v[2:3]
	s_add_u32 s28, s28, 8
	s_addc_u32 s29, s29, 0
	v_or_b32_e32 v28, v4, v28
	s_cmp_lg_u32 s34, s30
	v_or_b32_e32 v29, v5, v29
	s_cbranch_scc1 .LBB1_72
	s_branch .LBB1_75
.LBB1_73:                               ;   in Loop: Header=BB1_38 Depth=2
	s_mov_b32 s9, 0
	s_branch .LBB1_76
.LBB1_74:                               ;   in Loop: Header=BB1_38 Depth=2
	v_mov_b64_e32 v[28:29], 0
.LBB1_75:                               ; %Flow715
                                        ;   in Loop: Header=BB1_38 Depth=2
	s_mov_b32 s9, 0
	s_cbranch_execnz .LBB1_77
.LBB1_76:                               ;   in Loop: Header=BB1_38 Depth=2
	global_load_dwordx2 v[28:29], v3, s[6:7]
	s_add_i32 s9, s34, -8
	s_add_u32 s6, s6, 8
	s_addc_u32 s7, s7, 0
.LBB1_77:                               ; %.loopexit23.i
                                        ;   in Loop: Header=BB1_38 Depth=2
	s_cmp_gt_u32 s9, 7
	s_cbranch_scc1 .LBB1_81
; %bb.78:                               ;   in Loop: Header=BB1_38 Depth=2
	s_cmp_eq_u32 s9, 0
	s_cbranch_scc1 .LBB1_82
; %bb.79:                               ; %.preheader20.i.preheader
                                        ;   in Loop: Header=BB1_38 Depth=2
	s_mov_b64 s[28:29], 0
	v_mov_b64_e32 v[30:31], 0
	s_mov_b64 s[30:31], 0
.LBB1_80:                               ; %.preheader20.i
                                        ;   Parent Loop BB1_3 Depth=1
                                        ;     Parent Loop BB1_38 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_add_u32 s34, s6, s30
	s_addc_u32 s35, s7, s31
	global_load_ubyte v1, v3, s[34:35]
	s_add_u32 s30, s30, 1
	s_addc_u32 s31, s31, 0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v2, 0xffff, v1
	v_lshlrev_b64 v[4:5], s28, v[2:3]
	s_add_u32 s28, s28, 8
	s_addc_u32 s29, s29, 0
	v_or_b32_e32 v30, v4, v30
	s_cmp_lg_u32 s9, s30
	v_or_b32_e32 v31, v5, v31
	s_cbranch_scc1 .LBB1_80
	s_branch .LBB1_83
.LBB1_81:                               ;   in Loop: Header=BB1_38 Depth=2
                                        ; implicit-def: $vgpr30_vgpr31
	s_mov_b32 s34, 0
	s_branch .LBB1_84
.LBB1_82:                               ;   in Loop: Header=BB1_38 Depth=2
	v_mov_b64_e32 v[30:31], 0
.LBB1_83:                               ; %Flow710
                                        ;   in Loop: Header=BB1_38 Depth=2
	s_mov_b32 s34, 0
	s_cbranch_execnz .LBB1_85
.LBB1_84:                               ;   in Loop: Header=BB1_38 Depth=2
	global_load_dwordx2 v[30:31], v3, s[6:7]
	s_add_i32 s34, s9, -8
	s_add_u32 s6, s6, 8
	s_addc_u32 s7, s7, 0
.LBB1_85:                               ; %.loopexit21.i
                                        ;   in Loop: Header=BB1_38 Depth=2
	s_cmp_gt_u32 s34, 7
	s_cbranch_scc1 .LBB1_89
; %bb.86:                               ;   in Loop: Header=BB1_38 Depth=2
	s_cmp_eq_u32 s34, 0
	s_cbranch_scc1 .LBB1_90
; %bb.87:                               ; %.preheader.i.preheader
                                        ;   in Loop: Header=BB1_38 Depth=2
	s_mov_b64 s[28:29], 0
	v_mov_b64_e32 v[32:33], 0
	s_mov_b64 s[30:31], s[6:7]
.LBB1_88:                               ; %.preheader.i
                                        ;   Parent Loop BB1_3 Depth=1
                                        ;     Parent Loop BB1_38 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	global_load_ubyte v1, v3, s[30:31]
	s_add_i32 s34, s34, -1
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v2, 0xffff, v1
	v_lshlrev_b64 v[4:5], s28, v[2:3]
	s_add_u32 s28, s28, 8
	s_addc_u32 s29, s29, 0
	s_add_u32 s30, s30, 1
	s_addc_u32 s31, s31, 0
	v_or_b32_e32 v32, v4, v32
	s_cmp_lg_u32 s34, 0
	v_or_b32_e32 v33, v5, v33
	s_cbranch_scc1 .LBB1_88
	s_branch .LBB1_91
.LBB1_89:                               ;   in Loop: Header=BB1_38 Depth=2
	s_branch .LBB1_92
.LBB1_90:                               ;   in Loop: Header=BB1_38 Depth=2
	v_mov_b64_e32 v[32:33], 0
.LBB1_91:                               ; %Flow705
                                        ;   in Loop: Header=BB1_38 Depth=2
	s_cbranch_execnz .LBB1_93
.LBB1_92:                               ;   in Loop: Header=BB1_38 Depth=2
	global_load_dwordx2 v[32:33], v3, s[6:7]
.LBB1_93:                               ; %.loopexit.i
                                        ;   in Loop: Header=BB1_38 Depth=2
	v_readfirstlane_b32 s6, v49
	v_mov_b64_e32 v[4:5], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[6:7], s6, v49
	s_and_saveexec_b64 s[28:29], s[6:7]
	s_cbranch_execz .LBB1_99
; %bb.94:                               ;   in Loop: Header=BB1_38 Depth=2
	global_load_dwordx2 v[36:37], v3, s[22:23] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[4:5], v3, s[22:23] offset:40
	global_load_dwordx2 v[12:13], v3, s[22:23]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v1, v4, v36
	v_and_b32_e32 v2, v5, v37
	v_mul_lo_u32 v2, v2, 24
	v_mul_hi_u32 v4, v1, 24
	v_add_u32_e32 v5, v4, v2
	v_mul_lo_u32 v4, v1, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[4:5], v[12:13], 0, v[4:5]
	global_load_dwordx2 v[34:35], v[4:5], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[4:5], v3, v[34:37], s[22:23] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[4:5], v[36:37]
	s_and_saveexec_b64 s[30:31], vcc
	s_cbranch_execz .LBB1_98
; %bb.95:                               ; %.preheader3.i.i18.i.preheader
                                        ;   in Loop: Header=BB1_38 Depth=2
	s_mov_b64 s[34:35], 0
.LBB1_96:                               ; %.preheader3.i.i18.i
                                        ;   Parent Loop BB1_3 Depth=1
                                        ;     Parent Loop BB1_38 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_sleep 1
	global_load_dwordx2 v[12:13], v3, s[22:23] offset:40
	global_load_dwordx2 v[34:35], v3, s[22:23]
	v_mov_b64_e32 v[36:37], v[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v12, v36
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[4:5], s[36:37], v2, 24, v[34:35]
	v_and_b32_e32 v1, v13, v37
	v_mov_b32_e32 v2, v5
	v_mad_u64_u32 v[12:13], s[36:37], v1, 24, v[2:3]
	v_mov_b32_e32 v5, v12
	global_load_dwordx2 v[34:35], v[4:5], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[4:5], v3, v[34:37], s[22:23] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[4:5], v[36:37]
	s_or_b64 s[34:35], vcc, s[34:35]
	s_andn2_b64 exec, exec, s[34:35]
	s_cbranch_execnz .LBB1_96
; %bb.97:                               ; %Flow700
                                        ;   in Loop: Header=BB1_38 Depth=2
	s_or_b64 exec, exec, s[34:35]
.LBB1_98:                               ; %Flow702
                                        ;   in Loop: Header=BB1_38 Depth=2
	s_or_b64 exec, exec, s[30:31]
.LBB1_99:                               ; %.loopexit4.i.i13.i
                                        ;   in Loop: Header=BB1_38 Depth=2
	s_or_b64 exec, exec, s[28:29]
	global_load_dwordx2 v[12:13], v3, s[22:23] offset:40
	global_load_dwordx4 v[34:37], v3, s[22:23]
	v_readfirstlane_b32 s28, v4
	v_readfirstlane_b32 s29, v5
	s_mov_b64 s[30:31], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s34, v12
	v_readfirstlane_b32 s35, v13
	s_and_b64 s[34:35], s[28:29], s[34:35]
	s_mul_i32 s9, s35, 24
	s_mul_hi_u32 s36, s34, 24
	s_add_i32 s37, s36, s9
	s_mul_i32 s36, s34, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[12:13], v[34:35], 0, s[36:37]
	s_and_saveexec_b64 s[36:37], s[6:7]
	s_cbranch_execz .LBB1_101
; %bb.100:                              ;   in Loop: Header=BB1_38 Depth=2
	v_mov_b64_e32 v[4:5], s[30:31]
	global_store_dwordx4 v[12:13], v[4:7], off offset:8
.LBB1_101:                              ;   in Loop: Header=BB1_38 Depth=2
	s_or_b64 exec, exec, s[36:37]
	v_cmp_lt_u64_e64 vcc, s[24:25], 57
	s_lshl_b32 s9, s26, 2
	s_lshl_b64 s[30:31], s[34:35], 12
	v_cndmask_b32_e32 v1, 0, v8, vcc
	s_add_i32 s9, s9, 28
	v_and_b32_e32 v2, 0xffffff1f, v18
	v_lshl_add_u64 v[4:5], v[36:37], 0, s[30:31]
	s_and_b32 s9, s9, 0x1e0
	v_or_b32_e32 v1, v2, v1
	v_or_b32_e32 v18, s9, v1
	v_readfirstlane_b32 s30, v4
	v_readfirstlane_b32 s31, v5
	s_nop 4
	global_store_dwordx4 v46, v[18:21], s[30:31]
	global_store_dwordx4 v46, v[22:25], s[30:31] offset:16
	global_store_dwordx4 v46, v[26:29], s[30:31] offset:32
	global_store_dwordx4 v46, v[30:33], s[30:31] offset:48
	s_and_saveexec_b64 s[30:31], s[6:7]
	s_cbranch_execz .LBB1_109
; %bb.102:                              ;   in Loop: Header=BB1_38 Depth=2
	global_load_dwordx2 v[26:27], v3, s[22:23] offset:32 sc0 sc1
	global_load_dwordx2 v[18:19], v3, s[22:23] offset:40
	v_mov_b32_e32 v24, s28
	v_mov_b32_e32 v25, s29
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s34, v18
	v_readfirstlane_b32 s35, v19
	s_and_b64 s[34:35], s[34:35], s[28:29]
	s_mul_i32 s9, s35, 24
	s_mul_hi_u32 s35, s34, 24
	s_mul_i32 s34, s34, 24
	s_add_i32 s35, s35, s9
	v_lshl_add_u64 v[22:23], v[34:35], 0, s[34:35]
	global_store_dwordx2 v[22:23], v[26:27], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[20:21], v3, v[24:27], s[22:23] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[20:21], v[26:27]
	s_and_saveexec_b64 s[34:35], vcc
	s_cbranch_execz .LBB1_105
; %bb.103:                              ; %.preheader1.i.i16.i.preheader
                                        ;   in Loop: Header=BB1_38 Depth=2
	s_mov_b64 s[36:37], 0
.LBB1_104:                              ; %.preheader1.i.i16.i
                                        ;   Parent Loop BB1_3 Depth=1
                                        ;     Parent Loop BB1_38 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_sleep 1
	global_store_dwordx2 v[22:23], v[20:21], off
	v_mov_b32_e32 v18, s28
	v_mov_b32_e32 v19, s29
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[18:19], v3, v[18:21], s[22:23] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[18:19], v[20:21]
	s_or_b64 s[36:37], vcc, s[36:37]
	v_mov_b64_e32 v[20:21], v[18:19]
	s_andn2_b64 exec, exec, s[36:37]
	s_cbranch_execnz .LBB1_104
.LBB1_105:                              ; %Flow698
                                        ;   in Loop: Header=BB1_38 Depth=2
	s_or_b64 exec, exec, s[34:35]
	global_load_dwordx2 v[18:19], v3, s[22:23] offset:16
	s_mov_b64 s[36:37], exec
	v_mbcnt_lo_u32_b32 v1, s36, 0
	v_mbcnt_hi_u32_b32 v1, s37, v1
	v_cmp_eq_u32_e32 vcc, 0, v1
	s_and_saveexec_b64 s[34:35], vcc
	s_cbranch_execz .LBB1_107
; %bb.106:                              ;   in Loop: Header=BB1_38 Depth=2
	s_bcnt1_i32_b64 s9, s[36:37]
	v_mov_b32_e32 v2, s9
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[18:19], v[2:3], off offset:8 sc1
.LBB1_107:                              ;   in Loop: Header=BB1_38 Depth=2
	s_or_b64 exec, exec, s[34:35]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[20:21], v[18:19], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[20:21]
	s_cbranch_vccnz .LBB1_109
; %bb.108:                              ;   in Loop: Header=BB1_38 Depth=2
	global_load_dword v2, v[18:19], off offset:24
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v1, 0xffffff, v2
	s_nop 0
	v_readfirstlane_b32 s9, v1
	s_mov_b32 m0, s9
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[20:21], v[2:3], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB1_109:                              ; %Flow699
                                        ;   in Loop: Header=BB1_38 Depth=2
	s_or_b64 exec, exec, s[30:31]
	v_mov_b32_e32 v47, v3
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[46:47]
	s_branch .LBB1_113
.LBB1_110:                              ;   in Loop: Header=BB1_113 Depth=3
	s_or_b64 exec, exec, s[30:31]
	v_readfirstlane_b32 s9, v1
	s_cmp_eq_u32 s9, 0
	s_cbranch_scc1 .LBB1_112
; %bb.111:                              ;   in Loop: Header=BB1_113 Depth=3
	s_sleep 1
	s_cbranch_execnz .LBB1_113
	s_branch .LBB1_115
.LBB1_112:                              ;   in Loop: Header=BB1_38 Depth=2
	s_branch .LBB1_115
.LBB1_113:                              ;   Parent Loop BB1_3 Depth=1
                                        ;     Parent Loop BB1_38 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	v_mov_b32_e32 v1, 1
	s_and_saveexec_b64 s[30:31], s[6:7]
	s_cbranch_execz .LBB1_110
; %bb.114:                              ;   in Loop: Header=BB1_113 Depth=3
	global_load_dword v1, v[12:13], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v1, 1, v1
	s_branch .LBB1_110
.LBB1_115:                              ;   in Loop: Header=BB1_38 Depth=2
	global_load_dwordx2 v[18:19], v[4:5], off
	s_and_saveexec_b64 s[30:31], s[6:7]
	s_cbranch_execz .LBB1_37
; %bb.116:                              ;   in Loop: Header=BB1_38 Depth=2
	global_load_dwordx2 v[4:5], v3, s[22:23] offset:40
	global_load_dwordx2 v[12:13], v3, s[22:23] offset:24 sc0 sc1
	global_load_dwordx2 v[24:25], v3, s[22:23]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[20:21], v[4:5], 0, 1
	v_lshl_add_u64 v[26:27], v[20:21], 0, s[28:29]
	v_cmp_eq_u64_e32 vcc, 0, v[26:27]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v22, v12
	v_mov_b32_e32 v23, v13
	v_cndmask_b32_e32 v21, v27, v21, vcc
	v_cndmask_b32_e32 v20, v26, v20, vcc
	v_and_b32_e32 v1, v21, v5
	v_and_b32_e32 v2, v20, v4
	v_mul_lo_u32 v1, v1, 24
	v_mul_hi_u32 v5, v2, 24
	v_mul_lo_u32 v4, v2, 24
	v_add_u32_e32 v5, v5, v1
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[4:5], v[24:25], 0, v[4:5]
	global_store_dwordx2 v[4:5], v[12:13], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[22:23], v3, v[20:23], s[22:23] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[22:23], v[12:13]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB1_37
; %bb.117:                              ; %.preheader.i.i15.i.preheader
                                        ;   in Loop: Header=BB1_38 Depth=2
	s_mov_b64 s[6:7], 0
.LBB1_118:                              ; %.preheader.i.i15.i
                                        ;   Parent Loop BB1_3 Depth=1
                                        ;     Parent Loop BB1_38 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_sleep 1
	global_store_dwordx2 v[4:5], v[22:23], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[12:13], v3, v[20:23], s[22:23] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[12:13], v[22:23]
	s_or_b64 s[6:7], vcc, s[6:7]
	v_mov_b64_e32 v[22:23], v[12:13]
	s_andn2_b64 exec, exec, s[6:7]
	s_cbranch_execnz .LBB1_118
	s_branch .LBB1_37
.LBB1_119:                              ;   in Loop: Header=BB1_3 Depth=1
                                        ; implicit-def: $vgpr4_vgpr5
	s_branch .LBB1_5
.LBB1_120:                              ; %Flow738
                                        ;   in Loop: Header=BB1_3 Depth=1
	s_branch .LBB1_149
.LBB1_121:                              ;   in Loop: Header=BB1_3 Depth=1
                                        ; implicit-def: $vgpr18_vgpr19
	s_cbranch_execz .LBB1_149
; %bb.122:                              ;   in Loop: Header=BB1_3 Depth=1
	v_readfirstlane_b32 s6, v49
	v_mov_b64_e32 v[4:5], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[6:7], s6, v49
	s_and_saveexec_b64 s[10:11], s[6:7]
	s_cbranch_execz .LBB1_128
; %bb.123:                              ;   in Loop: Header=BB1_3 Depth=1
	global_load_dwordx2 v[20:21], v3, s[22:23] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[4:5], v3, s[22:23] offset:40
	global_load_dwordx2 v[8:9], v3, s[22:23]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v1, v4, v20
	v_and_b32_e32 v2, v5, v21
	v_mul_lo_u32 v2, v2, 24
	v_mul_hi_u32 v4, v1, 24
	v_add_u32_e32 v5, v4, v2
	v_mul_lo_u32 v4, v1, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[4:5], v[8:9], 0, v[4:5]
	global_load_dwordx2 v[18:19], v[4:5], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[4:5], v3, v[18:21], s[22:23] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[4:5], v[20:21]
	s_and_saveexec_b64 s[24:25], vcc
	s_cbranch_execz .LBB1_127
; %bb.124:                              ; %.preheader3.i.i.i23.preheader
                                        ;   in Loop: Header=BB1_3 Depth=1
	s_mov_b64 s[26:27], 0
.LBB1_125:                              ; %.preheader3.i.i.i23
                                        ;   Parent Loop BB1_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_load_dwordx2 v[8:9], v3, s[22:23] offset:40
	global_load_dwordx2 v[12:13], v3, s[22:23]
	v_mov_b64_e32 v[20:21], v[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v8, v20
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[4:5], s[28:29], v2, 24, v[12:13]
	v_and_b32_e32 v1, v9, v21
	v_mov_b32_e32 v2, v5
	v_mad_u64_u32 v[8:9], s[28:29], v1, 24, v[2:3]
	v_mov_b32_e32 v5, v8
	global_load_dwordx2 v[18:19], v[4:5], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[4:5], v3, v[18:21], s[22:23] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[4:5], v[20:21]
	s_or_b64 s[26:27], vcc, s[26:27]
	s_andn2_b64 exec, exec, s[26:27]
	s_cbranch_execnz .LBB1_125
; %bb.126:                              ; %Flow751
                                        ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[26:27]
.LBB1_127:                              ; %Flow753
                                        ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[24:25]
.LBB1_128:                              ; %.loopexit4.i.i.i18
                                        ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[10:11]
	global_load_dwordx2 v[8:9], v3, s[22:23] offset:40
	global_load_dwordx4 v[20:23], v3, s[22:23]
	v_readfirstlane_b32 s24, v4
	v_readfirstlane_b32 s25, v5
	s_mov_b64 s[10:11], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s26, v8
	v_readfirstlane_b32 s27, v9
	s_and_b64 s[26:27], s[24:25], s[26:27]
	s_mul_i32 s9, s27, 24
	s_mul_hi_u32 s28, s26, 24
	s_add_i32 s29, s28, s9
	s_mul_i32 s28, s26, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[8:9], v[20:21], 0, s[28:29]
	s_and_saveexec_b64 s[28:29], s[6:7]
	s_cbranch_execz .LBB1_130
; %bb.129:                              ;   in Loop: Header=BB1_3 Depth=1
	v_mov_b64_e32 v[4:5], s[10:11]
	global_store_dwordx4 v[8:9], v[4:7], off offset:8
.LBB1_130:                              ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[28:29]
	s_lshl_b64 s[10:11], s[26:27], 12
	v_lshl_add_u64 v[4:5], v[22:23], 0, s[10:11]
	v_and_or_b32 v16, v16, s39, 32
	v_mov_b32_e32 v18, v3
	v_mov_b32_e32 v19, v3
	v_readfirstlane_b32 s26, v4
	v_readfirstlane_b32 s27, v5
	s_mov_b32 s10, s8
	s_mov_b32 s11, s8
	s_mov_b32 s9, s8
	s_nop 1
	global_store_dwordx4 v46, v[16:19], s[26:27]
	s_nop 1
	v_mov_b64_e32 v[18:19], s[10:11]
	v_mov_b64_e32 v[16:17], s[8:9]
	global_store_dwordx4 v46, v[16:19], s[26:27] offset:16
	global_store_dwordx4 v46, v[16:19], s[26:27] offset:32
	global_store_dwordx4 v46, v[16:19], s[26:27] offset:48
	s_and_saveexec_b64 s[10:11], s[6:7]
	s_cbranch_execz .LBB1_138
; %bb.131:                              ;   in Loop: Header=BB1_3 Depth=1
	global_load_dwordx2 v[24:25], v3, s[22:23] offset:32 sc0 sc1
	global_load_dwordx2 v[12:13], v3, s[22:23] offset:40
	v_mov_b32_e32 v22, s24
	v_mov_b32_e32 v23, s25
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s26, v12
	v_readfirstlane_b32 s27, v13
	s_and_b64 s[26:27], s[26:27], s[24:25]
	s_mul_i32 s9, s27, 24
	s_mul_hi_u32 s27, s26, 24
	s_mul_i32 s26, s26, 24
	s_add_i32 s27, s27, s9
	v_lshl_add_u64 v[12:13], v[20:21], 0, s[26:27]
	global_store_dwordx2 v[12:13], v[24:25], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[18:19], v3, v[22:25], s[22:23] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[18:19], v[24:25]
	s_and_saveexec_b64 s[26:27], vcc
	s_cbranch_execz .LBB1_134
; %bb.132:                              ; %.preheader1.i.i.i21.preheader
                                        ;   in Loop: Header=BB1_3 Depth=1
	s_mov_b64 s[28:29], 0
.LBB1_133:                              ; %.preheader1.i.i.i21
                                        ;   Parent Loop BB1_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[12:13], v[18:19], off
	v_mov_b32_e32 v16, s24
	v_mov_b32_e32 v17, s25
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[16:17], v3, v[16:19], s[22:23] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[16:17], v[18:19]
	s_or_b64 s[28:29], vcc, s[28:29]
	v_mov_b64_e32 v[18:19], v[16:17]
	s_andn2_b64 exec, exec, s[28:29]
	s_cbranch_execnz .LBB1_133
.LBB1_134:                              ; %Flow749
                                        ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[26:27]
	global_load_dwordx2 v[12:13], v3, s[22:23] offset:16
	s_mov_b64 s[28:29], exec
	v_mbcnt_lo_u32_b32 v1, s28, 0
	v_mbcnt_hi_u32_b32 v1, s29, v1
	v_cmp_eq_u32_e32 vcc, 0, v1
	s_and_saveexec_b64 s[26:27], vcc
	s_cbranch_execz .LBB1_136
; %bb.135:                              ;   in Loop: Header=BB1_3 Depth=1
	s_bcnt1_i32_b64 s9, s[28:29]
	v_mov_b32_e32 v2, s9
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[12:13], v[2:3], off offset:8 sc1
.LBB1_136:                              ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[26:27]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[16:17], v[12:13], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[16:17]
	s_cbranch_vccnz .LBB1_138
; %bb.137:                              ;   in Loop: Header=BB1_3 Depth=1
	global_load_dword v2, v[12:13], off offset:24
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v1, 0xffffff, v2
	s_nop 0
	v_readfirstlane_b32 s9, v1
	s_mov_b32 m0, s9
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[16:17], v[2:3], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB1_138:                              ; %Flow750
                                        ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[10:11]
	v_mov_b32_e32 v47, v3
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[46:47]
	s_branch .LBB1_142
.LBB1_139:                              ;   in Loop: Header=BB1_142 Depth=2
	s_or_b64 exec, exec, s[10:11]
	v_readfirstlane_b32 s9, v1
	s_cmp_eq_u32 s9, 0
	s_cbranch_scc1 .LBB1_141
; %bb.140:                              ;   in Loop: Header=BB1_142 Depth=2
	s_sleep 1
	s_cbranch_execnz .LBB1_142
	s_branch .LBB1_144
.LBB1_141:                              ;   in Loop: Header=BB1_3 Depth=1
	s_branch .LBB1_144
.LBB1_142:                              ;   Parent Loop BB1_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v1, 1
	s_and_saveexec_b64 s[10:11], s[6:7]
	s_cbranch_execz .LBB1_139
; %bb.143:                              ;   in Loop: Header=BB1_142 Depth=2
	global_load_dword v1, v[8:9], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v1, 1, v1
	s_branch .LBB1_139
.LBB1_144:                              ;   in Loop: Header=BB1_3 Depth=1
	global_load_dwordx2 v[18:19], v[4:5], off
	s_and_saveexec_b64 s[10:11], s[6:7]
	s_cbranch_execz .LBB1_148
; %bb.145:                              ;   in Loop: Header=BB1_3 Depth=1
	global_load_dwordx2 v[4:5], v3, s[22:23] offset:40
	global_load_dwordx2 v[8:9], v3, s[22:23] offset:24 sc0 sc1
	global_load_dwordx2 v[12:13], v3, s[22:23]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[16:17], v[4:5], 0, 1
	v_lshl_add_u64 v[20:21], v[16:17], 0, s[24:25]
	v_cmp_eq_u64_e32 vcc, 0, v[20:21]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v22, v8
	v_mov_b32_e32 v23, v9
	v_cndmask_b32_e32 v21, v21, v17, vcc
	v_cndmask_b32_e32 v20, v20, v16, vcc
	v_and_b32_e32 v1, v21, v5
	v_and_b32_e32 v2, v20, v4
	v_mul_lo_u32 v1, v1, 24
	v_mul_hi_u32 v5, v2, 24
	v_mul_lo_u32 v4, v2, 24
	v_add_u32_e32 v5, v5, v1
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[4:5], v[12:13], 0, v[4:5]
	global_store_dwordx2 v[4:5], v[8:9], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[22:23], v3, v[20:23], s[22:23] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[22:23], v[8:9]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB1_148
; %bb.146:                              ; %.preheader.i.i.i20.preheader
                                        ;   in Loop: Header=BB1_3 Depth=1
	s_mov_b64 s[6:7], 0
.LBB1_147:                              ; %.preheader.i.i.i20
                                        ;   Parent Loop BB1_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[4:5], v[22:23], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[8:9], v3, v[20:23], s[22:23] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[8:9], v[22:23]
	s_or_b64 s[6:7], vcc, s[6:7]
	v_mov_b64_e32 v[22:23], v[8:9]
	s_andn2_b64 exec, exec, s[6:7]
	s_cbranch_execnz .LBB1_147
.LBB1_148:                              ; %Flow742
                                        ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[10:11]
.LBB1_149:                              ; %__ockl_printf_append_string_n.exit
                                        ;   in Loop: Header=BB1_3 Depth=1
	v_readfirstlane_b32 s6, v49
	v_mov_b64_e32 v[4:5], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[6:7], s6, v49
	s_and_saveexec_b64 s[10:11], s[6:7]
	s_cbranch_execz .LBB1_155
; %bb.150:                              ;   in Loop: Header=BB1_3 Depth=1
	global_load_dwordx2 v[22:23], v3, s[22:23] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[4:5], v3, s[22:23] offset:40
	global_load_dwordx2 v[8:9], v3, s[22:23]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v1, v4, v22
	v_and_b32_e32 v2, v5, v23
	v_mul_lo_u32 v2, v2, 24
	v_mul_hi_u32 v4, v1, 24
	v_add_u32_e32 v5, v4, v2
	v_mul_lo_u32 v4, v1, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[4:5], v[8:9], 0, v[4:5]
	global_load_dwordx2 v[20:21], v[4:5], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[4:5], v3, v[20:23], s[22:23] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[4:5], v[22:23]
	s_and_saveexec_b64 s[24:25], vcc
	s_cbranch_execz .LBB1_154
; %bb.151:                              ; %.preheader3.i.i.i30.preheader
                                        ;   in Loop: Header=BB1_3 Depth=1
	s_mov_b64 s[26:27], 0
.LBB1_152:                              ; %.preheader3.i.i.i30
                                        ;   Parent Loop BB1_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_load_dwordx2 v[8:9], v3, s[22:23] offset:40
	global_load_dwordx2 v[12:13], v3, s[22:23]
	v_mov_b64_e32 v[22:23], v[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v8, v22
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[4:5], s[28:29], v2, 24, v[12:13]
	v_and_b32_e32 v1, v9, v23
	v_mov_b32_e32 v2, v5
	v_mad_u64_u32 v[8:9], s[28:29], v1, 24, v[2:3]
	v_mov_b32_e32 v5, v8
	global_load_dwordx2 v[20:21], v[4:5], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[4:5], v3, v[20:23], s[22:23] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[4:5], v[22:23]
	s_or_b64 s[26:27], vcc, s[26:27]
	s_andn2_b64 exec, exec, s[26:27]
	s_cbranch_execnz .LBB1_152
; %bb.153:                              ; %Flow686
                                        ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[26:27]
.LBB1_154:                              ; %Flow688
                                        ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[24:25]
.LBB1_155:                              ; %.loopexit4.i.i.i24
                                        ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[10:11]
	global_load_dwordx2 v[8:9], v3, s[22:23] offset:40
	global_load_dwordx4 v[22:25], v3, s[22:23]
	v_readfirstlane_b32 s24, v4
	v_readfirstlane_b32 s25, v5
	s_mov_b64 s[10:11], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s26, v8
	v_readfirstlane_b32 s27, v9
	s_and_b64 s[26:27], s[24:25], s[26:27]
	s_mul_i32 s9, s27, 24
	s_mul_hi_u32 s28, s26, 24
	s_add_i32 s29, s28, s9
	s_mul_i32 s28, s26, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[8:9], v[22:23], 0, s[28:29]
	s_and_saveexec_b64 s[28:29], s[6:7]
	s_cbranch_execz .LBB1_157
; %bb.156:                              ;   in Loop: Header=BB1_3 Depth=1
	v_mov_b64_e32 v[4:5], s[10:11]
	global_store_dwordx4 v[8:9], v[4:7], off offset:8
.LBB1_157:                              ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[28:29]
	s_lshl_b64 s[10:11], s[26:27], 12
	v_lshl_add_u64 v[4:5], v[24:25], 0, s[10:11]
	v_and_or_b32 v18, v18, s39, 32
	v_mov_b32_e32 v20, v42
	v_mov_b32_e32 v21, v43
	v_readfirstlane_b32 s26, v4
	v_readfirstlane_b32 s27, v5
	s_mov_b32 s10, s8
	s_mov_b32 s11, s8
	s_mov_b32 s9, s8
	s_nop 1
	global_store_dwordx4 v46, v[18:21], s[26:27]
	s_nop 1
	v_mov_b64_e32 v[18:19], s[10:11]
	v_mov_b64_e32 v[16:17], s[8:9]
	global_store_dwordx4 v46, v[16:19], s[26:27] offset:16
	global_store_dwordx4 v46, v[16:19], s[26:27] offset:32
	global_store_dwordx4 v46, v[16:19], s[26:27] offset:48
	s_and_saveexec_b64 s[10:11], s[6:7]
	s_cbranch_execz .LBB1_165
; %bb.158:                              ;   in Loop: Header=BB1_3 Depth=1
	global_load_dwordx2 v[20:21], v3, s[22:23] offset:32 sc0 sc1
	global_load_dwordx2 v[12:13], v3, s[22:23] offset:40
	v_mov_b32_e32 v18, s24
	v_mov_b32_e32 v19, s25
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s26, v12
	v_readfirstlane_b32 s27, v13
	s_and_b64 s[26:27], s[26:27], s[24:25]
	s_mul_i32 s9, s27, 24
	s_mul_hi_u32 s27, s26, 24
	s_mul_i32 s26, s26, 24
	s_add_i32 s27, s27, s9
	v_lshl_add_u64 v[12:13], v[22:23], 0, s[26:27]
	global_store_dwordx2 v[12:13], v[20:21], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[18:19], v3, v[18:21], s[22:23] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[18:19], v[20:21]
	s_and_saveexec_b64 s[26:27], vcc
	s_cbranch_execz .LBB1_161
; %bb.159:                              ; %.preheader1.i.i.i28.preheader
                                        ;   in Loop: Header=BB1_3 Depth=1
	s_mov_b64 s[28:29], 0
.LBB1_160:                              ; %.preheader1.i.i.i28
                                        ;   Parent Loop BB1_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[12:13], v[18:19], off
	v_mov_b32_e32 v16, s24
	v_mov_b32_e32 v17, s25
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[16:17], v3, v[16:19], s[22:23] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[16:17], v[18:19]
	s_or_b64 s[28:29], vcc, s[28:29]
	v_mov_b64_e32 v[18:19], v[16:17]
	s_andn2_b64 exec, exec, s[28:29]
	s_cbranch_execnz .LBB1_160
.LBB1_161:                              ; %Flow684
                                        ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[26:27]
	global_load_dwordx2 v[12:13], v3, s[22:23] offset:16
	s_mov_b64 s[28:29], exec
	v_mbcnt_lo_u32_b32 v1, s28, 0
	v_mbcnt_hi_u32_b32 v1, s29, v1
	v_cmp_eq_u32_e32 vcc, 0, v1
	s_and_saveexec_b64 s[26:27], vcc
	s_cbranch_execz .LBB1_163
; %bb.162:                              ;   in Loop: Header=BB1_3 Depth=1
	s_bcnt1_i32_b64 s9, s[28:29]
	v_mov_b32_e32 v2, s9
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[12:13], v[2:3], off offset:8 sc1
.LBB1_163:                              ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[26:27]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[16:17], v[12:13], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[16:17]
	s_cbranch_vccnz .LBB1_165
; %bb.164:                              ;   in Loop: Header=BB1_3 Depth=1
	global_load_dword v2, v[12:13], off offset:24
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v1, 0xffffff, v2
	s_nop 0
	v_readfirstlane_b32 s9, v1
	s_mov_b32 m0, s9
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[16:17], v[2:3], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB1_165:                              ; %Flow685
                                        ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[10:11]
	v_mov_b32_e32 v47, v3
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[46:47]
	s_branch .LBB1_169
.LBB1_166:                              ;   in Loop: Header=BB1_169 Depth=2
	s_or_b64 exec, exec, s[10:11]
	v_readfirstlane_b32 s9, v1
	s_cmp_eq_u32 s9, 0
	s_cbranch_scc1 .LBB1_168
; %bb.167:                              ;   in Loop: Header=BB1_169 Depth=2
	s_sleep 1
	s_cbranch_execnz .LBB1_169
	s_branch .LBB1_171
.LBB1_168:                              ;   in Loop: Header=BB1_3 Depth=1
	s_branch .LBB1_171
.LBB1_169:                              ;   Parent Loop BB1_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v1, 1
	s_and_saveexec_b64 s[10:11], s[6:7]
	s_cbranch_execz .LBB1_166
; %bb.170:                              ;   in Loop: Header=BB1_169 Depth=2
	global_load_dword v1, v[8:9], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v1, 1, v1
	s_branch .LBB1_166
.LBB1_171:                              ;   in Loop: Header=BB1_3 Depth=1
	global_load_dwordx2 v[12:13], v[4:5], off
	s_and_saveexec_b64 s[10:11], s[6:7]
	s_cbranch_execz .LBB1_175
; %bb.172:                              ;   in Loop: Header=BB1_3 Depth=1
	global_load_dwordx2 v[4:5], v3, s[22:23] offset:40
	global_load_dwordx2 v[8:9], v3, s[22:23] offset:24 sc0 sc1
	global_load_dwordx2 v[20:21], v3, s[22:23]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[16:17], v[4:5], 0, 1
	v_lshl_add_u64 v[22:23], v[16:17], 0, s[24:25]
	v_cmp_eq_u64_e32 vcc, 0, v[22:23]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v18, v8
	v_mov_b32_e32 v19, v9
	v_cndmask_b32_e32 v17, v23, v17, vcc
	v_cndmask_b32_e32 v16, v22, v16, vcc
	v_and_b32_e32 v1, v17, v5
	v_and_b32_e32 v2, v16, v4
	v_mul_lo_u32 v1, v1, 24
	v_mul_hi_u32 v5, v2, 24
	v_mul_lo_u32 v4, v2, 24
	v_add_u32_e32 v5, v5, v1
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[4:5], v[20:21], 0, v[4:5]
	global_store_dwordx2 v[4:5], v[8:9], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[18:19], v3, v[16:19], s[22:23] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[18:19], v[8:9]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB1_175
; %bb.173:                              ; %.preheader.i.i.i27.preheader
                                        ;   in Loop: Header=BB1_3 Depth=1
	s_mov_b64 s[6:7], 0
.LBB1_174:                              ; %.preheader.i.i.i27
                                        ;   Parent Loop BB1_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[4:5], v[18:19], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[8:9], v3, v[16:19], s[22:23] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[8:9], v[18:19]
	s_or_b64 s[6:7], vcc, s[6:7]
	v_mov_b64_e32 v[18:19], v[8:9]
	s_andn2_b64 exec, exec, s[6:7]
	s_cbranch_execnz .LBB1_174
.LBB1_175:                              ; %__ockl_printf_append_args.exit
                                        ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[10:11]
	v_readfirstlane_b32 s6, v49
	v_mov_b64_e32 v[4:5], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[6:7], s6, v49
	s_and_saveexec_b64 s[10:11], s[6:7]
	s_cbranch_execz .LBB1_181
; %bb.176:                              ;   in Loop: Header=BB1_3 Depth=1
	global_load_dwordx2 v[18:19], v3, s[22:23] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[4:5], v3, s[22:23] offset:40
	global_load_dwordx2 v[8:9], v3, s[22:23]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v1, v4, v18
	v_and_b32_e32 v2, v5, v19
	v_mul_lo_u32 v2, v2, 24
	v_mul_hi_u32 v4, v1, 24
	v_add_u32_e32 v5, v4, v2
	v_mul_lo_u32 v4, v1, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[4:5], v[8:9], 0, v[4:5]
	global_load_dwordx2 v[16:17], v[4:5], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[4:5], v3, v[16:19], s[22:23] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[4:5], v[18:19]
	s_and_saveexec_b64 s[24:25], vcc
	s_cbranch_execz .LBB1_180
; %bb.177:                              ; %.preheader3.i.i.i37.preheader
                                        ;   in Loop: Header=BB1_3 Depth=1
	s_mov_b64 s[26:27], 0
.LBB1_178:                              ; %.preheader3.i.i.i37
                                        ;   Parent Loop BB1_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_load_dwordx2 v[8:9], v3, s[22:23] offset:40
	global_load_dwordx2 v[16:17], v3, s[22:23]
	v_mov_b64_e32 v[18:19], v[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v8, v18
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[4:5], s[28:29], v2, 24, v[16:17]
	v_and_b32_e32 v1, v9, v19
	v_mov_b32_e32 v2, v5
	v_mad_u64_u32 v[8:9], s[28:29], v1, 24, v[2:3]
	v_mov_b32_e32 v5, v8
	global_load_dwordx2 v[16:17], v[4:5], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[4:5], v3, v[16:19], s[22:23] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[4:5], v[18:19]
	s_or_b64 s[26:27], vcc, s[26:27]
	s_andn2_b64 exec, exec, s[26:27]
	s_cbranch_execnz .LBB1_178
; %bb.179:                              ; %Flow672
                                        ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[26:27]
.LBB1_180:                              ; %Flow674
                                        ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[24:25]
.LBB1_181:                              ; %.loopexit4.i.i.i31
                                        ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[10:11]
	global_load_dwordx2 v[8:9], v3, s[22:23] offset:40
	global_load_dwordx4 v[16:19], v3, s[22:23]
	v_readfirstlane_b32 s24, v4
	v_readfirstlane_b32 s25, v5
	s_mov_b64 s[10:11], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s26, v8
	v_readfirstlane_b32 s27, v9
	s_and_b64 s[26:27], s[24:25], s[26:27]
	s_mul_i32 s9, s27, 24
	s_mul_hi_u32 s28, s26, 24
	s_add_i32 s29, s28, s9
	s_mul_i32 s28, s26, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[8:9], v[16:17], 0, s[28:29]
	s_and_saveexec_b64 s[28:29], s[6:7]
	s_cbranch_execz .LBB1_183
; %bb.182:                              ;   in Loop: Header=BB1_3 Depth=1
	v_mov_b64_e32 v[4:5], s[10:11]
	global_store_dwordx4 v[8:9], v[4:7], off offset:8
.LBB1_183:                              ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[28:29]
	s_lshl_b64 s[10:11], s[26:27], 12
	v_lshl_add_u64 v[4:5], v[18:19], 0, s[10:11]
	v_and_or_b32 v12, v12, s39, 32
	v_readfirstlane_b32 s26, v4
	v_readfirstlane_b32 s27, v5
	s_mov_b32 s10, s8
	s_mov_b32 s11, s8
	s_mov_b32 s9, s8
	s_nop 1
	global_store_dwordx4 v46, v[12:15], s[26:27]
	s_nop 1
	v_mov_b64_e32 v[14:15], s[10:11]
	v_mov_b64_e32 v[12:13], s[8:9]
	global_store_dwordx4 v46, v[12:15], s[26:27] offset:16
	global_store_dwordx4 v46, v[12:15], s[26:27] offset:32
	global_store_dwordx4 v46, v[12:15], s[26:27] offset:48
	s_and_saveexec_b64 s[10:11], s[6:7]
	s_cbranch_execz .LBB1_191
; %bb.184:                              ;   in Loop: Header=BB1_3 Depth=1
	global_load_dwordx2 v[20:21], v3, s[22:23] offset:32 sc0 sc1
	global_load_dwordx2 v[12:13], v3, s[22:23] offset:40
	v_mov_b32_e32 v18, s24
	v_mov_b32_e32 v19, s25
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s26, v12
	v_readfirstlane_b32 s27, v13
	s_and_b64 s[26:27], s[26:27], s[24:25]
	s_mul_i32 s9, s27, 24
	s_mul_hi_u32 s27, s26, 24
	s_mul_i32 s26, s26, 24
	s_add_i32 s27, s27, s9
	v_lshl_add_u64 v[16:17], v[16:17], 0, s[26:27]
	global_store_dwordx2 v[16:17], v[20:21], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[14:15], v3, v[18:21], s[22:23] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[14:15], v[20:21]
	s_and_saveexec_b64 s[26:27], vcc
	s_cbranch_execz .LBB1_187
; %bb.185:                              ; %.preheader1.i.i.i35.preheader
                                        ;   in Loop: Header=BB1_3 Depth=1
	s_mov_b64 s[28:29], 0
.LBB1_186:                              ; %.preheader1.i.i.i35
                                        ;   Parent Loop BB1_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[16:17], v[14:15], off
	v_mov_b32_e32 v12, s24
	v_mov_b32_e32 v13, s25
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[12:13], v3, v[12:15], s[22:23] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[12:13], v[14:15]
	s_or_b64 s[28:29], vcc, s[28:29]
	v_mov_b64_e32 v[14:15], v[12:13]
	s_andn2_b64 exec, exec, s[28:29]
	s_cbranch_execnz .LBB1_186
.LBB1_187:                              ; %Flow670
                                        ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[26:27]
	global_load_dwordx2 v[12:13], v3, s[22:23] offset:16
	s_mov_b64 s[28:29], exec
	v_mbcnt_lo_u32_b32 v1, s28, 0
	v_mbcnt_hi_u32_b32 v1, s29, v1
	v_cmp_eq_u32_e32 vcc, 0, v1
	s_and_saveexec_b64 s[26:27], vcc
	s_cbranch_execz .LBB1_189
; %bb.188:                              ;   in Loop: Header=BB1_3 Depth=1
	s_bcnt1_i32_b64 s9, s[28:29]
	v_mov_b32_e32 v2, s9
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[12:13], v[2:3], off offset:8 sc1
.LBB1_189:                              ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[26:27]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[14:15], v[12:13], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[14:15]
	s_cbranch_vccnz .LBB1_191
; %bb.190:                              ;   in Loop: Header=BB1_3 Depth=1
	global_load_dword v2, v[12:13], off offset:24
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v1, 0xffffff, v2
	s_nop 0
	v_readfirstlane_b32 s9, v1
	s_mov_b32 m0, s9
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[14:15], v[2:3], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB1_191:                              ; %Flow671
                                        ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[10:11]
	v_mov_b32_e32 v47, v3
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[46:47]
	s_branch .LBB1_195
.LBB1_192:                              ;   in Loop: Header=BB1_195 Depth=2
	s_or_b64 exec, exec, s[10:11]
	v_readfirstlane_b32 s9, v1
	s_cmp_eq_u32 s9, 0
	s_cbranch_scc1 .LBB1_194
; %bb.193:                              ;   in Loop: Header=BB1_195 Depth=2
	s_sleep 1
	s_cbranch_execnz .LBB1_195
	s_branch .LBB1_197
.LBB1_194:                              ;   in Loop: Header=BB1_3 Depth=1
	s_branch .LBB1_197
.LBB1_195:                              ;   Parent Loop BB1_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v1, 1
	s_and_saveexec_b64 s[10:11], s[6:7]
	s_cbranch_execz .LBB1_192
; %bb.196:                              ;   in Loop: Header=BB1_195 Depth=2
	global_load_dword v1, v[8:9], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v1, 1, v1
	s_branch .LBB1_192
.LBB1_197:                              ;   in Loop: Header=BB1_3 Depth=1
	global_load_dwordx2 v[8:9], v[4:5], off
	s_and_saveexec_b64 s[10:11], s[6:7]
	s_cbranch_execz .LBB1_201
; %bb.198:                              ;   in Loop: Header=BB1_3 Depth=1
	global_load_dwordx2 v[4:5], v3, s[22:23] offset:40
	global_load_dwordx2 v[16:17], v3, s[22:23] offset:24 sc0 sc1
	global_load_dwordx2 v[18:19], v3, s[22:23]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[12:13], v[4:5], 0, 1
	v_lshl_add_u64 v[20:21], v[12:13], 0, s[24:25]
	v_cmp_eq_u64_e32 vcc, 0, v[20:21]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v14, v16
	v_mov_b32_e32 v15, v17
	v_cndmask_b32_e32 v13, v21, v13, vcc
	v_cndmask_b32_e32 v12, v20, v12, vcc
	v_and_b32_e32 v1, v13, v5
	v_and_b32_e32 v2, v12, v4
	v_mul_lo_u32 v1, v1, 24
	v_mul_hi_u32 v5, v2, 24
	v_mul_lo_u32 v4, v2, 24
	v_add_u32_e32 v5, v5, v1
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[4:5], v[18:19], 0, v[4:5]
	global_store_dwordx2 v[4:5], v[16:17], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[14:15], v3, v[12:15], s[22:23] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[14:15], v[16:17]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB1_201
; %bb.199:                              ; %.preheader.i.i.i34.preheader
                                        ;   in Loop: Header=BB1_3 Depth=1
	s_mov_b64 s[6:7], 0
.LBB1_200:                              ; %.preheader.i.i.i34
                                        ;   Parent Loop BB1_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[4:5], v[14:15], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[16:17], v3, v[12:15], s[22:23] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[16:17], v[14:15]
	s_or_b64 s[6:7], vcc, s[6:7]
	v_mov_b64_e32 v[14:15], v[16:17]
	s_andn2_b64 exec, exec, s[6:7]
	s_cbranch_execnz .LBB1_200
.LBB1_201:                              ; %__ockl_printf_append_args.exit38
                                        ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[10:11]
	v_readfirstlane_b32 s6, v49
	v_mov_b64_e32 v[4:5], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[6:7], s6, v49
	s_and_saveexec_b64 s[10:11], s[6:7]
	s_cbranch_execz .LBB1_207
; %bb.202:                              ;   in Loop: Header=BB1_3 Depth=1
	global_load_dwordx2 v[14:15], v3, s[22:23] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[4:5], v3, s[22:23] offset:40
	global_load_dwordx2 v[12:13], v3, s[22:23]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v1, v4, v14
	v_and_b32_e32 v2, v5, v15
	v_mul_lo_u32 v2, v2, 24
	v_mul_hi_u32 v4, v1, 24
	v_add_u32_e32 v5, v4, v2
	v_mul_lo_u32 v4, v1, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[4:5], v[12:13], 0, v[4:5]
	global_load_dwordx2 v[12:13], v[4:5], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[4:5], v3, v[12:15], s[22:23] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[4:5], v[14:15]
	s_and_saveexec_b64 s[24:25], vcc
	s_cbranch_execz .LBB1_206
; %bb.203:                              ; %.preheader3.i.i.i45.preheader
                                        ;   in Loop: Header=BB1_3 Depth=1
	s_mov_b64 s[26:27], 0
.LBB1_204:                              ; %.preheader3.i.i.i45
                                        ;   Parent Loop BB1_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_load_dwordx2 v[12:13], v3, s[22:23] offset:40
	global_load_dwordx2 v[16:17], v3, s[22:23]
	v_mov_b64_e32 v[14:15], v[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v12, v14
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[4:5], s[28:29], v2, 24, v[16:17]
	v_and_b32_e32 v1, v13, v15
	v_mov_b32_e32 v2, v5
	v_mad_u64_u32 v[12:13], s[28:29], v1, 24, v[2:3]
	v_mov_b32_e32 v5, v12
	global_load_dwordx2 v[12:13], v[4:5], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[4:5], v3, v[12:15], s[22:23] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[4:5], v[14:15]
	s_or_b64 s[26:27], vcc, s[26:27]
	s_andn2_b64 exec, exec, s[26:27]
	s_cbranch_execnz .LBB1_204
; %bb.205:                              ; %Flow658
                                        ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[26:27]
.LBB1_206:                              ; %Flow660
                                        ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[24:25]
.LBB1_207:                              ; %.loopexit4.i.i.i39
                                        ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[10:11]
	global_load_dwordx2 v[16:17], v3, s[22:23] offset:40
	global_load_dwordx4 v[12:15], v3, s[22:23]
	v_readfirstlane_b32 s24, v4
	v_readfirstlane_b32 s25, v5
	s_mov_b64 s[10:11], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s26, v16
	v_readfirstlane_b32 s27, v17
	s_and_b64 s[26:27], s[24:25], s[26:27]
	s_mul_i32 s9, s27, 24
	s_mul_hi_u32 s28, s26, 24
	s_add_i32 s29, s28, s9
	s_mul_i32 s28, s26, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[16:17], v[12:13], 0, s[28:29]
	s_and_saveexec_b64 s[28:29], s[6:7]
	s_cbranch_execz .LBB1_209
; %bb.208:                              ;   in Loop: Header=BB1_3 Depth=1
	v_mov_b64_e32 v[4:5], s[10:11]
	global_store_dwordx4 v[16:17], v[4:7], off offset:8
.LBB1_209:                              ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[28:29]
	s_lshl_b64 s[10:11], s[26:27], 12
	v_lshl_add_u64 v[4:5], v[14:15], 0, s[10:11]
	v_and_or_b32 v8, v8, s40, 34
	v_readfirstlane_b32 s26, v4
	v_readfirstlane_b32 s27, v5
	s_mov_b32 s9, s8
	s_mov_b32 s10, s8
	s_mov_b32 s11, s8
	s_nop 1
	global_store_dwordx4 v46, v[8:11], s[26:27]
	s_nop 1
	v_mov_b64_e32 v[8:9], s[8:9]
	v_mov_b64_e32 v[10:11], s[10:11]
	global_store_dwordx4 v46, v[8:11], s[26:27] offset:16
	global_store_dwordx4 v46, v[8:11], s[26:27] offset:32
	global_store_dwordx4 v46, v[8:11], s[26:27] offset:48
	s_and_saveexec_b64 s[10:11], s[6:7]
	s_cbranch_execz .LBB1_217
; %bb.210:                              ;   in Loop: Header=BB1_3 Depth=1
	global_load_dwordx2 v[20:21], v3, s[22:23] offset:32 sc0 sc1
	global_load_dwordx2 v[4:5], v3, s[22:23] offset:40
	v_mov_b32_e32 v18, s24
	v_mov_b32_e32 v19, s25
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s26, v4
	v_readfirstlane_b32 s27, v5
	s_and_b64 s[26:27], s[26:27], s[24:25]
	s_mul_i32 s9, s27, 24
	s_mul_hi_u32 s27, s26, 24
	s_mul_i32 s26, s26, 24
	s_add_i32 s27, s27, s9
	v_lshl_add_u64 v[4:5], v[12:13], 0, s[26:27]
	global_store_dwordx2 v[4:5], v[20:21], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v3, v[18:21], s[22:23] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[10:11], v[20:21]
	s_and_saveexec_b64 s[26:27], vcc
	s_cbranch_execz .LBB1_213
; %bb.211:                              ; %.preheader1.i.i.i43.preheader
                                        ;   in Loop: Header=BB1_3 Depth=1
	s_mov_b64 s[28:29], 0
.LBB1_212:                              ; %.preheader1.i.i.i43
                                        ;   Parent Loop BB1_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[4:5], v[10:11], off
	v_mov_b32_e32 v8, s24
	v_mov_b32_e32 v9, s25
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[8:9], v3, v[8:11], s[22:23] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[8:9], v[10:11]
	s_or_b64 s[28:29], vcc, s[28:29]
	v_mov_b64_e32 v[10:11], v[8:9]
	s_andn2_b64 exec, exec, s[28:29]
	s_cbranch_execnz .LBB1_212
.LBB1_213:                              ; %Flow656
                                        ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[26:27]
	global_load_dwordx2 v[4:5], v3, s[22:23] offset:16
	s_mov_b64 s[28:29], exec
	v_mbcnt_lo_u32_b32 v1, s28, 0
	v_mbcnt_hi_u32_b32 v1, s29, v1
	v_cmp_eq_u32_e32 vcc, 0, v1
	s_and_saveexec_b64 s[26:27], vcc
	s_cbranch_execz .LBB1_215
; %bb.214:                              ;   in Loop: Header=BB1_3 Depth=1
	s_bcnt1_i32_b64 s9, s[28:29]
	v_mov_b32_e32 v2, s9
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[4:5], v[2:3], off offset:8 sc1
.LBB1_215:                              ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[26:27]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[8:9], v[4:5], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[8:9]
	s_cbranch_vccnz .LBB1_217
; %bb.216:                              ;   in Loop: Header=BB1_3 Depth=1
	global_load_dword v2, v[4:5], off offset:24
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v1, 0xffffff, v2
	s_nop 0
	v_readfirstlane_b32 s9, v1
	s_mov_b32 m0, s9
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[8:9], v[2:3], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB1_217:                              ; %Flow657
                                        ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[10:11]
	s_branch .LBB1_221
.LBB1_218:                              ;   in Loop: Header=BB1_221 Depth=2
	s_or_b64 exec, exec, s[10:11]
	v_readfirstlane_b32 s9, v1
	s_cmp_eq_u32 s9, 0
	s_cbranch_scc1 .LBB1_220
; %bb.219:                              ;   in Loop: Header=BB1_221 Depth=2
	s_sleep 1
	s_cbranch_execnz .LBB1_221
	s_branch .LBB1_223
.LBB1_220:                              ;   in Loop: Header=BB1_3 Depth=1
	s_branch .LBB1_223
.LBB1_221:                              ;   Parent Loop BB1_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v1, 1
	s_and_saveexec_b64 s[10:11], s[6:7]
	s_cbranch_execz .LBB1_218
; %bb.222:                              ;   in Loop: Header=BB1_221 Depth=2
	global_load_dword v1, v[16:17], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v1, 1, v1
	s_branch .LBB1_218
.LBB1_223:                              ;   in Loop: Header=BB1_3 Depth=1
	s_and_saveexec_b64 s[10:11], s[6:7]
	s_cbranch_execz .LBB1_227
; %bb.224:                              ;   in Loop: Header=BB1_3 Depth=1
	global_load_dwordx2 v[4:5], v3, s[22:23] offset:40
	global_load_dwordx2 v[12:13], v3, s[22:23] offset:24 sc0 sc1
	global_load_dwordx2 v[14:15], v3, s[22:23]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[8:9], v[4:5], 0, 1
	v_lshl_add_u64 v[16:17], v[8:9], 0, s[24:25]
	v_cmp_eq_u64_e32 vcc, 0, v[16:17]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v10, v12
	v_mov_b32_e32 v11, v13
	v_cndmask_b32_e32 v9, v17, v9, vcc
	v_cndmask_b32_e32 v8, v16, v8, vcc
	v_and_b32_e32 v1, v9, v5
	v_and_b32_e32 v2, v8, v4
	v_mul_lo_u32 v1, v1, 24
	v_mul_hi_u32 v5, v2, 24
	v_mul_lo_u32 v4, v2, 24
	v_add_u32_e32 v5, v5, v1
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[4:5], v[14:15], 0, v[4:5]
	global_store_dwordx2 v[4:5], v[12:13], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v3, v[8:11], s[22:23] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[10:11], v[12:13]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB1_227
; %bb.225:                              ; %.preheader.i.i.i42.preheader
                                        ;   in Loop: Header=BB1_3 Depth=1
	s_mov_b64 s[6:7], 0
.LBB1_226:                              ; %.preheader.i.i.i42
                                        ;   Parent Loop BB1_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[4:5], v[10:11], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[12:13], v3, v[8:11], s[22:23] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[12:13], v[10:11]
	s_or_b64 s[6:7], vcc, s[6:7]
	v_mov_b64_e32 v[10:11], v[12:13]
	s_andn2_b64 exec, exec, s[6:7]
	s_cbranch_execnz .LBB1_226
.LBB1_227:                              ; %__ockl_printf_append_args.exit46
                                        ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[10:11]
	ds_read2_b64 v[8:11], v48 offset1:16
	s_waitcnt lgkmcnt(0)
	v_add_f64 v[4:5], v[10:11], v[8:9]
	ds_write_b64 v48, v[4:5]
.LBB1_228:                              ; %Flow771
                                        ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[6:7], s[4:5], s[0:1]
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_and_saveexec_b64 s[4:5], s[6:7]
	s_cbranch_execz .LBB1_230
; %bb.229:                              ;   in Loop: Header=BB1_3 Depth=1
	ds_read_b64 v[4:5], v48
	s_waitcnt lgkmcnt(0)
	global_store_dwordx2 v[44:45], v[4:5], off
.LBB1_230:                              ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[4:5]
	s_barrier
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB1_2
; %bb.231:                              ;   in Loop: Header=BB1_3 Depth=1
	s_load_dwordx2 s[20:21], s[16:17], 0x50
	v_readfirstlane_b32 s4, v49
	v_mov_b64_e32 v[4:5], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[4:5], s4, v49
	s_and_saveexec_b64 s[10:11], s[4:5]
	s_cbranch_execz .LBB1_237
; %bb.232:                              ;   in Loop: Header=BB1_3 Depth=1
	s_waitcnt lgkmcnt(0)
	global_load_dwordx2 v[10:11], v3, s[20:21] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[4:5], v3, s[20:21] offset:40
	global_load_dwordx2 v[8:9], v3, s[20:21]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v1, v4, v10
	v_and_b32_e32 v2, v5, v11
	v_mul_lo_u32 v2, v2, 24
	v_mul_hi_u32 v4, v1, 24
	v_add_u32_e32 v5, v4, v2
	v_mul_lo_u32 v4, v1, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[4:5], v[8:9], 0, v[4:5]
	global_load_dwordx2 v[8:9], v[4:5], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[4:5], v3, v[8:11], s[20:21] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[4:5], v[10:11]
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB1_236
; %bb.233:                              ; %.preheader3.i.i.i53.preheader
                                        ;   in Loop: Header=BB1_3 Depth=1
	s_mov_b64 s[24:25], 0
.LBB1_234:                              ; %.preheader3.i.i.i53
                                        ;   Parent Loop BB1_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_load_dwordx2 v[8:9], v3, s[20:21] offset:40
	global_load_dwordx2 v[12:13], v3, s[20:21]
	v_mov_b64_e32 v[10:11], v[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v8, v10
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[4:5], s[26:27], v2, 24, v[12:13]
	v_and_b32_e32 v1, v9, v11
	v_mov_b32_e32 v2, v5
	v_mad_u64_u32 v[8:9], s[26:27], v1, 24, v[2:3]
	v_mov_b32_e32 v5, v8
	global_load_dwordx2 v[8:9], v[4:5], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[4:5], v3, v[8:11], s[20:21] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[4:5], v[10:11]
	s_or_b64 s[24:25], vcc, s[24:25]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBB1_234
; %bb.235:                              ; %Flow643
                                        ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[24:25]
.LBB1_236:                              ; %Flow645
                                        ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[22:23]
.LBB1_237:                              ; %.loopexit4.i.i.i47
                                        ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[10:11]
	s_waitcnt lgkmcnt(0)
	global_load_dwordx2 v[12:13], v3, s[20:21] offset:40
	global_load_dwordx4 v[8:11], v3, s[20:21]
	v_readfirstlane_b32 s22, v4
	v_readfirstlane_b32 s23, v5
	s_mov_b64 s[10:11], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s24, v12
	v_readfirstlane_b32 s25, v13
	s_and_b64 s[24:25], s[22:23], s[24:25]
	s_mul_i32 s9, s25, 24
	s_mul_hi_u32 s26, s24, 24
	s_add_i32 s27, s26, s9
	s_mul_i32 s26, s24, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[12:13], v[8:9], 0, s[26:27]
	s_and_saveexec_b64 s[26:27], s[4:5]
	s_cbranch_execz .LBB1_239
; %bb.238:                              ;   in Loop: Header=BB1_3 Depth=1
	v_mov_b64_e32 v[4:5], s[10:11]
	global_store_dwordx4 v[12:13], v[4:7], off offset:8
.LBB1_239:                              ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[26:27]
	s_lshl_b64 s[10:11], s[24:25], 12
	v_lshl_add_u64 v[4:5], v[10:11], 0, s[10:11]
	s_mov_b32 s10, s8
	s_mov_b32 s11, s8
	s_mov_b32 s9, s8
	v_mov_b64_e32 v[16:17], s[10:11]
	v_lshlrev_b32_e32 v30, 6, v49
	v_mov_b32_e32 v1, v3
	v_mov_b32_e32 v2, v3
	v_readfirstlane_b32 s24, v4
	v_readfirstlane_b32 s25, v5
	v_mov_b64_e32 v[14:15], s[8:9]
	s_nop 3
	global_store_dwordx4 v30, v[0:3], s[24:25]
	global_store_dwordx4 v30, v[14:17], s[24:25] offset:16
	global_store_dwordx4 v30, v[14:17], s[24:25] offset:32
	global_store_dwordx4 v30, v[14:17], s[24:25] offset:48
	s_and_saveexec_b64 s[10:11], s[4:5]
	s_cbranch_execz .LBB1_247
; %bb.240:                              ;   in Loop: Header=BB1_3 Depth=1
	global_load_dwordx2 v[18:19], v3, s[20:21] offset:32 sc0 sc1
	global_load_dwordx2 v[10:11], v3, s[20:21] offset:40
	v_mov_b32_e32 v16, s22
	v_mov_b32_e32 v17, s23
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v1, s22, v10
	v_and_b32_e32 v2, s23, v11
	v_mul_lo_u32 v2, v2, 24
	v_mul_hi_u32 v11, v1, 24
	v_mul_lo_u32 v10, v1, 24
	v_add_u32_e32 v11, v11, v2
	v_lshl_add_u64 v[14:15], v[8:9], 0, v[10:11]
	global_store_dwordx2 v[14:15], v[18:19], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v3, v[16:19], s[20:21] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[10:11], v[18:19]
	s_and_saveexec_b64 s[24:25], vcc
	s_cbranch_execz .LBB1_243
; %bb.241:                              ; %.preheader1.i.i.i51.preheader
                                        ;   in Loop: Header=BB1_3 Depth=1
	s_mov_b64 s[26:27], 0
.LBB1_242:                              ; %.preheader1.i.i.i51
                                        ;   Parent Loop BB1_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[14:15], v[10:11], off
	v_mov_b32_e32 v8, s22
	v_mov_b32_e32 v9, s23
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[8:9], v3, v[8:11], s[20:21] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[8:9], v[10:11]
	s_or_b64 s[26:27], vcc, s[26:27]
	v_mov_b64_e32 v[10:11], v[8:9]
	s_andn2_b64 exec, exec, s[26:27]
	s_cbranch_execnz .LBB1_242
.LBB1_243:                              ; %Flow641
                                        ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[24:25]
	global_load_dwordx2 v[8:9], v3, s[20:21] offset:16
	s_mov_b64 s[26:27], exec
	v_mbcnt_lo_u32_b32 v1, s26, 0
	v_mbcnt_hi_u32_b32 v1, s27, v1
	v_cmp_eq_u32_e32 vcc, 0, v1
	s_and_saveexec_b64 s[24:25], vcc
	s_cbranch_execz .LBB1_245
; %bb.244:                              ;   in Loop: Header=BB1_3 Depth=1
	s_bcnt1_i32_b64 s9, s[26:27]
	v_mov_b32_e32 v2, s9
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[8:9], v[2:3], off offset:8 sc1
.LBB1_245:                              ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[24:25]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[10:11], v[8:9], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[10:11]
	s_cbranch_vccnz .LBB1_247
; %bb.246:                              ;   in Loop: Header=BB1_3 Depth=1
	global_load_dword v2, v[8:9], off offset:24
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v1, 0xffffff, v2
	s_nop 0
	v_readfirstlane_b32 s9, v1
	s_mov_b32 m0, s9
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[10:11], v[2:3], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB1_247:                              ; %Flow642
                                        ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[10:11]
	v_mov_b32_e32 v31, v3
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[30:31]
	s_branch .LBB1_251
.LBB1_248:                              ;   in Loop: Header=BB1_251 Depth=2
	s_or_b64 exec, exec, s[10:11]
	v_readfirstlane_b32 s9, v1
	s_cmp_eq_u32 s9, 0
	s_cbranch_scc1 .LBB1_250
; %bb.249:                              ;   in Loop: Header=BB1_251 Depth=2
	s_sleep 1
	s_cbranch_execnz .LBB1_251
	s_branch .LBB1_253
.LBB1_250:                              ;   in Loop: Header=BB1_3 Depth=1
	s_branch .LBB1_253
.LBB1_251:                              ;   Parent Loop BB1_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v1, 1
	s_and_saveexec_b64 s[10:11], s[4:5]
	s_cbranch_execz .LBB1_248
; %bb.252:                              ;   in Loop: Header=BB1_251 Depth=2
	global_load_dword v1, v[12:13], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v1, 1, v1
	s_branch .LBB1_248
.LBB1_253:                              ;   in Loop: Header=BB1_3 Depth=1
	global_load_dwordx2 v[8:9], v[4:5], off
	s_and_saveexec_b64 s[10:11], s[4:5]
	s_cbranch_execz .LBB1_257
; %bb.254:                              ;   in Loop: Header=BB1_3 Depth=1
	global_load_dwordx2 v[4:5], v3, s[20:21] offset:40
	global_load_dwordx2 v[14:15], v3, s[20:21] offset:24 sc0 sc1
	global_load_dwordx2 v[16:17], v3, s[20:21]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[10:11], v[4:5], 0, 1
	v_lshl_add_u64 v[18:19], v[10:11], 0, s[22:23]
	v_cmp_eq_u64_e32 vcc, 0, v[18:19]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v12, v14
	v_mov_b32_e32 v13, v15
	v_cndmask_b32_e32 v11, v19, v11, vcc
	v_cndmask_b32_e32 v10, v18, v10, vcc
	v_and_b32_e32 v1, v11, v5
	v_and_b32_e32 v2, v10, v4
	v_mul_lo_u32 v1, v1, 24
	v_mul_hi_u32 v5, v2, 24
	v_mul_lo_u32 v4, v2, 24
	v_add_u32_e32 v5, v5, v1
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[4:5], v[16:17], 0, v[4:5]
	global_store_dwordx2 v[4:5], v[14:15], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[12:13], v3, v[10:13], s[20:21] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[12:13], v[14:15]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB1_257
; %bb.255:                              ; %.preheader.i.i.i50.preheader
                                        ;   in Loop: Header=BB1_3 Depth=1
	s_mov_b64 s[4:5], 0
.LBB1_256:                              ; %.preheader.i.i.i50
                                        ;   Parent Loop BB1_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[4:5], v[12:13], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[14:15], v3, v[10:13], s[20:21] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[14:15], v[12:13]
	s_or_b64 s[4:5], vcc, s[4:5]
	v_mov_b64_e32 v[12:13], v[14:15]
	s_andn2_b64 exec, exec, s[4:5]
	s_cbranch_execnz .LBB1_256
.LBB1_257:                              ; %__ockl_printf_begin.exit54
                                        ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[10:11]
	s_and_b64 vcc, exec, s[12:13]
	s_cbranch_vccz .LBB1_342
; %bb.258:                              ;   in Loop: Header=BB1_3 Depth=1
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v10, -3, v8
	v_mov_b32_e32 v11, v9
	s_mov_b64 s[22:23], 45
	s_getpc_b64 s[10:11]
	s_add_u32 s10, s10, .str.1@rel32@lo+4
	s_addc_u32 s11, s11, .str.1@rel32@hi+12
	s_branch .LBB1_260
.LBB1_259:                              ; %__ockl_hostcall_preview.exit19.i71
                                        ;   in Loop: Header=BB1_260 Depth=2
	s_or_b64 exec, exec, s[28:29]
	s_sub_u32 s22, s22, s24
	s_subb_u32 s23, s23, s25
	s_add_u32 s10, s10, s24
	s_addc_u32 s11, s11, s25
	s_cmp_lg_u64 s[22:23], 0
	s_cbranch_scc0 .LBB1_341
.LBB1_260:                              ;   Parent Loop BB1_3 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB1_263 Depth 3
                                        ;       Child Loop BB1_269 Depth 3
                                        ;       Child Loop BB1_278 Depth 3
                                        ;       Child Loop BB1_286 Depth 3
                                        ;       Child Loop BB1_294 Depth 3
                                        ;       Child Loop BB1_302 Depth 3
                                        ;       Child Loop BB1_310 Depth 3
                                        ;       Child Loop BB1_318 Depth 3
                                        ;       Child Loop BB1_326 Depth 3
                                        ;       Child Loop BB1_335 Depth 3
                                        ;       Child Loop BB1_340 Depth 3
	v_cmp_lt_u64_e64 s[4:5], s[22:23], 56
	s_and_b64 s[4:5], s[4:5], exec
	v_cmp_gt_u64_e64 s[4:5], s[22:23], 7
	s_cselect_b32 s25, s23, 0
	s_cselect_b32 s24, s22, 56
	s_and_b64 vcc, exec, s[4:5]
	s_cbranch_vccnz .LBB1_270
; %bb.261:                              ;   in Loop: Header=BB1_260 Depth=2
	s_mov_b64 s[4:5], 0
	s_cmp_eq_u64 s[22:23], 0
	s_waitcnt vmcnt(0)
	v_mov_b64_e32 v[12:13], 0
	s_cbranch_scc1 .LBB1_264
; %bb.262:                              ; %.preheader30.i55.preheader
                                        ;   in Loop: Header=BB1_260 Depth=2
	s_lshl_b64 s[26:27], s[24:25], 3
	s_mov_b64 s[28:29], 0
	v_mov_b64_e32 v[12:13], 0
	s_mov_b64 s[30:31], s[10:11]
.LBB1_263:                              ; %.preheader30.i55
                                        ;   Parent Loop BB1_3 Depth=1
                                        ;     Parent Loop BB1_260 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	global_load_ubyte v1, v3, s[30:31]
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v2, 0xffff, v1
	v_lshlrev_b64 v[4:5], s28, v[2:3]
	s_add_u32 s28, s28, 8
	s_addc_u32 s29, s29, 0
	s_add_u32 s30, s30, 1
	s_addc_u32 s31, s31, 0
	v_or_b32_e32 v12, v4, v12
	s_cmp_lg_u32 s26, s28
	v_or_b32_e32 v13, v5, v13
	s_cbranch_scc1 .LBB1_263
.LBB1_264:                              ; %Flow612
                                        ;   in Loop: Header=BB1_260 Depth=2
	s_mov_b32 s9, 0
	s_andn2_b64 vcc, exec, s[4:5]
	s_mov_b64 s[4:5], s[10:11]
	s_cbranch_vccnz .LBB1_266
.LBB1_265:                              ;   in Loop: Header=BB1_260 Depth=2
	global_load_dwordx2 v[12:13], v3, s[10:11]
	s_add_i32 s9, s24, -8
	s_add_u32 s4, s10, 8
	s_addc_u32 s5, s11, 0
.LBB1_266:                              ; %.loopexit31.i56
                                        ;   in Loop: Header=BB1_260 Depth=2
	s_cmp_gt_u32 s9, 7
	s_cbranch_scc1 .LBB1_271
; %bb.267:                              ;   in Loop: Header=BB1_260 Depth=2
	s_cmp_eq_u32 s9, 0
	s_cbranch_scc1 .LBB1_272
; %bb.268:                              ; %.preheader28.i57.preheader
                                        ;   in Loop: Header=BB1_260 Depth=2
	s_mov_b64 s[26:27], 0
	v_mov_b64_e32 v[14:15], 0
	s_mov_b64 s[28:29], 0
.LBB1_269:                              ; %.preheader28.i57
                                        ;   Parent Loop BB1_3 Depth=1
                                        ;     Parent Loop BB1_260 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_add_u32 s30, s4, s28
	s_addc_u32 s31, s5, s29
	global_load_ubyte v1, v3, s[30:31]
	s_add_u32 s28, s28, 1
	s_addc_u32 s29, s29, 0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v2, 0xffff, v1
	v_lshlrev_b64 v[4:5], s26, v[2:3]
	s_add_u32 s26, s26, 8
	s_addc_u32 s27, s27, 0
	v_or_b32_e32 v14, v4, v14
	s_cmp_lg_u32 s9, s28
	v_or_b32_e32 v15, v5, v15
	s_cbranch_scc1 .LBB1_269
	s_branch .LBB1_273
.LBB1_270:                              ;   in Loop: Header=BB1_260 Depth=2
	s_mov_b32 s9, 0
	s_mov_b64 s[4:5], s[10:11]
	s_branch .LBB1_265
.LBB1_271:                              ;   in Loop: Header=BB1_260 Depth=2
                                        ; implicit-def: $vgpr14_vgpr15
	s_mov_b32 s30, 0
	s_branch .LBB1_274
.LBB1_272:                              ;   in Loop: Header=BB1_260 Depth=2
	v_mov_b64_e32 v[14:15], 0
.LBB1_273:                              ; %Flow605
                                        ;   in Loop: Header=BB1_260 Depth=2
	s_mov_b32 s30, 0
	s_cbranch_execnz .LBB1_275
.LBB1_274:                              ;   in Loop: Header=BB1_260 Depth=2
	global_load_dwordx2 v[14:15], v3, s[4:5]
	s_add_i32 s30, s9, -8
	s_add_u32 s4, s4, 8
	s_addc_u32 s5, s5, 0
.LBB1_275:                              ; %.loopexit29.i58
                                        ;   in Loop: Header=BB1_260 Depth=2
	s_cmp_gt_u32 s30, 7
	s_cbranch_scc1 .LBB1_279
; %bb.276:                              ;   in Loop: Header=BB1_260 Depth=2
	s_cmp_eq_u32 s30, 0
	s_cbranch_scc1 .LBB1_280
; %bb.277:                              ; %.preheader26.i59.preheader
                                        ;   in Loop: Header=BB1_260 Depth=2
	s_mov_b64 s[26:27], 0
	v_mov_b64_e32 v[16:17], 0
	s_mov_b64 s[28:29], 0
.LBB1_278:                              ; %.preheader26.i59
                                        ;   Parent Loop BB1_3 Depth=1
                                        ;     Parent Loop BB1_260 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_add_u32 s34, s4, s28
	s_addc_u32 s35, s5, s29
	global_load_ubyte v1, v3, s[34:35]
	s_add_u32 s28, s28, 1
	s_addc_u32 s29, s29, 0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v2, 0xffff, v1
	v_lshlrev_b64 v[4:5], s26, v[2:3]
	s_add_u32 s26, s26, 8
	s_addc_u32 s27, s27, 0
	v_or_b32_e32 v16, v4, v16
	s_cmp_lg_u32 s30, s28
	v_or_b32_e32 v17, v5, v17
	s_cbranch_scc1 .LBB1_278
	s_branch .LBB1_281
.LBB1_279:                              ;   in Loop: Header=BB1_260 Depth=2
	s_mov_b32 s9, 0
	s_branch .LBB1_282
.LBB1_280:                              ;   in Loop: Header=BB1_260 Depth=2
	v_mov_b64_e32 v[16:17], 0
.LBB1_281:                              ; %Flow600
                                        ;   in Loop: Header=BB1_260 Depth=2
	s_mov_b32 s9, 0
	s_cbranch_execnz .LBB1_283
.LBB1_282:                              ;   in Loop: Header=BB1_260 Depth=2
	global_load_dwordx2 v[16:17], v3, s[4:5]
	s_add_i32 s9, s30, -8
	s_add_u32 s4, s4, 8
	s_addc_u32 s5, s5, 0
.LBB1_283:                              ; %.loopexit27.i60
                                        ;   in Loop: Header=BB1_260 Depth=2
	s_cmp_gt_u32 s9, 7
	s_cbranch_scc1 .LBB1_287
; %bb.284:                              ;   in Loop: Header=BB1_260 Depth=2
	s_cmp_eq_u32 s9, 0
	s_cbranch_scc1 .LBB1_288
; %bb.285:                              ; %.preheader24.i61.preheader
                                        ;   in Loop: Header=BB1_260 Depth=2
	s_mov_b64 s[26:27], 0
	v_mov_b64_e32 v[18:19], 0
	s_mov_b64 s[28:29], 0
.LBB1_286:                              ; %.preheader24.i61
                                        ;   Parent Loop BB1_3 Depth=1
                                        ;     Parent Loop BB1_260 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_add_u32 s30, s4, s28
	s_addc_u32 s31, s5, s29
	global_load_ubyte v1, v3, s[30:31]
	s_add_u32 s28, s28, 1
	s_addc_u32 s29, s29, 0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v2, 0xffff, v1
	v_lshlrev_b64 v[4:5], s26, v[2:3]
	s_add_u32 s26, s26, 8
	s_addc_u32 s27, s27, 0
	v_or_b32_e32 v18, v4, v18
	s_cmp_lg_u32 s9, s28
	v_or_b32_e32 v19, v5, v19
	s_cbranch_scc1 .LBB1_286
	s_branch .LBB1_289
.LBB1_287:                              ;   in Loop: Header=BB1_260 Depth=2
                                        ; implicit-def: $vgpr18_vgpr19
	s_mov_b32 s30, 0
	s_branch .LBB1_290
.LBB1_288:                              ;   in Loop: Header=BB1_260 Depth=2
	v_mov_b64_e32 v[18:19], 0
.LBB1_289:                              ; %Flow595
                                        ;   in Loop: Header=BB1_260 Depth=2
	s_mov_b32 s30, 0
	s_cbranch_execnz .LBB1_291
.LBB1_290:                              ;   in Loop: Header=BB1_260 Depth=2
	global_load_dwordx2 v[18:19], v3, s[4:5]
	s_add_i32 s30, s9, -8
	s_add_u32 s4, s4, 8
	s_addc_u32 s5, s5, 0
.LBB1_291:                              ; %.loopexit25.i62
                                        ;   in Loop: Header=BB1_260 Depth=2
	s_cmp_gt_u32 s30, 7
	s_cbranch_scc1 .LBB1_295
; %bb.292:                              ;   in Loop: Header=BB1_260 Depth=2
	s_cmp_eq_u32 s30, 0
	s_cbranch_scc1 .LBB1_296
; %bb.293:                              ; %.preheader22.i63.preheader
                                        ;   in Loop: Header=BB1_260 Depth=2
	s_mov_b64 s[26:27], 0
	v_mov_b64_e32 v[20:21], 0
	s_mov_b64 s[28:29], 0
.LBB1_294:                              ; %.preheader22.i63
                                        ;   Parent Loop BB1_3 Depth=1
                                        ;     Parent Loop BB1_260 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_add_u32 s34, s4, s28
	s_addc_u32 s35, s5, s29
	global_load_ubyte v1, v3, s[34:35]
	s_add_u32 s28, s28, 1
	s_addc_u32 s29, s29, 0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v2, 0xffff, v1
	v_lshlrev_b64 v[4:5], s26, v[2:3]
	s_add_u32 s26, s26, 8
	s_addc_u32 s27, s27, 0
	v_or_b32_e32 v20, v4, v20
	s_cmp_lg_u32 s30, s28
	v_or_b32_e32 v21, v5, v21
	s_cbranch_scc1 .LBB1_294
	s_branch .LBB1_297
.LBB1_295:                              ;   in Loop: Header=BB1_260 Depth=2
	s_mov_b32 s9, 0
	s_branch .LBB1_298
.LBB1_296:                              ;   in Loop: Header=BB1_260 Depth=2
	v_mov_b64_e32 v[20:21], 0
.LBB1_297:                              ; %Flow590
                                        ;   in Loop: Header=BB1_260 Depth=2
	s_mov_b32 s9, 0
	s_cbranch_execnz .LBB1_299
.LBB1_298:                              ;   in Loop: Header=BB1_260 Depth=2
	global_load_dwordx2 v[20:21], v3, s[4:5]
	s_add_i32 s9, s30, -8
	s_add_u32 s4, s4, 8
	s_addc_u32 s5, s5, 0
.LBB1_299:                              ; %.loopexit23.i64
                                        ;   in Loop: Header=BB1_260 Depth=2
	s_cmp_gt_u32 s9, 7
	s_cbranch_scc1 .LBB1_303
; %bb.300:                              ;   in Loop: Header=BB1_260 Depth=2
	s_cmp_eq_u32 s9, 0
	s_cbranch_scc1 .LBB1_304
; %bb.301:                              ; %.preheader20.i65.preheader
                                        ;   in Loop: Header=BB1_260 Depth=2
	s_mov_b64 s[26:27], 0
	v_mov_b64_e32 v[22:23], 0
	s_mov_b64 s[28:29], 0
.LBB1_302:                              ; %.preheader20.i65
                                        ;   Parent Loop BB1_3 Depth=1
                                        ;     Parent Loop BB1_260 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_add_u32 s30, s4, s28
	s_addc_u32 s31, s5, s29
	global_load_ubyte v1, v3, s[30:31]
	s_add_u32 s28, s28, 1
	s_addc_u32 s29, s29, 0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v2, 0xffff, v1
	v_lshlrev_b64 v[4:5], s26, v[2:3]
	s_add_u32 s26, s26, 8
	s_addc_u32 s27, s27, 0
	v_or_b32_e32 v22, v4, v22
	s_cmp_lg_u32 s9, s28
	v_or_b32_e32 v23, v5, v23
	s_cbranch_scc1 .LBB1_302
	s_branch .LBB1_305
.LBB1_303:                              ;   in Loop: Header=BB1_260 Depth=2
                                        ; implicit-def: $vgpr22_vgpr23
	s_mov_b32 s30, 0
	s_branch .LBB1_306
.LBB1_304:                              ;   in Loop: Header=BB1_260 Depth=2
	v_mov_b64_e32 v[22:23], 0
.LBB1_305:                              ; %Flow585
                                        ;   in Loop: Header=BB1_260 Depth=2
	s_mov_b32 s30, 0
	s_cbranch_execnz .LBB1_307
.LBB1_306:                              ;   in Loop: Header=BB1_260 Depth=2
	global_load_dwordx2 v[22:23], v3, s[4:5]
	s_add_i32 s30, s9, -8
	s_add_u32 s4, s4, 8
	s_addc_u32 s5, s5, 0
.LBB1_307:                              ; %.loopexit21.i66
                                        ;   in Loop: Header=BB1_260 Depth=2
	s_cmp_gt_u32 s30, 7
	s_cbranch_scc1 .LBB1_311
; %bb.308:                              ;   in Loop: Header=BB1_260 Depth=2
	s_cmp_eq_u32 s30, 0
	s_cbranch_scc1 .LBB1_312
; %bb.309:                              ; %.preheader.i67.preheader
                                        ;   in Loop: Header=BB1_260 Depth=2
	s_mov_b64 s[26:27], 0
	v_mov_b64_e32 v[24:25], 0
	s_mov_b64 s[28:29], s[4:5]
.LBB1_310:                              ; %.preheader.i67
                                        ;   Parent Loop BB1_3 Depth=1
                                        ;     Parent Loop BB1_260 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	global_load_ubyte v1, v3, s[28:29]
	s_add_i32 s30, s30, -1
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v2, 0xffff, v1
	v_lshlrev_b64 v[4:5], s26, v[2:3]
	s_add_u32 s26, s26, 8
	s_addc_u32 s27, s27, 0
	s_add_u32 s28, s28, 1
	s_addc_u32 s29, s29, 0
	v_or_b32_e32 v24, v4, v24
	s_cmp_lg_u32 s30, 0
	v_or_b32_e32 v25, v5, v25
	s_cbranch_scc1 .LBB1_310
	s_branch .LBB1_313
.LBB1_311:                              ;   in Loop: Header=BB1_260 Depth=2
	s_branch .LBB1_314
.LBB1_312:                              ;   in Loop: Header=BB1_260 Depth=2
	v_mov_b64_e32 v[24:25], 0
.LBB1_313:                              ; %Flow580
                                        ;   in Loop: Header=BB1_260 Depth=2
	s_cbranch_execnz .LBB1_315
.LBB1_314:                              ;   in Loop: Header=BB1_260 Depth=2
	global_load_dwordx2 v[24:25], v3, s[4:5]
.LBB1_315:                              ; %.loopexit.i68
                                        ;   in Loop: Header=BB1_260 Depth=2
	v_readfirstlane_b32 s4, v49
	v_mov_b64_e32 v[4:5], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[4:5], s4, v49
	s_and_saveexec_b64 s[26:27], s[4:5]
	s_cbranch_execz .LBB1_321
; %bb.316:                              ;   in Loop: Header=BB1_260 Depth=2
	global_load_dwordx2 v[28:29], v3, s[20:21] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[4:5], v3, s[20:21] offset:40
	global_load_dwordx2 v[26:27], v3, s[20:21]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v1, v4, v28
	v_and_b32_e32 v2, v5, v29
	v_mul_lo_u32 v2, v2, 24
	v_mul_hi_u32 v4, v1, 24
	v_add_u32_e32 v5, v4, v2
	v_mul_lo_u32 v4, v1, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[4:5], v[26:27], 0, v[4:5]
	global_load_dwordx2 v[26:27], v[4:5], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[4:5], v3, v[26:29], s[20:21] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[4:5], v[28:29]
	s_and_saveexec_b64 s[28:29], vcc
	s_cbranch_execz .LBB1_320
; %bb.317:                              ; %.preheader3.i.i18.i75.preheader
                                        ;   in Loop: Header=BB1_260 Depth=2
	s_mov_b64 s[30:31], 0
.LBB1_318:                              ; %.preheader3.i.i18.i75
                                        ;   Parent Loop BB1_3 Depth=1
                                        ;     Parent Loop BB1_260 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_sleep 1
	global_load_dwordx2 v[26:27], v3, s[20:21] offset:40
	global_load_dwordx2 v[32:33], v3, s[20:21]
	v_mov_b64_e32 v[28:29], v[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v26, v28
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[4:5], s[34:35], v2, 24, v[32:33]
	v_and_b32_e32 v1, v27, v29
	v_mov_b32_e32 v2, v5
	v_mad_u64_u32 v[26:27], s[34:35], v1, 24, v[2:3]
	v_mov_b32_e32 v5, v26
	global_load_dwordx2 v[26:27], v[4:5], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[4:5], v3, v[26:29], s[20:21] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[4:5], v[28:29]
	s_or_b64 s[30:31], vcc, s[30:31]
	s_andn2_b64 exec, exec, s[30:31]
	s_cbranch_execnz .LBB1_318
; %bb.319:                              ; %Flow575
                                        ;   in Loop: Header=BB1_260 Depth=2
	s_or_b64 exec, exec, s[30:31]
.LBB1_320:                              ; %Flow577
                                        ;   in Loop: Header=BB1_260 Depth=2
	s_or_b64 exec, exec, s[28:29]
.LBB1_321:                              ; %.loopexit4.i.i13.i69
                                        ;   in Loop: Header=BB1_260 Depth=2
	s_or_b64 exec, exec, s[26:27]
	global_load_dwordx2 v[32:33], v3, s[20:21] offset:40
	global_load_dwordx4 v[26:29], v3, s[20:21]
	v_readfirstlane_b32 s26, v4
	v_readfirstlane_b32 s27, v5
	s_mov_b64 s[28:29], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s30, v32
	v_readfirstlane_b32 s31, v33
	s_and_b64 s[30:31], s[26:27], s[30:31]
	s_mul_i32 s9, s31, 24
	s_mul_hi_u32 s34, s30, 24
	s_add_i32 s35, s34, s9
	s_mul_i32 s34, s30, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[32:33], v[26:27], 0, s[34:35]
	s_and_saveexec_b64 s[34:35], s[4:5]
	s_cbranch_execz .LBB1_323
; %bb.322:                              ;   in Loop: Header=BB1_260 Depth=2
	v_mov_b64_e32 v[4:5], s[28:29]
	global_store_dwordx4 v[32:33], v[4:7], off offset:8
.LBB1_323:                              ;   in Loop: Header=BB1_260 Depth=2
	s_or_b64 exec, exec, s[34:35]
	s_lshl_b64 s[28:29], s[30:31], 12
	v_lshl_add_u64 v[4:5], v[28:29], 0, s[28:29]
	v_cmp_gt_u64_e64 s[28:29], s[22:23], 56
	s_and_b64 s[28:29], s[28:29], exec
	s_cselect_b32 s9, 0, 2
	s_lshl_b32 s28, s24, 2
	s_add_i32 s28, s28, 28
	v_and_b32_e32 v1, 0xffffff1f, v10
	s_and_b32 s28, s28, 0x1e0
	v_or_b32_e32 v1, s9, v1
	v_or_b32_e32 v10, s28, v1
	v_readfirstlane_b32 s28, v4
	v_readfirstlane_b32 s29, v5
	s_nop 4
	global_store_dwordx4 v30, v[10:13], s[28:29]
	global_store_dwordx4 v30, v[14:17], s[28:29] offset:16
	global_store_dwordx4 v30, v[18:21], s[28:29] offset:32
	global_store_dwordx4 v30, v[22:25], s[28:29] offset:48
	s_and_saveexec_b64 s[28:29], s[4:5]
	s_cbranch_execz .LBB1_331
; %bb.324:                              ;   in Loop: Header=BB1_260 Depth=2
	global_load_dwordx2 v[18:19], v3, s[20:21] offset:32 sc0 sc1
	global_load_dwordx2 v[10:11], v3, s[20:21] offset:40
	v_mov_b32_e32 v16, s26
	v_mov_b32_e32 v17, s27
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s30, v10
	v_readfirstlane_b32 s31, v11
	s_and_b64 s[30:31], s[30:31], s[26:27]
	s_mul_i32 s9, s31, 24
	s_mul_hi_u32 s31, s30, 24
	s_mul_i32 s30, s30, 24
	s_add_i32 s31, s31, s9
	v_lshl_add_u64 v[14:15], v[26:27], 0, s[30:31]
	global_store_dwordx2 v[14:15], v[18:19], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[12:13], v3, v[16:19], s[20:21] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[12:13], v[18:19]
	s_and_saveexec_b64 s[30:31], vcc
	s_cbranch_execz .LBB1_327
; %bb.325:                              ; %.preheader1.i.i16.i73.preheader
                                        ;   in Loop: Header=BB1_260 Depth=2
	s_mov_b64 s[34:35], 0
.LBB1_326:                              ; %.preheader1.i.i16.i73
                                        ;   Parent Loop BB1_3 Depth=1
                                        ;     Parent Loop BB1_260 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_sleep 1
	global_store_dwordx2 v[14:15], v[12:13], off
	v_mov_b32_e32 v10, s26
	v_mov_b32_e32 v11, s27
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v3, v[10:13], s[20:21] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[10:11], v[12:13]
	s_or_b64 s[34:35], vcc, s[34:35]
	v_mov_b64_e32 v[12:13], v[10:11]
	s_andn2_b64 exec, exec, s[34:35]
	s_cbranch_execnz .LBB1_326
.LBB1_327:                              ; %Flow573
                                        ;   in Loop: Header=BB1_260 Depth=2
	s_or_b64 exec, exec, s[30:31]
	global_load_dwordx2 v[10:11], v3, s[20:21] offset:16
	s_mov_b64 s[34:35], exec
	v_mbcnt_lo_u32_b32 v1, s34, 0
	v_mbcnt_hi_u32_b32 v1, s35, v1
	v_cmp_eq_u32_e32 vcc, 0, v1
	s_and_saveexec_b64 s[30:31], vcc
	s_cbranch_execz .LBB1_329
; %bb.328:                              ;   in Loop: Header=BB1_260 Depth=2
	s_bcnt1_i32_b64 s9, s[34:35]
	v_mov_b32_e32 v2, s9
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[10:11], v[2:3], off offset:8 sc1
.LBB1_329:                              ;   in Loop: Header=BB1_260 Depth=2
	s_or_b64 exec, exec, s[30:31]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[12:13], v[10:11], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[12:13]
	s_cbranch_vccnz .LBB1_331
; %bb.330:                              ;   in Loop: Header=BB1_260 Depth=2
	global_load_dword v2, v[10:11], off offset:24
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v1, 0xffffff, v2
	s_nop 0
	v_readfirstlane_b32 s9, v1
	s_mov_b32 m0, s9
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[12:13], v[2:3], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB1_331:                              ; %Flow574
                                        ;   in Loop: Header=BB1_260 Depth=2
	s_or_b64 exec, exec, s[28:29]
	v_mov_b32_e32 v31, v3
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[30:31]
	s_branch .LBB1_335
.LBB1_332:                              ;   in Loop: Header=BB1_335 Depth=3
	s_or_b64 exec, exec, s[28:29]
	v_readfirstlane_b32 s9, v1
	s_cmp_eq_u32 s9, 0
	s_cbranch_scc1 .LBB1_334
; %bb.333:                              ;   in Loop: Header=BB1_335 Depth=3
	s_sleep 1
	s_cbranch_execnz .LBB1_335
	s_branch .LBB1_337
.LBB1_334:                              ;   in Loop: Header=BB1_260 Depth=2
	s_branch .LBB1_337
.LBB1_335:                              ;   Parent Loop BB1_3 Depth=1
                                        ;     Parent Loop BB1_260 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	v_mov_b32_e32 v1, 1
	s_and_saveexec_b64 s[28:29], s[4:5]
	s_cbranch_execz .LBB1_332
; %bb.336:                              ;   in Loop: Header=BB1_335 Depth=3
	global_load_dword v1, v[32:33], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v1, 1, v1
	s_branch .LBB1_332
.LBB1_337:                              ;   in Loop: Header=BB1_260 Depth=2
	global_load_dwordx4 v[10:13], v[4:5], off
	s_and_saveexec_b64 s[28:29], s[4:5]
	s_cbranch_execz .LBB1_259
; %bb.338:                              ;   in Loop: Header=BB1_260 Depth=2
	global_load_dwordx2 v[4:5], v3, s[20:21] offset:40
	global_load_dwordx2 v[16:17], v3, s[20:21] offset:24 sc0 sc1
	global_load_dwordx2 v[18:19], v3, s[20:21]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[12:13], v[4:5], 0, 1
	v_lshl_add_u64 v[20:21], v[12:13], 0, s[26:27]
	v_cmp_eq_u64_e32 vcc, 0, v[20:21]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v14, v16
	v_mov_b32_e32 v15, v17
	v_cndmask_b32_e32 v13, v21, v13, vcc
	v_cndmask_b32_e32 v12, v20, v12, vcc
	v_and_b32_e32 v1, v13, v5
	v_and_b32_e32 v2, v12, v4
	v_mul_lo_u32 v1, v1, 24
	v_mul_hi_u32 v5, v2, 24
	v_mul_lo_u32 v4, v2, 24
	v_add_u32_e32 v5, v5, v1
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[4:5], v[18:19], 0, v[4:5]
	global_store_dwordx2 v[4:5], v[16:17], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[14:15], v3, v[12:15], s[20:21] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[14:15], v[16:17]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB1_259
; %bb.339:                              ; %.preheader.i.i15.i72.preheader
                                        ;   in Loop: Header=BB1_260 Depth=2
	s_mov_b64 s[4:5], 0
.LBB1_340:                              ; %.preheader.i.i15.i72
                                        ;   Parent Loop BB1_3 Depth=1
                                        ;     Parent Loop BB1_260 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_sleep 1
	global_store_dwordx2 v[4:5], v[14:15], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[16:17], v3, v[12:15], s[20:21] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[16:17], v[14:15]
	s_or_b64 s[4:5], vcc, s[4:5]
	v_mov_b64_e32 v[14:15], v[16:17]
	s_andn2_b64 exec, exec, s[4:5]
	s_cbranch_execnz .LBB1_340
	s_branch .LBB1_259
.LBB1_341:                              ; %Flow613
                                        ;   in Loop: Header=BB1_3 Depth=1
	s_branch .LBB1_2
.LBB1_342:                              ;   in Loop: Header=BB1_3 Depth=1
	s_cbranch_execz .LBB1_2
; %bb.343:                              ;   in Loop: Header=BB1_3 Depth=1
	v_readfirstlane_b32 s4, v49
	v_mov_b64_e32 v[4:5], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[4:5], s4, v49
	s_and_saveexec_b64 s[10:11], s[4:5]
	s_cbranch_execz .LBB1_349
; %bb.344:                              ;   in Loop: Header=BB1_3 Depth=1
	global_load_dwordx2 v[12:13], v3, s[20:21] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[4:5], v3, s[20:21] offset:40
	global_load_dwordx2 v[10:11], v3, s[20:21]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v1, v4, v12
	v_and_b32_e32 v2, v5, v13
	v_mul_lo_u32 v2, v2, 24
	v_mul_hi_u32 v4, v1, 24
	v_add_u32_e32 v5, v4, v2
	v_mul_lo_u32 v4, v1, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[4:5], v[10:11], 0, v[4:5]
	global_load_dwordx2 v[10:11], v[4:5], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[4:5], v3, v[10:13], s[20:21] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[4:5], v[12:13]
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB1_348
; %bb.345:                              ; %.preheader3.i.i.i82.preheader
                                        ;   in Loop: Header=BB1_3 Depth=1
	s_mov_b64 s[24:25], 0
.LBB1_346:                              ; %.preheader3.i.i.i82
                                        ;   Parent Loop BB1_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_load_dwordx2 v[10:11], v3, s[20:21] offset:40
	global_load_dwordx2 v[14:15], v3, s[20:21]
	v_mov_b64_e32 v[12:13], v[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v10, v12
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[4:5], s[26:27], v2, 24, v[14:15]
	v_and_b32_e32 v1, v11, v13
	v_mov_b32_e32 v2, v5
	v_mad_u64_u32 v[10:11], s[26:27], v1, 24, v[2:3]
	v_mov_b32_e32 v5, v10
	global_load_dwordx2 v[10:11], v[4:5], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[4:5], v3, v[10:13], s[20:21] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[4:5], v[12:13]
	s_or_b64 s[24:25], vcc, s[24:25]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBB1_346
; %bb.347:                              ; %Flow626
                                        ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[24:25]
.LBB1_348:                              ; %Flow628
                                        ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[22:23]
.LBB1_349:                              ; %.loopexit4.i.i.i76
                                        ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[10:11]
	global_load_dwordx2 v[10:11], v3, s[20:21] offset:40
	global_load_dwordx4 v[12:15], v3, s[20:21]
	v_readfirstlane_b32 s22, v4
	v_readfirstlane_b32 s23, v5
	s_mov_b64 s[10:11], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s24, v10
	v_readfirstlane_b32 s25, v11
	s_and_b64 s[24:25], s[22:23], s[24:25]
	s_mul_i32 s9, s25, 24
	s_mul_hi_u32 s26, s24, 24
	s_add_i32 s27, s26, s9
	s_mul_i32 s26, s24, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[16:17], v[12:13], 0, s[26:27]
	s_and_saveexec_b64 s[26:27], s[4:5]
	s_cbranch_execz .LBB1_351
; %bb.350:                              ;   in Loop: Header=BB1_3 Depth=1
	v_mov_b64_e32 v[4:5], s[10:11]
	global_store_dwordx4 v[16:17], v[4:7], off offset:8
.LBB1_351:                              ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[26:27]
	s_lshl_b64 s[10:11], s[24:25], 12
	v_lshl_add_u64 v[4:5], v[14:15], 0, s[10:11]
	v_and_or_b32 v8, v8, s40, 34
	v_mov_b32_e32 v10, v3
	v_mov_b32_e32 v11, v3
	v_readfirstlane_b32 s24, v4
	v_readfirstlane_b32 s25, v5
	s_mov_b32 s9, s8
	s_mov_b32 s10, s8
	s_mov_b32 s11, s8
	s_nop 1
	global_store_dwordx4 v30, v[8:11], s[24:25]
	s_nop 1
	v_mov_b64_e32 v[8:9], s[8:9]
	v_mov_b64_e32 v[10:11], s[10:11]
	global_store_dwordx4 v30, v[8:11], s[24:25] offset:16
	global_store_dwordx4 v30, v[8:11], s[24:25] offset:32
	global_store_dwordx4 v30, v[8:11], s[24:25] offset:48
	s_and_saveexec_b64 s[10:11], s[4:5]
	s_cbranch_execz .LBB1_359
; %bb.352:                              ;   in Loop: Header=BB1_3 Depth=1
	global_load_dwordx2 v[20:21], v3, s[20:21] offset:32 sc0 sc1
	global_load_dwordx2 v[4:5], v3, s[20:21] offset:40
	v_mov_b32_e32 v18, s22
	v_mov_b32_e32 v19, s23
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s24, v4
	v_readfirstlane_b32 s25, v5
	s_and_b64 s[24:25], s[24:25], s[22:23]
	s_mul_i32 s9, s25, 24
	s_mul_hi_u32 s25, s24, 24
	s_mul_i32 s24, s24, 24
	s_add_i32 s25, s25, s9
	v_lshl_add_u64 v[4:5], v[12:13], 0, s[24:25]
	global_store_dwordx2 v[4:5], v[20:21], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v3, v[18:21], s[20:21] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[10:11], v[20:21]
	s_and_saveexec_b64 s[24:25], vcc
	s_cbranch_execz .LBB1_355
; %bb.353:                              ; %.preheader1.i.i.i80.preheader
                                        ;   in Loop: Header=BB1_3 Depth=1
	s_mov_b64 s[26:27], 0
.LBB1_354:                              ; %.preheader1.i.i.i80
                                        ;   Parent Loop BB1_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[4:5], v[10:11], off
	v_mov_b32_e32 v8, s22
	v_mov_b32_e32 v9, s23
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[8:9], v3, v[8:11], s[20:21] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[8:9], v[10:11]
	s_or_b64 s[26:27], vcc, s[26:27]
	v_mov_b64_e32 v[10:11], v[8:9]
	s_andn2_b64 exec, exec, s[26:27]
	s_cbranch_execnz .LBB1_354
.LBB1_355:                              ; %Flow624
                                        ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[24:25]
	global_load_dwordx2 v[4:5], v3, s[20:21] offset:16
	s_mov_b64 s[26:27], exec
	v_mbcnt_lo_u32_b32 v1, s26, 0
	v_mbcnt_hi_u32_b32 v1, s27, v1
	v_cmp_eq_u32_e32 vcc, 0, v1
	s_and_saveexec_b64 s[24:25], vcc
	s_cbranch_execz .LBB1_357
; %bb.356:                              ;   in Loop: Header=BB1_3 Depth=1
	s_bcnt1_i32_b64 s9, s[26:27]
	v_mov_b32_e32 v2, s9
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[4:5], v[2:3], off offset:8 sc1
.LBB1_357:                              ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[24:25]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[8:9], v[4:5], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[8:9]
	s_cbranch_vccnz .LBB1_359
; %bb.358:                              ;   in Loop: Header=BB1_3 Depth=1
	global_load_dword v2, v[4:5], off offset:24
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v1, 0xffffff, v2
	s_nop 0
	v_readfirstlane_b32 s9, v1
	s_mov_b32 m0, s9
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[8:9], v[2:3], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB1_359:                              ; %Flow625
                                        ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[10:11]
	s_branch .LBB1_363
.LBB1_360:                              ;   in Loop: Header=BB1_363 Depth=2
	s_or_b64 exec, exec, s[10:11]
	v_readfirstlane_b32 s9, v1
	s_cmp_eq_u32 s9, 0
	s_cbranch_scc1 .LBB1_362
; %bb.361:                              ;   in Loop: Header=BB1_363 Depth=2
	s_sleep 1
	s_cbranch_execnz .LBB1_363
	s_branch .LBB1_365
.LBB1_362:                              ;   in Loop: Header=BB1_3 Depth=1
	s_branch .LBB1_365
.LBB1_363:                              ;   Parent Loop BB1_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v1, 1
	s_and_saveexec_b64 s[10:11], s[4:5]
	s_cbranch_execz .LBB1_360
; %bb.364:                              ;   in Loop: Header=BB1_363 Depth=2
	global_load_dword v1, v[16:17], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v1, 1, v1
	s_branch .LBB1_360
.LBB1_365:                              ;   in Loop: Header=BB1_3 Depth=1
	s_and_b64 exec, exec, s[4:5]
	s_cbranch_execz .LBB1_2
; %bb.366:                              ;   in Loop: Header=BB1_3 Depth=1
	global_load_dwordx2 v[4:5], v3, s[20:21] offset:40
	global_load_dwordx2 v[12:13], v3, s[20:21] offset:24 sc0 sc1
	global_load_dwordx2 v[14:15], v3, s[20:21]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[8:9], v[4:5], 0, 1
	v_lshl_add_u64 v[16:17], v[8:9], 0, s[22:23]
	v_cmp_eq_u64_e32 vcc, 0, v[16:17]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v10, v12
	v_mov_b32_e32 v11, v13
	v_cndmask_b32_e32 v9, v17, v9, vcc
	v_cndmask_b32_e32 v8, v16, v8, vcc
	v_and_b32_e32 v1, v9, v5
	v_and_b32_e32 v2, v8, v4
	v_mul_lo_u32 v1, v1, 24
	v_mul_hi_u32 v5, v2, 24
	v_mul_lo_u32 v4, v2, 24
	v_add_u32_e32 v5, v5, v1
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[4:5], v[14:15], 0, v[4:5]
	global_store_dwordx2 v[4:5], v[12:13], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v3, v[8:11], s[20:21] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[10:11], v[12:13]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB1_2
; %bb.367:                              ; %.preheader.i.i.i79.preheader
                                        ;   in Loop: Header=BB1_3 Depth=1
	s_mov_b64 s[4:5], 0
.LBB1_368:                              ; %.preheader.i.i.i79
                                        ;   Parent Loop BB1_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[4:5], v[10:11], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[12:13], v3, v[8:11], s[20:21] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[12:13], v[10:11]
	s_or_b64 s[4:5], vcc, s[4:5]
	v_mov_b64_e32 v[10:11], v[12:13]
	s_andn2_b64 exec, exec, s[4:5]
	s_cbranch_execnz .LBB1_368
	s_branch .LBB1_2
.LBB1_369:                              ; %._crit_edge
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z8vecMult2Pdmi
		.amdhsa_group_segment_fixed_size 256
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 280
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
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 50
		.amdhsa_next_free_sgpr 41
		.amdhsa_accum_offset 52
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
.Lfunc_end1:
	.size	_Z8vecMult2Pdmi, .Lfunc_end1-_Z8vecMult2Pdmi
                                        ; -- End function
	.set _Z8vecMult2Pdmi.num_vgpr, 50
	.set _Z8vecMult2Pdmi.num_agpr, 0
	.set _Z8vecMult2Pdmi.numbered_sgpr, 41
	.set _Z8vecMult2Pdmi.private_seg_size, 0
	.set _Z8vecMult2Pdmi.uses_vcc, 1
	.set _Z8vecMult2Pdmi.uses_flat_scratch, 0
	.set _Z8vecMult2Pdmi.has_dyn_sized_stack, 0
	.set _Z8vecMult2Pdmi.has_recursion, 0
	.set _Z8vecMult2Pdmi.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 13576
; TotalNumSgprs: 47
; NumVgprs: 50
; NumAgprs: 0
; TotalNumVgprs: 50
; ScratchSize: 0
; MemoryBound: 1
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 256 bytes/workgroup (compile time only)
; SGPRBlocks: 5
; VGPRBlocks: 6
; NumSGPRsForWavesPerEU: 47
; NumVGPRsForWavesPerEU: 50
; AccumOffset: 52
; Occupancy: 8
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 12
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.type	.str,@object                    ; @.str
	.section	.rodata.str1.1,"aMS",@progbits,1
.str:
	.asciz	"global:%d||%.2lf||%.2lf||\n"
	.size	.str, 27

	.type	.str.1,@object                  ; @.str.1
.str.1:
	.asciz	"-------------------------------------------\n"
	.size	.str.1, 45

	.type	__hip_cuid_e512ba6ea0c9d1f9,@object ; @__hip_cuid_e512ba6ea0c9d1f9
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_e512ba6ea0c9d1f9
__hip_cuid_e512ba6ea0c9d1f9:
	.byte	0                               ; 0x0
	.size	__hip_cuid_e512ba6ea0c9d1f9, 1

	.ident	"AMD clang version 20.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.0.1 25314 f4087f6b428f0e6f575ebac8a8a724dab123d06e)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_e512ba6ea0c9d1f9
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
      - .offset:         16
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         20
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         24
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         28
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         30
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         32
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         34
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         36
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         38
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         56
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         64
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         72
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         80
        .size:           2
        .value_kind:     hidden_grid_dims
    .group_segment_fixed_size: 4224
    .kernarg_segment_align: 8
    .kernarg_segment_size: 272
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z6matDetPdS_
    .private_segment_fixed_size: 0
    .sgpr_count:     17
    .sgpr_spill_count: 0
    .symbol:         _Z6matDetPdS_.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     54
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .agpr_count:     0
    .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .offset:         8
        .size:           8
        .value_kind:     by_value
      - .offset:         16
        .size:           4
        .value_kind:     by_value
      - .offset:         24
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         28
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         32
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         36
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         38
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         40
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         42
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         44
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         46
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         64
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         72
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         80
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         88
        .size:           2
        .value_kind:     hidden_grid_dims
      - .offset:         104
        .size:           8
        .value_kind:     hidden_hostcall_buffer
    .group_segment_fixed_size: 256
    .kernarg_segment_align: 8
    .kernarg_segment_size: 280
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z8vecMult2Pdmi
    .private_segment_fixed_size: 0
    .sgpr_count:     47
    .sgpr_spill_count: 0
    .symbol:         _Z8vecMult2Pdmi.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     50
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
