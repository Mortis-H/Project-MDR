.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6

; ============================================================================
; Kernel #1
; ============================================================================
.text
	.protected	_Z9vectorAddPKfS0_Pfi   ; -- Begin function _Z9vectorAddPKfS0_Pfi
	.globl	_Z9vectorAddPKfS0_Pfi
	.p2align	8
	.type	_Z9vectorAddPKfS0_Pfi,@function
_Z9vectorAddPKfS0_Pfi:                  ; @_Z9vectorAddPKfS0_Pfi
; %bb.0:
	s_load_dword s3, s[0:1], 0x2c
	s_load_dword s4, s[0:1], 0x18
	s_waitcnt lgkmcnt(0)
	s_and_b32 s3, s3, 0xffff
	s_mul_i32 s2, s2, s3
	v_add_u32_e32 v0, s2, v0
	v_cmp_gt_i32_e32 vcc, s4, v0
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_2
; %bb.1:
	s_load_dwordx4 s[4:7], s[0:1], 0x0
	s_load_dwordx2 s[2:3], s[0:1], 0x10
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshlrev_b64 v[0:1], 2, v[0:1]
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[4:5], s[4:5], 0, v[0:1]
	v_lshl_add_u64 v[2:3], s[6:7], 0, v[0:1]
	global_load_dword v6, v[4:5], off
	global_load_dword v7, v[2:3], off
	v_lshl_add_u64 v[0:1], s[2:3], 0, v[0:1]
	s_waitcnt vmcnt(0)
	v_add_f32_e32 v2, v6, v7
	global_store_dword v[0:1], v2, off
.LBB0_2:
	s_endpgm

; ============================================================================
; Kernel #2
; ============================================================================
.text
	.protected	_Z9scalarOpsPii         ; -- Begin function _Z9scalarOpsPii
	.globl	_Z9scalarOpsPii
	.p2align	8
	.type	_Z9scalarOpsPii,@function
_Z9scalarOpsPii:                        ; @_Z9scalarOpsPii
; %bb.0:
	s_load_dword s3, s[0:1], 0x1c
	s_load_dword s4, s[0:1], 0x8
	s_waitcnt lgkmcnt(0)
	s_and_b32 s3, s3, 0xffff
	s_mul_i32 s2, s2, s3
	v_add_u32_e32 v0, s2, v0
	v_cmp_gt_i32_e32 vcc, s4, v0
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_2
; %bb.1:
	s_load_dwordx2 s[0:1], s[0:1], 0x0
	v_add_u32_e32 v1, 1, v0
	v_and_b32_e32 v2, v1, v0
	v_or_b32_e32 v3, v1, v0
	v_ashrrev_i32_e32 v4, 2, v0
	v_add_u32_e32 v5, 7, v0
	v_xor_b32_e32 v1, v1, v0
	v_mul_lo_u32 v5, v5, v0
	v_add3_u32 v2, v2, v4, v3
	v_add3_u32 v2, v2, v1, v5
	v_ashrrev_i32_e32 v1, 31, v0
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[0:1], v[0:1], 2, s[0:1]
	global_store_dword v[0:1], v2, off
.LBB0_2:
	s_endpgm

; ============================================================================
; Kernel #3
; ============================================================================
.text
	.protected	_Z9memoryOpsPKiPii      ; -- Begin function _Z9memoryOpsPKiPii
	.globl	_Z9memoryOpsPKiPii
	.p2align	8
	.type	_Z9memoryOpsPKiPii,@function
_Z9memoryOpsPKiPii:                     ; @_Z9memoryOpsPKiPii
; %bb.0:
	s_load_dword s3, s[0:1], 0x24
	s_load_dword s4, s[0:1], 0x10
	s_waitcnt lgkmcnt(0)
	s_and_b32 s3, s3, 0xffff
	s_mul_i32 s2, s2, s3
	v_add_u32_e32 v0, s2, v0
	v_cmp_gt_i32_e32 vcc, s4, v0
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_2
; %bb.1:
	s_load_dwordx4 s[0:3], s[0:1], 0x0
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshlrev_b64 v[0:1], 2, v[0:1]
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[2:3], s[0:1], 0, v[0:1]
	global_load_dword v2, v[2:3], off
	v_lshl_add_u64 v[0:1], s[2:3], 0, v[0:1]
	s_waitcnt vmcnt(0)
	v_lshl_or_b32 v2, v2, 1, 1
	global_store_dword v[0:1], v2, off
.LBB0_2:
	s_endpgm

; ============================================================================
; Metadata for Kernel #1
; ============================================================================
.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z9vectorAddPKfS0_Pfi
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 288
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
		.amdhsa_next_free_vgpr 8
		.amdhsa_next_free_sgpr 8
		.amdhsa_accum_offset 8
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

; ============================================================================
; Metadata for Kernel #2
; ============================================================================
.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z9scalarOpsPii
		.amdhsa_group_segment_fixed_size 0
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
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 6
		.amdhsa_next_free_sgpr 5
		.amdhsa_accum_offset 8
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

; ============================================================================
; Metadata for Kernel #3
; ============================================================================
.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z9memoryOpsPKiPii
		.amdhsa_group_segment_fixed_size 0
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
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 5
		.amdhsa_accum_offset 4
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
