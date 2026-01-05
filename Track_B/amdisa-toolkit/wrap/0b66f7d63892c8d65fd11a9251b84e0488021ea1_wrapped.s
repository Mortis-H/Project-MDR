.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.amdhsa_code_object_version 6
.text
.protected	vec_add
.globl	vec_add
.p2align	8
.type	vec_add,@function

vec_add:        ; @vec_add
; %bb.0:
	s_load_dword s8, s[4:5], 0x34
	s_load_dwordx4 s[0:3], s[4:5], 0x18
	v_bfe_u32 v1, v0, 10, 10
	v_and_b32_e32 v0, 0x3ff, v0
	s_waitcnt lgkmcnt(0)
	s_lshr_b32 s3, s8, 16
	s_mul_i32 s7, s7, s3
	s_and_b32 s8, s8, 0xffff
	v_add_u32_e32 v1, s7, v1
	s_mul_i32 s6, s6, s8
	v_mul_lo_u32 v1, v1, s0
	v_add3_u32 v0, s6, v0, v1
	s_mul_i32 s0, s1, s0
	v_add_u32_e32 v2, s2, v0
	v_cmp_gt_i32_e32 vcc, s0, v2
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_2
; %bb.1:
	s_load_dwordx4 s[0:3], s[4:5], 0x0
	s_load_dwordx2 s[6:7], s[4:5], 0x10
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshlrev_b64 v[2:3], 2, v[2:3]
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v1, s3
	v_add_co_u32_e32 v4, vcc, s2, v2
	v_addc_co_u32_e32 v5, vcc, v1, v3, vcc
	v_mov_b32_e32 v1, s1
	v_add_co_u32_e32 v2, vcc, s0, v2
	v_addc_co_u32_e32 v3, vcc, v1, v3, vcc
	global_load_dword v6, v[2:3], off
	global_load_dword v7, v[4:5], off
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshlrev_b64 v[0:1], 2, v[0:1]
	v_mov_b32_e32 v2, s7
	v_add_co_u32_e32 v0, vcc, s6, v0
	v_addc_co_u32_e32 v1, vcc, v2, v1, vcc
	s_waitcnt vmcnt(0)
	v_add_f32_e32 v2, v6, v7
	global_store_dword v[0:1], v2, off
.LBB0_2:
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel vec_add
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 128
		.amdhsa_user_sgpr_count 6
		.amdhsa_user_sgpr_dispatch_ptr 1
		.amdhsa_user_sgpr_queue_ptr 1
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
		.amdhsa_system_vgpr_workitem_id 2
		.amdhsa_next_free_vgpr 8
		.amdhsa_next_free_sgpr 9
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
	.text
.Lfunc_end0:
	.size	vec_add, .Lfunc_end0-vec_add

	.amdgpu_metadata
---
amdhsa.version:
  - 1
  - 2
amdhsa.kernels:
  - .name:           vec_add
    .symbol:         vec_add.kd
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .kernarg_segment_size: 128
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .sgpr_count:     9
    .vgpr_count:     8
    .max_flat_workgroup_size: 256
    .wavefront_size: 64
    .sgpr_spill_count: 0
    .vgpr_spill_count: 0
    .args:
      - .name:           arg0
        .type_name:      'float*'
        .size:           8
        .offset:         0
        .value_kind:     global_buffer
        .address_space:  global
      - .name:           arg1
        .type_name:      'float*'
        .size:           8
        .offset:         8
        .value_kind:     global_buffer
        .address_space:  global
      - .name:           arg2
        .type_name:      'float*'
        .size:           8
        .offset:         16
        .value_kind:     global_buffer
        .address_space:  global
      - .name:           arg3
        .type_name:      'int'
        .size:           4
        .offset:         24
        .value_kind:     by_value
      - .name:           arg4
        .type_name:      'int'
        .size:           4
        .offset:         28
        .value_kind:     by_value
      - .name:           arg5
        .type_name:      'int'
        .size:           4
        .offset:         32
        .value_kind:     by_value
      - .offset:         52
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         54
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         56
        .size:           2
        .value_kind:     hidden_group_size_z
...
	.end_amdgpu_metadata

