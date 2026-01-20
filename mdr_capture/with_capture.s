	.text
	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.protected	_Z9vectorAddPfS_S_i
	.globl	_Z9vectorAddPfS_S_i
	.p2align	8
	.type	_Z9vectorAddPfS_S_i,@function
_Z9vectorAddPfS_S_i:
	s_load_dword s4, s[0:1], 0x18
	s_waitcnt lgkmcnt(0)
	s_and_b32 s4, s4, 0xffff
	v_cmp_gt_u32_e32 vcc, s4, v0
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_2
.LBB0_1:
	s_load_dwordx2 s[6:7], s[0:1], 0x0
	v_lshlrev_b32_e32 v1, 2, v0
	s_load_dwordx2 s[8:9], s[0:1], 0x8
	s_load_dwordx2 s[4:5], s[0:1], 0x10
	s_waitcnt lgkmcnt(0)
	global_load_dword v2, v1, s[6:7]
	global_load_dword v3, v1, s[8:9]
	s_waitcnt vmcnt(0)
; @CAPTURE cond=tid_eq(0) reg=v2,v3 type=f32,f32
	v_add_f32_e32 v2, v2, v3
; @CAPTURE cond=tid_eq(0) reg=v2 expr="v2*2.0" type=f32,f32
	global_store_dword v1, v2, s[4:5]
.LBB0_2:
	s_endpgm
	.section	.rodata,#alloc
	.p2align	6, 0x0
	.amdhsa_kernel _Z9vectorAddPfS_S_i
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 24
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_private_segment_buffer 0
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 0
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 10
		.amdhsa_reserve_vcc 1
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_reserve_xnack_mask 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 0
		.amdhsa_shared_vgpr_count 0
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
	.size	_Z9vectorAddPfS_S_i, .Lfunc_end0-_Z9vectorAddPfS_S_i

	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
        .name:           A
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
        .name:           B
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
        .name:           C
      - .offset:         24
        .size:           4
        .value_kind:     by_value
        .name:           n
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 28
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _Z9vectorAddPfS_S_i
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .sgpr_spill_count: 0
    .symbol:         _Z9vectorAddPfS_S_i.kd
    .vgpr_count:     4
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.version:
  - 1
  - 2
...
	.end_amdgpu_metadata
