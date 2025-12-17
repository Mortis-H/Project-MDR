	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.text
	.protected	_Z17conditionalKernelPKiPii ; -- Begin function _Z17conditionalKernelPKiPii
	.globl	_Z17conditionalKernelPKiPii
	.p2align	8
	.type	_Z17conditionalKernelPKiPii,@function
_Z17conditionalKernelPKiPii:            ; @_Z17conditionalKernelPKiPii
	s_load_dword s3, s[0:1], 0x24
	s_load_dword s4, s[0:1], 0x10
	s_waitcnt lgkmcnt(0)
	s_and_b32 s3, s3, 0xffff
	s_mul_i32 s2, s2, s3
	v_add_u32_e32 v0, s2, v0
	v_cmp_gt_i32_e32 vcc, s4, v0
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_6
	s_load_dwordx4 s[0:3], s[0:1], 0x0
	v_ashrrev_i32_e32 v1, 31, v0
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v2, s0
	v_mov_b32_e32 v3, s1
	v_lshl_add_u64 v[2:3], v[0:1], 2, v[2:3]
	global_load_dword v3, v[2:3], off
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v2, 1, v3
	v_cmp_eq_u32_e32 vcc, 1, v2
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	v_mad_u64_u32 v[2:3], s[4:5], v3, 3, 1
	s_andn2_saveexec_b64 s[0:1], s[0:1]
	v_lshlrev_b32_e32 v2, 1, v3
	s_or_b64 exec, exec, s[0:1]
	v_mov_b32_e32 v4, s2
	v_mov_b32_e32 v5, s3
	v_lshl_add_u64 v[0:1], v[0:1], 2, v[4:5]
	global_store_dword v[0:1], v2, off
.LBB0_6:
	s_endpgm
                                        ; implicit-def: $vgpr2
                                        ; implicit-def: $vgpr3
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z17conditionalKernelPKiPii
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
		.amdhsa_next_free_vgpr 6
		.amdhsa_next_free_sgpr 6
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
	.size	_Z17conditionalKernelPKiPii, .Lfunc_end0-_Z17conditionalKernelPKiPii
                                        ; -- End function
	.set _Z17conditionalKernelPKiPii.num_vgpr, 6
	.set _Z17conditionalKernelPKiPii.num_agpr, 0
	.set _Z17conditionalKernelPKiPii.numbered_sgpr, 6
	.set _Z17conditionalKernelPKiPii.private_seg_size, 0
	.set _Z17conditionalKernelPKiPii.uses_vcc, 1
	.set _Z17conditionalKernelPKiPii.uses_flat_scratch, 0
	.set _Z17conditionalKernelPKiPii.has_dyn_sized_stack, 0
	.set _Z17conditionalKernelPKiPii.has_recursion, 0
	.set _Z17conditionalKernelPKiPii.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.type	__hip_cuid_df040c0751af06d2,@object ; @__hip_cuid_df040c0751af06d2
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_df040c0751af06d2
__hip_cuid_df040c0751af06d2:
	.byte	0                               ; 0x0
	.size	__hip_cuid_df040c0751af06d2, 1

	.ident	"AMD clang version 20.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.0.1 25314 f4087f6b428f0e6f575ebac8a8a724dab123d06e)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_df040c0751af06d2

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
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 280
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z17conditionalKernelPKiPii
    .private_segment_fixed_size: 0
    .sgpr_count:     12
    .sgpr_spill_count: 0
    .symbol:         _Z17conditionalKernelPKiPii.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     6
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata


