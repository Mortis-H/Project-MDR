
# __CLANG_OFFLOAD_BUNDLE____START__ hip-amdgcn-amd-amdhsa--gfx950
	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
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
	.text
.Lfunc_end0:
	.size	_Z9vectorAddPKfS0_Pfi, .Lfunc_end0-_Z9vectorAddPKfS0_Pfi
                                        ; -- End function
	.set _Z9vectorAddPKfS0_Pfi.num_vgpr, 8
	.set _Z9vectorAddPKfS0_Pfi.num_agpr, 0
	.set _Z9vectorAddPKfS0_Pfi.numbered_sgpr, 8
	.set _Z9vectorAddPKfS0_Pfi.private_seg_size, 0
	.set _Z9vectorAddPKfS0_Pfi.uses_vcc, 1
	.set _Z9vectorAddPKfS0_Pfi.uses_flat_scratch, 0
	.set _Z9vectorAddPKfS0_Pfi.has_dyn_sized_stack, 0
	.set _Z9vectorAddPKfS0_Pfi.has_recursion, 0
	.set _Z9vectorAddPKfS0_Pfi.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 140
; TotalNumSgprs: 14
; NumVgprs: 8
; NumAgprs: 0
; TotalNumVgprs: 8
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 1
; VGPRBlocks: 0
; NumSGPRsForWavesPerEU: 14
; NumVGPRsForWavesPerEU: 8
; AccumOffset: 8
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 1
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.type	__hip_cuid_b6a544f012941045,@object ; @__hip_cuid_b6a544f012941045
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_b6a544f012941045
__hip_cuid_b6a544f012941045:
	.byte	0                               ; 0x0
	.size	__hip_cuid_b6a544f012941045, 1

	.ident	"AMD clang version 20.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.0.1 25314 f4087f6b428f0e6f575ebac8a8a724dab123d06e)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_b6a544f012941045
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .offset:         24
        .size:           4
        .value_kind:     by_value
      - .offset:         32
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         36
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         40
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         44
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         46
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         48
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         50
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         52
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         54
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         72
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         80
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         88
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         96
        .size:           2
        .value_kind:     hidden_grid_dims
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 288
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z9vectorAddPKfS0_Pfi
    .private_segment_fixed_size: 0
    .sgpr_count:     14
    .sgpr_spill_count: 0
    .symbol:         _Z9vectorAddPKfS0_Pfi.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     8
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata

# __CLANG_OFFLOAD_BUNDLE____END__ hip-amdgcn-amd-amdhsa--gfx950

# __CLANG_OFFLOAD_BUNDLE____START__ host-x86_64-unknown-linux-gnu-
	.file	"test_01_vector_add.hip"
	.text
	.globl	_Z24__device_stub__vectorAddPKfS0_Pfi # -- Begin function _Z24__device_stub__vectorAddPKfS0_Pfi
	.p2align	4
	.type	_Z24__device_stub__vectorAddPKfS0_Pfi,@function
_Z24__device_stub__vectorAddPKfS0_Pfi:  # @_Z24__device_stub__vectorAddPKfS0_Pfi
	.cfi_startproc
# %bb.0:
	subq	$120, %rsp
	.cfi_def_cfa_offset 128
	movq	%rdi, 72(%rsp)
	movq	%rsi, 64(%rsp)
	movq	%rdx, 56(%rsp)
	movl	%ecx, 4(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 80(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 88(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 104(%rsp)
	leaq	40(%rsp), %rdi
	leaq	24(%rsp), %rsi
	leaq	16(%rsp), %rdx
	leaq	8(%rsp), %rcx
	callq	__hipPopCallConfiguration
	movq	40(%rsp), %rsi
	movl	48(%rsp), %edx
	movq	24(%rsp), %rcx
	movl	32(%rsp), %r8d
	leaq	80(%rsp), %r9
	movl	$_Z9vectorAddPKfS0_Pfi, %edi
	pushq	8(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$136, %rsp
	.cfi_adjust_cfa_offset -136
	retq
.Lfunc_end0:
	.size	_Z24__device_stub__vectorAddPKfS0_Pfi, .Lfunc_end0-_Z24__device_stub__vectorAddPKfS0_Pfi
	.cfi_endproc
                                        # -- End function
	.section	.rodata.cst16,"aM",@progbits,16
	.p2align	4, 0x0                          # -- Begin function verify_vector_add
.LCPI1_0:
	.long	0x7fffffff                      # float NaN
	.long	0x7fffffff                      # float NaN
	.long	0x7fffffff                      # float NaN
	.long	0x7fffffff                      # float NaN
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3, 0x0
.LCPI1_1:
	.quad	0x3ee4f8b588e368f1              # double 1.0000000000000001E-5
	.text
	.globl	verify_vector_add
	.p2align	4
	.type	verify_vector_add,@function
verify_vector_add:                      # @verify_vector_add
	.cfi_startproc
# %bb.0:
	testl	%ecx, %ecx
	jle	.LBB1_1
# %bb.3:
	movl	%ecx, %ecx
	decq	%rcx
	xorl	%r8d, %r8d
	movaps	.LCPI1_0(%rip), %xmm0           # xmm0 = [NaN,NaN,NaN,NaN]
	movsd	.LCPI1_1(%rip), %xmm1           # xmm1 = [1.0000000000000001E-5,0.0E+0]
	.p2align	4
.LBB1_4:                                # =>This Inner Loop Header: Depth=1
	movss	(%rdi,%r8,4), %xmm2             # xmm2 = mem[0],zero,zero,zero
	addss	(%rsi,%r8,4), %xmm2
	movss	(%rdx,%r8,4), %xmm3             # xmm3 = mem[0],zero,zero,zero
	subss	%xmm2, %xmm3
	andps	%xmm0, %xmm3
	xorps	%xmm2, %xmm2
	cvtss2sd	%xmm3, %xmm2
	ucomisd	%xmm1, %xmm2
	setbe	%al
	ja	.LBB1_2
# %bb.5:                                #   in Loop: Header=BB1_4 Depth=1
	leaq	1(%r8), %r9
	cmpq	%r8, %rcx
	movq	%r9, %r8
	jne	.LBB1_4
.LBB1_2:
	retq
.LBB1_1:
	movb	$1, %al
	retq
.Lfunc_end1:
	.size	verify_vector_add, .Lfunc_end1-verify_vector_add
	.cfi_endproc
                                        # -- End function
	.section	.rodata.cst16,"aM",@progbits,16
	.p2align	4, 0x0                          # -- Begin function main
.LCPI2_0:
	.long	0                               # 0x0
	.long	1                               # 0x1
	.long	2                               # 0x2
	.long	3                               # 0x3
.LCPI2_1:
	.long	4                               # 0x4
	.long	4                               # 0x4
	.long	4                               # 0x4
	.long	4                               # 0x4
.LCPI2_2:
	.long	8                               # 0x8
	.long	8                               # 0x8
	.long	8                               # 0x8
	.long	8                               # 0x8
.LCPI2_3:
	.long	0x7fffffff                      # float NaN
	.long	0x7fffffff                      # float NaN
	.long	0x7fffffff                      # float NaN
	.long	0x7fffffff                      # float NaN
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3, 0x0
.LCPI2_4:
	.quad	0x3ee4f8b588e368f1              # double 1.0000000000000001E-5
	.text
	.globl	main
	.p2align	4
	.type	main,@function
main:                                   # @main
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%rbx
	.cfi_def_cfa_offset 40
	subq	$152, %rsp
	.cfi_def_cfa_offset 192
	.cfi_offset %rbx, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movl	$4096, %edi                     # imm = 0x1000
	callq	_Znam
	movq	%rax, %rbx
	movl	$4096, %edi                     # imm = 0x1000
	callq	_Znam
	movq	%rax, %r14
	movl	$4096, %edi                     # imm = 0x1000
	callq	_Znam
	movq	%rax, %r15
	movdqa	.LCPI2_0(%rip), %xmm0           # xmm0 = [0,1,2,3]
	movl	$4, %eax
	movdqa	.LCPI2_1(%rip), %xmm1           # xmm1 = [4,4,4,4]
	movdqa	.LCPI2_2(%rip), %xmm2           # xmm2 = [8,8,8,8]
	movdqa	%xmm0, %xmm3
	.p2align	4
.LBB2_1:                                # =>This Inner Loop Header: Depth=1
	movdqa	%xmm0, %xmm4
	paddd	%xmm1, %xmm4
	cvtdq2ps	%xmm0, %xmm5
	cvtdq2ps	%xmm4, %xmm4
	movups	%xmm5, -16(%rbx,%rax,4)
	movups	%xmm4, (%rbx,%rax,4)
	movdqa	%xmm3, %xmm4
	paddd	%xmm3, %xmm4
	cvtdq2ps	%xmm4, %xmm5
	paddd	%xmm2, %xmm4
	cvtdq2ps	%xmm4, %xmm4
	movups	%xmm5, -16(%r14,%rax,4)
	movups	%xmm4, (%r14,%rax,4)
	paddd	%xmm2, %xmm0
	paddd	%xmm2, %xmm3
	addq	$8, %rax
	cmpq	$1028, %rax                     # imm = 0x404
	jne	.LBB2_1
# %bb.2:
	leaq	24(%rsp), %rdi
	movl	$4096, %esi                     # imm = 0x1000
	callq	hipMalloc
	leaq	16(%rsp), %rdi
	movl	$4096, %esi                     # imm = 0x1000
	callq	hipMalloc
	leaq	8(%rsp), %rdi
	movl	$4096, %esi                     # imm = 0x1000
	callq	hipMalloc
	movq	24(%rsp), %rdi
	movl	$4096, %edx                     # imm = 0x1000
	movq	%rbx, %rsi
	movl	$1, %ecx
	callq	hipMemcpy
	movq	16(%rsp), %rdi
	movl	$4096, %edx                     # imm = 0x1000
	movq	%r14, %rsi
	movl	$1, %ecx
	callq	hipMemcpy
	movabsq	$4294967300, %rdi               # imm = 0x100000004
	leaq	252(%rdi), %rdx
	movl	$1, %esi
	movl	$1, %ecx
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration
	testl	%eax, %eax
	jne	.LBB2_4
# %bb.3:
	movq	24(%rsp), %rax
	movq	16(%rsp), %rcx
	movq	8(%rsp), %rdx
	movq	%rax, 104(%rsp)
	movq	%rcx, 96(%rsp)
	movq	%rdx, 88(%rsp)
	movl	$1024, 36(%rsp)                 # imm = 0x400
	leaq	104(%rsp), %rax
	movq	%rax, 112(%rsp)
	leaq	96(%rsp), %rax
	movq	%rax, 120(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	36(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	72(%rsp), %rdi
	leaq	56(%rsp), %rsi
	leaq	48(%rsp), %rdx
	leaq	40(%rsp), %rcx
	callq	__hipPopCallConfiguration
	movq	72(%rsp), %rsi
	movl	80(%rsp), %edx
	movq	56(%rsp), %rcx
	movl	64(%rsp), %r8d
	leaq	112(%rsp), %r9
	movl	$_Z9vectorAddPKfS0_Pfi, %edi
	pushq	40(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	56(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.LBB2_4:
	movq	8(%rsp), %rsi
	movl	$4096, %edx                     # imm = 0x1000
	movq	%r15, %rdi
	movl	$2, %ecx
	callq	hipMemcpy
	xorl	%eax, %eax
	movaps	.LCPI2_3(%rip), %xmm0           # xmm0 = [NaN,NaN,NaN,NaN]
	movsd	.LCPI2_4(%rip), %xmm1           # xmm1 = [1.0000000000000001E-5,0.0E+0]
	.p2align	4
.LBB2_5:                                # =>This Inner Loop Header: Depth=1
	movss	(%rbx,%rax,4), %xmm2            # xmm2 = mem[0],zero,zero,zero
	addss	(%r14,%rax,4), %xmm2
	movss	(%r15,%rax,4), %xmm3            # xmm3 = mem[0],zero,zero,zero
	subss	%xmm2, %xmm3
	andps	%xmm0, %xmm3
	xorps	%xmm2, %xmm2
	cvtss2sd	%xmm3, %xmm2
	ucomisd	%xmm1, %xmm2
	ja	.LBB2_7
# %bb.6:                                #   in Loop: Header=BB2_5 Depth=1
	leaq	1(%rax), %rcx
	cmpq	$1023, %rax                     # imm = 0x3FF
	movq	%rcx, %rax
	jne	.LBB2_5
.LBB2_7:
	xorl	%ebp, %ebp
	ucomisd	.LCPI2_4(%rip), %xmm2
	seta	%bpl
	movq	%rbx, %rdi
	callq	_ZdaPv
	movq	%r14, %rdi
	callq	_ZdaPv
	movq	%r15, %rdi
	callq	_ZdaPv
	movq	24(%rsp), %rdi
	callq	hipFree
	movq	16(%rsp), %rdi
	callq	hipFree
	movq	8(%rsp), %rdi
	callq	hipFree
	movl	%ebp, %eax
	addq	$152, %rsp
	.cfi_def_cfa_offset 40
	popq	%rbx
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end2:
	.size	main, .Lfunc_end2-main
	.cfi_endproc
                                        # -- End function
	.p2align	4                               # -- Begin function __hip_module_ctor
	.type	__hip_module_ctor,@function
__hip_module_ctor:                      # @__hip_module_ctor
	.cfi_startproc
# %bb.0:
	subq	$40, %rsp
	.cfi_def_cfa_offset 48
	movq	__hip_gpubin_handle_b6a544f012941045(%rip), %rdi
	testq	%rdi, %rdi
	jne	.LBB3_2
# %bb.1:
	movl	$__hip_fatbin_wrapper, %edi
	callq	__hipRegisterFatBinary
	movq	%rax, %rdi
	movq	%rax, __hip_gpubin_handle_b6a544f012941045(%rip)
.LBB3_2:
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movl	$_Z9vectorAddPKfS0_Pfi, %esi
	movl	$.L__unnamed_1, %edx
	movl	$.L__unnamed_1, %ecx
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction
	movl	$__hip_module_dtor, %edi
	addq	$40, %rsp
	.cfi_def_cfa_offset 8
	jmp	atexit                          # TAILCALL
.Lfunc_end3:
	.size	__hip_module_ctor, .Lfunc_end3-__hip_module_ctor
	.cfi_endproc
                                        # -- End function
	.p2align	4                               # -- Begin function __hip_module_dtor
	.type	__hip_module_dtor,@function
__hip_module_dtor:                      # @__hip_module_dtor
	.cfi_startproc
# %bb.0:
	movq	__hip_gpubin_handle_b6a544f012941045(%rip), %rdi
	testq	%rdi, %rdi
	je	.LBB4_2
# %bb.1:
	pushq	%rax
	.cfi_def_cfa_offset 16
	callq	__hipUnregisterFatBinary
	movq	$0, __hip_gpubin_handle_b6a544f012941045(%rip)
	addq	$8, %rsp
	.cfi_def_cfa_offset 8
.LBB4_2:
	retq
.Lfunc_end4:
	.size	__hip_module_dtor, .Lfunc_end4-__hip_module_dtor
	.cfi_endproc
                                        # -- End function
	.type	_Z9vectorAddPKfS0_Pfi,@object   # @_Z9vectorAddPKfS0_Pfi
	.section	.rodata,"a",@progbits
	.globl	_Z9vectorAddPKfS0_Pfi
	.p2align	3, 0x0
_Z9vectorAddPKfS0_Pfi:
	.quad	_Z24__device_stub__vectorAddPKfS0_Pfi
	.size	_Z9vectorAddPKfS0_Pfi, 8

	.type	.L__unnamed_1,@object           # @0
	.section	.rodata.str1.1,"aMS",@progbits,1
.L__unnamed_1:
	.asciz	"_Z9vectorAddPKfS0_Pfi"
	.size	.L__unnamed_1, 22

	.type	__hip_fatbin_wrapper,@object    # @__hip_fatbin_wrapper
	.section	.hipFatBinSegment,"a",@progbits
	.p2align	3, 0x0
__hip_fatbin_wrapper:
	.long	1212764230                      # 0x48495046
	.long	1                               # 0x1
	.quad	__hip_fatbin_b6a544f012941045
	.quad	0
	.size	__hip_fatbin_wrapper, 24

	.type	__hip_gpubin_handle_b6a544f012941045,@object # @__hip_gpubin_handle_b6a544f012941045
	.local	__hip_gpubin_handle_b6a544f012941045
	.comm	__hip_gpubin_handle_b6a544f012941045,8,8
	.section	.init_array,"aw",@init_array
	.p2align	3, 0x0
	.quad	__hip_module_ctor
	.type	__hip_cuid_b6a544f012941045,@object # @__hip_cuid_b6a544f012941045
	.bss
	.globl	__hip_cuid_b6a544f012941045
__hip_cuid_b6a544f012941045:
	.byte	0                               # 0x0
	.size	__hip_cuid_b6a544f012941045, 1

	.ident	"AMD clang version 20.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.0.1 25314 f4087f6b428f0e6f575ebac8a8a724dab123d06e)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym _Z24__device_stub__vectorAddPKfS0_Pfi
	.addrsig_sym __hip_module_ctor
	.addrsig_sym __hip_module_dtor
	.addrsig_sym _Z9vectorAddPKfS0_Pfi
	.addrsig_sym __hip_fatbin_b6a544f012941045
	.addrsig_sym __hip_fatbin_wrapper
	.addrsig_sym __hip_cuid_b6a544f012941045

# __CLANG_OFFLOAD_BUNDLE____END__ host-x86_64-unknown-linux-gnu-
