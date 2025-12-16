
# __CLANG_OFFLOAD_BUNDLE____START__ hip-amdgcn-amd-amdhsa--gfx950
	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.text
	.protected	_Z17conditionalKernelPKiPii ; -- Begin function _Z17conditionalKernelPKiPii
	.globl	_Z17conditionalKernelPKiPii
	.p2align	8
	.type	_Z17conditionalKernelPKiPii,@function
_Z17conditionalKernelPKiPii:            ; @_Z17conditionalKernelPKiPii
; %bb.0:
	s_load_dword s3, s[0:1], 0x24
	s_load_dword s4, s[0:1], 0x10
	s_waitcnt lgkmcnt(0)
	s_and_b32 s3, s3, 0xffff
	s_mul_i32 s2, s2, s3
	v_add_u32_e32 v0, s2, v0
	v_cmp_gt_i32_e32 vcc, s4, v0
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_6
; %bb.1:
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
                                        ; implicit-def: $vgpr2
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.2:
	v_mad_u64_u32 v[2:3], s[4:5], v3, 3, 1
                                        ; implicit-def: $vgpr3
; %bb.3:
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.4:
	v_lshlrev_b32_e32 v2, 1, v3
; %bb.5:
	s_or_b64 exec, exec, s[0:1]
	v_mov_b32_e32 v4, s2
	v_mov_b32_e32 v5, s3
	v_lshl_add_u64 v[0:1], v[0:1], 2, v[4:5]
	global_store_dword v[0:1], v2, off
.LBB0_6:
	s_endpgm
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
; Kernel info:
; codeLenInByte = 156
; TotalNumSgprs: 12
; NumVgprs: 6
; NumAgprs: 0
; TotalNumVgprs: 6
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 1
; VGPRBlocks: 0
; NumSGPRsForWavesPerEU: 12
; NumVGPRsForWavesPerEU: 6
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

# __CLANG_OFFLOAD_BUNDLE____END__ hip-amdgcn-amd-amdhsa--gfx950

# __CLANG_OFFLOAD_BUNDLE____START__ host-x86_64-unknown-linux-gnu-
	.file	"test_04_conditional.hip"
	.text
	.globl	_Z32__device_stub__conditionalKernelPKiPii # -- Begin function _Z32__device_stub__conditionalKernelPKiPii
	.p2align	4
	.type	_Z32__device_stub__conditionalKernelPKiPii,@function
_Z32__device_stub__conditionalKernelPKiPii: # @_Z32__device_stub__conditionalKernelPKiPii
	.cfi_startproc
# %bb.0:
	subq	$104, %rsp
	.cfi_def_cfa_offset 112
	movq	%rdi, 72(%rsp)
	movq	%rsi, 64(%rsp)
	movl	%edx, 12(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 80(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 88(%rsp)
	leaq	12(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	48(%rsp), %rdi
	leaq	32(%rsp), %rsi
	leaq	24(%rsp), %rdx
	leaq	16(%rsp), %rcx
	callq	__hipPopCallConfiguration
	movq	48(%rsp), %rsi
	movl	56(%rsp), %edx
	movq	32(%rsp), %rcx
	movl	40(%rsp), %r8d
	leaq	80(%rsp), %r9
	movl	$_Z17conditionalKernelPKiPii, %edi
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	32(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$120, %rsp
	.cfi_adjust_cfa_offset -120
	retq
.Lfunc_end0:
	.size	_Z32__device_stub__conditionalKernelPKiPii, .Lfunc_end0-_Z32__device_stub__conditionalKernelPKiPii
	.cfi_endproc
                                        # -- End function
	.section	.rodata.cst16,"aM",@progbits,16
	.p2align	4, 0x0                          # -- Begin function main
.LCPI1_0:
	.long	0                               # 0x0
	.long	1                               # 0x1
	.long	2                               # 0x2
	.long	3                               # 0x3
.LCPI1_1:
	.long	4                               # 0x4
	.long	4                               # 0x4
	.long	4                               # 0x4
	.long	4                               # 0x4
.LCPI1_2:
	.long	8                               # 0x8
	.long	8                               # 0x8
	.long	8                               # 0x8
	.long	8                               # 0x8
.LCPI1_3:
	.long	12                              # 0xc
	.long	12                              # 0xc
	.long	12                              # 0xc
	.long	12                              # 0xc
.LCPI1_4:
	.long	16                              # 0x10
	.long	16                              # 0x10
	.long	16                              # 0x10
	.long	16                              # 0x10
.LCPI1_5:
	.long	20                              # 0x14
	.long	20                              # 0x14
	.long	20                              # 0x14
	.long	20                              # 0x14
.LCPI1_6:
	.long	24                              # 0x18
	.long	24                              # 0x18
	.long	24                              # 0x18
	.long	24                              # 0x18
.LCPI1_7:
	.long	28                              # 0x1c
	.long	28                              # 0x1c
	.long	28                              # 0x1c
	.long	28                              # 0x1c
.LCPI1_8:
	.long	32                              # 0x20
	.long	32                              # 0x20
	.long	32                              # 0x20
	.long	32                              # 0x20
	.text
	.globl	main
	.p2align	4
	.type	main,@function
main:                                   # @main
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%rbx
	.cfi_def_cfa_offset 32
	subq	$128, %rsp
	.cfi_def_cfa_offset 160
	.cfi_offset %rbx, -32
	.cfi_offset %r14, -24
	.cfi_offset %rbp, -16
	movl	$2048, %edi                     # imm = 0x800
	callq	_Znam
	movq	%rax, %rbx
	movl	$2048, %edi                     # imm = 0x800
	callq	_Znam
	movq	%rax, %r14
	movdqa	.LCPI1_0(%rip), %xmm0           # xmm0 = [0,1,2,3]
	movl	$28, %eax
	movdqa	.LCPI1_1(%rip), %xmm1           # xmm1 = [4,4,4,4]
	movdqa	.LCPI1_2(%rip), %xmm2           # xmm2 = [8,8,8,8]
	movdqa	.LCPI1_3(%rip), %xmm3           # xmm3 = [12,12,12,12]
	movdqa	.LCPI1_4(%rip), %xmm4           # xmm4 = [16,16,16,16]
	movdqa	.LCPI1_5(%rip), %xmm5           # xmm5 = [20,20,20,20]
	movdqa	.LCPI1_6(%rip), %xmm6           # xmm6 = [24,24,24,24]
	movdqa	.LCPI1_7(%rip), %xmm7           # xmm7 = [28,28,28,28]
	movdqa	.LCPI1_8(%rip), %xmm8           # xmm8 = [32,32,32,32]
	.p2align	4
.LBB1_1:                                # =>This Inner Loop Header: Depth=1
	movdqa	%xmm0, %xmm9
	paddd	%xmm1, %xmm9
	movdqu	%xmm0, -112(%rbx,%rax,4)
	movdqu	%xmm9, -96(%rbx,%rax,4)
	movdqa	%xmm0, %xmm9
	paddd	%xmm2, %xmm9
	movdqa	%xmm0, %xmm10
	paddd	%xmm3, %xmm10
	movdqu	%xmm9, -80(%rbx,%rax,4)
	movdqu	%xmm10, -64(%rbx,%rax,4)
	movdqa	%xmm0, %xmm9
	paddd	%xmm4, %xmm9
	movdqa	%xmm0, %xmm10
	paddd	%xmm5, %xmm10
	movdqu	%xmm9, -48(%rbx,%rax,4)
	movdqu	%xmm10, -32(%rbx,%rax,4)
	movdqa	%xmm0, %xmm9
	paddd	%xmm6, %xmm9
	movdqa	%xmm0, %xmm10
	paddd	%xmm7, %xmm10
	movdqu	%xmm9, -16(%rbx,%rax,4)
	movdqu	%xmm10, (%rbx,%rax,4)
	paddd	%xmm8, %xmm0
	addq	$32, %rax
	cmpq	$540, %rax                      # imm = 0x21C
	jne	.LBB1_1
# %bb.2:
	leaq	16(%rsp), %rdi
	movl	$2048, %esi                     # imm = 0x800
	callq	hipMalloc
	leaq	8(%rsp), %rdi
	movl	$2048, %esi                     # imm = 0x800
	callq	hipMalloc
	movq	16(%rsp), %rdi
	movl	$2048, %edx                     # imm = 0x800
	movq	%rbx, %rsi
	movl	$1, %ecx
	callq	hipMemcpy
	movabsq	$4294967298, %rdi               # imm = 0x100000002
	leaq	254(%rdi), %rdx
	movl	$1, %esi
	movl	$1, %ecx
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration
	testl	%eax, %eax
	jne	.LBB1_4
# %bb.3:
	movq	16(%rsp), %rax
	movq	8(%rsp), %rcx
	movq	%rax, 88(%rsp)
	movq	%rcx, 80(%rsp)
	movl	$512, 28(%rsp)                  # imm = 0x200
	leaq	88(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 104(%rsp)
	leaq	28(%rsp), %rax
	movq	%rax, 112(%rsp)
	leaq	64(%rsp), %rdi
	leaq	48(%rsp), %rsi
	leaq	40(%rsp), %rdx
	leaq	32(%rsp), %rcx
	callq	__hipPopCallConfiguration
	movq	64(%rsp), %rsi
	movl	72(%rsp), %edx
	movq	48(%rsp), %rcx
	movl	56(%rsp), %r8d
	leaq	96(%rsp), %r9
	movl	$_Z17conditionalKernelPKiPii, %edi
	pushq	32(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	48(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.LBB1_4:
	movq	8(%rsp), %rsi
	movl	$2048, %edx                     # imm = 0x800
	movq	%r14, %rdi
	movl	$2, %ecx
	callq	hipMemcpy
	movl	$1, %eax
	movl	$1, %ebp
	.p2align	4
.LBB1_5:                                # =>This Inner Loop Header: Depth=1
	movl	-4(%rbx,%rax,4), %ecx
	leal	(%rcx,%rcx), %edx
	testb	$1, %cl
	leal	1(%rcx,%rcx,2), %ecx
	cmovel	%edx, %ecx
	cmpl	%ecx, -4(%r14,%rax,4)
	jne	.LBB1_9
# %bb.6:                                #   in Loop: Header=BB1_5 Depth=1
	movl	(%rbx,%rax,4), %ecx
	leal	(%rcx,%rcx), %edx
	testb	$1, %cl
	leal	1(%rcx,%rcx,2), %ecx
	cmovel	%edx, %ecx
	cmpl	%ecx, (%r14,%rax,4)
	jne	.LBB1_9
# %bb.7:                                #   in Loop: Header=BB1_5 Depth=1
	addq	$2, %rax
	cmpq	$513, %rax                      # imm = 0x201
	jne	.LBB1_5
# %bb.8:
	xorl	%ebp, %ebp
.LBB1_9:
	movq	%rbx, %rdi
	callq	_ZdaPv
	movq	%r14, %rdi
	callq	_ZdaPv
	movq	16(%rsp), %rdi
	callq	hipFree
	movq	8(%rsp), %rdi
	callq	hipFree
	movl	%ebp, %eax
	addq	$128, %rsp
	.cfi_def_cfa_offset 32
	popq	%rbx
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end1:
	.size	main, .Lfunc_end1-main
	.cfi_endproc
                                        # -- End function
	.p2align	4                               # -- Begin function __hip_module_ctor
	.type	__hip_module_ctor,@function
__hip_module_ctor:                      # @__hip_module_ctor
	.cfi_startproc
# %bb.0:
	subq	$40, %rsp
	.cfi_def_cfa_offset 48
	movq	__hip_gpubin_handle_df040c0751af06d2(%rip), %rdi
	testq	%rdi, %rdi
	jne	.LBB2_2
# %bb.1:
	movl	$__hip_fatbin_wrapper, %edi
	callq	__hipRegisterFatBinary
	movq	%rax, %rdi
	movq	%rax, __hip_gpubin_handle_df040c0751af06d2(%rip)
.LBB2_2:
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movl	$_Z17conditionalKernelPKiPii, %esi
	movl	$.L__unnamed_1, %edx
	movl	$.L__unnamed_1, %ecx
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction
	movl	$__hip_module_dtor, %edi
	addq	$40, %rsp
	.cfi_def_cfa_offset 8
	jmp	atexit                          # TAILCALL
.Lfunc_end2:
	.size	__hip_module_ctor, .Lfunc_end2-__hip_module_ctor
	.cfi_endproc
                                        # -- End function
	.p2align	4                               # -- Begin function __hip_module_dtor
	.type	__hip_module_dtor,@function
__hip_module_dtor:                      # @__hip_module_dtor
	.cfi_startproc
# %bb.0:
	movq	__hip_gpubin_handle_df040c0751af06d2(%rip), %rdi
	testq	%rdi, %rdi
	je	.LBB3_2
# %bb.1:
	pushq	%rax
	.cfi_def_cfa_offset 16
	callq	__hipUnregisterFatBinary
	movq	$0, __hip_gpubin_handle_df040c0751af06d2(%rip)
	addq	$8, %rsp
	.cfi_def_cfa_offset 8
.LBB3_2:
	retq
.Lfunc_end3:
	.size	__hip_module_dtor, .Lfunc_end3-__hip_module_dtor
	.cfi_endproc
                                        # -- End function
	.type	_Z17conditionalKernelPKiPii,@object # @_Z17conditionalKernelPKiPii
	.section	.rodata,"a",@progbits
	.globl	_Z17conditionalKernelPKiPii
	.p2align	3, 0x0
_Z17conditionalKernelPKiPii:
	.quad	_Z32__device_stub__conditionalKernelPKiPii
	.size	_Z17conditionalKernelPKiPii, 8

	.type	.L__unnamed_1,@object           # @0
	.section	.rodata.str1.1,"aMS",@progbits,1
.L__unnamed_1:
	.asciz	"_Z17conditionalKernelPKiPii"
	.size	.L__unnamed_1, 28

	.type	__hip_fatbin_wrapper,@object    # @__hip_fatbin_wrapper
	.section	.hipFatBinSegment,"a",@progbits
	.p2align	3, 0x0
__hip_fatbin_wrapper:
	.long	1212764230                      # 0x48495046
	.long	1                               # 0x1
	.quad	__hip_fatbin_df040c0751af06d2
	.quad	0
	.size	__hip_fatbin_wrapper, 24

	.type	__hip_gpubin_handle_df040c0751af06d2,@object # @__hip_gpubin_handle_df040c0751af06d2
	.local	__hip_gpubin_handle_df040c0751af06d2
	.comm	__hip_gpubin_handle_df040c0751af06d2,8,8
	.section	.init_array,"aw",@init_array
	.p2align	3, 0x0
	.quad	__hip_module_ctor
	.type	__hip_cuid_df040c0751af06d2,@object # @__hip_cuid_df040c0751af06d2
	.bss
	.globl	__hip_cuid_df040c0751af06d2
__hip_cuid_df040c0751af06d2:
	.byte	0                               # 0x0
	.size	__hip_cuid_df040c0751af06d2, 1

	.ident	"AMD clang version 20.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.0.1 25314 f4087f6b428f0e6f575ebac8a8a724dab123d06e)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym _Z32__device_stub__conditionalKernelPKiPii
	.addrsig_sym __hip_module_ctor
	.addrsig_sym __hip_module_dtor
	.addrsig_sym _Z17conditionalKernelPKiPii
	.addrsig_sym __hip_fatbin_df040c0751af06d2
	.addrsig_sym __hip_fatbin_wrapper
	.addrsig_sym __hip_cuid_df040c0751af06d2

# __CLANG_OFFLOAD_BUNDLE____END__ host-x86_64-unknown-linux-gnu-
