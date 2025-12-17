
# __CLANG_OFFLOAD_BUNDLE____START__ hip-amdgcn-amd-amdhsa--gfx950
	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.text
	.protected	_Z15sharedMemKernelPKiPii ; -- Begin function _Z15sharedMemKernelPKiPii
	.globl	_Z15sharedMemKernelPKiPii
	.p2align	8
	.type	_Z15sharedMemKernelPKiPii,@function
_Z15sharedMemKernelPKiPii:              ; @_Z15sharedMemKernelPKiPii
; %bb.0:
	s_load_dword s3, s[0:1], 0x24
	s_load_dword s8, s[0:1], 0x10
	s_load_dwordx4 s[4:7], s[0:1], 0x0
	v_mov_b32_e32 v3, 0
	s_waitcnt lgkmcnt(0)
	s_and_b32 s3, s3, 0xffff
	s_mul_i32 s0, s2, s3
	v_add_u32_e32 v2, s0, v0
	v_cmp_gt_i32_e32 vcc, s8, v2
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_2
; %bb.1:
	v_mov_b32_e32 v4, s4
	v_mov_b32_e32 v5, s5
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], v[2:3], 2, v[4:5]
	global_load_dword v3, v[2:3], off
.LBB0_2:
	s_or_b64 exec, exec, s[0:1]
	v_lshlrev_b32_e32 v1, 2, v0
	s_cmp_lt_u32 s3, 2
	s_waitcnt vmcnt(0)
	ds_write_b32 v1, v3
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cbranch_scc0 .LBB0_7
.LBB0_3:
	s_mov_b32 s3, 0
	v_cmp_eq_u32_e32 vcc, 0, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_5
; %bb.4:
	v_mov_b32_e32 v0, 0
	ds_read_b32 v1, v0
	s_lshl_b64 s[0:1], s[2:3], 2
	s_add_u32 s0, s6, s0
	s_addc_u32 s1, s7, s1
	s_waitcnt lgkmcnt(0)
	global_store_dword v0, v1, s[0:1]
.LBB0_5:
	s_endpgm
.LBB0_6:                                ;   in Loop: Header=BB0_7 Depth=1
	s_or_b64 exec, exec, s[0:1]
	s_cmp_lt_u32 s3, 4
	s_mov_b32 s3, s4
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cbranch_scc1 .LBB0_3
.LBB0_7:                                ; =>This Inner Loop Header: Depth=1
	s_lshr_b32 s4, s3, 1
	v_cmp_gt_u32_e32 vcc, s4, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_6
; %bb.8:                                ;   in Loop: Header=BB0_7 Depth=1
	v_lshl_add_u32 v2, s4, 2, v1
	ds_read_b32 v2, v2
	ds_read_b32 v3, v1
	s_waitcnt lgkmcnt(0)
	v_add_u32_e32 v2, v3, v2
	ds_write_b32 v1, v2
	s_branch .LBB0_6
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z15sharedMemKernelPKiPii
		.amdhsa_group_segment_fixed_size 1024
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
	.size	_Z15sharedMemKernelPKiPii, .Lfunc_end0-_Z15sharedMemKernelPKiPii
                                        ; -- End function
	.set _Z15sharedMemKernelPKiPii.num_vgpr, 6
	.set _Z15sharedMemKernelPKiPii.num_agpr, 0
	.set _Z15sharedMemKernelPKiPii.numbered_sgpr, 9
	.set _Z15sharedMemKernelPKiPii.private_seg_size, 0
	.set _Z15sharedMemKernelPKiPii.uses_vcc, 1
	.set _Z15sharedMemKernelPKiPii.uses_flat_scratch, 0
	.set _Z15sharedMemKernelPKiPii.has_dyn_sized_stack, 0
	.set _Z15sharedMemKernelPKiPii.has_recursion, 0
	.set _Z15sharedMemKernelPKiPii.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 264
; TotalNumSgprs: 15
; NumVgprs: 6
; NumAgprs: 0
; TotalNumVgprs: 6
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 1024 bytes/workgroup (compile time only)
; SGPRBlocks: 1
; VGPRBlocks: 0
; NumSGPRsForWavesPerEU: 15
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
	.type	__hip_cuid_17741c13aa0204ab,@object ; @__hip_cuid_17741c13aa0204ab
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_17741c13aa0204ab
__hip_cuid_17741c13aa0204ab:
	.byte	0                               ; 0x0
	.size	__hip_cuid_17741c13aa0204ab, 1

	.ident	"AMD clang version 20.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.0.1 25314 f4087f6b428f0e6f575ebac8a8a724dab123d06e)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_17741c13aa0204ab
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
    .group_segment_fixed_size: 1024
    .kernarg_segment_align: 8
    .kernarg_segment_size: 280
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z15sharedMemKernelPKiPii
    .private_segment_fixed_size: 0
    .sgpr_count:     15
    .sgpr_spill_count: 0
    .symbol:         _Z15sharedMemKernelPKiPii.kd
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
	.file	"test_06_shared_memory.hip"
	.text
	.globl	_Z30__device_stub__sharedMemKernelPKiPii # -- Begin function _Z30__device_stub__sharedMemKernelPKiPii
	.p2align	4
	.type	_Z30__device_stub__sharedMemKernelPKiPii,@function
_Z30__device_stub__sharedMemKernelPKiPii: # @_Z30__device_stub__sharedMemKernelPKiPii
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
	movl	$_Z15sharedMemKernelPKiPii, %edi
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	32(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$120, %rsp
	.cfi_adjust_cfa_offset -120
	retq
.Lfunc_end0:
	.size	_Z30__device_stub__sharedMemKernelPKiPii, .Lfunc_end0-_Z30__device_stub__sharedMemKernelPKiPii
	.cfi_endproc
                                        # -- End function
	.section	.rodata.cst16,"aM",@progbits,16
	.p2align	4, 0x0                          # -- Begin function main
.LCPI1_0:
	.long	1                               # 0x1
	.long	1                               # 0x1
	.long	1                               # 0x1
	.long	1                               # 0x1
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
	movl	$4096, %edi                     # imm = 0x1000
	callq	_Znam
	movq	%rax, %rbx
	movl	$16, %edi
	callq	_Znam
	movq	%rax, %r14
	movl	$28, %eax
	movaps	.LCPI1_0(%rip), %xmm0           # xmm0 = [1,1,1,1]
	.p2align	4
.LBB1_1:                                # =>This Inner Loop Header: Depth=1
	movups	%xmm0, -112(%rbx,%rax,4)
	movups	%xmm0, -96(%rbx,%rax,4)
	movups	%xmm0, -80(%rbx,%rax,4)
	movups	%xmm0, -64(%rbx,%rax,4)
	movups	%xmm0, -48(%rbx,%rax,4)
	movups	%xmm0, -32(%rbx,%rax,4)
	movups	%xmm0, -16(%rbx,%rax,4)
	movups	%xmm0, (%rbx,%rax,4)
	addq	$32, %rax
	cmpq	$1052, %rax                     # imm = 0x41C
	jne	.LBB1_1
# %bb.2:
	leaq	16(%rsp), %rdi
	movl	$4096, %esi                     # imm = 0x1000
	callq	hipMalloc
	leaq	8(%rsp), %rdi
	movl	$16, %esi
	callq	hipMalloc
	movq	16(%rsp), %rdi
	movl	$1, %ebp
	movl	$4096, %edx                     # imm = 0x1000
	movq	%rbx, %rsi
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
	jne	.LBB1_4
# %bb.3:
	movq	16(%rsp), %rax
	movq	8(%rsp), %rcx
	movq	%rax, 88(%rsp)
	movq	%rcx, 80(%rsp)
	movl	$1024, 28(%rsp)                 # imm = 0x400
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
	movl	$_Z15sharedMemKernelPKiPii, %edi
	pushq	32(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	48(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.LBB1_4:
	movq	8(%rsp), %rsi
	movl	$16, %edx
	movq	%r14, %rdi
	movl	$2, %ecx
	callq	hipMemcpy
	cmpl	$256, (%r14)                    # imm = 0x100
	jne	.LBB1_8
# %bb.5:
	cmpl	$256, 4(%r14)                   # imm = 0x100
	jne	.LBB1_8
# %bb.6:
	cmpl	$256, 8(%r14)                   # imm = 0x100
	jne	.LBB1_8
# %bb.7:
	xorl	%ebp, %ebp
	cmpl	$256, 12(%r14)                  # imm = 0x100
	setne	%bpl
.LBB1_8:
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
	movq	__hip_gpubin_handle_17741c13aa0204ab(%rip), %rdi
	testq	%rdi, %rdi
	jne	.LBB2_2
# %bb.1:
	movl	$__hip_fatbin_wrapper, %edi
	callq	__hipRegisterFatBinary
	movq	%rax, %rdi
	movq	%rax, __hip_gpubin_handle_17741c13aa0204ab(%rip)
.LBB2_2:
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movl	$_Z15sharedMemKernelPKiPii, %esi
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
	movq	__hip_gpubin_handle_17741c13aa0204ab(%rip), %rdi
	testq	%rdi, %rdi
	je	.LBB3_2
# %bb.1:
	pushq	%rax
	.cfi_def_cfa_offset 16
	callq	__hipUnregisterFatBinary
	movq	$0, __hip_gpubin_handle_17741c13aa0204ab(%rip)
	addq	$8, %rsp
	.cfi_def_cfa_offset 8
.LBB3_2:
	retq
.Lfunc_end3:
	.size	__hip_module_dtor, .Lfunc_end3-__hip_module_dtor
	.cfi_endproc
                                        # -- End function
	.type	_Z15sharedMemKernelPKiPii,@object # @_Z15sharedMemKernelPKiPii
	.section	.rodata,"a",@progbits
	.globl	_Z15sharedMemKernelPKiPii
	.p2align	3, 0x0
_Z15sharedMemKernelPKiPii:
	.quad	_Z30__device_stub__sharedMemKernelPKiPii
	.size	_Z15sharedMemKernelPKiPii, 8

	.type	.L__unnamed_1,@object           # @0
	.section	.rodata.str1.1,"aMS",@progbits,1
.L__unnamed_1:
	.asciz	"_Z15sharedMemKernelPKiPii"
	.size	.L__unnamed_1, 26

	.type	__hip_fatbin_wrapper,@object    # @__hip_fatbin_wrapper
	.section	.hipFatBinSegment,"a",@progbits
	.p2align	3, 0x0
__hip_fatbin_wrapper:
	.long	1212764230                      # 0x48495046
	.long	1                               # 0x1
	.quad	__hip_fatbin_17741c13aa0204ab
	.quad	0
	.size	__hip_fatbin_wrapper, 24

	.type	__hip_gpubin_handle_17741c13aa0204ab,@object # @__hip_gpubin_handle_17741c13aa0204ab
	.local	__hip_gpubin_handle_17741c13aa0204ab
	.comm	__hip_gpubin_handle_17741c13aa0204ab,8,8
	.section	.init_array,"aw",@init_array
	.p2align	3, 0x0
	.quad	__hip_module_ctor
	.type	__hip_cuid_17741c13aa0204ab,@object # @__hip_cuid_17741c13aa0204ab
	.bss
	.globl	__hip_cuid_17741c13aa0204ab
__hip_cuid_17741c13aa0204ab:
	.byte	0                               # 0x0
	.size	__hip_cuid_17741c13aa0204ab, 1

	.ident	"AMD clang version 20.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.0.1 25314 f4087f6b428f0e6f575ebac8a8a724dab123d06e)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym _Z30__device_stub__sharedMemKernelPKiPii
	.addrsig_sym __hip_module_ctor
	.addrsig_sym __hip_module_dtor
	.addrsig_sym _Z15sharedMemKernelPKiPii
	.addrsig_sym __hip_fatbin_17741c13aa0204ab
	.addrsig_sym __hip_fatbin_wrapper
	.addrsig_sym __hip_cuid_17741c13aa0204ab

# __CLANG_OFFLOAD_BUNDLE____END__ host-x86_64-unknown-linux-gnu-
