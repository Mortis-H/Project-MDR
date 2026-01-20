	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.text
	.globl	_Z9vectorAddPKfS0_Pfi
	.p2align	8
	.type	_Z9vectorAddPKfS0_Pfi,@function
_Z9vectorAddPKfS0_Pfi:
	;;#ASMSTART
	s_mov_b64 s[18:19], s[0:1]
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
		s_load_dword s3, s[0:1], 0x2c
	;;#ASMEND
	;;#ASMSTART
		s_load_dword s4, s[0:1], 0x18
	;;#ASMEND
	;;#ASMSTART
		s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
		s_and_b32 s3, s3, 0xffff
	;;#ASMEND
	;;#ASMSTART
		s_mul_i32 s2, s2, s3
	;;#ASMEND
	;;#ASMSTART
		v_add_u32_e32 v0, s2, v0
	;;#ASMEND
	;;#ASMSTART
		v_cmp_gt_i32_e32 vcc, s4, v0
	;;#ASMEND
	;;#ASMSTART
		s_and_saveexec_b64 s[2:3], vcc
	;;#ASMEND
	;;#ASMSTART
		s_cbranch_execz .LBB0_2
	;;#ASMEND
	;;#ASMSTART
		s_load_dwordx4 s[4:7], s[0:1], 0x0
	;;#ASMEND
	;;#ASMSTART
		s_load_dwordx2 s[2:3], s[0:1], 0x10
	;;#ASMEND
	;;#ASMSTART
		v_ashrrev_i32_e32 v1, 31, v0
	;;#ASMEND
	;;#ASMSTART
		v_lshlrev_b64 v[0:1], 2, v[0:1]
	;;#ASMEND
	;;#ASMSTART
		s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
		v_lshl_add_u64 v[4:5], s[4:5], 0, v[0:1]
	;;#ASMEND
	;;#ASMSTART
		v_lshl_add_u64 v[2:3], s[6:7], 0, v[0:1]
	;;#ASMEND
	;;#ASMSTART
		global_load_dword v6, v[4:5], off           ; Load A[tid]
	;;#ASMEND
	;;#ASMSTART
		global_load_dword v7, v[2:3], off           ; Load B[tid]
	;;#ASMEND
	;;#ASMSTART
		v_lshl_add_u64 v[0:1], s[2:3], 0, v[0:1]
	;;#ASMEND
	;;#ASMSTART
		s_waitcnt vmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v8, v6
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v9, v7
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v10, v2
	;;#ASMEND
	;;#ASMSTART
		v_add_f32_e32 v2, v6, v7                    ; C = A + B
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v11, v6
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v12, v7
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v13, v2
	;;#ASMEND
	;;#ASMSTART
		global_store_dword v[0:1], v2, off
	;;#ASMEND
	;;#ASMSTART
	s_mov_b64 s[0:1], s[18:19]
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v27, v8
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v30, v9
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v31, v10
	;;#ASMEND
	s_load_dwordx2 s[2:3], s[0:1], 0xa8
	v_mbcnt_lo_u32_b32 v16, -1, 0
	v_mbcnt_hi_u32_b32 v26, -1, v16
	v_mov_b64_e32 v[20:21], 0
	v_readfirstlane_b32 s0, v26
	s_nop 1
	v_cmp_eq_u32_e64 s[0:1], s0, v26
	s_and_saveexec_b64 s[20:21], s[0:1]
	s_cbranch_execz .LBB0_6
	v_mov_b32_e32 v22, 0
	s_waitcnt lgkmcnt(0)
	global_load_dwordx2 v[18:19], v22, s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[16:17], v22, s[2:3] offset:40
	global_load_dwordx2 v[20:21], v22, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v16, v16, v18
	v_and_b32_e32 v17, v17, v19
	v_mul_lo_u32 v17, v17, 24
	v_mul_hi_u32 v23, v16, 24
	v_add_u32_e32 v17, v23, v17
	v_mul_lo_u32 v16, v16, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[16:17], v[20:21], 0, v[16:17]
	global_load_dwordx2 v[16:17], v[16:17], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[20:21], v22, v[16:19], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[20:21], v[18:19]
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_5
	s_mov_b64 s[24:25], 0
.LBB0_3:
	s_sleep 1
	global_load_dwordx2 v[16:17], v22, s[2:3] offset:40
	global_load_dwordx2 v[18:19], v22, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v16, v16, v20
	v_and_b32_e32 v23, v17, v21
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[16:17], s[26:27], v16, 24, v[18:19]
	v_mov_b32_e32 v18, v17
	v_mad_u64_u32 v[18:19], s[26:27], v23, 24, v[18:19]
	v_mov_b32_e32 v17, v18
	global_load_dwordx2 v[18:19], v[16:17], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[16:17], v22, v[18:21], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[16:17], v[20:21]
	s_or_b64 s[24:25], vcc, s[24:25]
	v_mov_b64_e32 v[20:21], v[16:17]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBB0_3
	s_or_b64 exec, exec, s[24:25]
	v_mov_b64_e32 v[20:21], v[16:17]
.LBB0_5:
	s_or_b64 exec, exec, s[22:23]
.LBB0_6:
	s_or_b64 exec, exec, s[20:21]
	v_mov_b32_e32 v23, 0
	s_waitcnt lgkmcnt(0)
	global_load_dwordx2 v[24:25], v23, s[2:3] offset:40
	global_load_dwordx4 v[16:19], v23, s[2:3]
	v_readfirstlane_b32 s21, v21
	v_readfirstlane_b32 s20, v20
	s_mov_b64 s[22:23], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s24, v24
	v_readfirstlane_b32 s25, v25
	s_and_b64 s[24:25], s[24:25], s[20:21]
	s_mul_i32 s26, s25, 24
	s_mul_hi_u32 s27, s24, 24
	s_add_i32 s27, s27, s26
	s_mul_i32 s26, s24, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[20:21], v[16:17], 0, s[26:27]
	s_and_saveexec_b64 s[26:27], s[0:1]
	s_cbranch_execz .LBB0_8
	v_mov_b64_e32 v[32:33], s[22:23]
	v_mov_b32_e32 v34, 2
	v_mov_b32_e32 v35, 1
	global_store_dwordx4 v[20:21], v[32:35], off offset:8
.LBB0_8:
	s_or_b64 exec, exec, s[26:27]
	s_lshl_b64 s[22:23], s[24:25], 12
	v_lshl_add_u64 v[24:25], v[18:19], 0, s[22:23]
	s_mov_b32 s24, 0
	v_lshlrev_b32_e32 v22, 6, v26
	v_mov_b32_e32 v32, 33
	v_mov_b32_e32 v33, v23
	v_mov_b32_e32 v34, v23
	v_mov_b32_e32 v35, v23
	v_readfirstlane_b32 s22, v24
	v_readfirstlane_b32 s23, v25
	s_mov_b32 s26, s24
	s_mov_b32 s27, s24
	s_mov_b32 s25, s24
	s_nop 1
	global_store_dwordx4 v22, v[32:35], s[22:23]
	s_nop 1
	v_mov_b64_e32 v[34:35], s[26:27]
	v_mov_b64_e32 v[32:33], s[24:25]
	global_store_dwordx4 v22, v[32:35], s[22:23] offset:16
	global_store_dwordx4 v22, v[32:35], s[22:23] offset:32
	global_store_dwordx4 v22, v[32:35], s[22:23] offset:48
	s_and_saveexec_b64 s[22:23], s[0:1]
	s_cbranch_execz .LBB0_16
	v_mov_b32_e32 v32, 0
	global_load_dwordx2 v[36:37], v32, s[2:3] offset:32 sc0 sc1
	global_load_dwordx2 v[18:19], v32, s[2:3] offset:40
	v_mov_b32_e32 v34, s20
	v_mov_b32_e32 v35, s21
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v18, s20, v18
	v_and_b32_e32 v19, s21, v19
	v_mul_lo_u32 v19, v19, 24
	v_mul_hi_u32 v28, v18, 24
	v_mul_lo_u32 v18, v18, 24
	v_add_u32_e32 v19, v28, v19
	v_lshl_add_u64 v[28:29], v[16:17], 0, v[18:19]
	global_store_dwordx2 v[28:29], v[36:37], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[18:19], v32, v[34:37], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[18:19], v[36:37]
	s_and_saveexec_b64 s[24:25], vcc
	s_cbranch_execz .LBB0_12
	s_mov_b64 s[26:27], 0
.LBB0_11:
	s_sleep 1
	global_store_dwordx2 v[28:29], v[18:19], off
	v_mov_b32_e32 v16, s20
	v_mov_b32_e32 v17, s21
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[16:17], v32, v[16:19], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[16:17], v[18:19]
	s_or_b64 s[26:27], vcc, s[26:27]
	v_mov_b64_e32 v[18:19], v[16:17]
	s_andn2_b64 exec, exec, s[26:27]
	s_cbranch_execnz .LBB0_11
.LBB0_12:
	s_or_b64 exec, exec, s[24:25]
	v_mov_b32_e32 v19, 0
	global_load_dwordx2 v[16:17], v19, s[2:3] offset:16
	s_mov_b64 s[24:25], exec
	v_mbcnt_lo_u32_b32 v18, s24, 0
	v_mbcnt_hi_u32_b32 v18, s25, v18
	v_cmp_eq_u32_e32 vcc, 0, v18
	s_and_saveexec_b64 s[26:27], vcc
	s_cbranch_execz .LBB0_14
	s_bcnt1_i32_b64 s24, s[24:25]
	v_mov_b32_e32 v18, s24
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[16:17], v[18:19], off offset:8 sc1
.LBB0_14:
	s_or_b64 exec, exec, s[26:27]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[18:19], v[16:17], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[18:19]
	s_cbranch_vccnz .LBB0_16
	global_load_dword v16, v[16:17], off offset:24
	v_mov_b32_e32 v17, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s24, v16
	s_and_b32 m0, s24, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[18:19], v[16:17], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_16:
	s_or_b64 exec, exec, s[22:23]
	v_lshl_add_u64 v[16:17], v[24:25], 0, v[22:23]
	s_branch .LBB0_20
.LBB0_17:
	s_or_b64 exec, exec, s[22:23]
	v_readfirstlane_b32 s22, v18
	s_cmp_eq_u32 s22, 0
	s_cbranch_scc1 .LBB0_19
	s_sleep 1
	s_cbranch_execnz .LBB0_20
	s_branch .LBB0_22
.LBB0_19:
	s_branch .LBB0_22
.LBB0_20:
	v_mov_b32_e32 v18, 1
	s_and_saveexec_b64 s[22:23], s[0:1]
	s_cbranch_execz .LBB0_17
	global_load_dword v18, v[20:21], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v18, 1, v18
	s_branch .LBB0_17
.LBB0_22:
	global_load_dwordx2 v[16:17], v[16:17], off
	s_and_saveexec_b64 s[22:23], s[0:1]
	s_cbranch_execz .LBB0_25
	v_mov_b32_e32 v24, 0
	global_load_dwordx2 v[18:19], v24, s[2:3] offset:40
	global_load_dwordx2 v[34:35], v24, s[2:3] offset:24 sc0 sc1
	global_load_dwordx2 v[20:21], v24, s[2:3]
	s_waitcnt vmcnt(2)
	v_readfirstlane_b32 s24, v18
	v_readfirstlane_b32 s25, v19
	s_add_u32 s26, s24, 1
	s_addc_u32 s27, s25, 0
	s_add_u32 s0, s26, s20
	s_addc_u32 s1, s27, s21
	s_cmp_eq_u64 s[0:1], 0
	s_cselect_b32 s1, s27, s1
	s_cselect_b32 s0, s26, s0
	s_and_b64 s[20:21], s[0:1], s[24:25]
	s_mul_i32 s21, s21, 24
	s_mul_hi_u32 s24, s20, 24
	s_mul_i32 s20, s20, 24
	s_add_i32 s21, s24, s21
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[22:23], v[20:21], 0, s[20:21]
	v_mov_b32_e32 v32, s0
	global_store_dwordx2 v[22:23], v[34:35], off
	v_mov_b32_e32 v33, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[20:21], v24, v[32:35], s[2:3] offset:24 sc0 sc1
	s_mov_b64 s[20:21], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[20:21], v[34:35]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_25
.LBB0_24:
	s_sleep 1
	global_store_dwordx2 v[22:23], v[20:21], off
	v_mov_b32_e32 v18, s0
	v_mov_b32_e32 v19, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[18:19], v24, v[18:21], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[18:19], v[20:21]
	s_or_b64 s[20:21], vcc, s[20:21]
	v_mov_b64_e32 v[20:21], v[18:19]
	s_andn2_b64 exec, exec, s[20:21]
	s_cbranch_execnz .LBB0_24
.LBB0_25:
	s_or_b64 exec, exec, s[22:23]
	v_readfirstlane_b32 s0, v26
	v_mov_b64_e32 v[24:25], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v26
	s_and_saveexec_b64 s[20:21], s[0:1]
	s_cbranch_execz .LBB0_31
	v_mov_b32_e32 v20, 0
	global_load_dwordx2 v[34:35], v20, s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[18:19], v20, s[2:3] offset:40
	global_load_dwordx2 v[22:23], v20, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v18, v18, v34
	v_and_b32_e32 v19, v19, v35
	v_mul_lo_u32 v19, v19, 24
	v_mul_hi_u32 v21, v18, 24
	v_add_u32_e32 v19, v21, v19
	v_mul_lo_u32 v18, v18, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[18:19], v[22:23], 0, v[18:19]
	global_load_dwordx2 v[32:33], v[18:19], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[24:25], v20, v[32:35], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[24:25], v[34:35]
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_30
	s_mov_b64 s[24:25], 0
.LBB0_28:
	s_sleep 1
	global_load_dwordx2 v[18:19], v20, s[2:3] offset:40
	global_load_dwordx2 v[22:23], v20, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v18, v18, v24
	v_and_b32_e32 v21, v19, v25
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[18:19], s[26:27], v18, 24, v[22:23]
	v_mov_b32_e32 v22, v19
	v_mad_u64_u32 v[22:23], s[26:27], v21, 24, v[22:23]
	v_mov_b32_e32 v19, v22
	global_load_dwordx2 v[22:23], v[18:19], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[18:19], v20, v[22:25], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[18:19], v[24:25]
	s_or_b64 s[24:25], vcc, s[24:25]
	v_mov_b64_e32 v[24:25], v[18:19]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBB0_28
	s_or_b64 exec, exec, s[24:25]
	v_mov_b64_e32 v[24:25], v[18:19]
.LBB0_30:
	s_or_b64 exec, exec, s[22:23]
.LBB0_31:
	s_or_b64 exec, exec, s[20:21]
	v_mov_b32_e32 v29, 0
	global_load_dwordx2 v[18:19], v29, s[2:3] offset:40
	global_load_dwordx4 v[20:23], v29, s[2:3]
	v_readfirstlane_b32 s21, v25
	v_readfirstlane_b32 s20, v24
	s_mov_b64 s[22:23], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s24, v18
	v_readfirstlane_b32 s25, v19
	s_and_b64 s[24:25], s[24:25], s[20:21]
	s_mul_i32 s26, s25, 24
	s_mul_hi_u32 s27, s24, 24
	s_add_i32 s27, s27, s26
	s_mul_i32 s26, s24, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[24:25], v[20:21], 0, s[26:27]
	s_and_saveexec_b64 s[26:27], s[0:1]
	s_cbranch_execz .LBB0_33
	v_mov_b64_e32 v[32:33], s[22:23]
	v_mov_b32_e32 v34, 2
	v_mov_b32_e32 v35, 1
	global_store_dwordx4 v[24:25], v[32:35], off offset:8
.LBB0_33:
	s_or_b64 exec, exec, s[26:27]
	s_lshl_b64 s[22:23], s[24:25], 12
	v_lshl_add_u64 v[22:23], v[22:23], 0, s[22:23]
	v_and_b32_e32 v16, 0xffffff1f, v16
	v_or_b32_e32 v16, 0xc0, v16
	v_lshlrev_b32_e32 v28, 6, v26
	v_mov_b32_e32 v19, 0x41206572
	v_mov_b32_e32 v18, 0x6f666542
	v_readfirstlane_b32 s22, v22
	v_readfirstlane_b32 s23, v23
	v_mov_b32_e32 v32, 0x3d293776
	v_mov_b32_e32 v33, 0x202c6625
	v_mov_b32_e32 v34, 0x32762843
	s_nop 1
	global_store_dwordx4 v28, v[16:19], s[22:23]
	s_nop 1
	v_mov_b32_e32 v16, 0x203a4444
	v_mov_b32_e32 v17, 0x36762841
	v_mov_b32_e32 v18, 0x66253d29
	v_mov_b32_e32 v19, 0x2842202c
	global_store_dwordx4 v28, v[16:19], s[22:23] offset:16
	v_mov_b32_e32 v35, v18
	global_store_dwordx4 v28, v[32:35], s[22:23] offset:32
	v_mov_b32_e32 v16, 10
	v_mov_b32_e32 v17, v29
	v_mov_b32_e32 v18, v29
	v_mov_b32_e32 v19, v29
	global_store_dwordx4 v28, v[16:19], s[22:23] offset:48
	s_and_saveexec_b64 s[22:23], s[0:1]
	s_cbranch_execz .LBB0_41
	v_mov_b32_e32 v32, 0
	global_load_dwordx2 v[36:37], v32, s[2:3] offset:32 sc0 sc1
	global_load_dwordx2 v[16:17], v32, s[2:3] offset:40
	v_mov_b32_e32 v34, s20
	v_mov_b32_e32 v35, s21
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s24, v16
	v_readfirstlane_b32 s25, v17
	s_and_b64 s[24:25], s[24:25], s[20:21]
	s_mul_i32 s25, s25, 24
	s_mul_hi_u32 s26, s24, 24
	s_mul_i32 s24, s24, 24
	s_add_i32 s25, s26, s25
	v_lshl_add_u64 v[20:21], v[20:21], 0, s[24:25]
	global_store_dwordx2 v[20:21], v[36:37], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[18:19], v32, v[34:37], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[18:19], v[36:37]
	s_and_saveexec_b64 s[24:25], vcc
	s_cbranch_execz .LBB0_37
	s_mov_b64 s[26:27], 0
.LBB0_36:
	s_sleep 1
	global_store_dwordx2 v[20:21], v[18:19], off
	v_mov_b32_e32 v16, s20
	v_mov_b32_e32 v17, s21
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[16:17], v32, v[16:19], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[16:17], v[18:19]
	s_or_b64 s[26:27], vcc, s[26:27]
	v_mov_b64_e32 v[18:19], v[16:17]
	s_andn2_b64 exec, exec, s[26:27]
	s_cbranch_execnz .LBB0_36
.LBB0_37:
	s_or_b64 exec, exec, s[24:25]
	v_mov_b32_e32 v19, 0
	global_load_dwordx2 v[16:17], v19, s[2:3] offset:16
	s_mov_b64 s[24:25], exec
	v_mbcnt_lo_u32_b32 v18, s24, 0
	v_mbcnt_hi_u32_b32 v18, s25, v18
	v_cmp_eq_u32_e32 vcc, 0, v18
	s_and_saveexec_b64 s[26:27], vcc
	s_cbranch_execz .LBB0_39
	s_bcnt1_i32_b64 s24, s[24:25]
	v_mov_b32_e32 v18, s24
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[16:17], v[18:19], off offset:8 sc1
.LBB0_39:
	s_or_b64 exec, exec, s[26:27]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[18:19], v[16:17], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[18:19]
	s_cbranch_vccnz .LBB0_41
	global_load_dword v16, v[16:17], off offset:24
	v_mov_b32_e32 v17, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s24, v16
	s_and_b32 m0, s24, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[18:19], v[16:17], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_41:
	s_or_b64 exec, exec, s[22:23]
	v_lshl_add_u64 v[16:17], v[22:23], 0, v[28:29]
	s_branch .LBB0_45
.LBB0_42:
	s_or_b64 exec, exec, s[22:23]
	v_readfirstlane_b32 s22, v18
	s_cmp_eq_u32 s22, 0
	s_cbranch_scc1 .LBB0_44
	s_sleep 1
	s_cbranch_execnz .LBB0_45
	s_branch .LBB0_47
.LBB0_44:
	s_branch .LBB0_47
.LBB0_45:
	v_mov_b32_e32 v18, 1
	s_and_saveexec_b64 s[22:23], s[0:1]
	s_cbranch_execz .LBB0_42
	global_load_dword v18, v[24:25], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v18, 1, v18
	s_branch .LBB0_42
.LBB0_47:
	global_load_dwordx2 v[16:17], v[16:17], off
	s_and_saveexec_b64 s[22:23], s[0:1]
	s_cbranch_execz .LBB0_50
	v_mov_b32_e32 v24, 0
	global_load_dwordx2 v[18:19], v24, s[2:3] offset:40
	global_load_dwordx2 v[34:35], v24, s[2:3] offset:24 sc0 sc1
	global_load_dwordx2 v[20:21], v24, s[2:3]
	s_waitcnt vmcnt(2)
	v_readfirstlane_b32 s24, v18
	v_readfirstlane_b32 s25, v19
	s_add_u32 s26, s24, 1
	s_addc_u32 s27, s25, 0
	s_add_u32 s0, s26, s20
	s_addc_u32 s1, s27, s21
	s_cmp_eq_u64 s[0:1], 0
	s_cselect_b32 s1, s27, s1
	s_cselect_b32 s0, s26, s0
	s_and_b64 s[20:21], s[0:1], s[24:25]
	s_mul_i32 s21, s21, 24
	s_mul_hi_u32 s24, s20, 24
	s_mul_i32 s20, s20, 24
	s_add_i32 s21, s24, s21
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[22:23], v[20:21], 0, s[20:21]
	v_mov_b32_e32 v32, s0
	global_store_dwordx2 v[22:23], v[34:35], off
	v_mov_b32_e32 v33, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[20:21], v24, v[32:35], s[2:3] offset:24 sc0 sc1
	s_mov_b64 s[20:21], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[20:21], v[34:35]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_50
.LBB0_49:
	s_sleep 1
	global_store_dwordx2 v[22:23], v[20:21], off
	v_mov_b32_e32 v18, s0
	v_mov_b32_e32 v19, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[18:19], v24, v[18:21], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[18:19], v[20:21]
	s_or_b64 s[20:21], vcc, s[20:21]
	v_mov_b64_e32 v[20:21], v[18:19]
	s_andn2_b64 exec, exec, s[20:21]
	s_cbranch_execnz .LBB0_49
.LBB0_50:
	s_or_b64 exec, exec, s[22:23]
	v_readfirstlane_b32 s0, v26
	v_mov_b64_e32 v[24:25], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v26
	s_and_saveexec_b64 s[20:21], s[0:1]
	s_cbranch_execz .LBB0_56
	v_mov_b32_e32 v20, 0
	global_load_dwordx2 v[34:35], v20, s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[18:19], v20, s[2:3] offset:40
	global_load_dwordx2 v[22:23], v20, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v18, v18, v34
	v_and_b32_e32 v19, v19, v35
	v_mul_lo_u32 v19, v19, 24
	v_mul_hi_u32 v21, v18, 24
	v_add_u32_e32 v19, v21, v19
	v_mul_lo_u32 v18, v18, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[18:19], v[22:23], 0, v[18:19]
	global_load_dwordx2 v[32:33], v[18:19], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[24:25], v20, v[32:35], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[24:25], v[34:35]
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_55
	s_mov_b64 s[24:25], 0
.LBB0_53:
	s_sleep 1
	global_load_dwordx2 v[18:19], v20, s[2:3] offset:40
	global_load_dwordx2 v[22:23], v20, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v18, v18, v24
	v_and_b32_e32 v21, v19, v25
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[18:19], s[26:27], v18, 24, v[22:23]
	v_mov_b32_e32 v22, v19
	v_mad_u64_u32 v[22:23], s[26:27], v21, 24, v[22:23]
	v_mov_b32_e32 v19, v22
	global_load_dwordx2 v[22:23], v[18:19], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[18:19], v20, v[22:25], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[18:19], v[24:25]
	s_or_b64 s[24:25], vcc, s[24:25]
	v_mov_b64_e32 v[24:25], v[18:19]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBB0_53
	s_or_b64 exec, exec, s[24:25]
	v_mov_b64_e32 v[24:25], v[18:19]
.LBB0_55:
	s_or_b64 exec, exec, s[22:23]
.LBB0_56:
	s_or_b64 exec, exec, s[20:21]
	v_mov_b32_e32 v29, 0
	global_load_dwordx2 v[18:19], v29, s[2:3] offset:40
	global_load_dwordx4 v[20:23], v29, s[2:3]
	v_readfirstlane_b32 s21, v25
	v_readfirstlane_b32 s20, v24
	s_mov_b64 s[22:23], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s24, v18
	v_readfirstlane_b32 s25, v19
	s_and_b64 s[24:25], s[24:25], s[20:21]
	s_mul_i32 s26, s25, 24
	s_mul_hi_u32 s27, s24, 24
	s_add_i32 s27, s27, s26
	s_mul_i32 s26, s24, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[24:25], v[20:21], 0, s[26:27]
	s_and_saveexec_b64 s[26:27], s[0:1]
	s_cbranch_execz .LBB0_58
	v_mov_b64_e32 v[32:33], s[22:23]
	v_mov_b32_e32 v34, 2
	v_mov_b32_e32 v35, 1
	global_store_dwordx4 v[24:25], v[32:35], off offset:8
.LBB0_58:
	s_or_b64 exec, exec, s[26:27]
	s_lshl_b64 s[22:23], s[24:25], 12
	v_lshl_add_u64 v[22:23], v[22:23], 0, s[22:23]
	v_and_b32_e32 v16, 0xffffff1d, v16
	s_mov_b32 s24, 0
	v_cvt_f64_f32_e32 v[18:19], v27
	v_or_b32_e32 v16, 0x62, v16
	v_readfirstlane_b32 s22, v22
	v_readfirstlane_b32 s23, v23
	s_mov_b32 s25, s24
	v_cvt_f64_f32_e32 v[32:33], v30
	v_cvt_f64_f32_e32 v[34:35], v31
	s_nop 1
	global_store_dwordx4 v28, v[16:19], s[22:23]
	global_store_dwordx4 v28, v[32:35], s[22:23] offset:16
	s_mov_b32 s26, s24
	s_mov_b32 s27, s24
	v_mov_b64_e32 v[16:17], s[24:25]
	v_mov_b64_e32 v[18:19], s[26:27]
	global_store_dwordx4 v28, v[16:19], s[22:23] offset:32
	global_store_dwordx4 v28, v[16:19], s[22:23] offset:48
	s_and_saveexec_b64 s[22:23], s[0:1]
	s_cbranch_execz .LBB0_66
	v_mov_b32_e32 v22, 0
	global_load_dwordx2 v[32:33], v22, s[2:3] offset:32 sc0 sc1
	global_load_dwordx2 v[16:17], v22, s[2:3] offset:40
	v_mov_b32_e32 v30, s20
	v_mov_b32_e32 v31, s21
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s24, v16
	v_readfirstlane_b32 s25, v17
	s_and_b64 s[24:25], s[24:25], s[20:21]
	s_mul_i32 s25, s25, 24
	s_mul_hi_u32 s26, s24, 24
	s_mul_i32 s24, s24, 24
	s_add_i32 s25, s26, s25
	v_lshl_add_u64 v[20:21], v[20:21], 0, s[24:25]
	global_store_dwordx2 v[20:21], v[32:33], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[18:19], v22, v[30:33], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[18:19], v[32:33]
	s_and_saveexec_b64 s[24:25], vcc
	s_cbranch_execz .LBB0_62
	s_mov_b64 s[26:27], 0
.LBB0_61:
	s_sleep 1
	global_store_dwordx2 v[20:21], v[18:19], off
	v_mov_b32_e32 v16, s20
	v_mov_b32_e32 v17, s21
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[16:17], v22, v[16:19], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[16:17], v[18:19]
	s_or_b64 s[26:27], vcc, s[26:27]
	v_mov_b64_e32 v[18:19], v[16:17]
	s_andn2_b64 exec, exec, s[26:27]
	s_cbranch_execnz .LBB0_61
.LBB0_62:
	s_or_b64 exec, exec, s[24:25]
	v_mov_b32_e32 v19, 0
	global_load_dwordx2 v[16:17], v19, s[2:3] offset:16
	s_mov_b64 s[24:25], exec
	v_mbcnt_lo_u32_b32 v18, s24, 0
	v_mbcnt_hi_u32_b32 v18, s25, v18
	v_cmp_eq_u32_e32 vcc, 0, v18
	s_and_saveexec_b64 s[26:27], vcc
	s_cbranch_execz .LBB0_64
	s_bcnt1_i32_b64 s24, s[24:25]
	v_mov_b32_e32 v18, s24
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[16:17], v[18:19], off offset:8 sc1
.LBB0_64:
	s_or_b64 exec, exec, s[26:27]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[18:19], v[16:17], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[18:19]
	s_cbranch_vccnz .LBB0_66
	global_load_dword v16, v[16:17], off offset:24
	v_mov_b32_e32 v17, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s24, v16
	s_and_b32 m0, s24, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[18:19], v[16:17], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_66:
	s_or_b64 exec, exec, s[22:23]
	s_branch .LBB0_70
.LBB0_67:
	s_or_b64 exec, exec, s[22:23]
	v_readfirstlane_b32 s22, v16
	s_cmp_eq_u32 s22, 0
	s_cbranch_scc1 .LBB0_69
	s_sleep 1
	s_cbranch_execnz .LBB0_70
	s_branch .LBB0_72
.LBB0_69:
	s_branch .LBB0_72
.LBB0_70:
	v_mov_b32_e32 v16, 1
	s_and_saveexec_b64 s[22:23], s[0:1]
	s_cbranch_execz .LBB0_67
	global_load_dword v16, v[24:25], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v16, 1, v16
	s_branch .LBB0_67
.LBB0_72:
	s_and_saveexec_b64 s[22:23], s[0:1]
	s_cbranch_execz .LBB0_75
	v_mov_b32_e32 v22, 0
	global_load_dwordx2 v[16:17], v22, s[2:3] offset:40
	global_load_dwordx2 v[32:33], v22, s[2:3] offset:24 sc0 sc1
	global_load_dwordx2 v[18:19], v22, s[2:3]
	s_waitcnt vmcnt(2)
	v_readfirstlane_b32 s24, v16
	v_readfirstlane_b32 s25, v17
	s_add_u32 s26, s24, 1
	s_addc_u32 s27, s25, 0
	s_add_u32 s0, s26, s20
	s_addc_u32 s1, s27, s21
	s_cmp_eq_u64 s[0:1], 0
	s_cselect_b32 s1, s27, s1
	s_cselect_b32 s0, s26, s0
	s_and_b64 s[20:21], s[0:1], s[24:25]
	s_mul_i32 s21, s21, 24
	s_mul_hi_u32 s24, s20, 24
	s_mul_i32 s20, s20, 24
	s_add_i32 s21, s24, s21
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[20:21], v[18:19], 0, s[20:21]
	v_mov_b32_e32 v30, s0
	global_store_dwordx2 v[20:21], v[32:33], off
	v_mov_b32_e32 v31, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[18:19], v22, v[30:33], s[2:3] offset:24 sc0 sc1
	s_mov_b64 s[20:21], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[18:19], v[32:33]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_75
.LBB0_74:
	s_sleep 1
	global_store_dwordx2 v[20:21], v[18:19], off
	v_mov_b32_e32 v16, s0
	v_mov_b32_e32 v17, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[16:17], v22, v[16:19], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[16:17], v[18:19]
	s_or_b64 s[20:21], vcc, s[20:21]
	v_mov_b64_e32 v[18:19], v[16:17]
	s_andn2_b64 exec, exec, s[20:21]
	s_cbranch_execnz .LBB0_74
.LBB0_75:
	s_or_b64 exec, exec, s[22:23]
	v_readfirstlane_b32 s0, v26
	v_mov_b64_e32 v[20:21], 0
	;;#ASMSTART
	v_mov_b32 v27, v11
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v30, v12
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v31, v13
	;;#ASMEND
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v26
	s_and_saveexec_b64 s[20:21], s[0:1]
	s_cbranch_execz .LBB0_81
	v_mov_b32_e32 v22, 0
	global_load_dwordx2 v[18:19], v22, s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[16:17], v22, s[2:3] offset:40
	global_load_dwordx2 v[20:21], v22, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v16, v16, v18
	v_and_b32_e32 v17, v17, v19
	v_mul_lo_u32 v17, v17, 24
	v_mul_hi_u32 v23, v16, 24
	v_add_u32_e32 v17, v23, v17
	v_mul_lo_u32 v16, v16, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[16:17], v[20:21], 0, v[16:17]
	global_load_dwordx2 v[16:17], v[16:17], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[20:21], v22, v[16:19], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[20:21], v[18:19]
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_80
	s_mov_b64 s[24:25], 0
.LBB0_78:
	s_sleep 1
	global_load_dwordx2 v[16:17], v22, s[2:3] offset:40
	global_load_dwordx2 v[18:19], v22, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v16, v16, v20
	v_and_b32_e32 v23, v17, v21
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[16:17], s[26:27], v16, 24, v[18:19]
	v_mov_b32_e32 v18, v17
	v_mad_u64_u32 v[18:19], s[26:27], v23, 24, v[18:19]
	v_mov_b32_e32 v17, v18
	global_load_dwordx2 v[18:19], v[16:17], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[16:17], v22, v[18:21], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[16:17], v[20:21]
	s_or_b64 s[24:25], vcc, s[24:25]
	v_mov_b64_e32 v[20:21], v[16:17]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBB0_78
	s_or_b64 exec, exec, s[24:25]
	v_mov_b64_e32 v[20:21], v[16:17]
.LBB0_80:
	s_or_b64 exec, exec, s[22:23]
.LBB0_81:
	s_or_b64 exec, exec, s[20:21]
	v_mov_b32_e32 v29, 0
	global_load_dwordx2 v[22:23], v29, s[2:3] offset:40
	global_load_dwordx4 v[16:19], v29, s[2:3]
	v_readfirstlane_b32 s21, v21
	v_readfirstlane_b32 s20, v20
	s_mov_b64 s[22:23], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s24, v22
	v_readfirstlane_b32 s25, v23
	s_and_b64 s[24:25], s[24:25], s[20:21]
	s_mul_i32 s26, s25, 24
	s_mul_hi_u32 s27, s24, 24
	s_add_i32 s27, s27, s26
	s_mul_i32 s26, s24, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[20:21], v[16:17], 0, s[26:27]
	s_and_saveexec_b64 s[26:27], s[0:1]
	s_cbranch_execz .LBB0_83
	v_mov_b64_e32 v[22:23], s[22:23]
	v_mov_b32_e32 v24, 2
	v_mov_b32_e32 v25, 1
	global_store_dwordx4 v[20:21], v[22:25], off offset:8
.LBB0_83:
	s_or_b64 exec, exec, s[26:27]
	s_lshl_b64 s[22:23], s[24:25], 12
	v_lshl_add_u64 v[22:23], v[18:19], 0, s[22:23]
	s_mov_b32 s24, 0
	v_mov_b32_e32 v32, 33
	v_mov_b32_e32 v33, v29
	v_mov_b32_e32 v34, v29
	v_mov_b32_e32 v35, v29
	v_readfirstlane_b32 s22, v22
	v_readfirstlane_b32 s23, v23
	s_mov_b32 s26, s24
	s_mov_b32 s27, s24
	s_mov_b32 s25, s24
	s_nop 1
	global_store_dwordx4 v28, v[32:35], s[22:23]
	s_nop 1
	v_mov_b64_e32 v[34:35], s[26:27]
	v_mov_b64_e32 v[32:33], s[24:25]
	global_store_dwordx4 v28, v[32:35], s[22:23] offset:16
	global_store_dwordx4 v28, v[32:35], s[22:23] offset:32
	global_store_dwordx4 v28, v[32:35], s[22:23] offset:48
	s_and_saveexec_b64 s[22:23], s[0:1]
	s_cbranch_execz .LBB0_91
	v_mov_b32_e32 v32, 0
	global_load_dwordx2 v[36:37], v32, s[2:3] offset:32 sc0 sc1
	global_load_dwordx2 v[18:19], v32, s[2:3] offset:40
	v_mov_b32_e32 v34, s20
	v_mov_b32_e32 v35, s21
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s24, v18
	v_readfirstlane_b32 s25, v19
	s_and_b64 s[24:25], s[24:25], s[20:21]
	s_mul_i32 s25, s25, 24
	s_mul_hi_u32 s26, s24, 24
	s_mul_i32 s24, s24, 24
	s_add_i32 s25, s26, s25
	v_lshl_add_u64 v[24:25], v[16:17], 0, s[24:25]
	global_store_dwordx2 v[24:25], v[36:37], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[18:19], v32, v[34:37], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[18:19], v[36:37]
	s_and_saveexec_b64 s[24:25], vcc
	s_cbranch_execz .LBB0_87
	s_mov_b64 s[26:27], 0
.LBB0_86:
	s_sleep 1
	global_store_dwordx2 v[24:25], v[18:19], off
	v_mov_b32_e32 v16, s20
	v_mov_b32_e32 v17, s21
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[16:17], v32, v[16:19], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[16:17], v[18:19]
	s_or_b64 s[26:27], vcc, s[26:27]
	v_mov_b64_e32 v[18:19], v[16:17]
	s_andn2_b64 exec, exec, s[26:27]
	s_cbranch_execnz .LBB0_86
.LBB0_87:
	s_or_b64 exec, exec, s[24:25]
	v_mov_b32_e32 v19, 0
	global_load_dwordx2 v[16:17], v19, s[2:3] offset:16
	s_mov_b64 s[24:25], exec
	v_mbcnt_lo_u32_b32 v18, s24, 0
	v_mbcnt_hi_u32_b32 v18, s25, v18
	v_cmp_eq_u32_e32 vcc, 0, v18
	s_and_saveexec_b64 s[26:27], vcc
	s_cbranch_execz .LBB0_89
	s_bcnt1_i32_b64 s24, s[24:25]
	v_mov_b32_e32 v18, s24
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[16:17], v[18:19], off offset:8 sc1
.LBB0_89:
	s_or_b64 exec, exec, s[26:27]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[18:19], v[16:17], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[18:19]
	s_cbranch_vccnz .LBB0_91
	global_load_dword v16, v[16:17], off offset:24
	v_mov_b32_e32 v17, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s24, v16
	s_and_b32 m0, s24, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[18:19], v[16:17], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_91:
	s_or_b64 exec, exec, s[22:23]
	v_lshl_add_u64 v[16:17], v[22:23], 0, v[28:29]
	s_branch .LBB0_95
.LBB0_92:
	s_or_b64 exec, exec, s[22:23]
	v_readfirstlane_b32 s22, v18
	s_cmp_eq_u32 s22, 0
	s_cbranch_scc1 .LBB0_94
	s_sleep 1
	s_cbranch_execnz .LBB0_95
	s_branch .LBB0_97
.LBB0_94:
	s_branch .LBB0_97
.LBB0_95:
	v_mov_b32_e32 v18, 1
	s_and_saveexec_b64 s[22:23], s[0:1]
	s_cbranch_execz .LBB0_92
	global_load_dword v18, v[20:21], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v18, 1, v18
	s_branch .LBB0_92
.LBB0_97:
	global_load_dwordx2 v[16:17], v[16:17], off
	s_and_saveexec_b64 s[22:23], s[0:1]
	s_cbranch_execz .LBB0_100
	v_mov_b32_e32 v24, 0
	global_load_dwordx2 v[18:19], v24, s[2:3] offset:40
	global_load_dwordx2 v[34:35], v24, s[2:3] offset:24 sc0 sc1
	global_load_dwordx2 v[20:21], v24, s[2:3]
	s_waitcnt vmcnt(2)
	v_readfirstlane_b32 s24, v18
	v_readfirstlane_b32 s25, v19
	s_add_u32 s26, s24, 1
	s_addc_u32 s27, s25, 0
	s_add_u32 s0, s26, s20
	s_addc_u32 s1, s27, s21
	s_cmp_eq_u64 s[0:1], 0
	s_cselect_b32 s1, s27, s1
	s_cselect_b32 s0, s26, s0
	s_and_b64 s[20:21], s[0:1], s[24:25]
	s_mul_i32 s21, s21, 24
	s_mul_hi_u32 s24, s20, 24
	s_mul_i32 s20, s20, 24
	s_add_i32 s21, s24, s21
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[22:23], v[20:21], 0, s[20:21]
	v_mov_b32_e32 v32, s0
	global_store_dwordx2 v[22:23], v[34:35], off
	v_mov_b32_e32 v33, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[20:21], v24, v[32:35], s[2:3] offset:24 sc0 sc1
	s_mov_b64 s[20:21], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[20:21], v[34:35]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_100
.LBB0_99:
	s_sleep 1
	global_store_dwordx2 v[22:23], v[20:21], off
	v_mov_b32_e32 v18, s0
	v_mov_b32_e32 v19, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[18:19], v24, v[18:21], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[18:19], v[20:21]
	s_or_b64 s[20:21], vcc, s[20:21]
	v_mov_b64_e32 v[20:21], v[18:19]
	s_andn2_b64 exec, exec, s[20:21]
	s_cbranch_execnz .LBB0_99
.LBB0_100:
	s_or_b64 exec, exec, s[22:23]
	v_readfirstlane_b32 s0, v26
	v_mov_b64_e32 v[24:25], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v26
	s_and_saveexec_b64 s[20:21], s[0:1]
	s_cbranch_execz .LBB0_106
	v_mov_b32_e32 v20, 0
	global_load_dwordx2 v[34:35], v20, s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[18:19], v20, s[2:3] offset:40
	global_load_dwordx2 v[22:23], v20, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v18, v18, v34
	v_and_b32_e32 v19, v19, v35
	v_mul_lo_u32 v19, v19, 24
	v_mul_hi_u32 v21, v18, 24
	v_add_u32_e32 v19, v21, v19
	v_mul_lo_u32 v18, v18, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[18:19], v[22:23], 0, v[18:19]
	global_load_dwordx2 v[32:33], v[18:19], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[24:25], v20, v[32:35], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[24:25], v[34:35]
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_105
	s_mov_b64 s[24:25], 0
.LBB0_103:
	s_sleep 1
	global_load_dwordx2 v[18:19], v20, s[2:3] offset:40
	global_load_dwordx2 v[22:23], v20, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v18, v18, v24
	v_and_b32_e32 v21, v19, v25
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[18:19], s[26:27], v18, 24, v[22:23]
	v_mov_b32_e32 v22, v19
	v_mad_u64_u32 v[22:23], s[26:27], v21, 24, v[22:23]
	v_mov_b32_e32 v19, v22
	global_load_dwordx2 v[22:23], v[18:19], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[18:19], v20, v[22:25], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[18:19], v[24:25]
	s_or_b64 s[24:25], vcc, s[24:25]
	v_mov_b64_e32 v[24:25], v[18:19]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBB0_103
	s_or_b64 exec, exec, s[24:25]
	v_mov_b64_e32 v[24:25], v[18:19]
.LBB0_105:
	s_or_b64 exec, exec, s[22:23]
.LBB0_106:
	s_or_b64 exec, exec, s[20:21]
	v_mov_b32_e32 v29, 0
	global_load_dwordx2 v[18:19], v29, s[2:3] offset:40
	global_load_dwordx4 v[20:23], v29, s[2:3]
	v_readfirstlane_b32 s21, v25
	v_readfirstlane_b32 s20, v24
	s_mov_b64 s[22:23], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s24, v18
	v_readfirstlane_b32 s25, v19
	s_and_b64 s[24:25], s[24:25], s[20:21]
	s_mul_i32 s26, s25, 24
	s_mul_hi_u32 s27, s24, 24
	s_add_i32 s27, s27, s26
	s_mul_i32 s26, s24, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[24:25], v[20:21], 0, s[26:27]
	s_and_saveexec_b64 s[26:27], s[0:1]
	s_cbranch_execz .LBB0_108
	v_mov_b64_e32 v[32:33], s[22:23]
	v_mov_b32_e32 v34, 2
	v_mov_b32_e32 v35, 1
	global_store_dwordx4 v[24:25], v[32:35], off offset:8
.LBB0_108:
	s_or_b64 exec, exec, s[26:27]
	s_lshl_b64 s[22:23], s[24:25], 12
	v_lshl_add_u64 v[22:23], v[22:23], 0, s[22:23]
	v_and_b32_e32 v16, 0xffffff1f, v16
	v_or_b32_e32 v16, 0xc0, v16
	v_mov_b32_e32 v19, 0x44412072
	v_mov_b32_e32 v18, 0x65746641
	v_readfirstlane_b32 s22, v22
	v_readfirstlane_b32 s23, v23
	v_mov_b32_e32 v32, 0x3d293776
	v_mov_b32_e32 v33, 0x202c6625
	v_mov_b32_e32 v34, 0x32762843
	s_nop 1
	global_store_dwordx4 v28, v[16:19], s[22:23]
	s_nop 1
	v_mov_b32_e32 v16, 0x20203a44
	v_mov_b32_e32 v17, 0x36762841
	v_mov_b32_e32 v18, 0x66253d29
	v_mov_b32_e32 v19, 0x2842202c
	global_store_dwordx4 v28, v[16:19], s[22:23] offset:16
	v_mov_b32_e32 v35, v18
	global_store_dwordx4 v28, v[32:35], s[22:23] offset:32
	v_mov_b32_e32 v16, 10
	v_mov_b32_e32 v17, v29
	v_mov_b32_e32 v18, v29
	v_mov_b32_e32 v19, v29
	global_store_dwordx4 v28, v[16:19], s[22:23] offset:48
	s_and_saveexec_b64 s[22:23], s[0:1]
	s_cbranch_execz .LBB0_116
	v_mov_b32_e32 v32, 0
	global_load_dwordx2 v[36:37], v32, s[2:3] offset:32 sc0 sc1
	global_load_dwordx2 v[16:17], v32, s[2:3] offset:40
	v_mov_b32_e32 v34, s20
	v_mov_b32_e32 v35, s21
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s24, v16
	v_readfirstlane_b32 s25, v17
	s_and_b64 s[24:25], s[24:25], s[20:21]
	s_mul_i32 s25, s25, 24
	s_mul_hi_u32 s26, s24, 24
	s_mul_i32 s24, s24, 24
	s_add_i32 s25, s26, s25
	v_lshl_add_u64 v[20:21], v[20:21], 0, s[24:25]
	global_store_dwordx2 v[20:21], v[36:37], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[18:19], v32, v[34:37], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[18:19], v[36:37]
	s_and_saveexec_b64 s[24:25], vcc
	s_cbranch_execz .LBB0_112
	s_mov_b64 s[26:27], 0
.LBB0_111:
	s_sleep 1
	global_store_dwordx2 v[20:21], v[18:19], off
	v_mov_b32_e32 v16, s20
	v_mov_b32_e32 v17, s21
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[16:17], v32, v[16:19], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[16:17], v[18:19]
	s_or_b64 s[26:27], vcc, s[26:27]
	v_mov_b64_e32 v[18:19], v[16:17]
	s_andn2_b64 exec, exec, s[26:27]
	s_cbranch_execnz .LBB0_111
.LBB0_112:
	s_or_b64 exec, exec, s[24:25]
	v_mov_b32_e32 v19, 0
	global_load_dwordx2 v[16:17], v19, s[2:3] offset:16
	s_mov_b64 s[24:25], exec
	v_mbcnt_lo_u32_b32 v18, s24, 0
	v_mbcnt_hi_u32_b32 v18, s25, v18
	v_cmp_eq_u32_e32 vcc, 0, v18
	s_and_saveexec_b64 s[26:27], vcc
	s_cbranch_execz .LBB0_114
	s_bcnt1_i32_b64 s24, s[24:25]
	v_mov_b32_e32 v18, s24
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[16:17], v[18:19], off offset:8 sc1
.LBB0_114:
	s_or_b64 exec, exec, s[26:27]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[18:19], v[16:17], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[18:19]
	s_cbranch_vccnz .LBB0_116
	global_load_dword v16, v[16:17], off offset:24
	v_mov_b32_e32 v17, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s24, v16
	s_and_b32 m0, s24, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[18:19], v[16:17], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_116:
	s_or_b64 exec, exec, s[22:23]
	v_lshl_add_u64 v[16:17], v[22:23], 0, v[28:29]
	s_branch .LBB0_120
.LBB0_117:
	s_or_b64 exec, exec, s[22:23]
	v_readfirstlane_b32 s22, v18
	s_cmp_eq_u32 s22, 0
	s_cbranch_scc1 .LBB0_119
	s_sleep 1
	s_cbranch_execnz .LBB0_120
	s_branch .LBB0_122
.LBB0_119:
	s_branch .LBB0_122
.LBB0_120:
	v_mov_b32_e32 v18, 1
	s_and_saveexec_b64 s[22:23], s[0:1]
	s_cbranch_execz .LBB0_117
	global_load_dword v18, v[24:25], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v18, 1, v18
	s_branch .LBB0_117
.LBB0_122:
	global_load_dwordx2 v[16:17], v[16:17], off
	s_and_saveexec_b64 s[22:23], s[0:1]
	s_cbranch_execz .LBB0_125
	v_mov_b32_e32 v24, 0
	global_load_dwordx2 v[18:19], v24, s[2:3] offset:40
	global_load_dwordx2 v[34:35], v24, s[2:3] offset:24 sc0 sc1
	global_load_dwordx2 v[20:21], v24, s[2:3]
	s_waitcnt vmcnt(2)
	v_readfirstlane_b32 s24, v18
	v_readfirstlane_b32 s25, v19
	s_add_u32 s26, s24, 1
	s_addc_u32 s27, s25, 0
	s_add_u32 s0, s26, s20
	s_addc_u32 s1, s27, s21
	s_cmp_eq_u64 s[0:1], 0
	s_cselect_b32 s1, s27, s1
	s_cselect_b32 s0, s26, s0
	s_and_b64 s[20:21], s[0:1], s[24:25]
	s_mul_i32 s21, s21, 24
	s_mul_hi_u32 s24, s20, 24
	s_mul_i32 s20, s20, 24
	s_add_i32 s21, s24, s21
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[22:23], v[20:21], 0, s[20:21]
	v_mov_b32_e32 v32, s0
	global_store_dwordx2 v[22:23], v[34:35], off
	v_mov_b32_e32 v33, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[20:21], v24, v[32:35], s[2:3] offset:24 sc0 sc1
	s_mov_b64 s[20:21], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[20:21], v[34:35]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_125
.LBB0_124:
	s_sleep 1
	global_store_dwordx2 v[22:23], v[20:21], off
	v_mov_b32_e32 v18, s0
	v_mov_b32_e32 v19, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[18:19], v24, v[18:21], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[18:19], v[20:21]
	s_or_b64 s[20:21], vcc, s[20:21]
	v_mov_b64_e32 v[20:21], v[18:19]
	s_andn2_b64 exec, exec, s[20:21]
	s_cbranch_execnz .LBB0_124
.LBB0_125:
	s_or_b64 exec, exec, s[22:23]
	v_readfirstlane_b32 s0, v26
	v_mov_b64_e32 v[24:25], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v26
	s_and_saveexec_b64 s[20:21], s[0:1]
	s_cbranch_execz .LBB0_131
	v_mov_b32_e32 v20, 0
	global_load_dwordx2 v[34:35], v20, s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[18:19], v20, s[2:3] offset:40
	global_load_dwordx2 v[22:23], v20, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v18, v18, v34
	v_and_b32_e32 v19, v19, v35
	v_mul_lo_u32 v19, v19, 24
	v_mul_hi_u32 v21, v18, 24
	v_add_u32_e32 v19, v21, v19
	v_mul_lo_u32 v18, v18, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[18:19], v[22:23], 0, v[18:19]
	global_load_dwordx2 v[32:33], v[18:19], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[24:25], v20, v[32:35], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[24:25], v[34:35]
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_130
	s_mov_b64 s[24:25], 0
.LBB0_128:
	s_sleep 1
	global_load_dwordx2 v[18:19], v20, s[2:3] offset:40
	global_load_dwordx2 v[22:23], v20, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v18, v18, v24
	v_and_b32_e32 v21, v19, v25
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[18:19], s[26:27], v18, 24, v[22:23]
	v_mov_b32_e32 v22, v19
	v_mad_u64_u32 v[22:23], s[26:27], v21, 24, v[22:23]
	v_mov_b32_e32 v19, v22
	global_load_dwordx2 v[22:23], v[18:19], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[18:19], v20, v[22:25], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[18:19], v[24:25]
	s_or_b64 s[24:25], vcc, s[24:25]
	v_mov_b64_e32 v[24:25], v[18:19]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBB0_128
	s_or_b64 exec, exec, s[24:25]
	v_mov_b64_e32 v[24:25], v[18:19]
.LBB0_130:
	s_or_b64 exec, exec, s[22:23]
.LBB0_131:
	s_or_b64 exec, exec, s[20:21]
	v_mov_b32_e32 v26, 0
	global_load_dwordx2 v[18:19], v26, s[2:3] offset:40
	global_load_dwordx4 v[20:23], v26, s[2:3]
	v_readfirstlane_b32 s21, v25
	v_readfirstlane_b32 s20, v24
	s_mov_b64 s[22:23], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s24, v18
	v_readfirstlane_b32 s25, v19
	s_and_b64 s[24:25], s[24:25], s[20:21]
	s_mul_i32 s26, s25, 24
	s_mul_hi_u32 s27, s24, 24
	s_add_i32 s27, s27, s26
	s_mul_i32 s26, s24, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[24:25], v[20:21], 0, s[26:27]
	s_and_saveexec_b64 s[26:27], s[0:1]
	s_cbranch_execz .LBB0_133
	v_mov_b64_e32 v[32:33], s[22:23]
	v_mov_b32_e32 v34, 2
	v_mov_b32_e32 v35, 1
	global_store_dwordx4 v[24:25], v[32:35], off offset:8
.LBB0_133:
	s_or_b64 exec, exec, s[26:27]
	s_lshl_b64 s[22:23], s[24:25], 12
	v_lshl_add_u64 v[22:23], v[22:23], 0, s[22:23]
	v_and_b32_e32 v16, 0xffffff1d, v16
	s_mov_b32 s24, 0
	v_cvt_f64_f32_e32 v[18:19], v27
	v_or_b32_e32 v16, 0x62, v16
	v_readfirstlane_b32 s22, v22
	v_readfirstlane_b32 s23, v23
	s_mov_b32 s25, s24
	v_cvt_f64_f32_e32 v[32:33], v30
	v_cvt_f64_f32_e32 v[34:35], v31
	s_nop 1
	global_store_dwordx4 v28, v[16:19], s[22:23]
	global_store_dwordx4 v28, v[32:35], s[22:23] offset:16
	s_mov_b32 s26, s24
	s_mov_b32 s27, s24
	v_mov_b64_e32 v[16:17], s[24:25]
	v_mov_b64_e32 v[18:19], s[26:27]
	global_store_dwordx4 v28, v[16:19], s[22:23] offset:32
	global_store_dwordx4 v28, v[16:19], s[22:23] offset:48
	s_and_saveexec_b64 s[22:23], s[0:1]
	s_cbranch_execz .LBB0_141
	v_mov_b32_e32 v22, 0
	global_load_dwordx2 v[28:29], v22, s[2:3] offset:32 sc0 sc1
	global_load_dwordx2 v[16:17], v22, s[2:3] offset:40
	v_mov_b32_e32 v26, s20
	v_mov_b32_e32 v27, s21
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s24, v16
	v_readfirstlane_b32 s25, v17
	s_and_b64 s[24:25], s[24:25], s[20:21]
	s_mul_i32 s25, s25, 24
	s_mul_hi_u32 s26, s24, 24
	s_mul_i32 s24, s24, 24
	s_add_i32 s25, s26, s25
	v_lshl_add_u64 v[20:21], v[20:21], 0, s[24:25]
	global_store_dwordx2 v[20:21], v[28:29], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[18:19], v22, v[26:29], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[18:19], v[28:29]
	s_and_saveexec_b64 s[24:25], vcc
	s_cbranch_execz .LBB0_137
	s_mov_b64 s[26:27], 0
.LBB0_136:
	s_sleep 1
	global_store_dwordx2 v[20:21], v[18:19], off
	v_mov_b32_e32 v16, s20
	v_mov_b32_e32 v17, s21
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[16:17], v22, v[16:19], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[16:17], v[18:19]
	s_or_b64 s[26:27], vcc, s[26:27]
	v_mov_b64_e32 v[18:19], v[16:17]
	s_andn2_b64 exec, exec, s[26:27]
	s_cbranch_execnz .LBB0_136
.LBB0_137:
	s_or_b64 exec, exec, s[24:25]
	v_mov_b32_e32 v19, 0
	global_load_dwordx2 v[16:17], v19, s[2:3] offset:16
	s_mov_b64 s[24:25], exec
	v_mbcnt_lo_u32_b32 v18, s24, 0
	v_mbcnt_hi_u32_b32 v18, s25, v18
	v_cmp_eq_u32_e32 vcc, 0, v18
	s_and_saveexec_b64 s[26:27], vcc
	s_cbranch_execz .LBB0_139
	s_bcnt1_i32_b64 s24, s[24:25]
	v_mov_b32_e32 v18, s24
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[16:17], v[18:19], off offset:8 sc1
.LBB0_139:
	s_or_b64 exec, exec, s[26:27]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[18:19], v[16:17], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[18:19]
	s_cbranch_vccnz .LBB0_141
	global_load_dword v16, v[16:17], off offset:24
	v_mov_b32_e32 v17, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s24, v16
	s_and_b32 m0, s24, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[18:19], v[16:17], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_141:
	s_or_b64 exec, exec, s[22:23]
	s_branch .LBB0_145
.LBB0_142:
	s_or_b64 exec, exec, s[22:23]
	v_readfirstlane_b32 s22, v16
	s_cmp_eq_u32 s22, 0
	s_cbranch_scc1 .LBB0_144
	s_sleep 1
	s_cbranch_execnz .LBB0_145
	s_branch .LBB0_147
.LBB0_144:
	s_branch .LBB0_147
.LBB0_145:
	v_mov_b32_e32 v16, 1
	s_and_saveexec_b64 s[22:23], s[0:1]
	s_cbranch_execz .LBB0_142
	global_load_dword v16, v[24:25], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v16, 1, v16
	s_branch .LBB0_142
.LBB0_147:
	s_and_saveexec_b64 s[22:23], s[0:1]
	s_cbranch_execz .LBB0_150
	v_mov_b32_e32 v22, 0
	global_load_dwordx2 v[16:17], v22, s[2:3] offset:40
	global_load_dwordx2 v[26:27], v22, s[2:3] offset:24 sc0 sc1
	global_load_dwordx2 v[18:19], v22, s[2:3]
	s_waitcnt vmcnt(2)
	v_readfirstlane_b32 s24, v16
	v_readfirstlane_b32 s25, v17
	s_add_u32 s26, s24, 1
	s_addc_u32 s27, s25, 0
	s_add_u32 s0, s26, s20
	s_addc_u32 s1, s27, s21
	s_cmp_eq_u64 s[0:1], 0
	s_cselect_b32 s1, s27, s1
	s_cselect_b32 s0, s26, s0
	s_and_b64 s[20:21], s[0:1], s[24:25]
	s_mul_i32 s21, s21, 24
	s_mul_hi_u32 s24, s20, 24
	s_mul_i32 s20, s20, 24
	s_add_i32 s21, s24, s21
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[20:21], v[18:19], 0, s[20:21]
	v_mov_b32_e32 v24, s0
	global_store_dwordx2 v[20:21], v[26:27], off
	v_mov_b32_e32 v25, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[18:19], v22, v[24:27], s[2:3] offset:24 sc0 sc1
	s_mov_b64 s[20:21], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[18:19], v[26:27]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_150
.LBB0_149:
	s_sleep 1
	global_store_dwordx2 v[20:21], v[18:19], off
	v_mov_b32_e32 v16, s0
	v_mov_b32_e32 v17, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[16:17], v22, v[16:19], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[16:17], v[18:19]
	s_or_b64 s[20:21], vcc, s[20:21]
	v_mov_b64_e32 v[18:19], v[16:17]
	s_andn2_b64 exec, exec, s[20:21]
	s_cbranch_execnz .LBB0_149
.LBB0_150:
	s_or_b64 exec, exec, s[22:23]
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	.LBB0_2:
	;;#ASMEND
	;;#ASMSTART
		s_endpgm
	;;#ASMEND
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
		.amdhsa_next_free_vgpr 38
		.amdhsa_next_free_sgpr 28
		.amdhsa_accum_offset 40
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

	.set _Z9vectorAddPKfS0_Pfi.num_vgpr, 38
	.set _Z9vectorAddPKfS0_Pfi.num_agpr, 0
	.set _Z9vectorAddPKfS0_Pfi.numbered_sgpr, 28
	.set _Z9vectorAddPKfS0_Pfi.num_named_barrier, 0
	.set _Z9vectorAddPKfS0_Pfi.private_seg_size, 0
	.set _Z9vectorAddPKfS0_Pfi.uses_vcc, 1
	.set _Z9vectorAddPKfS0_Pfi.uses_flat_scratch, 0
	.set _Z9vectorAddPKfS0_Pfi.has_dyn_sized_stack, 0
	.set _Z9vectorAddPKfS0_Pfi.has_recursion, 0
	.set _Z9vectorAddPKfS0_Pfi.has_indirect_call, 0
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.text
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
- .agpr_count: 0
  .args:
  - .address_space: global
    .offset: 0
    .size: 8
    .value_kind: global_buffer
  - .address_space: global
    .offset: 8
    .size: 8
    .value_kind: global_buffer
  - .address_space: global
    .offset: 16
    .size: 8
    .value_kind: global_buffer
  - .offset: 24
    .size: 4
    .value_kind: by_value
  - .offset: 88
    .size: 4
    .value_kind: hidden_block_count_x
  - .offset: 92
    .size: 4
    .value_kind: hidden_block_count_y
  - .offset: 96
    .size: 4
    .value_kind: hidden_block_count_z
  - .offset: 100
    .size: 2
    .value_kind: hidden_group_size_x
  - .offset: 102
    .size: 2
    .value_kind: hidden_group_size_y
  - .offset: 104
    .size: 2
    .value_kind: hidden_group_size_z
  - .offset: 106
    .size: 2
    .value_kind: hidden_remainder_x
  - .offset: 108
    .size: 2
    .value_kind: hidden_remainder_y
  - .offset: 110
    .size: 2
    .value_kind: hidden_remainder_z
  - .offset: 128
    .size: 8
    .value_kind: hidden_global_offset_x
  - .offset: 136
    .size: 8
    .value_kind: hidden_global_offset_y
  - .offset: 144
    .size: 8
    .value_kind: hidden_global_offset_z
  - .offset: 152
    .size: 2
    .value_kind: hidden_grid_dims
  - .offset: 168
    .size: 8
    .value_kind: hidden_hostcall_buffer
  .group_segment_fixed_size: 0
  .kernarg_segment_align: 8
  .kernarg_segment_size: 288
  .max_flat_workgroup_size: 256
  .name: _Z9vectorAddPKfS0_Pfi
  .private_segment_fixed_size: 0
  .sgpr_count: 34
  .sgpr_spill_count: 0
  .symbol: _Z9vectorAddPKfS0_Pfi.kd
  .uniform_work_group_size: 1
  .uses_dynamic_stack: false
  .vgpr_count: 38
  .vgpr_spill_count: 0
  .wavefront_size: 64
amdhsa.target: amdgcn-amd-amdhsa--gfx950
amdhsa.version:
- 1
- 2
...
...

	.end_amdgpu_metadata
