	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.text
	.globl	_Z15sharedMemKernelPKiPii
	.p2align	8
	.type	_Z15sharedMemKernelPKiPii,@function
_Z15sharedMemKernelPKiPii:
	;;#ASMSTART
	s_mov_b64 s[18:19], s[0:1]
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
		s_load_dword s3, s[0:1], 0x24
	;;#ASMEND
	;;#ASMSTART
		s_load_dword s8, s[0:1], 0x10
	;;#ASMEND
	;;#ASMSTART
		s_load_dwordx4 s[4:7], s[0:1], 0x0
	;;#ASMEND
	;;#ASMSTART
		v_mov_b32_e32 v3, 0
	;;#ASMEND
	;;#ASMSTART
		s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
		s_and_b32 s3, s3, 0xffff
	;;#ASMEND
	;;#ASMSTART
		s_mul_i32 s0, s2, s3
	;;#ASMEND
	;;#ASMSTART
		v_add_u32_e32 v2, s0, v0
	;;#ASMEND
	;;#ASMSTART
		v_cmp_gt_i32_e32 vcc, s8, v2
	;;#ASMEND
	;;#ASMSTART
		s_and_saveexec_b64 s[0:1], vcc
	;;#ASMEND
	;;#ASMSTART
		s_cbranch_execz .LBB0_2
	;;#ASMEND
	;;#ASMSTART
		v_mov_b32_e32 v4, s4
	;;#ASMEND
	;;#ASMSTART
		v_mov_b32_e32 v5, s5
	;;#ASMEND
	;;#ASMSTART
		v_ashrrev_i32_e32 v3, 31, v2
	;;#ASMEND
	;;#ASMSTART
		v_lshl_add_u64 v[2:3], v[2:3], 2, v[4:5]
	;;#ASMEND
	;;#ASMSTART
		global_load_dword v3, v[2:3], off
	;;#ASMEND
	;;#ASMSTART
	s_mov_b64 s[0:1], s[18:19]
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v10, v6
	;;#ASMEND
	s_load_dwordx2 s[2:3], s[0:1], 0xa0
	v_mbcnt_lo_u32_b32 v8, -1, 0
	v_mbcnt_hi_u32_b32 v22, -1, v8
	v_mov_b64_e32 v[16:17], 0
	v_readfirstlane_b32 s0, v22
	s_nop 1
	v_cmp_eq_u32_e64 s[0:1], s0, v22
	s_and_saveexec_b64 s[20:21], s[0:1]
	s_cbranch_execz .LBBPRINTF0_6
	v_mov_b32_e32 v11, 0
	s_waitcnt lgkmcnt(0)
	global_load_dwordx2 v[14:15], v11, s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[8:9], v11, s[2:3] offset:40
	global_load_dwordx2 v[12:13], v11, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v8, v8, v14
	v_and_b32_e32 v9, v9, v15
	v_mul_lo_u32 v9, v9, 24
	v_mul_hi_u32 v16, v8, 24
	v_add_u32_e32 v9, v16, v9
	v_mul_lo_u32 v8, v8, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[8:9], v[12:13], 0, v[8:9]
	global_load_dwordx2 v[12:13], v[8:9], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[16:17], v11, v[12:15], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[16:17], v[14:15]
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBBPRINTF0_5
	s_mov_b64 s[24:25], 0
.LBBPRINTF0_3:
	s_sleep 1
	global_load_dwordx2 v[8:9], v11, s[2:3] offset:40
	global_load_dwordx2 v[12:13], v11, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v8, v8, v16
	v_and_b32_e32 v14, v9, v17
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[8:9], s[26:27], v8, 24, v[12:13]
	v_mov_b32_e32 v12, v9
	v_mad_u64_u32 v[12:13], s[26:27], v14, 24, v[12:13]
	v_mov_b32_e32 v9, v12
	global_load_dwordx2 v[14:15], v[8:9], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[8:9], v11, v[14:17], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[8:9], v[16:17]
	s_or_b64 s[24:25], vcc, s[24:25]
	v_mov_b64_e32 v[16:17], v[8:9]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBBPRINTF0_3
	s_or_b64 exec, exec, s[24:25]
	v_mov_b64_e32 v[16:17], v[8:9]
.LBBPRINTF0_5:
	s_or_b64 exec, exec, s[22:23]
.LBBPRINTF0_6:
	s_or_b64 exec, exec, s[20:21]
	v_mov_b32_e32 v19, 0
	s_waitcnt lgkmcnt(0)
	global_load_dwordx2 v[8:9], v19, s[2:3] offset:40
	global_load_dwordx4 v[12:15], v19, s[2:3]
	v_readfirstlane_b32 s21, v17
	v_readfirstlane_b32 s20, v16
	s_mov_b64 s[22:23], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s24, v8
	v_readfirstlane_b32 s25, v9
	s_and_b64 s[24:25], s[24:25], s[20:21]
	s_mul_i32 s26, s25, 24
	s_mul_hi_u32 s27, s24, 24
	s_add_i32 s27, s27, s26
	s_mul_i32 s26, s24, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[8:9], v[12:13], 0, s[26:27]
	s_and_saveexec_b64 s[26:27], s[0:1]
	s_cbranch_execz .LBB0_8
	v_mov_b64_e32 v[24:25], s[22:23]
	v_mov_b32_e32 v26, 2
	v_mov_b32_e32 v27, 1
	global_store_dwordx4 v[8:9], v[24:27], off offset:8
.LBB0_8:
	s_or_b64 exec, exec, s[26:27]
	s_lshl_b64 s[22:23], s[24:25], 12
	v_lshl_add_u64 v[16:17], v[14:15], 0, s[22:23]
	s_mov_b32 s24, 0
	v_lshlrev_b32_e32 v18, 6, v22
	v_mov_b32_e32 v24, 33
	v_mov_b32_e32 v25, v19
	v_mov_b32_e32 v26, v19
	v_mov_b32_e32 v27, v19
	v_readfirstlane_b32 s22, v16
	v_readfirstlane_b32 s23, v17
	s_mov_b32 s25, s24
	s_mov_b32 s26, s24
	s_mov_b32 s27, s24
	s_nop 1
	global_store_dwordx4 v18, v[24:27], s[22:23]
	s_nop 1
	v_mov_b64_e32 v[24:25], s[24:25]
	v_mov_b64_e32 v[26:27], s[26:27]
	global_store_dwordx4 v18, v[24:27], s[22:23] offset:16
	global_store_dwordx4 v18, v[24:27], s[22:23] offset:32
	global_store_dwordx4 v18, v[24:27], s[22:23] offset:48
	s_and_saveexec_b64 s[22:23], s[0:1]
	s_cbranch_execz .LBB0_16
	v_mov_b32_e32 v11, 0
	global_load_dwordx2 v[26:27], v11, s[2:3] offset:32 sc0 sc1
	global_load_dwordx2 v[14:15], v11, s[2:3] offset:40
	v_mov_b32_e32 v24, s20
	v_mov_b32_e32 v25, s21
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v14, s20, v14
	v_and_b32_e32 v15, s21, v15
	v_mul_lo_u32 v15, v15, 24
	v_mul_hi_u32 v20, v14, 24
	v_mul_lo_u32 v14, v14, 24
	v_add_u32_e32 v15, v20, v15
	v_lshl_add_u64 v[20:21], v[12:13], 0, v[14:15]
	global_store_dwordx2 v[20:21], v[26:27], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[14:15], v11, v[24:27], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[14:15], v[26:27]
	s_and_saveexec_b64 s[24:25], vcc
	s_cbranch_execz .LBB0_12
	s_mov_b64 s[26:27], 0
.LBB0_11:
	s_sleep 1
	global_store_dwordx2 v[20:21], v[14:15], off
	v_mov_b32_e32 v12, s20
	v_mov_b32_e32 v13, s21
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[12:13], v11, v[12:15], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[12:13], v[14:15]
	s_or_b64 s[26:27], vcc, s[26:27]
	v_mov_b64_e32 v[14:15], v[12:13]
	s_andn2_b64 exec, exec, s[26:27]
	s_cbranch_execnz .LBB0_11
.LBB0_12:
	s_or_b64 exec, exec, s[24:25]
	v_mov_b32_e32 v15, 0
	global_load_dwordx2 v[12:13], v15, s[2:3] offset:16
	s_mov_b64 s[24:25], exec
	v_mbcnt_lo_u32_b32 v11, s24, 0
	v_mbcnt_hi_u32_b32 v11, s25, v11
	v_cmp_eq_u32_e32 vcc, 0, v11
	s_and_saveexec_b64 s[26:27], vcc
	s_cbranch_execz .LBB0_14
	s_bcnt1_i32_b64 s24, s[24:25]
	v_mov_b32_e32 v14, s24
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[12:13], v[14:15], off offset:8 sc1
.LBB0_14:
	s_or_b64 exec, exec, s[26:27]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[14:15], v[12:13], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[14:15]
	s_cbranch_vccnz .LBB0_16
	global_load_dword v12, v[12:13], off offset:24
	v_mov_b32_e32 v13, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s24, v12
	s_and_b32 m0, s24, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[14:15], v[12:13], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_16:
	s_or_b64 exec, exec, s[22:23]
	v_lshl_add_u64 v[12:13], v[16:17], 0, v[18:19]
	s_branch .LBB0_20
.LBB0_17:
	s_or_b64 exec, exec, s[22:23]
	v_readfirstlane_b32 s22, v11
	s_cmp_eq_u32 s22, 0
	s_cbranch_scc1 .LBB0_19
	s_sleep 1
	s_cbranch_execnz .LBB0_20
	s_branch .LBB0_22
.LBB0_19:
	s_branch .LBB0_22
.LBB0_20:
	v_mov_b32_e32 v11, 1
	s_and_saveexec_b64 s[22:23], s[0:1]
	s_cbranch_execz .LBB0_17
	global_load_dword v11, v[8:9], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v11, 1, v11
	s_branch .LBB0_17
.LBB0_22:
	global_load_dwordx2 v[12:13], v[12:13], off
	s_and_saveexec_b64 s[22:23], s[0:1]
	s_cbranch_execz .LBB0_25
	v_mov_b32_e32 v11, 0
	global_load_dwordx2 v[8:9], v11, s[2:3] offset:40
	global_load_dwordx2 v[18:19], v11, s[2:3] offset:24 sc0 sc1
	global_load_dwordx2 v[14:15], v11, s[2:3]
	s_waitcnt vmcnt(2)
	v_readfirstlane_b32 s24, v8
	v_readfirstlane_b32 s25, v9
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
	v_lshl_add_u64 v[8:9], v[14:15], 0, s[20:21]
	v_mov_b32_e32 v16, s0
	global_store_dwordx2 v[8:9], v[18:19], off
	v_mov_b32_e32 v17, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[16:17], v11, v[16:19], s[2:3] offset:24 sc0 sc1
	s_mov_b64 s[20:21], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[16:17], v[18:19]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_25
.LBB0_24:
	s_sleep 1
	global_store_dwordx2 v[8:9], v[16:17], off
	v_mov_b32_e32 v14, s0
	v_mov_b32_e32 v15, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[14:15], v11, v[14:17], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[14:15], v[16:17]
	s_or_b64 s[20:21], vcc, s[20:21]
	v_mov_b64_e32 v[16:17], v[14:15]
	s_andn2_b64 exec, exec, s[20:21]
	s_cbranch_execnz .LBB0_24
.LBB0_25:
	s_or_b64 exec, exec, s[22:23]
	v_readfirstlane_b32 s0, v22
	v_mov_b64_e32 v[20:21], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v22
	s_and_saveexec_b64 s[20:21], s[0:1]
	s_cbranch_execz .LBB0_31
	v_mov_b32_e32 v11, 0
	global_load_dwordx2 v[16:17], v11, s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[8:9], v11, s[2:3] offset:40
	global_load_dwordx2 v[14:15], v11, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v8, v8, v16
	v_and_b32_e32 v9, v9, v17
	v_mul_lo_u32 v9, v9, 24
	v_mul_hi_u32 v18, v8, 24
	v_add_u32_e32 v9, v18, v9
	v_mul_lo_u32 v8, v8, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[8:9], v[14:15], 0, v[8:9]
	global_load_dwordx2 v[14:15], v[8:9], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[20:21], v11, v[14:17], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[20:21], v[16:17]
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_30
	s_mov_b64 s[24:25], 0
.LBB0_28:
	s_sleep 1
	global_load_dwordx2 v[8:9], v11, s[2:3] offset:40
	global_load_dwordx2 v[14:15], v11, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v8, v8, v20
	v_and_b32_e32 v16, v9, v21
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[8:9], s[26:27], v8, 24, v[14:15]
	v_mov_b32_e32 v14, v9
	v_mad_u64_u32 v[14:15], s[26:27], v16, 24, v[14:15]
	v_mov_b32_e32 v9, v14
	global_load_dwordx2 v[18:19], v[8:9], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[8:9], v11, v[18:21], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[8:9], v[20:21]
	s_or_b64 s[24:25], vcc, s[24:25]
	v_mov_b64_e32 v[20:21], v[8:9]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBB0_28
	s_or_b64 exec, exec, s[24:25]
	v_mov_b64_e32 v[20:21], v[8:9]
.LBB0_30:
	s_or_b64 exec, exec, s[22:23]
.LBB0_31:
	s_or_b64 exec, exec, s[20:21]
	v_mov_b32_e32 v25, 0
	global_load_dwordx2 v[8:9], v25, s[2:3] offset:40
	global_load_dwordx4 v[16:19], v25, s[2:3]
	v_readfirstlane_b32 s21, v21
	v_readfirstlane_b32 s20, v20
	s_mov_b64 s[22:23], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s24, v8
	v_readfirstlane_b32 s25, v9
	s_and_b64 s[24:25], s[24:25], s[20:21]
	s_mul_i32 s26, s25, 24
	s_mul_hi_u32 s27, s24, 24
	s_add_i32 s27, s27, s26
	s_mul_i32 s26, s24, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[8:9], v[16:17], 0, s[26:27]
	s_and_saveexec_b64 s[26:27], s[0:1]
	s_cbranch_execz .LBB0_33
	v_mov_b64_e32 v[26:27], s[22:23]
	v_mov_b32_e32 v28, 2
	v_mov_b32_e32 v29, 1
	global_store_dwordx4 v[8:9], v[26:29], off offset:8
.LBB0_33:
	s_or_b64 exec, exec, s[26:27]
	s_lshl_b64 s[22:23], s[24:25], 12
	v_lshl_add_u64 v[18:19], v[18:19], 0, s[22:23]
	v_and_b32_e32 v11, 0xffffff1f, v12
	v_or_b32_e32 v12, 0xc0, v11
	v_lshlrev_b32_e32 v24, 6, v22
	v_mov_b32_e32 v15, 0x62207265
	v_mov_b32_e32 v14, 0x7466415b
	v_readfirstlane_b32 s22, v18
	v_readfirstlane_b32 s23, v19
	s_nop 4
	global_store_dwordx4 v24, v[12:15], s[22:23]
	s_nop 1
	v_mov_b32_e32 v12, 0x69727261
	v_mov_b32_e32 v13, 0x205d7265
	v_mov_b32_e32 v14, 0x75646552
	v_mov_b32_e32 v15, 0x6f697463
	global_store_dwordx4 v24, v[12:15], s[22:23] offset:16
	s_nop 1
	v_mov_b32_e32 v12, 0x6572206e
	v_mov_b32_e32 v13, 0x746c7573
	v_mov_b32_e32 v14, 0x3176203a
	v_mov_b32_e32 v15, 0xa64253d
	global_store_dwordx4 v24, v[12:15], s[22:23] offset:32
	s_nop 1
	v_mov_b32_e32 v12, v25
	v_mov_b32_e32 v13, v25
	v_mov_b32_e32 v14, v25
	v_mov_b32_e32 v15, v25
	global_store_dwordx4 v24, v[12:15], s[22:23] offset:48
	s_and_saveexec_b64 s[22:23], s[0:1]
	s_cbranch_execz .LBB0_41
	v_mov_b32_e32 v11, 0
	global_load_dwordx2 v[28:29], v11, s[2:3] offset:32 sc0 sc1
	global_load_dwordx2 v[12:13], v11, s[2:3] offset:40
	v_mov_b32_e32 v26, s20
	v_mov_b32_e32 v27, s21
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s24, v12
	v_readfirstlane_b32 s25, v13
	s_and_b64 s[24:25], s[24:25], s[20:21]
	s_mul_i32 s25, s25, 24
	s_mul_hi_u32 s26, s24, 24
	s_mul_i32 s24, s24, 24
	s_add_i32 s25, s26, s25
	v_lshl_add_u64 v[16:17], v[16:17], 0, s[24:25]
	global_store_dwordx2 v[16:17], v[28:29], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[14:15], v11, v[26:29], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[14:15], v[28:29]
	s_and_saveexec_b64 s[24:25], vcc
	s_cbranch_execz .LBB0_37
	s_mov_b64 s[26:27], 0
.LBB0_36:
	s_sleep 1
	global_store_dwordx2 v[16:17], v[14:15], off
	v_mov_b32_e32 v12, s20
	v_mov_b32_e32 v13, s21
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[12:13], v11, v[12:15], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[12:13], v[14:15]
	s_or_b64 s[26:27], vcc, s[26:27]
	v_mov_b64_e32 v[14:15], v[12:13]
	s_andn2_b64 exec, exec, s[26:27]
	s_cbranch_execnz .LBB0_36
.LBB0_37:
	s_or_b64 exec, exec, s[24:25]
	v_mov_b32_e32 v15, 0
	global_load_dwordx2 v[12:13], v15, s[2:3] offset:16
	s_mov_b64 s[24:25], exec
	v_mbcnt_lo_u32_b32 v11, s24, 0
	v_mbcnt_hi_u32_b32 v11, s25, v11
	v_cmp_eq_u32_e32 vcc, 0, v11
	s_and_saveexec_b64 s[26:27], vcc
	s_cbranch_execz .LBB0_39
	s_bcnt1_i32_b64 s24, s[24:25]
	v_mov_b32_e32 v14, s24
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[12:13], v[14:15], off offset:8 sc1
.LBB0_39:
	s_or_b64 exec, exec, s[26:27]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[14:15], v[12:13], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[14:15]
	s_cbranch_vccnz .LBB0_41
	global_load_dword v12, v[12:13], off offset:24
	v_mov_b32_e32 v13, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s24, v12
	s_and_b32 m0, s24, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[14:15], v[12:13], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_41:
	s_or_b64 exec, exec, s[22:23]
	v_lshl_add_u64 v[12:13], v[18:19], 0, v[24:25]
	s_branch .LBB0_45
.LBB0_42:
	s_or_b64 exec, exec, s[22:23]
	v_readfirstlane_b32 s22, v11
	s_cmp_eq_u32 s22, 0
	s_cbranch_scc1 .LBB0_44
	s_sleep 1
	s_cbranch_execnz .LBB0_45
	s_branch .LBB0_47
.LBB0_44:
	s_branch .LBB0_47
.LBB0_45:
	v_mov_b32_e32 v11, 1
	s_and_saveexec_b64 s[22:23], s[0:1]
	s_cbranch_execz .LBB0_42
	global_load_dword v11, v[8:9], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v11, 1, v11
	s_branch .LBB0_42
.LBB0_47:
	global_load_dwordx2 v[8:9], v[12:13], off
	s_and_saveexec_b64 s[22:23], s[0:1]
	s_cbranch_execz .LBB0_50
	v_mov_b32_e32 v11, 0
	global_load_dwordx2 v[12:13], v11, s[2:3] offset:40
	global_load_dwordx2 v[20:21], v11, s[2:3] offset:24 sc0 sc1
	global_load_dwordx2 v[14:15], v11, s[2:3]
	s_waitcnt vmcnt(2)
	v_readfirstlane_b32 s24, v12
	v_readfirstlane_b32 s25, v13
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
	v_lshl_add_u64 v[16:17], v[14:15], 0, s[20:21]
	v_mov_b32_e32 v18, s0
	global_store_dwordx2 v[16:17], v[20:21], off
	v_mov_b32_e32 v19, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[14:15], v11, v[18:21], s[2:3] offset:24 sc0 sc1
	s_mov_b64 s[20:21], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[14:15], v[20:21]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_50
.LBB0_49:
	s_sleep 1
	global_store_dwordx2 v[16:17], v[14:15], off
	v_mov_b32_e32 v12, s0
	v_mov_b32_e32 v13, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[12:13], v11, v[12:15], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[12:13], v[14:15]
	s_or_b64 s[20:21], vcc, s[20:21]
	v_mov_b64_e32 v[14:15], v[12:13]
	s_andn2_b64 exec, exec, s[20:21]
	s_cbranch_execnz .LBB0_49
.LBB0_50:
	s_or_b64 exec, exec, s[22:23]
	v_readfirstlane_b32 s0, v22
	v_mov_b64_e32 v[16:17], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v22
	s_and_saveexec_b64 s[20:21], s[0:1]
	s_cbranch_execz .LBB0_56
	v_mov_b32_e32 v11, 0
	global_load_dwordx2 v[14:15], v11, s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[12:13], v11, s[2:3] offset:40
	global_load_dwordx2 v[16:17], v11, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v12, v12, v14
	v_and_b32_e32 v13, v13, v15
	v_mul_lo_u32 v13, v13, 24
	v_mul_hi_u32 v18, v12, 24
	v_add_u32_e32 v13, v18, v13
	v_mul_lo_u32 v12, v12, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[12:13], v[16:17], 0, v[12:13]
	global_load_dwordx2 v[12:13], v[12:13], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[16:17], v11, v[12:15], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[16:17], v[14:15]
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_55
	s_mov_b64 s[24:25], 0
.LBB0_53:
	s_sleep 1
	global_load_dwordx2 v[12:13], v11, s[2:3] offset:40
	global_load_dwordx2 v[14:15], v11, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v12, v12, v16
	v_and_b32_e32 v18, v13, v17
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[12:13], s[26:27], v12, 24, v[14:15]
	v_mov_b32_e32 v14, v13
	v_mad_u64_u32 v[14:15], s[26:27], v18, 24, v[14:15]
	v_mov_b32_e32 v13, v14
	global_load_dwordx2 v[14:15], v[12:13], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[12:13], v11, v[14:17], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[12:13], v[16:17]
	s_or_b64 s[24:25], vcc, s[24:25]
	v_mov_b64_e32 v[16:17], v[12:13]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBB0_53
	s_or_b64 exec, exec, s[24:25]
	v_mov_b64_e32 v[16:17], v[12:13]
.LBB0_55:
	s_or_b64 exec, exec, s[22:23]
.LBB0_56:
	s_or_b64 exec, exec, s[20:21]
	v_mov_b32_e32 v11, 0
	global_load_dwordx2 v[18:19], v11, s[2:3] offset:40
	global_load_dwordx4 v[12:15], v11, s[2:3]
	v_readfirstlane_b32 s21, v17
	v_readfirstlane_b32 s20, v16
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
	v_lshl_add_u64 v[16:17], v[12:13], 0, s[26:27]
	s_and_saveexec_b64 s[26:27], s[0:1]
	s_cbranch_execz .LBB0_58
	v_mov_b64_e32 v[18:19], s[22:23]
	v_mov_b32_e32 v20, 2
	v_mov_b32_e32 v21, 1
	global_store_dwordx4 v[16:17], v[18:21], off offset:8
.LBB0_58:
	s_or_b64 exec, exec, s[26:27]
	s_lshl_b64 s[22:23], s[24:25], 12
	v_lshl_add_u64 v[14:15], v[14:15], 0, s[22:23]
	s_movk_i32 s22, 0xff1d
	s_mov_b32 s24, 0
	v_and_or_b32 v8, v8, s22, 34
	v_readfirstlane_b32 s22, v14
	v_readfirstlane_b32 s23, v15
	s_mov_b32 s25, s24
	s_mov_b32 s26, s24
	s_mov_b32 s27, s24
	s_nop 1
	global_store_dwordx4 v24, v[8:11], s[22:23]
	s_nop 1
	v_mov_b64_e32 v[8:9], s[24:25]
	v_mov_b64_e32 v[10:11], s[26:27]
	global_store_dwordx4 v24, v[8:11], s[22:23] offset:16
	global_store_dwordx4 v24, v[8:11], s[22:23] offset:32
	global_store_dwordx4 v24, v[8:11], s[22:23] offset:48
	s_and_saveexec_b64 s[22:23], s[0:1]
	s_cbranch_execz .LBB0_66
	v_mov_b32_e32 v14, 0
	global_load_dwordx2 v[20:21], v14, s[2:3] offset:32 sc0 sc1
	global_load_dwordx2 v[8:9], v14, s[2:3] offset:40
	v_mov_b32_e32 v18, s20
	v_mov_b32_e32 v19, s21
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s24, v8
	v_readfirstlane_b32 s25, v9
	s_and_b64 s[24:25], s[24:25], s[20:21]
	s_mul_i32 s25, s25, 24
	s_mul_hi_u32 s26, s24, 24
	s_mul_i32 s24, s24, 24
	s_add_i32 s25, s26, s25
	v_lshl_add_u64 v[12:13], v[12:13], 0, s[24:25]
	global_store_dwordx2 v[12:13], v[20:21], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v14, v[18:21], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[10:11], v[20:21]
	s_and_saveexec_b64 s[24:25], vcc
	s_cbranch_execz .LBB0_62
	s_mov_b64 s[26:27], 0
.LBB0_61:
	s_sleep 1
	global_store_dwordx2 v[12:13], v[10:11], off
	v_mov_b32_e32 v8, s20
	v_mov_b32_e32 v9, s21
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[8:9], v14, v[8:11], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[8:9], v[10:11]
	s_or_b64 s[26:27], vcc, s[26:27]
	v_mov_b64_e32 v[10:11], v[8:9]
	s_andn2_b64 exec, exec, s[26:27]
	s_cbranch_execnz .LBB0_61
.LBB0_62:
	s_or_b64 exec, exec, s[24:25]
	v_mov_b32_e32 v11, 0
	global_load_dwordx2 v[8:9], v11, s[2:3] offset:16
	s_mov_b64 s[24:25], exec
	v_mbcnt_lo_u32_b32 v10, s24, 0
	v_mbcnt_hi_u32_b32 v10, s25, v10
	v_cmp_eq_u32_e32 vcc, 0, v10
	s_and_saveexec_b64 s[26:27], vcc
	s_cbranch_execz .LBB0_64
	s_bcnt1_i32_b64 s24, s[24:25]
	v_mov_b32_e32 v10, s24
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[8:9], v[10:11], off offset:8 sc1
.LBB0_64:
	s_or_b64 exec, exec, s[26:27]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[10:11], v[8:9], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[10:11]
	s_cbranch_vccnz .LBB0_66
	global_load_dword v8, v[8:9], off offset:24
	v_mov_b32_e32 v9, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s24, v8
	s_and_b32 m0, s24, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[10:11], v[8:9], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_66:
	s_or_b64 exec, exec, s[22:23]
	s_branch .LBB0_70
.LBB0_67:
	s_or_b64 exec, exec, s[22:23]
	v_readfirstlane_b32 s22, v8
	s_cmp_eq_u32 s22, 0
	s_cbranch_scc1 .LBB0_69
	s_sleep 1
	s_cbranch_execnz .LBB0_70
	s_branch .LBB0_72
.LBB0_69:
	s_branch .LBB0_72
.LBB0_70:
	v_mov_b32_e32 v8, 1
	s_and_saveexec_b64 s[22:23], s[0:1]
	s_cbranch_execz .LBB0_67
	global_load_dword v8, v[16:17], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v8, 1, v8
	s_branch .LBB0_67
.LBB0_72:
	s_and_saveexec_b64 s[22:23], s[0:1]
	s_cbranch_execz .LBB0_75
	v_mov_b32_e32 v14, 0
	global_load_dwordx2 v[8:9], v14, s[2:3] offset:40
	global_load_dwordx2 v[18:19], v14, s[2:3] offset:24 sc0 sc1
	global_load_dwordx2 v[10:11], v14, s[2:3]
	s_waitcnt vmcnt(2)
	v_readfirstlane_b32 s24, v8
	v_readfirstlane_b32 s25, v9
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
	v_lshl_add_u64 v[12:13], v[10:11], 0, s[20:21]
	v_mov_b32_e32 v16, s0
	global_store_dwordx2 v[12:13], v[18:19], off
	v_mov_b32_e32 v17, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v14, v[16:19], s[2:3] offset:24 sc0 sc1
	s_mov_b64 s[20:21], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[10:11], v[18:19]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_75
.LBB0_74:
	s_sleep 1
	global_store_dwordx2 v[12:13], v[10:11], off
	v_mov_b32_e32 v8, s0
	v_mov_b32_e32 v9, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[8:9], v14, v[8:11], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[8:9], v[10:11]
	s_or_b64 s[20:21], vcc, s[20:21]
	v_mov_b64_e32 v[10:11], v[8:9]
	s_andn2_b64 exec, exec, s[20:21]
	s_cbranch_execnz .LBB0_74
.LBB0_75:
	s_or_b64 exec, exec, s[22:23]
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	.LBB0_2:
	;;#ASMEND
	;;#ASMSTART
		s_or_b64 exec, exec, s[0:1]
	;;#ASMEND
	;;#ASMSTART
		v_lshlrev_b32_e32 v1, 2, v0
	;;#ASMEND
	;;#ASMSTART
		s_cmp_lt_u32 s3, 2
	;;#ASMEND
	;;#ASMSTART
		s_waitcnt vmcnt(0)
	;;#ASMEND
	;;#ASMSTART
		ds_write_b32 v1, v3
	;;#ASMEND
	;;#ASMSTART
		s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
		s_barrier
	;;#ASMEND
	;;#ASMSTART
		s_cbranch_scc0 .LBB0_7
	;;#ASMEND
	;;#ASMSTART
	.LBB0_3:
	;;#ASMEND
	;;#ASMSTART
		s_mov_b32 s3, 0
	;;#ASMEND
	;;#ASMSTART
		v_cmp_eq_u32_e32 vcc, 0, v0
	;;#ASMEND
	;;#ASMSTART
		s_and_saveexec_b64 s[0:1], vcc
	;;#ASMEND
	;;#ASMSTART
		s_cbranch_execz .LBB0_5
	;;#ASMEND
	;;#ASMSTART
		v_mov_b32_e32 v0, 0
	;;#ASMEND
	;;#ASMSTART
		ds_read_b32 v1, v0                          ; Read reduction result from LDS
	;;#ASMEND
	;;#ASMSTART
		s_lshl_b64 s[0:1], s[2:3], 2
	;;#ASMEND
	;;#ASMSTART
		s_add_u32 s0, s6, s0
	;;#ASMEND
	;;#ASMSTART
		s_addc_u32 s1, s7, s1
	;;#ASMEND
	;;#ASMSTART
		s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v6, v1
	;;#ASMEND
	;;#ASMSTART
		global_store_dword v0, v1, s[0:1]
	;;#ASMEND
	;;#ASMSTART
	.LBB0_5:
	;;#ASMEND
	;;#ASMSTART
		s_endpgm
	;;#ASMEND
	;;#ASMSTART
	.LBB0_6:                                ;   in Loop: Header=BB0_7 Depth=1:
	;;#ASMEND
	;;#ASMSTART
		s_or_b64 exec, exec, s[0:1]
	;;#ASMEND
	;;#ASMSTART
		s_cmp_lt_u32 s3, 4
	;;#ASMEND
	;;#ASMSTART
		s_mov_b32 s3, s4
	;;#ASMEND
	;;#ASMSTART
		s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
		s_barrier
	;;#ASMEND
	;;#ASMSTART
		s_cbranch_scc1 .LBB0_3
	;;#ASMEND
	;;#ASMSTART
	.LBB0_7:                                ; =>This Inner Loop Header: Depth=1:
	;;#ASMEND
	;;#ASMSTART
		s_lshr_b32 s4, s3, 1
	;;#ASMEND
	;;#ASMSTART
		v_cmp_gt_u32_e32 vcc, s4, v0
	;;#ASMEND
	;;#ASMSTART
		s_and_saveexec_b64 s[0:1], vcc
	;;#ASMEND
	;;#ASMSTART
		s_cbranch_execz .LBB0_6
	;;#ASMEND
	;;#ASMSTART
		v_lshl_add_u32 v2, s4, 2, v1
	;;#ASMEND
	;;#ASMSTART
		ds_read_b32 v2, v2
	;;#ASMEND
	;;#ASMSTART
		ds_read_b32 v3, v1
	;;#ASMEND
	;;#ASMSTART
		s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
		v_add_u32_e32 v2, v3, v2
	;;#ASMEND
	;;#ASMSTART
		ds_write_b32 v1, v2
	;;#ASMEND
	;;#ASMSTART
		s_branch .LBB0_6
	;;#ASMEND
	s_endpgm
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
		.amdhsa_next_free_vgpr 30
		.amdhsa_next_free_sgpr 28
		.amdhsa_accum_offset 32
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

	.set _Z15sharedMemKernelPKiPii.num_vgpr, 30
	.set _Z15sharedMemKernelPKiPii.num_agpr, 0
	.set _Z15sharedMemKernelPKiPii.numbered_sgpr, 28
	.set _Z15sharedMemKernelPKiPii.num_named_barrier, 0
	.set _Z15sharedMemKernelPKiPii.private_seg_size, 0
	.set _Z15sharedMemKernelPKiPii.uses_vcc, 1
	.set _Z15sharedMemKernelPKiPii.uses_flat_scratch, 0
	.set _Z15sharedMemKernelPKiPii.has_dyn_sized_stack, 0
	.set _Z15sharedMemKernelPKiPii.has_recursion, 0
	.set _Z15sharedMemKernelPKiPii.has_indirect_call, 0
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
  - .offset: 16
    .size: 4
    .value_kind: by_value
  - .offset: 80
    .size: 4
    .value_kind: hidden_block_count_x
  - .offset: 84
    .size: 4
    .value_kind: hidden_block_count_y
  - .offset: 88
    .size: 4
    .value_kind: hidden_block_count_z
  - .offset: 92
    .size: 2
    .value_kind: hidden_group_size_x
  - .offset: 94
    .size: 2
    .value_kind: hidden_group_size_y
  - .offset: 96
    .size: 2
    .value_kind: hidden_group_size_z
  - .offset: 98
    .size: 2
    .value_kind: hidden_remainder_x
  - .offset: 100
    .size: 2
    .value_kind: hidden_remainder_y
  - .offset: 102
    .size: 2
    .value_kind: hidden_remainder_z
  - .offset: 120
    .size: 8
    .value_kind: hidden_global_offset_x
  - .offset: 128
    .size: 8
    .value_kind: hidden_global_offset_y
  - .offset: 136
    .size: 8
    .value_kind: hidden_global_offset_z
  - .offset: 144
    .size: 2
    .value_kind: hidden_grid_dims
  - .offset: 160
    .size: 8
    .value_kind: hidden_hostcall_buffer
  .group_segment_fixed_size: 1024
  .kernarg_segment_align: 8
  .kernarg_segment_size: 280
  .max_flat_workgroup_size: 256
  .name: _Z15sharedMemKernelPKiPii
  .private_segment_fixed_size: 0
  .sgpr_count: 34
  .sgpr_spill_count: 0
  .symbol: _Z15sharedMemKernelPKiPii.kd
  .uniform_work_group_size: 1
  .uses_dynamic_stack: false
  .vgpr_count: 30
  .vgpr_spill_count: 0
  .wavefront_size: 64
amdhsa.target: amdgcn-amd-amdhsa--gfx950
amdhsa.version:
- 1
- 2
...
...

	.end_amdgpu_metadata
