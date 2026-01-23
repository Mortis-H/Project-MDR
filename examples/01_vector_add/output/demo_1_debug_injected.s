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
	v_mov_b32 v62, v0
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v63, v0
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
		global_load_dword v6, v[4:5], off
	;;#ASMEND
	;;#ASMSTART
		global_load_dword v7, v[2:3], off
	;;#ASMEND
	;;#ASMSTART
		v_lshl_add_u64 v[0:1], s[2:3], 0, v[0:1]
	;;#ASMEND
	;;#ASMSTART
		s_waitcnt vmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v56, v6
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v57, v7
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v58, v2
	;;#ASMEND
	;;#ASMSTART
		v_add_f32_e32 v2, v6, v7
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v59, v6
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v60, v7
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v61, v2
	;;#ASMEND
	;;#ASMSTART
		global_store_dword v[0:1], v2, off
	;;#ASMEND
	;;#ASMSTART
	s_mov_b64 s[0:1], s[18:19]
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v2, v63
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v18, v56
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v19, v57
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v20, v58
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v0, v63
	;;#ASMEND
	v_mbcnt_lo_u32_b32 v15, -1, 0
	v_cmp_gt_i32_e32 vcc, 4, v0
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_76
	s_load_dwordx2 s[6:7], s[0:1], 0x70
	v_mbcnt_hi_u32_b32 v14, -1, v15
	v_mov_b64_e32 v[8:9], 0
	v_readfirstlane_b32 s2, v14
	s_nop 1
	v_cmp_eq_u32_e64 s[2:3], s2, v14
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_7
	v_mov_b32_e32 v3, 0
	s_waitcnt lgkmcnt(0)
	global_load_dwordx2 v[6:7], v3, s[6:7] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[0:1], v3, s[6:7] offset:40
	global_load_dwordx2 v[4:5], v3, s[6:7]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v0, v0, v6
	v_and_b32_e32 v1, v1, v7
	v_mul_lo_u32 v1, v1, 24
	v_mul_hi_u32 v8, v0, 24
	v_add_u32_e32 v1, v8, v1
	v_mul_lo_u32 v0, v0, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[0:1], v[4:5], 0, v[0:1]
	global_load_dwordx2 v[4:5], v[0:1], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[8:9], v3, v[4:7], s[6:7] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[8:9], v[6:7]
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_6
	s_mov_b64 s[12:13], 0
.LBB0_4:
	s_sleep 1
	global_load_dwordx2 v[0:1], v3, s[6:7] offset:40
	global_load_dwordx2 v[4:5], v3, s[6:7]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v0, v0, v8
	v_and_b32_e32 v6, v1, v9
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[0:1], s[14:15], v0, 24, v[4:5]
	v_mov_b32_e32 v4, v1
	v_mad_u64_u32 v[4:5], s[14:15], v6, 24, v[4:5]
	v_mov_b32_e32 v1, v4
	global_load_dwordx2 v[6:7], v[0:1], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[0:1], v3, v[6:9], s[6:7] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[0:1], v[8:9]
	s_or_b64 s[12:13], vcc, s[12:13]
	v_mov_b64_e32 v[8:9], v[0:1]
	s_andn2_b64 exec, exec, s[12:13]
	s_cbranch_execnz .LBB0_4
	s_or_b64 exec, exec, s[12:13]
	v_mov_b64_e32 v[8:9], v[0:1]
.LBB0_6:
	s_or_b64 exec, exec, s[10:11]
.LBB0_7:
	s_or_b64 exec, exec, s[8:9]
	v_mov_b32_e32 v11, 0
	s_waitcnt lgkmcnt(0)
	global_load_dwordx2 v[0:1], v11, s[6:7] offset:40
	global_load_dwordx4 v[4:7], v11, s[6:7]
	v_readfirstlane_b32 s9, v9
	v_readfirstlane_b32 s8, v8
	s_mov_b64 s[10:11], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s12, v0
	v_readfirstlane_b32 s13, v1
	s_and_b64 s[12:13], s[12:13], s[8:9]
	s_mul_i32 s14, s13, 24
	s_mul_hi_u32 s15, s12, 24
	s_add_i32 s15, s15, s14
	s_mul_i32 s14, s12, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[0:1], v[4:5], 0, s[14:15]
	s_and_saveexec_b64 s[14:15], s[2:3]
	s_cbranch_execz .LBB0_9
	v_mov_b64_e32 v[22:23], s[10:11]
	v_mov_b32_e32 v24, 2
	v_mov_b32_e32 v25, 1
	global_store_dwordx4 v[0:1], v[22:25], off offset:8
.LBB0_9:
	s_or_b64 exec, exec, s[14:15]
	s_lshl_b64 s[10:11], s[12:13], 12
	v_lshl_add_u64 v[8:9], v[6:7], 0, s[10:11]
	s_mov_b32 s12, 0
	v_lshlrev_b32_e32 v10, 6, v14
	v_mov_b32_e32 v22, 33
	v_mov_b32_e32 v23, v11
	v_mov_b32_e32 v24, v11
	v_mov_b32_e32 v25, v11
	v_readfirstlane_b32 s10, v8
	v_readfirstlane_b32 s11, v9
	s_mov_b32 s14, s12
	s_mov_b32 s15, s12
	s_mov_b32 s13, s12
	s_nop 1
	global_store_dwordx4 v10, v[22:25], s[10:11]
	s_nop 1
	v_mov_b64_e32 v[24:25], s[14:15]
	v_mov_b64_e32 v[22:23], s[12:13]
	global_store_dwordx4 v10, v[22:25], s[10:11] offset:16
	global_store_dwordx4 v10, v[22:25], s[10:11] offset:32
	global_store_dwordx4 v10, v[22:25], s[10:11] offset:48
	s_and_saveexec_b64 s[10:11], s[2:3]
	s_cbranch_execz .LBB0_17
	v_mov_b32_e32 v3, 0
	global_load_dwordx2 v[24:25], v3, s[6:7] offset:32 sc0 sc1
	global_load_dwordx2 v[6:7], v3, s[6:7] offset:40
	v_mov_b32_e32 v22, s8
	v_mov_b32_e32 v23, s9
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v6, s8, v6
	v_and_b32_e32 v7, s9, v7
	v_mul_lo_u32 v7, v7, 24
	v_mul_hi_u32 v12, v6, 24
	v_mul_lo_u32 v6, v6, 24
	v_add_u32_e32 v7, v12, v7
	v_lshl_add_u64 v[12:13], v[4:5], 0, v[6:7]
	global_store_dwordx2 v[12:13], v[24:25], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[6:7], v3, v[22:25], s[6:7] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[6:7], v[24:25]
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB0_13
	s_mov_b64 s[14:15], 0
.LBB0_12:
	s_sleep 1
	global_store_dwordx2 v[12:13], v[6:7], off
	v_mov_b32_e32 v4, s8
	v_mov_b32_e32 v5, s9
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[4:5], v3, v[4:7], s[6:7] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[4:5], v[6:7]
	s_or_b64 s[14:15], vcc, s[14:15]
	v_mov_b64_e32 v[6:7], v[4:5]
	s_andn2_b64 exec, exec, s[14:15]
	s_cbranch_execnz .LBB0_12
.LBB0_13:
	s_or_b64 exec, exec, s[12:13]
	v_mov_b32_e32 v7, 0
	global_load_dwordx2 v[4:5], v7, s[6:7] offset:16
	s_mov_b64 s[12:13], exec
	v_mbcnt_lo_u32_b32 v3, s12, 0
	v_mbcnt_hi_u32_b32 v3, s13, v3
	v_cmp_eq_u32_e32 vcc, 0, v3
	s_and_saveexec_b64 s[14:15], vcc
	s_cbranch_execz .LBB0_15
	s_bcnt1_i32_b64 s12, s[12:13]
	v_mov_b32_e32 v6, s12
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[4:5], v[6:7], off offset:8 sc1
.LBB0_15:
	s_or_b64 exec, exec, s[14:15]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[6:7], v[4:5], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[6:7]
	s_cbranch_vccnz .LBB0_17
	global_load_dword v4, v[4:5], off offset:24
	v_mov_b32_e32 v5, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s12, v4
	s_and_b32 m0, s12, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[6:7], v[4:5], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_17:
	s_or_b64 exec, exec, s[10:11]
	v_lshl_add_u64 v[4:5], v[8:9], 0, v[10:11]
	s_branch .LBB0_21
.LBB0_18:
	s_or_b64 exec, exec, s[10:11]
	v_readfirstlane_b32 s10, v3
	s_cmp_eq_u32 s10, 0
	s_cbranch_scc1 .LBB0_20
	s_sleep 1
	s_cbranch_execnz .LBB0_21
	s_branch .LBB0_23
.LBB0_20:
	s_branch .LBB0_23
.LBB0_21:
	v_mov_b32_e32 v3, 1
	s_and_saveexec_b64 s[10:11], s[2:3]
	s_cbranch_execz .LBB0_18
	global_load_dword v3, v[0:1], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v3, 1, v3
	s_branch .LBB0_18
.LBB0_23:
	global_load_dwordx2 v[4:5], v[4:5], off
	s_and_saveexec_b64 s[10:11], s[2:3]
	s_cbranch_execz .LBB0_26
	v_mov_b32_e32 v3, 0
	global_load_dwordx2 v[0:1], v3, s[6:7] offset:40
	global_load_dwordx2 v[10:11], v3, s[6:7] offset:24 sc0 sc1
	global_load_dwordx2 v[6:7], v3, s[6:7]
	s_waitcnt vmcnt(2)
	v_readfirstlane_b32 s12, v0
	v_readfirstlane_b32 s13, v1
	s_add_u32 s14, s12, 1
	s_addc_u32 s15, s13, 0
	s_add_u32 s2, s14, s8
	s_addc_u32 s3, s15, s9
	s_cmp_eq_u64 s[2:3], 0
	s_cselect_b32 s3, s15, s3
	s_cselect_b32 s2, s14, s2
	s_and_b64 s[8:9], s[2:3], s[12:13]
	s_mul_i32 s9, s9, 24
	s_mul_hi_u32 s12, s8, 24
	s_mul_i32 s8, s8, 24
	s_add_i32 s9, s12, s9
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[0:1], v[6:7], 0, s[8:9]
	v_mov_b32_e32 v8, s2
	global_store_dwordx2 v[0:1], v[10:11], off
	v_mov_b32_e32 v9, s3
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[8:9], v3, v[8:11], s[6:7] offset:24 sc0 sc1
	s_mov_b64 s[8:9], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[8:9], v[10:11]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_26
.LBB0_25:
	s_sleep 1
	global_store_dwordx2 v[0:1], v[8:9], off
	v_mov_b32_e32 v6, s2
	v_mov_b32_e32 v7, s3
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[6:7], v3, v[6:9], s[6:7] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[6:7], v[8:9]
	s_or_b64 s[8:9], vcc, s[8:9]
	v_mov_b64_e32 v[8:9], v[6:7]
	s_andn2_b64 exec, exec, s[8:9]
	s_cbranch_execnz .LBB0_25
.LBB0_26:
	s_or_b64 exec, exec, s[10:11]
	v_readfirstlane_b32 s2, v14
	v_mov_b64_e32 v[12:13], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[2:3], s2, v14
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_32
	v_mov_b32_e32 v3, 0
	global_load_dwordx2 v[8:9], v3, s[6:7] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[0:1], v3, s[6:7] offset:40
	global_load_dwordx2 v[6:7], v3, s[6:7]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v0, v0, v8
	v_and_b32_e32 v1, v1, v9
	v_mul_lo_u32 v1, v1, 24
	v_mul_hi_u32 v10, v0, 24
	v_add_u32_e32 v1, v10, v1
	v_mul_lo_u32 v0, v0, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[0:1], v[6:7], 0, v[0:1]
	global_load_dwordx2 v[6:7], v[0:1], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[12:13], v3, v[6:9], s[6:7] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[12:13], v[8:9]
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_31
	s_mov_b64 s[12:13], 0
.LBB0_29:
	s_sleep 1
	global_load_dwordx2 v[0:1], v3, s[6:7] offset:40
	global_load_dwordx2 v[6:7], v3, s[6:7]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v0, v0, v12
	v_and_b32_e32 v8, v1, v13
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[0:1], s[14:15], v0, 24, v[6:7]
	v_mov_b32_e32 v6, v1
	v_mad_u64_u32 v[6:7], s[14:15], v8, 24, v[6:7]
	v_mov_b32_e32 v1, v6
	global_load_dwordx2 v[10:11], v[0:1], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[0:1], v3, v[10:13], s[6:7] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[0:1], v[12:13]
	s_or_b64 s[12:13], vcc, s[12:13]
	v_mov_b64_e32 v[12:13], v[0:1]
	s_andn2_b64 exec, exec, s[12:13]
	s_cbranch_execnz .LBB0_29
	s_or_b64 exec, exec, s[12:13]
	v_mov_b64_e32 v[12:13], v[0:1]
.LBB0_31:
	s_or_b64 exec, exec, s[10:11]
.LBB0_32:
	s_or_b64 exec, exec, s[8:9]
	v_mov_b32_e32 v17, 0
	global_load_dwordx2 v[0:1], v17, s[6:7] offset:40
	global_load_dwordx4 v[8:11], v17, s[6:7]
	v_readfirstlane_b32 s9, v13
	v_readfirstlane_b32 s8, v12
	s_mov_b64 s[10:11], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s12, v0
	v_readfirstlane_b32 s13, v1
	s_and_b64 s[12:13], s[12:13], s[8:9]
	s_mul_i32 s14, s13, 24
	s_mul_hi_u32 s15, s12, 24
	s_add_i32 s15, s15, s14
	s_mul_i32 s14, s12, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[0:1], v[8:9], 0, s[14:15]
	s_and_saveexec_b64 s[14:15], s[2:3]
	s_cbranch_execz .LBB0_34
	v_mov_b64_e32 v[22:23], s[10:11]
	v_mov_b32_e32 v24, 2
	v_mov_b32_e32 v25, 1
	global_store_dwordx4 v[0:1], v[22:25], off offset:8
.LBB0_34:
	s_or_b64 exec, exec, s[14:15]
	s_lshl_b64 s[10:11], s[12:13], 12
	v_lshl_add_u64 v[10:11], v[10:11], 0, s[10:11]
	v_and_b32_e32 v3, 0xffffff1f, v4
	v_or_b32_e32 v4, 0xc0, v3
	v_lshlrev_b32_e32 v16, 6, v14
	v_mov_b32_e32 v7, 0x2065726f
	v_mov_b32_e32 v6, 0x6665425b
	v_readfirstlane_b32 s10, v10
	v_readfirstlane_b32 s11, v11
	s_nop 4
	global_store_dwordx4 v16, v[4:7], s[10:11]
	s_nop 1
	v_mov_b32_e32 v4, 0x5d444441
	v_mov_b32_e32 v5, 0x64697420
	v_mov_b32_e32 v6, 0x3a64253d
	v_mov_b32_e32 v7, 0x253d4120
	global_store_dwordx4 v16, v[4:7], s[10:11] offset:16
	s_nop 1
	v_mov_b32_e32 v4, 0x2c66322e
	v_mov_b32_e32 v5, 0x253d4220
	v_mov_b32_e32 v7, 0x253d4320
	v_mov_b32_e32 v6, v4
	global_store_dwordx4 v16, v[4:7], s[10:11] offset:32
	s_nop 1
	v_mov_b32_e32 v4, 0xa66322e
	v_mov_b32_e32 v5, v17
	v_mov_b32_e32 v6, v17
	v_mov_b32_e32 v7, v17
	global_store_dwordx4 v16, v[4:7], s[10:11] offset:48
	s_and_saveexec_b64 s[10:11], s[2:3]
	s_cbranch_execz .LBB0_42
	v_mov_b32_e32 v3, 0
	global_load_dwordx2 v[24:25], v3, s[6:7] offset:32 sc0 sc1
	global_load_dwordx2 v[4:5], v3, s[6:7] offset:40
	v_mov_b32_e32 v22, s8
	v_mov_b32_e32 v23, s9
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s12, v4
	v_readfirstlane_b32 s13, v5
	s_and_b64 s[12:13], s[12:13], s[8:9]
	s_mul_i32 s13, s13, 24
	s_mul_hi_u32 s14, s12, 24
	s_mul_i32 s12, s12, 24
	s_add_i32 s13, s14, s13
	v_lshl_add_u64 v[8:9], v[8:9], 0, s[12:13]
	global_store_dwordx2 v[8:9], v[24:25], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[6:7], v3, v[22:25], s[6:7] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[6:7], v[24:25]
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB0_38
	s_mov_b64 s[14:15], 0
.LBB0_37:
	s_sleep 1
	global_store_dwordx2 v[8:9], v[6:7], off
	v_mov_b32_e32 v4, s8
	v_mov_b32_e32 v5, s9
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[4:5], v3, v[4:7], s[6:7] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[4:5], v[6:7]
	s_or_b64 s[14:15], vcc, s[14:15]
	v_mov_b64_e32 v[6:7], v[4:5]
	s_andn2_b64 exec, exec, s[14:15]
	s_cbranch_execnz .LBB0_37
.LBB0_38:
	s_or_b64 exec, exec, s[12:13]
	v_mov_b32_e32 v7, 0
	global_load_dwordx2 v[4:5], v7, s[6:7] offset:16
	s_mov_b64 s[12:13], exec
	v_mbcnt_lo_u32_b32 v3, s12, 0
	v_mbcnt_hi_u32_b32 v3, s13, v3
	v_cmp_eq_u32_e32 vcc, 0, v3
	s_and_saveexec_b64 s[14:15], vcc
	s_cbranch_execz .LBB0_40
	s_bcnt1_i32_b64 s12, s[12:13]
	v_mov_b32_e32 v6, s12
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[4:5], v[6:7], off offset:8 sc1
.LBB0_40:
	s_or_b64 exec, exec, s[14:15]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[6:7], v[4:5], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[6:7]
	s_cbranch_vccnz .LBB0_42
	global_load_dword v4, v[4:5], off offset:24
	v_mov_b32_e32 v5, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s12, v4
	s_and_b32 m0, s12, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[6:7], v[4:5], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_42:
	s_or_b64 exec, exec, s[10:11]
	v_lshl_add_u64 v[4:5], v[10:11], 0, v[16:17]
	s_branch .LBB0_46
.LBB0_43:
	s_or_b64 exec, exec, s[10:11]
	v_readfirstlane_b32 s10, v3
	s_cmp_eq_u32 s10, 0
	s_cbranch_scc1 .LBB0_45
	s_sleep 1
	s_cbranch_execnz .LBB0_46
	s_branch .LBB0_48
.LBB0_45:
	s_branch .LBB0_48
.LBB0_46:
	v_mov_b32_e32 v3, 1
	s_and_saveexec_b64 s[10:11], s[2:3]
	s_cbranch_execz .LBB0_43
	global_load_dword v3, v[0:1], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v3, 1, v3
	s_branch .LBB0_43
.LBB0_48:
	global_load_dwordx2 v[0:1], v[4:5], off
	s_and_saveexec_b64 s[10:11], s[2:3]
	s_cbranch_execz .LBB0_51
	v_mov_b32_e32 v3, 0
	global_load_dwordx2 v[4:5], v3, s[6:7] offset:40
	global_load_dwordx2 v[12:13], v3, s[6:7] offset:24 sc0 sc1
	global_load_dwordx2 v[6:7], v3, s[6:7]
	s_waitcnt vmcnt(2)
	v_readfirstlane_b32 s12, v4
	v_readfirstlane_b32 s13, v5
	s_add_u32 s14, s12, 1
	s_addc_u32 s15, s13, 0
	s_add_u32 s2, s14, s8
	s_addc_u32 s3, s15, s9
	s_cmp_eq_u64 s[2:3], 0
	s_cselect_b32 s3, s15, s3
	s_cselect_b32 s2, s14, s2
	s_and_b64 s[8:9], s[2:3], s[12:13]
	s_mul_i32 s9, s9, 24
	s_mul_hi_u32 s12, s8, 24
	s_mul_i32 s8, s8, 24
	s_add_i32 s9, s12, s9
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[8:9], v[6:7], 0, s[8:9]
	v_mov_b32_e32 v10, s2
	global_store_dwordx2 v[8:9], v[12:13], off
	v_mov_b32_e32 v11, s3
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[6:7], v3, v[10:13], s[6:7] offset:24 sc0 sc1
	s_mov_b64 s[8:9], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[6:7], v[12:13]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_51
.LBB0_50:
	s_sleep 1
	global_store_dwordx2 v[8:9], v[6:7], off
	v_mov_b32_e32 v4, s2
	v_mov_b32_e32 v5, s3
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[4:5], v3, v[4:7], s[6:7] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[4:5], v[6:7]
	s_or_b64 s[8:9], vcc, s[8:9]
	v_mov_b64_e32 v[6:7], v[4:5]
	s_andn2_b64 exec, exec, s[8:9]
	s_cbranch_execnz .LBB0_50
.LBB0_51:
	s_or_b64 exec, exec, s[10:11]
	v_readfirstlane_b32 s2, v14
	v_mov_b64_e32 v[10:11], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[2:3], s2, v14
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_57
	v_mov_b32_e32 v3, 0
	global_load_dwordx2 v[6:7], v3, s[6:7] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[4:5], v3, s[6:7] offset:40
	global_load_dwordx2 v[8:9], v3, s[6:7]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v4, v4, v6
	v_and_b32_e32 v5, v5, v7
	v_mul_lo_u32 v5, v5, 24
	v_mul_hi_u32 v10, v4, 24
	v_add_u32_e32 v5, v10, v5
	v_mul_lo_u32 v4, v4, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[4:5], v[8:9], 0, v[4:5]
	global_load_dwordx2 v[4:5], v[4:5], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v3, v[4:7], s[6:7] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[10:11], v[6:7]
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_56
	s_mov_b64 s[12:13], 0
.LBB0_54:
	s_sleep 1
	global_load_dwordx2 v[4:5], v3, s[6:7] offset:40
	global_load_dwordx2 v[6:7], v3, s[6:7]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v4, v4, v10
	v_and_b32_e32 v8, v5, v11
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[4:5], s[14:15], v4, 24, v[6:7]
	v_mov_b32_e32 v6, v5
	v_mad_u64_u32 v[6:7], s[14:15], v8, 24, v[6:7]
	v_mov_b32_e32 v5, v6
	global_load_dwordx2 v[8:9], v[4:5], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[4:5], v3, v[8:11], s[6:7] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[4:5], v[10:11]
	s_or_b64 s[12:13], vcc, s[12:13]
	v_mov_b64_e32 v[10:11], v[4:5]
	s_andn2_b64 exec, exec, s[12:13]
	s_cbranch_execnz .LBB0_54
	s_or_b64 exec, exec, s[12:13]
	v_mov_b64_e32 v[10:11], v[4:5]
.LBB0_56:
	s_or_b64 exec, exec, s[10:11]
.LBB0_57:
	s_or_b64 exec, exec, s[8:9]
	v_mov_b32_e32 v3, 0
	global_load_dwordx2 v[4:5], v3, s[6:7] offset:40
	global_load_dwordx4 v[6:9], v3, s[6:7]
	v_readfirstlane_b32 s9, v11
	v_readfirstlane_b32 s8, v10
	s_mov_b64 s[10:11], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s12, v4
	v_readfirstlane_b32 s13, v5
	s_and_b64 s[12:13], s[12:13], s[8:9]
	s_mul_i32 s14, s13, 24
	s_mul_hi_u32 s15, s12, 24
	s_add_i32 s15, s15, s14
	s_mul_i32 s14, s12, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[10:11], v[6:7], 0, s[14:15]
	s_and_saveexec_b64 s[14:15], s[2:3]
	s_cbranch_execz .LBB0_59
	v_mov_b64_e32 v[22:23], s[10:11]
	v_mov_b32_e32 v24, 2
	v_mov_b32_e32 v25, 1
	global_store_dwordx4 v[10:11], v[22:25], off offset:8
.LBB0_59:
	s_or_b64 exec, exec, s[14:15]
	s_lshl_b64 s[10:11], s[12:13], 12
	v_lshl_add_u64 v[4:5], v[8:9], 0, s[10:11]
	v_and_b32_e32 v0, 0xffffff1d, v0
	v_or_b32_e32 v0, 0x82, v0
	v_readfirstlane_b32 s10, v4
	v_readfirstlane_b32 s11, v5
	s_mov_b32 s12, 0
	v_cvt_f64_f32_e32 v[22:23], v18
	v_cvt_f64_f32_e32 v[24:25], v19
	v_cvt_f64_f32_e32 v[18:19], v20
	s_nop 0
	global_store_dwordx4 v16, v[0:3], s[10:11]
	global_store_dwordx4 v16, v[22:25], s[10:11] offset:16
	v_mov_b32_e32 v20, s12
	v_mov_b32_e32 v21, s12
	v_mov_b32_e32 v2, v3
	v_mov_b32_e32 v4, v3
	v_mov_b32_e32 v5, v3
	global_store_dwordx4 v16, v[18:21], s[10:11] offset:32
	global_store_dwordx4 v16, v[2:5], s[10:11] offset:48
	s_and_saveexec_b64 s[10:11], s[2:3]
	s_cbranch_execz .LBB0_67
	v_mov_b32_e32 v8, 0
	global_load_dwordx2 v[18:19], v8, s[6:7] offset:32 sc0 sc1
	global_load_dwordx2 v[0:1], v8, s[6:7] offset:40
	v_mov_b32_e32 v16, s8
	v_mov_b32_e32 v17, s9
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s12, v0
	v_readfirstlane_b32 s13, v1
	s_and_b64 s[12:13], s[12:13], s[8:9]
	s_mul_i32 s13, s13, 24
	s_mul_hi_u32 s14, s12, 24
	s_mul_i32 s12, s12, 24
	s_add_i32 s13, s14, s13
	v_lshl_add_u64 v[4:5], v[6:7], 0, s[12:13]
	global_store_dwordx2 v[4:5], v[18:19], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v8, v[16:19], s[6:7] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[2:3], v[18:19]
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB0_63
	s_mov_b64 s[14:15], 0
.LBB0_62:
	s_sleep 1
	global_store_dwordx2 v[4:5], v[2:3], off
	v_mov_b32_e32 v0, s8
	v_mov_b32_e32 v1, s9
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[0:1], v8, v[0:3], s[6:7] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[0:1], v[2:3]
	s_or_b64 s[14:15], vcc, s[14:15]
	v_mov_b64_e32 v[2:3], v[0:1]
	s_andn2_b64 exec, exec, s[14:15]
	s_cbranch_execnz .LBB0_62
.LBB0_63:
	s_or_b64 exec, exec, s[12:13]
	v_mov_b32_e32 v3, 0
	global_load_dwordx2 v[0:1], v3, s[6:7] offset:16
	s_mov_b64 s[12:13], exec
	v_mbcnt_lo_u32_b32 v2, s12, 0
	v_mbcnt_hi_u32_b32 v2, s13, v2
	v_cmp_eq_u32_e32 vcc, 0, v2
	s_and_saveexec_b64 s[14:15], vcc
	s_cbranch_execz .LBB0_65
	s_bcnt1_i32_b64 s12, s[12:13]
	v_mov_b32_e32 v2, s12
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[0:1], v[2:3], off offset:8 sc1
.LBB0_65:
	s_or_b64 exec, exec, s[14:15]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[2:3], v[0:1], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[2:3]
	s_cbranch_vccnz .LBB0_67
	global_load_dword v0, v[0:1], off offset:24
	v_mov_b32_e32 v1, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s12, v0
	s_and_b32 m0, s12, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[2:3], v[0:1], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_67:
	s_or_b64 exec, exec, s[10:11]
	s_branch .LBB0_71
.LBB0_68:
	s_or_b64 exec, exec, s[10:11]
	v_readfirstlane_b32 s10, v0
	s_cmp_eq_u32 s10, 0
	s_cbranch_scc1 .LBB0_70
	s_sleep 1
	s_cbranch_execnz .LBB0_71
	s_branch .LBB0_73
.LBB0_70:
	s_branch .LBB0_73
.LBB0_71:
	v_mov_b32_e32 v0, 1
	s_and_saveexec_b64 s[10:11], s[2:3]
	s_cbranch_execz .LBB0_68
	global_load_dword v0, v[10:11], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v0, 1, v0
	s_branch .LBB0_68
.LBB0_73:
	s_and_b64 exec, exec, s[2:3]
	s_cbranch_execz .LBB0_76
	v_mov_b32_e32 v6, 0
	global_load_dwordx2 v[0:1], v6, s[6:7] offset:40
	global_load_dwordx2 v[10:11], v6, s[6:7] offset:24 sc0 sc1
	global_load_dwordx2 v[2:3], v6, s[6:7]
	s_waitcnt vmcnt(2)
	v_readfirstlane_b32 s10, v0
	v_readfirstlane_b32 s11, v1
	s_add_u32 s12, s10, 1
	s_addc_u32 s13, s11, 0
	s_add_u32 s2, s12, s8
	s_addc_u32 s3, s13, s9
	s_cmp_eq_u64 s[2:3], 0
	s_cselect_b32 s3, s13, s3
	s_cselect_b32 s2, s12, s2
	s_and_b64 s[8:9], s[2:3], s[10:11]
	s_mul_i32 s9, s9, 24
	s_mul_hi_u32 s10, s8, 24
	s_mul_i32 s8, s8, 24
	s_add_i32 s9, s10, s9
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[4:5], v[2:3], 0, s[8:9]
	v_mov_b32_e32 v8, s2
	global_store_dwordx2 v[4:5], v[10:11], off
	v_mov_b32_e32 v9, s3
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v6, v[8:11], s[6:7] offset:24 sc0 sc1
	s_mov_b64 s[8:9], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[2:3], v[10:11]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_76
.LBB0_75:
	s_sleep 1
	global_store_dwordx2 v[4:5], v[2:3], off
	v_mov_b32_e32 v0, s2
	v_mov_b32_e32 v1, s3
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[0:1], v6, v[0:3], s[6:7] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[0:1], v[2:3]
	s_or_b64 s[8:9], vcc, s[8:9]
	v_mov_b64_e32 v[2:3], v[0:1]
	s_andn2_b64 exec, exec, s[8:9]
	s_cbranch_execnz .LBB0_75
.LBB0_76:
	s_or_b64 exec, exec, s[4:5]
	;;#ASMSTART
	v_mov_b32 v2, v63
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v18, v59
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v19, v60
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v20, v61
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v0, v63
	;;#ASMEND
	s_nop 0
	v_cmp_gt_i32_e32 vcc, 4, v0
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_152
	s_load_dwordx2 s[4:5], s[0:1], 0x70
	v_mbcnt_hi_u32_b32 v14, -1, v15
	v_mov_b64_e32 v[8:9], 0
	v_readfirstlane_b32 s0, v14
	s_nop 1
	v_cmp_eq_u32_e64 s[0:1], s0, v14
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_83
	v_mov_b32_e32 v3, 0
	s_waitcnt lgkmcnt(0)
	global_load_dwordx2 v[6:7], v3, s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[0:1], v3, s[4:5] offset:40
	global_load_dwordx2 v[4:5], v3, s[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v0, v0, v6
	v_and_b32_e32 v1, v1, v7
	v_mul_lo_u32 v1, v1, 24
	v_mul_hi_u32 v8, v0, 24
	v_add_u32_e32 v1, v8, v1
	v_mul_lo_u32 v0, v0, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[0:1], v[4:5], 0, v[0:1]
	global_load_dwordx2 v[4:5], v[0:1], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[8:9], v3, v[4:7], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[8:9], v[6:7]
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_82
	s_mov_b64 s[10:11], 0
.LBB0_80:
	s_sleep 1
	global_load_dwordx2 v[0:1], v3, s[4:5] offset:40
	global_load_dwordx2 v[4:5], v3, s[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v0, v0, v8
	v_and_b32_e32 v6, v1, v9
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[0:1], s[12:13], v0, 24, v[4:5]
	v_mov_b32_e32 v4, v1
	v_mad_u64_u32 v[4:5], s[12:13], v6, 24, v[4:5]
	v_mov_b32_e32 v1, v4
	global_load_dwordx2 v[6:7], v[0:1], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[0:1], v3, v[6:9], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[0:1], v[8:9]
	s_or_b64 s[10:11], vcc, s[10:11]
	v_mov_b64_e32 v[8:9], v[0:1]
	s_andn2_b64 exec, exec, s[10:11]
	s_cbranch_execnz .LBB0_80
	s_or_b64 exec, exec, s[10:11]
	v_mov_b64_e32 v[8:9], v[0:1]
.LBB0_82:
	s_or_b64 exec, exec, s[8:9]
.LBB0_83:
	s_or_b64 exec, exec, s[6:7]
	v_mov_b32_e32 v11, 0
	s_waitcnt lgkmcnt(0)
	global_load_dwordx2 v[0:1], v11, s[4:5] offset:40
	global_load_dwordx4 v[4:7], v11, s[4:5]
	v_readfirstlane_b32 s7, v9
	v_readfirstlane_b32 s6, v8
	s_mov_b64 s[8:9], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s10, v0
	v_readfirstlane_b32 s11, v1
	s_and_b64 s[10:11], s[10:11], s[6:7]
	s_mul_i32 s12, s11, 24
	s_mul_hi_u32 s13, s10, 24
	s_add_i32 s13, s13, s12
	s_mul_i32 s12, s10, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[0:1], v[4:5], 0, s[12:13]
	s_and_saveexec_b64 s[12:13], s[0:1]
	s_cbranch_execz .LBB0_85
	v_mov_b64_e32 v[22:23], s[8:9]
	v_mov_b32_e32 v24, 2
	v_mov_b32_e32 v25, 1
	global_store_dwordx4 v[0:1], v[22:25], off offset:8
.LBB0_85:
	s_or_b64 exec, exec, s[12:13]
	s_lshl_b64 s[8:9], s[10:11], 12
	v_lshl_add_u64 v[8:9], v[6:7], 0, s[8:9]
	s_mov_b32 s8, 0
	v_lshlrev_b32_e32 v10, 6, v14
	v_mov_b32_e32 v22, 33
	v_mov_b32_e32 v23, v11
	v_mov_b32_e32 v24, v11
	v_mov_b32_e32 v25, v11
	v_readfirstlane_b32 s12, v8
	v_readfirstlane_b32 s13, v9
	s_mov_b32 s10, s8
	s_mov_b32 s11, s8
	s_mov_b32 s9, s8
	s_nop 1
	global_store_dwordx4 v10, v[22:25], s[12:13]
	s_nop 1
	v_mov_b64_e32 v[24:25], s[10:11]
	v_mov_b64_e32 v[22:23], s[8:9]
	global_store_dwordx4 v10, v[22:25], s[12:13] offset:16
	global_store_dwordx4 v10, v[22:25], s[12:13] offset:32
	global_store_dwordx4 v10, v[22:25], s[12:13] offset:48
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_93
	v_mov_b32_e32 v3, 0
	global_load_dwordx2 v[24:25], v3, s[4:5] offset:32 sc0 sc1
	global_load_dwordx2 v[6:7], v3, s[4:5] offset:40
	v_mov_b32_e32 v22, s6
	v_mov_b32_e32 v23, s7
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v6, s6, v6
	v_and_b32_e32 v7, s7, v7
	v_mul_lo_u32 v7, v7, 24
	v_mul_hi_u32 v12, v6, 24
	v_mul_lo_u32 v6, v6, 24
	v_add_u32_e32 v7, v12, v7
	v_lshl_add_u64 v[12:13], v[4:5], 0, v[6:7]
	global_store_dwordx2 v[12:13], v[24:25], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[6:7], v3, v[22:25], s[4:5] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[6:7], v[24:25]
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_89
	s_mov_b64 s[12:13], 0
.LBB0_88:
	s_sleep 1
	global_store_dwordx2 v[12:13], v[6:7], off
	v_mov_b32_e32 v4, s6
	v_mov_b32_e32 v5, s7
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[4:5], v3, v[4:7], s[4:5] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[4:5], v[6:7]
	s_or_b64 s[12:13], vcc, s[12:13]
	v_mov_b64_e32 v[6:7], v[4:5]
	s_andn2_b64 exec, exec, s[12:13]
	s_cbranch_execnz .LBB0_88
.LBB0_89:
	s_or_b64 exec, exec, s[10:11]
	v_mov_b32_e32 v7, 0
	global_load_dwordx2 v[4:5], v7, s[4:5] offset:16
	s_mov_b64 s[10:11], exec
	v_mbcnt_lo_u32_b32 v3, s10, 0
	v_mbcnt_hi_u32_b32 v3, s11, v3
	v_cmp_eq_u32_e32 vcc, 0, v3
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB0_91
	s_bcnt1_i32_b64 s10, s[10:11]
	v_mov_b32_e32 v6, s10
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[4:5], v[6:7], off offset:8 sc1
.LBB0_91:
	s_or_b64 exec, exec, s[12:13]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[6:7], v[4:5], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[6:7]
	s_cbranch_vccnz .LBB0_93
	global_load_dword v4, v[4:5], off offset:24
	v_mov_b32_e32 v5, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v4
	s_and_b32 m0, s10, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[6:7], v[4:5], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_93:
	s_or_b64 exec, exec, s[8:9]
	v_lshl_add_u64 v[4:5], v[8:9], 0, v[10:11]
	s_branch .LBB0_97
.LBB0_94:
	s_or_b64 exec, exec, s[8:9]
	v_readfirstlane_b32 s8, v3
	s_cmp_eq_u32 s8, 0
	s_cbranch_scc1 .LBB0_96
	s_sleep 1
	s_cbranch_execnz .LBB0_97
	s_branch .LBB0_99
.LBB0_96:
	s_branch .LBB0_99
.LBB0_97:
	v_mov_b32_e32 v3, 1
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_94
	global_load_dword v3, v[0:1], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v3, 1, v3
	s_branch .LBB0_94
.LBB0_99:
	global_load_dwordx2 v[4:5], v[4:5], off
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_102
	v_mov_b32_e32 v3, 0
	global_load_dwordx2 v[0:1], v3, s[4:5] offset:40
	global_load_dwordx2 v[10:11], v3, s[4:5] offset:24 sc0 sc1
	global_load_dwordx2 v[6:7], v3, s[4:5]
	s_waitcnt vmcnt(2)
	v_readfirstlane_b32 s10, v0
	v_readfirstlane_b32 s11, v1
	s_add_u32 s12, s10, 1
	s_addc_u32 s13, s11, 0
	s_add_u32 s0, s12, s6
	s_addc_u32 s1, s13, s7
	s_cmp_eq_u64 s[0:1], 0
	s_cselect_b32 s1, s13, s1
	s_cselect_b32 s0, s12, s0
	s_and_b64 s[6:7], s[0:1], s[10:11]
	s_mul_i32 s7, s7, 24
	s_mul_hi_u32 s10, s6, 24
	s_mul_i32 s6, s6, 24
	s_add_i32 s7, s10, s7
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[0:1], v[6:7], 0, s[6:7]
	v_mov_b32_e32 v8, s0
	global_store_dwordx2 v[0:1], v[10:11], off
	v_mov_b32_e32 v9, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[8:9], v3, v[8:11], s[4:5] offset:24 sc0 sc1
	s_mov_b64 s[6:7], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[8:9], v[10:11]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_102
.LBB0_101:
	s_sleep 1
	global_store_dwordx2 v[0:1], v[8:9], off
	v_mov_b32_e32 v6, s0
	v_mov_b32_e32 v7, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[6:7], v3, v[6:9], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[6:7], v[8:9]
	s_or_b64 s[6:7], vcc, s[6:7]
	v_mov_b64_e32 v[8:9], v[6:7]
	s_andn2_b64 exec, exec, s[6:7]
	s_cbranch_execnz .LBB0_101
.LBB0_102:
	s_or_b64 exec, exec, s[8:9]
	v_readfirstlane_b32 s0, v14
	v_mov_b64_e32 v[12:13], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v14
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_108
	v_mov_b32_e32 v3, 0
	global_load_dwordx2 v[8:9], v3, s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[0:1], v3, s[4:5] offset:40
	global_load_dwordx2 v[6:7], v3, s[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v0, v0, v8
	v_and_b32_e32 v1, v1, v9
	v_mul_lo_u32 v1, v1, 24
	v_mul_hi_u32 v10, v0, 24
	v_add_u32_e32 v1, v10, v1
	v_mul_lo_u32 v0, v0, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[0:1], v[6:7], 0, v[0:1]
	global_load_dwordx2 v[6:7], v[0:1], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[12:13], v3, v[6:9], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[12:13], v[8:9]
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_107
	s_mov_b64 s[10:11], 0
.LBB0_105:
	s_sleep 1
	global_load_dwordx2 v[0:1], v3, s[4:5] offset:40
	global_load_dwordx2 v[6:7], v3, s[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v0, v0, v12
	v_and_b32_e32 v8, v1, v13
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[0:1], s[12:13], v0, 24, v[6:7]
	v_mov_b32_e32 v6, v1
	v_mad_u64_u32 v[6:7], s[12:13], v8, 24, v[6:7]
	v_mov_b32_e32 v1, v6
	global_load_dwordx2 v[10:11], v[0:1], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[0:1], v3, v[10:13], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[0:1], v[12:13]
	s_or_b64 s[10:11], vcc, s[10:11]
	v_mov_b64_e32 v[12:13], v[0:1]
	s_andn2_b64 exec, exec, s[10:11]
	s_cbranch_execnz .LBB0_105
	s_or_b64 exec, exec, s[10:11]
	v_mov_b64_e32 v[12:13], v[0:1]
.LBB0_107:
	s_or_b64 exec, exec, s[8:9]
.LBB0_108:
	s_or_b64 exec, exec, s[6:7]
	v_mov_b32_e32 v17, 0
	global_load_dwordx2 v[0:1], v17, s[4:5] offset:40
	global_load_dwordx4 v[8:11], v17, s[4:5]
	v_readfirstlane_b32 s7, v13
	v_readfirstlane_b32 s6, v12
	s_mov_b64 s[8:9], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s10, v0
	v_readfirstlane_b32 s11, v1
	s_and_b64 s[10:11], s[10:11], s[6:7]
	s_mul_i32 s12, s11, 24
	s_mul_hi_u32 s13, s10, 24
	s_add_i32 s13, s13, s12
	s_mul_i32 s12, s10, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[0:1], v[8:9], 0, s[12:13]
	s_and_saveexec_b64 s[12:13], s[0:1]
	s_cbranch_execz .LBB0_110
	v_mov_b64_e32 v[22:23], s[8:9]
	v_mov_b32_e32 v24, 2
	v_mov_b32_e32 v25, 1
	global_store_dwordx4 v[0:1], v[22:25], off offset:8
.LBB0_110:
	s_or_b64 exec, exec, s[12:13]
	s_lshl_b64 s[8:9], s[10:11], 12
	v_lshl_add_u64 v[10:11], v[10:11], 0, s[8:9]
	v_and_b32_e32 v3, 0xffffff1f, v4
	v_or_b32_e32 v4, 0xc0, v3
	v_lshlrev_b32_e32 v16, 6, v14
	v_mov_b32_e32 v7, 0x41207265
	v_mov_b32_e32 v6, 0x7466415b
	v_readfirstlane_b32 s8, v10
	v_readfirstlane_b32 s9, v11
	s_nop 4
	global_store_dwordx4 v16, v[4:7], s[8:9]
	s_nop 1
	v_mov_b32_e32 v4, 0x205d4444
	v_mov_b32_e32 v5, 0x3d646974
	v_mov_b32_e32 v6, 0x203a6425
	v_mov_b32_e32 v7, 0x2e253d41
	global_store_dwordx4 v16, v[4:7], s[8:9] offset:16
	s_nop 1
	v_mov_b32_e32 v4, 0x202c6633
	v_mov_b32_e32 v5, 0x2e253d42
	v_mov_b32_e32 v7, 0x2e253d43
	v_mov_b32_e32 v6, v4
	global_store_dwordx4 v16, v[4:7], s[8:9] offset:32
	s_nop 1
	v_mov_b32_e32 v4, 0xa6633
	v_mov_b32_e32 v5, v17
	v_mov_b32_e32 v6, v17
	v_mov_b32_e32 v7, v17
	global_store_dwordx4 v16, v[4:7], s[8:9] offset:48
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_118
	v_mov_b32_e32 v3, 0
	global_load_dwordx2 v[24:25], v3, s[4:5] offset:32 sc0 sc1
	global_load_dwordx2 v[4:5], v3, s[4:5] offset:40
	v_mov_b32_e32 v22, s6
	v_mov_b32_e32 v23, s7
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v4
	v_readfirstlane_b32 s11, v5
	s_and_b64 s[10:11], s[10:11], s[6:7]
	s_mul_i32 s11, s11, 24
	s_mul_hi_u32 s12, s10, 24
	s_mul_i32 s10, s10, 24
	s_add_i32 s11, s12, s11
	v_lshl_add_u64 v[8:9], v[8:9], 0, s[10:11]
	global_store_dwordx2 v[8:9], v[24:25], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[6:7], v3, v[22:25], s[4:5] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[6:7], v[24:25]
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_114
	s_mov_b64 s[12:13], 0
.LBB0_113:
	s_sleep 1
	global_store_dwordx2 v[8:9], v[6:7], off
	v_mov_b32_e32 v4, s6
	v_mov_b32_e32 v5, s7
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[4:5], v3, v[4:7], s[4:5] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[4:5], v[6:7]
	s_or_b64 s[12:13], vcc, s[12:13]
	v_mov_b64_e32 v[6:7], v[4:5]
	s_andn2_b64 exec, exec, s[12:13]
	s_cbranch_execnz .LBB0_113
.LBB0_114:
	s_or_b64 exec, exec, s[10:11]
	v_mov_b32_e32 v7, 0
	global_load_dwordx2 v[4:5], v7, s[4:5] offset:16
	s_mov_b64 s[10:11], exec
	v_mbcnt_lo_u32_b32 v3, s10, 0
	v_mbcnt_hi_u32_b32 v3, s11, v3
	v_cmp_eq_u32_e32 vcc, 0, v3
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB0_116
	s_bcnt1_i32_b64 s10, s[10:11]
	v_mov_b32_e32 v6, s10
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[4:5], v[6:7], off offset:8 sc1
.LBB0_116:
	s_or_b64 exec, exec, s[12:13]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[6:7], v[4:5], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[6:7]
	s_cbranch_vccnz .LBB0_118
	global_load_dword v4, v[4:5], off offset:24
	v_mov_b32_e32 v5, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v4
	s_and_b32 m0, s10, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[6:7], v[4:5], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_118:
	s_or_b64 exec, exec, s[8:9]
	v_lshl_add_u64 v[4:5], v[10:11], 0, v[16:17]
	s_branch .LBB0_122
.LBB0_119:
	s_or_b64 exec, exec, s[8:9]
	v_readfirstlane_b32 s8, v3
	s_cmp_eq_u32 s8, 0
	s_cbranch_scc1 .LBB0_121
	s_sleep 1
	s_cbranch_execnz .LBB0_122
	s_branch .LBB0_124
.LBB0_121:
	s_branch .LBB0_124
.LBB0_122:
	v_mov_b32_e32 v3, 1
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_119
	global_load_dword v3, v[0:1], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v3, 1, v3
	s_branch .LBB0_119
.LBB0_124:
	global_load_dwordx2 v[0:1], v[4:5], off
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_127
	v_mov_b32_e32 v3, 0
	global_load_dwordx2 v[4:5], v3, s[4:5] offset:40
	global_load_dwordx2 v[12:13], v3, s[4:5] offset:24 sc0 sc1
	global_load_dwordx2 v[6:7], v3, s[4:5]
	s_waitcnt vmcnt(2)
	v_readfirstlane_b32 s10, v4
	v_readfirstlane_b32 s11, v5
	s_add_u32 s12, s10, 1
	s_addc_u32 s13, s11, 0
	s_add_u32 s0, s12, s6
	s_addc_u32 s1, s13, s7
	s_cmp_eq_u64 s[0:1], 0
	s_cselect_b32 s1, s13, s1
	s_cselect_b32 s0, s12, s0
	s_and_b64 s[6:7], s[0:1], s[10:11]
	s_mul_i32 s7, s7, 24
	s_mul_hi_u32 s10, s6, 24
	s_mul_i32 s6, s6, 24
	s_add_i32 s7, s10, s7
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[8:9], v[6:7], 0, s[6:7]
	v_mov_b32_e32 v10, s0
	global_store_dwordx2 v[8:9], v[12:13], off
	v_mov_b32_e32 v11, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[6:7], v3, v[10:13], s[4:5] offset:24 sc0 sc1
	s_mov_b64 s[6:7], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[6:7], v[12:13]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_127
.LBB0_126:
	s_sleep 1
	global_store_dwordx2 v[8:9], v[6:7], off
	v_mov_b32_e32 v4, s0
	v_mov_b32_e32 v5, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[4:5], v3, v[4:7], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[4:5], v[6:7]
	s_or_b64 s[6:7], vcc, s[6:7]
	v_mov_b64_e32 v[6:7], v[4:5]
	s_andn2_b64 exec, exec, s[6:7]
	s_cbranch_execnz .LBB0_126
.LBB0_127:
	s_or_b64 exec, exec, s[8:9]
	v_readfirstlane_b32 s0, v14
	v_mov_b64_e32 v[10:11], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v14
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_133
	v_mov_b32_e32 v3, 0
	global_load_dwordx2 v[6:7], v3, s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[4:5], v3, s[4:5] offset:40
	global_load_dwordx2 v[8:9], v3, s[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v4, v4, v6
	v_and_b32_e32 v5, v5, v7
	v_mul_lo_u32 v5, v5, 24
	v_mul_hi_u32 v10, v4, 24
	v_add_u32_e32 v5, v10, v5
	v_mul_lo_u32 v4, v4, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[4:5], v[8:9], 0, v[4:5]
	global_load_dwordx2 v[4:5], v[4:5], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v3, v[4:7], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[10:11], v[6:7]
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_132
	s_mov_b64 s[10:11], 0
.LBB0_130:
	s_sleep 1
	global_load_dwordx2 v[4:5], v3, s[4:5] offset:40
	global_load_dwordx2 v[6:7], v3, s[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v4, v4, v10
	v_and_b32_e32 v8, v5, v11
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[4:5], s[12:13], v4, 24, v[6:7]
	v_mov_b32_e32 v6, v5
	v_mad_u64_u32 v[6:7], s[12:13], v8, 24, v[6:7]
	v_mov_b32_e32 v5, v6
	global_load_dwordx2 v[8:9], v[4:5], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[4:5], v3, v[8:11], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[4:5], v[10:11]
	s_or_b64 s[10:11], vcc, s[10:11]
	v_mov_b64_e32 v[10:11], v[4:5]
	s_andn2_b64 exec, exec, s[10:11]
	s_cbranch_execnz .LBB0_130
	s_or_b64 exec, exec, s[10:11]
	v_mov_b64_e32 v[10:11], v[4:5]
.LBB0_132:
	s_or_b64 exec, exec, s[8:9]
.LBB0_133:
	s_or_b64 exec, exec, s[6:7]
	v_mov_b32_e32 v3, 0
	global_load_dwordx2 v[4:5], v3, s[4:5] offset:40
	global_load_dwordx4 v[6:9], v3, s[4:5]
	v_readfirstlane_b32 s7, v11
	v_readfirstlane_b32 s6, v10
	s_mov_b64 s[8:9], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s10, v4
	v_readfirstlane_b32 s11, v5
	s_and_b64 s[10:11], s[10:11], s[6:7]
	s_mul_i32 s12, s11, 24
	s_mul_hi_u32 s13, s10, 24
	s_add_i32 s13, s13, s12
	s_mul_i32 s12, s10, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[10:11], v[6:7], 0, s[12:13]
	s_and_saveexec_b64 s[12:13], s[0:1]
	s_cbranch_execz .LBB0_135
	v_mov_b64_e32 v[12:13], s[8:9]
	v_mov_b32_e32 v14, 2
	v_mov_b32_e32 v15, 1
	global_store_dwordx4 v[10:11], v[12:15], off offset:8
.LBB0_135:
	s_or_b64 exec, exec, s[12:13]
	s_lshl_b64 s[8:9], s[10:11], 12
	v_lshl_add_u64 v[4:5], v[8:9], 0, s[8:9]
	v_and_b32_e32 v0, 0xffffff1d, v0
	v_or_b32_e32 v0, 0x82, v0
	v_readfirstlane_b32 s8, v4
	v_readfirstlane_b32 s9, v5
	s_mov_b32 s10, 0
	v_cvt_f64_f32_e32 v[12:13], v18
	v_cvt_f64_f32_e32 v[14:15], v19
	v_cvt_f64_f32_e32 v[18:19], v20
	s_nop 0
	global_store_dwordx4 v16, v[0:3], s[8:9]
	global_store_dwordx4 v16, v[12:15], s[8:9] offset:16
	v_mov_b32_e32 v20, s10
	v_mov_b32_e32 v21, s10
	v_mov_b32_e32 v2, v3
	v_mov_b32_e32 v4, v3
	v_mov_b32_e32 v5, v3
	global_store_dwordx4 v16, v[18:21], s[8:9] offset:32
	global_store_dwordx4 v16, v[2:5], s[8:9] offset:48
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_143
	v_mov_b32_e32 v8, 0
	global_load_dwordx2 v[14:15], v8, s[4:5] offset:32 sc0 sc1
	global_load_dwordx2 v[0:1], v8, s[4:5] offset:40
	v_mov_b32_e32 v12, s6
	v_mov_b32_e32 v13, s7
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v0
	v_readfirstlane_b32 s11, v1
	s_and_b64 s[10:11], s[10:11], s[6:7]
	s_mul_i32 s11, s11, 24
	s_mul_hi_u32 s12, s10, 24
	s_mul_i32 s10, s10, 24
	s_add_i32 s11, s12, s11
	v_lshl_add_u64 v[4:5], v[6:7], 0, s[10:11]
	global_store_dwordx2 v[4:5], v[14:15], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v8, v[12:15], s[4:5] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[2:3], v[14:15]
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_139
	s_mov_b64 s[12:13], 0
.LBB0_138:
	s_sleep 1
	global_store_dwordx2 v[4:5], v[2:3], off
	v_mov_b32_e32 v0, s6
	v_mov_b32_e32 v1, s7
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[0:1], v8, v[0:3], s[4:5] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[0:1], v[2:3]
	s_or_b64 s[12:13], vcc, s[12:13]
	v_mov_b64_e32 v[2:3], v[0:1]
	s_andn2_b64 exec, exec, s[12:13]
	s_cbranch_execnz .LBB0_138
.LBB0_139:
	s_or_b64 exec, exec, s[10:11]
	v_mov_b32_e32 v3, 0
	global_load_dwordx2 v[0:1], v3, s[4:5] offset:16
	s_mov_b64 s[10:11], exec
	v_mbcnt_lo_u32_b32 v2, s10, 0
	v_mbcnt_hi_u32_b32 v2, s11, v2
	v_cmp_eq_u32_e32 vcc, 0, v2
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB0_141
	s_bcnt1_i32_b64 s10, s[10:11]
	v_mov_b32_e32 v2, s10
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[0:1], v[2:3], off offset:8 sc1
.LBB0_141:
	s_or_b64 exec, exec, s[12:13]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[2:3], v[0:1], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[2:3]
	s_cbranch_vccnz .LBB0_143
	global_load_dword v0, v[0:1], off offset:24
	v_mov_b32_e32 v1, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v0
	s_and_b32 m0, s10, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[2:3], v[0:1], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_143:
	s_or_b64 exec, exec, s[8:9]
	s_branch .LBB0_147
.LBB0_144:
	s_or_b64 exec, exec, s[8:9]
	v_readfirstlane_b32 s8, v0
	s_cmp_eq_u32 s8, 0
	s_cbranch_scc1 .LBB0_146
	s_sleep 1
	s_cbranch_execnz .LBB0_147
	s_branch .LBB0_149
.LBB0_146:
	s_branch .LBB0_149
.LBB0_147:
	v_mov_b32_e32 v0, 1
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_144
	global_load_dword v0, v[10:11], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v0, 1, v0
	s_branch .LBB0_144
.LBB0_149:
	s_and_b64 exec, exec, s[0:1]
	s_cbranch_execz .LBB0_152
	v_mov_b32_e32 v6, 0
	global_load_dwordx2 v[0:1], v6, s[4:5] offset:40
	global_load_dwordx2 v[10:11], v6, s[4:5] offset:24 sc0 sc1
	global_load_dwordx2 v[2:3], v6, s[4:5]
	s_waitcnt vmcnt(2)
	v_readfirstlane_b32 s8, v0
	v_readfirstlane_b32 s9, v1
	s_add_u32 s10, s8, 1
	s_addc_u32 s11, s9, 0
	s_add_u32 s0, s10, s6
	s_addc_u32 s1, s11, s7
	s_cmp_eq_u64 s[0:1], 0
	s_cselect_b32 s1, s11, s1
	s_cselect_b32 s0, s10, s0
	s_and_b64 s[6:7], s[0:1], s[8:9]
	s_mul_i32 s7, s7, 24
	s_mul_hi_u32 s8, s6, 24
	s_mul_i32 s6, s6, 24
	s_add_i32 s7, s8, s7
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[4:5], v[2:3], 0, s[6:7]
	v_mov_b32_e32 v8, s0
	global_store_dwordx2 v[4:5], v[10:11], off
	v_mov_b32_e32 v9, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v6, v[8:11], s[4:5] offset:24 sc0 sc1
	s_mov_b64 s[6:7], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[2:3], v[10:11]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_152
.LBB0_151:
	s_sleep 1
	global_store_dwordx2 v[4:5], v[2:3], off
	v_mov_b32_e32 v0, s0
	v_mov_b32_e32 v1, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[0:1], v6, v[0:3], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[0:1], v[2:3]
	s_or_b64 s[6:7], vcc, s[6:7]
	v_mov_b64_e32 v[2:3], v[0:1]
	s_andn2_b64 exec, exec, s[6:7]
	s_cbranch_execnz .LBB0_151
.LBB0_152:
	s_or_b64 exec, exec, s[2:3]
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
		.amdhsa_next_free_vgpr 64
		.amdhsa_next_free_sgpr 20
		.amdhsa_accum_offset 28
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

	.set _Z9vectorAddPKfS0_Pfi.num_vgpr, 26
	.set _Z9vectorAddPKfS0_Pfi.num_agpr, 0
	.set _Z9vectorAddPKfS0_Pfi.numbered_sgpr, 16
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
  - .offset: 32
    .size: 4
    .value_kind: hidden_block_count_x
  - .offset: 36
    .size: 4
    .value_kind: hidden_block_count_y
  - .offset: 40
    .size: 4
    .value_kind: hidden_block_count_z
  - .offset: 44
    .size: 2
    .value_kind: hidden_group_size_x
  - .offset: 46
    .size: 2
    .value_kind: hidden_group_size_y
  - .offset: 48
    .size: 2
    .value_kind: hidden_group_size_z
  - .offset: 50
    .size: 2
    .value_kind: hidden_remainder_x
  - .offset: 52
    .size: 2
    .value_kind: hidden_remainder_y
  - .offset: 54
    .size: 2
    .value_kind: hidden_remainder_z
  - .offset: 72
    .size: 8
    .value_kind: hidden_global_offset_x
  - .offset: 80
    .size: 8
    .value_kind: hidden_global_offset_y
  - .offset: 88
    .size: 8
    .value_kind: hidden_global_offset_z
  - .offset: 96
    .size: 2
    .value_kind: hidden_grid_dims
  - .offset: 112
    .size: 8
    .value_kind: hidden_hostcall_buffer
  .group_segment_fixed_size: 0
  .kernarg_segment_align: 8
  .kernarg_segment_size: 288
  .max_flat_workgroup_size: 256
  .name: _Z9vectorAddPKfS0_Pfi
  .private_segment_fixed_size: 0
  .sgpr_count: 22
  .sgpr_spill_count: 0
  .symbol: _Z9vectorAddPKfS0_Pfi.kd
  .uniform_work_group_size: 1
  .uses_dynamic_stack: false
  .vgpr_count: 26
  .vgpr_spill_count: 0
  .wavefront_size: 64
amdhsa.target: amdgcn-amd-amdhsa--gfx950
amdhsa.version:
- 1
- 2
...
...

	.end_amdgpu_metadata
