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
	v_mov_b32 v98, v0
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v99, v0
	;;#ASMEND
	;;#ASMSTART
	v_mbcnt_lo_u32_b32 v100, -1, 0
	;;#ASMEND
	;;#ASMSTART
	v_mbcnt_hi_u32_b32 v100, -1, v100
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
	s_mov_b32 s20, s4
	;;#ASMEND
	;;#ASMSTART
	s_mov_b32 s21, s2
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
	v_mov_b32 v88, v6
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v89, v7
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v90, v2
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v91, v6
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v92, v7
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v91, v6
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v92, v7
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v93, v6
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v94, v7
	;;#ASMEND
	;;#ASMSTART
		v_add_f32_e32 v2, v6, v7
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v95, v6
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v96, v7
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v97, v2
	;;#ASMEND
	;;#ASMSTART
		global_store_dword v[0:1], v2, off
	;;#ASMEND
	;;#ASMSTART
	s_mov_b64 s[0:1], s[18:19]
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v4, s20
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v0, s21
	;;#ASMEND
	s_load_dwordx2 s[2:3], s[0:1], 0x70
	v_mbcnt_lo_u32_b32 v1, -1, 0
	v_mbcnt_hi_u32_b32 v16, -1, v1
	v_mov_b64_e32 v[10:11], 0
	v_readfirstlane_b32 s0, v16
	s_nop 1
	v_cmp_eq_u32_e64 s[0:1], s0, v16
	s_and_saveexec_b64 s[4:5], s[0:1]
	s_cbranch_execz .LBB0_6
	v_mov_b32_e32 v1, 0
	s_waitcnt lgkmcnt(0)
	global_load_dwordx2 v[8:9], v1, s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[2:3], v1, s[2:3] offset:40
	global_load_dwordx2 v[6:7], v1, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v8
	v_and_b32_e32 v3, v3, v9
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v5, v2, 24
	v_add_u32_e32 v3, v5, v3
	v_mul_lo_u32 v2, v2, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[6:7], 0, v[2:3]
	global_load_dwordx2 v[6:7], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v1, v[6:9], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[10:11], v[8:9]
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_5
	s_mov_b64 s[8:9], 0
.LBB0_3:
	s_sleep 1
	global_load_dwordx2 v[2:3], v1, s[2:3] offset:40
	global_load_dwordx2 v[6:7], v1, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v10
	v_and_b32_e32 v5, v3, v11
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[2:3], s[10:11], v2, 24, v[6:7]
	v_mov_b32_e32 v6, v3
	v_mad_u64_u32 v[6:7], s[10:11], v5, 24, v[6:7]
	v_mov_b32_e32 v3, v6
	global_load_dwordx2 v[8:9], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v1, v[8:11], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[2:3], v[10:11]
	s_or_b64 s[8:9], vcc, s[8:9]
	v_mov_b64_e32 v[10:11], v[2:3]
	s_andn2_b64 exec, exec, s[8:9]
	s_cbranch_execnz .LBB0_3
	s_or_b64 exec, exec, s[8:9]
	v_mov_b64_e32 v[10:11], v[2:3]
.LBB0_5:
	s_or_b64 exec, exec, s[6:7]
.LBB0_6:
	s_or_b64 exec, exec, s[4:5]
	v_mov_b32_e32 v13, 0
	s_waitcnt lgkmcnt(0)
	global_load_dwordx2 v[2:3], v13, s[2:3] offset:40
	global_load_dwordx4 v[6:9], v13, s[2:3]
	v_readfirstlane_b32 s5, v11
	v_readfirstlane_b32 s4, v10
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s8, v2
	v_readfirstlane_b32 s9, v3
	s_and_b64 s[8:9], s[8:9], s[4:5]
	s_mul_i32 s10, s9, 24
	s_mul_hi_u32 s11, s8, 24
	s_add_i32 s11, s11, s10
	s_mul_i32 s10, s8, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[6:7], 0, s[10:11]
	s_and_saveexec_b64 s[10:11], s[0:1]
	s_cbranch_execz .LBB0_8
	v_mov_b64_e32 v[18:19], s[6:7]
	v_mov_b32_e32 v20, 2
	v_mov_b32_e32 v21, 1
	global_store_dwordx4 v[2:3], v[18:21], off offset:8
.LBB0_8:
	s_or_b64 exec, exec, s[10:11]
	s_lshl_b64 s[6:7], s[8:9], 12
	v_lshl_add_u64 v[10:11], v[8:9], 0, s[6:7]
	s_mov_b32 s8, 0
	v_lshlrev_b32_e32 v12, 6, v16
	v_mov_b32_e32 v18, 33
	v_mov_b32_e32 v19, v13
	v_mov_b32_e32 v20, v13
	v_mov_b32_e32 v21, v13
	v_readfirstlane_b32 s6, v10
	v_readfirstlane_b32 s7, v11
	s_mov_b32 s10, s8
	s_mov_b32 s11, s8
	s_mov_b32 s9, s8
	s_nop 1
	global_store_dwordx4 v12, v[18:21], s[6:7]
	s_nop 1
	v_mov_b64_e32 v[20:21], s[10:11]
	v_mov_b64_e32 v[18:19], s[8:9]
	global_store_dwordx4 v12, v[18:21], s[6:7] offset:16
	global_store_dwordx4 v12, v[18:21], s[6:7] offset:32
	global_store_dwordx4 v12, v[18:21], s[6:7] offset:48
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_16
	v_mov_b32_e32 v1, 0
	global_load_dwordx2 v[20:21], v1, s[2:3] offset:32 sc0 sc1
	global_load_dwordx2 v[8:9], v1, s[2:3] offset:40
	v_mov_b32_e32 v18, s4
	v_mov_b32_e32 v19, s5
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v5, s4, v8
	v_and_b32_e32 v8, s5, v9
	v_mul_lo_u32 v9, v8, 24
	v_mul_hi_u32 v14, v5, 24
	v_mul_lo_u32 v8, v5, 24
	v_add_u32_e32 v9, v14, v9
	v_lshl_add_u64 v[14:15], v[6:7], 0, v[8:9]
	global_store_dwordx2 v[14:15], v[20:21], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[8:9], v1, v[18:21], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[8:9], v[20:21]
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_12
	s_mov_b64 s[10:11], 0
.LBB0_11:
	s_sleep 1
	global_store_dwordx2 v[14:15], v[8:9], off
	v_mov_b32_e32 v6, s4
	v_mov_b32_e32 v7, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[6:7], v1, v[6:9], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[6:7], v[8:9]
	s_or_b64 s[10:11], vcc, s[10:11]
	v_mov_b64_e32 v[8:9], v[6:7]
	s_andn2_b64 exec, exec, s[10:11]
	s_cbranch_execnz .LBB0_11
.LBB0_12:
	s_or_b64 exec, exec, s[8:9]
	v_mov_b32_e32 v9, 0
	global_load_dwordx2 v[6:7], v9, s[2:3] offset:16
	s_mov_b64 s[8:9], exec
	v_mbcnt_lo_u32_b32 v1, s8, 0
	v_mbcnt_hi_u32_b32 v1, s9, v1
	v_cmp_eq_u32_e32 vcc, 0, v1
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_14
	s_bcnt1_i32_b64 s8, s[8:9]
	v_mov_b32_e32 v8, s8
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[6:7], v[8:9], off offset:8 sc1
.LBB0_14:
	s_or_b64 exec, exec, s[10:11]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[8:9], v[6:7], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[8:9]
	s_cbranch_vccnz .LBB0_16
	global_load_dword v6, v[6:7], off offset:24
	v_mov_b32_e32 v7, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s8, v6
	s_and_b32 m0, s8, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[8:9], v[6:7], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_16:
	s_or_b64 exec, exec, s[6:7]
	v_lshl_add_u64 v[6:7], v[10:11], 0, v[12:13]
	s_branch .LBB0_20
.LBB0_17:
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s6, v1
	s_cmp_eq_u32 s6, 0
	s_cbranch_scc1 .LBB0_19
	s_sleep 1
	s_cbranch_execnz .LBB0_20
	s_branch .LBB0_22
.LBB0_19:
	s_branch .LBB0_22
.LBB0_20:
	v_mov_b32_e32 v1, 1
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_17
	global_load_dword v1, v[2:3], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v1, 1, v1
	s_branch .LBB0_17
.LBB0_22:
	global_load_dwordx2 v[6:7], v[6:7], off
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_25
	v_mov_b32_e32 v1, 0
	global_load_dwordx2 v[2:3], v1, s[2:3] offset:40
	global_load_dwordx2 v[12:13], v1, s[2:3] offset:24 sc0 sc1
	global_load_dwordx2 v[8:9], v1, s[2:3]
	s_waitcnt vmcnt(2)
	v_readfirstlane_b32 s8, v2
	v_readfirstlane_b32 s9, v3
	s_add_u32 s10, s8, 1
	s_addc_u32 s11, s9, 0
	s_add_u32 s0, s10, s4
	s_addc_u32 s1, s11, s5
	s_cmp_eq_u64 s[0:1], 0
	s_cselect_b32 s1, s11, s1
	s_cselect_b32 s0, s10, s0
	s_and_b64 s[4:5], s[0:1], s[8:9]
	s_mul_i32 s5, s5, 24
	s_mul_hi_u32 s8, s4, 24
	s_mul_i32 s4, s4, 24
	s_add_i32 s5, s8, s5
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[8:9], 0, s[4:5]
	v_mov_b32_e32 v10, s0
	global_store_dwordx2 v[2:3], v[12:13], off
	v_mov_b32_e32 v11, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v1, v[10:13], s[2:3] offset:24 sc0 sc1
	s_mov_b64 s[4:5], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[10:11], v[12:13]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_25
.LBB0_24:
	s_sleep 1
	global_store_dwordx2 v[2:3], v[10:11], off
	v_mov_b32_e32 v8, s0
	v_mov_b32_e32 v9, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[8:9], v1, v[8:11], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[8:9], v[10:11]
	s_or_b64 s[4:5], vcc, s[4:5]
	v_mov_b64_e32 v[10:11], v[8:9]
	s_andn2_b64 exec, exec, s[4:5]
	s_cbranch_execnz .LBB0_24
.LBB0_25:
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s0, v16
	v_mov_b64_e32 v[14:15], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v16
	s_and_saveexec_b64 s[4:5], s[0:1]
	s_cbranch_execz .LBB0_31
	v_mov_b32_e32 v1, 0
	global_load_dwordx2 v[10:11], v1, s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[2:3], v1, s[2:3] offset:40
	global_load_dwordx2 v[8:9], v1, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v10
	v_and_b32_e32 v3, v3, v11
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v5, v2, 24
	v_add_u32_e32 v3, v5, v3
	v_mul_lo_u32 v2, v2, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[8:9], 0, v[2:3]
	global_load_dwordx2 v[8:9], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[14:15], v1, v[8:11], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[14:15], v[10:11]
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_30
	s_mov_b64 s[8:9], 0
.LBB0_28:
	s_sleep 1
	global_load_dwordx2 v[2:3], v1, s[2:3] offset:40
	global_load_dwordx2 v[8:9], v1, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v14
	v_and_b32_e32 v5, v3, v15
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[2:3], s[10:11], v2, 24, v[8:9]
	v_mov_b32_e32 v8, v3
	v_mad_u64_u32 v[8:9], s[10:11], v5, 24, v[8:9]
	v_mov_b32_e32 v3, v8
	global_load_dwordx2 v[12:13], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v1, v[12:15], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[2:3], v[14:15]
	s_or_b64 s[8:9], vcc, s[8:9]
	v_mov_b64_e32 v[14:15], v[2:3]
	s_andn2_b64 exec, exec, s[8:9]
	s_cbranch_execnz .LBB0_28
	s_or_b64 exec, exec, s[8:9]
	v_mov_b64_e32 v[14:15], v[2:3]
.LBB0_30:
	s_or_b64 exec, exec, s[6:7]
.LBB0_31:
	s_or_b64 exec, exec, s[4:5]
	v_mov_b32_e32 v19, 0
	global_load_dwordx2 v[2:3], v19, s[2:3] offset:40
	global_load_dwordx4 v[10:13], v19, s[2:3]
	v_readfirstlane_b32 s5, v15
	v_readfirstlane_b32 s4, v14
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s8, v2
	v_readfirstlane_b32 s9, v3
	s_and_b64 s[8:9], s[8:9], s[4:5]
	s_mul_i32 s10, s9, 24
	s_mul_hi_u32 s11, s8, 24
	s_add_i32 s11, s11, s10
	s_mul_i32 s10, s8, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[10:11], 0, s[10:11]
	s_and_saveexec_b64 s[10:11], s[0:1]
	s_cbranch_execz .LBB0_33
	v_mov_b64_e32 v[20:21], s[6:7]
	v_mov_b32_e32 v22, 2
	v_mov_b32_e32 v23, 1
	global_store_dwordx4 v[2:3], v[20:23], off offset:8
.LBB0_33:
	s_or_b64 exec, exec, s[10:11]
	s_lshl_b64 s[6:7], s[8:9], 12
	v_lshl_add_u64 v[12:13], v[12:13], 0, s[6:7]
	v_and_b32_e32 v1, 0xffffff1f, v6
	v_or_b32_e32 v6, 0x80, v1
	v_lshlrev_b32_e32 v18, 6, v16
	v_mov_b32_e32 v9, 0x65542052
	v_mov_b32_e32 v8, 0x5047535b
	v_readfirstlane_b32 s6, v12
	v_readfirstlane_b32 s7, v13
	s_nop 4
	global_store_dwordx4 v18, v[6:9], s[6:7]
	s_nop 1
	v_mov_b32_e32 v6, 0x205d7473
	v_mov_b32_e32 v7, 0x64253d6e
	v_mov_b32_e32 v8, 0x6162202c
	v_mov_b32_e32 v9, 0x695f6573
	global_store_dwordx4 v18, v[6:9], s[6:7] offset:16
	s_nop 1
	v_mov_b32_e32 v6, 0x253d7864
	v_mov_b32_e32 v7, 0xa64
	v_mov_b32_e32 v8, v19
	v_mov_b32_e32 v9, v19
	global_store_dwordx4 v18, v[6:9], s[6:7] offset:32
	s_nop 1
	v_mov_b32_e32 v6, v19
	v_mov_b32_e32 v7, v19
	global_store_dwordx4 v18, v[6:9], s[6:7] offset:48
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_41
	v_mov_b32_e32 v1, 0
	global_load_dwordx2 v[22:23], v1, s[2:3] offset:32 sc0 sc1
	global_load_dwordx2 v[6:7], v1, s[2:3] offset:40
	v_mov_b32_e32 v20, s4
	v_mov_b32_e32 v21, s5
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s8, v6
	v_readfirstlane_b32 s9, v7
	s_and_b64 s[8:9], s[8:9], s[4:5]
	s_mul_i32 s9, s9, 24
	s_mul_hi_u32 s10, s8, 24
	s_mul_i32 s8, s8, 24
	s_add_i32 s9, s10, s9
	v_lshl_add_u64 v[10:11], v[10:11], 0, s[8:9]
	global_store_dwordx2 v[10:11], v[22:23], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[8:9], v1, v[20:23], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[8:9], v[22:23]
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_37
	s_mov_b64 s[10:11], 0
.LBB0_36:
	s_sleep 1
	global_store_dwordx2 v[10:11], v[8:9], off
	v_mov_b32_e32 v6, s4
	v_mov_b32_e32 v7, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[6:7], v1, v[6:9], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[6:7], v[8:9]
	s_or_b64 s[10:11], vcc, s[10:11]
	v_mov_b64_e32 v[8:9], v[6:7]
	s_andn2_b64 exec, exec, s[10:11]
	s_cbranch_execnz .LBB0_36
.LBB0_37:
	s_or_b64 exec, exec, s[8:9]
	v_mov_b32_e32 v9, 0
	global_load_dwordx2 v[6:7], v9, s[2:3] offset:16
	s_mov_b64 s[8:9], exec
	v_mbcnt_lo_u32_b32 v1, s8, 0
	v_mbcnt_hi_u32_b32 v1, s9, v1
	v_cmp_eq_u32_e32 vcc, 0, v1
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_39
	s_bcnt1_i32_b64 s8, s[8:9]
	v_mov_b32_e32 v8, s8
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[6:7], v[8:9], off offset:8 sc1
.LBB0_39:
	s_or_b64 exec, exec, s[10:11]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[8:9], v[6:7], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[8:9]
	s_cbranch_vccnz .LBB0_41
	global_load_dword v6, v[6:7], off offset:24
	v_mov_b32_e32 v7, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s8, v6
	s_and_b32 m0, s8, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[8:9], v[6:7], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_41:
	s_or_b64 exec, exec, s[6:7]
	v_lshl_add_u64 v[6:7], v[12:13], 0, v[18:19]
	s_branch .LBB0_45
.LBB0_42:
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s6, v1
	s_cmp_eq_u32 s6, 0
	s_cbranch_scc1 .LBB0_44
	s_sleep 1
	s_cbranch_execnz .LBB0_45
	s_branch .LBB0_47
.LBB0_44:
	s_branch .LBB0_47
.LBB0_45:
	v_mov_b32_e32 v1, 1
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_42
	global_load_dword v1, v[2:3], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v1, 1, v1
	s_branch .LBB0_42
.LBB0_47:
	global_load_dwordx2 v[2:3], v[6:7], off
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_50
	v_mov_b32_e32 v1, 0
	global_load_dwordx2 v[6:7], v1, s[2:3] offset:40
	global_load_dwordx2 v[14:15], v1, s[2:3] offset:24 sc0 sc1
	global_load_dwordx2 v[8:9], v1, s[2:3]
	s_waitcnt vmcnt(2)
	v_readfirstlane_b32 s8, v6
	v_readfirstlane_b32 s9, v7
	s_add_u32 s10, s8, 1
	s_addc_u32 s11, s9, 0
	s_add_u32 s0, s10, s4
	s_addc_u32 s1, s11, s5
	s_cmp_eq_u64 s[0:1], 0
	s_cselect_b32 s1, s11, s1
	s_cselect_b32 s0, s10, s0
	s_and_b64 s[4:5], s[0:1], s[8:9]
	s_mul_i32 s5, s5, 24
	s_mul_hi_u32 s8, s4, 24
	s_mul_i32 s4, s4, 24
	s_add_i32 s5, s8, s5
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[10:11], v[8:9], 0, s[4:5]
	v_mov_b32_e32 v12, s0
	global_store_dwordx2 v[10:11], v[14:15], off
	v_mov_b32_e32 v13, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[8:9], v1, v[12:15], s[2:3] offset:24 sc0 sc1
	s_mov_b64 s[4:5], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[8:9], v[14:15]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_50
.LBB0_49:
	s_sleep 1
	global_store_dwordx2 v[10:11], v[8:9], off
	v_mov_b32_e32 v6, s0
	v_mov_b32_e32 v7, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[6:7], v1, v[6:9], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[6:7], v[8:9]
	s_or_b64 s[4:5], vcc, s[4:5]
	v_mov_b64_e32 v[8:9], v[6:7]
	s_andn2_b64 exec, exec, s[4:5]
	s_cbranch_execnz .LBB0_49
.LBB0_50:
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s0, v16
	v_mov_b64_e32 v[10:11], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v16
	s_and_saveexec_b64 s[4:5], s[0:1]
	s_cbranch_execz .LBB0_56
	v_mov_b32_e32 v1, 0
	global_load_dwordx2 v[8:9], v1, s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[6:7], v1, s[2:3] offset:40
	global_load_dwordx2 v[10:11], v1, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v5, v6, v8
	v_and_b32_e32 v6, v7, v9
	v_mul_lo_u32 v6, v6, 24
	v_mul_hi_u32 v7, v5, 24
	v_add_u32_e32 v7, v7, v6
	v_mul_lo_u32 v6, v5, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[6:7], v[10:11], 0, v[6:7]
	global_load_dwordx2 v[6:7], v[6:7], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v1, v[6:9], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[10:11], v[8:9]
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_55
	s_mov_b64 s[8:9], 0
.LBB0_53:
	s_sleep 1
	global_load_dwordx2 v[6:7], v1, s[2:3] offset:40
	global_load_dwordx2 v[8:9], v1, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v6, v6, v10
	v_and_b32_e32 v5, v7, v11
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[6:7], s[10:11], v6, 24, v[8:9]
	v_mov_b32_e32 v8, v7
	v_mad_u64_u32 v[8:9], s[10:11], v5, 24, v[8:9]
	v_mov_b32_e32 v7, v8
	global_load_dwordx2 v[8:9], v[6:7], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[6:7], v1, v[8:11], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[6:7], v[10:11]
	s_or_b64 s[8:9], vcc, s[8:9]
	v_mov_b64_e32 v[10:11], v[6:7]
	s_andn2_b64 exec, exec, s[8:9]
	s_cbranch_execnz .LBB0_53
	s_or_b64 exec, exec, s[8:9]
	v_mov_b64_e32 v[10:11], v[6:7]
.LBB0_55:
	s_or_b64 exec, exec, s[6:7]
.LBB0_56:
	s_or_b64 exec, exec, s[4:5]
	v_mov_b32_e32 v5, 0
	global_load_dwordx2 v[12:13], v5, s[2:3] offset:40
	global_load_dwordx4 v[6:9], v5, s[2:3]
	v_readfirstlane_b32 s5, v11
	v_readfirstlane_b32 s4, v10
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s8, v12
	v_readfirstlane_b32 s9, v13
	s_and_b64 s[8:9], s[8:9], s[4:5]
	s_mul_i32 s10, s9, 24
	s_mul_hi_u32 s11, s8, 24
	s_add_i32 s11, s11, s10
	s_mul_i32 s10, s8, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[10:11], v[6:7], 0, s[10:11]
	s_and_saveexec_b64 s[10:11], s[0:1]
	s_cbranch_execz .LBB0_58
	v_mov_b64_e32 v[12:13], s[6:7]
	v_mov_b32_e32 v14, 2
	v_mov_b32_e32 v15, 1
	global_store_dwordx4 v[10:11], v[12:15], off offset:8
.LBB0_58:
	s_or_b64 exec, exec, s[10:11]
	s_lshl_b64 s[6:7], s[8:9], 12
	v_lshl_add_u64 v[8:9], v[8:9], 0, s[6:7]
	v_and_b32_e32 v1, 0xffffff1d, v2
	v_or_b32_e32 v2, 0x42, v1
	v_readfirstlane_b32 s6, v8
	v_readfirstlane_b32 s7, v9
	s_mov_b32 s8, 0
	v_mov_b32_e32 v1, s8
	s_mov_b32 s9, s8
	s_mov_b32 s10, s8
	s_mov_b32 s11, s8
	global_store_dwordx4 v18, v[2:5], s[6:7]
	s_nop 1
	v_mov_b32_e32 v2, s8
	v_mov_b32_e32 v3, s8
	global_store_dwordx4 v18, v[0:3], s[6:7] offset:16
	s_nop 1
	v_mov_b64_e32 v[0:1], s[8:9]
	v_mov_b64_e32 v[2:3], s[10:11]
	global_store_dwordx4 v18, v[0:3], s[6:7] offset:32
	global_store_dwordx4 v18, v[0:3], s[6:7] offset:48
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_66
	v_mov_b32_e32 v8, 0
	global_load_dwordx2 v[14:15], v8, s[2:3] offset:32 sc0 sc1
	global_load_dwordx2 v[0:1], v8, s[2:3] offset:40
	v_mov_b32_e32 v12, s4
	v_mov_b32_e32 v13, s5
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s8, v0
	v_readfirstlane_b32 s9, v1
	s_and_b64 s[8:9], s[8:9], s[4:5]
	s_mul_i32 s9, s9, 24
	s_mul_hi_u32 s10, s8, 24
	s_mul_i32 s8, s8, 24
	s_add_i32 s9, s10, s9
	v_lshl_add_u64 v[4:5], v[6:7], 0, s[8:9]
	global_store_dwordx2 v[4:5], v[14:15], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v8, v[12:15], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[2:3], v[14:15]
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_62
	s_mov_b64 s[10:11], 0
.LBB0_61:
	s_sleep 1
	global_store_dwordx2 v[4:5], v[2:3], off
	v_mov_b32_e32 v0, s4
	v_mov_b32_e32 v1, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[0:1], v8, v[0:3], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[0:1], v[2:3]
	s_or_b64 s[10:11], vcc, s[10:11]
	v_mov_b64_e32 v[2:3], v[0:1]
	s_andn2_b64 exec, exec, s[10:11]
	s_cbranch_execnz .LBB0_61
.LBB0_62:
	s_or_b64 exec, exec, s[8:9]
	v_mov_b32_e32 v3, 0
	global_load_dwordx2 v[0:1], v3, s[2:3] offset:16
	s_mov_b64 s[8:9], exec
	v_mbcnt_lo_u32_b32 v2, s8, 0
	v_mbcnt_hi_u32_b32 v2, s9, v2
	v_cmp_eq_u32_e32 vcc, 0, v2
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_64
	s_bcnt1_i32_b64 s8, s[8:9]
	v_mov_b32_e32 v2, s8
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[0:1], v[2:3], off offset:8 sc1
.LBB0_64:
	s_or_b64 exec, exec, s[10:11]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[2:3], v[0:1], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[2:3]
	s_cbranch_vccnz .LBB0_66
	global_load_dword v0, v[0:1], off offset:24
	v_mov_b32_e32 v1, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s8, v0
	s_and_b32 m0, s8, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[2:3], v[0:1], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_66:
	s_or_b64 exec, exec, s[6:7]
	s_branch .LBB0_70
.LBB0_67:
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s6, v0
	s_cmp_eq_u32 s6, 0
	s_cbranch_scc1 .LBB0_69
	s_sleep 1
	s_cbranch_execnz .LBB0_70
	s_branch .LBB0_72
.LBB0_69:
	s_branch .LBB0_72
.LBB0_70:
	v_mov_b32_e32 v0, 1
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_67
	global_load_dword v0, v[10:11], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v0, 1, v0
	s_branch .LBB0_67
.LBB0_72:
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_75
	v_mov_b32_e32 v6, 0
	global_load_dwordx2 v[0:1], v6, s[2:3] offset:40
	global_load_dwordx2 v[10:11], v6, s[2:3] offset:24 sc0 sc1
	global_load_dwordx2 v[2:3], v6, s[2:3]
	s_waitcnt vmcnt(2)
	v_readfirstlane_b32 s8, v0
	v_readfirstlane_b32 s9, v1
	s_add_u32 s10, s8, 1
	s_addc_u32 s11, s9, 0
	s_add_u32 s0, s10, s4
	s_addc_u32 s1, s11, s5
	s_cmp_eq_u64 s[0:1], 0
	s_cselect_b32 s1, s11, s1
	s_cselect_b32 s0, s10, s0
	s_and_b64 s[4:5], s[0:1], s[8:9]
	s_mul_i32 s5, s5, 24
	s_mul_hi_u32 s8, s4, 24
	s_mul_i32 s4, s4, 24
	s_add_i32 s5, s8, s5
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[4:5], v[2:3], 0, s[4:5]
	v_mov_b32_e32 v8, s0
	global_store_dwordx2 v[4:5], v[10:11], off
	v_mov_b32_e32 v9, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v6, v[8:11], s[2:3] offset:24 sc0 sc1
	s_mov_b64 s[4:5], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[2:3], v[10:11]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_75
.LBB0_74:
	s_sleep 1
	global_store_dwordx2 v[4:5], v[2:3], off
	v_mov_b32_e32 v0, s0
	v_mov_b32_e32 v1, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[0:1], v6, v[0:3], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[0:1], v[2:3]
	s_or_b64 s[4:5], vcc, s[4:5]
	v_mov_b64_e32 v[2:3], v[0:1]
	s_andn2_b64 exec, exec, s[4:5]
	s_cbranch_execnz .LBB0_74
.LBB0_75:
	s_or_b64 exec, exec, s[6:7]
	;;#ASMSTART
	v_mov_b32 v4, v99
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v0, v100
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v1, v99
	;;#ASMEND
	s_nop 0
	v_cmp_gt_i32_e32 vcc, 4, v1
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_151
	v_readfirstlane_b32 s0, v16
	v_mov_b64_e32 v[10:11], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v16
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_82
	v_mov_b32_e32 v1, 0
	global_load_dwordx2 v[8:9], v1, s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[2:3], v1, s[2:3] offset:40
	global_load_dwordx2 v[6:7], v1, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v8
	v_and_b32_e32 v3, v3, v9
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v5, v2, 24
	v_add_u32_e32 v3, v5, v3
	v_mul_lo_u32 v2, v2, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[6:7], 0, v[2:3]
	global_load_dwordx2 v[6:7], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v1, v[6:9], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[10:11], v[8:9]
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_81
	s_mov_b64 s[10:11], 0
.LBB0_79:
	s_sleep 1
	global_load_dwordx2 v[2:3], v1, s[2:3] offset:40
	global_load_dwordx2 v[6:7], v1, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v10
	v_and_b32_e32 v5, v3, v11
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[2:3], s[12:13], v2, 24, v[6:7]
	v_mov_b32_e32 v6, v3
	v_mad_u64_u32 v[6:7], s[12:13], v5, 24, v[6:7]
	v_mov_b32_e32 v3, v6
	global_load_dwordx2 v[8:9], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v1, v[8:11], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[2:3], v[10:11]
	s_or_b64 s[10:11], vcc, s[10:11]
	v_mov_b64_e32 v[10:11], v[2:3]
	s_andn2_b64 exec, exec, s[10:11]
	s_cbranch_execnz .LBB0_79
	s_or_b64 exec, exec, s[10:11]
	v_mov_b64_e32 v[10:11], v[2:3]
.LBB0_81:
	s_or_b64 exec, exec, s[8:9]
.LBB0_82:
	s_or_b64 exec, exec, s[6:7]
	v_mov_b32_e32 v19, 0
	global_load_dwordx2 v[2:3], v19, s[2:3] offset:40
	global_load_dwordx4 v[6:9], v19, s[2:3]
	v_readfirstlane_b32 s7, v11
	v_readfirstlane_b32 s6, v10
	s_mov_b64 s[8:9], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s10, v2
	v_readfirstlane_b32 s11, v3
	s_and_b64 s[10:11], s[10:11], s[6:7]
	s_mul_i32 s12, s11, 24
	s_mul_hi_u32 s13, s10, 24
	s_add_i32 s13, s13, s12
	s_mul_i32 s12, s10, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[6:7], 0, s[12:13]
	s_and_saveexec_b64 s[12:13], s[0:1]
	s_cbranch_execz .LBB0_84
	v_mov_b64_e32 v[10:11], s[8:9]
	v_mov_b32_e32 v12, 2
	v_mov_b32_e32 v13, 1
	global_store_dwordx4 v[2:3], v[10:13], off offset:8
.LBB0_84:
	s_or_b64 exec, exec, s[12:13]
	s_lshl_b64 s[8:9], s[10:11], 12
	v_lshl_add_u64 v[10:11], v[8:9], 0, s[8:9]
	s_mov_b32 s8, 0
	v_mov_b32_e32 v12, 33
	v_mov_b32_e32 v13, v19
	v_mov_b32_e32 v14, v19
	v_mov_b32_e32 v15, v19
	v_readfirstlane_b32 s12, v10
	v_readfirstlane_b32 s13, v11
	s_mov_b32 s10, s8
	s_mov_b32 s11, s8
	s_mov_b32 s9, s8
	s_nop 1
	global_store_dwordx4 v18, v[12:15], s[12:13]
	s_nop 1
	v_mov_b64_e32 v[14:15], s[10:11]
	v_mov_b64_e32 v[12:13], s[8:9]
	global_store_dwordx4 v18, v[12:15], s[12:13] offset:16
	global_store_dwordx4 v18, v[12:15], s[12:13] offset:32
	global_store_dwordx4 v18, v[12:15], s[12:13] offset:48
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_92
	v_mov_b32_e32 v1, 0
	global_load_dwordx2 v[22:23], v1, s[2:3] offset:32 sc0 sc1
	global_load_dwordx2 v[8:9], v1, s[2:3] offset:40
	v_mov_b32_e32 v20, s6
	v_mov_b32_e32 v21, s7
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v8
	v_readfirstlane_b32 s11, v9
	s_and_b64 s[10:11], s[10:11], s[6:7]
	s_mul_i32 s11, s11, 24
	s_mul_hi_u32 s12, s10, 24
	s_mul_i32 s10, s10, 24
	s_add_i32 s11, s12, s11
	v_lshl_add_u64 v[12:13], v[6:7], 0, s[10:11]
	global_store_dwordx2 v[12:13], v[22:23], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[8:9], v1, v[20:23], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[8:9], v[22:23]
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_88
	s_mov_b64 s[12:13], 0
.LBB0_87:
	s_sleep 1
	global_store_dwordx2 v[12:13], v[8:9], off
	v_mov_b32_e32 v6, s6
	v_mov_b32_e32 v7, s7
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[6:7], v1, v[6:9], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[6:7], v[8:9]
	s_or_b64 s[12:13], vcc, s[12:13]
	v_mov_b64_e32 v[8:9], v[6:7]
	s_andn2_b64 exec, exec, s[12:13]
	s_cbranch_execnz .LBB0_87
.LBB0_88:
	s_or_b64 exec, exec, s[10:11]
	v_mov_b32_e32 v9, 0
	global_load_dwordx2 v[6:7], v9, s[2:3] offset:16
	s_mov_b64 s[10:11], exec
	v_mbcnt_lo_u32_b32 v1, s10, 0
	v_mbcnt_hi_u32_b32 v1, s11, v1
	v_cmp_eq_u32_e32 vcc, 0, v1
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB0_90
	s_bcnt1_i32_b64 s10, s[10:11]
	v_mov_b32_e32 v8, s10
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[6:7], v[8:9], off offset:8 sc1
.LBB0_90:
	s_or_b64 exec, exec, s[12:13]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[8:9], v[6:7], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[8:9]
	s_cbranch_vccnz .LBB0_92
	global_load_dword v6, v[6:7], off offset:24
	v_mov_b32_e32 v7, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v6
	s_and_b32 m0, s10, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[8:9], v[6:7], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_92:
	s_or_b64 exec, exec, s[8:9]
	v_lshl_add_u64 v[6:7], v[10:11], 0, v[18:19]
	s_branch .LBB0_96
.LBB0_93:
	s_or_b64 exec, exec, s[8:9]
	v_readfirstlane_b32 s8, v1
	s_cmp_eq_u32 s8, 0
	s_cbranch_scc1 .LBB0_95
	s_sleep 1
	s_cbranch_execnz .LBB0_96
	s_branch .LBB0_98
.LBB0_95:
	s_branch .LBB0_98
.LBB0_96:
	v_mov_b32_e32 v1, 1
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_93
	global_load_dword v1, v[2:3], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v1, 1, v1
	s_branch .LBB0_93
.LBB0_98:
	global_load_dwordx2 v[6:7], v[6:7], off
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_101
	v_mov_b32_e32 v1, 0
	global_load_dwordx2 v[2:3], v1, s[2:3] offset:40
	global_load_dwordx2 v[12:13], v1, s[2:3] offset:24 sc0 sc1
	global_load_dwordx2 v[8:9], v1, s[2:3]
	s_waitcnt vmcnt(2)
	v_readfirstlane_b32 s10, v2
	v_readfirstlane_b32 s11, v3
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
	v_lshl_add_u64 v[2:3], v[8:9], 0, s[6:7]
	v_mov_b32_e32 v10, s0
	global_store_dwordx2 v[2:3], v[12:13], off
	v_mov_b32_e32 v11, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v1, v[10:13], s[2:3] offset:24 sc0 sc1
	s_mov_b64 s[6:7], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[10:11], v[12:13]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_101
.LBB0_100:
	s_sleep 1
	global_store_dwordx2 v[2:3], v[10:11], off
	v_mov_b32_e32 v8, s0
	v_mov_b32_e32 v9, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[8:9], v1, v[8:11], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[8:9], v[10:11]
	s_or_b64 s[6:7], vcc, s[6:7]
	v_mov_b64_e32 v[10:11], v[8:9]
	s_andn2_b64 exec, exec, s[6:7]
	s_cbranch_execnz .LBB0_100
.LBB0_101:
	s_or_b64 exec, exec, s[8:9]
	v_readfirstlane_b32 s0, v16
	v_mov_b64_e32 v[14:15], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v16
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_107
	v_mov_b32_e32 v1, 0
	global_load_dwordx2 v[10:11], v1, s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[2:3], v1, s[2:3] offset:40
	global_load_dwordx2 v[8:9], v1, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v10
	v_and_b32_e32 v3, v3, v11
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v5, v2, 24
	v_add_u32_e32 v3, v5, v3
	v_mul_lo_u32 v2, v2, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[8:9], 0, v[2:3]
	global_load_dwordx2 v[8:9], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[14:15], v1, v[8:11], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[14:15], v[10:11]
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_106
	s_mov_b64 s[10:11], 0
.LBB0_104:
	s_sleep 1
	global_load_dwordx2 v[2:3], v1, s[2:3] offset:40
	global_load_dwordx2 v[8:9], v1, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v14
	v_and_b32_e32 v5, v3, v15
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[2:3], s[12:13], v2, 24, v[8:9]
	v_mov_b32_e32 v8, v3
	v_mad_u64_u32 v[8:9], s[12:13], v5, 24, v[8:9]
	v_mov_b32_e32 v3, v8
	global_load_dwordx2 v[12:13], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v1, v[12:15], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[2:3], v[14:15]
	s_or_b64 s[10:11], vcc, s[10:11]
	v_mov_b64_e32 v[14:15], v[2:3]
	s_andn2_b64 exec, exec, s[10:11]
	s_cbranch_execnz .LBB0_104
	s_or_b64 exec, exec, s[10:11]
	v_mov_b64_e32 v[14:15], v[2:3]
.LBB0_106:
	s_or_b64 exec, exec, s[8:9]
.LBB0_107:
	s_or_b64 exec, exec, s[6:7]
	v_mov_b32_e32 v19, 0
	global_load_dwordx2 v[2:3], v19, s[2:3] offset:40
	global_load_dwordx4 v[10:13], v19, s[2:3]
	v_readfirstlane_b32 s7, v15
	v_readfirstlane_b32 s6, v14
	s_mov_b64 s[8:9], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s10, v2
	v_readfirstlane_b32 s11, v3
	s_and_b64 s[10:11], s[10:11], s[6:7]
	s_mul_i32 s12, s11, 24
	s_mul_hi_u32 s13, s10, 24
	s_add_i32 s13, s13, s12
	s_mul_i32 s12, s10, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[10:11], 0, s[12:13]
	s_and_saveexec_b64 s[12:13], s[0:1]
	s_cbranch_execz .LBB0_109
	v_mov_b64_e32 v[20:21], s[8:9]
	v_mov_b32_e32 v22, 2
	v_mov_b32_e32 v23, 1
	global_store_dwordx4 v[2:3], v[20:23], off offset:8
.LBB0_109:
	s_or_b64 exec, exec, s[12:13]
	s_lshl_b64 s[8:9], s[10:11], 12
	v_lshl_add_u64 v[12:13], v[12:13], 0, s[8:9]
	v_and_b32_e32 v1, 0xffffff1f, v6
	v_or_b32_e32 v6, 0x80, v1
	v_mov_b32_e32 v9, 0x205d343c
	v_mov_b32_e32 v8, 0x6469745b
	v_readfirstlane_b32 s12, v12
	v_readfirstlane_b32 s13, v13
	s_mov_b32 s8, 0
	s_mov_b32 s9, s8
	s_mov_b32 s10, s8
	s_mov_b32 s11, s8
	s_nop 0
	global_store_dwordx4 v18, v[6:9], s[12:13]
	s_nop 1
	v_mov_b32_e32 v6, 0x3d646974
	v_mov_b32_e32 v7, 0x202c6425
	v_mov_b32_e32 v8, 0x656e616c
	v_mov_b32_e32 v9, 0xa64253d
	global_store_dwordx4 v18, v[6:9], s[12:13] offset:16
	s_nop 1
	v_mov_b64_e32 v[6:7], s[8:9]
	v_mov_b64_e32 v[8:9], s[10:11]
	global_store_dwordx4 v18, v[6:9], s[12:13] offset:32
	global_store_dwordx4 v18, v[6:9], s[12:13] offset:48
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_117
	v_mov_b32_e32 v1, 0
	global_load_dwordx2 v[22:23], v1, s[2:3] offset:32 sc0 sc1
	global_load_dwordx2 v[6:7], v1, s[2:3] offset:40
	v_mov_b32_e32 v20, s6
	v_mov_b32_e32 v21, s7
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v6
	v_readfirstlane_b32 s11, v7
	s_and_b64 s[10:11], s[10:11], s[6:7]
	s_mul_i32 s11, s11, 24
	s_mul_hi_u32 s12, s10, 24
	s_mul_i32 s10, s10, 24
	s_add_i32 s11, s12, s11
	v_lshl_add_u64 v[10:11], v[10:11], 0, s[10:11]
	global_store_dwordx2 v[10:11], v[22:23], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[8:9], v1, v[20:23], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[8:9], v[22:23]
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_113
	s_mov_b64 s[12:13], 0
.LBB0_112:
	s_sleep 1
	global_store_dwordx2 v[10:11], v[8:9], off
	v_mov_b32_e32 v6, s6
	v_mov_b32_e32 v7, s7
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[6:7], v1, v[6:9], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[6:7], v[8:9]
	s_or_b64 s[12:13], vcc, s[12:13]
	v_mov_b64_e32 v[8:9], v[6:7]
	s_andn2_b64 exec, exec, s[12:13]
	s_cbranch_execnz .LBB0_112
.LBB0_113:
	s_or_b64 exec, exec, s[10:11]
	v_mov_b32_e32 v9, 0
	global_load_dwordx2 v[6:7], v9, s[2:3] offset:16
	s_mov_b64 s[10:11], exec
	v_mbcnt_lo_u32_b32 v1, s10, 0
	v_mbcnt_hi_u32_b32 v1, s11, v1
	v_cmp_eq_u32_e32 vcc, 0, v1
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB0_115
	s_bcnt1_i32_b64 s10, s[10:11]
	v_mov_b32_e32 v8, s10
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[6:7], v[8:9], off offset:8 sc1
.LBB0_115:
	s_or_b64 exec, exec, s[12:13]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[8:9], v[6:7], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[8:9]
	s_cbranch_vccnz .LBB0_117
	global_load_dword v6, v[6:7], off offset:24
	v_mov_b32_e32 v7, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v6
	s_and_b32 m0, s10, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[8:9], v[6:7], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_117:
	s_or_b64 exec, exec, s[8:9]
	v_lshl_add_u64 v[6:7], v[12:13], 0, v[18:19]
	s_branch .LBB0_121
.LBB0_118:
	s_or_b64 exec, exec, s[8:9]
	v_readfirstlane_b32 s8, v1
	s_cmp_eq_u32 s8, 0
	s_cbranch_scc1 .LBB0_120
	s_sleep 1
	s_cbranch_execnz .LBB0_121
	s_branch .LBB0_123
.LBB0_120:
	s_branch .LBB0_123
.LBB0_121:
	v_mov_b32_e32 v1, 1
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_118
	global_load_dword v1, v[2:3], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v1, 1, v1
	s_branch .LBB0_118
.LBB0_123:
	global_load_dwordx2 v[2:3], v[6:7], off
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_126
	v_mov_b32_e32 v1, 0
	global_load_dwordx2 v[6:7], v1, s[2:3] offset:40
	global_load_dwordx2 v[14:15], v1, s[2:3] offset:24 sc0 sc1
	global_load_dwordx2 v[8:9], v1, s[2:3]
	s_waitcnt vmcnt(2)
	v_readfirstlane_b32 s10, v6
	v_readfirstlane_b32 s11, v7
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
	v_lshl_add_u64 v[10:11], v[8:9], 0, s[6:7]
	v_mov_b32_e32 v12, s0
	global_store_dwordx2 v[10:11], v[14:15], off
	v_mov_b32_e32 v13, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[8:9], v1, v[12:15], s[2:3] offset:24 sc0 sc1
	s_mov_b64 s[6:7], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[8:9], v[14:15]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_126
.LBB0_125:
	s_sleep 1
	global_store_dwordx2 v[10:11], v[8:9], off
	v_mov_b32_e32 v6, s0
	v_mov_b32_e32 v7, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[6:7], v1, v[6:9], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[6:7], v[8:9]
	s_or_b64 s[6:7], vcc, s[6:7]
	v_mov_b64_e32 v[8:9], v[6:7]
	s_andn2_b64 exec, exec, s[6:7]
	s_cbranch_execnz .LBB0_125
.LBB0_126:
	s_or_b64 exec, exec, s[8:9]
	v_readfirstlane_b32 s0, v16
	v_mov_b64_e32 v[10:11], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v16
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_132
	v_mov_b32_e32 v1, 0
	global_load_dwordx2 v[8:9], v1, s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[6:7], v1, s[2:3] offset:40
	global_load_dwordx2 v[10:11], v1, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v5, v6, v8
	v_and_b32_e32 v6, v7, v9
	v_mul_lo_u32 v6, v6, 24
	v_mul_hi_u32 v7, v5, 24
	v_add_u32_e32 v7, v7, v6
	v_mul_lo_u32 v6, v5, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[6:7], v[10:11], 0, v[6:7]
	global_load_dwordx2 v[6:7], v[6:7], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v1, v[6:9], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[10:11], v[8:9]
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_131
	s_mov_b64 s[10:11], 0
.LBB0_129:
	s_sleep 1
	global_load_dwordx2 v[6:7], v1, s[2:3] offset:40
	global_load_dwordx2 v[8:9], v1, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v6, v6, v10
	v_and_b32_e32 v5, v7, v11
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[6:7], s[12:13], v6, 24, v[8:9]
	v_mov_b32_e32 v8, v7
	v_mad_u64_u32 v[8:9], s[12:13], v5, 24, v[8:9]
	v_mov_b32_e32 v7, v8
	global_load_dwordx2 v[8:9], v[6:7], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[6:7], v1, v[8:11], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[6:7], v[10:11]
	s_or_b64 s[10:11], vcc, s[10:11]
	v_mov_b64_e32 v[10:11], v[6:7]
	s_andn2_b64 exec, exec, s[10:11]
	s_cbranch_execnz .LBB0_129
	s_or_b64 exec, exec, s[10:11]
	v_mov_b64_e32 v[10:11], v[6:7]
.LBB0_131:
	s_or_b64 exec, exec, s[8:9]
.LBB0_132:
	s_or_b64 exec, exec, s[6:7]
	v_mov_b32_e32 v5, 0
	global_load_dwordx2 v[12:13], v5, s[2:3] offset:40
	global_load_dwordx4 v[6:9], v5, s[2:3]
	v_readfirstlane_b32 s7, v11
	v_readfirstlane_b32 s6, v10
	s_mov_b64 s[8:9], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s10, v12
	v_readfirstlane_b32 s11, v13
	s_and_b64 s[10:11], s[10:11], s[6:7]
	s_mul_i32 s12, s11, 24
	s_mul_hi_u32 s13, s10, 24
	s_add_i32 s13, s13, s12
	s_mul_i32 s12, s10, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[10:11], v[6:7], 0, s[12:13]
	s_and_saveexec_b64 s[12:13], s[0:1]
	s_cbranch_execz .LBB0_134
	v_mov_b64_e32 v[12:13], s[8:9]
	v_mov_b32_e32 v14, 2
	v_mov_b32_e32 v15, 1
	global_store_dwordx4 v[10:11], v[12:15], off offset:8
.LBB0_134:
	s_or_b64 exec, exec, s[12:13]
	s_lshl_b64 s[8:9], s[10:11], 12
	v_lshl_add_u64 v[8:9], v[8:9], 0, s[8:9]
	v_and_b32_e32 v1, 0xffffff1d, v2
	v_or_b32_e32 v2, 0x42, v1
	v_readfirstlane_b32 s12, v8
	v_readfirstlane_b32 s13, v9
	s_mov_b32 s8, 0
	v_mov_b32_e32 v1, s8
	s_mov_b32 s9, s8
	s_mov_b32 s10, s8
	s_mov_b32 s11, s8
	global_store_dwordx4 v18, v[2:5], s[12:13]
	s_nop 1
	v_mov_b32_e32 v2, s8
	v_mov_b32_e32 v3, s8
	global_store_dwordx4 v18, v[0:3], s[12:13] offset:16
	s_nop 1
	v_mov_b64_e32 v[0:1], s[8:9]
	v_mov_b64_e32 v[2:3], s[10:11]
	global_store_dwordx4 v18, v[0:3], s[12:13] offset:32
	global_store_dwordx4 v18, v[0:3], s[12:13] offset:48
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_142
	v_mov_b32_e32 v8, 0
	global_load_dwordx2 v[14:15], v8, s[2:3] offset:32 sc0 sc1
	global_load_dwordx2 v[0:1], v8, s[2:3] offset:40
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
	global_atomic_cmpswap_x2 v[2:3], v8, v[12:15], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[2:3], v[14:15]
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_138
	s_mov_b64 s[12:13], 0
.LBB0_137:
	s_sleep 1
	global_store_dwordx2 v[4:5], v[2:3], off
	v_mov_b32_e32 v0, s6
	v_mov_b32_e32 v1, s7
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[0:1], v8, v[0:3], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[0:1], v[2:3]
	s_or_b64 s[12:13], vcc, s[12:13]
	v_mov_b64_e32 v[2:3], v[0:1]
	s_andn2_b64 exec, exec, s[12:13]
	s_cbranch_execnz .LBB0_137
.LBB0_138:
	s_or_b64 exec, exec, s[10:11]
	v_mov_b32_e32 v3, 0
	global_load_dwordx2 v[0:1], v3, s[2:3] offset:16
	s_mov_b64 s[10:11], exec
	v_mbcnt_lo_u32_b32 v2, s10, 0
	v_mbcnt_hi_u32_b32 v2, s11, v2
	v_cmp_eq_u32_e32 vcc, 0, v2
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB0_140
	s_bcnt1_i32_b64 s10, s[10:11]
	v_mov_b32_e32 v2, s10
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[0:1], v[2:3], off offset:8 sc1
.LBB0_140:
	s_or_b64 exec, exec, s[12:13]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[2:3], v[0:1], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[2:3]
	s_cbranch_vccnz .LBB0_142
	global_load_dword v0, v[0:1], off offset:24
	v_mov_b32_e32 v1, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v0
	s_and_b32 m0, s10, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[2:3], v[0:1], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_142:
	s_or_b64 exec, exec, s[8:9]
	s_branch .LBB0_146
.LBB0_143:
	s_or_b64 exec, exec, s[8:9]
	v_readfirstlane_b32 s8, v0
	s_cmp_eq_u32 s8, 0
	s_cbranch_scc1 .LBB0_145
	s_sleep 1
	s_cbranch_execnz .LBB0_146
	s_branch .LBB0_148
.LBB0_145:
	s_branch .LBB0_148
.LBB0_146:
	v_mov_b32_e32 v0, 1
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_143
	global_load_dword v0, v[10:11], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v0, 1, v0
	s_branch .LBB0_143
.LBB0_148:
	s_and_b64 exec, exec, s[0:1]
	s_cbranch_execz .LBB0_151
	v_mov_b32_e32 v6, 0
	global_load_dwordx2 v[0:1], v6, s[2:3] offset:40
	global_load_dwordx2 v[10:11], v6, s[2:3] offset:24 sc0 sc1
	global_load_dwordx2 v[2:3], v6, s[2:3]
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
	global_atomic_cmpswap_x2 v[2:3], v6, v[8:11], s[2:3] offset:24 sc0 sc1
	s_mov_b64 s[6:7], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[2:3], v[10:11]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_151
.LBB0_150:
	s_sleep 1
	global_store_dwordx2 v[4:5], v[2:3], off
	v_mov_b32_e32 v0, s0
	v_mov_b32_e32 v1, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[0:1], v6, v[0:3], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[0:1], v[2:3]
	s_or_b64 s[6:7], vcc, s[6:7]
	v_mov_b64_e32 v[2:3], v[0:1]
	s_andn2_b64 exec, exec, s[6:7]
	s_cbranch_execnz .LBB0_150
.LBB0_151:
	s_or_b64 exec, exec, s[4:5]
	;;#ASMSTART
	v_mov_b32 v2, v99
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v0, v100
	;;#ASMEND
	s_nop 0
	v_cmp_eq_u32_e32 vcc, 0, v0
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_227
	v_readfirstlane_b32 s0, v16
	v_mov_b64_e32 v[8:9], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v16
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_158
	v_mov_b32_e32 v3, 0
	global_load_dwordx2 v[6:7], v3, s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[0:1], v3, s[2:3] offset:40
	global_load_dwordx2 v[4:5], v3, s[2:3]
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
	global_atomic_cmpswap_x2 v[8:9], v3, v[4:7], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[8:9], v[6:7]
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_157
	s_mov_b64 s[10:11], 0
.LBB0_155:
	s_sleep 1
	global_load_dwordx2 v[0:1], v3, s[2:3] offset:40
	global_load_dwordx2 v[4:5], v3, s[2:3]
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
	global_atomic_cmpswap_x2 v[0:1], v3, v[6:9], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[0:1], v[8:9]
	s_or_b64 s[10:11], vcc, s[10:11]
	v_mov_b64_e32 v[8:9], v[0:1]
	s_andn2_b64 exec, exec, s[10:11]
	s_cbranch_execnz .LBB0_155
	s_or_b64 exec, exec, s[10:11]
	v_mov_b64_e32 v[8:9], v[0:1]
.LBB0_157:
	s_or_b64 exec, exec, s[8:9]
.LBB0_158:
	s_or_b64 exec, exec, s[6:7]
	v_mov_b32_e32 v19, 0
	global_load_dwordx2 v[0:1], v19, s[2:3] offset:40
	global_load_dwordx4 v[4:7], v19, s[2:3]
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
	s_cbranch_execz .LBB0_160
	v_mov_b64_e32 v[8:9], s[8:9]
	v_mov_b32_e32 v10, 2
	v_mov_b32_e32 v11, 1
	global_store_dwordx4 v[0:1], v[8:11], off offset:8
.LBB0_160:
	s_or_b64 exec, exec, s[12:13]
	s_lshl_b64 s[8:9], s[10:11], 12
	v_lshl_add_u64 v[8:9], v[6:7], 0, s[8:9]
	s_mov_b32 s8, 0
	v_mov_b32_e32 v10, 33
	v_mov_b32_e32 v11, v19
	v_mov_b32_e32 v12, v19
	v_mov_b32_e32 v13, v19
	v_readfirstlane_b32 s12, v8
	v_readfirstlane_b32 s13, v9
	s_mov_b32 s10, s8
	s_mov_b32 s11, s8
	s_mov_b32 s9, s8
	s_nop 1
	global_store_dwordx4 v18, v[10:13], s[12:13]
	s_nop 1
	v_mov_b64_e32 v[12:13], s[10:11]
	v_mov_b64_e32 v[10:11], s[8:9]
	global_store_dwordx4 v18, v[10:13], s[12:13] offset:16
	global_store_dwordx4 v18, v[10:13], s[12:13] offset:32
	global_store_dwordx4 v18, v[10:13], s[12:13] offset:48
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_168
	v_mov_b32_e32 v3, 0
	global_load_dwordx2 v[14:15], v3, s[2:3] offset:32 sc0 sc1
	global_load_dwordx2 v[6:7], v3, s[2:3] offset:40
	v_mov_b32_e32 v12, s6
	v_mov_b32_e32 v13, s7
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v6
	v_readfirstlane_b32 s11, v7
	s_and_b64 s[10:11], s[10:11], s[6:7]
	s_mul_i32 s11, s11, 24
	s_mul_hi_u32 s12, s10, 24
	s_mul_i32 s10, s10, 24
	s_add_i32 s11, s12, s11
	v_lshl_add_u64 v[10:11], v[4:5], 0, s[10:11]
	global_store_dwordx2 v[10:11], v[14:15], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[6:7], v3, v[12:15], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[6:7], v[14:15]
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_164
	s_mov_b64 s[12:13], 0
.LBB0_163:
	s_sleep 1
	global_store_dwordx2 v[10:11], v[6:7], off
	v_mov_b32_e32 v4, s6
	v_mov_b32_e32 v5, s7
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[4:5], v3, v[4:7], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[4:5], v[6:7]
	s_or_b64 s[12:13], vcc, s[12:13]
	v_mov_b64_e32 v[6:7], v[4:5]
	s_andn2_b64 exec, exec, s[12:13]
	s_cbranch_execnz .LBB0_163
.LBB0_164:
	s_or_b64 exec, exec, s[10:11]
	v_mov_b32_e32 v7, 0
	global_load_dwordx2 v[4:5], v7, s[2:3] offset:16
	s_mov_b64 s[10:11], exec
	v_mbcnt_lo_u32_b32 v3, s10, 0
	v_mbcnt_hi_u32_b32 v3, s11, v3
	v_cmp_eq_u32_e32 vcc, 0, v3
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB0_166
	s_bcnt1_i32_b64 s10, s[10:11]
	v_mov_b32_e32 v6, s10
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[4:5], v[6:7], off offset:8 sc1
.LBB0_166:
	s_or_b64 exec, exec, s[12:13]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[6:7], v[4:5], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[6:7]
	s_cbranch_vccnz .LBB0_168
	global_load_dword v4, v[4:5], off offset:24
	v_mov_b32_e32 v5, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v4
	s_and_b32 m0, s10, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[6:7], v[4:5], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_168:
	s_or_b64 exec, exec, s[8:9]
	v_lshl_add_u64 v[4:5], v[8:9], 0, v[18:19]
	s_branch .LBB0_172
.LBB0_169:
	s_or_b64 exec, exec, s[8:9]
	v_readfirstlane_b32 s8, v3
	s_cmp_eq_u32 s8, 0
	s_cbranch_scc1 .LBB0_171
	s_sleep 1
	s_cbranch_execnz .LBB0_172
	s_branch .LBB0_174
.LBB0_171:
	s_branch .LBB0_174
.LBB0_172:
	v_mov_b32_e32 v3, 1
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_169
	global_load_dword v3, v[0:1], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v3, 1, v3
	s_branch .LBB0_169
.LBB0_174:
	global_load_dwordx2 v[4:5], v[4:5], off
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_177
	v_mov_b32_e32 v3, 0
	global_load_dwordx2 v[0:1], v3, s[2:3] offset:40
	global_load_dwordx2 v[10:11], v3, s[2:3] offset:24 sc0 sc1
	global_load_dwordx2 v[6:7], v3, s[2:3]
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
	global_atomic_cmpswap_x2 v[8:9], v3, v[8:11], s[2:3] offset:24 sc0 sc1
	s_mov_b64 s[6:7], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[8:9], v[10:11]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_177
.LBB0_176:
	s_sleep 1
	global_store_dwordx2 v[0:1], v[8:9], off
	v_mov_b32_e32 v6, s0
	v_mov_b32_e32 v7, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[6:7], v3, v[6:9], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[6:7], v[8:9]
	s_or_b64 s[6:7], vcc, s[6:7]
	v_mov_b64_e32 v[8:9], v[6:7]
	s_andn2_b64 exec, exec, s[6:7]
	s_cbranch_execnz .LBB0_176
.LBB0_177:
	s_or_b64 exec, exec, s[8:9]
	v_readfirstlane_b32 s0, v16
	v_mov_b64_e32 v[12:13], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v16
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_183
	v_mov_b32_e32 v3, 0
	global_load_dwordx2 v[8:9], v3, s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[0:1], v3, s[2:3] offset:40
	global_load_dwordx2 v[6:7], v3, s[2:3]
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
	global_atomic_cmpswap_x2 v[12:13], v3, v[6:9], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[12:13], v[8:9]
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_182
	s_mov_b64 s[10:11], 0
.LBB0_180:
	s_sleep 1
	global_load_dwordx2 v[0:1], v3, s[2:3] offset:40
	global_load_dwordx2 v[6:7], v3, s[2:3]
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
	global_atomic_cmpswap_x2 v[0:1], v3, v[10:13], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[0:1], v[12:13]
	s_or_b64 s[10:11], vcc, s[10:11]
	v_mov_b64_e32 v[12:13], v[0:1]
	s_andn2_b64 exec, exec, s[10:11]
	s_cbranch_execnz .LBB0_180
	s_or_b64 exec, exec, s[10:11]
	v_mov_b64_e32 v[12:13], v[0:1]
.LBB0_182:
	s_or_b64 exec, exec, s[8:9]
.LBB0_183:
	s_or_b64 exec, exec, s[6:7]
	v_mov_b32_e32 v19, 0
	global_load_dwordx2 v[0:1], v19, s[2:3] offset:40
	global_load_dwordx4 v[8:11], v19, s[2:3]
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
	s_cbranch_execz .LBB0_185
	v_mov_b64_e32 v[12:13], s[8:9]
	v_mov_b32_e32 v14, 2
	v_mov_b32_e32 v15, 1
	global_store_dwordx4 v[0:1], v[12:15], off offset:8
.LBB0_185:
	s_or_b64 exec, exec, s[12:13]
	s_lshl_b64 s[8:9], s[10:11], 12
	v_lshl_add_u64 v[10:11], v[10:11], 0, s[8:9]
	v_and_b32_e32 v3, 0xffffff1f, v4
	v_or_b32_e32 v4, 0xa0, v3
	v_mov_b32_e32 v7, 0x303d3d65
	v_mov_b32_e32 v6, 0x6e616c5b
	v_readfirstlane_b32 s8, v10
	v_readfirstlane_b32 s9, v11
	s_nop 4
	global_store_dwordx4 v18, v[4:7], s[8:9]
	s_nop 1
	v_mov_b32_e32 v4, 0x6974205d
	v_mov_b32_e32 v5, 0x64253d64
	v_mov_b32_e32 v6, 0x61772820
	v_mov_b32_e32 v7, 0x72666576
	global_store_dwordx4 v18, v[4:7], s[8:9] offset:16
	s_nop 1
	v_mov_b32_e32 v4, 0x20746e6f
	v_mov_b32_e32 v5, 0x6461656c
	v_mov_b32_e32 v6, 0xa297265
	v_mov_b32_e32 v7, v19
	global_store_dwordx4 v18, v[4:7], s[8:9] offset:32
	s_nop 1
	v_mov_b32_e32 v4, v19
	v_mov_b32_e32 v5, v19
	v_mov_b32_e32 v6, v19
	global_store_dwordx4 v18, v[4:7], s[8:9] offset:48
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_193
	v_mov_b32_e32 v3, 0
	global_load_dwordx2 v[14:15], v3, s[2:3] offset:32 sc0 sc1
	global_load_dwordx2 v[4:5], v3, s[2:3] offset:40
	v_mov_b32_e32 v12, s6
	v_mov_b32_e32 v13, s7
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v4
	v_readfirstlane_b32 s11, v5
	s_and_b64 s[10:11], s[10:11], s[6:7]
	s_mul_i32 s11, s11, 24
	s_mul_hi_u32 s12, s10, 24
	s_mul_i32 s10, s10, 24
	s_add_i32 s11, s12, s11
	v_lshl_add_u64 v[8:9], v[8:9], 0, s[10:11]
	global_store_dwordx2 v[8:9], v[14:15], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[6:7], v3, v[12:15], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[6:7], v[14:15]
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_189
	s_mov_b64 s[12:13], 0
.LBB0_188:
	s_sleep 1
	global_store_dwordx2 v[8:9], v[6:7], off
	v_mov_b32_e32 v4, s6
	v_mov_b32_e32 v5, s7
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[4:5], v3, v[4:7], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[4:5], v[6:7]
	s_or_b64 s[12:13], vcc, s[12:13]
	v_mov_b64_e32 v[6:7], v[4:5]
	s_andn2_b64 exec, exec, s[12:13]
	s_cbranch_execnz .LBB0_188
.LBB0_189:
	s_or_b64 exec, exec, s[10:11]
	v_mov_b32_e32 v7, 0
	global_load_dwordx2 v[4:5], v7, s[2:3] offset:16
	s_mov_b64 s[10:11], exec
	v_mbcnt_lo_u32_b32 v3, s10, 0
	v_mbcnt_hi_u32_b32 v3, s11, v3
	v_cmp_eq_u32_e32 vcc, 0, v3
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB0_191
	s_bcnt1_i32_b64 s10, s[10:11]
	v_mov_b32_e32 v6, s10
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[4:5], v[6:7], off offset:8 sc1
.LBB0_191:
	s_or_b64 exec, exec, s[12:13]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[6:7], v[4:5], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[6:7]
	s_cbranch_vccnz .LBB0_193
	global_load_dword v4, v[4:5], off offset:24
	v_mov_b32_e32 v5, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v4
	s_and_b32 m0, s10, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[6:7], v[4:5], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_193:
	s_or_b64 exec, exec, s[8:9]
	v_lshl_add_u64 v[4:5], v[10:11], 0, v[18:19]
	s_branch .LBB0_197
.LBB0_194:
	s_or_b64 exec, exec, s[8:9]
	v_readfirstlane_b32 s8, v3
	s_cmp_eq_u32 s8, 0
	s_cbranch_scc1 .LBB0_196
	s_sleep 1
	s_cbranch_execnz .LBB0_197
	s_branch .LBB0_199
.LBB0_196:
	s_branch .LBB0_199
.LBB0_197:
	v_mov_b32_e32 v3, 1
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_194
	global_load_dword v3, v[0:1], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v3, 1, v3
	s_branch .LBB0_194
.LBB0_199:
	global_load_dwordx2 v[0:1], v[4:5], off
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_202
	v_mov_b32_e32 v3, 0
	global_load_dwordx2 v[4:5], v3, s[2:3] offset:40
	global_load_dwordx2 v[12:13], v3, s[2:3] offset:24 sc0 sc1
	global_load_dwordx2 v[6:7], v3, s[2:3]
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
	global_atomic_cmpswap_x2 v[6:7], v3, v[10:13], s[2:3] offset:24 sc0 sc1
	s_mov_b64 s[6:7], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[6:7], v[12:13]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_202
.LBB0_201:
	s_sleep 1
	global_store_dwordx2 v[8:9], v[6:7], off
	v_mov_b32_e32 v4, s0
	v_mov_b32_e32 v5, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[4:5], v3, v[4:7], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[4:5], v[6:7]
	s_or_b64 s[6:7], vcc, s[6:7]
	v_mov_b64_e32 v[6:7], v[4:5]
	s_andn2_b64 exec, exec, s[6:7]
	s_cbranch_execnz .LBB0_201
.LBB0_202:
	s_or_b64 exec, exec, s[8:9]
	v_readfirstlane_b32 s0, v16
	v_mov_b64_e32 v[8:9], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v16
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_208
	v_mov_b32_e32 v3, 0
	global_load_dwordx2 v[6:7], v3, s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[4:5], v3, s[2:3] offset:40
	global_load_dwordx2 v[8:9], v3, s[2:3]
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
	global_atomic_cmpswap_x2 v[8:9], v3, v[4:7], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[8:9], v[6:7]
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_207
	s_mov_b64 s[10:11], 0
.LBB0_205:
	s_sleep 1
	global_load_dwordx2 v[4:5], v3, s[2:3] offset:40
	global_load_dwordx2 v[6:7], v3, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v4, v4, v8
	v_and_b32_e32 v10, v5, v9
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[4:5], s[12:13], v4, 24, v[6:7]
	v_mov_b32_e32 v6, v5
	v_mad_u64_u32 v[6:7], s[12:13], v10, 24, v[6:7]
	v_mov_b32_e32 v5, v6
	global_load_dwordx2 v[6:7], v[4:5], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[4:5], v3, v[6:9], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[4:5], v[8:9]
	s_or_b64 s[10:11], vcc, s[10:11]
	v_mov_b64_e32 v[8:9], v[4:5]
	s_andn2_b64 exec, exec, s[10:11]
	s_cbranch_execnz .LBB0_205
	s_or_b64 exec, exec, s[10:11]
	v_mov_b64_e32 v[8:9], v[4:5]
.LBB0_207:
	s_or_b64 exec, exec, s[8:9]
.LBB0_208:
	s_or_b64 exec, exec, s[6:7]
	v_mov_b32_e32 v3, 0
	global_load_dwordx2 v[10:11], v3, s[2:3] offset:40
	global_load_dwordx4 v[4:7], v3, s[2:3]
	v_readfirstlane_b32 s7, v9
	v_readfirstlane_b32 s6, v8
	s_mov_b64 s[8:9], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s10, v10
	v_readfirstlane_b32 s11, v11
	s_and_b64 s[10:11], s[10:11], s[6:7]
	s_mul_i32 s12, s11, 24
	s_mul_hi_u32 s13, s10, 24
	s_add_i32 s13, s13, s12
	s_mul_i32 s12, s10, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[8:9], v[4:5], 0, s[12:13]
	s_and_saveexec_b64 s[12:13], s[0:1]
	s_cbranch_execz .LBB0_210
	v_mov_b64_e32 v[10:11], s[8:9]
	v_mov_b32_e32 v12, 2
	v_mov_b32_e32 v13, 1
	global_store_dwordx4 v[8:9], v[10:13], off offset:8
.LBB0_210:
	s_or_b64 exec, exec, s[12:13]
	s_lshl_b64 s[8:9], s[10:11], 12
	v_lshl_add_u64 v[6:7], v[6:7], 0, s[8:9]
	s_movk_i32 s8, 0xff1d
	v_and_or_b32 v0, v0, s8, 34
	s_mov_b32 s8, 0
	v_readfirstlane_b32 s12, v6
	v_readfirstlane_b32 s13, v7
	s_mov_b32 s9, s8
	s_mov_b32 s10, s8
	s_mov_b32 s11, s8
	s_nop 1
	global_store_dwordx4 v18, v[0:3], s[12:13]
	s_nop 1
	v_mov_b64_e32 v[0:1], s[8:9]
	v_mov_b64_e32 v[2:3], s[10:11]
	global_store_dwordx4 v18, v[0:3], s[12:13] offset:16
	global_store_dwordx4 v18, v[0:3], s[12:13] offset:32
	global_store_dwordx4 v18, v[0:3], s[12:13] offset:48
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_218
	v_mov_b32_e32 v6, 0
	global_load_dwordx2 v[12:13], v6, s[2:3] offset:32 sc0 sc1
	global_load_dwordx2 v[0:1], v6, s[2:3] offset:40
	v_mov_b32_e32 v10, s6
	v_mov_b32_e32 v11, s7
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v0
	v_readfirstlane_b32 s11, v1
	s_and_b64 s[10:11], s[10:11], s[6:7]
	s_mul_i32 s11, s11, 24
	s_mul_hi_u32 s12, s10, 24
	s_mul_i32 s10, s10, 24
	s_add_i32 s11, s12, s11
	v_lshl_add_u64 v[4:5], v[4:5], 0, s[10:11]
	global_store_dwordx2 v[4:5], v[12:13], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v6, v[10:13], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[2:3], v[12:13]
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_214
	s_mov_b64 s[12:13], 0
.LBB0_213:
	s_sleep 1
	global_store_dwordx2 v[4:5], v[2:3], off
	v_mov_b32_e32 v0, s6
	v_mov_b32_e32 v1, s7
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[0:1], v6, v[0:3], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[0:1], v[2:3]
	s_or_b64 s[12:13], vcc, s[12:13]
	v_mov_b64_e32 v[2:3], v[0:1]
	s_andn2_b64 exec, exec, s[12:13]
	s_cbranch_execnz .LBB0_213
.LBB0_214:
	s_or_b64 exec, exec, s[10:11]
	v_mov_b32_e32 v3, 0
	global_load_dwordx2 v[0:1], v3, s[2:3] offset:16
	s_mov_b64 s[10:11], exec
	v_mbcnt_lo_u32_b32 v2, s10, 0
	v_mbcnt_hi_u32_b32 v2, s11, v2
	v_cmp_eq_u32_e32 vcc, 0, v2
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB0_216
	s_bcnt1_i32_b64 s10, s[10:11]
	v_mov_b32_e32 v2, s10
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[0:1], v[2:3], off offset:8 sc1
.LBB0_216:
	s_or_b64 exec, exec, s[12:13]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[2:3], v[0:1], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[2:3]
	s_cbranch_vccnz .LBB0_218
	global_load_dword v0, v[0:1], off offset:24
	v_mov_b32_e32 v1, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v0
	s_and_b32 m0, s10, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[2:3], v[0:1], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_218:
	s_or_b64 exec, exec, s[8:9]
	s_branch .LBB0_222
.LBB0_219:
	s_or_b64 exec, exec, s[8:9]
	v_readfirstlane_b32 s8, v0
	s_cmp_eq_u32 s8, 0
	s_cbranch_scc1 .LBB0_221
	s_sleep 1
	s_cbranch_execnz .LBB0_222
	s_branch .LBB0_224
.LBB0_221:
	s_branch .LBB0_224
.LBB0_222:
	v_mov_b32_e32 v0, 1
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_219
	global_load_dword v0, v[8:9], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v0, 1, v0
	s_branch .LBB0_219
.LBB0_224:
	s_and_b64 exec, exec, s[0:1]
	s_cbranch_execz .LBB0_227
	v_mov_b32_e32 v6, 0
	global_load_dwordx2 v[0:1], v6, s[2:3] offset:40
	global_load_dwordx2 v[10:11], v6, s[2:3] offset:24 sc0 sc1
	global_load_dwordx2 v[2:3], v6, s[2:3]
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
	global_atomic_cmpswap_x2 v[2:3], v6, v[8:11], s[2:3] offset:24 sc0 sc1
	s_mov_b64 s[6:7], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[2:3], v[10:11]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_227
.LBB0_226:
	s_sleep 1
	global_store_dwordx2 v[4:5], v[2:3], off
	v_mov_b32_e32 v0, s0
	v_mov_b32_e32 v1, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[0:1], v6, v[0:3], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[0:1], v[2:3]
	s_or_b64 s[6:7], vcc, s[6:7]
	v_mov_b64_e32 v[2:3], v[0:1]
	s_andn2_b64 exec, exec, s[6:7]
	s_cbranch_execnz .LBB0_226
.LBB0_227:
	s_or_b64 exec, exec, s[4:5]
	v_readfirstlane_b32 s0, v16
	v_mov_b64_e32 v[4:5], 0
	;;#ASMSTART
	v_mov_b32 v10, v88
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v11, v89
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v12, v90
	;;#ASMEND
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v16
	s_and_saveexec_b64 s[4:5], s[0:1]
	s_cbranch_execz .LBB0_233
	v_mov_b32_e32 v6, 0
	global_load_dwordx2 v[2:3], v6, s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[0:1], v6, s[2:3] offset:40
	global_load_dwordx2 v[4:5], v6, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v0, v0, v2
	v_and_b32_e32 v1, v1, v3
	v_mul_lo_u32 v1, v1, 24
	v_mul_hi_u32 v7, v0, 24
	v_add_u32_e32 v1, v7, v1
	v_mul_lo_u32 v0, v0, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[0:1], v[4:5], 0, v[0:1]
	global_load_dwordx2 v[0:1], v[0:1], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[4:5], v6, v[0:3], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[4:5], v[2:3]
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_232
	s_mov_b64 s[8:9], 0
.LBB0_230:
	s_sleep 1
	global_load_dwordx2 v[0:1], v6, s[2:3] offset:40
	global_load_dwordx2 v[2:3], v6, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v0, v0, v4
	v_and_b32_e32 v7, v1, v5
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[0:1], s[10:11], v0, 24, v[2:3]
	v_mov_b32_e32 v2, v1
	v_mad_u64_u32 v[2:3], s[10:11], v7, 24, v[2:3]
	v_mov_b32_e32 v1, v2
	global_load_dwordx2 v[2:3], v[0:1], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[0:1], v6, v[2:5], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[0:1], v[4:5]
	s_or_b64 s[8:9], vcc, s[8:9]
	v_mov_b64_e32 v[4:5], v[0:1]
	s_andn2_b64 exec, exec, s[8:9]
	s_cbranch_execnz .LBB0_230
	s_or_b64 exec, exec, s[8:9]
	v_mov_b64_e32 v[4:5], v[0:1]
.LBB0_232:
	s_or_b64 exec, exec, s[6:7]
.LBB0_233:
	s_or_b64 exec, exec, s[4:5]
	v_mov_b32_e32 v19, 0
	global_load_dwordx2 v[6:7], v19, s[2:3] offset:40
	global_load_dwordx4 v[0:3], v19, s[2:3]
	v_readfirstlane_b32 s5, v5
	v_readfirstlane_b32 s4, v4
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s8, v6
	v_readfirstlane_b32 s9, v7
	s_and_b64 s[8:9], s[8:9], s[4:5]
	s_mul_i32 s10, s9, 24
	s_mul_hi_u32 s11, s8, 24
	s_add_i32 s11, s11, s10
	s_mul_i32 s10, s8, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[4:5], v[0:1], 0, s[10:11]
	s_and_saveexec_b64 s[10:11], s[0:1]
	s_cbranch_execz .LBB0_235
	v_mov_b64_e32 v[6:7], s[6:7]
	v_mov_b32_e32 v8, 2
	v_mov_b32_e32 v9, 1
	global_store_dwordx4 v[4:5], v[6:9], off offset:8
.LBB0_235:
	s_or_b64 exec, exec, s[10:11]
	s_lshl_b64 s[6:7], s[8:9], 12
	v_lshl_add_u64 v[6:7], v[2:3], 0, s[6:7]
	s_mov_b32 s8, 0
	v_mov_b32_e32 v20, 33
	v_mov_b32_e32 v21, v19
	v_mov_b32_e32 v22, v19
	v_mov_b32_e32 v23, v19
	v_readfirstlane_b32 s6, v6
	v_readfirstlane_b32 s7, v7
	s_mov_b32 s10, s8
	s_mov_b32 s11, s8
	s_mov_b32 s9, s8
	s_nop 1
	global_store_dwordx4 v18, v[20:23], s[6:7]
	s_nop 1
	v_mov_b64_e32 v[22:23], s[10:11]
	v_mov_b64_e32 v[20:21], s[8:9]
	global_store_dwordx4 v18, v[20:23], s[6:7] offset:16
	global_store_dwordx4 v18, v[20:23], s[6:7] offset:32
	global_store_dwordx4 v18, v[20:23], s[6:7] offset:48
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_243
	v_mov_b32_e32 v13, 0
	global_load_dwordx2 v[22:23], v13, s[2:3] offset:32 sc0 sc1
	global_load_dwordx2 v[2:3], v13, s[2:3] offset:40
	v_mov_b32_e32 v20, s4
	v_mov_b32_e32 v21, s5
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s8, v2
	v_readfirstlane_b32 s9, v3
	s_and_b64 s[8:9], s[8:9], s[4:5]
	s_mul_i32 s9, s9, 24
	s_mul_hi_u32 s10, s8, 24
	s_mul_i32 s8, s8, 24
	s_add_i32 s9, s10, s9
	v_lshl_add_u64 v[8:9], v[0:1], 0, s[8:9]
	global_store_dwordx2 v[8:9], v[22:23], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v13, v[20:23], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[2:3], v[22:23]
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_239
	s_mov_b64 s[10:11], 0
.LBB0_238:
	s_sleep 1
	global_store_dwordx2 v[8:9], v[2:3], off
	v_mov_b32_e32 v0, s4
	v_mov_b32_e32 v1, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[0:1], v13, v[0:3], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[0:1], v[2:3]
	s_or_b64 s[10:11], vcc, s[10:11]
	v_mov_b64_e32 v[2:3], v[0:1]
	s_andn2_b64 exec, exec, s[10:11]
	s_cbranch_execnz .LBB0_238
.LBB0_239:
	s_or_b64 exec, exec, s[8:9]
	v_mov_b32_e32 v3, 0
	global_load_dwordx2 v[0:1], v3, s[2:3] offset:16
	s_mov_b64 s[8:9], exec
	v_mbcnt_lo_u32_b32 v2, s8, 0
	v_mbcnt_hi_u32_b32 v2, s9, v2
	v_cmp_eq_u32_e32 vcc, 0, v2
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_241
	s_bcnt1_i32_b64 s8, s[8:9]
	v_mov_b32_e32 v2, s8
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[0:1], v[2:3], off offset:8 sc1
.LBB0_241:
	s_or_b64 exec, exec, s[10:11]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[2:3], v[0:1], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[2:3]
	s_cbranch_vccnz .LBB0_243
	global_load_dword v0, v[0:1], off offset:24
	v_mov_b32_e32 v1, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s8, v0
	s_and_b32 m0, s8, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[2:3], v[0:1], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_243:
	s_or_b64 exec, exec, s[6:7]
	v_lshl_add_u64 v[0:1], v[6:7], 0, v[18:19]
	s_branch .LBB0_247
.LBB0_244:
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s6, v2
	s_cmp_eq_u32 s6, 0
	s_cbranch_scc1 .LBB0_246
	s_sleep 1
	s_cbranch_execnz .LBB0_247
	s_branch .LBB0_249
.LBB0_246:
	s_branch .LBB0_249
.LBB0_247:
	v_mov_b32_e32 v2, 1
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_244
	global_load_dword v2, v[4:5], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v2, 1, v2
	s_branch .LBB0_244
.LBB0_249:
	global_load_dwordx2 v[0:1], v[0:1], off
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_252
	v_mov_b32_e32 v8, 0
	global_load_dwordx2 v[2:3], v8, s[2:3] offset:40
	global_load_dwordx2 v[22:23], v8, s[2:3] offset:24 sc0 sc1
	global_load_dwordx2 v[4:5], v8, s[2:3]
	s_waitcnt vmcnt(2)
	v_readfirstlane_b32 s8, v2
	v_readfirstlane_b32 s9, v3
	s_add_u32 s10, s8, 1
	s_addc_u32 s11, s9, 0
	s_add_u32 s0, s10, s4
	s_addc_u32 s1, s11, s5
	s_cmp_eq_u64 s[0:1], 0
	s_cselect_b32 s1, s11, s1
	s_cselect_b32 s0, s10, s0
	s_and_b64 s[4:5], s[0:1], s[8:9]
	s_mul_i32 s5, s5, 24
	s_mul_hi_u32 s8, s4, 24
	s_mul_i32 s4, s4, 24
	s_add_i32 s5, s8, s5
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[6:7], v[4:5], 0, s[4:5]
	v_mov_b32_e32 v20, s0
	global_store_dwordx2 v[6:7], v[22:23], off
	v_mov_b32_e32 v21, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[4:5], v8, v[20:23], s[2:3] offset:24 sc0 sc1
	s_mov_b64 s[4:5], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[4:5], v[22:23]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_252
.LBB0_251:
	s_sleep 1
	global_store_dwordx2 v[6:7], v[4:5], off
	v_mov_b32_e32 v2, s0
	v_mov_b32_e32 v3, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v8, v[2:5], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[2:3], v[4:5]
	s_or_b64 s[4:5], vcc, s[4:5]
	v_mov_b64_e32 v[4:5], v[2:3]
	s_andn2_b64 exec, exec, s[4:5]
	s_cbranch_execnz .LBB0_251
.LBB0_252:
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s0, v16
	v_mov_b64_e32 v[8:9], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v16
	s_and_saveexec_b64 s[4:5], s[0:1]
	s_cbranch_execz .LBB0_258
	v_mov_b32_e32 v4, 0
	global_load_dwordx2 v[22:23], v4, s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[2:3], v4, s[2:3] offset:40
	global_load_dwordx2 v[6:7], v4, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v22
	v_and_b32_e32 v3, v3, v23
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v5, v2, 24
	v_add_u32_e32 v3, v5, v3
	v_mul_lo_u32 v2, v2, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[6:7], 0, v[2:3]
	global_load_dwordx2 v[20:21], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[8:9], v4, v[20:23], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[8:9], v[22:23]
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_257
	s_mov_b64 s[8:9], 0
.LBB0_255:
	s_sleep 1
	global_load_dwordx2 v[2:3], v4, s[2:3] offset:40
	global_load_dwordx2 v[6:7], v4, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v8
	v_and_b32_e32 v5, v3, v9
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[2:3], s[10:11], v2, 24, v[6:7]
	v_mov_b32_e32 v6, v3
	v_mad_u64_u32 v[6:7], s[10:11], v5, 24, v[6:7]
	v_mov_b32_e32 v3, v6
	global_load_dwordx2 v[6:7], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v4, v[6:9], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[2:3], v[8:9]
	s_or_b64 s[8:9], vcc, s[8:9]
	v_mov_b64_e32 v[8:9], v[2:3]
	s_andn2_b64 exec, exec, s[8:9]
	s_cbranch_execnz .LBB0_255
	s_or_b64 exec, exec, s[8:9]
	v_mov_b64_e32 v[8:9], v[2:3]
.LBB0_257:
	s_or_b64 exec, exec, s[6:7]
.LBB0_258:
	s_or_b64 exec, exec, s[4:5]
	v_mov_b32_e32 v19, 0
	global_load_dwordx2 v[2:3], v19, s[2:3] offset:40
	global_load_dwordx4 v[4:7], v19, s[2:3]
	v_readfirstlane_b32 s5, v9
	v_readfirstlane_b32 s4, v8
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s8, v2
	v_readfirstlane_b32 s9, v3
	s_and_b64 s[8:9], s[8:9], s[4:5]
	s_mul_i32 s10, s9, 24
	s_mul_hi_u32 s11, s8, 24
	s_add_i32 s11, s11, s10
	s_mul_i32 s10, s8, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[8:9], v[4:5], 0, s[10:11]
	s_and_saveexec_b64 s[10:11], s[0:1]
	s_cbranch_execz .LBB0_260
	v_mov_b64_e32 v[20:21], s[6:7]
	v_mov_b32_e32 v22, 2
	v_mov_b32_e32 v23, 1
	global_store_dwordx4 v[8:9], v[20:23], off offset:8
.LBB0_260:
	s_or_b64 exec, exec, s[10:11]
	s_lshl_b64 s[6:7], s[8:9], 12
	v_lshl_add_u64 v[6:7], v[6:7], 0, s[6:7]
	v_and_b32_e32 v0, 0xffffff1f, v0
	s_mov_b32 s6, 0x66332e25
	v_or_b32_e32 v0, 0x80, v0
	v_mov_b32_e32 v2, 0x203d2041
	v_mov_b32_e32 v3, s6
	v_readfirstlane_b32 s6, v6
	v_readfirstlane_b32 s7, v7
	s_nop 4
	global_store_dwordx4 v18, v[0:3], s[6:7]
	s_nop 1
	v_mov_b32_e32 v0, 0x2042202c
	v_mov_b32_e32 v1, 0x2e25203d
	v_mov_b32_e32 v2, 0x202c6633
	v_mov_b32_e32 v3, 0x203d2043
	global_store_dwordx4 v18, v[0:3], s[6:7] offset:16
	s_nop 1
	v_mov_b32_e32 v0, 0x66332e25
	v_mov_b32_e32 v1, 10
	v_mov_b32_e32 v2, v19
	v_mov_b32_e32 v3, v19
	global_store_dwordx4 v18, v[0:3], s[6:7] offset:32
	s_nop 1
	v_mov_b32_e32 v0, v19
	v_mov_b32_e32 v1, v19
	global_store_dwordx4 v18, v[0:3], s[6:7] offset:48
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_268
	v_mov_b32_e32 v13, 0
	global_load_dwordx2 v[22:23], v13, s[2:3] offset:32 sc0 sc1
	global_load_dwordx2 v[0:1], v13, s[2:3] offset:40
	v_mov_b32_e32 v20, s4
	v_mov_b32_e32 v21, s5
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s8, v0
	v_readfirstlane_b32 s9, v1
	s_and_b64 s[8:9], s[8:9], s[4:5]
	s_mul_i32 s9, s9, 24
	s_mul_hi_u32 s10, s8, 24
	s_mul_i32 s8, s8, 24
	s_add_i32 s9, s10, s9
	v_lshl_add_u64 v[4:5], v[4:5], 0, s[8:9]
	global_store_dwordx2 v[4:5], v[22:23], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v13, v[20:23], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[2:3], v[22:23]
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_264
	s_mov_b64 s[10:11], 0
.LBB0_263:
	s_sleep 1
	global_store_dwordx2 v[4:5], v[2:3], off
	v_mov_b32_e32 v0, s4
	v_mov_b32_e32 v1, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[0:1], v13, v[0:3], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[0:1], v[2:3]
	s_or_b64 s[10:11], vcc, s[10:11]
	v_mov_b64_e32 v[2:3], v[0:1]
	s_andn2_b64 exec, exec, s[10:11]
	s_cbranch_execnz .LBB0_263
.LBB0_264:
	s_or_b64 exec, exec, s[8:9]
	v_mov_b32_e32 v3, 0
	global_load_dwordx2 v[0:1], v3, s[2:3] offset:16
	s_mov_b64 s[8:9], exec
	v_mbcnt_lo_u32_b32 v2, s8, 0
	v_mbcnt_hi_u32_b32 v2, s9, v2
	v_cmp_eq_u32_e32 vcc, 0, v2
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_266
	s_bcnt1_i32_b64 s8, s[8:9]
	v_mov_b32_e32 v2, s8
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[0:1], v[2:3], off offset:8 sc1
.LBB0_266:
	s_or_b64 exec, exec, s[10:11]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[2:3], v[0:1], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[2:3]
	s_cbranch_vccnz .LBB0_268
	global_load_dword v0, v[0:1], off offset:24
	v_mov_b32_e32 v1, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s8, v0
	s_and_b32 m0, s8, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[2:3], v[0:1], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_268:
	s_or_b64 exec, exec, s[6:7]
	v_lshl_add_u64 v[0:1], v[6:7], 0, v[18:19]
	s_branch .LBB0_272
.LBB0_269:
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s6, v2
	s_cmp_eq_u32 s6, 0
	s_cbranch_scc1 .LBB0_271
	s_sleep 1
	s_cbranch_execnz .LBB0_272
	s_branch .LBB0_274
.LBB0_271:
	s_branch .LBB0_274
.LBB0_272:
	v_mov_b32_e32 v2, 1
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_269
	global_load_dword v2, v[8:9], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v2, 1, v2
	s_branch .LBB0_269
.LBB0_274:
	global_load_dwordx2 v[0:1], v[0:1], off
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_277
	v_mov_b32_e32 v8, 0
	global_load_dwordx2 v[2:3], v8, s[2:3] offset:40
	global_load_dwordx2 v[22:23], v8, s[2:3] offset:24 sc0 sc1
	global_load_dwordx2 v[4:5], v8, s[2:3]
	s_waitcnt vmcnt(2)
	v_readfirstlane_b32 s8, v2
	v_readfirstlane_b32 s9, v3
	s_add_u32 s10, s8, 1
	s_addc_u32 s11, s9, 0
	s_add_u32 s0, s10, s4
	s_addc_u32 s1, s11, s5
	s_cmp_eq_u64 s[0:1], 0
	s_cselect_b32 s1, s11, s1
	s_cselect_b32 s0, s10, s0
	s_and_b64 s[4:5], s[0:1], s[8:9]
	s_mul_i32 s5, s5, 24
	s_mul_hi_u32 s8, s4, 24
	s_mul_i32 s4, s4, 24
	s_add_i32 s5, s8, s5
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[6:7], v[4:5], 0, s[4:5]
	v_mov_b32_e32 v20, s0
	global_store_dwordx2 v[6:7], v[22:23], off
	v_mov_b32_e32 v21, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[4:5], v8, v[20:23], s[2:3] offset:24 sc0 sc1
	s_mov_b64 s[4:5], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[4:5], v[22:23]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_277
.LBB0_276:
	s_sleep 1
	global_store_dwordx2 v[6:7], v[4:5], off
	v_mov_b32_e32 v2, s0
	v_mov_b32_e32 v3, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v8, v[2:5], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[2:3], v[4:5]
	s_or_b64 s[4:5], vcc, s[4:5]
	v_mov_b64_e32 v[4:5], v[2:3]
	s_andn2_b64 exec, exec, s[4:5]
	s_cbranch_execnz .LBB0_276
.LBB0_277:
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s0, v16
	v_mov_b64_e32 v[8:9], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v16
	s_and_saveexec_b64 s[4:5], s[0:1]
	s_cbranch_execz .LBB0_283
	v_mov_b32_e32 v4, 0
	global_load_dwordx2 v[22:23], v4, s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[2:3], v4, s[2:3] offset:40
	global_load_dwordx2 v[6:7], v4, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v22
	v_and_b32_e32 v3, v3, v23
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v5, v2, 24
	v_add_u32_e32 v3, v5, v3
	v_mul_lo_u32 v2, v2, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[6:7], 0, v[2:3]
	global_load_dwordx2 v[20:21], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[8:9], v4, v[20:23], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[8:9], v[22:23]
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_282
	s_mov_b64 s[8:9], 0
.LBB0_280:
	s_sleep 1
	global_load_dwordx2 v[2:3], v4, s[2:3] offset:40
	global_load_dwordx2 v[6:7], v4, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v8
	v_and_b32_e32 v5, v3, v9
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[2:3], s[10:11], v2, 24, v[6:7]
	v_mov_b32_e32 v6, v3
	v_mad_u64_u32 v[6:7], s[10:11], v5, 24, v[6:7]
	v_mov_b32_e32 v3, v6
	global_load_dwordx2 v[6:7], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v4, v[6:9], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[2:3], v[8:9]
	s_or_b64 s[8:9], vcc, s[8:9]
	v_mov_b64_e32 v[8:9], v[2:3]
	s_andn2_b64 exec, exec, s[8:9]
	s_cbranch_execnz .LBB0_280
	s_or_b64 exec, exec, s[8:9]
	v_mov_b64_e32 v[8:9], v[2:3]
.LBB0_282:
	s_or_b64 exec, exec, s[6:7]
.LBB0_283:
	s_or_b64 exec, exec, s[4:5]
	v_mov_b32_e32 v13, 0
	global_load_dwordx2 v[2:3], v13, s[2:3] offset:40
	global_load_dwordx4 v[4:7], v13, s[2:3]
	v_readfirstlane_b32 s5, v9
	v_readfirstlane_b32 s4, v8
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s8, v2
	v_readfirstlane_b32 s9, v3
	s_and_b64 s[8:9], s[8:9], s[4:5]
	s_mul_i32 s10, s9, 24
	s_mul_hi_u32 s11, s8, 24
	s_add_i32 s11, s11, s10
	s_mul_i32 s10, s8, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[8:9], v[4:5], 0, s[10:11]
	s_and_saveexec_b64 s[10:11], s[0:1]
	s_cbranch_execz .LBB0_285
	v_mov_b64_e32 v[20:21], s[6:7]
	v_mov_b32_e32 v22, 2
	v_mov_b32_e32 v23, 1
	global_store_dwordx4 v[8:9], v[20:23], off offset:8
.LBB0_285:
	s_or_b64 exec, exec, s[10:11]
	s_lshl_b64 s[6:7], s[8:9], 12
	v_lshl_add_u64 v[6:7], v[6:7], 0, s[6:7]
	v_and_b32_e32 v0, 0xffffff1d, v0
	s_mov_b32 s8, 0
	v_cvt_f64_f32_e32 v[2:3], v10
	v_or_b32_e32 v0, 0x62, v0
	v_readfirstlane_b32 s6, v6
	v_readfirstlane_b32 s7, v7
	s_mov_b32 s9, s8
	v_cvt_f64_f32_e32 v[10:11], v11
	v_cvt_f64_f32_e32 v[12:13], v12
	s_nop 1
	global_store_dwordx4 v18, v[0:3], s[6:7]
	global_store_dwordx4 v18, v[10:13], s[6:7] offset:16
	s_mov_b32 s10, s8
	s_mov_b32 s11, s8
	v_mov_b64_e32 v[0:1], s[8:9]
	v_mov_b64_e32 v[2:3], s[10:11]
	global_store_dwordx4 v18, v[0:3], s[6:7] offset:32
	global_store_dwordx4 v18, v[0:3], s[6:7] offset:48
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_293
	v_mov_b32_e32 v6, 0
	global_load_dwordx2 v[12:13], v6, s[2:3] offset:32 sc0 sc1
	global_load_dwordx2 v[0:1], v6, s[2:3] offset:40
	v_mov_b32_e32 v10, s4
	v_mov_b32_e32 v11, s5
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s8, v0
	v_readfirstlane_b32 s9, v1
	s_and_b64 s[8:9], s[8:9], s[4:5]
	s_mul_i32 s9, s9, 24
	s_mul_hi_u32 s10, s8, 24
	s_mul_i32 s8, s8, 24
	s_add_i32 s9, s10, s9
	v_lshl_add_u64 v[4:5], v[4:5], 0, s[8:9]
	global_store_dwordx2 v[4:5], v[12:13], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v6, v[10:13], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[2:3], v[12:13]
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_289
	s_mov_b64 s[10:11], 0
.LBB0_288:
	s_sleep 1
	global_store_dwordx2 v[4:5], v[2:3], off
	v_mov_b32_e32 v0, s4
	v_mov_b32_e32 v1, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[0:1], v6, v[0:3], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[0:1], v[2:3]
	s_or_b64 s[10:11], vcc, s[10:11]
	v_mov_b64_e32 v[2:3], v[0:1]
	s_andn2_b64 exec, exec, s[10:11]
	s_cbranch_execnz .LBB0_288
.LBB0_289:
	s_or_b64 exec, exec, s[8:9]
	v_mov_b32_e32 v3, 0
	global_load_dwordx2 v[0:1], v3, s[2:3] offset:16
	s_mov_b64 s[8:9], exec
	v_mbcnt_lo_u32_b32 v2, s8, 0
	v_mbcnt_hi_u32_b32 v2, s9, v2
	v_cmp_eq_u32_e32 vcc, 0, v2
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_291
	s_bcnt1_i32_b64 s8, s[8:9]
	v_mov_b32_e32 v2, s8
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[0:1], v[2:3], off offset:8 sc1
.LBB0_291:
	s_or_b64 exec, exec, s[10:11]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[2:3], v[0:1], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[2:3]
	s_cbranch_vccnz .LBB0_293
	global_load_dword v0, v[0:1], off offset:24
	v_mov_b32_e32 v1, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s8, v0
	s_and_b32 m0, s8, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[2:3], v[0:1], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_293:
	s_or_b64 exec, exec, s[6:7]
	s_branch .LBB0_297
.LBB0_294:
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s6, v0
	s_cmp_eq_u32 s6, 0
	s_cbranch_scc1 .LBB0_296
	s_sleep 1
	s_cbranch_execnz .LBB0_297
	s_branch .LBB0_299
.LBB0_296:
	s_branch .LBB0_299
.LBB0_297:
	v_mov_b32_e32 v0, 1
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_294
	global_load_dword v0, v[8:9], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v0, 1, v0
	s_branch .LBB0_294
.LBB0_299:
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_302
	v_mov_b32_e32 v6, 0
	global_load_dwordx2 v[0:1], v6, s[2:3] offset:40
	global_load_dwordx2 v[10:11], v6, s[2:3] offset:24 sc0 sc1
	global_load_dwordx2 v[2:3], v6, s[2:3]
	s_waitcnt vmcnt(2)
	v_readfirstlane_b32 s8, v0
	v_readfirstlane_b32 s9, v1
	s_add_u32 s10, s8, 1
	s_addc_u32 s11, s9, 0
	s_add_u32 s0, s10, s4
	s_addc_u32 s1, s11, s5
	s_cmp_eq_u64 s[0:1], 0
	s_cselect_b32 s1, s11, s1
	s_cselect_b32 s0, s10, s0
	s_and_b64 s[4:5], s[0:1], s[8:9]
	s_mul_i32 s5, s5, 24
	s_mul_hi_u32 s8, s4, 24
	s_mul_i32 s4, s4, 24
	s_add_i32 s5, s8, s5
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[4:5], v[2:3], 0, s[4:5]
	v_mov_b32_e32 v8, s0
	global_store_dwordx2 v[4:5], v[10:11], off
	v_mov_b32_e32 v9, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v6, v[8:11], s[2:3] offset:24 sc0 sc1
	s_mov_b64 s[4:5], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[2:3], v[10:11]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_302
.LBB0_301:
	s_sleep 1
	global_store_dwordx2 v[4:5], v[2:3], off
	v_mov_b32_e32 v0, s0
	v_mov_b32_e32 v1, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[0:1], v6, v[0:3], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[0:1], v[2:3]
	s_or_b64 s[4:5], vcc, s[4:5]
	v_mov_b64_e32 v[2:3], v[0:1]
	s_andn2_b64 exec, exec, s[4:5]
	s_cbranch_execnz .LBB0_301
.LBB0_302:
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s0, v16
	v_mov_b64_e32 v[4:5], 0
	;;#ASMSTART
	v_mov_b32 v14, v91
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v15, v92
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v10, v91
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v11, v92
	;;#ASMEND
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v16
	;;#ASMSTART
	v_mov_b32 v12, v91
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v13, v92
	;;#ASMEND
	s_and_saveexec_b64 s[4:5], s[0:1]
	s_cbranch_execz .LBB0_308
	v_mov_b32_e32 v6, 0
	global_load_dwordx2 v[2:3], v6, s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[0:1], v6, s[2:3] offset:40
	global_load_dwordx2 v[4:5], v6, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v0, v0, v2
	v_and_b32_e32 v1, v1, v3
	v_mul_lo_u32 v1, v1, 24
	v_mul_hi_u32 v7, v0, 24
	v_add_u32_e32 v1, v7, v1
	v_mul_lo_u32 v0, v0, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[0:1], v[4:5], 0, v[0:1]
	global_load_dwordx2 v[0:1], v[0:1], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[4:5], v6, v[0:3], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[4:5], v[2:3]
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_307
	s_mov_b64 s[8:9], 0
.LBB0_305:
	s_sleep 1
	global_load_dwordx2 v[0:1], v6, s[2:3] offset:40
	global_load_dwordx2 v[2:3], v6, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v0, v0, v4
	v_and_b32_e32 v7, v1, v5
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[0:1], s[10:11], v0, 24, v[2:3]
	v_mov_b32_e32 v2, v1
	v_mad_u64_u32 v[2:3], s[10:11], v7, 24, v[2:3]
	v_mov_b32_e32 v1, v2
	global_load_dwordx2 v[2:3], v[0:1], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[0:1], v6, v[2:5], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[0:1], v[4:5]
	s_or_b64 s[8:9], vcc, s[8:9]
	v_mov_b64_e32 v[4:5], v[0:1]
	s_andn2_b64 exec, exec, s[8:9]
	s_cbranch_execnz .LBB0_305
	s_or_b64 exec, exec, s[8:9]
	v_mov_b64_e32 v[4:5], v[0:1]
.LBB0_307:
	s_or_b64 exec, exec, s[6:7]
.LBB0_308:
	s_or_b64 exec, exec, s[4:5]
	v_mov_b32_e32 v19, 0
	global_load_dwordx2 v[6:7], v19, s[2:3] offset:40
	global_load_dwordx4 v[0:3], v19, s[2:3]
	v_readfirstlane_b32 s5, v5
	v_readfirstlane_b32 s4, v4
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s8, v6
	v_readfirstlane_b32 s9, v7
	s_and_b64 s[8:9], s[8:9], s[4:5]
	s_mul_i32 s10, s9, 24
	s_mul_hi_u32 s11, s8, 24
	s_add_i32 s11, s11, s10
	s_mul_i32 s10, s8, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[4:5], v[0:1], 0, s[10:11]
	s_and_saveexec_b64 s[10:11], s[0:1]
	s_cbranch_execz .LBB0_310
	v_mov_b64_e32 v[6:7], s[6:7]
	v_mov_b32_e32 v8, 2
	v_mov_b32_e32 v9, 1
	global_store_dwordx4 v[4:5], v[6:9], off offset:8
.LBB0_310:
	s_or_b64 exec, exec, s[10:11]
	s_lshl_b64 s[6:7], s[8:9], 12
	v_lshl_add_u64 v[6:7], v[2:3], 0, s[6:7]
	s_mov_b32 s8, 0
	v_mov_b32_e32 v20, 33
	v_mov_b32_e32 v21, v19
	v_mov_b32_e32 v22, v19
	v_mov_b32_e32 v23, v19
	v_readfirstlane_b32 s6, v6
	v_readfirstlane_b32 s7, v7
	s_mov_b32 s10, s8
	s_mov_b32 s11, s8
	s_mov_b32 s9, s8
	s_nop 1
	global_store_dwordx4 v18, v[20:23], s[6:7]
	s_nop 1
	v_mov_b64_e32 v[22:23], s[10:11]
	v_mov_b64_e32 v[20:21], s[8:9]
	global_store_dwordx4 v18, v[20:23], s[6:7] offset:16
	global_store_dwordx4 v18, v[20:23], s[6:7] offset:32
	global_store_dwordx4 v18, v[20:23], s[6:7] offset:48
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_318
	v_mov_b32_e32 v17, 0
	global_load_dwordx2 v[22:23], v17, s[2:3] offset:32 sc0 sc1
	global_load_dwordx2 v[2:3], v17, s[2:3] offset:40
	v_mov_b32_e32 v20, s4
	v_mov_b32_e32 v21, s5
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s8, v2
	v_readfirstlane_b32 s9, v3
	s_and_b64 s[8:9], s[8:9], s[4:5]
	s_mul_i32 s9, s9, 24
	s_mul_hi_u32 s10, s8, 24
	s_mul_i32 s8, s8, 24
	s_add_i32 s9, s10, s9
	v_lshl_add_u64 v[8:9], v[0:1], 0, s[8:9]
	global_store_dwordx2 v[8:9], v[22:23], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v17, v[20:23], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[2:3], v[22:23]
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_314
	s_mov_b64 s[10:11], 0
.LBB0_313:
	s_sleep 1
	global_store_dwordx2 v[8:9], v[2:3], off
	v_mov_b32_e32 v0, s4
	v_mov_b32_e32 v1, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[0:1], v17, v[0:3], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[0:1], v[2:3]
	s_or_b64 s[10:11], vcc, s[10:11]
	v_mov_b64_e32 v[2:3], v[0:1]
	s_andn2_b64 exec, exec, s[10:11]
	s_cbranch_execnz .LBB0_313
.LBB0_314:
	s_or_b64 exec, exec, s[8:9]
	v_mov_b32_e32 v3, 0
	global_load_dwordx2 v[0:1], v3, s[2:3] offset:16
	s_mov_b64 s[8:9], exec
	v_mbcnt_lo_u32_b32 v2, s8, 0
	v_mbcnt_hi_u32_b32 v2, s9, v2
	v_cmp_eq_u32_e32 vcc, 0, v2
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_316
	s_bcnt1_i32_b64 s8, s[8:9]
	v_mov_b32_e32 v2, s8
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[0:1], v[2:3], off offset:8 sc1
.LBB0_316:
	s_or_b64 exec, exec, s[10:11]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[2:3], v[0:1], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[2:3]
	s_cbranch_vccnz .LBB0_318
	global_load_dword v0, v[0:1], off offset:24
	v_mov_b32_e32 v1, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s8, v0
	s_and_b32 m0, s8, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[2:3], v[0:1], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_318:
	s_or_b64 exec, exec, s[6:7]
	v_lshl_add_u64 v[0:1], v[6:7], 0, v[18:19]
	s_branch .LBB0_322
.LBB0_319:
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s6, v2
	s_cmp_eq_u32 s6, 0
	s_cbranch_scc1 .LBB0_321
	s_sleep 1
	s_cbranch_execnz .LBB0_322
	s_branch .LBB0_324
.LBB0_321:
	s_branch .LBB0_324
.LBB0_322:
	v_mov_b32_e32 v2, 1
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_319
	global_load_dword v2, v[4:5], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v2, 1, v2
	s_branch .LBB0_319
.LBB0_324:
	global_load_dwordx2 v[0:1], v[0:1], off
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_327
	v_mov_b32_e32 v8, 0
	global_load_dwordx2 v[2:3], v8, s[2:3] offset:40
	global_load_dwordx2 v[22:23], v8, s[2:3] offset:24 sc0 sc1
	global_load_dwordx2 v[4:5], v8, s[2:3]
	s_waitcnt vmcnt(2)
	v_readfirstlane_b32 s8, v2
	v_readfirstlane_b32 s9, v3
	s_add_u32 s10, s8, 1
	s_addc_u32 s11, s9, 0
	s_add_u32 s0, s10, s4
	s_addc_u32 s1, s11, s5
	s_cmp_eq_u64 s[0:1], 0
	s_cselect_b32 s1, s11, s1
	s_cselect_b32 s0, s10, s0
	s_and_b64 s[4:5], s[0:1], s[8:9]
	s_mul_i32 s5, s5, 24
	s_mul_hi_u32 s8, s4, 24
	s_mul_i32 s4, s4, 24
	s_add_i32 s5, s8, s5
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[6:7], v[4:5], 0, s[4:5]
	v_mov_b32_e32 v20, s0
	global_store_dwordx2 v[6:7], v[22:23], off
	v_mov_b32_e32 v21, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[4:5], v8, v[20:23], s[2:3] offset:24 sc0 sc1
	s_mov_b64 s[4:5], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[4:5], v[22:23]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_327
.LBB0_326:
	s_sleep 1
	global_store_dwordx2 v[6:7], v[4:5], off
	v_mov_b32_e32 v2, s0
	v_mov_b32_e32 v3, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v8, v[2:5], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[2:3], v[4:5]
	s_or_b64 s[4:5], vcc, s[4:5]
	v_mov_b64_e32 v[4:5], v[2:3]
	s_andn2_b64 exec, exec, s[4:5]
	s_cbranch_execnz .LBB0_326
.LBB0_327:
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s0, v16
	v_mov_b64_e32 v[8:9], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v16
	s_and_saveexec_b64 s[4:5], s[0:1]
	s_cbranch_execz .LBB0_333
	v_mov_b32_e32 v4, 0
	global_load_dwordx2 v[22:23], v4, s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[2:3], v4, s[2:3] offset:40
	global_load_dwordx2 v[6:7], v4, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v22
	v_and_b32_e32 v3, v3, v23
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v5, v2, 24
	v_add_u32_e32 v3, v5, v3
	v_mul_lo_u32 v2, v2, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[6:7], 0, v[2:3]
	global_load_dwordx2 v[20:21], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[8:9], v4, v[20:23], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[8:9], v[22:23]
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_332
	s_mov_b64 s[8:9], 0
.LBB0_330:
	s_sleep 1
	global_load_dwordx2 v[2:3], v4, s[2:3] offset:40
	global_load_dwordx2 v[6:7], v4, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v8
	v_and_b32_e32 v5, v3, v9
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[2:3], s[10:11], v2, 24, v[6:7]
	v_mov_b32_e32 v6, v3
	v_mad_u64_u32 v[6:7], s[10:11], v5, 24, v[6:7]
	v_mov_b32_e32 v3, v6
	global_load_dwordx2 v[6:7], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v4, v[6:9], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[2:3], v[8:9]
	s_or_b64 s[8:9], vcc, s[8:9]
	v_mov_b64_e32 v[8:9], v[2:3]
	s_andn2_b64 exec, exec, s[8:9]
	s_cbranch_execnz .LBB0_330
	s_or_b64 exec, exec, s[8:9]
	v_mov_b64_e32 v[8:9], v[2:3]
.LBB0_332:
	s_or_b64 exec, exec, s[6:7]
.LBB0_333:
	s_or_b64 exec, exec, s[4:5]
	v_mov_b32_e32 v19, 0
	global_load_dwordx2 v[2:3], v19, s[2:3] offset:40
	global_load_dwordx4 v[4:7], v19, s[2:3]
	v_readfirstlane_b32 s5, v9
	v_readfirstlane_b32 s4, v8
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s8, v2
	v_readfirstlane_b32 s9, v3
	s_and_b64 s[8:9], s[8:9], s[4:5]
	s_mul_i32 s10, s9, 24
	s_mul_hi_u32 s11, s8, 24
	s_add_i32 s11, s11, s10
	s_mul_i32 s10, s8, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[8:9], v[4:5], 0, s[10:11]
	s_and_saveexec_b64 s[10:11], s[0:1]
	s_cbranch_execz .LBB0_335
	v_mov_b64_e32 v[20:21], s[6:7]
	v_mov_b32_e32 v22, 2
	v_mov_b32_e32 v23, 1
	global_store_dwordx4 v[8:9], v[20:23], off offset:8
.LBB0_335:
	s_or_b64 exec, exec, s[10:11]
	s_lshl_b64 s[6:7], s[8:9], 12
	v_lshl_add_u64 v[6:7], v[6:7], 0, s[6:7]
	v_and_b32_e32 v0, 0xffffff1f, v0
	v_or_b32_e32 v0, 0xc0, v0
	v_mov_b32_e32 v3, 0x28205d72
	v_mov_b32_e32 v2, 0x7078455b
	v_readfirstlane_b32 s6, v6
	v_readfirstlane_b32 s7, v7
	v_mov_b32_e32 v20, 0x253d422d
	v_mov_b32_e32 v21, 0x2c66322e
	v_mov_b32_e32 v22, 0x422a4120
	s_nop 1
	global_store_dwordx4 v18, v[0:3], s[6:7]
	s_nop 1
	v_mov_b32_e32 v0, 0x29422b41
	v_mov_b32_e32 v1, 0x372f3278
	v_mov_b32_e32 v2, 0x322e253d
	v_mov_b32_e32 v3, 0x41202c66
	global_store_dwordx4 v18, v[0:3], s[6:7] offset:16
	v_mov_b32_e32 v23, v2
	global_store_dwordx4 v18, v[20:23], s[6:7] offset:32
	v_mov_b32_e32 v0, 0xa66
	v_mov_b32_e32 v1, v19
	v_mov_b32_e32 v2, v19
	v_mov_b32_e32 v3, v19
	global_store_dwordx4 v18, v[0:3], s[6:7] offset:48
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_343
	v_mov_b32_e32 v17, 0
	global_load_dwordx2 v[22:23], v17, s[2:3] offset:32 sc0 sc1
	global_load_dwordx2 v[0:1], v17, s[2:3] offset:40
	v_mov_b32_e32 v20, s4
	v_mov_b32_e32 v21, s5
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s8, v0
	v_readfirstlane_b32 s9, v1
	s_and_b64 s[8:9], s[8:9], s[4:5]
	s_mul_i32 s9, s9, 24
	s_mul_hi_u32 s10, s8, 24
	s_mul_i32 s8, s8, 24
	s_add_i32 s9, s10, s9
	v_lshl_add_u64 v[4:5], v[4:5], 0, s[8:9]
	global_store_dwordx2 v[4:5], v[22:23], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v17, v[20:23], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[2:3], v[22:23]
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_339
	s_mov_b64 s[10:11], 0
.LBB0_338:
	s_sleep 1
	global_store_dwordx2 v[4:5], v[2:3], off
	v_mov_b32_e32 v0, s4
	v_mov_b32_e32 v1, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[0:1], v17, v[0:3], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[0:1], v[2:3]
	s_or_b64 s[10:11], vcc, s[10:11]
	v_mov_b64_e32 v[2:3], v[0:1]
	s_andn2_b64 exec, exec, s[10:11]
	s_cbranch_execnz .LBB0_338
.LBB0_339:
	s_or_b64 exec, exec, s[8:9]
	v_mov_b32_e32 v3, 0
	global_load_dwordx2 v[0:1], v3, s[2:3] offset:16
	s_mov_b64 s[8:9], exec
	v_mbcnt_lo_u32_b32 v2, s8, 0
	v_mbcnt_hi_u32_b32 v2, s9, v2
	v_cmp_eq_u32_e32 vcc, 0, v2
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_341
	s_bcnt1_i32_b64 s8, s[8:9]
	v_mov_b32_e32 v2, s8
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[0:1], v[2:3], off offset:8 sc1
.LBB0_341:
	s_or_b64 exec, exec, s[10:11]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[2:3], v[0:1], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[2:3]
	s_cbranch_vccnz .LBB0_343
	global_load_dword v0, v[0:1], off offset:24
	v_mov_b32_e32 v1, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s8, v0
	s_and_b32 m0, s8, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[2:3], v[0:1], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_343:
	s_or_b64 exec, exec, s[6:7]
	v_lshl_add_u64 v[0:1], v[6:7], 0, v[18:19]
	s_branch .LBB0_347
.LBB0_344:
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s6, v2
	s_cmp_eq_u32 s6, 0
	s_cbranch_scc1 .LBB0_346
	s_sleep 1
	s_cbranch_execnz .LBB0_347
	s_branch .LBB0_349
.LBB0_346:
	s_branch .LBB0_349
.LBB0_347:
	v_mov_b32_e32 v2, 1
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_344
	global_load_dword v2, v[8:9], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v2, 1, v2
	s_branch .LBB0_344
.LBB0_349:
	global_load_dwordx2 v[0:1], v[0:1], off
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_352
	v_mov_b32_e32 v8, 0
	global_load_dwordx2 v[2:3], v8, s[2:3] offset:40
	global_load_dwordx2 v[22:23], v8, s[2:3] offset:24 sc0 sc1
	global_load_dwordx2 v[4:5], v8, s[2:3]
	s_waitcnt vmcnt(2)
	v_readfirstlane_b32 s8, v2
	v_readfirstlane_b32 s9, v3
	s_add_u32 s10, s8, 1
	s_addc_u32 s11, s9, 0
	s_add_u32 s0, s10, s4
	s_addc_u32 s1, s11, s5
	s_cmp_eq_u64 s[0:1], 0
	s_cselect_b32 s1, s11, s1
	s_cselect_b32 s0, s10, s0
	s_and_b64 s[4:5], s[0:1], s[8:9]
	s_mul_i32 s5, s5, 24
	s_mul_hi_u32 s8, s4, 24
	s_mul_i32 s4, s4, 24
	s_add_i32 s5, s8, s5
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[6:7], v[4:5], 0, s[4:5]
	v_mov_b32_e32 v20, s0
	global_store_dwordx2 v[6:7], v[22:23], off
	v_mov_b32_e32 v21, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[4:5], v8, v[20:23], s[2:3] offset:24 sc0 sc1
	s_mov_b64 s[4:5], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[4:5], v[22:23]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_352
.LBB0_351:
	s_sleep 1
	global_store_dwordx2 v[6:7], v[4:5], off
	v_mov_b32_e32 v2, s0
	v_mov_b32_e32 v3, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v8, v[2:5], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[2:3], v[4:5]
	s_or_b64 s[4:5], vcc, s[4:5]
	v_mov_b64_e32 v[4:5], v[2:3]
	s_andn2_b64 exec, exec, s[4:5]
	s_cbranch_execnz .LBB0_351
.LBB0_352:
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s0, v16
	v_mov_b64_e32 v[8:9], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v16
	s_and_saveexec_b64 s[4:5], s[0:1]
	s_cbranch_execz .LBB0_358
	v_mov_b32_e32 v4, 0
	global_load_dwordx2 v[22:23], v4, s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[2:3], v4, s[2:3] offset:40
	global_load_dwordx2 v[6:7], v4, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v22
	v_and_b32_e32 v3, v3, v23
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v5, v2, 24
	v_add_u32_e32 v3, v5, v3
	v_mul_lo_u32 v2, v2, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[6:7], 0, v[2:3]
	global_load_dwordx2 v[20:21], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[8:9], v4, v[20:23], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[8:9], v[22:23]
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_357
	s_mov_b64 s[8:9], 0
.LBB0_355:
	s_sleep 1
	global_load_dwordx2 v[2:3], v4, s[2:3] offset:40
	global_load_dwordx2 v[6:7], v4, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v8
	v_and_b32_e32 v5, v3, v9
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[2:3], s[10:11], v2, 24, v[6:7]
	v_mov_b32_e32 v6, v3
	v_mad_u64_u32 v[6:7], s[10:11], v5, 24, v[6:7]
	v_mov_b32_e32 v3, v6
	global_load_dwordx2 v[6:7], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v4, v[6:9], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[2:3], v[8:9]
	s_or_b64 s[8:9], vcc, s[8:9]
	v_mov_b64_e32 v[8:9], v[2:3]
	s_andn2_b64 exec, exec, s[8:9]
	s_cbranch_execnz .LBB0_355
	s_or_b64 exec, exec, s[8:9]
	v_mov_b64_e32 v[8:9], v[2:3]
.LBB0_357:
	s_or_b64 exec, exec, s[6:7]
.LBB0_358:
	s_or_b64 exec, exec, s[4:5]
	v_mov_b32_e32 v17, 0
	global_load_dwordx2 v[2:3], v17, s[2:3] offset:40
	global_load_dwordx4 v[4:7], v17, s[2:3]
	v_readfirstlane_b32 s5, v9
	v_readfirstlane_b32 s4, v8
	s_mov_b64 s[8:9], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s6, v2
	v_readfirstlane_b32 s7, v3
	s_and_b64 s[6:7], s[6:7], s[4:5]
	s_mul_i32 s10, s7, 24
	s_mul_hi_u32 s11, s6, 24
	s_add_i32 s11, s11, s10
	s_mul_i32 s10, s6, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[8:9], v[4:5], 0, s[10:11]
	s_and_saveexec_b64 s[10:11], s[0:1]
	s_cbranch_execz .LBB0_360
	v_mov_b64_e32 v[20:21], s[8:9]
	v_mov_b32_e32 v22, 2
	v_mov_b32_e32 v23, 1
	global_store_dwordx4 v[8:9], v[20:23], off offset:8
.LBB0_360:
	s_or_b64 exec, exec, s[10:11]
	v_add_f32_e32 v2, v14, v15
	v_add_f32_e32 v2, v2, v2
	s_mov_b32 s10, 0x40e00000
	v_div_scale_f32 v3, s[8:9], s10, s10, v2
	v_rcp_f32_e32 v14, v3
	s_lshl_b64 s[6:7], s[6:7], 12
	v_lshl_add_u64 v[6:7], v[6:7], 0, s[6:7]
	v_and_b32_e32 v0, 0xffffff1d, v0
	v_fma_f32 v15, -v3, v14, 1.0
	v_fmac_f32_e32 v14, v15, v14
	v_div_scale_f32 v15, vcc, v2, s10, v2
	v_mul_f32_e32 v17, v15, v14
	v_fma_f32 v19, -v3, v17, v15
	v_fmac_f32_e32 v17, v19, v14
	v_fma_f32 v3, -v3, v17, v15
	v_div_fmas_f32 v3, v3, v14, v17
	v_div_fixup_f32 v2, v3, s10, v2
	s_mov_b32 s8, 0
	v_sub_f32_e32 v10, v10, v11
	v_mul_f32_e32 v12, v12, v13
	v_cvt_f64_f32_e32 v[2:3], v2
	v_or_b32_e32 v0, 0x62, v0
	v_readfirstlane_b32 s6, v6
	v_readfirstlane_b32 s7, v7
	s_mov_b32 s9, s8
	v_cvt_f64_f32_e32 v[10:11], v10
	v_cvt_f64_f32_e32 v[12:13], v12
	s_nop 1
	global_store_dwordx4 v18, v[0:3], s[6:7]
	global_store_dwordx4 v18, v[10:13], s[6:7] offset:16
	s_mov_b32 s10, s8
	s_mov_b32 s11, s8
	v_mov_b64_e32 v[0:1], s[8:9]
	v_mov_b64_e32 v[2:3], s[10:11]
	global_store_dwordx4 v18, v[0:3], s[6:7] offset:32
	global_store_dwordx4 v18, v[0:3], s[6:7] offset:48
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_368
	v_mov_b32_e32 v6, 0
	global_load_dwordx2 v[12:13], v6, s[2:3] offset:32 sc0 sc1
	global_load_dwordx2 v[0:1], v6, s[2:3] offset:40
	v_mov_b32_e32 v10, s4
	v_mov_b32_e32 v11, s5
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s8, v0
	v_readfirstlane_b32 s9, v1
	s_and_b64 s[8:9], s[8:9], s[4:5]
	s_mul_i32 s9, s9, 24
	s_mul_hi_u32 s10, s8, 24
	s_mul_i32 s8, s8, 24
	s_add_i32 s9, s10, s9
	v_lshl_add_u64 v[4:5], v[4:5], 0, s[8:9]
	global_store_dwordx2 v[4:5], v[12:13], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v6, v[10:13], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[2:3], v[12:13]
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_364
	s_mov_b64 s[10:11], 0
.LBB0_363:
	s_sleep 1
	global_store_dwordx2 v[4:5], v[2:3], off
	v_mov_b32_e32 v0, s4
	v_mov_b32_e32 v1, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[0:1], v6, v[0:3], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[0:1], v[2:3]
	s_or_b64 s[10:11], vcc, s[10:11]
	v_mov_b64_e32 v[2:3], v[0:1]
	s_andn2_b64 exec, exec, s[10:11]
	s_cbranch_execnz .LBB0_363
.LBB0_364:
	s_or_b64 exec, exec, s[8:9]
	v_mov_b32_e32 v3, 0
	global_load_dwordx2 v[0:1], v3, s[2:3] offset:16
	s_mov_b64 s[8:9], exec
	v_mbcnt_lo_u32_b32 v2, s8, 0
	v_mbcnt_hi_u32_b32 v2, s9, v2
	v_cmp_eq_u32_e32 vcc, 0, v2
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_366
	s_bcnt1_i32_b64 s8, s[8:9]
	v_mov_b32_e32 v2, s8
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[0:1], v[2:3], off offset:8 sc1
.LBB0_366:
	s_or_b64 exec, exec, s[10:11]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[2:3], v[0:1], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[2:3]
	s_cbranch_vccnz .LBB0_368
	global_load_dword v0, v[0:1], off offset:24
	v_mov_b32_e32 v1, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s8, v0
	s_and_b32 m0, s8, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[2:3], v[0:1], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_368:
	s_or_b64 exec, exec, s[6:7]
	s_branch .LBB0_372
.LBB0_369:
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s6, v0
	s_cmp_eq_u32 s6, 0
	s_cbranch_scc1 .LBB0_371
	s_sleep 1
	s_cbranch_execnz .LBB0_372
	s_branch .LBB0_374
.LBB0_371:
	s_branch .LBB0_374
.LBB0_372:
	v_mov_b32_e32 v0, 1
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_369
	global_load_dword v0, v[8:9], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v0, 1, v0
	s_branch .LBB0_369
.LBB0_374:
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_377
	v_mov_b32_e32 v6, 0
	global_load_dwordx2 v[0:1], v6, s[2:3] offset:40
	global_load_dwordx2 v[10:11], v6, s[2:3] offset:24 sc0 sc1
	global_load_dwordx2 v[2:3], v6, s[2:3]
	s_waitcnt vmcnt(2)
	v_readfirstlane_b32 s8, v0
	v_readfirstlane_b32 s9, v1
	s_add_u32 s10, s8, 1
	s_addc_u32 s11, s9, 0
	s_add_u32 s0, s10, s4
	s_addc_u32 s1, s11, s5
	s_cmp_eq_u64 s[0:1], 0
	s_cselect_b32 s1, s11, s1
	s_cselect_b32 s0, s10, s0
	s_and_b64 s[4:5], s[0:1], s[8:9]
	s_mul_i32 s5, s5, 24
	s_mul_hi_u32 s8, s4, 24
	s_mul_i32 s4, s4, 24
	s_add_i32 s5, s8, s5
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[4:5], v[2:3], 0, s[4:5]
	v_mov_b32_e32 v8, s0
	global_store_dwordx2 v[4:5], v[10:11], off
	v_mov_b32_e32 v9, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v6, v[8:11], s[2:3] offset:24 sc0 sc1
	s_mov_b64 s[4:5], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[2:3], v[10:11]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_377
.LBB0_376:
	s_sleep 1
	global_store_dwordx2 v[4:5], v[2:3], off
	v_mov_b32_e32 v0, s0
	v_mov_b32_e32 v1, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[0:1], v6, v[0:3], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[0:1], v[2:3]
	s_or_b64 s[4:5], vcc, s[4:5]
	v_mov_b64_e32 v[2:3], v[0:1]
	s_andn2_b64 exec, exec, s[4:5]
	s_cbranch_execnz .LBB0_376
.LBB0_377:
	s_or_b64 exec, exec, s[6:7]
	;;#ASMSTART
	v_mov_b32 v10, v93
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v11, v94
	;;#ASMEND
	s_nop 0
	v_cmp_ge_f32_e32 vcc, 2.0, v10
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_453
	v_readfirstlane_b32 s0, v16
	v_mov_b64_e32 v[4:5], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v16
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_384
	v_mov_b32_e32 v6, 0
	global_load_dwordx2 v[2:3], v6, s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[0:1], v6, s[2:3] offset:40
	global_load_dwordx2 v[4:5], v6, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v0, v0, v2
	v_and_b32_e32 v1, v1, v3
	v_mul_lo_u32 v1, v1, 24
	v_mul_hi_u32 v7, v0, 24
	v_add_u32_e32 v1, v7, v1
	v_mul_lo_u32 v0, v0, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[0:1], v[4:5], 0, v[0:1]
	global_load_dwordx2 v[0:1], v[0:1], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[4:5], v6, v[0:3], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[4:5], v[2:3]
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_383
	s_mov_b64 s[10:11], 0
.LBB0_381:
	s_sleep 1
	global_load_dwordx2 v[0:1], v6, s[2:3] offset:40
	global_load_dwordx2 v[2:3], v6, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v0, v0, v4
	v_and_b32_e32 v7, v1, v5
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[0:1], s[12:13], v0, 24, v[2:3]
	v_mov_b32_e32 v2, v1
	v_mad_u64_u32 v[2:3], s[12:13], v7, 24, v[2:3]
	v_mov_b32_e32 v1, v2
	global_load_dwordx2 v[2:3], v[0:1], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[0:1], v6, v[2:5], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[0:1], v[4:5]
	s_or_b64 s[10:11], vcc, s[10:11]
	v_mov_b64_e32 v[4:5], v[0:1]
	s_andn2_b64 exec, exec, s[10:11]
	s_cbranch_execnz .LBB0_381
	s_or_b64 exec, exec, s[10:11]
	v_mov_b64_e32 v[4:5], v[0:1]
.LBB0_383:
	s_or_b64 exec, exec, s[8:9]
.LBB0_384:
	s_or_b64 exec, exec, s[6:7]
	v_mov_b32_e32 v19, 0
	global_load_dwordx2 v[6:7], v19, s[2:3] offset:40
	global_load_dwordx4 v[0:3], v19, s[2:3]
	v_readfirstlane_b32 s7, v5
	v_readfirstlane_b32 s6, v4
	s_mov_b64 s[8:9], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s10, v6
	v_readfirstlane_b32 s11, v7
	s_and_b64 s[10:11], s[10:11], s[6:7]
	s_mul_i32 s12, s11, 24
	s_mul_hi_u32 s13, s10, 24
	s_add_i32 s13, s13, s12
	s_mul_i32 s12, s10, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[4:5], v[0:1], 0, s[12:13]
	s_and_saveexec_b64 s[12:13], s[0:1]
	s_cbranch_execz .LBB0_386
	v_mov_b64_e32 v[6:7], s[8:9]
	v_mov_b32_e32 v8, 2
	v_mov_b32_e32 v9, 1
	global_store_dwordx4 v[4:5], v[6:9], off offset:8
.LBB0_386:
	s_or_b64 exec, exec, s[12:13]
	s_lshl_b64 s[8:9], s[10:11], 12
	v_lshl_add_u64 v[6:7], v[2:3], 0, s[8:9]
	s_mov_b32 s8, 0
	v_mov_b32_e32 v12, 33
	v_mov_b32_e32 v13, v19
	v_mov_b32_e32 v14, v19
	v_mov_b32_e32 v15, v19
	v_readfirstlane_b32 s12, v6
	v_readfirstlane_b32 s13, v7
	s_mov_b32 s10, s8
	s_mov_b32 s11, s8
	s_mov_b32 s9, s8
	s_nop 1
	global_store_dwordx4 v18, v[12:15], s[12:13]
	s_nop 1
	v_mov_b64_e32 v[14:15], s[10:11]
	v_mov_b64_e32 v[12:13], s[8:9]
	global_store_dwordx4 v18, v[12:15], s[12:13] offset:16
	global_store_dwordx4 v18, v[12:15], s[12:13] offset:32
	global_store_dwordx4 v18, v[12:15], s[12:13] offset:48
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_394
	v_mov_b32_e32 v12, 0
	global_load_dwordx2 v[22:23], v12, s[2:3] offset:32 sc0 sc1
	global_load_dwordx2 v[2:3], v12, s[2:3] offset:40
	v_mov_b32_e32 v20, s6
	v_mov_b32_e32 v21, s7
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v2
	v_readfirstlane_b32 s11, v3
	s_and_b64 s[10:11], s[10:11], s[6:7]
	s_mul_i32 s11, s11, 24
	s_mul_hi_u32 s12, s10, 24
	s_mul_i32 s10, s10, 24
	s_add_i32 s11, s12, s11
	v_lshl_add_u64 v[8:9], v[0:1], 0, s[10:11]
	global_store_dwordx2 v[8:9], v[22:23], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v12, v[20:23], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[2:3], v[22:23]
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_390
	s_mov_b64 s[12:13], 0
.LBB0_389:
	s_sleep 1
	global_store_dwordx2 v[8:9], v[2:3], off
	v_mov_b32_e32 v0, s6
	v_mov_b32_e32 v1, s7
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[0:1], v12, v[0:3], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[0:1], v[2:3]
	s_or_b64 s[12:13], vcc, s[12:13]
	v_mov_b64_e32 v[2:3], v[0:1]
	s_andn2_b64 exec, exec, s[12:13]
	s_cbranch_execnz .LBB0_389
.LBB0_390:
	s_or_b64 exec, exec, s[10:11]
	v_mov_b32_e32 v3, 0
	global_load_dwordx2 v[0:1], v3, s[2:3] offset:16
	s_mov_b64 s[10:11], exec
	v_mbcnt_lo_u32_b32 v2, s10, 0
	v_mbcnt_hi_u32_b32 v2, s11, v2
	v_cmp_eq_u32_e32 vcc, 0, v2
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB0_392
	s_bcnt1_i32_b64 s10, s[10:11]
	v_mov_b32_e32 v2, s10
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[0:1], v[2:3], off offset:8 sc1
.LBB0_392:
	s_or_b64 exec, exec, s[12:13]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[2:3], v[0:1], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[2:3]
	s_cbranch_vccnz .LBB0_394
	global_load_dword v0, v[0:1], off offset:24
	v_mov_b32_e32 v1, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v0
	s_and_b32 m0, s10, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[2:3], v[0:1], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_394:
	s_or_b64 exec, exec, s[8:9]
	v_lshl_add_u64 v[0:1], v[6:7], 0, v[18:19]
	s_branch .LBB0_398
.LBB0_395:
	s_or_b64 exec, exec, s[8:9]
	v_readfirstlane_b32 s8, v2
	s_cmp_eq_u32 s8, 0
	s_cbranch_scc1 .LBB0_397
	s_sleep 1
	s_cbranch_execnz .LBB0_398
	s_branch .LBB0_400
.LBB0_397:
	s_branch .LBB0_400
.LBB0_398:
	v_mov_b32_e32 v2, 1
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_395
	global_load_dword v2, v[4:5], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v2, 1, v2
	s_branch .LBB0_395
.LBB0_400:
	global_load_dwordx2 v[0:1], v[0:1], off
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_403
	v_mov_b32_e32 v8, 0
	global_load_dwordx2 v[2:3], v8, s[2:3] offset:40
	global_load_dwordx2 v[14:15], v8, s[2:3] offset:24 sc0 sc1
	global_load_dwordx2 v[4:5], v8, s[2:3]
	s_waitcnt vmcnt(2)
	v_readfirstlane_b32 s10, v2
	v_readfirstlane_b32 s11, v3
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
	v_lshl_add_u64 v[6:7], v[4:5], 0, s[6:7]
	v_mov_b32_e32 v12, s0
	global_store_dwordx2 v[6:7], v[14:15], off
	v_mov_b32_e32 v13, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[4:5], v8, v[12:15], s[2:3] offset:24 sc0 sc1
	s_mov_b64 s[6:7], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[4:5], v[14:15]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_403
.LBB0_402:
	s_sleep 1
	global_store_dwordx2 v[6:7], v[4:5], off
	v_mov_b32_e32 v2, s0
	v_mov_b32_e32 v3, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v8, v[2:5], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[2:3], v[4:5]
	s_or_b64 s[6:7], vcc, s[6:7]
	v_mov_b64_e32 v[4:5], v[2:3]
	s_andn2_b64 exec, exec, s[6:7]
	s_cbranch_execnz .LBB0_402
.LBB0_403:
	s_or_b64 exec, exec, s[8:9]
	v_readfirstlane_b32 s0, v16
	v_mov_b64_e32 v[8:9], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v16
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_409
	v_mov_b32_e32 v4, 0
	global_load_dwordx2 v[14:15], v4, s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[2:3], v4, s[2:3] offset:40
	global_load_dwordx2 v[6:7], v4, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v14
	v_and_b32_e32 v3, v3, v15
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v5, v2, 24
	v_add_u32_e32 v3, v5, v3
	v_mul_lo_u32 v2, v2, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[6:7], 0, v[2:3]
	global_load_dwordx2 v[12:13], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[8:9], v4, v[12:15], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[8:9], v[14:15]
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_408
	s_mov_b64 s[10:11], 0
.LBB0_406:
	s_sleep 1
	global_load_dwordx2 v[2:3], v4, s[2:3] offset:40
	global_load_dwordx2 v[6:7], v4, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v8
	v_and_b32_e32 v5, v3, v9
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[2:3], s[12:13], v2, 24, v[6:7]
	v_mov_b32_e32 v6, v3
	v_mad_u64_u32 v[6:7], s[12:13], v5, 24, v[6:7]
	v_mov_b32_e32 v3, v6
	global_load_dwordx2 v[6:7], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v4, v[6:9], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[2:3], v[8:9]
	s_or_b64 s[10:11], vcc, s[10:11]
	v_mov_b64_e32 v[8:9], v[2:3]
	s_andn2_b64 exec, exec, s[10:11]
	s_cbranch_execnz .LBB0_406
	s_or_b64 exec, exec, s[10:11]
	v_mov_b64_e32 v[8:9], v[2:3]
.LBB0_408:
	s_or_b64 exec, exec, s[8:9]
.LBB0_409:
	s_or_b64 exec, exec, s[6:7]
	v_mov_b32_e32 v19, 0
	global_load_dwordx2 v[2:3], v19, s[2:3] offset:40
	global_load_dwordx4 v[4:7], v19, s[2:3]
	v_readfirstlane_b32 s7, v9
	v_readfirstlane_b32 s6, v8
	s_mov_b64 s[8:9], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s10, v2
	v_readfirstlane_b32 s11, v3
	s_and_b64 s[10:11], s[10:11], s[6:7]
	s_mul_i32 s12, s11, 24
	s_mul_hi_u32 s13, s10, 24
	s_add_i32 s13, s13, s12
	s_mul_i32 s12, s10, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[8:9], v[4:5], 0, s[12:13]
	s_and_saveexec_b64 s[12:13], s[0:1]
	s_cbranch_execz .LBB0_411
	v_mov_b64_e32 v[12:13], s[8:9]
	v_mov_b32_e32 v14, 2
	v_mov_b32_e32 v15, 1
	global_store_dwordx4 v[8:9], v[12:15], off offset:8
.LBB0_411:
	s_or_b64 exec, exec, s[12:13]
	s_lshl_b64 s[8:9], s[10:11], 12
	v_lshl_add_u64 v[6:7], v[6:7], 0, s[8:9]
	v_and_b32_e32 v0, 0xffffff1f, v0
	v_or_b32_e32 v0, 0xc0, v0
	v_mov_b32_e32 v3, 0x41206572
	v_mov_b32_e32 v2, 0x6f666542
	v_readfirstlane_b32 s8, v6
	v_readfirstlane_b32 s9, v7
	s_nop 4
	global_store_dwordx4 v18, v[0:3], s[8:9]
	s_nop 1
	v_mov_b32_e32 v0, 0x28204444
	v_mov_b32_e32 v1, 0x3d3c3676
	v_mov_b32_e32 v2, 0x203a2932
	v_mov_b32_e32 v3, 0x3d203676
	global_store_dwordx4 v18, v[0:3], s[8:9] offset:16
	s_nop 1
	v_mov_b32_e32 v0, 0x332e2520
	v_mov_b32_e32 v1, 0x76202c66
	v_mov_b32_e32 v2, 0x203d2037
	v_mov_b32_e32 v3, 0x66322e25
	global_store_dwordx4 v18, v[0:3], s[8:9] offset:32
	s_nop 1
	v_mov_b32_e32 v0, 10
	v_mov_b32_e32 v1, v19
	v_mov_b32_e32 v2, v19
	v_mov_b32_e32 v3, v19
	global_store_dwordx4 v18, v[0:3], s[8:9] offset:48
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_419
	v_mov_b32_e32 v12, 0
	global_load_dwordx2 v[22:23], v12, s[2:3] offset:32 sc0 sc1
	global_load_dwordx2 v[0:1], v12, s[2:3] offset:40
	v_mov_b32_e32 v20, s6
	v_mov_b32_e32 v21, s7
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v0
	v_readfirstlane_b32 s11, v1
	s_and_b64 s[10:11], s[10:11], s[6:7]
	s_mul_i32 s11, s11, 24
	s_mul_hi_u32 s12, s10, 24
	s_mul_i32 s10, s10, 24
	s_add_i32 s11, s12, s11
	v_lshl_add_u64 v[4:5], v[4:5], 0, s[10:11]
	global_store_dwordx2 v[4:5], v[22:23], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v12, v[20:23], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[2:3], v[22:23]
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_415
	s_mov_b64 s[12:13], 0
.LBB0_414:
	s_sleep 1
	global_store_dwordx2 v[4:5], v[2:3], off
	v_mov_b32_e32 v0, s6
	v_mov_b32_e32 v1, s7
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[0:1], v12, v[0:3], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[0:1], v[2:3]
	s_or_b64 s[12:13], vcc, s[12:13]
	v_mov_b64_e32 v[2:3], v[0:1]
	s_andn2_b64 exec, exec, s[12:13]
	s_cbranch_execnz .LBB0_414
.LBB0_415:
	s_or_b64 exec, exec, s[10:11]
	v_mov_b32_e32 v3, 0
	global_load_dwordx2 v[0:1], v3, s[2:3] offset:16
	s_mov_b64 s[10:11], exec
	v_mbcnt_lo_u32_b32 v2, s10, 0
	v_mbcnt_hi_u32_b32 v2, s11, v2
	v_cmp_eq_u32_e32 vcc, 0, v2
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB0_417
	s_bcnt1_i32_b64 s10, s[10:11]
	v_mov_b32_e32 v2, s10
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[0:1], v[2:3], off offset:8 sc1
.LBB0_417:
	s_or_b64 exec, exec, s[12:13]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[2:3], v[0:1], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[2:3]
	s_cbranch_vccnz .LBB0_419
	global_load_dword v0, v[0:1], off offset:24
	v_mov_b32_e32 v1, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v0
	s_and_b32 m0, s10, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[2:3], v[0:1], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_419:
	s_or_b64 exec, exec, s[8:9]
	v_lshl_add_u64 v[0:1], v[6:7], 0, v[18:19]
	s_branch .LBB0_423
.LBB0_420:
	s_or_b64 exec, exec, s[8:9]
	v_readfirstlane_b32 s8, v2
	s_cmp_eq_u32 s8, 0
	s_cbranch_scc1 .LBB0_422
	s_sleep 1
	s_cbranch_execnz .LBB0_423
	s_branch .LBB0_425
.LBB0_422:
	s_branch .LBB0_425
.LBB0_423:
	v_mov_b32_e32 v2, 1
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_420
	global_load_dword v2, v[8:9], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v2, 1, v2
	s_branch .LBB0_420
.LBB0_425:
	global_load_dwordx2 v[0:1], v[0:1], off
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_428
	v_mov_b32_e32 v8, 0
	global_load_dwordx2 v[2:3], v8, s[2:3] offset:40
	global_load_dwordx2 v[14:15], v8, s[2:3] offset:24 sc0 sc1
	global_load_dwordx2 v[4:5], v8, s[2:3]
	s_waitcnt vmcnt(2)
	v_readfirstlane_b32 s10, v2
	v_readfirstlane_b32 s11, v3
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
	v_lshl_add_u64 v[6:7], v[4:5], 0, s[6:7]
	v_mov_b32_e32 v12, s0
	global_store_dwordx2 v[6:7], v[14:15], off
	v_mov_b32_e32 v13, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[4:5], v8, v[12:15], s[2:3] offset:24 sc0 sc1
	s_mov_b64 s[6:7], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[4:5], v[14:15]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_428
.LBB0_427:
	s_sleep 1
	global_store_dwordx2 v[6:7], v[4:5], off
	v_mov_b32_e32 v2, s0
	v_mov_b32_e32 v3, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v8, v[2:5], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[2:3], v[4:5]
	s_or_b64 s[6:7], vcc, s[6:7]
	v_mov_b64_e32 v[4:5], v[2:3]
	s_andn2_b64 exec, exec, s[6:7]
	s_cbranch_execnz .LBB0_427
.LBB0_428:
	s_or_b64 exec, exec, s[8:9]
	v_readfirstlane_b32 s0, v16
	v_mov_b64_e32 v[8:9], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v16
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_434
	v_mov_b32_e32 v4, 0
	global_load_dwordx2 v[14:15], v4, s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[2:3], v4, s[2:3] offset:40
	global_load_dwordx2 v[6:7], v4, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v14
	v_and_b32_e32 v3, v3, v15
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v5, v2, 24
	v_add_u32_e32 v3, v5, v3
	v_mul_lo_u32 v2, v2, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[6:7], 0, v[2:3]
	global_load_dwordx2 v[12:13], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[8:9], v4, v[12:15], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[8:9], v[14:15]
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_433
	s_mov_b64 s[10:11], 0
.LBB0_431:
	s_sleep 1
	global_load_dwordx2 v[2:3], v4, s[2:3] offset:40
	global_load_dwordx2 v[6:7], v4, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v8
	v_and_b32_e32 v5, v3, v9
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[2:3], s[12:13], v2, 24, v[6:7]
	v_mov_b32_e32 v6, v3
	v_mad_u64_u32 v[6:7], s[12:13], v5, 24, v[6:7]
	v_mov_b32_e32 v3, v6
	global_load_dwordx2 v[6:7], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v4, v[6:9], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[2:3], v[8:9]
	s_or_b64 s[10:11], vcc, s[10:11]
	v_mov_b64_e32 v[8:9], v[2:3]
	s_andn2_b64 exec, exec, s[10:11]
	s_cbranch_execnz .LBB0_431
	s_or_b64 exec, exec, s[10:11]
	v_mov_b64_e32 v[8:9], v[2:3]
.LBB0_433:
	s_or_b64 exec, exec, s[8:9]
.LBB0_434:
	s_or_b64 exec, exec, s[6:7]
	v_mov_b32_e32 v12, 0
	global_load_dwordx2 v[2:3], v12, s[2:3] offset:40
	global_load_dwordx4 v[4:7], v12, s[2:3]
	v_readfirstlane_b32 s7, v9
	v_readfirstlane_b32 s6, v8
	s_mov_b64 s[8:9], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s10, v2
	v_readfirstlane_b32 s11, v3
	s_and_b64 s[10:11], s[10:11], s[6:7]
	s_mul_i32 s12, s11, 24
	s_mul_hi_u32 s13, s10, 24
	s_add_i32 s13, s13, s12
	s_mul_i32 s12, s10, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[8:9], v[4:5], 0, s[12:13]
	s_and_saveexec_b64 s[12:13], s[0:1]
	s_cbranch_execz .LBB0_436
	v_mov_b64_e32 v[12:13], s[8:9]
	v_mov_b32_e32 v14, 2
	v_mov_b32_e32 v15, 1
	global_store_dwordx4 v[8:9], v[12:15], off offset:8
.LBB0_436:
	s_or_b64 exec, exec, s[12:13]
	s_lshl_b64 s[8:9], s[10:11], 12
	v_lshl_add_u64 v[6:7], v[6:7], 0, s[8:9]
	v_and_b32_e32 v0, 0xffffff1d, v0
	s_mov_b32 s8, 0
	v_cvt_f64_f32_e32 v[2:3], v10
	v_or_b32_e32 v0, 0x42, v0
	v_readfirstlane_b32 s12, v6
	v_readfirstlane_b32 s13, v7
	s_mov_b32 s9, s8
	s_mov_b32 s10, s8
	s_mov_b32 s11, s8
	v_cvt_f64_f32_e32 v[10:11], v11
	v_mov_b32_e32 v12, s8
	global_store_dwordx4 v18, v[0:3], s[12:13]
	v_mov_b32_e32 v13, s8
	global_store_dwordx4 v18, v[10:13], s[12:13] offset:16
	v_mov_b64_e32 v[0:1], s[8:9]
	v_mov_b64_e32 v[2:3], s[10:11]
	global_store_dwordx4 v18, v[0:3], s[12:13] offset:32
	global_store_dwordx4 v18, v[0:3], s[12:13] offset:48
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_444
	v_mov_b32_e32 v6, 0
	global_load_dwordx2 v[12:13], v6, s[2:3] offset:32 sc0 sc1
	global_load_dwordx2 v[0:1], v6, s[2:3] offset:40
	v_mov_b32_e32 v10, s6
	v_mov_b32_e32 v11, s7
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v0
	v_readfirstlane_b32 s11, v1
	s_and_b64 s[10:11], s[10:11], s[6:7]
	s_mul_i32 s11, s11, 24
	s_mul_hi_u32 s12, s10, 24
	s_mul_i32 s10, s10, 24
	s_add_i32 s11, s12, s11
	v_lshl_add_u64 v[4:5], v[4:5], 0, s[10:11]
	global_store_dwordx2 v[4:5], v[12:13], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v6, v[10:13], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[2:3], v[12:13]
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_440
	s_mov_b64 s[12:13], 0
.LBB0_439:
	s_sleep 1
	global_store_dwordx2 v[4:5], v[2:3], off
	v_mov_b32_e32 v0, s6
	v_mov_b32_e32 v1, s7
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[0:1], v6, v[0:3], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[0:1], v[2:3]
	s_or_b64 s[12:13], vcc, s[12:13]
	v_mov_b64_e32 v[2:3], v[0:1]
	s_andn2_b64 exec, exec, s[12:13]
	s_cbranch_execnz .LBB0_439
.LBB0_440:
	s_or_b64 exec, exec, s[10:11]
	v_mov_b32_e32 v3, 0
	global_load_dwordx2 v[0:1], v3, s[2:3] offset:16
	s_mov_b64 s[10:11], exec
	v_mbcnt_lo_u32_b32 v2, s10, 0
	v_mbcnt_hi_u32_b32 v2, s11, v2
	v_cmp_eq_u32_e32 vcc, 0, v2
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB0_442
	s_bcnt1_i32_b64 s10, s[10:11]
	v_mov_b32_e32 v2, s10
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[0:1], v[2:3], off offset:8 sc1
.LBB0_442:
	s_or_b64 exec, exec, s[12:13]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[2:3], v[0:1], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[2:3]
	s_cbranch_vccnz .LBB0_444
	global_load_dword v0, v[0:1], off offset:24
	v_mov_b32_e32 v1, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v0
	s_and_b32 m0, s10, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[2:3], v[0:1], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_444:
	s_or_b64 exec, exec, s[8:9]
	s_branch .LBB0_448
.LBB0_445:
	s_or_b64 exec, exec, s[8:9]
	v_readfirstlane_b32 s8, v0
	s_cmp_eq_u32 s8, 0
	s_cbranch_scc1 .LBB0_447
	s_sleep 1
	s_cbranch_execnz .LBB0_448
	s_branch .LBB0_450
.LBB0_447:
	s_branch .LBB0_450
.LBB0_448:
	v_mov_b32_e32 v0, 1
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_445
	global_load_dword v0, v[8:9], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v0, 1, v0
	s_branch .LBB0_445
.LBB0_450:
	s_and_b64 exec, exec, s[0:1]
	s_cbranch_execz .LBB0_453
	v_mov_b32_e32 v6, 0
	global_load_dwordx2 v[0:1], v6, s[2:3] offset:40
	global_load_dwordx2 v[10:11], v6, s[2:3] offset:24 sc0 sc1
	global_load_dwordx2 v[2:3], v6, s[2:3]
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
	global_atomic_cmpswap_x2 v[2:3], v6, v[8:11], s[2:3] offset:24 sc0 sc1
	s_mov_b64 s[6:7], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[2:3], v[10:11]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_453
.LBB0_452:
	s_sleep 1
	global_store_dwordx2 v[4:5], v[2:3], off
	v_mov_b32_e32 v0, s0
	v_mov_b32_e32 v1, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[0:1], v6, v[0:3], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[0:1], v[2:3]
	s_or_b64 s[6:7], vcc, s[6:7]
	v_mov_b64_e32 v[2:3], v[0:1]
	s_andn2_b64 exec, exec, s[6:7]
	s_cbranch_execnz .LBB0_452
.LBB0_453:
	s_or_b64 exec, exec, s[4:5]
	;;#ASMSTART
	v_mov_b32 v10, v95
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v11, v96
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v12, v97
	;;#ASMEND
	s_nop 0
	v_cmp_lt_f32_e32 vcc, 2.0, v10
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_529
	v_readfirstlane_b32 s0, v16
	v_mov_b64_e32 v[4:5], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v16
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_460
	v_mov_b32_e32 v6, 0
	global_load_dwordx2 v[2:3], v6, s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[0:1], v6, s[2:3] offset:40
	global_load_dwordx2 v[4:5], v6, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v0, v0, v2
	v_and_b32_e32 v1, v1, v3
	v_mul_lo_u32 v1, v1, 24
	v_mul_hi_u32 v7, v0, 24
	v_add_u32_e32 v1, v7, v1
	v_mul_lo_u32 v0, v0, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[0:1], v[4:5], 0, v[0:1]
	global_load_dwordx2 v[0:1], v[0:1], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[4:5], v6, v[0:3], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[4:5], v[2:3]
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_459
	s_mov_b64 s[10:11], 0
.LBB0_457:
	s_sleep 1
	global_load_dwordx2 v[0:1], v6, s[2:3] offset:40
	global_load_dwordx2 v[2:3], v6, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v0, v0, v4
	v_and_b32_e32 v7, v1, v5
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[0:1], s[12:13], v0, 24, v[2:3]
	v_mov_b32_e32 v2, v1
	v_mad_u64_u32 v[2:3], s[12:13], v7, 24, v[2:3]
	v_mov_b32_e32 v1, v2
	global_load_dwordx2 v[2:3], v[0:1], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[0:1], v6, v[2:5], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[0:1], v[4:5]
	s_or_b64 s[10:11], vcc, s[10:11]
	v_mov_b64_e32 v[4:5], v[0:1]
	s_andn2_b64 exec, exec, s[10:11]
	s_cbranch_execnz .LBB0_457
	s_or_b64 exec, exec, s[10:11]
	v_mov_b64_e32 v[4:5], v[0:1]
.LBB0_459:
	s_or_b64 exec, exec, s[8:9]
.LBB0_460:
	s_or_b64 exec, exec, s[6:7]
	v_mov_b32_e32 v19, 0
	global_load_dwordx2 v[6:7], v19, s[2:3] offset:40
	global_load_dwordx4 v[0:3], v19, s[2:3]
	v_readfirstlane_b32 s7, v5
	v_readfirstlane_b32 s6, v4
	s_mov_b64 s[8:9], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s10, v6
	v_readfirstlane_b32 s11, v7
	s_and_b64 s[10:11], s[10:11], s[6:7]
	s_mul_i32 s12, s11, 24
	s_mul_hi_u32 s13, s10, 24
	s_add_i32 s13, s13, s12
	s_mul_i32 s12, s10, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[4:5], v[0:1], 0, s[12:13]
	s_and_saveexec_b64 s[12:13], s[0:1]
	s_cbranch_execz .LBB0_462
	v_mov_b64_e32 v[6:7], s[8:9]
	v_mov_b32_e32 v8, 2
	v_mov_b32_e32 v9, 1
	global_store_dwordx4 v[4:5], v[6:9], off offset:8
.LBB0_462:
	s_or_b64 exec, exec, s[12:13]
	s_lshl_b64 s[8:9], s[10:11], 12
	v_lshl_add_u64 v[6:7], v[2:3], 0, s[8:9]
	s_mov_b32 s8, 0
	v_mov_b32_e32 v20, 33
	v_mov_b32_e32 v21, v19
	v_mov_b32_e32 v22, v19
	v_mov_b32_e32 v23, v19
	v_readfirstlane_b32 s12, v6
	v_readfirstlane_b32 s13, v7
	s_mov_b32 s10, s8
	s_mov_b32 s11, s8
	s_mov_b32 s9, s8
	s_nop 1
	global_store_dwordx4 v18, v[20:23], s[12:13]
	s_nop 1
	v_mov_b64_e32 v[22:23], s[10:11]
	v_mov_b64_e32 v[20:21], s[8:9]
	global_store_dwordx4 v18, v[20:23], s[12:13] offset:16
	global_store_dwordx4 v18, v[20:23], s[12:13] offset:32
	global_store_dwordx4 v18, v[20:23], s[12:13] offset:48
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_470
	v_mov_b32_e32 v13, 0
	global_load_dwordx2 v[22:23], v13, s[2:3] offset:32 sc0 sc1
	global_load_dwordx2 v[2:3], v13, s[2:3] offset:40
	v_mov_b32_e32 v20, s6
	v_mov_b32_e32 v21, s7
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v2
	v_readfirstlane_b32 s11, v3
	s_and_b64 s[10:11], s[10:11], s[6:7]
	s_mul_i32 s11, s11, 24
	s_mul_hi_u32 s12, s10, 24
	s_mul_i32 s10, s10, 24
	s_add_i32 s11, s12, s11
	v_lshl_add_u64 v[8:9], v[0:1], 0, s[10:11]
	global_store_dwordx2 v[8:9], v[22:23], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v13, v[20:23], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[2:3], v[22:23]
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_466
	s_mov_b64 s[12:13], 0
.LBB0_465:
	s_sleep 1
	global_store_dwordx2 v[8:9], v[2:3], off
	v_mov_b32_e32 v0, s6
	v_mov_b32_e32 v1, s7
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[0:1], v13, v[0:3], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[0:1], v[2:3]
	s_or_b64 s[12:13], vcc, s[12:13]
	v_mov_b64_e32 v[2:3], v[0:1]
	s_andn2_b64 exec, exec, s[12:13]
	s_cbranch_execnz .LBB0_465
.LBB0_466:
	s_or_b64 exec, exec, s[10:11]
	v_mov_b32_e32 v3, 0
	global_load_dwordx2 v[0:1], v3, s[2:3] offset:16
	s_mov_b64 s[10:11], exec
	v_mbcnt_lo_u32_b32 v2, s10, 0
	v_mbcnt_hi_u32_b32 v2, s11, v2
	v_cmp_eq_u32_e32 vcc, 0, v2
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB0_468
	s_bcnt1_i32_b64 s10, s[10:11]
	v_mov_b32_e32 v2, s10
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[0:1], v[2:3], off offset:8 sc1
.LBB0_468:
	s_or_b64 exec, exec, s[12:13]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[2:3], v[0:1], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[2:3]
	s_cbranch_vccnz .LBB0_470
	global_load_dword v0, v[0:1], off offset:24
	v_mov_b32_e32 v1, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v0
	s_and_b32 m0, s10, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[2:3], v[0:1], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_470:
	s_or_b64 exec, exec, s[8:9]
	v_lshl_add_u64 v[0:1], v[6:7], 0, v[18:19]
	s_branch .LBB0_474
.LBB0_471:
	s_or_b64 exec, exec, s[8:9]
	v_readfirstlane_b32 s8, v2
	s_cmp_eq_u32 s8, 0
	s_cbranch_scc1 .LBB0_473
	s_sleep 1
	s_cbranch_execnz .LBB0_474
	s_branch .LBB0_476
.LBB0_473:
	s_branch .LBB0_476
.LBB0_474:
	v_mov_b32_e32 v2, 1
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_471
	global_load_dword v2, v[4:5], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v2, 1, v2
	s_branch .LBB0_471
.LBB0_476:
	global_load_dwordx2 v[0:1], v[0:1], off
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_479
	v_mov_b32_e32 v8, 0
	global_load_dwordx2 v[2:3], v8, s[2:3] offset:40
	global_load_dwordx2 v[22:23], v8, s[2:3] offset:24 sc0 sc1
	global_load_dwordx2 v[4:5], v8, s[2:3]
	s_waitcnt vmcnt(2)
	v_readfirstlane_b32 s10, v2
	v_readfirstlane_b32 s11, v3
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
	v_lshl_add_u64 v[6:7], v[4:5], 0, s[6:7]
	v_mov_b32_e32 v20, s0
	global_store_dwordx2 v[6:7], v[22:23], off
	v_mov_b32_e32 v21, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[4:5], v8, v[20:23], s[2:3] offset:24 sc0 sc1
	s_mov_b64 s[6:7], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[4:5], v[22:23]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_479
.LBB0_478:
	s_sleep 1
	global_store_dwordx2 v[6:7], v[4:5], off
	v_mov_b32_e32 v2, s0
	v_mov_b32_e32 v3, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v8, v[2:5], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[2:3], v[4:5]
	s_or_b64 s[6:7], vcc, s[6:7]
	v_mov_b64_e32 v[4:5], v[2:3]
	s_andn2_b64 exec, exec, s[6:7]
	s_cbranch_execnz .LBB0_478
.LBB0_479:
	s_or_b64 exec, exec, s[8:9]
	v_readfirstlane_b32 s0, v16
	v_mov_b64_e32 v[8:9], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v16
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_485
	v_mov_b32_e32 v4, 0
	global_load_dwordx2 v[22:23], v4, s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[2:3], v4, s[2:3] offset:40
	global_load_dwordx2 v[6:7], v4, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v22
	v_and_b32_e32 v3, v3, v23
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v5, v2, 24
	v_add_u32_e32 v3, v5, v3
	v_mul_lo_u32 v2, v2, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[6:7], 0, v[2:3]
	global_load_dwordx2 v[20:21], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[8:9], v4, v[20:23], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[8:9], v[22:23]
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_484
	s_mov_b64 s[10:11], 0
.LBB0_482:
	s_sleep 1
	global_load_dwordx2 v[2:3], v4, s[2:3] offset:40
	global_load_dwordx2 v[6:7], v4, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v8
	v_and_b32_e32 v5, v3, v9
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[2:3], s[12:13], v2, 24, v[6:7]
	v_mov_b32_e32 v6, v3
	v_mad_u64_u32 v[6:7], s[12:13], v5, 24, v[6:7]
	v_mov_b32_e32 v3, v6
	global_load_dwordx2 v[6:7], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v4, v[6:9], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[2:3], v[8:9]
	s_or_b64 s[10:11], vcc, s[10:11]
	v_mov_b64_e32 v[8:9], v[2:3]
	s_andn2_b64 exec, exec, s[10:11]
	s_cbranch_execnz .LBB0_482
	s_or_b64 exec, exec, s[10:11]
	v_mov_b64_e32 v[8:9], v[2:3]
.LBB0_484:
	s_or_b64 exec, exec, s[8:9]
.LBB0_485:
	s_or_b64 exec, exec, s[6:7]
	v_mov_b32_e32 v19, 0
	global_load_dwordx2 v[2:3], v19, s[2:3] offset:40
	global_load_dwordx4 v[4:7], v19, s[2:3]
	v_readfirstlane_b32 s7, v9
	v_readfirstlane_b32 s6, v8
	s_mov_b64 s[8:9], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s10, v2
	v_readfirstlane_b32 s11, v3
	s_and_b64 s[10:11], s[10:11], s[6:7]
	s_mul_i32 s12, s11, 24
	s_mul_hi_u32 s13, s10, 24
	s_add_i32 s13, s13, s12
	s_mul_i32 s12, s10, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[8:9], v[4:5], 0, s[12:13]
	s_and_saveexec_b64 s[12:13], s[0:1]
	s_cbranch_execz .LBB0_487
	v_mov_b64_e32 v[20:21], s[8:9]
	v_mov_b32_e32 v22, 2
	v_mov_b32_e32 v23, 1
	global_store_dwordx4 v[8:9], v[20:23], off offset:8
.LBB0_487:
	s_or_b64 exec, exec, s[12:13]
	s_lshl_b64 s[8:9], s[10:11], 12
	v_lshl_add_u64 v[6:7], v[6:7], 0, s[8:9]
	v_and_b32_e32 v0, 0xffffff1f, v0
	v_or_b32_e32 v0, 0xa0, v0
	v_mov_b32_e32 v3, 0x36762072
	v_mov_b32_e32 v2, 0x65746641
	v_readfirstlane_b32 s8, v6
	v_readfirstlane_b32 s9, v7
	s_nop 4
	global_store_dwordx4 v18, v[0:3], s[8:9]
	s_nop 1
	v_mov_b32_e32 v0, 0x25203d20
	v_mov_b32_e32 v1, 0x2c66322e
	v_mov_b32_e32 v2, 0x20377620
	v_mov_b32_e32 v3, 0x2e25203d
	global_store_dwordx4 v18, v[0:3], s[8:9] offset:16
	s_nop 1
	v_mov_b32_e32 v0, 0x202c6632
	v_mov_b32_e32 v1, 0x3d203276
	v_mov_b32_e32 v2, 0x322e2520
	v_mov_b32_e32 v3, 0xa66
	global_store_dwordx4 v18, v[0:3], s[8:9] offset:32
	s_nop 1
	v_mov_b32_e32 v0, v19
	v_mov_b32_e32 v1, v19
	v_mov_b32_e32 v2, v19
	v_mov_b32_e32 v3, v19
	global_store_dwordx4 v18, v[0:3], s[8:9] offset:48
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_495
	v_mov_b32_e32 v13, 0
	global_load_dwordx2 v[22:23], v13, s[2:3] offset:32 sc0 sc1
	global_load_dwordx2 v[0:1], v13, s[2:3] offset:40
	v_mov_b32_e32 v20, s6
	v_mov_b32_e32 v21, s7
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v0
	v_readfirstlane_b32 s11, v1
	s_and_b64 s[10:11], s[10:11], s[6:7]
	s_mul_i32 s11, s11, 24
	s_mul_hi_u32 s12, s10, 24
	s_mul_i32 s10, s10, 24
	s_add_i32 s11, s12, s11
	v_lshl_add_u64 v[4:5], v[4:5], 0, s[10:11]
	global_store_dwordx2 v[4:5], v[22:23], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v13, v[20:23], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[2:3], v[22:23]
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_491
	s_mov_b64 s[12:13], 0
.LBB0_490:
	s_sleep 1
	global_store_dwordx2 v[4:5], v[2:3], off
	v_mov_b32_e32 v0, s6
	v_mov_b32_e32 v1, s7
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[0:1], v13, v[0:3], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[0:1], v[2:3]
	s_or_b64 s[12:13], vcc, s[12:13]
	v_mov_b64_e32 v[2:3], v[0:1]
	s_andn2_b64 exec, exec, s[12:13]
	s_cbranch_execnz .LBB0_490
.LBB0_491:
	s_or_b64 exec, exec, s[10:11]
	v_mov_b32_e32 v3, 0
	global_load_dwordx2 v[0:1], v3, s[2:3] offset:16
	s_mov_b64 s[10:11], exec
	v_mbcnt_lo_u32_b32 v2, s10, 0
	v_mbcnt_hi_u32_b32 v2, s11, v2
	v_cmp_eq_u32_e32 vcc, 0, v2
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB0_493
	s_bcnt1_i32_b64 s10, s[10:11]
	v_mov_b32_e32 v2, s10
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[0:1], v[2:3], off offset:8 sc1
.LBB0_493:
	s_or_b64 exec, exec, s[12:13]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[2:3], v[0:1], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[2:3]
	s_cbranch_vccnz .LBB0_495
	global_load_dword v0, v[0:1], off offset:24
	v_mov_b32_e32 v1, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v0
	s_and_b32 m0, s10, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[2:3], v[0:1], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_495:
	s_or_b64 exec, exec, s[8:9]
	v_lshl_add_u64 v[0:1], v[6:7], 0, v[18:19]
	s_branch .LBB0_499
.LBB0_496:
	s_or_b64 exec, exec, s[8:9]
	v_readfirstlane_b32 s8, v2
	s_cmp_eq_u32 s8, 0
	s_cbranch_scc1 .LBB0_498
	s_sleep 1
	s_cbranch_execnz .LBB0_499
	s_branch .LBB0_501
.LBB0_498:
	s_branch .LBB0_501
.LBB0_499:
	v_mov_b32_e32 v2, 1
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_496
	global_load_dword v2, v[8:9], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v2, 1, v2
	s_branch .LBB0_496
.LBB0_501:
	global_load_dwordx2 v[0:1], v[0:1], off
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_504
	v_mov_b32_e32 v8, 0
	global_load_dwordx2 v[2:3], v8, s[2:3] offset:40
	global_load_dwordx2 v[22:23], v8, s[2:3] offset:24 sc0 sc1
	global_load_dwordx2 v[4:5], v8, s[2:3]
	s_waitcnt vmcnt(2)
	v_readfirstlane_b32 s10, v2
	v_readfirstlane_b32 s11, v3
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
	v_lshl_add_u64 v[6:7], v[4:5], 0, s[6:7]
	v_mov_b32_e32 v20, s0
	global_store_dwordx2 v[6:7], v[22:23], off
	v_mov_b32_e32 v21, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[4:5], v8, v[20:23], s[2:3] offset:24 sc0 sc1
	s_mov_b64 s[6:7], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[4:5], v[22:23]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_504
.LBB0_503:
	s_sleep 1
	global_store_dwordx2 v[6:7], v[4:5], off
	v_mov_b32_e32 v2, s0
	v_mov_b32_e32 v3, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v8, v[2:5], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[2:3], v[4:5]
	s_or_b64 s[6:7], vcc, s[6:7]
	v_mov_b64_e32 v[4:5], v[2:3]
	s_andn2_b64 exec, exec, s[6:7]
	s_cbranch_execnz .LBB0_503
.LBB0_504:
	s_or_b64 exec, exec, s[8:9]
	v_readfirstlane_b32 s0, v16
	v_mov_b64_e32 v[8:9], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v16
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_510
	v_mov_b32_e32 v4, 0
	global_load_dwordx2 v[16:17], v4, s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[2:3], v4, s[2:3] offset:40
	global_load_dwordx2 v[6:7], v4, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v16
	v_and_b32_e32 v3, v3, v17
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v5, v2, 24
	v_add_u32_e32 v3, v5, v3
	v_mul_lo_u32 v2, v2, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[6:7], 0, v[2:3]
	global_load_dwordx2 v[14:15], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[8:9], v4, v[14:17], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[8:9], v[16:17]
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_509
	s_mov_b64 s[10:11], 0
.LBB0_507:
	s_sleep 1
	global_load_dwordx2 v[2:3], v4, s[2:3] offset:40
	global_load_dwordx2 v[6:7], v4, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v8
	v_and_b32_e32 v5, v3, v9
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[2:3], s[12:13], v2, 24, v[6:7]
	v_mov_b32_e32 v6, v3
	v_mad_u64_u32 v[6:7], s[12:13], v5, 24, v[6:7]
	v_mov_b32_e32 v3, v6
	global_load_dwordx2 v[6:7], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v4, v[6:9], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[2:3], v[8:9]
	s_or_b64 s[10:11], vcc, s[10:11]
	v_mov_b64_e32 v[8:9], v[2:3]
	s_andn2_b64 exec, exec, s[10:11]
	s_cbranch_execnz .LBB0_507
	s_or_b64 exec, exec, s[10:11]
	v_mov_b64_e32 v[8:9], v[2:3]
.LBB0_509:
	s_or_b64 exec, exec, s[8:9]
.LBB0_510:
	s_or_b64 exec, exec, s[6:7]
	v_mov_b32_e32 v13, 0
	global_load_dwordx2 v[2:3], v13, s[2:3] offset:40
	global_load_dwordx4 v[4:7], v13, s[2:3]
	v_readfirstlane_b32 s7, v9
	v_readfirstlane_b32 s6, v8
	s_mov_b64 s[8:9], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s10, v2
	v_readfirstlane_b32 s11, v3
	s_and_b64 s[10:11], s[10:11], s[6:7]
	s_mul_i32 s12, s11, 24
	s_mul_hi_u32 s13, s10, 24
	s_add_i32 s13, s13, s12
	s_mul_i32 s12, s10, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[8:9], v[4:5], 0, s[12:13]
	s_and_saveexec_b64 s[12:13], s[0:1]
	s_cbranch_execz .LBB0_512
	v_mov_b64_e32 v[14:15], s[8:9]
	v_mov_b32_e32 v16, 2
	v_mov_b32_e32 v17, 1
	global_store_dwordx4 v[8:9], v[14:17], off offset:8
.LBB0_512:
	s_or_b64 exec, exec, s[12:13]
	s_lshl_b64 s[8:9], s[10:11], 12
	v_lshl_add_u64 v[6:7], v[6:7], 0, s[8:9]
	v_and_b32_e32 v0, 0xffffff1d, v0
	s_mov_b32 s8, 0
	v_cvt_f64_f32_e32 v[2:3], v10
	v_or_b32_e32 v0, 0x62, v0
	v_readfirstlane_b32 s12, v6
	v_readfirstlane_b32 s13, v7
	s_mov_b32 s9, s8
	v_cvt_f64_f32_e32 v[10:11], v11
	v_cvt_f64_f32_e32 v[12:13], v12
	s_nop 1
	global_store_dwordx4 v18, v[0:3], s[12:13]
	global_store_dwordx4 v18, v[10:13], s[12:13] offset:16
	s_mov_b32 s10, s8
	s_mov_b32 s11, s8
	v_mov_b64_e32 v[0:1], s[8:9]
	v_mov_b64_e32 v[2:3], s[10:11]
	global_store_dwordx4 v18, v[0:3], s[12:13] offset:32
	global_store_dwordx4 v18, v[0:3], s[12:13] offset:48
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_520
	v_mov_b32_e32 v6, 0
	global_load_dwordx2 v[12:13], v6, s[2:3] offset:32 sc0 sc1
	global_load_dwordx2 v[0:1], v6, s[2:3] offset:40
	v_mov_b32_e32 v10, s6
	v_mov_b32_e32 v11, s7
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v0
	v_readfirstlane_b32 s11, v1
	s_and_b64 s[10:11], s[10:11], s[6:7]
	s_mul_i32 s11, s11, 24
	s_mul_hi_u32 s12, s10, 24
	s_mul_i32 s10, s10, 24
	s_add_i32 s11, s12, s11
	v_lshl_add_u64 v[4:5], v[4:5], 0, s[10:11]
	global_store_dwordx2 v[4:5], v[12:13], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v6, v[10:13], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[2:3], v[12:13]
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_516
	s_mov_b64 s[12:13], 0
.LBB0_515:
	s_sleep 1
	global_store_dwordx2 v[4:5], v[2:3], off
	v_mov_b32_e32 v0, s6
	v_mov_b32_e32 v1, s7
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[0:1], v6, v[0:3], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[0:1], v[2:3]
	s_or_b64 s[12:13], vcc, s[12:13]
	v_mov_b64_e32 v[2:3], v[0:1]
	s_andn2_b64 exec, exec, s[12:13]
	s_cbranch_execnz .LBB0_515
.LBB0_516:
	s_or_b64 exec, exec, s[10:11]
	v_mov_b32_e32 v3, 0
	global_load_dwordx2 v[0:1], v3, s[2:3] offset:16
	s_mov_b64 s[10:11], exec
	v_mbcnt_lo_u32_b32 v2, s10, 0
	v_mbcnt_hi_u32_b32 v2, s11, v2
	v_cmp_eq_u32_e32 vcc, 0, v2
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB0_518
	s_bcnt1_i32_b64 s10, s[10:11]
	v_mov_b32_e32 v2, s10
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[0:1], v[2:3], off offset:8 sc1
.LBB0_518:
	s_or_b64 exec, exec, s[12:13]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[2:3], v[0:1], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[2:3]
	s_cbranch_vccnz .LBB0_520
	global_load_dword v0, v[0:1], off offset:24
	v_mov_b32_e32 v1, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v0
	s_and_b32 m0, s10, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[2:3], v[0:1], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_520:
	s_or_b64 exec, exec, s[8:9]
	s_branch .LBB0_524
.LBB0_521:
	s_or_b64 exec, exec, s[8:9]
	v_readfirstlane_b32 s8, v0
	s_cmp_eq_u32 s8, 0
	s_cbranch_scc1 .LBB0_523
	s_sleep 1
	s_cbranch_execnz .LBB0_524
	s_branch .LBB0_526
.LBB0_523:
	s_branch .LBB0_526
.LBB0_524:
	v_mov_b32_e32 v0, 1
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_521
	global_load_dword v0, v[8:9], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v0, 1, v0
	s_branch .LBB0_521
.LBB0_526:
	s_and_b64 exec, exec, s[0:1]
	s_cbranch_execz .LBB0_529
	v_mov_b32_e32 v6, 0
	global_load_dwordx2 v[0:1], v6, s[2:3] offset:40
	global_load_dwordx2 v[10:11], v6, s[2:3] offset:24 sc0 sc1
	global_load_dwordx2 v[2:3], v6, s[2:3]
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
	global_atomic_cmpswap_x2 v[2:3], v6, v[8:11], s[2:3] offset:24 sc0 sc1
	s_mov_b64 s[6:7], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[2:3], v[10:11]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_529
.LBB0_528:
	s_sleep 1
	global_store_dwordx2 v[4:5], v[2:3], off
	v_mov_b32_e32 v0, s0
	v_mov_b32_e32 v1, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[0:1], v6, v[0:3], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[0:1], v[2:3]
	s_or_b64 s[6:7], vcc, s[6:7]
	v_mov_b64_e32 v[2:3], v[0:1]
	s_andn2_b64 exec, exec, s[6:7]
	s_cbranch_execnz .LBB0_528
.LBB0_529:
	s_or_b64 exec, exec, s[4:5]
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
		.amdhsa_next_free_vgpr 101
		.amdhsa_next_free_sgpr 22
		.amdhsa_accum_offset 24
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

	.set _Z9vectorAddPKfS0_Pfi.num_vgpr, 24
	.set _Z9vectorAddPKfS0_Pfi.num_agpr, 0
	.set _Z9vectorAddPKfS0_Pfi.numbered_sgpr, 14
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
  .sgpr_count: 20
  .sgpr_spill_count: 0
  .symbol: _Z9vectorAddPKfS0_Pfi.kd
  .uniform_work_group_size: 1
  .uses_dynamic_stack: false
  .vgpr_count: 24
  .vgpr_spill_count: 0
  .wavefront_size: 64
amdhsa.target: amdgcn-amd-amdhsa--gfx950
amdhsa.version:
- 1
- 2
...
...

	.end_amdgpu_metadata
