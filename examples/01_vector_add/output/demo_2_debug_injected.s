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
	v_mov_b32 v48, v6
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v49, v7
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v50, v2
	;;#ASMEND
	;;#ASMSTART
		v_add_f32_e32 v2, v6, v7
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v51, v6
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v52, v7
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v53, v2
	;;#ASMEND
	;;#ASMSTART
		global_store_dword v[0:1], v2, off
	;;#ASMEND
	;;#ASMSTART
	s_mov_b64 s[0:1], s[18:19]
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v11, v48
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v14, v49
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v15, v50
	;;#ASMEND
	s_load_dwordx2 s[2:3], s[0:1], 0x70
	v_mbcnt_lo_u32_b32 v0, -1, 0
	v_mbcnt_hi_u32_b32 v10, -1, v0
	v_mov_b64_e32 v[4:5], 0
	v_readfirstlane_b32 s0, v10
	s_nop 1
	v_cmp_eq_u32_e64 s[0:1], s0, v10
	s_and_saveexec_b64 s[4:5], s[0:1]
	s_cbranch_execz .LBB0_6
	v_mov_b32_e32 v6, 0
	s_waitcnt lgkmcnt(0)
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
	s_cbranch_execz .LBB0_5
	s_mov_b64 s[8:9], 0
.LBB0_3:
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
	s_cbranch_execnz .LBB0_3
	s_or_b64 exec, exec, s[8:9]
	v_mov_b64_e32 v[4:5], v[0:1]
.LBB0_5:
	s_or_b64 exec, exec, s[6:7]
.LBB0_6:
	s_or_b64 exec, exec, s[4:5]
	v_mov_b32_e32 v7, 0
	s_waitcnt lgkmcnt(0)
	global_load_dwordx2 v[8:9], v7, s[2:3] offset:40
	global_load_dwordx4 v[0:3], v7, s[2:3]
	v_readfirstlane_b32 s5, v5
	v_readfirstlane_b32 s4, v4
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s8, v8
	v_readfirstlane_b32 s9, v9
	s_and_b64 s[8:9], s[8:9], s[4:5]
	s_mul_i32 s10, s9, 24
	s_mul_hi_u32 s11, s8, 24
	s_add_i32 s11, s11, s10
	s_mul_i32 s10, s8, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[4:5], v[0:1], 0, s[10:11]
	s_and_saveexec_b64 s[10:11], s[0:1]
	s_cbranch_execz .LBB0_8
	v_mov_b64_e32 v[16:17], s[6:7]
	v_mov_b32_e32 v18, 2
	v_mov_b32_e32 v19, 1
	global_store_dwordx4 v[4:5], v[16:19], off offset:8
.LBB0_8:
	s_or_b64 exec, exec, s[10:11]
	s_lshl_b64 s[6:7], s[8:9], 12
	v_lshl_add_u64 v[8:9], v[2:3], 0, s[6:7]
	s_mov_b32 s8, 0
	v_lshlrev_b32_e32 v6, 6, v10
	v_mov_b32_e32 v16, 33
	v_mov_b32_e32 v17, v7
	v_mov_b32_e32 v18, v7
	v_mov_b32_e32 v19, v7
	v_readfirstlane_b32 s6, v8
	v_readfirstlane_b32 s7, v9
	s_mov_b32 s10, s8
	s_mov_b32 s11, s8
	s_mov_b32 s9, s8
	s_nop 1
	global_store_dwordx4 v6, v[16:19], s[6:7]
	s_nop 1
	v_mov_b64_e32 v[18:19], s[10:11]
	v_mov_b64_e32 v[16:17], s[8:9]
	global_store_dwordx4 v6, v[16:19], s[6:7] offset:16
	global_store_dwordx4 v6, v[16:19], s[6:7] offset:32
	global_store_dwordx4 v6, v[16:19], s[6:7] offset:48
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_16
	v_mov_b32_e32 v16, 0
	global_load_dwordx2 v[20:21], v16, s[2:3] offset:32 sc0 sc1
	global_load_dwordx2 v[2:3], v16, s[2:3] offset:40
	v_mov_b32_e32 v18, s4
	v_mov_b32_e32 v19, s5
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v2, s4, v2
	v_and_b32_e32 v3, s5, v3
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v12, v2, 24
	v_mul_lo_u32 v2, v2, 24
	v_add_u32_e32 v3, v12, v3
	v_lshl_add_u64 v[12:13], v[0:1], 0, v[2:3]
	global_store_dwordx2 v[12:13], v[20:21], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v16, v[18:21], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[2:3], v[20:21]
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_12
	s_mov_b64 s[10:11], 0
.LBB0_11:
	s_sleep 1
	global_store_dwordx2 v[12:13], v[2:3], off
	v_mov_b32_e32 v0, s4
	v_mov_b32_e32 v1, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[0:1], v16, v[0:3], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[0:1], v[2:3]
	s_or_b64 s[10:11], vcc, s[10:11]
	v_mov_b64_e32 v[2:3], v[0:1]
	s_andn2_b64 exec, exec, s[10:11]
	s_cbranch_execnz .LBB0_11
.LBB0_12:
	s_or_b64 exec, exec, s[8:9]
	v_mov_b32_e32 v3, 0
	global_load_dwordx2 v[0:1], v3, s[2:3] offset:16
	s_mov_b64 s[8:9], exec
	v_mbcnt_lo_u32_b32 v2, s8, 0
	v_mbcnt_hi_u32_b32 v2, s9, v2
	v_cmp_eq_u32_e32 vcc, 0, v2
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_14
	s_bcnt1_i32_b64 s8, s[8:9]
	v_mov_b32_e32 v2, s8
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[0:1], v[2:3], off offset:8 sc1
.LBB0_14:
	s_or_b64 exec, exec, s[10:11]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[2:3], v[0:1], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[2:3]
	s_cbranch_vccnz .LBB0_16
	global_load_dword v0, v[0:1], off offset:24
	v_mov_b32_e32 v1, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s8, v0
	s_and_b32 m0, s8, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[2:3], v[0:1], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_16:
	s_or_b64 exec, exec, s[6:7]
	v_lshl_add_u64 v[0:1], v[8:9], 0, v[6:7]
	s_branch .LBB0_20
.LBB0_17:
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s6, v2
	s_cmp_eq_u32 s6, 0
	s_cbranch_scc1 .LBB0_19
	s_sleep 1
	s_cbranch_execnz .LBB0_20
	s_branch .LBB0_22
.LBB0_19:
	s_branch .LBB0_22
.LBB0_20:
	v_mov_b32_e32 v2, 1
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_17
	global_load_dword v2, v[4:5], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v2, 1, v2
	s_branch .LBB0_17
.LBB0_22:
	global_load_dwordx2 v[0:1], v[0:1], off
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_25
	v_mov_b32_e32 v8, 0
	global_load_dwordx2 v[2:3], v8, s[2:3] offset:40
	global_load_dwordx2 v[18:19], v8, s[2:3] offset:24 sc0 sc1
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
	v_mov_b32_e32 v16, s0
	global_store_dwordx2 v[6:7], v[18:19], off
	v_mov_b32_e32 v17, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[4:5], v8, v[16:19], s[2:3] offset:24 sc0 sc1
	s_mov_b64 s[4:5], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[4:5], v[18:19]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_25
.LBB0_24:
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
	s_cbranch_execnz .LBB0_24
.LBB0_25:
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s0, v10
	v_mov_b64_e32 v[8:9], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v10
	s_and_saveexec_b64 s[4:5], s[0:1]
	s_cbranch_execz .LBB0_31
	v_mov_b32_e32 v4, 0
	global_load_dwordx2 v[18:19], v4, s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[2:3], v4, s[2:3] offset:40
	global_load_dwordx2 v[6:7], v4, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v18
	v_and_b32_e32 v3, v3, v19
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v5, v2, 24
	v_add_u32_e32 v3, v5, v3
	v_mul_lo_u32 v2, v2, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[6:7], 0, v[2:3]
	global_load_dwordx2 v[16:17], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[8:9], v4, v[16:19], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[8:9], v[18:19]
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_30
	s_mov_b64 s[8:9], 0
.LBB0_28:
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
	s_cbranch_execnz .LBB0_28
	s_or_b64 exec, exec, s[8:9]
	v_mov_b64_e32 v[8:9], v[2:3]
.LBB0_30:
	s_or_b64 exec, exec, s[6:7]
.LBB0_31:
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
	s_cbranch_execz .LBB0_33
	v_mov_b64_e32 v[16:17], s[6:7]
	v_mov_b32_e32 v18, 2
	v_mov_b32_e32 v19, 1
	global_store_dwordx4 v[8:9], v[16:19], off offset:8
.LBB0_33:
	s_or_b64 exec, exec, s[10:11]
	s_lshl_b64 s[6:7], s[8:9], 12
	v_lshl_add_u64 v[6:7], v[6:7], 0, s[6:7]
	v_and_b32_e32 v0, 0xffffff1f, v0
	v_or_b32_e32 v0, 0xa0, v0
	v_lshlrev_b32_e32 v12, 6, v10
	v_mov_b32_e32 v3, 0x2065726f
	v_mov_b32_e32 v2, 0x6665425b
	v_readfirstlane_b32 s6, v6
	v_readfirstlane_b32 s7, v7
	v_mov_b32_e32 v16, 0x5d444441
	v_mov_b32_e32 v17, 0x253d4120
	v_mov_b32_e32 v18, 0x2c66322e
	v_mov_b32_e32 v19, 0x253d4220
	v_mov_b32_e32 v20, 0xa66322e
	global_store_dwordx4 v12, v[0:3], s[6:7]
	global_store_dwordx4 v12, v[16:19], s[6:7] offset:16
	v_mov_b32_e32 v21, v13
	v_mov_b32_e32 v0, v13
	v_mov_b32_e32 v19, 0x253d4320
	v_mov_b32_e32 v1, v13
	v_mov_b32_e32 v2, v13
	v_mov_b32_e32 v3, v13
	global_store_dwordx4 v12, v[18:21], s[6:7] offset:32
	global_store_dwordx4 v12, v[0:3], s[6:7] offset:48
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_41
	v_mov_b32_e32 v16, 0
	global_load_dwordx2 v[20:21], v16, s[2:3] offset:32 sc0 sc1
	global_load_dwordx2 v[0:1], v16, s[2:3] offset:40
	v_mov_b32_e32 v18, s4
	v_mov_b32_e32 v19, s5
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s8, v0
	v_readfirstlane_b32 s9, v1
	s_and_b64 s[8:9], s[8:9], s[4:5]
	s_mul_i32 s9, s9, 24
	s_mul_hi_u32 s10, s8, 24
	s_mul_i32 s8, s8, 24
	s_add_i32 s9, s10, s9
	v_lshl_add_u64 v[4:5], v[4:5], 0, s[8:9]
	global_store_dwordx2 v[4:5], v[20:21], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v16, v[18:21], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[2:3], v[20:21]
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_37
	s_mov_b64 s[10:11], 0
.LBB0_36:
	s_sleep 1
	global_store_dwordx2 v[4:5], v[2:3], off
	v_mov_b32_e32 v0, s4
	v_mov_b32_e32 v1, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[0:1], v16, v[0:3], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[0:1], v[2:3]
	s_or_b64 s[10:11], vcc, s[10:11]
	v_mov_b64_e32 v[2:3], v[0:1]
	s_andn2_b64 exec, exec, s[10:11]
	s_cbranch_execnz .LBB0_36
.LBB0_37:
	s_or_b64 exec, exec, s[8:9]
	v_mov_b32_e32 v3, 0
	global_load_dwordx2 v[0:1], v3, s[2:3] offset:16
	s_mov_b64 s[8:9], exec
	v_mbcnt_lo_u32_b32 v2, s8, 0
	v_mbcnt_hi_u32_b32 v2, s9, v2
	v_cmp_eq_u32_e32 vcc, 0, v2
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_39
	s_bcnt1_i32_b64 s8, s[8:9]
	v_mov_b32_e32 v2, s8
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[0:1], v[2:3], off offset:8 sc1
.LBB0_39:
	s_or_b64 exec, exec, s[10:11]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[2:3], v[0:1], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[2:3]
	s_cbranch_vccnz .LBB0_41
	global_load_dword v0, v[0:1], off offset:24
	v_mov_b32_e32 v1, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s8, v0
	s_and_b32 m0, s8, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[2:3], v[0:1], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_41:
	s_or_b64 exec, exec, s[6:7]
	v_lshl_add_u64 v[0:1], v[6:7], 0, v[12:13]
	s_branch .LBB0_45
.LBB0_42:
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s6, v2
	s_cmp_eq_u32 s6, 0
	s_cbranch_scc1 .LBB0_44
	s_sleep 1
	s_cbranch_execnz .LBB0_45
	s_branch .LBB0_47
.LBB0_44:
	s_branch .LBB0_47
.LBB0_45:
	v_mov_b32_e32 v2, 1
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_42
	global_load_dword v2, v[8:9], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v2, 1, v2
	s_branch .LBB0_42
.LBB0_47:
	global_load_dwordx2 v[0:1], v[0:1], off
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_50
	v_mov_b32_e32 v8, 0
	global_load_dwordx2 v[2:3], v8, s[2:3] offset:40
	global_load_dwordx2 v[18:19], v8, s[2:3] offset:24 sc0 sc1
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
	v_mov_b32_e32 v16, s0
	global_store_dwordx2 v[6:7], v[18:19], off
	v_mov_b32_e32 v17, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[4:5], v8, v[16:19], s[2:3] offset:24 sc0 sc1
	s_mov_b64 s[4:5], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[4:5], v[18:19]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_50
.LBB0_49:
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
	s_cbranch_execnz .LBB0_49
.LBB0_50:
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s0, v10
	v_mov_b64_e32 v[8:9], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v10
	s_and_saveexec_b64 s[4:5], s[0:1]
	s_cbranch_execz .LBB0_56
	v_mov_b32_e32 v4, 0
	global_load_dwordx2 v[18:19], v4, s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[2:3], v4, s[2:3] offset:40
	global_load_dwordx2 v[6:7], v4, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v18
	v_and_b32_e32 v3, v3, v19
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v5, v2, 24
	v_add_u32_e32 v3, v5, v3
	v_mul_lo_u32 v2, v2, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[6:7], 0, v[2:3]
	global_load_dwordx2 v[16:17], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[8:9], v4, v[16:19], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[8:9], v[18:19]
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_55
	s_mov_b64 s[8:9], 0
.LBB0_53:
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
	s_cbranch_execnz .LBB0_53
	s_or_b64 exec, exec, s[8:9]
	v_mov_b64_e32 v[8:9], v[2:3]
.LBB0_55:
	s_or_b64 exec, exec, s[6:7]
.LBB0_56:
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
	s_cbranch_execz .LBB0_58
	v_mov_b64_e32 v[16:17], s[6:7]
	v_mov_b32_e32 v18, 2
	v_mov_b32_e32 v19, 1
	global_store_dwordx4 v[8:9], v[16:19], off offset:8
.LBB0_58:
	s_or_b64 exec, exec, s[10:11]
	s_lshl_b64 s[6:7], s[8:9], 12
	v_lshl_add_u64 v[6:7], v[6:7], 0, s[6:7]
	v_and_b32_e32 v0, 0xffffff1d, v0
	s_mov_b32 s8, 0
	v_cvt_f64_f32_e32 v[2:3], v11
	v_or_b32_e32 v0, 0x62, v0
	v_readfirstlane_b32 s6, v6
	v_readfirstlane_b32 s7, v7
	s_mov_b32 s9, s8
	v_cvt_f64_f32_e32 v[16:17], v14
	v_cvt_f64_f32_e32 v[18:19], v15
	s_nop 1
	global_store_dwordx4 v12, v[0:3], s[6:7]
	global_store_dwordx4 v12, v[16:19], s[6:7] offset:16
	s_mov_b32 s10, s8
	s_mov_b32 s11, s8
	v_mov_b64_e32 v[0:1], s[8:9]
	v_mov_b64_e32 v[2:3], s[10:11]
	global_store_dwordx4 v12, v[0:3], s[6:7] offset:32
	global_store_dwordx4 v12, v[0:3], s[6:7] offset:48
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_66
	v_mov_b32_e32 v6, 0
	global_load_dwordx2 v[16:17], v6, s[2:3] offset:32 sc0 sc1
	global_load_dwordx2 v[0:1], v6, s[2:3] offset:40
	v_mov_b32_e32 v14, s4
	v_mov_b32_e32 v15, s5
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s8, v0
	v_readfirstlane_b32 s9, v1
	s_and_b64 s[8:9], s[8:9], s[4:5]
	s_mul_i32 s9, s9, 24
	s_mul_hi_u32 s10, s8, 24
	s_mul_i32 s8, s8, 24
	s_add_i32 s9, s10, s9
	v_lshl_add_u64 v[4:5], v[4:5], 0, s[8:9]
	global_store_dwordx2 v[4:5], v[16:17], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v6, v[14:17], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[2:3], v[16:17]
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
	global_atomic_cmpswap_x2 v[0:1], v6, v[0:3], s[2:3] offset:32 sc0 sc1
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
	global_load_dword v0, v[8:9], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v0, 1, v0
	s_branch .LBB0_67
.LBB0_72:
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_75
	v_mov_b32_e32 v6, 0
	global_load_dwordx2 v[0:1], v6, s[2:3] offset:40
	global_load_dwordx2 v[16:17], v6, s[2:3] offset:24 sc0 sc1
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
	v_mov_b32_e32 v14, s0
	global_store_dwordx2 v[4:5], v[16:17], off
	v_mov_b32_e32 v15, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v6, v[14:17], s[2:3] offset:24 sc0 sc1
	s_mov_b64 s[4:5], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[2:3], v[16:17]
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
	v_readfirstlane_b32 s0, v10
	v_mov_b64_e32 v[4:5], 0
	;;#ASMSTART
	v_mov_b32 v11, v51
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v14, v52
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v15, v53
	;;#ASMEND
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v10
	s_and_saveexec_b64 s[4:5], s[0:1]
	s_cbranch_execz .LBB0_81
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
	s_cbranch_execz .LBB0_80
	s_mov_b64 s[8:9], 0
.LBB0_78:
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
	s_cbranch_execnz .LBB0_78
	s_or_b64 exec, exec, s[8:9]
	v_mov_b64_e32 v[4:5], v[0:1]
.LBB0_80:
	s_or_b64 exec, exec, s[6:7]
.LBB0_81:
	s_or_b64 exec, exec, s[4:5]
	v_mov_b32_e32 v13, 0
	global_load_dwordx2 v[6:7], v13, s[2:3] offset:40
	global_load_dwordx4 v[0:3], v13, s[2:3]
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
	s_cbranch_execz .LBB0_83
	v_mov_b64_e32 v[6:7], s[6:7]
	v_mov_b32_e32 v8, 2
	v_mov_b32_e32 v9, 1
	global_store_dwordx4 v[4:5], v[6:9], off offset:8
.LBB0_83:
	s_or_b64 exec, exec, s[10:11]
	s_lshl_b64 s[6:7], s[8:9], 12
	v_lshl_add_u64 v[6:7], v[2:3], 0, s[6:7]
	s_mov_b32 s8, 0
	v_mov_b32_e32 v16, 33
	v_mov_b32_e32 v17, v13
	v_mov_b32_e32 v18, v13
	v_mov_b32_e32 v19, v13
	v_readfirstlane_b32 s6, v6
	v_readfirstlane_b32 s7, v7
	s_mov_b32 s10, s8
	s_mov_b32 s11, s8
	s_mov_b32 s9, s8
	s_nop 1
	global_store_dwordx4 v12, v[16:19], s[6:7]
	s_nop 1
	v_mov_b64_e32 v[18:19], s[10:11]
	v_mov_b64_e32 v[16:17], s[8:9]
	global_store_dwordx4 v12, v[16:19], s[6:7] offset:16
	global_store_dwordx4 v12, v[16:19], s[6:7] offset:32
	global_store_dwordx4 v12, v[16:19], s[6:7] offset:48
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_91
	v_mov_b32_e32 v16, 0
	global_load_dwordx2 v[20:21], v16, s[2:3] offset:32 sc0 sc1
	global_load_dwordx2 v[2:3], v16, s[2:3] offset:40
	v_mov_b32_e32 v18, s4
	v_mov_b32_e32 v19, s5
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s8, v2
	v_readfirstlane_b32 s9, v3
	s_and_b64 s[8:9], s[8:9], s[4:5]
	s_mul_i32 s9, s9, 24
	s_mul_hi_u32 s10, s8, 24
	s_mul_i32 s8, s8, 24
	s_add_i32 s9, s10, s9
	v_lshl_add_u64 v[8:9], v[0:1], 0, s[8:9]
	global_store_dwordx2 v[8:9], v[20:21], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v16, v[18:21], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[2:3], v[20:21]
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_87
	s_mov_b64 s[10:11], 0
.LBB0_86:
	s_sleep 1
	global_store_dwordx2 v[8:9], v[2:3], off
	v_mov_b32_e32 v0, s4
	v_mov_b32_e32 v1, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[0:1], v16, v[0:3], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[0:1], v[2:3]
	s_or_b64 s[10:11], vcc, s[10:11]
	v_mov_b64_e32 v[2:3], v[0:1]
	s_andn2_b64 exec, exec, s[10:11]
	s_cbranch_execnz .LBB0_86
.LBB0_87:
	s_or_b64 exec, exec, s[8:9]
	v_mov_b32_e32 v3, 0
	global_load_dwordx2 v[0:1], v3, s[2:3] offset:16
	s_mov_b64 s[8:9], exec
	v_mbcnt_lo_u32_b32 v2, s8, 0
	v_mbcnt_hi_u32_b32 v2, s9, v2
	v_cmp_eq_u32_e32 vcc, 0, v2
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_89
	s_bcnt1_i32_b64 s8, s[8:9]
	v_mov_b32_e32 v2, s8
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[0:1], v[2:3], off offset:8 sc1
.LBB0_89:
	s_or_b64 exec, exec, s[10:11]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[2:3], v[0:1], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[2:3]
	s_cbranch_vccnz .LBB0_91
	global_load_dword v0, v[0:1], off offset:24
	v_mov_b32_e32 v1, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s8, v0
	s_and_b32 m0, s8, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[2:3], v[0:1], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_91:
	s_or_b64 exec, exec, s[6:7]
	v_lshl_add_u64 v[0:1], v[6:7], 0, v[12:13]
	s_branch .LBB0_95
.LBB0_92:
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s6, v2
	s_cmp_eq_u32 s6, 0
	s_cbranch_scc1 .LBB0_94
	s_sleep 1
	s_cbranch_execnz .LBB0_95
	s_branch .LBB0_97
.LBB0_94:
	s_branch .LBB0_97
.LBB0_95:
	v_mov_b32_e32 v2, 1
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_92
	global_load_dword v2, v[4:5], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v2, 1, v2
	s_branch .LBB0_92
.LBB0_97:
	global_load_dwordx2 v[0:1], v[0:1], off
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_100
	v_mov_b32_e32 v8, 0
	global_load_dwordx2 v[2:3], v8, s[2:3] offset:40
	global_load_dwordx2 v[18:19], v8, s[2:3] offset:24 sc0 sc1
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
	v_mov_b32_e32 v16, s0
	global_store_dwordx2 v[6:7], v[18:19], off
	v_mov_b32_e32 v17, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[4:5], v8, v[16:19], s[2:3] offset:24 sc0 sc1
	s_mov_b64 s[4:5], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[4:5], v[18:19]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_100
.LBB0_99:
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
	s_cbranch_execnz .LBB0_99
.LBB0_100:
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s0, v10
	v_mov_b64_e32 v[8:9], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v10
	s_and_saveexec_b64 s[4:5], s[0:1]
	s_cbranch_execz .LBB0_106
	v_mov_b32_e32 v4, 0
	global_load_dwordx2 v[18:19], v4, s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[2:3], v4, s[2:3] offset:40
	global_load_dwordx2 v[6:7], v4, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v18
	v_and_b32_e32 v3, v3, v19
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v5, v2, 24
	v_add_u32_e32 v3, v5, v3
	v_mul_lo_u32 v2, v2, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[6:7], 0, v[2:3]
	global_load_dwordx2 v[16:17], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[8:9], v4, v[16:19], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[8:9], v[18:19]
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_105
	s_mov_b64 s[8:9], 0
.LBB0_103:
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
	s_cbranch_execnz .LBB0_103
	s_or_b64 exec, exec, s[8:9]
	v_mov_b64_e32 v[8:9], v[2:3]
.LBB0_105:
	s_or_b64 exec, exec, s[6:7]
.LBB0_106:
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
	s_cbranch_execz .LBB0_108
	v_mov_b64_e32 v[16:17], s[6:7]
	v_mov_b32_e32 v18, 2
	v_mov_b32_e32 v19, 1
	global_store_dwordx4 v[8:9], v[16:19], off offset:8
.LBB0_108:
	s_or_b64 exec, exec, s[10:11]
	s_lshl_b64 s[6:7], s[8:9], 12
	v_lshl_add_u64 v[6:7], v[6:7], 0, s[6:7]
	v_and_b32_e32 v0, 0xffffff1f, v0
	v_or_b32_e32 v0, 0xa0, v0
	v_mov_b32_e32 v3, 0x41207265
	v_mov_b32_e32 v2, 0x7466415b
	v_readfirstlane_b32 s6, v6
	v_readfirstlane_b32 s7, v7
	v_mov_b32_e32 v16, 0x205d4444
	v_mov_b32_e32 v17, 0x2e253d41
	v_mov_b32_e32 v18, 0x202c6633
	v_mov_b32_e32 v19, 0x2e253d42
	v_mov_b32_e32 v20, 0xa6633
	global_store_dwordx4 v12, v[0:3], s[6:7]
	global_store_dwordx4 v12, v[16:19], s[6:7] offset:16
	v_mov_b32_e32 v21, v13
	v_mov_b32_e32 v0, v13
	v_mov_b32_e32 v19, 0x2e253d43
	v_mov_b32_e32 v1, v13
	v_mov_b32_e32 v2, v13
	v_mov_b32_e32 v3, v13
	global_store_dwordx4 v12, v[18:21], s[6:7] offset:32
	global_store_dwordx4 v12, v[0:3], s[6:7] offset:48
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_116
	v_mov_b32_e32 v16, 0
	global_load_dwordx2 v[20:21], v16, s[2:3] offset:32 sc0 sc1
	global_load_dwordx2 v[0:1], v16, s[2:3] offset:40
	v_mov_b32_e32 v18, s4
	v_mov_b32_e32 v19, s5
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s8, v0
	v_readfirstlane_b32 s9, v1
	s_and_b64 s[8:9], s[8:9], s[4:5]
	s_mul_i32 s9, s9, 24
	s_mul_hi_u32 s10, s8, 24
	s_mul_i32 s8, s8, 24
	s_add_i32 s9, s10, s9
	v_lshl_add_u64 v[4:5], v[4:5], 0, s[8:9]
	global_store_dwordx2 v[4:5], v[20:21], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v16, v[18:21], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[2:3], v[20:21]
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_112
	s_mov_b64 s[10:11], 0
.LBB0_111:
	s_sleep 1
	global_store_dwordx2 v[4:5], v[2:3], off
	v_mov_b32_e32 v0, s4
	v_mov_b32_e32 v1, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[0:1], v16, v[0:3], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[0:1], v[2:3]
	s_or_b64 s[10:11], vcc, s[10:11]
	v_mov_b64_e32 v[2:3], v[0:1]
	s_andn2_b64 exec, exec, s[10:11]
	s_cbranch_execnz .LBB0_111
.LBB0_112:
	s_or_b64 exec, exec, s[8:9]
	v_mov_b32_e32 v3, 0
	global_load_dwordx2 v[0:1], v3, s[2:3] offset:16
	s_mov_b64 s[8:9], exec
	v_mbcnt_lo_u32_b32 v2, s8, 0
	v_mbcnt_hi_u32_b32 v2, s9, v2
	v_cmp_eq_u32_e32 vcc, 0, v2
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_114
	s_bcnt1_i32_b64 s8, s[8:9]
	v_mov_b32_e32 v2, s8
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[0:1], v[2:3], off offset:8 sc1
.LBB0_114:
	s_or_b64 exec, exec, s[10:11]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[2:3], v[0:1], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[2:3]
	s_cbranch_vccnz .LBB0_116
	global_load_dword v0, v[0:1], off offset:24
	v_mov_b32_e32 v1, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s8, v0
	s_and_b32 m0, s8, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[2:3], v[0:1], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_116:
	s_or_b64 exec, exec, s[6:7]
	v_lshl_add_u64 v[0:1], v[6:7], 0, v[12:13]
	s_branch .LBB0_120
.LBB0_117:
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s6, v2
	s_cmp_eq_u32 s6, 0
	s_cbranch_scc1 .LBB0_119
	s_sleep 1
	s_cbranch_execnz .LBB0_120
	s_branch .LBB0_122
.LBB0_119:
	s_branch .LBB0_122
.LBB0_120:
	v_mov_b32_e32 v2, 1
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_117
	global_load_dword v2, v[8:9], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v2, 1, v2
	s_branch .LBB0_117
.LBB0_122:
	global_load_dwordx2 v[0:1], v[0:1], off
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_125
	v_mov_b32_e32 v8, 0
	global_load_dwordx2 v[2:3], v8, s[2:3] offset:40
	global_load_dwordx2 v[18:19], v8, s[2:3] offset:24 sc0 sc1
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
	v_mov_b32_e32 v16, s0
	global_store_dwordx2 v[6:7], v[18:19], off
	v_mov_b32_e32 v17, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[4:5], v8, v[16:19], s[2:3] offset:24 sc0 sc1
	s_mov_b64 s[4:5], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[4:5], v[18:19]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_125
.LBB0_124:
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
	s_cbranch_execnz .LBB0_124
.LBB0_125:
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s0, v10
	v_mov_b64_e32 v[8:9], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v10
	s_and_saveexec_b64 s[4:5], s[0:1]
	s_cbranch_execz .LBB0_131
	v_mov_b32_e32 v4, 0
	global_load_dwordx2 v[18:19], v4, s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[2:3], v4, s[2:3] offset:40
	global_load_dwordx2 v[6:7], v4, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v18
	v_and_b32_e32 v3, v3, v19
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v5, v2, 24
	v_add_u32_e32 v3, v5, v3
	v_mul_lo_u32 v2, v2, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[6:7], 0, v[2:3]
	global_load_dwordx2 v[16:17], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[8:9], v4, v[16:19], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[8:9], v[18:19]
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_130
	s_mov_b64 s[8:9], 0
.LBB0_128:
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
	s_cbranch_execnz .LBB0_128
	s_or_b64 exec, exec, s[8:9]
	v_mov_b64_e32 v[8:9], v[2:3]
.LBB0_130:
	s_or_b64 exec, exec, s[6:7]
.LBB0_131:
	s_or_b64 exec, exec, s[4:5]
	v_mov_b32_e32 v10, 0
	global_load_dwordx2 v[2:3], v10, s[2:3] offset:40
	global_load_dwordx4 v[4:7], v10, s[2:3]
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
	s_cbranch_execz .LBB0_133
	v_mov_b64_e32 v[16:17], s[6:7]
	v_mov_b32_e32 v18, 2
	v_mov_b32_e32 v19, 1
	global_store_dwordx4 v[8:9], v[16:19], off offset:8
.LBB0_133:
	s_or_b64 exec, exec, s[10:11]
	s_lshl_b64 s[6:7], s[8:9], 12
	v_lshl_add_u64 v[6:7], v[6:7], 0, s[6:7]
	v_and_b32_e32 v0, 0xffffff1d, v0
	s_mov_b32 s8, 0
	v_cvt_f64_f32_e32 v[2:3], v11
	v_or_b32_e32 v0, 0x62, v0
	v_readfirstlane_b32 s6, v6
	v_readfirstlane_b32 s7, v7
	s_mov_b32 s9, s8
	v_cvt_f64_f32_e32 v[16:17], v14
	v_cvt_f64_f32_e32 v[18:19], v15
	s_nop 1
	global_store_dwordx4 v12, v[0:3], s[6:7]
	global_store_dwordx4 v12, v[16:19], s[6:7] offset:16
	s_mov_b32 s10, s8
	s_mov_b32 s11, s8
	v_mov_b64_e32 v[0:1], s[8:9]
	v_mov_b64_e32 v[2:3], s[10:11]
	global_store_dwordx4 v12, v[0:3], s[6:7] offset:32
	global_store_dwordx4 v12, v[0:3], s[6:7] offset:48
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_141
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
	s_cbranch_execz .LBB0_137
	s_mov_b64 s[10:11], 0
.LBB0_136:
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
	s_cbranch_execnz .LBB0_136
.LBB0_137:
	s_or_b64 exec, exec, s[8:9]
	v_mov_b32_e32 v3, 0
	global_load_dwordx2 v[0:1], v3, s[2:3] offset:16
	s_mov_b64 s[8:9], exec
	v_mbcnt_lo_u32_b32 v2, s8, 0
	v_mbcnt_hi_u32_b32 v2, s9, v2
	v_cmp_eq_u32_e32 vcc, 0, v2
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_139
	s_bcnt1_i32_b64 s8, s[8:9]
	v_mov_b32_e32 v2, s8
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[0:1], v[2:3], off offset:8 sc1
.LBB0_139:
	s_or_b64 exec, exec, s[10:11]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[2:3], v[0:1], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[2:3]
	s_cbranch_vccnz .LBB0_141
	global_load_dword v0, v[0:1], off offset:24
	v_mov_b32_e32 v1, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s8, v0
	s_and_b32 m0, s8, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[2:3], v[0:1], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_141:
	s_or_b64 exec, exec, s[6:7]
	s_branch .LBB0_145
.LBB0_142:
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s6, v0
	s_cmp_eq_u32 s6, 0
	s_cbranch_scc1 .LBB0_144
	s_sleep 1
	s_cbranch_execnz .LBB0_145
	s_branch .LBB0_147
.LBB0_144:
	s_branch .LBB0_147
.LBB0_145:
	v_mov_b32_e32 v0, 1
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_142
	global_load_dword v0, v[8:9], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v0, 1, v0
	s_branch .LBB0_142
.LBB0_147:
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_150
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
	s_cbranch_execz .LBB0_150
.LBB0_149:
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
	s_cbranch_execnz .LBB0_149
.LBB0_150:
	s_or_b64 exec, exec, s[6:7]
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
		.amdhsa_next_free_vgpr 54
		.amdhsa_next_free_sgpr 20
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

	.set _Z9vectorAddPKfS0_Pfi.num_vgpr, 22
	.set _Z9vectorAddPKfS0_Pfi.num_agpr, 0
	.set _Z9vectorAddPKfS0_Pfi.numbered_sgpr, 12
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
  .sgpr_count: 18
  .sgpr_spill_count: 0
  .symbol: _Z9vectorAddPKfS0_Pfi.kd
  .uniform_work_group_size: 1
  .uses_dynamic_stack: false
  .vgpr_count: 22
  .vgpr_spill_count: 0
  .wavefront_size: 64
amdhsa.target: amdgcn-amd-amdhsa--gfx950
amdhsa.version:
- 1
- 2
...
...

	.end_amdgpu_metadata
