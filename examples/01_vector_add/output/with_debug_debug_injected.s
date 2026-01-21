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
	v_mov_b32 v71, v0
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
		global_load_dword v6, v[4:5], off      ; Ã¨Â¼ÂÃ¥ÂÂ¥ A[tid]
	;;#ASMEND
	;;#ASMSTART
		global_load_dword v7, v[2:3], off      ; Ã¨Â¼ÂÃ¥ÂÂ¥ B[tid]
	;;#ASMEND
	;;#ASMSTART
		v_lshl_add_u64 v[0:1], s[2:3], 0, v[0:1]
	;;#ASMEND
	;;#ASMSTART
		s_waitcnt vmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v64, v6
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v65, v7
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v64, v6
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v65, v7
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v66, v6
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v67, v7
	;;#ASMEND
	;;#ASMSTART
		v_add_f32_e32 v2, v6, v7               ; C = A + B
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v68, v6
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v69, v7
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v70, v2
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
	v_readfirstlane_b32 s0, v16
	v_mov_b64_e32 v[4:5], 0
	;;#ASMSTART
	v_mov_b32 v14, v64
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v15, v65
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v10, v64
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v11, v65
	;;#ASMEND
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v16
	;;#ASMSTART
	v_mov_b32 v12, v64
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v13, v65
	;;#ASMEND
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
	s_cbranch_execz .LBB0_91
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
	s_cbranch_execz .LBB0_87
	s_mov_b64 s[10:11], 0
.LBB0_86:
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
	v_lshl_add_u64 v[0:1], v[6:7], 0, v[18:19]
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
	v_readfirstlane_b32 s0, v16
	v_mov_b64_e32 v[8:9], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v16
	s_and_saveexec_b64 s[4:5], s[0:1]
	s_cbranch_execz .LBB0_106
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
	s_cbranch_execz .LBB0_108
	v_mov_b64_e32 v[20:21], s[6:7]
	v_mov_b32_e32 v22, 2
	v_mov_b32_e32 v23, 1
	global_store_dwordx4 v[8:9], v[20:23], off offset:8
.LBB0_108:
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
	s_cbranch_execz .LBB0_116
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
	s_cbranch_execz .LBB0_112
	s_mov_b64 s[10:11], 0
.LBB0_111:
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
	v_lshl_add_u64 v[0:1], v[6:7], 0, v[18:19]
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
	v_readfirstlane_b32 s0, v16
	v_mov_b64_e32 v[8:9], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v16
	s_and_saveexec_b64 s[4:5], s[0:1]
	s_cbranch_execz .LBB0_131
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
	s_cbranch_execz .LBB0_133
	v_mov_b64_e32 v[20:21], s[8:9]
	v_mov_b32_e32 v22, 2
	v_mov_b32_e32 v23, 1
	global_store_dwordx4 v[8:9], v[20:23], off offset:8
.LBB0_133:
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
	v_mov_b32 v10, v66
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v11, v67
	;;#ASMEND
	s_nop 0
	v_cmp_ge_f32_e32 vcc, 2.0, v10
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_226
	v_readfirstlane_b32 s0, v16
	v_mov_b64_e32 v[4:5], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v16
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_157
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
	s_cbranch_execz .LBB0_156
	s_mov_b64 s[10:11], 0
.LBB0_154:
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
	s_cbranch_execnz .LBB0_154
	s_or_b64 exec, exec, s[10:11]
	v_mov_b64_e32 v[4:5], v[0:1]
.LBB0_156:
	s_or_b64 exec, exec, s[8:9]
.LBB0_157:
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
	s_cbranch_execz .LBB0_159
	v_mov_b64_e32 v[6:7], s[8:9]
	v_mov_b32_e32 v8, 2
	v_mov_b32_e32 v9, 1
	global_store_dwordx4 v[4:5], v[6:9], off offset:8
.LBB0_159:
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
	s_cbranch_execz .LBB0_167
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
	s_cbranch_execz .LBB0_163
	s_mov_b64 s[12:13], 0
.LBB0_162:
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
	s_cbranch_execnz .LBB0_162
.LBB0_163:
	s_or_b64 exec, exec, s[10:11]
	v_mov_b32_e32 v3, 0
	global_load_dwordx2 v[0:1], v3, s[2:3] offset:16
	s_mov_b64 s[10:11], exec
	v_mbcnt_lo_u32_b32 v2, s10, 0
	v_mbcnt_hi_u32_b32 v2, s11, v2
	v_cmp_eq_u32_e32 vcc, 0, v2
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB0_165
	s_bcnt1_i32_b64 s10, s[10:11]
	v_mov_b32_e32 v2, s10
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[0:1], v[2:3], off offset:8 sc1
.LBB0_165:
	s_or_b64 exec, exec, s[12:13]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[2:3], v[0:1], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[2:3]
	s_cbranch_vccnz .LBB0_167
	global_load_dword v0, v[0:1], off offset:24
	v_mov_b32_e32 v1, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v0
	s_and_b32 m0, s10, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[2:3], v[0:1], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_167:
	s_or_b64 exec, exec, s[8:9]
	v_lshl_add_u64 v[0:1], v[6:7], 0, v[18:19]
	s_branch .LBB0_171
.LBB0_168:
	s_or_b64 exec, exec, s[8:9]
	v_readfirstlane_b32 s8, v2
	s_cmp_eq_u32 s8, 0
	s_cbranch_scc1 .LBB0_170
	s_sleep 1
	s_cbranch_execnz .LBB0_171
	s_branch .LBB0_173
.LBB0_170:
	s_branch .LBB0_173
.LBB0_171:
	v_mov_b32_e32 v2, 1
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_168
	global_load_dword v2, v[4:5], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v2, 1, v2
	s_branch .LBB0_168
.LBB0_173:
	global_load_dwordx2 v[0:1], v[0:1], off
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_176
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
	s_cbranch_execz .LBB0_176
.LBB0_175:
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
	s_cbranch_execnz .LBB0_175
.LBB0_176:
	s_or_b64 exec, exec, s[8:9]
	v_readfirstlane_b32 s0, v16
	v_mov_b64_e32 v[8:9], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v16
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_182
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
	s_cbranch_execz .LBB0_181
	s_mov_b64 s[10:11], 0
.LBB0_179:
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
	s_cbranch_execnz .LBB0_179
	s_or_b64 exec, exec, s[10:11]
	v_mov_b64_e32 v[8:9], v[2:3]
.LBB0_181:
	s_or_b64 exec, exec, s[8:9]
.LBB0_182:
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
	s_cbranch_execz .LBB0_184
	v_mov_b64_e32 v[12:13], s[8:9]
	v_mov_b32_e32 v14, 2
	v_mov_b32_e32 v15, 1
	global_store_dwordx4 v[8:9], v[12:15], off offset:8
.LBB0_184:
	s_or_b64 exec, exec, s[12:13]
	s_lshl_b64 s[8:9], s[10:11], 12
	v_lshl_add_u64 v[6:7], v[6:7], 0, s[8:9]
	v_and_b32_e32 v0, 0xffffff1f, v0
	v_or_b32_e32 v0, 0xa0, v0
	v_mov_b32_e32 v3, 0x2065726f
	v_mov_b32_e32 v2, 0x66654220
	v_readfirstlane_b32 s8, v6
	v_readfirstlane_b32 s9, v7
	s_nop 4
	global_store_dwordx4 v18, v[0:3], s[8:9]
	s_nop 1
	v_mov_b32_e32 v0, 0x20444441
	v_mov_b32_e32 v1, 0x323c4128
	v_mov_b32_e32 v2, 0x41203a29
	v_mov_b32_e32 v3, 0x332e253d
	global_store_dwordx4 v18, v[0:3], s[8:9] offset:16
	s_nop 1
	v_mov_b32_e32 v0, 0x42202c66
	v_mov_b32_e32 v1, 0x322e253d
	v_mov_b32_e32 v2, 0xa66
	v_mov_b32_e32 v3, v19
	global_store_dwordx4 v18, v[0:3], s[8:9] offset:32
	s_nop 1
	v_mov_b32_e32 v0, v19
	v_mov_b32_e32 v1, v19
	v_mov_b32_e32 v2, v19
	global_store_dwordx4 v18, v[0:3], s[8:9] offset:48
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_192
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
	s_cbranch_execz .LBB0_188
	s_mov_b64 s[12:13], 0
.LBB0_187:
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
	s_cbranch_execnz .LBB0_187
.LBB0_188:
	s_or_b64 exec, exec, s[10:11]
	v_mov_b32_e32 v3, 0
	global_load_dwordx2 v[0:1], v3, s[2:3] offset:16
	s_mov_b64 s[10:11], exec
	v_mbcnt_lo_u32_b32 v2, s10, 0
	v_mbcnt_hi_u32_b32 v2, s11, v2
	v_cmp_eq_u32_e32 vcc, 0, v2
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB0_190
	s_bcnt1_i32_b64 s10, s[10:11]
	v_mov_b32_e32 v2, s10
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[0:1], v[2:3], off offset:8 sc1
.LBB0_190:
	s_or_b64 exec, exec, s[12:13]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[2:3], v[0:1], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[2:3]
	s_cbranch_vccnz .LBB0_192
	global_load_dword v0, v[0:1], off offset:24
	v_mov_b32_e32 v1, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v0
	s_and_b32 m0, s10, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[2:3], v[0:1], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_192:
	s_or_b64 exec, exec, s[8:9]
	v_lshl_add_u64 v[0:1], v[6:7], 0, v[18:19]
	s_branch .LBB0_196
.LBB0_193:
	s_or_b64 exec, exec, s[8:9]
	v_readfirstlane_b32 s8, v2
	s_cmp_eq_u32 s8, 0
	s_cbranch_scc1 .LBB0_195
	s_sleep 1
	s_cbranch_execnz .LBB0_196
	s_branch .LBB0_198
.LBB0_195:
	s_branch .LBB0_198
.LBB0_196:
	v_mov_b32_e32 v2, 1
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_193
	global_load_dword v2, v[8:9], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v2, 1, v2
	s_branch .LBB0_193
.LBB0_198:
	global_load_dwordx2 v[0:1], v[0:1], off
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_201
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
	s_cbranch_execz .LBB0_201
.LBB0_200:
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
	s_cbranch_execnz .LBB0_200
.LBB0_201:
	s_or_b64 exec, exec, s[8:9]
	v_readfirstlane_b32 s0, v16
	v_mov_b64_e32 v[8:9], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v16
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_207
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
	s_cbranch_execz .LBB0_206
	s_mov_b64 s[10:11], 0
.LBB0_204:
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
	s_cbranch_execnz .LBB0_204
	s_or_b64 exec, exec, s[10:11]
	v_mov_b64_e32 v[8:9], v[2:3]
.LBB0_206:
	s_or_b64 exec, exec, s[8:9]
.LBB0_207:
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
	s_cbranch_execz .LBB0_209
	v_mov_b64_e32 v[12:13], s[8:9]
	v_mov_b32_e32 v14, 2
	v_mov_b32_e32 v15, 1
	global_store_dwordx4 v[8:9], v[12:15], off offset:8
.LBB0_209:
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
	s_cbranch_execz .LBB0_217
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
	s_cbranch_execz .LBB0_213
	s_mov_b64 s[12:13], 0
.LBB0_212:
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
	s_cbranch_execnz .LBB0_212
.LBB0_213:
	s_or_b64 exec, exec, s[10:11]
	v_mov_b32_e32 v3, 0
	global_load_dwordx2 v[0:1], v3, s[2:3] offset:16
	s_mov_b64 s[10:11], exec
	v_mbcnt_lo_u32_b32 v2, s10, 0
	v_mbcnt_hi_u32_b32 v2, s11, v2
	v_cmp_eq_u32_e32 vcc, 0, v2
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB0_215
	s_bcnt1_i32_b64 s10, s[10:11]
	v_mov_b32_e32 v2, s10
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[0:1], v[2:3], off offset:8 sc1
.LBB0_215:
	s_or_b64 exec, exec, s[12:13]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[2:3], v[0:1], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[2:3]
	s_cbranch_vccnz .LBB0_217
	global_load_dword v0, v[0:1], off offset:24
	v_mov_b32_e32 v1, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v0
	s_and_b32 m0, s10, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[2:3], v[0:1], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_217:
	s_or_b64 exec, exec, s[8:9]
	s_branch .LBB0_221
.LBB0_218:
	s_or_b64 exec, exec, s[8:9]
	v_readfirstlane_b32 s8, v0
	s_cmp_eq_u32 s8, 0
	s_cbranch_scc1 .LBB0_220
	s_sleep 1
	s_cbranch_execnz .LBB0_221
	s_branch .LBB0_223
.LBB0_220:
	s_branch .LBB0_223
.LBB0_221:
	v_mov_b32_e32 v0, 1
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_218
	global_load_dword v0, v[8:9], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v0, 1, v0
	s_branch .LBB0_218
.LBB0_223:
	s_and_b64 exec, exec, s[0:1]
	s_cbranch_execz .LBB0_226
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
	s_cbranch_execz .LBB0_226
.LBB0_225:
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
	s_cbranch_execnz .LBB0_225
.LBB0_226:
	s_or_b64 exec, exec, s[4:5]
	;;#ASMSTART
	v_mov_b32 v10, v68
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v11, v69
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v12, v70
	;;#ASMEND
	s_nop 0
	v_cmp_lt_f32_e32 vcc, 2.0, v10
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_302
	v_readfirstlane_b32 s0, v16
	v_mov_b64_e32 v[4:5], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v16
	s_and_saveexec_b64 s[6:7], s[0:1]
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
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_232
	s_mov_b64 s[10:11], 0
.LBB0_230:
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
	s_cbranch_execnz .LBB0_230
	s_or_b64 exec, exec, s[10:11]
	v_mov_b64_e32 v[4:5], v[0:1]
.LBB0_232:
	s_or_b64 exec, exec, s[8:9]
.LBB0_233:
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
	s_cbranch_execz .LBB0_235
	v_mov_b64_e32 v[6:7], s[8:9]
	v_mov_b32_e32 v8, 2
	v_mov_b32_e32 v9, 1
	global_store_dwordx4 v[4:5], v[6:9], off offset:8
.LBB0_235:
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
	s_cbranch_execz .LBB0_243
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
	s_cbranch_execz .LBB0_239
	s_mov_b64 s[12:13], 0
.LBB0_238:
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
	s_cbranch_execnz .LBB0_238
.LBB0_239:
	s_or_b64 exec, exec, s[10:11]
	v_mov_b32_e32 v3, 0
	global_load_dwordx2 v[0:1], v3, s[2:3] offset:16
	s_mov_b64 s[10:11], exec
	v_mbcnt_lo_u32_b32 v2, s10, 0
	v_mbcnt_hi_u32_b32 v2, s11, v2
	v_cmp_eq_u32_e32 vcc, 0, v2
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB0_241
	s_bcnt1_i32_b64 s10, s[10:11]
	v_mov_b32_e32 v2, s10
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[0:1], v[2:3], off offset:8 sc1
.LBB0_241:
	s_or_b64 exec, exec, s[12:13]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[2:3], v[0:1], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[2:3]
	s_cbranch_vccnz .LBB0_243
	global_load_dword v0, v[0:1], off offset:24
	v_mov_b32_e32 v1, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v0
	s_and_b32 m0, s10, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[2:3], v[0:1], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_243:
	s_or_b64 exec, exec, s[8:9]
	v_lshl_add_u64 v[0:1], v[6:7], 0, v[18:19]
	s_branch .LBB0_247
.LBB0_244:
	s_or_b64 exec, exec, s[8:9]
	v_readfirstlane_b32 s8, v2
	s_cmp_eq_u32 s8, 0
	s_cbranch_scc1 .LBB0_246
	s_sleep 1
	s_cbranch_execnz .LBB0_247
	s_branch .LBB0_249
.LBB0_246:
	s_branch .LBB0_249
.LBB0_247:
	v_mov_b32_e32 v2, 1
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_244
	global_load_dword v2, v[4:5], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v2, 1, v2
	s_branch .LBB0_244
.LBB0_249:
	global_load_dwordx2 v[0:1], v[0:1], off
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_252
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
	s_or_b64 s[6:7], vcc, s[6:7]
	v_mov_b64_e32 v[4:5], v[2:3]
	s_andn2_b64 exec, exec, s[6:7]
	s_cbranch_execnz .LBB0_251
.LBB0_252:
	s_or_b64 exec, exec, s[8:9]
	v_readfirstlane_b32 s0, v16
	v_mov_b64_e32 v[8:9], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v16
	s_and_saveexec_b64 s[6:7], s[0:1]
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
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_257
	s_mov_b64 s[10:11], 0
.LBB0_255:
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
	s_cbranch_execnz .LBB0_255
	s_or_b64 exec, exec, s[10:11]
	v_mov_b64_e32 v[8:9], v[2:3]
.LBB0_257:
	s_or_b64 exec, exec, s[8:9]
.LBB0_258:
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
	s_cbranch_execz .LBB0_260
	v_mov_b64_e32 v[20:21], s[8:9]
	v_mov_b32_e32 v22, 2
	v_mov_b32_e32 v23, 1
	global_store_dwordx4 v[8:9], v[20:23], off offset:8
.LBB0_260:
	s_or_b64 exec, exec, s[12:13]
	s_lshl_b64 s[8:9], s[10:11], 12
	v_lshl_add_u64 v[6:7], v[6:7], 0, s[8:9]
	v_and_b32_e32 v0, 0xffffff1f, v0
	v_or_b32_e32 v0, 0xa0, v0
	v_mov_b32_e32 v3, 0x76207265
	v_mov_b32_e32 v2, 0x74664120
	v_readfirstlane_b32 s8, v6
	v_readfirstlane_b32 s9, v7
	s_nop 4
	global_store_dwordx4 v18, v[0:3], s[8:9]
	s_nop 1
	v_mov_b32_e32 v0, 0x203d2036
	v_mov_b32_e32 v1, 0x66322e25
	v_mov_b32_e32 v2, 0x3776202c
	v_mov_b32_e32 v3, 0x25203d20
	global_store_dwordx4 v18, v[0:3], s[8:9] offset:16
	s_nop 1
	v_mov_b32_e32 v0, 0x2c66322e
	v_mov_b32_e32 v1, 0x253d4320
	v_mov_b32_e32 v2, 0xa66332e
	v_mov_b32_e32 v3, v19
	global_store_dwordx4 v18, v[0:3], s[8:9] offset:32
	s_nop 1
	v_mov_b32_e32 v0, v19
	v_mov_b32_e32 v1, v19
	v_mov_b32_e32 v2, v19
	global_store_dwordx4 v18, v[0:3], s[8:9] offset:48
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_268
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
	s_cbranch_execz .LBB0_264
	s_mov_b64 s[12:13], 0
.LBB0_263:
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
	s_cbranch_execnz .LBB0_263
.LBB0_264:
	s_or_b64 exec, exec, s[10:11]
	v_mov_b32_e32 v3, 0
	global_load_dwordx2 v[0:1], v3, s[2:3] offset:16
	s_mov_b64 s[10:11], exec
	v_mbcnt_lo_u32_b32 v2, s10, 0
	v_mbcnt_hi_u32_b32 v2, s11, v2
	v_cmp_eq_u32_e32 vcc, 0, v2
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB0_266
	s_bcnt1_i32_b64 s10, s[10:11]
	v_mov_b32_e32 v2, s10
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[0:1], v[2:3], off offset:8 sc1
.LBB0_266:
	s_or_b64 exec, exec, s[12:13]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[2:3], v[0:1], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[2:3]
	s_cbranch_vccnz .LBB0_268
	global_load_dword v0, v[0:1], off offset:24
	v_mov_b32_e32 v1, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v0
	s_and_b32 m0, s10, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[2:3], v[0:1], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_268:
	s_or_b64 exec, exec, s[8:9]
	v_lshl_add_u64 v[0:1], v[6:7], 0, v[18:19]
	s_branch .LBB0_272
.LBB0_269:
	s_or_b64 exec, exec, s[8:9]
	v_readfirstlane_b32 s8, v2
	s_cmp_eq_u32 s8, 0
	s_cbranch_scc1 .LBB0_271
	s_sleep 1
	s_cbranch_execnz .LBB0_272
	s_branch .LBB0_274
.LBB0_271:
	s_branch .LBB0_274
.LBB0_272:
	v_mov_b32_e32 v2, 1
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_269
	global_load_dword v2, v[8:9], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v2, 1, v2
	s_branch .LBB0_269
.LBB0_274:
	global_load_dwordx2 v[0:1], v[0:1], off
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_277
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
	s_or_b64 s[6:7], vcc, s[6:7]
	v_mov_b64_e32 v[4:5], v[2:3]
	s_andn2_b64 exec, exec, s[6:7]
	s_cbranch_execnz .LBB0_276
.LBB0_277:
	s_or_b64 exec, exec, s[8:9]
	v_readfirstlane_b32 s0, v16
	v_mov_b64_e32 v[8:9], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v16
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_283
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
	s_cbranch_execz .LBB0_282
	s_mov_b64 s[10:11], 0
.LBB0_280:
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
	s_cbranch_execnz .LBB0_280
	s_or_b64 exec, exec, s[10:11]
	v_mov_b64_e32 v[8:9], v[2:3]
.LBB0_282:
	s_or_b64 exec, exec, s[8:9]
.LBB0_283:
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
	s_cbranch_execz .LBB0_285
	v_mov_b64_e32 v[14:15], s[8:9]
	v_mov_b32_e32 v16, 2
	v_mov_b32_e32 v17, 1
	global_store_dwordx4 v[8:9], v[14:17], off offset:8
.LBB0_285:
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
	s_cbranch_execz .LBB0_293
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
	s_cbranch_execz .LBB0_289
	s_mov_b64 s[12:13], 0
.LBB0_288:
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
	s_cbranch_execnz .LBB0_288
.LBB0_289:
	s_or_b64 exec, exec, s[10:11]
	v_mov_b32_e32 v3, 0
	global_load_dwordx2 v[0:1], v3, s[2:3] offset:16
	s_mov_b64 s[10:11], exec
	v_mbcnt_lo_u32_b32 v2, s10, 0
	v_mbcnt_hi_u32_b32 v2, s11, v2
	v_cmp_eq_u32_e32 vcc, 0, v2
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB0_291
	s_bcnt1_i32_b64 s10, s[10:11]
	v_mov_b32_e32 v2, s10
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[0:1], v[2:3], off offset:8 sc1
.LBB0_291:
	s_or_b64 exec, exec, s[12:13]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[2:3], v[0:1], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[2:3]
	s_cbranch_vccnz .LBB0_293
	global_load_dword v0, v[0:1], off offset:24
	v_mov_b32_e32 v1, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v0
	s_and_b32 m0, s10, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[2:3], v[0:1], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_293:
	s_or_b64 exec, exec, s[8:9]
	s_branch .LBB0_297
.LBB0_294:
	s_or_b64 exec, exec, s[8:9]
	v_readfirstlane_b32 s8, v0
	s_cmp_eq_u32 s8, 0
	s_cbranch_scc1 .LBB0_296
	s_sleep 1
	s_cbranch_execnz .LBB0_297
	s_branch .LBB0_299
.LBB0_296:
	s_branch .LBB0_299
.LBB0_297:
	v_mov_b32_e32 v0, 1
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_294
	global_load_dword v0, v[8:9], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v0, 1, v0
	s_branch .LBB0_294
.LBB0_299:
	s_and_b64 exec, exec, s[0:1]
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
	s_or_b64 s[6:7], vcc, s[6:7]
	v_mov_b64_e32 v[2:3], v[0:1]
	s_andn2_b64 exec, exec, s[6:7]
	s_cbranch_execnz .LBB0_301
.LBB0_302:
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
		.amdhsa_next_free_vgpr 72
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
