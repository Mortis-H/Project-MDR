	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.text
	.globl	_Z9vectorAddPKfS0_Pfi
	.p2align	8
	.type	_Z9vectorAddPKfS0_Pfi,@function
_Z9vectorAddPKfS0_Pfi:
	v_mov_b32_e32 v8, v0
	v_cmp_eq_u32_e32 vcc, 0, v8
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
		v_add_f32_e32 v2, v6, v7               ; C = A + B
	;;#ASMEND
	;;#ASMSTART
		global_store_dword v[0:1], v2, off
	;;#ASMEND
	;;#ASMSTART
	s_mov_b64 s[0:1], s[18:19]
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v19, v6
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v22, v7
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v23, v6
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v24, v7
	;;#ASMEND
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_76
	s_load_dwordx2 s[20:21], s[0:1], 0x70
	v_mbcnt_lo_u32_b32 v8, -1, 0
	v_mbcnt_hi_u32_b32 v18, -1, v8
	v_mov_b64_e32 v[12:13], 0
	v_readfirstlane_b32 s0, v18
	s_nop 1
	v_cmp_eq_u32_e64 s[0:1], s0, v18
	s_and_saveexec_b64 s[22:23], s[0:1]
	s_cbranch_execz .LBB0_7
	v_mov_b32_e32 v14, 0
	s_waitcnt lgkmcnt(0)
	global_load_dwordx2 v[10:11], v14, s[20:21] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[8:9], v14, s[20:21] offset:40
	global_load_dwordx2 v[12:13], v14, s[20:21]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v8, v8, v10
	v_and_b32_e32 v9, v9, v11
	v_mul_lo_u32 v9, v9, 24
	v_mul_hi_u32 v15, v8, 24
	v_add_u32_e32 v9, v15, v9
	v_mul_lo_u32 v8, v8, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[8:9], v[12:13], 0, v[8:9]
	global_load_dwordx2 v[8:9], v[8:9], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[12:13], v14, v[8:11], s[20:21] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[12:13], v[10:11]
	s_and_saveexec_b64 s[24:25], vcc
	s_cbranch_execz .LBB0_6
	s_mov_b64 s[26:27], 0
.LBB0_4:
	s_sleep 1
	global_load_dwordx2 v[8:9], v14, s[20:21] offset:40
	global_load_dwordx2 v[10:11], v14, s[20:21]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v8, v8, v12
	v_and_b32_e32 v15, v9, v13
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[8:9], s[28:29], v8, 24, v[10:11]
	v_mov_b32_e32 v10, v9
	v_mad_u64_u32 v[10:11], s[28:29], v15, 24, v[10:11]
	v_mov_b32_e32 v9, v10
	global_load_dwordx2 v[10:11], v[8:9], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[8:9], v14, v[10:13], s[20:21] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[8:9], v[12:13]
	s_or_b64 s[26:27], vcc, s[26:27]
	v_mov_b64_e32 v[12:13], v[8:9]
	s_andn2_b64 exec, exec, s[26:27]
	s_cbranch_execnz .LBB0_4
	s_or_b64 exec, exec, s[26:27]
	v_mov_b64_e32 v[12:13], v[8:9]
.LBB0_6:
	s_or_b64 exec, exec, s[24:25]
.LBB0_7:
	s_or_b64 exec, exec, s[22:23]
	v_mov_b32_e32 v15, 0
	s_waitcnt lgkmcnt(0)
	global_load_dwordx2 v[16:17], v15, s[20:21] offset:40
	global_load_dwordx4 v[8:11], v15, s[20:21]
	v_readfirstlane_b32 s23, v13
	v_readfirstlane_b32 s22, v12
	s_mov_b64 s[24:25], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s26, v16
	v_readfirstlane_b32 s27, v17
	s_and_b64 s[26:27], s[26:27], s[22:23]
	s_mul_i32 s28, s27, 24
	s_mul_hi_u32 s29, s26, 24
	s_add_i32 s29, s29, s28
	s_mul_i32 s28, s26, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[12:13], v[8:9], 0, s[28:29]
	s_and_saveexec_b64 s[28:29], s[0:1]
	s_cbranch_execz .LBB0_9
	v_mov_b64_e32 v[26:27], s[24:25]
	v_mov_b32_e32 v28, 2
	v_mov_b32_e32 v29, 1
	global_store_dwordx4 v[12:13], v[26:29], off offset:8
.LBB0_9:
	s_or_b64 exec, exec, s[28:29]
	s_lshl_b64 s[24:25], s[26:27], 12
	v_lshl_add_u64 v[16:17], v[10:11], 0, s[24:25]
	s_mov_b32 s24, 0
	v_lshlrev_b32_e32 v14, 6, v18
	v_mov_b32_e32 v26, 33
	v_mov_b32_e32 v27, v15
	v_mov_b32_e32 v28, v15
	v_mov_b32_e32 v29, v15
	v_readfirstlane_b32 s28, v16
	v_readfirstlane_b32 s29, v17
	s_mov_b32 s26, s24
	s_mov_b32 s27, s24
	s_mov_b32 s25, s24
	s_nop 1
	global_store_dwordx4 v14, v[26:29], s[28:29]
	s_nop 1
	v_mov_b64_e32 v[28:29], s[26:27]
	v_mov_b64_e32 v[26:27], s[24:25]
	global_store_dwordx4 v14, v[26:29], s[28:29] offset:16
	global_store_dwordx4 v14, v[26:29], s[28:29] offset:32
	global_store_dwordx4 v14, v[26:29], s[28:29] offset:48
	s_and_saveexec_b64 s[24:25], s[0:1]
	s_cbranch_execz .LBB0_17
	v_mov_b32_e32 v25, 0
	global_load_dwordx2 v[28:29], v25, s[20:21] offset:32 sc0 sc1
	global_load_dwordx2 v[10:11], v25, s[20:21] offset:40
	v_mov_b32_e32 v26, s22
	v_mov_b32_e32 v27, s23
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v10, s22, v10
	v_and_b32_e32 v11, s23, v11
	v_mul_lo_u32 v11, v11, 24
	v_mul_hi_u32 v20, v10, 24
	v_mul_lo_u32 v10, v10, 24
	v_add_u32_e32 v11, v20, v11
	v_lshl_add_u64 v[20:21], v[8:9], 0, v[10:11]
	global_store_dwordx2 v[20:21], v[28:29], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v25, v[26:29], s[20:21] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[10:11], v[28:29]
	s_and_saveexec_b64 s[26:27], vcc
	s_cbranch_execz .LBB0_13
	s_mov_b64 s[28:29], 0
.LBB0_12:
	s_sleep 1
	global_store_dwordx2 v[20:21], v[10:11], off
	v_mov_b32_e32 v8, s22
	v_mov_b32_e32 v9, s23
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[8:9], v25, v[8:11], s[20:21] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[8:9], v[10:11]
	s_or_b64 s[28:29], vcc, s[28:29]
	v_mov_b64_e32 v[10:11], v[8:9]
	s_andn2_b64 exec, exec, s[28:29]
	s_cbranch_execnz .LBB0_12
.LBB0_13:
	s_or_b64 exec, exec, s[26:27]
	v_mov_b32_e32 v11, 0
	global_load_dwordx2 v[8:9], v11, s[20:21] offset:16
	s_mov_b64 s[26:27], exec
	v_mbcnt_lo_u32_b32 v10, s26, 0
	v_mbcnt_hi_u32_b32 v10, s27, v10
	v_cmp_eq_u32_e32 vcc, 0, v10
	s_and_saveexec_b64 s[28:29], vcc
	s_cbranch_execz .LBB0_15
	s_bcnt1_i32_b64 s26, s[26:27]
	v_mov_b32_e32 v10, s26
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[8:9], v[10:11], off offset:8 sc1
.LBB0_15:
	s_or_b64 exec, exec, s[28:29]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[10:11], v[8:9], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[10:11]
	s_cbranch_vccnz .LBB0_17
	global_load_dword v8, v[8:9], off offset:24
	v_mov_b32_e32 v9, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s26, v8
	s_and_b32 m0, s26, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[10:11], v[8:9], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_17:
	s_or_b64 exec, exec, s[24:25]
	v_lshl_add_u64 v[8:9], v[16:17], 0, v[14:15]
	s_branch .LBB0_21
.LBB0_18:
	s_or_b64 exec, exec, s[24:25]
	v_readfirstlane_b32 s24, v10
	s_cmp_eq_u32 s24, 0
	s_cbranch_scc1 .LBB0_20
	s_sleep 1
	s_cbranch_execnz .LBB0_21
	s_branch .LBB0_23
.LBB0_20:
	s_branch .LBB0_23
.LBB0_21:
	v_mov_b32_e32 v10, 1
	s_and_saveexec_b64 s[24:25], s[0:1]
	s_cbranch_execz .LBB0_18
	global_load_dword v10, v[12:13], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v10, 1, v10
	s_branch .LBB0_18
.LBB0_23:
	global_load_dwordx2 v[8:9], v[8:9], off
	s_and_saveexec_b64 s[24:25], s[0:1]
	s_cbranch_execz .LBB0_26
	v_mov_b32_e32 v16, 0
	global_load_dwordx2 v[10:11], v16, s[20:21] offset:40
	global_load_dwordx2 v[28:29], v16, s[20:21] offset:24 sc0 sc1
	global_load_dwordx2 v[12:13], v16, s[20:21]
	s_waitcnt vmcnt(2)
	v_readfirstlane_b32 s26, v10
	v_readfirstlane_b32 s27, v11
	s_add_u32 s28, s26, 1
	s_addc_u32 s29, s27, 0
	s_add_u32 s0, s28, s22
	s_addc_u32 s1, s29, s23
	s_cmp_eq_u64 s[0:1], 0
	s_cselect_b32 s1, s29, s1
	s_cselect_b32 s0, s28, s0
	s_and_b64 s[22:23], s[0:1], s[26:27]
	s_mul_i32 s23, s23, 24
	s_mul_hi_u32 s26, s22, 24
	s_mul_i32 s22, s22, 24
	s_add_i32 s23, s26, s23
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[14:15], v[12:13], 0, s[22:23]
	v_mov_b32_e32 v26, s0
	global_store_dwordx2 v[14:15], v[28:29], off
	v_mov_b32_e32 v27, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[12:13], v16, v[26:29], s[20:21] offset:24 sc0 sc1
	s_mov_b64 s[22:23], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[12:13], v[28:29]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_26
.LBB0_25:
	s_sleep 1
	global_store_dwordx2 v[14:15], v[12:13], off
	v_mov_b32_e32 v10, s0
	v_mov_b32_e32 v11, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v16, v[10:13], s[20:21] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[10:11], v[12:13]
	s_or_b64 s[22:23], vcc, s[22:23]
	v_mov_b64_e32 v[12:13], v[10:11]
	s_andn2_b64 exec, exec, s[22:23]
	s_cbranch_execnz .LBB0_25
.LBB0_26:
	s_or_b64 exec, exec, s[24:25]
	v_readfirstlane_b32 s0, v18
	v_mov_b64_e32 v[16:17], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v18
	s_and_saveexec_b64 s[22:23], s[0:1]
	s_cbranch_execz .LBB0_32
	v_mov_b32_e32 v12, 0
	global_load_dwordx2 v[28:29], v12, s[20:21] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[10:11], v12, s[20:21] offset:40
	global_load_dwordx2 v[14:15], v12, s[20:21]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v10, v10, v28
	v_and_b32_e32 v11, v11, v29
	v_mul_lo_u32 v11, v11, 24
	v_mul_hi_u32 v13, v10, 24
	v_add_u32_e32 v11, v13, v11
	v_mul_lo_u32 v10, v10, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[10:11], v[14:15], 0, v[10:11]
	global_load_dwordx2 v[26:27], v[10:11], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[16:17], v12, v[26:29], s[20:21] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[16:17], v[28:29]
	s_and_saveexec_b64 s[24:25], vcc
	s_cbranch_execz .LBB0_31
	s_mov_b64 s[26:27], 0
.LBB0_29:
	s_sleep 1
	global_load_dwordx2 v[10:11], v12, s[20:21] offset:40
	global_load_dwordx2 v[14:15], v12, s[20:21]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v10, v10, v16
	v_and_b32_e32 v13, v11, v17
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[10:11], s[28:29], v10, 24, v[14:15]
	v_mov_b32_e32 v14, v11
	v_mad_u64_u32 v[14:15], s[28:29], v13, 24, v[14:15]
	v_mov_b32_e32 v11, v14
	global_load_dwordx2 v[14:15], v[10:11], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v12, v[14:17], s[20:21] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[10:11], v[16:17]
	s_or_b64 s[26:27], vcc, s[26:27]
	v_mov_b64_e32 v[16:17], v[10:11]
	s_andn2_b64 exec, exec, s[26:27]
	s_cbranch_execnz .LBB0_29
	s_or_b64 exec, exec, s[26:27]
	v_mov_b64_e32 v[16:17], v[10:11]
.LBB0_31:
	s_or_b64 exec, exec, s[24:25]
.LBB0_32:
	s_or_b64 exec, exec, s[22:23]
	v_mov_b32_e32 v21, 0
	global_load_dwordx2 v[10:11], v21, s[20:21] offset:40
	global_load_dwordx4 v[12:15], v21, s[20:21]
	v_readfirstlane_b32 s23, v17
	v_readfirstlane_b32 s22, v16
	s_mov_b64 s[24:25], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s26, v10
	v_readfirstlane_b32 s27, v11
	s_and_b64 s[26:27], s[26:27], s[22:23]
	s_mul_i32 s28, s27, 24
	s_mul_hi_u32 s29, s26, 24
	s_add_i32 s29, s29, s28
	s_mul_i32 s28, s26, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[16:17], v[12:13], 0, s[28:29]
	s_and_saveexec_b64 s[28:29], s[0:1]
	s_cbranch_execz .LBB0_34
	v_mov_b64_e32 v[26:27], s[24:25]
	v_mov_b32_e32 v28, 2
	v_mov_b32_e32 v29, 1
	global_store_dwordx4 v[16:17], v[26:29], off offset:8
.LBB0_34:
	s_or_b64 exec, exec, s[28:29]
	s_lshl_b64 s[24:25], s[26:27], 12
	v_lshl_add_u64 v[14:15], v[14:15], 0, s[24:25]
	v_and_b32_e32 v8, 0xffffff1f, v8
	v_or_b32_e32 v8, 0x60, v8
	v_lshlrev_b32_e32 v20, 6, v18
	v_mov_b32_e32 v11, 0x3d42202c
	v_mov_b32_e32 v10, 0x66253d41
	v_readfirstlane_b32 s28, v14
	v_readfirstlane_b32 s29, v15
	s_mov_b32 s24, 0
	s_mov_b32 s25, s24
	s_mov_b32 s26, s24
	s_mov_b32 s27, s24
	s_nop 0
	global_store_dwordx4 v20, v[8:11], s[28:29]
	s_nop 1
	v_mov_b32_e32 v8, 0x202c6625
	v_mov_b32_e32 v9, 0x3d422a41
	v_mov_b32_e32 v10, 0xa6625
	v_mov_b32_e32 v11, v21
	global_store_dwordx4 v20, v[8:11], s[28:29] offset:16
	s_nop 1
	v_mov_b64_e32 v[8:9], s[24:25]
	v_mov_b64_e32 v[10:11], s[26:27]
	global_store_dwordx4 v20, v[8:11], s[28:29] offset:32
	global_store_dwordx4 v20, v[8:11], s[28:29] offset:48
	s_and_saveexec_b64 s[24:25], s[0:1]
	s_cbranch_execz .LBB0_42
	v_mov_b32_e32 v25, 0
	global_load_dwordx2 v[28:29], v25, s[20:21] offset:32 sc0 sc1
	global_load_dwordx2 v[8:9], v25, s[20:21] offset:40
	v_mov_b32_e32 v26, s22
	v_mov_b32_e32 v27, s23
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s26, v8
	v_readfirstlane_b32 s27, v9
	s_and_b64 s[26:27], s[26:27], s[22:23]
	s_mul_i32 s27, s27, 24
	s_mul_hi_u32 s28, s26, 24
	s_mul_i32 s26, s26, 24
	s_add_i32 s27, s28, s27
	v_lshl_add_u64 v[12:13], v[12:13], 0, s[26:27]
	global_store_dwordx2 v[12:13], v[28:29], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v25, v[26:29], s[20:21] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[10:11], v[28:29]
	s_and_saveexec_b64 s[26:27], vcc
	s_cbranch_execz .LBB0_38
	s_mov_b64 s[28:29], 0
.LBB0_37:
	s_sleep 1
	global_store_dwordx2 v[12:13], v[10:11], off
	v_mov_b32_e32 v8, s22
	v_mov_b32_e32 v9, s23
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[8:9], v25, v[8:11], s[20:21] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[8:9], v[10:11]
	s_or_b64 s[28:29], vcc, s[28:29]
	v_mov_b64_e32 v[10:11], v[8:9]
	s_andn2_b64 exec, exec, s[28:29]
	s_cbranch_execnz .LBB0_37
.LBB0_38:
	s_or_b64 exec, exec, s[26:27]
	v_mov_b32_e32 v11, 0
	global_load_dwordx2 v[8:9], v11, s[20:21] offset:16
	s_mov_b64 s[26:27], exec
	v_mbcnt_lo_u32_b32 v10, s26, 0
	v_mbcnt_hi_u32_b32 v10, s27, v10
	v_cmp_eq_u32_e32 vcc, 0, v10
	s_and_saveexec_b64 s[28:29], vcc
	s_cbranch_execz .LBB0_40
	s_bcnt1_i32_b64 s26, s[26:27]
	v_mov_b32_e32 v10, s26
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[8:9], v[10:11], off offset:8 sc1
.LBB0_40:
	s_or_b64 exec, exec, s[28:29]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[10:11], v[8:9], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[10:11]
	s_cbranch_vccnz .LBB0_42
	global_load_dword v8, v[8:9], off offset:24
	v_mov_b32_e32 v9, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s26, v8
	s_and_b32 m0, s26, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[10:11], v[8:9], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_42:
	s_or_b64 exec, exec, s[24:25]
	v_lshl_add_u64 v[8:9], v[14:15], 0, v[20:21]
	s_branch .LBB0_46
.LBB0_43:
	s_or_b64 exec, exec, s[24:25]
	v_readfirstlane_b32 s24, v10
	s_cmp_eq_u32 s24, 0
	s_cbranch_scc1 .LBB0_45
	s_sleep 1
	s_cbranch_execnz .LBB0_46
	s_branch .LBB0_48
.LBB0_45:
	s_branch .LBB0_48
.LBB0_46:
	v_mov_b32_e32 v10, 1
	s_and_saveexec_b64 s[24:25], s[0:1]
	s_cbranch_execz .LBB0_43
	global_load_dword v10, v[16:17], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v10, 1, v10
	s_branch .LBB0_43
.LBB0_48:
	global_load_dwordx2 v[8:9], v[8:9], off
	s_and_saveexec_b64 s[24:25], s[0:1]
	s_cbranch_execz .LBB0_51
	v_mov_b32_e32 v16, 0
	global_load_dwordx2 v[10:11], v16, s[20:21] offset:40
	global_load_dwordx2 v[28:29], v16, s[20:21] offset:24 sc0 sc1
	global_load_dwordx2 v[12:13], v16, s[20:21]
	s_waitcnt vmcnt(2)
	v_readfirstlane_b32 s26, v10
	v_readfirstlane_b32 s27, v11
	s_add_u32 s28, s26, 1
	s_addc_u32 s29, s27, 0
	s_add_u32 s0, s28, s22
	s_addc_u32 s1, s29, s23
	s_cmp_eq_u64 s[0:1], 0
	s_cselect_b32 s1, s29, s1
	s_cselect_b32 s0, s28, s0
	s_and_b64 s[22:23], s[0:1], s[26:27]
	s_mul_i32 s23, s23, 24
	s_mul_hi_u32 s26, s22, 24
	s_mul_i32 s22, s22, 24
	s_add_i32 s23, s26, s23
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[14:15], v[12:13], 0, s[22:23]
	v_mov_b32_e32 v26, s0
	global_store_dwordx2 v[14:15], v[28:29], off
	v_mov_b32_e32 v27, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[12:13], v16, v[26:29], s[20:21] offset:24 sc0 sc1
	s_mov_b64 s[22:23], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[12:13], v[28:29]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_51
.LBB0_50:
	s_sleep 1
	global_store_dwordx2 v[14:15], v[12:13], off
	v_mov_b32_e32 v10, s0
	v_mov_b32_e32 v11, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v16, v[10:13], s[20:21] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[10:11], v[12:13]
	s_or_b64 s[22:23], vcc, s[22:23]
	v_mov_b64_e32 v[12:13], v[10:11]
	s_andn2_b64 exec, exec, s[22:23]
	s_cbranch_execnz .LBB0_50
.LBB0_51:
	s_or_b64 exec, exec, s[24:25]
	v_readfirstlane_b32 s0, v18
	v_mov_b64_e32 v[16:17], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v18
	s_and_saveexec_b64 s[22:23], s[0:1]
	s_cbranch_execz .LBB0_57
	v_mov_b32_e32 v12, 0
	global_load_dwordx2 v[28:29], v12, s[20:21] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[10:11], v12, s[20:21] offset:40
	global_load_dwordx2 v[14:15], v12, s[20:21]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v10, v10, v28
	v_and_b32_e32 v11, v11, v29
	v_mul_lo_u32 v11, v11, 24
	v_mul_hi_u32 v13, v10, 24
	v_add_u32_e32 v11, v13, v11
	v_mul_lo_u32 v10, v10, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[10:11], v[14:15], 0, v[10:11]
	global_load_dwordx2 v[26:27], v[10:11], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[16:17], v12, v[26:29], s[20:21] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[16:17], v[28:29]
	s_and_saveexec_b64 s[24:25], vcc
	s_cbranch_execz .LBB0_56
	s_mov_b64 s[26:27], 0
.LBB0_54:
	s_sleep 1
	global_load_dwordx2 v[10:11], v12, s[20:21] offset:40
	global_load_dwordx2 v[14:15], v12, s[20:21]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v10, v10, v16
	v_and_b32_e32 v13, v11, v17
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[10:11], s[28:29], v10, 24, v[14:15]
	v_mov_b32_e32 v14, v11
	v_mad_u64_u32 v[14:15], s[28:29], v13, 24, v[14:15]
	v_mov_b32_e32 v11, v14
	global_load_dwordx2 v[14:15], v[10:11], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v12, v[14:17], s[20:21] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[10:11], v[16:17]
	s_or_b64 s[26:27], vcc, s[26:27]
	v_mov_b64_e32 v[16:17], v[10:11]
	s_andn2_b64 exec, exec, s[26:27]
	s_cbranch_execnz .LBB0_54
	s_or_b64 exec, exec, s[26:27]
	v_mov_b64_e32 v[16:17], v[10:11]
.LBB0_56:
	s_or_b64 exec, exec, s[24:25]
.LBB0_57:
	s_or_b64 exec, exec, s[22:23]
	v_mov_b32_e32 v18, 0
	global_load_dwordx2 v[10:11], v18, s[20:21] offset:40
	global_load_dwordx4 v[12:15], v18, s[20:21]
	v_readfirstlane_b32 s23, v17
	v_readfirstlane_b32 s22, v16
	s_mov_b64 s[24:25], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s26, v10
	v_readfirstlane_b32 s27, v11
	s_and_b64 s[26:27], s[26:27], s[22:23]
	s_mul_i32 s28, s27, 24
	s_mul_hi_u32 s29, s26, 24
	s_add_i32 s29, s29, s28
	s_mul_i32 s28, s26, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[16:17], v[12:13], 0, s[28:29]
	s_and_saveexec_b64 s[28:29], s[0:1]
	s_cbranch_execz .LBB0_59
	v_mov_b64_e32 v[26:27], s[24:25]
	v_mov_b32_e32 v28, 2
	v_mov_b32_e32 v29, 1
	global_store_dwordx4 v[16:17], v[26:29], off offset:8
.LBB0_59:
	s_or_b64 exec, exec, s[28:29]
	s_lshl_b64 s[24:25], s[26:27], 12
	v_lshl_add_u64 v[14:15], v[14:15], 0, s[24:25]
	v_and_b32_e32 v8, 0xffffff1d, v8
	s_mov_b32 s24, 0
	v_mul_f32_e32 v18, v23, v24
	v_cvt_f64_f32_e32 v[10:11], v19
	v_or_b32_e32 v8, 0x62, v8
	v_readfirstlane_b32 s28, v14
	v_readfirstlane_b32 s29, v15
	s_mov_b32 s25, s24
	v_cvt_f64_f32_e32 v[22:23], v22
	v_cvt_f64_f32_e32 v[24:25], v18
	s_nop 1
	global_store_dwordx4 v20, v[8:11], s[28:29]
	global_store_dwordx4 v20, v[22:25], s[28:29] offset:16
	s_mov_b32 s26, s24
	s_mov_b32 s27, s24
	v_mov_b64_e32 v[8:9], s[24:25]
	v_mov_b64_e32 v[10:11], s[26:27]
	global_store_dwordx4 v20, v[8:11], s[28:29] offset:32
	global_store_dwordx4 v20, v[8:11], s[28:29] offset:48
	s_and_saveexec_b64 s[24:25], s[0:1]
	s_cbranch_execz .LBB0_67
	v_mov_b32_e32 v14, 0
	global_load_dwordx2 v[20:21], v14, s[20:21] offset:32 sc0 sc1
	global_load_dwordx2 v[8:9], v14, s[20:21] offset:40
	v_mov_b32_e32 v18, s22
	v_mov_b32_e32 v19, s23
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s26, v8
	v_readfirstlane_b32 s27, v9
	s_and_b64 s[26:27], s[26:27], s[22:23]
	s_mul_i32 s27, s27, 24
	s_mul_hi_u32 s28, s26, 24
	s_mul_i32 s26, s26, 24
	s_add_i32 s27, s28, s27
	v_lshl_add_u64 v[12:13], v[12:13], 0, s[26:27]
	global_store_dwordx2 v[12:13], v[20:21], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v14, v[18:21], s[20:21] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[10:11], v[20:21]
	s_and_saveexec_b64 s[26:27], vcc
	s_cbranch_execz .LBB0_63
	s_mov_b64 s[28:29], 0
.LBB0_62:
	s_sleep 1
	global_store_dwordx2 v[12:13], v[10:11], off
	v_mov_b32_e32 v8, s22
	v_mov_b32_e32 v9, s23
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[8:9], v14, v[8:11], s[20:21] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[8:9], v[10:11]
	s_or_b64 s[28:29], vcc, s[28:29]
	v_mov_b64_e32 v[10:11], v[8:9]
	s_andn2_b64 exec, exec, s[28:29]
	s_cbranch_execnz .LBB0_62
.LBB0_63:
	s_or_b64 exec, exec, s[26:27]
	v_mov_b32_e32 v11, 0
	global_load_dwordx2 v[8:9], v11, s[20:21] offset:16
	s_mov_b64 s[26:27], exec
	v_mbcnt_lo_u32_b32 v10, s26, 0
	v_mbcnt_hi_u32_b32 v10, s27, v10
	v_cmp_eq_u32_e32 vcc, 0, v10
	s_and_saveexec_b64 s[28:29], vcc
	s_cbranch_execz .LBB0_65
	s_bcnt1_i32_b64 s26, s[26:27]
	v_mov_b32_e32 v10, s26
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[8:9], v[10:11], off offset:8 sc1
.LBB0_65:
	s_or_b64 exec, exec, s[28:29]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[10:11], v[8:9], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[10:11]
	s_cbranch_vccnz .LBB0_67
	global_load_dword v8, v[8:9], off offset:24
	v_mov_b32_e32 v9, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s26, v8
	s_and_b32 m0, s26, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[10:11], v[8:9], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_67:
	s_or_b64 exec, exec, s[24:25]
	s_branch .LBB0_71
.LBB0_68:
	s_or_b64 exec, exec, s[24:25]
	v_readfirstlane_b32 s24, v8
	s_cmp_eq_u32 s24, 0
	s_cbranch_scc1 .LBB0_70
	s_sleep 1
	s_cbranch_execnz .LBB0_71
	s_branch .LBB0_73
.LBB0_70:
	s_branch .LBB0_73
.LBB0_71:
	v_mov_b32_e32 v8, 1
	s_and_saveexec_b64 s[24:25], s[0:1]
	s_cbranch_execz .LBB0_68
	global_load_dword v8, v[16:17], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v8, 1, v8
	s_branch .LBB0_68
.LBB0_73:
	s_and_b64 exec, exec, s[0:1]
	s_cbranch_execz .LBB0_76
	v_mov_b32_e32 v14, 0
	global_load_dwordx2 v[8:9], v14, s[20:21] offset:40
	global_load_dwordx2 v[18:19], v14, s[20:21] offset:24 sc0 sc1
	global_load_dwordx2 v[10:11], v14, s[20:21]
	s_waitcnt vmcnt(2)
	v_readfirstlane_b32 s24, v8
	v_readfirstlane_b32 s25, v9
	s_add_u32 s26, s24, 1
	s_addc_u32 s27, s25, 0
	s_add_u32 s0, s26, s22
	s_addc_u32 s1, s27, s23
	s_cmp_eq_u64 s[0:1], 0
	s_cselect_b32 s1, s27, s1
	s_cselect_b32 s0, s26, s0
	s_and_b64 s[22:23], s[0:1], s[24:25]
	s_mul_i32 s23, s23, 24
	s_mul_hi_u32 s24, s22, 24
	s_mul_i32 s22, s22, 24
	s_add_i32 s23, s24, s23
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[12:13], v[10:11], 0, s[22:23]
	v_mov_b32_e32 v16, s0
	global_store_dwordx2 v[12:13], v[18:19], off
	v_mov_b32_e32 v17, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v14, v[16:19], s[20:21] offset:24 sc0 sc1
	s_mov_b64 s[22:23], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[10:11], v[18:19]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_76
.LBB0_75:
	s_sleep 1
	global_store_dwordx2 v[12:13], v[10:11], off
	v_mov_b32_e32 v8, s0
	v_mov_b32_e32 v9, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[8:9], v14, v[8:11], s[20:21] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[8:9], v[10:11]
	s_or_b64 s[22:23], vcc, s[22:23]
	v_mov_b64_e32 v[10:11], v[8:9]
	s_andn2_b64 exec, exec, s[22:23]
	s_cbranch_execnz .LBB0_75
.LBB0_76:
	s_or_b64 exec, exec, s[2:3]
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
		.amdhsa_next_free_vgpr 30
		.amdhsa_next_free_sgpr 30
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
	.size	_Z9vectorAddPKfS0_Pfi, .Lfunc_end0-_Z9vectorAddPKfS0_Pfi

	.set _Z9vectorAddPKfS0_Pfi.num_vgpr, 30
	.set _Z9vectorAddPKfS0_Pfi.num_agpr, 0
	.set _Z9vectorAddPKfS0_Pfi.numbered_sgpr, 30
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
  .sgpr_count: 36
  .sgpr_spill_count: 0
  .symbol: _Z9vectorAddPKfS0_Pfi.kd
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
