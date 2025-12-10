	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.text
	.globl	vec_add
	.p2align	8
	.type	vec_add,@function
vec_add:
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
	        v_add_f32_e32 v2, v6, v7
	;;#ASMEND
	;;#ASMSTART
	        global_store_dword v[0:1], v2, off
	;;#ASMEND
	;;#ASMSTART
	.LBB0_2:
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v43, v2
	;;#ASMEND
	v_mbcnt_lo_u32_b32 v32, -1, 0
	;;#ASMSTART
	;;#ASMEND
	s_load_dwordx2 s[2:3], s[0:1], 0x70
	v_mbcnt_hi_u32_b32 v42, -1, v32
	v_mov_b64_e32 v[36:37], 0
	v_readfirstlane_b32 s0, v42
	s_nop 1
	v_cmp_eq_u32_e64 s[0:1], s0, v42
	s_and_saveexec_b64 s[4:5], s[0:1]
	s_cbranch_execz .LBB0_6
	v_mov_b32_e32 v38, 0
	s_waitcnt lgkmcnt(0)
	global_load_dwordx2 v[34:35], v38, s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[32:33], v38, s[2:3] offset:40
	global_load_dwordx2 v[36:37], v38, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v32, v32, v34
	v_and_b32_e32 v33, v33, v35
	v_mul_lo_u32 v33, v33, 24
	v_mul_hi_u32 v39, v32, 24
	v_add_u32_e32 v33, v39, v33
	v_mul_lo_u32 v32, v32, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[32:33], v[36:37], 0, v[32:33]
	global_load_dwordx2 v[32:33], v[32:33], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v38, v[32:35], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[36:37], v[34:35]
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_5
	s_mov_b64 s[8:9], 0
.LBB0_3:
	s_sleep 1
	global_load_dwordx2 v[32:33], v38, s[2:3] offset:40
	global_load_dwordx2 v[34:35], v38, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v32, v32, v36
	v_and_b32_e32 v39, v33, v37
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[32:33], s[10:11], v32, 24, v[34:35]
	v_mov_b32_e32 v34, v33
	v_mad_u64_u32 v[34:35], s[10:11], v39, 24, v[34:35]
	v_mov_b32_e32 v33, v34
	global_load_dwordx2 v[34:35], v[32:33], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v38, v[34:37], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[32:33], v[36:37]
	s_or_b64 s[8:9], vcc, s[8:9]
	v_mov_b64_e32 v[36:37], v[32:33]
	s_andn2_b64 exec, exec, s[8:9]
	s_cbranch_execnz .LBB0_3
	s_or_b64 exec, exec, s[8:9]
	v_mov_b64_e32 v[36:37], v[32:33]
.LBB0_5:
	s_or_b64 exec, exec, s[6:7]
.LBB0_6:
	s_or_b64 exec, exec, s[4:5]
	v_mov_b32_e32 v39, 0
	s_waitcnt lgkmcnt(0)
	global_load_dwordx2 v[40:41], v39, s[2:3] offset:40
	global_load_dwordx4 v[32:35], v39, s[2:3]
	v_readfirstlane_b32 s5, v37
	v_readfirstlane_b32 s4, v36
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s8, v40
	v_readfirstlane_b32 s9, v41
	s_and_b64 s[8:9], s[8:9], s[4:5]
	s_mul_i32 s10, s9, 24
	s_mul_hi_u32 s11, s8, 24
	s_add_i32 s11, s11, s10
	s_mul_i32 s10, s8, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[36:37], v[32:33], 0, s[10:11]
	s_and_saveexec_b64 s[10:11], s[0:1]
	s_cbranch_execz .LBB0_8
	v_mov_b64_e32 v[44:45], s[6:7]
	v_mov_b32_e32 v46, 2
	v_mov_b32_e32 v47, 1
	global_store_dwordx4 v[36:37], v[44:47], off offset:8
.LBB0_8:
	s_or_b64 exec, exec, s[10:11]
	s_lshl_b64 s[6:7], s[8:9], 12
	v_lshl_add_u64 v[40:41], v[34:35], 0, s[6:7]
	s_mov_b32 s8, 0
	v_lshlrev_b32_e32 v38, 6, v42
	v_mov_b32_e32 v44, 33
	v_mov_b32_e32 v45, v39
	v_mov_b32_e32 v46, v39
	v_mov_b32_e32 v47, v39
	v_readfirstlane_b32 s6, v40
	v_readfirstlane_b32 s7, v41
	s_mov_b32 s10, s8
	s_mov_b32 s11, s8
	s_mov_b32 s9, s8
	s_nop 1
	global_store_dwordx4 v38, v[44:47], s[6:7]
	s_nop 1
	v_mov_b64_e32 v[46:47], s[10:11]
	v_mov_b64_e32 v[44:45], s[8:9]
	global_store_dwordx4 v38, v[44:47], s[6:7] offset:16
	global_store_dwordx4 v38, v[44:47], s[6:7] offset:32
	global_store_dwordx4 v38, v[44:47], s[6:7] offset:48
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_16
	v_mov_b32_e32 v46, 0
	global_load_dwordx2 v[50:51], v46, s[2:3] offset:32 sc0 sc1
	global_load_dwordx2 v[34:35], v46, s[2:3] offset:40
	v_mov_b32_e32 v48, s4
	v_mov_b32_e32 v49, s5
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v34, s4, v34
	v_and_b32_e32 v35, s5, v35
	v_mul_lo_u32 v35, v35, 24
	v_mul_hi_u32 v44, v34, 24
	v_mul_lo_u32 v34, v34, 24
	v_add_u32_e32 v35, v44, v35
	v_lshl_add_u64 v[44:45], v[32:33], 0, v[34:35]
	global_store_dwordx2 v[44:45], v[50:51], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[34:35], v46, v[48:51], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[34:35], v[50:51]
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_12
	s_mov_b64 s[10:11], 0
.LBB0_11:
	s_sleep 1
	global_store_dwordx2 v[44:45], v[34:35], off
	v_mov_b32_e32 v32, s4
	v_mov_b32_e32 v33, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v46, v[32:35], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[32:33], v[34:35]
	s_or_b64 s[10:11], vcc, s[10:11]
	v_mov_b64_e32 v[34:35], v[32:33]
	s_andn2_b64 exec, exec, s[10:11]
	s_cbranch_execnz .LBB0_11
.LBB0_12:
	s_or_b64 exec, exec, s[8:9]
	v_mov_b32_e32 v35, 0
	global_load_dwordx2 v[32:33], v35, s[2:3] offset:16
	s_mov_b64 s[8:9], exec
	v_mbcnt_lo_u32_b32 v34, s8, 0
	v_mbcnt_hi_u32_b32 v34, s9, v34
	v_cmp_eq_u32_e32 vcc, 0, v34
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_14
	s_bcnt1_i32_b64 s8, s[8:9]
	v_mov_b32_e32 v34, s8
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[32:33], v[34:35], off offset:8 sc1
.LBB0_14:
	s_or_b64 exec, exec, s[10:11]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[34:35], v[32:33], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[34:35]
	s_cbranch_vccnz .LBB0_16
	global_load_dword v32, v[32:33], off offset:24
	v_mov_b32_e32 v33, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s8, v32
	s_and_b32 m0, s8, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[34:35], v[32:33], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_16:
	s_or_b64 exec, exec, s[6:7]
	v_lshl_add_u64 v[32:33], v[40:41], 0, v[38:39]
	s_branch .LBB0_20
.LBB0_17:
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s6, v34
	s_cmp_eq_u32 s6, 0
	s_cbranch_scc1 .LBB0_19
	s_sleep 1
	s_cbranch_execnz .LBB0_20
	s_branch .LBB0_22
.LBB0_19:
	s_branch .LBB0_22
.LBB0_20:
	v_mov_b32_e32 v34, 1
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_17
	global_load_dword v34, v[36:37], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v34, 1, v34
	s_branch .LBB0_17
.LBB0_22:
	global_load_dwordx2 v[32:33], v[32:33], off
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_25
	v_mov_b32_e32 v40, 0
	global_load_dwordx2 v[34:35], v40, s[2:3] offset:40
	global_load_dwordx2 v[46:47], v40, s[2:3] offset:24 sc0 sc1
	global_load_dwordx2 v[36:37], v40, s[2:3]
	s_waitcnt vmcnt(2)
	v_readfirstlane_b32 s8, v34
	v_readfirstlane_b32 s9, v35
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
	v_lshl_add_u64 v[38:39], v[36:37], 0, s[4:5]
	v_mov_b32_e32 v44, s0
	global_store_dwordx2 v[38:39], v[46:47], off
	v_mov_b32_e32 v45, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v40, v[44:47], s[2:3] offset:24 sc0 sc1
	s_mov_b64 s[4:5], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[36:37], v[46:47]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_25
.LBB0_24:
	s_sleep 1
	global_store_dwordx2 v[38:39], v[36:37], off
	v_mov_b32_e32 v34, s0
	v_mov_b32_e32 v35, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[34:35], v40, v[34:37], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[34:35], v[36:37]
	s_or_b64 s[4:5], vcc, s[4:5]
	v_mov_b64_e32 v[36:37], v[34:35]
	s_andn2_b64 exec, exec, s[4:5]
	s_cbranch_execnz .LBB0_24
.LBB0_25:
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s0, v42
	v_mov_b64_e32 v[40:41], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v42
	s_and_saveexec_b64 s[4:5], s[0:1]
	s_cbranch_execz .LBB0_31
	v_mov_b32_e32 v36, 0
	global_load_dwordx2 v[46:47], v36, s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[34:35], v36, s[2:3] offset:40
	global_load_dwordx2 v[38:39], v36, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v34, v34, v46
	v_and_b32_e32 v35, v35, v47
	v_mul_lo_u32 v35, v35, 24
	v_mul_hi_u32 v37, v34, 24
	v_add_u32_e32 v35, v37, v35
	v_mul_lo_u32 v34, v34, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[34:35], v[38:39], 0, v[34:35]
	global_load_dwordx2 v[44:45], v[34:35], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[40:41], v36, v[44:47], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[40:41], v[46:47]
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_30
	s_mov_b64 s[8:9], 0
.LBB0_28:
	s_sleep 1
	global_load_dwordx2 v[34:35], v36, s[2:3] offset:40
	global_load_dwordx2 v[38:39], v36, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v34, v34, v40
	v_and_b32_e32 v37, v35, v41
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[34:35], s[10:11], v34, 24, v[38:39]
	v_mov_b32_e32 v38, v35
	v_mad_u64_u32 v[38:39], s[10:11], v37, 24, v[38:39]
	v_mov_b32_e32 v35, v38
	global_load_dwordx2 v[38:39], v[34:35], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[34:35], v36, v[38:41], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[34:35], v[40:41]
	s_or_b64 s[8:9], vcc, s[8:9]
	v_mov_b64_e32 v[40:41], v[34:35]
	s_andn2_b64 exec, exec, s[8:9]
	s_cbranch_execnz .LBB0_28
	s_or_b64 exec, exec, s[8:9]
	v_mov_b64_e32 v[40:41], v[34:35]
.LBB0_30:
	s_or_b64 exec, exec, s[6:7]
.LBB0_31:
	s_or_b64 exec, exec, s[4:5]
	v_mov_b32_e32 v45, 0
	global_load_dwordx2 v[34:35], v45, s[2:3] offset:40
	global_load_dwordx4 v[36:39], v45, s[2:3]
	v_readfirstlane_b32 s5, v41
	v_readfirstlane_b32 s4, v40
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s8, v34
	v_readfirstlane_b32 s9, v35
	s_and_b64 s[8:9], s[8:9], s[4:5]
	s_mul_i32 s10, s9, 24
	s_mul_hi_u32 s11, s8, 24
	s_add_i32 s11, s11, s10
	s_mul_i32 s10, s8, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[40:41], v[36:37], 0, s[10:11]
	s_and_saveexec_b64 s[10:11], s[0:1]
	s_cbranch_execz .LBB0_33
	v_mov_b64_e32 v[46:47], s[6:7]
	v_mov_b32_e32 v48, 2
	v_mov_b32_e32 v49, 1
	global_store_dwordx4 v[40:41], v[46:49], off offset:8
.LBB0_33:
	s_or_b64 exec, exec, s[10:11]
	s_lshl_b64 s[6:7], s[8:9], 12
	v_lshl_add_u64 v[38:39], v[38:39], 0, s[6:7]
	v_and_b32_e32 v32, 0xffffff1f, v32
	v_or_b32_e32 v32, 0xa0, v32
	v_lshlrev_b32_e32 v44, 6, v42
	v_mov_b32_e32 v35, 0x65746e69
	v_mov_b32_e32 v34, 0x72702043
	v_readfirstlane_b32 s6, v38
	v_readfirstlane_b32 s7, v39
	s_nop 4
	global_store_dwordx4 v44, v[32:35], s[6:7]
	s_nop 1
	v_mov_b32_e32 v32, 0x6e692064
	v_mov_b32_e32 v33, 0x65646973
	v_mov_b32_e32 v34, 0x72656b20
	v_mov_b32_e32 v35, 0x206c656e
	global_store_dwordx4 v44, v[32:35], s[6:7] offset:16
	s_nop 1
	v_mov_b32_e32 v32, 0x3425203d
	v_mov_b32_e32 v33, 0xa66332e
	v_mov_b32_e32 v34, v45
	v_mov_b32_e32 v35, v45
	global_store_dwordx4 v44, v[32:35], s[6:7] offset:32
	s_nop 1
	v_mov_b32_e32 v32, v45
	v_mov_b32_e32 v33, v45
	global_store_dwordx4 v44, v[32:35], s[6:7] offset:48
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_41
	v_mov_b32_e32 v46, 0
	global_load_dwordx2 v[50:51], v46, s[2:3] offset:32 sc0 sc1
	global_load_dwordx2 v[32:33], v46, s[2:3] offset:40
	v_mov_b32_e32 v48, s4
	v_mov_b32_e32 v49, s5
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s8, v32
	v_readfirstlane_b32 s9, v33
	s_and_b64 s[8:9], s[8:9], s[4:5]
	s_mul_i32 s9, s9, 24
	s_mul_hi_u32 s10, s8, 24
	s_mul_i32 s8, s8, 24
	s_add_i32 s9, s10, s9
	v_lshl_add_u64 v[36:37], v[36:37], 0, s[8:9]
	global_store_dwordx2 v[36:37], v[50:51], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[34:35], v46, v[48:51], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[34:35], v[50:51]
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_37
	s_mov_b64 s[10:11], 0
.LBB0_36:
	s_sleep 1
	global_store_dwordx2 v[36:37], v[34:35], off
	v_mov_b32_e32 v32, s4
	v_mov_b32_e32 v33, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v46, v[32:35], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[32:33], v[34:35]
	s_or_b64 s[10:11], vcc, s[10:11]
	v_mov_b64_e32 v[34:35], v[32:33]
	s_andn2_b64 exec, exec, s[10:11]
	s_cbranch_execnz .LBB0_36
.LBB0_37:
	s_or_b64 exec, exec, s[8:9]
	v_mov_b32_e32 v35, 0
	global_load_dwordx2 v[32:33], v35, s[2:3] offset:16
	s_mov_b64 s[8:9], exec
	v_mbcnt_lo_u32_b32 v34, s8, 0
	v_mbcnt_hi_u32_b32 v34, s9, v34
	v_cmp_eq_u32_e32 vcc, 0, v34
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_39
	s_bcnt1_i32_b64 s8, s[8:9]
	v_mov_b32_e32 v34, s8
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[32:33], v[34:35], off offset:8 sc1
.LBB0_39:
	s_or_b64 exec, exec, s[10:11]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[34:35], v[32:33], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[34:35]
	s_cbranch_vccnz .LBB0_41
	global_load_dword v32, v[32:33], off offset:24
	v_mov_b32_e32 v33, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s8, v32
	s_and_b32 m0, s8, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[34:35], v[32:33], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_41:
	s_or_b64 exec, exec, s[6:7]
	v_lshl_add_u64 v[32:33], v[38:39], 0, v[44:45]
	s_branch .LBB0_45
.LBB0_42:
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s6, v34
	s_cmp_eq_u32 s6, 0
	s_cbranch_scc1 .LBB0_44
	s_sleep 1
	s_cbranch_execnz .LBB0_45
	s_branch .LBB0_47
.LBB0_44:
	s_branch .LBB0_47
.LBB0_45:
	v_mov_b32_e32 v34, 1
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_42
	global_load_dword v34, v[40:41], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v34, 1, v34
	s_branch .LBB0_42
.LBB0_47:
	global_load_dwordx2 v[32:33], v[32:33], off
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_50
	v_mov_b32_e32 v40, 0
	global_load_dwordx2 v[34:35], v40, s[2:3] offset:40
	global_load_dwordx2 v[48:49], v40, s[2:3] offset:24 sc0 sc1
	global_load_dwordx2 v[36:37], v40, s[2:3]
	s_waitcnt vmcnt(2)
	v_readfirstlane_b32 s8, v34
	v_readfirstlane_b32 s9, v35
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
	v_lshl_add_u64 v[38:39], v[36:37], 0, s[4:5]
	v_mov_b32_e32 v46, s0
	global_store_dwordx2 v[38:39], v[48:49], off
	v_mov_b32_e32 v47, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v40, v[46:49], s[2:3] offset:24 sc0 sc1
	s_mov_b64 s[4:5], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[36:37], v[48:49]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_50
.LBB0_49:
	s_sleep 1
	global_store_dwordx2 v[38:39], v[36:37], off
	v_mov_b32_e32 v34, s0
	v_mov_b32_e32 v35, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[34:35], v40, v[34:37], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[34:35], v[36:37]
	s_or_b64 s[4:5], vcc, s[4:5]
	v_mov_b64_e32 v[36:37], v[34:35]
	s_andn2_b64 exec, exec, s[4:5]
	s_cbranch_execnz .LBB0_49
.LBB0_50:
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s0, v42
	v_mov_b64_e32 v[40:41], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v42
	s_and_saveexec_b64 s[4:5], s[0:1]
	s_cbranch_execz .LBB0_56
	v_mov_b32_e32 v36, 0
	global_load_dwordx2 v[48:49], v36, s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[34:35], v36, s[2:3] offset:40
	global_load_dwordx2 v[38:39], v36, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v34, v34, v48
	v_and_b32_e32 v35, v35, v49
	v_mul_lo_u32 v35, v35, 24
	v_mul_hi_u32 v37, v34, 24
	v_add_u32_e32 v35, v37, v35
	v_mul_lo_u32 v34, v34, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[34:35], v[38:39], 0, v[34:35]
	global_load_dwordx2 v[46:47], v[34:35], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[40:41], v36, v[46:49], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[40:41], v[48:49]
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_55
	s_mov_b64 s[8:9], 0
.LBB0_53:
	s_sleep 1
	global_load_dwordx2 v[34:35], v36, s[2:3] offset:40
	global_load_dwordx2 v[38:39], v36, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v34, v34, v40
	v_and_b32_e32 v37, v35, v41
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[34:35], s[10:11], v34, 24, v[38:39]
	v_mov_b32_e32 v38, v35
	v_mad_u64_u32 v[38:39], s[10:11], v37, 24, v[38:39]
	v_mov_b32_e32 v35, v38
	global_load_dwordx2 v[38:39], v[34:35], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[34:35], v36, v[38:41], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[34:35], v[40:41]
	s_or_b64 s[8:9], vcc, s[8:9]
	v_mov_b64_e32 v[40:41], v[34:35]
	s_andn2_b64 exec, exec, s[8:9]
	s_cbranch_execnz .LBB0_53
	s_or_b64 exec, exec, s[8:9]
	v_mov_b64_e32 v[40:41], v[34:35]
.LBB0_55:
	s_or_b64 exec, exec, s[6:7]
.LBB0_56:
	s_or_b64 exec, exec, s[4:5]
	v_mov_b32_e32 v42, 0
	global_load_dwordx2 v[34:35], v42, s[2:3] offset:40
	global_load_dwordx4 v[36:39], v42, s[2:3]
	v_readfirstlane_b32 s5, v41
	v_readfirstlane_b32 s4, v40
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s8, v34
	v_readfirstlane_b32 s9, v35
	s_and_b64 s[8:9], s[8:9], s[4:5]
	s_mul_i32 s10, s9, 24
	s_mul_hi_u32 s11, s8, 24
	s_add_i32 s11, s11, s10
	s_mul_i32 s10, s8, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[40:41], v[36:37], 0, s[10:11]
	s_and_saveexec_b64 s[10:11], s[0:1]
	s_cbranch_execz .LBB0_58
	v_mov_b64_e32 v[46:47], s[6:7]
	v_mov_b32_e32 v48, 2
	v_mov_b32_e32 v49, 1
	global_store_dwordx4 v[40:41], v[46:49], off offset:8
.LBB0_58:
	s_or_b64 exec, exec, s[10:11]
	s_lshl_b64 s[6:7], s[8:9], 12
	v_lshl_add_u64 v[38:39], v[38:39], 0, s[6:7]
	s_movk_i32 s6, 0xff1d
	s_mov_b32 s8, 0
	v_cvt_f64_f32_e32 v[34:35], v43
	v_and_or_b32 v32, v32, s6, 34
	v_readfirstlane_b32 s6, v38
	v_readfirstlane_b32 s7, v39
	s_mov_b32 s10, s8
	s_mov_b32 s11, s8
	s_mov_b32 s9, s8
	s_nop 1
	global_store_dwordx4 v44, v[32:35], s[6:7]
	s_nop 1
	v_mov_b64_e32 v[34:35], s[10:11]
	v_mov_b64_e32 v[32:33], s[8:9]
	global_store_dwordx4 v44, v[32:35], s[6:7] offset:16
	global_store_dwordx4 v44, v[32:35], s[6:7] offset:32
	global_store_dwordx4 v44, v[32:35], s[6:7] offset:48
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_66
	v_mov_b32_e32 v38, 0
	global_load_dwordx2 v[44:45], v38, s[2:3] offset:32 sc0 sc1
	global_load_dwordx2 v[32:33], v38, s[2:3] offset:40
	v_mov_b32_e32 v42, s4
	v_mov_b32_e32 v43, s5
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s8, v32
	v_readfirstlane_b32 s9, v33
	s_and_b64 s[8:9], s[8:9], s[4:5]
	s_mul_i32 s9, s9, 24
	s_mul_hi_u32 s10, s8, 24
	s_mul_i32 s8, s8, 24
	s_add_i32 s9, s10, s9
	v_lshl_add_u64 v[36:37], v[36:37], 0, s[8:9]
	global_store_dwordx2 v[36:37], v[44:45], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[34:35], v38, v[42:45], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[34:35], v[44:45]
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_62
	s_mov_b64 s[10:11], 0
.LBB0_61:
	s_sleep 1
	global_store_dwordx2 v[36:37], v[34:35], off
	v_mov_b32_e32 v32, s4
	v_mov_b32_e32 v33, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v38, v[32:35], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[32:33], v[34:35]
	s_or_b64 s[10:11], vcc, s[10:11]
	v_mov_b64_e32 v[34:35], v[32:33]
	s_andn2_b64 exec, exec, s[10:11]
	s_cbranch_execnz .LBB0_61
.LBB0_62:
	s_or_b64 exec, exec, s[8:9]
	v_mov_b32_e32 v35, 0
	global_load_dwordx2 v[32:33], v35, s[2:3] offset:16
	s_mov_b64 s[8:9], exec
	v_mbcnt_lo_u32_b32 v34, s8, 0
	v_mbcnt_hi_u32_b32 v34, s9, v34
	v_cmp_eq_u32_e32 vcc, 0, v34
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_64
	s_bcnt1_i32_b64 s8, s[8:9]
	v_mov_b32_e32 v34, s8
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[32:33], v[34:35], off offset:8 sc1
.LBB0_64:
	s_or_b64 exec, exec, s[10:11]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[34:35], v[32:33], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[34:35]
	s_cbranch_vccnz .LBB0_66
	global_load_dword v32, v[32:33], off offset:24
	v_mov_b32_e32 v33, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s8, v32
	s_and_b32 m0, s8, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[34:35], v[32:33], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_66:
	s_or_b64 exec, exec, s[6:7]
	s_branch .LBB0_70
.LBB0_67:
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s6, v32
	s_cmp_eq_u32 s6, 0
	s_cbranch_scc1 .LBB0_69
	s_sleep 1
	s_cbranch_execnz .LBB0_70
	s_branch .LBB0_72
.LBB0_69:
	s_branch .LBB0_72
.LBB0_70:
	v_mov_b32_e32 v32, 1
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_67
	global_load_dword v32, v[40:41], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v32, 1, v32
	s_branch .LBB0_67
.LBB0_72:
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_75
	v_mov_b32_e32 v38, 0
	global_load_dwordx2 v[32:33], v38, s[2:3] offset:40
	global_load_dwordx2 v[42:43], v38, s[2:3] offset:24 sc0 sc1
	global_load_dwordx2 v[34:35], v38, s[2:3]
	s_waitcnt vmcnt(2)
	v_readfirstlane_b32 s8, v32
	v_readfirstlane_b32 s9, v33
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
	v_lshl_add_u64 v[36:37], v[34:35], 0, s[4:5]
	v_mov_b32_e32 v40, s0
	global_store_dwordx2 v[36:37], v[42:43], off
	v_mov_b32_e32 v41, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[34:35], v38, v[40:43], s[2:3] offset:24 sc0 sc1
	s_mov_b64 s[4:5], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[34:35], v[42:43]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_75
.LBB0_74:
	s_sleep 1
	global_store_dwordx2 v[36:37], v[34:35], off
	v_mov_b32_e32 v32, s0
	v_mov_b32_e32 v33, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v38, v[32:35], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[32:33], v[34:35]
	s_or_b64 s[4:5], vcc, s[4:5]
	v_mov_b64_e32 v[34:35], v[32:33]
	s_andn2_b64 exec, exec, s[4:5]
	s_cbranch_execnz .LBB0_74
.LBB0_75:
	s_or_b64 exec, exec, s[6:7]
	;;#ASMSTART
	;;#ASMEND
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel vec_add
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
		.amdhsa_next_free_vgpr 52
		.amdhsa_next_free_sgpr 12
		.amdhsa_accum_offset 52
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
	.size	vec_add, .Lfunc_end0-vec_add

	.set vec_add.num_vgpr, 52
	.set vec_add.num_agpr, 0
	.set vec_add.numbered_sgpr, 12
	.set vec_add.num_named_barrier, 0
	.set vec_add.private_seg_size, 0
	.set vec_add.uses_vcc, 1
	.set vec_add.uses_flat_scratch, 0
	.set vec_add.has_dyn_sized_stack, 0
	.set vec_add.has_recursion, 0
	.set vec_add.has_indirect_call, 0
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
  - .agpr_count:     0
    .args:
      - .address_space:  generic
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  generic
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  generic
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
      - .offset:         112
        .size:           8
        .value_kind:     hidden_hostcall_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 288
    .max_flat_workgroup_size: 256
    .name:           vec_add
    .private_segment_fixed_size: 0
    .sgpr_count:     18
    .sgpr_spill_count: 0
    .symbol:         vec_add.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     52
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
