	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.text
	.globl	vec_add
	.p2align	8
	.type	vec_add,@function
vec_add:
	v_mov_b32_e32 v34, v0
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
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v50, v6
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v51, v7
	;;#ASMEND
	s_load_dwordx2 s[4:5], s[0:1], 0x70
	v_mbcnt_lo_u32_b32 v32, -1, 0
	v_mbcnt_hi_u32_b32 v46, -1, v32
	v_mov_b64_e32 v[40:41], 0
	v_readfirstlane_b32 s0, v46
	s_nop 1
	v_cmp_eq_u32_e64 s[0:1], s0, v46
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_cbranch_execz .LBB0_6
	v_mov_b32_e32 v35, 0
	s_waitcnt lgkmcnt(0)
	global_load_dwordx2 v[38:39], v35, s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[32:33], v35, s[4:5] offset:40
	global_load_dwordx2 v[36:37], v35, s[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v32, v32, v38
	v_and_b32_e32 v33, v33, v39
	v_mul_lo_u32 v33, v33, 24
	v_mul_hi_u32 v40, v32, 24
	v_add_u32_e32 v33, v40, v33
	v_mul_lo_u32 v32, v32, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[32:33], v[36:37], 0, v[32:33]
	global_load_dwordx2 v[36:37], v[32:33], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[40:41], v35, v[36:39], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[40:41], v[38:39]
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_5
	s_mov_b64 s[8:9], 0
.LBB0_3:
	s_sleep 1
	global_load_dwordx2 v[32:33], v35, s[4:5] offset:40
	global_load_dwordx2 v[36:37], v35, s[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v32, v32, v40
	v_and_b32_e32 v38, v33, v41
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[32:33], s[10:11], v32, 24, v[36:37]
	v_mov_b32_e32 v36, v33
	v_mad_u64_u32 v[36:37], s[10:11], v38, 24, v[36:37]
	v_mov_b32_e32 v33, v36
	global_load_dwordx2 v[38:39], v[32:33], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v35, v[38:41], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[32:33], v[40:41]
	s_or_b64 s[8:9], vcc, s[8:9]
	v_mov_b64_e32 v[40:41], v[32:33]
	s_andn2_b64 exec, exec, s[8:9]
	s_cbranch_execnz .LBB0_3
	s_or_b64 exec, exec, s[8:9]
	v_mov_b64_e32 v[40:41], v[32:33]
.LBB0_5:
	s_or_b64 exec, exec, s[6:7]
.LBB0_6:
	s_or_b64 exec, exec, s[2:3]
	v_mov_b32_e32 v43, 0
	s_waitcnt lgkmcnt(0)
	global_load_dwordx2 v[32:33], v43, s[4:5] offset:40
	global_load_dwordx4 v[36:39], v43, s[4:5]
	v_readfirstlane_b32 s3, v41
	v_readfirstlane_b32 s2, v40
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s8, v32
	v_readfirstlane_b32 s9, v33
	s_and_b64 s[8:9], s[8:9], s[2:3]
	s_mul_i32 s10, s9, 24
	s_mul_hi_u32 s11, s8, 24
	s_add_i32 s11, s11, s10
	s_mul_i32 s10, s8, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[32:33], v[36:37], 0, s[10:11]
	s_and_saveexec_b64 s[10:11], s[0:1]
	s_cbranch_execz .LBB0_8
	v_mov_b32_e32 v52, s6
	v_mov_b32_e32 v53, s7
	v_mov_b32_e32 v54, 2
	v_mov_b32_e32 v55, 1
	global_store_dwordx4 v[32:33], v[52:55], off offset:8
.LBB0_8:
	s_or_b64 exec, exec, s[10:11]
	s_lshl_b64 s[6:7], s[8:9], 12
	v_lshl_add_u64 v[40:41], v[38:39], 0, s[6:7]
	s_mov_b32 s8, 0
	v_lshlrev_b32_e32 v42, 6, v46
	v_mov_b32_e32 v52, 33
	v_mov_b32_e32 v53, v43
	v_mov_b32_e32 v54, v43
	v_mov_b32_e32 v55, v43
	v_readfirstlane_b32 s6, v40
	v_readfirstlane_b32 s7, v41
	s_mov_b32 s10, s8
	s_mov_b32 s11, s8
	s_mov_b32 s9, s8
	s_nop 1
	global_store_dwordx4 v42, v[52:55], s[6:7]
	s_nop 1
	v_mov_b64_e32 v[54:55], s[10:11]
	v_mov_b64_e32 v[52:53], s[8:9]
	global_store_dwordx4 v42, v[52:55], s[6:7] offset:16
	global_store_dwordx4 v42, v[52:55], s[6:7] offset:32
	global_store_dwordx4 v42, v[52:55], s[6:7] offset:48
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_16
	v_mov_b32_e32 v35, 0
	global_load_dwordx2 v[54:55], v35, s[4:5] offset:32 sc0 sc1
	global_load_dwordx2 v[38:39], v35, s[4:5] offset:40
	v_mov_b32_e32 v52, s2
	v_mov_b32_e32 v53, s3
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v38, s2, v38
	v_and_b32_e32 v39, s3, v39
	v_mul_lo_u32 v39, v39, 24
	v_mul_hi_u32 v44, v38, 24
	v_mul_lo_u32 v38, v38, 24
	v_add_u32_e32 v39, v44, v39
	v_lshl_add_u64 v[44:45], v[36:37], 0, v[38:39]
	global_store_dwordx2 v[44:45], v[54:55], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[38:39], v35, v[52:55], s[4:5] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[38:39], v[54:55]
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_12
	s_mov_b64 s[10:11], 0
.LBB0_11:
	s_sleep 1
	global_store_dwordx2 v[44:45], v[38:39], off
	v_mov_b32_e32 v36, s2
	v_mov_b32_e32 v37, s3
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v35, v[36:39], s[4:5] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[36:37], v[38:39]
	s_or_b64 s[10:11], vcc, s[10:11]
	v_mov_b64_e32 v[38:39], v[36:37]
	s_andn2_b64 exec, exec, s[10:11]
	s_cbranch_execnz .LBB0_11
.LBB0_12:
	s_or_b64 exec, exec, s[8:9]
	v_mov_b32_e32 v39, 0
	global_load_dwordx2 v[36:37], v39, s[4:5] offset:16
	s_mov_b64 s[8:9], exec
	v_mbcnt_lo_u32_b32 v35, s8, 0
	v_mbcnt_hi_u32_b32 v35, s9, v35
	v_cmp_eq_u32_e32 vcc, 0, v35
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_14
	s_bcnt1_i32_b64 s8, s[8:9]
	v_mov_b32_e32 v38, s8
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[36:37], v[38:39], off offset:8 sc1
.LBB0_14:
	s_or_b64 exec, exec, s[10:11]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[38:39], v[36:37], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[38:39]
	s_cbranch_vccnz .LBB0_16
	global_load_dword v36, v[36:37], off offset:24
	v_mov_b32_e32 v37, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s8, v36
	s_and_b32 m0, s8, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[38:39], v[36:37], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_16:
	s_or_b64 exec, exec, s[6:7]
	v_lshl_add_u64 v[36:37], v[40:41], 0, v[42:43]
	s_branch .LBB0_20
.LBB0_17:
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s6, v35
	s_cmp_eq_u32 s6, 0
	s_cbranch_scc1 .LBB0_19
	s_sleep 1
	s_cbranch_execnz .LBB0_20
	s_branch .LBB0_22
.LBB0_19:
	s_branch .LBB0_22
.LBB0_20:
	v_mov_b32_e32 v35, 1
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_17
	global_load_dword v35, v[32:33], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v35, 1, v35
	s_branch .LBB0_17
.LBB0_22:
	global_load_dwordx2 v[36:37], v[36:37], off
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_25
	v_mov_b32_e32 v35, 0
	global_load_dwordx2 v[32:33], v35, s[4:5] offset:40
	global_load_dwordx2 v[42:43], v35, s[4:5] offset:24 sc0 sc1
	global_load_dwordx2 v[38:39], v35, s[4:5]
	s_waitcnt vmcnt(2)
	v_readfirstlane_b32 s8, v32
	v_readfirstlane_b32 s9, v33
	s_add_u32 s10, s8, 1
	s_addc_u32 s11, s9, 0
	s_add_u32 s0, s10, s2
	s_addc_u32 s1, s11, s3
	s_cmp_eq_u64 s[0:1], 0
	s_cselect_b32 s1, s11, s1
	s_cselect_b32 s0, s10, s0
	s_and_b64 s[2:3], s[0:1], s[8:9]
	s_mul_i32 s3, s3, 24
	s_mul_hi_u32 s8, s2, 24
	s_mul_i32 s2, s2, 24
	s_add_i32 s3, s8, s3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[32:33], v[38:39], 0, s[2:3]
	v_mov_b32_e32 v40, s0
	global_store_dwordx2 v[32:33], v[42:43], off
	v_mov_b32_e32 v41, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[40:41], v35, v[40:43], s[4:5] offset:24 sc0 sc1
	s_mov_b64 s[2:3], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[40:41], v[42:43]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_25
.LBB0_24:
	s_sleep 1
	global_store_dwordx2 v[32:33], v[40:41], off
	v_mov_b32_e32 v38, s0
	v_mov_b32_e32 v39, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[38:39], v35, v[38:41], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[38:39], v[40:41]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[40:41], v[38:39]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_24
.LBB0_25:
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s0, v46
	v_mov_b64_e32 v[44:45], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v46
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_cbranch_execz .LBB0_31
	v_mov_b32_e32 v35, 0
	global_load_dwordx2 v[40:41], v35, s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[32:33], v35, s[4:5] offset:40
	global_load_dwordx2 v[38:39], v35, s[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v32, v32, v40
	v_and_b32_e32 v33, v33, v41
	v_mul_lo_u32 v33, v33, 24
	v_mul_hi_u32 v42, v32, 24
	v_add_u32_e32 v33, v42, v33
	v_mul_lo_u32 v32, v32, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[32:33], v[38:39], 0, v[32:33]
	global_load_dwordx2 v[38:39], v[32:33], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[44:45], v35, v[38:41], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[44:45], v[40:41]
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_30
	s_mov_b64 s[8:9], 0
.LBB0_28:
	s_sleep 1
	global_load_dwordx2 v[32:33], v35, s[4:5] offset:40
	global_load_dwordx2 v[38:39], v35, s[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v32, v32, v44
	v_and_b32_e32 v40, v33, v45
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[32:33], s[10:11], v32, 24, v[38:39]
	v_mov_b32_e32 v38, v33
	v_mad_u64_u32 v[38:39], s[10:11], v40, 24, v[38:39]
	v_mov_b32_e32 v33, v38
	global_load_dwordx2 v[42:43], v[32:33], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v35, v[42:45], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[32:33], v[44:45]
	s_or_b64 s[8:9], vcc, s[8:9]
	v_mov_b64_e32 v[44:45], v[32:33]
	s_andn2_b64 exec, exec, s[8:9]
	s_cbranch_execnz .LBB0_28
	s_or_b64 exec, exec, s[8:9]
	v_mov_b64_e32 v[44:45], v[32:33]
.LBB0_30:
	s_or_b64 exec, exec, s[6:7]
.LBB0_31:
	s_or_b64 exec, exec, s[2:3]
	v_mov_b32_e32 v49, 0
	global_load_dwordx2 v[32:33], v49, s[4:5] offset:40
	global_load_dwordx4 v[40:43], v49, s[4:5]
	v_readfirstlane_b32 s3, v45
	v_readfirstlane_b32 s2, v44
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s8, v32
	v_readfirstlane_b32 s9, v33
	s_and_b64 s[8:9], s[8:9], s[2:3]
	s_mul_i32 s10, s9, 24
	s_mul_hi_u32 s11, s8, 24
	s_add_i32 s11, s11, s10
	s_mul_i32 s10, s8, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[32:33], v[40:41], 0, s[10:11]
	s_and_saveexec_b64 s[10:11], s[0:1]
	s_cbranch_execz .LBB0_33
	v_mov_b32_e32 v52, s6
	v_mov_b32_e32 v53, s7
	v_mov_b32_e32 v54, 2
	v_mov_b32_e32 v55, 1
	global_store_dwordx4 v[32:33], v[52:55], off offset:8
.LBB0_33:
	s_or_b64 exec, exec, s[10:11]
	s_lshl_b64 s[6:7], s[8:9], 12
	v_lshl_add_u64 v[42:43], v[42:43], 0, s[6:7]
	v_or_b32_e32 v36, 0xe0, v36
	v_lshlrev_b32_e32 v48, 6, v46
	v_mov_b32_e32 v39, 0x6425203d
	v_mov_b32_e32 v38, 0x20444954
	v_readfirstlane_b32 s6, v42
	v_readfirstlane_b32 s7, v43
	s_nop 4
	global_store_dwordx4 v48, v[36:39], s[6:7]
	s_nop 1
	v_mov_b32_e32 v36, 0x4c46202c
	v_mov_b32_e32 v37, 0x3d204741
	v_mov_b32_e32 v38, 0x2c642520
	v_mov_b32_e32 v39, 0x6e6f6320
	global_store_dwordx4 v48, v[36:39], s[6:7] offset:16
	s_nop 1
	v_mov_b32_e32 v36, 0x69746964
	v_mov_b32_e32 v37, 0x3d206e6f
	v_mov_b32_e32 v38, 0x2c716520
	v_mov_b32_e32 v39, 0x5f736920
	global_store_dwordx4 v48, v[36:39], s[6:7] offset:32
	s_nop 1
	v_mov_b32_e32 v36, 0x69736f70
	v_mov_b32_e32 v37, 0x65766974
	v_mov_b32_e32 v38, 0x25203d20
	v_mov_b32_e32 v39, 0xa64
	global_store_dwordx4 v48, v[36:39], s[6:7] offset:48
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_41
	v_mov_b32_e32 v35, 0
	global_load_dwordx2 v[54:55], v35, s[4:5] offset:32 sc0 sc1
	global_load_dwordx2 v[36:37], v35, s[4:5] offset:40
	v_mov_b32_e32 v52, s2
	v_mov_b32_e32 v53, s3
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s8, v36
	v_readfirstlane_b32 s9, v37
	s_and_b64 s[8:9], s[8:9], s[2:3]
	s_mul_i32 s9, s9, 24
	s_mul_hi_u32 s10, s8, 24
	s_mul_i32 s8, s8, 24
	s_add_i32 s9, s10, s9
	v_lshl_add_u64 v[40:41], v[40:41], 0, s[8:9]
	global_store_dwordx2 v[40:41], v[54:55], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[38:39], v35, v[52:55], s[4:5] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[38:39], v[54:55]
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_37
	s_mov_b64 s[10:11], 0
.LBB0_36:
	s_sleep 1
	global_store_dwordx2 v[40:41], v[38:39], off
	v_mov_b32_e32 v36, s2
	v_mov_b32_e32 v37, s3
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v35, v[36:39], s[4:5] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[36:37], v[38:39]
	s_or_b64 s[10:11], vcc, s[10:11]
	v_mov_b64_e32 v[38:39], v[36:37]
	s_andn2_b64 exec, exec, s[10:11]
	s_cbranch_execnz .LBB0_36
.LBB0_37:
	s_or_b64 exec, exec, s[8:9]
	v_mov_b32_e32 v39, 0
	global_load_dwordx2 v[36:37], v39, s[4:5] offset:16
	s_mov_b64 s[8:9], exec
	v_mbcnt_lo_u32_b32 v35, s8, 0
	v_mbcnt_hi_u32_b32 v35, s9, v35
	v_cmp_eq_u32_e32 vcc, 0, v35
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_39
	s_bcnt1_i32_b64 s8, s[8:9]
	v_mov_b32_e32 v38, s8
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[36:37], v[38:39], off offset:8 sc1
.LBB0_39:
	s_or_b64 exec, exec, s[10:11]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[38:39], v[36:37], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[38:39]
	s_cbranch_vccnz .LBB0_41
	global_load_dword v36, v[36:37], off offset:24
	v_mov_b32_e32 v37, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s8, v36
	s_and_b32 m0, s8, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[38:39], v[36:37], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_41:
	s_or_b64 exec, exec, s[6:7]
	v_lshl_add_u64 v[36:37], v[42:43], 0, v[48:49]
	s_branch .LBB0_45
.LBB0_42:
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s6, v35
	s_cmp_eq_u32 s6, 0
	s_cbranch_scc1 .LBB0_44
	s_sleep 1
	s_cbranch_execnz .LBB0_45
	s_branch .LBB0_47
.LBB0_44:
	s_branch .LBB0_47
.LBB0_45:
	v_mov_b32_e32 v35, 1
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_42
	global_load_dword v35, v[32:33], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v35, 1, v35
	s_branch .LBB0_42
.LBB0_47:
	global_load_dwordx2 v[32:33], v[36:37], off
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_50
	v_mov_b32_e32 v35, 0
	global_load_dwordx2 v[36:37], v35, s[4:5] offset:40
	global_load_dwordx2 v[44:45], v35, s[4:5] offset:24 sc0 sc1
	global_load_dwordx2 v[38:39], v35, s[4:5]
	s_waitcnt vmcnt(2)
	v_readfirstlane_b32 s8, v36
	v_readfirstlane_b32 s9, v37
	s_add_u32 s10, s8, 1
	s_addc_u32 s11, s9, 0
	s_add_u32 s0, s10, s2
	s_addc_u32 s1, s11, s3
	s_cmp_eq_u64 s[0:1], 0
	s_cselect_b32 s1, s11, s1
	s_cselect_b32 s0, s10, s0
	s_and_b64 s[2:3], s[0:1], s[8:9]
	s_mul_i32 s3, s3, 24
	s_mul_hi_u32 s8, s2, 24
	s_mul_i32 s2, s2, 24
	s_add_i32 s3, s8, s3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[40:41], v[38:39], 0, s[2:3]
	v_mov_b32_e32 v42, s0
	global_store_dwordx2 v[40:41], v[44:45], off
	v_mov_b32_e32 v43, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[38:39], v35, v[42:45], s[4:5] offset:24 sc0 sc1
	s_mov_b64 s[2:3], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[38:39], v[44:45]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_50
.LBB0_49:
	s_sleep 1
	global_store_dwordx2 v[40:41], v[38:39], off
	v_mov_b32_e32 v36, s0
	v_mov_b32_e32 v37, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v35, v[36:39], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[36:37], v[38:39]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[38:39], v[36:37]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_49
.LBB0_50:
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s0, v46
	v_mov_b64_e32 v[40:41], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v46
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_cbranch_execz .LBB0_56
	v_mov_b32_e32 v35, 0
	global_load_dwordx2 v[38:39], v35, s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[36:37], v35, s[4:5] offset:40
	global_load_dwordx2 v[40:41], v35, s[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v36, v36, v38
	v_and_b32_e32 v37, v37, v39
	v_mul_lo_u32 v37, v37, 24
	v_mul_hi_u32 v42, v36, 24
	v_add_u32_e32 v37, v42, v37
	v_mul_lo_u32 v36, v36, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[36:37], v[40:41], 0, v[36:37]
	global_load_dwordx2 v[36:37], v[36:37], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[40:41], v35, v[36:39], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[40:41], v[38:39]
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_55
	s_mov_b64 s[8:9], 0
.LBB0_53:
	s_sleep 1
	global_load_dwordx2 v[36:37], v35, s[4:5] offset:40
	global_load_dwordx2 v[38:39], v35, s[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v36, v36, v40
	v_and_b32_e32 v42, v37, v41
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[36:37], s[10:11], v36, 24, v[38:39]
	v_mov_b32_e32 v38, v37
	v_mad_u64_u32 v[38:39], s[10:11], v42, 24, v[38:39]
	v_mov_b32_e32 v37, v38
	global_load_dwordx2 v[38:39], v[36:37], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v35, v[38:41], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[36:37], v[40:41]
	s_or_b64 s[8:9], vcc, s[8:9]
	v_mov_b64_e32 v[40:41], v[36:37]
	s_andn2_b64 exec, exec, s[8:9]
	s_cbranch_execnz .LBB0_53
	s_or_b64 exec, exec, s[8:9]
	v_mov_b64_e32 v[40:41], v[36:37]
.LBB0_55:
	s_or_b64 exec, exec, s[6:7]
.LBB0_56:
	s_or_b64 exec, exec, s[2:3]
	v_mov_b32_e32 v35, 0
	global_load_dwordx2 v[42:43], v35, s[4:5] offset:40
	global_load_dwordx4 v[36:39], v35, s[4:5]
	v_readfirstlane_b32 s7, v41
	v_readfirstlane_b32 s6, v40
	s_mov_b64 s[2:3], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s8, v42
	v_readfirstlane_b32 s9, v43
	s_and_b64 s[8:9], s[8:9], s[6:7]
	s_mul_i32 s10, s9, 24
	s_mul_hi_u32 s11, s8, 24
	s_add_i32 s11, s11, s10
	s_mul_i32 s10, s8, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[40:41], v[36:37], 0, s[10:11]
	s_and_saveexec_b64 s[10:11], s[0:1]
	s_cbranch_execz .LBB0_58
	v_mov_b32_e32 v42, s2
	v_mov_b32_e32 v43, s3
	v_mov_b32_e32 v44, 2
	v_mov_b32_e32 v45, 1
	global_store_dwordx4 v[40:41], v[42:45], off offset:8
.LBB0_58:
	s_or_b64 exec, exec, s[10:11]
	s_lshl_b64 s[2:3], s[8:9], 12
	v_lshl_add_u64 v[38:39], v[38:39], 0, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], 3, v34
	s_mov_b32 s8, 0
	v_mov_b32_e32 v42, 3
	v_cndmask_b32_e64 v44, 0, 1, s[2:3]
	v_mov_b32_e32 v45, s8
	v_readfirstlane_b32 s12, v38
	v_readfirstlane_b32 s13, v39
	v_mov_b32_e32 v43, s8
	s_mov_b32 s10, s8
	s_mov_b32 s11, s8
	v_and_b32_e32 v32, 0xffffff1d, v32
	s_mov_b32 s9, s8
	global_store_dwordx4 v48, v[42:45], s[12:13] offset:16
	v_or_b32_e32 v32, 0x62, v32
	global_store_dwordx4 v48, v[32:35], s[12:13]
	v_mov_b64_e32 v[44:45], s[10:11]
	v_mov_b64_e32 v[42:43], s[8:9]
	global_store_dwordx4 v48, v[42:45], s[12:13] offset:32
	global_store_dwordx4 v48, v[42:45], s[12:13] offset:48
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_66
	v_mov_b32_e32 v42, 0
	global_load_dwordx2 v[54:55], v42, s[4:5] offset:32 sc0 sc1
	global_load_dwordx2 v[32:33], v42, s[4:5] offset:40
	v_mov_b32_e32 v52, s6
	v_mov_b32_e32 v53, s7
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v32
	v_readfirstlane_b32 s11, v33
	s_and_b64 s[10:11], s[10:11], s[6:7]
	s_mul_i32 s11, s11, 24
	s_mul_hi_u32 s12, s10, 24
	s_mul_i32 s10, s10, 24
	s_add_i32 s11, s12, s11
	v_lshl_add_u64 v[32:33], v[36:37], 0, s[10:11]
	global_store_dwordx2 v[32:33], v[54:55], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[38:39], v42, v[52:55], s[4:5] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[38:39], v[54:55]
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_62
	s_mov_b64 s[12:13], 0
.LBB0_61:
	s_sleep 1
	global_store_dwordx2 v[32:33], v[38:39], off
	v_mov_b32_e32 v36, s6
	v_mov_b32_e32 v37, s7
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v42, v[36:39], s[4:5] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[36:37], v[38:39]
	s_or_b64 s[12:13], vcc, s[12:13]
	v_mov_b64_e32 v[38:39], v[36:37]
	s_andn2_b64 exec, exec, s[12:13]
	s_cbranch_execnz .LBB0_61
.LBB0_62:
	s_or_b64 exec, exec, s[10:11]
	v_mov_b32_e32 v37, 0
	global_load_dwordx2 v[32:33], v37, s[4:5] offset:16
	s_mov_b64 s[10:11], exec
	v_mbcnt_lo_u32_b32 v36, s10, 0
	v_mbcnt_hi_u32_b32 v36, s11, v36
	v_cmp_eq_u32_e32 vcc, 0, v36
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB0_64
	s_bcnt1_i32_b64 s10, s[10:11]
	v_mov_b32_e32 v36, s10
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[32:33], v[36:37], off offset:8 sc1
.LBB0_64:
	s_or_b64 exec, exec, s[12:13]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[36:37], v[32:33], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[36:37]
	s_cbranch_vccnz .LBB0_66
	global_load_dword v32, v[32:33], off offset:24
	v_mov_b32_e32 v33, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v32
	s_and_b32 m0, s10, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[36:37], v[32:33], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_66:
	s_or_b64 exec, exec, s[8:9]
	s_branch .LBB0_70
.LBB0_67:
	s_or_b64 exec, exec, s[8:9]
	v_readfirstlane_b32 s8, v32
	s_cmp_eq_u32 s8, 0
	s_cbranch_scc1 .LBB0_69
	s_sleep 1
	s_cbranch_execnz .LBB0_70
	s_branch .LBB0_72
.LBB0_69:
	s_branch .LBB0_72
.LBB0_70:
	v_mov_b32_e32 v32, 1
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_67
	global_load_dword v32, v[40:41], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v32, 1, v32
	s_branch .LBB0_67
.LBB0_72:
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_75
	v_mov_b32_e32 v40, 0
	global_load_dwordx2 v[32:33], v40, s[4:5] offset:40
	global_load_dwordx2 v[44:45], v40, s[4:5] offset:24 sc0 sc1
	global_load_dwordx2 v[36:37], v40, s[4:5]
	s_waitcnt vmcnt(2)
	v_readfirstlane_b32 s10, v32
	v_readfirstlane_b32 s11, v33
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
	v_lshl_add_u64 v[32:33], v[36:37], 0, s[6:7]
	v_mov_b32_e32 v42, s0
	global_store_dwordx2 v[32:33], v[44:45], off
	v_mov_b32_e32 v43, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[38:39], v40, v[42:45], s[4:5] offset:24 sc0 sc1
	s_mov_b64 s[6:7], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[38:39], v[44:45]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_75
.LBB0_74:
	s_sleep 1
	global_store_dwordx2 v[32:33], v[38:39], off
	v_mov_b32_e32 v36, s0
	v_mov_b32_e32 v37, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v40, v[36:39], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[36:37], v[38:39]
	s_or_b64 s[6:7], vcc, s[6:7]
	v_mov_b64_e32 v[38:39], v[36:37]
	s_andn2_b64 exec, exec, s[6:7]
	s_cbranch_execnz .LBB0_74
.LBB0_75:
	s_or_b64 exec, exec, s[8:9]
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_226
	v_readfirstlane_b32 s0, v46
	v_mov_b64_e32 v[40:41], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v46
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_cbranch_execz .LBB0_82
	v_mov_b32_e32 v36, 0
	global_load_dwordx2 v[42:43], v36, s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[32:33], v36, s[4:5] offset:40
	global_load_dwordx2 v[38:39], v36, s[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v32, v32, v42
	v_and_b32_e32 v33, v33, v43
	v_mul_lo_u32 v33, v33, 24
	v_mul_hi_u32 v37, v32, 24
	v_add_u32_e32 v33, v37, v33
	v_mul_lo_u32 v32, v32, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[32:33], v[38:39], 0, v[32:33]
	global_load_dwordx2 v[40:41], v[32:33], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[40:41], v36, v[40:43], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[40:41], v[42:43]
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_81
	s_mov_b64 s[10:11], 0
.LBB0_79:
	s_sleep 1
	global_load_dwordx2 v[32:33], v36, s[4:5] offset:40
	global_load_dwordx2 v[38:39], v36, s[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v32, v32, v40
	v_and_b32_e32 v37, v33, v41
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[32:33], s[12:13], v32, 24, v[38:39]
	v_mov_b32_e32 v38, v33
	v_mad_u64_u32 v[38:39], s[12:13], v37, 24, v[38:39]
	v_mov_b32_e32 v33, v38
	global_load_dwordx2 v[38:39], v[32:33], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v36, v[38:41], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[32:33], v[40:41]
	s_or_b64 s[10:11], vcc, s[10:11]
	v_mov_b64_e32 v[40:41], v[32:33]
	s_andn2_b64 exec, exec, s[10:11]
	s_cbranch_execnz .LBB0_79
	s_or_b64 exec, exec, s[10:11]
	v_mov_b64_e32 v[40:41], v[32:33]
.LBB0_81:
	s_or_b64 exec, exec, s[8:9]
.LBB0_82:
	s_or_b64 exec, exec, s[2:3]
	v_mov_b32_e32 v49, 0
	global_load_dwordx2 v[32:33], v49, s[4:5] offset:40
	global_load_dwordx4 v[36:39], v49, s[4:5]
	v_readfirstlane_b32 s3, v41
	v_readfirstlane_b32 s2, v40
	s_mov_b64 s[8:9], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s10, v32
	v_readfirstlane_b32 s11, v33
	s_and_b64 s[10:11], s[10:11], s[2:3]
	s_mul_i32 s12, s11, 24
	s_mul_hi_u32 s13, s10, 24
	s_add_i32 s13, s13, s12
	s_mul_i32 s12, s10, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[32:33], v[36:37], 0, s[12:13]
	s_and_saveexec_b64 s[12:13], s[0:1]
	s_cbranch_execz .LBB0_84
	v_mov_b32_e32 v40, s8
	v_mov_b32_e32 v41, s9
	v_mov_b32_e32 v42, 2
	v_mov_b32_e32 v43, 1
	global_store_dwordx4 v[32:33], v[40:43], off offset:8
.LBB0_84:
	s_or_b64 exec, exec, s[12:13]
	s_lshl_b64 s[8:9], s[10:11], 12
	v_lshl_add_u64 v[40:41], v[38:39], 0, s[8:9]
	s_mov_b32 s8, 0
	v_mov_b32_e32 v42, 33
	v_mov_b32_e32 v43, v49
	v_mov_b32_e32 v44, v49
	v_mov_b32_e32 v45, v49
	v_readfirstlane_b32 s12, v40
	v_readfirstlane_b32 s13, v41
	s_mov_b32 s10, s8
	s_mov_b32 s11, s8
	s_mov_b32 s9, s8
	s_nop 1
	global_store_dwordx4 v48, v[42:45], s[12:13]
	s_nop 1
	v_mov_b64_e32 v[44:45], s[10:11]
	v_mov_b64_e32 v[42:43], s[8:9]
	global_store_dwordx4 v48, v[42:45], s[12:13] offset:16
	global_store_dwordx4 v48, v[42:45], s[12:13] offset:32
	global_store_dwordx4 v48, v[42:45], s[12:13] offset:48
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_92
	v_mov_b32_e32 v44, 0
	global_load_dwordx2 v[54:55], v44, s[4:5] offset:32 sc0 sc1
	global_load_dwordx2 v[38:39], v44, s[4:5] offset:40
	v_mov_b32_e32 v52, s2
	v_mov_b32_e32 v53, s3
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v38
	v_readfirstlane_b32 s11, v39
	s_and_b64 s[10:11], s[10:11], s[2:3]
	s_mul_i32 s11, s11, 24
	s_mul_hi_u32 s12, s10, 24
	s_mul_i32 s10, s10, 24
	s_add_i32 s11, s12, s11
	v_lshl_add_u64 v[42:43], v[36:37], 0, s[10:11]
	global_store_dwordx2 v[42:43], v[54:55], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[38:39], v44, v[52:55], s[4:5] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[38:39], v[54:55]
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_88
	s_mov_b64 s[12:13], 0
.LBB0_87:
	s_sleep 1
	global_store_dwordx2 v[42:43], v[38:39], off
	v_mov_b32_e32 v36, s2
	v_mov_b32_e32 v37, s3
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v44, v[36:39], s[4:5] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[36:37], v[38:39]
	s_or_b64 s[12:13], vcc, s[12:13]
	v_mov_b64_e32 v[38:39], v[36:37]
	s_andn2_b64 exec, exec, s[12:13]
	s_cbranch_execnz .LBB0_87
.LBB0_88:
	s_or_b64 exec, exec, s[10:11]
	v_mov_b32_e32 v39, 0
	global_load_dwordx2 v[36:37], v39, s[4:5] offset:16
	s_mov_b64 s[10:11], exec
	v_mbcnt_lo_u32_b32 v38, s10, 0
	v_mbcnt_hi_u32_b32 v38, s11, v38
	v_cmp_eq_u32_e32 vcc, 0, v38
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB0_90
	s_bcnt1_i32_b64 s10, s[10:11]
	v_mov_b32_e32 v38, s10
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[36:37], v[38:39], off offset:8 sc1
.LBB0_90:
	s_or_b64 exec, exec, s[12:13]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[38:39], v[36:37], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[38:39]
	s_cbranch_vccnz .LBB0_92
	global_load_dword v36, v[36:37], off offset:24
	v_mov_b32_e32 v37, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v36
	s_and_b32 m0, s10, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[38:39], v[36:37], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_92:
	s_or_b64 exec, exec, s[8:9]
	v_lshl_add_u64 v[36:37], v[40:41], 0, v[48:49]
	s_branch .LBB0_96
.LBB0_93:
	s_or_b64 exec, exec, s[8:9]
	v_readfirstlane_b32 s8, v38
	s_cmp_eq_u32 s8, 0
	s_cbranch_scc1 .LBB0_95
	s_sleep 1
	s_cbranch_execnz .LBB0_96
	s_branch .LBB0_98
.LBB0_95:
	s_branch .LBB0_98
.LBB0_96:
	v_mov_b32_e32 v38, 1
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_93
	global_load_dword v38, v[32:33], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v38, 1, v38
	s_branch .LBB0_93
.LBB0_98:
	global_load_dwordx2 v[36:37], v[36:37], off
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_101
	v_mov_b32_e32 v42, 0
	global_load_dwordx2 v[32:33], v42, s[4:5] offset:40
	global_load_dwordx2 v[54:55], v42, s[4:5] offset:24 sc0 sc1
	global_load_dwordx2 v[38:39], v42, s[4:5]
	s_waitcnt vmcnt(2)
	v_readfirstlane_b32 s10, v32
	v_readfirstlane_b32 s11, v33
	s_add_u32 s12, s10, 1
	s_addc_u32 s13, s11, 0
	s_add_u32 s0, s12, s2
	s_addc_u32 s1, s13, s3
	s_cmp_eq_u64 s[0:1], 0
	s_cselect_b32 s1, s13, s1
	s_cselect_b32 s0, s12, s0
	s_and_b64 s[2:3], s[0:1], s[10:11]
	s_mul_i32 s3, s3, 24
	s_mul_hi_u32 s10, s2, 24
	s_mul_i32 s2, s2, 24
	s_add_i32 s3, s10, s3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[32:33], v[38:39], 0, s[2:3]
	v_mov_b32_e32 v52, s0
	global_store_dwordx2 v[32:33], v[54:55], off
	v_mov_b32_e32 v53, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[40:41], v42, v[52:55], s[4:5] offset:24 sc0 sc1
	s_mov_b64 s[2:3], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[40:41], v[54:55]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_101
.LBB0_100:
	s_sleep 1
	global_store_dwordx2 v[32:33], v[40:41], off
	v_mov_b32_e32 v38, s0
	v_mov_b32_e32 v39, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[38:39], v42, v[38:41], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[38:39], v[40:41]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[40:41], v[38:39]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_100
.LBB0_101:
	s_or_b64 exec, exec, s[8:9]
	v_readfirstlane_b32 s0, v46
	v_mov_b64_e32 v[44:45], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v46
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_cbranch_execz .LBB0_107
	v_mov_b32_e32 v38, 0
	global_load_dwordx2 v[42:43], v38, s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[32:33], v38, s[4:5] offset:40
	global_load_dwordx2 v[40:41], v38, s[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v32, v32, v42
	v_and_b32_e32 v33, v33, v43
	v_mul_lo_u32 v33, v33, 24
	v_mul_hi_u32 v39, v32, 24
	v_add_u32_e32 v33, v39, v33
	v_mul_lo_u32 v32, v32, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[32:33], v[40:41], 0, v[32:33]
	global_load_dwordx2 v[40:41], v[32:33], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[44:45], v38, v[40:43], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[44:45], v[42:43]
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_106
	s_mov_b64 s[10:11], 0
.LBB0_104:
	s_sleep 1
	global_load_dwordx2 v[32:33], v38, s[4:5] offset:40
	global_load_dwordx2 v[40:41], v38, s[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v32, v32, v44
	v_and_b32_e32 v39, v33, v45
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[32:33], s[12:13], v32, 24, v[40:41]
	v_mov_b32_e32 v40, v33
	v_mad_u64_u32 v[40:41], s[12:13], v39, 24, v[40:41]
	v_mov_b32_e32 v33, v40
	global_load_dwordx2 v[42:43], v[32:33], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v38, v[42:45], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[32:33], v[44:45]
	s_or_b64 s[10:11], vcc, s[10:11]
	v_mov_b64_e32 v[44:45], v[32:33]
	s_andn2_b64 exec, exec, s[10:11]
	s_cbranch_execnz .LBB0_104
	s_or_b64 exec, exec, s[10:11]
	v_mov_b64_e32 v[44:45], v[32:33]
.LBB0_106:
	s_or_b64 exec, exec, s[8:9]
.LBB0_107:
	s_or_b64 exec, exec, s[2:3]
	v_mov_b32_e32 v49, 0
	global_load_dwordx2 v[32:33], v49, s[4:5] offset:40
	global_load_dwordx4 v[40:43], v49, s[4:5]
	v_readfirstlane_b32 s3, v45
	v_readfirstlane_b32 s2, v44
	s_mov_b64 s[8:9], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s10, v32
	v_readfirstlane_b32 s11, v33
	s_and_b64 s[10:11], s[10:11], s[2:3]
	s_mul_i32 s12, s11, 24
	s_mul_hi_u32 s13, s10, 24
	s_add_i32 s13, s13, s12
	s_mul_i32 s12, s10, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[32:33], v[40:41], 0, s[12:13]
	s_and_saveexec_b64 s[12:13], s[0:1]
	s_cbranch_execz .LBB0_109
	v_mov_b32_e32 v52, s8
	v_mov_b32_e32 v53, s9
	v_mov_b32_e32 v54, 2
	v_mov_b32_e32 v55, 1
	global_store_dwordx4 v[32:33], v[52:55], off offset:8
.LBB0_109:
	s_or_b64 exec, exec, s[12:13]
	s_lshl_b64 s[8:9], s[10:11], 12
	v_lshl_add_u64 v[42:43], v[42:43], 0, s[8:9]
	v_and_b32_e32 v36, 0xffffff1f, v36
	v_or_b32_e32 v36, 0xa0, v36
	v_mov_b32_e32 v39, 0x7270205d
	v_mov_b32_e32 v38, 0x64255b41
	v_readfirstlane_b32 s8, v42
	v_readfirstlane_b32 s9, v43
	s_nop 4
	global_store_dwordx4 v48, v[36:39], s[8:9]
	s_nop 1
	v_mov_b32_e32 v36, 0x65746e69
	v_mov_b32_e32 v37, 0x6e692064
	v_mov_b32_e32 v38, 0x65646973
	v_mov_b32_e32 v39, 0x72656b20
	global_store_dwordx4 v48, v[36:39], s[8:9] offset:16
	s_nop 1
	v_mov_b32_e32 v36, 0x206c656e
	v_mov_b32_e32 v37, 0x3425203d
	v_mov_b32_e32 v38, 0xa66332e
	v_mov_b32_e32 v39, v49
	global_store_dwordx4 v48, v[36:39], s[8:9] offset:32
	s_nop 1
	v_mov_b32_e32 v36, v49
	v_mov_b32_e32 v37, v49
	v_mov_b32_e32 v38, v49
	global_store_dwordx4 v48, v[36:39], s[8:9] offset:48
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_117
	v_mov_b32_e32 v44, 0
	global_load_dwordx2 v[54:55], v44, s[4:5] offset:32 sc0 sc1
	global_load_dwordx2 v[36:37], v44, s[4:5] offset:40
	v_mov_b32_e32 v52, s2
	v_mov_b32_e32 v53, s3
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v36
	v_readfirstlane_b32 s11, v37
	s_and_b64 s[10:11], s[10:11], s[2:3]
	s_mul_i32 s11, s11, 24
	s_mul_hi_u32 s12, s10, 24
	s_mul_i32 s10, s10, 24
	s_add_i32 s11, s12, s11
	v_lshl_add_u64 v[40:41], v[40:41], 0, s[10:11]
	global_store_dwordx2 v[40:41], v[54:55], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[38:39], v44, v[52:55], s[4:5] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[38:39], v[54:55]
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_113
	s_mov_b64 s[12:13], 0
.LBB0_112:
	s_sleep 1
	global_store_dwordx2 v[40:41], v[38:39], off
	v_mov_b32_e32 v36, s2
	v_mov_b32_e32 v37, s3
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v44, v[36:39], s[4:5] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[36:37], v[38:39]
	s_or_b64 s[12:13], vcc, s[12:13]
	v_mov_b64_e32 v[38:39], v[36:37]
	s_andn2_b64 exec, exec, s[12:13]
	s_cbranch_execnz .LBB0_112
.LBB0_113:
	s_or_b64 exec, exec, s[10:11]
	v_mov_b32_e32 v39, 0
	global_load_dwordx2 v[36:37], v39, s[4:5] offset:16
	s_mov_b64 s[10:11], exec
	v_mbcnt_lo_u32_b32 v38, s10, 0
	v_mbcnt_hi_u32_b32 v38, s11, v38
	v_cmp_eq_u32_e32 vcc, 0, v38
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB0_115
	s_bcnt1_i32_b64 s10, s[10:11]
	v_mov_b32_e32 v38, s10
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[36:37], v[38:39], off offset:8 sc1
.LBB0_115:
	s_or_b64 exec, exec, s[12:13]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[38:39], v[36:37], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[38:39]
	s_cbranch_vccnz .LBB0_117
	global_load_dword v36, v[36:37], off offset:24
	v_mov_b32_e32 v37, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v36
	s_and_b32 m0, s10, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[38:39], v[36:37], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_117:
	s_or_b64 exec, exec, s[8:9]
	v_lshl_add_u64 v[36:37], v[42:43], 0, v[48:49]
	s_branch .LBB0_121
.LBB0_118:
	s_or_b64 exec, exec, s[8:9]
	v_readfirstlane_b32 s8, v38
	s_cmp_eq_u32 s8, 0
	s_cbranch_scc1 .LBB0_120
	s_sleep 1
	s_cbranch_execnz .LBB0_121
	s_branch .LBB0_123
.LBB0_120:
	s_branch .LBB0_123
.LBB0_121:
	v_mov_b32_e32 v38, 1
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_118
	global_load_dword v38, v[32:33], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v38, 1, v38
	s_branch .LBB0_118
.LBB0_123:
	global_load_dwordx2 v[32:33], v[36:37], off
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_126
	v_mov_b32_e32 v42, 0
	global_load_dwordx2 v[36:37], v42, s[4:5] offset:40
	global_load_dwordx2 v[54:55], v42, s[4:5] offset:24 sc0 sc1
	global_load_dwordx2 v[38:39], v42, s[4:5]
	s_waitcnt vmcnt(2)
	v_readfirstlane_b32 s10, v36
	v_readfirstlane_b32 s11, v37
	s_add_u32 s12, s10, 1
	s_addc_u32 s13, s11, 0
	s_add_u32 s0, s12, s2
	s_addc_u32 s1, s13, s3
	s_cmp_eq_u64 s[0:1], 0
	s_cselect_b32 s1, s13, s1
	s_cselect_b32 s0, s12, s0
	s_and_b64 s[2:3], s[0:1], s[10:11]
	s_mul_i32 s3, s3, 24
	s_mul_hi_u32 s10, s2, 24
	s_mul_i32 s2, s2, 24
	s_add_i32 s3, s10, s3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[40:41], v[38:39], 0, s[2:3]
	v_mov_b32_e32 v52, s0
	global_store_dwordx2 v[40:41], v[54:55], off
	v_mov_b32_e32 v53, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[38:39], v42, v[52:55], s[4:5] offset:24 sc0 sc1
	s_mov_b64 s[2:3], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[38:39], v[54:55]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_126
.LBB0_125:
	s_sleep 1
	global_store_dwordx2 v[40:41], v[38:39], off
	v_mov_b32_e32 v36, s0
	v_mov_b32_e32 v37, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v42, v[36:39], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[36:37], v[38:39]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[38:39], v[36:37]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_125
.LBB0_126:
	s_or_b64 exec, exec, s[8:9]
	v_readfirstlane_b32 s0, v46
	v_mov_b64_e32 v[40:41], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v46
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_cbranch_execz .LBB0_132
	v_mov_b32_e32 v42, 0
	global_load_dwordx2 v[38:39], v42, s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[36:37], v42, s[4:5] offset:40
	global_load_dwordx2 v[40:41], v42, s[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v36, v36, v38
	v_and_b32_e32 v37, v37, v39
	v_mul_lo_u32 v37, v37, 24
	v_mul_hi_u32 v43, v36, 24
	v_add_u32_e32 v37, v43, v37
	v_mul_lo_u32 v36, v36, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[36:37], v[40:41], 0, v[36:37]
	global_load_dwordx2 v[36:37], v[36:37], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[40:41], v42, v[36:39], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[40:41], v[38:39]
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_131
	s_mov_b64 s[10:11], 0
.LBB0_129:
	s_sleep 1
	global_load_dwordx2 v[36:37], v42, s[4:5] offset:40
	global_load_dwordx2 v[38:39], v42, s[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v36, v36, v40
	v_and_b32_e32 v43, v37, v41
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[36:37], s[12:13], v36, 24, v[38:39]
	v_mov_b32_e32 v38, v37
	v_mad_u64_u32 v[38:39], s[12:13], v43, 24, v[38:39]
	v_mov_b32_e32 v37, v38
	global_load_dwordx2 v[38:39], v[36:37], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v42, v[38:41], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[36:37], v[40:41]
	s_or_b64 s[10:11], vcc, s[10:11]
	v_mov_b64_e32 v[40:41], v[36:37]
	s_andn2_b64 exec, exec, s[10:11]
	s_cbranch_execnz .LBB0_129
	s_or_b64 exec, exec, s[10:11]
	v_mov_b64_e32 v[40:41], v[36:37]
.LBB0_131:
	s_or_b64 exec, exec, s[8:9]
.LBB0_132:
	s_or_b64 exec, exec, s[2:3]
	v_mov_b32_e32 v44, 0
	global_load_dwordx2 v[42:43], v44, s[4:5] offset:40
	global_load_dwordx4 v[36:39], v44, s[4:5]
	v_readfirstlane_b32 s3, v41
	v_readfirstlane_b32 s2, v40
	s_mov_b64 s[8:9], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s10, v42
	v_readfirstlane_b32 s11, v43
	s_and_b64 s[10:11], s[10:11], s[2:3]
	s_mul_i32 s12, s11, 24
	s_mul_hi_u32 s13, s10, 24
	s_add_i32 s13, s13, s12
	s_mul_i32 s12, s10, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[40:41], v[36:37], 0, s[12:13]
	s_and_saveexec_b64 s[12:13], s[0:1]
	s_cbranch_execz .LBB0_134
	v_mov_b32_e32 v42, s8
	v_mov_b32_e32 v43, s9
	v_mov_b32_e32 v44, 2
	v_mov_b32_e32 v45, 1
	global_store_dwordx4 v[40:41], v[42:45], off offset:8
.LBB0_134:
	s_or_b64 exec, exec, s[12:13]
	s_lshl_b64 s[8:9], s[10:11], 12
	v_lshl_add_u64 v[38:39], v[38:39], 0, s[8:9]
	s_mov_b32 s8, 0
	v_cvt_f64_f32_e32 v[42:43], v50
	v_readfirstlane_b32 s12, v38
	v_readfirstlane_b32 s13, v39
	v_mov_b32_e32 v44, s8
	v_mov_b32_e32 v45, s8
	s_mov_b32 s10, s8
	s_mov_b32 s11, s8
	v_and_b32_e32 v32, 0xffffff1d, v32
	global_store_dwordx4 v48, v[42:45], s[12:13] offset:16
	s_mov_b32 s9, s8
	v_or_b32_e32 v32, 0x42, v32
	v_mov_b64_e32 v[44:45], s[10:11]
	v_mov_b64_e32 v[42:43], s[8:9]
	global_store_dwordx4 v48, v[32:35], s[12:13]
	global_store_dwordx4 v48, v[42:45], s[12:13] offset:32
	global_store_dwordx4 v48, v[42:45], s[12:13] offset:48
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_142
	v_mov_b32_e32 v42, 0
	global_load_dwordx2 v[54:55], v42, s[4:5] offset:32 sc0 sc1
	global_load_dwordx2 v[32:33], v42, s[4:5] offset:40
	v_mov_b32_e32 v52, s2
	v_mov_b32_e32 v53, s3
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v32
	v_readfirstlane_b32 s11, v33
	s_and_b64 s[10:11], s[10:11], s[2:3]
	s_mul_i32 s11, s11, 24
	s_mul_hi_u32 s12, s10, 24
	s_mul_i32 s10, s10, 24
	s_add_i32 s11, s12, s11
	v_lshl_add_u64 v[32:33], v[36:37], 0, s[10:11]
	global_store_dwordx2 v[32:33], v[54:55], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[38:39], v42, v[52:55], s[4:5] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[38:39], v[54:55]
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_138
	s_mov_b64 s[12:13], 0
.LBB0_137:
	s_sleep 1
	global_store_dwordx2 v[32:33], v[38:39], off
	v_mov_b32_e32 v36, s2
	v_mov_b32_e32 v37, s3
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v42, v[36:39], s[4:5] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[36:37], v[38:39]
	s_or_b64 s[12:13], vcc, s[12:13]
	v_mov_b64_e32 v[38:39], v[36:37]
	s_andn2_b64 exec, exec, s[12:13]
	s_cbranch_execnz .LBB0_137
.LBB0_138:
	s_or_b64 exec, exec, s[10:11]
	v_mov_b32_e32 v37, 0
	global_load_dwordx2 v[32:33], v37, s[4:5] offset:16
	s_mov_b64 s[10:11], exec
	v_mbcnt_lo_u32_b32 v36, s10, 0
	v_mbcnt_hi_u32_b32 v36, s11, v36
	v_cmp_eq_u32_e32 vcc, 0, v36
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB0_140
	s_bcnt1_i32_b64 s10, s[10:11]
	v_mov_b32_e32 v36, s10
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[32:33], v[36:37], off offset:8 sc1
.LBB0_140:
	s_or_b64 exec, exec, s[12:13]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[36:37], v[32:33], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[36:37]
	s_cbranch_vccnz .LBB0_142
	global_load_dword v32, v[32:33], off offset:24
	v_mov_b32_e32 v33, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v32
	s_and_b32 m0, s10, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[36:37], v[32:33], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_142:
	s_or_b64 exec, exec, s[8:9]
	s_branch .LBB0_146
.LBB0_143:
	s_or_b64 exec, exec, s[8:9]
	v_readfirstlane_b32 s8, v32
	s_cmp_eq_u32 s8, 0
	s_cbranch_scc1 .LBB0_145
	s_sleep 1
	s_cbranch_execnz .LBB0_146
	s_branch .LBB0_148
.LBB0_145:
	s_branch .LBB0_148
.LBB0_146:
	v_mov_b32_e32 v32, 1
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_143
	global_load_dword v32, v[40:41], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v32, 1, v32
	s_branch .LBB0_143
.LBB0_148:
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_151
	v_mov_b32_e32 v40, 0
	global_load_dwordx2 v[32:33], v40, s[4:5] offset:40
	global_load_dwordx2 v[44:45], v40, s[4:5] offset:24 sc0 sc1
	global_load_dwordx2 v[36:37], v40, s[4:5]
	s_waitcnt vmcnt(2)
	v_readfirstlane_b32 s10, v32
	v_readfirstlane_b32 s11, v33
	s_add_u32 s12, s10, 1
	s_addc_u32 s13, s11, 0
	s_add_u32 s0, s12, s2
	s_addc_u32 s1, s13, s3
	s_cmp_eq_u64 s[0:1], 0
	s_cselect_b32 s1, s13, s1
	s_cselect_b32 s0, s12, s0
	s_and_b64 s[2:3], s[0:1], s[10:11]
	s_mul_i32 s3, s3, 24
	s_mul_hi_u32 s10, s2, 24
	s_mul_i32 s2, s2, 24
	s_add_i32 s3, s10, s3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[32:33], v[36:37], 0, s[2:3]
	v_mov_b32_e32 v42, s0
	global_store_dwordx2 v[32:33], v[44:45], off
	v_mov_b32_e32 v43, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[38:39], v40, v[42:45], s[4:5] offset:24 sc0 sc1
	s_mov_b64 s[2:3], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[38:39], v[44:45]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_151
.LBB0_150:
	s_sleep 1
	global_store_dwordx2 v[32:33], v[38:39], off
	v_mov_b32_e32 v36, s0
	v_mov_b32_e32 v37, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v40, v[36:39], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[36:37], v[38:39]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[38:39], v[36:37]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_150
.LBB0_151:
	s_or_b64 exec, exec, s[8:9]
	v_readfirstlane_b32 s0, v46
	v_mov_b64_e32 v[40:41], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v46
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_cbranch_execz .LBB0_157
	v_mov_b32_e32 v36, 0
	global_load_dwordx2 v[42:43], v36, s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[32:33], v36, s[4:5] offset:40
	global_load_dwordx2 v[38:39], v36, s[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v32, v32, v42
	v_and_b32_e32 v33, v33, v43
	v_mul_lo_u32 v33, v33, 24
	v_mul_hi_u32 v37, v32, 24
	v_add_u32_e32 v33, v37, v33
	v_mul_lo_u32 v32, v32, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[32:33], v[38:39], 0, v[32:33]
	global_load_dwordx2 v[40:41], v[32:33], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[40:41], v36, v[40:43], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[40:41], v[42:43]
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_156
	s_mov_b64 s[10:11], 0
.LBB0_154:
	s_sleep 1
	global_load_dwordx2 v[32:33], v36, s[4:5] offset:40
	global_load_dwordx2 v[38:39], v36, s[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v32, v32, v40
	v_and_b32_e32 v37, v33, v41
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[32:33], s[12:13], v32, 24, v[38:39]
	v_mov_b32_e32 v38, v33
	v_mad_u64_u32 v[38:39], s[12:13], v37, 24, v[38:39]
	v_mov_b32_e32 v33, v38
	global_load_dwordx2 v[38:39], v[32:33], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v36, v[38:41], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[32:33], v[40:41]
	s_or_b64 s[10:11], vcc, s[10:11]
	v_mov_b64_e32 v[40:41], v[32:33]
	s_andn2_b64 exec, exec, s[10:11]
	s_cbranch_execnz .LBB0_154
	s_or_b64 exec, exec, s[10:11]
	v_mov_b64_e32 v[40:41], v[32:33]
.LBB0_156:
	s_or_b64 exec, exec, s[8:9]
.LBB0_157:
	s_or_b64 exec, exec, s[2:3]
	v_mov_b32_e32 v49, 0
	global_load_dwordx2 v[32:33], v49, s[4:5] offset:40
	global_load_dwordx4 v[36:39], v49, s[4:5]
	v_readfirstlane_b32 s3, v41
	v_readfirstlane_b32 s2, v40
	s_mov_b64 s[8:9], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s10, v32
	v_readfirstlane_b32 s11, v33
	s_and_b64 s[10:11], s[10:11], s[2:3]
	s_mul_i32 s12, s11, 24
	s_mul_hi_u32 s13, s10, 24
	s_add_i32 s13, s13, s12
	s_mul_i32 s12, s10, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[32:33], v[36:37], 0, s[12:13]
	s_and_saveexec_b64 s[12:13], s[0:1]
	s_cbranch_execz .LBB0_159
	v_mov_b32_e32 v40, s8
	v_mov_b32_e32 v41, s9
	v_mov_b32_e32 v42, 2
	v_mov_b32_e32 v43, 1
	global_store_dwordx4 v[32:33], v[40:43], off offset:8
.LBB0_159:
	s_or_b64 exec, exec, s[12:13]
	s_lshl_b64 s[8:9], s[10:11], 12
	v_lshl_add_u64 v[40:41], v[38:39], 0, s[8:9]
	s_mov_b32 s8, 0
	v_mov_b32_e32 v42, 33
	v_mov_b32_e32 v43, v49
	v_mov_b32_e32 v44, v49
	v_mov_b32_e32 v45, v49
	v_readfirstlane_b32 s12, v40
	v_readfirstlane_b32 s13, v41
	s_mov_b32 s10, s8
	s_mov_b32 s11, s8
	s_mov_b32 s9, s8
	s_nop 1
	global_store_dwordx4 v48, v[42:45], s[12:13]
	s_nop 1
	v_mov_b64_e32 v[44:45], s[10:11]
	v_mov_b64_e32 v[42:43], s[8:9]
	global_store_dwordx4 v48, v[42:45], s[12:13] offset:16
	global_store_dwordx4 v48, v[42:45], s[12:13] offset:32
	global_store_dwordx4 v48, v[42:45], s[12:13] offset:48
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_167
	v_mov_b32_e32 v44, 0
	global_load_dwordx2 v[54:55], v44, s[4:5] offset:32 sc0 sc1
	global_load_dwordx2 v[38:39], v44, s[4:5] offset:40
	v_mov_b32_e32 v52, s2
	v_mov_b32_e32 v53, s3
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v38
	v_readfirstlane_b32 s11, v39
	s_and_b64 s[10:11], s[10:11], s[2:3]
	s_mul_i32 s11, s11, 24
	s_mul_hi_u32 s12, s10, 24
	s_mul_i32 s10, s10, 24
	s_add_i32 s11, s12, s11
	v_lshl_add_u64 v[42:43], v[36:37], 0, s[10:11]
	global_store_dwordx2 v[42:43], v[54:55], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[38:39], v44, v[52:55], s[4:5] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[38:39], v[54:55]
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_163
	s_mov_b64 s[12:13], 0
.LBB0_162:
	s_sleep 1
	global_store_dwordx2 v[42:43], v[38:39], off
	v_mov_b32_e32 v36, s2
	v_mov_b32_e32 v37, s3
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v44, v[36:39], s[4:5] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[36:37], v[38:39]
	s_or_b64 s[12:13], vcc, s[12:13]
	v_mov_b64_e32 v[38:39], v[36:37]
	s_andn2_b64 exec, exec, s[12:13]
	s_cbranch_execnz .LBB0_162
.LBB0_163:
	s_or_b64 exec, exec, s[10:11]
	v_mov_b32_e32 v39, 0
	global_load_dwordx2 v[36:37], v39, s[4:5] offset:16
	s_mov_b64 s[10:11], exec
	v_mbcnt_lo_u32_b32 v38, s10, 0
	v_mbcnt_hi_u32_b32 v38, s11, v38
	v_cmp_eq_u32_e32 vcc, 0, v38
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB0_165
	s_bcnt1_i32_b64 s10, s[10:11]
	v_mov_b32_e32 v38, s10
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[36:37], v[38:39], off offset:8 sc1
.LBB0_165:
	s_or_b64 exec, exec, s[12:13]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[38:39], v[36:37], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[38:39]
	s_cbranch_vccnz .LBB0_167
	global_load_dword v36, v[36:37], off offset:24
	v_mov_b32_e32 v37, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v36
	s_and_b32 m0, s10, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[38:39], v[36:37], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_167:
	s_or_b64 exec, exec, s[8:9]
	v_lshl_add_u64 v[36:37], v[40:41], 0, v[48:49]
	s_branch .LBB0_171
.LBB0_168:
	s_or_b64 exec, exec, s[8:9]
	v_readfirstlane_b32 s8, v38
	s_cmp_eq_u32 s8, 0
	s_cbranch_scc1 .LBB0_170
	s_sleep 1
	s_cbranch_execnz .LBB0_171
	s_branch .LBB0_173
.LBB0_170:
	s_branch .LBB0_173
.LBB0_171:
	v_mov_b32_e32 v38, 1
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_168
	global_load_dword v38, v[32:33], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v38, 1, v38
	s_branch .LBB0_168
.LBB0_173:
	global_load_dwordx2 v[36:37], v[36:37], off
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_176
	v_mov_b32_e32 v42, 0
	global_load_dwordx2 v[32:33], v42, s[4:5] offset:40
	global_load_dwordx2 v[54:55], v42, s[4:5] offset:24 sc0 sc1
	global_load_dwordx2 v[38:39], v42, s[4:5]
	s_waitcnt vmcnt(2)
	v_readfirstlane_b32 s10, v32
	v_readfirstlane_b32 s11, v33
	s_add_u32 s12, s10, 1
	s_addc_u32 s13, s11, 0
	s_add_u32 s0, s12, s2
	s_addc_u32 s1, s13, s3
	s_cmp_eq_u64 s[0:1], 0
	s_cselect_b32 s1, s13, s1
	s_cselect_b32 s0, s12, s0
	s_and_b64 s[2:3], s[0:1], s[10:11]
	s_mul_i32 s3, s3, 24
	s_mul_hi_u32 s10, s2, 24
	s_mul_i32 s2, s2, 24
	s_add_i32 s3, s10, s3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[32:33], v[38:39], 0, s[2:3]
	v_mov_b32_e32 v52, s0
	global_store_dwordx2 v[32:33], v[54:55], off
	v_mov_b32_e32 v53, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[40:41], v42, v[52:55], s[4:5] offset:24 sc0 sc1
	s_mov_b64 s[2:3], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[40:41], v[54:55]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_176
.LBB0_175:
	s_sleep 1
	global_store_dwordx2 v[32:33], v[40:41], off
	v_mov_b32_e32 v38, s0
	v_mov_b32_e32 v39, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[38:39], v42, v[38:41], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[38:39], v[40:41]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[40:41], v[38:39]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_175
.LBB0_176:
	s_or_b64 exec, exec, s[8:9]
	v_readfirstlane_b32 s0, v46
	v_mov_b64_e32 v[44:45], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v46
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_cbranch_execz .LBB0_182
	v_mov_b32_e32 v38, 0
	global_load_dwordx2 v[42:43], v38, s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[32:33], v38, s[4:5] offset:40
	global_load_dwordx2 v[40:41], v38, s[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v32, v32, v42
	v_and_b32_e32 v33, v33, v43
	v_mul_lo_u32 v33, v33, 24
	v_mul_hi_u32 v39, v32, 24
	v_add_u32_e32 v33, v39, v33
	v_mul_lo_u32 v32, v32, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[32:33], v[40:41], 0, v[32:33]
	global_load_dwordx2 v[40:41], v[32:33], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[44:45], v38, v[40:43], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[44:45], v[42:43]
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_181
	s_mov_b64 s[10:11], 0
.LBB0_179:
	s_sleep 1
	global_load_dwordx2 v[32:33], v38, s[4:5] offset:40
	global_load_dwordx2 v[40:41], v38, s[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v32, v32, v44
	v_and_b32_e32 v39, v33, v45
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[32:33], s[12:13], v32, 24, v[40:41]
	v_mov_b32_e32 v40, v33
	v_mad_u64_u32 v[40:41], s[12:13], v39, 24, v[40:41]
	v_mov_b32_e32 v33, v40
	global_load_dwordx2 v[42:43], v[32:33], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v38, v[42:45], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[32:33], v[44:45]
	s_or_b64 s[10:11], vcc, s[10:11]
	v_mov_b64_e32 v[44:45], v[32:33]
	s_andn2_b64 exec, exec, s[10:11]
	s_cbranch_execnz .LBB0_179
	s_or_b64 exec, exec, s[10:11]
	v_mov_b64_e32 v[44:45], v[32:33]
.LBB0_181:
	s_or_b64 exec, exec, s[8:9]
.LBB0_182:
	s_or_b64 exec, exec, s[2:3]
	v_mov_b32_e32 v49, 0
	global_load_dwordx2 v[32:33], v49, s[4:5] offset:40
	global_load_dwordx4 v[40:43], v49, s[4:5]
	v_readfirstlane_b32 s3, v45
	v_readfirstlane_b32 s2, v44
	s_mov_b64 s[8:9], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s10, v32
	v_readfirstlane_b32 s11, v33
	s_and_b64 s[10:11], s[10:11], s[2:3]
	s_mul_i32 s12, s11, 24
	s_mul_hi_u32 s13, s10, 24
	s_add_i32 s13, s13, s12
	s_mul_i32 s12, s10, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[32:33], v[40:41], 0, s[12:13]
	s_and_saveexec_b64 s[12:13], s[0:1]
	s_cbranch_execz .LBB0_184
	v_mov_b32_e32 v52, s8
	v_mov_b32_e32 v53, s9
	v_mov_b32_e32 v54, 2
	v_mov_b32_e32 v55, 1
	global_store_dwordx4 v[32:33], v[52:55], off offset:8
.LBB0_184:
	s_or_b64 exec, exec, s[12:13]
	s_lshl_b64 s[8:9], s[10:11], 12
	v_lshl_add_u64 v[42:43], v[42:43], 0, s[8:9]
	v_and_b32_e32 v36, 0xffffff1f, v36
	v_or_b32_e32 v36, 0xa0, v36
	v_mov_b32_e32 v39, 0x7270205d
	v_mov_b32_e32 v38, 0x64255b42
	v_readfirstlane_b32 s8, v42
	v_readfirstlane_b32 s9, v43
	s_nop 4
	global_store_dwordx4 v48, v[36:39], s[8:9]
	s_nop 1
	v_mov_b32_e32 v36, 0x65746e69
	v_mov_b32_e32 v37, 0x6e692064
	v_mov_b32_e32 v38, 0x65646973
	v_mov_b32_e32 v39, 0x72656b20
	global_store_dwordx4 v48, v[36:39], s[8:9] offset:16
	s_nop 1
	v_mov_b32_e32 v36, 0x206c656e
	v_mov_b32_e32 v37, 0x3425203d
	v_mov_b32_e32 v38, 0xa66332e
	v_mov_b32_e32 v39, v49
	global_store_dwordx4 v48, v[36:39], s[8:9] offset:32
	s_nop 1
	v_mov_b32_e32 v36, v49
	v_mov_b32_e32 v37, v49
	v_mov_b32_e32 v38, v49
	global_store_dwordx4 v48, v[36:39], s[8:9] offset:48
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_192
	v_mov_b32_e32 v44, 0
	global_load_dwordx2 v[54:55], v44, s[4:5] offset:32 sc0 sc1
	global_load_dwordx2 v[36:37], v44, s[4:5] offset:40
	v_mov_b32_e32 v52, s2
	v_mov_b32_e32 v53, s3
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v36
	v_readfirstlane_b32 s11, v37
	s_and_b64 s[10:11], s[10:11], s[2:3]
	s_mul_i32 s11, s11, 24
	s_mul_hi_u32 s12, s10, 24
	s_mul_i32 s10, s10, 24
	s_add_i32 s11, s12, s11
	v_lshl_add_u64 v[40:41], v[40:41], 0, s[10:11]
	global_store_dwordx2 v[40:41], v[54:55], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[38:39], v44, v[52:55], s[4:5] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[38:39], v[54:55]
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_188
	s_mov_b64 s[12:13], 0
.LBB0_187:
	s_sleep 1
	global_store_dwordx2 v[40:41], v[38:39], off
	v_mov_b32_e32 v36, s2
	v_mov_b32_e32 v37, s3
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v44, v[36:39], s[4:5] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[36:37], v[38:39]
	s_or_b64 s[12:13], vcc, s[12:13]
	v_mov_b64_e32 v[38:39], v[36:37]
	s_andn2_b64 exec, exec, s[12:13]
	s_cbranch_execnz .LBB0_187
.LBB0_188:
	s_or_b64 exec, exec, s[10:11]
	v_mov_b32_e32 v39, 0
	global_load_dwordx2 v[36:37], v39, s[4:5] offset:16
	s_mov_b64 s[10:11], exec
	v_mbcnt_lo_u32_b32 v38, s10, 0
	v_mbcnt_hi_u32_b32 v38, s11, v38
	v_cmp_eq_u32_e32 vcc, 0, v38
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB0_190
	s_bcnt1_i32_b64 s10, s[10:11]
	v_mov_b32_e32 v38, s10
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[36:37], v[38:39], off offset:8 sc1
.LBB0_190:
	s_or_b64 exec, exec, s[12:13]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[38:39], v[36:37], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[38:39]
	s_cbranch_vccnz .LBB0_192
	global_load_dword v36, v[36:37], off offset:24
	v_mov_b32_e32 v37, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v36
	s_and_b32 m0, s10, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[38:39], v[36:37], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_192:
	s_or_b64 exec, exec, s[8:9]
	v_lshl_add_u64 v[36:37], v[42:43], 0, v[48:49]
	s_branch .LBB0_196
.LBB0_193:
	s_or_b64 exec, exec, s[8:9]
	v_readfirstlane_b32 s8, v38
	s_cmp_eq_u32 s8, 0
	s_cbranch_scc1 .LBB0_195
	s_sleep 1
	s_cbranch_execnz .LBB0_196
	s_branch .LBB0_198
.LBB0_195:
	s_branch .LBB0_198
.LBB0_196:
	v_mov_b32_e32 v38, 1
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_193
	global_load_dword v38, v[32:33], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v38, 1, v38
	s_branch .LBB0_193
.LBB0_198:
	global_load_dwordx2 v[32:33], v[36:37], off
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_201
	v_mov_b32_e32 v42, 0
	global_load_dwordx2 v[36:37], v42, s[4:5] offset:40
	global_load_dwordx2 v[54:55], v42, s[4:5] offset:24 sc0 sc1
	global_load_dwordx2 v[38:39], v42, s[4:5]
	s_waitcnt vmcnt(2)
	v_readfirstlane_b32 s10, v36
	v_readfirstlane_b32 s11, v37
	s_add_u32 s12, s10, 1
	s_addc_u32 s13, s11, 0
	s_add_u32 s0, s12, s2
	s_addc_u32 s1, s13, s3
	s_cmp_eq_u64 s[0:1], 0
	s_cselect_b32 s1, s13, s1
	s_cselect_b32 s0, s12, s0
	s_and_b64 s[2:3], s[0:1], s[10:11]
	s_mul_i32 s3, s3, 24
	s_mul_hi_u32 s10, s2, 24
	s_mul_i32 s2, s2, 24
	s_add_i32 s3, s10, s3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[40:41], v[38:39], 0, s[2:3]
	v_mov_b32_e32 v52, s0
	global_store_dwordx2 v[40:41], v[54:55], off
	v_mov_b32_e32 v53, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[38:39], v42, v[52:55], s[4:5] offset:24 sc0 sc1
	s_mov_b64 s[2:3], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[38:39], v[54:55]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_201
.LBB0_200:
	s_sleep 1
	global_store_dwordx2 v[40:41], v[38:39], off
	v_mov_b32_e32 v36, s0
	v_mov_b32_e32 v37, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v42, v[36:39], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[36:37], v[38:39]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[38:39], v[36:37]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_200
.LBB0_201:
	s_or_b64 exec, exec, s[8:9]
	v_readfirstlane_b32 s0, v46
	v_mov_b64_e32 v[40:41], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v46
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_cbranch_execz .LBB0_207
	v_mov_b32_e32 v42, 0
	global_load_dwordx2 v[38:39], v42, s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[36:37], v42, s[4:5] offset:40
	global_load_dwordx2 v[40:41], v42, s[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v36, v36, v38
	v_and_b32_e32 v37, v37, v39
	v_mul_lo_u32 v37, v37, 24
	v_mul_hi_u32 v43, v36, 24
	v_add_u32_e32 v37, v43, v37
	v_mul_lo_u32 v36, v36, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[36:37], v[40:41], 0, v[36:37]
	global_load_dwordx2 v[36:37], v[36:37], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[40:41], v42, v[36:39], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[40:41], v[38:39]
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_206
	s_mov_b64 s[10:11], 0
.LBB0_204:
	s_sleep 1
	global_load_dwordx2 v[36:37], v42, s[4:5] offset:40
	global_load_dwordx2 v[38:39], v42, s[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v36, v36, v40
	v_and_b32_e32 v43, v37, v41
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[36:37], s[12:13], v36, 24, v[38:39]
	v_mov_b32_e32 v38, v37
	v_mad_u64_u32 v[38:39], s[12:13], v43, 24, v[38:39]
	v_mov_b32_e32 v37, v38
	global_load_dwordx2 v[38:39], v[36:37], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v42, v[38:41], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[36:37], v[40:41]
	s_or_b64 s[10:11], vcc, s[10:11]
	v_mov_b64_e32 v[40:41], v[36:37]
	s_andn2_b64 exec, exec, s[10:11]
	s_cbranch_execnz .LBB0_204
	s_or_b64 exec, exec, s[10:11]
	v_mov_b64_e32 v[40:41], v[36:37]
.LBB0_206:
	s_or_b64 exec, exec, s[8:9]
.LBB0_207:
	s_or_b64 exec, exec, s[2:3]
	v_mov_b32_e32 v44, 0
	global_load_dwordx2 v[42:43], v44, s[4:5] offset:40
	global_load_dwordx4 v[36:39], v44, s[4:5]
	v_readfirstlane_b32 s3, v41
	v_readfirstlane_b32 s2, v40
	s_mov_b64 s[8:9], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s10, v42
	v_readfirstlane_b32 s11, v43
	s_and_b64 s[10:11], s[10:11], s[2:3]
	s_mul_i32 s12, s11, 24
	s_mul_hi_u32 s13, s10, 24
	s_add_i32 s13, s13, s12
	s_mul_i32 s12, s10, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[40:41], v[36:37], 0, s[12:13]
	s_and_saveexec_b64 s[12:13], s[0:1]
	s_cbranch_execz .LBB0_209
	v_mov_b32_e32 v42, s8
	v_mov_b32_e32 v43, s9
	v_mov_b32_e32 v44, 2
	v_mov_b32_e32 v45, 1
	global_store_dwordx4 v[40:41], v[42:45], off offset:8
.LBB0_209:
	s_or_b64 exec, exec, s[12:13]
	s_lshl_b64 s[8:9], s[10:11], 12
	v_lshl_add_u64 v[38:39], v[38:39], 0, s[8:9]
	s_mov_b32 s8, 0
	v_cvt_f64_f32_e32 v[42:43], v51
	v_readfirstlane_b32 s12, v38
	v_readfirstlane_b32 s13, v39
	v_mov_b32_e32 v44, s8
	v_mov_b32_e32 v45, s8
	s_mov_b32 s10, s8
	s_mov_b32 s11, s8
	v_and_b32_e32 v32, 0xffffff1d, v32
	global_store_dwordx4 v48, v[42:45], s[12:13] offset:16
	s_mov_b32 s9, s8
	v_or_b32_e32 v32, 0x42, v32
	v_mov_b64_e32 v[44:45], s[10:11]
	v_mov_b64_e32 v[42:43], s[8:9]
	global_store_dwordx4 v48, v[32:35], s[12:13]
	global_store_dwordx4 v48, v[42:45], s[12:13] offset:32
	global_store_dwordx4 v48, v[42:45], s[12:13] offset:48
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_217
	v_mov_b32_e32 v42, 0
	global_load_dwordx2 v[54:55], v42, s[4:5] offset:32 sc0 sc1
	global_load_dwordx2 v[32:33], v42, s[4:5] offset:40
	v_mov_b32_e32 v52, s2
	v_mov_b32_e32 v53, s3
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v32
	v_readfirstlane_b32 s11, v33
	s_and_b64 s[10:11], s[10:11], s[2:3]
	s_mul_i32 s11, s11, 24
	s_mul_hi_u32 s12, s10, 24
	s_mul_i32 s10, s10, 24
	s_add_i32 s11, s12, s11
	v_lshl_add_u64 v[32:33], v[36:37], 0, s[10:11]
	global_store_dwordx2 v[32:33], v[54:55], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[38:39], v42, v[52:55], s[4:5] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[38:39], v[54:55]
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_213
	s_mov_b64 s[12:13], 0
.LBB0_212:
	s_sleep 1
	global_store_dwordx2 v[32:33], v[38:39], off
	v_mov_b32_e32 v36, s2
	v_mov_b32_e32 v37, s3
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v42, v[36:39], s[4:5] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[36:37], v[38:39]
	s_or_b64 s[12:13], vcc, s[12:13]
	v_mov_b64_e32 v[38:39], v[36:37]
	s_andn2_b64 exec, exec, s[12:13]
	s_cbranch_execnz .LBB0_212
.LBB0_213:
	s_or_b64 exec, exec, s[10:11]
	v_mov_b32_e32 v37, 0
	global_load_dwordx2 v[32:33], v37, s[4:5] offset:16
	s_mov_b64 s[10:11], exec
	v_mbcnt_lo_u32_b32 v36, s10, 0
	v_mbcnt_hi_u32_b32 v36, s11, v36
	v_cmp_eq_u32_e32 vcc, 0, v36
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB0_215
	s_bcnt1_i32_b64 s10, s[10:11]
	v_mov_b32_e32 v36, s10
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[32:33], v[36:37], off offset:8 sc1
.LBB0_215:
	s_or_b64 exec, exec, s[12:13]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[36:37], v[32:33], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[36:37]
	s_cbranch_vccnz .LBB0_217
	global_load_dword v32, v[32:33], off offset:24
	v_mov_b32_e32 v33, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v32
	s_and_b32 m0, s10, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[36:37], v[32:33], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_217:
	s_or_b64 exec, exec, s[8:9]
	s_branch .LBB0_221
.LBB0_218:
	s_or_b64 exec, exec, s[8:9]
	v_readfirstlane_b32 s8, v32
	s_cmp_eq_u32 s8, 0
	s_cbranch_scc1 .LBB0_220
	s_sleep 1
	s_cbranch_execnz .LBB0_221
	s_branch .LBB0_223
.LBB0_220:
	s_branch .LBB0_223
.LBB0_221:
	v_mov_b32_e32 v32, 1
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_218
	global_load_dword v32, v[40:41], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v32, 1, v32
	s_branch .LBB0_218
.LBB0_223:
	s_and_b64 exec, exec, s[0:1]
	s_cbranch_execz .LBB0_226
	v_mov_b32_e32 v40, 0
	global_load_dwordx2 v[32:33], v40, s[4:5] offset:40
	global_load_dwordx2 v[44:45], v40, s[4:5] offset:24 sc0 sc1
	global_load_dwordx2 v[36:37], v40, s[4:5]
	s_waitcnt vmcnt(2)
	v_readfirstlane_b32 s8, v32
	v_readfirstlane_b32 s9, v33
	s_add_u32 s10, s8, 1
	s_addc_u32 s11, s9, 0
	s_add_u32 s0, s10, s2
	s_addc_u32 s1, s11, s3
	s_cmp_eq_u64 s[0:1], 0
	s_cselect_b32 s1, s11, s1
	s_cselect_b32 s0, s10, s0
	s_and_b64 s[2:3], s[0:1], s[8:9]
	s_mul_i32 s3, s3, 24
	s_mul_hi_u32 s8, s2, 24
	s_mul_i32 s2, s2, 24
	s_add_i32 s3, s8, s3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[32:33], v[36:37], 0, s[2:3]
	v_mov_b32_e32 v42, s0
	global_store_dwordx2 v[32:33], v[44:45], off
	v_mov_b32_e32 v43, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[38:39], v40, v[42:45], s[4:5] offset:24 sc0 sc1
	s_mov_b64 s[2:3], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[38:39], v[44:45]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_226
.LBB0_225:
	s_sleep 1
	global_store_dwordx2 v[32:33], v[38:39], off
	v_mov_b32_e32 v36, s0
	v_mov_b32_e32 v37, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v40, v[36:39], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[36:37], v[38:39]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[38:39], v[36:37]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_225
.LBB0_226:
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s0, v46
	v_mov_b64_e32 v[40:41], 0
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
	v_mov_b32 v52, v2
	;;#ASMEND
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v46
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_cbranch_execz .LBB0_232
	v_mov_b32_e32 v36, 0
	global_load_dwordx2 v[42:43], v36, s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[32:33], v36, s[4:5] offset:40
	global_load_dwordx2 v[38:39], v36, s[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v32, v32, v42
	v_and_b32_e32 v33, v33, v43
	v_mul_lo_u32 v33, v33, 24
	v_mul_hi_u32 v37, v32, 24
	v_add_u32_e32 v33, v37, v33
	v_mul_lo_u32 v32, v32, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[32:33], v[38:39], 0, v[32:33]
	global_load_dwordx2 v[40:41], v[32:33], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[40:41], v36, v[40:43], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[40:41], v[42:43]
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_231
	s_mov_b64 s[8:9], 0
.LBB0_229:
	s_sleep 1
	global_load_dwordx2 v[32:33], v36, s[4:5] offset:40
	global_load_dwordx2 v[38:39], v36, s[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v32, v32, v40
	v_and_b32_e32 v37, v33, v41
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[32:33], s[10:11], v32, 24, v[38:39]
	v_mov_b32_e32 v38, v33
	v_mad_u64_u32 v[38:39], s[10:11], v37, 24, v[38:39]
	v_mov_b32_e32 v33, v38
	global_load_dwordx2 v[38:39], v[32:33], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v36, v[38:41], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[32:33], v[40:41]
	s_or_b64 s[8:9], vcc, s[8:9]
	v_mov_b64_e32 v[40:41], v[32:33]
	s_andn2_b64 exec, exec, s[8:9]
	s_cbranch_execnz .LBB0_229
	s_or_b64 exec, exec, s[8:9]
	v_mov_b64_e32 v[40:41], v[32:33]
.LBB0_231:
	s_or_b64 exec, exec, s[6:7]
.LBB0_232:
	s_or_b64 exec, exec, s[2:3]
	v_mov_b32_e32 v49, 0
	global_load_dwordx2 v[32:33], v49, s[4:5] offset:40
	global_load_dwordx4 v[36:39], v49, s[4:5]
	v_readfirstlane_b32 s3, v41
	v_readfirstlane_b32 s2, v40
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s8, v32
	v_readfirstlane_b32 s9, v33
	s_and_b64 s[8:9], s[8:9], s[2:3]
	s_mul_i32 s10, s9, 24
	s_mul_hi_u32 s11, s8, 24
	s_add_i32 s11, s11, s10
	s_mul_i32 s10, s8, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[32:33], v[36:37], 0, s[10:11]
	s_and_saveexec_b64 s[10:11], s[0:1]
	s_cbranch_execz .LBB0_234
	v_mov_b32_e32 v40, s6
	v_mov_b32_e32 v41, s7
	v_mov_b32_e32 v42, 2
	v_mov_b32_e32 v43, 1
	global_store_dwordx4 v[32:33], v[40:43], off offset:8
.LBB0_234:
	s_or_b64 exec, exec, s[10:11]
	s_lshl_b64 s[6:7], s[8:9], 12
	v_lshl_add_u64 v[40:41], v[38:39], 0, s[6:7]
	s_mov_b32 s8, 0
	v_mov_b32_e32 v42, 33
	v_mov_b32_e32 v43, v49
	v_mov_b32_e32 v44, v49
	v_mov_b32_e32 v45, v49
	v_readfirstlane_b32 s6, v40
	v_readfirstlane_b32 s7, v41
	s_mov_b32 s10, s8
	s_mov_b32 s11, s8
	s_mov_b32 s9, s8
	s_nop 1
	global_store_dwordx4 v48, v[42:45], s[6:7]
	s_nop 1
	v_mov_b64_e32 v[44:45], s[10:11]
	v_mov_b64_e32 v[42:43], s[8:9]
	global_store_dwordx4 v48, v[42:45], s[6:7] offset:16
	global_store_dwordx4 v48, v[42:45], s[6:7] offset:32
	global_store_dwordx4 v48, v[42:45], s[6:7] offset:48
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_242
	v_mov_b32_e32 v44, 0
	global_load_dwordx2 v[56:57], v44, s[4:5] offset:32 sc0 sc1
	global_load_dwordx2 v[38:39], v44, s[4:5] offset:40
	v_mov_b32_e32 v54, s2
	v_mov_b32_e32 v55, s3
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s8, v38
	v_readfirstlane_b32 s9, v39
	s_and_b64 s[8:9], s[8:9], s[2:3]
	s_mul_i32 s9, s9, 24
	s_mul_hi_u32 s10, s8, 24
	s_mul_i32 s8, s8, 24
	s_add_i32 s9, s10, s9
	v_lshl_add_u64 v[42:43], v[36:37], 0, s[8:9]
	global_store_dwordx2 v[42:43], v[56:57], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[38:39], v44, v[54:57], s[4:5] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[38:39], v[56:57]
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_238
	s_mov_b64 s[10:11], 0
.LBB0_237:
	s_sleep 1
	global_store_dwordx2 v[42:43], v[38:39], off
	v_mov_b32_e32 v36, s2
	v_mov_b32_e32 v37, s3
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v44, v[36:39], s[4:5] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[36:37], v[38:39]
	s_or_b64 s[10:11], vcc, s[10:11]
	v_mov_b64_e32 v[38:39], v[36:37]
	s_andn2_b64 exec, exec, s[10:11]
	s_cbranch_execnz .LBB0_237
.LBB0_238:
	s_or_b64 exec, exec, s[8:9]
	v_mov_b32_e32 v39, 0
	global_load_dwordx2 v[36:37], v39, s[4:5] offset:16
	s_mov_b64 s[8:9], exec
	v_mbcnt_lo_u32_b32 v38, s8, 0
	v_mbcnt_hi_u32_b32 v38, s9, v38
	v_cmp_eq_u32_e32 vcc, 0, v38
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_240
	s_bcnt1_i32_b64 s8, s[8:9]
	v_mov_b32_e32 v38, s8
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[36:37], v[38:39], off offset:8 sc1
.LBB0_240:
	s_or_b64 exec, exec, s[10:11]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[38:39], v[36:37], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[38:39]
	s_cbranch_vccnz .LBB0_242
	global_load_dword v36, v[36:37], off offset:24
	v_mov_b32_e32 v37, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s8, v36
	s_and_b32 m0, s8, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[38:39], v[36:37], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_242:
	s_or_b64 exec, exec, s[6:7]
	v_lshl_add_u64 v[36:37], v[40:41], 0, v[48:49]
	s_branch .LBB0_246
.LBB0_243:
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s6, v38
	s_cmp_eq_u32 s6, 0
	s_cbranch_scc1 .LBB0_245
	s_sleep 1
	s_cbranch_execnz .LBB0_246
	s_branch .LBB0_248
.LBB0_245:
	s_branch .LBB0_248
.LBB0_246:
	v_mov_b32_e32 v38, 1
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_243
	global_load_dword v38, v[32:33], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v38, 1, v38
	s_branch .LBB0_243
.LBB0_248:
	global_load_dwordx2 v[36:37], v[36:37], off
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_251
	v_mov_b32_e32 v42, 0
	global_load_dwordx2 v[32:33], v42, s[4:5] offset:40
	global_load_dwordx2 v[56:57], v42, s[4:5] offset:24 sc0 sc1
	global_load_dwordx2 v[38:39], v42, s[4:5]
	s_waitcnt vmcnt(2)
	v_readfirstlane_b32 s8, v32
	v_readfirstlane_b32 s9, v33
	s_add_u32 s10, s8, 1
	s_addc_u32 s11, s9, 0
	s_add_u32 s0, s10, s2
	s_addc_u32 s1, s11, s3
	s_cmp_eq_u64 s[0:1], 0
	s_cselect_b32 s1, s11, s1
	s_cselect_b32 s0, s10, s0
	s_and_b64 s[2:3], s[0:1], s[8:9]
	s_mul_i32 s3, s3, 24
	s_mul_hi_u32 s8, s2, 24
	s_mul_i32 s2, s2, 24
	s_add_i32 s3, s8, s3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[32:33], v[38:39], 0, s[2:3]
	v_mov_b32_e32 v54, s0
	global_store_dwordx2 v[32:33], v[56:57], off
	v_mov_b32_e32 v55, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[40:41], v42, v[54:57], s[4:5] offset:24 sc0 sc1
	s_mov_b64 s[2:3], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[40:41], v[56:57]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_251
.LBB0_250:
	s_sleep 1
	global_store_dwordx2 v[32:33], v[40:41], off
	v_mov_b32_e32 v38, s0
	v_mov_b32_e32 v39, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[38:39], v42, v[38:41], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[38:39], v[40:41]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[40:41], v[38:39]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_250
.LBB0_251:
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s0, v46
	v_mov_b64_e32 v[44:45], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v46
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_cbranch_execz .LBB0_257
	v_mov_b32_e32 v38, 0
	global_load_dwordx2 v[42:43], v38, s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[32:33], v38, s[4:5] offset:40
	global_load_dwordx2 v[40:41], v38, s[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v32, v32, v42
	v_and_b32_e32 v33, v33, v43
	v_mul_lo_u32 v33, v33, 24
	v_mul_hi_u32 v39, v32, 24
	v_add_u32_e32 v33, v39, v33
	v_mul_lo_u32 v32, v32, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[32:33], v[40:41], 0, v[32:33]
	global_load_dwordx2 v[40:41], v[32:33], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[44:45], v38, v[40:43], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[44:45], v[42:43]
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_256
	s_mov_b64 s[8:9], 0
.LBB0_254:
	s_sleep 1
	global_load_dwordx2 v[32:33], v38, s[4:5] offset:40
	global_load_dwordx2 v[40:41], v38, s[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v32, v32, v44
	v_and_b32_e32 v39, v33, v45
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[32:33], s[10:11], v32, 24, v[40:41]
	v_mov_b32_e32 v40, v33
	v_mad_u64_u32 v[40:41], s[10:11], v39, 24, v[40:41]
	v_mov_b32_e32 v33, v40
	global_load_dwordx2 v[42:43], v[32:33], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v38, v[42:45], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[32:33], v[44:45]
	s_or_b64 s[8:9], vcc, s[8:9]
	v_mov_b64_e32 v[44:45], v[32:33]
	s_andn2_b64 exec, exec, s[8:9]
	s_cbranch_execnz .LBB0_254
	s_or_b64 exec, exec, s[8:9]
	v_mov_b64_e32 v[44:45], v[32:33]
.LBB0_256:
	s_or_b64 exec, exec, s[6:7]
.LBB0_257:
	s_or_b64 exec, exec, s[2:3]
	v_mov_b32_e32 v49, 0
	global_load_dwordx2 v[32:33], v49, s[4:5] offset:40
	global_load_dwordx4 v[40:43], v49, s[4:5]
	v_readfirstlane_b32 s3, v45
	v_readfirstlane_b32 s2, v44
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s8, v32
	v_readfirstlane_b32 s9, v33
	s_and_b64 s[8:9], s[8:9], s[2:3]
	s_mul_i32 s10, s9, 24
	s_mul_hi_u32 s11, s8, 24
	s_add_i32 s11, s11, s10
	s_mul_i32 s10, s8, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[32:33], v[40:41], 0, s[10:11]
	s_and_saveexec_b64 s[10:11], s[0:1]
	s_cbranch_execz .LBB0_259
	v_mov_b32_e32 v54, s6
	v_mov_b32_e32 v55, s7
	v_mov_b32_e32 v56, 2
	v_mov_b32_e32 v57, 1
	global_store_dwordx4 v[32:33], v[54:57], off offset:8
.LBB0_259:
	s_or_b64 exec, exec, s[10:11]
	s_lshl_b64 s[6:7], s[8:9], 12
	v_lshl_add_u64 v[42:43], v[42:43], 0, s[6:7]
	v_or_b32_e32 v36, 0xe0, v36
	v_mov_b32_e32 v39, 0x6425203d
	v_mov_b32_e32 v38, 0x20444954
	v_readfirstlane_b32 s6, v42
	v_readfirstlane_b32 s7, v43
	s_nop 4
	global_store_dwordx4 v48, v[36:39], s[6:7]
	s_nop 1
	v_mov_b32_e32 v36, 0x4c46202c
	v_mov_b32_e32 v37, 0x3d204741
	v_mov_b32_e32 v38, 0x2c642520
	v_mov_b32_e32 v39, 0x6e6f6320
	global_store_dwordx4 v48, v[36:39], s[6:7] offset:16
	s_nop 1
	v_mov_b32_e32 v36, 0x69746964
	v_mov_b32_e32 v37, 0x3d206e6f
	v_mov_b32_e32 v38, 0x2c716520
	v_mov_b32_e32 v39, 0x5f736920
	global_store_dwordx4 v48, v[36:39], s[6:7] offset:32
	s_nop 1
	v_mov_b32_e32 v36, 0x69736f70
	v_mov_b32_e32 v37, 0x65766974
	v_mov_b32_e32 v38, 0x25203d20
	v_mov_b32_e32 v39, 0xa64
	global_store_dwordx4 v48, v[36:39], s[6:7] offset:48
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_267
	v_mov_b32_e32 v44, 0
	global_load_dwordx2 v[56:57], v44, s[4:5] offset:32 sc0 sc1
	global_load_dwordx2 v[36:37], v44, s[4:5] offset:40
	v_mov_b32_e32 v54, s2
	v_mov_b32_e32 v55, s3
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s8, v36
	v_readfirstlane_b32 s9, v37
	s_and_b64 s[8:9], s[8:9], s[2:3]
	s_mul_i32 s9, s9, 24
	s_mul_hi_u32 s10, s8, 24
	s_mul_i32 s8, s8, 24
	s_add_i32 s9, s10, s9
	v_lshl_add_u64 v[40:41], v[40:41], 0, s[8:9]
	global_store_dwordx2 v[40:41], v[56:57], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[38:39], v44, v[54:57], s[4:5] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[38:39], v[56:57]
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_263
	s_mov_b64 s[10:11], 0
.LBB0_262:
	s_sleep 1
	global_store_dwordx2 v[40:41], v[38:39], off
	v_mov_b32_e32 v36, s2
	v_mov_b32_e32 v37, s3
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v44, v[36:39], s[4:5] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[36:37], v[38:39]
	s_or_b64 s[10:11], vcc, s[10:11]
	v_mov_b64_e32 v[38:39], v[36:37]
	s_andn2_b64 exec, exec, s[10:11]
	s_cbranch_execnz .LBB0_262
.LBB0_263:
	s_or_b64 exec, exec, s[8:9]
	v_mov_b32_e32 v39, 0
	global_load_dwordx2 v[36:37], v39, s[4:5] offset:16
	s_mov_b64 s[8:9], exec
	v_mbcnt_lo_u32_b32 v38, s8, 0
	v_mbcnt_hi_u32_b32 v38, s9, v38
	v_cmp_eq_u32_e32 vcc, 0, v38
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_265
	s_bcnt1_i32_b64 s8, s[8:9]
	v_mov_b32_e32 v38, s8
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[36:37], v[38:39], off offset:8 sc1
.LBB0_265:
	s_or_b64 exec, exec, s[10:11]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[38:39], v[36:37], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[38:39]
	s_cbranch_vccnz .LBB0_267
	global_load_dword v36, v[36:37], off offset:24
	v_mov_b32_e32 v37, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s8, v36
	s_and_b32 m0, s8, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[38:39], v[36:37], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_267:
	s_or_b64 exec, exec, s[6:7]
	v_lshl_add_u64 v[36:37], v[42:43], 0, v[48:49]
	s_branch .LBB0_271
.LBB0_268:
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s6, v38
	s_cmp_eq_u32 s6, 0
	s_cbranch_scc1 .LBB0_270
	s_sleep 1
	s_cbranch_execnz .LBB0_271
	s_branch .LBB0_273
.LBB0_270:
	s_branch .LBB0_273
.LBB0_271:
	v_mov_b32_e32 v38, 1
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_268
	global_load_dword v38, v[32:33], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v38, 1, v38
	s_branch .LBB0_268
.LBB0_273:
	global_load_dwordx2 v[32:33], v[36:37], off
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_276
	v_mov_b32_e32 v42, 0
	global_load_dwordx2 v[36:37], v42, s[4:5] offset:40
	global_load_dwordx2 v[56:57], v42, s[4:5] offset:24 sc0 sc1
	global_load_dwordx2 v[38:39], v42, s[4:5]
	s_waitcnt vmcnt(2)
	v_readfirstlane_b32 s8, v36
	v_readfirstlane_b32 s9, v37
	s_add_u32 s10, s8, 1
	s_addc_u32 s11, s9, 0
	s_add_u32 s0, s10, s2
	s_addc_u32 s1, s11, s3
	s_cmp_eq_u64 s[0:1], 0
	s_cselect_b32 s1, s11, s1
	s_cselect_b32 s0, s10, s0
	s_and_b64 s[2:3], s[0:1], s[8:9]
	s_mul_i32 s3, s3, 24
	s_mul_hi_u32 s8, s2, 24
	s_mul_i32 s2, s2, 24
	s_add_i32 s3, s8, s3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[40:41], v[38:39], 0, s[2:3]
	v_mov_b32_e32 v54, s0
	global_store_dwordx2 v[40:41], v[56:57], off
	v_mov_b32_e32 v55, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[38:39], v42, v[54:57], s[4:5] offset:24 sc0 sc1
	s_mov_b64 s[2:3], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[38:39], v[56:57]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_276
.LBB0_275:
	s_sleep 1
	global_store_dwordx2 v[40:41], v[38:39], off
	v_mov_b32_e32 v36, s0
	v_mov_b32_e32 v37, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v42, v[36:39], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[36:37], v[38:39]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[38:39], v[36:37]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_275
.LBB0_276:
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s0, v46
	v_mov_b64_e32 v[40:41], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v46
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_cbranch_execz .LBB0_282
	v_mov_b32_e32 v42, 0
	global_load_dwordx2 v[38:39], v42, s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[36:37], v42, s[4:5] offset:40
	global_load_dwordx2 v[40:41], v42, s[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v36, v36, v38
	v_and_b32_e32 v37, v37, v39
	v_mul_lo_u32 v37, v37, 24
	v_mul_hi_u32 v43, v36, 24
	v_add_u32_e32 v37, v43, v37
	v_mul_lo_u32 v36, v36, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[36:37], v[40:41], 0, v[36:37]
	global_load_dwordx2 v[36:37], v[36:37], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[40:41], v42, v[36:39], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[40:41], v[38:39]
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_281
	s_mov_b64 s[8:9], 0
.LBB0_279:
	s_sleep 1
	global_load_dwordx2 v[36:37], v42, s[4:5] offset:40
	global_load_dwordx2 v[38:39], v42, s[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v36, v36, v40
	v_and_b32_e32 v43, v37, v41
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[36:37], s[10:11], v36, 24, v[38:39]
	v_mov_b32_e32 v38, v37
	v_mad_u64_u32 v[38:39], s[10:11], v43, 24, v[38:39]
	v_mov_b32_e32 v37, v38
	global_load_dwordx2 v[38:39], v[36:37], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v42, v[38:41], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[36:37], v[40:41]
	s_or_b64 s[8:9], vcc, s[8:9]
	v_mov_b64_e32 v[40:41], v[36:37]
	s_andn2_b64 exec, exec, s[8:9]
	s_cbranch_execnz .LBB0_279
	s_or_b64 exec, exec, s[8:9]
	v_mov_b64_e32 v[40:41], v[36:37]
.LBB0_281:
	s_or_b64 exec, exec, s[6:7]
.LBB0_282:
	s_or_b64 exec, exec, s[2:3]
	v_mov_b32_e32 v44, 0
	global_load_dwordx2 v[42:43], v44, s[4:5] offset:40
	global_load_dwordx4 v[36:39], v44, s[4:5]
	v_readfirstlane_b32 s7, v41
	v_readfirstlane_b32 s6, v40
	s_mov_b64 s[2:3], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s8, v42
	v_readfirstlane_b32 s9, v43
	s_and_b64 s[8:9], s[8:9], s[6:7]
	s_mul_i32 s10, s9, 24
	s_mul_hi_u32 s11, s8, 24
	s_add_i32 s11, s11, s10
	s_mul_i32 s10, s8, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[40:41], v[36:37], 0, s[10:11]
	s_and_saveexec_b64 s[10:11], s[0:1]
	s_cbranch_execz .LBB0_284
	v_mov_b32_e32 v42, s2
	v_mov_b32_e32 v43, s3
	v_mov_b32_e32 v44, 2
	v_mov_b32_e32 v45, 1
	global_store_dwordx4 v[40:41], v[42:45], off offset:8
.LBB0_284:
	s_or_b64 exec, exec, s[10:11]
	s_lshl_b64 s[2:3], s[8:9], 12
	v_lshl_add_u64 v[38:39], v[38:39], 0, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], 2, v34
	s_mov_b32 s8, 0
	v_mov_b32_e32 v42, 2
	v_cndmask_b32_e64 v44, 0, 1, s[2:3]
	v_mov_b32_e32 v45, s8
	v_readfirstlane_b32 s12, v38
	v_readfirstlane_b32 s13, v39
	v_mov_b32_e32 v43, s8
	s_mov_b32 s10, s8
	s_mov_b32 s11, s8
	v_and_b32_e32 v32, 0xffffff1d, v32
	s_mov_b32 s9, s8
	global_store_dwordx4 v48, v[42:45], s[12:13] offset:16
	v_or_b32_e32 v32, 0x62, v32
	global_store_dwordx4 v48, v[32:35], s[12:13]
	v_mov_b64_e32 v[44:45], s[10:11]
	v_mov_b64_e32 v[42:43], s[8:9]
	global_store_dwordx4 v48, v[42:45], s[12:13] offset:32
	global_store_dwordx4 v48, v[42:45], s[12:13] offset:48
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_292
	v_mov_b32_e32 v42, 0
	global_load_dwordx2 v[56:57], v42, s[4:5] offset:32 sc0 sc1
	global_load_dwordx2 v[32:33], v42, s[4:5] offset:40
	v_mov_b32_e32 v54, s6
	v_mov_b32_e32 v55, s7
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v32
	v_readfirstlane_b32 s11, v33
	s_and_b64 s[10:11], s[10:11], s[6:7]
	s_mul_i32 s11, s11, 24
	s_mul_hi_u32 s12, s10, 24
	s_mul_i32 s10, s10, 24
	s_add_i32 s11, s12, s11
	v_lshl_add_u64 v[32:33], v[36:37], 0, s[10:11]
	global_store_dwordx2 v[32:33], v[56:57], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[38:39], v42, v[54:57], s[4:5] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[38:39], v[56:57]
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_288
	s_mov_b64 s[12:13], 0
.LBB0_287:
	s_sleep 1
	global_store_dwordx2 v[32:33], v[38:39], off
	v_mov_b32_e32 v36, s6
	v_mov_b32_e32 v37, s7
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v42, v[36:39], s[4:5] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[36:37], v[38:39]
	s_or_b64 s[12:13], vcc, s[12:13]
	v_mov_b64_e32 v[38:39], v[36:37]
	s_andn2_b64 exec, exec, s[12:13]
	s_cbranch_execnz .LBB0_287
.LBB0_288:
	s_or_b64 exec, exec, s[10:11]
	v_mov_b32_e32 v37, 0
	global_load_dwordx2 v[32:33], v37, s[4:5] offset:16
	s_mov_b64 s[10:11], exec
	v_mbcnt_lo_u32_b32 v36, s10, 0
	v_mbcnt_hi_u32_b32 v36, s11, v36
	v_cmp_eq_u32_e32 vcc, 0, v36
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB0_290
	s_bcnt1_i32_b64 s10, s[10:11]
	v_mov_b32_e32 v36, s10
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[32:33], v[36:37], off offset:8 sc1
.LBB0_290:
	s_or_b64 exec, exec, s[12:13]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[36:37], v[32:33], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[36:37]
	s_cbranch_vccnz .LBB0_292
	global_load_dword v32, v[32:33], off offset:24
	v_mov_b32_e32 v33, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v32
	s_and_b32 m0, s10, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[36:37], v[32:33], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_292:
	s_or_b64 exec, exec, s[8:9]
	s_branch .LBB0_296
.LBB0_293:
	s_or_b64 exec, exec, s[8:9]
	v_readfirstlane_b32 s8, v32
	s_cmp_eq_u32 s8, 0
	s_cbranch_scc1 .LBB0_295
	s_sleep 1
	s_cbranch_execnz .LBB0_296
	s_branch .LBB0_298
.LBB0_295:
	s_branch .LBB0_298
.LBB0_296:
	v_mov_b32_e32 v32, 1
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_293
	global_load_dword v32, v[40:41], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v32, 1, v32
	s_branch .LBB0_293
.LBB0_298:
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_301
	v_mov_b32_e32 v40, 0
	global_load_dwordx2 v[32:33], v40, s[4:5] offset:40
	global_load_dwordx2 v[44:45], v40, s[4:5] offset:24 sc0 sc1
	global_load_dwordx2 v[36:37], v40, s[4:5]
	s_waitcnt vmcnt(2)
	v_readfirstlane_b32 s10, v32
	v_readfirstlane_b32 s11, v33
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
	v_lshl_add_u64 v[32:33], v[36:37], 0, s[6:7]
	v_mov_b32_e32 v42, s0
	global_store_dwordx2 v[32:33], v[44:45], off
	v_mov_b32_e32 v43, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[38:39], v40, v[42:45], s[4:5] offset:24 sc0 sc1
	s_mov_b64 s[6:7], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[38:39], v[44:45]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_301
.LBB0_300:
	s_sleep 1
	global_store_dwordx2 v[32:33], v[38:39], off
	v_mov_b32_e32 v36, s0
	v_mov_b32_e32 v37, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v40, v[36:39], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[36:37], v[38:39]
	s_or_b64 s[6:7], vcc, s[6:7]
	v_mov_b64_e32 v[38:39], v[36:37]
	s_andn2_b64 exec, exec, s[6:7]
	s_cbranch_execnz .LBB0_300
.LBB0_301:
	s_or_b64 exec, exec, s[8:9]
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_377
	v_readfirstlane_b32 s0, v46
	v_mov_b64_e32 v[40:41], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v46
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_cbranch_execz .LBB0_308
	v_mov_b32_e32 v36, 0
	global_load_dwordx2 v[42:43], v36, s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[32:33], v36, s[4:5] offset:40
	global_load_dwordx2 v[38:39], v36, s[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v32, v32, v42
	v_and_b32_e32 v33, v33, v43
	v_mul_lo_u32 v33, v33, 24
	v_mul_hi_u32 v37, v32, 24
	v_add_u32_e32 v33, v37, v33
	v_mul_lo_u32 v32, v32, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[32:33], v[38:39], 0, v[32:33]
	global_load_dwordx2 v[40:41], v[32:33], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[40:41], v36, v[40:43], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[40:41], v[42:43]
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_307
	s_mov_b64 s[10:11], 0
.LBB0_305:
	s_sleep 1
	global_load_dwordx2 v[32:33], v36, s[4:5] offset:40
	global_load_dwordx2 v[38:39], v36, s[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v32, v32, v40
	v_and_b32_e32 v37, v33, v41
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[32:33], s[12:13], v32, 24, v[38:39]
	v_mov_b32_e32 v38, v33
	v_mad_u64_u32 v[38:39], s[12:13], v37, 24, v[38:39]
	v_mov_b32_e32 v33, v38
	global_load_dwordx2 v[38:39], v[32:33], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v36, v[38:41], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[32:33], v[40:41]
	s_or_b64 s[10:11], vcc, s[10:11]
	v_mov_b64_e32 v[40:41], v[32:33]
	s_andn2_b64 exec, exec, s[10:11]
	s_cbranch_execnz .LBB0_305
	s_or_b64 exec, exec, s[10:11]
	v_mov_b64_e32 v[40:41], v[32:33]
.LBB0_307:
	s_or_b64 exec, exec, s[8:9]
.LBB0_308:
	s_or_b64 exec, exec, s[2:3]
	v_mov_b32_e32 v49, 0
	global_load_dwordx2 v[32:33], v49, s[4:5] offset:40
	global_load_dwordx4 v[36:39], v49, s[4:5]
	v_readfirstlane_b32 s3, v41
	v_readfirstlane_b32 s2, v40
	s_mov_b64 s[8:9], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s10, v32
	v_readfirstlane_b32 s11, v33
	s_and_b64 s[10:11], s[10:11], s[2:3]
	s_mul_i32 s12, s11, 24
	s_mul_hi_u32 s13, s10, 24
	s_add_i32 s13, s13, s12
	s_mul_i32 s12, s10, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[32:33], v[36:37], 0, s[12:13]
	s_and_saveexec_b64 s[12:13], s[0:1]
	s_cbranch_execz .LBB0_310
	v_mov_b32_e32 v40, s8
	v_mov_b32_e32 v41, s9
	v_mov_b32_e32 v42, 2
	v_mov_b32_e32 v43, 1
	global_store_dwordx4 v[32:33], v[40:43], off offset:8
.LBB0_310:
	s_or_b64 exec, exec, s[12:13]
	s_lshl_b64 s[8:9], s[10:11], 12
	v_lshl_add_u64 v[40:41], v[38:39], 0, s[8:9]
	s_mov_b32 s8, 0
	v_mov_b32_e32 v42, 33
	v_mov_b32_e32 v43, v49
	v_mov_b32_e32 v44, v49
	v_mov_b32_e32 v45, v49
	v_readfirstlane_b32 s12, v40
	v_readfirstlane_b32 s13, v41
	s_mov_b32 s10, s8
	s_mov_b32 s11, s8
	s_mov_b32 s9, s8
	s_nop 1
	global_store_dwordx4 v48, v[42:45], s[12:13]
	s_nop 1
	v_mov_b64_e32 v[44:45], s[10:11]
	v_mov_b64_e32 v[42:43], s[8:9]
	global_store_dwordx4 v48, v[42:45], s[12:13] offset:16
	global_store_dwordx4 v48, v[42:45], s[12:13] offset:32
	global_store_dwordx4 v48, v[42:45], s[12:13] offset:48
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_318
	v_mov_b32_e32 v44, 0
	global_load_dwordx2 v[56:57], v44, s[4:5] offset:32 sc0 sc1
	global_load_dwordx2 v[38:39], v44, s[4:5] offset:40
	v_mov_b32_e32 v54, s2
	v_mov_b32_e32 v55, s3
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v38
	v_readfirstlane_b32 s11, v39
	s_and_b64 s[10:11], s[10:11], s[2:3]
	s_mul_i32 s11, s11, 24
	s_mul_hi_u32 s12, s10, 24
	s_mul_i32 s10, s10, 24
	s_add_i32 s11, s12, s11
	v_lshl_add_u64 v[42:43], v[36:37], 0, s[10:11]
	global_store_dwordx2 v[42:43], v[56:57], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[38:39], v44, v[54:57], s[4:5] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[38:39], v[56:57]
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_314
	s_mov_b64 s[12:13], 0
.LBB0_313:
	s_sleep 1
	global_store_dwordx2 v[42:43], v[38:39], off
	v_mov_b32_e32 v36, s2
	v_mov_b32_e32 v37, s3
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v44, v[36:39], s[4:5] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[36:37], v[38:39]
	s_or_b64 s[12:13], vcc, s[12:13]
	v_mov_b64_e32 v[38:39], v[36:37]
	s_andn2_b64 exec, exec, s[12:13]
	s_cbranch_execnz .LBB0_313
.LBB0_314:
	s_or_b64 exec, exec, s[10:11]
	v_mov_b32_e32 v39, 0
	global_load_dwordx2 v[36:37], v39, s[4:5] offset:16
	s_mov_b64 s[10:11], exec
	v_mbcnt_lo_u32_b32 v38, s10, 0
	v_mbcnt_hi_u32_b32 v38, s11, v38
	v_cmp_eq_u32_e32 vcc, 0, v38
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB0_316
	s_bcnt1_i32_b64 s10, s[10:11]
	v_mov_b32_e32 v38, s10
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[36:37], v[38:39], off offset:8 sc1
.LBB0_316:
	s_or_b64 exec, exec, s[12:13]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[38:39], v[36:37], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[38:39]
	s_cbranch_vccnz .LBB0_318
	global_load_dword v36, v[36:37], off offset:24
	v_mov_b32_e32 v37, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v36
	s_and_b32 m0, s10, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[38:39], v[36:37], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_318:
	s_or_b64 exec, exec, s[8:9]
	v_lshl_add_u64 v[36:37], v[40:41], 0, v[48:49]
	s_branch .LBB0_322
.LBB0_319:
	s_or_b64 exec, exec, s[8:9]
	v_readfirstlane_b32 s8, v38
	s_cmp_eq_u32 s8, 0
	s_cbranch_scc1 .LBB0_321
	s_sleep 1
	s_cbranch_execnz .LBB0_322
	s_branch .LBB0_324
.LBB0_321:
	s_branch .LBB0_324
.LBB0_322:
	v_mov_b32_e32 v38, 1
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_319
	global_load_dword v38, v[32:33], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v38, 1, v38
	s_branch .LBB0_319
.LBB0_324:
	global_load_dwordx2 v[36:37], v[36:37], off
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_327
	v_mov_b32_e32 v42, 0
	global_load_dwordx2 v[32:33], v42, s[4:5] offset:40
	global_load_dwordx2 v[56:57], v42, s[4:5] offset:24 sc0 sc1
	global_load_dwordx2 v[38:39], v42, s[4:5]
	s_waitcnt vmcnt(2)
	v_readfirstlane_b32 s10, v32
	v_readfirstlane_b32 s11, v33
	s_add_u32 s12, s10, 1
	s_addc_u32 s13, s11, 0
	s_add_u32 s0, s12, s2
	s_addc_u32 s1, s13, s3
	s_cmp_eq_u64 s[0:1], 0
	s_cselect_b32 s1, s13, s1
	s_cselect_b32 s0, s12, s0
	s_and_b64 s[2:3], s[0:1], s[10:11]
	s_mul_i32 s3, s3, 24
	s_mul_hi_u32 s10, s2, 24
	s_mul_i32 s2, s2, 24
	s_add_i32 s3, s10, s3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[32:33], v[38:39], 0, s[2:3]
	v_mov_b32_e32 v54, s0
	global_store_dwordx2 v[32:33], v[56:57], off
	v_mov_b32_e32 v55, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[40:41], v42, v[54:57], s[4:5] offset:24 sc0 sc1
	s_mov_b64 s[2:3], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[40:41], v[56:57]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_327
.LBB0_326:
	s_sleep 1
	global_store_dwordx2 v[32:33], v[40:41], off
	v_mov_b32_e32 v38, s0
	v_mov_b32_e32 v39, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[38:39], v42, v[38:41], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[38:39], v[40:41]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[40:41], v[38:39]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_326
.LBB0_327:
	s_or_b64 exec, exec, s[8:9]
	v_readfirstlane_b32 s0, v46
	v_mov_b64_e32 v[44:45], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v46
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_cbranch_execz .LBB0_333
	v_mov_b32_e32 v38, 0
	global_load_dwordx2 v[42:43], v38, s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[32:33], v38, s[4:5] offset:40
	global_load_dwordx2 v[40:41], v38, s[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v32, v32, v42
	v_and_b32_e32 v33, v33, v43
	v_mul_lo_u32 v33, v33, 24
	v_mul_hi_u32 v39, v32, 24
	v_add_u32_e32 v33, v39, v33
	v_mul_lo_u32 v32, v32, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[32:33], v[40:41], 0, v[32:33]
	global_load_dwordx2 v[40:41], v[32:33], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[44:45], v38, v[40:43], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[44:45], v[42:43]
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_332
	s_mov_b64 s[10:11], 0
.LBB0_330:
	s_sleep 1
	global_load_dwordx2 v[32:33], v38, s[4:5] offset:40
	global_load_dwordx2 v[40:41], v38, s[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v32, v32, v44
	v_and_b32_e32 v39, v33, v45
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[32:33], s[12:13], v32, 24, v[40:41]
	v_mov_b32_e32 v40, v33
	v_mad_u64_u32 v[40:41], s[12:13], v39, 24, v[40:41]
	v_mov_b32_e32 v33, v40
	global_load_dwordx2 v[42:43], v[32:33], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v38, v[42:45], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[32:33], v[44:45]
	s_or_b64 s[10:11], vcc, s[10:11]
	v_mov_b64_e32 v[44:45], v[32:33]
	s_andn2_b64 exec, exec, s[10:11]
	s_cbranch_execnz .LBB0_330
	s_or_b64 exec, exec, s[10:11]
	v_mov_b64_e32 v[44:45], v[32:33]
.LBB0_332:
	s_or_b64 exec, exec, s[8:9]
.LBB0_333:
	s_or_b64 exec, exec, s[2:3]
	v_mov_b32_e32 v49, 0
	global_load_dwordx2 v[32:33], v49, s[4:5] offset:40
	global_load_dwordx4 v[40:43], v49, s[4:5]
	v_readfirstlane_b32 s3, v45
	v_readfirstlane_b32 s2, v44
	s_mov_b64 s[8:9], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s10, v32
	v_readfirstlane_b32 s11, v33
	s_and_b64 s[10:11], s[10:11], s[2:3]
	s_mul_i32 s12, s11, 24
	s_mul_hi_u32 s13, s10, 24
	s_add_i32 s13, s13, s12
	s_mul_i32 s12, s10, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[32:33], v[40:41], 0, s[12:13]
	s_and_saveexec_b64 s[12:13], s[0:1]
	s_cbranch_execz .LBB0_335
	v_mov_b32_e32 v54, s8
	v_mov_b32_e32 v55, s9
	v_mov_b32_e32 v56, 2
	v_mov_b32_e32 v57, 1
	global_store_dwordx4 v[32:33], v[54:57], off offset:8
.LBB0_335:
	s_or_b64 exec, exec, s[12:13]
	s_lshl_b64 s[8:9], s[10:11], 12
	v_lshl_add_u64 v[42:43], v[42:43], 0, s[8:9]
	v_and_b32_e32 v36, 0xffffff1f, v36
	v_or_b32_e32 v36, 0xa0, v36
	v_mov_b32_e32 v39, 0x7270205d
	v_mov_b32_e32 v38, 0x64255b43
	v_readfirstlane_b32 s8, v42
	v_readfirstlane_b32 s9, v43
	s_nop 4
	global_store_dwordx4 v48, v[36:39], s[8:9]
	s_nop 1
	v_mov_b32_e32 v36, 0x65746e69
	v_mov_b32_e32 v37, 0x6e692064
	v_mov_b32_e32 v38, 0x65646973
	v_mov_b32_e32 v39, 0x72656b20
	global_store_dwordx4 v48, v[36:39], s[8:9] offset:16
	s_nop 1
	v_mov_b32_e32 v36, 0x206c656e
	v_mov_b32_e32 v37, 0x3425203d
	v_mov_b32_e32 v38, 0xa66332e
	v_mov_b32_e32 v39, v49
	global_store_dwordx4 v48, v[36:39], s[8:9] offset:32
	s_nop 1
	v_mov_b32_e32 v36, v49
	v_mov_b32_e32 v37, v49
	v_mov_b32_e32 v38, v49
	global_store_dwordx4 v48, v[36:39], s[8:9] offset:48
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_343
	v_mov_b32_e32 v44, 0
	global_load_dwordx2 v[56:57], v44, s[4:5] offset:32 sc0 sc1
	global_load_dwordx2 v[36:37], v44, s[4:5] offset:40
	v_mov_b32_e32 v54, s2
	v_mov_b32_e32 v55, s3
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v36
	v_readfirstlane_b32 s11, v37
	s_and_b64 s[10:11], s[10:11], s[2:3]
	s_mul_i32 s11, s11, 24
	s_mul_hi_u32 s12, s10, 24
	s_mul_i32 s10, s10, 24
	s_add_i32 s11, s12, s11
	v_lshl_add_u64 v[40:41], v[40:41], 0, s[10:11]
	global_store_dwordx2 v[40:41], v[56:57], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[38:39], v44, v[54:57], s[4:5] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[38:39], v[56:57]
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_339
	s_mov_b64 s[12:13], 0
.LBB0_338:
	s_sleep 1
	global_store_dwordx2 v[40:41], v[38:39], off
	v_mov_b32_e32 v36, s2
	v_mov_b32_e32 v37, s3
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v44, v[36:39], s[4:5] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[36:37], v[38:39]
	s_or_b64 s[12:13], vcc, s[12:13]
	v_mov_b64_e32 v[38:39], v[36:37]
	s_andn2_b64 exec, exec, s[12:13]
	s_cbranch_execnz .LBB0_338
.LBB0_339:
	s_or_b64 exec, exec, s[10:11]
	v_mov_b32_e32 v39, 0
	global_load_dwordx2 v[36:37], v39, s[4:5] offset:16
	s_mov_b64 s[10:11], exec
	v_mbcnt_lo_u32_b32 v38, s10, 0
	v_mbcnt_hi_u32_b32 v38, s11, v38
	v_cmp_eq_u32_e32 vcc, 0, v38
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB0_341
	s_bcnt1_i32_b64 s10, s[10:11]
	v_mov_b32_e32 v38, s10
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[36:37], v[38:39], off offset:8 sc1
.LBB0_341:
	s_or_b64 exec, exec, s[12:13]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[38:39], v[36:37], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[38:39]
	s_cbranch_vccnz .LBB0_343
	global_load_dword v36, v[36:37], off offset:24
	v_mov_b32_e32 v37, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v36
	s_and_b32 m0, s10, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[38:39], v[36:37], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_343:
	s_or_b64 exec, exec, s[8:9]
	v_lshl_add_u64 v[36:37], v[42:43], 0, v[48:49]
	s_branch .LBB0_347
.LBB0_344:
	s_or_b64 exec, exec, s[8:9]
	v_readfirstlane_b32 s8, v38
	s_cmp_eq_u32 s8, 0
	s_cbranch_scc1 .LBB0_346
	s_sleep 1
	s_cbranch_execnz .LBB0_347
	s_branch .LBB0_349
.LBB0_346:
	s_branch .LBB0_349
.LBB0_347:
	v_mov_b32_e32 v38, 1
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_344
	global_load_dword v38, v[32:33], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v38, 1, v38
	s_branch .LBB0_344
.LBB0_349:
	global_load_dwordx2 v[32:33], v[36:37], off
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_352
	v_mov_b32_e32 v42, 0
	global_load_dwordx2 v[36:37], v42, s[4:5] offset:40
	global_load_dwordx2 v[56:57], v42, s[4:5] offset:24 sc0 sc1
	global_load_dwordx2 v[38:39], v42, s[4:5]
	s_waitcnt vmcnt(2)
	v_readfirstlane_b32 s10, v36
	v_readfirstlane_b32 s11, v37
	s_add_u32 s12, s10, 1
	s_addc_u32 s13, s11, 0
	s_add_u32 s0, s12, s2
	s_addc_u32 s1, s13, s3
	s_cmp_eq_u64 s[0:1], 0
	s_cselect_b32 s1, s13, s1
	s_cselect_b32 s0, s12, s0
	s_and_b64 s[2:3], s[0:1], s[10:11]
	s_mul_i32 s3, s3, 24
	s_mul_hi_u32 s10, s2, 24
	s_mul_i32 s2, s2, 24
	s_add_i32 s3, s10, s3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[40:41], v[38:39], 0, s[2:3]
	v_mov_b32_e32 v54, s0
	global_store_dwordx2 v[40:41], v[56:57], off
	v_mov_b32_e32 v55, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[38:39], v42, v[54:57], s[4:5] offset:24 sc0 sc1
	s_mov_b64 s[2:3], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[38:39], v[56:57]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_352
.LBB0_351:
	s_sleep 1
	global_store_dwordx2 v[40:41], v[38:39], off
	v_mov_b32_e32 v36, s0
	v_mov_b32_e32 v37, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v42, v[36:39], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[36:37], v[38:39]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[38:39], v[36:37]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_351
.LBB0_352:
	s_or_b64 exec, exec, s[8:9]
	v_readfirstlane_b32 s0, v46
	v_mov_b64_e32 v[40:41], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v46
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_cbranch_execz .LBB0_358
	v_mov_b32_e32 v42, 0
	global_load_dwordx2 v[38:39], v42, s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[36:37], v42, s[4:5] offset:40
	global_load_dwordx2 v[40:41], v42, s[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v36, v36, v38
	v_and_b32_e32 v37, v37, v39
	v_mul_lo_u32 v37, v37, 24
	v_mul_hi_u32 v43, v36, 24
	v_add_u32_e32 v37, v43, v37
	v_mul_lo_u32 v36, v36, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[36:37], v[40:41], 0, v[36:37]
	global_load_dwordx2 v[36:37], v[36:37], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[40:41], v42, v[36:39], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[40:41], v[38:39]
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_357
	s_mov_b64 s[10:11], 0
.LBB0_355:
	s_sleep 1
	global_load_dwordx2 v[36:37], v42, s[4:5] offset:40
	global_load_dwordx2 v[38:39], v42, s[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v36, v36, v40
	v_and_b32_e32 v43, v37, v41
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[36:37], s[12:13], v36, 24, v[38:39]
	v_mov_b32_e32 v38, v37
	v_mad_u64_u32 v[38:39], s[12:13], v43, 24, v[38:39]
	v_mov_b32_e32 v37, v38
	global_load_dwordx2 v[38:39], v[36:37], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v42, v[38:41], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[36:37], v[40:41]
	s_or_b64 s[10:11], vcc, s[10:11]
	v_mov_b64_e32 v[40:41], v[36:37]
	s_andn2_b64 exec, exec, s[10:11]
	s_cbranch_execnz .LBB0_355
	s_or_b64 exec, exec, s[10:11]
	v_mov_b64_e32 v[40:41], v[36:37]
.LBB0_357:
	s_or_b64 exec, exec, s[8:9]
.LBB0_358:
	s_or_b64 exec, exec, s[2:3]
	v_mov_b32_e32 v44, 0
	global_load_dwordx2 v[42:43], v44, s[4:5] offset:40
	global_load_dwordx4 v[36:39], v44, s[4:5]
	v_readfirstlane_b32 s3, v41
	v_readfirstlane_b32 s2, v40
	s_mov_b64 s[8:9], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s10, v42
	v_readfirstlane_b32 s11, v43
	s_and_b64 s[10:11], s[10:11], s[2:3]
	s_mul_i32 s12, s11, 24
	s_mul_hi_u32 s13, s10, 24
	s_add_i32 s13, s13, s12
	s_mul_i32 s12, s10, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[40:41], v[36:37], 0, s[12:13]
	s_and_saveexec_b64 s[12:13], s[0:1]
	s_cbranch_execz .LBB0_360
	v_mov_b32_e32 v42, s8
	v_mov_b32_e32 v43, s9
	v_mov_b32_e32 v44, 2
	v_mov_b32_e32 v45, 1
	global_store_dwordx4 v[40:41], v[42:45], off offset:8
.LBB0_360:
	s_or_b64 exec, exec, s[12:13]
	s_lshl_b64 s[8:9], s[10:11], 12
	v_lshl_add_u64 v[38:39], v[38:39], 0, s[8:9]
	s_mov_b32 s8, 0
	v_cvt_f64_f32_e32 v[42:43], v52
	v_readfirstlane_b32 s12, v38
	v_readfirstlane_b32 s13, v39
	v_mov_b32_e32 v44, s8
	v_mov_b32_e32 v45, s8
	s_mov_b32 s10, s8
	s_mov_b32 s11, s8
	v_and_b32_e32 v32, 0xffffff1d, v32
	global_store_dwordx4 v48, v[42:45], s[12:13] offset:16
	s_mov_b32 s9, s8
	v_or_b32_e32 v32, 0x42, v32
	v_mov_b64_e32 v[44:45], s[10:11]
	v_mov_b64_e32 v[42:43], s[8:9]
	global_store_dwordx4 v48, v[32:35], s[12:13]
	global_store_dwordx4 v48, v[42:45], s[12:13] offset:32
	global_store_dwordx4 v48, v[42:45], s[12:13] offset:48
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_368
	v_mov_b32_e32 v42, 0
	global_load_dwordx2 v[56:57], v42, s[4:5] offset:32 sc0 sc1
	global_load_dwordx2 v[32:33], v42, s[4:5] offset:40
	v_mov_b32_e32 v54, s2
	v_mov_b32_e32 v55, s3
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v32
	v_readfirstlane_b32 s11, v33
	s_and_b64 s[10:11], s[10:11], s[2:3]
	s_mul_i32 s11, s11, 24
	s_mul_hi_u32 s12, s10, 24
	s_mul_i32 s10, s10, 24
	s_add_i32 s11, s12, s11
	v_lshl_add_u64 v[32:33], v[36:37], 0, s[10:11]
	global_store_dwordx2 v[32:33], v[56:57], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[38:39], v42, v[54:57], s[4:5] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[38:39], v[56:57]
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_364
	s_mov_b64 s[12:13], 0
.LBB0_363:
	s_sleep 1
	global_store_dwordx2 v[32:33], v[38:39], off
	v_mov_b32_e32 v36, s2
	v_mov_b32_e32 v37, s3
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v42, v[36:39], s[4:5] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[36:37], v[38:39]
	s_or_b64 s[12:13], vcc, s[12:13]
	v_mov_b64_e32 v[38:39], v[36:37]
	s_andn2_b64 exec, exec, s[12:13]
	s_cbranch_execnz .LBB0_363
.LBB0_364:
	s_or_b64 exec, exec, s[10:11]
	v_mov_b32_e32 v37, 0
	global_load_dwordx2 v[32:33], v37, s[4:5] offset:16
	s_mov_b64 s[10:11], exec
	v_mbcnt_lo_u32_b32 v36, s10, 0
	v_mbcnt_hi_u32_b32 v36, s11, v36
	v_cmp_eq_u32_e32 vcc, 0, v36
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB0_366
	s_bcnt1_i32_b64 s10, s[10:11]
	v_mov_b32_e32 v36, s10
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[32:33], v[36:37], off offset:8 sc1
.LBB0_366:
	s_or_b64 exec, exec, s[12:13]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[36:37], v[32:33], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[36:37]
	s_cbranch_vccnz .LBB0_368
	global_load_dword v32, v[32:33], off offset:24
	v_mov_b32_e32 v33, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v32
	s_and_b32 m0, s10, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[36:37], v[32:33], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_368:
	s_or_b64 exec, exec, s[8:9]
	s_branch .LBB0_372
.LBB0_369:
	s_or_b64 exec, exec, s[8:9]
	v_readfirstlane_b32 s8, v32
	s_cmp_eq_u32 s8, 0
	s_cbranch_scc1 .LBB0_371
	s_sleep 1
	s_cbranch_execnz .LBB0_372
	s_branch .LBB0_374
.LBB0_371:
	s_branch .LBB0_374
.LBB0_372:
	v_mov_b32_e32 v32, 1
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_369
	global_load_dword v32, v[40:41], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v32, 1, v32
	s_branch .LBB0_369
.LBB0_374:
	s_and_b64 exec, exec, s[0:1]
	s_cbranch_execz .LBB0_377
	v_mov_b32_e32 v40, 0
	global_load_dwordx2 v[32:33], v40, s[4:5] offset:40
	global_load_dwordx2 v[44:45], v40, s[4:5] offset:24 sc0 sc1
	global_load_dwordx2 v[36:37], v40, s[4:5]
	s_waitcnt vmcnt(2)
	v_readfirstlane_b32 s8, v32
	v_readfirstlane_b32 s9, v33
	s_add_u32 s10, s8, 1
	s_addc_u32 s11, s9, 0
	s_add_u32 s0, s10, s2
	s_addc_u32 s1, s11, s3
	s_cmp_eq_u64 s[0:1], 0
	s_cselect_b32 s1, s11, s1
	s_cselect_b32 s0, s10, s0
	s_and_b64 s[2:3], s[0:1], s[8:9]
	s_mul_i32 s3, s3, 24
	s_mul_hi_u32 s8, s2, 24
	s_mul_i32 s2, s2, 24
	s_add_i32 s3, s8, s3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[32:33], v[36:37], 0, s[2:3]
	v_mov_b32_e32 v42, s0
	global_store_dwordx2 v[32:33], v[44:45], off
	v_mov_b32_e32 v43, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[38:39], v40, v[42:45], s[4:5] offset:24 sc0 sc1
	s_mov_b64 s[2:3], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[38:39], v[44:45]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_377
.LBB0_376:
	s_sleep 1
	global_store_dwordx2 v[32:33], v[38:39], off
	v_mov_b32_e32 v36, s0
	v_mov_b32_e32 v37, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v40, v[36:39], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[36:37], v[38:39]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[38:39], v[36:37]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_376
.LBB0_377:
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s0, v46
	v_mov_b64_e32 v[40:41], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v46
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_cbranch_execz .LBB0_383
	v_mov_b32_e32 v36, 0
	global_load_dwordx2 v[42:43], v36, s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[32:33], v36, s[4:5] offset:40
	global_load_dwordx2 v[38:39], v36, s[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v32, v32, v42
	v_and_b32_e32 v33, v33, v43
	v_mul_lo_u32 v33, v33, 24
	v_mul_hi_u32 v37, v32, 24
	v_add_u32_e32 v33, v37, v33
	v_mul_lo_u32 v32, v32, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[32:33], v[38:39], 0, v[32:33]
	global_load_dwordx2 v[40:41], v[32:33], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[40:41], v36, v[40:43], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[40:41], v[42:43]
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_382
	s_mov_b64 s[8:9], 0
.LBB0_380:
	s_sleep 1
	global_load_dwordx2 v[32:33], v36, s[4:5] offset:40
	global_load_dwordx2 v[38:39], v36, s[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v32, v32, v40
	v_and_b32_e32 v37, v33, v41
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[32:33], s[10:11], v32, 24, v[38:39]
	v_mov_b32_e32 v38, v33
	v_mad_u64_u32 v[38:39], s[10:11], v37, 24, v[38:39]
	v_mov_b32_e32 v33, v38
	global_load_dwordx2 v[38:39], v[32:33], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v36, v[38:41], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[32:33], v[40:41]
	s_or_b64 s[8:9], vcc, s[8:9]
	v_mov_b64_e32 v[40:41], v[32:33]
	s_andn2_b64 exec, exec, s[8:9]
	s_cbranch_execnz .LBB0_380
	s_or_b64 exec, exec, s[8:9]
	v_mov_b64_e32 v[40:41], v[32:33]
.LBB0_382:
	s_or_b64 exec, exec, s[6:7]
.LBB0_383:
	s_or_b64 exec, exec, s[2:3]
	v_mov_b32_e32 v49, 0
	global_load_dwordx2 v[32:33], v49, s[4:5] offset:40
	global_load_dwordx4 v[36:39], v49, s[4:5]
	v_readfirstlane_b32 s3, v41
	v_readfirstlane_b32 s2, v40
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s8, v32
	v_readfirstlane_b32 s9, v33
	s_and_b64 s[8:9], s[8:9], s[2:3]
	s_mul_i32 s10, s9, 24
	s_mul_hi_u32 s11, s8, 24
	s_add_i32 s11, s11, s10
	s_mul_i32 s10, s8, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[32:33], v[36:37], 0, s[10:11]
	s_and_saveexec_b64 s[10:11], s[0:1]
	s_cbranch_execz .LBB0_385
	v_mov_b32_e32 v40, s6
	v_mov_b32_e32 v41, s7
	v_mov_b32_e32 v42, 2
	v_mov_b32_e32 v43, 1
	global_store_dwordx4 v[32:33], v[40:43], off offset:8
.LBB0_385:
	s_or_b64 exec, exec, s[10:11]
	s_lshl_b64 s[6:7], s[8:9], 12
	v_lshl_add_u64 v[40:41], v[38:39], 0, s[6:7]
	s_mov_b32 s8, 0
	v_mov_b32_e32 v42, 33
	v_mov_b32_e32 v43, v49
	v_mov_b32_e32 v44, v49
	v_mov_b32_e32 v45, v49
	v_readfirstlane_b32 s6, v40
	v_readfirstlane_b32 s7, v41
	s_mov_b32 s10, s8
	s_mov_b32 s11, s8
	s_mov_b32 s9, s8
	s_nop 1
	global_store_dwordx4 v48, v[42:45], s[6:7]
	s_nop 1
	v_mov_b64_e32 v[44:45], s[10:11]
	v_mov_b64_e32 v[42:43], s[8:9]
	global_store_dwordx4 v48, v[42:45], s[6:7] offset:16
	global_store_dwordx4 v48, v[42:45], s[6:7] offset:32
	global_store_dwordx4 v48, v[42:45], s[6:7] offset:48
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_393
	v_mov_b32_e32 v44, 0
	global_load_dwordx2 v[56:57], v44, s[4:5] offset:32 sc0 sc1
	global_load_dwordx2 v[38:39], v44, s[4:5] offset:40
	v_mov_b32_e32 v54, s2
	v_mov_b32_e32 v55, s3
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s8, v38
	v_readfirstlane_b32 s9, v39
	s_and_b64 s[8:9], s[8:9], s[2:3]
	s_mul_i32 s9, s9, 24
	s_mul_hi_u32 s10, s8, 24
	s_mul_i32 s8, s8, 24
	s_add_i32 s9, s10, s9
	v_lshl_add_u64 v[42:43], v[36:37], 0, s[8:9]
	global_store_dwordx2 v[42:43], v[56:57], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[38:39], v44, v[54:57], s[4:5] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[38:39], v[56:57]
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_389
	s_mov_b64 s[10:11], 0
.LBB0_388:
	s_sleep 1
	global_store_dwordx2 v[42:43], v[38:39], off
	v_mov_b32_e32 v36, s2
	v_mov_b32_e32 v37, s3
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v44, v[36:39], s[4:5] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[36:37], v[38:39]
	s_or_b64 s[10:11], vcc, s[10:11]
	v_mov_b64_e32 v[38:39], v[36:37]
	s_andn2_b64 exec, exec, s[10:11]
	s_cbranch_execnz .LBB0_388
.LBB0_389:
	s_or_b64 exec, exec, s[8:9]
	v_mov_b32_e32 v39, 0
	global_load_dwordx2 v[36:37], v39, s[4:5] offset:16
	s_mov_b64 s[8:9], exec
	v_mbcnt_lo_u32_b32 v38, s8, 0
	v_mbcnt_hi_u32_b32 v38, s9, v38
	v_cmp_eq_u32_e32 vcc, 0, v38
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_391
	s_bcnt1_i32_b64 s8, s[8:9]
	v_mov_b32_e32 v38, s8
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[36:37], v[38:39], off offset:8 sc1
.LBB0_391:
	s_or_b64 exec, exec, s[10:11]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[38:39], v[36:37], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[38:39]
	s_cbranch_vccnz .LBB0_393
	global_load_dword v36, v[36:37], off offset:24
	v_mov_b32_e32 v37, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s8, v36
	s_and_b32 m0, s8, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[38:39], v[36:37], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_393:
	s_or_b64 exec, exec, s[6:7]
	v_lshl_add_u64 v[36:37], v[40:41], 0, v[48:49]
	s_branch .LBB0_397
.LBB0_394:
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s6, v38
	s_cmp_eq_u32 s6, 0
	s_cbranch_scc1 .LBB0_396
	s_sleep 1
	s_cbranch_execnz .LBB0_397
	s_branch .LBB0_399
.LBB0_396:
	s_branch .LBB0_399
.LBB0_397:
	v_mov_b32_e32 v38, 1
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_394
	global_load_dword v38, v[32:33], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v38, 1, v38
	s_branch .LBB0_394
.LBB0_399:
	global_load_dwordx2 v[36:37], v[36:37], off
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_402
	v_mov_b32_e32 v42, 0
	global_load_dwordx2 v[32:33], v42, s[4:5] offset:40
	global_load_dwordx2 v[56:57], v42, s[4:5] offset:24 sc0 sc1
	global_load_dwordx2 v[38:39], v42, s[4:5]
	s_waitcnt vmcnt(2)
	v_readfirstlane_b32 s8, v32
	v_readfirstlane_b32 s9, v33
	s_add_u32 s10, s8, 1
	s_addc_u32 s11, s9, 0
	s_add_u32 s0, s10, s2
	s_addc_u32 s1, s11, s3
	s_cmp_eq_u64 s[0:1], 0
	s_cselect_b32 s1, s11, s1
	s_cselect_b32 s0, s10, s0
	s_and_b64 s[2:3], s[0:1], s[8:9]
	s_mul_i32 s3, s3, 24
	s_mul_hi_u32 s8, s2, 24
	s_mul_i32 s2, s2, 24
	s_add_i32 s3, s8, s3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[32:33], v[38:39], 0, s[2:3]
	v_mov_b32_e32 v54, s0
	global_store_dwordx2 v[32:33], v[56:57], off
	v_mov_b32_e32 v55, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[40:41], v42, v[54:57], s[4:5] offset:24 sc0 sc1
	s_mov_b64 s[2:3], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[40:41], v[56:57]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_402
.LBB0_401:
	s_sleep 1
	global_store_dwordx2 v[32:33], v[40:41], off
	v_mov_b32_e32 v38, s0
	v_mov_b32_e32 v39, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[38:39], v42, v[38:41], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[38:39], v[40:41]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[40:41], v[38:39]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_401
.LBB0_402:
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s0, v46
	v_mov_b64_e32 v[44:45], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v46
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_cbranch_execz .LBB0_408
	v_mov_b32_e32 v38, 0
	global_load_dwordx2 v[42:43], v38, s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[32:33], v38, s[4:5] offset:40
	global_load_dwordx2 v[40:41], v38, s[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v32, v32, v42
	v_and_b32_e32 v33, v33, v43
	v_mul_lo_u32 v33, v33, 24
	v_mul_hi_u32 v39, v32, 24
	v_add_u32_e32 v33, v39, v33
	v_mul_lo_u32 v32, v32, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[32:33], v[40:41], 0, v[32:33]
	global_load_dwordx2 v[40:41], v[32:33], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[44:45], v38, v[40:43], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[44:45], v[42:43]
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_407
	s_mov_b64 s[8:9], 0
.LBB0_405:
	s_sleep 1
	global_load_dwordx2 v[32:33], v38, s[4:5] offset:40
	global_load_dwordx2 v[40:41], v38, s[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v32, v32, v44
	v_and_b32_e32 v39, v33, v45
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[32:33], s[10:11], v32, 24, v[40:41]
	v_mov_b32_e32 v40, v33
	v_mad_u64_u32 v[40:41], s[10:11], v39, 24, v[40:41]
	v_mov_b32_e32 v33, v40
	global_load_dwordx2 v[42:43], v[32:33], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v38, v[42:45], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[32:33], v[44:45]
	s_or_b64 s[8:9], vcc, s[8:9]
	v_mov_b64_e32 v[44:45], v[32:33]
	s_andn2_b64 exec, exec, s[8:9]
	s_cbranch_execnz .LBB0_405
	s_or_b64 exec, exec, s[8:9]
	v_mov_b64_e32 v[44:45], v[32:33]
.LBB0_407:
	s_or_b64 exec, exec, s[6:7]
.LBB0_408:
	s_or_b64 exec, exec, s[2:3]
	v_mov_b32_e32 v49, 0
	global_load_dwordx2 v[32:33], v49, s[4:5] offset:40
	global_load_dwordx4 v[40:43], v49, s[4:5]
	v_readfirstlane_b32 s3, v45
	v_readfirstlane_b32 s2, v44
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s8, v32
	v_readfirstlane_b32 s9, v33
	s_and_b64 s[8:9], s[8:9], s[2:3]
	s_mul_i32 s10, s9, 24
	s_mul_hi_u32 s11, s8, 24
	s_add_i32 s11, s11, s10
	s_mul_i32 s10, s8, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[32:33], v[40:41], 0, s[10:11]
	s_and_saveexec_b64 s[10:11], s[0:1]
	s_cbranch_execz .LBB0_410
	v_mov_b32_e32 v54, s6
	v_mov_b32_e32 v55, s7
	v_mov_b32_e32 v56, 2
	v_mov_b32_e32 v57, 1
	global_store_dwordx4 v[32:33], v[54:57], off offset:8
.LBB0_410:
	s_or_b64 exec, exec, s[10:11]
	s_lshl_b64 s[6:7], s[8:9], 12
	v_lshl_add_u64 v[42:43], v[42:43], 0, s[6:7]
	v_or_b32_e32 v36, 0xe0, v36
	v_mov_b32_e32 v39, 0x6425203d
	v_mov_b32_e32 v38, 0x20444954
	v_readfirstlane_b32 s6, v42
	v_readfirstlane_b32 s7, v43
	s_nop 4
	global_store_dwordx4 v48, v[36:39], s[6:7]
	s_nop 1
	v_mov_b32_e32 v36, 0x4c46202c
	v_mov_b32_e32 v37, 0x3d204741
	v_mov_b32_e32 v38, 0x2c642520
	v_mov_b32_e32 v39, 0x6e6f6320
	global_store_dwordx4 v48, v[36:39], s[6:7] offset:16
	s_nop 1
	v_mov_b32_e32 v36, 0x69746964
	v_mov_b32_e32 v37, 0x3d206e6f
	v_mov_b32_e32 v38, 0x746c7320
	v_mov_b32_e32 v39, 0x7369202c
	global_store_dwordx4 v48, v[36:39], s[6:7] offset:32
	s_nop 1
	v_mov_b32_e32 v36, 0x736f705f
	v_mov_b32_e32 v37, 0x76697469
	v_mov_b32_e32 v38, 0x203d2065
	v_mov_b32_e32 v39, 0xa6425
	global_store_dwordx4 v48, v[36:39], s[6:7] offset:48
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_418
	v_mov_b32_e32 v44, 0
	global_load_dwordx2 v[56:57], v44, s[4:5] offset:32 sc0 sc1
	global_load_dwordx2 v[36:37], v44, s[4:5] offset:40
	v_mov_b32_e32 v54, s2
	v_mov_b32_e32 v55, s3
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s8, v36
	v_readfirstlane_b32 s9, v37
	s_and_b64 s[8:9], s[8:9], s[2:3]
	s_mul_i32 s9, s9, 24
	s_mul_hi_u32 s10, s8, 24
	s_mul_i32 s8, s8, 24
	s_add_i32 s9, s10, s9
	v_lshl_add_u64 v[40:41], v[40:41], 0, s[8:9]
	global_store_dwordx2 v[40:41], v[56:57], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[38:39], v44, v[54:57], s[4:5] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[38:39], v[56:57]
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_414
	s_mov_b64 s[10:11], 0
.LBB0_413:
	s_sleep 1
	global_store_dwordx2 v[40:41], v[38:39], off
	v_mov_b32_e32 v36, s2
	v_mov_b32_e32 v37, s3
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v44, v[36:39], s[4:5] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[36:37], v[38:39]
	s_or_b64 s[10:11], vcc, s[10:11]
	v_mov_b64_e32 v[38:39], v[36:37]
	s_andn2_b64 exec, exec, s[10:11]
	s_cbranch_execnz .LBB0_413
.LBB0_414:
	s_or_b64 exec, exec, s[8:9]
	v_mov_b32_e32 v39, 0
	global_load_dwordx2 v[36:37], v39, s[4:5] offset:16
	s_mov_b64 s[8:9], exec
	v_mbcnt_lo_u32_b32 v38, s8, 0
	v_mbcnt_hi_u32_b32 v38, s9, v38
	v_cmp_eq_u32_e32 vcc, 0, v38
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_416
	s_bcnt1_i32_b64 s8, s[8:9]
	v_mov_b32_e32 v38, s8
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[36:37], v[38:39], off offset:8 sc1
.LBB0_416:
	s_or_b64 exec, exec, s[10:11]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[38:39], v[36:37], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[38:39]
	s_cbranch_vccnz .LBB0_418
	global_load_dword v36, v[36:37], off offset:24
	v_mov_b32_e32 v37, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s8, v36
	s_and_b32 m0, s8, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[38:39], v[36:37], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_418:
	s_or_b64 exec, exec, s[6:7]
	v_lshl_add_u64 v[36:37], v[42:43], 0, v[48:49]
	s_branch .LBB0_422
.LBB0_419:
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s6, v38
	s_cmp_eq_u32 s6, 0
	s_cbranch_scc1 .LBB0_421
	s_sleep 1
	s_cbranch_execnz .LBB0_422
	s_branch .LBB0_424
.LBB0_421:
	s_branch .LBB0_424
.LBB0_422:
	v_mov_b32_e32 v38, 1
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_419
	global_load_dword v38, v[32:33], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v38, 1, v38
	s_branch .LBB0_419
.LBB0_424:
	global_load_dwordx2 v[32:33], v[36:37], off
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_427
	v_mov_b32_e32 v42, 0
	global_load_dwordx2 v[36:37], v42, s[4:5] offset:40
	global_load_dwordx2 v[56:57], v42, s[4:5] offset:24 sc0 sc1
	global_load_dwordx2 v[38:39], v42, s[4:5]
	s_waitcnt vmcnt(2)
	v_readfirstlane_b32 s8, v36
	v_readfirstlane_b32 s9, v37
	s_add_u32 s10, s8, 1
	s_addc_u32 s11, s9, 0
	s_add_u32 s0, s10, s2
	s_addc_u32 s1, s11, s3
	s_cmp_eq_u64 s[0:1], 0
	s_cselect_b32 s1, s11, s1
	s_cselect_b32 s0, s10, s0
	s_and_b64 s[2:3], s[0:1], s[8:9]
	s_mul_i32 s3, s3, 24
	s_mul_hi_u32 s8, s2, 24
	s_mul_i32 s2, s2, 24
	s_add_i32 s3, s8, s3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[40:41], v[38:39], 0, s[2:3]
	v_mov_b32_e32 v54, s0
	global_store_dwordx2 v[40:41], v[56:57], off
	v_mov_b32_e32 v55, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[38:39], v42, v[54:57], s[4:5] offset:24 sc0 sc1
	s_mov_b64 s[2:3], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[38:39], v[56:57]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_427
.LBB0_426:
	s_sleep 1
	global_store_dwordx2 v[40:41], v[38:39], off
	v_mov_b32_e32 v36, s0
	v_mov_b32_e32 v37, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v42, v[36:39], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[36:37], v[38:39]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[38:39], v[36:37]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_426
.LBB0_427:
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s0, v46
	v_mov_b64_e32 v[40:41], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v46
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_cbranch_execz .LBB0_433
	v_mov_b32_e32 v42, 0
	global_load_dwordx2 v[38:39], v42, s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[36:37], v42, s[4:5] offset:40
	global_load_dwordx2 v[40:41], v42, s[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v36, v36, v38
	v_and_b32_e32 v37, v37, v39
	v_mul_lo_u32 v37, v37, 24
	v_mul_hi_u32 v43, v36, 24
	v_add_u32_e32 v37, v43, v37
	v_mul_lo_u32 v36, v36, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[36:37], v[40:41], 0, v[36:37]
	global_load_dwordx2 v[36:37], v[36:37], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[40:41], v42, v[36:39], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[40:41], v[38:39]
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_432
	s_mov_b64 s[8:9], 0
.LBB0_430:
	s_sleep 1
	global_load_dwordx2 v[36:37], v42, s[4:5] offset:40
	global_load_dwordx2 v[38:39], v42, s[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v36, v36, v40
	v_and_b32_e32 v43, v37, v41
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[36:37], s[10:11], v36, 24, v[38:39]
	v_mov_b32_e32 v38, v37
	v_mad_u64_u32 v[38:39], s[10:11], v43, 24, v[38:39]
	v_mov_b32_e32 v37, v38
	global_load_dwordx2 v[38:39], v[36:37], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v42, v[38:41], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[36:37], v[40:41]
	s_or_b64 s[8:9], vcc, s[8:9]
	v_mov_b64_e32 v[40:41], v[36:37]
	s_andn2_b64 exec, exec, s[8:9]
	s_cbranch_execnz .LBB0_430
	s_or_b64 exec, exec, s[8:9]
	v_mov_b64_e32 v[40:41], v[36:37]
.LBB0_432:
	s_or_b64 exec, exec, s[6:7]
.LBB0_433:
	s_or_b64 exec, exec, s[2:3]
	v_mov_b32_e32 v44, 0
	global_load_dwordx2 v[42:43], v44, s[4:5] offset:40
	global_load_dwordx4 v[36:39], v44, s[4:5]
	v_readfirstlane_b32 s7, v41
	v_readfirstlane_b32 s6, v40
	s_mov_b64 s[2:3], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s8, v42
	v_readfirstlane_b32 s9, v43
	s_and_b64 s[8:9], s[8:9], s[6:7]
	s_mul_i32 s10, s9, 24
	s_mul_hi_u32 s11, s8, 24
	s_add_i32 s11, s11, s10
	s_mul_i32 s10, s8, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[40:41], v[36:37], 0, s[10:11]
	s_and_saveexec_b64 s[10:11], s[0:1]
	s_cbranch_execz .LBB0_435
	v_mov_b32_e32 v42, s2
	v_mov_b32_e32 v43, s3
	v_mov_b32_e32 v44, 2
	v_mov_b32_e32 v45, 1
	global_store_dwordx4 v[40:41], v[42:45], off offset:8
.LBB0_435:
	s_or_b64 exec, exec, s[10:11]
	s_lshl_b64 s[2:3], s[8:9], 12
	v_lshl_add_u64 v[38:39], v[38:39], 0, s[2:3]
	v_cmp_gt_u32_e64 s[2:3], 4, v34
	s_mov_b32 s8, 0
	v_mov_b32_e32 v42, 4
	v_cndmask_b32_e64 v44, 0, 1, s[2:3]
	v_mov_b32_e32 v45, s8
	v_readfirstlane_b32 s12, v38
	v_readfirstlane_b32 s13, v39
	v_mov_b32_e32 v43, s8
	s_mov_b32 s10, s8
	s_mov_b32 s11, s8
	v_and_b32_e32 v32, 0xffffff1d, v32
	s_mov_b32 s9, s8
	global_store_dwordx4 v48, v[42:45], s[12:13] offset:16
	v_or_b32_e32 v32, 0x62, v32
	global_store_dwordx4 v48, v[32:35], s[12:13]
	v_mov_b64_e32 v[44:45], s[10:11]
	v_mov_b64_e32 v[42:43], s[8:9]
	global_store_dwordx4 v48, v[42:45], s[12:13] offset:32
	global_store_dwordx4 v48, v[42:45], s[12:13] offset:48
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_443
	v_mov_b32_e32 v42, 0
	global_load_dwordx2 v[56:57], v42, s[4:5] offset:32 sc0 sc1
	global_load_dwordx2 v[32:33], v42, s[4:5] offset:40
	v_mov_b32_e32 v54, s6
	v_mov_b32_e32 v55, s7
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v32
	v_readfirstlane_b32 s11, v33
	s_and_b64 s[10:11], s[10:11], s[6:7]
	s_mul_i32 s11, s11, 24
	s_mul_hi_u32 s12, s10, 24
	s_mul_i32 s10, s10, 24
	s_add_i32 s11, s12, s11
	v_lshl_add_u64 v[32:33], v[36:37], 0, s[10:11]
	global_store_dwordx2 v[32:33], v[56:57], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[38:39], v42, v[54:57], s[4:5] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[38:39], v[56:57]
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_439
	s_mov_b64 s[12:13], 0
.LBB0_438:
	s_sleep 1
	global_store_dwordx2 v[32:33], v[38:39], off
	v_mov_b32_e32 v36, s6
	v_mov_b32_e32 v37, s7
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v42, v[36:39], s[4:5] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[36:37], v[38:39]
	s_or_b64 s[12:13], vcc, s[12:13]
	v_mov_b64_e32 v[38:39], v[36:37]
	s_andn2_b64 exec, exec, s[12:13]
	s_cbranch_execnz .LBB0_438
.LBB0_439:
	s_or_b64 exec, exec, s[10:11]
	v_mov_b32_e32 v37, 0
	global_load_dwordx2 v[32:33], v37, s[4:5] offset:16
	s_mov_b64 s[10:11], exec
	v_mbcnt_lo_u32_b32 v36, s10, 0
	v_mbcnt_hi_u32_b32 v36, s11, v36
	v_cmp_eq_u32_e32 vcc, 0, v36
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB0_441
	s_bcnt1_i32_b64 s10, s[10:11]
	v_mov_b32_e32 v36, s10
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[32:33], v[36:37], off offset:8 sc1
.LBB0_441:
	s_or_b64 exec, exec, s[12:13]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[36:37], v[32:33], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[36:37]
	s_cbranch_vccnz .LBB0_443
	global_load_dword v32, v[32:33], off offset:24
	v_mov_b32_e32 v33, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v32
	s_and_b32 m0, s10, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[36:37], v[32:33], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_443:
	s_or_b64 exec, exec, s[8:9]
	s_branch .LBB0_447
.LBB0_444:
	s_or_b64 exec, exec, s[8:9]
	v_readfirstlane_b32 s8, v32
	s_cmp_eq_u32 s8, 0
	s_cbranch_scc1 .LBB0_446
	s_sleep 1
	s_cbranch_execnz .LBB0_447
	s_branch .LBB0_449
.LBB0_446:
	s_branch .LBB0_449
.LBB0_447:
	v_mov_b32_e32 v32, 1
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_444
	global_load_dword v32, v[40:41], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v32, 1, v32
	s_branch .LBB0_444
.LBB0_449:
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_452
	v_mov_b32_e32 v40, 0
	global_load_dwordx2 v[32:33], v40, s[4:5] offset:40
	global_load_dwordx2 v[44:45], v40, s[4:5] offset:24 sc0 sc1
	global_load_dwordx2 v[36:37], v40, s[4:5]
	s_waitcnt vmcnt(2)
	v_readfirstlane_b32 s10, v32
	v_readfirstlane_b32 s11, v33
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
	v_lshl_add_u64 v[32:33], v[36:37], 0, s[6:7]
	v_mov_b32_e32 v42, s0
	global_store_dwordx2 v[32:33], v[44:45], off
	v_mov_b32_e32 v43, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[38:39], v40, v[42:45], s[4:5] offset:24 sc0 sc1
	s_mov_b64 s[6:7], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[38:39], v[44:45]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_452
.LBB0_451:
	s_sleep 1
	global_store_dwordx2 v[32:33], v[38:39], off
	v_mov_b32_e32 v36, s0
	v_mov_b32_e32 v37, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v40, v[36:39], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[36:37], v[38:39]
	s_or_b64 s[6:7], vcc, s[6:7]
	v_mov_b64_e32 v[38:39], v[36:37]
	s_andn2_b64 exec, exec, s[6:7]
	s_cbranch_execnz .LBB0_451
.LBB0_452:
	s_or_b64 exec, exec, s[8:9]
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_678
	v_readfirstlane_b32 s0, v46
	v_mov_b64_e32 v[40:41], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v46
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_cbranch_execz .LBB0_459
	v_mov_b32_e32 v36, 0
	global_load_dwordx2 v[42:43], v36, s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[32:33], v36, s[4:5] offset:40
	global_load_dwordx2 v[38:39], v36, s[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v32, v32, v42
	v_and_b32_e32 v33, v33, v43
	v_mul_lo_u32 v33, v33, 24
	v_mul_hi_u32 v37, v32, 24
	v_add_u32_e32 v33, v37, v33
	v_mul_lo_u32 v32, v32, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[32:33], v[38:39], 0, v[32:33]
	global_load_dwordx2 v[40:41], v[32:33], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[40:41], v36, v[40:43], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[40:41], v[42:43]
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_458
	s_mov_b64 s[10:11], 0
.LBB0_456:
	s_sleep 1
	global_load_dwordx2 v[32:33], v36, s[4:5] offset:40
	global_load_dwordx2 v[38:39], v36, s[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v32, v32, v40
	v_and_b32_e32 v37, v33, v41
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[32:33], s[12:13], v32, 24, v[38:39]
	v_mov_b32_e32 v38, v33
	v_mad_u64_u32 v[38:39], s[12:13], v37, 24, v[38:39]
	v_mov_b32_e32 v33, v38
	global_load_dwordx2 v[38:39], v[32:33], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v36, v[38:41], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[32:33], v[40:41]
	s_or_b64 s[10:11], vcc, s[10:11]
	v_mov_b64_e32 v[40:41], v[32:33]
	s_andn2_b64 exec, exec, s[10:11]
	s_cbranch_execnz .LBB0_456
	s_or_b64 exec, exec, s[10:11]
	v_mov_b64_e32 v[40:41], v[32:33]
.LBB0_458:
	s_or_b64 exec, exec, s[8:9]
.LBB0_459:
	s_or_b64 exec, exec, s[2:3]
	v_mov_b32_e32 v49, 0
	global_load_dwordx2 v[32:33], v49, s[4:5] offset:40
	global_load_dwordx4 v[36:39], v49, s[4:5]
	v_readfirstlane_b32 s3, v41
	v_readfirstlane_b32 s2, v40
	s_mov_b64 s[8:9], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s10, v32
	v_readfirstlane_b32 s11, v33
	s_and_b64 s[10:11], s[10:11], s[2:3]
	s_mul_i32 s12, s11, 24
	s_mul_hi_u32 s13, s10, 24
	s_add_i32 s13, s13, s12
	s_mul_i32 s12, s10, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[32:33], v[36:37], 0, s[12:13]
	s_and_saveexec_b64 s[12:13], s[0:1]
	s_cbranch_execz .LBB0_461
	v_mov_b32_e32 v40, s8
	v_mov_b32_e32 v41, s9
	v_mov_b32_e32 v42, 2
	v_mov_b32_e32 v43, 1
	global_store_dwordx4 v[32:33], v[40:43], off offset:8
.LBB0_461:
	s_or_b64 exec, exec, s[12:13]
	s_lshl_b64 s[8:9], s[10:11], 12
	v_lshl_add_u64 v[40:41], v[38:39], 0, s[8:9]
	s_mov_b32 s8, 0
	v_mov_b32_e32 v42, 33
	v_mov_b32_e32 v43, v49
	v_mov_b32_e32 v44, v49
	v_mov_b32_e32 v45, v49
	v_readfirstlane_b32 s12, v40
	v_readfirstlane_b32 s13, v41
	s_mov_b32 s10, s8
	s_mov_b32 s11, s8
	s_mov_b32 s9, s8
	s_nop 1
	global_store_dwordx4 v48, v[42:45], s[12:13]
	s_nop 1
	v_mov_b64_e32 v[44:45], s[10:11]
	v_mov_b64_e32 v[42:43], s[8:9]
	global_store_dwordx4 v48, v[42:45], s[12:13] offset:16
	global_store_dwordx4 v48, v[42:45], s[12:13] offset:32
	global_store_dwordx4 v48, v[42:45], s[12:13] offset:48
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_469
	v_mov_b32_e32 v44, 0
	global_load_dwordx2 v[56:57], v44, s[4:5] offset:32 sc0 sc1
	global_load_dwordx2 v[38:39], v44, s[4:5] offset:40
	v_mov_b32_e32 v54, s2
	v_mov_b32_e32 v55, s3
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v38
	v_readfirstlane_b32 s11, v39
	s_and_b64 s[10:11], s[10:11], s[2:3]
	s_mul_i32 s11, s11, 24
	s_mul_hi_u32 s12, s10, 24
	s_mul_i32 s10, s10, 24
	s_add_i32 s11, s12, s11
	v_lshl_add_u64 v[42:43], v[36:37], 0, s[10:11]
	global_store_dwordx2 v[42:43], v[56:57], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[38:39], v44, v[54:57], s[4:5] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[38:39], v[56:57]
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_465
	s_mov_b64 s[12:13], 0
.LBB0_464:
	s_sleep 1
	global_store_dwordx2 v[42:43], v[38:39], off
	v_mov_b32_e32 v36, s2
	v_mov_b32_e32 v37, s3
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v44, v[36:39], s[4:5] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[36:37], v[38:39]
	s_or_b64 s[12:13], vcc, s[12:13]
	v_mov_b64_e32 v[38:39], v[36:37]
	s_andn2_b64 exec, exec, s[12:13]
	s_cbranch_execnz .LBB0_464
.LBB0_465:
	s_or_b64 exec, exec, s[10:11]
	v_mov_b32_e32 v39, 0
	global_load_dwordx2 v[36:37], v39, s[4:5] offset:16
	s_mov_b64 s[10:11], exec
	v_mbcnt_lo_u32_b32 v38, s10, 0
	v_mbcnt_hi_u32_b32 v38, s11, v38
	v_cmp_eq_u32_e32 vcc, 0, v38
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB0_467
	s_bcnt1_i32_b64 s10, s[10:11]
	v_mov_b32_e32 v38, s10
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[36:37], v[38:39], off offset:8 sc1
.LBB0_467:
	s_or_b64 exec, exec, s[12:13]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[38:39], v[36:37], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[38:39]
	s_cbranch_vccnz .LBB0_469
	global_load_dword v36, v[36:37], off offset:24
	v_mov_b32_e32 v37, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v36
	s_and_b32 m0, s10, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[38:39], v[36:37], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_469:
	s_or_b64 exec, exec, s[8:9]
	v_lshl_add_u64 v[36:37], v[40:41], 0, v[48:49]
	s_branch .LBB0_473
.LBB0_470:
	s_or_b64 exec, exec, s[8:9]
	v_readfirstlane_b32 s8, v38
	s_cmp_eq_u32 s8, 0
	s_cbranch_scc1 .LBB0_472
	s_sleep 1
	s_cbranch_execnz .LBB0_473
	s_branch .LBB0_475
.LBB0_472:
	s_branch .LBB0_475
.LBB0_473:
	v_mov_b32_e32 v38, 1
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_470
	global_load_dword v38, v[32:33], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v38, 1, v38
	s_branch .LBB0_470
.LBB0_475:
	global_load_dwordx2 v[36:37], v[36:37], off
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_478
	v_mov_b32_e32 v42, 0
	global_load_dwordx2 v[32:33], v42, s[4:5] offset:40
	global_load_dwordx2 v[56:57], v42, s[4:5] offset:24 sc0 sc1
	global_load_dwordx2 v[38:39], v42, s[4:5]
	s_waitcnt vmcnt(2)
	v_readfirstlane_b32 s10, v32
	v_readfirstlane_b32 s11, v33
	s_add_u32 s12, s10, 1
	s_addc_u32 s13, s11, 0
	s_add_u32 s0, s12, s2
	s_addc_u32 s1, s13, s3
	s_cmp_eq_u64 s[0:1], 0
	s_cselect_b32 s1, s13, s1
	s_cselect_b32 s0, s12, s0
	s_and_b64 s[2:3], s[0:1], s[10:11]
	s_mul_i32 s3, s3, 24
	s_mul_hi_u32 s10, s2, 24
	s_mul_i32 s2, s2, 24
	s_add_i32 s3, s10, s3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[32:33], v[38:39], 0, s[2:3]
	v_mov_b32_e32 v54, s0
	global_store_dwordx2 v[32:33], v[56:57], off
	v_mov_b32_e32 v55, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[40:41], v42, v[54:57], s[4:5] offset:24 sc0 sc1
	s_mov_b64 s[2:3], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[40:41], v[56:57]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_478
.LBB0_477:
	s_sleep 1
	global_store_dwordx2 v[32:33], v[40:41], off
	v_mov_b32_e32 v38, s0
	v_mov_b32_e32 v39, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[38:39], v42, v[38:41], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[38:39], v[40:41]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[40:41], v[38:39]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_477
.LBB0_478:
	s_or_b64 exec, exec, s[8:9]
	v_readfirstlane_b32 s0, v46
	v_mov_b64_e32 v[44:45], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v46
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_cbranch_execz .LBB0_484
	v_mov_b32_e32 v38, 0
	global_load_dwordx2 v[42:43], v38, s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[32:33], v38, s[4:5] offset:40
	global_load_dwordx2 v[40:41], v38, s[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v32, v32, v42
	v_and_b32_e32 v33, v33, v43
	v_mul_lo_u32 v33, v33, 24
	v_mul_hi_u32 v39, v32, 24
	v_add_u32_e32 v33, v39, v33
	v_mul_lo_u32 v32, v32, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[32:33], v[40:41], 0, v[32:33]
	global_load_dwordx2 v[40:41], v[32:33], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[44:45], v38, v[40:43], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[44:45], v[42:43]
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_483
	s_mov_b64 s[10:11], 0
.LBB0_481:
	s_sleep 1
	global_load_dwordx2 v[32:33], v38, s[4:5] offset:40
	global_load_dwordx2 v[40:41], v38, s[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v32, v32, v44
	v_and_b32_e32 v39, v33, v45
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[32:33], s[12:13], v32, 24, v[40:41]
	v_mov_b32_e32 v40, v33
	v_mad_u64_u32 v[40:41], s[12:13], v39, 24, v[40:41]
	v_mov_b32_e32 v33, v40
	global_load_dwordx2 v[42:43], v[32:33], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v38, v[42:45], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[32:33], v[44:45]
	s_or_b64 s[10:11], vcc, s[10:11]
	v_mov_b64_e32 v[44:45], v[32:33]
	s_andn2_b64 exec, exec, s[10:11]
	s_cbranch_execnz .LBB0_481
	s_or_b64 exec, exec, s[10:11]
	v_mov_b64_e32 v[44:45], v[32:33]
.LBB0_483:
	s_or_b64 exec, exec, s[8:9]
.LBB0_484:
	s_or_b64 exec, exec, s[2:3]
	v_mov_b32_e32 v49, 0
	global_load_dwordx2 v[32:33], v49, s[4:5] offset:40
	global_load_dwordx4 v[40:43], v49, s[4:5]
	v_readfirstlane_b32 s3, v45
	v_readfirstlane_b32 s2, v44
	s_mov_b64 s[8:9], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s10, v32
	v_readfirstlane_b32 s11, v33
	s_and_b64 s[10:11], s[10:11], s[2:3]
	s_mul_i32 s12, s11, 24
	s_mul_hi_u32 s13, s10, 24
	s_add_i32 s13, s13, s12
	s_mul_i32 s12, s10, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[32:33], v[40:41], 0, s[12:13]
	s_and_saveexec_b64 s[12:13], s[0:1]
	s_cbranch_execz .LBB0_486
	v_mov_b32_e32 v54, s8
	v_mov_b32_e32 v55, s9
	v_mov_b32_e32 v56, 2
	v_mov_b32_e32 v57, 1
	global_store_dwordx4 v[32:33], v[54:57], off offset:8
.LBB0_486:
	s_or_b64 exec, exec, s[12:13]
	s_lshl_b64 s[8:9], s[10:11], 12
	v_lshl_add_u64 v[42:43], v[42:43], 0, s[8:9]
	v_and_b32_e32 v36, 0xffffff1f, v36
	v_or_b32_e32 v36, 0xa0, v36
	v_mov_b32_e32 v39, 0x7270205d
	v_mov_b32_e32 v38, 0x64255b41
	v_readfirstlane_b32 s8, v42
	v_readfirstlane_b32 s9, v43
	s_nop 4
	global_store_dwordx4 v48, v[36:39], s[8:9]
	s_nop 1
	v_mov_b32_e32 v36, 0x65746e69
	v_mov_b32_e32 v37, 0x6e692064
	v_mov_b32_e32 v38, 0x65646973
	v_mov_b32_e32 v39, 0x72656b20
	global_store_dwordx4 v48, v[36:39], s[8:9] offset:16
	s_nop 1
	v_mov_b32_e32 v36, 0x206c656e
	v_mov_b32_e32 v37, 0x3425203d
	v_mov_b32_e32 v38, 0xa66332e
	v_mov_b32_e32 v39, v49
	global_store_dwordx4 v48, v[36:39], s[8:9] offset:32
	s_nop 1
	v_mov_b32_e32 v36, v49
	v_mov_b32_e32 v37, v49
	v_mov_b32_e32 v38, v49
	global_store_dwordx4 v48, v[36:39], s[8:9] offset:48
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_494
	v_mov_b32_e32 v44, 0
	global_load_dwordx2 v[56:57], v44, s[4:5] offset:32 sc0 sc1
	global_load_dwordx2 v[36:37], v44, s[4:5] offset:40
	v_mov_b32_e32 v54, s2
	v_mov_b32_e32 v55, s3
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v36
	v_readfirstlane_b32 s11, v37
	s_and_b64 s[10:11], s[10:11], s[2:3]
	s_mul_i32 s11, s11, 24
	s_mul_hi_u32 s12, s10, 24
	s_mul_i32 s10, s10, 24
	s_add_i32 s11, s12, s11
	v_lshl_add_u64 v[40:41], v[40:41], 0, s[10:11]
	global_store_dwordx2 v[40:41], v[56:57], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[38:39], v44, v[54:57], s[4:5] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[38:39], v[56:57]
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_490
	s_mov_b64 s[12:13], 0
.LBB0_489:
	s_sleep 1
	global_store_dwordx2 v[40:41], v[38:39], off
	v_mov_b32_e32 v36, s2
	v_mov_b32_e32 v37, s3
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v44, v[36:39], s[4:5] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[36:37], v[38:39]
	s_or_b64 s[12:13], vcc, s[12:13]
	v_mov_b64_e32 v[38:39], v[36:37]
	s_andn2_b64 exec, exec, s[12:13]
	s_cbranch_execnz .LBB0_489
.LBB0_490:
	s_or_b64 exec, exec, s[10:11]
	v_mov_b32_e32 v39, 0
	global_load_dwordx2 v[36:37], v39, s[4:5] offset:16
	s_mov_b64 s[10:11], exec
	v_mbcnt_lo_u32_b32 v38, s10, 0
	v_mbcnt_hi_u32_b32 v38, s11, v38
	v_cmp_eq_u32_e32 vcc, 0, v38
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB0_492
	s_bcnt1_i32_b64 s10, s[10:11]
	v_mov_b32_e32 v38, s10
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[36:37], v[38:39], off offset:8 sc1
.LBB0_492:
	s_or_b64 exec, exec, s[12:13]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[38:39], v[36:37], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[38:39]
	s_cbranch_vccnz .LBB0_494
	global_load_dword v36, v[36:37], off offset:24
	v_mov_b32_e32 v37, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v36
	s_and_b32 m0, s10, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[38:39], v[36:37], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_494:
	s_or_b64 exec, exec, s[8:9]
	v_lshl_add_u64 v[36:37], v[42:43], 0, v[48:49]
	s_branch .LBB0_498
.LBB0_495:
	s_or_b64 exec, exec, s[8:9]
	v_readfirstlane_b32 s8, v38
	s_cmp_eq_u32 s8, 0
	s_cbranch_scc1 .LBB0_497
	s_sleep 1
	s_cbranch_execnz .LBB0_498
	s_branch .LBB0_500
.LBB0_497:
	s_branch .LBB0_500
.LBB0_498:
	v_mov_b32_e32 v38, 1
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_495
	global_load_dword v38, v[32:33], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v38, 1, v38
	s_branch .LBB0_495
.LBB0_500:
	global_load_dwordx2 v[32:33], v[36:37], off
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_503
	v_mov_b32_e32 v42, 0
	global_load_dwordx2 v[36:37], v42, s[4:5] offset:40
	global_load_dwordx2 v[56:57], v42, s[4:5] offset:24 sc0 sc1
	global_load_dwordx2 v[38:39], v42, s[4:5]
	s_waitcnt vmcnt(2)
	v_readfirstlane_b32 s10, v36
	v_readfirstlane_b32 s11, v37
	s_add_u32 s12, s10, 1
	s_addc_u32 s13, s11, 0
	s_add_u32 s0, s12, s2
	s_addc_u32 s1, s13, s3
	s_cmp_eq_u64 s[0:1], 0
	s_cselect_b32 s1, s13, s1
	s_cselect_b32 s0, s12, s0
	s_and_b64 s[2:3], s[0:1], s[10:11]
	s_mul_i32 s3, s3, 24
	s_mul_hi_u32 s10, s2, 24
	s_mul_i32 s2, s2, 24
	s_add_i32 s3, s10, s3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[40:41], v[38:39], 0, s[2:3]
	v_mov_b32_e32 v54, s0
	global_store_dwordx2 v[40:41], v[56:57], off
	v_mov_b32_e32 v55, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[38:39], v42, v[54:57], s[4:5] offset:24 sc0 sc1
	s_mov_b64 s[2:3], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[38:39], v[56:57]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_503
.LBB0_502:
	s_sleep 1
	global_store_dwordx2 v[40:41], v[38:39], off
	v_mov_b32_e32 v36, s0
	v_mov_b32_e32 v37, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v42, v[36:39], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[36:37], v[38:39]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[38:39], v[36:37]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_502
.LBB0_503:
	s_or_b64 exec, exec, s[8:9]
	v_readfirstlane_b32 s0, v46
	v_mov_b64_e32 v[40:41], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v46
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_cbranch_execz .LBB0_509
	v_mov_b32_e32 v42, 0
	global_load_dwordx2 v[38:39], v42, s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[36:37], v42, s[4:5] offset:40
	global_load_dwordx2 v[40:41], v42, s[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v36, v36, v38
	v_and_b32_e32 v37, v37, v39
	v_mul_lo_u32 v37, v37, 24
	v_mul_hi_u32 v43, v36, 24
	v_add_u32_e32 v37, v43, v37
	v_mul_lo_u32 v36, v36, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[36:37], v[40:41], 0, v[36:37]
	global_load_dwordx2 v[36:37], v[36:37], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[40:41], v42, v[36:39], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[40:41], v[38:39]
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_508
	s_mov_b64 s[10:11], 0
.LBB0_506:
	s_sleep 1
	global_load_dwordx2 v[36:37], v42, s[4:5] offset:40
	global_load_dwordx2 v[38:39], v42, s[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v36, v36, v40
	v_and_b32_e32 v43, v37, v41
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[36:37], s[12:13], v36, 24, v[38:39]
	v_mov_b32_e32 v38, v37
	v_mad_u64_u32 v[38:39], s[12:13], v43, 24, v[38:39]
	v_mov_b32_e32 v37, v38
	global_load_dwordx2 v[38:39], v[36:37], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v42, v[38:41], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[36:37], v[40:41]
	s_or_b64 s[10:11], vcc, s[10:11]
	v_mov_b64_e32 v[40:41], v[36:37]
	s_andn2_b64 exec, exec, s[10:11]
	s_cbranch_execnz .LBB0_506
	s_or_b64 exec, exec, s[10:11]
	v_mov_b64_e32 v[40:41], v[36:37]
.LBB0_508:
	s_or_b64 exec, exec, s[8:9]
.LBB0_509:
	s_or_b64 exec, exec, s[2:3]
	v_mov_b32_e32 v44, 0
	global_load_dwordx2 v[42:43], v44, s[4:5] offset:40
	global_load_dwordx4 v[36:39], v44, s[4:5]
	v_readfirstlane_b32 s3, v41
	v_readfirstlane_b32 s2, v40
	s_mov_b64 s[8:9], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s10, v42
	v_readfirstlane_b32 s11, v43
	s_and_b64 s[10:11], s[10:11], s[2:3]
	s_mul_i32 s12, s11, 24
	s_mul_hi_u32 s13, s10, 24
	s_add_i32 s13, s13, s12
	s_mul_i32 s12, s10, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[40:41], v[36:37], 0, s[12:13]
	s_and_saveexec_b64 s[12:13], s[0:1]
	s_cbranch_execz .LBB0_511
	v_mov_b32_e32 v42, s8
	v_mov_b32_e32 v43, s9
	v_mov_b32_e32 v44, 2
	v_mov_b32_e32 v45, 1
	global_store_dwordx4 v[40:41], v[42:45], off offset:8
.LBB0_511:
	s_or_b64 exec, exec, s[12:13]
	s_lshl_b64 s[8:9], s[10:11], 12
	v_lshl_add_u64 v[38:39], v[38:39], 0, s[8:9]
	s_mov_b32 s8, 0
	v_cvt_f64_f32_e32 v[42:43], v50
	v_readfirstlane_b32 s12, v38
	v_readfirstlane_b32 s13, v39
	v_mov_b32_e32 v44, s8
	v_mov_b32_e32 v45, s8
	s_mov_b32 s10, s8
	s_mov_b32 s11, s8
	v_and_b32_e32 v32, 0xffffff1d, v32
	global_store_dwordx4 v48, v[42:45], s[12:13] offset:16
	s_mov_b32 s9, s8
	v_or_b32_e32 v32, 0x42, v32
	v_mov_b64_e32 v[44:45], s[10:11]
	v_mov_b64_e32 v[42:43], s[8:9]
	global_store_dwordx4 v48, v[32:35], s[12:13]
	global_store_dwordx4 v48, v[42:45], s[12:13] offset:32
	global_store_dwordx4 v48, v[42:45], s[12:13] offset:48
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_519
	v_mov_b32_e32 v42, 0
	global_load_dwordx2 v[56:57], v42, s[4:5] offset:32 sc0 sc1
	global_load_dwordx2 v[32:33], v42, s[4:5] offset:40
	v_mov_b32_e32 v54, s2
	v_mov_b32_e32 v55, s3
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v32
	v_readfirstlane_b32 s11, v33
	s_and_b64 s[10:11], s[10:11], s[2:3]
	s_mul_i32 s11, s11, 24
	s_mul_hi_u32 s12, s10, 24
	s_mul_i32 s10, s10, 24
	s_add_i32 s11, s12, s11
	v_lshl_add_u64 v[32:33], v[36:37], 0, s[10:11]
	global_store_dwordx2 v[32:33], v[56:57], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[38:39], v42, v[54:57], s[4:5] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[38:39], v[56:57]
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_515
	s_mov_b64 s[12:13], 0
.LBB0_514:
	s_sleep 1
	global_store_dwordx2 v[32:33], v[38:39], off
	v_mov_b32_e32 v36, s2
	v_mov_b32_e32 v37, s3
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v42, v[36:39], s[4:5] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[36:37], v[38:39]
	s_or_b64 s[12:13], vcc, s[12:13]
	v_mov_b64_e32 v[38:39], v[36:37]
	s_andn2_b64 exec, exec, s[12:13]
	s_cbranch_execnz .LBB0_514
.LBB0_515:
	s_or_b64 exec, exec, s[10:11]
	v_mov_b32_e32 v37, 0
	global_load_dwordx2 v[32:33], v37, s[4:5] offset:16
	s_mov_b64 s[10:11], exec
	v_mbcnt_lo_u32_b32 v36, s10, 0
	v_mbcnt_hi_u32_b32 v36, s11, v36
	v_cmp_eq_u32_e32 vcc, 0, v36
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB0_517
	s_bcnt1_i32_b64 s10, s[10:11]
	v_mov_b32_e32 v36, s10
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[32:33], v[36:37], off offset:8 sc1
.LBB0_517:
	s_or_b64 exec, exec, s[12:13]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[36:37], v[32:33], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[36:37]
	s_cbranch_vccnz .LBB0_519
	global_load_dword v32, v[32:33], off offset:24
	v_mov_b32_e32 v33, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v32
	s_and_b32 m0, s10, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[36:37], v[32:33], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_519:
	s_or_b64 exec, exec, s[8:9]
	s_branch .LBB0_523
.LBB0_520:
	s_or_b64 exec, exec, s[8:9]
	v_readfirstlane_b32 s8, v32
	s_cmp_eq_u32 s8, 0
	s_cbranch_scc1 .LBB0_522
	s_sleep 1
	s_cbranch_execnz .LBB0_523
	s_branch .LBB0_525
.LBB0_522:
	s_branch .LBB0_525
.LBB0_523:
	v_mov_b32_e32 v32, 1
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_520
	global_load_dword v32, v[40:41], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v32, 1, v32
	s_branch .LBB0_520
.LBB0_525:
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_528
	v_mov_b32_e32 v40, 0
	global_load_dwordx2 v[32:33], v40, s[4:5] offset:40
	global_load_dwordx2 v[44:45], v40, s[4:5] offset:24 sc0 sc1
	global_load_dwordx2 v[36:37], v40, s[4:5]
	s_waitcnt vmcnt(2)
	v_readfirstlane_b32 s10, v32
	v_readfirstlane_b32 s11, v33
	s_add_u32 s12, s10, 1
	s_addc_u32 s13, s11, 0
	s_add_u32 s0, s12, s2
	s_addc_u32 s1, s13, s3
	s_cmp_eq_u64 s[0:1], 0
	s_cselect_b32 s1, s13, s1
	s_cselect_b32 s0, s12, s0
	s_and_b64 s[2:3], s[0:1], s[10:11]
	s_mul_i32 s3, s3, 24
	s_mul_hi_u32 s10, s2, 24
	s_mul_i32 s2, s2, 24
	s_add_i32 s3, s10, s3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[32:33], v[36:37], 0, s[2:3]
	v_mov_b32_e32 v42, s0
	global_store_dwordx2 v[32:33], v[44:45], off
	v_mov_b32_e32 v43, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[38:39], v40, v[42:45], s[4:5] offset:24 sc0 sc1
	s_mov_b64 s[2:3], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[38:39], v[44:45]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_528
.LBB0_527:
	s_sleep 1
	global_store_dwordx2 v[32:33], v[38:39], off
	v_mov_b32_e32 v36, s0
	v_mov_b32_e32 v37, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v40, v[36:39], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[36:37], v[38:39]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[38:39], v[36:37]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_527
.LBB0_528:
	s_or_b64 exec, exec, s[8:9]
	v_readfirstlane_b32 s0, v46
	v_mov_b64_e32 v[40:41], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v46
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_cbranch_execz .LBB0_534
	v_mov_b32_e32 v36, 0
	global_load_dwordx2 v[42:43], v36, s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[32:33], v36, s[4:5] offset:40
	global_load_dwordx2 v[38:39], v36, s[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v32, v32, v42
	v_and_b32_e32 v33, v33, v43
	v_mul_lo_u32 v33, v33, 24
	v_mul_hi_u32 v37, v32, 24
	v_add_u32_e32 v33, v37, v33
	v_mul_lo_u32 v32, v32, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[32:33], v[38:39], 0, v[32:33]
	global_load_dwordx2 v[40:41], v[32:33], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[40:41], v36, v[40:43], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[40:41], v[42:43]
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_533
	s_mov_b64 s[10:11], 0
.LBB0_531:
	s_sleep 1
	global_load_dwordx2 v[32:33], v36, s[4:5] offset:40
	global_load_dwordx2 v[38:39], v36, s[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v32, v32, v40
	v_and_b32_e32 v37, v33, v41
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[32:33], s[12:13], v32, 24, v[38:39]
	v_mov_b32_e32 v38, v33
	v_mad_u64_u32 v[38:39], s[12:13], v37, 24, v[38:39]
	v_mov_b32_e32 v33, v38
	global_load_dwordx2 v[38:39], v[32:33], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v36, v[38:41], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[32:33], v[40:41]
	s_or_b64 s[10:11], vcc, s[10:11]
	v_mov_b64_e32 v[40:41], v[32:33]
	s_andn2_b64 exec, exec, s[10:11]
	s_cbranch_execnz .LBB0_531
	s_or_b64 exec, exec, s[10:11]
	v_mov_b64_e32 v[40:41], v[32:33]
.LBB0_533:
	s_or_b64 exec, exec, s[8:9]
.LBB0_534:
	s_or_b64 exec, exec, s[2:3]
	v_mov_b32_e32 v49, 0
	global_load_dwordx2 v[32:33], v49, s[4:5] offset:40
	global_load_dwordx4 v[36:39], v49, s[4:5]
	v_readfirstlane_b32 s3, v41
	v_readfirstlane_b32 s2, v40
	s_mov_b64 s[8:9], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s10, v32
	v_readfirstlane_b32 s11, v33
	s_and_b64 s[10:11], s[10:11], s[2:3]
	s_mul_i32 s12, s11, 24
	s_mul_hi_u32 s13, s10, 24
	s_add_i32 s13, s13, s12
	s_mul_i32 s12, s10, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[32:33], v[36:37], 0, s[12:13]
	s_and_saveexec_b64 s[12:13], s[0:1]
	s_cbranch_execz .LBB0_536
	v_mov_b32_e32 v40, s8
	v_mov_b32_e32 v41, s9
	v_mov_b32_e32 v42, 2
	v_mov_b32_e32 v43, 1
	global_store_dwordx4 v[32:33], v[40:43], off offset:8
.LBB0_536:
	s_or_b64 exec, exec, s[12:13]
	s_lshl_b64 s[8:9], s[10:11], 12
	v_lshl_add_u64 v[40:41], v[38:39], 0, s[8:9]
	s_mov_b32 s8, 0
	v_mov_b32_e32 v42, 33
	v_mov_b32_e32 v43, v49
	v_mov_b32_e32 v44, v49
	v_mov_b32_e32 v45, v49
	v_readfirstlane_b32 s12, v40
	v_readfirstlane_b32 s13, v41
	s_mov_b32 s10, s8
	s_mov_b32 s11, s8
	s_mov_b32 s9, s8
	s_nop 1
	global_store_dwordx4 v48, v[42:45], s[12:13]
	s_nop 1
	v_mov_b64_e32 v[44:45], s[10:11]
	v_mov_b64_e32 v[42:43], s[8:9]
	global_store_dwordx4 v48, v[42:45], s[12:13] offset:16
	global_store_dwordx4 v48, v[42:45], s[12:13] offset:32
	global_store_dwordx4 v48, v[42:45], s[12:13] offset:48
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_544
	v_mov_b32_e32 v44, 0
	global_load_dwordx2 v[56:57], v44, s[4:5] offset:32 sc0 sc1
	global_load_dwordx2 v[38:39], v44, s[4:5] offset:40
	v_mov_b32_e32 v54, s2
	v_mov_b32_e32 v55, s3
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v38
	v_readfirstlane_b32 s11, v39
	s_and_b64 s[10:11], s[10:11], s[2:3]
	s_mul_i32 s11, s11, 24
	s_mul_hi_u32 s12, s10, 24
	s_mul_i32 s10, s10, 24
	s_add_i32 s11, s12, s11
	v_lshl_add_u64 v[42:43], v[36:37], 0, s[10:11]
	global_store_dwordx2 v[42:43], v[56:57], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[38:39], v44, v[54:57], s[4:5] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[38:39], v[56:57]
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_540
	s_mov_b64 s[12:13], 0
.LBB0_539:
	s_sleep 1
	global_store_dwordx2 v[42:43], v[38:39], off
	v_mov_b32_e32 v36, s2
	v_mov_b32_e32 v37, s3
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v44, v[36:39], s[4:5] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[36:37], v[38:39]
	s_or_b64 s[12:13], vcc, s[12:13]
	v_mov_b64_e32 v[38:39], v[36:37]
	s_andn2_b64 exec, exec, s[12:13]
	s_cbranch_execnz .LBB0_539
.LBB0_540:
	s_or_b64 exec, exec, s[10:11]
	v_mov_b32_e32 v39, 0
	global_load_dwordx2 v[36:37], v39, s[4:5] offset:16
	s_mov_b64 s[10:11], exec
	v_mbcnt_lo_u32_b32 v38, s10, 0
	v_mbcnt_hi_u32_b32 v38, s11, v38
	v_cmp_eq_u32_e32 vcc, 0, v38
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB0_542
	s_bcnt1_i32_b64 s10, s[10:11]
	v_mov_b32_e32 v38, s10
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[36:37], v[38:39], off offset:8 sc1
.LBB0_542:
	s_or_b64 exec, exec, s[12:13]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[38:39], v[36:37], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[38:39]
	s_cbranch_vccnz .LBB0_544
	global_load_dword v36, v[36:37], off offset:24
	v_mov_b32_e32 v37, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v36
	s_and_b32 m0, s10, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[38:39], v[36:37], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_544:
	s_or_b64 exec, exec, s[8:9]
	v_lshl_add_u64 v[36:37], v[40:41], 0, v[48:49]
	s_branch .LBB0_548
.LBB0_545:
	s_or_b64 exec, exec, s[8:9]
	v_readfirstlane_b32 s8, v38
	s_cmp_eq_u32 s8, 0
	s_cbranch_scc1 .LBB0_547
	s_sleep 1
	s_cbranch_execnz .LBB0_548
	s_branch .LBB0_550
.LBB0_547:
	s_branch .LBB0_550
.LBB0_548:
	v_mov_b32_e32 v38, 1
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_545
	global_load_dword v38, v[32:33], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v38, 1, v38
	s_branch .LBB0_545
.LBB0_550:
	global_load_dwordx2 v[36:37], v[36:37], off
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_553
	v_mov_b32_e32 v42, 0
	global_load_dwordx2 v[32:33], v42, s[4:5] offset:40
	global_load_dwordx2 v[56:57], v42, s[4:5] offset:24 sc0 sc1
	global_load_dwordx2 v[38:39], v42, s[4:5]
	s_waitcnt vmcnt(2)
	v_readfirstlane_b32 s10, v32
	v_readfirstlane_b32 s11, v33
	s_add_u32 s12, s10, 1
	s_addc_u32 s13, s11, 0
	s_add_u32 s0, s12, s2
	s_addc_u32 s1, s13, s3
	s_cmp_eq_u64 s[0:1], 0
	s_cselect_b32 s1, s13, s1
	s_cselect_b32 s0, s12, s0
	s_and_b64 s[2:3], s[0:1], s[10:11]
	s_mul_i32 s3, s3, 24
	s_mul_hi_u32 s10, s2, 24
	s_mul_i32 s2, s2, 24
	s_add_i32 s3, s10, s3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[32:33], v[38:39], 0, s[2:3]
	v_mov_b32_e32 v54, s0
	global_store_dwordx2 v[32:33], v[56:57], off
	v_mov_b32_e32 v55, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[40:41], v42, v[54:57], s[4:5] offset:24 sc0 sc1
	s_mov_b64 s[2:3], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[40:41], v[56:57]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_553
.LBB0_552:
	s_sleep 1
	global_store_dwordx2 v[32:33], v[40:41], off
	v_mov_b32_e32 v38, s0
	v_mov_b32_e32 v39, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[38:39], v42, v[38:41], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[38:39], v[40:41]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[40:41], v[38:39]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_552
.LBB0_553:
	s_or_b64 exec, exec, s[8:9]
	v_readfirstlane_b32 s0, v46
	v_mov_b64_e32 v[44:45], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v46
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_cbranch_execz .LBB0_559
	v_mov_b32_e32 v38, 0
	global_load_dwordx2 v[42:43], v38, s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[32:33], v38, s[4:5] offset:40
	global_load_dwordx2 v[40:41], v38, s[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v32, v32, v42
	v_and_b32_e32 v33, v33, v43
	v_mul_lo_u32 v33, v33, 24
	v_mul_hi_u32 v39, v32, 24
	v_add_u32_e32 v33, v39, v33
	v_mul_lo_u32 v32, v32, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[32:33], v[40:41], 0, v[32:33]
	global_load_dwordx2 v[40:41], v[32:33], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[44:45], v38, v[40:43], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[44:45], v[42:43]
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_558
	s_mov_b64 s[10:11], 0
.LBB0_556:
	s_sleep 1
	global_load_dwordx2 v[32:33], v38, s[4:5] offset:40
	global_load_dwordx2 v[40:41], v38, s[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v32, v32, v44
	v_and_b32_e32 v39, v33, v45
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[32:33], s[12:13], v32, 24, v[40:41]
	v_mov_b32_e32 v40, v33
	v_mad_u64_u32 v[40:41], s[12:13], v39, 24, v[40:41]
	v_mov_b32_e32 v33, v40
	global_load_dwordx2 v[42:43], v[32:33], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v38, v[42:45], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[32:33], v[44:45]
	s_or_b64 s[10:11], vcc, s[10:11]
	v_mov_b64_e32 v[44:45], v[32:33]
	s_andn2_b64 exec, exec, s[10:11]
	s_cbranch_execnz .LBB0_556
	s_or_b64 exec, exec, s[10:11]
	v_mov_b64_e32 v[44:45], v[32:33]
.LBB0_558:
	s_or_b64 exec, exec, s[8:9]
.LBB0_559:
	s_or_b64 exec, exec, s[2:3]
	v_mov_b32_e32 v49, 0
	global_load_dwordx2 v[32:33], v49, s[4:5] offset:40
	global_load_dwordx4 v[40:43], v49, s[4:5]
	v_readfirstlane_b32 s3, v45
	v_readfirstlane_b32 s2, v44
	s_mov_b64 s[8:9], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s10, v32
	v_readfirstlane_b32 s11, v33
	s_and_b64 s[10:11], s[10:11], s[2:3]
	s_mul_i32 s12, s11, 24
	s_mul_hi_u32 s13, s10, 24
	s_add_i32 s13, s13, s12
	s_mul_i32 s12, s10, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[32:33], v[40:41], 0, s[12:13]
	s_and_saveexec_b64 s[12:13], s[0:1]
	s_cbranch_execz .LBB0_561
	v_mov_b32_e32 v54, s8
	v_mov_b32_e32 v55, s9
	v_mov_b32_e32 v56, 2
	v_mov_b32_e32 v57, 1
	global_store_dwordx4 v[32:33], v[54:57], off offset:8
.LBB0_561:
	s_or_b64 exec, exec, s[12:13]
	s_lshl_b64 s[8:9], s[10:11], 12
	v_lshl_add_u64 v[42:43], v[42:43], 0, s[8:9]
	v_and_b32_e32 v36, 0xffffff1f, v36
	v_or_b32_e32 v36, 0xa0, v36
	v_mov_b32_e32 v39, 0x7270205d
	v_mov_b32_e32 v38, 0x64255b42
	v_readfirstlane_b32 s8, v42
	v_readfirstlane_b32 s9, v43
	s_nop 4
	global_store_dwordx4 v48, v[36:39], s[8:9]
	s_nop 1
	v_mov_b32_e32 v36, 0x65746e69
	v_mov_b32_e32 v37, 0x6e692064
	v_mov_b32_e32 v38, 0x65646973
	v_mov_b32_e32 v39, 0x72656b20
	global_store_dwordx4 v48, v[36:39], s[8:9] offset:16
	s_nop 1
	v_mov_b32_e32 v36, 0x206c656e
	v_mov_b32_e32 v37, 0x3425203d
	v_mov_b32_e32 v38, 0xa66332e
	v_mov_b32_e32 v39, v49
	global_store_dwordx4 v48, v[36:39], s[8:9] offset:32
	s_nop 1
	v_mov_b32_e32 v36, v49
	v_mov_b32_e32 v37, v49
	v_mov_b32_e32 v38, v49
	global_store_dwordx4 v48, v[36:39], s[8:9] offset:48
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_569
	v_mov_b32_e32 v44, 0
	global_load_dwordx2 v[56:57], v44, s[4:5] offset:32 sc0 sc1
	global_load_dwordx2 v[36:37], v44, s[4:5] offset:40
	v_mov_b32_e32 v54, s2
	v_mov_b32_e32 v55, s3
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v36
	v_readfirstlane_b32 s11, v37
	s_and_b64 s[10:11], s[10:11], s[2:3]
	s_mul_i32 s11, s11, 24
	s_mul_hi_u32 s12, s10, 24
	s_mul_i32 s10, s10, 24
	s_add_i32 s11, s12, s11
	v_lshl_add_u64 v[40:41], v[40:41], 0, s[10:11]
	global_store_dwordx2 v[40:41], v[56:57], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[38:39], v44, v[54:57], s[4:5] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[38:39], v[56:57]
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_565
	s_mov_b64 s[12:13], 0
.LBB0_564:
	s_sleep 1
	global_store_dwordx2 v[40:41], v[38:39], off
	v_mov_b32_e32 v36, s2
	v_mov_b32_e32 v37, s3
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v44, v[36:39], s[4:5] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[36:37], v[38:39]
	s_or_b64 s[12:13], vcc, s[12:13]
	v_mov_b64_e32 v[38:39], v[36:37]
	s_andn2_b64 exec, exec, s[12:13]
	s_cbranch_execnz .LBB0_564
.LBB0_565:
	s_or_b64 exec, exec, s[10:11]
	v_mov_b32_e32 v39, 0
	global_load_dwordx2 v[36:37], v39, s[4:5] offset:16
	s_mov_b64 s[10:11], exec
	v_mbcnt_lo_u32_b32 v38, s10, 0
	v_mbcnt_hi_u32_b32 v38, s11, v38
	v_cmp_eq_u32_e32 vcc, 0, v38
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB0_567
	s_bcnt1_i32_b64 s10, s[10:11]
	v_mov_b32_e32 v38, s10
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[36:37], v[38:39], off offset:8 sc1
.LBB0_567:
	s_or_b64 exec, exec, s[12:13]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[38:39], v[36:37], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[38:39]
	s_cbranch_vccnz .LBB0_569
	global_load_dword v36, v[36:37], off offset:24
	v_mov_b32_e32 v37, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v36
	s_and_b32 m0, s10, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[38:39], v[36:37], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_569:
	s_or_b64 exec, exec, s[8:9]
	v_lshl_add_u64 v[36:37], v[42:43], 0, v[48:49]
	s_branch .LBB0_573
.LBB0_570:
	s_or_b64 exec, exec, s[8:9]
	v_readfirstlane_b32 s8, v38
	s_cmp_eq_u32 s8, 0
	s_cbranch_scc1 .LBB0_572
	s_sleep 1
	s_cbranch_execnz .LBB0_573
	s_branch .LBB0_575
.LBB0_572:
	s_branch .LBB0_575
.LBB0_573:
	v_mov_b32_e32 v38, 1
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_570
	global_load_dword v38, v[32:33], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v38, 1, v38
	s_branch .LBB0_570
.LBB0_575:
	global_load_dwordx2 v[32:33], v[36:37], off
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_578
	v_mov_b32_e32 v42, 0
	global_load_dwordx2 v[36:37], v42, s[4:5] offset:40
	global_load_dwordx2 v[56:57], v42, s[4:5] offset:24 sc0 sc1
	global_load_dwordx2 v[38:39], v42, s[4:5]
	s_waitcnt vmcnt(2)
	v_readfirstlane_b32 s10, v36
	v_readfirstlane_b32 s11, v37
	s_add_u32 s12, s10, 1
	s_addc_u32 s13, s11, 0
	s_add_u32 s0, s12, s2
	s_addc_u32 s1, s13, s3
	s_cmp_eq_u64 s[0:1], 0
	s_cselect_b32 s1, s13, s1
	s_cselect_b32 s0, s12, s0
	s_and_b64 s[2:3], s[0:1], s[10:11]
	s_mul_i32 s3, s3, 24
	s_mul_hi_u32 s10, s2, 24
	s_mul_i32 s2, s2, 24
	s_add_i32 s3, s10, s3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[40:41], v[38:39], 0, s[2:3]
	v_mov_b32_e32 v54, s0
	global_store_dwordx2 v[40:41], v[56:57], off
	v_mov_b32_e32 v55, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[38:39], v42, v[54:57], s[4:5] offset:24 sc0 sc1
	s_mov_b64 s[2:3], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[38:39], v[56:57]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_578
.LBB0_577:
	s_sleep 1
	global_store_dwordx2 v[40:41], v[38:39], off
	v_mov_b32_e32 v36, s0
	v_mov_b32_e32 v37, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v42, v[36:39], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[36:37], v[38:39]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[38:39], v[36:37]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_577
.LBB0_578:
	s_or_b64 exec, exec, s[8:9]
	v_readfirstlane_b32 s0, v46
	v_mov_b64_e32 v[40:41], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v46
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_cbranch_execz .LBB0_584
	v_mov_b32_e32 v42, 0
	global_load_dwordx2 v[38:39], v42, s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[36:37], v42, s[4:5] offset:40
	global_load_dwordx2 v[40:41], v42, s[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v36, v36, v38
	v_and_b32_e32 v37, v37, v39
	v_mul_lo_u32 v37, v37, 24
	v_mul_hi_u32 v43, v36, 24
	v_add_u32_e32 v37, v43, v37
	v_mul_lo_u32 v36, v36, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[36:37], v[40:41], 0, v[36:37]
	global_load_dwordx2 v[36:37], v[36:37], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[40:41], v42, v[36:39], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[40:41], v[38:39]
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_583
	s_mov_b64 s[10:11], 0
.LBB0_581:
	s_sleep 1
	global_load_dwordx2 v[36:37], v42, s[4:5] offset:40
	global_load_dwordx2 v[38:39], v42, s[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v36, v36, v40
	v_and_b32_e32 v43, v37, v41
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[36:37], s[12:13], v36, 24, v[38:39]
	v_mov_b32_e32 v38, v37
	v_mad_u64_u32 v[38:39], s[12:13], v43, 24, v[38:39]
	v_mov_b32_e32 v37, v38
	global_load_dwordx2 v[38:39], v[36:37], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v42, v[38:41], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[36:37], v[40:41]
	s_or_b64 s[10:11], vcc, s[10:11]
	v_mov_b64_e32 v[40:41], v[36:37]
	s_andn2_b64 exec, exec, s[10:11]
	s_cbranch_execnz .LBB0_581
	s_or_b64 exec, exec, s[10:11]
	v_mov_b64_e32 v[40:41], v[36:37]
.LBB0_583:
	s_or_b64 exec, exec, s[8:9]
.LBB0_584:
	s_or_b64 exec, exec, s[2:3]
	v_mov_b32_e32 v44, 0
	global_load_dwordx2 v[42:43], v44, s[4:5] offset:40
	global_load_dwordx4 v[36:39], v44, s[4:5]
	v_readfirstlane_b32 s3, v41
	v_readfirstlane_b32 s2, v40
	s_mov_b64 s[8:9], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s10, v42
	v_readfirstlane_b32 s11, v43
	s_and_b64 s[10:11], s[10:11], s[2:3]
	s_mul_i32 s12, s11, 24
	s_mul_hi_u32 s13, s10, 24
	s_add_i32 s13, s13, s12
	s_mul_i32 s12, s10, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[40:41], v[36:37], 0, s[12:13]
	s_and_saveexec_b64 s[12:13], s[0:1]
	s_cbranch_execz .LBB0_586
	v_mov_b32_e32 v42, s8
	v_mov_b32_e32 v43, s9
	v_mov_b32_e32 v44, 2
	v_mov_b32_e32 v45, 1
	global_store_dwordx4 v[40:41], v[42:45], off offset:8
.LBB0_586:
	s_or_b64 exec, exec, s[12:13]
	s_lshl_b64 s[8:9], s[10:11], 12
	v_lshl_add_u64 v[38:39], v[38:39], 0, s[8:9]
	s_mov_b32 s8, 0
	v_cvt_f64_f32_e32 v[42:43], v51
	v_readfirstlane_b32 s12, v38
	v_readfirstlane_b32 s13, v39
	v_mov_b32_e32 v44, s8
	v_mov_b32_e32 v45, s8
	s_mov_b32 s10, s8
	s_mov_b32 s11, s8
	v_and_b32_e32 v32, 0xffffff1d, v32
	global_store_dwordx4 v48, v[42:45], s[12:13] offset:16
	s_mov_b32 s9, s8
	v_or_b32_e32 v32, 0x42, v32
	v_mov_b64_e32 v[44:45], s[10:11]
	v_mov_b64_e32 v[42:43], s[8:9]
	global_store_dwordx4 v48, v[32:35], s[12:13]
	global_store_dwordx4 v48, v[42:45], s[12:13] offset:32
	global_store_dwordx4 v48, v[42:45], s[12:13] offset:48
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_594
	v_mov_b32_e32 v42, 0
	global_load_dwordx2 v[56:57], v42, s[4:5] offset:32 sc0 sc1
	global_load_dwordx2 v[32:33], v42, s[4:5] offset:40
	v_mov_b32_e32 v54, s2
	v_mov_b32_e32 v55, s3
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v32
	v_readfirstlane_b32 s11, v33
	s_and_b64 s[10:11], s[10:11], s[2:3]
	s_mul_i32 s11, s11, 24
	s_mul_hi_u32 s12, s10, 24
	s_mul_i32 s10, s10, 24
	s_add_i32 s11, s12, s11
	v_lshl_add_u64 v[32:33], v[36:37], 0, s[10:11]
	global_store_dwordx2 v[32:33], v[56:57], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[38:39], v42, v[54:57], s[4:5] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[38:39], v[56:57]
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_590
	s_mov_b64 s[12:13], 0
.LBB0_589:
	s_sleep 1
	global_store_dwordx2 v[32:33], v[38:39], off
	v_mov_b32_e32 v36, s2
	v_mov_b32_e32 v37, s3
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v42, v[36:39], s[4:5] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[36:37], v[38:39]
	s_or_b64 s[12:13], vcc, s[12:13]
	v_mov_b64_e32 v[38:39], v[36:37]
	s_andn2_b64 exec, exec, s[12:13]
	s_cbranch_execnz .LBB0_589
.LBB0_590:
	s_or_b64 exec, exec, s[10:11]
	v_mov_b32_e32 v37, 0
	global_load_dwordx2 v[32:33], v37, s[4:5] offset:16
	s_mov_b64 s[10:11], exec
	v_mbcnt_lo_u32_b32 v36, s10, 0
	v_mbcnt_hi_u32_b32 v36, s11, v36
	v_cmp_eq_u32_e32 vcc, 0, v36
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB0_592
	s_bcnt1_i32_b64 s10, s[10:11]
	v_mov_b32_e32 v36, s10
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[32:33], v[36:37], off offset:8 sc1
.LBB0_592:
	s_or_b64 exec, exec, s[12:13]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[36:37], v[32:33], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[36:37]
	s_cbranch_vccnz .LBB0_594
	global_load_dword v32, v[32:33], off offset:24
	v_mov_b32_e32 v33, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v32
	s_and_b32 m0, s10, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[36:37], v[32:33], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_594:
	s_or_b64 exec, exec, s[8:9]
	s_branch .LBB0_598
.LBB0_595:
	s_or_b64 exec, exec, s[8:9]
	v_readfirstlane_b32 s8, v32
	s_cmp_eq_u32 s8, 0
	s_cbranch_scc1 .LBB0_597
	s_sleep 1
	s_cbranch_execnz .LBB0_598
	s_branch .LBB0_600
.LBB0_597:
	s_branch .LBB0_600
.LBB0_598:
	v_mov_b32_e32 v32, 1
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_595
	global_load_dword v32, v[40:41], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v32, 1, v32
	s_branch .LBB0_595
.LBB0_600:
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_603
	v_mov_b32_e32 v40, 0
	global_load_dwordx2 v[32:33], v40, s[4:5] offset:40
	global_load_dwordx2 v[44:45], v40, s[4:5] offset:24 sc0 sc1
	global_load_dwordx2 v[36:37], v40, s[4:5]
	s_waitcnt vmcnt(2)
	v_readfirstlane_b32 s10, v32
	v_readfirstlane_b32 s11, v33
	s_add_u32 s12, s10, 1
	s_addc_u32 s13, s11, 0
	s_add_u32 s0, s12, s2
	s_addc_u32 s1, s13, s3
	s_cmp_eq_u64 s[0:1], 0
	s_cselect_b32 s1, s13, s1
	s_cselect_b32 s0, s12, s0
	s_and_b64 s[2:3], s[0:1], s[10:11]
	s_mul_i32 s3, s3, 24
	s_mul_hi_u32 s10, s2, 24
	s_mul_i32 s2, s2, 24
	s_add_i32 s3, s10, s3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[32:33], v[36:37], 0, s[2:3]
	v_mov_b32_e32 v42, s0
	global_store_dwordx2 v[32:33], v[44:45], off
	v_mov_b32_e32 v43, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[38:39], v40, v[42:45], s[4:5] offset:24 sc0 sc1
	s_mov_b64 s[2:3], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[38:39], v[44:45]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_603
.LBB0_602:
	s_sleep 1
	global_store_dwordx2 v[32:33], v[38:39], off
	v_mov_b32_e32 v36, s0
	v_mov_b32_e32 v37, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v40, v[36:39], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[36:37], v[38:39]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[38:39], v[36:37]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_602
.LBB0_603:
	s_or_b64 exec, exec, s[8:9]
	v_readfirstlane_b32 s0, v46
	v_mov_b64_e32 v[40:41], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v46
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_cbranch_execz .LBB0_609
	v_mov_b32_e32 v36, 0
	global_load_dwordx2 v[42:43], v36, s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[32:33], v36, s[4:5] offset:40
	global_load_dwordx2 v[38:39], v36, s[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v32, v32, v42
	v_and_b32_e32 v33, v33, v43
	v_mul_lo_u32 v33, v33, 24
	v_mul_hi_u32 v37, v32, 24
	v_add_u32_e32 v33, v37, v33
	v_mul_lo_u32 v32, v32, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[32:33], v[38:39], 0, v[32:33]
	global_load_dwordx2 v[40:41], v[32:33], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[40:41], v36, v[40:43], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[40:41], v[42:43]
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_608
	s_mov_b64 s[10:11], 0
.LBB0_606:
	s_sleep 1
	global_load_dwordx2 v[32:33], v36, s[4:5] offset:40
	global_load_dwordx2 v[38:39], v36, s[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v32, v32, v40
	v_and_b32_e32 v37, v33, v41
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[32:33], s[12:13], v32, 24, v[38:39]
	v_mov_b32_e32 v38, v33
	v_mad_u64_u32 v[38:39], s[12:13], v37, 24, v[38:39]
	v_mov_b32_e32 v33, v38
	global_load_dwordx2 v[38:39], v[32:33], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v36, v[38:41], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[32:33], v[40:41]
	s_or_b64 s[10:11], vcc, s[10:11]
	v_mov_b64_e32 v[40:41], v[32:33]
	s_andn2_b64 exec, exec, s[10:11]
	s_cbranch_execnz .LBB0_606
	s_or_b64 exec, exec, s[10:11]
	v_mov_b64_e32 v[40:41], v[32:33]
.LBB0_608:
	s_or_b64 exec, exec, s[8:9]
.LBB0_609:
	s_or_b64 exec, exec, s[2:3]
	v_mov_b32_e32 v49, 0
	global_load_dwordx2 v[32:33], v49, s[4:5] offset:40
	global_load_dwordx4 v[36:39], v49, s[4:5]
	v_readfirstlane_b32 s3, v41
	v_readfirstlane_b32 s2, v40
	s_mov_b64 s[8:9], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s10, v32
	v_readfirstlane_b32 s11, v33
	s_and_b64 s[10:11], s[10:11], s[2:3]
	s_mul_i32 s12, s11, 24
	s_mul_hi_u32 s13, s10, 24
	s_add_i32 s13, s13, s12
	s_mul_i32 s12, s10, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[32:33], v[36:37], 0, s[12:13]
	s_and_saveexec_b64 s[12:13], s[0:1]
	s_cbranch_execz .LBB0_611
	v_mov_b32_e32 v40, s8
	v_mov_b32_e32 v41, s9
	v_mov_b32_e32 v42, 2
	v_mov_b32_e32 v43, 1
	global_store_dwordx4 v[32:33], v[40:43], off offset:8
.LBB0_611:
	s_or_b64 exec, exec, s[12:13]
	s_lshl_b64 s[8:9], s[10:11], 12
	v_lshl_add_u64 v[40:41], v[38:39], 0, s[8:9]
	s_mov_b32 s8, 0
	v_mov_b32_e32 v42, 33
	v_mov_b32_e32 v43, v49
	v_mov_b32_e32 v44, v49
	v_mov_b32_e32 v45, v49
	v_readfirstlane_b32 s12, v40
	v_readfirstlane_b32 s13, v41
	s_mov_b32 s10, s8
	s_mov_b32 s11, s8
	s_mov_b32 s9, s8
	s_nop 1
	global_store_dwordx4 v48, v[42:45], s[12:13]
	s_nop 1
	v_mov_b64_e32 v[44:45], s[10:11]
	v_mov_b64_e32 v[42:43], s[8:9]
	global_store_dwordx4 v48, v[42:45], s[12:13] offset:16
	global_store_dwordx4 v48, v[42:45], s[12:13] offset:32
	global_store_dwordx4 v48, v[42:45], s[12:13] offset:48
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_619
	v_mov_b32_e32 v44, 0
	global_load_dwordx2 v[56:57], v44, s[4:5] offset:32 sc0 sc1
	global_load_dwordx2 v[38:39], v44, s[4:5] offset:40
	v_mov_b32_e32 v54, s2
	v_mov_b32_e32 v55, s3
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v38
	v_readfirstlane_b32 s11, v39
	s_and_b64 s[10:11], s[10:11], s[2:3]
	s_mul_i32 s11, s11, 24
	s_mul_hi_u32 s12, s10, 24
	s_mul_i32 s10, s10, 24
	s_add_i32 s11, s12, s11
	v_lshl_add_u64 v[42:43], v[36:37], 0, s[10:11]
	global_store_dwordx2 v[42:43], v[56:57], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[38:39], v44, v[54:57], s[4:5] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[38:39], v[56:57]
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_615
	s_mov_b64 s[12:13], 0
.LBB0_614:
	s_sleep 1
	global_store_dwordx2 v[42:43], v[38:39], off
	v_mov_b32_e32 v36, s2
	v_mov_b32_e32 v37, s3
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v44, v[36:39], s[4:5] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[36:37], v[38:39]
	s_or_b64 s[12:13], vcc, s[12:13]
	v_mov_b64_e32 v[38:39], v[36:37]
	s_andn2_b64 exec, exec, s[12:13]
	s_cbranch_execnz .LBB0_614
.LBB0_615:
	s_or_b64 exec, exec, s[10:11]
	v_mov_b32_e32 v39, 0
	global_load_dwordx2 v[36:37], v39, s[4:5] offset:16
	s_mov_b64 s[10:11], exec
	v_mbcnt_lo_u32_b32 v38, s10, 0
	v_mbcnt_hi_u32_b32 v38, s11, v38
	v_cmp_eq_u32_e32 vcc, 0, v38
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB0_617
	s_bcnt1_i32_b64 s10, s[10:11]
	v_mov_b32_e32 v38, s10
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[36:37], v[38:39], off offset:8 sc1
.LBB0_617:
	s_or_b64 exec, exec, s[12:13]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[38:39], v[36:37], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[38:39]
	s_cbranch_vccnz .LBB0_619
	global_load_dword v36, v[36:37], off offset:24
	v_mov_b32_e32 v37, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v36
	s_and_b32 m0, s10, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[38:39], v[36:37], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_619:
	s_or_b64 exec, exec, s[8:9]
	v_lshl_add_u64 v[36:37], v[40:41], 0, v[48:49]
	s_branch .LBB0_623
.LBB0_620:
	s_or_b64 exec, exec, s[8:9]
	v_readfirstlane_b32 s8, v38
	s_cmp_eq_u32 s8, 0
	s_cbranch_scc1 .LBB0_622
	s_sleep 1
	s_cbranch_execnz .LBB0_623
	s_branch .LBB0_625
.LBB0_622:
	s_branch .LBB0_625
.LBB0_623:
	v_mov_b32_e32 v38, 1
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_620
	global_load_dword v38, v[32:33], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v38, 1, v38
	s_branch .LBB0_620
.LBB0_625:
	global_load_dwordx2 v[36:37], v[36:37], off
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_628
	v_mov_b32_e32 v42, 0
	global_load_dwordx2 v[32:33], v42, s[4:5] offset:40
	global_load_dwordx2 v[56:57], v42, s[4:5] offset:24 sc0 sc1
	global_load_dwordx2 v[38:39], v42, s[4:5]
	s_waitcnt vmcnt(2)
	v_readfirstlane_b32 s10, v32
	v_readfirstlane_b32 s11, v33
	s_add_u32 s12, s10, 1
	s_addc_u32 s13, s11, 0
	s_add_u32 s0, s12, s2
	s_addc_u32 s1, s13, s3
	s_cmp_eq_u64 s[0:1], 0
	s_cselect_b32 s1, s13, s1
	s_cselect_b32 s0, s12, s0
	s_and_b64 s[2:3], s[0:1], s[10:11]
	s_mul_i32 s3, s3, 24
	s_mul_hi_u32 s10, s2, 24
	s_mul_i32 s2, s2, 24
	s_add_i32 s3, s10, s3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[32:33], v[38:39], 0, s[2:3]
	v_mov_b32_e32 v54, s0
	global_store_dwordx2 v[32:33], v[56:57], off
	v_mov_b32_e32 v55, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[40:41], v42, v[54:57], s[4:5] offset:24 sc0 sc1
	s_mov_b64 s[2:3], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[40:41], v[56:57]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_628
.LBB0_627:
	s_sleep 1
	global_store_dwordx2 v[32:33], v[40:41], off
	v_mov_b32_e32 v38, s0
	v_mov_b32_e32 v39, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[38:39], v42, v[38:41], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[38:39], v[40:41]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[40:41], v[38:39]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_627
.LBB0_628:
	s_or_b64 exec, exec, s[8:9]
	v_readfirstlane_b32 s0, v46
	v_mov_b64_e32 v[44:45], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v46
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_cbranch_execz .LBB0_634
	v_mov_b32_e32 v38, 0
	global_load_dwordx2 v[42:43], v38, s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[32:33], v38, s[4:5] offset:40
	global_load_dwordx2 v[40:41], v38, s[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v32, v32, v42
	v_and_b32_e32 v33, v33, v43
	v_mul_lo_u32 v33, v33, 24
	v_mul_hi_u32 v39, v32, 24
	v_add_u32_e32 v33, v39, v33
	v_mul_lo_u32 v32, v32, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[32:33], v[40:41], 0, v[32:33]
	global_load_dwordx2 v[40:41], v[32:33], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[44:45], v38, v[40:43], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[44:45], v[42:43]
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_633
	s_mov_b64 s[10:11], 0
.LBB0_631:
	s_sleep 1
	global_load_dwordx2 v[32:33], v38, s[4:5] offset:40
	global_load_dwordx2 v[40:41], v38, s[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v32, v32, v44
	v_and_b32_e32 v39, v33, v45
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[32:33], s[12:13], v32, 24, v[40:41]
	v_mov_b32_e32 v40, v33
	v_mad_u64_u32 v[40:41], s[12:13], v39, 24, v[40:41]
	v_mov_b32_e32 v33, v40
	global_load_dwordx2 v[42:43], v[32:33], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v38, v[42:45], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[32:33], v[44:45]
	s_or_b64 s[10:11], vcc, s[10:11]
	v_mov_b64_e32 v[44:45], v[32:33]
	s_andn2_b64 exec, exec, s[10:11]
	s_cbranch_execnz .LBB0_631
	s_or_b64 exec, exec, s[10:11]
	v_mov_b64_e32 v[44:45], v[32:33]
.LBB0_633:
	s_or_b64 exec, exec, s[8:9]
.LBB0_634:
	s_or_b64 exec, exec, s[2:3]
	v_mov_b32_e32 v49, 0
	global_load_dwordx2 v[32:33], v49, s[4:5] offset:40
	global_load_dwordx4 v[40:43], v49, s[4:5]
	v_readfirstlane_b32 s3, v45
	v_readfirstlane_b32 s2, v44
	s_mov_b64 s[8:9], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s10, v32
	v_readfirstlane_b32 s11, v33
	s_and_b64 s[10:11], s[10:11], s[2:3]
	s_mul_i32 s12, s11, 24
	s_mul_hi_u32 s13, s10, 24
	s_add_i32 s13, s13, s12
	s_mul_i32 s12, s10, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[32:33], v[40:41], 0, s[12:13]
	s_and_saveexec_b64 s[12:13], s[0:1]
	s_cbranch_execz .LBB0_636
	v_mov_b32_e32 v54, s8
	v_mov_b32_e32 v55, s9
	v_mov_b32_e32 v56, 2
	v_mov_b32_e32 v57, 1
	global_store_dwordx4 v[32:33], v[54:57], off offset:8
.LBB0_636:
	s_or_b64 exec, exec, s[12:13]
	s_lshl_b64 s[8:9], s[10:11], 12
	v_lshl_add_u64 v[42:43], v[42:43], 0, s[8:9]
	v_and_b32_e32 v36, 0xffffff1f, v36
	v_or_b32_e32 v36, 0xa0, v36
	v_mov_b32_e32 v39, 0x7270205d
	v_mov_b32_e32 v38, 0x64255b43
	v_readfirstlane_b32 s8, v42
	v_readfirstlane_b32 s9, v43
	s_nop 4
	global_store_dwordx4 v48, v[36:39], s[8:9]
	s_nop 1
	v_mov_b32_e32 v36, 0x65746e69
	v_mov_b32_e32 v37, 0x6e692064
	v_mov_b32_e32 v38, 0x65646973
	v_mov_b32_e32 v39, 0x72656b20
	global_store_dwordx4 v48, v[36:39], s[8:9] offset:16
	s_nop 1
	v_mov_b32_e32 v36, 0x206c656e
	v_mov_b32_e32 v37, 0x3425203d
	v_mov_b32_e32 v38, 0xa66332e
	v_mov_b32_e32 v39, v49
	global_store_dwordx4 v48, v[36:39], s[8:9] offset:32
	s_nop 1
	v_mov_b32_e32 v36, v49
	v_mov_b32_e32 v37, v49
	v_mov_b32_e32 v38, v49
	global_store_dwordx4 v48, v[36:39], s[8:9] offset:48
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_644
	v_mov_b32_e32 v44, 0
	global_load_dwordx2 v[56:57], v44, s[4:5] offset:32 sc0 sc1
	global_load_dwordx2 v[36:37], v44, s[4:5] offset:40
	v_mov_b32_e32 v54, s2
	v_mov_b32_e32 v55, s3
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v36
	v_readfirstlane_b32 s11, v37
	s_and_b64 s[10:11], s[10:11], s[2:3]
	s_mul_i32 s11, s11, 24
	s_mul_hi_u32 s12, s10, 24
	s_mul_i32 s10, s10, 24
	s_add_i32 s11, s12, s11
	v_lshl_add_u64 v[40:41], v[40:41], 0, s[10:11]
	global_store_dwordx2 v[40:41], v[56:57], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[38:39], v44, v[54:57], s[4:5] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[38:39], v[56:57]
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_640
	s_mov_b64 s[12:13], 0
.LBB0_639:
	s_sleep 1
	global_store_dwordx2 v[40:41], v[38:39], off
	v_mov_b32_e32 v36, s2
	v_mov_b32_e32 v37, s3
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v44, v[36:39], s[4:5] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[36:37], v[38:39]
	s_or_b64 s[12:13], vcc, s[12:13]
	v_mov_b64_e32 v[38:39], v[36:37]
	s_andn2_b64 exec, exec, s[12:13]
	s_cbranch_execnz .LBB0_639
.LBB0_640:
	s_or_b64 exec, exec, s[10:11]
	v_mov_b32_e32 v39, 0
	global_load_dwordx2 v[36:37], v39, s[4:5] offset:16
	s_mov_b64 s[10:11], exec
	v_mbcnt_lo_u32_b32 v38, s10, 0
	v_mbcnt_hi_u32_b32 v38, s11, v38
	v_cmp_eq_u32_e32 vcc, 0, v38
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB0_642
	s_bcnt1_i32_b64 s10, s[10:11]
	v_mov_b32_e32 v38, s10
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[36:37], v[38:39], off offset:8 sc1
.LBB0_642:
	s_or_b64 exec, exec, s[12:13]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[38:39], v[36:37], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[38:39]
	s_cbranch_vccnz .LBB0_644
	global_load_dword v36, v[36:37], off offset:24
	v_mov_b32_e32 v37, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v36
	s_and_b32 m0, s10, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[38:39], v[36:37], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_644:
	s_or_b64 exec, exec, s[8:9]
	v_lshl_add_u64 v[36:37], v[42:43], 0, v[48:49]
	s_branch .LBB0_648
.LBB0_645:
	s_or_b64 exec, exec, s[8:9]
	v_readfirstlane_b32 s8, v38
	s_cmp_eq_u32 s8, 0
	s_cbranch_scc1 .LBB0_647
	s_sleep 1
	s_cbranch_execnz .LBB0_648
	s_branch .LBB0_650
.LBB0_647:
	s_branch .LBB0_650
.LBB0_648:
	v_mov_b32_e32 v38, 1
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_645
	global_load_dword v38, v[32:33], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v38, 1, v38
	s_branch .LBB0_645
.LBB0_650:
	global_load_dwordx2 v[32:33], v[36:37], off
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_653
	v_mov_b32_e32 v42, 0
	global_load_dwordx2 v[36:37], v42, s[4:5] offset:40
	global_load_dwordx2 v[56:57], v42, s[4:5] offset:24 sc0 sc1
	global_load_dwordx2 v[38:39], v42, s[4:5]
	s_waitcnt vmcnt(2)
	v_readfirstlane_b32 s10, v36
	v_readfirstlane_b32 s11, v37
	s_add_u32 s12, s10, 1
	s_addc_u32 s13, s11, 0
	s_add_u32 s0, s12, s2
	s_addc_u32 s1, s13, s3
	s_cmp_eq_u64 s[0:1], 0
	s_cselect_b32 s1, s13, s1
	s_cselect_b32 s0, s12, s0
	s_and_b64 s[2:3], s[0:1], s[10:11]
	s_mul_i32 s3, s3, 24
	s_mul_hi_u32 s10, s2, 24
	s_mul_i32 s2, s2, 24
	s_add_i32 s3, s10, s3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[40:41], v[38:39], 0, s[2:3]
	v_mov_b32_e32 v54, s0
	global_store_dwordx2 v[40:41], v[56:57], off
	v_mov_b32_e32 v55, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[38:39], v42, v[54:57], s[4:5] offset:24 sc0 sc1
	s_mov_b64 s[2:3], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[38:39], v[56:57]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_653
.LBB0_652:
	s_sleep 1
	global_store_dwordx2 v[40:41], v[38:39], off
	v_mov_b32_e32 v36, s0
	v_mov_b32_e32 v37, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v42, v[36:39], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[36:37], v[38:39]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[38:39], v[36:37]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_652
.LBB0_653:
	s_or_b64 exec, exec, s[8:9]
	v_readfirstlane_b32 s0, v46
	v_mov_b64_e32 v[40:41], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v46
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_cbranch_execz .LBB0_659
	v_mov_b32_e32 v42, 0
	global_load_dwordx2 v[38:39], v42, s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[36:37], v42, s[4:5] offset:40
	global_load_dwordx2 v[40:41], v42, s[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v36, v36, v38
	v_and_b32_e32 v37, v37, v39
	v_mul_lo_u32 v37, v37, 24
	v_mul_hi_u32 v43, v36, 24
	v_add_u32_e32 v37, v43, v37
	v_mul_lo_u32 v36, v36, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[36:37], v[40:41], 0, v[36:37]
	global_load_dwordx2 v[36:37], v[36:37], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[40:41], v42, v[36:39], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[40:41], v[38:39]
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_658
	s_mov_b64 s[10:11], 0
.LBB0_656:
	s_sleep 1
	global_load_dwordx2 v[36:37], v42, s[4:5] offset:40
	global_load_dwordx2 v[38:39], v42, s[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v36, v36, v40
	v_and_b32_e32 v43, v37, v41
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[36:37], s[12:13], v36, 24, v[38:39]
	v_mov_b32_e32 v38, v37
	v_mad_u64_u32 v[38:39], s[12:13], v43, 24, v[38:39]
	v_mov_b32_e32 v37, v38
	global_load_dwordx2 v[38:39], v[36:37], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v42, v[38:41], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[36:37], v[40:41]
	s_or_b64 s[10:11], vcc, s[10:11]
	v_mov_b64_e32 v[40:41], v[36:37]
	s_andn2_b64 exec, exec, s[10:11]
	s_cbranch_execnz .LBB0_656
	s_or_b64 exec, exec, s[10:11]
	v_mov_b64_e32 v[40:41], v[36:37]
.LBB0_658:
	s_or_b64 exec, exec, s[8:9]
.LBB0_659:
	s_or_b64 exec, exec, s[2:3]
	v_mov_b32_e32 v44, 0
	global_load_dwordx2 v[42:43], v44, s[4:5] offset:40
	global_load_dwordx4 v[36:39], v44, s[4:5]
	v_readfirstlane_b32 s3, v41
	v_readfirstlane_b32 s2, v40
	s_mov_b64 s[8:9], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s10, v42
	v_readfirstlane_b32 s11, v43
	s_and_b64 s[10:11], s[10:11], s[2:3]
	s_mul_i32 s12, s11, 24
	s_mul_hi_u32 s13, s10, 24
	s_add_i32 s13, s13, s12
	s_mul_i32 s12, s10, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[40:41], v[36:37], 0, s[12:13]
	s_and_saveexec_b64 s[12:13], s[0:1]
	s_cbranch_execz .LBB0_661
	v_mov_b32_e32 v42, s8
	v_mov_b32_e32 v43, s9
	v_mov_b32_e32 v44, 2
	v_mov_b32_e32 v45, 1
	global_store_dwordx4 v[40:41], v[42:45], off offset:8
.LBB0_661:
	s_or_b64 exec, exec, s[12:13]
	s_lshl_b64 s[8:9], s[10:11], 12
	v_lshl_add_u64 v[38:39], v[38:39], 0, s[8:9]
	s_mov_b32 s8, 0
	v_cvt_f64_f32_e32 v[42:43], v52
	v_readfirstlane_b32 s12, v38
	v_readfirstlane_b32 s13, v39
	v_mov_b32_e32 v44, s8
	v_mov_b32_e32 v45, s8
	s_mov_b32 s10, s8
	s_mov_b32 s11, s8
	v_and_b32_e32 v32, 0xffffff1d, v32
	global_store_dwordx4 v48, v[42:45], s[12:13] offset:16
	s_mov_b32 s9, s8
	v_or_b32_e32 v32, 0x42, v32
	v_mov_b64_e32 v[44:45], s[10:11]
	v_mov_b64_e32 v[42:43], s[8:9]
	global_store_dwordx4 v48, v[32:35], s[12:13]
	global_store_dwordx4 v48, v[42:45], s[12:13] offset:32
	global_store_dwordx4 v48, v[42:45], s[12:13] offset:48
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_669
	v_mov_b32_e32 v42, 0
	global_load_dwordx2 v[56:57], v42, s[4:5] offset:32 sc0 sc1
	global_load_dwordx2 v[32:33], v42, s[4:5] offset:40
	v_mov_b32_e32 v54, s2
	v_mov_b32_e32 v55, s3
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v32
	v_readfirstlane_b32 s11, v33
	s_and_b64 s[10:11], s[10:11], s[2:3]
	s_mul_i32 s11, s11, 24
	s_mul_hi_u32 s12, s10, 24
	s_mul_i32 s10, s10, 24
	s_add_i32 s11, s12, s11
	v_lshl_add_u64 v[32:33], v[36:37], 0, s[10:11]
	global_store_dwordx2 v[32:33], v[56:57], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[38:39], v42, v[54:57], s[4:5] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[38:39], v[56:57]
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_665
	s_mov_b64 s[12:13], 0
.LBB0_664:
	s_sleep 1
	global_store_dwordx2 v[32:33], v[38:39], off
	v_mov_b32_e32 v36, s2
	v_mov_b32_e32 v37, s3
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v42, v[36:39], s[4:5] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[36:37], v[38:39]
	s_or_b64 s[12:13], vcc, s[12:13]
	v_mov_b64_e32 v[38:39], v[36:37]
	s_andn2_b64 exec, exec, s[12:13]
	s_cbranch_execnz .LBB0_664
.LBB0_665:
	s_or_b64 exec, exec, s[10:11]
	v_mov_b32_e32 v37, 0
	global_load_dwordx2 v[32:33], v37, s[4:5] offset:16
	s_mov_b64 s[10:11], exec
	v_mbcnt_lo_u32_b32 v36, s10, 0
	v_mbcnt_hi_u32_b32 v36, s11, v36
	v_cmp_eq_u32_e32 vcc, 0, v36
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB0_667
	s_bcnt1_i32_b64 s10, s[10:11]
	v_mov_b32_e32 v36, s10
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[32:33], v[36:37], off offset:8 sc1
.LBB0_667:
	s_or_b64 exec, exec, s[12:13]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[36:37], v[32:33], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[36:37]
	s_cbranch_vccnz .LBB0_669
	global_load_dword v32, v[32:33], off offset:24
	v_mov_b32_e32 v33, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v32
	s_and_b32 m0, s10, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[36:37], v[32:33], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_669:
	s_or_b64 exec, exec, s[8:9]
	s_branch .LBB0_673
.LBB0_670:
	s_or_b64 exec, exec, s[8:9]
	v_readfirstlane_b32 s8, v32
	s_cmp_eq_u32 s8, 0
	s_cbranch_scc1 .LBB0_672
	s_sleep 1
	s_cbranch_execnz .LBB0_673
	s_branch .LBB0_675
.LBB0_672:
	s_branch .LBB0_675
.LBB0_673:
	v_mov_b32_e32 v32, 1
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_670
	global_load_dword v32, v[40:41], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v32, 1, v32
	s_branch .LBB0_670
.LBB0_675:
	s_and_b64 exec, exec, s[0:1]
	s_cbranch_execz .LBB0_678
	v_mov_b32_e32 v40, 0
	global_load_dwordx2 v[32:33], v40, s[4:5] offset:40
	global_load_dwordx2 v[44:45], v40, s[4:5] offset:24 sc0 sc1
	global_load_dwordx2 v[36:37], v40, s[4:5]
	s_waitcnt vmcnt(2)
	v_readfirstlane_b32 s8, v32
	v_readfirstlane_b32 s9, v33
	s_add_u32 s10, s8, 1
	s_addc_u32 s11, s9, 0
	s_add_u32 s0, s10, s2
	s_addc_u32 s1, s11, s3
	s_cmp_eq_u64 s[0:1], 0
	s_cselect_b32 s1, s11, s1
	s_cselect_b32 s0, s10, s0
	s_and_b64 s[2:3], s[0:1], s[8:9]
	s_mul_i32 s3, s3, 24
	s_mul_hi_u32 s8, s2, 24
	s_mul_i32 s2, s2, 24
	s_add_i32 s3, s8, s3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[32:33], v[36:37], 0, s[2:3]
	v_mov_b32_e32 v42, s0
	global_store_dwordx2 v[32:33], v[44:45], off
	v_mov_b32_e32 v43, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[38:39], v40, v[42:45], s[4:5] offset:24 sc0 sc1
	s_mov_b64 s[2:3], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[38:39], v[44:45]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_678
.LBB0_677:
	s_sleep 1
	global_store_dwordx2 v[32:33], v[38:39], off
	v_mov_b32_e32 v36, s0
	v_mov_b32_e32 v37, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v40, v[36:39], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[36:37], v[38:39]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[38:39], v[36:37]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_677
.LBB0_678:
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s0, v46
	v_mov_b64_e32 v[40:41], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v46
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_cbranch_execz .LBB0_684
	v_mov_b32_e32 v36, 0
	global_load_dwordx2 v[42:43], v36, s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[32:33], v36, s[4:5] offset:40
	global_load_dwordx2 v[38:39], v36, s[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v32, v32, v42
	v_and_b32_e32 v33, v33, v43
	v_mul_lo_u32 v33, v33, 24
	v_mul_hi_u32 v37, v32, 24
	v_add_u32_e32 v33, v37, v33
	v_mul_lo_u32 v32, v32, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[32:33], v[38:39], 0, v[32:33]
	global_load_dwordx2 v[40:41], v[32:33], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[40:41], v36, v[40:43], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[40:41], v[42:43]
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_683
	s_mov_b64 s[8:9], 0
.LBB0_681:
	s_sleep 1
	global_load_dwordx2 v[32:33], v36, s[4:5] offset:40
	global_load_dwordx2 v[38:39], v36, s[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v32, v32, v40
	v_and_b32_e32 v37, v33, v41
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[32:33], s[10:11], v32, 24, v[38:39]
	v_mov_b32_e32 v38, v33
	v_mad_u64_u32 v[38:39], s[10:11], v37, 24, v[38:39]
	v_mov_b32_e32 v33, v38
	global_load_dwordx2 v[38:39], v[32:33], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v36, v[38:41], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[32:33], v[40:41]
	s_or_b64 s[8:9], vcc, s[8:9]
	v_mov_b64_e32 v[40:41], v[32:33]
	s_andn2_b64 exec, exec, s[8:9]
	s_cbranch_execnz .LBB0_681
	s_or_b64 exec, exec, s[8:9]
	v_mov_b64_e32 v[40:41], v[32:33]
.LBB0_683:
	s_or_b64 exec, exec, s[6:7]
.LBB0_684:
	s_or_b64 exec, exec, s[2:3]
	v_mov_b32_e32 v49, 0
	global_load_dwordx2 v[32:33], v49, s[4:5] offset:40
	global_load_dwordx4 v[36:39], v49, s[4:5]
	v_readfirstlane_b32 s3, v41
	v_readfirstlane_b32 s2, v40
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s8, v32
	v_readfirstlane_b32 s9, v33
	s_and_b64 s[8:9], s[8:9], s[2:3]
	s_mul_i32 s10, s9, 24
	s_mul_hi_u32 s11, s8, 24
	s_add_i32 s11, s11, s10
	s_mul_i32 s10, s8, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[32:33], v[36:37], 0, s[10:11]
	s_and_saveexec_b64 s[10:11], s[0:1]
	s_cbranch_execz .LBB0_686
	v_mov_b32_e32 v40, s6
	v_mov_b32_e32 v41, s7
	v_mov_b32_e32 v42, 2
	v_mov_b32_e32 v43, 1
	global_store_dwordx4 v[32:33], v[40:43], off offset:8
.LBB0_686:
	s_or_b64 exec, exec, s[10:11]
	s_lshl_b64 s[6:7], s[8:9], 12
	v_lshl_add_u64 v[40:41], v[38:39], 0, s[6:7]
	s_mov_b32 s8, 0
	v_mov_b32_e32 v42, 33
	v_mov_b32_e32 v43, v49
	v_mov_b32_e32 v44, v49
	v_mov_b32_e32 v45, v49
	v_readfirstlane_b32 s6, v40
	v_readfirstlane_b32 s7, v41
	s_mov_b32 s10, s8
	s_mov_b32 s11, s8
	s_mov_b32 s9, s8
	s_nop 1
	global_store_dwordx4 v48, v[42:45], s[6:7]
	s_nop 1
	v_mov_b64_e32 v[44:45], s[10:11]
	v_mov_b64_e32 v[42:43], s[8:9]
	global_store_dwordx4 v48, v[42:45], s[6:7] offset:16
	global_store_dwordx4 v48, v[42:45], s[6:7] offset:32
	global_store_dwordx4 v48, v[42:45], s[6:7] offset:48
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_694
	v_mov_b32_e32 v44, 0
	global_load_dwordx2 v[56:57], v44, s[4:5] offset:32 sc0 sc1
	global_load_dwordx2 v[38:39], v44, s[4:5] offset:40
	v_mov_b32_e32 v54, s2
	v_mov_b32_e32 v55, s3
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s8, v38
	v_readfirstlane_b32 s9, v39
	s_and_b64 s[8:9], s[8:9], s[2:3]
	s_mul_i32 s9, s9, 24
	s_mul_hi_u32 s10, s8, 24
	s_mul_i32 s8, s8, 24
	s_add_i32 s9, s10, s9
	v_lshl_add_u64 v[42:43], v[36:37], 0, s[8:9]
	global_store_dwordx2 v[42:43], v[56:57], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[38:39], v44, v[54:57], s[4:5] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[38:39], v[56:57]
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_690
	s_mov_b64 s[10:11], 0
.LBB0_689:
	s_sleep 1
	global_store_dwordx2 v[42:43], v[38:39], off
	v_mov_b32_e32 v36, s2
	v_mov_b32_e32 v37, s3
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v44, v[36:39], s[4:5] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[36:37], v[38:39]
	s_or_b64 s[10:11], vcc, s[10:11]
	v_mov_b64_e32 v[38:39], v[36:37]
	s_andn2_b64 exec, exec, s[10:11]
	s_cbranch_execnz .LBB0_689
.LBB0_690:
	s_or_b64 exec, exec, s[8:9]
	v_mov_b32_e32 v39, 0
	global_load_dwordx2 v[36:37], v39, s[4:5] offset:16
	s_mov_b64 s[8:9], exec
	v_mbcnt_lo_u32_b32 v38, s8, 0
	v_mbcnt_hi_u32_b32 v38, s9, v38
	v_cmp_eq_u32_e32 vcc, 0, v38
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_692
	s_bcnt1_i32_b64 s8, s[8:9]
	v_mov_b32_e32 v38, s8
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[36:37], v[38:39], off offset:8 sc1
.LBB0_692:
	s_or_b64 exec, exec, s[10:11]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[38:39], v[36:37], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[38:39]
	s_cbranch_vccnz .LBB0_694
	global_load_dword v36, v[36:37], off offset:24
	v_mov_b32_e32 v37, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s8, v36
	s_and_b32 m0, s8, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[38:39], v[36:37], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_694:
	s_or_b64 exec, exec, s[6:7]
	v_lshl_add_u64 v[36:37], v[40:41], 0, v[48:49]
	s_branch .LBB0_698
.LBB0_695:
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s6, v38
	s_cmp_eq_u32 s6, 0
	s_cbranch_scc1 .LBB0_697
	s_sleep 1
	s_cbranch_execnz .LBB0_698
	s_branch .LBB0_700
.LBB0_697:
	s_branch .LBB0_700
.LBB0_698:
	v_mov_b32_e32 v38, 1
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_695
	global_load_dword v38, v[32:33], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v38, 1, v38
	s_branch .LBB0_695
.LBB0_700:
	global_load_dwordx2 v[36:37], v[36:37], off
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_703
	v_mov_b32_e32 v42, 0
	global_load_dwordx2 v[32:33], v42, s[4:5] offset:40
	global_load_dwordx2 v[56:57], v42, s[4:5] offset:24 sc0 sc1
	global_load_dwordx2 v[38:39], v42, s[4:5]
	s_waitcnt vmcnt(2)
	v_readfirstlane_b32 s8, v32
	v_readfirstlane_b32 s9, v33
	s_add_u32 s10, s8, 1
	s_addc_u32 s11, s9, 0
	s_add_u32 s0, s10, s2
	s_addc_u32 s1, s11, s3
	s_cmp_eq_u64 s[0:1], 0
	s_cselect_b32 s1, s11, s1
	s_cselect_b32 s0, s10, s0
	s_and_b64 s[2:3], s[0:1], s[8:9]
	s_mul_i32 s3, s3, 24
	s_mul_hi_u32 s8, s2, 24
	s_mul_i32 s2, s2, 24
	s_add_i32 s3, s8, s3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[32:33], v[38:39], 0, s[2:3]
	v_mov_b32_e32 v54, s0
	global_store_dwordx2 v[32:33], v[56:57], off
	v_mov_b32_e32 v55, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[40:41], v42, v[54:57], s[4:5] offset:24 sc0 sc1
	s_mov_b64 s[2:3], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[40:41], v[56:57]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_703
.LBB0_702:
	s_sleep 1
	global_store_dwordx2 v[32:33], v[40:41], off
	v_mov_b32_e32 v38, s0
	v_mov_b32_e32 v39, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[38:39], v42, v[38:41], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[38:39], v[40:41]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[40:41], v[38:39]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_702
.LBB0_703:
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s0, v46
	v_mov_b64_e32 v[44:45], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v46
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_cbranch_execz .LBB0_709
	v_mov_b32_e32 v38, 0
	global_load_dwordx2 v[42:43], v38, s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[32:33], v38, s[4:5] offset:40
	global_load_dwordx2 v[40:41], v38, s[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v32, v32, v42
	v_and_b32_e32 v33, v33, v43
	v_mul_lo_u32 v33, v33, 24
	v_mul_hi_u32 v39, v32, 24
	v_add_u32_e32 v33, v39, v33
	v_mul_lo_u32 v32, v32, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[32:33], v[40:41], 0, v[32:33]
	global_load_dwordx2 v[40:41], v[32:33], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[44:45], v38, v[40:43], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[44:45], v[42:43]
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_708
	s_mov_b64 s[8:9], 0
.LBB0_706:
	s_sleep 1
	global_load_dwordx2 v[32:33], v38, s[4:5] offset:40
	global_load_dwordx2 v[40:41], v38, s[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v32, v32, v44
	v_and_b32_e32 v39, v33, v45
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[32:33], s[10:11], v32, 24, v[40:41]
	v_mov_b32_e32 v40, v33
	v_mad_u64_u32 v[40:41], s[10:11], v39, 24, v[40:41]
	v_mov_b32_e32 v33, v40
	global_load_dwordx2 v[42:43], v[32:33], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v38, v[42:45], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[32:33], v[44:45]
	s_or_b64 s[8:9], vcc, s[8:9]
	v_mov_b64_e32 v[44:45], v[32:33]
	s_andn2_b64 exec, exec, s[8:9]
	s_cbranch_execnz .LBB0_706
	s_or_b64 exec, exec, s[8:9]
	v_mov_b64_e32 v[44:45], v[32:33]
.LBB0_708:
	s_or_b64 exec, exec, s[6:7]
.LBB0_709:
	s_or_b64 exec, exec, s[2:3]
	v_mov_b32_e32 v49, 0
	global_load_dwordx2 v[32:33], v49, s[4:5] offset:40
	global_load_dwordx4 v[40:43], v49, s[4:5]
	v_readfirstlane_b32 s3, v45
	v_readfirstlane_b32 s2, v44
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s8, v32
	v_readfirstlane_b32 s9, v33
	s_and_b64 s[8:9], s[8:9], s[2:3]
	s_mul_i32 s10, s9, 24
	s_mul_hi_u32 s11, s8, 24
	s_add_i32 s11, s11, s10
	s_mul_i32 s10, s8, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[32:33], v[40:41], 0, s[10:11]
	s_and_saveexec_b64 s[10:11], s[0:1]
	s_cbranch_execz .LBB0_711
	v_mov_b32_e32 v54, s6
	v_mov_b32_e32 v55, s7
	v_mov_b32_e32 v56, 2
	v_mov_b32_e32 v57, 1
	global_store_dwordx4 v[32:33], v[54:57], off offset:8
.LBB0_711:
	s_or_b64 exec, exec, s[10:11]
	s_lshl_b64 s[6:7], s[8:9], 12
	v_lshl_add_u64 v[42:43], v[42:43], 0, s[6:7]
	v_and_b32_e32 v36, 0xffffff1f, v36
	v_or_b32_e32 v36, 0x80, v36
	v_mov_b32_e32 v39, 0x2d325e5d
	v_mov_b32_e32 v38, 0x64255b42
	v_readfirstlane_b32 s6, v42
	v_readfirstlane_b32 s7, v43
	s_nop 4
	global_store_dwordx4 v48, v[36:39], s[6:7]
	s_nop 1
	v_mov_b32_e32 v36, 0x255b4134
	v_mov_b32_e32 v37, 0x5b435d64
	v_mov_b32_e32 v38, 0x205d6425
	v_mov_b32_e32 v39, 0x3425203d
	global_store_dwordx4 v48, v[36:39], s[6:7] offset:16
	s_nop 1
	v_mov_b32_e32 v36, 0xa66332e
	v_mov_b32_e32 v37, v49
	v_mov_b32_e32 v38, v49
	v_mov_b32_e32 v39, v49
	global_store_dwordx4 v48, v[36:39], s[6:7] offset:32
	s_nop 1
	v_mov_b32_e32 v36, v49
	global_store_dwordx4 v48, v[36:39], s[6:7] offset:48
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_719
	v_mov_b32_e32 v44, 0
	global_load_dwordx2 v[56:57], v44, s[4:5] offset:32 sc0 sc1
	global_load_dwordx2 v[36:37], v44, s[4:5] offset:40
	v_mov_b32_e32 v54, s2
	v_mov_b32_e32 v55, s3
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s8, v36
	v_readfirstlane_b32 s9, v37
	s_and_b64 s[8:9], s[8:9], s[2:3]
	s_mul_i32 s9, s9, 24
	s_mul_hi_u32 s10, s8, 24
	s_mul_i32 s8, s8, 24
	s_add_i32 s9, s10, s9
	v_lshl_add_u64 v[40:41], v[40:41], 0, s[8:9]
	global_store_dwordx2 v[40:41], v[56:57], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[38:39], v44, v[54:57], s[4:5] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[38:39], v[56:57]
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_715
	s_mov_b64 s[10:11], 0
.LBB0_714:
	s_sleep 1
	global_store_dwordx2 v[40:41], v[38:39], off
	v_mov_b32_e32 v36, s2
	v_mov_b32_e32 v37, s3
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v44, v[36:39], s[4:5] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[36:37], v[38:39]
	s_or_b64 s[10:11], vcc, s[10:11]
	v_mov_b64_e32 v[38:39], v[36:37]
	s_andn2_b64 exec, exec, s[10:11]
	s_cbranch_execnz .LBB0_714
.LBB0_715:
	s_or_b64 exec, exec, s[8:9]
	v_mov_b32_e32 v39, 0
	global_load_dwordx2 v[36:37], v39, s[4:5] offset:16
	s_mov_b64 s[8:9], exec
	v_mbcnt_lo_u32_b32 v38, s8, 0
	v_mbcnt_hi_u32_b32 v38, s9, v38
	v_cmp_eq_u32_e32 vcc, 0, v38
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_717
	s_bcnt1_i32_b64 s8, s[8:9]
	v_mov_b32_e32 v38, s8
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[36:37], v[38:39], off offset:8 sc1
.LBB0_717:
	s_or_b64 exec, exec, s[10:11]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[38:39], v[36:37], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[38:39]
	s_cbranch_vccnz .LBB0_719
	global_load_dword v36, v[36:37], off offset:24
	v_mov_b32_e32 v37, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s8, v36
	s_and_b32 m0, s8, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[38:39], v[36:37], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_719:
	s_or_b64 exec, exec, s[6:7]
	v_lshl_add_u64 v[36:37], v[42:43], 0, v[48:49]
	s_branch .LBB0_723
.LBB0_720:
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s6, v38
	s_cmp_eq_u32 s6, 0
	s_cbranch_scc1 .LBB0_722
	s_sleep 1
	s_cbranch_execnz .LBB0_723
	s_branch .LBB0_725
.LBB0_722:
	s_branch .LBB0_725
.LBB0_723:
	v_mov_b32_e32 v38, 1
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_720
	global_load_dword v38, v[32:33], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v38, 1, v38
	s_branch .LBB0_720
.LBB0_725:
	global_load_dwordx2 v[32:33], v[36:37], off
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_728
	v_mov_b32_e32 v42, 0
	global_load_dwordx2 v[36:37], v42, s[4:5] offset:40
	global_load_dwordx2 v[56:57], v42, s[4:5] offset:24 sc0 sc1
	global_load_dwordx2 v[38:39], v42, s[4:5]
	s_waitcnt vmcnt(2)
	v_readfirstlane_b32 s8, v36
	v_readfirstlane_b32 s9, v37
	s_add_u32 s10, s8, 1
	s_addc_u32 s11, s9, 0
	s_add_u32 s0, s10, s2
	s_addc_u32 s1, s11, s3
	s_cmp_eq_u64 s[0:1], 0
	s_cselect_b32 s1, s11, s1
	s_cselect_b32 s0, s10, s0
	s_and_b64 s[2:3], s[0:1], s[8:9]
	s_mul_i32 s3, s3, 24
	s_mul_hi_u32 s8, s2, 24
	s_mul_i32 s2, s2, 24
	s_add_i32 s3, s8, s3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[40:41], v[38:39], 0, s[2:3]
	v_mov_b32_e32 v54, s0
	global_store_dwordx2 v[40:41], v[56:57], off
	v_mov_b32_e32 v55, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[38:39], v42, v[54:57], s[4:5] offset:24 sc0 sc1
	s_mov_b64 s[2:3], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[38:39], v[56:57]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_728
.LBB0_727:
	s_sleep 1
	global_store_dwordx2 v[40:41], v[38:39], off
	v_mov_b32_e32 v36, s0
	v_mov_b32_e32 v37, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v42, v[36:39], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[36:37], v[38:39]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[38:39], v[36:37]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_727
.LBB0_728:
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s0, v46
	v_mov_b64_e32 v[44:45], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v46
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_cbranch_execz .LBB0_734
	v_mov_b32_e32 v38, 0
	global_load_dwordx2 v[42:43], v38, s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[36:37], v38, s[4:5] offset:40
	global_load_dwordx2 v[40:41], v38, s[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v36, v36, v42
	v_and_b32_e32 v37, v37, v43
	v_mul_lo_u32 v37, v37, 24
	v_mul_hi_u32 v39, v36, 24
	v_add_u32_e32 v37, v39, v37
	v_mul_lo_u32 v36, v36, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[36:37], v[40:41], 0, v[36:37]
	global_load_dwordx2 v[40:41], v[36:37], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[44:45], v38, v[40:43], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[44:45], v[42:43]
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_733
	s_mov_b64 s[8:9], 0
.LBB0_731:
	s_sleep 1
	global_load_dwordx2 v[36:37], v38, s[4:5] offset:40
	global_load_dwordx2 v[40:41], v38, s[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v36, v36, v44
	v_and_b32_e32 v39, v37, v45
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[36:37], s[10:11], v36, 24, v[40:41]
	v_mov_b32_e32 v40, v37
	v_mad_u64_u32 v[40:41], s[10:11], v39, 24, v[40:41]
	v_mov_b32_e32 v37, v40
	global_load_dwordx2 v[42:43], v[36:37], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[36:37], v38, v[42:45], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[36:37], v[44:45]
	s_or_b64 s[8:9], vcc, s[8:9]
	v_mov_b64_e32 v[44:45], v[36:37]
	s_andn2_b64 exec, exec, s[8:9]
	s_cbranch_execnz .LBB0_731
	s_or_b64 exec, exec, s[8:9]
	v_mov_b64_e32 v[44:45], v[36:37]
.LBB0_733:
	s_or_b64 exec, exec, s[6:7]
.LBB0_734:
	s_or_b64 exec, exec, s[2:3]
	v_mov_b32_e32 v42, 0
	global_load_dwordx2 v[36:37], v42, s[4:5] offset:40
	global_load_dwordx4 v[38:41], v42, s[4:5]
	v_readfirstlane_b32 s3, v45
	v_readfirstlane_b32 s2, v44
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s8, v36
	v_readfirstlane_b32 s9, v37
	s_and_b64 s[8:9], s[8:9], s[2:3]
	s_mul_i32 s10, s9, 24
	s_mul_hi_u32 s11, s8, 24
	s_add_i32 s11, s11, s10
	s_mul_i32 s10, s8, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[46:47], v[38:39], 0, s[10:11]
	s_and_saveexec_b64 s[10:11], s[0:1]
	s_cbranch_execz .LBB0_736
	v_mov_b32_e32 v54, s6
	v_mov_b32_e32 v55, s7
	v_mov_b32_e32 v56, 2
	v_mov_b32_e32 v57, 1
	global_store_dwordx4 v[46:47], v[54:57], off offset:8
.LBB0_736:
	s_or_b64 exec, exec, s[10:11]
	s_lshl_b64 s[6:7], s[8:9], 12
	v_lshl_add_u64 v[36:37], v[40:41], 0, s[6:7]
	v_mul_f32_e32 v41, v50, v52
	v_mul_f32_e32 v40, v51, v51
	v_mul_f32_e32 v41, -4.0, v41
	v_add_f32_e32 v40, v40, v41
	v_and_b32_e32 v32, 0xffffff1d, v32
	s_mov_b32 s8, 0
	v_cvt_f64_f32_e32 v[50:51], v40
	v_or_b32_e32 v32, 0x82, v32
	v_readfirstlane_b32 s6, v36
	v_readfirstlane_b32 s7, v37
	v_mov_b32_e32 v36, v34
	v_mov_b32_e32 v37, v35
	v_mov_b32_e32 v52, s8
	v_mov_b32_e32 v53, s8
	v_mov_b32_e32 v43, v42
	v_mov_b32_e32 v44, v42
	v_mov_b32_e32 v45, v42
	global_store_dwordx4 v48, v[32:35], s[6:7]
	global_store_dwordx4 v48, v[34:37], s[6:7] offset:16
	global_store_dwordx4 v48, v[50:53], s[6:7] offset:32
	global_store_dwordx4 v48, v[42:45], s[6:7] offset:48
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_744
	v_mov_b32_e32 v40, 0
	global_load_dwordx2 v[44:45], v40, s[4:5] offset:32 sc0 sc1
	global_load_dwordx2 v[32:33], v40, s[4:5] offset:40
	v_mov_b32_e32 v42, s2
	v_mov_b32_e32 v43, s3
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s8, v32
	v_readfirstlane_b32 s9, v33
	s_and_b64 s[8:9], s[8:9], s[2:3]
	s_mul_i32 s9, s9, 24
	s_mul_hi_u32 s10, s8, 24
	s_mul_i32 s8, s8, 24
	s_add_i32 s9, s10, s9
	v_lshl_add_u64 v[36:37], v[38:39], 0, s[8:9]
	global_store_dwordx2 v[36:37], v[44:45], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[34:35], v40, v[42:45], s[4:5] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[34:35], v[44:45]
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_740
	s_mov_b64 s[10:11], 0
.LBB0_739:
	s_sleep 1
	global_store_dwordx2 v[36:37], v[34:35], off
	v_mov_b32_e32 v32, s2
	v_mov_b32_e32 v33, s3
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v40, v[32:35], s[4:5] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[32:33], v[34:35]
	s_or_b64 s[10:11], vcc, s[10:11]
	v_mov_b64_e32 v[34:35], v[32:33]
	s_andn2_b64 exec, exec, s[10:11]
	s_cbranch_execnz .LBB0_739
.LBB0_740:
	s_or_b64 exec, exec, s[8:9]
	v_mov_b32_e32 v35, 0
	global_load_dwordx2 v[32:33], v35, s[4:5] offset:16
	s_mov_b64 s[8:9], exec
	v_mbcnt_lo_u32_b32 v34, s8, 0
	v_mbcnt_hi_u32_b32 v34, s9, v34
	v_cmp_eq_u32_e32 vcc, 0, v34
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_742
	s_bcnt1_i32_b64 s8, s[8:9]
	v_mov_b32_e32 v34, s8
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[32:33], v[34:35], off offset:8 sc1
.LBB0_742:
	s_or_b64 exec, exec, s[10:11]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[34:35], v[32:33], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[34:35]
	s_cbranch_vccnz .LBB0_744
	global_load_dword v32, v[32:33], off offset:24
	v_mov_b32_e32 v33, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s8, v32
	s_and_b32 m0, s8, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[34:35], v[32:33], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_744:
	s_or_b64 exec, exec, s[6:7]
	s_branch .LBB0_748
.LBB0_745:
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s6, v32
	s_cmp_eq_u32 s6, 0
	s_cbranch_scc1 .LBB0_747
	s_sleep 1
	s_cbranch_execnz .LBB0_748
	s_branch .LBB0_750
.LBB0_747:
	s_branch .LBB0_750
.LBB0_748:
	v_mov_b32_e32 v32, 1
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_745
	global_load_dword v32, v[46:47], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v32, 1, v32
	s_branch .LBB0_745
.LBB0_750:
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_753
	v_mov_b32_e32 v38, 0
	global_load_dwordx2 v[32:33], v38, s[4:5] offset:40
	global_load_dwordx2 v[42:43], v38, s[4:5] offset:24 sc0 sc1
	global_load_dwordx2 v[34:35], v38, s[4:5]
	s_waitcnt vmcnt(2)
	v_readfirstlane_b32 s8, v32
	v_readfirstlane_b32 s9, v33
	s_add_u32 s10, s8, 1
	s_addc_u32 s11, s9, 0
	s_add_u32 s0, s10, s2
	s_addc_u32 s1, s11, s3
	s_cmp_eq_u64 s[0:1], 0
	s_cselect_b32 s1, s11, s1
	s_cselect_b32 s0, s10, s0
	s_and_b64 s[2:3], s[0:1], s[8:9]
	s_mul_i32 s3, s3, 24
	s_mul_hi_u32 s8, s2, 24
	s_mul_i32 s2, s2, 24
	s_add_i32 s3, s8, s3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[36:37], v[34:35], 0, s[2:3]
	v_mov_b32_e32 v40, s0
	global_store_dwordx2 v[36:37], v[42:43], off
	v_mov_b32_e32 v41, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[34:35], v38, v[40:43], s[4:5] offset:24 sc0 sc1
	s_mov_b64 s[2:3], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[34:35], v[42:43]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_753
.LBB0_752:
	s_sleep 1
	global_store_dwordx2 v[36:37], v[34:35], off
	v_mov_b32_e32 v32, s0
	v_mov_b32_e32 v33, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[32:33], v38, v[32:35], s[4:5] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[32:33], v[34:35]
	s_or_b64 s[2:3], vcc, s[2:3]
	v_mov_b64_e32 v[34:35], v[32:33]
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_752
.LBB0_753:
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
		.amdhsa_next_free_vgpr 58
		.amdhsa_next_free_sgpr 14
		.amdhsa_accum_offset 60
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

	.set vec_add.num_vgpr, 58
	.set vec_add.num_agpr, 0
	.set vec_add.numbered_sgpr, 14
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
    .sgpr_count:     20
    .sgpr_spill_count: 0
    .symbol:         vec_add.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     58
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
