
_Z18MatrixAddGlobalMemPfS_S_iii:        ; @_Z18MatrixAddGlobalMemPfS_S_iii
; %bb.0:
	s_load_dword s8, s[4:5], 0x34
	s_load_dwordx4 s[0:3], s[4:5], 0x18
	v_bfe_u32 v1, v0, 10, 10
	v_and_b32_e32 v0, 0x3ff, v0
	s_waitcnt lgkmcnt(0)
	s_lshr_b32 s3, s8, 16
	s_mul_i32 s7, s7, s3
	s_and_b32 s8, s8, 0xffff
	v_add_u32_e32 v1, s7, v1
	s_mul_i32 s6, s6, s8
	v_mul_lo_u32 v1, v1, s0
	v_add3_u32 v0, s6, v0, v1
	s_mul_i32 s0, s1, s0
	v_add_u32_e32 v2, s2, v0
	v_cmp_gt_i32_e32 vcc, s0, v2
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_2
; %bb.1:
	s_load_dwordx4 s[0:3], s[4:5], 0x0
	s_load_dwordx2 s[6:7], s[4:5], 0x10
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshlrev_b64 v[2:3], 2, v[2:3]
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v1, s3
	v_add_co_u32_e32 v4, vcc, s2, v2
	v_addc_co_u32_e32 v5, vcc, v1, v3, vcc
	v_mov_b32_e32 v1, s1
	v_add_co_u32_e32 v2, vcc, s0, v2
	v_addc_co_u32_e32 v3, vcc, v1, v3, vcc
	global_load_dword v6, v[2:3], off
	global_load_dword v7, v[4:5], off
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshlrev_b64 v[0:1], 2, v[0:1]
	v_mov_b32_e32 v2, s7
	v_add_co_u32_e32 v0, vcc, s6, v0
	v_addc_co_u32_e32 v1, vcc, v2, v1, vcc
	s_waitcnt vmcnt(0)
	v_add_f32_e32 v2, v6, v7
	global_store_dword v[0:1], v2, off
.LBB0_2:
	s_endpgm