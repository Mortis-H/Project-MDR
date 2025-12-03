	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

	.amdhsa_code_object_version 6

	.section	.text._Z23attend_bwd_combined_kerILi128EEv25attn_bwd_combined_globalsIXT_EE,"axG",@progbits,_Z23attend_bwd_combined_kerILi128EEv25attn_bwd_combined_globalsIXT_EE,comdat
	.protected	_Z23attend_bwd_combined_kerILi128EEv25attn_bwd_combined_globalsIXT_EE
	.globl	_Z23attend_bwd_combined_kerILi128EEv25attn_bwd_combined_globalsIXT_EE
	.p2align	8
	.type	_Z23attend_bwd_combined_kerILi128EEv25attn_bwd_combined_globalsIXT_EE,@function
_Z23attend_bwd_combined_kerILi128EEv25attn_bwd_combined_globalsIXT_EE:
	s_load_dwordx2 s[50:51], s[0:1], 0x0
	s_load_dwordx4 s[20:23], s[0:1], 0x10
	s_load_dword s33, s[0:1], 0x20
	s_load_dwordx2 s[6:7], s[0:1], 0x30
	s_load_dwordx4 s[28:31], s[0:1], 0x40
	s_lshl_b32 s61, s2, 3
	s_waitcnt lgkmcnt(0)
	s_load_dword s21, s[0:1], 0x50
	s_load_dwordx8 s[8:15], s[0:1], 0x60
	s_cmp_lg_u32 0, -1
	s_mov_b64 s[34:35], src_shared_base
	s_waitcnt lgkmcnt(0)
	s_cselect_b32 s13, 0, 0
	s_cselect_b32 s5, s35, 0
	s_and_b32 s58, s13, 15
	s_and_b32 s15, s13, -16
	s_load_dword s11, s[0:1], 0x80
	s_load_dwordx2 s[52:53], s[0:1], 0x90
	s_load_dwordx4 s[24:27], s[0:1], 0xa0
	s_load_dword s60, s[0:1], 0xb0
	s_load_dwordx2 s[54:55], s[0:1], 0x150
	s_load_dword s63, s[0:1], 0x170
	s_load_dwordx4 s[36:39], s[0:1], 0x160
	s_load_dword s65, s[0:1], 0x1a0
	s_load_dwordx2 s[56:57], s[0:1], 0x180
	s_load_dwordx4 s[16:19], s[0:1], 0x190
	s_add_u32 s15, s15, 16
	s_mov_b32 s59, 0
	s_waitcnt lgkmcnt(0)
	s_addc_u32 s17, s5, 0
	s_cmp_eq_u64 s[58:59], 0
	s_cselect_b32 s68, s13, s15
	s_cselect_b32 s5, s5, s17
	s_add_u32 s13, s68, 0x10000
	s_addc_u32 s5, s5, 0
	s_and_b32 s58, s13, 15
	s_and_b32 s15, s13, -16
	s_add_u32 s15, s15, 16
	s_addc_u32 s17, s5, 0
	s_cmp_eq_u64 s[58:59], 0
	s_cselect_b32 s27, s13, s15
	s_cselect_b32 s5, s5, s17
	s_add_u32 s13, s27, 0x8000
	s_addc_u32 s5, s5, 0
	s_and_b32 s58, s13, 15
	s_and_b32 s15, s13, -16
	s_add_u32 s15, s15, 16
	s_addc_u32 s17, s5, 0
	s_cmp_eq_u64 s[58:59], 0
	s_cselect_b32 s62, s13, s15
	s_cselect_b32 s5, s5, s17
	s_add_u32 s13, s62, 0x8000
	s_addc_u32 s5, s5, 0
	s_and_b32 s58, s13, 15
	s_and_b32 s15, s13, -16
	s_add_u32 s15, s15, 16
	s_addc_u32 s17, s5, 0
	s_cmp_eq_u64 s[58:59], 0
	s_cselect_b32 s69, s13, s15
	s_cselect_b32 s5, s5, s17
	s_add_u32 s13, s69, 0x2000
	s_addc_u32 s5, s5, 0
	s_and_b32 s58, s13, 15
	s_and_b32 s15, s13, -16
	s_add_u32 s15, s15, 16
	s_addc_u32 s17, s5, 0
	s_cmp_eq_u64 s[58:59], 0
	s_cselect_b32 s5, s5, s17
	s_cselect_b32 s5, s13, s15
	s_add_u32 s13, s5, 0x200
	s_and_b32 s15, s13, -16
	s_and_b32 s58, s13, 15
	s_add_u32 s15, s15, 16
	s_cmp_eq_u64 s[58:59], 0
	s_cselect_b32 s23, s13, s15
	s_lshl_b32 s3, s3, 8
	s_mul_i32 s17, s4, s28
	s_add_i32 s17, s17, s3
	v_lshlrev_b32_e32 v6, 4, v0
	s_mul_i32 s17, s17, s30
	v_bitop3_b32 v1, v0, v6, 32 bitop3:0x6c
	v_lshrrev_b32_e32 v2, 1, v0
	s_add_i32 s17, s17, s2
	v_lshrrev_b32_e32 v1, 1, v1
	v_and_b32_e32 v4, 0x60, v2
	s_mul_i32 s28, s17, s21
	s_mul_i32 s13, s22, s33
	v_bfe_u32 v3, v0, 2, 4
	v_and_or_b32 v4, v1, 24, v4
	s_ashr_i32 s29, s28, 31
	v_mad_u64_u32 v[4:5], s[34:35], v3, s13, v[4:5]
	s_lshl_b32 s15, s13, 4
	s_lshl_b64 s[28:29], s[28:29], 1
	v_lshlrev_b32_e32 v1, 3, v0
	v_lshlrev_b32_e32 v14, 1, v4
	v_add_lshl_u32 v15, v4, s15, 1
	s_add_u32 s28, s6, s28
	v_and_b32_e32 v4, 8, v1
	s_movk_i32 s6, 0x70
	s_mul_i32 s15, s30, s21
	v_and_b32_e32 v3, 0xc00, v6
	v_bfe_u32 v5, v0, 1, 4
	v_and_or_b32 v4, v2, s6, v4
	s_addc_u32 s29, s7, s29
	v_add_u32_e32 v3, s68, v3
	v_mad_u64_u32 v[4:5], s[6:7], v5, s15, v[4:5]
	v_readfirstlane_b32 s6, v3
	s_mov_b32 m0, s6
	s_lshl_b32 s6, s15, 4
	v_add_u32_e32 v7, 0x1000, v3
	s_lshl_b32 s30, s15, 9
	s_mov_b32 s31, 0x110000
	v_lshlrev_b32_e32 v5, 1, v4
	v_add_u32_e32 v4, s6, v4
	v_readfirstlane_b32 s7, v7
	v_add_u32_e32 v7, 0x2000, v3
	buffer_load_dwordx4 v5, s[28:31], 0 offen lds
	v_lshlrev_b32_e32 v5, 1, v4
	s_mov_b32 m0, s7
	v_add_u32_e32 v4, s6, v4
	v_readfirstlane_b32 s7, v7
	v_add_u32_e32 v7, 0x3000, v3
	buffer_load_dwordx4 v5, s[28:31], 0 offen lds
	v_lshlrev_b32_e32 v5, 1, v4
	s_mov_b32 m0, s7
	v_add_u32_e32 v4, s6, v4
	v_readfirstlane_b32 s7, v7
	v_add_u32_e32 v7, 0x4000, v3
	buffer_load_dwordx4 v5, s[28:31], 0 offen lds
	v_lshlrev_b32_e32 v5, 1, v4
	s_mov_b32 m0, s7
	v_add_u32_e32 v4, s6, v4
	v_readfirstlane_b32 s7, v7
	v_add_u32_e32 v7, 0x5000, v3
	buffer_load_dwordx4 v5, s[28:31], 0 offen lds
	v_lshlrev_b32_e32 v5, 1, v4
	s_mov_b32 m0, s7
	v_add_u32_e32 v4, s6, v4
	v_readfirstlane_b32 s7, v7
	v_add_u32_e32 v7, 0x6000, v3
	buffer_load_dwordx4 v5, s[28:31], 0 offen lds
	v_lshlrev_b32_e32 v5, 1, v4
	s_mov_b32 m0, s7
	v_add_u32_e32 v4, s6, v4
	v_readfirstlane_b32 s7, v7
	v_add_u32_e32 v7, 0x7000, v3
	buffer_load_dwordx4 v5, s[28:31], 0 offen lds
	v_lshlrev_b32_e32 v5, 1, v4
	s_mov_b32 m0, s7
	v_add_u32_e32 v4, s6, v4
	v_readfirstlane_b32 s7, v7
	v_add_u32_e32 v7, 0x8000, v3
	buffer_load_dwordx4 v5, s[28:31], 0 offen lds
	v_lshlrev_b32_e32 v5, 1, v4
	s_mov_b32 m0, s7
	v_add_u32_e32 v4, s6, v4
	v_readfirstlane_b32 s7, v7
	v_add_u32_e32 v7, 0x9000, v3
	buffer_load_dwordx4 v5, s[28:31], 0 offen lds
	v_lshlrev_b32_e32 v5, 1, v4
	s_mov_b32 m0, s7
	v_add_u32_e32 v4, s6, v4
	v_readfirstlane_b32 s7, v7
	v_add_u32_e32 v7, 0xa000, v3
	buffer_load_dwordx4 v5, s[28:31], 0 offen lds
	v_lshlrev_b32_e32 v5, 1, v4
	s_mov_b32 m0, s7
	v_add_u32_e32 v4, s6, v4
	v_readfirstlane_b32 s7, v7
	v_add_u32_e32 v7, 0xb000, v3
	buffer_load_dwordx4 v5, s[28:31], 0 offen lds
	v_lshlrev_b32_e32 v5, 1, v4
	s_mov_b32 m0, s7
	v_add_u32_e32 v4, s6, v4
	v_readfirstlane_b32 s7, v7
	v_add_u32_e32 v7, 0xc000, v3
	buffer_load_dwordx4 v5, s[28:31], 0 offen lds
	v_lshlrev_b32_e32 v5, 1, v4
	s_mov_b32 m0, s7
	v_add_u32_e32 v4, s6, v4
	v_readfirstlane_b32 s7, v7
	v_add_u32_e32 v7, 0xd000, v3
	buffer_load_dwordx4 v5, s[28:31], 0 offen lds
	v_lshlrev_b32_e32 v5, 1, v4
	s_mov_b32 m0, s7
	v_add_u32_e32 v4, s6, v4
	v_readfirstlane_b32 s7, v7
	v_add_u32_e32 v7, 0xe000, v3
	buffer_load_dwordx4 v5, s[28:31], 0 offen lds
	v_lshlrev_b32_e32 v5, 1, v4
	s_mov_b32 m0, s7
	v_add_u32_e32 v4, s6, v4
	v_readfirstlane_b32 s7, v7
	v_add_u32_e32 v3, 0xf000, v3
	buffer_load_dwordx4 v5, s[28:31], 0 offen lds
	v_lshlrev_b32_e32 v5, 1, v4
	s_mov_b32 m0, s7
	v_add_lshl_u32 v4, v4, s6, 1
	v_readfirstlane_b32 s6, v3
	buffer_load_dwordx4 v5, s[28:31], 0 offen lds
	s_mov_b32 m0, s6
	s_mul_i32 s6, s4, s12
	s_mul_i32 s6, s6, s14
	s_add_i32 s6, s6, s2
	s_mul_i32 s6, s6, s11
	s_ashr_i32 s7, s6, 31
	buffer_load_dwordx4 v4, s[28:31], 0 offen lds
	s_lshl_b64 s[6:7], s[6:7], 1
	v_and_b32_e32 v4, 24, v2
	v_and_b32_e32 v2, 0xc0, v0
	s_add_u32 s8, s8, s6
	s_mul_i32 s6, s11, s14
	v_or_b32_e32 v3, s3, v2
	v_and_b32_e32 v5, 15, v0
	v_mul_lo_u32 v2, s6, v3
	s_addc_u32 s9, s9, s7
	s_mul_i32 s7, s6, s10
	v_mul_lo_u32 v5, v5, s6
	v_or_b32_e32 v2, v2, v4
	s_mul_i32 s7, s7, s12
	v_add_lshl_u32 v7, v5, v2, 1
	s_lshl_b32 s10, s7, 1
	s_mov_b32 s11, 0x20000
	buffer_load_dwordx4 a[48:51], v7, s[8:11], 0 offen
	v_or_b32_e32 v7, 32, v2
	v_add_lshl_u32 v8, v5, v7, 1
	buffer_load_dwordx4 a[52:55], v8, s[8:11], 0 offen
	v_add_u32_e32 v8, 64, v2
	v_add_lshl_u32 v9, v5, v8, 1
	buffer_load_dwordx4 a[56:59], v9, s[8:11], 0 offen
	v_add_u32_e32 v9, 0x60, v2
	s_lshl_b32 s3, s6, 4
	v_add_lshl_u32 v10, v5, v9, 1
	v_add_u32_e32 v5, s3, v5
	buffer_load_dwordx4 a[60:63], v10, s[8:11], 0 offen
	v_add_lshl_u32 v10, v5, v2, 1
	buffer_load_dwordx4 a[64:67], v10, s[8:11], 0 offen
	v_add_lshl_u32 v10, v5, v7, 1
	buffer_load_dwordx4 a[68:71], v10, s[8:11], 0 offen
	v_add_lshl_u32 v10, v5, v8, 1
	buffer_load_dwordx4 a[72:75], v10, s[8:11], 0 offen
	v_add_lshl_u32 v10, v5, v9, 1
	v_add_u32_e32 v5, s3, v5
	buffer_load_dwordx4 a[76:79], v10, s[8:11], 0 offen
	v_add_lshl_u32 v10, v5, v2, 1
	buffer_load_dwordx4 a[80:83], v10, s[8:11], 0 offen
	v_add_lshl_u32 v10, v5, v7, 1
	buffer_load_dwordx4 a[84:87], v10, s[8:11], 0 offen
	v_add_lshl_u32 v10, v5, v8, 1
	s_mul_i32 s58, s4, s36
	buffer_load_dwordx4 a[88:91], v10, s[8:11], 0 offen
	v_add_lshl_u32 v10, v5, v9, 1
	v_add_u32_e32 v5, s3, v5
	s_add_i32 s3, s58, s61
	s_mul_i32 s63, s63, s38
	s_mul_i32 s6, s63, s3
	s_ashr_i32 s7, s6, 31
	s_lshl_b64 s[6:7], s[6:7], 2
	s_add_u32 s40, s54, s6
	s_mul_i32 s64, s4, s16
	buffer_load_dwordx4 a[92:95], v10, s[8:11], 0 offen
	s_addc_u32 s41, s55, s7
	s_add_i32 s3, s64, s61
	s_mul_i32 s65, s65, s18
	v_add_lshl_u32 v2, v5, v2, 1
	buffer_load_dwordx4 a[96:99], v2, s[8:11], 0 offen
	s_mul_i32 s6, s65, s3
	v_add_lshl_u32 v2, v5, v7, 1
	buffer_load_dwordx4 a[100:103], v2, s[8:11], 0 offen
	s_ashr_i32 s7, s6, 31
	v_add_lshl_u32 v2, v5, v8, 1
	buffer_load_dwordx4 a[104:107], v2, s[8:11], 0 offen
	s_lshl_b64 s[6:7], s[6:7], 2
	s_mul_i32 s66, s4, s20
	v_add_lshl_u32 v2, v5, v9, 1
	buffer_load_dwordx4 a[108:111], v2, s[8:11], 0 offen
	s_add_u32 s8, s56, s6
	s_mul_i32 s3, s66, s22
	s_addc_u32 s9, s57, s7
	s_add_i32 s3, s3, s61
	s_mul_i32 s6, s3, s33
	v_and_b32_e32 v2, 0x600, v1
	v_lshlrev_b32_e32 v7, 2, v0
	s_ashr_i32 s7, s6, 31
	v_lshlrev_b32_e32 v17, 1, v2
	s_movk_i32 s42, 0x100
	s_mov_b32 s43, s31
	v_and_b32_e32 v16, 0xfc, v7
	s_mov_b32 m0, s5
	s_mov_b64 s[16:17], s[40:41]
	s_lshl_b64 s[6:7], s[6:7], 1
	v_add_u32_e32 v5, s27, v17
	buffer_load_dword v16, s[40:43], 0 offen lds
	s_mov_b64 s[18:19], s[42:43]
	s_mov_b32 s16, s8
	s_mov_b32 s17, s9
	s_mov_b32 m0, s23
	s_add_u32 s28, s50, s6
	v_readfirstlane_b32 s6, v5
	v_add_u32_e32 v5, 0x1000, v5
	buffer_load_dword v16, s[16:19], 0 offen lds
	s_addc_u32 s29, s51, s7
	s_lshl_b32 s30, s13, 6
	s_mov_b32 m0, s6
	v_readfirstlane_b32 s6, v5
	s_mul_i32 s67, s4, s24
	buffer_load_dwordx4 v14, s[28:31], 0 offen lds
	s_mov_b32 m0, s6
	s_mul_i32 s6, s67, s26
	s_add_i32 s12, s6, s61
	s_mul_i32 s6, s12, s60
	s_ashr_i32 s7, s6, 31
	s_lshl_b64 s[6:7], s[6:7], 1
	s_mul_i32 s10, s26, s60
	s_add_u32 s44, s52, s6
	v_add_u32_e32 v5, s62, v17
	s_addc_u32 s45, s53, s7
	s_lshl_b32 s46, s10, 6
	v_readfirstlane_b32 s6, v5
	v_add_u32_e32 v5, 0x1000, v5
	s_lshl_b32 s10, s22, 5
	buffer_load_dwordx4 v15, s[28:31], 0 offen lds
	s_mov_b32 s47, s31
	s_mov_b32 m0, s6
	v_readfirstlane_b32 s6, v5
	s_add_i32 s72, s3, s10
	buffer_load_dwordx4 v14, s[44:47], 0 offen lds
	s_mov_b32 m0, s6
	s_mul_i32 s6, s72, s33
	s_add_i32 s70, s27, 0x2000
	s_ashr_i32 s7, s6, 31
	s_lshl_b64 s[6:7], s[6:7], 1
	v_add_u32_e32 v5, s70, v17
	s_add_u32 s28, s50, s6
	v_readfirstlane_b32 s3, v5
	v_add_u32_e32 v5, 0x1000, v5
	buffer_load_dwordx4 v15, s[44:47], 0 offen lds
	s_addc_u32 s29, s51, s7
	s_mov_b32 m0, s3
	v_readfirstlane_b32 s3, v5
	buffer_load_dwordx4 v14, s[28:31], 0 offen lds
	s_mov_b32 m0, s3
	s_lshl_b32 s3, s26, 5
	s_add_i32 s3, s12, s3
	s_mul_i32 s6, s3, s60
	s_add_i32 s71, s62, 0x2000
	s_ashr_i32 s7, s6, 31
	s_lshl_b64 s[6:7], s[6:7], 1
	v_add_u32_e32 v5, s71, v17
	s_add_u32 s44, s52, s6
	v_readfirstlane_b32 s3, v5
	v_add_u32_e32 v5, 0x1000, v5
	buffer_load_dwordx4 v15, s[28:31], 0 offen lds
	s_addc_u32 s45, s53, s7
	s_mov_b32 m0, s3
	v_readfirstlane_b32 s3, v5
	buffer_load_dwordx4 v14, s[44:47], 0 offen lds
	s_mov_b32 m0, s3
	v_lshrrev_b32_e32 v9, 6, v0
	buffer_load_dwordx4 v15, s[44:47], 0 offen lds
	s_load_dwordx2 s[34:35], s[0:1], 0x140
	s_load_dwordx2 s[6:7], s[0:1], 0xc0
	s_load_dwordx4 s[36:39], s[0:1], 0xd0
	s_load_dword s3, s[0:1], 0xe0
	s_load_dwordx2 s[20:21], s[0:1], 0xf0
	s_load_dwordx2 s[48:49], s[0:1], 0x120
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	s_barrier
	s_load_dwordx2 s[24:25], s[0:1], 0x110
	s_load_dwordx4 s[12:15], s[0:1], 0x100
	s_load_dwordx4 s[16:19], s[0:1], 0x130
	v_and_b32_e32 v8, 32, v0
	s_mov_b32 s37, 2
	v_lshrrev_b32_e32 v11, 2, v0
	s_mov_b32 s35, 1
	s_waitcnt lgkmcnt(0)
	s_add_i32 s25, s5, 0x100
	v_lshlrev_b32_e32 v13, 5, v0
	s_add_u32 s40, s40, 0x100
	v_and_b32_e32 v10, 0x200, v6
	v_and_b32_e32 v12, 16, v0
	v_and_b32_e32 v13, 0x1e0, v13
	s_addc_u32 s41, s41, 0
	s_add_i32 s0, s72, s10
	v_lshlrev_b32_e32 v5, 14, v9
	v_or3_b32 v10, v10, v13, v12
	v_and_b32_e32 v11, 12, v11
	v_and_b32_e32 v18, 0xf0, v6
	v_and_b32_e32 v19, 12, v0
	s_mul_i32 s0, s0, s33
	v_add3_u32 v10, s68, v5, v10
	v_lshlrev_b32_e32 v5, 10, v9
	v_lshl_add_u32 v9, v9, 11, s69
	v_bitop3_b32 v11, v18, v19, v11 bitop3:0x36
	s_add_i32 s19, s27, 0x4000
	s_ashr_i32 s1, s0, 31
	v_lshl_add_u32 v9, v11, 1, v9
	s_lshl_b64 s[0:1], s[0:1], 1
	v_add_u32_e32 v11, s19, v17
	s_mov_b32 m0, s25
	s_add_u32 s28, s50, s0
	v_readfirstlane_b32 s0, v11
	v_add_u32_e32 v11, 0x1000, v11
	v_and_b32_e32 v13, 0x1f8, v1
	buffer_load_dword v16, s[40:43], 0 offen lds
	s_addc_u32 s29, s51, s1
	s_mov_b32 m0, s0
	v_readfirstlane_b32 s0, v11
	v_lshlrev_b32_e32 v11, 6, v0
	v_add3_u32 v1, s68, v5, v13
	v_xad_u32 v4, v13, v4, s69
	buffer_load_dwordx4 v14, s[28:31], 0 offen lds
	s_mov_b32 m0, s0
	v_and_b32_e32 v13, 48, v0
	v_and_b32_e32 v11, 0x3c0, v11
	v_and_b32_e32 v18, 32, v7
	buffer_load_dwordx4 v15, s[28:31], 0 offen lds
	v_bitop3_b32 v11, v11, v18, v13 bitop3:0x36
	v_add_u32_e32 v18, s27, v11
	ds_read_b128 a[112:115], v18
	ds_read_b128 a[116:119], v18 offset:1024
	ds_read_b128 a[120:123], v18 offset:2048
	v_and_b32_e32 v7, 12, v7
	ds_read_b128 a[124:127], v18 offset:3072
	v_or_b32_e32 v13, v7, v13
	v_add_u32_e32 v18, s5, v13
	ds_read_b32 v126, v18

	v_add_u32_e32 v18, s23, v13
	ds_read_b32 v127, v18

	ds_read_b128 a[0:3], v10
	ds_read_b128 a[4:7], v10 offset:1024
	ds_read_b128 a[8:11], v10 offset:2048
	ds_read_b128 a[12:15], v10 offset:3072
	ds_read_b128 a[16:19], v10 offset:4096
	ds_read_b128 a[20:23], v10 offset:5120
	ds_read_b128 a[24:27], v10 offset:6144
	ds_read_b128 a[28:31], v10 offset:7168
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[46:49], a[112:115], a[0:3], 0
	ds_read_b128 a[32:35], v10 offset:8192
	ds_read_b128 a[36:39], v10 offset:9216
	v_mfma_f32_16x16x32_bf16 v[46:49], a[116:119], a[4:7], v[46:49]
	v_mul_f32_e32 v126, 0x3fb8aa3b, v126
	v_mfma_f32_16x16x32_bf16 v[46:49], a[120:123], a[8:11], v[46:49]
	ds_read_b128 a[40:43], v10 offset:10240
	ds_read_b128 a[44:47], v10 offset:11264
	v_mfma_f32_16x16x32_bf16 v[46:49], a[124:127], a[12:15], v[46:49]
	v_mfma_f32_16x16x32_bf16 v[50:53], a[112:115], a[16:19], 0
	ds_read_b128 v[62:65], v10 offset:12288
	ds_read_b128 v[66:69], v10 offset:13312
	v_mfma_f32_16x16x32_bf16 v[50:53], a[116:119], a[20:23], v[50:53]
	v_mfma_f32_16x16x32_bf16 v[50:53], a[120:123], a[24:27], v[50:53]
	ds_read_b128 v[70:73], v10 offset:14336
	v_or_b32_e32 v7, v7, v12
	ds_read_b128 v[74:77], v10 offset:15360
	v_mfma_f32_16x16x32_bf16 v[50:53], a[124:127], a[28:31], v[50:53]
	v_mul_f32_e32 v46, 0x3e0293ee, v46
	v_mul_f32_e32 v47, 0x3e0293ee, v47
	v_mul_f32_e32 v48, 0x3e0293ee, v48
	v_mul_f32_e32 v49, 0x3e0293ee, v49
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_bf16 v[54:57], a[112:115], a[32:35], 0
	v_add_u32_e32 v18, s62, v11
	ds_read_b128 v[78:81], v18
	v_and_b32_e32 v6, 0x2c0, v6
	v_lshlrev_b32_e32 v7, 1, v7
	ds_read_b128 v[82:85], v18 offset:1024
	v_mfma_f32_16x16x32_bf16 v[54:57], a[116:119], a[36:39], v[54:57]
	v_subrev_f32_dpp v46, v126, v46 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v47, v126, v47 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v48, v126, v48 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v49, v126, v49 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_bf16 v[54:57], a[120:123], a[40:43], v[54:57]
	ds_read_b128 v[86:89], v18 offset:2048
	v_bitop3_b32 v12, v7, v8, v6 bitop3:0x36
	ds_read_b128 v[90:93], v18 offset:3072
	v_mfma_f32_16x16x32_bf16 v[54:57], a[124:127], a[44:47], v[54:57]
	v_mul_f32_e32 v50, 0x3e0293ee, v50
	v_mul_f32_e32 v51, 0x3e0293ee, v51
	v_mul_f32_e32 v52, 0x3e0293ee, v52
	v_mul_f32_e32 v53, 0x3e0293ee, v53
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_bf16 v[58:61], a[112:115], v[62:65], 0
	v_add_u32_e32 v6, s62, v12
	ds_read_b64_tr_b16 v[94:95], v6
	ds_read_b64_tr_b16 v[96:97], v6 offset:256
	ds_read_b64_tr_b16 v[98:99], v6 offset:1024
	ds_read_b64_tr_b16 v[100:101], v6 offset:1280
	v_mfma_f32_16x16x32_bf16 v[58:61], a[116:119], v[66:69], v[58:61]
	v_subrev_f32_dpp v50, v126, v50 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v51, v126, v51 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v52, v126, v52 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v53, v126, v53 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_16x16x32_bf16 v[58:61], a[120:123], v[70:73], v[58:61]
	ds_read_b64_tr_b16 v[102:103], v6 offset:2048
	ds_read_b64_tr_b16 v[104:105], v6 offset:2304
	ds_read_b64_tr_b16 v[106:107], v6 offset:3072
	ds_read_b64_tr_b16 v[108:109], v6 offset:3328
	v_mfma_f32_16x16x32_bf16 v[58:61], a[124:127], v[74:77], v[58:61]
	v_mul_f32_e32 v54, 0x3e0293ee, v54
	v_mul_f32_e32 v55, 0x3e0293ee, v55
	v_mul_f32_e32 v56, 0x3e0293ee, v56
	v_mul_f32_e32 v57, 0x3e0293ee, v57
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_16x16x32_bf16 v[62:65], v[78:81], a[48:51], 0
	v_subrev_f32_dpp v54, v126, v54 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v55, v126, v55 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v56, v126, v56 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v57, v126, v57 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	v_mfma_f32_16x16x32_bf16 v[62:65], v[82:85], a[52:55], v[62:65]
	v_exp_f32_e32 v46, v46
	v_exp_f32_e32 v47, v47
	v_exp_f32_e32 v48, v48
	v_exp_f32_e32 v49, v49
	v_mfma_f32_16x16x32_bf16 v[62:65], v[86:89], a[56:59], v[62:65]
	v_add_u32_e32 v6, s27, v12
	ds_read_b64_tr_b16 a[112:113], v6
	ds_read_b64_tr_b16 a[114:115], v6 offset:256
	v_mfma_f32_16x16x32_bf16 v[62:65], v[90:93], a[60:63], v[62:65]
	v_exp_f32_e32 v50, v50
	v_exp_f32_e32 v51, v51
	v_exp_f32_e32 v52, v52
	v_exp_f32_e32 v53, v53
	v_mfma_f32_16x16x32_bf16 v[66:69], v[78:81], a[64:67], 0
	ds_read_b64_tr_b16 a[116:117], v6 offset:1024
	ds_read_b64_tr_b16 a[118:119], v6 offset:1280
	v_mfma_f32_16x16x32_bf16 v[66:69], v[82:85], a[68:71], v[66:69]
	v_mul_f32_e32 v58, 0x3e0293ee, v58
	v_mul_f32_e32 v59, 0x3e0293ee, v59
	v_mul_f32_e32 v60, 0x3e0293ee, v60
	v_mul_f32_e32 v61, 0x3e0293ee, v61
	v_mfma_f32_16x16x32_bf16 v[66:69], v[86:89], a[72:75], v[66:69]
	v_subrev_f32_dpp v58, v126, v58 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v59, v126, v59 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v60, v126, v60 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v61, v126, v61 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	v_mfma_f32_16x16x32_bf16 v[66:69], v[90:93], a[76:79], v[66:69]
	v_cvt_pk_bf16_f32 v118, v46, v47
	v_cvt_pk_bf16_f32 v119, v48, v49
	v_mfma_f32_16x16x32_bf16 v[70:73], v[78:81], a[80:83], 0
	v_exp_f32_e32 v54, v54
	v_exp_f32_e32 v55, v55
	v_exp_f32_e32 v56, v56
	v_exp_f32_e32 v57, v57
	v_mfma_f32_16x16x32_bf16 v[70:73], v[82:85], a[84:87], v[70:73]
	v_cvt_pk_bf16_f32 v120, v50, v51
	v_cvt_pk_bf16_f32 v121, v52, v53
	v_mfma_f32_16x16x32_bf16 v[70:73], v[86:89], a[88:91], v[70:73]
	ds_read_b64_tr_b16 a[120:121], v6 offset:2048
	ds_read_b64_tr_b16 a[122:123], v6 offset:2304
	v_mfma_f32_16x16x32_bf16 v[70:73], v[90:93], a[92:95], v[70:73]
	v_exp_f32_e32 v58, v58
	v_exp_f32_e32 v59, v59
	v_exp_f32_e32 v60, v60
	v_exp_f32_e32 v61, v61
	v_mfma_f32_16x16x32_bf16 v[74:77], v[78:81], a[96:99], 0
	ds_read_b64_tr_b16 a[124:125], v6 offset:3072
	ds_read_b64_tr_b16 a[126:127], v6 offset:3328
	v_mfma_f32_16x16x32_bf16 v[74:77], v[82:85], a[100:103], v[74:77]
	v_cvt_pk_bf16_f32 v122, v54, v55
	v_cvt_pk_bf16_f32 v123, v56, v57
	v_cvt_pk_bf16_f32 v124, v58, v59
	v_cvt_pk_bf16_f32 v125, v60, v61
	v_mfma_f32_16x16x32_bf16 v[74:77], v[86:89], a[104:107], v[74:77]
	v_permlane16_swap_b32_e32 v118, v120
	v_permlane16_swap_b32_e32 v119, v121
	v_permlane16_swap_b32_e32 v122, v124
	v_permlane16_swap_b32_e32 v123, v125
	v_mfma_f32_16x16x32_bf16 v[74:77], v[90:93], a[108:111], v[74:77]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x16_bf16 v[128:143], v[94:97], v[118:121], 0
	ds_read_b64_tr_b16 a[0:1], v1
	ds_read_b64_tr_b16 a[2:3], v1 offset:4096
	ds_read_b64_tr_b16 a[4:5], v1 offset:512
	ds_read_b64_tr_b16 a[6:7], v1 offset:4608
	v_mfma_f32_32x32x16_bf16 v[144:159], v[94:97], v[122:125], 0
	v_subrev_f32_dpp v62, v127, v62 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v63, v127, v63 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v64, v127, v64 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v65, v127, v65 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v66, v127, v66 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v67, v127, v67 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v68, v127, v68 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v69, v127, v69 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	v_mfma_f32_32x32x16_bf16 v[160:175], v[98:101], v[118:121], 0
	ds_read_b64_tr_b16 a[8:9], v1 offset:8192
	ds_read_b64_tr_b16 a[10:11], v1 offset:12288
	ds_read_b64_tr_b16 a[12:13], v1 offset:8704
	ds_read_b64_tr_b16 a[14:15], v1 offset:12800
	v_mfma_f32_32x32x16_bf16 v[176:191], v[98:101], v[122:125], 0
	v_mul_f32_e32 v62, v46, v62
	v_mul_f32_e32 v63, v47, v63
	v_mul_f32_e32 v64, v48, v64
	v_mul_f32_e32 v65, v49, v65
	v_mul_f32_e32 v66, v50, v66
	v_mul_f32_e32 v67, v51, v67
	v_mul_f32_e32 v68, v52, v68
	v_mul_f32_e32 v69, v53, v69
	v_cvt_pk_bf16_f32 v62, v62, v63
	v_cvt_pk_bf16_f32 v63, v64, v65
	v_cvt_pk_bf16_f32 v64, v66, v67
	v_cvt_pk_bf16_f32 v65, v68, v69
	v_subrev_f32_dpp v70, v127, v70 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v71, v127, v71 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v72, v127, v72 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v73, v127, v73 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	v_mfma_f32_32x32x16_bf16 v[192:207], v[102:105], v[118:121], 0
	ds_read_b64_tr_b16 a[16:17], v1 offset:16384
	ds_read_b64_tr_b16 a[18:19], v1 offset:20480
	ds_write_b64 v9, v[62:63]
	ds_write_b64 v9, v[64:65] offset:512
	v_mfma_f32_32x32x16_bf16 v[208:223], v[102:105], v[122:125], 0
	v_subrev_f32_dpp v74, v127, v74 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v75, v127, v75 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v76, v127, v76 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v77, v127, v77 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	v_mul_f32_e32 v70, v54, v70
	v_mul_f32_e32 v71, v55, v71
	v_mul_f32_e32 v72, v56, v72
	v_mul_f32_e32 v73, v57, v73
	v_mul_f32_e32 v74, v58, v74
	v_mul_f32_e32 v75, v59, v75
	v_mul_f32_e32 v76, v60, v76
	v_mul_f32_e32 v77, v61, v77
	v_cvt_pk_bf16_f32 v66, v70, v71
	v_cvt_pk_bf16_f32 v67, v72, v73
	v_cvt_pk_bf16_f32 v68, v74, v75
	v_cvt_pk_bf16_f32 v69, v76, v77
	v_mfma_f32_32x32x16_bf16 v[224:239], v[106:109], v[118:121], 0
	v_add3_u32 v6, s5, 64, v13
	ds_read_b32 v126, v6

	v_add3_u32 v6, s23, 64, v13
	ds_read_b32 v127, v6

	ds_write_b64 v9, v[66:67] offset:1024
	ds_write_b64 v9, v[68:69] offset:1536
	v_mfma_f32_32x32x16_bf16 v[240:255], v[106:109], v[122:125], 0
	v_permlane16_swap_b32_e32 v62, v64
	v_permlane16_swap_b32_e32 v63, v65
	v_permlane16_swap_b32_e32 v66, v68
	v_permlane16_swap_b32_e32 v67, v69
	s_waitcnt lgkmcnt(12)
	v_mfma_f32_32x32x16_bf16 a[128:143], a[112:115], v[62:65], 0
	ds_read_b64_tr_b16 a[20:21], v1 offset:16896
	ds_read_b64_tr_b16 a[22:23], v1 offset:20992
	ds_read_b64_tr_b16 a[24:25], v1 offset:24576
	ds_read_b64_tr_b16 a[26:27], v1 offset:28672
	ds_read_b64_tr_b16 a[28:29], v1 offset:25088
	ds_read_b64_tr_b16 a[30:31], v1 offset:29184
	ds_read_b64_tr_b16 a[32:33], v1 offset:32768
	ds_read_b64_tr_b16 a[34:35], v1 offset:36864
	v_mfma_f32_32x32x16_bf16 a[144:159], a[112:115], v[66:69], 0
	s_waitcnt lgkmcnt(8)
	s_barrier
	v_mfma_f32_32x32x16_bf16 a[160:175], a[116:119], v[62:65], 0
	ds_read_b64_tr_b16 v[30:31], v4
	ds_read_b64_tr_b16 v[32:33], v4 offset:512
	ds_read_b64_tr_b16 v[34:35], v4 offset:1024
	ds_read_b64_tr_b16 v[36:37], v4 offset:1536
	ds_read_b64_tr_b16 v[38:39], v4 offset:2048
	ds_read_b64_tr_b16 v[40:41], v4 offset:2560
	ds_read_b64_tr_b16 v[42:43], v4 offset:3072
	ds_read_b64_tr_b16 v[44:45], v4 offset:3584
	v_mfma_f32_32x32x16_bf16 a[176:191], a[116:119], v[66:69], 0
	v_mul_f32_e32 v126, 0x3fb8aa3b, v126
	v_mfma_f32_32x32x16_bf16 a[192:207], a[120:123], v[62:65], 0
	ds_read_b64_tr_b16 v[46:47], v4 offset:4096
	ds_read_b64_tr_b16 v[48:49], v4 offset:4608
	ds_read_b64_tr_b16 v[50:51], v4 offset:5120
	ds_read_b64_tr_b16 v[52:53], v4 offset:5632
	ds_read_b64_tr_b16 a[36:37], v1 offset:33280
	ds_read_b64_tr_b16 a[38:39], v1 offset:37376
	ds_read_b64_tr_b16 a[40:41], v1 offset:40960
	ds_read_b64_tr_b16 a[42:43], v1 offset:45056
	v_mfma_f32_32x32x16_bf16 a[208:223], a[120:123], v[66:69], 0
	v_mfma_f32_32x32x16_bf16 a[224:239], a[124:127], v[62:65], 0
	ds_read_b64_tr_b16 v[54:55], v4 offset:6144
	s_add_i32 s0, s67, 64
	ds_read_b64_tr_b16 v[56:57], v4 offset:6656
	s_mul_i32 s0, s0, s26
	ds_read_b64_tr_b16 v[58:59], v4 offset:7168
	s_add_i32 s0, s0, s61
	ds_read_b64_tr_b16 v[60:61], v4 offset:7680
	s_mul_i32 s0, s0, s60
	ds_read_b64_tr_b16 a[44:45], v1 offset:41472
	s_add_i32 s10, s62, 0x4000
	s_ashr_i32 s1, s0, 31
	ds_read_b64_tr_b16 a[46:47], v1 offset:45568
	v_mfma_f32_32x32x16_bf16 a[240:255], a[124:127], v[66:69], 0
	s_waitcnt vmcnt(0) lgkmcnt(6)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[110:113], a[0:3], v[30:33], 0
	ds_read_b64_tr_b16 v[62:63], v1 offset:49152
	s_lshl_b64 s[0:1], s[0:1], 1
	v_add_u32_e32 v6, s10, v17
	ds_read_b64_tr_b16 v[64:65], v1 offset:53248
	s_add_u32 s44, s52, s0
	v_readfirstlane_b32 s0, v6
	v_add_u32_e32 v6, 0x1000, v6
	ds_read_b64_tr_b16 v[66:67], v1 offset:49664
	s_addc_u32 s45, s53, s1
	s_mov_b32 m0, s0
	v_readfirstlane_b32 s0, v6
	ds_read_b64_tr_b16 v[68:69], v1 offset:53760
	v_mfma_f32_16x16x32_bf16 v[110:113], a[8:11], v[34:37], v[110:113]
	buffer_load_dwordx4 v14, s[44:47], 0 offen lds
	s_mov_b32 m0, s0
	s_add_i32 s39, s23, 0x100
	buffer_load_dwordx4 v15, s[44:47], 0 offen lds
	v_mfma_f32_16x16x32_bf16 v[110:113], a[16:19], v[38:41], v[110:113]
	ds_read_b64_tr_b16 v[70:71], v1 offset:57344
	ds_read_b64_tr_b16 v[72:73], v1 offset:61440
	s_add_u32 s40, s8, 0x100
	ds_read_b64_tr_b16 v[74:75], v1 offset:57856
	s_addc_u32 s41, s9, 0
	s_mov_b32 m0, s39
	s_add_i32 s0, s27, 0x1000
	ds_read_b64_tr_b16 v[76:77], v1 offset:61952
	v_mfma_f32_16x16x32_bf16 v[110:113], a[24:27], v[42:45], v[110:113]
	buffer_load_dword v16, s[40:43], 0 offen lds
	v_mfma_f32_16x16x32_bf16 v[110:113], a[32:35], v[46:49], v[110:113]
	v_add_u32_e32 v6, s0, v11
	ds_read_b128 a[112:115], v6
	ds_read_b128 a[116:119], v6 offset:1024
	v_mfma_f32_16x16x32_bf16 v[110:113], a[40:43], v[50:53], v[110:113]
	s_waitcnt lgkmcnt(4)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[110:113], v[62:65], v[54:57], v[110:113]
	ds_read_b128 a[120:123], v6 offset:2048
	ds_read_b128 a[124:127], v6 offset:3072
	v_mfma_f32_16x16x32_bf16 v[110:113], v[70:73], v[58:61], v[110:113]
	v_mfma_f32_16x16x32_bf16 v[114:117], a[4:7], v[30:33], 0
	ds_read_b128 a[0:3], v10
	ds_read_b128 a[4:7], v10 offset:1024
	v_mfma_f32_16x16x32_bf16 v[114:117], a[12:15], v[34:37], v[114:117]
	v_mfma_f32_16x16x32_bf16 v[114:117], a[20:23], v[38:41], v[114:117]
	ds_read_b128 a[8:11], v10 offset:2048
	ds_read_b128 a[12:15], v10 offset:3072
	v_mfma_f32_16x16x32_bf16 v[114:117], a[28:31], v[42:45], v[114:117]
	v_mul_f32_e32 v110, 0x3db504f3, v110
	v_mul_f32_e32 v111, 0x3db504f3, v111
	v_mfma_f32_16x16x32_bf16 v[114:117], a[36:39], v[46:49], v[114:117]
	ds_read_b128 a[16:19], v10 offset:4096
	ds_read_b128 a[20:23], v10 offset:5120
	v_mfma_f32_16x16x32_bf16 v[114:117], a[44:47], v[50:53], v[114:117]
	v_mul_f32_e32 v112, 0x3db504f3, v112
	v_mul_f32_e32 v113, 0x3db504f3, v113
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_16x16x32_bf16 v[114:117], v[66:69], v[54:57], v[114:117]
	ds_read_b128 a[24:27], v10 offset:6144
	ds_read_b128 a[28:31], v10 offset:7168
	v_mfma_f32_16x16x32_bf16 v[114:117], v[74:77], v[58:61], v[114:117]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x32_bf16 v[46:49], a[112:115], a[0:3], 0
	ds_read_b128 a[32:35], v10 offset:8192
	ds_read_b128 a[36:39], v10 offset:9216
	v_mfma_f32_16x16x32_bf16 v[46:49], a[116:119], a[4:7], v[46:49]
	v_mfma_f32_16x16x32_bf16 v[46:49], a[120:123], a[8:11], v[46:49]
	ds_read_b128 a[40:43], v10 offset:10240
	ds_read_b128 a[44:47], v10 offset:11264
	v_mfma_f32_16x16x32_bf16 v[46:49], a[124:127], a[12:15], v[46:49]
	v_mul_f32_e32 v114, 0x3db504f3, v114
	v_mul_f32_e32 v115, 0x3db504f3, v115
	v_mfma_f32_16x16x32_bf16 v[50:53], a[112:115], a[16:19], 0
	ds_read_b128 v[62:65], v10 offset:12288
	ds_read_b128 v[66:69], v10 offset:13312
	v_mfma_f32_16x16x32_bf16 v[50:53], a[116:119], a[20:23], v[50:53]
	v_mul_f32_e32 v116, 0x3db504f3, v116
	v_mul_f32_e32 v117, 0x3db504f3, v117
	v_mfma_f32_16x16x32_bf16 v[50:53], a[120:123], a[24:27], v[50:53]
	ds_read_b128 v[70:73], v10 offset:14336
	s_add_i32 s1, s62, 0x1000
	ds_read_b128 v[74:77], v10 offset:15360
	v_mfma_f32_16x16x32_bf16 v[50:53], a[124:127], a[28:31], v[50:53]
	v_mul_f32_e32 v46, 0x3e0293ee, v46
	v_mul_f32_e32 v47, 0x3e0293ee, v47
	v_mul_f32_e32 v48, 0x3e0293ee, v48
	v_mul_f32_e32 v49, 0x3e0293ee, v49
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_bf16 v[54:57], a[112:115], a[32:35], 0
	v_add_u32_e32 v6, s1, v11
	ds_read_b128 v[78:81], v6
	ds_read_b128 v[82:85], v6 offset:1024
	v_mfma_f32_16x16x32_bf16 v[54:57], a[116:119], a[36:39], v[54:57]
	v_subrev_f32_dpp v46, v126, v46 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v47, v126, v47 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v48, v126, v48 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v49, v126, v49 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_bf16 v[54:57], a[120:123], a[40:43], v[54:57]
	ds_read_b128 v[86:89], v6 offset:2048
	ds_read_b128 v[90:93], v6 offset:3072
	v_mfma_f32_16x16x32_bf16 v[54:57], a[124:127], a[44:47], v[54:57]
	v_mul_f32_e32 v50, 0x3e0293ee, v50
	v_mul_f32_e32 v51, 0x3e0293ee, v51
	v_mul_f32_e32 v52, 0x3e0293ee, v52
	v_mul_f32_e32 v53, 0x3e0293ee, v53
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_bf16 v[58:61], a[112:115], v[62:65], 0
	v_add_u32_e32 v6, s1, v12
	ds_read_b64_tr_b16 v[94:95], v6
	ds_read_b64_tr_b16 v[96:97], v6 offset:256
	ds_read_b64_tr_b16 v[98:99], v6 offset:1024
	ds_read_b64_tr_b16 v[100:101], v6 offset:1280
	v_mfma_f32_16x16x32_bf16 v[58:61], a[116:119], v[66:69], v[58:61]
	v_subrev_f32_dpp v50, v126, v50 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v51, v126, v51 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v52, v126, v52 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v53, v126, v53 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_16x16x32_bf16 v[58:61], a[120:123], v[70:73], v[58:61]
	ds_read_b64_tr_b16 v[102:103], v6 offset:2048
	ds_read_b64_tr_b16 v[104:105], v6 offset:2304
	ds_read_b64_tr_b16 v[106:107], v6 offset:3072
	ds_read_b64_tr_b16 v[108:109], v6 offset:3328
	v_mfma_f32_16x16x32_bf16 v[58:61], a[124:127], v[74:77], v[58:61]
	v_mul_f32_e32 v54, 0x3e0293ee, v54
	v_mul_f32_e32 v55, 0x3e0293ee, v55
	v_mul_f32_e32 v56, 0x3e0293ee, v56
	v_mul_f32_e32 v57, 0x3e0293ee, v57
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_16x16x32_bf16 v[62:65], v[78:81], a[48:51], 0
	v_subrev_f32_dpp v54, v126, v54 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v55, v126, v55 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v56, v126, v56 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v57, v126, v57 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	v_mfma_f32_16x16x32_bf16 v[62:65], v[82:85], a[52:55], v[62:65]
	v_exp_f32_e32 v46, v46
	v_exp_f32_e32 v47, v47
	v_exp_f32_e32 v48, v48
	v_exp_f32_e32 v49, v49
	v_mfma_f32_16x16x32_bf16 v[62:65], v[86:89], a[56:59], v[62:65]
	v_add_u32_e32 v6, s0, v12
	ds_read_b64_tr_b16 a[112:113], v6
	ds_read_b64_tr_b16 a[114:115], v6 offset:256
	v_mfma_f32_16x16x32_bf16 v[62:65], v[90:93], a[60:63], v[62:65]
	v_exp_f32_e32 v50, v50
	v_exp_f32_e32 v51, v51
	v_exp_f32_e32 v52, v52
	v_exp_f32_e32 v53, v53
	v_mfma_f32_16x16x32_bf16 v[66:69], v[78:81], a[64:67], 0
	ds_read_b64_tr_b16 a[116:117], v6 offset:1024
	ds_read_b64_tr_b16 a[118:119], v6 offset:1280
	v_mfma_f32_16x16x32_bf16 v[66:69], v[82:85], a[68:71], v[66:69]
	v_mul_f32_e32 v58, 0x3e0293ee, v58
	v_mul_f32_e32 v59, 0x3e0293ee, v59
	v_mul_f32_e32 v60, 0x3e0293ee, v60
	v_mul_f32_e32 v61, 0x3e0293ee, v61
	v_mfma_f32_16x16x32_bf16 v[66:69], v[86:89], a[72:75], v[66:69]
	v_subrev_f32_dpp v58, v126, v58 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v59, v126, v59 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v60, v126, v60 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v61, v126, v61 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	v_mfma_f32_16x16x32_bf16 v[66:69], v[90:93], a[76:79], v[66:69]
	v_cvt_pk_bf16_f32 v118, v46, v47
	v_cvt_pk_bf16_f32 v119, v48, v49
	v_mfma_f32_16x16x32_bf16 v[70:73], v[78:81], a[80:83], 0
	v_exp_f32_e32 v54, v54
	v_exp_f32_e32 v55, v55
	v_exp_f32_e32 v56, v56
	v_exp_f32_e32 v57, v57
	v_mfma_f32_16x16x32_bf16 v[70:73], v[82:85], a[84:87], v[70:73]
	v_cvt_pk_bf16_f32 v120, v50, v51
	v_cvt_pk_bf16_f32 v121, v52, v53
	v_mfma_f32_16x16x32_bf16 v[70:73], v[86:89], a[88:91], v[70:73]
	ds_read_b64_tr_b16 a[120:121], v6 offset:2048
	ds_read_b64_tr_b16 a[122:123], v6 offset:2304
	v_mfma_f32_16x16x32_bf16 v[70:73], v[90:93], a[92:95], v[70:73]
	v_exp_f32_e32 v58, v58
	v_exp_f32_e32 v59, v59
	v_exp_f32_e32 v60, v60
	v_exp_f32_e32 v61, v61
	v_mfma_f32_16x16x32_bf16 v[74:77], v[78:81], a[96:99], 0
	ds_read_b64_tr_b16 a[124:125], v6 offset:3072
	ds_read_b64_tr_b16 a[126:127], v6 offset:3328
	v_mfma_f32_16x16x32_bf16 v[74:77], v[82:85], a[100:103], v[74:77]
	v_cvt_pk_bf16_f32 v122, v54, v55
	v_cvt_pk_bf16_f32 v123, v56, v57
	v_cvt_pk_bf16_f32 v124, v58, v59
	v_cvt_pk_bf16_f32 v125, v60, v61
	v_mfma_f32_16x16x32_bf16 v[74:77], v[86:89], a[104:107], v[74:77]
	v_permlane16_swap_b32_e32 v118, v120
	v_permlane16_swap_b32_e32 v119, v121
	v_permlane16_swap_b32_e32 v122, v124
	v_permlane16_swap_b32_e32 v123, v125
	v_mfma_f32_16x16x32_bf16 v[74:77], v[90:93], a[108:111], v[74:77]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x16_bf16 v[128:143], v[94:97], v[118:121], v[128:143]
	ds_read_b64_tr_b16 a[0:1], v1
	ds_read_b64_tr_b16 a[2:3], v1 offset:4096
	ds_read_b64_tr_b16 a[4:5], v1 offset:512
	ds_read_b64_tr_b16 a[6:7], v1 offset:4608
	v_mfma_f32_32x32x16_bf16 v[144:159], v[94:97], v[122:125], v[144:159]
	v_subrev_f32_dpp v62, v127, v62 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v63, v127, v63 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v64, v127, v64 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v65, v127, v65 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v66, v127, v66 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v67, v127, v67 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v68, v127, v68 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v69, v127, v69 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	v_mfma_f32_32x32x16_bf16 v[160:175], v[98:101], v[118:121], v[160:175]
	ds_read_b64_tr_b16 a[8:9], v1 offset:8192
	ds_read_b64_tr_b16 a[10:11], v1 offset:12288
	ds_read_b64_tr_b16 a[12:13], v1 offset:8704
	ds_read_b64_tr_b16 a[14:15], v1 offset:12800
	v_mfma_f32_32x32x16_bf16 v[176:191], v[98:101], v[122:125], v[176:191]
	v_mul_f32_e32 v62, v46, v62
	v_mul_f32_e32 v63, v47, v63
	v_mul_f32_e32 v64, v48, v64
	v_mul_f32_e32 v65, v49, v65
	v_mul_f32_e32 v66, v50, v66
	v_mul_f32_e32 v67, v51, v67
	v_mul_f32_e32 v68, v52, v68
	v_mul_f32_e32 v69, v53, v69
	v_cvt_pk_bf16_f32 v62, v62, v63
	v_cvt_pk_bf16_f32 v63, v64, v65
	v_cvt_pk_bf16_f32 v64, v66, v67
	v_cvt_pk_bf16_f32 v65, v68, v69
	v_subrev_f32_dpp v70, v127, v70 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v71, v127, v71 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v72, v127, v72 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v73, v127, v73 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	v_mfma_f32_32x32x16_bf16 v[192:207], v[102:105], v[118:121], v[192:207]
	ds_read_b64_tr_b16 a[16:17], v1 offset:16384
	ds_read_b64_tr_b16 a[18:19], v1 offset:20480
	ds_write_b64 v9, v[62:63]
	s_add_i32 s0, s5, 0x80
	ds_write_b64 v9, v[64:65] offset:512
	v_mfma_f32_32x32x16_bf16 v[208:223], v[102:105], v[122:125], v[208:223]
	v_subrev_f32_dpp v74, v127, v74 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v75, v127, v75 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v76, v127, v76 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v77, v127, v77 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	v_mul_f32_e32 v70, v54, v70
	v_mul_f32_e32 v71, v55, v71
	v_mul_f32_e32 v72, v56, v72
	v_mul_f32_e32 v73, v57, v73
	v_mul_f32_e32 v74, v58, v74
	v_mul_f32_e32 v75, v59, v75
	v_mul_f32_e32 v76, v60, v76
	v_mul_f32_e32 v77, v61, v77
	v_cvt_pk_bf16_f32 v66, v70, v71
	v_cvt_pk_bf16_f32 v67, v72, v73
	v_cvt_pk_bf16_f32 v68, v74, v75
	v_cvt_pk_bf16_f32 v69, v76, v77
	v_mfma_f32_32x32x16_bf16 v[224:239], v[106:109], v[118:121], v[224:239]
	v_add_u32_e32 v6, s0, v13
	ds_read_b32 v126, v6

	s_add_i32 s0, s23, 0x80
	v_add_u32_e32 v6, s0, v13
	ds_read_b32 v127, v6

	ds_write_b64 v9, v[66:67] offset:1024
	ds_write_b64 v9, v[68:69] offset:1536
	v_mfma_f32_32x32x16_bf16 v[240:255], v[106:109], v[122:125], v[240:255]
	v_permlane16_swap_b32_e32 v62, v64
	v_permlane16_swap_b32_e32 v63, v65
	v_permlane16_swap_b32_e32 v66, v68
	v_permlane16_swap_b32_e32 v67, v69
	s_waitcnt lgkmcnt(12)
	v_mfma_f32_32x32x16_bf16 a[128:143], a[112:115], v[62:65], a[128:143]
	ds_read_b64_tr_b16 a[20:21], v1 offset:16896
	ds_read_b64_tr_b16 a[22:23], v1 offset:20992
	ds_read_b64_tr_b16 a[24:25], v1 offset:24576
	ds_read_b64_tr_b16 a[26:27], v1 offset:28672
	ds_read_b64_tr_b16 a[28:29], v1 offset:25088
	ds_read_b64_tr_b16 a[30:31], v1 offset:29184
	ds_read_b64_tr_b16 a[32:33], v1 offset:32768
	ds_read_b64_tr_b16 a[34:35], v1 offset:36864
	v_mfma_f32_32x32x16_bf16 a[144:159], a[112:115], v[66:69], a[144:159]
	s_waitcnt lgkmcnt(8)
	s_barrier
	v_mfma_f32_32x32x16_bf16 a[160:175], a[116:119], v[62:65], a[160:175]
	ds_read_b64_tr_b16 v[30:31], v4
	s_mul_i32 s15, s4, s36
	ds_read_b64_tr_b16 v[32:33], v4 offset:512
	s_add_i32 s15, s15, s61
	ds_read_b64_tr_b16 v[34:35], v4 offset:1024
	s_mul_i32 s0, s15, s38
	ds_read_b64_tr_b16 v[36:37], v4 offset:1536
	s_mul_i32 s0, s0, s3
	ds_read_b64_tr_b16 v[38:39], v4 offset:2048
	s_ashr_i32 s1, s0, 31
	ds_read_b64_tr_b16 v[40:41], v4 offset:2560
	s_lshl_b64 s[8:9], s[0:1], 1
	ds_read_b64_tr_b16 v[42:43], v4 offset:3072
	s_add_u32 s8, s6, s8
	ds_read_b64_tr_b16 v[44:45], v4 offset:3584
	v_mfma_f32_32x32x16_bf16 a[176:191], a[116:119], v[66:69], a[176:191]
	v_mul_f32_e32 v126, 0x3fb8aa3b, v126
	s_addc_u32 s9, s7, s9
	s_lshl_b32 s10, s3, 5
	v_or_b32_e32 v5, v5, v16
	v_cvt_pk_bf16_f32 v110, v110, v111
	v_cvt_pk_bf16_f32 v111, v112, v113
	buffer_atomic_pk_add_bf16 v110, v5, s[8:11], 0 offen
	v_or_b32_e32 v6, 0x100, v5
	buffer_atomic_pk_add_bf16 v111, v6, s[8:11], 0 offen
	v_mfma_f32_32x32x16_bf16 a[192:207], a[120:123], v[62:65], a[192:207]
	ds_read_b64_tr_b16 v[46:47], v4 offset:4096
	ds_read_b64_tr_b16 v[48:49], v4 offset:4608
	ds_read_b64_tr_b16 v[50:51], v4 offset:5120
	ds_read_b64_tr_b16 v[52:53], v4 offset:5632
	ds_read_b64_tr_b16 a[36:37], v1 offset:33280
	ds_read_b64_tr_b16 a[38:39], v1 offset:37376
	ds_read_b64_tr_b16 a[40:41], v1 offset:40960
	ds_read_b64_tr_b16 a[42:43], v1 offset:45056
	v_mfma_f32_32x32x16_bf16 a[208:223], a[120:123], v[66:69], a[208:223]
	v_cvt_pk_bf16_f32 v114, v114, v115
	v_cvt_pk_bf16_f32 v115, v116, v117
	v_or_b32_e32 v7, 0x200, v5
	buffer_atomic_pk_add_bf16 v114, v7, s[8:11], 0 offen
	v_or_b32_e32 v8, 0x300, v5
	buffer_atomic_pk_add_bf16 v115, v8, s[8:11], 0 offen
	v_mfma_f32_32x32x16_bf16 a[224:239], a[124:127], v[62:65], a[224:239]
	ds_read_b64_tr_b16 v[54:55], v4 offset:6144
	s_add_i32 s8, s66, 0x60
	ds_read_b64_tr_b16 v[56:57], v4 offset:6656
	s_mul_i32 s8, s8, s22
	ds_read_b64_tr_b16 v[58:59], v4 offset:7168
	s_add_i32 s8, s8, s61
	ds_read_b64_tr_b16 v[60:61], v4 offset:7680
	s_mul_i32 s8, s8, s33
	ds_read_b64_tr_b16 a[44:45], v1 offset:41472
	s_add_i32 s1, s27, 0x6000
	s_ashr_i32 s9, s8, 31
	ds_read_b64_tr_b16 a[46:47], v1 offset:45568
	v_mfma_f32_32x32x16_bf16 a[240:255], a[124:127], v[66:69], a[240:255]
	s_waitcnt vmcnt(4) lgkmcnt(6)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[110:113], a[0:3], v[30:33], 0
	ds_read_b64_tr_b16 v[62:63], v1 offset:49152
	s_lshl_b64 s[8:9], s[8:9], 1
	v_add_u32_e32 v18, s1, v17
	ds_read_b64_tr_b16 v[64:65], v1 offset:53248
	s_add_u32 s28, s50, s8
	v_readfirstlane_b32 s1, v18
	v_add_u32_e32 v18, 0x1000, v18
	ds_read_b64_tr_b16 v[66:67], v1 offset:49664
	s_addc_u32 s29, s51, s9
	s_mov_b32 m0, s1
	v_readfirstlane_b32 s1, v18
	ds_read_b64_tr_b16 v[68:69], v1 offset:53760
	v_mfma_f32_16x16x32_bf16 v[110:113], a[8:11], v[34:37], v[110:113]
	buffer_load_dwordx4 v14, s[28:31], 0 offen lds
	s_mov_b32 m0, s1
	v_add_u32_e32 v18, s70, v11
	buffer_load_dwordx4 v15, s[28:31], 0 offen lds
	v_mfma_f32_16x16x32_bf16 v[110:113], a[16:19], v[38:41], v[110:113]
	ds_read_b64_tr_b16 v[70:71], v1 offset:57344
	ds_read_b64_tr_b16 v[72:73], v1 offset:61440
	ds_read_b64_tr_b16 v[74:75], v1 offset:57856
	ds_read_b64_tr_b16 v[76:77], v1 offset:61952
	v_mfma_f32_16x16x32_bf16 v[110:113], a[24:27], v[42:45], v[110:113]
	v_mfma_f32_16x16x32_bf16 v[110:113], a[32:35], v[46:49], v[110:113]
	ds_read_b128 a[112:115], v18
	ds_read_b128 a[116:119], v18 offset:1024
	v_mfma_f32_16x16x32_bf16 v[110:113], a[40:43], v[50:53], v[110:113]
	s_waitcnt lgkmcnt(4)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[110:113], v[62:65], v[54:57], v[110:113]
	ds_read_b128 a[120:123], v18 offset:2048
	ds_read_b128 a[124:127], v18 offset:3072
	v_mfma_f32_16x16x32_bf16 v[110:113], v[70:73], v[58:61], v[110:113]
	v_mfma_f32_16x16x32_bf16 v[114:117], a[4:7], v[30:33], 0
	ds_read_b128 a[0:3], v10
	ds_read_b128 a[4:7], v10 offset:1024
	v_mfma_f32_16x16x32_bf16 v[114:117], a[12:15], v[34:37], v[114:117]
	v_mfma_f32_16x16x32_bf16 v[114:117], a[20:23], v[38:41], v[114:117]
	ds_read_b128 a[8:11], v10 offset:2048
	ds_read_b128 a[12:15], v10 offset:3072
	v_mfma_f32_16x16x32_bf16 v[114:117], a[28:31], v[42:45], v[114:117]
	v_mul_f32_e32 v110, 0x3db504f3, v110
	v_mul_f32_e32 v111, 0x3db504f3, v111
	v_mfma_f32_16x16x32_bf16 v[114:117], a[36:39], v[46:49], v[114:117]
	ds_read_b128 a[16:19], v10 offset:4096
	ds_read_b128 a[20:23], v10 offset:5120
	v_mfma_f32_16x16x32_bf16 v[114:117], a[44:47], v[50:53], v[114:117]
	v_mul_f32_e32 v112, 0x3db504f3, v112
	v_mul_f32_e32 v113, 0x3db504f3, v113
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_16x16x32_bf16 v[114:117], v[66:69], v[54:57], v[114:117]
	ds_read_b128 a[24:27], v10 offset:6144
	ds_read_b128 a[28:31], v10 offset:7168
	v_mfma_f32_16x16x32_bf16 v[114:117], v[74:77], v[58:61], v[114:117]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x32_bf16 v[46:49], a[112:115], a[0:3], 0
	ds_read_b128 a[32:35], v10 offset:8192
	ds_read_b128 a[36:39], v10 offset:9216
	v_mfma_f32_16x16x32_bf16 v[46:49], a[116:119], a[4:7], v[46:49]
	v_mfma_f32_16x16x32_bf16 v[46:49], a[120:123], a[8:11], v[46:49]
	ds_read_b128 a[40:43], v10 offset:10240
	ds_read_b128 a[44:47], v10 offset:11264
	v_mfma_f32_16x16x32_bf16 v[46:49], a[124:127], a[12:15], v[46:49]
	v_mul_f32_e32 v114, 0x3db504f3, v114
	v_mul_f32_e32 v115, 0x3db504f3, v115
	v_mfma_f32_16x16x32_bf16 v[50:53], a[112:115], a[16:19], 0
	ds_read_b128 v[62:65], v10 offset:12288
	ds_read_b128 v[66:69], v10 offset:13312
	v_mfma_f32_16x16x32_bf16 v[50:53], a[116:119], a[20:23], v[50:53]
	v_mul_f32_e32 v116, 0x3db504f3, v116
	v_mul_f32_e32 v117, 0x3db504f3, v117
	v_mfma_f32_16x16x32_bf16 v[50:53], a[120:123], a[24:27], v[50:53]
	ds_read_b128 v[70:73], v10 offset:14336
	ds_read_b128 v[74:77], v10 offset:15360
	v_mfma_f32_16x16x32_bf16 v[50:53], a[124:127], a[28:31], v[50:53]
	v_mul_f32_e32 v46, 0x3e0293ee, v46
	v_mul_f32_e32 v47, 0x3e0293ee, v47
	v_mul_f32_e32 v48, 0x3e0293ee, v48
	v_mul_f32_e32 v49, 0x3e0293ee, v49
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_bf16 v[54:57], a[112:115], a[32:35], 0
	v_add_u32_e32 v18, s71, v11
	ds_read_b128 v[78:81], v18
	ds_read_b128 v[82:85], v18 offset:1024
	v_mfma_f32_16x16x32_bf16 v[54:57], a[116:119], a[36:39], v[54:57]
	v_subrev_f32_dpp v46, v126, v46 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v47, v126, v47 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v48, v126, v48 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v49, v126, v49 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_bf16 v[54:57], a[120:123], a[40:43], v[54:57]
	ds_read_b128 v[86:89], v18 offset:2048
	ds_read_b128 v[90:93], v18 offset:3072
	v_mfma_f32_16x16x32_bf16 v[54:57], a[124:127], a[44:47], v[54:57]
	v_mul_f32_e32 v50, 0x3e0293ee, v50
	v_mul_f32_e32 v51, 0x3e0293ee, v51
	v_mul_f32_e32 v52, 0x3e0293ee, v52
	v_mul_f32_e32 v53, 0x3e0293ee, v53
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_bf16 v[58:61], a[112:115], v[62:65], 0
	v_add_u32_e32 v18, s71, v12
	ds_read_b64_tr_b16 v[94:95], v18
	ds_read_b64_tr_b16 v[96:97], v18 offset:256
	ds_read_b64_tr_b16 v[98:99], v18 offset:1024
	ds_read_b64_tr_b16 v[100:101], v18 offset:1280
	v_mfma_f32_16x16x32_bf16 v[58:61], a[116:119], v[66:69], v[58:61]
	v_subrev_f32_dpp v50, v126, v50 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v51, v126, v51 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v52, v126, v52 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v53, v126, v53 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_16x16x32_bf16 v[58:61], a[120:123], v[70:73], v[58:61]
	ds_read_b64_tr_b16 v[102:103], v18 offset:2048
	ds_read_b64_tr_b16 v[104:105], v18 offset:2304
	ds_read_b64_tr_b16 v[106:107], v18 offset:3072
	ds_read_b64_tr_b16 v[108:109], v18 offset:3328
	v_mfma_f32_16x16x32_bf16 v[58:61], a[124:127], v[74:77], v[58:61]
	v_mul_f32_e32 v54, 0x3e0293ee, v54
	v_mul_f32_e32 v55, 0x3e0293ee, v55
	v_mul_f32_e32 v56, 0x3e0293ee, v56
	v_mul_f32_e32 v57, 0x3e0293ee, v57
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_16x16x32_bf16 v[62:65], v[78:81], a[48:51], 0
	v_subrev_f32_dpp v54, v126, v54 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v55, v126, v55 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v56, v126, v56 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v57, v126, v57 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	v_mfma_f32_16x16x32_bf16 v[62:65], v[82:85], a[52:55], v[62:65]
	v_exp_f32_e32 v46, v46
	v_exp_f32_e32 v47, v47
	v_exp_f32_e32 v48, v48
	v_exp_f32_e32 v49, v49
	v_mfma_f32_16x16x32_bf16 v[62:65], v[86:89], a[56:59], v[62:65]
	v_add_u32_e32 v18, s70, v12
	ds_read_b64_tr_b16 a[112:113], v18
	ds_read_b64_tr_b16 a[114:115], v18 offset:256
	v_mfma_f32_16x16x32_bf16 v[62:65], v[90:93], a[60:63], v[62:65]
	v_exp_f32_e32 v50, v50
	v_exp_f32_e32 v51, v51
	v_exp_f32_e32 v52, v52
	v_exp_f32_e32 v53, v53
	v_mfma_f32_16x16x32_bf16 v[66:69], v[78:81], a[64:67], 0
	ds_read_b64_tr_b16 a[116:117], v18 offset:1024
	ds_read_b64_tr_b16 a[118:119], v18 offset:1280
	v_mfma_f32_16x16x32_bf16 v[66:69], v[82:85], a[68:71], v[66:69]
	v_mul_f32_e32 v58, 0x3e0293ee, v58
	v_mul_f32_e32 v59, 0x3e0293ee, v59
	v_mul_f32_e32 v60, 0x3e0293ee, v60
	v_mul_f32_e32 v61, 0x3e0293ee, v61
	v_mfma_f32_16x16x32_bf16 v[66:69], v[86:89], a[72:75], v[66:69]
	v_subrev_f32_dpp v58, v126, v58 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v59, v126, v59 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v60, v126, v60 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v61, v126, v61 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	v_mfma_f32_16x16x32_bf16 v[66:69], v[90:93], a[76:79], v[66:69]
	v_cvt_pk_bf16_f32 v118, v46, v47
	v_cvt_pk_bf16_f32 v119, v48, v49
	v_mfma_f32_16x16x32_bf16 v[70:73], v[78:81], a[80:83], 0
	v_exp_f32_e32 v54, v54
	v_exp_f32_e32 v55, v55
	v_exp_f32_e32 v56, v56
	v_exp_f32_e32 v57, v57
	v_mfma_f32_16x16x32_bf16 v[70:73], v[82:85], a[84:87], v[70:73]
	v_cvt_pk_bf16_f32 v120, v50, v51
	v_cvt_pk_bf16_f32 v121, v52, v53
	v_mfma_f32_16x16x32_bf16 v[70:73], v[86:89], a[88:91], v[70:73]
	ds_read_b64_tr_b16 a[120:121], v18 offset:2048
	ds_read_b64_tr_b16 a[122:123], v18 offset:2304
	v_mfma_f32_16x16x32_bf16 v[70:73], v[90:93], a[92:95], v[70:73]
	v_exp_f32_e32 v58, v58
	v_exp_f32_e32 v59, v59
	v_exp_f32_e32 v60, v60
	v_exp_f32_e32 v61, v61
	v_mfma_f32_16x16x32_bf16 v[74:77], v[78:81], a[96:99], 0
	ds_read_b64_tr_b16 a[124:125], v18 offset:3072
	ds_read_b64_tr_b16 a[126:127], v18 offset:3328
	v_mfma_f32_16x16x32_bf16 v[74:77], v[82:85], a[100:103], v[74:77]
	v_cvt_pk_bf16_f32 v122, v54, v55
	v_cvt_pk_bf16_f32 v123, v56, v57
	v_cvt_pk_bf16_f32 v124, v58, v59
	v_cvt_pk_bf16_f32 v125, v60, v61
	v_mfma_f32_16x16x32_bf16 v[74:77], v[86:89], a[104:107], v[74:77]
	v_permlane16_swap_b32_e32 v118, v120
	v_permlane16_swap_b32_e32 v119, v121
	v_permlane16_swap_b32_e32 v122, v124
	v_permlane16_swap_b32_e32 v123, v125
	v_mfma_f32_16x16x32_bf16 v[74:77], v[90:93], a[108:111], v[74:77]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x16_bf16 v[128:143], v[94:97], v[118:121], v[128:143]
	ds_read_b64_tr_b16 a[0:1], v1
	ds_read_b64_tr_b16 a[2:3], v1 offset:4096
	ds_read_b64_tr_b16 a[4:5], v1 offset:512
	ds_read_b64_tr_b16 a[6:7], v1 offset:4608
	v_mfma_f32_32x32x16_bf16 v[144:159], v[94:97], v[122:125], v[144:159]
	v_subrev_f32_dpp v62, v127, v62 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v63, v127, v63 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v64, v127, v64 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v65, v127, v65 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v66, v127, v66 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v67, v127, v67 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v68, v127, v68 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v69, v127, v69 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	v_mfma_f32_32x32x16_bf16 v[160:175], v[98:101], v[118:121], v[160:175]
	ds_read_b64_tr_b16 a[8:9], v1 offset:8192
	ds_read_b64_tr_b16 a[10:11], v1 offset:12288
	ds_read_b64_tr_b16 a[12:13], v1 offset:8704
	ds_read_b64_tr_b16 a[14:15], v1 offset:12800
	v_mfma_f32_32x32x16_bf16 v[176:191], v[98:101], v[122:125], v[176:191]
	v_mul_f32_e32 v62, v46, v62
	v_mul_f32_e32 v63, v47, v63
	v_mul_f32_e32 v64, v48, v64
	v_mul_f32_e32 v65, v49, v65
	v_mul_f32_e32 v66, v50, v66
	v_mul_f32_e32 v67, v51, v67
	v_mul_f32_e32 v68, v52, v68
	v_mul_f32_e32 v69, v53, v69
	v_cvt_pk_bf16_f32 v62, v62, v63
	v_cvt_pk_bf16_f32 v63, v64, v65
	v_cvt_pk_bf16_f32 v64, v66, v67
	v_cvt_pk_bf16_f32 v65, v68, v69
	v_subrev_f32_dpp v70, v127, v70 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v71, v127, v71 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v72, v127, v72 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v73, v127, v73 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	v_mfma_f32_32x32x16_bf16 v[192:207], v[102:105], v[118:121], v[192:207]
	ds_read_b64_tr_b16 a[16:17], v1 offset:16384
	ds_read_b64_tr_b16 a[18:19], v1 offset:20480
	ds_write_b64 v9, v[62:63]
	s_add_i32 s1, s5, 0xc0
	ds_write_b64 v9, v[64:65] offset:512
	v_mfma_f32_32x32x16_bf16 v[208:223], v[102:105], v[122:125], v[208:223]
	v_subrev_f32_dpp v74, v127, v74 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v75, v127, v75 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v76, v127, v76 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v77, v127, v77 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	v_mul_f32_e32 v70, v54, v70
	v_mul_f32_e32 v71, v55, v71
	v_mul_f32_e32 v72, v56, v72
	v_mul_f32_e32 v73, v57, v73
	v_mul_f32_e32 v74, v58, v74
	v_mul_f32_e32 v75, v59, v75
	v_mul_f32_e32 v76, v60, v76
	v_mul_f32_e32 v77, v61, v77
	v_cvt_pk_bf16_f32 v66, v70, v71
	v_cvt_pk_bf16_f32 v67, v72, v73
	v_cvt_pk_bf16_f32 v68, v74, v75
	v_cvt_pk_bf16_f32 v69, v76, v77
	v_mfma_f32_32x32x16_bf16 v[224:239], v[106:109], v[118:121], v[224:239]
	v_add_u32_e32 v18, s1, v13
	ds_read_b32 v126, v18

	s_add_i32 s1, s23, 0xc0
	v_add_u32_e32 v18, s1, v13
	ds_read_b32 v127, v18

	ds_write_b64 v9, v[66:67] offset:1024
	ds_write_b64 v9, v[68:69] offset:1536
	v_mfma_f32_32x32x16_bf16 v[240:255], v[106:109], v[122:125], v[240:255]
	v_permlane16_swap_b32_e32 v62, v64
	v_permlane16_swap_b32_e32 v63, v65
	v_permlane16_swap_b32_e32 v66, v68
	v_permlane16_swap_b32_e32 v67, v69
	s_waitcnt lgkmcnt(12)
	v_mfma_f32_32x32x16_bf16 a[128:143], a[112:115], v[62:65], a[128:143]
	ds_read_b64_tr_b16 a[20:21], v1 offset:16896
	ds_read_b64_tr_b16 a[22:23], v1 offset:20992
	ds_read_b64_tr_b16 a[24:25], v1 offset:24576
	ds_read_b64_tr_b16 a[26:27], v1 offset:28672
	ds_read_b64_tr_b16 a[28:29], v1 offset:25088
	ds_read_b64_tr_b16 a[30:31], v1 offset:29184
	ds_read_b64_tr_b16 a[32:33], v1 offset:32768
	ds_read_b64_tr_b16 a[34:35], v1 offset:36864
	v_mfma_f32_32x32x16_bf16 a[144:159], a[112:115], v[66:69], a[144:159]
	s_waitcnt lgkmcnt(8)
	s_barrier
	v_mfma_f32_32x32x16_bf16 a[160:175], a[116:119], v[62:65], a[160:175]
	ds_read_b64_tr_b16 v[30:31], v4
	ds_read_b64_tr_b16 v[32:33], v4 offset:512
	ds_read_b64_tr_b16 v[34:35], v4 offset:1024
	s_lshl_b32 s13, s3, 4
	ds_read_b64_tr_b16 v[36:37], v4 offset:1536
	s_add_i32 s0, s0, s13
	ds_read_b64_tr_b16 v[38:39], v4 offset:2048
	s_ashr_i32 s1, s0, 31
	ds_read_b64_tr_b16 v[40:41], v4 offset:2560
	s_lshl_b64 s[8:9], s[0:1], 1
	ds_read_b64_tr_b16 v[42:43], v4 offset:3072
	s_add_u32 s8, s6, s8
	ds_read_b64_tr_b16 v[44:45], v4 offset:3584
	v_mfma_f32_32x32x16_bf16 a[176:191], a[116:119], v[66:69], a[176:191]
	v_mul_f32_e32 v126, 0x3fb8aa3b, v126
	s_addc_u32 s9, s7, s9
	v_cvt_pk_bf16_f32 v110, v110, v111
	v_cvt_pk_bf16_f32 v111, v112, v113
	buffer_atomic_pk_add_bf16 v110, v5, s[8:11], 0 offen
	buffer_atomic_pk_add_bf16 v111, v6, s[8:11], 0 offen
	v_mfma_f32_32x32x16_bf16 a[192:207], a[120:123], v[62:65], a[192:207]
	ds_read_b64_tr_b16 v[46:47], v4 offset:4096
	ds_read_b64_tr_b16 v[48:49], v4 offset:4608
	ds_read_b64_tr_b16 v[50:51], v4 offset:5120
	ds_read_b64_tr_b16 v[52:53], v4 offset:5632
	ds_read_b64_tr_b16 a[36:37], v1 offset:33280
	ds_read_b64_tr_b16 a[38:39], v1 offset:37376
	ds_read_b64_tr_b16 a[40:41], v1 offset:40960
	ds_read_b64_tr_b16 a[42:43], v1 offset:45056
	v_mfma_f32_32x32x16_bf16 a[208:223], a[120:123], v[66:69], a[208:223]
	v_cvt_pk_bf16_f32 v114, v114, v115
	v_cvt_pk_bf16_f32 v115, v116, v117
	buffer_atomic_pk_add_bf16 v114, v7, s[8:11], 0 offen
	buffer_atomic_pk_add_bf16 v115, v8, s[8:11], 0 offen
	v_mfma_f32_32x32x16_bf16 a[224:239], a[124:127], v[62:65], a[224:239]
	ds_read_b64_tr_b16 v[54:55], v4 offset:6144
	s_add_i32 s8, s67, 0x60
	ds_read_b64_tr_b16 v[56:57], v4 offset:6656
	s_mul_i32 s8, s8, s26
	ds_read_b64_tr_b16 v[58:59], v4 offset:7168
	s_add_i32 s8, s8, s61
	ds_read_b64_tr_b16 v[60:61], v4 offset:7680
	s_mul_i32 s8, s8, s60
	ds_read_b64_tr_b16 a[44:45], v1 offset:41472
	s_add_i32 s1, s62, 0x6000
	s_ashr_i32 s9, s8, 31
	ds_read_b64_tr_b16 a[46:47], v1 offset:45568
	v_mfma_f32_32x32x16_bf16 a[240:255], a[124:127], v[66:69], a[240:255]
	s_waitcnt vmcnt(4) lgkmcnt(6)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[110:113], a[0:3], v[30:33], 0
	ds_read_b64_tr_b16 v[62:63], v1 offset:49152
	s_lshl_b64 s[8:9], s[8:9], 1
	v_add_u32_e32 v17, s1, v17
	ds_read_b64_tr_b16 v[64:65], v1 offset:53248
	s_add_u32 s44, s52, s8
	v_readfirstlane_b32 s1, v17
	v_add_u32_e32 v17, 0x1000, v17
	ds_read_b64_tr_b16 v[66:67], v1 offset:49664
	s_addc_u32 s45, s53, s9
	s_mov_b32 m0, s1
	v_readfirstlane_b32 s1, v17
	ds_read_b64_tr_b16 v[68:69], v1 offset:53760
	v_mfma_f32_16x16x32_bf16 v[110:113], a[8:11], v[34:37], v[110:113]
	buffer_load_dwordx4 v14, s[44:47], 0 offen lds
	s_mov_b32 m0, s1
	s_add_i32 s1, s27, 0x3000
	buffer_load_dwordx4 v15, s[44:47], 0 offen lds
	v_mfma_f32_16x16x32_bf16 v[110:113], a[16:19], v[38:41], v[110:113]
	ds_read_b64_tr_b16 v[70:71], v1 offset:57344
	ds_read_b64_tr_b16 v[72:73], v1 offset:61440
	ds_read_b64_tr_b16 v[74:75], v1 offset:57856
	ds_read_b64_tr_b16 v[76:77], v1 offset:61952
	v_mfma_f32_16x16x32_bf16 v[110:113], a[24:27], v[42:45], v[110:113]
	v_mfma_f32_16x16x32_bf16 v[110:113], a[32:35], v[46:49], v[110:113]
	v_add_u32_e32 v17, s1, v11
	ds_read_b128 a[112:115], v17
	ds_read_b128 a[116:119], v17 offset:1024
	v_mfma_f32_16x16x32_bf16 v[110:113], a[40:43], v[50:53], v[110:113]
	s_waitcnt lgkmcnt(4)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[110:113], v[62:65], v[54:57], v[110:113]
	ds_read_b128 a[120:123], v17 offset:2048
	ds_read_b128 a[124:127], v17 offset:3072
	v_mfma_f32_16x16x32_bf16 v[110:113], v[70:73], v[58:61], v[110:113]
	v_mfma_f32_16x16x32_bf16 v[114:117], a[4:7], v[30:33], 0
	ds_read_b128 a[0:3], v10
	ds_read_b128 a[4:7], v10 offset:1024
	v_mfma_f32_16x16x32_bf16 v[114:117], a[12:15], v[34:37], v[114:117]
	v_mfma_f32_16x16x32_bf16 v[114:117], a[20:23], v[38:41], v[114:117]
	ds_read_b128 a[8:11], v10 offset:2048
	ds_read_b128 a[12:15], v10 offset:3072
	v_mfma_f32_16x16x32_bf16 v[114:117], a[28:31], v[42:45], v[114:117]
	v_mul_f32_e32 v110, 0x3db504f3, v110
	v_mul_f32_e32 v111, 0x3db504f3, v111
	v_mfma_f32_16x16x32_bf16 v[114:117], a[36:39], v[46:49], v[114:117]
	ds_read_b128 a[16:19], v10 offset:4096
	ds_read_b128 a[20:23], v10 offset:5120
	v_mfma_f32_16x16x32_bf16 v[114:117], a[44:47], v[50:53], v[114:117]
	v_mul_f32_e32 v112, 0x3db504f3, v112
	v_mul_f32_e32 v113, 0x3db504f3, v113
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_16x16x32_bf16 v[114:117], v[66:69], v[54:57], v[114:117]
	ds_read_b128 a[24:27], v10 offset:6144
	ds_read_b128 a[28:31], v10 offset:7168
	v_mfma_f32_16x16x32_bf16 v[114:117], v[74:77], v[58:61], v[114:117]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x32_bf16 v[46:49], a[112:115], a[0:3], 0
	ds_read_b128 a[32:35], v10 offset:8192
	ds_read_b128 a[36:39], v10 offset:9216
	v_mfma_f32_16x16x32_bf16 v[46:49], a[116:119], a[4:7], v[46:49]
	v_mfma_f32_16x16x32_bf16 v[46:49], a[120:123], a[8:11], v[46:49]
	ds_read_b128 a[40:43], v10 offset:10240
	ds_read_b128 a[44:47], v10 offset:11264
	v_mfma_f32_16x16x32_bf16 v[46:49], a[124:127], a[12:15], v[46:49]
	v_mul_f32_e32 v114, 0x3db504f3, v114
	v_mul_f32_e32 v115, 0x3db504f3, v115
	v_mfma_f32_16x16x32_bf16 v[50:53], a[112:115], a[16:19], 0
	ds_read_b128 v[62:65], v10 offset:12288
	ds_read_b128 v[66:69], v10 offset:13312
	v_mfma_f32_16x16x32_bf16 v[50:53], a[116:119], a[20:23], v[50:53]
	v_mul_f32_e32 v116, 0x3db504f3, v116
	v_mul_f32_e32 v117, 0x3db504f3, v117
	v_mfma_f32_16x16x32_bf16 v[50:53], a[120:123], a[24:27], v[50:53]
	ds_read_b128 v[70:73], v10 offset:14336
	s_add_i32 s8, s62, 0x3000
	ds_read_b128 v[74:77], v10 offset:15360
	v_mfma_f32_16x16x32_bf16 v[50:53], a[124:127], a[28:31], v[50:53]
	v_mul_f32_e32 v46, 0x3e0293ee, v46
	v_mul_f32_e32 v47, 0x3e0293ee, v47
	v_mul_f32_e32 v48, 0x3e0293ee, v48
	v_mul_f32_e32 v49, 0x3e0293ee, v49
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_bf16 v[54:57], a[112:115], a[32:35], 0
	v_add_u32_e32 v17, s8, v11
	ds_read_b128 v[78:81], v17
	ds_read_b128 v[82:85], v17 offset:1024
	v_mfma_f32_16x16x32_bf16 v[54:57], a[116:119], a[36:39], v[54:57]
	v_subrev_f32_dpp v46, v126, v46 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v47, v126, v47 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v48, v126, v48 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v49, v126, v49 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_bf16 v[54:57], a[120:123], a[40:43], v[54:57]
	ds_read_b128 v[86:89], v17 offset:2048
	ds_read_b128 v[90:93], v17 offset:3072
	v_mfma_f32_16x16x32_bf16 v[54:57], a[124:127], a[44:47], v[54:57]
	v_mul_f32_e32 v50, 0x3e0293ee, v50
	v_mul_f32_e32 v51, 0x3e0293ee, v51
	v_mul_f32_e32 v52, 0x3e0293ee, v52
	v_mul_f32_e32 v53, 0x3e0293ee, v53
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_bf16 v[58:61], a[112:115], v[62:65], 0
	v_add_u32_e32 v17, s8, v12
	ds_read_b64_tr_b16 v[94:95], v17
	ds_read_b64_tr_b16 v[96:97], v17 offset:256
	ds_read_b64_tr_b16 v[98:99], v17 offset:1024
	ds_read_b64_tr_b16 v[100:101], v17 offset:1280
	v_mfma_f32_16x16x32_bf16 v[58:61], a[116:119], v[66:69], v[58:61]
	v_subrev_f32_dpp v50, v126, v50 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v51, v126, v51 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v52, v126, v52 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v53, v126, v53 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_16x16x32_bf16 v[58:61], a[120:123], v[70:73], v[58:61]
	ds_read_b64_tr_b16 v[102:103], v17 offset:2048
	ds_read_b64_tr_b16 v[104:105], v17 offset:2304
	ds_read_b64_tr_b16 v[106:107], v17 offset:3072
	ds_read_b64_tr_b16 v[108:109], v17 offset:3328
	v_mfma_f32_16x16x32_bf16 v[58:61], a[124:127], v[74:77], v[58:61]
	v_mul_f32_e32 v54, 0x3e0293ee, v54
	v_mul_f32_e32 v55, 0x3e0293ee, v55
	v_mul_f32_e32 v56, 0x3e0293ee, v56
	v_mul_f32_e32 v57, 0x3e0293ee, v57
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_16x16x32_bf16 v[62:65], v[78:81], a[48:51], 0
	v_subrev_f32_dpp v54, v126, v54 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v55, v126, v55 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v56, v126, v56 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v57, v126, v57 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	v_mfma_f32_16x16x32_bf16 v[62:65], v[82:85], a[52:55], v[62:65]
	v_exp_f32_e32 v46, v46
	v_exp_f32_e32 v47, v47
	v_exp_f32_e32 v48, v48
	v_exp_f32_e32 v49, v49
	v_mfma_f32_16x16x32_bf16 v[62:65], v[86:89], a[56:59], v[62:65]
	v_add_u32_e32 v17, s1, v12
	ds_read_b64_tr_b16 a[112:113], v17
	ds_read_b64_tr_b16 a[114:115], v17 offset:256
	v_mfma_f32_16x16x32_bf16 v[62:65], v[90:93], a[60:63], v[62:65]
	v_exp_f32_e32 v50, v50
	v_exp_f32_e32 v51, v51
	v_exp_f32_e32 v52, v52
	v_exp_f32_e32 v53, v53
	v_mfma_f32_16x16x32_bf16 v[66:69], v[78:81], a[64:67], 0
	ds_read_b64_tr_b16 a[116:117], v17 offset:1024
	ds_read_b64_tr_b16 a[118:119], v17 offset:1280
	v_mfma_f32_16x16x32_bf16 v[66:69], v[82:85], a[68:71], v[66:69]
	v_mul_f32_e32 v58, 0x3e0293ee, v58
	v_mul_f32_e32 v59, 0x3e0293ee, v59
	v_mul_f32_e32 v60, 0x3e0293ee, v60
	v_mul_f32_e32 v61, 0x3e0293ee, v61
	v_mfma_f32_16x16x32_bf16 v[66:69], v[86:89], a[72:75], v[66:69]
	v_subrev_f32_dpp v58, v126, v58 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v59, v126, v59 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v60, v126, v60 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v61, v126, v61 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	v_mfma_f32_16x16x32_bf16 v[66:69], v[90:93], a[76:79], v[66:69]
	v_cvt_pk_bf16_f32 v118, v46, v47
	v_cvt_pk_bf16_f32 v119, v48, v49
	v_mfma_f32_16x16x32_bf16 v[70:73], v[78:81], a[80:83], 0
	v_exp_f32_e32 v54, v54
	v_exp_f32_e32 v55, v55
	v_exp_f32_e32 v56, v56
	v_exp_f32_e32 v57, v57
	v_mfma_f32_16x16x32_bf16 v[70:73], v[82:85], a[84:87], v[70:73]
	v_cvt_pk_bf16_f32 v120, v50, v51
	v_cvt_pk_bf16_f32 v121, v52, v53
	v_mfma_f32_16x16x32_bf16 v[70:73], v[86:89], a[88:91], v[70:73]
	ds_read_b64_tr_b16 a[120:121], v17 offset:2048
	ds_read_b64_tr_b16 a[122:123], v17 offset:2304
	v_mfma_f32_16x16x32_bf16 v[70:73], v[90:93], a[92:95], v[70:73]
	v_exp_f32_e32 v58, v58
	v_exp_f32_e32 v59, v59
	v_exp_f32_e32 v60, v60
	v_exp_f32_e32 v61, v61
	v_mfma_f32_16x16x32_bf16 v[74:77], v[78:81], a[96:99], 0
	ds_read_b64_tr_b16 a[124:125], v17 offset:3072
	ds_read_b64_tr_b16 a[126:127], v17 offset:3328
	v_mfma_f32_16x16x32_bf16 v[74:77], v[82:85], a[100:103], v[74:77]
	v_cvt_pk_bf16_f32 v122, v54, v55
	v_cvt_pk_bf16_f32 v123, v56, v57
	v_cvt_pk_bf16_f32 v124, v58, v59
	v_cvt_pk_bf16_f32 v125, v60, v61
	v_mfma_f32_16x16x32_bf16 v[74:77], v[86:89], a[104:107], v[74:77]
	v_permlane16_swap_b32_e32 v118, v120
	v_permlane16_swap_b32_e32 v119, v121
	v_permlane16_swap_b32_e32 v122, v124
	v_permlane16_swap_b32_e32 v123, v125
	v_mfma_f32_16x16x32_bf16 v[74:77], v[90:93], a[108:111], v[74:77]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x16_bf16 v[128:143], v[94:97], v[118:121], v[128:143]
	ds_read_b64_tr_b16 a[0:1], v1
	ds_read_b64_tr_b16 a[2:3], v1 offset:4096
	ds_read_b64_tr_b16 a[4:5], v1 offset:512
	ds_read_b64_tr_b16 a[6:7], v1 offset:4608
	v_mfma_f32_32x32x16_bf16 v[144:159], v[94:97], v[122:125], v[144:159]
	v_subrev_f32_dpp v62, v127, v62 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v63, v127, v63 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v64, v127, v64 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v65, v127, v65 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v66, v127, v66 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v67, v127, v67 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v68, v127, v68 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v69, v127, v69 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	v_mfma_f32_32x32x16_bf16 v[160:175], v[98:101], v[118:121], v[160:175]
	ds_read_b64_tr_b16 a[8:9], v1 offset:8192
	ds_read_b64_tr_b16 a[10:11], v1 offset:12288
	ds_read_b64_tr_b16 a[12:13], v1 offset:8704
	ds_read_b64_tr_b16 a[14:15], v1 offset:12800
	v_mfma_f32_32x32x16_bf16 v[176:191], v[98:101], v[122:125], v[176:191]
	v_mul_f32_e32 v62, v46, v62
	v_mul_f32_e32 v63, v47, v63
	v_mul_f32_e32 v64, v48, v64
	v_mul_f32_e32 v65, v49, v65
	v_mul_f32_e32 v66, v50, v66
	v_mul_f32_e32 v67, v51, v67
	v_mul_f32_e32 v68, v52, v68
	v_mul_f32_e32 v69, v53, v69
	v_cvt_pk_bf16_f32 v62, v62, v63
	v_cvt_pk_bf16_f32 v63, v64, v65
	v_cvt_pk_bf16_f32 v64, v66, v67
	v_cvt_pk_bf16_f32 v65, v68, v69
	v_subrev_f32_dpp v70, v127, v70 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v71, v127, v71 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v72, v127, v72 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v73, v127, v73 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	v_mfma_f32_32x32x16_bf16 v[192:207], v[102:105], v[118:121], v[192:207]
	ds_read_b64_tr_b16 a[16:17], v1 offset:16384
	ds_read_b64_tr_b16 a[18:19], v1 offset:20480
	ds_write_b64 v9, v[62:63]
	ds_write_b64 v9, v[64:65] offset:512
	v_mfma_f32_32x32x16_bf16 v[208:223], v[102:105], v[122:125], v[208:223]
	v_subrev_f32_dpp v74, v127, v74 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v75, v127, v75 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v76, v127, v76 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v77, v127, v77 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	v_mul_f32_e32 v70, v54, v70
	v_mul_f32_e32 v71, v55, v71
	v_mul_f32_e32 v72, v56, v72
	v_mul_f32_e32 v73, v57, v73
	v_mul_f32_e32 v74, v58, v74
	v_mul_f32_e32 v75, v59, v75
	v_mul_f32_e32 v76, v60, v76
	v_mul_f32_e32 v77, v61, v77
	v_cvt_pk_bf16_f32 v66, v70, v71
	v_cvt_pk_bf16_f32 v67, v72, v73
	v_cvt_pk_bf16_f32 v68, v74, v75
	v_cvt_pk_bf16_f32 v69, v76, v77
	v_mfma_f32_32x32x16_bf16 v[224:239], v[106:109], v[118:121], v[224:239]
	v_add_u32_e32 v17, s25, v13
	ds_read_b32 v126, v17

	v_add_u32_e32 v17, s39, v13
	ds_read_b32 v127, v17

	ds_write_b64 v9, v[66:67] offset:1024
	ds_write_b64 v9, v[68:69] offset:1536
	v_mfma_f32_32x32x16_bf16 v[240:255], v[106:109], v[122:125], v[240:255]
	v_permlane16_swap_b32_e32 v62, v64
	v_permlane16_swap_b32_e32 v63, v65
	v_permlane16_swap_b32_e32 v66, v68
	v_permlane16_swap_b32_e32 v67, v69
	s_waitcnt lgkmcnt(12)
	v_mfma_f32_32x32x16_bf16 a[128:143], a[112:115], v[62:65], a[128:143]
	ds_read_b64_tr_b16 a[20:21], v1 offset:16896
	ds_read_b64_tr_b16 a[22:23], v1 offset:20992
	ds_read_b64_tr_b16 a[24:25], v1 offset:24576
	ds_read_b64_tr_b16 a[26:27], v1 offset:28672
	ds_read_b64_tr_b16 a[28:29], v1 offset:25088
	ds_read_b64_tr_b16 a[30:31], v1 offset:29184
	ds_read_b64_tr_b16 a[32:33], v1 offset:32768
	ds_read_b64_tr_b16 a[34:35], v1 offset:36864
	v_mfma_f32_32x32x16_bf16 a[144:159], a[112:115], v[66:69], a[144:159]
	s_waitcnt lgkmcnt(8)
	s_barrier
	v_mfma_f32_32x32x16_bf16 a[160:175], a[116:119], v[62:65], a[160:175]
	ds_read_b64_tr_b16 v[30:31], v4
	ds_read_b64_tr_b16 v[32:33], v4 offset:512
	ds_read_b64_tr_b16 v[34:35], v4 offset:1024
	ds_read_b64_tr_b16 v[36:37], v4 offset:1536
	s_add_i32 s0, s0, s13
	ds_read_b64_tr_b16 v[38:39], v4 offset:2048
	s_ashr_i32 s1, s0, 31
	ds_read_b64_tr_b16 v[40:41], v4 offset:2560
	s_lshl_b64 s[0:1], s[0:1], 1
	ds_read_b64_tr_b16 v[42:43], v4 offset:3072
	s_add_u32 s8, s6, s0
	ds_read_b64_tr_b16 v[44:45], v4 offset:3584
	v_mfma_f32_32x32x16_bf16 a[176:191], a[116:119], v[66:69], a[176:191]
	v_mul_f32_e32 v126, 0x3fb8aa3b, v126
	s_addc_u32 s9, s7, s1
	v_cvt_pk_bf16_f32 v110, v110, v111
	v_cvt_pk_bf16_f32 v111, v112, v113
	buffer_atomic_pk_add_bf16 v110, v5, s[8:11], 0 offen
	buffer_atomic_pk_add_bf16 v111, v6, s[8:11], 0 offen
	v_mfma_f32_32x32x16_bf16 a[192:207], a[120:123], v[62:65], a[192:207]
	ds_read_b64_tr_b16 v[46:47], v4 offset:4096
	ds_read_b64_tr_b16 v[48:49], v4 offset:4608
	ds_read_b64_tr_b16 v[50:51], v4 offset:5120
	ds_read_b64_tr_b16 v[52:53], v4 offset:5632
	ds_read_b64_tr_b16 a[36:37], v1 offset:33280
	ds_read_b64_tr_b16 a[38:39], v1 offset:37376
	ds_read_b64_tr_b16 a[40:41], v1 offset:40960
	ds_read_b64_tr_b16 a[42:43], v1 offset:45056
	v_mfma_f32_32x32x16_bf16 a[208:223], a[120:123], v[66:69], a[208:223]
	v_cvt_pk_bf16_f32 v114, v114, v115
	v_cvt_pk_bf16_f32 v115, v116, v117
	buffer_atomic_pk_add_bf16 v114, v7, s[8:11], 0 offen
	buffer_atomic_pk_add_bf16 v115, v8, s[8:11], 0 offen
	v_mfma_f32_32x32x16_bf16 a[224:239], a[124:127], v[62:65], a[224:239]
	ds_read_b64_tr_b16 v[54:55], v4 offset:6144
	ds_read_b64_tr_b16 v[56:57], v4 offset:6656
	ds_read_b64_tr_b16 v[58:59], v4 offset:7168
	ds_read_b64_tr_b16 v[60:61], v4 offset:7680
	ds_read_b64_tr_b16 a[44:45], v1 offset:41472
	ds_read_b64_tr_b16 a[46:47], v1 offset:45568
	v_mfma_f32_32x32x16_bf16 a[240:255], a[124:127], v[66:69], a[240:255]
	s_waitcnt vmcnt(4) lgkmcnt(6)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[110:113], a[0:3], v[30:33], 0
	ds_read_b64_tr_b16 v[62:63], v1 offset:49152
	ds_read_b64_tr_b16 v[64:65], v1 offset:53248
	ds_read_b64_tr_b16 v[66:67], v1 offset:49664
	ds_read_b64_tr_b16 v[68:69], v1 offset:53760
	v_mfma_f32_16x16x32_bf16 v[110:113], a[8:11], v[34:37], v[110:113]
	v_mfma_f32_16x16x32_bf16 v[110:113], a[16:19], v[38:41], v[110:113]
	ds_read_b64_tr_b16 v[70:71], v1 offset:57344
	ds_read_b64_tr_b16 v[72:73], v1 offset:61440
	ds_read_b64_tr_b16 v[74:75], v1 offset:57856
	ds_read_b64_tr_b16 v[76:77], v1 offset:61952
	v_mfma_f32_16x16x32_bf16 v[110:113], a[24:27], v[42:45], v[110:113]
	v_mfma_f32_16x16x32_bf16 v[110:113], a[32:35], v[46:49], v[110:113]
	v_add_u32_e32 v17, s19, v11
	ds_read_b128 a[112:115], v17
	ds_read_b128 a[116:119], v17 offset:1024
	v_mfma_f32_16x16x32_bf16 v[110:113], a[40:43], v[50:53], v[110:113]
	s_waitcnt lgkmcnt(4)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[110:113], v[62:65], v[54:57], v[110:113]
	ds_read_b128 a[120:123], v17 offset:2048
	ds_read_b128 a[124:127], v17 offset:3072
	v_mfma_f32_16x16x32_bf16 v[110:113], v[70:73], v[58:61], v[110:113]
	v_mfma_f32_16x16x32_bf16 v[114:117], a[4:7], v[30:33], 0
	ds_read_b128 a[0:3], v10
	ds_read_b128 a[4:7], v10 offset:1024
	v_mfma_f32_16x16x32_bf16 v[114:117], a[12:15], v[34:37], v[114:117]
	v_mfma_f32_16x16x32_bf16 v[114:117], a[20:23], v[38:41], v[114:117]
	ds_read_b128 a[8:11], v10 offset:2048
	ds_read_b128 a[12:15], v10 offset:3072
	v_mfma_f32_16x16x32_bf16 v[114:117], a[28:31], v[42:45], v[114:117]
	v_mul_f32_e32 v110, 0x3db504f3, v110
	v_mul_f32_e32 v111, 0x3db504f3, v111
	v_mfma_f32_16x16x32_bf16 v[114:117], a[36:39], v[46:49], v[114:117]
	ds_read_b128 a[16:19], v10 offset:4096
	ds_read_b128 a[20:23], v10 offset:5120
	v_mfma_f32_16x16x32_bf16 v[114:117], a[44:47], v[50:53], v[114:117]
	v_mul_f32_e32 v112, 0x3db504f3, v112
	v_mul_f32_e32 v113, 0x3db504f3, v113
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_16x16x32_bf16 v[114:117], v[66:69], v[54:57], v[114:117]
	ds_read_b128 a[24:27], v10 offset:6144
	ds_read_b128 a[28:31], v10 offset:7168
	s_movk_i32 s17, 0x80
	s_add_i32 s19, s66, 32
	s_add_i32 s25, s67, 32
	v_lshlrev_b32_e32 v2, 1, v2
	v_mfma_f32_16x16x32_bf16 v[114:117], v[74:77], v[58:61], v[114:117]
	s_waitcnt lgkmcnt(2)
.LBB0_1:
	s_add_i32 s1, s37, -1
	s_and_b32 s8, s1, 0xffff
	s_and_b32 s36, s17, 0x1fc0
	s_add_i32 s8, s8, 0xffff
	s_lshr_b32 s39, s37, 7
	s_add_i32 s9, s36, s66
	s_sext_i32_i16 s28, s8
	s_add_i32 s39, s39, s61
	s_mul_i32 s9, s9, s22
	s_bfe_u32 s29, s28, 0x70018
	s_add_i32 s9, s9, s39
	s_add_i32 s41, s8, s29
	s_mul_i32 s28, s9, s33
	s_sext_i32_i16 s41, s41
	s_lshl_b32 s0, s59, 14
	s_ashr_i32 s29, s28, 31
	s_ashr_i32 s44, s41, 7
	s_and_b32 s41, s41, 0xffffff80
	s_lshr_b32 s68, s1, 7
	s_add_i32 s1, s27, s0
	s_lshl_b64 s[28:29], s[28:29], 1
	s_sub_i32 s8, s8, s41
	s_add_u32 s28, s50, s28
	v_add_u32_e32 v18, s1, v2
	s_sext_i32_i16 s45, s8
	s_addc_u32 s29, s51, s29
	s_add_i32 s8, s39, s58
	v_readfirstlane_b32 s40, v18
	v_add_u32_e32 v18, 0x1000, v18
	s_mul_i32 s8, s63, s8
	v_readfirstlane_b32 s9, v18
	s_mov_b32 m0, s40
	s_add_i32 s8, s8, s36
	s_lshl_b32 s72, s59, 8
	buffer_load_dwordx4 v14, s[28:31], 0 offen lds
	s_mov_b32 m0, s9
	s_ashr_i32 s9, s8, 31
	s_add_i32 s69, s5, s72
	s_lshl_b64 s[8:9], s[8:9], 2
	s_add_u32 s40, s54, s8
	s_mov_b32 s43, s31
	buffer_load_dwordx4 v15, s[28:31], 0 offen lds
	v_mfma_f32_16x16x32_bf16 v[46:49], a[112:115], a[0:3], 0
	ds_read_b128 a[32:35], v10 offset:8192
	s_addc_u32 s41, s55, s9
	s_mov_b32 m0, s69
	ds_read_b128 a[36:39], v10 offset:9216
	v_mfma_f32_16x16x32_bf16 v[46:49], a[116:119], a[4:7], v[46:49]
	s_lshl_b32 s8, s35, 14
	buffer_load_dword v16, s[40:43], 0 offen lds
	v_mfma_f32_16x16x32_bf16 v[46:49], a[120:123], a[8:11], v[46:49]
	ds_read_b128 a[40:43], v10 offset:10240
	ds_read_b128 a[44:47], v10 offset:11264
	v_mfma_f32_16x16x32_bf16 v[46:49], a[124:127], a[12:15], v[46:49]
	v_mul_f32_e32 v114, 0x3db504f3, v114
	v_mul_f32_e32 v115, 0x3db504f3, v115
	v_mfma_f32_16x16x32_bf16 v[50:53], a[112:115], a[16:19], 0
	ds_read_b128 v[62:65], v10 offset:12288
	v_add_u32_e32 v18, s69, v13
	s_add_i32 s69, s62, s8
	ds_read_b128 v[66:69], v10 offset:13312
	v_mfma_f32_16x16x32_bf16 v[50:53], a[116:119], a[20:23], v[50:53]
	v_mul_f32_e32 v116, 0x3db504f3, v116
	v_mul_f32_e32 v117, 0x3db504f3, v117
	v_mfma_f32_16x16x32_bf16 v[50:53], a[120:123], a[24:27], v[50:53]
	ds_read_b128 v[70:73], v10 offset:14336
	v_add_u32_e32 v21, s69, v11
	ds_read_b128 v[74:77], v10 offset:15360
	v_mfma_f32_16x16x32_bf16 v[50:53], a[124:127], a[28:31], v[50:53]
	v_mul_f32_e32 v46, 0x3e0293ee, v46
	v_mul_f32_e32 v47, 0x3e0293ee, v47
	v_mul_f32_e32 v48, 0x3e0293ee, v48
	v_mul_f32_e32 v49, 0x3e0293ee, v49
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_bf16 v[54:57], a[112:115], a[32:35], 0
	ds_read_b128 v[78:81], v21
	ds_read_b128 v[82:85], v21 offset:1024
	v_mfma_f32_16x16x32_bf16 v[54:57], a[116:119], a[36:39], v[54:57]
	v_subrev_f32_dpp v46, v126, v46 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v47, v126, v47 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v48, v126, v48 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v49, v126, v49 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_bf16 v[54:57], a[120:123], a[40:43], v[54:57]
	ds_read_b128 v[86:89], v21 offset:2048
	v_add_u32_e32 v20, s69, v12
	ds_read_b128 v[90:93], v21 offset:3072
	v_mfma_f32_16x16x32_bf16 v[54:57], a[124:127], a[44:47], v[54:57]
	v_mul_f32_e32 v50, 0x3e0293ee, v50
	v_mul_f32_e32 v51, 0x3e0293ee, v51
	v_mul_f32_e32 v52, 0x3e0293ee, v52
	v_mul_f32_e32 v53, 0x3e0293ee, v53
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_bf16 v[58:61], a[112:115], v[62:65], 0
	ds_read_b64_tr_b16 v[94:95], v20
	ds_read_b64_tr_b16 v[96:97], v20 offset:256
	ds_read_b64_tr_b16 v[98:99], v20 offset:1024
	ds_read_b64_tr_b16 v[100:101], v20 offset:1280
	v_mfma_f32_16x16x32_bf16 v[58:61], a[116:119], v[66:69], v[58:61]
	v_subrev_f32_dpp v50, v126, v50 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v51, v126, v51 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v52, v126, v52 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v53, v126, v53 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_16x16x32_bf16 v[58:61], a[120:123], v[70:73], v[58:61]
	ds_read_b64_tr_b16 v[102:103], v20 offset:2048
	ds_read_b64_tr_b16 v[104:105], v20 offset:2304
	s_add_i32 s70, s27, s8
	ds_read_b64_tr_b16 v[106:107], v20 offset:3072
	v_add_u32_e32 v19, s70, v12
	ds_read_b64_tr_b16 v[108:109], v20 offset:3328
	v_mfma_f32_16x16x32_bf16 v[58:61], a[124:127], v[74:77], v[58:61]
	v_mul_f32_e32 v54, 0x3e0293ee, v54
	v_mul_f32_e32 v55, 0x3e0293ee, v55
	v_mul_f32_e32 v56, 0x3e0293ee, v56
	v_mul_f32_e32 v57, 0x3e0293ee, v57
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_16x16x32_bf16 v[62:65], v[78:81], a[48:51], 0
	v_subrev_f32_dpp v54, v126, v54 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v55, v126, v55 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v56, v126, v56 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v57, v126, v57 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	v_mfma_f32_16x16x32_bf16 v[62:65], v[82:85], a[52:55], v[62:65]
	v_exp_f32_e32 v46, v46
	v_exp_f32_e32 v47, v47
	v_exp_f32_e32 v48, v48
	v_exp_f32_e32 v49, v49
	v_mfma_f32_16x16x32_bf16 v[62:65], v[86:89], a[56:59], v[62:65]
	ds_read_b64_tr_b16 a[112:113], v19
	ds_read_b64_tr_b16 a[114:115], v19 offset:256
	v_mfma_f32_16x16x32_bf16 v[62:65], v[90:93], a[60:63], v[62:65]
	v_exp_f32_e32 v50, v50
	v_exp_f32_e32 v51, v51
	v_exp_f32_e32 v52, v52
	v_exp_f32_e32 v53, v53
	v_mfma_f32_16x16x32_bf16 v[66:69], v[78:81], a[64:67], 0
	ds_read_b64_tr_b16 a[116:117], v19 offset:1024
	ds_read_b64_tr_b16 a[118:119], v19 offset:1280
	v_mfma_f32_16x16x32_bf16 v[66:69], v[82:85], a[68:71], v[66:69]
	v_mul_f32_e32 v58, 0x3e0293ee, v58
	v_mul_f32_e32 v59, 0x3e0293ee, v59
	v_mul_f32_e32 v60, 0x3e0293ee, v60
	v_mul_f32_e32 v61, 0x3e0293ee, v61
	v_mfma_f32_16x16x32_bf16 v[66:69], v[86:89], a[72:75], v[66:69]
	v_subrev_f32_dpp v58, v126, v58 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v59, v126, v59 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v60, v126, v60 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v61, v126, v61 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	v_mfma_f32_16x16x32_bf16 v[66:69], v[90:93], a[76:79], v[66:69]
	v_cvt_pk_bf16_f32 v118, v46, v47
	v_cvt_pk_bf16_f32 v119, v48, v49
	v_mfma_f32_16x16x32_bf16 v[70:73], v[78:81], a[80:83], 0
	v_exp_f32_e32 v54, v54
	v_exp_f32_e32 v55, v55
	v_exp_f32_e32 v56, v56
	v_exp_f32_e32 v57, v57
	v_mfma_f32_16x16x32_bf16 v[70:73], v[82:85], a[84:87], v[70:73]
	v_cvt_pk_bf16_f32 v120, v50, v51
	v_cvt_pk_bf16_f32 v121, v52, v53
	v_mfma_f32_16x16x32_bf16 v[70:73], v[86:89], a[88:91], v[70:73]
	ds_read_b64_tr_b16 a[120:121], v19 offset:2048
	ds_read_b64_tr_b16 a[122:123], v19 offset:2304
	v_mfma_f32_16x16x32_bf16 v[70:73], v[90:93], a[92:95], v[70:73]
	v_exp_f32_e32 v58, v58
	v_exp_f32_e32 v59, v59
	v_exp_f32_e32 v60, v60
	v_exp_f32_e32 v61, v61
	v_mfma_f32_16x16x32_bf16 v[74:77], v[78:81], a[96:99], 0
	ds_read_b64_tr_b16 a[124:125], v19 offset:3072
	ds_read_b64_tr_b16 a[126:127], v19 offset:3328
	v_mfma_f32_16x16x32_bf16 v[74:77], v[82:85], a[100:103], v[74:77]
	v_cvt_pk_bf16_f32 v122, v54, v55
	v_cvt_pk_bf16_f32 v123, v56, v57
	v_cvt_pk_bf16_f32 v124, v58, v59
	v_cvt_pk_bf16_f32 v125, v60, v61
	v_mfma_f32_16x16x32_bf16 v[74:77], v[86:89], a[104:107], v[74:77]
	v_permlane16_swap_b32_e32 v118, v120
	v_permlane16_swap_b32_e32 v119, v121
	v_permlane16_swap_b32_e32 v122, v124
	v_permlane16_swap_b32_e32 v123, v125
	v_mfma_f32_16x16x32_bf16 v[74:77], v[90:93], a[108:111], v[74:77]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x16_bf16 v[128:143], v[94:97], v[118:121], v[128:143]
	ds_read_b64_tr_b16 a[0:1], v1
	ds_read_b64_tr_b16 a[2:3], v1 offset:4096
	ds_read_b64_tr_b16 a[4:5], v1 offset:512
	ds_read_b64_tr_b16 a[6:7], v1 offset:4608
	v_mfma_f32_32x32x16_bf16 v[144:159], v[94:97], v[122:125], v[144:159]
	v_subrev_f32_dpp v62, v127, v62 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v63, v127, v63 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v64, v127, v64 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v65, v127, v65 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v66, v127, v66 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v67, v127, v67 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v68, v127, v68 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v69, v127, v69 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	v_mfma_f32_32x32x16_bf16 v[160:175], v[98:101], v[118:121], v[160:175]
	ds_read_b64_tr_b16 a[8:9], v1 offset:8192
	ds_read_b64_tr_b16 a[10:11], v1 offset:12288
	ds_read_b64_tr_b16 a[12:13], v1 offset:8704
	ds_read_b64_tr_b16 a[14:15], v1 offset:12800
	v_mfma_f32_32x32x16_bf16 v[176:191], v[98:101], v[122:125], v[176:191]
	v_mul_f32_e32 v62, v46, v62
	v_mul_f32_e32 v63, v47, v63
	v_mul_f32_e32 v64, v48, v64
	v_mul_f32_e32 v65, v49, v65
	v_mul_f32_e32 v66, v50, v66
	v_mul_f32_e32 v67, v51, v67
	v_mul_f32_e32 v68, v52, v68
	v_mul_f32_e32 v69, v53, v69
	v_cvt_pk_bf16_f32 v62, v62, v63
	v_cvt_pk_bf16_f32 v63, v64, v65
	v_cvt_pk_bf16_f32 v64, v66, v67
	v_cvt_pk_bf16_f32 v65, v68, v69
	v_subrev_f32_dpp v70, v127, v70 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v71, v127, v71 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v72, v127, v72 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v73, v127, v73 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	v_mfma_f32_32x32x16_bf16 v[192:207], v[102:105], v[118:121], v[192:207]
	ds_read_b64_tr_b16 a[16:17], v1 offset:16384
	s_lshl_b32 s9, s35, 8
	ds_read_b64_tr_b16 a[18:19], v1 offset:20480
	s_add_i32 s74, s5, s9
	ds_write_b64 v9, v[62:63]
	s_add_i32 s75, s23, s9
	v_add3_u32 v24, s74, 64, v13
	ds_write_b64 v9, v[64:65] offset:512
	v_mfma_f32_32x32x16_bf16 v[208:223], v[102:105], v[122:125], v[208:223]
	v_subrev_f32_dpp v74, v127, v74 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v75, v127, v75 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v76, v127, v76 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v77, v127, v77 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	v_mul_f32_e32 v70, v54, v70
	v_mul_f32_e32 v71, v55, v71
	v_mul_f32_e32 v72, v56, v72
	v_mul_f32_e32 v73, v57, v73
	v_mul_f32_e32 v74, v58, v74
	v_mul_f32_e32 v75, v59, v75
	v_mul_f32_e32 v76, v60, v76
	v_mul_f32_e32 v77, v61, v77
	v_cvt_pk_bf16_f32 v66, v70, v71
	v_cvt_pk_bf16_f32 v67, v72, v73
	v_cvt_pk_bf16_f32 v68, v74, v75
	v_cvt_pk_bf16_f32 v69, v76, v77
	v_mfma_f32_32x32x16_bf16 v[224:239], v[106:109], v[118:121], v[224:239]
	ds_read_b32 v126, v24

	v_add3_u32 v22, s75, 64, v13
	ds_read_b32 v127, v22

	ds_write_b64 v9, v[66:67] offset:1024
	ds_write_b64 v9, v[68:69] offset:1536
	v_mfma_f32_32x32x16_bf16 v[240:255], v[106:109], v[122:125], v[240:255]
	v_permlane16_swap_b32_e32 v62, v64
	v_permlane16_swap_b32_e32 v63, v65
	v_permlane16_swap_b32_e32 v66, v68
	v_permlane16_swap_b32_e32 v67, v69
	s_waitcnt lgkmcnt(12)
	v_mfma_f32_32x32x16_bf16 a[128:143], a[112:115], v[62:65], a[128:143]
	ds_read_b64_tr_b16 a[20:21], v1 offset:16896
	ds_read_b64_tr_b16 a[22:23], v1 offset:20992
	ds_read_b64_tr_b16 a[24:25], v1 offset:24576
	ds_read_b64_tr_b16 a[26:27], v1 offset:28672
	ds_read_b64_tr_b16 a[28:29], v1 offset:25088
	ds_read_b64_tr_b16 a[30:31], v1 offset:29184
	s_add_i32 s29, s15, s44
	ds_read_b64_tr_b16 a[32:33], v1 offset:32768
	s_lshl_b32 s28, s45, 6
	s_mul_i32 s8, s29, s38
	ds_read_b64_tr_b16 a[34:35], v1 offset:36864
	v_mfma_f32_32x32x16_bf16 a[144:159], a[112:115], v[66:69], a[144:159]
	s_waitcnt lgkmcnt(8)
	s_barrier
	v_mfma_f32_32x32x16_bf16 a[160:175], a[116:119], v[62:65], a[160:175]
	ds_read_b64_tr_b16 v[30:31], v4
	s_add_i32 s8, s8, s28
	ds_read_b64_tr_b16 v[32:33], v4 offset:512
	s_add_i32 s8, s8, 48
	ds_read_b64_tr_b16 v[34:35], v4 offset:1024
	s_mul_i32 s8, s8, s3
	ds_read_b64_tr_b16 v[36:37], v4 offset:1536
	s_ashr_i32 s9, s8, 31
	ds_read_b64_tr_b16 v[38:39], v4 offset:2048
	s_lshl_b64 s[8:9], s[8:9], 1
	ds_read_b64_tr_b16 v[40:41], v4 offset:2560
	s_add_u32 s8, s6, s8
	ds_read_b64_tr_b16 v[42:43], v4 offset:3072
	s_addc_u32 s9, s7, s9
	ds_read_b64_tr_b16 v[44:45], v4 offset:3584
	v_mfma_f32_32x32x16_bf16 a[176:191], a[116:119], v[66:69], a[176:191]
	v_mul_f32_e32 v126, 0x3fb8aa3b, v126
	v_cvt_pk_bf16_f32 v110, v110, v111
	v_cvt_pk_bf16_f32 v111, v112, v113
	buffer_atomic_pk_add_bf16 v110, v5, s[8:11], 0 offen
	buffer_atomic_pk_add_bf16 v111, v6, s[8:11], 0 offen
	v_mfma_f32_32x32x16_bf16 a[192:207], a[120:123], v[62:65], a[192:207]
	ds_read_b64_tr_b16 v[46:47], v4 offset:4096
	ds_read_b64_tr_b16 v[48:49], v4 offset:4608
	ds_read_b64_tr_b16 v[50:51], v4 offset:5120
	ds_read_b64_tr_b16 v[52:53], v4 offset:5632
	ds_read_b64_tr_b16 a[36:37], v1 offset:33280
	ds_read_b64_tr_b16 a[38:39], v1 offset:37376
	s_add_i32 s71, s62, s0
	s_add_i32 s0, s36, s67
	ds_read_b64_tr_b16 a[40:41], v1 offset:40960
	s_mul_i32 s0, s0, s26
	ds_read_b64_tr_b16 a[42:43], v1 offset:45056
	v_mfma_f32_32x32x16_bf16 a[208:223], a[120:123], v[66:69], a[208:223]
	v_cvt_pk_bf16_f32 v114, v114, v115
	v_cvt_pk_bf16_f32 v115, v116, v117
	buffer_atomic_pk_add_bf16 v114, v7, s[8:11], 0 offen
	s_add_i32 s0, s0, s39
	buffer_atomic_pk_add_bf16 v115, v8, s[8:11], 0 offen
	v_mfma_f32_32x32x16_bf16 a[224:239], a[124:127], v[62:65], a[224:239]
	ds_read_b64_tr_b16 v[54:55], v4 offset:6144
	s_mul_i32 s40, s0, s60
	ds_read_b64_tr_b16 v[56:57], v4 offset:6656
	s_ashr_i32 s41, s40, 31
	ds_read_b64_tr_b16 v[58:59], v4 offset:7168
	s_lshl_b64 s[40:41], s[40:41], 1
	ds_read_b64_tr_b16 v[60:61], v4 offset:7680
	s_add_u32 s44, s52, s40
	ds_read_b64_tr_b16 a[44:45], v1 offset:41472
	v_add_u32_e32 v21, s71, v2
	s_addc_u32 s45, s53, s41
	s_add_i32 s28, s39, s64
	ds_read_b64_tr_b16 a[46:47], v1 offset:45568
	v_mfma_f32_32x32x16_bf16 a[240:255], a[124:127], v[66:69], a[240:255]
	s_waitcnt vmcnt(4) lgkmcnt(6)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[110:113], a[0:3], v[30:33], 0
	ds_read_b64_tr_b16 v[62:63], v1 offset:49152
	v_readfirstlane_b32 s29, v21
	v_add_u32_e32 v21, 0x1000, v21
	s_mul_i32 s28, s65, s28
	ds_read_b64_tr_b16 v[64:65], v1 offset:53248
	s_mov_b32 s47, s31
	v_readfirstlane_b32 s0, v21
	s_add_i32 s40, s28, s36
	ds_read_b64_tr_b16 v[66:67], v1 offset:49664
	s_mov_b32 m0, s29
	s_ashr_i32 s41, s40, 31
	ds_read_b64_tr_b16 v[68:69], v1 offset:53760
	v_mfma_f32_16x16x32_bf16 v[110:113], a[8:11], v[34:37], v[110:113]
	buffer_load_dwordx4 v14, s[44:47], 0 offen lds
	s_mov_b32 m0, s0
	s_add_i32 s78, s23, s72
	s_lshl_b64 s[40:41], s[40:41], 2
	buffer_load_dwordx4 v15, s[44:47], 0 offen lds
	v_mfma_f32_16x16x32_bf16 v[110:113], a[16:19], v[38:41], v[110:113]
	ds_read_b64_tr_b16 v[70:71], v1 offset:57344
	s_add_u32 s40, s56, s40
	ds_read_b64_tr_b16 v[72:73], v1 offset:61440
	s_addc_u32 s41, s57, s41
	s_add_i32 s28, s70, 0x1000
	ds_read_b64_tr_b16 v[74:75], v1 offset:57856
	s_mov_b32 m0, s78
	v_add_u32_e32 v27, s28, v11
	ds_read_b64_tr_b16 v[76:77], v1 offset:61952
	v_mfma_f32_16x16x32_bf16 v[110:113], a[24:27], v[42:45], v[110:113]
	buffer_load_dword v16, s[40:43], 0 offen lds
	v_mfma_f32_16x16x32_bf16 v[110:113], a[32:35], v[46:49], v[110:113]
	ds_read_b128 a[112:115], v27
	ds_read_b128 a[116:119], v27 offset:1024
	v_mfma_f32_16x16x32_bf16 v[110:113], a[40:43], v[50:53], v[110:113]
	s_waitcnt lgkmcnt(4)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[110:113], v[62:65], v[54:57], v[110:113]
	ds_read_b128 a[120:123], v27 offset:2048
	ds_read_b128 a[124:127], v27 offset:3072
	v_mfma_f32_16x16x32_bf16 v[110:113], v[70:73], v[58:61], v[110:113]
	v_mfma_f32_16x16x32_bf16 v[114:117], a[4:7], v[30:33], 0
	ds_read_b128 a[0:3], v10
	ds_read_b128 a[4:7], v10 offset:1024
	v_mfma_f32_16x16x32_bf16 v[114:117], a[12:15], v[34:37], v[114:117]
	v_mfma_f32_16x16x32_bf16 v[114:117], a[20:23], v[38:41], v[114:117]
	ds_read_b128 a[8:11], v10 offset:2048
	ds_read_b128 a[12:15], v10 offset:3072
	v_mfma_f32_16x16x32_bf16 v[114:117], a[28:31], v[42:45], v[114:117]
	v_mul_f32_e32 v110, 0x3db504f3, v110
	v_mul_f32_e32 v111, 0x3db504f3, v111
	v_mfma_f32_16x16x32_bf16 v[114:117], a[36:39], v[46:49], v[114:117]
	ds_read_b128 a[16:19], v10 offset:4096
	ds_read_b128 a[20:23], v10 offset:5120
	v_mfma_f32_16x16x32_bf16 v[114:117], a[44:47], v[50:53], v[114:117]
	v_mul_f32_e32 v112, 0x3db504f3, v112
	v_mul_f32_e32 v113, 0x3db504f3, v113
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_16x16x32_bf16 v[114:117], v[66:69], v[54:57], v[114:117]
	ds_read_b128 a[24:27], v10 offset:6144
	ds_read_b128 a[28:31], v10 offset:7168
	v_mfma_f32_16x16x32_bf16 v[114:117], v[74:77], v[58:61], v[114:117]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x32_bf16 v[46:49], a[112:115], a[0:3], 0
	ds_read_b128 a[32:35], v10 offset:8192
	ds_read_b128 a[36:39], v10 offset:9216
	v_mfma_f32_16x16x32_bf16 v[46:49], a[116:119], a[4:7], v[46:49]
	v_mfma_f32_16x16x32_bf16 v[46:49], a[120:123], a[8:11], v[46:49]
	ds_read_b128 a[40:43], v10 offset:10240
	ds_read_b128 a[44:47], v10 offset:11264
	v_mfma_f32_16x16x32_bf16 v[46:49], a[124:127], a[12:15], v[46:49]
	v_mul_f32_e32 v114, 0x3db504f3, v114
	v_mul_f32_e32 v115, 0x3db504f3, v115
	v_mfma_f32_16x16x32_bf16 v[50:53], a[112:115], a[16:19], 0
	ds_read_b128 v[62:65], v10 offset:12288
	s_add_i32 s72, s69, 0x1000
	ds_read_b128 v[66:69], v10 offset:13312
	v_mfma_f32_16x16x32_bf16 v[50:53], a[116:119], a[20:23], v[50:53]
	v_mul_f32_e32 v116, 0x3db504f3, v116
	v_mul_f32_e32 v117, 0x3db504f3, v117
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_bf16 v[50:53], a[120:123], a[24:27], v[50:53]
	ds_read_b128 v[70:73], v10 offset:14336
	v_add_u32_e32 v25, s72, v11
	ds_read_b128 v[74:77], v10 offset:15360
	v_mfma_f32_16x16x32_bf16 v[50:53], a[124:127], a[28:31], v[50:53]
	v_mul_f32_e32 v46, 0x3e0293ee, v46
	v_mul_f32_e32 v47, 0x3e0293ee, v47
	v_mul_f32_e32 v48, 0x3e0293ee, v48
	v_mul_f32_e32 v49, 0x3e0293ee, v49
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_bf16 v[54:57], a[112:115], a[32:35], 0
	ds_read_b128 v[78:81], v25
	ds_read_b128 v[82:85], v25 offset:1024
	v_mfma_f32_16x16x32_bf16 v[54:57], a[116:119], a[36:39], v[54:57]
	v_subrev_f32_dpp v46, v126, v46 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v47, v126, v47 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v48, v126, v48 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v49, v126, v49 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_bf16 v[54:57], a[120:123], a[40:43], v[54:57]
	ds_read_b128 v[86:89], v25 offset:2048
	v_add_u32_e32 v20, s72, v12
	ds_read_b128 v[90:93], v25 offset:3072
	v_mfma_f32_16x16x32_bf16 v[54:57], a[124:127], a[44:47], v[54:57]
	v_mul_f32_e32 v50, 0x3e0293ee, v50
	v_mul_f32_e32 v51, 0x3e0293ee, v51
	v_mul_f32_e32 v52, 0x3e0293ee, v52
	v_mul_f32_e32 v53, 0x3e0293ee, v53
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_bf16 v[58:61], a[112:115], v[62:65], 0
	ds_read_b64_tr_b16 v[94:95], v20
	ds_read_b64_tr_b16 v[96:97], v20 offset:256
	ds_read_b64_tr_b16 v[98:99], v20 offset:1024
	ds_read_b64_tr_b16 v[100:101], v20 offset:1280
	v_mfma_f32_16x16x32_bf16 v[58:61], a[116:119], v[66:69], v[58:61]
	v_subrev_f32_dpp v50, v126, v50 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v51, v126, v51 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v52, v126, v52 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v53, v126, v53 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_16x16x32_bf16 v[58:61], a[120:123], v[70:73], v[58:61]
	ds_read_b64_tr_b16 v[102:103], v20 offset:2048
	ds_read_b64_tr_b16 v[104:105], v20 offset:2304
	ds_read_b64_tr_b16 v[106:107], v20 offset:3072
	v_add_u32_e32 v26, s28, v12
	ds_read_b64_tr_b16 v[108:109], v20 offset:3328
	v_mfma_f32_16x16x32_bf16 v[58:61], a[124:127], v[74:77], v[58:61]
	v_mul_f32_e32 v54, 0x3e0293ee, v54
	v_mul_f32_e32 v55, 0x3e0293ee, v55
	v_mul_f32_e32 v56, 0x3e0293ee, v56
	v_mul_f32_e32 v57, 0x3e0293ee, v57
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_16x16x32_bf16 v[62:65], v[78:81], a[48:51], 0
	v_subrev_f32_dpp v54, v126, v54 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v55, v126, v55 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v56, v126, v56 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v57, v126, v57 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	v_mfma_f32_16x16x32_bf16 v[62:65], v[82:85], a[52:55], v[62:65]
	v_exp_f32_e32 v46, v46
	v_exp_f32_e32 v47, v47
	v_exp_f32_e32 v48, v48
	v_exp_f32_e32 v49, v49
	v_mfma_f32_16x16x32_bf16 v[62:65], v[86:89], a[56:59], v[62:65]
	ds_read_b64_tr_b16 a[112:113], v26
	ds_read_b64_tr_b16 a[114:115], v26 offset:256
	v_mfma_f32_16x16x32_bf16 v[62:65], v[90:93], a[60:63], v[62:65]
	v_exp_f32_e32 v50, v50
	v_exp_f32_e32 v51, v51
	v_exp_f32_e32 v52, v52
	v_exp_f32_e32 v53, v53
	v_mfma_f32_16x16x32_bf16 v[66:69], v[78:81], a[64:67], 0
	ds_read_b64_tr_b16 a[116:117], v26 offset:1024
	ds_read_b64_tr_b16 a[118:119], v26 offset:1280
	v_mfma_f32_16x16x32_bf16 v[66:69], v[82:85], a[68:71], v[66:69]
	v_mul_f32_e32 v58, 0x3e0293ee, v58
	v_mul_f32_e32 v59, 0x3e0293ee, v59
	v_mul_f32_e32 v60, 0x3e0293ee, v60
	v_mul_f32_e32 v61, 0x3e0293ee, v61
	v_mfma_f32_16x16x32_bf16 v[66:69], v[86:89], a[72:75], v[66:69]
	v_subrev_f32_dpp v58, v126, v58 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v59, v126, v59 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v60, v126, v60 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v61, v126, v61 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	v_mfma_f32_16x16x32_bf16 v[66:69], v[90:93], a[76:79], v[66:69]
	v_cvt_pk_bf16_f32 v118, v46, v47
	v_cvt_pk_bf16_f32 v119, v48, v49
	v_mfma_f32_16x16x32_bf16 v[70:73], v[78:81], a[80:83], 0
	v_exp_f32_e32 v54, v54
	v_exp_f32_e32 v55, v55
	v_exp_f32_e32 v56, v56
	v_exp_f32_e32 v57, v57
	v_mfma_f32_16x16x32_bf16 v[70:73], v[82:85], a[84:87], v[70:73]
	v_cvt_pk_bf16_f32 v120, v50, v51
	v_cvt_pk_bf16_f32 v121, v52, v53
	v_mfma_f32_16x16x32_bf16 v[70:73], v[86:89], a[88:91], v[70:73]
	ds_read_b64_tr_b16 a[120:121], v26 offset:2048
	ds_read_b64_tr_b16 a[122:123], v26 offset:2304
	v_mfma_f32_16x16x32_bf16 v[70:73], v[90:93], a[92:95], v[70:73]
	v_exp_f32_e32 v58, v58
	v_exp_f32_e32 v59, v59
	v_exp_f32_e32 v60, v60
	v_exp_f32_e32 v61, v61
	v_mfma_f32_16x16x32_bf16 v[74:77], v[78:81], a[96:99], 0
	ds_read_b64_tr_b16 a[124:125], v26 offset:3072
	ds_read_b64_tr_b16 a[126:127], v26 offset:3328
	v_mfma_f32_16x16x32_bf16 v[74:77], v[82:85], a[100:103], v[74:77]
	v_cvt_pk_bf16_f32 v122, v54, v55
	v_cvt_pk_bf16_f32 v123, v56, v57
	v_cvt_pk_bf16_f32 v124, v58, v59
	v_cvt_pk_bf16_f32 v125, v60, v61
	v_mfma_f32_16x16x32_bf16 v[74:77], v[86:89], a[104:107], v[74:77]
	v_permlane16_swap_b32_e32 v118, v120
	v_permlane16_swap_b32_e32 v119, v121
	v_permlane16_swap_b32_e32 v122, v124
	v_permlane16_swap_b32_e32 v123, v125
	v_mfma_f32_16x16x32_bf16 v[74:77], v[90:93], a[108:111], v[74:77]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x16_bf16 v[128:143], v[94:97], v[118:121], v[128:143]
	ds_read_b64_tr_b16 a[0:1], v1
	ds_read_b64_tr_b16 a[2:3], v1 offset:4096
	ds_read_b64_tr_b16 a[4:5], v1 offset:512
	ds_read_b64_tr_b16 a[6:7], v1 offset:4608
	v_mfma_f32_32x32x16_bf16 v[144:159], v[94:97], v[122:125], v[144:159]
	v_subrev_f32_dpp v62, v127, v62 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v63, v127, v63 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v64, v127, v64 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v65, v127, v65 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v66, v127, v66 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v67, v127, v67 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v68, v127, v68 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v69, v127, v69 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	v_mfma_f32_32x32x16_bf16 v[160:175], v[98:101], v[118:121], v[160:175]
	ds_read_b64_tr_b16 a[8:9], v1 offset:8192
	ds_read_b64_tr_b16 a[10:11], v1 offset:12288
	ds_read_b64_tr_b16 a[12:13], v1 offset:8704
	ds_read_b64_tr_b16 a[14:15], v1 offset:12800
	v_mfma_f32_32x32x16_bf16 v[176:191], v[98:101], v[122:125], v[176:191]
	v_mul_f32_e32 v62, v46, v62
	v_mul_f32_e32 v63, v47, v63
	v_mul_f32_e32 v64, v48, v64
	v_mul_f32_e32 v65, v49, v65
	v_mul_f32_e32 v66, v50, v66
	v_mul_f32_e32 v67, v51, v67
	v_mul_f32_e32 v68, v52, v68
	v_mul_f32_e32 v69, v53, v69
	v_cvt_pk_bf16_f32 v62, v62, v63
	v_cvt_pk_bf16_f32 v63, v64, v65
	v_cvt_pk_bf16_f32 v64, v66, v67
	v_cvt_pk_bf16_f32 v65, v68, v69
	v_subrev_f32_dpp v70, v127, v70 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v71, v127, v71 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v72, v127, v72 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v73, v127, v73 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	v_mfma_f32_32x32x16_bf16 v[192:207], v[102:105], v[118:121], v[192:207]
	ds_read_b64_tr_b16 a[16:17], v1 offset:16384
	ds_read_b64_tr_b16 a[18:19], v1 offset:20480
	s_add_i32 s73, s74, 0x80
	ds_write_b64 v9, v[62:63]
	s_add_i32 s76, s75, 0x80
	v_add_u32_e32 v21, s73, v13
	ds_write_b64 v9, v[64:65] offset:512
	v_mfma_f32_32x32x16_bf16 v[208:223], v[102:105], v[122:125], v[208:223]
	v_subrev_f32_dpp v74, v127, v74 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v75, v127, v75 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v76, v127, v76 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v77, v127, v77 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	v_mul_f32_e32 v70, v54, v70
	v_mul_f32_e32 v71, v55, v71
	v_mul_f32_e32 v72, v56, v72
	v_mul_f32_e32 v73, v57, v73
	v_mul_f32_e32 v74, v58, v74
	v_mul_f32_e32 v75, v59, v75
	v_mul_f32_e32 v76, v60, v76
	v_mul_f32_e32 v77, v61, v77
	v_cvt_pk_bf16_f32 v66, v70, v71
	v_cvt_pk_bf16_f32 v67, v72, v73
	v_cvt_pk_bf16_f32 v68, v74, v75
	v_cvt_pk_bf16_f32 v69, v76, v77
	v_mfma_f32_32x32x16_bf16 v[224:239], v[106:109], v[118:121], v[224:239]
	ds_read_b32 v126, v21

	v_add_u32_e32 v23, s76, v13
	ds_read_b32 v127, v23

	ds_write_b64 v9, v[66:67] offset:1024
	ds_write_b64 v9, v[68:69] offset:1536
	v_mfma_f32_32x32x16_bf16 v[240:255], v[106:109], v[122:125], v[240:255]
	v_permlane16_swap_b32_e32 v62, v64
	v_permlane16_swap_b32_e32 v63, v65
	v_permlane16_swap_b32_e32 v66, v68
	v_permlane16_swap_b32_e32 v67, v69
	s_waitcnt lgkmcnt(12)
	v_mfma_f32_32x32x16_bf16 a[128:143], a[112:115], v[62:65], a[128:143]
	ds_read_b64_tr_b16 a[20:21], v1 offset:16896
	ds_read_b64_tr_b16 a[22:23], v1 offset:20992
	ds_read_b64_tr_b16 a[24:25], v1 offset:24576
	ds_read_b64_tr_b16 a[26:27], v1 offset:28672
	ds_read_b64_tr_b16 a[28:29], v1 offset:25088
	ds_read_b64_tr_b16 a[30:31], v1 offset:29184
	ds_read_b64_tr_b16 a[32:33], v1 offset:32768
	s_sub_i32 s72, s17, 64
	s_add_i32 s68, s15, s68
	ds_read_b64_tr_b16 a[34:35], v1 offset:36864
	v_mfma_f32_32x32x16_bf16 a[144:159], a[112:115], v[66:69], a[144:159]
	s_waitcnt lgkmcnt(8)
	s_barrier
	v_mfma_f32_32x32x16_bf16 a[160:175], a[116:119], v[62:65], a[160:175]
	ds_read_b64_tr_b16 v[30:31], v4
	s_and_b32 s28, s72, 0x1fc0
	s_mul_i32 s68, s68, s38
	ds_read_b64_tr_b16 v[32:33], v4 offset:512
	s_add_i32 s68, s68, s28
	ds_read_b64_tr_b16 v[34:35], v4 offset:1024
	s_mul_i32 s72, s68, s3
	ds_read_b64_tr_b16 v[36:37], v4 offset:1536
	s_ashr_i32 s73, s72, 31
	ds_read_b64_tr_b16 v[38:39], v4 offset:2048
	s_lshl_b64 s[72:73], s[72:73], 1
	ds_read_b64_tr_b16 v[40:41], v4 offset:2560
	s_add_u32 s8, s6, s72
	ds_read_b64_tr_b16 v[42:43], v4 offset:3072
	s_addc_u32 s9, s7, s73
	ds_read_b64_tr_b16 v[44:45], v4 offset:3584
	v_mfma_f32_32x32x16_bf16 a[176:191], a[116:119], v[66:69], a[176:191]
	v_mul_f32_e32 v126, 0x3fb8aa3b, v126
	v_cvt_pk_bf16_f32 v110, v110, v111
	v_cvt_pk_bf16_f32 v111, v112, v113
	buffer_atomic_pk_add_bf16 v110, v5, s[8:11], 0 offen
	buffer_atomic_pk_add_bf16 v111, v6, s[8:11], 0 offen
	v_mfma_f32_32x32x16_bf16 a[192:207], a[120:123], v[62:65], a[192:207]
	ds_read_b64_tr_b16 v[46:47], v4 offset:4096
	ds_read_b64_tr_b16 v[48:49], v4 offset:4608
	ds_read_b64_tr_b16 v[50:51], v4 offset:5120
	ds_read_b64_tr_b16 v[52:53], v4 offset:5632
	ds_read_b64_tr_b16 a[36:37], v1 offset:33280
	ds_read_b64_tr_b16 a[38:39], v1 offset:37376
	ds_read_b64_tr_b16 a[40:41], v1 offset:40960
	ds_read_b64_tr_b16 a[42:43], v1 offset:45056
	v_mfma_f32_32x32x16_bf16 a[208:223], a[120:123], v[66:69], a[208:223]
	v_cvt_pk_bf16_f32 v114, v114, v115
	v_cvt_pk_bf16_f32 v115, v116, v117
	buffer_atomic_pk_add_bf16 v114, v7, s[8:11], 0 offen
	s_add_i32 s28, s36, s19
	buffer_atomic_pk_add_bf16 v115, v8, s[8:11], 0 offen
	v_mfma_f32_32x32x16_bf16 a[224:239], a[124:127], v[62:65], a[224:239]
	ds_read_b64_tr_b16 v[54:55], v4 offset:6144
	v_add_u32_e32 v17, s1, v11
	s_addk_i32 s1, 0x2000
	s_mul_i32 s28, s28, s22
	ds_read_b64_tr_b16 v[56:57], v4 offset:6656
	v_add_u32_e32 v22, s1, v2
	s_add_i32 s1, s28, s39
	ds_read_b64_tr_b16 v[58:59], v4 offset:7168
	s_mul_i32 s76, s1, s33
	ds_read_b64_tr_b16 v[60:61], v4 offset:7680
	s_ashr_i32 s77, s76, 31
	ds_read_b64_tr_b16 a[44:45], v1 offset:41472
	s_lshl_b64 s[76:77], s[76:77], 1
	ds_read_b64_tr_b16 a[46:47], v1 offset:45568
	v_mfma_f32_32x32x16_bf16 a[240:255], a[124:127], v[66:69], a[240:255]
	s_waitcnt vmcnt(4) lgkmcnt(6)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[110:113], a[0:3], v[30:33], 0
	ds_read_b64_tr_b16 v[62:63], v1 offset:49152
	v_readfirstlane_b32 s73, v22
	v_add_u32_e32 v22, 0x1000, v22
	s_add_u32 s28, s50, s76
	ds_read_b64_tr_b16 v[64:65], v1 offset:53248
	v_readfirstlane_b32 s72, v22
	s_addc_u32 s29, s51, s77
	ds_read_b64_tr_b16 v[66:67], v1 offset:49664
	s_mov_b32 m0, s73
	ds_read_b64_tr_b16 v[68:69], v1 offset:53760
	v_mfma_f32_16x16x32_bf16 v[110:113], a[8:11], v[34:37], v[110:113]
	s_add_i32 s1, s70, 0x2000
	buffer_load_dwordx4 v14, s[28:31], 0 offen lds
	s_mov_b32 m0, s72
	v_add_u32_e32 v28, s1, v11
	buffer_load_dwordx4 v15, s[28:31], 0 offen lds
	v_mfma_f32_16x16x32_bf16 v[110:113], a[16:19], v[38:41], v[110:113]
	ds_read_b64_tr_b16 v[70:71], v1 offset:57344
	ds_read_b64_tr_b16 v[72:73], v1 offset:61440
	ds_read_b64_tr_b16 v[74:75], v1 offset:57856
	ds_read_b64_tr_b16 v[76:77], v1 offset:61952
	v_mfma_f32_16x16x32_bf16 v[110:113], a[24:27], v[42:45], v[110:113]
	v_mfma_f32_16x16x32_bf16 v[110:113], a[32:35], v[46:49], v[110:113]
	ds_read_b128 a[112:115], v28
	ds_read_b128 a[116:119], v28 offset:1024
	v_mfma_f32_16x16x32_bf16 v[110:113], a[40:43], v[50:53], v[110:113]
	s_waitcnt lgkmcnt(4)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[110:113], v[62:65], v[54:57], v[110:113]
	ds_read_b128 a[120:123], v28 offset:2048
	ds_read_b128 a[124:127], v28 offset:3072
	v_mfma_f32_16x16x32_bf16 v[110:113], v[70:73], v[58:61], v[110:113]
	v_mfma_f32_16x16x32_bf16 v[114:117], a[4:7], v[30:33], 0
	ds_read_b128 a[0:3], v10
	ds_read_b128 a[4:7], v10 offset:1024
	v_mfma_f32_16x16x32_bf16 v[114:117], a[12:15], v[34:37], v[114:117]
	v_mfma_f32_16x16x32_bf16 v[114:117], a[20:23], v[38:41], v[114:117]
	ds_read_b128 a[8:11], v10 offset:2048
	ds_read_b128 a[12:15], v10 offset:3072
	v_mfma_f32_16x16x32_bf16 v[114:117], a[28:31], v[42:45], v[114:117]
	v_mul_f32_e32 v110, 0x3db504f3, v110
	v_mul_f32_e32 v111, 0x3db504f3, v111
	v_mfma_f32_16x16x32_bf16 v[114:117], a[36:39], v[46:49], v[114:117]
	ds_read_b128 a[16:19], v10 offset:4096
	ds_read_b128 a[20:23], v10 offset:5120
	v_mfma_f32_16x16x32_bf16 v[114:117], a[44:47], v[50:53], v[114:117]
	v_mul_f32_e32 v112, 0x3db504f3, v112
	v_mul_f32_e32 v113, 0x3db504f3, v113
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_16x16x32_bf16 v[114:117], v[66:69], v[54:57], v[114:117]
	ds_read_b128 a[24:27], v10 offset:6144
	ds_read_b128 a[28:31], v10 offset:7168
	v_mfma_f32_16x16x32_bf16 v[114:117], v[74:77], v[58:61], v[114:117]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x32_bf16 v[46:49], a[112:115], a[0:3], 0
	ds_read_b128 a[32:35], v10 offset:8192
	ds_read_b128 a[36:39], v10 offset:9216
	v_mfma_f32_16x16x32_bf16 v[46:49], a[116:119], a[4:7], v[46:49]
	v_mfma_f32_16x16x32_bf16 v[46:49], a[120:123], a[8:11], v[46:49]
	ds_read_b128 a[40:43], v10 offset:10240
	ds_read_b128 a[44:47], v10 offset:11264
	v_mfma_f32_16x16x32_bf16 v[46:49], a[124:127], a[12:15], v[46:49]
	v_mul_f32_e32 v114, 0x3db504f3, v114
	v_mul_f32_e32 v115, 0x3db504f3, v115
	v_mfma_f32_16x16x32_bf16 v[50:53], a[112:115], a[16:19], 0
	ds_read_b128 v[62:65], v10 offset:12288
	s_add_i32 s76, s69, 0x2000
	ds_read_b128 v[66:69], v10 offset:13312
	v_mfma_f32_16x16x32_bf16 v[50:53], a[116:119], a[20:23], v[50:53]
	v_mul_f32_e32 v116, 0x3db504f3, v116
	v_mul_f32_e32 v117, 0x3db504f3, v117
	v_mfma_f32_16x16x32_bf16 v[50:53], a[120:123], a[24:27], v[50:53]
	ds_read_b128 v[70:73], v10 offset:14336
	v_add_u32_e32 v22, s76, v11
	ds_read_b128 v[74:77], v10 offset:15360
	v_mfma_f32_16x16x32_bf16 v[50:53], a[124:127], a[28:31], v[50:53]
	v_mul_f32_e32 v46, 0x3e0293ee, v46
	v_mul_f32_e32 v47, 0x3e0293ee, v47
	v_mul_f32_e32 v48, 0x3e0293ee, v48
	v_mul_f32_e32 v49, 0x3e0293ee, v49
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_bf16 v[54:57], a[112:115], a[32:35], 0
	ds_read_b128 v[78:81], v22
	ds_read_b128 v[82:85], v22 offset:1024
	v_mfma_f32_16x16x32_bf16 v[54:57], a[116:119], a[36:39], v[54:57]
	v_subrev_f32_dpp v46, v126, v46 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v47, v126, v47 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v48, v126, v48 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v49, v126, v49 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_bf16 v[54:57], a[120:123], a[40:43], v[54:57]
	ds_read_b128 v[86:89], v22 offset:2048
	v_add_u32_e32 v24, s76, v12
	ds_read_b128 v[90:93], v22 offset:3072
	v_mfma_f32_16x16x32_bf16 v[54:57], a[124:127], a[44:47], v[54:57]
	v_mul_f32_e32 v50, 0x3e0293ee, v50
	v_mul_f32_e32 v51, 0x3e0293ee, v51
	v_mul_f32_e32 v52, 0x3e0293ee, v52
	v_mul_f32_e32 v53, 0x3e0293ee, v53
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_bf16 v[58:61], a[112:115], v[62:65], 0
	ds_read_b64_tr_b16 v[94:95], v24
	ds_read_b64_tr_b16 v[96:97], v24 offset:256
	ds_read_b64_tr_b16 v[98:99], v24 offset:1024
	ds_read_b64_tr_b16 v[100:101], v24 offset:1280
	v_mfma_f32_16x16x32_bf16 v[58:61], a[116:119], v[66:69], v[58:61]
	v_subrev_f32_dpp v50, v126, v50 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v51, v126, v51 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v52, v126, v52 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v53, v126, v53 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_16x16x32_bf16 v[58:61], a[120:123], v[70:73], v[58:61]
	ds_read_b64_tr_b16 v[102:103], v24 offset:2048
	ds_read_b64_tr_b16 v[104:105], v24 offset:2304
	ds_read_b64_tr_b16 v[106:107], v24 offset:3072
	v_add_u32_e32 v25, s1, v12
	ds_read_b64_tr_b16 v[108:109], v24 offset:3328
	v_mfma_f32_16x16x32_bf16 v[58:61], a[124:127], v[74:77], v[58:61]
	v_mul_f32_e32 v54, 0x3e0293ee, v54
	v_mul_f32_e32 v55, 0x3e0293ee, v55
	v_mul_f32_e32 v56, 0x3e0293ee, v56
	v_mul_f32_e32 v57, 0x3e0293ee, v57
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_16x16x32_bf16 v[62:65], v[78:81], a[48:51], 0
	v_subrev_f32_dpp v54, v126, v54 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v55, v126, v55 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v56, v126, v56 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v57, v126, v57 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	v_mfma_f32_16x16x32_bf16 v[62:65], v[82:85], a[52:55], v[62:65]
	v_exp_f32_e32 v46, v46
	v_exp_f32_e32 v47, v47
	v_exp_f32_e32 v48, v48
	v_exp_f32_e32 v49, v49
	v_mfma_f32_16x16x32_bf16 v[62:65], v[86:89], a[56:59], v[62:65]
	ds_read_b64_tr_b16 a[112:113], v25
	ds_read_b64_tr_b16 a[114:115], v25 offset:256
	v_mfma_f32_16x16x32_bf16 v[62:65], v[90:93], a[60:63], v[62:65]
	v_exp_f32_e32 v50, v50
	v_exp_f32_e32 v51, v51
	v_exp_f32_e32 v52, v52
	v_exp_f32_e32 v53, v53
	v_mfma_f32_16x16x32_bf16 v[66:69], v[78:81], a[64:67], 0
	ds_read_b64_tr_b16 a[116:117], v25 offset:1024
	ds_read_b64_tr_b16 a[118:119], v25 offset:1280
	v_mfma_f32_16x16x32_bf16 v[66:69], v[82:85], a[68:71], v[66:69]
	v_mul_f32_e32 v58, 0x3e0293ee, v58
	v_mul_f32_e32 v59, 0x3e0293ee, v59
	v_mul_f32_e32 v60, 0x3e0293ee, v60
	v_mul_f32_e32 v61, 0x3e0293ee, v61
	v_mfma_f32_16x16x32_bf16 v[66:69], v[86:89], a[72:75], v[66:69]
	v_subrev_f32_dpp v58, v126, v58 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v59, v126, v59 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v60, v126, v60 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v61, v126, v61 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	v_mfma_f32_16x16x32_bf16 v[66:69], v[90:93], a[76:79], v[66:69]
	v_cvt_pk_bf16_f32 v118, v46, v47
	v_cvt_pk_bf16_f32 v119, v48, v49
	v_mfma_f32_16x16x32_bf16 v[70:73], v[78:81], a[80:83], 0
	v_exp_f32_e32 v54, v54
	v_exp_f32_e32 v55, v55
	v_exp_f32_e32 v56, v56
	v_exp_f32_e32 v57, v57
	v_mfma_f32_16x16x32_bf16 v[70:73], v[82:85], a[84:87], v[70:73]
	v_cvt_pk_bf16_f32 v120, v50, v51
	v_cvt_pk_bf16_f32 v121, v52, v53
	v_mfma_f32_16x16x32_bf16 v[70:73], v[86:89], a[88:91], v[70:73]
	ds_read_b64_tr_b16 a[120:121], v25 offset:2048
	ds_read_b64_tr_b16 a[122:123], v25 offset:2304
	v_mfma_f32_16x16x32_bf16 v[70:73], v[90:93], a[92:95], v[70:73]
	v_exp_f32_e32 v58, v58
	v_exp_f32_e32 v59, v59
	v_exp_f32_e32 v60, v60
	v_exp_f32_e32 v61, v61
	v_mfma_f32_16x16x32_bf16 v[74:77], v[78:81], a[96:99], 0
	ds_read_b64_tr_b16 a[124:125], v25 offset:3072
	ds_read_b64_tr_b16 a[126:127], v25 offset:3328
	v_mfma_f32_16x16x32_bf16 v[74:77], v[82:85], a[100:103], v[74:77]
	v_cvt_pk_bf16_f32 v122, v54, v55
	v_cvt_pk_bf16_f32 v123, v56, v57
	v_cvt_pk_bf16_f32 v124, v58, v59
	v_cvt_pk_bf16_f32 v125, v60, v61
	v_mfma_f32_16x16x32_bf16 v[74:77], v[86:89], a[104:107], v[74:77]
	v_permlane16_swap_b32_e32 v118, v120
	v_permlane16_swap_b32_e32 v119, v121
	v_permlane16_swap_b32_e32 v122, v124
	v_permlane16_swap_b32_e32 v123, v125
	v_mfma_f32_16x16x32_bf16 v[74:77], v[90:93], a[108:111], v[74:77]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x16_bf16 v[128:143], v[94:97], v[118:121], v[128:143]
	ds_read_b64_tr_b16 a[0:1], v1
	ds_read_b64_tr_b16 a[2:3], v1 offset:4096
	ds_read_b64_tr_b16 a[4:5], v1 offset:512
	ds_read_b64_tr_b16 a[6:7], v1 offset:4608
	v_mfma_f32_32x32x16_bf16 v[144:159], v[94:97], v[122:125], v[144:159]
	v_subrev_f32_dpp v62, v127, v62 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v63, v127, v63 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v64, v127, v64 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v65, v127, v65 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v66, v127, v66 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v67, v127, v67 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v68, v127, v68 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v69, v127, v69 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	v_mfma_f32_32x32x16_bf16 v[160:175], v[98:101], v[118:121], v[160:175]
	ds_read_b64_tr_b16 a[8:9], v1 offset:8192
	ds_read_b64_tr_b16 a[10:11], v1 offset:12288
	ds_read_b64_tr_b16 a[12:13], v1 offset:8704
	ds_read_b64_tr_b16 a[14:15], v1 offset:12800
	v_mfma_f32_32x32x16_bf16 v[176:191], v[98:101], v[122:125], v[176:191]
	v_mul_f32_e32 v62, v46, v62
	v_mul_f32_e32 v63, v47, v63
	v_mul_f32_e32 v64, v48, v64
	v_mul_f32_e32 v65, v49, v65
	v_mul_f32_e32 v66, v50, v66
	v_mul_f32_e32 v67, v51, v67
	v_mul_f32_e32 v68, v52, v68
	v_mul_f32_e32 v69, v53, v69
	v_cvt_pk_bf16_f32 v62, v62, v63
	v_cvt_pk_bf16_f32 v63, v64, v65
	v_cvt_pk_bf16_f32 v64, v66, v67
	v_cvt_pk_bf16_f32 v65, v68, v69
	v_subrev_f32_dpp v70, v127, v70 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v71, v127, v71 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v72, v127, v72 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v73, v127, v73 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	v_mfma_f32_32x32x16_bf16 v[192:207], v[102:105], v[118:121], v[192:207]
	ds_read_b64_tr_b16 a[16:17], v1 offset:16384
	ds_read_b64_tr_b16 a[18:19], v1 offset:20480
	s_addk_i32 s74, 0xc0
	ds_write_b64 v9, v[62:63]
	s_addk_i32 s75, 0xc0
	v_add_u32_e32 v20, s74, v13
	ds_write_b64 v9, v[64:65] offset:512
	v_mfma_f32_32x32x16_bf16 v[208:223], v[102:105], v[122:125], v[208:223]
	v_subrev_f32_dpp v74, v127, v74 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v75, v127, v75 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v76, v127, v76 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v77, v127, v77 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	v_mul_f32_e32 v70, v54, v70
	v_mul_f32_e32 v71, v55, v71
	v_mul_f32_e32 v72, v56, v72
	v_mul_f32_e32 v73, v57, v73
	v_mul_f32_e32 v74, v58, v74
	v_mul_f32_e32 v75, v59, v75
	v_mul_f32_e32 v76, v60, v76
	v_mul_f32_e32 v77, v61, v77
	v_cvt_pk_bf16_f32 v66, v70, v71
	v_cvt_pk_bf16_f32 v67, v72, v73
	v_cvt_pk_bf16_f32 v68, v74, v75
	v_cvt_pk_bf16_f32 v69, v76, v77
	v_mfma_f32_32x32x16_bf16 v[224:239], v[106:109], v[118:121], v[224:239]
	ds_read_b32 v126, v20

	v_add_u32_e32 v26, s75, v13
	ds_read_b32 v127, v26

	ds_write_b64 v9, v[66:67] offset:1024
	ds_write_b64 v9, v[68:69] offset:1536
	v_mfma_f32_32x32x16_bf16 v[240:255], v[106:109], v[122:125], v[240:255]
	v_permlane16_swap_b32_e32 v62, v64
	v_permlane16_swap_b32_e32 v63, v65
	v_permlane16_swap_b32_e32 v66, v68
	v_permlane16_swap_b32_e32 v67, v69
	s_waitcnt lgkmcnt(12)
	v_mfma_f32_32x32x16_bf16 a[128:143], a[112:115], v[62:65], a[128:143]
	ds_read_b64_tr_b16 a[20:21], v1 offset:16896
	ds_read_b64_tr_b16 a[22:23], v1 offset:20992
	ds_read_b64_tr_b16 a[24:25], v1 offset:24576
	ds_read_b64_tr_b16 a[26:27], v1 offset:28672
	ds_read_b64_tr_b16 a[28:29], v1 offset:25088
	ds_read_b64_tr_b16 a[30:31], v1 offset:29184
	ds_read_b64_tr_b16 a[32:33], v1 offset:32768
	ds_read_b64_tr_b16 a[34:35], v1 offset:36864
	v_mfma_f32_32x32x16_bf16 a[144:159], a[112:115], v[66:69], a[144:159]
	s_waitcnt lgkmcnt(8)
	s_barrier
	v_mfma_f32_32x32x16_bf16 a[160:175], a[116:119], v[62:65], a[160:175]
	ds_read_b64_tr_b16 v[30:31], v4
	ds_read_b64_tr_b16 v[32:33], v4 offset:512
	s_add_i32 s77, s68, 16
	ds_read_b64_tr_b16 v[34:35], v4 offset:1024
	s_mul_i32 s0, s77, s3
	ds_read_b64_tr_b16 v[36:37], v4 offset:1536
	s_ashr_i32 s1, s0, 31
	ds_read_b64_tr_b16 v[38:39], v4 offset:2048
	s_lshl_b64 s[0:1], s[0:1], 1
	ds_read_b64_tr_b16 v[40:41], v4 offset:2560
	s_add_u32 s8, s6, s0
	ds_read_b64_tr_b16 v[42:43], v4 offset:3072
	s_addc_u32 s9, s7, s1
	ds_read_b64_tr_b16 v[44:45], v4 offset:3584
	v_mfma_f32_32x32x16_bf16 a[176:191], a[116:119], v[66:69], a[176:191]
	v_mul_f32_e32 v126, 0x3fb8aa3b, v126
	v_cvt_pk_bf16_f32 v110, v110, v111
	v_cvt_pk_bf16_f32 v111, v112, v113
	buffer_atomic_pk_add_bf16 v110, v5, s[8:11], 0 offen
	buffer_atomic_pk_add_bf16 v111, v6, s[8:11], 0 offen
	v_mfma_f32_32x32x16_bf16 a[192:207], a[120:123], v[62:65], a[192:207]
	ds_read_b64_tr_b16 v[46:47], v4 offset:4096
	ds_read_b64_tr_b16 v[48:49], v4 offset:4608
	ds_read_b64_tr_b16 v[50:51], v4 offset:5120
	ds_read_b64_tr_b16 v[52:53], v4 offset:5632
	ds_read_b64_tr_b16 a[36:37], v1 offset:33280
	ds_read_b64_tr_b16 a[38:39], v1 offset:37376
	ds_read_b64_tr_b16 a[40:41], v1 offset:40960
	ds_read_b64_tr_b16 a[42:43], v1 offset:45056
	v_mfma_f32_32x32x16_bf16 a[208:223], a[120:123], v[66:69], a[208:223]
	v_cvt_pk_bf16_f32 v114, v114, v115
	v_cvt_pk_bf16_f32 v115, v116, v117
	buffer_atomic_pk_add_bf16 v114, v7, s[8:11], 0 offen
	s_add_i32 s0, s36, s25
	buffer_atomic_pk_add_bf16 v115, v8, s[8:11], 0 offen
	v_mfma_f32_32x32x16_bf16 a[224:239], a[124:127], v[62:65], a[224:239]
	ds_read_b64_tr_b16 v[54:55], v4 offset:6144
	s_mul_i32 s0, s0, s26
	ds_read_b64_tr_b16 v[56:57], v4 offset:6656
	s_add_i32 s0, s0, s39
	ds_read_b64_tr_b16 v[58:59], v4 offset:7168
	s_mul_i32 s0, s0, s60
	ds_read_b64_tr_b16 v[60:61], v4 offset:7680
	s_addk_i32 s71, 0x2000
	s_ashr_i32 s1, s0, 31
	ds_read_b64_tr_b16 a[44:45], v1 offset:41472
	v_add_u32_e32 v21, s71, v2
	s_lshl_b64 s[0:1], s[0:1], 1
	ds_read_b64_tr_b16 a[46:47], v1 offset:45568
	v_mfma_f32_32x32x16_bf16 a[240:255], a[124:127], v[66:69], a[240:255]
	s_waitcnt vmcnt(4) lgkmcnt(6)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[110:113], a[0:3], v[30:33], 0
	ds_read_b64_tr_b16 v[62:63], v1 offset:49152
	v_readfirstlane_b32 s36, v21
	v_add_u32_e32 v21, 0x1000, v21
	s_add_u32 s44, s52, s0
	ds_read_b64_tr_b16 v[64:65], v1 offset:53248
	v_readfirstlane_b32 s39, v21
	s_addc_u32 s45, s53, s1
	ds_read_b64_tr_b16 v[66:67], v1 offset:49664
	s_mov_b32 m0, s36
	ds_read_b64_tr_b16 v[68:69], v1 offset:53760
	v_mfma_f32_16x16x32_bf16 v[110:113], a[8:11], v[34:37], v[110:113]
	s_addk_i32 s70, 0x3000
	buffer_load_dwordx4 v14, s[44:47], 0 offen lds
	s_mov_b32 m0, s39
	v_add_u32_e32 v27, s70, v11
	buffer_load_dwordx4 v15, s[44:47], 0 offen lds
	v_mfma_f32_16x16x32_bf16 v[110:113], a[16:19], v[38:41], v[110:113]
	ds_read_b64_tr_b16 v[70:71], v1 offset:57344
	ds_read_b64_tr_b16 v[72:73], v1 offset:61440
	ds_read_b64_tr_b16 v[74:75], v1 offset:57856
	ds_read_b64_tr_b16 v[76:77], v1 offset:61952
	v_mfma_f32_16x16x32_bf16 v[110:113], a[24:27], v[42:45], v[110:113]
	v_mfma_f32_16x16x32_bf16 v[110:113], a[32:35], v[46:49], v[110:113]
	ds_read_b128 a[112:115], v27
	ds_read_b128 a[116:119], v27 offset:1024
	v_mfma_f32_16x16x32_bf16 v[110:113], a[40:43], v[50:53], v[110:113]
	s_waitcnt lgkmcnt(4)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[110:113], v[62:65], v[54:57], v[110:113]
	ds_read_b128 a[120:123], v27 offset:2048
	ds_read_b128 a[124:127], v27 offset:3072
	v_mfma_f32_16x16x32_bf16 v[110:113], v[70:73], v[58:61], v[110:113]
	v_mfma_f32_16x16x32_bf16 v[114:117], a[4:7], v[30:33], 0
	ds_read_b128 a[0:3], v10
	ds_read_b128 a[4:7], v10 offset:1024
	v_mfma_f32_16x16x32_bf16 v[114:117], a[12:15], v[34:37], v[114:117]
	v_mfma_f32_16x16x32_bf16 v[114:117], a[20:23], v[38:41], v[114:117]
	ds_read_b128 a[8:11], v10 offset:2048
	ds_read_b128 a[12:15], v10 offset:3072
	v_mfma_f32_16x16x32_bf16 v[114:117], a[28:31], v[42:45], v[114:117]
	v_mul_f32_e32 v110, 0x3db504f3, v110
	v_mul_f32_e32 v111, 0x3db504f3, v111
	v_mfma_f32_16x16x32_bf16 v[114:117], a[36:39], v[46:49], v[114:117]
	ds_read_b128 a[16:19], v10 offset:4096
	ds_read_b128 a[20:23], v10 offset:5120
	v_mfma_f32_16x16x32_bf16 v[114:117], a[44:47], v[50:53], v[114:117]
	v_mul_f32_e32 v112, 0x3db504f3, v112
	v_mul_f32_e32 v113, 0x3db504f3, v113
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_16x16x32_bf16 v[114:117], v[66:69], v[54:57], v[114:117]
	ds_read_b128 a[24:27], v10 offset:6144
	ds_read_b128 a[28:31], v10 offset:7168
	v_mfma_f32_16x16x32_bf16 v[114:117], v[74:77], v[58:61], v[114:117]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x32_bf16 v[46:49], a[112:115], a[0:3], 0
	ds_read_b128 a[32:35], v10 offset:8192
	ds_read_b128 a[36:39], v10 offset:9216
	v_mfma_f32_16x16x32_bf16 v[46:49], a[116:119], a[4:7], v[46:49]
	v_mfma_f32_16x16x32_bf16 v[46:49], a[120:123], a[8:11], v[46:49]
	ds_read_b128 a[40:43], v10 offset:10240
	ds_read_b128 a[44:47], v10 offset:11264
	v_mfma_f32_16x16x32_bf16 v[46:49], a[124:127], a[12:15], v[46:49]
	v_mul_f32_e32 v114, 0x3db504f3, v114
	v_mul_f32_e32 v115, 0x3db504f3, v115
	v_mfma_f32_16x16x32_bf16 v[50:53], a[112:115], a[16:19], 0
	ds_read_b128 v[62:65], v10 offset:12288
	s_addk_i32 s69, 0x3000
	ds_read_b128 v[66:69], v10 offset:13312
	v_mfma_f32_16x16x32_bf16 v[50:53], a[116:119], a[20:23], v[50:53]
	v_mul_f32_e32 v116, 0x3db504f3, v116
	v_mul_f32_e32 v117, 0x3db504f3, v117
	v_mfma_f32_16x16x32_bf16 v[50:53], a[120:123], a[24:27], v[50:53]
	ds_read_b128 v[70:73], v10 offset:14336
	v_add_u32_e32 v23, s69, v11
	ds_read_b128 v[74:77], v10 offset:15360
	v_mfma_f32_16x16x32_bf16 v[50:53], a[124:127], a[28:31], v[50:53]
	v_mul_f32_e32 v46, 0x3e0293ee, v46
	v_mul_f32_e32 v47, 0x3e0293ee, v47
	v_mul_f32_e32 v48, 0x3e0293ee, v48
	v_mul_f32_e32 v49, 0x3e0293ee, v49
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_bf16 v[54:57], a[112:115], a[32:35], 0
	ds_read_b128 v[78:81], v23
	ds_read_b128 v[82:85], v23 offset:1024
	v_mfma_f32_16x16x32_bf16 v[54:57], a[116:119], a[36:39], v[54:57]
	v_subrev_f32_dpp v46, v126, v46 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v47, v126, v47 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v48, v126, v48 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v49, v126, v49 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_bf16 v[54:57], a[120:123], a[40:43], v[54:57]
	ds_read_b128 v[86:89], v23 offset:2048
	v_add_u32_e32 v21, s69, v12
	ds_read_b128 v[90:93], v23 offset:3072
	v_mfma_f32_16x16x32_bf16 v[54:57], a[124:127], a[44:47], v[54:57]
	v_mul_f32_e32 v50, 0x3e0293ee, v50
	v_mul_f32_e32 v51, 0x3e0293ee, v51
	v_mul_f32_e32 v52, 0x3e0293ee, v52
	v_mul_f32_e32 v53, 0x3e0293ee, v53
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_bf16 v[58:61], a[112:115], v[62:65], 0
	ds_read_b64_tr_b16 v[94:95], v21
	ds_read_b64_tr_b16 v[96:97], v21 offset:256
	ds_read_b64_tr_b16 v[98:99], v21 offset:1024
	ds_read_b64_tr_b16 v[100:101], v21 offset:1280
	v_mfma_f32_16x16x32_bf16 v[58:61], a[116:119], v[66:69], v[58:61]
	v_subrev_f32_dpp v50, v126, v50 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v51, v126, v51 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v52, v126, v52 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v53, v126, v53 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_16x16x32_bf16 v[58:61], a[120:123], v[70:73], v[58:61]
	ds_read_b64_tr_b16 v[102:103], v21 offset:2048
	ds_read_b64_tr_b16 v[104:105], v21 offset:2304
	ds_read_b64_tr_b16 v[106:107], v21 offset:3072
	v_add_u32_e32 v22, s70, v12
	ds_read_b64_tr_b16 v[108:109], v21 offset:3328
	v_mfma_f32_16x16x32_bf16 v[58:61], a[124:127], v[74:77], v[58:61]
	v_mul_f32_e32 v54, 0x3e0293ee, v54
	v_mul_f32_e32 v55, 0x3e0293ee, v55
	v_mul_f32_e32 v56, 0x3e0293ee, v56
	v_mul_f32_e32 v57, 0x3e0293ee, v57
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_16x16x32_bf16 v[62:65], v[78:81], a[48:51], 0
	v_subrev_f32_dpp v54, v126, v54 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v55, v126, v55 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v56, v126, v56 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v57, v126, v57 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	v_mfma_f32_16x16x32_bf16 v[62:65], v[82:85], a[52:55], v[62:65]
	v_exp_f32_e32 v46, v46
	v_exp_f32_e32 v47, v47
	v_exp_f32_e32 v48, v48
	v_exp_f32_e32 v49, v49
	v_mfma_f32_16x16x32_bf16 v[62:65], v[86:89], a[56:59], v[62:65]
	ds_read_b64_tr_b16 a[112:113], v22
	ds_read_b64_tr_b16 a[114:115], v22 offset:256
	v_mfma_f32_16x16x32_bf16 v[62:65], v[90:93], a[60:63], v[62:65]
	v_exp_f32_e32 v50, v50
	v_exp_f32_e32 v51, v51
	v_exp_f32_e32 v52, v52
	v_exp_f32_e32 v53, v53
	v_mfma_f32_16x16x32_bf16 v[66:69], v[78:81], a[64:67], 0
	ds_read_b64_tr_b16 a[116:117], v22 offset:1024
	ds_read_b64_tr_b16 a[118:119], v22 offset:1280
	v_mfma_f32_16x16x32_bf16 v[66:69], v[82:85], a[68:71], v[66:69]
	v_mul_f32_e32 v58, 0x3e0293ee, v58
	v_mul_f32_e32 v59, 0x3e0293ee, v59
	v_mul_f32_e32 v60, 0x3e0293ee, v60
	v_mul_f32_e32 v61, 0x3e0293ee, v61
	v_mfma_f32_16x16x32_bf16 v[66:69], v[86:89], a[72:75], v[66:69]
	v_subrev_f32_dpp v58, v126, v58 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v59, v126, v59 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v60, v126, v60 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v61, v126, v61 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	v_mfma_f32_16x16x32_bf16 v[66:69], v[90:93], a[76:79], v[66:69]
	v_cvt_pk_bf16_f32 v118, v46, v47
	v_cvt_pk_bf16_f32 v119, v48, v49
	v_mfma_f32_16x16x32_bf16 v[70:73], v[78:81], a[80:83], 0
	v_exp_f32_e32 v54, v54
	v_exp_f32_e32 v55, v55
	v_exp_f32_e32 v56, v56
	v_exp_f32_e32 v57, v57
	v_mfma_f32_16x16x32_bf16 v[70:73], v[82:85], a[84:87], v[70:73]
	v_cvt_pk_bf16_f32 v120, v50, v51
	v_cvt_pk_bf16_f32 v121, v52, v53
	v_mfma_f32_16x16x32_bf16 v[70:73], v[86:89], a[88:91], v[70:73]
	ds_read_b64_tr_b16 a[120:121], v22 offset:2048
	ds_read_b64_tr_b16 a[122:123], v22 offset:2304
	v_mfma_f32_16x16x32_bf16 v[70:73], v[90:93], a[92:95], v[70:73]
	v_exp_f32_e32 v58, v58
	v_exp_f32_e32 v59, v59
	v_exp_f32_e32 v60, v60
	v_exp_f32_e32 v61, v61
	v_mfma_f32_16x16x32_bf16 v[74:77], v[78:81], a[96:99], 0
	ds_read_b64_tr_b16 a[124:125], v22 offset:3072
	ds_read_b64_tr_b16 a[126:127], v22 offset:3328
	v_mfma_f32_16x16x32_bf16 v[74:77], v[82:85], a[100:103], v[74:77]
	v_cvt_pk_bf16_f32 v122, v54, v55
	v_cvt_pk_bf16_f32 v123, v56, v57
	v_cvt_pk_bf16_f32 v124, v58, v59
	v_cvt_pk_bf16_f32 v125, v60, v61
	v_mfma_f32_16x16x32_bf16 v[74:77], v[86:89], a[104:107], v[74:77]
	v_permlane16_swap_b32_e32 v118, v120
	v_permlane16_swap_b32_e32 v119, v121
	v_permlane16_swap_b32_e32 v122, v124
	v_permlane16_swap_b32_e32 v123, v125
	v_mfma_f32_16x16x32_bf16 v[74:77], v[90:93], a[108:111], v[74:77]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x16_bf16 v[128:143], v[94:97], v[118:121], v[128:143]
	ds_read_b64_tr_b16 a[0:1], v1
	ds_read_b64_tr_b16 a[2:3], v1 offset:4096
	ds_read_b64_tr_b16 a[4:5], v1 offset:512
	ds_read_b64_tr_b16 a[6:7], v1 offset:4608
	v_mfma_f32_32x32x16_bf16 v[144:159], v[94:97], v[122:125], v[144:159]
	v_subrev_f32_dpp v62, v127, v62 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v63, v127, v63 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v64, v127, v64 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v65, v127, v65 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v66, v127, v66 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v67, v127, v67 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v68, v127, v68 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v69, v127, v69 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	v_mfma_f32_32x32x16_bf16 v[160:175], v[98:101], v[118:121], v[160:175]
	ds_read_b64_tr_b16 a[8:9], v1 offset:8192
	ds_read_b64_tr_b16 a[10:11], v1 offset:12288
	ds_read_b64_tr_b16 a[12:13], v1 offset:8704
	ds_read_b64_tr_b16 a[14:15], v1 offset:12800
	v_mfma_f32_32x32x16_bf16 v[176:191], v[98:101], v[122:125], v[176:191]
	v_mul_f32_e32 v62, v46, v62
	v_mul_f32_e32 v63, v47, v63
	v_mul_f32_e32 v64, v48, v64
	v_mul_f32_e32 v65, v49, v65
	v_mul_f32_e32 v66, v50, v66
	v_mul_f32_e32 v67, v51, v67
	v_mul_f32_e32 v68, v52, v68
	v_mul_f32_e32 v69, v53, v69
	v_cvt_pk_bf16_f32 v62, v62, v63
	v_cvt_pk_bf16_f32 v63, v64, v65
	v_cvt_pk_bf16_f32 v64, v66, v67
	v_cvt_pk_bf16_f32 v65, v68, v69
	v_subrev_f32_dpp v70, v127, v70 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v71, v127, v71 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v72, v127, v72 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v73, v127, v73 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	v_mfma_f32_32x32x16_bf16 v[192:207], v[102:105], v[118:121], v[192:207]
	ds_read_b64_tr_b16 a[16:17], v1 offset:16384
	ds_read_b64_tr_b16 a[18:19], v1 offset:20480
	ds_write_b64 v9, v[62:63]
	ds_write_b64 v9, v[64:65] offset:512
	v_mfma_f32_32x32x16_bf16 v[208:223], v[102:105], v[122:125], v[208:223]
	v_subrev_f32_dpp v74, v127, v74 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v75, v127, v75 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v76, v127, v76 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v77, v127, v77 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	v_mul_f32_e32 v70, v54, v70
	v_mul_f32_e32 v71, v55, v71
	v_mul_f32_e32 v72, v56, v72
	v_mul_f32_e32 v73, v57, v73
	v_mul_f32_e32 v74, v58, v74
	v_mul_f32_e32 v75, v59, v75
	v_mul_f32_e32 v76, v60, v76
	v_mul_f32_e32 v77, v61, v77
	v_cvt_pk_bf16_f32 v66, v70, v71
	v_cvt_pk_bf16_f32 v67, v72, v73
	v_cvt_pk_bf16_f32 v68, v74, v75
	v_cvt_pk_bf16_f32 v69, v76, v77
	v_mfma_f32_32x32x16_bf16 v[224:239], v[106:109], v[118:121], v[224:239]
	ds_read_b32 v126, v18

	v_add_u32_e32 v19, s78, v13
	ds_read_b32 v127, v19

	ds_write_b64 v9, v[66:67] offset:1024
	ds_write_b64 v9, v[68:69] offset:1536
	v_mfma_f32_32x32x16_bf16 v[240:255], v[106:109], v[122:125], v[240:255]
	v_permlane16_swap_b32_e32 v62, v64
	v_permlane16_swap_b32_e32 v63, v65
	v_permlane16_swap_b32_e32 v66, v68
	v_permlane16_swap_b32_e32 v67, v69
	s_waitcnt lgkmcnt(12)
	v_mfma_f32_32x32x16_bf16 a[128:143], a[112:115], v[62:65], a[128:143]
	ds_read_b64_tr_b16 a[20:21], v1 offset:16896
	ds_read_b64_tr_b16 a[22:23], v1 offset:20992
	ds_read_b64_tr_b16 a[24:25], v1 offset:24576
	ds_read_b64_tr_b16 a[26:27], v1 offset:28672
	ds_read_b64_tr_b16 a[28:29], v1 offset:25088
	ds_read_b64_tr_b16 a[30:31], v1 offset:29184
	ds_read_b64_tr_b16 a[32:33], v1 offset:32768
	ds_read_b64_tr_b16 a[34:35], v1 offset:36864
	v_mfma_f32_32x32x16_bf16 a[144:159], a[112:115], v[66:69], a[144:159]
	s_waitcnt lgkmcnt(8)
	s_barrier
	v_mfma_f32_32x32x16_bf16 a[160:175], a[116:119], v[62:65], a[160:175]
	ds_read_b64_tr_b16 v[30:31], v4
	ds_read_b64_tr_b16 v[32:33], v4 offset:512
	s_add_i32 s0, s68, 32
	ds_read_b64_tr_b16 v[34:35], v4 offset:1024
	s_mul_i32 s0, s0, s3
	ds_read_b64_tr_b16 v[36:37], v4 offset:1536
	s_ashr_i32 s1, s0, 31
	ds_read_b64_tr_b16 v[38:39], v4 offset:2048
	s_lshl_b64 s[0:1], s[0:1], 1
	ds_read_b64_tr_b16 v[40:41], v4 offset:2560
	s_add_u32 s8, s6, s0
	ds_read_b64_tr_b16 v[42:43], v4 offset:3072
	s_addc_u32 s9, s7, s1
	ds_read_b64_tr_b16 v[44:45], v4 offset:3584
	v_mfma_f32_32x32x16_bf16 a[176:191], a[116:119], v[66:69], a[176:191]
	v_mul_f32_e32 v126, 0x3fb8aa3b, v126
	v_cvt_pk_bf16_f32 v110, v110, v111
	v_cvt_pk_bf16_f32 v111, v112, v113
	buffer_atomic_pk_add_bf16 v110, v5, s[8:11], 0 offen
	buffer_atomic_pk_add_bf16 v111, v6, s[8:11], 0 offen
	v_mfma_f32_32x32x16_bf16 a[192:207], a[120:123], v[62:65], a[192:207]
	ds_read_b64_tr_b16 v[46:47], v4 offset:4096
	ds_read_b64_tr_b16 v[48:49], v4 offset:4608
	ds_read_b64_tr_b16 v[50:51], v4 offset:5120
	ds_read_b64_tr_b16 v[52:53], v4 offset:5632
	ds_read_b64_tr_b16 a[36:37], v1 offset:33280
	ds_read_b64_tr_b16 a[38:39], v1 offset:37376
	ds_read_b64_tr_b16 a[40:41], v1 offset:40960
	ds_read_b64_tr_b16 a[42:43], v1 offset:45056
	v_mfma_f32_32x32x16_bf16 a[208:223], a[120:123], v[66:69], a[208:223]
	v_cvt_pk_bf16_f32 v114, v114, v115
	v_cvt_pk_bf16_f32 v115, v116, v117
	buffer_atomic_pk_add_bf16 v114, v7, s[8:11], 0 offen
	buffer_atomic_pk_add_bf16 v115, v8, s[8:11], 0 offen
	v_mfma_f32_32x32x16_bf16 a[224:239], a[124:127], v[62:65], a[224:239]
	ds_read_b64_tr_b16 v[54:55], v4 offset:6144
	ds_read_b64_tr_b16 v[56:57], v4 offset:6656
	ds_read_b64_tr_b16 v[58:59], v4 offset:7168
	ds_read_b64_tr_b16 v[60:61], v4 offset:7680
	ds_read_b64_tr_b16 a[44:45], v1 offset:41472
	ds_read_b64_tr_b16 a[46:47], v1 offset:45568
	v_mfma_f32_32x32x16_bf16 a[240:255], a[124:127], v[66:69], a[240:255]
	s_waitcnt vmcnt(4) lgkmcnt(6)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[110:113], a[0:3], v[30:33], 0
	ds_read_b64_tr_b16 v[62:63], v1 offset:49152
	ds_read_b64_tr_b16 v[64:65], v1 offset:53248
	ds_read_b64_tr_b16 v[66:67], v1 offset:49664
	ds_read_b64_tr_b16 v[68:69], v1 offset:53760
	v_mfma_f32_16x16x32_bf16 v[110:113], a[8:11], v[34:37], v[110:113]
	v_mfma_f32_16x16x32_bf16 v[110:113], a[16:19], v[38:41], v[110:113]
	ds_read_b64_tr_b16 v[70:71], v1 offset:57344
	ds_read_b64_tr_b16 v[72:73], v1 offset:61440
	ds_read_b64_tr_b16 v[74:75], v1 offset:57856
	ds_read_b64_tr_b16 v[76:77], v1 offset:61952
	v_mfma_f32_16x16x32_bf16 v[110:113], a[24:27], v[42:45], v[110:113]
	v_mfma_f32_16x16x32_bf16 v[110:113], a[32:35], v[46:49], v[110:113]
	ds_read_b128 a[112:115], v17
	ds_read_b128 a[116:119], v17 offset:1024
	v_mfma_f32_16x16x32_bf16 v[110:113], a[40:43], v[50:53], v[110:113]
	s_waitcnt lgkmcnt(4)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[110:113], v[62:65], v[54:57], v[110:113]
	ds_read_b128 a[120:123], v17 offset:2048
	ds_read_b128 a[124:127], v17 offset:3072
	v_mfma_f32_16x16x32_bf16 v[110:113], v[70:73], v[58:61], v[110:113]
	v_mfma_f32_16x16x32_bf16 v[114:117], a[4:7], v[30:33], 0
	ds_read_b128 a[0:3], v10
	ds_read_b128 a[4:7], v10 offset:1024
	v_mfma_f32_16x16x32_bf16 v[114:117], a[12:15], v[34:37], v[114:117]
	v_mfma_f32_16x16x32_bf16 v[114:117], a[20:23], v[38:41], v[114:117]
	ds_read_b128 a[8:11], v10 offset:2048
	ds_read_b128 a[12:15], v10 offset:3072
	v_mfma_f32_16x16x32_bf16 v[114:117], a[28:31], v[42:45], v[114:117]
	v_mul_f32_e32 v110, 0x3db504f3, v110
	v_mul_f32_e32 v111, 0x3db504f3, v111
	v_mfma_f32_16x16x32_bf16 v[114:117], a[36:39], v[46:49], v[114:117]
	ds_read_b128 a[16:19], v10 offset:4096
	ds_read_b128 a[20:23], v10 offset:5120
	v_mfma_f32_16x16x32_bf16 v[114:117], a[44:47], v[50:53], v[114:117]
	v_mul_f32_e32 v112, 0x3db504f3, v112
	v_mul_f32_e32 v113, 0x3db504f3, v113
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_16x16x32_bf16 v[114:117], v[66:69], v[54:57], v[114:117]
	ds_read_b128 a[24:27], v10 offset:6144
	ds_read_b128 a[28:31], v10 offset:7168
	s_xor_b32 s35, s35, 1
	s_xor_b32 s59, s59, 1
	s_add_i32 s37, s37, 1
	s_add_i32 s17, s17, 64
	s_cmpk_eq_i32 s37, 0x400
	v_mfma_f32_16x16x32_bf16 v[114:117], v[74:77], v[58:61], v[114:117]
	s_waitcnt lgkmcnt(2)
	s_cbranch_scc0 .LBB0_1
	v_mfma_f32_16x16x32_bf16 v[46:49], a[112:115], a[0:3], 0
	ds_read_b128 a[32:35], v10 offset:8192
	ds_read_b128 a[36:39], v10 offset:9216
	v_mfma_f32_16x16x32_bf16 v[46:49], a[116:119], a[4:7], v[46:49]
	v_mfma_f32_16x16x32_bf16 v[46:49], a[120:123], a[8:11], v[46:49]
	ds_read_b128 a[40:43], v10 offset:10240
	ds_read_b128 a[44:47], v10 offset:11264
	v_mfma_f32_16x16x32_bf16 v[46:49], a[124:127], a[12:15], v[46:49]
	v_mul_f32_e32 v114, 0x3db504f3, v114
	v_mul_f32_e32 v115, 0x3db504f3, v115
	v_mfma_f32_16x16x32_bf16 v[50:53], a[112:115], a[16:19], 0
	ds_read_b128 v[62:65], v10 offset:12288
	s_lshl_b32 s0, s35, 14
	ds_read_b128 v[66:69], v10 offset:13312
	v_mfma_f32_16x16x32_bf16 v[50:53], a[116:119], a[20:23], v[50:53]
	v_mul_f32_e32 v116, 0x3db504f3, v116
	v_mul_f32_e32 v117, 0x3db504f3, v117
	v_mfma_f32_16x16x32_bf16 v[50:53], a[120:123], a[24:27], v[50:53]
	ds_read_b128 v[70:73], v10 offset:14336
	s_add_i32 s17, s62, s0
	ds_read_b128 v[74:77], v10 offset:15360
	v_mfma_f32_16x16x32_bf16 v[50:53], a[124:127], a[28:31], v[50:53]
	v_mul_f32_e32 v46, 0x3e0293ee, v46
	v_mul_f32_e32 v47, 0x3e0293ee, v47
	v_mul_f32_e32 v48, 0x3e0293ee, v48
	v_mul_f32_e32 v49, 0x3e0293ee, v49
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_bf16 v[54:57], a[112:115], a[32:35], 0
	v_add_u32_e32 v2, s17, v11
	ds_read_b128 v[78:81], v2
	ds_read_b128 v[82:85], v2 offset:1024
	v_mfma_f32_16x16x32_bf16 v[54:57], a[116:119], a[36:39], v[54:57]
	v_subrev_f32_dpp v46, v126, v46 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v47, v126, v47 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v48, v126, v48 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v49, v126, v49 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_bf16 v[54:57], a[120:123], a[40:43], v[54:57]
	ds_read_b128 v[86:89], v2 offset:2048
	ds_read_b128 v[90:93], v2 offset:3072
	v_mfma_f32_16x16x32_bf16 v[54:57], a[124:127], a[44:47], v[54:57]
	v_mul_f32_e32 v50, 0x3e0293ee, v50
	v_mul_f32_e32 v51, 0x3e0293ee, v51
	v_mul_f32_e32 v52, 0x3e0293ee, v52
	v_mul_f32_e32 v53, 0x3e0293ee, v53
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_bf16 v[58:61], a[112:115], v[62:65], 0
	v_add_u32_e32 v2, s17, v12
	ds_read_b64_tr_b16 v[94:95], v2
	ds_read_b64_tr_b16 v[96:97], v2 offset:256
	ds_read_b64_tr_b16 v[98:99], v2 offset:1024
	ds_read_b64_tr_b16 v[100:101], v2 offset:1280
	v_mfma_f32_16x16x32_bf16 v[58:61], a[116:119], v[66:69], v[58:61]
	v_subrev_f32_dpp v50, v126, v50 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v51, v126, v51 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v52, v126, v52 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v53, v126, v53 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_16x16x32_bf16 v[58:61], a[120:123], v[70:73], v[58:61]
	ds_read_b64_tr_b16 v[102:103], v2 offset:2048
	ds_read_b64_tr_b16 v[104:105], v2 offset:2304
	ds_read_b64_tr_b16 v[106:107], v2 offset:3072
	s_add_i32 s19, s27, s0
	ds_read_b64_tr_b16 v[108:109], v2 offset:3328
	v_mfma_f32_16x16x32_bf16 v[58:61], a[124:127], v[74:77], v[58:61]
	v_mul_f32_e32 v54, 0x3e0293ee, v54
	v_mul_f32_e32 v55, 0x3e0293ee, v55
	v_mul_f32_e32 v56, 0x3e0293ee, v56
	v_mul_f32_e32 v57, 0x3e0293ee, v57
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_16x16x32_bf16 v[62:65], v[78:81], a[48:51], 0
	v_subrev_f32_dpp v54, v126, v54 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v55, v126, v55 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v56, v126, v56 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v57, v126, v57 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	v_mfma_f32_16x16x32_bf16 v[62:65], v[82:85], a[52:55], v[62:65]
	v_exp_f32_e32 v46, v46
	v_exp_f32_e32 v47, v47
	v_exp_f32_e32 v48, v48
	v_exp_f32_e32 v49, v49
	v_mfma_f32_16x16x32_bf16 v[62:65], v[86:89], a[56:59], v[62:65]
	v_add_u32_e32 v2, s19, v12
	ds_read_b64_tr_b16 a[112:113], v2
	ds_read_b64_tr_b16 a[114:115], v2 offset:256
	v_mfma_f32_16x16x32_bf16 v[62:65], v[90:93], a[60:63], v[62:65]
	v_exp_f32_e32 v50, v50
	v_exp_f32_e32 v51, v51
	v_exp_f32_e32 v52, v52
	v_exp_f32_e32 v53, v53
	v_mfma_f32_16x16x32_bf16 v[66:69], v[78:81], a[64:67], 0
	ds_read_b64_tr_b16 a[116:117], v2 offset:1024
	ds_read_b64_tr_b16 a[118:119], v2 offset:1280
	v_mfma_f32_16x16x32_bf16 v[66:69], v[82:85], a[68:71], v[66:69]
	v_mul_f32_e32 v58, 0x3e0293ee, v58
	v_mul_f32_e32 v59, 0x3e0293ee, v59
	v_mul_f32_e32 v60, 0x3e0293ee, v60
	v_mul_f32_e32 v61, 0x3e0293ee, v61
	v_mfma_f32_16x16x32_bf16 v[66:69], v[86:89], a[72:75], v[66:69]
	v_subrev_f32_dpp v58, v126, v58 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v59, v126, v59 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v60, v126, v60 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v61, v126, v61 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	v_mfma_f32_16x16x32_bf16 v[66:69], v[90:93], a[76:79], v[66:69]
	v_cvt_pk_bf16_f32 v118, v46, v47
	v_cvt_pk_bf16_f32 v119, v48, v49
	v_mfma_f32_16x16x32_bf16 v[70:73], v[78:81], a[80:83], 0
	v_exp_f32_e32 v54, v54
	v_exp_f32_e32 v55, v55
	v_exp_f32_e32 v56, v56
	v_exp_f32_e32 v57, v57
	v_mfma_f32_16x16x32_bf16 v[70:73], v[82:85], a[84:87], v[70:73]
	v_cvt_pk_bf16_f32 v120, v50, v51
	v_cvt_pk_bf16_f32 v121, v52, v53
	v_mfma_f32_16x16x32_bf16 v[70:73], v[86:89], a[88:91], v[70:73]
	ds_read_b64_tr_b16 a[120:121], v2 offset:2048
	ds_read_b64_tr_b16 a[122:123], v2 offset:2304
	v_mfma_f32_16x16x32_bf16 v[70:73], v[90:93], a[92:95], v[70:73]
	v_exp_f32_e32 v58, v58
	v_exp_f32_e32 v59, v59
	v_exp_f32_e32 v60, v60
	v_exp_f32_e32 v61, v61
	v_mfma_f32_16x16x32_bf16 v[74:77], v[78:81], a[96:99], 0
	ds_read_b64_tr_b16 a[124:125], v2 offset:3072
	ds_read_b64_tr_b16 a[126:127], v2 offset:3328
	v_mfma_f32_16x16x32_bf16 v[74:77], v[82:85], a[100:103], v[74:77]
	v_cvt_pk_bf16_f32 v122, v54, v55
	v_cvt_pk_bf16_f32 v123, v56, v57
	v_cvt_pk_bf16_f32 v124, v58, v59
	v_cvt_pk_bf16_f32 v125, v60, v61
	v_mfma_f32_16x16x32_bf16 v[74:77], v[86:89], a[104:107], v[74:77]
	v_permlane16_swap_b32_e32 v118, v120
	v_permlane16_swap_b32_e32 v119, v121
	v_permlane16_swap_b32_e32 v122, v124
	v_permlane16_swap_b32_e32 v123, v125
	v_mfma_f32_16x16x32_bf16 v[74:77], v[90:93], a[108:111], v[74:77]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x16_bf16 v[128:143], v[94:97], v[118:121], v[128:143]
	ds_read_b64_tr_b16 a[0:1], v1
	ds_read_b64_tr_b16 a[2:3], v1 offset:4096
	ds_read_b64_tr_b16 a[4:5], v1 offset:512
	ds_read_b64_tr_b16 a[6:7], v1 offset:4608
	v_mfma_f32_32x32x16_bf16 v[144:159], v[94:97], v[122:125], v[144:159]
	v_subrev_f32_dpp v62, v127, v62 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v63, v127, v63 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v64, v127, v64 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v65, v127, v65 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v66, v127, v66 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v67, v127, v67 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v68, v127, v68 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v69, v127, v69 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	v_mfma_f32_32x32x16_bf16 v[160:175], v[98:101], v[118:121], v[160:175]
	ds_read_b64_tr_b16 a[8:9], v1 offset:8192
	ds_read_b64_tr_b16 a[10:11], v1 offset:12288
	ds_read_b64_tr_b16 a[12:13], v1 offset:8704
	ds_read_b64_tr_b16 a[14:15], v1 offset:12800
	v_mfma_f32_32x32x16_bf16 v[176:191], v[98:101], v[122:125], v[176:191]
	v_mul_f32_e32 v62, v46, v62
	v_mul_f32_e32 v63, v47, v63
	v_mul_f32_e32 v64, v48, v64
	v_mul_f32_e32 v65, v49, v65
	v_mul_f32_e32 v66, v50, v66
	v_mul_f32_e32 v67, v51, v67
	v_mul_f32_e32 v68, v52, v68
	v_mul_f32_e32 v69, v53, v69
	v_cvt_pk_bf16_f32 v62, v62, v63
	v_cvt_pk_bf16_f32 v63, v64, v65
	v_cvt_pk_bf16_f32 v64, v66, v67
	v_cvt_pk_bf16_f32 v65, v68, v69
	v_subrev_f32_dpp v70, v127, v70 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v71, v127, v71 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v72, v127, v72 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v73, v127, v73 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	v_mfma_f32_32x32x16_bf16 v[192:207], v[102:105], v[118:121], v[192:207]
	ds_read_b64_tr_b16 a[16:17], v1 offset:16384
	ds_read_b64_tr_b16 a[18:19], v1 offset:20480
	s_lshl_b32 s0, s35, 8
	ds_write_b64 v9, v[62:63]
	s_add_i32 s22, s5, s0
	ds_write_b64 v9, v[64:65] offset:512
	v_mfma_f32_32x32x16_bf16 v[208:223], v[102:105], v[122:125], v[208:223]
	v_subrev_f32_dpp v74, v127, v74 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v75, v127, v75 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v76, v127, v76 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v77, v127, v77 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	v_mul_f32_e32 v70, v54, v70
	v_mul_f32_e32 v71, v55, v71
	v_mul_f32_e32 v72, v56, v72
	v_mul_f32_e32 v73, v57, v73
	v_mul_f32_e32 v74, v58, v74
	v_mul_f32_e32 v75, v59, v75
	v_mul_f32_e32 v76, v60, v76
	v_mul_f32_e32 v77, v61, v77
	v_cvt_pk_bf16_f32 v66, v70, v71
	v_cvt_pk_bf16_f32 v67, v72, v73
	v_cvt_pk_bf16_f32 v68, v74, v75
	v_cvt_pk_bf16_f32 v69, v76, v77
	v_mfma_f32_32x32x16_bf16 v[224:239], v[106:109], v[118:121], v[224:239]
	v_add3_u32 v2, s22, 64, v13
	ds_read_b32 v126, v2

	s_add_i32 s23, s23, s0
	v_add3_u32 v2, s23, 64, v13
	ds_read_b32 v127, v2

	ds_write_b64 v9, v[66:67] offset:1024
	ds_write_b64 v9, v[68:69] offset:1536
	v_mfma_f32_32x32x16_bf16 v[240:255], v[106:109], v[122:125], v[240:255]
	v_permlane16_swap_b32_e32 v62, v64
	v_permlane16_swap_b32_e32 v63, v65
	v_permlane16_swap_b32_e32 v66, v68
	v_permlane16_swap_b32_e32 v67, v69
	s_waitcnt lgkmcnt(12)
	v_mfma_f32_32x32x16_bf16 a[128:143], a[112:115], v[62:65], a[128:143]
	ds_read_b64_tr_b16 a[20:21], v1 offset:16896
	ds_read_b64_tr_b16 a[22:23], v1 offset:20992
	ds_read_b64_tr_b16 a[24:25], v1 offset:24576
	ds_read_b64_tr_b16 a[26:27], v1 offset:28672
	ds_read_b64_tr_b16 a[28:29], v1 offset:25088
	ds_read_b64_tr_b16 a[30:31], v1 offset:29184
	ds_read_b64_tr_b16 a[32:33], v1 offset:32768
	ds_read_b64_tr_b16 a[34:35], v1 offset:36864
	v_mfma_f32_32x32x16_bf16 a[144:159], a[112:115], v[66:69], a[144:159]
	s_waitcnt lgkmcnt(8)
	s_barrier
	v_mfma_f32_32x32x16_bf16 a[160:175], a[116:119], v[62:65], a[160:175]
	ds_read_b64_tr_b16 v[30:31], v4
	s_add_i32 s5, s15, 7
	ds_read_b64_tr_b16 v[32:33], v4 offset:512
	s_mul_i32 s5, s5, s38
	ds_read_b64_tr_b16 v[34:35], v4 offset:1024
	s_add_i32 s0, s5, 0x1fb0
	ds_read_b64_tr_b16 v[36:37], v4 offset:1536
	s_mul_i32 s0, s0, s3
	ds_read_b64_tr_b16 v[38:39], v4 offset:2048
	s_ashr_i32 s1, s0, 31
	ds_read_b64_tr_b16 v[40:41], v4 offset:2560
	s_lshl_b64 s[8:9], s[0:1], 1
	ds_read_b64_tr_b16 v[42:43], v4 offset:3072
	s_add_u32 s8, s6, s8
	ds_read_b64_tr_b16 v[44:45], v4 offset:3584
	v_mfma_f32_32x32x16_bf16 a[176:191], a[116:119], v[66:69], a[176:191]
	v_mul_f32_e32 v126, 0x3fb8aa3b, v126
	s_addc_u32 s9, s7, s9
	s_mov_b32 s11, 0x20000
	v_cvt_pk_bf16_f32 v110, v110, v111
	v_cvt_pk_bf16_f32 v111, v112, v113
	buffer_atomic_pk_add_bf16 v110, v5, s[8:11], 0 offen
	buffer_atomic_pk_add_bf16 v111, v6, s[8:11], 0 offen
	v_mfma_f32_32x32x16_bf16 a[192:207], a[120:123], v[62:65], a[192:207]
	ds_read_b64_tr_b16 v[46:47], v4 offset:4096
	ds_read_b64_tr_b16 v[48:49], v4 offset:4608
	ds_read_b64_tr_b16 v[50:51], v4 offset:5120
	ds_read_b64_tr_b16 v[52:53], v4 offset:5632
	ds_read_b64_tr_b16 a[36:37], v1 offset:33280
	ds_read_b64_tr_b16 a[38:39], v1 offset:37376
	ds_read_b64_tr_b16 a[40:41], v1 offset:40960
	ds_read_b64_tr_b16 a[42:43], v1 offset:45056
	v_mfma_f32_32x32x16_bf16 a[208:223], a[120:123], v[66:69], a[208:223]
	v_cvt_pk_bf16_f32 v114, v114, v115
	v_cvt_pk_bf16_f32 v115, v116, v117
	buffer_atomic_pk_add_bf16 v114, v7, s[8:11], 0 offen
	buffer_atomic_pk_add_bf16 v115, v8, s[8:11], 0 offen
	v_mfma_f32_32x32x16_bf16 a[224:239], a[124:127], v[62:65], a[224:239]
	ds_read_b64_tr_b16 v[54:55], v4 offset:6144
	ds_read_b64_tr_b16 v[56:57], v4 offset:6656
	ds_read_b64_tr_b16 v[58:59], v4 offset:7168
	ds_read_b64_tr_b16 v[60:61], v4 offset:7680
	ds_read_b64_tr_b16 a[44:45], v1 offset:41472
	ds_read_b64_tr_b16 a[46:47], v1 offset:45568
	v_mfma_f32_32x32x16_bf16 a[240:255], a[124:127], v[66:69], a[240:255]
	s_waitcnt lgkmcnt(6)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[110:113], a[0:3], v[30:33], 0
	ds_read_b64_tr_b16 v[62:63], v1 offset:49152
	ds_read_b64_tr_b16 v[64:65], v1 offset:53248
	ds_read_b64_tr_b16 v[66:67], v1 offset:49664
	ds_read_b64_tr_b16 v[68:69], v1 offset:53760
	v_mfma_f32_16x16x32_bf16 v[110:113], a[8:11], v[34:37], v[110:113]
	v_mfma_f32_16x16x32_bf16 v[110:113], a[16:19], v[38:41], v[110:113]
	ds_read_b64_tr_b16 v[70:71], v1 offset:57344
	ds_read_b64_tr_b16 v[72:73], v1 offset:61440
	ds_read_b64_tr_b16 v[74:75], v1 offset:57856
	s_add_i32 s1, s19, 0x1000
	ds_read_b64_tr_b16 v[76:77], v1 offset:61952
	v_mfma_f32_16x16x32_bf16 v[110:113], a[24:27], v[42:45], v[110:113]
	v_mfma_f32_16x16x32_bf16 v[110:113], a[32:35], v[46:49], v[110:113]
	v_add_u32_e32 v2, s1, v11
	ds_read_b128 a[112:115], v2
	ds_read_b128 a[116:119], v2 offset:1024
	v_mfma_f32_16x16x32_bf16 v[110:113], a[40:43], v[50:53], v[110:113]
	s_waitcnt lgkmcnt(4)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[110:113], v[62:65], v[54:57], v[110:113]
	ds_read_b128 a[120:123], v2 offset:2048
	ds_read_b128 a[124:127], v2 offset:3072
	v_mfma_f32_16x16x32_bf16 v[110:113], v[70:73], v[58:61], v[110:113]
	v_mfma_f32_16x16x32_bf16 v[114:117], a[4:7], v[30:33], 0
	ds_read_b128 a[0:3], v10
	ds_read_b128 a[4:7], v10 offset:1024
	v_mfma_f32_16x16x32_bf16 v[114:117], a[12:15], v[34:37], v[114:117]
	v_mfma_f32_16x16x32_bf16 v[114:117], a[20:23], v[38:41], v[114:117]
	ds_read_b128 a[8:11], v10 offset:2048
	ds_read_b128 a[12:15], v10 offset:3072
	v_mfma_f32_16x16x32_bf16 v[114:117], a[28:31], v[42:45], v[114:117]
	v_mul_f32_e32 v110, 0x3db504f3, v110
	v_mul_f32_e32 v111, 0x3db504f3, v111
	v_mfma_f32_16x16x32_bf16 v[114:117], a[36:39], v[46:49], v[114:117]
	ds_read_b128 a[16:19], v10 offset:4096
	ds_read_b128 a[20:23], v10 offset:5120
	v_mfma_f32_16x16x32_bf16 v[114:117], a[44:47], v[50:53], v[114:117]
	v_mul_f32_e32 v112, 0x3db504f3, v112
	v_mul_f32_e32 v113, 0x3db504f3, v113
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_16x16x32_bf16 v[114:117], v[66:69], v[54:57], v[114:117]
	ds_read_b128 a[24:27], v10 offset:6144
	ds_read_b128 a[28:31], v10 offset:7168
	v_mfma_f32_16x16x32_bf16 v[114:117], v[74:77], v[58:61], v[114:117]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x32_bf16 v[46:49], a[112:115], a[0:3], 0
	ds_read_b128 a[32:35], v10 offset:8192
	ds_read_b128 a[36:39], v10 offset:9216
	v_mfma_f32_16x16x32_bf16 v[46:49], a[116:119], a[4:7], v[46:49]
	v_mfma_f32_16x16x32_bf16 v[46:49], a[120:123], a[8:11], v[46:49]
	ds_read_b128 a[40:43], v10 offset:10240
	ds_read_b128 a[44:47], v10 offset:11264
	v_mfma_f32_16x16x32_bf16 v[46:49], a[124:127], a[12:15], v[46:49]
	v_mul_f32_e32 v114, 0x3db504f3, v114
	v_mul_f32_e32 v115, 0x3db504f3, v115
	v_mfma_f32_16x16x32_bf16 v[50:53], a[112:115], a[16:19], 0
	ds_read_b128 v[62:65], v10 offset:12288
	ds_read_b128 v[66:69], v10 offset:13312
	v_mfma_f32_16x16x32_bf16 v[50:53], a[116:119], a[20:23], v[50:53]
	v_mul_f32_e32 v116, 0x3db504f3, v116
	v_mul_f32_e32 v117, 0x3db504f3, v117
	v_mfma_f32_16x16x32_bf16 v[50:53], a[120:123], a[24:27], v[50:53]
	ds_read_b128 v[70:73], v10 offset:14336
	s_add_i32 s8, s17, 0x1000
	ds_read_b128 v[74:77], v10 offset:15360
	v_mfma_f32_16x16x32_bf16 v[50:53], a[124:127], a[28:31], v[50:53]
	v_mul_f32_e32 v46, 0x3e0293ee, v46
	v_mul_f32_e32 v47, 0x3e0293ee, v47
	v_mul_f32_e32 v48, 0x3e0293ee, v48
	v_mul_f32_e32 v49, 0x3e0293ee, v49
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_bf16 v[54:57], a[112:115], a[32:35], 0
	v_add_u32_e32 v2, s8, v11
	ds_read_b128 v[78:81], v2
	ds_read_b128 v[82:85], v2 offset:1024
	v_mfma_f32_16x16x32_bf16 v[54:57], a[116:119], a[36:39], v[54:57]
	v_subrev_f32_dpp v46, v126, v46 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v47, v126, v47 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v48, v126, v48 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v49, v126, v49 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_bf16 v[54:57], a[120:123], a[40:43], v[54:57]
	ds_read_b128 v[86:89], v2 offset:2048
	ds_read_b128 v[90:93], v2 offset:3072
	v_mfma_f32_16x16x32_bf16 v[54:57], a[124:127], a[44:47], v[54:57]
	v_mul_f32_e32 v50, 0x3e0293ee, v50
	v_mul_f32_e32 v51, 0x3e0293ee, v51
	v_mul_f32_e32 v52, 0x3e0293ee, v52
	v_mul_f32_e32 v53, 0x3e0293ee, v53
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_bf16 v[58:61], a[112:115], v[62:65], 0
	v_add_u32_e32 v2, s8, v12
	ds_read_b64_tr_b16 v[94:95], v2
	ds_read_b64_tr_b16 v[96:97], v2 offset:256
	ds_read_b64_tr_b16 v[98:99], v2 offset:1024
	ds_read_b64_tr_b16 v[100:101], v2 offset:1280
	v_mfma_f32_16x16x32_bf16 v[58:61], a[116:119], v[66:69], v[58:61]
	v_subrev_f32_dpp v50, v126, v50 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v51, v126, v51 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v52, v126, v52 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v53, v126, v53 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_16x16x32_bf16 v[58:61], a[120:123], v[70:73], v[58:61]
	ds_read_b64_tr_b16 v[102:103], v2 offset:2048
	ds_read_b64_tr_b16 v[104:105], v2 offset:2304
	ds_read_b64_tr_b16 v[106:107], v2 offset:3072
	ds_read_b64_tr_b16 v[108:109], v2 offset:3328
	v_mfma_f32_16x16x32_bf16 v[58:61], a[124:127], v[74:77], v[58:61]
	v_mul_f32_e32 v54, 0x3e0293ee, v54
	v_mul_f32_e32 v55, 0x3e0293ee, v55
	v_mul_f32_e32 v56, 0x3e0293ee, v56
	v_mul_f32_e32 v57, 0x3e0293ee, v57
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_16x16x32_bf16 v[62:65], v[78:81], a[48:51], 0
	v_subrev_f32_dpp v54, v126, v54 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v55, v126, v55 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v56, v126, v56 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v57, v126, v57 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	v_mfma_f32_16x16x32_bf16 v[62:65], v[82:85], a[52:55], v[62:65]
	v_exp_f32_e32 v46, v46
	v_exp_f32_e32 v47, v47
	v_exp_f32_e32 v48, v48
	v_exp_f32_e32 v49, v49
	v_mfma_f32_16x16x32_bf16 v[62:65], v[86:89], a[56:59], v[62:65]
	v_add_u32_e32 v2, s1, v12
	ds_read_b64_tr_b16 a[112:113], v2
	ds_read_b64_tr_b16 a[114:115], v2 offset:256
	v_mfma_f32_16x16x32_bf16 v[62:65], v[90:93], a[60:63], v[62:65]
	v_exp_f32_e32 v50, v50
	v_exp_f32_e32 v51, v51
	v_exp_f32_e32 v52, v52
	v_exp_f32_e32 v53, v53
	v_mfma_f32_16x16x32_bf16 v[66:69], v[78:81], a[64:67], 0
	ds_read_b64_tr_b16 a[116:117], v2 offset:1024
	ds_read_b64_tr_b16 a[118:119], v2 offset:1280
	v_mfma_f32_16x16x32_bf16 v[66:69], v[82:85], a[68:71], v[66:69]
	v_mul_f32_e32 v58, 0x3e0293ee, v58
	v_mul_f32_e32 v59, 0x3e0293ee, v59
	v_mul_f32_e32 v60, 0x3e0293ee, v60
	v_mul_f32_e32 v61, 0x3e0293ee, v61
	v_mfma_f32_16x16x32_bf16 v[66:69], v[86:89], a[72:75], v[66:69]
	v_subrev_f32_dpp v58, v126, v58 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v59, v126, v59 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v60, v126, v60 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v61, v126, v61 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	v_mfma_f32_16x16x32_bf16 v[66:69], v[90:93], a[76:79], v[66:69]
	v_cvt_pk_bf16_f32 v118, v46, v47
	v_cvt_pk_bf16_f32 v119, v48, v49
	v_mfma_f32_16x16x32_bf16 v[70:73], v[78:81], a[80:83], 0
	v_exp_f32_e32 v54, v54
	v_exp_f32_e32 v55, v55
	v_exp_f32_e32 v56, v56
	v_exp_f32_e32 v57, v57
	v_mfma_f32_16x16x32_bf16 v[70:73], v[82:85], a[84:87], v[70:73]
	v_cvt_pk_bf16_f32 v120, v50, v51
	v_cvt_pk_bf16_f32 v121, v52, v53
	v_mfma_f32_16x16x32_bf16 v[70:73], v[86:89], a[88:91], v[70:73]
	ds_read_b64_tr_b16 a[120:121], v2 offset:2048
	ds_read_b64_tr_b16 a[122:123], v2 offset:2304
	v_mfma_f32_16x16x32_bf16 v[70:73], v[90:93], a[92:95], v[70:73]
	v_exp_f32_e32 v58, v58
	v_exp_f32_e32 v59, v59
	v_exp_f32_e32 v60, v60
	v_exp_f32_e32 v61, v61
	v_mfma_f32_16x16x32_bf16 v[74:77], v[78:81], a[96:99], 0
	ds_read_b64_tr_b16 a[124:125], v2 offset:3072
	ds_read_b64_tr_b16 a[126:127], v2 offset:3328
	v_mfma_f32_16x16x32_bf16 v[74:77], v[82:85], a[100:103], v[74:77]
	v_cvt_pk_bf16_f32 v122, v54, v55
	v_cvt_pk_bf16_f32 v123, v56, v57
	v_cvt_pk_bf16_f32 v124, v58, v59
	v_cvt_pk_bf16_f32 v125, v60, v61
	v_mfma_f32_16x16x32_bf16 v[74:77], v[86:89], a[104:107], v[74:77]
	v_permlane16_swap_b32_e32 v118, v120
	v_permlane16_swap_b32_e32 v119, v121
	v_permlane16_swap_b32_e32 v122, v124
	v_permlane16_swap_b32_e32 v123, v125
	v_mfma_f32_16x16x32_bf16 v[74:77], v[90:93], a[108:111], v[74:77]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x16_bf16 v[128:143], v[94:97], v[118:121], v[128:143]
	ds_read_b64_tr_b16 a[0:1], v1
	ds_read_b64_tr_b16 a[2:3], v1 offset:4096
	ds_read_b64_tr_b16 a[4:5], v1 offset:512
	ds_read_b64_tr_b16 a[6:7], v1 offset:4608
	v_mfma_f32_32x32x16_bf16 v[144:159], v[94:97], v[122:125], v[144:159]
	v_subrev_f32_dpp v62, v127, v62 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v63, v127, v63 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v64, v127, v64 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v65, v127, v65 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v66, v127, v66 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v67, v127, v67 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v68, v127, v68 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v69, v127, v69 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	v_mfma_f32_32x32x16_bf16 v[160:175], v[98:101], v[118:121], v[160:175]
	ds_read_b64_tr_b16 a[8:9], v1 offset:8192
	ds_read_b64_tr_b16 a[10:11], v1 offset:12288
	ds_read_b64_tr_b16 a[12:13], v1 offset:8704
	ds_read_b64_tr_b16 a[14:15], v1 offset:12800
	v_mfma_f32_32x32x16_bf16 v[176:191], v[98:101], v[122:125], v[176:191]
	v_mul_f32_e32 v62, v46, v62
	v_mul_f32_e32 v63, v47, v63
	v_mul_f32_e32 v64, v48, v64
	v_mul_f32_e32 v65, v49, v65
	v_mul_f32_e32 v66, v50, v66
	v_mul_f32_e32 v67, v51, v67
	v_mul_f32_e32 v68, v52, v68
	v_mul_f32_e32 v69, v53, v69
	v_cvt_pk_bf16_f32 v62, v62, v63
	v_cvt_pk_bf16_f32 v63, v64, v65
	v_cvt_pk_bf16_f32 v64, v66, v67
	v_cvt_pk_bf16_f32 v65, v68, v69
	v_subrev_f32_dpp v70, v127, v70 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v71, v127, v71 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v72, v127, v72 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v73, v127, v73 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	v_mfma_f32_32x32x16_bf16 v[192:207], v[102:105], v[118:121], v[192:207]
	ds_read_b64_tr_b16 a[16:17], v1 offset:16384
	ds_read_b64_tr_b16 a[18:19], v1 offset:20480
	ds_write_b64 v9, v[62:63]
	s_add_i32 s1, s22, 0x80
	ds_write_b64 v9, v[64:65] offset:512
	v_mfma_f32_32x32x16_bf16 v[208:223], v[102:105], v[122:125], v[208:223]
	v_subrev_f32_dpp v74, v127, v74 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v75, v127, v75 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v76, v127, v76 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v77, v127, v77 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	v_mul_f32_e32 v70, v54, v70
	v_mul_f32_e32 v71, v55, v71
	v_mul_f32_e32 v72, v56, v72
	v_mul_f32_e32 v73, v57, v73
	v_mul_f32_e32 v74, v58, v74
	v_mul_f32_e32 v75, v59, v75
	v_mul_f32_e32 v76, v60, v76
	v_mul_f32_e32 v77, v61, v77
	v_cvt_pk_bf16_f32 v66, v70, v71
	v_cvt_pk_bf16_f32 v67, v72, v73
	v_cvt_pk_bf16_f32 v68, v74, v75
	v_cvt_pk_bf16_f32 v69, v76, v77
	v_mfma_f32_32x32x16_bf16 v[224:239], v[106:109], v[118:121], v[224:239]
	v_add_u32_e32 v2, s1, v13
	ds_read_b32 v126, v2

	s_add_i32 s1, s23, 0x80
	v_add_u32_e32 v2, s1, v13
	ds_read_b32 v127, v2

	ds_write_b64 v9, v[66:67] offset:1024
	ds_write_b64 v9, v[68:69] offset:1536
	v_mfma_f32_32x32x16_bf16 v[240:255], v[106:109], v[122:125], v[240:255]
	v_permlane16_swap_b32_e32 v62, v64
	v_permlane16_swap_b32_e32 v63, v65
	v_permlane16_swap_b32_e32 v66, v68
	v_permlane16_swap_b32_e32 v67, v69
	s_waitcnt lgkmcnt(12)
	v_mfma_f32_32x32x16_bf16 a[128:143], a[112:115], v[62:65], a[128:143]
	ds_read_b64_tr_b16 a[20:21], v1 offset:16896
	ds_read_b64_tr_b16 a[22:23], v1 offset:20992
	ds_read_b64_tr_b16 a[24:25], v1 offset:24576
	ds_read_b64_tr_b16 a[26:27], v1 offset:28672
	ds_read_b64_tr_b16 a[28:29], v1 offset:25088
	ds_read_b64_tr_b16 a[30:31], v1 offset:29184
	ds_read_b64_tr_b16 a[32:33], v1 offset:32768
	ds_read_b64_tr_b16 a[34:35], v1 offset:36864
	v_mfma_f32_32x32x16_bf16 a[144:159], a[112:115], v[66:69], a[144:159]
	s_waitcnt lgkmcnt(8)
	s_barrier
	v_mfma_f32_32x32x16_bf16 a[160:175], a[116:119], v[62:65], a[160:175]
	ds_read_b64_tr_b16 v[30:31], v4
	ds_read_b64_tr_b16 v[32:33], v4 offset:512
	ds_read_b64_tr_b16 v[34:35], v4 offset:1024
	ds_read_b64_tr_b16 v[36:37], v4 offset:1536
	s_add_i32 s0, s0, s13
	ds_read_b64_tr_b16 v[38:39], v4 offset:2048
	s_ashr_i32 s1, s0, 31
	ds_read_b64_tr_b16 v[40:41], v4 offset:2560
	s_lshl_b64 s[8:9], s[0:1], 1
	ds_read_b64_tr_b16 v[42:43], v4 offset:3072
	s_add_u32 s8, s6, s8
	ds_read_b64_tr_b16 v[44:45], v4 offset:3584
	v_mfma_f32_32x32x16_bf16 a[176:191], a[116:119], v[66:69], a[176:191]
	v_mul_f32_e32 v126, 0x3fb8aa3b, v126
	s_addc_u32 s9, s7, s9
	v_cvt_pk_bf16_f32 v110, v110, v111
	v_cvt_pk_bf16_f32 v111, v112, v113
	buffer_atomic_pk_add_bf16 v110, v5, s[8:11], 0 offen
	buffer_atomic_pk_add_bf16 v111, v6, s[8:11], 0 offen
	v_mfma_f32_32x32x16_bf16 a[192:207], a[120:123], v[62:65], a[192:207]
	ds_read_b64_tr_b16 v[46:47], v4 offset:4096
	ds_read_b64_tr_b16 v[48:49], v4 offset:4608
	ds_read_b64_tr_b16 v[50:51], v4 offset:5120
	ds_read_b64_tr_b16 v[52:53], v4 offset:5632
	ds_read_b64_tr_b16 a[36:37], v1 offset:33280
	ds_read_b64_tr_b16 a[38:39], v1 offset:37376
	ds_read_b64_tr_b16 a[40:41], v1 offset:40960
	ds_read_b64_tr_b16 a[42:43], v1 offset:45056
	v_mfma_f32_32x32x16_bf16 a[208:223], a[120:123], v[66:69], a[208:223]
	v_cvt_pk_bf16_f32 v114, v114, v115
	v_cvt_pk_bf16_f32 v115, v116, v117
	buffer_atomic_pk_add_bf16 v114, v7, s[8:11], 0 offen
	buffer_atomic_pk_add_bf16 v115, v8, s[8:11], 0 offen
	v_mfma_f32_32x32x16_bf16 a[224:239], a[124:127], v[62:65], a[224:239]
	ds_read_b64_tr_b16 v[54:55], v4 offset:6144
	ds_read_b64_tr_b16 v[56:57], v4 offset:6656
	ds_read_b64_tr_b16 v[58:59], v4 offset:7168
	ds_read_b64_tr_b16 v[60:61], v4 offset:7680
	ds_read_b64_tr_b16 a[44:45], v1 offset:41472
	ds_read_b64_tr_b16 a[46:47], v1 offset:45568
	v_mfma_f32_32x32x16_bf16 a[240:255], a[124:127], v[66:69], a[240:255]
	s_waitcnt lgkmcnt(6)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[110:113], a[0:3], v[30:33], 0
	ds_read_b64_tr_b16 v[62:63], v1 offset:49152
	ds_read_b64_tr_b16 v[64:65], v1 offset:53248
	ds_read_b64_tr_b16 v[66:67], v1 offset:49664
	ds_read_b64_tr_b16 v[68:69], v1 offset:53760
	v_mfma_f32_16x16x32_bf16 v[110:113], a[8:11], v[34:37], v[110:113]
	v_mfma_f32_16x16x32_bf16 v[110:113], a[16:19], v[38:41], v[110:113]
	ds_read_b64_tr_b16 v[70:71], v1 offset:57344
	ds_read_b64_tr_b16 v[72:73], v1 offset:61440
	ds_read_b64_tr_b16 v[74:75], v1 offset:57856
	s_add_i32 s1, s19, 0x2000
	ds_read_b64_tr_b16 v[76:77], v1 offset:61952
	v_mfma_f32_16x16x32_bf16 v[110:113], a[24:27], v[42:45], v[110:113]
	v_mfma_f32_16x16x32_bf16 v[110:113], a[32:35], v[46:49], v[110:113]
	v_add_u32_e32 v2, s1, v11
	ds_read_b128 a[112:115], v2
	ds_read_b128 a[116:119], v2 offset:1024
	v_mfma_f32_16x16x32_bf16 v[110:113], a[40:43], v[50:53], v[110:113]
	s_waitcnt lgkmcnt(4)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[110:113], v[62:65], v[54:57], v[110:113]
	ds_read_b128 a[120:123], v2 offset:2048
	ds_read_b128 a[124:127], v2 offset:3072
	v_mfma_f32_16x16x32_bf16 v[110:113], v[70:73], v[58:61], v[110:113]
	v_mfma_f32_16x16x32_bf16 v[114:117], a[4:7], v[30:33], 0
	ds_read_b128 a[0:3], v10
	ds_read_b128 a[4:7], v10 offset:1024
	v_mfma_f32_16x16x32_bf16 v[114:117], a[12:15], v[34:37], v[114:117]
	v_mfma_f32_16x16x32_bf16 v[114:117], a[20:23], v[38:41], v[114:117]
	ds_read_b128 a[8:11], v10 offset:2048
	ds_read_b128 a[12:15], v10 offset:3072
	v_mfma_f32_16x16x32_bf16 v[114:117], a[28:31], v[42:45], v[114:117]
	v_mul_f32_e32 v110, 0x3db504f3, v110
	v_mul_f32_e32 v111, 0x3db504f3, v111
	v_mfma_f32_16x16x32_bf16 v[114:117], a[36:39], v[46:49], v[114:117]
	ds_read_b128 a[16:19], v10 offset:4096
	ds_read_b128 a[20:23], v10 offset:5120
	v_mfma_f32_16x16x32_bf16 v[114:117], a[44:47], v[50:53], v[114:117]
	v_mul_f32_e32 v112, 0x3db504f3, v112
	v_mul_f32_e32 v113, 0x3db504f3, v113
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_16x16x32_bf16 v[114:117], v[66:69], v[54:57], v[114:117]
	ds_read_b128 a[24:27], v10 offset:6144
	ds_read_b128 a[28:31], v10 offset:7168
	v_mfma_f32_16x16x32_bf16 v[114:117], v[74:77], v[58:61], v[114:117]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x32_bf16 v[46:49], a[112:115], a[0:3], 0
	ds_read_b128 a[32:35], v10 offset:8192
	ds_read_b128 a[36:39], v10 offset:9216
	v_mfma_f32_16x16x32_bf16 v[46:49], a[116:119], a[4:7], v[46:49]
	v_mfma_f32_16x16x32_bf16 v[46:49], a[120:123], a[8:11], v[46:49]
	ds_read_b128 a[40:43], v10 offset:10240
	ds_read_b128 a[44:47], v10 offset:11264
	v_mfma_f32_16x16x32_bf16 v[46:49], a[124:127], a[12:15], v[46:49]
	v_mul_f32_e32 v114, 0x3db504f3, v114
	v_mul_f32_e32 v115, 0x3db504f3, v115
	v_mfma_f32_16x16x32_bf16 v[50:53], a[112:115], a[16:19], 0
	ds_read_b128 v[62:65], v10 offset:12288
	ds_read_b128 v[66:69], v10 offset:13312
	v_mfma_f32_16x16x32_bf16 v[50:53], a[116:119], a[20:23], v[50:53]
	v_mul_f32_e32 v116, 0x3db504f3, v116
	v_mul_f32_e32 v117, 0x3db504f3, v117
	v_mfma_f32_16x16x32_bf16 v[50:53], a[120:123], a[24:27], v[50:53]
	ds_read_b128 v[70:73], v10 offset:14336
	s_add_i32 s8, s17, 0x2000
	ds_read_b128 v[74:77], v10 offset:15360
	v_mfma_f32_16x16x32_bf16 v[50:53], a[124:127], a[28:31], v[50:53]
	v_mul_f32_e32 v46, 0x3e0293ee, v46
	v_mul_f32_e32 v47, 0x3e0293ee, v47
	v_mul_f32_e32 v48, 0x3e0293ee, v48
	v_mul_f32_e32 v49, 0x3e0293ee, v49
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_bf16 v[54:57], a[112:115], a[32:35], 0
	v_add_u32_e32 v2, s8, v11
	ds_read_b128 v[78:81], v2
	ds_read_b128 v[82:85], v2 offset:1024
	v_mfma_f32_16x16x32_bf16 v[54:57], a[116:119], a[36:39], v[54:57]
	v_subrev_f32_dpp v46, v126, v46 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v47, v126, v47 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v48, v126, v48 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v49, v126, v49 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_bf16 v[54:57], a[120:123], a[40:43], v[54:57]
	ds_read_b128 v[86:89], v2 offset:2048
	ds_read_b128 v[90:93], v2 offset:3072
	v_mfma_f32_16x16x32_bf16 v[54:57], a[124:127], a[44:47], v[54:57]
	v_mul_f32_e32 v50, 0x3e0293ee, v50
	v_mul_f32_e32 v51, 0x3e0293ee, v51
	v_mul_f32_e32 v52, 0x3e0293ee, v52
	v_mul_f32_e32 v53, 0x3e0293ee, v53
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_bf16 v[58:61], a[112:115], v[62:65], 0
	v_add_u32_e32 v2, s8, v12
	ds_read_b64_tr_b16 v[94:95], v2
	ds_read_b64_tr_b16 v[96:97], v2 offset:256
	ds_read_b64_tr_b16 v[98:99], v2 offset:1024
	ds_read_b64_tr_b16 v[100:101], v2 offset:1280
	v_mfma_f32_16x16x32_bf16 v[58:61], a[116:119], v[66:69], v[58:61]
	v_subrev_f32_dpp v50, v126, v50 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v51, v126, v51 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v52, v126, v52 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v53, v126, v53 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_16x16x32_bf16 v[58:61], a[120:123], v[70:73], v[58:61]
	ds_read_b64_tr_b16 v[102:103], v2 offset:2048
	ds_read_b64_tr_b16 v[104:105], v2 offset:2304
	ds_read_b64_tr_b16 v[106:107], v2 offset:3072
	ds_read_b64_tr_b16 v[108:109], v2 offset:3328
	v_mfma_f32_16x16x32_bf16 v[58:61], a[124:127], v[74:77], v[58:61]
	v_mul_f32_e32 v54, 0x3e0293ee, v54
	v_mul_f32_e32 v55, 0x3e0293ee, v55
	v_mul_f32_e32 v56, 0x3e0293ee, v56
	v_mul_f32_e32 v57, 0x3e0293ee, v57
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_16x16x32_bf16 v[62:65], v[78:81], a[48:51], 0
	v_subrev_f32_dpp v54, v126, v54 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v55, v126, v55 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v56, v126, v56 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v57, v126, v57 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	v_mfma_f32_16x16x32_bf16 v[62:65], v[82:85], a[52:55], v[62:65]
	v_exp_f32_e32 v46, v46
	v_exp_f32_e32 v47, v47
	v_exp_f32_e32 v48, v48
	v_exp_f32_e32 v49, v49
	v_mfma_f32_16x16x32_bf16 v[62:65], v[86:89], a[56:59], v[62:65]
	v_add_u32_e32 v2, s1, v12
	ds_read_b64_tr_b16 a[112:113], v2
	ds_read_b64_tr_b16 a[114:115], v2 offset:256
	v_mfma_f32_16x16x32_bf16 v[62:65], v[90:93], a[60:63], v[62:65]
	v_exp_f32_e32 v50, v50
	v_exp_f32_e32 v51, v51
	v_exp_f32_e32 v52, v52
	v_exp_f32_e32 v53, v53
	v_mfma_f32_16x16x32_bf16 v[66:69], v[78:81], a[64:67], 0
	ds_read_b64_tr_b16 a[116:117], v2 offset:1024
	ds_read_b64_tr_b16 a[118:119], v2 offset:1280
	v_mfma_f32_16x16x32_bf16 v[66:69], v[82:85], a[68:71], v[66:69]
	v_mul_f32_e32 v58, 0x3e0293ee, v58
	v_mul_f32_e32 v59, 0x3e0293ee, v59
	v_mul_f32_e32 v60, 0x3e0293ee, v60
	v_mul_f32_e32 v61, 0x3e0293ee, v61
	v_mfma_f32_16x16x32_bf16 v[66:69], v[86:89], a[72:75], v[66:69]
	v_subrev_f32_dpp v58, v126, v58 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v59, v126, v59 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v60, v126, v60 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v61, v126, v61 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	v_mfma_f32_16x16x32_bf16 v[66:69], v[90:93], a[76:79], v[66:69]
	v_cvt_pk_bf16_f32 v118, v46, v47
	v_cvt_pk_bf16_f32 v119, v48, v49
	v_mfma_f32_16x16x32_bf16 v[70:73], v[78:81], a[80:83], 0
	v_exp_f32_e32 v54, v54
	v_exp_f32_e32 v55, v55
	v_exp_f32_e32 v56, v56
	v_exp_f32_e32 v57, v57
	v_mfma_f32_16x16x32_bf16 v[70:73], v[82:85], a[84:87], v[70:73]
	v_cvt_pk_bf16_f32 v120, v50, v51
	v_cvt_pk_bf16_f32 v121, v52, v53
	v_mfma_f32_16x16x32_bf16 v[70:73], v[86:89], a[88:91], v[70:73]
	ds_read_b64_tr_b16 a[120:121], v2 offset:2048
	ds_read_b64_tr_b16 a[122:123], v2 offset:2304
	v_mfma_f32_16x16x32_bf16 v[70:73], v[90:93], a[92:95], v[70:73]
	v_exp_f32_e32 v58, v58
	v_exp_f32_e32 v59, v59
	v_exp_f32_e32 v60, v60
	v_exp_f32_e32 v61, v61
	v_mfma_f32_16x16x32_bf16 v[74:77], v[78:81], a[96:99], 0
	ds_read_b64_tr_b16 a[124:125], v2 offset:3072
	ds_read_b64_tr_b16 a[126:127], v2 offset:3328
	v_mfma_f32_16x16x32_bf16 v[74:77], v[82:85], a[100:103], v[74:77]
	v_cvt_pk_bf16_f32 v122, v54, v55
	v_cvt_pk_bf16_f32 v123, v56, v57
	v_cvt_pk_bf16_f32 v124, v58, v59
	v_cvt_pk_bf16_f32 v125, v60, v61
	v_mfma_f32_16x16x32_bf16 v[74:77], v[86:89], a[104:107], v[74:77]
	v_permlane16_swap_b32_e32 v118, v120
	v_permlane16_swap_b32_e32 v119, v121
	v_permlane16_swap_b32_e32 v122, v124
	v_permlane16_swap_b32_e32 v123, v125
	v_mfma_f32_16x16x32_bf16 v[74:77], v[90:93], a[108:111], v[74:77]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x16_bf16 v[128:143], v[94:97], v[118:121], v[128:143]
	ds_read_b64_tr_b16 a[0:1], v1
	ds_read_b64_tr_b16 a[2:3], v1 offset:4096
	ds_read_b64_tr_b16 a[4:5], v1 offset:512
	ds_read_b64_tr_b16 a[6:7], v1 offset:4608
	v_mfma_f32_32x32x16_bf16 v[144:159], v[94:97], v[122:125], v[144:159]
	v_subrev_f32_dpp v62, v127, v62 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v63, v127, v63 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v64, v127, v64 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v65, v127, v65 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v66, v127, v66 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v67, v127, v67 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v68, v127, v68 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v69, v127, v69 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	v_mfma_f32_32x32x16_bf16 v[160:175], v[98:101], v[118:121], v[160:175]
	ds_read_b64_tr_b16 a[8:9], v1 offset:8192
	ds_read_b64_tr_b16 a[10:11], v1 offset:12288
	ds_read_b64_tr_b16 a[12:13], v1 offset:8704
	ds_read_b64_tr_b16 a[14:15], v1 offset:12800
	v_mfma_f32_32x32x16_bf16 v[176:191], v[98:101], v[122:125], v[176:191]
	v_mul_f32_e32 v62, v46, v62
	v_mul_f32_e32 v63, v47, v63
	v_mul_f32_e32 v64, v48, v64
	v_mul_f32_e32 v65, v49, v65
	v_mul_f32_e32 v66, v50, v66
	v_mul_f32_e32 v67, v51, v67
	v_mul_f32_e32 v68, v52, v68
	v_mul_f32_e32 v69, v53, v69
	v_cvt_pk_bf16_f32 v62, v62, v63
	v_cvt_pk_bf16_f32 v63, v64, v65
	v_cvt_pk_bf16_f32 v64, v66, v67
	v_cvt_pk_bf16_f32 v65, v68, v69
	v_subrev_f32_dpp v70, v127, v70 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v71, v127, v71 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v72, v127, v72 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v73, v127, v73 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	v_mfma_f32_32x32x16_bf16 v[192:207], v[102:105], v[118:121], v[192:207]
	ds_read_b64_tr_b16 a[16:17], v1 offset:16384
	ds_read_b64_tr_b16 a[18:19], v1 offset:20480
	ds_write_b64 v9, v[62:63]
	s_addk_i32 s22, 0xc0
	ds_write_b64 v9, v[64:65] offset:512
	v_mfma_f32_32x32x16_bf16 v[208:223], v[102:105], v[122:125], v[208:223]
	v_subrev_f32_dpp v74, v127, v74 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v75, v127, v75 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v76, v127, v76 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v77, v127, v77 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	v_mul_f32_e32 v70, v54, v70
	v_mul_f32_e32 v71, v55, v71
	v_mul_f32_e32 v72, v56, v72
	v_mul_f32_e32 v73, v57, v73
	v_mul_f32_e32 v74, v58, v74
	v_mul_f32_e32 v75, v59, v75
	v_mul_f32_e32 v76, v60, v76
	v_mul_f32_e32 v77, v61, v77
	v_cvt_pk_bf16_f32 v66, v70, v71
	v_cvt_pk_bf16_f32 v67, v72, v73
	v_cvt_pk_bf16_f32 v68, v74, v75
	v_cvt_pk_bf16_f32 v69, v76, v77
	v_mfma_f32_32x32x16_bf16 v[224:239], v[106:109], v[118:121], v[224:239]
	v_add_u32_e32 v2, s22, v13
	ds_read_b32 v126, v2

	s_addk_i32 s23, 0xc0
	v_add_u32_e32 v2, s23, v13
	ds_read_b32 v127, v2

	ds_write_b64 v9, v[66:67] offset:1024
	ds_write_b64 v9, v[68:69] offset:1536
	v_mfma_f32_32x32x16_bf16 v[240:255], v[106:109], v[122:125], v[240:255]
	v_permlane16_swap_b32_e32 v62, v64
	v_permlane16_swap_b32_e32 v63, v65
	v_permlane16_swap_b32_e32 v66, v68
	v_permlane16_swap_b32_e32 v67, v69
	s_waitcnt lgkmcnt(12)
	v_mfma_f32_32x32x16_bf16 a[128:143], a[112:115], v[62:65], a[128:143]
	ds_read_b64_tr_b16 a[20:21], v1 offset:16896
	ds_read_b64_tr_b16 a[22:23], v1 offset:20992
	ds_read_b64_tr_b16 a[24:25], v1 offset:24576
	ds_read_b64_tr_b16 a[26:27], v1 offset:28672
	ds_read_b64_tr_b16 a[28:29], v1 offset:25088
	ds_read_b64_tr_b16 a[30:31], v1 offset:29184
	ds_read_b64_tr_b16 a[32:33], v1 offset:32768
	ds_read_b64_tr_b16 a[34:35], v1 offset:36864
	v_mfma_f32_32x32x16_bf16 a[144:159], a[112:115], v[66:69], a[144:159]
	s_waitcnt lgkmcnt(8)
	s_barrier
	v_mfma_f32_32x32x16_bf16 a[160:175], a[116:119], v[62:65], a[160:175]
	ds_read_b64_tr_b16 v[30:31], v4
	ds_read_b64_tr_b16 v[32:33], v4 offset:512
	ds_read_b64_tr_b16 v[34:35], v4 offset:1024
	ds_read_b64_tr_b16 v[36:37], v4 offset:1536
	s_add_i32 s0, s0, s13
	ds_read_b64_tr_b16 v[38:39], v4 offset:2048
	s_ashr_i32 s1, s0, 31
	ds_read_b64_tr_b16 v[40:41], v4 offset:2560
	s_lshl_b64 s[8:9], s[0:1], 1
	ds_read_b64_tr_b16 v[42:43], v4 offset:3072
	s_add_u32 s8, s6, s8
	ds_read_b64_tr_b16 v[44:45], v4 offset:3584
	v_mfma_f32_32x32x16_bf16 a[176:191], a[116:119], v[66:69], a[176:191]
	v_mul_f32_e32 v126, 0x3fb8aa3b, v126
	s_addc_u32 s9, s7, s9
	v_cvt_pk_bf16_f32 v110, v110, v111
	v_cvt_pk_bf16_f32 v111, v112, v113
	buffer_atomic_pk_add_bf16 v110, v5, s[8:11], 0 offen
	buffer_atomic_pk_add_bf16 v111, v6, s[8:11], 0 offen
	v_mfma_f32_32x32x16_bf16 a[192:207], a[120:123], v[62:65], a[192:207]
	ds_read_b64_tr_b16 v[46:47], v4 offset:4096
	ds_read_b64_tr_b16 v[48:49], v4 offset:4608
	ds_read_b64_tr_b16 v[50:51], v4 offset:5120
	ds_read_b64_tr_b16 v[52:53], v4 offset:5632
	ds_read_b64_tr_b16 a[36:37], v1 offset:33280
	ds_read_b64_tr_b16 a[38:39], v1 offset:37376
	ds_read_b64_tr_b16 a[40:41], v1 offset:40960
	ds_read_b64_tr_b16 a[42:43], v1 offset:45056
	v_mfma_f32_32x32x16_bf16 a[208:223], a[120:123], v[66:69], a[208:223]
	v_cvt_pk_bf16_f32 v114, v114, v115
	v_cvt_pk_bf16_f32 v115, v116, v117
	buffer_atomic_pk_add_bf16 v114, v7, s[8:11], 0 offen
	buffer_atomic_pk_add_bf16 v115, v8, s[8:11], 0 offen
	v_mfma_f32_32x32x16_bf16 a[224:239], a[124:127], v[62:65], a[224:239]
	ds_read_b64_tr_b16 v[54:55], v4 offset:6144
	ds_read_b64_tr_b16 v[56:57], v4 offset:6656
	ds_read_b64_tr_b16 v[58:59], v4 offset:7168
	ds_read_b64_tr_b16 v[60:61], v4 offset:7680
	ds_read_b64_tr_b16 a[44:45], v1 offset:41472
	ds_read_b64_tr_b16 a[46:47], v1 offset:45568
	v_mfma_f32_32x32x16_bf16 a[240:255], a[124:127], v[66:69], a[240:255]
	s_waitcnt lgkmcnt(6)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[110:113], a[0:3], v[30:33], 0
	ds_read_b64_tr_b16 v[62:63], v1 offset:49152
	ds_read_b64_tr_b16 v[64:65], v1 offset:53248
	ds_read_b64_tr_b16 v[66:67], v1 offset:49664
	ds_read_b64_tr_b16 v[68:69], v1 offset:53760
	v_mfma_f32_16x16x32_bf16 v[110:113], a[8:11], v[34:37], v[110:113]
	v_mfma_f32_16x16x32_bf16 v[110:113], a[16:19], v[38:41], v[110:113]
	ds_read_b64_tr_b16 v[70:71], v1 offset:57344
	ds_read_b64_tr_b16 v[72:73], v1 offset:61440
	ds_read_b64_tr_b16 v[74:75], v1 offset:57856
	s_addk_i32 s19, 0x3000
	ds_read_b64_tr_b16 v[76:77], v1 offset:61952
	v_mfma_f32_16x16x32_bf16 v[110:113], a[24:27], v[42:45], v[110:113]
	v_mfma_f32_16x16x32_bf16 v[110:113], a[32:35], v[46:49], v[110:113]
	v_add_u32_e32 v2, s19, v11
	ds_read_b128 a[112:115], v2
	ds_read_b128 a[116:119], v2 offset:1024
	v_mfma_f32_16x16x32_bf16 v[110:113], a[40:43], v[50:53], v[110:113]
	s_waitcnt lgkmcnt(4)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[110:113], v[62:65], v[54:57], v[110:113]
	ds_read_b128 a[120:123], v2 offset:2048
	ds_read_b128 a[124:127], v2 offset:3072
	v_mfma_f32_16x16x32_bf16 v[110:113], v[70:73], v[58:61], v[110:113]
	v_mfma_f32_16x16x32_bf16 v[114:117], a[4:7], v[30:33], 0
	ds_read_b128 a[0:3], v10
	ds_read_b128 a[4:7], v10 offset:1024
	v_mfma_f32_16x16x32_bf16 v[114:117], a[12:15], v[34:37], v[114:117]
	v_mfma_f32_16x16x32_bf16 v[114:117], a[20:23], v[38:41], v[114:117]
	ds_read_b128 a[8:11], v10 offset:2048
	ds_read_b128 a[12:15], v10 offset:3072
	v_mfma_f32_16x16x32_bf16 v[114:117], a[28:31], v[42:45], v[114:117]
	v_mul_f32_e32 v110, 0x3db504f3, v110
	v_mul_f32_e32 v111, 0x3db504f3, v111
	v_mfma_f32_16x16x32_bf16 v[114:117], a[36:39], v[46:49], v[114:117]
	ds_read_b128 a[16:19], v10 offset:4096
	ds_read_b128 a[20:23], v10 offset:5120
	v_mfma_f32_16x16x32_bf16 v[114:117], a[44:47], v[50:53], v[114:117]
	v_mul_f32_e32 v112, 0x3db504f3, v112
	v_mul_f32_e32 v113, 0x3db504f3, v113
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_16x16x32_bf16 v[114:117], v[66:69], v[54:57], v[114:117]
	ds_read_b128 a[24:27], v10 offset:6144
	ds_read_b128 a[28:31], v10 offset:7168
	v_mfma_f32_16x16x32_bf16 v[114:117], v[74:77], v[58:61], v[114:117]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x32_bf16 v[46:49], a[112:115], a[0:3], 0
	ds_read_b128 a[32:35], v10 offset:8192
	ds_read_b128 a[36:39], v10 offset:9216
	v_mfma_f32_16x16x32_bf16 v[46:49], a[116:119], a[4:7], v[46:49]
	v_mfma_f32_16x16x32_bf16 v[46:49], a[120:123], a[8:11], v[46:49]
	ds_read_b128 a[40:43], v10 offset:10240
	ds_read_b128 a[44:47], v10 offset:11264
	v_mfma_f32_16x16x32_bf16 v[46:49], a[124:127], a[12:15], v[46:49]
	v_mul_f32_e32 v114, 0x3db504f3, v114
	v_mul_f32_e32 v115, 0x3db504f3, v115
	v_mfma_f32_16x16x32_bf16 v[50:53], a[112:115], a[16:19], 0
	ds_read_b128 v[62:65], v10 offset:12288
	ds_read_b128 v[66:69], v10 offset:13312
	v_mfma_f32_16x16x32_bf16 v[50:53], a[116:119], a[20:23], v[50:53]
	v_mul_f32_e32 v116, 0x3db504f3, v116
	v_mul_f32_e32 v117, 0x3db504f3, v117
	v_mfma_f32_16x16x32_bf16 v[50:53], a[120:123], a[24:27], v[50:53]
	ds_read_b128 v[70:73], v10 offset:14336
	s_addk_i32 s17, 0x3000
	ds_read_b128 v[74:77], v10 offset:15360
	v_mfma_f32_16x16x32_bf16 v[50:53], a[124:127], a[28:31], v[50:53]
	v_mul_f32_e32 v46, 0x3e0293ee, v46
	v_mul_f32_e32 v47, 0x3e0293ee, v47
	v_mul_f32_e32 v48, 0x3e0293ee, v48
	v_mul_f32_e32 v49, 0x3e0293ee, v49
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_bf16 v[54:57], a[112:115], a[32:35], 0
	v_add_u32_e32 v2, s17, v11
	ds_read_b128 v[78:81], v2
	ds_read_b128 v[82:85], v2 offset:1024
	v_mfma_f32_16x16x32_bf16 v[54:57], a[116:119], a[36:39], v[54:57]
	v_subrev_f32_dpp v46, v126, v46 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v47, v126, v47 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v48, v126, v48 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v49, v126, v49 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_bf16 v[54:57], a[120:123], a[40:43], v[54:57]
	ds_read_b128 v[86:89], v2 offset:2048
	ds_read_b128 v[90:93], v2 offset:3072
	v_mfma_f32_16x16x32_bf16 v[54:57], a[124:127], a[44:47], v[54:57]
	v_mul_f32_e32 v50, 0x3e0293ee, v50
	v_mul_f32_e32 v51, 0x3e0293ee, v51
	v_mul_f32_e32 v52, 0x3e0293ee, v52
	v_mul_f32_e32 v53, 0x3e0293ee, v53
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_bf16 v[58:61], a[112:115], v[62:65], 0
	v_add_u32_e32 v2, s17, v12
	ds_read_b64_tr_b16 v[94:95], v2
	ds_read_b64_tr_b16 v[96:97], v2 offset:256
	ds_read_b64_tr_b16 v[98:99], v2 offset:1024
	ds_read_b64_tr_b16 v[100:101], v2 offset:1280
	v_mfma_f32_16x16x32_bf16 v[58:61], a[116:119], v[66:69], v[58:61]
	v_subrev_f32_dpp v50, v126, v50 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v51, v126, v51 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v52, v126, v52 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v53, v126, v53 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_16x16x32_bf16 v[58:61], a[120:123], v[70:73], v[58:61]
	ds_read_b64_tr_b16 v[102:103], v2 offset:2048
	ds_read_b64_tr_b16 v[104:105], v2 offset:2304
	ds_read_b64_tr_b16 v[106:107], v2 offset:3072
	ds_read_b64_tr_b16 v[108:109], v2 offset:3328
	v_mfma_f32_16x16x32_bf16 v[58:61], a[124:127], v[74:77], v[58:61]
	v_mul_f32_e32 v54, 0x3e0293ee, v54
	v_mul_f32_e32 v55, 0x3e0293ee, v55
	v_mul_f32_e32 v56, 0x3e0293ee, v56
	v_mul_f32_e32 v57, 0x3e0293ee, v57
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_16x16x32_bf16 v[62:65], v[78:81], a[48:51], 0
	v_subrev_f32_dpp v54, v126, v54 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v55, v126, v55 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v56, v126, v56 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v57, v126, v57 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	v_mfma_f32_16x16x32_bf16 v[62:65], v[82:85], a[52:55], v[62:65]
	v_exp_f32_e32 v46, v46
	v_exp_f32_e32 v47, v47
	v_exp_f32_e32 v48, v48
	v_exp_f32_e32 v49, v49
	v_mfma_f32_16x16x32_bf16 v[62:65], v[86:89], a[56:59], v[62:65]
	v_add_u32_e32 v2, s19, v12
	ds_read_b64_tr_b16 a[112:113], v2
	ds_read_b64_tr_b16 a[114:115], v2 offset:256
	v_mfma_f32_16x16x32_bf16 v[62:65], v[90:93], a[60:63], v[62:65]
	v_exp_f32_e32 v50, v50
	v_exp_f32_e32 v51, v51
	v_exp_f32_e32 v52, v52
	v_exp_f32_e32 v53, v53
	v_mfma_f32_16x16x32_bf16 v[66:69], v[78:81], a[64:67], 0
	ds_read_b64_tr_b16 a[116:117], v2 offset:1024
	ds_read_b64_tr_b16 a[118:119], v2 offset:1280
	v_mfma_f32_16x16x32_bf16 v[66:69], v[82:85], a[68:71], v[66:69]
	v_mul_f32_e32 v58, 0x3e0293ee, v58
	v_mul_f32_e32 v59, 0x3e0293ee, v59
	v_mul_f32_e32 v60, 0x3e0293ee, v60
	v_mul_f32_e32 v61, 0x3e0293ee, v61
	v_mfma_f32_16x16x32_bf16 v[66:69], v[86:89], a[72:75], v[66:69]
	v_subrev_f32_dpp v58, v126, v58 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v59, v126, v59 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v60, v126, v60 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v61, v126, v61 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	v_mfma_f32_16x16x32_bf16 v[66:69], v[90:93], a[76:79], v[66:69]
	v_cvt_pk_bf16_f32 v118, v46, v47
	v_cvt_pk_bf16_f32 v119, v48, v49
	v_mfma_f32_16x16x32_bf16 v[70:73], v[78:81], a[80:83], 0
	v_exp_f32_e32 v54, v54
	v_exp_f32_e32 v55, v55
	v_exp_f32_e32 v56, v56
	v_exp_f32_e32 v57, v57
	v_mfma_f32_16x16x32_bf16 v[70:73], v[82:85], a[84:87], v[70:73]
	v_cvt_pk_bf16_f32 v120, v50, v51
	v_cvt_pk_bf16_f32 v121, v52, v53
	v_mfma_f32_16x16x32_bf16 v[70:73], v[86:89], a[88:91], v[70:73]
	ds_read_b64_tr_b16 a[120:121], v2 offset:2048
	ds_read_b64_tr_b16 a[122:123], v2 offset:2304
	v_mfma_f32_16x16x32_bf16 v[70:73], v[90:93], a[92:95], v[70:73]
	v_exp_f32_e32 v58, v58
	v_exp_f32_e32 v59, v59
	v_exp_f32_e32 v60, v60
	v_exp_f32_e32 v61, v61
	v_mfma_f32_16x16x32_bf16 v[74:77], v[78:81], a[96:99], 0
	ds_read_b64_tr_b16 a[124:125], v2 offset:3072
	ds_read_b64_tr_b16 a[126:127], v2 offset:3328
	v_mfma_f32_16x16x32_bf16 v[74:77], v[82:85], a[100:103], v[74:77]
	v_cvt_pk_bf16_f32 v122, v54, v55
	v_cvt_pk_bf16_f32 v123, v56, v57
	v_cvt_pk_bf16_f32 v124, v58, v59
	v_cvt_pk_bf16_f32 v125, v60, v61
	v_mfma_f32_16x16x32_bf16 v[74:77], v[86:89], a[104:107], v[74:77]
	v_permlane16_swap_b32_e32 v118, v120
	v_permlane16_swap_b32_e32 v119, v121
	v_permlane16_swap_b32_e32 v122, v124
	v_permlane16_swap_b32_e32 v123, v125
	v_mfma_f32_16x16x32_bf16 v[74:77], v[90:93], a[108:111], v[74:77]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x16_bf16 v[128:143], v[94:97], v[118:121], v[128:143]
	ds_read_b64_tr_b16 a[0:1], v1
	ds_read_b64_tr_b16 a[2:3], v1 offset:4096
	ds_read_b64_tr_b16 a[4:5], v1 offset:512
	ds_read_b64_tr_b16 a[6:7], v1 offset:4608
	v_mfma_f32_32x32x16_bf16 v[144:159], v[94:97], v[122:125], v[144:159]
	v_subrev_f32_dpp v62, v127, v62 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v63, v127, v63 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v64, v127, v64 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v65, v127, v65 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v66, v127, v66 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v67, v127, v67 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v68, v127, v68 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v69, v127, v69 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	v_mfma_f32_32x32x16_bf16 v[160:175], v[98:101], v[118:121], v[160:175]
	ds_read_b64_tr_b16 a[8:9], v1 offset:8192
	ds_read_b64_tr_b16 a[10:11], v1 offset:12288
	ds_read_b64_tr_b16 a[12:13], v1 offset:8704
	ds_read_b64_tr_b16 a[14:15], v1 offset:12800
	v_mfma_f32_32x32x16_bf16 v[176:191], v[98:101], v[122:125], v[176:191]
	v_mul_f32_e32 v62, v46, v62
	v_mul_f32_e32 v63, v47, v63
	v_mul_f32_e32 v64, v48, v64
	v_mul_f32_e32 v65, v49, v65
	v_mul_f32_e32 v66, v50, v66
	v_mul_f32_e32 v67, v51, v67
	v_mul_f32_e32 v68, v52, v68
	v_mul_f32_e32 v69, v53, v69
	v_cvt_pk_bf16_f32 v62, v62, v63
	v_cvt_pk_bf16_f32 v63, v64, v65
	v_cvt_pk_bf16_f32 v64, v66, v67
	v_cvt_pk_bf16_f32 v65, v68, v69
	v_subrev_f32_dpp v70, v127, v70 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v71, v127, v71 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v72, v127, v72 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v73, v127, v73 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	v_mfma_f32_32x32x16_bf16 v[192:207], v[102:105], v[118:121], v[192:207]
	ds_read_b64_tr_b16 a[16:17], v1 offset:16384
	ds_read_b64_tr_b16 a[18:19], v1 offset:20480
	ds_write_b64 v9, v[62:63]
	ds_write_b64 v9, v[64:65] offset:512
	v_mfma_f32_32x32x16_bf16 v[208:223], v[102:105], v[122:125], v[208:223]
	v_subrev_f32_dpp v74, v127, v74 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v75, v127, v75 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v76, v127, v76 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
	v_subrev_f32_dpp v77, v127, v77 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
	v_mul_f32_e32 v70, v54, v70
	v_mul_f32_e32 v71, v55, v71
	v_mul_f32_e32 v72, v56, v72
	v_mul_f32_e32 v73, v57, v73
	v_mul_f32_e32 v74, v58, v74
	v_mul_f32_e32 v75, v59, v75
	v_mul_f32_e32 v76, v60, v76
	v_mul_f32_e32 v77, v61, v77
	v_cvt_pk_bf16_f32 v66, v70, v71
	v_cvt_pk_bf16_f32 v67, v72, v73
	v_cvt_pk_bf16_f32 v68, v74, v75
	v_cvt_pk_bf16_f32 v69, v76, v77
	v_mfma_f32_32x32x16_bf16 v[224:239], v[106:109], v[118:121], v[224:239]
	ds_write_b64 v9, v[66:67] offset:1024
	ds_write_b64 v9, v[68:69] offset:1536
	v_mfma_f32_32x32x16_bf16 v[240:255], v[106:109], v[122:125], v[240:255]
	v_permlane16_swap_b32_e32 v62, v64
	v_permlane16_swap_b32_e32 v63, v65
	v_permlane16_swap_b32_e32 v66, v68
	v_permlane16_swap_b32_e32 v67, v69
	s_waitcnt lgkmcnt(12)
	v_mfma_f32_32x32x16_bf16 a[128:143], a[112:115], v[62:65], a[128:143]
	ds_read_b64_tr_b16 a[20:21], v1 offset:16896
	ds_read_b64_tr_b16 a[22:23], v1 offset:20992
	ds_read_b64_tr_b16 a[24:25], v1 offset:24576
	ds_read_b64_tr_b16 a[26:27], v1 offset:28672
	ds_read_b64_tr_b16 a[28:29], v1 offset:25088
	ds_read_b64_tr_b16 a[30:31], v1 offset:29184
	ds_read_b64_tr_b16 a[32:33], v1 offset:32768
	ds_read_b64_tr_b16 a[34:35], v1 offset:36864
	v_mfma_f32_32x32x16_bf16 a[144:159], a[112:115], v[66:69], a[144:159]
	s_waitcnt lgkmcnt(8)
	s_barrier
	v_mfma_f32_32x32x16_bf16 a[160:175], a[116:119], v[62:65], a[160:175]
	ds_read_b64_tr_b16 v[30:31], v4
	ds_read_b64_tr_b16 v[32:33], v4 offset:512
	ds_read_b64_tr_b16 v[34:35], v4 offset:1024
	ds_read_b64_tr_b16 v[36:37], v4 offset:1536
	s_add_i32 s0, s0, s13
	ds_read_b64_tr_b16 v[38:39], v4 offset:2048
	s_ashr_i32 s1, s0, 31
	ds_read_b64_tr_b16 v[40:41], v4 offset:2560
	s_lshl_b64 s[0:1], s[0:1], 1
	ds_read_b64_tr_b16 v[42:43], v4 offset:3072
	s_add_u32 s8, s6, s0
	ds_read_b64_tr_b16 v[44:45], v4 offset:3584
	v_mfma_f32_32x32x16_bf16 a[176:191], a[116:119], v[66:69], a[176:191]
	s_addc_u32 s9, s7, s1
	v_cvt_pk_bf16_f32 v110, v110, v111
	v_cvt_pk_bf16_f32 v111, v112, v113
	buffer_atomic_pk_add_bf16 v110, v5, s[8:11], 0 offen
	buffer_atomic_pk_add_bf16 v111, v6, s[8:11], 0 offen
	v_mfma_f32_32x32x16_bf16 a[192:207], a[120:123], v[62:65], a[192:207]
	ds_read_b64_tr_b16 v[46:47], v4 offset:4096
	ds_read_b64_tr_b16 v[48:49], v4 offset:4608
	ds_read_b64_tr_b16 v[50:51], v4 offset:5120
	ds_read_b64_tr_b16 v[52:53], v4 offset:5632
	ds_read_b64_tr_b16 a[36:37], v1 offset:33280
	ds_read_b64_tr_b16 a[38:39], v1 offset:37376
	ds_read_b64_tr_b16 a[40:41], v1 offset:40960
	ds_read_b64_tr_b16 a[42:43], v1 offset:45056
	v_mfma_f32_32x32x16_bf16 a[208:223], a[120:123], v[66:69], a[208:223]
	v_cvt_pk_bf16_f32 v114, v114, v115
	v_cvt_pk_bf16_f32 v115, v116, v117
	buffer_atomic_pk_add_bf16 v114, v7, s[8:11], 0 offen
	buffer_atomic_pk_add_bf16 v115, v8, s[8:11], 0 offen
	v_mfma_f32_32x32x16_bf16 a[224:239], a[124:127], v[62:65], a[224:239]
	ds_read_b64_tr_b16 v[54:55], v4 offset:6144
	ds_read_b64_tr_b16 v[56:57], v4 offset:6656
	ds_read_b64_tr_b16 v[58:59], v4 offset:7168
	ds_read_b64_tr_b16 v[60:61], v4 offset:7680
	ds_read_b64_tr_b16 a[44:45], v1 offset:41472
	s_mul_i32 s0, s4, s16
	ds_read_b64_tr_b16 a[46:47], v1 offset:45568
	v_mfma_f32_32x32x16_bf16 a[240:255], a[124:127], v[66:69], a[240:255]
	s_waitcnt lgkmcnt(6)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[110:113], a[0:3], v[30:33], 0
	ds_read_b64_tr_b16 v[62:63], v1 offset:49152
	s_mul_i32 s0, s0, s18
	ds_read_b64_tr_b16 v[64:65], v1 offset:53248
	s_add_i32 s0, s0, s2
	ds_read_b64_tr_b16 v[66:67], v1 offset:49664
	s_mul_i32 s0, s0, s34
	ds_read_b64_tr_b16 v[68:69], v1 offset:53760
	v_mfma_f32_16x16x32_bf16 v[110:113], a[8:11], v[34:37], v[110:113]
	v_mfma_f32_16x16x32_bf16 v[110:113], a[16:19], v[38:41], v[110:113]
	ds_read_b64_tr_b16 v[70:71], v1 offset:57344
	s_ashr_i32 s1, s0, 31
	ds_read_b64_tr_b16 v[72:73], v1 offset:61440
	s_lshl_b64 s[0:1], s[0:1], 1
	s_mul_i32 s8, s18, s34
	ds_read_b64_tr_b16 v[74:75], v1 offset:57856
	s_add_u32 s16, s48, s0
	v_mul_lo_u32 v4, s8, v3
	v_and_b32_e32 v23, 31, v0
	v_lshrrev_b32_e32 v0, 3, v0
	ds_read_b64_tr_b16 v[76:77], v1 offset:61952
	s_addc_u32 s17, s49, s1
	v_and_b32_e32 v2, 4, v0
	v_mad_u64_u32 v[0:1], s[0:1], v23, s8, v[4:5]
	v_add_lshl_u32 v1, v0, v2, 1
	v_or_b32_e32 v10, 8, v2
	v_mfma_f32_16x16x32_bf16 v[110:113], a[24:27], v[42:45], v[110:113]
	v_mfma_f32_16x16x32_bf16 v[110:113], a[32:35], v[46:49], v[110:113]
	v_mfma_f32_16x16x32_bf16 v[110:113], a[40:43], v[50:53], v[110:113]
	s_waitcnt lgkmcnt(4)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[110:113], v[62:65], v[54:57], v[110:113]
	v_mfma_f32_16x16x32_bf16 v[110:113], v[70:73], v[58:61], v[110:113]
	v_mfma_f32_16x16x32_bf16 v[114:117], a[4:7], v[30:33], 0
	v_mfma_f32_16x16x32_bf16 v[114:117], a[12:15], v[34:37], v[114:117]
	v_mfma_f32_16x16x32_bf16 v[114:117], a[20:23], v[38:41], v[114:117]
	v_mfma_f32_16x16x32_bf16 v[114:117], a[28:31], v[42:45], v[114:117]
	v_mfma_f32_16x16x32_bf16 v[114:117], a[36:39], v[46:49], v[114:117]
	v_mfma_f32_16x16x32_bf16 v[114:117], a[44:47], v[50:53], v[114:117]
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_16x16x32_bf16 v[114:117], v[66:69], v[54:57], v[114:117]
	v_mfma_f32_16x16x32_bf16 v[114:117], v[74:77], v[58:61], v[114:117]
	s_waitcnt lgkmcnt(2)
	s_mov_b32 s18, -1
	s_mov_b32 s19, s11
	v_cvt_pk_bf16_f32 v128, v128, v129
	v_cvt_pk_bf16_f32 v129, v130, v131
	v_cvt_pk_bf16_f32 v130, v132, v133
	v_cvt_pk_bf16_f32 v131, v134, v135
	v_cvt_pk_bf16_f32 v132, v136, v137
	v_cvt_pk_bf16_f32 v133, v138, v139
	v_cvt_pk_bf16_f32 v134, v140, v141
	v_cvt_pk_bf16_f32 v135, v142, v143
	buffer_store_dwordx2 v[128:129], v1, s[16:19], 0 offen
	v_add_lshl_u32 v1, v0, v10, 1
	v_or_b32_e32 v11, 16, v2
	buffer_store_dwordx2 v[130:131], v1, s[16:19], 0 offen
	v_add_lshl_u32 v1, v0, v11, 1
	v_or_b32_e32 v12, 24, v2
	buffer_store_dwordx2 v[132:133], v1, s[16:19], 0 offen
	v_add_lshl_u32 v1, v0, v12, 1
	v_or_b32_e32 v4, 32, v2
	buffer_store_dwordx2 v[134:135], v1, s[16:19], 0 offen
	v_add_lshl_u32 v1, v0, v4, 1
	v_or_b32_e32 v14, 40, v2
	v_cvt_pk_bf16_f32 v160, v160, v161
	v_cvt_pk_bf16_f32 v161, v162, v163
	v_cvt_pk_bf16_f32 v162, v164, v165
	v_cvt_pk_bf16_f32 v163, v166, v167
	v_cvt_pk_bf16_f32 v164, v168, v169
	v_cvt_pk_bf16_f32 v165, v170, v171
	v_cvt_pk_bf16_f32 v166, v172, v173
	v_cvt_pk_bf16_f32 v167, v174, v175
	buffer_store_dwordx2 v[160:161], v1, s[16:19], 0 offen
	v_add_lshl_u32 v1, v0, v14, 1
	v_or_b32_e32 v15, 48, v2
	buffer_store_dwordx2 v[162:163], v1, s[16:19], 0 offen
	v_add_lshl_u32 v1, v0, v15, 1
	v_or_b32_e32 v16, 56, v2
	buffer_store_dwordx2 v[164:165], v1, s[16:19], 0 offen
	v_add_lshl_u32 v1, v0, v16, 1
	v_or_b32_e32 v9, 64, v2
	buffer_store_dwordx2 v[166:167], v1, s[16:19], 0 offen
	v_add_lshl_u32 v1, v0, v9, 1
	v_or_b32_e32 v17, 0x48, v2
	v_cvt_pk_bf16_f32 v192, v192, v193
	v_cvt_pk_bf16_f32 v193, v194, v195
	v_cvt_pk_bf16_f32 v194, v196, v197
	v_cvt_pk_bf16_f32 v195, v198, v199
	v_cvt_pk_bf16_f32 v196, v200, v201
	v_cvt_pk_bf16_f32 v197, v202, v203
	v_cvt_pk_bf16_f32 v198, v204, v205
	v_cvt_pk_bf16_f32 v199, v206, v207
	buffer_store_dwordx2 v[192:193], v1, s[16:19], 0 offen
	v_add_lshl_u32 v1, v0, v17, 1
	v_or_b32_e32 v18, 0x50, v2
	buffer_store_dwordx2 v[194:195], v1, s[16:19], 0 offen
	v_add_lshl_u32 v1, v0, v18, 1
	v_or_b32_e32 v19, 0x58, v2
	buffer_store_dwordx2 v[196:197], v1, s[16:19], 0 offen
	v_add_lshl_u32 v1, v0, v19, 1
	v_or_b32_e32 v13, 0x60, v2
	buffer_store_dwordx2 v[198:199], v1, s[16:19], 0 offen
	v_add_lshl_u32 v1, v0, v13, 1
	v_or_b32_e32 v20, 0x68, v2
	v_cvt_pk_bf16_f32 v224, v224, v225
	v_cvt_pk_bf16_f32 v225, v226, v227
	v_cvt_pk_bf16_f32 v226, v228, v229
	v_cvt_pk_bf16_f32 v227, v230, v231
	v_cvt_pk_bf16_f32 v228, v232, v233
	v_cvt_pk_bf16_f32 v229, v234, v235
	v_cvt_pk_bf16_f32 v230, v236, v237
	v_cvt_pk_bf16_f32 v231, v238, v239
	buffer_store_dwordx2 v[224:225], v1, s[16:19], 0 offen
	v_add_lshl_u32 v1, v0, v20, 1
	v_or_b32_e32 v21, 0x70, v2
	buffer_store_dwordx2 v[226:227], v1, s[16:19], 0 offen
	v_add_lshl_u32 v1, v0, v21, 1
	v_or_b32_e32 v22, 0x78, v2
	buffer_store_dwordx2 v[228:229], v1, s[16:19], 0 offen
	v_add_lshl_u32 v1, v0, v22, 1
	v_lshl_add_u32 v0, s8, 5, v0
	buffer_store_dwordx2 v[230:231], v1, s[16:19], 0 offen
	v_add_lshl_u32 v1, v0, v2, 1
	v_cvt_pk_bf16_f32 v144, v144, v145
	v_cvt_pk_bf16_f32 v145, v146, v147
	v_cvt_pk_bf16_f32 v146, v148, v149
	v_cvt_pk_bf16_f32 v147, v150, v151
	v_cvt_pk_bf16_f32 v148, v152, v153
	v_cvt_pk_bf16_f32 v149, v154, v155
	v_cvt_pk_bf16_f32 v150, v156, v157
	v_cvt_pk_bf16_f32 v151, v158, v159
	buffer_store_dwordx2 v[144:145], v1, s[16:19], 0 offen
	v_add_lshl_u32 v1, v0, v10, 1
	buffer_store_dwordx2 v[146:147], v1, s[16:19], 0 offen
	v_add_lshl_u32 v1, v0, v11, 1
	buffer_store_dwordx2 v[148:149], v1, s[16:19], 0 offen
	v_add_lshl_u32 v1, v0, v12, 1
	buffer_store_dwordx2 v[150:151], v1, s[16:19], 0 offen
	v_add_lshl_u32 v1, v0, v4, 1
	v_cvt_pk_bf16_f32 v176, v176, v177
	v_cvt_pk_bf16_f32 v177, v178, v179
	v_cvt_pk_bf16_f32 v178, v180, v181
	v_cvt_pk_bf16_f32 v179, v182, v183
	v_cvt_pk_bf16_f32 v180, v184, v185
	v_cvt_pk_bf16_f32 v181, v186, v187
	v_cvt_pk_bf16_f32 v182, v188, v189
	v_cvt_pk_bf16_f32 v183, v190, v191
	buffer_store_dwordx2 v[176:177], v1, s[16:19], 0 offen
	v_add_lshl_u32 v1, v0, v14, 1
	buffer_store_dwordx2 v[178:179], v1, s[16:19], 0 offen
	v_add_lshl_u32 v1, v0, v15, 1
	buffer_store_dwordx2 v[180:181], v1, s[16:19], 0 offen
	v_add_lshl_u32 v1, v0, v16, 1
	buffer_store_dwordx2 v[182:183], v1, s[16:19], 0 offen
	v_add_lshl_u32 v1, v0, v9, 1
	v_cvt_pk_bf16_f32 v208, v208, v209
	v_cvt_pk_bf16_f32 v209, v210, v211
	v_cvt_pk_bf16_f32 v210, v212, v213
	v_cvt_pk_bf16_f32 v211, v214, v215
	v_cvt_pk_bf16_f32 v212, v216, v217
	v_cvt_pk_bf16_f32 v213, v218, v219
	v_cvt_pk_bf16_f32 v214, v220, v221
	v_cvt_pk_bf16_f32 v215, v222, v223
	buffer_store_dwordx2 v[208:209], v1, s[16:19], 0 offen
	v_add_lshl_u32 v1, v0, v17, 1
	s_mul_i32 s0, s4, s12
	buffer_store_dwordx2 v[210:211], v1, s[16:19], 0 offen
	v_add_lshl_u32 v1, v0, v18, 1
	s_mul_i32 s0, s0, s14
	buffer_store_dwordx2 v[212:213], v1, s[16:19], 0 offen
	v_add_lshl_u32 v1, v0, v19, 1
	s_add_i32 s0, s0, s2
	buffer_store_dwordx2 v[214:215], v1, s[16:19], 0 offen
	v_cvt_pk_bf16_f32 v240, v240, v241
	v_cvt_pk_bf16_f32 v241, v242, v243
	v_cvt_pk_bf16_f32 v242, v244, v245
	v_cvt_pk_bf16_f32 v243, v246, v247
	v_cvt_pk_bf16_f32 v244, v248, v249
	v_cvt_pk_bf16_f32 v245, v250, v251
	v_cvt_pk_bf16_f32 v246, v252, v253
	v_cvt_pk_bf16_f32 v247, v254, v255
	v_add_lshl_u32 v1, v0, v13, 1
	buffer_store_dwordx2 v[240:241], v1, s[16:19], 0 offen
	s_mul_i32 s0, s0, s24
	v_add_lshl_u32 v1, v0, v20, 1
	buffer_store_dwordx2 v[242:243], v1, s[16:19], 0 offen
	s_ashr_i32 s1, s0, 31
	v_add_lshl_u32 v1, v0, v21, 1
	buffer_store_dwordx2 v[244:245], v1, s[16:19], 0 offen
	v_add_lshl_u32 v0, v0, v22, 1
	s_lshl_b64 s[0:1], s[0:1], 1
	s_mul_i32 s2, s14, s24
	buffer_store_dwordx2 v[246:247], v0, s[16:19], 0 offen
	s_add_u32 s16, s20, s0
	v_mul_lo_u32 v0, s2, v3
	s_addc_u32 s17, s21, s1
	v_mad_u64_u32 v[0:1], s[0:1], v23, s2, v[0:1]
	v_add_lshl_u32 v1, v0, v2, 1
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	s_barrier
	v_accvgpr_read_b32 v128, a128
	v_accvgpr_read_b32 v129, a129
	v_accvgpr_read_b32 v130, a130
	v_accvgpr_read_b32 v131, a131
	v_accvgpr_read_b32 v132, a132
	v_accvgpr_read_b32 v133, a133
	v_accvgpr_read_b32 v134, a134
	v_accvgpr_read_b32 v135, a135
	v_accvgpr_read_b32 v136, a136
	v_accvgpr_read_b32 v137, a137
	v_accvgpr_read_b32 v138, a138
	v_accvgpr_read_b32 v139, a139
	v_accvgpr_read_b32 v140, a140
	v_accvgpr_read_b32 v141, a141
	v_accvgpr_read_b32 v142, a142
	v_accvgpr_read_b32 v143, a143
	v_accvgpr_read_b32 v144, a144
	v_accvgpr_read_b32 v145, a145
	v_accvgpr_read_b32 v146, a146
	v_accvgpr_read_b32 v147, a147
	v_accvgpr_read_b32 v148, a148
	v_accvgpr_read_b32 v149, a149
	v_accvgpr_read_b32 v150, a150
	v_accvgpr_read_b32 v151, a151
	v_accvgpr_read_b32 v152, a152
	v_accvgpr_read_b32 v153, a153
	v_accvgpr_read_b32 v154, a154
	v_accvgpr_read_b32 v155, a155
	v_accvgpr_read_b32 v156, a156
	v_accvgpr_read_b32 v157, a157
	v_accvgpr_read_b32 v158, a158
	v_accvgpr_read_b32 v159, a159
	v_accvgpr_read_b32 v160, a160
	v_accvgpr_read_b32 v161, a161
	v_accvgpr_read_b32 v162, a162
	v_accvgpr_read_b32 v163, a163
	v_accvgpr_read_b32 v164, a164
	v_accvgpr_read_b32 v165, a165
	v_accvgpr_read_b32 v166, a166
	v_accvgpr_read_b32 v167, a167
	v_accvgpr_read_b32 v168, a168
	v_accvgpr_read_b32 v169, a169
	v_accvgpr_read_b32 v170, a170
	v_accvgpr_read_b32 v171, a171
	v_accvgpr_read_b32 v172, a172
	v_accvgpr_read_b32 v173, a173
	v_accvgpr_read_b32 v174, a174
	v_accvgpr_read_b32 v175, a175
	v_accvgpr_read_b32 v176, a176
	v_accvgpr_read_b32 v177, a177
	v_accvgpr_read_b32 v178, a178
	v_accvgpr_read_b32 v179, a179
	v_accvgpr_read_b32 v180, a180
	v_accvgpr_read_b32 v181, a181
	v_accvgpr_read_b32 v182, a182
	v_accvgpr_read_b32 v183, a183
	v_accvgpr_read_b32 v184, a184
	v_accvgpr_read_b32 v185, a185
	v_accvgpr_read_b32 v186, a186
	v_accvgpr_read_b32 v187, a187
	v_accvgpr_read_b32 v188, a188
	v_accvgpr_read_b32 v189, a189
	v_accvgpr_read_b32 v190, a190
	v_accvgpr_read_b32 v191, a191
	v_accvgpr_read_b32 v192, a192
	v_accvgpr_read_b32 v193, a193
	v_accvgpr_read_b32 v194, a194
	v_accvgpr_read_b32 v195, a195
	v_accvgpr_read_b32 v196, a196
	v_accvgpr_read_b32 v197, a197
	v_accvgpr_read_b32 v198, a198
	v_accvgpr_read_b32 v199, a199
	v_accvgpr_read_b32 v200, a200
	v_accvgpr_read_b32 v201, a201
	v_accvgpr_read_b32 v202, a202
	v_accvgpr_read_b32 v203, a203
	v_accvgpr_read_b32 v204, a204
	v_accvgpr_read_b32 v205, a205
	v_accvgpr_read_b32 v206, a206
	v_accvgpr_read_b32 v207, a207
	v_accvgpr_read_b32 v208, a208
	v_accvgpr_read_b32 v209, a209
	v_accvgpr_read_b32 v210, a210
	v_accvgpr_read_b32 v211, a211
	v_accvgpr_read_b32 v212, a212
	v_accvgpr_read_b32 v213, a213
	v_accvgpr_read_b32 v214, a214
	v_accvgpr_read_b32 v215, a215
	v_accvgpr_read_b32 v216, a216
	v_accvgpr_read_b32 v217, a217
	v_accvgpr_read_b32 v218, a218
	v_accvgpr_read_b32 v219, a219
	v_accvgpr_read_b32 v220, a220
	v_accvgpr_read_b32 v221, a221
	v_accvgpr_read_b32 v222, a222
	v_accvgpr_read_b32 v223, a223
	v_accvgpr_read_b32 v224, a224
	v_accvgpr_read_b32 v225, a225
	v_accvgpr_read_b32 v226, a226
	v_accvgpr_read_b32 v227, a227
	v_accvgpr_read_b32 v228, a228
	v_accvgpr_read_b32 v229, a229
	v_accvgpr_read_b32 v230, a230
	v_accvgpr_read_b32 v231, a231
	v_accvgpr_read_b32 v232, a232
	v_accvgpr_read_b32 v233, a233
	v_accvgpr_read_b32 v234, a234
	v_accvgpr_read_b32 v235, a235
	v_accvgpr_read_b32 v236, a236
	v_accvgpr_read_b32 v237, a237
	v_accvgpr_read_b32 v238, a238
	v_accvgpr_read_b32 v239, a239
	v_accvgpr_read_b32 v240, a240
	v_accvgpr_read_b32 v241, a241
	v_accvgpr_read_b32 v242, a242
	v_accvgpr_read_b32 v243, a243
	v_accvgpr_read_b32 v244, a244
	v_accvgpr_read_b32 v245, a245
	v_accvgpr_read_b32 v246, a246
	v_accvgpr_read_b32 v247, a247
	v_accvgpr_read_b32 v248, a248
	v_accvgpr_read_b32 v249, a249
	v_accvgpr_read_b32 v250, a250
	v_accvgpr_read_b32 v251, a251
	v_accvgpr_read_b32 v252, a252
	v_accvgpr_read_b32 v253, a253
	v_accvgpr_read_b32 v254, a254
	v_accvgpr_read_b32 v255, a255
	v_mul_f32_e32 v128, 0x3db504f3, v128
	v_mul_f32_e32 v129, 0x3db504f3, v129
	v_mul_f32_e32 v130, 0x3db504f3, v130
	v_mul_f32_e32 v131, 0x3db504f3, v131
	v_mul_f32_e32 v132, 0x3db504f3, v132
	v_mul_f32_e32 v133, 0x3db504f3, v133
	v_mul_f32_e32 v134, 0x3db504f3, v134
	v_mul_f32_e32 v135, 0x3db504f3, v135
	v_mul_f32_e32 v136, 0x3db504f3, v136
	v_mul_f32_e32 v137, 0x3db504f3, v137
	v_mul_f32_e32 v138, 0x3db504f3, v138
	v_mul_f32_e32 v139, 0x3db504f3, v139
	v_mul_f32_e32 v140, 0x3db504f3, v140
	v_mul_f32_e32 v141, 0x3db504f3, v141
	v_mul_f32_e32 v142, 0x3db504f3, v142
	v_mul_f32_e32 v143, 0x3db504f3, v143
	v_mul_f32_e32 v144, 0x3db504f3, v144
	v_mul_f32_e32 v145, 0x3db504f3, v145
	v_mul_f32_e32 v146, 0x3db504f3, v146
	v_mul_f32_e32 v147, 0x3db504f3, v147
	v_mul_f32_e32 v148, 0x3db504f3, v148
	v_mul_f32_e32 v149, 0x3db504f3, v149
	v_mul_f32_e32 v150, 0x3db504f3, v150
	v_mul_f32_e32 v151, 0x3db504f3, v151
	v_mul_f32_e32 v152, 0x3db504f3, v152
	v_mul_f32_e32 v153, 0x3db504f3, v153
	v_mul_f32_e32 v154, 0x3db504f3, v154
	v_mul_f32_e32 v155, 0x3db504f3, v155
	v_mul_f32_e32 v156, 0x3db504f3, v156
	v_mul_f32_e32 v157, 0x3db504f3, v157
	v_mul_f32_e32 v158, 0x3db504f3, v158
	v_mul_f32_e32 v159, 0x3db504f3, v159
	v_mul_f32_e32 v160, 0x3db504f3, v160
	v_mul_f32_e32 v161, 0x3db504f3, v161
	v_mul_f32_e32 v162, 0x3db504f3, v162
	v_mul_f32_e32 v163, 0x3db504f3, v163
	v_mul_f32_e32 v164, 0x3db504f3, v164
	v_mul_f32_e32 v165, 0x3db504f3, v165
	v_mul_f32_e32 v166, 0x3db504f3, v166
	v_mul_f32_e32 v167, 0x3db504f3, v167
	v_mul_f32_e32 v168, 0x3db504f3, v168
	v_mul_f32_e32 v169, 0x3db504f3, v169
	v_mul_f32_e32 v170, 0x3db504f3, v170
	v_mul_f32_e32 v171, 0x3db504f3, v171
	v_mul_f32_e32 v172, 0x3db504f3, v172
	v_mul_f32_e32 v173, 0x3db504f3, v173
	v_mul_f32_e32 v174, 0x3db504f3, v174
	v_mul_f32_e32 v175, 0x3db504f3, v175
	v_mul_f32_e32 v176, 0x3db504f3, v176
	v_mul_f32_e32 v177, 0x3db504f3, v177
	v_mul_f32_e32 v178, 0x3db504f3, v178
	v_mul_f32_e32 v179, 0x3db504f3, v179
	v_mul_f32_e32 v180, 0x3db504f3, v180
	v_mul_f32_e32 v181, 0x3db504f3, v181
	v_mul_f32_e32 v182, 0x3db504f3, v182
	v_mul_f32_e32 v183, 0x3db504f3, v183
	v_mul_f32_e32 v184, 0x3db504f3, v184
	v_mul_f32_e32 v185, 0x3db504f3, v185
	v_mul_f32_e32 v186, 0x3db504f3, v186
	v_mul_f32_e32 v187, 0x3db504f3, v187
	v_mul_f32_e32 v188, 0x3db504f3, v188
	v_mul_f32_e32 v189, 0x3db504f3, v189
	v_mul_f32_e32 v190, 0x3db504f3, v190
	v_mul_f32_e32 v191, 0x3db504f3, v191
	v_mul_f32_e32 v192, 0x3db504f3, v192
	v_mul_f32_e32 v193, 0x3db504f3, v193
	v_mul_f32_e32 v194, 0x3db504f3, v194
	v_mul_f32_e32 v195, 0x3db504f3, v195
	v_mul_f32_e32 v196, 0x3db504f3, v196
	v_mul_f32_e32 v197, 0x3db504f3, v197
	v_mul_f32_e32 v198, 0x3db504f3, v198
	v_mul_f32_e32 v199, 0x3db504f3, v199
	v_mul_f32_e32 v200, 0x3db504f3, v200
	v_mul_f32_e32 v201, 0x3db504f3, v201
	v_mul_f32_e32 v202, 0x3db504f3, v202
	v_mul_f32_e32 v203, 0x3db504f3, v203
	v_mul_f32_e32 v204, 0x3db504f3, v204
	v_mul_f32_e32 v205, 0x3db504f3, v205
	v_mul_f32_e32 v206, 0x3db504f3, v206
	v_mul_f32_e32 v207, 0x3db504f3, v207
	v_mul_f32_e32 v208, 0x3db504f3, v208
	v_mul_f32_e32 v209, 0x3db504f3, v209
	v_mul_f32_e32 v210, 0x3db504f3, v210
	v_mul_f32_e32 v211, 0x3db504f3, v211
	v_mul_f32_e32 v212, 0x3db504f3, v212
	v_mul_f32_e32 v213, 0x3db504f3, v213
	v_mul_f32_e32 v214, 0x3db504f3, v214
	v_mul_f32_e32 v215, 0x3db504f3, v215
	v_mul_f32_e32 v216, 0x3db504f3, v216
	v_mul_f32_e32 v217, 0x3db504f3, v217
	v_mul_f32_e32 v218, 0x3db504f3, v218
	v_mul_f32_e32 v219, 0x3db504f3, v219
	v_mul_f32_e32 v220, 0x3db504f3, v220
	v_mul_f32_e32 v221, 0x3db504f3, v221
	v_mul_f32_e32 v222, 0x3db504f3, v222
	v_mul_f32_e32 v223, 0x3db504f3, v223
	v_mul_f32_e32 v224, 0x3db504f3, v224
	v_mul_f32_e32 v225, 0x3db504f3, v225
	v_mul_f32_e32 v226, 0x3db504f3, v226
	v_mul_f32_e32 v227, 0x3db504f3, v227
	v_mul_f32_e32 v228, 0x3db504f3, v228
	v_mul_f32_e32 v229, 0x3db504f3, v229
	v_mul_f32_e32 v230, 0x3db504f3, v230
	v_mul_f32_e32 v231, 0x3db504f3, v231
	v_mul_f32_e32 v232, 0x3db504f3, v232
	v_mul_f32_e32 v233, 0x3db504f3, v233
	v_mul_f32_e32 v234, 0x3db504f3, v234
	v_mul_f32_e32 v235, 0x3db504f3, v235
	v_mul_f32_e32 v236, 0x3db504f3, v236
	v_mul_f32_e32 v237, 0x3db504f3, v237
	v_mul_f32_e32 v238, 0x3db504f3, v238
	v_mul_f32_e32 v239, 0x3db504f3, v239
	v_mul_f32_e32 v240, 0x3db504f3, v240
	v_mul_f32_e32 v241, 0x3db504f3, v241
	v_mul_f32_e32 v242, 0x3db504f3, v242
	v_mul_f32_e32 v243, 0x3db504f3, v243
	v_mul_f32_e32 v244, 0x3db504f3, v244
	v_mul_f32_e32 v245, 0x3db504f3, v245
	v_mul_f32_e32 v246, 0x3db504f3, v246
	v_mul_f32_e32 v247, 0x3db504f3, v247
	v_mul_f32_e32 v248, 0x3db504f3, v248
	v_mul_f32_e32 v249, 0x3db504f3, v249
	v_mul_f32_e32 v250, 0x3db504f3, v250
	v_mul_f32_e32 v251, 0x3db504f3, v251
	v_mul_f32_e32 v252, 0x3db504f3, v252
	v_mul_f32_e32 v253, 0x3db504f3, v253
	v_mul_f32_e32 v254, 0x3db504f3, v254
	v_mul_f32_e32 v255, 0x3db504f3, v255
	v_cvt_pk_bf16_f32 v128, v128, v129
	v_cvt_pk_bf16_f32 v129, v130, v131
	v_cvt_pk_bf16_f32 v130, v132, v133
	v_cvt_pk_bf16_f32 v131, v134, v135
	v_cvt_pk_bf16_f32 v132, v136, v137
	v_cvt_pk_bf16_f32 v133, v138, v139
	v_cvt_pk_bf16_f32 v134, v140, v141
	v_cvt_pk_bf16_f32 v135, v142, v143
	buffer_store_dwordx2 v[128:129], v1, s[16:19], 0 offen
	v_add_lshl_u32 v1, v0, v10, 1
	buffer_store_dwordx2 v[130:131], v1, s[16:19], 0 offen
	v_add_lshl_u32 v1, v0, v11, 1
	buffer_store_dwordx2 v[132:133], v1, s[16:19], 0 offen
	v_add_lshl_u32 v1, v0, v12, 1
	buffer_store_dwordx2 v[134:135], v1, s[16:19], 0 offen
	v_add_lshl_u32 v1, v0, v4, 1
	v_cvt_pk_bf16_f32 v160, v160, v161
	v_cvt_pk_bf16_f32 v161, v162, v163
	v_cvt_pk_bf16_f32 v162, v164, v165
	v_cvt_pk_bf16_f32 v163, v166, v167
	v_cvt_pk_bf16_f32 v164, v168, v169
	v_cvt_pk_bf16_f32 v165, v170, v171
	v_cvt_pk_bf16_f32 v166, v172, v173
	v_cvt_pk_bf16_f32 v167, v174, v175
	buffer_store_dwordx2 v[160:161], v1, s[16:19], 0 offen
	v_add_lshl_u32 v1, v0, v14, 1
	buffer_store_dwordx2 v[162:163], v1, s[16:19], 0 offen
	v_add_lshl_u32 v1, v0, v15, 1
	buffer_store_dwordx2 v[164:165], v1, s[16:19], 0 offen
	v_add_lshl_u32 v1, v0, v16, 1
	buffer_store_dwordx2 v[166:167], v1, s[16:19], 0 offen
	v_add_lshl_u32 v1, v0, v9, 1
	v_cvt_pk_bf16_f32 v192, v192, v193
	v_cvt_pk_bf16_f32 v193, v194, v195
	v_cvt_pk_bf16_f32 v194, v196, v197
	v_cvt_pk_bf16_f32 v195, v198, v199
	v_cvt_pk_bf16_f32 v196, v200, v201
	v_cvt_pk_bf16_f32 v197, v202, v203
	v_cvt_pk_bf16_f32 v198, v204, v205
	v_cvt_pk_bf16_f32 v199, v206, v207
	buffer_store_dwordx2 v[192:193], v1, s[16:19], 0 offen
	v_add_lshl_u32 v1, v0, v17, 1
	buffer_store_dwordx2 v[194:195], v1, s[16:19], 0 offen
	v_add_lshl_u32 v1, v0, v18, 1
	buffer_store_dwordx2 v[196:197], v1, s[16:19], 0 offen
	v_add_lshl_u32 v1, v0, v19, 1
	buffer_store_dwordx2 v[198:199], v1, s[16:19], 0 offen
	v_add_lshl_u32 v1, v0, v13, 1
	v_cvt_pk_bf16_f32 v224, v224, v225
	v_cvt_pk_bf16_f32 v225, v226, v227
	v_cvt_pk_bf16_f32 v226, v228, v229
	v_cvt_pk_bf16_f32 v227, v230, v231
	v_cvt_pk_bf16_f32 v228, v232, v233
	v_cvt_pk_bf16_f32 v229, v234, v235
	v_cvt_pk_bf16_f32 v230, v236, v237
	v_cvt_pk_bf16_f32 v231, v238, v239
	buffer_store_dwordx2 v[224:225], v1, s[16:19], 0 offen
	v_add_lshl_u32 v1, v0, v20, 1
	buffer_store_dwordx2 v[226:227], v1, s[16:19], 0 offen
	v_add_lshl_u32 v1, v0, v21, 1
	buffer_store_dwordx2 v[228:229], v1, s[16:19], 0 offen
	v_add_lshl_u32 v1, v0, v22, 1
	v_lshl_add_u32 v0, s2, 5, v0
	buffer_store_dwordx2 v[230:231], v1, s[16:19], 0 offen
	v_add_lshl_u32 v1, v0, v2, 1
	v_cvt_pk_bf16_f32 v144, v144, v145
	v_cvt_pk_bf16_f32 v145, v146, v147
	v_cvt_pk_bf16_f32 v146, v148, v149
	v_cvt_pk_bf16_f32 v147, v150, v151
	v_cvt_pk_bf16_f32 v148, v152, v153
	v_cvt_pk_bf16_f32 v149, v154, v155
	v_cvt_pk_bf16_f32 v150, v156, v157
	v_cvt_pk_bf16_f32 v151, v158, v159
	buffer_store_dwordx2 v[144:145], v1, s[16:19], 0 offen
	v_add_lshl_u32 v1, v0, v10, 1
	buffer_store_dwordx2 v[146:147], v1, s[16:19], 0 offen
	v_add_lshl_u32 v1, v0, v11, 1
	buffer_store_dwordx2 v[148:149], v1, s[16:19], 0 offen
	v_add_lshl_u32 v1, v0, v12, 1
	buffer_store_dwordx2 v[150:151], v1, s[16:19], 0 offen
	v_add_lshl_u32 v1, v0, v4, 1
	v_cvt_pk_bf16_f32 v176, v176, v177
	v_cvt_pk_bf16_f32 v177, v178, v179
	v_cvt_pk_bf16_f32 v178, v180, v181
	v_cvt_pk_bf16_f32 v179, v182, v183
	v_cvt_pk_bf16_f32 v180, v184, v185
	v_cvt_pk_bf16_f32 v181, v186, v187
	v_cvt_pk_bf16_f32 v182, v188, v189
	v_cvt_pk_bf16_f32 v183, v190, v191
	buffer_store_dwordx2 v[176:177], v1, s[16:19], 0 offen
	v_add_lshl_u32 v1, v0, v14, 1
	buffer_store_dwordx2 v[178:179], v1, s[16:19], 0 offen
	v_add_lshl_u32 v1, v0, v15, 1
	buffer_store_dwordx2 v[180:181], v1, s[16:19], 0 offen
	v_add_lshl_u32 v1, v0, v16, 1
	buffer_store_dwordx2 v[182:183], v1, s[16:19], 0 offen
	v_add_lshl_u32 v1, v0, v9, 1
	v_cvt_pk_bf16_f32 v208, v208, v209
	v_cvt_pk_bf16_f32 v209, v210, v211
	v_cvt_pk_bf16_f32 v210, v212, v213
	v_cvt_pk_bf16_f32 v211, v214, v215
	v_cvt_pk_bf16_f32 v212, v216, v217
	v_cvt_pk_bf16_f32 v213, v218, v219
	v_cvt_pk_bf16_f32 v214, v220, v221
	v_cvt_pk_bf16_f32 v215, v222, v223
	buffer_store_dwordx2 v[208:209], v1, s[16:19], 0 offen
	v_add_lshl_u32 v1, v0, v17, 1
	buffer_store_dwordx2 v[210:211], v1, s[16:19], 0 offen
	v_add_lshl_u32 v1, v0, v18, 1
	buffer_store_dwordx2 v[212:213], v1, s[16:19], 0 offen
	v_add_lshl_u32 v1, v0, v19, 1
	s_add_i32 s0, s5, 0x1ff0
	buffer_store_dwordx2 v[214:215], v1, s[16:19], 0 offen
	v_cvt_pk_bf16_f32 v240, v240, v241
	v_cvt_pk_bf16_f32 v241, v242, v243
	v_cvt_pk_bf16_f32 v242, v244, v245
	v_cvt_pk_bf16_f32 v243, v246, v247
	v_cvt_pk_bf16_f32 v244, v248, v249
	v_cvt_pk_bf16_f32 v245, v250, v251
	v_cvt_pk_bf16_f32 v246, v252, v253
	v_cvt_pk_bf16_f32 v247, v254, v255
	v_add_lshl_u32 v1, v0, v13, 1
	buffer_store_dwordx2 v[240:241], v1, s[16:19], 0 offen
	s_mul_i32 s0, s0, s3
	v_add_lshl_u32 v1, v0, v20, 1
	buffer_store_dwordx2 v[242:243], v1, s[16:19], 0 offen
	s_ashr_i32 s1, s0, 31
	v_add_lshl_u32 v1, v0, v21, 1
	buffer_store_dwordx2 v[244:245], v1, s[16:19], 0 offen
	s_lshl_b64 s[0:1], s[0:1], 1
	v_add_lshl_u32 v0, v0, v22, 1
	buffer_store_dwordx2 v[246:247], v0, s[16:19], 0 offen
	s_add_u32 s8, s6, s0
	v_mul_f32_e32 v110, 0x3db504f3, v110
	v_mul_f32_e32 v111, 0x3db504f3, v111
	v_mul_f32_e32 v112, 0x3db504f3, v112
	v_mul_f32_e32 v113, 0x3db504f3, v113
	v_mul_f32_e32 v114, 0x3db504f3, v114
	v_mul_f32_e32 v115, 0x3db504f3, v115
	v_mul_f32_e32 v116, 0x3db504f3, v116
	v_mul_f32_e32 v117, 0x3db504f3, v117
	s_addc_u32 s9, s7, s1
	v_cvt_pk_bf16_f32 v110, v110, v111
	v_cvt_pk_bf16_f32 v111, v112, v113
	buffer_atomic_pk_add_bf16 v110, v5, s[8:11], 0 offen
	buffer_atomic_pk_add_bf16 v111, v6, s[8:11], 0 offen
	v_cvt_pk_bf16_f32 v114, v114, v115
	v_cvt_pk_bf16_f32 v115, v116, v117
	buffer_atomic_pk_add_bf16 v114, v7, s[8:11], 0 offen
	buffer_atomic_pk_add_bf16 v115, v8, s[8:11], 0 offen
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z23attend_bwd_combined_kerILi128EEv25attn_bwd_combined_globalsIXT_EE
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 440
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
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 1
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 512
		.amdhsa_next_free_sgpr 79
		.amdhsa_accum_offset 256
		.amdhsa_reserve_vcc 0
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

	.section	.text._Z23attend_bwd_combined_kerILi128EEv25attn_bwd_combined_globalsIXT_EE,"axG",@progbits,_Z23attend_bwd_combined_kerILi128EEv25attn_bwd_combined_globalsIXT_EE,comdat
.Lfunc_end0:
	.size	_Z23attend_bwd_combined_kerILi128EEv25attn_bwd_combined_globalsIXT_EE, .Lfunc_end0-_Z23attend_bwd_combined_kerILi128EEv25attn_bwd_combined_globalsIXT_EE
.set _Z23attend_bwd_combined_kerILi128EEv25attn_bwd_combined_globalsIXT_EE.num_vgpr, 256
	.no_dead_strip	_Z23attend_bwd_combined_kerILi128EEv25attn_bwd_combined_globalsIXT_EE.num_vgpr
.set _Z23attend_bwd_combined_kerILi128EEv25attn_bwd_combined_globalsIXT_EE.num_agpr, 256
	.no_dead_strip	_Z23attend_bwd_combined_kerILi128EEv25attn_bwd_combined_globalsIXT_EE.num_agpr
.set _Z23attend_bwd_combined_kerILi128EEv25attn_bwd_combined_globalsIXT_EE.numbered_sgpr, 79
	.no_dead_strip	_Z23attend_bwd_combined_kerILi128EEv25attn_bwd_combined_globalsIXT_EE.numbered_sgpr
.set _Z23attend_bwd_combined_kerILi128EEv25attn_bwd_combined_globalsIXT_EE.private_seg_size, 0
	.no_dead_strip	_Z23attend_bwd_combined_kerILi128EEv25attn_bwd_combined_globalsIXT_EE.private_seg_size
.set _Z23attend_bwd_combined_kerILi128EEv25attn_bwd_combined_globalsIXT_EE.uses_vcc, 0
	.no_dead_strip	_Z23attend_bwd_combined_kerILi128EEv25attn_bwd_combined_globalsIXT_EE.uses_vcc
.set _Z23attend_bwd_combined_kerILi128EEv25attn_bwd_combined_globalsIXT_EE.uses_flat_scratch, 0
	.no_dead_strip	_Z23attend_bwd_combined_kerILi128EEv25attn_bwd_combined_globalsIXT_EE.uses_flat_scratch
.set _Z23attend_bwd_combined_kerILi128EEv25attn_bwd_combined_globalsIXT_EE.has_dyn_sized_stack, 0
	.no_dead_strip	_Z23attend_bwd_combined_kerILi128EEv25attn_bwd_combined_globalsIXT_EE.has_dyn_sized_stack
.set _Z23attend_bwd_combined_kerILi128EEv25attn_bwd_combined_globalsIXT_EE.has_recursion, 0
	.no_dead_strip	_Z23attend_bwd_combined_kerILi128EEv25attn_bwd_combined_globalsIXT_EE.has_recursion
.set _Z23attend_bwd_combined_kerILi128EEv25attn_bwd_combined_globalsIXT_EE.has_indirect_call, 0
	.no_dead_strip	_Z23attend_bwd_combined_kerILi128EEv25attn_bwd_combined_globalsIXT_EE.has_indirect_call
	.section	.AMDGPU.csdata,"",@progbits
	.text
.p2alignl 6, 0xbf800000
	.fill	256, 4, 0xbf800000
	.section	.AMDGPU.gpr_maximums,"",@progbits
.set amdgpu.max_num_vgpr, 0
	.no_dead_strip	amdgpu.max_num_vgpr
.set amdgpu.max_num_agpr, 0
	.no_dead_strip	amdgpu.max_num_agpr
.set amdgpu.max_num_sgpr, 0
	.no_dead_strip	amdgpu.max_num_sgpr
	.text
	.type	__hip_cuid_5e2152b1227460d,@object
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_5e2152b1227460d
__hip_cuid_5e2152b1227460d:
	.byte	0
	.size	__hip_cuid_5e2152b1227460d, 1

	.ident	"AMD clang version 20.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.0.0 25304 82aed4e69d70bef3c89c38a2ee85c8c41294dfc9)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __shm
	.addrsig_sym __hip_cuid_5e2152b1227460d
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     256
    .args:
      - .offset:         0
        .size:           440
        .value_kind:     by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 440
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _Z23attend_bwd_combined_kerILi128EEv25attn_bwd_combined_globalsIXT_EE
    .private_segment_fixed_size: 0
    .sgpr_count:     85
    .sgpr_spill_count: 0
    .symbol:         _Z23attend_bwd_combined_kerILi128EEv25attn_bwd_combined_globalsIXT_EE.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     512
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata

