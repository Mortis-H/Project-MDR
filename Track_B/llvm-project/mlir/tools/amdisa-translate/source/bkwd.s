	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.section	.text._Z23attend_bwd_combined_kerILi128EEv25attn_bwd_combined_globalsIXT_EE,"axG",@progbits,_Z23attend_bwd_combined_kerILi128EEv25attn_bwd_combined_globalsIXT_EE,comdat
	.protected	_Z23attend_bwd_combined_kerILi128EEv25attn_bwd_combined_globalsIXT_EE ; -- Begin function _Z23attend_bwd_combined_kerILi128EEv25attn_bwd_combined_globalsIXT_EE
	.globl	_Z23attend_bwd_combined_kerILi128EEv25attn_bwd_combined_globalsIXT_EE
	.p2align	8
	.type	_Z23attend_bwd_combined_kerILi128EEv25attn_bwd_combined_globalsIXT_EE,@function
_Z23attend_bwd_combined_kerILi128EEv25attn_bwd_combined_globalsIXT_EE: ; @_Z23attend_bwd_combined_kerILi128EEv25attn_bwd_combined_globalsIXT_EE
; %bb.0:
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
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s6
	s_lshl_b32 s6, s15, 4
	v_add_u32_e32 v7, 0x1000, v3
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_lshl_b32 s30, s15, 9
	s_mov_b32 s31, 0x110000
	v_lshlrev_b32_e32 v5, 1, v4
	v_add_u32_e32 v4, s6, v4
	v_readfirstlane_b32 s7, v7
	v_add_u32_e32 v7, 0x2000, v3
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
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
	;;#ASMSTART
	buffer_load_dwordx4 a[48:51], v7, s[8:11], 0 offen offset:0
	;;#ASMEND
	v_or_b32_e32 v7, 32, v2
	v_add_lshl_u32 v8, v5, v7, 1
	;;#ASMSTART
	buffer_load_dwordx4 a[52:55], v8, s[8:11], 0 offen offset:0
	;;#ASMEND
	v_add_u32_e32 v8, 64, v2
	v_add_lshl_u32 v9, v5, v8, 1
	;;#ASMSTART
	buffer_load_dwordx4 a[56:59], v9, s[8:11], 0 offen offset:0
	;;#ASMEND
	v_add_u32_e32 v9, 0x60, v2
	s_lshl_b32 s3, s6, 4
	v_add_lshl_u32 v10, v5, v9, 1
	v_add_u32_e32 v5, s3, v5
	;;#ASMSTART
	buffer_load_dwordx4 a[60:63], v10, s[8:11], 0 offen offset:0
	;;#ASMEND
	v_add_lshl_u32 v10, v5, v2, 1
	;;#ASMSTART
	buffer_load_dwordx4 a[64:0x43], v10, s[8:11], 0 offen offset:0
	;;#ASMEND
	v_add_lshl_u32 v10, v5, v7, 1
	;;#ASMSTART
	buffer_load_dwordx4 a[0x44:0x47], v10, s[8:11], 0 offen offset:0
	;;#ASMEND
	v_add_lshl_u32 v10, v5, v8, 1
	;;#ASMSTART
	buffer_load_dwordx4 a[0x48:0x4b], v10, s[8:11], 0 offen offset:0
	;;#ASMEND
	v_add_lshl_u32 v10, v5, v9, 1
	v_add_u32_e32 v5, s3, v5
	;;#ASMSTART
	buffer_load_dwordx4 a[0x4c:0x4f], v10, s[8:11], 0 offen offset:0
	;;#ASMEND
	v_add_lshl_u32 v10, v5, v2, 1
	;;#ASMSTART
	buffer_load_dwordx4 a[0x50:0x53], v10, s[8:11], 0 offen offset:0
	;;#ASMEND
	v_add_lshl_u32 v10, v5, v7, 1
	;;#ASMSTART
	buffer_load_dwordx4 a[0x54:0x57], v10, s[8:11], 0 offen offset:0
	;;#ASMEND
	v_add_lshl_u32 v10, v5, v8, 1
	s_mul_i32 s58, s4, s36
	;;#ASMSTART
	buffer_load_dwordx4 a[0x58:0x5b], v10, s[8:11], 0 offen offset:0
	;;#ASMEND
	v_add_lshl_u32 v10, v5, v9, 1
	v_add_u32_e32 v5, s3, v5
	s_add_i32 s3, s58, s61
	s_mul_i32 s63, s63, s38
	s_mul_i32 s6, s63, s3
	s_ashr_i32 s7, s6, 31
	s_lshl_b64 s[6:7], s[6:7], 2
	s_add_u32 s40, s54, s6
	s_mul_i32 s64, s4, s16
	;;#ASMSTART
	buffer_load_dwordx4 a[0x5c:0x5f], v10, s[8:11], 0 offen offset:0
	;;#ASMEND
	s_addc_u32 s41, s55, s7
	s_add_i32 s3, s64, s61
	s_mul_i32 s65, s65, s18
	v_add_lshl_u32 v2, v5, v2, 1
	;;#ASMSTART
	buffer_load_dwordx4 a[0x60:0x63], v2, s[8:11], 0 offen offset:0
	;;#ASMEND
	s_mul_i32 s6, s65, s3
	v_add_lshl_u32 v2, v5, v7, 1
	;;#ASMSTART
	buffer_load_dwordx4 a[0x64:0x67], v2, s[8:11], 0 offen offset:0
	;;#ASMEND
	s_ashr_i32 s7, s6, 31
	v_add_lshl_u32 v2, v5, v8, 1
	;;#ASMSTART
	buffer_load_dwordx4 a[0x68:0x6b], v2, s[8:11], 0 offen offset:0
	;;#ASMEND
	s_lshl_b64 s[6:7], s[6:7], 2
	s_mul_i32 s66, s4, s20
	v_add_lshl_u32 v2, v5, v9, 1
	;;#ASMSTART
	buffer_load_dwordx4 a[0x6c:0x6f], v2, s[8:11], 0 offen offset:0
	;;#ASMEND
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
	; sched_barrier mask(0x00000000)
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
	;;#ASMSTART
	ds_read_b128 a[0x70:0x73], v18 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[0x74:0x77], v18 offset:0x400
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[0x78:0x7b], v18 offset:0x800
	;;#ASMEND
	v_and_b32_e32 v7, 12, v7
	;;#ASMSTART
	ds_read_b128 a[0x7c:0x7f], v18 offset:0xc00
	;;#ASMEND
	v_or_b32_e32 v13, v7, v13
	v_add_u32_e32 v18, s5, v13
	;;#ASMSTART
	ds_read_b32 v[0x7e], v18 offset:0

	;;#ASMEND
	v_add_u32_e32 v18, s23, v13
	;;#ASMSTART
	ds_read_b32 v[0x7f], v18 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[0:3], v10 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[4:7], v10 offset:0x400
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[8:11], v10 offset:0x800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[12:15], v10 offset:0xc00
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[16:19], v10 offset:0x1000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[20:23], v10 offset:0x1400
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[24:27], v10 offset:0x1800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[28:31], v10 offset:0x1c00
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[46:49], a[0x70:0x73], a[0:3], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[32:35], v10 offset:0x2000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[36:39], v10 offset:0x2400
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[46:49], a[0x74:0x77], a[4:7], v[46:49]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x7e], 0x3fb8aa3b, v[0x7e]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[46:49], a[0x78:0x7b], a[8:11], v[46:49]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[40:43], v10 offset:0x2800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[44:47], v10 offset:0x2c00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[46:49], a[0x7c:0x7f], a[12:15], v[46:49]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[50:53], a[0x70:0x73], a[16:19], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[62:0x41], v10 offset:0x3000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[0x42:0x45], v10 offset:0x3400
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[50:53], a[0x74:0x77], a[20:23], v[50:53]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[50:53], a[0x78:0x7b], a[24:27], v[50:53]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[0x46:0x49], v10 offset:0x3800
	;;#ASMEND
	v_or_b32_e32 v7, v7, v12
	;;#ASMSTART
	ds_read_b128 v[0x4a:0x4d], v10 offset:0x3c00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[50:53], a[0x7c:0x7f], a[28:31], v[50:53]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[46], 0x3e0293ee, v[46]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[47], 0x3e0293ee, v[47]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[48], 0x3e0293ee, v[48]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[49], 0x3e0293ee, v[49]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(6)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[54:57], a[0x70:0x73], a[32:35], 0
	;;#ASMEND
	v_add_u32_e32 v18, s62, v11
	;;#ASMSTART
	ds_read_b128 v[0x4e:0x51], v18 offset:0
	;;#ASMEND
	v_and_b32_e32 v6, 0x2c0, v6
	v_lshlrev_b32_e32 v7, 1, v7
	;;#ASMSTART
	ds_read_b128 v[0x52:0x55], v18 offset:0x400
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[54:57], a[0x74:0x77], a[36:39], v[54:57]
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[46], v[0x7e], v[46] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[47], v[0x7e], v[47] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[48], v[0x7e], v[48] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[49], v[0x7e], v[49] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(6)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[54:57], a[0x78:0x7b], a[40:43], v[54:57]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[0x56:0x59], v18 offset:0x800
	;;#ASMEND
	v_bitop3_b32 v12, v7, v8, v6 bitop3:0x36
	;;#ASMSTART
	ds_read_b128 v[0x5a:0x5d], v18 offset:0xc00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[54:57], a[0x7c:0x7f], a[44:47], v[54:57]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[50], 0x3e0293ee, v[50]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[51], 0x3e0293ee, v[51]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[52], 0x3e0293ee, v[52]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[53], 0x3e0293ee, v[53]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(6)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[58:61], a[0x70:0x73], v[62:0x41], 0
	;;#ASMEND
	v_add_u32_e32 v6, s62, v12
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x5e:0x5f], v6 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x60:0x61], v6 offset:0x100
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x62:0x63], v6 offset:0x400
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x64:0x65], v6 offset:0x500
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[58:61], a[0x74:0x77], v[0x42:0x45], v[58:61]
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[50], v[0x7e], v[50] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[51], v[0x7e], v[51] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[52], v[0x7e], v[52] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[53], v[0x7e], v[53] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[58:61], a[0x78:0x7b], v[0x46:0x49], v[58:61]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x66:0x67], v6 offset:0x800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x68:0x69], v6 offset:0x900
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x6a:0x6b], v6 offset:0xc00
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x6c:0x6d], v6 offset:0xd00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[58:61], a[0x7c:0x7f], v[0x4a:0x4d], v[58:61]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[54], 0x3e0293ee, v[54]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[55], 0x3e0293ee, v[55]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[56], 0x3e0293ee, v[56]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[57], 0x3e0293ee, v[57]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[62:0x41], v[0x4e:0x51], a[48:51], 0
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[54], v[0x7e], v[54] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[55], v[0x7e], v[55] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[56], v[0x7e], v[56] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[57], v[0x7e], v[57] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[62:0x41], v[0x52:0x55], a[52:55], v[62:0x41]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[46], v[46]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[47], v[47]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[48], v[48]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[49], v[49]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[62:0x41], v[0x56:0x59], a[56:59], v[62:0x41]
	;;#ASMEND
	v_add_u32_e32 v6, s27, v12
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x70:0x71], v6 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x72:0x73], v6 offset:0x100
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[62:0x41], v[0x5a:0x5d], a[60:63], v[62:0x41]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[50], v[50]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[51], v[51]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[52], v[52]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[53], v[53]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x42:0x45], v[0x4e:0x51], a[64:0x43], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x74:0x75], v6 offset:0x400
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x76:0x77], v6 offset:0x500
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x42:0x45], v[0x52:0x55], a[0x44:0x47], v[0x42:0x45]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[58], 0x3e0293ee, v[58]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[59], 0x3e0293ee, v[59]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[60], 0x3e0293ee, v[60]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[61], 0x3e0293ee, v[61]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x42:0x45], v[0x56:0x59], a[0x48:0x4b], v[0x42:0x45]
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[58], v[0x7e], v[58] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[59], v[0x7e], v[59] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[60], v[0x7e], v[60] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[61], v[0x7e], v[61] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x42:0x45], v[0x5a:0x5d], a[0x4c:0x4f], v[0x42:0x45]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x76], v[46], v[47]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x77], v[48], v[49]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x46:0x49], v[0x4e:0x51], a[0x50:0x53], 0
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[54], v[54]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[55], v[55]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[56], v[56]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[57], v[57]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x46:0x49], v[0x52:0x55], a[0x54:0x57], v[0x46:0x49]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x78], v[50], v[51]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x79], v[52], v[53]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x46:0x49], v[0x56:0x59], a[0x58:0x5b], v[0x46:0x49]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x78:0x79], v6 offset:0x800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x7a:0x7b], v6 offset:0x900
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x46:0x49], v[0x5a:0x5d], a[0x5c:0x5f], v[0x46:0x49]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[58], v[58]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[59], v[59]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[60], v[60]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[61], v[61]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x4a:0x4d], v[0x4e:0x51], a[0x60:0x63], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x7c:0x7d], v6 offset:0xc00
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x7e:0x7f], v6 offset:0xd00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x4a:0x4d], v[0x52:0x55], a[0x64:0x67], v[0x4a:0x4d]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x7a], v[54], v[55]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x7b], v[56], v[57]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x7c], v[58], v[59]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x7d], v[60], v[61]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x4a:0x4d], v[0x56:0x59], a[0x68:0x6b], v[0x4a:0x4d]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[0x76], v[0x78]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[0x77], v[0x79]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[0x7a], v[0x7c]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[0x7b], v[0x7d]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x4a:0x4d], v[0x5a:0x5d], a[0x6c:0x6f], v[0x4a:0x4d]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0x80:0x8f], v[0x5e:0x61], v[0x76:0x79], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0:1], v1 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[2:3], v1 offset:0x1000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[4:5], v1 offset:0x200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[6:7], v1 offset:0x1200
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0x90:0x9f], v[0x5e:0x61], v[0x7a:0x7d], 0
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[62], v[0x7f], v[62] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[63], v[0x7f], v[63] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[64], v[0x7f], v[64] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x41], v[0x7f], v[0x41] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x42], v[0x7f], v[0x42] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x43], v[0x7f], v[0x43] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x44], v[0x7f], v[0x44] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x45], v[0x7f], v[0x45] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0xa0:0xaf], v[0x62:0x65], v[0x76:0x79], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[8:9], v1 offset:0x2000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[10:11], v1 offset:0x3000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[12:13], v1 offset:0x2200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[14:15], v1 offset:0x3200
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0xb0:0xbf], v[0x62:0x65], v[0x7a:0x7d], 0
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[62], v[46], v[62]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[63], v[47], v[63]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[64], v[48], v[64]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x41], v[49], v[0x41]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x42], v[50], v[0x42]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x43], v[51], v[0x43]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x44], v[52], v[0x44]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x45], v[53], v[0x45]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[62], v[62], v[63]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[63], v[64], v[0x41]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[64], v[0x42], v[0x43]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x41], v[0x44], v[0x45]
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x46], v[0x7f], v[0x46] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x47], v[0x7f], v[0x47] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x48], v[0x7f], v[0x48] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x49], v[0x7f], v[0x49] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0xc0:0xcf], v[0x66:0x69], v[0x76:0x79], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[16:17], v1 offset:0x4000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[18:19], v1 offset:0x5000
	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v9, v[62:63], offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v9, v[64:0x41], offset:0x200
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0xd0:0xdf], v[0x66:0x69], v[0x7a:0x7d], 0
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x4a], v[0x7f], v[0x4a] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x4b], v[0x7f], v[0x4b] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x4c], v[0x7f], v[0x4c] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x4d], v[0x7f], v[0x4d] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x46], v[54], v[0x46]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x47], v[55], v[0x47]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x48], v[56], v[0x48]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x49], v[57], v[0x49]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x4a], v[58], v[0x4a]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x4b], v[59], v[0x4b]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x4c], v[60], v[0x4c]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x4d], v[61], v[0x4d]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x42], v[0x46], v[0x47]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x43], v[0x48], v[0x49]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x44], v[0x4a], v[0x4b]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x45], v[0x4c], v[0x4d]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0xe0:0xef], v[0x6a:0x6d], v[0x76:0x79], 0
	;;#ASMEND
	v_add3_u32 v6, s5, 64, v13
	;;#ASMSTART
	ds_read_b32 v[0x7e], v6 offset:0

	;;#ASMEND
	v_add3_u32 v6, s23, 64, v13
	;;#ASMSTART
	ds_read_b32 v[0x7f], v6 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v9, v[0x42:0x43], offset:0x400
	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v9, v[0x44:0x45], offset:0x600
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0xf0:0xff], v[0x6a:0x6d], v[0x7a:0x7d], 0
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[62], v[64]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[63], v[0x41]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[0x42], v[0x44]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[0x43], v[0x45]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(12)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0x80:0x8f], a[0x70:0x73], v[62:0x41], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[20:21], v1 offset:0x4200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[22:23], v1 offset:0x5200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[24:25], v1 offset:0x6000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[26:27], v1 offset:0x7000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[28:29], v1 offset:0x6200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[30:31], v1 offset:0x7200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[32:33], v1 offset:0x8000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[34:35], v1 offset:0x9000
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0x90:0x9f], a[0x70:0x73], v[0x42:0x45], 0
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0xa0:0xaf], a[0x74:0x77], v[62:0x41], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[30:31], v4 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[32:33], v4 offset:0x200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[34:35], v4 offset:0x400
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[36:37], v4 offset:0x600
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[38:39], v4 offset:0x800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[40:41], v4 offset:0xa00
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[42:43], v4 offset:0xc00
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[44:45], v4 offset:0xe00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0xb0:0xbf], a[0x74:0x77], v[0x42:0x45], 0
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x7e], 0x3fb8aa3b, v[0x7e]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0xc0:0xcf], a[0x78:0x7b], v[62:0x41], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[46:47], v4 offset:0x1000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[48:49], v4 offset:0x1200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[50:51], v4 offset:0x1400
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[52:53], v4 offset:0x1600
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[36:37], v1 offset:0x8200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[38:39], v1 offset:0x9200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[40:41], v1 offset:0xa000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[42:43], v1 offset:0xb000
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0xd0:0xdf], a[0x78:0x7b], v[0x42:0x45], 0
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0xe0:0xef], a[0x7c:0x7f], v[62:0x41], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[54:55], v4 offset:0x1800
	;;#ASMEND
	s_add_i32 s0, s67, 64
	;;#ASMSTART
	ds_read_b64_tr_b16 v[56:57], v4 offset:0x1a00
	;;#ASMEND
	s_mul_i32 s0, s0, s26
	;;#ASMSTART
	ds_read_b64_tr_b16 v[58:59], v4 offset:0x1c00
	;;#ASMEND
	s_add_i32 s0, s0, s61
	;;#ASMSTART
	ds_read_b64_tr_b16 v[60:61], v4 offset:0x1e00
	;;#ASMEND
	s_mul_i32 s0, s0, s60
	;;#ASMSTART
	ds_read_b64_tr_b16 a[44:45], v1 offset:0xa200
	;;#ASMEND
	s_add_i32 s10, s62, 0x4000
	s_ashr_i32 s1, s0, 31
	;;#ASMSTART
	ds_read_b64_tr_b16 a[46:47], v1 offset:0xb200
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0xf0:0xff], a[0x7c:0x7f], v[0x42:0x45], 0
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0) lgkmcnt(6)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], a[0:3], v[30:33], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[62:63], v1 offset:0xc000
	;;#ASMEND
	s_lshl_b64 s[0:1], s[0:1], 1
	v_add_u32_e32 v6, s10, v17
	;;#ASMSTART
	ds_read_b64_tr_b16 v[64:0x41], v1 offset:0xd000
	;;#ASMEND
	s_add_u32 s44, s52, s0
	v_readfirstlane_b32 s0, v6
	v_add_u32_e32 v6, 0x1000, v6
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x42:0x43], v1 offset:0xc200
	;;#ASMEND
	s_addc_u32 s45, s53, s1
	s_mov_b32 m0, s0
	v_readfirstlane_b32 s0, v6
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x44:0x45], v1 offset:0xd200
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], a[8:11], v[34:37], v[0x6e:0x71]
	;;#ASMEND
	buffer_load_dwordx4 v14, s[44:47], 0 offen lds
	s_mov_b32 m0, s0
	s_add_i32 s39, s23, 0x100
	buffer_load_dwordx4 v15, s[44:47], 0 offen lds
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], a[16:19], v[38:41], v[0x6e:0x71]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x46:0x47], v1 offset:0xe000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x48:0x49], v1 offset:0xf000
	;;#ASMEND
	s_add_u32 s40, s8, 0x100
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x4a:0x4b], v1 offset:0xe200
	;;#ASMEND
	s_addc_u32 s41, s9, 0
	s_mov_b32 m0, s39
	s_add_i32 s0, s27, 0x1000
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x4c:0x4d], v1 offset:0xf200
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], a[24:27], v[42:45], v[0x6e:0x71]
	;;#ASMEND
	buffer_load_dword v16, s[40:43], 0 offen lds
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], a[32:35], v[46:49], v[0x6e:0x71]
	;;#ASMEND
	v_add_u32_e32 v6, s0, v11
	;;#ASMSTART
	ds_read_b128 a[0x70:0x73], v6 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[0x74:0x77], v6 offset:0x400
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], a[40:43], v[50:53], v[0x6e:0x71]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], v[62:0x41], v[54:57], v[0x6e:0x71]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[0x78:0x7b], v6 offset:0x800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[0x7c:0x7f], v6 offset:0xc00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], v[0x46:0x49], v[58:61], v[0x6e:0x71]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], a[4:7], v[30:33], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[0:3], v10 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[4:7], v10 offset:0x400
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], a[12:15], v[34:37], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], a[20:23], v[38:41], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[8:11], v10 offset:0x800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[12:15], v10 offset:0xc00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], a[28:31], v[42:45], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x6e], 0x3db504f3, v[0x6e]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x6f], 0x3db504f3, v[0x6f]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], a[36:39], v[46:49], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[16:19], v10 offset:0x1000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[20:23], v10 offset:0x1400
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], a[44:47], v[50:53], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x70], 0x3db504f3, v[0x70]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x71], 0x3db504f3, v[0x71]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(10)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], v[0x42:0x45], v[54:57], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[24:27], v10 offset:0x1800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[28:31], v10 offset:0x1c00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], v[0x4a:0x4d], v[58:61], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(2)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[46:49], a[0x70:0x73], a[0:3], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[32:35], v10 offset:0x2000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[36:39], v10 offset:0x2400
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[46:49], a[0x74:0x77], a[4:7], v[46:49]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[46:49], a[0x78:0x7b], a[8:11], v[46:49]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[40:43], v10 offset:0x2800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[44:47], v10 offset:0x2c00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[46:49], a[0x7c:0x7f], a[12:15], v[46:49]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x72], 0x3db504f3, v[0x72]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x73], 0x3db504f3, v[0x73]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[50:53], a[0x70:0x73], a[16:19], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[62:0x41], v10 offset:0x3000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[0x42:0x45], v10 offset:0x3400
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[50:53], a[0x74:0x77], a[20:23], v[50:53]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x74], 0x3db504f3, v[0x74]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x75], 0x3db504f3, v[0x75]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[50:53], a[0x78:0x7b], a[24:27], v[50:53]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[0x46:0x49], v10 offset:0x3800
	;;#ASMEND
	s_add_i32 s1, s62, 0x1000
	;;#ASMSTART
	ds_read_b128 v[0x4a:0x4d], v10 offset:0x3c00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[50:53], a[0x7c:0x7f], a[28:31], v[50:53]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[46], 0x3e0293ee, v[46]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[47], 0x3e0293ee, v[47]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[48], 0x3e0293ee, v[48]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[49], 0x3e0293ee, v[49]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(6)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[54:57], a[0x70:0x73], a[32:35], 0
	;;#ASMEND
	v_add_u32_e32 v6, s1, v11
	;;#ASMSTART
	ds_read_b128 v[0x4e:0x51], v6 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[0x52:0x55], v6 offset:0x400
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[54:57], a[0x74:0x77], a[36:39], v[54:57]
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[46], v[0x7e], v[46] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[47], v[0x7e], v[47] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[48], v[0x7e], v[48] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[49], v[0x7e], v[49] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(6)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[54:57], a[0x78:0x7b], a[40:43], v[54:57]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[0x56:0x59], v6 offset:0x800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[0x5a:0x5d], v6 offset:0xc00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[54:57], a[0x7c:0x7f], a[44:47], v[54:57]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[50], 0x3e0293ee, v[50]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[51], 0x3e0293ee, v[51]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[52], 0x3e0293ee, v[52]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[53], 0x3e0293ee, v[53]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(6)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[58:61], a[0x70:0x73], v[62:0x41], 0
	;;#ASMEND
	v_add_u32_e32 v6, s1, v12
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x5e:0x5f], v6 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x60:0x61], v6 offset:0x100
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x62:0x63], v6 offset:0x400
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x64:0x65], v6 offset:0x500
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[58:61], a[0x74:0x77], v[0x42:0x45], v[58:61]
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[50], v[0x7e], v[50] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[51], v[0x7e], v[51] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[52], v[0x7e], v[52] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[53], v[0x7e], v[53] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[58:61], a[0x78:0x7b], v[0x46:0x49], v[58:61]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x66:0x67], v6 offset:0x800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x68:0x69], v6 offset:0x900
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x6a:0x6b], v6 offset:0xc00
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x6c:0x6d], v6 offset:0xd00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[58:61], a[0x7c:0x7f], v[0x4a:0x4d], v[58:61]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[54], 0x3e0293ee, v[54]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[55], 0x3e0293ee, v[55]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[56], 0x3e0293ee, v[56]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[57], 0x3e0293ee, v[57]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[62:0x41], v[0x4e:0x51], a[48:51], 0
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[54], v[0x7e], v[54] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[55], v[0x7e], v[55] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[56], v[0x7e], v[56] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[57], v[0x7e], v[57] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[62:0x41], v[0x52:0x55], a[52:55], v[62:0x41]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[46], v[46]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[47], v[47]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[48], v[48]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[49], v[49]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[62:0x41], v[0x56:0x59], a[56:59], v[62:0x41]
	;;#ASMEND
	v_add_u32_e32 v6, s0, v12
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x70:0x71], v6 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x72:0x73], v6 offset:0x100
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[62:0x41], v[0x5a:0x5d], a[60:63], v[62:0x41]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[50], v[50]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[51], v[51]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[52], v[52]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[53], v[53]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x42:0x45], v[0x4e:0x51], a[64:0x43], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x74:0x75], v6 offset:0x400
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x76:0x77], v6 offset:0x500
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x42:0x45], v[0x52:0x55], a[0x44:0x47], v[0x42:0x45]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[58], 0x3e0293ee, v[58]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[59], 0x3e0293ee, v[59]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[60], 0x3e0293ee, v[60]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[61], 0x3e0293ee, v[61]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x42:0x45], v[0x56:0x59], a[0x48:0x4b], v[0x42:0x45]
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[58], v[0x7e], v[58] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[59], v[0x7e], v[59] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[60], v[0x7e], v[60] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[61], v[0x7e], v[61] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x42:0x45], v[0x5a:0x5d], a[0x4c:0x4f], v[0x42:0x45]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x76], v[46], v[47]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x77], v[48], v[49]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x46:0x49], v[0x4e:0x51], a[0x50:0x53], 0
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[54], v[54]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[55], v[55]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[56], v[56]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[57], v[57]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x46:0x49], v[0x52:0x55], a[0x54:0x57], v[0x46:0x49]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x78], v[50], v[51]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x79], v[52], v[53]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x46:0x49], v[0x56:0x59], a[0x58:0x5b], v[0x46:0x49]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x78:0x79], v6 offset:0x800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x7a:0x7b], v6 offset:0x900
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x46:0x49], v[0x5a:0x5d], a[0x5c:0x5f], v[0x46:0x49]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[58], v[58]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[59], v[59]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[60], v[60]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[61], v[61]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x4a:0x4d], v[0x4e:0x51], a[0x60:0x63], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x7c:0x7d], v6 offset:0xc00
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x7e:0x7f], v6 offset:0xd00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x4a:0x4d], v[0x52:0x55], a[0x64:0x67], v[0x4a:0x4d]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x7a], v[54], v[55]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x7b], v[56], v[57]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x7c], v[58], v[59]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x7d], v[60], v[61]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x4a:0x4d], v[0x56:0x59], a[0x68:0x6b], v[0x4a:0x4d]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[0x76], v[0x78]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[0x77], v[0x79]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[0x7a], v[0x7c]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[0x7b], v[0x7d]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x4a:0x4d], v[0x5a:0x5d], a[0x6c:0x6f], v[0x4a:0x4d]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0x80:0x8f], v[0x5e:0x61], v[0x76:0x79], v[0x80:0x8f]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0:1], v1 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[2:3], v1 offset:0x1000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[4:5], v1 offset:0x200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[6:7], v1 offset:0x1200
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0x90:0x9f], v[0x5e:0x61], v[0x7a:0x7d], v[0x90:0x9f]
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[62], v[0x7f], v[62] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[63], v[0x7f], v[63] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[64], v[0x7f], v[64] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x41], v[0x7f], v[0x41] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x42], v[0x7f], v[0x42] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x43], v[0x7f], v[0x43] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x44], v[0x7f], v[0x44] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x45], v[0x7f], v[0x45] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0xa0:0xaf], v[0x62:0x65], v[0x76:0x79], v[0xa0:0xaf]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[8:9], v1 offset:0x2000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[10:11], v1 offset:0x3000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[12:13], v1 offset:0x2200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[14:15], v1 offset:0x3200
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0xb0:0xbf], v[0x62:0x65], v[0x7a:0x7d], v[0xb0:0xbf]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[62], v[46], v[62]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[63], v[47], v[63]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[64], v[48], v[64]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x41], v[49], v[0x41]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x42], v[50], v[0x42]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x43], v[51], v[0x43]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x44], v[52], v[0x44]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x45], v[53], v[0x45]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[62], v[62], v[63]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[63], v[64], v[0x41]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[64], v[0x42], v[0x43]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x41], v[0x44], v[0x45]
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x46], v[0x7f], v[0x46] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x47], v[0x7f], v[0x47] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x48], v[0x7f], v[0x48] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x49], v[0x7f], v[0x49] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0xc0:0xcf], v[0x66:0x69], v[0x76:0x79], v[0xc0:0xcf]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[16:17], v1 offset:0x4000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[18:19], v1 offset:0x5000
	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v9, v[62:63], offset:0
	;;#ASMEND
	s_add_i32 s0, s5, 0x80
	;;#ASMSTART
	ds_write_b64 v9, v[64:0x41], offset:0x200
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0xd0:0xdf], v[0x66:0x69], v[0x7a:0x7d], v[0xd0:0xdf]
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x4a], v[0x7f], v[0x4a] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x4b], v[0x7f], v[0x4b] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x4c], v[0x7f], v[0x4c] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x4d], v[0x7f], v[0x4d] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x46], v[54], v[0x46]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x47], v[55], v[0x47]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x48], v[56], v[0x48]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x49], v[57], v[0x49]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x4a], v[58], v[0x4a]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x4b], v[59], v[0x4b]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x4c], v[60], v[0x4c]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x4d], v[61], v[0x4d]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x42], v[0x46], v[0x47]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x43], v[0x48], v[0x49]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x44], v[0x4a], v[0x4b]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x45], v[0x4c], v[0x4d]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0xe0:0xef], v[0x6a:0x6d], v[0x76:0x79], v[0xe0:0xef]
	;;#ASMEND
	v_add_u32_e32 v6, s0, v13
	;;#ASMSTART
	ds_read_b32 v[0x7e], v6 offset:0

	;;#ASMEND
	s_add_i32 s0, s23, 0x80
	v_add_u32_e32 v6, s0, v13
	;;#ASMSTART
	ds_read_b32 v[0x7f], v6 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v9, v[0x42:0x43], offset:0x400
	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v9, v[0x44:0x45], offset:0x600
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0xf0:0xff], v[0x6a:0x6d], v[0x7a:0x7d], v[0xf0:0xff]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[62], v[64]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[63], v[0x41]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[0x42], v[0x44]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[0x43], v[0x45]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(12)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0x80:0x8f], a[0x70:0x73], v[62:0x41], a[0x80:0x8f]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[20:21], v1 offset:0x4200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[22:23], v1 offset:0x5200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[24:25], v1 offset:0x6000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[26:27], v1 offset:0x7000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[28:29], v1 offset:0x6200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[30:31], v1 offset:0x7200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[32:33], v1 offset:0x8000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[34:35], v1 offset:0x9000
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0x90:0x9f], a[0x70:0x73], v[0x42:0x45], a[0x90:0x9f]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0xa0:0xaf], a[0x74:0x77], v[62:0x41], a[0xa0:0xaf]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[30:31], v4 offset:0
	;;#ASMEND
	s_mul_i32 s15, s4, s36
	;;#ASMSTART
	ds_read_b64_tr_b16 v[32:33], v4 offset:0x200
	;;#ASMEND
	s_add_i32 s15, s15, s61
	;;#ASMSTART
	ds_read_b64_tr_b16 v[34:35], v4 offset:0x400
	;;#ASMEND
	s_mul_i32 s0, s15, s38
	;;#ASMSTART
	ds_read_b64_tr_b16 v[36:37], v4 offset:0x600
	;;#ASMEND
	s_mul_i32 s0, s0, s3
	;;#ASMSTART
	ds_read_b64_tr_b16 v[38:39], v4 offset:0x800
	;;#ASMEND
	s_ashr_i32 s1, s0, 31
	;;#ASMSTART
	ds_read_b64_tr_b16 v[40:41], v4 offset:0xa00
	;;#ASMEND
	s_lshl_b64 s[8:9], s[0:1], 1
	;;#ASMSTART
	ds_read_b64_tr_b16 v[42:43], v4 offset:0xc00
	;;#ASMEND
	s_add_u32 s8, s6, s8
	;;#ASMSTART
	ds_read_b64_tr_b16 v[44:45], v4 offset:0xe00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0xb0:0xbf], a[0x74:0x77], v[0x42:0x45], a[0xb0:0xbf]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x7e], 0x3fb8aa3b, v[0x7e]
	;;#ASMEND
	s_addc_u32 s9, s7, s9
	s_lshl_b32 s10, s3, 5
	v_or_b32_e32 v5, v5, v16
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x6e], v[0x6e], v[0x6f]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x6f], v[0x70], v[0x71]
	;;#ASMEND
	;;#ASMSTART
	buffer_atomic_pk_add_bf16 v[0x6e], v5, s[8:11], 0 offen
	;;#ASMEND
	v_or_b32_e32 v6, 0x100, v5
	;;#ASMSTART
	buffer_atomic_pk_add_bf16 v[0x6f], v6, s[8:11], 0 offen
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0xc0:0xcf], a[0x78:0x7b], v[62:0x41], a[0xc0:0xcf]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[46:47], v4 offset:0x1000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[48:49], v4 offset:0x1200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[50:51], v4 offset:0x1400
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[52:53], v4 offset:0x1600
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[36:37], v1 offset:0x8200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[38:39], v1 offset:0x9200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[40:41], v1 offset:0xa000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[42:43], v1 offset:0xb000
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0xd0:0xdf], a[0x78:0x7b], v[0x42:0x45], a[0xd0:0xdf]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x72], v[0x72], v[0x73]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x73], v[0x74], v[0x75]
	;;#ASMEND
	v_or_b32_e32 v7, 0x200, v5
	;;#ASMSTART
	buffer_atomic_pk_add_bf16 v[0x72], v7, s[8:11], 0 offen
	;;#ASMEND
	v_or_b32_e32 v8, 0x300, v5
	;;#ASMSTART
	buffer_atomic_pk_add_bf16 v[0x73], v8, s[8:11], 0 offen
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0xe0:0xef], a[0x7c:0x7f], v[62:0x41], a[0xe0:0xef]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[54:55], v4 offset:0x1800
	;;#ASMEND
	s_add_i32 s8, s66, 0x60
	;;#ASMSTART
	ds_read_b64_tr_b16 v[56:57], v4 offset:0x1a00
	;;#ASMEND
	s_mul_i32 s8, s8, s22
	;;#ASMSTART
	ds_read_b64_tr_b16 v[58:59], v4 offset:0x1c00
	;;#ASMEND
	s_add_i32 s8, s8, s61
	;;#ASMSTART
	ds_read_b64_tr_b16 v[60:61], v4 offset:0x1e00
	;;#ASMEND
	s_mul_i32 s8, s8, s33
	;;#ASMSTART
	ds_read_b64_tr_b16 a[44:45], v1 offset:0xa200
	;;#ASMEND
	s_add_i32 s1, s27, 0x6000
	s_ashr_i32 s9, s8, 31
	;;#ASMSTART
	ds_read_b64_tr_b16 a[46:47], v1 offset:0xb200
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0xf0:0xff], a[0x7c:0x7f], v[0x42:0x45], a[0xf0:0xff]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(4) lgkmcnt(6)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], a[0:3], v[30:33], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[62:63], v1 offset:0xc000
	;;#ASMEND
	s_lshl_b64 s[8:9], s[8:9], 1
	v_add_u32_e32 v18, s1, v17
	;;#ASMSTART
	ds_read_b64_tr_b16 v[64:0x41], v1 offset:0xd000
	;;#ASMEND
	s_add_u32 s28, s50, s8
	v_readfirstlane_b32 s1, v18
	v_add_u32_e32 v18, 0x1000, v18
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x42:0x43], v1 offset:0xc200
	;;#ASMEND
	s_addc_u32 s29, s51, s9
	s_mov_b32 m0, s1
	v_readfirstlane_b32 s1, v18
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x44:0x45], v1 offset:0xd200
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], a[8:11], v[34:37], v[0x6e:0x71]
	;;#ASMEND
	buffer_load_dwordx4 v14, s[28:31], 0 offen lds
	s_mov_b32 m0, s1
	v_add_u32_e32 v18, s70, v11
	buffer_load_dwordx4 v15, s[28:31], 0 offen lds
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], a[16:19], v[38:41], v[0x6e:0x71]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x46:0x47], v1 offset:0xe000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x48:0x49], v1 offset:0xf000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x4a:0x4b], v1 offset:0xe200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x4c:0x4d], v1 offset:0xf200
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], a[24:27], v[42:45], v[0x6e:0x71]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], a[32:35], v[46:49], v[0x6e:0x71]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[0x70:0x73], v18 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[0x74:0x77], v18 offset:0x400
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], a[40:43], v[50:53], v[0x6e:0x71]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], v[62:0x41], v[54:57], v[0x6e:0x71]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[0x78:0x7b], v18 offset:0x800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[0x7c:0x7f], v18 offset:0xc00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], v[0x46:0x49], v[58:61], v[0x6e:0x71]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], a[4:7], v[30:33], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[0:3], v10 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[4:7], v10 offset:0x400
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], a[12:15], v[34:37], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], a[20:23], v[38:41], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[8:11], v10 offset:0x800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[12:15], v10 offset:0xc00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], a[28:31], v[42:45], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x6e], 0x3db504f3, v[0x6e]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x6f], 0x3db504f3, v[0x6f]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], a[36:39], v[46:49], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[16:19], v10 offset:0x1000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[20:23], v10 offset:0x1400
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], a[44:47], v[50:53], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x70], 0x3db504f3, v[0x70]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x71], 0x3db504f3, v[0x71]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(10)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], v[0x42:0x45], v[54:57], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[24:27], v10 offset:0x1800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[28:31], v10 offset:0x1c00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], v[0x4a:0x4d], v[58:61], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(2)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[46:49], a[0x70:0x73], a[0:3], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[32:35], v10 offset:0x2000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[36:39], v10 offset:0x2400
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[46:49], a[0x74:0x77], a[4:7], v[46:49]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[46:49], a[0x78:0x7b], a[8:11], v[46:49]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[40:43], v10 offset:0x2800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[44:47], v10 offset:0x2c00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[46:49], a[0x7c:0x7f], a[12:15], v[46:49]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x72], 0x3db504f3, v[0x72]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x73], 0x3db504f3, v[0x73]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[50:53], a[0x70:0x73], a[16:19], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[62:0x41], v10 offset:0x3000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[0x42:0x45], v10 offset:0x3400
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[50:53], a[0x74:0x77], a[20:23], v[50:53]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x74], 0x3db504f3, v[0x74]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x75], 0x3db504f3, v[0x75]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[50:53], a[0x78:0x7b], a[24:27], v[50:53]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[0x46:0x49], v10 offset:0x3800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[0x4a:0x4d], v10 offset:0x3c00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[50:53], a[0x7c:0x7f], a[28:31], v[50:53]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[46], 0x3e0293ee, v[46]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[47], 0x3e0293ee, v[47]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[48], 0x3e0293ee, v[48]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[49], 0x3e0293ee, v[49]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(6)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[54:57], a[0x70:0x73], a[32:35], 0
	;;#ASMEND
	v_add_u32_e32 v18, s71, v11
	;;#ASMSTART
	ds_read_b128 v[0x4e:0x51], v18 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[0x52:0x55], v18 offset:0x400
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[54:57], a[0x74:0x77], a[36:39], v[54:57]
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[46], v[0x7e], v[46] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[47], v[0x7e], v[47] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[48], v[0x7e], v[48] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[49], v[0x7e], v[49] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(6)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[54:57], a[0x78:0x7b], a[40:43], v[54:57]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[0x56:0x59], v18 offset:0x800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[0x5a:0x5d], v18 offset:0xc00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[54:57], a[0x7c:0x7f], a[44:47], v[54:57]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[50], 0x3e0293ee, v[50]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[51], 0x3e0293ee, v[51]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[52], 0x3e0293ee, v[52]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[53], 0x3e0293ee, v[53]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(6)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[58:61], a[0x70:0x73], v[62:0x41], 0
	;;#ASMEND
	v_add_u32_e32 v18, s71, v12
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x5e:0x5f], v18 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x60:0x61], v18 offset:0x100
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x62:0x63], v18 offset:0x400
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x64:0x65], v18 offset:0x500
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[58:61], a[0x74:0x77], v[0x42:0x45], v[58:61]
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[50], v[0x7e], v[50] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[51], v[0x7e], v[51] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[52], v[0x7e], v[52] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[53], v[0x7e], v[53] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[58:61], a[0x78:0x7b], v[0x46:0x49], v[58:61]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x66:0x67], v18 offset:0x800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x68:0x69], v18 offset:0x900
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x6a:0x6b], v18 offset:0xc00
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x6c:0x6d], v18 offset:0xd00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[58:61], a[0x7c:0x7f], v[0x4a:0x4d], v[58:61]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[54], 0x3e0293ee, v[54]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[55], 0x3e0293ee, v[55]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[56], 0x3e0293ee, v[56]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[57], 0x3e0293ee, v[57]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[62:0x41], v[0x4e:0x51], a[48:51], 0
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[54], v[0x7e], v[54] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[55], v[0x7e], v[55] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[56], v[0x7e], v[56] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[57], v[0x7e], v[57] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[62:0x41], v[0x52:0x55], a[52:55], v[62:0x41]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[46], v[46]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[47], v[47]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[48], v[48]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[49], v[49]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[62:0x41], v[0x56:0x59], a[56:59], v[62:0x41]
	;;#ASMEND
	v_add_u32_e32 v18, s70, v12
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x70:0x71], v18 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x72:0x73], v18 offset:0x100
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[62:0x41], v[0x5a:0x5d], a[60:63], v[62:0x41]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[50], v[50]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[51], v[51]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[52], v[52]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[53], v[53]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x42:0x45], v[0x4e:0x51], a[64:0x43], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x74:0x75], v18 offset:0x400
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x76:0x77], v18 offset:0x500
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x42:0x45], v[0x52:0x55], a[0x44:0x47], v[0x42:0x45]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[58], 0x3e0293ee, v[58]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[59], 0x3e0293ee, v[59]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[60], 0x3e0293ee, v[60]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[61], 0x3e0293ee, v[61]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x42:0x45], v[0x56:0x59], a[0x48:0x4b], v[0x42:0x45]
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[58], v[0x7e], v[58] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[59], v[0x7e], v[59] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[60], v[0x7e], v[60] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[61], v[0x7e], v[61] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x42:0x45], v[0x5a:0x5d], a[0x4c:0x4f], v[0x42:0x45]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x76], v[46], v[47]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x77], v[48], v[49]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x46:0x49], v[0x4e:0x51], a[0x50:0x53], 0
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[54], v[54]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[55], v[55]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[56], v[56]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[57], v[57]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x46:0x49], v[0x52:0x55], a[0x54:0x57], v[0x46:0x49]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x78], v[50], v[51]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x79], v[52], v[53]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x46:0x49], v[0x56:0x59], a[0x58:0x5b], v[0x46:0x49]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x78:0x79], v18 offset:0x800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x7a:0x7b], v18 offset:0x900
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x46:0x49], v[0x5a:0x5d], a[0x5c:0x5f], v[0x46:0x49]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[58], v[58]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[59], v[59]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[60], v[60]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[61], v[61]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x4a:0x4d], v[0x4e:0x51], a[0x60:0x63], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x7c:0x7d], v18 offset:0xc00
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x7e:0x7f], v18 offset:0xd00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x4a:0x4d], v[0x52:0x55], a[0x64:0x67], v[0x4a:0x4d]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x7a], v[54], v[55]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x7b], v[56], v[57]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x7c], v[58], v[59]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x7d], v[60], v[61]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x4a:0x4d], v[0x56:0x59], a[0x68:0x6b], v[0x4a:0x4d]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[0x76], v[0x78]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[0x77], v[0x79]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[0x7a], v[0x7c]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[0x7b], v[0x7d]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x4a:0x4d], v[0x5a:0x5d], a[0x6c:0x6f], v[0x4a:0x4d]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0x80:0x8f], v[0x5e:0x61], v[0x76:0x79], v[0x80:0x8f]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0:1], v1 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[2:3], v1 offset:0x1000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[4:5], v1 offset:0x200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[6:7], v1 offset:0x1200
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0x90:0x9f], v[0x5e:0x61], v[0x7a:0x7d], v[0x90:0x9f]
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[62], v[0x7f], v[62] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[63], v[0x7f], v[63] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[64], v[0x7f], v[64] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x41], v[0x7f], v[0x41] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x42], v[0x7f], v[0x42] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x43], v[0x7f], v[0x43] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x44], v[0x7f], v[0x44] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x45], v[0x7f], v[0x45] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0xa0:0xaf], v[0x62:0x65], v[0x76:0x79], v[0xa0:0xaf]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[8:9], v1 offset:0x2000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[10:11], v1 offset:0x3000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[12:13], v1 offset:0x2200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[14:15], v1 offset:0x3200
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0xb0:0xbf], v[0x62:0x65], v[0x7a:0x7d], v[0xb0:0xbf]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[62], v[46], v[62]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[63], v[47], v[63]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[64], v[48], v[64]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x41], v[49], v[0x41]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x42], v[50], v[0x42]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x43], v[51], v[0x43]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x44], v[52], v[0x44]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x45], v[53], v[0x45]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[62], v[62], v[63]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[63], v[64], v[0x41]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[64], v[0x42], v[0x43]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x41], v[0x44], v[0x45]
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x46], v[0x7f], v[0x46] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x47], v[0x7f], v[0x47] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x48], v[0x7f], v[0x48] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x49], v[0x7f], v[0x49] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0xc0:0xcf], v[0x66:0x69], v[0x76:0x79], v[0xc0:0xcf]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[16:17], v1 offset:0x4000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[18:19], v1 offset:0x5000
	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v9, v[62:63], offset:0
	;;#ASMEND
	s_add_i32 s1, s5, 0xc0
	;;#ASMSTART
	ds_write_b64 v9, v[64:0x41], offset:0x200
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0xd0:0xdf], v[0x66:0x69], v[0x7a:0x7d], v[0xd0:0xdf]
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x4a], v[0x7f], v[0x4a] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x4b], v[0x7f], v[0x4b] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x4c], v[0x7f], v[0x4c] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x4d], v[0x7f], v[0x4d] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x46], v[54], v[0x46]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x47], v[55], v[0x47]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x48], v[56], v[0x48]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x49], v[57], v[0x49]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x4a], v[58], v[0x4a]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x4b], v[59], v[0x4b]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x4c], v[60], v[0x4c]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x4d], v[61], v[0x4d]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x42], v[0x46], v[0x47]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x43], v[0x48], v[0x49]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x44], v[0x4a], v[0x4b]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x45], v[0x4c], v[0x4d]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0xe0:0xef], v[0x6a:0x6d], v[0x76:0x79], v[0xe0:0xef]
	;;#ASMEND
	v_add_u32_e32 v18, s1, v13
	;;#ASMSTART
	ds_read_b32 v[0x7e], v18 offset:0

	;;#ASMEND
	s_add_i32 s1, s23, 0xc0
	v_add_u32_e32 v18, s1, v13
	;;#ASMSTART
	ds_read_b32 v[0x7f], v18 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v9, v[0x42:0x43], offset:0x400
	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v9, v[0x44:0x45], offset:0x600
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0xf0:0xff], v[0x6a:0x6d], v[0x7a:0x7d], v[0xf0:0xff]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[62], v[64]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[63], v[0x41]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[0x42], v[0x44]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[0x43], v[0x45]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(12)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0x80:0x8f], a[0x70:0x73], v[62:0x41], a[0x80:0x8f]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[20:21], v1 offset:0x4200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[22:23], v1 offset:0x5200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[24:25], v1 offset:0x6000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[26:27], v1 offset:0x7000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[28:29], v1 offset:0x6200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[30:31], v1 offset:0x7200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[32:33], v1 offset:0x8000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[34:35], v1 offset:0x9000
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0x90:0x9f], a[0x70:0x73], v[0x42:0x45], a[0x90:0x9f]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0xa0:0xaf], a[0x74:0x77], v[62:0x41], a[0xa0:0xaf]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[30:31], v4 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[32:33], v4 offset:0x200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[34:35], v4 offset:0x400
	;;#ASMEND
	s_lshl_b32 s13, s3, 4
	;;#ASMSTART
	ds_read_b64_tr_b16 v[36:37], v4 offset:0x600
	;;#ASMEND
	s_add_i32 s0, s0, s13
	;;#ASMSTART
	ds_read_b64_tr_b16 v[38:39], v4 offset:0x800
	;;#ASMEND
	s_ashr_i32 s1, s0, 31
	;;#ASMSTART
	ds_read_b64_tr_b16 v[40:41], v4 offset:0xa00
	;;#ASMEND
	s_lshl_b64 s[8:9], s[0:1], 1
	;;#ASMSTART
	ds_read_b64_tr_b16 v[42:43], v4 offset:0xc00
	;;#ASMEND
	s_add_u32 s8, s6, s8
	;;#ASMSTART
	ds_read_b64_tr_b16 v[44:45], v4 offset:0xe00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0xb0:0xbf], a[0x74:0x77], v[0x42:0x45], a[0xb0:0xbf]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x7e], 0x3fb8aa3b, v[0x7e]
	;;#ASMEND
	s_addc_u32 s9, s7, s9
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x6e], v[0x6e], v[0x6f]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x6f], v[0x70], v[0x71]
	;;#ASMEND
	;;#ASMSTART
	buffer_atomic_pk_add_bf16 v[0x6e], v5, s[8:11], 0 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_atomic_pk_add_bf16 v[0x6f], v6, s[8:11], 0 offen
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0xc0:0xcf], a[0x78:0x7b], v[62:0x41], a[0xc0:0xcf]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[46:47], v4 offset:0x1000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[48:49], v4 offset:0x1200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[50:51], v4 offset:0x1400
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[52:53], v4 offset:0x1600
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[36:37], v1 offset:0x8200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[38:39], v1 offset:0x9200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[40:41], v1 offset:0xa000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[42:43], v1 offset:0xb000
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0xd0:0xdf], a[0x78:0x7b], v[0x42:0x45], a[0xd0:0xdf]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x72], v[0x72], v[0x73]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x73], v[0x74], v[0x75]
	;;#ASMEND
	;;#ASMSTART
	buffer_atomic_pk_add_bf16 v[0x72], v7, s[8:11], 0 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_atomic_pk_add_bf16 v[0x73], v8, s[8:11], 0 offen
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0xe0:0xef], a[0x7c:0x7f], v[62:0x41], a[0xe0:0xef]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[54:55], v4 offset:0x1800
	;;#ASMEND
	s_add_i32 s8, s67, 0x60
	;;#ASMSTART
	ds_read_b64_tr_b16 v[56:57], v4 offset:0x1a00
	;;#ASMEND
	s_mul_i32 s8, s8, s26
	;;#ASMSTART
	ds_read_b64_tr_b16 v[58:59], v4 offset:0x1c00
	;;#ASMEND
	s_add_i32 s8, s8, s61
	;;#ASMSTART
	ds_read_b64_tr_b16 v[60:61], v4 offset:0x1e00
	;;#ASMEND
	s_mul_i32 s8, s8, s60
	;;#ASMSTART
	ds_read_b64_tr_b16 a[44:45], v1 offset:0xa200
	;;#ASMEND
	s_add_i32 s1, s62, 0x6000
	s_ashr_i32 s9, s8, 31
	;;#ASMSTART
	ds_read_b64_tr_b16 a[46:47], v1 offset:0xb200
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0xf0:0xff], a[0x7c:0x7f], v[0x42:0x45], a[0xf0:0xff]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(4) lgkmcnt(6)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], a[0:3], v[30:33], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[62:63], v1 offset:0xc000
	;;#ASMEND
	s_lshl_b64 s[8:9], s[8:9], 1
	v_add_u32_e32 v17, s1, v17
	;;#ASMSTART
	ds_read_b64_tr_b16 v[64:0x41], v1 offset:0xd000
	;;#ASMEND
	s_add_u32 s44, s52, s8
	v_readfirstlane_b32 s1, v17
	v_add_u32_e32 v17, 0x1000, v17
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x42:0x43], v1 offset:0xc200
	;;#ASMEND
	s_addc_u32 s45, s53, s9
	s_mov_b32 m0, s1
	v_readfirstlane_b32 s1, v17
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x44:0x45], v1 offset:0xd200
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], a[8:11], v[34:37], v[0x6e:0x71]
	;;#ASMEND
	buffer_load_dwordx4 v14, s[44:47], 0 offen lds
	s_mov_b32 m0, s1
	s_add_i32 s1, s27, 0x3000
	buffer_load_dwordx4 v15, s[44:47], 0 offen lds
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], a[16:19], v[38:41], v[0x6e:0x71]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x46:0x47], v1 offset:0xe000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x48:0x49], v1 offset:0xf000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x4a:0x4b], v1 offset:0xe200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x4c:0x4d], v1 offset:0xf200
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], a[24:27], v[42:45], v[0x6e:0x71]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], a[32:35], v[46:49], v[0x6e:0x71]
	;;#ASMEND
	v_add_u32_e32 v17, s1, v11
	;;#ASMSTART
	ds_read_b128 a[0x70:0x73], v17 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[0x74:0x77], v17 offset:0x400
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], a[40:43], v[50:53], v[0x6e:0x71]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], v[62:0x41], v[54:57], v[0x6e:0x71]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[0x78:0x7b], v17 offset:0x800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[0x7c:0x7f], v17 offset:0xc00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], v[0x46:0x49], v[58:61], v[0x6e:0x71]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], a[4:7], v[30:33], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[0:3], v10 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[4:7], v10 offset:0x400
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], a[12:15], v[34:37], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], a[20:23], v[38:41], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[8:11], v10 offset:0x800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[12:15], v10 offset:0xc00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], a[28:31], v[42:45], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x6e], 0x3db504f3, v[0x6e]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x6f], 0x3db504f3, v[0x6f]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], a[36:39], v[46:49], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[16:19], v10 offset:0x1000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[20:23], v10 offset:0x1400
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], a[44:47], v[50:53], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x70], 0x3db504f3, v[0x70]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x71], 0x3db504f3, v[0x71]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(10)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], v[0x42:0x45], v[54:57], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[24:27], v10 offset:0x1800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[28:31], v10 offset:0x1c00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], v[0x4a:0x4d], v[58:61], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(2)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[46:49], a[0x70:0x73], a[0:3], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[32:35], v10 offset:0x2000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[36:39], v10 offset:0x2400
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[46:49], a[0x74:0x77], a[4:7], v[46:49]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[46:49], a[0x78:0x7b], a[8:11], v[46:49]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[40:43], v10 offset:0x2800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[44:47], v10 offset:0x2c00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[46:49], a[0x7c:0x7f], a[12:15], v[46:49]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x72], 0x3db504f3, v[0x72]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x73], 0x3db504f3, v[0x73]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[50:53], a[0x70:0x73], a[16:19], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[62:0x41], v10 offset:0x3000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[0x42:0x45], v10 offset:0x3400
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[50:53], a[0x74:0x77], a[20:23], v[50:53]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x74], 0x3db504f3, v[0x74]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x75], 0x3db504f3, v[0x75]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[50:53], a[0x78:0x7b], a[24:27], v[50:53]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[0x46:0x49], v10 offset:0x3800
	;;#ASMEND
	s_add_i32 s8, s62, 0x3000
	;;#ASMSTART
	ds_read_b128 v[0x4a:0x4d], v10 offset:0x3c00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[50:53], a[0x7c:0x7f], a[28:31], v[50:53]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[46], 0x3e0293ee, v[46]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[47], 0x3e0293ee, v[47]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[48], 0x3e0293ee, v[48]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[49], 0x3e0293ee, v[49]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(6)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[54:57], a[0x70:0x73], a[32:35], 0
	;;#ASMEND
	v_add_u32_e32 v17, s8, v11
	;;#ASMSTART
	ds_read_b128 v[0x4e:0x51], v17 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[0x52:0x55], v17 offset:0x400
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[54:57], a[0x74:0x77], a[36:39], v[54:57]
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[46], v[0x7e], v[46] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[47], v[0x7e], v[47] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[48], v[0x7e], v[48] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[49], v[0x7e], v[49] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(6)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[54:57], a[0x78:0x7b], a[40:43], v[54:57]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[0x56:0x59], v17 offset:0x800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[0x5a:0x5d], v17 offset:0xc00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[54:57], a[0x7c:0x7f], a[44:47], v[54:57]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[50], 0x3e0293ee, v[50]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[51], 0x3e0293ee, v[51]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[52], 0x3e0293ee, v[52]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[53], 0x3e0293ee, v[53]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(6)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[58:61], a[0x70:0x73], v[62:0x41], 0
	;;#ASMEND
	v_add_u32_e32 v17, s8, v12
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x5e:0x5f], v17 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x60:0x61], v17 offset:0x100
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x62:0x63], v17 offset:0x400
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x64:0x65], v17 offset:0x500
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[58:61], a[0x74:0x77], v[0x42:0x45], v[58:61]
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[50], v[0x7e], v[50] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[51], v[0x7e], v[51] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[52], v[0x7e], v[52] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[53], v[0x7e], v[53] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[58:61], a[0x78:0x7b], v[0x46:0x49], v[58:61]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x66:0x67], v17 offset:0x800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x68:0x69], v17 offset:0x900
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x6a:0x6b], v17 offset:0xc00
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x6c:0x6d], v17 offset:0xd00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[58:61], a[0x7c:0x7f], v[0x4a:0x4d], v[58:61]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[54], 0x3e0293ee, v[54]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[55], 0x3e0293ee, v[55]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[56], 0x3e0293ee, v[56]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[57], 0x3e0293ee, v[57]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[62:0x41], v[0x4e:0x51], a[48:51], 0
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[54], v[0x7e], v[54] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[55], v[0x7e], v[55] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[56], v[0x7e], v[56] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[57], v[0x7e], v[57] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[62:0x41], v[0x52:0x55], a[52:55], v[62:0x41]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[46], v[46]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[47], v[47]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[48], v[48]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[49], v[49]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[62:0x41], v[0x56:0x59], a[56:59], v[62:0x41]
	;;#ASMEND
	v_add_u32_e32 v17, s1, v12
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x70:0x71], v17 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x72:0x73], v17 offset:0x100
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[62:0x41], v[0x5a:0x5d], a[60:63], v[62:0x41]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[50], v[50]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[51], v[51]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[52], v[52]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[53], v[53]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x42:0x45], v[0x4e:0x51], a[64:0x43], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x74:0x75], v17 offset:0x400
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x76:0x77], v17 offset:0x500
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x42:0x45], v[0x52:0x55], a[0x44:0x47], v[0x42:0x45]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[58], 0x3e0293ee, v[58]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[59], 0x3e0293ee, v[59]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[60], 0x3e0293ee, v[60]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[61], 0x3e0293ee, v[61]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x42:0x45], v[0x56:0x59], a[0x48:0x4b], v[0x42:0x45]
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[58], v[0x7e], v[58] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[59], v[0x7e], v[59] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[60], v[0x7e], v[60] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[61], v[0x7e], v[61] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x42:0x45], v[0x5a:0x5d], a[0x4c:0x4f], v[0x42:0x45]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x76], v[46], v[47]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x77], v[48], v[49]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x46:0x49], v[0x4e:0x51], a[0x50:0x53], 0
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[54], v[54]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[55], v[55]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[56], v[56]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[57], v[57]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x46:0x49], v[0x52:0x55], a[0x54:0x57], v[0x46:0x49]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x78], v[50], v[51]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x79], v[52], v[53]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x46:0x49], v[0x56:0x59], a[0x58:0x5b], v[0x46:0x49]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x78:0x79], v17 offset:0x800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x7a:0x7b], v17 offset:0x900
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x46:0x49], v[0x5a:0x5d], a[0x5c:0x5f], v[0x46:0x49]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[58], v[58]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[59], v[59]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[60], v[60]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[61], v[61]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x4a:0x4d], v[0x4e:0x51], a[0x60:0x63], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x7c:0x7d], v17 offset:0xc00
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x7e:0x7f], v17 offset:0xd00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x4a:0x4d], v[0x52:0x55], a[0x64:0x67], v[0x4a:0x4d]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x7a], v[54], v[55]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x7b], v[56], v[57]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x7c], v[58], v[59]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x7d], v[60], v[61]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x4a:0x4d], v[0x56:0x59], a[0x68:0x6b], v[0x4a:0x4d]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[0x76], v[0x78]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[0x77], v[0x79]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[0x7a], v[0x7c]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[0x7b], v[0x7d]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x4a:0x4d], v[0x5a:0x5d], a[0x6c:0x6f], v[0x4a:0x4d]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0x80:0x8f], v[0x5e:0x61], v[0x76:0x79], v[0x80:0x8f]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0:1], v1 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[2:3], v1 offset:0x1000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[4:5], v1 offset:0x200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[6:7], v1 offset:0x1200
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0x90:0x9f], v[0x5e:0x61], v[0x7a:0x7d], v[0x90:0x9f]
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[62], v[0x7f], v[62] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[63], v[0x7f], v[63] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[64], v[0x7f], v[64] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x41], v[0x7f], v[0x41] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x42], v[0x7f], v[0x42] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x43], v[0x7f], v[0x43] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x44], v[0x7f], v[0x44] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x45], v[0x7f], v[0x45] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0xa0:0xaf], v[0x62:0x65], v[0x76:0x79], v[0xa0:0xaf]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[8:9], v1 offset:0x2000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[10:11], v1 offset:0x3000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[12:13], v1 offset:0x2200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[14:15], v1 offset:0x3200
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0xb0:0xbf], v[0x62:0x65], v[0x7a:0x7d], v[0xb0:0xbf]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[62], v[46], v[62]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[63], v[47], v[63]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[64], v[48], v[64]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x41], v[49], v[0x41]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x42], v[50], v[0x42]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x43], v[51], v[0x43]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x44], v[52], v[0x44]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x45], v[53], v[0x45]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[62], v[62], v[63]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[63], v[64], v[0x41]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[64], v[0x42], v[0x43]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x41], v[0x44], v[0x45]
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x46], v[0x7f], v[0x46] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x47], v[0x7f], v[0x47] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x48], v[0x7f], v[0x48] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x49], v[0x7f], v[0x49] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0xc0:0xcf], v[0x66:0x69], v[0x76:0x79], v[0xc0:0xcf]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[16:17], v1 offset:0x4000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[18:19], v1 offset:0x5000
	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v9, v[62:63], offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v9, v[64:0x41], offset:0x200
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0xd0:0xdf], v[0x66:0x69], v[0x7a:0x7d], v[0xd0:0xdf]
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x4a], v[0x7f], v[0x4a] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x4b], v[0x7f], v[0x4b] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x4c], v[0x7f], v[0x4c] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x4d], v[0x7f], v[0x4d] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x46], v[54], v[0x46]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x47], v[55], v[0x47]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x48], v[56], v[0x48]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x49], v[57], v[0x49]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x4a], v[58], v[0x4a]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x4b], v[59], v[0x4b]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x4c], v[60], v[0x4c]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x4d], v[61], v[0x4d]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x42], v[0x46], v[0x47]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x43], v[0x48], v[0x49]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x44], v[0x4a], v[0x4b]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x45], v[0x4c], v[0x4d]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0xe0:0xef], v[0x6a:0x6d], v[0x76:0x79], v[0xe0:0xef]
	;;#ASMEND
	v_add_u32_e32 v17, s25, v13
	;;#ASMSTART
	ds_read_b32 v[0x7e], v17 offset:0

	;;#ASMEND
	v_add_u32_e32 v17, s39, v13
	;;#ASMSTART
	ds_read_b32 v[0x7f], v17 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v9, v[0x42:0x43], offset:0x400
	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v9, v[0x44:0x45], offset:0x600
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0xf0:0xff], v[0x6a:0x6d], v[0x7a:0x7d], v[0xf0:0xff]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[62], v[64]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[63], v[0x41]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[0x42], v[0x44]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[0x43], v[0x45]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(12)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0x80:0x8f], a[0x70:0x73], v[62:0x41], a[0x80:0x8f]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[20:21], v1 offset:0x4200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[22:23], v1 offset:0x5200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[24:25], v1 offset:0x6000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[26:27], v1 offset:0x7000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[28:29], v1 offset:0x6200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[30:31], v1 offset:0x7200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[32:33], v1 offset:0x8000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[34:35], v1 offset:0x9000
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0x90:0x9f], a[0x70:0x73], v[0x42:0x45], a[0x90:0x9f]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0xa0:0xaf], a[0x74:0x77], v[62:0x41], a[0xa0:0xaf]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[30:31], v4 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[32:33], v4 offset:0x200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[34:35], v4 offset:0x400
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[36:37], v4 offset:0x600
	;;#ASMEND
	s_add_i32 s0, s0, s13
	;;#ASMSTART
	ds_read_b64_tr_b16 v[38:39], v4 offset:0x800
	;;#ASMEND
	s_ashr_i32 s1, s0, 31
	;;#ASMSTART
	ds_read_b64_tr_b16 v[40:41], v4 offset:0xa00
	;;#ASMEND
	s_lshl_b64 s[0:1], s[0:1], 1
	;;#ASMSTART
	ds_read_b64_tr_b16 v[42:43], v4 offset:0xc00
	;;#ASMEND
	s_add_u32 s8, s6, s0
	;;#ASMSTART
	ds_read_b64_tr_b16 v[44:45], v4 offset:0xe00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0xb0:0xbf], a[0x74:0x77], v[0x42:0x45], a[0xb0:0xbf]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x7e], 0x3fb8aa3b, v[0x7e]
	;;#ASMEND
	s_addc_u32 s9, s7, s1
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x6e], v[0x6e], v[0x6f]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x6f], v[0x70], v[0x71]
	;;#ASMEND
	;;#ASMSTART
	buffer_atomic_pk_add_bf16 v[0x6e], v5, s[8:11], 0 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_atomic_pk_add_bf16 v[0x6f], v6, s[8:11], 0 offen
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0xc0:0xcf], a[0x78:0x7b], v[62:0x41], a[0xc0:0xcf]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[46:47], v4 offset:0x1000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[48:49], v4 offset:0x1200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[50:51], v4 offset:0x1400
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[52:53], v4 offset:0x1600
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[36:37], v1 offset:0x8200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[38:39], v1 offset:0x9200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[40:41], v1 offset:0xa000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[42:43], v1 offset:0xb000
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0xd0:0xdf], a[0x78:0x7b], v[0x42:0x45], a[0xd0:0xdf]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x72], v[0x72], v[0x73]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x73], v[0x74], v[0x75]
	;;#ASMEND
	;;#ASMSTART
	buffer_atomic_pk_add_bf16 v[0x72], v7, s[8:11], 0 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_atomic_pk_add_bf16 v[0x73], v8, s[8:11], 0 offen
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0xe0:0xef], a[0x7c:0x7f], v[62:0x41], a[0xe0:0xef]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[54:55], v4 offset:0x1800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[56:57], v4 offset:0x1a00
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[58:59], v4 offset:0x1c00
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[60:61], v4 offset:0x1e00
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[44:45], v1 offset:0xa200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[46:47], v1 offset:0xb200
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0xf0:0xff], a[0x7c:0x7f], v[0x42:0x45], a[0xf0:0xff]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(4) lgkmcnt(6)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], a[0:3], v[30:33], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[62:63], v1 offset:0xc000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[64:0x41], v1 offset:0xd000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x42:0x43], v1 offset:0xc200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x44:0x45], v1 offset:0xd200
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], a[8:11], v[34:37], v[0x6e:0x71]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], a[16:19], v[38:41], v[0x6e:0x71]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x46:0x47], v1 offset:0xe000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x48:0x49], v1 offset:0xf000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x4a:0x4b], v1 offset:0xe200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x4c:0x4d], v1 offset:0xf200
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], a[24:27], v[42:45], v[0x6e:0x71]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], a[32:35], v[46:49], v[0x6e:0x71]
	;;#ASMEND
	v_add_u32_e32 v17, s19, v11
	;;#ASMSTART
	ds_read_b128 a[0x70:0x73], v17 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[0x74:0x77], v17 offset:0x400
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], a[40:43], v[50:53], v[0x6e:0x71]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], v[62:0x41], v[54:57], v[0x6e:0x71]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[0x78:0x7b], v17 offset:0x800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[0x7c:0x7f], v17 offset:0xc00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], v[0x46:0x49], v[58:61], v[0x6e:0x71]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], a[4:7], v[30:33], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[0:3], v10 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[4:7], v10 offset:0x400
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], a[12:15], v[34:37], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], a[20:23], v[38:41], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[8:11], v10 offset:0x800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[12:15], v10 offset:0xc00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], a[28:31], v[42:45], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x6e], 0x3db504f3, v[0x6e]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x6f], 0x3db504f3, v[0x6f]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], a[36:39], v[46:49], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[16:19], v10 offset:0x1000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[20:23], v10 offset:0x1400
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], a[44:47], v[50:53], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x70], 0x3db504f3, v[0x70]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x71], 0x3db504f3, v[0x71]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(10)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], v[0x42:0x45], v[54:57], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[24:27], v10 offset:0x1800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[28:31], v10 offset:0x1c00
	;;#ASMEND
	s_movk_i32 s17, 0x80
	s_add_i32 s19, s66, 32
	s_add_i32 s25, s67, 32
	v_lshlrev_b32_e32 v2, 1, v2
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], v[0x4a:0x4d], v[58:61], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(2)
	;;#ASMEND
.LBB0_1:                                ; =>This Inner Loop Header: Depth=1
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
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[46:49], a[0x70:0x73], a[0:3], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[32:35], v10 offset:0x2000
	;;#ASMEND
	s_addc_u32 s41, s55, s9
	s_mov_b32 m0, s69
	;;#ASMSTART
	ds_read_b128 a[36:39], v10 offset:0x2400
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[46:49], a[0x74:0x77], a[4:7], v[46:49]
	;;#ASMEND
	s_lshl_b32 s8, s35, 14
	buffer_load_dword v16, s[40:43], 0 offen lds
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[46:49], a[0x78:0x7b], a[8:11], v[46:49]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[40:43], v10 offset:0x2800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[44:47], v10 offset:0x2c00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[46:49], a[0x7c:0x7f], a[12:15], v[46:49]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x72], 0x3db504f3, v[0x72]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x73], 0x3db504f3, v[0x73]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[50:53], a[0x70:0x73], a[16:19], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[62:0x41], v10 offset:0x3000
	;;#ASMEND
	v_add_u32_e32 v18, s69, v13
	s_add_i32 s69, s62, s8
	;;#ASMSTART
	ds_read_b128 v[0x42:0x45], v10 offset:0x3400
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[50:53], a[0x74:0x77], a[20:23], v[50:53]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x74], 0x3db504f3, v[0x74]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x75], 0x3db504f3, v[0x75]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[50:53], a[0x78:0x7b], a[24:27], v[50:53]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[0x46:0x49], v10 offset:0x3800
	;;#ASMEND
	v_add_u32_e32 v21, s69, v11
	;;#ASMSTART
	ds_read_b128 v[0x4a:0x4d], v10 offset:0x3c00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[50:53], a[0x7c:0x7f], a[28:31], v[50:53]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[46], 0x3e0293ee, v[46]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[47], 0x3e0293ee, v[47]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[48], 0x3e0293ee, v[48]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[49], 0x3e0293ee, v[49]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(6)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[54:57], a[0x70:0x73], a[32:35], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[0x4e:0x51], v21 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[0x52:0x55], v21 offset:0x400
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[54:57], a[0x74:0x77], a[36:39], v[54:57]
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[46], v[0x7e], v[46] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[47], v[0x7e], v[47] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[48], v[0x7e], v[48] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[49], v[0x7e], v[49] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(6)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[54:57], a[0x78:0x7b], a[40:43], v[54:57]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[0x56:0x59], v21 offset:0x800
	;;#ASMEND
	v_add_u32_e32 v20, s69, v12
	;;#ASMSTART
	ds_read_b128 v[0x5a:0x5d], v21 offset:0xc00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[54:57], a[0x7c:0x7f], a[44:47], v[54:57]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[50], 0x3e0293ee, v[50]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[51], 0x3e0293ee, v[51]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[52], 0x3e0293ee, v[52]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[53], 0x3e0293ee, v[53]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(6)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[58:61], a[0x70:0x73], v[62:0x41], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x5e:0x5f], v20 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x60:0x61], v20 offset:0x100
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x62:0x63], v20 offset:0x400
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x64:0x65], v20 offset:0x500
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[58:61], a[0x74:0x77], v[0x42:0x45], v[58:61]
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[50], v[0x7e], v[50] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[51], v[0x7e], v[51] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[52], v[0x7e], v[52] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[53], v[0x7e], v[53] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[58:61], a[0x78:0x7b], v[0x46:0x49], v[58:61]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x66:0x67], v20 offset:0x800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x68:0x69], v20 offset:0x900
	;;#ASMEND
	s_add_i32 s70, s27, s8
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x6a:0x6b], v20 offset:0xc00
	;;#ASMEND
	v_add_u32_e32 v19, s70, v12
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x6c:0x6d], v20 offset:0xd00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[58:61], a[0x7c:0x7f], v[0x4a:0x4d], v[58:61]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[54], 0x3e0293ee, v[54]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[55], 0x3e0293ee, v[55]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[56], 0x3e0293ee, v[56]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[57], 0x3e0293ee, v[57]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[62:0x41], v[0x4e:0x51], a[48:51], 0
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[54], v[0x7e], v[54] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[55], v[0x7e], v[55] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[56], v[0x7e], v[56] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[57], v[0x7e], v[57] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[62:0x41], v[0x52:0x55], a[52:55], v[62:0x41]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[46], v[46]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[47], v[47]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[48], v[48]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[49], v[49]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[62:0x41], v[0x56:0x59], a[56:59], v[62:0x41]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x70:0x71], v19 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x72:0x73], v19 offset:0x100
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[62:0x41], v[0x5a:0x5d], a[60:63], v[62:0x41]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[50], v[50]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[51], v[51]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[52], v[52]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[53], v[53]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x42:0x45], v[0x4e:0x51], a[64:0x43], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x74:0x75], v19 offset:0x400
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x76:0x77], v19 offset:0x500
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x42:0x45], v[0x52:0x55], a[0x44:0x47], v[0x42:0x45]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[58], 0x3e0293ee, v[58]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[59], 0x3e0293ee, v[59]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[60], 0x3e0293ee, v[60]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[61], 0x3e0293ee, v[61]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x42:0x45], v[0x56:0x59], a[0x48:0x4b], v[0x42:0x45]
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[58], v[0x7e], v[58] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[59], v[0x7e], v[59] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[60], v[0x7e], v[60] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[61], v[0x7e], v[61] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x42:0x45], v[0x5a:0x5d], a[0x4c:0x4f], v[0x42:0x45]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x76], v[46], v[47]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x77], v[48], v[49]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x46:0x49], v[0x4e:0x51], a[0x50:0x53], 0
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[54], v[54]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[55], v[55]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[56], v[56]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[57], v[57]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x46:0x49], v[0x52:0x55], a[0x54:0x57], v[0x46:0x49]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x78], v[50], v[51]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x79], v[52], v[53]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x46:0x49], v[0x56:0x59], a[0x58:0x5b], v[0x46:0x49]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x78:0x79], v19 offset:0x800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x7a:0x7b], v19 offset:0x900
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x46:0x49], v[0x5a:0x5d], a[0x5c:0x5f], v[0x46:0x49]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[58], v[58]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[59], v[59]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[60], v[60]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[61], v[61]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x4a:0x4d], v[0x4e:0x51], a[0x60:0x63], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x7c:0x7d], v19 offset:0xc00
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x7e:0x7f], v19 offset:0xd00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x4a:0x4d], v[0x52:0x55], a[0x64:0x67], v[0x4a:0x4d]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x7a], v[54], v[55]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x7b], v[56], v[57]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x7c], v[58], v[59]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x7d], v[60], v[61]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x4a:0x4d], v[0x56:0x59], a[0x68:0x6b], v[0x4a:0x4d]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[0x76], v[0x78]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[0x77], v[0x79]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[0x7a], v[0x7c]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[0x7b], v[0x7d]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x4a:0x4d], v[0x5a:0x5d], a[0x6c:0x6f], v[0x4a:0x4d]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0x80:0x8f], v[0x5e:0x61], v[0x76:0x79], v[0x80:0x8f]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0:1], v1 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[2:3], v1 offset:0x1000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[4:5], v1 offset:0x200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[6:7], v1 offset:0x1200
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0x90:0x9f], v[0x5e:0x61], v[0x7a:0x7d], v[0x90:0x9f]
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[62], v[0x7f], v[62] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[63], v[0x7f], v[63] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[64], v[0x7f], v[64] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x41], v[0x7f], v[0x41] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x42], v[0x7f], v[0x42] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x43], v[0x7f], v[0x43] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x44], v[0x7f], v[0x44] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x45], v[0x7f], v[0x45] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0xa0:0xaf], v[0x62:0x65], v[0x76:0x79], v[0xa0:0xaf]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[8:9], v1 offset:0x2000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[10:11], v1 offset:0x3000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[12:13], v1 offset:0x2200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[14:15], v1 offset:0x3200
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0xb0:0xbf], v[0x62:0x65], v[0x7a:0x7d], v[0xb0:0xbf]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[62], v[46], v[62]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[63], v[47], v[63]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[64], v[48], v[64]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x41], v[49], v[0x41]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x42], v[50], v[0x42]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x43], v[51], v[0x43]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x44], v[52], v[0x44]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x45], v[53], v[0x45]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[62], v[62], v[63]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[63], v[64], v[0x41]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[64], v[0x42], v[0x43]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x41], v[0x44], v[0x45]
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x46], v[0x7f], v[0x46] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x47], v[0x7f], v[0x47] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x48], v[0x7f], v[0x48] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x49], v[0x7f], v[0x49] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0xc0:0xcf], v[0x66:0x69], v[0x76:0x79], v[0xc0:0xcf]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[16:17], v1 offset:0x4000
	;;#ASMEND
	s_lshl_b32 s9, s35, 8
	;;#ASMSTART
	ds_read_b64_tr_b16 a[18:19], v1 offset:0x5000
	;;#ASMEND
	s_add_i32 s74, s5, s9
	;;#ASMSTART
	ds_write_b64 v9, v[62:63], offset:0
	;;#ASMEND
	s_add_i32 s75, s23, s9
	v_add3_u32 v24, s74, 64, v13
	;;#ASMSTART
	ds_write_b64 v9, v[64:0x41], offset:0x200
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0xd0:0xdf], v[0x66:0x69], v[0x7a:0x7d], v[0xd0:0xdf]
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x4a], v[0x7f], v[0x4a] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x4b], v[0x7f], v[0x4b] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x4c], v[0x7f], v[0x4c] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x4d], v[0x7f], v[0x4d] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x46], v[54], v[0x46]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x47], v[55], v[0x47]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x48], v[56], v[0x48]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x49], v[57], v[0x49]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x4a], v[58], v[0x4a]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x4b], v[59], v[0x4b]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x4c], v[60], v[0x4c]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x4d], v[61], v[0x4d]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x42], v[0x46], v[0x47]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x43], v[0x48], v[0x49]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x44], v[0x4a], v[0x4b]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x45], v[0x4c], v[0x4d]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0xe0:0xef], v[0x6a:0x6d], v[0x76:0x79], v[0xe0:0xef]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b32 v[0x7e], v24 offset:0

	;;#ASMEND
	v_add3_u32 v22, s75, 64, v13
	;;#ASMSTART
	ds_read_b32 v[0x7f], v22 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v9, v[0x42:0x43], offset:0x400
	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v9, v[0x44:0x45], offset:0x600
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0xf0:0xff], v[0x6a:0x6d], v[0x7a:0x7d], v[0xf0:0xff]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[62], v[64]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[63], v[0x41]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[0x42], v[0x44]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[0x43], v[0x45]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(12)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0x80:0x8f], a[0x70:0x73], v[62:0x41], a[0x80:0x8f]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[20:21], v1 offset:0x4200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[22:23], v1 offset:0x5200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[24:25], v1 offset:0x6000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[26:27], v1 offset:0x7000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[28:29], v1 offset:0x6200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[30:31], v1 offset:0x7200
	;;#ASMEND
	s_add_i32 s29, s15, s44
	;;#ASMSTART
	ds_read_b64_tr_b16 a[32:33], v1 offset:0x8000
	;;#ASMEND
	s_lshl_b32 s28, s45, 6
	s_mul_i32 s8, s29, s38
	;;#ASMSTART
	ds_read_b64_tr_b16 a[34:35], v1 offset:0x9000
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0x90:0x9f], a[0x70:0x73], v[0x42:0x45], a[0x90:0x9f]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0xa0:0xaf], a[0x74:0x77], v[62:0x41], a[0xa0:0xaf]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[30:31], v4 offset:0
	;;#ASMEND
	s_add_i32 s8, s8, s28
	;;#ASMSTART
	ds_read_b64_tr_b16 v[32:33], v4 offset:0x200
	;;#ASMEND
	s_add_i32 s8, s8, 48
	;;#ASMSTART
	ds_read_b64_tr_b16 v[34:35], v4 offset:0x400
	;;#ASMEND
	s_mul_i32 s8, s8, s3
	;;#ASMSTART
	ds_read_b64_tr_b16 v[36:37], v4 offset:0x600
	;;#ASMEND
	s_ashr_i32 s9, s8, 31
	;;#ASMSTART
	ds_read_b64_tr_b16 v[38:39], v4 offset:0x800
	;;#ASMEND
	s_lshl_b64 s[8:9], s[8:9], 1
	;;#ASMSTART
	ds_read_b64_tr_b16 v[40:41], v4 offset:0xa00
	;;#ASMEND
	s_add_u32 s8, s6, s8
	;;#ASMSTART
	ds_read_b64_tr_b16 v[42:43], v4 offset:0xc00
	;;#ASMEND
	s_addc_u32 s9, s7, s9
	;;#ASMSTART
	ds_read_b64_tr_b16 v[44:45], v4 offset:0xe00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0xb0:0xbf], a[0x74:0x77], v[0x42:0x45], a[0xb0:0xbf]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x7e], 0x3fb8aa3b, v[0x7e]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x6e], v[0x6e], v[0x6f]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x6f], v[0x70], v[0x71]
	;;#ASMEND
	;;#ASMSTART
	buffer_atomic_pk_add_bf16 v[0x6e], v5, s[8:11], 0 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_atomic_pk_add_bf16 v[0x6f], v6, s[8:11], 0 offen
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0xc0:0xcf], a[0x78:0x7b], v[62:0x41], a[0xc0:0xcf]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[46:47], v4 offset:0x1000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[48:49], v4 offset:0x1200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[50:51], v4 offset:0x1400
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[52:53], v4 offset:0x1600
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[36:37], v1 offset:0x8200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[38:39], v1 offset:0x9200
	;;#ASMEND
	s_add_i32 s71, s62, s0
	s_add_i32 s0, s36, s67
	;;#ASMSTART
	ds_read_b64_tr_b16 a[40:41], v1 offset:0xa000
	;;#ASMEND
	s_mul_i32 s0, s0, s26
	;;#ASMSTART
	ds_read_b64_tr_b16 a[42:43], v1 offset:0xb000
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0xd0:0xdf], a[0x78:0x7b], v[0x42:0x45], a[0xd0:0xdf]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x72], v[0x72], v[0x73]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x73], v[0x74], v[0x75]
	;;#ASMEND
	;;#ASMSTART
	buffer_atomic_pk_add_bf16 v[0x72], v7, s[8:11], 0 offen
	;;#ASMEND
	s_add_i32 s0, s0, s39
	;;#ASMSTART
	buffer_atomic_pk_add_bf16 v[0x73], v8, s[8:11], 0 offen
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0xe0:0xef], a[0x7c:0x7f], v[62:0x41], a[0xe0:0xef]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[54:55], v4 offset:0x1800
	;;#ASMEND
	s_mul_i32 s40, s0, s60
	;;#ASMSTART
	ds_read_b64_tr_b16 v[56:57], v4 offset:0x1a00
	;;#ASMEND
	s_ashr_i32 s41, s40, 31
	;;#ASMSTART
	ds_read_b64_tr_b16 v[58:59], v4 offset:0x1c00
	;;#ASMEND
	s_lshl_b64 s[40:41], s[40:41], 1
	;;#ASMSTART
	ds_read_b64_tr_b16 v[60:61], v4 offset:0x1e00
	;;#ASMEND
	s_add_u32 s44, s52, s40
	;;#ASMSTART
	ds_read_b64_tr_b16 a[44:45], v1 offset:0xa200
	;;#ASMEND
	v_add_u32_e32 v21, s71, v2
	s_addc_u32 s45, s53, s41
	s_add_i32 s28, s39, s64
	;;#ASMSTART
	ds_read_b64_tr_b16 a[46:47], v1 offset:0xb200
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0xf0:0xff], a[0x7c:0x7f], v[0x42:0x45], a[0xf0:0xff]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(4) lgkmcnt(6)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], a[0:3], v[30:33], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[62:63], v1 offset:0xc000
	;;#ASMEND
	v_readfirstlane_b32 s29, v21
	v_add_u32_e32 v21, 0x1000, v21
	s_mul_i32 s28, s65, s28
	;;#ASMSTART
	ds_read_b64_tr_b16 v[64:0x41], v1 offset:0xd000
	;;#ASMEND
	s_mov_b32 s47, s31
	v_readfirstlane_b32 s0, v21
	s_add_i32 s40, s28, s36
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x42:0x43], v1 offset:0xc200
	;;#ASMEND
	s_mov_b32 m0, s29
	s_ashr_i32 s41, s40, 31
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x44:0x45], v1 offset:0xd200
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], a[8:11], v[34:37], v[0x6e:0x71]
	;;#ASMEND
	buffer_load_dwordx4 v14, s[44:47], 0 offen lds
	s_mov_b32 m0, s0
	s_add_i32 s78, s23, s72
	s_lshl_b64 s[40:41], s[40:41], 2
	buffer_load_dwordx4 v15, s[44:47], 0 offen lds
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], a[16:19], v[38:41], v[0x6e:0x71]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x46:0x47], v1 offset:0xe000
	;;#ASMEND
	s_add_u32 s40, s56, s40
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x48:0x49], v1 offset:0xf000
	;;#ASMEND
	s_addc_u32 s41, s57, s41
	s_add_i32 s28, s70, 0x1000
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x4a:0x4b], v1 offset:0xe200
	;;#ASMEND
	s_mov_b32 m0, s78
	v_add_u32_e32 v27, s28, v11
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x4c:0x4d], v1 offset:0xf200
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], a[24:27], v[42:45], v[0x6e:0x71]
	;;#ASMEND
	buffer_load_dword v16, s[40:43], 0 offen lds
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], a[32:35], v[46:49], v[0x6e:0x71]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[0x70:0x73], v27 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[0x74:0x77], v27 offset:0x400
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], a[40:43], v[50:53], v[0x6e:0x71]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], v[62:0x41], v[54:57], v[0x6e:0x71]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[0x78:0x7b], v27 offset:0x800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[0x7c:0x7f], v27 offset:0xc00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], v[0x46:0x49], v[58:61], v[0x6e:0x71]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], a[4:7], v[30:33], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[0:3], v10 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[4:7], v10 offset:0x400
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], a[12:15], v[34:37], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], a[20:23], v[38:41], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[8:11], v10 offset:0x800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[12:15], v10 offset:0xc00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], a[28:31], v[42:45], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x6e], 0x3db504f3, v[0x6e]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x6f], 0x3db504f3, v[0x6f]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], a[36:39], v[46:49], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[16:19], v10 offset:0x1000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[20:23], v10 offset:0x1400
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], a[44:47], v[50:53], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x70], 0x3db504f3, v[0x70]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x71], 0x3db504f3, v[0x71]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(10)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], v[0x42:0x45], v[54:57], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[24:27], v10 offset:0x1800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[28:31], v10 offset:0x1c00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], v[0x4a:0x4d], v[58:61], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(2)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[46:49], a[0x70:0x73], a[0:3], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[32:35], v10 offset:0x2000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[36:39], v10 offset:0x2400
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[46:49], a[0x74:0x77], a[4:7], v[46:49]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[46:49], a[0x78:0x7b], a[8:11], v[46:49]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[40:43], v10 offset:0x2800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[44:47], v10 offset:0x2c00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[46:49], a[0x7c:0x7f], a[12:15], v[46:49]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x72], 0x3db504f3, v[0x72]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x73], 0x3db504f3, v[0x73]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[50:53], a[0x70:0x73], a[16:19], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[62:0x41], v10 offset:0x3000
	;;#ASMEND
	s_add_i32 s72, s69, 0x1000
	;;#ASMSTART
	ds_read_b128 v[0x42:0x45], v10 offset:0x3400
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[50:53], a[0x74:0x77], a[20:23], v[50:53]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x74], 0x3db504f3, v[0x74]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x75], 0x3db504f3, v[0x75]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(6)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[50:53], a[0x78:0x7b], a[24:27], v[50:53]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[0x46:0x49], v10 offset:0x3800
	;;#ASMEND
	v_add_u32_e32 v25, s72, v11
	;;#ASMSTART
	ds_read_b128 v[0x4a:0x4d], v10 offset:0x3c00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[50:53], a[0x7c:0x7f], a[28:31], v[50:53]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[46], 0x3e0293ee, v[46]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[47], 0x3e0293ee, v[47]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[48], 0x3e0293ee, v[48]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[49], 0x3e0293ee, v[49]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(6)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[54:57], a[0x70:0x73], a[32:35], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[0x4e:0x51], v25 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[0x52:0x55], v25 offset:0x400
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[54:57], a[0x74:0x77], a[36:39], v[54:57]
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[46], v[0x7e], v[46] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[47], v[0x7e], v[47] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[48], v[0x7e], v[48] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[49], v[0x7e], v[49] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(6)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[54:57], a[0x78:0x7b], a[40:43], v[54:57]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[0x56:0x59], v25 offset:0x800
	;;#ASMEND
	v_add_u32_e32 v20, s72, v12
	;;#ASMSTART
	ds_read_b128 v[0x5a:0x5d], v25 offset:0xc00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[54:57], a[0x7c:0x7f], a[44:47], v[54:57]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[50], 0x3e0293ee, v[50]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[51], 0x3e0293ee, v[51]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[52], 0x3e0293ee, v[52]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[53], 0x3e0293ee, v[53]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(6)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[58:61], a[0x70:0x73], v[62:0x41], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x5e:0x5f], v20 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x60:0x61], v20 offset:0x100
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x62:0x63], v20 offset:0x400
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x64:0x65], v20 offset:0x500
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[58:61], a[0x74:0x77], v[0x42:0x45], v[58:61]
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[50], v[0x7e], v[50] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[51], v[0x7e], v[51] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[52], v[0x7e], v[52] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[53], v[0x7e], v[53] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[58:61], a[0x78:0x7b], v[0x46:0x49], v[58:61]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x66:0x67], v20 offset:0x800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x68:0x69], v20 offset:0x900
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x6a:0x6b], v20 offset:0xc00
	;;#ASMEND
	v_add_u32_e32 v26, s28, v12
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x6c:0x6d], v20 offset:0xd00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[58:61], a[0x7c:0x7f], v[0x4a:0x4d], v[58:61]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[54], 0x3e0293ee, v[54]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[55], 0x3e0293ee, v[55]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[56], 0x3e0293ee, v[56]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[57], 0x3e0293ee, v[57]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[62:0x41], v[0x4e:0x51], a[48:51], 0
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[54], v[0x7e], v[54] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[55], v[0x7e], v[55] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[56], v[0x7e], v[56] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[57], v[0x7e], v[57] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[62:0x41], v[0x52:0x55], a[52:55], v[62:0x41]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[46], v[46]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[47], v[47]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[48], v[48]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[49], v[49]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[62:0x41], v[0x56:0x59], a[56:59], v[62:0x41]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x70:0x71], v26 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x72:0x73], v26 offset:0x100
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[62:0x41], v[0x5a:0x5d], a[60:63], v[62:0x41]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[50], v[50]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[51], v[51]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[52], v[52]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[53], v[53]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x42:0x45], v[0x4e:0x51], a[64:0x43], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x74:0x75], v26 offset:0x400
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x76:0x77], v26 offset:0x500
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x42:0x45], v[0x52:0x55], a[0x44:0x47], v[0x42:0x45]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[58], 0x3e0293ee, v[58]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[59], 0x3e0293ee, v[59]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[60], 0x3e0293ee, v[60]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[61], 0x3e0293ee, v[61]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x42:0x45], v[0x56:0x59], a[0x48:0x4b], v[0x42:0x45]
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[58], v[0x7e], v[58] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[59], v[0x7e], v[59] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[60], v[0x7e], v[60] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[61], v[0x7e], v[61] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x42:0x45], v[0x5a:0x5d], a[0x4c:0x4f], v[0x42:0x45]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x76], v[46], v[47]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x77], v[48], v[49]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x46:0x49], v[0x4e:0x51], a[0x50:0x53], 0
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[54], v[54]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[55], v[55]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[56], v[56]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[57], v[57]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x46:0x49], v[0x52:0x55], a[0x54:0x57], v[0x46:0x49]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x78], v[50], v[51]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x79], v[52], v[53]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x46:0x49], v[0x56:0x59], a[0x58:0x5b], v[0x46:0x49]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x78:0x79], v26 offset:0x800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x7a:0x7b], v26 offset:0x900
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x46:0x49], v[0x5a:0x5d], a[0x5c:0x5f], v[0x46:0x49]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[58], v[58]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[59], v[59]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[60], v[60]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[61], v[61]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x4a:0x4d], v[0x4e:0x51], a[0x60:0x63], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x7c:0x7d], v26 offset:0xc00
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x7e:0x7f], v26 offset:0xd00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x4a:0x4d], v[0x52:0x55], a[0x64:0x67], v[0x4a:0x4d]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x7a], v[54], v[55]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x7b], v[56], v[57]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x7c], v[58], v[59]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x7d], v[60], v[61]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x4a:0x4d], v[0x56:0x59], a[0x68:0x6b], v[0x4a:0x4d]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[0x76], v[0x78]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[0x77], v[0x79]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[0x7a], v[0x7c]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[0x7b], v[0x7d]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x4a:0x4d], v[0x5a:0x5d], a[0x6c:0x6f], v[0x4a:0x4d]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0x80:0x8f], v[0x5e:0x61], v[0x76:0x79], v[0x80:0x8f]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0:1], v1 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[2:3], v1 offset:0x1000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[4:5], v1 offset:0x200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[6:7], v1 offset:0x1200
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0x90:0x9f], v[0x5e:0x61], v[0x7a:0x7d], v[0x90:0x9f]
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[62], v[0x7f], v[62] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[63], v[0x7f], v[63] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[64], v[0x7f], v[64] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x41], v[0x7f], v[0x41] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x42], v[0x7f], v[0x42] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x43], v[0x7f], v[0x43] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x44], v[0x7f], v[0x44] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x45], v[0x7f], v[0x45] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0xa0:0xaf], v[0x62:0x65], v[0x76:0x79], v[0xa0:0xaf]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[8:9], v1 offset:0x2000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[10:11], v1 offset:0x3000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[12:13], v1 offset:0x2200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[14:15], v1 offset:0x3200
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0xb0:0xbf], v[0x62:0x65], v[0x7a:0x7d], v[0xb0:0xbf]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[62], v[46], v[62]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[63], v[47], v[63]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[64], v[48], v[64]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x41], v[49], v[0x41]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x42], v[50], v[0x42]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x43], v[51], v[0x43]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x44], v[52], v[0x44]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x45], v[53], v[0x45]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[62], v[62], v[63]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[63], v[64], v[0x41]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[64], v[0x42], v[0x43]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x41], v[0x44], v[0x45]
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x46], v[0x7f], v[0x46] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x47], v[0x7f], v[0x47] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x48], v[0x7f], v[0x48] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x49], v[0x7f], v[0x49] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0xc0:0xcf], v[0x66:0x69], v[0x76:0x79], v[0xc0:0xcf]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[16:17], v1 offset:0x4000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[18:19], v1 offset:0x5000
	;;#ASMEND
	s_add_i32 s73, s74, 0x80
	;;#ASMSTART
	ds_write_b64 v9, v[62:63], offset:0
	;;#ASMEND
	s_add_i32 s76, s75, 0x80
	v_add_u32_e32 v21, s73, v13
	;;#ASMSTART
	ds_write_b64 v9, v[64:0x41], offset:0x200
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0xd0:0xdf], v[0x66:0x69], v[0x7a:0x7d], v[0xd0:0xdf]
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x4a], v[0x7f], v[0x4a] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x4b], v[0x7f], v[0x4b] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x4c], v[0x7f], v[0x4c] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x4d], v[0x7f], v[0x4d] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x46], v[54], v[0x46]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x47], v[55], v[0x47]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x48], v[56], v[0x48]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x49], v[57], v[0x49]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x4a], v[58], v[0x4a]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x4b], v[59], v[0x4b]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x4c], v[60], v[0x4c]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x4d], v[61], v[0x4d]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x42], v[0x46], v[0x47]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x43], v[0x48], v[0x49]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x44], v[0x4a], v[0x4b]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x45], v[0x4c], v[0x4d]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0xe0:0xef], v[0x6a:0x6d], v[0x76:0x79], v[0xe0:0xef]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b32 v[0x7e], v21 offset:0

	;;#ASMEND
	v_add_u32_e32 v23, s76, v13
	;;#ASMSTART
	ds_read_b32 v[0x7f], v23 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v9, v[0x42:0x43], offset:0x400
	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v9, v[0x44:0x45], offset:0x600
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0xf0:0xff], v[0x6a:0x6d], v[0x7a:0x7d], v[0xf0:0xff]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[62], v[64]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[63], v[0x41]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[0x42], v[0x44]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[0x43], v[0x45]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(12)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0x80:0x8f], a[0x70:0x73], v[62:0x41], a[0x80:0x8f]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[20:21], v1 offset:0x4200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[22:23], v1 offset:0x5200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[24:25], v1 offset:0x6000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[26:27], v1 offset:0x7000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[28:29], v1 offset:0x6200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[30:31], v1 offset:0x7200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[32:33], v1 offset:0x8000
	;;#ASMEND
	s_sub_i32 s72, s17, 64
	s_add_i32 s68, s15, s68
	;;#ASMSTART
	ds_read_b64_tr_b16 a[34:35], v1 offset:0x9000
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0x90:0x9f], a[0x70:0x73], v[0x42:0x45], a[0x90:0x9f]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0xa0:0xaf], a[0x74:0x77], v[62:0x41], a[0xa0:0xaf]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[30:31], v4 offset:0
	;;#ASMEND
	s_and_b32 s28, s72, 0x1fc0
	s_mul_i32 s68, s68, s38
	;;#ASMSTART
	ds_read_b64_tr_b16 v[32:33], v4 offset:0x200
	;;#ASMEND
	s_add_i32 s68, s68, s28
	;;#ASMSTART
	ds_read_b64_tr_b16 v[34:35], v4 offset:0x400
	;;#ASMEND
	s_mul_i32 s72, s68, s3
	;;#ASMSTART
	ds_read_b64_tr_b16 v[36:37], v4 offset:0x600
	;;#ASMEND
	s_ashr_i32 s73, s72, 31
	;;#ASMSTART
	ds_read_b64_tr_b16 v[38:39], v4 offset:0x800
	;;#ASMEND
	s_lshl_b64 s[72:73], s[72:73], 1
	;;#ASMSTART
	ds_read_b64_tr_b16 v[40:41], v4 offset:0xa00
	;;#ASMEND
	s_add_u32 s8, s6, s72
	;;#ASMSTART
	ds_read_b64_tr_b16 v[42:43], v4 offset:0xc00
	;;#ASMEND
	s_addc_u32 s9, s7, s73
	;;#ASMSTART
	ds_read_b64_tr_b16 v[44:45], v4 offset:0xe00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0xb0:0xbf], a[0x74:0x77], v[0x42:0x45], a[0xb0:0xbf]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x7e], 0x3fb8aa3b, v[0x7e]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x6e], v[0x6e], v[0x6f]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x6f], v[0x70], v[0x71]
	;;#ASMEND
	;;#ASMSTART
	buffer_atomic_pk_add_bf16 v[0x6e], v5, s[8:11], 0 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_atomic_pk_add_bf16 v[0x6f], v6, s[8:11], 0 offen
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0xc0:0xcf], a[0x78:0x7b], v[62:0x41], a[0xc0:0xcf]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[46:47], v4 offset:0x1000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[48:49], v4 offset:0x1200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[50:51], v4 offset:0x1400
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[52:53], v4 offset:0x1600
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[36:37], v1 offset:0x8200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[38:39], v1 offset:0x9200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[40:41], v1 offset:0xa000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[42:43], v1 offset:0xb000
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0xd0:0xdf], a[0x78:0x7b], v[0x42:0x45], a[0xd0:0xdf]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x72], v[0x72], v[0x73]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x73], v[0x74], v[0x75]
	;;#ASMEND
	;;#ASMSTART
	buffer_atomic_pk_add_bf16 v[0x72], v7, s[8:11], 0 offen
	;;#ASMEND
	s_add_i32 s28, s36, s19
	;;#ASMSTART
	buffer_atomic_pk_add_bf16 v[0x73], v8, s[8:11], 0 offen
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0xe0:0xef], a[0x7c:0x7f], v[62:0x41], a[0xe0:0xef]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[54:55], v4 offset:0x1800
	;;#ASMEND
	v_add_u32_e32 v17, s1, v11
	s_addk_i32 s1, 0x2000
	s_mul_i32 s28, s28, s22
	;;#ASMSTART
	ds_read_b64_tr_b16 v[56:57], v4 offset:0x1a00
	;;#ASMEND
	v_add_u32_e32 v22, s1, v2
	s_add_i32 s1, s28, s39
	;;#ASMSTART
	ds_read_b64_tr_b16 v[58:59], v4 offset:0x1c00
	;;#ASMEND
	s_mul_i32 s76, s1, s33
	;;#ASMSTART
	ds_read_b64_tr_b16 v[60:61], v4 offset:0x1e00
	;;#ASMEND
	s_ashr_i32 s77, s76, 31
	;;#ASMSTART
	ds_read_b64_tr_b16 a[44:45], v1 offset:0xa200
	;;#ASMEND
	s_lshl_b64 s[76:77], s[76:77], 1
	;;#ASMSTART
	ds_read_b64_tr_b16 a[46:47], v1 offset:0xb200
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0xf0:0xff], a[0x7c:0x7f], v[0x42:0x45], a[0xf0:0xff]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(4) lgkmcnt(6)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], a[0:3], v[30:33], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[62:63], v1 offset:0xc000
	;;#ASMEND
	v_readfirstlane_b32 s73, v22
	v_add_u32_e32 v22, 0x1000, v22
	s_add_u32 s28, s50, s76
	;;#ASMSTART
	ds_read_b64_tr_b16 v[64:0x41], v1 offset:0xd000
	;;#ASMEND
	v_readfirstlane_b32 s72, v22
	s_addc_u32 s29, s51, s77
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x42:0x43], v1 offset:0xc200
	;;#ASMEND
	s_mov_b32 m0, s73
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x44:0x45], v1 offset:0xd200
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], a[8:11], v[34:37], v[0x6e:0x71]
	;;#ASMEND
	s_add_i32 s1, s70, 0x2000
	buffer_load_dwordx4 v14, s[28:31], 0 offen lds
	s_mov_b32 m0, s72
	v_add_u32_e32 v28, s1, v11
	buffer_load_dwordx4 v15, s[28:31], 0 offen lds
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], a[16:19], v[38:41], v[0x6e:0x71]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x46:0x47], v1 offset:0xe000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x48:0x49], v1 offset:0xf000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x4a:0x4b], v1 offset:0xe200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x4c:0x4d], v1 offset:0xf200
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], a[24:27], v[42:45], v[0x6e:0x71]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], a[32:35], v[46:49], v[0x6e:0x71]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[0x70:0x73], v28 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[0x74:0x77], v28 offset:0x400
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], a[40:43], v[50:53], v[0x6e:0x71]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], v[62:0x41], v[54:57], v[0x6e:0x71]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[0x78:0x7b], v28 offset:0x800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[0x7c:0x7f], v28 offset:0xc00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], v[0x46:0x49], v[58:61], v[0x6e:0x71]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], a[4:7], v[30:33], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[0:3], v10 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[4:7], v10 offset:0x400
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], a[12:15], v[34:37], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], a[20:23], v[38:41], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[8:11], v10 offset:0x800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[12:15], v10 offset:0xc00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], a[28:31], v[42:45], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x6e], 0x3db504f3, v[0x6e]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x6f], 0x3db504f3, v[0x6f]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], a[36:39], v[46:49], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[16:19], v10 offset:0x1000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[20:23], v10 offset:0x1400
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], a[44:47], v[50:53], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x70], 0x3db504f3, v[0x70]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x71], 0x3db504f3, v[0x71]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(10)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], v[0x42:0x45], v[54:57], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[24:27], v10 offset:0x1800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[28:31], v10 offset:0x1c00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], v[0x4a:0x4d], v[58:61], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(2)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[46:49], a[0x70:0x73], a[0:3], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[32:35], v10 offset:0x2000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[36:39], v10 offset:0x2400
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[46:49], a[0x74:0x77], a[4:7], v[46:49]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[46:49], a[0x78:0x7b], a[8:11], v[46:49]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[40:43], v10 offset:0x2800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[44:47], v10 offset:0x2c00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[46:49], a[0x7c:0x7f], a[12:15], v[46:49]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x72], 0x3db504f3, v[0x72]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x73], 0x3db504f3, v[0x73]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[50:53], a[0x70:0x73], a[16:19], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[62:0x41], v10 offset:0x3000
	;;#ASMEND
	s_add_i32 s76, s69, 0x2000
	;;#ASMSTART
	ds_read_b128 v[0x42:0x45], v10 offset:0x3400
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[50:53], a[0x74:0x77], a[20:23], v[50:53]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x74], 0x3db504f3, v[0x74]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x75], 0x3db504f3, v[0x75]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[50:53], a[0x78:0x7b], a[24:27], v[50:53]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[0x46:0x49], v10 offset:0x3800
	;;#ASMEND
	v_add_u32_e32 v22, s76, v11
	;;#ASMSTART
	ds_read_b128 v[0x4a:0x4d], v10 offset:0x3c00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[50:53], a[0x7c:0x7f], a[28:31], v[50:53]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[46], 0x3e0293ee, v[46]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[47], 0x3e0293ee, v[47]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[48], 0x3e0293ee, v[48]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[49], 0x3e0293ee, v[49]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(6)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[54:57], a[0x70:0x73], a[32:35], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[0x4e:0x51], v22 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[0x52:0x55], v22 offset:0x400
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[54:57], a[0x74:0x77], a[36:39], v[54:57]
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[46], v[0x7e], v[46] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[47], v[0x7e], v[47] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[48], v[0x7e], v[48] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[49], v[0x7e], v[49] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(6)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[54:57], a[0x78:0x7b], a[40:43], v[54:57]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[0x56:0x59], v22 offset:0x800
	;;#ASMEND
	v_add_u32_e32 v24, s76, v12
	;;#ASMSTART
	ds_read_b128 v[0x5a:0x5d], v22 offset:0xc00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[54:57], a[0x7c:0x7f], a[44:47], v[54:57]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[50], 0x3e0293ee, v[50]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[51], 0x3e0293ee, v[51]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[52], 0x3e0293ee, v[52]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[53], 0x3e0293ee, v[53]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(6)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[58:61], a[0x70:0x73], v[62:0x41], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x5e:0x5f], v24 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x60:0x61], v24 offset:0x100
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x62:0x63], v24 offset:0x400
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x64:0x65], v24 offset:0x500
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[58:61], a[0x74:0x77], v[0x42:0x45], v[58:61]
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[50], v[0x7e], v[50] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[51], v[0x7e], v[51] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[52], v[0x7e], v[52] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[53], v[0x7e], v[53] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[58:61], a[0x78:0x7b], v[0x46:0x49], v[58:61]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x66:0x67], v24 offset:0x800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x68:0x69], v24 offset:0x900
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x6a:0x6b], v24 offset:0xc00
	;;#ASMEND
	v_add_u32_e32 v25, s1, v12
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x6c:0x6d], v24 offset:0xd00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[58:61], a[0x7c:0x7f], v[0x4a:0x4d], v[58:61]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[54], 0x3e0293ee, v[54]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[55], 0x3e0293ee, v[55]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[56], 0x3e0293ee, v[56]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[57], 0x3e0293ee, v[57]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[62:0x41], v[0x4e:0x51], a[48:51], 0
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[54], v[0x7e], v[54] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[55], v[0x7e], v[55] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[56], v[0x7e], v[56] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[57], v[0x7e], v[57] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[62:0x41], v[0x52:0x55], a[52:55], v[62:0x41]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[46], v[46]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[47], v[47]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[48], v[48]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[49], v[49]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[62:0x41], v[0x56:0x59], a[56:59], v[62:0x41]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x70:0x71], v25 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x72:0x73], v25 offset:0x100
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[62:0x41], v[0x5a:0x5d], a[60:63], v[62:0x41]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[50], v[50]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[51], v[51]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[52], v[52]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[53], v[53]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x42:0x45], v[0x4e:0x51], a[64:0x43], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x74:0x75], v25 offset:0x400
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x76:0x77], v25 offset:0x500
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x42:0x45], v[0x52:0x55], a[0x44:0x47], v[0x42:0x45]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[58], 0x3e0293ee, v[58]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[59], 0x3e0293ee, v[59]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[60], 0x3e0293ee, v[60]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[61], 0x3e0293ee, v[61]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x42:0x45], v[0x56:0x59], a[0x48:0x4b], v[0x42:0x45]
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[58], v[0x7e], v[58] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[59], v[0x7e], v[59] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[60], v[0x7e], v[60] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[61], v[0x7e], v[61] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x42:0x45], v[0x5a:0x5d], a[0x4c:0x4f], v[0x42:0x45]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x76], v[46], v[47]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x77], v[48], v[49]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x46:0x49], v[0x4e:0x51], a[0x50:0x53], 0
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[54], v[54]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[55], v[55]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[56], v[56]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[57], v[57]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x46:0x49], v[0x52:0x55], a[0x54:0x57], v[0x46:0x49]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x78], v[50], v[51]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x79], v[52], v[53]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x46:0x49], v[0x56:0x59], a[0x58:0x5b], v[0x46:0x49]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x78:0x79], v25 offset:0x800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x7a:0x7b], v25 offset:0x900
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x46:0x49], v[0x5a:0x5d], a[0x5c:0x5f], v[0x46:0x49]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[58], v[58]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[59], v[59]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[60], v[60]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[61], v[61]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x4a:0x4d], v[0x4e:0x51], a[0x60:0x63], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x7c:0x7d], v25 offset:0xc00
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x7e:0x7f], v25 offset:0xd00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x4a:0x4d], v[0x52:0x55], a[0x64:0x67], v[0x4a:0x4d]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x7a], v[54], v[55]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x7b], v[56], v[57]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x7c], v[58], v[59]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x7d], v[60], v[61]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x4a:0x4d], v[0x56:0x59], a[0x68:0x6b], v[0x4a:0x4d]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[0x76], v[0x78]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[0x77], v[0x79]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[0x7a], v[0x7c]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[0x7b], v[0x7d]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x4a:0x4d], v[0x5a:0x5d], a[0x6c:0x6f], v[0x4a:0x4d]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0x80:0x8f], v[0x5e:0x61], v[0x76:0x79], v[0x80:0x8f]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0:1], v1 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[2:3], v1 offset:0x1000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[4:5], v1 offset:0x200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[6:7], v1 offset:0x1200
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0x90:0x9f], v[0x5e:0x61], v[0x7a:0x7d], v[0x90:0x9f]
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[62], v[0x7f], v[62] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[63], v[0x7f], v[63] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[64], v[0x7f], v[64] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x41], v[0x7f], v[0x41] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x42], v[0x7f], v[0x42] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x43], v[0x7f], v[0x43] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x44], v[0x7f], v[0x44] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x45], v[0x7f], v[0x45] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0xa0:0xaf], v[0x62:0x65], v[0x76:0x79], v[0xa0:0xaf]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[8:9], v1 offset:0x2000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[10:11], v1 offset:0x3000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[12:13], v1 offset:0x2200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[14:15], v1 offset:0x3200
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0xb0:0xbf], v[0x62:0x65], v[0x7a:0x7d], v[0xb0:0xbf]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[62], v[46], v[62]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[63], v[47], v[63]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[64], v[48], v[64]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x41], v[49], v[0x41]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x42], v[50], v[0x42]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x43], v[51], v[0x43]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x44], v[52], v[0x44]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x45], v[53], v[0x45]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[62], v[62], v[63]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[63], v[64], v[0x41]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[64], v[0x42], v[0x43]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x41], v[0x44], v[0x45]
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x46], v[0x7f], v[0x46] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x47], v[0x7f], v[0x47] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x48], v[0x7f], v[0x48] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x49], v[0x7f], v[0x49] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0xc0:0xcf], v[0x66:0x69], v[0x76:0x79], v[0xc0:0xcf]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[16:17], v1 offset:0x4000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[18:19], v1 offset:0x5000
	;;#ASMEND
	s_addk_i32 s74, 0xc0
	;;#ASMSTART
	ds_write_b64 v9, v[62:63], offset:0
	;;#ASMEND
	s_addk_i32 s75, 0xc0
	v_add_u32_e32 v20, s74, v13
	;;#ASMSTART
	ds_write_b64 v9, v[64:0x41], offset:0x200
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0xd0:0xdf], v[0x66:0x69], v[0x7a:0x7d], v[0xd0:0xdf]
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x4a], v[0x7f], v[0x4a] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x4b], v[0x7f], v[0x4b] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x4c], v[0x7f], v[0x4c] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x4d], v[0x7f], v[0x4d] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x46], v[54], v[0x46]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x47], v[55], v[0x47]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x48], v[56], v[0x48]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x49], v[57], v[0x49]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x4a], v[58], v[0x4a]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x4b], v[59], v[0x4b]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x4c], v[60], v[0x4c]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x4d], v[61], v[0x4d]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x42], v[0x46], v[0x47]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x43], v[0x48], v[0x49]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x44], v[0x4a], v[0x4b]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x45], v[0x4c], v[0x4d]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0xe0:0xef], v[0x6a:0x6d], v[0x76:0x79], v[0xe0:0xef]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b32 v[0x7e], v20 offset:0

	;;#ASMEND
	v_add_u32_e32 v26, s75, v13
	;;#ASMSTART
	ds_read_b32 v[0x7f], v26 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v9, v[0x42:0x43], offset:0x400
	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v9, v[0x44:0x45], offset:0x600
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0xf0:0xff], v[0x6a:0x6d], v[0x7a:0x7d], v[0xf0:0xff]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[62], v[64]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[63], v[0x41]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[0x42], v[0x44]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[0x43], v[0x45]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(12)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0x80:0x8f], a[0x70:0x73], v[62:0x41], a[0x80:0x8f]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[20:21], v1 offset:0x4200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[22:23], v1 offset:0x5200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[24:25], v1 offset:0x6000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[26:27], v1 offset:0x7000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[28:29], v1 offset:0x6200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[30:31], v1 offset:0x7200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[32:33], v1 offset:0x8000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[34:35], v1 offset:0x9000
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0x90:0x9f], a[0x70:0x73], v[0x42:0x45], a[0x90:0x9f]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0xa0:0xaf], a[0x74:0x77], v[62:0x41], a[0xa0:0xaf]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[30:31], v4 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[32:33], v4 offset:0x200
	;;#ASMEND
	s_add_i32 s77, s68, 16
	;;#ASMSTART
	ds_read_b64_tr_b16 v[34:35], v4 offset:0x400
	;;#ASMEND
	s_mul_i32 s0, s77, s3
	;;#ASMSTART
	ds_read_b64_tr_b16 v[36:37], v4 offset:0x600
	;;#ASMEND
	s_ashr_i32 s1, s0, 31
	;;#ASMSTART
	ds_read_b64_tr_b16 v[38:39], v4 offset:0x800
	;;#ASMEND
	s_lshl_b64 s[0:1], s[0:1], 1
	;;#ASMSTART
	ds_read_b64_tr_b16 v[40:41], v4 offset:0xa00
	;;#ASMEND
	s_add_u32 s8, s6, s0
	;;#ASMSTART
	ds_read_b64_tr_b16 v[42:43], v4 offset:0xc00
	;;#ASMEND
	s_addc_u32 s9, s7, s1
	;;#ASMSTART
	ds_read_b64_tr_b16 v[44:45], v4 offset:0xe00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0xb0:0xbf], a[0x74:0x77], v[0x42:0x45], a[0xb0:0xbf]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x7e], 0x3fb8aa3b, v[0x7e]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x6e], v[0x6e], v[0x6f]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x6f], v[0x70], v[0x71]
	;;#ASMEND
	;;#ASMSTART
	buffer_atomic_pk_add_bf16 v[0x6e], v5, s[8:11], 0 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_atomic_pk_add_bf16 v[0x6f], v6, s[8:11], 0 offen
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0xc0:0xcf], a[0x78:0x7b], v[62:0x41], a[0xc0:0xcf]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[46:47], v4 offset:0x1000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[48:49], v4 offset:0x1200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[50:51], v4 offset:0x1400
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[52:53], v4 offset:0x1600
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[36:37], v1 offset:0x8200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[38:39], v1 offset:0x9200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[40:41], v1 offset:0xa000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[42:43], v1 offset:0xb000
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0xd0:0xdf], a[0x78:0x7b], v[0x42:0x45], a[0xd0:0xdf]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x72], v[0x72], v[0x73]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x73], v[0x74], v[0x75]
	;;#ASMEND
	;;#ASMSTART
	buffer_atomic_pk_add_bf16 v[0x72], v7, s[8:11], 0 offen
	;;#ASMEND
	s_add_i32 s0, s36, s25
	;;#ASMSTART
	buffer_atomic_pk_add_bf16 v[0x73], v8, s[8:11], 0 offen
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0xe0:0xef], a[0x7c:0x7f], v[62:0x41], a[0xe0:0xef]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[54:55], v4 offset:0x1800
	;;#ASMEND
	s_mul_i32 s0, s0, s26
	;;#ASMSTART
	ds_read_b64_tr_b16 v[56:57], v4 offset:0x1a00
	;;#ASMEND
	s_add_i32 s0, s0, s39
	;;#ASMSTART
	ds_read_b64_tr_b16 v[58:59], v4 offset:0x1c00
	;;#ASMEND
	s_mul_i32 s0, s0, s60
	;;#ASMSTART
	ds_read_b64_tr_b16 v[60:61], v4 offset:0x1e00
	;;#ASMEND
	s_addk_i32 s71, 0x2000
	s_ashr_i32 s1, s0, 31
	;;#ASMSTART
	ds_read_b64_tr_b16 a[44:45], v1 offset:0xa200
	;;#ASMEND
	v_add_u32_e32 v21, s71, v2
	s_lshl_b64 s[0:1], s[0:1], 1
	;;#ASMSTART
	ds_read_b64_tr_b16 a[46:47], v1 offset:0xb200
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0xf0:0xff], a[0x7c:0x7f], v[0x42:0x45], a[0xf0:0xff]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(4) lgkmcnt(6)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], a[0:3], v[30:33], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[62:63], v1 offset:0xc000
	;;#ASMEND
	v_readfirstlane_b32 s36, v21
	v_add_u32_e32 v21, 0x1000, v21
	s_add_u32 s44, s52, s0
	;;#ASMSTART
	ds_read_b64_tr_b16 v[64:0x41], v1 offset:0xd000
	;;#ASMEND
	v_readfirstlane_b32 s39, v21
	s_addc_u32 s45, s53, s1
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x42:0x43], v1 offset:0xc200
	;;#ASMEND
	s_mov_b32 m0, s36
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x44:0x45], v1 offset:0xd200
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], a[8:11], v[34:37], v[0x6e:0x71]
	;;#ASMEND
	s_addk_i32 s70, 0x3000
	buffer_load_dwordx4 v14, s[44:47], 0 offen lds
	s_mov_b32 m0, s39
	v_add_u32_e32 v27, s70, v11
	buffer_load_dwordx4 v15, s[44:47], 0 offen lds
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], a[16:19], v[38:41], v[0x6e:0x71]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x46:0x47], v1 offset:0xe000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x48:0x49], v1 offset:0xf000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x4a:0x4b], v1 offset:0xe200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x4c:0x4d], v1 offset:0xf200
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], a[24:27], v[42:45], v[0x6e:0x71]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], a[32:35], v[46:49], v[0x6e:0x71]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[0x70:0x73], v27 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[0x74:0x77], v27 offset:0x400
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], a[40:43], v[50:53], v[0x6e:0x71]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], v[62:0x41], v[54:57], v[0x6e:0x71]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[0x78:0x7b], v27 offset:0x800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[0x7c:0x7f], v27 offset:0xc00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], v[0x46:0x49], v[58:61], v[0x6e:0x71]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], a[4:7], v[30:33], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[0:3], v10 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[4:7], v10 offset:0x400
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], a[12:15], v[34:37], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], a[20:23], v[38:41], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[8:11], v10 offset:0x800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[12:15], v10 offset:0xc00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], a[28:31], v[42:45], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x6e], 0x3db504f3, v[0x6e]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x6f], 0x3db504f3, v[0x6f]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], a[36:39], v[46:49], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[16:19], v10 offset:0x1000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[20:23], v10 offset:0x1400
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], a[44:47], v[50:53], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x70], 0x3db504f3, v[0x70]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x71], 0x3db504f3, v[0x71]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(10)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], v[0x42:0x45], v[54:57], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[24:27], v10 offset:0x1800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[28:31], v10 offset:0x1c00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], v[0x4a:0x4d], v[58:61], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(2)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[46:49], a[0x70:0x73], a[0:3], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[32:35], v10 offset:0x2000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[36:39], v10 offset:0x2400
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[46:49], a[0x74:0x77], a[4:7], v[46:49]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[46:49], a[0x78:0x7b], a[8:11], v[46:49]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[40:43], v10 offset:0x2800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[44:47], v10 offset:0x2c00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[46:49], a[0x7c:0x7f], a[12:15], v[46:49]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x72], 0x3db504f3, v[0x72]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x73], 0x3db504f3, v[0x73]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[50:53], a[0x70:0x73], a[16:19], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[62:0x41], v10 offset:0x3000
	;;#ASMEND
	s_addk_i32 s69, 0x3000
	;;#ASMSTART
	ds_read_b128 v[0x42:0x45], v10 offset:0x3400
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[50:53], a[0x74:0x77], a[20:23], v[50:53]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x74], 0x3db504f3, v[0x74]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x75], 0x3db504f3, v[0x75]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[50:53], a[0x78:0x7b], a[24:27], v[50:53]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[0x46:0x49], v10 offset:0x3800
	;;#ASMEND
	v_add_u32_e32 v23, s69, v11
	;;#ASMSTART
	ds_read_b128 v[0x4a:0x4d], v10 offset:0x3c00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[50:53], a[0x7c:0x7f], a[28:31], v[50:53]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[46], 0x3e0293ee, v[46]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[47], 0x3e0293ee, v[47]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[48], 0x3e0293ee, v[48]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[49], 0x3e0293ee, v[49]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(6)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[54:57], a[0x70:0x73], a[32:35], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[0x4e:0x51], v23 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[0x52:0x55], v23 offset:0x400
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[54:57], a[0x74:0x77], a[36:39], v[54:57]
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[46], v[0x7e], v[46] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[47], v[0x7e], v[47] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[48], v[0x7e], v[48] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[49], v[0x7e], v[49] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(6)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[54:57], a[0x78:0x7b], a[40:43], v[54:57]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[0x56:0x59], v23 offset:0x800
	;;#ASMEND
	v_add_u32_e32 v21, s69, v12
	;;#ASMSTART
	ds_read_b128 v[0x5a:0x5d], v23 offset:0xc00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[54:57], a[0x7c:0x7f], a[44:47], v[54:57]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[50], 0x3e0293ee, v[50]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[51], 0x3e0293ee, v[51]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[52], 0x3e0293ee, v[52]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[53], 0x3e0293ee, v[53]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(6)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[58:61], a[0x70:0x73], v[62:0x41], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x5e:0x5f], v21 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x60:0x61], v21 offset:0x100
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x62:0x63], v21 offset:0x400
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x64:0x65], v21 offset:0x500
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[58:61], a[0x74:0x77], v[0x42:0x45], v[58:61]
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[50], v[0x7e], v[50] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[51], v[0x7e], v[51] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[52], v[0x7e], v[52] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[53], v[0x7e], v[53] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[58:61], a[0x78:0x7b], v[0x46:0x49], v[58:61]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x66:0x67], v21 offset:0x800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x68:0x69], v21 offset:0x900
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x6a:0x6b], v21 offset:0xc00
	;;#ASMEND
	v_add_u32_e32 v22, s70, v12
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x6c:0x6d], v21 offset:0xd00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[58:61], a[0x7c:0x7f], v[0x4a:0x4d], v[58:61]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[54], 0x3e0293ee, v[54]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[55], 0x3e0293ee, v[55]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[56], 0x3e0293ee, v[56]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[57], 0x3e0293ee, v[57]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[62:0x41], v[0x4e:0x51], a[48:51], 0
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[54], v[0x7e], v[54] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[55], v[0x7e], v[55] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[56], v[0x7e], v[56] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[57], v[0x7e], v[57] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[62:0x41], v[0x52:0x55], a[52:55], v[62:0x41]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[46], v[46]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[47], v[47]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[48], v[48]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[49], v[49]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[62:0x41], v[0x56:0x59], a[56:59], v[62:0x41]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x70:0x71], v22 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x72:0x73], v22 offset:0x100
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[62:0x41], v[0x5a:0x5d], a[60:63], v[62:0x41]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[50], v[50]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[51], v[51]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[52], v[52]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[53], v[53]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x42:0x45], v[0x4e:0x51], a[64:0x43], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x74:0x75], v22 offset:0x400
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x76:0x77], v22 offset:0x500
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x42:0x45], v[0x52:0x55], a[0x44:0x47], v[0x42:0x45]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[58], 0x3e0293ee, v[58]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[59], 0x3e0293ee, v[59]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[60], 0x3e0293ee, v[60]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[61], 0x3e0293ee, v[61]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x42:0x45], v[0x56:0x59], a[0x48:0x4b], v[0x42:0x45]
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[58], v[0x7e], v[58] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[59], v[0x7e], v[59] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[60], v[0x7e], v[60] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[61], v[0x7e], v[61] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x42:0x45], v[0x5a:0x5d], a[0x4c:0x4f], v[0x42:0x45]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x76], v[46], v[47]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x77], v[48], v[49]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x46:0x49], v[0x4e:0x51], a[0x50:0x53], 0
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[54], v[54]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[55], v[55]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[56], v[56]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[57], v[57]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x46:0x49], v[0x52:0x55], a[0x54:0x57], v[0x46:0x49]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x78], v[50], v[51]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x79], v[52], v[53]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x46:0x49], v[0x56:0x59], a[0x58:0x5b], v[0x46:0x49]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x78:0x79], v22 offset:0x800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x7a:0x7b], v22 offset:0x900
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x46:0x49], v[0x5a:0x5d], a[0x5c:0x5f], v[0x46:0x49]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[58], v[58]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[59], v[59]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[60], v[60]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[61], v[61]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x4a:0x4d], v[0x4e:0x51], a[0x60:0x63], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x7c:0x7d], v22 offset:0xc00
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x7e:0x7f], v22 offset:0xd00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x4a:0x4d], v[0x52:0x55], a[0x64:0x67], v[0x4a:0x4d]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x7a], v[54], v[55]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x7b], v[56], v[57]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x7c], v[58], v[59]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x7d], v[60], v[61]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x4a:0x4d], v[0x56:0x59], a[0x68:0x6b], v[0x4a:0x4d]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[0x76], v[0x78]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[0x77], v[0x79]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[0x7a], v[0x7c]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[0x7b], v[0x7d]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x4a:0x4d], v[0x5a:0x5d], a[0x6c:0x6f], v[0x4a:0x4d]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0x80:0x8f], v[0x5e:0x61], v[0x76:0x79], v[0x80:0x8f]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0:1], v1 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[2:3], v1 offset:0x1000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[4:5], v1 offset:0x200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[6:7], v1 offset:0x1200
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0x90:0x9f], v[0x5e:0x61], v[0x7a:0x7d], v[0x90:0x9f]
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[62], v[0x7f], v[62] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[63], v[0x7f], v[63] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[64], v[0x7f], v[64] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x41], v[0x7f], v[0x41] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x42], v[0x7f], v[0x42] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x43], v[0x7f], v[0x43] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x44], v[0x7f], v[0x44] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x45], v[0x7f], v[0x45] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0xa0:0xaf], v[0x62:0x65], v[0x76:0x79], v[0xa0:0xaf]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[8:9], v1 offset:0x2000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[10:11], v1 offset:0x3000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[12:13], v1 offset:0x2200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[14:15], v1 offset:0x3200
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0xb0:0xbf], v[0x62:0x65], v[0x7a:0x7d], v[0xb0:0xbf]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[62], v[46], v[62]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[63], v[47], v[63]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[64], v[48], v[64]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x41], v[49], v[0x41]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x42], v[50], v[0x42]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x43], v[51], v[0x43]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x44], v[52], v[0x44]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x45], v[53], v[0x45]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[62], v[62], v[63]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[63], v[64], v[0x41]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[64], v[0x42], v[0x43]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x41], v[0x44], v[0x45]
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x46], v[0x7f], v[0x46] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x47], v[0x7f], v[0x47] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x48], v[0x7f], v[0x48] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x49], v[0x7f], v[0x49] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0xc0:0xcf], v[0x66:0x69], v[0x76:0x79], v[0xc0:0xcf]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[16:17], v1 offset:0x4000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[18:19], v1 offset:0x5000
	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v9, v[62:63], offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v9, v[64:0x41], offset:0x200
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0xd0:0xdf], v[0x66:0x69], v[0x7a:0x7d], v[0xd0:0xdf]
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x4a], v[0x7f], v[0x4a] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x4b], v[0x7f], v[0x4b] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x4c], v[0x7f], v[0x4c] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x4d], v[0x7f], v[0x4d] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x46], v[54], v[0x46]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x47], v[55], v[0x47]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x48], v[56], v[0x48]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x49], v[57], v[0x49]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x4a], v[58], v[0x4a]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x4b], v[59], v[0x4b]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x4c], v[60], v[0x4c]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x4d], v[61], v[0x4d]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x42], v[0x46], v[0x47]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x43], v[0x48], v[0x49]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x44], v[0x4a], v[0x4b]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x45], v[0x4c], v[0x4d]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0xe0:0xef], v[0x6a:0x6d], v[0x76:0x79], v[0xe0:0xef]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b32 v[0x7e], v18 offset:0

	;;#ASMEND
	v_add_u32_e32 v19, s78, v13
	;;#ASMSTART
	ds_read_b32 v[0x7f], v19 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v9, v[0x42:0x43], offset:0x400
	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v9, v[0x44:0x45], offset:0x600
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0xf0:0xff], v[0x6a:0x6d], v[0x7a:0x7d], v[0xf0:0xff]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[62], v[64]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[63], v[0x41]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[0x42], v[0x44]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[0x43], v[0x45]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(12)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0x80:0x8f], a[0x70:0x73], v[62:0x41], a[0x80:0x8f]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[20:21], v1 offset:0x4200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[22:23], v1 offset:0x5200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[24:25], v1 offset:0x6000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[26:27], v1 offset:0x7000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[28:29], v1 offset:0x6200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[30:31], v1 offset:0x7200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[32:33], v1 offset:0x8000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[34:35], v1 offset:0x9000
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0x90:0x9f], a[0x70:0x73], v[0x42:0x45], a[0x90:0x9f]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0xa0:0xaf], a[0x74:0x77], v[62:0x41], a[0xa0:0xaf]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[30:31], v4 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[32:33], v4 offset:0x200
	;;#ASMEND
	s_add_i32 s0, s68, 32
	;;#ASMSTART
	ds_read_b64_tr_b16 v[34:35], v4 offset:0x400
	;;#ASMEND
	s_mul_i32 s0, s0, s3
	;;#ASMSTART
	ds_read_b64_tr_b16 v[36:37], v4 offset:0x600
	;;#ASMEND
	s_ashr_i32 s1, s0, 31
	;;#ASMSTART
	ds_read_b64_tr_b16 v[38:39], v4 offset:0x800
	;;#ASMEND
	s_lshl_b64 s[0:1], s[0:1], 1
	;;#ASMSTART
	ds_read_b64_tr_b16 v[40:41], v4 offset:0xa00
	;;#ASMEND
	s_add_u32 s8, s6, s0
	;;#ASMSTART
	ds_read_b64_tr_b16 v[42:43], v4 offset:0xc00
	;;#ASMEND
	s_addc_u32 s9, s7, s1
	;;#ASMSTART
	ds_read_b64_tr_b16 v[44:45], v4 offset:0xe00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0xb0:0xbf], a[0x74:0x77], v[0x42:0x45], a[0xb0:0xbf]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x7e], 0x3fb8aa3b, v[0x7e]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x6e], v[0x6e], v[0x6f]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x6f], v[0x70], v[0x71]
	;;#ASMEND
	;;#ASMSTART
	buffer_atomic_pk_add_bf16 v[0x6e], v5, s[8:11], 0 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_atomic_pk_add_bf16 v[0x6f], v6, s[8:11], 0 offen
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0xc0:0xcf], a[0x78:0x7b], v[62:0x41], a[0xc0:0xcf]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[46:47], v4 offset:0x1000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[48:49], v4 offset:0x1200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[50:51], v4 offset:0x1400
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[52:53], v4 offset:0x1600
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[36:37], v1 offset:0x8200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[38:39], v1 offset:0x9200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[40:41], v1 offset:0xa000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[42:43], v1 offset:0xb000
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0xd0:0xdf], a[0x78:0x7b], v[0x42:0x45], a[0xd0:0xdf]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x72], v[0x72], v[0x73]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x73], v[0x74], v[0x75]
	;;#ASMEND
	;;#ASMSTART
	buffer_atomic_pk_add_bf16 v[0x72], v7, s[8:11], 0 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_atomic_pk_add_bf16 v[0x73], v8, s[8:11], 0 offen
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0xe0:0xef], a[0x7c:0x7f], v[62:0x41], a[0xe0:0xef]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[54:55], v4 offset:0x1800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[56:57], v4 offset:0x1a00
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[58:59], v4 offset:0x1c00
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[60:61], v4 offset:0x1e00
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[44:45], v1 offset:0xa200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[46:47], v1 offset:0xb200
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0xf0:0xff], a[0x7c:0x7f], v[0x42:0x45], a[0xf0:0xff]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(4) lgkmcnt(6)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], a[0:3], v[30:33], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[62:63], v1 offset:0xc000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[64:0x41], v1 offset:0xd000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x42:0x43], v1 offset:0xc200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x44:0x45], v1 offset:0xd200
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], a[8:11], v[34:37], v[0x6e:0x71]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], a[16:19], v[38:41], v[0x6e:0x71]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x46:0x47], v1 offset:0xe000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x48:0x49], v1 offset:0xf000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x4a:0x4b], v1 offset:0xe200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x4c:0x4d], v1 offset:0xf200
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], a[24:27], v[42:45], v[0x6e:0x71]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], a[32:35], v[46:49], v[0x6e:0x71]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[0x70:0x73], v17 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[0x74:0x77], v17 offset:0x400
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], a[40:43], v[50:53], v[0x6e:0x71]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], v[62:0x41], v[54:57], v[0x6e:0x71]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[0x78:0x7b], v17 offset:0x800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[0x7c:0x7f], v17 offset:0xc00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], v[0x46:0x49], v[58:61], v[0x6e:0x71]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], a[4:7], v[30:33], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[0:3], v10 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[4:7], v10 offset:0x400
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], a[12:15], v[34:37], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], a[20:23], v[38:41], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[8:11], v10 offset:0x800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[12:15], v10 offset:0xc00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], a[28:31], v[42:45], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x6e], 0x3db504f3, v[0x6e]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x6f], 0x3db504f3, v[0x6f]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], a[36:39], v[46:49], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[16:19], v10 offset:0x1000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[20:23], v10 offset:0x1400
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], a[44:47], v[50:53], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x70], 0x3db504f3, v[0x70]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x71], 0x3db504f3, v[0x71]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(10)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], v[0x42:0x45], v[54:57], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[24:27], v10 offset:0x1800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[28:31], v10 offset:0x1c00
	;;#ASMEND
	s_xor_b32 s35, s35, 1
	s_xor_b32 s59, s59, 1
	s_add_i32 s37, s37, 1
	s_add_i32 s17, s17, 64
	s_cmpk_eq_i32 s37, 0x400
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], v[0x4a:0x4d], v[58:61], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(2)
	;;#ASMEND
	s_cbranch_scc0 .LBB0_1
; %bb.2:
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[46:49], a[0x70:0x73], a[0:3], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[32:35], v10 offset:0x2000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[36:39], v10 offset:0x2400
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[46:49], a[0x74:0x77], a[4:7], v[46:49]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[46:49], a[0x78:0x7b], a[8:11], v[46:49]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[40:43], v10 offset:0x2800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[44:47], v10 offset:0x2c00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[46:49], a[0x7c:0x7f], a[12:15], v[46:49]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x72], 0x3db504f3, v[0x72]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x73], 0x3db504f3, v[0x73]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[50:53], a[0x70:0x73], a[16:19], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[62:0x41], v10 offset:0x3000
	;;#ASMEND
	s_lshl_b32 s0, s35, 14
	;;#ASMSTART
	ds_read_b128 v[0x42:0x45], v10 offset:0x3400
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[50:53], a[0x74:0x77], a[20:23], v[50:53]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x74], 0x3db504f3, v[0x74]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x75], 0x3db504f3, v[0x75]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[50:53], a[0x78:0x7b], a[24:27], v[50:53]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[0x46:0x49], v10 offset:0x3800
	;;#ASMEND
	s_add_i32 s17, s62, s0
	;;#ASMSTART
	ds_read_b128 v[0x4a:0x4d], v10 offset:0x3c00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[50:53], a[0x7c:0x7f], a[28:31], v[50:53]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[46], 0x3e0293ee, v[46]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[47], 0x3e0293ee, v[47]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[48], 0x3e0293ee, v[48]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[49], 0x3e0293ee, v[49]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(6)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[54:57], a[0x70:0x73], a[32:35], 0
	;;#ASMEND
	v_add_u32_e32 v2, s17, v11
	;;#ASMSTART
	ds_read_b128 v[0x4e:0x51], v2 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[0x52:0x55], v2 offset:0x400
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[54:57], a[0x74:0x77], a[36:39], v[54:57]
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[46], v[0x7e], v[46] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[47], v[0x7e], v[47] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[48], v[0x7e], v[48] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[49], v[0x7e], v[49] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(6)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[54:57], a[0x78:0x7b], a[40:43], v[54:57]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[0x56:0x59], v2 offset:0x800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[0x5a:0x5d], v2 offset:0xc00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[54:57], a[0x7c:0x7f], a[44:47], v[54:57]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[50], 0x3e0293ee, v[50]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[51], 0x3e0293ee, v[51]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[52], 0x3e0293ee, v[52]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[53], 0x3e0293ee, v[53]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(6)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[58:61], a[0x70:0x73], v[62:0x41], 0
	;;#ASMEND
	v_add_u32_e32 v2, s17, v12
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x5e:0x5f], v2 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x60:0x61], v2 offset:0x100
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x62:0x63], v2 offset:0x400
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x64:0x65], v2 offset:0x500
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[58:61], a[0x74:0x77], v[0x42:0x45], v[58:61]
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[50], v[0x7e], v[50] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[51], v[0x7e], v[51] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[52], v[0x7e], v[52] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[53], v[0x7e], v[53] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[58:61], a[0x78:0x7b], v[0x46:0x49], v[58:61]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x66:0x67], v2 offset:0x800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x68:0x69], v2 offset:0x900
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x6a:0x6b], v2 offset:0xc00
	;;#ASMEND
	s_add_i32 s19, s27, s0
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x6c:0x6d], v2 offset:0xd00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[58:61], a[0x7c:0x7f], v[0x4a:0x4d], v[58:61]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[54], 0x3e0293ee, v[54]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[55], 0x3e0293ee, v[55]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[56], 0x3e0293ee, v[56]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[57], 0x3e0293ee, v[57]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[62:0x41], v[0x4e:0x51], a[48:51], 0
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[54], v[0x7e], v[54] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[55], v[0x7e], v[55] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[56], v[0x7e], v[56] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[57], v[0x7e], v[57] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[62:0x41], v[0x52:0x55], a[52:55], v[62:0x41]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[46], v[46]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[47], v[47]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[48], v[48]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[49], v[49]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[62:0x41], v[0x56:0x59], a[56:59], v[62:0x41]
	;;#ASMEND
	v_add_u32_e32 v2, s19, v12
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x70:0x71], v2 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x72:0x73], v2 offset:0x100
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[62:0x41], v[0x5a:0x5d], a[60:63], v[62:0x41]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[50], v[50]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[51], v[51]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[52], v[52]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[53], v[53]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x42:0x45], v[0x4e:0x51], a[64:0x43], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x74:0x75], v2 offset:0x400
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x76:0x77], v2 offset:0x500
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x42:0x45], v[0x52:0x55], a[0x44:0x47], v[0x42:0x45]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[58], 0x3e0293ee, v[58]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[59], 0x3e0293ee, v[59]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[60], 0x3e0293ee, v[60]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[61], 0x3e0293ee, v[61]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x42:0x45], v[0x56:0x59], a[0x48:0x4b], v[0x42:0x45]
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[58], v[0x7e], v[58] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[59], v[0x7e], v[59] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[60], v[0x7e], v[60] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[61], v[0x7e], v[61] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x42:0x45], v[0x5a:0x5d], a[0x4c:0x4f], v[0x42:0x45]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x76], v[46], v[47]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x77], v[48], v[49]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x46:0x49], v[0x4e:0x51], a[0x50:0x53], 0
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[54], v[54]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[55], v[55]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[56], v[56]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[57], v[57]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x46:0x49], v[0x52:0x55], a[0x54:0x57], v[0x46:0x49]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x78], v[50], v[51]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x79], v[52], v[53]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x46:0x49], v[0x56:0x59], a[0x58:0x5b], v[0x46:0x49]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x78:0x79], v2 offset:0x800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x7a:0x7b], v2 offset:0x900
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x46:0x49], v[0x5a:0x5d], a[0x5c:0x5f], v[0x46:0x49]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[58], v[58]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[59], v[59]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[60], v[60]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[61], v[61]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x4a:0x4d], v[0x4e:0x51], a[0x60:0x63], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x7c:0x7d], v2 offset:0xc00
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x7e:0x7f], v2 offset:0xd00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x4a:0x4d], v[0x52:0x55], a[0x64:0x67], v[0x4a:0x4d]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x7a], v[54], v[55]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x7b], v[56], v[57]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x7c], v[58], v[59]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x7d], v[60], v[61]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x4a:0x4d], v[0x56:0x59], a[0x68:0x6b], v[0x4a:0x4d]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[0x76], v[0x78]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[0x77], v[0x79]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[0x7a], v[0x7c]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[0x7b], v[0x7d]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x4a:0x4d], v[0x5a:0x5d], a[0x6c:0x6f], v[0x4a:0x4d]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0x80:0x8f], v[0x5e:0x61], v[0x76:0x79], v[0x80:0x8f]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0:1], v1 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[2:3], v1 offset:0x1000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[4:5], v1 offset:0x200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[6:7], v1 offset:0x1200
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0x90:0x9f], v[0x5e:0x61], v[0x7a:0x7d], v[0x90:0x9f]
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[62], v[0x7f], v[62] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[63], v[0x7f], v[63] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[64], v[0x7f], v[64] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x41], v[0x7f], v[0x41] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x42], v[0x7f], v[0x42] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x43], v[0x7f], v[0x43] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x44], v[0x7f], v[0x44] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x45], v[0x7f], v[0x45] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0xa0:0xaf], v[0x62:0x65], v[0x76:0x79], v[0xa0:0xaf]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[8:9], v1 offset:0x2000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[10:11], v1 offset:0x3000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[12:13], v1 offset:0x2200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[14:15], v1 offset:0x3200
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0xb0:0xbf], v[0x62:0x65], v[0x7a:0x7d], v[0xb0:0xbf]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[62], v[46], v[62]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[63], v[47], v[63]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[64], v[48], v[64]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x41], v[49], v[0x41]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x42], v[50], v[0x42]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x43], v[51], v[0x43]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x44], v[52], v[0x44]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x45], v[53], v[0x45]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[62], v[62], v[63]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[63], v[64], v[0x41]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[64], v[0x42], v[0x43]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x41], v[0x44], v[0x45]
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x46], v[0x7f], v[0x46] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x47], v[0x7f], v[0x47] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x48], v[0x7f], v[0x48] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x49], v[0x7f], v[0x49] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0xc0:0xcf], v[0x66:0x69], v[0x76:0x79], v[0xc0:0xcf]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[16:17], v1 offset:0x4000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[18:19], v1 offset:0x5000
	;;#ASMEND
	s_lshl_b32 s0, s35, 8
	;;#ASMSTART
	ds_write_b64 v9, v[62:63], offset:0
	;;#ASMEND
	s_add_i32 s22, s5, s0
	;;#ASMSTART
	ds_write_b64 v9, v[64:0x41], offset:0x200
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0xd0:0xdf], v[0x66:0x69], v[0x7a:0x7d], v[0xd0:0xdf]
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x4a], v[0x7f], v[0x4a] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x4b], v[0x7f], v[0x4b] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x4c], v[0x7f], v[0x4c] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x4d], v[0x7f], v[0x4d] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x46], v[54], v[0x46]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x47], v[55], v[0x47]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x48], v[56], v[0x48]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x49], v[57], v[0x49]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x4a], v[58], v[0x4a]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x4b], v[59], v[0x4b]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x4c], v[60], v[0x4c]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x4d], v[61], v[0x4d]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x42], v[0x46], v[0x47]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x43], v[0x48], v[0x49]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x44], v[0x4a], v[0x4b]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x45], v[0x4c], v[0x4d]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0xe0:0xef], v[0x6a:0x6d], v[0x76:0x79], v[0xe0:0xef]
	;;#ASMEND
	v_add3_u32 v2, s22, 64, v13
	;;#ASMSTART
	ds_read_b32 v[0x7e], v2 offset:0

	;;#ASMEND
	s_add_i32 s23, s23, s0
	v_add3_u32 v2, s23, 64, v13
	;;#ASMSTART
	ds_read_b32 v[0x7f], v2 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v9, v[0x42:0x43], offset:0x400
	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v9, v[0x44:0x45], offset:0x600
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0xf0:0xff], v[0x6a:0x6d], v[0x7a:0x7d], v[0xf0:0xff]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[62], v[64]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[63], v[0x41]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[0x42], v[0x44]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[0x43], v[0x45]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(12)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0x80:0x8f], a[0x70:0x73], v[62:0x41], a[0x80:0x8f]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[20:21], v1 offset:0x4200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[22:23], v1 offset:0x5200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[24:25], v1 offset:0x6000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[26:27], v1 offset:0x7000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[28:29], v1 offset:0x6200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[30:31], v1 offset:0x7200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[32:33], v1 offset:0x8000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[34:35], v1 offset:0x9000
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0x90:0x9f], a[0x70:0x73], v[0x42:0x45], a[0x90:0x9f]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0xa0:0xaf], a[0x74:0x77], v[62:0x41], a[0xa0:0xaf]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[30:31], v4 offset:0
	;;#ASMEND
	s_add_i32 s5, s15, 7
	;;#ASMSTART
	ds_read_b64_tr_b16 v[32:33], v4 offset:0x200
	;;#ASMEND
	s_mul_i32 s5, s5, s38
	;;#ASMSTART
	ds_read_b64_tr_b16 v[34:35], v4 offset:0x400
	;;#ASMEND
	s_add_i32 s0, s5, 0x1fb0
	;;#ASMSTART
	ds_read_b64_tr_b16 v[36:37], v4 offset:0x600
	;;#ASMEND
	s_mul_i32 s0, s0, s3
	;;#ASMSTART
	ds_read_b64_tr_b16 v[38:39], v4 offset:0x800
	;;#ASMEND
	s_ashr_i32 s1, s0, 31
	;;#ASMSTART
	ds_read_b64_tr_b16 v[40:41], v4 offset:0xa00
	;;#ASMEND
	s_lshl_b64 s[8:9], s[0:1], 1
	;;#ASMSTART
	ds_read_b64_tr_b16 v[42:43], v4 offset:0xc00
	;;#ASMEND
	s_add_u32 s8, s6, s8
	;;#ASMSTART
	ds_read_b64_tr_b16 v[44:45], v4 offset:0xe00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0xb0:0xbf], a[0x74:0x77], v[0x42:0x45], a[0xb0:0xbf]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x7e], 0x3fb8aa3b, v[0x7e]
	;;#ASMEND
	s_addc_u32 s9, s7, s9
	s_mov_b32 s11, 0x20000
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x6e], v[0x6e], v[0x6f]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x6f], v[0x70], v[0x71]
	;;#ASMEND
	;;#ASMSTART
	buffer_atomic_pk_add_bf16 v[0x6e], v5, s[8:11], 0 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_atomic_pk_add_bf16 v[0x6f], v6, s[8:11], 0 offen
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0xc0:0xcf], a[0x78:0x7b], v[62:0x41], a[0xc0:0xcf]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[46:47], v4 offset:0x1000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[48:49], v4 offset:0x1200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[50:51], v4 offset:0x1400
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[52:53], v4 offset:0x1600
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[36:37], v1 offset:0x8200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[38:39], v1 offset:0x9200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[40:41], v1 offset:0xa000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[42:43], v1 offset:0xb000
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0xd0:0xdf], a[0x78:0x7b], v[0x42:0x45], a[0xd0:0xdf]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x72], v[0x72], v[0x73]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x73], v[0x74], v[0x75]
	;;#ASMEND
	;;#ASMSTART
	buffer_atomic_pk_add_bf16 v[0x72], v7, s[8:11], 0 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_atomic_pk_add_bf16 v[0x73], v8, s[8:11], 0 offen
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0xe0:0xef], a[0x7c:0x7f], v[62:0x41], a[0xe0:0xef]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[54:55], v4 offset:0x1800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[56:57], v4 offset:0x1a00
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[58:59], v4 offset:0x1c00
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[60:61], v4 offset:0x1e00
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[44:45], v1 offset:0xa200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[46:47], v1 offset:0xb200
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0xf0:0xff], a[0x7c:0x7f], v[0x42:0x45], a[0xf0:0xff]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(6)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], a[0:3], v[30:33], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[62:63], v1 offset:0xc000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[64:0x41], v1 offset:0xd000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x42:0x43], v1 offset:0xc200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x44:0x45], v1 offset:0xd200
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], a[8:11], v[34:37], v[0x6e:0x71]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], a[16:19], v[38:41], v[0x6e:0x71]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x46:0x47], v1 offset:0xe000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x48:0x49], v1 offset:0xf000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x4a:0x4b], v1 offset:0xe200
	;;#ASMEND
	s_add_i32 s1, s19, 0x1000
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x4c:0x4d], v1 offset:0xf200
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], a[24:27], v[42:45], v[0x6e:0x71]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], a[32:35], v[46:49], v[0x6e:0x71]
	;;#ASMEND
	v_add_u32_e32 v2, s1, v11
	;;#ASMSTART
	ds_read_b128 a[0x70:0x73], v2 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[0x74:0x77], v2 offset:0x400
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], a[40:43], v[50:53], v[0x6e:0x71]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], v[62:0x41], v[54:57], v[0x6e:0x71]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[0x78:0x7b], v2 offset:0x800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[0x7c:0x7f], v2 offset:0xc00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], v[0x46:0x49], v[58:61], v[0x6e:0x71]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], a[4:7], v[30:33], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[0:3], v10 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[4:7], v10 offset:0x400
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], a[12:15], v[34:37], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], a[20:23], v[38:41], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[8:11], v10 offset:0x800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[12:15], v10 offset:0xc00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], a[28:31], v[42:45], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x6e], 0x3db504f3, v[0x6e]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x6f], 0x3db504f3, v[0x6f]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], a[36:39], v[46:49], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[16:19], v10 offset:0x1000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[20:23], v10 offset:0x1400
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], a[44:47], v[50:53], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x70], 0x3db504f3, v[0x70]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x71], 0x3db504f3, v[0x71]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(10)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], v[0x42:0x45], v[54:57], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[24:27], v10 offset:0x1800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[28:31], v10 offset:0x1c00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], v[0x4a:0x4d], v[58:61], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(2)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[46:49], a[0x70:0x73], a[0:3], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[32:35], v10 offset:0x2000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[36:39], v10 offset:0x2400
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[46:49], a[0x74:0x77], a[4:7], v[46:49]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[46:49], a[0x78:0x7b], a[8:11], v[46:49]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[40:43], v10 offset:0x2800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[44:47], v10 offset:0x2c00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[46:49], a[0x7c:0x7f], a[12:15], v[46:49]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x72], 0x3db504f3, v[0x72]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x73], 0x3db504f3, v[0x73]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[50:53], a[0x70:0x73], a[16:19], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[62:0x41], v10 offset:0x3000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[0x42:0x45], v10 offset:0x3400
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[50:53], a[0x74:0x77], a[20:23], v[50:53]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x74], 0x3db504f3, v[0x74]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x75], 0x3db504f3, v[0x75]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[50:53], a[0x78:0x7b], a[24:27], v[50:53]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[0x46:0x49], v10 offset:0x3800
	;;#ASMEND
	s_add_i32 s8, s17, 0x1000
	;;#ASMSTART
	ds_read_b128 v[0x4a:0x4d], v10 offset:0x3c00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[50:53], a[0x7c:0x7f], a[28:31], v[50:53]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[46], 0x3e0293ee, v[46]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[47], 0x3e0293ee, v[47]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[48], 0x3e0293ee, v[48]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[49], 0x3e0293ee, v[49]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(6)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[54:57], a[0x70:0x73], a[32:35], 0
	;;#ASMEND
	v_add_u32_e32 v2, s8, v11
	;;#ASMSTART
	ds_read_b128 v[0x4e:0x51], v2 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[0x52:0x55], v2 offset:0x400
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[54:57], a[0x74:0x77], a[36:39], v[54:57]
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[46], v[0x7e], v[46] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[47], v[0x7e], v[47] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[48], v[0x7e], v[48] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[49], v[0x7e], v[49] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(6)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[54:57], a[0x78:0x7b], a[40:43], v[54:57]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[0x56:0x59], v2 offset:0x800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[0x5a:0x5d], v2 offset:0xc00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[54:57], a[0x7c:0x7f], a[44:47], v[54:57]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[50], 0x3e0293ee, v[50]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[51], 0x3e0293ee, v[51]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[52], 0x3e0293ee, v[52]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[53], 0x3e0293ee, v[53]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(6)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[58:61], a[0x70:0x73], v[62:0x41], 0
	;;#ASMEND
	v_add_u32_e32 v2, s8, v12
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x5e:0x5f], v2 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x60:0x61], v2 offset:0x100
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x62:0x63], v2 offset:0x400
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x64:0x65], v2 offset:0x500
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[58:61], a[0x74:0x77], v[0x42:0x45], v[58:61]
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[50], v[0x7e], v[50] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[51], v[0x7e], v[51] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[52], v[0x7e], v[52] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[53], v[0x7e], v[53] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[58:61], a[0x78:0x7b], v[0x46:0x49], v[58:61]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x66:0x67], v2 offset:0x800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x68:0x69], v2 offset:0x900
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x6a:0x6b], v2 offset:0xc00
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x6c:0x6d], v2 offset:0xd00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[58:61], a[0x7c:0x7f], v[0x4a:0x4d], v[58:61]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[54], 0x3e0293ee, v[54]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[55], 0x3e0293ee, v[55]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[56], 0x3e0293ee, v[56]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[57], 0x3e0293ee, v[57]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[62:0x41], v[0x4e:0x51], a[48:51], 0
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[54], v[0x7e], v[54] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[55], v[0x7e], v[55] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[56], v[0x7e], v[56] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[57], v[0x7e], v[57] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[62:0x41], v[0x52:0x55], a[52:55], v[62:0x41]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[46], v[46]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[47], v[47]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[48], v[48]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[49], v[49]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[62:0x41], v[0x56:0x59], a[56:59], v[62:0x41]
	;;#ASMEND
	v_add_u32_e32 v2, s1, v12
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x70:0x71], v2 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x72:0x73], v2 offset:0x100
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[62:0x41], v[0x5a:0x5d], a[60:63], v[62:0x41]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[50], v[50]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[51], v[51]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[52], v[52]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[53], v[53]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x42:0x45], v[0x4e:0x51], a[64:0x43], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x74:0x75], v2 offset:0x400
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x76:0x77], v2 offset:0x500
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x42:0x45], v[0x52:0x55], a[0x44:0x47], v[0x42:0x45]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[58], 0x3e0293ee, v[58]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[59], 0x3e0293ee, v[59]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[60], 0x3e0293ee, v[60]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[61], 0x3e0293ee, v[61]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x42:0x45], v[0x56:0x59], a[0x48:0x4b], v[0x42:0x45]
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[58], v[0x7e], v[58] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[59], v[0x7e], v[59] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[60], v[0x7e], v[60] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[61], v[0x7e], v[61] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x42:0x45], v[0x5a:0x5d], a[0x4c:0x4f], v[0x42:0x45]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x76], v[46], v[47]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x77], v[48], v[49]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x46:0x49], v[0x4e:0x51], a[0x50:0x53], 0
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[54], v[54]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[55], v[55]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[56], v[56]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[57], v[57]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x46:0x49], v[0x52:0x55], a[0x54:0x57], v[0x46:0x49]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x78], v[50], v[51]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x79], v[52], v[53]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x46:0x49], v[0x56:0x59], a[0x58:0x5b], v[0x46:0x49]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x78:0x79], v2 offset:0x800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x7a:0x7b], v2 offset:0x900
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x46:0x49], v[0x5a:0x5d], a[0x5c:0x5f], v[0x46:0x49]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[58], v[58]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[59], v[59]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[60], v[60]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[61], v[61]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x4a:0x4d], v[0x4e:0x51], a[0x60:0x63], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x7c:0x7d], v2 offset:0xc00
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x7e:0x7f], v2 offset:0xd00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x4a:0x4d], v[0x52:0x55], a[0x64:0x67], v[0x4a:0x4d]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x7a], v[54], v[55]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x7b], v[56], v[57]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x7c], v[58], v[59]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x7d], v[60], v[61]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x4a:0x4d], v[0x56:0x59], a[0x68:0x6b], v[0x4a:0x4d]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[0x76], v[0x78]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[0x77], v[0x79]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[0x7a], v[0x7c]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[0x7b], v[0x7d]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x4a:0x4d], v[0x5a:0x5d], a[0x6c:0x6f], v[0x4a:0x4d]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0x80:0x8f], v[0x5e:0x61], v[0x76:0x79], v[0x80:0x8f]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0:1], v1 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[2:3], v1 offset:0x1000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[4:5], v1 offset:0x200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[6:7], v1 offset:0x1200
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0x90:0x9f], v[0x5e:0x61], v[0x7a:0x7d], v[0x90:0x9f]
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[62], v[0x7f], v[62] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[63], v[0x7f], v[63] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[64], v[0x7f], v[64] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x41], v[0x7f], v[0x41] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x42], v[0x7f], v[0x42] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x43], v[0x7f], v[0x43] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x44], v[0x7f], v[0x44] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x45], v[0x7f], v[0x45] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0xa0:0xaf], v[0x62:0x65], v[0x76:0x79], v[0xa0:0xaf]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[8:9], v1 offset:0x2000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[10:11], v1 offset:0x3000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[12:13], v1 offset:0x2200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[14:15], v1 offset:0x3200
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0xb0:0xbf], v[0x62:0x65], v[0x7a:0x7d], v[0xb0:0xbf]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[62], v[46], v[62]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[63], v[47], v[63]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[64], v[48], v[64]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x41], v[49], v[0x41]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x42], v[50], v[0x42]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x43], v[51], v[0x43]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x44], v[52], v[0x44]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x45], v[53], v[0x45]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[62], v[62], v[63]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[63], v[64], v[0x41]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[64], v[0x42], v[0x43]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x41], v[0x44], v[0x45]
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x46], v[0x7f], v[0x46] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x47], v[0x7f], v[0x47] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x48], v[0x7f], v[0x48] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x49], v[0x7f], v[0x49] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0xc0:0xcf], v[0x66:0x69], v[0x76:0x79], v[0xc0:0xcf]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[16:17], v1 offset:0x4000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[18:19], v1 offset:0x5000
	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v9, v[62:63], offset:0
	;;#ASMEND
	s_add_i32 s1, s22, 0x80
	;;#ASMSTART
	ds_write_b64 v9, v[64:0x41], offset:0x200
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0xd0:0xdf], v[0x66:0x69], v[0x7a:0x7d], v[0xd0:0xdf]
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x4a], v[0x7f], v[0x4a] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x4b], v[0x7f], v[0x4b] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x4c], v[0x7f], v[0x4c] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x4d], v[0x7f], v[0x4d] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x46], v[54], v[0x46]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x47], v[55], v[0x47]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x48], v[56], v[0x48]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x49], v[57], v[0x49]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x4a], v[58], v[0x4a]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x4b], v[59], v[0x4b]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x4c], v[60], v[0x4c]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x4d], v[61], v[0x4d]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x42], v[0x46], v[0x47]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x43], v[0x48], v[0x49]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x44], v[0x4a], v[0x4b]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x45], v[0x4c], v[0x4d]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0xe0:0xef], v[0x6a:0x6d], v[0x76:0x79], v[0xe0:0xef]
	;;#ASMEND
	v_add_u32_e32 v2, s1, v13
	;;#ASMSTART
	ds_read_b32 v[0x7e], v2 offset:0

	;;#ASMEND
	s_add_i32 s1, s23, 0x80
	v_add_u32_e32 v2, s1, v13
	;;#ASMSTART
	ds_read_b32 v[0x7f], v2 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v9, v[0x42:0x43], offset:0x400
	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v9, v[0x44:0x45], offset:0x600
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0xf0:0xff], v[0x6a:0x6d], v[0x7a:0x7d], v[0xf0:0xff]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[62], v[64]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[63], v[0x41]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[0x42], v[0x44]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[0x43], v[0x45]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(12)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0x80:0x8f], a[0x70:0x73], v[62:0x41], a[0x80:0x8f]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[20:21], v1 offset:0x4200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[22:23], v1 offset:0x5200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[24:25], v1 offset:0x6000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[26:27], v1 offset:0x7000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[28:29], v1 offset:0x6200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[30:31], v1 offset:0x7200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[32:33], v1 offset:0x8000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[34:35], v1 offset:0x9000
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0x90:0x9f], a[0x70:0x73], v[0x42:0x45], a[0x90:0x9f]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0xa0:0xaf], a[0x74:0x77], v[62:0x41], a[0xa0:0xaf]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[30:31], v4 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[32:33], v4 offset:0x200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[34:35], v4 offset:0x400
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[36:37], v4 offset:0x600
	;;#ASMEND
	s_add_i32 s0, s0, s13
	;;#ASMSTART
	ds_read_b64_tr_b16 v[38:39], v4 offset:0x800
	;;#ASMEND
	s_ashr_i32 s1, s0, 31
	;;#ASMSTART
	ds_read_b64_tr_b16 v[40:41], v4 offset:0xa00
	;;#ASMEND
	s_lshl_b64 s[8:9], s[0:1], 1
	;;#ASMSTART
	ds_read_b64_tr_b16 v[42:43], v4 offset:0xc00
	;;#ASMEND
	s_add_u32 s8, s6, s8
	;;#ASMSTART
	ds_read_b64_tr_b16 v[44:45], v4 offset:0xe00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0xb0:0xbf], a[0x74:0x77], v[0x42:0x45], a[0xb0:0xbf]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x7e], 0x3fb8aa3b, v[0x7e]
	;;#ASMEND
	s_addc_u32 s9, s7, s9
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x6e], v[0x6e], v[0x6f]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x6f], v[0x70], v[0x71]
	;;#ASMEND
	;;#ASMSTART
	buffer_atomic_pk_add_bf16 v[0x6e], v5, s[8:11], 0 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_atomic_pk_add_bf16 v[0x6f], v6, s[8:11], 0 offen
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0xc0:0xcf], a[0x78:0x7b], v[62:0x41], a[0xc0:0xcf]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[46:47], v4 offset:0x1000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[48:49], v4 offset:0x1200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[50:51], v4 offset:0x1400
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[52:53], v4 offset:0x1600
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[36:37], v1 offset:0x8200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[38:39], v1 offset:0x9200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[40:41], v1 offset:0xa000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[42:43], v1 offset:0xb000
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0xd0:0xdf], a[0x78:0x7b], v[0x42:0x45], a[0xd0:0xdf]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x72], v[0x72], v[0x73]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x73], v[0x74], v[0x75]
	;;#ASMEND
	;;#ASMSTART
	buffer_atomic_pk_add_bf16 v[0x72], v7, s[8:11], 0 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_atomic_pk_add_bf16 v[0x73], v8, s[8:11], 0 offen
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0xe0:0xef], a[0x7c:0x7f], v[62:0x41], a[0xe0:0xef]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[54:55], v4 offset:0x1800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[56:57], v4 offset:0x1a00
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[58:59], v4 offset:0x1c00
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[60:61], v4 offset:0x1e00
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[44:45], v1 offset:0xa200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[46:47], v1 offset:0xb200
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0xf0:0xff], a[0x7c:0x7f], v[0x42:0x45], a[0xf0:0xff]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(6)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], a[0:3], v[30:33], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[62:63], v1 offset:0xc000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[64:0x41], v1 offset:0xd000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x42:0x43], v1 offset:0xc200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x44:0x45], v1 offset:0xd200
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], a[8:11], v[34:37], v[0x6e:0x71]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], a[16:19], v[38:41], v[0x6e:0x71]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x46:0x47], v1 offset:0xe000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x48:0x49], v1 offset:0xf000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x4a:0x4b], v1 offset:0xe200
	;;#ASMEND
	s_add_i32 s1, s19, 0x2000
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x4c:0x4d], v1 offset:0xf200
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], a[24:27], v[42:45], v[0x6e:0x71]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], a[32:35], v[46:49], v[0x6e:0x71]
	;;#ASMEND
	v_add_u32_e32 v2, s1, v11
	;;#ASMSTART
	ds_read_b128 a[0x70:0x73], v2 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[0x74:0x77], v2 offset:0x400
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], a[40:43], v[50:53], v[0x6e:0x71]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], v[62:0x41], v[54:57], v[0x6e:0x71]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[0x78:0x7b], v2 offset:0x800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[0x7c:0x7f], v2 offset:0xc00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], v[0x46:0x49], v[58:61], v[0x6e:0x71]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], a[4:7], v[30:33], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[0:3], v10 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[4:7], v10 offset:0x400
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], a[12:15], v[34:37], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], a[20:23], v[38:41], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[8:11], v10 offset:0x800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[12:15], v10 offset:0xc00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], a[28:31], v[42:45], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x6e], 0x3db504f3, v[0x6e]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x6f], 0x3db504f3, v[0x6f]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], a[36:39], v[46:49], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[16:19], v10 offset:0x1000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[20:23], v10 offset:0x1400
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], a[44:47], v[50:53], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x70], 0x3db504f3, v[0x70]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x71], 0x3db504f3, v[0x71]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(10)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], v[0x42:0x45], v[54:57], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[24:27], v10 offset:0x1800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[28:31], v10 offset:0x1c00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], v[0x4a:0x4d], v[58:61], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(2)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[46:49], a[0x70:0x73], a[0:3], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[32:35], v10 offset:0x2000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[36:39], v10 offset:0x2400
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[46:49], a[0x74:0x77], a[4:7], v[46:49]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[46:49], a[0x78:0x7b], a[8:11], v[46:49]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[40:43], v10 offset:0x2800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[44:47], v10 offset:0x2c00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[46:49], a[0x7c:0x7f], a[12:15], v[46:49]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x72], 0x3db504f3, v[0x72]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x73], 0x3db504f3, v[0x73]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[50:53], a[0x70:0x73], a[16:19], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[62:0x41], v10 offset:0x3000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[0x42:0x45], v10 offset:0x3400
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[50:53], a[0x74:0x77], a[20:23], v[50:53]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x74], 0x3db504f3, v[0x74]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x75], 0x3db504f3, v[0x75]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[50:53], a[0x78:0x7b], a[24:27], v[50:53]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[0x46:0x49], v10 offset:0x3800
	;;#ASMEND
	s_add_i32 s8, s17, 0x2000
	;;#ASMSTART
	ds_read_b128 v[0x4a:0x4d], v10 offset:0x3c00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[50:53], a[0x7c:0x7f], a[28:31], v[50:53]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[46], 0x3e0293ee, v[46]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[47], 0x3e0293ee, v[47]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[48], 0x3e0293ee, v[48]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[49], 0x3e0293ee, v[49]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(6)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[54:57], a[0x70:0x73], a[32:35], 0
	;;#ASMEND
	v_add_u32_e32 v2, s8, v11
	;;#ASMSTART
	ds_read_b128 v[0x4e:0x51], v2 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[0x52:0x55], v2 offset:0x400
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[54:57], a[0x74:0x77], a[36:39], v[54:57]
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[46], v[0x7e], v[46] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[47], v[0x7e], v[47] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[48], v[0x7e], v[48] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[49], v[0x7e], v[49] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(6)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[54:57], a[0x78:0x7b], a[40:43], v[54:57]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[0x56:0x59], v2 offset:0x800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[0x5a:0x5d], v2 offset:0xc00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[54:57], a[0x7c:0x7f], a[44:47], v[54:57]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[50], 0x3e0293ee, v[50]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[51], 0x3e0293ee, v[51]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[52], 0x3e0293ee, v[52]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[53], 0x3e0293ee, v[53]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(6)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[58:61], a[0x70:0x73], v[62:0x41], 0
	;;#ASMEND
	v_add_u32_e32 v2, s8, v12
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x5e:0x5f], v2 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x60:0x61], v2 offset:0x100
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x62:0x63], v2 offset:0x400
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x64:0x65], v2 offset:0x500
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[58:61], a[0x74:0x77], v[0x42:0x45], v[58:61]
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[50], v[0x7e], v[50] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[51], v[0x7e], v[51] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[52], v[0x7e], v[52] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[53], v[0x7e], v[53] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[58:61], a[0x78:0x7b], v[0x46:0x49], v[58:61]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x66:0x67], v2 offset:0x800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x68:0x69], v2 offset:0x900
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x6a:0x6b], v2 offset:0xc00
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x6c:0x6d], v2 offset:0xd00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[58:61], a[0x7c:0x7f], v[0x4a:0x4d], v[58:61]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[54], 0x3e0293ee, v[54]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[55], 0x3e0293ee, v[55]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[56], 0x3e0293ee, v[56]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[57], 0x3e0293ee, v[57]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[62:0x41], v[0x4e:0x51], a[48:51], 0
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[54], v[0x7e], v[54] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[55], v[0x7e], v[55] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[56], v[0x7e], v[56] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[57], v[0x7e], v[57] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[62:0x41], v[0x52:0x55], a[52:55], v[62:0x41]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[46], v[46]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[47], v[47]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[48], v[48]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[49], v[49]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[62:0x41], v[0x56:0x59], a[56:59], v[62:0x41]
	;;#ASMEND
	v_add_u32_e32 v2, s1, v12
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x70:0x71], v2 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x72:0x73], v2 offset:0x100
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[62:0x41], v[0x5a:0x5d], a[60:63], v[62:0x41]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[50], v[50]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[51], v[51]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[52], v[52]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[53], v[53]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x42:0x45], v[0x4e:0x51], a[64:0x43], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x74:0x75], v2 offset:0x400
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x76:0x77], v2 offset:0x500
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x42:0x45], v[0x52:0x55], a[0x44:0x47], v[0x42:0x45]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[58], 0x3e0293ee, v[58]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[59], 0x3e0293ee, v[59]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[60], 0x3e0293ee, v[60]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[61], 0x3e0293ee, v[61]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x42:0x45], v[0x56:0x59], a[0x48:0x4b], v[0x42:0x45]
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[58], v[0x7e], v[58] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[59], v[0x7e], v[59] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[60], v[0x7e], v[60] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[61], v[0x7e], v[61] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x42:0x45], v[0x5a:0x5d], a[0x4c:0x4f], v[0x42:0x45]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x76], v[46], v[47]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x77], v[48], v[49]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x46:0x49], v[0x4e:0x51], a[0x50:0x53], 0
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[54], v[54]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[55], v[55]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[56], v[56]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[57], v[57]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x46:0x49], v[0x52:0x55], a[0x54:0x57], v[0x46:0x49]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x78], v[50], v[51]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x79], v[52], v[53]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x46:0x49], v[0x56:0x59], a[0x58:0x5b], v[0x46:0x49]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x78:0x79], v2 offset:0x800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x7a:0x7b], v2 offset:0x900
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x46:0x49], v[0x5a:0x5d], a[0x5c:0x5f], v[0x46:0x49]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[58], v[58]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[59], v[59]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[60], v[60]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[61], v[61]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x4a:0x4d], v[0x4e:0x51], a[0x60:0x63], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x7c:0x7d], v2 offset:0xc00
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x7e:0x7f], v2 offset:0xd00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x4a:0x4d], v[0x52:0x55], a[0x64:0x67], v[0x4a:0x4d]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x7a], v[54], v[55]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x7b], v[56], v[57]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x7c], v[58], v[59]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x7d], v[60], v[61]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x4a:0x4d], v[0x56:0x59], a[0x68:0x6b], v[0x4a:0x4d]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[0x76], v[0x78]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[0x77], v[0x79]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[0x7a], v[0x7c]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[0x7b], v[0x7d]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x4a:0x4d], v[0x5a:0x5d], a[0x6c:0x6f], v[0x4a:0x4d]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0x80:0x8f], v[0x5e:0x61], v[0x76:0x79], v[0x80:0x8f]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0:1], v1 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[2:3], v1 offset:0x1000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[4:5], v1 offset:0x200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[6:7], v1 offset:0x1200
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0x90:0x9f], v[0x5e:0x61], v[0x7a:0x7d], v[0x90:0x9f]
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[62], v[0x7f], v[62] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[63], v[0x7f], v[63] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[64], v[0x7f], v[64] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x41], v[0x7f], v[0x41] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x42], v[0x7f], v[0x42] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x43], v[0x7f], v[0x43] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x44], v[0x7f], v[0x44] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x45], v[0x7f], v[0x45] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0xa0:0xaf], v[0x62:0x65], v[0x76:0x79], v[0xa0:0xaf]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[8:9], v1 offset:0x2000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[10:11], v1 offset:0x3000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[12:13], v1 offset:0x2200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[14:15], v1 offset:0x3200
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0xb0:0xbf], v[0x62:0x65], v[0x7a:0x7d], v[0xb0:0xbf]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[62], v[46], v[62]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[63], v[47], v[63]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[64], v[48], v[64]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x41], v[49], v[0x41]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x42], v[50], v[0x42]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x43], v[51], v[0x43]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x44], v[52], v[0x44]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x45], v[53], v[0x45]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[62], v[62], v[63]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[63], v[64], v[0x41]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[64], v[0x42], v[0x43]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x41], v[0x44], v[0x45]
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x46], v[0x7f], v[0x46] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x47], v[0x7f], v[0x47] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x48], v[0x7f], v[0x48] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x49], v[0x7f], v[0x49] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0xc0:0xcf], v[0x66:0x69], v[0x76:0x79], v[0xc0:0xcf]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[16:17], v1 offset:0x4000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[18:19], v1 offset:0x5000
	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v9, v[62:63], offset:0
	;;#ASMEND
	s_addk_i32 s22, 0xc0
	;;#ASMSTART
	ds_write_b64 v9, v[64:0x41], offset:0x200
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0xd0:0xdf], v[0x66:0x69], v[0x7a:0x7d], v[0xd0:0xdf]
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x4a], v[0x7f], v[0x4a] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x4b], v[0x7f], v[0x4b] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x4c], v[0x7f], v[0x4c] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x4d], v[0x7f], v[0x4d] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x46], v[54], v[0x46]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x47], v[55], v[0x47]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x48], v[56], v[0x48]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x49], v[57], v[0x49]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x4a], v[58], v[0x4a]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x4b], v[59], v[0x4b]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x4c], v[60], v[0x4c]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x4d], v[61], v[0x4d]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x42], v[0x46], v[0x47]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x43], v[0x48], v[0x49]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x44], v[0x4a], v[0x4b]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x45], v[0x4c], v[0x4d]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0xe0:0xef], v[0x6a:0x6d], v[0x76:0x79], v[0xe0:0xef]
	;;#ASMEND
	v_add_u32_e32 v2, s22, v13
	;;#ASMSTART
	ds_read_b32 v[0x7e], v2 offset:0

	;;#ASMEND
	s_addk_i32 s23, 0xc0
	v_add_u32_e32 v2, s23, v13
	;;#ASMSTART
	ds_read_b32 v[0x7f], v2 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v9, v[0x42:0x43], offset:0x400
	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v9, v[0x44:0x45], offset:0x600
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0xf0:0xff], v[0x6a:0x6d], v[0x7a:0x7d], v[0xf0:0xff]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[62], v[64]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[63], v[0x41]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[0x42], v[0x44]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[0x43], v[0x45]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(12)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0x80:0x8f], a[0x70:0x73], v[62:0x41], a[0x80:0x8f]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[20:21], v1 offset:0x4200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[22:23], v1 offset:0x5200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[24:25], v1 offset:0x6000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[26:27], v1 offset:0x7000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[28:29], v1 offset:0x6200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[30:31], v1 offset:0x7200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[32:33], v1 offset:0x8000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[34:35], v1 offset:0x9000
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0x90:0x9f], a[0x70:0x73], v[0x42:0x45], a[0x90:0x9f]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0xa0:0xaf], a[0x74:0x77], v[62:0x41], a[0xa0:0xaf]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[30:31], v4 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[32:33], v4 offset:0x200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[34:35], v4 offset:0x400
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[36:37], v4 offset:0x600
	;;#ASMEND
	s_add_i32 s0, s0, s13
	;;#ASMSTART
	ds_read_b64_tr_b16 v[38:39], v4 offset:0x800
	;;#ASMEND
	s_ashr_i32 s1, s0, 31
	;;#ASMSTART
	ds_read_b64_tr_b16 v[40:41], v4 offset:0xa00
	;;#ASMEND
	s_lshl_b64 s[8:9], s[0:1], 1
	;;#ASMSTART
	ds_read_b64_tr_b16 v[42:43], v4 offset:0xc00
	;;#ASMEND
	s_add_u32 s8, s6, s8
	;;#ASMSTART
	ds_read_b64_tr_b16 v[44:45], v4 offset:0xe00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0xb0:0xbf], a[0x74:0x77], v[0x42:0x45], a[0xb0:0xbf]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x7e], 0x3fb8aa3b, v[0x7e]
	;;#ASMEND
	s_addc_u32 s9, s7, s9
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x6e], v[0x6e], v[0x6f]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x6f], v[0x70], v[0x71]
	;;#ASMEND
	;;#ASMSTART
	buffer_atomic_pk_add_bf16 v[0x6e], v5, s[8:11], 0 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_atomic_pk_add_bf16 v[0x6f], v6, s[8:11], 0 offen
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0xc0:0xcf], a[0x78:0x7b], v[62:0x41], a[0xc0:0xcf]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[46:47], v4 offset:0x1000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[48:49], v4 offset:0x1200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[50:51], v4 offset:0x1400
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[52:53], v4 offset:0x1600
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[36:37], v1 offset:0x8200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[38:39], v1 offset:0x9200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[40:41], v1 offset:0xa000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[42:43], v1 offset:0xb000
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0xd0:0xdf], a[0x78:0x7b], v[0x42:0x45], a[0xd0:0xdf]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x72], v[0x72], v[0x73]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x73], v[0x74], v[0x75]
	;;#ASMEND
	;;#ASMSTART
	buffer_atomic_pk_add_bf16 v[0x72], v7, s[8:11], 0 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_atomic_pk_add_bf16 v[0x73], v8, s[8:11], 0 offen
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0xe0:0xef], a[0x7c:0x7f], v[62:0x41], a[0xe0:0xef]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[54:55], v4 offset:0x1800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[56:57], v4 offset:0x1a00
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[58:59], v4 offset:0x1c00
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[60:61], v4 offset:0x1e00
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[44:45], v1 offset:0xa200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[46:47], v1 offset:0xb200
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0xf0:0xff], a[0x7c:0x7f], v[0x42:0x45], a[0xf0:0xff]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(6)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], a[0:3], v[30:33], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[62:63], v1 offset:0xc000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[64:0x41], v1 offset:0xd000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x42:0x43], v1 offset:0xc200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x44:0x45], v1 offset:0xd200
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], a[8:11], v[34:37], v[0x6e:0x71]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], a[16:19], v[38:41], v[0x6e:0x71]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x46:0x47], v1 offset:0xe000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x48:0x49], v1 offset:0xf000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x4a:0x4b], v1 offset:0xe200
	;;#ASMEND
	s_addk_i32 s19, 0x3000
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x4c:0x4d], v1 offset:0xf200
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], a[24:27], v[42:45], v[0x6e:0x71]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], a[32:35], v[46:49], v[0x6e:0x71]
	;;#ASMEND
	v_add_u32_e32 v2, s19, v11
	;;#ASMSTART
	ds_read_b128 a[0x70:0x73], v2 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[0x74:0x77], v2 offset:0x400
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], a[40:43], v[50:53], v[0x6e:0x71]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], v[62:0x41], v[54:57], v[0x6e:0x71]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[0x78:0x7b], v2 offset:0x800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[0x7c:0x7f], v2 offset:0xc00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], v[0x46:0x49], v[58:61], v[0x6e:0x71]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], a[4:7], v[30:33], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[0:3], v10 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[4:7], v10 offset:0x400
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], a[12:15], v[34:37], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], a[20:23], v[38:41], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[8:11], v10 offset:0x800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[12:15], v10 offset:0xc00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], a[28:31], v[42:45], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x6e], 0x3db504f3, v[0x6e]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x6f], 0x3db504f3, v[0x6f]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], a[36:39], v[46:49], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[16:19], v10 offset:0x1000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[20:23], v10 offset:0x1400
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], a[44:47], v[50:53], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x70], 0x3db504f3, v[0x70]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x71], 0x3db504f3, v[0x71]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(10)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], v[0x42:0x45], v[54:57], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[24:27], v10 offset:0x1800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[28:31], v10 offset:0x1c00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], v[0x4a:0x4d], v[58:61], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(2)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[46:49], a[0x70:0x73], a[0:3], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[32:35], v10 offset:0x2000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[36:39], v10 offset:0x2400
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[46:49], a[0x74:0x77], a[4:7], v[46:49]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[46:49], a[0x78:0x7b], a[8:11], v[46:49]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[40:43], v10 offset:0x2800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[44:47], v10 offset:0x2c00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[46:49], a[0x7c:0x7f], a[12:15], v[46:49]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x72], 0x3db504f3, v[0x72]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x73], 0x3db504f3, v[0x73]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[50:53], a[0x70:0x73], a[16:19], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[62:0x41], v10 offset:0x3000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[0x42:0x45], v10 offset:0x3400
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[50:53], a[0x74:0x77], a[20:23], v[50:53]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x74], 0x3db504f3, v[0x74]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x75], 0x3db504f3, v[0x75]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[50:53], a[0x78:0x7b], a[24:27], v[50:53]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[0x46:0x49], v10 offset:0x3800
	;;#ASMEND
	s_addk_i32 s17, 0x3000
	;;#ASMSTART
	ds_read_b128 v[0x4a:0x4d], v10 offset:0x3c00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[50:53], a[0x7c:0x7f], a[28:31], v[50:53]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[46], 0x3e0293ee, v[46]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[47], 0x3e0293ee, v[47]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[48], 0x3e0293ee, v[48]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[49], 0x3e0293ee, v[49]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(6)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[54:57], a[0x70:0x73], a[32:35], 0
	;;#ASMEND
	v_add_u32_e32 v2, s17, v11
	;;#ASMSTART
	ds_read_b128 v[0x4e:0x51], v2 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[0x52:0x55], v2 offset:0x400
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[54:57], a[0x74:0x77], a[36:39], v[54:57]
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[46], v[0x7e], v[46] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[47], v[0x7e], v[47] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[48], v[0x7e], v[48] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[49], v[0x7e], v[49] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(6)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[54:57], a[0x78:0x7b], a[40:43], v[54:57]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[0x56:0x59], v2 offset:0x800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[0x5a:0x5d], v2 offset:0xc00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[54:57], a[0x7c:0x7f], a[44:47], v[54:57]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[50], 0x3e0293ee, v[50]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[51], 0x3e0293ee, v[51]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[52], 0x3e0293ee, v[52]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[53], 0x3e0293ee, v[53]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(6)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[58:61], a[0x70:0x73], v[62:0x41], 0
	;;#ASMEND
	v_add_u32_e32 v2, s17, v12
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x5e:0x5f], v2 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x60:0x61], v2 offset:0x100
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x62:0x63], v2 offset:0x400
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x64:0x65], v2 offset:0x500
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[58:61], a[0x74:0x77], v[0x42:0x45], v[58:61]
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[50], v[0x7e], v[50] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[51], v[0x7e], v[51] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[52], v[0x7e], v[52] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[53], v[0x7e], v[53] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[58:61], a[0x78:0x7b], v[0x46:0x49], v[58:61]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x66:0x67], v2 offset:0x800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x68:0x69], v2 offset:0x900
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x6a:0x6b], v2 offset:0xc00
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x6c:0x6d], v2 offset:0xd00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[58:61], a[0x7c:0x7f], v[0x4a:0x4d], v[58:61]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[54], 0x3e0293ee, v[54]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[55], 0x3e0293ee, v[55]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[56], 0x3e0293ee, v[56]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[57], 0x3e0293ee, v[57]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[62:0x41], v[0x4e:0x51], a[48:51], 0
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[54], v[0x7e], v[54] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[55], v[0x7e], v[55] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[56], v[0x7e], v[56] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[57], v[0x7e], v[57] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[62:0x41], v[0x52:0x55], a[52:55], v[62:0x41]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[46], v[46]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[47], v[47]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[48], v[48]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[49], v[49]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[62:0x41], v[0x56:0x59], a[56:59], v[62:0x41]
	;;#ASMEND
	v_add_u32_e32 v2, s19, v12
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x70:0x71], v2 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x72:0x73], v2 offset:0x100
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[62:0x41], v[0x5a:0x5d], a[60:63], v[62:0x41]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[50], v[50]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[51], v[51]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[52], v[52]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[53], v[53]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x42:0x45], v[0x4e:0x51], a[64:0x43], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x74:0x75], v2 offset:0x400
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x76:0x77], v2 offset:0x500
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x42:0x45], v[0x52:0x55], a[0x44:0x47], v[0x42:0x45]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[58], 0x3e0293ee, v[58]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[59], 0x3e0293ee, v[59]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[60], 0x3e0293ee, v[60]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[61], 0x3e0293ee, v[61]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x42:0x45], v[0x56:0x59], a[0x48:0x4b], v[0x42:0x45]
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[58], v[0x7e], v[58] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[59], v[0x7e], v[59] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[60], v[0x7e], v[60] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[61], v[0x7e], v[61] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x42:0x45], v[0x5a:0x5d], a[0x4c:0x4f], v[0x42:0x45]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x76], v[46], v[47]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x77], v[48], v[49]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x46:0x49], v[0x4e:0x51], a[0x50:0x53], 0
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[54], v[54]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[55], v[55]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[56], v[56]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[57], v[57]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x46:0x49], v[0x52:0x55], a[0x54:0x57], v[0x46:0x49]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x78], v[50], v[51]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x79], v[52], v[53]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x46:0x49], v[0x56:0x59], a[0x58:0x5b], v[0x46:0x49]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x78:0x79], v2 offset:0x800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x7a:0x7b], v2 offset:0x900
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x46:0x49], v[0x5a:0x5d], a[0x5c:0x5f], v[0x46:0x49]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[58], v[58]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[59], v[59]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[60], v[60]
	;;#ASMEND
	;;#ASMSTART
	v_exp_f32_e32 v[61], v[61]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x4a:0x4d], v[0x4e:0x51], a[0x60:0x63], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x7c:0x7d], v2 offset:0xc00
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0x7e:0x7f], v2 offset:0xd00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x4a:0x4d], v[0x52:0x55], a[0x64:0x67], v[0x4a:0x4d]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x7a], v[54], v[55]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x7b], v[56], v[57]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x7c], v[58], v[59]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x7d], v[60], v[61]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x4a:0x4d], v[0x56:0x59], a[0x68:0x6b], v[0x4a:0x4d]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[0x76], v[0x78]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[0x77], v[0x79]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[0x7a], v[0x7c]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[0x7b], v[0x7d]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x4a:0x4d], v[0x5a:0x5d], a[0x6c:0x6f], v[0x4a:0x4d]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0x80:0x8f], v[0x5e:0x61], v[0x76:0x79], v[0x80:0x8f]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[0:1], v1 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[2:3], v1 offset:0x1000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[4:5], v1 offset:0x200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[6:7], v1 offset:0x1200
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0x90:0x9f], v[0x5e:0x61], v[0x7a:0x7d], v[0x90:0x9f]
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[62], v[0x7f], v[62] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[63], v[0x7f], v[63] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[64], v[0x7f], v[64] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x41], v[0x7f], v[0x41] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x42], v[0x7f], v[0x42] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x43], v[0x7f], v[0x43] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x44], v[0x7f], v[0x44] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x45], v[0x7f], v[0x45] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0xa0:0xaf], v[0x62:0x65], v[0x76:0x79], v[0xa0:0xaf]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[8:9], v1 offset:0x2000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[10:11], v1 offset:0x3000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[12:13], v1 offset:0x2200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[14:15], v1 offset:0x3200
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0xb0:0xbf], v[0x62:0x65], v[0x7a:0x7d], v[0xb0:0xbf]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[62], v[46], v[62]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[63], v[47], v[63]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[64], v[48], v[64]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x41], v[49], v[0x41]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x42], v[50], v[0x42]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x43], v[51], v[0x43]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x44], v[52], v[0x44]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x45], v[53], v[0x45]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[62], v[62], v[63]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[63], v[64], v[0x41]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[64], v[0x42], v[0x43]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x41], v[0x44], v[0x45]
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x46], v[0x7f], v[0x46] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x47], v[0x7f], v[0x47] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x48], v[0x7f], v[0x48] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x49], v[0x7f], v[0x49] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0xc0:0xcf], v[0x66:0x69], v[0x76:0x79], v[0xc0:0xcf]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[16:17], v1 offset:0x4000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[18:19], v1 offset:0x5000
	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v9, v[62:63], offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v9, v[64:0x41], offset:0x200
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0xd0:0xdf], v[0x66:0x69], v[0x7a:0x7d], v[0xd0:0xdf]
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x4a], v[0x7f], v[0x4a] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x4b], v[0x7f], v[0x4b] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x4c], v[0x7f], v[0x4c] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_subrev_f32_dpp v[0x4d], v[0x7f], v[0x4d] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x46], v[54], v[0x46]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x47], v[55], v[0x47]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x48], v[56], v[0x48]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x49], v[57], v[0x49]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x4a], v[58], v[0x4a]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x4b], v[59], v[0x4b]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x4c], v[60], v[0x4c]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x4d], v[61], v[0x4d]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x42], v[0x46], v[0x47]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x43], v[0x48], v[0x49]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x44], v[0x4a], v[0x4b]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x45], v[0x4c], v[0x4d]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0xe0:0xef], v[0x6a:0x6d], v[0x76:0x79], v[0xe0:0xef]
	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v9, v[0x42:0x43], offset:0x400
	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v9, v[0x44:0x45], offset:0x600
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 v[0xf0:0xff], v[0x6a:0x6d], v[0x7a:0x7d], v[0xf0:0xff]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[62], v[64]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[63], v[0x41]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[0x42], v[0x44]
	;;#ASMEND
	;;#ASMSTART
	v_permlane16_swap_b32_e32 v[0x43], v[0x45]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(12)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0x80:0x8f], a[0x70:0x73], v[62:0x41], a[0x80:0x8f]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[20:21], v1 offset:0x4200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[22:23], v1 offset:0x5200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[24:25], v1 offset:0x6000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[26:27], v1 offset:0x7000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[28:29], v1 offset:0x6200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[30:31], v1 offset:0x7200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[32:33], v1 offset:0x8000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[34:35], v1 offset:0x9000
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0x90:0x9f], a[0x70:0x73], v[0x42:0x45], a[0x90:0x9f]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0xa0:0xaf], a[0x74:0x77], v[62:0x41], a[0xa0:0xaf]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[30:31], v4 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[32:33], v4 offset:0x200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[34:35], v4 offset:0x400
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[36:37], v4 offset:0x600
	;;#ASMEND
	s_add_i32 s0, s0, s13
	;;#ASMSTART
	ds_read_b64_tr_b16 v[38:39], v4 offset:0x800
	;;#ASMEND
	s_ashr_i32 s1, s0, 31
	;;#ASMSTART
	ds_read_b64_tr_b16 v[40:41], v4 offset:0xa00
	;;#ASMEND
	s_lshl_b64 s[0:1], s[0:1], 1
	;;#ASMSTART
	ds_read_b64_tr_b16 v[42:43], v4 offset:0xc00
	;;#ASMEND
	s_add_u32 s8, s6, s0
	;;#ASMSTART
	ds_read_b64_tr_b16 v[44:45], v4 offset:0xe00
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0xb0:0xbf], a[0x74:0x77], v[0x42:0x45], a[0xb0:0xbf]
	;;#ASMEND
	s_addc_u32 s9, s7, s1
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x6e], v[0x6e], v[0x6f]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x6f], v[0x70], v[0x71]
	;;#ASMEND
	;;#ASMSTART
	buffer_atomic_pk_add_bf16 v[0x6e], v5, s[8:11], 0 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_atomic_pk_add_bf16 v[0x6f], v6, s[8:11], 0 offen
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0xc0:0xcf], a[0x78:0x7b], v[62:0x41], a[0xc0:0xcf]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[46:47], v4 offset:0x1000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[48:49], v4 offset:0x1200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[50:51], v4 offset:0x1400
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[52:53], v4 offset:0x1600
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[36:37], v1 offset:0x8200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[38:39], v1 offset:0x9200
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[40:41], v1 offset:0xa000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[42:43], v1 offset:0xb000
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0xd0:0xdf], a[0x78:0x7b], v[0x42:0x45], a[0xd0:0xdf]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x72], v[0x72], v[0x73]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x73], v[0x74], v[0x75]
	;;#ASMEND
	;;#ASMSTART
	buffer_atomic_pk_add_bf16 v[0x72], v7, s[8:11], 0 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_atomic_pk_add_bf16 v[0x73], v8, s[8:11], 0 offen
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0xe0:0xef], a[0x7c:0x7f], v[62:0x41], a[0xe0:0xef]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[54:55], v4 offset:0x1800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[56:57], v4 offset:0x1a00
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[58:59], v4 offset:0x1c00
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[60:61], v4 offset:0x1e00
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 a[44:45], v1 offset:0xa200
	;;#ASMEND
	s_mul_i32 s0, s4, s16
	;;#ASMSTART
	ds_read_b64_tr_b16 a[46:47], v1 offset:0xb200
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_32x32x16_bf16 a[0xf0:0xff], a[0x7c:0x7f], v[0x42:0x45], a[0xf0:0xff]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(6)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], a[0:3], v[30:33], 0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[62:63], v1 offset:0xc000
	;;#ASMEND
	s_mul_i32 s0, s0, s18
	;;#ASMSTART
	ds_read_b64_tr_b16 v[64:0x41], v1 offset:0xd000
	;;#ASMEND
	s_add_i32 s0, s0, s2
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x42:0x43], v1 offset:0xc200
	;;#ASMEND
	s_mul_i32 s0, s0, s34
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x44:0x45], v1 offset:0xd200
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], a[8:11], v[34:37], v[0x6e:0x71]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], a[16:19], v[38:41], v[0x6e:0x71]
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x46:0x47], v1 offset:0xe000
	;;#ASMEND
	s_ashr_i32 s1, s0, 31
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x48:0x49], v1 offset:0xf000
	;;#ASMEND
	s_lshl_b64 s[0:1], s[0:1], 1
	s_mul_i32 s8, s18, s34
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x4a:0x4b], v1 offset:0xe200
	;;#ASMEND
	s_add_u32 s16, s48, s0
	v_mul_lo_u32 v4, s8, v3
	v_and_b32_e32 v23, 31, v0
	v_lshrrev_b32_e32 v0, 3, v0
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0x4c:0x4d], v1 offset:0xf200
	;;#ASMEND
	s_addc_u32 s17, s49, s1
	v_and_b32_e32 v2, 4, v0
	v_mad_u64_u32 v[0:1], s[0:1], v23, s8, v[4:5]
	v_add_lshl_u32 v1, v0, v2, 1
	v_or_b32_e32 v10, 8, v2
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], a[24:27], v[42:45], v[0x6e:0x71]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], a[32:35], v[46:49], v[0x6e:0x71]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], a[40:43], v[50:53], v[0x6e:0x71]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], v[62:0x41], v[54:57], v[0x6e:0x71]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x6e:0x71], v[0x46:0x49], v[58:61], v[0x6e:0x71]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], a[4:7], v[30:33], 0
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], a[12:15], v[34:37], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], a[20:23], v[38:41], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], a[28:31], v[42:45], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], a[36:39], v[46:49], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], a[44:47], v[50:53], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(10)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], v[0x42:0x45], v[54:57], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[0x72:0x75], v[0x4a:0x4d], v[58:61], v[0x72:0x75]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(2)
	;;#ASMEND
	s_mov_b32 s18, -1
	s_mov_b32 s19, s11
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x80], v[0x80], v[0x81]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x81], v[0x82], v[0x83]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x82], v[0x84], v[0x85]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x83], v[0x86], v[0x87]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x84], v[0x88], v[0x89]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x85], v[0x8a], v[0x8b]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x86], v[0x8c], v[0x8d]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x87], v[0x8e], v[0x8f]
	;;#ASMEND
	;;#ASMSTART
	buffer_store_dwordx2 v[0x80:0x81], v1, s[16:19], 0 offen
	;;#ASMEND
	v_add_lshl_u32 v1, v0, v10, 1
	v_or_b32_e32 v11, 16, v2
	;;#ASMSTART
	buffer_store_dwordx2 v[0x82:0x83], v1, s[16:19], 0 offen
	;;#ASMEND
	v_add_lshl_u32 v1, v0, v11, 1
	v_or_b32_e32 v12, 24, v2
	;;#ASMSTART
	buffer_store_dwordx2 v[0x84:0x85], v1, s[16:19], 0 offen
	;;#ASMEND
	v_add_lshl_u32 v1, v0, v12, 1
	v_or_b32_e32 v4, 32, v2
	;;#ASMSTART
	buffer_store_dwordx2 v[0x86:0x87], v1, s[16:19], 0 offen
	;;#ASMEND
	v_add_lshl_u32 v1, v0, v4, 1
	v_or_b32_e32 v14, 40, v2
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xa0], v[0xa0], v[0xa1]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xa1], v[0xa2], v[0xa3]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xa2], v[0xa4], v[0xa5]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xa3], v[0xa6], v[0xa7]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xa4], v[0xa8], v[0xa9]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xa5], v[0xaa], v[0xab]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xa6], v[0xac], v[0xad]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xa7], v[0xae], v[0xaf]
	;;#ASMEND
	;;#ASMSTART
	buffer_store_dwordx2 v[0xa0:0xa1], v1, s[16:19], 0 offen
	;;#ASMEND
	v_add_lshl_u32 v1, v0, v14, 1
	v_or_b32_e32 v15, 48, v2
	;;#ASMSTART
	buffer_store_dwordx2 v[0xa2:0xa3], v1, s[16:19], 0 offen
	;;#ASMEND
	v_add_lshl_u32 v1, v0, v15, 1
	v_or_b32_e32 v16, 56, v2
	;;#ASMSTART
	buffer_store_dwordx2 v[0xa4:0xa5], v1, s[16:19], 0 offen
	;;#ASMEND
	v_add_lshl_u32 v1, v0, v16, 1
	v_or_b32_e32 v9, 64, v2
	;;#ASMSTART
	buffer_store_dwordx2 v[0xa6:0xa7], v1, s[16:19], 0 offen
	;;#ASMEND
	v_add_lshl_u32 v1, v0, v9, 1
	v_or_b32_e32 v17, 0x48, v2
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xc0], v[0xc0], v[0xc1]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xc1], v[0xc2], v[0xc3]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xc2], v[0xc4], v[0xc5]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xc3], v[0xc6], v[0xc7]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xc4], v[0xc8], v[0xc9]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xc5], v[0xca], v[0xcb]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xc6], v[0xcc], v[0xcd]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xc7], v[0xce], v[0xcf]
	;;#ASMEND
	;;#ASMSTART
	buffer_store_dwordx2 v[0xc0:0xc1], v1, s[16:19], 0 offen
	;;#ASMEND
	v_add_lshl_u32 v1, v0, v17, 1
	v_or_b32_e32 v18, 0x50, v2
	;;#ASMSTART
	buffer_store_dwordx2 v[0xc2:0xc3], v1, s[16:19], 0 offen
	;;#ASMEND
	v_add_lshl_u32 v1, v0, v18, 1
	v_or_b32_e32 v19, 0x58, v2
	;;#ASMSTART
	buffer_store_dwordx2 v[0xc4:0xc5], v1, s[16:19], 0 offen
	;;#ASMEND
	v_add_lshl_u32 v1, v0, v19, 1
	v_or_b32_e32 v13, 0x60, v2
	;;#ASMSTART
	buffer_store_dwordx2 v[0xc6:0xc7], v1, s[16:19], 0 offen
	;;#ASMEND
	v_add_lshl_u32 v1, v0, v13, 1
	v_or_b32_e32 v20, 0x68, v2
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xe0], v[0xe0], v[0xe1]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xe1], v[0xe2], v[0xe3]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xe2], v[0xe4], v[0xe5]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xe3], v[0xe6], v[0xe7]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xe4], v[0xe8], v[0xe9]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xe5], v[0xea], v[0xeb]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xe6], v[0xec], v[0xed]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xe7], v[0xee], v[0xef]
	;;#ASMEND
	;;#ASMSTART
	buffer_store_dwordx2 v[0xe0:0xe1], v1, s[16:19], 0 offen
	;;#ASMEND
	v_add_lshl_u32 v1, v0, v20, 1
	v_or_b32_e32 v21, 0x70, v2
	;;#ASMSTART
	buffer_store_dwordx2 v[0xe2:0xe3], v1, s[16:19], 0 offen
	;;#ASMEND
	v_add_lshl_u32 v1, v0, v21, 1
	v_or_b32_e32 v22, 0x78, v2
	;;#ASMSTART
	buffer_store_dwordx2 v[0xe4:0xe5], v1, s[16:19], 0 offen
	;;#ASMEND
	v_add_lshl_u32 v1, v0, v22, 1
	v_lshl_add_u32 v0, s8, 5, v0
	;;#ASMSTART
	buffer_store_dwordx2 v[0xe6:0xe7], v1, s[16:19], 0 offen
	;;#ASMEND
	v_add_lshl_u32 v1, v0, v2, 1
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x90], v[0x90], v[0x91]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x91], v[0x92], v[0x93]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x92], v[0x94], v[0x95]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x93], v[0x96], v[0x97]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x94], v[0x98], v[0x99]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x95], v[0x9a], v[0x9b]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x96], v[0x9c], v[0x9d]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x97], v[0x9e], v[0x9f]
	;;#ASMEND
	;;#ASMSTART
	buffer_store_dwordx2 v[0x90:0x91], v1, s[16:19], 0 offen
	;;#ASMEND
	v_add_lshl_u32 v1, v0, v10, 1
	;;#ASMSTART
	buffer_store_dwordx2 v[0x92:0x93], v1, s[16:19], 0 offen
	;;#ASMEND
	v_add_lshl_u32 v1, v0, v11, 1
	;;#ASMSTART
	buffer_store_dwordx2 v[0x94:0x95], v1, s[16:19], 0 offen
	;;#ASMEND
	v_add_lshl_u32 v1, v0, v12, 1
	;;#ASMSTART
	buffer_store_dwordx2 v[0x96:0x97], v1, s[16:19], 0 offen
	;;#ASMEND
	v_add_lshl_u32 v1, v0, v4, 1
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xb0], v[0xb0], v[0xb1]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xb1], v[0xb2], v[0xb3]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xb2], v[0xb4], v[0xb5]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xb3], v[0xb6], v[0xb7]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xb4], v[0xb8], v[0xb9]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xb5], v[0xba], v[0xbb]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xb6], v[0xbc], v[0xbd]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xb7], v[0xbe], v[0xbf]
	;;#ASMEND
	;;#ASMSTART
	buffer_store_dwordx2 v[0xb0:0xb1], v1, s[16:19], 0 offen
	;;#ASMEND
	v_add_lshl_u32 v1, v0, v14, 1
	;;#ASMSTART
	buffer_store_dwordx2 v[0xb2:0xb3], v1, s[16:19], 0 offen
	;;#ASMEND
	v_add_lshl_u32 v1, v0, v15, 1
	;;#ASMSTART
	buffer_store_dwordx2 v[0xb4:0xb5], v1, s[16:19], 0 offen
	;;#ASMEND
	v_add_lshl_u32 v1, v0, v16, 1
	;;#ASMSTART
	buffer_store_dwordx2 v[0xb6:0xb7], v1, s[16:19], 0 offen
	;;#ASMEND
	v_add_lshl_u32 v1, v0, v9, 1
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xd0], v[0xd0], v[0xd1]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xd1], v[0xd2], v[0xd3]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xd2], v[0xd4], v[0xd5]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xd3], v[0xd6], v[0xd7]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xd4], v[0xd8], v[0xd9]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xd5], v[0xda], v[0xdb]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xd6], v[0xdc], v[0xdd]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xd7], v[0xde], v[0xdf]
	;;#ASMEND
	;;#ASMSTART
	buffer_store_dwordx2 v[0xd0:0xd1], v1, s[16:19], 0 offen
	;;#ASMEND
	v_add_lshl_u32 v1, v0, v17, 1
	s_mul_i32 s0, s4, s12
	;;#ASMSTART
	buffer_store_dwordx2 v[0xd2:0xd3], v1, s[16:19], 0 offen
	;;#ASMEND
	v_add_lshl_u32 v1, v0, v18, 1
	s_mul_i32 s0, s0, s14
	;;#ASMSTART
	buffer_store_dwordx2 v[0xd4:0xd5], v1, s[16:19], 0 offen
	;;#ASMEND
	v_add_lshl_u32 v1, v0, v19, 1
	s_add_i32 s0, s0, s2
	;;#ASMSTART
	buffer_store_dwordx2 v[0xd6:0xd7], v1, s[16:19], 0 offen
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xf0], v[0xf0], v[0xf1]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xf1], v[0xf2], v[0xf3]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xf2], v[0xf4], v[0xf5]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xf3], v[0xf6], v[0xf7]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xf4], v[0xf8], v[0xf9]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xf5], v[0xfa], v[0xfb]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xf6], v[0xfc], v[0xfd]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xf7], v[0xfe], v[0xff]
	;;#ASMEND
	v_add_lshl_u32 v1, v0, v13, 1
	;;#ASMSTART
	buffer_store_dwordx2 v[0xf0:0xf1], v1, s[16:19], 0 offen
	;;#ASMEND
	s_mul_i32 s0, s0, s24
	v_add_lshl_u32 v1, v0, v20, 1
	;;#ASMSTART
	buffer_store_dwordx2 v[0xf2:0xf3], v1, s[16:19], 0 offen
	;;#ASMEND
	s_ashr_i32 s1, s0, 31
	v_add_lshl_u32 v1, v0, v21, 1
	;;#ASMSTART
	buffer_store_dwordx2 v[0xf4:0xf5], v1, s[16:19], 0 offen
	;;#ASMEND
	v_add_lshl_u32 v0, v0, v22, 1
	s_lshl_b64 s[0:1], s[0:1], 1
	s_mul_i32 s2, s14, s24
	;;#ASMSTART
	buffer_store_dwordx2 v[0xf6:0xf7], v0, s[16:19], 0 offen
	;;#ASMEND
	s_add_u32 s16, s20, s0
	v_mul_lo_u32 v0, s2, v3
	s_addc_u32 s17, s21, s1
	v_mad_u64_u32 v[0:1], s[0:1], v23, s2, v[0:1]
	v_add_lshl_u32 v1, v0, v2, 1
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	s_barrier
	;;#ASMSTART
	v_accvgpr_read_b32 v[0x80], a[0x80]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0x81], a[0x81]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0x82], a[0x82]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0x83], a[0x83]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0x84], a[0x84]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0x85], a[0x85]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0x86], a[0x86]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0x87], a[0x87]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0x88], a[0x88]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0x89], a[0x89]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0x8a], a[0x8a]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0x8b], a[0x8b]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0x8c], a[0x8c]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0x8d], a[0x8d]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0x8e], a[0x8e]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0x8f], a[0x8f]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0x90], a[0x90]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0x91], a[0x91]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0x92], a[0x92]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0x93], a[0x93]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0x94], a[0x94]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0x95], a[0x95]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0x96], a[0x96]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0x97], a[0x97]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0x98], a[0x98]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0x99], a[0x99]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0x9a], a[0x9a]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0x9b], a[0x9b]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0x9c], a[0x9c]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0x9d], a[0x9d]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0x9e], a[0x9e]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0x9f], a[0x9f]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xa0], a[0xa0]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xa1], a[0xa1]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xa2], a[0xa2]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xa3], a[0xa3]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xa4], a[0xa4]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xa5], a[0xa5]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xa6], a[0xa6]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xa7], a[0xa7]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xa8], a[0xa8]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xa9], a[0xa9]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xaa], a[0xaa]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xab], a[0xab]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xac], a[0xac]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xad], a[0xad]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xae], a[0xae]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xaf], a[0xaf]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xb0], a[0xb0]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xb1], a[0xb1]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xb2], a[0xb2]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xb3], a[0xb3]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xb4], a[0xb4]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xb5], a[0xb5]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xb6], a[0xb6]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xb7], a[0xb7]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xb8], a[0xb8]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xb9], a[0xb9]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xba], a[0xba]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xbb], a[0xbb]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xbc], a[0xbc]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xbd], a[0xbd]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xbe], a[0xbe]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xbf], a[0xbf]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xc0], a[0xc0]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xc1], a[0xc1]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xc2], a[0xc2]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xc3], a[0xc3]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xc4], a[0xc4]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xc5], a[0xc5]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xc6], a[0xc6]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xc7], a[0xc7]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xc8], a[0xc8]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xc9], a[0xc9]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xca], a[0xca]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xcb], a[0xcb]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xcc], a[0xcc]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xcd], a[0xcd]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xce], a[0xce]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xcf], a[0xcf]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xd0], a[0xd0]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xd1], a[0xd1]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xd2], a[0xd2]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xd3], a[0xd3]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xd4], a[0xd4]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xd5], a[0xd5]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xd6], a[0xd6]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xd7], a[0xd7]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xd8], a[0xd8]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xd9], a[0xd9]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xda], a[0xda]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xdb], a[0xdb]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xdc], a[0xdc]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xdd], a[0xdd]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xde], a[0xde]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xdf], a[0xdf]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xe0], a[0xe0]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xe1], a[0xe1]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xe2], a[0xe2]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xe3], a[0xe3]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xe4], a[0xe4]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xe5], a[0xe5]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xe6], a[0xe6]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xe7], a[0xe7]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xe8], a[0xe8]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xe9], a[0xe9]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xea], a[0xea]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xeb], a[0xeb]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xec], a[0xec]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xed], a[0xed]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xee], a[0xee]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xef], a[0xef]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xf0], a[0xf0]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xf1], a[0xf1]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xf2], a[0xf2]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xf3], a[0xf3]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xf4], a[0xf4]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xf5], a[0xf5]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xf6], a[0xf6]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xf7], a[0xf7]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xf8], a[0xf8]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xf9], a[0xf9]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xfa], a[0xfa]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xfb], a[0xfb]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xfc], a[0xfc]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xfd], a[0xfd]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xfe], a[0xfe]
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_read_b32 v[0xff], a[0xff]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x80], 0x3db504f3, v[0x80]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x81], 0x3db504f3, v[0x81]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x82], 0x3db504f3, v[0x82]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x83], 0x3db504f3, v[0x83]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x84], 0x3db504f3, v[0x84]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x85], 0x3db504f3, v[0x85]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x86], 0x3db504f3, v[0x86]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x87], 0x3db504f3, v[0x87]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x88], 0x3db504f3, v[0x88]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x89], 0x3db504f3, v[0x89]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x8a], 0x3db504f3, v[0x8a]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x8b], 0x3db504f3, v[0x8b]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x8c], 0x3db504f3, v[0x8c]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x8d], 0x3db504f3, v[0x8d]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x8e], 0x3db504f3, v[0x8e]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x8f], 0x3db504f3, v[0x8f]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x90], 0x3db504f3, v[0x90]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x91], 0x3db504f3, v[0x91]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x92], 0x3db504f3, v[0x92]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x93], 0x3db504f3, v[0x93]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x94], 0x3db504f3, v[0x94]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x95], 0x3db504f3, v[0x95]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x96], 0x3db504f3, v[0x96]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x97], 0x3db504f3, v[0x97]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x98], 0x3db504f3, v[0x98]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x99], 0x3db504f3, v[0x99]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x9a], 0x3db504f3, v[0x9a]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x9b], 0x3db504f3, v[0x9b]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x9c], 0x3db504f3, v[0x9c]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x9d], 0x3db504f3, v[0x9d]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x9e], 0x3db504f3, v[0x9e]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x9f], 0x3db504f3, v[0x9f]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xa0], 0x3db504f3, v[0xa0]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xa1], 0x3db504f3, v[0xa1]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xa2], 0x3db504f3, v[0xa2]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xa3], 0x3db504f3, v[0xa3]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xa4], 0x3db504f3, v[0xa4]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xa5], 0x3db504f3, v[0xa5]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xa6], 0x3db504f3, v[0xa6]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xa7], 0x3db504f3, v[0xa7]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xa8], 0x3db504f3, v[0xa8]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xa9], 0x3db504f3, v[0xa9]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xaa], 0x3db504f3, v[0xaa]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xab], 0x3db504f3, v[0xab]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xac], 0x3db504f3, v[0xac]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xad], 0x3db504f3, v[0xad]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xae], 0x3db504f3, v[0xae]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xaf], 0x3db504f3, v[0xaf]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xb0], 0x3db504f3, v[0xb0]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xb1], 0x3db504f3, v[0xb1]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xb2], 0x3db504f3, v[0xb2]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xb3], 0x3db504f3, v[0xb3]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xb4], 0x3db504f3, v[0xb4]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xb5], 0x3db504f3, v[0xb5]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xb6], 0x3db504f3, v[0xb6]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xb7], 0x3db504f3, v[0xb7]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xb8], 0x3db504f3, v[0xb8]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xb9], 0x3db504f3, v[0xb9]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xba], 0x3db504f3, v[0xba]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xbb], 0x3db504f3, v[0xbb]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xbc], 0x3db504f3, v[0xbc]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xbd], 0x3db504f3, v[0xbd]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xbe], 0x3db504f3, v[0xbe]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xbf], 0x3db504f3, v[0xbf]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xc0], 0x3db504f3, v[0xc0]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xc1], 0x3db504f3, v[0xc1]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xc2], 0x3db504f3, v[0xc2]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xc3], 0x3db504f3, v[0xc3]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xc4], 0x3db504f3, v[0xc4]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xc5], 0x3db504f3, v[0xc5]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xc6], 0x3db504f3, v[0xc6]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xc7], 0x3db504f3, v[0xc7]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xc8], 0x3db504f3, v[0xc8]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xc9], 0x3db504f3, v[0xc9]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xca], 0x3db504f3, v[0xca]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xcb], 0x3db504f3, v[0xcb]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xcc], 0x3db504f3, v[0xcc]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xcd], 0x3db504f3, v[0xcd]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xce], 0x3db504f3, v[0xce]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xcf], 0x3db504f3, v[0xcf]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xd0], 0x3db504f3, v[0xd0]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xd1], 0x3db504f3, v[0xd1]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xd2], 0x3db504f3, v[0xd2]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xd3], 0x3db504f3, v[0xd3]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xd4], 0x3db504f3, v[0xd4]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xd5], 0x3db504f3, v[0xd5]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xd6], 0x3db504f3, v[0xd6]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xd7], 0x3db504f3, v[0xd7]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xd8], 0x3db504f3, v[0xd8]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xd9], 0x3db504f3, v[0xd9]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xda], 0x3db504f3, v[0xda]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xdb], 0x3db504f3, v[0xdb]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xdc], 0x3db504f3, v[0xdc]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xdd], 0x3db504f3, v[0xdd]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xde], 0x3db504f3, v[0xde]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xdf], 0x3db504f3, v[0xdf]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xe0], 0x3db504f3, v[0xe0]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xe1], 0x3db504f3, v[0xe1]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xe2], 0x3db504f3, v[0xe2]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xe3], 0x3db504f3, v[0xe3]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xe4], 0x3db504f3, v[0xe4]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xe5], 0x3db504f3, v[0xe5]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xe6], 0x3db504f3, v[0xe6]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xe7], 0x3db504f3, v[0xe7]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xe8], 0x3db504f3, v[0xe8]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xe9], 0x3db504f3, v[0xe9]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xea], 0x3db504f3, v[0xea]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xeb], 0x3db504f3, v[0xeb]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xec], 0x3db504f3, v[0xec]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xed], 0x3db504f3, v[0xed]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xee], 0x3db504f3, v[0xee]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xef], 0x3db504f3, v[0xef]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xf0], 0x3db504f3, v[0xf0]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xf1], 0x3db504f3, v[0xf1]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xf2], 0x3db504f3, v[0xf2]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xf3], 0x3db504f3, v[0xf3]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xf4], 0x3db504f3, v[0xf4]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xf5], 0x3db504f3, v[0xf5]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xf6], 0x3db504f3, v[0xf6]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xf7], 0x3db504f3, v[0xf7]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xf8], 0x3db504f3, v[0xf8]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xf9], 0x3db504f3, v[0xf9]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xfa], 0x3db504f3, v[0xfa]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xfb], 0x3db504f3, v[0xfb]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xfc], 0x3db504f3, v[0xfc]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xfd], 0x3db504f3, v[0xfd]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xfe], 0x3db504f3, v[0xfe]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0xff], 0x3db504f3, v[0xff]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x80], v[0x80], v[0x81]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x81], v[0x82], v[0x83]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x82], v[0x84], v[0x85]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x83], v[0x86], v[0x87]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x84], v[0x88], v[0x89]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x85], v[0x8a], v[0x8b]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x86], v[0x8c], v[0x8d]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x87], v[0x8e], v[0x8f]
	;;#ASMEND
	;;#ASMSTART
	buffer_store_dwordx2 v[0x80:0x81], v1, s[16:19], 0 offen
	;;#ASMEND
	v_add_lshl_u32 v1, v0, v10, 1
	;;#ASMSTART
	buffer_store_dwordx2 v[0x82:0x83], v1, s[16:19], 0 offen
	;;#ASMEND
	v_add_lshl_u32 v1, v0, v11, 1
	;;#ASMSTART
	buffer_store_dwordx2 v[0x84:0x85], v1, s[16:19], 0 offen
	;;#ASMEND
	v_add_lshl_u32 v1, v0, v12, 1
	;;#ASMSTART
	buffer_store_dwordx2 v[0x86:0x87], v1, s[16:19], 0 offen
	;;#ASMEND
	v_add_lshl_u32 v1, v0, v4, 1
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xa0], v[0xa0], v[0xa1]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xa1], v[0xa2], v[0xa3]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xa2], v[0xa4], v[0xa5]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xa3], v[0xa6], v[0xa7]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xa4], v[0xa8], v[0xa9]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xa5], v[0xaa], v[0xab]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xa6], v[0xac], v[0xad]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xa7], v[0xae], v[0xaf]
	;;#ASMEND
	;;#ASMSTART
	buffer_store_dwordx2 v[0xa0:0xa1], v1, s[16:19], 0 offen
	;;#ASMEND
	v_add_lshl_u32 v1, v0, v14, 1
	;;#ASMSTART
	buffer_store_dwordx2 v[0xa2:0xa3], v1, s[16:19], 0 offen
	;;#ASMEND
	v_add_lshl_u32 v1, v0, v15, 1
	;;#ASMSTART
	buffer_store_dwordx2 v[0xa4:0xa5], v1, s[16:19], 0 offen
	;;#ASMEND
	v_add_lshl_u32 v1, v0, v16, 1
	;;#ASMSTART
	buffer_store_dwordx2 v[0xa6:0xa7], v1, s[16:19], 0 offen
	;;#ASMEND
	v_add_lshl_u32 v1, v0, v9, 1
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xc0], v[0xc0], v[0xc1]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xc1], v[0xc2], v[0xc3]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xc2], v[0xc4], v[0xc5]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xc3], v[0xc6], v[0xc7]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xc4], v[0xc8], v[0xc9]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xc5], v[0xca], v[0xcb]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xc6], v[0xcc], v[0xcd]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xc7], v[0xce], v[0xcf]
	;;#ASMEND
	;;#ASMSTART
	buffer_store_dwordx2 v[0xc0:0xc1], v1, s[16:19], 0 offen
	;;#ASMEND
	v_add_lshl_u32 v1, v0, v17, 1
	;;#ASMSTART
	buffer_store_dwordx2 v[0xc2:0xc3], v1, s[16:19], 0 offen
	;;#ASMEND
	v_add_lshl_u32 v1, v0, v18, 1
	;;#ASMSTART
	buffer_store_dwordx2 v[0xc4:0xc5], v1, s[16:19], 0 offen
	;;#ASMEND
	v_add_lshl_u32 v1, v0, v19, 1
	;;#ASMSTART
	buffer_store_dwordx2 v[0xc6:0xc7], v1, s[16:19], 0 offen
	;;#ASMEND
	v_add_lshl_u32 v1, v0, v13, 1
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xe0], v[0xe0], v[0xe1]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xe1], v[0xe2], v[0xe3]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xe2], v[0xe4], v[0xe5]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xe3], v[0xe6], v[0xe7]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xe4], v[0xe8], v[0xe9]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xe5], v[0xea], v[0xeb]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xe6], v[0xec], v[0xed]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xe7], v[0xee], v[0xef]
	;;#ASMEND
	;;#ASMSTART
	buffer_store_dwordx2 v[0xe0:0xe1], v1, s[16:19], 0 offen
	;;#ASMEND
	v_add_lshl_u32 v1, v0, v20, 1
	;;#ASMSTART
	buffer_store_dwordx2 v[0xe2:0xe3], v1, s[16:19], 0 offen
	;;#ASMEND
	v_add_lshl_u32 v1, v0, v21, 1
	;;#ASMSTART
	buffer_store_dwordx2 v[0xe4:0xe5], v1, s[16:19], 0 offen
	;;#ASMEND
	v_add_lshl_u32 v1, v0, v22, 1
	v_lshl_add_u32 v0, s2, 5, v0
	;;#ASMSTART
	buffer_store_dwordx2 v[0xe6:0xe7], v1, s[16:19], 0 offen
	;;#ASMEND
	v_add_lshl_u32 v1, v0, v2, 1
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x90], v[0x90], v[0x91]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x91], v[0x92], v[0x93]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x92], v[0x94], v[0x95]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x93], v[0x96], v[0x97]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x94], v[0x98], v[0x99]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x95], v[0x9a], v[0x9b]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x96], v[0x9c], v[0x9d]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x97], v[0x9e], v[0x9f]
	;;#ASMEND
	;;#ASMSTART
	buffer_store_dwordx2 v[0x90:0x91], v1, s[16:19], 0 offen
	;;#ASMEND
	v_add_lshl_u32 v1, v0, v10, 1
	;;#ASMSTART
	buffer_store_dwordx2 v[0x92:0x93], v1, s[16:19], 0 offen
	;;#ASMEND
	v_add_lshl_u32 v1, v0, v11, 1
	;;#ASMSTART
	buffer_store_dwordx2 v[0x94:0x95], v1, s[16:19], 0 offen
	;;#ASMEND
	v_add_lshl_u32 v1, v0, v12, 1
	;;#ASMSTART
	buffer_store_dwordx2 v[0x96:0x97], v1, s[16:19], 0 offen
	;;#ASMEND
	v_add_lshl_u32 v1, v0, v4, 1
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xb0], v[0xb0], v[0xb1]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xb1], v[0xb2], v[0xb3]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xb2], v[0xb4], v[0xb5]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xb3], v[0xb6], v[0xb7]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xb4], v[0xb8], v[0xb9]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xb5], v[0xba], v[0xbb]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xb6], v[0xbc], v[0xbd]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xb7], v[0xbe], v[0xbf]
	;;#ASMEND
	;;#ASMSTART
	buffer_store_dwordx2 v[0xb0:0xb1], v1, s[16:19], 0 offen
	;;#ASMEND
	v_add_lshl_u32 v1, v0, v14, 1
	;;#ASMSTART
	buffer_store_dwordx2 v[0xb2:0xb3], v1, s[16:19], 0 offen
	;;#ASMEND
	v_add_lshl_u32 v1, v0, v15, 1
	;;#ASMSTART
	buffer_store_dwordx2 v[0xb4:0xb5], v1, s[16:19], 0 offen
	;;#ASMEND
	v_add_lshl_u32 v1, v0, v16, 1
	;;#ASMSTART
	buffer_store_dwordx2 v[0xb6:0xb7], v1, s[16:19], 0 offen
	;;#ASMEND
	v_add_lshl_u32 v1, v0, v9, 1
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xd0], v[0xd0], v[0xd1]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xd1], v[0xd2], v[0xd3]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xd2], v[0xd4], v[0xd5]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xd3], v[0xd6], v[0xd7]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xd4], v[0xd8], v[0xd9]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xd5], v[0xda], v[0xdb]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xd6], v[0xdc], v[0xdd]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xd7], v[0xde], v[0xdf]
	;;#ASMEND
	;;#ASMSTART
	buffer_store_dwordx2 v[0xd0:0xd1], v1, s[16:19], 0 offen
	;;#ASMEND
	v_add_lshl_u32 v1, v0, v17, 1
	;;#ASMSTART
	buffer_store_dwordx2 v[0xd2:0xd3], v1, s[16:19], 0 offen
	;;#ASMEND
	v_add_lshl_u32 v1, v0, v18, 1
	;;#ASMSTART
	buffer_store_dwordx2 v[0xd4:0xd5], v1, s[16:19], 0 offen
	;;#ASMEND
	v_add_lshl_u32 v1, v0, v19, 1
	s_add_i32 s0, s5, 0x1ff0
	;;#ASMSTART
	buffer_store_dwordx2 v[0xd6:0xd7], v1, s[16:19], 0 offen
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xf0], v[0xf0], v[0xf1]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xf1], v[0xf2], v[0xf3]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xf2], v[0xf4], v[0xf5]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xf3], v[0xf6], v[0xf7]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xf4], v[0xf8], v[0xf9]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xf5], v[0xfa], v[0xfb]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xf6], v[0xfc], v[0xfd]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0xf7], v[0xfe], v[0xff]
	;;#ASMEND
	v_add_lshl_u32 v1, v0, v13, 1
	;;#ASMSTART
	buffer_store_dwordx2 v[0xf0:0xf1], v1, s[16:19], 0 offen
	;;#ASMEND
	s_mul_i32 s0, s0, s3
	v_add_lshl_u32 v1, v0, v20, 1
	;;#ASMSTART
	buffer_store_dwordx2 v[0xf2:0xf3], v1, s[16:19], 0 offen
	;;#ASMEND
	s_ashr_i32 s1, s0, 31
	v_add_lshl_u32 v1, v0, v21, 1
	;;#ASMSTART
	buffer_store_dwordx2 v[0xf4:0xf5], v1, s[16:19], 0 offen
	;;#ASMEND
	s_lshl_b64 s[0:1], s[0:1], 1
	v_add_lshl_u32 v0, v0, v22, 1
	;;#ASMSTART
	buffer_store_dwordx2 v[0xf6:0xf7], v0, s[16:19], 0 offen
	;;#ASMEND
	s_add_u32 s8, s6, s0
	;;#ASMSTART
	v_mul_f32_e32 v[0x6e], 0x3db504f3, v[0x6e]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x6f], 0x3db504f3, v[0x6f]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x70], 0x3db504f3, v[0x70]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x71], 0x3db504f3, v[0x71]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x72], 0x3db504f3, v[0x72]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x73], 0x3db504f3, v[0x73]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x74], 0x3db504f3, v[0x74]
	;;#ASMEND
	;;#ASMSTART
	v_mul_f32_e32 v[0x75], 0x3db504f3, v[0x75]
	;;#ASMEND
	s_addc_u32 s9, s7, s1
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x6e], v[0x6e], v[0x6f]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x6f], v[0x70], v[0x71]
	;;#ASMEND
	;;#ASMSTART
	buffer_atomic_pk_add_bf16 v[0x6e], v5, s[8:11], 0 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_atomic_pk_add_bf16 v[0x6f], v6, s[8:11], 0 offen
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x72], v[0x72], v[0x73]
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v[0x73], v[0x74], v[0x75]
	;;#ASMEND
	;;#ASMSTART
	buffer_atomic_pk_add_bf16 v[0x72], v7, s[8:11], 0 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_atomic_pk_add_bf16 v[0x73], v8, s[8:11], 0 offen
	;;#ASMEND
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
                                        ; -- End function
	.set _Z23attend_bwd_combined_kerILi128EEv25attn_bwd_combined_globalsIXT_EE.num_vgpr, 256
	.set _Z23attend_bwd_combined_kerILi128EEv25attn_bwd_combined_globalsIXT_EE.num_agpr, 256
	.set _Z23attend_bwd_combined_kerILi128EEv25attn_bwd_combined_globalsIXT_EE.numbered_sgpr, 79
	.set _Z23attend_bwd_combined_kerILi128EEv25attn_bwd_combined_globalsIXT_EE.private_seg_size, 0
	.set _Z23attend_bwd_combined_kerILi128EEv25attn_bwd_combined_globalsIXT_EE.uses_vcc, 0
	.set _Z23attend_bwd_combined_kerILi128EEv25attn_bwd_combined_globalsIXT_EE.uses_flat_scratch, 0
	.set _Z23attend_bwd_combined_kerILi128EEv25attn_bwd_combined_globalsIXT_EE.has_dyn_sized_stack, 0
	.set _Z23attend_bwd_combined_kerILi128EEv25attn_bwd_combined_globalsIXT_EE.has_recursion, 0
	.set _Z23attend_bwd_combined_kerILi128EEv25attn_bwd_combined_globalsIXT_EE.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 68160
; TotalNumSgprs: 85
; NumVgprs: 256
; NumAgprs: 256
; TotalNumVgprs: 512
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 10
; VGPRBlocks: 63
; NumSGPRsForWavesPerEU: 85
; NumVGPRsForWavesPerEU: 512
; AccumOffset: 256
; Occupancy: 1
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 1
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 63
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.type	__hip_cuid_5e2152b1227460d,@object ; @__hip_cuid_5e2152b1227460d
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_5e2152b1227460d
__hip_cuid_5e2152b1227460d:
	.byte	0                               ; 0x0
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
