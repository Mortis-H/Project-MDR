	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.text
	.protected	_Z27uniqueIdxCalcUsingThreadIdxPi ; -- Begin function _Z27uniqueIdxCalcUsingThreadIdxPi
	.globl	_Z27uniqueIdxCalcUsingThreadIdxPi
	.p2align	8
	.type	_Z27uniqueIdxCalcUsingThreadIdxPi,@function
_Z27uniqueIdxCalcUsingThreadIdxPi:      ; @_Z27uniqueIdxCalcUsingThreadIdxPi
; %bb.0:
	s_load_dwordx2 s[4:5], s[0:1], 0x0
	s_load_dwordx2 s[2:3], s[0:1], 0x58
	v_lshlrev_b32_e32 v1, 2, v0
	v_mbcnt_lo_u32_b32 v3, -1, 0
	v_mbcnt_hi_u32_b32 v3, -1, v3
	s_waitcnt lgkmcnt(0)
	global_load_dword v2, v1, s[4:5]
	v_readfirstlane_b32 s0, v3
	v_mov_b32_e32 v1, 0
	v_mov_b64_e32 v[8:9], 0
	v_cmp_eq_u32_e64 s[0:1], s0, v3
	s_and_saveexec_b64 s[4:5], s[0:1]
	s_cbranch_execz .LBB0_6
; %bb.1:
	global_load_dwordx2 v[6:7], v1, s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[4:5], v1, s[2:3] offset:40
	global_load_dwordx2 v[8:9], v1, s[2:3]
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
	global_atomic_cmpswap_x2 v[8:9], v1, v[4:7], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[8:9], v[6:7]
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_5
; %bb.2:                                ; %.preheader3.i.i.i.preheader
	s_mov_b64 s[8:9], 0
	v_mov_b32_e32 v4, 0
.LBB0_3:                                ; %.preheader3.i.i.i
                                        ; =>This Inner Loop Header: Depth=1
	s_sleep 1
	global_load_dwordx2 v[6:7], v4, s[2:3] offset:40
	global_load_dwordx2 v[12:13], v4, s[2:3]
	v_mov_b64_e32 v[10:11], v[8:9]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v6, v6, v10
	v_and_b32_e32 v5, v7, v11
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[6:7], s[10:11], v6, 24, v[12:13]
	v_mov_b32_e32 v8, v7
	v_mad_u64_u32 v[8:9], s[10:11], v5, 24, v[8:9]
	v_mov_b32_e32 v7, v8
	global_load_dwordx2 v[8:9], v[6:7], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[8:9], v4, v[8:11], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[8:9], v[10:11]
	s_or_b64 s[8:9], vcc, s[8:9]
	s_andn2_b64 exec, exec, s[8:9]
	s_cbranch_execnz .LBB0_3
; %bb.4:                                ; %Flow372
	s_or_b64 exec, exec, s[8:9]
.LBB0_5:                                ; %Flow374
	s_or_b64 exec, exec, s[6:7]
.LBB0_6:                                ; %.loopexit4.i.i.i
	s_or_b64 exec, exec, s[4:5]
	global_load_dwordx2 v[10:11], v1, s[2:3] offset:40
	global_load_dwordx4 v[4:7], v1, s[2:3]
	v_readfirstlane_b32 s4, v8
	v_readfirstlane_b32 s5, v9
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s8, v10
	v_readfirstlane_b32 s9, v11
	s_and_b64 s[8:9], s[4:5], s[8:9]
	s_mul_i32 s10, s9, 24
	s_mul_hi_u32 s11, s8, 24
	s_add_i32 s11, s11, s10
	s_mul_i32 s10, s8, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[8:9], v[4:5], 0, s[10:11]
	s_and_saveexec_b64 s[10:11], s[0:1]
	s_cbranch_execz .LBB0_8
; %bb.7:
	v_mov_b64_e32 v[10:11], s[6:7]
	v_mov_b32_e32 v12, 2
	v_mov_b32_e32 v13, 1
	global_store_dwordx4 v[8:9], v[10:13], off offset:8
.LBB0_8:
	s_or_b64 exec, exec, s[10:11]
	s_lshl_b64 s[6:7], s[8:9], 12
	v_lshl_add_u64 v[10:11], v[6:7], 0, s[6:7]
	s_mov_b32 s8, 0
	v_mov_b32_e32 v31, 0
	v_lshlrev_b32_e32 v30, 6, v3
	v_mov_b32_e32 v12, 33
	v_mov_b32_e32 v13, v31
	v_mov_b32_e32 v14, v31
	v_mov_b32_e32 v15, v31
	v_readfirstlane_b32 s6, v10
	v_readfirstlane_b32 s7, v11
	s_mov_b32 s10, s8
	s_mov_b32 s11, s8
	s_mov_b32 s9, s8
	s_nop 1
	global_store_dwordx4 v30, v[12:15], s[6:7]
	s_nop 1
	v_mov_b64_e32 v[14:15], s[10:11]
	v_mov_b64_e32 v[12:13], s[8:9]
	global_store_dwordx4 v30, v[12:15], s[6:7] offset:16
	global_store_dwordx4 v30, v[12:15], s[6:7] offset:32
	global_store_dwordx4 v30, v[12:15], s[6:7] offset:48
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_16
; %bb.9:
	global_load_dwordx2 v[16:17], v31, s[2:3] offset:32 sc0 sc1
	global_load_dwordx2 v[6:7], v31, s[2:3] offset:40
	v_mov_b32_e32 v14, s4
	v_mov_b32_e32 v15, s5
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v1, s4, v6
	v_and_b32_e32 v6, s5, v7
	v_mul_lo_u32 v7, v6, 24
	v_mul_hi_u32 v12, v1, 24
	v_mul_lo_u32 v6, v1, 24
	v_add_u32_e32 v7, v12, v7
	v_lshl_add_u64 v[12:13], v[4:5], 0, v[6:7]
	global_store_dwordx2 v[12:13], v[16:17], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[6:7], v31, v[14:17], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[6:7], v[16:17]
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_12
; %bb.10:                               ; %.preheader1.i.i.i.preheader
	s_mov_b64 s[10:11], 0
.LBB0_11:                               ; %.preheader1.i.i.i
                                        ; =>This Inner Loop Header: Depth=1
	s_sleep 1
	global_store_dwordx2 v[12:13], v[6:7], off
	v_mov_b32_e32 v4, s4
	v_mov_b32_e32 v5, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[4:5], v31, v[4:7], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[4:5], v[6:7]
	s_or_b64 s[10:11], vcc, s[10:11]
	v_mov_b64_e32 v[6:7], v[4:5]
	s_andn2_b64 exec, exec, s[10:11]
	s_cbranch_execnz .LBB0_11
.LBB0_12:                               ; %Flow370
	s_or_b64 exec, exec, s[8:9]
	v_mov_b32_e32 v7, 0
	global_load_dwordx2 v[4:5], v7, s[2:3] offset:16
	s_mov_b64 s[8:9], exec
	v_mbcnt_lo_u32_b32 v1, s8, 0
	v_mbcnt_hi_u32_b32 v1, s9, v1
	v_cmp_eq_u32_e32 vcc, 0, v1
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_14
; %bb.13:
	s_bcnt1_i32_b64 s8, s[8:9]
	v_mov_b32_e32 v6, s8
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[4:5], v[6:7], off offset:8 sc1
.LBB0_14:
	s_or_b64 exec, exec, s[10:11]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[6:7], v[4:5], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[6:7]
	s_cbranch_vccnz .LBB0_16
; %bb.15:
	global_load_dword v4, v[4:5], off offset:24
	v_mov_b32_e32 v5, 0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v1, 0xffffff, v4
	s_nop 0
	v_readfirstlane_b32 s8, v1
	s_mov_b32 m0, s8
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[6:7], v[4:5], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_16:                               ; %Flow371
	s_or_b64 exec, exec, s[6:7]
	v_lshl_add_u64 v[4:5], v[10:11], 0, v[30:31]
	s_branch .LBB0_20
.LBB0_17:                               ;   in Loop: Header=BB0_20 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s6, v1
	s_cmp_eq_u32 s6, 0
	s_cbranch_scc1 .LBB0_19
; %bb.18:                               ;   in Loop: Header=BB0_20 Depth=1
	s_sleep 1
	s_cbranch_execnz .LBB0_20
	s_branch .LBB0_22
.LBB0_19:
	s_branch .LBB0_22
.LBB0_20:                               ; =>This Inner Loop Header: Depth=1
	v_mov_b32_e32 v1, 1
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_17
; %bb.21:                               ;   in Loop: Header=BB0_20 Depth=1
	global_load_dword v1, v[8:9], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v1, 1, v1
	s_branch .LBB0_17
.LBB0_22:
	global_load_dwordx2 v[4:5], v[4:5], off
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_25
; %bb.23:
	v_mov_b32_e32 v1, 0
	global_load_dwordx2 v[10:11], v1, s[2:3] offset:40
	global_load_dwordx2 v[12:13], v1, s[2:3] offset:24 sc0 sc1
	global_load_dwordx2 v[14:15], v1, s[2:3]
	s_mov_b64 s[0:1], 0
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[6:7], v[10:11], 0, 1
	v_lshl_add_u64 v[16:17], v[6:7], 0, s[4:5]
	v_cmp_eq_u64_e32 vcc, 0, v[16:17]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v8, v12
	v_cndmask_b32_e32 v7, v17, v7, vcc
	v_cndmask_b32_e32 v6, v16, v6, vcc
	v_and_b32_e32 v9, v7, v11
	v_and_b32_e32 v10, v6, v10
	v_mul_lo_u32 v9, v9, 24
	v_mul_hi_u32 v11, v10, 24
	v_mul_lo_u32 v10, v10, 24
	v_add_u32_e32 v11, v11, v9
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[10:11], v[14:15], 0, v[10:11]
	global_store_dwordx2 v[10:11], v[12:13], off
	v_mov_b32_e32 v9, v13
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[8:9], v1, v[6:9], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[8:9], v[12:13]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_25
.LBB0_24:                               ; %.preheader.i.i.i
                                        ; =>This Inner Loop Header: Depth=1
	s_sleep 1
	global_store_dwordx2 v[10:11], v[8:9], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[12:13], v1, v[6:9], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[12:13], v[8:9]
	s_or_b64 s[0:1], vcc, s[0:1]
	v_mov_b64_e32 v[8:9], v[12:13]
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_24
.LBB0_25:                               ; %__ockl_printf_begin.exit
	s_or_b64 exec, exec, s[6:7]
	s_getpc_b64 s[4:5]
	s_add_u32 s4, s4, .str@rel32@lo+4
	s_addc_u32 s5, s5, .str@rel32@hi+12
	s_cmp_lg_u64 s[4:5], 0
	s_cbranch_scc0 .LBB0_110
; %bb.26:
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v32, 2, v4
	v_mov_b32_e32 v35, 0
	v_and_b32_e32 v6, -3, v4
	v_mov_b32_e32 v7, v5
	s_mov_b64 s[6:7], 31
	v_mov_b32_e32 v12, 2
	v_mov_b32_e32 v13, 1
	s_branch .LBB0_28
.LBB0_27:                               ; %__ockl_hostcall_preview.exit19.i
                                        ;   in Loop: Header=BB0_28 Depth=1
	s_or_b64 exec, exec, s[12:13]
	s_sub_u32 s6, s6, s8
	s_subb_u32 s7, s7, s9
	s_add_u32 s4, s4, s8
	s_addc_u32 s5, s5, s9
	s_cmp_lg_u64 s[6:7], 0
	s_cbranch_scc0 .LBB0_109
.LBB0_28:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_31 Depth 2
                                        ;     Child Loop BB0_37 Depth 2
                                        ;     Child Loop BB0_46 Depth 2
                                        ;     Child Loop BB0_54 Depth 2
                                        ;     Child Loop BB0_62 Depth 2
                                        ;     Child Loop BB0_70 Depth 2
                                        ;     Child Loop BB0_78 Depth 2
                                        ;     Child Loop BB0_86 Depth 2
                                        ;     Child Loop BB0_94 Depth 2
                                        ;     Child Loop BB0_103 Depth 2
                                        ;     Child Loop BB0_108 Depth 2
	v_cmp_lt_u64_e64 s[0:1], s[6:7], 56
	s_and_b64 s[0:1], s[0:1], exec
	v_cmp_gt_u64_e64 s[0:1], s[6:7], 7
	s_cselect_b32 s9, s7, 0
	s_cselect_b32 s8, s6, 56
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB0_38
; %bb.29:                               ;   in Loop: Header=BB0_28 Depth=1
	s_mov_b64 s[0:1], 0
	s_cmp_eq_u64 s[6:7], 0
	v_mov_b64_e32 v[8:9], 0
	s_cbranch_scc1 .LBB0_32
; %bb.30:                               ; %.preheader30.i.preheader
                                        ;   in Loop: Header=BB0_28 Depth=1
	s_lshl_b64 s[10:11], s[8:9], 3
	s_mov_b64 s[12:13], 0
	v_mov_b64_e32 v[8:9], 0
	s_mov_b64 s[14:15], s[4:5]
.LBB0_31:                               ; %.preheader30.i
                                        ;   Parent Loop BB0_28 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	global_load_ubyte v1, v35, s[14:15]
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v34, 0xffff, v1
	v_lshlrev_b64 v[10:11], s12, v[34:35]
	s_add_u32 s12, s12, 8
	s_addc_u32 s13, s13, 0
	s_add_u32 s14, s14, 1
	s_addc_u32 s15, s15, 0
	v_or_b32_e32 v8, v10, v8
	s_cmp_lg_u32 s10, s12
	v_or_b32_e32 v9, v11, v9
	s_cbranch_scc1 .LBB0_31
.LBB0_32:                               ; %Flow341
                                        ;   in Loop: Header=BB0_28 Depth=1
	s_mov_b32 s14, 0
	s_andn2_b64 vcc, exec, s[0:1]
	s_mov_b64 s[0:1], s[4:5]
	s_cbranch_vccnz .LBB0_34
.LBB0_33:                               ;   in Loop: Header=BB0_28 Depth=1
	global_load_dwordx2 v[8:9], v35, s[4:5]
	s_add_i32 s14, s8, -8
	s_add_u32 s0, s4, 8
	s_addc_u32 s1, s5, 0
.LBB0_34:                               ; %.loopexit31.i
                                        ;   in Loop: Header=BB0_28 Depth=1
	s_cmp_gt_u32 s14, 7
	s_cbranch_scc1 .LBB0_39
; %bb.35:                               ;   in Loop: Header=BB0_28 Depth=1
	s_cmp_eq_u32 s14, 0
	s_cbranch_scc1 .LBB0_40
; %bb.36:                               ; %.preheader28.i.preheader
                                        ;   in Loop: Header=BB0_28 Depth=1
	s_mov_b64 s[10:11], 0
	v_mov_b64_e32 v[14:15], 0
	s_mov_b64 s[12:13], 0
.LBB0_37:                               ; %.preheader28.i
                                        ;   Parent Loop BB0_28 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_add_u32 s16, s0, s12
	s_addc_u32 s17, s1, s13
	global_load_ubyte v1, v35, s[16:17]
	s_add_u32 s12, s12, 1
	s_addc_u32 s13, s13, 0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v34, 0xffff, v1
	v_lshlrev_b64 v[10:11], s10, v[34:35]
	s_add_u32 s10, s10, 8
	s_addc_u32 s11, s11, 0
	v_or_b32_e32 v14, v10, v14
	s_cmp_lg_u32 s14, s12
	v_or_b32_e32 v15, v11, v15
	s_cbranch_scc1 .LBB0_37
	s_branch .LBB0_41
.LBB0_38:                               ;   in Loop: Header=BB0_28 Depth=1
	s_mov_b32 s14, 0
	s_mov_b64 s[0:1], s[4:5]
	s_branch .LBB0_33
.LBB0_39:                               ;   in Loop: Header=BB0_28 Depth=1
                                        ; implicit-def: $vgpr14_vgpr15
	s_mov_b32 s15, 0
	s_branch .LBB0_42
.LBB0_40:                               ;   in Loop: Header=BB0_28 Depth=1
	v_mov_b64_e32 v[14:15], 0
.LBB0_41:                               ; %Flow334
                                        ;   in Loop: Header=BB0_28 Depth=1
	s_mov_b32 s15, 0
	s_cbranch_execnz .LBB0_43
.LBB0_42:                               ;   in Loop: Header=BB0_28 Depth=1
	global_load_dwordx2 v[14:15], v35, s[0:1]
	s_add_i32 s15, s14, -8
	s_add_u32 s0, s0, 8
	s_addc_u32 s1, s1, 0
.LBB0_43:                               ; %.loopexit29.i
                                        ;   in Loop: Header=BB0_28 Depth=1
	s_cmp_gt_u32 s15, 7
	s_cbranch_scc1 .LBB0_47
; %bb.44:                               ;   in Loop: Header=BB0_28 Depth=1
	s_cmp_eq_u32 s15, 0
	s_cbranch_scc1 .LBB0_48
; %bb.45:                               ; %.preheader26.i.preheader
                                        ;   in Loop: Header=BB0_28 Depth=1
	s_mov_b64 s[10:11], 0
	v_mov_b64_e32 v[16:17], 0
	s_mov_b64 s[12:13], 0
.LBB0_46:                               ; %.preheader26.i
                                        ;   Parent Loop BB0_28 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_add_u32 s16, s0, s12
	s_addc_u32 s17, s1, s13
	global_load_ubyte v1, v35, s[16:17]
	s_add_u32 s12, s12, 1
	s_addc_u32 s13, s13, 0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v34, 0xffff, v1
	v_lshlrev_b64 v[10:11], s10, v[34:35]
	s_add_u32 s10, s10, 8
	s_addc_u32 s11, s11, 0
	v_or_b32_e32 v16, v10, v16
	s_cmp_lg_u32 s15, s12
	v_or_b32_e32 v17, v11, v17
	s_cbranch_scc1 .LBB0_46
	s_branch .LBB0_49
.LBB0_47:                               ;   in Loop: Header=BB0_28 Depth=1
	s_mov_b32 s14, 0
	s_branch .LBB0_50
.LBB0_48:                               ;   in Loop: Header=BB0_28 Depth=1
	v_mov_b64_e32 v[16:17], 0
.LBB0_49:                               ; %Flow329
                                        ;   in Loop: Header=BB0_28 Depth=1
	s_mov_b32 s14, 0
	s_cbranch_execnz .LBB0_51
.LBB0_50:                               ;   in Loop: Header=BB0_28 Depth=1
	global_load_dwordx2 v[16:17], v35, s[0:1]
	s_add_i32 s14, s15, -8
	s_add_u32 s0, s0, 8
	s_addc_u32 s1, s1, 0
.LBB0_51:                               ; %.loopexit27.i
                                        ;   in Loop: Header=BB0_28 Depth=1
	s_cmp_gt_u32 s14, 7
	s_cbranch_scc1 .LBB0_55
; %bb.52:                               ;   in Loop: Header=BB0_28 Depth=1
	s_cmp_eq_u32 s14, 0
	s_cbranch_scc1 .LBB0_56
; %bb.53:                               ; %.preheader24.i.preheader
                                        ;   in Loop: Header=BB0_28 Depth=1
	s_mov_b64 s[10:11], 0
	v_mov_b64_e32 v[18:19], 0
	s_mov_b64 s[12:13], 0
.LBB0_54:                               ; %.preheader24.i
                                        ;   Parent Loop BB0_28 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_add_u32 s16, s0, s12
	s_addc_u32 s17, s1, s13
	global_load_ubyte v1, v35, s[16:17]
	s_add_u32 s12, s12, 1
	s_addc_u32 s13, s13, 0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v34, 0xffff, v1
	v_lshlrev_b64 v[10:11], s10, v[34:35]
	s_add_u32 s10, s10, 8
	s_addc_u32 s11, s11, 0
	v_or_b32_e32 v18, v10, v18
	s_cmp_lg_u32 s14, s12
	v_or_b32_e32 v19, v11, v19
	s_cbranch_scc1 .LBB0_54
	s_branch .LBB0_57
.LBB0_55:                               ;   in Loop: Header=BB0_28 Depth=1
                                        ; implicit-def: $vgpr18_vgpr19
	s_mov_b32 s15, 0
	s_branch .LBB0_58
.LBB0_56:                               ;   in Loop: Header=BB0_28 Depth=1
	v_mov_b64_e32 v[18:19], 0
.LBB0_57:                               ; %Flow324
                                        ;   in Loop: Header=BB0_28 Depth=1
	s_mov_b32 s15, 0
	s_cbranch_execnz .LBB0_59
.LBB0_58:                               ;   in Loop: Header=BB0_28 Depth=1
	global_load_dwordx2 v[18:19], v35, s[0:1]
	s_add_i32 s15, s14, -8
	s_add_u32 s0, s0, 8
	s_addc_u32 s1, s1, 0
.LBB0_59:                               ; %.loopexit25.i
                                        ;   in Loop: Header=BB0_28 Depth=1
	s_cmp_gt_u32 s15, 7
	s_cbranch_scc1 .LBB0_63
; %bb.60:                               ;   in Loop: Header=BB0_28 Depth=1
	s_cmp_eq_u32 s15, 0
	s_cbranch_scc1 .LBB0_64
; %bb.61:                               ; %.preheader22.i.preheader
                                        ;   in Loop: Header=BB0_28 Depth=1
	s_mov_b64 s[10:11], 0
	v_mov_b64_e32 v[20:21], 0
	s_mov_b64 s[12:13], 0
.LBB0_62:                               ; %.preheader22.i
                                        ;   Parent Loop BB0_28 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_add_u32 s16, s0, s12
	s_addc_u32 s17, s1, s13
	global_load_ubyte v1, v35, s[16:17]
	s_add_u32 s12, s12, 1
	s_addc_u32 s13, s13, 0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v34, 0xffff, v1
	v_lshlrev_b64 v[10:11], s10, v[34:35]
	s_add_u32 s10, s10, 8
	s_addc_u32 s11, s11, 0
	v_or_b32_e32 v20, v10, v20
	s_cmp_lg_u32 s15, s12
	v_or_b32_e32 v21, v11, v21
	s_cbranch_scc1 .LBB0_62
	s_branch .LBB0_65
.LBB0_63:                               ;   in Loop: Header=BB0_28 Depth=1
	s_mov_b32 s14, 0
	s_branch .LBB0_66
.LBB0_64:                               ;   in Loop: Header=BB0_28 Depth=1
	v_mov_b64_e32 v[20:21], 0
.LBB0_65:                               ; %Flow319
                                        ;   in Loop: Header=BB0_28 Depth=1
	s_mov_b32 s14, 0
	s_cbranch_execnz .LBB0_67
.LBB0_66:                               ;   in Loop: Header=BB0_28 Depth=1
	global_load_dwordx2 v[20:21], v35, s[0:1]
	s_add_i32 s14, s15, -8
	s_add_u32 s0, s0, 8
	s_addc_u32 s1, s1, 0
.LBB0_67:                               ; %.loopexit23.i
                                        ;   in Loop: Header=BB0_28 Depth=1
	s_cmp_gt_u32 s14, 7
	s_cbranch_scc1 .LBB0_71
; %bb.68:                               ;   in Loop: Header=BB0_28 Depth=1
	s_cmp_eq_u32 s14, 0
	s_cbranch_scc1 .LBB0_72
; %bb.69:                               ; %.preheader20.i.preheader
                                        ;   in Loop: Header=BB0_28 Depth=1
	s_mov_b64 s[10:11], 0
	v_mov_b64_e32 v[22:23], 0
	s_mov_b64 s[12:13], 0
.LBB0_70:                               ; %.preheader20.i
                                        ;   Parent Loop BB0_28 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_add_u32 s16, s0, s12
	s_addc_u32 s17, s1, s13
	global_load_ubyte v1, v35, s[16:17]
	s_add_u32 s12, s12, 1
	s_addc_u32 s13, s13, 0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v34, 0xffff, v1
	v_lshlrev_b64 v[10:11], s10, v[34:35]
	s_add_u32 s10, s10, 8
	s_addc_u32 s11, s11, 0
	v_or_b32_e32 v22, v10, v22
	s_cmp_lg_u32 s14, s12
	v_or_b32_e32 v23, v11, v23
	s_cbranch_scc1 .LBB0_70
	s_branch .LBB0_73
.LBB0_71:                               ;   in Loop: Header=BB0_28 Depth=1
                                        ; implicit-def: $vgpr22_vgpr23
	s_mov_b32 s15, 0
	s_branch .LBB0_74
.LBB0_72:                               ;   in Loop: Header=BB0_28 Depth=1
	v_mov_b64_e32 v[22:23], 0
.LBB0_73:                               ; %Flow314
                                        ;   in Loop: Header=BB0_28 Depth=1
	s_mov_b32 s15, 0
	s_cbranch_execnz .LBB0_75
.LBB0_74:                               ;   in Loop: Header=BB0_28 Depth=1
	global_load_dwordx2 v[22:23], v35, s[0:1]
	s_add_i32 s15, s14, -8
	s_add_u32 s0, s0, 8
	s_addc_u32 s1, s1, 0
.LBB0_75:                               ; %.loopexit21.i
                                        ;   in Loop: Header=BB0_28 Depth=1
	s_cmp_gt_u32 s15, 7
	s_cbranch_scc1 .LBB0_79
; %bb.76:                               ;   in Loop: Header=BB0_28 Depth=1
	s_cmp_eq_u32 s15, 0
	s_cbranch_scc1 .LBB0_80
; %bb.77:                               ; %.preheader.i.preheader
                                        ;   in Loop: Header=BB0_28 Depth=1
	s_mov_b64 s[10:11], 0
	v_mov_b64_e32 v[24:25], 0
	s_mov_b64 s[12:13], s[0:1]
.LBB0_78:                               ; %.preheader.i
                                        ;   Parent Loop BB0_28 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	global_load_ubyte v1, v35, s[12:13]
	s_add_i32 s15, s15, -1
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v34, 0xffff, v1
	v_lshlrev_b64 v[10:11], s10, v[34:35]
	s_add_u32 s10, s10, 8
	s_addc_u32 s11, s11, 0
	s_add_u32 s12, s12, 1
	s_addc_u32 s13, s13, 0
	v_or_b32_e32 v24, v10, v24
	s_cmp_lg_u32 s15, 0
	v_or_b32_e32 v25, v11, v25
	s_cbranch_scc1 .LBB0_78
	s_branch .LBB0_81
.LBB0_79:                               ;   in Loop: Header=BB0_28 Depth=1
	s_branch .LBB0_82
.LBB0_80:                               ;   in Loop: Header=BB0_28 Depth=1
	v_mov_b64_e32 v[24:25], 0
.LBB0_81:                               ; %Flow309
                                        ;   in Loop: Header=BB0_28 Depth=1
	s_cbranch_execnz .LBB0_83
.LBB0_82:                               ;   in Loop: Header=BB0_28 Depth=1
	global_load_dwordx2 v[24:25], v35, s[0:1]
.LBB0_83:                               ; %.loopexit.i
                                        ;   in Loop: Header=BB0_28 Depth=1
	v_readfirstlane_b32 s0, v3
	v_mov_b64_e32 v[10:11], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v3
	s_and_saveexec_b64 s[10:11], s[0:1]
	s_cbranch_execz .LBB0_89
; %bb.84:                               ;   in Loop: Header=BB0_28 Depth=1
	global_load_dwordx2 v[28:29], v35, s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[10:11], v35, s[2:3] offset:40
	global_load_dwordx2 v[26:27], v35, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v1, v10, v28
	v_and_b32_e32 v10, v11, v29
	v_mul_lo_u32 v10, v10, 24
	v_mul_hi_u32 v11, v1, 24
	v_add_u32_e32 v11, v11, v10
	v_mul_lo_u32 v10, v1, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[10:11], v[26:27], 0, v[10:11]
	global_load_dwordx2 v[26:27], v[10:11], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v35, v[26:29], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[10:11], v[28:29]
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB0_88
; %bb.85:                               ; %.preheader3.i.i18.i.preheader
                                        ;   in Loop: Header=BB0_28 Depth=1
	s_mov_b64 s[14:15], 0
.LBB0_86:                               ; %.preheader3.i.i18.i
                                        ;   Parent Loop BB0_28 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_load_dwordx2 v[26:27], v35, s[2:3] offset:40
	global_load_dwordx2 v[36:37], v35, s[2:3]
	v_mov_b64_e32 v[28:29], v[10:11]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v10, v26, v28
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[10:11], s[16:17], v10, 24, v[36:37]
	v_and_b32_e32 v1, v27, v29
	v_mov_b32_e32 v26, v11
	v_mad_u64_u32 v[26:27], s[16:17], v1, 24, v[26:27]
	v_mov_b32_e32 v11, v26
	global_load_dwordx2 v[26:27], v[10:11], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v35, v[26:29], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[10:11], v[28:29]
	s_or_b64 s[14:15], vcc, s[14:15]
	s_andn2_b64 exec, exec, s[14:15]
	s_cbranch_execnz .LBB0_86
; %bb.87:                               ; %Flow304
                                        ;   in Loop: Header=BB0_28 Depth=1
	s_or_b64 exec, exec, s[14:15]
.LBB0_88:                               ; %Flow306
                                        ;   in Loop: Header=BB0_28 Depth=1
	s_or_b64 exec, exec, s[12:13]
.LBB0_89:                               ; %.loopexit4.i.i13.i
                                        ;   in Loop: Header=BB0_28 Depth=1
	s_or_b64 exec, exec, s[10:11]
	global_load_dwordx2 v[36:37], v35, s[2:3] offset:40
	global_load_dwordx4 v[26:29], v35, s[2:3]
	v_readfirstlane_b32 s10, v10
	v_readfirstlane_b32 s11, v11
	s_mov_b64 s[12:13], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s14, v36
	v_readfirstlane_b32 s15, v37
	s_and_b64 s[14:15], s[10:11], s[14:15]
	s_mul_i32 s16, s15, 24
	s_mul_hi_u32 s17, s14, 24
	s_add_i32 s17, s17, s16
	s_mul_i32 s16, s14, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[36:37], v[26:27], 0, s[16:17]
	s_and_saveexec_b64 s[16:17], s[0:1]
	s_cbranch_execz .LBB0_91
; %bb.90:                               ;   in Loop: Header=BB0_28 Depth=1
	v_mov_b64_e32 v[10:11], s[12:13]
	global_store_dwordx4 v[36:37], v[10:13], off offset:8
.LBB0_91:                               ;   in Loop: Header=BB0_28 Depth=1
	s_or_b64 exec, exec, s[16:17]
	s_lshl_b64 s[12:13], s[14:15], 12
	v_lshl_add_u64 v[10:11], v[28:29], 0, s[12:13]
	v_cmp_lt_u64_e64 vcc, s[6:7], 57
	s_lshl_b32 s12, s8, 2
	s_add_i32 s12, s12, 28
	v_cndmask_b32_e32 v1, 0, v32, vcc
	v_and_b32_e32 v6, 0xffffff1f, v6
	s_and_b32 s12, s12, 0x1e0
	v_or_b32_e32 v1, v6, v1
	v_or_b32_e32 v6, s12, v1
	v_readfirstlane_b32 s12, v10
	v_readfirstlane_b32 s13, v11
	s_nop 4
	global_store_dwordx4 v30, v[6:9], s[12:13]
	global_store_dwordx4 v30, v[14:17], s[12:13] offset:16
	global_store_dwordx4 v30, v[18:21], s[12:13] offset:32
	global_store_dwordx4 v30, v[22:25], s[12:13] offset:48
	s_and_saveexec_b64 s[12:13], s[0:1]
	s_cbranch_execz .LBB0_99
; %bb.92:                               ;   in Loop: Header=BB0_28 Depth=1
	global_load_dwordx2 v[18:19], v35, s[2:3] offset:32 sc0 sc1
	global_load_dwordx2 v[6:7], v35, s[2:3] offset:40
	v_mov_b32_e32 v16, s10
	v_mov_b32_e32 v17, s11
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s14, v6
	v_readfirstlane_b32 s15, v7
	s_and_b64 s[14:15], s[14:15], s[10:11]
	s_mul_i32 s15, s15, 24
	s_mul_hi_u32 s16, s14, 24
	s_mul_i32 s14, s14, 24
	s_add_i32 s15, s16, s15
	v_lshl_add_u64 v[14:15], v[26:27], 0, s[14:15]
	global_store_dwordx2 v[14:15], v[18:19], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[8:9], v35, v[16:19], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[8:9], v[18:19]
	s_and_saveexec_b64 s[14:15], vcc
	s_cbranch_execz .LBB0_95
; %bb.93:                               ; %.preheader1.i.i16.i.preheader
                                        ;   in Loop: Header=BB0_28 Depth=1
	s_mov_b64 s[16:17], 0
.LBB0_94:                               ; %.preheader1.i.i16.i
                                        ;   Parent Loop BB0_28 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[14:15], v[8:9], off
	v_mov_b32_e32 v6, s10
	v_mov_b32_e32 v7, s11
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[6:7], v35, v[6:9], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[6:7], v[8:9]
	s_or_b64 s[16:17], vcc, s[16:17]
	v_mov_b64_e32 v[8:9], v[6:7]
	s_andn2_b64 exec, exec, s[16:17]
	s_cbranch_execnz .LBB0_94
.LBB0_95:                               ; %Flow302
                                        ;   in Loop: Header=BB0_28 Depth=1
	s_or_b64 exec, exec, s[14:15]
	global_load_dwordx2 v[6:7], v35, s[2:3] offset:16
	s_mov_b64 s[16:17], exec
	v_mbcnt_lo_u32_b32 v1, s16, 0
	v_mbcnt_hi_u32_b32 v1, s17, v1
	v_cmp_eq_u32_e32 vcc, 0, v1
	s_and_saveexec_b64 s[14:15], vcc
	s_cbranch_execz .LBB0_97
; %bb.96:                               ;   in Loop: Header=BB0_28 Depth=1
	s_bcnt1_i32_b64 s16, s[16:17]
	v_mov_b32_e32 v34, s16
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[6:7], v[34:35], off offset:8 sc1
.LBB0_97:                               ;   in Loop: Header=BB0_28 Depth=1
	s_or_b64 exec, exec, s[14:15]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[8:9], v[6:7], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[8:9]
	s_cbranch_vccnz .LBB0_99
; %bb.98:                               ;   in Loop: Header=BB0_28 Depth=1
	global_load_dword v34, v[6:7], off offset:24
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v1, 0xffffff, v34
	s_nop 0
	v_readfirstlane_b32 s14, v1
	s_mov_b32 m0, s14
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[8:9], v[34:35], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_99:                               ; %Flow303
                                        ;   in Loop: Header=BB0_28 Depth=1
	s_or_b64 exec, exec, s[12:13]
	v_mov_b32_e32 v31, v35
	v_lshl_add_u64 v[6:7], v[10:11], 0, v[30:31]
	s_branch .LBB0_103
.LBB0_100:                              ;   in Loop: Header=BB0_103 Depth=2
	s_or_b64 exec, exec, s[12:13]
	v_readfirstlane_b32 s12, v1
	s_cmp_eq_u32 s12, 0
	s_cbranch_scc1 .LBB0_102
; %bb.101:                              ;   in Loop: Header=BB0_103 Depth=2
	s_sleep 1
	s_cbranch_execnz .LBB0_103
	s_branch .LBB0_105
.LBB0_102:                              ;   in Loop: Header=BB0_28 Depth=1
	s_branch .LBB0_105
.LBB0_103:                              ;   Parent Loop BB0_28 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v1, 1
	s_and_saveexec_b64 s[12:13], s[0:1]
	s_cbranch_execz .LBB0_100
; %bb.104:                              ;   in Loop: Header=BB0_103 Depth=2
	global_load_dword v1, v[36:37], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v1, 1, v1
	s_branch .LBB0_100
.LBB0_105:                              ;   in Loop: Header=BB0_28 Depth=1
	global_load_dwordx2 v[6:7], v[6:7], off
	s_and_saveexec_b64 s[12:13], s[0:1]
	s_cbranch_execz .LBB0_27
; %bb.106:                              ;   in Loop: Header=BB0_28 Depth=1
	global_load_dwordx2 v[14:15], v35, s[2:3] offset:40
	global_load_dwordx2 v[16:17], v35, s[2:3] offset:24 sc0 sc1
	global_load_dwordx2 v[18:19], v35, s[2:3]
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[8:9], v[14:15], 0, 1
	v_lshl_add_u64 v[20:21], v[8:9], 0, s[10:11]
	v_cmp_eq_u64_e32 vcc, 0, v[20:21]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v10, v16
	v_cndmask_b32_e32 v9, v21, v9, vcc
	v_cndmask_b32_e32 v8, v20, v8, vcc
	v_and_b32_e32 v1, v9, v15
	v_and_b32_e32 v11, v8, v14
	v_mul_lo_u32 v1, v1, 24
	v_mul_hi_u32 v15, v11, 24
	v_mul_lo_u32 v14, v11, 24
	v_add_u32_e32 v15, v15, v1
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[14:15], v[18:19], 0, v[14:15]
	global_store_dwordx2 v[14:15], v[16:17], off
	v_mov_b32_e32 v11, v17
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v35, v[8:11], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[10:11], v[16:17]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_27
; %bb.107:                              ; %.preheader.i.i15.i.preheader
                                        ;   in Loop: Header=BB0_28 Depth=1
	s_mov_b64 s[0:1], 0
.LBB0_108:                              ; %.preheader.i.i15.i
                                        ;   Parent Loop BB0_28 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[14:15], v[10:11], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[16:17], v35, v[8:11], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[16:17], v[10:11]
	s_or_b64 s[0:1], vcc, s[0:1]
	v_mov_b64_e32 v[10:11], v[16:17]
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_108
	s_branch .LBB0_27
.LBB0_109:                              ; %Flow342
	s_branch .LBB0_137
.LBB0_110:
                                        ; implicit-def: $vgpr6_vgpr7
	s_cbranch_execz .LBB0_137
; %bb.111:
	v_readfirstlane_b32 s0, v3
	s_waitcnt vmcnt(0)
	v_mov_b64_e32 v[6:7], 0
	v_cmp_eq_u32_e64 s[0:1], s0, v3
	s_and_saveexec_b64 s[4:5], s[0:1]
	s_cbranch_execz .LBB0_117
; %bb.112:
	v_mov_b32_e32 v1, 0
	global_load_dwordx2 v[8:9], v1, s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[6:7], v1, s[2:3] offset:40
	global_load_dwordx2 v[10:11], v1, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v6, v6, v8
	v_and_b32_e32 v7, v7, v9
	v_mul_lo_u32 v7, v7, 24
	v_mul_hi_u32 v12, v6, 24
	v_add_u32_e32 v7, v12, v7
	v_mul_lo_u32 v6, v6, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[6:7], v[10:11], 0, v[6:7]
	global_load_dwordx2 v[6:7], v[6:7], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[6:7], v1, v[6:9], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[6:7], v[8:9]
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_116
; %bb.113:                              ; %.preheader3.i.i.i8.preheader
	s_mov_b64 s[8:9], 0
.LBB0_114:                              ; %.preheader3.i.i.i8
                                        ; =>This Inner Loop Header: Depth=1
	s_sleep 1
	global_load_dwordx2 v[10:11], v1, s[2:3] offset:40
	global_load_dwordx2 v[12:13], v1, s[2:3]
	v_mov_b64_e32 v[8:9], v[6:7]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v6, v10, v8
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[6:7], s[10:11], v6, 24, v[12:13]
	v_and_b32_e32 v11, v11, v9
	v_mov_b32_e32 v10, v7
	v_mad_u64_u32 v[10:11], s[10:11], v11, 24, v[10:11]
	v_mov_b32_e32 v7, v10
	global_load_dwordx2 v[6:7], v[6:7], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[6:7], v1, v[6:9], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[6:7], v[8:9]
	s_or_b64 s[8:9], vcc, s[8:9]
	s_andn2_b64 exec, exec, s[8:9]
	s_cbranch_execnz .LBB0_114
; %bb.115:                              ; %Flow355
	s_or_b64 exec, exec, s[8:9]
.LBB0_116:                              ; %Flow357
	s_or_b64 exec, exec, s[6:7]
.LBB0_117:                              ; %.loopexit4.i.i.i3
	s_or_b64 exec, exec, s[4:5]
	v_mov_b32_e32 v31, 0
	global_load_dwordx2 v[12:13], v31, s[2:3] offset:40
	global_load_dwordx4 v[8:11], v31, s[2:3]
	v_readfirstlane_b32 s4, v6
	v_readfirstlane_b32 s5, v7
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s8, v12
	v_readfirstlane_b32 s9, v13
	s_and_b64 s[8:9], s[4:5], s[8:9]
	s_mul_i32 s10, s9, 24
	s_mul_hi_u32 s11, s8, 24
	s_add_i32 s11, s11, s10
	s_mul_i32 s10, s8, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[12:13], v[8:9], 0, s[10:11]
	s_and_saveexec_b64 s[10:11], s[0:1]
	s_cbranch_execz .LBB0_119
; %bb.118:
	v_mov_b64_e32 v[14:15], s[6:7]
	v_mov_b32_e32 v16, 2
	v_mov_b32_e32 v17, 1
	global_store_dwordx4 v[12:13], v[14:17], off offset:8
.LBB0_119:
	s_or_b64 exec, exec, s[10:11]
	s_lshl_b64 s[6:7], s[8:9], 12
	v_lshl_add_u64 v[10:11], v[10:11], 0, s[6:7]
	s_movk_i32 s6, 0xff1f
	s_mov_b32 s8, 0
	v_and_or_b32 v4, v4, s6, 32
	v_mov_b32_e32 v6, v31
	v_mov_b32_e32 v7, v31
	v_readfirstlane_b32 s6, v10
	v_readfirstlane_b32 s7, v11
	s_mov_b32 s9, s8
	s_mov_b32 s10, s8
	s_mov_b32 s11, s8
	s_nop 1
	global_store_dwordx4 v30, v[4:7], s[6:7]
	s_nop 1
	v_mov_b64_e32 v[4:5], s[8:9]
	v_mov_b64_e32 v[6:7], s[10:11]
	global_store_dwordx4 v30, v[4:7], s[6:7] offset:16
	global_store_dwordx4 v30, v[4:7], s[6:7] offset:32
	global_store_dwordx4 v30, v[4:7], s[6:7] offset:48
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_127
; %bb.120:
	v_mov_b32_e32 v1, 0
	global_load_dwordx2 v[16:17], v1, s[2:3] offset:32 sc0 sc1
	global_load_dwordx2 v[4:5], v1, s[2:3] offset:40
	v_mov_b32_e32 v14, s4
	v_mov_b32_e32 v15, s5
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s8, v4
	v_readfirstlane_b32 s9, v5
	s_and_b64 s[8:9], s[8:9], s[4:5]
	s_mul_i32 s9, s9, 24
	s_mul_hi_u32 s10, s8, 24
	s_mul_i32 s8, s8, 24
	s_add_i32 s9, s10, s9
	v_lshl_add_u64 v[8:9], v[8:9], 0, s[8:9]
	global_store_dwordx2 v[8:9], v[16:17], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[6:7], v1, v[14:17], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[6:7], v[16:17]
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_123
; %bb.121:                              ; %.preheader1.i.i.i6.preheader
	s_mov_b64 s[10:11], 0
.LBB0_122:                              ; %.preheader1.i.i.i6
                                        ; =>This Inner Loop Header: Depth=1
	s_sleep 1
	global_store_dwordx2 v[8:9], v[6:7], off
	v_mov_b32_e32 v4, s4
	v_mov_b32_e32 v5, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[4:5], v1, v[4:7], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[4:5], v[6:7]
	s_or_b64 s[10:11], vcc, s[10:11]
	v_mov_b64_e32 v[6:7], v[4:5]
	s_andn2_b64 exec, exec, s[10:11]
	s_cbranch_execnz .LBB0_122
.LBB0_123:                              ; %Flow353
	s_or_b64 exec, exec, s[8:9]
	v_mov_b32_e32 v7, 0
	global_load_dwordx2 v[4:5], v7, s[2:3] offset:16
	s_mov_b64 s[8:9], exec
	v_mbcnt_lo_u32_b32 v1, s8, 0
	v_mbcnt_hi_u32_b32 v1, s9, v1
	v_cmp_eq_u32_e32 vcc, 0, v1
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_125
; %bb.124:
	s_bcnt1_i32_b64 s8, s[8:9]
	v_mov_b32_e32 v6, s8
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[4:5], v[6:7], off offset:8 sc1
.LBB0_125:
	s_or_b64 exec, exec, s[10:11]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[6:7], v[4:5], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[6:7]
	s_cbranch_vccnz .LBB0_127
; %bb.126:
	global_load_dword v4, v[4:5], off offset:24
	v_mov_b32_e32 v5, 0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v1, 0xffffff, v4
	s_nop 0
	v_readfirstlane_b32 s8, v1
	s_mov_b32 m0, s8
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[6:7], v[4:5], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_127:                              ; %Flow354
	s_or_b64 exec, exec, s[6:7]
	v_lshl_add_u64 v[4:5], v[10:11], 0, v[30:31]
	s_branch .LBB0_131
.LBB0_128:                              ;   in Loop: Header=BB0_131 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s6, v1
	s_cmp_eq_u32 s6, 0
	s_cbranch_scc1 .LBB0_130
; %bb.129:                              ;   in Loop: Header=BB0_131 Depth=1
	s_sleep 1
	s_cbranch_execnz .LBB0_131
	s_branch .LBB0_133
.LBB0_130:
	s_branch .LBB0_133
.LBB0_131:                              ; =>This Inner Loop Header: Depth=1
	v_mov_b32_e32 v1, 1
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_128
; %bb.132:                              ;   in Loop: Header=BB0_131 Depth=1
	global_load_dword v1, v[12:13], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v1, 1, v1
	s_branch .LBB0_128
.LBB0_133:
	global_load_dwordx2 v[6:7], v[4:5], off
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_136
; %bb.134:
	v_mov_b32_e32 v1, 0
	global_load_dwordx2 v[4:5], v1, s[2:3] offset:40
	global_load_dwordx2 v[12:13], v1, s[2:3] offset:24 sc0 sc1
	global_load_dwordx2 v[14:15], v1, s[2:3]
	s_mov_b64 s[0:1], 0
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[8:9], v[4:5], 0, 1
	v_lshl_add_u64 v[16:17], v[8:9], 0, s[4:5]
	v_cmp_eq_u64_e32 vcc, 0, v[16:17]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v10, v12
	v_cndmask_b32_e32 v9, v17, v9, vcc
	v_cndmask_b32_e32 v8, v16, v8, vcc
	v_and_b32_e32 v5, v9, v5
	v_and_b32_e32 v4, v8, v4
	v_mul_lo_u32 v5, v5, 24
	v_mul_hi_u32 v11, v4, 24
	v_mul_lo_u32 v4, v4, 24
	v_add_u32_e32 v5, v11, v5
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[4:5], v[14:15], 0, v[4:5]
	global_store_dwordx2 v[4:5], v[12:13], off
	v_mov_b32_e32 v11, v13
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v1, v[8:11], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[10:11], v[12:13]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_136
.LBB0_135:                              ; %.preheader.i.i.i5
                                        ; =>This Inner Loop Header: Depth=1
	s_sleep 1
	global_store_dwordx2 v[4:5], v[10:11], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[12:13], v1, v[8:11], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[12:13], v[10:11]
	s_or_b64 s[0:1], vcc, s[0:1]
	v_mov_b64_e32 v[10:11], v[12:13]
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_135
.LBB0_136:                              ; %Flow346
	s_or_b64 exec, exec, s[6:7]
.LBB0_137:                              ; %__ockl_printf_append_string_n.exit
	v_readfirstlane_b32 s0, v3
	s_waitcnt vmcnt(0)
	v_mov_b64_e32 v[4:5], 0
	v_cmp_eq_u32_e64 s[0:1], s0, v3
	s_and_saveexec_b64 s[4:5], s[0:1]
	s_cbranch_execz .LBB0_143
; %bb.138:
	v_mov_b32_e32 v1, 0
	global_load_dwordx2 v[10:11], v1, s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[4:5], v1, s[2:3] offset:40
	global_load_dwordx2 v[8:9], v1, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v4, v4, v10
	v_and_b32_e32 v5, v5, v11
	v_mul_lo_u32 v5, v5, 24
	v_mul_hi_u32 v12, v4, 24
	v_add_u32_e32 v5, v12, v5
	v_mul_lo_u32 v4, v4, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[4:5], v[8:9], 0, v[4:5]
	global_load_dwordx2 v[8:9], v[4:5], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[4:5], v1, v[8:11], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[4:5], v[10:11]
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_142
; %bb.139:                              ; %.preheader3.i.i.i15.preheader
	s_mov_b64 s[8:9], 0
.LBB0_140:                              ; %.preheader3.i.i.i15
                                        ; =>This Inner Loop Header: Depth=1
	s_sleep 1
	global_load_dwordx2 v[8:9], v1, s[2:3] offset:40
	global_load_dwordx2 v[12:13], v1, s[2:3]
	v_mov_b64_e32 v[10:11], v[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v4, v8, v10
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[4:5], s[10:11], v4, 24, v[12:13]
	v_and_b32_e32 v9, v9, v11
	v_mov_b32_e32 v8, v5
	v_mad_u64_u32 v[8:9], s[10:11], v9, 24, v[8:9]
	v_mov_b32_e32 v5, v8
	global_load_dwordx2 v[8:9], v[4:5], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[4:5], v1, v[8:11], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[4:5], v[10:11]
	s_or_b64 s[8:9], vcc, s[8:9]
	s_andn2_b64 exec, exec, s[8:9]
	s_cbranch_execnz .LBB0_140
; %bb.141:                              ; %Flow290
	s_or_b64 exec, exec, s[8:9]
.LBB0_142:                              ; %Flow292
	s_or_b64 exec, exec, s[6:7]
.LBB0_143:                              ; %.loopexit4.i.i.i9
	s_or_b64 exec, exec, s[4:5]
	v_mov_b32_e32 v31, 0
	global_load_dwordx2 v[8:9], v31, s[2:3] offset:40
	global_load_dwordx4 v[10:13], v31, s[2:3]
	v_readfirstlane_b32 s4, v4
	v_readfirstlane_b32 s5, v5
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s8, v8
	v_readfirstlane_b32 s9, v9
	s_and_b64 s[8:9], s[4:5], s[8:9]
	s_mul_i32 s10, s9, 24
	s_mul_hi_u32 s11, s8, 24
	s_add_i32 s11, s11, s10
	s_mul_i32 s10, s8, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[14:15], v[10:11], 0, s[10:11]
	s_and_saveexec_b64 s[10:11], s[0:1]
	s_cbranch_execz .LBB0_145
; %bb.144:
	v_mov_b64_e32 v[16:17], s[6:7]
	v_mov_b32_e32 v18, 2
	v_mov_b32_e32 v19, 1
	global_store_dwordx4 v[14:15], v[16:19], off offset:8
.LBB0_145:
	s_or_b64 exec, exec, s[10:11]
	s_lshl_b64 s[6:7], s[8:9], 12
	v_lshl_add_u64 v[12:13], v[12:13], 0, s[6:7]
	s_movk_i32 s6, 0xff1f
	s_mov_b32 s8, 0
	v_and_or_b32 v6, v6, s6, 32
	v_mov_b32_e32 v8, v0
	v_mov_b32_e32 v9, v31
	v_readfirstlane_b32 s6, v12
	v_readfirstlane_b32 s7, v13
	s_mov_b32 s9, s8
	s_mov_b32 s10, s8
	s_mov_b32 s11, s8
	s_nop 1
	global_store_dwordx4 v30, v[6:9], s[6:7]
	v_mov_b64_e32 v[4:5], s[8:9]
	s_nop 0
	v_mov_b64_e32 v[6:7], s[10:11]
	global_store_dwordx4 v30, v[4:7], s[6:7] offset:16
	global_store_dwordx4 v30, v[4:7], s[6:7] offset:32
	global_store_dwordx4 v30, v[4:7], s[6:7] offset:48
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_153
; %bb.146:
	v_mov_b32_e32 v8, 0
	global_load_dwordx2 v[18:19], v8, s[2:3] offset:32 sc0 sc1
	global_load_dwordx2 v[0:1], v8, s[2:3] offset:40
	v_mov_b32_e32 v16, s4
	v_mov_b32_e32 v17, s5
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s8, v0
	v_readfirstlane_b32 s9, v1
	s_and_b64 s[8:9], s[8:9], s[4:5]
	s_mul_i32 s9, s9, 24
	s_mul_hi_u32 s10, s8, 24
	s_mul_i32 s8, s8, 24
	s_add_i32 s9, s10, s9
	v_lshl_add_u64 v[0:1], v[10:11], 0, s[8:9]
	global_store_dwordx2 v[0:1], v[18:19], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[6:7], v8, v[16:19], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[6:7], v[18:19]
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_149
; %bb.147:                              ; %.preheader1.i.i.i13.preheader
	s_mov_b64 s[10:11], 0
.LBB0_148:                              ; %.preheader1.i.i.i13
                                        ; =>This Inner Loop Header: Depth=1
	s_sleep 1
	global_store_dwordx2 v[0:1], v[6:7], off
	v_mov_b32_e32 v4, s4
	v_mov_b32_e32 v5, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[4:5], v8, v[4:7], s[2:3] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[4:5], v[6:7]
	s_or_b64 s[10:11], vcc, s[10:11]
	v_mov_b64_e32 v[6:7], v[4:5]
	s_andn2_b64 exec, exec, s[10:11]
	s_cbranch_execnz .LBB0_148
.LBB0_149:                              ; %Flow288
	s_or_b64 exec, exec, s[8:9]
	v_mov_b32_e32 v5, 0
	global_load_dwordx2 v[0:1], v5, s[2:3] offset:16
	s_mov_b64 s[8:9], exec
	v_mbcnt_lo_u32_b32 v4, s8, 0
	v_mbcnt_hi_u32_b32 v4, s9, v4
	v_cmp_eq_u32_e32 vcc, 0, v4
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_151
; %bb.150:
	s_bcnt1_i32_b64 s8, s[8:9]
	v_mov_b32_e32 v4, s8
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[0:1], v[4:5], off offset:8 sc1
.LBB0_151:
	s_or_b64 exec, exec, s[10:11]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[4:5], v[0:1], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[4:5]
	s_cbranch_vccnz .LBB0_153
; %bb.152:
	global_load_dword v0, v[0:1], off offset:24
	v_mov_b32_e32 v1, 0
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_store_dwordx2 v[4:5], v[0:1], off sc0 sc1
	v_and_b32_e32 v0, 0xffffff, v0
	s_nop 0
	v_readfirstlane_b32 s8, v0
	s_mov_b32 m0, s8
	s_nop 0
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_153:                              ; %Flow289
	s_or_b64 exec, exec, s[6:7]
	v_lshl_add_u64 v[0:1], v[12:13], 0, v[30:31]
	s_branch .LBB0_157
.LBB0_154:                              ;   in Loop: Header=BB0_157 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s6, v4
	s_cmp_eq_u32 s6, 0
	s_cbranch_scc1 .LBB0_156
; %bb.155:                              ;   in Loop: Header=BB0_157 Depth=1
	s_sleep 1
	s_cbranch_execnz .LBB0_157
	s_branch .LBB0_159
.LBB0_156:
	s_branch .LBB0_159
.LBB0_157:                              ; =>This Inner Loop Header: Depth=1
	v_mov_b32_e32 v4, 1
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_154
; %bb.158:                              ;   in Loop: Header=BB0_157 Depth=1
	global_load_dword v4, v[14:15], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v4, 1, v4
	s_branch .LBB0_154
.LBB0_159:
	global_load_dwordx2 v[0:1], v[0:1], off
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_162
; %bb.160:
	v_mov_b32_e32 v10, 0
	global_load_dwordx2 v[8:9], v10, s[2:3] offset:40
	global_load_dwordx2 v[12:13], v10, s[2:3] offset:24 sc0 sc1
	global_load_dwordx2 v[14:15], v10, s[2:3]
	s_mov_b64 s[0:1], 0
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[4:5], v[8:9], 0, 1
	v_lshl_add_u64 v[16:17], v[4:5], 0, s[4:5]
	v_cmp_eq_u64_e32 vcc, 0, v[16:17]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v6, v12
	v_cndmask_b32_e32 v5, v17, v5, vcc
	v_cndmask_b32_e32 v4, v16, v4, vcc
	v_and_b32_e32 v7, v5, v9
	v_and_b32_e32 v8, v4, v8
	v_mul_lo_u32 v7, v7, 24
	v_mul_hi_u32 v9, v8, 24
	v_mul_lo_u32 v8, v8, 24
	v_add_u32_e32 v9, v9, v7
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[8:9], v[14:15], 0, v[8:9]
	global_store_dwordx2 v[8:9], v[12:13], off
	v_mov_b32_e32 v7, v13
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[6:7], v10, v[4:7], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[6:7], v[12:13]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_162
.LBB0_161:                              ; %.preheader.i.i.i12
                                        ; =>This Inner Loop Header: Depth=1
	s_sleep 1
	global_store_dwordx2 v[8:9], v[6:7], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[12:13], v10, v[4:7], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[12:13], v[6:7]
	s_or_b64 s[0:1], vcc, s[0:1]
	v_mov_b64_e32 v[6:7], v[12:13]
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_161
.LBB0_162:                              ; %__ockl_printf_append_args.exit
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s0, v3
	v_mov_b64_e32 v[8:9], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v3
	s_and_saveexec_b64 s[4:5], s[0:1]
	s_cbranch_execz .LBB0_168
; %bb.163:
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
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_167
; %bb.164:                              ; %.preheader3.i.i.i22.preheader
	s_mov_b64 s[8:9], 0
.LBB0_165:                              ; %.preheader3.i.i.i22
                                        ; =>This Inner Loop Header: Depth=1
	s_sleep 1
	global_load_dwordx2 v[4:5], v3, s[2:3] offset:40
	global_load_dwordx2 v[10:11], v3, s[2:3]
	v_mov_b64_e32 v[6:7], v[8:9]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v4, v4, v6
	v_and_b32_e32 v9, v5, v7
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[4:5], s[10:11], v4, 24, v[10:11]
	v_mov_b32_e32 v8, v5
	v_mad_u64_u32 v[8:9], s[10:11], v9, 24, v[8:9]
	v_mov_b32_e32 v5, v8
	global_load_dwordx2 v[4:5], v[4:5], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[8:9], v3, v[4:7], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[8:9], v[6:7]
	s_or_b64 s[8:9], vcc, s[8:9]
	s_andn2_b64 exec, exec, s[8:9]
	s_cbranch_execnz .LBB0_165
; %bb.166:                              ; %Flow276
	s_or_b64 exec, exec, s[8:9]
.LBB0_167:                              ; %Flow278
	s_or_b64 exec, exec, s[6:7]
.LBB0_168:                              ; %.loopexit4.i.i.i16
	s_or_b64 exec, exec, s[4:5]
	v_mov_b32_e32 v3, 0
	global_load_dwordx2 v[10:11], v3, s[2:3] offset:40
	global_load_dwordx4 v[4:7], v3, s[2:3]
	v_readfirstlane_b32 s4, v8
	v_readfirstlane_b32 s5, v9
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s8, v10
	v_readfirstlane_b32 s9, v11
	s_and_b64 s[8:9], s[4:5], s[8:9]
	s_mul_i32 s10, s9, 24
	s_mul_hi_u32 s11, s8, 24
	s_add_i32 s11, s11, s10
	s_mul_i32 s10, s8, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[8:9], v[4:5], 0, s[10:11]
	s_and_saveexec_b64 s[10:11], s[0:1]
	s_cbranch_execz .LBB0_170
; %bb.169:
	v_mov_b64_e32 v[10:11], s[6:7]
	v_mov_b32_e32 v12, 2
	v_mov_b32_e32 v13, 1
	global_store_dwordx4 v[8:9], v[10:13], off offset:8
.LBB0_170:
	s_or_b64 exec, exec, s[10:11]
	s_lshl_b64 s[6:7], s[8:9], 12
	v_lshl_add_u64 v[6:7], v[6:7], 0, s[6:7]
	s_movk_i32 s6, 0xff1d
	s_mov_b32 s8, 0
	v_and_or_b32 v0, v0, s6, 34
	v_readfirstlane_b32 s6, v6
	v_readfirstlane_b32 s7, v7
	s_mov_b32 s9, s8
	s_mov_b32 s10, s8
	s_mov_b32 s11, s8
	s_nop 1
	global_store_dwordx4 v30, v[0:3], s[6:7]
	s_nop 1
	v_mov_b64_e32 v[0:1], s[8:9]
	v_mov_b64_e32 v[2:3], s[10:11]
	global_store_dwordx4 v30, v[0:3], s[6:7] offset:16
	global_store_dwordx4 v30, v[0:3], s[6:7] offset:32
	global_store_dwordx4 v30, v[0:3], s[6:7] offset:48
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_178
; %bb.171:
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
	s_cbranch_execz .LBB0_174
; %bb.172:                              ; %.preheader1.i.i.i20.preheader
	s_mov_b64 s[10:11], 0
.LBB0_173:                              ; %.preheader1.i.i.i20
                                        ; =>This Inner Loop Header: Depth=1
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
	s_cbranch_execnz .LBB0_173
.LBB0_174:                              ; %Flow274
	s_or_b64 exec, exec, s[8:9]
	v_mov_b32_e32 v3, 0
	global_load_dwordx2 v[0:1], v3, s[2:3] offset:16
	s_mov_b64 s[8:9], exec
	v_mbcnt_lo_u32_b32 v2, s8, 0
	v_mbcnt_hi_u32_b32 v2, s9, v2
	v_cmp_eq_u32_e32 vcc, 0, v2
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_176
; %bb.175:
	s_bcnt1_i32_b64 s8, s[8:9]
	v_mov_b32_e32 v2, s8
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[0:1], v[2:3], off offset:8 sc1
.LBB0_176:
	s_or_b64 exec, exec, s[10:11]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[2:3], v[0:1], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[2:3]
	s_cbranch_vccnz .LBB0_178
; %bb.177:
	global_load_dword v0, v[0:1], off offset:24
	v_mov_b32_e32 v1, 0
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_store_dwordx2 v[2:3], v[0:1], off sc0 sc1
	v_and_b32_e32 v0, 0xffffff, v0
	s_nop 0
	v_readfirstlane_b32 s8, v0
	s_mov_b32 m0, s8
	s_nop 0
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_178:                              ; %Flow275
	s_or_b64 exec, exec, s[6:7]
	s_branch .LBB0_182
.LBB0_179:                              ;   in Loop: Header=BB0_182 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s6, v0
	s_cmp_eq_u32 s6, 0
	s_cbranch_scc1 .LBB0_181
; %bb.180:                              ;   in Loop: Header=BB0_182 Depth=1
	s_sleep 1
	s_cbranch_execnz .LBB0_182
	s_branch .LBB0_184
.LBB0_181:
	s_branch .LBB0_184
.LBB0_182:                              ; =>This Inner Loop Header: Depth=1
	v_mov_b32_e32 v0, 1
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_179
; %bb.183:                              ;   in Loop: Header=BB0_182 Depth=1
	global_load_dword v0, v[8:9], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v0, 1, v0
	s_branch .LBB0_179
.LBB0_184:
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_187
; %bb.185:
	v_mov_b32_e32 v6, 0
	global_load_dwordx2 v[4:5], v6, s[2:3] offset:40
	global_load_dwordx2 v[8:9], v6, s[2:3] offset:24 sc0 sc1
	global_load_dwordx2 v[10:11], v6, s[2:3]
	s_mov_b64 s[0:1], 0
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[0:1], v[4:5], 0, 1
	v_lshl_add_u64 v[12:13], v[0:1], 0, s[4:5]
	v_cmp_eq_u64_e32 vcc, 0, v[12:13]
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v2, v8
	v_cndmask_b32_e32 v1, v13, v1, vcc
	v_cndmask_b32_e32 v0, v12, v0, vcc
	v_and_b32_e32 v3, v1, v5
	v_and_b32_e32 v4, v0, v4
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v5, v4, 24
	v_mul_lo_u32 v4, v4, 24
	v_add_u32_e32 v5, v5, v3
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[4:5], v[10:11], 0, v[4:5]
	global_store_dwordx2 v[4:5], v[8:9], off
	v_mov_b32_e32 v3, v9
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v6, v[0:3], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[2:3], v[8:9]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_187
.LBB0_186:                              ; %.preheader.i.i.i19
                                        ; =>This Inner Loop Header: Depth=1
	s_sleep 1
	global_store_dwordx2 v[4:5], v[2:3], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[8:9], v6, v[0:3], s[2:3] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[8:9], v[2:3]
	s_or_b64 s[0:1], vcc, s[0:1]
	v_mov_b64_e32 v[2:3], v[8:9]
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_186
.LBB0_187:                              ; %__ockl_printf_append_args.exit23
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z27uniqueIdxCalcUsingThreadIdxPi
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 264
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
		.amdhsa_next_free_vgpr 38
		.amdhsa_next_free_sgpr 18
		.amdhsa_accum_offset 40
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
	.size	_Z27uniqueIdxCalcUsingThreadIdxPi, .Lfunc_end0-_Z27uniqueIdxCalcUsingThreadIdxPi
                                        ; -- End function
	.set _Z27uniqueIdxCalcUsingThreadIdxPi.num_vgpr, 38
	.set _Z27uniqueIdxCalcUsingThreadIdxPi.num_agpr, 0
	.set _Z27uniqueIdxCalcUsingThreadIdxPi.numbered_sgpr, 18
	.set _Z27uniqueIdxCalcUsingThreadIdxPi.private_seg_size, 0
	.set _Z27uniqueIdxCalcUsingThreadIdxPi.uses_vcc, 1
	.set _Z27uniqueIdxCalcUsingThreadIdxPi.uses_flat_scratch, 0
	.set _Z27uniqueIdxCalcUsingThreadIdxPi.has_dyn_sized_stack, 0
	.set _Z27uniqueIdxCalcUsingThreadIdxPi.has_recursion, 0
	.set _Z27uniqueIdxCalcUsingThreadIdxPi.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 7016
; TotalNumSgprs: 24
; NumVgprs: 38
; NumAgprs: 0
; TotalNumVgprs: 38
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 2
; VGPRBlocks: 4
; NumSGPRsForWavesPerEU: 24
; NumVGPRsForWavesPerEU: 38
; AccumOffset: 40
; Occupancy: 8
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 9
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.type	.str,@object                    ; @.str
	.section	.rodata.str1.1,"aMS",@progbits,1
.str:
	.asciz	"threadIdx : %d,   value : %d \n"
	.size	.str, 31

	.type	__hip_cuid_34339226a7dc1ad2,@object ; @__hip_cuid_34339226a7dc1ad2
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_34339226a7dc1ad2
__hip_cuid_34339226a7dc1ad2:
	.byte	0                               ; 0x0
	.size	__hip_cuid_34339226a7dc1ad2, 1

	.ident	"AMD clang version 20.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.0.1 25314 f4087f6b428f0e6f575ebac8a8a724dab123d06e)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_34339226a7dc1ad2
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .offset:         8
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         12
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         16
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         20
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         22
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         24
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         26
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         28
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         30
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         48
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         56
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         64
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         72
        .size:           2
        .value_kind:     hidden_grid_dims
      - .offset:         88
        .size:           8
        .value_kind:     hidden_hostcall_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 264
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z27uniqueIdxCalcUsingThreadIdxPi
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .sgpr_spill_count: 0
    .symbol:         _Z27uniqueIdxCalcUsingThreadIdxPi.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     38
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
