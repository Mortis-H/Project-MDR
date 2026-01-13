
_Z27uniqueIdxCalcUsingThreadIdxPi:      ; @_Z27uniqueIdxCalcUsingThreadIdxPi
; %bb.0:
	s_load_dwordx2 s[0:1], s[4:5], 0x0
	s_load_dwordx2 s[2:3], s[4:5], 0x58
	v_lshlrev_b32_e32 v1, 2, v0
	v_mbcnt_lo_u32_b32 v3, -1, 0
	v_mbcnt_hi_u32_b32 v3, -1, v3
	s_waitcnt lgkmcnt(0)
	global_load_dword v2, v1, s[0:1]
	v_readfirstlane_b32 s0, v3
	v_mov_b32_e32 v1, 0
	v_cmp_eq_u32_e64 s[0:1], s0, v3
	v_pk_mov_b32 v[8:9], 0, 0
	s_and_saveexec_b64 s[4:5], s[0:1]
	s_cbranch_execz .LBB0_6
; %bb.1:
	global_load_dwordx2 v[6:7], v1, s[2:3] offset:24 glc
	s_waitcnt vmcnt(0)
	buffer_invl2
	buffer_wbinvl1_vol
	global_load_dwordx2 v[4:5], v1, s[2:3] offset:40
	global_load_dwordx2 v[8:9], v1, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v4, v4, v6
	v_and_b32_e32 v5, v5, v7
	v_mul_lo_u32 v5, v5, 24
	v_mul_hi_u32 v10, v4, 24
	v_mul_lo_u32 v4, v4, 24
	v_add_u32_e32 v5, v10, v5
	s_waitcnt vmcnt(0)
	v_add_co_u32_e32 v4, vcc, v8, v4
	v_addc_co_u32_e32 v5, vcc, v9, v5, vcc
	global_load_dwordx2 v[4:5], v[4:5], off glc
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[8:9], v1, v[4:7], s[2:3] offset:24 glc
	s_waitcnt vmcnt(0)
	buffer_invl2
	buffer_wbinvl1_vol
	v_cmp_ne_u64_e32 vcc, v[8:9], v[6:7]
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_5
; %bb.2:
	s_mov_b64 s[8:9], 0
	v_mov_b32_e32 v4, 0
.LBB0_3:                                ; =>This Inner Loop Header: Depth=1
	s_sleep 1
	global_load_dwordx2 v[6:7], v4, s[2:3] offset:40
	global_load_dwordx2 v[12:13], v4, s[2:3]
	v_pk_mov_b32 v[10:11], v[8:9], v[8:9] op_sel:[0,1]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v6, v6, v10
	v_and_b32_e32 v5, v7, v11
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[6:7], s[10:11], v6, 24, v[12:13]
	v_mov_b32_e32 v8, v7
	v_mad_u64_u32 v[8:9], s[10:11], v5, 24, v[8:9]
	v_mov_b32_e32 v7, v8
	global_load_dwordx2 v[8:9], v[6:7], off glc
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[8:9], v4, v[8:11], s[2:3] offset:24 glc
	s_waitcnt vmcnt(0)
	buffer_invl2
	buffer_wbinvl1_vol
	v_cmp_eq_u64_e32 vcc, v[8:9], v[10:11]
	s_or_b64 s[8:9], vcc, s[8:9]
	s_andn2_b64 exec, exec, s[8:9]
	s_cbranch_execnz .LBB0_3
; %bb.4:
	s_or_b64 exec, exec, s[8:9]
.LBB0_5:
	s_or_b64 exec, exec, s[6:7]
.LBB0_6:
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
	s_mul_i32 s12, s8, 24
	s_add_i32 s10, s11, s10
	v_mov_b32_e32 v1, s10
	s_waitcnt vmcnt(0)
	v_add_co_u32_e32 v12, vcc, s12, v4
	v_addc_co_u32_e32 v13, vcc, v5, v1, vcc
	s_and_saveexec_b64 s[10:11], s[0:1]
	s_cbranch_execz .LBB0_8
; %bb.7:
	v_pk_mov_b32 v[8:9], s[6:7], s[6:7] op_sel:[0,1]
	v_mov_b32_e32 v10, 2
	v_mov_b32_e32 v11, 1
	global_store_dwordx4 v[12:13], v[8:11], off offset:8
.LBB0_8:
	s_or_b64 exec, exec, s[10:11]
	s_lshl_b64 s[6:7], s[8:9], 12
	v_mov_b32_e32 v1, s7
	v_add_co_u32_e32 v10, vcc, s6, v6
	v_addc_co_u32_e32 v1, vcc, v7, v1, vcc
	s_mov_b32 s8, 0
	v_mov_b32_e32 v7, 0
	v_lshlrev_b32_e32 v33, 6, v3
	v_mov_b32_e32 v6, 33
	v_mov_b32_e32 v8, v7
	v_mov_b32_e32 v9, v7
	v_readfirstlane_b32 s6, v10
	v_readfirstlane_b32 s7, v1
	s_mov_b32 s9, s8
	v_add_co_u32_e32 v14, vcc, v10, v33
	s_mov_b32 s10, s8
	s_mov_b32 s11, s8
	s_nop 0
	global_store_dwordx4 v33, v[6:9], s[6:7]
	v_addc_co_u32_e32 v15, vcc, 0, v1, vcc
	v_pk_mov_b32 v[8:9], s[8:9], s[8:9] op_sel:[0,1]
	v_pk_mov_b32 v[10:11], s[10:11], s[10:11] op_sel:[0,1]
	global_store_dwordx4 v33, v[8:11], s[6:7] offset:16
	global_store_dwordx4 v33, v[8:11], s[6:7] offset:32
	global_store_dwordx4 v33, v[8:11], s[6:7] offset:48
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_16
; %bb.9:
	global_load_dwordx2 v[18:19], v7, s[2:3] offset:32 glc
	global_load_dwordx2 v[8:9], v7, s[2:3] offset:40
	v_mov_b32_e32 v16, s4
	v_mov_b32_e32 v17, s5
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v1, s4, v8
	v_and_b32_e32 v6, s5, v9
	v_mul_lo_u32 v6, v6, 24
	v_mul_hi_u32 v8, v1, 24
	v_mul_lo_u32 v1, v1, 24
	v_add_u32_e32 v6, v8, v6
	v_add_co_u32_e32 v4, vcc, v4, v1
	v_addc_co_u32_e32 v5, vcc, v5, v6, vcc
	global_store_dwordx2 v[4:5], v[18:19], off
	buffer_wbl2
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v7, v[16:19], s[2:3] offset:32 glc
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[10:11], v[18:19]
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_12
; %bb.10:
	s_mov_b64 s[10:11], 0
.LBB0_11:                               ; =>This Inner Loop Header: Depth=1
	s_sleep 1
	global_store_dwordx2 v[4:5], v[10:11], off
	v_mov_b32_e32 v8, s4
	v_mov_b32_e32 v9, s5
	buffer_wbl2
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[8:9], v7, v[8:11], s[2:3] offset:32 glc
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[8:9], v[10:11]
	s_or_b64 s[10:11], vcc, s[10:11]
	v_pk_mov_b32 v[10:11], v[8:9], v[8:9] op_sel:[0,1]
	s_andn2_b64 exec, exec, s[10:11]
	s_cbranch_execnz .LBB0_11
.LBB0_12:
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
	buffer_wbl2
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[4:5], v[6:7], off offset:8
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
	v_readfirstlane_b32 s8, v1
	s_mov_b32 m0, s8
	buffer_wbl2
	global_store_dwordx2 v[6:7], v[4:5], off
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_16:
	s_or_b64 exec, exec, s[6:7]
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
	global_load_dword v1, v[12:13], off offset:20 glc
	s_waitcnt vmcnt(0)
	buffer_invl2
	buffer_wbinvl1_vol
	v_and_b32_e32 v1, 1, v1
	s_branch .LBB0_17
.LBB0_22:
	global_load_dwordx2 v[4:5], v[14:15], off
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_25
; %bb.23:
	v_mov_b32_e32 v1, 0
	global_load_dwordx2 v[10:11], v1, s[2:3] offset:40
	global_load_dwordx2 v[12:13], v1, s[2:3] offset:24 glc
	global_load_dwordx2 v[14:15], v1, s[2:3]
	v_mov_b32_e32 v7, s5
	s_mov_b64 s[0:1], 0
	s_waitcnt vmcnt(2)
	v_add_co_u32_e32 v9, vcc, 1, v10
	v_addc_co_u32_e32 v16, vcc, 0, v11, vcc
	v_add_co_u32_e32 v6, vcc, s4, v9
	v_addc_co_u32_e32 v7, vcc, v16, v7, vcc
	v_cmp_eq_u64_e32 vcc, 0, v[6:7]
	v_cndmask_b32_e32 v7, v7, v16, vcc
	v_cndmask_b32_e32 v6, v6, v9, vcc
	v_and_b32_e32 v9, v7, v11
	v_and_b32_e32 v10, v6, v10
	v_mul_lo_u32 v9, v9, 24
	v_mul_hi_u32 v11, v10, 24
	v_mul_lo_u32 v10, v10, 24
	v_add_u32_e32 v9, v11, v9
	s_waitcnt vmcnt(0)
	v_add_co_u32_e32 v10, vcc, v14, v10
	v_addc_co_u32_e32 v11, vcc, v15, v9, vcc
	v_mov_b32_e32 v8, v12
	global_store_dwordx2 v[10:11], v[12:13], off
	v_mov_b32_e32 v9, v13
	buffer_wbl2
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[8:9], v1, v[6:9], s[2:3] offset:24 glc
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[8:9], v[12:13]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_25
.LBB0_24:                               ; =>This Inner Loop Header: Depth=1
	s_sleep 1
	global_store_dwordx2 v[10:11], v[8:9], off
	buffer_wbl2
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[12:13], v1, v[6:9], s[2:3] offset:24 glc
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[12:13], v[8:9]
	s_or_b64 s[0:1], vcc, s[0:1]
	v_pk_mov_b32 v[8:9], v[12:13], v[12:13] op_sel:[0,1]
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_24
.LBB0_25:
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
	v_mov_b32_e32 v10, 2
	v_mov_b32_e32 v11, 1
	s_branch .LBB0_28
.LBB0_27:                               ;   in Loop: Header=BB0_28 Depth=1
	s_or_b64 exec, exec, s[12:13]
	s_sub_u32 s6, s6, s8
	s_subb_u32 s7, s7, s9
	s_add_u32 s4, s4, s8
	s_addc_u32 s5, s5, s9
	s_cmp_lg_u64 s[6:7], 0
	s_cbranch_scc0 .LBB0_109
.LBB0_28:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_31 Depth 2
                                        ;     Child Loop BB0_38 Depth 2
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
	s_cbranch_vccnz .LBB0_33
; %bb.29:                               ;   in Loop: Header=BB0_28 Depth=1
	s_mov_b64 s[0:1], 0
	s_cmp_eq_u64 s[6:7], 0
	v_pk_mov_b32 v[14:15], 0, 0
	s_cbranch_scc1 .LBB0_32
; %bb.30:                               ;   in Loop: Header=BB0_28 Depth=1
	s_lshl_b64 s[10:11], s[8:9], 3
	s_mov_b64 s[12:13], 0
	v_pk_mov_b32 v[14:15], 0, 0
	s_mov_b64 s[14:15], s[4:5]
.LBB0_31:                               ;   Parent Loop BB0_28 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	global_load_ubyte v1, v35, s[14:15]
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v34, 0xffff, v1
	v_lshlrev_b64 v[8:9], s12, v[34:35]
	s_add_u32 s12, s12, 8
	s_addc_u32 s13, s13, 0
	s_add_u32 s14, s14, 1
	s_addc_u32 s15, s15, 0
	v_or_b32_e32 v14, v8, v14
	s_cmp_lg_u32 s10, s12
	v_or_b32_e32 v15, v9, v15
	s_cbranch_scc1 .LBB0_31
.LBB0_32:                               ;   in Loop: Header=BB0_28 Depth=1
	s_mov_b32 s14, 0
	s_andn2_b64 vcc, exec, s[0:1]
	s_mov_b64 s[0:1], s[4:5]
	s_cbranch_vccz .LBB0_34
	s_branch .LBB0_35
.LBB0_33:                               ;   in Loop: Header=BB0_28 Depth=1
                                        ; implicit-def: $vgpr14_vgpr15
                                        ; implicit-def: $sgpr14
	s_mov_b64 s[0:1], s[4:5]
.LBB0_34:                               ;   in Loop: Header=BB0_28 Depth=1
	global_load_dwordx2 v[14:15], v35, s[4:5]
	s_add_i32 s14, s8, -8
	s_add_u32 s0, s4, 8
	s_addc_u32 s1, s5, 0
.LBB0_35:                               ;   in Loop: Header=BB0_28 Depth=1
	s_cmp_gt_u32 s14, 7
	s_cbranch_scc1 .LBB0_39
; %bb.36:                               ;   in Loop: Header=BB0_28 Depth=1
	s_cmp_eq_u32 s14, 0
	s_cbranch_scc1 .LBB0_40
; %bb.37:                               ;   in Loop: Header=BB0_28 Depth=1
	s_mov_b64 s[10:11], 0
	v_pk_mov_b32 v[16:17], 0, 0
	s_mov_b64 s[12:13], 0
.LBB0_38:                               ;   Parent Loop BB0_28 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_add_u32 s16, s0, s12
	s_addc_u32 s17, s1, s13
	global_load_ubyte v1, v35, s[16:17]
	s_add_u32 s12, s12, 1
	s_addc_u32 s13, s13, 0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v34, 0xffff, v1
	v_lshlrev_b64 v[8:9], s10, v[34:35]
	s_add_u32 s10, s10, 8
	s_addc_u32 s11, s11, 0
	v_or_b32_e32 v16, v8, v16
	s_cmp_lg_u32 s14, s12
	v_or_b32_e32 v17, v9, v17
	s_cbranch_scc1 .LBB0_38
	s_branch .LBB0_41
.LBB0_39:                               ;   in Loop: Header=BB0_28 Depth=1
                                        ; implicit-def: $vgpr16_vgpr17
                                        ; implicit-def: $sgpr15
	s_branch .LBB0_42
.LBB0_40:                               ;   in Loop: Header=BB0_28 Depth=1
	v_pk_mov_b32 v[16:17], 0, 0
.LBB0_41:                               ;   in Loop: Header=BB0_28 Depth=1
	s_mov_b32 s15, 0
	s_cbranch_execnz .LBB0_43
.LBB0_42:                               ;   in Loop: Header=BB0_28 Depth=1
	global_load_dwordx2 v[16:17], v35, s[0:1]
	s_add_i32 s15, s14, -8
	s_add_u32 s0, s0, 8
	s_addc_u32 s1, s1, 0
.LBB0_43:                               ;   in Loop: Header=BB0_28 Depth=1
	s_cmp_gt_u32 s15, 7
	s_cbranch_scc1 .LBB0_47
; %bb.44:                               ;   in Loop: Header=BB0_28 Depth=1
	s_cmp_eq_u32 s15, 0
	s_cbranch_scc1 .LBB0_48
; %bb.45:                               ;   in Loop: Header=BB0_28 Depth=1
	s_mov_b64 s[10:11], 0
	v_pk_mov_b32 v[18:19], 0, 0
	s_mov_b64 s[12:13], 0
.LBB0_46:                               ;   Parent Loop BB0_28 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_add_u32 s16, s0, s12
	s_addc_u32 s17, s1, s13
	global_load_ubyte v1, v35, s[16:17]
	s_add_u32 s12, s12, 1
	s_addc_u32 s13, s13, 0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v34, 0xffff, v1
	v_lshlrev_b64 v[8:9], s10, v[34:35]
	s_add_u32 s10, s10, 8
	s_addc_u32 s11, s11, 0
	v_or_b32_e32 v18, v8, v18
	s_cmp_lg_u32 s15, s12
	v_or_b32_e32 v19, v9, v19
	s_cbranch_scc1 .LBB0_46
	s_branch .LBB0_49
.LBB0_47:                               ;   in Loop: Header=BB0_28 Depth=1
                                        ; implicit-def: $sgpr14
	s_branch .LBB0_50
.LBB0_48:                               ;   in Loop: Header=BB0_28 Depth=1
	v_pk_mov_b32 v[18:19], 0, 0
.LBB0_49:                               ;   in Loop: Header=BB0_28 Depth=1
	s_mov_b32 s14, 0
	s_cbranch_execnz .LBB0_51
.LBB0_50:                               ;   in Loop: Header=BB0_28 Depth=1
	global_load_dwordx2 v[18:19], v35, s[0:1]
	s_add_i32 s14, s15, -8
	s_add_u32 s0, s0, 8
	s_addc_u32 s1, s1, 0
.LBB0_51:                               ;   in Loop: Header=BB0_28 Depth=1
	s_cmp_gt_u32 s14, 7
	s_cbranch_scc1 .LBB0_55
; %bb.52:                               ;   in Loop: Header=BB0_28 Depth=1
	s_cmp_eq_u32 s14, 0
	s_cbranch_scc1 .LBB0_56
; %bb.53:                               ;   in Loop: Header=BB0_28 Depth=1
	s_mov_b64 s[10:11], 0
	v_pk_mov_b32 v[20:21], 0, 0
	s_mov_b64 s[12:13], 0
.LBB0_54:                               ;   Parent Loop BB0_28 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_add_u32 s16, s0, s12
	s_addc_u32 s17, s1, s13
	global_load_ubyte v1, v35, s[16:17]
	s_add_u32 s12, s12, 1
	s_addc_u32 s13, s13, 0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v34, 0xffff, v1
	v_lshlrev_b64 v[8:9], s10, v[34:35]
	s_add_u32 s10, s10, 8
	s_addc_u32 s11, s11, 0
	v_or_b32_e32 v20, v8, v20
	s_cmp_lg_u32 s14, s12
	v_or_b32_e32 v21, v9, v21
	s_cbranch_scc1 .LBB0_54
	s_branch .LBB0_57
.LBB0_55:                               ;   in Loop: Header=BB0_28 Depth=1
                                        ; implicit-def: $vgpr20_vgpr21
                                        ; implicit-def: $sgpr15
	s_branch .LBB0_58
.LBB0_56:                               ;   in Loop: Header=BB0_28 Depth=1
	v_pk_mov_b32 v[20:21], 0, 0
.LBB0_57:                               ;   in Loop: Header=BB0_28 Depth=1
	s_mov_b32 s15, 0
	s_cbranch_execnz .LBB0_59
.LBB0_58:                               ;   in Loop: Header=BB0_28 Depth=1
	global_load_dwordx2 v[20:21], v35, s[0:1]
	s_add_i32 s15, s14, -8
	s_add_u32 s0, s0, 8
	s_addc_u32 s1, s1, 0
.LBB0_59:                               ;   in Loop: Header=BB0_28 Depth=1
	s_cmp_gt_u32 s15, 7
	s_cbranch_scc1 .LBB0_63
; %bb.60:                               ;   in Loop: Header=BB0_28 Depth=1
	s_cmp_eq_u32 s15, 0
	s_cbranch_scc1 .LBB0_64
; %bb.61:                               ;   in Loop: Header=BB0_28 Depth=1
	s_mov_b64 s[10:11], 0
	v_pk_mov_b32 v[22:23], 0, 0
	s_mov_b64 s[12:13], 0
.LBB0_62:                               ;   Parent Loop BB0_28 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_add_u32 s16, s0, s12
	s_addc_u32 s17, s1, s13
	global_load_ubyte v1, v35, s[16:17]
	s_add_u32 s12, s12, 1
	s_addc_u32 s13, s13, 0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v34, 0xffff, v1
	v_lshlrev_b64 v[8:9], s10, v[34:35]
	s_add_u32 s10, s10, 8
	s_addc_u32 s11, s11, 0
	v_or_b32_e32 v22, v8, v22
	s_cmp_lg_u32 s15, s12
	v_or_b32_e32 v23, v9, v23
	s_cbranch_scc1 .LBB0_62
	s_branch .LBB0_65
.LBB0_63:                               ;   in Loop: Header=BB0_28 Depth=1
                                        ; implicit-def: $sgpr14
	s_branch .LBB0_66
.LBB0_64:                               ;   in Loop: Header=BB0_28 Depth=1
	v_pk_mov_b32 v[22:23], 0, 0
.LBB0_65:                               ;   in Loop: Header=BB0_28 Depth=1
	s_mov_b32 s14, 0
	s_cbranch_execnz .LBB0_67
.LBB0_66:                               ;   in Loop: Header=BB0_28 Depth=1
	global_load_dwordx2 v[22:23], v35, s[0:1]
	s_add_i32 s14, s15, -8
	s_add_u32 s0, s0, 8
	s_addc_u32 s1, s1, 0
.LBB0_67:                               ;   in Loop: Header=BB0_28 Depth=1
	s_cmp_gt_u32 s14, 7
	s_cbranch_scc1 .LBB0_71
; %bb.68:                               ;   in Loop: Header=BB0_28 Depth=1
	s_cmp_eq_u32 s14, 0
	s_cbranch_scc1 .LBB0_72
; %bb.69:                               ;   in Loop: Header=BB0_28 Depth=1
	s_mov_b64 s[10:11], 0
	v_pk_mov_b32 v[24:25], 0, 0
	s_mov_b64 s[12:13], 0
.LBB0_70:                               ;   Parent Loop BB0_28 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_add_u32 s16, s0, s12
	s_addc_u32 s17, s1, s13
	global_load_ubyte v1, v35, s[16:17]
	s_add_u32 s12, s12, 1
	s_addc_u32 s13, s13, 0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v34, 0xffff, v1
	v_lshlrev_b64 v[8:9], s10, v[34:35]
	s_add_u32 s10, s10, 8
	s_addc_u32 s11, s11, 0
	v_or_b32_e32 v24, v8, v24
	s_cmp_lg_u32 s14, s12
	v_or_b32_e32 v25, v9, v25
	s_cbranch_scc1 .LBB0_70
	s_branch .LBB0_73
.LBB0_71:                               ;   in Loop: Header=BB0_28 Depth=1
                                        ; implicit-def: $vgpr24_vgpr25
                                        ; implicit-def: $sgpr15
	s_branch .LBB0_74
.LBB0_72:                               ;   in Loop: Header=BB0_28 Depth=1
	v_pk_mov_b32 v[24:25], 0, 0
.LBB0_73:                               ;   in Loop: Header=BB0_28 Depth=1
	s_mov_b32 s15, 0
	s_cbranch_execnz .LBB0_75
.LBB0_74:                               ;   in Loop: Header=BB0_28 Depth=1
	global_load_dwordx2 v[24:25], v35, s[0:1]
	s_add_i32 s15, s14, -8
	s_add_u32 s0, s0, 8
	s_addc_u32 s1, s1, 0
.LBB0_75:                               ;   in Loop: Header=BB0_28 Depth=1
	s_cmp_gt_u32 s15, 7
	s_cbranch_scc1 .LBB0_79
; %bb.76:                               ;   in Loop: Header=BB0_28 Depth=1
	s_cmp_eq_u32 s15, 0
	s_cbranch_scc1 .LBB0_80
; %bb.77:                               ;   in Loop: Header=BB0_28 Depth=1
	s_mov_b64 s[10:11], 0
	v_pk_mov_b32 v[26:27], 0, 0
	s_mov_b64 s[12:13], s[0:1]
.LBB0_78:                               ;   Parent Loop BB0_28 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	global_load_ubyte v1, v35, s[12:13]
	s_add_i32 s15, s15, -1
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v34, 0xffff, v1
	v_lshlrev_b64 v[8:9], s10, v[34:35]
	s_add_u32 s10, s10, 8
	s_addc_u32 s11, s11, 0
	s_add_u32 s12, s12, 1
	s_addc_u32 s13, s13, 0
	v_or_b32_e32 v26, v8, v26
	s_cmp_lg_u32 s15, 0
	v_or_b32_e32 v27, v9, v27
	s_cbranch_scc1 .LBB0_78
	s_branch .LBB0_81
.LBB0_79:                               ;   in Loop: Header=BB0_28 Depth=1
	s_branch .LBB0_82
.LBB0_80:                               ;   in Loop: Header=BB0_28 Depth=1
	v_pk_mov_b32 v[26:27], 0, 0
.LBB0_81:                               ;   in Loop: Header=BB0_28 Depth=1
	s_cbranch_execnz .LBB0_83
.LBB0_82:                               ;   in Loop: Header=BB0_28 Depth=1
	global_load_dwordx2 v[26:27], v35, s[0:1]
.LBB0_83:                               ;   in Loop: Header=BB0_28 Depth=1
	v_readfirstlane_b32 s0, v3
	v_cmp_eq_u32_e64 s[0:1], s0, v3
	s_waitcnt vmcnt(0)
	v_pk_mov_b32 v[8:9], 0, 0
	s_and_saveexec_b64 s[10:11], s[0:1]
	s_cbranch_execz .LBB0_89
; %bb.84:                               ;   in Loop: Header=BB0_28 Depth=1
	global_load_dwordx2 v[30:31], v35, s[2:3] offset:24 glc
	s_waitcnt vmcnt(0)
	buffer_invl2
	buffer_wbinvl1_vol
	global_load_dwordx2 v[8:9], v35, s[2:3] offset:40
	global_load_dwordx2 v[12:13], v35, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v1, v8, v30
	v_and_b32_e32 v8, v9, v31
	v_mul_lo_u32 v8, v8, 24
	v_mul_hi_u32 v9, v1, 24
	v_mul_lo_u32 v1, v1, 24
	v_add_u32_e32 v9, v9, v8
	s_waitcnt vmcnt(0)
	v_add_co_u32_e32 v8, vcc, v12, v1
	v_addc_co_u32_e32 v9, vcc, v13, v9, vcc
	global_load_dwordx2 v[28:29], v[8:9], off glc
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[8:9], v35, v[28:31], s[2:3] offset:24 glc
	s_waitcnt vmcnt(0)
	buffer_invl2
	buffer_wbinvl1_vol
	v_cmp_ne_u64_e32 vcc, v[8:9], v[30:31]
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB0_88
; %bb.85:                               ;   in Loop: Header=BB0_28 Depth=1
	s_mov_b64 s[14:15], 0
.LBB0_86:                               ;   Parent Loop BB0_28 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_load_dwordx2 v[12:13], v35, s[2:3] offset:40
	global_load_dwordx2 v[28:29], v35, s[2:3]
	v_pk_mov_b32 v[30:31], v[8:9], v[8:9] op_sel:[0,1]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v8, v12, v30
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[8:9], s[16:17], v8, 24, v[28:29]
	v_and_b32_e32 v1, v13, v31
	v_mov_b32_e32 v12, v9
	v_mad_u64_u32 v[12:13], s[16:17], v1, 24, v[12:13]
	v_mov_b32_e32 v9, v12
	global_load_dwordx2 v[28:29], v[8:9], off glc
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[8:9], v35, v[28:31], s[2:3] offset:24 glc
	s_waitcnt vmcnt(0)
	buffer_invl2
	buffer_wbinvl1_vol
	v_cmp_eq_u64_e32 vcc, v[8:9], v[30:31]
	s_or_b64 s[14:15], vcc, s[14:15]
	s_andn2_b64 exec, exec, s[14:15]
	s_cbranch_execnz .LBB0_86
; %bb.87:                               ;   in Loop: Header=BB0_28 Depth=1
	s_or_b64 exec, exec, s[14:15]
.LBB0_88:                               ;   in Loop: Header=BB0_28 Depth=1
	s_or_b64 exec, exec, s[12:13]
.LBB0_89:                               ;   in Loop: Header=BB0_28 Depth=1
	s_or_b64 exec, exec, s[10:11]
	global_load_dwordx2 v[12:13], v35, s[2:3] offset:40
	global_load_dwordx4 v[28:31], v35, s[2:3]
	v_readfirstlane_b32 s10, v8
	v_readfirstlane_b32 s11, v9
	s_mov_b64 s[12:13], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s14, v12
	v_readfirstlane_b32 s15, v13
	s_and_b64 s[14:15], s[10:11], s[14:15]
	s_mul_i32 s16, s15, 24
	s_mul_hi_u32 s17, s14, 24
	s_mul_i32 s18, s14, 24
	s_add_i32 s16, s17, s16
	v_mov_b32_e32 v1, s16
	s_waitcnt vmcnt(0)
	v_add_co_u32_e32 v36, vcc, s18, v28
	v_addc_co_u32_e32 v37, vcc, v29, v1, vcc
	s_and_saveexec_b64 s[16:17], s[0:1]
	s_cbranch_execz .LBB0_91
; %bb.90:                               ;   in Loop: Header=BB0_28 Depth=1
	v_pk_mov_b32 v[8:9], s[12:13], s[12:13] op_sel:[0,1]
	global_store_dwordx4 v[36:37], v[8:11], off offset:8
.LBB0_91:                               ;   in Loop: Header=BB0_28 Depth=1
	s_or_b64 exec, exec, s[16:17]
	s_lshl_b64 s[12:13], s[14:15], 12
	v_mov_b32_e32 v1, s13
	v_add_co_u32_e32 v30, vcc, s12, v30
	v_addc_co_u32_e32 v1, vcc, v31, v1, vcc
	v_or_b32_e32 v9, v6, v32
	v_cmp_gt_u64_e64 vcc, s[6:7], 56
	s_lshl_b32 s12, s8, 2
	v_cndmask_b32_e32 v6, v9, v6, vcc
	s_add_i32 s12, s12, 28
	v_or_b32_e32 v8, 0, v7
	s_and_b32 s12, s12, 0x1e0
	v_and_b32_e32 v6, 0xffffff1f, v6
	v_cndmask_b32_e32 v13, v8, v7, vcc
	v_or_b32_e32 v12, s12, v6
	v_readfirstlane_b32 s12, v30
	v_readfirstlane_b32 s13, v1
	s_nop 4
	global_store_dwordx4 v33, v[12:15], s[12:13]
	global_store_dwordx4 v33, v[16:19], s[12:13] offset:16
	global_store_dwordx4 v33, v[20:23], s[12:13] offset:32
	global_store_dwordx4 v33, v[24:27], s[12:13] offset:48
	s_and_saveexec_b64 s[12:13], s[0:1]
	s_cbranch_execz .LBB0_99
; %bb.92:                               ;   in Loop: Header=BB0_28 Depth=1
	global_load_dwordx2 v[16:17], v35, s[2:3] offset:32 glc
	global_load_dwordx2 v[6:7], v35, s[2:3] offset:40
	v_mov_b32_e32 v14, s10
	v_mov_b32_e32 v15, s11
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s14, v6
	v_readfirstlane_b32 s15, v7
	s_and_b64 s[14:15], s[14:15], s[10:11]
	s_mul_i32 s15, s15, 24
	s_mul_hi_u32 s16, s14, 24
	s_mul_i32 s14, s14, 24
	s_add_i32 s15, s16, s15
	v_mov_b32_e32 v6, s15
	v_add_co_u32_e32 v12, vcc, s14, v28
	v_addc_co_u32_e32 v13, vcc, v29, v6, vcc
	global_store_dwordx2 v[12:13], v[16:17], off
	buffer_wbl2
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[8:9], v35, v[14:17], s[2:3] offset:32 glc
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[8:9], v[16:17]
	s_and_saveexec_b64 s[14:15], vcc
	s_cbranch_execz .LBB0_95
; %bb.93:                               ;   in Loop: Header=BB0_28 Depth=1
	s_mov_b64 s[16:17], 0
.LBB0_94:                               ;   Parent Loop BB0_28 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[12:13], v[8:9], off
	v_mov_b32_e32 v6, s10
	v_mov_b32_e32 v7, s11
	buffer_wbl2
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[6:7], v35, v[6:9], s[2:3] offset:32 glc
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[6:7], v[8:9]
	s_or_b64 s[16:17], vcc, s[16:17]
	v_pk_mov_b32 v[8:9], v[6:7], v[6:7] op_sel:[0,1]
	s_andn2_b64 exec, exec, s[16:17]
	s_cbranch_execnz .LBB0_94
.LBB0_95:                               ;   in Loop: Header=BB0_28 Depth=1
	s_or_b64 exec, exec, s[14:15]
	global_load_dwordx2 v[6:7], v35, s[2:3] offset:16
	s_mov_b64 s[16:17], exec
	v_mbcnt_lo_u32_b32 v8, s16, 0
	v_mbcnt_hi_u32_b32 v8, s17, v8
	v_cmp_eq_u32_e32 vcc, 0, v8
	s_and_saveexec_b64 s[14:15], vcc
	s_cbranch_execz .LBB0_97
; %bb.96:                               ;   in Loop: Header=BB0_28 Depth=1
	s_bcnt1_i32_b64 s16, s[16:17]
	v_mov_b32_e32 v34, s16
	buffer_wbl2
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[6:7], v[34:35], off offset:8
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
	v_and_b32_e32 v6, 0xffffff, v34
	v_readfirstlane_b32 s14, v6
	s_mov_b32 m0, s14
	buffer_wbl2
	global_store_dwordx2 v[8:9], v[34:35], off
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_99:                               ;   in Loop: Header=BB0_28 Depth=1
	s_or_b64 exec, exec, s[12:13]
	v_add_co_u32_e32 v6, vcc, v30, v33
	v_addc_co_u32_e32 v7, vcc, 0, v1, vcc
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
	global_load_dword v1, v[36:37], off offset:20 glc
	s_waitcnt vmcnt(0)
	buffer_invl2
	buffer_wbinvl1_vol
	v_and_b32_e32 v1, 1, v1
	s_branch .LBB0_100
.LBB0_105:                              ;   in Loop: Header=BB0_28 Depth=1
	global_load_dwordx4 v[6:9], v[6:7], off
	s_and_saveexec_b64 s[12:13], s[0:1]
	s_cbranch_execz .LBB0_27
; %bb.106:                              ;   in Loop: Header=BB0_28 Depth=1
	global_load_dwordx2 v[8:9], v35, s[2:3] offset:40
	global_load_dwordx2 v[16:17], v35, s[2:3] offset:24 glc
	global_load_dwordx2 v[18:19], v35, s[2:3]
	v_mov_b32_e32 v1, s11
	s_waitcnt vmcnt(2)
	v_add_co_u32_e32 v15, vcc, 1, v8
	v_addc_co_u32_e32 v20, vcc, 0, v9, vcc
	v_add_co_u32_e32 v12, vcc, s10, v15
	v_addc_co_u32_e32 v13, vcc, v20, v1, vcc
	v_cmp_eq_u64_e32 vcc, 0, v[12:13]
	v_cndmask_b32_e32 v13, v13, v20, vcc
	v_cndmask_b32_e32 v12, v12, v15, vcc
	v_and_b32_e32 v1, v13, v9
	v_and_b32_e32 v8, v12, v8
	v_mul_lo_u32 v1, v1, 24
	v_mul_hi_u32 v9, v8, 24
	v_mul_lo_u32 v8, v8, 24
	v_add_u32_e32 v1, v9, v1
	s_waitcnt vmcnt(0)
	v_add_co_u32_e32 v8, vcc, v18, v8
	v_addc_co_u32_e32 v9, vcc, v19, v1, vcc
	v_mov_b32_e32 v14, v16
	global_store_dwordx2 v[8:9], v[16:17], off
	v_mov_b32_e32 v15, v17
	buffer_wbl2
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[14:15], v35, v[12:15], s[2:3] offset:24 glc
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[14:15], v[16:17]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_27
; %bb.107:                              ;   in Loop: Header=BB0_28 Depth=1
	s_mov_b64 s[0:1], 0
.LBB0_108:                              ;   Parent Loop BB0_28 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_sleep 1
	global_store_dwordx2 v[8:9], v[14:15], off
	buffer_wbl2
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[16:17], v35, v[12:15], s[2:3] offset:24 glc
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[16:17], v[14:15]
	s_or_b64 s[0:1], vcc, s[0:1]
	v_pk_mov_b32 v[14:15], v[16:17], v[16:17] op_sel:[0,1]
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_108
	s_branch .LBB0_27
.LBB0_109:
	s_branch .LBB0_137
.LBB0_110:
                                        ; implicit-def: $vgpr6_vgpr7
	s_cbranch_execz .LBB0_137
; %bb.111:
	v_readfirstlane_b32 s0, v3
	v_cmp_eq_u32_e64 s[0:1], s0, v3
	v_pk_mov_b32 v[12:13], 0, 0
	s_and_saveexec_b64 s[4:5], s[0:1]
	s_cbranch_execz .LBB0_117
; %bb.112:
	v_mov_b32_e32 v1, 0
	global_load_dwordx2 v[8:9], v1, s[2:3] offset:24 glc
	s_waitcnt vmcnt(0)
	buffer_invl2
	buffer_wbinvl1_vol
	global_load_dwordx2 v[6:7], v1, s[2:3] offset:40
	global_load_dwordx2 v[10:11], v1, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v6, v6, v8
	v_and_b32_e32 v7, v7, v9
	v_mul_lo_u32 v7, v7, 24
	v_mul_hi_u32 v12, v6, 24
	v_mul_lo_u32 v6, v6, 24
	v_add_u32_e32 v7, v12, v7
	s_waitcnt vmcnt(0)
	v_add_co_u32_e32 v6, vcc, v10, v6
	v_addc_co_u32_e32 v7, vcc, v11, v7, vcc
	global_load_dwordx2 v[6:7], v[6:7], off glc
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[12:13], v1, v[6:9], s[2:3] offset:24 glc
	s_waitcnt vmcnt(0)
	buffer_invl2
	buffer_wbinvl1_vol
	v_cmp_ne_u64_e32 vcc, v[12:13], v[8:9]
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_116
; %bb.113:
	s_mov_b64 s[8:9], 0
.LBB0_114:                              ; =>This Inner Loop Header: Depth=1
	s_sleep 1
	global_load_dwordx2 v[6:7], v1, s[2:3] offset:40
	global_load_dwordx2 v[10:11], v1, s[2:3]
	v_pk_mov_b32 v[8:9], v[12:13], v[12:13] op_sel:[0,1]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v6, v6, v8
	v_and_b32_e32 v12, v7, v9
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[6:7], s[10:11], v6, 24, v[10:11]
	v_mov_b32_e32 v10, v7
	v_mad_u64_u32 v[10:11], s[10:11], v12, 24, v[10:11]
	v_mov_b32_e32 v7, v10
	global_load_dwordx2 v[6:7], v[6:7], off glc
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[12:13], v1, v[6:9], s[2:3] offset:24 glc
	s_waitcnt vmcnt(0)
	buffer_invl2
	buffer_wbinvl1_vol
	v_cmp_eq_u64_e32 vcc, v[12:13], v[8:9]
	s_or_b64 s[8:9], vcc, s[8:9]
	s_andn2_b64 exec, exec, s[8:9]
	s_cbranch_execnz .LBB0_114
; %bb.115:
	s_or_b64 exec, exec, s[8:9]
.LBB0_116:
	s_or_b64 exec, exec, s[6:7]
.LBB0_117:
	s_or_b64 exec, exec, s[4:5]
	s_waitcnt vmcnt(0)
	v_mov_b32_e32 v6, 0
	global_load_dwordx2 v[14:15], v6, s[2:3] offset:40
	global_load_dwordx4 v[8:11], v6, s[2:3]
	v_readfirstlane_b32 s4, v12
	v_readfirstlane_b32 s5, v13
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s8, v14
	v_readfirstlane_b32 s9, v15
	s_and_b64 s[8:9], s[4:5], s[8:9]
	s_mul_i32 s10, s9, 24
	s_mul_hi_u32 s11, s8, 24
	s_mul_i32 s12, s8, 24
	s_add_i32 s10, s11, s10
	v_mov_b32_e32 v1, s10
	s_waitcnt vmcnt(0)
	v_add_co_u32_e32 v12, vcc, s12, v8
	v_addc_co_u32_e32 v13, vcc, v9, v1, vcc
	s_and_saveexec_b64 s[10:11], s[0:1]
	s_cbranch_execz .LBB0_119
; %bb.118:
	v_pk_mov_b32 v[14:15], s[6:7], s[6:7] op_sel:[0,1]
	v_mov_b32_e32 v16, 2
	v_mov_b32_e32 v17, 1
	global_store_dwordx4 v[12:13], v[14:17], off offset:8
.LBB0_119:
	s_or_b64 exec, exec, s[10:11]
	s_lshl_b64 s[6:7], s[8:9], 12
	v_mov_b32_e32 v1, s7
	v_add_co_u32_e32 v14, vcc, s6, v10
	v_addc_co_u32_e32 v1, vcc, v11, v1, vcc
	s_movk_i32 s6, 0xff1f
	s_mov_b32 s8, 0
	v_and_or_b32 v4, v4, s6, 32
	v_mov_b32_e32 v7, v6
	v_readfirstlane_b32 s6, v14
	v_readfirstlane_b32 s7, v1
	s_mov_b32 s9, s8
	v_add_co_u32_e32 v10, vcc, v14, v33
	s_mov_b32 s10, s8
	s_mov_b32 s11, s8
	s_nop 0
	global_store_dwordx4 v33, v[4:7], s[6:7]
	v_addc_co_u32_e32 v11, vcc, 0, v1, vcc
	v_pk_mov_b32 v[4:5], s[8:9], s[8:9] op_sel:[0,1]
	v_pk_mov_b32 v[6:7], s[10:11], s[10:11] op_sel:[0,1]
	global_store_dwordx4 v33, v[4:7], s[6:7] offset:16
	global_store_dwordx4 v33, v[4:7], s[6:7] offset:32
	global_store_dwordx4 v33, v[4:7], s[6:7] offset:48
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_127
; %bb.120:
	v_mov_b32_e32 v1, 0
	global_load_dwordx2 v[16:17], v1, s[2:3] offset:32 glc
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
	v_mov_b32_e32 v4, s9
	v_add_co_u32_e32 v8, vcc, s8, v8
	v_addc_co_u32_e32 v9, vcc, v9, v4, vcc
	global_store_dwordx2 v[8:9], v[16:17], off
	buffer_wbl2
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[6:7], v1, v[14:17], s[2:3] offset:32 glc
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[6:7], v[16:17]
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_123
; %bb.121:
	s_mov_b64 s[10:11], 0
.LBB0_122:                              ; =>This Inner Loop Header: Depth=1
	s_sleep 1
	global_store_dwordx2 v[8:9], v[6:7], off
	v_mov_b32_e32 v4, s4
	v_mov_b32_e32 v5, s5
	buffer_wbl2
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[4:5], v1, v[4:7], s[2:3] offset:32 glc
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[4:5], v[6:7]
	s_or_b64 s[10:11], vcc, s[10:11]
	v_pk_mov_b32 v[6:7], v[4:5], v[4:5] op_sel:[0,1]
	s_andn2_b64 exec, exec, s[10:11]
	s_cbranch_execnz .LBB0_122
.LBB0_123:
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
	buffer_wbl2
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[4:5], v[6:7], off offset:8
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
	v_readfirstlane_b32 s8, v1
	s_mov_b32 m0, s8
	buffer_wbl2
	global_store_dwordx2 v[6:7], v[4:5], off
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_127:
	s_or_b64 exec, exec, s[6:7]
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
	global_load_dword v1, v[12:13], off offset:20 glc
	s_waitcnt vmcnt(0)
	buffer_invl2
	buffer_wbinvl1_vol
	v_and_b32_e32 v1, 1, v1
	s_branch .LBB0_128
.LBB0_133:
	global_load_dwordx2 v[6:7], v[10:11], off
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_136
; %bb.134:
	v_mov_b32_e32 v1, 0
	global_load_dwordx2 v[4:5], v1, s[2:3] offset:40
	global_load_dwordx2 v[12:13], v1, s[2:3] offset:24 glc
	global_load_dwordx2 v[14:15], v1, s[2:3]
	v_mov_b32_e32 v9, s5
	s_mov_b64 s[0:1], 0
	s_waitcnt vmcnt(2)
	v_add_co_u32_e32 v11, vcc, 1, v4
	v_addc_co_u32_e32 v16, vcc, 0, v5, vcc
	v_add_co_u32_e32 v8, vcc, s4, v11
	v_addc_co_u32_e32 v9, vcc, v16, v9, vcc
	v_cmp_eq_u64_e32 vcc, 0, v[8:9]
	v_cndmask_b32_e32 v9, v9, v16, vcc
	v_cndmask_b32_e32 v8, v8, v11, vcc
	v_and_b32_e32 v5, v9, v5
	v_and_b32_e32 v4, v8, v4
	v_mul_lo_u32 v5, v5, 24
	v_mul_hi_u32 v11, v4, 24
	v_mul_lo_u32 v4, v4, 24
	v_add_u32_e32 v5, v11, v5
	s_waitcnt vmcnt(0)
	v_add_co_u32_e32 v4, vcc, v14, v4
	v_addc_co_u32_e32 v5, vcc, v15, v5, vcc
	v_mov_b32_e32 v10, v12
	global_store_dwordx2 v[4:5], v[12:13], off
	v_mov_b32_e32 v11, v13
	buffer_wbl2
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v1, v[8:11], s[2:3] offset:24 glc
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[10:11], v[12:13]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_136
.LBB0_135:                              ; =>This Inner Loop Header: Depth=1
	s_sleep 1
	global_store_dwordx2 v[4:5], v[10:11], off
	buffer_wbl2
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[12:13], v1, v[8:11], s[2:3] offset:24 glc
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[12:13], v[10:11]
	s_or_b64 s[0:1], vcc, s[0:1]
	v_pk_mov_b32 v[10:11], v[12:13], v[12:13] op_sel:[0,1]
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_135
.LBB0_136:
	s_or_b64 exec, exec, s[6:7]
.LBB0_137:
	v_readfirstlane_b32 s0, v3
	v_cmp_eq_u32_e64 s[0:1], s0, v3
	s_waitcnt vmcnt(0)
	v_pk_mov_b32 v[4:5], 0, 0
	s_and_saveexec_b64 s[4:5], s[0:1]
	s_cbranch_execz .LBB0_143
; %bb.138:
	v_mov_b32_e32 v1, 0
	global_load_dwordx2 v[10:11], v1, s[2:3] offset:24 glc
	s_waitcnt vmcnt(0)
	buffer_invl2
	buffer_wbinvl1_vol
	global_load_dwordx2 v[4:5], v1, s[2:3] offset:40
	global_load_dwordx2 v[8:9], v1, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v4, v4, v10
	v_and_b32_e32 v5, v5, v11
	v_mul_lo_u32 v5, v5, 24
	v_mul_hi_u32 v12, v4, 24
	v_mul_lo_u32 v4, v4, 24
	v_add_u32_e32 v5, v12, v5
	s_waitcnt vmcnt(0)
	v_add_co_u32_e32 v4, vcc, v8, v4
	v_addc_co_u32_e32 v5, vcc, v9, v5, vcc
	global_load_dwordx2 v[8:9], v[4:5], off glc
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[4:5], v1, v[8:11], s[2:3] offset:24 glc
	s_waitcnt vmcnt(0)
	buffer_invl2
	buffer_wbinvl1_vol
	v_cmp_ne_u64_e32 vcc, v[4:5], v[10:11]
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_142
; %bb.139:
	s_mov_b64 s[8:9], 0
.LBB0_140:                              ; =>This Inner Loop Header: Depth=1
	s_sleep 1
	global_load_dwordx2 v[8:9], v1, s[2:3] offset:40
	global_load_dwordx2 v[12:13], v1, s[2:3]
	v_pk_mov_b32 v[10:11], v[4:5], v[4:5] op_sel:[0,1]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v4, v8, v10
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[4:5], s[10:11], v4, 24, v[12:13]
	v_and_b32_e32 v9, v9, v11
	v_mov_b32_e32 v8, v5
	v_mad_u64_u32 v[8:9], s[10:11], v9, 24, v[8:9]
	v_mov_b32_e32 v5, v8
	global_load_dwordx2 v[8:9], v[4:5], off glc
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[4:5], v1, v[8:11], s[2:3] offset:24 glc
	s_waitcnt vmcnt(0)
	buffer_invl2
	buffer_wbinvl1_vol
	v_cmp_eq_u64_e32 vcc, v[4:5], v[10:11]
	s_or_b64 s[8:9], vcc, s[8:9]
	s_andn2_b64 exec, exec, s[8:9]
	s_cbranch_execnz .LBB0_140
; %bb.141:
	s_or_b64 exec, exec, s[8:9]
.LBB0_142:
	s_or_b64 exec, exec, s[6:7]
.LBB0_143:
	s_or_b64 exec, exec, s[4:5]
	v_mov_b32_e32 v9, 0
	global_load_dwordx2 v[14:15], v9, s[2:3] offset:40
	global_load_dwordx4 v[10:13], v9, s[2:3]
	v_readfirstlane_b32 s4, v4
	v_readfirstlane_b32 s5, v5
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s8, v14
	v_readfirstlane_b32 s9, v15
	s_and_b64 s[8:9], s[4:5], s[8:9]
	s_mul_i32 s10, s9, 24
	s_mul_hi_u32 s11, s8, 24
	s_mul_i32 s12, s8, 24
	s_add_i32 s10, s11, s10
	v_mov_b32_e32 v1, s10
	s_waitcnt vmcnt(0)
	v_add_co_u32_e32 v14, vcc, s12, v10
	v_addc_co_u32_e32 v15, vcc, v11, v1, vcc
	s_and_saveexec_b64 s[10:11], s[0:1]
	s_cbranch_execz .LBB0_145
; %bb.144:
	v_pk_mov_b32 v[16:17], s[6:7], s[6:7] op_sel:[0,1]
	v_mov_b32_e32 v18, 2
	v_mov_b32_e32 v19, 1
	global_store_dwordx4 v[14:15], v[16:19], off offset:8
.LBB0_145:
	s_or_b64 exec, exec, s[10:11]
	s_lshl_b64 s[6:7], s[8:9], 12
	v_mov_b32_e32 v1, s7
	v_add_co_u32_e32 v4, vcc, s6, v12
	v_addc_co_u32_e32 v1, vcc, v13, v1, vcc
	s_movk_i32 s6, 0xff1f
	s_mov_b32 s8, 0
	v_and_or_b32 v6, v6, s6, 32
	v_mov_b32_e32 v8, v0
	v_readfirstlane_b32 s6, v4
	v_readfirstlane_b32 s7, v1
	s_mov_b32 s9, s8
	v_add_co_u32_e32 v12, vcc, v4, v33
	s_mov_b32 s10, s8
	s_mov_b32 s11, s8
	s_nop 0
	global_store_dwordx4 v33, v[6:9], s[6:7]
	v_pk_mov_b32 v[4:5], s[8:9], s[8:9] op_sel:[0,1]
	v_addc_co_u32_e32 v13, vcc, 0, v1, vcc
	v_pk_mov_b32 v[6:7], s[10:11], s[10:11] op_sel:[0,1]
	global_store_dwordx4 v33, v[4:7], s[6:7] offset:16
	global_store_dwordx4 v33, v[4:7], s[6:7] offset:32
	global_store_dwordx4 v33, v[4:7], s[6:7] offset:48
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_153
; %bb.146:
	v_mov_b32_e32 v8, 0
	global_load_dwordx2 v[18:19], v8, s[2:3] offset:32 glc
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
	v_mov_b32_e32 v1, s9
	v_add_co_u32_e32 v0, vcc, s8, v10
	v_addc_co_u32_e32 v1, vcc, v11, v1, vcc
	global_store_dwordx2 v[0:1], v[18:19], off
	buffer_wbl2
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[6:7], v8, v[16:19], s[2:3] offset:32 glc
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[6:7], v[18:19]
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_149
; %bb.147:
	s_mov_b64 s[10:11], 0
.LBB0_148:                              ; =>This Inner Loop Header: Depth=1
	s_sleep 1
	global_store_dwordx2 v[0:1], v[6:7], off
	v_mov_b32_e32 v4, s4
	v_mov_b32_e32 v5, s5
	buffer_wbl2
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[4:5], v8, v[4:7], s[2:3] offset:32 glc
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[4:5], v[6:7]
	s_or_b64 s[10:11], vcc, s[10:11]
	v_pk_mov_b32 v[6:7], v[4:5], v[4:5] op_sel:[0,1]
	s_andn2_b64 exec, exec, s[10:11]
	s_cbranch_execnz .LBB0_148
.LBB0_149:
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
	buffer_wbl2
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[0:1], v[4:5], off offset:8
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
	buffer_wbl2
	s_waitcnt vmcnt(0)
	global_store_dwordx2 v[4:5], v[0:1], off
	v_and_b32_e32 v0, 0xffffff, v0
	v_readfirstlane_b32 s8, v0
	s_mov_b32 m0, s8
	s_nop 0
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_153:
	s_or_b64 exec, exec, s[6:7]
	s_branch .LBB0_157
.LBB0_154:                              ;   in Loop: Header=BB0_157 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s6, v0
	s_cmp_eq_u32 s6, 0
	s_cbranch_scc1 .LBB0_156
; %bb.155:                              ;   in Loop: Header=BB0_157 Depth=1
	s_sleep 1
	s_cbranch_execnz .LBB0_157
	s_branch .LBB0_159
.LBB0_156:
	s_branch .LBB0_159
.LBB0_157:                              ; =>This Inner Loop Header: Depth=1
	v_mov_b32_e32 v0, 1
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_154
; %bb.158:                              ;   in Loop: Header=BB0_157 Depth=1
	global_load_dword v0, v[14:15], off offset:20 glc
	s_waitcnt vmcnt(0)
	buffer_invl2
	buffer_wbinvl1_vol
	v_and_b32_e32 v0, 1, v0
	s_branch .LBB0_154
.LBB0_159:
	global_load_dwordx2 v[0:1], v[12:13], off
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_162
; %bb.160:
	v_mov_b32_e32 v10, 0
	global_load_dwordx2 v[8:9], v10, s[2:3] offset:40
	global_load_dwordx2 v[12:13], v10, s[2:3] offset:24 glc
	global_load_dwordx2 v[14:15], v10, s[2:3]
	v_mov_b32_e32 v5, s5
	s_mov_b64 s[0:1], 0
	s_waitcnt vmcnt(2)
	v_add_co_u32_e32 v7, vcc, 1, v8
	v_addc_co_u32_e32 v11, vcc, 0, v9, vcc
	v_add_co_u32_e32 v4, vcc, s4, v7
	v_addc_co_u32_e32 v5, vcc, v11, v5, vcc
	v_cmp_eq_u64_e32 vcc, 0, v[4:5]
	v_cndmask_b32_e32 v5, v5, v11, vcc
	v_cndmask_b32_e32 v4, v4, v7, vcc
	v_and_b32_e32 v7, v5, v9
	v_and_b32_e32 v8, v4, v8
	v_mul_lo_u32 v7, v7, 24
	v_mul_hi_u32 v9, v8, 24
	v_mul_lo_u32 v8, v8, 24
	v_add_u32_e32 v7, v9, v7
	s_waitcnt vmcnt(0)
	v_add_co_u32_e32 v8, vcc, v14, v8
	v_addc_co_u32_e32 v9, vcc, v15, v7, vcc
	v_mov_b32_e32 v6, v12
	global_store_dwordx2 v[8:9], v[12:13], off
	v_mov_b32_e32 v7, v13
	buffer_wbl2
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[6:7], v10, v[4:7], s[2:3] offset:24 glc
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[6:7], v[12:13]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_162
.LBB0_161:                              ; =>This Inner Loop Header: Depth=1
	s_sleep 1
	global_store_dwordx2 v[8:9], v[6:7], off
	buffer_wbl2
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[12:13], v10, v[4:7], s[2:3] offset:24 glc
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[12:13], v[6:7]
	s_or_b64 s[0:1], vcc, s[0:1]
	v_pk_mov_b32 v[6:7], v[12:13], v[12:13] op_sel:[0,1]
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_161
.LBB0_162:
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s0, v3
	v_cmp_eq_u32_e64 s[0:1], s0, v3
	v_pk_mov_b32 v[8:9], 0, 0
	s_and_saveexec_b64 s[4:5], s[0:1]
	s_cbranch_execz .LBB0_168
; %bb.163:
	v_mov_b32_e32 v3, 0
	global_load_dwordx2 v[6:7], v3, s[2:3] offset:24 glc
	s_waitcnt vmcnt(0)
	buffer_invl2
	buffer_wbinvl1_vol
	global_load_dwordx2 v[4:5], v3, s[2:3] offset:40
	global_load_dwordx2 v[8:9], v3, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v4, v4, v6
	v_and_b32_e32 v5, v5, v7
	v_mul_lo_u32 v5, v5, 24
	v_mul_hi_u32 v10, v4, 24
	v_mul_lo_u32 v4, v4, 24
	v_add_u32_e32 v5, v10, v5
	s_waitcnt vmcnt(0)
	v_add_co_u32_e32 v4, vcc, v8, v4
	v_addc_co_u32_e32 v5, vcc, v9, v5, vcc
	global_load_dwordx2 v[4:5], v[4:5], off glc
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[8:9], v3, v[4:7], s[2:3] offset:24 glc
	s_waitcnt vmcnt(0)
	buffer_invl2
	buffer_wbinvl1_vol
	v_cmp_ne_u64_e32 vcc, v[8:9], v[6:7]
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_167
; %bb.164:
	s_mov_b64 s[8:9], 0
.LBB0_165:                              ; =>This Inner Loop Header: Depth=1
	s_sleep 1
	global_load_dwordx2 v[4:5], v3, s[2:3] offset:40
	global_load_dwordx2 v[10:11], v3, s[2:3]
	v_pk_mov_b32 v[6:7], v[8:9], v[8:9] op_sel:[0,1]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v4, v4, v6
	v_and_b32_e32 v9, v5, v7
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[4:5], s[10:11], v4, 24, v[10:11]
	v_mov_b32_e32 v8, v5
	v_mad_u64_u32 v[8:9], s[10:11], v9, 24, v[8:9]
	v_mov_b32_e32 v5, v8
	global_load_dwordx2 v[4:5], v[4:5], off glc
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[8:9], v3, v[4:7], s[2:3] offset:24 glc
	s_waitcnt vmcnt(0)
	buffer_invl2
	buffer_wbinvl1_vol
	v_cmp_eq_u64_e32 vcc, v[8:9], v[6:7]
	s_or_b64 s[8:9], vcc, s[8:9]
	s_andn2_b64 exec, exec, s[8:9]
	s_cbranch_execnz .LBB0_165
; %bb.166:
	s_or_b64 exec, exec, s[8:9]
.LBB0_167:
	s_or_b64 exec, exec, s[6:7]
.LBB0_168:
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
	s_mul_i32 s12, s8, 24
	s_add_i32 s10, s11, s10
	v_mov_b32_e32 v9, s10
	s_waitcnt vmcnt(0)
	v_add_co_u32_e32 v8, vcc, s12, v4
	v_addc_co_u32_e32 v9, vcc, v5, v9, vcc
	s_and_saveexec_b64 s[10:11], s[0:1]
	s_cbranch_execz .LBB0_170
; %bb.169:
	v_pk_mov_b32 v[10:11], s[6:7], s[6:7] op_sel:[0,1]
	v_mov_b32_e32 v12, 2
	v_mov_b32_e32 v13, 1
	global_store_dwordx4 v[8:9], v[10:13], off offset:8
.LBB0_170:
	s_or_b64 exec, exec, s[10:11]
	s_lshl_b64 s[6:7], s[8:9], 12
	v_mov_b32_e32 v10, s7
	v_add_co_u32_e32 v6, vcc, s6, v6
	v_addc_co_u32_e32 v7, vcc, v7, v10, vcc
	s_movk_i32 s6, 0xff1d
	s_mov_b32 s8, 0
	v_and_or_b32 v0, v0, s6, 34
	v_readfirstlane_b32 s6, v6
	v_readfirstlane_b32 s7, v7
	s_mov_b32 s9, s8
	s_mov_b32 s10, s8
	s_mov_b32 s11, s8
	s_nop 1
	global_store_dwordx4 v33, v[0:3], s[6:7]
	s_nop 0
	v_pk_mov_b32 v[0:1], s[8:9], s[8:9] op_sel:[0,1]
	v_pk_mov_b32 v[2:3], s[10:11], s[10:11] op_sel:[0,1]
	global_store_dwordx4 v33, v[0:3], s[6:7] offset:16
	global_store_dwordx4 v33, v[0:3], s[6:7] offset:32
	global_store_dwordx4 v33, v[0:3], s[6:7] offset:48
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_178
; %bb.171:
	v_mov_b32_e32 v6, 0
	global_load_dwordx2 v[12:13], v6, s[2:3] offset:32 glc
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
	v_mov_b32_e32 v0, s9
	v_add_co_u32_e32 v4, vcc, s8, v4
	v_addc_co_u32_e32 v5, vcc, v5, v0, vcc
	global_store_dwordx2 v[4:5], v[12:13], off
	buffer_wbl2
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v6, v[10:13], s[2:3] offset:32 glc
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[2:3], v[12:13]
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_174
; %bb.172:
	s_mov_b64 s[10:11], 0
.LBB0_173:                              ; =>This Inner Loop Header: Depth=1
	s_sleep 1
	global_store_dwordx2 v[4:5], v[2:3], off
	v_mov_b32_e32 v0, s4
	v_mov_b32_e32 v1, s5
	buffer_wbl2
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[0:1], v6, v[0:3], s[2:3] offset:32 glc
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[0:1], v[2:3]
	s_or_b64 s[10:11], vcc, s[10:11]
	v_pk_mov_b32 v[2:3], v[0:1], v[0:1] op_sel:[0,1]
	s_andn2_b64 exec, exec, s[10:11]
	s_cbranch_execnz .LBB0_173
.LBB0_174:
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
	buffer_wbl2
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[0:1], v[2:3], off offset:8
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
	buffer_wbl2
	s_waitcnt vmcnt(0)
	global_store_dwordx2 v[2:3], v[0:1], off
	v_and_b32_e32 v0, 0xffffff, v0
	v_readfirstlane_b32 s8, v0
	s_mov_b32 m0, s8
	s_nop 0
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_178:
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
	global_load_dword v0, v[8:9], off offset:20 glc
	s_waitcnt vmcnt(0)
	buffer_invl2
	buffer_wbinvl1_vol
	v_and_b32_e32 v0, 1, v0
	s_branch .LBB0_179
.LBB0_184:
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_187
; %bb.185:
	v_mov_b32_e32 v6, 0
	global_load_dwordx2 v[4:5], v6, s[2:3] offset:40
	global_load_dwordx2 v[8:9], v6, s[2:3] offset:24 glc
	global_load_dwordx2 v[10:11], v6, s[2:3]
	v_mov_b32_e32 v1, s5
	s_mov_b64 s[0:1], 0
	s_waitcnt vmcnt(2)
	v_add_co_u32_e32 v3, vcc, 1, v4
	v_addc_co_u32_e32 v7, vcc, 0, v5, vcc
	v_add_co_u32_e32 v0, vcc, s4, v3
	v_addc_co_u32_e32 v1, vcc, v7, v1, vcc
	v_cmp_eq_u64_e32 vcc, 0, v[0:1]
	v_cndmask_b32_e32 v1, v1, v7, vcc
	v_cndmask_b32_e32 v0, v0, v3, vcc
	v_and_b32_e32 v3, v1, v5
	v_and_b32_e32 v4, v0, v4
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v5, v4, 24
	v_mul_lo_u32 v4, v4, 24
	v_add_u32_e32 v3, v5, v3
	s_waitcnt vmcnt(0)
	v_add_co_u32_e32 v4, vcc, v10, v4
	v_addc_co_u32_e32 v5, vcc, v11, v3, vcc
	v_mov_b32_e32 v2, v8
	global_store_dwordx2 v[4:5], v[8:9], off
	v_mov_b32_e32 v3, v9
	buffer_wbl2
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v6, v[0:3], s[2:3] offset:24 glc
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[2:3], v[8:9]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_187
.LBB0_186:                              ; =>This Inner Loop Header: Depth=1
	s_sleep 1
	global_store_dwordx2 v[4:5], v[2:3], off
	buffer_wbl2
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[8:9], v6, v[0:3], s[2:3] offset:24 glc
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[8:9], v[2:3]
	s_or_b64 s[0:1], vcc, s[0:1]
	v_pk_mov_b32 v[2:3], v[8:9], v[8:9] op_sel:[0,1]
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_186
.LBB0_187:
	s_endpgm