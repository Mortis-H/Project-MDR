#!/usr/bin/env python3
"""
自動包裝純函數級 AMD GPU 組合語言
將只有指令的 .s 文件轉換為完整的可執行 kernel

用法:
    python3 wrap_pure_asm.py input.s output.s --name vec_add --arch gfx950
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def demangle_cpp_kernel(mangled_name: str) -> Tuple[str, List[str]]:
    """
    簡單的 C++ name demangling (支持常見的 kernel 簽名)
    
    Returns:
        (function_name, param_types)
        
    Example:
        "_Z18MatrixAddGlobalMemPfS_S_iii" -> ("MatrixAddGlobalMem", ["Pf", "Pf", "Pf", "i", "i", "i"])
    """
    if not mangled_name.startswith("_Z"):
        print(f"[WARNING] 不是 C++ mangled name: {mangled_name}")
        return mangled_name, []
    
    # 移除 _Z 前綴
    rest = mangled_name[2:]
    
    # 解析函數名長度
    name_len = 0
    i = 0
    while i < len(rest) and rest[i].isdigit():
        name_len = name_len * 10 + int(rest[i])
        i += 1
    
    if name_len == 0:
        print(f"[WARNING] 無法解析函數名長度: {mangled_name}")
        return mangled_name, []
    
    # 提取函數名
    func_name = rest[i:i+name_len]
    params_str = rest[i+name_len:]
    
    # 解析參數類型
    param_types = []
    substitutions = []  # 存儲已見過的類型用於替換
    
    j = 0
    while j < len(params_str):
        if params_str[j] == 'P':  # Pointer
            if j + 1 < len(params_str):
                base_type = params_str[j+1]
                type_str = 'P' + base_type
                param_types.append(type_str)
                substitutions.append(type_str)
                j += 2
            else:
                j += 1
        elif params_str[j:j+2] == 'S_':  # Substitution - 引用之前的類型
            if substitutions:
                param_types.append(substitutions[0])  # 引用第一個類型
            j += 2
        elif params_str[j] in ['i', 'l', 'f', 'd', 'v', 'b', 's', 'h', 'j', 'x', 'y']:
            # 基本類型: i=int, l=long, f=float, d=double, v=void, 
            #          b=bool, s=short, h=unsigned char, j=unsigned, x=long long, y=unsigned long long
            param_types.append(params_str[j])
            j += 1
        else:
            # 未知類型，跳過
            j += 1
    
    return func_name, param_types


def cpp_type_to_size_and_kind(type_code: str) -> Tuple[int, str, str]:
    """
    將 C++ mangled type code 轉換為 size 和 value_kind
    
    Returns:
        (size, value_kind, type_name)
    """
    # Pointer 類型
    if type_code.startswith('P'):
        base = type_code[1]
        type_map = {
            'f': ('float*', 8, 'global_buffer'),
            'd': ('double*', 8, 'global_buffer'),
            'i': ('int*', 8, 'global_buffer'),
            'c': ('char*', 8, 'global_buffer'),
            'v': ('void*', 8, 'global_buffer'),
        }
        if base in type_map:
            type_name, size, value_kind = type_map[base]
            return size, value_kind, type_name
        return 8, 'global_buffer', 'void*'  # 預設指標
    
    # 標量類型
    scalar_map = {
        'i': (4, 'by_value', 'int'),
        'j': (4, 'by_value', 'unsigned int'),
        'l': (8, 'by_value', 'long'),
        'x': (8, 'by_value', 'long long'),
        'y': (8, 'by_value', 'unsigned long long'),
        'f': (4, 'by_value', 'float'),
        'd': (8, 'by_value', 'double'),
        'b': (1, 'by_value', 'bool'),
        's': (2, 'by_value', 'short'),
        'h': (1, 'by_value', 'unsigned char'),
    }
    
    if type_code in scalar_map:
        return scalar_map[type_code]
    
    return 4, 'by_value', 'int'  # 預設


def analyze_sgpr_mapping(asm_code: str) -> Dict[str, int]:
    """
    分析代碼的 SGPR 映射，確定 kernarg_ptr 位置
    
    Returns:
        {
            'kernarg_base': 基址 (0 或 4),
            'dispatch_ptr': 0 或 1,
            'queue_ptr': 0 或 1,
            'workgroup_id_x': 在代碼中的 SGPR 編號,
            'workgroup_id_y': 在代碼中的 SGPR 編號 (或 None),
            'workgroup_id_z': 在代碼中的 SGPR 編號 (或 None)
        }
    """
    # 查找所有 s_load 指令使用的基址
    s_load_pattern = r's_load\w*\s+s\[\d+:\d+\],\s*s\[(\d+):(\d+)\]'
    matches = re.findall(s_load_pattern, asm_code)
    
    if not matches:
        print("[WARNING] 未找到 s_load 指令，假設 kernarg_ptr 在 s[0:1]")
        kernarg_base = 0
    else:
        # 取最常見的基址作為 kernarg_ptr
        bases = [int(m[0]) for m in matches]
        kernarg_base = max(set(bases), key=bases.count)
    
    print(f"[分析] kernarg_ptr 位於 s[{kernarg_base}:{kernarg_base+1}]")
    
    # 根據 kernarg_base 決定需要啟用的 user SGPRs
    if kernarg_base == 0:
        dispatch_ptr = 0
        queue_ptr = 0
    elif kernarg_base == 4:
        dispatch_ptr = 1  # 佔用 s[0:1]
        queue_ptr = 1     # 佔用 s[2:3]
    else:
        print(f"[WARNING] 不常見的 kernarg_base: {kernarg_base}")
        dispatch_ptr = 0
        queue_ptr = 0
    
    # 尋找 workgroup_id 的使用 (s6, s7, s8 通常是 workgroup IDs)
    workgroup_pattern = r'\bs(6|7|8)\b'
    used_sgprs = set(re.findall(workgroup_pattern, asm_code))
    
    # 推斷哪些維度被使用
    workgroup_ids = {}
    base_sgpr = kernarg_base + 2  # kernarg_ptr 後面
    
    if '6' in used_sgprs or '7' in used_sgprs:
        workgroup_ids['x'] = 6
        workgroup_ids['y'] = 7
    if '8' in used_sgprs:
        workgroup_ids['z'] = 8
    
    return {
        'kernarg_base': kernarg_base,
        'dispatch_ptr': dispatch_ptr,
        'queue_ptr': queue_ptr,
        'workgroup_ids': workgroup_ids
    }


def analyze_workitem_id_encoding(asm_code: str) -> int:
    """
    分析 workitem_id (v0) 的編碼模式
    
    Returns:
        0: 只有 x
        1: x, y packed
        2: x, y, z packed (3D)
    """
    # 查找 bit extraction 操作
    # v_bfe_u32 v1, v0, 10, 10  表示提取 v0[19:10] (y 維度)
    if re.search(r'v_bfe_u32\s+v\d+,\s*v0,\s*10,\s*10', asm_code):
        print("[分析] 檢測到 y 維度解包 → workitem_id mode = 2")
        return 2
    
    # 如果只有 v_and 提取 x 維度
    if re.search(r'v_and.*v0.*0x3ff', asm_code):
        print("[分析] 只檢測到 x 維度 → workitem_id mode = 0")
        return 0
    
    print("[分析] 未檢測到 workitem_id 解包 → 預設 mode = 0")
    return 0


def analyze_register_usage(asm_code: str) -> Tuple[int, int]:
    """
    分析暫存器使用量
    
    Returns:
        (vgpr_count, sgpr_count)
    """
    # 查找所有 VGPR 使用
    vgpr_pattern = r'\bv(\d+)\b'
    vgprs = [int(m) for m in re.findall(vgpr_pattern, asm_code)]
    max_vgpr = max(vgprs) if vgprs else 0
    vgpr_count = max_vgpr + 1
    
    # 查找所有 SGPR 使用
    sgpr_pattern = r'\bs(\d+)\b'
    sgprs = [int(m) for m in re.findall(sgpr_pattern, asm_code)]
    max_sgpr = max(sgprs) if sgprs else 0
    sgpr_count = max_sgpr + 1
    
    print(f"[分析] VGPR 使用: v0-v{max_vgpr} (共 {vgpr_count} 個)")
    print(f"[分析] SGPR 使用: s0-s{max_sgpr} (共 {sgpr_count} 個)")
    
    return vgpr_count, sgpr_count


def analyze_kernarg_accesses(asm_code: str, sgpr_base: int, 
                            explicit_args_size: int = 0) -> Tuple[int, List[Dict]]:
    """
    分析 kernarg 訪問，推斷大小和隱藏參數
    
    Args:
        asm_code: 組合語言代碼
        sgpr_base: kernarg_ptr 所在的 SGPR 編號
        explicit_args_size: 顯式參數的總大小（用於判斷 hidden args）
    
    Returns:
        (kernarg_size, hidden_args)
    """
    # 查找所有從 kernarg_ptr 的 load 操作（支持兩種格式）
    # 格式 1: s_load_* sN, s[base:base+1], 0xOFFSET
    load_pattern1 = rf's_load\w*\s+s\d+,\s*s\[{sgpr_base}:{sgpr_base+1}\],\s*0x([0-9a-fA-F]+)'
    # 格式 2: s_load_* s[N:M], s[base:base+1], 0xOFFSET
    load_pattern2 = rf's_load\w*\s+s\[\d+:\d+\],\s*s\[{sgpr_base}:{sgpr_base+1}\],\s*0x([0-9a-fA-F]+)'
    
    matches1 = re.findall(load_pattern1, asm_code)
    matches2 = re.findall(load_pattern2, asm_code)
    
    offsets = [int(m, 16) for m in matches1 + matches2]
    
    if not offsets:
        print("[分析] 未檢測到 kernarg 訪問，使用預設大小 64")
        return 64, []
    
    offsets_sorted = sorted(set(offsets))
    max_offset = max(offsets)
    print(f"[分析] 檢測到的 kernarg offsets: {[f'0x{o:x}' for o in offsets_sorted]}")
    print(f"[分析] 最大 kernarg offset: 0x{max_offset:x} ({max_offset})")
    
    # 推斷 hidden arguments 的起始位置
    # 顯式參數後通常對齊到 8 或 16 bytes
    if explicit_args_size > 0:
        # 對齊到 16 bytes 邊界
        aligned_explicit_size = ((explicit_args_size + 15) // 16) * 16
        print(f"[分析] 顯式參數大小: {explicit_args_size} bytes (對齊後: {aligned_explicit_size})")
    else:
        # 如果不知道顯式參數大小，假設最小的 offset 是顯式參數
        # Hidden args 通常在最大顯式 offset + padding 之後
        explicit_offsets = [o for o in offsets if o < 48]  # 假設 < 48 是顯式參數
        if explicit_offsets:
            aligned_explicit_size = ((max(explicit_offsets) + 16) // 16) * 16
        else:
            aligned_explicit_size = 48  # 預設值
    
    # 找出超出顯式參數範圍的訪問（可能是 hidden arguments）
    hidden_offsets = [o for o in offsets_sorted if o >= aligned_explicit_size]
    
    hidden_args = []
    if hidden_offsets:
        print(f"[分析] 檢測到疑似 hidden arguments 的訪問: {[f'0x{o:x}' for o in hidden_offsets]}")
        
        # 分析 hidden offsets 的模式
        hidden_args = infer_hidden_args_from_offsets(hidden_offsets, aligned_explicit_size)
    else:
        print("[分析] 未檢測到 hidden arguments")
    
    # kernarg_size 向上取整到 64 的倍數
    kernarg_size = ((max_offset + 16) // 64 + 1) * 64
    print(f"[分析] kernarg_size: {kernarg_size}")
    
    return kernarg_size, hidden_args


def infer_hidden_args_from_offsets(hidden_offsets: List[int], 
                                   base_offset: int) -> List[Dict]:
    """
    從 hidden offsets 推斷 hidden arguments 的類型
    
    Args:
        hidden_offsets: 超出顯式參數的訪問 offsets
        base_offset: hidden arguments 的起始位置
    
    Returns:
        hidden arguments 列表
    """
    if not hidden_offsets:
        return []
    
    hidden_args = []
    processed_offsets = set()
    max_hidden_offset = max(hidden_offsets)
    
    for offset in sorted(hidden_offsets):
        if offset in processed_offsets:
            continue
        
        # 模式 1: 連續的 3 個 2-byte 值 → hidden_group_size_x/y/z (blockDim)
        # 檢查當前 offset 是否可能是 group_size 的起始
        # 通常 group_size 會通過一個 dword load (4 bytes) 訪問，讀取 x 和 y
        # 或者分別訪問，但位置是連續的
        
        # 簡化邏輯：如果是第一個 hidden offset，且範圍足夠，假設是 group_size
        if offset == min(hidden_offsets) and offset + 4 <= max_hidden_offset + 2:
            print(f"[推斷] offset 0x{offset:x} → hidden_group_size_x/y/z")
            hidden_args.extend([
                {'offset': offset, 'size': 2, 'value_kind': 'hidden_group_size_x'},
                {'offset': offset + 2, 'size': 2, 'value_kind': 'hidden_group_size_y'},
                {'offset': offset + 4, 'size': 2, 'value_kind': 'hidden_group_size_z'},
            ])
            processed_offsets.update([offset, offset + 2, offset + 4])
            break  # 通常只有一組 group_size
        
        # 模式 2: 8-byte 對齊的值 → 可能是 global_offset 或其他指針
        elif offset % 8 == 0 and offset not in processed_offsets:
            print(f"[推斷] offset 0x{offset:x} → 可能是其他 hidden argument")
            # 暫時不添加，避免錯誤推斷
            processed_offsets.add(offset)
    
    if not hidden_args:
        # 如果無法精確推斷，檢查是否有常見的 hidden_group_size 位置
        # AMD HSA 通常將 group_size 放在特定對齊位置
        for offset in hidden_offsets:
            # 如果 offset 是 4 的倍數，可能是 group_size
            if offset % 4 == 0:
                print(f"[推斷] offset 0x{offset:x} → hidden_group_size_x/y/z (基於對齊)")
                hidden_args.extend([
                    {'offset': offset, 'size': 2, 'value_kind': 'hidden_group_size_x'},
                    {'offset': offset + 2, 'size': 2, 'value_kind': 'hidden_group_size_y'},
                    {'offset': offset + 4, 'size': 2, 'value_kind': 'hidden_group_size_z'},
                ])
                break
    
    return hidden_args


def infer_explicit_args(asm_code: str, mangled_name: Optional[str] = None) -> List[Dict]:
    """
    從代碼推斷顯式參數
    
    如果提供 mangled_name，會嘗試從 C++ name mangling 解析參數類型
    否則使用預設模板
    """
    if mangled_name:
        func_name, param_types = demangle_cpp_kernel(mangled_name)
        
        if param_types:
            print(f"[分析] 從 mangled name 解析出 {len(param_types)} 個參數")
            print(f"[分析] 函數名: {func_name}")
            print(f"[分析] 參數類型: {param_types}")
            
            args = []
            offset = 0
            
            for i, type_code in enumerate(param_types):
                size, value_kind, type_name = cpp_type_to_size_and_kind(type_code)
                
                arg = {
                    'name': f'arg{i}',
                    'type_name': type_name,
                    'size': size,
                    'offset': offset,
                    'value_kind': value_kind,
                }
                
                # 如果是指標類型，添加 address_space
                if value_kind == 'global_buffer':
                    arg['address_space'] = 'global'
                
                args.append(arg)
                offset += size
            
            print(f"[分析] 推斷出 kernarg 大小: {offset} bytes")
            return args
    
    # 沒有 mangled name，使用預設模板
    print("[分析] 推斷顯式參數（使用預設模板）")
    return [
        {'name': 'A', 'type_name': 'float*', 'size': 8, 'offset': 0, 
         'value_kind': 'global_buffer', 'address_space': 'global'},
        {'name': 'B', 'type_name': 'float*', 'size': 8, 'offset': 8,
         'value_kind': 'global_buffer', 'address_space': 'global'},
        {'name': 'S', 'type_name': 'float*', 'size': 8, 'offset': 16,
         'value_kind': 'global_buffer', 'address_space': 'global'},
        {'name': 'arg3', 'type_name': 'int', 'size': 4, 'offset': 24,
         'value_kind': 'by_value'},
        {'name': 'arg4', 'type_name': 'int', 'size': 4, 'offset': 28,
         'value_kind': 'by_value'},
        {'name': 'arg5', 'type_name': 'int', 'size': 4, 'offset': 32,
         'value_kind': 'by_value'},
    ]


def generate_header(kernel_name: str, arch: str = "gfx950") -> str:
    """生成文件頭部"""
    return f""".amdgcn_target "amdgcn-amd-amdhsa--{arch}"
.amdhsa_code_object_version 6
.text
.protected\t{kernel_name}
.globl\t{kernel_name}
.p2align\t8
.type\t{kernel_name},@function

"""


def generate_amdhsa_metadata(kernel_name: str, config: Dict) -> str:
    """生成 AMDHSA kernel metadata"""
    lines = [
        "\t.section\t.rodata,\"a\",@progbits",
        "\t.p2align\t6, 0x0",
        f"\t.amdhsa_kernel {kernel_name}",
        "\t\t.amdhsa_group_segment_fixed_size 0",
        "\t\t.amdhsa_private_segment_fixed_size 0",
        f"\t\t.amdhsa_kernarg_size {config['kernarg_size']}",
    ]
    
    # User SGPR 配置
    user_sgpr_count = 2  # kernarg_ptr
    if config['dispatch_ptr']:
        user_sgpr_count += 2
    if config['queue_ptr']:
        user_sgpr_count += 2
    
    lines.extend([
        f"\t\t.amdhsa_user_sgpr_count {user_sgpr_count}",
        f"\t\t.amdhsa_user_sgpr_dispatch_ptr {config['dispatch_ptr']}",
        f"\t\t.amdhsa_user_sgpr_queue_ptr {config['queue_ptr']}",
        "\t\t.amdhsa_user_sgpr_kernarg_segment_ptr 1",
        "\t\t.amdhsa_user_sgpr_dispatch_id 0",
        "\t\t.amdhsa_user_sgpr_kernarg_preload_length 0",
        "\t\t.amdhsa_user_sgpr_kernarg_preload_offset 0",
        "\t\t.amdhsa_user_sgpr_private_segment_size 0",
        "\t\t.amdhsa_uses_dynamic_stack 0",
        "\t\t.amdhsa_enable_private_segment 0",
    ])
    
    # System SGPR (workgroup IDs)
    wg_ids = config['workgroup_ids']
    lines.extend([
        f"\t\t.amdhsa_system_sgpr_workgroup_id_x {1 if 'x' in wg_ids else 0}",
        f"\t\t.amdhsa_system_sgpr_workgroup_id_y {1 if 'y' in wg_ids else 0}",
        f"\t\t.amdhsa_system_sgpr_workgroup_id_z {1 if 'z' in wg_ids else 0}",
        "\t\t.amdhsa_system_sgpr_workgroup_info 0",
    ])
    
    # VGPR workitem_id
    lines.append(f"\t\t.amdhsa_system_vgpr_workitem_id {config['workitem_id_mode']}")
    
    # 暫存器計數
    # accum_offset 必須是 4 的倍數，並且至少為 4
    accum_offset = max(4, ((config['vgpr_count'] + 3) // 4) * 4)
    lines.extend([
        f"\t\t.amdhsa_next_free_vgpr {config['vgpr_count']}",
        f"\t\t.amdhsa_next_free_sgpr {config['sgpr_count']}",
        f"\t\t.amdhsa_accum_offset {accum_offset}",
        "\t\t.amdhsa_reserve_vcc 1",
    ])
    
    # 浮點和例外設置
    lines.extend([
        "\t\t.amdhsa_float_round_mode_32 0",
        "\t\t.amdhsa_float_round_mode_16_64 0",
        "\t\t.amdhsa_float_denorm_mode_32 3",
        "\t\t.amdhsa_float_denorm_mode_16_64 3",
        "\t\t.amdhsa_dx10_clamp 1",
        "\t\t.amdhsa_ieee_mode 1",
        "\t\t.amdhsa_fp16_overflow 0",
        "\t\t.amdhsa_tg_split 0",
        "\t\t.amdhsa_exception_fp_ieee_invalid_op 0",
        "\t\t.amdhsa_exception_fp_denorm_src 0",
        "\t\t.amdhsa_exception_fp_ieee_div_zero 0",
        "\t\t.amdhsa_exception_fp_ieee_overflow 0",
        "\t\t.amdhsa_exception_fp_ieee_underflow 0",
        "\t\t.amdhsa_exception_fp_ieee_inexact 0",
        "\t\t.amdhsa_exception_int_div_zero 0",
        "\t.end_amdhsa_kernel",
    ])
    
    return "\n".join(lines) + "\n"


def generate_footer(kernel_name: str) -> str:
    """生成文件尾部"""
    return f"""\t.text
.Lfunc_end0:
\t.size\t{kernel_name}, .Lfunc_end0-{kernel_name}

"""


def generate_yaml_metadata(kernel_name: str, config: Dict, 
                           explicit_args: List[Dict], 
                           hidden_args: List[Dict]) -> str:
    """生成 AMDGPU YAML metadata"""
    lines = [
        "\t.amdgpu_metadata",
        "---",
        "amdhsa.version:",
        "  - 1",
        "  - 2",
        "amdhsa.kernels:",
        f"  - .name:           {kernel_name}",
        f"    .symbol:         {kernel_name}.kd",
        "    .language:       OpenCL C",
        "    .language_version:",
        "      - 2",
        "      - 0",
        f"    .kernarg_segment_size: {config['kernarg_size']}",
        "    .kernarg_segment_align: 8",
        "    .group_segment_fixed_size: 0",
        "    .private_segment_fixed_size: 0",
        f"    .sgpr_count:     {config['sgpr_count']}",
        f"    .vgpr_count:     {config['vgpr_count']}",
        "    .max_flat_workgroup_size: 256",
        "    .wavefront_size: 64",
        "    .sgpr_spill_count: 0",
        "    .vgpr_spill_count: 0",
        "    .args:",
    ]
    
    # 添加顯式參數
    for arg in explicit_args:
        lines.append(f"      - .name:           {arg['name']}")
        if 'type_name' in arg:
            lines.append(f"        .type_name:      '{arg['type_name']}'")
        lines.append(f"        .size:           {arg['size']}")
        lines.append(f"        .offset:         {arg['offset']}")
        lines.append(f"        .value_kind:     {arg['value_kind']}")
        if 'address_space' in arg:
            lines.append(f"        .address_space:  {arg['address_space']}")
    
    # 添加隱藏參數
    for arg in hidden_args:
        lines.append(f"      - .offset:         {arg['offset']}")
        lines.append(f"        .size:           {arg['size']}")
        lines.append(f"        .value_kind:     {arg['value_kind']}")
    
    lines.extend([
        "...",
        "\t.end_amdgpu_metadata",
        ""
    ])
    
    return "\n".join(lines) + "\n"


def extract_mangled_name_from_asm(asm_code: str) -> Optional[str]:
    """
    從組合語言文件中提取 C++ mangled name
    
    查找格式如: _Z18MatrixAddGlobalMemPfS_S_iii:        ; @_Z18MatrixAddGlobalMemPfS_S_iii
    """
    lines = asm_code.split('\n')
    for line in lines[:10]:  # 只檢查前10行
        # 匹配標籤行: <name>:
        match = re.match(r'^(_Z\w+):\s*(?:;|$)', line)
        if match:
            mangled = match.group(1)
            print(f"[檢測] 從文件中提取到 mangled name: {mangled}")
            return mangled
    return None


def wrap_pure_asm(input_file: Path, output_file: Path, 
                  kernel_name: str, arch: str = "gfx950",
                  mangled_name: Optional[str] = None):
    """主函數：包裝純函數級組合語言"""
    
    print(f"\n{'='*60}")
    print(f"包裝純函數級組合語言")
    print(f"{'='*60}")
    print(f"輸入: {input_file}")
    print(f"輸出: {output_file}")
    print(f"Kernel: {kernel_name}")
    print(f"架構: {arch}")
    if mangled_name:
        print(f"手動指定 Mangled name: {mangled_name}")
    print()
    
    # 讀取輸入文件
    asm_code = input_file.read_text()
    
    # 如果沒有手動指定 mangled_name，嘗試從文件中自動提取
    if not mangled_name:
        mangled_name = extract_mangled_name_from_asm(asm_code)
    
    # 移除文件開頭的 mangled name 標籤行（如果存在）
    lines = asm_code.split('\n')
    cleaned_lines = []
    skip_first_label = True
    
    for line in lines:
        # 跳過第一個 C++ mangled name 標籤
        if skip_first_label and re.match(r'^_Z\w+:\s*(?:;|$)', line):
            print(f"[處理] 移除原始標籤行: {line.strip()}")
            skip_first_label = False
            continue
        # 跳過空行（直到遇到第一條指令）
        if not cleaned_lines and not line.strip():
            continue
        cleaned_lines.append(line)
    
    asm_code = '\n'.join(cleaned_lines)
    
    # 步驟 1: 分析代碼
    print("\n步驟 1: 分析代碼結構")
    print("-" * 60)
    
    sgpr_mapping = analyze_sgpr_mapping(asm_code)
    workitem_id_mode = analyze_workitem_id_encoding(asm_code)
    vgpr_count, sgpr_count = analyze_register_usage(asm_code)
    
    # 先推斷顯式參數（如果有 mangled name）
    explicit_args = infer_explicit_args(asm_code, mangled_name)
    
    # 計算顯式參數的總大小
    explicit_args_size = 0
    if explicit_args:
        # 找出最大的 offset + size
        for arg in explicit_args:
            arg_end = arg['offset'] + arg['size']
            if arg_end > explicit_args_size:
                explicit_args_size = arg_end
    
    # 使用顯式參數大小來幫助推斷 hidden arguments
    kernarg_size, hidden_args = analyze_kernarg_accesses(
        asm_code, sgpr_mapping['kernarg_base'], explicit_args_size
    )
    
    # 整合配置
    config = {
        'dispatch_ptr': sgpr_mapping['dispatch_ptr'],
        'queue_ptr': sgpr_mapping['queue_ptr'],
        'workgroup_ids': sgpr_mapping['workgroup_ids'],
        'workitem_id_mode': workitem_id_mode,
        'vgpr_count': vgpr_count,
        'sgpr_count': sgpr_count,
        'kernarg_size': kernarg_size,
    }
    
    print()
    print("步驟 2: 生成完整的 kernel 文件")
    print("-" * 60)
    
    # 生成各部分
    header = generate_header(kernel_name, arch)
    
    # 包裝函數體（添加 kernel 名稱標籤）
    function_body = f"{kernel_name}:        ; @{kernel_name}\n"
    function_body += asm_code
    if not function_body.endswith('\n'):
        function_body += '\n'
    
    amdhsa_metadata = generate_amdhsa_metadata(kernel_name, config)
    footer = generate_footer(kernel_name)
    yaml_metadata = generate_yaml_metadata(
        kernel_name, config, explicit_args, hidden_args
    )
    
    # 組合完整文件
    full_content = (
        header +
        function_body +
        "\n" +
        amdhsa_metadata +
        footer +
        yaml_metadata
    )
    
    # 寫入輸出文件
    output_file.write_text(full_content)
    
    print(f"\n✓ 成功生成: {output_file}")
    print(f"  總行數: {len(full_content.splitlines())}")
    print(f"  檔案大小: {len(full_content)} bytes")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="自動包裝純函數級 AMD GPU 組合語言",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 最簡用法（自動提取 mangled name 並推斷參數類型）
  %(prog)s input.s -n vec_add
  
  # 完全自動（從文件名推斷 kernel 名稱）
  %(prog)s input.s
  
  # 手動指定 mangled name（覆蓋自動提取）
  %(prog)s input.s -n vec_add -m _Z18MatrixAddGlobalMemPfS_S_iii
  
  # 指定輸出文件和架構
  %(prog)s input.s output.s -a gfx908
  
  # 自定義後綴
  %(prog)s input.s --suffix _auto
  
  # 完整範例
  %(prog)s pure_kernel.s wrapped.s -n my_kernel -a gfx908
  
  # 工作流程：
  #   1. 自動從文件第一行提取 C++ mangled name (如 _Z18MatrixAddGlobalMemPfS_S_iii)
  #   2. 解析 mangled name 得到參數類型 (float*, float*, float*, int, int, int)
  #   3. 生成正確的 kernarg metadata
  #   4. 輸出包裝後的完整 kernel
  
  # 輸出說明：
  #   input.s → input_wrapped.s (kernel名: input)
  #   my_kernel_pure.s → my_kernel_pure_wrapped.s (kernel名: my_kernel)
        """
    )
    
    parser.add_argument(
        "input",
        type=Path,
        help="輸入的純函數級組合語言文件"
    )
    parser.add_argument(
        "output",
        type=Path,
        nargs='?',
        default=None,
        help="輸出的完整 kernel 文件 [default: <input>_wrapped.s]"
    )
    parser.add_argument(
        "-n", "--name",
        default="vec_add",
        help="Kernel 函數名稱 [default: vec_add]"
    )
    parser.add_argument(
        "-a", "--arch",
        default="gfx950",
        help="目標 GPU 架構 [default: gfx950]"
    )
    parser.add_argument(
        "-s", "--suffix",
        default="_wrapped",
        help="輸出文件名後綴 [default: _wrapped]"
    )
    parser.add_argument(
        "-m", "--mangled-name",
        default=None,
        help="C++ mangled kernel 名稱 (可選，會自動從文件中提取)"
    )
    
    args = parser.parse_args()
    
    # 檢查輸入文件存在
    if not args.input.exists():
        print(f"錯誤: 輸入文件不存在: {args.input}", file=sys.stderr)
        return 1
    
    input_stem = args.input.stem
    input_parent = args.input.parent
    
    # 如果沒有指定 kernel 名稱，從輸入文件名推斷
    if args.name is None:
        # 移除常見的後綴
        kernel_name = input_stem
        for suffix in ['_pure', '_raw', '_function', '_kernel', '_asm']:
            if kernel_name.endswith(suffix):
                kernel_name = kernel_name[:-len(suffix)]
                break
        args.name = kernel_name
        print(f"[INFO] 自動設定 kernel 名稱: {args.name}")
    
    # 如果沒有指定輸出文件，自動生成
    if args.output is None:
        args.output = input_parent / f"{input_stem}{args.suffix}.s"
        print(f"[INFO] 自動設定輸出文件: {args.output}")
    
    try:
        wrap_pure_asm(args.input, args.output, args.name, args.arch, args.mangled_name)
        return 0
    except Exception as e:
        print(f"\n錯誤: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

