#!/usr/bin/env python3
"""
AMD ISA Assembly Debug Tool
============================

在組合語言（.s 檔案）中插入 printf 除錯功能。

使用方式：
1. 在 .s 檔案中以註解形式標註要印出的內容（f-string 風格）：
   ; @PRINT f"value: {v6:.3f}"
   ; @PRINT f"idx: {v0:d}"
   ; @PRINT f"regs: v4={v4:.3f}, v5={v5:.3f}"

2. 執行此工具：
   # 基本用法（生成 HSACO）
   python3 asm_debug.py input.s --output-dir debug_output

   # 純功能驗證（不注入 printf）
   python3 asm_debug.py input.s --output-dir output --no-printf --test

3. 工具會：
   - 解析 @PRINT 標記
   - 轉換為 GPU MLIR（使用 amdisa-translate）
   - 注入 printf 指令（包含 register clobbering）
   - 修復 ISA metadata（保留 hidden_hostcall_buffer）
   - 生成除錯版本的 HSACO

支援的類型：
- f32, f64: 浮點數
- i32, i64: 整數
- ptr: 指標（列印為 hex）

注意事項：
- printf 功能需要 HIP runtime 的 hostcall 支援
- 使用 hipModuleLaunchKernel 載入的 HSACO 可能無法正確顯示 printf 輸出
- 建議先使用 --no-printf --test 驗證基本功能
"""

import argparse
import pathlib
import re
import shutil
import subprocess
import sys
import yaml
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple


BUILTIN_VARS = {
    '$tid',
    '$lane',
    '$tgid_x',
    '$tgid_y',
    '$tgid_z',
    '$block_x',
    '$block_y',
    '$block_z',
    '$blockidx_x',
    '$blockidx_y',
    '$blockidx_z',
}

BUILTIN_ALIASES = {
    '$block_x': '$tgid_x',
    '$block_y': '$tgid_y',
    '$block_z': '$tgid_z',
    '$blockidx_x': '$tgid_x',
    '$blockidx_y': '$tgid_y',
    '$blockidx_z': '$tgid_z',
}


def normalize_builtin(var_name: str) -> str:
    return BUILTIN_ALIASES.get(var_name, var_name)


# ============================================================
# @PRINT 標記解析
# ============================================================

@dataclass
class PrintDirective:
    """
    代表一個 @PRINT 指令
    
    支援兩種模式：
    1. 直接印出暫存器值：reg=v6,v7 type=f32,f32
    2. 計算表達式後印出：expr="v6 * v7" type=f32
    
    條件式 printf：
    - cond=v6_gt(2.0) - 只印出 v6 > 2.0 的 thread
    - 條件暫存器和類型自動從 cond= 提取
    - 如果條件暫存器不在 reg= 中，會自動創建快照
    
    內建變數：
    - {$tid} - Local Thread ID (workitem_id_x, 0 ~ workgroup_size-1)
    - {$lane} - Wavefront Lane ID (0-63)
    - {$tgid_x/$tgid_y/$tgid_z} - Workgroup ID (block index)
      (aliases: {$block_x/$block_y/$block_z}, {$blockidx_x/$blockidx_y/$blockidx_z})
    - {$tgid_x/$tgid_y/$tgid_z} - Workgroup ID (block index)
      (aliases: {$block_x/$block_y/$block_z}, {$blockidx_x/$blockidx_y/$blockidx_z})
    """
    line_number: int           # 在 .s 檔案中的行號（0-based）
    format_string: str         # printf 格式字串
    registers: List[str]       # 要印出的暫存器列表（如 ["v6", "v7"]）
    types: List[str]           # 對應的類型（如 ["f32", "f32"]）
    condition: Optional[str] = None  # 條件表達式（如 "v6_eq(0.0)"）
    condition_register: Optional[str] = None  # 條件暫存器（自動從 cond 提取）
    condition_type: Optional[str] = None  # 條件暫存器的類型（自動從比較值推導）
    next_instruction: Optional[str] = None  # @PRINT 之後的第一條 ISA 指令
    expressions: Optional[List[str]] = None  # 計算表達式列表（如 ["v6 * v7"]）
    uses_tid: bool = False     # 是否使用了 $tid 內建變數
    uses_lane: bool = False    # 是否使用了 $lane 內建變數
    uses_tgid: bool = False    # 是否使用了 $tgid_x/$tgid_y/$tgid_z 內建變數
    all_placeholders: Optional[List[Tuple[str, str]]] = None  # 所有 placeholder 的原始順序 [(value, type), ...]
    trace_max: Optional[int] = None  # 最大 trace 次數（使用 VGPR 快照多次）
    
    def __str__(self):
        cond_str = f" [cond={self.condition}]" if self.condition else ""
        builtin_str = ""
        if self.uses_tid or self.uses_lane or self.uses_tgid:
            builtins = []
            if self.uses_tid:
                builtins.append("$tid")
            if self.uses_lane:
                builtins.append("$lane")
            if self.uses_tgid:
                builtins.append("$tgid_*")
            builtin_str = f" [builtins={','.join(builtins)}]"
        trace_str = f" [max={self.trace_max}]" if self.trace_max else ""
        if self.expressions:
            return f"@PRINT at line {self.line_number}: {self.format_string} (expr={self.expressions}){cond_str}{builtin_str}{trace_str}"
        return f"@PRINT at line {self.line_number}: {self.format_string} ({self.registers}){cond_str}{builtin_str}{trace_str}"


@dataclass
class TimestampDirective:
    """
    代表一個 @TIMESTAMP_START 或 @TIMESTAMP_END 指令
    
    用於 kernel 內部時間測量。根據模式選擇不同的時間戳指令：
    - local 模式: s_memtime（高精度，~1.7 GHz，單 wavefront 內使用）
    - wallclock/atomic 模式: s_memrealtime（全域同步，~100 MHz，跨 workgroup 使用）
    
    使用方式：
    ; @TIMESTAMP_START [label="section_name"]
    ; ... kernel code ...
    ; @TIMESTAMP_END [label="section_name"]
    
    簡單模式（local，每個 wavefront 獨立測量）：
    ; @TIMESTAMP_START
    ; ... kernel code ...
    ; @TIMESTAMP_END
    
    Wallclock 模式（輸出絕對時間戳，用於計算 kernel-wide 時間）：
    ; @TIMESTAMP_START mode=wallclock
    ; ... kernel code ...
    ; @TIMESTAMP_END mode=wallclock
    
    Atomic 模式（使用 global atomic 操作獲得真正的 kernel wall time）：
    ; @TIMESTAMP_START mode=atomic
    ; ... kernel code ...
    ; @TIMESTAMP_END mode=atomic
    
    輸出格式：
    - local 模式: [Timestamp label] elapsed = X ticks (~1.7 GHz)
    - Wallclock 模式: [WallClock label] start=HI:LO elapsed=X ticks (~100 MHz)
    - Atomic 模式: Host 端讀取 timestamp buffer 計算 kernel_wall_time (~100 MHz)
    
    Atomic 模式說明：
    - 使用 s_memrealtime（全域同步計數器，~100 MHz）
    - Buffer layout: [min_start(8), max_end(8)] 位於 C + N * sizeof(float)
    - 使用 global_atomic_umin_x2 和 global_atomic_umax_x2 (GFX9/CDNA)
    - 與 rocprofv3 測量結果誤差約 3-15%
    """
    line_number: int           # 在 .s 檔案中的行號（0-based）
    directive_type: str        # "start" 或 "end"
    label: Optional[str] = None  # 可選的標籤名稱
    next_instruction: Optional[str] = None  # directive 之後的第一條 ISA 指令
    condition: Optional[str] = None  # 條件式（如 "if $lane == 0:"）
    mode: str = "local"        # 模式: "local", "wallclock", 或 "atomic"
    
    def __str__(self):
        label_str = f" [{self.label}]" if self.label else ""
        cond_str = f" [cond={self.condition}]" if self.condition else ""
        mode_str = f" [mode={self.mode}]" if self.mode != "local" else ""
        return f"@TIMESTAMP_{self.directive_type.upper()} at line {self.line_number + 1}{label_str}{mode_str}{cond_str}"


def validate_variable(var_name: str, line_number: int = None) -> bool:
    """
    驗證變數名稱是否合法
    
    合法的變數：
    - v\\d+ - VGPR（如 v0, v6, v123）
    - s\\d+ - SGPR（如 s0, s4, s19）
    - $tid - 內建變數（Local Thread ID）
    - $lane - 內建變數（Wavefront Lane ID）
    - 表達式（包含 +, -, *, / 的組合）
    
    Returns:
        True 如果合法
    
    Raises:
        ValueError 如果不合法
    """
    var_name = var_name.strip()
    
    # 內建變數
    if var_name in BUILTIN_VARS:
        return True
    
    # 簡單暫存器 v\d+, s\d+, or a\d+ (accumulator VGPR)
    if re.match(r'^[vsa]\d+$', var_name):
        return True
    
    # 表達式：檢查是否包含運算符
    if any(op in var_name for op in ['+', '-', '*', '/', '(', ')']):
        # 提取表達式中的所有變數
        tokens = re.findall(r'[vsa]\d+|\$\w+', var_name)
        for token in tokens:
            if token.startswith('$'):
                if token not in BUILTIN_VARS:
                    line_info = f" (line {line_number + 1})" if line_number is not None else ""
                    raise ValueError(
                        f"Unknown built-in variable '{token}'{line_info}. "
                        "Valid built-in variables are: $tid, $lane, $tgid_x/$tgid_y/$tgid_z "
                        "(aliases: $block_x/$block_y/$block_z, $blockidx_x/$blockidx_y/$blockidx_z)"
                    )
            elif not re.match(r'^[vsa]\d+$', token):
                line_info = f" (line {line_number + 1})" if line_number is not None else ""
                raise ValueError(f"Invalid register '{token}'{line_info}. Expected format: v<N>, s<N>, or a<N>")
        return True
    
    # 如果以 $ 開頭但不是已知的內建變數
    if var_name.startswith('$'):
        line_info = f" (line {line_number + 1})" if line_number is not None else ""
        raise ValueError(
            f"Unknown built-in variable '{var_name}'{line_info}. "
            "Valid built-in variables are: $tid, $lane, $tgid_x/$tgid_y/$tgid_z "
            "(aliases: $block_x/$block_y/$block_z, $blockidx_x/$blockidx_y/$blockidx_z)"
        )
    
    # 不是暫存器也不是表達式
    line_info = f" (line {line_number + 1})" if line_number is not None else ""
    raise ValueError(f"Invalid variable '{var_name}'{line_info}. Expected: v<N>, s<N>, $tid, $lane, or an expression")


def parse_fstring_format(fstring: str, line_number: int = None) -> Tuple[str, List[str], List[str]]:
    """
    解析 Python f-string 風格的格式字串
    
    輸入: f"Before: A={v6:.3f}, B={v7:.2f}, idx={s4:d}"
    輸出: ("Before: A=%0.3f, B=%0.2f, idx=%d", ["v6", "v7", "s4"], ["f32", "f32", "i32"])
    
    支援的格式說明符：
    - {v6} - 預設格式（VGPR 預設 f32，SGPR 預設 i32）
    - {v6:f} 或 {v6:.f} - 浮點數 f32
    - {v6:.3f} - 浮點數，3 位小數
    - {v6:d} - 整數 i32
    - {v6:ld} - 長整數 i64
    - {expr} - 表達式（如 {v6*v7:.2f}）
    
    內建變數：
    - {$tid} - Local Thread ID (workitem_id_x, 0 ~ workgroup_size-1)
    - {$lane} - Wavefront Lane ID (0-63)
    """
    registers = []
    types = []
    
    # 尋找所有 {xxx} 或 {xxx:格式} 的 placeholder
    # 支援暫存器名稱或表達式
    pattern = r'\{([^}]+)\}'
    
    def replace_placeholder(match):
        content = match.group(1)
        
        # 分離暫存器/表達式和格式說明符
        if ':' in content:
            reg_or_expr, fmt_spec = content.rsplit(':', 1)
        else:
            reg_or_expr = content
            fmt_spec = ''
        
        reg_or_expr = reg_or_expr.strip()
        fmt_spec = fmt_spec.strip()
        
        # 驗證變數名稱
        validate_variable(reg_or_expr, line_number)
        
        # 檢查是否是內建變數
        reg_or_expr = normalize_builtin(reg_or_expr)
        if reg_or_expr in ('$tid', '$lane', '$tgid_x', '$tgid_y', '$tgid_z'):
            # 內建變數：$tid (local thread ID) 或 $lane (wavefront lane ID)
            registers.append(reg_or_expr)
            types.append('i32')  # thread ID 總是 i32
            return '%d'
        
        # 判斷是暫存器還是表達式
        is_simple_reg = re.match(r'^[vsa]\d+$', reg_or_expr)
        
        registers.append(reg_or_expr)
        
        # 根據格式說明符推導類型和 printf 格式
        if fmt_spec:
            if fmt_spec == 'd':
                types.append('i32')
                return '%d'
            elif fmt_spec == 'ld':
                types.append('i64')
                return '%ld'
            elif fmt_spec == 'x':
                types.append('i32')
                return '%x'
            elif fmt_spec == 'u':
                types.append('i32')
                return '%u'
            elif fmt_spec == 'f' or fmt_spec == '.f':
                types.append('f32')
                return '%f'
            elif fmt_spec == 'lf':
                types.append('f64')
                return '%f'
            elif re.match(r'\.?\d*f$', fmt_spec):
                # 如 .3f, 3f, .f
                types.append('f32')
                # 轉換為 printf 格式
                if fmt_spec.startswith('.'):
                    return f'%{fmt_spec[:-1]}f'  # .3f -> %0.3f
                else:
                    return f'%.{fmt_spec[:-1]}f'  # 3f -> %.3f
            else:
                # 未知格式，預設
                if is_simple_reg and reg_or_expr.startswith('s'):
                    types.append('i32')
                    return '%d'
                else:
                    types.append('f32')
                    return '%f'
        else:
            # 無格式說明符，根據暫存器類型推導
            if is_simple_reg and reg_or_expr.startswith('s'):
                types.append('i32')
                return '%d'
            else:
                types.append('f32')
                return '%f'
    
    printf_format = re.sub(pattern, replace_placeholder, fstring)
    
    return printf_format, registers, types


def parse_condition_pythonic(cond_str: str, line_number: int = None) -> Optional[str]:
    """
    解析 Python 風格的條件式
    
    輸入: "if v6 > 2.0:" 或 "if s4 == 64:" 或 "if $tid < 10:" 或 "if $lane == 0:"
    輸出: "v6_gt(2.0)" 或 "s4_eq(64)" 或 "$tid_lt(10)" 或 "$lane_eq(0)"
    
    支援的運算子：
    - == -> eq
    - != -> ne
    - < -> lt
    - <= -> le
    - > -> gt
    - >= -> ge
    
    支援的變數：
    - [vs]\\d+ (暫存器，如 v6, s4)
    - $tid (local thread ID)
    - $lane (wavefront lane ID)
    - $tgid_x/$tgid_y/$tgid_z (workgroup ID)
    
    支援 AND：
    - if $tid == 0 && $tgid_x == 0:
    - if $tid == 0 and $tgid_x == 0:
    """
    cond_str = cond_str.strip()
    if cond_str.startswith('if '):
        cond_str = cond_str[3:]
    if cond_str.endswith(':'):
        cond_str = cond_str[:-1]
    cond_str = cond_str.strip()

    parts = [p.strip() for p in re.split(r'\s*(?:&&|\band\b)\s*', cond_str) if p.strip()]
    if not parts:
        return None
    if len(parts) > 1:
        parsed_parts = []
        for part in parts:
            parsed = _parse_condition_single(part, line_number)
            if not parsed:
                return None
            parsed_parts.append(parsed)
        return "&&".join(parsed_parts)

    return _parse_condition_single(parts[0], line_number)


def _parse_condition_single(cond_str: str, line_number: int = None) -> Optional[str]:
    op_map = {
        '==': 'eq',
        '!=': 'ne',
        '<=': 'le',
        '>=': 'ge',
        '<': 'lt',
        '>': 'gt'
    }

    for op_symbol, op_name in sorted(op_map.items(), key=lambda x: -len(x[0])):
        if op_symbol in cond_str:
            parts = cond_str.split(op_symbol, 1)
            if len(parts) == 2:
                reg = parts[0].strip()
                value = parts[1].strip()
                reg = normalize_builtin(reg)
                validate_variable(reg, line_number)
                return f"{reg}_{op_name}({value})"
    return None


def parse_print_directive(line: str, line_number: int) -> Optional[PrintDirective]:
    """
    解析 @PRINT 指令
    
    支援語法（Python f-string 風格）：
    ; @PRINT f"Before: A={v6:.3f}, B={v7:.2f}"
    ; @PRINT if v6 > 2.0: f"A={v6:.3f}, B={v7:.2f}"
    
    格式說明符：
    - {v6} - 預設格式（VGPR 預設 f32，SGPR 預設 i32）
    - {v6:.3f} - 浮點數，3 位小數
    - {v6:d} - 整數
    - {v6*v7:.2f} - 表達式
    
    條件式：
    - if v6 > 2.0:
    - if s4 == 64:
    - if $tid == 0 && $tgid_x == 0:
    - 支援 ==, !=, <, <=, >, >= 與 AND (&& / and)
    
    注意：僅支援 f-string 風格，舊語法不再解析。
    """
    # 匹配 @PRINT 指令（支援 ; 或 # 作為註解前綴）
    match = re.search(r'[;#]\s*@PRINT\s+(.+)', line)
    if not match:
        return None
    
    directive_content = match.group(1).strip()
    
    # ========== 嘗試新語法（Python f-string 風格）==========
    # 格式: [if CONDITION:] f"..."
    fstring_match = re.search(r'f"([^"]*)"', directive_content)
    
    if fstring_match:
        fstring_content = fstring_match.group(1)

        # 解析 trace 上限（例如 max=16）
        trace_max = None
        max_match = re.search(r'\bmax\s*=\s*(\d+)\b', directive_content)
        if max_match:
            trace_max = int(max_match.group(1))
            if trace_max <= 0:
                trace_max = None
            else:
                print(f"[Info] Trace max set to {trace_max} at line {line_number + 1}")
        
        # 檢查是否有條件式 (if ... :)
        condition = None
        condition_register = None
        condition_type = None
        
        # 尋找 "if ... :" 部分
        cond_match = re.match(r'(if\s+[^:]+:)\s*f"', directive_content)
        if cond_match:
            cond_str = cond_match.group(1)
            # 轉換為內部格式
            internal_cond = parse_condition_pythonic(cond_str, line_number)
            if internal_cond:
                condition = internal_cond
                print(f"[Info] Parsed condition: '{cond_str}' -> {condition}")

        # 如果啟用 trace 且沒有條件，預設只在單一 thread 打印，避免 hostcall buffer 爆量
        if trace_max and condition is None:
            condition = "$tgid_x_eq(0)&&$tgid_y_eq(0)&&$tgid_z_eq(0)&&$tid_eq(0)"
            print(f"[Info] Trace max without condition at line {line_number + 1}; defaulting to {condition}")
        
        # 解析 f-string
        printf_format, parsed_regs, parsed_types = parse_fstring_format(fstring_content, line_number)
        
        # 分離暫存器、表達式、和內建變數
        # 重要：維護各自的類型列表，確保順序正確
        # 同時保存原始順序 (all_placeholders) 用於 printf 參數對齊
        registers = []
        reg_types = []
        expressions = []
        expr_types = []
        all_placeholders = []  # 保存原始順序 [(value, type), ...]
        uses_tid = False
        uses_lane = False
        uses_tgid = False
        
        for reg_or_expr, typ in zip(parsed_regs, parsed_types):
            reg_or_expr = normalize_builtin(reg_or_expr)
            # 保存原始順序
            all_placeholders.append((reg_or_expr, typ))
            
            # 檢查是否是內建變數
            if reg_or_expr == '$tid':
                uses_tid = True
                expressions.append(reg_or_expr)
                expr_types.append(typ)
            elif reg_or_expr == '$lane':
                uses_lane = True
                expressions.append(reg_or_expr)
                expr_types.append(typ)
            elif reg_or_expr in ('$tgid_x', '$tgid_y', '$tgid_z'):
                uses_tgid = True
                expressions.append(reg_or_expr)
                expr_types.append(typ)
            # 判斷是暫存器還是表達式
            elif re.match(r'^[vsa]\d+$', reg_or_expr):
                registers.append(reg_or_expr)
                reg_types.append(typ)
            else:
                expressions.append(reg_or_expr)
                expr_types.append(typ)
        
        # 合併類型列表：先 registers 類型，後 expressions 類型
        types = reg_types + expr_types
        
        # 輸出內建變數使用資訊
        if uses_tid:
            print(f"[Info] Using $tid (local thread ID) at line {line_number + 1}")
        if uses_lane:
            print(f"[Info] Using $lane (wavefront lane ID) at line {line_number + 1}")
        if uses_tgid:
            print(f"[Info] Using $tgid_x/$tgid_y/$tgid_z (workgroup ID) at line {line_number + 1}")
        
        # 處理條件暫存器
        if condition:
            cond_parts = [p.strip() for p in condition.split('&&') if p.strip()]
            for cond_part in cond_parts:
                # 先檢查是否是內建變數條件
                builtin_cond_parse = re.match(r'(\$tid|\$lane|\$tgid_x|\$tgid_y|\$tgid_z)_(\w+)\((-?\d+(?:\.\d+)?)\)', cond_part)
                if builtin_cond_parse:
                    builtin_var, _, _ = builtin_cond_parse.groups()
                    if builtin_var == '$tid':
                        uses_tid = True
                        print(f"[Info] Condition uses $tid at line {line_number + 1}")
                    elif builtin_var == '$lane':
                        uses_lane = True
                        print(f"[Info] Condition uses $lane at line {line_number + 1}")
                    else:
                        uses_tgid = True
                        print(f"[Info] Condition uses {builtin_var} at line {line_number + 1}")
                    continue

                # 檢查是否是暫存器條件
                cond_parse = re.match(r'([vs]\d+)_(\w+)\((-?\d+(?:\.\d+)?)\)', cond_part)
                if cond_parse:
                    cond_reg, _, cond_value = cond_parse.groups()
                    if '.' in cond_value:
                        cond_type = 'f32'
                    else:
                        cond_type = 'i32'
                    if cond_reg not in registers:
                        if not condition_register:
                            condition_register = cond_reg
                            condition_type = cond_type
                            print(f"[Info] Condition register {cond_reg} (type={cond_type}) needs separate snapshot")
                        elif condition_register != cond_reg:
                            print(
                                f"[Warning] Multiple condition registers detected ({condition_register}, {cond_reg}); "
                                "only the first will be snapshotted."
                            )
        
        if not registers and not expressions:
            print(f"[Warning] @PRINT f-string has no placeholders at line {line_number + 1}")
            return None
        
        return PrintDirective(
            line_number=line_number,
            format_string=printf_format,
            registers=registers,
            types=types,
            condition=condition,
            condition_register=condition_register,
            condition_type=condition_type,
            expressions=expressions if expressions else None,
            uses_tid=uses_tid,
            uses_lane=uses_lane,
            uses_tgid=uses_tgid,
            all_placeholders=all_placeholders,
            trace_max=trace_max
        )
    
    # 僅支援 f-string，其他格式直接忽略
    print(f"[Warning] Invalid @PRINT format at line {line_number + 1}: {line.strip()}")
    print("         Use f-string format, e.g. ; @PRINT f\"v0={v0:d}\"")
    return None


def parse_timestamp_directive(line: str, line_number: int) -> Optional[TimestampDirective]:
    """
    解析 @TIMESTAMP_START 或 @TIMESTAMP_END 指令
    
    語法：
    ; @TIMESTAMP_START [label="section_name"] [mode=local|wallclock|atomic] [if $lane == 0:]
    ; @TIMESTAMP_END [label="section_name"] [mode=local|wallclock|atomic] [if $lane == 0:]
    
    簡單模式：
    ; @TIMESTAMP_START
    ; @TIMESTAMP_END
    
    Wallclock 模式（輸出絕對時間戳）：
    ; @TIMESTAMP_START mode=wallclock
    ; @TIMESTAMP_END mode=wallclock if $lane == 0:
    
    Atomic 模式（使用 global atomic 獲得真正的 kernel wall time）：
    ; @TIMESTAMP_START mode=atomic
    ; @TIMESTAMP_END mode=atomic
    """
    # 匹配 @TIMESTAMP_START 或 @TIMESTAMP_END
    start_match = re.search(r'[;#]\s*@TIMESTAMP_START\b(.*)$', line)
    end_match = re.search(r'[;#]\s*@TIMESTAMP_END\b(.*)$', line)
    
    if not start_match and not end_match:
        return None
    
    if start_match:
        directive_type = "start"
        args_str = start_match.group(1).strip()
    else:
        directive_type = "end"
        args_str = end_match.group(1).strip()
    
    # 解析可選的 label
    label = None
    label_match = re.search(r'label\s*=\s*"([^"]*)"', args_str)
    if label_match:
        label = label_match.group(1)
    
    # 解析可選的 mode
    mode = "local"  # 預設模式
    mode_match = re.search(r'mode\s*=\s*(\w+)', args_str)
    if mode_match:
        mode_value = mode_match.group(1).lower()
        if mode_value in ("local", "wallclock", "atomic"):
            mode = mode_value
        else:
            print(f"[Warning] Unknown timestamp mode '{mode_value}' at line {line_number + 1}, using 'local'")
    
    # 解析可選的條件式
    condition = None
    cond_match = re.search(r'(if\s+[^:]+:)', args_str)
    if cond_match:
        cond_str = cond_match.group(1)
        condition = parse_condition_pythonic(cond_str, line_number)
    
    return TimestampDirective(
        line_number=line_number,
        directive_type=directive_type,
        label=label,
        condition=condition,
        mode=mode
    )


def parse_asm_file(asm_path: pathlib.Path) -> Tuple[List[str], List[PrintDirective], bool, List[TimestampDirective]]:
    """
    解析 .s 檔案，提取 @PRINT 和 @TIMESTAMP 指令
    
    對於每個 directive，找到它之後的第一條非註解 ISA 指令，作為插入點的參考
    
    Returns:
        (lines, print_directives, has_barrier, timestamp_directives): 
        原始行列表、解析出的 @PRINT 指令、是否有 s_barrier、解析出的 @TIMESTAMP 指令
    """
    lines = asm_path.read_text().split('\n')
    print_directives = []
    timestamp_directives = []
    has_barrier = False
    
    def find_next_instruction(start_idx: int) -> Optional[str]:
        """找到指定位置之後的第一條 ISA 指令"""
        for j in range(start_idx + 1, min(start_idx + 10, len(lines))):
            next_line = lines[j].strip()
            # 跳過空行、註解、標籤
            if next_line and not next_line.startswith(';') and not next_line.startswith('#'):
                if not next_line.endswith(':'):  # 不是標籤
                    return next_line
        return None
    
    for i, line in enumerate(lines):
        # 檢測 s_barrier 指令
        if 's_barrier' in line and not line.strip().startswith(';'):
            has_barrier = True
        
        # 解析 @PRINT directive
        if '@PRINT' in line and '@TIMESTAMP' not in line:
            directive = parse_print_directive(line, i)
            if directive:
                directive.next_instruction = find_next_instruction(i)
                print_directives.append(directive)
                print(f"[Info] Found: {directive}")
                if directive.next_instruction:
                    instr_preview = directive.next_instruction[:50]
                    print(f"       After: {instr_preview}...")
        
        # 解析 @TIMESTAMP directive
        if '@TIMESTAMP_START' in line or '@TIMESTAMP_END' in line:
            ts_directive = parse_timestamp_directive(line, i)
            if ts_directive:
                ts_directive.next_instruction = find_next_instruction(i)
                timestamp_directives.append(ts_directive)
                print(f"[Info] Found: {ts_directive}")
                if ts_directive.next_instruction:
                    instr_preview = ts_directive.next_instruction[:50]
                    print(f"       After: {instr_preview}...")
    
    # 驗證 timestamp directive 配對
    starts = [d for d in timestamp_directives if d.directive_type == "start"]
    ends = [d for d in timestamp_directives if d.directive_type == "end"]
    
    if len(starts) != len(ends):
        print(f"[Warning] Mismatched @TIMESTAMP_START ({len(starts)}) and @TIMESTAMP_END ({len(ends)})")
    
    return lines, print_directives, has_barrier, timestamp_directives


# ============================================================
# 暫存器分析
# ============================================================

def analyze_registers(isa_code: str) -> Dict[str, int]:
    """
    分析 ISA 程式碼中的暫存器使用量
    
    Returns:
        {'vgpr': max_vgpr, 'sgpr': max_sgpr, 'agpr': max_agpr}
    """
    max_vgpr = 0
    max_sgpr = 0
    max_agpr = 0
    
    # 匹配暫存器模式
    vgpr_pattern = re.compile(r'\bv(\d+)\b')
    sgpr_pattern = re.compile(r'\bs(\d+)\b')
    agpr_pattern = re.compile(r'\ba(\d+)\b')
    vgpr_range_pattern = re.compile(r'\bv\[(\d+):(\d+)\]')
    sgpr_range_pattern = re.compile(r'\bs\[(\d+):(\d+)\]')
    agpr_range_pattern = re.compile(r'\ba\[(\d+):(\d+)\]')
    
    for line in isa_code.split('\n'):
        # 移除註解
        code_part = line.split(';')[0].split('#')[0].strip()
        if not code_part or code_part.endswith(':'):
            continue
        
        # 匹配暫存器範圍（如 v[0:3]）
        for match in vgpr_range_pattern.finditer(code_part):
            end = int(match.group(2))
            max_vgpr = max(max_vgpr, end + 1)
        
        for match in sgpr_range_pattern.finditer(code_part):
            end = int(match.group(2))
            max_sgpr = max(max_sgpr, end + 1)
        
        for match in agpr_range_pattern.finditer(code_part):
            end = int(match.group(2))
            max_agpr = max(max_agpr, end + 1)
        
        # 匹配單個暫存器（如 v6）
        for match in vgpr_pattern.finditer(code_part):
            num = int(match.group(1))
            max_vgpr = max(max_vgpr, num + 1)
        
        for match in sgpr_pattern.finditer(code_part):
            num = int(match.group(1))
            max_sgpr = max(max_sgpr, num + 1)
        
        for match in agpr_pattern.finditer(code_part):
            num = int(match.group(1))
            max_agpr = max(max_agpr, num + 1)
    
    return {'vgpr': max_vgpr, 'sgpr': max_sgpr, 'agpr': max_agpr}


# ============================================================
# GPU MLIR 操作
# ============================================================

def map_type_to_mlir(type_str: str) -> str:
    """將類型字串轉換為 MLIR 類型"""
    type_map = {
        'f32': 'f32',
        'f64': 'f64',
        'i32': 'i32',
        'i64': 'i64',
        'ptr': 'i64',
        'index': 'index'
    }
    return type_map.get(type_str, 'f32')


# ============================================================
# 表達式解析與編譯（用於 @PRINT expr="..."）
# ============================================================

def parse_expression(expr: str) -> List[Tuple[str, str]]:
    """
    解析簡單的算術表達式為 token 列表
    
    支援：
    - 暫存器：v0, v1, s0, s1, a0 等
    - 常數：1.0, 2.5, 4, -3.14 等
    - 運算符：+, -, *, /
    - 括號：( )
    
    Returns:
        List of (token_type, token_value) tuples
        token_type: 'REG', 'NUM', 'OP', 'LPAREN', 'RPAREN'
    """
    tokens = []
    i = 0
    expr = expr.strip()
    
    while i < len(expr):
        ch = expr[i]
        
        # 跳過空白
        if ch.isspace():
            i += 1
            continue
        
        # 暫存器：v0, v1, s0, s1, a0 等
        if ch in 'vsa' and i + 1 < len(expr) and expr[i + 1].isdigit():
            j = i + 1
            while j < len(expr) and expr[j].isdigit():
                j += 1
            tokens.append(('REG', expr[i:j]))
            i = j
            continue
        
        # 數字（包括小數和負數）
        if ch.isdigit() or (ch == '-' and i + 1 < len(expr) and expr[i + 1].isdigit()):
            j = i
            if ch == '-':
                j += 1
            while j < len(expr) and (expr[j].isdigit() or expr[j] == '.'):
                j += 1
            tokens.append(('NUM', expr[i:j]))
            i = j
            continue
        
        # 運算符
        if ch in '+-*/':
            tokens.append(('OP', ch))
            i += 1
            continue
        
        # 括號
        if ch == '(':
            tokens.append(('LPAREN', ch))
            i += 1
            continue
        if ch == ')':
            tokens.append(('RPAREN', ch))
            i += 1
            continue
        
        # 未知字符
        raise ValueError(f"Unknown character in expression: '{ch}' at position {i}")
    
    return tokens


def compile_expression_to_mlir(expr: str, result_type: str, unique_id: int, expr_idx: int,
                                snapshot_map: Dict[str, str] = None,
                                scratch_expr_offsets: Dict[str, int] = None) -> Tuple[str, str]:
    """
    將表達式編譯為 MLIR 代碼
    
    使用 arith dialect 進行計算。
    
    Args:
        expr: 表達式字串（如 "v6 * v7" 或 "v7*v7 - 4.0*v6*v2"）
        result_type: 結果類型（f32, i32 等）
        unique_id: 唯一識別符
        expr_idx: 表達式索引
        snapshot_map: 暫存器快照映射 {原始暫存器: 快照暫存器}，如 {"v6": "v60", "v7": "v61"}
        scratch_expr_offsets: scratch offset 映射 {原始暫存器: scratch_offset}（scratch mode）
    
    Returns:
        (mlir_code, result_var_name): MLIR 代碼和結果變數名
    """
    tokens = parse_expression(expr)
    mlir_type = map_type_to_mlir(result_type)
    lines = []
    
    # 追蹤已綁定的暫存器
    bound_regs = {}
    var_counter = [0]
    
    def get_operand(token_type: str, token_value: str) -> str:
        """獲取運算元的 MLIR 變數名"""
        if token_type == 'REG':
            if token_value not in bound_regs:
                var_name = f'expr_{unique_id}_{expr_idx}_reg_{var_counter[0]}'
                var_counter[0] += 1
                
                # Check scratch mode first
                if scratch_expr_offsets and token_value in scratch_expr_offsets:
                    s_off = scratch_expr_offsets[token_value]
                    lines.append(f'              // Using scratch [{s_off}] for {token_value}')
                    lines.append(_gen_scratch_load_mlir(s_off, mlir_type, var_name))
                    bound_regs[token_value] = var_name
                    return f'%{bound_regs[token_value]}'
                
                # 檢查是否有快照映射
                actual_reg = token_value
                if snapshot_map and token_value in snapshot_map:
                    actual_reg = snapshot_map[token_value]
                    lines.append(f'              // Using snapshot {actual_reg} for {token_value}')
                
                # 使用 value binding 讀取暫存器值
                reg_type = actual_reg[0]  # v, s, or a
                
                if reg_type == 'v':
                    lines.append(f'              %{var_name} = llvm.inline_asm has_side_effects asm_dialect = att "v_mov_b32 $0, {actual_reg}", "=v": () -> {mlir_type}')
                elif reg_type == 's':
                    lines.append(f'              %{var_name} = llvm.inline_asm has_side_effects asm_dialect = att "v_mov_b32 $0, {actual_reg}", "=v": () -> {mlir_type}')
                else:  # a
                    lines.append(f'              %{var_name} = llvm.inline_asm has_side_effects asm_dialect = att "v_accvgpr_read_b32 $0, {actual_reg}", "=v": () -> {mlir_type}')
                
                bound_regs[token_value] = var_name
            return f'%{bound_regs[token_value]}'
        elif token_type == 'NUM':
            var_name = f'expr_{unique_id}_{expr_idx}_const_{var_counter[0]}'
            var_counter[0] += 1
            # 確保浮點類型的常數使用浮點格式
            const_value = token_value
            if mlir_type in ('f32', 'f64') and '.' not in str(const_value):
                const_value = f'{const_value}.0'
            lines.append(f'              %{var_name} = arith.constant {const_value} : {mlir_type}')
            return f'%{var_name}'
        else:
            raise ValueError(f"Unexpected token type: {token_type}")
    
    # 簡單的表達式求值（遞歸下降解析器）
    # 支援優先級：* / 高於 + -
    
    pos = [0]  # 使用列表以便在嵌套函數中修改
    
    def parse_primary() -> str:
        """解析基本元素（數字、暫存器、括號表達式）"""
        if pos[0] >= len(tokens):
            raise ValueError("Unexpected end of expression")
        
        token_type, token_value = tokens[pos[0]]
        
        if token_type in ('REG', 'NUM'):
            pos[0] += 1
            return get_operand(token_type, token_value)
        elif token_type == 'LPAREN':
            pos[0] += 1
            result = parse_add_sub()
            if pos[0] >= len(tokens) or tokens[pos[0]][0] != 'RPAREN':
                raise ValueError("Missing closing parenthesis")
            pos[0] += 1
            return result
        else:
            raise ValueError(f"Unexpected token: {token_type} {token_value}")
    
    def parse_mul_div() -> str:
        """解析乘除表達式"""
        left = parse_primary()
        
        while pos[0] < len(tokens) and tokens[pos[0]][0] == 'OP' and tokens[pos[0]][1] in '*/':
            op = tokens[pos[0]][1]
            pos[0] += 1
            right = parse_primary()
            
            result_var = f'expr_{unique_id}_{expr_idx}_tmp_{var_counter[0]}'
            var_counter[0] += 1
            
            if result_type.startswith('f'):
                op_name = 'mulf' if op == '*' else 'divf'
            else:
                op_name = 'muli' if op == '*' else 'divi_signed'
            
            lines.append(f'              %{result_var} = arith.{op_name} {left}, {right} : {mlir_type}')
            left = f'%{result_var}'
        
        return left
    
    def parse_add_sub() -> str:
        """解析加減表達式"""
        left = parse_mul_div()
        
        while pos[0] < len(tokens) and tokens[pos[0]][0] == 'OP' and tokens[pos[0]][1] in '+-':
            op = tokens[pos[0]][1]
            pos[0] += 1
            right = parse_mul_div()
            
            result_var = f'expr_{unique_id}_{expr_idx}_tmp_{var_counter[0]}'
            var_counter[0] += 1
            
            if result_type.startswith('f'):
                op_name = 'addf' if op == '+' else 'subf'
            else:
                op_name = 'addi' if op == '+' else 'subi'
            
            lines.append(f'              %{result_var} = arith.{op_name} {left}, {right} : {mlir_type}')
            left = f'%{result_var}'
        
        return left
    
    final_result = parse_add_sub()
    
    # 確保結果變數名是純淨的（不帶 %）
    if final_result.startswith('%'):
        final_var = final_result[1:]
    else:
        final_var = final_result
    
    return '\n'.join(lines), final_var


def generate_value_binding(reg: str, type_str: str, var_name: str) -> str:
    """
    生成讀取暫存器值的 MLIR 程式碼
    
    Args:
        reg: 暫存器名稱（如 "v6"）
        type_str: 類型字串（如 "f32"）
        var_name: 生成的 SSA 變數名稱
    
    Returns:
        MLIR 程式碼行
    """
    mlir_type = map_type_to_mlir(type_str)
    
    # 處理不同的暫存器格式
    if reg.startswith('a') and not reg.startswith('acc'):
        # AccVGPR (a0, a1, ...) — read via v_accvgpr_read_b32
        asm_instr = f"v_accvgpr_read_b32 $0, {reg}"
    elif reg.startswith('v'):
        # VGPR
        asm_instr = f"v_mov_b32 $0, {reg}"
    elif reg.startswith('s'):
        # SGPR - 需要先 mov 到 VGPR
        asm_instr = f"v_mov_b32 $0, {reg}"
    else:
        asm_instr = f"v_mov_b32 $0, {reg}"
    
    return f'              %{var_name} = llvm.inline_asm has_side_effects asm_dialect = att "{asm_instr}", "=v": () -> {mlir_type}'


def _gen_scratch_load_mlir(scratch_offset: int, mlir_type: str, var_name: str) -> str:
    """Generate MLIR inline_asm to load a value from scratch memory."""
    return (
        f'              %{var_name} = llvm.inline_asm has_side_effects asm_dialect = att '
        f'"scratch_load_dword $0, off, off offset:{scratch_offset}\\0A'
        f's_waitcnt vmcnt(0)", "=v": () -> {mlir_type}'
    )


def generate_printf_with_snapshot(directive: PrintDirective, unique_id: int, 
                                   vgpr_snapshots: List[Tuple[int, str, int]],
                                   sgpr_snapshots: List[Tuple[int, str, int]] = None,
                                   workitem_id_backup_vgpr: int = None,
                                   cond_snapshot: Tuple[str, int, bool, str] = None,
                                   expr_snapshot: Dict[str, str] = None,
                                   builtin_vars: Dict[str, int] = None,
                                   scratch_snapshot_map: Dict = None) -> str:
    """
    生成使用快照暫存器的 gpu.printf MLIR 程式碼
    
    Args:
        directive: @PRINT 指令
        unique_id: 唯一識別碼
        vgpr_snapshots: [(reg_idx, original_reg, snapshot_vgpr), ...] VGPR 快照暫存器映射
        sgpr_snapshots: [(reg_idx, original_reg, snapshot_sgpr), ...] SGPR 快照暫存器映射
        workitem_id_backup_vgpr: workitem_id 備份暫存器（未使用，保留以相容）
        cond_snapshot: 條件暫存器快照 (cond_reg, snapshot_num, is_vgpr, cond_type)
        expr_snapshot: 表達式暫存器快照映射 {原始暫存器: 快照暫存器}
        builtin_vars: 內建變數備份暫存器 {'tid_vgpr': int, 'lane_vgpr': int}
    
    Returns:
        MLIR 程式碼字串
    """
    lines = []
    
    # 生成註解標記
    lines.append(f'              // === @PRINT from line {directive.line_number + 1} (using snapshots) ===')
    
    var_names = []
    
    # 建立暫存器名稱 -> (snapshot_reg, snapshot_type) 的映射
    reg_snapshot_map = {}
    for r_idx, orig_reg, snap_vgpr in vgpr_snapshots:
        reg_snapshot_map[orig_reg] = (f'v{snap_vgpr}', 'v')
    if sgpr_snapshots:
        for r_idx, orig_reg, snap_sgpr in sgpr_snapshots:
            reg_snapshot_map[orig_reg] = (f's{snap_sgpr}', 's')
    
    # 建立完整的表達式快照映射（用於表達式計算）
    full_expr_snapshot = {}
    if expr_snapshot:
        full_expr_snapshot.update(expr_snapshot)
    for orig_reg, (snap_reg, _) in reg_snapshot_map.items():
        full_expr_snapshot[orig_reg] = snap_reg
    
    # Build scratch expression offset map for compile_expression_to_mlir
    scratch_expr_offsets = None
    if scratch_snapshot_map:
        scratch_expr_offsets = {}
        # Merge all scratch reg offsets for expression use
        if 'reg_snapshots' in scratch_snapshot_map:
            scratch_expr_offsets.update(scratch_snapshot_map['reg_snapshots'])
        if 'expr' in scratch_snapshot_map:
            scratch_expr_offsets.update(scratch_snapshot_map['expr'])
    
    # --- Solution A: When in scratch snapshot mode, defer print value generation ---
    # This ensures LLVM keeps the scratch_load results in registers only briefly
    # (right before gpu.printf) instead of across 600+ hostcall setup instructions.
    _use_deferred = scratch_snapshot_map is not None
    _deferred_print_val_lines: List[str] = []  # collected, inserted inside scf.if later
    
    # === 按照原始順序生成所有值（使用 all_placeholders）===
    # 這確保 printf 參數順序與格式字串中的 placeholder 順序一致
    if directive.all_placeholders:
        # When in scratch deferred mode, print value lines go to _deferred list
        _pv_target = _deferred_print_val_lines if _use_deferred else lines
        for i, (value, typ) in enumerate(directive.all_placeholders):
            var_name = f'print_val_{unique_id}_{i}'
            mlir_type = map_type_to_mlir(typ)
            
            # 檢查是否是內建變數
            if value == '$tid':
                # $tid: Local Thread ID (workitem_id_x)
                if scratch_snapshot_map and scratch_snapshot_map.get('tid') is not None:
                    tid_off = scratch_snapshot_map['tid']
                    _pv_target.append(f'              // Built-in $tid: read from scratch[{tid_off}]')
                    _pv_target.append(_gen_scratch_load_mlir(tid_off, 'i32', var_name))
                elif builtin_vars and builtin_vars.get('tid_vgpr') is not None:
                    tid_vgpr = builtin_vars['tid_vgpr']
                    _pv_target.append(f'              // Built-in $tid: read from v{tid_vgpr}')
                    _pv_target.append(f'              %{var_name} = llvm.inline_asm has_side_effects asm_dialect = att "v_mov_b32 $0, v{tid_vgpr}", "=v": () -> i32')
                else:
                    _pv_target.append(f'              // Built-in $tid: WARNING - reading v0 directly (may be modified)')
                    _pv_target.append(f'              %{var_name} = llvm.inline_asm has_side_effects asm_dialect = att "v_mov_b32 $0, v0", "=v": () -> i32')
                var_names.append(var_name)
            
            elif value == '$lane':
                # $lane: Wavefront Lane ID (0-63)
                if scratch_snapshot_map and scratch_snapshot_map.get('lane') is not None:
                    lane_off = scratch_snapshot_map['lane']
                    _pv_target.append(f'              // Built-in $lane: read from scratch[{lane_off}]')
                    _pv_target.append(_gen_scratch_load_mlir(lane_off, 'i32', var_name))
                elif builtin_vars and builtin_vars.get('lane_vgpr') is not None:
                    lane_vgpr = builtin_vars['lane_vgpr']
                    _pv_target.append(f'              // Built-in $lane: read from v{lane_vgpr}')
                    _pv_target.append(f'              %{var_name} = llvm.inline_asm has_side_effects asm_dialect = att "v_mov_b32 $0, v{lane_vgpr}", "=v": () -> i32')
                else:
                    _pv_target.append(f'              // Built-in $lane: calculating lane ID on the fly')
                    _pv_target.append(f'              %{var_name}_lo = llvm.inline_asm has_side_effects asm_dialect = att "v_mbcnt_lo_u32_b32 $0, -1, 0", "=v": () -> i32')
                    _pv_target.append(f'              %{var_name} = llvm.inline_asm has_side_effects asm_dialect = att "v_mbcnt_hi_u32_b32 $0, -1, $1", "=v,v": (%{var_name}_lo) -> i32')
                var_names.append(var_name)
            
            elif value in ('$tgid_x', '$tgid_y', '$tgid_z'):
                # $tgid_x/y/z: Workgroup ID (block index)
                scratch_tgid_key = value.replace('$', '')  # 'tgid_x', 'tgid_y', 'tgid_z'
                if scratch_snapshot_map and scratch_snapshot_map.get(scratch_tgid_key) is not None:
                    tgid_off = scratch_snapshot_map[scratch_tgid_key]
                    _pv_target.append(f'              // Built-in {value}: read from scratch[{tgid_off}]')
                    _pv_target.append(_gen_scratch_load_mlir(tgid_off, 'i32', var_name))
                else:
                    tgid_vgpr_map = {
                        '$tgid_x': 'tgid_x_vgpr',
                        '$tgid_y': 'tgid_y_vgpr',
                        '$tgid_z': 'tgid_z_vgpr',
                    }
                    tgid_sgpr_map = {
                        '$tgid_x': 'tgid_x_sgpr',
                        '$tgid_y': 'tgid_y_sgpr',
                        '$tgid_z': 'tgid_z_sgpr',
                    }
                    tgid_vgpr = None
                    sgpr_idx = None
                    if builtin_vars:
                        tgid_vgpr = builtin_vars.get(tgid_vgpr_map[value])
                        sgpr_idx = builtin_vars.get(tgid_sgpr_map[value])
                    if tgid_vgpr is not None:
                        _pv_target.append(f'              // Built-in {value}: read from v{tgid_vgpr}')
                        _pv_target.append(
                            f'              %{var_name} = llvm.inline_asm has_side_effects asm_dialect = att '
                            f'"v_mov_b32 $0, v{tgid_vgpr}", "=v": () -> i32'
                        )
                    else:
                        if sgpr_idx is None:
                            sgpr_idx = 2 if value == '$tgid_x' else 3 if value == '$tgid_y' else 4
                        _pv_target.append(f'              // Built-in {value}: read from s{sgpr_idx}')
                        _pv_target.append(
                            f'              %{var_name} = llvm.inline_asm has_side_effects asm_dialect = att '
                            f'"v_mov_b32 $0, s{sgpr_idx}", "=v": () -> i32'
                        )
                var_names.append(var_name)
            
            elif re.match(r'^[vsa]\d+$', value):
                # 簡單暫存器 (v=VGPR, s=SGPR, a=AccVGPR)
                if scratch_snapshot_map and value in scratch_snapshot_map.get('reg_snapshots', {}):
                    s_off = scratch_snapshot_map['reg_snapshots'][value]
                    _pv_target.append(f'              // Using scratch snapshot [{s_off}] for {value}')
                    _pv_target.append(_gen_scratch_load_mlir(s_off, mlir_type, var_name))
                elif value in reg_snapshot_map:
                    snap_reg, _ = reg_snapshot_map[value]
                    _pv_target.append(f'              // Using snapshot {snap_reg} for {value}')
                    _pv_target.append(generate_value_binding(snap_reg, typ, var_name))
                else:
                    _pv_target.append(generate_value_binding(value, typ, var_name))
                var_names.append(var_name)
            
            else:
                # 表達式
                try:
                    expr_code, result_var = compile_expression_to_mlir(value, typ, unique_id, i, full_expr_snapshot, scratch_expr_offsets)
                    if expr_code:
                        _pv_target.append(f'              // Expression: {value} (using snapshots)')
                        _pv_target.append(expr_code)
                    var_names.append(result_var)
                except Exception as e:
                    print(f"[Warning] Failed to compile expression '{value}': {e}")
                    if typ.startswith('f'):
                        _pv_target.append(f'              %{var_name} = arith.constant 0.0 : {mlir_type}')
                    else:
                        _pv_target.append(f'              %{var_name} = arith.constant 0 : {mlir_type}')
                    var_names.append(var_name)
    else:
        # Fallback: 使用舊的處理方式（向後兼容）
        reg_types = directive.types[:len(directive.registers)]
        expr_types = directive.types[len(directive.registers):]
        
        for i, (reg, typ) in enumerate(zip(directive.registers, reg_types)):
            var_name = f'print_val_{unique_id}_{i}'
            if reg in reg_snapshot_map:
                snap_reg, _ = reg_snapshot_map[reg]
                lines.append(f'              // Using snapshot {snap_reg} for {reg}')
                lines.append(generate_value_binding(snap_reg, typ, var_name))
            else:
                lines.append(generate_value_binding(reg, typ, var_name))
            var_names.append(var_name)
        
        if directive.expressions:
            for i, (expr, typ) in enumerate(zip(directive.expressions, expr_types)):
                var_name = f'expr_val_{unique_id}_{i}'
                if expr == '$tid':
                    if builtin_vars and builtin_vars.get('tid_vgpr') is not None:
                        tid_vgpr = builtin_vars['tid_vgpr']
                        lines.append(f'              %{var_name} = llvm.inline_asm has_side_effects asm_dialect = att "v_mov_b32 $0, v{tid_vgpr}", "=v": () -> i32')
                    else:
                        lines.append(f'              %{var_name} = llvm.inline_asm has_side_effects asm_dialect = att "v_mov_b32 $0, v0", "=v": () -> i32')
                    var_names.append(var_name)
                elif expr == '$lane':
                    if builtin_vars and builtin_vars.get('lane_vgpr') is not None:
                        lane_vgpr = builtin_vars['lane_vgpr']
                        lines.append(f'              %{var_name} = llvm.inline_asm has_side_effects asm_dialect = att "v_mov_b32 $0, v{lane_vgpr}", "=v": () -> i32')
                    else:
                        lines.append(f'              %{var_name}_lo = llvm.inline_asm has_side_effects asm_dialect = att "v_mbcnt_lo_u32_b32 $0, -1, 0", "=v": () -> i32')
                        lines.append(f'              %{var_name} = llvm.inline_asm has_side_effects asm_dialect = att "v_mbcnt_hi_u32_b32 $0, -1, $1", "=v,v": (%{var_name}_lo) -> i32')
                    var_names.append(var_name)
                elif expr in ('$tgid_x', '$tgid_y', '$tgid_z'):
                    tgid_vgpr_map = {
                        '$tgid_x': 'tgid_x_vgpr',
                        '$tgid_y': 'tgid_y_vgpr',
                        '$tgid_z': 'tgid_z_vgpr',
                    }
                    tgid_sgpr_map = {
                        '$tgid_x': 'tgid_x_sgpr',
                        '$tgid_y': 'tgid_y_sgpr',
                        '$tgid_z': 'tgid_z_sgpr',
                    }
                    tgid_vgpr = None
                    sgpr_idx = None
                    if builtin_vars:
                        tgid_vgpr = builtin_vars.get(tgid_vgpr_map[expr])
                        sgpr_idx = builtin_vars.get(tgid_sgpr_map[expr])
                    if tgid_vgpr is not None:
                        lines.append(
                            f'              %{var_name} = llvm.inline_asm has_side_effects asm_dialect = att '
                            f'"v_mov_b32 $0, v{tgid_vgpr}", "=v": () -> i32'
                        )
                    else:
                        if sgpr_idx is None:
                            sgpr_idx = 2 if expr == '$tgid_x' else 3 if expr == '$tgid_y' else 4
                        lines.append(
                            f'              %{var_name} = llvm.inline_asm has_side_effects asm_dialect = att '
                            f'"v_mov_b32 $0, s{sgpr_idx}", "=v": () -> i32'
                        )
                    var_names.append(var_name)
                else:
                    try:
                        expr_code, result_var = compile_expression_to_mlir(expr, typ, unique_id, i, full_expr_snapshot, scratch_expr_offsets)
                        if expr_code:
                            lines.append(expr_code)
                        var_names.append(result_var)
                    except Exception as e:
                        mlir_type = map_type_to_mlir(typ)
                        if typ.startswith('f'):
                            lines.append(f'              %{var_name} = arith.constant 0.0 : {mlir_type}')
                        else:
                            lines.append(f'              %{var_name} = arith.constant 0 : {mlir_type}')
                        var_names.append(var_name)
    
    # 準備 printf 的參數列表
    args = ', '.join(f'%{name}' for name in var_names)
    # 使用 all_placeholders 中的類型順序（如果有的話）
    if directive.all_placeholders:
        types = ', '.join(map_type_to_mlir(t) for _, t in directive.all_placeholders)
    else:
        types = ', '.join(map_type_to_mlir(t) for t in directive.types)
    
    # MLIR string 中的換行需使用 \0A
    escaped_format = directive.format_string.replace('\\n', '\\0A')
    if not escaped_format.endswith('\\0A'):
        escaped_format += '\\0A'
    
    # 條件式 printf 支援
    # 
    # 支援基於暫存器值的條件：REG_XX(N)
    # 例如：v0_eq(1), s4_gt(0), v6_lt(10.0)
    # 
    # 注意：條件暫存器必須在 reg= 中列出，才能使用快照值
    
    if directive.condition:
        cond_parts = [p.strip() for p in directive.condition.split('&&') if p.strip()]
        cond_vars: List[str] = []
        cond_supported = True

        for cond_idx, cond_part in enumerate(cond_parts):
            cond_tag = f"{unique_id}_{cond_idx}"
            builtin_cond_match = re.match(
                r'(\$tid|\$lane|\$tgid_x|\$tgid_y|\$tgid_z)_(\w+)\((-?\d+(?:\.\d+)?)\)',
                cond_part,
            )
            if builtin_cond_match:
                builtin_var, cmp_op, cmp_value = builtin_cond_match.groups()
                snapshot_var = f'%cond_builtin_{cond_tag}'
                scratch_cond_key = builtin_var.replace('$', '')  # 'tid', 'lane', 'tgid_x', etc.
                
                # Try scratch mode first
                if scratch_snapshot_map and scratch_snapshot_map.get(scratch_cond_key) is not None:
                    s_off = scratch_snapshot_map[scratch_cond_key]
                    lines.append(f'              // Conditional printf: read {builtin_var} from scratch[{s_off}]')
                    lines.append(_gen_scratch_load_mlir(s_off, 'i32', f'cond_builtin_{cond_tag}'))
                elif builtin_var == '$tid' and builtin_vars and builtin_vars.get('tid_vgpr') is not None:
                    tid_vgpr = builtin_vars['tid_vgpr']
                    lines.append(f'              // Conditional printf: read $tid from v{tid_vgpr}')
                    lines.append(f'              {snapshot_var} = llvm.inline_asm has_side_effects asm_dialect = att "v_mov_b32 $0, v{tid_vgpr}", "=v": () -> i32')
                elif builtin_var == '$lane' and builtin_vars and builtin_vars.get('lane_vgpr') is not None:
                    lane_vgpr = builtin_vars['lane_vgpr']
                    lines.append(f'              // Conditional printf: read $lane from v{lane_vgpr}')
                    lines.append(f'              {snapshot_var} = llvm.inline_asm has_side_effects asm_dialect = att "v_mov_b32 $0, v{lane_vgpr}", "=v": () -> i32')
                elif builtin_var in ('$tgid_x', '$tgid_y', '$tgid_z'):
                    key_map_vgpr = {
                        '$tgid_x': 'tgid_x_vgpr',
                        '$tgid_y': 'tgid_y_vgpr',
                        '$tgid_z': 'tgid_z_vgpr',
                    }
                    key_map_sgpr = {
                        '$tgid_x': 'tgid_x_sgpr',
                        '$tgid_y': 'tgid_y_sgpr',
                        '$tgid_z': 'tgid_z_sgpr',
                    }
                    tgid_vgpr = None
                    sgpr_idx = None
                    if builtin_vars:
                        tgid_vgpr = builtin_vars.get(key_map_vgpr[builtin_var])
                        sgpr_idx = builtin_vars.get(key_map_sgpr[builtin_var])
                    if tgid_vgpr is not None:
                        lines.append(f'              // Conditional printf: read {builtin_var} from v{tgid_vgpr}')
                        lines.append(
                            f'              {snapshot_var} = llvm.inline_asm has_side_effects asm_dialect = att '
                            f'"v_mov_b32 $0, v{tgid_vgpr}", "=v": () -> i32'
                        )
                    else:
                        if sgpr_idx is None:
                            sgpr_idx = 2 if builtin_var == '$tgid_x' else 3 if builtin_var == '$tgid_y' else 4
                        lines.append(f'              // Conditional printf: read {builtin_var} from s{sgpr_idx}')
                        lines.append(
                            f'              {snapshot_var} = llvm.inline_asm has_side_effects asm_dialect = att '
                            f'"v_mov_b32 $0, s{sgpr_idx}", "=v": () -> i32'
                        )
                else:
                    if builtin_var == '$tid':
                        lines.append(f'              // Conditional printf: read $tid from v0 (WARNING: may be modified)')
                        lines.append(f'              {snapshot_var} = llvm.inline_asm has_side_effects asm_dialect = att "v_mov_b32 $0, v0", "=v": () -> i32')
                    else:
                        lines.append(f'              // Conditional printf: calculate $lane on the fly')
                        lines.append(f'              %cond_lane_lo_{cond_tag} = llvm.inline_asm has_side_effects asm_dialect = att "v_mbcnt_lo_u32_b32 $0, -1, 0", "=v": () -> i32')
                        lines.append(f'              {snapshot_var} = llvm.inline_asm has_side_effects asm_dialect = att "v_mbcnt_hi_u32_b32 $0, -1, $1", "=v,v": (%cond_lane_lo_{cond_tag}) -> i32')

                cmp_ops = {'eq': 'eq', 'ne': 'ne', 'lt': 'slt', 'le': 'sle', 'gt': 'sgt', 'ge': 'sge'}
                mlir_cmp = cmp_ops.get(cmp_op, 'eq')
                cmp_value_int = int(float(cmp_value))
                cmp_const = f'%cmp_val_{cond_tag}'
                cond_var = f'%cond_{cond_tag}'
                lines.append(f'              {cmp_const} = arith.constant {cmp_value_int} : i32')
                lines.append(f'              {cond_var} = arith.cmpi {mlir_cmp}, {snapshot_var}, {cmp_const} : i32')
                cond_vars.append(cond_var)
                print(f"[Info] Conditional printf (builtin): {builtin_var} {cmp_op} {cmp_value}")
                continue

            reg_cond_match = re.match(r'([vs]\d+)_(\w+)\((-?\d+(?:\.\d+)?)\)', cond_part)
            if reg_cond_match:
                cond_reg, cmp_op, cmp_value = reg_cond_match.groups()
                snapshot_var = None
                snapshot_type = None

                if vgpr_snapshots:
                    for r_idx, orig_reg, _ in vgpr_snapshots:
                        if orig_reg == cond_reg:
                            snapshot_var = f'%print_val_{unique_id}_{r_idx}'
                            if r_idx < len(directive.types):
                                snapshot_type = directive.types[r_idx]
                            break

                if not snapshot_var and sgpr_snapshots:
                    for r_idx, orig_reg, _ in sgpr_snapshots:
                        if orig_reg == cond_reg:
                            snapshot_var = f'%print_val_{unique_id}_{r_idx}'
                            if r_idx < len(directive.types):
                                snapshot_type = directive.types[r_idx]
                            break

                # Try scratch snapshot for condition register
                if not snapshot_var and scratch_snapshot_map and 'cond' in scratch_snapshot_map:
                    cond_scratch_offsets = scratch_snapshot_map['cond']
                    if cond_reg in cond_scratch_offsets:
                        s_off = cond_scratch_offsets[cond_reg]
                        if cond_snapshot:
                            _, _, _, cond_type = cond_snapshot
                            snapshot_type = cond_type
                        else:
                            snapshot_type = 'i32'
                        mlir_type = map_type_to_mlir(snapshot_type)
                        snapshot_var = f'%cond_snap_val_{cond_tag}'
                        lines.append(f'              // Using scratch condition snapshot: {cond_reg} -> scratch[{s_off}]')
                        lines.append(_gen_scratch_load_mlir(s_off, mlir_type, f'cond_snap_val_{cond_tag}'))

                if not snapshot_var and cond_snapshot:
                    cond_snap_reg, snap_num, is_vgpr, cond_type = cond_snapshot
                    if cond_snap_reg == cond_reg:
                        snapshot_type = cond_type
                        snapshot_reg = f'v{snap_num}' if is_vgpr else f's{snap_num}'
                        mlir_type = map_type_to_mlir(cond_type)
                        snapshot_var = f'%cond_snap_val_{cond_tag}'
                        lines.append(f'              // Using condition register snapshot: {cond_reg} -> {snapshot_reg}')
                        lines.append(f'              {snapshot_var} = llvm.inline_asm has_side_effects asm_dialect = att "v_mov_b32 $0, {snapshot_reg}", "=v": () -> {mlir_type}')

                cond_var = f'%cond_{cond_tag}'
                cmp_const = f'%cmp_val_{cond_tag}'
                if snapshot_var and snapshot_type:
                    cmp_ops = {
                        'eq': 'eq', 'ne': 'ne',
                        'lt': 'slt', 'le': 'sle',
                        'gt': 'sgt', 'ge': 'sge'
                    }
                    mlir_cmp = cmp_ops.get(cmp_op, 'eq')
                    if snapshot_type in ['f32', 'f64']:
                        float_cmp_ops = {
                            'eq': 'oeq', 'ne': 'one',
                            'lt': 'olt', 'le': 'ole',
                            'gt': 'ogt', 'ge': 'oge'
                        }
                        mlir_cmp = float_cmp_ops.get(cmp_op, 'oeq')
                        mlir_type = 'f32' if snapshot_type == 'f32' else 'f64'
                        lines.append(f'              // Conditional printf: {cond_reg} {cmp_op} {cmp_value}')
                        lines.append(f'              {cmp_const} = arith.constant {cmp_value} : {mlir_type}')
                        lines.append(f'              {cond_var} = arith.cmpf {mlir_cmp}, {snapshot_var}, {cmp_const} : {mlir_type}')
                    else:
                        mlir_type = 'i32' if snapshot_type == 'i32' else 'i64'
                        cmp_value_int = int(float(cmp_value))
                        lines.append(f'              // Conditional printf: {cond_reg} {cmp_op} {cmp_value_int}')
                        lines.append(f'              {cmp_const} = arith.constant {cmp_value_int} : {mlir_type}')
                        lines.append(f'              {cond_var} = arith.cmpi {mlir_cmp}, {snapshot_var}, {cmp_const} : {mlir_type}')
                    print(f"[Info] Conditional printf (snapshot): {cond_reg} {cmp_op} {cmp_value}")
                else:
                    print(f"[Warning] Register {cond_reg} not in snapshot, using direct read (value at printf time, not @PRINT time)")
                    lines.append(f'              // Conditional printf: read {cond_reg} for condition (direct read)')
                    lines.append(f'              %cond_reg_val_{cond_tag} = llvm.inline_asm has_side_effects asm_dialect = att "v_mov_b32 $0, {cond_reg}", "=v": () -> i32')
                    cmp_ops = {'eq': 'eq', 'ne': 'ne', 'lt': 'slt', 'le': 'sle', 'gt': 'sgt', 'ge': 'sge'}
                    mlir_cmp = cmp_ops.get(cmp_op, 'eq')
                    cmp_value_int = int(float(cmp_value))
                    lines.append(f'              {cmp_const} = arith.constant {cmp_value_int} : i32')
                    lines.append(f'              {cond_var} = arith.cmpi {mlir_cmp}, %cond_reg_val_{cond_tag}, {cmp_const} : i32')
                    print(f"[Info] Conditional printf (direct read): {cond_reg} {cmp_op} {cmp_value}")

                cond_vars.append(cond_var)
                continue

            cond_supported = False
            break

        if cond_supported and cond_vars:
            cond_final = cond_vars[0]
            for idx in range(1, len(cond_vars)):
                merged = f'%cond_and_{unique_id}_{idx}'
                lines.append(f'              {merged} = arith.andi {cond_final}, {cond_vars[idx]} : i1')
                cond_final = merged
            lines.append(f'              scf.if {cond_final} {{')
            # Solution A: insert deferred print value loads INSIDE scf.if
            if _deferred_print_val_lines:
                lines.append(f'                // --- deferred scratch loads (inside scf.if to reduce reg pressure) ---')
                lines.extend(_deferred_print_val_lines)
            lines.append(f'                gpu.printf "{escaped_format}", {args} : {types}')
            lines.append(f'              }}')
        else:
            print(f"[Warning] Unknown condition format: {directive.condition}, printing unconditionally")
            if _deferred_print_val_lines:
                lines.extend(_deferred_print_val_lines)
            lines.append(f'              gpu.printf "{escaped_format}", {args} : {types}')
    else:
        if _deferred_print_val_lines:
            lines.extend(_deferred_print_val_lines)
        lines.append(f'              gpu.printf "{escaped_format}", {args} : {types}')
    
    lines.append(f'              // === End @PRINT ===')
    
    return '\n'.join(lines)


def inject_printf_into_mlir(gpumlir_text: str, directives: List[PrintDirective], reg_info: Dict[str, int],
                            timestamp_directives: List[TimestampDirective] = None,
                            print_mode: str = "defer") -> str:
    """
    將 printf 指令注入到 GPU MLIR 中

    策略（快照版）：
    1. 在每個 @PRINT 的實際位置插入快照指令（v_mov_b32），保存當時的暫存器值
    2. 在每個 @TIMESTAMP_START 位置插入 s_memtime 並備份到 VGPR
    3. 在每個 @TIMESTAMP_END 位置插入 s_memtime 並計算差值
    4. 在 .LBB0_2: 標籤之前或 s_endpgm 之前（defer 模式）：
       - Restore kernarg pointer
       - 使用快照暫存器執行 printf
       - 輸出 timestamp 結果
       - Register clobbering 結束
    5. 這樣可以真正觀察到「當時」的暫存器值和時間

    Args:
        gpumlir_text: 原始 GPU MLIR 文字
        directives: @PRINT 指令列表
        reg_info: 暫存器使用量資訊
        timestamp_directives: @TIMESTAMP 指令列表

    Returns:
        修改後的 GPU MLIR 文字
    """
    if timestamp_directives is None:
        timestamp_directives = []
    
    if not directives and not timestamp_directives:
        return gpumlir_text

    inline_mode = (print_mode == "inline")
    
    # === 計算快照暫存器需求 ===
    # 每個 @PRINT 的每個 VGPR/SGPR 都需要一個快照暫存器
    original_vgpr = reg_info.get('vgpr', 0)
    original_sgpr = reg_info.get('sgpr', 0)
    
    # VGPR 快照
    # 注意：printf/hostcall 代碼會使用大量 VGPR
    # 為了避免被覆蓋，快照暫存器必須從足夠高的編號開始
    # 
    # 動態計算 printf/hostcall 使用的 VGPR 開銷：
    # - 基礎開銷：hostcall 機制約需 16 個 VGPR
    # - 每個 printf 值：約需 4 個 VGPR（參數處理、格式化等）
    # - 安全邊界：+8 個 VGPR
    # - 最小值：24（即使沒有 printf 也預留空間）
    PRINTF_BASE_OVERHEAD = 16      # hostcall 基礎開銷
    PRINTF_PER_VALUE_OVERHEAD = 4  # 每個 printf 值的開銷
    PRINTF_SAFETY_MARGIN = 8       # 安全邊界
    PRINTF_MIN_OVERHEAD = 24       # 最小開銷
    
    # 計算所有 @PRINT 指令中的值數量（暫存器 + 表達式）
    total_printf_values = 0
    for d in directives:
        total_printf_values += len(d.registers)
        if d.expressions:
            total_printf_values += len(d.expressions)
    
    # 動態計算 VGPR 開銷
    printf_vgpr_overhead = PRINTF_BASE_OVERHEAD + PRINTF_PER_VALUE_OVERHEAD * total_printf_values + PRINTF_SAFETY_MARGIN
    printf_vgpr_overhead = max(printf_vgpr_overhead, PRINTF_MIN_OVERHEAD)
    
    print(f"[Info] Printf VGPR overhead: {printf_vgpr_overhead} (base={PRINTF_BASE_OVERHEAD} + {total_printf_values} values * {PRINTF_PER_VALUE_OVERHEAD} + margin={PRINTF_SAFETY_MARGIN})")
    
    snapshot_vgprs = []  # [(directive_idx, reg_idx, reg_name, snapshot_vgpr_num), ...]
    snapshot_vgpr_start = max(original_vgpr, printf_vgpr_overhead)  # 確保在 printf 使用範圍之後
    vgpr_snapshot_idx = 0
    
    # SGPR 快照 - 需要避開系統保留的和 kernarg backup 使用的暫存器
    # s[0:3] 系統保留, s[18:19] 用於 kernarg backup
    # SGPR 快照從 max(original_sgpr, 20) 開始，確保不與 kernarg backup 衝突
    SGPR_SNAPSHOT_START_MIN = 20  # 確保在 s[18:19] 之後
    snapshot_sgprs = []  # [(directive_idx, reg_idx, reg_name, snapshot_sgpr_num), ...]
    snapshot_sgpr_start = max(original_sgpr, SGPR_SNAPSHOT_START_MIN)
    sgpr_snapshot_idx = 0
    
    # 條件暫存器快照（獨立於要印出的暫存器）
    # directive_cond_snapshots[d_idx] = (cond_reg, snapshot_num, is_vgpr, cond_type)
    directive_cond_snapshots = {}
    
    # 表達式暫存器快照映射
    # directive_expr_snapshots[d_idx] = {原始暫存器: 快照暫存器}
    directive_expr_snapshots = {}
    
    # 用於從表達式中提取暫存器的正則表達式
    reg_pattern = re.compile(r'[vsa]\d+')
    
    for d_idx, directive in enumerate(directives):
        # 收集這個 directive 已經快照的暫存器
        snapped_regs = set()
        
        # 處理要印出的暫存器
        for r_idx, reg in enumerate(directive.registers):
            if reg.startswith('v') and not reg.startswith('vcc'):
                # VGPR 快照
                snapshot_vgpr_num = snapshot_vgpr_start + vgpr_snapshot_idx
                snapshot_vgprs.append((d_idx, r_idx, reg, snapshot_vgpr_num))
                snapped_regs.add(reg)
                vgpr_snapshot_idx += 1
            elif reg.startswith('a') and not reg.startswith('acc'):
                # AccVGPR (a0, a1, ...) — saved to VGPR snapshot slot
                snapshot_vgpr_num = snapshot_vgpr_start + vgpr_snapshot_idx
                snapshot_vgprs.append((d_idx, r_idx, reg, snapshot_vgpr_num))
                snapped_regs.add(reg)
                vgpr_snapshot_idx += 1
            elif reg.startswith('s') and not reg.startswith('scc'):
                # SGPR 快照
                snapshot_sgpr_num = snapshot_sgpr_start + sgpr_snapshot_idx
                snapshot_sgprs.append((d_idx, r_idx, reg, snapshot_sgpr_num))
                snapped_regs.add(reg)
                sgpr_snapshot_idx += 1
        
        # 處理表達式中的暫存器（如果有表達式且暫存器不在 registers 中）
        if directive.expressions:
            expr_snap_map = {}
            for expr in directive.expressions:
                for reg in reg_pattern.findall(expr):
                    if reg not in snapped_regs and reg not in expr_snap_map:
                        if reg.startswith('v') and not reg.startswith('vcc'):
                            snapshot_vgpr_num = snapshot_vgpr_start + vgpr_snapshot_idx
                            # 用特殊的 r_idx (-1) 標記這是表達式暫存器
                            snapshot_vgprs.append((d_idx, -1, reg, snapshot_vgpr_num))
                            expr_snap_map[reg] = f'v{snapshot_vgpr_num}'
                            vgpr_snapshot_idx += 1
                        elif reg.startswith('a') and not reg.startswith('acc'):
                            snapshot_vgpr_num = snapshot_vgpr_start + vgpr_snapshot_idx
                            snapshot_vgprs.append((d_idx, -1, reg, snapshot_vgpr_num))
                            expr_snap_map[reg] = f'v{snapshot_vgpr_num}'
                            vgpr_snapshot_idx += 1
                        elif reg.startswith('s') and not reg.startswith('scc'):
                            snapshot_sgpr_num = snapshot_sgpr_start + sgpr_snapshot_idx
                            snapshot_sgprs.append((d_idx, -1, reg, snapshot_sgpr_num))
                            expr_snap_map[reg] = f's{snapshot_sgpr_num}'
                            sgpr_snapshot_idx += 1
            
            if expr_snap_map:
                directive_expr_snapshots[d_idx] = expr_snap_map
                print(f"[Info] Expression register snapshots for @PRINT #{d_idx+1}: {expr_snap_map}")
        
        # 處理條件暫存器（如果有且不在 registers 中）
        if directive.condition_register and directive.condition_register not in directive.registers:
            cond_reg = directive.condition_register
            cond_type = directive.condition_type or 'f32'
            
            if cond_reg.startswith('v') and not cond_reg.startswith('vcc'):
                # VGPR 條件暫存器
                snapshot_vgpr_num = snapshot_vgpr_start + vgpr_snapshot_idx
                directive_cond_snapshots[d_idx] = (cond_reg, snapshot_vgpr_num, True, cond_type)
                vgpr_snapshot_idx += 1
                print(f"[Info] Condition register snapshot: {cond_reg} -> v{snapshot_vgpr_num} (type={cond_type})")
            elif cond_reg.startswith('s') and not cond_reg.startswith('scc'):
                # SGPR 條件暫存器
                snapshot_sgpr_num = snapshot_sgpr_start + sgpr_snapshot_idx
                directive_cond_snapshots[d_idx] = (cond_reg, snapshot_sgpr_num, False, cond_type)
                sgpr_snapshot_idx += 1
                print(f"[Info] Condition register snapshot: {cond_reg} -> s{snapshot_sgpr_num} (type={cond_type})")
    
    total_snapshot_vgprs = vgpr_snapshot_idx
    total_snapshot_sgprs = sgpr_snapshot_idx
    
    # 輸出快照資訊
    if total_snapshot_vgprs > 0:
        print(f"[Info] VGPR Snapshots: {total_snapshot_vgprs} (v{snapshot_vgpr_start} - v{snapshot_vgpr_start + total_snapshot_vgprs - 1})")
    if total_snapshot_sgprs > 0:
        print(f"[Info] SGPR Snapshots: {total_snapshot_sgprs} (s{snapshot_sgpr_start} - s{snapshot_sgpr_start + total_snapshot_sgprs - 1})")
    
    # === Scratch snapshot mode detection ===
    # Estimate total VGPR need: snapshots + builtins (workitem, tid, lane, tgid_xyz, kernarg_lo/hi, inline_s0 save)
    estimated_builtin_vgprs = 10
    VGPR_HW_LIMIT = 256
    use_scratch_snapshot = (snapshot_vgpr_start + total_snapshot_vgprs + estimated_builtin_vgprs > VGPR_HW_LIMIT)
    
    # Scratch offset tracking (only used when use_scratch_snapshot=True)
    SCRATCH_V0_SPILL_OFFSET = 0  # bytes 0-3 reserved for v0 temp spill
    scratch_offset_next = 4      # next available scratch offset (bytes)
    scratch_vgpr_map = {}        # (d_idx, r_idx) -> scratch_offset (for VGPR snapshots)
    scratch_sgpr_map = {}        # (d_idx, r_idx) -> scratch_offset (for SGPR snapshots)
    scratch_expr_map = {}        # d_idx -> {orig_reg: scratch_offset}
    scratch_cond_map = {}        # d_idx -> scratch_offset
    scratch_builtin_map = {}     # 'tid' / 'lane' / 'tgid_x' etc -> scratch_offset
    scratch_kernarg_map = {}     # 'lo' / 'hi' -> scratch_offset
    
    if use_scratch_snapshot:
        print(f"[Info] VGPR overflow detected: snapshot_vgpr_start({snapshot_vgpr_start}) + snapshots({total_snapshot_vgprs}) + builtins(~{estimated_builtin_vgprs}) > {VGPR_HW_LIMIT}")
        print(f"[Info] Switching to scratch memory snapshot mode")
        if inline_mode:
            print(f"[Info] Inline printf mode disabled in scratch snapshot mode; using defer mode")
            inline_mode = False
        
        # Assign scratch offsets for VGPR snapshots
        for d_idx, r_idx, reg, _snap_vgpr in snapshot_vgprs:
            scratch_vgpr_map[(d_idx, r_idx)] = scratch_offset_next
            scratch_offset_next += 4
        
        # Assign scratch offsets for SGPR snapshots
        for d_idx, r_idx, reg, _snap_sgpr in snapshot_sgprs:
            scratch_sgpr_map[(d_idx, r_idx)] = scratch_offset_next
            scratch_offset_next += 4
        
        # Assign scratch offsets for expression snapshots
        for d_idx, expr_snap_map in directive_expr_snapshots.items():
            scratch_expr_map[d_idx] = {}
            for orig_reg in expr_snap_map:
                scratch_expr_map[d_idx][orig_reg] = scratch_offset_next
                scratch_offset_next += 4
        
        # Assign scratch offsets for condition snapshots
        for d_idx, (cond_reg, _snap_num, _is_vgpr, _cond_type) in directive_cond_snapshots.items():
            scratch_cond_map[d_idx] = scratch_offset_next
            scratch_offset_next += 4
        
        # Built-in vars will be assigned below when we know which are needed
        # Kernarg backup will also be assigned below
        
        print(f"[Info] Scratch snapshot offsets: {scratch_offset_next} bytes allocated so far (snapshot slots)")

    # 建立 directive -> snapshot 映射
    # directive_vgpr_snapshots[d_idx] = [(r_idx, original_reg, snapshot_vgpr), ...]
    # directive_sgpr_snapshots[d_idx] = [(r_idx, original_reg, snapshot_sgpr), ...]
    directive_vgpr_snapshots = {}
    directive_sgpr_snapshots = {}
    
    for d_idx, r_idx, reg, snap_vgpr in snapshot_vgprs:
        if d_idx not in directive_vgpr_snapshots:
            directive_vgpr_snapshots[d_idx] = []
        directive_vgpr_snapshots[d_idx].append((r_idx, reg, snap_vgpr))
    
    for d_idx, r_idx, reg, snap_sgpr in snapshot_sgprs:
        if d_idx not in directive_sgpr_snapshots:
            directive_sgpr_snapshots[d_idx] = []
        directive_sgpr_snapshots[d_idx].append((r_idx, reg, snap_sgpr))
    
    # === 動態計算 clobber 範圍 ===
    MAX_VECTOR_SIZE = 32      # LLVM inline_asm vector 類型限制
    if use_scratch_snapshot:
        SGPR_KERNARG_BACKUP = max(original_sgpr, SGPR_SNAPSHOT_START_MIN)
    else:
        SGPR_KERNARG_BACKUP = snapshot_sgpr_start + total_snapshot_sgprs
    if SGPR_KERNARG_BACKUP < SGPR_SNAPSHOT_START_MIN:
        SGPR_KERNARG_BACKUP = SGPR_SNAPSHOT_START_MIN
    SGPR_RESERVED_START = 4   # s[0:3] 是系統保留的，從 s4 開始保護
    
    original_agpr = reg_info.get('agpr', 0)
    
    # 向上取整到最近的 2 的冪
    def next_power_of_2(n):
        if n <= 0:
            return 0
        n -= 1
        n |= n >> 1
        n |= n >> 2
        n |= n >> 4
        n |= n >> 8
        n |= n >> 16
        return n + 1
    
    # VGPR clobber 需要包含快照暫存器，並向上取整到 2 的冪
    vgpr_count_needed = max(1, original_vgpr + total_snapshot_vgprs) if original_vgpr > 0 else total_snapshot_vgprs
    total_vgpr = next_power_of_2(vgpr_count_needed)
    total_agpr = max(1, original_agpr) if original_agpr > 0 else 0
    
    # SGPR: 保護 s[4:...] 包含原始 SGPR 和快照 SGPR
    # s[0:3] 系統保留, s[18:19] 用於 kernarg backup
    # SGPR 快照從 snapshot_sgpr_start 開始
    # 需要保護的範圍：s[4] 到 s[snapshot_sgpr_start + total_snapshot_sgprs - 1]
    # 在 scratch snapshot mode 下，SGPR snapshot 已存入 scratch memory，不需要額外物理 SGPR
    sgpr_start = SGPR_RESERVED_START
    if use_scratch_snapshot:
        sgpr_end_needed = max(original_sgpr, SGPR_SNAPSHOT_START_MIN)
    else:
        sgpr_end_needed = snapshot_sgpr_start + total_snapshot_sgprs if total_snapshot_sgprs > 0 else max(original_sgpr, SGPR_SNAPSHOT_START_MIN)
    sgpr_end_needed = max(sgpr_end_needed, SGPR_KERNARG_BACKUP + 2)
    sgpr_count_needed = sgpr_end_needed - sgpr_start
    
    total_sgpr = next_power_of_2(sgpr_count_needed) if sgpr_count_needed > 0 else 0
    # 限制最大為 32（因為 MAX_VECTOR_SIZE = 32）
    total_sgpr = min(total_sgpr, MAX_VECTOR_SIZE)
    
    # 計算每種暫存器需要多少個 clobber 區塊
    num_vgpr_blocks = (total_vgpr + MAX_VECTOR_SIZE - 1) // MAX_VECTOR_SIZE if total_vgpr > 0 else 0
    num_sgpr_blocks = (total_sgpr + MAX_VECTOR_SIZE - 1) // MAX_VECTOR_SIZE if total_sgpr > 0 else 0
    num_agpr_blocks = (total_agpr + MAX_VECTOR_SIZE - 1) // MAX_VECTOR_SIZE if total_agpr > 0 else 0
    
    print(f"[Info] Register clobbering:")
    if total_vgpr > 0:
        print(f"       VGPR: v[0:{total_vgpr - 1}] ({total_vgpr} registers, {num_vgpr_blocks} block(s))")
    if total_sgpr > 0:
        print(f"       SGPR: s[{sgpr_start}:{sgpr_start + total_sgpr - 1}] ({total_sgpr} registers, {num_sgpr_blocks} block(s))")
    if total_agpr > 0:
        print(f"       AGPR: a[0:{total_agpr - 1}] ({total_agpr} registers, {num_agpr_blocks} block(s))")
    
    # === 生成 kernarg 保存程式碼 ===
    # 注意：移除了有問題的 clobber 機制（會導致 LLVM 生成 spill 代碼）
    # 改為只保存 kernarg pointer，快照機制會在適當位置保存暫存器值
    clobber_start_lines = ['              // === Kernarg Backup Start ===']
    
    kernarg_backup_in_vgpr = any(d.trace_max for d in directives)
    
    # 檢查是否有條件式 printf，如果有則保存 workitem_id (v0)
    # 使用在快照暫存器之後的 VGPR
    WORKITEM_ID_BACKUP_VGPR = snapshot_vgpr_start + total_snapshot_vgprs
    has_conditional_printf = any(d.condition for d in directives)
    
    # === 內建變數支援：$tid / $lane / $tgid_* ===
    # 檢查是否有任何 directive 使用了內建變數
    uses_tid = any(d.uses_tid for d in directives)
    uses_lane = any(d.uses_lane for d in directives)
    uses_tgid = any(getattr(d, 'uses_tgid', False) for d in directives)
    
    # 也檢查 timestamp directive 的條件是否使用了 $tid / $lane / $tgid_*
    for ts in timestamp_directives:
        if ts.condition:
            if '$tid' in ts.condition:
                uses_tid = True
            if '$lane' in ts.condition:
                uses_lane = True
            if '$tgid_x' in ts.condition or '$tgid_y' in ts.condition or '$tgid_z' in ts.condition:
                uses_tgid = True
    
    if use_scratch_snapshot:
        # =====================================================================
        # Scratch snapshot mode: all backups go to scratch memory, no extra VGPRs
        # =====================================================================
        next_builtin_vgpr = original_vgpr  # Do NOT inflate VGPR count
        
        # Allocate scratch offsets for built-in vars
        if uses_tid:
            scratch_builtin_map['tid'] = scratch_offset_next
            scratch_offset_next += 4
        if uses_lane:
            scratch_builtin_map['lane'] = scratch_offset_next
            scratch_offset_next += 4
        if uses_tgid:
            scratch_builtin_map['tgid_x'] = scratch_offset_next
            scratch_offset_next += 4
            scratch_builtin_map['tgid_y'] = scratch_offset_next
            scratch_offset_next += 4
            scratch_builtin_map['tgid_z'] = scratch_offset_next
            scratch_offset_next += 4
        
        # Kernarg backup always goes to scratch in scratch mode
        scratch_kernarg_map['lo'] = scratch_offset_next
        scratch_offset_next += 4
        scratch_kernarg_map['hi'] = scratch_offset_next
        scratch_offset_next += 4
        
        # Helper: generate inline_asm to store SGPR to scratch (needs v0 spill)
        def _scratch_store_sgpr_asm(sgpr_name: str, scratch_off: int) -> str:
            """Generate multi-instruction asm to store an SGPR to scratch via v0 temp."""
            return (
                f"scratch_store_dword off, v0, off offset:{SCRATCH_V0_SPILL_OFFSET}\\0A"
                f"v_mov_b32 v0, {sgpr_name}\\0A"
                f"scratch_store_dword off, v0, off offset:{scratch_off}\\0A"
                f"scratch_load_dword v0, off, off offset:{SCRATCH_V0_SPILL_OFFSET}\\0A"
                f"s_waitcnt vmcnt(0)"
            )
        
        # Kernarg backup: s0/s1 -> scratch
        clobber_start_lines.append('              // Save kernarg pointer for printf (to scratch memory)')
        clobber_start_lines.append(
            f'              llvm.inline_asm has_side_effects "{_scratch_store_sgpr_asm("s0", scratch_kernarg_map["lo"])}", ""  : () -> ()'
        )
        clobber_start_lines.append(
            f'              llvm.inline_asm has_side_effects "{_scratch_store_sgpr_asm("s1", scratch_kernarg_map["hi"])}", ""  : () -> ()'
        )
        
        # $tid: v0 -> scratch (v0 is VGPR, can store directly)
        if uses_tid:
            tid_off = scratch_builtin_map['tid']
            clobber_start_lines.append(f'              // === Built-in variable: $tid (local thread ID) -> scratch[{tid_off}] ===')
            clobber_start_lines.append(
                f'              llvm.inline_asm has_side_effects "scratch_store_dword off, v0, off offset:{tid_off}", ""  : () -> ()'
            )
            print(f"[Info] Built-in $tid: v0 -> scratch[{tid_off}]")
        
        # $lane: compute v_mbcnt into v0 (spill v0 first), store to scratch, restore v0
        if uses_lane:
            lane_off = scratch_builtin_map['lane']
            clobber_start_lines.append(f'              // === Built-in variable: $lane (wavefront lane ID) -> scratch[{lane_off}] ===')
            clobber_start_lines.append(
                f'              llvm.inline_asm has_side_effects "'
                f'scratch_store_dword off, v0, off offset:{SCRATCH_V0_SPILL_OFFSET}\\0A'
                f'v_mbcnt_lo_u32_b32 v0, -1, 0\\0A'
                f'v_mbcnt_hi_u32_b32 v0, -1, v0\\0A'
                f'scratch_store_dword off, v0, off offset:{lane_off}\\0A'
                f'scratch_load_dword v0, off, off offset:{SCRATCH_V0_SPILL_OFFSET}\\0A'
                f's_waitcnt vmcnt(0)", ""  : () -> ()'
            )
            print(f"[Info] Built-in $lane: v_mbcnt -> scratch[{lane_off}]")
        
        # $tgid_x/y/z: s2/s3/s4 -> scratch (SGPRs, need v0 temp)
        if uses_tgid:
            clobber_start_lines.append('              // === Built-in variable: $tgid_x/$tgid_y/$tgid_z (workgroup ID) -> scratch ===')
            for tgid_name, sgpr_name in [('tgid_x', 's2'), ('tgid_y', 's3'), ('tgid_z', 's4')]:
                tgid_off = scratch_builtin_map[tgid_name]
                clobber_start_lines.append(
                    f'              llvm.inline_asm has_side_effects "{_scratch_store_sgpr_asm(sgpr_name, tgid_off)}", ""  : () -> ()'
                )
            print(f"[Info] Built-in $tgid_x/y/z: s2/s3/s4 -> scratch[{scratch_builtin_map.get('tgid_x')}/{scratch_builtin_map.get('tgid_y')}/{scratch_builtin_map.get('tgid_z')}]")
        
        # Set VGPR backup vars to None (not used in scratch mode)
        VGPR_KERNARG_BACKUP_LO = None
        VGPR_KERNARG_BACKUP_HI = None
        kernarg_backup_in_vgpr = False
        TID_BACKUP_VGPR = None
        LANE_BACKUP_VGPR = None
        TGID_X_BACKUP_VGPR = None
        TGID_Y_BACKUP_VGPR = None
        TGID_Z_BACKUP_VGPR = None
        VGPR_INLINE_S0_SAVE_LO = None
        VGPR_INLINE_S0_SAVE_HI = None
        
        print(f"[Info] Scratch snapshot: total {scratch_offset_next} bytes (aligned to 16: {((scratch_offset_next + 15) // 16) * 16})")
        
    else:
        # =====================================================================
        # Original VGPR snapshot mode
        # =====================================================================
        if has_conditional_printf:
            clobber_start_lines.append(f'              // Save workitem_id (v0) to v{WORKITEM_ID_BACKUP_VGPR} for conditional printf')
            clobber_start_lines.append(f'              llvm.inline_asm has_side_effects "v_mov_b32 v{WORKITEM_ID_BACKUP_VGPR}, v0", ""  : () -> ()')
        
        # 分配專用 VGPR 給內建變數（在 WORKITEM_ID_BACKUP 之後）
        next_builtin_vgpr = WORKITEM_ID_BACKUP_VGPR + (1 if has_conditional_printf else 0)

        # 保存 kernarg pointer（trace 模式用 VGPR 備份，避免佔用 SGPR）
        VGPR_KERNARG_BACKUP_LO = None
        VGPR_KERNARG_BACKUP_HI = None
        if kernarg_backup_in_vgpr:
            VGPR_KERNARG_BACKUP_LO = next_builtin_vgpr
            VGPR_KERNARG_BACKUP_HI = next_builtin_vgpr + 1
            next_builtin_vgpr += 2
            clobber_start_lines.append('              // Save kernarg pointer for printf (to VGPRs)')
            clobber_start_lines.append(
                f'              llvm.inline_asm has_side_effects "v_mov_b32 v{VGPR_KERNARG_BACKUP_LO}, s0", ""  : () -> ()'
            )
            clobber_start_lines.append(
                f'              llvm.inline_asm has_side_effects "v_mov_b32 v{VGPR_KERNARG_BACKUP_HI}, s1", ""  : () -> ()'
            )
        else:
            clobber_start_lines.append('              // Save kernarg pointer for printf (to s[18:19])')
            clobber_start_lines.append(
                f'              llvm.inline_asm has_side_effects "s_mov_b64 s[{SGPR_KERNARG_BACKUP}:{SGPR_KERNARG_BACKUP+1}], s[0:1]", ""  : () -> ()'
            )
        
        # $tid: Local Thread ID (workitem_id_x) - 備份 v0
        TID_BACKUP_VGPR = None
        if uses_tid:
            TID_BACKUP_VGPR = next_builtin_vgpr
            next_builtin_vgpr += 1
            clobber_start_lines.append(f'              // === Built-in variable: $tid (local thread ID) ===')
            clobber_start_lines.append(f'              // Backup workitem_id_x (v0) to v{TID_BACKUP_VGPR}')
            clobber_start_lines.append(f'              llvm.inline_asm has_side_effects "v_mov_b32 v{TID_BACKUP_VGPR}, v0", ""  : () -> ()')
            print(f"[Info] Built-in $tid: v0 -> v{TID_BACKUP_VGPR}")
        
        # $lane: Wavefront Lane ID (0-63) - 使用 v_mbcnt 計算
        LANE_BACKUP_VGPR = None
        if uses_lane:
            LANE_BACKUP_VGPR = next_builtin_vgpr
            next_builtin_vgpr += 1
            clobber_start_lines.append(f'              // === Built-in variable: $lane (wavefront lane ID) ===')
            clobber_start_lines.append(f'              // Calculate lane ID using v_mbcnt instructions')
            clobber_start_lines.append(f'              llvm.inline_asm has_side_effects "v_mbcnt_lo_u32_b32 v{LANE_BACKUP_VGPR}, -1, 0", ""  : () -> ()')
            clobber_start_lines.append(f'              llvm.inline_asm has_side_effects "v_mbcnt_hi_u32_b32 v{LANE_BACKUP_VGPR}, -1, v{LANE_BACKUP_VGPR}", ""  : () -> ()')
            print(f"[Info] Built-in $lane: v_mbcnt -> v{LANE_BACKUP_VGPR}")

        # $tgid_x/$tgid_y/$tgid_z: Workgroup ID - 備份 s2/s3/s4，避免後續被覆寫
        TGID_X_BACKUP_VGPR = None
        TGID_Y_BACKUP_VGPR = None
        TGID_Z_BACKUP_VGPR = None
        if uses_tgid:
            TGID_X_BACKUP_VGPR = next_builtin_vgpr
            TGID_Y_BACKUP_VGPR = next_builtin_vgpr + 1
            TGID_Z_BACKUP_VGPR = next_builtin_vgpr + 2
            next_builtin_vgpr += 3
            clobber_start_lines.append('              // === Built-in variable: $tgid_x/$tgid_y/$tgid_z (workgroup ID) ===')
            clobber_start_lines.append(f'              llvm.inline_asm has_side_effects "v_mov_b32 v{TGID_X_BACKUP_VGPR}, s2", ""  : () -> ()')
            clobber_start_lines.append(f'              llvm.inline_asm has_side_effects "v_mov_b32 v{TGID_Y_BACKUP_VGPR}, s3", ""  : () -> ()')
            clobber_start_lines.append(f'              llvm.inline_asm has_side_effects "v_mov_b32 v{TGID_Z_BACKUP_VGPR}, s4", ""  : () -> ()')
            print(f"[Info] Built-in $tgid_x/y/z: s2/s3/s4 -> v{TGID_X_BACKUP_VGPR}/v{TGID_Y_BACKUP_VGPR}/v{TGID_Z_BACKUP_VGPR}")

        # Inline printf 需要備份 s0/s1，使用 VGPR 避免超出 SGPR 上限
        VGPR_INLINE_S0_SAVE_LO = None
        VGPR_INLINE_S0_SAVE_HI = None
        if inline_mode:
            VGPR_INLINE_S0_SAVE_LO = next_builtin_vgpr
            VGPR_INLINE_S0_SAVE_HI = next_builtin_vgpr + 1
            next_builtin_vgpr += 2
            print(f"[Info] Inline printf: s0/s1 backup -> v{VGPR_INLINE_S0_SAVE_LO}:v{VGPR_INLINE_S0_SAVE_HI}")

    # === Trace buffer (VGPR) ===
    # 依據 max= 配置，為每個 @PRINT 分配 VGPR trace buffer
    trace_buffers = {}  # d_idx -> {'max': int, 'counter': v, 'regs': [r...], 'slots': {reg: [v...]}, 'tmp': {reg: v}, 'vcc_save': (v_lo, v_hi)}
    VGPR_LIMIT = 256  # v0 ~ v255
    for d_idx, directive in enumerate(directives):
        if not directive.trace_max:
            continue
        if use_scratch_snapshot:
            print(f"[Warning] Trace max for @PRINT #{d_idx+1} disabled in scratch snapshot mode (insufficient VGPRs)")
            continue
        trace_max = directive.trace_max
        # 僅支援 32-bit 值
        if directive.all_placeholders:
            unsupported = [t for _, t in directive.all_placeholders if t in ("i64", "f64", "ptr")]
            if unsupported:
                print(f"[Warning] Trace max for @PRINT #{d_idx+1} skipped (unsupported types: {unsupported})")
                continue
        # trace 需要保存寄存器（包含 expressions 內的寄存器）
        trace_regs = set(directive.registers)
        if directive.expressions:
            for expr in directive.expressions:
                for reg in reg_pattern.findall(expr):
                    trace_regs.add(reg)
        if not trace_regs:
            print(f"[Warning] Trace max for @PRINT #{d_idx+1} skipped (no registers to trace)")
            continue
        if trace_max > 64:
            print(f"[Warning] Trace max {trace_max} may consume many VGPRs; consider smaller max.")
        # 對 VGPR 上限進行保護
        ordered_regs = sorted(trace_regs)
        reg_count = len(ordered_regs)
        base_need = 3 + reg_count  # counter + vcc_save(lo/hi) + tmp per reg
        avail = VGPR_LIMIT - next_builtin_vgpr
        max_possible = 0
        if avail > base_need and reg_count > 0:
            max_possible = (avail - base_need) // reg_count
        if max_possible <= 0:
            print(f"[Warning] Trace max for @PRINT #{d_idx+1} disabled (insufficient VGPRs)")
            continue
        if trace_max > max_possible:
            print(f"[Warning] Trace max {trace_max} capped to {max_possible} due to VGPR limit")
            trace_max = max_possible
        counter_vgpr = next_builtin_vgpr
        next_builtin_vgpr += 1
        vcc_save_lo = next_builtin_vgpr
        vcc_save_hi = next_builtin_vgpr + 1
        next_builtin_vgpr += 2
        tmp_regs = {}
        slots = {}
        for reg in ordered_regs:
            tmp_regs[reg] = next_builtin_vgpr
            next_builtin_vgpr += 1
            slots[reg] = []
            for _ in range(trace_max):
                slots[reg].append(next_builtin_vgpr)
                next_builtin_vgpr += 1
        trace_buffers[d_idx] = {
            'max': trace_max,
            'counter': counter_vgpr,
            'regs': ordered_regs,
            'slots': slots,
            'tmp': tmp_regs,
            'vcc_save': (vcc_save_lo, vcc_save_hi),
        }
        clobber_start_lines.append(f'              // Trace counter for @PRINT #{d_idx+1}')
        clobber_start_lines.append(f'              llvm.inline_asm has_side_effects "v_mov_b32 v{counter_vgpr}, 0", ""  : () -> ()')
        print(f"[Info] Trace buffer for @PRINT #{d_idx+1}: max={trace_max}, counter=v{counter_vgpr}, regs={ordered_regs}")
    
    # === Timestamp Profiling 支援 ===
    # 為每對 @TIMESTAMP_START/@TIMESTAMP_END 分配 VGPR
    # - 2 個 VGPR 用於保存開始時間（低 32 位和高 32 位）
    # - 結束時間和差值在 printf 時計算
    timestamp_vgpr_map = {}  # {label: {'start_lo': vgpr, 'start_hi': vgpr}}
    
    # 匹配 start 和 end directive
    ts_starts = [d for d in timestamp_directives if d.directive_type == "start"]
    ts_ends = [d for d in timestamp_directives if d.directive_type == "end"]
    
    # 檢測是否有 atomic 模式
    has_atomic_timestamp = any(d.mode == "atomic" for d in timestamp_directives)
    
    # Atomic 模式需要額外的 SGPR/VGPR 來存儲 buffer 地址
    # SGPR_TS_BUFFER: 存儲 timestamp buffer 地址
    SGPR_TS_BUFFER = 22  # s[22:23] 用於 timestamp buffer 地址
    # VGPR for timestamp buffer address (for global addressing)
    VGPR_TS_ADDR_LO = None
    VGPR_TS_ADDR_HI = None
    
    if has_atomic_timestamp:
        # 完整 atomic timestamp 實現
        clobber_start_lines.append(f'              // === Atomic Timestamp Setup ===')
        
        # 分配 VGPR 用於存儲 buffer 地址
        VGPR_TS_ADDR_LO = next_builtin_vgpr
        VGPR_TS_ADDR_HI = next_builtin_vgpr + 1
        next_builtin_vgpr += 2
        
        print(f"[Info] Atomic timestamp: VGPRs v{VGPR_TS_ADDR_LO}:v{VGPR_TS_ADDR_HI} for buffer address")
    
    if ts_starts:
        clobber_start_lines.append(f'              // === Timestamp Profiling Setup ===')
        
        for ts_idx, ts_start in enumerate(ts_starts):
            label = ts_start.label or f"default_{ts_idx}"
            
            # 分配 2 個 VGPR 用於保存開始時間
            start_lo_vgpr = next_builtin_vgpr
            start_hi_vgpr = next_builtin_vgpr + 1
            next_builtin_vgpr += 2
            
            timestamp_vgpr_map[label] = {
                'start_lo': start_lo_vgpr,
                'start_hi': start_hi_vgpr,
                'ts_start': ts_start,
                'ts_end': None,  # 稍後匹配
                'ts_addr_lo': VGPR_TS_ADDR_LO,  # For atomic mode
                'ts_addr_hi': VGPR_TS_ADDR_HI   # For atomic mode
            }
            
            print(f"[Info] Timestamp [{label}]: start time -> v{start_lo_vgpr}:v{start_hi_vgpr}")
        
        # 匹配 end directive
        for ts_end in ts_ends:
            label = ts_end.label or "default_0"
            if label in timestamp_vgpr_map:
                timestamp_vgpr_map[label]['ts_end'] = ts_end
            else:
                # 嘗試匹配第一個沒有 label 的 start
                for lbl, info in timestamp_vgpr_map.items():
                    if info['ts_end'] is None:
                        timestamp_vgpr_map[lbl]['ts_end'] = ts_end
                        break
    
    clobber_start_lines.append('              // === Kernarg Backup End ===')
    clobber_start = '\n'.join(clobber_start_lines) + '\n'
    
    # 生成 printf 區塊（使用快照暫存器）- 包含 kernarg restore
    need_deferred_printf_section = (not inline_mode) or bool(timestamp_directives)
    printf_blocks = []
    if need_deferred_printf_section:
        if use_scratch_snapshot:
            # Scratch mode: restore kernarg from scratch memory (v0 is freely available at kernel end)
            kernarg_lo_off = scratch_kernarg_map['lo']
            kernarg_hi_off = scratch_kernarg_map['hi']
            printf_blocks.append('              // Restore kernarg pointer for printf (from scratch memory)')
            printf_blocks.append(
                f'              llvm.inline_asm has_side_effects "'
                f'scratch_load_dword v0, off, off offset:{kernarg_lo_off}\\0A'
                f's_waitcnt vmcnt(0)\\0A'
                f'v_readfirstlane_b32 s0, v0\\0A'
                f'scratch_load_dword v0, off, off offset:{kernarg_hi_off}\\0A'
                f's_waitcnt vmcnt(0)\\0A'
                f'v_readfirstlane_b32 s1, v0", ""  : () -> ()'
            )
        elif kernarg_backup_in_vgpr:
            printf_blocks.append('              // Restore kernarg pointer for printf (from VGPRs)')
            printf_blocks.append(
                f'              llvm.inline_asm has_side_effects "v_readfirstlane_b32 s0, v{VGPR_KERNARG_BACKUP_LO}", ""  : () -> ()'
            )
            printf_blocks.append(
                f'              llvm.inline_asm has_side_effects "v_readfirstlane_b32 s1, v{VGPR_KERNARG_BACKUP_HI}", ""  : () -> ()'
            )
        else:
            printf_blocks.append('              // Restore kernarg pointer for printf (from s[18:19])')
            printf_blocks.append(
                f'              llvm.inline_asm has_side_effects "s_mov_b64 s[0:1], s[{SGPR_KERNARG_BACKUP}:{SGPR_KERNARG_BACKUP+1}]", ""  : () -> ()'
            )
        if not inline_mode:
            def _indent_block(block: str, prefix: str = "  ") -> str:
                return "\n".join(prefix + ln if ln.strip() else ln for ln in block.split("\n"))

            for i, directive in enumerate(directives):
                # 使用 VGPR 和 SGPR 快照暫存器生成 printf
                vgpr_snaps = directive_vgpr_snapshots.get(i, [])
                sgpr_snaps = directive_sgpr_snapshots.get(i, [])
                cond_snap = directive_cond_snapshots.get(i, None)
                expr_snap = directive_expr_snapshots.get(i, None)
                # 如果有條件式 printf，傳入 workitem_id 備份暫存器
                tid_backup = WORKITEM_ID_BACKUP_VGPR if has_conditional_printf else None
                # 傳入內建變數備份暫存器
                builtin_vars = {
                    'tid_vgpr': TID_BACKUP_VGPR,
                    'lane_vgpr': LANE_BACKUP_VGPR,
                    'tgid_x_vgpr': TGID_X_BACKUP_VGPR,
                    'tgid_y_vgpr': TGID_Y_BACKUP_VGPR,
                    'tgid_z_vgpr': TGID_Z_BACKUP_VGPR,
                    'tgid_x_sgpr': 2,
                    'tgid_y_sgpr': 3,
                    'tgid_z_sgpr': 4,
                }
                
                # Build per-directive scratch_snapshot_map for generate_printf_with_snapshot
                dir_scratch_map = None
                if use_scratch_snapshot:
                    dir_scratch_map = {
                        'tid': scratch_builtin_map.get('tid'),
                        'lane': scratch_builtin_map.get('lane'),
                        'tgid_x': scratch_builtin_map.get('tgid_x'),
                        'tgid_y': scratch_builtin_map.get('tgid_y'),
                        'tgid_z': scratch_builtin_map.get('tgid_z'),
                        'reg_snapshots': {},
                        'expr': {},
                        'cond': {},
                    }
                    # Collect VGPR snapshot offsets for this directive
                    for d_idx, r_idx, reg, _snap_vgpr in snapshot_vgprs:
                        if d_idx == i:
                            s_off = scratch_vgpr_map.get((d_idx, r_idx))
                            if s_off is not None:
                                dir_scratch_map['reg_snapshots'][reg] = s_off
                    # Collect SGPR snapshot offsets for this directive
                    for d_idx, r_idx, reg, _snap_sgpr in snapshot_sgprs:
                        if d_idx == i:
                            s_off = scratch_sgpr_map.get((d_idx, r_idx))
                            if s_off is not None:
                                dir_scratch_map['reg_snapshots'][reg] = s_off
                    # Collect expression snapshot offsets
                    if i in scratch_expr_map:
                        dir_scratch_map['expr'] = scratch_expr_map[i]
                    # Collect condition snapshot offset
                    if i in scratch_cond_map:
                        cond_reg_name = directive_cond_snapshots[i][0] if i in directive_cond_snapshots else None
                        if cond_reg_name:
                            dir_scratch_map['cond'][cond_reg_name] = scratch_cond_map[i]

                if i in trace_buffers:
                    trace_cfg = trace_buffers[i]
                    trace_max = trace_cfg['max']
                    trace_counter = trace_cfg['counter']
                    trace_regs = trace_cfg['regs']
                    trace_slots = trace_cfg['slots']

                    printf_blocks.append(f'              // === Trace printf for @PRINT #{i+1} (max={trace_max}) ===')
                    printf_blocks.append(
                        f'              %trace_count_{i} = llvm.inline_asm has_side_effects asm_dialect = att "v_mov_b32 $0, v{trace_counter}", "=v": () -> i32'
                    )
                    for idx in range(trace_max):
                        printf_blocks.append(
                            f'              %trace_idx_const_{i}_{idx} = arith.constant {idx} : i32'
                        )
                        printf_blocks.append(
                            f'              %trace_idx_cmp_{i}_{idx} = arith.cmpi ult, %trace_idx_const_{i}_{idx}, %trace_count_{i} : i32'
                        )
                        printf_blocks.append(f'              scf.if %trace_idx_cmp_{i}_{idx} {{')
                        iter_vgpr_snaps = [(0, reg, trace_slots[reg][idx]) for reg in trace_regs]
                        block = generate_printf_with_snapshot(
                            directive, i, iter_vgpr_snaps, [], tid_backup, cond_snap, expr_snap, builtin_vars
                        )
                        printf_blocks.append(_indent_block(block))
                        printf_blocks.append(f'              }}')
                    printf_blocks.append(f'              // === End Trace printf ===')
                else:
                    printf_blocks.append(
                        generate_printf_with_snapshot(
                            directive, i, vgpr_snaps, sgpr_snaps, tid_backup, cond_snap, expr_snap, builtin_vars,
                            scratch_snapshot_map=dir_scratch_map
                        )
                    )
    
    # === 生成 Timestamp 輸出 ===
    # 在 printf section 的最後，記錄結束時間並計算經過時間
    # 注意：atomic 模式不需要 printf 輸出，host 會直接讀取 buffer
    for label, ts_info in timestamp_vgpr_map.items():
        ts_start = ts_info['ts_start']
        ts_end = ts_info.get('ts_end')
        start_lo = ts_info['start_lo']
        start_hi = ts_info['start_hi']
        
        if ts_end is None:
            print(f"[Warning] @TIMESTAMP_START [{label}] has no matching @TIMESTAMP_END")
            continue
        
        # Atomic 模式：把 atomic_max 操作放在 printf section 的開頭
        # 這樣可以在 printf 之前記錄結束時間，避免 printf 開銷影響 kernel wall time
        is_atomic = (ts_start.mode == "atomic" or ts_end.mode == "atomic")
        if is_atomic:
            printf_blocks.append(f'              // === Timestamp [{label}] (mode=atomic) ===')
            
            # === ATOMIC_MAX 操作（放在 printf 之前）===
            # 重新計算 timestamp buffer 地址 (C + N * 4 + 8)
            printf_blocks.append(f'              // Atomic max: record end time BEFORE printf')
            printf_blocks.append(f'              llvm.inline_asm has_side_effects "s_load_dwordx2 s[{SGPR_TS_BUFFER}:{SGPR_TS_BUFFER+1}], s[{SGPR_KERNARG_BACKUP}:{SGPR_KERNARG_BACKUP+1}], 0x10\\0As_waitcnt lgkmcnt(0)", ""  : () -> ()')
            printf_blocks.append(f'              llvm.inline_asm has_side_effects "s_load_dword s24, s[{SGPR_KERNARG_BACKUP}:{SGPR_KERNARG_BACKUP+1}], 0x18\\0As_waitcnt lgkmcnt(0)", ""  : () -> ()')
            printf_blocks.append(f'              llvm.inline_asm has_side_effects "s_lshl_b32 s24, s24, 2\\0As_add_u32 s{SGPR_TS_BUFFER}, s{SGPR_TS_BUFFER}, s24\\0As_addc_u32 s{SGPR_TS_BUFFER+1}, s{SGPR_TS_BUFFER+1}, 0\\0As_add_u32 s{SGPR_TS_BUFFER}, s{SGPR_TS_BUFFER}, 8\\0As_addc_u32 s{SGPR_TS_BUFFER+1}, s{SGPR_TS_BUFFER+1}, 0", ""  : () -> ()')
            # 查找這個 timestamp 的 VGPR 地址
            ts_info = timestamp_vgpr_map.get(label, {})
            ts_addr_lo = ts_info.get('ts_addr_lo')
            ts_addr_hi = ts_info.get('ts_addr_hi')
            end_lo = ts_info.get('start_lo', 26)  # 重用 start VGPRs
            end_hi = ts_info.get('start_hi', 27)
            if ts_addr_lo is not None:
                printf_blocks.append(f'              llvm.inline_asm has_side_effects "v_mov_b32 v{ts_addr_lo}, s{SGPR_TS_BUFFER}\\0Av_mov_b32 v{ts_addr_hi}, s{SGPR_TS_BUFFER+1}", ""  : () -> ()')
                # Atomic 模式使用 s_memrealtime（跨 workgroup 同步，~100 MHz）
                printf_blocks.append(f'              llvm.inline_asm has_side_effects "s_memrealtime s[20:21]\\0As_waitcnt lgkmcnt(0)", ""  : () -> ()')
                printf_blocks.append(f'              llvm.inline_asm has_side_effects "v_mov_b32 v{end_lo}, s20\\0Av_mov_b32 v{end_hi}, s21", ""  : () -> ()')
                printf_blocks.append(f'              llvm.inline_asm has_side_effects "global_atomic_umax_x2 v[{ts_addr_lo}:{ts_addr_hi}], v[{end_lo}:{end_hi}], off", ""  : () -> ()')
            
            # === Printf 部分（用於保持 MLIR 代碼結構，但只輸出最少信息）===
            # 只讓 lane 0 輸出，進一步減少 printf 開銷
            printf_blocks.append(f'              %ts_elapsed_{label} = llvm.inline_asm has_side_effects asm_dialect = att')
            printf_blocks.append(f'                "v_sub_u32 $0, s20, v{start_lo}", "=v": () -> i32')
            # 無條件 printf（保持 MLIR 結構完整）
            printf_blocks.append(f'              gpu.printf ""')  # 空字串 printf 開銷最小
            printf_blocks.append(f'              // === End Timestamp ===')
            continue
        
        # 檢查是否為 wallclock 模式
        is_wallclock = (ts_start.mode == "wallclock" or ts_end.mode == "wallclock")
        
        printf_blocks.append(f'              // === Timestamp END [{label}] (mode={"wallclock" if is_wallclock else "local"}) ===')
        
        if is_wallclock:
            # Wallclock 模式：輸出 64-bit START 時間戳 + elapsed
            # Host 端計算: end = start + elapsed, kernel_time = max(end) - min(start)
            # 
            # 注意：我們只輸出 START 的絕對時間戳（已正確快照）
            # END 時間 = START + elapsed，避免 printf section 時間不一致的問題
            printf_blocks.append(f'              // Wallclock mode: output 64-bit START timestamp + elapsed')
            printf_blocks.append(f'              // Host calculates: end = start + elapsed')
            printf_blocks.append(f'              // Read start time (64-bit) from snapshot VGPRs')
            printf_blocks.append(f'              %ts_start_lo_{label} = llvm.inline_asm has_side_effects asm_dialect = att "v_mov_b32 $0, v{start_lo}", "=v": () -> i32')
            printf_blocks.append(f'              %ts_start_hi_{label} = llvm.inline_asm has_side_effects asm_dialect = att "v_mov_b32 $0, v{start_hi}", "=v": () -> i32')
            printf_blocks.append(f'              // Calculate elapsed using current time - start time')
            printf_blocks.append(f'              // Wallclock mode uses s_memrealtime (globally synchronized, ~100 MHz)')
            printf_blocks.append(f'              %ts_elapsed_{label} = llvm.inline_asm has_side_effects asm_dialect = att')
            printf_blocks.append(f'                "s_memrealtime s[20:21]\\0As_waitcnt lgkmcnt(0)\\0Av_sub_u32 $0, s20, v{start_lo}", "=v": () -> i32')
        else:
            # 普通模式（local）：只輸出 elapsed，使用 s_memtime（高精度，1.7 GHz）
            printf_blocks.append(f'              // Calculate elapsed time using GPU ISA directly')
            printf_blocks.append(f'              // Local mode uses s_memtime (high precision, ~1.7 GHz)')
            printf_blocks.append(f'              %ts_elapsed_{label} = llvm.inline_asm has_side_effects asm_dialect = att')
            printf_blocks.append(f'                "s_memtime s[20:21]\\0As_waitcnt lgkmcnt(0)\\0Av_sub_u32 $0, s20, v{start_lo}", "=v": () -> i32')
        
        # 生成 printf 格式字串
        if is_wallclock:
            # 輸出 64-bit START 時間戳（hi:lo 格式）+ elapsed
            # Host 計算: end = start + elapsed, kernel_time = max(end) - min(start)
            printf_fmt = f'[WallClock {label}] start=%u:%u elapsed=%u ticks\\0A'
            printf_args = f'%ts_start_hi_{label}, %ts_start_lo_{label}, %ts_elapsed_{label}'
            printf_types = 'i32, i32, i32'
        else:
            printf_fmt = f'[Timestamp {label}] elapsed = %u ticks\\0A'
            printf_args = f'%ts_elapsed_{label}'
            printf_types = 'i32'
        
        # 輸出結果
        # 根據 ts_end 的 condition 決定是否有條件輸出
        if ts_end.condition:
            # 有條件輸出（例如 if $lane == 0:）
            cond_match = re.match(r'(\$tid|\$lane)_(\w+)\((-?\d+)\)', ts_end.condition)
            if cond_match:
                builtin_var, cmp_op, cmp_value = cond_match.groups()
                
                if builtin_var == '$lane' and LANE_BACKUP_VGPR is not None:
                    printf_blocks.append(f'              // Conditional timestamp output: {builtin_var} {cmp_op} {cmp_value}')
                    printf_blocks.append(f'              %ts_lane_{label} = llvm.inline_asm has_side_effects asm_dialect = att "v_mov_b32 $0, v{LANE_BACKUP_VGPR}", "=v": () -> i32')
                    cmp_ops = {'eq': 'eq', 'ne': 'ne', 'lt': 'slt', 'le': 'sle', 'gt': 'sgt', 'ge': 'sge'}
                    mlir_cmp = cmp_ops.get(cmp_op, 'eq')
                    printf_blocks.append(f'              %ts_cmp_val_{label} = arith.constant {cmp_value} : i32')
                    printf_blocks.append(f'              %ts_cond_{label} = arith.cmpi {mlir_cmp}, %ts_lane_{label}, %ts_cmp_val_{label} : i32')
                    printf_blocks.append(f'              scf.if %ts_cond_{label} {{')
                    printf_blocks.append(f'                gpu.printf "{printf_fmt}", {printf_args} : {printf_types}')
                    printf_blocks.append(f'              }}')
                elif builtin_var == '$tid' and TID_BACKUP_VGPR is not None:
                    printf_blocks.append(f'              // Conditional timestamp output: {builtin_var} {cmp_op} {cmp_value}')
                    printf_blocks.append(f'              %ts_tid_{label} = llvm.inline_asm has_side_effects asm_dialect = att "v_mov_b32 $0, v{TID_BACKUP_VGPR}", "=v": () -> i32')
                    cmp_ops = {'eq': 'eq', 'ne': 'ne', 'lt': 'slt', 'le': 'sle', 'gt': 'sgt', 'ge': 'sge'}
                    mlir_cmp = cmp_ops.get(cmp_op, 'eq')
                    printf_blocks.append(f'              %ts_cmp_val_{label} = arith.constant {cmp_value} : i32')
                    printf_blocks.append(f'              %ts_cond_{label} = arith.cmpi {mlir_cmp}, %ts_tid_{label}, %ts_cmp_val_{label} : i32')
                    printf_blocks.append(f'              scf.if %ts_cond_{label} {{')
                    printf_blocks.append(f'                gpu.printf "{printf_fmt}", {printf_args} : {printf_types}')
                    printf_blocks.append(f'              }}')
                else:
                    # Fallback: 無條件輸出
                    printf_blocks.append(f'              gpu.printf "{printf_fmt}", {printf_args} : {printf_types}')
            else:
                # 無法解析條件，無條件輸出
                printf_blocks.append(f'              gpu.printf "{printf_fmt}", {printf_args} : {printf_types}')
        else:
            # 無條件輸出（每個 thread 都輸出）
            printf_blocks.append(f'              gpu.printf "{printf_fmt}", {printf_args} : {printf_types}')
        
        printf_blocks.append(f'              // === End Timestamp END ===')
    
    printf_section = '\n'.join(printf_blocks) if need_deferred_printf_section else ''
    
    # 移除了有問題的 clobber end 代碼（會導致 LLVM 生成 spill 代碼）
    # 快照機制已經保存了需要的暫存器值，不需要額外的 clobber/restore
    clobber_end = '              // === Printf Section End ===\n'
    
    # === 生成每個 @PRINT 的快照指令 ===
    # 建立 next_instruction -> snapshot_code_list 的映射（支援多個 @PRINT 共用同一位置）
    snapshot_insertions = {}  # {clean_instr: [snapshot_code1, snapshot_code2, ...]}
    for d_idx, directive in enumerate(directives):
        has_vgpr_snap = d_idx in directive_vgpr_snapshots
        has_sgpr_snap = d_idx in directive_sgpr_snapshots
        has_cond_snap = d_idx in directive_cond_snapshots
        has_expr_snap = d_idx in directive_expr_snapshots
        
        if directive.next_instruction and (has_vgpr_snap or has_sgpr_snap or has_cond_snap or has_expr_snap):
            # 清理 next_instruction 以便匹配
            clean_instr = directive.next_instruction.strip()
            if ';' in clean_instr:
                clean_instr = clean_instr.split(';')[0].strip()
            clean_instr = ' '.join(clean_instr.split())
            
            # 生成快照指令
            snap_lines = [f'              // === Snapshot for @PRINT at line {directive.line_number + 1} ===']
            
            if use_scratch_snapshot:
                # --- Scratch snapshot mode ---
                # VGPR/AccVGPR snapshots
                if has_vgpr_snap:
                    for r_idx, orig_reg, _snap_vgpr in directive_vgpr_snapshots[d_idx]:
                        s_off = scratch_vgpr_map.get((d_idx, r_idx))
                        if s_off is not None:
                            if orig_reg.startswith('a') and not orig_reg.startswith('acc'):
                                snap_lines.append(f'              // Scratch snapshot (AccVGPR): {orig_reg} -> scratch[{s_off}]')
                                snap_lines.append(
                                    f'              llvm.inline_asm has_side_effects "'
                                    f'scratch_store_dword off, v0, off offset:{SCRATCH_V0_SPILL_OFFSET}\\0A'
                                    f'v_accvgpr_read_b32 v0, {orig_reg}\\0A'
                                    f'scratch_store_dword off, v0, off offset:{s_off}\\0A'
                                    f'scratch_load_dword v0, off, off offset:{SCRATCH_V0_SPILL_OFFSET}\\0A'
                                    f's_waitcnt vmcnt(0)", ""  : () -> ()'
                                )
                            else:
                                snap_lines.append(f'              // Scratch snapshot: {orig_reg} -> scratch[{s_off}]')
                                snap_lines.append(f'              llvm.inline_asm has_side_effects "scratch_store_dword off, {orig_reg}, off offset:{s_off}", ""  : () -> ()')
                
                # SGPR snapshots: need v0 temp
                if has_sgpr_snap:
                    for r_idx, orig_reg, _snap_sgpr in directive_sgpr_snapshots[d_idx]:
                        s_off = scratch_sgpr_map.get((d_idx, r_idx))
                        if s_off is not None:
                            snap_lines.append(f'              // Scratch snapshot (SGPR): {orig_reg} -> scratch[{s_off}]')
                            snap_lines.append(
                                f'              llvm.inline_asm has_side_effects "'
                                f'scratch_store_dword off, v0, off offset:{SCRATCH_V0_SPILL_OFFSET}\\0A'
                                f'v_mov_b32 v0, {orig_reg}\\0A'
                                f'scratch_store_dword off, v0, off offset:{s_off}\\0A'
                                f'scratch_load_dword v0, off, off offset:{SCRATCH_V0_SPILL_OFFSET}\\0A'
                                f's_waitcnt vmcnt(0)", ""  : () -> ()'
                            )
                
                # Expression register snapshots
                if has_expr_snap and d_idx in scratch_expr_map:
                    for orig_reg, s_off in scratch_expr_map[d_idx].items():
                        snap_lines.append(f'              // Scratch expr snapshot: {orig_reg} -> scratch[{s_off}]')
                        if orig_reg.startswith('v'):
                            snap_lines.append(f'              llvm.inline_asm has_side_effects "scratch_store_dword off, {orig_reg}, off offset:{s_off}", ""  : () -> ()')
                        elif orig_reg.startswith('a') and not orig_reg.startswith('acc'):
                            snap_lines.append(
                                f'              llvm.inline_asm has_side_effects "'
                                f'scratch_store_dword off, v0, off offset:{SCRATCH_V0_SPILL_OFFSET}\\0A'
                                f'v_accvgpr_read_b32 v0, {orig_reg}\\0A'
                                f'scratch_store_dword off, v0, off offset:{s_off}\\0A'
                                f'scratch_load_dword v0, off, off offset:{SCRATCH_V0_SPILL_OFFSET}\\0A'
                                f's_waitcnt vmcnt(0)", ""  : () -> ()'
                            )
                        else:
                            snap_lines.append(
                                f'              llvm.inline_asm has_side_effects "'
                                f'scratch_store_dword off, v0, off offset:{SCRATCH_V0_SPILL_OFFSET}\\0A'
                                f'v_mov_b32 v0, {orig_reg}\\0A'
                                f'scratch_store_dword off, v0, off offset:{s_off}\\0A'
                                f'scratch_load_dword v0, off, off offset:{SCRATCH_V0_SPILL_OFFSET}\\0A'
                                f's_waitcnt vmcnt(0)", ""  : () -> ()'
                            )
                
                # Condition register snapshot
                if has_cond_snap and d_idx in scratch_cond_map:
                    cond_reg, _snap_num, _is_vgpr, _cond_type = directive_cond_snapshots[d_idx]
                    s_off = scratch_cond_map[d_idx]
                    snap_lines.append(f'              // Scratch cond snapshot: {cond_reg} -> scratch[{s_off}]')
                    if cond_reg.startswith('v'):
                        snap_lines.append(f'              llvm.inline_asm has_side_effects "scratch_store_dword off, {cond_reg}, off offset:{s_off}", ""  : () -> ()')
                    else:
                        snap_lines.append(
                            f'              llvm.inline_asm has_side_effects "'
                            f'scratch_store_dword off, v0, off offset:{SCRATCH_V0_SPILL_OFFSET}\\0A'
                            f'v_mov_b32 v0, {cond_reg}\\0A'
                            f'scratch_store_dword off, v0, off offset:{s_off}\\0A'
                            f'scratch_load_dword v0, off, off offset:{SCRATCH_V0_SPILL_OFFSET}\\0A'
                            f's_waitcnt vmcnt(0)", ""  : () -> ()'
                        )
            else:
                # --- Original VGPR snapshot mode ---
                # VGPR/AccVGPR 快照
                if has_vgpr_snap:
                    for r_idx, orig_reg, snap_vgpr in directive_vgpr_snapshots[d_idx]:
                        if orig_reg.startswith('a') and not orig_reg.startswith('acc'):
                            snap_lines.append(f'              llvm.inline_asm has_side_effects "v_accvgpr_read_b32 v{snap_vgpr}, {orig_reg}", ""  : () -> ()')
                        else:
                            snap_lines.append(f'              llvm.inline_asm has_side_effects "v_mov_b32 v{snap_vgpr}, {orig_reg}", ""  : () -> ()')
                
                # SGPR 快照：使用 s_mov_b32
                if has_sgpr_snap:
                    for r_idx, orig_reg, snap_sgpr in directive_sgpr_snapshots[d_idx]:
                        snap_lines.append(f'              llvm.inline_asm has_side_effects "s_mov_b32 s{snap_sgpr}, {orig_reg}", ""  : () -> ()')
                
                # 表達式暫存器快照
                if has_expr_snap:
                    for orig_reg, snap_reg in directive_expr_snapshots[d_idx].items():
                        if orig_reg.startswith('a') and not orig_reg.startswith('acc'):
                            snap_lines.append(f'              // Expression register snapshot (AccVGPR): {orig_reg} -> {snap_reg}')
                            snap_lines.append(f'              llvm.inline_asm has_side_effects "v_accvgpr_read_b32 {snap_reg}, {orig_reg}", ""  : () -> ()')
                        elif snap_reg.startswith('v'):
                            snap_lines.append(f'              // Expression register snapshot: {orig_reg} -> {snap_reg}')
                            snap_lines.append(f'              llvm.inline_asm has_side_effects "v_mov_b32 {snap_reg}, {orig_reg}", ""  : () -> ()')
                        else:
                            snap_lines.append(f'              // Expression register snapshot: {orig_reg} -> {snap_reg}')
                            snap_lines.append(f'              llvm.inline_asm has_side_effects "s_mov_b32 {snap_reg}, {orig_reg}", ""  : () -> ()')
                
                # 條件暫存器快照
                if has_cond_snap:
                    cond_reg, snap_num, is_vgpr, cond_type = directive_cond_snapshots[d_idx]
                    if is_vgpr:
                        snap_lines.append(f'              // Condition register snapshot: {cond_reg} -> v{snap_num}')
                        snap_lines.append(f'              llvm.inline_asm has_side_effects "v_mov_b32 v{snap_num}, {cond_reg}", ""  : () -> ()')
                    else:
                        snap_lines.append(f'              // Condition register snapshot: {cond_reg} -> s{snap_num}')
                        snap_lines.append(f'              llvm.inline_asm has_side_effects "s_mov_b32 s{snap_num}, {cond_reg}", ""  : () -> ()')
            
            snap_lines.append(f'              // === End Snapshot ===')

            # Trace capture: save per-iteration values into VGPR slots
            if d_idx in trace_buffers:
                trace_cfg = trace_buffers[d_idx]
                trace_max = trace_cfg['max']
                trace_counter = trace_cfg['counter']
                trace_regs = trace_cfg['regs']
                trace_slots = trace_cfg['slots']
                trace_tmps = trace_cfg['tmp']
                vcc_save_lo, vcc_save_hi = trace_cfg['vcc_save']

                # Build reg -> snapshot reg mapping for capture
                reg_snapshot_map = {}
                for _, orig_reg, snap_vgpr in directive_vgpr_snapshots.get(d_idx, []):
                    reg_snapshot_map[orig_reg] = f'v{snap_vgpr}'
                for _, orig_reg, snap_sgpr in directive_sgpr_snapshots.get(d_idx, []):
                    reg_snapshot_map[orig_reg] = f's{snap_sgpr}'
                if d_idx in directive_expr_snapshots:
                    for orig_reg, snap_reg in directive_expr_snapshots[d_idx].items():
                        reg_snapshot_map[orig_reg] = snap_reg

                snap_lines.append(f'              // === Trace capture for @PRINT #{d_idx+1} (max={trace_max}) ===')
                # Save vcc
                snap_lines.append(
                    f'              llvm.inline_asm has_side_effects "v_mov_b32 v{vcc_save_lo}, vcc_lo", ""  : () -> ()'
                )
                snap_lines.append(
                    f'              llvm.inline_asm has_side_effects "v_mov_b32 v{vcc_save_hi}, vcc_hi", ""  : () -> ()'
                )
                # Load current values into temp VGPRs
                for reg in trace_regs:
                    src_reg = reg_snapshot_map.get(reg, reg)
                    tmp_vgpr = trace_tmps[reg]
                    snap_lines.append(
                        f'              llvm.inline_asm has_side_effects "v_mov_b32 v{tmp_vgpr}, {src_reg}", ""  : () -> ()'
                    )
                # Conditional capture using vcc (no exec modification)
                for idx in range(trace_max):
                    snap_lines.append(
                        f'              llvm.inline_asm has_side_effects "v_cmp_eq_u32 vcc, v{trace_counter}, {idx}", ""  : () -> ()'
                    )
                    for reg in trace_regs:
                        dst_vgpr = trace_slots[reg][idx]
                        tmp_vgpr = trace_tmps[reg]
                        snap_lines.append(
                            f'              llvm.inline_asm has_side_effects "v_cndmask_b32 v{dst_vgpr}, v{dst_vgpr}, v{tmp_vgpr}, vcc", ""  : () -> ()'
                        )
                snap_lines.append(
                    f'              llvm.inline_asm has_side_effects "v_add_u32 v{trace_counter}, 1, v{trace_counter}", ""  : () -> ()'
                )
                # Restore vcc
                snap_lines.append(
                    f'              llvm.inline_asm has_side_effects "v_readfirstlane_b32 s{SGPR_KERNARG_BACKUP}, v{vcc_save_lo}", ""  : () -> ()'
                )
                snap_lines.append(
                    f'              llvm.inline_asm has_side_effects "v_readfirstlane_b32 s{SGPR_KERNARG_BACKUP+1}, v{vcc_save_hi}", ""  : () -> ()'
                )
                snap_lines.append(
                    f'              llvm.inline_asm has_side_effects "s_mov_b32 vcc_lo, s{SGPR_KERNARG_BACKUP}", ""  : () -> ()'
                )
                snap_lines.append(
                    f'              llvm.inline_asm has_side_effects "s_mov_b32 vcc_hi, s{SGPR_KERNARG_BACKUP+1}", ""  : () -> ()'
                )
                snap_lines.append(f'              // === End Trace capture ===')

            if inline_mode and not use_scratch_snapshot:
                vgpr_snaps = directive_vgpr_snapshots.get(d_idx, [])
                sgpr_snaps = directive_sgpr_snapshots.get(d_idx, [])
                cond_snap = directive_cond_snapshots.get(d_idx, None)
                expr_snap = directive_expr_snapshots.get(d_idx, None)
                tid_backup = WORKITEM_ID_BACKUP_VGPR if has_conditional_printf else None
                builtin_vars = {
                    'tid_vgpr': TID_BACKUP_VGPR,
                    'lane_vgpr': LANE_BACKUP_VGPR,
                    'tgid_x_sgpr': 2,
                    'tgid_y_sgpr': 3,
                    'tgid_z_sgpr': 4,
                }
                snap_lines.append('              // === Inline printf (save s0/s1, restore kernarg) ===')
                snap_lines.append(
                    f'              llvm.inline_asm has_side_effects "v_mov_b32 v{VGPR_INLINE_S0_SAVE_LO}, s0", ""  : () -> ()'
                )
                snap_lines.append(
                    f'              llvm.inline_asm has_side_effects "v_mov_b32 v{VGPR_INLINE_S0_SAVE_HI}, s1", ""  : () -> ()'
                )
                snap_lines.append(
                    f'              llvm.inline_asm has_side_effects "s_mov_b64 s[0:1], s[{SGPR_KERNARG_BACKUP}:{SGPR_KERNARG_BACKUP+1}]", ""  : () -> ()'
                )
                snap_lines.append(
                    generate_printf_with_snapshot(
                        directive, d_idx, vgpr_snaps, sgpr_snaps, tid_backup, cond_snap, expr_snap, builtin_vars
                    )
                )
                snap_lines.append(
                    f'              llvm.inline_asm has_side_effects "v_readfirstlane_b32 s0, v{VGPR_INLINE_S0_SAVE_LO}", ""  : () -> ()'
                )
                snap_lines.append(
                    f'              llvm.inline_asm has_side_effects "v_readfirstlane_b32 s1, v{VGPR_INLINE_S0_SAVE_HI}", ""  : () -> ()'
                )
                snap_lines.append('              // === End Inline printf ===')
            
            # 合併相同位置的快照（多個 @PRINT 可能共用同一個 next_instruction）
            if clean_instr not in snapshot_insertions:
                snapshot_insertions[clean_instr] = []
            snapshot_insertions[clean_instr].append('\n'.join(snap_lines))
            print(f"[Info] @PRINT #{d_idx+1} snapshot at: {clean_instr[:50]}...")
    
    # === 生成 @TIMESTAMP_START 的快照指令 ===
    # 在 @TIMESTAMP_START 位置注入：s_memrealtime + 備份到 VGPR
    # 對於 atomic 模式，額外執行 global_atomic_min_u64
    for label, ts_info in timestamp_vgpr_map.items():
        ts_start = ts_info['ts_start']
        start_lo = ts_info['start_lo']
        start_hi = ts_info['start_hi']
        ts_addr_lo = ts_info.get('ts_addr_lo')
        ts_addr_hi = ts_info.get('ts_addr_hi')
        
        if ts_start.next_instruction:
            clean_instr = ts_start.next_instruction.strip()
            if ';' in clean_instr:
                clean_instr = clean_instr.split(';')[0].strip()
            clean_instr = ' '.join(clean_instr.split())
            
            is_atomic = (ts_start.mode == "atomic")
            is_wallclock = (ts_start.mode == "wallclock")
            
            # 生成 timestamp start 指令
            ts_lines = [f'              // === Timestamp START [{label}] (mode={ts_start.mode}) at line {ts_start.line_number + 1} ===']
            # 選擇正確的時間戳指令：
            # - atomic/wallclock: s_memrealtime（跨 workgroup 同步，~100 MHz）
            # - local: s_memtime（高精度，~1.7 GHz，單 wavefront 內使用）
            if is_atomic or is_wallclock:
                ts_lines.append(f'              // Using s_memrealtime (globally synchronized, ~100 MHz)')
                ts_lines.append(f'              llvm.inline_asm has_side_effects "s_memrealtime s[20:21]\\0As_waitcnt lgkmcnt(0)", ""  : () -> ()')
            else:
                ts_lines.append(f'              // Using s_memtime (high precision, ~1.7 GHz)')
                ts_lines.append(f'              llvm.inline_asm has_side_effects "s_memtime s[20:21]\\0As_waitcnt lgkmcnt(0)", ""  : () -> ()')
            # 備份到 VGPR
            ts_lines.append(f'              llvm.inline_asm has_side_effects "v_mov_b32 v{start_lo}, s20", ""  : () -> ()')
            ts_lines.append(f'              llvm.inline_asm has_side_effects "v_mov_b32 v{start_hi}, s21", ""  : () -> ()')
            
            if is_atomic and ts_addr_lo is not None:
                # 完整 Atomic 模式：計算地址 + 執行 global_atomic_umin_x2 更新 min_start
                ts_lines.append(f'              // === Atomic: compute address and update min_start ===')
                # 計算 timestamp buffer 地址: C + N * 4
                ts_lines.append(f'              llvm.inline_asm has_side_effects "s_load_dwordx2 s[{SGPR_TS_BUFFER}:{SGPR_TS_BUFFER+1}], s[{SGPR_KERNARG_BACKUP}:{SGPR_KERNARG_BACKUP+1}], 0x10\\0As_waitcnt lgkmcnt(0)", ""  : () -> ()')  # C
                ts_lines.append(f'              llvm.inline_asm has_side_effects "s_load_dword s24, s[{SGPR_KERNARG_BACKUP}:{SGPR_KERNARG_BACKUP+1}], 0x18\\0As_waitcnt lgkmcnt(0)", ""  : () -> ()')  # N
                ts_lines.append(f'              llvm.inline_asm has_side_effects "s_lshl_b32 s24, s24, 2", ""  : () -> ()')  # N * 4
                ts_lines.append(f'              llvm.inline_asm has_side_effects "s_add_u32 s{SGPR_TS_BUFFER}, s{SGPR_TS_BUFFER}, s24", ""  : () -> ()')  # C + N*4
                ts_lines.append(f'              llvm.inline_asm has_side_effects "s_addc_u32 s{SGPR_TS_BUFFER+1}, s{SGPR_TS_BUFFER+1}, 0", ""  : () -> ()')
                ts_lines.append(f'              // Copy to VGPR for global addressing')
                ts_lines.append(f'              llvm.inline_asm has_side_effects "v_mov_b32 v{ts_addr_lo}, s{SGPR_TS_BUFFER}", ""  : () -> ()')
                ts_lines.append(f'              llvm.inline_asm has_side_effects "v_mov_b32 v{ts_addr_hi}, s{SGPR_TS_BUFFER+1}", ""  : () -> ()')
                # Atomic min operation
                ts_lines.append(f'              llvm.inline_asm has_side_effects "global_atomic_umin_x2 v[{ts_addr_lo}:{ts_addr_hi}], v[{start_lo}:{start_hi}], off", ""  : () -> ()')
            
            ts_lines.append(f'              // === End Timestamp START ===')
            
            if clean_instr not in snapshot_insertions:
                snapshot_insertions[clean_instr] = []
            snapshot_insertions[clean_instr].append('\n'.join(ts_lines))
            print(f"[Info] Timestamp START [{label}] at: {clean_instr[:50]}...")
    
    # === 生成 @TIMESTAMP_END 的 atomic 操作（僅限 atomic 模式）===
    # 注意：atomic 模式的 END 操作已移到 printf_blocks 中（在 printf 之前執行）
    # 這裡只打印信息，不再注入到 s_endpgm 位置
    for label, ts_info in timestamp_vgpr_map.items():
        ts_end = ts_info.get('ts_end')
        ts_addr_lo = ts_info.get('ts_addr_lo')
        
        if ts_end and ts_end.mode == "atomic" and ts_addr_lo is not None:
            # Atomic END 已移到 printf_blocks，這裡只保留注釋
            print(f"[Info] Timestamp END [{label}] (atomic) - injected in printf section for accurate timing")
    
    # === 注入邏輯 ===
    lines = gpumlir_text.split('\n')
    modified_lines = []
    in_func = False
    clobber_inserted = False
    printf_inserted = False
    inserted_snapshots = set()
    
    # 先掃描是否有 .LBB0_2: 標籤
    has_lbb0_2_label = any('.LBB0_2:' in line or '.LBB0_2"' in line for line in lines)
    
    for i, line in enumerate(lines):
        # 檢測是否進入 gpu.func
        if 'gpu.func @' in line and 'kernel' in line:
            in_func = True
        
        # 在第一個 llvm.inline_asm 之前插入 clobber_start
        if in_func and not clobber_inserted and 'llvm.inline_asm' in line:
            modified_lines.append(clobber_start)
            clobber_inserted = True
        
        # 檢查是否需要在這一行之前插入快照指令
        if in_func and 'llvm.inline_asm' in line:
            mlir_match = re.search(r'"([^"]*)"', line)
            if mlir_match:
                mlir_instr = mlir_match.group(1)
                mlir_instr = mlir_instr.lstrip('\\09').lstrip('\\t').strip()
                if ';' in mlir_instr:
                    mlir_instr = mlir_instr.split(';')[0].strip()
                mlir_instr = ' '.join(mlir_instr.split())
                
                # 檢查是否匹配任何 next_instruction
                for clean_instr, snap_code_list in snapshot_insertions.items():
                    if clean_instr not in inserted_snapshots:
                        if clean_instr == mlir_instr or clean_instr.startswith(mlir_instr) or mlir_instr.startswith(clean_instr):
                            # 插入所有共用這個位置的快照（可能有多個 @PRINT 共用）
                            for snap_code in snap_code_list:
                                modified_lines.append(snap_code)
                            inserted_snapshots.add(clean_instr)
        
        # 優先在 .LBB0_2: 標籤之前插入 printf
        is_lbb0_2_label = '.LBB0_2:' in line or ('.LBB0_2"' in line and 's_cbranch' not in line)
        if in_func and not printf_inserted and need_deferred_printf_section and has_lbb0_2_label and is_lbb0_2_label:
            modified_lines.append(printf_section)
            modified_lines.append(clobber_end)
            printf_inserted = True
        
        # 如果沒有 .LBB0_2: 標籤，則在 s_endpgm 之前插入
        if in_func and not printf_inserted and need_deferred_printf_section and not has_lbb0_2_label and 's_endpgm' in line:
            modified_lines.append(printf_section)
            modified_lines.append(clobber_end)
            printf_inserted = True
        
        modified_lines.append(line)
        
        # 離開 gpu.func
        if in_func and 'gpu.return' in line:
            in_func = False
    
    # 警告未插入的快照
    not_inserted = set(snapshot_insertions.keys()) - inserted_snapshots
    if not_inserted:
        print(f"[Warning] {len(not_inserted)} snapshot(s) could not be matched to MLIR instructions")
        for instr in not_inserted:
            print(f"          - {instr[:60]}...")
    
    # 計算所需的暫存器數量（用於後續 ISA metadata 修復）
    # 需要包含：快照 VGPR + workitem_id backup VGPR + 內建變數 VGPR
    # next_builtin_vgpr 已經計算了所有需要的 VGPR（包括 conditional、$tid、$lane）
    if use_scratch_snapshot:
        required_vgpr = original_vgpr  # No extra VGPRs needed in scratch mode
    else:
        required_vgpr = next_builtin_vgpr
    if use_scratch_snapshot:
        required_sgpr = SGPR_KERNARG_BACKUP + 2
    else:
        required_sgpr = max(snapshot_sgpr_start + total_snapshot_sgprs, SGPR_KERNARG_BACKUP + 2)
    
    # Calculate required private segment size for scratch snapshots
    required_private_size = 0
    if use_scratch_snapshot:
        # Align to 16 bytes
        required_private_size = ((scratch_offset_next + 15) // 16) * 16
        print(f"[Info] Scratch snapshot: required private segment size = {required_private_size} bytes")
    
    return '\n'.join(modified_lines), required_vgpr, required_sgpr, required_private_size


# ============================================================
# Pipeline 整合
# ============================================================

def run_cmd(cmd, cwd=None):
    """執行外部命令"""
    print("[$]", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, cwd=cwd)


def ensure_tool(name: str):
    """確認工具存在於 PATH 中"""
    if shutil.which(name) is None:
        raise RuntimeError(f"Required tool '{name}' not found in PATH")


def auto_detect_mlir_libs():
    """
    自動偵測 libmlir_rocm_runtime.so 和 libmlir_runner_utils.so
    """
    runner_path = shutil.which("mlir-runner") or shutil.which("mlir-opt")
    if not runner_path:
        return None, None

    bin_dir = pathlib.Path(runner_path).parent
    lib_dir = bin_dir.parent / "lib"

    rocm_rt = lib_dir / "libmlir_rocm_runtime.so"
    runner_utils = lib_dir / "libmlir_runner_utils.so"

    if rocm_rt.exists() and runner_utils.exists():
        return rocm_rt, runner_utils

    return None, None


def generate_host_wrapper_mlir(gpu_mlir_path: pathlib.Path, test_size: int, workdir: pathlib.Path, 
                                directives: List[PrintDirective] = None) -> pathlib.Path:
    """
    生成帶有 host wrapper 的完整 MLIR（包含 @main 函數）
    用於 mlir-runner 執行帶有 printf 的 kernel
    
    策略：使用 gpu.launch_func 搭配簡化的 kernel 簽名
    注意：kernel body 仍依賴原始 ISA 的 s[0:1] kernarg pointer 行為
    """
    import textwrap
    
    # 讀取 GPU MLIR
    gpu_mlir_text = gpu_mlir_path.read_text()
    
    # 提取 kernel 名稱
    kernel_match = re.search(r'gpu\.func\s+@([^\s(]+)\s*\(', gpu_mlir_text)
    if not kernel_match:
        raise RuntimeError("Cannot find kernel function in GPU MLIR")
    original_kernel_name = kernel_match.group(1)
    print(f"[Info] Found original kernel: {original_kernel_name}")
    print(f"[Info] Using gpu.launch_func with simplified kernel signature")
    
    # 提取 kernel body（從 attributes {} 後的 { 到 gpu.return 之前）
    def extract_kernel_body(text):
        """提取 kernel body 內容"""
        # 找到 gpu.func 的 body 開始位置
        func_start = text.find('gpu.func')
        if func_start == -1:
            return None
        
        # 找到 kernel attributes {...} 後的第二個 {
        # 格式: gpu.func @name(...) kernel attributes {...} { body }
        attr_start = text.find('kernel attributes', func_start)
        if attr_start == -1:
            # 沒有 attributes，直接找 kernel 後的 {
            kernel_pos = text.find('kernel', func_start)
            body_start = text.find('{', kernel_pos)
        else:
            # 有 attributes，找到 attributes {} 後的 {
            first_brace = text.find('{', attr_start)
            # 跳過 attributes 的 {}
            depth = 1
            pos = first_brace + 1
            while pos < len(text) and depth > 0:
                if text[pos] == '{':
                    depth += 1
                elif text[pos] == '}':
                    depth -= 1
                pos += 1
            # 找到 body 的 {
            body_start = text.find('{', pos)
        
        if body_start == -1:
            return None
        
        # 找到 gpu.return 的位置
        return_pos = text.find('gpu.return', body_start)
        if return_pos == -1:
            return None
        
        # 提取 body（不包括 gpu.return）
        body = text[body_start + 1:return_pos].strip()
        return body
    
    kernel_body = extract_kernel_body(gpu_mlir_text)
    if not kernel_body:
        raise RuntimeError("Cannot extract kernel body")
    
    # 移除 amdisa 特定的屬性（mlir-opt 不認識這些）
    kernel_body = re.sub(r'amdisa\.[a-z_]+\s*=\s*\d+\s*:\s*i32,?\s*', '', kernel_body)
    
    # 縮進 kernel body（用於 gpu.launch 內部）
    indented_body = '\n'.join('        ' + line for line in kernel_body.split('\n'))
    
    # 根據原始 kernel 名稱判斷參數類型
    # test_01: vectorAdd(A, B, C, n) - f32
    # test_02, test_05: scalarOps/loopKernel(output, n) - i32
    # test_03, test_04, test_06: memoryOps/conditionalKernel/sharedMemKernel(in, out, n) - i32
    is_float_kernel = 'vectorAdd' in original_kernel_name or 'float' in original_kernel_name.lower()
    is_single_output = 'scalarOps' in original_kernel_name or 'loopKernel' in original_kernel_name
    
    # 確定數據類型和參數數量
    data_type = "f32" if is_float_kernel else "i32"
    
    if is_single_output:
        param_count = 2
    elif is_float_kernel:
        param_count = 4
    else:
        param_count = 3
    
    print(f"[Info] Detected kernel type: {'float' if is_float_kernel else 'int'}, {param_count} params")
    
    # 生成完整的 MLIR 模組
    # 使用 gpu.launch_func，需要定義獨立的 gpu.func kernel
    # 注意：原始 ISA 代碼假設 s[0:1] 是 kernarg segment pointer
    
    module_name = "debug_module"
    kernel_name = "debug_kernel"
    
    if is_single_output:
        kernel_params_decl = "%Out: !llvm.ptr, %N: i32"
        kernel_args = "%Cptr : !llvm.ptr, %N_val : i32"
    elif is_float_kernel:
        kernel_params_decl = "%A: !llvm.ptr, %B: !llvm.ptr, %C: !llvm.ptr, %N: i32"
        kernel_args = "%Aptr : !llvm.ptr, %Bptr : !llvm.ptr, %Cptr : !llvm.ptr, %N_i32 : i32"
    else:
        kernel_params_decl = "%In: !llvm.ptr, %Out: !llvm.ptr, %N: i32"
        kernel_args = "%Aptr : !llvm.ptr, %Cptr : !llvm.ptr, %N_val : i32"
    
    if is_float_kernel:
        # Float kernel: A, B, C, n - 使用 f32 數組
        host_wrapper = textwrap.dedent(f'''
  // Host main function for testing with mlir-runner
  func.func @main() {{
    // Allocate arrays on host
    %A = memref.alloc() : memref<{test_size}xf32>
    %B = memref.alloc() : memref<{test_size}xf32>
    %C = memref.alloc() : memref<{test_size}xf32>
    
    // Initialize arrays: A[i] = i+1, B[i] = 2*(i+1), C[i] = 0
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cN = arith.constant {test_size} : index
    
    scf.for %i = %c0 to %cN step %c1 {{
      %fi = arith.index_cast %i : index to i32
      %fval = arith.sitofp %fi : i32 to f32
      %one = arith.constant 1.0 : f32
      %aval = arith.addf %fval, %one : f32
      memref.store %aval, %A[%i] : memref<{test_size}xf32>
      
      %two = arith.constant 2.0 : f32
      %bval = arith.mulf %aval, %two : f32
      memref.store %bval, %B[%i] : memref<{test_size}xf32>
      
      %zero = arith.constant 0.0 : f32
      memref.store %zero, %C[%i] : memref<{test_size}xf32>
    }}
    
    // Register for unified memory
    %castA = memref.cast %A : memref<{test_size}xf32> to memref<*xf32>
    %castB = memref.cast %B : memref<{test_size}xf32> to memref<*xf32>
    %castC = memref.cast %C : memref<{test_size}xf32> to memref<*xf32>
    gpu.host_register %castA : memref<*xf32>
    gpu.host_register %castB : memref<*xf32>
    gpu.host_register %castC : memref<*xf32>
    
    // Get raw pointers for kernel
    %Aptr_idx = memref.extract_aligned_pointer_as_index %A : memref<{test_size}xf32> -> index
    %Bptr_idx = memref.extract_aligned_pointer_as_index %B : memref<{test_size}xf32> -> index
    %Cptr_idx = memref.extract_aligned_pointer_as_index %C : memref<{test_size}xf32> -> index
    
    %Aptr_i64 = arith.index_cast %Aptr_idx : index to i64
    %Bptr_i64 = arith.index_cast %Bptr_idx : index to i64
    %Cptr_i64 = arith.index_cast %Cptr_idx : index to i64
    
    %Aptr = llvm.inttoptr %Aptr_i64 : i64 to !llvm.ptr
    %Bptr = llvm.inttoptr %Bptr_i64 : i64 to !llvm.ptr
    %Cptr = llvm.inttoptr %Cptr_i64 : i64 to !llvm.ptr
    
    %N_i32 = arith.constant {test_size} : i32
    
    // Launch kernel using gpu.launch_func
    %block_size = arith.constant 256 : index
    %grid_size = arith.ceildivui %cN, %block_size : index
    gpu.launch_func @{module_name}::@{kernel_name}
        blocks in (%grid_size, %c1, %c1)
        threads in (%block_size, %c1, %c1)
        args({kernel_args})
    
    // Print results
    call @printMemrefF32(%castC) : (memref<*xf32>) -> ()
    
    return
  }}
  
  func.func private @printMemrefF32(memref<*xf32>)
''')
    else:
        # Integer kernel: 使用 i32 數組
        host_wrapper = textwrap.dedent(f'''
// Host main function for testing with mlir-runner
func.func @main() {{
  // Allocate arrays on host (i32)
  %A = memref.alloc() : memref<{test_size}xi32>
  %C = memref.alloc() : memref<{test_size}xi32>
  
  // Initialize arrays: A[i] = i+1, C[i] = 0
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cN = arith.constant {test_size} : index
  
  scf.for %i = %c0 to %cN step %c1 {{
    %fi = arith.index_cast %i : index to i32
    %one = arith.constant 1 : i32
    %aval = arith.addi %fi, %one : i32
    memref.store %aval, %A[%i] : memref<{test_size}xi32>
    
    %zero = arith.constant 0 : i32
    memref.store %zero, %C[%i] : memref<{test_size}xi32>
  }}
  
  // Register for unified memory
  %castA = memref.cast %A : memref<{test_size}xi32> to memref<*xi32>
  %castC = memref.cast %C : memref<{test_size}xi32> to memref<*xi32>
  gpu.host_register %castA : memref<*xi32>
  gpu.host_register %castC : memref<*xi32>
  
  // Get raw pointers for kernel
  %Aptr_idx = memref.extract_aligned_pointer_as_index %A : memref<{test_size}xi32> -> index
  %Cptr_idx = memref.extract_aligned_pointer_as_index %C : memref<{test_size}xi32> -> index
  
  %Aptr_i64 = arith.index_cast %Aptr_idx : index to i64
  %Cptr_i64 = arith.index_cast %Cptr_idx : index to i64
  
  %Aptr = llvm.inttoptr %Aptr_i64 : i64 to !llvm.ptr
  %Cptr = llvm.inttoptr %Cptr_i64 : i64 to !llvm.ptr
  
  %N_val = arith.constant {test_size} : i32
  
  // Launch kernel
  %block_size = arith.constant 256 : index
  %grid_size = arith.ceildivui %cN, %block_size : index
  gpu.launch_func @{module_name}::@{kernel_name}
      blocks in (%grid_size, %c1, %c1)
      threads in (%block_size, %c1, %c1)
      args({kernel_args})
  
  // Print results
  call @printMemrefI32(%castC) : (memref<*xi32>) -> ()
  
  return
}}

func.func private @printMemrefI32(memref<*xi32>)
''')
    
    full_mlir = f'''module attributes {{gpu.container_module}} {{
  gpu.module @{module_name} {{
    gpu.func @{kernel_name}({kernel_params_decl}) kernel {{
{indented_body}
      gpu.return
    }}
  }}

{host_wrapper}
}}
'''
    
    # 保存
    output_path = workdir / f"{gpu_mlir_path.stem}_with_host.mlir"
    output_path.write_text(full_mlir)
    print(f"Generated host wrapper MLIR: {output_path}")
    
    return output_path


def run_with_mlir_runner(host_mlir_path: pathlib.Path, chip: str, workdir: pathlib.Path,
                         rocm_runtime_lib: str = None, runner_utils_lib: str = None):
    """
    使用 mlir-runner 執行帶有 host wrapper 的 MLIR
    """
    ensure_tool("mlir-opt")
    ensure_tool("mlir-runner")
    
    # 自動偵測 libs
    auto_rocm_rt, auto_runner_utils = auto_detect_mlir_libs()
    
    rocm_lib = pathlib.Path(rocm_runtime_lib) if rocm_runtime_lib else auto_rocm_rt
    runner_lib = pathlib.Path(runner_utils_lib) if runner_utils_lib else auto_runner_utils
    
    if not rocm_lib or not rocm_lib.exists():
        raise RuntimeError(
            "Cannot find libmlir_rocm_runtime.so. "
            "Please specify with --rocm-runtime-lib"
        )
    if not runner_lib or not runner_lib.exists():
        raise RuntimeError(
            "Cannot find libmlir_runner_utils.so. "
            "Please specify with --runner-utils-lib"
        )
    
    print(f"\n=== Running with mlir-runner ===")
    print(f"ROCm runtime lib: {rocm_lib}")
    print(f"Runner utils lib: {runner_lib}")
    
    # 使用官方 ROCM integration test 的 pipeline 順序
    step1 = workdir / f"{host_mlir_path.stem}_step1.mlir"
    step2 = workdir / f"{host_mlir_path.stem}_step2.mlir"
    step3 = workdir / f"{host_mlir_path.stem}_step3.mlir"
    final_mlir = workdir / f"{host_mlir_path.stem}_final.mlir"
    
    # Step 1: convert-scf-to-cf
    cmd1 = ["mlir-opt", str(host_mlir_path), "-convert-scf-to-cf", "-o", str(step1)]
    run_cmd(cmd1)
    
    # Step 2: gpu-kernel-outlining
    cmd2 = ["mlir-opt", str(step1), "-gpu-kernel-outlining", "-o", str(step2)]
    run_cmd(cmd2)
    
    # Step 3: GPU to ROCDL conversion (runtime=HIP is required for gpu.printf support)
    pipeline3 = f"builtin.module(gpu.module(strip-debuginfo,convert-gpu-to-rocdl{{index-bitwidth=32 runtime=HIP}}),rocdl-attach-target{{chip={chip}}})"
    cmd3 = ["mlir-opt", str(step2), f"--pass-pipeline={pipeline3}", "-o", str(step3)]
    run_cmd(cmd3)
    
    # Step 4: Final lowering
    cmd4 = [
        "mlir-opt", str(step3),
        "-gpu-to-llvm",
        "-reconcile-unrealized-casts",
        "-gpu-module-to-binary",
        "-o", str(final_mlir)
    ]
    run_cmd(cmd4)
    
    # Step 3: Run with mlir-runner
    cmd3 = [
        "mlir-runner",
        str(final_mlir),
        f"--shared-libs={rocm_lib}",
        f"--shared-libs={runner_lib}",
        "--entry-point-result=void"
    ]
    run_cmd(cmd3)


def translate_to_gpumlir(asm_path: pathlib.Path, workdir: pathlib.Path) -> pathlib.Path:
    """
    使用 amdisa-translate 將 .s 轉換為 GPU MLIR
    """
    ensure_tool("amdisa-translate")
    
    gpumlir_path = workdir / f"{asm_path.stem}_debug.gpumlir"
    
    print(f"\n=== Stage 1: Translating {asm_path.name} to GPU MLIR ===")
    
    # 直接使用 -emit=gpu 一步到位
    cmd = [
        "amdisa-translate",
        "-x", "s",
        "-emit=gpu",
        str(asm_path),
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    gpumlir_path.write_text(result.stdout)
    
    print(f"Generated GPU MLIR: {gpumlir_path}")
    return gpumlir_path


def rename_conflicting_labels(isa_text: str, original_isa_file: pathlib.Path) -> str:
    """
    重命名 MLIR pipeline 生成的標籤，避免與原始 ISA 中的標籤衝突
    
    MLIR pipeline 會生成 .LBB* 格式的標籤（用於 printf 的條件分支等），
    這些標籤可能會與原始 ISA 中已有的標籤重複。
    
    關鍵觀察：
    - 原始 ISA 的標籤在 ;;#ASMSTART / ;;#ASMEND 塊中（因為是 llvm.inline_asm 生成的）
    - printf 生成的標籤不在這些塊中
    
    策略：只重命名不在 ASMSTART/ASMEND 塊中的衝突標籤
    
    Args:
        isa_text: 需要修復的 ISA 文本
        original_isa_file: 原始 ISA 文件路徑
    
    Returns:
        修復後的 ISA 文本
    """
    if not original_isa_file.exists():
        return isa_text
    
    original_isa_text = original_isa_file.read_text()
    
    # 提取原始 ISA 中的所有 .LBB* 標籤
    original_labels = set(re.findall(r'\.LBB\d+_\d+', original_isa_text))
    
    if not original_labels:
        return isa_text  # 原始 ISA 沒有 .LBB 標籤，不需要重命名
    
    # 找出需要重命名的標籤：那些不在 ASMSTART/ASMEND 塊中的衝突標籤
    # 我們需要逐行處理，追蹤是否在 ASM 塊中
    lines = isa_text.split('\n')
    modified_lines = []
    in_asm_block = False
    renamed_count = 0
    
    for line in lines:
        if ';;#ASMSTART' in line:
            in_asm_block = True
            modified_lines.append(line)
            continue
        elif ';;#ASMEND' in line:
            in_asm_block = False
            modified_lines.append(line)
            continue
        
        if not in_asm_block:
            # 不在 ASM 塊中，檢查是否有衝突的標籤需要重命名
            modified_line = line
            for label in sorted(original_labels, key=lambda x: -len(x)):  # 從長到短避免部分替換
                if label in modified_line:
                    new_label = label.replace('.LBB', '.LBBPRINTF')
                    # 使用負向前瞻確保不會匹配到更長的標籤（如 .LBB0_60）
                    # 標籤後面要麼是冒號、空格、逗號、或行尾
                    pattern = rf'{re.escape(label)}(?![0-9_])'
                    old_line = modified_line
                    modified_line = re.sub(pattern, new_label, modified_line)
                    if modified_line != old_line:
                        renamed_count += 1
            modified_lines.append(modified_line)
        else:
            # 在 ASM 塊中，保持原樣
            modified_lines.append(line)
    
    if renamed_count > 0:
        print(f"[Info] Renamed {renamed_count} label references to avoid conflicts")
    
    return '\n'.join(modified_lines)


def fix_isa_metadata(isa_text: str, original_isa_file: pathlib.Path, has_printf: bool = True,
                     required_vgpr: int = 0, required_sgpr: int = 0,
                     required_private_size: int = 0) -> str:
    """
    修復 ISA metadata：從原始 ISA 文件提取正確的 metadata
    
    MLIR pipeline (convert-gpu-to-rocdl, gpu-module-to-binary) 會丟失原始 kernel 的參數信息，
    導致生成的 ISA metadata 只有 hidden_* 參數，缺少實際的 kernel args (A, B, C, n 等)。
    
    重要：如果有 printf 注入，需要保留 MLIR pipeline 生成的 hidden_hostcall_buffer 參數！
    否則 printf 會因為找不到 hostcall buffer 而導致 illegal memory access。
    
    Args:
        isa_text: 需要修復的 ISA 文本
        original_isa_file: 原始 ISA 文件路徑
        has_printf: 是否有 printf 注入（影響 args 合併策略）
        required_vgpr: printf 快照所需的 VGPR 數量（將更新 .amdhsa_next_free_vgpr）
        required_sgpr: printf 快照所需的 SGPR 數量（將更新 .amdhsa_next_free_sgpr）
        required_private_size: scratch snapshot 所需的 private segment 大小（bytes）
    
    Returns:
        修復後的 ISA 文本
    """
    if not original_isa_file.exists():
        print(f"[Warning] Original ISA file not found: {original_isa_file}")
        return isa_text
    
    print(f"\n=== Fixing ISA metadata from original: {original_isa_file.name} ===")
    original_isa_text = original_isa_file.read_text()
    
    attrs = {}
    original_non_hidden_args = []  # 只保存非 hidden 參數
    original_all_args = []  # 保存所有參數（用於無 printf 情況）
    computed_kernarg_size = None

    def compute_kernarg_size(arg_list):
        max_end = 0
        for arg in arg_list:
            offset = arg.get('.offset', arg.get('offset'))
            size = arg.get('.size', arg.get('size', 0))
            if offset is None:
                continue
            try:
                end = int(offset) + int(size)
            except Exception:
                continue
            if end > max_end:
                max_end = end
        # Align to 8 bytes for kernarg segment
        if max_end % 8 != 0:
            max_end = ((max_end + 7) // 8) * 8
        return max_end

    def compute_user_sgpr_count(attr_map):
        count = 0
        if attr_map.get('dispatch_ptr', 0):
            count += 2
        if attr_map.get('queue_ptr', 0):
            count += 2
        if attr_map.get('kernarg_segment_ptr', 0):
            count += 2
        if attr_map.get('dispatch_id', 0):
            count += 2
        if attr_map.get('private_segment_size', 0):
            count += 1
        return count
    
    # 提取 .amdhsa_* directives
    amdhsa_patterns = {
        'kernarg_segment_size': r'\.amdhsa_kernarg_size\s+(\d+)',
        'group_segment_fixed_size': r'\.amdhsa_group_segment_fixed_size\s+(\d+)',
        'user_sgpr_count': r'\.amdhsa_user_sgpr_count\s+(\d+)',
        'dispatch_ptr': r'\.amdhsa_user_sgpr_dispatch_ptr\s+(\d+)',
        'queue_ptr': r'\.amdhsa_user_sgpr_queue_ptr\s+(\d+)',
        'kernarg_segment_ptr': r'\.amdhsa_user_sgpr_kernarg_segment_ptr\s+(\d+)',  # 關鍵！
        'private_segment_size': r'\.amdhsa_user_sgpr_private_segment_size\s+(\d+)',
        'workitem_id': r'\.amdhsa_system_vgpr_workitem_id\s+(\d+)',
        'workgroup_id_x': r'\.amdhsa_system_sgpr_workgroup_id_x\s+(\d+)',
        'accum_offset': r'\.amdhsa_accum_offset\s+(\d+)',  # AGPR offset - 重要
        'reserve_vcc': r'\.amdhsa_reserve_vcc\s+(\d+)',  # VCC 保留 - kernel 使用 vcc 時需要
    }
    
    for attr_name, pattern in amdhsa_patterns.items():
        match = re.search(pattern, original_isa_text)
        if match:
            attrs[attr_name] = int(match.group(1))
    
    # 提取 YAML metadata 中的 args
    yaml_match = re.search(r'\.amdgpu_metadata\s+---\s+(.*?)\.\.\.\s+\.end_amdgpu_metadata', 
                           original_isa_text, re.DOTALL)
    if yaml_match:
        try:
            yaml_text = yaml_match.group(1)
            metadata = yaml.safe_load(yaml_text)
            
            if 'amdhsa.kernels' in metadata and len(metadata['amdhsa.kernels']) > 0:
                kernel = metadata['amdhsa.kernels'][0]
                
                # 提取 args
                if '.args' in kernel and isinstance(kernel['.args'], list):
                    for arg in kernel['.args']:
                        arg_dict = {}
                        for k, v in arg.items():
                            key_name = k[1:] if k.startswith('.') else k
                            arg_dict[key_name] = v
                        
                        # 保存所有參數
                        original_all_args.append(arg_dict)
                        
                        # 只保存非 hidden 參數
                        if 'value_kind' in arg_dict and not arg_dict['value_kind'].startswith('hidden_'):
                            original_non_hidden_args.append(arg_dict)
            
            print(f"[Info] Extracted {len(attrs)} metadata attributes and {len(original_non_hidden_args)} non-hidden args from original ISA")
            
        except Exception as e:
            print(f"[Warning] Failed to parse original ISA YAML metadata: {e}")
    
    if not original_non_hidden_args and not attrs:
        print("[Warning] No metadata found in original ISA")
        return isa_text
    
    # 找到生成的 ISA 中的 .amdgpu_metadata 部分
    metadata_start = isa_text.find('.amdgpu_metadata')
    metadata_end = isa_text.find('.end_amdgpu_metadata')
    
    if metadata_start == -1 or metadata_end == -1:
        print("[Warning] No .amdgpu_metadata section found in generated ISA")
        return isa_text
    
    # 提取 YAML metadata
    yaml_start = isa_text.find('---', metadata_start)
    yaml_end = isa_text.find('...', yaml_start)
    
    if yaml_start == -1 or yaml_end == -1:
        print("[Warning] Invalid YAML in .amdgpu_metadata")
        return isa_text
    
    yaml_text = isa_text[yaml_start+3:yaml_end].strip()
    
    try:
        gen_metadata = yaml.safe_load(yaml_text)
    except Exception as e:
        print(f"[Warning] Failed to parse generated ISA metadata YAML: {e}")
        return isa_text
    
    # 修復 metadata
    if 'amdhsa.kernels' in gen_metadata and len(gen_metadata['amdhsa.kernels']) > 0:
        kernel = gen_metadata['amdhsa.kernels'][0]
        
        # 修復 kernarg_segment_size
        if 'kernarg_segment_size' in attrs:
            kernel['.kernarg_segment_size'] = attrs['kernarg_segment_size']
        
        # 修復 group_segment_fixed_size (LDS / shared memory 大小)
        if 'group_segment_fixed_size' in attrs:
            kernel['.group_segment_fixed_size'] = attrs['group_segment_fixed_size']
        
        # === 關鍵修復：合併 args ===
        if has_printf and original_non_hidden_args:
            # 有 printf：保留 MLIR pipeline 生成的 hidden 參數（特別是 hidden_hostcall_buffer）
            # 只替換前面的非 hidden 參數
            gen_args = kernel.get('.args', [])
            
            # 找出生成的 args 中 hidden 參數的起始位置
            hidden_start_idx = len(gen_args)
            for i, arg in enumerate(gen_args):
                value_kind = arg.get('.value_kind', '')
                if value_kind.startswith('hidden_'):
                    hidden_start_idx = i
                    break
            
            # 構建新的 args 列表：原始非 hidden + 生成的 hidden
            new_args = []
            for arg in original_non_hidden_args:
                yaml_arg = {}
                if 'address_space' in arg:
                    yaml_arg['.address_space'] = arg['address_space']
                if 'offset' in arg:
                    yaml_arg['.offset'] = arg['offset']
                if 'size' in arg:
                    yaml_arg['.size'] = arg['size']
                if 'value_kind' in arg:
                    yaml_arg['.value_kind'] = arg['value_kind']
                if 'name' in arg:
                    yaml_arg['.name'] = arg['name']
                new_args.append(yaml_arg)
            
            # 添加生成的 hidden 參數（包含 hidden_hostcall_buffer）
            hidden_args = gen_args[hidden_start_idx:]
            if hidden_args:
                base_offset = compute_kernarg_size(original_non_hidden_args)
                hidden_offsets = [arg.get('.offset') for arg in hidden_args if arg.get('.offset') is not None]
                if hidden_offsets:
                    shift = base_offset - min(hidden_offsets)
                    if shift:
                        for arg in hidden_args:
                            if arg.get('.offset') is not None:
                                arg['.offset'] = int(arg['.offset']) + shift
                        print(f"[Info] Shifted hidden args by {shift} (base_offset={base_offset})")
            new_args.extend(hidden_args)
            
            kernel['.args'] = new_args
            computed_kernarg_size = compute_kernarg_size(new_args)
            if computed_kernarg_size:
                kernel['.kernarg_segment_size'] = computed_kernarg_size
                print(f"[Info] Recomputed kernarg_segment_size: {computed_kernarg_size}")
            
            # 檢查是否有 hidden_hostcall_buffer
            has_hostcall = any(arg.get('.value_kind') == 'hidden_hostcall_buffer' for arg in new_args)
            print(f"[Info] Merged {len(original_non_hidden_args)} original args + {len(gen_args) - hidden_start_idx} hidden args")
            if has_hostcall:
                print(f"[Info] ✓ hidden_hostcall_buffer preserved for printf support")
            else:
                print(f"[Warning] hidden_hostcall_buffer NOT found - printf may fail!")
        
        elif original_all_args:
            # 無 printf：恢復原始 args 並添加必要的 hidden 參數（用於多 workgroup 支持）
            kernel['.args'] = []
            max_offset = 0
            for arg in original_all_args:
                yaml_arg = {}
                if 'address_space' in arg:
                    yaml_arg['.address_space'] = arg['address_space']
                if 'offset' in arg:
                    yaml_arg['.offset'] = arg['offset']
                    # 追蹤最大 offset 以便添加 hidden args
                    arg_end = arg['offset'] + arg.get('size', 8)
                    max_offset = max(max_offset, arg_end)
                if 'size' in arg:
                    yaml_arg['.size'] = arg['size']
                if 'value_kind' in arg:
                    yaml_arg['.value_kind'] = arg['value_kind']
                if 'name' in arg:
                    yaml_arg['.name'] = arg['name']
                kernel['.args'].append(yaml_arg)
            
            # 添加標準的 hidden 參數（對多 workgroup kernel 執行是必要的）
            # 這些參數是 HIP runtime 用來傳遞 grid/block 配置的
            # 對齊到 8 bytes
            hidden_offset = ((max_offset + 7) // 8) * 8
            hidden_args = [
                {'.offset': hidden_offset, '.size': 4, '.value_kind': 'hidden_block_count_x'},
                {'.offset': hidden_offset + 4, '.size': 4, '.value_kind': 'hidden_block_count_y'},
                {'.offset': hidden_offset + 8, '.size': 4, '.value_kind': 'hidden_block_count_z'},
                {'.offset': hidden_offset + 12, '.size': 2, '.value_kind': 'hidden_group_size_x'},
                {'.offset': hidden_offset + 14, '.size': 2, '.value_kind': 'hidden_group_size_y'},
                {'.offset': hidden_offset + 16, '.size': 2, '.value_kind': 'hidden_group_size_z'},
                {'.offset': hidden_offset + 18, '.size': 2, '.value_kind': 'hidden_remainder_x'},
                {'.offset': hidden_offset + 20, '.size': 2, '.value_kind': 'hidden_remainder_y'},
                {'.offset': hidden_offset + 22, '.size': 2, '.value_kind': 'hidden_remainder_z'},
                # 跳到 offset 40 (對齊到 8 bytes 後開始 64-bit 值)
                {'.offset': hidden_offset + 40, '.size': 8, '.value_kind': 'hidden_global_offset_x'},
                {'.offset': hidden_offset + 48, '.size': 8, '.value_kind': 'hidden_global_offset_y'},
                {'.offset': hidden_offset + 56, '.size': 8, '.value_kind': 'hidden_global_offset_z'},
                {'.offset': hidden_offset + 64, '.size': 8, '.value_kind': 'hidden_printf_buffer'},
                {'.offset': hidden_offset + 80, '.size': 8, '.value_kind': 'hidden_default_queue'},
                {'.offset': hidden_offset + 88, '.size': 8, '.value_kind': 'hidden_completion_action'},
                {'.offset': hidden_offset + 104, '.size': 4, '.value_kind': 'hidden_grid_dims'},
            ]
            kernel['.args'].extend(hidden_args)
            computed_kernarg_size = compute_kernarg_size(kernel['.args'])
            if computed_kernarg_size:
                kernel['.kernarg_segment_size'] = computed_kernarg_size
                print(f"[Info] Recomputed kernarg_segment_size: {computed_kernarg_size}")
            print(f"[Info] Restored {len(original_all_args)} original kernel arguments + {len(hidden_args)} hidden args for multi-workgroup support")
    
    # 重新生成 YAML
    fixed_yaml = yaml.dump(gen_metadata, default_flow_style=False, sort_keys=False)
    
    # 替換 ISA 中的 metadata
    before_metadata = isa_text[:yaml_start]
    after_metadata = isa_text[yaml_end:]
    
    fixed_isa = before_metadata + "---\n" + fixed_yaml + "...\n" + after_metadata
    
    # 同時修復 .amdhsa_* 指令
    if computed_kernarg_size:
        if re.search(r'\.amdhsa_kernarg_size\s+\d+', fixed_isa):
            fixed_isa = re.sub(
                r'(\.amdhsa_kernarg_size)\s+\d+',
                rf'\1 {computed_kernarg_size}',
                fixed_isa
            )
            print(f"[Info] Fixed .amdhsa_kernarg_size: {computed_kernarg_size}")
        else:
            fixed_isa = re.sub(
                r'(\.amdhsa_private_segment_fixed_size\s+\d+)',
                rf'\1\n\t\t.amdhsa_kernarg_size {computed_kernarg_size}',
                fixed_isa,
                count=1
            )
            print(f"[Info] Inserted .amdhsa_kernarg_size: {computed_kernarg_size}")

    if computed_kernarg_size is None and 'kernarg_segment_size' in attrs:
        fixed_isa = re.sub(
            r'(\.amdhsa_kernarg_size)\s+\d+',
            rf'\1 {attrs["kernarg_segment_size"]}',
            fixed_isa
        )
    
    if 'group_segment_fixed_size' in attrs:
        fixed_isa = re.sub(
            r'(\.amdhsa_group_segment_fixed_size)\s+\d+',
            rf'\1 {attrs["group_segment_fixed_size"]}',
            fixed_isa
        )
    
    user_sgpr_count = attrs.get('user_sgpr_count', 0)
    if user_sgpr_count <= 0:
        user_sgpr_count = compute_user_sgpr_count(attrs)
    if user_sgpr_count > 0:
        fixed_isa = re.sub(
            r'(\.amdhsa_user_sgpr_count)\s+\d+',
            rf'\1 {user_sgpr_count}',
            fixed_isa
        )
    
    # 修復 kernarg_segment_ptr - 關鍵！沒有這個 kernel 無法讀取參數
    if 'kernarg_segment_ptr' in attrs:
        fixed_isa = re.sub(
            r'(\.amdhsa_user_sgpr_kernarg_segment_ptr)\s+\d+',
            rf'\1 {attrs["kernarg_segment_ptr"]}',
            fixed_isa
        )
        print(f"[Info] Fixed kernarg_segment_ptr = {attrs['kernarg_segment_ptr']}")
    
    # 修復 reserve_vcc - kernel 使用 vcc 時必須設為 1
    if 'reserve_vcc' in attrs:
        fixed_isa = re.sub(
            r'(\.amdhsa_reserve_vcc)\s+\d+',
            rf'\1 {attrs["reserve_vcc"]}',
            fixed_isa
        )
        print(f"[Info] Fixed reserve_vcc = {attrs['reserve_vcc']}")
    
    # === 修復快照暫存器分配 ===
    # printf 快照使用的 VGPR/SGPR 可能超出 MLIR 生成的範圍，需要更新 .amdhsa_next_free_vgpr/sgpr
    if required_vgpr > 0:
        # 找出目前的 next_free_vgpr 值
        vgpr_match = re.search(r'\.amdhsa_next_free_vgpr\s+(\d+)', fixed_isa)
        if vgpr_match:
            current_vgpr = int(vgpr_match.group(1))
            if required_vgpr > current_vgpr:
                fixed_isa = re.sub(
                    r'(\.amdhsa_next_free_vgpr)\s+\d+',
                    rf'\1 {required_vgpr}',
                    fixed_isa
                )
                print(f"[Info] Fixed next_free_vgpr: {current_vgpr} -> {required_vgpr} (for snapshot registers)")
        
        # 修復 accum_offset - 必須 >= next_free_vgpr 以避免 VGPR/AGPR 衝突
        accum_match = re.search(r'\.amdhsa_accum_offset\s+(\d+)', fixed_isa)
        if accum_match:
            current_accum = int(accum_match.group(1))
            # accum_offset 必須是 4 的倍數，且 >= required_vgpr
            required_accum = ((required_vgpr + 3) // 4) * 4  # 向上取整到 4 的倍數
            if required_accum > 256:
                required_accum = 256
            if required_accum > current_accum:
                fixed_isa = re.sub(
                    r'(\.amdhsa_accum_offset)\s+\d+',
                    rf'\1 {required_accum}',
                    fixed_isa
                )
                print(f"[Info] Fixed accum_offset: {current_accum} -> {required_accum}")
    
    if required_sgpr > 0:
        # 找出目前的 next_free_sgpr 值
        sgpr_match = re.search(r'\.amdhsa_next_free_sgpr\s+(\d+)', fixed_isa)
        if sgpr_match:
            current_sgpr = int(sgpr_match.group(1))
            if required_sgpr > current_sgpr:
                fixed_isa = re.sub(
                    r'(\.amdhsa_next_free_sgpr)\s+\d+',
                    rf'\1 {required_sgpr}',
                    fixed_isa
                )
                print(f"[Info] Fixed next_free_sgpr: {current_sgpr} -> {required_sgpr} (for snapshot registers)")
    
    # === 修復 YAML metadata 的 .sgpr_count 和 .vgpr_count ===
    # MLIR 生成的 YAML metadata 可能不正確（特別是當沒有 printf 時）
    # 需要根據原始 kernel 的值和 snapshot 需求來修正
    
    # 先從原始 ISA 提取 vgpr_count 和 sgpr_count
    original_vgpr_count = 0
    original_sgpr_count = 0
    original_vgpr_match = re.search(r'\.vgpr_count:\s*(\d+)', original_isa_text)
    original_sgpr_match = re.search(r'\.sgpr_count:\s*(\d+)', original_isa_text)
    original_next_free_vgpr_match = re.search(r'\.amdhsa_next_free_vgpr\s+(\d+)', original_isa_text)
    original_next_free_sgpr_match = re.search(r'\.amdhsa_next_free_sgpr\s+(\d+)', original_isa_text)
    
    if original_vgpr_match:
        original_vgpr_count = int(original_vgpr_match.group(1))
    if original_sgpr_match:
        original_sgpr_count = int(original_sgpr_match.group(1))
    if original_next_free_vgpr_match:
        original_next_free_vgpr = int(original_next_free_vgpr_match.group(1))
        original_vgpr_count = max(original_vgpr_count, original_next_free_vgpr)
    if original_next_free_sgpr_match:
        original_next_free_sgpr = int(original_next_free_sgpr_match.group(1))
        original_sgpr_count = max(original_sgpr_count, original_next_free_sgpr)
    
    # 計算需要的 vgpr/sgpr：取原始值和 required 的最大值
    needed_vgpr = max(required_vgpr, original_vgpr_count)
    needed_sgpr = max(required_sgpr, original_sgpr_count)
    
    if needed_vgpr > 0:
        yaml_vgpr_match = re.search(r'\.vgpr_count:\s*(\d+)', fixed_isa)
        if yaml_vgpr_match:
            current_yaml_vgpr = int(yaml_vgpr_match.group(1))
            if needed_vgpr > current_yaml_vgpr:
                fixed_isa = re.sub(
                    r'(\.vgpr_count:)\s*\d+',
                    rf'\1 {needed_vgpr}',
                    fixed_isa
                )
                print(f"[Info] Fixed YAML .vgpr_count: {current_yaml_vgpr} -> {needed_vgpr} (original={original_vgpr_count}, required={required_vgpr})")
        else:
            # 如果沒有 .vgpr_count 條目，在 .sgpr_count 後面添加
            fixed_isa = re.sub(
                r'(\.sgpr_count:\s*\d+)',
                rf'\1\n  .vgpr_count: {needed_vgpr}',
                fixed_isa
            )
            print(f"[Info] Added YAML .vgpr_count: {needed_vgpr}")
        
        # 也修復 .amdhsa_next_free_vgpr
        vgpr_directive_match = re.search(r'\.amdhsa_next_free_vgpr\s+(\d+)', fixed_isa)
        if vgpr_directive_match:
            current_directive_vgpr = int(vgpr_directive_match.group(1))
            if needed_vgpr > current_directive_vgpr:
                fixed_isa = re.sub(
                    r'(\.amdhsa_next_free_vgpr)\s+\d+',
                    rf'\1 {needed_vgpr}',
                    fixed_isa
                )
                print(f"[Info] Fixed .amdhsa_next_free_vgpr: {current_directive_vgpr} -> {needed_vgpr}")
        
        # 修復 accum_offset - 必須 >= next_free_vgpr 以避免 VGPR/AGPR 衝突
        # 獲取原始 accum_offset
        original_accum = attrs.get('accum_offset', needed_vgpr)
        accum_match = re.search(r'\.amdhsa_accum_offset\s+(\d+)', fixed_isa)
        if accum_match:
            current_accum = int(accum_match.group(1))
            # accum_offset 必須是 4 的倍數，且 >= needed_vgpr，且 >= 原始值
            required_accum = ((max(needed_vgpr, original_accum) + 3) // 4) * 4
            if required_accum > 256:
                required_accum = 256
            if required_accum > current_accum:
                fixed_isa = re.sub(
                    r'(\.amdhsa_accum_offset)\s+\d+',
                    rf'\1 {required_accum}',
                    fixed_isa
                )
                print(f"[Info] Fixed accum_offset: {current_accum} -> {required_accum} (original={original_accum})")
    
    if needed_sgpr > 0:
        yaml_sgpr_match = re.search(r'\.sgpr_count:\s*(\d+)', fixed_isa)
        if yaml_sgpr_match:
            current_yaml_sgpr = int(yaml_sgpr_match.group(1))
            if needed_sgpr > current_yaml_sgpr:
                fixed_isa = re.sub(
                    r'(\.sgpr_count:)\s*\d+',
                    rf'\1 {needed_sgpr}',
                    fixed_isa
                )
                print(f"[Info] Fixed YAML .sgpr_count: {current_yaml_sgpr} -> {needed_sgpr} (original={original_sgpr_count}, required={required_sgpr})")
        
        # 也修復 .amdhsa_next_free_sgpr
        sgpr_directive_match = re.search(r'\.amdhsa_next_free_sgpr\s+(\d+)', fixed_isa)
        if sgpr_directive_match:
            current_directive_sgpr = int(sgpr_directive_match.group(1))
            if needed_sgpr > current_directive_sgpr:
                fixed_isa = re.sub(
                    r'(\.amdhsa_next_free_sgpr)\s+\d+',
                    rf'\1 {needed_sgpr}',
                    fixed_isa
                )
                print(f"[Info] Fixed .amdhsa_next_free_sgpr: {current_directive_sgpr} -> {needed_sgpr}")
    
    # Ensure .amdhsa_private_segment_fixed_size is large enough for scratch snapshots
    if required_private_size > 0:
        priv_match = re.search(r'\.amdhsa_private_segment_fixed_size\s+(\d+)', fixed_isa)
        if priv_match:
            current_priv = int(priv_match.group(1))
            needed_priv = max(current_priv, required_private_size)
            if needed_priv > current_priv:
                fixed_isa = re.sub(
                    r'(\.amdhsa_private_segment_fixed_size)\s+\d+',
                    rf'\1 {needed_priv}',
                    fixed_isa
                )
                print(f"[Info] Fixed .amdhsa_private_segment_fixed_size: {current_priv} -> {needed_priv}")
        
        # Also ensure enable_private_segment is 1
        enable_match = re.search(r'\.amdhsa_enable_private_segment\s+(\d+)', fixed_isa)
        if enable_match and int(enable_match.group(1)) == 0:
            fixed_isa = re.sub(
                r'(\.amdhsa_enable_private_segment)\s+\d+',
                rf'\1 1',
                fixed_isa
            )
            print(f"[Info] Enabled .amdhsa_enable_private_segment for scratch snapshots")
    
    return fixed_isa


def build_debug_hsaco(gpumlir_path: pathlib.Path, chip: str, workdir: pathlib.Path, 
                      original_isa_file: pathlib.Path = None, has_printf: bool = False,
                      required_vgpr: int = 0, required_sgpr: int = 0,
                      required_private_size: int = 0):
    """
    從修改後的 GPU MLIR 生成 HSACO
    
    Args:
        gpumlir_path: 修改後的 GPU MLIR 文件
        chip: 目標 GPU 架構
        workdir: 工作目錄
        original_isa_file: 原始 ISA 文件路徑（用於提取 metadata）
        has_printf: 是否有 printf 注入（影響 metadata 合併策略）
        required_vgpr: printf 快照所需的 VGPR 數量
        required_sgpr: printf 快照所需的 SGPR 數量
        required_private_size: scratch snapshot 所需的 private segment 大小（bytes）
    """
    for tool in ["mlir-opt", "llvm-mc", "lld"]:
        ensure_tool(tool)
    
    stem = gpumlir_path.stem
    binary_mlir = workdir / f"{stem}_binary.mlir"
    isa_path = workdir / f"{stem}.s"
    obj_path = workdir / f"{stem}.o"
    hsaco_path = workdir / f"{stem}.hsaco"
    
    print(f"\n=== Stage 2: Running MLIR optimization pipeline ===")
    
    # 完整的 pipeline
    # 注意：convert-gpu-to-rocdl 需要在 convert-scf-to-cf 之後
    # gpu-to-llvm 會自動處理 arith 和 cf 操作的轉換
    pipeline = (
        f"builtin.module("
        f"gpu-kernel-outlining,"
        f"rocdl-attach-target{{chip={chip}}},"
        f"gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl{{index-bitwidth=32 runtime=HIP}}),"
        f"gpu-to-llvm,"
        f"gpu-module-to-binary{{format=isa}}"
        f")"
    )
    
    cmd = [
        "mlir-opt",
        str(gpumlir_path),
        f"--pass-pipeline={pipeline}",
        "-o",
        str(binary_mlir),
    ]
    run_cmd(cmd)
    
    # 提取 ISA
    binary_text = binary_mlir.read_text()
    
    # 解碼 ISA 字串
    asm_match = re.search(
        r'gpu\.binary\b.*?assembly\s*=\s*"((?:[^"\\]|\\.)*)"',
        binary_text, re.DOTALL
    )
    
    if not asm_match:
        raise RuntimeError("No gpu.binary assembly found in MLIR output")
    
    def decode_mlir_string(raw: str) -> str:
        def _hex_repl(m):
            return chr(int(m.group(1), 16))
        s = re.sub(r"\\([0-9A-Fa-f]{2})", _hex_repl, raw)
        return bytes(s, "utf-8").decode("unicode_escape")
    
    isa_text = decode_mlir_string(asm_match.group(1))
    
    # 修復 ISA metadata（從原始 ISA 提取 kernel args 等）
    # 重要：如果有 printf，需要保留 hidden_hostcall_buffer
    if original_isa_file is not None:
        isa_text = fix_isa_metadata(isa_text, original_isa_file, has_printf, required_vgpr, required_sgpr,
                                    required_private_size)
        # 重命名衝突的標籤（printf 生成的 .LBB* 與原始 ISA 衝突）
        isa_text = rename_conflicting_labels(isa_text, original_isa_file)
    
    isa_path.write_text(isa_text)
    print(f"Generated ISA: {isa_path}")
    
    print(f"\n=== Stage 3: Assembling ISA to object file ===")
    cmd = [
        "llvm-mc",
        "-triple", "amdgcn-amd-amdhsa",
        f"-mcpu={chip}",
        "-filetype=obj",
        str(isa_path),
        "-o",
        str(obj_path),
    ]
    run_cmd(cmd)
    
    print(f"\n=== Stage 4: Linking to HSACO ===")
    cmd = [
        "lld",
        "-flavor", "gnu",
        "-m", "elf64_amdgpu",
        "--no-undefined",
        "-shared",
        "-plugin-opt=-amdgpu-internalize-symbols",
        f"-plugin-opt=mcpu={chip}",
        "--whole-archive",
        str(obj_path),
        "--no-whole-archive",
        "-o",
        str(hsaco_path),
    ]
    run_cmd(cmd)
    
    print(f"\n✓ Successfully generated debug HSACO: {hsaco_path}")
    return hsaco_path


# ============================================================
# 主程式
# ============================================================

def detect_kernel_info(asm_text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    從 ISA 程式碼自動偵測 kernel 名稱和類型
    
    Returns:
        (kernel_name, kernel_type)
    """
    kernel_name = None
    kernel_type = None
    
    # 提取 kernel 名稱
    kernel_match = re.search(r'\.globl\s+(\S+)', asm_text)
    if kernel_match:
        kernel_name = kernel_match.group(1)
        # 忽略 __hip_cuid_* 這類輔助符號
        if kernel_name.startswith('__hip_cuid'):
            # 嘗試找第二個 .globl
            all_matches = re.findall(r'\.globl\s+(\S+)', asm_text)
            for name in all_matches:
                if not name.startswith('__hip_cuid'):
                    kernel_name = name
                    break
    
    # 根據 kernel 名稱推斷類型
    if kernel_name:
        name_lower = kernel_name.lower()
        if 'vectoradd' in name_lower:
            kernel_type = 'float_add'
        elif 'vectormul' in name_lower:
            kernel_type = 'float_mul'
        elif 'vectordot' in name_lower:
            kernel_type = 'float_dot'
        elif 'saxpy' in name_lower:
            kernel_type = 'float_saxpy'
        elif 'conditionalops' in name_lower:
            kernel_type = 'float_cond'
        elif 'scalarops' in name_lower:
            kernel_type = 'int_scalar'
        elif 'memoryops' in name_lower:
            kernel_type = 'int_mem'
        elif 'conditionalkernel' in name_lower:
            kernel_type = 'int_cond'
        elif 'loopkernel' in name_lower:
            kernel_type = 'int_loop'
        elif 'sharedmemkernel' in name_lower:
            kernel_type = 'int_shared'
    
    return kernel_name, kernel_type


def main():
    ap = argparse.ArgumentParser(
        description="AMD ISA Assembly Debug Tool - 在組合語言中插入 printf 除錯功能",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例：
  1. 在 .s 檔案中標註：
     ; @PRINT f"v6={v6:.3f}"
     ; @PRINT if v6 == 0.0: f"A={v6:.3f}, B={v7:.3f}"

  2. 執行工具：
     python3 asm_debug.py input.s --output-dir debug_output

  3. 直接測試（需要 universal_hsaco_runner）：
     python3 asm_debug.py input.s --test --test-size 64
        """
    )
    
    ap.add_argument(
        "input_file",
        help="輸入的 .s 組合語言檔案"
    )
    ap.add_argument(
        "--output-dir",
        default="debug_output",
        help="輸出目錄（預設：debug_output）"
    )
    ap.add_argument(
        "--chip",
        default="gfx950",
        help="目標 GPU 架構（預設：gfx950）"
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="只解析 @PRINT 指令，不執行編譯"
    )
    ap.add_argument(
        "--no-printf",
        action="store_true",
        help="禁用 printf 注入（用於純功能驗證）"
    )
    ap.add_argument(
        "--print-mode",
        choices=["defer", "inline"],
        default="defer",
        help="printf 注入模式：defer（預設，延後到結尾）或 inline（插入原位置）"
    )
    ap.add_argument(
        "--test",
        action="store_true",
        help="使用 universal_hsaco_runner 執行測試"
    )
    ap.add_argument(
        "--test-size",
        type=int,
        default=64,
        help="測試數據大小（預設：64）"
    )
    ap.add_argument(
        "--kernel-name",
        help="Kernel 名稱（可自動偵測）"
    )
    ap.add_argument(
        "--kernel-type",
        help="Kernel 類型：float_add, int_scalar, int_mem, int_cond, int_loop, int_shared 等"
    )
    ap.add_argument(
        "--runner-path",
        help="universal_hsaco_runner 的路徑"
    )
    ap.add_argument(
        "--run",
        action="store_true",
        help="使用 mlir-runner 執行帶有 printf 的 kernel（較舊的方式）"
    )
    ap.add_argument(
        "--rocm-runtime-lib",
        help="libmlir_rocm_runtime.so 路徑（用於 --run）"
    )
    ap.add_argument(
        "--runner-utils-lib",
        help="libmlir_runner_utils.so 路徑（用於 --run）"
    )
    
    args = ap.parse_args()
    
    input_path = pathlib.Path(args.input_file).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    workdir = pathlib.Path(args.output_dir)
    workdir.mkdir(parents=True, exist_ok=True)
    
    # 讀取 ASM 檔案
    asm_text = input_path.read_text()
    
    # 自動偵測 kernel 資訊
    auto_kernel_name, auto_kernel_type = detect_kernel_info(asm_text)
    kernel_name = args.kernel_name or auto_kernel_name
    kernel_type = args.kernel_type or auto_kernel_type
    
    print(f"=== AMD ISA Assembly Debug Tool ===")
    print(f"Input: {input_path}")
    print(f"Output: {workdir}")
    print(f"Chip: {args.chip}")
    if kernel_name:
        print(f"Kernel Name: {kernel_name}")
    if kernel_type:
        print(f"Kernel Type: {kernel_type}")
    
    # 1. 解析 @PRINT 和 @TIMESTAMP 指令
    print(f"\n=== Parsing @PRINT and @TIMESTAMP directives ===")
    try:
        lines, directives, has_barrier, timestamp_directives = parse_asm_file(input_path)
    except ValueError as e:
        print(f"\n❌ ERROR: {e}")
        print(f"\n   Please check your directive syntax.")
        sys.exit(1)
    
    if not directives and not timestamp_directives:
        if args.no_printf:
            print("[Info] No @PRINT/@TIMESTAMP directives found (--no-printf mode)")
        else:
            print("[Info] No @PRINT/@TIMESTAMP directives found. Will generate HSACO without printf.")
    else:
        if directives:
            print(f"\nFound {len(directives)} @PRINT directive(s)")
        if timestamp_directives:
            print(f"Found {len(timestamp_directives)} @TIMESTAMP directive(s)")
    
    # 警告：s_barrier 與 printf 不兼容
    if has_barrier and (directives or timestamp_directives) and not args.no_printf:
        print("\n" + "=" * 60)
        print("⚠️  WARNING: Kernel contains s_barrier instruction!")
        print("   gpu.printf's hostcall mechanism may conflict with barrier")
        print("   synchronization, causing kernel to hang or crash.")
        print("")
        print("   Recommendations:")
        print("   1. Use --no-printf for functional verification")
        print("   2. Use cond=REG_eq(N) to limit printf (e.g., v6_eq(0.0))")
        print("   3. Place @PRINT only after all barriers complete")
        print("=" * 60 + "\n")

        if args.print_mode == "inline":
            print("[Info] Inline printf disabled due to s_barrier; falling back to defer mode.")
            args.print_mode = "defer"
    
    if args.dry_run:
        print("\n[Dry Run] Stopping here.")
        return
    
    # 2. 分析暫存器使用量
    print(f"\n=== Analyzing register usage ===")
    reg_info = analyze_registers(asm_text)
    print(f"  VGPR: {reg_info['vgpr']}, SGPR: {reg_info['sgpr']}, AGPR: {reg_info['agpr']}")
    
    # 3. 轉換為 GPU MLIR
    gpumlir_path = translate_to_gpumlir(input_path, workdir)
    
    # 4. 注入 printf/timestamp 程式碼（如果有 @PRINT/@TIMESTAMP 且未禁用）
    has_printf = (bool(directives) or bool(timestamp_directives)) and not args.no_printf
    
    required_vgpr = 0
    required_sgpr = 0
    
    if (not directives and not timestamp_directives) or args.no_printf:
        print(f"\nUsing original GPU MLIR (no printf injection)")
        modified_path = gpumlir_path
    else:
        print(f"\n=== Injecting printf/timestamp code ===")
        gpumlir_text = gpumlir_path.read_text()
        modified_mlir, required_vgpr, required_sgpr, required_private_size = inject_printf_into_mlir(
            gpumlir_text, directives, reg_info, timestamp_directives, print_mode=args.print_mode
        )
        
        modified_path = workdir / f"{input_path.stem}_debug_injected.gpumlir"
        modified_path.write_text(modified_mlir)
        print(f"Generated modified GPU MLIR: {modified_path}")
    
    # 5. 如果使用 --run（舊方式：mlir-runner），則使用 mlir-runner 執行
    if args.run:
        if args.no_printf:
            print("\n[Warning] --run 與 --no-printf 一起使用意義不大")
        
        # 生成 host wrapper 並執行
        host_mlir_path = generate_host_wrapper_mlir(modified_path, args.test_size, workdir)
        run_with_mlir_runner(
            host_mlir_path, 
            args.chip, 
            workdir,
            args.rocm_runtime_lib,
            args.runner_utils_lib
        )
        return
    
    # 6. 生成 HSACO
    hsaco_path = build_debug_hsaco(modified_path, args.chip, workdir, input_path, has_printf,
                                    required_vgpr, required_sgpr, required_private_size)
    
    print(f"\n=== Done ===")
    print(f"Debug HSACO: {hsaco_path}")
    
    # 7. 如果使用 --test，執行 universal_hsaco_runner
    if args.test:
        if not kernel_name:
            print("\n[Error] Cannot run test: kernel name not detected.")
            print("        Please specify with --kernel-name")
            return
        if not kernel_type:
            print("\n[Error] Cannot run test: kernel type not detected.")
            print("        Please specify with --kernel-type")
            return
        
        # 尋找 universal_hsaco_runner
        runner_path = args.runner_path
        if not runner_path:
            # 嘗試在 Project-MDR 目錄下尋找
            project_root = pathlib.Path(__file__).parent
            possible_paths = [
                project_root / "Track_B" / "kernel_testcases" / "universal_hsaco_runner",
                project_root / "universal_hsaco_runner",
                pathlib.Path("universal_hsaco_runner"),
            ]
            for p in possible_paths:
                if p.exists():
                    runner_path = str(p)
                    break
        
        if not runner_path or not pathlib.Path(runner_path).exists():
            print("\n[Error] universal_hsaco_runner not found.")
            print("        Please compile it or specify path with --runner-path")
            return
        
        print(f"\n=== Running test with universal_hsaco_runner ===")
        cmd = [
            runner_path,
            str(hsaco_path),
            kernel_name,
            kernel_type,
            str(args.test_size)
        ]
        run_cmd(cmd)
    else:
        # 顯示如何手動測試
        print(f"\n[Info] 可以使用 universal_hsaco_runner 測試此 HSACO:")
        if kernel_name and kernel_type:
            print(f"       Track_B/kernel_testcases/universal_hsaco_runner {hsaco_path} {kernel_name} {kernel_type} {args.test_size}")


if __name__ == "__main__":
    main()
