#!/usr/bin/env python3
"""
AMD ISA Assembly Capture Tool (Register-based)
===============================================

在組合語言中插入暫存器快照功能，將值保存到指定的 register 中。

使用方式：
1. 在 .s 檔案中標註要捕獲的內容（建議使用 f-string）：
   ; @CAPTURE f"A={v2}, B={v3}" dst=v10,v11
   ; @CAPTURE if $tid == 0: f"A={v2}, A*2={(v2*2.0):.2f}" dst=v12,v13
   ; @CAPTURE f"addr={(s[12:13] + s[16:17]*4):ld}" dst=s[12:13]

2. 執行此工具：
   python3 mdr_cap.py input.s --output-dir output

3. 工具會：
   - 在 @CAPTURE 標記處直接插入 ISA 指令
   - 生成 injected.s
   - 轉換為 MLIR 並加入 register clobbering
   - 生成最終的 HSACO

優點：
- 純 register 操作，零記憶體開銷
- 精確控制插入位置
- 自動保護所有使用的 registers
"""

import argparse
import pathlib
import re
import shutil
import subprocess
import sys
import yaml
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Set


# 常數定義
MAX_VECTOR_SIZE = 8  # inline_asm 約束的最大 vector 大小（降低以避免 AMDGPU backend 問題）


# ============================================================
# @CAPTURE 標記解析
# ============================================================

@dataclass
class CaptureDirective:
    """代表一個 @CAPTURE 指令"""
    line_number: int
    source_registers: List[str]   # 來源 register (src)
    target_registers: List[str]   # 目標 register (dst)
    types: List[str]              # 類型
    condition: Optional[str] = None
    expressions: Optional[List[str]] = None
    ordered_values: Optional[List[Tuple[str, bool]]] = None  # [(value, is_expression), ...]
    
    def __str__(self):
        cond_str = f" [cond={self.condition}]" if self.condition else ""
        if self.expressions:
            return f"@CAPTURE at line {self.line_number + 1}: {self.source_registers} + expr={self.expressions} → {self.target_registers}{cond_str}"
        return f"@CAPTURE at line {self.line_number + 1}: {self.source_registers} → {self.target_registers}{cond_str}"


def _split_fstring_format_spec(content: str) -> Tuple[str, str]:
    """
    在 f-string placeholder 中切分 format spec，忽略括號內的冒號。
    例如: "s[12:13]:ld" -> ("s[12:13]", "ld")
    """
    depth_paren = 0
    depth_bracket = 0
    split_at = None
    for idx, ch in enumerate(content):
        if ch == '(':
            depth_paren += 1
        elif ch == ')':
            depth_paren = max(depth_paren - 1, 0)
        elif ch == '[':
            depth_bracket += 1
        elif ch == ']':
            depth_bracket = max(depth_bracket - 1, 0)
        elif ch == ':' and depth_paren == 0 and depth_bracket == 0:
            split_at = idx
    if split_at is None:
        return content, ''
    return content[:split_at], content[split_at + 1:]


def parse_capture_fstring(fstring: str) -> Tuple[List[str], List[str]]:
    """
    解析 @CAPTURE 的 Python f-string placeholder
    
    輸入: f"A={v6:.3f}, B={(v6+v7):.2f}"
    輸出: (["v6", "(v6+v7)"], ["f32", "f32"])
    
    支援的格式說明符：
    - {v6} - 預設格式（VGPR 預設 f32，SGPR 預設 i32）
    - {v6:f} 或 {v6:.f} - 浮點數 f32
    - {v6:.3f} - 浮點數（精度僅影響輸出顯示）
    - {v6:d} - 整數 i32
    - {v6:ld} - 長整數 i64
    - {s[12:13]:ld} - SGPR 64-bit pair（使用 s[lo:hi] 格式）
    - {expr} - 表達式（如 {(v6*v7):.2f}）
    """
    values = []
    types = []
    pattern = r'\{([^}]+)\}'
    
    for match in re.finditer(pattern, fstring):
        content = match.group(1)
        if re.match(r'^[sva]\d+:(?:[sva])?\d+$', content.strip()):
            raise ValueError("Use s[lo:hi] format for register pairs in f-strings (e.g., s[12:13])")
        reg_or_expr, fmt_spec = _split_fstring_format_spec(content)
        
        reg_or_expr = reg_or_expr.strip()
        fmt_spec = fmt_spec.strip()

        if re.match(r'^[sva]\d+:(?:[sva])?\d+$', reg_or_expr):
            raise ValueError("Use s[lo:hi] format for register pairs in f-strings (e.g., s[12:13])")
        
        if reg_or_expr in ('$tid', '$lane'):
            raise ValueError("Builtin $tid/$lane is not supported in @CAPTURE")
        
        values.append(reg_or_expr)

        pair_match = re.match(r'^([sva])\[(\d+):(\d+)\]$', reg_or_expr)
        is_reg_pair = pair_match is not None
        if is_reg_pair and pair_match.group(1) != 's':
            raise ValueError("Only SGPR pair s[lo:hi] is supported in f-string placeholders")

        is_simple_reg = re.match(r'^[sva]\d+$', reg_or_expr)
        if fmt_spec:
            if fmt_spec == 'd':
                types.append('i32')
            elif fmt_spec == 'ld':
                types.append('i64')
            elif fmt_spec == 'f' or fmt_spec == '.f':
                types.append('f32')
            elif fmt_spec == 'lf':
                types.append('f64')
            elif fmt_spec.endswith('f'):
                # 忽略精度，僅判斷為浮點
                types.append('f32')
            else:
                if is_reg_pair and reg_or_expr.startswith('s'):
                    types.append('i64')
                elif is_simple_reg and reg_or_expr.startswith('s'):
                    types.append('i32')
                elif re.search(r's\[\d+:\d+\]', reg_or_expr):
                    types.append('i64')
                else:
                    types.append('f32')
        else:
            if is_reg_pair and reg_or_expr.startswith('s'):
                types.append('i64')
            elif is_simple_reg and reg_or_expr.startswith('s'):
                types.append('i32')
            elif re.search(r's\[\d+:\d+\]', reg_or_expr):
                types.append('i64')
            else:
                types.append('f32')
    
    return values, types


def _parse_reg_pair(reg: str) -> Optional[Tuple[str, int, int]]:
    """解析 s/v/a 暫存器 pair：s[12:13]（建議）或 s12:s13（舊式）"""
    m = re.match(r'^([sva])\[(\d+):(\d+)\]$', reg)
    if m:
        reg_type, lo, hi = m.group(1), int(m.group(2)), int(m.group(3))
        return reg_type, lo, hi
    m = re.match(r'^([sva])(\d+):(?:[sva])?(\d+)$', reg)
    if m:
        reg_type, lo, hi = m.group(1), int(m.group(2)), int(m.group(3))
        return reg_type, lo, hi
    return None


def _is_reg_pair(reg: str, reg_type: Optional[str] = None) -> bool:
    parsed = _parse_reg_pair(reg)
    if not parsed:
        return False
    if reg_type is None:
        return True
    return parsed[0] == reg_type


def _format_sgpr_pair(reg: str) -> Tuple[int, int]:
    parsed = _parse_reg_pair(reg)
    if not parsed or parsed[0] != 's':
        raise ValueError(f"Invalid SGPR pair: {reg}")
    _, lo, hi = parsed
    return lo, hi


def _is_deprecated_reg_pair(reg: str) -> bool:
    return re.match(r'^[sva]\d+:(?:[sva])?\d+$', reg) is not None


def _normalize_reg_pair_style(reg: str, line_number: int, field: str) -> str:
    if _is_deprecated_reg_pair(reg):
        parsed = _parse_reg_pair(reg)
        if parsed:
            reg_type, lo, hi = parsed
            fixed = f"{reg_type}[{lo}:{hi}]"
            print(f"[Warning] Deprecated register pair format at line {line_number + 1} in {field}: {reg} -> {fixed}")
            return fixed
    return reg


def _is_plain_register_token(token: str) -> bool:
    return (
        re.match(r'^[sva]\d+$', token) is not None
        or re.match(r'^[sva]\[\d+:\d+\]$', token) is not None
    )


@dataclass
class CaptureMapping:
    """記錄 capture 的結果存放在哪個 register"""
    directive_id: int
    source: str                   # 來源：register 名稱或表達式
    target_register: str          # 目標 register（如 "v10"）
    type_str: str                 # 類型（如 "f32"）
    is_expression: bool = False   # 是否為表達式


def parse_capture_directive(line: str, line_number: int) -> Optional[CaptureDirective]:
    """
    解析 @CAPTURE 指令
    
    支援語法：
    1. 舊格式（向後兼容）：reg=v2,v3 type=f32,f32
       - 自動使用相同的 register 作為目標（in-place capture）
    2. 新格式：src=v2,v3 dst=v10,v11 type=f32,f32
       - 顯式指定目標 register
    3. Pythonic 條件式（僅支援 $tid / tid）：
       - if $tid == 0: src=... dst=... type=...
       - if tid >= 8: src=... dst=... type=...
    4. Python f-string 風格（建議，從 placeholder 推導 type）：
       - f"A={v2}, B={v3}" dst=v10,v11
       - if $tid == 0: f"A={v2}, A*2={(v2*2.0):.2f}" dst=v12,v13
       - f"addr={(s[12:13] + s[16:17]*4):ld}" dst=s[12:13]
    5. expr=（舊式，仍支援但建議改用 f-string）：
       - expr="v2*2.0" dst=v10 type=f32
    """
    match = re.search(r'[;#]\s*@CAPTURE\s+(.+)', line)
    if not match:
        return None
    
    directive_content = match.group(1).strip()
    
    # 解析 Pythonic 條件式（if ... :)
    condition = None
    cond_prefix_match = re.match(r'(if\s+[^:]+:)\s*(.*)$', directive_content)
    if cond_prefix_match:
        cond_str, remainder = cond_prefix_match.groups()
        internal_cond = parse_condition_pythonic(cond_str)
        if internal_cond:
            condition = internal_cond
            directive_content = remainder.strip()
        else:
            print(f"[Warning] Invalid pythonic condition at line {line_number + 1}: {cond_str}")
    
    # 解析各個屬性
    # 舊格式：reg=...
    reg_match = re.search(r'\breg\s*=\s*([\w,\[\]:\s]+?)(?=\s+(?:type|cond|expr)\b|$)', directive_content)
    # 新格式：src=... dst=...
    src_match = re.search(r'\bsrc\s*=\s*([\w,\[\]:\s]+?)(?=\s+(?:dst|type|cond|expr)\b|$)', directive_content)
    dst_match = re.search(r'\bdst\s*=\s*([\w,\[\]:\s]+?)(?=\s+(?:src|type|cond|expr)\b|$)', directive_content)
    
    type_match = re.search(r'type\s*=\s*([\w,\s]+?)(?=\s+(?:reg|src|dst|cond|expr)\b|$)', directive_content)
    cond_match = re.search(r'cond\s*=\s*(\w+\([^)]+\))', directive_content)
    expr_match = re.search(r'expr\s*=\s*"([^"]+)"', directive_content)
    fstring_match = re.search(r'f"([^"]*)"', directive_content)
    
    if not type_match and not fstring_match:
        print(f"[Warning] @CAPTURE missing 'type' at line {line_number + 1}")
        return None
    
    # 解析類型
    types = []
    if type_match:
        type_str = type_match.group(1).strip().rstrip(',')
        types = [t.strip() for t in type_str.split(',')]
    
    # 解析 source 和 target registers
    source_registers = []
    target_registers = []
    expressions = []
    ordered_values = []
    
    # f-string 模式：從 placeholder 推導來源/表達式與類型
    if fstring_match:
        if not dst_match:
            print(f"[Warning] @CAPTURE f-string missing 'dst' at line {line_number + 1}")
            return None
        
        fstring_content = fstring_match.group(1)
        try:
            parsed_values, parsed_types = parse_capture_fstring(fstring_content)
        except Exception as e:
            print(f"[Warning] Failed to parse @CAPTURE f-string at line {line_number + 1}: {e}")
            return None
        
        if not parsed_values:
            print(f"[Warning] @CAPTURE f-string has no placeholders at line {line_number + 1}")
            return None
        
        # f-string 使用 placeholder 的順序
        ordered_values = []
        for val in parsed_values:
            if _is_plain_register_token(val):
                source_registers.append(val)
                ordered_values.append((val, False))
            else:
                expressions.append(val)
                ordered_values.append((val, True))
        
        types = parsed_types
        
        dst_str = dst_match.group(1).strip().rstrip(',')
        target_registers = [
            _normalize_reg_pair_style(r.strip(), line_number, "dst")
            for r in dst_str.split(',')
        ]
        
        if reg_match or src_match or expr_match or type_match:
            print(f"[Info] @CAPTURE f-string ignores reg/src/expr/type at line {line_number + 1}")
    # 優先使用新格式（src/dst）
    elif src_match and dst_match:
        # 新格式：src=... dst=...
        src_str = src_match.group(1).strip().rstrip(',')
        source_registers = [
            _normalize_reg_pair_style(r.strip(), line_number, "src")
            for r in src_str.split(',')
        ]
        
        dst_str = dst_match.group(1).strip().rstrip(',')
        target_registers = [
            _normalize_reg_pair_style(r.strip(), line_number, "dst")
            for r in dst_str.split(',')
        ]
    elif dst_match and not src_match:
        # 只有 dst（可能只有表達式）
        dst_str = dst_match.group(1).strip().rstrip(',')
        target_registers = [
            _normalize_reg_pair_style(r.strip(), line_number, "dst")
            for r in dst_str.split(',')
        ]
        # source_registers 留空，稍後會檢查是否有表達式
    elif reg_match:
        # 舊格式：reg=... (in-place capture)
        reg_str = reg_match.group(1).strip().rstrip(',')
        registers = [
            _normalize_reg_pair_style(r.strip(), line_number, "reg")
            for r in reg_str.split(',')
        ]
        source_registers = registers
        target_registers = registers  # 使用相同的 register
        print(f"[Info] Old format detected at line {line_number + 1}, using in-place capture: {registers}")
    elif src_match:
        # 只有 src 沒有 dst（錯誤）
        print(f"[Warning] @CAPTURE has 'src' but missing 'dst' at line {line_number + 1}")
        return None
    else:
        print(f"[Warning] @CAPTURE must have 'reg', 'src'/'dst', or 'dst' with 'expr' at line {line_number + 1}")
        return None
    
    # 解析表達式
    if expr_match and not fstring_match:
        expr_str = expr_match.group(1).strip()
        if re.search(r'\b[sva]\d+:(?:[sva])?\d+\b', expr_str):
            print(f"[Warning] Deprecated register pair format in expr at line {line_number + 1}; use s[lo:hi]")
        expressions = [e.strip() for e in expr_str.split(';')]
    
    # 條件（cond= 會覆蓋 pythonic 條件式）
    if cond_match:
        condition = cond_match.group(1)
    
    # 驗證
    if not source_registers and not expressions:
        print(f"[Warning] @CAPTURE must have source registers or expressions at line {line_number + 1}")
        return None
    
    total_values = len(source_registers) + len(expressions)
    if total_values != len(types):
        print(f"[Warning] Value/type count mismatch at line {line_number + 1}: {total_values} values vs {len(types)} types")
        return None
    
    # 如果使用舊格式且有表達式，需要額外的目標 registers
    if reg_match and expressions:
        print(f"[Warning] Old format with expressions at line {line_number + 1}")
        print(f"          Please use new format or f-string to specify target registers for expressions")
        print(f"          Example: f\"A={{v2}}, A*2={{(v2*2.0):.2f}}\" dst=v2,v10")
        return None
    
    if total_values != len(target_registers):
        print(f"[Warning] Source/target register count mismatch at line {line_number + 1}: {total_values} sources vs {len(target_registers)} targets")
        return None

    # 驗證 i64 需要 SGPR pair 目標
    for typ, dst in zip(types, target_registers):
        if typ == 'i64':
            if not _is_reg_pair(dst, 's'):
                print(f"[Warning] i64 target must be SGPR pair (s[lo:hi]) at line {line_number + 1}: {dst}")
                return None
        else:
            if _is_reg_pair(dst):
                print(f"[Warning] Non-i64 target cannot be register pair at line {line_number + 1}: {dst}")
                return None
    
    if not ordered_values:
        ordered_values = [(r, False) for r in source_registers]
        if expressions:
            ordered_values.extend((e, True) for e in expressions)
    
    # 驗證 i64 來源（若為 register）
    for (value, is_expr), typ in zip(ordered_values, types):
        if typ == 'i64':
            if not is_expr and not _is_reg_pair(value, 's'):
                print(f"[Warning] i64 source must be SGPR pair (s[lo:hi]) at line {line_number + 1}: {value}")
                return None
        else:
            if not is_expr and _is_reg_pair(value):
                print(f"[Warning] Non-i64 source cannot be register pair at line {line_number + 1}: {value}")
                return None
    
    return CaptureDirective(
        line_number=line_number,
        source_registers=source_registers,
        target_registers=target_registers,
        types=types,
        condition=condition,
        expressions=expressions if expressions else None,
        ordered_values=ordered_values if ordered_values else None
    )


def parse_asm_file(asm_path: pathlib.Path) -> Tuple[List[str], List[CaptureDirective]]:
    """解析 .s 檔案，提取 @CAPTURE 指令"""
    lines = asm_path.read_text().split('\n')
    directives = []
    
    for i, line in enumerate(lines):
        if '@CAPTURE' in line:
            directive = parse_capture_directive(line, i)
            if directive:
                directives.append(directive)
                print(f"[Info] Found: {directive}")
    
    return lines, directives


def parse_condition_pythonic(cond_str: str) -> Optional[str]:
    """
    解析 Python 風格的條件式（僅支援 $tid / tid）
    
    輸入: "if $tid > 2:" 或 "if tid == 0:"
    輸出: "tid_gt(2)" 或 "tid_eq(0)"
    
    支援的運算子：
    - == -> eq
    - != -> ne
    - < -> lt
    - <= -> le
    - > -> gt
    - >= -> ge
    """
    # 移除 "if " 前綴和 ":" 後綴
    cond_str = cond_str.strip()
    if cond_str.startswith('if '):
        cond_str = cond_str[3:]
    if cond_str.endswith(':'):
        cond_str = cond_str[:-1]
    cond_str = cond_str.strip()
    
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
                if reg in ('$tid', 'tid'):
                    return f"tid_{op_name}({value})"
                print(f"[Warning] Unsupported pythonic condition register: {reg} (only $tid/tid)")
                return None
    
    return None


# ============================================================
# Register 分析
# ============================================================

def analyze_registers(isa_code: str) -> Dict[str, int]:
    """分析 ISA 程式碼中的暫存器使用量"""
    max_vgpr = 0
    max_sgpr = 0
    max_agpr = 0
    
    vgpr_pattern = re.compile(r'\bv(\d+)\b')
    sgpr_pattern = re.compile(r'\bs(\d+)\b')
    agpr_pattern = re.compile(r'\ba(\d+)\b')
    vgpr_range_pattern = re.compile(r'\bv\[(\d+):(\d+)\]')
    sgpr_range_pattern = re.compile(r'\bs\[(\d+):(\d+)\]')
    agpr_range_pattern = re.compile(r'\ba\[(\d+):(\d+)\]')
    
    for line in isa_code.split('\n'):
        code_part = line.split(';')[0].split('#')[0].strip()
        if not code_part or code_part.endswith(':'):
            continue
        
        for match in vgpr_range_pattern.finditer(code_part):
            end = int(match.group(2))
            max_vgpr = max(max_vgpr, end + 1)
        
        for match in sgpr_range_pattern.finditer(code_part):
            end = int(match.group(2))
            max_sgpr = max(max_sgpr, end + 1)
        
        for match in agpr_range_pattern.finditer(code_part):
            end = int(match.group(2))
            max_agpr = max(max_agpr, end + 1)
        
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


def collect_all_target_registers(directives: List[CaptureDirective]) -> Set[str]:
    """收集所有用戶定義的目標 registers"""
    all_targets = set()
    for directive in directives:
        for reg in directive.target_registers:
            if _is_reg_pair(reg):
                reg_type, lo, hi = _parse_reg_pair(reg)
                all_targets.add(f"{reg_type}{lo}")
                all_targets.add(f"{reg_type}{hi}")
            else:
                all_targets.add(reg)
    return all_targets


def calculate_final_register_usage(original_usage: Dict[str, int], 
                                   target_registers: Set[str]) -> Dict[str, int]:
    """計算包含目標 registers 的最終使用量"""
    final_usage = original_usage.copy()
    
    # 分析目標 registers 的最大編號
    for reg in target_registers:
        if _is_reg_pair(reg):
            reg_type, lo, hi = _parse_reg_pair(reg)
            if reg_type == 'v':
                final_usage['vgpr'] = max(final_usage['vgpr'], hi + 1)
            elif reg_type == 's':
                final_usage['sgpr'] = max(final_usage['sgpr'], hi + 1)
            elif reg_type == 'a':
                final_usage['agpr'] = max(final_usage['agpr'], hi + 1)
        else:
            if reg.startswith('v'):
                num = int(reg[1:])
                final_usage['vgpr'] = max(final_usage['vgpr'], num + 1)
            elif reg.startswith('s'):
                num = int(reg[1:])
                final_usage['sgpr'] = max(final_usage['sgpr'], num + 1)
            elif reg.startswith('a'):
                num = int(reg[1:])
                final_usage['agpr'] = max(final_usage['agpr'], num + 1)
    
    return final_usage


# ============================================================
# 表達式編譯為 ISA 指令
# ============================================================

def parse_expression(expr: str) -> List[Tuple[str, str]]:
    """解析算術表達式為 token 列表"""
    tokens = []
    i = 0
    expr = expr.strip()
    
    while i < len(expr):
        ch = expr[i]
        
        if ch.isspace():
            i += 1
            continue
        
        # SGPR/VGPR/AGPR pair: s[12:13] (preferred) or s12:s13 (legacy)
        pair_match = re.match(r'([sva])\[(\d+):(\d+)\]', expr[i:])
        if pair_match:
            reg_type, lo, hi = pair_match.group(1), pair_match.group(2), pair_match.group(3)
            tokens.append(('REG64', f"{reg_type}{lo}:{reg_type}{hi}"))
            i += len(pair_match.group(0))
            continue
        pair_match = re.match(r'([sva])(\d+):(?:[sva])?(\d+)', expr[i:])
        if pair_match:
            reg_type, lo, hi = pair_match.group(1), pair_match.group(2), pair_match.group(3)
            tokens.append(('REG64', f"{reg_type}{lo}:{reg_type}{hi}"))
            i += len(pair_match.group(0))
            continue
        
        if ch in 'vsa' and i + 1 < len(expr) and expr[i + 1].isdigit():
            j = i + 1
            while j < len(expr) and expr[j].isdigit():
                j += 1
            tokens.append(('REG', expr[i:j]))
            i = j
            continue
        
        if ch.isdigit() or (ch == '-' and i + 1 < len(expr) and (expr[i + 1].isdigit() or expr[i + 1] == '.')):
            j = i
            if ch == '-':
                j += 1
            while j < len(expr) and (expr[j].isdigit() or expr[j] == '.'):
                j += 1
            tokens.append(('NUM', expr[i:j]))
            i = j
            continue
        
        if ch in '+-*/':
            tokens.append(('OP', ch))
            i += 1
            continue
        
        if ch == '(':
            tokens.append(('LPAREN', ch))
            i += 1
            continue
        if ch == ')':
            tokens.append(('RPAREN', ch))
            i += 1
            continue
        
        raise ValueError(f"Unknown character in expression: '{ch}' at position {i}")
    
    return tokens


def count_ops_in_expression(expr: str) -> int:
    """估算表達式的運算子數量，用於分配暫存 VGPR。"""
    expr = expr.strip()
    # 移除最外層括號（避免 token 過多時誤判）
    while expr.startswith('(') and expr.endswith(')'):
        depth = 0
        balanced = True
        for i, ch in enumerate(expr):
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0 and i != len(expr) - 1:
                    balanced = False
                    break
        if balanced and depth == 0:
            expr = expr[1:-1].strip()
        else:
            break
    try:
        tokens = parse_expression(expr)
    except Exception:
        return 0
    return sum(1 for t in tokens if t[0] == 'OP')


def count_i64_temp_pairs(expr: str) -> int:
    """估算 i64 表達式所需的 SGPR pair 暫存數量（保守估計）。"""
    try:
        tokens = parse_expression(expr)
    except Exception:
        return 0
    op_count = sum(1 for t in tokens if t[0] == 'OP')
    non_pair_operands = sum(1 for t in tokens if t[0] in ('REG', 'NUM'))
    # 保守估計：每個運算可能需要額外暫存（乘法需要中間值）
    return op_count * 2 + non_pair_operands


def compile_expression_to_isa(expr: str, result_type: str, target_reg: str,
                              temp_pool: Optional[List[str]] = None,
                              temp_sgpr_pairs: Optional[List[Tuple[int, int]]] = None) -> List[str]:
    """
    將表達式編譯為 ISA 指令列表
    
    Args:
        expr: 表達式字串（如 "v2*2.0"）
        result_type: 結果類型（f32, i32 等）
        target_reg: 目標 register（如 "v10"）
    
    Returns:
        ISA 指令列表
    """
    expr = expr.strip()
    
    # 移除最外層多餘的括號，例如 "(v2*2.0)"
    def strip_outer_parens(text: str) -> str:
        while text.startswith('(') and text.endswith(')'):
            depth = 0
            balanced = True
            for i, ch in enumerate(text):
                if ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
                    if depth == 0 and i != len(text) - 1:
                        balanced = False
                        break
            if balanced and depth == 0:
                text = text[1:-1].strip()
            else:
                break
        return text
    
    expr = strip_outer_parens(expr)
    tokens = parse_expression(expr)
    instructions = []
    
    def to_rpn(expr_tokens: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        precedence = {'+': 1, '-': 1, '*': 2, '/': 2}
        output = []
        op_stack = []
        for tok_type, tok_val in expr_tokens:
            if tok_type in ('REG', 'NUM', 'REG64'):
                output.append((tok_type, tok_val))
            elif tok_type == 'OP':
                while op_stack and op_stack[-1][0] == 'OP' and precedence[op_stack[-1][1]] >= precedence[tok_val]:
                    output.append(op_stack.pop())
                op_stack.append((tok_type, tok_val))
            elif tok_type == 'LPAREN':
                op_stack.append((tok_type, tok_val))
            elif tok_type == 'RPAREN':
                while op_stack and op_stack[-1][0] != 'LPAREN':
                    output.append(op_stack.pop())
                if not op_stack:
                    raise ValueError(f"Mismatched parentheses in expression: {expr}")
                op_stack.pop()  # 移除 LPAREN
            else:
                raise ValueError(f"Unsupported token: {tok_type}")
        while op_stack:
            if op_stack[-1][0] == 'LPAREN':
                raise ValueError(f"Mismatched parentheses in expression: {expr}")
            output.append(op_stack.pop())
        return output
    
    def format_const(value: str) -> str:
        if result_type.startswith('f') and '.' not in value:
            return f"{value}.0"
        return value
    
    # i64 or REG64: 使用 SGPR pair 指令序列
    if result_type == 'i64' or any(t[0] == 'REG64' for t in tokens):
        rpn = to_rpn(tokens)
        op_count = sum(1 for t in rpn if t[0] == 'OP')
        if op_count == 0:
            raise ValueError(f"Unsupported expression format: {expr}")
        
        sgpr_pool = [] if temp_sgpr_pairs is None else list(temp_sgpr_pairs)
        
        def alloc_pair() -> Tuple[int, int]:
            if not sgpr_pool:
                raise ValueError("i64 expression needs temp SGPR pairs")
            return sgpr_pool.pop(0)
        
        def format_imm32(val: int) -> str:
            if val < 0:
                val &= 0xffffffff
            if val > 9:
                return hex(val)
            return str(val)
        
        def ensure_pair(item: Tuple[str, str]) -> Tuple[int, int]:
            kind, value = item
            if kind == 'PAIR':
                return value
            if kind == 'REG':
                lo, hi = alloc_pair()
                instructions.append(f"\ts_mov_b32 s{lo}, {value}")
                instructions.append(f"\ts_mov_b32 s{hi}, 0")
                return lo, hi
            if kind == 'NUM':
                lo, hi = alloc_pair()
                num = int(float(value))
                lo_val = num & 0xffffffff
                hi_val = (num >> 32) & 0xffffffff
                instructions.append(f"\ts_mov_b32 s{lo}, {format_imm32(lo_val)}")
                instructions.append(f"\ts_mov_b32 s{hi}, {format_imm32(hi_val)}")
                return lo, hi
            raise ValueError(f"Unsupported operand for i64: {kind}")
        
        def ensure_reg32(item: Tuple[str, str]) -> str:
            kind, value = item
            if kind == 'REG':
                return value
            if kind == 'NUM':
                lo, _ = alloc_pair()
                num = int(float(value))
                instructions.append(f"\ts_mov_b32 s{lo}, {format_imm32(num)}")
                return f"s{lo}"
            raise ValueError(f"Unsupported operand for i64 mul: {kind}")
        
        def parse_pair_str(reg: str) -> Tuple[int, int]:
            lo, hi = _format_sgpr_pair(reg)
            return lo, hi
        
        stack: List[Tuple[str, Tuple[int, int] or str]] = []
        op_seen = 0
        target_lo, target_hi = parse_pair_str(target_reg)
        
        for tok_type, tok_val in rpn:
            if tok_type == 'REG64':
                if not _is_reg_pair(tok_val, 's'):
                    raise ValueError(f"Only SGPR pair supported for i64: {tok_val}")
                stack.append(('PAIR', parse_pair_str(tok_val)))
                continue
            if tok_type in ('REG', 'NUM'):
                stack.append((tok_type, tok_val))
                continue
            if tok_type != 'OP':
                raise ValueError(f"Unexpected token in i64 RPN: {tok_type}")
            if len(stack) < 2:
                raise ValueError(f"Invalid expression: {expr}")
            
            op_seen += 1
            right = stack.pop()
            left = stack.pop()
            
            # 選擇目的暫存器
            if op_seen == op_count:
                dest_lo, dest_hi = target_lo, target_hi
            else:
                dest_lo, dest_hi = alloc_pair()
            
            if tok_val in ('+', '-'):
                l_lo, l_hi = ensure_pair(left)
                r_lo, r_hi = ensure_pair(right)
                if tok_val == '+':
                    instructions.append(f"\ts_add_u32 s{dest_lo}, s{l_lo}, s{r_lo}")
                    instructions.append(f"\ts_addc_u32 s{dest_hi}, s{l_hi}, s{r_hi}")
                else:
                    instructions.append(f"\ts_sub_u32 s{dest_lo}, s{l_lo}, s{r_lo}")
                    instructions.append(f"\ts_subb_u32 s{dest_hi}, s{l_hi}, s{r_hi}")
                stack.append(('PAIR', (dest_lo, dest_hi)))
                continue
            
            if tok_val == '*':
                # 64-bit pair * 32-bit 或 32-bit * 32-bit
                if left[0] == 'PAIR' and right[0] != 'PAIR':
                    l_lo, l_hi = left[1]
                    rhs = ensure_reg32(right)
                    tmp_lo, tmp_hi = alloc_pair()
                    instructions.append(f"\ts_mul_i32 s{dest_lo}, s{l_lo}, {rhs}")
                    instructions.append(f"\ts_mul_hi_u32 s{tmp_lo}, s{l_lo}, {rhs}")
                    instructions.append(f"\ts_mul_i32 s{tmp_hi}, s{l_hi}, {rhs}")
                    instructions.append(f"\ts_add_i32 s{dest_hi}, s{tmp_lo}, s{tmp_hi}")
                    stack.append(('PAIR', (dest_lo, dest_hi)))
                    continue
                if right[0] == 'PAIR' and left[0] != 'PAIR':
                    r_lo, r_hi = right[1]
                    rhs = ensure_reg32(left)
                    tmp_lo, tmp_hi = alloc_pair()
                    instructions.append(f"\ts_mul_i32 s{dest_lo}, s{r_lo}, {rhs}")
                    instructions.append(f"\ts_mul_hi_u32 s{tmp_lo}, s{r_lo}, {rhs}")
                    instructions.append(f"\ts_mul_i32 s{tmp_hi}, s{r_hi}, {rhs}")
                    instructions.append(f"\ts_add_i32 s{dest_hi}, s{tmp_lo}, s{tmp_hi}")
                    stack.append(('PAIR', (dest_lo, dest_hi)))
                    continue
                if left[0] != 'PAIR' and right[0] != 'PAIR':
                    l_reg = ensure_reg32(left)
                    r_reg = ensure_reg32(right)
                    instructions.append(f"\ts_mul_i32 s{dest_lo}, {l_reg}, {r_reg}")
                    instructions.append(f"\ts_mul_hi_u32 s{dest_hi}, {l_reg}, {r_reg}")
                    stack.append(('PAIR', (dest_lo, dest_hi)))
                    continue
                raise ValueError("i64 multiply of two SGPR pairs is not supported")
            
            raise ValueError(f"Unsupported i64 operation: {tok_val}")
        
        return instructions
    
    rpn = to_rpn(tokens)
    op_count = sum(1 for t in rpn if t[0] == 'OP')
    if op_count == 0:
        raise ValueError(f"Unsupported expression format: {expr}")
    
    # 預留足夠的暫存器，若無則退回舊行為
    temp_pool = [] if temp_pool is None else list(temp_pool)
    
    stack: List[Tuple[str, str]] = []
    op_seen = 0
    
    for tok_type, tok_val in rpn:
        if tok_type in ('REG', 'NUM'):
            stack.append((tok_type, tok_val))
            continue
        
        if tok_type != 'OP':
            raise ValueError(f"Unexpected token in RPN: {tok_type}")
        
        if len(stack) < 2:
            raise ValueError(f"Invalid expression: {expr}")
        
        op_seen += 1
        right_type, right_val = stack.pop()
        left_type, left_val = stack.pop()
        
        # 常數折疊
        if left_type == 'NUM' and right_type == 'NUM':
            left_num = float(left_val) if result_type.startswith('f') else int(left_val)
            right_num = float(right_val) if result_type.startswith('f') else int(right_val)
            if tok_val == '+':
                result_num = left_num + right_num
            elif tok_val == '-':
                result_num = left_num - right_num
            elif tok_val == '*':
                result_num = left_num * right_num
            elif tok_val == '/':
                if result_type.startswith('f'):
                    result_num = left_num / right_num
                else:
                    raise ValueError("Integer division is not supported")
            else:
                raise ValueError(f"Unsupported operation: {tok_val}")
            stack.append(('NUM', str(result_num)))
            continue
        
        # 選擇目的暫存器
        if op_seen == op_count:
            dest = target_reg
        else:
            if not temp_pool:
                raise ValueError("Complex expression needs temp VGPRs")
            dest = temp_pool.pop(0)
        
        # 運算元格式化
        left_op = format_const(left_val) if left_type == 'NUM' else left_val
        right_op = format_const(right_val) if right_type == 'NUM' else right_val
        
        if result_type.startswith('f'):
            if tok_val == '*':
                instructions.append(f"\tv_mul_f32 {dest}, {left_op}, {right_op}")
            elif tok_val == '+':
                instructions.append(f"\tv_add_f32 {dest}, {left_op}, {right_op}")
            elif tok_val == '-':
                instructions.append(f"\tv_sub_f32 {dest}, {left_op}, {right_op}")
            elif tok_val == '/':
                instructions.append(f"\tv_div_f32 {dest}, {left_op}, {right_op}")
            else:
                raise ValueError(f"Unsupported operation: {tok_val}")
        else:
            if tok_val == '*':
                instructions.append(f"\tv_mul_lo_u32 {dest}, {left_op}, {right_op}")
            elif tok_val == '+':
                instructions.append(f"\tv_add_u32 {dest}, {left_op}, {right_op}")
            elif tok_val == '-':
                instructions.append(f"\tv_sub_u32 {dest}, {left_op}, {right_op}")
            elif tok_val == '/':
                raise ValueError("Integer division is not supported")
            else:
                raise ValueError(f"Unsupported operation: {tok_val}")
        
        stack.append(('REG', dest))
    
    return instructions


# ============================================================
# 直接在 .s 文件中插入 ISA 指令
# ============================================================

def generate_capture_isa_code(directive: CaptureDirective, unique_id: int,
                              temp_alloc: Optional[Dict[int, Dict[str, int]]] = None
                              ) -> Tuple[List[str], List[CaptureMapping]]:
    """
    生成 capture 的 ISA 指令
    
    Returns:
        (isa_lines, mappings): ISA 指令列表和 register 映射列表
    """
    lines = []
    mappings = []
    
    lines.append(f"; === @CAPTURE #{unique_id} from line {directive.line_number + 1} ===")
    
    # 1. 條件判斷（如果有）
    temp_sgpr_start = None
    temp_vgpr = None
    if directive.condition:
        match = re.match(r'tid_(\w+)\((\d+)\)', directive.condition)
        if match:
            cmp_type, value = match.groups()
            
            cmp_isa_ops = {
                'eq': 'v_cmp_eq_u32_e32',
                'ne': 'v_cmp_ne_u32_e32',
                'lt': 'v_cmp_lt_u32_e32',
                'le': 'v_cmp_le_u32_e32',
                'gt': 'v_cmp_gt_u32_e32',
                'ge': 'v_cmp_ge_u32_e32'
            }
            cmp_instr = cmp_isa_ops.get(cmp_type, 'v_cmp_eq_u32_e32')
            
            # 使用預先分配的暫存器，避免超出宣告範圍
            if temp_alloc and unique_id in temp_alloc:
                temp_sgpr_start = temp_alloc[unique_id]["sgpr_start"]
                temp_vgpr = temp_alloc[unique_id]["vgpr"]
            else:
                raise RuntimeError(f"Missing temp register allocation for @CAPTURE #{unique_id}")
            
            lines.append(f"; Condition: tid_{cmp_type}({value})")
            lines.append(f"; Re-compute thread ID (v0 may have been modified)")
            lines.append(f"\tv_mbcnt_lo_u32_b32 {temp_vgpr}, -1, 0")
            lines.append(f"\tv_mbcnt_hi_u32_b32 {temp_vgpr}, -1, {temp_vgpr}")
            temp_sgpr_end = temp_sgpr_start + 1
            lines.append(f"\ts_mov_b64 s[{temp_sgpr_start}:{temp_sgpr_end}], exec")
            lines.append(f"\t{cmp_instr} vcc, {value}, v{temp_vgpr}")
            lines.append(f"\ts_and_b64 exec, exec, vcc")
    
    # 2. 生成 register 複製指令與表達式計算（依原始順序）
    ordered_values = directive.ordered_values
    if not ordered_values:
        ordered_values = [(r, False) for r in directive.source_registers]
        if directive.expressions:
            ordered_values.extend((e, True) for e in directive.expressions)
    
    expr_temp_pool = []
    if temp_alloc and unique_id in temp_alloc and "expr_vgprs" in temp_alloc[unique_id]:
        expr_temp_pool = [f"v{n}" for n in temp_alloc[unique_id]["expr_vgprs"]]
    expr_sgpr_pool = []
    if temp_alloc and unique_id in temp_alloc and "expr_sgpr_pairs" in temp_alloc[unique_id]:
        expr_sgpr_pool = temp_alloc[unique_id]["expr_sgpr_pairs"]
    
    for (value, is_expr), dst_reg, typ in zip(ordered_values, directive.target_registers, directive.types):
        if not is_expr:
            lines.append(f"; Capture: {value} → {dst_reg}")
            if typ == 'i64':
                if not _is_reg_pair(dst_reg, 's') or not _is_reg_pair(value, 's'):
                    raise ValueError(f"i64 capture requires SGPR pair src/dst: {value} -> {dst_reg}")
                dst_lo, dst_hi = _format_sgpr_pair(dst_reg)
                src_lo, src_hi = _format_sgpr_pair(value)
                lines.append(f"\ts_mov_b32 s{dst_lo}, s{src_lo}")
                lines.append(f"\ts_mov_b32 s{dst_hi}, s{src_hi}")
            else:
                lines.append(f"\tv_mov_b32 {dst_reg}, {value}")
            
            mappings.append(CaptureMapping(
                directive_id=unique_id,
                source=value,
                target_register=dst_reg,
                type_str=typ,
                is_expression=False
            ))
            continue
        
        try:
            lines.append(f"; Expression: {value} → {dst_reg}")
            expr_instructions = compile_expression_to_isa(value, typ, dst_reg, expr_temp_pool, expr_sgpr_pool)
            lines.extend(expr_instructions)
            
            mappings.append(CaptureMapping(
                directive_id=unique_id,
                source=value,
                target_register=dst_reg,
                type_str=typ,
                is_expression=True
            ))
        except Exception as e:
            print(f"[Warning] Failed to compile expression '{value}': {e}")
            # Fallback: 複製第一個 register
            fallback_src = directive.source_registers[0] if directive.source_registers else "v0"
            lines.append(f"; Fallback for failed expression: {value}")
            if typ == 'i64':
                if _is_reg_pair(dst_reg, 's') and _is_reg_pair(fallback_src, 's'):
                    dst_lo, dst_hi = _format_sgpr_pair(dst_reg)
                    src_lo, src_hi = _format_sgpr_pair(fallback_src)
                    lines.append(f"\ts_mov_b32 s{dst_lo}, s{src_lo}")
                    lines.append(f"\ts_mov_b32 s{dst_hi}, s{src_hi}")
                else:
                    dst_lo, dst_hi = _format_sgpr_pair(dst_reg)
                    lines.append(f"\ts_mov_b32 s{dst_lo}, 0")
                    lines.append(f"\ts_mov_b32 s{dst_hi}, 0")
            else:
                lines.append(f"\tv_mov_b32 {dst_reg}, {fallback_src}")
            
            mappings.append(CaptureMapping(
                directive_id=unique_id,
                source=f"{value} (fallback: {fallback_src})",
                target_register=dst_reg,
                type_str=typ,
                is_expression=True
            ))
    
    # 4. 恢復 exec mask（如果有條件）
    if temp_sgpr_start is not None:
        temp_sgpr_end = temp_sgpr_start + 1
        lines.append(f"; Restore exec mask")
        lines.append(f"\ts_mov_b64 exec, s[{temp_sgpr_start}:{temp_sgpr_end}]")
    
    lines.append(f"; === End @CAPTURE #{unique_id} ===")
    
    return lines, mappings


def inject_captures_into_asm(asm_lines: List[str], 
                             directives: List[CaptureDirective],
                             temp_alloc: Optional[Dict[int, Dict[str, int]]] = None
                             ) -> Tuple[List[str], List[CaptureMapping]]:
    """
    在 .s 文件中直接插入 capture ISA 指令
    
    Returns:
        (modified_lines, all_mappings): 修改後的行和所有映射
    """
    if not directives:
        return asm_lines, []
    
    all_mappings = []
    modified_lines = []
    
    # 建立 line_number → directive 的映射
    directive_map = {d.line_number: d for d in directives}
    
    for i, line in enumerate(asm_lines):
        # 先加入原始行
        modified_lines.append(line)
        
        # 如果這行有 @CAPTURE，在其後插入代碼
        if i in directive_map:
            directive = directive_map[i]
            directive_id = directives.index(directive)
            
            capture_lines, mappings = generate_capture_isa_code(directive, directive_id, temp_alloc)
            modified_lines.extend(capture_lines)
            all_mappings.extend(mappings)
    
    return modified_lines, all_mappings


def update_asm_metadata(asm_lines: List[str], new_vgpr_count: int, 
                       new_sgpr_count: int, new_agpr_count: int) -> List[str]:
    """更新 .s 文件中的 metadata"""
    modified_lines = []
    
    for line in asm_lines:
        # 更新 .amdhsa_next_free_vgpr
        if '.amdhsa_next_free_vgpr' in line:
            line = re.sub(r'\.amdhsa_next_free_vgpr\s+\d+', 
                         f'.amdhsa_next_free_vgpr {new_vgpr_count}', line)
        
        # 更新 .amdhsa_next_free_sgpr（避免降低原值）
        if '.amdhsa_next_free_sgpr' in line:
            def _sgpr_next_free_repl(m):
                current = int(m.group(1))
                return f'.amdhsa_next_free_sgpr {max(current, new_sgpr_count)}'
            line = re.sub(r'\.amdhsa_next_free_sgpr\s+(\d+)', _sgpr_next_free_repl, line)
        
        # 更新 YAML 中的 .vgpr_count
        if '.vgpr_count:' in line:
            line = re.sub(r'\.vgpr_count:\s+\d+', 
                         f'.vgpr_count: {new_vgpr_count}', line)
        
        # 更新 YAML 中的 .sgpr_count（避免降低原值）
        if '.sgpr_count:' in line:
            def _sgpr_yaml_repl(m):
                current = int(m.group(1))
                return f'.sgpr_count:     {max(current, new_sgpr_count)}'
            line = re.sub(r'\.sgpr_count:\s+(\d+)', _sgpr_yaml_repl, line)
        
        # 更新 YAML 中的 .agpr_count（避免降低原值）
        if '.agpr_count:' in line:
            def _agpr_yaml_repl(m):
                current = int(m.group(1))
                return f'.agpr_count:     {max(current, new_agpr_count)}'
            line = re.sub(r'\.agpr_count:\s+(\d+)', _agpr_yaml_repl, line)
        
        modified_lines.append(line)
    
    return modified_lines


# ============================================================
# Register Clobber 生成（MLIR）
# ============================================================

def generate_register_clobber(reg_info: Dict[str, int]) -> str:
    """
    生成 register clobber 代碼，保護所有使用的 registers（包括原始 + target）
    
    參考 mdr_printf.py 的實現
    """
    lines = []
    lines.append('              // === Register Clobbering Start ===')
    
    total_vgpr = reg_info['vgpr']
    total_sgpr = reg_info['sgpr']
    total_agpr = reg_info['agpr']
    
    # SGPR 起始位置（跳過系統保留的 s[0:3]）
    sgpr_start = 4
    if total_sgpr > sgpr_start:
        total_sgpr = total_sgpr - sgpr_start
    else:
        total_sgpr = 0
    
    # 計算需要的 block 數量
    num_vgpr_blocks = (total_vgpr + MAX_VECTOR_SIZE - 1) // MAX_VECTOR_SIZE
    num_sgpr_blocks = (total_sgpr + MAX_VECTOR_SIZE - 1) // MAX_VECTOR_SIZE
    num_agpr_blocks = (total_agpr + MAX_VECTOR_SIZE - 1) // MAX_VECTOR_SIZE
    
    # VGPR clobber
    if total_vgpr > 0:
        lines.append(f'              // Protecting {total_vgpr} VGPRs (including original + target registers)')
        for block_idx in range(num_vgpr_blocks):
            start_reg = block_idx * MAX_VECTOR_SIZE
            end_reg = min((block_idx + 1) * MAX_VECTOR_SIZE, total_vgpr) - 1
            block_size = end_reg - start_reg + 1
            lines.append(
                f'              %reserved_vgpr_{block_idx} = llvm.inline_asm has_side_effects asm_dialect = att "", '
                f'"={{v[{start_reg}:{end_reg}]}}": () -> vector<{block_size}xi32>'
            )
    
    # SGPR clobber (跳過系統保留的 s[0:3])
    if total_sgpr > 0:
        lines.append(f'              // Protecting {total_sgpr} SGPRs (s[{sgpr_start}:{sgpr_start + total_sgpr - 1}])')
        for block_idx in range(num_sgpr_blocks):
            start_reg = sgpr_start + block_idx * MAX_VECTOR_SIZE
            end_reg = min(sgpr_start + (block_idx + 1) * MAX_VECTOR_SIZE, sgpr_start + total_sgpr) - 1
            block_size = end_reg - start_reg + 1
            lines.append(
                f'              %reserved_sgpr_{block_idx} = llvm.inline_asm has_side_effects asm_dialect = att "", '
                f'"={{s[{start_reg}:{end_reg}]}}": () -> vector<{block_size}xi32>'
            )
    
    # AGPR clobber
    if total_agpr > 0:
        lines.append(f'              // Protecting {total_agpr} AGPRs')
        for block_idx in range(num_agpr_blocks):
            start_reg = block_idx * MAX_VECTOR_SIZE
            end_reg = min((block_idx + 1) * MAX_VECTOR_SIZE, total_agpr) - 1
            block_size = end_reg - start_reg + 1
            lines.append(
                f'              %reserved_agpr_{block_idx} = llvm.inline_asm has_side_effects asm_dialect = att "", '
                f'"={{a[{start_reg}:{end_reg}]}}": () -> vector<{block_size}xi32>'
            )
    
    lines.append('              // === Register Clobbering End ===')
    
    return '\n'.join(lines)


def inject_clobber_into_mlir(gpumlir_text: str, clobber_code: str) -> str:
    """在 MLIR 中插入 register clobber 代碼"""
    lines = gpumlir_text.split('\n')
    modified_lines = []
    in_func = False
    clobber_inserted = False
    
    for i, line in enumerate(lines):
        # 檢測進入 kernel 函數
        if 'gpu.func @' in line and 'kernel' in line:
            in_func = True
            modified_lines.append(line)
            continue
        
        # 在函數開頭插入 clobber
        if in_func and not clobber_inserted:
            # 在第一個實際指令之前插入
            if 'llvm.inline_asm' in line or 's_endpgm' in line or 'gpu.return' in line:
                modified_lines.append(clobber_code)
                clobber_inserted = True
        
        modified_lines.append(line)
        
        if in_func and 'gpu.return' in line:
            in_func = False
    
    return '\n'.join(modified_lines)


# ============================================================
# Metadata 修復
# ============================================================

def extract_original_metadata(original_isa_file: pathlib.Path) -> Dict:
    """從原始 ISA 檔案中提取 metadata"""
    if not original_isa_file or not original_isa_file.exists():
        return {}
    
    isa_text = original_isa_file.read_text()
    attrs = {}
    original_args = []
    
    # 提取 .amdhsa_* directives
    amdhsa_patterns = {
        'kernarg_size': r'\.amdhsa_kernarg_size\s+(\d+)',
        'group_segment_fixed_size': r'\.amdhsa_group_segment_fixed_size\s+(\d+)',
        'user_sgpr_count': r'\.amdhsa_user_sgpr_count\s+(\d+)',
        'dispatch_ptr': r'\.amdhsa_user_sgpr_dispatch_ptr\s+(\d+)',
        'queue_ptr': r'\.amdhsa_user_sgpr_queue_ptr\s+(\d+)',
        'kernarg_segment_ptr': r'\.amdhsa_user_sgpr_kernarg_segment_ptr\s+(\d+)',
        'next_free_sgpr': r'\.amdhsa_next_free_sgpr\s+(\d+)',
        'reserve_vcc': r'\.amdhsa_reserve_vcc\s+(\d+)',
        'accum_offset': r'\.amdhsa_accum_offset\s+(\d+)',
        'system_sgpr_workgroup_id_x': r'\.amdhsa_system_sgpr_workgroup_id_x\s+(\d+)',
        'system_sgpr_workgroup_id_y': r'\.amdhsa_system_sgpr_workgroup_id_y\s+(\d+)',
        'system_sgpr_workgroup_id_z': r'\.amdhsa_system_sgpr_workgroup_id_z\s+(\d+)',
        'system_sgpr_workgroup_info': r'\.amdhsa_system_sgpr_workgroup_info\s+(\d+)',
        'system_vgpr_workitem_id': r'\.amdhsa_system_vgpr_workitem_id\s+(\d+)',
    }
    
    for attr_name, pattern in amdhsa_patterns.items():
        match = re.search(pattern, isa_text)
        if match:
            attrs[attr_name] = int(match.group(1))
    
    # 提取 YAML metadata
    yaml_match = re.search(r'\.amdgpu_metadata\s+---\s+(.*?)\.\.\.',
                          isa_text, re.DOTALL | re.MULTILINE)
    if yaml_match:
        try:
            yaml_text = yaml_match.group(1)
            metadata = yaml.safe_load(yaml_text)
            
            if 'amdhsa.kernels' in metadata and len(metadata['amdhsa.kernels']) > 0:
                kernel = metadata['amdhsa.kernels'][0]
                
                result = {'attrs': attrs, 'args': original_args, 'yaml_orig_kernel': kernel.copy()}
                
                if '.args' in kernel and isinstance(kernel['.args'], list):
                    for arg in kernel['.args']:
                        arg_dict = {}
                        for k, v in arg.items():
                            key_name = k[1:] if k.startswith('.') else k
                            if key_name not in ['name', 'actual_access']:
                                arg_dict[key_name] = v
                        
                        if arg_dict:
                            original_args.append(arg_dict)
                
                if '.agpr_count' in kernel:
                    attrs['agpr_count'] = kernel['.agpr_count']
            
            print(f"[Info] Extracted {len(attrs)} metadata attributes from {original_isa_file.name}")
            return result
            
        except Exception as e:
            print(f"[Warning] Failed to parse original ISA YAML metadata: {e}")
    
    return {'attrs': attrs, 'args': original_args}


def update_isa_metadata(isa_text: str, new_vgpr_count: int, 
                        new_sgpr_count: int, original_metadata: Dict = None) -> str:
    """更新 ISA metadata"""
    
    # 更新 .amdhsa_next_free_vgpr
    isa_text = re.sub(
        r'(\.amdhsa_next_free_vgpr)\s+\d+',
        rf'\1 {new_vgpr_count}',
        isa_text
    )
    
    # 恢復原始 metadata
    if original_metadata and 'attrs' in original_metadata:
        attrs = original_metadata['attrs']
        
        if 'kernarg_size' in attrs:
            isa_text = re.sub(
                r'(\.amdhsa_kernarg_size)\s+\d+',
                rf'\1 {attrs["kernarg_size"]}',
                isa_text
            )
        
        if 'user_sgpr_count' in attrs:
            isa_text = re.sub(
                r'(\.amdhsa_user_sgpr_count)\s+\d+',
                rf'\1 {attrs["user_sgpr_count"]}',
                isa_text
            )
        
        if 'kernarg_segment_ptr' in attrs:
            isa_text = re.sub(
                r'(\.amdhsa_user_sgpr_kernarg_segment_ptr)\s+\d+',
                rf'\1 {attrs["kernarg_segment_ptr"]}',
                isa_text
            )
        
        if 'next_free_sgpr' in attrs:
            isa_text = re.sub(
                r'(\.amdhsa_next_free_sgpr)\s+\d+',
                rf'\1 {max(attrs["next_free_sgpr"], new_sgpr_count)}',
                isa_text
            )
        
        if 'reserve_vcc' in attrs:
            isa_text = re.sub(
                r'(\.amdhsa_reserve_vcc)\s+\d+',
                rf'\1 {attrs["reserve_vcc"]}',
                isa_text
            )
        
        if 'group_segment_fixed_size' in attrs:
            isa_text = re.sub(
                r'(\.amdhsa_group_segment_fixed_size)\s+\d+',
                rf'\1 {attrs["group_segment_fixed_size"]}',
                isa_text
            )
        
        if 'system_sgpr_workgroup_id_x' in attrs:
            isa_text = re.sub(
                r'(\.amdhsa_system_sgpr_workgroup_id_x)\s+\d+',
                rf'\1 {attrs["system_sgpr_workgroup_id_x"]}',
                isa_text
            )
        
        if 'system_sgpr_workgroup_id_y' in attrs:
            isa_text = re.sub(
                r'(\.amdhsa_system_sgpr_workgroup_id_y)\s+\d+',
                rf'\1 {attrs["system_sgpr_workgroup_id_y"]}',
                isa_text
            )
        
        if 'system_sgpr_workgroup_id_z' in attrs:
            isa_text = re.sub(
                r'(\.amdhsa_system_sgpr_workgroup_id_z)\s+\d+',
                rf'\1 {attrs["system_sgpr_workgroup_id_z"]}',
                isa_text
            )
        
        if 'system_sgpr_workgroup_info' in attrs:
            isa_text = re.sub(
                r'(\.amdhsa_system_sgpr_workgroup_info)\s+\d+',
                rf'\1 {attrs["system_sgpr_workgroup_info"]}',
                isa_text
            )
        
        if 'system_vgpr_workitem_id' in attrs:
            isa_text = re.sub(
                r'(\.amdhsa_system_vgpr_workitem_id)\s+\d+',
                rf'\1 {attrs["system_vgpr_workitem_id"]}',
                isa_text
            )
    
    # 更新 YAML metadata
    yaml_match = re.search(r'\.amdgpu_metadata\s+---\s+(.*?)\.\.\.',
                          isa_text, re.DOTALL | re.MULTILINE)
    if yaml_match:
        try:
            yaml_start = isa_text.find('---', isa_text.find('.amdgpu_metadata'))
            yaml_end = isa_text.find('...', yaml_start)
            yaml_text = isa_text[yaml_start+3:yaml_end].strip()
            
            gen_metadata = yaml.safe_load(yaml_text)
            
            if 'amdhsa.kernels' in gen_metadata and len(gen_metadata['amdhsa.kernels']) > 0:
                kernel = gen_metadata['amdhsa.kernels'][0]
                
                kernel['.vgpr_count'] = new_vgpr_count
                orig_sgpr = 0
                if original_metadata and 'yaml_orig_kernel' in original_metadata:
                    orig_sgpr = original_metadata['yaml_orig_kernel'].get('.sgpr_count', 0)
                kernel['.sgpr_count'] = max(new_sgpr_count, orig_sgpr)
                
                # 恢復原始 args
                if original_metadata and 'args' in original_metadata:
                    original_args = original_metadata['args']
                    
                    kernel['.args'] = []
                    for arg in original_args:
                        yaml_arg = {}
                        if 'address_space' in arg:
                            yaml_arg['.address_space'] = arg['address_space']
                        if 'offset' in arg:
                            yaml_arg['.offset'] = arg['offset']
                        if 'size' in arg:
                            yaml_arg['.size'] = arg['size']
                        if 'value_kind' in arg:
                            yaml_arg['.value_kind'] = arg['value_kind']
                        kernel['.args'].append(yaml_arg)
                
                # 恢復其他欄位
                if original_metadata and 'attrs' in original_metadata:
                    attrs = original_metadata['attrs']
                    
                    if 'kernarg_size' in attrs:
                        kernel['.kernarg_segment_size'] = attrs['kernarg_size']
                    
                    if 'group_segment_fixed_size' in attrs:
                        kernel['.group_segment_fixed_size'] = attrs['group_segment_fixed_size']
                    
                    if '.agpr_count' not in kernel:
                        kernel['.agpr_count'] = attrs.get('agpr_count', 0)
                
                # 恢復其他欄位
                if original_metadata and 'yaml_orig_kernel' in original_metadata:
                    orig_kernel = original_metadata['yaml_orig_kernel']
                    
                    restore_fields = [
                        '.kernarg_segment_size',
                        '.kernarg_segment_align',
                        '.max_flat_workgroup_size',
                        '.language',
                        '.language_version',
                    ]
                    
                    for field in restore_fields:
                        if field in orig_kernel:
                            kernel[field] = orig_kernel[field]
                
                fixed_yaml = yaml.dump(gen_metadata, default_flow_style=False, sort_keys=False)
                
                before_metadata = isa_text[:yaml_start]
                after_metadata = isa_text[yaml_end+3:]
                
                isa_text = before_metadata + "---\n" + fixed_yaml + "..." + after_metadata
        
        except Exception as e:
            print(f"[Warning] Failed to update YAML metadata: {e}")
            import traceback
            traceback.print_exc()
    
    return isa_text


def allocate_temp_registers(directives: List[CaptureDirective],
                            base_reg_info: Dict[str, int]) -> Tuple[Dict[int, Dict[str, int]], Dict[str, int]]:
    """
    為帶條件的 @CAPTURE 分配臨時暫存器。
    會從目前使用量之後開始分配，並回傳更新後的最終使用量。
    """
    temp_alloc: Dict[int, Dict[str, int]] = {}
    final_reg_info = base_reg_info.copy()
    
    next_vgpr = final_reg_info['vgpr']
    next_sgpr = max(final_reg_info['sgpr'], 4)  # 避免使用系統保留 s[0:3]
    
    for directive_id, directive in enumerate(directives):
        entry: Dict[str, int] = {}
        
        if directive.condition:
            entry["vgpr"] = next_vgpr
            entry["sgpr_start"] = next_sgpr
            next_vgpr += 1
            next_sgpr += 2
        
        expr_op_count = 0
        expr_sgpr_pairs_needed = 0
        ordered_values = directive.ordered_values
        if not ordered_values:
            ordered_values = [(r, False) for r in directive.source_registers]
            if directive.expressions:
                ordered_values.extend((e, True) for e in directive.expressions)
        for (value, is_expr), typ in zip(ordered_values, directive.types):
            if not is_expr:
                continue
            if typ == 'i64':
                expr_sgpr_pairs_needed += count_i64_temp_pairs(value)
            else:
                expr_op_count += count_ops_in_expression(value)
        
        if expr_op_count > 0:
            entry["expr_vgprs"] = list(range(next_vgpr, next_vgpr + expr_op_count))
            next_vgpr += expr_op_count
        if expr_sgpr_pairs_needed > 0:
            entry["expr_sgpr_pairs"] = [
                (lo, lo + 1) for lo in range(next_sgpr, next_sgpr + expr_sgpr_pairs_needed * 2, 2)
            ]
            next_sgpr += expr_sgpr_pairs_needed * 2
        
        if entry:
            temp_alloc[directive_id] = entry
    
    final_reg_info['vgpr'] = next_vgpr
    final_reg_info['sgpr'] = next_sgpr
    
    return temp_alloc, final_reg_info


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


def translate_to_gpumlir(asm_path: pathlib.Path, workdir: pathlib.Path) -> pathlib.Path:
    """使用 amdisa-translate 將 .s 轉換為 GPU MLIR"""
    ensure_tool("amdisa-translate")
    
    gpumlir_path = workdir / f"{asm_path.stem}.gpumlir"
    
    print(f"\n=== Stage 2: Translating {asm_path.name} to GPU MLIR ===")
    
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


def build_capture_hsaco(gpumlir_path: pathlib.Path, chip: str, workdir: pathlib.Path,
                        new_vgpr_count: int, new_sgpr_count: int,
                        original_isa_file: pathlib.Path = None):
    """從修改後的 GPU MLIR 生成 HSACO"""
    for tool in ["mlir-opt", "llvm-mc", "lld"]:
        ensure_tool(tool)
    
    stem = gpumlir_path.stem
    binary_mlir = workdir / f"{stem}_binary.mlir"
    isa_path = workdir / f"{stem}_final.s"
    obj_path = workdir / f"{stem}.o"
    hsaco_path = workdir / f"{stem}.hsaco"
    
    # 提取原始 metadata
    original_metadata = extract_original_metadata(original_isa_file) if original_isa_file else {}
    
    print(f"\n=== Stage 4: Running MLIR optimization pipeline ===")
    
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
    
    # 更新 metadata
    isa_text = update_isa_metadata(isa_text, new_vgpr_count, new_sgpr_count, original_metadata)
    
    isa_path.write_text(isa_text)
    print(f"Generated final ISA: {isa_path}")
    
    print(f"\n=== Stage 5: Assembling ISA to object file ===")
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
    
    print(f"\n=== Stage 6: Linking to HSACO ===")
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
    
    print(f"\n✓ Successfully generated capture HSACO: {hsaco_path}")
    return hsaco_path, isa_path


# ============================================================
# 映射文件生成
# ============================================================

def generate_mapping_file(workdir: pathlib.Path, 
                          directives: List[CaptureDirective],
                          mappings: List[CaptureMapping],
                          isa_path: pathlib.Path):
    """生成 register 映射文件"""
    
    mapping_file = workdir / "capture_mapping.txt"
    
    content = []
    content.append("=" * 70)
    content.append("CAPTURE Register Mapping")
    content.append("=" * 70)
    content.append("")
    content.append(f"Generated from: {isa_path.name}")
    content.append(f"Total captures: {len(directives)}")
    content.append(f"Total values: {len(mappings)}")
    content.append("")
    
    # 按 directive 分組
    for directive in directives:
        directive_id = directives.index(directive)
        directive_mappings = [m for m in mappings if m.directive_id == directive_id]
        
        content.append("-" * 70)
        content.append(f"CAPTURE #{directive_id} (Line {directive.line_number + 1})")
        
        if directive.condition:
            content.append(f"  Condition: {directive.condition}")
        
        content.append("")
        
        for mapping in directive_mappings:
            source_label = "Expression" if mapping.is_expression else "Register"
            content.append(f"  {source_label:12} : {mapping.source:20} → {mapping.target_register:6} ({mapping.type_str})")
        
        content.append("")
    
    content.append("=" * 70)
    
    mapping_file.write_text('\n'.join(content))
    print(f"\n✓ Generated mapping file: {mapping_file}")


# ============================================================
# 主程式
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="AMD ISA Assembly Capture Tool (Register-based) - 在組合語言中保存 register 快照",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例：
  1. 在 .s 檔案中標註（建議使用 f-string）：
     ; @CAPTURE f"A={v2}, B={v3}" dst=v10,v11
     ; @CAPTURE if $tid == 0: f"A={v2}, A*2={(v2*2.0):.2f}" dst=v12,v13
     ; @CAPTURE f"addr={(s[12:13] + s[16:17]*4):ld}" dst=s[12:13]

  2. 執行工具：
     python3 mdr_cap.py input.s --output-dir output

  3. 查看映射文件：
     cat output/capture_mapping.txt
        """
    )
    
    ap.add_argument(
        "input_file",
        help="輸入的 .s 組合語言檔案"
    )
    ap.add_argument(
        "--output-dir",
        default="cap_output",
        help="輸出目錄（預設：cap_output）"
    )
    ap.add_argument(
        "--chip",
        default="gfx950",
        help="目標 GPU 架構（預設：gfx950）"
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="只解析和顯示將執行的操作，不執行編譯"
    )
    
    args = ap.parse_args()
    
    input_path = pathlib.Path(args.input_file).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    workdir = pathlib.Path(args.output_dir)
    workdir.mkdir(parents=True, exist_ok=True)
    
    # 讀取 ASM 檔案
    asm_text = input_path.read_text()
    
    print(f"=== AMD ISA Assembly Capture Tool (Register-based) ===")
    print(f"Input: {input_path}")
    print(f"Output: {workdir}")
    print(f"Chip: {args.chip}")
    
    # 1. 解析 @CAPTURE 指令
    print(f"\n=== Stage 1: Parsing @CAPTURE directives ===")
    asm_lines, directives = parse_asm_file(input_path)
    
    if not directives:
        print("[Error] No @CAPTURE directives found")
        return
    
    print(f"Found {len(directives)} @CAPTURE directive(s)")
    
    # 2. 分析暫存器使用量
    print(f"\n=== Analyzing original register usage ===")
    original_reg_info = analyze_registers(asm_text)
    print(f"  Original - VGPR: v0-v{original_reg_info['vgpr']-1} ({original_reg_info['vgpr']} total)")
    print(f"             SGPR: s0-s{original_reg_info['sgpr']-1} ({original_reg_info['sgpr']} total)")
    print(f"             AGPR: a0-a{original_reg_info['agpr']-1} ({original_reg_info['agpr']} total)")
    
    # 3. 收集目標 registers 並計算最終使用量
    all_target_regs = collect_all_target_registers(directives)
    print(f"\n  User-defined target registers: {', '.join(sorted(all_target_regs))}")
    
    final_reg_info = calculate_final_register_usage(original_reg_info, all_target_regs)
    temp_alloc, final_reg_info = allocate_temp_registers(directives, final_reg_info)
    print(f"\n  Final usage (original + targets + temp):")
    print(f"    VGPR: v0-v{final_reg_info['vgpr']-1} ({final_reg_info['vgpr']} total)")
    print(f"    SGPR: s0-s{final_reg_info['sgpr']-1} ({final_reg_info['sgpr']} total)")
    print(f"    AGPR: a0-a{final_reg_info['agpr']-1} ({final_reg_info['agpr']} total)")
    
    # 檢查硬體限制
    if final_reg_info['vgpr'] > 256:
        print(f"\n❌ ERROR: VGPR count exceeds hardware limit (256)!")
        return
    
    if args.dry_run:
        print("\n[Dry Run] Would generate the following operations:")
        print(f"  1. Insert capture ISA instructions at @CAPTURE locations")
        for i, directive in enumerate(directives):
            print(f"     CAPTURE #{i} at line {directive.line_number + 1}:")
            ordered_values = directive.ordered_values
            if not ordered_values:
                ordered_values = [(r, False) for r in directive.source_registers]
                if directive.expressions:
                    ordered_values.extend((e, True) for e in directive.expressions)
            for (value, is_expr), dst, typ in zip(ordered_values, directive.target_registers, directive.types):
                label = "expr" if is_expr else "reg"
                print(f"       {label}: {value} → {dst} ({typ})")
        print(f"  2. Update metadata: VGPR count = {final_reg_info['vgpr']}")
        print(f"  3. Generate injected.s")
        print(f"  4. Convert to MLIR with clobber protection")
        print(f"  5. Build HSACO")
        print("\n[Dry Run] Stopping here.")
        return
    
    # 4. 在 .s 文件中插入 capture ISA 指令
    print(f"\n=== Stage 1: Injecting capture ISA instructions ===")
    modified_asm_lines, all_mappings = inject_captures_into_asm(asm_lines, directives, temp_alloc)
    
    # 5. 更新 metadata
    modified_asm_lines = update_asm_metadata(
        modified_asm_lines,
        final_reg_info['vgpr'],
        final_reg_info['sgpr'],
        final_reg_info['agpr']
    )
    
    # 6. 保存 injected.s
    injected_asm_path = workdir / f"{input_path.stem}_injected.s"
    injected_asm_path.write_text('\n'.join(modified_asm_lines))
    print(f"✓ Generated injected ASM: {injected_asm_path}")
    
    # 7. 轉換為 GPU MLIR
    gpumlir_path = translate_to_gpumlir(injected_asm_path, workdir)
    
    # 8. 在 MLIR 中插入 register clobber
    print(f"\n=== Stage 3: Injecting register clobber into MLIR ===")
    clobber_code = generate_register_clobber(final_reg_info)
    
    gpumlir_text = gpumlir_path.read_text()
    modified_mlir = inject_clobber_into_mlir(gpumlir_text, clobber_code)
    
    clobber_mlir_path = workdir / f"{input_path.stem}_clobber.gpumlir"
    clobber_mlir_path.write_text(modified_mlir)
    print(f"✓ Generated MLIR with clobber: {clobber_mlir_path}")
    
    # 9. 生成 HSACO
    hsaco_path, final_isa_path = build_capture_hsaco(
        clobber_mlir_path, args.chip, workdir,
        final_reg_info['vgpr'], final_reg_info['sgpr'],
        input_path
    )
    
    # 10. 生成映射文件
    generate_mapping_file(workdir, directives, all_mappings, final_isa_path)
    
    print(f"\n=== Done ===")
    print(f"Injected ASM: {injected_asm_path}")
    print(f"Capture HSACO: {hsaco_path}")
    print(f"Final ISA: {final_isa_path}")
    print(f"Mapping file: {workdir}/capture_mapping.txt")


if __name__ == "__main__":
    main()
