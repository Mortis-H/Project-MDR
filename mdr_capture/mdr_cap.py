#!/usr/bin/env python3
"""
AMD ISA Assembly Capture Tool (Register-based)
===============================================

在組合語言中插入暫存器快照功能，將值保存到指定的 register 中。

使用方式：
1. 在 .s 檔案中標註要捕獲的內容：
   ; @CAPTURE src=v2,v3 dst=v10,v11 type=f32,f32
   ; @CAPTURE cond=tid_eq(0) src=v2 dst=v10 expr="v2*2.0" type=f32,f32

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
    
    def __str__(self):
        cond_str = f" [cond={self.condition}]" if self.condition else ""
        if self.expressions:
            return f"@CAPTURE at line {self.line_number + 1}: {self.source_registers} + expr={self.expressions} → {self.target_registers}{cond_str}"
        return f"@CAPTURE at line {self.line_number + 1}: {self.source_registers} → {self.target_registers}{cond_str}"


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
    
    支援兩種格式：
    1. 舊格式（向後兼容）：reg=v2,v3 type=f32,f32
       - 自動使用相同的 register 作為目標（in-place capture）
    2. 新格式：src=v2,v3 dst=v10,v11 type=f32,f32
       - 顯式指定目標 register
    """
    match = re.search(r'[;#]\s*@CAPTURE\s+(.+)', line)
    if not match:
        return None
    
    directive_content = match.group(1).strip()
    
    # 解析各個屬性
    # 舊格式：reg=...
    reg_match = re.search(r'\breg\s*=\s*([\w,\[\]:\s]+?)(?:\s+(?:type|cond|expr|$))', directive_content)
    # 新格式：src=... dst=...
    src_match = re.search(r'\bsrc\s*=\s*([\w,\[\]:\s]+?)(?:\s+(?:dst|type|cond|expr|$))', directive_content)
    dst_match = re.search(r'\bdst\s*=\s*([\w,\[\]:\s]+?)(?:\s+(?:src|type|cond|expr|$))', directive_content)
    
    type_match = re.search(r'type\s*=\s*([\w,\s]+?)(?:\s+(?:reg|src|dst|cond|expr|$)|$)', directive_content)
    cond_match = re.search(r'cond\s*=\s*(\w+\([^)]+\))', directive_content)
    expr_match = re.search(r'expr\s*=\s*"([^"]+)"', directive_content)
    
    if not type_match:
        print(f"[Warning] @CAPTURE missing 'type' at line {line_number + 1}")
        return None
    
    # 解析類型
    type_str = type_match.group(1).strip().rstrip(',')
    types = [t.strip() for t in type_str.split(',')]
    
    # 解析 source 和 target registers
    source_registers = []
    target_registers = []
    
    # 優先使用新格式（src/dst）
    if src_match and dst_match:
        # 新格式：src=... dst=...
        src_str = src_match.group(1).strip().rstrip(',')
        source_registers = [r.strip() for r in src_str.split(',')]
        
        dst_str = dst_match.group(1).strip().rstrip(',')
        target_registers = [r.strip() for r in dst_str.split(',')]
    elif dst_match and not src_match:
        # 只有 dst（可能只有表達式）
        dst_str = dst_match.group(1).strip().rstrip(',')
        target_registers = [r.strip() for r in dst_str.split(',')]
        # source_registers 留空，稍後會檢查是否有表達式
    elif reg_match:
        # 舊格式：reg=... (in-place capture)
        reg_str = reg_match.group(1).strip().rstrip(',')
        registers = [r.strip() for r in reg_str.split(',')]
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
    expressions = []
    if expr_match:
        expr_str = expr_match.group(1).strip()
        expressions = [e.strip() for e in expr_str.split(';')]
    
    # 條件
    condition = cond_match.group(1) if cond_match else None
    
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
        print(f"          Please use new format: src=... dst=... to specify target registers for expressions")
        print(f"          Example: src=v2 dst=v2,v10 expr=\"v2*2.0\" type=f32,f32")
        return None
    
    if total_values != len(target_registers):
        print(f"[Warning] Source/target register count mismatch at line {line_number + 1}: {total_values} sources vs {len(target_registers)} targets")
        return None
    
    return CaptureDirective(
        line_number=line_number,
        source_registers=source_registers,
        target_registers=target_registers,
        types=types,
        condition=condition,
        expressions=expressions if expressions else None
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
        all_targets.update(directive.target_registers)
    return all_targets


def calculate_final_register_usage(original_usage: Dict[str, int], 
                                   target_registers: Set[str]) -> Dict[str, int]:
    """計算包含目標 registers 的最終使用量"""
    final_usage = original_usage.copy()
    
    # 分析目標 registers 的最大編號
    for reg in target_registers:
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


def compile_expression_to_isa(expr: str, result_type: str, target_reg: str) -> List[str]:
    """
    將表達式編譯為 ISA 指令列表
    
    Args:
        expr: 表達式字串（如 "v2*2.0"）
        result_type: 結果類型（f32, i32 等）
        target_reg: 目標 register（如 "v10"）
    
    Returns:
        ISA 指令列表
    """
    tokens = parse_expression(expr)
    instructions = []
    
    # 簡化版：只支持簡單的二元運算（a op b）
    if len(tokens) == 3 and tokens[1][0] == 'OP':
        left_type, left_val = tokens[0]
        op = tokens[1][1]
        right_type, right_val = tokens[2]
        
        # 確定操作數和常數
        if left_type == 'REG' and right_type == 'NUM':
            reg = left_val
            const = right_val
        elif left_type == 'NUM' and right_type == 'REG':
            const = left_val
            reg = right_val
        elif left_type == 'REG' and right_type == 'REG':
            # 兩個 register 的運算
            if result_type.startswith('f'):
                if op == '*':
                    instructions.append(f"\tv_mul_f32 {target_reg}, {left_val}, {right_val}")
                elif op == '+':
                    instructions.append(f"\tv_add_f32 {target_reg}, {left_val}, {right_val}")
                elif op == '-':
                    instructions.append(f"\tv_sub_f32 {target_reg}, {left_val}, {right_val}")
                else:
                    raise ValueError(f"Unsupported operation: {op}")
            else:
                if op == '*':
                    instructions.append(f"\tv_mul_lo_u32 {target_reg}, {left_val}, {right_val}")
                elif op == '+':
                    instructions.append(f"\tv_add_u32 {target_reg}, {left_val}, {right_val}")
                elif op == '-':
                    instructions.append(f"\tv_sub_u32 {target_reg}, {left_val}, {right_val}")
                else:
                    raise ValueError(f"Unsupported operation: {op}")
            return instructions
        else:
            raise ValueError(f"Unsupported expression format: {expr}")
        
        # 生成 ISA 指令（register op constant）
        if result_type.startswith('f'):
            # 浮點運算
            if op == '*':
                instructions.append(f"\tv_mul_f32 {target_reg}, {reg}, {const}")
            elif op == '+':
                instructions.append(f"\tv_add_f32 {target_reg}, {reg}, {const}")
            elif op == '-':
                instructions.append(f"\tv_sub_f32 {target_reg}, {reg}, {const}")
            elif op == '/':
                raise ValueError("Division by constant not yet supported")
            else:
                raise ValueError(f"Unsupported operation: {op}")
        else:
            # 整數運算
            if op == '*':
                instructions.append(f"\tv_mul_lo_u32 {target_reg}, {reg}, {const}")
            elif op == '+':
                instructions.append(f"\tv_add_u32 {target_reg}, {reg}, {const}")
            elif op == '-':
                instructions.append(f"\tv_sub_u32 {target_reg}, {reg}, {const}")
            else:
                raise ValueError(f"Unsupported operation: {op}")
    else:
        raise ValueError(f"Complex expressions not yet supported: {expr}")
    
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
    
    # 2. 生成 register 複製指令
    for i, (src_reg, dst_reg, typ) in enumerate(zip(
        directive.source_registers, 
        directive.target_registers[:len(directive.source_registers)],
        directive.types[:len(directive.source_registers)]
    )):
        lines.append(f"; Capture: {src_reg} → {dst_reg}")
        lines.append(f"\tv_mov_b32 {dst_reg}, {src_reg}")
        
        mappings.append(CaptureMapping(
            directive_id=unique_id,
            source=src_reg,
            target_register=dst_reg,
            type_str=typ,
            is_expression=False
        ))
    
    # 3. 生成表達式計算（如果有）
    if directive.expressions:
        expr_types = directive.types[len(directive.source_registers):]
        expr_targets = directive.target_registers[len(directive.source_registers):]
        
        for i, (expr, dst_reg, typ) in enumerate(zip(directive.expressions, expr_targets, expr_types)):
            try:
                lines.append(f"; Expression: {expr} → {dst_reg}")
                expr_instructions = compile_expression_to_isa(expr, typ, dst_reg)
                lines.extend(expr_instructions)
                
                mappings.append(CaptureMapping(
                    directive_id=unique_id,
                    source=expr,
                    target_register=dst_reg,
                    type_str=typ,
                    is_expression=True
                ))
            except Exception as e:
                print(f"[Warning] Failed to compile expression '{expr}': {e}")
                # Fallback: 複製第一個 register
                fallback_src = directive.source_registers[0] if directive.source_registers else "v0"
                lines.append(f"; Fallback for failed expression: {expr}")
                lines.append(f"\tv_mov_b32 {dst_reg}, {fallback_src}")
                
                mappings.append(CaptureMapping(
                    directive_id=unique_id,
                    source=f"{expr} (fallback: {fallback_src})",
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
        if not directive.condition:
            continue
        
        temp_alloc[directive_id] = {
            "vgpr": next_vgpr,
            "sgpr_start": next_sgpr,
        }
        next_vgpr += 1
        next_sgpr += 2
    
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
  1. 在 .s 檔案中標註：
     ; @CAPTURE src=v2,v3 dst=v10,v11 type=f32,f32
     ; @CAPTURE cond=tid_eq(0) src=v2 dst=v10 expr="v2*2.0" type=f32,f32

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
            for src, dst, typ in zip(directive.source_registers, directive.target_registers, directive.types):
                print(f"       {src} → {dst} ({typ})")
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
