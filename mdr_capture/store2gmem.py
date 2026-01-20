#!/usr/bin/env python3
"""
AMD ISA Assembly Capture Tool
==============================

在組合語言（.s 檔案）中插入暫存器捕獲功能，將資料寫回 host memory。

使用方式：
1. 在 .s 檔案中以註解形式標註要捕獲的內容：
   ; @CAPTURE reg=v6,v7 type=f32,f32
   ; @CAPTURE cond=tid_lt(10) reg=v0,v1,v2 type=i32,i32,i32
   ; @CAPTURE reg=v6 expr="v6*v7" type=f32,f32

2. 執行此工具：
   python3 mdr_capture.py input.s --output-dir output

3. 工具會：
   - 解析 @CAPTURE 標記
   - 轉換為 GPU MLIR（使用 amdisa-translate）
   - 注入 capture 指令（直接寫入 global memory）
   - 修復 ISA metadata（添加 debug_buffer 參數）
   - 生成除錯版本的 HSACO

4. 使用生成的 host wrapper：
   ./output/host_runner input_debug.hsaco

優點：
- 無 hostcall 開銷，效能影響小
- 與 s_barrier 完全相容
- 可捕獲所有 thread 的資料
- 支援大量資料捕獲
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


# ============================================================
# @CAPTURE 標記解析
# ============================================================

@dataclass
class CaptureDirective:
    """
    代表一個 @CAPTURE 指令
    
    支援：
    1. 直接捕獲暫存器：reg=v6,v7 type=f32,f32
    2. 捕獲表達式結果：expr="v6*v7" type=f32
    3. 混合模式：reg=v6 expr="v6*2" type=f32,f32
    """
    line_number: int
    registers: List[str]
    types: List[str]
    condition: Optional[str] = None
    expressions: Optional[List[str]] = None
    buffer_name: str = "debug_buffer"  # 預設 buffer 名稱
    
    def __str__(self):
        cond_str = f" [cond={self.condition}]" if self.condition else ""
        if self.expressions:
            return f"@CAPTURE at line {self.line_number}: {self.registers} + expr={self.expressions}{cond_str}"
        return f"@CAPTURE at line {self.line_number}: {self.registers}{cond_str}"


def parse_capture_directive(line: str, line_number: int) -> Optional[CaptureDirective]:
    """
    解析 @CAPTURE 指令
    
    格式：
    ; @CAPTURE reg=v6,v7 type=f32,f32
    ; @CAPTURE cond=tid_eq(0) reg=v0 type=i32
    ; @CAPTURE reg=v6 expr="v6*v7" type=f32,f32
    """
    match = re.search(r'[;#]\s*@CAPTURE\s+(.+)', line)
    if not match:
        return None
    
    directive_content = match.group(1).strip()
    
    # 解析各個屬性
    reg_match = re.search(r'reg\s*=\s*([\w,\[\]:\s]+?)(?:\s+(?:type|cond|expr|buffer|$))', directive_content)
    type_match = re.search(r'type\s*=\s*([\w,\s]+?)(?:\s+(?:reg|cond|expr|buffer|$)|$)', directive_content)
    cond_match = re.search(r'cond\s*=\s*(\w+\([^)]+\))', directive_content)
    expr_match = re.search(r'expr\s*=\s*"([^"]+)"', directive_content)
    buffer_match = re.search(r'buffer\s*=\s*(\w+)', directive_content)
    
    if not type_match:
        print(f"[Warning] @CAPTURE missing 'type' at line {line_number + 1}")
        return None
    
    # 解析類型列表
    type_str = type_match.group(1).strip().rstrip(',')
    types = [t.strip() for t in type_str.split(',')]
    
    # 解析暫存器列表（可選）
    registers = []
    if reg_match:
        reg_str = reg_match.group(1).strip().rstrip(',')
        registers = [r.strip() for r in reg_str.split(',')]
    
    # 解析表達式（可選）
    expressions = []
    if expr_match:
        expr_str = expr_match.group(1).strip()
        expressions = [e.strip() for e in expr_str.split(';')]
    
    # 條件（可選）
    condition = cond_match.group(1) if cond_match else None
    
    # Buffer 名稱（可選）
    buffer_name = buffer_match.group(1) if buffer_match else "debug_buffer"
    
    # 驗證：必須有 reg 或 expr
    if not registers and not expressions:
        print(f"[Warning] @CAPTURE must have 'reg' or 'expr' at line {line_number + 1}")
        return None
    
    # 驗證數量匹配
    total_values = len(registers) + len(expressions)
    if total_values != len(types):
        print(f"[Warning] Value/type count mismatch at line {line_number + 1}")
        return None
    
    return CaptureDirective(
        line_number=line_number,
        registers=registers,
        types=types,
        condition=condition,
        expressions=expressions if expressions else None,
        buffer_name=buffer_name
    )


def parse_asm_file(asm_path: pathlib.Path) -> Tuple[List[str], List[CaptureDirective], bool]:
    """解析 .s 檔案，提取 @CAPTURE 指令"""
    lines = asm_path.read_text().split('\n')
    directives = []
    has_barrier = False
    
    for i, line in enumerate(lines):
        if 's_barrier' in line and not line.strip().startswith(';'):
            has_barrier = True
        
        if '@CAPTURE' in line:
            directive = parse_capture_directive(line, i)
            if directive:
                directives.append(directive)
                print(f"[Info] Found: {directive}")
    
    return lines, directives, has_barrier


# ============================================================
# 暫存器分析（從 mdr_printf.py 複製）
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


# ============================================================
# 表達式編譯（從 mdr_printf.py 複製核心部分）
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
        
        if ch.isdigit() or (ch == '-' and i + 1 < len(expr) and expr[i + 1].isdigit()):
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


def compile_expression_to_mlir(expr: str, result_type: str, unique_id: int, expr_idx: int) -> Tuple[str, str]:
    """將表達式編譯為 MLIR 代碼"""
    tokens = parse_expression(expr)
    mlir_type = map_type_to_mlir(result_type)
    lines = []
    
    bound_regs = {}
    var_counter = [0]
    
    def get_operand(token_type: str, token_value: str) -> str:
        if token_type == 'REG':
            if token_value not in bound_regs:
                var_name = f'cap_expr_{unique_id}_{expr_idx}_reg_{var_counter[0]}'
                var_counter[0] += 1
                reg_type = token_value[0]
                
                if reg_type == 'v':
                    lines.append(f'              %{var_name} = llvm.inline_asm has_side_effects asm_dialect = att "v_mov_b32 $0, {token_value}", "=v": () -> {mlir_type}')
                elif reg_type == 's':
                    lines.append(f'              %{var_name} = llvm.inline_asm has_side_effects asm_dialect = att "s_mov_b32 $0, {token_value}", "=s": () -> {mlir_type}')
                else:
                    lines.append(f'              %{var_name} = llvm.inline_asm has_side_effects asm_dialect = att "v_accvgpr_read_b32 $0, {token_value}", "=v": () -> {mlir_type}')
                
                bound_regs[token_value] = var_name
            return f'%{bound_regs[token_value]}'
        elif token_type == 'NUM':
            var_name = f'cap_expr_{unique_id}_{expr_idx}_const_{var_counter[0]}'
            var_counter[0] += 1
            lines.append(f'              %{var_name} = arith.constant {token_value} : {mlir_type}')
            return f'%{var_name}'
        else:
            raise ValueError(f"Unexpected token type: {token_type}")
    
    pos = [0]
    
    def parse_primary() -> str:
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
        left = parse_primary()
        
        while pos[0] < len(tokens) and tokens[pos[0]][0] == 'OP' and tokens[pos[0]][1] in '*/':
            op = tokens[pos[0]][1]
            pos[0] += 1
            right = parse_primary()
            
            result_var = f'cap_expr_{unique_id}_{expr_idx}_tmp_{var_counter[0]}'
            var_counter[0] += 1
            
            if result_type.startswith('f'):
                op_name = 'mulf' if op == '*' else 'divf'
            else:
                op_name = 'muli' if op == '*' else 'divi_signed'
            
            lines.append(f'              %{result_var} = arith.{op_name} {left}, {right} : {mlir_type}')
            left = f'%{result_var}'
        
        return left
    
    def parse_add_sub() -> str:
        left = parse_mul_div()
        
        while pos[0] < len(tokens) and tokens[pos[0]][0] == 'OP' and tokens[pos[0]][1] in '+-':
            op = tokens[pos[0]][1]
            pos[0] += 1
            right = parse_mul_div()
            
            result_var = f'cap_expr_{unique_id}_{expr_idx}_tmp_{var_counter[0]}'
            var_counter[0] += 1
            
            if result_type.startswith('f'):
                op_name = 'addf' if op == '+' else 'subf'
            else:
                op_name = 'addi' if op == '+' else 'subi'
            
            lines.append(f'              %{result_var} = arith.{op_name} {left}, {right} : {mlir_type}')
            left = f'%{result_var}'
        
        return left
    
    final_result = parse_add_sub()
    
    if final_result.startswith('%'):
        final_var = final_result[1:]
    else:
        final_var = final_result
    
    return '\n'.join(lines), final_var


# ============================================================
# Capture 生成（核心功能）
# ============================================================

def generate_value_binding(reg: str, type_str: str, var_name: str) -> str:
    """生成讀取暫存器值的 MLIR 程式碼"""
    mlir_type = map_type_to_mlir(type_str)
    
    if reg.startswith('v'):
        asm_instr = f"v_mov_b32 $0, {reg}"
    elif reg.startswith('s'):
        asm_instr = f"v_mov_b32 $0, {reg}"
    else:
        asm_instr = f"v_mov_b32 $0, {reg}"
    
    return f'              %{var_name} = llvm.inline_asm has_side_effects asm_dialect = att "{asm_instr}", "=v": () -> {mlir_type}'


def generate_capture(directive: CaptureDirective, unique_id: int, total_values_per_thread: int, debug_buffer_offset: int = 28, value_offset: int = 0) -> str:
    """
    生成將暫存器值寫入 global memory 的 MLIR 程式碼
    
    策略：
    1. 從 kernarg segment 讀取 debug_buffer 指針（從指定 offset）
    2. 讀取暫存器值（value binding）
    3. 計算表達式（如果有）
    4. 計算寫入位置：base_offset = tid * total_values_per_thread + value_offset
    5. 依序寫入每個值到 debug_buffer[base_offset + i]
    
    Args:
        value_offset: 此 capture 之前已經有多少個值（用於計算全局 offset）
    """
    lines = []
    
    lines.append(f'              // === @CAPTURE from line {directive.line_number + 1} ===')
    
    # 生成從 kernarg segment 讀取 debug_buffer 指針的代碼（只在第一個 capture 時）
    if unique_id == 0:
        lines.append(f'              // Load debug_buffer pointer from kernarg segment at offset {debug_buffer_offset}')
        lines.append(f'              %debug_buffer_ptr = llvm.inline_asm has_side_effects asm_dialect = att "s_load_dwordx2 $0, s[0:1], {debug_buffer_offset}", "=s": () -> i64')
        lines.append(f'              %arg_debug_buffer = llvm.inttoptr %debug_buffer_ptr : i64 to !llvm.ptr')
        lines.append(f'              llvm.inline_asm has_side_effects asm_dialect = att "s_waitcnt lgkmcnt(0)", "" : () -> ()')
    
    var_names = []
    reg_types = directive.types[:len(directive.registers)]
    expr_types = directive.types[len(directive.registers):]
    
    # 1. 生成 value bindings（讀取暫存器）
    for i, (reg, typ) in enumerate(zip(directive.registers, reg_types)):
        var_name = f'cap_val_{unique_id}_{i}'
        var_names.append(var_name)
        lines.append(generate_value_binding(reg, typ, var_name))
    
    # 2. 生成表達式計算（如果有）
    if directive.expressions:
        for i, (expr, typ) in enumerate(zip(directive.expressions, expr_types)):
            try:
                expr_code, result_var = compile_expression_to_mlir(expr, typ, unique_id, i)
                if expr_code:
                    lines.append(f'              // Expression: {expr}')
                    lines.append(expr_code)
                var_names.append(result_var)
            except Exception as e:
                print(f"[Warning] Failed to compile expression '{expr}': {e}")
                fallback_var = f'cap_fallback_{unique_id}_{i}'
                mlir_type = map_type_to_mlir(typ)
                if typ.startswith('f'):
                    lines.append(f'              %{fallback_var} = arith.constant 0.0 : {mlir_type}')
                else:
                    lines.append(f'              %{fallback_var} = arith.constant 0 : {mlir_type}')
                var_names.append(fallback_var)
    
    # 3. 計算每個 thread 的寫入起始位置
    num_values = len(var_names)
    lines.append(f'              // Calculate write offset: tid * {total_values_per_thread} + {value_offset}')
    lines.append(f'              %cap_tid_{unique_id} = gpu.thread_id x')
    lines.append(f'              %cap_total_vals_{unique_id} = arith.constant {total_values_per_thread} : index')
    lines.append(f'              %cap_tid_offset_{unique_id} = arith.muli %cap_tid_{unique_id}, %cap_total_vals_{unique_id} : index')
    if value_offset > 0:
        lines.append(f'              %cap_value_offset_{unique_id} = arith.constant {value_offset} : index')
        lines.append(f'              %cap_base_offset_{unique_id} = arith.addi %cap_tid_offset_{unique_id}, %cap_value_offset_{unique_id} : index')
        base_offset_var = f'cap_base_offset_{unique_id}'
    else:
        # 直接使用 cap_tid_offset 作為 base offset，無需額外操作
        base_offset_var = f'cap_tid_offset_{unique_id}'
    
    # 4. 條件判斷（如果有）- 使用 inline_asm 直接控制 exec mask
    if directive.condition:
        match = re.match(r'tid_(\w+)\((\d+)\)', directive.condition)
        if match:
            cmp_type, value = match.groups()
            
            # 生成比較指令（比較 v0 = thread_id_x 與 value）
            cmp_isa_ops = {
                'eq': 'v_cmp_eq_u32_e32',
                'ne': 'v_cmp_ne_u32_e32',
                'lt': 'v_cmp_lt_u32_e32',
                'le': 'v_cmp_le_u32_e32',
                'gt': 'v_cmp_gt_u32_e32',
                'ge': 'v_cmp_ge_u32_e32'
            }
            cmp_instr = cmp_isa_ops.get(cmp_type, 'v_cmp_eq_u32_e32')
            
            # 使用臨時 SGPR 對（避免與其他代碼衝突）
            temp_sgpr_start = 20 + unique_id * 2
            temp_sgpr_end = temp_sgpr_start + 1
            
            lines.append(f'              // Condition: tid_{cmp_type}({value})')
            # 保存當前 exec mask
            lines.append(f'              llvm.inline_asm has_side_effects asm_dialect = att "s_mov_b64 s[{temp_sgpr_start}:{temp_sgpr_end}], exec", "" : () -> ()')
            # 比較並設置 VCC
            lines.append(f'              llvm.inline_asm has_side_effects asm_dialect = att "{cmp_instr} vcc, {value}, v0", "" : () -> ()')
            # 根據 VCC 更新 exec mask (只有滿足條件的 threads 繼續執行)
            lines.append(f'              llvm.inline_asm has_side_effects asm_dialect = att "s_and_b64 exec, exec, vcc", "" : () -> ()')
            indent = '              '
            has_condition = True
        else:
            indent = '              '
            has_condition = False
    else:
        indent = '              '
        has_condition = False
    
    # 5. 依序寫入每個值
    for i, (var_name, typ) in enumerate(zip(var_names, directive.types)):
        mlir_type = map_type_to_mlir(typ)
        
        # 計算此值的 offset
        if i == 0:
            # 第一個值直接使用 base offset
            offset_var = base_offset_var
        else:
            # 後續值需要加上偏移量
            lines.append(f'{indent}%cap_i_{unique_id}_{i} = arith.constant {i} : index')
            lines.append(f'{indent}%cap_offset_{unique_id}_{i} = arith.addi %{base_offset_var}, %cap_i_{unique_id}_{i} : index')
            offset_var = f'cap_offset_{unique_id}_{i}'
        
        # 使用 LLVM GEP + store 寫入（比 inline_asm 更可靠）
        lines.append(f'{indent}%cap_offset_i64_{unique_id}_{i} = arith.index_cast %{offset_var} : index to i64')
        lines.append(f'{indent}%cap_ptr_{unique_id}_{i} = llvm.getelementptr %arg_debug_buffer[%cap_offset_i64_{unique_id}_{i}] : (!llvm.ptr, i64) -> !llvm.ptr, {mlir_type}')
        lines.append(f'{indent}llvm.store %{var_name}, %cap_ptr_{unique_id}_{i} : {mlir_type}, !llvm.ptr')
    
    # 6. 結束條件區塊（如果有）- 恢復 exec mask
    if has_condition:
        # 恢復原始 exec mask
        temp_sgpr_start = 20 + unique_id * 2
        temp_sgpr_end = temp_sgpr_start + 1
        lines.append(f'              // Restore exec mask')
        lines.append(f'              llvm.inline_asm has_side_effects asm_dialect = att "s_mov_b64 exec, s[{temp_sgpr_start}:{temp_sgpr_end}]", "" : () -> ()')
    
    lines.append(f'              // === End @CAPTURE ===')
    
    return '\n'.join(lines)


def inject_capture_into_mlir(gpumlir_text: str, directives: List[CaptureDirective], reg_info: Dict[str, int]) -> str:
    """
    將 capture 指令注入到 GPU MLIR 中
    
    與 printf 的主要差異：
    1. 不需要 hostcall，所以不保存/恢復 kernarg pointer
    2. 需要添加 debug_buffer 參數到 kernel 簽名
    3. 使用 llvm.store 寫入，而非 gpu.printf
    """
    if not directives:
        return gpumlir_text
    
    # 計算每個 thread 需要寫入的值的總數
    total_values = sum(len(d.registers) + (len(d.expressions) if d.expressions else 0) for d in directives)
    
    print(f"[Info] Each thread will capture {total_values} values")
    
    # === Register Clobbering（與 mdr_printf.py 相同）===
    MAX_VECTOR_SIZE = 32
    SGPR_RESERVED_START = 4
    
    original_vgpr = reg_info.get('vgpr', 0)
    original_sgpr = reg_info.get('sgpr', 0)
    original_agpr = reg_info.get('agpr', 0)
    
    total_vgpr = max(1, original_vgpr) if original_vgpr > 0 else 0
    total_agpr = max(1, original_agpr) if original_agpr > 0 else 0
    
    sgpr_start = SGPR_RESERVED_START
    sgpr_count_needed = max(original_sgpr, 20) - sgpr_start
    
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
    
    total_sgpr = next_power_of_2(sgpr_count_needed) if sgpr_count_needed > 0 else 0
    total_sgpr = min(total_sgpr, MAX_VECTOR_SIZE)
    
    num_vgpr_blocks = (total_vgpr + MAX_VECTOR_SIZE - 1) // MAX_VECTOR_SIZE if total_vgpr > 0 else 0
    num_sgpr_blocks = (total_sgpr + MAX_VECTOR_SIZE - 1) // MAX_VECTOR_SIZE if total_sgpr > 0 else 0
    num_agpr_blocks = (total_agpr + MAX_VECTOR_SIZE - 1) // MAX_VECTOR_SIZE if total_agpr > 0 else 0
    
    print(f"[Info] Register clobbering:")
    if total_vgpr > 0:
        print(f"       VGPR: v[0:{total_vgpr - 1}] ({total_vgpr} registers, {num_vgpr_blocks} block(s))")
    if total_sgpr > 0:
        print(f"       SGPR: s[{sgpr_start}:{sgpr_start + total_sgpr - 1}] ({total_sgpr} registers, {num_sgpr_blocks} block(s))")
    
    # === 生成 clobber 開始 ===
    clobber_start_lines = ['              // === Register Clobbering Start ===']
    
    # Capture 不需要保存 kernarg pointer（因為不用 hostcall）
    
    if total_vgpr > 0:
        clobber_start_lines.append(f'              // Protecting {total_vgpr} VGPRs')
        for block_idx in range(num_vgpr_blocks):
            start_reg = block_idx * MAX_VECTOR_SIZE
            end_reg = min((block_idx + 1) * MAX_VECTOR_SIZE, total_vgpr) - 1
            block_size = end_reg - start_reg + 1
            clobber_start_lines.append(
                f'              %reserved_vgpr_{block_idx} = llvm.inline_asm has_side_effects asm_dialect = att "", '
                f'"={{v[{start_reg}:{end_reg}]}}": () -> vector<{block_size}xi32>'
            )
    
    if total_sgpr > 0:
        clobber_start_lines.append(f'              // Protecting {total_sgpr} SGPRs')
        for block_idx in range(num_sgpr_blocks):
            start_reg = sgpr_start + block_idx * MAX_VECTOR_SIZE
            end_reg = min(sgpr_start + (block_idx + 1) * MAX_VECTOR_SIZE, sgpr_start + total_sgpr) - 1
            block_size = end_reg - start_reg + 1
            clobber_start_lines.append(
                f'              %reserved_sgpr_{block_idx} = llvm.inline_asm has_side_effects asm_dialect = att "", '
                f'"={{s[{start_reg}:{end_reg}]}}": () -> vector<{block_size}xi32>'
            )
    
    clobber_start_lines.append('              // === End Clobbering Start ===')
    clobber_start = '\n'.join(clobber_start_lines) + '\n'
    
    # === 生成 capture 區塊 ===
    # 計算 debug_buffer 的 offset（從原始 ISA metadata 中獲取）
    # 假設原始 kernel 有 N 個參數，debug_buffer 是最後一個參數
    # 從 directives 中我們知道原始參數的信息，debug_buffer offset = 最後一個參數的 offset + size
    # 但更簡單的方法是直接使用 28 (0x1c)，因為這是在 fix_isa_metadata 中計算的
    debug_buffer_offset = 28  # 將在後續改進中從 metadata 動態計算
    
    capture_blocks = []
    value_offset = 0  # 追蹤前面已經有多少個值
    for i, directive in enumerate(directives):
        num_values = len(directive.registers) + (len(directive.expressions) if directive.expressions else 0)
        capture_blocks.append(generate_capture(directive, i, total_values, debug_buffer_offset, value_offset))
        value_offset += num_values
    
    capture_section = '\n'.join(capture_blocks)
    
    # === 生成 clobber 結束 ===
    clobber_end_lines = ['              // === Register Clobbering End ===']
    
    if total_vgpr > 0:
        for block_idx in range(num_vgpr_blocks):
            start_reg = block_idx * MAX_VECTOR_SIZE
            end_reg = min((block_idx + 1) * MAX_VECTOR_SIZE, total_vgpr) - 1
            block_size = end_reg - start_reg + 1
            clobber_end_lines.append(
                f'              llvm.inline_asm has_side_effects asm_dialect = att "", '
                f'"{{v[{start_reg}:{end_reg}]}}" %reserved_vgpr_{block_idx} : (vector<{block_size}xi32>)-> ()'
            )
    
    if total_sgpr > 0:
        for block_idx in range(num_sgpr_blocks):
            start_reg = sgpr_start + block_idx * MAX_VECTOR_SIZE
            end_reg = min(sgpr_start + (block_idx + 1) * MAX_VECTOR_SIZE, sgpr_start + total_sgpr) - 1
            block_size = end_reg - start_reg + 1
            clobber_end_lines.append(
                f'              llvm.inline_asm has_side_effects asm_dialect = att "", '
                f'"{{s[{start_reg}:{end_reg}]}}" %reserved_sgpr_{block_idx} : (vector<{block_size}xi32>)-> ()'
            )
    
    clobber_end_lines.append('              // === End Clobbering End ===')
    clobber_end = '\n'.join(clobber_end_lines) + '\n'
    
    # === 注意：不修改 kernel 簽名 ===
    # 我們通過手動從 kernarg segment 讀取 debug_buffer 指針，
    # 而不是添加函數參數，因為 ROCDL lowering 會將其映射到錯誤的 offset
    
    # === 插入 clobber 和 capture 程式碼 ===
    lines = gpumlir_text.split('\n')
    modified_lines = []
    in_func = False
    clobber_inserted = False
    capture_inserted = False
    
    has_lbb0_2_label = any('.LBB0_2:' in line or '.LBB0_2"' in line for line in lines)
    
    for i, line in enumerate(lines):
        if 'gpu.func @' in line and 'kernel' in line:
            in_func = True
        
        if in_func and not clobber_inserted and 'llvm.inline_asm' in line:
            modified_lines.append(clobber_start)
            clobber_inserted = True
        
        is_lbb0_2_label = '.LBB0_2:' in line or ('.LBB0_2"' in line and 's_cbranch' not in line)
        if in_func and not capture_inserted and has_lbb0_2_label and is_lbb0_2_label:
            # 先添加 label 行（不要跳過！）
            modified_lines.append(line)
            # 然後插入 capture 代碼
            modified_lines.append(capture_section)
            modified_lines.append(clobber_end)
            capture_inserted = True
            # 已經處理了這一行，繼續下一行
            continue
        
        if in_func and not capture_inserted and not has_lbb0_2_label and 's_endpgm' in line:
            modified_lines.append(capture_section)
            modified_lines.append(clobber_end)
            capture_inserted = True
        
        modified_lines.append(line)
        
        if in_func and 'gpu.return' in line:
            in_func = False
    
    return '\n'.join(modified_lines)


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
    
    gpumlir_path = workdir / f"{asm_path.stem}_capture.gpumlir"
    
    print(f"\n=== Stage 1: Translating {asm_path.name} to GPU MLIR ===")
    
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


def fix_isa_metadata(isa_text: str, original_isa_file: pathlib.Path, has_printf: bool = True) -> str:
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
    
    # 提取 .amdhsa_* directives
    amdhsa_patterns = {
        'kernarg_segment_size': r'\.amdhsa_kernarg_size\s+(\d+)',
        'group_segment_fixed_size': r'\.amdhsa_group_segment_fixed_size\s+(\d+)',
        'user_sgpr_count': r'\.amdhsa_user_sgpr_count\s+(\d+)',
        'dispatch_ptr': r'\.amdhsa_user_sgpr_dispatch_ptr\s+(\d+)',
        'queue_ptr': r'\.amdhsa_user_sgpr_queue_ptr\s+(\d+)',
        'kernarg_segment_ptr': r'\.amdhsa_user_sgpr_kernarg_segment_ptr\s+(\d+)',  # 關鍵！
        'workitem_id': r'\.amdhsa_system_vgpr_workitem_id\s+(\d+)',
        'workgroup_id_x': r'\.amdhsa_system_sgpr_workgroup_id_x\s+(\d+)',
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
            new_args.extend(gen_args[hidden_start_idx:])
            
            kernel['.args'] = new_args
            
            # 檢查是否有 hidden_hostcall_buffer
            has_hostcall = any(arg.get('.value_kind') == 'hidden_hostcall_buffer' for arg in new_args)
            print(f"[Info] Merged {len(original_non_hidden_args)} original args + {len(gen_args) - hidden_start_idx} hidden args")
            if has_hostcall:
                print(f"[Info] ✓ hidden_hostcall_buffer preserved for printf support")
            else:
                print(f"[Warning] hidden_hostcall_buffer NOT found - printf may fail!")
        
        elif original_all_args:
            # 無 printf，但需要為 capture 添加 debug_buffer 參數
            kernel['.args'] = []
            
            # 添加原始的所有參數（注意：不包含 .name 字段，因為會導致 llvm-mc 報錯）
            for arg in original_all_args:
                yaml_arg = {}
                if 'address_space' in arg:
                    yaml_arg['.address_space'] = arg['address_space']
                if 'offset' in arg:
                    yaml_arg['.offset'] = arg['offset']
                if 'size' in arg:
                    yaml_arg['.size'] = arg['size']
                if 'value_kind' in arg:
                    yaml_arg['.value_kind'] = arg['value_kind']
                # 注意：不添加 'name' 字段，llvm-mc 無法處理它
                kernel['.args'].append(yaml_arg)
            
            # 計算 debug_buffer 的 offset（所有原始參數的總大小）
            max_offset = 0
            max_size = 0
            for arg in original_all_args:
                if 'offset' in arg:
                    offset = arg['offset']
                    size = arg.get('size', 0)
                    if offset + size > max_offset + max_size:
                        max_offset = offset
                        max_size = size
            
            debug_buffer_offset = max_offset + max_size
            
            # 添加 debug_buffer 參數（用於 capture，不包含 .name 字段）
            kernel['.args'].append({
                '.address_space': 'global',
                '.offset': debug_buffer_offset,
                '.size': 8,
                '.value_kind': 'global_buffer'
                # 注意：不添加 .name 字段
            })
            
            # 更新 kernarg_segment_size
            new_kernarg_size = debug_buffer_offset + 8
            kernel['.kernarg_segment_size'] = new_kernarg_size
            
            print(f"[Info] Restored {len(original_all_args)} original args + 1 debug_buffer arg")
            print(f"[Info] debug_buffer offset: {debug_buffer_offset}, total kernarg_size: {new_kernarg_size}")
    
    # 重新生成 YAML
    fixed_yaml = yaml.dump(gen_metadata, default_flow_style=False, sort_keys=False)
    
    # 替換 ISA 中的 metadata
    before_metadata = isa_text[:yaml_start]
    after_metadata = isa_text[yaml_end+3:]  # 跳過原始的 '...'（3 個字符）
    
    # 構建修復後的 ISA
    fixed_isa = before_metadata + "---\n" + fixed_yaml + "..." + after_metadata
    
    # 獲取更新後的 kernarg_segment_size（從 metadata 中）
    updated_kernarg_size = kernel.get('.kernarg_segment_size', attrs.get('kernarg_segment_size', 0))
    
    # 同時修復 .amdhsa_* 指令
    fixed_isa = re.sub(
        r'(\.amdhsa_kernarg_size)\s+\d+',
        rf'\1 {updated_kernarg_size}',
        fixed_isa
    )
    
    if 'group_segment_fixed_size' in attrs:
        fixed_isa = re.sub(
            r'(\.amdhsa_group_segment_fixed_size)\s+\d+',
            rf'\1 {attrs["group_segment_fixed_size"]}',
            fixed_isa
        )
    
    if 'user_sgpr_count' in attrs and attrs['user_sgpr_count'] > 0:
        fixed_isa = re.sub(
            r'(\.amdhsa_user_sgpr_count)\s+\d+',
            rf'\1 {attrs["user_sgpr_count"]}',
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
    
    return fixed_isa


def build_capture_hsaco(gpumlir_path: pathlib.Path, chip: str, workdir: pathlib.Path, 
                        original_isa_file: pathlib.Path = None):
    """從修改後的 GPU MLIR 生成 HSACO"""
    for tool in ["mlir-opt", "llvm-mc", "lld"]:
        ensure_tool(tool)
    
    stem = gpumlir_path.stem
    binary_mlir = workdir / f"{stem}_binary.mlir"
    isa_path = workdir / f"{stem}.s"
    obj_path = workdir / f"{stem}.o"
    hsaco_path = workdir / f"{stem}.hsaco"
    
    print(f"\n=== Stage 2: Running MLIR optimization pipeline ===")
    
    # 使用 runtime=HIP（參考 pipeline.py）
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
    
    # 修復 ISA metadata（has_printf=False 因為我們用的是 capture 而非 printf）
    if original_isa_file is not None:
        isa_text = fix_isa_metadata(isa_text, original_isa_file, has_printf=False)
    
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
    
    print(f"\n✓ Successfully generated capture HSACO: {hsaco_path}")
    return hsaco_path


def detect_kernel_info(asm_text: str) -> Tuple[Optional[str], Optional[str]]:
    """從 ISA 程式碼自動偵測 kernel 名稱和類型"""
    kernel_name = None
    kernel_type = None
    
    kernel_match = re.search(r'\.globl\s+(\S+)', asm_text)
    if kernel_match:
        kernel_name = kernel_match.group(1)
        if kernel_name.startswith('__hip_cuid'):
            all_matches = re.findall(r'\.globl\s+(\S+)', asm_text)
            for name in all_matches:
                if not name.startswith('__hip_cuid'):
                    kernel_name = name
                    break
    
    if kernel_name:
        name_lower = kernel_name.lower()
        if 'vectoradd' in name_lower:
            kernel_type = 'float_add'
        elif 'vectormul' in name_lower:
            kernel_type = 'float_mul'
        elif 'scalarops' in name_lower:
            kernel_type = 'int_scalar'
        elif 'memoryops' in name_lower:
            kernel_type = 'int_mem'
    
    return kernel_name, kernel_type


# ============================================================
# 主程式
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="AMD ISA Assembly Capture Tool - 在組合語言中插入資料捕獲功能",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例：
  1. 在 .s 檔案中標註：
     ; @CAPTURE reg=v6,v7 type=f32,f32
     ; @CAPTURE cond=tid_lt(10) reg=v0 type=i32

  2. 執行工具：
     python3 mdr_capture.py input.s --output-dir output

  3. 工具會生成：
     - output/input_capture.hsaco (修改後的 kernel)
     - output/host_runner.cpp (範例 host 程式碼)
        """
    )
    
    ap.add_argument(
        "input_file",
        help="輸入的 .s 組合語言檔案"
    )
    ap.add_argument(
        "--output-dir",
        default="capture_output",
        help="輸出目錄（預設：capture_output）"
    )
    ap.add_argument(
        "--chip",
        default="gfx950",
        help="目標 GPU 架構（預設：gfx950）"
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="只解析 @CAPTURE 指令，不執行編譯"
    )
    ap.add_argument(
        "--generate-host",
        action="store_true",
        help="生成完整的 host 程式碼（快速測試用，僅支援 vectorAdd）"
    )
    ap.add_argument(
        "--generate-skeleton",
        action="store_true",
        help="生成 host code 骨架（需要用戶填寫參數）"
    )
    ap.add_argument(
        "--inject-host",
        metavar="TEMPLATE",
        help="將 capture 程式碼注入到現有的 host template"
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
    kernel_name, kernel_type = detect_kernel_info(asm_text)
    
    print(f"=== AMD ISA Assembly Capture Tool ===")
    print(f"Input: {input_path}")
    print(f"Output: {workdir}")
    print(f"Chip: {args.chip}")
    if kernel_name:
        print(f"Kernel Name: {kernel_name}")
    if kernel_type:
        print(f"Kernel Type: {kernel_type}")
    
    # 1. 解析 @CAPTURE 指令
    print(f"\n=== Parsing @CAPTURE directives ===")
    lines, directives, has_barrier = parse_asm_file(input_path)
    
    if not directives:
        print("[Error] No @CAPTURE directives found")
        return
    
    print(f"\nFound {len(directives)} @CAPTURE directive(s)")
    
    # 計算總共需要捕獲的值數量
    total_values = sum(len(d.registers) + (len(d.expressions) if d.expressions else 0) for d in directives)
    print(f"[Info] Total values per thread: {total_values}")
    
    if has_barrier:
        print("[Info] ✓ Kernel contains s_barrier (Capture 完全相容)")
    
    if args.dry_run:
        print("\n[Dry Run] Stopping here.")
        return
    
    # 2. 分析暫存器使用量
    print(f"\n=== Analyzing register usage ===")
    reg_info = analyze_registers(asm_text)
    print(f"  VGPR: {reg_info['vgpr']}, SGPR: {reg_info['sgpr']}, AGPR: {reg_info['agpr']}")
    
    # 3. 轉換為 GPU MLIR
    gpumlir_path = translate_to_gpumlir(input_path, workdir)
    
    # 4. 注入 capture 程式碼
    print(f"\n=== Injecting capture code ===")
    gpumlir_text = gpumlir_path.read_text()
    modified_mlir = inject_capture_into_mlir(gpumlir_text, directives, reg_info)
    
    modified_path = workdir / f"{input_path.stem}_capture_injected.gpumlir"
    modified_path.write_text(modified_mlir)
    print(f"Generated modified GPU MLIR: {modified_path}")
    
    # 5. 生成 HSACO
    hsaco_path = build_capture_hsaco(modified_path, args.chip, workdir, input_path)
    
    print(f"\n=== Done ===")
    print(f"Capture HSACO: {hsaco_path}")
    print(f"Values per thread: {total_values}")
    
    # 6. Host code 生成/注入
    if args.generate_host:
        # 舊方式：生成完整的 host code（快速測試）
        generate_host_code(workdir, hsaco_path, kernel_name, kernel_type, total_values)
    elif args.generate_skeleton:
        # 新方式：生成骨架
        generate_host_skeleton(workdir, hsaco_path, kernel_name, total_values)
    elif args.inject_host:
        # 新方式：注入到現有 template
        inject_capture_to_host(args.inject_host, workdir, hsaco_path, kernel_name, total_values)


def generate_host_code(workdir: pathlib.Path, hsaco_path: pathlib.Path, 
                       kernel_name: str, kernel_type: str, values_per_thread: int):
    """生成範例 host 程式碼"""
    host_code = f"""#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <fstream>

#define HIP_CHECK(call) \\
    do {{ \\
        hipError_t err = call; \\
        if (err != hipSuccess) {{ \\
            std::cerr << "HIP Error: " << hipGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \\
            exit(1); \\
        }} \\
    }} while(0)

int main(int argc, char** argv) {{
    // 參數
    const int N = (argc > 1) ? std::atoi(argv[1]) : 256;
    const int values_per_thread = {values_per_thread};
    
    std::cout << "=== Capture Test ===" << std::endl;
    std::cout << "Threads: " << N << std::endl;
    std::cout << "Values per thread: " << values_per_thread << std::endl;
    
    // 分配 host memory
    std::vector<float> h_A(N), h_B(N), h_C(N);
    std::vector<float> h_debug(N * values_per_thread, -999.0f);  // 初始化為特殊值
    
    // 初始化輸入
    for (int i = 0; i < N; i++) {{
        h_A[i] = i + 1.0f;
        h_B[i] = 2.0f * (i + 1.0f);
        h_C[i] = 0.0f;
    }}
    
    // 分配 device memory
    float *d_A, *d_B, *d_C, *d_debug;
    HIP_CHECK(hipMalloc(&d_A, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_B, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_C, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_debug, N * values_per_thread * sizeof(float)));
    
    // 複製到 device
    HIP_CHECK(hipMemcpy(d_A, h_A.data(), N * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B.data(), N * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_C, h_C.data(), N * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_debug, h_debug.data(), N * values_per_thread * sizeof(float), hipMemcpyHostToDevice));
    
    // 載入 HSACO
    hipModule_t module;
    hipFunction_t kernel;
    HIP_CHECK(hipModuleLoad(&module, "{hsaco_path.name}"));
    HIP_CHECK(hipModuleGetFunction(&kernel, module, "{kernel_name}"));
    
    // 準備 kernel 參數（包含 debug_buffer）
    int n_val = N;
    void* args[] = {{&d_A, &d_B, &d_C, &n_val, &d_debug}};
    
    // 啟動 kernel
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
    
    std::cout << "\\nLaunching kernel..." << std::endl;
    HIP_CHECK(hipModuleLaunchKernel(kernel,
        gridSize.x, gridSize.y, gridSize.z,
        blockSize.x, blockSize.y, blockSize.z,
        0, nullptr, args, nullptr));
    
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "Kernel finished." << std::endl;
    
    // 複製結果回 host
    HIP_CHECK(hipMemcpy(h_C.data(), d_C, N * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_debug.data(), d_debug, N * values_per_thread * sizeof(float), hipMemcpyDeviceToHost));
    
    // 顯示計算結果（前 10 個）
    std::cout << "\\n=== Computation Results (first 10) ===" << std::endl;
    for (int i = 0; i < std::min(10, N); i++) {{
        std::cout << "C[" << i << "] = " << h_C[i] << std::endl;
    }}
    
    // 顯示 captured 資料（前 10 個 thread）
    std::cout << "\\n=== Captured Debug Data (first 10 threads) ===" << std::endl;
    for (int tid = 0; tid < std::min(10, N); tid++) {{
        std::cout << "Thread " << tid << ": ";
        for (int v = 0; v < values_per_thread; v++) {{
            std::cout << h_debug[tid * values_per_thread + v];
            if (v < values_per_thread - 1) std::cout << ", ";
        }}
        std::cout << std::endl;
    }}
    
    // 統計分析
    std::cout << "\\n=== Statistics ===" << std::endl;
    int non_default_count = 0;
    for (const auto& val : h_debug) {{
        if (val != -999.0f) non_default_count++;
    }}
    std::cout << "Non-default values: " << non_default_count << " / " << h_debug.size() << std::endl;
    
    // 將完整資料寫入檔案
    std::ofstream outfile("debug_data.txt");
    for (int tid = 0; tid < N; tid++) {{
        outfile << tid;
        for (int v = 0; v < values_per_thread; v++) {{
            outfile << " " << h_debug[tid * values_per_thread + v];
        }}
        outfile << "\\n";
    }}
    outfile.close();
    std::cout << "Full data saved to debug_data.txt" << std::endl;
    
    // 清理
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));
    HIP_CHECK(hipFree(d_debug));
    HIP_CHECK(hipModuleUnload(module));
    
    std::cout << "\\n✓ Test completed successfully!" << std::endl;
    
    return 0;
}}
"""
    
    host_cpp_path = workdir / "host_runner.cpp"
    host_cpp_path.write_text(host_code)
    print(f"\n✓ Generated host code: {host_cpp_path}")
    print(f"\n編譯與執行：")
    print(f"  cd {workdir}")
    print(f"  hipcc host_runner.cpp -o host_runner")
    print(f"  ./host_runner 256")


def generate_host_skeleton(workdir: pathlib.Path, hsaco_path: pathlib.Path, 
                           kernel_name: str, values_per_thread: int):
    """生成 host code 骨架（需要用戶填寫）"""
    
    skeleton = f"""#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <fstream>

#define HIP_CHECK(call) \\
    do {{ \\
        hipError_t err = call; \\
        if (err != hipSuccess) {{ \\
            std::cerr << "HIP Error: " << hipGetErrorString(err) \\
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \\
            exit(1); \\
        }} \\
    }} while(0)

int main(int argc, char** argv) {{
    const int N = (argc > 1) ? std::atoi(argv[1]) : 256;
    
    // ========================================
    // TODO: 配置您的 kernel 參數和資料
    // ========================================
    
    // 範例（請根據您的 kernel 修改）：
    // float *d_input, *d_output;
    // HIP_CHECK(hipMalloc(&d_input, N * sizeof(float)));
    // HIP_CHECK(hipMalloc(&d_output, N * sizeof(float)));
    // 
    // std::vector<float> h_input(N);
    // for (int i = 0; i < N; i++) h_input[i] = i;
    // HIP_CHECK(hipMemcpy(d_input, h_input.data(), 
    //                     N * sizeof(float), hipMemcpyHostToDevice));
    
    // ===== CAPTURE_INJECT_ALLOC =====
    // ⚠️ 請保持這個標記！mdr_capture.py 會在這裡自動注入 debug buffer 分配
    
    // 載入 HSACO
    hipModule_t module;
    hipFunction_t kernel;
    HIP_CHECK(hipModuleLoad(&module, "{hsaco_path.name}"));
    HIP_CHECK(hipModuleGetFunction(&kernel, module, "{kernel_name}"));
    
    // ========================================
    // TODO: 準備 kernel 參數（重要！）
    // ========================================
    
    // ⚠️ 注意：mdr_capture.py 會自動在 args 陣列末尾添加 &d_debug
    // 您只需要列出原始的 kernel 參數，不要手動添加 d_debug
    
    // 範例（請根據您的 kernel 修改）：
    // int n_val = N;
    // void* args[] = {{&d_input, &d_output, &n_val}};
    //                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    //                列出您的原始參數（末尾不要加 d_debug）
    
    // TODO: 填寫您的參數
    void* args[] = {{/* TODO: 填寫您的 kernel 參數 */}};
    
    // 啟動 kernel
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
    
    std::cout << "Launching kernel..." << std::endl;
    HIP_CHECK(hipModuleLaunchKernel(kernel,
        gridSize.x, gridSize.y, gridSize.z,
        blockSize.x, blockSize.y, blockSize.z,
        0, nullptr, args, nullptr));
    
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "Kernel finished." << std::endl;
    
    // ===== CAPTURE_INJECT_READ =====
    // ⚠️ 請保持這個標記！mdr_capture.py 會在這裡自動注入讀取和顯示 debug 資料
    
    // ========================================
    // TODO: 顯示您的計算結果（可選）
    // ========================================
    
    // 範例：
    // std::vector<float> h_output(N);
    // HIP_CHECK(hipMemcpy(h_output.data(), d_output, 
    //                     N * sizeof(float), hipMemcpyDeviceToHost));
    // for (int i = 0; i < 10; i++) {{
    //     std::cout << "Output[" << i << "] = " << h_output[i] << std::endl;
    // }}
    
    // ========================================
    // TODO: 清理資源
    // ========================================
    
    // 範例：
    // HIP_CHECK(hipFree(d_input));
    // HIP_CHECK(hipFree(d_output));
    
    // ===== CAPTURE_INJECT_CLEANUP =====
    // ⚠️ 請保持這個標記！mdr_capture.py 會在這裡自動注入清理 debug buffer
    
    HIP_CHECK(hipModuleUnload(module));
    
    return 0;
}}

/*
========================================
使用說明：
========================================

1. 填寫所有 TODO 部分：
   - 配置您的 kernel 參數和資料
   - 準備 args 陣列（不要手動添加 d_debug）
   - （可選）顯示計算結果
   - 清理資源

2. 使用 mdr_capture.py 注入 capture 程式碼：
   python3 mdr_capture.py input.s --inject-host host_skeleton.cpp

3. 編譯生成的檔案：
   hipcc host_skeleton_with_capture.cpp -o host_runner

4. 執行：
   ./host_runner 256

注意事項：
- 保持所有 CAPTURE_INJECT_* 標記
- args 陣列末尾不要手動添加 &d_debug（工具會自動添加）
- Values per thread: {values_per_thread}（自動計算）
*/
"""
    
    skeleton_path = workdir / "host_skeleton.cpp"
    skeleton_path.write_text(skeleton)
    print(f"\n✓ Generated host skeleton: {skeleton_path}")
    print(f"\n請按照以下步驟操作：")
    print(f"  1. 編輯 {skeleton_path}，填寫 TODO 部分")
    print(f"  2. 運行：python3 mdr_capture.py <input.s> --inject-host {skeleton_path}")
    print(f"  3. 編譯：hipcc host_skeleton_with_capture.cpp -o host_runner")
    print(f"  4. 執行：./host_runner 256")


def inject_capture_to_host(template_path: str, workdir: pathlib.Path, 
                           hsaco_path: pathlib.Path, kernel_name: str, 
                           values_per_thread: int):
    """將 capture 程式碼注入到用戶的 host template"""
    
    template_file = pathlib.Path(template_path)
    if not template_file.exists():
        raise FileNotFoundError(f"Host template not found: {template_path}")
    
    print(f"\n=== Injecting capture code into host template ===")
    print(f"Template: {template_path}")
    print(f"Values per thread: {values_per_thread}")
    
    # 讀取 template
    host_code = template_file.read_text()
    
    # 檢查並添加必要的頭文件
    required_headers = ['<fstream>', '<vector>', '<iostream>']
    headers_to_add = []
    
    for header in required_headers:
        if f'#include {header}' not in host_code:
            headers_to_add.append(header)
    
    if headers_to_add:
        # 找到第一個 #include 的位置並在其後添加
        first_include = host_code.find('#include')
        if first_include != -1:
            # 找到這一行的末尾
            line_end = host_code.find('\n', first_include)
            if line_end != -1:
                # 在第一個 #include 之後添加缺失的頭文件
                new_includes = '\n'.join(f'#include {h}' for h in headers_to_add)
                host_code = host_code[:line_end+1] + new_includes + '\n' + host_code[line_end+1:]
                print(f"✓ Added missing headers: {', '.join(headers_to_add)}")
    
    # 檢查必要的標記
    required_markers = [
        "CAPTURE_INJECT_ALLOC",
        "CAPTURE_INJECT_READ"
    ]
    
    missing_markers = []
    for marker in required_markers:
        if marker not in host_code:
            missing_markers.append(marker)
    
    if missing_markers:
        print(f"\n❌ 錯誤：Host template 缺少以下標記：")
        for marker in missing_markers:
            print(f"   - // ===== {marker} =====")
        print(f"\n請使用 --generate-skeleton 生成範例，或手動添加這些標記。")
        return
    
    # 1. 注入 debug buffer 分配
    alloc_code = f"""    // ===== CAPTURE_INJECT_ALLOC =====
    // === Auto-injected by mdr_capture.py ===
    float *d_debug;
    std::vector<float> h_debug(N * {values_per_thread}, -999.0f);
    HIP_CHECK(hipMalloc(&d_debug, N * {values_per_thread} * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_debug, h_debug.data(), 
                        N * {values_per_thread} * sizeof(float), 
                        hipMemcpyHostToDevice));
    std::cout << "Debug buffer allocated: " << N * {values_per_thread} << " floats" << std::endl;
    // === End auto-injection ==="""
    
    host_code = host_code.replace("    // ===== CAPTURE_INJECT_ALLOC =====", alloc_code)
    
    # 2. 修改 args 陣列（在末尾添加 &d_debug）
    args_pattern = re.compile(r'(void\*\s+args\[\]\s*=\s*\{)([^\}]*)(\};)', re.DOTALL)
    
    def add_debug_arg(match):
        prefix = match.group(1)
        args_content = match.group(2).strip()
        suffix = match.group(3)
        
        # 如果是空的或只有註解，不添加逗號
        if not args_content or args_content.startswith('/*'):
            return f"{prefix}{args_content}&d_debug{suffix}  // Auto-added by mdr_capture.py"
        else:
            # 移除末尾的逗號（如果有）
            args_content = args_content.rstrip(', ')
            return f"{prefix}{args_content}, &d_debug{suffix}  // Auto-added by mdr_capture.py"
    
    modified = args_pattern.sub(add_debug_arg, host_code)
    
    if modified == host_code:
        print(f"\n⚠️  警告：未找到 'void* args[] = {{...}};' 模式")
        print(f"   請確認您的 host code 包含正確的參數陣列定義")
    else:
        print(f"✓ Modified args array (added &d_debug)")
        host_code = modified
    
    # 3. 注入讀取和顯示程式碼
    read_code = f"""    // ===== CAPTURE_INJECT_READ =====
    // === Auto-injected by mdr_capture.py ===
    HIP_CHECK(hipMemcpy(h_debug.data(), d_debug, 
                        N * {values_per_thread} * sizeof(float), 
                        hipMemcpyDeviceToHost));
    
    std::cout << "\\n=== Captured Debug Data (first 10 threads) ===" << std::endl;
    for (int tid = 0; tid < std::min(10, N); tid++) {{
        std::cout << "Thread " << tid << ": ";
        for (int v = 0; v < {values_per_thread}; v++) {{
            std::cout << h_debug[tid * {values_per_thread} + v];
            if (v < {values_per_thread} - 1) std::cout << ", ";
        }}
        std::cout << std::endl;
    }}
    
    // 統計分析
    std::cout << "\\n=== Statistics ===" << std::endl;
    int non_default_count = 0;
    for (const auto& val : h_debug) {{
        if (val != -999.0f) non_default_count++;
    }}
    std::cout << "Non-default values: " << non_default_count << " / " << h_debug.size() << std::endl;
    
    // 儲存完整資料
    std::ofstream outfile("debug_data.txt");
    for (int tid = 0; tid < N; tid++) {{
        outfile << tid;
        for (int v = 0; v < {values_per_thread}; v++) {{
            outfile << " " << h_debug[tid * {values_per_thread} + v];
        }}
        outfile << "\\n";
    }}
    outfile.close();
    std::cout << "Full data saved to debug_data.txt" << std::endl;
    // === End auto-injection ==="""
    
    host_code = host_code.replace("    // ===== CAPTURE_INJECT_READ =====", read_code)
    
    # 4. 注入清理程式碼
    cleanup_code = f"""    // ===== CAPTURE_INJECT_CLEANUP =====
    // === Auto-injected by mdr_capture.py ===
    HIP_CHECK(hipFree(d_debug));
    // === End auto-injection ==="""
    
    host_code = host_code.replace("    // ===== CAPTURE_INJECT_CLEANUP =====", cleanup_code)
    
    # 5. 更新 HSACO 和 kernel 名稱
    # 支持兩種方式：
    # (a) 使用 placeholder: {{HSACO_PATH}}, {{KERNEL_NAME}}
    # (b) 自動替換現有的 .hsaco 文件名和 kernel 名稱
    
    # 方式 (a): 替換 placeholder
    host_code = host_code.replace("{{HSACO_PATH}}", hsaco_path.name)
    host_code = host_code.replace("{{KERNEL_NAME}}", kernel_name)
    
    # 方式 (b): 自動替換 hipModuleLoad 中的 .hsaco 文件名
    # 匹配模式: hipModuleLoad(&module, "xxx.hsaco")
    hsaco_pattern = re.compile(r'(hipModuleLoad\s*\([^,]+,\s*")([^"]+\.hsaco)("\))', re.MULTILINE)
    if hsaco_pattern.search(host_code):
        host_code = hsaco_pattern.sub(rf'\g<1>{hsaco_path.name}\g<3>', host_code)
        print(f"✓ Updated HSACO filename to: {hsaco_path.name}")
    
    # 自動替換 hipModuleGetFunction 中的 kernel 名稱
    kernel_pattern = re.compile(r'(hipModuleGetFunction\s*\([^,]+,\s*[^,]+,\s*")([^"]+)("\))', re.MULTILINE)
    detected_kernels = kernel_pattern.findall(host_code)
    if detected_kernels:
        old_kernel_name = detected_kernels[0][1]  # 獲取中間的 group (kernel 名稱)
        if old_kernel_name != kernel_name:
            host_code = kernel_pattern.sub(rf'\g<1>{kernel_name}\g<3>', host_code)
            print(f"✓ Updated kernel name: {old_kernel_name} -> {kernel_name}")
        else:
            print(f"✓ Kernel name already correct: {kernel_name}")
    
    # 儲存修改後的檔案
    output_name = template_file.stem + "_with_capture" + template_file.suffix
    output_path = workdir / output_name
    output_path.write_text(host_code)
    
    print(f"\n✓ Generated: {output_path}")
    print(f"\n編譯與執行：")
    print(f"  cd {workdir}")
    print(f"  hipcc {output_name} -o host_runner")
    print(f"  ./host_runner 256")


if __name__ == "__main__":
    main()
