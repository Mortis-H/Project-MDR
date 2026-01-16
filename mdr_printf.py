#!/usr/bin/env python3
"""
AMD ISA Assembly Debug Tool
============================

在組合語言（.s 檔案）中插入 printf 除錯功能。

使用方式：
1. 在 .s 檔案中以註解形式標註要印出的內容：
   ; @PRINT fmt="value: %f" reg=v6 type=f32
   ; @PRINT fmt="idx: %d" reg=v0 type=i32
   ; @PRINT fmt="regs: v4=%f, v5=%f" reg=v4,v5 type=f32,f32

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
    """
    line_number: int           # 在 .s 檔案中的行號（0-based）
    format_string: str         # printf 格式字串
    registers: List[str]       # 要印出的暫存器列表（如 ["v6", "v7"]）
    types: List[str]           # 對應的類型（如 ["f32", "f32"]）
    condition: Optional[str] = None  # 條件表達式（如 "tid_eq(3)"）
    next_instruction: Optional[str] = None  # @PRINT 之後的第一條 ISA 指令
    expressions: Optional[List[str]] = None  # 計算表達式列表（如 ["v6 * v7"]）
    
    def __str__(self):
        cond_str = f" [cond={self.condition}]" if self.condition else ""
        if self.expressions:
            return f"@PRINT at line {self.line_number}: {self.format_string} (expr={self.expressions}){cond_str}"
        return f"@PRINT at line {self.line_number}: {self.format_string} ({self.registers}){cond_str}"


def parse_print_directive(line: str, line_number: int) -> Optional[PrintDirective]:
    """
    解析 @PRINT 指令
    
    支援格式：
    1. 直接印出暫存器：
       ; @PRINT fmt="value: %f" reg=v6 type=f32
       ; @PRINT cond=tid_eq(3) fmt="hello" reg=v0 type=i32
       
    2. 計算表達式後印出：
       ; @PRINT fmt="product=%f" expr="v6 * v7" type=f32
       ; @PRINT fmt="B^2-4AC=%f" expr="v7*v7 - 4.0*v6*v2" type=f32
       
    3. 混合模式（先印暫存器，再印表達式）：
       ; @PRINT fmt="a=%f, b=%f, a*b=%f" reg=v6,v7 expr="v6*v7" type=f32,f32,f32
    """
    # 匹配 @PRINT 指令（支援 ; 或 # 作為註解前綴）
    match = re.search(r'[;#]\s*@PRINT\s+(.+)', line)
    if not match:
        return None
    
    directive_content = match.group(1).strip()
    
    # 解析各個屬性
    fmt_match = re.search(r'fmt\s*=\s*"([^"]*)"', directive_content)
    reg_match = re.search(r'reg\s*=\s*([\w,\[\]:\s]+?)(?:\s+(?:type|cond|expr|$))', directive_content)
    type_match = re.search(r'type\s*=\s*([\w,\s]+?)(?:\s+(?:reg|cond|fmt|expr|$)|$)', directive_content)
    cond_match = re.search(r'cond\s*=\s*(\w+\([^)]+\))', directive_content)
    # 表達式匹配：支援 expr="..." 或 expr='...'
    expr_match = re.search(r'expr\s*=\s*"([^"]+)"', directive_content)
    
    if not fmt_match or not type_match:
        print(f"[Warning] Incomplete @PRINT at line {line_number + 1}: {line.strip()}")
        return None
    
    format_string = fmt_match.group(1)
    
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
        # 支持多個表達式用 ; 分隔
        expressions = [e.strip() for e in expr_str.split(';')]
    
    # 條件（可選）
    condition = cond_match.group(1) if cond_match else None
    
    # 驗證：必須有 reg 或 expr
    if not registers and not expressions:
        print(f"[Warning] @PRINT must have 'reg' or 'expr' at line {line_number + 1}")
        return None
    
    # 驗證數量匹配
    total_values = len(registers) + len(expressions)
    if total_values != len(types):
        print(f"[Warning] Value/type count mismatch at line {line_number + 1}")
        print(f"  Registers: {registers}, Expressions: {expressions}")
        print(f"  Types: {types}")
        return None
    
    return PrintDirective(
        line_number=line_number,
        format_string=format_string,
        registers=registers,
        types=types,
        condition=condition,
        expressions=expressions if expressions else None
    )


def parse_asm_file(asm_path: pathlib.Path) -> Tuple[List[str], List[PrintDirective], bool]:
    """
    解析 .s 檔案，提取 @PRINT 指令
    
    對於每個 @PRINT，找到它之後的第一條非註解 ISA 指令，作為插入點的參考
    
    Returns:
        (lines, print_directives, has_barrier): 原始行列表、解析出的 @PRINT 指令、是否有 s_barrier
    """
    lines = asm_path.read_text().split('\n')
    directives = []
    has_barrier = False
    
    for i, line in enumerate(lines):
        # 檢測 s_barrier 指令
        if 's_barrier' in line and not line.strip().startswith(';'):
            has_barrier = True
        
        if '@PRINT' in line:
            directive = parse_print_directive(line, i)
            if directive:
                # 找到 @PRINT 之後的第一條 ISA 指令（用於精確定位插入點）
                for j in range(i + 1, min(i + 10, len(lines))):
                    next_line = lines[j].strip()
                    # 跳過空行、註解、標籤
                    if next_line and not next_line.startswith(';') and not next_line.startswith('#'):
                        if not next_line.endswith(':'):  # 不是標籤
                            directive.next_instruction = next_line
                            break
                directives.append(directive)
                print(f"[Info] Found: {directive}")
                if directive.next_instruction:
                    instr_preview = directive.next_instruction[:50]
                    print(f"       After: {instr_preview}...")
    
    return lines, directives, has_barrier


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


def compile_expression_to_mlir(expr: str, result_type: str, unique_id: int, expr_idx: int) -> Tuple[str, str]:
    """
    將表達式編譯為 MLIR 代碼
    
    使用 arith dialect 進行計算。
    
    Args:
        expr: 表達式字串（如 "v6 * v7" 或 "v7*v7 - 4.0*v6*v2"）
        result_type: 結果類型（f32, i32 等）
        unique_id: 唯一識別符
        expr_idx: 表達式索引
    
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
                # 使用 value binding 讀取暫存器值
                reg_num = token_value[1:]
                reg_type = token_value[0]  # v, s, or a
                
                if reg_type == 'v':
                    lines.append(f'              %{var_name} = llvm.inline_asm has_side_effects asm_dialect = att "v_mov_b32 $0, {token_value}", "=v": () -> {mlir_type}')
                elif reg_type == 's':
                    lines.append(f'              %{var_name} = llvm.inline_asm has_side_effects asm_dialect = att "s_mov_b32 $0, {token_value}", "=s": () -> {mlir_type}')
                else:  # a
                    lines.append(f'              %{var_name} = llvm.inline_asm has_side_effects asm_dialect = att "v_accvgpr_read_b32 $0, {token_value}", "=v": () -> {mlir_type}')
                
                bound_regs[token_value] = var_name
            return f'%{bound_regs[token_value]}'
        elif token_type == 'NUM':
            var_name = f'expr_{unique_id}_{expr_idx}_const_{var_counter[0]}'
            var_counter[0] += 1
            lines.append(f'              %{var_name} = arith.constant {token_value} : {mlir_type}')
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
    if reg.startswith('v'):
        # VGPR
        asm_instr = f"v_mov_b32 $0, {reg}"
    elif reg.startswith('s'):
        # SGPR - 需要先 mov 到 VGPR
        asm_instr = f"v_mov_b32 $0, {reg}"
    else:
        # 預設
        asm_instr = f"v_mov_b32 $0, {reg}"
    
    return f'              %{var_name} = llvm.inline_asm has_side_effects asm_dialect = att "{asm_instr}", "=v": () -> {mlir_type}'


def generate_condition_check(condition: str, true_block: str, merge_block: str, unique_id: int) -> Tuple[str, str, str]:
    """
    生成條件檢查的 MLIR 程式碼
    
    Args:
        condition: 條件表達式（如 "tid_eq(3)"）
        true_block: 成立時跳轉的 block 名稱
        merge_block: 合併 block 名稱
        unique_id: 唯一識別碼
    
    Returns:
        (cond_check_code, branch_code)
    """
    # 解析條件
    match = re.match(r'tid_(\w+)\((\d+)\)', condition)
    if not match:
        return ("", "")
    
    cmp_type = match.group(1)  # eq, lt, le, gt, ge
    value = match.group(2)
    
    cmp_map = {
        'eq': 'eq',
        'lt': 'slt',
        'le': 'sle',
        'gt': 'sgt',
        'ge': 'sge'
    }
    
    mlir_cmp = cmp_map.get(cmp_type, 'eq')
    
    tid_code = f'              %debug_tid_{unique_id} = gpu.thread_id x'
    flag_code = f'              %debug_flag_{unique_id} = arith.constant {value} : index'
    cmp_code = f'              %debug_cond_{unique_id} = arith.cmpi {mlir_cmp}, %debug_tid_{unique_id}, %debug_flag_{unique_id} : index'
    branch_code = f'              cf.cond_br %debug_cond_{unique_id}, ^{true_block}, ^{merge_block}'
    
    return (f'{tid_code}\n{flag_code}\n{cmp_code}', branch_code)


def generate_printf(directive: PrintDirective, unique_id: int) -> str:
    """
    生成 gpu.printf 的完整 MLIR 程式碼區塊
    
    包含：
    - Value bindings（直接讀取暫存器）
    - 表達式計算（如果有 expr 參數）
    - 條件檢查（如果有條件，使用 scf.if）
    - gpu.printf 呼叫
    
    對於有 s_barrier 的 kernel，建議使用 cond=tid_eq(0) 來避免同步問題
    """
    lines = []
    
    # 生成註解標記
    lines.append(f'              // === @PRINT from line {directive.line_number + 1} ===')
    
    var_names = []
    reg_types = directive.types[:len(directive.registers)]  # 暫存器對應的類型
    expr_types = directive.types[len(directive.registers):]  # 表達式對應的類型
    
    # 1. 生成 value bindings（直接讀取暫存器）
    for i, (reg, typ) in enumerate(zip(directive.registers, reg_types)):
        var_name = f'print_val_{unique_id}_{i}'
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
                # 使用常數 0 作為備用
                fallback_var = f'expr_fallback_{unique_id}_{i}'
                mlir_type = map_type_to_mlir(typ)
                if typ.startswith('f'):
                    lines.append(f'              %{fallback_var} = arith.constant 0.0 : {mlir_type}')
                else:
                    lines.append(f'              %{fallback_var} = arith.constant 0 : {mlir_type}')
                var_names.append(fallback_var)
    
    # 準備 printf 的參數列表
    args = ', '.join(f'%{name}' for name in var_names)
    types = ', '.join(map_type_to_mlir(t) for t in directive.types)
    
    # MLIR string 中的換行需使用 \0A
    escaped_format = directive.format_string.replace('\\n', '\\0A')
    if not escaped_format.endswith('\\0A'):
        escaped_format += '\\0A'
    
    # 條件式 printf 支援
    if directive.condition:
        # 解析條件：支援 tid_eq(N), tid_lt(N), tid_le(N), tid_gt(N), tid_ge(N)
        match = re.match(r'tid_(\w+)\((\d+)\)', directive.condition)
        if match:
            cmp_type, value = match.groups()
            
            # 獲取 thread_id
            lines.append(f'              %tid_x_{unique_id} = gpu.thread_id x')
            lines.append(f'              %cmp_val_{unique_id} = arith.constant {value} : index')
            
            # 根據條件類型選擇比較操作
            cmp_ops = {
                'eq': 'eq',
                'ne': 'ne', 
                'lt': 'slt',
                'le': 'sle',
                'gt': 'sgt',
                'ge': 'sge'
            }
            mlir_cmp = cmp_ops.get(cmp_type, 'eq')
            
            lines.append(f'              %cond_{unique_id} = arith.cmpi {mlir_cmp}, %tid_x_{unique_id}, %cmp_val_{unique_id} : index')
            
            # 使用 scf.if 包裹 printf
            lines.append(f'              scf.if %cond_{unique_id} {{')
            lines.append(f'                gpu.printf "{escaped_format}", {args} : {types}')
            lines.append(f'              }}')
            
            print(f"[Info] Conditional printf: {directive.condition}")
        else:
            print(f"[Warning] Unknown condition format: {directive.condition}, printing unconditionally")
            lines.append(f'              gpu.printf "{escaped_format}", {args} : {types}')
    else:
        # 無條件印出
        lines.append(f'              gpu.printf "{escaped_format}", {args} : {types}')
    
    lines.append(f'              // === End @PRINT ===')
    
    return '\n'.join(lines)


def inject_printf_into_mlir(gpumlir_text: str, directives: List[PrintDirective], reg_info: Dict[str, int]) -> str:
    """
    將 printf 指令注入到 GPU MLIR 中
    
    策略（與原版 /home/morhuang/my_tmp/asm_debug.py 一致）：
    1. 在 kernel 函數開頭添加 register clobbering 開始
    2. 在 .LBB0_2: 標籤之前（有效 thread 區域內）添加：
       - Restore kernarg pointer
       - 各個 @PRINT 的 value binding + printf
       - Register clobbering 結束
    3. 如果沒有 .LBB0_2: 標籤，則在 s_endpgm 之前插入
    
    Args:
        gpumlir_text: 原始 GPU MLIR 文字
        directives: @PRINT 指令列表
        reg_info: 暫存器使用量資訊
    
    Returns:
        修改後的 GPU MLIR 文字
    """
    if not directives:
        return gpumlir_text
    
    # === 動態計算 clobber 範圍 ===
    MAX_VECTOR_SIZE = 32      # LLVM inline_asm vector 類型限制
    SGPR_KERNARG_BACKUP = 18  # 保存 kernarg pointer 到 s[18:19]
    SGPR_RESERVED_START = 4   # s[0:3] 是系統保留的，從 s4 開始保護
    SGPR_RESERVED_END = 17    # 保護到 s17（不含 s[18:19] 用於 kernarg backup）
    
    original_vgpr = reg_info.get('vgpr', 0)
    original_sgpr = reg_info.get('sgpr', 0)
    original_agpr = reg_info.get('agpr', 0)
    
    # 只 clobber 原始 kernel 使用的暫存器數量
    total_vgpr = max(1, original_vgpr) if original_vgpr > 0 else 0
    total_agpr = max(1, original_agpr) if original_agpr > 0 else 0
    
    # SGPR: 保護 s[4:17]（14 個 SGPR，但 vector 必須是 2 的冪）
    # 將範圍調整為 s[4:19]（16 個 SGPR）以符合 vector<16xi32> 的要求
    # 但同時我們會保存 kernarg pointer 到 s[18:19]，所以需要在保存前後處理
    sgpr_start = SGPR_RESERVED_START
    sgpr_count_needed = max(original_sgpr, 20) - sgpr_start  # 確保至少到 s19
    
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
    
    # === 生成 clobber 開始程式碼 ===
    clobber_start_lines = ['              // === Register Clobbering Start ===']
    
    # 保存 kernarg pointer (s[0:1]) 到 s[SGPR_KERNARG_BACKUP:SGPR_KERNARG_BACKUP+1]
    # 這必須在 SGPR clobbering 開始之前執行
    clobber_start_lines.append('              // Save kernarg pointer for printf (to s[18:19])')
    clobber_start_lines.append(f'              llvm.inline_asm has_side_effects "s_mov_b64 s[{SGPR_KERNARG_BACKUP}:{SGPR_KERNARG_BACKUP+1}], s[0:1]", ""  : () -> ()')
    
    # VGPR clobber
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
    
    # SGPR clobber (跳過系統保留的 s[0:3])
    if total_sgpr > 0:
        clobber_start_lines.append(f'              // Protecting {total_sgpr} SGPRs (s[{sgpr_start}:{sgpr_start + total_sgpr - 1}])')
        for block_idx in range(num_sgpr_blocks):
            start_reg = sgpr_start + block_idx * MAX_VECTOR_SIZE
            end_reg = min(sgpr_start + (block_idx + 1) * MAX_VECTOR_SIZE, sgpr_start + total_sgpr) - 1
            block_size = end_reg - start_reg + 1
            clobber_start_lines.append(
                f'              %reserved_sgpr_{block_idx} = llvm.inline_asm has_side_effects asm_dialect = att "", '
                f'"={{s[{start_reg}:{end_reg}]}}": () -> vector<{block_size}xi32>'
            )
    
    # AGPR clobber
    if total_agpr > 0:
        clobber_start_lines.append(f'              // Protecting {total_agpr} AGPRs')
        for block_idx in range(num_agpr_blocks):
            start_reg = block_idx * MAX_VECTOR_SIZE
            end_reg = min((block_idx + 1) * MAX_VECTOR_SIZE, total_agpr) - 1
            block_size = end_reg - start_reg + 1
            clobber_start_lines.append(
                f'              %reserved_agpr_{block_idx} = llvm.inline_asm has_side_effects asm_dialect = att "", '
                f'"={{a[{start_reg}:{end_reg}]}}": () -> vector<{block_size}xi32>'
            )
    
    clobber_start_lines.append('              // === End Clobbering Start ===')
    clobber_start = '\n'.join(clobber_start_lines) + '\n'
    
    # 生成 printf 區塊（所有 @PRINT 指令）- 包含 kernarg restore
    printf_blocks = ['              // Restore kernarg pointer for printf (from s[18:19])']
    printf_blocks.append(f'              llvm.inline_asm has_side_effects "s_mov_b64 s[0:1], s[{SGPR_KERNARG_BACKUP}:{SGPR_KERNARG_BACKUP+1}]", ""  : () -> ()')
    for i, directive in enumerate(directives):
        printf_blocks.append(generate_printf(directive, i))
    
    printf_section = '\n'.join(printf_blocks)
    
    # === 生成 clobber 結束程式碼 ===
    clobber_end_lines = ['              // === Register Clobbering End ===']
    
    # VGPR restore
    if total_vgpr > 0:
        for block_idx in range(num_vgpr_blocks):
            start_reg = block_idx * MAX_VECTOR_SIZE
            end_reg = min((block_idx + 1) * MAX_VECTOR_SIZE, total_vgpr) - 1
            block_size = end_reg - start_reg + 1
            clobber_end_lines.append(
                f'              llvm.inline_asm has_side_effects asm_dialect = att "", '
                f'"{{v[{start_reg}:{end_reg}]}}" %reserved_vgpr_{block_idx} : (vector<{block_size}xi32>)-> ()'
            )
    
    # SGPR restore
    if total_sgpr > 0:
        for block_idx in range(num_sgpr_blocks):
            start_reg = sgpr_start + block_idx * MAX_VECTOR_SIZE
            end_reg = min(sgpr_start + (block_idx + 1) * MAX_VECTOR_SIZE, sgpr_start + total_sgpr) - 1
            block_size = end_reg - start_reg + 1
            clobber_end_lines.append(
                f'              llvm.inline_asm has_side_effects asm_dialect = att "", '
                f'"{{s[{start_reg}:{end_reg}]}}" %reserved_sgpr_{block_idx} : (vector<{block_size}xi32>)-> ()'
            )
    
    # AGPR restore
    if total_agpr > 0:
        for block_idx in range(num_agpr_blocks):
            start_reg = block_idx * MAX_VECTOR_SIZE
            end_reg = min((block_idx + 1) * MAX_VECTOR_SIZE, total_agpr) - 1
            block_size = end_reg - start_reg + 1
            clobber_end_lines.append(
                f'              llvm.inline_asm has_side_effects asm_dialect = att "", '
                f'"{{a[{start_reg}:{end_reg}]}}" %reserved_agpr_{block_idx} : (vector<{block_size}xi32>)-> ()'
            )
    
    clobber_end_lines.append('              // === End Clobbering End ===')
    clobber_end = '\n'.join(clobber_end_lines) + '\n'
    
    # 找到函數開頭，在第一個 llvm.inline_asm 之前插入 clobber_start
    # 在 .LBB0_2: 標籤之前插入 printf（因為跳轉目標之後是無效的 thread）
    # 如果沒有 .LBB0_2:，則在 s_endpgm 之前插入
    lines = gpumlir_text.split('\n')
    modified_lines = []
    in_func = False
    clobber_inserted = False
    printf_inserted = False
    
    # 先掃描是否有 .LBB0_2: 標籤（注意：必須是標籤定義，結尾有冒號）
    # 匹配 '.LBB0_2:' 或 '.LBB0_2"' (在 inline_asm 字串中)
    has_lbb0_2_label = any('.LBB0_2:' in line or '.LBB0_2"' in line for line in lines)
    
    for i, line in enumerate(lines):
        # 檢測是否進入 gpu.func
        if 'gpu.func @' in line and 'kernel' in line:
            in_func = True
        
        # 在第一個 llvm.inline_asm 之前插入 clobber_start
        if in_func and not clobber_inserted and 'llvm.inline_asm' in line:
            modified_lines.append(clobber_start)
            clobber_inserted = True
        
        # 優先在 .LBB0_2: 標籤之前插入 printf（在計算分支內）
        # 注意：只匹配標籤定義（結尾有冒號），不匹配跳轉指令中的引用
        # 標籤在 inline_asm 中會是 '.LBB0_2:' 或 '.LBB0_2"'
        is_lbb0_2_label = '.LBB0_2:' in line or ('.LBB0_2"' in line and 's_cbranch' not in line)
        if in_func and not printf_inserted and has_lbb0_2_label and is_lbb0_2_label:
            modified_lines.append(printf_section)
            modified_lines.append(clobber_end)
            printf_inserted = True
        
        # 如果沒有 .LBB0_2: 標籤，則在 s_endpgm 之前插入
        if in_func and not printf_inserted and not has_lbb0_2_label and 's_endpgm' in line:
            modified_lines.append(printf_section)
            modified_lines.append(clobber_end)
            printf_inserted = True
        
        modified_lines.append(line)
        
        # 離開 gpu.func
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
            # 無 printf：直接使用原始的完整 args
            kernel['.args'] = []
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
                if 'name' in arg:
                    yaml_arg['.name'] = arg['name']
                kernel['.args'].append(yaml_arg)
            print(f"[Info] Restored {len(original_all_args)} original kernel arguments")
    
    # 重新生成 YAML
    fixed_yaml = yaml.dump(gen_metadata, default_flow_style=False, sort_keys=False)
    
    # 替換 ISA 中的 metadata
    before_metadata = isa_text[:yaml_start]
    after_metadata = isa_text[yaml_end:]
    
    fixed_isa = before_metadata + "---\n" + fixed_yaml + "...\n" + after_metadata
    
    # 同時修復 .amdhsa_* 指令
    if 'kernarg_segment_size' in attrs:
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


def build_debug_hsaco(gpumlir_path: pathlib.Path, chip: str, workdir: pathlib.Path, 
                      original_isa_file: pathlib.Path = None, has_printf: bool = False):
    """
    從修改後的 GPU MLIR 生成 HSACO
    
    Args:
        gpumlir_path: 修改後的 GPU MLIR 文件
        chip: 目標 GPU 架構
        workdir: 工作目錄
        original_isa_file: 原始 ISA 文件路徑（用於提取 metadata）
        has_printf: 是否有 printf 注入（影響 metadata 合併策略）
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
        isa_text = fix_isa_metadata(isa_text, original_isa_file, has_printf)
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
     ; @PRINT fmt="v6=%f" reg=v6 type=f32
     ; @PRINT cond=tid_eq(0) fmt="s[0:1]=%p" reg=s0 type=ptr

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
    
    # 1. 解析 @PRINT 指令
    print(f"\n=== Parsing @PRINT directives ===")
    lines, directives, has_barrier = parse_asm_file(input_path)
    
    if not directives:
        if args.no_printf:
            print("[Info] No @PRINT directives found (--no-printf mode)")
        else:
            print("[Info] No @PRINT directives found. Will generate HSACO without printf.")
    else:
        print(f"\nFound {len(directives)} @PRINT directive(s)")
    
    # 警告：s_barrier 與 printf 不兼容
    if has_barrier and directives and not args.no_printf:
        print("\n" + "=" * 60)
        print("⚠️  WARNING: Kernel contains s_barrier instruction!")
        print("   gpu.printf's hostcall mechanism may conflict with barrier")
        print("   synchronization, causing kernel to hang or crash.")
        print("")
        print("   Recommendations:")
        print("   1. Use --no-printf for functional verification")
        print("   2. Use cond=tid_eq(0) to limit printf to single thread")
        print("   3. Place @PRINT only after all barriers complete")
        print("=" * 60 + "\n")
    
    if args.dry_run:
        print("\n[Dry Run] Stopping here.")
        return
    
    # 2. 分析暫存器使用量
    print(f"\n=== Analyzing register usage ===")
    reg_info = analyze_registers(asm_text)
    print(f"  VGPR: {reg_info['vgpr']}, SGPR: {reg_info['sgpr']}, AGPR: {reg_info['agpr']}")
    
    # 3. 轉換為 GPU MLIR
    gpumlir_path = translate_to_gpumlir(input_path, workdir)
    
    # 4. 注入 printf 程式碼（如果有 @PRINT 且未禁用）
    has_printf = bool(directives) and not args.no_printf
    
    if not directives or args.no_printf:
        print(f"\nUsing original GPU MLIR (no printf injection)")
        modified_path = gpumlir_path
    else:
        print(f"\n=== Injecting printf code ===")
        gpumlir_text = gpumlir_path.read_text()
        modified_mlir = inject_printf_into_mlir(gpumlir_text, directives, reg_info)
        
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
    hsaco_path = build_debug_hsaco(modified_path, args.chip, workdir, input_path, has_printf)
    
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
