#!/usr/bin/env python3
"""
AMD ISA Assembly Capture Tool (Register-based)
===============================================

在組合語言中插入暫存器快照功能，將值保存到安全的 register 中。

使用方式：
1. 在 .s 檔案中標註要捕獲的內容：
   ; @CAPTURE reg=v2,v3 type=f32,f32
   ; @CAPTURE cond=tid_eq(0) reg=v2 expr="v2*2.0" type=f32,f32

2. 執行此工具：
   python3 mdr_capture_v2.py input.s --output-dir output

3. 工具會：
   - 分析 register 使用量
   - 自動分配未使用的 register
   - 生成 capture 代碼（使用 inline_asm）
   - 生成映射文件（告訴您值存在哪些 register）

優點：
- 純 register 操作，零記憶體開銷
- 不影響原始 kernel 邏輯
- 後續可在 assembly 中繼續使用這些 register
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
    """代表一個 @CAPTURE 指令"""
    line_number: int
    registers: List[str]          # 要捕獲的原始 register
    types: List[str]              # 類型
    condition: Optional[str] = None
    expressions: Optional[List[str]] = None
    
    def __str__(self):
        cond_str = f" [cond={self.condition}]" if self.condition else ""
        if self.expressions:
            return f"@CAPTURE at line {self.line_number + 1}: {self.registers} + expr={self.expressions}{cond_str}"
        return f"@CAPTURE at line {self.line_number + 1}: {self.registers}{cond_str}"


@dataclass
class CaptureMapping:
    """記錄 capture 的結果存放在哪個 register"""
    directive_id: int
    source: str                   # 來源：register 名稱或表達式
    target_register: str          # 目標 register（如 "v10"）
    type_str: str                 # 類型（如 "f32"）
    is_expression: bool = False   # 是否為表達式


def parse_capture_directive(line: str, line_number: int) -> Optional[CaptureDirective]:
    """解析 @CAPTURE 指令"""
    match = re.search(r'[;#]\s*@CAPTURE\s+(.+)', line)
    if not match:
        return None
    
    directive_content = match.group(1).strip()
    
    # 解析各個屬性
    reg_match = re.search(r'reg\s*=\s*([\w,\[\]:\s]+?)(?:\s+(?:type|cond|expr|$))', directive_content)
    type_match = re.search(r'type\s*=\s*([\w,\s]+?)(?:\s+(?:reg|cond|expr|$)|$)', directive_content)
    cond_match = re.search(r'cond\s*=\s*(\w+\([^)]+\))', directive_content)
    expr_match = re.search(r'expr\s*=\s*"([^"]+)"', directive_content)
    
    if not type_match:
        print(f"[Warning] @CAPTURE missing 'type' at line {line_number + 1}")
        return None
    
    # 解析類型
    type_str = type_match.group(1).strip().rstrip(',')
    types = [t.strip() for t in type_str.split(',')]
    
    # 解析暫存器
    registers = []
    if reg_match:
        reg_str = reg_match.group(1).strip().rstrip(',')
        registers = [r.strip() for r in reg_str.split(',')]
    
    # 解析表達式
    expressions = []
    if expr_match:
        expr_str = expr_match.group(1).strip()
        expressions = [e.strip() for e in expr_str.split(';')]
    
    # 條件
    condition = cond_match.group(1) if cond_match else None
    
    # 驗證
    if not registers and not expressions:
        print(f"[Warning] @CAPTURE must have 'reg' or 'expr' at line {line_number + 1}")
        return None
    
    total_values = len(registers) + len(expressions)
    if total_values != len(types):
        print(f"[Warning] Value/type count mismatch at line {line_number + 1}")
        return None
    
    return CaptureDirective(
        line_number=line_number,
        registers=registers,
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
# Register 分析與分配
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


class RegisterAllocator:
    """自動分配未使用的 register"""
    
    def __init__(self, max_vgpr: int, max_sgpr: int, max_agpr: int):
        self.next_vgpr = max_vgpr
        self.next_sgpr = max_sgpr
        self.next_agpr = max_agpr
        
        # 硬體限制（GFX9/GFX10 架構）
        self.VGPR_LIMIT = 256
        self.SGPR_LIMIT = 102  # 實際可用（總共 104，但部分保留）
        self.AGPR_LIMIT = 256
    
    def allocate_vgpr(self, count: int = 1) -> List[str]:
        """分配 VGPR"""
        if self.next_vgpr + count > self.VGPR_LIMIT:
            raise RuntimeError(
                f"VGPR exhausted! Need {count} more, but only "
                f"{self.VGPR_LIMIT - self.next_vgpr} available. "
                f"Current usage: v0-v{self.next_vgpr-1}"
            )
        
        allocated = [f"v{i}" for i in range(self.next_vgpr, self.next_vgpr + count)]
        self.next_vgpr += count
        return allocated
    
    def allocate_sgpr(self, count: int = 1) -> List[str]:
        """分配 SGPR"""
        if self.next_sgpr + count > self.SGPR_LIMIT:
            raise RuntimeError(
                f"SGPR exhausted! Need {count} more, but only "
                f"{self.SGPR_LIMIT - self.next_sgpr} available."
            )
        
        allocated = [f"s{i}" for i in range(self.next_sgpr, self.next_sgpr + count)]
        self.next_sgpr += count
        return allocated
    
    def get_final_usage(self) -> Dict[str, int]:
        """獲取最終的 register 使用量"""
        return {
            'vgpr': self.next_vgpr,
            'sgpr': self.next_sgpr,
            'agpr': self.next_agpr
        }


# ============================================================
# 表達式編譯（簡化版）
# ============================================================

def map_type_to_mlir(type_str: str) -> str:
    """將類型字串轉換為 MLIR 類型"""
    type_map = {
        'f32': 'f32',
        'f64': 'f64',
        'i32': 'i32',
        'i64': 'i64',
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


def compile_expression_to_mlir(expr: str, result_type: str, target_reg: str, 
                                unique_id: int, expr_idx: int) -> str:
    """
    將表達式編譯為 MLIR inline_asm 代碼
    
    Args:
        expr: 表達式字串（如 "v2*2.0"）
        result_type: 結果類型（f32, i32 等）
        target_reg: 目標 register（如 "v10"）
        unique_id: 唯一識別符
        expr_idx: 表達式索引
    
    Returns:
        MLIR 代碼字串
    """
    tokens = parse_expression(expr)
    mlir_type = map_type_to_mlir(result_type)
    lines = []
    
    # 簡化版：只支持簡單的二元運算（a op b）
    # 格式：reg op num 或 num op reg
    
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
                    asm_code = f"v_mul_f32 {target_reg}, {left_val}, {right_val}"
                elif op == '+':
                    asm_code = f"v_add_f32 {target_reg}, {left_val}, {right_val}"
                elif op == '-':
                    asm_code = f"v_sub_f32 {target_reg}, {left_val}, {right_val}"
                else:
                    raise ValueError(f"Unsupported operation: {op}")
            else:
                if op == '*':
                    asm_code = f"v_mul_lo_u32 {target_reg}, {left_val}, {right_val}"
                elif op == '+':
                    asm_code = f"v_add_u32 {target_reg}, {left_val}, {right_val}"
                elif op == '-':
                    asm_code = f"v_sub_u32 {target_reg}, {left_val}, {right_val}"
                else:
                    raise ValueError(f"Unsupported operation: {op}")
            
            comment = f"// Expression: {expr} → {target_reg}"
            lines.append(f'              {comment}')
            lines.append(f'              llvm.inline_asm has_side_effects asm_dialect = att "{asm_code}", "": () -> ()')
            return '\n'.join(lines)
        else:
            raise ValueError(f"Unsupported expression format: {expr}")
        
        # 生成 ISA 指令
        if result_type.startswith('f'):
            # 浮點運算
            if op == '*':
                asm_code = f"v_mul_f32 {target_reg}, {reg}, {const}"
            elif op == '+':
                asm_code = f"v_add_f32 {target_reg}, {reg}, {const}"
            elif op == '-':
                asm_code = f"v_sub_f32 {target_reg}, {reg}, {const}"
            elif op == '/':
                # 除法需要先載入常數到 register
                raise ValueError("Division by constant not yet supported (需要額外的 register)")
            else:
                raise ValueError(f"Unsupported operation: {op}")
        else:
            # 整數運算
            if op == '*':
                asm_code = f"v_mul_lo_u32 {target_reg}, {reg}, {const}"
            elif op == '+':
                asm_code = f"v_add_u32 {target_reg}, {reg}, {const}"
            elif op == '-':
                asm_code = f"v_sub_u32 {target_reg}, {reg}, {const}"
            else:
                raise ValueError(f"Unsupported operation: {op}")
        
        comment = f"// Expression: {expr} → {target_reg}"
        lines.append(f'              {comment}')
        lines.append(f'              llvm.inline_asm has_side_effects asm_dialect = att "{asm_code}", "": () -> ()')
        
    else:
        raise ValueError(f"Complex expressions not yet supported: {expr}")
    
    return '\n'.join(lines)


# ============================================================
# Capture 生成（核心功能）
# ============================================================

def generate_register_capture(directive: CaptureDirective, 
                               unique_id: int,
                               allocator: RegisterAllocator) -> Tuple[str, List[CaptureMapping]]:
    """
    生成 register-based capture 的 MLIR 程式碼
    
    Returns:
        (mlir_code, mappings): MLIR 代碼和 register 映射列表
    """
    lines = []
    mappings = []
    
    lines.append(f'              // === @CAPTURE #{unique_id} from line {directive.line_number + 1} ===')
    
    # 1. 分配 register
    num_values = len(directive.registers) + (len(directive.expressions) if directive.expressions else 0)
    target_registers = allocator.allocate_vgpr(num_values)
    
    print(f"[Info] Allocated registers for CAPTURE #{unique_id}: {', '.join(target_registers)}")
    
    # 2. 條件判斷（如果有）
    has_condition = False
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
            
            # 使用臨時 SGPR 對保存 exec mask
            temp_sgpr_start = 20 + unique_id * 2
            temp_sgpr_end = temp_sgpr_start + 1
            
            lines.append(f'              // Condition: tid_{cmp_type}({value})')
            lines.append(f'              llvm.inline_asm has_side_effects asm_dialect = att "s_mov_b64 s[{temp_sgpr_start}:{temp_sgpr_end}], exec", "" : () -> ()')
            lines.append(f'              llvm.inline_asm has_side_effects asm_dialect = att "{cmp_instr} vcc, {value}, v0", "" : () -> ()')
            lines.append(f'              llvm.inline_asm has_side_effects asm_dialect = att "s_and_b64 exec, exec, vcc", "" : () -> ()')
            has_condition = True
    
    # 3. 生成 register 複製指令（直接複製 register 值）
    reg_idx = 0
    for i, (src_reg, typ) in enumerate(zip(directive.registers, directive.types[:len(directive.registers)])):
        target_reg = target_registers[reg_idx]
        
        lines.append(f'              // Capture: {src_reg} → {target_reg}')
        lines.append(f'              llvm.inline_asm has_side_effects asm_dialect = att "v_mov_b32 {target_reg}, {src_reg}", "": () -> ()')
        
        mappings.append(CaptureMapping(
            directive_id=unique_id,
            source=src_reg,
            target_register=target_reg,
            type_str=typ,
            is_expression=False
        ))
        
        reg_idx += 1
    
    # 4. 生成表達式計算（如果有）
    if directive.expressions:
        expr_types = directive.types[len(directive.registers):]
        for i, (expr, typ) in enumerate(zip(directive.expressions, expr_types)):
            target_reg = target_registers[reg_idx]
            
            try:
                expr_code = compile_expression_to_mlir(expr, typ, target_reg, unique_id, i)
                lines.append(expr_code)
                
                mappings.append(CaptureMapping(
                    directive_id=unique_id,
                    source=expr,
                    target_register=target_reg,
                    type_str=typ,
                    is_expression=True
                ))
            except Exception as e:
                print(f"[Warning] Failed to compile expression '{expr}': {e}")
                # Fallback: 複製第一個 register
                fallback_src = directive.registers[0] if directive.registers else "v0"
                lines.append(f'              // Fallback for failed expression: {expr}')
                lines.append(f'              llvm.inline_asm has_side_effects asm_dialect = att "v_mov_b32 {target_reg}, {fallback_src}", "": () -> ()')
                
                mappings.append(CaptureMapping(
                    directive_id=unique_id,
                    source=f"{expr} (fallback: {fallback_src})",
                    target_register=target_reg,
                    type_str=typ,
                    is_expression=True
                ))
            
            reg_idx += 1
    
    # 5. 恢復 exec mask（如果有條件）
    if has_condition:
        temp_sgpr_start = 20 + unique_id * 2
        temp_sgpr_end = temp_sgpr_start + 1
        lines.append(f'              // Restore exec mask')
        lines.append(f'              llvm.inline_asm has_side_effects asm_dialect = att "s_mov_b64 exec, s[{temp_sgpr_start}:{temp_sgpr_end}]", "" : () -> ()')
    
    lines.append(f'              // === End @CAPTURE #{unique_id} ===')
    
    return '\n'.join(lines), mappings


def inject_capture_into_mlir(gpumlir_text: str, 
                              directives: List[CaptureDirective],
                              allocator: RegisterAllocator) -> Tuple[str, List[CaptureMapping]]:
    """
    將 capture 指令注入到 GPU MLIR 中
    
    Returns:
        (modified_mlir, all_mappings): 修改後的 MLIR 和所有映射
    """
    if not directives:
        return gpumlir_text, []
    
    all_mappings = []
    capture_blocks = []
    
    # 生成所有 capture 代碼
    for i, directive in enumerate(directives):
        capture_code, mappings = generate_register_capture(directive, i, allocator)
        capture_blocks.append(capture_code)
        all_mappings.extend(mappings)
    
    capture_section = '\n'.join(capture_blocks)
    
    # 找到插入點：在 s_endpgm 之前插入，並移除原本的 s_endpgm
    lines = gpumlir_text.split('\n')
    modified_lines = []
    in_func = False
    capture_inserted = False
    
    for i, line in enumerate(lines):
        if 'gpu.func @' in line and 'kernel' in line:
            in_func = True
        
        # 在 s_endpgm 之前插入 capture 代碼，並跳過原本的 s_endpgm
        if in_func and not capture_inserted and 's_endpgm' in line:
            # 插入 capture 代碼（已包含 s_endpgm）
            modified_lines.append(capture_section)
            capture_inserted = True
            # 跳過原本的 s_endpgm 行（不添加到 modified_lines）
            continue
        
        modified_lines.append(line)
        
        if in_func and 'gpu.return' in line:
            in_func = False
    
    return '\n'.join(modified_lines), all_mappings


# ============================================================
# Metadata 修復（參考 mdr_capture.py 的實現）
# ============================================================

def extract_original_metadata(original_isa_file: pathlib.Path) -> Dict:
    """從原始 ISA 檔案中提取 metadata（參考 mdr_capture.py）"""
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
    
    # 提取 YAML metadata 中的 args
    yaml_match = re.search(r'\.amdgpu_metadata\s+---\s+(.*?)\.\.\.',
                          isa_text, re.DOTALL | re.MULTILINE)
    if yaml_match:
        try:
            yaml_text = yaml_match.group(1)
            metadata = yaml.safe_load(yaml_text)
            
            if 'amdhsa.kernels' in metadata and len(metadata['amdhsa.kernels']) > 0:
                kernel = metadata['amdhsa.kernels'][0]
                
                # 保存原始 kernel 定義（用於恢復其他欄位）
                result = {'attrs': attrs, 'args': original_args, 'yaml_orig_kernel': kernel.copy()}
                
                # 提取 args（不包含 .name 欄位，因為 llvm-mc 無法處理）
                if '.args' in kernel and isinstance(kernel['.args'], list):
                    for arg in kernel['.args']:
                        arg_dict = {}
                        for k, v in arg.items():
                            key_name = k[1:] if k.startswith('.') else k
                            # 跳過 .name 和 .actual_access 欄位（llvm-mc 不支持）
                            if key_name not in ['name', 'actual_access']:
                                arg_dict[key_name] = v
                        
                        if arg_dict:  # 只添加非空的 arg
                            original_args.append(arg_dict)
                
                # 檢查是否有 .agpr_count
                if '.agpr_count' in kernel:
                    attrs['agpr_count'] = kernel['.agpr_count']
            
            print(f"[Info] Extracted {len(attrs)} metadata attributes and {len(original_args)} args from {original_isa_file.name}")
            return result
            
        except Exception as e:
            print(f"[Warning] Failed to parse original ISA YAML metadata: {e}")
    
    return {'attrs': attrs, 'args': original_args}


def update_isa_metadata(isa_text: str, new_vgpr_count: int, 
                        new_sgpr_count: int, original_metadata: Dict = None) -> str:
    
    # 更新 .amdhsa_next_free_vgpr
    isa_text = re.sub(
        r'(\.amdhsa_next_free_vgpr)\s+\d+',
        rf'\1 {new_vgpr_count}',
        isa_text
    )
    
    # 如果有原始 metadata，恢復關鍵設定
    if original_metadata and 'attrs' in original_metadata:
        attrs = original_metadata['attrs']
        
        # 恢復關鍵的 .amdhsa_* 欄位
        if 'kernarg_size' in attrs:
            isa_text = re.sub(
                r'(\.amdhsa_kernarg_size)\s+\d+',
                rf'\1 {attrs["kernarg_size"]}',
                isa_text
            )
            print(f"[Info] Restored .amdhsa_kernarg_size = {attrs['kernarg_size']}")
        
        if 'user_sgpr_count' in attrs:
            isa_text = re.sub(
                r'(\.amdhsa_user_sgpr_count)\s+\d+',
                rf'\1 {attrs["user_sgpr_count"]}',
                isa_text
            )
            print(f"[Info] Restored .amdhsa_user_sgpr_count = {attrs['user_sgpr_count']}")
        
        if 'kernarg_segment_ptr' in attrs:
            isa_text = re.sub(
                r'(\.amdhsa_user_sgpr_kernarg_segment_ptr)\s+\d+',
                rf'\1 {attrs["kernarg_segment_ptr"]}',
                isa_text
            )
            print(f"[Info] Restored .amdhsa_user_sgpr_kernarg_segment_ptr = {attrs['kernarg_segment_ptr']}")
        
        if 'next_free_sgpr' in attrs:
            isa_text = re.sub(
                r'(\.amdhsa_next_free_sgpr)\s+\d+',
                rf'\1 {attrs["next_free_sgpr"]}',
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
                
                # 更新 VGPR count
                kernel['.vgpr_count'] = new_vgpr_count
                
                # 恢復原始 args（不包含 .name 欄位）
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
                        # 注意：不添加 .name 欄位（llvm-mc 無法處理）
                        kernel['.args'].append(yaml_arg)
                    
                    print(f"[Info] Restored {len(original_args)} args (without .name fields)")
                
                # 恢復其他關鍵欄位（從原始 YAML metadata）
                if original_metadata and 'attrs' in original_metadata:
                    attrs = original_metadata['attrs']
                    
                    if 'kernarg_size' in attrs:
                        kernel['.kernarg_segment_size'] = attrs['kernarg_size']
                    
                    if 'group_segment_fixed_size' in attrs:
                        kernel['.group_segment_fixed_size'] = attrs['group_segment_fixed_size']
                    
                    # 確保有 .agpr_count（llvm-mc 需要）
                    if '.agpr_count' not in kernel:
                        kernel['.agpr_count'] = attrs.get('agpr_count', 0)
                        print(f"[Info] Added .agpr_count = {kernel['.agpr_count']}")
                
                # 從原始 YAML 中恢復其他欄位（如 .sgpr_count, .kernarg_segment_size 等）
                if original_metadata and 'yaml_orig_kernel' in original_metadata:
                    orig_kernel = original_metadata['yaml_orig_kernel']
                    
                    # 恢復這些欄位（如果原始檔案有）
                    restore_fields = [
                        '.kernarg_segment_size',
                        '.kernarg_segment_align',
                        '.sgpr_count',
                        '.max_flat_workgroup_size',
                        '.language',
                        '.language_version',
                    ]
                    
                    for field in restore_fields:
                        if field in orig_kernel:
                            kernel[field] = orig_kernel[field]
                
                # 重新生成 YAML（移除 .name 欄位後）
                fixed_yaml = yaml.dump(gen_metadata, default_flow_style=False, sort_keys=False)
                
                before_metadata = isa_text[:yaml_start]
                after_metadata = isa_text[yaml_end+3:]
                
                isa_text = before_metadata + "---\n" + fixed_yaml + "..." + after_metadata
                
                print(f"[Info] Updated YAML metadata: VGPR count = {new_vgpr_count}")
        
        except Exception as e:
            print(f"[Warning] Failed to update YAML metadata: {e}")
            import traceback
            traceback.print_exc()
    
    return isa_text


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


def build_capture_hsaco(gpumlir_path: pathlib.Path, chip: str, workdir: pathlib.Path,
                        new_vgpr_count: int, new_sgpr_count: int,
                        original_isa_file: pathlib.Path = None):
    """從修改後的 GPU MLIR 生成 HSACO"""
    for tool in ["mlir-opt", "llvm-mc", "lld"]:
        ensure_tool(tool)
    
    stem = gpumlir_path.stem
    binary_mlir = workdir / f"{stem}_binary.mlir"
    isa_path = workdir / f"{stem}.s"
    obj_path = workdir / f"{stem}.o"
    hsaco_path = workdir / f"{stem}.hsaco"
    
    # 提取原始 metadata
    original_metadata = extract_original_metadata(original_isa_file) if original_isa_file else {}
    
    print(f"\n=== Stage 2: Running MLIR optimization pipeline ===")
    
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
    
    # 更新 register metadata 並恢復原始 metadata
    isa_text = update_isa_metadata(isa_text, new_vgpr_count, new_sgpr_count, original_metadata)
    
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
        # 使用索引來匹配 directive_id
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
    content.append("")
    content.append("使用說明：")
    content.append("  1. 這些 register 保存了捕獲時刻的快照")
    content.append("  2. 您可以在後續的 assembly 代碼中繼續使用這些 register")
    content.append("  3. 注意不要在後續代碼中覆蓋這些 register 的值")
    content.append("")
    content.append("範例：")
    content.append("  ; 假設 v2 被捕獲到 v10")
    content.append("  ; 原始代碼修改了 v2：")
    content.append("  v_add_f32 v2, v2, v3")
    content.append("  ; 但 v10 仍保留原始的 v2 值，可以繼續使用：")
    content.append("  v_mul_f32 v11, v10, 2.0  ; 使用捕獲的原始值")
    content.append("")
    
    mapping_file.write_text('\n'.join(content))
    print(f"\n✓ Generated mapping file: {mapping_file}")
    
    # 也在終端輸出
    print("\n" + "\n".join(content[:20]))  # 只顯示前 20 行


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
     ; @CAPTURE reg=v2,v3 type=f32,f32
     ; @CAPTURE cond=tid_eq(0) reg=v2 expr="v2*2.0" type=f32,f32

  2. 執行工具：
     python3 mdr_capture_v2.py input.s --output-dir output

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
        help="只解析和分配 register，不執行編譯"
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
    print(f"\n=== Parsing @CAPTURE directives ===")
    lines, directives = parse_asm_file(input_path)
    
    if not directives:
        print("[Error] No @CAPTURE directives found")
        return
    
    print(f"\nFound {len(directives)} @CAPTURE directive(s)")
    
    # 2. 分析暫存器使用量
    print(f"\n=== Analyzing register usage ===")
    reg_info = analyze_registers(asm_text)
    print(f"  Original usage - VGPR: v0-v{reg_info['vgpr']-1} ({reg_info['vgpr']} total)")
    print(f"                   SGPR: s0-s{reg_info['sgpr']-1} ({reg_info['sgpr']} total)")
    
    # 3. 創建 register allocator
    allocator = RegisterAllocator(reg_info['vgpr'], reg_info['sgpr'], reg_info['agpr'])
    
    # 計算需要的 register 數量
    total_values = sum(
        len(d.registers) + (len(d.expressions) if d.expressions else 0) 
        for d in directives
    )
    print(f"\n  Need to allocate: {total_values} VGPR(s)")
    
    # 檢查是否會超過限制
    if allocator.next_vgpr + total_values > allocator.VGPR_LIMIT:
        print(f"\n❌ ERROR: Not enough VGPR available!")
        print(f"   Current usage: v0-v{allocator.next_vgpr-1}")
        print(f"   Required: {total_values} more")
        print(f"   Limit: v0-v{allocator.VGPR_LIMIT-1}")
        print(f"\n   建議：")
        print(f"   1. 減少 @CAPTURE 的數量")
        print(f"   2. 優化原始代碼以減少 register 使用")
        print(f"   3. 使用條件捕獲（cond=...）減少執行的 threads")
        return
    
    print(f"  Will allocate: v{allocator.next_vgpr}-v{allocator.next_vgpr + total_values - 1}")
    
    if args.dry_run:
        print("\n[Dry Run] Would generate the following mapping:")
        # 模擬分配
        temp_allocator = RegisterAllocator(reg_info['vgpr'], reg_info['sgpr'], reg_info['agpr'])
        for i, directive in enumerate(directives):
            num_vals = len(directive.registers) + (len(directive.expressions) if directive.expressions else 0)
            allocated = temp_allocator.allocate_vgpr(num_vals)
            print(f"\n  CAPTURE #{i} (line {directive.line_number + 1}):")
            
            idx = 0
            for reg in directive.registers:
                print(f"    {reg} → {allocated[idx]}")
                idx += 1
            
            if directive.expressions:
                for expr in directive.expressions:
                    print(f"    {expr} → {allocated[idx]}")
                    idx += 1
        
        print("\n[Dry Run] Stopping here.")
        return
    
    # 4. 轉換為 GPU MLIR
    gpumlir_path = translate_to_gpumlir(input_path, workdir)
    
    # 5. 注入 capture 程式碼
    print(f"\n=== Injecting capture code ===")
    gpumlir_text = gpumlir_path.read_text()
    
    try:
        modified_mlir, mappings = inject_capture_into_mlir(gpumlir_text, directives, allocator)
    except RuntimeError as e:
        print(f"\n❌ ERROR: {e}")
        return
    
    modified_path = workdir / f"{input_path.stem}_capture_injected.gpumlir"
    modified_path.write_text(modified_mlir)
    print(f"Generated modified GPU MLIR: {modified_path}")
    
    # 6. 獲取最終的 register 使用量
    final_usage = allocator.get_final_usage()
    print(f"\n=== Final register usage ===")
    print(f"  VGPR: v0-v{final_usage['vgpr']-1} ({final_usage['vgpr']} total)")
    print(f"  SGPR: s0-s{final_usage['sgpr']-1} ({final_usage['sgpr']} total)")
    
    # 7. 生成 HSACO
    hsaco_path, isa_path = build_capture_hsaco(
        modified_path, args.chip, workdir,
        final_usage['vgpr'], final_usage['sgpr'],
        input_path
    )
    
    # 8. 生成映射文件
    generate_mapping_file(workdir, directives, mappings, isa_path)
    
    print(f"\n=== Done ===")
    print(f"Capture HSACO: {hsaco_path}")
    print(f"Generated ISA: {isa_path}")
    print(f"Mapping file: {workdir}/capture_mapping.txt")
    print(f"\n您現在可以：")
    print(f"  1. 查看映射文件瞭解值存在哪些 register")
    print(f"  2. 在生成的 .s 文件中繼續使用這些 register")
    print(f"  3. 使用 {hsaco_path} 替換原始的 HSACO")


if __name__ == "__main__":
    main()
