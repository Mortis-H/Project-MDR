#!/usr/bin/env python3
"""
自動為 GPU MLIR 添加 Register Clobber
分析 inline_asm 使用的暫存器範圍，並在 gpu.func 開頭插入 clobber 聲明
"""

import re
import pathlib
import argparse
import sys


def analyze_registers_in_gpumlir(mlir_content: str) -> tuple[int, int]:
    """
    分析 GPU MLIR 中所有 inline_asm 使用的暫存器
    返回: (max_vgpr, max_sgpr)
    """
    vgprs = set()
    sgprs = set()
    
    # 匹配所有 llvm.inline_asm 中的暫存器使用
    # 支持多種格式: v0, v[0:1], s2, s[4:5]
    asm_pattern = r'llvm\.inline_asm.*?"([^"]*)"'
    
    for match in re.finditer(asm_pattern, mlir_content, re.DOTALL):
        asm_code = match.group(1)
        
        # 匹配 v123 格式
        for v_match in re.finditer(r'\bv(\d+)\b', asm_code):
            vgprs.add(int(v_match.group(1)))
        
        # 匹配 v[123:456] 格式
        for v_range_match in re.finditer(r'\bv\[(\d+):(\d+)\]', asm_code):
            start = int(v_range_match.group(1))
            end = int(v_range_match.group(2))
            for i in range(start, end + 1):
                vgprs.add(i)
        
        # 匹配 s123 格式
        for s_match in re.finditer(r'\bs(\d+)\b', asm_code):
            sgprs.add(int(s_match.group(1)))
        
        # 匹配 s[123:456] 格式
        for s_range_match in re.finditer(r'\bs\[(\d+):(\d+)\]', asm_code):
            start = int(s_range_match.group(1))
            end = int(s_range_match.group(2))
            for i in range(start, end + 1):
                sgprs.add(i)
    
    max_vgpr = max(vgprs) if vgprs else -1
    max_sgpr = max(sgprs) if sgprs else -1
    
    return max_vgpr, max_sgpr


def generate_clobber_lines(max_vgpr: int, max_sgpr: int) -> list[str]:
    """
    生成 clobber 保留（reserve）的 MLIR 代碼
    """
    lines = []
    
    if max_vgpr >= 0:
        vgpr_count = max_vgpr + 1
        lines.append(f'    // Reserve VGPR: v[0:{max_vgpr}] ({vgpr_count} registers)')
        lines.append(f'    %vgpr_reserved = llvm.inline_asm has_side_effects asm_dialect = att "", "={{v[0:{max_vgpr}]}}" : () -> vector<{vgpr_count}xi32>')
    
    if max_sgpr >= 0:
        sgpr_count = max_sgpr + 1
        lines.append(f'    // Reserve SGPR: s[0:{max_sgpr}] ({sgpr_count} registers)')
        lines.append(f'    %sgpr_reserved = llvm.inline_asm has_side_effects asm_dialect = att "", "={{s[0:{max_sgpr}]}}" : () -> vector<{sgpr_count}xi32>')
    
    return lines


def generate_clobber_release_lines(max_vgpr: int, max_sgpr: int) -> list[str]:
    """
    生成 clobber 釋放的 MLIR 代碼
    """
    lines = []
    
    if max_vgpr >= 0:
        vgpr_count = max_vgpr + 1
        lines.append(f'    // Release VGPR clobber: v[0:{max_vgpr}]')
        lines.append(f'    llvm.inline_asm has_side_effects asm_dialect = att "", "{{v[0:{max_vgpr}]}}" %vgpr_reserved : (vector<{vgpr_count}xi32>) -> ()')
    
    if max_sgpr >= 0:
        sgpr_count = max_sgpr + 1
        lines.append(f'    // Release SGPR clobber: s[0:{max_sgpr}]')
        lines.append(f'    llvm.inline_asm has_side_effects asm_dialect = att "", "{{s[0:{max_sgpr}]}}" %sgpr_reserved : (vector<{sgpr_count}xi32>) -> ()')
    
    return lines


def add_clobber_to_gpumlir(input_content: str, max_vgpr: int, max_sgpr: int) -> str:
    """
    在 gpu.func 開頭和結尾添加 register clobber (reserve & release)
    """
    lines = input_content.splitlines()
    new_lines = []
    
    # 狀態機
    clobber_inserted = False
    release_inserted = False
    
    for line in lines:
        # 找到 gpu.return，在之前插入 release
        if 'gpu.return' in line and not release_inserted:
            new_lines.append('')
            new_lines.append('    // ===== Release Register Clobber =====')
            release_lines = generate_clobber_release_lines(max_vgpr, max_sgpr)
            new_lines.extend(release_lines)
            new_lines.append('    // =====================================')
            new_lines.append('')
            release_inserted = True
        
        new_lines.append(line)
        
        # 找到 gpu.func ... kernel {
        # 支持: gpu.func @name() kernel {
        #      gpu.func @name() kernel attributes {...} {
        if re.search(r'gpu\.func\s+@\S+.*\bkernel\b.*\{', line) and not clobber_inserted:
            # 在下一行插入 clobber reserve
            new_lines.append('')
            new_lines.append('    // ===== Reserve Register Clobber =====')
            clobber_lines = generate_clobber_lines(max_vgpr, max_sgpr)
            new_lines.extend(clobber_lines)
            new_lines.append('    // =====================================')
            new_lines.append('')
            clobber_inserted = True
    
    if not clobber_inserted:
        print("[Warning] No gpu.func kernel found, clobber not inserted", file=sys.stderr)
    
    if not release_inserted:
        print("[Warning] No gpu.return found, clobber not released", file=sys.stderr)
    
    return '\n'.join(new_lines)


def main():
    parser = argparse.ArgumentParser(
        description="自動為 GPU MLIR 添加 Register Clobber，讓 LLVM 正確計算資源需求"
    )
    parser.add_argument("input_mlir", type=pathlib.Path, help="輸入 .gpumlir 文件")
    parser.add_argument("-o", "--output", type=pathlib.Path, help="輸出文件（默認覆蓋輸入）")
    parser.add_argument("--dry-run", action="store_true", help="只分析暫存器使用，不修改文件")
    
    args = parser.parse_args()
    
    if not args.input_mlir.exists():
        print(f"[Error] Input file not found: {args.input_mlir}", file=sys.stderr)
        sys.exit(1)
    
    # 讀取輸入
    input_content = args.input_mlir.read_text()
    
    # 分析暫存器使用
    max_vgpr, max_sgpr = analyze_registers_in_gpumlir(input_content)
    
    print(f"[Info] Register usage analysis:", file=sys.stderr)
    print(f"  - VGPR: 0-{max_vgpr} ({max_vgpr+1 if max_vgpr >= 0 else 0} registers)", file=sys.stderr)
    print(f"  - SGPR: 0-{max_sgpr} ({max_sgpr+1 if max_sgpr >= 0 else 0} registers)", file=sys.stderr)
    
    if args.dry_run:
        print("[Info] Dry-run mode, no files modified", file=sys.stderr)
        return
    
    # 添加 clobber
    output_content = add_clobber_to_gpumlir(input_content, max_vgpr, max_sgpr)
    
    # 寫入輸出
    output_file = args.output or args.input_mlir
    output_file.write_text(output_content)
    
    print(f"✓ Successfully added register clobber to: {output_file}", file=sys.stderr)


if __name__ == "__main__":
    main()

