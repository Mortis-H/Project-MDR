#!/usr/bin/env python3
"""
ISA Wrapper Tool
將 LLM 生成的純指令 ISA 包裝成完整的 ISA 文件
"""

import re
import argparse
from pathlib import Path
from typing import Tuple, Set, Dict


class ISAWrapper:
    def __init__(self, old_isa_path: str, new_isa_path: str, output_path: str):
        self.old_isa_path = Path(old_isa_path)
        self.new_isa_path = Path(new_isa_path)
        self.output_path = Path(output_path)
        
    def read_file(self, path: Path) -> str:
        """讀取文件內容"""
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def write_file(self, path: Path, content: str):
        """寫入文件內容"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def extract_function_name(self, content: str) -> str:
        """從 ISA 中提取函數名稱"""
        # 尋找函數標籤，例如：_Z6matDetPdS_:
        match = re.search(r'^([a-zA-Z_][\w]*):.*?@\1', content, re.MULTILINE)
        if match:
            return match.group(1)
        # 如果沒找到，嘗試尋找其他格式
        match = re.search(r'^\.globl\s+([a-zA-Z_][\w]*)', content, re.MULTILINE)
        if match:
            return match.group(1)
        raise ValueError("無法從 ISA 中找到函數名稱")
    
    def split_old_isa(self, content: str, func_name: str) -> Tuple[str, str, str]:
        """
        將舊的 ISA 分成三部分：頭部、函數體、尾部
        返回：(header, body, footer)
        """
        lines = content.split('\n')
        
        # 找到函數開始的位置（函數標籤）
        func_label = f"{func_name}:"
        func_start_idx = -1
        for i, line in enumerate(lines):
            if line.startswith(func_label):
                func_start_idx = i
                break
        
        if func_start_idx == -1:
            raise ValueError(f"無法在舊 ISA 中找到函數標籤: {func_label}")
        
        # 找到函數結束的位置（s_endpgm 之後）
        func_end_idx = -1
        for i in range(func_start_idx, len(lines)):
            if 's_endpgm' in lines[i]:
                func_end_idx = i
                break
        
        if func_end_idx == -1:
            raise ValueError("無法在舊 ISA 中找到 s_endpgm")
        
        # 頭部：從開始到函數標籤之前
        header = '\n'.join(lines[:func_start_idx])
        
        # 函數體：從函數標籤到 s_endpgm
        body = '\n'.join(lines[func_start_idx:func_end_idx + 1])
        
        # 尾部：從 s_endpgm 之後到下一個函數或文件結束
        footer_start_idx = func_end_idx + 1
        footer_end_idx = len(lines)
        
        # 尋找下一個函數的開始（通過 .protected 或 .globl）
        for i in range(footer_start_idx, len(lines)):
            if lines[i].strip().startswith('.protected') or \
               (lines[i].strip().startswith('.globl') and i > footer_start_idx + 10):
                footer_end_idx = i
                break
        
        footer = '\n'.join(lines[footer_start_idx:footer_end_idx])
        
        return header, body, footer
    
    def analyze_registers(self, isa_code: str) -> Dict[str, int]:
        """
        分析 ISA 指令中使用的寄存器
        返回：{'vgpr': max_vgpr, 'sgpr': max_sgpr, 'agpr': max_agpr, 'uses_vcc': 0/1}
        """
        max_vgpr = 0
        max_sgpr = 0
        max_agpr = 0
        uses_vcc = 0
        
        # 正則表達式模式
        vgpr_pattern = re.compile(r'\bv(\d+)\b')
        sgpr_pattern = re.compile(r'\bs(\d+)\b')
        agpr_pattern = re.compile(r'\ba(\d+)\b')
        vgpr_range_pattern = re.compile(r'\bv\[(\d+):(\d+)\]')
        sgpr_range_pattern = re.compile(r'\bs\[(\d+):(\d+)\]')
        agpr_range_pattern = re.compile(r'\ba\[(\d+):(\d+)\]')
        vcc_pattern = re.compile(r'\bvcc\b')
        
        for line in isa_code.split('\n'):
            # 跳過註釋和標籤
            code_part = line.split(';')[0].split('#')[0].strip()
            if not code_part or code_part.endswith(':'):
                continue
            
            # 檢查 VCC 使用
            if vcc_pattern.search(code_part):
                uses_vcc = 1
            
            # 檢查 VGPR 範圍 v[start:end]
            for match in vgpr_range_pattern.finditer(code_part):
                start, end = int(match.group(1)), int(match.group(2))
                max_vgpr = max(max_vgpr, end + 1)
            
            # 檢查 SGPR 範圍 s[start:end]
            for match in sgpr_range_pattern.finditer(code_part):
                start, end = int(match.group(1)), int(match.group(2))
                max_sgpr = max(max_sgpr, end + 1)
            
            # 檢查 AGPR 範圍 a[start:end]
            for match in agpr_range_pattern.finditer(code_part):
                start, end = int(match.group(1)), int(match.group(2))
                max_agpr = max(max_agpr, end + 1)
            
            # 檢查單個 VGPR
            for match in vgpr_pattern.finditer(code_part):
                vgpr_num = int(match.group(1))
                max_vgpr = max(max_vgpr, vgpr_num + 1)
            
            # 檢查單個 SGPR
            for match in sgpr_pattern.finditer(code_part):
                sgpr_num = int(match.group(1))
                max_sgpr = max(max_sgpr, sgpr_num + 1)
            
            # 檢查單個 AGPR
            for match in agpr_pattern.finditer(code_part):
                agpr_num = int(match.group(1))
                max_agpr = max(max_agpr, agpr_num + 1)
        
        return {
            'vgpr': max_vgpr,
            'sgpr': max_sgpr,
            'agpr': max_agpr,
            'uses_vcc': uses_vcc
        }
    
    def update_register_metadata(self, footer: str, reg_info: Dict[str, int]) -> str:
        """
        更新尾部元數據中的寄存器信息
        """
        lines = footer.split('\n')
        updated_lines = []
        
        # 計算 accum_offset（AGPR 的起始位置）
        # accum_offset 必須是 4 的倍數，且 >= VGPR 數量
        # 公式：將 VGPR 向上取整到下一個 4 的倍數
        accum_offset = ((reg_info['vgpr'] + 3) // 4) * 4
        # 確保在有效範圍內 [4..256]
        accum_offset = max(4, min(256, accum_offset))
        
        # 計算 TotalNumSgprs（需要考慮 VCC、Flat Scratch 等）
        # 通常 TotalNumSgprs = numbered_sgpr + (uses_vcc ? 2 : 0) + ...
        # 這裡簡化為 sgpr + 2（如果使用 VCC）+ 4（系統寄存器）
        total_sgpr = reg_info['sgpr'] + (2 if reg_info['uses_vcc'] else 0) + 4
        
        # SGPR blocks = ceil((total_sgpr - 1) / 8)
        sgpr_blocks = (total_sgpr + 7) // 8
        
        # VGPR blocks = ceil((vgpr - 1) / 4)  for gfx9+
        vgpr_blocks = (reg_info['vgpr'] + 3) // 4
        
        for line in lines:
            updated_line = line
            
            # 更新 .amdhsa_next_free_vgpr
            if '.amdhsa_next_free_vgpr' in line:
                updated_line = re.sub(r'\.amdhsa_next_free_vgpr\s+\d+', 
                                     f'.amdhsa_next_free_vgpr {reg_info["vgpr"]}', line)
            
            # 更新 .amdhsa_next_free_sgpr
            elif '.amdhsa_next_free_sgpr' in line:
                updated_line = re.sub(r'\.amdhsa_next_free_sgpr\s+\d+', 
                                     f'.amdhsa_next_free_sgpr {reg_info["sgpr"]}', line)
            
            # 更新 .amdhsa_accum_offset
            elif '.amdhsa_accum_offset' in line:
                updated_line = re.sub(r'\.amdhsa_accum_offset\s+\d+', 
                                     f'.amdhsa_accum_offset {accum_offset}', line)
            
            # 更新 .amdhsa_reserve_vcc
            elif '.amdhsa_reserve_vcc' in line:
                updated_line = re.sub(r'\.amdhsa_reserve_vcc\s+\d+', 
                                     f'.amdhsa_reserve_vcc {reg_info["uses_vcc"]}', line)
            
            # 更新 .set num_vgpr
            elif '.num_vgpr' in line:
                updated_line = re.sub(r'\.num_vgpr,\s*\d+', 
                                     f'.num_vgpr, {reg_info["vgpr"]}', line)
            
            # 更新 .set num_agpr
            elif '.num_agpr' in line:
                updated_line = re.sub(r'\.num_agpr,\s*\d+', 
                                     f'.num_agpr, {reg_info["agpr"]}', line)
            
            # 更新 .set numbered_sgpr
            elif '.numbered_sgpr' in line:
                updated_line = re.sub(r'\.numbered_sgpr,\s*\d+', 
                                     f'.numbered_sgpr, {reg_info["sgpr"]}', line)
            
            # 更新 .set uses_vcc
            elif '.uses_vcc' in line:
                updated_line = re.sub(r'\.uses_vcc,\s*\d+', 
                                     f'.uses_vcc, {reg_info["uses_vcc"]}', line)
            
            # 更新註釋中的 NumVgprs
            elif '; NumVgprs:' in line:
                updated_line = re.sub(r'; NumVgprs:\s*\d+', 
                                     f'; NumVgprs: {reg_info["vgpr"]}', line)
            
            # 更新註釋中的 NumAgprs
            elif '; NumAgprs:' in line:
                updated_line = re.sub(r'; NumAgprs:\s*\d+', 
                                     f'; NumAgprs: {reg_info["agpr"]}', line)
            
            # 更新註釋中的 TotalNumSgprs
            elif '; TotalNumSgprs:' in line:
                updated_line = re.sub(r'; TotalNumSgprs:\s*\d+', 
                                     f'; TotalNumSgprs: {total_sgpr}', line)
            
            # 更新註釋中的 TotalNumVgprs
            elif '; TotalNumVgprs:' in line:
                updated_line = re.sub(r'; TotalNumVgprs:\s*\d+', 
                                     f'; TotalNumVgprs: {reg_info["vgpr"]}', line)
            
            # 更新註釋中的 SGPRBlocks
            elif '; SGPRBlocks:' in line:
                updated_line = re.sub(r'; SGPRBlocks:\s*\d+', 
                                     f'; SGPRBlocks: {sgpr_blocks}', line)
            
            # 更新註釋中的 VGPRBlocks
            elif '; VGPRBlocks:' in line:
                updated_line = re.sub(r'; VGPRBlocks:\s*\d+', 
                                     f'; VGPRBlocks: {vgpr_blocks}', line)
            
            # 更新註釋中的 NumSGPRsForWavesPerEU
            elif '; NumSGPRsForWavesPerEU:' in line:
                updated_line = re.sub(r'; NumSGPRsForWavesPerEU:\s*\d+', 
                                     f'; NumSGPRsForWavesPerEU: {total_sgpr}', line)
            
            # 更新註釋中的 NumVGPRsForWavesPerEU
            elif '; NumVGPRsForWavesPerEU:' in line:
                updated_line = re.sub(r'; NumVGPRsForWavesPerEU:\s*\d+', 
                                     f'; NumVGPRsForWavesPerEU: {reg_info["vgpr"]}', line)
            
            updated_lines.append(updated_line)
        
        return '\n'.join(updated_lines)
    
    def wrap(self):
        """主要的包裝流程"""
        print(f"讀取舊的完整 ISA: {self.old_isa_path}")
        old_isa_content = self.read_file(self.old_isa_path)
        
        print(f"讀取新的純指令 ISA: {self.new_isa_path}")
        new_isa_content = self.read_file(self.new_isa_path)
        
        # 提取函數名稱
        func_name = self.extract_function_name(new_isa_content)
        print(f"檢測到函數名稱: {func_name}")
        
        # 分割舊的 ISA
        print("分割舊的 ISA...")
        header, old_body, footer = self.split_old_isa(old_isa_content, func_name)
        
        # 分析新 ISA 的寄存器使用
        print("分析新 ISA 的寄存器使用...")
        reg_info = self.analyze_registers(new_isa_content)
        print(f"  VGPR: {reg_info['vgpr']}")
        print(f"  SGPR: {reg_info['sgpr']}")
        print(f"  AGPR: {reg_info['agpr']}")
        print(f"  使用 VCC: {'是' if reg_info['uses_vcc'] else '否'}")
        
        # 更新尾部元數據
        print("更新寄存器元數據...")
        updated_footer = self.update_register_metadata(footer, reg_info)
        
        # 組合完整的 ISA
        print("組合完整的新 ISA...")
        complete_isa = header + '\n' + new_isa_content + '\n' + updated_footer
        
        # 寫入輸出文件
        print(f"寫入輸出文件: {self.output_path}")
        self.write_file(self.output_path, complete_isa)
        
        print("完成！")


def main():
    parser = argparse.ArgumentParser(
        description='將 LLM 生成的純指令 ISA 包裝成完整的 ISA 文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  %(prog)s -o old.s -n new_func.s -O output.s
  %(prog)s --old-isa old.s --new-isa new_func.s --output output.s
        """
    )
    
    parser.add_argument('-o', '--old-isa', required=True,
                       help='舊的完整 ISA 文件路徑')
    parser.add_argument('-n', '--new-isa', required=True,
                       help='新的純指令 ISA 文件路徑')
    parser.add_argument('-O', '--output', required=True,
                       help='輸出的完整 ISA 文件路徑')
    
    args = parser.parse_args()
    
    try:
        wrapper = ISAWrapper(args.old_isa, args.new_isa, args.output)
        wrapper.wrap()
    except Exception as e:
        print(f"錯誤: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
