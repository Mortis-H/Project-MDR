#!/usr/bin/env python3
import argparse
import pathlib
import re
import shutil
import subprocess
import sys
import textwrap
import yaml


# ------------------------------------------------------------
# Helper: Run external command and check for failure.
# ------------------------------------------------------------

def run_cmd(cmd, cwd=None):
    print("[$]", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, cwd=cwd)

def ensure_tool(name: str):
    if shutil.which(name) is None:
        raise RuntimeError(f"Required tool '{name}' not found in PATH")

# ======================================
# String decoding for gpu.binary assembly (ISA) – same as before
# ======================================

ASM_ATTR_RE = re.compile(
    r'gpu\.binary\b.*?assembly\s*=\s*"((?:[^"\\]|\\.)*)"',
    re.DOTALL,
)


def decode_mlir_string(raw: str) -> str:
    """
    Decode an MLIR string literal into a Python str.

    MLIR for assembly often uses:
      \09 -> 0x09 (tab)
      \22 -> 0x22 (double quote)
      \0A -> 0x0A (newline)
    which is "backslash + two hex digits".
    """

    def _hex_repl(m: re.Match) -> str:
        hex_val = m.group(1)
        try:
            return chr(int(hex_val, 16))
        except ValueError:
            return "\\" + hex_val

    # First, convert \XX (two hex digits) to the corresponding character
    s = re.sub(r"\\([0-9A-Fa-f]{2})", _hex_repl, raw)
    # Then let unicode_escape handle \n, \t, \", \\, etc.
    return bytes(s, "utf-8").decode("unicode_escape")


def extract_assembly_strings(mlir_text: str):
    out = []
    for raw in ASM_ATTR_RE.findall(mlir_text):
        out.append(decode_mlir_string(raw))
    return out


# ======================================
# String decoding for offload/offloading bitcode (format=llvm)
# ======================================

BITCODE_ATTR_RE = re.compile(
    r'(?:offload|offloading)\s*=\s*"((?:[^"\\]|\\.)*)"',
    re.DOTALL,
)


def decode_mlir_bytes(raw: str) -> bytes:
    """
    Decode an MLIR string literal into raw bytes.

    Example:
      "BC\\C0\\DE5\\14\\00\\00..."  (backslash + two hex digits)
    """

    out = bytearray()
    i = 0
    n = len(raw)

    while i < n:
        ch = raw[i]
        if ch == "\\" and i + 1 < n:
            nxt = raw[i + 1]
            # \XX where X are hex digits
            if i + 2 < n and nxt in "0123456789abcdefABCDEF" and raw[i + 2] in "0123456789abcdefABCDEF":
                val = int(raw[i + 1 : i + 3], 16)
                out.append(val)
                i += 3
                continue
            # Some common escapes, just in case
            if nxt == "n":
                out.append(ord("\n"))
                i += 2
                continue
            if nxt == "t":
                out.append(ord("\t"))
                i += 2
                continue
            if nxt == "r":
                out.append(ord("\r"))
                i += 2
                continue
            if nxt == '"':
                out.append(ord('"'))
                i += 2
                continue
            if nxt == "\\":
                out.append(ord("\\"))
                i += 2
                continue
            # Fallback: keep the backslash as-is
            out.append(ord(ch))
            i += 1
        else:
            out.append(ord(ch))
            i += 1

    return bytes(out)


def extract_bitcode_bytes(mlir_text: str):
    out = []
    for raw in BITCODE_ATTR_RE.findall(mlir_text):
        out.append(decode_mlir_bytes(raw))
    return out


def auto_detect_mlir_libs() -> tuple[pathlib.Path, pathlib.Path] | tuple[None, None]:
    """
    嘗試自動偵測 libmlir_rocm_runtime.so 和 libmlir_runner_utils.so
    基於 PATH 中 mlir-runner (或 mlir-opt) 的位置。
    
    假設標準的 LLVM 構建布局：build/bin, build/lib。
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


def translate_asm_to_gpu(asm_file: pathlib.Path, workdir: pathlib.Path, output_prefix: str = None):
    """
    使用 amdisa-translate 將 .s 文件轉換為 .amdisamlir 和 .gpumlir
    
    Args:
        asm_file: 輸入的 .s assembly 文件
        workdir: 工作目錄
        output_prefix: 輸出文件前綴（如果為 None，使用輸入文件名）
    
    Returns:
        (amdisamlir_path, gpumlir_path): 生成的文件路徑
    """
    ensure_tool("amdisa-translate")
    
    asm_stem = output_prefix if output_prefix else asm_file.stem
    amdisamlir_file = workdir / f"{asm_stem}.amdisamlir"
    gpumlir_file = workdir / f"{asm_stem}.gpumlir"
    
    # 首先從原始 ISA 文件提取 group_segment_fixed_size（因為 amdisa-import 會丟失這個值）
    asm_text = asm_file.read_text()
    group_segment_size = None
    match = re.search(r'\.amdhsa_group_segment_fixed_size\s+(\d+)', asm_text)
    if match:
        group_segment_size = int(match.group(1))
        print(f"[Info] Extracted group_segment_fixed_size from original ISA: {group_segment_size}")
    
    # Step 1: .s -> .amdisamlir
    print(f"\n=== Stage 1: Translating {asm_file.name} to AMDISA MLIR ===")
    amdisa_cmd = [
        "amdisa-translate",
        "-x", "s",
        "-emit=mlir",
        str(asm_file),
    ]
    result = subprocess.run(amdisa_cmd, capture_output=True, text=True, check=True)
    amdisamlir_file.write_text(result.stdout)
    print(f"Generated AMDISA MLIR: {amdisamlir_file}")
    
    # Step 2: .amdisamlir -> .gpumlir
    print(f"\n=== Stage 2: Lowering AMDISA MLIR to GPU MLIR ===")
    gpu_cmd = [
        "amdisa-translate",
        "-x", "mlir",
        "-emit=gpu",
        str(amdisamlir_file),
    ]
    result = subprocess.run(gpu_cmd, capture_output=True, text=True, check=True)
    gpumlir_text = result.stdout
    
    # 手動注入 group_segment_fixed_size 到 GPU MLIR 的 module attributes
    if group_segment_size is not None:
        # 在 module attributes 中添加 amdisa.group_segment_fixed_size
        # 找到 module attributes {... 的位置
        module_attr_pattern = r'(module attributes \{[^}]*)'
        match = re.search(module_attr_pattern, gpumlir_text)
        if match:
            attrs_str = match.group(1)
            # 在最後添加 group_segment_fixed_size attribute（在 } 之前）
            new_attrs_str = attrs_str + f', amdisa.group_segment_fixed_size = {group_segment_size} : i32'
            gpumlir_text = gpumlir_text.replace(attrs_str, new_attrs_str)
            print(f"[Info] Injected amdisa.group_segment_fixed_size = {group_segment_size} into GPU MLIR")
        else:
            print("[Warning] Could not find module attributes in GPU MLIR to inject group_segment_fixed_size")
    
    gpumlir_file.write_text(gpumlir_text)
    print(f"Generated GPU MLIR: {gpumlir_file}")
    
    return amdisamlir_file, gpumlir_file


def append_global_symbols(isa_text: str, original_isa_file: pathlib.Path) -> str:
    """
    從原始 ISA 文件提取全局變數定義並附加到重建的 ISA
    
    只提取 .bss, .data, .rodata 段中的全局變數，排除函數定義和已存在的符號。
    **重要**：保留 .protected, .hidden, .type 等可見性和類型屬性，這些屬性對於
    支持 PC-relative 重定位（如 @rel32@lo/hi）至關重要。
    
    Args:
        isa_text: 重建的 ISA 文本
        original_isa_file: 原始 ISA 文件路徑
    
    Returns:
        附加了全局符號定義的 ISA 文本
    """
    if not original_isa_file or not original_isa_file.exists():
        return isa_text
    
    original_text = original_isa_file.read_text()
    lines = original_text.split('\n')
    
    # 提取 rebuilt ISA 中已經存在的符號，避免重複定義
    existing_symbols = set()
    for match in re.finditer(r'\.(globl|protected|hidden)\s+([\w.$]+)', isa_text):
        existing_symbols.add(match.group(2))
    for match in re.finditer(r'\.type\s+([\w.$]+),@object', isa_text):
        existing_symbols.add(match.group(1))
    # 匹配標籤定義，允許前導空白，匹配包含 . 的符號
    for match in re.finditer(r'^\s*([\w.$]+):', isa_text, re.MULTILINE):
        symbol = match.group(1)
        if not symbol.startswith('.L'): # 排除局部標籤
            existing_symbols.add(symbol)
    
    # 找到所有全局/protected/hidden 符號
    # 我們採用基於符號的策略：找到每個符號的完整定義塊
    symbol_blocks = {}  # symbol_name -> (start_line, end_line)
    
    # 第一步：找到所有可能的全局符號定義
    # 查找所有 .globl, .protected, .hidden 聲明，以及 .type xxx,@object 聲明
    for i, line in enumerate(lines):
        # 匹配 .globl/.protected/.hidden 指令
        match = re.match(r'^\s*\.(globl|protected|hidden)\s+([\w.$]+)', line)
        if match:
            symbol_name = match.group(2)
            if symbol_name not in existing_symbols:
                if symbol_name not in symbol_blocks:
                    symbol_blocks[symbol_name] = {'start': i, 'end': i, 'has_definition': False}
        
        # 也匹配 .type xxx,@object 聲明（用於字符串常量等沒有可見性聲明的符號）
        type_match = re.match(r'^\s*\.type\s+([\w.$]+),@object', line)
        if type_match:
            symbol_name = type_match.group(1)
            if symbol_name not in existing_symbols:
                if symbol_name not in symbol_blocks:
                    symbol_blocks[symbol_name] = {'start': i, 'end': i, 'has_definition': False}
    
    # 第二步：為每個符號找到其定義範圍
    for symbol_name in list(symbol_blocks.keys()):
        start_line = symbol_blocks[symbol_name]['start']
        
        # 向前查找，看是否還有相關的屬性指令（.type, .protected 等）
        # 例如：.protected 在前，.type 在後
        for i in range(start_line - 1, max(0, start_line - 10), -1):
            line = lines[i].strip()
            # 如果是空行，跳過
            if not line:
                continue
            # 如果是與當前符號相關的屬性指令，擴展範圍
            if re.match(rf'^\.(protected|hidden|type|weak|local)\s+{re.escape(symbol_name)}\b', line):
                symbol_blocks[symbol_name]['start'] = i
            # 如果遇到其他內容（不相關的指令），停止
            elif not re.match(r'^\s*[;#]', line):  # 允許註釋
                break
        
        # 向後查找符號定義和內容
        found_label = False
        found_section = False
        for i in range(start_line, len(lines)):
            line = lines[i]
            stripped = line.strip()
            
            # 檢查是否到達符號定義（標籤）
            if re.match(rf'^{re.escape(symbol_name)}:', stripped):
                found_label = True
                symbol_blocks[symbol_name]['has_definition'] = True
                symbol_blocks[symbol_name]['end'] = i
                continue
            
            # 找到 .section 指令
            if re.match(r'^\s*\.section', line):
                found_section = True
                symbol_blocks[symbol_name]['end'] = i
                continue
            
            # 如果已經找到標籤，繼續往後看內容（.zero, .size 等）
            if found_label:
                # 如果遇到下一個段、另一個符號定義、或者特殊段標記，停止
                if (stripped.startswith('.section') or 
                    stripped.startswith('.text') or
                    stripped.startswith('.amdgpu_metadata') or
                    stripped.startswith('.ident') or
                    stripped.startswith('.addrsig') or
                    (re.match(r'^[\w.$]+:', stripped) and stripped.split(':')[0] != symbol_name)):
                    break
                
                # 如果是符號相關的指令或數據，包含它
                if (stripped.startswith('.') or 
                    stripped.startswith('#') or 
                    stripped.startswith(';') or
                    not stripped):
                    symbol_blocks[symbol_name]['end'] = i
                else:
                    # 可能是符號的實際數據內容
                    symbol_blocks[symbol_name]['end'] = i
    
    # 第三步：追蹤每個符號所在的段
    # 掃描整個文件，追蹤當前活動段
    current_section = None
    for i, line in enumerate(lines):
        # 檢查段切換（支持兩種格式：.section .data 和 .data）
        section_match = re.search(r'\.section\s+\.(bss|data|rodata)(\.[\w.]+)?', line)
        if section_match:
            current_section = section_match.group(1)  # bss, data, or rodata
            continue
        
        # 檢查簡短格式的段切換（如 .data, .bss, .rodata）
        short_section_match = re.match(r'^\s*\.(bss|data|rodata)(\s|$)', line)
        if short_section_match:
            current_section = short_section_match.group(1)
            continue
        
        # 檢查其他會切換段的指令
        if re.match(r'^\s*\.(text|amdgpu_metadata)', line):
            current_section = None
            continue
        
        # 為符號塊記錄段信息
        for symbol_name, info in symbol_blocks.items():
            if info['start'] <= i <= info['end']:
                if 'section' not in info:
                    info['section'] = current_section
    
    # 第四步：提取符號塊並檢查是否在允許的段中
    global_sections = []
    new_symbols_found = []
    
    for symbol_name, info in symbol_blocks.items():
        if not info['has_definition']:
            continue
        
        # 提取符號塊
        start = info['start']
        end = info['end'] + 1
        symbol_block_lines = lines[start:end]
        symbol_block = '\n'.join(symbol_block_lines)
        
        # 檢查是否包含函數定義
        if re.search(r'\.type\s+\w+,@function', symbol_block):
            continue
        
        # 檢查是否在允許的段中
        # 方法1：符號塊內有 .section 指令或簡短格式段聲明
        has_section_directive = False
        for line in symbol_block_lines:
            # 檢查完整格式：.section .data
            if re.search(r'\.section\s+\.(bss|data|rodata)(\.|,|\s|$)', line):
                has_section_directive = True
                break
            # 檢查簡短格式：.data, .bss, .rodata
            if re.match(r'^\s*\.(bss|data|rodata)(\s|$)', line):
                has_section_directive = True
                break
        
        # 方法2：符號繼承了當前活動段
        in_data_section = info.get('section') in ('bss', 'data', 'rodata')
        
        if has_section_directive or in_data_section:
            # 如果符號沒有自己的 .section 指令，需要添加一個
            if not has_section_directive and in_data_section:
                section_name = info.get('section')
                if section_name == 'bss':
                    section_directive = f'\t.section\t.{section_name},"aw",@nobits'
                elif section_name == 'data':
                    section_directive = f'\t.section\t.{section_name},"aw",@progbits'
                else:  # rodata
                    section_directive = f'\t.section\t.{section_name},"a",@progbits'
                symbol_block = section_directive + '\n' + symbol_block
            
            global_sections.append(symbol_block.strip())
            new_symbols_found.append(symbol_name)
            print(f"[Info] Found global data symbols (not in rebuilt ISA): {symbol_name}")
    
    # 如果沒有找到任何新的全局變數段，直接返回
    if not global_sections:
        print("[Info] No additional global data symbols found in original ISA")
        return isa_text
    
    # 找到 .amdgpu_metadata 的位置
    metadata_pos = isa_text.find('.amdgpu_metadata')
    
    if metadata_pos == -1:
        # 如果沒有 metadata 段，附加到文件末尾
        print(f"[Info] Appending {len(global_sections)} global data section(s) to end of ISA")
        return isa_text + '\n\n' + '\n\n'.join(global_sections) + '\n'
    
    # 插入到 metadata 之前
    before_metadata = isa_text[:metadata_pos].rstrip()
    metadata_and_after = isa_text[metadata_pos:]
    
    print(f"[Info] Appending {len(global_sections)} global data section(s) before .amdgpu_metadata")
    
    global_text = '\n\n'.join(global_sections)
    return f"{before_metadata}\n\n{global_text}\n\n\t{metadata_and_after}"


def append_device_functions(isa_text: str, original_isa_file: pathlib.Path) -> str:
    """
    從原始 ISA 文件提取設備函數（device functions）並附加到重建的 ISA
    
    設備函數是非 kernel 函數，它們：
    - 有 .type xxx,@function 標記
    - 沒有 .amdhsa_kernel metadata
    - 通常被 kernels 或 vtables 調用
    
    Args:
        isa_text: 重建的 ISA 文本
        original_isa_file: 原始 ISA 文件路徑
    
    Returns:
        附加了設備函數的 ISA 文本
    """
    if not original_isa_file or not original_isa_file.exists():
        return isa_text
    
    original_text = original_isa_file.read_text()
    lines = original_text.split('\n')
    
    # 提取 rebuilt ISA 中已存在的函數名稱（kernels）
    existing_functions = set()
    for match in re.finditer(r'\.amdhsa_kernel\s+([\w.$]+)', isa_text):
        existing_functions.add(match.group(1))
    
    # 查找原始文件中的所有函數
    device_functions = []
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # 查找 .type xxx,@function 聲明
        func_match = re.match(r'^\s*\.type\s+([\w.$]+),@function', line)
        if func_match:
            func_name = func_match.group(1)
            
            # 檢查是否是 kernel（有 .amdhsa_kernel）
            is_kernel = False
            # 向前查找 50 行，看是否有 .amdhsa_kernel
            for j in range(i, min(i + 50, len(lines))):
                if re.search(rf'\.amdhsa_kernel\s+{re.escape(func_name)}', lines[j]):
                    is_kernel = True
                    break
            
            # 如果不是 kernel 且不在 rebuilt ISA 中，則提取它
            if not is_kernel and func_name not in existing_functions:
                # 找到函數的開始位置（向前查找可見性聲明）
                func_start = i
                for k in range(i - 1, max(0, i - 10), -1):
                    if re.match(r'^\s*\.(protected|hidden|globl|weak)\s+' + re.escape(func_name), lines[k]):
                        func_start = k
                    elif re.match(r'^\s*\.p2align', lines[k]):
                        func_start = k
                    elif lines[k].strip() and not re.match(r'^\s*[;#]', lines[k]):
                        break
                
                # 找到函數的結束位置（查找 .size 指令或下一個函數/段）
                func_end = i + 1
                for j in range(i + 1, len(lines)):
                    if re.search(rf'\.size\s+{re.escape(func_name)}', lines[j]):
                        func_end = j + 1
                        break
                    # 如果遇到下一個函數或段定義，停止
                    if (re.match(r'^\s*\.type\s+[\w.$]+,@(function|object)', lines[j]) or
                        re.match(r'^\s*\.section', lines[j]) or
                        re.match(r'^\s*\.amdgpu_metadata', lines[j])):
                        break
                    func_end = j + 1
                
                # 提取函數代碼
                func_lines = lines[func_start:func_end]
                func_text = '\n'.join(func_lines)
                
                # 重命名局部標籤以避免衝突
                # 局部標籤格式：.Lxxx（如 .Lfunc_end0, .LBB0_1）
                # 為每個函數的局部標籤添加唯一後綴
                local_label_suffix = f"_devfunc_{len(device_functions)}"
                # 匹配所有局部標籤的定義和引用
                # 不使用 \b，因為 . 不是單詞字符
                func_text = re.sub(r'(\.L[\w]+)', rf'\1{local_label_suffix}', func_text)
                
                device_functions.append(func_text.strip())
                existing_functions.add(func_name)
                print(f"[Info] Found device function (not in rebuilt ISA): {func_name}")
        
        i += 1
    
    if not device_functions:
        print("[Info] No device functions to append")
        return isa_text
    
    # 找到 .amdgpu_metadata 或全局數據段的位置，在其之前插入設備函數
    metadata_pos = isa_text.find('.amdgpu_metadata')
    
    if metadata_pos == -1:
        # 如果沒有 metadata，附加到末尾
        print(f"[Info] Appending {len(device_functions)} device function(s) to end of ISA")
        device_text = '\n\n'.join(device_functions)
        return f"{isa_text}\n\n{device_text}"
    
    # 在 metadata 之前插入
    before_metadata = isa_text[:metadata_pos].rstrip()
    metadata_and_after = isa_text[metadata_pos:]
    
    print(f"[Info] Appending {len(device_functions)} device function(s) before .amdgpu_metadata")
    
    # 設備函數必須在 .text 段中
    device_text = '\t.text\n\n' + '\n\n'.join(device_functions)
    return f"{before_metadata}\n\n{device_text}\n\n\t{metadata_and_after}"


def fix_isa_metadata(isa_text: str, gpumlir_file: pathlib.Path, original_isa_file: pathlib.Path = None) -> str:
    """
    修復 ISA metadata：從原始 ISA 文件或 GPU MLIR attributes 提取正確的 metadata
    
    MLIR pipeline (convert-gpu-to-rocdl, gpu-module-to-binary) 會丟失自定義的
    amdisa.* attributes，導致生成的 ISA metadata 不正確。
    
    此函數支持多 kernel 的 ISA 文件，為每個 kernel 分別提取和修復 metadata。
    
    Args:
        isa_text: 需要修復的 ISA 文本
        gpumlir_file: GPU MLIR 文件路徑（用於嘗試提取 amdisa.* attributes）
        original_isa_file: 原始 ISA 文件路徑（如果提供，優先從此提取 metadata）
    
    Returns:
        修復後的 ISA 文本
    """
    
    # Step 1: 識別重建 ISA 中的所有 kernel
    rebuilt_kernels = []
    for match in re.finditer(r'\.amdhsa_kernel\s+([\w.$]+)', isa_text):
        kernel_name = match.group(1)
        kernel_pos = match.start()
        rebuilt_kernels.append({'name': kernel_name, 'pos': kernel_pos})
    
    if not rebuilt_kernels:
        print("[Warning] No kernels found in rebuilt ISA")
        return isa_text
    
    print(f"[Info] Found {len(rebuilt_kernels)} kernel(s) in rebuilt ISA: {[k['name'] for k in rebuilt_kernels]}")
    
    # Step 2: 從原始 ISA 提取每個 kernel 的 metadata
    kernel_metadata_map = {}  # kernel_name -> {attrs, args_list}
    
    if original_isa_file is not None and original_isa_file.exists():
        print(f"[Info] Extracting metadata from original ISA: {original_isa_file.name}")
        original_isa_text = original_isa_file.read_text()
        
        # 找到原始 ISA 中的所有 kernel 及其位置
        original_kernels = []
        for match in re.finditer(r'\.amdhsa_kernel\s+([\w.$]+)', original_isa_text):
            kernel_name = match.group(1)
            kernel_start = match.start()
            original_kernels.append({'name': kernel_name, 'start': kernel_start})
        
        # 為每個 kernel 確定其 .amdhsa_* 指令的範圍
        for i, kernel_info in enumerate(original_kernels):
            # 找到下一個 kernel 的位置或 .end_amdhsa_kernel
            kernel_name = kernel_info['name']
            kernel_start = kernel_info['start']
            
            # 找到當前 kernel 的 .end_amdhsa_kernel
            end_pattern = r'\.end_amdhsa_kernel'
            end_match = re.search(end_pattern, original_isa_text[kernel_start:])
            if end_match:
                kernel_end = kernel_start + end_match.end()
            else:
                # 如果沒找到，使用下一個 kernel 的開始位置
                if i + 1 < len(original_kernels):
                    kernel_end = original_kernels[i + 1]['start']
                else:
                    kernel_end = len(original_isa_text)
            
            kernel_text = original_isa_text[kernel_start:kernel_end]
            
            # 從這個 kernel 的文本中提取 metadata
            attrs = {}
            # 修改正則表達式以支持符號表達式（不僅是數字）
            # 使用 .+?(?=\s*$) 匹配到行尾（支持符號表達式如 max(...)）
            amdhsa_patterns = {
                'kernarg_segment_size': r'\.amdhsa_kernarg_size\s+(.+?)(?:\s*$)',
                'group_segment_fixed_size': r'\.amdhsa_group_segment_fixed_size\s+(.+?)(?:\s*$)',
                'vgpr_count': r'\.amdhsa_next_free_vgpr\s+(.+?)(?:\s*$)',
                'amdhsa_next_free_sgpr': r'\.amdhsa_next_free_sgpr\s+(.+?)(?:\s*$)',  # 支持符號表達式
                'user_sgpr_count': r'\.amdhsa_user_sgpr_count\s+(.+?)(?:\s*$)',
                'dispatch_ptr': r'\.amdhsa_user_sgpr_dispatch_ptr\s+(.+?)(?:\s*$)',
                'queue_ptr': r'\.amdhsa_user_sgpr_queue_ptr\s+(.+?)(?:\s*$)',
                'workitem_id': r'\.amdhsa_system_vgpr_workitem_id\s+(.+?)(?:\s*$)',
                'workgroup_id_x': r'\.amdhsa_system_sgpr_workgroup_id_x\s+(.+?)(?:\s*$)',
                'workgroup_id_y': r'\.amdhsa_system_sgpr_workgroup_id_y\s+(.+?)(?:\s*$)',
                'workgroup_id_z': r'\.amdhsa_system_sgpr_workgroup_id_z\s+(.+?)(?:\s*$)',
                'reserve_vcc': r'\.amdhsa_reserve_vcc\s+(.+?)(?:\s*$)',
                'accum_offset': r'\.amdhsa_accum_offset\s+(.+?)(?:\s*$)',
            }
            
            for attr_name, pattern in amdhsa_patterns.items():
                match = re.search(pattern, kernel_text, re.MULTILINE)
                if match:
                    value_str = match.group(1).strip()
                    # 嘗試轉換為整數，如果失敗則保留為字符串（符號表達式）
                    try:
                        attrs[attr_name] = int(value_str)
                    except ValueError:
                        # 保留符號表達式（如 max(totalnumvgprs(...), 1, 0)）
                        attrs[attr_name] = value_str
                        print(f"  - {kernel_name}: {attr_name} 使用符號表達式: {value_str}")
            
            # 初始化 kernel metadata，包含組譯指令的值和 YAML 的值
            kernel_metadata_map[kernel_name] = {
                'attrs': attrs,  # 來自組譯指令（可能包含符號表達式）
                'args_list': [], 
                'yaml_vgpr_count': None,  # 來自 YAML（始終是數字）
                'yaml_agpr_count': None,  # 來自 YAML（始終是數字）
                'yaml_sgpr_count': None   # 來自 YAML（始終是數字）
            }
        
        # 從 YAML metadata 中提取 args 和其他信息
        yaml_match = re.search(r'\.amdgpu_metadata\s+---\s+(.*?)\.\.\.\s+\.end_amdgpu_metadata', 
                               original_isa_text, re.DOTALL)
        if yaml_match:
            try:
                yaml_text = yaml_match.group(1)
                metadata = yaml.safe_load(yaml_text)
                
                if 'amdhsa.kernels' in metadata:
                    for kernel in metadata['amdhsa.kernels']:
                        if '.name' in kernel:
                            kernel_name = kernel['.name']
                            
                            if kernel_name in kernel_metadata_map:
                                # 提取 YAML 中的 GPR counts（用於修復 YAML metadata）
                                # 注意：不覆蓋組譯指令中的值（可能是符號表達式）
                                # 而是單獨保存為 yaml_*_count
                                if '.vgpr_count' in kernel:
                                    kernel_metadata_map[kernel_name]['yaml_vgpr_count'] = kernel['.vgpr_count']
                                
                                if '.agpr_count' in kernel:
                                    kernel_metadata_map[kernel_name]['yaml_agpr_count'] = kernel['.agpr_count']
                                
                                # 保存 YAML 中的 .sgpr_count（可能 > 102）
                                if '.sgpr_count' in kernel:
                                    kernel_metadata_map[kernel_name]['yaml_sgpr_count'] = kernel['.sgpr_count']
                                
                                # 提取 args
                                if '.args' in kernel and isinstance(kernel['.args'], list):
                                    args_list = []
                                    for arg in kernel['.args']:
                                        arg_dict = {}
                                        for k, v in arg.items():
                                            key_name = k[1:] if k.startswith('.') else k
                                            arg_dict[key_name] = v
                                        args_list.append(arg_dict)
                                    kernel_metadata_map[kernel_name]['args_list'] = args_list
                
                print(f"[Info] Extracted metadata for {len(kernel_metadata_map)} kernel(s) from original ISA")
                
                # 輸出每個 kernel 的 metadata 信息
                for kname, kdata in kernel_metadata_map.items():
                    attrs = kdata['attrs']
                    args_list = kdata['args_list']
                    amdhsa_vgpr = attrs.get('vgpr_count', '?')
                    amdhsa_sgpr = attrs.get('amdhsa_next_free_sgpr', '?')
                    yaml_vgpr = kdata['yaml_vgpr_count']
                    yaml_sgpr = kdata['yaml_sgpr_count']
                    print(f"  - {kname}: amdhsa_vgpr={amdhsa_vgpr}, amdhsa_sgpr={amdhsa_sgpr}, "
                          f"yaml_vgpr={yaml_vgpr}, yaml_sgpr={yaml_sgpr}, "
                          f"kernarg_size={attrs.get('kernarg_segment_size', '?')}, "
                          f"lds={attrs.get('group_segment_fixed_size', '?')}, args={len(args_list)}")
            
            except Exception as e:
                print(f"[Warning] Failed to parse original ISA YAML metadata: {e}")
    
    # 如果沒有從原始 ISA 提取到 metadata，嘗試從 GPU MLIR 提取（向後兼容）
    if not kernel_metadata_map:
        print(f"[Info] Attempting to extract metadata from GPU MLIR: {gpumlir_file.name}")
        # 回退到舊的單 kernel 邏輯
        # 這裡保留原有邏輯以支持沒有 original_isa_file 的情況
        pass
    
    # Step 3: 修復 YAML metadata
    metadata_start = isa_text.find('.amdgpu_metadata')
    metadata_end = isa_text.find('.end_amdgpu_metadata')
    
    if metadata_start == -1 or metadata_end == -1:
        print("[Warning] No .amdgpu_metadata section found in ISA")
        return isa_text
    
    yaml_start = isa_text.find('---', metadata_start)
    yaml_end = isa_text.find('...', yaml_start)
    
    if yaml_start == -1 or yaml_end == -1:
        print("[Warning] Invalid YAML in .amdgpu_metadata")
        return isa_text
    
    yaml_text = isa_text[yaml_start+3:yaml_end].strip()
    
    try:
        metadata = yaml.safe_load(yaml_text)
    except Exception as e:
        print(f"[Warning] Failed to parse ISA metadata YAML: {e}")
        return isa_text
    
    # 為每個 kernel 修復 YAML metadata
    if 'amdhsa.kernels' in metadata:
        for i, kernel in enumerate(metadata['amdhsa.kernels']):
            if '.name' in kernel:
                kernel_name = kernel['.name']
                
                if kernel_name in kernel_metadata_map:
                    kdata = kernel_metadata_map[kernel_name]
                    attrs = kdata['attrs']
                    args_list = kdata['args_list']
                    yaml_vgpr_count = kdata['yaml_vgpr_count']
                    yaml_agpr_count = kdata['yaml_agpr_count']
                    yaml_sgpr_count = kdata['yaml_sgpr_count']
                    
                    # 修復 GPR counts（使用 YAML 中的數值）
                    if yaml_vgpr_count is not None:
                        kernel['.vgpr_count'] = yaml_vgpr_count
                    if yaml_agpr_count is not None:
                        kernel['.agpr_count'] = yaml_agpr_count
                    if yaml_sgpr_count is not None:
                        kernel['.sgpr_count'] = yaml_sgpr_count
                    
                    # 修復 kernarg_segment_size
                    if 'kernarg_segment_size' in attrs:
                        kernel['.kernarg_segment_size'] = attrs['kernarg_segment_size']
                    
                    # 修復 group_segment_fixed_size (LDS / shared memory 大小)
                    if 'group_segment_fixed_size' in attrs:
                        kernel['.group_segment_fixed_size'] = attrs['group_segment_fixed_size']
                    
                    # 修復 args
                    if args_list:
                        kernel['.args'] = []
                        for arg in args_list:
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
    
    # 重新生成 YAML
    fixed_yaml = yaml.dump(metadata, default_flow_style=False, sort_keys=False)
    
    # 替換 ISA 中的 metadata
    before_metadata = isa_text[:yaml_start]
    after_metadata = isa_text[yaml_end:]
    
    fixed_isa = before_metadata + "---\n" + fixed_yaml + "...\n" + after_metadata
    
    # Step 4: 修復 .amdhsa_* 指令（為每個 kernel 分別修復）
    for kernel_info in rebuilt_kernels:
        kernel_name = kernel_info['name']
        
        if kernel_name not in kernel_metadata_map:
            print(f"[Warning] No metadata found for kernel {kernel_name}")
            continue
        
        attrs = kernel_metadata_map[kernel_name]['attrs']
        
        # 找到這個 kernel 的 .amdhsa_kernel 到 .end_amdhsa_kernel 的範圍
        kernel_pattern = rf'\.amdhsa_kernel\s+{re.escape(kernel_name)}(.*?)\.end_amdhsa_kernel'
        match = re.search(kernel_pattern, fixed_isa, re.DOTALL)
        
        if not match:
            print(f"[Warning] Could not find kernel block for {kernel_name}")
            continue
        
        kernel_block_start = match.start()
        kernel_block_end = match.end()
        kernel_block = match.group(0)
        
        # 在這個 kernel block 內進行替換
        modified_block = kernel_block
        
        # 定義需要修復的 amdhsa 指令
        amdhsa_replacements = {
            'kernarg_segment_size': 'amdhsa_kernarg_size',
            'group_segment_fixed_size': 'amdhsa_group_segment_fixed_size',
            'vgpr_count': 'amdhsa_next_free_vgpr',
            'amdhsa_next_free_sgpr': 'amdhsa_next_free_sgpr',  # 使用組譯指令的值（<= 102）
            'user_sgpr_count': 'amdhsa_user_sgpr_count',
            'dispatch_ptr': 'amdhsa_user_sgpr_dispatch_ptr',
            'queue_ptr': 'amdhsa_user_sgpr_queue_ptr',
            'workitem_id': 'amdhsa_system_vgpr_workitem_id',
            'workgroup_id_x': 'amdhsa_system_sgpr_workgroup_id_x',
            'workgroup_id_y': 'amdhsa_system_sgpr_workgroup_id_y',
            'workgroup_id_z': 'amdhsa_system_sgpr_workgroup_id_z',
            'reserve_vcc': 'amdhsa_reserve_vcc',
            'accum_offset': 'amdhsa_accum_offset',
        }
        
        for attr_key, amdhsa_directive in amdhsa_replacements.items():
            if attr_key in attrs:
                value = attrs[attr_key]
                # 跳過某些條件（只對整數值判斷）
                if attr_key in ['vgpr_count', 'amdhsa_next_free_sgpr'] and value == 0:
                    continue
                
                # 修改模式以匹配數字或符號表達式（匹配到行尾）
                # 使用 re.MULTILINE 使 $ 匹配行尾
                pattern = rf'(\.{amdhsa_directive})\s+.+?(?=\s*$)'
                replacement = rf'\1 {value}'
                modified_block = re.sub(pattern, replacement, modified_block, flags=re.MULTILINE)
        
        # 特殊處理：根據 kernarg_segment_size 決定是否需要 kernarg pointer
        if 'kernarg_segment_size' in attrs and attrs['kernarg_segment_size'] > 0:
            modified_block = re.sub(
                r'(\.amdhsa_user_sgpr_kernarg_segment_ptr)\s+\d+',
                r'\1 1',
                modified_block
            )
        
        # 替換原始 ISA 中的這個 kernel block
        fixed_isa = fixed_isa[:kernel_block_start] + modified_block + fixed_isa[kernel_block_end:]
    
    print(f"[Info] Fixed ISA metadata for {len(rebuilt_kernels)} kernel(s)")
    
    return fixed_isa


def build_isa_and_hsaco(kernel_mlir: pathlib.Path, chip: str, workdir: pathlib.Path, output_prefix: str = None, original_isa_file: pathlib.Path = None):
    """
    從 GPU MLIR 繼續執行 pipeline，生成 ISA 和 HSACO
    
    Args:
        kernel_mlir: GPU MLIR 文件 (帶有 gpu.func)
        chip: 目標晶片型號
        workdir: 工作目錄
        output_prefix: 輸出文件前綴（如果為 None，使用輸入文件名）
        original_isa_file: 原始 ISA 文件路徑（如果提供，用於提取 metadata）
    """
    for tool in ["mlir-opt", "llvm-mc", "lld"]:
        ensure_tool(tool)

    kernel_stem = output_prefix if output_prefix else kernel_mlir.stem

    kernel_binary_mlir = workdir / f"{kernel_stem}_binary_isa.mlir"
    kernel_isa_s         = workdir / f"{kernel_stem}.s"
    kernel_o             = workdir / f"{kernel_stem}.o"
    kernel_hsaco         = workdir / f"{kernel_stem}.hsaco"

    print(f"\n=== Stage 3: Running MLIR optimization pipeline ===")
    pipeline = (
        f"builtin.module("
        f"gpu-kernel-outlining,"
        f"rocdl-attach-target{{chip={chip}}},"
        f"gpu.module(convert-gpu-to-rocdl{{index-bitwidth=32 runtime=HIP}}),"
        f"gpu-to-llvm,"
        f"gpu-module-to-binary{{format=isa}}"
        f")"
    )

    mlir_opt_cmd = [
        "mlir-opt",
        str(kernel_mlir),
        f"--pass-pipeline={pipeline}",
        "-o",
        str(kernel_binary_mlir),
    ]
    run_cmd(mlir_opt_cmd)

    binary_text = kernel_binary_mlir.read_text()
    isa_list = extract_assembly_strings(binary_text)

    if not isa_list:
        raise RuntimeError("No gpu.binary assembly attribute (ISA) found in MLIR output")

    if len(isa_list) > 1:
        print(f"[!] Found {len(isa_list)} gpu.binary entries (ISA), using the first one.")

    isa = isa_list[0]
    
    # Fix ISA metadata: extract from original ISA (if provided) or GPU MLIR attributes
    isa = fix_isa_metadata(isa, kernel_mlir, original_isa_file)
    
    # Append global symbols from original ISA (if provided)
    if original_isa_file is not None:
        isa = append_global_symbols(isa, original_isa_file)
        isa = append_device_functions(isa, original_isa_file)
    
    kernel_isa_s.write_text(isa)
    print(f"Wrote ISA assembly to {kernel_isa_s}")

    print(f"\n=== Stage 4: Assembling ISA to object file ===")
    llvm_mc_cmd = [
        "llvm-mc",
        "-triple", "amdgcn-amd-amdhsa",
        f"-mcpu={chip}",
        "-filetype=obj",
        str(kernel_isa_s),
        "-o",
        str(kernel_o),
    ]
    run_cmd(llvm_mc_cmd)

    print(f"\n=== Stage 5: Linking to HSACO ===")
    ld_cmd = [
        "lld",
        "-flavor", "gnu",
        "-m", "elf64_amdgpu",
        "--no-undefined",
        "-shared",
        "-plugin-opt=-amdgpu-internalize-symbols",
        f"-plugin-opt=mcpu={chip}",
        "--whole-archive",
        str(kernel_o),
        "--no-whole-archive",
        "-o",
        str(kernel_hsaco),
    ]
    run_cmd(ld_cmd)

    print(f"\n✓ Successfully generated HSACO: {kernel_hsaco}")


def build_llvm_ir_via_binary(kernel_mlir: pathlib.Path, chip: str, workdir: pathlib.Path, output_prefix: str = None):
    """
    從 GPU MLIR 生成 LLVM IR（用於調試和分析）
    
    使用 format=llvm 而不是 format=isa，生成 LLVM bitcode 並轉換為人類可讀的 .ll 文件
    
    Args:
        kernel_mlir: GPU MLIR 文件 (帶有 gpu.func)
        chip: 目標晶片型號
        workdir: 工作目錄
        output_prefix: 輸出文件前綴（如果為 None，使用輸入文件名）
    """
    for tool in ["mlir-opt", "llvm-dis"]:
        ensure_tool(tool)
    
    kernel_stem = output_prefix if output_prefix else kernel_mlir.stem
    
    kernel_binary_mlir = workdir / f"{kernel_stem}_binary_llvm.mlir"
    kernel_llvm_bc = workdir / f"{kernel_stem}_llvm.bc"
    kernel_llvm_ll = workdir / f"{kernel_stem}_llvm.ll"
    
    print(f"\n=== Generating LLVM IR (for debugging) ===")
    pipeline = (
        f"builtin.module("
        f"gpu-kernel-outlining,"
        f"rocdl-attach-target{{chip={chip}}},"
        f"gpu.module(convert-gpu-to-rocdl{{index-bitwidth=32 runtime=HIP}}),"
        f"gpu-to-llvm,"
        f"gpu-module-to-binary{{format=llvm}}"
        f")"
    )
    
    mlir_opt_cmd = [
        "mlir-opt",
        str(kernel_mlir),
        f"--pass-pipeline={pipeline}",
        "-o",
        str(kernel_binary_mlir),
    ]
    run_cmd(mlir_opt_cmd)
    
    binary_text = kernel_binary_mlir.read_text()
    bc_list = extract_bitcode_bytes(binary_text)
    
    if not bc_list:
        raise RuntimeError("No offload/offloading bitcode attribute found in MLIR output")
    
    if len(bc_list) > 1:
        print(f"[!] Found {len(bc_list)} offload entries (bitcode), using the first one.")
    
    bitcode = bc_list[0]
    kernel_llvm_bc.write_bytes(bitcode)
    print(f"Wrote LLVM bitcode to {kernel_llvm_bc}")
    
    llvm_dis_cmd = [
        "llvm-dis",
        str(kernel_llvm_bc),
        "-o",
        str(kernel_llvm_ll),
    ]
    
    # llvm-dis 可能因版本不兼容失败（如 LLVM 22 生成的 bitcode vs LLVM 18 的 llvm-dis）
    # 這不影響 HSACO 生成，所以我們只是警告而不失敗
    try:
        run_cmd(llvm_dis_cmd)
        print(f"✓ Wrote human-readable LLVM IR to {kernel_llvm_ll}")
    except subprocess.CalledProcessError as e:
        print(f"[Warning] llvm-dis 失敗（可能是版本不兼容）")
        print(f"[Reason] LLVM bitcode 由 LLVM 22 產生，但系統的 llvm-dis 是較舊版本")
        print(f"[Info] LLVM bitcode 已保存在: {kernel_llvm_bc}")
        print(f"[Info] 您可以使用匹配版本的 llvm-dis 來反組譯此 bitcode")
        print(f"[Info] 這不影響 HSACO 生成，可以安全忽略")

# ------------------------------------------------------------
# Main Logic
# ------------------------------------------------------------
def main():

    ap = argparse.ArgumentParser(
        description="AMD ISA to MLIR pipeline: Translate .s assembly to MLIR, then build HSACO / LLVM IR."
    )
    ap.add_argument(
        "--chip",
        default="gfx950",
        help="AMDGPU chip (for rocdl-attach-target & llvm-mc -mcpu) [default: gfx950]",
    )
    ap.add_argument(
        "--workdir",
        default="pipeline_output",
        help="directory to put intermediate files [default: \"pipeline_output\"]",
    )
    ap.add_argument(
        "--emit-isa",
        action="store_true",
        default=True,
        help="run device-only pipeline with gpu-module-to-binary{format=isa} and build HSACO (default: enabled)",
    )
    ap.add_argument(
        "--no-emit-isa",
        dest="emit_isa",
        action="store_false",
        help="disable ISA / HSACO generation",
    )
    ap.add_argument(
        "--emit-llvm-ir",
        action="store_true",
        help="also run device-only pipeline with gpu-module-to-binary{format=llvm} and dump LLVM IR (via llvm-dis)",
    )
    ap.add_argument(
        "--output-prefix",
        default=None,
        help="output file prefix [default: <input_stem>_rebuilt for .s, <input_stem> for .mlir]",
    )
    ap.add_argument(
        "--original-isa",
        default=None,
        help="original complete ISA file (for extracting metadata and global symbols, optional)",
    )
    ap.add_argument(
        "input_file",
        help="Input file: either .s (AMD ISA assembly) or .mlir (GPU MLIR with gpu.func kernel)"
    )

    args = ap.parse_args()

    input_file = pathlib.Path(args.input_file).resolve()

    if not input_file.exists():
        raise FileNotFoundError(input_file)

    # 如果未指定 workdir，則使用 input_file 的檔名
    if args.workdir is None:
        workdir = pathlib.Path(input_file.stem)
    else:
        workdir = pathlib.Path(args.workdir)
    
    workdir.mkdir(parents=True, exist_ok=True)

    suffix = input_file.suffix.lower()
    
    # 決定輸出文件前綴
    if args.output_prefix is None:
        if suffix == ".s":
            # 對於 .s 文件，自動添加 _rebuilt 後綴以區分原始和重建版本
            output_prefix = f"{input_file.stem}_rebuilt"
        else:
            # 對於 .mlir 文件，使用原始文件名
            output_prefix = input_file.stem
    else:
        output_prefix = args.output_prefix
    
    print(f"[INFO] 輸出文件前綴: {output_prefix}")
    
    # 用於記錄最終生成的 GPU MLIR 文件（用於後續的 run-host）
    final_gpu_mlir = None
    
    # 處理 --original-isa 參數
    original_isa_for_metadata = None
    if args.original_isa:
        original_isa_for_metadata = pathlib.Path(args.original_isa).resolve()
        if not original_isa_for_metadata.exists():
            print(f"[Warning] --original-isa file not found: {original_isa_for_metadata}")
            original_isa_for_metadata = None
        else:
            print(f"[Info] Using original ISA for metadata extraction: {original_isa_for_metadata.name}")
    
    if suffix == ".s":
        # 完整流程：.s -> .amdisamlir -> .gpumlir -> ISA/HSACO
        print(f"=== Processing AMD ISA Assembly: {input_file.name} ===")
        amdisamlir_file, gpumlir_file = translate_asm_to_gpu(input_file, workdir, output_prefix)
        final_gpu_mlir = gpumlir_file
        
        if args.emit_isa:
            # 如果提供了 --original-isa，使用它；否則使用 input_file
            isa_for_metadata = original_isa_for_metadata if original_isa_for_metadata else input_file
            build_isa_and_hsaco(gpumlir_file, args.chip, workdir, output_prefix, original_isa_file=isa_for_metadata)
        
        if args.emit_llvm_ir:
            build_llvm_ir_via_binary(gpumlir_file, args.chip, workdir, output_prefix)
    
    elif suffix in [".mlir", ".gpumlir"]:
        # 從 GPU MLIR 開始（原本的流程）
        print(f"=== Processing GPU MLIR: {input_file.name} ===")
        final_gpu_mlir = input_file
        
        if args.emit_isa:
            build_isa_and_hsaco(input_file, args.chip, workdir, output_prefix)
        
        if args.emit_llvm_ir:
            build_llvm_ir_via_binary(input_file, args.chip, workdir, output_prefix)
    
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Expected .s or .mlir")

if __name__ == "__main__":
    main()