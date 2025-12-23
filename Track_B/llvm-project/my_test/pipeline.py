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
# Register Clobber Functions
# ======================================

def analyze_registers_in_gpumlir(mlir_content: str) -> tuple:
    """
    分析 GPU MLIR 中所有 inline_asm 使用的暫存器
    返回: (max_vgpr, max_sgpr, max_agpr)
    """
    vgprs = set()
    sgprs = set()
    agprs = set()
    
    # 匹配所有 llvm.inline_asm 中的暫存器使用
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
        
        # 匹配 a123 格式 (AGPR - Accumulation GPR)
        for a_match in re.finditer(r'\ba(\d+)\b', asm_code):
            agprs.add(int(a_match.group(1)))
        
        # 匹配 a[123:456] 格式
        for a_range_match in re.finditer(r'\ba\[(\d+):(\d+)\]', asm_code):
            start = int(a_range_match.group(1))
            end = int(a_range_match.group(2))
            for i in range(start, end + 1):
                agprs.add(i)
        
        # 匹配 a[0xNN:0xNN] 格式 (十六進制)
        for a_hex_match in re.finditer(r'\ba\[0x([0-9a-fA-F]+):0x([0-9a-fA-F]+)\]', asm_code):
            start = int(a_hex_match.group(1), 16)
            end = int(a_hex_match.group(2), 16)
            for i in range(start, end + 1):
                agprs.add(i)
    
    max_vgpr = max(vgprs) if vgprs else -1
    max_sgpr = max(sgprs) if sgprs else -1
    max_agpr = max(agprs) if agprs else -1
    
    return max_vgpr, max_sgpr, max_agpr


def add_register_clobber_to_gpumlir(mlir_content: str) -> str:
    """
    自動分析並添加 register clobber (reserve & release) 到 GPU MLIR
    這是新的默認行為，確保 LLVM 能正確計算資源需求
    支持 VGPR, SGPR, 和 AGPR (Accumulation GPR)
    """
    # 分析暫存器使用
    max_vgpr, max_sgpr, max_agpr = analyze_registers_in_gpumlir(mlir_content)
    
    if max_vgpr < 0 and max_sgpr < 0 and max_agpr < 0:
        print("[Info] No inline_asm register usage detected, skipping clobber")
        return mlir_content
    
    vgpr_count = max_vgpr + 1 if max_vgpr >= 0 else 0
    sgpr_count = max_sgpr + 1 if max_sgpr >= 0 else 0
    agpr_count = max_agpr + 1 if max_agpr >= 0 else 0
    
    info_parts = []
    if max_vgpr >= 0:
        info_parts.append(f"VGPR=0-{max_vgpr} ({vgpr_count})")
    if max_sgpr >= 0:
        info_parts.append(f"SGPR=0-{max_sgpr} ({sgpr_count})")
    if max_agpr >= 0:
        info_parts.append(f"AGPR=0-{max_agpr} ({agpr_count})")
    print(f"[Info] Detected register usage: {', '.join(info_parts)}")
    
    # 生成 reserve 和 release 代碼
    reserve_lines = []
    release_lines = []
    
    if max_vgpr >= 0:
        reserve_lines.append(f'    // Auto: Reserve VGPR v[0:{max_vgpr}] ({vgpr_count} registers)')
        reserve_lines.append(f'    %vgpr_reserved = llvm.inline_asm has_side_effects asm_dialect = att "", "={{v[0:{max_vgpr}]}}" : () -> vector<{vgpr_count}xi32>')
        release_lines.append(f'    // Release VGPR clobber')
        release_lines.append(f'    llvm.inline_asm has_side_effects asm_dialect = att "", "{{v[0:{max_vgpr}]}}" %vgpr_reserved : (vector<{vgpr_count}xi32>) -> ()')
    
    if max_sgpr >= 0:
        reserve_lines.append(f'    // Auto: Reserve SGPR s[0:{max_sgpr}] ({sgpr_count} registers)')
        reserve_lines.append(f'    %sgpr_reserved = llvm.inline_asm has_side_effects asm_dialect = att "", "={{s[0:{max_sgpr}]}}" : () -> vector<{sgpr_count}xi32>')
        release_lines.append(f'    // Release SGPR clobber')
        release_lines.append(f'    llvm.inline_asm has_side_effects asm_dialect = att "", "{{s[0:{max_sgpr}]}}" %sgpr_reserved : (vector<{sgpr_count}xi32>) -> ()')
    
    # AGPR Clobber: LLVM AMDGPU backend 不支持 'a' 約束字符的 inline asm clobber
    # 目前需要依賴 LLVM backend 自動檢測 AGPR 使用或手動修復 metadata
    if max_agpr >= 0:
        reserve_lines.append(f'    // ⚠️  Detected AGPR usage: a[0:{max_agpr}] ({agpr_count} registers)')
        reserve_lines.append(f'    // Note: LLVM does not support AGPR clobber constraints')
        reserve_lines.append(f'    // AGPR count may need manual adjustment in metadata')
    
    # 插入 reserve 和 release
    lines = mlir_content.splitlines()
    new_lines = []
    clobber_inserted = False
    release_inserted = False
    
    for line in lines:
        # 找到 gpu.return，在之前插入 release
        if 'gpu.return' in line and not release_inserted:
            new_lines.append('')
            new_lines.append('    // ===== Register Clobber Release =====')
            new_lines.extend(release_lines)
            new_lines.append('    // ====================================')
            new_lines.append('')
            release_inserted = True
        
        new_lines.append(line)
        
        # 找到 gpu.func ... kernel {
        if re.search(r'gpu\.func\s+@\S+.*\bkernel\b.*\{', line) and not clobber_inserted:
            new_lines.append('')
            new_lines.append('    // ===== Register Clobber Reserve =====')
            new_lines.extend(reserve_lines)
            new_lines.append('    // ====================================')
            new_lines.append('')
            clobber_inserted = True
    
    if not clobber_inserted:
        print("[Warning] No gpu.func kernel found, clobber not inserted")
        return mlir_content
    
    if not release_inserted:
        print("[Warning] No gpu.return found, clobber not released")
    
    print("[Info] ✅ Register clobber (reserve + release) added")
    return '\n'.join(new_lines)

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


def translate_asm_to_gpu(asm_file: pathlib.Path, workdir: pathlib.Path, output_prefix: str = None, auto_add_clobber: bool = False):
    """
    使用 amdisa-translate 將 .s 文件轉換為 .amdisamlir 和 .gpumlir
    
    Args:
        asm_file: 輸入的 .s assembly 文件
        workdir: 工作目錄
        output_prefix: 輸出文件前綴（如果為 None，使用輸入文件名）
        auto_add_clobber: 是否自動添加 register clobber
    
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
    
    # 自動添加 register clobber（新的默認行為）
    if auto_add_clobber:
        print(f"\n=== Stage 2.5: Adding Register Clobber ===")
        gpumlir_text = add_register_clobber_to_gpumlir(gpumlir_text)
    
    gpumlir_file.write_text(gpumlir_text)
    print(f"Generated GPU MLIR: {gpumlir_file}")
    
    return amdisamlir_file, gpumlir_file


def fix_isa_metadata(isa_text: str, gpumlir_file: pathlib.Path) -> str:
    """
    修復 ISA metadata：從 GPU MLIR attributes 提取 kernarg_segment_size 等信息
    
    注意：資源計數（VGPR/SGPR）現在由 LLVM 通過 register clobber 自動計算
    這個函數只修復 kernarg_segment_size 和 hidden parameters 等非資源相關的 metadata
    
    Args:
        isa_text: ISA assembly 文本
        gpumlir_file: GPU MLIR 文件路徑
    """
    
    # 讀取 GPU MLIR 獲取 attributes
    gpumlir_text = gpumlir_file.read_text()
    
    # 提取 module attributes
    attrs = {}
    
    # 提取 vgpr_count, sgpr_count, agpr_count, kernarg_segment_size, group_segment_fixed_size
    for attr_name in ['vgpr_count', 'sgpr_count', 'agpr_count', 'kernarg_segment_size', 'group_segment_fixed_size']:
        pattern = rf'amdisa\.{attr_name}\s*=\s*(\d+)'
        match = re.search(pattern, gpumlir_text)
        if match:
            attrs[attr_name] = int(match.group(1))
    
    # 提取 kernargs array
    kernargs_pattern = r'amdisa\.kernargs\s*=\s*\[(.*?)\](?=,\s+amdisa\.|,\s+llvm\.|}\s+{)'
    match = re.search(kernargs_pattern, gpumlir_text, re.DOTALL)
    
    args_list = []
    if match:
        kernargs_str = match.group(1)
        # 簡單解析：找到所有 {..} 字典
        dict_pattern = r'\{([^{}]*?)\}'
        for dict_match in re.finditer(dict_pattern, kernargs_str):
            arg_dict = {}
            dict_content = dict_match.group(1)
            
            # 解析每個屬性
            for prop_match in re.finditer(r'(\w+)\s*=\s*"([^"]+)"', dict_content):
                arg_dict[prop_match.group(1)] = prop_match.group(2)
            for prop_match in re.finditer(r'(\w+)\s*=\s*(\d+)', dict_content):
                arg_dict[prop_match.group(1)] = int(prop_match.group(2))
            
            if arg_dict:
                args_list.append(arg_dict)
    
    if not args_list and not attrs:
        print("[Warning] No amdisa attributes found in GPU MLIR, ISA metadata may be incorrect")
        return isa_text
    
    # 找到 ISA 中的 .amdgpu_metadata 部分
    metadata_start = isa_text.find('.amdgpu_metadata')
    metadata_end = isa_text.find('.end_amdgpu_metadata')
    
    if metadata_start == -1 or metadata_end == -1:
        print("[Warning] No .amdgpu_metadata section found in ISA")
        return isa_text
    
    # 提取 YAML metadata
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
    
    # 修復 metadata
    if 'amdhsa.kernels' in metadata and len(metadata['amdhsa.kernels']) > 0:
        kernel = metadata['amdhsa.kernels'][0]
        
        # 注意：不再修復 VGPR/SGPR counts，由 LLVM 通過 register clobber 自動計算
        # 但是 AGPR 必須手動修復，因為 LLVM 不支持 AGPR clobber約束
        print("[Info] Trusting LLVM for resource counts (VGPR/SGPR)")
        
        # 修復 AGPR count (LLVM 無法通過 clobber 計算)
        if 'agpr_count' in attrs:
            kernel['.agpr_count'] = attrs['agpr_count']
            print(f"[Info] Fixed AGPR count: {attrs['agpr_count']} (LLVM limitation workaround)")
        
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
    
    # 同時修復 .amdhsa_* 指令（這些指令會被 assembler 使用）
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
    
    # 注意：不再修復 .amdhsa_next_free_vgpr/sgpr，由 LLVM 自動計算
    
    print(f"[Info] Fixed ISA metadata (non-resource fields):")
    if 'kernarg_segment_size' in attrs:
        print(f"  - kernarg_segment_size: {attrs['kernarg_segment_size']}")
    if 'group_segment_fixed_size' in attrs:
        print(f"  - group_segment_fixed_size: {attrs['group_segment_fixed_size']} (LDS/Shared Memory)")
    if args_list:
        print(f"  - args count: {len(args_list)}")
    
    return fixed_isa


def build_isa_and_hsaco(kernel_mlir: pathlib.Path, chip: str, workdir: pathlib.Path, output_prefix: str = None):
    """
    從 GPU MLIR 繼續執行 pipeline，生成 ISA 和 HSACO
    
    注意：資源計數現在由 LLVM 通過 register clobber 自動計算
    
    Args:
        kernel_mlir: GPU MLIR 文件 (帶有 gpu.func，應包含 register clobber)
        chip: 目標晶片型號
        workdir: 工作目錄
        output_prefix: 輸出文件前綴（如果為 None，使用輸入文件名）
    """
    for tool in ["mlir-opt", "llvm-mc", "ld.lld"]:
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
    
    # Fix ISA metadata: extract kernarg_segment_size etc. from GPU MLIR attributes
    # Note: Resource counts are now handled by LLVM via register clobber
    isa = fix_isa_metadata(isa, kernel_mlir)
    
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
        "ld.lld",
        "-shared",
        str(kernel_o),
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
    run_cmd(llvm_dis_cmd)
    
    print(f"✓ Wrote human-readable LLVM IR to {kernel_llvm_ll}")

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
    
    if suffix == ".s":
        # 完整流程：.s -> .amdisamlir -> .gpumlir (with clobber) -> ISA/HSACO
        print(f"=== Processing AMD ISA Assembly: {input_file.name} ===")
        amdisamlir_file, gpumlir_file = translate_asm_to_gpu(input_file, workdir, output_prefix, auto_add_clobber=True)
        final_gpu_mlir = gpumlir_file
        
        if args.emit_isa:
            build_isa_and_hsaco(gpumlir_file, args.chip, workdir, output_prefix)
        
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
