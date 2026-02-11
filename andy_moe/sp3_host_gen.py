#!/usr/bin/env python3
"""
sp3_host_gen.py -- Generate a standalone HIP C++ host program from an SP3 shader file.

Parses the SP3 header (parameter table, grid formulas, constants) and produces
a .cpp file that:
  1. Loads an HSACO via hipModuleLoad
  2. Allocates GPU buffers for each pointer parameter
  3. Assembles the kernarg buffer at exact SP3 offsets
  4. Launches the kernel with correct grid/block dimensions
  5. gpu.printf output appears automatically on stderr

Usage:
  python3 sp3_host_gen.py INPUT.sp3 -o test_kernel.cpp [--compile]
"""

import argparse
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SP3Param:
    """A single parameter from the SP3 parameter table."""
    name: str
    size: int          # always 4 in current SP3 format
    offset: int        # hex offset in kernarg buffer
    comment: str
    sreg: str
    is_pointer: bool = False

    def __post_init__(self):
        # Classify: pointer if comment mentions "address" or sreg ends with _buf
        c = self.comment.lower()
        s = self.sreg.lower()
        self.is_pointer = ('address' in c) or s.endswith('_buf')


@dataclass
class GridFormulas:
    """Grid/block dimension formulas extracted from SP3 header."""
    global_size_x: str = '256'
    global_size_y: str = '1'
    global_size_z: str = '1'
    local_size_x: str = '256'
    local_size_y: str = '1'
    local_size_z: str = '1'


@dataclass
class SP3Info:
    """All information extracted from an SP3 file header."""
    filename: str = ''
    params: List[SP3Param] = field(default_factory=list)
    grid: GridFormulas = field(default_factory=GridFormulas)
    constants: Dict[str, str] = field(default_factory=dict)
    # derived
    kernarg_size: int = 0   # total kernarg buffer size (bytes)
    is_fmoe: bool = False   # whether this looks like an FMOE kernel


# ---------------------------------------------------------------------------
# SP3 Header Parser
# ---------------------------------------------------------------------------

def parse_sp3_header(sp3_path: str) -> SP3Info:
    """Parse an SP3 file header and extract parameter table, grid formulas, constants."""
    with open(sp3_path, 'r') as f:
        lines = f.readlines()

    info = SP3Info(filename=os.path.basename(sp3_path))

    # --- 1. Find parameter table boundaries ---
    param_start = None
    param_end = None
    for i, line in enumerate(lines):
        if 'parameter description beginning' in line.lower() or \
           'parameter description beginning' in line:
            param_start = i
        if re.search(r'par[am]*eter description end', line, re.IGNORECASE):
            param_end = i
            break

    if param_start is None:
        print(f'[Warning] No parameter table found in {sp3_path}', file=sys.stderr)
        return info

    # --- 2. Parse parameter lines ---
    # Format: //   NAME   SIZE   0xOFFSET   COMMENT   SREG
    param_re = re.compile(
        r'//\s+'
        r'(\S+)\s+'          # name
        r'(\d+)\s+'          # size
        r'(0x[0-9A-Fa-f]+)\s+'  # offset
        r'(.+?)\s{2,}'      # comment (followed by 2+ spaces)
        r'(\S+)\s*$'         # sreg
    )

    end_boundary = param_end if param_end else len(lines)
    for i in range(param_start, end_boundary):
        line = lines[i]
        # Skip header/separator lines
        if '----' in line or 'name' in line.lower() and 'size' in line.lower() and 'offset' in line.lower():
            continue
        m = param_re.match(line)
        if m:
            name = m.group(1)
            size = int(m.group(2))
            offset = int(m.group(3), 16)
            comment = m.group(4).strip()
            sreg = m.group(5).strip()
            info.params.append(SP3Param(name, size, offset, comment, sreg))

    # Compute total kernarg size: last param offset + 16 (each slot is 0x10)
    if info.params:
        max_offset = max(p.offset for p in info.params)
        info.kernarg_size = max_offset + 0x10
    else:
        info.kernarg_size = 256  # fallback

    # --- 2b. Cross-check with actual s_load instructions in kernel body ---
    sload_params = _parse_sload_instructions(lines)
    if sload_params:
        info.params = _reconcile_params(info.params, sload_params)
        if info.params:
            max_offset = max(p.offset for p in info.params)
            info.kernarg_size = max_offset + 0x10

    # --- 3. Parse grid dimension formulas ---
    grid_re = re.compile(r'//\s*(\d+)\.\s*(global_size_[xyz]|local_size_[xyz])\s*=\s*(.+)')
    for i in range(param_start, end_boundary):
        line = lines[i].strip()
        m = grid_re.match(line)
        if m:
            var_name = m.group(2)
            expr = m.group(3).strip()
            # Remove trailing comments
            expr = re.sub(r'//.*$', '', expr).strip()
            if var_name == 'global_size_x':
                info.grid.global_size_x = expr
            elif var_name == 'global_size_y':
                info.grid.global_size_y = expr
            elif var_name == 'global_size_z':
                info.grid.global_size_z = expr
            elif var_name == 'local_size_x':
                info.grid.local_size_x = expr
            elif var_name == 'local_size_y':
                info.grid.local_size_y = expr
            elif var_name == 'local_size_z':
                info.grid.local_size_z = expr

    # --- 4. Parse constants (var NAME = EXPR) ---
    # Scan from after param_end to first non-comment, non-var, non-blank line
    # that looks like actual code (e.g., "shader main")
    const_re = re.compile(r'^var\s+(\w+)\s*=\s*(.+?)(?:\s*//.*)?$')
    start_scan = (param_end + 1) if param_end else 0
    for i in range(start_scan, min(start_scan + 100, len(lines))):
        line = lines[i].strip()
        m = const_re.match(line)
        if m:
            name = m.group(1)
            expr = m.group(2).strip()
            info.constants[name] = expr

    # Evaluate constants in dependency order
    evaluated = {}
    for name, expr in info.constants.items():
        evaluated[name] = _eval_const_expr(expr, evaluated)
    info.constants = {k: str(v) for k, v in evaluated.items()}

    # Detect FMOE family
    fname = info.filename.upper()
    info.is_fmoe = 'FMOE' in fname or 'MOE' in fname

    return info


def _eval_const_expr(expr: str, known: Dict[str, int]) -> int:
    """Evaluate a simple arithmetic expression with known constants."""
    # Replace known variable references
    result_expr = expr
    for name, val in known.items():
        result_expr = re.sub(r'\b' + re.escape(name) + r'\b', str(val), result_expr)
    try:
        # Only allow safe arithmetic
        val = eval(result_expr, {"__builtins__": {}}, {})
        return int(val)
    except Exception:
        # If it can't be evaluated, try to parse as int
        try:
            return int(expr, 0)
        except ValueError:
            return 0


# ---------------------------------------------------------------------------
# Kernel body s_load scanner  (ground-truth kernarg offsets)
# ---------------------------------------------------------------------------

# Regex: s_load_dwordx2  s_regs(_s_R_buf, 0),  s[0:1], 0x00  // comment
_SLOAD_PTR_RE = re.compile(
    r's_load_dwordx2\s+'
    r's_regs\((_s_\w+)\s*,\s*\d+\)'   # target var: _s_R_buf etc.
    r'\s*,\s*s\[0:1\]\s*,\s*'
    r'(0x[0-9A-Fa-f]+)'               # offset
    r'(?:\s*//\s*(.*))?'               # optional inline comment
)

# Regex: s_load_dword  s_Xs,  s[0:1], 0x100  // comment
_SLOAD_SCALAR_RE = re.compile(
    r's_load_dword\s+'
    r'(s_\w+)'                         # target var: s_Xs, s_dim_len, etc.
    r'\s*,\s*s\[0:1\]\s*,\s*'
    r'(0x[0-9A-Fa-f]+)'               # offset
    r'(?:\s*//\s*(.*))?'               # optional inline comment
)


def _parse_sload_instructions(lines: List[str]) -> List[SP3Param]:
    """Scan all SP3 lines for s_load_dword* from s[0:1] and build ground-truth param list."""
    results: List[SP3Param] = []
    seen_offsets = set()

    for line in lines:
        stripped = line.strip()

        # --- pointer loads (s_load_dwordx2) ---
        m = _SLOAD_PTR_RE.search(stripped)
        if m:
            var_name = m.group(1)           # e.g. _s_R_buf
            offset = int(m.group(2), 16)
            comment = (m.group(3) or '').strip()
            if offset in seen_offsets:
                continue
            seen_offsets.add(offset)

            # Derive short name: _s_R_buf -> R, _s_XQ_buf -> XQ
            short = var_name
            if short.startswith('_s_'):
                short = short[3:]
            elif short.startswith('s_'):
                short = short[2:]
            if short.endswith('_buf'):
                short = short[:-4]

            # sreg = variable without leading underscore  (s_R_buf)
            sreg = var_name.lstrip('_')

            results.append(SP3Param(
                name=short, size=8, offset=offset,
                comment=comment if comment else f'{short} address',
                sreg=sreg, is_pointer=True))
            continue

        # --- scalar loads (s_load_dword) ---
        m = _SLOAD_SCALAR_RE.search(stripped)
        if m:
            var_name = m.group(1)           # e.g. s_Xs, s_dim_len
            offset = int(m.group(2), 16)
            comment = (m.group(3) or '').strip()
            if offset in seen_offsets:
                continue
            seen_offsets.add(offset)

            # Derive short name: s_Xs -> Xs, s_dim_len -> dim_len
            short = var_name
            if short.startswith('s_'):
                short = short[2:]

            sreg = var_name

            results.append(SP3Param(
                name=short, size=4, offset=offset,
                comment=comment if comment else short,
                sreg=sreg, is_pointer=False))
            continue

    return results


def _reconcile_params(header_params: List[SP3Param],
                      sload_params: List[SP3Param]) -> List[SP3Param]:
    """Merge header-parsed params with ground-truth s_load params.

    s_load offsets always win when they conflict with header offsets.
    New params found only in s_load are appended.
    Header params not found in s_load are kept (e.g. compile-time constants).
    """
    # Build header lookup: normalized sreg -> SP3Param
    def _norm(s: str) -> str:
        return s.lower().lstrip('_')

    header_by_sreg: Dict[str, SP3Param] = {}
    for p in header_params:
        header_by_sreg[_norm(p.sreg)] = p

    matched_header_keys = set()
    merged: List[SP3Param] = []

    for sp in sload_params:
        key = _norm(sp.sreg)
        hp = header_by_sreg.get(key)
        if hp is not None:
            # Match found -- prefer s_load offset; keep header name/comment
            if hp.offset != sp.offset:
                print(f'  [sload-fix] {hp.name}: header offset 0x{hp.offset:X} '
                      f'-> actual 0x{sp.offset:X}', file=sys.stderr)
            merged.append(SP3Param(
                name=hp.name, size=sp.size, offset=sp.offset,
                comment=hp.comment, sreg=hp.sreg,
                is_pointer=sp.is_pointer))
            matched_header_keys.add(key)
        else:
            # New param only in kernel body
            print(f'  [sload-new] {sp.name} at 0x{sp.offset:X} '
                  f'({"ptr" if sp.is_pointer else "scalar"})',
                  file=sys.stderr)
            merged.append(sp)

    # Keep header-only params (e.g. log2e) that had no s_load match
    for p in header_params:
        if _norm(p.sreg) not in matched_header_keys:
            # Only keep if no sload param occupies the same offset
            sload_offsets = {sp.offset for sp in sload_params}
            if p.offset not in sload_offsets:
                merged.append(p)
            else:
                print(f'  [sload-drop] header param "{p.name}" at 0x{p.offset:X} '
                      f'superseded by s_load param', file=sys.stderr)

    # Sort by offset for readability
    merged.sort(key=lambda p: p.offset)
    return merged


# ---------------------------------------------------------------------------
# C++ Code Generator
# ---------------------------------------------------------------------------

def generate_cpp(info: SP3Info, output_path: str) -> str:
    """Generate a standalone HIP C++ host program from parsed SP3 info."""

    pointer_params = [p for p in info.params if p.is_pointer]
    scalar_params = [p for p in info.params if not p.is_pointer]

    # Get key constants with defaults
    SUB_X = info.constants.get('SUB_X', '32')
    SUB_GU = info.constants.get('SUB_GU', '512')
    Bpp = info.constants.get('Bpp', '1')
    DO_UP = info.constants.get('DO_UP', '1')

    # Translate grid formulas to C++ expressions
    gdx_expr = _grid_formula_to_cpp(info.grid.global_size_x, info.grid.local_size_x, info)
    gdy_expr = _grid_formula_to_cpp(info.grid.global_size_y, info.grid.local_size_y, info)
    gdz_expr = _grid_formula_to_cpp(info.grid.global_size_z, info.grid.local_size_z, info)
    bdx_expr = info.grid.local_size_x
    bdy_expr = info.grid.local_size_y
    bdz_expr = info.grid.local_size_z

    # Build the C++ source
    lines = []
    lines.append(_gen_header(info))
    lines.append(_gen_cli_parser(info, pointer_params, scalar_params))
    lines.append(_gen_constants(info))
    lines.append(_gen_moe_routing(info))
    lines.append(_gen_buffer_alloc(info, pointer_params))
    lines.append(_gen_kernarg_assembly(info, pointer_params, scalar_params))
    lines.append(_gen_launch(info, gdx_expr, gdy_expr, gdz_expr,
                             bdx_expr, bdy_expr, bdz_expr))
    lines.append(_gen_cleanup(pointer_params))

    code = '\n'.join(lines)

    with open(output_path, 'w') as f:
        f.write(code)

    return code


def _grid_formula_to_cpp(global_expr: str, local_expr: str, info: SP3Info) -> str:
    """Convert SP3 grid formula to C++ grid-dim expression (grid = global / local)."""
    # Replace SP3 variable names with C++ variable names
    g = global_expr
    l = local_expr

    # sub_GU -> SUB_GU (use the C++ constant name)
    g = re.sub(r'\bsub_GU\b', 'SUB_GU', g)
    g = re.sub(r'\bsub_X_cnt\b', 'sub_X_cnt', g)

    try:
        local_val = int(l)
    except ValueError:
        local_val = None

    # If global already contains /256 pattern with local=256, simplify
    if local_val and local_val > 1:
        # Check if global already divides by local
        pattern = f'*{local_val}'
        if pattern in g:
            # e.g., (hidden_dim/SUB_GU)*256 with local=256 -> (hidden_dim/SUB_GU)
            return g.replace(f'*{local_val}', '')
        else:
            return f'({g}) / {local_val}'
    elif local_val == 1:
        return g
    else:
        return f'({g}) / ({l})'


def _gen_header(info: SP3Info) -> str:
    return f'''\
// ============================================================================
// Auto-generated by sp3_host_gen.py from {info.filename}
// This is a standalone HIP test driver for debugging gpu.printf output.
//
// Compile: hipcc --offload-arch=gfx942 -O0 -g {os.path.basename(info.filename).replace('.sp3', '_test.cpp')} -o test_kernel
// Run:     ./test_kernel --hsaco <path> --kernel <symbol> [options]
// ============================================================================
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#define HIP_CHECK(cmd) do {{ \\
    hipError_t err = (cmd); \\
    if (err != hipSuccess) {{ \\
        fprintf(stderr, "HIP error %d (%s) at %s:%d\\n", \\
                (int)err, hipGetErrorString(err), __FILE__, __LINE__); \\
        exit(1); \\
    }} \\
}} while(0)

static void print_usage(const char *prog) {{
    fprintf(stderr,
        "Usage: %s --hsaco <path> --kernel <symbol> [options]\\n"
        "Options:\\n"
        "  --hsaco <path>     Path to HSACO file (required)\\n"
        "  --kernel <symbol>  Kernel symbol name (required)\\n"
        "  --dim <N>          dim value (default: 7168)\\n"
        "  --hidden-dim <N>   hidden_dim value (default: 2048)\\n"
        "  --batch <N>        Batch/token count (default: 64)\\n"
        "  --topk <N>         topk value (default: 8)\\n"
        "  --eprt <N>         Expert count (default: 128)\\n"
        "  --buf-size <MB>    Fallback buffer size in MB (default: 64)\\n"
        "  --device <N>       GPU device index (default: 0)\\n"
        "  --help             Show this help\\n",
        prog);
}}
'''


def _gen_cli_parser(info: SP3Info, pointer_params: list, scalar_params: list) -> str:
    return '''\
int main(int argc, char **argv) {
    // ---- CLI argument parsing ----
    std::string hsaco_path;
    std::string kernel_sym;
    int dim        = 7168;
    int hidden_dim = 2048;
    int batch      = 64;
    int topk       = 8;
    int eprt       = 128;
    int buf_size_mb = 64;
    int device_id  = 0;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--hsaco"      && i+1 < argc) { hsaco_path  = argv[++i]; }
        else if (arg == "--kernel"     && i+1 < argc) { kernel_sym  = argv[++i]; }
        else if (arg == "--dim"        && i+1 < argc) { dim         = atoi(argv[++i]); }
        else if (arg == "--hidden-dim" && i+1 < argc) { hidden_dim  = atoi(argv[++i]); }
        else if (arg == "--batch"      && i+1 < argc) { batch       = atoi(argv[++i]); }
        else if (arg == "--topk"       && i+1 < argc) { topk        = atoi(argv[++i]); }
        else if (arg == "--eprt"       && i+1 < argc) { eprt        = atoi(argv[++i]); }
        else if (arg == "--buf-size"   && i+1 < argc) { buf_size_mb = atoi(argv[++i]); }
        else if (arg == "--device"     && i+1 < argc) { device_id   = atoi(argv[++i]); }
        else if (arg == "--help") { print_usage(argv[0]); return 0; }
        else { fprintf(stderr, "Unknown arg: %s\\n", argv[i]); print_usage(argv[0]); return 1; }
    }

    if (hsaco_path.empty() || kernel_sym.empty()) {
        fprintf(stderr, "Error: --hsaco and --kernel are required.\\n");
        print_usage(argv[0]);
        return 1;
    }

    printf("=== SP3 Host Test Driver ===\\n");
    printf("HSACO      : %s\\n", hsaco_path.c_str());
    printf("Kernel     : %s\\n", kernel_sym.c_str());
    printf("dim        : %d\\n", dim);
    printf("hidden_dim : %d\\n", hidden_dim);
    printf("batch      : %d\\n", batch);
    printf("topk       : %d\\n", topk);
    printf("eprt       : %d\\n", eprt);
    printf("buf_size   : %d MB\\n", buf_size_mb);
    printf("device     : %d\\n", device_id);
'''


def _gen_constants(info: SP3Info) -> str:
    SUB_X = info.constants.get('SUB_X', '32')
    SUB_GU = info.constants.get('SUB_GU', '512')
    Bpp = info.constants.get('Bpp', '1')
    DO_UP = info.constants.get('DO_UP', '1')

    return f'''
    // ---- SP3 Constants ----
    const int SUB_X  = {SUB_X};
    const int SUB_GU = {SUB_GU};
    const int Bpp    = {Bpp};        // bytes per point
    const int DO_UP  = {DO_UP};      // G1U1 vs G1U0

    // ---- Derived dimensions ----
    const int u1_factor = (DO_UP == 1) ? 2 : 1;  // GU combined if u1
'''


def _gen_moe_routing(info: SP3Info) -> str:
    """Generate MoE routing data (STP/SW/SEP) initialization."""
    return '''
    // ---- MoE routing data (uniform round-robin for testing) ----
    // STP: sorted_token_ids  -- which token each slot maps to
    // SW:  sorted_weights    -- weight for each slot
    // SEP: sorted_expert_ids -- expert-id for each sub_X group

    int total_tokens = batch * topk;
    int sz_stp = total_tokens + eprt * SUB_X - topk;  // max length
    if (sz_stp < total_tokens) sz_stp = total_tokens;
    int sz_sw  = sz_stp;
    int sz_sep = (sz_stp + SUB_X - 1) / SUB_X;  // max groups
    int sub_X_cnt = sz_sep;   // for grid dim calculation

    // Allocate host routing arrays
    std::vector<uint32_t> h_stp(sz_stp, 0);
    std::vector<float>    h_sw(sz_sw, 1.0f);
    std::vector<uint32_t> h_sep(sz_sep, 0);

    // Round-robin assignment: token i -> expert (i % eprt)
    // Group tokens by expert in sub_X-sized chunks
    {
        int slot = 0;
        int group = 0;
        for (int e = 0; e < eprt && group < sz_sep; e++) {
            h_sep[group] = e;
            int tokens_for_expert = 0;
            for (int t = 0; t < batch; t++) {
                for (int k = 0; k < topk; k++) {
                    int token_expert = (t * topk + k) % eprt;
                    if (token_expert == e && slot < sz_stp) {
                        h_stp[slot] = t;
                        h_sw[slot]  = 1.0f;
                        slot++;
                        tokens_for_expert++;
                    }
                }
            }
            // Pad to SUB_X boundary with dummy token ids
            while (tokens_for_expert % SUB_X != 0 && slot < sz_stp) {
                h_stp[slot] = batch;  // out-of-range token (safe: will read zeros)
                h_sw[slot]  = 0.0f;
                slot++;
                tokens_for_expert++;
            }
            group += tokens_for_expert / SUB_X;
        }
        sub_X_cnt = group;  // actual number of groups
        printf("sub_X_cnt  : %d (routing groups)\\n", sub_X_cnt);
    }
'''


def _gen_buffer_alloc(info: SP3Info, pointer_params: list) -> str:
    """Generate hipMalloc calls for each pointer parameter."""
    lines = []
    lines.append('''
    // ---- HIP setup & buffer allocation ----
    HIP_CHECK(hipSetDevice(device_id));

    hipModule_t   module;
    hipFunction_t kernel_func;
    HIP_CHECK(hipModuleLoad(&module, hsaco_path.c_str()));
    HIP_CHECK(hipModuleGetFunction(&kernel_func, module, kernel_sym.c_str()));
    printf("Kernel loaded successfully.\\n");

    size_t fallback_buf = (size_t)buf_size_mb * 1024 * 1024;
''')

    # For each pointer param, compute size and allocate
    for p in pointer_params:
        var = f'dev_{p.name}'
        size_expr = _buffer_size_expr(p, info)
        lines.append(f'''
    // {p.name}: {p.comment}
    void *{var} = nullptr;
    {{
        size_t sz = {size_expr};
        if (sz == 0) sz = fallback_buf;
        HIP_CHECK(hipMalloc(&{var}, sz));
        HIP_CHECK(hipMemset({var}, 0, sz));
        printf("{p.name:20s}: %p  (%zu bytes)\\n", {var}, sz);
    }}''')

    # Upload routing data for STP, SW, SEP
    lines.append('''
    // Upload routing data
    if (dev_STP && sz_stp > 0)
        HIP_CHECK(hipMemcpy(dev_STP, h_stp.data(), sz_stp * sizeof(uint32_t), hipMemcpyHostToDevice));
    if (dev_SW && sz_sw > 0)
        HIP_CHECK(hipMemcpy(dev_SW, h_sw.data(), sz_sw * sizeof(float), hipMemcpyHostToDevice));
    if (dev_SEP && sz_sep > 0)
        HIP_CHECK(hipMemcpy(dev_SEP, h_sep.data(), sub_X_cnt * sizeof(uint32_t), hipMemcpyHostToDevice));
''')

    return '\n'.join(lines)


def _buffer_size_expr(param: SP3Param, info: SP3Info) -> str:
    """Return a C++ expression for the buffer size of a pointer parameter."""
    name = param.name.upper()
    if not info.is_fmoe:
        return 'fallback_buf'

    # FMOE-family buffer size formulas
    size_map = {
        'R':    '(size_t)batch * dim * 2',            # bf16 output
        'X':    '(size_t)batch * dim * Bpp',           # input tokens
        'G':    '(size_t)eprt * hidden_dim * dim * Bpp',  # gate weights
        'U':    '(size_t)eprt * hidden_dim * dim * Bpp',  # up weights
        'D':    '(size_t)eprt * dim * hidden_dim * Bpp',  # down weights
        'STP':  '(size_t)sz_stp * sizeof(uint32_t)',       # sorted_token_ids
        'SW':   '(size_t)sz_sw * sizeof(float)',           # sorted_weights
        'SEP':  '(size_t)sz_sep * sizeof(uint32_t)',       # expert_offsets
        # Quantization-related buffers (INT8/FP8 kernels)
        'XC':   'fallback_buf',       # reserved / XC buffer
        'XQ':   'fallback_buf',       # X-quantization scales
        'GQ':   'fallback_buf',       # G-quantization scales
        'DQ':   'fallback_buf',       # D-quantization scales
        'GSMQ': 'fallback_buf',       # smooth quantization scales
    }
    return size_map.get(name, 'fallback_buf')


def _gen_kernarg_assembly(info: SP3Info, pointer_params: list, scalar_params: list) -> str:
    """Generate the flat kernarg buffer assembly."""
    lines = []
    lines.append(f'''
    // ---- Kernarg buffer assembly ----
    // Each parameter slot is 0x10 (16 bytes) wide.
    // Total kernarg size: 0x{info.kernarg_size:X} ({info.kernarg_size} bytes)
    uint8_t kernarg[{info.kernarg_size}];
    memset(kernarg, 0, sizeof(kernarg));
''')

    # Stride computations
    lines.append('''    // Compute stride values
    int stride_X          = dim * Bpp;
    int stride_GU         = dim * Bpp;
    int stride_D          = hidden_dim * Bpp;
    int stride_R          = dim * 2;  // bf16 output
    int stride_expert_GU  = stride_GU * hidden_dim * u1_factor;
    int stride_expert_D   = stride_D * dim;
''')

    # Write pointer params
    for p in pointer_params:
        var = f'dev_{p.name}'
        lines.append(f'    *(void**)(kernarg + 0x{p.offset:X}) = {var};'
                      f'  // {p.name} ({p.comment})')

    lines.append('')

    # Write scalar params
    for p in scalar_params:
        val_expr = _scalar_value_expr(p)
        lines.append(f'    *(uint32_t*)(kernarg + 0x{p.offset:X}) = (uint32_t){val_expr};'
                      f'  // {p.name} ({p.comment})')

    return '\n'.join(lines)


def _scalar_value_expr(param: SP3Param) -> str:
    """Return a C++ expression for a scalar parameter value."""
    name = param.name.lower()

    # Map known scalar params to their C++ variable names.
    # Includes both header names (dim, hidden_dim, stride_x, ...) and
    # kernel-body variable names (dim_len, hidden_len, Xs, token_cnt, ...).
    scalar_map = {
        # --- original header names ---
        'log2e':           '0x3FB8AA3B',         # float bits for log2(e) = 1.4427
        'dim':             'dim',
        'hidden_dim':      'hidden_dim',
        'topk':            'topk',
        'eprt':            'eprt',
        'stride_x':        'stride_X',
        'stride_gu':       'stride_GU',
        'stride_d':        'stride_D',
        'stride_r':        'stride_R',
        'stride_expert_gu':'stride_expert_GU',
        'stride_expert_d': 'stride_expert_D',
        # --- kernel-body s_load variable names (aliases) ---
        'dim_len':         'dim',
        'hidden_len':      'hidden_dim',
        'token_cnt':       'batch * topk',
        'eprt_cnt':        'eprt',
        'xs':              'stride_X',
        'gus':             'stride_GU',
        'ds':              'stride_D',
        'rs':              'stride_R',
        'egus':            'stride_expert_GU',
        'eds':             'stride_expert_D',
        # --- quantization strides (zero for test) ---
        'eguqs':           '0',
        'edqs':            '0',
        'egsmqs':          '0',
    }
    return scalar_map.get(name, '0  /* unknown */')


def _gen_launch(info: SP3Info, gdx: str, gdy: str, gdz: str,
                bdx: str, bdy: str, bdz: str) -> str:
    return f'''
    // ---- Kernel launch ----
    int gdx = {gdx};
    int gdy = {gdy};
    int gdz = {gdz};
    int bdx = {bdx};
    int bdy = {bdy};
    int bdz = {bdz};

    if (gdx <= 0) gdx = 1;
    if (gdy <= 0) gdy = 1;
    if (gdz <= 0) gdz = 1;

    printf("Grid  : (%d, %d, %d)\\n", gdx, gdy, gdz);
    printf("Block : (%d, %d, %d)\\n", bdx, bdy, bdz);

    size_t arg_size = sizeof(kernarg);
    void *config[] = {{
        HIP_LAUNCH_PARAM_BUFFER_POINTER, kernarg,
        HIP_LAUNCH_PARAM_BUFFER_SIZE,    &arg_size,
        HIP_LAUNCH_PARAM_END
    }};

    printf("Launching kernel...\\n");
    fflush(stdout);
    fflush(stderr);

    HIP_CHECK(hipModuleLaunchKernel(
        kernel_func,
        gdx, gdy, gdz,
        bdx, bdy, bdz,
        0,        // shared mem
        nullptr,  // stream (default)
        nullptr,  // kernel params (using config instead)
        (void **)&config
    ));

    HIP_CHECK(hipDeviceSynchronize());
    printf("Kernel completed.\\n");
    fflush(stdout);
    fflush(stderr);
'''


def _gen_cleanup(pointer_params: list) -> str:
    lines = ['\n    // ---- Cleanup ----']
    for p in pointer_params:
        lines.append(f'    if (dev_{p.name}) (void)hipFree(dev_{p.name});')
    lines.append('    (void)hipModuleUnload(module);')
    lines.append('    printf("Done.\\n");')
    lines.append('    return 0;')
    lines.append('}')
    lines.append('')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Compile helper
# ---------------------------------------------------------------------------

def compile_cpp(cpp_path: str, output_path: str = None, arch: str = 'gfx942') -> bool:
    """Compile the generated C++ file with hipcc."""
    if output_path is None:
        output_path = cpp_path.replace('.cpp', '')
        if output_path == cpp_path:
            output_path += '.exe'

    cmd = [
        '/opt/rocm/bin/hipcc',
        f'--offload-arch={arch}',
        '-O0', '-g',
        cpp_path,
        '-o', output_path
    ]
    print(f'[Compile] {" ".join(cmd)}')
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f'[Error] Compilation failed:', file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        return False
    print(f'[OK] Compiled: {output_path}')
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate HIP C++ host code from an SP3 shader file.')
    parser.add_argument('sp3_file', help='Input SP3 shader file')
    parser.add_argument('-o', '--output', default=None,
                        help='Output C++ file path (default: <sp3_name>_test.cpp)')
    parser.add_argument('--compile', action='store_true',
                        help='Also compile the generated C++ with hipcc')
    parser.add_argument('--arch', default='gfx942',
                        help='GPU architecture for hipcc (default: gfx942)')
    parser.add_argument('--exe', default=None,
                        help='Output executable path (with --compile)')
    args = parser.parse_args()

    if not os.path.isfile(args.sp3_file):
        print(f'Error: File not found: {args.sp3_file}', file=sys.stderr)
        sys.exit(1)

    # Default output name
    if args.output is None:
        base = os.path.splitext(os.path.basename(args.sp3_file))[0]
        args.output = base + '_test.cpp'

    print(f'[Parse] {args.sp3_file}')
    info = parse_sp3_header(args.sp3_file)

    print(f'  Parameters: {len(info.params)} '
          f'({sum(1 for p in info.params if p.is_pointer)} pointers, '
          f'{sum(1 for p in info.params if not p.is_pointer)} scalars)')
    print(f'  Kernarg size: 0x{info.kernarg_size:X} ({info.kernarg_size} bytes)')
    print(f'  Constants: {dict(list(info.constants.items())[:8])}...')
    print(f'  Grid: global=({info.grid.global_size_x}, {info.grid.global_size_y}, '
          f'{info.grid.global_size_z}), '
          f'local=({info.grid.local_size_x}, {info.grid.local_size_y}, '
          f'{info.grid.local_size_z})')
    print(f'  FMOE family: {info.is_fmoe}')

    print(f'[Generate] {args.output}')
    generate_cpp(info, args.output)
    print(f'[OK] Generated: {args.output}')

    if args.compile:
        compile_cpp(args.output, args.exe, args.arch)


if __name__ == '__main__':
    main()
