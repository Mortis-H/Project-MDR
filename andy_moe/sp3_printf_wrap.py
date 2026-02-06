#!/usr/bin/env python3
"""
Wrapper: keep @PRINT/@TIMESTAMP annotations from SP3, re-inject into .s,
then call gpr_printf_tool.py to generate HSACO.
"""
from __future__ import annotations

import argparse
import difflib
import os
import pathlib
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent

DEFAULT_SP3_DIR = "/home/root123/andy-mdr/poc_kl/mi300/fused_moe_asm/mi300_sp3_to_asm"
DEFAULT_MOE_CVT = "/home/root123/andy-mdr/poc_kl/scripts/fused_moe/moe_cvt.py"
DEFAULT_FIX_SCRIPT = "/home/root123/andy-mdr/poc_kl/mi300/fix_atomic_add.py"
DEFAULT_KERNEL_SYMBOL = "_ZN5aiter45fmoe_bf16_pertokenFp8_g1u1_vs_silu_1tg_32x192E"
DEFAULT_GPR_TOOL = str(PROJECT_ROOT / "gpr_printf_tool.py")

DIRECTIVE_RE = re.compile(r"@PRINT\b|@TIMESTAMP_START\b|@TIMESTAMP_END\b")


@dataclass
class LineInfo:
    func: Optional[str]
    opcode: Optional[str]
    instr_index: Optional[int]


@dataclass
class Directive:
    kind: str
    text: str
    func: str
    anchor_index: int
    line_no: int


def run_cmd(cmd: List[str], env: Optional[Dict[str, str]] = None) -> None:
    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def strip_inline_comment(line: str) -> str:
    indices = []
    for marker in ("//", "#", ";"):
        idx = line.find(marker)
        if idx != -1:
            indices.append(idx)
    if not indices:
        return line
    return line[: min(indices)]


def extract_opcode(line: str) -> Optional[str]:
    no_comment = strip_inline_comment(line)
    stripped = no_comment.strip()
    if not stripped:
        return None
    if stripped.startswith(("//", "#", ";")):
        return None
    if stripped.endswith(":"):
        return None
    lowered = stripped.lower()
    if lowered.startswith((
        "shader ",
        "type(",
        "asic(",
        "user_sgpr_count",
        "tgid_x_en",
        "tgid_y_en",
        "tgid_z_en",
        "tidig_comp_cnt",
        "var ",
        "if ",
        "for ",
        "while ",
        "else",
        "end",
        "function ",
        "macro ",
        "print ",
        ".",
    )):
        return None
    return stripped.split()[0]


def current_function(stack: List[Tuple[str, str]]) -> Optional[str]:
    for block_type, name in reversed(stack):
        if block_type in ("function", "shader"):
            return name
    return None


def parse_sp3_source(path: pathlib.Path) -> Tuple[List[str], List[LineInfo], Dict[str, List[str]]]:
    lines = path.read_text().splitlines()
    stack: List[Tuple[str, str]] = []
    func_ops: Dict[str, List[str]] = {}
    func_instr_counts: Dict[str, int] = {}
    line_infos: List[LineInfo] = []

    func_start_re = re.compile(r"^\s*function\s+([A-Za-z0-9_]+)")

    for line in lines:
        stripped = line.strip()

        func_match = func_start_re.match(stripped)
        if func_match:
            func_name = func_match.group(1)
            stack.append(("function", func_name))
            line_infos.append(LineInfo(current_function(stack), None, None))
            continue

        if stripped.startswith("shader main"):
            stack.append(("shader", "__shader_main__"))
            line_infos.append(LineInfo(current_function(stack), None, None))
            continue

        if re.match(r"^\s*(if|for|while)\b", stripped):
            stack.append(("block", ""))

        func = current_function(stack)
        opcode = extract_opcode(line)
        instr_index = None
        if opcode and func:
            func_ops.setdefault(func, [])
            func_instr_counts.setdefault(func, 0)
            instr_index = func_instr_counts[func]
            func_instr_counts[func] += 1
            func_ops[func].append(opcode)

        line_infos.append(LineInfo(func, opcode, instr_index))

        if re.match(r"^\s*end\b", stripped):
            if stack:
                stack.pop()

    return lines, line_infos, func_ops


def extract_directive_text(line: str) -> Optional[str]:
    match = re.search(r"@(PRINT|TIMESTAMP_START|TIMESTAMP_END)\b.*", line)
    if not match:
        return None
    return line[match.start():].strip()


def collect_directives(
    lines: List[str],
    line_infos: List[LineInfo],
) -> List[Directive]:
    directives: List[Directive] = []
    for idx, line in enumerate(lines):
        if not DIRECTIVE_RE.search(line):
            continue
        directive_text = extract_directive_text(line)
        if not directive_text:
            continue
        info = line_infos[idx]
        if not info.func:
            print(f"[Warn] Directive outside any function at line {idx + 1}, skip.")
            continue
        anchor_index = info.instr_index
        if anchor_index is None:
            for j in range(idx + 1, len(lines)):
                next_info = line_infos[j]
                if next_info.func == info.func and next_info.instr_index is not None:
                    anchor_index = next_info.instr_index
                    break
        if anchor_index is None:
            print(f"[Warn] Cannot find anchor instruction for directive at line {idx + 1}, skip.")
            continue
        kind = "PRINT" if "@PRINT" in directive_text else "TIMESTAMP"
        directives.append(
            Directive(
                kind=kind,
                text=directive_text,
                func=info.func,
                anchor_index=anchor_index,
                line_no=idx + 1,
            )
        )
    return directives


def parse_disasm(path: pathlib.Path) -> Tuple[List[str], List[str], Dict[int, int], Dict[int, int]]:
    lines = path.read_text().splitlines()
    opcodes: List[str] = []
    op_index_to_line: Dict[int, int] = {}
    line_to_op_index: Dict[int, int] = {}
    for idx, line in enumerate(lines):
        opcode = extract_opcode(line)
        if opcode:
            op_index = len(opcodes)
            opcodes.append(opcode)
            op_index_to_line[op_index] = idx
            line_to_op_index[idx] = op_index
    return lines, opcodes, op_index_to_line, line_to_op_index


def build_index_map(src_ops: List[str], dst_ops: List[str]) -> Dict[int, int]:
    matcher = difflib.SequenceMatcher(a=src_ops, b=dst_ops, autojunk=False)
    mapping: Dict[int, int] = {}
    for a0, b0, size in matcher.get_matching_blocks():
        for i in range(size):
            mapping[a0 + i] = b0 + i
    return mapping


def find_pattern_matches(dis_ops: List[str], pattern: List[str]) -> List[int]:
    if not pattern:
        return []
    matches: List[int] = []
    plen = len(pattern)
    for i in range(len(dis_ops) - plen + 1):
        if dis_ops[i : i + plen] == pattern:
            matches.append(i)
    return matches


def build_pattern(func_ops: List[str], idx: int, window: int) -> Tuple[List[str], int]:
    start = max(0, idx - window)
    end = min(len(func_ops), idx + window + 1)
    return func_ops[start:end], idx - start


def insert_directives_into_disasm(
    disasm_lines: List[str],
    dis_ops: List[str],
    line_to_op_index: Dict[int, int],
    directives: List[Directive],
    func_ops: Dict[str, List[str]],
) -> List[str]:
    insertions: Dict[int, List[str]] = {}

    main_ops = func_ops.get("__shader_main__")
    main_map = build_index_map(main_ops, dis_ops) if main_ops else {}

    for d in directives:
        if d.func == "__shader_main__":
            dis_idx = main_map.get(d.anchor_index)
            if dis_idx is None:
                print(f"[Warn] Cannot map main directive (line {d.line_no}) to disasm.")
                continue
            insertions.setdefault(dis_idx, []).append(d.text)
            continue

        func_list = func_ops.get(d.func)
        if not func_list:
            print(f"[Warn] Missing function op list for {d.func}, skip directive at line {d.line_no}.")
            continue

        matched = False
        for window in (3, 2, 1, 0):
            pattern, offset = build_pattern(func_list, d.anchor_index, window)
            match_starts = find_pattern_matches(dis_ops, pattern)
            if match_starts:
                matched = True
                for start in match_starts:
                    dis_idx = start + offset
                    insertions.setdefault(dis_idx, []).append(d.text)
                if len(match_starts) > 20:
                    print(
                        f"[Warn] Directive at line {d.line_no} matched {len(match_starts)} times; "
                        "please verify placement."
                    )
                break
        if not matched:
            print(f"[Warn] Cannot map directive (line {d.line_no}) from function {d.func} to disasm.")

    new_lines: List[str] = []
    for line_idx, line in enumerate(disasm_lines):
        op_index = line_to_op_index.get(line_idx)
        if op_index is not None and op_index in insertions:
            indent = re.match(r"^(\s*)", line).group(1)
            for text in insertions[op_index]:
                new_lines.append(f"{indent}// {text}")
        new_lines.append(line)

    return new_lines


def normalize_directive_comments(input_path: pathlib.Path, output_path: pathlib.Path) -> None:
    lines = input_path.read_text().splitlines()
    out_lines: List[str] = []
    for line in lines:
        if "@PRINT" in line or "@TIMESTAMP_START" in line or "@TIMESTAMP_END" in line:
            line = re.sub(r"//\s*(?=@PRINT|@TIMESTAMP_)", "; ", line)
        out_lines.append(line)
    output_path.write_text("\n".join(out_lines) + "\n")


def fix_atomic_add_syntax(input_path: pathlib.Path, output_path: pathlib.Path) -> None:
    lines = input_path.read_text().splitlines()
    out_lines: List[str] = []
    atomic_re = re.compile(
        r"^(?P<indent>\s*)global_atomic_add\s+v0,\s*v\[(?P<addr>\d+):\d+\],\s*"
        r"(?P<val>v\d+),\s*(?P<sreg>s\[\d+:\d+\])\s*(?P<offset>inst_offset:\d+)?"
    )
    for line in lines:
        if re.match(r"^\s*type\(CS\)\s*$", line):
            continue
        match = atomic_re.match(line)
        if match:
            indent = match.group("indent")
            addr = match.group("addr")
            val = match.group("val")
            sreg = match.group("sreg")
            offset = match.group("offset")
            offset_str = f" {offset}" if offset else ""
            out_lines.append(
                f"{indent}global_atomic_add   v{addr}, {val}, {sreg}{offset_str}"
            )
            continue
        out_lines.append(line)
    output_path.write_text("\n".join(out_lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="SP3 -> .s -> gpr_printf wrapper with @PRINT re-injection")
    ap.add_argument("input_sp3", help="原始 SP3 檔案（含 @PRINT/@TIMESTAMP 註解）")
    ap.add_argument("--output-dir", required=True, help="輸出目錄")
    ap.add_argument("--prefix", default="kernel", help="輸出檔名前綴（預設：kernel）")
    ap.add_argument("--sp3-dir", default=DEFAULT_SP3_DIR, help="SP3 compiler 目錄")
    ap.add_argument("--moe-cvt", default=DEFAULT_MOE_CVT, help="moe_cvt.py 路徑")
    ap.add_argument("--fix-script", default=DEFAULT_FIX_SCRIPT, help="fix_atomic_add.py 路徑")
    ap.add_argument(
        "--atomic-fix-mode",
        choices=("syntax", "long"),
        default="syntax",
        help="atomic add 修復模式：syntax(保留指令) 或 long(轉為 .long)",
    )
    ap.add_argument("--kernel-symbol", default=DEFAULT_KERNEL_SYMBOL, help="symbol 重命名")
    ap.add_argument("--asic", default="MI300", help="SP3 asic (預設 MI300)")
    ap.add_argument("--wave-size", default="64", help="SP3 wave_size (預設 64)")
    ap.add_argument("--chip", default="gfx942", help="gpr_printf_tool 目標架構")
    ap.add_argument("--gpr-output-dir", default=None, help="gpr_printf_tool 輸出目錄")
    ap.add_argument("--skip-gpr", action="store_true", help="不執行 gpr_printf_tool")
    ap.add_argument("--keep-disasm", action="store_true", help="保留中間檔案")
    args, gpr_args = ap.parse_known_args()

    input_sp3 = pathlib.Path(args.input_sp3).resolve()
    if not input_sp3.exists():
        raise FileNotFoundError(f"Input SP3 not found: {input_sp3}")

    output_dir = pathlib.Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = args.prefix
    bin_path = output_dir / f"{prefix}.bin"
    disasm_path = output_dir / f"{prefix}_disasm.sp3"
    disasm_print_path = output_dir / f"{prefix}_disasm_print.sp3"
    raw_s_path = output_dir / f"{prefix}_raw.s"
    fixed_s_path = output_dir / f"{prefix}_fixed.s"
    fixed_print_s_path = output_dir / f"{prefix}_fixed_print.s"

    print("=== Parse SP3 directives ===")
    src_lines, line_infos, func_ops = parse_sp3_source(input_sp3)
    directives = collect_directives(src_lines, line_infos)
    print(f"Found {len(directives)} directive(s)")

    print("\n=== Step 1: SP3 -> Binary ===")
    sp3_bin = pathlib.Path(args.sp3_dir) / "sp3"
    env = os.environ.copy()
    existing_ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{args.sp3_dir}:{existing_ld}" if existing_ld else args.sp3_dir
    run_cmd([str(sp3_bin), str(input_sp3), f"asic={args.asic}", "-binary", str(bin_path)], env=env)

    print("\n=== Step 2: Binary -> Disasm SP3 ===")
    run_cmd([
        str(sp3_bin),
        "-binary",
        str(bin_path),
        f"asic={args.asic}",
        "type=CS",
        f"wave_size={args.wave_size}",
        str(disasm_path),
    ], env=env)

    print("\n=== Step 3: Inject directives into disasm ===")
    disasm_lines, dis_ops, _, line_to_op_index = parse_disasm(disasm_path)
    if directives:
        disasm_with_print = insert_directives_into_disasm(
            disasm_lines, dis_ops, line_to_op_index, directives, func_ops
        )
        disasm_print_path.write_text("\n".join(disasm_with_print) + "\n")
    else:
        disasm_print_path.write_text("\n".join(disasm_lines) + "\n")
    print(f"Annotated disasm: {disasm_print_path}")

    print("\n=== Step 4: SP3 -> LLVM Assembly ===")
    run_cmd([
        sys.executable,
        args.moe_cvt,
        str(input_sp3),
        str(disasm_print_path),
        str(raw_s_path),
    ])

    print("\n=== Step 5: Fix Atomic Add Syntax ===")
    if args.atomic_fix_mode == "long":
        run_cmd([sys.executable, args.fix_script, str(raw_s_path), str(fixed_s_path)])
    else:
        fix_atomic_add_syntax(raw_s_path, fixed_s_path)

    print("\n=== Step 6: Rename Symbol ===")
    fixed_text = fixed_s_path.read_text()
    fixed_text = fixed_text.replace("fmoe_kernel_func", args.kernel_symbol)
    fixed_s_path.write_text(fixed_text)

    print("\n=== Step 7: Normalize directive comment prefix ===")
    normalize_directive_comments(fixed_s_path, fixed_print_s_path)

    if args.skip_gpr:
        print("\n=== Done (skip gpr_printf_tool) ===")
        print(f"Annotated .s: {fixed_print_s_path}")
        return 0

    gpr_output_dir = pathlib.Path(args.gpr_output_dir) if args.gpr_output_dir else output_dir / "gpr_printf_out"
    gpr_output_dir = gpr_output_dir.resolve()
    gpr_output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Step 8: gpr_printf_tool ===")
    gpr_cmd = [
        sys.executable,
        DEFAULT_GPR_TOOL,
        str(fixed_print_s_path),
        "--output-dir",
        str(gpr_output_dir),
        "--chip",
        args.chip,
    ] + gpr_args
    run_cmd(gpr_cmd)

    print("\n=== Done ===")
    print(f"Final .s: {fixed_print_s_path}")
    print(f"gpr_printf output: {gpr_output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
