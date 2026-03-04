#!/usr/bin/env python3
"""
SP3 @CAPTURE Wrapper
====================

從 SP3 原始碼中提取 @CAPTURE 標記，經 SP3 編譯 -> 反組譯 -> 重新注入 @CAPTURE
-> 轉換為 LLVM Assembly (.s) -> 呼叫 mdr_cap.py 生成含有 register snapshot 的 HSACO。

@CAPTURE 讓使用者在 SP3 中標記 f-string 計算式，工具會自動生成 ISA 指令，
將計算結果存入指定的目標暫存器。

SP3 中的使用方式：
  // @CAPTURE f"A={v2}, B={v3}" dst=v10,v11
  // @CAPTURE if $tid == 0: f"A*2={(v2*2.0):.2f}" dst=v12
  // @CAPTURE f"addr={(s[12:13] + s[16:17]*4):ld}" dst=s[12:13]

Pipeline:
  1. 解析 SP3 原始碼中的 @CAPTURE 指令
  2. SP3 -> Binary（編譯，註解被移除）
  3. Binary -> Disasm SP3（反組譯）
  4. 將 @CAPTURE 指令重新注入反組譯結果（透過 token 比對找到正確位置）
  5. SP3 -> LLVM Assembly (.s)
  6. 修正 atomic add 語法 / symbol 重命名
  7. 正規化 @CAPTURE 的註解前綴（// -> ;）
  8. 呼叫 mdr_cap.py 解析 @CAPTURE，插入 ISA 指令，生成 HSACO
"""
from __future__ import annotations

import argparse
import os
import pathlib
import re
import sys
from typing import Dict, List, Optional

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import sp3_printf_wrap as spw

MDR_CAPTURE_DIR = PROJECT_ROOT.parent / "mdr_capture"
DEFAULT_CAP_TOOL = str(MDR_CAPTURE_DIR / "mdr_cap.py")

DEFAULT_SP3_DIR = spw.DEFAULT_SP3_DIR
DEFAULT_MOE_CVT = spw.DEFAULT_MOE_CVT
DEFAULT_FIX_SCRIPT = spw.DEFAULT_FIX_SCRIPT
DEFAULT_KERNEL_SYMBOL = spw.DEFAULT_KERNEL_SYMBOL

CAPTURE_DIRECTIVE_RE = re.compile(r"@CAPTURE\b")


def extract_capture_directive_text(line: str) -> Optional[str]:
    """從一行中提取 @CAPTURE 指令文字。"""
    match = re.search(r"@CAPTURE\b.*", line)
    if not match:
        return None
    return line[match.start():].strip()


def collect_capture_directives(
    lines: List[str],
    line_infos: List[spw.LineInfo],
) -> List[spw.Directive]:
    """
    從 SP3 原始碼中收集 @CAPTURE 指令。

    邏輯與 sp3_printf_wrap.collect_directives 相同，
    但匹配 @CAPTURE 而非 @PRINT/@TIMESTAMP。
    """
    directives: List[spw.Directive] = []
    for idx, line in enumerate(lines):
        if not CAPTURE_DIRECTIVE_RE.search(line):
            continue
        directive_text = extract_capture_directive_text(line)
        if not directive_text:
            continue
        info = line_infos[idx]
        if not info.func:
            print(f"[Warn] @CAPTURE outside any function at line {idx + 1}, skip.")
            continue

        anchor_index = info.instr_index
        insert_after = False
        if anchor_index is None:
            for j in range(idx - 1, -1, -1):
                prev_info = line_infos[j]
                if prev_info.func == info.func and prev_info.instr_index is not None:
                    anchor_index = prev_info.instr_index
                    insert_after = True
                    break
        if anchor_index is None:
            for j in range(idx + 1, len(lines)):
                next_info = line_infos[j]
                if next_info.func == info.func and next_info.instr_index is not None:
                    anchor_index = next_info.instr_index
                    break
        if anchor_index is None:
            print(f"[Warn] Cannot find anchor instruction for @CAPTURE at line {idx + 1}, skip.")
            continue

        anchor_raw_operands = None
        if spw.SP3_MACRO_RE.search(directive_text):
            anchor_line_idx = None
            if insert_after:
                for j in range(idx - 1, -1, -1):
                    pinfo = line_infos[j]
                    if pinfo.func == info.func and pinfo.instr_index == anchor_index:
                        anchor_line_idx = j
                        break
            else:
                if info.instr_index == anchor_index:
                    anchor_line_idx = idx
                else:
                    for j in range(idx + 1, len(lines)):
                        ninfo = line_infos[j]
                        if ninfo.func == info.func and ninfo.instr_index == anchor_index:
                            anchor_line_idx = j
                            break
            if anchor_line_idx is not None:
                anchor_raw_operands = spw.extract_raw_operands(lines[anchor_line_idx])

        directives.append(
            spw.Directive(
                kind="CAPTURE",
                text=directive_text,
                func=info.func,
                anchor_index=anchor_index,
                line_no=idx + 1,
                insert_after=insert_after,
                anchor_raw_operands=anchor_raw_operands,
            )
        )
    return directives


def normalize_capture_comments(input_path: pathlib.Path, output_path: pathlib.Path) -> None:
    """將 // @CAPTURE 轉換為 ; @CAPTURE（LLVM assembly 格式）。"""
    lines = input_path.read_text().splitlines()
    out_lines: List[str] = []
    for line in lines:
        if "@CAPTURE" in line:
            line = re.sub(r"//\s*(?=@CAPTURE)", "; ", line)
        out_lines.append(line)
    output_path.write_text("\n".join(out_lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="SP3 -> .s -> mdr_cap wrapper：從 SP3 提取 @CAPTURE 並生成 register snapshot HSACO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例：
  1. 在 SP3 中標註 @CAPTURE：
     // @CAPTURE f"A={v2}, B={v3}" dst=v10,v11
     // @CAPTURE if $tid == 0: f"A*2={(v2*2.0):.2f}" dst=v12
     // @CAPTURE f"addr={(s[12:13] + s[16:17]*4):ld}" dst=s[12:13]

  2. 執行工具：
     python3 sp3_capture_wrap.py input.sp3 --output-dir cap_output --insert-mode all

  3. 測試生成的 HSACO：
     ./test_bin --hsaco cap_output/mdr_cap_out/<stem>_injected.hsaco

  4. 查看 register 映射：
     cat cap_output/mdr_cap_out/capture_mapping.txt
        """
    )
    ap.add_argument("input_sp3", help="含 @CAPTURE 標記的 SP3 原始檔")
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
    ap.add_argument("--kernel-symbol", default=DEFAULT_KERNEL_SYMBOL, help="symbol 重命名目標")
    ap.add_argument("--asic", default="MI300", help="SP3 asic（預設 MI300）")
    ap.add_argument("--wave-size", default="64", help="SP3 wave_size（預設 64）")
    ap.add_argument("--chip", default="gfx942", help="mdr_cap.py 目標 GPU 架構")
    ap.add_argument("--cap-output-dir", default=None, help="mdr_cap.py 輸出目錄")
    ap.add_argument("--cap-tool", default=DEFAULT_CAP_TOOL, help="mdr_cap.py 路徑")
    ap.add_argument("--skip-cap", action="store_true", help="不執行 mdr_cap.py（僅產生 .s）")
    ap.add_argument("--dry-run", action="store_true", help="傳遞 --dry-run 至 mdr_cap.py")
    ap.add_argument("--keep-disasm", action="store_true", help="保留中間檔案")
    ap.add_argument("--insert-mode", default="all", help=argparse.SUPPRESS)
    ap.add_argument(
        "--insert-all-score-margin", type=int, default=2,
        help="保留 >= (max_score - margin) 的候選位置",
    )
    ap.add_argument(
        "--insert-all-max", type=int, default=20,
        help="每個 directive 最多插入幾個位置 (0=不限)",
    )
    ap.add_argument("--debug-match", action="store_true", help="輸出匹配除錯資訊")
    ap.add_argument("--debug-max-patterns", type=int, default=200)
    ap.add_argument("--debug-max-matches", type=int, default=50)
    args = ap.parse_args()

    input_sp3 = pathlib.Path(args.input_sp3).resolve()
    if not input_sp3.exists():
        raise FileNotFoundError(f"Input SP3 not found: {input_sp3}")

    output_dir = pathlib.Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = args.prefix
    bin_path = output_dir / f"{prefix}.bin"
    disasm_path = output_dir / f"{prefix}_disasm.sp3"
    disasm_cap_path = output_dir / f"{prefix}_disasm_capture.sp3"
    raw_s_path = output_dir / f"{prefix}_raw.s"
    fixed_s_path = output_dir / f"{prefix}_fixed.s"
    fixed_cap_s_path = output_dir / f"{prefix}_fixed_capture.s"

    # ── Parse SP3 @CAPTURE directives ──────────────────────────────────
    print("=== Parse SP3 @CAPTURE directives ===")
    src_lines, line_infos, func_tokens, func_meta = spw.parse_sp3_source(input_sp3)
    directives = collect_capture_directives(src_lines, line_infos)
    spw.expand_thin_functions(src_lines, line_infos, func_tokens, func_meta, directives)

    var_register_map = spw.parse_var_register_aliases(src_lines)
    var_constants = spw.parse_sp3_var_constants(src_lines)
    if var_register_map or var_constants:
        print(f"Found {len(var_register_map)} register alias(es), {len(var_constants)} var constant(s)")
        for d in directives:
            resolved = spw.resolve_symbolic_registers(d.text, var_register_map, var_constants)
            if resolved != d.text:
                print(f"  [Resolve] L{d.line_no}: ...{d.text[-60:]}")
                print(f"         -> ...{resolved[-60:]}")
                d.text = resolved

    print(f"Found {len(directives)} @CAPTURE directive(s)")
    if not directives:
        print("[Error] No @CAPTURE directives found in SP3 source.")
        return 1

    # ── Step 1: SP3 -> Binary ──────────────────────────────────────────
    print("\n=== Step 1: SP3 -> Binary ===")
    sp3_bin = pathlib.Path(args.sp3_dir) / "sp3"
    env = os.environ.copy()
    existing_ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{args.sp3_dir}:{existing_ld}" if existing_ld else args.sp3_dir
    spw.run_cmd(
        [str(sp3_bin), str(input_sp3), f"asic={args.asic}", "-binary", str(bin_path)],
        env=env,
    )

    # ── Step 2: Binary -> Disasm SP3 ───────────────────────────────────
    print("\n=== Step 2: Binary -> Disasm SP3 ===")
    spw.run_cmd(
        [
            str(sp3_bin),
            "-binary", str(bin_path),
            f"asic={args.asic}",
            "type=CS",
            f"wave_size={args.wave_size}",
            str(disasm_path),
        ],
        env=env,
    )

    # ── Step 3: Inject @CAPTURE directives into disasm ─────────────────
    print("\n=== Step 3: Inject @CAPTURE directives into disasm ===")
    disasm_lines, dis_tokens, op_index_to_line, line_to_op_index, dis_meta = spw.parse_disasm(
        disasm_path
    )
    disasm_with_cap = spw.insert_directives_into_disasm(
        disasm_lines,
        dis_tokens,
        dis_meta,
        line_to_op_index,
        op_index_to_line,
        directives,
        func_tokens,
        func_meta,
        insert_mode=args.insert_mode,
        insert_all_score_margin=args.insert_all_score_margin,
        insert_all_max=args.insert_all_max,
        debug_match=args.debug_match,
        debug_max_patterns=args.debug_max_patterns,
        debug_max_matches=args.debug_max_matches,
    )
    disasm_cap_path.write_text("\n".join(disasm_with_cap) + "\n")
    print(f"Annotated disasm: {disasm_cap_path}")

    # ── Step 4: SP3 -> LLVM Assembly ───────────────────────────────────
    print("\n=== Step 4: SP3 -> LLVM Assembly ===")
    spw.run_cmd([
        sys.executable,
        args.moe_cvt,
        str(input_sp3),
        str(disasm_cap_path),
        str(raw_s_path),
    ])

    # ── Step 5: Fix Atomic Add Syntax ──────────────────────────────────
    print("\n=== Step 5: Fix Atomic Add Syntax ===")
    if args.atomic_fix_mode == "long":
        spw.run_cmd([sys.executable, args.fix_script, str(raw_s_path), str(fixed_s_path)])
    else:
        spw.fix_atomic_add_syntax(raw_s_path, fixed_s_path)

    # ── Step 6: Rename Symbol ──────────────────────────────────────────
    print("\n=== Step 6: Rename Symbol ===")
    fixed_text = fixed_s_path.read_text()
    fixed_text = fixed_text.replace("fmoe_kernel_func", args.kernel_symbol)
    fixed_s_path.write_text(fixed_text)

    # ── Step 7: Normalize @CAPTURE comment prefix ──────────────────────
    print("\n=== Step 7: Normalize @CAPTURE comment prefix ===")
    normalize_capture_comments(fixed_s_path, fixed_cap_s_path)
    print(f"Annotated .s: {fixed_cap_s_path}")

    if args.skip_cap:
        print("\n=== Done (skip mdr_cap.py) ===")
        print(f"Annotated .s: {fixed_cap_s_path}")
        return 0

    # ── Step 8: mdr_cap.py ─────────────────────────────────────────────
    cap_output_dir = (
        pathlib.Path(args.cap_output_dir)
        if args.cap_output_dir
        else output_dir / "mdr_cap_out"
    )
    cap_output_dir = cap_output_dir.resolve()
    cap_output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Step 8: mdr_cap.py ===")
    cap_cmd = [
        sys.executable,
        args.cap_tool,
        str(fixed_cap_s_path),
        "--output-dir", str(cap_output_dir),
        "--chip", args.chip,
    ]
    if args.dry_run:
        cap_cmd.append("--dry-run")
    spw.run_cmd(cap_cmd)

    print("\n=== Done ===")
    print(f"Annotated .s:      {fixed_cap_s_path}")
    print(f"mdr_cap output:    {cap_output_dir}")
    print(f"Mapping file:      {cap_output_dir / 'capture_mapping.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
