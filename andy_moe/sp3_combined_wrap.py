#!/usr/bin/env python3
"""
SP3 Combined Wrapper (@PRINT + @CAPTURE)
=========================================

同時處理 SP3 中的 @PRINT 與 @CAPTURE 標記，一次生成包含
printf 輸出 + capture 計算的合併 HSACO。

Pipeline:
  1. 解析 SP3：收集 @PRINT 和 @CAPTURE 指令
  2. SP3 -> Binary
  3. Binary -> Disasm SP3
  4. 將 @PRINT + @CAPTURE 統一注入反組譯結果
  5-7. SP3 -> .s（moe_cvt, fix_atomic, rename, normalize）
  8. mdr_cap.py --inject-only（注入 capture ISA，保留 @PRINT 註解）
  9. gpr_printf_tool.py（注入 printf ISA + 生成 HSACO）

使用方式：
  python3 sp3_combined_wrap.py input.sp3 --output-dir combined_out

  SP3 中：
    // @PRINT if $tgid_x == 0 && $tid == 0: f"val={v128:x}"
    // @CAPTURE f"fma={(v128*v129+v130)}" dst=v58
    // @PRINT if $tgid_x == 0 && $tid == 0: f"result={v58:x}"
"""
from __future__ import annotations

import argparse
import os
import pathlib
import re
import sys
from typing import List

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import sp3_printf_wrap as spw
import sp3_capture_wrap as scw

MDR_CAPTURE_DIR = PROJECT_ROOT.parent / "mdr_capture"
DEFAULT_CAP_TOOL = str(MDR_CAPTURE_DIR / "mdr_cap.py")
DEFAULT_GPR_TOOL = str(PROJECT_ROOT / "gpr_printf_tool.py")

DEFAULT_SP3_DIR = spw.DEFAULT_SP3_DIR
DEFAULT_MOE_CVT = spw.DEFAULT_MOE_CVT
DEFAULT_FIX_SCRIPT = spw.DEFAULT_FIX_SCRIPT
DEFAULT_KERNEL_SYMBOL = spw.DEFAULT_KERNEL_SYMBOL


def normalize_all_directive_comments(input_path: pathlib.Path, output_path: pathlib.Path) -> None:
    """將 // @PRINT, // @TIMESTAMP_*, // @CAPTURE 統一轉為 ; 前綴。"""
    lines = input_path.read_text().splitlines()
    out_lines: List[str] = []
    directive_re = re.compile(r"//\s*(?=@(?:PRINT|CAPTURE|TIMESTAMP_))")
    for line in lines:
        out_lines.append(directive_re.sub("; ", line))
    output_path.write_text("\n".join(out_lines) + "\n")


def reorder_prints_after_capture_isa(injected_s_path: pathlib.Path) -> int:
    """
    將引用 capture 目標暫存器的 @PRINT 移到 capture ISA 之後。

    mdr_cap.py 注入 capture ISA 時會留下 '; Expression: ... → vXX' 標記。
    如果有 @PRINT 引用了該 vXX 但位置在 capture ISA 之前，
    則將該 @PRINT 移到 capture ISA block 結束之後，
    確保 printf snapshot 拿到的是計算後的值。
    """
    lines = injected_s_path.read_text().splitlines()
    expr_re = re.compile(r";\s*Expression:\s*.+→\s*(v\d+)")

    capture_blocks: List[dict] = []
    i = 0
    while i < len(lines):
        m = expr_re.search(lines[i])
        if m:
            dst_reg = m.group(1)
            block_end = i + 1
            while block_end < len(lines):
                stripped = lines[block_end].strip()
                if stripped.startswith(("v_", "s_")) and dst_reg in stripped:
                    block_end += 1
                else:
                    break
            capture_blocks.append({"start": i, "end": block_end, "dst": dst_reg})
            i = block_end
        else:
            i += 1

    moved = 0
    prev_block_end = 0
    for block in capture_blocks:
        dst = block["dst"]
        to_remove_indices: List[int] = []
        to_move_lines: List[str] = []

        search_floor = max(0, prev_block_end)
        for j in range(block["start"] - 1, search_floor - 1, -1):
            stripped = lines[j].strip()
            if stripped.startswith("; @PRINT") and f"{{{dst}:" in stripped:
                to_remove_indices.append(j)
                to_move_lines.insert(0, lines[j])

        if to_move_lines:
            for idx in sorted(to_remove_indices, reverse=True):
                lines.pop(idx)

            n_removed = len(to_remove_indices)
            insert_pos = block["end"] - n_removed

            for i, line_content in enumerate(to_move_lines):
                lines.insert(insert_pos + i, line_content)

            moved += n_removed

        prev_block_end = block["end"]

    if moved:
        injected_s_path.write_text("\n".join(lines) + "\n")
    return moved


def main() -> int:
    ap = argparse.ArgumentParser(
        description="SP3 -> HSACO：同時處理 @PRINT + @CAPTURE，生成合併的 HSACO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例：
  python3 sp3_combined_wrap.py input.sp3 --output-dir combined_out
  python3 sp3_combined_wrap.py input.sp3 --output-dir combined_out --insert-mode all

  SP3 中標記範例（支援 SP3 符號名稱 _v_NAME[offset]）：
    // @PRINT if $tid == 0: f"Z0={_v_Z[0]:f} Z1={_v_Z[1]:f}"
    // @CAPTURE f"prod={(_v_Z[0]*_v_Z[1])}" dst=_v_tmp[0]
    // @PRINT if $tid == 0: f"result={_v_tmp[0]:f}"
        """,
    )
    ap.add_argument("input_sp3", help="含 @PRINT/@CAPTURE 的 SP3 原始檔")
    ap.add_argument("--output-dir", required=True, help="輸出目錄")
    ap.add_argument("--prefix", default="kernel", help="輸出檔名前綴（預設：kernel）")
    ap.add_argument("--sp3-dir", default=DEFAULT_SP3_DIR, help="SP3 compiler 目錄")
    ap.add_argument("--moe-cvt", default=DEFAULT_MOE_CVT, help="moe_cvt.py 路徑")
    ap.add_argument("--fix-script", default=DEFAULT_FIX_SCRIPT, help="fix_atomic_add.py 路徑")
    ap.add_argument(
        "--atomic-fix-mode",
        choices=("syntax", "long"),
        default="syntax",
        help="atomic add 修復模式",
    )
    ap.add_argument("--kernel-symbol", default=DEFAULT_KERNEL_SYMBOL, help="symbol 重命名目標")
    ap.add_argument("--asic", default="MI300", help="SP3 asic")
    ap.add_argument("--wave-size", default="64", help="SP3 wave_size")
    ap.add_argument("--chip", default="gfx942", help="目標 GPU 架構")

    ap.add_argument("--cap-tool", default=DEFAULT_CAP_TOOL, help="mdr_cap.py 路徑")
    ap.add_argument("--gpr-tool", default=DEFAULT_GPR_TOOL, help="gpr_printf_tool.py 路徑")
    ap.add_argument("--dry-run", action="store_true", help="只解析，不執行編譯")
    ap.add_argument("--keep-disasm", action="store_true", help="保留中間檔案")

    ap.add_argument(
        "--insert-mode",
        choices=("best", "all"),
        default="best",
        help="指令插入模式：best(選最佳位置) 或 all(插入所有匹配)",
    )
    ap.add_argument("--insert-all-score-margin", type=int, default=2)
    ap.add_argument("--insert-all-max", type=int, default=20)
    ap.add_argument("--debug-match", action="store_true")
    ap.add_argument("--debug-max-patterns", type=int, default=200)
    ap.add_argument("--debug-max-matches", type=int, default=50)

    args, gpr_extra_args = ap.parse_known_args()

    input_sp3 = pathlib.Path(args.input_sp3).resolve()
    if not input_sp3.exists():
        raise FileNotFoundError(f"Input SP3 not found: {input_sp3}")

    output_dir = pathlib.Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = args.prefix
    bin_path = output_dir / f"{prefix}.bin"
    disasm_path = output_dir / f"{prefix}_disasm.sp3"
    disasm_combined_path = output_dir / f"{prefix}_disasm_combined.sp3"
    raw_s_path = output_dir / f"{prefix}_raw.s"
    fixed_s_path = output_dir / f"{prefix}_fixed.s"
    fixed_combined_s_path = output_dir / f"{prefix}_fixed_combined.s"

    # ── Parse SP3: collect BOTH @PRINT and @CAPTURE ─────────────────────
    print("=" * 60)
    print("SP3 Combined Wrapper (@PRINT + @CAPTURE)")
    print("=" * 60)

    print("\n=== Parsing SP3 directives ===")
    src_lines, line_infos, func_tokens, func_meta = spw.parse_sp3_source(input_sp3)

    capture_directives = scw.collect_capture_directives(src_lines, line_infos)
    print_directives = spw.collect_directives(src_lines, line_infos)

    spw.expand_thin_functions(src_lines, line_infos, func_tokens, func_meta, capture_directives)
    spw.expand_thin_functions(src_lines, line_infos, func_tokens, func_meta, print_directives)

    var_register_map = spw.parse_var_register_aliases(src_lines)
    var_constants = spw.parse_sp3_var_constants(src_lines)
    if var_register_map or var_constants:
        print(f"Found {len(var_register_map)} register alias(es), {len(var_constants)} var constant(s)")
        for d in capture_directives + print_directives:
            resolved = spw.resolve_symbolic_registers(d.text, var_register_map, var_constants)
            if resolved != d.text:
                print(f"  [Resolve] L{d.line_no}: ...{d.text[-60:]}")
                print(f"         -> ...{resolved[-60:]}")
                d.text = resolved

    has_capture = len(capture_directives) > 0
    has_print = len(print_directives) > 0

    print(f"\n  @CAPTURE directives: {len(capture_directives)}")
    print(f"  @PRINT   directives: {len(print_directives)}")

    if not has_capture and not has_print:
        print("[Error] No @CAPTURE or @PRINT directives found.")
        return 1

    all_directives = capture_directives + print_directives
    all_directives.sort(key=lambda d: d.line_no)

    # ── Step 1: SP3 -> Binary ───────────────────────────────────────────
    print("\n=== Step 1: SP3 -> Binary ===")
    sp3_bin = pathlib.Path(args.sp3_dir) / "sp3"
    env = os.environ.copy()
    existing_ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{args.sp3_dir}:{existing_ld}" if existing_ld else args.sp3_dir
    spw.run_cmd(
        [str(sp3_bin), str(input_sp3), f"asic={args.asic}", "-binary", str(bin_path)],
        env=env,
    )

    # ── Step 2: Binary -> Disasm SP3 ────────────────────────────────────
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

    # ── Step 3: Inject ALL directives into disasm ───────────────────────
    print("\n=== Step 3: Inject @PRINT + @CAPTURE into disasm ===")
    disasm_lines, dis_tokens, op_index_to_line, line_to_op_index, dis_meta = spw.parse_disasm(
        disasm_path
    )
    disasm_with_all = spw.insert_directives_into_disasm(
        disasm_lines,
        dis_tokens,
        dis_meta,
        line_to_op_index,
        op_index_to_line,
        all_directives,
        func_tokens,
        func_meta,
        insert_mode=args.insert_mode,
        insert_all_score_margin=args.insert_all_score_margin,
        insert_all_max=args.insert_all_max,
        debug_match=args.debug_match,
        debug_max_patterns=args.debug_max_patterns,
        debug_max_matches=args.debug_max_matches,
    )
    disasm_combined_path.write_text("\n".join(disasm_with_all) + "\n")
    print(f"Annotated disasm: {disasm_combined_path}")

    # ── Step 4: SP3 -> LLVM Assembly ────────────────────────────────────
    print("\n=== Step 4: SP3 -> LLVM Assembly ===")
    spw.run_cmd([
        sys.executable,
        args.moe_cvt,
        str(input_sp3),
        str(disasm_combined_path),
        str(raw_s_path),
    ])

    # ── Step 5: Fix Atomic Add Syntax ───────────────────────────────────
    print("\n=== Step 5: Fix Atomic Add Syntax ===")
    if args.atomic_fix_mode == "long":
        spw.run_cmd([sys.executable, args.fix_script, str(raw_s_path), str(fixed_s_path)])
    else:
        spw.fix_atomic_add_syntax(raw_s_path, fixed_s_path)

    # ── Step 6: Rename Symbol ───────────────────────────────────────────
    print("\n=== Step 6: Rename Symbol ===")
    fixed_text = fixed_s_path.read_text()
    fixed_text = fixed_text.replace("fmoe_kernel_func", args.kernel_symbol)
    fixed_s_path.write_text(fixed_text)

    # ── Step 7: Normalize ALL directive comment prefixes ────────────────
    print("\n=== Step 7: Normalize directive comments (// -> ;) ===")
    normalize_all_directive_comments(fixed_s_path, fixed_combined_s_path)
    print(f"Annotated .s: {fixed_combined_s_path}")

    if args.dry_run:
        print(f"\n[Dry Run] Stopping here.")
        print(f"  .s with @PRINT + @CAPTURE: {fixed_combined_s_path}")
        return 0

    # ── Determine pipeline based on available directives ────────────────
    gpr_input_s = fixed_combined_s_path

    if has_capture:
        # Step 8: mdr_cap.py --inject-only → capture ISA 注入，@PRINT 保留
        cap_output_dir = (output_dir / "mdr_cap_out").resolve()
        cap_output_dir.mkdir(parents=True, exist_ok=True)

        inject_only = has_print
        print(f"\n=== Step 8: mdr_cap.py {'(inject-only)' if inject_only else ''} ===")
        cap_cmd = [
            sys.executable,
            args.cap_tool,
            str(fixed_combined_s_path),
            "--output-dir", str(cap_output_dir),
            "--chip", args.chip,
        ]
        if inject_only:
            cap_cmd.append("--inject-only")
        spw.run_cmd(cap_cmd)

        if inject_only:
            gpr_input_s = cap_output_dir / f"{fixed_combined_s_path.stem}_injected.s"
            if not gpr_input_s.exists():
                print(f"[Error] Expected injected .s not found: {gpr_input_s}")
                return 1
            print(f"  Capture ISA injected: {gpr_input_s}")

            n_moved = reorder_prints_after_capture_isa(gpr_input_s)
            if n_moved:
                print(f"  Reordered {n_moved} @PRINT(s) to after capture ISA")

    if has_print:
        # Step 9: gpr_printf_tool.py → printf ISA 注入 + HSACO 生成
        gpr_output_dir = (output_dir / "gpr_printf_out").resolve()
        gpr_output_dir.mkdir(parents=True, exist_ok=True)

        step_label = "Step 9" if has_capture else "Step 8"
        print(f"\n=== {step_label}: gpr_printf_tool.py ===")
        gpr_cmd = [
            sys.executable,
            args.gpr_tool,
            str(gpr_input_s),
            "--output-dir", str(gpr_output_dir),
            "--chip", args.chip,
        ] + gpr_extra_args
        spw.run_cmd(gpr_cmd)

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Combined Pipeline Done")
    print("=" * 60)
    print(f"  Input SP3:      {input_sp3.name}")
    print(f"  @CAPTURE:       {len(capture_directives)} directive(s)")
    print(f"  @PRINT:         {len(print_directives)} directive(s)")
    print(f"  Annotated .s:   {fixed_combined_s_path}")

    if has_capture:
        print(f"  Capture output: {cap_output_dir}")
        if has_print:
            print(f"  Injected .s:    {gpr_input_s}")

    if has_print:
        hsaco_stem = f"{gpr_input_s.stem}_debug_injected"
        hsaco_path = gpr_output_dir / f"{hsaco_stem}.hsaco"
        print(f"  Printf output:  {gpr_output_dir}")
        print(f"  Final HSACO:    {hsaco_path}")
    elif has_capture and not has_print:
        hsaco_path = cap_output_dir / f"{fixed_combined_s_path.stem}_clobber.hsaco"
        print(f"  Final HSACO:    {hsaco_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
