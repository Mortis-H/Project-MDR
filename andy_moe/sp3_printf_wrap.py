#!/usr/bin/env python3
"""
Wrapper: keep @PRINT/@TIMESTAMP annotations from SP3, re-inject into .s,
then call gpr_printf_tool.py to generate HSACO.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import os
import pathlib
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent

DEFAULT_SP3_DIR = "../../poc_kl/mi300/fused_moe_asm/mi300_sp3_to_asm"
DEFAULT_MOE_CVT = "../..//poc_kl/scripts/fused_moe/moe_cvt.py"
DEFAULT_FIX_SCRIPT = "../..//poc_kl/mi300/fix_atomic_add.py"
DEFAULT_KERNEL_SYMBOL = "_ZN5aiter45fmoe_bf16_pertokenFp8_g1u1_vs_silu_1tg_32x192E"
DEFAULT_GPR_TOOL = str(PROJECT_ROOT / "gpr_printf_tool.py")

DIRECTIVE_RE = re.compile(r"@PRINT\b|@TIMESTAMP_START\b|@TIMESTAMP_END\b")
LABEL_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_\.\[\]]*):")
REGISTER_RANGE_RE = re.compile(r"^([a-z]+)\[\d+:\d+\]$")
REGISTER_RE = re.compile(r"^([sv])\d+$")
ACC_RE = re.compile(r"^acc\d+$")
AREG_RE = re.compile(r"^a\d+$")
IMM_RE = re.compile(r"^-?(0x[0-9a-fA-F]+|\d+)$")
FLOAT_RE = re.compile(r"^-?\d*\.\d+([eE][-+]?\d+)?$")
FUNC_CALL_RE = re.compile(r"^([A-Za-z_]\w*)\((.*)\)$")
LABEL_TOKEN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_\.\[\]]*$")
IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
VAR_REGISTER_RE = re.compile(
    r'^\s*var\s+(\w+)\s*=\s*([svSV]\d+|acc\d+|[aA]\d+)\s*(?://.*)?$'
)


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
    insert_after: bool = False
    anchor_raw_operands: Optional[List[str]] = None  # raw operands from SP3 anchor instruction


# Regex to match SP3 macro references in @PRINT text:
# - Full form: v_regs(_v_X_addr, i_idx) or s_regs(_s_X_buf, 0)
# - Bare form: v_regs or s_regs (used as variable name in format string {v_regs:d})
SP3_MACRO_RE = re.compile(r'(?:[vs]_regs|acc_regs)(?:\s*\([^)]*\))?')
# Regex to find bare v_regs/s_regs inside format string placeholders like {v_regs:d}
SP3_BARE_MACRO_IN_FMT_RE = re.compile(r'\{(v_regs|s_regs|acc_regs)(:[^}]*)?\}')


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


def extract_label(line: str) -> Optional[str]:
    no_comment = strip_inline_comment(line)
    match = LABEL_RE.match(no_comment)
    if not match:
        return None
    return match.group(1)


def normalize_operand(token: str) -> Optional[str]:
    tok = token.strip()
    if not tok:
        return None
    tok = tok.replace(" ", "")
    lower_tok = tok.lower()
    if lower_tok.startswith(("row_", "bank_", "format:", "cbid:", "dmask:", "swz:")):
        return None
    range_match = REGISTER_RANGE_RE.match(tok)
    if range_match:
        return range_match.group(1)
    reg_match = REGISTER_RE.match(tok)
    if reg_match:
        return reg_match.group(1)
    if ACC_RE.match(tok):
        return "acc"
    if AREG_RE.match(tok):
        return "a"
    if IMM_RE.match(tok) or FLOAT_RE.match(tok):
        return "imm"
    if tok.startswith("inst_offset:"):
        return "inst_offset"
    if tok.startswith("offset:"):
        return "offset"
    call_match = FUNC_CALL_RE.match(tok)
    if call_match:
        func = call_match.group(1).lower()
        if func in ("v_regs", "v_reg", "vreg"):
            return "v"
        if func in ("s_regs", "s_reg", "sreg"):
            return "s"
        if func in ("acc_regs", "acc_reg", "accreg"):
            return "acc"
        if func in ("a_regs", "a_reg", "areg"):
            return "a"
        inner = call_match.group(2)
        inner_norm = normalize_operand(inner) if inner else ""
        return f"{func}({inner_norm})" if inner_norm else func
    if LABEL_TOKEN_RE.match(tok):
        if tok.startswith(("s_", "S_")):
            return "s"
        if tok.startswith(("v_", "V_")):
            return "v"
        if tok.startswith(("acc_", "ACC_")):
            return "acc"
        if tok.startswith(("a_", "A_")):
            return "a"
        if tok.startswith(("Vsrc", "vsrc")):
            return "imm"
        if tok.isupper():
            return "imm"
        if tok.startswith(("label", "lbl", "L")):
            return "label"
        if IDENT_RE.match(tok):
            return "imm"
    return tok.lower()


def extract_raw_operands(line: str) -> List[str]:
    """Extract raw (un-normalized) operands from an instruction line."""
    no_comment = strip_inline_comment(line)
    stripped = no_comment.strip()
    if not stripped:
        return []
    parts = stripped.split(None, 1)
    if len(parts) < 2:
        return []
    return split_operands(parts[1])


_REG_RANGE_SIMPLIFY_RE = re.compile(r'^([a-z]+)\[(\d+):\d+\]$')


def _simplify_reg_range(operand: str) -> str:
    """Convert register range like s[28:31] to s28, acc[0:1] to a0."""
    m = _REG_RANGE_SIMPLIFY_RE.match(operand)
    if m:
        prefix = m.group(1)
        if prefix == 'acc':
            prefix = 'a'
        return f"{prefix}{m.group(2)}"
    return operand


def resolve_sp3_macros(directive_text: str,
                       sp3_raw_operands: Optional[List[str]],
                       disasm_line: str) -> str:
    """Resolve v_regs(...)/s_regs(...) references in a directive by mapping
    SP3 operand positions to the matched disasm instruction's operands.

    Supports two forms:
      1. Full form in directive text: v_regs(_v_X_addr, i_idx) -> v26
      2. Bare form in format string: {v_regs:d} -> {v26:d}
         (resolves to the first v_regs(...) operand found in the anchor)

    For example:
      SP3 instruction: buffer_load_dword v0, v_regs(_v_X_addr,i_idx), s_regs(_s_X_buf,0), 0 lds:1 offen:1
      Disasm instruction: buffer_load_dword v26, s[20:23], 0 offen lds
      Directive text:  f"X_mem_load v_regs={v_regs:d}"
      Result:          f"X_mem_load v_regs={v26:d}"
    """
    if not sp3_raw_operands or not SP3_MACRO_RE.search(directive_text):
        return directive_text

    disasm_raw = extract_raw_operands(disasm_line)
    if not disasm_raw:
        return directive_text

    # Build mapping: SP3 macro operands -> concrete disasm registers.
    #
    # Strategy: align SP3 and disasm operands by position. When the number
    # of operands differs (e.g., lds drops vdst in some ISA formats), use
    # normalized-type matching within each type group, preserving order.
    # When counts match, use direct positional mapping.
    sp3_norm = [(op, normalize_operand(op)) for op in sp3_raw_operands]
    dis_norm = [(op, normalize_operand(op)) for op in disasm_raw]

    macro_map: Dict[str, str] = {}        # full macro text -> concrete
    bare_macro_map: Dict[str, str] = {}   # 'v_regs' or 's_regs' -> first concrete

    if len(sp3_norm) == len(dis_norm):
        # Same operand count -> direct positional mapping
        for (sp3_op, sp3_n), (dis_op, dis_n) in zip(sp3_norm, dis_norm):
            clean_op = sp3_op.strip().replace(' ', '')
            call_match = FUNC_CALL_RE.match(clean_op)
            if call_match:
                resolved = _simplify_reg_range(dis_op.strip())
                macro_map[clean_op] = resolved
                func_name = call_match.group(1).lower()
                if func_name in ('v_regs', 'v_reg', 'vreg',
                                 's_regs', 's_reg', 'sreg',
                                 'acc_regs', 'acc_reg', 'accreg'):
                    if func_name.startswith('acc'):
                        base_name = 'acc_regs'
                    elif func_name.startswith('v'):
                        base_name = 'v_regs'
                    else:
                        base_name = 's_regs'
                    if base_name not in bare_macro_map:
                        bare_macro_map[base_name] = resolved
    else:
        # Different operand count (e.g., lds drops vdst in ISA disasm).
        # Match within each normalized-type group, only counting macro ops.
        # Group disasm operands by normalized type
        dis_by_norm: Dict[str, List[str]] = {}
        for raw_op, norm in dis_norm:
            if norm:
                dis_by_norm.setdefault(norm, []).append(raw_op.strip())

        # For each SP3 macro operand, find its match within the type group
        macro_idx_by_norm: Dict[str, int] = {}
        for raw_op, norm in sp3_norm:
            if norm is None:
                continue
            clean_op = raw_op.strip().replace(' ', '')
            call_match = FUNC_CALL_RE.match(clean_op)
            if call_match:
                idx = macro_idx_by_norm.get(norm, 0)
                macro_idx_by_norm[norm] = idx + 1
                candidates = dis_by_norm.get(norm, [])
                if idx < len(candidates):
                    concrete = _simplify_reg_range(candidates[idx])
                    macro_map[clean_op] = concrete
                    func_name = call_match.group(1).lower()
                    if func_name in ('v_regs', 'v_reg', 'vreg',
                                     's_regs', 's_reg', 'sreg',
                                     'acc_regs', 'acc_reg', 'accreg'):
                        if func_name.startswith('acc'):
                            base_name = 'acc_regs'
                        elif func_name.startswith('v'):
                            base_name = 'v_regs'
                        else:
                            base_name = 's_regs'
                        if base_name not in bare_macro_map:
                            bare_macro_map[base_name] = concrete

    result = directive_text

    # 1. Replace full-form macro calls (e.g., v_regs(_v_X_addr, i_idx) -> v26)
    for macro, concrete in macro_map.items():
        func_match = FUNC_CALL_RE.match(macro)
        if func_match:
            func_name = func_match.group(1)
            func_args = func_match.group(2)
            escaped_name = re.escape(func_name)
            args_parts = [re.escape(a.strip()) for a in func_args.split(',')]
            args_pattern = r'\s*,\s*'.join(args_parts)
            pattern = escaped_name + r'\s*\(\s*' + args_pattern + r'\s*\)'
            result = re.sub(pattern, concrete, result)

    # 2. Replace bare-form macros in format strings (e.g., {v_regs:d} -> {v26:d})
    if bare_macro_map:
        def _replace_bare(m):
            macro_name = m.group(1)  # 'v_regs' or 's_regs'
            fmt_spec = m.group(2) or ''  # ':d' or '' etc.
            concrete = bare_macro_map.get(macro_name)
            if concrete:
                return '{' + concrete + fmt_spec + '}'
            return m.group(0)
        result = SP3_BARE_MACRO_IN_FMT_RE.sub(_replace_bare, result)

    # 3. Replace bare v_regs/s_regs in plain text (outside {}) for readability
    #    e.g., f"X_mem_load v_regs={v26:d}" -> f"X_mem_load v26={v26:d}"
    if bare_macro_map:
        for macro_name, concrete in bare_macro_map.items():
            # Only replace occurrences NOT inside { } (already handled above)
            # Match: word-boundary v_regs/s_regs NOT preceded by { or followed by :}
            result = re.sub(
                r'(?<!\{)\b' + re.escape(macro_name) + r'\b(?![:\}])',
                concrete, result)

    return result


def split_operands(text: str) -> List[str]:
    parts: List[str] = []
    buf: List[str] = []
    depth = 0
    for ch in text:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth = max(depth - 1, 0)
        if ch == "," and depth == 0:
            part = "".join(buf).strip()
            if part:
                parts.append(strip_operand_modifier(part))
            buf = []
            continue
        buf.append(ch)
    if buf:
        part = "".join(buf).strip()
        if part:
            parts.append(strip_operand_modifier(part))
    return parts


def strip_operand_modifier(text: str) -> str:
    depth = 0
    for idx, ch in enumerate(text):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth = max(depth - 1, 0)
        elif ch.isspace() and depth == 0:
            return text[:idx].strip()
    return text.strip()


def tokenize_instruction(line: str) -> Optional[str]:
    no_comment = strip_inline_comment(line)
    stripped = no_comment.strip()
    if not stripped or stripped.endswith(":"):
        return None
    parts = stripped.split(None, 1)
    if not parts:
        return None
    opcode = parts[0].lower()
    if "(" in opcode:
        return None
    operands_part = parts[1] if len(parts) > 1 else ""
    if not operands_part:
        return opcode
    modifiers = re.findall(r"\b(row_[A-Za-z0-9_]+):", operands_part)
    modifiers += re.findall(r"\b(bank_[A-Za-z0-9_]+):", operands_part)
    lower_operands = operands_part.lower()
    for flag in ("lds", "offen"):
        if re.search(rf"\b{flag}\b", lower_operands):
            modifiers.append(flag)
    operands = []
    for raw in split_operands(operands_part):
        norm = normalize_operand(raw)
        if norm:
            operands.append(norm)
    if not operands:
        return opcode
    token = f"{opcode} " + ",".join(operands)
    if modifiers:
        unique_mods = "+".join(sorted(set(modifiers)))
        token = f"{token}|{unique_mods}"
    return token


def hash_tokens(tokens: List[str]) -> str:
    data = "\n".join(tokens).encode("utf-8")
    return hashlib.sha1(data).hexdigest()[:12]


def build_block_meta(
    tokens: List[str],
    labels: List[Optional[str]],
) -> Tuple[List[str], List[int], List[int]]:
    block_ids: List[str] = []
    current = "__entry__"
    for label in labels:
        if label:
            current = label
        block_ids.append(current)

    block_tokens: Dict[str, List[str]] = {}
    for tok, bid in zip(tokens, block_ids):
        block_tokens.setdefault(bid, []).append(tok)

    block_hash = {bid: hash_tokens(seq) for bid, seq in block_tokens.items()}

    block_pos: List[int] = []
    block_len: List[int] = []
    pos_counter: Dict[str, int] = {}
    for bid in block_ids:
        pos = pos_counter.get(bid, 0)
        block_pos.append(pos)
        pos_counter[bid] = pos + 1
        block_len.append(len(block_tokens[bid]))

    block_hashes = [block_hash[bid] for bid in block_ids]
    return block_hashes, block_pos, block_len


def current_function(stack: List[Tuple[str, str]]) -> Optional[str]:
    for block_type, name in reversed(stack):
        if block_type in ("function", "shader"):
            return name
    return None


def expand_thin_functions(
    lines: List[str],
    line_infos: List[LineInfo],
    func_tokens: Dict[str, List[str]],
    func_meta: Dict[str, Dict[str, List]],
    directives: List["Directive"],
    min_tokens_threshold: int = 10,
) -> None:
    """Inline called-function tokens for functions with few direct instructions.

    When a function body is mostly function calls (e.g. cl_actv_reshape_D_load),
    the token list is too short for reliable pattern matching.  By inlining the
    called functions' tokens we provide enough surrounding context to
    distinguish otherwise-identical instruction pairs (e.g. two s_barrier).
    Modifies *func_tokens*, *func_meta* and directive anchor indices in place.
    """
    thin_funcs = set()
    for d in directives:
        if d.func in func_tokens and len(func_tokens[d.func]) < min_tokens_threshold:
            thin_funcs.add(d.func)
    if not thin_funcs:
        return

    for func_name in thin_funcs:
        original_tokens = list(func_tokens[func_name])
        expanded: List[str] = []
        anchor_remap: Dict[int, int] = {}

        for idx, line in enumerate(lines):
            info = line_infos[idx]
            if info.func != func_name:
                continue

            if info.opcode and info.instr_index is not None:
                anchor_remap[info.instr_index] = len(expanded)
                expanded.append(original_tokens[info.instr_index])
            elif info.opcode and info.instr_index is None:
                called = info.opcode.split("(")[0]
                if called in func_tokens and called != func_name:
                    expanded.extend(func_tokens[called])

        if len(expanded) <= len(original_tokens):
            continue

        func_tokens[func_name] = expanded

        labels_list: List[Optional[str]] = [None] * len(expanded)
        bh, bp, bl = build_block_meta(expanded, labels_list)
        func_meta[func_name] = {
            "labels": labels_list,
            "block_hashes": bh,
            "block_pos": bp,
            "block_len": bl,
        }

        for d in directives:
            if d.func == func_name and d.anchor_index in anchor_remap:
                d.anchor_index = anchor_remap[d.anchor_index]

        print(
            f"[Info] Expanded '{func_name}': "
            f"{len(original_tokens)} -> {len(expanded)} tokens (inlined)"
        )


def parse_var_register_aliases(lines: List[str]) -> Dict[str, str]:
    """Parse 'var NAME = sN/vN/accN' declarations to build symbolic -> physical register map."""
    var_map: Dict[str, str] = {}
    for line in lines:
        m = VAR_REGISTER_RE.match(line)
        if m:
            var_map[m.group(1)] = m.group(2).lower()
    return var_map


def resolve_symbolic_registers(text: str, var_map: Dict[str, str]) -> str:
    """Replace symbolic register names in {name:fmt} format placeholders with physical registers.

    Example: var_map = {"s_log2e": "s6"} turns '{s_log2e:x}' into '{s6:x}'.
    """
    if not var_map:
        return text

    def _replace(m: re.Match) -> str:
        name = m.group(1)
        if name in var_map:
            fmt = m.group(2) or ''
            return '{' + var_map[name] + fmt + '}'
        return m.group(0)

    return re.sub(r'\{(\w+)(:[^}]*)?\}', _replace, text)


def parse_sp3_source(
    path: pathlib.Path,
) -> Tuple[
    List[str],
    List[LineInfo],
    Dict[str, List[str]],
    Dict[str, Dict[str, List]],
]:
    lines = path.read_text().splitlines()
    stack: List[Tuple[str, str]] = []
    func_tokens: Dict[str, List[str]] = {}
    func_labels: Dict[str, List[Optional[str]]] = {}
    func_instr_counts: Dict[str, int] = {}
    line_infos: List[LineInfo] = []
    current_label_by_func: Dict[str, Optional[str]] = {}

    func_start_re = re.compile(r"^\s*function\s+([A-Za-z0-9_]+)")

    for line in lines:
        stripped = line.strip()

        func_match = func_start_re.match(stripped)
        if func_match:
            func_name = func_match.group(1)
            stack.append(("function", func_name))
            current_label_by_func[func_name] = None
            line_infos.append(LineInfo(current_function(stack), None, None))
            continue

        if stripped.startswith("shader main"):
            stack.append(("shader", "__shader_main__"))
            current_label_by_func["__shader_main__"] = None
            line_infos.append(LineInfo(current_function(stack), None, None))
            continue

        if re.match(r"^\s*(if|for|while)\b", stripped):
            stack.append(("block", ""))

        func = current_function(stack)
        label = extract_label(line)
        if label and func:
            current_label_by_func[func] = label

        opcode = extract_opcode(line)
        instr_index = None
        token = tokenize_instruction(line) if opcode else None
        if opcode and func and token:
            func_tokens.setdefault(func, [])
            func_labels.setdefault(func, [])
            func_instr_counts.setdefault(func, 0)
            instr_index = func_instr_counts[func]
            func_instr_counts[func] += 1
            func_tokens[func].append(token)
            func_labels[func].append(current_label_by_func.get(func))

        line_infos.append(LineInfo(func, opcode, instr_index))

        if re.match(r"^\s*end\b", stripped):
            if stack:
                stack.pop()

    func_meta: Dict[str, Dict[str, List]] = {}
    for func, tokens in func_tokens.items():
        labels = func_labels.get(func, [None] * len(tokens))
        block_hashes, block_pos, block_len = build_block_meta(tokens, labels)
        func_meta[func] = {
            "labels": labels,
            "block_hashes": block_hashes,
            "block_pos": block_pos,
            "block_len": block_len,
        }

    return lines, line_infos, func_tokens, func_meta


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
            print(f"[Warn] Cannot find anchor instruction for directive at line {idx + 1}, skip.")
            continue
        kind = "PRINT" if "@PRINT" in directive_text else "TIMESTAMP"

        # If directive contains v_regs/s_regs macros, extract the anchor
        # instruction's raw operands for later resolution.
        anchor_raw_operands = None
        if SP3_MACRO_RE.search(directive_text):
            # Find the anchor instruction line
            anchor_line_idx = None
            if insert_after:
                # anchor is before the directive
                for j in range(idx - 1, -1, -1):
                    pinfo = line_infos[j]
                    if pinfo.func == info.func and pinfo.instr_index == anchor_index:
                        anchor_line_idx = j
                        break
            else:
                # anchor is on the same line or after
                if info.instr_index == anchor_index:
                    anchor_line_idx = idx
                else:
                    for j in range(idx + 1, len(lines)):
                        ninfo = line_infos[j]
                        if ninfo.func == info.func and ninfo.instr_index == anchor_index:
                            anchor_line_idx = j
                            break
            if anchor_line_idx is not None:
                anchor_raw_operands = extract_raw_operands(lines[anchor_line_idx])

        directives.append(
            Directive(
                kind=kind,
                text=directive_text,
                func=info.func,
                anchor_index=anchor_index,
                line_no=idx + 1,
                insert_after=insert_after,
                anchor_raw_operands=anchor_raw_operands,
            )
        )
    return directives


def parse_disasm(
    path: pathlib.Path,
) -> Tuple[
    List[str],
    List[str],
    Dict[int, int],
    Dict[int, int],
    Dict[str, List],
]:
    lines = path.read_text().splitlines()
    tokens: List[str] = []
    labels: List[Optional[str]] = []
    op_index_to_line: Dict[int, int] = {}
    line_to_op_index: Dict[int, int] = {}
    current_label: Optional[str] = None
    for idx, line in enumerate(lines):
        label = extract_label(line)
        if label:
            current_label = label
            continue
        token = tokenize_instruction(line)
        if token:
            op_index = len(tokens)
            tokens.append(token)
            labels.append(current_label)
            op_index_to_line[op_index] = idx
            line_to_op_index[idx] = op_index

    block_hashes, block_pos, block_len = build_block_meta(tokens, labels)
    dis_meta = {
        "labels": labels,
        "block_hashes": block_hashes,
        "block_pos": block_pos,
        "block_len": block_len,
    }
    return lines, tokens, op_index_to_line, line_to_op_index, dis_meta


def build_index_map(src_ops: List[str], dst_ops: List[str]) -> Dict[int, int]:
    matcher = difflib.SequenceMatcher(a=src_ops, b=dst_ops, autojunk=False)
    mapping: Dict[int, int] = {}
    for a0, b0, size in matcher.get_matching_blocks():
        for i in range(size):
            mapping[a0 + i] = b0 + i
    return mapping


def find_pattern_matches(dis_tokens: List[str], pattern: List[str]) -> List[int]:
    if not pattern:
        return []
    matches: List[int] = []
    plen = len(pattern)
    for i in range(len(dis_tokens) - plen + 1):
        if dis_tokens[i : i + plen] == pattern:
            matches.append(i)
    return matches


def build_pattern(func_tokens: List[str], idx: int, window: int) -> Tuple[List[str], int]:
    start = max(0, idx - window)
    end = min(len(func_tokens), idx + window + 1)
    return func_tokens[start:end], idx - start


def collect_candidates(
    func_tokens: List[str],
    func_meta: Dict[str, List],
    dis_tokens: List[str],
    dis_meta: Dict[str, List],
    anchor_index: int,
    windows: Tuple[int, ...] = (5, 4, 3, 2, 1, 0),
    anchor_scan: int = 12,
    debug: bool = False,
    debug_max_patterns: int = 200,
    debug_max_matches: int = 50,
    debug_disasm_lines: Optional[List[str]] = None,
    debug_op_index_to_line: Optional[Dict[int, int]] = None,
) -> Dict[int, int]:
    if anchor_index < 0 or anchor_index >= len(func_tokens):
        return {}

    cand_scores: Dict[int, int] = {}
    func_labels = func_meta.get("labels", [])
    func_block_hashes = func_meta.get("block_hashes", [])
    func_block_pos = func_meta.get("block_pos", [])
    func_block_len = func_meta.get("block_len", [])

    dis_labels = dis_meta.get("labels", [])
    dis_block_pos = dis_meta.get("block_pos", [])
    dis_block_len = dis_meta.get("block_len", [])
    dis_block_pos = dis_meta.get("block_pos", [])
    dis_block_len = dis_meta.get("block_len", [])
    dis_block_hashes = dis_meta.get("block_hashes", [])
    dis_block_pos = dis_meta.get("block_pos", [])
    dis_block_len = dis_meta.get("block_len", [])

    debug_patterns = 0
    debug_match_budget = debug_max_matches
    for shift in range(anchor_scan + 1):
        idx = anchor_index + shift
        if idx >= len(func_tokens):
            break
        for window in windows:
            pattern, offset = build_pattern(func_tokens, idx, window)
            if not pattern:
                continue
            match_starts = find_pattern_matches(dis_tokens, pattern)
            if not match_starts:
                continue
            if debug and (debug_max_patterns <= 0 or debug_patterns < debug_max_patterns):
                debug_patterns += 1
                pattern_text = " | ".join(pattern)
                print(
                    f"[DEBUG] shift={shift} window={window} pattern_len={len(pattern)} "
                    f"offset={offset} matches={len(match_starts)}"
                )
                print(f"[DEBUG] pattern: {pattern_text}")
            for start in match_starts:
                dis_idx = start + offset
                base_score = window * 5 - shift * 2
                score = base_score
                hash_match = (
                    func_block_hashes
                    and dis_block_hashes
                    and func_block_hashes[idx] == dis_block_hashes[dis_idx]
                )
                if hash_match:
                    score += 4
                label_match = (
                    func_labels
                    and dis_labels
                    and func_labels[idx]
                    and func_labels[idx] == dis_labels[dis_idx]
                )
                if label_match:
                    score += 2
                block_match = (
                    func_block_len
                    and dis_block_len
                    and func_block_len[idx] == dis_block_len[dis_idx]
                    and func_block_pos
                    and dis_block_pos
                    and abs(func_block_pos[idx] - dis_block_pos[dis_idx]) <= 1
                )
                if block_match:
                    score += 1
                if debug and (debug_max_matches <= 0 or debug_match_budget > 0):
                    # only print up to debug_max_matches total if > 0
                    if debug_max_matches > 0:
                        debug_match_budget -= 1
                    line_text = ""
                    if debug_disasm_lines is not None and debug_op_index_to_line is not None:
                        line_idx = debug_op_index_to_line.get(dis_idx)
                        if line_idx is not None and line_idx < len(debug_disasm_lines):
                            line_text = debug_disasm_lines[line_idx]
                    calc = (
                        f"base={base_score} "
                        f"+hash({4 if hash_match else 0}) "
                        f"+label({2 if label_match else 0}) "
                        f"+block({1 if block_match else 0})"
                    )
                    print(
                        "[DEBUG]  match dis_idx="
                        f"{dis_idx} score={score} {calc} "
                        f"token={dis_tokens[dis_idx]}"
                    )
                    if line_text:
                        print(f"[DEBUG]   disasm: {line_text}")
                prev = cand_scores.get(dis_idx)
                if prev is None or score > prev:
                    cand_scores[dis_idx] = score
    return cand_scores


def select_positions(
    directives: List[Directive],
    candidates: List[Dict[int, int]],
    max_candidates: int = 200,
) -> List[Optional[int]]:
    if not directives:
        return []

    cand_lists: List[List[Tuple[int, int]]] = []
    for cand in candidates:
        if not cand:
            cand_lists.append([])
            continue
        ranked = sorted(cand.items(), key=lambda x: (-x[1], x[0]))[:max_candidates]
        cand_lists.append(sorted(ranked, key=lambda x: x[0]))

    # DP for monotonic increasing positions
    trace_list: List[Dict[int, Optional[int]]] = []
    prev_scores: Dict[int, int] = {}
    success = True
    for i, cand_list in enumerate(cand_lists):
        curr_scores: Dict[int, int] = {}
        curr_trace: Dict[int, Optional[int]] = {}
        if not cand_list:
            trace_list.append(curr_trace)
            prev_scores = {}
            success = False
            continue
        if i == 0 or not prev_scores:
            for pos, score in cand_list:
                curr_scores[pos] = score
                curr_trace[pos] = None
        else:
            prev_items = sorted(prev_scores.items(), key=lambda x: x[0])
            best_score = None
            best_pos = None
            idx_prev = 0
            for pos, score in cand_list:
                while idx_prev < len(prev_items) and prev_items[idx_prev][0] < pos:
                    ppos, pscore = prev_items[idx_prev]
                    if best_score is None or pscore > best_score:
                        best_score = pscore
                        best_pos = ppos
                    idx_prev += 1
                if best_score is None:
                    continue
                curr_scores[pos] = best_score + score
                curr_trace[pos] = best_pos
            if not curr_scores:
                success = False
                for pos, score in cand_list:
                    curr_scores[pos] = score
                    curr_trace[pos] = None
        trace_list.append(curr_trace)
        prev_scores = curr_scores

    if not prev_scores:
        return [None for _ in directives]

    if not success:
        result: List[Optional[int]] = []
        for cand_list in cand_lists:
            if not cand_list:
                result.append(None)
            else:
                result.append(max(cand_list, key=lambda x: x[1])[0])
        return result

    best_end = max(prev_scores.items(), key=lambda x: x[1])[0]
    positions: List[Optional[int]] = [None for _ in directives]
    for i in range(len(directives) - 1, -1, -1):
        positions[i] = best_end
        best_end = trace_list[i].get(best_end)
        if best_end is None and i > 0:
            break
    return positions


def insert_directives_into_disasm(
    disasm_lines: List[str],
    dis_tokens: List[str],
    dis_meta: Dict[str, List],
    line_to_op_index: Dict[int, int],
    op_index_to_line: Dict[int, int],
    directives: List[Directive],
    func_tokens: Dict[str, List[str]],
    func_meta: Dict[str, Dict[str, List]],
    insert_mode: str = "best",
    insert_all_score_margin: int = 2,
    insert_all_max: int = 20,
    debug_match: bool = False,
    debug_max_patterns: int = 200,
    debug_max_matches: int = 50,
) -> List[str]:
    insertions: Dict[int, List[Tuple[str, bool]]] = {}
    if not directives:
        return disasm_lines

    directives_by_func: Dict[str, List[Directive]] = {}
    for d in directives:
        directives_by_func.setdefault(d.func, []).append(d)

    main_tokens = func_tokens.get("__shader_main__")
    main_map = build_index_map(main_tokens, dis_tokens) if main_tokens else {}

    dis_labels = dis_meta.get("labels", [])
    dis_block_pos = dis_meta.get("block_pos", [])
    dis_block_len = dis_meta.get("block_len", [])

    for func, dirs in directives_by_func.items():
        tokens = func_tokens.get(func)
        if not tokens:
            print(f"[Warn] Missing function token list for {func}, skip directives.")
            continue
        meta = func_meta.get(func, {})
        func_map = build_index_map(tokens, dis_tokens)

        dir_entries = list(enumerate(dirs))
        dir_entries.sort(key=lambda x: (x[1].anchor_index, x[1].line_no))
        sorted_dirs = [d for _, d in dir_entries]

        cand_scores_list: List[Dict[int, int]] = []
        for d in sorted_dirs:
            if debug_match:
                anchor_token = (
                    tokens[d.anchor_index] if 0 <= d.anchor_index < len(tokens) else "n/a"
                )
                print(
                    f"[DEBUG] directive line={d.line_no} func={d.func} "
                    f"anchor_index={d.anchor_index} anchor_token={anchor_token}"
                )
            cand_scores = collect_candidates(
                tokens,
                meta,
                dis_tokens,
                dis_meta,
                d.anchor_index,
                debug=debug_match,
                debug_max_patterns=debug_max_patterns,
                debug_max_matches=debug_max_matches,
                debug_disasm_lines=disasm_lines,
                debug_op_index_to_line=op_index_to_line,
            )
            if func_map:
                for shift in range(13):
                    idx = d.anchor_index + shift
                    if idx >= len(tokens):
                        break
                    mapped = func_map.get(idx)
                    if mapped is not None:
                        base_score = 30 - shift
                        prev_score = cand_scores.get(mapped, 0)
                        cand_scores[mapped] = max(prev_score, base_score)
                        if debug_match:
                            print(
                                "[DEBUG] func_map boost "
                                f"src_idx={idx} shift={shift} dis_idx={mapped} "
                                f"base=30-shift={base_score} prev={prev_score} "
                                f"score={cand_scores[mapped]}"
                            )
            if func == "__shader_main__":
                mapped = main_map.get(d.anchor_index)
                if mapped is not None:
                    cand_scores[mapped] = max(cand_scores.get(mapped, 0), 40)
                    if debug_match:
                        print(
                            "[DEBUG] main_map boost "
                            f"src_idx={d.anchor_index} dis_idx={mapped} score={cand_scores[mapped]}"
                        )
            if insert_mode == "all" and cand_scores:
                if 0 <= d.anchor_index < len(tokens):
                    anchor_token = tokens[d.anchor_index]
                    filtered = {
                        idx: score
                        for idx, score in cand_scores.items()
                        if idx < len(dis_tokens) and dis_tokens[idx] == anchor_token
                    }
                    if filtered:
                        cand_scores = filtered
            if cand_scores and insert_mode == "best":
                max_score = max(cand_scores.values())
                if max_score >= 28:
                    cand_scores = {
                        idx: score
                        for idx, score in cand_scores.items()
                        if score >= max_score - 2
                    }
            if debug_match and cand_scores:
                top = sorted(cand_scores.items(), key=lambda x: (-x[1], x[0]))[:10]
                top_text = ", ".join(f"{idx}:{score}" for idx, score in top)
                print(f"[DEBUG] cand_scores top: {top_text}")
            cand_scores_list.append(cand_scores)
            if not cand_scores:
                print(f"[Warn] Cannot map directive (line {d.line_no}) from {d.func}.")
            elif len(cand_scores) > 200:
                print(
                    f"[Warn] Directive at line {d.line_no} matched {len(cand_scores)} candidates; "
                    "placement may be ambiguous."
                )

        if insert_mode == "all":
            for d, cand_scores in zip(sorted_dirs, cand_scores_list):
                if not cand_scores:
                    continue
                candidates = list(cand_scores.items())
                if not candidates:
                    continue
                max_score = max(score for _, score in candidates)

                if d.func == "__shader_main__":
                    best_idx = max(candidates, key=lambda x: (x[1], -x[0]))[0]
                    insertions.setdefault(best_idx, []).append(
                        (d.text, d.insert_after, d.anchor_raw_operands))
                    continue

                base_candidates = [
                    (idx, score)
                    for idx, score in candidates
                    if score >= max_score - insert_all_score_margin
                ]
                if not base_candidates:
                    continue

                top_idx = max(base_candidates, key=lambda x: (x[1], -x[0]))[0]
                anchor_token = (
                    tokens[d.anchor_index]
                    if 0 <= d.anchor_index < len(tokens)
                    else None
                )

                expanded = {idx for idx, _ in base_candidates}

                if anchor_token:
                    best_ctx_positions: Optional[set] = None
                    best_ctx_count = 0
                    for ctx_w in range(6, 0, -1):
                        ctx_start = max(0, top_idx - ctx_w)
                        ctx_end = min(len(dis_tokens), top_idx + ctx_w + 1)
                        ctx_pattern = dis_tokens[ctx_start:ctx_end]
                        ctx_offset = top_idx - ctx_start
                        ctx_matches = find_pattern_matches(dis_tokens, ctx_pattern)
                        if not ctx_matches:
                            continue
                        new_positions = set()
                        for m_start in ctx_matches:
                            p = m_start + ctx_offset
                            if 0 <= p < len(dis_tokens) and dis_tokens[p] == anchor_token:
                                new_positions.add(p)
                        if not new_positions:
                            continue
                        count = len(new_positions)
                        if insert_all_max > 0 and count > insert_all_max:
                            continue
                        if count > best_ctx_count:
                            best_ctx_count = count
                            best_ctx_positions = new_positions
                    if best_ctx_positions:
                        expanded |= best_ctx_positions

                expanded_list = sorted(expanded)
                if insert_all_max and insert_all_max > 0:
                    expanded_list = expanded_list[:insert_all_max]
                if not expanded_list:
                    print(
                        f"[Warn] No candidates after filtering for directive (line {d.line_no}) from {d.func}."
                    )
                    continue
                for dis_idx in expanded_list:
                    insertions.setdefault(dis_idx, []).append(
                        (d.text, d.insert_after, d.anchor_raw_operands))
        else:
            positions = select_positions(sorted_dirs, cand_scores_list)
            for d, dis_idx in zip(sorted_dirs, positions):
                if dis_idx is None:
                    print(f"[Warn] Cannot resolve directive (line {d.line_no}) from {d.func}.")
                    continue
                insertions.setdefault(dis_idx, []).append(
                    (d.text, d.insert_after, d.anchor_raw_operands))

    new_lines: List[str] = []
    for line_idx, line in enumerate(disasm_lines):
        op_index = line_to_op_index.get(line_idx)
        if op_index is not None and op_index in insertions:
            indent = re.match(r"^(\s*)", line).group(1)
            for text, insert_after, anchor_ops in insertions[op_index]:
                if not insert_after:
                    resolved = resolve_sp3_macros(text, anchor_ops, line)
                    new_lines.append(f"{indent}// {resolved}")
            new_lines.append(line)
            for text, insert_after, anchor_ops in insertions[op_index]:
                if insert_after:
                    resolved = resolve_sp3_macros(text, anchor_ops, line)
                    new_lines.append(f"{indent}// {resolved}")
        else:
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
    ap.add_argument(
        "--insert-mode",
        choices=("best", "all"),
        default="best",
        help="PRINT/TIMESTAMP 插入模式：best(選最佳位置) 或 all(插入多個候選位置)",
    )
    ap.add_argument(
        "--insert-all-score-margin",
        type=int,
        default=2,
        help="insert-mode=all 時，保留 >= (max_score - margin) 的候選",
    )
    ap.add_argument(
        "--insert-all-max",
        type=int,
        default=20,
        help="insert-mode=all 時，每個 directive 最多插入幾個位置 (0=不限)",
    )
    ap.add_argument(
        "--debug-match",
        action="store_true",
        help="輸出 pattern/shift/anchor 的匹配除錯資訊",
    )
    ap.add_argument(
        "--debug-max-patterns",
        type=int,
        default=200,
        help="debug 模式最多列印幾個 pattern (0=不限)",
    )
    ap.add_argument(
        "--debug-max-matches",
        type=int,
        default=50,
        help="debug 模式最多列印幾個 match (0=不限)",
    )
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
    src_lines, line_infos, func_tokens, func_meta = parse_sp3_source(input_sp3)
    directives = collect_directives(src_lines, line_infos)
    expand_thin_functions(src_lines, line_infos, func_tokens, func_meta, directives)

    var_register_map = parse_var_register_aliases(src_lines)
    if var_register_map:
        print(f"Found {len(var_register_map)} register alias(es)")
        for d in directives:
            resolved = resolve_symbolic_registers(d.text, var_register_map)
            if resolved != d.text:
                print(f"  [Resolve] L{d.line_no}: ...{d.text[-60:]}")
                print(f"         -> ...{resolved[-60:]}")
                d.text = resolved

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
    disasm_lines, dis_tokens, op_index_to_line, line_to_op_index, dis_meta = parse_disasm(disasm_path)
    if directives:
        disasm_with_print = insert_directives_into_disasm(
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
