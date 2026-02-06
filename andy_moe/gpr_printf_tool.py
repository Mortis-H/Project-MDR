#!/usr/bin/env python3
# GPR usage + MLIR printf injection tool.
# Combines:
# - pipeline.py GPR usage parsing (real usage from ISA text)
# - mdr_printf.py @PRINT/@TIMESTAMP parsing + MLIR printf injection pipeline
import argparse
import pathlib
import re
import yaml
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
# mdr_printf.py
sys.path.append(str(PROJECT_ROOT))
# pipeline.py (amdisa-toolkit examples)
sys.path.append(str(PROJECT_ROOT / 'Track_B' / 'amdisa-toolkit' / 'examples'))

import mdr_printf
import pipeline as pipeline_mod


def collect_kernel_names(asm_text: str) -> list[str]:
    names = re.findall(r'\.amdhsa_kernel\s+([\w.$]+)', asm_text)
    if names:
        return names
    globl = re.findall(r'^\s*\.globl\s+([\w.$]+)', asm_text, re.MULTILINE)
    return globl


def report_gpr_usage(asm_text: str) -> None:
    kernel_names = collect_kernel_names(asm_text)
    if not kernel_names:
        kernel_names = ['__all__']

    print('=== GPR Usage (parsed from ISA) ===')
    usage_map = pipeline_mod.compute_gpr_usage_map(asm_text, kernel_names)
    for name, usage in usage_map.items():
        print(
            '- {name}: sgpr={sgpr} (max={sgpr_max}), vgpr={vgpr} (max={vgpr_max}), '
            'agpr={agpr} (max={agpr_max})'.format(
                name=name,
                sgpr=usage['sgpr_count'],
                sgpr_max=usage['sgpr_max'],
                vgpr=usage['vgpr_count'],
                vgpr_max=usage['vgpr_max'],
                agpr=usage['agpr_count'],
                agpr_max=usage['agpr_max'],
            )
        )


def get_kernarg_segment_size(asm_text: str) -> int | None:
    match = re.search(
        r'\.amdgpu_metadata\s+---\s+(.*?)\.\.\.\s+\.end_amdgpu_metadata',
        asm_text,
        re.DOTALL,
    )
    if not match:
        return None
    try:
        meta = yaml.safe_load(match.group(1))
        kernels = meta.get('amdhsa.kernels', [])
        if not kernels:
            return None
        return kernels[0].get('.kernarg_segment_size')
    except Exception:
        return None


def inject_kernarg_padding_arg(gpumlir_text: str, kernarg_size: int | None) -> str:
    if not kernarg_size:
        return gpumlir_text
    # Insert a single padding arg so hidden args are placed after original kernarg bytes.
    pattern = r'(gpu\.func\s+@[\w.$]+\s*)\(\)\s+kernel'
    if re.search(pattern, gpumlir_text):
        return re.sub(
            pattern,
            lambda m: f"{m.group(1)}(%kernarg_pad: !llvm.array<{kernarg_size} x i8>) kernel",
            gpumlir_text,
            count=1,
        )
    return gpumlir_text


def compute_kernarg_size_from_args(args_list: list[dict]) -> int:
    max_end = 0
    for arg in args_list:
        offset = arg.get('.offset')
        size = arg.get('.size', 0)
        if offset is None:
            continue
        end = int(offset) + int(size)
        if end > max_end:
            max_end = end
    if max_end % 8 != 0:
        max_end = ((max_end + 7) // 8) * 8
    return max_end


def merge_hidden_args(fixed_isa: str, generated_isa: str) -> str:
    def _extract_metadata(text: str):
        match = re.search(
            r'\.amdgpu_metadata\s+---\s+(.*?)\.\.\.\s+\.end_amdgpu_metadata',
            text,
            re.DOTALL,
        )
        if not match:
            return None, None, None
        meta = yaml.safe_load(match.group(1))
        return meta, match.start(1), match.end(1)

    fixed_meta, fixed_start, fixed_end = _extract_metadata(fixed_isa)
    gen_meta, _, _ = _extract_metadata(generated_isa)
    if not fixed_meta or not gen_meta:
        return fixed_isa

    fixed_kernels = fixed_meta.get('amdhsa.kernels', [])
    gen_kernels = gen_meta.get('amdhsa.kernels', [])
    if not fixed_kernels or not gen_kernels:
        return fixed_isa

    for f_kernel, g_kernel in zip(fixed_kernels, gen_kernels):
        fixed_args = f_kernel.get('.args', [])
        gen_args = g_kernel.get('.args', [])
        hidden_args = [a for a in gen_args if a.get('.value_kind', '').startswith('hidden_')]
        if not hidden_args:
            continue

        existing_hidden = {a.get('.value_kind') for a in fixed_args if a.get('.value_kind', '').startswith('hidden_')}
        for arg in hidden_args:
            if arg.get('.value_kind') in existing_hidden:
                continue
            fixed_args.append(arg)
        f_kernel['.args'] = fixed_args
        f_kernel['.kernarg_segment_size'] = compute_kernarg_size_from_args(fixed_args)

    fixed_yaml = yaml.dump(fixed_meta, default_flow_style=False, sort_keys=False)
    fixed_isa = fixed_isa[:fixed_start] + fixed_yaml + fixed_isa[fixed_end:]

    kernarg_size = fixed_kernels[0].get('.kernarg_segment_size')
    if kernarg_size:
        fixed_isa = re.sub(
            r'(\.amdhsa_kernarg_size)\s+\d+',
            rf'\1 {kernarg_size}',
            fixed_isa,
        )
    return fixed_isa


def apply_required_gpr_counts(isa_text: str, required_vgpr: int, required_sgpr: int) -> str:
    updated = isa_text
    if required_vgpr > 0:
        vgpr_match = re.search(r'\.amdhsa_next_free_vgpr\s+(\d+)', updated)
        if vgpr_match:
            current_vgpr = int(vgpr_match.group(1))
            if required_vgpr > current_vgpr:
                updated = re.sub(
                    r'(\.amdhsa_next_free_vgpr)\s+\d+',
                    rf'\1 {required_vgpr}',
                    updated,
                )
        updated = re.sub(
            r'(\.vgpr_count:)\s*\d+',
            lambda m: f"{m.group(1)} {required_vgpr}",
            updated,
        )
        accum_match = re.search(r'\.amdhsa_accum_offset\s+(\d+)', updated)
        if accum_match:
            current_accum = int(accum_match.group(1))
            required_accum = ((required_vgpr + 3) // 4) * 4
            if required_accum > 256:
                required_accum = 256
            if required_accum > current_accum:
                updated = re.sub(
                    r'(\.amdhsa_accum_offset)\s+\d+',
                    rf'\1 {required_accum}',
                    updated,
                )
    if required_sgpr > 0:
        sgpr_match = re.search(r'\.amdhsa_next_free_sgpr\s+(\d+)', updated)
        if sgpr_match:
            current_sgpr = int(sgpr_match.group(1))
            if required_sgpr > current_sgpr:
                updated = re.sub(
                    r'(\.amdhsa_next_free_sgpr)\s+\d+',
                    rf'\1 {required_sgpr}',
                    updated,
                )
        updated = re.sub(
            r'(\.sgpr_count:)\s*\d+',
            lambda m: f"{m.group(1)} {required_sgpr}",
            updated,
        )
    return updated


def build_hsaco_with_pipeline(
    kernel_mlir: pathlib.Path,
    chip: str,
    workdir: pathlib.Path,
    output_prefix: str,
    original_isa_file: pathlib.Path,
    has_printf: bool,
    required_vgpr: int,
    required_sgpr: int,
) -> pathlib.Path:
    kernel_binary_mlir = workdir / f"{output_prefix}_binary_isa.mlir"
    kernel_isa_s = workdir / f"{output_prefix}.s"
    kernel_o = workdir / f"{output_prefix}.o"
    kernel_hsaco = workdir / f"{output_prefix}.hsaco"

    pipeline = (
        f"builtin.module("
        f"gpu-kernel-outlining,"
        f"rocdl-attach-target{{chip={chip}}},"
        f"gpu.module(convert-gpu-to-rocdl{{index-bitwidth=32 runtime=HIP}}),"
        f"convert-scf-to-cf,"
        f"convert-cf-to-llvm,"
        f"gpu-to-llvm,"
        f"gpu-module-to-binary{{format=isa}}"
        f")"
    )
    pipeline_mod.run_cmd([
        "mlir-opt",
        str(kernel_mlir),
        f"--pass-pipeline={pipeline}",
        "-o",
        str(kernel_binary_mlir),
    ])

    binary_text = kernel_binary_mlir.read_text()
    isa_list = pipeline_mod.extract_assembly_strings(binary_text)
    if not isa_list:
        raise RuntimeError("No gpu.binary assembly attribute (ISA) found in MLIR output")
    generated_isa = isa_list[0]

    isa = pipeline_mod.fix_isa_metadata(generated_isa, kernel_mlir, original_isa_file)
    if original_isa_file is not None:
        isa = pipeline_mod.append_global_symbols(isa, original_isa_file)
        isa = pipeline_mod.append_device_functions(isa, original_isa_file)

    if has_printf:
        isa = merge_hidden_args(isa, generated_isa)
        isa = apply_required_gpr_counts(isa, required_vgpr, required_sgpr)

    kernel_isa_s.write_text(isa)

    target_triple = "amdgcn--amdhsa"
    target_match = re.search(r'\.amdgcn_target\s+"([^"]+)"', isa)
    if target_match:
        target_id = target_match.group(1)
        target_triple = target_id.split("--", 1)[0]

    pipeline_mod.run_cmd([
        "/opt/rocm/llvm/bin/clang++",
        "-x", "assembler",
        "-target", target_triple,
        f"--offload-arch={chip}",
        "-c",
        str(kernel_isa_s),
        "-o",
        str(kernel_o),
    ])
    pipeline_mod.run_cmd([
        "/opt/rocm/llvm/bin/ld.lld",
        "-shared",
        str(kernel_o),
        "-o",
        str(kernel_hsaco),
    ])
    return kernel_hsaco

def main() -> int:
    ap = argparse.ArgumentParser(
        description='AMD ISA debug tool with GPR usage + printf injection'
    )
    ap.add_argument('input_file', help='輸入的 .s 組合語言檔案')
    ap.add_argument(
        '--output-dir',
        default='debug_output',
        help='輸出目錄（預設：debug_output）',
    )
    ap.add_argument(
        '--chip',
        default='gfx942',
        help='目標 GPU 架構（預設：gfx942）',
    )
    ap.add_argument(
        '--dry-run',
        action='store_true',
        help='只解析 @PRINT 指令，不執行編譯',
    )
    ap.add_argument(
        '--no-printf',
        action='store_true',
        help='禁用 printf 注入（用於純功能驗證）',
    )
    ap.add_argument(
        '--test',
        action='store_true',
        help='使用 universal_hsaco_runner 執行測試',
    )
    ap.add_argument(
        '--test-size',
        type=int,
        default=64,
        help='測試數據大小（預設：64）',
    )
    ap.add_argument(
        '--kernel-name',
        help='Kernel 名稱（可自動偵測）',
    )
    ap.add_argument(
        '--kernel-type',
        help='Kernel 類型：float_add, int_scalar, int_mem, int_cond, int_loop, int_shared 等',
    )
    ap.add_argument(
        '--runner-path',
        help='universal_hsaco_runner 的路徑',
    )
    ap.add_argument(
        '--run',
        action='store_true',
        help='使用 mlir-runner 執行帶有 printf 的 kernel',
    )
    ap.add_argument(
        '--rocm-runtime-lib',
        help='libmlir_rocm_runtime.so 路徑（用於 --run）',
    )
    ap.add_argument(
        '--runner-utils-lib',
        help='libmlir_runner_utils.so 路徑（用於 --run）',
    )
    args = ap.parse_args()

    input_path = pathlib.Path(args.input_file).resolve()
    if not input_path.exists():
        raise FileNotFoundError('Input file not found: {}'.format(input_path))

    workdir = pathlib.Path(args.output_dir)
    workdir.mkdir(parents=True, exist_ok=True)

    asm_text = input_path.read_text()
    kernarg_size = get_kernarg_segment_size(asm_text)

    # 0. GPR usage report (ISA-based)
    report_gpr_usage(asm_text)

    # 1. 自動偵測 kernel 資訊
    auto_kernel_name, auto_kernel_type = mdr_printf.detect_kernel_info(asm_text)
    kernel_name = args.kernel_name or auto_kernel_name
    kernel_type = args.kernel_type or auto_kernel_type

    print('\n=== AMD ISA Debug Tool ===')
    print('Input: {}'.format(input_path))
    print('Output: {}'.format(workdir))
    print('Chip: {}'.format(args.chip))
    if kernel_name:
        print('Kernel Name: {}'.format(kernel_name))
    if kernel_type:
        print('Kernel Type: {}'.format(kernel_type))

    # 2. 解析 @PRINT / @TIMESTAMP
    print('\n=== Parsing @PRINT and @TIMESTAMP directives ===')
    try:
        _, directives, has_barrier, timestamp_directives = mdr_printf.parse_asm_file(input_path)
    except ValueError as e:
        print('\n❌ ERROR: {}'.format(e))
        print('\n   Please check your directive syntax.')
        return 1

    if not directives and not timestamp_directives:
        if args.no_printf:
            print('[Info] No @PRINT/@TIMESTAMP directives found (--no-printf mode)')
        else:
            print('[Info] No @PRINT/@TIMESTAMP directives found. Will generate HSACO without printf.')
    else:
        if directives:
            print('\nFound {} @PRINT directive(s)'.format(len(directives)))
        if timestamp_directives:
            print('Found {} @TIMESTAMP directive(s)'.format(len(timestamp_directives)))

    if has_barrier and (directives or timestamp_directives) and not args.no_printf:
        print('\n' + '=' * 60)
        print('⚠️  WARNING: Kernel contains s_barrier instruction!')
        print('   gpu.printf hostcall mechanism may conflict with barrier')
        print('   synchronization, causing kernel to hang or crash.')
        print('')
        print('   Recommendations:')
        print('   1. Use --no-printf for functional verification')
        print('   2. Use cond=REG_eq(N) to limit printf (e.g., v6_eq(0.0))')
        print('   3. Place @PRINT only after all barriers complete')
        print('=' * 60 + '\n')

    if args.dry_run:
        print('\n[Dry Run] Stopping here.')
        return 0

    # 3. 分析暫存器使用量
    print('\n=== Analyzing register usage ===')
    reg_info = mdr_printf.analyze_registers(asm_text)
    print('  VGPR: {}, SGPR: {}, AGPR: {}'.format(reg_info['vgpr'], reg_info['sgpr'], reg_info['agpr']))

    # 4. 轉換為 GPU MLIR（使用 pipeline.py 的流程）
    _, gpumlir_path = pipeline_mod.translate_asm_to_gpu(input_path, workdir, output_prefix=input_path.stem)

    # 5. 注入 printf/timestamp
    has_printf = (bool(directives) or bool(timestamp_directives)) and not args.no_printf
    required_vgpr = 0
    required_sgpr = 0

    if (not directives and not timestamp_directives) or args.no_printf:
        print('\nUsing original GPU MLIR (no printf injection)')
        modified_path = gpumlir_path
    else:
        print('\n=== Injecting printf/timestamp code ===')
        gpumlir_text = gpumlir_path.read_text()
        if kernarg_size:
            print('[Info] Padding kernarg segment to {} bytes for hidden args'.format(kernarg_size))
            gpumlir_text = inject_kernarg_padding_arg(gpumlir_text, kernarg_size)
        modified_mlir, required_vgpr, required_sgpr = mdr_printf.inject_printf_into_mlir(
            gpumlir_text, directives, reg_info, timestamp_directives
        )
        modified_path = workdir / '{}_debug_injected.gpumlir'.format(input_path.stem)
        modified_path.write_text(modified_mlir)
        print('Generated modified GPU MLIR: {}'.format(modified_path))

    # 6. 可選：使用 mlir-runner 執行
    if args.run:
        if args.no_printf:
            print('\n[Warning] --run 與 --no-printf 一起使用意義不大')
        host_mlir_path = mdr_printf.generate_host_wrapper_mlir(modified_path, args.test_size, workdir, directives)
        mdr_printf.run_with_mlir_runner(
            host_mlir_path,
            args.chip,
            workdir,
            args.rocm_runtime_lib,
            args.runner_utils_lib,
        )
        return 0

    # 7. 生成 HSACO（沿用 pipeline.py 的流程）
    output_prefix = "{}_debug_injected".format(input_path.stem) if has_printf else "{}_debug".format(input_path.stem)
    hsaco_path = build_hsaco_with_pipeline(
        modified_path,
        args.chip,
        workdir,
        output_prefix,
        input_path,
        has_printf,
        required_vgpr,
        required_sgpr,
    )

    print('\n=== Done ===')
    print('Debug HSACO: {}'.format(hsaco_path))

    # 8. 可選：使用 universal_hsaco_runner
    if args.test:
        if not kernel_name:
            print('\n[Error] Cannot run test: kernel name not detected.')
            print('        Please specify with --kernel-name')
            return 1
        if not kernel_type:
            print('\n[Error] Cannot run test: kernel type not detected.')
            print('        Please specify with --kernel-type')
            return 1

        runner_path = args.runner_path
        if not runner_path:
            project_root = pathlib.Path(__file__).parent
            possible_paths = [
                project_root / 'Track_B' / 'kernel_testcases' / 'universal_hsaco_runner',
                project_root / 'universal_hsaco_runner',
                pathlib.Path('universal_hsaco_runner'),
            ]
            for p in possible_paths:
                if p.exists():
                    runner_path = str(p)
                    break

        if not runner_path or not pathlib.Path(runner_path).exists():
            print('\n[Error] universal_hsaco_runner not found.')
            print('        Please compile it or specify path with --runner-path')
            return 1

        print('\n=== Running test with universal_hsaco_runner ===')
        cmd = [
            runner_path,
            str(hsaco_path),
            kernel_name,
            kernel_type,
            str(args.test_size),
        ]
        mdr_printf.run_cmd(cmd)
    else:
        if kernel_name and kernel_type:
            print('\n[Info] 可以使用 universal_hsaco_runner 測試此 HSACO:')
            print(
                '       Track_B/kernel_testcases/universal_hsaco_runner {} {} {} {}'.format(
                    hsaco_path, kernel_name, kernel_type, args.test_size
                )
            )

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
