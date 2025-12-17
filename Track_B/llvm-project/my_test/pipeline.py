#!/usr/bin/env python3
import argparse
import pathlib
import re
import shutil
import subprocess
import sys
import textwrap


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


def translate_asm_to_gpu(asm_file: pathlib.Path, workdir: pathlib.Path):
    """
    使用 amdisa-translate 將 .s 文件轉換為 .amdisamlir 和 .gpumlir
    
    Args:
        asm_file: 輸入的 .s assembly 文件
        workdir: 工作目錄
    
    Returns:
        (amdisamlir_path, gpumlir_path): 生成的文件路徑
    """
    ensure_tool("amdisa-translate")
    
    asm_stem = asm_file.stem
    amdisamlir_file = workdir / f"{asm_stem}.amdisamlir"
    gpumlir_file = workdir / f"{asm_stem}.gpumlir"
    
    # Step 1: .s -> .amdisamlir
    print(f"\n=== Step 1: Translating {asm_file.name} to AMDISA MLIR ===")
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
    print(f"\n=== Step 2: Lowering AMDISA MLIR to GPU MLIR ===")
    gpu_cmd = [
        "amdisa-translate",
        "-x", "mlir",
        "-emit=gpu",
        str(amdisamlir_file),
    ]
    result = subprocess.run(gpu_cmd, capture_output=True, text=True, check=True)
    gpumlir_file.write_text(result.stdout)
    print(f"Generated GPU MLIR: {gpumlir_file}")
    
    return amdisamlir_file, gpumlir_file


def build_isa_and_hsaco(kernel_mlir: pathlib.Path, chip: str, workdir: pathlib.Path):
    """
    從 GPU MLIR 繼續執行原本的 pipeline，生成 ISA 和 HSACO
    
    Args:
        kernel_mlir: GPU MLIR 文件 (帶有 gpu.func)
        chip: 目標晶片型號
        workdir: 工作目錄
    """
    for tool in ["mlir-opt", "llvm-mc", "ld.lld"]:
        ensure_tool(tool)

    kernel_stem = kernel_mlir.stem

    kernel_binary_mlir = workdir / f"{kernel_stem}_binary_isa.mlir"
    kernel_isa_s         = workdir / f"{kernel_stem}.s"
    kernel_o             = workdir / f"{kernel_stem}.o"
    kernel_hsaco         = workdir / f"{kernel_stem}.hsaco"

    print(f"\n=== Step 3: Running MLIR optimization pipeline ===")
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
    kernel_isa_s.write_text(isa)
    print(f"Wrote ISA assembly to {kernel_isa_s}")

    print(f"\n=== Step 4: Assembling ISA to object file ===")
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

    print(f"\n=== Step 5: Linking to HSACO ===")
    ld_cmd = [
        "ld.lld",
        "-shared",
        str(kernel_o),
        "-o",
        str(kernel_hsaco),
    ]
    run_cmd(ld_cmd)

    print(f"\n✓ Successfully generated HSACO: {kernel_hsaco}")


def build_and_run_host(kernel_mlir: pathlib.Path,
                       chip: str,
                       workdir: pathlib.Path,
                       rocm_runtime_lib: str | None,
                       runner_utils_lib: str | None):
    """
    降階 host + device 並使用 mlir-runner 執行。
    
    Pipeline（分為兩個 mlir-opt 調用，然後是 mlir-runner）：
    
      1) mlir-opt -pass-pipeline="
           builtin.module(
             gpu.module(strip-debuginfo,
                        convert-gpu-to-rocdl{index-bitwidth=32 runtime=HIP}),
             rocdl-attach-target{chip=<chip>}
           )"
      
      2) mlir-opt -gpu-to-llvm -reconcile-unrealized-casts -gpu-module-to-binary
      
      3) mlir-runner --shared-libs=... \
                     --shared-libs=... \
                     --entry-point-result=void
    """
    for tool in ["mlir-opt", "mlir-runner"]:
        ensure_tool(tool)
    
    # 嘗試自動偵測庫（如果沒有提供）
    auto_rocm_rt, auto_runner_utils = auto_detect_mlir_libs()
    
    rocm_runtime_lib_path = pathlib.Path(
        rocm_runtime_lib if rocm_runtime_lib is not None else (
            str(auto_rocm_rt) if auto_rocm_rt is not None else ""
        )
    )
    runner_utils_lib_path = pathlib.Path(
        runner_utils_lib if runner_utils_lib is not None else (
            str(auto_runner_utils) if auto_runner_utils is not None else ""
        )
    )
    
    if not rocm_runtime_lib_path.is_file() or not runner_utils_lib_path.is_file():
        raise RuntimeError(
            "無法找到 libmlir_rocm_runtime.so 或 libmlir_runner_utils.so。\n"
            "請透過以下方式明確傳遞它們：\n"
            "  --rocm-runtime-lib /path/to/libmlir_rocm_runtime.so\n"
            "  --runner-utils-lib /path/to/libmlir_runner_utils.so\n"
            "或確保使用標準 LLVM 構建布局，以便可以自動偵測。"
        )
    
    print(f"使用 ROCm runtime lib: {rocm_runtime_lib_path}")
    print(f"使用 runner utils lib: {runner_utils_lib_path}")
    
    kernel_stem = kernel_mlir.stem
    host_step1_mlir = workdir / f"{kernel_stem}_host_step1.mlir"
    host_final_mlir = workdir / f"{kernel_stem}_host_final.mlir"
    
    # Step 1: gpu.module(strip-debuginfo, convert-gpu-to-rocdl{...}), rocdl-attach-target
    print(f"\n=== Step: Lowering GPU module to ROCDL ===")
    pipeline1 = (
        f"builtin.module("
        f"gpu.module(strip-debuginfo,convert-gpu-to-rocdl{{index-bitwidth=32 runtime=HIP}}),"
        f"rocdl-attach-target{{chip={chip}}}"
        f")"
    )
    
    mlir_opt_cmd1 = [
        "mlir-opt",
        str(kernel_mlir),
        f"--pass-pipeline={pipeline1}",
        "-o",
        str(host_step1_mlir),
    ]
    run_cmd(mlir_opt_cmd1)
    
    # Step 2: -gpu-to-llvm -reconcile-unrealized-casts -gpu-module-to-binary
    print(f"\n=== Step: Converting to LLVM and creating binary ===")
    mlir_opt_cmd2 = [
        "mlir-opt",
        str(host_step1_mlir),
        "-gpu-to-llvm",
        "-reconcile-unrealized-casts",
        "-gpu-module-to-binary",
        "-o",
        str(host_final_mlir),
    ]
    run_cmd(mlir_opt_cmd2)
    
    # Step 3: mlir-runner
    print(f"\n=== Step: Running kernel via mlir-runner ===")
    mlir_runner_cmd = [
        "mlir-runner",
        str(host_final_mlir),
        f"--shared-libs={rocm_runtime_lib_path}",
        f"--shared-libs={runner_utils_lib_path}",
        "--entry-point-result=void",
    ]
    run_cmd(mlir_runner_cmd)
    
    print(f"\n✓ Kernel execution completed")


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
        default=None,
        help="directory to put intermediate files [default: <kernel_mlir_stem>]",
    )
    ap.add_argument(
        "--emit-isa",
        action="store_true",
        default=None,
        help="run device-only pipeline with gpu-module-to-binary{format=isa} and build HSACO",
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
        "--run-host",
        action="store_true",
        help="lower host+device and run @main via mlir-runner",
    )
    ap.add_argument(
        "--rocm-runtime-lib",
        help="path to libmlir_rocm_runtime.so (optional; will be auto-detected if possible)",
    )
    ap.add_argument(
        "--runner-utils-lib",
        help="path to libmlir_runner_utils.so (optional; will be auto-detected if possible)",
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
    
    # 智能決定是否需要 emit_isa
    # 如果用戶沒有明確指定 --emit-isa 或 --no-emit-isa
    if args.emit_isa is None:
        if args.run_host:
            # 使用 --run-host 時，預設不生成 HSACO（避免重複編譯）
            args.emit_isa = False
            print("[INFO] --run-host 已啟用，自動停用 HSACO 生成以避免重複編譯")
            print("[INFO] 如需同時生成 HSACO，請明確使用 --emit-isa --run-host")
        else:
            # 不執行時，預設生成 HSACO
            args.emit_isa = True

    # 根據文件副檔名決定處理流程
    suffix = input_file.suffix.lower()
    
    # 用於記錄最終生成的 GPU MLIR 文件（用於後續的 run-host）
    final_gpu_mlir = None
    
    if suffix == ".s":
        # 完整流程：.s -> .amdisamlir -> .gpumlir -> ISA/HSACO
        print(f"=== Processing AMD ISA Assembly: {input_file.name} ===")
        amdisamlir_file, gpumlir_file = translate_asm_to_gpu(input_file, workdir)
        final_gpu_mlir = gpumlir_file
        
        if args.emit_isa:
            build_isa_and_hsaco(gpumlir_file, args.chip, workdir)
    
    elif suffix in [".mlir", ".gpumlir"]:
        # 從 GPU MLIR 開始（原本的流程）
        print(f"=== Processing GPU MLIR: {input_file.name} ===")
        final_gpu_mlir = input_file
        
        if args.emit_isa:
            build_isa_and_hsaco(input_file, args.chip, workdir)
    
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Expected .s or .mlir")
    
    # 如果指定了 --run-host，則執行 kernel
    if args.run_host:
        if final_gpu_mlir is None:
            raise RuntimeError("No GPU MLIR file available for host execution")
        
        print(f"\n{'='*60}")
        print(f"=== Running host + device execution ===")
        print(f"{'='*60}")
        build_and_run_host(
            kernel_mlir=final_gpu_mlir,
            chip=args.chip,
            workdir=workdir,
            rocm_runtime_lib=args.rocm_runtime_lib,
            runner_utils_lib=args.runner_utils_lib,
        )



if __name__ == "__main__":
    main()
