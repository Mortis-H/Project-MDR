#!/usr/bin/env python3
"""
端到端測試腳本：驗證 pipeline 轉換的正確性

支援兩種測試模式：

模式 1 (原本模式):
1. 編譯原始 kernel → 提取 .s 文件
2. 執行 host 程式 → 記錄輸出 (結果 A)
3. 用 pipeline.py 處理 .s → 生成新的 .hsaco
4. 替換 .hsaco 後再執行 → 記錄輸出 (結果 B)
5. 比較 A == B，驗證 pipeline 正確性

模式 2 (Universal Runner 模式):
1. 編譯原始 kernel → 提取 .s 和 .hsaco
2. 用 universal_hsaco_runner 執行原始 .hsaco → 記錄輸出 (結果 A)
3. 用 pipeline.py 處理 .s → 生成新的 .hsaco
4. 用 universal_hsaco_runner 執行重建 .hsaco → 記錄輸出 (結果 B)
5. 比較 A == B，驗證 pipeline 正確性
"""

import argparse
import pathlib
import re
import shutil
import subprocess
import sys
import difflib
import hashlib
from typing import Tuple


def run_cmd(cmd, cwd=None, capture=False):
    """執行命令"""
    print(f"[$] {' '.join(str(c) for c in cmd)}")
    if capture:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            print(f"\n✗ 命令執行失敗 (exit code: {result.returncode})")
            if result.stdout:
                print("--- stdout ---")
                print(result.stdout)
            if result.stderr:
                print("--- stderr ---")
                print(result.stderr)
            raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
        return result.stdout, result.stderr
    else:
        subprocess.run(cmd, cwd=cwd, check=True)
        return None, None


def ensure_tool(name: str):
    """確認工具存在"""
    if shutil.which(name) is None:
        raise RuntimeError(f"Required tool '{name}' not found in PATH")


def calculate_file_hash(file_path: pathlib.Path) -> str:
    """計算檔案的 SHA256 hash"""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def verify_symlink_target(symlink: pathlib.Path, expected_target: pathlib.Path):
    """
    驗證符號連結指向正確的目標
    
    Returns:
        實際目標的絕對路徑
    """
    if not symlink.is_symlink():
        raise RuntimeError(f"{symlink} 不是符號連結！")
    
    # 解析符號連結的實際目標
    actual_target = symlink.resolve()
    expected_target_resolved = expected_target.resolve()
    
    print(f"\n[驗證] 符號連結檢查:")
    print(f"  符號連結: {symlink}")
    print(f"  指向目標: {symlink.readlink()} (相對路徑)")
    print(f"  實際路徑: {actual_target}")
    print(f"  預期路徑: {expected_target_resolved}")
    
    if actual_target != expected_target_resolved:
        raise RuntimeError(
            f"符號連結指向錯誤的目標！\n"
            f"  實際: {actual_target}\n"
            f"  預期: {expected_target_resolved}"
        )
    
    print(f"  ✓ 符號連結正確")
    return actual_target


def extract_isa_from_hipcc_temps(workdir: pathlib.Path, arch: str) -> pathlib.Path:
    """
    從 hipcc --save-temps 產生的臨時文件中提取 ISA assembly
    
    hipcc 會生成類似這樣的文件：
    - vec_add_kernel-gfx950.s (已組裝的 ISA)
    
    Returns:
        提取出來的 .s 文件路徑
    """
    # 尋找 hipcc 生成的 .s 文件
    isa_files = list(workdir.glob(f"*-{arch}.s"))
    
    if not isa_files:
        raise RuntimeError(f"No ISA assembly file (*-{arch}.s) found in {workdir}")
    
    if len(isa_files) > 1:
        print(f"[!] Found multiple ISA files: {isa_files}, using the first one")
    
    return isa_files[0]


def step1_compile_original(kernel_src: pathlib.Path,
                           host_src: pathlib.Path,
                           arch: str,
                           workdir: pathlib.Path) -> Tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    """
    Step 1: 編譯原始 kernel 和 host 程式
    
    使用 hipcc 編譯 kernel，直接使用其生成的原始文件名，不做額外複製。
    
    Returns:
        (hsaco_path, isa_path, executable_path)
    """
    ensure_tool("hipcc")
    
    print("\n" + "="*60)
    print("Step 1 (HIPCC): 編譯原始 kernel")
    print("="*60)
    
    # 編譯 kernel 到 code object (並保存臨時文件)
    # hipcc 會自動生成完整的文件名（包含架構信息）
    print("\n[1.1] 編譯 kernel 到 code object...")
    kernel_base_name = kernel_src.stem
    compile_kernel_cmd = [
        "hipcc",
        "--genco",
        f"--offload-arch={arch}",
        "--save-temps",
        str(kernel_src),
        "-o", str(workdir / f"{kernel_base_name}.out")
    ]
    run_cmd(compile_kernel_cmd, cwd=workdir)
    
    # 查找實際生成的 code object 文件（hipcc 會生成 *-{arch}.out）
    print("\n[1.2] 定位生成的 code object...")
    code_objects = list(workdir.glob(f"*-{arch}.out"))
    if not code_objects:
        raise RuntimeError(f"找不到生成的 code object (*-{arch}.out) in {workdir}")
    hsaco_original = code_objects[0]
    print(f"使用 code object: {hsaco_original}")
    
    # 從生成的 code object 名稱中提取完整的基礎名稱（包含架構信息）
    # 例如: vec_add_kernel-hip-amdgcn-amd-amdhsa-gfx950.out -> vec_add_kernel-hip-amdgcn-amd-amdhsa-gfx950
    kernel_full_name = hsaco_original.stem
    print(f"完整 kernel 名稱: {kernel_full_name}")
    
    # 提取 ISA (hipcc 已經生成，直接使用)
    print("\n[1.3] 定位 ISA assembly...")
    isa_original = extract_isa_from_hipcc_temps(workdir, arch)
    print(f"使用 ISA 文件: {isa_original}")
    
    # 編譯 host 程式（如果提供了 host_src）
    executable = None
    if host_src is not None:
        print("\n[1.4] 編譯 host 程式（使用完整的 kernel 名稱）...")
        executable = workdir / kernel_full_name
        compile_host_cmd = [
            "hipcc",
            str(host_src),
            "-o", str(executable)
        ]
        run_cmd(compile_host_cmd, cwd=workdir)
        print(f"  - 執行檔: {executable}")
    
    print(f"\n✓ 原始編譯完成:")
    print(f"  - HSACO: {hsaco_original}")
    print(f"  - ISA:   {isa_original}")
    if executable:
        print(f"  - 執行檔: {executable}")
    
    return hsaco_original, isa_original, executable


def step1_assemble_mdr(kernel_src: pathlib.Path,
                       host_src: pathlib.Path,
                       mdr_isa: pathlib.Path,
                       arch: str,
                       workdir: pathlib.Path) -> Tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    """
    Step 1 (Assemble 版本): 使用 MDR 優化的 ISA 組合成 HSACO
    
    參照 Makefile 的 assemble target 流程：
    1. clang -cc1as: 組合 .s 成 .o
    2. ld.lld: 連結 .o 成 .out
    3. clang-offload-bundler: 打包成 .hsaco
    
    Args:
        kernel_src: 原始 kernel 源碼（用於命名）
        host_src: Host 程式源碼
        mdr_isa: MDR 優化後的 ISA 組合語言文件
        arch: GPU 架構（如 gfx950）
        workdir: 工作目錄
    
    Returns:
        (hsaco_path, isa_path, executable_path)
    """
    ensure_tool("clang")
    ensure_tool("ld.lld")
    ensure_tool("clang-offload-bundler")
    ensure_tool("hipcc")
    
    print("\n" + "="*60)
    print("Step 1 (Assemble): 使用 MDR 優化的 ISA")
    print("="*60)
    
    kernel_base_name = kernel_src.stem
    
    # 複製 MDR ISA 到工作目錄
    print(f"\n[1.1] 複製 MDR ISA: {mdr_isa}")
    kernel_s = workdir / f"{kernel_base_name}-{arch}.s"
    shutil.copy(mdr_isa, kernel_s)
    print(f"  -> {kernel_s}")
    
    # Step 1: clang -cc1as 組合成 .o
    print(f"\n[1.2] 組合 ISA 成 object 文件...")
    kernel_o = workdir / f"{kernel_base_name}-{arch}.o"
    assemble_cmd = [
        "clang",
        "-cc1as",
        "-triple", "amdgcn-amd-amdhsa",
        "-filetype", "obj",
        "-main-file-name", kernel_src.name,
        "-target-cpu", arch,
        "-fdebug-compilation-dir=" + str(workdir.resolve()),
        "-dwarf-version=5",
        "-mrelocation-model", "pic",
        "-o", str(kernel_o),
        str(kernel_s)
    ]
    run_cmd(assemble_cmd, cwd=workdir)
    
    # Step 2: ld.lld 連結成 .out
    print(f"\n[1.3] 連結 object 文件...")
    kernel_out = workdir / f"{kernel_base_name}-{arch}.out"
    link_cmd = [
        "ld.lld",
        "-flavor", "gnu",
        "-m", "elf64_amdgpu",
        "--no-undefined",
        "-shared",
        "-plugin-opt=-amdgpu-internalize-symbols",
        "--lto-partitions=8",
        f"-plugin-opt=mcpu={arch}",
        "-plugin-opt=O3",
        "--lto-CGO3",
        "--whole-archive",
        "-o", str(kernel_out),
        str(kernel_o),
        "--no-whole-archive"
    ]
    run_cmd(link_cmd, cwd=workdir)
    
    # Step 3: clang-offload-bundler 打包成 .hsaco
    print(f"\n[1.4] 打包成 HSACO...")
    hsaco_path = workdir / f"{kernel_base_name}.hsaco"
    
    # 清理可能存在的舊文件或符號連結（避免循環引用錯誤）
    if hsaco_path.exists() or hsaco_path.is_symlink():
        print(f"  清理舊的 HSACO 檔案: {hsaco_path}")
        hsaco_path.unlink()
    
    bundle_cmd = [
        "clang-offload-bundler",
        "-type=o",
        "-bundle-align=4096",
        f"-targets=host-x86_64-unknown-linux-gnu,hipv4-amdgcn-amd-amdhsa--{arch}",
        "-input=/dev/null",
        f"-input={kernel_out}",
        f"-output={hsaco_path}",
        "-verbose"
    ]
    run_cmd(bundle_cmd, cwd=workdir)
    
    # 編譯 host 程式（如果提供了 host_src）
    executable = None
    if host_src is not None:
        print(f"\n[1.5] 編譯 host 程式...")
        executable = workdir / kernel_base_name
        compile_host_cmd = [
            "hipcc",
            str(host_src),
            "-o", str(executable)
        ]
        run_cmd(compile_host_cmd, cwd=workdir)
        print(f"  - 執行檔: {executable}")
    
    print(f"\n✓ MDR Assemble 完成:")
    print(f"  - HSACO: {hsaco_path}")
    print(f"  - ISA:   {kernel_s}")
    if executable:
        print(f"  - 執行檔: {executable}")
    
    return hsaco_path, kernel_s, executable


def step2_run_original(executable: pathlib.Path,
                      hsaco_original: pathlib.Path,
                      workdir: pathlib.Path,
                      hsaco_name: str) -> Tuple[str, str]:
    """
    Step 2: 執行原始版本並記錄輸出
    
    Args:
        hsaco_name: Host 程式期望載入的 HSACO 文件名
    
    Returns:
        (程式輸出, HSACO hash)
    """
    print("\n" + "="*60)
    print("Step 2: 執行原始版本")
    print("="*60)
    
    # 創建符號連結，指向原始 HSACO（使用相對路徑，便於目錄移動）
    hsaco_link = workdir / hsaco_name
    
    # 檢查是否需要創建符號連結（避免循環引用）
    # 當 hsaco_link 和 hsaco_original 指向同一個文件時，不需要創建符號連結
    if hsaco_link.absolute() == hsaco_original.absolute():
        print(f"HSACO 檔案已存在於預期位置: {hsaco_name}")
        print(f"  路徑: {hsaco_original}")
    else:
        # 需要創建符號連結
        if hsaco_link.exists() or hsaco_link.is_symlink():
            hsaco_link.unlink()
        # 計算相對路徑
        relative_path = hsaco_original.relative_to(workdir)
        hsaco_link.symlink_to(relative_path)
        print(f"創建符號連結: {hsaco_name} -> {relative_path}")
        
        # 驗證符號連結指向正確的檔案
        verify_symlink_target(hsaco_link, hsaco_original)
    
    # 計算原始 HSACO 的 hash
    original_hash = calculate_file_hash(hsaco_original)
    print(f"\n[驗證] 原始 HSACO 檔案資訊:")
    print(f"  路徑: {hsaco_original}")
    print(f"  大小: {hsaco_original.stat().st_size} bytes")
    print(f"  SHA256: {original_hash}")
    
    print(f"\n執行: {executable}")
    stdout, stderr = run_cmd([str(executable)], cwd=workdir, capture=True)
    
    output = stdout + stderr
    print("\n--- 原始版本輸出 ---")
    print(output)
    print("--- 輸出結束 ---")
    
    return output, original_hash


def step3_rebuild_with_pipeline(isa_file: pathlib.Path,
                                pipeline_script: pathlib.Path,
                                arch: str,
                                workdir: pathlib.Path) -> pathlib.Path:
    """
    Step 3: 用 pipeline.py 重建 kernel
    
    Returns:
        新生成的 .hsaco 路徑
    """
    print("\n" + "="*60)
    print("Step 3: 用 pipeline 重建 kernel")
    print("="*60)
    
    ensure_tool("python3")
    
    pipeline_workdir = workdir / "pipeline_output"
    pipeline_workdir.mkdir(exist_ok=True)
    
    print(f"\n執行 pipeline: {pipeline_script}")
    pipeline_cmd = [
        "python3",
        str(pipeline_script),
        str(isa_file),
        f"--chip={arch}",
        f"--workdir={pipeline_workdir}",
        "--emit-llvm-ir"
    ]
    run_cmd(pipeline_cmd, cwd=workdir)
    
    # Pipeline 會自動為 .s 文件添加 _rebuilt 後綴
    # 所以 kernel_original.s -> kernel_original_rebuilt.hsaco
    hsaco_rebuilt = pipeline_workdir / f"{isa_file.stem}_rebuilt.hsaco"
    
    if not hsaco_rebuilt.exists():
        raise RuntimeError(f"Pipeline 未生成預期的 HSACO: {hsaco_rebuilt}")
    
    print(f"\n✓ Pipeline 重建完成: {hsaco_rebuilt}")
    return hsaco_rebuilt


def step4_run_rebuilt(executable: pathlib.Path,
                     hsaco_rebuilt: pathlib.Path,
                     workdir: pathlib.Path,
                     hsaco_name: str,
                     original_hash: str) -> str:
    """
    Step 4: 用重建的 HSACO 執行並記錄輸出
    
    Args:
        hsaco_name: Host 程式期望載入的 HSACO 文件名
        original_hash: 原始 HSACO 的 hash，用於驗證確實使用了不同的檔案
    
    Returns:
        程式輸出
    """
    print("\n" + "="*60)
    print("Step 4: 執行重建版本")
    print("="*60)
    
    # 計算重建 HSACO 的 hash
    rebuilt_hash = calculate_file_hash(hsaco_rebuilt)
    print(f"\n[驗證] 重建 HSACO 檔案資訊:")
    print(f"  路徑: {hsaco_rebuilt}")
    print(f"  大小: {hsaco_rebuilt.stat().st_size} bytes")
    print(f"  SHA256: {rebuilt_hash}")
    
    # 重要驗證：確認重建的 HSACO 與原始的不同
    print(f"\n[驗證] Hash 比較:")
    print(f"  原始 HSACO:  {original_hash}")
    print(f"  重建 HSACO:  {rebuilt_hash}")
    
    if original_hash == rebuilt_hash:
        print("  ⚠️  警告：重建的 HSACO 與原始檔案的 hash 相同！")
        print("      這可能表示 pipeline 沒有進行任何修改。")
    else:
        print("  ✓ Hash 不同，確認使用了重建的檔案")
    
    # 更新符號連結，指向重建的 HSACO（使用相對路徑，便於目錄移動）
    hsaco_link = workdir / hsaco_name
    
    # 檢查是否需要創建符號連結（避免循環引用）
    if hsaco_link.absolute() == hsaco_rebuilt.absolute():
        print(f"\n重建的 HSACO 已存在於預期位置: {hsaco_name}")
        print(f"  路徑: {hsaco_rebuilt}")
        actual_target = hsaco_rebuilt
    else:
        # 需要創建或更新符號連結
        if hsaco_link.exists() or hsaco_link.is_symlink():
            hsaco_link.unlink()
        # 計算相對路徑（rebuilt 在子目錄 pipeline_output/ 下）
        relative_path = hsaco_rebuilt.relative_to(workdir)
        hsaco_link.symlink_to(relative_path)
        print(f"\n更新符號連結: {hsaco_name} -> {relative_path}")
        
        # 驗證符號連結指向正確的檔案
        actual_target = verify_symlink_target(hsaco_link, hsaco_rebuilt)
    
    # 再次驗證：執行前確認符號連結指向的檔案的 hash
    link_target_hash = calculate_file_hash(actual_target)
    print(f"\n[最終驗證] 即將執行的 HSACO:")
    print(f"  符號連結: {hsaco_link}")
    print(f"  實際檔案: {actual_target}")
    print(f"  SHA256:   {link_target_hash}")
    
    if link_target_hash != rebuilt_hash:
        raise RuntimeError(
            f"符號連結目標的 hash 與重建 HSACO 不匹配！\n"
            f"  連結目標 hash: {link_target_hash}\n"
            f"  重建檔案 hash: {rebuilt_hash}"
        )
    
    if link_target_hash == original_hash:
        raise RuntimeError(
            f"錯誤：符號連結仍指向原始 HSACO！\n"
            f"  當前 hash: {link_target_hash}\n"
            f"  原始 hash: {original_hash}"
        )
    
    print("  ✓ 確認將使用重建的 HSACO 執行")
    
    print(f"\n執行: {executable}")
    stdout, stderr = run_cmd([str(executable)], cwd=workdir, capture=True)
    
    output = stdout + stderr
    print("\n--- 重建版本輸出 ---")
    print(output)
    print("--- 輸出結束 ---")
    
    return output


def step5_compare_outputs(output_original: str, output_rebuilt: str) -> bool:
    """
    Step 5: 比較兩次執行的輸出（忽略 HSACO 路徑差異）
    
    Returns:
        True 如果輸出相同
    """
    print("\n" + "="*60)
    print("Step 5: 比較輸出")
    print("="*60)
    
    # 過濾輸出：移除包含 HSACO 路徑的行
    def filter_output(output):
        """過濾掉路徑相關的行，只保留實際計算結果"""
        lines = output.splitlines()
        filtered = []
        for line in lines:
            # 跳過包含 HSACO 路徑的行
            if "HSACO:" in line and ("/" in line or "\\" in line):
                continue
            filtered.append(line)
        return '\n'.join(filtered)
    
    # 先比較原始輸出（完全相同最好）
    if output_original == output_rebuilt:
        print("\n✓ 測試通過！兩次執行的輸出完全相同。")
        return True
    
    # 如果不完全相同，過濾後再比較（忽略路徑差異）
    filtered_original = filter_output(output_original)
    filtered_rebuilt = filter_output(output_rebuilt)
    
    if filtered_original == filtered_rebuilt:
        print("\n✓ 測試通過！計算結果相同（已忽略 HSACO 路徑差異）。")
        print("\n註：HSACO 路徑不同是正常的（原始 vs 重建），不影響測試結果。")
        return True
    else:
        print("\n✗ 測試失敗！計算結果不同。\n")
        print("差異如下（已過濾路徑）:")
        print("-" * 60)
        
        diff = difflib.unified_diff(
            filtered_original.splitlines(keepends=True),
            filtered_rebuilt.splitlines(keepends=True),
            fromfile="原始版本",
            tofile="重建版本",
            lineterm=""
        )
        print("".join(diff))
        
        return False


# ============================================================================
# Universal Runner 模式函數
# ============================================================================

def step2_run_with_universal_runner(hsaco_path: pathlib.Path,
                                    runner_path: pathlib.Path,
                                    kernel_name: str,
                                    kernel_type: str,
                                    test_size: int,
                                    workdir: pathlib.Path,
                                    label: str = "原始") -> Tuple[str, str]:
    """
    使用 universal_hsaco_runner 執行 HSACO 並記錄輸出
    
    Args:
        hsaco_path: HSACO 檔案路徑
        runner_path: universal_hsaco_runner 可執行檔路徑
        kernel_name: Kernel 函數名稱 (mangled name)
        kernel_type: Kernel 類型 (float_add, int_scalar, etc.)
        test_size: 測試資料大小
        workdir: 工作目錄
        label: 標籤（用於顯示，如 "原始" 或 "重建"）
    
    Returns:
        (程式輸出, HSACO hash)
    """
    print("\n" + "="*60)
    print(f"執行{label}版本 (Universal Runner)")
    print("="*60)
    
    # 計算 HSACO 的 hash
    hsaco_hash = calculate_file_hash(hsaco_path)
    print(f"\n[驗證] {label} HSACO 檔案資訊:")
    print(f"  路徑: {hsaco_path}")
    print(f"  大小: {hsaco_path.stat().st_size} bytes")
    print(f"  SHA256: {hsaco_hash}")
    
    # 使用 universal_hsaco_runner 執行
    print(f"\n執行 universal_hsaco_runner:")
    print(f"  HSACO:  {hsaco_path}")
    print(f"  Kernel: {kernel_name}")
    print(f"  Type:   {kernel_type}")
    print(f"  Size:   {test_size}")
    
    runner_cmd = [
        str(runner_path),
        str(hsaco_path),
        kernel_name,
        kernel_type,
        str(test_size)
    ]
    
    stdout, stderr = run_cmd(runner_cmd, cwd=workdir, capture=True)
    
    output = stdout + stderr
    print(f"\n--- {label}版本輸出 ---")
    print(output)
    print("--- 輸出結束 ---")
    
    return output, hsaco_hash


def main():
    ap = argparse.ArgumentParser(
        description="E2E 測試：驗證 pipeline 轉換的正確性（支援兩種模式）"
    )
    
    # 通用參數
    ap.add_argument(
        "--kernel",
        type=pathlib.Path,
        default=pathlib.Path("../../../Track_A/e2e_test/vec_add_kernel.hip"),
        help="Kernel 源碼文件 [default: ../../../Track_A/e2e_test/vec_add_kernel.hip]"
    )
    ap.add_argument(
        "--pipeline",
        type=pathlib.Path,
        default=pathlib.Path("pipeline.py"),
        help="Pipeline 腳本路徑 [default: pipeline.py]"
    )
    ap.add_argument(
        "--arch",
        default="gfx950",
        help="GPU 架構 [default: gfx950]"
    )
    ap.add_argument(
        "--workdir",
        type=pathlib.Path,
        default=pathlib.Path("output"),
        help="工作目錄 [default: output]"
    )
    ap.add_argument(
        "--use-mdr-isa",
        type=pathlib.Path,
        default=None,
        help="使用 MDR 優化後的 ISA 組合語言文件進行 assemble [若不指定則使用一般 hipcc 編譯]"
    )
    
    # 模式選擇
    ap.add_argument(
        "--use-universal-runner",
        action="store_true",
        help="使用 universal_hsaco_runner 模式（不需要 host source）"
    )
    
    # 原本模式的參數（當不使用 universal runner 時需要）
    ap.add_argument(
        "--host",
        type=pathlib.Path,
        default=pathlib.Path("../../../Track_A/e2e_test/main.cpp"),
        help="Host 程式源碼 [default: ../../../Track_A/e2e_test/main.cpp] (僅原本模式需要)"
    )
    ap.add_argument(
        "--hsaco-name",
        default=None,
        help="Host 程式期望載入的 HSACO 文件名 [default: 從 kernel 文件名自動提取] (僅原本模式需要)"
    )
    
    # Universal Runner 模式的參數（當使用 universal runner 時需要）
    ap.add_argument(
        "--runner",
        type=pathlib.Path,
        default=pathlib.Path("../mlir/test/Dialect/AMDISA/universal_hsaco_runner"),
        help="universal_hsaco_runner 可執行檔路徑 [default: ../mlir/test/Dialect/AMDISA/universal_hsaco_runner]"
    )
    ap.add_argument(
        "--kernel-name",
        default=None,
        help="Kernel 函數名稱 (mangled name), 例如: _Z9vectorAddPKfS0_Pfi (Universal Runner 模式需要)"
    )
    ap.add_argument(
        "--kernel-type",
        choices=["float_add", "int_scalar", "int_mem", "int_cond", "int_loop", "int_shared"],
        default=None,
        help="Kernel 類型 (Universal Runner 模式需要)"
    )
    ap.add_argument(
        "--test-size",
        type=int,
        default=1024,
        help="測試資料大小 [default: 1024] (Universal Runner 模式需要)"
    )
    
    args = ap.parse_args()
    
    # 解析路徑
    kernel_src = args.kernel.resolve()
    pipeline_script = args.pipeline.resolve()
    workdir = args.workdir.resolve()
    
    # 檢查必要文件存在
    if not kernel_src.exists():
        raise FileNotFoundError(f"Kernel 源碼不存在: {kernel_src}")
    if not pipeline_script.exists():
        raise FileNotFoundError(f"Pipeline 腳本不存在: {pipeline_script}")
    
    # 根據模式驗證參數
    if args.use_universal_runner:
        # Universal Runner 模式
        if args.kernel_name is None:
            raise ValueError("Universal Runner 模式需要指定 --kernel-name")
        if args.kernel_type is None:
            raise ValueError("Universal Runner 模式需要指定 --kernel-type")
        
        runner_path = args.runner.resolve()
        if not runner_path.exists():
            raise FileNotFoundError(f"universal_hsaco_runner 不存在: {runner_path}")
        
        print("="*60)
        print("E2E 測試開始 (Universal Runner 模式)")
        print("="*60)
        print(f"Kernel:      {kernel_src}")
        print(f"Runner:      {runner_path}")
        print(f"Kernel Name: {args.kernel_name}")
        print(f"Kernel Type: {args.kernel_type}")
        print(f"Test Size:   {args.test_size}")
        print(f"Pipeline:    {pipeline_script}")
        print(f"架構:        {args.arch}")
        print(f"工作目錄:    {workdir}")
        if args.use_mdr_isa:
            print(f"MDR ISA:     {args.use_mdr_isa.resolve()}")
            print(f"編譯模式:    MDR Assemble")
        else:
            print(f"編譯模式:    一般 HIPCC")
    else:
        # 原本模式
        host_src = args.host.resolve()
        if not host_src.exists():
            raise FileNotFoundError(f"Host 源碼不存在: {host_src}")
        
        # 如果沒有指定 hsaco-name，從 kernel 文件名自動提取
        if args.hsaco_name is None:
            args.hsaco_name = f"{kernel_src.stem}.hsaco"
            print(f"[INFO] 自動設定 HSACO 名稱: {args.hsaco_name}")
        
        print("="*60)
        print("E2E 測試開始 (原本模式)")
        print("="*60)
        print(f"Kernel:     {kernel_src}")
        print(f"Host:       {host_src}")
        print(f"Pipeline:   {pipeline_script}")
        print(f"架構:       {args.arch}")
        print(f"工作目錄:   {workdir}")
        print(f"HSACO 名稱: {args.hsaco_name}")
        if args.use_mdr_isa:
            print(f"MDR ISA:    {args.use_mdr_isa.resolve()}")
            print(f"編譯模式:   MDR Assemble")
        else:
            print(f"編譯模式:   一般 HIPCC")
    
    # 創建工作目錄
    workdir.mkdir(parents=True, exist_ok=True)
    
    try:
        if args.use_universal_runner:
            # ============================================================
            # Universal Runner 模式
            # ============================================================
            
            # Step 1: 編譯 kernel（根據是否使用 MDR ISA 選擇）
            if args.use_mdr_isa:
                # 使用 MDR assemble 模式
                if not args.use_mdr_isa.exists():
                    raise FileNotFoundError(f"MDR ISA 文件不存在: {args.use_mdr_isa}")
                
                print("\n[模式] 使用 MDR Assemble 模式")
                hsaco_original, isa_original, _ = step1_assemble_mdr(
                    kernel_src, None, args.use_mdr_isa, args.arch, workdir
                )
            else:
                # 使用一般 hipcc 編譯模式（只編譯 kernel，不編譯 host）
                print("\n[模式] 使用一般 HIPCC 編譯模式")
                hsaco_original, isa_original, executable = step1_compile_original(
                    kernel_src, None, args.arch, workdir
                )
            
            # Step 2: 用 universal_hsaco_runner 執行原始版本
            output_original, original_hash = step2_run_with_universal_runner(
                hsaco_original, runner_path, args.kernel_name,
                args.kernel_type, args.test_size, workdir, "原始"
            )
            
            # Step 3: 用 pipeline 重建
            hsaco_rebuilt = step3_rebuild_with_pipeline(
                isa_original, pipeline_script, args.arch, workdir
            )
            
            # Step 4: 用 universal_hsaco_runner 執行重建版本
            output_rebuilt, rebuilt_hash = step2_run_with_universal_runner(
                hsaco_rebuilt, runner_path, args.kernel_name,
                args.kernel_type, args.test_size, workdir, "重建"
            )
            
            # 驗證使用了不同的 HSACO
            if original_hash == rebuilt_hash:
                print("\n⚠️  警告：原始和重建的 HSACO hash 相同！")
            else:
                print(f"\n✓ 確認使用了不同的 HSACO")
                print(f"  原始: {original_hash[:16]}...")
                print(f"  重建: {rebuilt_hash[:16]}...")
            
            # Step 5: 比較結果
            success = step5_compare_outputs(output_original, output_rebuilt)
            
        else:
            # ============================================================
            # 原本模式
            # ============================================================
            
            # Step 1: 編譯（根據模式選擇）
            if args.use_mdr_isa:
                # 使用 MDR assemble 模式
                if not args.use_mdr_isa.exists():
                    raise FileNotFoundError(f"MDR ISA 文件不存在: {args.use_mdr_isa}")
                
                print("\n[模式] 使用 MDR Assemble 模式")
                hsaco_original, isa_original, executable = step1_assemble_mdr(
                    kernel_src, host_src, args.use_mdr_isa, args.arch, workdir
                )
            else:
                # 使用一般 hipcc 編譯模式
                print("\n[模式] 使用一般 HIPCC 編譯模式")
                hsaco_original, isa_original, executable = step1_compile_original(
                    kernel_src, host_src, args.arch, workdir
                )
            
            # Step 2: 執行原始版本
            output_original, original_hash = step2_run_original(
                executable, hsaco_original, workdir, args.hsaco_name
            )
            
            # Step 3: 用 pipeline 重建
            hsaco_rebuilt = step3_rebuild_with_pipeline(
                isa_original, pipeline_script, args.arch, workdir
            )
            
            # Step 4: 執行重建版本
            output_rebuilt = step4_run_rebuilt(
                executable, hsaco_rebuilt, workdir, args.hsaco_name, original_hash
            )
            
            # Step 5: 比較結果
            success = step5_compare_outputs(output_original, output_rebuilt)
        
        # 顯示最終結果
        print("\n" + "="*60)
        if success:
            mode_str = "Universal Runner" if args.use_universal_runner else "原本"
            print(f"測試結果: ✓ 通過 ({mode_str}模式)")
            print("="*60)
            return 0
        else:
            mode_str = "Universal Runner" if args.use_universal_runner else "原本"
            print(f"測試結果: ✗ 失敗 ({mode_str}模式)")
            print("="*60)
            return 1
            
    except Exception as e:
        print(f"\n✗ 測試過程中發生錯誤: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
