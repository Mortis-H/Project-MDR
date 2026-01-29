#!/usr/bin/env python3
"""
rocprofv2 穩定性測試：每個 kernel 測量 100 次
"""
import subprocess
import re
import statistics
import os
import glob
import shutil

# 測試配置
NUM_RUNS = 100
RUNNER = "./Track_B/kernel_testcases/universal_hsaco_runner"

# Kernel 配置 (baseline, 無 printf)
KERNELS = {
    "scalar_ops": {
        "hsaco": "experiments/dispatch_overhead_test/baseline_scalar/original_debug.hsaco",
        "name": "_Z9scalarOpsPii",
        "type": "int_scalar",
        "kernel_match": "scalarOps",
    },
    "memory_ops": {
        "hsaco": "experiments/dispatch_overhead_test/baseline_memory/original_debug.hsaco",
        "name": "_Z9memoryOpsPKiPii",
        "type": "int_mem",
        "kernel_match": "memoryOps",
    },
    "conditional": {
        "hsaco": "experiments/dispatch_overhead_test/baseline_conditional/original_debug.hsaco",
        "name": "_Z17conditionalKernelPKiPii",
        "type": "int_cond",
        "kernel_match": "conditionalKernel",
    },
    "loop": {
        "hsaco": "experiments/dispatch_overhead_test/baseline_loop/original_debug.hsaco",
        "name": "_Z10loopKernelPii",
        "type": "int_loop",
        "kernel_match": "loopKernel",
    },
}

def run_rocprof_test(kernel_config, output_dir):
    """運行一次 rocprofv2 測量"""
    # 清理舊的結果
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    cmd = [
        "rocprofv2", "--kernel-trace", "-d", output_dir,
        RUNNER, kernel_config["hsaco"], kernel_config["name"], 
        kernel_config["type"], "64"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # 讀取結果 CSV
    csv_files = glob.glob(f"{output_dir}/results_*.csv")
    if not csv_files:
        return None
    
    with open(csv_files[0], 'r') as f:
        lines = f.readlines()
    
    # 找到目標 kernel 的時間
    # CSV 格式有問題（kernel name 包含逗號），使用正則表達式
    kernel_match = kernel_config["kernel_match"]
    for line in lines[1:]:  # 跳過 header
        if kernel_match in line:
            # 使用正則表達式提取時間戳
            match = re.search(r'(\d{15,}),(\d{15,}),\d+$', line.strip())
            if match:
                start = int(match.group(1))
                end = int(match.group(2))
                return end - start
    
    return None

def analyze_results(values, name):
    """分析測量結果"""
    if len(values) < 3:
        print(f"  {name}: 數據不足")
        return None
    
    # 排序
    sorted_vals = sorted(values)
    
    # 去掉最大和最小
    trimmed = sorted_vals[1:-1]
    
    # 計算統計
    median = statistics.median(trimmed)
    mean = statistics.mean(trimmed)
    stdev = statistics.stdev(trimmed) if len(trimmed) > 1 else 0
    min_val = min(trimmed)
    max_val = max(trimmed)
    
    return {
        "count": len(values),
        "trimmed_count": len(trimmed),
        "median": median,
        "mean": mean,
        "stdev": stdev,
        "min": min_val,
        "max": max_val,
        "range": max_val - min_val,
        "cv": (stdev / mean * 100) if mean > 0 else 0,
        "original_min": sorted_vals[0],
        "original_max": sorted_vals[-1],
    }

def main():
    os.chdir("/home/morhuang/Project-MDR")
    
    print("=" * 60)
    print("rocprofv2 穩定性測試")
    print(f"每個 kernel 測量 {NUM_RUNS} 次")
    print("=" * 60)
    
    all_results = {}
    
    for kernel_name, config in KERNELS.items():
        print(f"\n測試 {kernel_name}...")
        values = []
        output_dir = f"experiments/dispatch_overhead_test/rocprof_temp/{kernel_name}"
        
        for i in range(NUM_RUNS):
            if (i + 1) % 20 == 0:
                print(f"  進度: {i + 1}/{NUM_RUNS}")
            
            val = run_rocprof_test(config, output_dir)
            if val is not None:
                values.append(val)
        
        # 分析結果
        stats = analyze_results(values, kernel_name)
        if stats:
            all_results[kernel_name] = stats
            print(f"  完成: {stats['count']} 次測量")
    
    # 輸出結果
    print("\n" + "=" * 60)
    print("rocprofv2 測量結果摘要")
    print("=" * 60)
    
    print("\n| Kernel | 中位數 | 平均值 | 標準差 | 變異係數 | 範圍 | 原始 Min/Max |")
    print("|--------|--------|--------|--------|----------|------|--------------|")
    
    for name, stats in all_results.items():
        print(f"| {name} | {stats['median']:.0f} | {stats['mean']:.1f} | {stats['stdev']:.1f} | {stats['cv']:.1f}% | {stats['range']} | {stats['original_min']}/{stats['original_max']} |")
    
    # 追加結果到文件
    with open("experiments/dispatch_overhead_test/stability_results.md", "a") as f:
        f.write("\n\n---\n\n")
        f.write("# rocprofv2 穩定性測試結果\n\n")
        f.write(f"測量次數: {NUM_RUNS} 次/kernel\n")
        f.write("統計方法: 去掉最大最小值後計算\n\n")
        
        f.write("## 結果摘要\n\n")
        f.write("| Kernel | 中位數 (ticks) | 平均值 | 標準差 | 變異係數 | 範圍 |\n")
        f.write("|--------|----------------|--------|--------|----------|------|\n")
        
        for name, stats in all_results.items():
            f.write(f"| {name} | {stats['median']:.0f} | {stats['mean']:.1f} | {stats['stdev']:.1f} | {stats['cv']:.1f}% | {stats['range']} |\n")
        
        f.write("\n## 詳細數據\n\n")
        for name, stats in all_results.items():
            f.write(f"### {name}\n")
            f.write(f"- 總測量次數: {stats['count']}\n")
            f.write(f"- 去掉極值後: {stats['trimmed_count']}\n")
            f.write(f"- 中位數: {stats['median']:.0f} ticks\n")
            f.write(f"- 平均值: {stats['mean']:.1f} ticks\n")
            f.write(f"- 標準差: {stats['stdev']:.1f} ticks\n")
            f.write(f"- 變異係數: {stats['cv']:.1f}%\n")
            f.write(f"- 範圍: {stats['min']} ~ {stats['max']} (差: {stats['range']})\n")
            f.write(f"- 原始 Min/Max: {stats['original_min']} / {stats['original_max']}\n\n")
    
    print("\n結果已追加到: experiments/dispatch_overhead_test/stability_results.md")
    
    return all_results

if __name__ == "__main__":
    main()
