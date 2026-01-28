---
name: rocprofv3-thread-trace
description: Instruction-level GPU profiling using rocprofv3 thread trace (ATT). Use when analyzing per-instruction latency, finding hotspots, debugging stalls, or when the user mentions rocprofv3, thread trace, instruction profiling, or ROCprof Compute Viewer.
---

# rocprofv3 Thread Trace (Instruction-Level Profiling)

Near cycle-accurate instruction tracing for AMD GPUs. Provides per-instruction latency, stall analysis, and hotspot detection.

## Prerequisites

### 1. Install ROCprof Trace Decoder

```bash
# Download from GitHub
curl -L -o /tmp/decoder.tar.gz \
  "https://github.com/ROCm/rocprof-trace-decoder/releases/download/0.1.6/rocprof-trace-decoder-manylinux-2.28-0.1.6-Linux.tar.gz"

# Extract
cd /tmp && tar -xzf decoder.tar.gz

# Set environment variable
export ROCPROF_ATT_LIBRARY_PATH="/tmp/rocprof-trace-decoder-manylinux-2.28-0.1.6-Linux/opt/rocm/lib"
```

### 2. Supported Hardware

- AMD Instinct: MI200, MI300 series
- AMD Radeon: gfx10, gfx11, gfx12

## Quick Start

```bash
# Collect thread trace
ROCPROF_ATT_LIBRARY_PATH="/path/to/decoder/lib" \
rocprofv3 --att -d output_dir -- ./your_application

# View instruction-level stats
cat output_dir/stats_*.csv
```

## Output Files

| File | Description |
|------|-------------|
| `stats_*.csv` | Per-instruction latency summary |
| `ui_output_*/` | Data for ROCprof Compute Viewer |
| `*.att` | Raw trace data |
| `*.out` | Code object binaries |

## Stats CSV Format

```csv
"CodeObj","Vaddr","Instruction","Hitcount","Latency","Stall","Idle","Source"
1,5648,"s_waitcnt lgkmcnt(0)",4,2560,2560,0,""
1,5752,"s_waitcnt vmcnt(0)",1,704,704,0,""
1,5756,"v_add_f32_e32 v2, v6, v7",1,4,0,0,""
```

| Column | Description |
|--------|-------------|
| Hitcount | Times instruction was executed |
| Latency | Total cycles (Stall + Issue/Execute) |
| Stall | Cycles waiting (backpressure, dependency) |
| Idle | Gap between instructions |

## Common Options

| Option | Description |
|--------|-------------|
| `--att` | Enable thread trace |
| `-d <dir>` | Output directory |
| `--att-target-cu <N>` | Target Compute Unit (default: 1) |
| `--att-buffer-size <bytes>` | Trace buffer size (default: 96MB) |
| `--att-activity <N>` | Enable perfcounter streaming (MI200/MI300) |

## Example: Analyze Hotspots

```bash
# Collect trace
rocprofv3 --att -d trace_output -- ./gpu_app

# Find top instructions by latency
cat trace_output/stats_*.csv | sort -t',' -k5 -nr | head -10
```

## Interpreting Results

### High Stall on s_waitcnt
```
s_waitcnt lgkmcnt(0)  Latency: 2560  Stall: 2560
```
→ Waiting for scalar memory operations (s_load)

### High Stall on vmcnt
```
s_waitcnt vmcnt(0)    Latency: 704   Stall: 704
```
→ Waiting for vector memory operations (global_load)

### Low Latency Compute
```
v_add_f32_e32         Latency: 4     Stall: 0
```
→ Efficient computation (no stalls)

## Visualization

Use ROCprof Compute Viewer for graphical analysis:
- Hotspot view: Instruction cost histogram
- Compute Unit view: Wave scheduling
- Instructions view: Per-instruction details with source mapping

## Troubleshooting

### Empty stats.csv
- Kernel may not have launched enough waves
- Try: `--att-target-cu 0` or `--att-shader-engine-mask 0xFFFFFFFF`

### INVALID_SHADER_DATA Error
- AQL profile version mismatch
- Update ROCm or rebuild aqlprofile from source

### Buffer Full Warnings
- Increase buffer: `--att-buffer-size 0x10000000` (256MB)

## Comparison with Other Tools

| Tool | Granularity | Invasive | Use Case |
|------|-------------|----------|----------|
| rocprofv2 | Kernel | No | Quick kernel timing |
| **rocprofv3 --att** | **Instruction** | No | **Deep analysis** |
| MDR @TIMESTAMP | Section | Yes | Debug + timing integration |
