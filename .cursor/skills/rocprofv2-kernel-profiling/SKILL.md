---
name: rocprofv2-kernel-profiling
description: Profile AMD GPU kernel execution time using rocprofv2. Use when measuring kernel performance, comparing kernel execution times, or when the user mentions rocprofv2, kernel profiling, or GPU performance measurement.
---

# rocprofv2 Kernel Profiling

AMD GPU kernel-level profiling tool for measuring kernel execution time.

## Quick Start

```bash
# Measure kernel execution time
rocprofv2 --kernel-trace -d output_dir -- ./your_application

# View results
cat output_dir/results_*.csv
```

## Output Format

Results are saved in CSV format with columns:

| Column | Description |
|--------|-------------|
| Kernel_Name | Name of the executed kernel |
| Start_Timestamp | Kernel start time (ticks) |
| End_Timestamp | Kernel end time (ticks) |
| Grid_Size | Total number of threads |
| Workgroup_Size | Threads per workgroup |

### Calculate Duration

```bash
# Duration = End_Timestamp - Start_Timestamp
# Example: 79809512877433 - 79809512756591 = 120842 ticks
```

## Common Options

| Option | Description |
|--------|-------------|
| `--kernel-trace` | Enable kernel dispatch tracing |
| `-d <dir>` | Output directory |
| `--stats` | Generate statistics summary |
| `--hip-trace` | Include HIP API traces |
| `--hsa-trace` | Include HSA API traces |

## Example Usage

```bash
# Basic kernel profiling
rocprofv2 --kernel-trace -d profiling_output -- ./gpu_app

# With statistics
rocprofv2 --kernel-trace --stats -d profiling_output -- ./gpu_app

# Profile specific application with arguments
rocprofv2 --kernel-trace -d output -- ./runner input.hsaco kernel_name float_add 64
```

## Interpreting Results

```csv
Kernel_Name,Start_Timestamp,End_Timestamp
"vectorAdd(.kd)",79809512756591,79809512877433
```

- **Duration**: 120,842 ticks
- **Clock**: Usually GPU core clock (~1-2 GHz)
- **Time**: ~60-120 μs (depending on clock frequency)

## Limitations

- Measures entire kernel execution (dispatch to completion)
- Cannot measure kernel internal sections
- Non-invasive (no code modification needed)

## See Also

- For instruction-level profiling, use `rocprofv3 --att` (Thread Trace)
- For kernel internal timing, use MDR `@TIMESTAMP` directives

## Notes

- rocprofv2 is being phased out in favor of rocprofv3
- Results are consistent with `s_memtime` hardware counter
- Typical overhead: negligible (hardware-based measurement)
