# Timestamp Profiling Example

## Atomic 模式範例

`vectoradd_atomic_v2.s` 展示如何使用 `@TIMESTAMP` 指令測量 kernel wall time。

### 使用方式

```bash
# 編譯
python3 ../../mdr_printf.py vectoradd_atomic_v2.s --output-dir output --chip gfx950

# 執行
../../Track_B/kernel_testcases/universal_hsaco_runner --timestamp-atomic \
    output/vectoradd_atomic_v2_debug_injected.hsaco \
    _Z9vectorAddPKfS0_Pfi float_add 4096
```

### 測試結果

| Size | Workgroups | MDR (us) | rocprofv3 (us) | Diff |
|------|------------|----------|----------------|------|
| 256 | 1 | 7.52 | 8.84 | -14.9% |
| 4,096 | 16 | 92.56 | 95.56 | -3.1% |
| 16,384 | 64 | 363.80 | 374.24 | -2.8% |

## 技術說明

- **Atomic 模式**使用 `s_memrealtime`（全域同步計數器，~100 MHz）
- 詳細設計文檔：`docs/timestamp_profiling_design.md`
