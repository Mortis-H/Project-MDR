# Timestamp Profiling 範例

## 概述

此範例展示如何使用 MDR 的 `@TIMESTAMP_START` 和 `@TIMESTAMP_END` directive 來測量 kernel 內部的執行時間。

## 原理

1. **`@TIMESTAMP_START`**：使用 `s_memtime` 指令記錄時間戳，並備份到 VGPR
2. **`@TIMESTAMP_END`**：再次記錄時間戳，計算差值並輸出

由於 MDR 使用「快照 + 延遲輸出」機制，printf 的 hostcall 開銷是在測量結束後才發生，因此**不會污染測量結果**。

### 實驗驗證（2026-01-28）

| 測量方式 | 時間 (ticks) | 說明 |
|---------|------------|------|
| rocprofv2 基準（硬體測量）| 1,960 | 業界標準 |
| MDR s_memtime | 1,768 | ✅ 誤差 < 10% |

**結論**：MDR timestamp 與硬體 profiler 精確度相當。

## 語法

### 基本用法

```asm
; @TIMESTAMP_START
; ... kernel code ...
; @TIMESTAMP_END
```

### 條件式輸出（推薦）

```asm
; @TIMESTAMP_START
; ... kernel code ...
; @TIMESTAMP_END if $lane == 0:
```

只有每個 wavefront 的第一個 lane 輸出（每 64 個 thread 輸出一次）。

### 帶標籤（多區段測量）

```asm
; @TIMESTAMP_START label="load"
global_load_dword v6, v[4:5], off
s_waitcnt vmcnt(0)
; @TIMESTAMP_END label="load" if $lane == 0:

; @TIMESTAMP_START label="compute"
v_add_f32_e32 v2, v6, v7
; @TIMESTAMP_END label="compute" if $lane == 0:
```

## 執行範例

```bash
# 編譯
python3 ../../mdr_printf.py vector_add_profiled.s --output-dir output

# 測試
../../Track_B/kernel_testcases/universal_hsaco_runner \
    output/vector_add_profiled_debug_injected.hsaco \
    _Z9vectorAddPKfS0_Pfi \
    float_add \
    64

# 預期輸出：
# [Timestamp default_0] elapsed = 1768 ticks
# ✅ PASS: All 64 elements correct
```

## 與其他 Profiling 工具比較

| 功能 | rocprofv2 | rocprofv3 --att | MDR @TIMESTAMP |
|------|-----------|-----------------|----------------|
| 整個 kernel 時間 | ✅ | ✅ | ✅ |
| kernel 內部區段 | ❌ | ❌ | ✅ |
| 每條指令時間 | ❌ | ✅ | ❌ |
| 與 @PRINT 整合 | ❌ | ❌ | ✅ |
| 非侵入式 | ✅ | ✅ | ❌ |

## 相關文檔

- [Timestamp Profiling 設計文檔](../../docs/timestamp_profiling_design.md) - 包含完整實驗記錄和技術細節
