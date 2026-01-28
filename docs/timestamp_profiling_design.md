# Kernel 內部 Timestamp/Profiling 功能設計

## 背景

目標：透過 MDR 的 printf 功能，在 kernel 內部實現 profiler 功能，測量 kernel 執行時間。

---

## 🎉 Phase 1 實驗結果（2026-01-28 驗證成功）

### 關鍵發現：s_waitcnt lgkmcnt(0) 是必要的！

**問題**：初始實驗測量結果不穩定（數億 ticks），與 rocprofv2 硬體測量差異巨大。

**根因**：`s_memtime` / `s_memrealtime` 是需要等待的指令，結果不會立即可用。

```asm
; ❌ 錯誤寫法（結果不穩定）
s_memtime s[20:21]
v_mov_b32 v25, s20    ; s20 還沒準備好！會讀到錯誤值

; ✅ 正確寫法
s_memtime s[20:21]
s_waitcnt lgkmcnt(0)  ; 必須等待！
v_mov_b32 v25, s20    ; 現在 s20 有正確的值
```

### 實驗數據比較

| 測量方式 | 時間 (ticks) | 說明 |
|---------|------------|------|
| rocprofv2 基準（無 printf）| **1,960** | 硬體測量，純 kernel 執行時間 |
| **MDR s_memtime**（計算部分）| **1,768** | ✅ 與硬體測量吻合！ |
| rocprofv2 帶 printf | 120,842 | 包含 printf hostcall 開銷 |

**關鍵結論**：
1. **MDR timestamp 準確度已驗證** - 與 rocprofv2 硬體測量吻合（誤差 < 10%）
2. **Printf 開銷約 61 倍** - 但不影響 timestamp 測量
3. **快照機制有效** - timestamp 在 printf 執行前完成，測量的是純計算時間

### 時鐘選擇：s_memtime vs s_memrealtime

兩者在加上 `s_waitcnt lgkmcnt(0)` 後都能穩定工作，但使用不同的時鐘源：

#### 實測數據比較

| 指令 | 測量結果 (ticks) | 穩定性 | 與 rocprofv2 (1,960) |
|------|-----------------|--------|---------------------|
| **s_memrealtime** | 68-88 | ✅ 非常穩定 | ⚠️ 頻率不同（約 23x 慢）|
| **s_memtime** | 1576-1932 | ✅ 穩定（±10%）| ✅ **吻合** |

#### 頻率分析

```
頻率比: s_memtime / s_memrealtime ≈ 1748 / 76 ≈ 23x

- s_memtime     → GPU 核心時鐘（~2 GHz）
- s_memrealtime → Real-time 時鐘（~100 MHz）
```

#### 選擇建議

| 使用場景 | 推薦指令 | 原因 |
|---------|---------|------|
| 與 rocprof/rocprofv2 比較 | **s_memtime** | 時鐘源一致，數值可直接比較 |
| 絕對時間測量（ns/μs） | s_memrealtime | 固定頻率，不受 DVFS 影響 |
| GPU 動態頻率調整環境 | s_memrealtime | 不受頻率變化影響 |
| **預設推薦** | **s_memtime** | 與標準 profiling 工具一致 |

---

## AMD GPU 時間戳指令

### 可用指令

| 指令 | 說明 | 輸出 | 特性 |
|------|------|------|------|
| `s_memtime` | GPU 時鐘計數器 | 64-bit → s[N:N+1] | **推薦**：與 rocprof 時鐘一致 |
| `s_memrealtime` | Real-time 計數器 | 64-bit → s[N:N+1] | 固定頻率，適合跨 kernel 比較 |

### 正確使用方式（必須包含 s_waitcnt！）

```asm
; 記錄時間戳到 SGPR
s_memtime s[10:11]               ; 64-bit 時間戳存入 s[10:11]
s_waitcnt lgkmcnt(0)             ; ⚠️ 必須等待！

; 如果需要每個 thread 獨立記錄，需要轉存到 VGPR
v_mov_b32 v14, s10               ; 低 32 位
v_mov_b32 v15, s11               ; 高 32 位
```

---

## 基本設計概念

### 已實現的語法

```asm
; ===== Kernel 開始 =====
; @TIMESTAMP_START                          ; 預設 label
; @TIMESTAMP_START label="my_section"       ; 自訂 label

; ... kernel 主體執行 ...

; ===== Kernel 結束前 =====
; @TIMESTAMP_END                            ; 每個 thread 輸出
; @TIMESTAMP_END if $lane == 0:             ; 只有 lane 0 輸出（推薦）
; @TIMESTAMP_END label="my_section" if $tid == 0:  ; 指定 label + 條件
```

### 內部實現（自動生成）

```asm
; ===== @TIMESTAMP_START 注入 =====
s_memtime s[20:21]               ; 記錄開始時間
s_waitcnt lgkmcnt(0)             ; ⚠️ 必須等待！
v_mov_b32 v_start_lo, s20        ; 備份到 VGPR（快照）
v_mov_b32 v_start_hi, s21

; ... kernel 主體執行 ...

; ===== @TIMESTAMP_END 注入（在 printf section）=====
s_memtime s[20:21]               ; 記錄結束時間
s_waitcnt lgkmcnt(0)             ; 等待
v_sub_u32 v_diff, s20, v_start_lo  ; 計算差值（低 32 位足夠）

; 透過 gpu.printf 輸出
gpu.printf "[Timestamp %s] elapsed = %u ticks", label, v_diff
```

### 快照機制的優勢

```
時間軸：
  @TIMESTAMP_START     @TIMESTAMP_END        gpu.printf
        |                    |                   |
        v                    v                   v
  [記錄開始時間]----[kernel 計算]----[記錄結束時間]----[hostcall 輸出]
        |<---- 這段是我們測量的 ---->|
                                              |<-- printf 開銷不影響測量 -->|
```

---

## 已知限制和挑戰

### 1. ~~Printf 開銷污染測量結果~~ ✅ 已解決！

**原本擔心**：
```
測量的時間 = 實際 kernel 時間 + printf 開銷?
```

**實驗證實**：透過快照機制，printf 開銷**不會**影響測量結果！

```
實測數據：
- 純 kernel 執行（rocprofv2）: 1,960 ticks
- MDR timestamp 測量:          1,768 ticks  ✅ 準確！
- 帶 printf kernel（rocprofv2）: 120,842 ticks（開銷在這裡）
```

**結論**：快照機制有效隔離了 printf 開銷，timestamp 測量的是純計算時間。

### 2. 64-bit 運算複雜性

GPU 沒有原生 64-bit 減法指令，需要拆分處理：

```asm
; 64-bit 減法：result = end - start
v_sub_co_u32 v_diff_lo, vcc, v_end_lo, v_start_lo    ; 低 32 位減法
v_subb_co_u32 v_diff_hi, vcc, v_end_hi, v_start_hi, vcc  ; 高 32 位 + borrow
```

**簡化方案**：
- 對於短時間測量（<4 秒 @ 1GHz），只使用低 32 位
- 高 32 位通常為 0，可以忽略

### 3. 時鐘頻率轉換

| 計數器 | 典型頻率 | 說明 |
|--------|----------|------|
| `s_memrealtime` | 通常是固定頻率（如 27MHz、100MHz） | 需要查詢硬體 |
| `s_memtime` | 與 memory clock 相關（如 1.2GHz） | 會隨 DVFS 變化 |

**獲取頻率的方法**：
```cpp
// Host 端查詢
hipDeviceGetAttribute(&freq, hipDeviceAttributeMemoryClockRate, device);
```

**建議**：先輸出原始 tick 數，Host 端再轉換

### 4. Wavefront 調度差異

```
時間軸：
        t0        t1        t2        t3
         |         |         |         |
WF 0:    [===== kernel =====]
WF 1:         [===== kernel =====]
WF 2:              [===== kernel =====]
```

- 不同 wavefront 的開始時間不同
- 每個 wavefront 測量的是「自己的執行時間」
- 無法直接得到「整個 kernel 的端到端時間」

**解決方案**：
- 使用 atomic_min/max 記錄全局最早開始/最晚結束時間
- 或在 Host 端分析所有 thread 的時間戳

### 5. SGPR 資源佔用

- 每個時間戳需要 2 個 SGPR（64-bit）
- 開始 + 結束需要 4 個 SGPR
- 需要確保不與 kernel 原有使用衝突

---

## 實現路徑

### ✅ Phase 1：概念驗證（Printf 輸出）— 已完成！

**目標**：驗證時間戳功能可以正常工作

**實現語法**：
```asm
; @TIMESTAMP_START label="kernel_total"
; ... kernel ...
; @TIMESTAMP_END label="kernel_total" if $lane == 0:
```

**實際輸出**：
```
[Timestamp kernel_total] elapsed = 1768 ticks
✅ PASS: All 64 elements correct
```

**重要發現**：
- ✅ 測量結果**不包含** printf 開銷（快照機制有效）
- ✅ 與 rocprofv2 硬體測量吻合（誤差 < 10%）
- ✅ 使用 `s_memtime` + `s_waitcnt lgkmcnt(0)` 組合

### Phase 2：Global Memory 記錄

**目標**：減少測量開銷

**設計**：
```asm
; 將時間戳寫入 global memory buffer
global_store_dwordx2 v[addr], v[time_lo:time_hi], off
```

**優點**：
- 開銷遠小於 printf
- 可以記錄所有 thread 的時間戳
- Host 端離線分析

**需求**：
- 需要額外的 buffer 參數
- Host 端需要讀取和解析

### Phase 3：進階功能

1. **區段計時**：
   ```asm
   ; @TIMESTAMP label="section_A"
   ; ... code section A ...
   ; @TIMESTAMP label="section_B"
   ; ... code section B ...
   ; @TIMESTAMP label="end"
   ```

2. **Atomic 統計**：
   ```asm
   ; 記錄全局最小開始時間
   global_atomic_min_u64 v[global_start_addr], v[my_start]
   ; 記錄全局最大結束時間
   global_atomic_max_u64 v[global_end_addr], v[my_end]
   ```

3. **統計輸出**：
   - Min/Max/Avg 執行時間
   - 時間分布直方圖

---

## 與 rocprof 的比較（2026-01-28 實驗驗證）

### 精確度比較

| 測量方式 | vectorAdd kernel 時間 | 說明 |
|---------|---------------------|------|
| rocprofv2（硬體）| 1,960 ticks | 業界標準 |
| MDR s_memtime | 1,768 ticks | **誤差 < 10%** ✅ |

### 功能比較

| 特性 | MDR s_memtime | rocprof |
|------|---------------|---------|
| **細粒度** | ✅ 可測量任意程式碼區段 | ❌ 只有 kernel 整體時間 |
| **每 thread 時間** | ✅ 每個 thread/wavefront 獨立 | ❌ 無法區分 |
| **測量開銷** | ✅ **快照機制避免 printf 開銷** | ✅ 硬體 counter，無開銷 |
| **侵入性** | ⚠️ 需要修改 kernel | ✅ 非侵入式 |
| **精確度** | ✅ cycle 級精確（已驗證） | ✅ cycle 級精確 |
| **使用難度** | ⚠️ 需要加 directive | ✅ 簡單命令列 |
| **輸出整合** | ✅ 與 @PRINT debug 整合 | ❌ 獨立工具 |

### 使用場景

| 場景 | 推薦工具 |
|------|---------|
| 測量 kernel 內部特定區段 | **MDR s_memtime** |
| 需要每 thread 獨立計時 | **MDR s_memtime** |
| 與 @PRINT debug 同時使用 | **MDR s_memtime** |
| 只需要整體 kernel 時間 | rocprof |
| 生產環境 profiling | rocprof |
| 非侵入式測量 | rocprof |

---

## 工具定位與價值分析（2026-01-28）

### 核心問題：MDR @TIMESTAMP 與 rocprofv2 有什麼不同？

**重要發現**：rocprofv2 的測量 overhead 極低（使用硬體時間戳），所以單純測量整個 kernel 時間時，MDR @TIMESTAMP 沒有明顯優勢。

**但這兩者的定位完全不同！**

### 測量範圍比較

| 工具 | 測量範圍 | 比喻 |
|------|---------|------|
| **rocprofv2** | 整個 kernel（黑盒）| 測量「整趟火車旅程」 |
| **MDR @TIMESTAMP** | kernel **內部任意區段**（白盒）| 測量「站與站之間」 |

### rocprofv2 的能力

```
Kernel Start ──────────────────────────> Kernel End
     │                                        │
     └─────────── 1,681 ticks ────────────────┘
                    ↑
              rocprofv2 可以測這個
```

### MDR @TIMESTAMP 的能力

```
Kernel Start ──> 區段A ──> 區段B ──> 區段C ──> Kernel End
     │              │         │         │          │
     │   @TS_START  │  @TS_END│  @TS_END│          │
     │      ↓       │    ↓    │    ↓    │          │
     └──────┴───────┴────┴────┴────┴────┴──────────┘
            500 ticks    800 ticks   381 ticks
                  ↑
       rocprofv2 無法做到這件事！
```

### 實際應用範例

```asm
; 複雜 kernel 範例：分析瓶頸在哪裡
kernel_start:
    ; @TIMESTAMP_START label="load"
    ; ... 資料載入 ...
    ; @TIMESTAMP_END label="load" if $lane == 0: f"Load: {$elapsed:d} ticks"
    
    ; @TIMESTAMP_START label="compute"
    ; ... 計算迴圈 ...
    ; @TIMESTAMP_END label="compute" if $lane == 0: f"Compute: {$elapsed:d} ticks"
    
    ; @TIMESTAMP_START label="store"
    ; ... 資料寫回 ...
    ; @TIMESTAMP_END label="store" if $lane == 0: f"Store: {$elapsed:d} ticks"
kernel_end:
```

**rocprofv2 輸出**：整個 kernel 花了 3000 ticks（無法知道瓶頸在哪）

**MDR @TIMESTAMP 輸出**：
```
[Timestamp load] elapsed = 500 ticks      (17%)
[Timestamp compute] elapsed = 2000 ticks  (67%) ← 瓶頸在這裡！
[Timestamp store] elapsed = 500 ticks     (17%)
```

### 為什麼 rocprofv2 的 Overhead 這麼低？

根據 [AMD ROCProfiler Specification](https://rocm.docs.amd.com/projects/rocprofiler/en/latest/reference/rocprofiler_spec.html)：

1. **硬體級別時間戳** - GPU 在 kernel dispatch/begin/end/complete 時自動記錄
2. **基於 AQL Profile HSA Extension** - 不需要軟體插樁
3. **User-Mode Queueing** - dispatch 不需要經過 OS kernel

詳細分析見：[rocprofv2_low_overhead_analysis.md](./rocprofv2_low_overhead_analysis.md)

### 工具選擇決策樹

```
需要測量什麼？
    │
    ├── 整個 kernel 的執行時間
    │       │
    │       └── 使用 rocprofv2 ✅（簡單、非侵入式）
    │
    ├── kernel 內部特定區段的時間
    │       │
    │       └── 使用 MDR @TIMESTAMP ✅（唯一選擇）
    │
    ├── 每條指令的 latency/stall
    │       │
    │       └── 使用 rocprofv3 --att (Thread Trace) ✅
    │
    └── 同時需要 debug + profiling
            │
            └── 使用 MDR @TIMESTAMP + @PRINT ✅（整合方案）
```

### 結論

| 場景 | 推薦工具 | 原因 |
|------|---------|------|
| 比較不同 kernel 的整體效能 | **rocprofv2** | 非侵入式、低 overhead |
| 找出 kernel **內部**的效能瓶頸 | **MDR @TIMESTAMP** | rocprofv2 做不到 |
| Instruction-level 分析 | **rocprofv3 --att** | 最細粒度 |
| 同時 debug + profiling | **MDR @TIMESTAMP + @PRINT** | 整合方案 |

**MDR @TIMESTAMP 的價值不是「取代 rocprofv2」，而是提供 kernel 內部區段分析的能力。**

---

## 問題狀態追蹤

### ✅ 已解決

1. [x] **確認 `s_memtime` 在 gfx950 上可用** - 2026-01-28 驗證成功
2. [x] **時鐘頻率問題** - `s_memtime` 與 rocprofv2 使用相同時鐘源，無需轉換
3. [x] **設計 directive 語法** - `@TIMESTAMP_START` / `@TIMESTAMP_END` 已實現
4. [x] **Printf 開銷問題** - 快照機制有效避免 printf 開銷影響測量
5. [x] **s_waitcnt 問題** - 發現並修復：必須在 `s_memtime` 後加 `s_waitcnt lgkmcnt(0)`

### 📋 待實現（Phase 2+）

1. [ ] Global memory buffer 記錄（完全消除 printf 開銷）
2. [ ] 多區段計時支援（多個 label）
3. [ ] Atomic 統計（min/max/avg）
4. [ ] 自動頻率查詢和時間轉換

---

## 實驗記錄

### 2026-01-28 Phase 1 驗證實驗

**環境**：
- GPU: AMD MI300 (gfx950)
- 測試 kernel: vectorAdd (64 elements)
- 比較工具: rocprofv2 --kernel-trace

**發現問題**：
初始實現結果不穩定（數億 ticks），經調查發現缺少 `s_waitcnt lgkmcnt(0)`。

**修復後結果**：
```
rocprofv2 基準（無 printf）:  1,960 ticks
MDR s_memtime:               1,768 ticks  (誤差 9.8%)
rocprofv2 帶 printf:        120,842 ticks (printf 開銷 61x)
```

**結論**：
1. `s_memtime` 需要 `s_waitcnt lgkmcnt(0)` 等待結果
2. 快照機制成功隔離 printf 開銷
3. MDR timestamp 與硬體 profiler 精確度相當

### 2026-01-28 s_memtime vs s_memrealtime 比較實驗

**目的**：確定應該使用哪個時間戳指令

**測試條件**：兩者都加上 `s_waitcnt lgkmcnt(0)`

**結果**：
```
s_memrealtime: 68, 88, 68, 72, 72 ticks  (平均 ~76 ticks，非常穩定)
s_memtime:     1576, 1876, 1748, 1608, 1932 ticks  (平均 ~1748 ticks，±10%)
rocprofv2:     1,960 ticks

頻率比: 1748 / 76 ≈ 23x
```

**分析**：
- `s_memtime` 使用 GPU 核心時鐘（~2 GHz），與 rocprofv2 一致
- `s_memrealtime` 使用 real-time 時鐘（~100 MHz），頻率較低但更穩定

**決定**：預設使用 `s_memtime`，因為與標準 profiling 工具時鐘源一致

---

## 使用的 Profiling 工具總覽

### 1. rocprofv2 (Kernel-Level Profiling)

```bash
# 測量 kernel 整體執行時間
rocprofv2 --kernel-trace -d output_dir -- ./your_app
```

**輸出**：`results.csv` 包含 Start_Timestamp, End_Timestamp
**用途**：測量整個 kernel 的執行時間（非侵入式）

### 2. rocprofv3 + Thread Trace (Instruction-Level Profiling)

```bash
# 需要先安裝 rocprof-trace-decoder
# 下載：https://github.com/ROCm/rocprof-trace-decoder/releases

# 設置 library path
export ROCPROF_ATT_LIBRARY_PATH="/path/to/rocprof-trace-decoder/lib"

# 收集 instruction-level trace
rocprofv3 --att -d output_dir -- ./your_app
```

**輸出**：
- `stats_*.csv` - 每條指令的 Hitcount, Latency, Stall, Idle
- `ui_output_*/` - ROCprof Compute Viewer 可視化數據

**用途**：分析每條指令的執行時間和 stall 原因

### 3. MDR s_memtime (區段計時，與 @PRINT 整合)

```asm
; 在 .s 文件中加入 directive
; @TIMESTAMP_START label="my_section"
; ... code ...
; @TIMESTAMP_END label="my_section" if $lane == 0:
```

```bash
# 編譯
python3 mdr_printf.py input.s --output-dir output

# 執行
./universal_hsaco_runner output/xxx_debug_injected.hsaco kernel_name type size
```

**輸出**：kernel 執行時輸出 elapsed ticks
**用途**：測量 kernel 內部特定區段，與 @PRINT debug 整合

### 工具選擇指南

| 需求 | 推薦工具 |
|------|---------|
| 快速測量整個 kernel 時間 | rocprofv2 |
| 分析每條指令的瓶頸 | rocprofv3 --att (Thread Trace) |
| 測量 kernel 內部區段 | MDR @TIMESTAMP |
| 同時 debug + profiling | MDR @TIMESTAMP + @PRINT |

---

## 參考資料

- AMD GCN ISA Reference Guide
- AMDGPU Assembly Guide
- rocprof / rocprofv2 / rocprofv3 documentation
- [ROCprof Compute Viewer](https://rocm.docs.amd.com/projects/rocprof-compute-viewer/)
- [Thread Trace 文檔](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-thread-trace.html)
- [AMD ISA: s_memtime/s_memrealtime 指令說明]
