# Kernel 內部 Timestamp/Profiling 功能設計

## 背景

目標：透過 MDR 的 printf 功能，在 kernel 內部實現 profiler 功能，測量 kernel 執行時間。

## AMD GPU 時間戳指令

### 可用指令

| 指令 | 說明 | 輸出 | 特性 |
|------|------|------|------|
| `s_memtime` | Memory clock 計數器 | 64-bit → s[N:N+1] | 頻率隨 memory clock 變化 |
| `s_memrealtime` | Real-time 計數器 | 64-bit → s[N:N+1] | 固定頻率，較穩定 |

### 使用方式

```asm
; 記錄時間戳到 SGPR
s_memrealtime s[10:11]           ; 64-bit 時間戳存入 s[10:11]

; 如果需要每個 thread 獨立記錄，需要轉存到 VGPR
v_mov_b32 v14, s10               ; 低 32 位
v_mov_b32 v15, s11               ; 高 32 位
```

---

## 基本設計概念

### 方案：開始/結束時間戳

```asm
; ===== Kernel 開始 =====
s_memrealtime s[10:11]           ; 記錄開始時間
v_mov_b32 v_start_lo, s10        ; 備份到 VGPR
v_mov_b32 v_start_hi, s11

; ... kernel 主體執行 ...

; ===== Kernel 結束前 =====
s_memrealtime s[12:13]           ; 記錄結束時間
; 計算差值（64-bit 減法）
v_sub_co_u32 v_diff_lo, vcc, s12, v_start_lo
v_subb_co_u32 v_diff_hi, vcc, s13, v_start_hi, vcc
; 輸出結果
; @PRINT f"[tid={$tid}] Kernel time: {v_diff_lo:d} ticks (low 32-bit)"
```

---

## 已知限制和挑戰

### 1. Printf 開銷污染測量結果

```
測量的時間 = 實際 kernel 時間 + printf 開銷
                                 ↑ 可能比 kernel 本身還大！
```

**影響程度**：
- Printf/hostcall 機制本身需要數千到數萬個 cycle
- 短 kernel（<1ms）的測量會被嚴重影響
- 長 kernel 的測量相對可接受

**緩解方案**：
- Phase 1：接受此限制，用於概念驗證
- Phase 2：改用 global memory 記錄，避免 printf 開銷

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

## 實現路徑建議

### Phase 1：概念驗證（Printf 輸出）

**目標**：驗證 `s_memrealtime` 可以正常工作

**設計**：
```asm
; 新 directive 語法
; @TIMESTAMP_START
; ... kernel ...
; @TIMESTAMP_END
```

**輸出**：
```
[tid=0] Kernel time: 12345 ticks
[tid=1] Kernel time: 12340 ticks
...
```

**限制**：
- 測量結果包含 printf 開銷
- 只適合概念驗證和長 kernel

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

## 與 rocprof 的比較

| 特性 | s_memtime 方案 | rocprof |
|------|---------------|---------|
| **細粒度** | ✅ 可測量任意程式碼區段 | ❌ 只有 kernel 整體時間 |
| **每 thread 時間** | ✅ 每個 thread/wavefront 獨立 | ❌ 無法區分 |
| **測量開銷** | ⚠️ printf 有開銷（Phase 1） | ✅ 硬體 counter，無開銷 |
| **侵入性** | ⚠️ 需要修改 kernel | ✅ 非侵入式 |
| **精確度** | ✅ cycle 級精確 | ✅ cycle 級精確 |
| **使用難度** | ⚠️ 需要理解 GPU 指令 | ✅ 簡單命令列 |

**使用場景**：
- **s_memtime 方案**：需要測量 kernel 內部特定區段、需要每 thread 獨立計時
- **rocprof**：只需要整體 kernel 時間、生產環境 profiling

---

## 待解決問題

1. [ ] 確認 `s_memrealtime` 在目標 GPU (gfx950) 上的可用性
2. [ ] 查詢 `s_memrealtime` 的時鐘頻率
3. [ ] 設計 directive 語法（`@TIMESTAMP_START`/`@TIMESTAMP_END` 或其他）
4. [ ] 決定是否支援 64-bit printf（目前 gpu.printf 可能有限制）
5. [ ] 評估 Phase 2 的 global memory buffer 設計

---

## 參考資料

- AMD GCN ISA Reference Guide
- AMDGPU Assembly Guide
- rocprof documentation
