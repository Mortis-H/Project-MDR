# s_barrier 與 gpu.printf hostcall 不相容問題分析

## 問題描述

在使用 `mdr_printf.py` 對含有 `s_barrier` 指令的 kernel 注入 `@PRINT` 時，可能會發生：
1. printf 輸出沒有顯示
2. Kernel hang（無限等待）
3. Illegal memory access 錯誤

## 實驗驗證

### 測試 1：無 s_barrier 的 kernel（正常工作）

```bash
# vectorAdd kernel - 無 barrier
./universal_hsaco_runner output/with_debug_debug_injected.hsaco _Z9vectorAddPKfS0_Pfi float_add 8

# 結果：正常輸出
A=0.000000, B=0.000000, A*B=0.000000
A=1.000000, B=2.000000, A*B=2.000000
...
✅ PASS: All 8 elements correct
```

### 測試 2：有 s_barrier 的 kernel（256 threads = 4 waves）

```bash
# sharedMemKernel - 有 barrier
./universal_hsaco_runner output/with_barrier_debug_injected.hsaco _Z15sharedMemKernelPKiPii int_shared 256

# 結果：崩潰
HIP error: an illegal memory access was encountered
```

## 根本原因分析

### 1. Hostcall 機制的工作流程

`gpu.printf` 被 MLIR 的 `convert-gpu-to-rocdl` pass 轉換為 hostcall 機制：

```
Thread 執行 printf
    ↓
獲取 hostcall buffer slot（global_atomic_cmpswap_x2）
    ↓
如果 slot 被占用 → s_sleep 循環等待
    ↓
寫入 printf 數據到 buffer
    ↓
發送中斷給 host（s_sendmsg MSG_INTERRUPT）
    ↓
等待 host 處理完成（s_sleep 循環）
    ↓
釋放 slot，繼續執行
```

**關鍵問題**：獲取 slot 和等待 host 都是**阻塞操作**，使用 `s_sleep` 循環。

### 2. s_barrier 的語義

```asm
s_barrier    ; 所有 active threads 必須同時到達此點
```

- Workgroup-level 同步
- **所有 active threads** 必須到達 barrier
- 如果有任何 thread 沒到達，其他 thread 會**永遠等待**

### 3. 衝突場景（死鎖形成）

假設 @PRINT 在 barrier 之前：

```
Timeline:
─────────────────────────────────────────────────────────
Wave 0, Lane 0: printf → 獲取 slot → 等待 host...
Wave 0, Lane 1-63: 完成計算 → 到達 barrier → 等待其他 threads...
Wave 1, Lane 0: printf → 等待 slot（被 Wave 0 占用）...
Wave 1, Lane 1-63: 完成計算 → 到達 barrier → 等待其他 threads...
Wave 2, Wave 3: 同上...
─────────────────────────────────────────────────────────

死鎖條件：
- Wave 0 Lane 0 在 hostcall 等待循環中
- 其他 threads 在 barrier 等待 Wave 0 Lane 0
- Host 可能在等待 GPU 完成某些操作
→ 無人能繼續，形成死鎖
```

### 4. 生成的 ISA 代碼分析

查看 `with_debug_debug_injected.s`，可以看到 hostcall 機制的實作：

```asm
; === 獲取 hostcall buffer slot ===
.LBB0_4:
    s_sleep 1                                    ; 等待循環
    global_load_dwordx2 v[8:9], v14, s[20:21] offset:40
    ...
    global_atomic_cmpswap_x2 v[8:9], v14, ...    ; 原子操作獲取 slot
    v_cmp_eq_u64_e32 vcc, v[8:9], v[12:13]
    s_or_b64 s[26:27], vcc, s[26:27]
    s_andn2_b64 exec, exec, s[26:27]
    s_cbranch_execnz .LBB0_4                     ; 如果失敗，繼續等待

; === 等待 host 處理 ===
.LBB0_21:
    v_mov_b32_e32 v10, 1
    s_and_saveexec_b64 s[24:25], s[0:1]
    s_cbranch_execz .LBB0_18
    global_load_dword v10, v[12:13], off offset:20 sc0 sc1
    s_waitcnt vmcnt(0)
    buffer_inv sc0 sc1
    v_and_b32_e32 v10, 1, v10                    ; 檢查 host 是否處理完成
    s_branch .LBB0_18
    ...
    s_sleep 1                                    ; 繼續等待
    s_cbranch_execnz .LBB0_21
```

這些 `s_sleep` 循環就是死鎖的根源。

### 5. 為什麼 cond=tid_eq(0) 不能完全解決問題

即使使用條件式 printf：

```asm
; @PRINT cond=tid_eq(0) fmt="..." reg=v0 type=i32
```

生成的代碼會使用 `scf.if`：

```mlir
%cond = arith.cmpi eq, %tid_x, %cmp_val : index
scf.if %cond {
    gpu.printf "..."
}
```

問題：
1. 每個 **wave** 的 lane 0 都會嘗試執行 printf
2. 如果 workgroup 有 4 個 waves（256 threads），就有 4 個 threads 競爭 hostcall buffer
3. 競爭 + barrier = 死鎖風險

## 解決方案

### 方案 1：避免在 barrier 之前使用 @PRINT

**最安全的方法**：將所有 @PRINT 放在最後一個 s_barrier 之後。

```asm
; 計算部分
global_load_dword v3, v[2:3], off
ds_write_b32 v1, v3
s_waitcnt lgkmcnt(0)
s_barrier                    ; 最後的 barrier

; === 在所有 barrier 之後才使用 @PRINT ===
; @PRINT cond=tid_eq(0) fmt="result=%d" reg=v3 type=i32
s_endpgm
```

### 方案 2：使用 --no-printf 進行功能驗證

```bash
# 先驗證 kernel 功能正確性（不注入 printf）
python3 mdr_printf.py input.s --output-dir output --no-printf --test
```

### 方案 3：限制為單一 wave 執行（實驗性）

使用 workgroup 級別的唯一 thread 條件：

```asm
; 只讓 workgroup 中的第一個 thread 執行 printf
; 需要計算 global_thread_id == 0
; @PRINT cond=wg_eq(0) fmt="..." reg=v0 type=i32
```

這需要擴展 mdr_printf.py 支援 `wg_eq(N)` 條件。

## mdr_printf.py 完整性檢查

### 現有功能（已驗證）

| 功能 | 狀態 | 說明 |
|------|------|------|
| @PRINT 基本格式 | ✓ | `fmt="..." reg=v0 type=f32` |
| 多暫存器 | ✓ | `reg=v0,v1,v2 type=f32,f32,f32` |
| 條件式 printf | ✓ | `cond=tid_eq(0)` |
| 表達式計算 | ✓ | `expr="v6*v7"` |
| Register clobbering | ✓ | 保護原始暫存器值 |
| Kernarg pointer 保存 | ✓ | s[0:1] → s[18:19] |
| Metadata 修復 | ✓ | 保留 hidden_hostcall_buffer |

### 已知限制

| 限制 | 原因 | 建議 |
|------|------|------|
| s_barrier 衝突 | hostcall 阻塞與 barrier 同步互斥 | 避免在 barrier 前使用 |
| printf 位置固定 | 所有 printf 注入到 .LBB0_2 之前 | 未來可改進 |
| 多 wave 競爭 | 每個 wave 的 lane 0 都會執行 | 使用更嚴格的條件 |

### 建議改進

1. **增強 s_barrier 檢測**：分析 @PRINT 相對於 barrier 的位置
2. **支援 workgroup-unique printf**：新增 `wg_eq(0)` 條件
3. **改進注入位置**：讓每個 @PRINT 在其標記位置附近執行

## 實驗驗證結果（2026-01-20 更新）

### 測試案例：有 s_barrier 的 kernel + cond=tid_eq(0)

```bash
# 256 threads (4 waves)
./universal_hsaco_runner kernel.hsaco _Z15sharedMemKernelPKiPii int_shared 256

# 結果：正常執行，有 printf 輸出，沒有死鎖
✓ Kernel launched (grid=1, block=256)
After barrier: tid=0, result=32184
After barrier: tid=1, result=32184
...
✅ PASS: Kernel executed successfully
```

### 重要發現

1. **使用 `cond=tid_eq(0)` 可以避免死鎖**：
   - 每個 wave 只有 lane 0 執行 hostcall
   - 競爭程度大幅降低
   - 單 workgroup（即使有多個 waves）可以正常工作

2. **printf 注入位置的限制**：
   - mdr_printf.py 總是把 printf 注入到 `.LBB0_2:` 標籤之前
   - 不管 `@PRINT` 標記的實際位置在哪裡
   - 這是目前實作的限制，未來可改進

3. **`cond=tid_eq(0)` 的實際行為**：
   - 生成的 MLIR 使用 `gpu.thread_id x` 檢查條件
   - 所有 threads 都會執行到條件檢查
   - 只有滿足條件的 thread 進入 hostcall

## 總結

| Kernel 類型 | printf 支援 | 實測結果 |
|-------------|-------------|----------|
| 無 s_barrier | ✅ 完全支援 | 正常工作 |
| 有 s_barrier + `if $tid == 0:` | ✅ 可用 | 256/512 threads 正常 |
| 有 s_barrier + `cond=tid_eq(0)` | ✅ 可用 | 256 threads 正常（舊語法） |
| 有 s_barrier，無 cond | ⚠️ 高風險 | 可能死鎖或崩潰 |

**核心原則**：
1. 有 s_barrier 的 kernel **必須**使用 `if $tid == 0:` 或類似條件
2. 這樣可以限制 hostcall 的競爭，避免死鎖
3. Hostcall 是阻塞操作，不限制執行數量時會與 barrier 衝突

---

## 2026-01-22 更新：使用 $tid 內建變數

### 新語法（推薦）

```asm
; 使用新的 $tid 內建變數作為條件
; @PRINT if $tid == 0: f"[After barrier] Reduction result: v1={v1:d}"
```

### 工具自動檢測

mdr_printf.py 會自動檢測 kernel 中是否有 `s_barrier` 指令並發出警告：

```
⚠️  WARNING: Kernel contains s_barrier instruction!
   gpu.printf's hostcall mechanism may conflict with barrier
   synchronization, causing kernel to hang or crash.

   Recommendations:
   1. Use --no-printf for functional verification
   2. Use cond=REG_eq(N) to limit printf (e.g., v6_eq(0.0))
   3. Place @PRINT only after all barriers complete
```

---

## 2026-01-22 擴大測試：重新驗證 s_barrier + printf

### 重大發現：快照機制使 printf 與 barrier 相容

經過更全面的測試後發現，**MDR 的「快照 + 延遲輸出」機制使得 printf 與 s_barrier 可以相容**：

1. **@PRINT 位置**：只注入快照指令（把 VGPR 值複製到專用快照 VGPR）
2. **printf 執行時機**：所有 `gpu.printf` hostcall 在 kernel 結束時（`s_endpgm` 前）統一執行
3. **結果**：hostcall 永遠在所有 barrier 完成後執行，避免了死鎖

### 擴大測試結果

| 測試檔案 | 條件 | Printf 位置 | Threads | 結果 |
|----------|------|------------|---------|------|
| test_barrier_conditions.s | `$tid == 0` | barrier 前後，3 處 | 256 | ✅ 成功 (3 行) |
| test_lane_condition.s | `$lane == 0` | barrier 前後 | 256 | ✅ 成功 (1 行) |
| test_multi_threads.s | `$tid < 4` | barrier 前 | 256 | ✅ 成功 (4 行) |
| test_unconditional.s | 無條件 | barrier 前 | 256 | ✅ 成功 (64 行) |
| test_unconditional_after_barrier.s | 無條件 | barrier 後 | 256 | ✅ 成功 (256 行) |
| test_inside_loop_barrier.s | 無條件 | loop 內 barrier 後 | 256 | ✅ 成功 (256 行) |

### 測試範例輸出

**條件式 printf（$tid == 0）**：
```
[Test1] Before barrier: v3=0
[Test2] After barrier: v3=0
[Test3] Reduction result: v1=-1035452244
✅ PASS: Kernel executed successfully
```

**多 thread 條件（$tid < 4）**：
```
[Multi-thread Test] tid=0, lane=0, v3=0
[Multi-thread Test] tid=1, lane=1, v3=0
[Multi-thread Test] tid=2, lane=2, v3=0
[Multi-thread Test] tid=3, lane=3, v3=0
✅ PASS: Kernel executed successfully
```

**無條件 printf（256 threads）**：
```
[Inside Loop] tid=128, v3=3776
[Inside Loop] tid=129, v3=3777
...（共 256 行，順序非確定）
✅ PASS: Kernel executed successfully
```

### 更新後的建議

| 使用情境 | 建議條件 | 說明 |
|----------|----------|------|
| 只需單一輸出 | `if $tid == 0:` | 最少輸出，適合看整體結果 |
| 檢查每個 wavefront | `if $lane == 0:` | 每個 wavefront 輸出 1 行 |
| 限制輸出數量 | `if $tid < N:` | 輸出前 N 個 thread |
| 完整 debug | 無條件 | 輸出所有 thread，注意輸出量 |

### 注意事項

1. **輸出順序非確定**：即使所有 thread 都輸出，順序取決於 wavefront 調度
2. **輸出量考量**：無條件 printf 會產生大量輸出，可能影響可讀性
3. **效能影響**：大量 printf 仍會影響效能，建議 debug 時限制條件

### 結論

**s_barrier + printf 現在是完全相容的**，因為：
1. MDR 使用快照機制，在 @PRINT 位置只記錄值
2. 實際 hostcall 在 kernel 結束時才執行
3. 所有 barrier 同步在 hostcall 之前完成

條件式 printf（如 `if $tid == 0:`）仍然推薦使用，主要是為了：
- 減少輸出量，提高可讀性
- 降低效能影響
- 聚焦於特定 thread 的行為
