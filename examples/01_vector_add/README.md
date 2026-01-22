# 範例 01: Vector Add

浮點數向量加法 kernel 的除錯範例，展示 SGPR、VGPR、條件式 printf、表達式計算和內建變數。

## 檔案說明

| 檔案 | 說明 |
|------|------|
| `original.s` | 原始程式碼（無 @PRINT） |
| `with_debug.s` | 加入 @PRINT 後的程式碼 |

## @PRINT 指令範例

### 1. SGPR 印出

```asm
; @PRINT f"[SGPR Test] n={s4}, base_idx={s2}"
```

### 2. 內建變數 `$tid` 和 `$lane`

```asm
; 只印出前 4 個 thread
; @PRINT if $tid < 4: f"[tid<4] tid={$tid}, lane={$lane}"

; 只印出每個 wavefront 的 leader (lane 0)
; @PRINT if $lane == 0: f"[lane==0] tid={$tid} (wavefront leader)"
```

| 變數 | 說明 | 範圍 |
|------|------|------|
| `$tid` | Local Thread ID (workitem_id_x) | 0 ~ workgroup_size-1 |
| `$lane` | Wavefront Lane ID | 0-63 |

### 3. 表達式計算

```asm
; @PRINT f"[Expr] (A+B)x2/7={(v6+v7)*2/7:.2f}, A-B={v6-v7:.2f}, A*B={v6*v7:.2f}"
```

### 4. 條件式 printf

```asm
; 只印出 A <= 2.0 的 thread
; @PRINT if v6 <= 2.0: f"Before ADD (A<=2): A={v6:.3f}, B={v7:.2f}"

; 只印出 A > 2.0 的 thread
; @PRINT if v6 > 2.0: f"After v6={v6:.2f}, v7={v7:.2f}, C={v2:.3f}"
```

## 使用方法

```bash
# 編譯並測試
python3 ../../mdr_printf.py with_debug.s --output-dir output --test --test-size 128

# 或分開執行
python3 ../../mdr_printf.py with_debug.s --output-dir output

../../Track_B/kernel_testcases/universal_hsaco_runner \
    output/with_debug_debug_injected.hsaco \
    _Z9vectorAddPKfS0_Pfi float_add 128
```

## 預期輸出

```
[SGPR Test] n=128, base_idx=0
...
[tid<4] tid=0, lane=0
[tid<4] tid=1, lane=1
[tid<4] tid=2, lane=2
[tid<4] tid=3, lane=3
[lane==0] tid=0 (wavefront leader)
[lane==0] tid=64 (wavefront leader)
[Expr] (A+B)x2/7=0.00, A-B=0.00, A*B=0.00
[Expr] (A+B)x2/7=0.86, A-B=-1.00, A*B=2.00
...
Before ADD (A<=2): A=0.000, B=0.00
Before ADD (A<=2): A=1.000, B=2.00
Before ADD (A<=2): A=2.000, B=4.00
After v6=3.00, v7=6.00, C=9.000
After v6=4.00, v7=8.00, C=12.000
...
✅ PASS: All 128 elements correct
```

## 相關文件

- [主 README](../../README.md) - 完整功能說明
- [Thread ID 範例](../03_thread_id_experiment/README.md) - 更多 `$tid` 和 `$lane` 範例
