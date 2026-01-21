# 範例 01: Vector Add

浮點數向量加法 kernel 的除錯範例，展示 SGPR、VGPR、條件式 printf 和表達式計算。

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

### 2. 表達式計算

```asm
; @PRINT f"[Expr] (A+B)x2/7={(v6+v7)*2/7:.2f}, A-B={v6-v7:.2f}, A*B={v6*v7:.2f}"
```

### 3. 條件式 printf

```asm
; 只印出 A <= 2.0 的 thread
; @PRINT if v6 <= 2.0: f"Before ADD (A<=2): A={v6:.3f}, B={v7:.2f}"

; 只印出 A > 2.0 的 thread
; @PRINT if v6 > 2.0: f"After v6={v6:.2f}, v7={v7:.2f}, C={v2:.3f}"
```

## 使用方法

```bash
# 編譯並測試
python3 ../../mdr_printf.py with_debug.s --output-dir output --test --test-size 8

# 或分開執行
python3 ../../mdr_printf.py with_debug.s --output-dir output

../../Track_B/kernel_testcases/universal_hsaco_runner \
    output/with_debug_debug_injected.hsaco \
    _Z9vectorAddPKfS0_Pfi float_add 8
```

## 預期輸出

```
[SGPR Test] n=8, base_idx=0
[SGPR Test] n=8, base_idx=0
...
[Expr] (A+B)x2/7=0.00, A-B=0.00, A*B=0.00
[Expr] (A+B)x2/7=0.86, A-B=-1.00, A*B=2.00
[Expr] (A+B)x2/7=1.71, A-B=-2.00, A*B=8.00
...
Before ADD (A<=2): A=0.000, B=0.00
Before ADD (A<=2): A=1.000, B=2.00
Before ADD (A<=2): A=2.000, B=4.00
After v6=3.00, v7=6.00, C=9.000
After v6=4.00, v7=8.00, C=12.000
...
✅ PASS: All 8 elements correct
```
