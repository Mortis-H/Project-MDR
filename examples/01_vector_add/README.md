# 範例 01: Vector Add

浮點數向量加法 kernel 的除錯範例。

## 檔案說明

| 檔案 | 說明 |
|------|------|
| `original.s` | 原始程式碼（無 @PRINT） |
| `with_debug.s` | 加入 @PRINT 後的程式碼 |

## @PRINT 指令

```asm
; @PRINT cond=tid_eq(0) fmt="A=%f, B=%f, A*B=%f" reg=v6,v7 expr="v6*v7" type=f32,f32,f32
```

這個指令會：
1. 印出 `v6`（A[tid]）的值
2. 印出 `v7`（B[tid]）的值
3. 計算並印出 `v6*v7`（A*B）的值

## 使用方法

```bash
# 編譯
python3 ../../mdr_printf.py with_debug.s --output-dir output

# 執行
../../Track_B/kernel_testcases/universal_hsaco_runner \
    output/with_debug_debug_injected.hsaco \
    _Z9vectorAddPKfS0_Pfi float_add 8
```

## 預期輸出

```
A=0.000000, B=0.000000, A*B=0.000000
A=1.000000, B=2.000000, A*B=2.000000
A=2.000000, B=4.000000, A*B=8.000000
...
✅ PASS: All 8 elements correct
```
