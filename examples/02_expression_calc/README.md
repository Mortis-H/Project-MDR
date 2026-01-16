# 範例 02: Expression Calculation

展示表達式計算功能的範例。

## 檔案說明

| 檔案 | 說明 |
|------|------|
| `with_expression.s` | 包含多個表達式計算的範例 |

## @PRINT 指令

```asm
; @PRINT cond=tid_eq(0) fmt="value=%d, v2*2=%d, v2+55=%d, v2*v2=%d" reg=v2 expr="v2*2; v2+55; v2*v2" type=i32,i32,i32,i32
```

這個指令會：
1. 印出 `v2` 的值（45）
2. 計算並印出 `v2*2`（90）
3. 計算並印出 `v2+55`（100）
4. 計算並印出 `v2*v2`（2025）

## 表達式語法

- 多個表達式用 `;` 分隔
- 支援 `+`, `-`, `*`, `/` 四則運算
- 支援括號 `()` 和常數

## 使用方法

```bash
# 編譯
python3 ../../mdr_printf.py with_expression.s --output-dir output

# 執行
../../Track_B/kernel_testcases/universal_hsaco_runner \
    output/with_expression_debug_injected.hsaco \
    _Z10loopKernelPii int_loop 8
```

## 預期輸出

```
value=45, v2*2=90, v2+55=100, v2*v2=2025
✅ PASS: Kernel executed successfully
```
