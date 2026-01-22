# Thread ID 內建變數範例

此目錄包含 `$tid` 和 `$lane` 內建變數的使用範例。

## 檔案說明

- `test_builtin_vars.s` - 展示如何使用 `$tid` 和 `$lane` 內建變數

## 使用方式

```bash
cd /home/morhuang/Project-MDR

# 執行測試（128 個 thread = 2 個 wavefront）
python3 mdr_printf.py examples/03_thread_id_experiment/test_builtin_vars.s \
    --output-dir examples/03_thread_id_experiment/output \
    --test --test-size 128
```

## 內建變數說明

| 變數 | 說明 | 範圍 |
|------|------|------|
| `$tid` | Local Thread ID (workitem_id_x) | 0 ~ workgroup_size-1 |
| `$lane` | Wavefront Lane ID | 0-63 |

## 範例輸出

```
[Builtin Test] $tid=0, $lane=0
[Builtin Test] $tid=1, $lane=1
...
[Builtin Test] $tid=63, $lane=63
[Builtin Test] $tid=64, $lane=0   ← 第二個 wavefront
[Builtin Test] $tid=65, $lane=1
...
```

## 相關文件

- [主 README](../../README.md) - 完整功能說明
- [with_debug.s](../01_vector_add/with_debug.s) - 更多使用範例（包含條件式）
